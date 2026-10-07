"""Unity Catalog layout and Delta writes from a workstation.

Rows are staged as Parquet in a UC Volume and loaded with ``read_files`` —
one round trip per table instead of row-by-row INSERTs.
"""
from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

STAGING_VOLUME = "staging"


@dataclass(frozen=True)
class Layout:
    catalog: str
    benchmark: str = "benchmark"
    traces: str = "traces"
    evaluation: str = "evaluation"

    def fq(self, schema: str, table: str) -> str:
        return f"`{self.catalog}`.`{schema}`.`{table}`"

    @property
    def project_schemas(self) -> tuple[str, ...]:
        return (self.benchmark, self.traces, self.evaluation)

    @property
    def staging_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.benchmark}/{STAGING_VOLUME}"


def ensure_layout(runner: Any, catalog: str, fallback_catalog: str | None, schemas: dict[str, str]) -> tuple[Layout, list[str]]:
    """Create catalog (or fall back), project schemas and the staging volume.

    ``runner`` needs ``run(sql)``. Returns the layout and a log of what happened.
    """
    notes: list[str] = []
    try:
        runner.run(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
        notes.append(f"catalog {catalog}: ok")
    except Exception as e:
        if not fallback_catalog:
            raise
        notes.append(f"catalog {catalog}: CREATE failed ({str(e)[:160]}); using fallback {fallback_catalog}")
        catalog = fallback_catalog
    layout = Layout(catalog, **{k: v for k, v in schemas.items() if k in Layout.__dataclass_fields__})
    for s in layout.project_schemas:
        runner.run(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{s}`")
    runner.run(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{layout.benchmark}`.`{STAGING_VOLUME}`")
    notes.append(f"schemas {', '.join(layout.project_schemas)} + volume {STAGING_VOLUME}: ok")
    return layout, notes


def upload_bytes(workspace_config: Any, volume_path: str, data: bytes) -> None:
    from databricks.sdk import WorkspaceClient

    WorkspaceClient(config=workspace_config).files.upload(volume_path, io.BytesIO(data), overwrite=True)


def upload_file(workspace_config: Any, volume_path: str, local_path: str) -> None:
    from databricks.sdk import WorkspaceClient

    with open(local_path, "rb") as f:
        WorkspaceClient(config=workspace_config).files.upload(volume_path, f, overwrite=True)


def rows_to_parquet(rows: list[dict[str, Any]], schema: pa.Schema) -> bytes:
    table = pa.Table.from_pylist(rows, schema=schema)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def write_rows(runner: Any, layout: Layout, schema_name: str, table: str, rows: list[dict[str, Any]],
               arrow_schema: pa.Schema, mode: str = "append") -> str:
    """Write rows to a Delta table. mode: append | overwrite | create_if_absent."""
    fq = layout.fq(schema_name, table)
    path = f"{layout.staging_root}/{schema_name}.{table}/{uuid.uuid4().hex}.parquet"
    upload_bytes(runner.workspace_config, path, rows_to_parquet(rows, arrow_schema))
    src = f"SELECT * FROM read_files('{path}', format => 'parquet')"
    if mode == "overwrite":
        runner.run(f"CREATE OR REPLACE TABLE {fq} AS {src}")
    elif mode == "create_if_absent":
        runner.run(f"CREATE TABLE {fq} AS {src}")
    else:
        runner.run(f"CREATE TABLE IF NOT EXISTS {fq} AS {src} LIMIT 0")
        ensure_columns(runner, fq, arrow_schema)
        runner.run(f"INSERT INTO {fq} BY NAME {src}")
    log.info("wrote %d rows to %s (%s)", len(rows), fq, mode)
    return fq


_SQL_TYPES = {"string": "STRING", "int64": "BIGINT", "bool": "BOOLEAN", "timestamp[us, tz=UTC]": "TIMESTAMP"}


def ensure_columns(runner: Any, fq: str, arrow_schema: pa.Schema) -> list[str]:
    """Additive schema evolution for append-only evidence tables: add columns
    that exist in ``arrow_schema`` but not yet in the Delta table."""
    existing = {str(r[0]).lower() for r in runner.run(f"DESCRIBE TABLE {fq}")
                if r and r[0] and not str(r[0]).startswith("#")}
    added = []
    for f in arrow_schema:
        if f.name.lower() not in existing:
            runner.run(f"ALTER TABLE {fq} ADD COLUMNS (`{f.name}` {_SQL_TYPES[str(f.type)]})")
            added.append(f.name)
    if added:
        log.info("added columns %s to %s", added, fq)
    return added


def table_exists(runner: Any, layout: Layout, schema_name: str, table: str) -> bool:
    rows = runner.run(f"SHOW TABLES IN `{layout.catalog}`.`{schema_name}` LIKE '{table}'")
    return len(rows) > 0

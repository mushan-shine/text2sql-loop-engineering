"""Phase 0.2b — restore BEAVER databases in Unity Catalog.

Source of truth is the BEAVER MySQL dump loaded into the reference MySQL
server. Each MySQL database ``db`` becomes the UC schema ``<catalog>.<db>``
(same name, so unqualified and db-qualified gold SQL both resolve). The FULL
database is replicated — not only gold tables, which would leak ground truth
into retrieval.

Pipeline per table: MySQL → Parquet (typed, lossless where possible) → UC
Volume → ``CREATE TABLE`` with explicit mapped DDL → ``INSERT ... SELECT CAST``.
Afterwards :func:`fidelity_check` compares row counts and column profiles on
both engines.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MySqlColumn:
    name: str
    data_type: str        # information_schema.COLUMNS.DATA_TYPE
    column_type: str      # e.g. "decimal(10,2)", "tinyint(1)", "int unsigned"
    precision: int | None
    scale: int | None
    collation: str | None


@dataclass(frozen=True)
class ColumnMapping:
    name: str
    mysql_type: str
    databricks_type: str   # DDL type, may include COLLATE
    arrow_type: pa.DataType
    staged_as_string: bool  # value staged as text and CAST on insert
    note: str | None = None


_INTS = {"tinyint": "TINYINT", "smallint": "SMALLINT", "mediumint": "INT", "int": "INT", "integer": "INT",
         "bigint": "BIGINT"}
_WIDER = {"TINYINT": "SMALLINT", "SMALLINT": "INT", "INT": "BIGINT", "BIGINT": "DECIMAL(20,0)"}
_STRINGS = {"char", "varchar", "tinytext", "text", "mediumtext", "longtext", "enum", "set", "json"}
_BINARY = {"binary", "varbinary", "tinyblob", "blob", "mediumblob", "longblob", "bit", "geometry"}


def map_column(c: MySqlColumn, collation_policy: str = "mirror_mysql_ci") -> ColumnMapping:
    t = c.data_type.lower()
    unsigned = "unsigned" in c.column_type.lower()
    if t in _INTS:
        dbx = _INTS[t]
        if unsigned:  # widen so the full unsigned range fits
            dbx = _WIDER[dbx]
        arrow = pa.decimal128(20, 0) if dbx.startswith("DECIMAL") else pa.int64()
        return ColumnMapping(c.name, c.column_type, dbx, arrow, False,
                             "unsigned widened" if unsigned else None)
    if t in ("decimal", "numeric"):
        p, s = c.precision or 10, c.scale or 0
        if p > 38:
            return ColumnMapping(c.name, c.column_type, f"DECIMAL(38,{min(s, 38)})", pa.string(), True,
                                 f"precision {p}>38 clipped")
        return ColumnMapping(c.name, c.column_type, f"DECIMAL({p},{s})", pa.decimal128(p, s), False)
    if t == "float":
        return ColumnMapping(c.name, c.column_type, "FLOAT", pa.float32(), False)
    if t in ("double", "real"):
        return ColumnMapping(c.name, c.column_type, "DOUBLE", pa.float64(), False)
    if t == "date":
        return ColumnMapping(c.name, c.column_type, "DATE", pa.date32(), False)
    if t in ("datetime", "timestamp"):
        # MySQL DATETIME has no time zone → TIMESTAMP_NTZ (no session-tz shift).
        return ColumnMapping(c.name, c.column_type, "TIMESTAMP_NTZ", pa.string(), True)
    if t == "year":
        return ColumnMapping(c.name, c.column_type, "INT", pa.int64(), False)
    if t == "time":
        return ColumnMapping(c.name, c.column_type, "STRING", pa.string(), False, "TIME stored as HH:MM:SS text")
    if t in _BINARY:
        return ColumnMapping(c.name, c.column_type, "BINARY", pa.binary(), False)
    if t in _STRINGS:
        coll = (c.collation or "").lower()
        if collation_policy == "mirror_mysql_ci" and coll.endswith("_ci"):
            note = "UTF8_LCASE mirrors case-insensitivity; accent-insensitivity/PAD SPACE not mirrored"
            return ColumnMapping(c.name, c.column_type, "STRING COLLATE UTF8_LCASE", pa.string(), False, note)
        return ColumnMapping(c.name, c.column_type, "STRING", pa.string(), False)
    return ColumnMapping(c.name, c.column_type, "STRING", pa.string(), False, f"unmapped type {t} → STRING")


# --------------------------------------------------------------------------- value staging


@dataclass
class StageStats:
    rows: int = 0
    nulled_invalid_dates: int = 0
    notes: list[str] = field(default_factory=list)


def stage_value(v: Any, m: ColumnMapping, stats: StageStats) -> Any:
    if v is None:
        return None
    dbx = m.databricks_type
    if dbx == "TIMESTAMP_NTZ":
        if isinstance(v, dt.datetime):
            return v.isoformat(sep=" ")
        stats.nulled_invalid_dates += 1  # pymysql returns zero-dates as str
        return None
    if dbx == "DATE":
        if isinstance(v, dt.date):
            return v
        stats.nulled_invalid_dates += 1
        return None
    if m.mysql_type.startswith("time"):
        if isinstance(v, dt.timedelta):
            total = int(v.total_seconds())
            sign, total = ("-" if total < 0 else ""), abs(total)
            return f"{sign}{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
        return str(v)
    if isinstance(m.arrow_type, pa.Decimal128Type):
        return Decimal(v) if not isinstance(v, Decimal) else v
    if pa.types.is_integer(m.arrow_type):
        if isinstance(v, (bytes, bytearray)):  # BIT columns
            return int.from_bytes(v, "big")
        return int(v)
    if pa.types.is_floating(m.arrow_type):
        return float(v)
    if pa.types.is_binary(m.arrow_type):
        return bytes(v) if not isinstance(v, bytes) else v
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return str(v) if not isinstance(v, str) else v


def write_parquet(batches: Iterable[list[tuple]], mappings: list[ColumnMapping], path: Path) -> StageStats:
    stats = StageStats()
    # Parquet field names are positional c0..cn: MySQL column names may contain
    # characters that are awkward in Parquet/Spark; the INSERT maps them back.
    schema = pa.schema([pa.field(f"c{i}", m.arrow_type) for i, m in enumerate(mappings)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema) as w:
        for batch in batches:
            cols = list(zip(*batch)) if batch else [[] for _ in mappings]
            arrays = [pa.array([stage_value(v, m, stats) for v in col], type=m.arrow_type)
                      for col, m in zip(cols, mappings)]
            w.write_table(pa.Table.from_arrays(arrays, schema=schema))
            stats.rows += len(batch)
    return stats


# --------------------------------------------------------------------------- DDL


def quote(ident: str) -> str:
    return "`" + ident.replace("`", "``") + "`"


def needs_column_mapping(names: list[str]) -> bool:
    return any(re.search(r"[ ,;{}()\n\t=]", n) for n in names)


def create_table_sql(fq_table: str, mappings: list[ColumnMapping]) -> str:
    cols = ",\n  ".join(f"{quote(m.name)} {m.databricks_type}" for m in mappings)
    props = ["'delta.feature.allowColumnDefaults' = 'supported'"]
    if needs_column_mapping([m.name for m in mappings]):
        props.append("'delta.columnMapping.mode' = 'name'")
    return f"CREATE OR REPLACE TABLE {fq_table} (\n  {cols}\n) TBLPROPERTIES ({', '.join(props)})"


def insert_from_parquet_sql(fq_table: str, mappings: list[ColumnMapping], volume_path: str) -> str:
    sel = ", ".join(
        f"CAST(c{i} AS {m.databricks_type.split(' COLLATE ')[0]}) AS {quote(m.name)}"
        if m.staged_as_string else f"c{i} AS {quote(m.name)}"
        for i, m in enumerate(mappings)
    )
    return (f"INSERT INTO {fq_table} SELECT {sel} "
            f"FROM read_files('{volume_path}', format => 'parquet')")


# --------------------------------------------------------------------------- fidelity


def profile_sql(fq_table: str, mappings: list[ColumnMapping], engine: str) -> str:
    """Engine-portable column profile: count, non-null count, distinct count per column.

    COUNT(DISTINCT) follows the column collation on both engines, so a MySQL
    ``_ci`` column is only matched by a UTF8_LCASE column on Databricks.
    """
    parts = ["COUNT(*)"]
    for m in mappings:
        c = quote(m.name)
        parts.append(f"COUNT({c})")
        if m.databricks_type == "BINARY" and engine == "databricks":
            parts.append(f"COUNT(DISTINCT hex({c}))")
        else:
            parts.append(f"COUNT(DISTINCT {c})")
    return f"SELECT {', '.join(parts)} FROM {fq_table}"


@dataclass
class TableFidelity:
    db: str
    table: str
    mysql_rows: int
    databricks_rows: int
    columns: int
    mismatched_columns: list[str]
    nulled_invalid_dates: int
    notes: list[str]

    @property
    def ok(self) -> bool:
        return self.mysql_rows == self.databricks_rows and not self.mismatched_columns


def compare_profiles(mappings: list[ColumnMapping], mysql_prof: tuple, dbx_prof: tuple,
                     nulled_invalid_dates: int) -> list[str]:
    bad = []
    for i, m in enumerate(mappings):
        nn_m, nd_m = mysql_prof[1 + 2 * i], mysql_prof[2 + 2 * i]
        nn_d, nd_d = dbx_prof[1 + 2 * i], dbx_prof[2 + 2 * i]
        tolerance = nulled_invalid_dates if m.databricks_type in ("DATE", "TIMESTAMP_NTZ") else 0
        if abs(int(nn_m) - int(nn_d)) > tolerance or (tolerance == 0 and int(nd_m) != int(nd_d)):
            bad.append(f"{m.name}: nonnull {nn_m}/{nn_d}, distinct {nd_m}/{nd_d}")
    return bad

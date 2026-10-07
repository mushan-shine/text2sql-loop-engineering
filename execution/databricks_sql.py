"""Databricks SQL executors.

* :class:`DatabricksSqlExecutor` — workstation → SQL Warehouse (databricks-sql-connector),
  authenticated through a ~/.databrickscfg profile via the Databricks SDK.
* :class:`SparkSqlExecutor` — inside a Databricks notebook / job (``spark.sql``).

Both apply the same session environment (see config/phase0.yaml ``environment``)
so gold SQL and, later, generated SQL run under identical settings.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from execution.base import ExecutionResult, Timer, is_read_only

log = logging.getLogger(__name__)

_ERROR_CLASS = re.compile(r"\[([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]")


def extract_error_class(message: str) -> str | None:
    m = _ERROR_CLASS.search(message or "")
    return m.group(1) if m else None


def session_settings(ansi_mode: bool, statement_timeout_s: int | None, disable_result_cache: bool) -> list[str]:
    stmts = [f"SET ANSI_MODE = {'true' if ansi_mode else 'false'}"]
    if statement_timeout_s:
        stmts.append(f"SET STATEMENT_TIMEOUT = {int(statement_timeout_s)}")
    if disable_result_cache:
        stmts.append("SET use_cached_result = false")
    return stmts


class DatabricksSqlExecutor:
    engine = "databricks"

    def __init__(
        self,
        catalog: str,
        profile: str | None = "DEFAULT",
        http_path: str | None = None,
        ansi_mode: bool = False,
        statement_timeout_s: int = 120,
        disable_result_cache: bool = True,
    ):
        from databricks import sql as dbsql
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config

        if os.environ.get("DATABRICKS_RUNTIME_VERSION"):  # notebook / job: the workspace authenticates us
            profile = None
        self._cfg = Config(profile=profile) if profile else Config()
        http_path = http_path or os.environ.get("DATABRICKS_HTTP_PATH") or self._discover_http_path(
            WorkspaceClient(config=self._cfg)
        )
        host = self._cfg.host.replace("https://", "").rstrip("/")
        self._conn = dbsql.connect(
            server_hostname=host,
            http_path=http_path,
            credentials_provider=lambda: self._cfg.authenticate,
        )
        self.catalog = catalog
        self.http_path = http_path
        self._db: str | None = None
        self._settings = session_settings(ansi_mode, statement_timeout_s, disable_result_cache)
        for s in self._settings:
            self.run(s)
        log.info("connected to %s (%s), catalog=%s", host, http_path, catalog)

    @staticmethod
    def _discover_http_path(w: Any) -> str:
        whs = list(w.warehouses.list())
        if not whs:
            raise RuntimeError("no SQL warehouse found; set DATABRICKS_HTTP_PATH")
        running = [x for x in whs if str(x.state).endswith("RUNNING")]
        wh = (running or whs)[0]
        return wh.odbc_params.path if wh.odbc_params else f"/sql/1.0/warehouses/{wh.id}"

    @property
    def workspace_config(self) -> Any:
        return self._cfg

    def run(self, sql: str) -> list[tuple]:
        """Run trusted project SQL (DDL / DML for our own tables)."""
        with self._conn.cursor() as cur:
            cur.execute(sql)
            try:
                return [tuple(r) for r in cur.fetchall()]
            except Exception:  # statements without a result set
                return []

    def query_df(self, sql: str):
        """Run trusted project SQL and return a pandas DataFrame (dashboards / analysis)."""
        return self.query_arrow(sql).to_pandas()

    def query_arrow(self, sql: str):
        """Run trusted project SQL and return a pyarrow Table (types kept as they are, e.g. for exports)."""
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall_arrow()

    def use(self, catalog: str, schema: str | None = None) -> None:
        self.run(f"USE CATALOG `{catalog}`")
        if schema:
            self.run(f"USE SCHEMA `{schema}`")
        self.catalog, self._db = catalog, schema

    def execute(self, sql: str, db: str, max_rows: int | None = None) -> ExecutionResult:
        """Run benchmark / generated SQL (read-only) against schema ``db``.

        ``max_rows`` guards against runaway results (e.g. an accidental cross join
        in generated SQL): larger results become status ``TOO_MANY_ROWS``.
        """
        if not is_read_only(sql, "mysql") and not is_read_only(sql, "databricks"):
            return ExecutionResult(self.engine, "REJECTED", error="not a single read-only statement")
        try:
            if db != self._db:
                self.use(self.catalog, db)
            with self._conn.cursor() as cur, Timer() as t:
                cur.execute(sql)
                rows = [tuple(r) for r in (cur.fetchall() if max_rows is None else cur.fetchmany(max_rows + 1))]
                cols = [d[0] for d in cur.description] if cur.description else []
            if max_rows is not None and len(rows) > max_rows:
                return ExecutionResult(self.engine, "TOO_MANY_ROWS", error=f"result exceeds {max_rows} rows",
                                       elapsed_ms=t.ms)
            return ExecutionResult(self.engine, "SUCCESS", rows, cols, elapsed_ms=t.ms)
        except Exception as e:  # connector raises ServerOperationError / DatabaseError
            msg = str(e)
            cls = extract_error_class(msg)
            timed_out = (cls is not None and "TIMEOUT" in cls) or "timed out" in msg.lower()
            status = "TIMEOUT" if timed_out else "ERROR"
            return ExecutionResult(self.engine, status, error=msg[:4000], error_class=cls)

    def close(self) -> None:
        self._conn.close()


class SparkSqlExecutor:
    """Same contract, for notebooks and jobs running on Databricks."""

    engine = "databricks"

    def __init__(self, spark: Any, catalog: str, ansi_mode: bool = False, disable_result_cache: bool = True):
        self.spark = spark
        self.catalog = catalog
        self._db: str | None = None
        for s in session_settings(ansi_mode, None, disable_result_cache):
            try:
                spark.sql(s)
            except Exception as e:  # some confs are warehouse-only
                log.warning("could not apply %r: %s", s, e)

    def run(self, sql: str) -> list[tuple]:
        return [tuple(r) for r in self.spark.sql(sql).collect()]

    def execute(self, sql: str, db: str) -> ExecutionResult:
        if not is_read_only(sql, "mysql") and not is_read_only(sql, "databricks"):
            return ExecutionResult(self.engine, "REJECTED", error="not a single read-only statement")
        try:
            if db != self._db:
                self.spark.sql(f"USE `{self.catalog}`.`{db}`")
                self._db = db
            with Timer() as t:
                df = self.spark.sql(sql)
                rows = [tuple(r) for r in df.collect()]
            return ExecutionResult(self.engine, "SUCCESS", rows, list(df.columns), elapsed_ms=t.ms)
        except Exception as e:
            msg = str(e)
            return ExecutionResult(self.engine, "ERROR", error=msg[:4000], error_class=extract_error_class(msg))

    def close(self) -> None:
        pass

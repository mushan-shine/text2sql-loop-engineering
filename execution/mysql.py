"""MySQL executor — BEAVER's official engine, used as the reference oracle."""
from __future__ import annotations

import logging
import os

from execution.base import ExecutionResult, Timer, is_read_only

log = logging.getLogger(__name__)


class MySqlExecutor:
    engine = "mysql"

    def __init__(self, host: str, port: int, user: str, password: str, timeout_s: int = 120):
        import pymysql

        self._pymysql = pymysql
        self._conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            charset="utf8mb4", autocommit=True, read_timeout=timeout_s + 5,
        )
        self._timeout_ms = timeout_s * 1000
        self._db: str | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "MySqlExecutor":
        user = os.environ.get(cfg.get("user_env", "MYSQL_USER"), "root")
        pwd_env = cfg.get("password_env", "MYSQL_PASSWORD")
        if pwd_env not in os.environ:
            raise RuntimeError(f"set {pwd_env} (e.g. in .env) for the MySQL reference engine")
        return cls(cfg.get("host", "localhost"), int(cfg.get("port", 3306)), user,
                   os.environ[pwd_env], int(cfg.get("timeout_s", 120)))

    def execute(self, sql: str, db: str) -> ExecutionResult:
        if not is_read_only(sql, "mysql"):
            return ExecutionResult(self.engine, "REJECTED", error="not a single read-only statement")
        with self._conn.cursor() as cur:
            try:
                if db != self._db:
                    self._conn.select_db(db)
                    self._db = db
                cur.execute(f"SET SESSION MAX_EXECUTION_TIME={self._timeout_ms}")
                with Timer() as t:
                    cur.execute(sql)
                    rows = [tuple(r) for r in cur.fetchall()] if cur.description else []
                cols = [d[0] for d in cur.description] if cur.description else []
                return ExecutionResult(self.engine, "SUCCESS", rows, cols, elapsed_ms=t.ms)
            except self._pymysql.MySQLError as e:
                code = e.args[0] if e.args else None
                status = "TIMEOUT" if code in (3024, 1969) else "ERROR"
                return ExecutionResult(self.engine, status, error=str(e), error_class=str(code))

    # ---- schema introspection (used by replication) ---------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [tuple(r) for r in cur.fetchall()]

    def iter_table(self, db: str, table: str, batch: int = 50_000):
        """Stream a table in batches with an unbuffered cursor."""
        cur = self._conn.cursor(self._pymysql.cursors.SSCursor)
        try:
            cur.execute(f"SELECT * FROM `{db}`.`{table}`")
            while chunk := cur.fetchmany(batch):
                yield [tuple(r) for r in chunk]
        finally:
            cur.close()

    def close(self) -> None:
        self._conn.close()

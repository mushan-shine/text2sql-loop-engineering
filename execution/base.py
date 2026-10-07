"""Engine-neutral SQL execution interface."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import sqlglot
from sqlglot import exp


@dataclass
class ExecutionResult:
    engine: str
    status: str  # SUCCESS | ERROR | TIMEOUT | REJECTED
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    error_class: str | None = None  # engine error class / code, when available
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "SUCCESS"


class SqlExecutor(Protocol):
    engine: str

    def execute(self, sql: str, db: str) -> ExecutionResult: ...

    def close(self) -> None: ...


_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create,
    exp.Alter, exp.Command, exp.TruncateTable,
)


def is_read_only(sql: str, dialect: str) -> bool:
    """True only for a single SELECT-like statement. Unparseable SQL is not
    rejected here (the engine reports the syntax error) unless it smells of DML."""
    try:
        stmts = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError:
        head = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
        return head in ("SELECT", "WITH", "(")
    if len(stmts) != 1:
        return False
    return not any(isinstance(n, _WRITE_NODES) for n in stmts[0].walk())


class Timer:
    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = int((time.perf_counter() - self._t0) * 1000)

from __future__ import annotations

from typing import Callable

import pytest

from benchmark.beaver.dataset import BeaverCase
from execution.base import ExecutionResult


class FakeExecutor:
    """Returns scripted results; ``script`` maps sql → result or a callable (call_no) → result."""

    def __init__(self, engine: str, script: dict[str, ExecutionResult | Callable[[int], ExecutionResult]]):
        self.engine = engine
        self.script = script
        self.calls: list[tuple[str, str]] = []

    def execute(self, sql: str, db: str) -> ExecutionResult:
        self.calls.append((sql, db))
        r = self.script[sql]
        return r(len(self.calls)) if callable(r) else r

    def close(self) -> None:
        pass


def ok(engine: str, rows: list[tuple]) -> ExecutionResult:
    return ExecutionResult(engine, "SUCCESS", rows, [f"c{i}" for i in range(len(rows[0]) if rows else 0)])


def err(engine: str, cls: str, msg: str = "boom") -> ExecutionResult:
    return ExecutionResult(engine, "ERROR", error=f"[{cls}] {msg}", error_class=cls)


@pytest.fixture
def make_case() -> Callable[..., BeaverCase]:
    def _make(sql: str = "SELECT 1", case_id: str = "1", db: str = "dw") -> BeaverCase:
        return BeaverCase.from_beaver({
            "id": case_id, "question": "How many buildings?", "db": db, "sql": sql,
            "tables": '["FCLT_BUILDING"]', "column_mapping": {"buildings": "FCLT_BUILDING.BUILDING_KEY"},
            "join_keys": [], "domain_knowledge": [], "sub_questions": [], "sub_sqls": [],
        }, "dw")
    return _make

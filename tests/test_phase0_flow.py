"""Phase-0 orchestration against in-memory fakes (no network)."""
import json

import pytest

from benchmark.beaver import phase0
from dbx.catalog import Layout
from execution.databricks_sql import extract_error_class, session_settings
from execution.base import ExecutionResult
from tests.conftest import FakeExecutor, ok


class FakeRunner:
    """Captures project SQL and Delta writes."""

    def __init__(self, existing_cases: dict | None = None):
        self.sql: list[str] = []
        self.written: dict[str, list[dict]] = {}
        self.existing_cases = existing_cases
        self.workspace_config = None

    def run(self, sql: str):
        self.sql.append(sql)
        if sql.startswith("SHOW TABLES"):
            return [("benchmark", "cases", False)] if self.existing_cases is not None else []
        if "source_sha256 FROM" in sql:
            return list(self.existing_cases.items())
        return []


@pytest.fixture
def patched_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(phase0, "RUNS_DIR", tmp_path)
    captured: dict[str, list] = {}

    def fake_write(runner, layout, schema, table, rows, arrow_schema, mode="append"):
        import pyarrow as pa
        pa.Table.from_pylist(rows, schema=arrow_schema)  # rows must fit the declared schema
        captured.setdefault(table, []).extend(rows)
        return table

    monkeypatch.setattr(phase0, "write_rows", fake_write)
    return captured


def test_import_creates_and_view_hides_gold(make_case, patched_writes):
    r = FakeRunner()
    s = phase0.import_cases(r, Layout("cat"), [make_case(case_id="1"), make_case(case_id="2")],
                            {"T": {"db": "dw", "column_names": ["a"], "column_types": ["int"], "example_rows": []}})
    assert s["cases"] == 2 and s["cases_table"] == "created"
    view = next(q for q in r.sql if "cases_agent_view" in q)
    assert "gold" not in view.split("AS", 1)[1]


def test_reimport_refuses_to_overwrite_changed_cases(make_case, patched_writes):
    case = make_case("SELECT 1", case_id="1")
    r = FakeRunner(existing_cases={case.case_id: "different-hash"})
    with pytest.raises(RuntimeError, match="never overwritten"):
        phase0.import_cases(r, Layout("cat"), [case], {})


def test_identical_reimport_is_noop(make_case, patched_writes):
    case = make_case("SELECT 1", case_id="1")
    r = FakeRunner(existing_cases={case.case_id: case.source_sha256})
    assert phase0.import_cases(r, Layout("cat"), [case], {})["cases_table"].startswith("unchanged")
    assert "cases" not in patched_writes


def test_compatibility_run_summary(make_case, patched_writes, tmp_path):
    good = make_case("SELECT 1", "1")
    bad = make_case("SELECT RANK() OVER (ORDER BY b ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t", "2")
    adapted = "SELECT RANK() OVER (ORDER BY b) FROM t"
    frame_error = ExecutionResult("databricks", "ERROR", error="Window Frame specifiedwindowframe(RowFrame, -2, "
                                  "currentrow$()) must match the required frame")
    mysql = FakeExecutor("mysql", {good.gold_sql: ok("mysql", [(1,)]), bad.gold_sql: ok("mysql", [(1,)])})
    dbx = FakeExecutor("databricks", {good.gold_sql: ok("databricks", [(1,)]),
                                      bad.gold_sql: frame_error,
                                      adapted: ok("databricks", [(1,)])})
    s = phase0.run_compatibility([good, bad], mysql, dbx, Layout("cat"),
                                 phase0.QualificationConfig(repeats=2))
    assert s["by_status"]["COMPATIBLE"] == 1 and s["by_status"]["INCOMPATIBLE_FUNCTION"] == 1
    assert s["executes_unmodified_on_databricks"] == 1
    assert s["adaptations"] == {"RESULT_EQUIVALENT": 1}
    assert s["adaptations_by_rule"] == {"nonaggregate_window_frame_ignored": 1}
    assert s["usable_with_adaptations"] == 2
    assert json.loads((tmp_path / "03_compatibility.json").read_text(encoding="utf-8"))["cases"] == 2


def test_error_class_extraction_and_session():
    assert extract_error_class("[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column ...") == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    assert extract_error_class("no class") is None
    s = session_settings(False, 60, True)
    assert "SET ANSI_MODE = false" in s and "SET use_cached_result = false" in s

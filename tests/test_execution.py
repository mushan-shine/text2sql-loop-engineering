"""Step 0: engine-neutral execution helpers."""
from execution.base import ExecutionResult, is_read_only
from execution.databricks_sql import extract_error_class, session_settings


def test_only_a_single_read_only_statement_is_allowed():
    assert is_read_only("SELECT 1", "databricks")
    assert is_read_only("WITH t AS (SELECT 1 AS a) SELECT a FROM t", "databricks")
    assert not is_read_only("DROP TABLE x", "databricks")
    assert not is_read_only("SELECT 1; SELECT 2", "databricks")


def test_error_class_is_extracted_from_the_engine_message():
    assert extract_error_class("[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column ...") == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    assert extract_error_class("plain error") is None


def test_session_settings_are_identical_for_every_query():
    assert session_settings(False, 120, True) == [
        "SET ANSI_MODE = false", "SET STATEMENT_TIMEOUT = 120", "SET use_cached_result = false"]
    assert ExecutionResult("databricks", "SUCCESS").ok and not ExecutionResult("databricks", "ERROR").ok

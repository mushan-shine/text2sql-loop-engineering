from decimal import Decimal

from benchmark.beaver import adapter, compatibility as C
from benchmark.beaver.compatibility import classify_error, static_hazards, validate_case
from execution.base import ExecutionResult, is_read_only
from tests.conftest import FakeExecutor, err, ok


class TestClassifyError:
    def test_categories(self):
        assert classify_error("PARSE_SYNTAX_ERROR", "") == C.INCOMPATIBLE_SYNTAX
        assert classify_error("UNRESOLVED_ROUTINE", "") == C.INCOMPATIBLE_FUNCTION
        assert classify_error("DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES", "") == C.INCOMPATIBLE_FUNCTION
        assert classify_error("TABLE_OR_VIEW_NOT_FOUND", "") == C.INCOMPATIBLE_SCHEMA
        assert classify_error("UNRESOLVED_COLUMN.WITH_SUGGESTION", "") == C.INCOMPATIBLE_SCHEMA
        assert classify_error("MISSING_AGGREGATION", "") == C.INCOMPATIBLE_SEMANTICS
        assert classify_error("DIVIDE_BY_ZERO", "") == C.INCOMPATIBLE_SEMANTICS
        assert classify_error(None, "something odd") == C.UNKNOWN


class TestStaticHazards:
    def test_flags(self):
        hz = static_hazards("SELECT a, COUNT(*) FROM t WHERE name = 'X' GROUP BY b LIMIT 5")
        assert {"LIMIT_WITHOUT_ORDER_BY", "NONAGGREGATED_COLUMN_IN_GROUP_BY",
                "STRING_COMPARISON_COLLATION"} <= set(hz)

    def test_clean_query(self):
        assert static_hazards("SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY a LIMIT 5") == []

    def test_qualified_table(self):
        assert "DB_QUALIFIED_TABLE" in static_hazards("SELECT * FROM dw.t")


class TestReadOnlyGuard:
    def test_select_ok(self):
        assert is_read_only("SELECT 1", "mysql")
        assert is_read_only("WITH x AS (SELECT 1) SELECT * FROM x", "mysql")

    def test_rejects_writes_and_multi(self):
        assert not is_read_only("DELETE FROM t", "mysql")
        assert not is_read_only("SELECT 1; DROP TABLE t", "mysql")
        assert not is_read_only("CREATE TABLE x AS SELECT 1", "mysql")


class TestValidateCase:
    def test_compatible_when_results_match_across_representations(self, make_case):
        case = make_case("SELECT AVG(x) FROM t")
        ref = FakeExecutor("mysql", {case.gold_sql: ok("mysql", [(Decimal("2.5000"),)])})
        cand = FakeExecutor("databricks", {case.gold_sql: ok("databricks", [(2.5,)])})
        rec, _, _ = validate_case(case, ref, cand, repeats=3)
        assert rec.compatibility_status == C.COMPATIBLE
        assert rec.reference_stable and rec.databricks_stable
        assert len(ref.calls) == 3 and len(cand.calls) == 3

    def test_executes_but_differs_is_semantics(self, make_case):
        case = make_case("SELECT COUNT(*) FROM t WHERE name = 'abc'")
        ref = FakeExecutor("mysql", {case.gold_sql: ok("mysql", [(3,)])})     # case-insensitive
        cand = FakeExecutor("databricks", {case.gold_sql: ok("databricks", [(1,)])})
        rec, _, _ = validate_case(case, ref, cand)
        assert rec.compatibility_status == C.INCOMPATIBLE_SEMANTICS
        assert rec.execution_status == "SUCCESS" and rec.set_match is False

    def test_error_is_classified(self, make_case):
        case = make_case("SELECT DATE_FORMAT(d, '%Y') FROM t")
        ref = FakeExecutor("mysql", {case.gold_sql: ok("mysql", [("2020",)])})
        cand = FakeExecutor("databricks", {case.gold_sql: err("databricks", "UNRESOLVED_ROUTINE")})
        rec, _, _ = validate_case(case, ref, cand)
        assert rec.compatibility_status == C.INCOMPATIBLE_FUNCTION

    def test_reference_failure(self, make_case):
        case = make_case("SELECT broken")
        ref = FakeExecutor("mysql", {case.gold_sql: err("mysql", "1054")})
        cand = FakeExecutor("databricks", {case.gold_sql: ok("databricks", [(1,)])})
        rec, _, _ = validate_case(case, ref, cand)
        assert rec.compatibility_status == C.REFERENCE_FAILED

    def test_unstable_result_is_not_compatible(self, make_case):
        case = make_case("SELECT id FROM t LIMIT 1")
        ref = FakeExecutor("mysql", {case.gold_sql: ok("mysql", [(1,)])})
        cand = FakeExecutor("databricks", {case.gold_sql: lambda n: ok("databricks", [(n % 2,)])})
        rec, _, _ = validate_case(case, ref, cand, repeats=3)
        assert rec.compatibility_status == C.INCOMPATIBLE_SEMANTICS
        assert not rec.databricks_stable
        assert "LIMIT_WITHOUT_ORDER_BY" in rec.static_hazards


class TestAdapterRules:
    def test_population_statistics(self):
        sql = "SELECT STD(x), VARIANCE(y) OVER (PARTITION BY a), stddev(z), t.std, 'VARIANCE(' FROM t"
        out, rules = adapter.apply_rules(sql)
        assert rules == ["mysql_population_statistics"]
        assert out == ("SELECT STDDEV_POP(x), VAR_POP(y) OVER (PARTITION BY a), STDDEV_POP(z), t.std, "
                       "'VARIANCE(' FROM t")  # qualified names and string literals untouched

    def test_sample_statistics_untouched(self):
        sql = "SELECT VAR_SAMP(x), STDDEV_SAMP(y), VAR_POP(z) FROM t"
        assert adapter.apply_rules(sql) == (sql, [])

    def test_nonaggregate_frame_removed(self):
        sql = ("SELECT RANK() OVER (PARTITION BY a ORDER BY b ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS r, "
               "LAG(x, 1) OVER (ORDER BY b ROWS 2 PRECEDING) AS l, "
               "SUM(x) OVER (ORDER BY b ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS s FROM t")
        out, rules = adapter.apply_rules(sql)
        assert rules == ["nonaggregate_window_frame_ignored"]
        assert "RANK() OVER (PARTITION BY a ORDER BY b) AS r" in out
        assert "LAG(x, 1) OVER (ORDER BY b) AS l" in out
        # aggregate window functions keep their frame: there it is meaningful
        assert "SUM(x) OVER (ORDER BY b ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS s" in out

    def test_rules_combine(self):
        out, rules = adapter.apply_rules("SELECT VARIANCE(x), ROW_NUMBER() OVER (ORDER BY y ROWS UNBOUNDED PRECEDING) FROM t")
        assert set(rules) == {"mysql_population_statistics", "nonaggregate_window_frame_ignored"}
        assert out == "SELECT VAR_POP(x), ROW_NUMBER() OVER (ORDER BY y) FROM t"


class TestAdapterValidation:
    def _ref_runs(self, case, rows):
        ref = FakeExecutor("mysql", {case.gold_sql: ok("mysql", rows)})
        _, ref_runs, _ = validate_case(case, ref, FakeExecutor("databricks", {case.gold_sql: err("databricks", "X")}))
        return ref_runs

    def test_adaptation_validated_by_result_equivalence(self, make_case):
        case = make_case("SELECT VARIANCE(a) FROM t")
        adapted, _ = adapter.apply_rules(case.gold_sql)
        cand = FakeExecutor("databricks", {adapted: ok("databricks", [(2.25,)])})
        rec = adapter.try_adapt(case, self._ref_runs(case, [(2.25,)]), cand, repeats=2)
        assert rec.semantic_validation == adapter.RESULT_EQUIVALENT
        assert rec.adaptation_rule == "mysql_population_statistics"
        assert rec.adapted_result_hash and "dev.mysql.com" in rec.detail
        assert rec.original_gold_sql == case.gold_sql  # original is never modified

    def test_adaptation_rejected_when_result_differs(self, make_case):
        case = make_case("SELECT VARIANCE(a) FROM t")
        adapted, _ = adapter.apply_rules(case.gold_sql)
        rec = adapter.try_adapt(case, self._ref_runs(case, [(2.25,)]),
                                FakeExecutor("databricks", {adapted: ok("databricks", [(3.0,)])}), 2)
        assert rec.semantic_validation == adapter.RESULT_DIFFERS

    def test_no_adaptation_when_no_rule_applies(self, make_case):
        case = make_case("SELECT a FROM t")
        rec = adapter.try_adapt(case, None, FakeExecutor("databricks", {}), 1)  # type: ignore[arg-type]
        assert rec.semantic_validation == adapter.NO_ADAPTATION


def test_window_frame_error_is_function_incompatibility():
    msg = "Window Frame specifiedwindowframe(RowFrame, -2, currentrow$()) must match the required frame"
    assert classify_error(None, msg) == C.INCOMPATIBLE_FUNCTION
    assert classify_error("INCOMPATIBLE_COLUMN_TYPE", "") == C.INCOMPATIBLE_SCHEMA


def test_execution_result_ok_flag():
    assert ExecutionResult("x", "SUCCESS").ok and not ExecutionResult("x", "ERROR").ok

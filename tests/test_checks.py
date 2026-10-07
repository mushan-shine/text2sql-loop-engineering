"""Semantic verifier checks (loop_engineer/checks.py): each fires on its target and stays quiet otherwise."""
from loop_engineer.checks import check_result, check_static, expected_outputs


def signals(q, sql, rows=None):
    return {f.signal for f in check_static(q, sql) + check_result(q, sql, rows)}


def test_missing_literal_quoted_and_compared_values():
    q = "For each term with status 'P' and more than 5 students in 2022, give the term code and the count."
    ok = "SELECT term_code, COUNT(*) FROM t WHERE status = 'P' AND n > 5 AND year = 2022 GROUP BY term_code"
    assert "missing_literal" not in signals(q, ok)
    bad = "SELECT term_code, COUNT(*) FROM t WHERE n > 5 AND year = 2022 GROUP BY term_code"
    assert "missing_literal" in signals(q, bad)
    # apostrophes and non-comparison numbers are not filter values
    assert "missing_literal" not in signals("Show each department's top 10 subjects' titles.", "SELECT a FROM t")


def test_missing_aggregate_and_grouping():
    q = "For each department, provide the department name and the average number of students."
    assert "missing_aggregate" in signals(q, "SELECT dept, SUM(n) FROM t GROUP BY dept")
    assert "missing_grouping" in signals(q, "SELECT dept, AVG(n) FROM t")
    assert not {"missing_aggregate", "missing_grouping"} & signals(q, "SELECT dept, AVG(n) FROM t GROUP BY dept")
    assert "missing_grouping" not in signals(q, "SELECT dept, AVG(n) OVER (PARTITION BY dept) FROM t")


def test_rounding_both_directions():
    assert "rounding" in signals("Give the average. Do not return any rounded answers.", "SELECT ROUND(AVG(x), 2) FROM t")
    assert "rounding" in signals("Give the average rounded to 2 decimals.", "SELECT AVG(x) FROM t")
    assert "rounding" not in signals("Give the average. Do not return any rounded answers.", "SELECT AVG(x) FROM t")


def test_join_checks():
    q = "List the names."
    assert "join_tautology" in signals(q, "SELECT a.x FROM a JOIN b ON a.k = a.k")
    assert "join_tautology" not in signals(q, "SELECT a.x FROM a JOIN b ON a.k = b.k")
    assert "join_without_condition" in signals(q, "SELECT a.x FROM a LEFT JOIN b")


def test_output_columns_and_result_checks():
    assert expected_outputs("provide the department name, the average, and the variance of units") == 3
    assert "output_columns" in signals("Provide the name, the average, and the variance.", "SELECT name, AVG(x) FROM t")
    rows = [("a", 1), ("a", 1), ("b", 2)]
    assert "duplicate_rows" in signals("List names.", "SELECT n, v FROM t", rows)
    assert "duplicate_rows" not in signals("List names.", "SELECT DISTINCT n, v FROM t", rows)
    assert "single_row_for_each" in signals("For each department, give the name.", "SELECT n FROM t", [("a",)])


# ---- numeric consistency ("calculator") checks

from loop_engineer.checks import check_numeric, output_roles  # noqa: E402

CTE = ("WITH c AS (SELECT dept, AVG(n) AS a, MIN(n) AS lo, MAX(n) AS hi, STDDEV_POP(n) AS s, VAR_POP(n) AS v, "
       "COUNT(*) AS k FROM t GROUP BY dept) SELECT c.dept, c.a, c.lo, c.hi, c.s, c.v, c.k FROM c")


def nsig(sql, rows):
    return {f.signal for f in check_numeric(sql, rows)}


def test_roles_follow_cte_aliases_and_skip_uncertain_columns():
    roles = output_roles(CTE)
    assert [r.role if r else None for r in roles] == [None, "avg", "min", "max", "std", "var", "count"]
    assert roles[4].family == "pop" and roles[1].arg == "n"
    assert output_roles("SELECT ROUND(AVG(x), 2), COUNT(DISTINCT x) FROM t") == [None, None]
    assert output_roles("SELECT STD(x) FROM t")[0].role == "std"          # MySQL STD, anonymous function


def test_consistent_numbers_pass_and_each_violation_is_caught():
    assert nsig(CTE, [("x", 5, 1, 9, 2, 4, 3), ("y", "2.5", "2", "3", "0.5", "0.25", "2")]) == set()
    assert nsig(CTE, [("x", 12, 1, 9, 2, 4, 3)]) == {"avg_outside_min_max"}
    assert nsig(CTE, [("x", 5, 9, 1, 2, 4, 3)]) >= {"min_greater_than_max"}
    assert nsig(CTE, [("x", 5, 1, 9, 2, 5, 3)]) == {"std_var_mismatch"}
    assert nsig(CTE, [("x", 5, 1, 9, 2, 4, 2.5)]) == {"count_not_integer"}
    assert nsig(CTE, [("x", 5, 1, 9, -2, 4, 3)]) >= {"negative_statistic"}
    assert nsig(CTE, [("x", 5, 4, 6, 3, 9, 3)]) == {"std_exceeds_range"}
    assert nsig(CTE, [("x", None, None, None, None, None, 0)]) == set()   # NULLs are not violations


def test_mixed_or_default_families_are_never_compared():
    sql = "SELECT STDDEV(x), VAR_POP(x) FROM t"          # sample std vs population var: no rule applies
    assert nsig(sql, [(2.0, 3.0)]) == set()
    assert nsig("SELECT STDDEV(x), VARIANCE(x) FROM t", [(2.0, 3.0)]) == set()

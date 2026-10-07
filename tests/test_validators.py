"""Tool-grounded validators (loop_engineer/validators.py) against a fake warehouse."""
from agent.retriever import SchemaCatalog
from execution.base import ExecutionResult
from loop_engineer.validators import ValidatorAgent

CAT = SchemaCatalog.build("dw", [
    ("dept", "CODE", "STRING"), ("dept", "NAME", "STRING"), ("dept", "SCHOOL", "STRING"),
    ("subj", "DEPT_CODE", "STRING"), ("subj", "ENROLLED", "BIGINT"),
    ("descr", "DEPT", "STRING"), ("descr", "TITLE", "STRING"),
], {})
VALUES = {("dept", "school"): ["School of Engineering", "Science"], ("dept", "name"): ["Mathematics", "Physics"]}
KEYS = {"dept": (10, 10), "subj": (500, 40), "descr": (695, 72)}   # rows, distinct join keys


class FakeWarehouse:
    def __init__(self):
        self.sql = []

    def execute(self, sql, db, max_rows=None):
        self.sql.append(sql)
        s = sql.lower()
        if s.startswith("select 1"):
            t, c = s.split("from `dw`.`")[1].split("`")[0], s.split("where `")[1].split("`")[0]
            v = sql.split("= '")[1].rsplit("'", 1)[0]
            ok = any(x.lower() == v.lower() for x in VALUES.get((t, c), []))
            return ExecutionResult("databricks", "SUCCESS", [(1,)] if ok else [], ["1"])
        if s.startswith("select distinct"):
            t, c = s.split("from `dw`.`")[1].split("`")[0], s.split("select distinct `")[1].split("`")[0]
            return ExecutionResult("databricks", "SUCCESS", [(v,) for v in VALUES.get((t, c), [])], [c])
        if s.startswith("select count(*)"):
            t = s.split("from `dw`.`")[1].split("`")[0]
            return ExecutionResult("databricks", "SUCCESS", [KEYS[t]], ["n", "d"])
        return ExecutionResult("databricks", "ERROR", error="unexpected probe")


def run(sql, **kw):
    agent = ValidatorAgent(FakeWarehouse(), CAT, **kw)
    return agent, {f.signal: f for f in agent.validate(sql)}


def test_missing_filter_value_with_closest_real_values():
    agent, fs = run("SELECT d.NAME FROM DEPT d WHERE d.SCHOOL = 'Engineering'")
    f = fs["filter_value_not_found"]
    assert f.evidence["closest"][0] == "School of Engineering" and "does not occur in dept.school" in f.hint
    assert "filter_value_not_found" not in run("SELECT d.NAME FROM DEPT d WHERE d.SCHOOL = 'science'")[1]  # case-insensitive
    assert "filter_value_not_found" in run("SELECT NAME FROM DEPT WHERE NAME IN ('Mathematics', 'Physicz')")[1]


def test_negations_numbers_and_non_string_columns_are_not_probed():
    agent, fs = run("SELECT NAME FROM DEPT WHERE NOT (SCHOOL = 'Nope') AND CODE <> 'x'")
    assert fs == {} and agent.probes_run == 0
    agent, fs = run("SELECT DEPT_CODE FROM SUBJ WHERE ENROLLED = '5'")
    assert fs == {} and agent.probes_run == 0


def test_fanout_only_when_the_other_side_key_is_not_unique():
    ok = "SELECT d.NAME, SUM(s.ENROLLED) FROM SUBJ s JOIN DEPT d ON s.DEPT_CODE = d.CODE GROUP BY d.NAME"
    assert "join_fanout_aggregate" not in run(ok)[1]          # dept.CODE is unique: no inflation
    bad = ("SELECT d.NAME, SUM(s.ENROLLED) FROM SUBJ s JOIN DEPT d ON s.DEPT_CODE = d.CODE "
           "JOIN DESCR x ON x.DEPT = d.CODE GROUP BY d.NAME")
    f = run(bad)[1]["join_fanout_aggregate"]
    assert f.evidence["joined_table"] == "descr" and f.evidence["rows"] == 695 and f.evidence["distinct_keys"] == 72
    assert "SUM(subj.enrolled)" in f.hint
    distinct = bad.replace("SUM(s.ENROLLED)", "COUNT(DISTINCT s.ENROLLED)")
    assert "join_fanout_aggregate" not in run(distinct)[1]


def test_probe_budget_and_cache():
    sql = "SELECT NAME FROM DEPT WHERE SCHOOL = 'a' AND NAME = 'b' AND CODE = 'c'"
    agent, _ = run(sql, max_probes=1)
    assert agent.probes_run == 1

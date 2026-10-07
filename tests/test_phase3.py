"""Phase 3: failure taxonomy labeler (evaluation only)."""
import pytest

from benchmark.beaver import subtasks as S
from benchmark.beaver.dataset import BeaverCase

SCHEMA = {
    "sis_department": {"department_code": "STRING", "department_name": "STRING"},
    "subject_offered": {"subject_id": "STRING", "department_code": "STRING", "num_enrolled": "INT", "term_code": "STRING"},
    "academic_terms": {"term_code": "STRING", "academic_year": "STRING"},
}
GOLD = ("SELECT d.DEPARTMENT_NAME, AVG(s.NUM_ENROLLED) FROM SIS_DEPARTMENT d "
        "JOIN SUBJECT_OFFERED s ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE "
        "WHERE d.DEPARTMENT_NAME = 'Mathematics' GROUP BY d.DEPARTMENT_NAME")


@pytest.fixture
def case() -> BeaverCase:
    return BeaverCase.from_beaver({
        "id": "1", "db": "dw", "sql": GOLD,
        "question": "For Course 18, what is the department name and average enrollment?",
        # ACADEMIC_TERMS is annotated but unused by the gold SQL (BEAVER over-lists tables)
        "tables": ["SIS_DEPARTMENT", "SUBJECT_OFFERED", "ACADEMIC_TERMS"],
        "column_mapping": {"department name": ["SIS_DEPARTMENT.DEPARTMENT_NAME"],
                           "average enrollment": ["SUBJECT_OFFERED.NUM_ENROLLED"]},
        "join_keys": [["SIS_DEPARTMENT.DEPARTMENT_CODE", "SUBJECT_OFFERED.DEPARTMENT_CODE"]],
        "domain_knowledge": ["\"Course 18\" is predicated by \"TABLE.DEPARTMENT_NAME = 'Mathematics'\""],
    }, "dw")


def lab(case, sql, status="SUCCESS", retrieved=("sis_department", "subject_offered")):
    return S.label_failure(case, sql, status, list(retrieved), SCHEMA)


def test_gold_passes_its_own_checks_and_unused_annotation_is_dropped(case):
    r = lab(case, GOLD)
    assert r.primary == S.UNKNOWN and r.failed_checks == []
    assert r.evidence["annotation_tables_unused_by_gold_sql"] == ["academic_terms"]


def test_equivalent_rewrite_with_other_aliases_passes(case):
    sql = ("SELECT x.department_name, avg(y.num_enrolled) FROM sis_department AS x, subject_offered AS y "
           "WHERE y.department_code = x.department_code AND x.department_name = 'Mathematics' GROUP BY 1")
    assert lab(case, sql).failed_checks == []


def test_table_retrieval(case):
    r = lab(case, "SELECT DEPARTMENT_NAME, 1 FROM SIS_DEPARTMENT WHERE DEPARTMENT_NAME = 'Mathematics' GROUP BY 1",
            retrieved=("sis_department",))
    assert r.primary == S.TABLE_RETRIEVAL
    assert r.evidence["missing_tables"] == {"subject_offered": "not_retrieved"}


def test_column_mapping_wrong_alias(case):
    # num_enrolled attached to the wrong table (the dominant baseline error)
    sql = ("SELECT d.DEPARTMENT_NAME, AVG(d.NUM_ENROLLED) FROM SIS_DEPARTMENT d JOIN SUBJECT_OFFERED s "
           "ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE WHERE d.DEPARTMENT_NAME = 'Mathematics' GROUP BY d.DEPARTMENT_NAME")
    r = lab(case, sql, status="ERROR")
    assert r.primary == S.COLUMN_MAPPING and "subject_offered.num_enrolled" in r.evidence["missing_columns"]


def test_join_key(case):
    sql = ("SELECT d.DEPARTMENT_NAME, AVG(s.NUM_ENROLLED) FROM SIS_DEPARTMENT d JOIN SUBJECT_OFFERED s "
           "ON d.DEPARTMENT_NAME = s.SUBJECT_ID WHERE d.DEPARTMENT_NAME = 'Mathematics' GROUP BY d.DEPARTMENT_NAME")
    assert lab(case, sql).primary == S.JOIN_KEY


def test_domain_knowledge(case):
    sql = GOLD.replace("'Mathematics'", "'Course 18'")
    r = lab(case, sql)
    assert r.primary == S.DOMAIN_KNOWLEDGE and r.evidence["missing_domain_knowledge"]


def test_query_decomposition(case):
    sql = ("SELECT d.DEPARTMENT_NAME, s.NUM_ENROLLED FROM SIS_DEPARTMENT d JOIN SUBJECT_OFFERED s "
           "ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE WHERE d.DEPARTMENT_NAME = 'Mathematics'")
    r = lab(case, sql)
    assert r.primary == S.QUERY_DECOMPOSITION and set(r.evidence["missing_operations"]) == {"avg", "group_by"}


def test_execution_and_unparseable(case):
    assert lab(case, GOLD, status="ERROR").primary == S.EXECUTION
    r = lab(case, "SELECT CASE WHEN FROM", status="ERROR")
    assert r.primary == S.EXECUTION and not r.evidence["sql_parsed"]


def test_priority_order():
    assert S.PRIORITY == (S.TABLE_RETRIEVAL, S.COLUMN_MAPPING, S.JOIN_KEY, S.DOMAIN_KNOWLEDGE,
                          S.QUERY_DECOMPOSITION, S.EXECUTION)


def test_intervention_hints_and_classification(case):
    from evaluation.intervention import MULTI_CAUSE, MULTIPLE_SUFFICIENT, UNRESOLVED, build_hints, classify
    h = build_hints(case, SCHEMA)
    assert set(h) == {"tables", "columns", "join_keys", "domain_knowledge"}  # single sub-question -> no decomposition
    assert "academic_terms" not in h["tables"].lines[0]  # annotation-only table is not hinted
    assert any("subject_offered.num_enrolled" in ln for ln in h["columns"].lines)
    assert "Course 18" in h["domain_knowledge"].lines[0]
    assert classify({"tables": False, "columns": True}, True) == (S.COLUMN_MAPPING, ["columns"])
    assert classify({"tables": True, "columns": True}, True)[0] == MULTIPLE_SUFFICIENT
    assert classify({"tables": False}, True)[0] == MULTI_CAUSE
    assert classify({"tables": False}, False)[0] == UNRESOLVED


def test_oracle_hints_only_appear_when_passed():
    from agent.generator import build_prompt
    from benchmark.beaver.dataset import AgentTask
    t = AgentTask("dw:1", "q?", "dw")
    assert "Verified facts" not in build_prompt(t, "TABLE x", [])
    assert "- use t.a" in build_prompt(t, "TABLE x", [], oracle_hints=["use t.a"])

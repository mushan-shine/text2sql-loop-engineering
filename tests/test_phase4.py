"""Phase 4: observer whitelist, rule diagnosis, LLM parsing, accuracy scoring."""
import dataclasses

import pytest

from agent.llm import LlmResponse
from agent.retriever import SchemaCatalog
from evaluation.diagnosis_eval import score
from loop_engineer import diagnose as D
from loop_engineer.observer import OBSERVABLE_FIELDS, Observation, observe

CAT = SchemaCatalog.build("dw", [
    ("sis_department", "DEPARTMENT_CODE", "STRING"), ("sis_department", "DEPARTMENT_NAME", "STRING"),
    ("subject_offered", "NUM_ENROLLED", "INT"), ("subject_offered", "DEPARTMENT_CODE", "STRING"),
    ("fclt_building_address", "POSTAL_CODE", "STRING"),
], {})
SQL = "SELECT d.DEPARTMENT_NAME, AVG(d.NUM_ENROLLED) FROM SIS_DEPARTMENT d JOIN SUBJECT_OFFERED s ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE"


def rec(**kw):
    base = {"case_id": "dw:1", "attempt_id": 1, "question": "q", "retrieved_tables": ["sis_department", "subject_offered"],
            "generated_sql": SQL, "parse_status": "OK", "execution_status": "ERROR", "execution_error": None,
            "result_row_count": None, "result_preview": None,
            # gold-derived / evaluation fields that must never pass the observer
            "correct": False, "eval_message": "Values mismatch", "eval_table_recall": 0.5, "category": "c"}
    return {**base, **kw}


def unresolved(col, qual="d"):
    return (f"[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column, variable, or function parameter with name `{qual}`.`{col}` "
            "cannot be resolved. Did you mean one of the following? [`s`.`NUM_ENROLLED`, `d`.`DEPARTMENT_NAME`].")


def test_observer_whitelist_blocks_gold_and_correctness():
    obs = observe(rec(), "dw")
    names = {f.name for f in dataclasses.fields(Observation)}
    for forbidden in ("correct", "eval_message", "eval_table_recall", "category", "gold_sql", "gold_tables"):
        assert forbidden not in names
    assert "correct" not in OBSERVABLE_FIELDS and obs.question == "q"


def test_observer_parses_engine_error():
    obs = observe(rec(execution_error=unresolved("NUM_ENROLLED")), "dw")
    assert obs.error_class == "UNRESOLVED_COLUMN" and obs.unresolved_qualifier == "d"
    assert obs.unresolved_column == "NUM_ENROLLED" and "s.NUM_ENROLLED" in obs.suggestions


@pytest.mark.parametrize("col,retrieved,sql,expected,case", [
    ("NUM_ENROLLED", ["sis_department", "subject_offered"], SQL, D.COLUMN_MAPPING, "wrong_alias"),
    ("POSTAL_CODE", ["sis_department", "subject_offered", "fclt_building_address"], SQL, D.TABLE_RETRIEVAL, "table_not_used"),
    ("POSTAL_CODE", ["sis_department", "subject_offered"], SQL, D.TABLE_RETRIEVAL, "not_retrieved"),
    ("IS_NO_COURSE_MATERIAL", ["sis_department"], SQL, D.COLUMN_MAPPING, "hallucinated"),
])
def test_unresolved_column_rules(col, retrieved, sql, expected, case):
    d = D.diagnose_by_rules(observe(rec(execution_error=unresolved(col), retrieved_tables=retrieved,
                                        generated_sql=sql), "dw"), CAT)
    assert d.failure_type == expected and d.repair_hints["case"] == case and d.source == "rule"


def test_other_rules():
    parse = observe(rec(execution_error="[PARSE_SYNTAX_ERROR] Syntax error at or near 'JOIN'"), "dw")
    assert D.diagnose_by_rules(parse, CAT).failure_type == D.EXECUTION
    nosql = observe(rec(parse_status="NO_SQL", execution_status="NO_SQL", generated_sql=""), "dw")
    assert D.diagnose_by_rules(nosql, CAT).failure_type == D.EXECUTION
    tnf = observe(rec(execution_error="[TABLE_OR_VIEW_NOT_FOUND] The table or view `fac_building` cannot be found."), "dw")
    d = D.diagnose_by_rules(tnf, CAT)
    assert d.failure_type == D.TABLE_RETRIEVAL and d.repair_hints["missing_table"] == "fac_building"
    big = observe(rec(execution_status="TOO_MANY_ROWS"), "dw")
    assert D.diagnose_by_rules(big, CAT).failure_type == D.JOIN_KEY
    ok = observe(rec(execution_status="SUCCESS", result_row_count=3, result_preview='[["a"]]'), "dw")
    assert D.diagnose_by_rules(ok, CAT) is None


def test_llm_stage_only_without_signal():
    class Chat:
        model, calls = "m", 0

        def complete(self, prompt, system=None):
            Chat.calls += 1
            return LlmResponse('```json\n{"failure_type": "domain_knowledge_failure", "confidence": 1.4, "reason": "x"}\n```',
                               "m", 10, 5, 3)

    dg = D.Diagnoser(CAT, Chat())
    d, usage = dg.diagnose(observe(rec(execution_status="SUCCESS", result_row_count=0), "dw"))
    assert d.failure_type == D.DOMAIN_KNOWLEDGE and d.confidence == 1.0 and d.source == "llm" and usage["input_tokens"] == 10
    dg.diagnose(observe(rec(execution_error=unresolved("NUM_ENROLLED")), "dw"))
    assert Chat.calls == 1  # explicit signal -> rules, no LLM call
    assert D.parse_llm_diagnosis("not json") is None
    assert D.parse_llm_diagnosis('{"failure_type": "SOMETHING_ELSE"}') is None


def test_rules_only_fallback():
    d, usage = D.Diagnoser(CAT).diagnose(observe(rec(execution_status="SUCCESS", result_row_count=1), "dw"))
    assert d.failure_type == D.UNKNOWN and usage == {}


def test_accuracy_strict_and_lenient():
    diags = {"a": {"failure_type": D.COLUMN_MAPPING, "confidence": 0.9, "source": "rule"},
             "b": {"failure_type": D.TABLE_RETRIEVAL, "confidence": 0.8, "source": "rule"},
             "c": {"failure_type": D.JOIN_KEY, "confidence": 0.4, "source": "llm"}}
    labels = {"a": {"actual_failure_type": D.TABLE_RETRIEVAL, "failed_checks": [D.TABLE_RETRIEVAL, D.COLUMN_MAPPING]},
              "b": {"actual_failure_type": D.TABLE_RETRIEVAL, "failed_checks": [D.TABLE_RETRIEVAL]},
              "c": {"actual_failure_type": D.EXECUTION, "failed_checks": [D.EXECUTION]}}
    s, rows = score(diags, labels)
    assert s["diagnosis_accuracy_strict"].startswith("1/3") and s["diagnosis_accuracy_lenient"].startswith("2/3")
    assert s["by_source"]["llm"]["n"] == 1 and len(rows) == 3


@pytest.mark.parametrize("ref", ["`fac_building`", "`dw`.`fac_building`", "dw.fac_building",
                                 "`self_healing_text2sql`.`dw`.`fac_building`"])
def test_missing_table_is_the_table_not_the_schema(ref):
    # models copy the "dw." prefix from the prompt's "TABLE dw.xxx"; the table is the last part of the reference
    err = f"[TABLE_OR_VIEW_NOT_FOUND] The table or view {ref} cannot be found. Verify the spelling and correctness."
    obs = observe(rec(execution_error=err), "dw")
    assert obs.error_class == "TABLE_OR_VIEW_NOT_FOUND" and obs.missing_table == "fac_building"
    assert D.diagnose_by_rules(obs, CAT).repair_hints["missing_table"] == "fac_building"

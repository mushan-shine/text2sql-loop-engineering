"""Phase 5: join graph, repair skills, policy, gold isolation."""
import ast
from pathlib import Path

import pytest

from agent.join_graph import connect, join_candidates
from agent.llm import LlmResponse
from agent.retriever import SchemaCatalog
from loop_engineer import diagnose as D
from loop_engineer.observer import observe
from loop_engineer.policy import FALLBACK, Policy
from skills.base import RepairContext
from skills.retrieve_again import RetrieveAgain
from skills.schema_search import SchemaSearch, fix_column_refs

CAT = SchemaCatalog.build("dw", [
    ("sis_department", "DEPARTMENT_CODE", "STRING"), ("sis_department", "DEPARTMENT_NAME", "STRING"),
    ("subject_offered", "DEPARTMENT_CODE", "STRING"), ("subject_offered", "NUM_ENROLLED", "INT"),
    ("subject_offered", "TERM_CODE", "STRING"),
    ("fclt_building", "FCLT_BUILDING_KEY", "STRING"), ("fclt_building", "BUILDING_NAME", "STRING"),
    ("fclt_building_address", "FCLT_BUILDING_KEY", "STRING"), ("fclt_building_address", "POSTAL_CODE", "STRING"),
    ("fclt_building_address", "WAREHOUSE_LOAD_DATE", "STRING"), ("fclt_building", "WAREHOUSE_LOAD_DATE", "STRING"),
], {})
SQL = ("SELECT d.DEPARTMENT_NAME, AVG(d.NUM_ENROLLED) AS a FROM SIS_DEPARTMENT d "
       "JOIN SUBJECT_OFFERED s ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE GROUP BY d.DEPARTMENT_NAME")


class Chat:
    model = "m"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return LlmResponse("```sql\nSELECT 1\n```", "m", 100, 10, 5)


def obs_for(err, sql=SQL, retrieved=("sis_department", "subject_offered")):
    return observe({"case_id": "dw:1", "attempt_id": 1, "question": "avg enrollment per department",
                    "retrieved_tables": list(retrieved), "generated_sql": sql, "parse_status": "OK",
                    "execution_status": "ERROR", "execution_error": err}, "dw")


def unresolved(q, col):
    return f"[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column ... with name `{q}`.`{col}` cannot be resolved. Did you mean one of the following? [`s`.`NUM_ENROLLED`]."


def test_join_graph_uses_key_like_shared_columns_only():
    js = {j.sql() for j in join_candidates(CAT, ["fclt_building", "fclt_building_address"])}
    assert js == {"fclt_building.FCLT_BUILDING_KEY = fclt_building_address.FCLT_BUILDING_KEY"}  # load date excluded
    assert connect(CAT, "subject_offered", ["sis_department"])[0].column == "DEPARTMENT_CODE"


def test_fix_column_refs_fixes_every_wrong_reference_in_one_pass():
    sql = ("SELECT d.DEPARTMENT_NAME, AVG(d.NUM_ENROLLED) AS a, MAX(d.TERM_CODE) AS t FROM SIS_DEPARTMENT d "
           "JOIN SUBJECT_OFFERED s ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE GROUP BY d.DEPARTMENT_NAME")
    fix = fix_column_refs(sql, CAT)
    assert fix.changes == ["d.NUM_ENROLLED -> s.NUM_ENROLLED", "d.TERM_CODE -> s.TERM_CODE"]
    assert not fix.unresolved and "AVG(s.NUM_ENROLLED)" in fix.sql and "MAX(s.TERM_CODE)" in fix.sql
    assert "d.DEPARTMENT_NAME" in fix.sql  # correct references untouched


def test_fix_column_refs_leaves_what_the_schema_cannot_decide():
    amb = fix_column_refs("SELECT d.X_UNKNOWN, s.DEPARTMENT_NAME FROM SIS_DEPARTMENT d JOIN SUBJECT_OFFERED s "
                          "ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE", CAT)
    assert amb.changes == ["s.DEPARTMENT_NAME -> d.DEPARTMENT_NAME"]
    assert amb.unresolved == ["d.X_UNKNOWN (no table in this scope has it)"]
    # DEPARTMENT_CODE is in both tables of the scope -> not a wrong reference; nothing to do
    ok = fix_column_refs("SELECT s.DEPARTMENT_CODE FROM SIS_DEPARTMENT d JOIN SUBJECT_OFFERED s "
                         "ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE", CAT)
    assert ok.changes == [] and ok.unresolved == []
    # references to a CTE are not checked against base tables
    cte = fix_column_refs("WITH x AS (SELECT d.DEPARTMENT_NAME AS n FROM SIS_DEPARTMENT d) SELECT x.n FROM x", CAT)
    assert cte.changes == [] and cte.unresolved == []
    assert not fix_column_refs("SELECT FROM WHERE", CAT).parsed


def test_fix_column_refs_is_scope_aware():
    # the alias s means different tables in the two scopes
    sql = ("WITH a AS (SELECT s.NUM_ENROLLED FROM SIS_DEPARTMENT s JOIN SUBJECT_OFFERED o "
           "ON s.DEPARTMENT_CODE = o.DEPARTMENT_CODE) "
           "SELECT s.DEPARTMENT_NAME FROM SUBJECT_OFFERED s JOIN SIS_DEPARTMENT t ON s.DEPARTMENT_CODE = t.DEPARTMENT_CODE")
    fix = fix_column_refs(sql, CAT)
    assert set(fix.changes) == {"s.NUM_ENROLLED -> o.NUM_ENROLLED", "s.DEPARTMENT_NAME -> t.DEPARTMENT_NAME"}


def test_schema_search_wrong_alias_uses_no_llm():
    obs = obs_for(unresolved("d", "NUM_ENROLLED"))
    diag = D.diagnose_by_rules(obs, CAT)
    chat = Chat()
    r = SchemaSearch().repair(obs, diag, RepairContext(CAT, chat))
    assert not r.used_llm and chat.prompts == [] and "s.NUM_ENROLLED" in r.repaired_sql
    assert "deterministic" in r.repair_action


def test_schema_search_hands_leftovers_to_llm_with_partly_fixed_sql():
    sql = ("SELECT d.DEPARTMENT_NAME, AVG(d.NUM_ENROLLED) AS a, d.NO_SUCH_COL AS z FROM SIS_DEPARTMENT d "
           "JOIN SUBJECT_OFFERED s ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE GROUP BY d.DEPARTMENT_NAME")
    obs = obs_for(unresolved("d", "NUM_ENROLLED"), sql=sql)
    chat = Chat()
    r = SchemaSearch().repair(obs, D.diagnose_by_rules(obs, CAT), RepairContext(CAT, chat))
    assert r.used_llm and "d.NUM_ENROLLED -> s.NUM_ENROLLED" in r.repair_action
    assert "AVG(s.NUM_ENROLLED)" in chat.prompts[0]  # the LLM starts from the partly fixed SQL
    assert "d.NO_SUCH_COL (no table in this scope has it)" in chat.prompts[0]


def test_retrieve_again_adds_owner_table_with_join_candidates():
    sql = "SELECT b.BUILDING_NAME, b.POSTAL_CODE FROM FCLT_BUILDING b"
    obs = obs_for(unresolved("b", "POSTAL_CODE"), sql=sql, retrieved=("fclt_building",))
    diag = D.diagnose_by_rules(obs, CAT)
    assert diag.repair_hints["case"] == "not_retrieved"
    chat = Chat()
    r = RetrieveAgain().repair(obs, diag, RepairContext(CAT, chat))
    assert r.used_llm and "fclt_building_address" in r.tables
    assert "fclt_building.FCLT_BUILDING_KEY = fclt_building_address.FCLT_BUILDING_KEY" in chat.prompts[0]
    assert "TABLE dw.fclt_building_address" in chat.prompts[0]


def test_policy_routes_and_ablations():
    d = lambda t: D.Diagnosis(t, 0.9, "r", "rule")  # noqa: E731
    p = Policy()
    assert p.route(d(D.TABLE_RETRIEVAL)).skill == "RetrieveAgain"
    assert p.route(d(D.COLUMN_MAPPING)).skill == "SchemaSearch"
    assert p.route(d(D.JOIN_KEY)).skill == "FindJoinPath"
    assert p.route(d(D.QUERY_DECOMPOSITION)).skill == "ReplanQuery"
    dk = p.route(d(D.DOMAIN_KNOWLEDGE))
    assert dk.skill == "ReplanQuery" and dk.fallback  # until RetrieveKnowledge exists
    assert p.route(d(D.UNKNOWN)).skill == FALLBACK
    assert Policy(mode="generic").route(d(D.TABLE_RETRIEVAL)).skill == FALLBACK
    r = Policy(disabled={"SchemaSearch"}).route(d(D.COLUMN_MAPPING))
    assert r.skill == FALLBACK and r.fallback


@pytest.mark.parametrize("path", sorted(Path("skills").glob("*.py")) + [Path("loop_engineer/policy.py"),
                                                                       Path("loop_engineer/diagnose.py"),
                                                                       Path("loop_engineer/observer.py")])
def test_loop_components_never_import_gold_or_evaluation(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("benchmark", "evaluation")) for m in mods), (path, mods)


def test_fix_never_turns_a_join_condition_into_a_self_comparison():
    """Regression (dw_4188): re-pointing a wrong column inside ON made 'sd.X = sd.X' - the query ran,
    but the tables were no longer related. Such references must be reverted and handed on."""
    cat = SchemaCatalog.build("dw", [
        ("academic_terms_all", "TERM_CODE", "STRING"), ("academic_terms_all", "ACADEMIC_YEAR", "STRING"),
        ("sis_department", "DEPARTMENT_CODE", "STRING"), ("sis_department", "DEPARTMENT_NAME", "STRING"),
        ("subject_offered", "TERM_CODE", "STRING"),
    ], {})
    sql = ("SELECT ata.ACADEMIC_YEAR, sd.DEPARTMENT_NAME, lib.ACADEMIC_YEAR AS y FROM ACADEMIC_TERMS_ALL ata "
           "JOIN SIS_DEPARTMENT sd ON ata.DEPARTMENT_CODE = sd.DEPARTMENT_CODE "
           "JOIN SUBJECT_OFFERED lib ON ata.TERM_CODE = lib.TERM_CODE AND ata.ACADEMIC_YEAR = lib.ACADEMIC_YEAR")
    fix = fix_column_refs(sql, cat)
    assert "sd.DEPARTMENT_CODE = sd.DEPARTMENT_CODE" not in fix.sql
    assert "ata.ACADEMIC_YEAR = ata.ACADEMIC_YEAR" not in fix.sql
    assert fix.changes == ["lib.ACADEMIC_YEAR -> ata.ACADEMIC_YEAR"]  # the SELECT reference is still fixed
    assert sum("needs a real join key" in u for u in fix.unresolved) == 2


def test_join_key_problems_reach_the_llm_with_schema_join_candidates():
    cat = SchemaCatalog.build("dw", [
        ("academic_terms_all", "TERM_CODE", "STRING"), ("sis_department", "DEPARTMENT_CODE", "STRING"),
        ("subject_offered", "TERM_CODE", "STRING"),  # only sd owns DEPARTMENT_CODE (as in dw_4188)
    ], {})
    sql = ("SELECT 1 FROM ACADEMIC_TERMS_ALL ata JOIN SIS_DEPARTMENT sd ON ata.DEPARTMENT_CODE = sd.DEPARTMENT_CODE "
           "JOIN SUBJECT_OFFERED s ON ata.TERM_CODE = s.TERM_CODE")
    err = ("[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column ... with name `ata`.`DEPARTMENT_CODE` cannot be resolved. "
           "Did you mean one of the following? [`sd`.`DEPARTMENT_CODE`].")
    obs = observe({"case_id": "dw:1", "attempt_id": 1, "question": "q", "retrieved_tables":
                   ["academic_terms_all", "sis_department", "subject_offered"], "generated_sql": sql,
                   "parse_status": "OK", "execution_status": "ERROR", "execution_error": err}, "dw")
    chat = Chat()
    r = SchemaSearch().repair(obs, D.diagnose_by_rules(obs, cat), RepairContext(cat, chat))
    assert r.used_llm and "needs a real join key" in chat.prompts[0]
    assert "academic_terms_all.TERM_CODE = subject_offered.TERM_CODE" in chat.prompts[0]

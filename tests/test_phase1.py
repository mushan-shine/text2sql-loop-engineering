import json
from decimal import Decimal

import pytest

from agent.generator import FewShotExample, FewShotGenerator, build_prompt, extract_sql, select_few_shot
from agent.llm import CachingChatClient, LlmError, LlmResponse, UsageMeter, LlmBudgetExceeded, check_api_key
from agent.retriever import BM25TableRetriever, SchemaCatalog, tokenize
from benchmark.beaver.evaluator import evaluate_against_gold, serialize_rows
from evaluation.baseline import BaselineConfig, gold_judge, run_case, select_pilot, summarize
from execution.base import ExecutionResult


# ----------------------------------------------------------------------------- fixtures

@pytest.fixture
def catalog() -> SchemaCatalog:
    cols = [("sis_department", "DEPARTMENT_NAME", "STRING"), ("sis_department", "DEPARTMENT_CODE", "STRING"),
            ("space_unit", "SPACE_UNIT", "STRING"), ("space_unit", "DLC_KEY", "STRING"),
            ("fclt_building", "BUILDING_NAME", "STRING"), ("fclt_building", "BUILDING_HEIGHT", "FLOAT")]
    meta = {"SIS_DEPARTMENT": {"db": "dw", "column_names": ["DEPARTMENT_NAME", "DEPARTMENT_CODE"],
                               "example_columns": [["Mathematics", "Physics"], ["18", "8"]]}}
    return SchemaCatalog.build("dw", cols, meta)


class FakeChat:
    model = "fake"

    def __init__(self, text: str):
        self.text, self.prompts = text, []
        self.meter = UsageMeter()

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return LlmResponse(self.text, "fake", 100, 20, 5)


class RecordingExecutor:
    def __init__(self, result: ExecutionResult):
        self.result, self.calls = result, []

    def execute(self, sql, db, max_rows=None):
        self.calls.append((sql, db, max_rows))
        return self.result


# ----------------------------------------------------------------------------- llm

def test_api_key_control_characters_rejected():
    with pytest.raises(LlmError, match="control character"):
        check_api_key("abc\x16def")
    assert check_api_key(" key ") == "key"


def test_budget_enforced():
    m = UsageMeter(max_calls=1)
    m.record(LlmResponse("x", "m", 1, 1, 1))
    with pytest.raises(LlmBudgetExceeded):
        m.check()


def test_cache_replays_identical_prompt(tmp_path):
    class Inner:
        model, params, meter = "m", {"do_sample": False}, UsageMeter()
        n = 0

        def complete(self, prompt, system=None):
            Inner.n += 1
            r = LlmResponse(f"answer {Inner.n}", "m", 3, 4, 5)
            self.meter.record(r)
            return r

    c = CachingChatClient(Inner(), tmp_path / "cache.jsonl")  # type: ignore[arg-type]
    a, b = c.complete("q", "s"), c.complete("q", "s")
    assert a.text == b.text == "answer 1" and b.cached and Inner.n == 1
    c2 = CachingChatClient(Inner(), tmp_path / "cache.jsonl")  # type: ignore[arg-type]
    assert c2.complete("q", "s").text == "answer 1"  # persisted across instances
    assert c.complete("q", "other system").text == "answer 2"


# ----------------------------------------------------------------------------- retrieval

def test_tokenize():
    assert tokenize("DEPARTMENT_NAMES of the Buildings") == ["department", "name", "building"]


def test_retrieval_ranks_by_names_and_values(catalog):
    r = BM25TableRetriever(catalog)
    assert r.retrieve("Which departments are in Mathematics?", 1).tables == ("sis_department",)
    assert r.retrieve("height of each building", 1).tables == ("fclt_building",)
    assert len(r.retrieve("anything", 2).tables) == 2


def test_catalog_json_roundtrip(catalog):
    again = SchemaCatalog.from_json(catalog.to_json())
    assert again.tables["sis_department"].columns[0].examples == ("Mathematics", "Physics")


# ----------------------------------------------------------------------------- generation

@pytest.mark.parametrize("text,sql,status", [
    ("```sql\nSELECT 1;\n```", "SELECT 1", "OK"),
    ("Here you go:\n```\nWITH x AS (SELECT 1) SELECT * FROM x\n```", "WITH x AS (SELECT 1) SELECT * FROM x", "OK"),
    ("SELECT a FROM t", "SELECT a FROM t", "OK"),
    ("```sql\nI cannot answer this question.\n```", "", "NO_SQL"),
    ("", "", "EMPTY_RESPONSE"),
])
def test_extract_sql(text, sql, status):
    assert extract_sql(text) == (sql, status)


def test_few_shot_excludes_evaluation_cases_and_is_deterministic():
    qs = [{"id": f"dw_{i}", "question": f"q{i}", "sql": f"SELECT VARIANCE(x) FROM t{i}", "tables": ["t"]}
          for i in range(20)]
    ex = select_few_shot(qs, exclude_ids={f"dw_{i}" for i in range(10)}, n=3, seed=1)
    assert len(ex) == 3 and all(int(e.question[1:]) >= 10 for e in ex)
    assert all("VAR_POP" in e.sql for e in ex)  # phase-0 adapter rules applied
    assert ex == select_few_shot(qs, exclude_ids={f"dw_{i}" for i in range(10)}, n=3, seed=1)


def test_prompt_contains_only_agent_visible_information(catalog, make_case):
    case = make_case("SELECT secret_gold FROM FCLT_BUILDING")
    prompt = build_prompt(case.agent_view(), "TABLE dw.x\n  a STRING", [FewShotExample("q", "SELECT 1")])
    assert case.question in prompt and "secret_gold" not in prompt and "FCLT_BUILDING" not in prompt


# ----------------------------------------------------------------------------- evaluation

def test_evaluate_against_frozen_gold():
    gold = serialize_rows([("Math", Decimal("1.50")), ("Physics", Decimal("2"))])
    assert evaluate_against_gold([("Physics", 2.0), ("Math", 1.5)], gold)[0]
    assert not evaluate_against_gold([("Math", 1.5)], gold)[0]
    assert not evaluate_against_gold([(1.5, "Math"), (2.0, "Physics")], gold)[0]  # column order matters
    assert evaluate_against_gold([], serialize_rows([]))[0]


def test_run_case_gives_agent_only_the_task(catalog, make_case):
    case = make_case("SELECT DEPARTMENT_NAME FROM SIS_DEPARTMENT", case_id="7")
    chat = FakeChat("```sql\nSELECT DEPARTMENT_NAME FROM dw.sis_department\n```")
    ex = RecordingExecutor(ExecutionResult("databricks", "SUCCESS", [("Mathematics",)], ["DEPARTMENT_NAME"]))
    rec = run_case(case, gold_judge(serialize_rows([("Mathematics",)])), BM25TableRetriever(catalog),
                   FewShotGenerator(chat, catalog, []), ex, BaselineConfig(top_k=2, max_result_rows=10))
    assert rec["correct"] and rec["attempt_id"] == 1 and ex.calls[0][2] == 10
    assert case.gold_sql not in chat.prompts[0]
    assert rec["eval_table_recall"] == 1.0  # "How many buildings?" retrieves the gold table FCLT_BUILDING


def test_no_sql_is_a_failed_attempt(catalog, make_case):
    rec = run_case(make_case(), gold_judge(serialize_rows([(1,)])), BM25TableRetriever(catalog),
                   FewShotGenerator(FakeChat("I cannot help."), catalog, []),
                   RecordingExecutor(ExecutionResult("databricks", "SUCCESS", [(1,)])), BaselineConfig())
    assert not rec["correct"] and rec["execution_status"] == "NO_SQL"


def test_pilot_selection_is_fixed():
    ids = [f"dw:dw_{i}" for i in range(89)]
    assert select_pilot(ids, 20, 5) == select_pilot(list(reversed(ids)), 20, 5)
    assert len(select_pilot(ids, 20, 5)) == 20


def test_summary_metrics():
    base = {"parse_status": "OK", "total_tokens": 10, "llm_latency_ms": 5, "latency_ms": 7, "finish_reason": "stop",
            "category": "c", "eval_table_recall": 1.0}
    recs = [{**base, "correct": True, "execution_status": "SUCCESS", "eval_all_gold_tables_retrieved": True},
            {**base, "correct": False, "execution_status": "ERROR", "eval_all_gold_tables_retrieved": False}]
    s = summarize(recs)
    assert s["first_pass_accuracy"] == 0.5 and s["executable_rate"] == 0.5
    assert s["accuracy_when_all_gold_tables_retrieved"] == "1/1"


# ----------------------------------------------------------------------------- dev set

def test_devset_value_roundtrip_keeps_decimal_scale():
    import datetime as dt
    from evaluation.devset import decode_rows, encode_rows
    rows = [(Decimal("6.8333"), dt.date(2020, 1, 2), dt.datetime(2020, 1, 2, 3, 4), None, "x", 3, 1.5)]
    again = decode_rows(json.loads(json.dumps(encode_rows(rows))))
    assert again == rows and again[0][0].as_tuple().exponent == -4


def test_devset_build_excludes_eval_and_few_shot_and_qualifies():
    from evaluation.devset import build_devset, dev_cases, dev_judges
    from tests.conftest import FakeExecutor, ok
    qs = [{"id": f"dw_{i}", "question": f"q{i}", "db": "dw", "sql": f"SELECT {i}", "tables": ["t"]} for i in range(8)]
    mysql = FakeExecutor("mysql", {f"SELECT {i}": ok("mysql", [(Decimal(f"{i}.5000"),)]) for i in range(8)})
    # dw_3 does not reproduce on databricks -> rejected
    dbx_script = {f"SELECT {i}": ok("databricks", [(i + 0.5,)]) for i in range(8)}
    dbx_script["SELECT 3"] = ok("databricks", [(99.0,)])

    class Dbx(FakeExecutor):
        def execute(self, sql, db, max_rows=None):
            return super().execute(sql, db)

    d = build_devset(qs, {"dw_0", "dw_1"}, n=10, seed=1, mysql=mysql, dbx=Dbx("databricks", dbx_script))
    ids = {c["id"] for c in d["cases"]}
    assert ids == {"dw_2", "dw_4", "dw_5", "dw_6", "dw_7"} and "dw_3" in d["rejected"]
    judges = dev_judges(json.loads(json.dumps(d)))
    assert judges["dw:dw_2"]([(2.5,)])[0] and not judges["dw:dw_2"]([(2.6,)])[0]
    assert {c.case_id for c in dev_cases(d)} == {f"dw:{i}" for i in ids}


def test_prompt_v2_maps_mysql_statistics():
    from agent.generator import PROMPT_VERSION, RULES
    assert PROMPT_VERSION == "baseline-v2"
    assert "never STDDEV_POP" in RULES and "STDDEV_POP(...)" in RULES and "SAMPLE" in RULES

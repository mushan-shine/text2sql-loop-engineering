"""Outer loop (evaluation/outer_loop.py) and knowledge items (agent/curated.py): mining, gate, prompt use."""
import json

from agent.curated import CuratedKnowledge, attach_curated
from agent.generator import FewShotGenerator
from agent.llm import LlmResponse
from agent.retriever import Retrieval, SchemaCatalog
from benchmark.beaver.dataset import AgentTask, BeaverCase
from benchmark.beaver.subtasks import FailureLabel
from evaluation import outer_loop as ol
from execution.base import ExecutionResult

CAT = SchemaCatalog.build("dw", [
    ("dept", "CODE", "STRING"), ("dept", "NAME", "STRING"),
    ("course_desc", "DEPT", "STRING"), ("course_desc", "LEVEL", "STRING"), ("course_desc", "TITLE", "STRING"),
    ("course_desc_hist", "DEPT", "STRING"), ("course_desc_hist", "LEVEL", "STRING"),
    ("course_desc_hist", "TITLE", "STRING"),
], {})

PREF = {"item_id": "kn-1", "kind": "table_preference",
        "content": {"prefer": "course_desc", "instead_of": "course_desc_hist", "keywords": ["graduate"],
                    "text": "For questions about graduate, this warehouse uses table course_desc."}}
JOIN = {"item_id": "kn-2", "kind": "join_rule",
        "content": {"tables": ["course_desc", "dept"], "condition": "course_desc.DEPT = dept.CODE", "join": "LEFT",
                    "text": "Join course_desc and dept on course_desc.DEPT = dept.CODE (LEFT JOIN)."}}
class Chat:
    model = "m"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return LlmResponse("```sql\nSELECT 1\n```", "m", 10, 2, 1)


# ---------------------------------------------------------------- reviewed knowledge in the prompt

def test_curated_notes_match_shown_tables_and_keywords():
    cur = CuratedKnowledge([PREF, JOIN])
    notes = cur.notes_for("list graduate courses of each department", ["dept", "course_desc"])
    assert "Reviewed usage notes" in notes and "uses table course_desc" in notes and "LEFT JOIN" in notes
    assert "uses table" not in cur.notes_for("list undergrad courses", ["dept", "course_desc"])   # keyword gate
    assert cur.notes_for("graduate courses", ["dept"]) == ""                                   # tables not shown
    assert len(cur.notes) == 2 and not CuratedKnowledge([])


def test_attach_curated_puts_notes_into_the_prompt():
    chat = Chat()
    gen = attach_curated(FewShotGenerator(chat, CAT, []), CuratedKnowledge([PREF, JOIN]))
    assert gen.prompt_version.endswith("+cur")
    gen.generate(AgentTask("u:1", "graduate courses per department", "dw"), ("dept", "course_desc"))
    assert "Reviewed usage notes" in chat.prompts[0]
    assert attach_curated(gen, CuratedKnowledge([])) is gen    # nothing to attach -> unchanged


# ---------------------------------------------------------------- mining

def result(cid, question, gen_tables, gold_tables, missing_joins=()):
    case = BeaverCase(cid, "dw", question, "dw", "")
    label = FailureLabel(cid, "JOIN_KEY_FAILURE", [], {"missing_join_keys": list(missing_joins)})
    return ol.CaseResult(case, "SELECT 1", "SUCCESS", False, [], 10, label, set(gen_tables), set(gold_tables))


def test_table_preferences_need_support_and_look_alike_tables():
    fails = [result("dw:1", "graduate courses", {"course_desc_hist"}, {"course_desc"}),
             result("dw:2", "graduate course titles", {"course_desc_hist", "dept"}, {"course_desc", "dept"}),
             result("dw:3", "department names", {"course_desc"}, {"dept"})]        # not look-alike tables
    got = ol.mine_table_preferences(fails, None, CAT, min_support=2)
    assert len(got) == 1
    c = got[0]
    assert c["content"]["prefer"] == "course_desc" and c["content"]["instead_of"] == "course_desc_hist"
    assert c["evidence"]["support"] == 2 and c["evidence"]["case_ids"] == ["dw:1", "dw:2"]
    assert ol.mine_table_preferences(fails, None, CAT, min_support=3) == []


class KB:  # the parts of WarehouseKnowledge the miners read
    n_queries = 100
    table_freq = {"course_desc": 20, "dept": 50}
    word_df = {"graduate": 10, "level": 40, "five": 8, "g2021": 3}
    word_tables = {"graduate": {"course_desc": 9}, "level": {"course_desc": 5, "dept": 30},
                   "five": {"course_desc": 8}, "g2021": {"course_desc": 3}}
    groups: list = []
    joins = {"course_desc|dept": {"keys": {"course_desc.DEPT = dept.CODE": 7}, "kinds": {"LEFT": 7}}}


def test_missing_tables_become_table_hints_that_extend_the_schema():
    fails = [result("dw:1", "graduate level courses", {"dept"}, {"dept", "course_desc"}),
             result("dw:2", "graduate course count", {"dept"}, {"dept", "course_desc"}),
             result("dw:3", "names", set(), {"course_desc"})]                    # nothing parsed: no evidence
    got = ol.mine_missing_tables(fails, KB, min_support=2)
    assert len(got) == 1
    c = got[0]["content"]
    assert c["table"] == "course_desc" and c["keywords"] == ["graduate"] and c["with"] == ["dept"]
    assert c["condition"] == "course_desc.DEPT = dept.CODE" and got[0]["evidence"]["case_ids"] == ["dw:1", "dw:2"]
    cur = ol.as_curated(got)
    assert cur.extra_tables("list graduate students by department", ["dept"]) == ["course_desc"]
    assert cur.extra_tables("list departments", ["dept"]) == []                  # keyword gate
    assert "also need table course_desc" in cur.notes_for("graduate students", ["dept", "course_desc"])
    chat = Chat()
    g = attach_curated(FewShotGenerator(chat, CAT, []), cur).generate(AgentTask("u:1", "graduate totals", "dw"), ("dept",))
    assert g.schema_tables == ("dept", "course_desc") and "TABLE dw.course_desc" in chat.prompts[0]


def test_join_rules_from_missing_join_keys():
    cond = "course_desc.dept = dept.code"
    fails = [result(f"dw:{i}", "q", {"dept", "course_desc"}, {"dept", "course_desc"}, [cond]) for i in range(2)]
    got = ol.mine_join_rules(fails, None, min_support=2)
    assert len(got) == 1 and got[0]["content"]["condition"] == "course_desc.DEPT = dept.CODE"
    assert got[0]["content"]["tables"] == ["course_desc", "dept"] and got[0]["content"]["join"] == "INNER"


# ---------------------------------------------------------------- regression gate + proposals

def runs(correct, tokens=10, prompts=None):
    return [ol.CaseResult(BeaverCase(f"dw:{i}", "dw", "q", "dw", ""), "S", "SUCCESS", ok, [], tokens,
                          prompt_hash=(prompts or {}).get(i, "p"))
            for i, ok in enumerate(correct)]


def test_keywords_need_shared_content_words_that_point_at_the_table():
    qs = ["graduate level courses with five units in g2021", "graduate level courses with five sections g2021"]
    # "five" (number word), "g2021" (digits) and "level" (points at dept, not course_desc) are dropped;
    # a word from only one supporting question would be dropped too
    assert ol._keywords(qs, "course_desc", KB) == ["graduate"]
    assert ol._keywords(["graduate courses"], "course_desc", KB) == ["graduate"]     # single question: n >= 1
    assert ol._keywords(qs, "dept", KB) == []                                        # nothing points at dept
    assert ol._keywords(qs, "course_desc", None) == []


def test_per_item_gate_isolates_a_harmful_item_and_checks_the_good_ones_together():
    before = runs([False, True, False])
    good1, good2, bad, idle = ({"kind": "join_rule", "title": t, "content": {"tables": [t], "text": t},
                                "evidence": {}} for t in ("g1", "g2", "bad", "idle"))
    outcome = {"g1": ([True, True, False], {0: "x"}), "g2": ([False, True, True], {2: "y"}),
               "bad": ([True, False, False], {0: "x", 1: "z"}), "idle": ([False, True, False], {})}
    calls = []

    def run(cur):
        titles = [i["content"]["text"] for i in cur.items]
        calls.append(titles)
        if len(titles) > 1:
            return runs([True, True, True], prompts={0: "x", 2: "y"})
        correct, prompts = outcome[titles[0]]
        return runs(correct, prompts=prompts)

    per_item, combined = ol.gate_candidates([good1, bad, good2, idle], before, run)
    assert [r["gate_passed"] for r in per_item] == [True, False, True, True]
    assert per_item[1]["harmed"] == ["dw:1"] and per_item[3]["affected"] == []
    assert calls[-1] == ["g1", "g2"] and combined["correct_after"] == 3 and combined["items"] == 2
    recs = [ol.recommendation(r) for r in per_item]
    assert recs[0].startswith("建议批准") and recs[1].startswith("建议驳回") and "无法验证" in recs[3]
    rows = ol.proposal_rows("b", [good1, bad, good2, idle], per_item, {"combined": {**combined, "gate_passed": False}})
    assert "合用时未通过" in rows[0]["recommendation"] and rows[1]["recommendation"].startswith("建议驳回")
    assert json.loads(rows[0]["regression_json"])["batch"]["combined"]["items"] == 2


def test_run_and_judge_reuses_unchanged_cases():
    class Ret:
        def retrieve(self, q, k):
            return Retrieval(("dept",), (1.0,))

    class Exec:
        calls = 0

        def execute(self, sql, db, max_rows=None):
            Exec.calls += 1
            return ExecutionResult("dbx", "SUCCESS", rows=[(1,)])

    cases = [BeaverCase("dw:a", "dw", "graduate courses", "dw", ""), BeaverCase("dw:b", "dw", "dept names", "dw", "")]
    judges = {c.case_id: (lambda rows: (True, "")) for c in cases}
    first = ol.run_and_judge(cases, judges, Ret(), FewShotGenerator(Chat(), CAT, []), Exec(), {}, 5, 10)
    assert Exec.calls == 2 and all(r.prompt_hash for r in first)
    hint = {"item_id": "kn-h", "kind": "table_hint",
            "content": {"table": "course_desc", "keywords": ["graduate"], "with": [], "text": "Use course_desc."}}
    hinted = attach_curated(FewShotGenerator(Chat(), CAT, []), CuratedKnowledge([hint]))   # fires on "graduate" only
    again = ol.run_and_judge(cases, judges, Ret(), hinted, Exec(), {}, 5, 10, reuse={r.case.case_id: r for r in first})
    assert Exec.calls == 3                                  # only the case whose prompt changed is executed again
    assert again[1] is first[1] and again[0].prompt_hash != first[0].prompt_hash
    assert ol.compare(first, again)["affected"] == ["dw:a"]



def test_gate_requires_no_harm_and_bounded_tokens():
    better = ol.compare(runs([False, True, False]), runs([True, True, False], 11))
    assert better["gate_passed"] and better["fixed"] == ["dw:0"] and "批准" in ol.recommendation(better)
    harmed = ol.compare(runs([False, True]), runs([True, False]))            # same total, but one harmed
    assert not harmed["gate_passed"] and harmed["harmed"] == ["dw:1"] and "驳回" in ol.recommendation(harmed)
    costly = ol.compare(runs([False, True]), runs([True, True], 13))         # +30% tokens
    assert not costly["gate_passed"]
    same = ol.compare(runs([True]), runs([True]))
    assert same["gate_passed"] and "中性" in ol.recommendation(same)


def test_proposal_rows_carry_json_and_recommendation():
    reg = ol.compare(runs([False]), runs([True]))
    rows = ol.proposal_rows("batch-x", [{"kind": "join_rule", "title": "t", "content": JOIN["content"],
                                          "evidence": {"support": 2}}], [reg])
    assert rows[0]["proposal_id"] == "batch-x-00" and json.loads(rows[0]["content_json"]) == JOIN["content"]
    assert json.loads(rows[0]["regression_json"])["gate_passed"] and rows[0]["recommendation"]
    assert ol.as_curated([{"kind": "join_rule", "content": JOIN["content"]}]).notes[0]["item_id"] == "cand-0"


def pref(prefer, instead_of, support):
    return {"kind": "table_preference", "title": f"{prefer}>{instead_of}",
            "content": {"prefer": prefer, "instead_of": instead_of, "keywords": [], "text": ""},
            "evidence": {"support": support}}


def test_opposite_table_preferences_are_resolved_by_support():
    join = {"kind": "join_rule", "title": "j", "content": {}, "evidence": {"support": 2}}
    close = [pref("a", "b", 3), pref("b", "a", 3), join]                 # same support: nobody knows -> drop both
    kept, dropped = ol.resolve_conflicts(close)
    assert kept == [join] and len(dropped) == 2
    clear = [pref("a", "b", 6), pref("b", "a", 3), pref("c", "d", 2)]    # 2x the support: keep the stronger one
    kept, dropped = ol.resolve_conflicts(clear)
    assert [c["title"] for c in kept] == ["a>b", "c>d"] and [c["title"] for c in dropped] == ["b>a"]


def test_dev_check_failure_downgrades_recommended_items():
    good = {**ol.compare(runs([False]), runs([True], prompts={0: "x"})), "gate_set": "val"}
    rows = ol.proposal_rows("b", [{"kind": "join_rule", "title": "t", "content": {}, "evidence": {}}], [good],
                            {"dev_check": {"gate_passed": False}})
    assert "开发集复核" in rows[0]["recommendation"]
    rows = ol.proposal_rows("b", [{"kind": "join_rule", "title": "t", "content": {}, "evidence": {}}], [good],
                            {"dev_check": {"gate_passed": True}})
    assert rows[0]["recommendation"].startswith("建议批准")


def test_judged_cases_drop_questions_whose_gold_does_not_run():
    class Exec:
        def execute(self, sql, db, max_rows=None):
            ok = "broken" not in sql
            return ExecutionResult("dbx", "SUCCESS" if ok else "ERROR", rows=[(1,)] if ok else [])

    qs = [{"id": 1, "question": "q1", "db": "dw", "sql": "SELECT 1"},
          {"id": 2, "question": "q2", "db": "dw", "sql": "SELECT broken"}]
    cases, judges = ol.judged_cases(qs, Exec(), "dw", 10)
    assert [c.case_id for c in cases] == ["dw:1"] and judges["dw:1"]([(1,)])[0] and not judges["dw:1"]([(2,)])[0]

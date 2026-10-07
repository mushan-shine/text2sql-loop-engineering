"""Phase 6: verifiers, loop controller (targeted / generic), loop metrics."""
from agent.generator import FewShotGenerator
from agent.llm import LlmResponse
from agent.retriever import BM25TableRetriever, SchemaCatalog
from benchmark.beaver.dataset import AgentTask
from evaluation.loop_run import summarize
from execution.base import ExecutionResult
from loop_engineer.controller import GENERIC_INSTRUCTION, LoopConfig, LoopController
from loop_engineer.diagnose import Diagnoser
from loop_engineer.policy import Policy
from loop_engineer.verifier import OracleVerifier, SelfVerifier
from skills.base import RepairContext

CAT = SchemaCatalog.build("dw", [
    ("sis_department", "DEPARTMENT_CODE", "STRING"), ("sis_department", "DEPARTMENT_NAME", "STRING"),
    ("subject_offered", "DEPARTMENT_CODE", "STRING"), ("subject_offered", "NUM_ENROLLED", "INT"),
], {})
BAD = ("SELECT d.DEPARTMENT_NAME, AVG(d.NUM_ENROLLED) FROM SIS_DEPARTMENT d "
       "JOIN SUBJECT_OFFERED s ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE GROUP BY d.DEPARTMENT_NAME")
ERR = ("[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column ... with name `d`.`NUM_ENROLLED` cannot be resolved. "
       "Did you mean one of the following? [`s`.`NUM_ENROLLED`].")


class Chat:
    model = "m"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return LlmResponse(self.reply(len(self.prompts)), "m", 100, 10, 5)


class Exec:
    """Errors on any SQL that still references d.NUM_ENROLLED, succeeds otherwise."""

    def __init__(self, rows=((("Math", 3.0),))):
        self.rows, self.calls = list(rows), []

    def execute(self, sql, db, max_rows=None):
        self.calls.append(sql)
        if "d.NUM_ENROLLED" in sql:
            return ExecutionResult("databricks", "ERROR", error=ERR, error_class="UNRESOLVED_COLUMN")
        return ExecutionResult("databricks", "SUCCESS", self.rows, ["a", "b"])


def controller(chat, strategy="targeted", ex=None):
    return LoopController(BM25TableRetriever(CAT), FewShotGenerator(chat, CAT, []), ex or Exec(), Diagnoser(CAT),
                          Policy(), RepairContext(CAT, chat), LoopConfig(strategy=strategy, top_k=2))


TASK = AgentTask("dw:1", "average enrollment per department", "dw")


def test_self_verifier_signals():
    v = SelfVerifier()
    assert v.verify({"execution_status": "ERROR"}, []).signals == ("execution_error",)
    assert v.verify({"execution_status": "NO_SQL"}, []).signals == ("no_sql",)
    assert v.verify({"execution_status": "TOO_MANY_ROWS"}, []).signals == ("too_many_rows",)
    assert v.verify({"execution_status": "SUCCESS"}, []).signals == ("empty_result",)
    assert v.verify({"execution_status": "SUCCESS"}, [("a", None), ("b", None)]).signals == ("all_null_column",)
    assert v.verify({"execution_status": "SUCCESS"}, [("a", 1)]).passed
    assert SelfVerifier(signals=("execution_error",)).verify({"execution_status": "SUCCESS"}, []).passed


def test_oracle_verifier_is_gold_based():
    o = OracleVerifier(lambda rows: (rows == [("x",)], ""))
    assert o.verify({"execution_status": "SUCCESS"}, [("x",)]).passed
    assert not o.verify({"execution_status": "SUCCESS"}, [("y",)]).passed and o.mode == "oracle"


def test_targeted_loop_repairs_deterministically():
    chat = Chat(lambda n: f"```sql\n{BAD}\n```")
    res = controller(chat).run(TASK, SelfVerifier())
    a1, a2 = res.attempts
    assert a1["verifier_decision"] == "FAIL" and a1["failure_type"] == "COLUMN_MAPPING_FAILURE"
    assert a1["repair_skill"] == "SchemaSearch" and a2["used_llm"] is False
    assert a2["execution_status"] == "SUCCESS" and a2["verifier_decision"] == "PASS"
    assert res.final_index == 1 and a2["final_status"] == "FINAL" and len(chat.prompts) == 1


def test_generic_retry_uses_same_budget_without_diagnosis():
    chat = Chat(lambda n: f"```sql\n{BAD}\n```" if n == 1 else "```sql\nSELECT s.NUM_ENROLLED FROM SUBJECT_OFFERED s\n```")
    res = controller(chat, "generic").run(TASK, SelfVerifier())
    assert len(res.attempts) == 2 and res.attempts[1]["repair_skill"] == "GenericRetry"
    assert GENERIC_INSTRUCTION in chat.prompts[1] and "(not diagnosed)" in chat.prompts[1]
    assert "failure_type" not in res.attempts[0]


def test_pass_on_first_attempt_stops_the_loop():
    ex = Exec()
    res = controller(Chat(lambda n: "```sql\nSELECT 1\n```"), ex=ex).run(TASK, SelfVerifier())
    assert len(res.attempts) == 1 and len(ex.calls) == 1


def test_final_answer_keeps_an_executable_first_attempt_when_repair_breaks_it():
    chat = Chat(lambda n: "```sql\nSELECT 1\n```" if n == 1 else f"```sql\n{BAD}\n```")
    ex = Exec(rows=[])  # attempt 1 runs but returns nothing -> empty_result -> retry, which errors
    res = controller(chat, "generic", ex).run(TASK, SelfVerifier())
    assert [a["execution_status"] for a in res.attempts] == ["SUCCESS", "ERROR"]
    assert res.final_index == 0  # last executed attempt, not the broken repair


def test_loop_metrics():
    def rec(first, final, trig, att=2):
        return {"first_correct": first, "final_correct": final, "attempt_correct": [first, final][:att],
                "n_attempts": att, "final_index": att - 1, "tokens": 10, "extra_tokens": 4, "latency_ms": 5,
                "attempts": [{"verifier_decision": "FAIL" if trig else "PASS", "execution_status": "SUCCESS"},
                             {"repair_skill": "SchemaSearch", "execution_status": "SUCCESS"}][:att]}
    s = summarize([rec(False, True, True), rec(False, False, True), rec(True, False, True), rec(True, True, False, 1),
                   rec(False, False, False, 1)])
    assert s["recovered"] == 1 and s["recovery_rate"] == round(1 / 3, 4)
    assert s["harmed"] == 1 and s["harm_rate"] == 0.5 and s["net_gain"] == 0
    assert s["verifier_confusion"] == {"triggered_and_wrong(hit)": 2, "triggered_but_right(false_alarm)": 1,
                                       "passed_but_wrong(miss)": 1, "passed_and_right": 1}
    assert s["per_skill"]["SchemaSearch"]["recovered"] == 1


def test_loop_records_flatten_to_one_trace_row_per_attempt():
    from dbx import tables
    from dbx.publish import build_rows, flatten_loop_records, mlflow_metrics, mlflow_params
    import pyarrow as pa
    rec = {"category": "c", "final_index": 1, "attempt_correct": [False, True], "attempts": [
        {"case_id": "dw:1", "attempt_id": 1, "question": "q", "retrieved_tables": ["a"], "generated_sql": "S1",
         "parse_status": "OK", "execution_status": "ERROR", "execution_error": "e", "verifier_mode": "self",
         "verifier_decision": "FAIL", "verifier_signals": ["execution_error"], "failure_type": "COLUMN_MAPPING_FAILURE",
         "diagnosis_confidence": 0.9, "repair_skill": "SchemaSearch", "repaired_sql": "S2", "final_status": "SUPERSEDED",
         "input_tokens": 10, "output_tokens": 2, "llm_latency_ms": 3, "exec_latency_ms": 1},
        {"case_id": "dw:1", "attempt_id": 2, "question": "q", "retrieved_tables": ["a"], "generated_sql": "S2",
         "parse_status": "OK", "execution_status": "SUCCESS", "result_row_count": 1, "result_preview": '[["x"]]',
         "verifier_mode": "self", "verifier_decision": "PASS", "verifier_signals": [], "final_status": "FINAL",
         "used_llm": False, "input_tokens": 0, "output_tokens": 0, "llm_latency_ms": 0, "exec_latency_ms": 2}]}
    flat = flatten_loop_records([rec])
    meta = {"run_id": "targeted-self-x", "mode": "loop", "split": "dev", "arm": "targeted-self", "model": "m"}
    summary = {"cases": 1, "final_correct": 1, "first_pass_accuracy": 0.0, "final_accuracy": 1.0, "executable_final": 1,
               "recovery_rate": 1.0, "net_gain": 1}
    rows = build_rows(meta, summary, flat, "targeted_loop")
    pa.Table.from_pylist(rows["execution_traces"], schema=tables.EXECUTION_TRACES)
    pa.Table.from_pylist(rows["evaluation_results"], schema=tables.EVALUATION_RESULTS)
    pa.Table.from_pylist(rows["runs"], schema=tables.RUNS)
    t1, t2 = rows["execution_traces"]
    assert t1["failure_type"] == "COLUMN_MAPPING_FAILURE" and t1["repair_skill"] == "SchemaSearch"
    assert t2["verifier_decision"] == "PASS" and t2["final_status"] == "FINAL" and "correct" not in t2
    assert [e["correct"] for e in rows["evaluation_results"]] == [False, True]
    assert rows["runs"][0]["correct"] == 1 and rows["runs"][0]["executable_rate"] == 1.0
    assert mlflow_params(meta, summary)["arm"] == "targeted-self" and mlflow_metrics(summary)["net_gain"] == 1.0


def test_events_follow_the_loop_steps_and_do_not_change_results():
    events = []
    chat = Chat(lambda n: f"```sql\n{BAD}\n```")
    res = controller(chat).run(TASK, SelfVerifier(), on_event=lambda step, p: events.append((step, p)))
    assert [s for s, _ in events] == ["retrieve", "generate", "execute", "verify", "observe", "diagnose", "route",
                                      "repair", "execute", "verify", "final"]
    d = dict(events)
    assert d["observe"]["unresolved_column"] == "NUM_ENROLLED" and d["observe"]["suggestions"] == ["s.NUM_ENROLLED"]
    assert "correct" not in d["observe"]["fields"]  # the observer whitelist, shown in the UI
    assert d["diagnose"]["failure_type"] == "COLUMN_MAPPING_FAILURE" and d["route"]["skill"] == "SchemaSearch"
    assert d["repair"]["before_sql"] == BAD and d["final"]["final_attempt"] == 2
    assert d["repair"]["details"]["deterministic_changes"] == ["d.NUM_ENROLLED -> s.NUM_ENROLLED"]
    assert "Question:" in d["generate"]["prompt"]
    plain = controller(Chat(lambda n: f"```sql\n{BAD}\n```")).run(TASK, SelfVerifier())
    assert [a["generated_sql"] for a in plain.attempts] == [a["generated_sql"] for a in res.attempts]


def test_a_failing_event_callback_never_breaks_the_loop():
    def boom(step, payload):
        raise RuntimeError("ui crashed")
    res = controller(Chat(lambda n: f"```sql\n{BAD}\n```")).run(TASK, SelfVerifier(), on_event=boom)
    assert res.attempts[-1]["verifier_decision"] == "PASS"


def test_generic_retry_event_carries_its_prompt_and_instruction():
    events = []
    chat = Chat(lambda n: f"```sql\n{BAD}\n```" if n == 1 else "```sql\nSELECT s.NUM_ENROLLED FROM SUBJECT_OFFERED s\n```")
    controller(chat, "generic").run(TASK, SelfVerifier(), on_event=lambda s, p: events.append((s, p)))
    rep = dict(events)["repair"]
    assert "observe" not in dict(events)  # the generic arm does not observe / diagnose
    assert rep["details"]["instruction"] == GENERIC_INSTRUCTION and "(not diagnosed)" in rep["details"]["prompt"]

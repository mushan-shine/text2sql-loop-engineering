"""Phase 2: trace / evaluation row building and the gold boundary."""
import json

from dbx import tables
from dbx.publish import build_rows, mlflow_metrics, mlflow_params, split_of

META = {"run_id": "baseline-x", "mode": "dev", "model": "glm-4-flash", "prompt_version": "baseline-v2", "top_k": 20,
        "few_shot": [{"id": "dw_1", "question": "q", "sql": "SELECT 1"}]}
SUMMARY = {"cases": 2, "correct": 1, "first_pass_accuracy": 0.5, "executable_rate": 1.0, "tokens_total": 30,
           "mean_table_recall": 0.75, "execution_status": {"SUCCESS": 2}}
REC = {"case_id": "dw:dw_9", "attempt_id": 1, "question": "How many?", "retrieved_tables": ["a", "b"],
       "generated_sql": "SELECT 1", "parse_status": "OK", "execution_status": "SUCCESS", "execution_error": None,
       "result_row_count": 1, "result_preview": json.dumps([["1"]]), "correct": True, "eval_message": "Match",
       "category": "complex query", "eval_table_recall": 0.5, "eval_all_gold_tables_retrieved": False,
       "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "llm_latency_ms": 7, "exec_latency_ms": 3,
       "latency_ms": 10, "llm_cached": False, "model": "glm-4-flash"}


def test_rows_fit_declared_schemas():
    import pyarrow as pa
    rows = build_rows(META, SUMMARY, [REC, {**REC, "case_id": "dw:dw_8", "execution_status": "ERROR",
                                            "result_preview": None, "correct": False}], "baseline", "mlf-1")
    pa.Table.from_pylist(rows["execution_traces"], schema=tables.EXECUTION_TRACES)
    pa.Table.from_pylist(rows["evaluation_results"], schema=tables.EVALUATION_RESULTS)
    pa.Table.from_pylist(rows["runs"], schema=tables.RUNS)
    assert rows["runs"][0]["mlflow_run_id"] == "mlf-1" and rows["runs"][0]["split"] == "dev"


def test_traces_carry_no_gold_derived_fields():
    """Observer / Diagnoser read traces; correctness and table recall come from gold."""
    gold_derived = {"correct", "eval_message", "eval_table_recall", "eval_all_gold_tables_retrieved"}
    assert not gold_derived & set(tables.EXECUTION_TRACES.names)
    trace = build_rows(META, SUMMARY, [REC], "baseline")["execution_traces"][0]
    assert not gold_derived & set(trace) and json.loads(trace["execution_result"])["row_count"] == 1
    ev = build_rows(META, SUMMARY, [REC], "baseline")["evaluation_results"][0]
    assert ev["correct"] is True and ev["eval_table_recall"] == 0.5


def test_mlflow_payload():
    p = mlflow_params(META, SUMMARY)
    assert p["split"] == "dev" and p["few_shot_ids"] == "dw_1" and p["prompt_version"] == "baseline-v2"
    m = mlflow_metrics(SUMMARY)
    assert m["first_pass_accuracy"] == 0.5 and "execution_status" not in m
    assert split_of({"mode": "baseline"}) == "eval"


def test_mlflow_params_accept_a_few_shot_mode_string():
    """Loop runs record the few-shot MODE (a string), baseline runs the example list."""
    from dbx.publish import mlflow_params
    p = mlflow_params({"run_id": "r", "arm": "targeted-self", "few_shot": "dynamic", "knowledge": "on"}, {"cases": 3})
    assert p["n_few_shot"] == 0 and p["few_shot_mode"] == "dynamic" and p["knowledge"] == "on"

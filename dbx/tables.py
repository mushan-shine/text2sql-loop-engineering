"""Arrow schemas of the project Delta tables (single source of truth for columns)."""
from __future__ import annotations

import pyarrow as pa

S, I, B, T = pa.string(), pa.int64(), pa.bool_(), pa.timestamp("us", tz="UTC")

CASES = pa.schema([
    ("case_id", S), ("split", S), ("question", S), ("db", S),
    ("gold_sql", S),                      # ORIGINAL BEAVER gold SQL
    ("gold_tables", S), ("gold_column_mapping", S), ("gold_join_keys", S),
    ("domain_knowledge", S), ("query_decomposition", S),
    ("category", S), ("detailed_category", S), ("contains_domain_knowledge", B),
    ("source_sha256", S), ("imported_at", T),
])

# Non-gold schema metadata from beaver-table (column names/types/example rows).
# Agent-visible: this is the "schema" BEAVER provides in setting=0.
TABLES_META = pa.schema([
    ("db", S), ("table_name", S), ("column_names", S), ("column_types", S),
    ("example_rows", S), ("imported_at", T),
])

REPLICATION = pa.schema([
    ("run_id", S), ("db", S), ("table_name", S), ("mysql_rows", I), ("databricks_rows", I),
    ("columns", I), ("column_types", S), ("mismatched_columns", S), ("nulled_invalid_dates", I),
    ("notes", S), ("fidelity_ok", B), ("created_at", T),
])

COMPATIBILITY = pa.schema([
    ("run_id", S), ("case_id", S), ("db", S), ("gold_sql", S),
    ("compatibility_status", S), ("reason", S),
    ("reference_status", S), ("reference_error", S), ("reference_row_count", I),
    ("reference_result_hash", S), ("reference_stable", B),
    ("execution_status", S), ("execution_error", S), ("execution_error_class", S),
    ("execution_time_ms", I), ("row_count", I), ("result_hash", S), ("databricks_stable", B),
    ("set_match", B), ("multiset_match", B), ("ordered_match", B),
    ("static_hazards", S), ("environment", S), ("created_at", T),
])

ADAPTATIONS = pa.schema([
    ("run_id", S), ("case_id", S), ("original_gold_sql", S), ("adapted_sql", S),
    ("adaptation_rule", S), ("semantic_validation", S), ("detail", S),
    ("adapted_result_hash", S),  # Databricks result hash of the adapted SQL (drift reference)
    ("created_at", T),
])

# ---------------------------------------------------------------------------- phase 2+
# Boundary: traces.* holds what the agent saw and did — the Observer / Diagnoser
# read it. Anything computed from gold (correctness, table recall) lives in
# evaluation.* so that self-verified loop components cannot see it.

EXECUTION_TRACES = pa.schema([
    ("run_id", S), ("experiment_id", S), ("case_id", S), ("split", S), ("attempt_id", I),
    ("question", S),
    ("retrieved_tables", S), ("retrieved_columns", S), ("retrieved_context", S),
    ("generated_sql", S), ("parse_status", S),
    ("execution_status", S), ("execution_error", S), ("execution_result", S),  # row count + preview
    ("verifier_mode", S), ("verifier_decision", S), ("verifier_signals", S),
    ("failure_type", S), ("diagnosis_confidence", pa.float64()), ("diagnosis_reason", S),
    ("repair_skill", S), ("repair_reason", S), ("repaired_sql", S),
    ("final_status", S),
    ("model", S), ("prompt_version", S),
    ("latency_ms", I), ("llm_latency_ms", I), ("exec_latency_ms", I),
    ("input_tokens", I), ("output_tokens", I), ("total_tokens", I), ("llm_cached", B),
    ("created_at", T),
])

EVALUATION_RESULTS = pa.schema([
    ("run_id", S), ("experiment_id", S), ("case_id", S), ("split", S), ("attempt_id", I),
    ("correct", B), ("eval_message", S), ("category", S),
    ("eval_table_recall", pa.float64()), ("eval_all_gold_tables_retrieved", B),
    ("created_at", T),
])

RUNS = pa.schema([
    ("run_id", S), ("experiment_id", S), ("split", S), ("mode", S), ("model", S), ("prompt_version", S),
    ("top_k", I), ("cases", I), ("correct", I), ("first_pass_accuracy", pa.float64()),
    ("executable_rate", pa.float64()), ("tokens_total", I), ("summary_json", S), ("meta_json", S),
    ("mlflow_run_id", S), ("created_at", T),
])

GOLD_RESULTS = pa.schema([
    ("case_id", S), ("db", S),
    ("gold_sql", S),                # original text, as executed
    ("executed_sql", S),            # == gold_sql unless an admitted adaptation
    ("gold_result", S),             # canonical JSON rows
    ("result_hash", S), ("row_count", I), ("column_names", S),
    ("execution_status", S), ("execution_time", I),
    ("gold_source", S),             # databricks_original_gold_sql | databricks_adapted_sql
    ("verified_against", S),        # reference engine the result was checked against
    ("reference_result_hash", S),
    ("evaluation_eligibility", S),  # PRIMARY | SECONDARY | EXCLUDED
    ("exclusion_reason", S),
    ("qualification_run_id", S), ("created_at", T),
])

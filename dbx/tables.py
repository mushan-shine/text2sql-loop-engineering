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

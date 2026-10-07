"""Diagnoser — predicts WHY an attempt failed, without any gold information.

Input: an ``Observation`` (question, retrieved tables, generated SQL, execution
status / error / result preview) plus the agent-visible schema catalog.
Output: ``Diagnosis(failure_type, confidence, reason, source, repair_hints)``.

Two stages:
1. **Rules** for explicit signals — engine errors carry most of the evidence
   (Databricks even names the unresolved column and suggests alternatives).
   An unresolved column is located in the schema:
     * owned by a table the SQL already uses       -> COLUMN_MAPPING (wrong alias)
     * owned by a retrieved table the SQL ignores  -> TABLE_RETRIEVAL (table not used)
     * owned only by a non-retrieved table         -> TABLE_RETRIEVAL (retrieval miss)
     * owned by no table                           -> COLUMN_MAPPING (hallucinated name)
2. **LLM** for failures without an explicit signal (the SQL ran, but the
   verifier distrusts the result). Output is strict JSON.

``repair_hints`` carries the evidence forward to the repair skills (phase 5).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.llm import ChatClient
from agent.retriever import SchemaCatalog
from agent.sql_analysis import alias_map
from loop_engineer.observer import Observation

TABLE_RETRIEVAL = "TABLE_RETRIEVAL_FAILURE"
COLUMN_MAPPING = "COLUMN_MAPPING_FAILURE"
JOIN_KEY = "JOIN_KEY_FAILURE"
DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE_FAILURE"
QUERY_DECOMPOSITION = "QUERY_DECOMPOSITION_FAILURE"
EXECUTION = "EXECUTION_FAILURE"
UNKNOWN = "UNKNOWN"
FAILURE_TYPES = (TABLE_RETRIEVAL, COLUMN_MAPPING, JOIN_KEY, DOMAIN_KNOWLEDGE, QUERY_DECOMPOSITION, EXECUTION)

DIAGNOSER_VERSION = "diagnoser-v1.1"   # v1 + collation mismatch in UNION

# engine error classes that are SQL-level problems (fixable without new schema knowledge)
_SQL_LEVEL = {"PARSE_SYNTAX_ERROR", "MISSING_AGGREGATION", "MISSING_GROUP_BY", "DATATYPE_MISMATCH",
              "DISTINCT_WINDOW_FUNCTION_UNSUPPORTED", "INVALID_WHERE_CONDITION", "UNSUPPORTED_EXPR_FOR_WINDOW",
              "WRONG_NUM_ARGS", "UNRESOLVED_ROUTINE", "INVALID_USAGE_OF_STAR_OR_REGEX", "GROUP_BY_AGGREGATE",
              "AMBIGUOUS_REFERENCE", "NO_SQL"}


# semantic self-verifier signal -> (failure type, confidence); the policy then routes as usual
_VERIFIER_TYPES = {
    "join_tautology": (JOIN_KEY, 0.85),          # -> FindJoinPath
    "join_without_condition": (JOIN_KEY, 0.85),  # -> FindJoinPath
    "missing_grouping": (QUERY_DECOMPOSITION, 0.8),  # -> ReplanQuery
    "rounding": (EXECUTION, 0.8),                # -> RepairSQL (small, local change)
    # numeric inconsistencies: related statistics computed over different row sets -> re-plan the computation
    "avg_outside_min_max": (QUERY_DECOMPOSITION, 0.75), "min_greater_than_max": (QUERY_DECOMPOSITION, 0.75),
    "std_var_mismatch": (QUERY_DECOMPOSITION, 0.75), "std_exceeds_range": (QUERY_DECOMPOSITION, 0.75),
    "negative_statistic": (EXECUTION, 0.75), "count_not_integer": (EXECUTION, 0.75),
}


@dataclass(frozen=True)
class Diagnosis:
    failure_type: str
    confidence: float
    reason: str
    source: str                      # rule | llm | fallback
    repair_hints: dict[str, Any] = field(default_factory=dict)
    version: str = DIAGNOSER_VERSION

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["repair_hints"] = json.dumps(self.repair_hints, ensure_ascii=False)
        return d


def column_owners(catalog: SchemaCatalog, column: str) -> list[str]:
    c = column.lower()
    return sorted(t for t, tb in catalog.tables.items() if any(col.name.lower() == c for col in tb.columns))


def diagnose_by_rules(obs: Observation, catalog: SchemaCatalog) -> Diagnosis | None:
    cls = obs.error_class
    if obs.execution_status in ("NO_SQL", "EMPTY_RESPONSE") or cls == "NO_SQL":
        return Diagnosis(EXECUTION, 0.9, "the model produced no SQL", "rule", {"signal": "no_sql"})

    if cls == "UNRESOLVED_COLUMN" and obs.unresolved_column:
        col = obs.unresolved_column
        aliases = alias_map(obs.generated_sql)
        used = set(aliases.values())
        owner = column_owners(catalog, col)
        retrieved = set(obs.retrieved_tables)
        hints = {"signal": "unresolved_column", "column": col, "qualifier": obs.unresolved_qualifier,
                 "qualifier_table": aliases.get((obs.unresolved_qualifier or "").lower()),
                 "owner_tables": owner, "suggestions": list(obs.suggestions)}
        in_used = [t for t in owner if t in used]
        in_retrieved = [t for t in owner if t in retrieved and t not in used]
        if in_used:
            return Diagnosis(COLUMN_MAPPING, 0.9, f"{col} exists in {in_used[0]}, which the SQL already uses, "
                             f"but was referenced through another alias", "rule", {**hints, "case": "wrong_alias"})
        if in_retrieved:
            return Diagnosis(TABLE_RETRIEVAL, 0.8, f"{col} lives in {in_retrieved}, retrieved but not used by the SQL",
                             "rule", {**hints, "case": "table_not_used", "tables_to_add": in_retrieved})
        if owner:
            return Diagnosis(TABLE_RETRIEVAL, 0.85, f"{col} lives only in non-retrieved table(s) {owner}", "rule",
                             {**hints, "case": "not_retrieved", "tables_to_add": owner})
        return Diagnosis(COLUMN_MAPPING, 0.7, f"no table has a column named {col} (hallucinated name)", "rule",
                         {**hints, "case": "hallucinated"})

    if cls == "TABLE_OR_VIEW_NOT_FOUND":
        return Diagnosis(TABLE_RETRIEVAL, 0.8, f"table {obs.missing_table or '?'} does not exist", "rule",
                         {"signal": "table_not_found", "missing_table": obs.missing_table})

    if obs.execution_status == "TOO_MANY_ROWS":
        return Diagnosis(JOIN_KEY, 0.6, "result exploded beyond the row limit: likely a missing or wrong join key",
                         "rule", {"signal": "too_many_rows"})

    # UNION of a UTF8_LCASE table column with a plain STRING (CAST(NULL AS STRING), a literal, ...):
    # the engine names the collation, so the fix is known without an LLM diagnosis
    if cls == "INCOMPATIBLE_COLUMN_TYPE" and "COLLATE" in (obs.execution_error or "").upper():
        return Diagnosis(EXECUTION, 0.9, "UNION combines a UTF8_LCASE string column with a plain STRING value "
                         "(e.g. CAST(NULL AS STRING) or a string literal)", "rule",
                         {"signal": "collation_mismatch", "error_class": cls})

    if obs.execution_status == "ERROR":
        if cls in _SQL_LEVEL or cls is None or "window frame" in (obs.execution_error or "").lower():
            return Diagnosis(EXECUTION, 0.8, f"SQL-level error {cls or 'unknown'}", "rule",
                             {"signal": "sql_error", "error_class": cls})
        return Diagnosis(EXECUTION, 0.6, f"engine error {cls}", "rule", {"signal": "sql_error", "error_class": cls})

    # ran, but the self-verifier's semantic checks found a structural problem (verifier.py / checks.py)
    for f in obs.verifier_findings:
        if f.get("signal") in _VERIFIER_TYPES:
            ftype, conf = _VERIFIER_TYPES[f["signal"]]
            return Diagnosis(ftype, conf, f.get("hint") or f.get("message") or f["signal"], "rule",
                             {"signal": f["signal"], "verifier_findings": list(obs.verifier_findings),
                              **(f.get("evidence") or {})})
    return None  # ran successfully: no explicit signal


LLM_SYSTEM = "You diagnose failed Text-to-SQL attempts. Answer with JSON only."

LLM_PROMPT = """A Text-to-SQL attempt produced a result that is probably wrong. Decide the single most likely root cause.

Failure types:
- TABLE_RETRIEVAL_FAILURE: the query uses the wrong tables or misses a table the question needs
- COLUMN_MAPPING_FAILURE: a phrase in the question is mapped to the wrong column
- JOIN_KEY_FAILURE: tables are joined on the wrong columns (duplicated / missing rows)
- DOMAIN_KNOWLEDGE_FAILURE: a business term is translated to the wrong filter or value (e.g. "Course 18", "current term")
- QUERY_DECOMPOSITION_FAILURE: the query structure is wrong (missing sub-question, wrong grouping / aggregation / ordering)
- EXECUTION_FAILURE: a SQL-level mistake unrelated to schema understanding

Question: {question}

Tables available to the attempt: {tables}

Generated SQL:
```sql
{sql}
```

Execution: {status}; rows returned: {rows}; first rows: {preview}

Answer exactly: {{"failure_type": "<one type>", "confidence": <0..1>, "reason": "<one sentence>"}}"""

_JSON = re.compile(r"\{.*\}", re.S)


def parse_llm_diagnosis(text: str) -> tuple[str, float, str] | None:
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ft = str(d.get("failure_type", "")).strip().upper()
    if ft not in FAILURE_TYPES:
        return None
    try:
        conf = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return ft, conf, str(d.get("reason", ""))[:500]


@dataclass
class Diagnoser:
    catalog: SchemaCatalog
    client: ChatClient | None = None  # None = rules only

    def diagnose(self, obs: Observation) -> tuple[Diagnosis, dict[str, Any]]:
        """Returns the diagnosis and LLM usage ({} when no LLM call was made)."""
        d = diagnose_by_rules(obs, self.catalog)
        # 如果可以通过规则诊断出对应的问题
        if d is not None:
            return d, {}

        # 如果LLM客户端不存在
        if self.client is None:
            return Diagnosis(UNKNOWN, 0.0, "no explicit signal and no LLM stage", "fallback"), {}
        # 如果规则诊断不了，LLM客户端存在，生成prompt，并调用LLM
        prompt = LLM_PROMPT.format(question=obs.question, tables=", ".join(obs.retrieved_tables),
                                   sql=obs.generated_sql, status=obs.execution_status,
                                   rows=obs.result_row_count, preview=json.dumps(obs.result_preview)[:600])
        r = self.client.complete(prompt, system=LLM_SYSTEM)
        usage = {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "latency_ms": r.latency_ms,
                 "cached": r.cached}
        parsed = parse_llm_diagnosis(r.text)
        if parsed is None:
            return Diagnosis(UNKNOWN, 0.0, f"unparseable LLM answer: {r.text[:120]}", "llm"), usage
        ft, conf, reason = parsed
        return Diagnosis(ft, conf, reason, "llm", {"signal": "llm"}), usage

"""Phase 3 — Failure taxonomy: ground-truth failure labels for failed attempts.

EVALUATION ONLY. This module reads gold annotations (gold tables, column
mapping, join keys, domain knowledge, decomposition) to decide *why* a failed
attempt failed. Its output is the reference that Diagnosis Accuracy is measured
against; it is never visible to the agent, the diagnoser or repair skills.

BEAVER's five fine-grained subtasks become deterministic checks on the parsed
generated SQL (aliases resolved to base tables). v1 records one primary failure
per attempt using the project's fixed priority, plus every failed check:

    TABLE_RETRIEVAL → COLUMN_MAPPING → JOIN_KEY → DOMAIN_KNOWLEDGE
    → QUERY_DECOMPOSITION → EXECUTION → UNKNOWN

Reference = annotation ∩ what the gold SQL actually does. BEAVER annotations
over-list: in 1,060 of 5,787 dw questions (18%) the annotated ``tables`` include
tables the gold SQL never references (10/30 dev, 15/100 evaluation sample).
Requiring such tables would label correct generations as retrieval failures, so
tables, columns and join keys count only if the gold SQL uses them.

The checks are heuristics (e.g. a generation may use an alternative but valid
column); their agreement with human judgement is measured by the spot check.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from agent.sql_analysis import sql_facts
from benchmark.beaver.dataset import BeaverCase

TABLE_RETRIEVAL = "TABLE_RETRIEVAL_FAILURE"
COLUMN_MAPPING = "COLUMN_MAPPING_FAILURE"
JOIN_KEY = "JOIN_KEY_FAILURE"
DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE_FAILURE"
QUERY_DECOMPOSITION = "QUERY_DECOMPOSITION_FAILURE"
EXECUTION = "EXECUTION_FAILURE"
UNKNOWN = "UNKNOWN"
PRIORITY = (TABLE_RETRIEVAL, COLUMN_MAPPING, JOIN_KEY, DOMAIN_KNOWLEDGE, QUERY_DECOMPOSITION, EXECUTION)

_DK = re.compile(r'"(?P<term>.+?)"\s+is predicated by\s+"(?P<pred>.+)"', re.S)


def _tc(ref: str) -> tuple[str, str] | None:
    parts = ref.strip().strip('"').split(".")
    return (parts[-2].lower(), parts[-1].lower()) if len(parts) >= 2 else None


@dataclass
class FailureLabel:
    case_id: str
    primary: str
    failed_checks: list[str]
    evidence: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoldReference:
    tables: set[str]
    columns: set[tuple[str, str]]
    join_keys: list[tuple[tuple[str, str], tuple[str, str]]]
    operations: set[str]
    literals: set[str]
    dropped_tables: set[str]  # annotated but not used by the gold SQL


def gold_reference(case: BeaverCase, schema: dict[str, dict[str, str]]) -> GoldReference:
    from benchmark.beaver.adapter import apply_rules  # gold SQL is MySQL; parse the adapted form

    g = sql_facts(apply_rules(case.gold_sql)[0], schema)
    ann_tables = {t.lower() for t in case.gold_tables}
    ann_cols = {tc for refs in case.gold_column_mapping.values() for r in (refs or []) if (tc := _tc(r))}
    ann_joins = [(a, b) for p in (case.gold_join_keys or []) if len(p) == 2 and (a := _tc(p[0])) and (b := _tc(p[1]))]
    if not g.parsed:  # fall back to the raw annotation
        return GoldReference(ann_tables, ann_cols, ann_joins, set(), set(), set())
    return GoldReference(
        tables=ann_tables & g.tables,
        columns=ann_cols & g.columns,
        join_keys=[(a, b) for a, b in ann_joins if frozenset((a, b)) in g.equalities],
        operations=g.operations,
        literals=g.literals,
        dropped_tables=ann_tables - g.tables,
    )


def label_failure(case: BeaverCase, generated_sql: str, execution_status: str,
                  retrieved_tables: list[str], schema: dict[str, dict[str, str]]) -> FailureLabel:
    facts = sql_facts(generated_sql, schema)
    ref = gold_reference(case, schema)
    failed: list[str] = []
    ev: dict[str, Any] = {"sql_parsed": facts.parsed, "execution_status": execution_status}
    if ref.dropped_tables:
        ev["annotation_tables_unused_by_gold_sql"] = sorted(ref.dropped_tables)
    if not facts.parsed:
        ev["parse_error"] = facts.error
        return FailureLabel(case.case_id, EXECUTION, [EXECUTION], ev)

    # 1. multi-table retrieval
    gold_tables = ref.tables
    missing_tables = sorted(gold_tables - facts.tables)
    if missing_tables:
        failed.append(TABLE_RETRIEVAL)
        retrieved = {t.lower() for t in retrieved_tables}
        ev["missing_tables"] = {t: ("retrieved_not_used" if t in retrieved else "not_retrieved") for t in missing_tables}

    # 2. column mapping (only columns of gold tables that the SQL does use)
    missing_cols = sorted(f"{t}.{c}" for t, c in ref.columns - facts.columns if t in facts.tables)
    if missing_cols:
        failed.append(COLUMN_MAPPING)
        ev["missing_columns"] = missing_cols

    # 3. join keys (pairs whose tables are both used)
    missing_joins = [f"{a[0]}.{a[1]} = {b[0]}.{b[1]}" for a, b in ref.join_keys
                     if a[0] in facts.tables and b[0] in facts.tables and frozenset((a, b)) not in facts.equalities]
    if missing_joins:
        failed.append(JOIN_KEY)
        ev["missing_join_keys"] = missing_joins

    # 4. domain knowledge (entries whose term appears in the question and whose value the gold SQL uses)
    missing_dk = []
    for entry in case.domain_knowledge or []:
        m = _DK.search(str(entry))
        if not m or m.group("term").lower() not in case.question.lower():
            continue
        values = [v.lower() for v in re.findall(r"'([^']*)'", m.group("pred")) if v.lower() in ref.literals]
        if values and not any(v in facts.literals for v in values):
            missing_dk.append(f"{m.group('term')} => {m.group('pred')}")
    if missing_dk:
        failed.append(DOMAIN_KNOWLEDGE)
        ev["missing_domain_knowledge"] = missing_dk

    # 5. query decomposition: operations the gold query needs but the generation lacks
    missing_ops = sorted(ref.operations - facts.operations)
    if missing_ops:
        failed.append(QUERY_DECOMPOSITION)
        ev["missing_operations"] = missing_ops

    if execution_status not in ("SUCCESS",):
        failed.append(EXECUTION)
    primary = next((p for p in PRIORITY if p in failed), UNKNOWN)
    return FailureLabel(case.case_id, primary, failed, ev)

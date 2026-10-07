"""Phase 3 — interventional failure attribution (OFFLINE ANALYSIS ONLY).

Comparing a failed SQL with gold annotations shows which subtasks deviate, not
which one *caused* the failure (errors cascade downstream). This experiment
supplies causal evidence: for each failed attempt it regenerates the SQL with
exactly ONE subtask's gold hint added (BEAVER setting=1/2 style), plus one run
with all hints as an upper bound.

    only hint X fixes it      -> causal type X
    several single hints fix  -> MULTIPLE_SUFFICIENT (any of them)
    only all hints together   -> MULTI_CAUSE
    nothing fixes it          -> UNRESOLVED (beyond the model / hints)

Hints use the same reference as the labeler (annotation ∩ what the gold SQL
uses). Decomposition hints give sub-questions only, never sub-SQL. Gold hints
are never available to the agent, the loop or any experiment arm.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from agent.generator import FewShotGenerator
from benchmark.beaver import subtasks as S
from benchmark.beaver.dataset import BeaverCase

log = logging.getLogger(__name__)

VARIANTS = ("tables", "columns", "join_keys", "domain_knowledge", "decomposition")
VARIANT_TYPE = {"tables": S.TABLE_RETRIEVAL, "columns": S.COLUMN_MAPPING, "join_keys": S.JOIN_KEY,
                "domain_knowledge": S.DOMAIN_KNOWLEDGE, "decomposition": S.QUERY_DECOMPOSITION}
MULTIPLE_SUFFICIENT, MULTI_CAUSE, UNRESOLVED = "MULTIPLE_SUFFICIENT", "MULTI_CAUSE", "UNRESOLVED"


@dataclass(frozen=True)
class Hint:
    lines: list[str]
    tables: set[str]  # tables the hint refers to; they are added to the schema shown


def build_hints(case: BeaverCase, schema: dict[str, dict[str, str]]) -> dict[str, Hint]:
    ref = S.gold_reference(case, schema)
    hints: dict[str, Hint] = {}
    if ref.tables:
        hints["tables"] = Hint([f"The query needs exactly these tables: {', '.join(sorted(ref.tables))}."], set(ref.tables))
    col_lines, col_tables = [], set()
    for phrase, refs in case.gold_column_mapping.items():
        cols = [f"{t}.{c}" for r in (refs or []) if (tc := S._tc(r)) and tc in ref.columns for t, c in [tc]]
        if cols:
            col_lines.append(f"\"{phrase}\" refers to column(s) {', '.join(cols)}.")
            col_tables |= {c.split(".")[0] for c in cols}
    if col_lines:
        hints["columns"] = Hint(col_lines, col_tables)
    if ref.join_keys:
        hints["join_keys"] = Hint([f"Join on {a[0]}.{a[1]} = {b[0]}.{b[1]}." for a, b in ref.join_keys],
                                  {a[0] for a, _ in ref.join_keys} | {b[0] for _, b in ref.join_keys})
    dk = []
    for entry in case.domain_knowledge or []:
        m = S._DK.search(str(entry))
        if m and m.group("term").lower() in case.question.lower():
            if any(v.lower() in ref.literals for v in re.findall(r"'([^']*)'", m.group("pred"))):
                dk.append(f"{m.group('term')} means: {m.group('pred')}.")
    if dk:
        hints["domain_knowledge"] = Hint(dk, set())
    subs = [q for q in case.sub_questions or [] if isinstance(q, str) and q.strip()]
    if len(subs) > 1:  # a single sub-question just restates the question
        hints["decomposition"] = Hint(["Solve it in these steps: " + " | ".join(f"({i}) {q}" for i, q in enumerate(subs, 1))],
                                      set())
    return hints


def classify(single: dict[str, bool], all_hints: bool | None) -> tuple[str, list[str]]:
    fixed = [v for v, ok in single.items() if ok]
    if len(fixed) == 1:
        return VARIANT_TYPE[fixed[0]], fixed
    if len(fixed) > 1:
        return MULTIPLE_SUFFICIENT, fixed
    if all_hints:
        return MULTI_CAUSE, []
    return UNRESOLVED, []


def run_intervention(cases: list[BeaverCase], retrieved: dict[str, list[str]], generator: FewShotGenerator,
                     executor: Any, judges: dict[str, Callable], schema: dict[str, dict[str, str]],
                     catalog_tables: set[str], workers: int = 4, max_rows: int = 500_000,
                     only: set[str] | None = None) -> list[dict]:
    """``only``: run just these variants (e.g. {"all"} for a cheap upper-bound probe of a new model)."""
    # 1) prompts for every (case, variant)
    jobs = []
    for c in cases:
        hints = build_hints(c, schema)
        variants = {v: hints[v] for v in VARIANTS if v in hints}
        if variants:
            variants["all"] = Hint([ln for h in variants.values() for ln in h.lines],
                                   set().union(*(h.tables for h in variants.values())))
        for name, h in variants.items():
            if only and name not in only:
                continue
            tables = list(retrieved[c.case_id]) + sorted(t for t in h.tables if t in catalog_tables
                                                         and t not in retrieved[c.case_id])
            jobs.append((c, name, h, tuple(tables)))

    # 2) generation in parallel (LLM only), 3) execution + judging sequentially (one DB session)
    def gen(job):
        c, name, h, tables = job
        return generator.generate(c.agent_view(), tables, oracle_hints=h.lines)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        generations = list(pool.map(gen, jobs))
    by_case: dict[str, dict[str, Any]] = {}
    for (c, name, h, tables), g in zip(jobs, generations):
        if g.sql:
            ex = executor.execute(g.sql, c.db, max_rows=max_rows)
            ok = judges[c.case_id](ex.rows)[0] if ex.ok else False
            status = ex.status
        else:
            ok, status = False, g.parse_status
        by_case.setdefault(c.case_id, {})[name] = {"correct": ok, "execution_status": status, "sql": g.sql,
                                                   "hints": h.lines, "tokens": g.llm.total_tokens}
        log.info("%s %-16s exec=%s correct=%s", c.case_id, name, status, ok)
    out = []
    for c in cases:
        res = by_case.get(c.case_id, {})
        single = {v: res[v]["correct"] for v in VARIANTS if v in res}
        causal, fixed_by = classify(single, res.get("all", {}).get("correct"))
        out.append({"case_id": c.case_id, "variants_available": sorted(single), "single_hint_correct": single,
                    "all_hints_correct": res.get("all", {}).get("correct"), "causal_type": causal,
                    "fixed_by": fixed_by, "details": res})
    return out

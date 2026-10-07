"""Outer loop — propose improvements from failures, gate them on held-out questions, hand them to a reviewer.

Company-style "evaluation-driven, human-in-the-loop" improvement cycle:

    ① sample the TRAINING split (stand-in for a company's verified-query log) and run the current system
    ② judge each answer against its gold (offline only) and label why wrong answers are wrong
    ③ mine candidate knowledge from the failures (table preferences among look-alike tables, missing tables,
       join rules); contradicting table preferences are resolved by support
    ④ regression gate, per item, on a held-out validation sample of the training split: current system vs
       current + this candidate (correct, harmed, tokens); recommended items re-checked together and on the dev set
    ⑤ write everything as proposals with a recommendation; a reviewer decides what to adopt —
       nothing goes live automatically.

Gold is used here to learn (training split) and to gate (validation / dev split); the evaluation set is never
touched, and nothing about the question being answered ever reaches the online system.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.curated import CuratedKnowledge
from agent.retriever import tokenize
from agent.sql_analysis import sql_facts
from benchmark.beaver.adapter import apply_rules
from benchmark.beaver.dataset import BeaverCase
from benchmark.beaver.evaluator import cross_engine_match
from benchmark.beaver.subtasks import gold_reference, label_failure

OUTER_LOOP_VERSION = "outer-v1"


@dataclass
class CaseResult:
    case: BeaverCase
    sql: str
    status: str
    correct: bool
    retrieved: list[str]
    tokens: int
    label: Any = None                       # FailureLabel for wrong answers
    gen_tables: set[str] = field(default_factory=set)
    gold_tables: set[str] = field(default_factory=set)
    prompt_hash: str = ""                   # same prompt -> same (cached, greedy) answer: result can be reused


# ---------------------------------------------------------------- ① + ② run and judge

def sample_training(queries: list[dict], exclude_ids: set[str], n: int, seed: int) -> list[dict]:
    pool = sorted((q for q in queries if str(q["id"]) not in exclude_ids and q.get("sql")), key=lambda q: str(q["id"]))
    return random.Random(seed).sample(pool, min(n, len(pool)))


def judged_cases(sample: list[dict], executor: Any, split: str, max_rows: int) -> tuple[list[BeaverCase], dict]:
    """Cases + gold judges for sampled training questions; questions whose gold does not run are dropped."""
    cases, judges = [], {}
    for q in sample:
        c = BeaverCase.from_beaver(q, split)
        j = gold_judge_from_sql(executor, q["sql"], c.db, max_rows)
        if j:
            cases.append(c)
            judges[c.case_id] = j
    return cases, judges


def gold_judge_from_sql(executor: Any, gold_sql: str, db: str, max_rows: int) -> Callable | None:
    """Judge built by executing the (adapted) gold SQL on Databricks; None if the gold does not run."""
    ex = executor.execute(apply_rules(gold_sql.strip().rstrip(";"))[0], db, max_rows=max_rows)
    if not ex.ok:
        return None
    ref = ex.rows

    def judge(rows):
        cmp = cross_engine_match(rows, ref)
        return cmp.set_match, cmp.reason
    return judge


def run_and_judge(cases: list[BeaverCase], judges: dict[str, Callable], retriever: Any, generator: Any,
                  executor: Any, schema: dict, top_k: int, max_rows: int, workers: int = 4,
                  reuse: dict[str, CaseResult] | None = None) -> list[CaseResult]:
    """First-attempt generation (the outer loop improves what the generator knows), parallel LLM calls,
    sequential execution on one warehouse session. ``reuse``: earlier results by case id; a case whose
    prompt is unchanged gets the earlier result without executing again (the LLM answer is cached)."""
    def gen(case):
        retrieval = retriever.retrieve(case.question, top_k)
        return retrieval, generator.generate(case.agent_view(), retrieval.tables)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        generated = list(pool.map(gen, cases))
    out = []
    for case, (retrieval, g) in zip(cases, generated):
        ph = hashlib.sha256(g.prompt.encode("utf-8")).hexdigest()[:16]
        prev = (reuse or {}).get(case.case_id)
        if prev is not None and prev.prompt_hash == ph and prev.sql == g.sql:
            out.append(prev)
            continue
        if g.sql:
            ex = executor.execute(g.sql, case.db, max_rows=max_rows)
            status, rows = ex.status, (ex.rows if ex.ok else [])
        else:
            status, rows = g.parse_status, []
        ok = bool(status == "SUCCESS" and judges[case.case_id](rows)[0])
        r = CaseResult(case, g.sql, status, ok, list(g.schema_tables or retrieval.tables),
                       g.llm.input_tokens + g.llm.output_tokens, prompt_hash=ph)
        if not ok:
            r.label = label_failure(case, g.sql, status, r.retrieved, schema)
            r.gen_tables = sql_facts(g.sql, schema).tables if g.sql else set()
            r.gold_tables = gold_reference(case, schema).tables
        out.append(r)
    return out


# ---------------------------------------------------------------- ③ mine candidates

def _related(a: str, b: str, kb: Any, catalog: Any) -> bool:
    """Look-alike tables: same knowledge-base group, or at least 40% shared column names."""
    if kb is not None and any(a in g and b in g for g in kb.groups):
        return True
    ca = {c.name.upper() for c in catalog.tables[a].columns} if a in catalog.tables else set()
    cb = {c.name.upper() for c in catalog.tables[b].columns} if b in catalog.tables else set()
    return bool(ca and cb) and len(ca & cb) / len(ca | cb) >= 0.4


NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                 "fifteen", "twenty", "thirty", "fifty", "hundred", "thousand", "first", "second", "third"}


def _keywords(questions: list[str], table: str, kb: Any, k: int = 3, min_share: float = 0.5,
              min_lift: float = 2.0) -> list[str]:
    """Content words that tie the supporting questions to ``table``. A word qualifies when it
    * occurs in at least two supporting questions (or the only one) — a pattern, not one question's wording;
    * is not a number (``2021fa``, ``12``, ``five``) and has at least 3 letters;
    * points at the table in the solved-query statistics: P(table | word) >= ``min_share`` and at least
      ``min_lift`` times the table's base rate P(table) — so words like "least" or "description" that
      occur everywhere do not qualify.
    Most shared first, then strongest pointer."""
    if kb is None or not kb.n_queries:
        return []
    base = kb.table_freq.get(table, 0) / kb.n_queries
    need = min(2, len(questions))
    counts = Counter(w for q in questions for w in set(tokenize(q)))
    scored = []
    for w, n in counts.items():
        if n < need or len(w) < 3 or any(ch.isdigit() for ch in w) or w in NUMBER_WORDS:
            continue
        df = kb.word_df.get(w) or 0
        if not df or df >= 0.5 * kb.n_queries:
            continue
        share = kb.word_tables.get(w, {}).get(table, 0) / df
        if share >= min_share and share >= min_lift * base:
            scored.append((n, share, w))
    return [w for _, _, w in sorted(scored, key=lambda x: (-x[0], -x[1], x[2]))[:k]]


def mine_table_preferences(failures: list[CaseResult], kb: Any, catalog: Any, min_support: int = 2) -> list[dict]:
    """Wrong answers that used table X where the gold used look-alike table Y -> 'prefer Y over X'."""
    groups: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    for f in failures:
        missing = f.gold_tables - f.gen_tables
        extra = f.gen_tables - f.gold_tables
        for y in missing:
            for x in extra:
                if _related(x, y, kb, catalog):
                    groups[(y, x)].append(f)
    out = []
    for (y, x), fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(fs) < min_support:
            continue
        kws = _keywords([f.case.question for f in fs], y, kb)
        text = (f"For questions about {', '.join(kws) if kws else 'this topic'}, this warehouse uses table {y}, "
                f"not the look-alike table {x}.")
        out.append({"kind": "table_preference", "title": f"优先用 {y}，而不是相似表 {x}",
                    "content": {"prefer": y, "instead_of": x, "keywords": kws, "text": text},
                    "evidence": {"support": len(fs), "case_ids": [f.case.case_id for f in fs][:10],
                                 "example_question": fs[0].case.question[:400],
                                 "example_wrong_sql": fs[0].sql[:1500]}})
    return out


def _join_condition(a: str, b: str, kb: Any) -> str:
    """Most common join key between two tables in the solved-query statistics ("" if never joined)."""
    keys = (kb.joins.get("|".join(sorted([a, b])), {}).get("keys", {}) if kb is not None else {})
    return max(keys, key=keys.get) if keys else ""


def mine_missing_tables(failures: list[CaseResult], kb: Any, min_support: int = 2) -> list[dict]:
    """Wrong answers that left out a table the gold joins (the most common table error of weak models)
    -> 'questions about <keywords> also need table T, joined on K'. Needs question keywords, so the hint
    only fires for similar questions."""
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for f in failures:
        if f.gen_tables:                                   # nothing parsed -> no table evidence
            for t in f.gold_tables - f.gen_tables:
                groups[t].append(f)
    out = []
    for t, fs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(fs) < min_support:
            continue
        kws = _keywords([f.case.question for f in fs], t, kb)
        if not kws:
            continue
        partners = Counter(p for f in fs for p in (f.gen_tables & f.gold_tables) if p != t)
        conds = [(p, _join_condition(t, p, kb)) for p, _ in partners.most_common()]
        conds = [(p, c) for p, c in conds if c]
        with_ = [p for p, _ in partners.most_common(3)]
        join = f", joined on {conds[0][1]}" if conds else ""
        out.append({"kind": "table_hint", "title": f"问到 {', '.join(kws)} 时，还需要关联表 {t}",
                    "content": {"table": t, "keywords": kws, "with": with_, "condition": conds[0][1] if conds else "",
                                "text": f"Questions about {', '.join(kws)} usually also need table {t}{join}."},
                    "evidence": {"support": len(fs), "case_ids": [f.case.case_id for f in fs][:10],
                                 "example_question": fs[0].case.question[:400],
                                 "example_wrong_sql": fs[0].sql[:1500]}})
    return out


def mine_join_rules(failures: list[CaseResult], kb: Any, min_support: int = 2) -> list[dict]:
    """Gold join conditions the wrong answers missed, between tables they did use."""
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for f in failures:
        for cond in (f.label.evidence.get("missing_join_keys") or []) if f.label else []:
            groups[cond].append(f)
    out = []
    for cond, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(fs) < min_support:
            continue
        a, b = (side.strip().split(".")[0] for side in cond.split("="))
        kinds = (kb.joins.get("|".join(sorted([a, b])), {}).get("kinds", {}) if kb is not None else {})
        join = "LEFT" if kinds and kinds.get("LEFT", 0) >= sum(kinds.values()) / 2 else "INNER"
        cond_u = " = ".join(f"{s.split('.')[0]}.{s.split('.')[1].upper()}" for s in (x.strip() for x in cond.split("=")))
        out.append({"kind": "join_rule", "title": f"{a} 与 {b} 关联：{cond_u}",
                    "content": {"tables": sorted([a, b]), "condition": cond_u, "join": join,
                                "text": f"Join {a} and {b} on {cond_u} ({join} JOIN)."},
                    "evidence": {"support": len(fs), "case_ids": [f.case.case_id for f in fs][:10],
                                 "example_question": fs[0].case.question[:400]}})
    return out


def resolve_conflicts(candidates: list[dict], min_ratio: float = 2.0) -> tuple[list[dict], list[dict]]:
    """Opposite table preferences ("prefer Y over X" and "prefer X over Y") contradict each other: keep the one
    with at least ``min_ratio`` times the support, otherwise drop both — the failures do not say which is
    conventional. Returns (kept, dropped)."""
    prefs: dict[frozenset, list[dict]] = defaultdict(list)
    for c in candidates:
        if c["kind"] == "table_preference":
            prefs[frozenset((c["content"]["prefer"], c["content"]["instead_of"]))].append(c)
    dropped: list[dict] = []
    for group in prefs.values():
        if len({c["content"]["prefer"] for c in group}) < 2:
            continue                                            # one direction only: no conflict
        group = sorted(group, key=lambda c: -c["evidence"]["support"])
        top, rest = group[0], group[1:]
        if top["evidence"]["support"] >= min_ratio * rest[0]["evidence"]["support"]:
            dropped += rest
        else:
            dropped += group
    ids = {id(c) for c in dropped}
    return [c for c in candidates if id(c) not in ids], dropped


def as_curated(candidates: list[dict], prefix: str = "cand") -> CuratedKnowledge:
    return CuratedKnowledge([{"item_id": f"{prefix}-{i}", "kind": c["kind"], "content": c["content"]}
                             for i, c in enumerate(candidates)])


# ---------------------------------------------------------------- ④ regression gate

def compare(before: list[CaseResult], after: list[CaseResult], max_token_increase: float = 0.2) -> dict:
    b = {r.case.case_id: r for r in before}
    a = {r.case.case_id: r for r in after}
    fixed = sorted(c for c in a if a[c].correct and not b[c].correct)
    harmed = sorted(c for c in a if b[c].correct and not a[c].correct)
    tb, ta = sum(r.tokens for r in before), sum(r.tokens for r in after)
    res = {
        "cases": len(before),
        "affected": sorted(c for c in a if a[c].prompt_hash != b[c].prompt_hash),   # prompt changed
        "correct_before": sum(r.correct for r in before), "correct_after": sum(r.correct for r in after),
        "executable_before": sum(r.status == "SUCCESS" for r in before),
        "executable_after": sum(r.status == "SUCCESS" for r in after),
        "fixed": fixed, "harmed": harmed,
        "tokens_before": tb, "tokens_after": ta, "token_increase": round((ta - tb) / tb, 4) if tb else 0.0,
    }
    ok = res["correct_after"] >= res["correct_before"] and not harmed and res["token_increase"] <= max_token_increase
    res["gate_passed"] = ok
    return res


def recommendation(reg: dict) -> str:
    if not reg["gate_passed"]:
        return "建议驳回：回归未通过（答对变少、有误伤或 token 增幅过大）"
    if reg["correct_after"] > reg["correct_before"]:
        return "建议批准：答对增加且没有误伤"
    if "affected" in reg and not reg["affected"]:
        return "中性：门禁题里没有会触发这条的题，回归无法验证，由审核人判断"
    return "中性：不退步，但也没有提升，由审核人判断"


def gate_candidates(candidates: list[dict], before: list[CaseResult],
                    run: Callable[[CuratedKnowledge], list[CaseResult]],
                    max_token_increase: float = 0.2) -> tuple[list[dict], dict | None]:
    """Per-item gate: each candidate alone vs the current system, so one bad item cannot sink good ones
    (and a batch gain cannot hide a harmful item). ``run`` should reuse unchanged cases (``reuse=``), so an
    item costs only the dev questions whose prompt it changes. The items recommended for approval are then
    checked together, since knowledge items can interact. Returns (per-item regressions, combined or None)."""
    per_item = [compare(before, run(as_curated([c])), max_token_increase) for c in candidates]
    good = [c for c, reg in zip(candidates, per_item) if reg["gate_passed"] and reg["fixed"]]
    combined = None
    if len(good) >= 2:
        combined = {**compare(before, run(as_curated(good)), max_token_increase), "items": len(good)}
    return per_item, combined


# ---------------------------------------------------------------- ⑤ proposal rows

def proposal_rows(batch_id: str, candidates: list[dict], regressions: list[dict],
                  batch: dict | None = None) -> list[dict]:
    """One row per candidate with its own regression and recommendation; ``batch`` (combined check of the
    recommended items, the training summary, model ...) is attached to every row of the batch."""
    rows = []
    for i, (c, reg) in enumerate(zip(candidates, regressions)):
        rec = recommendation(reg)
        good = reg["gate_passed"] and reg["fixed"]
        if good and batch and batch.get("combined") and not batch["combined"]["gate_passed"]:
            rec = "中性：单独回归通过，但与本批其他建议批准的条目合用时未通过，建议逐条批准并观察"
        elif good and batch and batch.get("dev_check") and not batch["dev_check"]["gate_passed"]:
            rec = "中性：验证集回归通过，但开发集复核时有退步，建议观察后再批准"
        rows.append({"proposal_id": f"{batch_id}-{i:02d}", "batch_id": batch_id, "kind": c["kind"], "title": c["title"],
                     "content_json": json.dumps(c["content"], ensure_ascii=False),
                     "evidence_json": json.dumps(c["evidence"], ensure_ascii=False, default=str),
                     "regression_json": json.dumps({**reg, "batch": batch or {}}, ensure_ascii=False, default=str),
                     "recommendation": rec})
    return rows


def failure_rows(failures: list[CaseResult], kb: Any, catalog: Any) -> list[dict]:
    """Per failure: tables used vs gold, and which wrong/missing pairs count as look-alike (mining input)."""
    return [{"case_id": f.case.case_id, "status": f.status, "primary": f.label.primary if f.label else None,
             "gen_tables": sorted(f.gen_tables), "gold_tables": sorted(f.gold_tables),
             "look_alike_pairs": [f"{y}<-{x}" for y in sorted(f.gold_tables - f.gen_tables)
                                  for x in sorted(f.gen_tables - f.gold_tables) if _related(x, y, kb, catalog)],
             "missing_join_keys": (f.label.evidence.get("missing_join_keys") or []) if f.label else []}
            for f in failures]


def failure_summary(results: list[CaseResult]) -> dict:
    wrong = [r for r in results if not r.correct]
    return {"cases": len(results), "correct": len(results) - len(wrong),
            "executable": sum(r.status == "SUCCESS" for r in results),
            "primary_labels": dict(Counter(r.label.primary for r in wrong if r.label))}

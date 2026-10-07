"""Phase 1 — Few-shot baseline run: one attempt per case, no retry.

Agent components (retriever, generator, executor) only ever receive an
``AgentTask`` (case_id, question, db). Gold information is used strictly after
the attempt, by the evaluation code in this module:
* execution accuracy via a per-case ``judge``: the frozen
  ``benchmark.gold_results`` for the evaluation set (``gold_judge``), MySQL gold
  rows for the development set (``evaluation/devset.py``);
* ``eval_table_recall``: share of gold tables that retrieval returned
  (diagnostic for Phase 3, never fed back to the agent).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import statistics
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.generator import PROMPT_VERSION, FewShotGenerator
from agent.retriever import BM25TableRetriever
from benchmark.beaver.dataset import BeaverCase
from benchmark.beaver.evaluator import evaluate_against_gold, serialize_rows

log = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    top_k: int = 20
    max_result_rows: int = 500_000


def select_pilot(primary_ids: list[str], n: int, seed: int) -> list[str]:
    """Fixed pilot subset of the PRIMARY cases (independent of run order)."""
    ids = sorted(primary_ids)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


Judge = Callable[[list], tuple[bool, str]]


def gold_judge(gold_json: str) -> Judge:
    return lambda rows: evaluate_against_gold(rows, gold_json)


def run_case(case: BeaverCase, judge: Judge, retriever: BM25TableRetriever, generator: FewShotGenerator,
             executor: Any, cfg: BaselineConfig) -> dict[str, Any]:
    task = case.agent_view()  # ← the only thing the agent sees
    retrieval = retriever.retrieve(task.question, cfg.top_k)
    gen = generator.generate(task, retrieval.tables)
    if gen.sql:
        ex = executor.execute(gen.sql, task.db, max_rows=cfg.max_result_rows)
        exec_status, exec_error, rows, exec_ms = ex.status, ex.error, ex.rows, ex.elapsed_ms
    else:
        exec_status, exec_error, rows, exec_ms = gen.parse_status, None, [], 0
    correct, message = judge(rows) if exec_status == "SUCCESS" else (False, exec_status)

    # ---- evaluation-only diagnostics (after the attempt; never shown to the agent)
    gold_tables = {t.lower() for t in case.gold_tables}
    got = set(retrieval.tables)
    return {
        "case_id": case.case_id, "attempt_id": 1, "question": task.question,
        "retrieved_tables": list(retrieval.tables), "retrieval_scores": list(retrieval.scores),
        "generated_sql": gen.sql, "raw_response": gen.raw, "parse_status": gen.parse_status,
        "execution_status": exec_status, "execution_error": (exec_error or "")[:2000] or None,
        "result_row_count": len(rows) if exec_status == "SUCCESS" else None,
        # what the agent itself would observe of its result (no gold): shape + first rows
        "result_preview": serialize_rows(rows[:5]) if exec_status == "SUCCESS" else None,
        "correct": correct, "eval_message": message,
        "input_tokens": gen.llm.input_tokens, "output_tokens": gen.llm.output_tokens,
        "total_tokens": gen.llm.total_tokens, "llm_latency_ms": gen.llm.latency_ms, "llm_cached": gen.llm.cached,
        "exec_latency_ms": exec_ms, "latency_ms": gen.llm.latency_ms + exec_ms,
        "finish_reason": gen.llm.finish_reason, "model": gen.llm.model,
        "eval_table_recall": round(len(gold_tables & got) / len(gold_tables), 4) if gold_tables else None,
        "eval_all_gold_tables_retrieved": gold_tables <= got,
        "category": case.category,
    }


def summarize(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    ok = [r for r in records if r["correct"]]
    by_cat: dict[str, list[bool]] = {}
    for r in records:
        by_cat.setdefault(r["category"] or "?", []).append(r["correct"])
    full = [r for r in records if r["eval_all_gold_tables_retrieved"]]
    return {
        "cases": n,
        "correct": len(ok),
        "first_pass_accuracy": round(len(ok) / n, 4) if n else None,
        "execution_status": dict(Counter(r["execution_status"] for r in records)),
        "parse_status": dict(Counter(r["parse_status"] for r in records)),
        "executable_rate": round(sum(r["execution_status"] == "SUCCESS" for r in records) / n, 4) if n else None,
        "accuracy_by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "accuracy_when_all_gold_tables_retrieved": f"{sum(r['correct'] for r in full)}/{len(full)}",
        "accuracy_when_some_gold_table_missing": f"{sum(r['correct'] for r in records if not r['eval_all_gold_tables_retrieved'])}/{n - len(full)}",
        "mean_table_recall": round(statistics.mean(r["eval_table_recall"] for r in records), 4) if n else None,
        "tokens_total": sum(r["total_tokens"] for r in records),
        "tokens_mean": round(statistics.mean(r["total_tokens"] for r in records)) if n else None,
        "llm_latency_ms_median": statistics.median(r["llm_latency_ms"] for r in records) if n else None,
        "latency_ms_median": statistics.median(r["latency_ms"] for r in records) if n else None,
        "finish_reason": dict(Counter(r["finish_reason"] for r in records)),
    }


def run_baseline(cases: list[BeaverCase], judges: dict[str, Judge], retriever: BM25TableRetriever,
                 generator: FewShotGenerator, executor: Any, cfg: BaselineConfig, out_root: Path,
                 run_meta: dict[str, Any]) -> tuple[str, dict]:
    run_id = f"baseline-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {**run_meta, "run_id": run_id, "prompt_version": getattr(generator, "prompt_version", PROMPT_VERSION), "top_k": cfg.top_k,
            "few_shot": [{"id": e.source_id, "question": e.question, "sql": e.sql} for e in generator.examples]}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    records = []
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            rec = run_case(case, judges[case.case_id], retriever, generator, executor, cfg)
            rec["run_id"] = run_id
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            log.info("[%d/%d] %s %s exec=%s correct=%s tokens=%d", i, len(cases), case.case_id,
                     rec["parse_status"], rec["execution_status"], rec["correct"], rec["total_tokens"])
    summary = {"run_id": run_id, **summarize(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_id, summary

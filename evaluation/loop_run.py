"""Phase 6/7 — run one experiment arm (strategy x verifier) and compute loop metrics.

Correctness is judged here, after the loop has finished, with gold-derived
judges (frozen gold results for the evaluation set, MySQL gold for the dev
set). The loop itself never sees these judgments — except through
``OracleVerifier``, whose results are labelled as an upper bound.

Metrics (docs/PROJECT_POSITIONING.md §3):
    recovery rate = first wrong & final right / first wrong
    harm rate     = first right & final wrong / first right
    net gain      = recovered - harmed
plus the verifier confusion on attempt 1 and cost (attempts, tokens, latency).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from benchmark.beaver.dataset import BeaverCase
from loop_engineer.controller import LoopController

log = logging.getLogger(__name__)


def _tokens(a: dict) -> int:
    return int(a.get("input_tokens") or 0) + int(a.get("output_tokens") or 0) + int(a.get("diag_tokens") or 0)


# 入口
def run_arm(cases: list[BeaverCase], judges: dict[str, Callable], controller: LoopController,
            verifier_for: Callable[[str], Any], arm: str, out_root: Path, meta: dict,
            on_event: Callable[[str, str, dict], None] | None = None) -> tuple[str, dict]:
    """``on_event(case_id, step, payload)``: live progress for UIs. The ``judged`` step comes from
    the evaluation side, after the loop for that case has finished."""
    run_id = f"{arm}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps({**meta, "run_id": run_id, "arm": arm}, indent=2,
                                                  ensure_ascii=False, default=str), encoding="utf-8")
    records = []
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            cb = (lambda step, payload, cid=case.case_id: on_event(cid, step, payload)) if on_event else None
            if on_event:
                on_event(case.case_id, "start", {"index": i, "total": len(cases), "question": case.question})
            # 运行示例
            res = controller.run(case.agent_view(), verifier_for(case.case_id), on_event=cb)
            judge = judges[case.case_id]
            correct = [judge(rows)[0] if a["execution_status"] == "SUCCESS" else False
                       for a, rows in zip(res.attempts, res.rows)]
            rec = {"run_id": run_id, "arm": arm, "case_id": case.case_id, "category": case.category,
                   "attempts": res.attempts, "attempt_correct": correct, "final_index": res.final_index,
                   "first_correct": correct[0], "final_correct": correct[res.final_index],
                   "n_attempts": len(res.attempts), "tokens": sum(_tokens(a) for a in res.attempts),
                   "extra_tokens": sum(_tokens(a) for a in res.attempts[1:]),
                   "latency_ms": sum(int(a.get("llm_latency_ms") or 0) + int(a.get("exec_latency_ms") or 0)
                                     for a in res.attempts)}
            records.append(rec)
            if on_event:
                on_event(case.case_id, "judged", {"attempt_correct": correct, "final_correct": rec["final_correct"],
                                                  "tokens": rec["tokens"], "latency_ms": rec["latency_ms"]})
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            a1 = res.attempts[0]
            log.info("[%d/%d] %s v1=%s %s -> attempts=%d final_correct=%s", i, len(cases), case.case_id,
                     a1["verifier_decision"], a1.get("repair_skill") or "", len(res.attempts), rec["final_correct"])
    summary = {"run_id": run_id, "arm": arm, **summarize(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_id, summary


def summarize(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    first = sum(r["first_correct"] for r in records)
    final = sum(r["final_correct"] for r in records)
    recovered = sum(not r["first_correct"] and r["final_correct"] for r in records)
    harmed = sum(r["first_correct"] and not r["final_correct"] for r in records)
    initial_failures = n - first
    # verifier on attempt 1 vs actual correctness
    conf = Counter()
    for r in records:
        trig = r["attempts"][0]["verifier_decision"] == "FAIL"
        conf[("triggered" if trig else "passed", "wrong" if not r["first_correct"] else "right")] += 1
    per_skill = defaultdict(lambda: {"repairs": 0, "recovered": 0, "executable_after": 0})
    for r in records:
        for k, a in enumerate(r["attempts"][1:], 1):
            s = per_skill[a.get("repair_skill") or "?"]
            s["repairs"] += 1
            s["executable_after"] += a["execution_status"] == "SUCCESS"
            s["recovered"] += bool(r["attempt_correct"][k]) and not r["first_correct"]
    tokens = [r["tokens"] for r in records]
    extra = sum(r["extra_tokens"] for r in records)
    net = recovered - harmed
    return {
        "cases": n,
        "first_pass_accuracy": round(first / n, 4) if n else None,
        "final_accuracy": round(final / n, 4) if n else None,
        "first_correct": first, "final_correct": final,
        "initial_failures": initial_failures, "recovered": recovered,
        "recovery_rate": round(recovered / initial_failures, 4) if initial_failures else None,
        "harmed": harmed, "harm_rate": round(harmed / first, 4) if first else None,
        "net_gain": net,
        "avg_attempts": round(statistics.mean(r["n_attempts"] for r in records), 3) if n else None,
        "tokens_total": sum(tokens), "tokens_mean": round(statistics.mean(tokens)) if n else None,
        "extra_tokens_total": extra,
        "extra_tokens_per_net_recovery": round(extra / net) if net > 0 else None,
        "latency_ms_median": statistics.median(r["latency_ms"] for r in records) if n else None,
        "executable_first": sum(r["attempts"][0]["execution_status"] == "SUCCESS" for r in records),
        "executable_final": sum(r["attempts"][r["final_index"]]["execution_status"] == "SUCCESS" for r in records),
        "verifier_confusion": {"triggered_and_wrong(hit)": conf[("triggered", "wrong")],
                               "triggered_but_right(false_alarm)": conf[("triggered", "right")],
                               "passed_but_wrong(miss)": conf[("passed", "wrong")],
                               "passed_and_right": conf[("passed", "right")]},
        "per_skill": dict(per_skill),
    }

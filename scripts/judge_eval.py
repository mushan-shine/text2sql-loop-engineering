"""Offline evaluation of the LLM judge (loop_engineer/judge.py) as a verifier signal.

    python scripts/judge_eval.py --model glm-4-flash
    python scripts/judge_eval.py --model deepseek-flash

Population (dev questions only, evaluation set excluded — decision D2):
  * WRONG   — executed-but-wrong attempts from existing runs (same set as scripts/verifier_eval.py);
  * CORRECT — the dev-set gold SQL with its gold rows, plus model attempts that were judged correct.
Reports hit rate (judge says "wrong" on WRONG) and false-alarm rate (judge says "wrong" on CORRECT),
overall and by confidence threshold, plus token cost. Gold is used only to label the population.
Note: gold SQL is written in MySQL style (BEAVER), model SQL in Databricks style — a possible bias.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from agent.llm import CachingChatClient, UsageMeter, make_client  # noqa: E402
from agent.retriever import SchemaCatalog  # noqa: E402
from evaluation.devset import decode_rows  # noqa: E402
from loop_engineer.judge import JUDGE_VERSION, LlmJudge  # noqa: E402
from scripts.verifier_eval import BASELINES, wrong_attempts  # noqa: E402


def correct_items(dev: dict) -> list[dict]:
    items = [{"case_id": f"dw:{c['id']}", "question": c["question"], "sql": c["sql"],
              "rows": decode_rows(c["mysql_rows"]), "n_rows": len(c["mysql_rows"]), "source": "gold"}
             for c in dev["cases"]]
    for d in BASELINES:
        meta = json.loads((ROOT / d / "run_meta.json").read_text(encoding="utf-8"))
        for line in (ROOT / d / "results.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["correct"]:
                items.append({"case_id": r["case_id"], "question": r["question"], "sql": r["generated_sql"],
                              "rows": [tuple(x) for x in json.loads(r["result_preview"] or "[]")],
                              "n_rows": r["result_row_count"], "source": f"attempt/{meta['model']}"})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="glm-4-flash")
    ap.add_argument("--config", default="config/phase1.yaml")
    args = ap.parse_args()
    for line in ((ROOT / ".env").read_text(encoding="utf-8") if (ROOT / ".env").exists() else "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    catalog = SchemaCatalog.from_json((ROOT / f"runs/phase1/schema_{cfg['beaver']['db']}.json").read_text(encoding="utf-8"))
    meter = UsageMeter(max_calls=150, max_tokens=1_500_000)
    client = CachingChatClient(make_client(model=args.model, max_output_tokens=512, meter=meter),
                               ROOT / cfg["llm"]["cache"])
    judge = LlmJudge(client, catalog)

    wrong = wrong_attempts()
    correct = correct_items(json.loads((ROOT / cfg["dev"]["path"]).read_text(encoding="utf-8")))
    results = []
    for label, items in (("wrong", wrong), ("correct", correct)):
        for it in items:
            j = judge.judge(it["question"], it["sql"], it["rows"], it["n_rows"])
            results.append({"label": label, "case_id": it["case_id"], "source": it["source"], "verdict": j.verdict,
                            "confidence": j.confidence, "problems": list(j.problems), "tokens": j.tokens})
            print(f"{label:7s} {it['case_id']:12s} {it['source']:28s} -> {j.verdict:11s} {j.confidence:.2f}")

    def rate(label, thr=0.0):
        xs = [r for r in results if r["label"] == label]
        k = sum(r["verdict"] == "wrong" and r["confidence"] >= thr for r in xs)
        return k, len(xs)

    table = []
    for thr in (0.0, 0.6, 0.7, 0.8, 0.9):
        (h, nw), (fa, nc) = rate("wrong", thr), rate("correct", thr)
        table.append({"threshold": thr, "hits": h, "wrong": nw, "false_alarms": fa, "correct": nc})
    unparse = sum(r["verdict"] == "unparseable" for r in results)
    fa_by_source = {}
    for r in results:
        if r["label"] == "correct":
            s = fa_by_source.setdefault(r["source"].split("/")[0], [0, 0])
            s[0] += r["verdict"] == "wrong"
            s[1] += 1
    summary = {"model": args.model, "judge": JUDGE_VERSION, "table": table, "unparseable": unparse,
               "false_alarms_by_source": fa_by_source, "llm_usage": meter.snapshot(), "results": results}
    out = ROOT / "runs" / "verifier_eval" / f"judge-{args.model}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmodel {args.model}  unparseable {unparse}  usage {meter.snapshot()}")
    print(f"{'conf>=':>7s} {'hits on wrong':>15s} {'false alarms on correct':>25s}")
    for t in table:
        print(f"{t['threshold']:7.1f} {t['hits']:>6d}/{t['wrong']:<8d} {t['false_alarms']:>12d}/{t['correct']:<12d}")
    print("false alarms by source:", fa_by_source)


if __name__ == "__main__":
    main()

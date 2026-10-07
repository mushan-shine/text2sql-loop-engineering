"""Offline evaluation of the tool-grounded validators (loop_engineer/validators.py). No LLM calls.

    python scripts/validator_eval.py --pool 300

Populations (evaluation set excluded everywhere, decision D2):
  * WRONG   — executed-but-wrong attempts from existing dev runs (same set as scripts/verifier_eval.py);
  * CORRECT — dev-set gold SQL + model attempts judged correct;
  * POOL    — N random non-evaluation gold SQL that contain a string filter or a join with an aggregate
              (correct by definition: every finding there is a false alarm).
The probes run on the Databricks copy of the warehouse. Gold only labels the populations.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from agent.retriever import SchemaCatalog  # noqa: E402
from benchmark.beaver.loader import load_cases, load_from_local_json  # noqa: E402
from execution.databricks_sql import DatabricksSqlExecutor  # noqa: E402
from loop_engineer.validators import VALIDATOR_SIGNALS, ValidatorAgent  # noqa: E402
from scripts.judge_eval import correct_items  # noqa: E402
from scripts.verifier_eval import wrong_attempts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--config", default="config/phase1.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    b, d = cfg["beaver"], cfg["databricks"]
    catalog = SchemaCatalog.from_json((ROOT / f"runs/phase1/schema_{b['db']}.json").read_text(encoding="utf-8"))
    dbx = DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), ansi_mode=bool(d["ansi_mode"]),
                                statement_timeout_s=int(d["statement_timeout_s"]))

    wrong = wrong_attempts()
    dev = json.loads((ROOT / cfg["dev"]["path"]).read_text(encoding="utf-8"))
    correct = correct_items(dev)
    queries, _ = load_from_local_json(str(ROOT / b["local_dir"]), b["split"])
    eval_ids = {c.case_id.split(":", 1)[1] for c in load_cases("local_json", b["split"], int(b["sample_size"]),
                                                               int(b["sample_seed"]), str(ROOT / b["local_dir"]))[0]}
    dev_ids = {str(c["id"]) for c in dev["cases"]}
    pool = [q for q in queries if str(q["id"]) not in eval_ids | dev_ids and q.get("sql")
            and ("'" in q["sql"] or ("JOIN" in q["sql"].upper() and any(f in q["sql"].upper() for f in ("SUM(", "AVG(", "COUNT("))))]
    random.Random(20260928).shuffle(pool)
    pool = [{"case_id": f"dw:{q['id']}", "sql": q["sql"], "source": "pool"} for q in pool[: args.pool]]

    results = []
    for label, items in (("wrong", wrong), ("correct", correct), ("pool", pool)):
        for it in items:
            agent = ValidatorAgent(dbx, catalog, b["db"])
            fs = agent.validate(it["sql"])
            results.append({"label": label, "case_id": it["case_id"], "source": it["source"], "probes": agent.probes_run,
                            "signals": sorted({f.signal for f in fs}), "messages": [f.message for f in fs]})
    dbx.close()

    table = []
    for s in VALIDATOR_SIGNALS:
        row = {"signal": s}
        for label in ("wrong", "correct", "pool"):
            xs = [r for r in results if r["label"] == label]
            row[label] = sum(s in r["signals"] for r in xs)
            row[f"{label}_total"] = len(xs)
            row[f"{label}_probed"] = sum(r["probes"] > 0 for r in xs)
        table.append(row)
    examples = defaultdict(list)
    for r in results:
        for m in r["messages"]:
            examples[r["label"]].append((r["case_id"], m))
    out = ROOT / "runs" / "verifier_eval" / "validators.json"
    out.write_text(json.dumps({"table": table, "results": results,
                               "examples": {k: v[:12] for k, v in examples.items()}}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"{'signal':24s} {'hits on wrong':>14s} {'FA dev correct':>15s} {'FA pool gold':>13s}")
    for t in table:
        print(f"{t['signal']:24s} {t['wrong']:>6d}/{t['wrong_total']:<7d} {t['correct']:>6d}/{t['correct_total']:<8d}"
              f" {t['pool']:>5d}/{t['pool_total']:<7d}")
    print("SQL where at least one probe ran:",
          {k: f"{table[0][k + '_probed']}/{table[0][k + '_total']}" for k in ("wrong", "correct", "pool")})
    print("probes total:", sum(r["probes"] for r in results))


if __name__ == "__main__":
    main()

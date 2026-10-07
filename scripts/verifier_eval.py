"""Offline evaluation of the semantic verifier checks (loop_engineer/checks.py). No LLM calls.

    python scripts/verifier_eval.py                 # preview rows only
    python scripts/verifier_eval.py --reexecute     # re-run the wrong attempts on Databricks for full result rows

For every signal:
  * hits          — how many executed-but-WRONG attempts it flags (what the verifier should catch);
                    taken from existing runs, deduplicated by (question, SQL);
  * false alarms  — how often it flags CORRECT SQL: the dev-set gold SQL with its gold rows (static + result
                    checks) and the gold SQL of every non-evaluation dw question (static checks only).
Gold is used here only to MEASURE the checks; the checks themselves never see it.
Evaluation-set questions are excluded everywhere (decision D2).
"""
from __future__ import annotations

import argparse
import json
import os
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

from benchmark.beaver.loader import load_cases, load_from_local_json  # noqa: E402
from evaluation.devset import decode_rows  # noqa: E402
from loop_engineer.checks import check_numeric, check_result, check_static, output_roles  # noqa: E402



def dev_baseline_dirs() -> list[str]:
    """Your own dev-set baseline runs (runs/phase1/baseline-*, mode "dev"), oldest first."""
    out = []
    for d in sorted((ROOT / "runs/phase1").glob("baseline-*")):
        meta = d / "run_meta.json"
        if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("mode") == "dev":
            out.append(str(d.relative_to(ROOT)))
    return out


BASELINES = dev_baseline_dirs()


def wrong_attempts() -> list[dict]:
    """Executed-but-wrong attempts from the dev-set runs, one per (case, SQL)."""
    out, seen = [], set()

    def add(case_id, question, sql, preview, n_rows, source):
        key = (case_id, " ".join((sql or "").split()))
        if key in seen or not sql:
            return
        seen.add(key)
        rows = [tuple(r) for r in json.loads(preview)] if preview else []
        out.append({"case_id": case_id, "question": question, "sql": sql, "rows": rows, "n_rows": n_rows,
                    "source": source})

    for d in BASELINES:
        meta = json.loads((ROOT / d / "run_meta.json").read_text(encoding="utf-8"))
        for line in (ROOT / d / "results.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["execution_status"] == "SUCCESS" and not r["correct"] and r.get("result_row_count"):
                add(r["case_id"], r["question"], r["generated_sql"], r.get("result_preview"), r["result_row_count"],
                    f"baseline/{meta['model']}")
    for d in sorted((ROOT / "runs/phase6").glob("*-self-*")) + sorted((ROOT / "runs/phase6").glob("*-oracle-*")):
        for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            for a, ok in zip(r["attempts"], r["attempt_correct"]):
                if a["execution_status"] == "SUCCESS" and not ok and a.get("result_row_count"):
                    add(r["case_id"], a["question"], a["generated_sql"], a.get("result_preview"),
                        a["result_row_count"], f"loop/{d.name.split('-2026')[0]}")
    return out


def reexecute(items: list[dict], cfg: dict) -> None:
    from execution.databricks_sql import DatabricksSqlExecutor
    d = cfg["databricks"]
    dbx = DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), ansi_mode=bool(d["ansi_mode"]),
                                statement_timeout_s=int(d["statement_timeout_s"]))
    for it in items:
        ex = dbx.execute(it["sql"], cfg["beaver"]["db"], max_rows=int(d["max_result_rows"]))
        if ex.ok:
            it["rows"] = ex.rows
    dbx.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reexecute", action="store_true", help="fetch full result rows of the wrong attempts")
    ap.add_argument("--mysql-pool", type=int, default=0,
                    help="also run N non-evaluation gold SQL with aggregates on local MySQL (numeric false alarms)")
    ap.add_argument("--config", default="config/phase1.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    b = cfg["beaver"]

    wrong = wrong_attempts()
    if args.reexecute:
        reexecute(wrong, cfg)
    dev = json.loads((ROOT / cfg["dev"]["path"]).read_text(encoding="utf-8"))
    queries, _ = load_from_local_json(str(ROOT / b["local_dir"]), b["split"])
    eval_ids = {c.case_id.split(":", 1)[1] for c in load_cases("local_json", b["split"], int(b["sample_size"]),
                                                               int(b["sample_seed"]), str(ROOT / b["local_dir"]))[0]}
    pool = [q for q in queries if str(q["id"]) not in eval_ids and q.get("sql")]

    hits, fa_dev, fa_pool = defaultdict(list), Counter(), Counter()
    def all_checks(q, sql, rows):
        return check_static(q, sql) + check_result(q, sql, rows) + check_numeric(sql, rows)

    def numeric_applicable(sql, rows):   # at least one comparable relation (e.g. avg+min+max, std+var, a count)
        roles = [r for r in output_roles(sql) if r]
        by_arg = Counter(r.arg for r in roles)
        return bool(rows) and (any(r.role in ("count", "std", "var") for r in roles) or any(v >= 2 for v in by_arg.values()))

    for it in wrong:
        for f in all_checks(it["question"], it["sql"], it["rows"]):
            hits[f.signal].append((it["case_id"], it["source"], f.message))
    caught = {(it["case_id"], it["sql"]) for it in wrong if all_checks(it["question"], it["sql"], it["rows"])}
    num_app_wrong = sum(numeric_applicable(it["sql"], it["rows"]) for it in wrong)
    num_app_dev = 0
    fa_examples = defaultdict(list)
    for c in dev["cases"]:
        rows = decode_rows(c["mysql_rows"])
        num_app_dev += numeric_applicable(c["sql"], rows)
        for f in all_checks(c["question"], c["sql"], rows):
            fa_dev[f.signal] += 1
            fa_examples[f.signal].append((c["id"], f.message))
    fa_mysql, mysql_run, mysql_app = Counter(), 0, 0
    if args.mysql_pool:
        from execution.mysql import MySqlExecutor
        for line in ((ROOT / ".env").read_text(encoding="utf-8") if (ROOT / ".env").exists() else "").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        my = MySqlExecutor.from_config({"host": "localhost", "port": 3306, "timeout_s": 30})
        cand = [q for q in pool if sum(1 for r in output_roles(q["sql"]) if r) >= 2]
        random.Random(20260928).shuffle(cand)
        for q in cand[: args.mysql_pool]:
            ex = my.execute(q["sql"], b["db"])
            if not ex.ok or not ex.rows:
                continue
            mysql_run += 1
            mysql_app += numeric_applicable(q["sql"], ex.rows)
            for f in check_numeric(q["sql"], ex.rows):
                fa_mysql[f.signal] += 1
                fa_examples[f.signal].append((q["id"], f.message))
        my.close()

    pool_parsed = 0
    for q in pool:
        fs = check_static(q["question"], q["sql"])
        pool_parsed += 1
        for s in {f.signal for f in fs}:
            fa_pool[s] += 1

    signals = sorted(set(hits) | set(fa_dev) | set(fa_pool) | set(fa_mysql))
    table = []
    for s in signals:
        h = len({(c, src) for c, src, _ in hits[s]})
        table.append({"signal": s, "hits_on_wrong": h, "wrong_total": len(wrong),
                      "false_alarm_dev_gold": fa_dev[s], "dev_gold_total": len(dev["cases"]),
                      "false_alarm_pool_gold": fa_pool[s], "pool_total": pool_parsed,
                      "false_alarm_rate_pool": round(fa_pool[s] / pool_parsed, 4) if pool_parsed else None,
                      "false_alarm_mysql_gold": fa_mysql[s], "mysql_gold_total": mysql_run})
    summary = {"wrong_attempts": len(wrong), "caught_by_any": len(caught), "sources": dict(Counter(i["source"] for i in wrong)),
               "reexecuted": args.reexecute, "table": table,
               "numeric_applicable": {"wrong": num_app_wrong, "dev_gold": num_app_dev, "mysql_gold": mysql_app,
                                      "mysql_gold_run": mysql_run},
               "examples": {s: hits[s][:6] for s in signals},
               "false_alarm_examples": {s: fa_examples[s][:6] for s in fa_examples}}
    out = ROOT / "runs" / "verifier_eval" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrong-but-executed attempts: {len(wrong)}  {dict(Counter(i['source'] for i in wrong))}")
    print(f"caught by at least one signal: {len(caught)}")
    print(f"numeric checks applicable: wrong {num_app_wrong}/{len(wrong)}, dev gold {num_app_dev}/{len(dev['cases'])}, "
          f"mysql gold {mysql_app}/{mysql_run}")
    print(f"{'signal':24s} {'hits(wrong)':>12s} {'FA dev gold':>12s} {'FA pool (static)':>17s} {'FA mysql gold':>14s}")
    for t in table:
        print(f"{t['signal']:24s} {t['hits_on_wrong']:>5d}/{t['wrong_total']:<6d} {t['false_alarm_dev_gold']:>5d}/{t['dev_gold_total']:<6d}"
              f" {t['false_alarm_pool_gold']:>7d}/{t['pool_total']:<8d} {t['false_alarm_mysql_gold']:>6d}/{t['mysql_gold_total']:<6d}")


if __name__ == "__main__":
    main()

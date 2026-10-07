"""Error analysis of one published dev-set run (e.g. a console run), against the dev-set gold.

    python scripts/analyze_run.py console-targeted-self-20260928T105340-eef1a9

Pulls the run's traces + judgments from Delta, takes each question's FINAL attempt and explains why a
wrong one is wrong: the deterministic failure labeler (tables / columns / join keys / literals /
operations vs. the gold SQL) plus a result-level comparison (column count, row count, values).
Offline analysis on the dev set only (decision D2); nothing here feeds back into the loop at runtime.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from agent.retriever import SchemaCatalog  # noqa: E402
from benchmark.beaver.evaluator import cross_engine_match  # noqa: E402
from benchmark.beaver.subtasks import label_failure  # noqa: E402
from evaluation.devset import decode_rows, dev_cases, load_devset  # noqa: E402
from dbx.runtime import project_catalog  # noqa: E402
from execution.databricks_sql import DatabricksSqlExecutor  # noqa: E402

CAT = project_catalog()


def main() -> None:
    run_id = sys.argv[1]
    cfg = yaml.safe_load((ROOT / "config/phase1.yaml").read_text(encoding="utf-8"))
    devset = load_devset(ROOT / cfg["dev"]["path"])
    cases = {c.case_id: c for c in dev_cases(devset, cfg["beaver"]["split"])}
    gold_rows = {f"{cfg['beaver']['split']}:{c['id']}": decode_rows(c["mysql_rows"]) for c in devset["cases"]}
    catalog = SchemaCatalog.from_json((ROOT / f"runs/phase1/schema_{cfg['beaver']['db']}.json").read_text(encoding="utf-8"))
    schema = {t: {c.name.lower(): c.type for c in tb.columns} for t, tb in catalog.tables.items()}

    dbx = DatabricksSqlExecutor(catalog=CAT, profile="DEFAULT")
    T = dbx.query_df(f"SELECT * FROM `{CAT}`.`traces`.`execution_traces` WHERE run_id = '{run_id}' ORDER BY case_id, attempt_id")
    E = dbx.query_df(f"SELECT case_id, attempt_id, correct FROM `{CAT}`.`evaluation`.`evaluation_results` WHERE run_id = '{run_id}'")
    correct = {(r.case_id, int(r.attempt_id)): bool(r.correct) for r in E.itertuples()}

    out = []
    for cid, g in T.groupby("case_id"):
        g = g.sort_values("attempt_id")
        fin = g[g["final_status"] == "FINAL"].iloc[0]
        ok = correct.get((cid, int(fin.attempt_id)), False)
        rec = {"case_id": cid, "attempts": len(g), "final_attempt": int(fin.attempt_id), "correct": ok,
               "exec": fin.execution_status, "skills": [s for s in g["repair_skill"].tolist() if s],
               "verifier": fin.verifier_decision}
        if not ok and fin.execution_status == "SUCCESS":
            case = cases[cid]
            lab = label_failure(case, fin.generated_sql, fin.execution_status, json.loads(fin.retrieved_tables), schema)
            ex = dbx.execute(fin.generated_sql, cfg["beaver"]["db"], max_rows=200_000)
            gold = gold_rows[cid]
            rows = ex.rows if ex.ok else []
            cmp = cross_engine_match(rows, gold)
            rec.update({"label": lab.primary, "failed_checks": lab.failed_checks,
                        "missing_tables": sorted(set(lab.evidence.get("missing_tables", {}))),
                        "extra_tables": sorted(set(lab.evidence.get("extra_tables", []) or [])),
                        "missing_columns": lab.evidence.get("missing_columns", [])[:6],
                        "missing_literals": lab.evidence.get("missing_literals", [])[:6],
                        "ops": {"gold": lab.evidence.get("gold_operations"), "gen": lab.evidence.get("generated_operations")},
                        "cols": (len(rows[0]) if rows else None, len(gold[0]) if gold else None),
                        "rows": (len(rows), len(gold)), "compare": cmp.reason,
                        "category": case.category, "question": case.question, "sql": fin.generated_sql,
                        "gold_sql": case.gold_sql})
        out.append(rec)
    dbx.close()

    wrong = [r for r in out if not r["correct"] and r["exec"] == "SUCCESS"]
    print(f"run {run_id}: {len(out)} questions, correct {sum(r['correct'] for r in out)}, executed-but-wrong {len(wrong)}, "
          f"not executed {sum(r['exec'] != 'SUCCESS' for r in out)}")
    print("primary label:", dict(Counter(r["label"] for r in wrong)))
    print("all failed checks:", dict(Counter(c for r in wrong for c in r["failed_checks"])))
    print("column count gen vs gold equal:", sum(r["cols"][0] == r["cols"][1] for r in wrong), "/", len(wrong))
    print("row count gen vs gold equal:", sum(r["rows"][0] == r["rows"][1] for r in wrong), "/", len(wrong))
    print("compare reasons:", dict(Counter(r["compare"].split(":")[0][:40] for r in wrong)))
    print("missing gold tables:", dict(Counter(t for r in wrong for t in r["missing_tables"])))
    for r in wrong:
        print(f"- {r['case_id']:10s} {r['label'][:18]:18s} cols {r['cols']} rows {r['rows']} checks {r['failed_checks']} "
              f"missing_tables {r['missing_tables']}")
    dest = ROOT / "runs" / "analysis" / f"{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("saved", dest)


if __name__ == "__main__":
    main()

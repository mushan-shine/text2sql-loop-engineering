"""Phase 3 CLI — failure taxonomy labels for a run (EVALUATION ONLY).

    python scripts/phase3.py label runs/phase1/<run_id>            # labels + distribution
    python scripts/phase3.py label runs/phase1/<run_id> --publish  # also write evaluation.failure_labels
    python scripts/phase3.py spotcheck runs/phase1/<run_id> --n 20 # markdown sheet for human review
    python scripts/phase3.py intervene runs/phase1/<dev_run_id>      # causal attribution with gold hints (dev only)

Labels are written under the run directory (``failure_labels.json``) and,
with --publish, to ``evaluation.failure_labels`` — never to traces.*.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pyarrow as pa  # noqa: E402

from agent.retriever import SchemaCatalog  # noqa: E402
from benchmark.beaver.loader import load_cases  # noqa: E402
from benchmark.beaver.subtasks import PRIORITY, UNKNOWN, label_failure  # noqa: E402
from evaluation.devset import dev_cases, load_devset  # noqa: E402

FAILURE_LABELS = pa.schema([
    ("run_id", pa.string()), ("case_id", pa.string()), ("split", pa.string()), ("attempt_id", pa.int64()),
    ("actual_failure_type", pa.string()), ("failed_checks", pa.string()), ("evidence", pa.string()),
    ("labeler_version", pa.string()), ("created_at", pa.timestamp("us", tz="UTC")),
])
LABELER_VERSION = "taxonomy-v1 (annotation ∩ gold SQL)"


def load(run_dir: Path):
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    recs = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if meta.get("mode") == "dev":
        cases = {c.case_id: c for c in dev_cases(load_devset(Path(meta["config"]["dev"]["path"])))}
        split = "dev"
    else:
        b = meta["config"]["beaver"]
        cases = {c.case_id: c for c in load_cases("local_json", b["split"], int(b["sample_size"]),
                                                  int(b["sample_seed"]), b["local_dir"])[0]}
        split = "eval"
    catalog = SchemaCatalog.from_json(Path(f"runs/phase1/schema_{meta['config']['beaver']['db']}.json").read_text(encoding="utf-8"))
    schema = {t: {c.name.lower(): c.type for c in tb.columns} for t, tb in catalog.tables.items()}
    return meta, recs, cases, schema, split


def label_run(run_dir: Path) -> tuple[dict, list[dict]]:
    meta, recs, cases, schema, split = load(run_dir)
    out = []
    for r in recs:
        if r["correct"]:
            continue
        lab = label_failure(cases[r["case_id"]], r["generated_sql"], r["execution_status"], r["retrieved_tables"], schema)
        out.append({"run_id": meta["run_id"], "case_id": r["case_id"], "split": split,
                    "attempt_id": r.get("attempt_id", 1), "actual_failure_type": lab.primary,
                    "failed_checks": lab.failed_checks, "evidence": lab.evidence, "labeler_version": LABELER_VERSION})
    prim = Counter(x["actual_failure_type"] for x in out)
    checks = Counter(c for x in out for c in x["failed_checks"])
    missing = Counter(k for x in out for k in x["evidence"].get("missing_tables", {}).values())
    summary = {"run_id": meta["run_id"], "split": split, "attempts": len(recs), "failures": len(out),
               "primary_distribution": {k: prim.get(k, 0) for k in (*PRIORITY, UNKNOWN)},
               "all_failed_checks": dict(checks),
               "missing_gold_tables": dict(missing),
               "labeler_version": LABELER_VERSION}
    (run_dir / "failure_labels.json").write_text(json.dumps({"summary": summary, "labels": out}, indent=2,
                                                            ensure_ascii=False), encoding="utf-8")
    return summary, out


def spotcheck(run_dir: Path, n: int, seed: int) -> Path:
    meta, recs, cases, schema, _ = load(run_dir)
    labels = {x["case_id"]: x for x in json.loads((run_dir / "failure_labels.json").read_text(encoding="utf-8"))["labels"]}
    ids = sorted(labels)
    pick = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    by_id = {r["case_id"]: r for r in recs}
    lines = [f"# Failure label spot check — {meta['run_id']}", "",
             "For each case: does `actual_failure_type` name the most important reason the attempt failed?",
             "Fill in **agree / disagree** and, when disagreeing, the label you would give.", ""]
    for i, cid in enumerate(pick, 1):
        lab, r, c = labels[cid], by_id[cid], cases[cid]
        lines += [f"## {i}. {cid}", "", f"**Question:** {c.question}", "",
                  f"**Label:** `{lab['actual_failure_type']}` · all failed checks: {', '.join(lab['failed_checks'])}", "",
                  "**Evidence:**", "```json", json.dumps(lab["evidence"], indent=2, ensure_ascii=False), "```", "",
                  f"**Execution:** {r['execution_status']} {(r.get('execution_error') or '')[:300]}", "",
                  "**Generated SQL:**", "```sql", r["generated_sql"] or "(none)", "```", "",
                  "**Gold SQL (evaluation only):**", "```sql", c.gold_sql, "```", "",
                  "**Review:** agree / disagree → ______", "", "---", ""]
    out = run_dir / "spotcheck.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def publish(labels: list[dict]) -> None:
    import logging
    logging.getLogger("databricks").setLevel(logging.WARNING)
    from dbx.catalog import Layout, table_exists, write_rows
    from execution.databricks_sql import DatabricksSqlExecutor

    from dbx.runtime import project_catalog
    dbx = DatabricksSqlExecutor(catalog=project_catalog())
    layout = Layout(project_catalog())
    now = dt.datetime.now(dt.timezone.utc)
    rows = [{**x, "failed_checks": json.dumps(x["failed_checks"]), "evidence": json.dumps(x["evidence"], ensure_ascii=False),
             "created_at": now} for x in labels]
    if rows and table_exists(dbx, layout, layout.evaluation, "failure_labels"):
        dbx.run(f"DELETE FROM {layout.fq(layout.evaluation, 'failure_labels')} WHERE run_id = '{rows[0]['run_id']}'")
    if rows:
        write_rows(dbx, layout, layout.evaluation, "failure_labels", rows, FAILURE_LABELS)
    dbx.close()


def intervene(run_dir: Path, workers: int, only: set[str] | None = None) -> dict:
    """Interventional attribution on a DEV run (offline analysis; gold hints never reach the agent)."""
    import logging

    from agent.generator import FewShotGenerator, select_few_shot
    from agent.llm import CachingChatClient, UsageMeter, make_client
    from benchmark.beaver.dataset import BeaverCase
    from benchmark.beaver.loader import load_from_local_json
    from evaluation.devset import dev_judges
    from evaluation.intervention import VARIANT_TYPE, run_intervention
    from execution.databricks_sql import DatabricksSqlExecutor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("databricks").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    for line in (Path(".env").read_text(encoding="utf-8") if Path(".env").exists() else "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    meta, recs, _, schema, split = load(run_dir)
    if split != "dev":
        raise SystemExit("interventions run on the dev set only (gold hints must not touch evaluation tuning)")
    cfg = meta["config"]
    b, lc, fs = cfg["beaver"], cfg["llm"], cfg["few_shot"]
    queries, _ = load_from_local_json(b["local_dir"], b["split"])
    by_id = {str(q["id"]): q for q in queries}
    failed = [r for r in recs if not r["correct"]]
    cases = [BeaverCase.from_beaver(by_id[r["case_id"].split(":", 1)[1]], b["split"]) for r in failed]
    retrieved = {r["case_id"]: r["retrieved_tables"] for r in failed}

    eval_ids = {c.case_id.split(":", 1)[1] for c in load_cases("local_json", b["split"], int(b["sample_size"]),
                                                               int(b["sample_seed"]), b["local_dir"])[0]}
    examples = select_few_shot(queries, eval_ids, int(fs["n_examples"]), int(fs["seed"]),
                               int(fs["max_tables"]), int(fs["max_sql_chars"]))
    assert [e.question for e in examples] == [e["question"] for e in meta["few_shot"]], "few-shot drifted"
    catalog = SchemaCatalog.from_json(Path(f"runs/phase1/schema_{b['db']}.json").read_text(encoding="utf-8"))
    client = make_client(model=meta["model"], max_output_tokens=int(lc["max_output_tokens"]),
                         meter=UsageMeter(max_calls=400, max_tokens=5_000_000))
    generator = FewShotGenerator(CachingChatClient(client, Path(lc["cache"])), catalog, examples)
    d = cfg["databricks"]
    dbx = DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), ansi_mode=bool(d["ansi_mode"]),
                                statement_timeout_s=int(d["statement_timeout_s"]))
    judges = dev_judges(load_devset(Path(cfg["dev"]["path"])), b["split"])
    results = run_intervention(cases, retrieved, generator, dbx, judges, schema, set(catalog.tables), workers, only=only)
    dbx.close()

    # compare with the deterministic labels of the same run
    det = {x["case_id"]: x for x in label_run(run_dir)[1]}
    unique = [r for r in results if r["causal_type"] in VARIANT_TYPE.values()]
    summary = {
        "dev_run": meta["run_id"], "model": meta["model"], "variants": sorted(only) if only else "all", "failures": len(results),
        "causal_distribution": dict(Counter(r["causal_type"] for r in results)),
        "single_hint_fix_rate": {v: f"{sum(r['single_hint_correct'].get(v, False) for r in results)}/"
                                    f"{sum(v in r['single_hint_correct'] for r in results)}" for v in VARIANT_TYPE},
        "all_hints_fix_rate": f"{sum(bool(r['all_hints_correct']) for r in results)}/{len(results)}",
        "unique_causal_cases": len(unique),
        "primary_agrees_with_causal": f"{sum(det[r['case_id']]['actual_failure_type'] == r['causal_type'] for r in unique)}/{len(unique)}",
        "causal_in_failed_checks": f"{sum(r['causal_type'] in det[r['case_id']]['failed_checks'] for r in unique)}/{len(unique)}",
        "llm_usage": client.meter.snapshot(),
    }
    for r in results:
        r["deterministic_primary"] = det[r["case_id"]]["actual_failure_type"]
        r["deterministic_failed_checks"] = det[r["case_id"]]["failed_checks"]
    out = Path("runs/phase3") / f"intervention-{meta['run_id']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2, ensure_ascii=False,
                                                 default=str), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["label", "spotcheck", "intervene"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("run_dir")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--variants", default="", help="intervene: comma-separated subset, e.g. 'all' (model probe)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if args.mode == "label":
        summary, labels = label_run(run_dir)
        if args.publish:
            publish(labels)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.mode == "intervene":
        print(json.dumps(intervene(run_dir, args.workers, {v for v in args.variants.split(",") if v} or None), indent=2, ensure_ascii=False))
    else:
        print(spotcheck(run_dir, args.n, args.seed))


if __name__ == "__main__":
    main()

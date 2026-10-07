"""Phase 4 CLI — diagnose the failed attempts of a run and score Diagnosis Accuracy.

    python scripts/phase4.py runs/phase1/<run_id>                 # rules only
    python scripts/phase4.py runs/phase1/<run_id> --llm           # rules + LLM for signal-less failures
    python scripts/phase4.py runs/phase1/<run_id> --llm --publish # also traces.diagnoses / evaluation.diagnosis_eval

Requires the phase-3 labels of the run (``python scripts/phase3.py label <run_dir>``).
Tune on dev runs; run the evaluation set once per frozen diagnoser.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pyarrow as pa  # noqa: E402

from agent.retriever import SchemaCatalog  # noqa: E402
from evaluation.diagnosis_eval import score  # noqa: E402
from loop_engineer.diagnose import DIAGNOSER_VERSION, Diagnoser  # noqa: E402
from loop_engineer.observer import observe  # noqa: E402

S, F, B, I, T = pa.string(), pa.float64(), pa.bool_(), pa.int64(), pa.timestamp("us", tz="UTC")
DIAGNOSES = pa.schema([("run_id", S), ("case_id", S), ("attempt_id", I), ("failure_type", S), ("confidence", F),
                       ("reason", S), ("source", S), ("repair_hints", S), ("version", S), ("llm_tokens", I),
                       ("created_at", T)])
DIAGNOSIS_EVAL = pa.schema([("run_id", S), ("case_id", S), ("predicted_failure_type", S), ("actual_failure_type", S),
                            ("actual_failed_checks", S), ("diagnosis_correct", B), ("diagnosis_correct_lenient", B),
                            ("diagnosis_confidence", F), ("diagnosis_source", S), ("created_at", T)])


def load_env() -> None:
    for line in (Path(".env").read_text(encoding="utf-8") if Path(".env").exists() else "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--llm", action="store_true", help="use the LLM stage for failures without an explicit signal")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("databricks").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    recs = [json.loads(x) for x in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    labels_path = run_dir / "failure_labels.json"
    if not labels_path.exists():
        raise SystemExit(f"missing {labels_path}; run: python scripts/phase3.py label {run_dir}")
    labels = {x["case_id"]: x for x in json.loads(labels_path.read_text(encoding="utf-8"))["labels"]}
    db = meta["config"]["beaver"]["db"]
    catalog = SchemaCatalog.from_json(Path(f"runs/phase1/schema_{db}.json").read_text(encoding="utf-8"))

    client = None
    if args.llm:
        from agent.llm import CachingChatClient, UsageMeter, make_client
        load_env()
        lc = meta["config"]["llm"]
        inner = make_client(model=meta["model"], max_output_tokens=512, meter=UsageMeter(max_calls=300))
        client = CachingChatClient(inner, Path(lc["cache"]))
    diagnoser = Diagnoser(catalog, client)

    diagnoses = {}
    for r in recs:
        if r["correct"]:  # diagnosable failures = the attempts that actually failed
            continue
        d, usage = diagnoser.diagnose(observe(r, db))  # observe() whitelists: no gold / correctness fields
        diagnoses[r["case_id"]] = {**d.to_row(), "llm_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)}

    summary, rows = score(diagnoses, labels)
    summary = {"run_id": meta["run_id"], "diagnoser": DIAGNOSER_VERSION, "llm_stage": args.llm, **summary}
    tag = "llm" if args.llm else "rules"
    (run_dir / f"diagnoses_{tag}.json").write_text(json.dumps({"summary": summary, "diagnoses": diagnoses,
                                                               "scored": rows}, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    if args.publish:
        from dbx.catalog import Layout, table_exists, write_rows
        from execution.databricks_sql import DatabricksSqlExecutor

        from dbx.runtime import project_catalog
        dbx = DatabricksSqlExecutor(catalog=project_catalog())
        layout, now = Layout(project_catalog()), dt.datetime.now(dt.timezone.utc)
        drows = [{"run_id": meta["run_id"], "case_id": cid, "attempt_id": 1, **{k: v for k, v in d.items()},
                  "created_at": now} for cid, d in diagnoses.items()]
        erows = [{"run_id": meta["run_id"], **{**r, "actual_failed_checks": json.dumps(r["actual_failed_checks"])},
                  "created_at": now} for r in rows]
        for schema_name, table, rs, arrow in ((layout.traces, "diagnoses", drows, DIAGNOSES),
                                              (layout.evaluation, "diagnosis_eval", erows, DIAGNOSIS_EVAL)):
            if table_exists(dbx, layout, schema_name, table):
                dbx.run(f"DELETE FROM {layout.fq(schema_name, table)} WHERE run_id = '{meta['run_id']}'")
            write_rows(dbx, layout, schema_name, table, rs, arrow)
        dbx.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

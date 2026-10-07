"""Publish local run directories to Delta (traces.*, evaluation.*) and MLflow.

    python scripts/publish_run.py runs/phase1/baseline-20260925T084206-51d855
    python scripts/publish_run.py runs/phase1/baseline-*          # several runs
    python scripts/publish_run.py <run_dir> --experiment-id baseline --no-mlflow
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dbx.catalog import Layout  # noqa: E402
from dbx.publish import MLFLOW_EXPERIMENT, publish_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--catalog", default=None, help="default: the project catalog (config)")
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--experiment-id", default="baseline", help="experiment arm: baseline | generic_retry | targeted_loop | ...")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    if not args.catalog:
        from dbx.runtime import project_catalog
        args.catalog = project_catalog()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("databricks").setLevel(logging.WARNING)
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", args.profile)

    from execution.databricks_sql import DatabricksSqlExecutor

    dbx = DatabricksSqlExecutor(catalog=args.catalog, profile=args.profile)
    user = dbx.workspace_config.username if getattr(dbx.workspace_config, "username", None) else None
    if not user:
        from databricks.sdk import WorkspaceClient
        user = WorkspaceClient(config=dbx.workspace_config).current_user.me().user_name
    experiment = None if args.no_mlflow else MLFLOW_EXPERIMENT.format(user=user)
    dirs = sorted({d for pattern in args.run_dirs for d in glob.glob(pattern)})
    out = [publish_run(dbx, Layout(args.catalog), Path(d), args.experiment_id, experiment) for d in dirs]
    dbx.close()
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

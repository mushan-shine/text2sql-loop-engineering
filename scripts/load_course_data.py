"""Load the course data package into your own Databricks workspace (replaces the MySQL-based step 1).

    python scripts/load_course_data.py                       # download the package from HuggingFace
    python scripts/load_course_data.py --from-dir course_data  # use an already downloaded package

What it does (idempotent: tables that already hold the expected rows are skipped):

1. creates the project catalog layout (catalog, schemas ``benchmark`` / ``traces`` / ``evaluation`` / ``dw``,
   staging volume) — the same as ``python scripts/phase0.py env``;
2. for every table in the package: creates it with its exact column types (string columns keep the
   ``UTF8_LCASE`` collation, which parquet cannot carry), uploads the parquet file to the staging volume and
   inserts it; then checks the row count against the manifest;
3. recreates the agent-facing view ``benchmark.cases_agent_view`` (question fields only, no gold);
4. copies the local artifacts (BEAVER question snapshot, dev set, step-1 result files) into ``data/`` and
   ``runs/`` of this project.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from dbx.catalog import ensure_layout, upload_file  # noqa: E402
from dbx.runtime import load_dotenv  # noqa: E402

log = logging.getLogger("load_course_data")


def fetch(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="HuggingFace dataset id of the package (default: course_data.repo in config)")
    ap.add_argument("--revision", help="package version (git revision / tag on HuggingFace)")
    ap.add_argument("--from-dir", default=os.environ.get("SHT_COURSE_DATA_DIR"),
                    help="use a package folder instead of downloading (default: $SHT_COURSE_DATA_DIR)")
    ap.add_argument("--config", default="config/phase0.yaml")
    ap.add_argument("--force", action="store_true", help="reload tables even when the row counts already match")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("databricks", "urllib3", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d = cfg["databricks"]
    pkg = Path(args.from_dir) if args.from_dir else fetch(args.repo or cfg["course_data"]["repo"], args.revision)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))

    from execution.databricks_sql import DatabricksSqlExecutor
    from dbx.runtime import project_catalog
    dbx = DatabricksSqlExecutor(catalog=project_catalog(), profile=d.get("profile"), http_path=d.get("http_path"),
                                ansi_mode=bool(cfg["environment"]["ansi_mode"]),
                                statement_timeout_s=int(d["statement_timeout_s"]))
    layout, notes = ensure_layout(dbx, project_catalog(), d.get("fallback_catalog"), cfg["schemas"])
    for n in notes:
        log.info(n)
    dbx.run(f"CREATE SCHEMA IF NOT EXISTS `{layout.catalog}`.`dw`")

    report = []
    for t in manifest["tables"]:
        fq = f"`{layout.catalog}`.`{t['schema']}`.`{t['table']}`"
        if not args.force:
            try:
                if int(dbx.run(f"SELECT COUNT(*) FROM {fq}")[0][0]) == t["rows"]:
                    report.append((t["schema"], t["table"], t["rows"], "already loaded"))
                    continue
            except Exception:  # table does not exist yet
                pass
        cols = ", ".join(f"`{c['name']}` {c['type']}" for c in t["columns"])
        dbx.run(f"CREATE OR REPLACE TABLE {fq} ({cols})")
        vol = f"{layout.staging_root}/course_data/{t['schema']}/{t['table']}.parquet"
        upload_file(dbx.workspace_config, vol, str(pkg / t["file"]))
        select = ", ".join(f"CAST(`{c['name']}` AS {c['type']}) AS `{c['name']}`" for c in t["columns"])
        # parquet.`path`, not read_files(): read_files adds its own _rescued_data column, which clashes with
        # tables that were themselves loaded through read_files and carry one
        dbx.run(f"INSERT INTO {fq} SELECT {select} FROM parquet.`{vol}`")
        n = int(dbx.run(f"SELECT COUNT(*) FROM {fq}")[0][0])
        if n != t["rows"]:
            raise SystemExit(f"{t['schema']}.{t['table']}: loaded {n} rows, expected {t['rows']}")
        report.append((t["schema"], t["table"], n, "loaded"))
        log.info("%s.%s: %d rows", t["schema"], t["table"], n)

    # agent-facing view: question fields only, gold columns are not selectable through it
    dbx.run(f"CREATE OR REPLACE VIEW {layout.fq(layout.benchmark, 'cases_agent_view')} AS "
            f"SELECT case_id, question, db FROM {layout.fq(layout.benchmark, 'cases')}")
    dbx.close()

    for rel in manifest["files"]:
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pkg / "files" / rel, dst)

    by_schema: dict[str, list] = {}
    for s, tb, n, state in report:
        by_schema.setdefault(s, []).append(n)
    print(json.dumps({"catalog": layout.catalog,
                      "tables": {s: {"count": len(v), "rows": sum(v)} for s, v in by_schema.items()},
                      "loaded_now": sum(1 for r in report if r[3] == "loaded"),
                      "already_loaded": sum(1 for r in report if r[3] == "already loaded"),
                      "files": manifest["files"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Phase 0 CLI — BEAVER → Databricks qualification, run from a workstation.

    python scripts/phase0.py env          # 01 connectivity + UC layout
    python scripts/phase0.py import       # 02 BEAVER cases + table metadata → benchmark.*
    python scripts/phase0.py replicate    # 02b MySQL (BEAVER dump) → UC schemas + fidelity check
    python scripts/phase0.py compat       # 03 gold SQL compatibility (MySQL oracle vs Databricks)
    python scripts/phase0.py gold         # 04 freeze benchmark.gold_results
    python scripts/phase0.py report       # answers Q1–Q5 → reports/phase0_report.md
    python scripts/phase0.py all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# Windows consoles default to GBK, which cannot print emoji / some model output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from benchmark.beaver import loader, phase0, report  # noqa: E402
from dbx.catalog import Layout, ensure_layout  # noqa: E402

log = logging.getLogger("phase0")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_config(path: Path, overrides: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if overrides.catalog:
        cfg["databricks"]["catalog"] = overrides.catalog
    if overrides.sample_size:
        cfg["beaver"]["sample_size"] = overrides.sample_size
    if overrides.split:
        cfg["beaver"]["split"] = overrides.split
    if overrides.source:
        cfg["beaver"]["source"] = overrides.source
    return cfg


def connect_databricks(cfg: dict):
    from execution.databricks_sql import DatabricksSqlExecutor

    d, env = cfg["databricks"], cfg["environment"]
    return DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), http_path=d.get("http_path"),
                                 ansi_mode=bool(env["ansi_mode"]), statement_timeout_s=int(d["statement_timeout_s"]))


def connect_mysql(cfg: dict):
    from execution.mysql import MySqlExecutor

    return MySqlExecutor.from_config(cfg["reference_engine"])


def layout_for(cfg: dict, dbx) -> Layout:
    layout, notes = ensure_layout(dbx, cfg["databricks"]["catalog"], cfg["databricks"].get("fallback_catalog"),
                                  cfg["schemas"])
    dbx.catalog = layout.catalog
    for n in notes:
        log.info(n)
    return layout


def load_cases(cfg: dict):
    b = cfg["beaver"]
    if b["source"] == "huggingface":
        queries, tables = loader.load_from_huggingface(b["split"])
        loader.save_local_snapshot(b["local_dir"], b["split"], queries, tables)  # frozen reference copy
        sampled = loader.sample_official(queries, int(b["sample_size"]), int(b["sample_seed"]))
        from benchmark.beaver.dataset import BeaverCase

        return [BeaverCase.from_beaver(e, b["split"]) for e in sampled], tables, len(queries)
    return loader.load_cases("local_json", b["split"], int(b["sample_size"]), int(b["sample_seed"]), b["local_dir"])


def qc_for(cfg: dict) -> phase0.QualificationConfig:
    q = cfg["qualification"]
    return phase0.QualificationConfig(int(q["stability_repeats"]), bool(q["try_adapter"]),
                                      bool(q["admit_validated_adaptations"]), json.dumps(cfg["environment"]))


# --------------------------------------------------------------------------- steps


def step_env(cfg: dict) -> dict:
    out: dict = {"config": {k: cfg[k] for k in ("databricks", "schemas", "environment", "beaver")}}
    mysql = connect_mysql(cfg)
    out["mysql_version"] = mysql.query("SELECT VERSION()")[0][0]
    out["mysql_lower_case_table_names"] = mysql.query("SELECT @@lower_case_table_names")[0][0]
    out["mysql_databases"] = [r[0] for r in mysql.query("SHOW DATABASES")]
    mysql.close()
    dbx = connect_databricks(cfg)
    layout = layout_for(cfg, dbx)
    out["databricks_http_path"] = dbx.http_path
    out["databricks_catalog"] = layout.catalog
    out["databricks_version"] = dbx.run("SELECT current_version()")[0][0]
    try:
        dbx.run("SELECT 'A' COLLATE UTF8_LCASE = 'a'")
        out["collation_supported"] = True
    except Exception as e:  # older runtimes
        out["collation_supported"] = False
        out["collation_error"] = str(e)[:200]
    dbx.close()
    phase0.save_local("01_environment.json", out)
    return out


def step_import(cfg: dict) -> dict:
    cases, tables, n_total = load_cases(cfg)
    dbx = connect_databricks(cfg)
    layout = layout_for(cfg, dbx)
    summary = phase0.import_cases(dbx, layout, cases, tables)
    summary["split_size"] = n_total
    phase0.save_local("02_import.json", summary)
    dbx.close()
    return summary


def step_replicate(cfg: dict) -> dict:
    cases, _, _ = loader.load_cases("local_json", cfg["beaver"]["split"], int(cfg["beaver"]["sample_size"]),
                                    int(cfg["beaver"]["sample_seed"]), cfg["beaver"]["local_dir"])
    mysql, dbx = connect_mysql(cfg), connect_databricks(cfg)
    layout = layout_for(cfg, dbx)
    s = phase0.replicate_databases(mysql, dbx, layout, sorted({c.db for c in cases}),
                                   cfg["environment"]["collation_policy"])
    mysql.close(); dbx.close()
    return s


def step_compat(cfg: dict) -> dict:
    cases, _, _ = loader.load_cases("local_json", cfg["beaver"]["split"], int(cfg["beaver"]["sample_size"]),
                                    int(cfg["beaver"]["sample_seed"]), cfg["beaver"]["local_dir"])
    mysql, dbx = connect_mysql(cfg), connect_databricks(cfg)
    layout = layout_for(cfg, dbx)
    s = phase0.run_compatibility(cases, mysql, dbx, layout, qc_for(cfg))
    mysql.close(); dbx.close()
    return s


def step_gold(cfg: dict, compat_run_id: str | None) -> dict:
    cases, _, _ = loader.load_cases("local_json", cfg["beaver"]["split"], int(cfg["beaver"]["sample_size"]),
                                    int(cfg["beaver"]["sample_seed"]), cfg["beaver"]["local_dir"])
    if not compat_run_id:
        compat_run_id = json.loads(Path("runs/phase0/03_compatibility.json").read_text(encoding="utf-8"))["run_id"]
    dbx = connect_databricks(cfg)  # fresh session: tests cross-session stability
    layout = layout_for(cfg, dbx)
    s = phase0.build_gold_results(cases, dbx, layout, compat_run_id, qc_for(cfg))
    dbx.close()
    return s


def step_report() -> dict:
    md, answers = report.build_report("runs/phase0")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase0_report.md").write_text(md, encoding="utf-8")
    return answers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["env", "import", "replicate", "compat", "gold", "report", "all"])
    ap.add_argument("--config", default="config/phase0.yaml")
    ap.add_argument("--catalog")
    ap.add_argument("--split")
    ap.add_argument("--source", choices=["huggingface", "local_json"])
    ap.add_argument("--sample-size", type=int)
    ap.add_argument("--compat-run-id")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.verbose:  # the SQL connector logs every HTTP round trip at INFO
        logging.getLogger("databricks").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    cfg = load_config(Path(args.config), args)

    steps = ["env", "import", "replicate", "compat", "gold", "report"] if args.step == "all" else [args.step]
    for s in steps:
        log.info("=== step %s ===", s)
        if s == "env":
            res = step_env(cfg)
        elif s == "import":
            res = step_import(cfg)
        elif s == "replicate":
            res = step_replicate(cfg)
        elif s == "compat":
            res = step_compat(cfg)
        elif s == "gold":
            res = step_gold(cfg, args.compat_run_id)
        else:
            res = step_report()
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

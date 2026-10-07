"""Phase 1 CLI — few-shot baseline (one attempt per case, no retry).

    python scripts/phase1.py build-dev    # build the development set (MySQL gold, non-evaluation questions)
    python scripts/phase1.py dev          # iterate here: prompt / model / parameters
    python scripts/phase1.py baseline     # all PRIMARY evaluation cases, once per frozen configuration
    python scripts/phase1.py pilot        # fixed 20-case subset of the evaluation cases (historical, v1)
    python scripts/phase1.py dev --limit 3   # smoke test
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
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from agent.generator import FewShotGenerator, select_few_shot  # noqa: E402
from agent.llm import CachingChatClient, UsageMeter, make_client  # noqa: E402
from agent.retriever import BM25TableRetriever, SchemaCatalog  # noqa: E402
from benchmark.beaver.loader import load_cases, load_from_local_json  # noqa: E402
from evaluation.baseline import BaselineConfig, gold_judge, run_baseline, select_pilot  # noqa: E402
from evaluation.devset import build_devset, dev_cases, dev_judges, load_devset, save_devset  # noqa: E402

log = logging.getLogger("phase1")
RUNS = Path("runs/phase1")


def load_dotenv(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def connect(cfg: dict):
    from execution.databricks_sql import DatabricksSqlExecutor

    d = cfg["databricks"]
    return DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), http_path=d.get("http_path"),
                                 ansi_mode=bool(d["ansi_mode"]), statement_timeout_s=int(d["statement_timeout_s"]))


def load_gold(dbx, catalog: str) -> dict[str, str]:
    """PRIMARY frozen gold results (evaluation only), cached locally under runs/ (git-ignored)."""
    cache = RUNS / "gold_primary.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = dbx.run(f"SELECT case_id, gold_result FROM `{catalog}`.`benchmark`.`gold_results` "
                   "WHERE evaluation_eligibility = 'PRIMARY'")
    gold = {r[0]: r[1] for r in rows}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(gold), encoding="utf-8")
    return gold


def load_catalog(dbx, cfg: dict, tables_meta: dict) -> SchemaCatalog:
    cache = RUNS / f"schema_{cfg['beaver']['db']}.json"
    if cache.exists():
        return SchemaCatalog.from_json(cache.read_text(encoding="utf-8"))
    cat = SchemaCatalog.from_databricks(dbx, cfg["databricks"]["catalog"], cfg["beaver"]["db"], tables_meta)
    cat.save(cache)
    return cat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["build-dev", "dev", "pilot", "baseline"])
    ap.add_argument("--config", default="config/phase1.yaml")
    ap.add_argument("--few-shot", choices=["static", "dynamic"], help="override few_shot.mode of the config")
    ap.add_argument("--limit", type=int, help="only the first N selected cases (smoke test)")
    ap.add_argument("--no-cache", action="store_true", help="do not replay cached LLM responses")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("databricks").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    b, lc, fs = cfg["beaver"], cfg["llm"], cfg["few_shot"]
    if args.few_shot:
        fs["mode"] = args.few_shot

    queries, tables_meta = load_from_local_json(b["local_dir"], b["split"])
    cases, _, _ = load_cases("local_json", b["split"], int(b["sample_size"]), int(b["sample_seed"]), b["local_dir"])
    eval_raw_ids = {c.case_id.split(":", 1)[1] for c in cases}

    dbx = connect(cfg)
    examples = select_few_shot(queries, eval_raw_ids, int(fs["n_examples"]), int(fs["seed"]),
                               int(fs["max_tables"]), int(fs["max_sql_chars"]))
    dev_path = Path(cfg["dev"]["path"])

    if args.mode == "build-dev":
        from execution.mysql import MySqlExecutor

        mysql = MySqlExecutor.from_config({"host": "localhost", "port": 3306})
        exclude = eval_raw_ids | {e.source_id for e in examples}
        devset = build_devset(queries, exclude, int(cfg["dev"]["n_cases"]), int(cfg["dev"]["seed"]),
                              mysql, dbx, b["db"], int(cfg["dev"]["max_candidates"]))
        save_devset(devset, dev_path)
        mysql.close(); dbx.close()
        print(json.dumps({"n": devset["n"], "rejected": len(devset["rejected"]),
                          "reject_reasons": devset["rejected"]}, indent=2, ensure_ascii=False))
        return

    if args.mode == "dev":
        devset = load_devset(dev_path)
        overlap = {c["id"] for c in devset["cases"]} & (eval_raw_ids | {e.source_id for e in examples})
        if overlap:
            raise SystemExit(f"dev set overlaps evaluation sample / few-shot examples: {sorted(overlap)}")
        selected, judges = dev_cases(devset, b["split"]), dev_judges(devset, b["split"])
    else:
        gold = load_gold(dbx, cfg["databricks"]["catalog"])
        primary = [c for c in cases if c.case_id in gold]
        if args.mode == "pilot":
            keep = set(select_pilot([c.case_id for c in primary], int(cfg["pilot"]["n_cases"]), int(cfg["pilot"]["seed"])))
            selected = [c for c in primary if c.case_id in keep]
        else:
            selected = primary
        judges = {c.case_id: gold_judge(gold[c.case_id]) for c in selected}
    if args.limit:
        selected = selected[: args.limit]

    catalog = load_catalog(dbx, cfg, tables_meta)
    meter = UsageMeter(max_calls=int(lc["max_calls"]), max_tokens=int(lc["max_tokens"]))
    client = make_client(lc, max_output_tokens=int(lc["max_output_tokens"]), meter=meter)
    chat = client if args.no_cache else CachingChatClient(client, Path(lc["cache"]))
    from agent.examples import build_generator_index
    dev_ids = {str(c["id"]) for c in load_devset(dev_path)["cases"]} if dev_path.exists() else set()
    index = build_generator_index(queries, eval_raw_ids, dev_ids, fs)
    generator = FewShotGenerator(chat, catalog, examples, index=index, k=int(fs.get("dynamic_k", 4)),
                                 max_extra_tables=int(fs.get("dynamic_max_extra_tables", 6)))
    retriever = BM25TableRetriever(catalog)

    log.info("mode=%s cases=%d model=%s top_k=%s few_shot=%d", args.mode, len(selected), client.model,
             cfg["retrieval"]["top_k"], len(examples))
    run_id, summary = run_baseline(
        selected, judges, retriever, generator, dbx,
        BaselineConfig(top_k=int(cfg["retrieval"]["top_k"]), max_result_rows=int(cfg["databricks"]["max_result_rows"])),
        RUNS, {"mode": args.mode, "model": client.model, "provider": client.provider, "config": cfg, "case_ids": [c.case_id for c in selected]})
    summary["llm_usage"] = meter.snapshot()
    (RUNS / run_id / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    dbx.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

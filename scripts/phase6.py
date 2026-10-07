"""Phase 6 CLI — run the loop for one arm.

    python scripts/phase6.py --strategy targeted --verifier self          # dev set (default)
    python scripts/phase6.py --strategy generic  --verifier self
    python scripts/phase6.py --strategy targeted --verifier oracle        # upper bound
    python scripts/phase6.py --strategy targeted --verifier self --split eval --eval   # phase 7 only

Arms share attempt 1 (same prompt, replayed from the LLM cache).
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
from evaluation.baseline import gold_judge  # noqa: E402
from evaluation.devset import dev_cases, dev_judges, load_devset  # noqa: E402
from evaluation.loop_run import run_arm  # noqa: E402
from loop_engineer.controller import LoopConfig, LoopController  # noqa: E402
from loop_engineer.diagnose import DIAGNOSER_VERSION, Diagnoser  # noqa: E402
from loop_engineer.policy import Policy  # noqa: E402
from loop_engineer.verifier import SELF_SIGNALS, VERIFIER_VERSION, OracleVerifier, SelfVerifier  # noqa: E402
from skills.base import RepairContext  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["targeted", "generic"], default="targeted")
    ap.add_argument("--verifier", choices=["self", "oracle"], default="self")
    ap.add_argument("--policy", choices=["targeted", "generic"], default="targeted", help="repair policy (ablation)")
    ap.add_argument("--disable", default="", help="comma-separated skills to disable (ablation)")
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--eval", action="store_true", help="confirm a run on the evaluation set (phase 7)")
    ap.add_argument("--max-repairs", type=int, default=1, help="repair rounds per question (attempts = +1)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--config", default="config/phase1.yaml")
    args = ap.parse_args()
    if args.split == "eval" and not args.eval:
        raise SystemExit("evaluation-set runs are for phase 7 with a frozen configuration; add --eval to confirm")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("databricks", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for line in (Path(".env").read_text(encoding="utf-8") if Path(".env").exists() else "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    b, lc, fs, d = cfg["beaver"], cfg["llm"], cfg["few_shot"], cfg["databricks"]
    queries, _ = load_from_local_json(b["local_dir"], b["split"])
    eval_cases = load_cases("local_json", b["split"], int(b["sample_size"]), int(b["sample_seed"]), b["local_dir"])[0]
    eval_ids = {c.case_id.split(":", 1)[1] for c in eval_cases}
    examples = select_few_shot(queries, eval_ids, int(fs["n_examples"]), int(fs["seed"]), int(fs["max_tables"]),
                               int(fs["max_sql_chars"]))
    catalog = SchemaCatalog.from_json(Path(f"runs/phase1/schema_{b['db']}.json").read_text(encoding="utf-8"))

    from execution.databricks_sql import DatabricksSqlExecutor
    dbx = DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), ansi_mode=bool(d["ansi_mode"]),
                                statement_timeout_s=int(d["statement_timeout_s"]))
    if args.split == "dev":
        devset = load_devset(Path(cfg["dev"]["path"]))
        cases, judges = dev_cases(devset, b["split"]), dev_judges(devset, b["split"])
    else:
        gold = json.loads(Path("runs/phase1/gold_primary.json").read_text(encoding="utf-8"))
        cases = [c for c in eval_cases if c.case_id in gold]
        judges = {c.case_id: gold_judge(gold[c.case_id]) for c in cases}
    if args.limit:
        cases = cases[: args.limit]

    meter = UsageMeter(max_calls=int(lc["max_calls"]), max_tokens=int(lc["max_tokens"]))
    inner = make_client(lc, max_output_tokens=int(lc["max_output_tokens"]), meter=meter)
    client = CachingChatClient(inner, Path(lc["cache"]))
    policy = Policy(mode=args.policy, disabled={s for s in args.disable.split(",") if s})
    # 1. 生成器
    generator = FewShotGenerator(client, catalog, examples)
    # 2. 循环控制器
    controller = LoopController(BM25TableRetriever(catalog), generator, dbx,
                                Diagnoser(catalog, client), policy, RepairContext(catalog, client, examples),
                                LoopConfig(strategy=args.strategy, max_attempts=args.max_repairs + 1,
                                           top_k=int(cfg["retrieval"]["top_k"]),
                                           max_result_rows=int(d["max_result_rows"])))
    verifier_for = (lambda cid: SelfVerifier()) if args.verifier == "self" else (lambda cid: OracleVerifier(judges[cid]))
    arm = f"{args.strategy}-{args.verifier}" + (f"-policy_{args.policy}" if args.policy != "targeted" else "") + \
          (f"-no_{args.disable.replace(',', '_')}" if args.disable else "")
    meta = {"split": args.split, "strategy": args.strategy, "verifier": args.verifier,
            "upper_bound": args.verifier == "oracle", "policy": args.policy, "disabled": args.disable,
            "self_signals": list(SELF_SIGNALS), "verifier_version": VERIFIER_VERSION, "model": inner.model, "provider": inner.provider, "prompt_version": generator.prompt_version,
            "diagnoser": DIAGNOSER_VERSION, "max_attempts": args.max_repairs + 1,
            "case_ids": [c.case_id for c in cases]}
    
    # 运行入口
    run_id, summary = run_arm(cases, judges, controller, verifier_for, arm, Path("runs/phase6"), meta)
    summary["llm_usage"] = meter.snapshot()
    (Path("runs/phase6") / run_id / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                               encoding="utf-8")
    dbx.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

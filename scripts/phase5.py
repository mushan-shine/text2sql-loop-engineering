"""Phase 5 CLI — repair-skill evaluation on the failures of a DEV run.

For every failed attempt: observe -> diagnose -> policy -> skill -> execute ->
judge. Measures what each skill can do *given* a failure (the loop controller
and self-verification come in phase 6).

    python scripts/phase5.py runs/phase1/<dev_run_id>
    python scripts/phase5.py runs/phase1/<dev_run_id> --policy generic
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from agent.generator import select_few_shot  # noqa: E402
from agent.llm import CachingChatClient, UsageMeter, make_client  # noqa: E402
from agent.retriever import SchemaCatalog  # noqa: E402
from benchmark.beaver.loader import load_cases, load_from_local_json  # noqa: E402
from evaluation.devset import dev_judges, load_devset  # noqa: E402
from loop_engineer.diagnose import Diagnoser  # noqa: E402
from loop_engineer.observer import observe  # noqa: E402
from loop_engineer.policy import Policy  # noqa: E402
from skills.base import RepairContext  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--policy", default="targeted", choices=["targeted", "generic"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("databricks", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for line in (Path(".env").read_text(encoding="utf-8") if Path(".env").exists() else "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if meta.get("mode") != "dev":
        raise SystemExit("repair skills are tuned on the dev set only")
    cfg = meta["config"]
    b, lc, fs, d = cfg["beaver"], cfg["llm"], cfg["few_shot"], cfg["databricks"]
    recs = [json.loads(x) for x in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    catalog = SchemaCatalog.from_json(Path(f"runs/phase1/schema_{b['db']}.json").read_text(encoding="utf-8"))
    queries, _ = load_from_local_json(b["local_dir"], b["split"])
    eval_ids = {c.case_id.split(":", 1)[1] for c in load_cases("local_json", b["split"], int(b["sample_size"]),
                                                               int(b["sample_seed"]), b["local_dir"])[0]}
    examples = select_few_shot(queries, eval_ids, int(fs["n_examples"]), int(fs["seed"]), int(fs["max_tables"]),
                               int(fs["max_sql_chars"]))
    inner = make_client(model=meta["model"], max_output_tokens=int(lc["max_output_tokens"]),
                        meter=UsageMeter(max_calls=200))
    client = CachingChatClient(inner, Path(lc["cache"]))
    diagnoser, policy = Diagnoser(catalog, client), Policy(mode=args.policy)
    ctx = RepairContext(catalog, client, examples)
    judges = dev_judges(load_devset(Path(cfg["dev"]["path"])), b["split"])

    from execution.databricks_sql import DatabricksSqlExecutor
    dbx = DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), ansi_mode=bool(d["ansi_mode"]),
                                statement_timeout_s=int(d["statement_timeout_s"]))
    out = []
    for r in recs:
        if r["correct"]:
            continue
        obs = observe(r, b["db"])
        diag, _ = diagnoser.diagnose(obs)
        route = policy.route(diag)
        res = policy.skill(route.skill).repair(obs, diag, ctx)
        if res.repaired_sql:
            ex = dbx.execute(res.repaired_sql, b["db"], max_rows=int(d["max_result_rows"]))
            ok = judges[r["case_id"]](ex.rows)[0] if ex.ok else False
            status, err = ex.status, ex.error
        else:
            ok, status, err = False, res.parse_status, None
        out.append({"case_id": r["case_id"], "before_status": r["execution_status"], "diagnosis": diag.failure_type,
                    "hint_case": diag.repair_hints.get("case"), "skill": route.skill, "fallback_route": route.fallback,
                    "used_llm": res.used_llm, "repair_action": res.repair_action, "after_status": status,
                    "after_error": (err or "")[:300] or None, "correct_after": ok,
                    "tokens": res.input_tokens + res.output_tokens, "repaired_sql": res.repaired_sql})
        logging.info("%s %s/%s -> %s exec=%s correct=%s", r["case_id"], diag.failure_type, route.skill,
                     "llm" if res.used_llm else "det", status, ok)
    dbx.close()

    per = defaultdict(lambda: {"n": 0, "executable_after": 0, "correct_after": 0, "was_error": 0,
                               "error_fixed": 0, "llm_calls": 0})
    for x in out:
        p = per[x["skill"]]
        p["n"] += 1
        p["executable_after"] += x["after_status"] == "SUCCESS"
        p["correct_after"] += x["correct_after"]
        p["llm_calls"] += x["used_llm"]
        if x["before_status"] != "SUCCESS":
            p["was_error"] += 1
            p["error_fixed"] += x["after_status"] == "SUCCESS"
    summary = {"dev_run": meta["run_id"], "policy": args.policy, "failures": len(out),
               "executable_before": sum(x["before_status"] == "SUCCESS" for x in out),
               "executable_after": sum(x["after_status"] == "SUCCESS" for x in out),
               "correct_after": sum(x["correct_after"] for x in out),
               "per_skill": dict(per), "llm_usage": inner.meter.snapshot()}
    target = Path("runs/phase5") / f"repair-{args.policy}-{meta['run_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"summary": summary, "cases": out}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

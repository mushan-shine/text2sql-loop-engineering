"""Outer loop, one iteration: propose improvements and gate them on held-out questions.

    python scripts/outer_loop.py --train-n 200                 # mine on 200, gate on 200 held-out
    python scripts/outer_loop.py --train-n 60 --val-n 0        # gate on the 30-question dev set

Steps (evaluation/outer_loop.py): sample the training split -> run the current system -> judge against gold
and label failures -> mine candidate knowledge (contradicting table preferences resolved) -> per-item regression
on a held-out validation sample of the training split (disjoint from the mining sample), recommended items
re-checked together and on the dev set -> write proposals with a recommendation to
runs/outer_loop/<batch_id>/proposals.json for a reviewer. The evaluation set is never used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from agent.curated import CuratedKnowledge, attach_curated  # noqa: E402
from agent.examples import ExampleIndex  # noqa: E402
from agent.generator import FewShotGenerator, select_few_shot  # noqa: E402
from agent.knowledge import load_knowledge  # noqa: E402
from agent.llm import CachingChatClient, UsageMeter, make_client  # noqa: E402
from agent.retriever import BM25TableRetriever, SchemaCatalog  # noqa: E402
from benchmark.beaver.loader import load_cases, load_from_local_json  # noqa: E402
from evaluation import outer_loop as ol  # noqa: E402
from evaluation.devset import dev_cases, dev_judges, load_devset  # noqa: E402
from execution.databricks_sql import DatabricksSqlExecutor  # noqa: E402

log = logging.getLogger("outer_loop")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-n", type=int, default=40, help="training questions to run this iteration")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--val-n", type=int, default=200,
                    help="held-out training questions for the per-item gate (0 = gate on the 30-question dev set)")
    ap.add_argument("--val-seed", type=int, default=20261001)
    ap.add_argument("--min-support", type=int, default=2, help="failures needed before a pattern becomes a proposal")
    ap.add_argument("--model", default=None, help="default: config llm.model (glm-4-flash)")
    ap.add_argument("--workers", type=int, default=4)
    # an iteration is much larger than one phase-1 run (config llm.max_calls / max_tokens): its own safety budget
    ap.add_argument("--max-calls", type=int, default=1500, help="LLM call budget for the whole iteration")
    ap.add_argument("--max-tokens", type=int, default=20_000_000, help="LLM token budget for the whole iteration")
    ap.add_argument("--config", default="config/phase1.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("databricks", "urllib3", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for line in ((ROOT / ".env").read_text(encoding="utf-8") if (ROOT / ".env").exists() else "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    b, lc, fs, d = cfg["beaver"], cfg["llm"], cfg["few_shot"], cfg["databricks"]
    catalog = SchemaCatalog.from_json((ROOT / f"runs/phase1/schema_{b['db']}.json").read_text(encoding="utf-8"))
    schema = {t: {c.name.lower(): c.type for c in tb.columns} for t, tb in catalog.tables.items()}
    queries, _ = load_from_local_json(str(ROOT / b["local_dir"]), b["split"])
    eval_ids = {c.case_id.split(":", 1)[1] for c in load_cases("local_json", b["split"], int(b["sample_size"]),
                                                               int(b["sample_seed"]), str(ROOT / b["local_dir"]))[0]}
    devset = load_devset(ROOT / cfg["dev"]["path"])
    dev_ids = {str(c["id"]) for c in devset["cases"]}
    kb = load_knowledge(ROOT / cfg.get("knowledge", {}).get("path", "runs/knowledge/kb.json"))
    examples = select_few_shot(queries, eval_ids, int(fs["n_examples"]), int(fs["seed"]), int(fs["max_tables"]),
                               int(fs["max_sql_chars"]))

    dbx = DatabricksSqlExecutor(catalog=d["catalog"], profile=d.get("profile"), ansi_mode=bool(d["ansi_mode"]),
                                statement_timeout_s=int(d["statement_timeout_s"]))
    max_rows = int(d["max_result_rows"])
    meter = UsageMeter(max_calls=args.max_calls, max_tokens=args.max_tokens)
    inner = make_client(model=args.model, max_output_tokens=int(lc["max_output_tokens"]), meter=meter) if args.model \
        else make_client(lc, max_output_tokens=int(lc["max_output_tokens"]), meter=meter)
    client = CachingChatClient(inner, ROOT / lc["cache"])
    retriever = BM25TableRetriever(catalog)
    log.info("model=%s", inner.model)

    def generator(pool_exclude: set[str], extra: CuratedKnowledge | None = None) -> FewShotGenerator:
        g = FewShotGenerator(client, catalog, examples,
                             index=ExampleIndex.build(queries, pool_exclude, int(fs.get("dynamic_max_sql_chars", 2500))),
                             k=int(fs.get("dynamic_k", 4)), max_extra_tables=int(fs.get("dynamic_max_extra_tables", 6)),
                             knowledge=kb)
        return attach_curated(g, extra)

    # ① + ② mining sample: the sampled questions must not be their own few-shot examples
    top_k = int(cfg["retrieval"]["top_k"])
    sample = ol.sample_training(queries, eval_ids | dev_ids, args.train_n, args.seed)
    mine_ids = {str(q["id"]) for q in sample}
    train_cases, train_judges = ol.judged_cases(sample, dbx, b["split"], max_rows)
    log.info("mining sample: %d questions (%d with runnable gold)", len(sample), len(train_cases))
    train = ol.run_and_judge(train_cases, train_judges, retriever, generator(eval_ids | dev_ids | mine_ids), dbx,
                             schema, top_k, max_rows, args.workers)
    failures = [r for r in train if not r.correct]
    log.info("training: %s", ol.failure_summary(train))

    # ③ candidates from failures
    candidates = ol.mine_table_preferences(failures, kb, catalog, args.min_support) + \
        ol.mine_missing_tables(failures, kb, args.min_support) + \
        ol.mine_join_rules(failures, kb, args.min_support)
    candidates, dropped = ol.resolve_conflicts(candidates)
    for c in dropped:
        log.info("dropped conflicting candidate: %s (support %s)", c["title"], c["evidence"].get("support"))
    log.info("candidates: %d (%s)", len(candidates), dict(ol.Counter(c["kind"] for c in candidates)))

    # ④ regression gate, per item: current vs current + this one candidate (unchanged cases reused).
    #    Gate set = held-out training questions (disjoint from the mining sample, dev and eval), large enough
    #    to contain questions each candidate touches; the 30-question dev set re-checks the recommended ones.
    dcases, djudges = dev_cases(devset, b["split"]), dev_judges(devset, b["split"])
    if args.val_n:
        val = ol.sample_training(queries, eval_ids | dev_ids | mine_ids, args.val_n, args.val_seed)
        val_ids = {str(q["id"]) for q in val}
        gcases, gjudges = ol.judged_cases(val, dbx, b["split"], max_rows)
        gate_set = "val"
        log.info("validation set: %d questions (%d with runnable gold)", len(val), len(gcases))
    else:
        val_ids, gcases, gjudges, gate_set = set(), dcases, djudges, "dev"
    exclude = eval_ids | dev_ids | mine_ids | val_ids          # no gate question is its own example

    def gate_run(cases, judges, extra=None, reuse=None):
        return ol.run_and_judge(cases, judges, retriever, generator(exclude, extra), dbx, schema,
                                top_k, max_rows, args.workers, reuse=reuse)

    before = gate_run(gcases, gjudges)
    reuse = {r.case.case_id: r for r in before}
    per_item, combined = ol.gate_candidates(candidates, before, lambda extra: gate_run(gcases, gjudges, extra, reuse))
    for c, reg in zip(candidates, per_item):
        reg["gate_set"] = gate_set
        log.info("gate %-16s %-70s affected=%d %d->%d fixed=%s harmed=%s passed=%s", c["kind"], c["title"][:70],
                 len(reg["affected"]), reg["correct_before"], reg["correct_after"], reg["fixed"], reg["harmed"],
                 reg["gate_passed"])
    good = [c for c, reg in zip(candidates, per_item) if reg["gate_passed"] and reg["fixed"]]
    dev_check = None
    if good and gate_set == "val":                             # the dev set re-checks what would be recommended
        dev_before = gate_run(dcases, djudges)
        dev_check = {**ol.compare(dev_before, gate_run(dcases, djudges, ol.as_curated(good),
                                                       {r.case.case_id: r for r in dev_before})),
                     "items": len(good), "gate_set": "dev"}
        log.info("dev check of %d recommended items: %d->%d harmed=%s", len(good), dev_check["correct_before"],
                 dev_check["correct_after"], dev_check["harmed"])
    batch = {"model": inner.model, "outer_loop": ol.OUTER_LOOP_VERSION, "train": ol.failure_summary(train),
             "gate_set": gate_set, "gate_correct": sum(r.correct for r in before), "gate_cases": len(before),
             "combined": {**combined, "gate_set": gate_set} if combined else None, "dev_check": dev_check,
             "dropped_conflicts": [c["title"] for c in dropped]}

    # ⑤ proposals for a reviewer
    batch_id = f"batch-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    rows = ol.proposal_rows(batch_id, candidates, per_item, batch)
    out = ROOT / "runs" / "outer_loop" / batch_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposals.json").write_text(json.dumps({"batch": batch, "per_item": per_item, "candidates": candidates,
                                                    "train_failures": ol.failure_rows(failures, kb, catalog)},
                                                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    dbx.close()
    print(json.dumps({"batch_id": batch_id, "proposals": len(rows),
                      "gate": f"{gate_set} {batch['gate_correct']}/{batch['gate_cases']}",
                      "dropped_conflicts": batch["dropped_conflicts"],
                      "items": [{"title": r["title"], "recommendation": r["recommendation"]} for r in rows],
                      **{name: {k: chk[k] for k in ("correct_before", "correct_after", "fixed", "harmed",
                                                    "gate_passed")} if chk else None
                         for name, chk in (("combined", combined), ("dev_check", dev_check))},
                      "llm_usage": meter.snapshot(), "local": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

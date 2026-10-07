"""Outer loop, step 1: mine warehouse usage knowledge from solved queries (agent/knowledge.py).

    python scripts/build_knowledge.py            # -> runs/knowledge/kb.json

Training split only: every dw question with gold SQL EXCEPT the evaluation sample and the dev set, so the
dev set keeps measuring what the knowledge does for unseen questions (decision D2). No LLM calls.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from agent.knowledge import WarehouseKnowledge  # noqa: E402
from agent.retriever import SchemaCatalog  # noqa: E402
from benchmark.beaver.loader import load_cases, load_from_local_json  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config/phase1.yaml").read_text(encoding="utf-8"))
    b = cfg["beaver"]
    catalog = SchemaCatalog.from_json((ROOT / f"runs/phase1/schema_{b['db']}.json").read_text(encoding="utf-8"))
    queries, _ = load_from_local_json(str(ROOT / b["local_dir"]), b["split"])
    eval_ids = {c.case_id.split(":", 1)[1] for c in load_cases("local_json", b["split"], int(b["sample_size"]),
                                                               int(b["sample_seed"]), str(ROOT / b["local_dir"]))[0]}
    dev_ids = {str(c["id"]) for c in json.loads((ROOT / cfg["dev"]["path"]).read_text(encoding="utf-8"))["cases"]}
    t = time.time()
    kb = WarehouseKnowledge.build(queries, eval_ids | dev_ids, catalog)
    out = ROOT / cfg.get("knowledge", {}).get("path", "runs/knowledge/kb.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(kb.to_json(), encoding="utf-8")
    print(f"{kb.version}: {kb.n_queries} solved queries, {len(kb.word_df)} words, {len(kb.joins)} join pairs, "
          f"{len(kb.groups)} look-alike groups in {time.time() - t:.0f}s -> {out}")
    for g in kb.groups:
        print("  look-alike:", ", ".join(g))


if __name__ == "__main__":
    main()

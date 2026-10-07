"""Build app/bundle/ — the data the console's "Run loop" page needs.

    python scripts/build_app_bundle.py

Writes schema_dw.json (agent-visible schema), few_shot.json (examples drawn from
non-evaluation questions, same selection as the CLI), devset.json (dev questions +
MySQL gold rows, used only to judge AFTER the loop), config.json (phase-1 config)
and a copy of the LLM cache so replays of recorded prompts are instant.

app/bundle/ holds BEAVER content (gated dataset): it is git-ignored and only
uploaded to the user's own Databricks workspace by scripts/deploy_app.py.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.generator import select_few_shot  # noqa: E402
from benchmark.beaver.loader import load_cases, load_from_local_json  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config/phase1.yaml").read_text(encoding="utf-8"))
    b, fs = cfg["beaver"], cfg["few_shot"]
    out = ROOT / "app" / "bundle"
    out.mkdir(parents=True, exist_ok=True)

    queries, _ = load_from_local_json(str(ROOT / b["local_dir"]), b["split"])
    eval_cases = load_cases("local_json", b["split"], int(b["sample_size"]), int(b["sample_seed"]),
                            str(ROOT / b["local_dir"]))[0]
    eval_ids = {c.case_id.split(":", 1)[1] for c in eval_cases}
    examples = select_few_shot(queries, eval_ids, int(fs["n_examples"]), int(fs["seed"]), int(fs["max_tables"]),
                               int(fs["max_sql_chars"]))
    (out / "few_shot.json").write_text(json.dumps([asdict(e) for e in examples], ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    # dynamic few-shot pool: solved questions minus the evaluation sample and the dev set (agent/examples.py)
    from agent.examples import ExampleIndex
    dev_ids = {str(c["id"]) for c in json.loads((ROOT / cfg["dev"]["path"]).read_text(encoding="utf-8"))["cases"]}
    pool = ExampleIndex.build(queries, eval_ids | dev_ids, int(fs.get("dynamic_max_sql_chars", 2500)))
    import gzip  # ~10 MB as JSON -> gzip keeps it well under the workspace upload limit
    (out / "examples_pool.json").unlink(missing_ok=True)
    with gzip.open(out / "examples_pool.json.gz", "wt", encoding="utf-8") as f:
        json.dump(pool.to_rows(), f, ensure_ascii=False)
    kb = ROOT / cfg.get("knowledge", {}).get("path", "runs/knowledge/kb.json")   # outer-loop usage notes
    if kb.exists():
        shutil.copy(kb, out / "kb.json")
    shutil.copy(ROOT / f"runs/phase1/schema_{b['db']}.json", out / f"schema_{b['db']}.json")
    shutil.copy(ROOT / cfg["dev"]["path"], out / "devset.json")
    cache = ROOT / cfg["llm"]["cache"]
    if cache.exists():
        shutil.copy(cache, out / "llm_cache.jsonl")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in sorted(out.iterdir()):
        print(f"{p.name:22s} {p.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()

"""Development set for Phase 1+ iteration — never the evaluation sample.

Prompt changes, model choices and parameters are tuned on this set only; the
89 PRIMARY evaluation cases are run once per frozen configuration.

Construction (deterministic):
* pool = dw questions minus the evaluation sample (seed 77) minus the few-shot
  examples used in prompts;
* candidates are visited in ``random.Random(seed)`` order;
* a candidate is kept only if its gold SQL runs on MySQL (BEAVER's engine) with
  a non-empty result, and the phase-0-adapted gold SQL reproduces that result
  on Databricks under the cross-engine rule — the same qualification as phase 0.

The MySQL gold rows are the dev ground truth; generated SQL (run on
Databricks) is judged against them with ``cross_engine_match``.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

from benchmark.beaver.adapter import apply_rules
from benchmark.beaver.dataset import BeaverCase
from benchmark.beaver.evaluator import cross_engine_match

log = logging.getLogger(__name__)


# ---- lossless JSON for MySQL values (Decimal scale matters to the cross-engine rule)

def encode_value(v: Any) -> Any:
    if isinstance(v, Decimal):
        return {"__decimal__": str(v)}
    if isinstance(v, dt.datetime):
        return {"__datetime__": v.isoformat()}
    if isinstance(v, dt.date):
        return {"__date__": v.isoformat()}
    if isinstance(v, dt.timedelta):
        return {"__timedelta__": v.total_seconds()}
    if isinstance(v, (bytes, bytearray)):
        return {"__bytes__": bytes(v).hex()}
    return v


def decode_value(v: Any) -> Any:
    if isinstance(v, dict) and len(v) == 1:
        (k, x), = v.items()
        return {"__decimal__": Decimal, "__datetime__": dt.datetime.fromisoformat,
                "__date__": dt.date.fromisoformat, "__timedelta__": lambda s: dt.timedelta(seconds=s),
                "__bytes__": bytes.fromhex}[k](x)
    return v


def encode_rows(rows: list[tuple]) -> list[list]:
    return [[encode_value(v) for v in r] for r in rows]


def decode_rows(rows: list[list]) -> list[tuple]:
    return [tuple(decode_value(v) for v in r) for r in rows]


def build_devset(queries: list[dict], exclude_ids: set[str], n: int, seed: int, mysql: Any, dbx: Any,
                 db: str = "dw", max_candidates: int = 200) -> dict[str, Any]:
    pool = sorted((e for e in queries if str(e["id"]) not in exclude_ids), key=lambda e: str(e["id"]))
    random.Random(seed).shuffle(pool)
    kept, rejected = [], {}
    for e in pool[:max_candidates]:
        if len(kept) >= n:
            break
        ref = mysql.execute(e["sql"], db)
        if not ref.ok or not ref.rows:
            rejected[str(e["id"])] = f"mysql: {ref.status if not ref.ok else 'empty result'}"
            continue
        adapted, rules = apply_rules(e["sql"])
        cand = dbx.execute(adapted, db, max_rows=500_000)
        if not cand.ok:
            rejected[str(e["id"])] = f"databricks: {cand.status} {cand.error_class or ''}"
            continue
        cmp = cross_engine_match(cand.rows, ref.rows)
        if not cmp.set_match:
            rejected[str(e["id"])] = f"not reproducible on databricks ({cmp.reason})"
            continue
        kept.append({**{k: e.get(k) for k in ("id", "question", "db", "sql", "tables", "category")},
                     "adapter_rules": rules, "mysql_rows": encode_rows(ref.rows)})
        log.info("dev %d/%d: %s", len(kept), n, e["id"])
    return {"seed": seed, "n": len(kept), "excluded_ids": sorted(exclude_ids), "cases": kept,
            "rejected": rejected}


def save_devset(devset: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(devset, ensure_ascii=False, default=str), encoding="utf-8")


def load_devset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dev_cases(devset: dict, split: str = "dw") -> list[BeaverCase]:
    return [BeaverCase.from_beaver({**c, "id": c["id"]}, split) for c in devset["cases"]]


def dev_judges(devset: dict, split: str = "dw"):
    """case_id -> judge(rows) -> (correct, message) against the MySQL gold rows."""
    judges = {}
    for c in devset["cases"]:
        ref = decode_rows(c["mysql_rows"])

        def judge(rows, ref=ref):
            cmp = cross_engine_match(rows, ref)
            return cmp.set_match, cmp.reason
        judges[f"{split}:{c['id']}"] = judge
    return judges

"""Load BEAVER queries and table metadata.

Mirrors BEAVER's official ``data/download_hf.py`` (field mapping and seed-77
sampling) so that the phase-0 sample equals the official ``dev_sampled.json``.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from benchmark.beaver.dataset import BeaverCase, _parse

log = logging.getLogger(__name__)

HF_QUERY_REPO = "beaverbench/beaver-query"
HF_TABLE_REPO = "beaverbench/beaver-table"


def sample_official(entries: list[dict[str, Any]], sample_size: int, seed: int = 77) -> list[dict[str, Any]]:
    """Exactly BEAVER's sampling: ``random.seed(77); random.sample(queries, n)``."""
    random.seed(seed)
    return random.sample(entries, min(sample_size, len(entries)))


def load_from_huggingface(split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (raw query entries, tables keyed by table name) for one split.

    The HF datasets are gated: accept the terms on huggingface.co and run
    ``hf auth login`` once before calling this.
    """
    from datasets import load_dataset  # heavy import; only needed here

    query_ds = load_dataset(HF_QUERY_REPO)
    table_ds = load_dataset(HF_TABLE_REPO)
    if split not in query_ds:
        raise KeyError(f"split {split!r} not in {HF_QUERY_REPO}; available: {list(query_ds.keys())}")
    queries = [_map_query(e) for e in query_ds[split]]
    table_split = "dw" if split == "dw_real" else split  # as in download_hf.py
    tables = {}
    for e in table_ds[table_split]:
        tables[e["table_name"]] = {
            "db": e.get("db"),
            "table_name": e["table_name"],
            "column_names": _parse(e.get("column_names"), list),
            "column_types": _parse(e.get("column_types"), list),
            "example_rows": _parse(e.get("example_rows"), list),
            "example_columns": _parse(e.get("example_columns"), list),
        }
    log.info("loaded %d queries / %d tables for split %s", len(queries), len(tables), split)
    return queries, tables


def load_from_local_json(root: str | Path, split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read ``<root>/<split>/dev.json`` and ``dev_tables.json`` (download_hf.py layout)."""
    d = Path(root) / split
    queries = json.loads((d / "dev.json").read_text(encoding="utf-8"))
    tables = json.loads((d / "dev_tables.json").read_text(encoding="utf-8"))
    return queries, tables


def load_cases(
    source: str, split: str, sample_size: int, seed: int = 77, local_dir: str | Path = "data/beaver"
) -> tuple[list[BeaverCase], dict[str, dict[str, Any]], int]:
    """Return (sampled cases, table metadata, size of the full split)."""
    if source == "huggingface":
        queries, tables = load_from_huggingface(split)
    elif source == "local_json":
        queries, tables = load_from_local_json(local_dir, split)
    else:
        raise ValueError(f"unknown BEAVER source {source!r}")
    sampled = sample_official(queries, sample_size, seed)
    cases = [BeaverCase.from_beaver(e, split) for e in sampled]
    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate BEAVER ids in sample — refusing to continue")
    return cases, tables, len(queries)


def save_local_snapshot(root: str | Path, split: str, queries: list[dict], tables: dict) -> Path:
    """Persist the raw download (read-only reference copy, never edited)."""
    d = Path(root) / split
    d.mkdir(parents=True, exist_ok=True)
    (d / "dev.json").write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "dev_tables.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def _map_query(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": e.get("id"),
        "question": e.get("question"),
        "db": e.get("db"),
        "sql": e.get("sql"),
        "tables": _parse(e.get("tables"), list),
        "column_mapping": _parse(e.get("column_mapping"), dict),
        "join_keys": _parse(e.get("join_keys"), list),
        "domain_knowledge": _parse(e.get("domain_knowledge"), list),
        "sub_questions": _parse(e.get("sub_questions"), list),
        "sub_sqls": _parse(e.get("sub_sqls"), list),
        "category": e.get("category"),
        "detailed_category": e.get("detailed_category"),
        "contains_domain_knowledge": e.get("contains_domain_knowledge"),
    }

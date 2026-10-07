"""Schema catalog and table retrieval (BEAVER setting=0).

The catalog is built only from agent-visible material:
* column names and **Databricks** types of the replicated database
  (``information_schema.columns``; the beaver-table metadata carries the
  original Oracle types such as VARCHAR2, which do not match the engine);
* example column values from beaver-table (``example_columns``).

No gold annotation (gold tables, join keys, column mapping, domain knowledge)
is read here.

Retrieval is lexical BM25 over table names, column names and example values.
A dw database has 97 tables / 1,530 columns — too much to put in every prompt,
and choosing the right tables is itself a failure mode the loop must handle
(TABLE_RETRIEVAL_FAILURE), so the baseline retrieves top-k tables.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by each for from give has have how i in include is it its list me of on or per "
    "provide show that the their them these they this those to what when where which who whose with within "
    "all any also along both do does find get number total average count than there".split())
# Name tokens count more than example values: a question names concepts, not data.
FIELD_WEIGHTS = {"table": 3, "column": 2, "value": 1}


def tokenize(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower().replace("_", " ")):
        if w in STOPWORDS or len(w) < 2:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]

    def document(self) -> list[str]:
        toks: list[str] = []
        toks += tokenize(self.name) * FIELD_WEIGHTS["table"]
        for c in self.columns:
            toks += tokenize(c.name) * FIELD_WEIGHTS["column"]
            for v in c.examples:
                toks += tokenize(v)[:6] * FIELD_WEIGHTS["value"]
        return toks


@dataclass
class SchemaCatalog:
    db: str
    tables: dict[str, Table]

    @classmethod
    def build(cls, db: str, columns: list[tuple[str, str, str]], tables_meta: dict[str, dict[str, Any]],
              examples_per_column: int = 3) -> "SchemaCatalog":
        """``columns``: (table, column, databricks_type) rows in ordinal order."""
        examples: dict[tuple[str, str], tuple[str, ...]] = {}
        for tname, meta in tables_meta.items():
            if meta.get("db") not in (None, db):
                continue
            for cname, vals in zip(meta.get("column_names") or [], meta.get("example_columns") or []):
                vs = [str(v) for v in (vals or []) if v is not None and str(v).strip()][:examples_per_column]
                examples[(tname.lower(), str(cname).lower())] = tuple(v[:40] for v in vs)
        by_table: dict[str, list[Column]] = {}
        for t, c, typ in columns:
            by_table.setdefault(t.lower(), []).append(Column(c, typ.upper(), examples.get((t.lower(), c.lower()), ())))
        return cls(db, {t: Table(t, tuple(cols)) for t, cols in by_table.items()})

    @classmethod
    def from_databricks(cls, runner: Any, catalog: str, db: str, tables_meta: dict[str, dict]) -> "SchemaCatalog":
        rows = runner.run(
            f"SELECT table_name, column_name, full_data_type FROM `{catalog}`.information_schema.columns "
            f"WHERE table_schema = '{db}' ORDER BY table_name, ordinal_position")
        cols = [(r[0], r[1], re.sub(r"\s+COLLATE\s+\w+", "", str(r[2]), flags=re.I)) for r in rows]
        return cls.build(db, cols, tables_meta)

    def to_json(self) -> str:
        return json.dumps({"db": self.db, "tables": {
            t: [[c.name, c.type, list(c.examples)] for c in tb.columns] for t, tb in self.tables.items()}},
            ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "SchemaCatalog":
        d = json.loads(text)
        return cls(d["db"], {t: Table(t, tuple(Column(c[0], c[1], tuple(c[2])) for c in cols))
                             for t, cols in d["tables"].items()})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")


@dataclass(frozen=True)
class Retrieval:
    tables: tuple[str, ...]
    scores: tuple[float, ...]


@dataclass
class BM25TableRetriever:
    catalog: SchemaCatalog
    k1: float = 1.2
    b: float = 0.75
    _docs: dict[str, Counter] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._docs = {t: Counter(tb.document()) for t, tb in self.catalog.tables.items()}
        self._len = {t: sum(c.values()) for t, c in self._docs.items()}
        self._avg = sum(self._len.values()) / max(1, len(self._len))
        df: Counter = Counter()
        for c in self._docs.values():
            df.update(c.keys())
        n = len(self._docs)
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def score(self, question: str) -> dict[str, float]:
        q = Counter(tokenize(question))
        out = {}
        for t, doc in self._docs.items():
            s = 0.0
            for w in q:
                tf = doc.get(w, 0)
                if tf:
                    s += self._idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self._len[t] / self._avg))
            out[t] = s
        return out

    def retrieve(self, question: str, k: int = 10) -> Retrieval:
        ranked = sorted(self.score(question).items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return Retrieval(tuple(t for t, _ in ranked), tuple(round(s, 4) for _, s in ranked))

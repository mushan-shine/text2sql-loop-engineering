"""Dynamic few-shot: retrieve solved questions similar to the new one (a "query log" of the warehouse).

Error analysis of a deepseek-flash dev run (scripts/analyze_run.py) showed that 22/26 executed-but-wrong
answers used different tables than the gold SQL, and 34 of 41 such missing tables WERE retrieved — the model
picked a plausible look-alike table (the warehouse has several near-duplicates, e.g. subject_offered /
subject_offered_summary / tip_subject_offered). Solved questions that resemble the new one show which tables
and joins this warehouse uses for that kind of question: the tables of the 5 most similar ones cover 88.5% of
the gold tables of the dev questions on average.

The pool is every dw question with gold SQL EXCEPT the evaluation sample and the dev set (so dev runs measure
generalisation the way evaluation runs will). SQL is BEAVER's MySQL gold, adapted with the documented phase-0
rules; examples whose SQL does not parse as Databricks SQL or is very long are skipped.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import sqlglot

from agent.generator import FewShotExample
from agent.retriever import tokenize
from agent.sql_analysis import alias_map
from benchmark.beaver.adapter import apply_rules


def build_generator_index(queries: list[dict], eval_ids: set[str], dev_ids: set[str], fs_cfg: dict):
    """The example pool for dynamic few-shot, or None in static mode. Always excludes the evaluation
    sample AND the dev set, so dev runs measure what evaluation runs will get."""
    if fs_cfg.get("mode", "static") != "dynamic":
        return None
    return ExampleIndex.build(queries, set(eval_ids) | set(dev_ids), int(fs_cfg.get("dynamic_max_sql_chars", 2500)))


@dataclass(frozen=True)
class PoolEntry:
    example: FewShotExample
    tables: tuple[str, ...]


@dataclass
class ExampleIndex:
    entries: list[PoolEntry]
    k1: float = 1.2
    b: float = 0.75
    _docs: list[list[str]] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avgdl: float = 1.0

    def __post_init__(self) -> None:
        self._docs = [tokenize(e.example.question) for e in self.entries]
        self._df = Counter(t for d in self._docs for t in set(d))
        self._avgdl = (sum(map(len, self._docs)) / len(self._docs)) if self._docs else 1.0

    @classmethod
    def build(cls, queries: list[dict], exclude_ids: set[str], max_sql_chars: int = 2500) -> "ExampleIndex":
        entries = []
        for q in queries:
            if str(q["id"]) in exclude_ids or not q.get("sql"):
                continue
            sql, _ = apply_rules(q["sql"].strip().rstrip(";"))
            if len(sql) > max_sql_chars:
                continue
            try:
                sqlglot.parse_one(sql, read="databricks")
            except sqlglot.errors.ParseError:
                continue
            tables = tuple(dict.fromkeys(alias_map(sql).values()))
            entries.append(PoolEntry(FewShotExample(q["question"].strip(), sql, str(q["id"])), tables))
        return cls(entries)

    def to_rows(self) -> list[dict]:
        return [{"id": e.example.source_id, "question": e.example.question, "sql": e.example.sql,
                 "tables": list(e.tables)} for e in self.entries]

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "ExampleIndex":
        return cls([PoolEntry(FewShotExample(r["question"], r["sql"], str(r["id"])), tuple(r["tables"])) for r in rows])

    def _score(self, qt: list[str], i: int) -> float:
        d, n = self._docs[i], len(self._docs)
        tf, s = Counter(d), 0.0
        for t in set(qt):
            if t in tf:
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self._avgdl))
        return s

    def top(self, question: str, k: int = 4) -> list[PoolEntry]:
        qt = tokenize(question)
        scored = sorted(range(len(self.entries)), key=lambda i: (-self._score(qt, i), self.entries[i].example.source_id))
        return [self.entries[i] for i in scored[:k]]

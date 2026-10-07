"""Warehouse usage knowledge — the outer loop's first knowledge asset, mined from solved queries.

Error analysis of the deepseek-flash dev run (EXECUTION_LOG G.9): 12 of 21 executed-but-wrong answers came
from not knowing this warehouse's conventions — WHICH table holds a concept among look-alike tables (8) and
HOW tables are joined (key, INNER vs LEFT; 4). Such conventions can be counted from solved questions:

* concept -> tables: for each question word, the tables its solved questions used (P(table | word));
* look-alike tables: tables whose column sets overlap strongly form a group; for a new question the
  word-weighted usage shares inside the group tell the model which one is conventional;
* join conventions: for each pair of tables, the join keys and join kinds (INNER / LEFT) used.

Built OFFLINE from the training split only (every dw question with gold SQL minus the evaluation sample and
the dev set). At run time only this aggregated knowledge is read — nothing gold-derived about the question
being answered. ``notes_for`` renders a short, question-specific "usage notes" block for the prompt.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from agent.retriever import SchemaCatalog, tokenize
from benchmark.beaver.adapter import apply_rules

KB_VERSION = "kb-v1"


def _scope_joins(sql: str, tables: set[str]) -> tuple[set[str], list[tuple[str, str, str, str, str]]]:
    """Base tables used and join edges (t1, c1, t2, c2, kind) of a query, aliases resolved per scope."""
    try:
        tree = sqlglot.parse_one(sql, read="databricks")
        scopes = traverse_scope(tree)
    except Exception:
        return set(), []
    used, edges = set(), []
    for sc in scopes:
        aliases = {a.lower(): s.name.lower() for a, s in sc.sources.items()
                   if isinstance(s, exp.Table) and s.name.lower() in tables}
        used |= set(aliases.values())
        node = sc.expression
        if not isinstance(node, exp.Select):
            continue
        for j in node.args.get("joins") or []:
            on = j.args.get("on")
            if on is None:
                continue
            kind = "LEFT" if (j.args.get("side") or "").upper() == "LEFT" else "INNER"
            for eq in on.find_all(exp.EQ):
                l, r = eq.left, eq.right
                if isinstance(l, exp.Column) and isinstance(r, exp.Column) and l.table and r.table:
                    t1, t2 = aliases.get(l.table.lower()), aliases.get(r.table.lower())
                    if t1 and t2 and t1 != t2:
                        edges.append((t1, l.name.upper(), t2, r.name.upper(), kind))
    return used, edges


@dataclass
class WarehouseKnowledge:
    n_queries: int = 0
    table_freq: dict[str, int] = field(default_factory=dict)
    word_df: dict[str, int] = field(default_factory=dict)                    # questions containing the word
    word_tables: dict[str, dict[str, int]] = field(default_factory=dict)     # word -> table -> questions
    groups: list[list[str]] = field(default_factory=list)                    # look-alike tables
    joins: dict[str, dict[str, Any]] = field(default_factory=dict)           # "a|b" -> {"keys": {..}, "kinds": {..}, "n"}
    version: str = KB_VERSION

    # ---------------------------------------------------------------- build (offline, training split)
    @classmethod
    def build(cls, queries: list[dict], exclude_ids: set[str], catalog: SchemaCatalog,
              min_word_df: int = 5, group_jaccard: float = 0.5) -> "WarehouseKnowledge":
        tables = set(catalog.tables)
        kb = cls()
        tf, wdf, wt = Counter(), Counter(), defaultdict(Counter)
        jn: dict[str, dict[str, Any]] = {}
        for q in queries:
            if str(q["id"]) in exclude_ids or not q.get("sql"):
                continue
            sql, _ = apply_rules(q["sql"].strip().rstrip(";"))
            used, edges = _scope_joins(sql, tables)
            if not used:
                continue
            kb.n_queries += 1
            tf.update(used)
            for w in set(tokenize(q["question"])):
                wdf[w] += 1
                wt[w].update(used)
            seen = set()
            for t1, c1, t2, c2, kind in edges:
                (a, ca), (b, cb) = sorted([(t1, c1), (t2, c2)])
                key = f"{a}|{b}"
                e = jn.setdefault(key, {"keys": Counter(), "kinds": Counter(), "n": 0})
                e["keys"][f"{a}.{ca} = {b}.{cb}"] += 1
                e["kinds"][kind] += 1
                if key not in seen:
                    e["n"] += 1
                    seen.add(key)
        kb.table_freq = dict(tf)
        kb.word_df = {w: n for w, n in wdf.items() if n >= min_word_df}
        kb.word_tables = {w: dict(wt[w]) for w in kb.word_df}
        kb.joins = {k: {"keys": dict(v["keys"]), "kinds": dict(v["kinds"]), "n": v["n"]} for k, v in jn.items()}
        cols = {t: {c.name.upper() for c in tb.columns} for t, tb in catalog.tables.items()}
        names = sorted(t for t in tables if tf.get(t))
        parent = {t: t for t in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter, union = len(cols[a] & cols[b]), len(cols[a] | cols[b])
                if union and inter / union >= group_jaccard and inter >= 4:
                    parent[find(a)] = find(b)
        grp = defaultdict(list)
        for t in names:
            grp[find(t)].append(t)
        kb.groups = sorted([sorted(g) for g in grp.values() if len(g) > 1], key=lambda g: -len(g))
        return kb

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "WarehouseKnowledge":
        return cls(**json.loads(text))

    # ---------------------------------------------------------------- use (run time)
    def table_scores(self, question: str) -> dict[str, float]:
        """Word-weighted share of solved questions using each table: sum_w idf(w) * P(table | w)."""
        scores: Counter = Counter()
        for w in set(tokenize(question)):
            df = self.word_df.get(w)
            if not df or df > 0.5 * self.n_queries:
                continue
            idf = math.log(self.n_queries / df)
            for t, n in self.word_tables[w].items():
                scores[t] += idf * n / df
        return dict(scores)

    def notes_for(self, question: str, schema_tables: list[str] | tuple[str, ...], max_tables: int = 6,
                  max_joins: int = 8) -> str:
        """Short usage notes for one question (empty string if nothing useful)."""
        scores = self.table_scores(question)
        if not scores:
            return ""
        shown = set(schema_tables)
        ranked = [t for t, _ in sorted(scores.items(), key=lambda kv: -kv[1]) if t in shown][:max_tables]
        if not ranked:
            return ""
        top = scores[ranked[0]]
        lines = [f"Warehouse usage notes (counted from {self.n_queries} solved questions of this warehouse; "
                 "conventions, not rules):",
                 "- Tables that solved questions with similar wording use most (strongest first): "
                 + ", ".join(f"{t} ({scores[t] / top:.0%})" for t in ranked)]
        for g in self.groups:
            members = [t for t in g if t in shown]
            if len(members) >= 2 and any(t in ranked for t in members):
                tot = sum(scores.get(t, 0.0) for t in members) or 1.0
                share = ", ".join(f"{t} {scores.get(t, 0.0) / tot:.0%}"
                                  for t in sorted(members, key=lambda t: -scores.get(t, 0.0)))
                lines.append(f"- Look-alike tables (similar columns) - for this kind of question solved queries used: {share}")
        pairs = []
        for i, a in enumerate(ranked):
            for b in ranked[i + 1:]:
                key = "|".join(sorted([a, b]))
                if key in self.joins:
                    pairs.append((self.joins[key]["n"], key))
        for n, key in sorted(pairs, reverse=True)[:max_joins]:
            e = self.joins[key]
            k, kn = max(e["keys"].items(), key=lambda kv: kv[1])
            kinds = e["kinds"]
            total = sum(kinds.values()) or 1
            left = kinds.get("LEFT", 0) / total
            kind = "LEFT JOIN" if left >= 0.5 else "INNER JOIN"
            pair = key.replace("|", " - ")
            lines.append(f"- Join {pair}: usually {k} ({kn}/{sum(e['keys'].values())} of "
                         f"{n} queries), {kind} ({max(left, 1 - left):.0%})")
        return "\n".join(lines)


def load_knowledge(path: Path) -> WarehouseKnowledge | None:
    return WarehouseKnowledge.from_json(path.read_text(encoding="utf-8")) if path.exists() else None


def knowledge_for(cfg: dict, root: Path) -> WarehouseKnowledge | None:
    """The knowledge to use per config (``knowledge.mode: on``), or None."""
    kc = cfg.get("knowledge") or {}
    if str(kc.get("mode", "off")).lower() not in ("on", "true", "1"):
        return None
    kb = load_knowledge(root / kc.get("path", "runs/knowledge/kb.json"))
    if kb is None:
        raise FileNotFoundError("knowledge file missing; run scripts/build_knowledge.py first")
    return kb

"""Tool-grounded validators — a validation sub-agent that checks an executed SQL against the DATABASE.

The LLM judge only read the SQL and gave opinions (unreliable on BEAVER: glm missed most errors, deepseek
flagged 31/33 correct answers). These validators instead gather evidence with small probe queries on the
warehouse; nothing is gold-derived and no LLM is involved:

* ``filter_value_not_found`` (schema validator) — a filter ``col = 'v'`` / ``col IN ('v', ...)`` whose value
  does not occur in that column at all (e.g. 'Engineering' where the column holds 'School of Engineering').
  Evidence: the value, the column, and the closest values that do exist.
* ``join_fanout_aggregate`` (logic validator) — SUM / AVG / COUNT(col) over a column of table T while T is
  joined to a table U whose join key is NOT unique in U: every T row is repeated once per matching U row, so
  the aggregate is inflated / re-weighted. Evidence: U's row count vs distinct join-key count.

Each probe is a read-only, bounded query; results are cached per validator instance. Whether a validator is
precise enough to trigger repairs is measured offline first (scripts/validator_eval.py).
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from loop_engineer.checks import Finding, parse

VALIDATOR_SIGNALS = ("filter_value_not_found", "join_fanout_aggregate")


def _q(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


@dataclass
class ValidatorAgent:
    executor: Any                       # DatabricksSqlExecutor-like: execute(sql, db, max_rows=...)
    catalog: Any                        # SchemaCatalog (agent-visible schema)
    db: str = "dw"
    max_probes: int = 12
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)
    probes_run: int = 0

    # ---------------------------------------------------------------- probes
    def _probe(self, sql: str, max_rows: int = 500) -> list[tuple] | None:
        if sql in self._cache:
            return self._cache[sql]
        if self.probes_run >= self.max_probes:
            return None
        self.probes_run += 1
        ex = self.executor.execute(sql, self.db, max_rows=max_rows)
        rows = ex.rows if ex.ok else None
        self._cache[sql] = rows
        return rows

    def _columns(self, table: str) -> dict[str, str]:
        tb = self.catalog.tables.get(table)
        return {c.name.lower(): c.type for c in tb.columns} if tb else {}

    # ---------------------------------------------------------------- scopes
    def _scopes(self, tree: exp.Expression) -> list[tuple[exp.Expression, dict[str, str]]]:
        """(select node, alias -> base table) for every scope, only real catalog tables."""
        out = []
        try:
            scopes = traverse_scope(tree)
        except Exception:
            return out
        for sc in scopes:
            m = {a.lower(): s.name.lower() for a, s in sc.sources.items()
                 if isinstance(s, exp.Table) and s.name.lower() in self.catalog.tables}
            out.append((sc.expression, m))
        return out

    @staticmethod
    def _resolve(col: exp.Column, aliases: dict[str, str], columns_of) -> tuple[str, str] | None:
        name = col.name.lower()
        if col.table:
            t = aliases.get(col.table.lower())
            return (t, name) if t and name in columns_of(t) else None
        owners = sorted({t for t in aliases.values() if name in columns_of(t)})
        return (owners[0], name) if len(owners) == 1 else None

    # ---------------------------------------------------------------- validator 1: filter values
    def filter_values(self, tree: exp.Expression) -> list[Finding]:
        found: list[Finding] = []
        seen = set()
        for node, aliases in self._scopes(tree):
            where = node.args.get("where") if isinstance(node, exp.Select) else None
            conds = [where] if where else []
            conds += [j.args.get("on") for j in (node.args.get("joins") or []) if j.args.get("on") is not None]
            for cond in conds:
                for pred in cond.find_all(exp.EQ, exp.In):
                    if pred.find_ancestor(exp.Not) or pred.find_ancestor(exp.Select) is not node:
                        continue
                    if isinstance(pred, exp.EQ):
                        col, lits = (pred.left, [pred.right]) if isinstance(pred.left, exp.Column) else (pred.right, [pred.left])
                    else:
                        col, lits = pred.this, list(pred.expressions)
                    if not isinstance(col, exp.Column):
                        continue
                    values = [lit.this for lit in lits if isinstance(lit, exp.Literal) and lit.is_string]
                    tc = self._resolve(col, aliases, self._columns)
                    if not values or tc is None or "STRING" not in self._columns(tc[0]).get(tc[1], "").upper():
                        continue
                    for v in values:
                        key = (tc, v.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        t, c = tc
                        rows = self._probe(f"SELECT 1 FROM `{self.db}`.`{t}` WHERE `{c}` = {_q(v)} LIMIT 1", 1)
                        if rows is None or rows:
                            continue  # probe budget exhausted / failed, or the value exists
                        dist = self._probe(f"SELECT DISTINCT `{c}` FROM `{self.db}`.`{t}` WHERE `{c}` IS NOT NULL LIMIT 500") or []
                        vals = [str(r[0]) for r in dist]
                        close = difflib.get_close_matches(v, vals, n=3, cutoff=0.5)
                        close += [x for x in vals if v.lower() in x.lower() and x not in close][:3]
                        found.append(Finding(
                            "filter_value_not_found",
                            f"过滤值 '{v}' 在 {t}.{c} 中不存在" + (f"，最接近的真实值：{', '.join(close[:4])}" if close else ""),
                            {"table": t, "column": c, "value": v, "closest": close[:4]},
                            f"The filter value '{v}' does not occur in {t}.{c}"
                            + (f"; existing values that look closest: {', '.join(repr(x) for x in close[:4])}." if close
                               else "; check which column or value the question refers to.")))
        return found

    # ---------------------------------------------------------------- validator 2: join fan-out
    def join_fanout(self, tree: exp.Expression) -> list[Finding]:
        found: list[Finding] = []
        for node, aliases in self._scopes(tree):
            if not isinstance(node, exp.Select) or not node.args.get("joins") or len(set(aliases.values())) < 2:
                continue
            # join keys per table: table -> set of its columns used in column = column join conditions
            keys: dict[str, set[str]] = {}
            conds = [j.args.get("on") for j in node.args["joins"] if j.args.get("on") is not None]
            for cond in conds:
                for eq in cond.find_all(exp.EQ):
                    if not (isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column)):
                        continue
                    a = self._resolve(eq.left, aliases, self._columns)
                    b = self._resolve(eq.right, aliases, self._columns)
                    if a and b and a[0] != b[0]:
                        keys.setdefault(a[0], set()).add(a[1])
                        keys.setdefault(b[0], set()).add(b[1])
            # aggregated columns in THIS select (not windows, not DISTINCT, not COUNT(*))
            aggs = []
            for f in node.expressions:
                for agg in f.find_all(exp.Sum, exp.Avg, exp.Count):
                    if agg.find_ancestor(exp.Window) or agg.find_ancestor(exp.Select) is not node:
                        continue
                    arg = agg.this
                    if isinstance(arg, exp.Distinct) or not isinstance(arg, exp.Column):
                        continue
                    tc = self._resolve(arg, aliases, self._columns)
                    if tc:
                        aggs.append((agg.key.upper(), tc))
            for fn, (t, c) in aggs:
                for u, ukeys in keys.items():
                    if u == t or not ukeys:
                        continue
                    cols = ", ".join(f"`{k}`" for k in sorted(ukeys))
                    r = self._probe(f"SELECT COUNT(*), COUNT(DISTINCT {cols}) FROM `{self.db}`.`{u}`", 1)
                    if not r:
                        continue
                    total, distinct = int(r[0][0] or 0), int(r[0][1] or 0)
                    if distinct and total > distinct * 1.05:
                        found.append(Finding(
                            "join_fanout_aggregate",
                            f"{fn}({t}.{c}) 在与 {u} 关联后计算，但 {u} 的关联键（{', '.join(sorted(ukeys))}）不唯一"
                            f"（{total} 行只有 {distinct} 个不同的键），{t} 的每行会被重复计算",
                            {"aggregate": fn, "table": t, "column": c, "joined_table": u, "keys": sorted(ukeys),
                             "rows": total, "distinct_keys": distinct},
                            f"{fn}({t}.{c}) is computed after joining {u}, whose join key ({', '.join(sorted(ukeys))}) is "
                            f"not unique ({total} rows, {distinct} distinct keys), so each {t} row is counted several "
                            f"times. Aggregate {t} before the join, or join on a unique key."))
                        break
        return found

    # ---------------------------------------------------------------- entry point
    def validate(self, sql: str) -> list[Finding]:
        tree = parse(sql)
        if tree is None:
            return []
        return self.filter_values(tree) + self.join_fanout(tree)

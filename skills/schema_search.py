"""SchemaSearch — COLUMN_MAPPING_FAILURE.

Deterministic first, LLM only for what the schema cannot decide:

1. ``fix_column_refs`` checks EVERY qualified column reference of the query,
   scope by scope. A reference ``a.COL`` whose table (alias ``a``) has no column
   COL is re-pointed to the one table *in the same scope* that has it. Engines
   report only the first unresolved column, so fixing just that one left the
   next wrong reference to fail the re-run (phase-5 dev evaluation).
2. References that stay unresolved (no table in scope has the column, or
   several do) go to the LLM together with the partly fixed SQL, the engine's
   suggestions and similarly named schema columns.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from agent.join_graph import join_candidates
from agent.retriever import SchemaCatalog
from agent.sql_analysis import alias_map
from loop_engineer.diagnose import Diagnosis
from loop_engineer.observer import Observation
from skills.base import RepairContext, RepairResult, as_hints, llm_repair


@dataclass
class ColumnFix:
    sql: str
    changes: list[str] = field(default_factory=list)      # "a.COL -> b.COL"
    unresolved: list[str] = field(default_factory=list)   # "a.COL (no table in scope has it)" / "(ambiguous: ...)"
    parsed: bool = True


def fix_column_refs(sql: str, catalog: SchemaCatalog) -> ColumnFix:
    """Re-point every wrongly qualified column to the unique in-scope table that owns it."""
    try:
        tree = sqlglot.parse_one(sql, read="databricks")
    except sqlglot.errors.ParseError:
        return ColumnFix(sql, parsed=False)
    cols = {t: {c.name.lower() for c in tb.columns} for t, tb in catalog.tables.items()}
    fix = ColumnFix(sql)
    moved: dict[int, tuple[exp.Column, str]] = {}  # id(column node) -> (node, original qualifier)
    for scope in traverse_scope(tree):
        # alias -> base table, only for sources that are real tables of the catalog
        tables = {a.lower(): s.name.lower() for a, s in scope.sources.items()
                  if isinstance(s, exp.Table) and s.name.lower() in cols}
        for col in scope.columns:
            q = (col.table or "").lower()
            if not q or q not in tables:        # unqualified, or qualified by a CTE / subquery
                continue
            name = col.name.lower()
            if name in cols[tables[q]]:
                continue
            owners = [a for a, t in tables.items() if name in cols[t]]
            owners = list(dict.fromkeys(owners))
            if len({tables[a] for a in owners}) == 1:
                target = owners[0]
                moved[id(col)] = (col, col.table)
                col.set("table", exp.to_identifier(target))
                fix.changes.append(f"{q}.{col.name} -> {target}.{col.name}")
            elif owners:
                fix.unresolved.append(f"{q}.{col.name} (ambiguous: {', '.join(tables[a] for a in owners)})")
            else:
                fix.unresolved.append(f"{q}.{col.name} (no table in this scope has it)")
    # Guard: a moved column inside a column-vs-column comparison must not end up on the same
    # table as the other side. That turns a join condition into "x = x" (or a same-table
    # predicate): the query runs, but the tables are no longer related. Such a reference
    # needs a real join key, which the schema alone cannot pick -> revert and hand it on.
    for cmp in tree.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE):
        left, right = cmp.left, cmp.right
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            continue
        if (left.table or "").lower() != (right.table or "").lower():
            continue
        for node in (left, right):
            if id(node) in moved:
                col, original = moved.pop(id(node))
                col.set("table", exp.to_identifier(original))
                fix.unresolved.append(f"{original.lower()}.{col.name} in join/compare condition "
                                      f"'{cmp.sql(dialect='databricks')}' (needs a real join key)")
    fix.changes = list(dict.fromkeys(f"{orig.lower()}.{col.name} -> {col.table}.{col.name}"
                                     for col, orig in moved.values()))  # what really stayed changed
    fix.unresolved = list(dict.fromkeys(fix.unresolved))
    if fix.changes:
        fix.sql = tree.sql(dialect="databricks", pretty=True)
    return fix


def similar_columns(ctx: RepairContext, column: str, tables: list[str], n: int = 8) -> list[str]:
    pool = [f"{t}.{c.name}" for t in tables if t in ctx.catalog.tables for c in ctx.catalog.tables[t].columns]
    names = [p.split(".", 1)[1] for p in pool]
    close = set(difflib.get_close_matches(column.upper(), [x.upper() for x in names], n=n, cutoff=0.5))
    return [p for p in pool if p.split(".", 1)[1].upper() in close][:n]


class SchemaSearch:
    name = "SchemaSearch"

    def repair(self, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext) -> RepairResult:
        h = as_hints(diagnosis)
        col, qual = h.get("column"), h.get("qualifier")
        fix = fix_column_refs(obs.generated_sql, ctx.catalog)
        if fix.changes and not fix.unresolved:
            return RepairResult(fix.sql, self.name,
                                f"re-pointed {len(fix.changes)} column reference(s), deterministic: {'; '.join(fix.changes)}",
                                diagnosis.reason, tuple(sorted(set(alias_map(fix.sql).values()))), False,
                                details={"deterministic_changes": fix.changes, "unresolved": [],
                                         "start_sql": obs.generated_sql})
        # the schema cannot decide the rest: LLM, starting from the partly fixed SQL
        used = sorted(set(alias_map(fix.sql).values()))
        names = [col] if col else []
        names += [u.split(" ")[0].split(".", 1)[-1] for u in fix.unresolved]
        candidates = list(dict.fromkeys([*h.get("suggestions", []),
                                         *[c for n in names for c in similar_columns(ctx, n, list(obs.retrieved_tables))]]))
        problems = fix.unresolved or [f"{qual + '.' if qual else ''}{col} does not exist there"]
        joins, cands = "", []
        if any("needs a real join key" in p for p in problems):
            cands = [j.sql() for j in join_candidates(ctx.catalog, used)][:10]
            joins = (f"Join keys shared by the tables in the query (from the schema): {'; '.join(cands) or '(none)'}. "
                     "If two tables share no key, join them through another table that does. ")
        instruction = ("These column references are wrong: " + "; ".join(problems) + ". " + joins +
                       f"Candidate columns: {', '.join(candidates) or '(none)'}. "
                       "Map every phrase of the question to a column that really exists in the schema; if the right "
                       "column belongs to a table the query does not use yet, join that table. Keep everything else.")
        start = obs if not fix.changes else _with_sql(obs, fix.sql)
        action = (f"deterministic: {'; '.join(fix.changes)}; " if fix.changes else "") + \
                 f"LLM rewrite for {len(problems)} unresolved reference(s)"
        return llm_repair(self.name, start, diagnosis, ctx, [*used, *obs.retrieved_tables], instruction, action,
                          {"deterministic_changes": fix.changes, "unresolved": problems,
                           "candidate_columns": candidates, "join_candidates": cands,
                           "sql_before_deterministic": obs.generated_sql})


def _with_sql(obs: Observation, sql: str) -> Observation:
    from dataclasses import replace
    return replace(obs, generated_sql=sql)

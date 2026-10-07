"""Structural analysis of a SQL query: base tables, (table, column) references
with aliases resolved, column = column equalities, literals and operations.

Neutral module — no benchmark or gold knowledge. Used by the loop at runtime
(diagnosis) and by the evaluation-only failure labeler.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

_STAT = {"avg", "sum", "count", "min", "max", "stddev", "stddev_pop", "stddev_samp", "std", "variance", "var_pop",
         "var_samp", "median", "percentile", "percentile_cont", "approx_percentile"}


@dataclass
class SqlFacts:
    """What the generated SQL touches, with aliases resolved to base tables."""
    parsed: bool
    tables: set[str] = field(default_factory=set)            # base tables
    columns: set[tuple[str, str]] = field(default_factory=set)  # (table, column), lower-case
    equalities: set[frozenset] = field(default_factory=set)  # {(t1,c1),(t2,c2)} column = column
    literals: set[str] = field(default_factory=set)
    operations: set[str] = field(default_factory=set)
    error: str | None = None


def _operations(tree: exp.Expression) -> set[str]:
    ops = set()
    for fn in tree.find_all(exp.AggFunc):
        name = fn.key.lower()
        # sqlglot names the statistics family inconsistently (stddev, stddevpop, variancepop, ...)
        ops.add("stat" if "std" in name or "var" in name else name)
    if any(True for _ in tree.find_all(exp.Window)):
        ops.add("window")
    if any(s.args.get("group") for s in tree.find_all(exp.Select)):
        ops.add("group_by")
    if any(s.args.get("having") for s in tree.find_all(exp.Select)):
        ops.add("having")
    if any(True for _ in tree.find_all(exp.Union, exp.Intersect, exp.Except)):
        ops.add("set_operation")
    if any(s.args.get("limit") and s.args.get("order") for s in tree.find_all(exp.Select)):
        ops.add("top_k")
    return {o for o in ops if o in _STAT or o in ("stat", "window", "group_by", "having", "set_operation", "top_k")}


def sql_facts(sql: str, schema: dict[str, dict[str, str]], dialect: str = "databricks") -> SqlFacts:
    """``schema``: {table: {column: type}} (lower-case) of the database."""
    if not sql or not sql.strip():
        return SqlFacts(False, error="no SQL")
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        return SqlFacts(False, error=f"parse error: {str(e)[:120]}")
    try:  # attach table qualifiers to unqualified columns where the schema allows it
        tree = qualify(tree, schema=schema, dialect=dialect, validate_qualify_columns=False,
                       identify=False, quote_identifiers=False)
    except Exception:
        pass
    f = SqlFacts(True, operations=_operations(tree))
    for scope in traverse_scope(tree):
        alias_to_table = {a: s.name.lower() for a, s in scope.sources.items() if isinstance(s, exp.Table)}
        for t in alias_to_table.values():
            if t in schema:
                f.tables.add(t)

        def resolve(col: exp.Column) -> tuple[str, str] | None:
            name = col.name.lower()
            if col.table:
                t = alias_to_table.get(col.table) or alias_to_table.get(col.table.lower())
                return (t, name) if t else None
            owners = [t for t in alias_to_table.values() if name in schema.get(t, {})]
            return (owners[0], name) if len(owners) == 1 else None

        for col in scope.columns:
            r = resolve(col)
            if r:
                f.columns.add(r)
        for eq in scope.expression.find_all(exp.EQ):
            if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
                a, b = resolve(eq.left), resolve(eq.right)
                if a and b:
                    f.equalities.add(frozenset((a, b)))
    for lit in tree.find_all(exp.Literal):
        f.literals.add(str(lit.this).strip().lower())
    return f


def alias_map(sql: str, dialect: str = "databricks") -> dict[str, str]:
    """alias (lower-case) -> base table (lower-case), over every scope; {} if unparseable."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return {}
    out: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        out[(t.alias or t.name).lower()] = name
        out.setdefault(name, name)
    return out

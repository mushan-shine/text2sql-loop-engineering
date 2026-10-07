"""Semantic checks for the self-verifier: does the SQL deliver what the question asks?

SelfVerifier's original signals only catch attempts that are visibly broken
(error / empty / all-NULL). These checks compare the attempt with the QUESTION
itself — BEAVER questions spell out outputs, filters, grouping and functions —
and with the attempt's own result. They use only agent-visible inputs
(question, generated SQL, result rows); nothing gold-derived.

Each check returns ``Finding(signal, message, evidence)`` or nothing. Which
signals are trusted enough to trigger a repair is decided with data
(scripts/verifier_eval.py: hits on wrong answers vs. false alarms on gold SQL),
not by intuition.

Static checks (question + SQL):
    missing_literal       a quoted value / year / compared number of the question is absent from the SQL
    missing_aggregate     the question asks for average / std / variance / min / max / count / median but the SQL lacks it
    missing_grouping      "for each / per" + an aggregate, but no GROUP BY / PARTITION BY
    rounding              "do not return rounded answers" but the SQL uses ROUND (or the reverse)
    join_tautology        a join / filter compares a column with itself (x = x)
    join_without_condition a JOIN without ON / USING (cross product)
    output_columns        the question enumerates more outputs than the SELECT returns
Result checks (question + SQL + rows):
    duplicate_rows        identical result rows although the SQL has no DISTINCT (typical of join fan-out)
    single_row_for_each   "for each ..." but exactly one row came back
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp


# Grades from scripts/verifier_eval.py (dev runs, 30 executed-but-wrong attempts; false alarms on the gold SQL
# of 30 dev questions + 5,687 non-evaluation questions), 2026-09-28:
#   trigger-grade (false-alarm rate ~0 on gold): join_tautology, join_without_condition, missing_grouping, rounding
#   advisory only (too noisy to trigger a repair): missing_literal 7.5%, output_columns 10.8%, missing_aggregate 4.2%,
#     duplicate_rows / single_row_for_each (gold results have them too: 5/30 and 4/30 dev questions)
#   numeric consistency (check_numeric): 0 false alarms on 21 dev gold + 357 MySQL-executed gold results that
#     have comparable columns; 0 hits on the 22 applicable wrong attempts (their numbers are consistent — the
#     errors are in WHAT is computed, not HOW). Trigger-grade: free, exact, catches statistics computed over
#     mismatched row sets whenever that happens.
NUMERIC_SIGNALS = ("negative_statistic", "count_not_integer", "min_greater_than_max", "avg_outside_min_max",
                   "std_var_mismatch", "std_exceeds_range")
TRIGGER_SIGNALS = ("join_tautology", "join_without_condition", "missing_grouping", "rounding") + NUMERIC_SIGNALS


@dataclass(frozen=True)
class Finding:
    signal: str
    message: str                   # for people (console / report)
    evidence: dict[str, Any] = field(default_factory=dict)
    hint: str = ""                 # for the repair prompt (English, like the rest of the prompt)


# ------------------------------------------------------------------ helpers

_AGG_WORDS = {  # phrase in the question -> acceptable SQL functions (upper-case, prefix match)
    "average": ("AVG",), "mean ": ("AVG",),
    "standard deviation": ("STDDEV", "STD"),
    "variance": ("VAR",),
    "minimum": ("MIN",), "maximum": ("MAX",),
    "median": ("MEDIAN", "PERCENTILE"),
    # not "number of": it usually names a column ("the number of enrolled students"), not a COUNT
}
_GROUP_WORDS = re.compile(r"\b(for each|for every|per|by each|grouped by|for all)\b", re.I)
_COMPARE_NUM = re.compile(  # only real comparisons, not "top 10" / "window of 3" / "Course 18"
    r"\b(?:greater than|more than|less than|fewer than|at least|at most|exceed(?:s|ing)?|"
    r"(?:greater|less) than or equal to)\s+(-?\d+(?:\.\d+)?)\b(?!\s*(?:preceding|following|rows|percent))", re.I)
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_QUOTED = re.compile(r"(?<![A-Za-z])'([^'\s][^']{0,40})'(?![A-Za-z])")  # not apostrophes ("department's")
_NO_ROUND = re.compile(r"(do not|don't|never|without)\s+(return\s+any\s+)?round", re.I)
_ROUND_TO = re.compile(r"\brounded to\b|\bround(?:ed)? (?:it |them )?to \d", re.I)
_ENUM_VERB = re.compile(r"\b(provide|show|list|return|give|display|retrieve|report|include|output|select)\b\s+"
                        r"(?:me\s+)?(?:the\s+)?", re.I)
_ENUM_STOP = re.compile(r"(,?\s+(?:considering|but only|only for|only|where|when|whose|restrict|restricting|"
                        r"filtered|limited|ordered|sorted|order(?:ed)? by|sort(?:ed)? by|excluding|including only|"
                        r"along with any|in which|within|if|that (?:are|have|is|has))\b|[.;]|$)", re.I)


def parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="databricks")
    except Exception:
        try:
            return sqlglot.parse_one(sql, read="mysql")
        except Exception:
            return None


def _sql_literals(tree: exp.Expression) -> set[str]:
    out = set()
    for lit in tree.find_all(exp.Literal):
        v = str(lit.this).strip().lower()
        out.add(v)
        try:
            f = float(v)
            out.add(str(int(f)) if f == int(f) else str(f))
        except ValueError:
            pass
    return out


def _functions(tree: exp.Expression) -> set[str]:
    names = set()
    for f in tree.find_all(exp.Func):
        names.add(f.sql_name().upper())
        if isinstance(f, exp.Anonymous):
            names.add(str(f.this).upper())
    for f in tree.find_all(exp.AggFunc):
        names.add(f.key.upper())
    return names


def expected_outputs(question: str) -> int | None:
    """Rough count of the outputs the question enumerates ("provide the a, b, and c" -> 3); None if unclear."""
    m = _ENUM_VERB.search(question)
    if not m:
        return None
    rest = question[m.end():]
    stop = _ENUM_STOP.search(rest)
    clause = rest[: stop.start()] if stop else rest
    items = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+(?:the\s+)?", clause) if x.strip()]
    return len(items) if len(items) >= 2 else None


def _select_width(tree: exp.Expression) -> int | None:
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if isinstance(tree, (exp.Union,)):
        sel = tree.left if isinstance(tree.left, exp.Select) else tree.find(exp.Select)
    if sel is None or any(isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star))
                          for e in sel.expressions):
        return None
    return len(sel.expressions)


# ------------------------------------------------------------------ static checks

def check_static(question: str, sql: str) -> list[Finding]:
    tree = parse(sql)
    if tree is None:
        return []
    q = question
    ql = q.lower()
    out: list[Finding] = []
    lits = _sql_literals(tree)
    sql_l = sql.lower()

    missing = [v for v in _QUOTED.findall(q) if v.strip().lower() not in lits and f"'{v.lower()}'" not in sql_l]
    nums = {n for n in _COMPARE_NUM.findall(q)} | set(_YEAR.findall(q))
    missing += [n for n in sorted(nums) if n not in lits and not re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", sql_l)]
    if missing:
        out.append(Finding("missing_literal", "题干中的这些值没有出现在 SQL 里：" + ", ".join(missing),
                           {"values": missing}, "These values from the question are not used in the SQL: "
                           + ", ".join(missing) + "."))

    fns = _functions(tree)
    lacking = [w.strip() for w, accepted in _AGG_WORDS.items()
               if w in ql and not any(fn.startswith(a) for fn in fns for a in accepted)]
    if lacking:
        out.append(Finding("missing_aggregate", "题干要求的统计量在 SQL 里没有对应的函数：" + ", ".join(lacking),
                           {"phrases": lacking}, "The question asks for " + ", ".join(lacking)
                           + " but the SQL computes no such aggregate."))

    has_agg = any(w in ql for w in _AGG_WORDS) or "total" in ql or "sum of" in ql
    sql_aggs = [f for f in tree.find_all(exp.AggFunc) if not f.find_ancestor(exp.Window)]
    grouped = any(s.args.get("group") for s in tree.find_all(exp.Select)) or tree.find(exp.Window) is not None
    if _GROUP_WORDS.search(q) and has_agg and sql_aggs and not grouped:
        phrase = _GROUP_WORDS.search(q).group(0)
        out.append(Finding("missing_grouping", "题干要求“对每个…”分别统计，但 SQL 没有 GROUP BY / PARTITION BY",
                           {"phrase": phrase}, f"The question asks for results '{phrase}' group, but the SQL "
                           "aggregates without GROUP BY (or PARTITION BY), so it returns one overall value."))

    has_round = "ROUND" in fns or "round(" in sql_l
    if _NO_ROUND.search(q) and has_round:
        out.append(Finding("rounding", "题干要求不要四舍五入，但 SQL 使用了 ROUND", {},
                           "The question says not to round, but the SQL uses ROUND. Remove the rounding."))
    elif _ROUND_TO.search(q) and not has_round:
        out.append(Finding("rounding", "题干要求四舍五入，但 SQL 没有 ROUND", {},
                           "The question asks for rounded values, but the SQL does not round."))

    for cmp in tree.find_all(exp.EQ):
        l, r = cmp.left, cmp.right
        if isinstance(l, exp.Column) and isinstance(r, exp.Column) and l.name.lower() == r.name.lower() \
                and (l.table or "").lower() == (r.table or "").lower():
            out.append(Finding("join_tautology", f"条件 {cmp.sql()} 两边是同一列，恒为真，表之间失去关联",
                               {"condition": cmp.sql()}, f"The condition {cmp.sql()} compares a column with "
                               "itself, so it is always true and the tables are no longer related. Use a real "
                               "join key between the two tables."))
            break
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        # sqlglot fills a missing ON with a literal TRUE; comma joins / CROSS JOIN are explicit cross products
        no_condition = on is None or (isinstance(on, exp.Boolean) and on.this is True)
        if no_condition and not j.args.get("using") and (j.args.get("kind") or "").upper() != "CROSS":
            out.append(Finding("join_without_condition", f"JOIN {j.this.sql()} 没有关联条件（笛卡尔积）",
                               {"table": j.this.sql()}, f"JOIN {j.this.sql()} has no ON condition, which "
                               "multiplies rows (cross product). Add the join key."))
            break

    exp_n, width = expected_outputs(q), _select_width(tree)
    if exp_n and width and width < exp_n:
        out.append(Finding("output_columns", f"题干列出约 {exp_n} 项输出，SQL 只返回 {width} 列",
                           {"expected": exp_n, "select": width}))
    return out


# ------------------------------------------------------------------ result checks

def check_result(question: str, sql: str, rows: list[tuple] | None) -> list[Finding]:
    if not rows:
        return []
    tree = parse(sql)
    out: list[Finding] = []
    distinct = tree is not None and any(s.args.get("distinct") for s in tree.find_all(exp.Select))
    if not distinct and len(rows) >= 2:
        seen, dup = set(), 0
        for r in rows:
            key = tuple(map(str, r))
            dup += key in seen
            seen.add(key)
        if dup and dup / len(rows) >= 0.2:
            out.append(Finding("duplicate_rows", f"结果中有 {dup} 行与其他行完全相同（占 {dup / len(rows):.0%}），"
                                                 "常见原因是关联导致行数翻倍", {"duplicates": dup, "rows": len(rows)}))
    if re.search(r"\bfor each\b|\bper\b", question, re.I) and len(rows) == 1:
        out.append(Finding("single_row_for_each", "题干要求“对每个…”分别给出结果，但只返回了 1 行", {"rows": 1}))
    return out


# ------------------------------------------------------------------ numeric consistency ("calculator") checks
#
# Question-independent: they only test relations that MUST hold between the numbers a query returns, so they
# are not affected by BEAVER's question / gold-SQL mismatches. All arithmetic is done here in code on the rows
# the database computed — never by an LLM. Each output column is mapped to the aggregate that produced it
# (following CTE aliases), so e.g. AVG(x), MIN(x), MAX(x) of the same x can be compared row by row.

_ROLE_OF = (("stddev", "std"), ("std", "std"), ("variance", "var"), ("var", "var"), ("avg", "avg"),
            ("min", "min"), ("max", "max"), ("count", "count"), ("sum", "sum"))


@dataclass(frozen=True)
class ColumnRole:
    role: str          # avg | min | max | std | var | count | sum
    arg: str           # normalized argument, e.g. "num_enrolled_students"
    family: str = ""   # pop | samp | "" (std / var only)


def _agg_role(node: exp.Expression) -> ColumnRole | None:
    """Role of an output expression that IS an aggregate (optionally windowed / cast), else None."""
    while isinstance(node, (exp.Alias, exp.Cast, exp.Paren)):
        node = node.this
    if isinstance(node, exp.Window):
        node = node.this
    if not isinstance(node, exp.Func):
        return None
    name = (node.sql_name() if not isinstance(node, exp.Anonymous) else str(node.this)).lower()
    role = next((r for key, r in _ROLE_OF if name.startswith(key)), None)
    if role is None:
        return None
    if isinstance(node, exp.Anonymous):     # e.g. MySQL STD(x): arguments live in .expressions
        arg = node.expressions[0] if node.expressions else None
    else:
        arg = node.this if isinstance(node.this, exp.Expression) else None
    if isinstance(arg, exp.Distinct):
        return None  # COUNT(DISTINCT x) etc.: different population, skip
    if arg is None or isinstance(arg, exp.Star):
        return ColumnRole(role, "*") if role == "count" else None
    col = arg.name.lower() if isinstance(arg, exp.Column) else arg.sql().lower()
    # population vs sample: only a std and a var of the same family must satisfy var = std²
    # ("default" = the engine's plain STDDEV / VARIANCE, which agree with each other within one engine)
    fam = ("pop" if "pop" in name else "samp" if "samp" in name else "default") if role in ("std", "var") else ""
    return ColumnRole(role, col, fam)


def output_roles(sql: str) -> list[ColumnRole | None]:
    """One entry per output column of the outermost SELECT (None when not an aggregate / unknown)."""
    tree = parse(sql)
    if tree is None:
        return []
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        return []
    # alias -> expression for every CTE / subquery select, to follow references like inner_cte.avg_x
    defs: dict[str, exp.Expression] = {}
    for s in tree.find_all(exp.Select):
        if s is sel:
            continue
        for e in s.expressions:
            if e.alias_or_name:
                defs.setdefault(e.alias_or_name.lower(), e)
    roles: list[ColumnRole | None] = []
    for e in sel.expressions:
        role, node, hops = _agg_role(e), e, 0
        while role is None and hops < 4:
            inner = node.this if isinstance(node, exp.Alias) else node
            if not isinstance(inner, exp.Column) or inner.name.lower() not in defs:
                break
            node = defs[inner.name.lower()]
            role, hops = _agg_role(node), hops + 1
        roles.append(role)
    return roles


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, dict) and "__decimal__" in v:
        v = v["__decimal__"]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def check_numeric(sql: str, rows: list[tuple] | None) -> list[Finding]:
    """Relations every correct aggregate result satisfies; a violation means the numbers cannot be right."""
    if not rows:
        return []
    roles = output_roles(sql)
    if not roles or not any(roles) or len(roles) != len(rows[0]):
        return []
    idx = defaultdict(dict)                      # arg -> role -> column index
    for i, r in enumerate(roles):
        if r:
            idx[r.arg].setdefault(r.role, i)
    tol = 1e-6
    bad: dict[str, list[str]] = defaultdict(list)

    for row_no, row in enumerate(rows):
        for i, r in enumerate(roles):
            v = _num(row[i]) if r else None
            if v is None:
                continue
            if r.role in ("std", "var", "count") and v < -tol:
                bad["negative_statistic"].append(f"第 {row_no + 1} 行第 {i + 1} 列 {r.role}({r.arg}) = {v}")
            if r.role == "count" and abs(v - round(v)) > tol:
                bad["count_not_integer"].append(f"第 {row_no + 1} 行第 {i + 1} 列 count({r.arg}) = {v}")
        for arg, cols in idx.items():
            val = {role: _num(row[c]) for role, c in cols.items()}
            lo, hi, avg = val.get("min"), val.get("max"), val.get("avg")
            if lo is not None and hi is not None and lo > hi + tol * max(1, abs(hi)):
                bad["min_greater_than_max"].append(f"第 {row_no + 1} 行 {arg}: min {lo} > max {hi}")
            if avg is not None and lo is not None and hi is not None:
                slack = tol * max(1.0, abs(lo), abs(hi))
                if avg < lo - slack or avg > hi + slack:
                    bad["avg_outside_min_max"].append(f"第 {row_no + 1} 行 {arg}: avg {avg} 不在 [{lo}, {hi}]")
            std, var = val.get("std"), val.get("var")
            if std is not None and var is not None and std >= 0 and var >= 0:
                fs, fv = roles[cols["std"]].family, roles[cols["var"]].family
                # only explicit, identical families: plain STDDEV / VARIANCE differ between engines and sqlglot
                # folds VAR_SAMP into VARIANCE, so "default" is never compared (no false alarms by construction)
                if fs == fv and fs in ("pop", "samp") and abs(std * std - var) > 1e-4 * max(1.0, var):
                    bad["std_var_mismatch"].append(f"第 {row_no + 1} 行 {arg}: std² = {std * std:.6g} ≠ var = {var:.6g}")
            if std is not None and lo is not None and hi is not None and std > (hi - lo) + tol * max(1, hi - lo):
                bad["std_exceeds_range"].append(f"第 {row_no + 1} 行 {arg}: std {std} > max - min = {hi - lo}")

    msgs = {"negative_statistic": "统计量（标准差 / 方差 / 计数）出现负数",
            "count_not_integer": "计数不是整数",
            "min_greater_than_max": "同一列的最小值大于最大值",
            "avg_outside_min_max": "平均值不在同一列的最小值和最大值之间",
            "std_var_mismatch": "同一列的方差不等于标准差的平方",
            "std_exceeds_range": "标准差大于同一列的极差（最大值 − 最小值）"}
    hints = {"negative_statistic": "a standard deviation / variance / count is negative",
             "count_not_integer": "a count is not an integer",
             "min_greater_than_max": "a minimum is larger than the maximum of the same column",
             "avg_outside_min_max": "an average lies outside the minimum..maximum of the same column",
             "std_var_mismatch": "a variance does not equal the square of the standard deviation of the same column",
             "std_exceeds_range": "a standard deviation exceeds max - min of the same column"}
    return [Finding(sig, f"{msgs[sig]}（{len(ex)} 处，例：{ex[0]}）", {"examples": ex[:5], "count": len(ex)},
                    f"The numbers in the result are inconsistent: {hints[sig]} (e.g. {ex[0]}). The aggregates are "
                    "probably computed over different row sets (e.g. a join that multiplies rows, or columns taken "
                    "from different groups). Compute related statistics over the same rows.")
            for sig, ex in bad.items()]


# 校验问题与sql之间的语义是否一致，问题与结果要求是否一致，数据一致性是否满足
def check_all(question: str, sql: str, rows: list[tuple] | None) -> list[Finding]:
    return check_static(question, sql) + check_result(question, sql, rows) + check_numeric(sql, rows)

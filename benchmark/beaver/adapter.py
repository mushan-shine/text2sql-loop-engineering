"""Benchmark Adapter — rule-based, documented, result-validated.

Only for gold SQL that is NOT ``COMPATIBLE``. The original gold SQL is never
modified; an adaptation is a separate record:

    original_gold_sql | adapted_sql | adaptation_rule | semantic_validation

Every rule rewrites a MySQL construct whose meaning is fixed by the MySQL
Reference Manual into the Databricks construct with the same meaning. Edits
are minimal text splices located with the sqlglot tokenizer (string literals
and qualified names such as ``t.std`` are never touched), so the diff between
original and adapted SQL is exactly the rule and nothing else.

An adaptation is ``RESULT_EQUIVALENT`` only when the adapted SQL, run on
Databricks, reproduces the MySQL (official engine) result and is stable.
Documentation fixes the meaning; the result check guards the implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import sqlglot
from sqlglot.tokens import TokenType

from benchmark.beaver.compatibility import EngineRuns, run_repeated
from benchmark.beaver.dataset import BeaverCase
from benchmark.beaver.evaluator import cross_engine_match
from execution.base import SqlExecutor

RESULT_EQUIVALENT = "RESULT_EQUIVALENT"
RESULT_DIFFERS = "RESULT_DIFFERS"
ADAPTED_SQL_FAILS = "ADAPTED_SQL_FAILS"
NO_ADAPTATION = "NO_ADAPTATION"  # no rule applies

MYSQL_AGG_DOC = "https://dev.mysql.com/doc/refman/8.0/en/aggregate-functions.html"
MYSQL_FRAME_DOC = "https://dev.mysql.com/doc/refman/8.0/en/window-functions-frames.html"

# rule name -> (what it does, documentation)
RULES: dict[str, tuple[str, str]] = {
    "mysql_population_statistics": (
        "MySQL VARIANCE()/STD()/STDDEV() are population statistics (documented synonyms of "
        "VAR_POP()/STDDEV_POP()); Databricks variance()/std()/stddev() are sample statistics. "
        "Rewrite to the explicit VAR_POP/STDDEV_POP.", MYSQL_AGG_DOC),
    "nonaggregate_window_frame_ignored": (
        "MySQL accepts but ignores a frame clause on nonaggregate window functions "
        "(ROW_NUMBER, RANK, DENSE_RANK, PERCENT_RANK, CUME_DIST, NTILE, LAG, LEAD); Databricks rejects it. "
        "Remove the ignored frame clause.", MYSQL_FRAME_DOC),
}

_POP = {"VARIANCE": "VAR_POP", "STD": "STDDEV_POP", "STDDEV": "STDDEV_POP"}
_NONAGG = {"ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE", "LAG", "LEAD"}


@dataclass
class AdaptationRecord:
    case_id: str
    original_gold_sql: str
    adapted_sql: str | None
    adaptation_rule: str
    semantic_validation: str
    detail: str
    adapted_result_hash: str | None = None  # Databricks hash of the adapted SQL, for drift checks

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- rules


def _matching_paren(tokens: list, i: int) -> int | None:
    """Index of the R_PAREN matching the L_PAREN at ``tokens[i]``."""
    depth = 0
    for j in range(i, len(tokens)):
        if tokens[j].token_type == TokenType.L_PAREN:
            depth += 1
        elif tokens[j].token_type == TokenType.R_PAREN:
            depth -= 1
            if depth == 0:
                return j
    return None


def _is_call(tokens: list, i: int) -> bool:
    """tokens[i] is a function name: followed by '(' and not a qualified name (``t.std``)."""
    return (i + 1 < len(tokens) and tokens[i + 1].token_type == TokenType.L_PAREN
            and not (i > 0 and tokens[i - 1].token_type == TokenType.DOT))


def _edits_population_statistics(sql: str, tokens: list) -> list[tuple[int, int, str]]:
    return [(t.start, t.end + 1, _POP[t.text.upper()])
            for i, t in enumerate(tokens)
            if t.token_type != TokenType.STRING and t.text.upper() in _POP and _is_call(tokens, i)]


def _edits_nonaggregate_frames(sql: str, tokens: list) -> list[tuple[int, int, str]]:
    edits = []
    for i, t in enumerate(tokens):
        if t.token_type == TokenType.STRING or t.text.upper() not in _NONAGG or not _is_call(tokens, i):
            continue
        close_args = _matching_paren(tokens, i + 1)
        if close_args is None or close_args + 2 >= len(tokens):
            continue
        if tokens[close_args + 1].token_type != TokenType.OVER or tokens[close_args + 2].token_type != TokenType.L_PAREN:
            continue
        open_w = close_args + 2
        close_w = _matching_paren(tokens, open_w)
        if close_w is None:
            continue
        depth = 0
        for k in range(open_w + 1, close_w):
            tt = tokens[k].token_type
            if tt == TokenType.L_PAREN:
                depth += 1
            elif tt == TokenType.R_PAREN:
                depth -= 1
            elif depth == 0 and tt in (TokenType.ROWS, TokenType.RANGE):
                start = tokens[k].start
                while start > 0 and sql[start - 1].isspace():  # also drop the whitespace before ROWS
                    start -= 1
                edits.append((start, tokens[close_w].start, ""))
                break
    return edits


_RULE_EDITS = {
    "mysql_population_statistics": _edits_population_statistics,
    "nonaggregate_window_frame_ignored": _edits_nonaggregate_frames,
}


def apply_rules(sql: str) -> tuple[str, list[str]]:
    """Apply every applicable rule. Returns (adapted_sql, rule names applied)."""
    try:
        tokens = sqlglot.tokenize(sql, read="mysql")
    except sqlglot.errors.TokenError:
        return sql, []
    edits: list[tuple[int, int, str]] = []
    applied: list[str] = []
    for name, fn in _RULE_EDITS.items():
        e = fn(sql, tokens)
        if e:
            edits.extend(e)
            applied.append(name)
    out = sql
    for start, end, repl in sorted(edits, key=lambda x: x[0], reverse=True):
        out = out[:start] + repl + out[end:]
    return out, applied


# --------------------------------------------------------------------------- validation


def try_adapt(case: BeaverCase, reference_runs: EngineRuns, candidate: SqlExecutor, repeats: int) -> AdaptationRecord:
    adapted, applied = apply_rules(case.gold_sql)
    rule = "+".join(applied) if applied else "none"
    if not applied:
        return AdaptationRecord(case.case_id, case.gold_sql, None, rule, NO_ADAPTATION, "no documented rule applies")
    docs = "; ".join(f"{r}: {RULES[r][1]}" for r in applied)
    runs = run_repeated(candidate, adapted, case.db, repeats)
    if runs.status != "SUCCESS":
        return AdaptationRecord(case.case_id, case.gold_sql, adapted, rule, ADAPTED_SQL_FAILS,
                                f"{runs.status}: {runs.error_class or (runs.error or '')[:200]} | {docs}")
    cmp = cross_engine_match(runs.rows, reference_runs.rows)
    ok = cmp.set_match and runs.stable and reference_runs.stable
    return AdaptationRecord(case.case_id, case.gold_sql, adapted, rule,
                            RESULT_EQUIVALENT if ok else RESULT_DIFFERS,
                            f"{cmp.reason}; stable={runs.stable} | {docs}",
                            runs.result_hashes[0] if runs.result_hashes else None)

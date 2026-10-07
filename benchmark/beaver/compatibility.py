"""Phase 0.3 — SQL compatibility validation of ORIGINAL BEAVER gold SQL.

For each case:
  1. run the original gold SQL on the reference engine (MySQL, BEAVER's official engine);
  2. run the same, unmodified text on Databricks SQL;
  3. repeat each ``stability_repeats`` times (result cache disabled);
  4. classify.

``COMPATIBLE`` means: executes on Databricks unmodified AND reproduces the
MySQL result (canonical set equality = official EX semantics) AND is stable.
Executing without error is NOT enough — that would hide semantic drift.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

from benchmark.beaver.dataset import BeaverCase
from benchmark.beaver.evaluator import cross_engine_match, result_hash
from execution.base import ExecutionResult, SqlExecutor

log = logging.getLogger(__name__)

COMPATIBLE = "COMPATIBLE"
INCOMPATIBLE_SYNTAX = "INCOMPATIBLE_SYNTAX"
INCOMPATIBLE_FUNCTION = "INCOMPATIBLE_FUNCTION"
INCOMPATIBLE_SEMANTICS = "INCOMPATIBLE_SEMANTICS"
INCOMPATIBLE_SCHEMA = "INCOMPATIBLE_SCHEMA"
UNKNOWN = "UNKNOWN"
# Gold SQL does not run on BEAVER's own engine → cannot serve as ground truth at all.
REFERENCE_FAILED = "REFERENCE_FAILED"

STATUSES = (COMPATIBLE, INCOMPATIBLE_SYNTAX, INCOMPATIBLE_FUNCTION, INCOMPATIBLE_SEMANTICS,
            INCOMPATIBLE_SCHEMA, UNKNOWN, REFERENCE_FAILED)

_SYNTAX = ("PARSE_SYNTAX_ERROR", "PARSE_EMPTY_STATEMENT", "INVALID_IDENTIFIER", "UNCLOSED_BRACKETED_COMMENT",
           "INVALID_SQL_SYNTAX", "INVALID_TYPED_LITERAL", "UNSUPPORTED_FEATURE", "INVALID_ESCAPE_CHAR")
_FUNCTION = ("UNRESOLVED_ROUTINE", "WRONG_NUM_ARGS", "DATATYPE_MISMATCH", "INVALID_PARAMETER_VALUE",
             "INVALID_FORMAT", "COLLATION_MISMATCH", "UNSUPPORTED_COLLATION")
_SCHEMA = ("TABLE_OR_VIEW_NOT_FOUND", "UNRESOLVED_COLUMN", "SCHEMA_NOT_FOUND", "AMBIGUOUS_REFERENCE",
           "AMBIGUOUS_COLUMN_OR_FIELD", "FIELD_NOT_FOUND", "UNRESOLVED_FIELD",
           # e.g. UNION of a UTF8_LCASE column with an untyped literal: caused by the replicated schema
           "INCOMPATIBLE_COLUMN_TYPE")
# Errors caused by a *semantic* difference between engines (MySQL would have
# returned something; Databricks refuses or computes differently).
_SEMANTICS = ("MISSING_AGGREGATION", "MISSING_GROUP_BY", "DIVIDE_BY_ZERO", "CAST_INVALID_INPUT",
              "CAST_OVERFLOW", "ARITHMETIC_OVERFLOW", "NUMERIC_VALUE_OUT_OF_RANGE",
              "INCONSISTENT_BEHAVIOR_CROSS_VERSION", "GROUP_BY_POS_OUT_OF_RANGE")


def classify_error(error_class: str | None, message: str | None) -> str:
    cls = (error_class or "").upper()
    head = cls.split(".")[0]
    msg = (message or "").lower()
    if head in _SYNTAX or "syntax error" in msg or "parseexception" in msg:
        return INCOMPATIBLE_SYNTAX
    if head in _SCHEMA:
        return INCOMPATIBLE_SCHEMA
    # Spark raises this without an error class for RANK()/ROW_NUMBER()/LAG() ... OVER (... ROWS ...)
    if head in _FUNCTION or "undefined function" in msg or "window frame" in msg:
        return INCOMPATIBLE_FUNCTION
    if head in _SEMANTICS or head.startswith("DATETIME_") or head.startswith("CAST_"):
        return INCOMPATIBLE_SEMANTICS
    return UNKNOWN


# --------------------------------------------------------------------------- static hazards


def static_hazards(sql: str) -> list[str]:
    """Constructs whose MySQL meaning may differ on Databricks. Informational —
    the verdict is decided by executed results, not by these flags."""
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except sqlglot.errors.ParseError:
        return ["UNPARSEABLE_BY_SQLGLOT_MYSQL"]
    hz: set[str] = set()
    for sel in tree.find_all(exp.Select):
        limit = sel.args.get("limit")
        if limit is not None and not sel.args.get("order"):
            hz.add("LIMIT_WITHOUT_ORDER_BY")
        group = sel.args.get("group")
        if group is not None:
            grouped = {g.sql() for g in group.expressions}
            for proj in sel.expressions:
                node = proj.this if isinstance(proj, exp.Alias) else proj
                if isinstance(node, exp.Column) and node.sql() not in grouped and proj.alias_or_name not in grouped:
                    hz.add("NONAGGREGATED_COLUMN_IN_GROUP_BY")
    for eq in tree.find_all(exp.EQ, exp.NEQ, exp.Like, exp.In):
        if any(isinstance(n, exp.Literal) and n.is_string for n in eq.walk()):
            hz.add("STRING_COMPARISON_COLLATION")
            break
    if any(isinstance(n, exp.IntDiv) for n in tree.walk()):
        hz.add("INTEGER_DIV")
    if any(isinstance(n, exp.Div) for n in tree.walk()):
        hz.add("DIVISION")
    for fn in tree.find_all(exp.Anonymous):
        hz.add(f"UNKNOWN_FUNCTION:{fn.name.upper()}")
    for tbl in tree.find_all(exp.Table):
        if tbl.args.get("db"):
            hz.add("DB_QUALIFIED_TABLE")
            break
    return sorted(hz)


# --------------------------------------------------------------------------- validation


@dataclass
class EngineRuns:
    status: str
    error: str | None
    error_class: str | None
    row_count: int | None
    result_hashes: list[str]
    elapsed_ms: list[int]
    rows: list[tuple] = field(default_factory=list, repr=False)
    columns: list[str] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        return self.status == "SUCCESS" and len(set(self.result_hashes)) == 1


def run_repeated(executor: SqlExecutor, sql: str, db: str, repeats: int) -> EngineRuns:
    hashes, times = [], []
    first: ExecutionResult | None = None
    for _ in range(max(1, repeats)):
        r = executor.execute(sql, db)
        if first is None:
            first = r
        if not r.ok:
            return EngineRuns(r.status, r.error, r.error_class, None, hashes, times)
        hashes.append(result_hash(r.rows))
        times.append(r.elapsed_ms)
    assert first is not None
    return EngineRuns("SUCCESS", None, None, len(first.rows), hashes, times, first.rows, first.columns)


@dataclass
class CompatibilityRecord:
    case_id: str
    db: str
    gold_sql: str
    compatibility_status: str
    reason: str
    reference_status: str
    reference_error: str | None
    reference_row_count: int | None
    reference_result_hash: str | None
    reference_stable: bool
    execution_status: str
    execution_error: str | None
    execution_error_class: str | None
    execution_time_ms: int | None
    row_count: int | None
    result_hash: str | None
    databricks_stable: bool
    set_match: bool | None
    multiset_match: bool | None
    ordered_match: bool | None
    static_hazards: list[str]

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["static_hazards"] = ",".join(self.static_hazards)
        return d


def validate_case(case: BeaverCase, reference: SqlExecutor, candidate: SqlExecutor,
                  repeats: int = 3) -> tuple[CompatibilityRecord, EngineRuns, EngineRuns]:
    ref = run_repeated(reference, case.gold_sql, case.db, repeats)
    cand = run_repeated(candidate, case.gold_sql, case.db, repeats)
    cmp = cross_engine_match(cand.rows, ref.rows) if ref.status == cand.status == "SUCCESS" else None

    if ref.status != "SUCCESS":
        status, reason = REFERENCE_FAILED, f"gold SQL fails on reference engine: {ref.status} {ref.error_class}"
    elif cand.status == "TIMEOUT":
        status, reason = UNKNOWN, "timeout on Databricks"
    elif cand.status != "SUCCESS":
        status = classify_error(cand.error_class, cand.error)
        reason = f"{cand.status}: {cand.error_class or (cand.error or '')[:120]}"
    elif not ref.stable or not cand.stable:
        # A non-deterministic gold result cannot be a fixed ground truth.
        status, reason = INCOMPATIBLE_SEMANTICS, (
            f"non-deterministic result (mysql_stable={ref.stable}, databricks_stable={cand.stable})")
    elif cmp is not None and cmp.set_match:
        status, reason = COMPATIBLE, f"results match ({cmp.reason})"
    else:
        status, reason = INCOMPATIBLE_SEMANTICS, f"executes but result differs ({cmp.reason if cmp else '?'})"

    rec = CompatibilityRecord(
        case_id=case.case_id, db=case.db, gold_sql=case.gold_sql,
        compatibility_status=status, reason=reason,
        reference_status=ref.status, reference_error=ref.error, reference_row_count=ref.row_count,
        reference_result_hash=ref.result_hashes[0] if ref.result_hashes else None,
        reference_stable=ref.stable,
        execution_status=cand.status, execution_error=cand.error, execution_error_class=cand.error_class,
        execution_time_ms=int(sum(cand.elapsed_ms) / len(cand.elapsed_ms)) if cand.elapsed_ms else None,
        row_count=cand.row_count,
        result_hash=cand.result_hashes[0] if cand.result_hashes else None,
        databricks_stable=cand.stable,
        set_match=cmp.set_match if cmp else None,
        multiset_match=cmp.multiset_match if cmp else None,
        ordered_match=cmp.ordered_match if cmp else None,
        static_hazards=static_hazards(case.gold_sql),
    )
    return rec, ref, cand

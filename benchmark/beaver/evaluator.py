"""Result comparison.

Two comparators, used for different purposes:

* :func:`official_match` — a faithful port of BEAVER ``eval/utils/ex_acc.py``
  ``compare_results``: values → ``str`` → strip, compare the *set* of rows,
  column order matters, column names ignored, both-empty counts as a match.
  This is the Execution Accuracy metric for all later experiments, where gold
  and generated results come from the SAME engine (Databricks).

* :func:`canonical_match` — engine-neutral comparison used only in phase 0 to
  decide whether Databricks reproduces the MySQL (official engine) result.
  Drivers render the same value differently (``AVG`` → ``DECIMAL 12.5000`` in
  MySQL vs ``DOUBLE 12.5`` in Spark), so values are canonicalised first.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

Row = Sequence[Any]

NUMERIC_DIGITS = 6  # decimal places kept by the canonical form


# --------------------------------------------------------------------------- official


def _official_str(v: Any) -> str:
    # pandas ``astype(str)`` renders None as 'None' and NaN as 'nan'.
    return str(v).strip()


def official_match(pred: list[Row] | None, gold: list[Row] | None) -> tuple[bool, str]:
    pred_empty = not pred
    gold_empty = not gold
    if pred_empty and gold_empty:
        return True, "Both empty"
    if pred_empty or gold_empty:
        return False, "One is empty, other is not"
    pred_rows = [tuple(_official_str(v) for v in r) for r in pred]
    gold_rows = [tuple(_official_str(v) for v in r) for r in gold]
    if len(pred_rows[0]) != len(gold_rows[0]):
        return False, f"Column count mismatch: pred {len(pred_rows[0])} vs gold {len(gold_rows[0])}"
    if set(pred_rows) == set(gold_rows):
        return True, "Match (values match, ignoring column names)"
    return False, "Values mismatch"


def evaluate_against_gold(generated_rows: list[Row] | None, gold_result_json: str) -> tuple[bool, str]:
    """Execution accuracy of generated SQL against a frozen gold result.

    Frozen gold results are stored as canonical rows (``serialize_rows``), so the
    generated rows are canonicalised the same way before BEAVER's official
    comparison — otherwise e.g. ``Decimal('1.50')`` vs ``'1.5'`` would differ
    only by formatting. Both sides come from the same engine (Databricks).
    """
    gold = [tuple(r) for r in json.loads(gold_result_json)]
    return official_match(canonical_rows(generated_rows), gold)


# --------------------------------------------------------------------------- canonical


def canonical_value(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        v = Decimal(repr(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, Decimal):
        try:
            q = v.quantize(Decimal(1).scaleb(-NUMERIC_DIGITS))
        except InvalidOperation:  # too many digits to quantize — keep as is
            q = v
        s = format(q.normalize(), "f")
        return "0" if s in ("-0", "0") else s
    if isinstance(v, dt.datetime):
        return v.replace(tzinfo=None).isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, dt.timedelta):  # MySQL TIME
        total = int(v.total_seconds())
        sign = "-" if total < 0 else ""
        total = abs(total)
        return f"{sign}{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    return str(v).strip()


def canonical_rows(rows: list[Row] | None) -> list[tuple[str | None, ...]]:
    return [tuple(canonical_value(v) for v in r) for r in (rows or [])]


@dataclass(frozen=True)
class CanonicalComparison:
    set_match: bool
    multiset_match: bool
    ordered_match: bool
    reason: str


def canonical_match(a: list[Row] | None, b: list[Row] | None) -> CanonicalComparison:
    ca, cb = canonical_rows(a), canonical_rows(b)
    if not ca and not cb:
        return CanonicalComparison(True, True, True, "Both empty")
    if not ca or not cb:
        return CanonicalComparison(False, False, False, "One is empty, other is not")
    if len(ca[0]) != len(cb[0]):
        return CanonicalComparison(False, False, False, f"Column count mismatch: {len(ca[0])} vs {len(cb[0])}")
    s = set(ca) == set(cb)
    m = Counter(ca) == Counter(cb)
    o = ca == cb
    reason = "match" if m else ("set match, duplicate counts differ" if s else "values differ")
    return CanonicalComparison(s, m, o, reason)


# --------------------------------------------------------------------------- cross-engine

FLOAT_REL_TOL = 1e-6
CROSS_ENGINE_RULE = ("cross_engine_v2: DECIMAL values compared at their own displayed scale "
                     f"(MySQL AVG/division keep 4 decimals); float vs float rel tol {FLOAT_REL_TOL:g}; "
                     "strings exact after strip; NULL only equals NULL")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def _scale(d: Decimal) -> int:
    exp_ = d.as_tuple().exponent
    return -exp_ if isinstance(exp_, int) and exp_ < 0 else 0


def cross_engine_value_equal(a: Any, b: Any) -> bool:
    """Same value, allowing for how each engine *represents* numbers.

    A DECIMAL carries the precision its engine computed it at (MySQL returns
    ``AVG(int)`` as ``6.8333``). The other engine's value is equal if it rounds
    to that DECIMAL at that scale — i.e. |a-b| <= half a unit in the last place.
    """
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        a, b = (int(a) if isinstance(a, bool) else a), (int(b) if isinstance(b, bool) else b)
    if _is_num(a) and _is_num(b):
        if isinstance(a, float) and math.isnan(a) or isinstance(b, float) and math.isnan(b):
            return isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        scales = [_scale(x) for x in (a, b) if isinstance(x, Decimal)]
        if scales:
            half_ulp = Decimal(5).scaleb(-min(scales) - 1)
            da = a if isinstance(a, Decimal) else Decimal(repr(a)) if isinstance(a, float) else Decimal(a)
            db = b if isinstance(b, Decimal) else Decimal(repr(b)) if isinstance(b, float) else Decimal(b)
            return abs(da - db) <= half_ulp * (1 + Decimal("1e-9"))
        return math.isclose(float(a), float(b), rel_tol=FLOAT_REL_TOL, abs_tol=1e-9)
    return canonical_value(a) == canonical_value(b)


def _row_equal(r: Row, s: Row) -> bool:
    return len(r) == len(s) and all(cross_engine_value_equal(x, y) for x, y in zip(r, s))


def _bucket_key(r: Row) -> tuple:
    # non-numeric values must match exactly, so they are a safe bucketing key
    return tuple(None if _is_num(v) else canonical_value(v) for v in r)


GREEDY_LIMIT = 200  # bucket size up to which rows are paired by exhaustive search


def _sort_key(r: Row) -> tuple:
    return tuple((0, "") if v is None else (1, float(v), "") if _is_num(v) else (2, 0.0, canonical_value(v))
                 for v in r)


def _buckets(rows: list[Row]) -> dict[tuple, list[Row]]:
    out: dict[tuple, list[Row]] = {}
    for r in rows:
        out.setdefault(_bucket_key(r), []).append(r)
    return out


def _bucket_equal(x: list[Row], y: list[Row]) -> bool:
    if len(x) != len(y):
        return False
    if len(x) <= GREEDY_LIMIT:
        pool = list(y)
        for r in x:
            hit = next((i for i, c in enumerate(pool) if _row_equal(r, c)), None)
            if hit is None:
                return False
            pool.pop(hit)
        return True
    # Large bucket: rows that differ only by numeric noise sort into the same
    # position, so pairwise comparison after sorting is O(n log n).
    return all(_row_equal(r, s) for r, s in zip(sorted(x, key=_sort_key), sorted(y, key=_sort_key)))


def _multiset_equal(a: list[Row], b: list[Row]) -> bool:
    if len(a) != len(b):
        return False
    ba, bb = _buckets(a), _buckets(b)
    return ba.keys() == bb.keys() and all(_bucket_equal(ba[k], bb[k]) for k in ba)


def _dedupe(rows: list[Row]) -> list[Row]:
    out: list[Row] = []
    for bucket in _buckets(rows).values():
        kept: list[Row] = []
        for r in sorted(bucket, key=_sort_key):
            if not (kept and _row_equal(r, kept[-1])):
                kept.append(r)
        out.extend(kept)
    return out


def cross_engine_match(candidate: list[Row] | None, reference: list[Row] | None) -> CanonicalComparison:
    """Compare results of the same SQL on two engines (Phase 0 qualification only)."""
    a, b = [tuple(r) for r in (candidate or [])], [tuple(r) for r in (reference or [])]
    if not a and not b:
        return CanonicalComparison(True, True, True, "Both empty")
    if not a or not b:
        return CanonicalComparison(False, False, False, "One is empty, other is not")
    if len(a[0]) != len(b[0]):
        return CanonicalComparison(False, False, False, f"Column count mismatch: {len(a[0])} vs {len(b[0])}")
    m = _multiset_equal(a, b)
    s = m or _multiset_equal(_dedupe(a), _dedupe(b))
    o = len(a) == len(b) and all(_row_equal(x, y) for x, y in zip(a, b))
    reason = "match" if m else ("set match, duplicate counts differ" if s else "values differ")
    return CanonicalComparison(s, m, o, reason)


def result_hash(rows: list[Row] | None) -> str:
    """Order- and duplicate-insensitive fingerprint (official EX semantics)."""
    canon = sorted(set(canonical_rows(rows)), key=lambda r: json.dumps(r))
    payload = json.dumps(canon, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_rows(rows: list[Row] | None) -> str:
    """Canonical JSON, lossless enough to recompute the hash and re-run official_match."""
    return json.dumps(canonical_rows(rows), ensure_ascii=False)

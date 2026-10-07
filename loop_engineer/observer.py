"""Observer — assembles what the loop may look at after a failed attempt.

The observation is built from a trace / attempt record through an explicit
whitelist, so gold-derived fields (correctness, eval_* diagnostics, gold SQL or
annotations) can never reach Diagnosis or Repair, even if the source record
carries them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Everything an agent legitimately observes about its own attempt.
OBSERVABLE_FIELDS = ("case_id", "attempt_id", "question", "retrieved_tables", "generated_sql", "parse_status",
                     "execution_status", "execution_error", "result_row_count", "result_preview",
                     # the self-verifier's own findings about this attempt (gold-free, see verifier.py)
                     "verifier_signals", "verifier_findings")

_ERROR_CLASS = re.compile(r"\[([A-Z][A-Z0-9_]+)(?:\.[A-Z0-9_]+)?\]")
_UNRESOLVED = re.compile(r"name `([^`]+)`(?:\.`([^`]+)`)? cannot be resolved")
_SUGGEST = re.compile(r"Did you mean one of the following\? \[([^\]]*)\]")
# The whole (possibly qualified) reference: `fac_building`, `dw`.`school`, dw.school, `cat`.`dw`.`school`
_TABLE_NOT_FOUND = re.compile(r"The table or view (\S+) cannot be found|TABLE_OR_VIEW_NOT_FOUND\][^`]*((?:`[^`]+`\.?)+)")
_NAME_PART = re.compile(r"`([^`]+)`|([^.`]+)")


def table_name(ref: str | None) -> str | None:
    """Last part of a table reference, without backticks: `dw`.`school` -> school (the table, not the schema)."""
    parts = [a or b for a, b in _NAME_PART.findall(ref or "") if (a or b).strip()]
    return parts[-1].strip() if parts else None


@dataclass(frozen=True)
class Observation:
    case_id: str
    attempt_id: int
    question: str
    db: str
    retrieved_tables: tuple[str, ...]
    generated_sql: str
    parse_status: str
    execution_status: str
    execution_error: str | None
    result_row_count: int | None
    result_preview: list | None
    # parsed signals (derived from the error text only)
    error_class: str | None = None
    unresolved_qualifier: str | None = None
    unresolved_column: str | None = None
    suggestions: tuple[str, ...] = field(default_factory=tuple)
    missing_table: str | None = None
    # what the self-verifier found on an attempt that ran (e.g. join_tautology), with evidence and repair hints
    verifier_signals: tuple[str, ...] = field(default_factory=tuple)
    verifier_findings: tuple[dict, ...] = field(default_factory=tuple)


def observe(record: dict[str, Any], db: str) -> Observation:
    r = {k: record.get(k) for k in OBSERVABLE_FIELDS}  # whitelist: nothing else passes
    err = r["execution_error"] or ""
    m_cls = _ERROR_CLASS.search(err)
    m_col = _UNRESOLVED.search(err)
    qualifier, column = (None, None)
    if m_col:
        qualifier, column = (m_col.group(1), m_col.group(2)) if m_col.group(2) else (None, m_col.group(1))
    m_sug = _SUGGEST.search(err)
    suggestions = tuple(s.strip().strip("`").replace("`.`", ".") for s in m_sug.group(1).split(",")) if m_sug else ()
    m_tab = _TABLE_NOT_FOUND.search(err)
    preview = json.loads(r["result_preview"]) if isinstance(r["result_preview"], str) and r["result_preview"] else None
    return Observation(
        case_id=r["case_id"], attempt_id=int(r["attempt_id"] or 1), question=r["question"] or "", db=db,
        retrieved_tables=tuple(r["retrieved_tables"] or ()), generated_sql=r["generated_sql"] or "",
        parse_status=r["parse_status"] or "", execution_status=r["execution_status"] or "",
        execution_error=r["execution_error"], result_row_count=r["result_row_count"], result_preview=preview,
        error_class=m_cls.group(1) if m_cls else ("NO_SQL" if r["parse_status"] in ("NO_SQL", "EMPTY_RESPONSE") else None),
        unresolved_qualifier=qualifier, unresolved_column=column, suggestions=suggestions,
        missing_table=table_name(m_tab.group(1) or m_tab.group(2)) if m_tab else None,
        verifier_signals=tuple(r["verifier_signals"] or ()),
        verifier_findings=tuple(json.loads(r["verifier_findings"]) if isinstance(r["verifier_findings"], str)
                                else (r["verifier_findings"] or ())),
    )

"""RepairSQL — EXECUTION_FAILURE (and the fallback for UNKNOWN).

Fixes the reported engine error with minimal changes, with notes on Databricks
SQL restrictions that the baseline hit (e.g. DISTINCT inside window functions).

Collation mismatch in UNION (diagnosis signal ``collation_mismatch``) is fixed
deterministically first: ``CAST(NULL AS STRING)`` -> ``NULL``, which takes the
column's type and collation from the other branch. Anything left (string
literals, ``CAST(x AS STRING)``) goes to the LLM with an explicit instruction.
"""
from __future__ import annotations

import re

from agent.sql_analysis import alias_map
from loop_engineer.diagnose import Diagnosis
from loop_engineer.observer import Observation
from skills.base import RepairContext, RepairResult, llm_repair

DATABRICKS_NOTES = ("Databricks SQL notes: clauses must be in the order SELECT, FROM/JOIN, WHERE, GROUP BY, HAVING, "
                    "ORDER BY, LIMIT; DISTINCT is not allowed inside window functions; every non-aggregated SELECT "
                    "column must be in GROUP BY; ranking functions (RANK, ROW_NUMBER, DENSE_RANK, LAG, LEAD) take no "
                    "ROWS/RANGE frame; use STDDEV_POP / VAR_POP for standard deviation / variance; string columns "
                    "have the UTF8_LCASE collation, so in UNION fill a missing column with a bare NULL (not "
                    "CAST(NULL AS STRING)) and write a string literal or CAST(... AS STRING) that sits opposite a "
                    "string column as <expression> COLLATE UTF8_LCASE.")

_NULL_AS_STRING = re.compile(r"CAST\s*\(\s*NULL\s+AS\s+STRING\s*\)", re.IGNORECASE)
COLLATION_INSTRUCTION = ("The UNION fails because a plain STRING value is combined with a UTF8_LCASE string column. "
                         "Keep the query as it is; only make the UNION branches type-compatible: use a bare NULL for "
                         "missing columns and append COLLATE UTF8_LCASE to string literals or CAST(... AS STRING) "
                         "values that sit opposite a string column.")


def fix_null_casts(sql: str) -> tuple[str, int]:
    """CAST(NULL AS STRING) -> NULL (a bare NULL adopts the other branch's type and collation)."""
    return _NULL_AS_STRING.subn("NULL", sql)


class RepairSQL:
    name = "RepairSQL"

    def repair(self, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext) -> RepairResult:
        used = sorted(set(alias_map(obs.generated_sql).values()) & set(ctx.catalog.tables))
        if diagnosis.repair_hints.get("signal") == "collation_mismatch":
            fixed, n = fix_null_casts(obs.generated_sql)
            if n:  # deterministic; if a literal is still incompatible, the next round goes to the LLM
                return RepairResult(fixed, self.name, f"replaced {n} CAST(NULL AS STRING) with NULL (deterministic)",
                                    diagnosis.reason, tuple(used), used_llm=False,
                                    details={"deterministic_changes": [f"CAST(NULL AS STRING) -> NULL x{n}"]})
            return llm_repair(self.name, obs, diagnosis, ctx, [*used, *obs.retrieved_tables],
                              COLLATION_INSTRUCTION, "fix UNION collation mismatch")
        instruction = ("Fix the error with the smallest change that keeps the intended meaning. " + DATABRICKS_NOTES)
        return llm_repair(self.name, obs, diagnosis, ctx, [*used, *obs.retrieved_tables], instruction,
                          f"fix engine error {obs.error_class or obs.execution_status}")

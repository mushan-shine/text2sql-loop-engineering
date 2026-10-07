"""RepairSQL — EXECUTION_FAILURE (and the fallback for UNKNOWN).

Fixes the reported engine error with minimal changes, with notes on Databricks
SQL restrictions that the baseline hit (e.g. DISTINCT inside window functions).
"""
from __future__ import annotations

from agent.sql_analysis import alias_map
from loop_engineer.diagnose import Diagnosis
from loop_engineer.observer import Observation
from skills.base import RepairContext, RepairResult, llm_repair

DATABRICKS_NOTES = ("Databricks SQL notes: clauses must be in the order SELECT, FROM/JOIN, WHERE, GROUP BY, HAVING, "
                    "ORDER BY, LIMIT; DISTINCT is not allowed inside window functions; every non-aggregated SELECT "
                    "column must be in GROUP BY; ranking functions (RANK, ROW_NUMBER, DENSE_RANK, LAG, LEAD) take no "
                    "ROWS/RANGE frame; use STDDEV_POP / VAR_POP for standard deviation / variance.")


class RepairSQL:
    name = "RepairSQL"

    def repair(self, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext) -> RepairResult:
        used = sorted(set(alias_map(obs.generated_sql).values()) & set(ctx.catalog.tables))
        instruction = ("Fix the error with the smallest change that keeps the intended meaning. " + DATABRICKS_NOTES)
        return llm_repair(self.name, obs, diagnosis, ctx, [*used, *obs.retrieved_tables], instruction,
                          f"fix engine error {obs.error_class or obs.execution_status}")

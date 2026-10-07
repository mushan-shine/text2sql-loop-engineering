"""FindJoinPath — JOIN_KEY_FAILURE.

Offers the key-like columns shared by the tables in the query (schema-inferred,
never BEAVER join_keys) and asks the LLM to re-check every join condition.
"""
from __future__ import annotations

from agent.join_graph import join_candidates
from agent.sql_analysis import alias_map
from loop_engineer.diagnose import Diagnosis
from loop_engineer.observer import Observation
from skills.base import RepairContext, RepairResult, llm_repair


class FindJoinPath:
    name = "FindJoinPath"

    def repair(self, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext) -> RepairResult:
        used = sorted(set(alias_map(obs.generated_sql).values()) & set(ctx.catalog.tables))
        joins = [j.sql() for j in join_candidates(ctx.catalog, used)][:12]
        instruction = ("The join conditions are probably wrong (rows are multiplied or lost). Key columns shared by "
                       f"the tables in the query: {'; '.join(joins) or '(none found)'}. Re-check every JOIN ... ON "
                       "condition, join on matching keys only, and aggregate before joining when a join would "
                       "duplicate rows.")
        return llm_repair(self.name, obs, diagnosis, ctx, [*used, *obs.retrieved_tables], instruction,
                          f"join candidates offered: {len(joins)}", {"tables_in_sql": used, "join_candidates": joins})

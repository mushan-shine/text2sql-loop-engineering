"""ReplanQuery — QUERY_DECOMPOSITION_FAILURE (and, until a non-gold knowledge
source exists, DOMAIN_KNOWLEDGE_FAILURE — see skills/retrieve_knowledge.py).

Asks the LLM to decompose the question into sub-questions, solve each as a CTE
and check every requested output, filter, grouping and ordering.
"""
from __future__ import annotations

from agent.sql_analysis import alias_map
from loop_engineer.diagnose import Diagnosis
from loop_engineer.observer import Observation
from skills.base import RepairContext, RepairResult, llm_repair

INSTRUCTION = ("Re-plan the query: (1) list the sub-questions the question contains as SQL comments at the top; "
               "(2) answer each sub-question in its own CTE at the right granularity; (3) combine the CTEs; "
               "(4) check that every output column, filter, grouping, ranking and ordering the question asks for "
               "is present, in the requested column order.")


class ReplanQuery:
    name = "ReplanQuery"

    def repair(self, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext) -> RepairResult:
        used = sorted(set(alias_map(obs.generated_sql).values()) & set(ctx.catalog.tables))
        return llm_repair(self.name, obs, diagnosis, ctx, [*used, *obs.retrieved_tables], INSTRUCTION,
                          "decompose into sub-questions / CTEs")

"""RetrieveAgain — TABLE_RETRIEVAL_FAILURE.

Phase 3/4 data: most missing tables were *retrieved but not used* (132 vs 32
not retrieved on the evaluation baseline), so this skill does more than
re-retrieval:
* table not used / not retrieved: the tables that own the missing column are
  added to the schema, with schema-inferred join candidates, and the LLM is told
  to join them;
* table not found: similarly named real tables are offered instead.
"""
from __future__ import annotations

import difflib

from agent.join_graph import connect
from agent.sql_analysis import alias_map
from loop_engineer.diagnose import Diagnosis
from loop_engineer.observer import Observation
from skills.base import RepairContext, RepairResult, as_hints, llm_repair


class RetrieveAgain:
    name = "RetrieveAgain"

    def repair(self, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext) -> RepairResult:
        h = as_hints(diagnosis)
        used = sorted(set(alias_map(obs.generated_sql).values()) & set(ctx.catalog.tables))
        joins: list[str] = []
        if h.get("signal") == "table_not_found":
            missing = (h.get("missing_table") or "").split(".")[-1]
            add = difflib.get_close_matches(missing.lower(), list(ctx.catalog.tables), n=4, cutoff=0.4)
            instruction = (f"Table {missing} does not exist. Similar existing tables: {', '.join(add) or '(none)'}. "
                           "Use only tables from the schema.")
            action = f"offered existing tables similar to {missing}: {add}"
        else:
            add = list(h.get("tables_to_add") or h.get("owner_tables") or [])[:4]
            joins = [j.sql() for t in add for j in connect(ctx.catalog, t, used)][:8]
            instruction = (f"Column {h.get('column')} is not in the table you used; it lives in {', '.join(add)}. "
                           f"Join the appropriate one of these tables. Join candidates inferred from the schema: "
                           f"{'; '.join(joins) or '(none found - choose matching key columns)'}. "
                           "Check that every column the question needs comes from a table that really has it.")
            action = f"added tables {add} ({h.get('case')}) with join candidates"
        return llm_repair(self.name, obs, diagnosis, ctx, [*used, *add, *obs.retrieved_tables], instruction, action,
                          {"tables_added": add, "tables_in_sql": used, "join_candidates": joins})

"""Repair skill contract.

A skill receives the observation of the failed attempt, the diagnosis and a
RepairContext (agent-visible resources only: schema catalog, LLM client,
few-shot examples). It returns a RepairResult with the repaired SQL.

Skills never read gold information: they import nothing from ``benchmark``
or ``evaluation`` (enforced by tests/test_phase5.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.generator import RULES, SYSTEM, FewShotExample, extract_sql, render_schema
from agent.llm import ChatClient
from agent.retriever import SchemaCatalog
from loop_engineer.diagnose import Diagnosis
from loop_engineer.observer import Observation


@dataclass
class RepairContext:
    catalog: SchemaCatalog
    client: ChatClient
    examples: list[FewShotExample] = field(default_factory=list)
    max_schema_tables: int = 24


@dataclass(frozen=True)
class RepairResult:
    repaired_sql: str
    repair_skill: str
    repair_action: str          # what the skill did, for the trace / debug console
    repair_reason: str          # why (from the diagnosis)
    tables: tuple[str, ...]     # schema shown to the model (or used by a deterministic fix)
    used_llm: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    parse_status: str = "OK"
    # what the skill did internally, for live UIs / reports (agent-visible data only):
    # deterministic_changes, unresolved, candidate_columns, join_candidates, instruction, prompt, ...
    details: dict[str, Any] = field(default_factory=dict)


class RepairSkill(Protocol):
    name: str

    def repair(self, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext) -> RepairResult: ...


REPAIR_TEMPLATE = """{rules}

Schema:
{schema}

Question: {question}

A previous attempt produced this SQL:
```sql
{sql}
```
{observed}

Diagnosis: {diagnosis}

{instruction}

Return the corrected query as ONE read-only SQL query inside a ```sql code fence. No explanation."""


def observed_text(obs: Observation) -> str:
    if obs.execution_status == "SUCCESS":
        found = [f.get("hint") or f.get("message") for f in obs.verifier_findings if f.get("hint") or f.get("message")]
        if found:
            return (f"It ran and returned {obs.result_row_count} row(s), but a check of the SQL against the question "
                    "found:\n- " + "\n- ".join(found))
        return f"It ran and returned {obs.result_row_count} row(s); the result is believed to be wrong."
    if obs.execution_error:
        return f"Executing it failed with:\n{obs.execution_error[:1200]}"
    return f"It could not be executed ({obs.execution_status})."


def llm_repair(skill: str, obs: Observation, diagnosis: Diagnosis, ctx: RepairContext, tables: list[str],
               instruction: str, action: str, details: dict[str, Any] | None = None) -> RepairResult:
    tables = [t for t in dict.fromkeys(tables) if t in ctx.catalog.tables][: ctx.max_schema_tables]
    prompt = REPAIR_TEMPLATE.format(rules=RULES, schema=render_schema(ctx.catalog, tuple(tables)),
                                    question=obs.question, sql=obs.generated_sql or "(no SQL was produced)",
                                    observed=observed_text(obs),
                                    diagnosis=f"{diagnosis.failure_type}: {diagnosis.reason}",
                                    instruction=instruction)
    r = ctx.client.complete(prompt, system=SYSTEM)
    sql, status = extract_sql(r.text)
    return RepairResult(sql, skill, action, diagnosis.reason, tuple(tables), True,
                        r.input_tokens, r.output_tokens, r.latency_ms, status,
                        {**(details or {}), "instruction": instruction, "schema_tables": list(tables),
                         "start_sql": obs.generated_sql, "prompt": prompt, "raw_response": r.text[:4000],
                         "cached": r.cached})


def as_hints(diagnosis: Diagnosis) -> dict[str, Any]:
    return diagnosis.repair_hints or {}

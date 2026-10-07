"""Loop Controller — Generate -> Execute -> Verify -> (Observe -> Diagnose -> Policy -> Repair) -> Execute -> Verify.

One controller drives both experiment arms; they differ only in how the next
attempt is produced after a failed verification:

* ``targeted`` — Observer -> Diagnoser -> Policy -> Repair Skill;
* ``generic``  — Generic Retry: the previous SQL and what was observed go back
  to the model with a generic "correct it" instruction. No diagnosis, no
  targeted schema information. Same verifier, same attempt budget.

Attempt 1 is identical across arms (same prompt; the LLM cache replays it).
Budget = ``max_attempts`` SQL attempts; LLM calls made by diagnosis are not
attempts but are counted as cost.

Final answer: the last attempt that passed verification; otherwise the last
attempt that executed; otherwise the last attempt.

``run(..., on_event=cb)`` reports every step as ``cb(step, payload)`` (retrieve,
generate, execute, verify, observe, diagnose, route, repair, final) for live UIs;
``repair`` carries the skill's internal ``details`` (deterministic edits, the
instruction and prompt given to the LLM). It only
observes: payloads are copies of what goes into the trace anyway, and a failing
callback never breaks the loop.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agent.generator import RULES, SYSTEM, extract_sql, render_schema
from agent.retriever import BM25TableRetriever
from benchmark.beaver.dataset import AgentTask
from loop_engineer.diagnose import Diagnoser
from loop_engineer.observer import OBSERVABLE_FIELDS, observe
from loop_engineer.policy import Policy
from skills.base import REPAIR_TEMPLATE, RepairContext, observed_text

log = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]

GENERIC_INSTRUCTION = "The query above is wrong. Write a corrected query."


@dataclass
class LoopConfig:
    strategy: str = "targeted"      # targeted | generic
    max_attempts: int = 2
    top_k: int = 20
    max_result_rows: int = 500_000


def _same_sql(a: str | None, b: str | None) -> bool:
    """Equal up to whitespace and a trailing semicolon: re-executing it cannot change the outcome."""
    norm = lambda x: " ".join((x or "").split()).rstrip(";").strip()
    return norm(a) == norm(b)


@dataclass
class LoopResult:
    attempts: list[dict[str, Any]]
    rows: list[list[tuple]] = field(repr=False)
    final_index: int = 0

    @property
    def final(self) -> dict[str, Any]:
        return self.attempts[self.final_index]


def _serialize_preview(rows: list[tuple]) -> str:
    from benchmark.beaver.evaluator import serialize_rows  # agent-visible formatting only (no gold)
    return serialize_rows(rows[:5])


@dataclass
class LoopController:
    retriever: BM25TableRetriever
    generator: Any                  # FewShotGenerator
    executor: Any
    diagnoser: Diagnoser
    policy: Policy
    ctx: RepairContext
    cfg: LoopConfig = field(default_factory=LoopConfig)

    def _execute(self, sql: str, db: str, parse_status: str) -> tuple[dict[str, Any], list[tuple]]:
        if not sql:
            return {"execution_status": parse_status, "execution_error": None, "result_row_count": None,
                    "result_preview": None, "exec_latency_ms": 0}, []
        ex = self.executor.execute(sql, db, max_rows=self.cfg.max_result_rows)
        rows = ex.rows if ex.ok else []
        return {"execution_status": ex.status, "execution_error": (ex.error or "")[:2000] or None,
                "result_row_count": len(rows) if ex.ok else None,
                "result_preview": _serialize_preview(rows) if ex.ok else None, "exec_latency_ms": ex.elapsed_ms}, rows

    def _generic_retry(self, attempt: dict[str, Any], tables: tuple[str, ...]) -> dict[str, Any]:
        obs = observe(attempt, "")
        prompt = REPAIR_TEMPLATE.format(rules=RULES,
                                        schema=render_schema(self.ctx.catalog, tables), question=obs.question,
                                        sql=obs.generated_sql or "(no SQL was produced)", observed=observed_text(obs),
                                        diagnosis="(not diagnosed)", instruction=GENERIC_INSTRUCTION)
        r = self.ctx.client.complete(prompt, system=SYSTEM)
        sql, status = extract_sql(r.text)
        self._last_details = {"instruction": GENERIC_INSTRUCTION, "schema_tables": list(tables),
                              "start_sql": obs.generated_sql, "prompt": prompt, "raw_response": r.text[:4000],
                              "cached": r.cached}
        return {"generated_sql": sql, "parse_status": status, "repair_skill": "GenericRetry",
                "repair_action": "generic correction prompt", "repair_reason": None, "repair_fallback": False,
                "used_llm": True, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "llm_latency_ms": r.latency_ms, "tables": list(tables)}

    def run(self, task: AgentTask, verifier: Any, on_event: EventCallback | None = None) -> LoopResult:
        def emit(step: str, **payload: Any) -> None:
            if on_event is None:
                return
            try:
                on_event(step, payload)
            except Exception:  # an observer must never break the loop
                log.exception("on_event(%s) failed", step)

        # 检索与问题相关的候选表 BM25
        retrieval = self.retriever.retrieve(task.question, self.cfg.top_k)
        emit("retrieve", tables=list(retrieval.tables), scores=list(retrieval.scores))
        # 生成SQL语句
        gen = self.generator.generate(task, retrieval.tables)
        # dynamic few-shot may add the tables of similar solved questions to the schema the model saw
        shown = list(getattr(gen, "schema_tables", ()) or retrieval.tables)
        attempt = {"case_id": task.case_id, "attempt_id": 1, "question": task.question,
                   "retrieved_tables": shown, "generated_sql": gen.sql,
                   "parse_status": gen.parse_status, "strategy": self.cfg.strategy,
                   "input_tokens": gen.llm.input_tokens, "output_tokens": gen.llm.output_tokens,
                   "llm_latency_ms": gen.llm.latency_ms, "used_llm": True, "diag_tokens": 0}
        emit("generate", attempt_id=1, sql=gen.sql, parse_status=gen.parse_status,
             tokens=gen.llm.input_tokens + gen.llm.output_tokens, input_tokens=gen.llm.input_tokens,
             output_tokens=gen.llm.output_tokens, latency_ms=gen.llm.latency_ms, cached=gen.llm.cached,
             schema_tables=len(shown), added_tables=[t for t in shown if t not in retrieval.tables],
             few_shot=len(getattr(gen, "example_ids", ()) or getattr(self.generator, "examples", []) or []),
             example_ids=list(getattr(gen, "example_ids", ()) or ()),
             prompt_version=getattr(self.generator, "prompt_version", None),
             knowledge_notes=getattr(gen, "notes", "") or "",
             prompt=gen.prompt, raw_response=gen.raw[:4000])
        attempts, all_rows = [], []
        # 开始尝试运行SQL
        for n in range(1, self.cfg.max_attempts + 1):
            # 执行SQL语句，exe返回的是摘要，rows返回的是原始全文
            exe, rows = self._execute(attempt["generated_sql"], task.db, attempt["parse_status"])
            attempt.update(exe)
            emit("execute", attempt_id=n, sql=attempt["generated_sql"], **exe)
            decision = verifier.verify(attempt, rows)
            findings = [asdict(f) for f in getattr(decision, "findings", ())]
            advisories = [asdict(f) for f in getattr(decision, "advisories", ())]
            attempt.update({"verifier_mode": decision.mode, "verifier_decision": "PASS" if decision.passed else "FAIL",
                            "verifier_signals": list(decision.signals),
                            "verifier_findings": json.dumps(findings, ensure_ascii=False) if findings else None,
                            "verifier_advisories": json.dumps(advisories, ensure_ascii=False) if advisories else None})
            emit("verify", attempt_id=n, mode=decision.mode, passed=decision.passed, signals=list(decision.signals),
                 findings=findings, advisories=advisories, last=n == self.cfg.max_attempts)
            attempts.append(attempt)
            all_rows.append(rows)
            # 验证pass或者优化次数已经达到上限，就跳出循环
            if decision.passed or n == self.cfg.max_attempts:
                break
            # ---- produce the next attempt
            nxt: dict[str, Any] = {"case_id": task.case_id, "attempt_id": n + 1, "question": task.question,
                                   "retrieved_tables": attempt["retrieved_tables"], "strategy": self.cfg.strategy,
                                   "diag_tokens": 0}
            details: dict[str, Any] = {}
            if self.cfg.strategy == "generic":
                nxt.update(self._generic_retry(attempt, tuple(attempt["retrieved_tables"])))
                details = self._last_details
            else:
                # 验证不通过的情况，对失败情况进行观察、诊断、修复
                # 1. 观察
                obs = observe(attempt, task.db)   # whitelist: nothing gold-derived reaches diagnosis / repair
                emit("observe", attempt_id=n, fields=list(OBSERVABLE_FIELDS), execution_status=obs.execution_status,
                     error_class=obs.error_class, unresolved_qualifier=obs.unresolved_qualifier,
                     unresolved_column=obs.unresolved_column, suggestions=list(obs.suggestions),
                     missing_table=obs.missing_table, result_row_count=obs.result_row_count,
                     retrieved_tables=len(obs.retrieved_tables), verifier_signals=list(obs.verifier_signals))
                # 2. 诊断
                diag, usage = self.diagnoser.diagnose(obs)
                emit("diagnose", attempt_id=n, failure_type=diag.failure_type, confidence=diag.confidence,
                     reason=diag.reason, source=diag.source, repair_hints=diag.repair_hints,
                     error_class=obs.error_class, unresolved_column=obs.unresolved_column)
                # 3. 路由
                route = self.policy.route(diag)
                emit("route", attempt_id=n, skill=route.skill, fallback=route.fallback, reason=route.reason)
                # 4. 执行skill
                res = self.policy.skill(route.skill).repair(obs, diag, self.ctx)
                attempt.update({"failure_type": diag.failure_type, "diagnosis_confidence": diag.confidence,
                                "diagnosis_reason": diag.reason, "diagnosis_source": diag.source,
                                "repair_hints": json.dumps(diag.repair_hints, ensure_ascii=False)})
                nxt.update({"generated_sql": res.repaired_sql, "parse_status": res.parse_status,
                            "repair_skill": route.skill, "repair_action": res.repair_action,
                            "repair_reason": res.repair_reason, "repair_fallback": route.fallback,
                            "used_llm": res.used_llm, "input_tokens": res.input_tokens,
                            "output_tokens": res.output_tokens, "llm_latency_ms": res.latency_ms,
                            "diag_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                            "tables": list(res.tables)})
                details = res.details
            attempt["repaired_sql"] = nxt["generated_sql"]
            attempt["repair_skill"] = nxt["repair_skill"]
            emit("repair", attempt_id=n + 1, skill=nxt["repair_skill"], action=nxt.get("repair_action"),
                 used_llm=nxt.get("used_llm"), before_sql=attempt["generated_sql"], sql=nxt["generated_sql"],
                 parse_status=nxt["parse_status"], latency_ms=nxt.get("llm_latency_ms") or 0,
                 diag_tokens=nxt.get("diag_tokens") or 0,
                 tokens=int(nxt.get("input_tokens") or 0) + int(nxt.get("output_tokens") or 0), details=details)
            # the repair reproduced a SQL that already ran: executing it again gives the same result, so stop
            repeat = next((a["attempt_id"] for a in attempts if _same_sql(a["generated_sql"], nxt["generated_sql"])), None)
            if repeat is not None:
                attempt["stop_reason"] = f"repair repeated the SQL of attempt {repeat}"
                emit("stop", attempt_id=n, repeated_attempt=repeat, reason=attempt["stop_reason"])
                break
            attempt = nxt
        passed = [i for i, a in enumerate(attempts) if a["verifier_decision"] == "PASS"]
        executed = [i for i, a in enumerate(attempts) if a["execution_status"] == "SUCCESS"]
        final = passed[-1] if passed else executed[-1] if executed else len(attempts) - 1
        for i, a in enumerate(attempts):
            a["final_status"] = "FINAL" if i == final else "SUPERSEDED"
        emit("final", final_attempt=final + 1, attempts=len(attempts),
             verifier_decision=attempts[final]["verifier_decision"],
             execution_status=attempts[final]["execution_status"], rows=attempts[final].get("result_row_count"),
             preview=attempts[final].get("result_preview"))
        return LoopResult(attempts, all_rows, final)

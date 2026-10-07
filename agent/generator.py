"""Few-shot Text-to-SQL generator — the baseline has exactly one attempt.

Prompt = rules + retrieved schema + fixed few-shot examples + question.

Two rules are *evaluation-environment conventions*, not gold information:
* target dialect is Databricks SQL and string columns compare case-insensitively
  (UTF8_LCASE, mirroring MySQL ``_ci`` collations);
* BEAVER questions were written for MySQL and often name MySQL functions
  ("using STDDEV only and never STDDEV_POP": 26/89 evaluation questions,
  2,116/5,787 dw questions). MySQL STDDEV/STD/VARIANCE are population
  statistics, Databricks stddev/variance are sample statistics, so the prompt
  maps the MySQL names to VAR_POP / STDDEV_POP (baseline-v2; v1 only said
  "use STDDEV_POP", which contradicted such questions).

Few-shot examples come from dw questions outside the evaluation sample; their
gold SQL is MySQL, so the phase-0 adapter rules are applied before use.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

import sqlglot

from agent.llm import ChatClient, LlmResponse
from agent.retriever import SchemaCatalog
from benchmark.beaver.adapter import apply_rules
from benchmark.beaver.dataset import AgentTask

PROMPT_VERSION = "baseline-v2"

SYSTEM = ("You are an expert data analyst who writes Databricks SQL (Spark SQL dialect) "
          "for an enterprise data warehouse.")

RULES = """Rules:
1. Answer with exactly ONE read-only SQL query (SELECT, CTEs allowed) inside a ```sql code fence. No explanation.
2. Use only the tables and columns listed in the schema. Qualify columns with table aliases when joining.
3. The dialect is Databricks SQL. String comparisons on table columns are case-insensitive.
4. Questions were written for MySQL. In MySQL, STDDEV(), STD() and VARIANCE() compute POPULATION statistics,
   but in Databricks the functions with those names compute SAMPLE statistics. So whenever a question asks for
   a standard deviation or variance - including when it says "use STDDEV" or "never STDDEV_POP" - write
   STDDEV_POP(...) or VAR_POP(...) in Databricks SQL. Never use STDDEV(), STD(), VARIANCE() or VAR_SAMP().
5. Many codes, years and dates are stored as strings (for example term codes like '2014FA').
6. Return only the columns the question asks for, in the order it mentions them."""

_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class FewShotExample:
    question: str
    sql: str
    source_id: str = ""  # BEAVER id, so dev/eval sets can exclude it


@dataclass(frozen=True)
class Generation:
    sql: str                     # "" when nothing usable was produced
    raw: str
    parse_status: str            # OK | NO_SQL | EMPTY_RESPONSE
    llm: LlmResponse
    prompt: str
    example_ids: tuple[str, ...] = ()    # few-shot examples used (dynamic mode: per question)
    schema_tables: tuple[str, ...] = ()  # tables whose schema was shown
    notes: str = ""                      # warehouse usage notes shown (knowledge mode)


def select_few_shot(queries: list[dict], exclude_ids: set, n: int = 3, seed: int = 20260925,
                    max_tables: int = 3, max_sql_chars: int = 900) -> list[FewShotExample]:
    """Deterministic few-shot pool from non-evaluation questions whose SQL parses as Databricks SQL
    after the documented adapter rules."""
    pool = []
    for e in queries:
        if e["id"] in exclude_ids or not e.get("sql") or len(e.get("tables") or []) > max_tables:
            continue
        sql, _ = apply_rules(e["sql"].strip().rstrip(";"))
        if len(sql) > max_sql_chars:
            continue
        try:
            sqlglot.parse_one(sql, read="databricks")
        except sqlglot.errors.ParseError:
            continue
        pool.append(FewShotExample(e["question"].strip(), sql, str(e["id"])))
    pool.sort(key=lambda x: x.question)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def render_schema(catalog: SchemaCatalog, tables: tuple[str, ...]) -> str:
    blocks = []
    for t in tables:
        tb = catalog.tables[t]
        lines = [f"TABLE {catalog.db}.{tb.name}"]
        for c in tb.columns:
            ex = f"  -- e.g. {', '.join(c.examples)}" if c.examples else ""
            lines.append(f"  {c.name} {c.type}{ex}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

# 创建prompt=规则+需要用到的表的说明+示例+总结的知识
def build_prompt(task: AgentTask, schema_text: str, examples: list[FewShotExample],
                 oracle_hints: list[str] | None = None, notes: str = "") -> str:
    parts = [RULES, "", "Schema:", schema_text, ""]
    if notes:  # warehouse usage notes mined from solved queries (agent/knowledge.py)
        parts += [notes, ""]
    if examples:
        parts.append("Examples (from the same warehouse):")
        for ex in examples:
            parts += [f"Question: {ex.question}", f"```sql\n{ex.sql}\n```", ""]
    if oracle_hints:
        parts.append("Verified facts about this question (use them):")
        parts += [f"- {h}" for h in oracle_hints]
        parts.append("")
    parts += [f"Question: {task.question}", "SQL:"]
    return "\n".join(parts)


def extract_sql(text: str) -> tuple[str, str]:
    """Return (sql, parse_status). Tolerates fenced and bare SQL."""
    if not text or not text.strip():
        return "", "EMPTY_RESPONSE"
    m = _FENCE.search(text)
    body = (m.group(1) if m else text).strip()
    if not re.match(r"(?is)^\s*(with|select|\()", body):
        return "", "NO_SQL"  # e.g. a refusal or prose (glm also wraps refusals in ```sql fences)
    return body.rstrip().rstrip(";").strip(), "OK"


PROMPT_VERSION_DYNAMIC = "baseline-v3-dynfs"


@dataclass
class FewShotGenerator:
    """``index`` (agent/examples.py ExampleIndex) switches to dynamic few-shot: the ``k`` solved questions most
    similar to the new one replace the fixed examples, and the tables they use are added to the schema shown
    (up to ``max_extra_tables``), so the model sees which tables this warehouse uses for such questions."""

    client: ChatClient
    catalog: SchemaCatalog
    examples: list[FewShotExample]
    index: Any = None
    k: int = 4
    max_extra_tables: int = 6
    knowledge: Any = None        # agent/knowledge.py WarehouseKnowledge: usage notes per question

    @property
    def prompt_version(self) -> str:
        base = PROMPT_VERSION_DYNAMIC if self.index is not None else PROMPT_VERSION
        return base + ("+kb" if self.knowledge is not None else "")

    def generate(self, task: AgentTask, tables: tuple[str, ...], oracle_hints: list[str] | None = None) -> Generation:
        """``oracle_hints`` carry GOLD annotations (BEAVER setting=1/2). Offline diagnostic analysis
        only (evaluation/intervention.py) — the agent, loop and experiments never pass them."""
        examples = self.examples
        # 只有 dynamic 模式才有示例库；static 模式 index 为 None，整段跳过。主要作用是看相似题怎么做的
        if self.index is not None:
            # 从已解题里，按问题文本 BM25 找最相似的 k=4 道
            # hits 里每一项是 PoolEntry，有两个字段：example（问题和 SQL）和 tables（这道题 SQL 里用到的表，建库时就已解析好）
            hits = self.index.top(task.question, self.k)
            # 用这 4 道题（问题 + SQL）替换固定的 3 个示例
            examples = [h.example for h in hits]
            # 把相似题用到的表摊成一个列表，只有 schema 里真实存在、且检索结果中还没有的表才加进来
            extra = [t for h in hits for t in h.tables if t in self.catalog.tables and t not in tables]
            # 最多补 6 张
            tables = tuple(tables) + tuple(dict.fromkeys(extra))[: self.max_extra_tables]
        # 根据问题检索表相关的知识，生成相关说明
        notes = self.knowledge.notes_for(task.question, tables) if self.knowledge is not None else ""
        # 生成提示词
        prompt = build_prompt(task, render_schema(self.catalog, tables), examples, oracle_hints, notes)
        # 调用LLM，生成SQL
        r = self.client.complete(prompt, system=SYSTEM)
        sql, status = extract_sql(r.text)
        return Generation(sql, r.text, status, r, prompt, tuple(e.source_id for e in examples), tuple(tables), notes)

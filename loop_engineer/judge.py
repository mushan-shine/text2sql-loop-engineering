"""LLM judge — asks a model whether an executed SQL answers the question (gold-free).

Input: the question, the schema of the tables the SQL uses, the SQL and a result
summary (row count + first rows). Output: verdict (correct / wrong), confidence
and the problems found, as strict JSON. A candidate verifier signal; whether it
is precise enough to trigger repairs is measured offline first
(scripts/judge_eval.py: hits on wrong attempts vs. false alarms on gold SQL).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agent.generator import render_schema
from agent.llm import ChatClient
from agent.retriever import SchemaCatalog
from agent.sql_analysis import alias_map

JUDGE_VERSION = "judge-v2"   # v2: warehouse conventions added (v1 flagged correct STDDEV_POP / VAR_POP use)
# General warehouse conventions the generator also follows (agent/generator.py RULES) — not question-specific.
CONVENTIONS = """Warehouse conventions (these are correct; do not report them as problems):
- In this warehouse a question's "standard deviation" / "variance" (even "use STDDEV only, never STDDEV_POP") means the
  population statistic: STDDEV_POP / VAR_POP in Databricks SQL, or STDDEV / STD / VARIANCE in MySQL, are all correct.
- The query may be written in Databricks SQL or in MySQL style; judge the logic, not the dialect.
- String comparisons are case-insensitive; codes and years are often stored as strings ('2022' and 2022 are equivalent).
- "(Course N)" in a question is the department code 'N' (e.g. Mathematics = '18')."""
JUDGE_SYSTEM = ("You are a meticulous reviewer of SQL written for analytics questions over a data warehouse. "
                "You judge whether a query answers the question exactly as asked. Reply with JSON only.")
JUDGE_PROMPT = """Question: {question}

Schema of the tables the query uses:
{schema}

Query:
```sql
{sql}
```
Result: {rows} row(s). First rows: {preview}

{conventions}

Check the query against the question, point by point:
1. Every output the question asks for is returned (no missing or extra columns, requested order).
2. Every filter / condition in the question is applied, with the right column and value.
3. Statistics and grouping match the question (which statistic, computed per what).
4. Joins relate the right tables on real key columns and do not multiply rows.
5. The result is plausible for the question.

Reply with JSON only:
{{"verdict": "correct" or "wrong", "confidence": number between 0 and 1, "problems": ["short description", ...]}}"""


@dataclass(frozen=True)
class Judgement:
    verdict: str                 # correct | wrong | unparseable
    confidence: float
    problems: tuple[str, ...] = field(default_factory=tuple)
    raw: str = ""
    tokens: int = 0


def parse_judgement(text: str) -> tuple[str, float, tuple[str, ...]] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    v = str(d.get("verdict", "")).strip().lower()
    if v not in ("correct", "wrong"):
        return None
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    probs = tuple(str(p) for p in (d.get("problems") or []) if str(p).strip())
    return v, conf, probs


@dataclass
class LlmJudge:
    client: ChatClient
    catalog: SchemaCatalog
    max_tables: int = 8

    def prompt(self, question: str, sql: str, rows: list | None, n_rows: int | None) -> str:
        tables = [t for t in dict.fromkeys(alias_map(sql).values()) if t in self.catalog.tables][: self.max_tables]
        preview = json.dumps([list(map(str, r)) for r in (rows or [])[:5]], ensure_ascii=False)[:800]
        return JUDGE_PROMPT.format(question=question, schema=render_schema(self.catalog, tuple(tables)) or "(unknown)",
                                   sql=sql, rows=n_rows if n_rows is not None else len(rows or []), preview=preview,
                                   conventions=CONVENTIONS)

    # 使用LLM评判 
    def judge(self, question: str, sql: str, rows: list | None, n_rows: int | None = None) -> Judgement:
        r = self.client.complete(self.prompt(question, sql, rows, n_rows), system=JUDGE_SYSTEM)
        parsed = parse_judgement(r.text)
        tokens = r.input_tokens + r.output_tokens
        if parsed is None:
            return Judgement("unparseable", 0.0, (), r.text[:300], tokens)
        v, conf, probs = parsed
        return Judgement(v, conf, probs, r.text[:300], tokens)

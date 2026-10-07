"""BEAVER case model with an explicit split between agent-visible and gold fields.

Benchmark-integrity rule: agents, diagnosis and repair skills only ever receive
an :class:`AgentTask`. Everything else on :class:`BeaverCase` is ground truth and
is reserved for evaluation / benchmark analysis.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Fields an agent may see (BEAVER setting=0).
AGENT_FIELDS = ("case_id", "question", "db")

# Ground-truth annotations. Never passed to Agent / Diagnosis / Repair.
GOLD_FIELDS = (
    "gold_sql",
    "gold_tables",
    "gold_column_mapping",
    "gold_join_keys",
    "domain_knowledge",
    "sub_questions",
    "sub_sqls",
)


@dataclass(frozen=True)
class AgentTask:
    """The only view of a case that the Text-to-SQL agent is allowed to receive."""

    case_id: str
    question: str
    db: str


@dataclass(frozen=True)
class BeaverCase:
    case_id: str
    split: str
    question: str
    db: str
    gold_sql: str  # ORIGINAL BEAVER gold SQL — never modified
    gold_tables: list[str] = field(default_factory=list)
    gold_column_mapping: dict[str, Any] = field(default_factory=dict)
    gold_join_keys: list[Any] = field(default_factory=list)
    domain_knowledge: list[Any] = field(default_factory=list)
    sub_questions: list[Any] = field(default_factory=list)
    sub_sqls: list[Any] = field(default_factory=list)
    category: str | None = None
    detailed_category: str | None = None
    contains_domain_knowledge: bool | None = None

    @classmethod
    def from_beaver(cls, entry: dict[str, Any], split: str) -> "BeaverCase":
        """Build from an entry shaped like BEAVER's data/download_hf.py output."""
        return cls(
            case_id=f"{split}:{entry['id']}",
            split=split,
            question=entry["question"],
            db=entry["db"],
            gold_sql=entry["sql"],
            gold_tables=list(_parse(entry.get("tables"), list)),
            gold_column_mapping=dict(_parse(entry.get("column_mapping"), dict)),
            gold_join_keys=list(_parse(entry.get("join_keys"), list)),
            domain_knowledge=list(_parse(entry.get("domain_knowledge"), list)),
            sub_questions=list(_parse(entry.get("sub_questions"), list)),
            sub_sqls=list(_parse(entry.get("sub_sqls"), list)),
            category=entry.get("category"),
            detailed_category=entry.get("detailed_category"),
            contains_domain_knowledge=entry.get("contains_domain_knowledge"),
        )

    def agent_view(self) -> AgentTask:
        return AgentTask(case_id=self.case_id, question=self.question, db=self.db)

    @property
    def source_sha256(self) -> str:
        """Fingerprint of the original record; used to refuse silent overwrites."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_row(self) -> dict[str, Any]:
        """Flat row for benchmark.cases (JSON-encode nested annotations)."""
        return {
            "case_id": self.case_id,
            "split": self.split,
            "question": self.question,
            "db": self.db,
            "gold_sql": self.gold_sql,
            "gold_tables": json.dumps(self.gold_tables, ensure_ascii=False),
            "gold_column_mapping": json.dumps(self.gold_column_mapping, ensure_ascii=False),
            "gold_join_keys": json.dumps(self.gold_join_keys, ensure_ascii=False),
            "domain_knowledge": json.dumps(self.domain_knowledge, ensure_ascii=False),
            "query_decomposition": json.dumps(
                {"sub_questions": self.sub_questions, "sub_sqls": self.sub_sqls},
                ensure_ascii=False,
            ),
            "category": self.category,
            "detailed_category": self.detailed_category,
            "contains_domain_knowledge": self.contains_domain_knowledge,
            "source_sha256": self.source_sha256,
        }


def _parse(val: Any, default_type: type) -> Any:
    """Same leniency as BEAVER's parse_if_string."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except ValueError:
            return default_type()
    return val if val is not None else default_type()

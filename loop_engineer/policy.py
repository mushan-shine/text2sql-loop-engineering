"""Repair Policy — maps a diagnosis to a repair skill.

The mapping is data, so ablations (phase 8) swap it instead of editing code:
* ``targeted``  – the default mapping below;
* ``generic``   – every failure goes to RepairSQL ("No Repair Policy");
* ``disabled``  – skills removed one at a time ("No Individual Repair Skill")
  fall back to RepairSQL.

DOMAIN_KNOWLEDGE_FAILURE goes to ReplanQuery until RetrieveKnowledge has a
non-gold knowledge source; the route is marked as a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loop_engineer.diagnose import (COLUMN_MAPPING, DOMAIN_KNOWLEDGE, EXECUTION, JOIN_KEY, QUERY_DECOMPOSITION,
                                    TABLE_RETRIEVAL, UNKNOWN, Diagnosis)
from skills.find_join_path import FindJoinPath
from skills.repair_sql import RepairSQL
from skills.replan_query import ReplanQuery
from skills.retrieve_again import RetrieveAgain
from skills.schema_search import SchemaSearch

SKILLS = {s.name: s for s in (RetrieveAgain(), SchemaSearch(), FindJoinPath(), ReplanQuery(), RepairSQL())}
FALLBACK = "RepairSQL"

# 失败类型对应的技能映射表
TARGETED = {
    TABLE_RETRIEVAL: "RetrieveAgain",
    COLUMN_MAPPING: "SchemaSearch",
    JOIN_KEY: "FindJoinPath",
    DOMAIN_KNOWLEDGE: "ReplanQuery",   # fallback until RetrieveKnowledge exists
    QUERY_DECOMPOSITION: "ReplanQuery",
    EXECUTION: "RepairSQL",
    UNKNOWN: "RepairSQL",
}
FALLBACK_ROUTES = {DOMAIN_KNOWLEDGE}


@dataclass(frozen=True)
class Route:
    skill: str
    fallback: bool
    reason: str


@dataclass
class Policy:
    mode: str = "targeted"                       # targeted | generic
    disabled: set[str] = field(default_factory=set)
    mapping: dict[str, str] = field(default_factory=lambda: dict(TARGETED))

    def route(self, diagnosis: Diagnosis) -> Route:
        if self.mode == "generic":
            return Route(FALLBACK, False, "policy disabled: generic repair")
        # 根据诊断到的失败类型映射到技能名称
        skill = self.mapping.get(diagnosis.failure_type, FALLBACK)
        if skill in self.disabled:
            return Route(FALLBACK, True, f"{skill} disabled (ablation)")
        return Route(skill, diagnosis.failure_type in FALLBACK_ROUTES,
                     f"{diagnosis.failure_type} -> {skill}")

    def skill(self, name: str):
        # 根据技能名称掉对应的函数
        return SKILLS[name]

"""Knowledge items mined by the outer loop (evaluation/outer_loop.py) and how the generator uses them.

During an outer-loop iteration each candidate is attached to the generator to measure its effect; an item that a
reviewer approves can be attached the same way. Three kinds:

* ``table_preference`` — {"prefer": Y, "instead_of": X, "keywords": [...], "text": ...}
  used when X or Y is in the schema shown and the question shares a keyword (or no keywords are given);
* ``table_hint``       — {"table": T, "keywords": [...], "with": [...], "condition": ..., "text": ...}
  a table weak models leave out: when the question shares a keyword and one of ``with`` is shown (or ``with`` is
  empty), T is added to the schema shown and the note tells the model to use it;
* ``join_rule``        — {"tables": [a, b], "condition": "a.c = b.d", "join": "INNER"|"LEFT", "text": ...}
  used when both tables are in the schema shown.

No gold about the question being answered is involved at run time: items are aggregated knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.retriever import tokenize

KINDS = ("table_preference", "table_hint", "join_rule")


@dataclass
class CuratedKnowledge:
    items: list[dict] = field(default_factory=list)      # rows with "item_id", "kind", "content"

    @property
    def notes(self) -> list[dict]:
        return [i for i in self.items if i["kind"] in KINDS]

    def __bool__(self) -> bool:
        return bool(self.items)

    @staticmethod
    def _hint_applies(c: dict, words: set[str], shown: set[str]) -> bool:
        kws = {w for k in c.get("keywords") or [] for w in tokenize(k)}
        return bool(kws & words) and (not c.get("with") or bool(set(c["with"]) & shown))

    def extra_tables(self, question: str, schema_tables: list[str] | tuple[str, ...]) -> list[str]:
        """Tables that table hints add to the schema shown for this question."""
        shown, words = set(schema_tables), set(tokenize(question))
        return list(dict.fromkeys(it["content"]["table"] for it in self.items if it["kind"] == "table_hint"
                                  and it["content"]["table"] not in shown
                                  and self._hint_applies(it["content"], words, shown)))

    def notes_for(self, question: str, schema_tables: list[str] | tuple[str, ...]) -> str:
        shown, words = set(schema_tables), set(tokenize(question))
        lines = []
        for it in self.notes:
            c = it["content"]
            if it["kind"] == "table_hint":
                if c.get("table") in shown and self._hint_applies(c, words, shown):
                    lines.append(f"- {c['text']}")
            elif it["kind"] == "table_preference":
                kws = {w for k in c.get("keywords") or [] for w in tokenize(k)}
                if ({c.get("prefer"), c.get("instead_of")} & shown) and (not kws or kws & words):
                    lines.append(f"- {c['text']}")
            elif it["kind"] == "join_rule":
                if set(c.get("tables") or []) <= shown:
                    lines.append(f"- {c['text']}")
        if not lines:
            return ""
        return "Reviewed usage notes (approved by this warehouse's data engineers; follow them):\n" + "\n".join(lines)


def attach_curated(generator: Any, curated: CuratedKnowledge | None) -> Any:
    """Give a FewShotGenerator the knowledge items: table hints extend the schema, notes go into the prompt."""
    if curated:
        generator.curated = curated
    return generator

"""Join candidates inferred from the schema only (never from BEAVER join_keys).

Two tables are joinable on a column they share by name when the name looks
like a key (``*_CODE``, ``*_KEY``, ``*_ID``, ...). This is the non-gold join
source decided in ROADMAP 5.3; FindJoinPath and RetrieveAgain use it to tell
the model how a table can be connected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from agent.retriever import SchemaCatalog

_KEYISH = re.compile(r"(_CODE|_KEY|_ID|_NUMBER|^DLC_KEY$|^TERM_CODE$|^COURSE$|^DEPARTMENT$)$", re.I)
_GENERIC = {"WAREHOUSE_LOAD_DATE", "LAST_MODIFIED_DATE", "LAST_ACTIVITY_DATE"}


@dataclass(frozen=True)
class JoinCandidate:
    left: str
    right: str
    column: str

    def sql(self) -> str:
        return f"{self.left}.{self.column} = {self.right}.{self.column}"


def join_candidates(catalog: SchemaCatalog, tables: list[str] | tuple[str, ...]) -> list[JoinCandidate]:
    """Shared key-like columns between every pair of the given tables."""
    cols = {t: {c.name.upper() for c in catalog.tables[t].columns} for t in tables if t in catalog.tables}
    out = []
    names = sorted(cols)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for c in sorted(cols[a] & cols[b]):
                if _KEYISH.search(c) and c not in _GENERIC:
                    out.append(JoinCandidate(a, b, c))
    return out


def connect(catalog: SchemaCatalog, new_table: str, used: list[str] | tuple[str, ...]) -> list[JoinCandidate]:
    """Candidates that connect ``new_table`` to any of the ``used`` tables."""
    return [j for j in join_candidates(catalog, [new_table, *used]) if new_table in (j.left, j.right)]

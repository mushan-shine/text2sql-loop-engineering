"""RetrieveKnowledge — DOMAIN_KNOWLEDGE_FAILURE. NOT IMPLEMENTED YET.

BEAVER's ``domain_knowledge`` annotations are gold and must not be used
(ROADMAP 5.4). A non-gold knowledge source is still undecided, e.g.:
* value grounding: look up distinct values of string columns that match terms
  of the question (runtime database lookups);
* a data dictionary / glossary curated from NON-evaluation questions.

Until then the policy routes DOMAIN_KNOWLEDGE_FAILURE to ReplanQuery and marks
the route as a fallback in the trace.
"""
from __future__ import annotations


class RetrieveKnowledge:
    name = "RetrieveKnowledge"

    def repair(self, *args, **kwargs):  # pragma: no cover - placeholder
        raise NotImplementedError("RetrieveKnowledge needs a non-gold knowledge source (ROADMAP 5.4)")

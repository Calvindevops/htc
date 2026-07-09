"""Synthesis — the capstone of the memory system: on-demand, LLM-synthesized,
strictly grounded and CITED answers to questions asked directly against the
whole memory (hybrid retrieval + optional rerank + optional knowledge graph),
plus an explicit gap analysis of what the memory doesn't know or is
uncertain about. gBrain's "gives you the answer, not raw search results."

Complements raw search (`memory.search`, precise but unsynthesized) and the
pre-built `wiki` (a fixed set of topic pages) — this is ad hoc, query-time
synthesis for any question, not a fixed page set.
"""

from __future__ import annotations

from .answer import answer_question
from .model import Answer

__all__ = ["Answer", "answer_question"]

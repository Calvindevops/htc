"""Answer — the synthesis layer's output: a grounded, cited answer to a
question asked directly against the whole memory, plus a gap analysis of
what the memory doesn't know or is uncertain about."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Answer:
    """One synthesized, cited answer, with an explicit gap analysis."""

    question: str
    answer_md: str
    citations: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    confidence: str = "low"

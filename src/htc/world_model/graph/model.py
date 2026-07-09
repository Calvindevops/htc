"""Knowledge-graph model — entities and relations.

Mirrors gBrain's approach: a self-wiring knowledge graph built with **zero
LLM calls** — cheap, fast, deterministic heuristic extraction, not an
API-cost generation step. See `extract.py` for the extraction logic and
`graph.py` for the `KnowledgeGraph` container."""

from __future__ import annotations

from dataclasses import dataclass

EntityKind = str  # "symbol" | "file" | "module" | "term" | "proper_noun"
RelationKind = str  # "co_occurs" | "contains" | "references"


@dataclass(frozen=True)
class Entity:
    """One node in the knowledge graph."""

    id: str  # normalized slug, e.g. "symbol:build-memory"
    name: str
    kind: EntityKind
    mentions: int = 0


@dataclass(frozen=True)
class Relation:
    """One directed edge in the knowledge graph."""

    source_id: str
    target_id: str
    kind: RelationKind
    weight: int = 1

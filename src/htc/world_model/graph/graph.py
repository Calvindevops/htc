"""KnowledgeGraph — the self-wiring knowledge graph container.

Built once by `extract.extract()` (zero LLM calls), persisted HTC-native to
`<root>/.htc/graph/graph.json`, and used as an additional retrieval signal
(see `memory/local.py`'s optional graph-boost). No network calls here.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from ..ingest.model import SourceChunk
from .extract import extract
from .model import Entity, Relation

_TOKEN_RE = re.compile(r"[a-z0-9]+")  # mirrors the BM25 tokenizer in memory/local.py

GRAPH_SUBDIR = Path(".htc") / "graph"
GRAPH_FILENAME = "graph.json"


def graph_json_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / GRAPH_SUBDIR / GRAPH_FILENAME


def _mermaid_node_id(entity_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", entity_id)


class KnowledgeGraph:
    """A small in-memory graph of `Entity` nodes + `Relation` edges."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._adjacency: dict[str, dict[str, int]] = {}

    def add_entities(self, entities: list[Entity]) -> None:
        for entity in entities:
            self._entities[entity.id] = entity

    def add_relations(self, relations: list[Relation]) -> None:
        for relation in relations:
            self._relations.append(relation)
            self._bump_adjacency(relation.source_id, relation.target_id, relation.weight)
            self._bump_adjacency(relation.target_id, relation.source_id, relation.weight)

    def _bump_adjacency(self, from_id: str, to_id: str, weight: int) -> None:
        neighbors = self._adjacency.setdefault(from_id, {})
        neighbors[to_id] = neighbors.get(to_id, 0) + weight

    def entities(self) -> list[Entity]:
        return sorted(self._entities.values(), key=lambda e: e.id)

    def relations(self) -> list[Relation]:
        return sorted(self._relations, key=lambda r: (r.source_id, r.target_id, r.kind))

    def neighbors(self, entity_id: str, k: int = 10) -> list[Entity]:
        """The `k` entities most strongly connected to `entity_id` (by
        summed relation weight), best-first."""
        adjacent = self._adjacency.get(entity_id, {})
        ordered_ids = sorted(adjacent, key=lambda nid: (-adjacent[nid], nid))
        return [self._entities[nid] for nid in ordered_ids[:k] if nid in self._entities]

    def top_entities(self, n: int = 20) -> list[Entity]:
        """The `n` most-mentioned entities, best-first."""
        return sorted(self._entities.values(), key=lambda e: (-e.mentions, e.id))[:n]

    def subgraph_for_query(self, query: str, k: int = 10) -> list[Entity]:
        """Entities whose name shares a token with `query`, plus their
        neighbors — a small subgraph relevant to `query`, capped at `k`."""
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return []

        def _matches(entity: Entity) -> bool:
            return bool(query_tokens & set(_TOKEN_RE.findall(entity.name.lower())))

        matched = sorted(
            (e for e in self._entities.values() if _matches(e)),
            key=lambda e: (-e.mentions, e.id),
        )
        collected: dict[str, Entity] = {e.id: e for e in matched}
        for entity in matched:
            for neighbor in self.neighbors(entity.id, k=k):
                collected.setdefault(neighbor.id, neighbor)
        ordered = sorted(collected.values(), key=lambda e: (-e.mentions, e.id))
        return ordered[:k]

    def to_mermaid(self, limit: int = 40) -> str:
        """Render the top `limit` entities (by mentions) and the relations
        between them as a fenced ```mermaid``` graph diagram."""
        top = self.top_entities(limit)
        top_ids = {e.id for e in top}
        lines = ["```mermaid", "graph LR"]
        for entity in sorted(top, key=lambda e: e.id):
            label = entity.name.replace('"', "'")
            lines.append(f'  {_mermaid_node_id(entity.id)}["{label}"]')
        edges = sorted(
            (r for r in self._relations if r.source_id in top_ids and r.target_id in top_ids),
            key=lambda r: (-r.weight, r.source_id, r.target_id, r.kind),
        )
        for relation in edges:
            source = _mermaid_node_id(relation.source_id)
            target = _mermaid_node_id(relation.target_id)
            lines.append(f"  {source} -->|{relation.kind}| {target}")
        lines.append("```")
        return "\n".join(lines) + "\n"

    def save(self, root: str | Path) -> Path:
        path = graph_json_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": [asdict(e) for e in self.entities()],
            "relations": [asdict(r) for r in self.relations()],
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, root: str | Path) -> "KnowledgeGraph":
        graph = cls()
        path = graph_json_path(root)
        if not path.is_file():
            return graph
        data = json.loads(path.read_text())
        graph.add_entities([Entity(**item) for item in data.get("entities", [])])
        graph.add_relations([Relation(**item) for item in data.get("relations", [])])
        return graph


def build_graph(chunks: list[SourceChunk], root: str | Path) -> KnowledgeGraph:
    """Extract entities/relations from `chunks` (zero LLM calls), populate a
    `KnowledgeGraph`, and persist it to `<root>/.htc/graph/graph.json`."""
    entities, relations = extract(chunks)
    graph = KnowledgeGraph()
    graph.add_entities(entities)
    graph.add_relations(relations)
    graph.save(root)
    return graph

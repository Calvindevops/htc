"""RetrievalPipeline — a single composed retrieval object.

Wires ingest (once) → memory store → optional knowledge graph (built from
the SAME ingested chunks, no second ingest) → query-transform → hybrid
search (with the graph signal) → rerank once after fusion
(`retrieve_with_transform` already does this), so callers configure
retrieval once instead of re-deriving corpus/graph/reranker wiring at every
call site.

Not wired into any consumer yet — that's a later pass; this module only
builds and tests the pipeline object itself.
"""

from __future__ import annotations

from pathlib import Path

from ..adapters.base import Source
from ..adapters.filesystem import FilesystemAdapter
from .build import _prepare_chunks
from .graph.graph import KnowledgeGraph, build_graph
from .ingest import ingest_sources
from .memory import MemoryStore, SearchResult, get_memory_store
from .query.retrieve import retrieve_with_transform
from .rerank import Reranker, get_reranker

__all__ = ["RetrievalPipeline", "build_pipeline"]


class RetrievalPipeline:
    """Holds a `store`, an optional `reranker` and knowledge `graph`, and a
    `query_transform` strategy name — `retrieve()` runs query-transform →
    hybrid search (with the graph signal) → rerank once after fusion."""

    def __init__(
        self,
        store: MemoryStore,
        reranker: Reranker | None = None,
        query_transform: str = "none",
        graph: KnowledgeGraph | None = None,
        model: str | None = None,
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.query_transform = query_transform
        self.graph = graph
        self.model = model

    def retrieve(self, query: str, k: int = 8) -> list[SearchResult]:
        return retrieve_with_transform(
            self.store,
            query,
            k,
            strategy=self.query_transform,
            model=self.model,
            graph=self.graph,
            reranker=self.reranker,
        )


def build_pipeline(
    root: str | Path,
    sources: list[Source] | None = None,
    *,
    backend: str = "local",
    rerank: str = "none",
    query_transform: str = "none",
    contextual: bool = False,
    graph: bool = False,
    model: str | None = None,
) -> RetrievalPipeline:
    """Build a `RetrievalPipeline` for `root`.

    Ingests `sources` (defaulting to the whole `root` filesystem via
    `FilesystemAdapter`) exactly ONCE, loads the resulting chunks into a
    memory store, and — when `graph=True` — builds the knowledge graph from
    those SAME chunks (no second `ingest_sources` call).
    """
    corpus = ingest_sources(sources or FilesystemAdapter(str(root)).sources(), root=Path(root))
    chunks = _prepare_chunks(corpus, contextual, model)

    store = get_memory_store(root, backend=backend)
    store.add_chunks(chunks)

    kg = build_graph(chunks, Path(root)) if graph else None
    reranker = get_reranker(rerank) if rerank != "none" else None

    return RetrievalPipeline(
        store=store,
        reranker=reranker,
        query_transform=query_transform,
        graph=kg,
        model=model,
    )

"""MemoryStore — the interface a persistent, queryable memory backend must
satisfy. `LocalMemoryStore` (self-contained, default) and `GBrainMemoryStore`
(optional external adapter) both implement this contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..ingest.model import SourceChunk

if TYPE_CHECKING:
    from ..graph.graph import KnowledgeGraph


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk, ranked by relevance to a query."""

    chunk: SourceChunk
    score: float


class MemoryStore(Protocol):
    """Persistent, queryable storage for ingested `SourceChunk`s."""

    def add_chunks(self, chunks: list[SourceChunk]) -> None:
        """Add chunks to the store (persisted for later sessions)."""
        ...

    def search(
        self, query: str, k: int = 5, graph: KnowledgeGraph | None = None
    ) -> list[SearchResult]:
        """Return the `k` most relevant stored chunks for `query`. `graph` is
        an optional additional retrieval signal (see `LocalMemoryStore`'s
        graph-boost); backends that don't support it simply ignore it."""
        ...

    def has_source(self, path: str) -> bool:
        """Whether any chunk from `path` is stored."""
        ...

    def count(self) -> int:
        """Total number of stored chunks."""
        ...

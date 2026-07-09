"""MemoryStore — the interface a persistent, queryable memory backend must
satisfy. `LocalMemoryStore` (self-contained, default) and `GBrainMemoryStore`
(optional external adapter) both implement this contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..ingest.model import SourceChunk


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

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the `k` most relevant stored chunks for `query`."""
        ...

    def has_source(self, path: str) -> bool:
        """Whether any chunk from `path` is stored."""
        ...

    def count(self) -> int:
        """Total number of stored chunks."""
        ...

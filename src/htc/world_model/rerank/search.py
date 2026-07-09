"""`search_with_rerank` — retrieve a larger candidate pool via a store's
existing hybrid search, then rerank down to `top_k`. Non-breaking: when
`reranker` is `None` (or the default `NoOpReranker`), behavior is IDENTICAL
to calling `store.search` directly — no pool expansion, no extra work.

`RerankingMemoryStore` wraps any `MemoryStore` so callers that already
accept an optional `memory: MemoryStore` (handbook/studio/wiki generators)
can opt into reranking with no change to their retrieval code.
"""

from __future__ import annotations

from typing import Any

from ..graph import KnowledgeGraph
from ..memory.store import MemoryStore, SearchResult
from .base import NoOpReranker, Reranker

_DEFAULT_POOL_MULTIPLIER = 4
_MIN_POOL = 20


def _pool_size(k: int, rerank_pool: int | None) -> int:
    return rerank_pool if rerank_pool is not None else max(k * _DEFAULT_POOL_MULTIPLIER, _MIN_POOL)


def _store_search(
    store: MemoryStore, query: str, k: int, graph: KnowledgeGraph | None
) -> list[SearchResult]:
    """Call `store.search`, passing `graph` through only if the store's
    `search` accepts it (only `LocalMemoryStore` does today)."""
    if graph is not None:
        try:
            return store.search(query, k=k, graph=graph)
        except TypeError:
            pass
    return store.search(query, k=k)


def search_with_rerank(
    store: MemoryStore,
    query: str,
    k: int = 5,
    reranker: Reranker | None = None,
    rerank_pool: int | None = None,
    graph: KnowledgeGraph | None = None,
) -> list[SearchResult]:
    """Retrieve `k` results, optionally reranked.

    When `reranker` is `None` or a `NoOpReranker`, this is IDENTICAL to
    `store.search(query, k=k[, graph=graph])` — no pool expansion. Otherwise
    a larger candidate pool (`rerank_pool`, default `max(k * 4, 20)`) is
    retrieved first and the reranker re-scores it down to `k`.
    """
    if reranker is None or isinstance(reranker, NoOpReranker):
        return _store_search(store, query, k, graph)
    pool = _pool_size(k, rerank_pool)
    candidates = _store_search(store, query, pool, graph)
    return reranker.rerank(query, candidates, top_k=k)


class RerankingMemoryStore:
    """Wraps any `MemoryStore`, overriding `search` to retrieve a larger pool
    and rerank it down to `k` via `search_with_rerank` — lets callers that
    already accept a `memory: MemoryStore` opt into reranking with no change
    to their retrieval code. Delegates every other method untouched."""

    def __init__(
        self, store: MemoryStore, reranker: Reranker, rerank_pool: int | None = None
    ) -> None:
        self._store = store
        self._reranker = reranker
        self._rerank_pool = rerank_pool

    def add_chunks(self, chunks) -> None:
        self._store.add_chunks(chunks)

    def search(self, query: str, k: int = 5, **kwargs: Any) -> list[SearchResult]:
        return search_with_rerank(
            self._store,
            query,
            k=k,
            reranker=self._reranker,
            rerank_pool=self._rerank_pool,
            graph=kwargs.get("graph"),
        )

    def has_source(self, path: str) -> bool:
        return self._store.has_source(path)

    def count(self) -> int:
        return self._store.count()

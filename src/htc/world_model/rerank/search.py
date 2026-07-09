"""`search_with_rerank` — retrieve a larger candidate pool via a store's
existing hybrid search, then rerank down to `top_k`. Non-breaking: when
`reranker` is `None` (or the default `NoOpReranker`), behavior is IDENTICAL
to calling `store.search` directly — no pool expansion, no extra work.

Used by `RetrievalPipeline` to apply reranking after fusion.
"""

from __future__ import annotations

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
    """Call `store.search`, forwarding `graph` — every `MemoryStore` accepts
    the `graph` param explicitly (backends that don't use it simply ignore
    it)."""
    return store.search(query, k=k, graph=graph)


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

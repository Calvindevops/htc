"""`retrieve_with_transform` — optional query-side transformation before
retrieval (gBrain's "expansion model"): transform the raw query into better
retrieval queries, retrieve for each, and fuse via Reciprocal Rank Fusion.
Pluggable and opt-in (costs LLM calls) — default "none" (or `HTC_QUERY_
TRANSFORM` unset) is byte-for-byte identical to `search_with_rerank`, no LLM
call, so zero-config behavior is unchanged.
"""

from __future__ import annotations

import os

from ..fusion import reciprocal_rank_fusion
from ..graph.graph import KnowledgeGraph
from ..memory.store import MemoryStore, SearchResult
from ..rerank.base import NoOpReranker, Reranker
from ..rerank.search import search_with_rerank
from .transform import decompose, expand, hyde, multi_query

STRATEGIES = ("none", "expand", "hyde", "decompose", "multi")


def _rrf_fuse_results(rankings: list[list[SearchResult]]) -> list[SearchResult]:
    """Reciprocal Rank Fusion across multiple per-variant result rankings,
    deduped by chunk id (the first ranking a chunk id appears in wins the
    `SearchResult` kept; only the RRF score is recomputed), best-first."""
    fused = reciprocal_rank_fusion(rankings, key=lambda result: result.chunk.id)
    return [SearchResult(chunk=result.chunk, score=score) for result, score in fused]


def retrieve_with_transform(
    store: MemoryStore,
    query: str,
    k: int = 5,
    *,
    strategy: str | None = None,
    model: str | None = None,
    graph: KnowledgeGraph | None = None,
    reranker: Reranker | None = None,
) -> list[SearchResult]:
    """Retrieve `k` results for `query`, optionally transforming it first.

    `strategy` (default: the `HTC_QUERY_TRANSFORM` env var, else "none"):
    - "none" — no transformation, no LLM call; IDENTICAL to
      `search_with_rerank(store, query, k, reranker, graph)`.
    - "hyde" — retrieve using a generated hypothetical document instead of
      the bare query.
    - "expand" / "decompose" / "multi" — generate multiple queries, retrieve
      for each (with `graph`, without an individual reranker), then fuse the
      rankings via RRF (deduped by chunk id).

    The reranker (if any) is always applied to the final pool — for "none"
    and "hyde" that's `search_with_rerank`'s own rerank step; for the
    multi-query strategies it's applied once, after fusion, to the whole
    fused pool.
    """
    resolved = strategy or os.environ.get("HTC_QUERY_TRANSFORM", "none")
    if resolved not in STRATEGIES:
        raise ValueError(
            f"unknown query-transform strategy '{resolved}' "
            f"(expected one of: {', '.join(STRATEGIES)})"
        )

    if resolved == "none":
        return search_with_rerank(store, query, k=k, reranker=reranker, graph=graph)

    if resolved == "hyde":
        transformed = hyde(query, model=model)
        return search_with_rerank(store, transformed, k=k, reranker=reranker, graph=graph)

    if resolved == "expand":
        queries = expand(query, model=model)
    elif resolved == "decompose":
        queries = decompose(query, model=model)
    else:  # "multi"
        queries = multi_query(query, model=model)

    rankings = [search_with_rerank(store, variant, k=k, graph=graph) for variant in queries]
    fused = _rrf_fuse_results(rankings)
    if reranker is not None and not isinstance(reranker, NoOpReranker):
        return reranker.rerank(query, fused, top_k=k)
    return fused[:k]

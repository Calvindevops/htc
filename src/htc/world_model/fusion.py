"""Reciprocal Rank Fusion (RRF) — the single canonical implementation shared
by `memory/local.py` (fusing BM25/semantic/graph id rankings) and
`query/retrieve.py` (fusing per-variant `SearchResult` rankings for
query-transform strategies)."""

from __future__ import annotations

from typing import Callable, TypeVar

# Reciprocal Rank Fusion constant (standard default).
RRF_K = 60

T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: list[list[T]], key: Callable[[T], str]
) -> list[tuple[T, float]]:
    """Fuse multiple best-first rankings of items into one fused ranking.

    Each item's fused score is the sum of `1 / (RRF_K + rank)` across every
    ranking it appears in (`rank` is 1-based); absence from a ranking simply
    contributes 0. `key` identifies an item across rankings (e.g. an id or a
    chunk id) — duplicate keys are deduped, keeping the first-seen item.
    Returns `(item, score)` pairs sorted by descending score, with `key` as a
    deterministic tiebreak.
    """
    scores: dict[str, float] = {}
    first_item: dict[str, T] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            item_key = key(item)
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (RRF_K + rank)
            first_item.setdefault(item_key, item)
    ordered_keys = sorted(scores, key=lambda k: (-scores[k], k))
    return [(first_item[k], scores[k]) for k in ordered_keys]

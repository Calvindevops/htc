"""Reranker — the interface a pluggable result-reranking backend must
satisfy. `NoOpReranker` (passthrough, truncated to top_k) is the default —
zero-config, dependency-light, matching HTC's local-first philosophy. Cloud
rerankers (`cloud.py`) and an optional local cross-encoder (`local.py`) are
opt-in BYO.
"""

from __future__ import annotations

from typing import Protocol

from ..memory.store import SearchResult


class Reranker(Protocol):
    """Re-scores hybrid-retrieval results for precision."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Return the `top_k` most relevant of `results`, best-first."""
        ...


class RerankerUnavailable(RuntimeError):
    """Raised when a requested reranker's dependency or API key isn't present."""


class NoOpReranker:
    """Passthrough reranker — the default. Returns `results` unchanged
    (already ranked by hybrid retrieval), truncated to `top_k`."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        return results[:top_k]

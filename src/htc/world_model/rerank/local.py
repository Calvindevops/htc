"""LocalCrossEncoderReranker — an optional, dependency-light local reranker
using a cross-encoder model via `sentence-transformers` (`pip install
htc[rerank]`). NOT a core dependency: raises `RerankerUnavailable` if the
library isn't installed, the same fail-loud-at-construction contract as the
cloud rerankers.
"""

from __future__ import annotations

import os

from ..memory.store import SearchResult
from .base import RerankerUnavailable

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LocalCrossEncoderReranker:
    """Cross-encoder reranker running locally via `sentence-transformers`.
    Model defaults to the small, CPU-friendly `cross-encoder/ms-marco-MiniLM-
    L-6-v2`; override with `HTC_RERANK_MODEL`."""

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as err:
            raise RerankerUnavailable(
                "Local cross-encoder reranker requested but 'sentence-transformers' "
                "is not installed. Install it with `pip install htc[rerank]`, or use "
                "the default no-op reranker (HTC_RERANKER=none)."
            ) from err
        resolved_model = model_name or os.environ.get("HTC_RERANK_MODEL", _DEFAULT_MODEL)
        self._model = CrossEncoder(resolved_model)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        scores = self._model.predict(pairs)
        scored = sorted(zip(scores, results), key=lambda pair: -pair[0])
        return [
            SearchResult(chunk=result.chunk, score=float(score)) for score, result in scored[:top_k]
        ]

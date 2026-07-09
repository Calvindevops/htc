"""Rerank: pluggable re-scoring of hybrid-retrieval results for precision.

`NoOpReranker` (passthrough) is the default — zero-config, dependency-light,
matching HTC's local-first philosophy. Cloud rerankers (ZeroEntropy zerank —
the gBrain default; Cohere; Voyage) are opt-in BYO via API key; an optional
local cross-encoder (`sentence-transformers`, `pip install htc[rerank]`) is
also available. Select via the `HTC_RERANKER` env var or the `get_reranker`
argument.
"""

from __future__ import annotations

import os

from .base import NoOpReranker, Reranker, RerankerUnavailable
from .cloud import CohereReranker, VoyageReranker, ZeroEntropyReranker
from .local import LocalCrossEncoderReranker
from .search import RerankingMemoryStore, search_with_rerank

__all__ = [
    "CohereReranker",
    "LocalCrossEncoderReranker",
    "NoOpReranker",
    "Reranker",
    "RerankerUnavailable",
    "RerankingMemoryStore",
    "VoyageReranker",
    "ZeroEntropyReranker",
    "get_reranker",
    "search_with_rerank",
]

_BUILTINS = {
    "none": NoOpReranker,
    "zerank": ZeroEntropyReranker,
    "cohere": CohereReranker,
    "voyage": VoyageReranker,
    "local": LocalCrossEncoderReranker,
}


def get_reranker(name: str | None = None) -> Reranker:
    """Return a `Reranker` by name.

    `name` selects the implementation; if omitted, falls back to the
    `HTC_RERANKER` env var, defaulting to "none" — the zero-config
    passthrough. Built-ins: "none", "zerank" (ZeroEntropy), "cohere",
    "voyage", "local" (cross-encoder via sentence-transformers).
    """
    resolved = name or os.environ.get("HTC_RERANKER", "none")
    cls = _BUILTINS.get(resolved)
    if cls is None:
        raise RerankerUnavailable(
            f"unknown reranker '{resolved}' (expected one of: {', '.join(sorted(_BUILTINS))})"
        )
    return cls()

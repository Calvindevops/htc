"""Cloud rerankers — BYO API key, lazy `httpx`, never crash the caller.

Fail-loud-at-construction, degrade-quietly-at-call-time (mirrors
`MemoryBackendUnavailable`'s contract): a missing API key raises
`RerankerUnavailable` immediately, but once constructed, any network/API
failure during `rerank()` degrades to a passthrough (original order,
truncated to `top_k`) with a `warnings.warn`, never an exception.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

from ..memory.secrets import load_secret
from ..memory.store import SearchResult
from .base import RerankerUnavailable

_TIMEOUT = 30.0


def _api_key(env_var: str, secret_name: str) -> str | None:
    return os.environ.get(env_var) or load_secret(secret_name)


def _passthrough(results: list[SearchResult], top_k: int, reason: str) -> list[SearchResult]:
    warnings.warn(reason, stacklevel=2)
    return results[:top_k]


def _call_rerank_api(
    url: str, headers: dict[str, str], body: dict[str, Any], results_key: str
) -> list[dict[str, Any]] | None:
    """POST `body` to `url` and return the raw per-document score list, or
    `None` on any failure (network error, non-2xx status, malformed JSON) —
    the caller degrades to passthrough rather than raising."""
    import httpx

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
    except Exception:  # noqa: BLE001 - any failure here must degrade, never crash
        return None
    return data.get(results_key)


def _reorder(
    results: list[SearchResult],
    scored: list[dict[str, Any]],
    top_k: int,
) -> list[SearchResult]:
    """Map a Cohere-shaped `[{"index": int, "relevance_score": float}, ...]`
    response back onto the original `SearchResult`s, best-first."""
    ordered = sorted(scored, key=lambda item: -item["relevance_score"])
    reranked = []
    for item in ordered[:top_k]:
        index = item["index"]
        if 0 <= index < len(results):
            original = results[index]
            reranked.append(
                SearchResult(chunk=original.chunk, score=float(item["relevance_score"]))
            )
    return reranked


class ZeroEntropyReranker:
    """ZeroEntropy zerank — the gBrain default. BYO key: `ZEROENTROPY_API_KEY`
    (env) or the encrypted secret store (`zeroentropy_api_key`).

    Documented-assumption HTTP contract (ZeroEntropy's rerank API follows the
    same request/response shape as Cohere/Voyage; adjust if the upstream
    contract changes):
      POST https://api.zeroentropy.dev/v1/models/rerank
        {"model": "zerank-2", "query": str, "documents": [str, ...], "top_n": int}
        -> {"results": [{"index": int, "relevance_score": float}, ...]}
    """

    _URL = "https://api.zeroentropy.dev/v1/models/rerank"
    _MODEL = "zerank-2"

    def __init__(self) -> None:
        key = _api_key("ZEROENTROPY_API_KEY", "zeroentropy_api_key")
        if not key:
            raise RerankerUnavailable(
                "ZeroEntropy reranker requested but ZEROENTROPY_API_KEY is not set. "
                "Get a key at https://zeroentropy.dev and set it, or use the default "
                "no-op reranker (HTC_RERANKER=none)."
            )
        self._key = key

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        body = {
            "model": self._MODEL,
            "query": query,
            "documents": [result.chunk.text for result in results],
            "top_n": top_k,
        }
        headers = {"authorization": f"Bearer {self._key}", "content-type": "application/json"}
        scored = _call_rerank_api(self._URL, headers, body, results_key="results")
        if scored is None:
            return _passthrough(results, top_k, "ZeroEntropy rerank call failed — passthrough.")
        return _reorder(results, scored, top_k)


class CohereReranker:
    """Cohere rerank API. BYO key: `COHERE_API_KEY` (env) or the encrypted
    secret store (`cohere_api_key`).

      POST https://api.cohere.com/v2/rerank
        {"model": "rerank-v3.5", "query": str, "documents": [str, ...], "top_n": int}
        -> {"results": [{"index": int, "relevance_score": float}, ...]}
    """

    _URL = "https://api.cohere.com/v2/rerank"
    _MODEL = "rerank-v3.5"

    def __init__(self) -> None:
        key = _api_key("COHERE_API_KEY", "cohere_api_key")
        if not key:
            raise RerankerUnavailable(
                "Cohere reranker requested but COHERE_API_KEY is not set. Get a key at "
                "https://dashboard.cohere.com and set it, or use the default no-op "
                "reranker (HTC_RERANKER=none)."
            )
        self._key = key

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        body = {
            "model": self._MODEL,
            "query": query,
            "documents": [result.chunk.text for result in results],
            "top_n": top_k,
        }
        headers = {"authorization": f"Bearer {self._key}", "content-type": "application/json"}
        scored = _call_rerank_api(self._URL, headers, body, results_key="results")
        if scored is None:
            return _passthrough(results, top_k, "Cohere rerank call failed — passthrough.")
        return _reorder(results, scored, top_k)


class VoyageReranker:
    """Voyage rerank API. BYO key: `VOYAGE_API_KEY` (env) or the encrypted
    secret store (`voyage_api_key`).

      POST https://api.voyageai.com/v1/rerank
        {"model": "rerank-2", "query": str, "documents": [str, ...], "top_k": int}
        -> {"data": [{"index": int, "relevance_score": float}, ...]}
    """

    _URL = "https://api.voyageai.com/v1/rerank"
    _MODEL = "rerank-2"

    def __init__(self) -> None:
        key = _api_key("VOYAGE_API_KEY", "voyage_api_key")
        if not key:
            raise RerankerUnavailable(
                "Voyage reranker requested but VOYAGE_API_KEY is not set. Get a key at "
                "https://dash.voyageai.com and set it, or use the default no-op "
                "reranker (HTC_RERANKER=none)."
            )
        self._key = key

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        body = {
            "model": self._MODEL,
            "query": query,
            "documents": [result.chunk.text for result in results],
            "top_k": top_k,
        }
        headers = {"authorization": f"Bearer {self._key}", "content-type": "application/json"}
        scored = _call_rerank_api(self._URL, headers, body, results_key="data")
        if scored is None:
            return _passthrough(results, top_k, "Voyage rerank call failed — passthrough.")
        return _reorder(results, scored, top_k)

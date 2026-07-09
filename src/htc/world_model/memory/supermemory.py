"""SupermemoryMemoryStore — optional adapter over the Supermemory API
(hosted at api.supermemory.ai, or self-hosted via `SUPERMEMORY_BASE_URL`).

Not required to use HTC: `LocalMemoryStore` is the self-contained default.
This adapter needs a Supermemory account/API key (or a self-hosted instance).

Documented-assumption HTTP contract (Supermemory does not publish a pinned
OpenAPI spec; adjust if the upstream contract changes):
  POST {base}/v3/documents  {"content": str, "metadata": {...}}
    -> add one memory document
  POST {base}/v3/search     {"q": str, "limit": int}
    -> {"results": [{"content": str, "metadata": {...}, "score": float}, ...]}

Network calls are best-effort: a failed add or search logs a warning and
degrades gracefully (skipped chunk / empty results) rather than crashing HTC.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

from ..ingest.model import SourceChunk
from .gbrain import MemoryBackendUnavailable
from .store import SearchResult

if TYPE_CHECKING:
    from ..graph.graph import KnowledgeGraph

_DEFAULT_BASE_URL = "https://api.supermemory.ai"


class SupermemoryMemoryStore:
    """Adapter over the Supermemory API. Requires `SUPERMEMORY_API_KEY`."""

    def __init__(self) -> None:
        key = os.environ.get("SUPERMEMORY_API_KEY")
        if not key:
            raise MemoryBackendUnavailable(
                "Supermemory backend requested but SUPERMEMORY_API_KEY is not set. "
                "Get a key at https://supermemory.ai and set it, or use the default "
                "local memory store (backend='local')."
            )
        self._key = key
        self._base = os.environ.get("SUPERMEMORY_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self._key}", "content-type": "application/json"}

    def add_chunks(self, chunks: list[SourceChunk]) -> None:
        import httpx

        with httpx.Client(timeout=30.0) as client:
            for chunk in chunks:
                body = {
                    "content": chunk.text,
                    "metadata": {
                        "chunk_id": chunk.id,
                        "source_path": chunk.source_path,
                        "kind": chunk.kind,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                }
                try:
                    res = client.post(
                        f"{self._base}/v3/documents", headers=self._headers(), json=body
                    )
                    res.raise_for_status()
                except httpx.HTTPError as err:
                    warnings.warn(
                        f"Supermemory add_chunks failed for chunk {chunk.id}: {err}",
                        stacklevel=2,
                    )

    def search(
        self, query: str, k: int = 5, graph: KnowledgeGraph | None = None
    ) -> list[SearchResult]:
        import httpx

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"{self._base}/v3/search",
                    headers=self._headers(),
                    json={"q": query, "limit": k},
                )
                res.raise_for_status()
                data = res.json()
        except httpx.HTTPError as err:
            warnings.warn(f"Supermemory search failed: {err}", stacklevel=2)
            return []

        results: list[SearchResult] = []
        for item in (data.get("results") or [])[:k]:
            metadata = item.get("metadata") or {}
            chunk = SourceChunk(
                id=metadata.get("chunk_id", ""),
                source_path=metadata.get("source_path", ""),
                kind=metadata.get("kind", "docs"),
                text=item.get("content", ""),
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", 0),
            )
            results.append(SearchResult(chunk=chunk, score=float(item.get("score", 0.0))))
        return results

    def has_source(self, path: str) -> bool:
        raise NotImplementedError("Supermemory backend does not support has_source lookups yet.")

    def count(self) -> int:
        raise NotImplementedError("Supermemory backend does not support count yet.")

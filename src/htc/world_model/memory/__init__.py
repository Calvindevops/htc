"""Memory: persistent, queryable storage for ingested `SourceChunk`s.

`LocalMemoryStore` is the default, self-contained backend (no external
service, no ML dependency — works offline out of the box). `GBrainMemoryStore`
is an optional adapter over the maintainer's external gBrain CLI.
"""

from __future__ import annotations

import os
from pathlib import Path

from .gbrain import GBrainMemoryStore, MemoryBackendUnavailable
from .local import LocalMemoryStore
from .store import MemoryStore, SearchResult

__all__ = [
    "GBrainMemoryStore",
    "LocalMemoryStore",
    "MemoryBackendUnavailable",
    "MemoryStore",
    "SearchResult",
    "get_memory_store",
]


def get_memory_store(root: str | Path, backend: str | None = None) -> MemoryStore:
    """Return a `MemoryStore` for `root`.

    `backend` selects the implementation ("local" or "gbrain"); if omitted,
    falls back to the `HTC_MEMORY_BACKEND` env var, defaulting to "local" —
    the self-contained backend HTC works with out of the box.
    """
    resolved = backend or os.environ.get("HTC_MEMORY_BACKEND", "local")
    if resolved == "gbrain":
        return GBrainMemoryStore()
    return LocalMemoryStore(root)

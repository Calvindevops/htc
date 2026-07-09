"""Memory: persistent, queryable storage for ingested `SourceChunk`s.

`LocalMemoryStore` is the default, self-contained backend (no external
service required — hybrid BM25+semantic retrieval, works offline out of the
box; semantic retrieval is opt-in via `HTC_EMBED_*` env vars).
`GBrainMemoryStore` and `SupermemoryMemoryStore` are optional adapters over
external services. A custom backend (any dotted class path satisfying the
`MemoryStore` protocol) can also be plugged in via `HTC_MEMORY_BACKEND`.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from .gbrain import GBrainMemoryStore, MemoryBackendUnavailable
from .local import LocalMemoryStore
from .store import MemoryStore, SearchResult
from .supermemory import SupermemoryMemoryStore

__all__ = [
    "GBrainMemoryStore",
    "LocalMemoryStore",
    "MemoryBackendUnavailable",
    "MemoryStore",
    "SearchResult",
    "SupermemoryMemoryStore",
    "get_memory_store",
]

_PROTOCOL_METHODS = ("add_chunks", "search", "has_source", "count")


def get_memory_store(root: str | Path, backend: str | None = None) -> MemoryStore:
    """Return a `MemoryStore` for `root`.

    `backend` selects the implementation; if omitted, falls back to the
    `HTC_MEMORY_BACKEND` env var, defaulting to "local" — the self-contained
    hybrid backend HTC works with out of the box. Built-ins: "local",
    "gbrain", "supermemory". Anything else is treated as a dotted class path
    (e.g. "mypkg.mymod.MyStore") and imported dynamically — it must satisfy
    the `MemoryStore` protocol.
    """
    resolved = backend or os.environ.get("HTC_MEMORY_BACKEND", "local")
    if resolved == "local":
        return LocalMemoryStore(root)
    if resolved == "gbrain":
        return GBrainMemoryStore()
    if resolved == "supermemory":
        return SupermemoryMemoryStore()
    return _load_custom_backend(resolved, root)


def _load_custom_backend(dotted_path: str, root: str | Path) -> MemoryStore:
    """Import and instantiate a custom `MemoryStore` from a dotted class path."""
    if "." not in dotted_path:
        raise MemoryBackendUnavailable(
            f"unknown memory backend '{dotted_path}' (expected 'local', 'gbrain', "
            "'supermemory', or a dotted class path like 'mypkg.mymod.MyStore')"
        )
    module_path, _, class_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as err:
        raise MemoryBackendUnavailable(
            f"could not import memory backend module '{module_path}': {err}"
        ) from err
    try:
        cls = getattr(module, class_name)
    except AttributeError as err:
        raise MemoryBackendUnavailable(
            f"module '{module_path}' has no class '{class_name}'"
        ) from err

    try:
        instance = cls()
    except TypeError:
        try:
            instance = cls(root)
        except TypeError as err:
            raise MemoryBackendUnavailable(
                f"could not instantiate '{dotted_path}' — its constructor must accept "
                "either no arguments or a single `root` argument"
            ) from err

    missing = [method for method in _PROTOCOL_METHODS if not hasattr(instance, method)]
    if missing:
        raise MemoryBackendUnavailable(
            f"'{dotted_path}' does not satisfy the MemoryStore protocol "
            f"(missing: {', '.join(missing)})"
        )
    return instance

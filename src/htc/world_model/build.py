"""Thin composition: ingest sources (Phase 1) into a memory store (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from ..adapters.base import Source
from .ingest import ingest_sources
from .memory import MemoryStore, get_memory_store


def build_memory(sources: list[Source], root: str | Path, backend: str = "local") -> MemoryStore:
    """Ingest `sources` under `root` and load the resulting chunks into a memory store."""
    corpus = ingest_sources(sources, root=Path(root))
    store = get_memory_store(root, backend=backend)
    store.add_chunks(corpus.all_chunks())
    return store

"""Thin composition: ingest sources (Phase 1) into a memory store (Phase 2)."""

from __future__ import annotations

import os
from pathlib import Path

from ..adapters.base import Source
from .ingest import ingest_sources
from .memory import MemoryStore, get_memory_store


def build_memory(
    sources: list[Source],
    root: str | Path,
    backend: str = "local",
    *,
    contextual: bool = False,
    model: str | None = None,
) -> MemoryStore:
    """Ingest `sources` under `root` and load the resulting chunks into a memory store.

    When `contextual` (or `HTC_CONTEXTUAL_RETRIEVAL=1`), each chunk is enriched
    with an Anthropic-style context blurb before embedding — one LLM call per
    chunk, so it's opt-in. Default is byte-for-byte the prior behavior.
    """
    corpus = ingest_sources(sources, root=Path(root))
    chunks = corpus.all_chunks()
    if contextual or os.environ.get("HTC_CONTEXTUAL_RETRIEVAL") == "1":
        from .ingest.contextual import contextualize_chunks

        # Reconstruct each source's full text from its (contiguous) chunks so the
        # context prompt sees the whole document.
        full_texts = {
            path: "".join(c.text for c in cs) for path, cs in corpus.chunks_by_path.items()
        }
        chunks = contextualize_chunks(chunks, full_texts, model=model)
    store = get_memory_store(root, backend=backend)
    store.add_chunks(chunks)
    return store

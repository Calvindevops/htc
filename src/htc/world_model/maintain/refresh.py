"""Incremental memory refresh — gBrain's self-healing loop for HTC memory.

Only new/changed sources are (re-)extracted, chunked, deduped, and embedded;
deleted sources have their chunks pruned; unchanged ("fresh") sources are
left completely untouched (no re-extraction, no re-embedding). Never a full
rebuild.
"""

from __future__ import annotations

from pathlib import Path

from ...adapters.base import Source
from ..ingest.chunker import chunk_text
from ..ingest.corpus import iter_ingestible_files
from ..ingest.extractors import extract_text
from ..ingest.model import SourceChunk
from ..memory.store import MemoryStore
from .dedup import dedup_chunks
from .staleness import check_staleness
from .state import content_hash, load_manifest, save_manifest


def _chunk_file(path: Path, root: Path) -> list[SourceChunk]:
    rel = path.relative_to(root).as_posix()
    text = extract_text(path)
    return chunk_text(text, source_path=rel, kind="docs")


def _remove_source(memory: MemoryStore, source_path: str) -> int:
    """Best-effort removal: only backends exposing `remove_source` (e.g.
    `LocalMemoryStore`) support pruning; others are left untouched."""
    remove = getattr(memory, "remove_source", None)
    if remove is None:
        return 0
    return remove(source_path)


def refresh_memory(memory: MemoryStore, sources: list[Source], root: str | Path) -> dict:
    """Refresh `memory` in place against the current on-disk state of
    `sources`. Returns a summary dict of counts:
    `new`/`changed`/`deleted`/`fresh`/`chunks_added`/`chunks_removed`/`deduped`.
    """
    root_path = Path(root).expanduser().resolve()
    manifest = load_manifest(root_path)
    staleness = check_staleness(sources, root_path, manifest)

    current_by_path = {
        f.relative_to(root_path).as_posix(): f for f in iter_ingestible_files(sources, root_path)
    }

    chunks_removed = 0
    for rel_path in staleness["deleted"]:
        chunks_removed += _remove_source(memory, rel_path)
        manifest.pop(rel_path, None)

    to_reingest = staleness["new"] + staleness["changed"]
    for rel_path in staleness["changed"]:
        chunks_removed += _remove_source(memory, rel_path)

    new_chunks: list[SourceChunk] = []
    for rel_path in to_reingest:
        new_chunks.extend(_chunk_file(current_by_path[rel_path], root_path))

    deduped = dedup_chunks(new_chunks)
    num_deduped = len(new_chunks) - len(deduped)

    if deduped:
        memory.add_chunks(deduped)

    for rel_path in to_reingest:
        chunk_ids = [chunk.id for chunk in deduped if chunk.source_path == rel_path]
        manifest[rel_path] = {
            "hash": content_hash(current_by_path[rel_path]),
            "chunk_ids": chunk_ids,
        }

    save_manifest(root_path, manifest)

    return {
        "new": len(staleness["new"]),
        "changed": len(staleness["changed"]),
        "deleted": len(staleness["deleted"]),
        "fresh": len(staleness["fresh"]),
        "chunks_added": len(deduped),
        "chunks_removed": chunks_removed,
        "deduped": num_deduped,
    }

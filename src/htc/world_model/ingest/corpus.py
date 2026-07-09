"""Corpus — ingested SourceChunks indexed by source path.

This is the generalized grounding surface for goldens: a golden's `artifact`
is valid if it is a real filesystem file (today's check, unchanged) OR a path
the corpus actually ingested (docs, transcripts, decks — anything under a
`docs`-kind `Source`). Repo-only mode never builds a corpus, so its behavior
is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...adapters.base import Source
from .chunker import chunk_text
from .extractors import KNOWN_SUFFIXES, MissingDependencyError, UnsupportedFormatError, extract_text
from .model import SourceChunk


@dataclass(frozen=True)
class Corpus:
    """Ingested SourceChunks indexed by source path (relative to the ingest root)."""

    chunks_by_path: dict[str, list[SourceChunk]] = field(default_factory=dict)

    def has_source(self, path: str) -> bool:
        return path.lstrip("/") in self.chunks_by_path

    def chunks_for(self, path: str) -> list[SourceChunk]:
        return self.chunks_by_path.get(path.lstrip("/"), [])

    def all_chunks(self) -> list[SourceChunk]:
        return [chunk for chunks in self.chunks_by_path.values() for chunk in chunks]


def _ingest_file(path: Path, root: Path) -> list[SourceChunk]:
    if path.suffix.lower() not in KNOWN_SUFFIXES:
        return []
    try:
        text = extract_text(path)
    except (UnsupportedFormatError, MissingDependencyError):
        return []
    rel = path.relative_to(root).as_posix()
    return chunk_text(text, source_path=rel, kind="docs")


def ingest_sources(sources: list[Source], root: Path) -> Corpus:
    """Ingest `sources` (files or directories, relative to `root` unless
    absolute) into a `Corpus` of `SourceChunk`s.

    Directories are walked recursively; files with an unrecognized extension
    are skipped rather than erroring the whole ingest (a mixed-format docs
    folder is the common case).
    """
    chunks_by_path: dict[str, list[SourceChunk]] = {}
    for source in sources:
        location = Path(source.path)
        if not location.is_absolute():
            location = root / location
        if location.is_file():
            chunks = _ingest_file(location, root)
            if chunks:
                chunks_by_path[chunks[0].source_path] = chunks
        elif location.is_dir():
            for file_path in sorted(location.rglob("*")):
                if not file_path.is_file():
                    continue
                chunks = _ingest_file(file_path, root)
                if chunks:
                    chunks_by_path[chunks[0].source_path] = chunks
    return Corpus(chunks_by_path=chunks_by_path)

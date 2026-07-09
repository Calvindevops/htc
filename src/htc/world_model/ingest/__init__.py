"""Ingest arbitrary local company sources (docs/PDFs/transcripts, not just
repo files) into citable `SourceChunk`s, indexed in a `Corpus`."""

from .chunker import MAX_CHARS, chunk_text
from .contextual import contextualize_chunks
from .corpus import Corpus, ingest_sources, iter_ingestible_files
from .extractors import MissingDependencyError, UnsupportedFormatError, extract_text
from .model import SourceChunk, chunk_id

__all__ = [
    "MAX_CHARS",
    "Corpus",
    "MissingDependencyError",
    "SourceChunk",
    "UnsupportedFormatError",
    "chunk_id",
    "chunk_text",
    "contextualize_chunks",
    "extract_text",
    "ingest_sources",
    "iter_ingestible_files",
]

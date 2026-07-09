"""Near-duplicate chunk collapsing — gBrain's self-healing "dedup" step.

Two passes, both deterministic:
1. Exact match after whitespace/case normalization (hash-based).
2. High-overlap match — token Jaccard similarity > 0.9 against every chunk
   already kept — the later chunk is dropped.

Chunks are processed in their given order; the FIRST occurrence of a
near-duplicate wins, later ones are dropped. No randomness, no wall-clock.
"""

from __future__ import annotations

import hashlib
import re

from ..ingest.model import SourceChunk

_WHITESPACE = re.compile(r"\s+")

_JACCARD_THRESHOLD = 0.9


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


def _normalized_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode()).hexdigest()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def dedup_chunks(chunks: list[SourceChunk]) -> list[SourceChunk]:
    """Return `chunks` with near-identical entries collapsed, preserving the
    order of the first occurrence of each surviving chunk."""
    kept: list[SourceChunk] = []
    kept_hashes: set[str] = set()
    kept_token_sets: list[set[str]] = []

    for chunk in chunks:
        normalized = _normalize(chunk.text)
        digest = _normalized_hash(normalized)
        if digest in kept_hashes:
            continue

        tokens = set(normalized.split())
        if any(_jaccard(tokens, existing) > _JACCARD_THRESHOLD for existing in kept_token_sets):
            continue

        kept.append(chunk)
        kept_hashes.add(digest)
        kept_token_sets.append(tokens)

    return kept

"""SourceChunk — the citation unit for ingested content.

Every chunk carries a stable, deterministic id (a hash of its source path and
offset) so the same source re-ingested twice yields identical ids — no
`random`/`uuid4`-style nondeterminism anywhere in this package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def chunk_id(source_path: str, start_char: int) -> str:
    """Stable id for a chunk: sha256(source_path + start offset), truncated."""
    digest = hashlib.sha256(f"{source_path}:{start_char}".encode()).hexdigest()
    return digest[:16]


@dataclass(frozen=True)
class SourceChunk:
    """One retrievable, citable slice of an ingested source."""

    id: str
    source_path: str
    kind: str
    text: str
    start_char: int
    end_char: int
    # Optional contextualized text (Anthropic-style Contextual Retrieval): when
    # set, the embedder embeds THIS instead of `text`, while `text` stays the
    # original slice for display/citation. `None` (the default) preserves the
    # exact prior behavior everywhere — embed `text` itself.
    embed_text: str | None = None

"""GBrainMemoryStore — optional adapter over the maintainer's external gBrain
CLI. Not required to use HTC: `LocalMemoryStore` is the self-contained
default. This adapter is only useful if you separately run gBrain.

Documented-assumption CLI contract (subject to change upstream):
  gbrain capture --file <path>        -> ingest one file's content
  gbrain query "<q>" --json           -> JSON list of {text, source_path, score, ...}
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..ingest.model import SourceChunk
from .store import SearchResult

if TYPE_CHECKING:
    from ..graph.graph import KnowledgeGraph


class MemoryBackendUnavailable(RuntimeError):
    """Raised when a requested memory backend's dependency isn't present."""


class GBrainMemoryStore:
    """Adapter over the external `gbrain` CLI. Requires `gbrain` on PATH."""

    def __init__(self) -> None:
        if shutil.which("gbrain") is None:
            raise MemoryBackendUnavailable(
                "gBrain backend requested but the 'gbrain' CLI is not installed or not on "
                "PATH. Install gBrain, or use the default local memory store "
                "(backend='local')."
            )

    def add_chunks(self, chunks: list[SourceChunk]) -> None:
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
                tmp_file.write(chunk.text)
                tmp_path = tmp_file.name
            try:
                subprocess.run(
                    ["gbrain", "capture", "--file", tmp_path],
                    check=True,
                    capture_output=True,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def search(
        self, query: str, k: int = 5, graph: KnowledgeGraph | None = None
    ) -> list[SearchResult]:
        result = subprocess.run(
            ["gbrain", "query", query, "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        raw_results = json.loads(result.stdout)
        return [
            SearchResult(
                chunk=SourceChunk(
                    id=item.get("id", ""),
                    source_path=item.get("source_path", ""),
                    kind=item.get("kind", "docs"),
                    text=item.get("text", ""),
                    start_char=item.get("start_char", 0),
                    end_char=item.get("end_char", 0),
                ),
                score=float(item.get("score", 0.0)),
            )
            for item in raw_results[:k]
        ]

    def has_source(self, path: str) -> bool:
        raise NotImplementedError("gBrain backend does not support has_source lookups yet.")

    def count(self) -> int:
        raise NotImplementedError("gBrain backend does not support count yet.")

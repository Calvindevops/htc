"""LocalMemoryStore — the default, self-contained memory backend.

No external service, no ML dependency: retrieval is BM25-style keyword
scoring implemented in pure Python over a lowercase term index. Chunks
persist to a `chunks.jsonl` file so memory survives across sessions.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..ingest.model import SourceChunk
from .store import SearchResult

_TOKEN = re.compile(r"[a-z0-9]+")

# BM25 constants (standard defaults).
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LocalMemoryStore:
    """Keyword-retrieval memory store, persisted to `<root>/.htc/memory/chunks.jsonl`."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._path = self._root / ".htc" / "memory" / "chunks.jsonl"
        self._chunks_by_id: dict[str, SourceChunk] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            chunk = SourceChunk(**data)
            self._chunks_by_id[chunk.id] = chunk

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(chunk), sort_keys=True) for chunk in self._all_sorted()]
        self._path.write_text("\n".join(lines) + ("\n" if lines else ""))

    def _all_sorted(self) -> list[SourceChunk]:
        return sorted(self._chunks_by_id.values(), key=lambda c: c.id)

    def add_chunks(self, chunks: list[SourceChunk]) -> None:
        for chunk in chunks:
            self._chunks_by_id[chunk.id] = chunk
        self._persist()

    def has_source(self, path: str) -> bool:
        target = path.lstrip("/")
        return any(chunk.source_path == target for chunk in self._chunks_by_id.values())

    def count(self) -> int:
        return len(self._chunks_by_id)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query_terms = _tokenize(query)
        if not query_terms or not self._chunks_by_id:
            return []

        chunks = self._all_sorted()
        doc_tokens = [_tokenize(chunk.text) for chunk in chunks]
        doc_freqs = [Counter(tokens) for tokens in doc_tokens]
        doc_lens = [len(tokens) for tokens in doc_tokens]
        n_docs = len(chunks)
        avg_len = (sum(doc_lens) / n_docs) if n_docs else 0.0

        # Document frequency per query term (how many chunks contain it).
        query_term_set = set(query_terms)
        doc_freq_for_term = {
            term: sum(1 for freqs in doc_freqs if term in freqs) for term in query_term_set
        }
        idf = {
            term: math.log(
                (n_docs - doc_freq_for_term[term] + 0.5) / (doc_freq_for_term[term] + 0.5) + 1
            )
            for term in query_term_set
        }

        scored: list[tuple[float, SourceChunk]] = []
        for chunk, freqs, doc_len in zip(chunks, doc_freqs, doc_lens):
            score = 0.0
            for term in query_term_set:
                freq = freqs.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + _K1 * (1 - _B + _B * doc_len / avg_len) if avg_len else freq
                score += idf[term] * (freq * (_K1 + 1)) / denom
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [SearchResult(chunk=chunk, score=score) for score, chunk in scored[:k]]

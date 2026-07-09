"""LocalMemoryStore — the default, self-contained memory backend.

No external service required: retrieval is BM25-style keyword scoring
implemented in pure Python over a lowercase term index — works offline out
of the box. If an OpenAI-compatible embeddings endpoint is configured
(`HTC_EMBED_BASE_URL` + `HTC_EMBED_API_KEY` + `HTC_EMBED_MODEL`), retrieval
becomes hybrid: BM25 and cosine-similarity semantic rankings are fused via
Reciprocal Rank Fusion (RRF). With no embedding endpoint configured, behavior
is unchanged — BM25 only.

Chunks persist to `chunks.jsonl`; embeddings (when computed) persist
alongside in a parallel `embeddings.jsonl`, so they aren't recomputed across
sessions.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ...llm import _post
from ..ingest.model import SourceChunk
from .store import SearchResult

_TOKEN = re.compile(r"[a-z0-9]+")

# BM25 constants (standard defaults).
_K1 = 1.5
_B = 0.75

# Reciprocal Rank Fusion constant (standard default).
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _embed_config() -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) for the embeddings endpoint, or
    `None` if not fully configured — the hybrid path is opt-in."""
    base = os.environ.get("HTC_EMBED_BASE_URL")
    key = os.environ.get("HTC_EMBED_API_KEY")
    model = os.environ.get("HTC_EMBED_MODEL")
    if not base or not key or not model:
        return None
    return base.rstrip("/"), key, model


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed `texts` via the configured OpenAI-compatible `/embeddings` endpoint."""
    config = _embed_config()
    assert config is not None, "_embed called without HTC_EMBED_* configured"
    base, key, model = config
    data = _post(
        f"{base}/embeddings",
        {"content-type": "application/json", "authorization": f"Bearer {key}"},
        {"model": model, "input": texts},
    )
    return [item["embedding"] for item in data["data"]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rrf_fuse(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: combine multiple best-first id rankings into
    one fused score per id. Missing from a ranking simply contributes 0."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, start=1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return scores


class LocalMemoryStore:
    """Hybrid-retrieval memory store, persisted to `<root>/.htc/memory/`."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._path = self._root / ".htc" / "memory" / "chunks.jsonl"
        self._embeddings_path = self._root / ".htc" / "memory" / "embeddings.jsonl"
        self._chunks_by_id: dict[str, SourceChunk] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                chunk = SourceChunk(**data)
                self._chunks_by_id[chunk.id] = chunk
        if self._embeddings_path.exists():
            for line in self._embeddings_path.read_text().splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                self._embeddings[data["id"]] = data["embedding"]

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(chunk), sort_keys=True) for chunk in self._all_sorted()]
        self._path.write_text("\n".join(lines) + ("\n" if lines else ""))

    def _persist_embeddings(self) -> None:
        self._embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"id": id_, "embedding": self._embeddings[id_]}, sort_keys=True)
            for id_ in sorted(self._embeddings)
        ]
        self._embeddings_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    def _all_sorted(self) -> list[SourceChunk]:
        return sorted(self._chunks_by_id.values(), key=lambda c: c.id)

    def add_chunks(self, chunks: list[SourceChunk]) -> None:
        for chunk in chunks:
            self._chunks_by_id[chunk.id] = chunk
        self._persist()

        if _embed_config() is None or not chunks:
            return
        vectors = _embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            self._embeddings[chunk.id] = vector
        self._persist_embeddings()

    def has_source(self, path: str) -> bool:
        target = path.lstrip("/")
        return any(chunk.source_path == target for chunk in self._chunks_by_id.values())

    def count(self) -> int:
        return len(self._chunks_by_id)

    def _bm25_scored(
        self, chunks: list[SourceChunk], query_term_set: set[str]
    ) -> list[tuple[float, SourceChunk]]:
        doc_tokens = [_tokenize(chunk.text) for chunk in chunks]
        doc_freqs = [Counter(tokens) for tokens in doc_tokens]
        doc_lens = [len(tokens) for tokens in doc_tokens]
        n_docs = len(chunks)
        avg_len = (sum(doc_lens) / n_docs) if n_docs else 0.0

        # Document frequency per query term (how many chunks contain it).
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
        return scored

    def _semantic_ranking(self, query: str) -> list[str]:
        """Best-first chunk ids by cosine similarity to `query`'s embedding.
        Only chunks with a stored embedding participate."""
        available_ids = [chunk.id for chunk in self._all_sorted() if chunk.id in self._embeddings]
        if not available_ids:
            return []
        query_vector = _embed([query])[0]
        scored = [(_cosine(query_vector, self._embeddings[id_]), id_) for id_ in available_ids]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [id_ for _, id_ in scored]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query_terms = _tokenize(query)
        if not query_terms or not self._chunks_by_id:
            return []

        chunks = self._all_sorted()
        scored = self._bm25_scored(chunks, set(query_terms))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))

        if _embed_config() is None:
            return [SearchResult(chunk=chunk, score=score) for score, chunk in scored[:k]]

        semantic_ranking = self._semantic_ranking(query)
        if not semantic_ranking:
            return [SearchResult(chunk=chunk, score=score) for score, chunk in scored[:k]]

        bm25_ranking = [chunk.id for _, chunk in scored]
        fused = _rrf_fuse([bm25_ranking, semantic_ranking])
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        ordered_ids = sorted(fused, key=lambda id_: (-fused[id_], id_))
        return [SearchResult(chunk=chunk_by_id[id_], score=fused[id_]) for id_ in ordered_ids[:k]]

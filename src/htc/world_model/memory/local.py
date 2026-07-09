"""LocalMemoryStore — the default, self-contained memory backend.

Retrieval is BM25-style keyword scoring implemented in pure Python over a
lowercase term index — works offline out of the box with zero dependencies.
On top of that, semantic search is on **by default** whenever an embedder is
available, fused with BM25 via Reciprocal Rank Fusion (RRF). Embedder
precedence, highest first:

1. A configured cloud endpoint (`HTC_EMBED_BASE_URL` + `HTC_EMBED_API_KEY` +
   `HTC_EMBED_MODEL`, any OpenAI-compatible `/embeddings` API).
2. Ollama (`HTC_OLLAMA_URL`, default `http://localhost:11434`) — the
   RECOMMENDED default, matching gBrain. Detected by probing the server; the
   default model is `nomic-embed-text` (override with `HTC_EMBED_MODEL`).
3. The bundled `fastembed` package (`pip install htc[embed]`) — a zero-config
   local CPU fallback so HTC never hard-fails when neither cloud nor Ollama
   is reachable.
4. BM25 only — last resort if nothing above is available.

The first time no embedder is configured or reachable, a short interactive
wizard offers to pick one and saves the choice to `~/.htc/config.json` so it
isn't asked again; non-interactive runs (`HTC_EMBED_NONINTERACTIVE=1` or no
TTY) skip the prompt and fall through the precedence order silently.

Chunks persist to `chunks.jsonl`; embeddings (when computed) persist
alongside in a parallel `embeddings.jsonl`, so they aren't recomputed across
sessions.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import httpx

from ...llm import _post
from ..graph.graph import KnowledgeGraph
from ..ingest.model import SourceChunk
from .secrets import load_secret, save_secret
from .store import SearchResult

_TOKEN = re.compile(r"[a-z0-9]+")

# BM25 constants (standard defaults).
_K1 = 1.5
_B = 0.75

# Reciprocal Rank Fusion constant (standard default).
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


_FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_fastembed_model_cache: dict[str, object] = {}

_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "nomic-embed-text"
_OLLAMA_PROBE_TIMEOUT = 0.75

# Global (not per-project) preference file — the first-boot wizard's answer,
# so it's asked at most once per machine.
_CONFIG_PATH = Path.home() / ".htc" / "config.json"

_WIZARD_PROMPT = """
No embedding backend is configured — HTC can search purely by keyword (BM25)
or semantically. Pick one:

  (a) Ollama local [recommended] — free, private (run: ollama pull nomic-embed-text)
  (b) Cloud endpoint — paste your base URL / model / API key (key is saved encrypted)
  (c) fastembed — bundled local CPU model, zero config (pip install htc[embed])
  (d) BM25 only — keyword search, no embeddings

Choice [a/b/c/d]: """

_WIZARD_CHOICES = {"a": "ollama", "b": "cloud", "c": "fastembed", "d": "bm25"}

_CLOUD_BASE_URL_PROMPT = "Cloud base URL (OpenAI-compatible /embeddings API): "
_CLOUD_MODEL_PROMPT = "Model name (e.g. text-embedding-3-small): "
_CLOUD_API_KEY_PROMPT = "API key (saved encrypted, never written to config.json): "


def _embed_config() -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) for the remote cloud embeddings
    endpoint, or `None` if not fully configured. Precedence per field: the
    `HTC_EMBED_*` env vars first; if those aren't all set, fall back to the
    wizard's saved cloud config — `embed_base_url`/`embed_model` from
    `~/.htc/config.json`, and the API key from the encrypted secret store
    (`secrets.load_secret`), never from the plaintext config file."""
    base = os.environ.get("HTC_EMBED_BASE_URL")
    key = os.environ.get("HTC_EMBED_API_KEY")
    model = os.environ.get("HTC_EMBED_MODEL")
    if not (base and key and model):
        config = _load_global_config()
        base = base or config.get("embed_base_url")
        model = model or config.get("embed_model")
        key = key or load_secret("embedding_api_key")
    if not base or not key or not model:
        return None
    return base.rstrip("/"), key, model


def _ollama_base_url() -> str:
    return os.environ.get("HTC_OLLAMA_URL", _OLLAMA_DEFAULT_URL).rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("HTC_EMBED_MODEL", _OLLAMA_DEFAULT_MODEL)


def _ollama_reachable() -> bool:
    """Best-effort probe: is an Ollama server responding at `HTC_OLLAMA_URL`
    (default `http://localhost:11434`)? Makes Ollama the recommended default
    embedder with zero configuration, matching gBrain."""
    try:
        with httpx.Client(timeout=_OLLAMA_PROBE_TIMEOUT) as client:
            response = client.get(f"{_ollama_base_url()}/api/tags")
            return response.status_code < 500
    except httpx.HTTPError:
        return False


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Embed `texts` one at a time via Ollama's `/api/embeddings` endpoint
    (no batch support in that API)."""
    base = _ollama_base_url()
    model = _ollama_model()
    vectors = []
    for text in texts:
        data = _post(
            f"{base}/api/embeddings",
            {"content-type": "application/json"},
            {"model": model, "prompt": text},
        )
        vectors.append(data["embedding"])
    return vectors


def _fastembed_available() -> bool:
    """Whether the optional `fastembed` package (`pip install htc[embed]`)
    is importable — the bundled, zero-config local fallback embedder."""
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def _fastembed_model():
    """Lazily import and cache the local fastembed model instance."""
    if "model" not in _fastembed_model_cache:
        from fastembed import TextEmbedding

        _fastembed_model_cache["model"] = TextEmbedding(model_name=_FASTEMBED_MODEL_NAME)
    return _fastembed_model_cache["model"]


def _fastembed_embed(texts: list[str]) -> list[list[float]]:
    model = _fastembed_model()
    return [[float(x) for x in vector] for vector in model.embed(texts)]


def _load_global_config() -> dict:
    if _CONFIG_PATH.is_file():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_global_config(config: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _prompt_embedder_choice() -> str:
    try:
        answer = input(_WIZARD_PROMPT).strip().lower()
    except (EOFError, OSError):
        return "ollama"
    return _WIZARD_CHOICES.get(answer, "ollama")


def _prompt_cloud_config() -> tuple[str, str, str]:
    """Collect the cloud/BYO provider's base URL, model, and API key. The
    key is only ever returned to the caller so it can be routed through
    `secrets.save_secret` — it must never land in the saved config dict."""
    try:
        base_url = input(_CLOUD_BASE_URL_PROMPT).strip()
        model = input(_CLOUD_MODEL_PROMPT).strip()
        api_key = input(_CLOUD_API_KEY_PROMPT).strip()
    except (EOFError, OSError):
        return "", "", ""
    return base_url, model, api_key


def _maybe_run_wizard(ollama_reachable: bool) -> None:
    """First-boot wizard: only runs when nothing is configured or reachable
    yet (no cloud config, no Ollama, no saved preference) and the session is
    interactive. Saves the choice to `~/.htc/config.json` so it's asked at
    most once. Non-interactive runs (`HTC_EMBED_NONINTERACTIVE=1` or no TTY)
    skip the prompt silently and fall through the precedence order. If the
    cloud/BYO provider is chosen, its API key is saved ENCRYPTED via
    `secrets.save_secret` — only the non-secret base URL/model land in the
    plaintext `~/.htc/config.json`."""
    if ollama_reachable:
        return
    config = _load_global_config()
    if "embed_backend" in config:
        return
    if os.environ.get("HTC_EMBED_NONINTERACTIVE") == "1" or not sys.stdin.isatty():
        return
    choice = _prompt_embedder_choice()
    updated = {**config, "embed_backend": choice}
    if choice == "cloud":
        base_url, model, api_key = _prompt_cloud_config()
        updated["embed_base_url"] = base_url
        updated["embed_model"] = model
        if api_key:
            save_secret("embedding_api_key", api_key)
    _save_global_config(updated)


def _saved_preference_is_bm25() -> bool:
    """Whether the user explicitly opted into BM25-only via the wizard —
    an explicit choice that overrides Ollama/fastembed auto-detection."""
    return _load_global_config().get("embed_backend") == "bm25"


def _embedder_available() -> bool:
    """Whether ANY embedder is available, in precedence order: a configured
    cloud endpoint, a reachable Ollama server, or the bundled fastembed
    fallback. If none, retrieval is BM25-only."""
    if _embed_config() is not None:
        return True
    ollama_ok = _ollama_reachable()
    _maybe_run_wizard(ollama_ok)
    if _saved_preference_is_bm25():
        return False
    return ollama_ok or _fastembed_available()


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed `texts`: cloud endpoint > Ollama > bundled fastembed model."""
    config = _embed_config()
    if config is not None:
        base, key, model = config
        data = _post(
            f"{base}/embeddings",
            {"content-type": "application/json", "authorization": f"Bearer {key}"},
            {"model": model, "input": texts},
        )
        return [item["embedding"] for item in data["data"]]
    if _ollama_reachable():
        return _ollama_embed(texts)
    assert _fastembed_available(), "_embed called with no embedder available"
    return _fastembed_embed(texts)


def _cosine(a: list[float], b: list[float]) -> float | None:
    """Cosine similarity, or `None` if `a`/`b` have mismatched dimensionality
    (e.g. a stale embedding left over from a different embedder/model) —
    callers must skip rather than mix dimensions, which would otherwise
    silently produce garbage similarity scores."""
    if len(a) != len(b):
        return None
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

        if not _embedder_available() or not chunks:
            return
        vectors = _embed([chunk.embed_text or chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            self._embeddings[chunk.id] = vector
        self._persist_embeddings()

    def remove_source(self, source_path: str) -> int:
        """Remove every chunk (and stored embedding) for `source_path`.
        Returns the count removed. Used by the maintenance/refresh layer to
        replace a stale source's chunks without rebuilding the whole store."""
        target = source_path.lstrip("/")
        stale_ids = [
            chunk_id
            for chunk_id, chunk in self._chunks_by_id.items()
            if chunk.source_path == target
        ]
        for chunk_id in stale_ids:
            del self._chunks_by_id[chunk_id]
            self._embeddings.pop(chunk_id, None)
        if stale_ids:
            self._persist()
            self._persist_embeddings()
        return len(stale_ids)

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
        Only chunks with a stored embedding participate. Stored embeddings
        whose dimensionality doesn't match the query's (e.g. a stale
        `embeddings.jsonl` left over from a different embedder/model) are
        skipped rather than mixed in, which would otherwise silently produce
        garbage similarity scores."""
        available_ids = [chunk.id for chunk in self._all_sorted() if chunk.id in self._embeddings]
        if not available_ids:
            return []
        query_vector = _embed([query])[0]
        scored = []
        skipped_dim_mismatch = 0
        for id_ in available_ids:
            similarity = _cosine(query_vector, self._embeddings[id_])
            if similarity is None:
                skipped_dim_mismatch += 1
                continue
            scored.append((similarity, id_))
        if skipped_dim_mismatch:
            print(
                f"  [htc] skipped {skipped_dim_mismatch} stored embedding(s) with mismatched "
                "dimensionality (stale embeddings.jsonl?)",
                file=sys.stderr,
            )
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [id_ for _, id_ in scored]

    def _graph_ranking(
        self, query: str, chunks: list[SourceChunk], graph: KnowledgeGraph
    ) -> list[str]:
        """Best-first chunk ids that mention a query-relevant graph entity
        (from `graph.subgraph_for_query`) in their path or text — an
        ADDITIONAL signal fused via RRF, never a replacement for BM25/semantic
        ranking. Chunks with no graph-entity match are simply absent."""
        relevant_names = {entity.name for entity in graph.subgraph_for_query(query, k=20)}
        if not relevant_names:
            return []
        scored: list[tuple[int, str]] = []
        for chunk in chunks:
            hits = sum(
                1 for name in relevant_names if name in chunk.source_path or name in chunk.text
            )
            if hits > 0:
                scored.append((hits, chunk.id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [chunk_id for _, chunk_id in scored]

    def search(
        self, query: str, k: int = 5, graph: KnowledgeGraph | None = None
    ) -> list[SearchResult]:
        query_terms = _tokenize(query)
        if not query_terms or not self._chunks_by_id:
            return []

        chunks = self._all_sorted()
        scored = self._bm25_scored(chunks, set(query_terms))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        bm25_ranking = [chunk.id for _, chunk in scored]

        rankings = [bm25_ranking]
        if _embedder_available():
            semantic_ranking = self._semantic_ranking(query)
            if semantic_ranking:
                rankings.append(semantic_ranking)
        if graph is not None:
            graph_ranking = self._graph_ranking(query, chunks, graph)
            if graph_ranking:
                rankings.append(graph_ranking)

        if len(rankings) == 1:
            return [SearchResult(chunk=chunk, score=score) for score, chunk in scored[:k]]

        fused = _rrf_fuse(rankings)
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        ordered_ids = sorted(fused, key=lambda id_: (-fused[id_], id_))
        return [SearchResult(chunk=chunk_by_id[id_], score=fused[id_]) for id_ in ordered_ids[:k]]

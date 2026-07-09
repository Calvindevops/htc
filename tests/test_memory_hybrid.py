"""Hybrid retrieval (BM25 + optional semantic via RRF), the Supermemory
adapter guard, and the pluggable backend registry (built-ins + custom
dotted-class paths). No network: the embeddings HTTP call is monkeypatched
to return canned vectors."""

from __future__ import annotations

import sys
import types

import pytest

from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory import (
    MemoryBackendUnavailable,
    SupermemoryMemoryStore,
    get_memory_store,
)
from htc.world_model.memory import local as local_module


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


def _fake_post_factory(vectors_by_text: dict[str, list[float]]):
    """Canned stand-in for `htc.llm._post`, mimicking an OpenAI-compatible
    `/embeddings` response shape."""

    def _fake_post(url, headers, body):
        inputs = body["input"]
        return {"data": [{"embedding": vectors_by_text[text]} for text in inputs]}

    return _fake_post


class TestHybridRetrieval:
    def test_hybrid_fuses_bm25_and_semantic_and_changes_ranking(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HTC_EMBED_BASE_URL", "https://embed.example.com")
        monkeypatch.setenv("HTC_EMBED_API_KEY", "test-key")
        monkeypatch.setenv("HTC_EMBED_MODEL", "test-embed-model")

        # "a" shares no keywords with the query but is semantically identical
        # (same canned vector as the query). "b" shares the keyword "refund"
        # but is semantically unrelated (orthogonal vector).
        texts = {
            "refund policy semantics": [1.0, 0.0, 0.0],  # query
            "totally unrelated content": [1.0, 0.0, 0.0],  # chunk a (semantic match)
            "refund policy details": [0.0, 1.0, 0.0],  # chunk b (BM25 match)
        }
        monkeypatch.setattr(local_module, "_post", _fake_post_factory(texts))

        store = local_module.LocalMemoryStore(tmp_path)
        store.add_chunks(
            [
                _chunk("a", "a.md", "totally unrelated content"),
                _chunk("b", "b.md", "refund policy details"),
            ]
        )

        # BM25-only ranking (no query terms match chunk "a" at all).
        bm25_scored = store._bm25_scored(
            store._all_sorted(), set(local_module._tokenize("refund policy semantics"))
        )
        bm25_scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        bm25_only_ranking = [chunk.id for _, chunk in bm25_scored]
        assert bm25_only_ranking == ["b"]  # "a" never appears — zero keyword overlap

        results = store.search("refund policy semantics", k=2)
        result_ids = [r.chunk.id for r in results]

        # Hybrid surfaces "a" (semantic match) even though BM25 alone would not.
        assert "a" in result_ids
        assert set(result_ids) != set(bm25_only_ranking)

    def test_no_embed_config_is_byte_for_byte_bm25(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HTC_EMBED_BASE_URL", raising=False)
        monkeypatch.delenv("HTC_EMBED_API_KEY", raising=False)
        monkeypatch.delenv("HTC_EMBED_MODEL", raising=False)

        store = local_module.LocalMemoryStore(tmp_path)
        store.add_chunks(
            [
                _chunk("a", "refunds.md", "All refunds are processed within 30 days."),
                _chunk("b", "shipping.md", "Shipping takes 5 to 7 business days."),
                _chunk("c", "returns.md", "Our refund policy covers unopened items."),
            ]
        )
        results = store.search("refund policy", k=2)
        assert results
        assert results[0].chunk.id in ("a", "c")
        assert results[0].chunk.id != "b"
        # No embeddings file should be written when no embed endpoint is configured.
        assert not (tmp_path / ".htc" / "memory" / "embeddings.jsonl").exists()

    def test_embeddings_persist_across_reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HTC_EMBED_BASE_URL", "https://embed.example.com")
        monkeypatch.setenv("HTC_EMBED_API_KEY", "test-key")
        monkeypatch.setenv("HTC_EMBED_MODEL", "test-embed-model")

        calls = {"count": 0}
        texts = {"hello world": [0.1, 0.2, 0.3]}

        def _counting_fake_post(url, headers, body):
            calls["count"] += 1
            return {"data": [{"embedding": texts[t]} for t in body["input"]]}

        monkeypatch.setattr(local_module, "_post", _counting_fake_post)

        store = local_module.LocalMemoryStore(tmp_path)
        store.add_chunks([_chunk("a", "x.md", "hello world")])
        assert calls["count"] == 1
        assert (tmp_path / ".htc" / "memory" / "embeddings.jsonl").exists()

        reloaded = local_module.LocalMemoryStore(tmp_path)
        assert reloaded._embeddings == {"a": [0.1, 0.2, 0.3]}

    def test_rrf_fuse_basic(self):
        fused = local_module._rrf_fuse([["a", "b", "c"], ["b", "a"]])
        # "a": rank1 in first + rank2 in second; "b": rank2 in first + rank1 in second.
        assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert fused["c"] == pytest.approx(1 / 63)


class TestSupermemoryMemoryStore:
    def test_raises_clean_error_when_key_absent(self, monkeypatch):
        monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
        with pytest.raises(MemoryBackendUnavailable, match="SUPERMEMORY_API_KEY"):
            SupermemoryMemoryStore()

    def test_constructs_with_key(self, monkeypatch):
        monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm_test_key")
        store = SupermemoryMemoryStore()
        assert store._base == "https://api.supermemory.ai"

    def test_respects_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm_test_key")
        monkeypatch.setenv("SUPERMEMORY_BASE_URL", "http://localhost:8787/")
        store = SupermemoryMemoryStore()
        assert store._base == "http://localhost:8787"


class _DummyCustomStore:
    """Minimal Protocol-satisfying store, used to test dynamic backend loading."""

    def __init__(self) -> None:
        self.added: list[SourceChunk] = []

    def add_chunks(self, chunks: list[SourceChunk]) -> None:
        self.added.extend(chunks)

    def search(self, query: str, k: int = 5) -> list:
        return []

    def has_source(self, path: str) -> bool:
        return False

    def count(self) -> int:
        return len(self.added)


class TestGetMemoryStoreCustomBackend:
    def test_loads_custom_dotted_class(self, tmp_path):
        module = types.ModuleType("htc_test_custom_backend_module")
        module.DummyCustomStore = _DummyCustomStore
        sys.modules["htc_test_custom_backend_module"] = module
        try:
            store = get_memory_store(
                tmp_path, backend="htc_test_custom_backend_module.DummyCustomStore"
            )
            assert isinstance(store, _DummyCustomStore)
            assert store.count() == 0
        finally:
            del sys.modules["htc_test_custom_backend_module"]

    def test_env_var_selects_custom_backend(self, tmp_path, monkeypatch):
        module = types.ModuleType("htc_test_custom_backend_module_env")
        module.DummyCustomStore = _DummyCustomStore
        sys.modules["htc_test_custom_backend_module_env"] = module
        try:
            monkeypatch.setenv(
                "HTC_MEMORY_BACKEND", "htc_test_custom_backend_module_env.DummyCustomStore"
            )
            store = get_memory_store(tmp_path)
            assert isinstance(store, _DummyCustomStore)
        finally:
            del sys.modules["htc_test_custom_backend_module_env"]

    def test_bad_dotted_path_raises_clear_error(self, tmp_path):
        with pytest.raises(MemoryBackendUnavailable, match="could not import"):
            get_memory_store(tmp_path, backend="does.not.exist.DummyStore")

    def test_missing_class_raises_clear_error(self, tmp_path):
        with pytest.raises(MemoryBackendUnavailable, match="no class"):
            get_memory_store(tmp_path, backend="htc.world_model.memory.local.NotAClass")

    def test_unknown_single_word_backend_raises_clear_error(self, tmp_path):
        with pytest.raises(MemoryBackendUnavailable, match="unknown memory backend"):
            get_memory_store(tmp_path, backend="notarealbackend")

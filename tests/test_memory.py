"""Memory: LocalMemoryStore (BM25 retrieval + persistence), GBrainMemoryStore
guard, get_memory_store factory, and build_memory wiring — no external
services, no network."""

from __future__ import annotations

import pytest

from htc.adapters.base import Source
from htc.world_model.build import build_memory
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory import (
    GBrainMemoryStore,
    LocalMemoryStore,
    MemoryBackendUnavailable,
    get_memory_store,
)


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class TestLocalMemoryStore:
    def test_search_ranks_relevant_chunk_first(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
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

    def test_search_empty_query_or_empty_store_returns_empty(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        assert store.search("anything") == []
        store.add_chunks([_chunk("a", "x.md", "some text here")])
        assert store.search("") == []

    def test_bm25_scoring_is_deterministic_across_runs(self, tmp_path):
        chunks = [
            _chunk("a", "one.md", "refund policy details for orders"),
            _chunk("b", "two.md", "shipping and delivery timelines"),
            _chunk("c", "three.md", "refund policy exceptions and edge cases"),
        ]
        store1 = LocalMemoryStore(tmp_path / "s1")
        store1.add_chunks(chunks)
        store2 = LocalMemoryStore(tmp_path / "s2")
        store2.add_chunks(chunks)
        results1 = store1.search("refund policy")
        results2 = store2.search("refund policy")
        assert [(r.chunk.id, r.score) for r in results1] == [
            (r.chunk.id, r.score) for r in results2
        ]

    def test_persistence_round_trips(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.add_chunks(
            [
                _chunk("a", "refunds.md", "All refunds are processed within 30 days."),
                _chunk("b", "shipping.md", "Shipping takes 5 to 7 business days."),
            ]
        )
        assert (tmp_path / ".htc" / "memory" / "chunks.jsonl").exists()

        reloaded = LocalMemoryStore(tmp_path)
        assert reloaded.count() == 2
        assert reloaded.search("refunds") == store.search("refunds")

    def test_has_source_and_count(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        assert store.count() == 0
        assert store.has_source("refunds.md") is False
        store.add_chunks([_chunk("a", "refunds.md", "refund text")])
        assert store.count() == 1
        assert store.has_source("refunds.md") is True
        assert store.has_source("ghost.md") is False

    def test_add_chunks_dedupes_by_id(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.add_chunks([_chunk("a", "x.md", "first version")])
        store.add_chunks([_chunk("a", "x.md", "updated version")])
        assert store.count() == 1


class TestGetMemoryStore:
    def test_returns_local_by_default(self, tmp_path):
        store = get_memory_store(tmp_path)
        assert isinstance(store, LocalMemoryStore)

    def test_returns_local_explicitly(self, tmp_path):
        store = get_memory_store(tmp_path, backend="local")
        assert isinstance(store, LocalMemoryStore)

    def test_env_var_selects_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HTC_MEMORY_BACKEND", "local")
        store = get_memory_store(tmp_path)
        assert isinstance(store, LocalMemoryStore)

    def test_gbrain_backend_raises_when_binary_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(MemoryBackendUnavailable):
            get_memory_store(tmp_path, backend="gbrain")


class TestGBrainMemoryStore:
    def test_raises_clean_error_when_gbrain_absent(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(MemoryBackendUnavailable, match="gbrain"):
            GBrainMemoryStore()


class TestBuildMemory:
    def test_builds_memory_from_ingested_sources(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "policy.md").write_text("Refunds are honored within 30 days of purchase.")

        store = build_memory([Source(path=str(docs), kind="docs")], root=tmp_path)

        assert store.count() >= 1
        assert store.has_source("docs/policy.md")
        results = store.search("refunds purchase")
        assert results
        assert results[0].chunk.source_path == "docs/policy.md"

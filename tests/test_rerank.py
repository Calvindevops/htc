"""Reranker layer: `NoOpReranker` passthrough, `get_reranker` selection, a
cloud reranker reordering + gracefully degrading on failure,
`RerankerUnavailable` when a key is absent, and `search_with_rerank` parity
with plain `store.search`. No network — httpx calls are monkeypatched."""

from __future__ import annotations

import httpx
import pytest

from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.store import SearchResult
from htc.world_model.rerank import (
    CohereReranker,
    NoOpReranker,
    RerankerUnavailable,
    ZeroEntropyReranker,
    get_reranker,
    search_with_rerank,
)
from htc.world_model.rerank import cloud as cloud_module


def _result(id_: str, text: str, score: float = 0.0) -> SearchResult:
    chunk = SourceChunk(
        id=id_, source_path=f"{id_}.md", kind="docs", text=text, start_char=0, end_char=len(text)
    )
    return SearchResult(chunk=chunk, score=score)


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_body: dict | None = None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


class _FakeStore:
    """Minimal `MemoryStore` stand-in that records every `search` call."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.search_calls: list[dict] = []

    def add_chunks(self, chunks) -> None:
        pass

    def search(self, query: str, k: int = 5, **kwargs) -> list[SearchResult]:
        self.search_calls.append({"query": query, "k": k, **kwargs})
        return self._results[:k]

    def has_source(self, path: str) -> bool:
        return False

    def count(self) -> int:
        return len(self._results)


class TestNoOpReranker:
    def test_passthrough_order_and_top_k_truncation(self):
        results = [_result("a", "a"), _result("b", "b"), _result("c", "c")]
        reranked = NoOpReranker().rerank("query", results, top_k=2)
        assert [r.chunk.id for r in reranked] == ["a", "b"]


class TestGetReranker:
    def test_default_is_noop(self, monkeypatch):
        monkeypatch.delenv("HTC_RERANKER", raising=False)
        assert isinstance(get_reranker(), NoOpReranker)

    def test_env_selects_backend(self, monkeypatch):
        monkeypatch.setenv("HTC_RERANKER", "zerank")
        monkeypatch.setenv("ZEROENTROPY_API_KEY", "ze-test-key")
        assert isinstance(get_reranker(), ZeroEntropyReranker)

    def test_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("HTC_RERANKER", "zerank")
        assert isinstance(get_reranker("none"), NoOpReranker)

    def test_unknown_backend_raises_clear_error(self):
        with pytest.raises(RerankerUnavailable, match="unknown reranker"):
            get_reranker("madeup")


class TestCloudRerankerKeyRequired:
    def test_zeroentropy_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("ZEROENTROPY_API_KEY", raising=False)
        monkeypatch.setattr(cloud_module, "load_secret", lambda name: None)
        with pytest.raises(RerankerUnavailable, match="ZEROENTROPY_API_KEY"):
            ZeroEntropyReranker()

    def test_cohere_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.setattr(cloud_module, "load_secret", lambda name: None)
        with pytest.raises(RerankerUnavailable, match="COHERE_API_KEY"):
            CohereReranker()


class TestCloudRerankerReorderAndDegrade:
    def test_reorders_by_relevance_score(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "co-test-key")
        results = [
            _result("a", "unrelated"),
            _result("b", "very relevant"),
            _result("c", "somewhat relevant"),
        ]

        def fake_post(self, url, headers=None, json=None):
            assert url == CohereReranker._URL
            assert json["documents"] == ["unrelated", "very relevant", "somewhat relevant"]
            return _FakeResponse(
                200,
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 2, "relevance_score": 0.5},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                },
            )

        monkeypatch.setattr("httpx.Client.post", fake_post)
        reranked = CohereReranker().rerank("query", results, top_k=2)
        assert [r.chunk.id for r in reranked] == ["b", "c"]
        assert reranked[0].score == pytest.approx(0.9)

    def test_degrades_to_passthrough_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "co-test-key")
        results = [_result("a", "a"), _result("b", "b")]

        def fake_post(self, url, headers=None, json=None):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr("httpx.Client.post", fake_post)
        with pytest.warns(UserWarning, match="passthrough"):
            reranked = CohereReranker().rerank("query", results, top_k=2)
        assert [r.chunk.id for r in reranked] == ["a", "b"]


class TestSearchWithRerank:
    def test_none_reranker_is_identical_to_plain_search(self):
        results = [_result("a", "a"), _result("b", "b"), _result("c", "c")]
        store = _FakeStore(results)
        direct = store.search("q", k=2)
        store.search_calls.clear()

        reranked = search_with_rerank(store, "q", k=2, reranker=None)

        assert [r.chunk.id for r in reranked] == [r.chunk.id for r in direct]
        assert store.search_calls == [{"query": "q", "k": 2}]

    def test_noop_reranker_is_identical_to_plain_search(self):
        results = [_result("a", "a"), _result("b", "b"), _result("c", "c")]
        store = _FakeStore(results)

        reranked = search_with_rerank(store, "q", k=2, reranker=NoOpReranker())

        assert [r.chunk.id for r in reranked] == ["a", "b"]
        assert store.search_calls == [{"query": "q", "k": 2}]

    def test_reranker_expands_pool_then_truncates(self):
        results = [_result(str(i), str(i)) for i in range(30)]
        store = _FakeStore(results)

        class _ReverseReranker:
            def rerank(self, query, results, top_k):
                return list(reversed(results))[:top_k]

        reranked = search_with_rerank(store, "q", k=3, reranker=_ReverseReranker())

        # Default pool = max(k * 4, 20) = 20, so the pool is ids "0".."19".
        assert store.search_calls == [{"query": "q", "k": 20}]
        assert [r.chunk.id for r in reranked] == ["19", "18", "17"]

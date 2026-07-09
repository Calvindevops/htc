"""Query transformation: LLM-driven retrieval-query transforms (expand,
hyde, decompose, multi_query) with deterministic guard-rails on bad/empty
model replies, and `retrieve_with_transform`'s "none" passthrough + RRF
fusion across per-variant retrieval for the multi-query strategies — no
network (`complete()` is monkeypatched)."""

from __future__ import annotations

import json

import pytest

from htc.llm import LLMResponse
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.store import SearchResult
from htc.world_model.query import decompose, expand, hyde, multi_query, retrieve_with_transform
from htc.world_model.query import transform as transform_module


def _chunk(id_: str, text: str = "text") -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=f"{id_}.md", kind="docs", text=text, start_char=0, end_char=len(text)
    )


def _result(id_: str, score: float = 0.0) -> SearchResult:
    return SearchResult(chunk=_chunk(id_), score=score)


class _FakeStore:
    """Returns a fixed ranking per query (default: none), recording every
    `search` call. Mimics `LocalMemoryStore.search`'s signature."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]] | None = None):
        self._results_by_query = results_by_query or {}
        self.search_calls: list[dict] = []

    def add_chunks(self, chunks):  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def search(self, query: str, k: int = 5, **kwargs) -> list[SearchResult]:
        self.search_calls.append({"query": query, "k": k, **kwargs})
        return self._results_by_query.get(query, [])[:k]

    def has_source(self, path: str) -> bool:  # pragma: no cover - unused
        return False

    def count(self) -> int:
        return sum(len(v) for v in self._results_by_query.values())


@pytest.fixture
def fake_complete(monkeypatch):
    """Records every prompt sent; returns whatever `fake_complete.state['reply']`
    holds as the model's raw text (mutate it per-test)."""
    calls: list[str] = []
    state = {"reply": "[]"}

    def _fake(system, messages, *, model=None, tools=None, max_tokens=4096):
        calls.append(messages[0]["content"])
        return LLMResponse(text=state["reply"])

    monkeypatch.setattr(transform_module, "complete", _fake)
    _fake.calls = calls
    _fake.state = state
    return _fake


class TestExpand:
    def test_returns_variants(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["q", "q alt one", "q alt two"])
        assert expand("q") == ["q", "q alt one", "q alt two"]
        assert fake_complete.calls == ["q"]

    def test_falls_back_on_non_list_reply(self, fake_complete):
        fake_complete.state["reply"] = json.dumps({"not": "a list"})
        assert expand("q") == ["q"]

    def test_falls_back_on_empty_list_reply(self, fake_complete):
        fake_complete.state["reply"] = "[]"
        assert expand("q") == ["q"]

    def test_falls_back_on_unparseable_reply(self, fake_complete):
        fake_complete.state["reply"] = "not json at all"
        assert expand("q") == ["q"]


class TestHyde:
    def test_returns_hypothetical_document(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(
            {"hypothetical_document": "a hypothetical answer"}
        )
        assert hyde("q") == "a hypothetical answer"

    def test_falls_back_on_non_dict_reply(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["not", "a", "dict"])
        assert hyde("q") == "q"

    def test_falls_back_on_empty_document(self, fake_complete):
        fake_complete.state["reply"] = json.dumps({"hypothetical_document": "  "})
        assert hyde("q") == "q"

    def test_falls_back_on_unparseable_reply(self, fake_complete):
        fake_complete.state["reply"] = "not json"
        assert hyde("q") == "q"


class TestDecompose:
    def test_returns_sub_questions(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["sub one", "sub two"])
        assert decompose("q1 and q2") == ["sub one", "sub two"]

    def test_returns_original_when_already_atomic(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["q"])
        assert decompose("q") == ["q"]

    def test_falls_back_on_bad_reply(self, fake_complete):
        fake_complete.state["reply"] = "[]"
        assert decompose("q") == ["q"]


class TestMultiQuery:
    def test_returns_n_variants(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["a", "b", "c"])
        assert multi_query("q", n=3) == ["a", "b", "c"]

    def test_falls_back_on_bad_reply(self, fake_complete):
        fake_complete.state["reply"] = "nonsense"
        assert multi_query("q") == ["q"]


class TestRetrieveWithTransform:
    def test_none_makes_zero_llm_calls_and_equals_plain_search(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("complete() must not be called for strategy='none'")

        monkeypatch.setattr(transform_module, "complete", _boom)
        results_a = [_result("a", 1.0), _result("b", 0.5)]
        store = _FakeStore({"q": results_a})
        got = retrieve_with_transform(store, "q", k=5, strategy="none")
        assert got == results_a
        assert store.search_calls == [{"query": "q", "k": 5}]

    def test_none_is_the_default_strategy(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("complete() must not be called by default")

        monkeypatch.setattr(transform_module, "complete", _boom)
        monkeypatch.delenv("HTC_QUERY_TRANSFORM", raising=False)
        store = _FakeStore({"q": [_result("a")]})
        got = retrieve_with_transform(store, "q", k=5)
        assert [r.chunk.id for r in got] == ["a"]

    def test_hyde_retrieves_using_hypothetical_text(self, fake_complete):
        fake_complete.state["reply"] = json.dumps({"hypothetical_document": "hypothetical doc"})
        store = _FakeStore({"hypothetical doc": [_result("a")]})
        got = retrieve_with_transform(store, "q", k=5, strategy="hyde")
        assert [r.chunk.id for r in got] == ["a"]
        assert store.search_calls == [{"query": "hypothetical doc", "k": 5}]

    def test_expand_retrieves_for_each_variant_and_fuses(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["q", "q alt"])
        store = _FakeStore(
            {
                "q": [_result("a"), _result("b")],
                "q alt": [_result("b"), _result("c")],
            }
        )
        got = retrieve_with_transform(store, "q", k=5, strategy="expand")
        ids = [r.chunk.id for r in got]
        # "b" is hit by both variants (rank 2 in "q", rank 1 in "q alt") so its
        # fused RRF score beats "a" and "c" (each hit once).
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}
        queried = {call["query"] for call in store.search_calls}
        assert queried == {"q", "q alt"}

    def test_multi_dedupes_and_truncates_to_k(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["q1", "q2", "q3"])
        store = _FakeStore(
            {
                "q1": [_result("a"), _result("b")],
                "q2": [_result("b"), _result("c")],
                "q3": [_result("c"), _result("d")],
            }
        )
        got = retrieve_with_transform(store, "q", k=2, strategy="multi")
        assert len(got) == 2

    def test_invalid_strategy_raises(self):
        store = _FakeStore()
        with pytest.raises(ValueError):
            retrieve_with_transform(store, "q", strategy="bogus")

    def test_determinism(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["q", "q alt"])
        store = _FakeStore(
            {
                "q": [_result("a"), _result("b")],
                "q alt": [_result("b"), _result("c")],
            }
        )
        first = retrieve_with_transform(store, "q", k=5, strategy="expand")
        second = retrieve_with_transform(store, "q", k=5, strategy="expand")
        assert [r.chunk.id for r in first] == [r.chunk.id for r in second]

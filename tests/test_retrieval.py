"""RetrievalPipeline: a single composed object wrapping ingest (once) →
memory store → optional knowledge graph (built from the SAME ingested
chunks) → query-transform → hybrid search → rerank once after fusion.

No network (`ingest_sources`/`complete` are monkeypatched where needed).
Not wired into any consumer yet — these tests only exercise the pipeline
object itself.
"""

from __future__ import annotations

import json

from htc.llm import LLMResponse
from htc.world_model import retrieval as retrieval_module
from htc.world_model.ingest.corpus import Corpus
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.local import LocalMemoryStore
from htc.world_model.memory.store import SearchResult
from htc.world_model.query import transform as transform_module
from htc.world_model.retrieval import RetrievalPipeline, build_pipeline


def _chunk(id_: str, text: str = "hello world", path: str | None = None) -> SourceChunk:
    return SourceChunk(
        id=id_,
        source_path=path or f"{id_}.md",
        kind="docs",
        text=text,
        start_char=0,
        end_char=len(text),
    )


def _fake_corpus() -> Corpus:
    chunks = [
        _chunk("a", "Shipping takes 5 to 7 business days for most orders."),
        _chunk("b", "Refund policy covers unopened items within 30 days."),
    ]
    return Corpus(chunks_by_path={c.source_path: [c] for c in chunks})


class TestBuildPipelineIngestsOnce:
    def test_ingest_sources_called_exactly_once_with_graph(self, tmp_path, monkeypatch):
        calls = []

        def _fake_ingest(sources, root):
            calls.append((sources, root))
            return _fake_corpus()

        monkeypatch.setattr(retrieval_module, "ingest_sources", _fake_ingest)
        pipeline = build_pipeline(tmp_path, graph=True)

        assert len(calls) == 1
        assert pipeline.graph is not None
        assert pipeline.store.count() == 2

    def test_ingest_sources_called_exactly_once_without_graph(self, tmp_path, monkeypatch):
        calls = []

        def _fake_ingest(sources, root):
            calls.append((sources, root))
            return _fake_corpus()

        monkeypatch.setattr(retrieval_module, "ingest_sources", _fake_ingest)
        pipeline = build_pipeline(tmp_path, graph=False)

        assert len(calls) == 1
        assert pipeline.graph is None


class TestRetrieveParity:
    def test_none_transform_no_reranker_matches_plain_search(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.add_chunks(
            [
                _chunk("a", "Shipping takes 5 to 7 business days for most orders."),
                _chunk("b", "Refund policy covers unopened items within 30 days."),
            ]
        )
        pipeline = RetrievalPipeline(store=store, reranker=None, query_transform="none")

        got = pipeline.retrieve("shipping refund", k=2)
        expected = store.search("shipping refund", k=2)
        assert got == expected


class _CountingReranker:
    """Records every `rerank` call and how many results it saw each time."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        self.calls.append([r.chunk.id for r in results])
        return results[:top_k]


class _FakeStore:
    """Returns a fixed ranking per query, recording every `search` call."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]]):
        self._results_by_query = results_by_query
        self.search_calls: list[dict] = []

    def add_chunks(self, chunks):  # pragma: no cover - unused
        raise NotImplementedError

    def search(self, query: str, k: int = 5, **kwargs) -> list[SearchResult]:
        self.search_calls.append({"query": query, "k": k, **kwargs})
        return self._results_by_query.get(query, [])[:k]

    def has_source(self, path: str) -> bool:  # pragma: no cover - unused
        return False

    def count(self) -> int:
        return sum(len(v) for v in self._results_by_query.values())


def _result(id_: str, score: float = 0.0) -> SearchResult:
    return SearchResult(chunk=_chunk(id_), score=score)


class TestRerankOncePerFusion:
    def test_multi_transform_reranks_fused_pool_exactly_once(self, monkeypatch):
        monkeypatch.setattr(
            transform_module,
            "complete",
            lambda *a, **k: LLMResponse(text=json.dumps(["q1", "q2"])),
        )
        store = _FakeStore(
            {
                "q1": [_result("a"), _result("b")],
                "q2": [_result("b"), _result("c")],
            }
        )
        reranker = _CountingReranker()
        pipeline = RetrievalPipeline(store=store, reranker=reranker, query_transform="multi")

        got = pipeline.retrieve("q", k=2)

        assert len(reranker.calls) == 1  # applied once, after fusion
        assert set(reranker.calls[0]) == {"a", "b", "c"}  # sees the whole fused pool
        assert len(got) == 2
        # per-variant search calls never carried a reranker of their own
        assert all("reranker" not in call for call in store.search_calls)


class TestBuildPipelineContextual:
    def test_contextual_true_contextualizes_chunks(self, tmp_path, monkeypatch):
        import htc.world_model.ingest.contextual as ctx

        (tmp_path / "a.md").write_text("hello world " * 20)
        monkeypatch.setattr(
            ctx, "complete", lambda *a, **k: LLMResponse(text="Situated context blurb.")
        )

        pipeline = build_pipeline(tmp_path, contextual=True)

        stored = pipeline.store._chunks_by_id  # noqa: SLF001 - test-only introspection
        assert stored
        assert all(
            chunk.embed_text == "Situated context blurb.\n\n" + chunk.text
            for chunk in stored.values()
        )

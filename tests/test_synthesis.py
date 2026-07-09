"""Synthesis: `answer_question` retrieves across memory (+ optional graph)
and asks the model ONE question to produce a grounded, cited `Answer` with
an explicit gap analysis — no network (complete() is monkeypatched)."""

from __future__ import annotations

import json

import pytest

from htc.llm import LLMResponse
from htc.world_model.graph.graph import KnowledgeGraph
from htc.world_model.graph.model import Entity
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.store import SearchResult
from htc.world_model.synthesis import Answer, answer_question


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class _FakeMemory:
    """Returns a fixed set of `SearchResult`s (or none), recording every call."""

    def __init__(self, results: list[SearchResult]):
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5, graph=None) -> list[SearchResult]:
        self.calls.append((query, k))
        return self._results[:k]

    def add_chunks(self, chunks):  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def has_source(self, path: str) -> bool:  # pragma: no cover - unused
        return False

    def count(self) -> int:
        return len(self._results)


_GROUNDED_RESULTS = [
    SearchResult(chunk=_chunk("a", "docs/billing.md", "Billing is monthly."), score=1.0),
    SearchResult(chunk=_chunk("b", "src/billing.py", "def charge(): ..."), score=0.8),
]


@pytest.fixture
def fake_complete(monkeypatch):
    """Records the prompt it was called with; returns whatever JSON payload
    the test assigns to `fake_complete.reply` (default: a well-formed reply)."""
    calls: list[str] = []
    state = {
        "reply": json.dumps(
            {
                "answer_md": "Billing runs monthly (see docs/billing.md).",
                "citations": ["docs/billing.md", "src/billing.py"],
                "gaps": ["Refund policy is not covered."],
                "confidence": "medium",
            }
        )
    }

    def _complete(system, messages, model=None):
        calls.append(messages[0]["content"])
        return LLMResponse(text=state["reply"])

    monkeypatch.setattr("htc.world_model.synthesis.answer.complete", _complete)
    _complete.calls = calls
    _complete.state = state
    return _complete


class TestAnswerQuestion:
    def test_returns_populated_answer_from_retrieved_chunks(self, fake_complete):
        memory = _FakeMemory(_GROUNDED_RESULTS)

        result = answer_question("How does billing work?", memory)

        assert isinstance(result, Answer)
        assert result.question == "How does billing work?"
        assert "monthly" in result.answer_md
        assert result.citations == ["docs/billing.md", "src/billing.py"]
        assert result.gaps == ["Refund policy is not covered."]
        assert result.confidence == "medium"
        assert memory.calls  # retrieval actually happened
        assert "docs/billing.md" in fake_complete.calls[0]

    def test_empty_retrieval_returns_low_confidence_dont_know(self, fake_complete):
        memory = _FakeMemory([])

        result = answer_question("What is our refund policy?", memory)

        assert result.confidence == "low"
        assert result.citations == []
        assert result.gaps
        assert "no relevant" in result.answer_md.lower() or "no" in result.answer_md.lower()
        assert not fake_complete.calls  # no LLM call for empty retrieval

    def test_garbled_non_json_reply_degrades_to_low_confidence_fallback(self, fake_complete):
        fake_complete.state["reply"] = "not json at all, just prose"
        memory = _FakeMemory(_GROUNDED_RESULTS)

        result = answer_question("How does billing work?", memory)

        assert result.confidence == "low"
        assert result.citations == []
        assert result.gaps

    def test_non_dict_json_reply_degrades_to_low_confidence_fallback(self, fake_complete):
        fake_complete.state["reply"] = json.dumps(["billing", "is", "monthly"])
        memory = _FakeMemory(_GROUNDED_RESULTS)

        result = answer_question("How does billing work?", memory)

        assert result.confidence == "low"
        assert result.citations == []
        assert result.gaps

    def test_graph_context_reaches_the_prompt(self, fake_complete):
        memory = _FakeMemory(_GROUNDED_RESULTS)
        graph = KnowledgeGraph()
        graph.add_entities(
            [
                Entity(id="file:src-billing-py", name="src/billing.py", kind="file", mentions=3),
                Entity(id="term:billing", name="billing", kind="term", mentions=5),
            ]
        )

        answer_question("How does billing work?", memory, graph=graph)

        prompt = fake_complete.calls[0]
        assert "KNOWLEDGE GRAPH" in prompt
        assert "src/billing.py" in prompt

"""Agentic/iterative retrieval: `answer_question(..., iterative=True)` loops
retrieve -> assess -> (maybe) followup-retrieve -> merge before the final
synthesis call. Opt-in and bounded — no network (complete() is monkeypatched,
retrieval is a fake pipeline)."""

from __future__ import annotations

import json

import pytest

from htc.llm import LLMResponse
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.store import SearchResult
from htc.world_model.synthesis import answer_question


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class _FakePipeline:
    """Returns results per-query from a dict keyed by the exact query string
    (falling back to an empty list), recording every `.retrieve` call."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]]):
        self._results_by_query = results_by_query
        self.graph = None
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        self.calls.append((query, k))
        return self._results_by_query.get(query, [])[:k]


_INITIAL_RESULTS = [
    SearchResult(chunk=_chunk("a", "docs/billing.md", "Billing is monthly."), score=1.0),
]
_FOLLOWUP_RESULTS = [
    SearchResult(chunk=_chunk("b", "docs/refunds.md", "Refunds within 30 days."), score=0.9),
    # "a" reappears in the followup results — must be deduped, not duplicated.
    SearchResult(chunk=_chunk("a", "docs/billing.md", "Billing is monthly."), score=0.5),
]

_SYNTH_REPLY = json.dumps(
    {
        "answer_md": "Billing runs monthly (see docs/billing.md).",
        "citations": ["docs/billing.md"],
        "gaps": [],
        "confidence": "high",
    }
)


def _assess_reply(sufficient: bool, followup_query: str | None = None) -> str:
    return json.dumps({"sufficient": sufficient, "followup_query": followup_query})


@pytest.fixture
def fake_complete(monkeypatch):
    """Records every prompt; returns replies from a queue the test populates
    via `.replies`, falling back to a well-formed synthesis reply."""
    calls: list[str] = []
    state = {"replies": []}

    def _complete(system, messages, model=None):
        calls.append(messages[0]["content"])
        if state["replies"]:
            return LLMResponse(text=state["replies"].pop(0))
        return LLMResponse(text=_SYNTH_REPLY)

    monkeypatch.setattr("htc.world_model.synthesis.answer.complete", _complete)
    _complete.calls = calls
    _complete.state = state
    return _complete


class TestIterativeOff:
    def test_default_iterative_false_makes_no_assess_calls_and_matches_single_shot(
        self, fake_complete
    ):
        pipeline = _FakePipeline({"How does billing work?": _INITIAL_RESULTS})

        single_shot = answer_question("How does billing work?", pipeline)
        # Reset the fake so we can call again for the "iterative=False, explicit" case.
        pipeline2 = _FakePipeline({"How does billing work?": _INITIAL_RESULTS})
        explicit_off = answer_question("How does billing work?", pipeline2, iterative=False)

        assert single_shot == explicit_off
        assert len(fake_complete.calls) == 2  # one synthesis call per invocation
        assert pipeline.calls == [("How does billing work?", 8)]


class TestIterativeLoop:
    def test_stops_when_assessor_says_sufficient(self, fake_complete):
        pipeline = _FakePipeline({"How does billing work?": _INITIAL_RESULTS})
        fake_complete.state["replies"] = [_assess_reply(True, None)]

        result = answer_question("How does billing work?", pipeline, iterative=True)

        assert result.confidence == "high"
        # 1 retrieval call, 1 assess call, 1 synthesis call.
        assert pipeline.calls == [("How does billing work?", 8)]
        assert len(fake_complete.calls) == 2

    def test_followup_retrieval_reaches_final_pool_deduped(self, fake_complete):
        pipeline = _FakePipeline(
            {
                "How does billing work?": _INITIAL_RESULTS,
                "refund policy": _FOLLOWUP_RESULTS,
            }
        )
        fake_complete.state["replies"] = [
            _assess_reply(False, "refund policy"),
            _assess_reply(True, None),
        ]

        answer_question("How does billing work?", pipeline, iterative=True, max_rounds=3)

        assert pipeline.calls == [
            ("How does billing work?", 8),
            ("refund policy", 8),
        ]
        # Final synthesis prompt is the last complete() call; must contain both
        # sources, with "a" appearing exactly once (deduped).
        synthesis_prompt = fake_complete.calls[-1]
        assert "docs/billing.md" in synthesis_prompt
        assert "docs/refunds.md" in synthesis_prompt
        assert synthesis_prompt.count("Billing is monthly.") == 1

    def test_stops_at_max_rounds_even_if_always_insufficient(self, fake_complete):
        pipeline = _FakePipeline(
            {
                "How does billing work?": _INITIAL_RESULTS,
                "refund policy": _FOLLOWUP_RESULTS,
            }
        )
        # Assessor always insufficient with the same followup query, forever.
        fake_complete.state["replies"] = [
            _assess_reply(False, "refund policy"),
            _assess_reply(False, "refund policy"),
            _assess_reply(False, "refund policy"),
            _assess_reply(False, "refund policy"),
        ]

        answer_question("How does billing work?", pipeline, iterative=True, max_rounds=3)

        # max_rounds=3 -> retrieval rounds: initial + 2 followups = 3 total.
        assert pipeline.calls == [
            ("How does billing work?", 8),
            ("refund policy", 8),
            ("refund policy", 8),
        ]
        # 2 assess calls (bounded by rounds_used < max_rounds) + 1 synthesis call.
        assert len(fake_complete.calls) == 3

    def test_bad_assessor_reply_treated_as_sufficient_stops_loop(self, fake_complete):
        pipeline = _FakePipeline({"How does billing work?": _INITIAL_RESULTS})
        fake_complete.state["replies"] = ["not json at all, just prose"]

        result = answer_question("How does billing work?", pipeline, iterative=True)

        assert result.confidence == "high"
        assert pipeline.calls == [("How does billing work?", 8)]
        assert len(fake_complete.calls) == 2  # 1 assess (bad) + 1 synthesis

    def test_empty_retrieval_skips_iterative_loop_entirely(self, fake_complete):
        pipeline = _FakePipeline({})

        result = answer_question("What is our refund policy?", pipeline, iterative=True)

        assert result.confidence == "low"
        assert not fake_complete.calls  # no assess, no synthesis call

"""Handbook: section generation grounded in memory search, never-overwrite,
no-grounding fallback — no network."""

from __future__ import annotations

import pytest

from htc.handbook.generator import DRAFT_NAME, NO_GROUNDING, SECTIONS, generate_handbook
from htc.llm import LLMResponse
from htc.world_model.memory.local import LocalMemoryStore
from htc.world_model.memory.store import SearchResult
from htc.world_model.retrieval import RetrievalPipeline
from htc.world_model.ingest.model import SourceChunk


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class _FakePipeline:
    """Records every `.retrieve` call and always returns one grounded chunk."""

    def __init__(self):
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        return [SearchResult(chunk=_chunk("a", "docs/notes.md", "some grounding text"), score=1.0)]


@pytest.fixture
def fake_complete(monkeypatch):
    calls: list[str] = []

    def _complete(system, messages, model=None):
        calls.append(system)
        return LLMResponse(text=f"body for section {len(calls)}")

    monkeypatch.setattr("htc.handbook.generator.complete", _complete)
    return calls


class TestGenerateHandbook:
    def test_produces_all_sections_and_calls_search(self, tmp_path, fake_complete):
        pipeline = _FakePipeline()
        markdown = generate_handbook(tmp_path, pipeline, model="test-model")

        for section in SECTIONS:
            assert f"## {section.heading}" in markdown
        assert pipeline.queries == [s.query for s in SECTIONS]
        assert len(fake_complete) == len(SECTIONS)

    def test_writes_draft_without_overwriting_existing_handbook(self, tmp_path, fake_complete):
        existing = tmp_path / "HANDBOOK.md"
        existing.write_text("# hand-written, do not touch\n")

        pipeline = _FakePipeline()
        markdown = generate_handbook(tmp_path, pipeline)

        draft = tmp_path / DRAFT_NAME
        assert draft.exists()
        assert draft.read_text() == markdown
        assert existing.read_text() == "# hand-written, do not touch\n"

    def test_returns_markdown_matching_written_draft(self, tmp_path, fake_complete):
        pipeline = _FakePipeline()
        markdown = generate_handbook(tmp_path, pipeline)
        draft = tmp_path / DRAFT_NAME
        assert draft.read_text() == markdown

    def test_no_grounding_section_skips_the_model(self, tmp_path, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            "htc.handbook.generator.complete",
            lambda *a, **k: calls.append(1) or LLMResponse(text="unused"),
        )
        store = LocalMemoryStore(tmp_path)  # empty store -> every search returns nothing
        pipeline = RetrievalPipeline(store=store)
        markdown = generate_handbook(tmp_path, pipeline)

        assert calls == []
        assert markdown.count(NO_GROUNDING) == len(SECTIONS)

    def test_empty_model_reply_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "htc.handbook.generator.complete", lambda *a, **k: LLMResponse(text="   ")
        )
        pipeline = _FakePipeline()
        with pytest.raises(RuntimeError, match="empty"):
            generate_handbook(tmp_path, pipeline)

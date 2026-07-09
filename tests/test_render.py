"""Render/studio: diagram + podcast generation grounded in memory search,
Mermaid validation with one retry, and the env-gated TTS stub — no network."""

from __future__ import annotations

import pytest

from htc.llm import LLMResponse
from htc.world_model.render.diagram import NO_DIAGRAM, generate_diagram
from htc.world_model.render.podcast import generate_podcast_script, render_audio
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.store import SearchResult


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class _FakeMemory:
    """Records every `search` call and always returns one grounded chunk."""

    def __init__(self):
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        return [
            SearchResult(
                chunk=_chunk("a", "src/app.py", "the app has a web server and a db"), score=1.0
            )
        ]

    def add_chunks(self, chunks):  # pragma: no cover - unused
        raise NotImplementedError

    def has_source(self, path: str) -> bool:  # pragma: no cover - unused
        return False

    def count(self) -> int:  # pragma: no cover - unused
        return 1


VALID_MERMAID = "```mermaid\nflowchart TD\n  A[Web Server] --> B[(DB)]\n```"


class TestGenerateDiagram:
    def test_returns_mermaid_fence_and_writes_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "htc.world_model.render.diagram.complete",
            lambda *a, **k: LLMResponse(text=VALID_MERMAID),
        )
        memory = _FakeMemory()
        markdown = generate_diagram(tmp_path, memory=memory, model="test-model")

        assert "```mermaid" in markdown
        assert "-->" in markdown
        out = tmp_path / ".htc" / "studio" / "architecture.mmd.md"
        assert out.is_file()
        assert out.read_text() == markdown
        assert memory.queries == [
            "architecture modules data flow components how system is put together"
        ]

    def test_retries_once_then_gives_up_on_junk(self, tmp_path, monkeypatch):
        calls: list[int] = []

        def _complete(system, messages, model=None):
            calls.append(1)
            return LLMResponse(text="not a diagram at all")

        monkeypatch.setattr("htc.world_model.render.diagram.complete", _complete)
        memory = _FakeMemory()
        markdown = generate_diagram(tmp_path, memory=memory)

        assert len(calls) == 2
        assert NO_DIAGRAM in markdown
        out = tmp_path / ".htc" / "studio" / "architecture.mmd.md"
        assert out.read_text() == markdown

    def test_succeeds_on_second_attempt(self, tmp_path, monkeypatch):
        replies = iter([LLMResponse(text="junk"), LLMResponse(text=VALID_MERMAID)])
        monkeypatch.setattr(
            "htc.world_model.render.diagram.complete", lambda *a, **k: next(replies)
        )
        memory = _FakeMemory()
        markdown = generate_diagram(tmp_path, memory=memory)

        assert "```mermaid" in markdown

    def test_mindmap_kind_uses_its_own_query(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "htc.world_model.render.diagram.complete",
            lambda *a, **k: LLMResponse(
                text="```mermaid\nmindmap\n  root((App))\n    Web\n    DB\n```"
            ),
        )
        memory = _FakeMemory()
        markdown = generate_diagram(tmp_path, memory=memory, kind="mindmap")

        assert "```mermaid" in markdown
        assert memory.queries == ["project overview purpose main components concepts"]

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown diagram kind"):
            generate_diagram(tmp_path, memory=_FakeMemory(), kind="bogus")


class TestGeneratePodcastScript:
    def test_returns_speaker_labeled_script_and_writes_file(self, tmp_path, monkeypatch):
        script_text = "Host A: Welcome!\nHost B: Let's dig in.\n"
        monkeypatch.setattr(
            "htc.world_model.render.podcast.complete",
            lambda *a, **k: LLMResponse(text=script_text),
        )
        memory = _FakeMemory()
        script = generate_podcast_script(tmp_path, memory=memory, model="test-model")

        assert "Host A:" in script
        assert "Host B:" in script
        out = tmp_path / ".htc" / "studio" / "overview-script.md"
        assert out.is_file()
        assert out.read_text() == script + "\n"
        assert memory.queries == [
            "project overview purpose architecture what this system does for a new team member"
        ]

    def test_empty_model_reply_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "htc.world_model.render.podcast.complete", lambda *a, **k: LLMResponse(text="   ")
        )
        with pytest.raises(RuntimeError, match="empty"):
            generate_podcast_script(tmp_path, memory=_FakeMemory())


class TestRenderAudio:
    def test_returns_none_without_importing_tts_lib_when_unconfigured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HTC_TTS_PROVIDER", raising=False)
        monkeypatch.delenv("HTC_TTS_API_KEY", raising=False)

        result = render_audio("Host A: hi\nHost B: hey", tmp_path / "overview.mp3")

        assert result is None

    def test_returns_none_when_provider_set_but_no_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HTC_TTS_PROVIDER", "elevenlabs")
        monkeypatch.delenv("HTC_TTS_API_KEY", raising=False)

        result = render_audio("Host A: hi\nHost B: hey", tmp_path / "overview.mp3")

        assert result is None

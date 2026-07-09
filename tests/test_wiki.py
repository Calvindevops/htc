"""Wiki: LLM-synthesized pages grounded in memory search, stored back into
memory as `kind="wiki"` chunks, written to disk for human browsing, and the
"unknown" fallback when a topic has no grounding — no network."""

from __future__ import annotations

import pytest

from htc.llm import LLMResponse
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.local import LocalMemoryStore
from htc.world_model.memory.store import SearchResult
from htc.world_model.wiki import UNKNOWN, WikiPage, add_wiki_to_memory, build_wiki, write_wiki_files
from htc.world_model.wiki.generator import WIKI_DIR


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class _FakeMemory:
    """Records every `search` call; returns one grounded chunk unless the
    query is in `empty_for`."""

    def __init__(self, empty_for: set[str] | None = None):
        self.queries: list[str] = []
        self._empty_for = empty_for or set()

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        if query in self._empty_for:
            return []
        return [SearchResult(chunk=_chunk("a", "docs/notes.md", "some grounding text"), score=1.0)]

    def add_chunks(self, chunks):  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def has_source(self, path: str) -> bool:  # pragma: no cover - unused
        return False

    def count(self) -> int:  # pragma: no cover - unused
        return 1


@pytest.fixture
def fake_complete(monkeypatch):
    calls: list[str] = []

    def _complete(system, messages, model=None):
        calls.append(messages[0]["content"])
        return LLMResponse(text=f"synthesized page body ({len(calls)})")

    monkeypatch.setattr("htc.world_model.wiki.generator.complete", _complete)
    return calls


class TestBuildWiki:
    def test_produces_a_page_per_explicit_topic_grounded_in_retrieved_chunks(self, fake_complete):
        memory = _FakeMemory()
        pages = build_wiki(memory, topics=["Auth", "Billing"], model="test-model")

        assert [p.title for p in pages] == ["Auth", "Billing"]
        assert memory.queries == ["Auth", "Billing"]
        for page in pages:
            assert page.source_paths == ["docs/notes.md"]
            assert "synthesized page body" in page.body_md
        assert len(fake_complete) == 2

    def test_infers_topics_from_memory_when_none_given(self, monkeypatch):
        memory = _FakeMemory()
        monkeypatch.setattr(
            "htc.world_model.wiki.generator.complete",
            lambda system, messages, model=None: LLMResponse(text='["Auth", "Billing"]'),
        )

        pages = build_wiki(memory)

        # First call is the topic-inference search; the rest ground each page.
        assert memory.queries[0] == "overview architecture components purpose"
        assert [p.title for p in pages] == ["Auth", "Billing"]

    def test_topic_with_no_grounding_returns_unknown_page(self):
        memory = _FakeMemory(empty_for={"Nonexistent"})
        pages = build_wiki(memory, topics=["Nonexistent"])

        assert pages == [WikiPage(title="Nonexistent", body_md=UNKNOWN, source_paths=[])]

    def test_model_reply_of_unknown_yields_unknown_page(self, monkeypatch):
        memory = _FakeMemory()
        monkeypatch.setattr(
            "htc.world_model.wiki.generator.complete",
            lambda system, messages, model=None: LLMResponse(text="unknown"),
        )

        pages = build_wiki(memory, topics=["Auth"])

        assert pages == [WikiPage(title="Auth", body_md=UNKNOWN, source_paths=[])]

    def test_empty_model_reply_raises(self, monkeypatch):
        memory = _FakeMemory()
        monkeypatch.setattr(
            "htc.world_model.wiki.generator.complete",
            lambda system, messages, model=None: LLMResponse(text="   "),
        )
        with pytest.raises(RuntimeError, match="empty"):
            build_wiki(memory, topics=["Auth"])

    def test_no_topics_inferred_and_none_given_yields_no_pages(self, monkeypatch):
        memory = _FakeMemory(empty_for={"overview architecture components purpose"})
        pages = build_wiki(memory)
        assert pages == []


class TestAddWikiToMemory:
    def test_inserted_pages_show_up_in_memory_search(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.add_chunks([_chunk("src-1", "src/auth.py", "unrelated raw source content")])

        page = WikiPage(
            title="Authentication",
            body_md="This project authenticates users via JWT (see src/auth.py).",
            source_paths=["src/auth.py"],
        )
        add_wiki_to_memory([page], store)

        results = store.search("authenticates users JWT", k=5)
        wiki_results = [r for r in results if r.chunk.kind == "wiki"]
        assert wiki_results
        assert wiki_results[0].chunk.source_path == f"{WIKI_DIR}/authentication.md"
        assert "Authentication" in wiki_results[0].chunk.text

    def test_no_pages_is_a_noop(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        add_wiki_to_memory([], store)
        assert store.count() == 0


class TestWriteWikiFiles:
    def test_writes_one_markdown_file_per_page_with_sources(self, tmp_path):
        pages = [
            WikiPage(title="Auth", body_md="Grounded body.", source_paths=["src/auth.py"]),
            WikiPage(title="Nonexistent", body_md=UNKNOWN, source_paths=[]),
        ]
        written = write_wiki_files(pages, tmp_path)

        assert len(written) == 2
        auth_path = tmp_path / WIKI_DIR / "auth.md"
        assert auth_path in written
        content = auth_path.read_text()
        assert "# Auth" in content
        assert "Grounded body." in content
        assert "src/auth.py" in content

        unknown_path = tmp_path / WIKI_DIR / "nonexistent.md"
        assert unknown_path.read_text().count("ungrounded") == 1

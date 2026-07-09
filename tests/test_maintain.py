"""Self-maintenance: code-file ingestion, staleness detection, incremental
refresh, and dedup — all deterministic, no network."""

from __future__ import annotations

from htc.adapters.base import Source
from htc.world_model.ingest.extractors import extract_text
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.maintain.dedup import dedup_chunks
from htc.world_model.maintain.refresh import refresh_memory
from htc.world_model.maintain.staleness import check_staleness
from htc.world_model.maintain.state import load_manifest
from htc.world_model.memory.local import LocalMemoryStore
from htc.world_model.memory import local as local_module


def _chunk(id_: str, source_path: str, text: str) -> SourceChunk:
    return SourceChunk(
        id=id_, source_path=source_path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class TestCodeExtraction:
    def test_python_file_extracted(self, tmp_path):
        path = tmp_path / "main.py"
        path.write_text("def foo():\n    return 1\n")
        assert extract_text(path) == "def foo():\n    return 1\n"

    def test_typescript_file_extracted(self, tmp_path):
        path = tmp_path / "index.ts"
        path.write_text("export const x = 1;\n")
        assert extract_text(path) == "export const x = 1;\n"

    def test_env_file_not_ingested(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / ".env").write_text("SECRET=abc\n")
        from htc.world_model.ingest import ingest_sources

        corpus = ingest_sources([Source(path=str(tmp_path), kind="repo")], root=tmp_path)
        assert corpus.has_source("a.py")
        assert not corpus.has_source(".env")


class TestStaleness:
    def test_classifies_new_changed_deleted_fresh(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n")
        (tmp_path / "b.py").write_text("b = 1\n")
        sources = [Source(path=str(tmp_path), kind="repo")]

        # First pass: manifest empty, everything is "new".
        first = check_staleness(sources, tmp_path, manifest={})
        assert first["new"] == ["a.py", "b.py"]
        assert first["changed"] == []
        assert first["deleted"] == []
        assert first["fresh"] == []

        # Build a manifest as if a.py/b.py were already ingested.
        from htc.world_model.maintain.state import content_hash

        manifest = {
            "a.py": {"hash": content_hash(tmp_path / "a.py"), "chunk_ids": []},
            "b.py": {"hash": content_hash(tmp_path / "b.py"), "chunk_ids": []},
        }

        # Nothing changed on disk -> both fresh.
        second = check_staleness(sources, tmp_path, manifest)
        assert second["fresh"] == ["a.py", "b.py"]
        assert second["new"] == []
        assert second["changed"] == []
        assert second["deleted"] == []

        # Modify a.py, delete b.py, add c.py.
        (tmp_path / "a.py").write_text("a = 2\n")
        (tmp_path / "b.py").unlink()
        (tmp_path / "c.py").write_text("c = 1\n")

        third = check_staleness(sources, tmp_path, manifest)
        assert third["changed"] == ["a.py"]
        assert third["deleted"] == ["b.py"]
        assert third["new"] == ["c.py"]
        assert third["fresh"] == []


class TestRefreshMemory:
    def test_adds_new_sources(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n")
        memory = LocalMemoryStore(tmp_path)
        sources = [Source(path=str(tmp_path), kind="repo")]

        summary = refresh_memory(memory, sources, tmp_path)
        assert summary["new"] == 1
        assert summary["chunks_added"] == 1
        assert memory.has_source("a.py")

        manifest = load_manifest(tmp_path)
        assert "a.py" in manifest

    def test_replaces_chunks_for_changed_source(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n")
        memory = LocalMemoryStore(tmp_path)
        sources = [Source(path=str(tmp_path), kind="repo")]
        refresh_memory(memory, sources, tmp_path)

        (tmp_path / "a.py").write_text("a = 999999\n")
        summary = refresh_memory(memory, sources, tmp_path)
        assert summary["changed"] == 1
        assert summary["fresh"] == 0

        remaining_texts = [c.text for c in memory._chunks_by_id.values()]
        assert "a = 1\n" not in remaining_texts
        assert "a = 999999\n" in remaining_texts

    def test_prunes_deleted_source(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n")
        (tmp_path / "b.py").write_text("b = 1\n")
        memory = LocalMemoryStore(tmp_path)
        sources = [Source(path=str(tmp_path), kind="repo")]
        refresh_memory(memory, sources, tmp_path)
        assert memory.has_source("b.py")

        (tmp_path / "b.py").unlink()
        summary = refresh_memory(memory, sources, tmp_path)
        assert summary["deleted"] == 1
        assert not memory.has_source("b.py")

        manifest = load_manifest(tmp_path)
        assert "b.py" not in manifest

    def test_updates_manifest(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n")
        memory = LocalMemoryStore(tmp_path)
        sources = [Source(path=str(tmp_path), kind="repo")]
        refresh_memory(memory, sources, tmp_path)

        manifest = load_manifest(tmp_path)
        assert manifest["a.py"]["chunk_ids"]

    def test_is_incremental_unchanged_sources_not_reembedded(self, tmp_path, monkeypatch):
        (tmp_path / "a.py").write_text("a = 1\n")
        (tmp_path / "b.py").write_text("b = 1\n")
        memory = LocalMemoryStore(tmp_path)
        sources = [Source(path=str(tmp_path), kind="repo")]
        refresh_memory(memory, sources, tmp_path)

        # Now enable the embedder and track exactly which texts get embedded.
        monkeypatch.setattr(local_module, "_embedder_available", lambda: True)
        embedded_calls: list[list[str]] = []

        def _fake_embed(texts: list[str]) -> list[list[float]]:
            embedded_calls.append(list(texts))
            return [[0.0] for _ in texts]

        monkeypatch.setattr(local_module, "_embed", _fake_embed)

        # Only a.py changes; b.py stays fresh and must never be re-embedded.
        (tmp_path / "a.py").write_text("a = 2\n")
        refresh_memory(memory, sources, tmp_path)

        embedded_texts = [text for call in embedded_calls for text in call]
        assert "b = 1\n" not in embedded_texts
        assert "a = 2\n" in embedded_texts


class TestDedup:
    def test_collapses_exact_duplicates(self):
        chunks = [
            _chunk("a", "one.md", "Refunds within 30 days."),
            _chunk("b", "two.md", "Refunds within 30 days."),
        ]
        result = dedup_chunks(chunks)
        assert len(result) == 1
        assert result[0].id == "a"

    def test_collapses_whitespace_and_case_variants(self):
        chunks = [
            _chunk("a", "one.md", "Refunds within 30 days."),
            _chunk("b", "two.md", "refunds   within\n30 days."),
        ]
        result = dedup_chunks(chunks)
        assert len(result) == 1

    def test_collapses_high_overlap_near_duplicates(self):
        # 20 shared tokens with only the last one swapped -> jaccard ~0.90 (>0.9 threshold).
        base = (
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
            "kilo lima mike november oscar papa quebec romeo sierra"
        )
        chunks = [
            _chunk("a", "one.md", base + " tango"),
            _chunk("b", "two.md", base + " uniform"),
        ]
        result = dedup_chunks(chunks)
        assert len(result) == 1
        assert result[0].id == "a"

    def test_keeps_distinct_chunks(self):
        chunks = [
            _chunk("a", "one.md", "Refunds within 30 days."),
            _chunk("b", "two.md", "Shipping takes 5 to 7 business days."),
        ]
        result = dedup_chunks(chunks)
        assert len(result) == 2

    def test_deterministic_across_calls(self):
        chunks = [
            _chunk("a", "one.md", "Refunds within 30 days."),
            _chunk("b", "two.md", "Refunds within 30 days."),
            _chunk("c", "three.md", "Shipping info."),
        ]
        first = dedup_chunks(chunks)
        second = dedup_chunks(chunks)
        assert [c.id for c in first] == [c.id for c in second]

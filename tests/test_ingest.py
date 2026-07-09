"""Ingest: extractor dispatch, chunker, SourceChunk determinism, Corpus,
and the generalized grounding gate accepting an ingested (non-file) path."""

from __future__ import annotations

import sys

import pytest

from htc.adapters.base import Source
from htc.goldens.generator import _artifact_is_grounded
from htc.world_model.ingest import Corpus, ingest_sources
from htc.world_model.ingest.chunker import chunk_text
from htc.world_model.ingest.extractors import (
    MissingDependencyError,
    UnsupportedFormatError,
    extract_text,
)
from htc.world_model.ingest.model import chunk_id


class TestExtractors:
    def test_txt(self, tmp_path):
        path = tmp_path / "note.txt"
        path.write_text("hello world")
        assert extract_text(path) == "hello world"

    def test_md(self, tmp_path):
        path = tmp_path / "note.md"
        path.write_text("# heading\n\nbody text")
        assert extract_text(path) == "# heading\n\nbody text"

    def test_html_strips_tags(self, tmp_path):
        path = tmp_path / "page.html"
        path.write_text("<html><body><p>Hello <b>World</b></p></body></html>")
        assert extract_text(path) == "Hello World"

    def test_html_skips_script_and_style(self, tmp_path):
        path = tmp_path / "page.html"
        path.write_text(
            "<html><head><style>.x{color:red}</style></head>"
            "<body><script>alert(1)</script><p>Visible</p></body></html>"
        )
        assert extract_text(path).strip() == "Visible"

    def test_vtt_strips_timestamps(self, tmp_path):
        path = tmp_path / "clip.vtt"
        path.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nHello there\n")
        assert extract_text(path) == "Hello there"

    def test_srt_strips_timestamps(self, tmp_path):
        path = tmp_path / "clip.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello there\n")
        assert extract_text(path) == "Hello there"

    def test_unknown_extension_raises(self, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"\x00\x01")
        with pytest.raises(UnsupportedFormatError):
            extract_text(path)

    def test_pdf_without_optional_dep_raises_clear_error(self, tmp_path, monkeypatch):
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setitem(sys.modules, "pypdf", None)
        with pytest.raises(MissingDependencyError, match="ingest"):
            extract_text(path)


class TestChunker:
    def test_respects_max_chars(self):
        text = "Sentence one. Sentence two. Sentence three. " * 30
        chunks = chunk_text(text, "doc.txt", "docs", max_chars=100)
        assert chunks
        assert all(len(c.text) <= 100 for c in chunks)

    def test_never_cuts_mid_word(self):
        # One giant "sentence" (no punctuation) forces whitespace hard-wrap.
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, "doc.txt", "docs", max_chars=50)
        words = set(text.split(" "))
        for chunk in chunks:
            for token in chunk.text.split(" "):
                assert token in words

    def test_splits_on_blank_lines(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, "doc.txt", "docs", max_chars=1000)
        assert [c.text for c in chunks] == ["Para one.", "Para two.", "Para three."]

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("   \n\n  ", "doc.txt", "docs") == []

    def test_offsets_recover_original_text(self):
        text = "Para one.\n\nPara two is a bit longer here."
        chunks = chunk_text(text, "doc.txt", "docs", max_chars=1000)
        for chunk in chunks:
            assert text[chunk.start_char : chunk.end_char] == chunk.text


class TestSourceChunkId:
    def test_deterministic_across_calls(self):
        assert chunk_id("docs/a.txt", 42) == chunk_id("docs/a.txt", 42)

    def test_differs_by_path_or_offset(self):
        assert chunk_id("docs/a.txt", 0) != chunk_id("docs/b.txt", 0)
        assert chunk_id("docs/a.txt", 0) != chunk_id("docs/a.txt", 10)

    def test_chunk_text_ids_are_stable_across_reingest(self):
        text = "Para one.\n\nPara two."
        first = chunk_text(text, "doc.txt", "docs")
        second = chunk_text(text, "doc.txt", "docs")
        assert [c.id for c in first] == [c.id for c in second]


class TestCorpus:
    def test_ingest_directory_and_has_source(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "policy.md").write_text("# Policy\n\nAll refunds within 30 days.")
        corpus = ingest_sources([Source(path=str(docs), kind="docs")], root=tmp_path)
        assert corpus.has_source("docs/policy.md")
        assert corpus.chunks_for("docs/policy.md")
        assert not corpus.has_source("docs/ghost.md")

    def test_ingest_single_file_source(self, tmp_path):
        doc = tmp_path / "handbook.txt"
        doc.write_text("Employees get unlimited PTO.")
        corpus = ingest_sources([Source(path=str(doc), kind="docs")], root=tmp_path)
        assert corpus.has_source("handbook.txt")

    def test_unknown_extension_files_are_skipped_not_errored(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "data.bin").write_bytes(b"\x00\x01")
        corpus = ingest_sources([Source(path=str(docs), kind="docs")], root=tmp_path)
        assert corpus.chunks_by_path == {}

    def test_empty_corpus_has_source_is_false(self):
        assert Corpus().has_source("anything") is False


class TestGeneralizedGrounding:
    def test_accepts_real_filesystem_file_without_corpus(self, tmp_path):
        (tmp_path / "main.py").write_text("PORT = 8080")
        assert _artifact_is_grounded("main.py", tmp_path, corpus=None) is True

    def test_rejects_bogus_path_without_corpus(self, tmp_path):
        assert _artifact_is_grounded("ghost.py", tmp_path, corpus=None) is False

    def test_accepts_ingested_non_file_source_path(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "policy.md").write_text("Refund window is 30 days.")
        corpus = ingest_sources([Source(path=str(docs), kind="docs")], root=tmp_path)
        assert _artifact_is_grounded("docs/policy.md", tmp_path, corpus=corpus) is True

    def test_still_rejects_bogus_path_with_corpus_present(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "policy.md").write_text("Refund window is 30 days.")
        corpus = ingest_sources([Source(path=str(docs), kind="docs")], root=tmp_path)
        assert _artifact_is_grounded("docs/ghost.md", tmp_path, corpus=corpus) is False

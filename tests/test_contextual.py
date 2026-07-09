"""Contextual Retrieval: chunk context prepended for embedding only, opt-in."""

from __future__ import annotations

from htc.world_model.build import build_memory
from htc.world_model.ingest.contextual import contextualize_chunks
from htc.world_model.ingest.model import SourceChunk
from htc.world_model.memory.local import LocalMemoryStore


def _chunk(cid: str, text: str, path: str = "doc.md") -> SourceChunk:
    return SourceChunk(
        id=cid, source_path=path, kind="docs", text=text, start_char=0, end_char=len(text)
    )


class TestContextualize:
    def test_prepends_context_into_embed_text_only(self, monkeypatch):
        import htc.world_model.ingest.contextual as ctx

        monkeypatch.setattr(
            ctx,
            "complete",
            lambda *a, **k: type("R", (), {"text": "This is section 2 of the guide."})(),
        )
        chunks = [_chunk("c1", "Run the migration before deploy.")]
        out = contextualize_chunks(chunks, {"doc.md": "full doc text"})
        assert out[0].text == "Run the migration before deploy."  # original untouched
        assert (
            out[0].embed_text
            == "This is section 2 of the guide.\n\nRun the migration before deploy."
        )

    def test_bad_reply_leaves_chunk_unchanged(self, monkeypatch):
        import htc.world_model.ingest.contextual as ctx

        monkeypatch.setattr(ctx, "complete", lambda *a, **k: type("R", (), {"text": "   "})())
        chunks = [_chunk("c1", "body")]
        out = contextualize_chunks(chunks, {"doc.md": "full"})
        assert out[0].embed_text is None

    def test_no_full_text_leaves_chunk_unchanged(self, monkeypatch):
        import htc.world_model.ingest.contextual as ctx

        called = []
        monkeypatch.setattr(
            ctx, "complete", lambda *a, **k: called.append(1) or type("R", (), {"text": "ctx"})()
        )
        out = contextualize_chunks([_chunk("c1", "body", "other.md")], {"doc.md": "full"})
        assert out[0].embed_text is None
        assert called == []  # no LLM call when the source has no full text


class TestEmbedderUsesEmbedText:
    def test_embeds_embed_text_when_present_else_text(self, tmp_path, monkeypatch):
        import htc.world_model.memory.local as local

        captured: list[str] = []

        def fake_embed(texts):
            captured.extend(texts)
            return [[float(len(t))] for t in texts]

        monkeypatch.setattr(local, "_embedder_available", lambda: True)
        monkeypatch.setattr(local, "_embed", fake_embed)

        store = LocalMemoryStore(root=tmp_path)
        plain = _chunk("a", "plain text", "a.md")
        ctxd = SourceChunk(
            id="b",
            source_path="b.md",
            kind="docs",
            text="original",
            start_char=0,
            end_char=8,
            embed_text="CONTEXT\n\noriginal",
        )
        store.add_chunks([plain, ctxd])
        assert "plain text" in captured  # no embed_text → embed text
        assert "CONTEXT\n\noriginal" in captured  # embed_text present → embed that
        assert "original" not in captured


class TestBuildMemoryContextualFlag:
    def test_contextual_false_makes_zero_context_calls(self, tmp_path, monkeypatch):
        import htc.world_model.ingest.contextual as ctx
        from htc.adapters.base import Source

        (tmp_path / "a.md").write_text("hello world " * 20)
        calls = []
        monkeypatch.setattr(
            ctx, "complete", lambda *a, **k: calls.append(1) or type("R", (), {"text": "c"})()
        )
        build_memory([Source(path=str(tmp_path), kind="docs")], tmp_path, contextual=False)
        assert calls == []

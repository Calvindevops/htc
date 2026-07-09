"""CLI: `studio --kind` mapping to the right generator argument — no network,
no real retrieval (the pipeline builder and generator are mocked)."""

from __future__ import annotations

from htc import cli


class TestStudioKindMapping:
    def test_kind_diagram_calls_generate_diagram_with_architecture(self, tmp_path, monkeypatch):
        calls: list[dict] = []

        def fake_build_pipeline(root, sources, *, rerank, model):
            return object()

        def fake_generate_diagram(root, pipeline, *, model=None, kind="architecture"):
            calls.append({"root": root, "kind": kind})
            return "```mermaid\nflowchart TD\n  A --> B\n```"

        monkeypatch.setattr("htc.world_model.retrieval.build_pipeline", fake_build_pipeline)
        monkeypatch.setattr("htc.world_model.render.generate_diagram", fake_generate_diagram)

        rc = cli.main(["studio", "--root", str(tmp_path), "--kind", "diagram"])

        assert rc == 0
        assert len(calls) == 1
        assert calls[0]["kind"] == "architecture"

    def test_kind_mindmap_calls_generate_diagram_with_mindmap(self, tmp_path, monkeypatch):
        calls: list[dict] = []

        def fake_build_pipeline(root, sources, *, rerank, model):
            return object()

        def fake_generate_diagram(root, pipeline, *, model=None, kind="architecture"):
            calls.append({"root": root, "kind": kind})
            return "```mermaid\nmindmap\n  root\n```"

        monkeypatch.setattr("htc.world_model.retrieval.build_pipeline", fake_build_pipeline)
        monkeypatch.setattr("htc.world_model.render.generate_diagram", fake_generate_diagram)

        rc = cli.main(["studio", "--root", str(tmp_path), "--kind", "mindmap"])

        assert rc == 0
        assert len(calls) == 1
        assert calls[0]["kind"] == "mindmap"

"""Test-wide safety net for the local memory embedder: never probe the real
network (Ollama) or touch the developer's real `~/.htc/config.json`, and
never block on the first-boot wizard's interactive prompt. Individual tests
that specifically exercise Ollama/the wizard override these via monkeypatch."""

from __future__ import annotations

import pytest

from htc.world_model.memory import local as local_module


@pytest.fixture(autouse=True)
def _no_real_ollama_or_wizard_io(tmp_path, monkeypatch):
    monkeypatch.setattr(local_module, "_ollama_reachable", lambda: False)
    monkeypatch.setattr(local_module, "_CONFIG_PATH", tmp_path / "htc-test-config.json")
    monkeypatch.setenv("HTC_EMBED_NONINTERACTIVE", "1")

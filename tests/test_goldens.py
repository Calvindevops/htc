"""Goldens: validation gate, save/load round-trip, JSON extraction."""

from __future__ import annotations

import json

import pytest

from htc.adapters.base import Source
from htc.goldens.generator import (
    BUSINESS_GENERATION_SYSTEM,
    GENERATION_SYSTEM,
    Golden,
    _is_test_file,
    _validate,
    generate_goldens,
    load_goldens,
    save_goldens,
)
from htc.llm import LLMError, LLMResponse, extract_json


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("PORT = 8080\n")
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_x(): pass\n")
    return tmp_path


def _item(**overrides):
    base = {
        "question": "What port does the app listen on?",
        "answer": "8080",
        "artifact": "app/main.py",
        "category": "config",
        "difficulty": 1,
    }
    base.update(overrides)
    return base


class TestValidate:
    def test_keeps_grounded_item(self, repo):
        goldens = _validate([_item()], repo)
        assert len(goldens) == 1
        assert goldens[0].artifact == "app/main.py"

    def test_drops_missing_artifact(self, repo):
        assert _validate([_item(artifact="app/ghost.py")], repo) == []

    def test_drops_bad_category(self, repo):
        assert _validate([_item(category="trivia")], repo) == []

    def test_drops_bad_difficulty(self, repo):
        assert _validate([_item(difficulty=9)], repo) == []

    def test_drops_empty_question(self, repo):
        assert _validate([_item(question="  ")], repo) == []

    def test_drops_malformed_item(self, repo):
        assert _validate([{"question": "?"}], repo) == []

    def test_strips_leading_slash_in_artifact(self, repo):
        goldens = _validate([_item(artifact="/app/main.py")], repo)
        assert len(goldens) == 1
        assert goldens[0].artifact == "app/main.py"

    def test_rejects_test_file_artifact(self, repo):
        assert _validate([_item(artifact="tests/test_main.py")], repo) == []


class TestIsTestFile:
    def test_classifies_test_file(self, repo):
        assert _is_test_file(repo / "tests" / "test_main.py", repo) is True

    def test_classifies_source_file(self, repo):
        assert _is_test_file(repo / "app" / "main.py", repo) is False


class TestRoundTrip:
    def test_save_and_load(self, repo, tmp_path):
        goldens = [Golden(**_item())]
        path = tmp_path / "out" / "goldens.json"
        save_goldens(goldens, path)
        assert load_goldens(path) == goldens

    def test_load_rejects_non_array(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a list"}')
        with pytest.raises(ValueError):
            load_goldens(path)


class TestExtractJson:
    def test_plain_array(self):
        assert extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_fenced(self):
        assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_prose_wrapped(self):
        assert extract_json('Here you go:\n[{"a": 1}]\nDone.') == [{"a": 1}]

    def test_object(self):
        assert extract_json('{"verdict": "correct"}') == {"verdict": "correct"}

    def test_garbage_raises(self):
        with pytest.raises(LLMError):
            extract_json("no json here at all")

    def test_single_line_fenced(self):
        assert extract_json('```json {"a": 1}```') == {"a": 1}

    def test_two_arrays_returns_first_balanced_span(self):
        text = 'First: [{"a": 1}]\nSecond: [{"b": 2}]'
        assert extract_json(text) == [{"a": 1}]

    def test_valid_array_after_prose(self):
        text = 'Sure, here is the array you asked for:\n[{"question": "q"}]'
        assert extract_json(text) == [{"question": "q"}]


class TestGenerateGoldensBatchGuard:
    def test_provider_error_skips_batch_without_raising(self, repo, monkeypatch, capsys):
        monkeypatch.setattr(
            "htc.goldens.generator.complete",
            lambda *a, **k: (_ for _ in ()).throw(LLMError("boom")),
        )
        goldens = generate_goldens(repo, count=3, seed=1)
        assert goldens == []
        assert "batch skipped" in capsys.readouterr().err


class TestGenerateGoldensBalance:
    def test_balance_prompts_mention_under_filled_categories(self, repo, monkeypatch):
        prompts: list[str] = []

        def fake_complete(system, messages, model=None):
            prompts.append(messages[0]["content"])
            n = len(prompts)
            item = {
                "question": f"q{n}",
                "answer": "a",
                # Every batch comes back "config" so architecture/behavior/ops
                # stay under-filled and should keep getting hinted.
                "artifact": "app/main.py",
                "category": "config",
                "difficulty": 1,
            }
            return LLMResponse(text=json.dumps([item]))

        monkeypatch.setattr("htc.goldens.generator.complete", fake_complete)
        generate_goldens(repo, count=8, seed=1, balance=True)
        assert len(prompts) > 1
        # The first batch has no counts yet, but a later batch must call out
        # the categories that are still under their fair share.
        assert any("Prioritize these under-covered categories" in p for p in prompts[1:])
        assert any("architecture" in p for p in prompts[1:])


class TestGenerateGoldensOnBatch:
    def test_on_batch_invoked_per_accepted_batch(self, repo, monkeypatch):
        calls: list[tuple[int, int]] = []

        def fake_complete(system, messages, model=None):
            item = {
                "question": f"q{len(calls)}",
                "answer": "a",
                "artifact": "app/main.py",
                "category": "config",
                "difficulty": 1,
            }
            return LLMResponse(text=json.dumps([item]))

        monkeypatch.setattr("htc.goldens.generator.complete", fake_complete)
        generate_goldens(repo, count=3, seed=1, on_batch=lambda i, total: calls.append((i, total)))
        assert calls
        assert calls[-1][1] == 3


class TestGenerateGoldensTopUp:
    def test_tops_up_beyond_the_fixed_batch_count(self, repo, monkeypatch):
        calls = {"n": 0}

        def fake_complete(system, messages, model=None):
            calls["n"] += 1
            item = {
                "question": f"q{calls['n']}",
                "answer": "a",
                "artifact": "app/main.py",
                "category": "config",
                "difficulty": 1,
            }
            return LLMResponse(text=json.dumps([item]))

        monkeypatch.setattr("htc.goldens.generator.complete", fake_complete)
        goldens = generate_goldens(repo, count=10, seed=1)
        assert len(goldens) == 10
        # `batches = (count + 4) // 5` alone would cap at 2 iterations for count=10 —
        # confirm the top-up loop kept attempting past that fixed count.
        assert calls["n"] > 2


class TestGenerateGoldensScope:
    def test_code_scope_default_uses_code_prompt(self, repo, monkeypatch):
        systems: list[str] = []

        def fake_complete(system, messages, model=None):
            systems.append(system)
            item = {
                "question": "q",
                "answer": "a",
                "artifact": "app/main.py",
                "category": "config",
                "difficulty": 1,
            }
            return LLMResponse(text=json.dumps([item]))

        monkeypatch.setattr("htc.goldens.generator.complete", fake_complete)
        goldens = generate_goldens(repo, count=1, seed=1)
        assert goldens
        assert systems[0] == GENERATION_SYSTEM

    def test_business_scope_uses_business_prompt_and_docs_artifact(self, repo, monkeypatch):
        docs = repo / "docs"
        docs.mkdir()
        (docs / "policy.md").write_text("We always ship changelogs with every release.")

        systems: list[str] = []

        def fake_complete(system, messages, model=None):
            systems.append(system)
            item = {
                "question": "Why do we ship changelogs with releases?",
                "answer": "House policy for transparency.",
                "artifact": "docs/policy.md",
                "category": "ops",
                "difficulty": 1,
            }
            return LLMResponse(text=json.dumps([item]))

        monkeypatch.setattr("htc.goldens.generator.complete", fake_complete)
        goldens = generate_goldens(
            repo,
            count=1,
            seed=1,
            scope="business",
            sources=[Source(path=str(docs), kind="docs")],
        )
        assert goldens
        assert goldens[0].artifact == "docs/policy.md"
        assert systems and all(s == BUSINESS_GENERATION_SYSTEM for s in systems)

    def test_business_scope_never_samples_repo_files(self, repo, monkeypatch):
        """With no ingested sources, business scope has nothing to sample and
        stops immediately rather than falling back to repo files."""
        calls = {"n": 0}
        monkeypatch.setattr(
            "htc.goldens.generator.complete",
            lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), LLMResponse(text="[]"))[1],
        )
        goldens = generate_goldens(repo, count=1, seed=1, scope="business")
        assert goldens == []
        assert calls["n"] == 0

    def test_unknown_scope_raises(self, repo):
        with pytest.raises(ValueError):
            generate_goldens(repo, count=1, scope="nonsense")


class TestSecretFileExclusion:
    def test_env_and_key_files_skipped(self, tmp_path):
        from htc.goldens.generator import _is_secret_file

        for name in (".env", ".env.local", "server.key", "prod.pem", "id_rsa", "credentials"):
            assert _is_secret_file(tmp_path / name) is True
        for name in ("main.py", "config.ts", "README.md", "settings.json"):
            assert _is_secret_file(tmp_path / name) is False

    def test_iter_files_excludes_secrets(self, tmp_path):
        from htc.goldens.generator import _iter_files

        (tmp_path / "app.py").write_text("x = 1")
        (tmp_path / ".env").write_text("SECRET_KEY=leaked")
        found = {p.name for p in _iter_files(tmp_path)}
        assert "app.py" in found
        assert ".env" not in found

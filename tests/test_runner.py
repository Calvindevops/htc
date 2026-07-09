"""Runner-level guards: judge verdict shape, malformed results.json handling,
and the parallel run_eval pipeline."""

from __future__ import annotations

import json

from htc.evaluation.runner import _judge, load_results, run_eval
from htc.goldens.generator import Golden
from htc.llm import LLMError, LLMResponse


def _golden(**overrides):
    base = dict(
        question="q",
        answer="a",
        artifact="app/main.py",
        category="config",
        difficulty=1,
    )
    base.update(overrides)
    return Golden(**base)


class TestJudgeArrayGuard:
    def test_array_reply_returns_wrong_without_raising(self, monkeypatch):
        monkeypatch.setattr(
            "htc.evaluation.runner.complete",
            lambda *a, **k: LLMResponse(text='[{"verdict": "correct"}]'),
        )
        verdict, reason = _judge(_golden(), "some answer")
        assert verdict == "wrong"
        assert "not a JSON object" in reason


class TestLoadResultsMalformed:
    def test_missing_root_and_agent_default_to_empty_string(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"items": []}))
        result = load_results(path)
        assert result.root == ""
        assert result.agent == ""


class TestRunEvalParallel:
    def test_all_items_present_and_scored_with_concurrency(self, tmp_path, monkeypatch):
        goldens = [_golden(question=f"q{i}") for i in range(6)]

        monkeypatch.setattr(
            "htc.evaluation.runner._builtin_agent",
            lambda root, question, model: f"answer to {question}",
        )
        monkeypatch.setattr(
            "htc.evaluation.runner._judge", lambda golden, answer: ("correct", "matches")
        )
        result = run_eval(tmp_path, goldens, concurrency=4)
        assert len(result.items) == 6
        assert {item.golden.question for item in result.items} == {g.question for g in goldens}
        assert all(item.verdict == "correct" for item in result.items)
        assert result.score == 100.0

    def test_agent_llm_error_is_skipped_not_fatal(self, tmp_path, monkeypatch, capsys):
        goldens = [_golden(question=f"q{i}") for i in range(6)]

        def flaky_agent(root, question, model):
            if question == "q3":
                raise LLMError("provider hiccup")
            return f"answer to {question}"

        monkeypatch.setattr("htc.evaluation.runner._builtin_agent", flaky_agent)
        monkeypatch.setattr(
            "htc.evaluation.runner._judge", lambda golden, answer: ("correct", "matches")
        )
        result = run_eval(tmp_path, goldens, concurrency=4)
        assert len(result.items) == 5
        assert "q3" not in {item.golden.question for item in result.items}
        assert "SKIPPED" in capsys.readouterr().err

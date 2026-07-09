"""Provider selection, OpenAI message translation, and the claude-cli lane."""

from __future__ import annotations

import json
import subprocess

import pytest

from htc.llm import (
    LLMError,
    ToolSpec,
    _claude_cli_call,
    _from_openai_message,
    _pick_provider,
    _post,
    _to_openai_messages,
)

TOOL = ToolSpec(name="read_file", description="d", input_schema={"type": "object"})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("HTC_PROVIDER", "ANTHROPIC_API_KEY", "HTC_LLM_BASE_URL", "HTC_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestPickProvider:
    def test_anthropic_key_wins(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        assert _pick_provider(None) == "anthropic"

    def test_openai_base_url(self, monkeypatch):
        monkeypatch.setenv("HTC_LLM_BASE_URL", "https://api.deepseek.com/v1")
        assert _pick_provider(None) == "openai"

    def test_explicit_override_beats_keys(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("HTC_PROVIDER", "claude-cli")
        assert _pick_provider(None) == "claude-cli"

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("HTC_PROVIDER", "bard")
        with pytest.raises(LLMError, match="unknown HTC_PROVIDER"):
            _pick_provider(None)

    def test_claude_cli_autodetect(self, monkeypatch):
        monkeypatch.setattr("htc.llm.shutil.which", lambda _: "/usr/bin/claude")
        assert _pick_provider(None) == "claude-cli"

    def test_nothing_configured_raises(self, monkeypatch):
        monkeypatch.setattr("htc.llm.shutil.which", lambda _: None)
        with pytest.raises(LLMError, match="No provider configured"):
            _pick_provider(None)


class TestOpenAITranslation:
    def test_plain_strings_pass_through(self):
        out = _to_openai_messages("sys", [{"role": "user", "content": "hi"}])
        assert out == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_tool_use_round_trip(self):
        # Assistant turn with a tool call, then the user turn carrying the result.
        history = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "data"}],
            },
        ]
        out = _to_openai_messages("sys", history)
        assistant = out[2]
        assert assistant["tool_calls"][0]["function"]["name"] == "read_file"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
        tool_msg = out[3]
        assert tool_msg == {"role": "tool", "tool_call_id": "t1", "content": "data"}

    def test_parse_tool_calls_from_response(self):
        response = _from_openai_message(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "t9",
                        "type": "function",
                        "function": {"name": "grep_content", "arguments": '{"pattern": "x"}'},
                    }
                ],
            }
        )
        assert response.wants_tools
        assert response.tool_calls[0].arguments == {"pattern": "x"}
        # raw_content is Anthropic-shaped so the agent loop can append it verbatim.
        assert response.raw_content[0]["type"] == "tool_use"

    def test_parse_plain_text_response(self):
        response = _from_openai_message({"content": "answer"})
        assert response.text == "answer"
        assert not response.wants_tools

    def test_tool_call_missing_id_gets_synthesized_id(self):
        response = _from_openai_message(
            {
                "content": None,
                "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a"}'}}],
            }
        )
        assert response.tool_calls[0].id == "call_0"
        assert response.tool_calls[0].name == "read_file"


class _FakeResponse:
    def __init__(self, status_code: int, json_body=None, text: str = "", headers=None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json


class TestPostRetry:
    def test_429_then_200_returns_body(self, monkeypatch):
        calls = {"n": 0}

        def fake_post(self, url, headers=None, json=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse(429, text="rate limited")
            return _FakeResponse(200, json_body={"ok": True})

        monkeypatch.setattr("httpx.Client.post", fake_post)
        monkeypatch.setattr("htc.llm.time.sleep", lambda _: None)
        assert _post("http://x", {}, {}) == {"ok": True}
        assert calls["n"] == 2

    def test_persistent_500_raises_after_retries(self, monkeypatch):
        calls = {"n": 0}

        def fake_post(self, url, headers=None, json=None):
            calls["n"] += 1
            return _FakeResponse(500, text="server error")

        monkeypatch.setattr("httpx.Client.post", fake_post)
        monkeypatch.setattr("htc.llm.time.sleep", lambda _: None)
        with pytest.raises(LLMError, match="500"):
            _post("http://x", {}, {})
        assert calls["n"] == len(("0.5", "1", "2", "4")) + 1

    def test_400_raises_immediately_without_retry(self, monkeypatch):
        calls = {"n": 0}

        def fake_post(self, url, headers=None, json=None):
            calls["n"] += 1
            return _FakeResponse(400, text="bad request")

        monkeypatch.setattr("httpx.Client.post", fake_post)
        monkeypatch.setattr("htc.llm.time.sleep", lambda _: None)
        with pytest.raises(LLMError, match="400"):
            _post("http://x", {}, {})
        assert calls["n"] == 1


class TestClaudeCli:
    def test_happy_path(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd == ["claude", "-p"]
            assert "sys" in kwargs["input"] and "question" in kwargs["input"]
            return subprocess.CompletedProcess(cmd, 0, stdout="the answer\n", stderr="")

        monkeypatch.setattr("htc.llm.subprocess.run", fake_run)
        response = _claude_cli_call("sys", [{"role": "user", "content": "question"}])
        assert response.text == "the answer"

    def test_nonzero_exit_raises(self, monkeypatch):
        monkeypatch.setattr(
            "htc.llm.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="boom"),
        )
        with pytest.raises(LLMError, match="claude -p failed"):
            _claude_cli_call("sys", [{"role": "user", "content": "q"}])

    def test_rejects_tool_history(self):
        with pytest.raises(LLMError, match="single-turn"):
            _claude_cli_call("sys", [{"role": "user", "content": [{"type": "tool_result"}]}])

"""Provider-agnostic LLM client for goldens generation, eval, and judging.

Three provider lanes, resolved by `HTC_PROVIDER` or auto-detected:

- **anthropic** — `ANTHROPIC_API_KEY` (default when set). Full tool use.
- **openai** — any OpenAI-compatible endpoint via `HTC_LLM_BASE_URL` +
  `HTC_LLM_API_KEY` (GLM, DeepSeek, Kimi, NVIDIA NIM, local servers, ...).
  Full tool use via function calling.
- **claude-cli** — the local Claude Code CLI (`claude -p`), billed to the
  user's subscription instead of API credits. Single-turn text only
  (generation + judging); the builtin tool-use agent needs an API lane.

One env var switch, no config files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_MODEL = "claude-sonnet-5"
ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class LLMError(RuntimeError):
    """Raised when the provider returns a non-retryable error."""


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call (Anthropic input_schema shape)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """One assistant turn: text and/or tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: list[dict[str, Any]] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0


def _openai_compatible() -> tuple[str, str] | None:
    base = os.environ.get("HTC_LLM_BASE_URL")
    if not base:
        return None
    key = os.environ.get("HTC_LLM_API_KEY", "")
    return base.rstrip("/"), key


def default_model() -> str:
    return os.environ.get("HTC_MODEL", DEFAULT_MODEL)


def judge_model() -> str:
    return os.environ.get("HTC_JUDGE_MODEL", default_model())


def _require_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError(
            "No provider configured. Set ANTHROPIC_API_KEY, or point HTC at any "
            "OpenAI-compatible endpoint with HTC_LLM_BASE_URL + HTC_LLM_API_KEY."
        )
    return key


RETRY_BACKOFFS = (0.5, 1.0, 2.0, 4.0)
MAX_RETRY_AFTER = 10.0


def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    """POST with exponential backoff on 429/5xx. Other 4xx fail fast (client error)."""
    last: httpx.Response | None = None
    with httpx.Client(timeout=TIMEOUT) as client:
        for attempt, backoff in enumerate((*RETRY_BACKOFFS, None)):
            res = client.post(url, headers=headers, json=body)
            if res.status_code < 400:
                return res.json()
            if res.status_code != 429 and res.status_code < 500:
                raise LLMError(f"{url} -> {res.status_code}: {res.text[:500]}")
            last = res
            if backoff is None:
                break
            wait = backoff
            retry_after = res.headers.get("retry-after")
            if retry_after:
                try:
                    wait = min(float(retry_after), MAX_RETRY_AFTER)
                except ValueError:
                    pass
            time.sleep(wait)
    assert last is not None
    raise LLMError(f"{url} -> {last.status_code} after retries: {last.text[:500]}")


def _anthropic_call(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[ToolSpec] | None,
    max_tokens: int,
) -> LLMResponse:
    key = _require_anthropic_key()
    body: dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
    data = _post(
        f"{ANTHROPIC_BASE}/messages",
        {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        body,
    )
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    content = data.get("content", [])
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block["input"]))
    return LLMResponse(text="".join(text_parts), tool_calls=calls, raw_content=content)


def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the internal (Anthropic-shaped) history to OpenAI chat messages.

    Anthropic shapes handled: plain-string content; assistant content-block lists
    (text + tool_use); user content-block lists of tool_result.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                text_parts.append(block["text"])
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                )
            elif kind == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    }
                )
        if tool_results:
            out.extend(tool_results)
        if text_parts or tool_calls:
            content_str = "".join(text_parts)
            entry: dict[str, Any] = {
                "role": message["role"],
                # Tool-call turns must carry a string content (even empty) — some
                # OpenAI-compatible providers reject `content: null` alongside tool_calls.
                "content": content_str if tool_calls else (content_str or None),
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
    return out


def _from_openai_message(message: dict[str, Any]) -> LLMResponse:
    """Parse an OpenAI chat message into LLMResponse with Anthropic-shaped raw_content."""
    text = message.get("content") or ""
    calls: list[ToolCall] = []
    raw: list[dict[str, Any]] = []
    if text:
        raw.append({"type": "text", "text": text})
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function", {})
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        call = ToolCall(
            id=tc.get("id") or f"call_{i}", name=fn.get("name", ""), arguments=arguments
        )
        calls.append(call)
        raw.append({"type": "tool_use", "id": call.id, "name": call.name, "input": arguments})
    return LLMResponse(text=text, tool_calls=calls, raw_content=raw)


def _openai_call(
    base: str,
    key: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[ToolSpec] | None,
    max_tokens: int,
) -> LLMResponse:
    body: dict[str, Any] = {
        "model": model,
        "messages": _to_openai_messages(system, messages),
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = _post(f"{base}/chat/completions", headers, body)
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as err:
        raise LLMError(f"unexpected completion shape: {json.dumps(data)[:300]}") from err
    return _from_openai_message(message)


def _flatten_for_cli(system: str, messages: list[dict[str, Any]]) -> str:
    """Concatenate a single-turn conversation into one prompt for `claude -p`."""
    parts = [system]
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        else:
            raise LLMError(
                "The claude-cli provider only supports single-turn text prompts "
                "(generation/judging). The builtin eval agent needs an API provider — "
                "or evaluate the CLI directly: htc eval --agent-cmd 'claude -p'."
            )
    return "\n\n".join(parts)


def _claude_cli_call(system: str, messages: list[dict[str, Any]]) -> LLMResponse:
    """Run the prompt through the local Claude Code CLI (bills the subscription).

    Transient CLI failures (empty output, nonzero exit with no stderr) happen
    under rapid sequential invocations, so failed calls retry with backoff.
    """
    prompt = _flatten_for_cli(system, messages)
    last_error = "unknown"
    for attempt in range(3):
        if attempt:
            time.sleep(3 * attempt)
        try:
            out = subprocess.run(
                ["claude", "-p"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as err:
            raise LLMError(
                "claude CLI not found on PATH (needed for HTC_PROVIDER=claude-cli)"
            ) from err
        except subprocess.TimeoutExpired:
            last_error = "timed out after 600s"
            continue
        if out.returncode != 0:
            last_error = f"exit {out.returncode}: {out.stderr[:300]}"
            continue
        text = out.stdout.strip()
        if not text:
            last_error = "produced no output"
            continue
        return LLMResponse(text=text)
    raise LLMError(f"claude -p failed after 3 attempts ({last_error})")


def _pick_provider(tools: list[ToolSpec] | None) -> str:
    """Resolve the provider: explicit HTC_PROVIDER, else key-based, else claude CLI."""
    explicit = os.environ.get("HTC_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in ("anthropic", "openai", "claude-cli"):
            raise LLMError(f"unknown HTC_PROVIDER '{explicit}' (anthropic | openai | claude-cli)")
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _openai_compatible():
        return "openai"
    if shutil.which("claude"):
        return "claude-cli"
    raise LLMError(
        "No provider configured. Set ANTHROPIC_API_KEY, or an OpenAI-compatible "
        "endpoint (HTC_LLM_BASE_URL + HTC_LLM_API_KEY), or install the Claude Code "
        "CLI and set HTC_PROVIDER=claude-cli."
    )


def complete(
    system: str,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[ToolSpec] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    """One model turn. `messages` uses Anthropic shape ({role, content}).

    Providers: anthropic (default when ANTHROPIC_API_KEY is set), openai (any
    OpenAI-compatible endpoint, tool use included), claude-cli (the local Claude
    Code CLI — subscription-billed, single-turn text only). Override with
    HTC_PROVIDER.
    """
    resolved = model or default_model()
    provider = _pick_provider(tools)
    if provider == "claude-cli":
        if tools:
            raise LLMError(
                "The claude-cli provider can't run the builtin tool-use agent. "
                "Use an API provider, or: htc eval --agent-cmd 'claude -p'."
            )
        return _claude_cli_call(system, messages)
    if provider == "openai":
        compat = _openai_compatible()
        if not compat:
            raise LLMError("HTC_PROVIDER=openai but HTC_LLM_BASE_URL is not set")
        base, key = compat
        return _openai_call(base, key, resolved, system, messages, tools, max_tokens)
    return _anthropic_call(resolved, system, messages, tools, max_tokens)


_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*")


def _find_balanced(s: str, opener: str, closer: str) -> str | None:
    """Return the first balanced `opener`..`closer` span in `s`, string-aware."""
    depth = 0
    begin: int | None = None
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == opener:
            if depth == 0:
                begin = i
            depth += 1
        elif ch == closer and depth > 0:
            depth -= 1
            if depth == 0 and begin is not None:
                return s[begin : i + 1]
    return None


def extract_json(text: str) -> Any:
    """Parse the first JSON value in a model reply (handles ```json fences,
    single-line fences, and prose containing multiple bracket groups)."""
    stripped = text.strip()
    stripped = _FENCE_OPEN_RE.sub("", stripped, count=1)
    if stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()[:-3]
    stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fall back to scanning for the first balanced JSON array/object span,
    # rather than find(opener)..rfind(closer) which breaks when the reply
    # contains more than one bracketed group.
    for opener, closer in (("[", "]"), ("{", "}")):
        span = _find_balanced(stripped, opener, closer)
        if span is None:
            continue
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"model reply contained no parseable JSON: {text[:300]}")

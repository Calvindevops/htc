# Providers

HTC talks to LLMs through one module, `src/htc/llm.py`. There is one env-var
switch and no config files. Three lanes:

- **anthropic** (default when `ANTHROPIC_API_KEY` is set) — full tool use,
  used by the builtin tool-use agent, goldens generation, and judging.
- **openai** — any OpenAI-compatible endpoint (`HTC_LLM_BASE_URL` +
  `HTC_LLM_API_KEY`): Groq, Nous Portal, DeepSeek, local servers, etc. Full
  tool use via OpenAI function calling, translated to/from HTC's internal
  Anthropic-shaped message format.
- **claude-cli** — the local Claude Code CLI (`claude -p`), billed to your
  Claude subscription instead of API credits. Single-turn text only.

## Env vars

| Var | Meaning |
|---|---|
| `HTC_PROVIDER` | Force a lane: `anthropic` \| `openai` \| `claude-cli`. Unset = auto-detect. |
| `HTC_MODEL` | Default model for generation/agent calls. Falls back to `claude-sonnet-5`. |
| `HTC_JUDGE_MODEL` | Model used for judging. Falls back to `HTC_MODEL`. |
| `HTC_LLM_BASE_URL` | Base URL for the `openai` lane (e.g. `https://api.groq.com/openai/v1`). |
| `HTC_LLM_API_KEY` | API key for the `openai` lane. |
| `ANTHROPIC_API_KEY` | API key for the `anthropic` lane. |

## Auto-detect precedence (`_pick_provider`)

1. Explicit `HTC_PROVIDER` — always wins if set.
2. `ANTHROPIC_API_KEY` present -> `anthropic`.
3. `HTC_LLM_BASE_URL` present -> `openai`.
4. `claude` binary on `PATH` -> `claude-cli`.
5. Otherwise: raises `LLMError` telling you to configure one of the above.

## Worked configs

### Groq

```bash
export HTC_PROVIDER=openai
export HTC_LLM_BASE_URL=https://api.groq.com/openai/v1
export HTC_LLM_API_KEY=gsk_...
export HTC_MODEL=openai/gpt-oss-120b
```

### Nous Portal

```bash
export HTC_PROVIDER=openai
export HTC_LLM_BASE_URL=https://inference-api.nousresearch.com/v1
export HTC_LLM_API_KEY=...
export HTC_MODEL=<model id from your Nous Portal account>
```

### Generic local server (vLLM, llama.cpp server, Ollama's OpenAI shim, etc.)

```bash
export HTC_PROVIDER=openai
export HTC_LLM_BASE_URL=http://localhost:8000/v1
export HTC_LLM_API_KEY=unused          # some servers require any non-empty value
export HTC_MODEL=<model name the server exposes>
```

### claude-cli lane

```bash
export HTC_PROVIDER=claude-cli
# no keys needed — uses your existing `claude` login/subscription
```

Requires the `claude` CLI on `PATH`. No `HTC_LLM_*` or `ANTHROPIC_API_KEY`
needed.

## claude-cli limitation: single-turn only

The claude-cli lane flattens the conversation into one prompt and shells out
to `claude -p`. It works for goldens generation and judging (both are
single-turn: system + one user message). It does **not** support the
builtin tool-use agent (`htc eval` without `--agent-cmd`) — that agent's loop
needs multi-turn tool_use/tool_result messages, which `claude -p` can't
consume. Calling `complete(..., tools=[...])` under `HTC_PROVIDER=claude-cli`
raises `LLMError` immediately.

If you want to evaluate Claude Code itself under this lane, don't use the
builtin agent — drive the CLI directly as the agent-under-test instead:

```bash
htc eval --root . --agent-cmd 'claude -p'
```

This pipes each golden's question to `claude -p`, which explores the repo
itself (it has real filesystem access), independent of `HTC_PROVIDER`. The
judge call afterward still goes through whatever `HTC_PROVIDER` lane you have
configured.

## Sandbox mode (Docker)

By default, `--agent-cmd` runs the agent command directly on the host
(`shell=True`, full filesystem access). `--sandbox` isolates that: the
command runs inside a Docker container with the repo mounted **read-only**
at `/repo` and no other host filesystem access.

Scope: `--sandbox` only applies to `--agent-cmd` mode. The builtin agent
(`htc eval` without `--agent-cmd`) runs in-process and is already
path-confined via its tool implementations — `--sandbox` has no effect on it
(a warning is printed if you pass `--sandbox` without `--agent-cmd`).

```bash
htc eval --root . --agent-cmd 'my-agent' --sandbox \
  --sandbox-network none \
  --sandbox-image python:3.12-slim
```

Flags:

| Flag | Meaning |
|---|---|
| `--sandbox` | enable Docker isolation for `--agent-cmd` |
| `--sandbox-image` | image to run the agent command in (default `python:3.12-slim`) |
| `--sandbox-network` | `bridge` (default, allows outbound API calls) or `none` (fully offline, for repo-only/local-model agents) |
| `--sandbox-env NAME` | forward a host env var (e.g. an API key) into the container by name; repeatable |

**Network tradeoff:** agents that call out to an LLM API need `--sandbox-network bridge`
plus `--sandbox-env` for their API key(s). Agents that only read the mounted
repo and use a local model can run fully offline with `--sandbox-network none`.

**Honest limitation:** host-authenticated CLIs like `claude -p` or `codex exec`
rely on local login state (browser session, keychain, config files) that
won't exist inside a fresh container. Sandbox mode is cleanest for
API-key-based agents (forward the key via `--sandbox-env`) and for
repo-only/local-model agents. To sandbox a host-authenticated CLI, build a
custom `--sandbox-image` that carries the tool and its auth baked in, or
forward whatever token/credential env var it accepts via `--sandbox-env`.

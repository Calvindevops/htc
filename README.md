# Hyperbolic Time Chamber (HTC)

**Onboard AI agents to your company in an afternoon, not a quarter.**

Your agents don't know your business. Every new agent session starts cold — and
teams burn weeks hand-writing context files, re-explaining the same tribal
knowledge, and guessing whether any of it worked. HTC compresses that time,
like its namesake chamber: **measure** what agents don't know about your
company, **generate** the knowledge pack that fixes it, and — when you want the
ceiling — **train** an open model that's natively fluent in your business.

```
Rung 1  MEASURE   htc goldens · htc eval   → repo-specific exam + Agent-Ready score
Rung 2  FIX       htc onboard              → context pack from the gaps, re-eval the delta
Rung 3  TRAIN     htc train [rl extra]     → GRPO-train an open model company-native
```

## Quickstart

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...   # or any OpenAI-compatible endpoint, see below

htc goldens --root /path/to/repo      # generate the knowledge exam (~20 questions)
htc eval    --root /path/to/repo      # score an agent → Agent-Ready scorecard
htc onboard --root /path/to/repo      # draft AGENTS.md from what it got wrong
# merge the draft, then prove the delta:
htc eval    --root /path/to/repo --compare /path/to/repo/.htc/results.json
```

![Agent-Ready](https://img.shields.io/badge/Agent--Ready-87%25-brightgreen)

Evaluate **your actual agent** instead of the builtin one:

```bash
htc eval --root . --agent-cmd 'claude -p'   # or any CLI that reads stdin, answers on stdout
```

## How it works

- **Goldens** — HTC samples your repo (weighted by git churn — the files that
  change hold the knowledge that matters) and generates grounded Q&A: every
  question names the `artifact` file a correct answer must cite, and items
  whose artifact doesn't exist are rejected. No generic trivia.
- **Eval** — an agent answers each golden using read-only, path-confined tools
  over your repo. An LLM judge grades against the reference answer; citing the
  right file is part of the rubric. Verdicts roll up into the **Agent-Ready
  score** (0–100) by category: architecture · config · behavior · ops.
- **Onboard** — the gaps become `AGENTS.md.htc-draft`: each missing fact stated
  as operating knowledge with its file pointer. Merge it, re-eval, screenshot
  the before/after.

## Providers

| Env | Meaning | Default |
|-----|---------|---------|
| `ANTHROPIC_API_KEY` | default provider | — |
| `HTC_MODEL` | agent/generation model | `claude-sonnet-5` |
| `HTC_JUDGE_MODEL` | judge model | `HTC_MODEL` |
| `HTC_LLM_BASE_URL` + `HTC_LLM_API_KEY` | any OpenAI-compatible endpoint | — |

## The extension point

Everything is generic via the **`CompanyAdapter`** protocol
(`src/htc/adapters/base.py`). Core ships `FilesystemAdapter` — point it at any
directory. A company adapter can bring more sources (`repo | docs | iac |
schema | tickets`), curated golden Q&A, and its own reward rubric — without
touching core. Multi-source connectors are the active roadmap.

## Rung 3 — the RL chamber

For teams that want more than context files: `pip install -e '.[rl]'` adds the
training chamber (built on [OpenPipe ART](https://github.com/openpipe/art),
GRPO). It reinforcement-trains an open-weight model (e.g. Qwen2.5-Coder LoRA)
against a read-only **twin** of your company — real reward signals from the
same goldens pipeline, so the model comes out *measurably* better at your
business, not just informed. Codebase-specific RL is sold today as a closed
service; HTC is the open on-ramp.

> Status: rungs 1–2 ship in v0.2. The RL chamber is functional scaffolding
> under active development — see `docs/spec.md` for the phased plan.

## Layout

| Path | What |
| --- | --- |
| `src/htc/adapters/` | `CompanyAdapter` protocol + `FilesystemAdapter` |
| `src/htc/goldens/` | repo → grounded golden Q&A |
| `src/htc/evaluation/` | agent runner · LLM judge · Agent-Ready scorecard |
| `src/htc/onboard/` | eval gaps → `AGENTS.md` context pack |
| `src/htc/twin/` | read-only MCP server over the company |
| `src/htc/llm.py` | provider-agnostic client (Anthropic / OpenAI-compatible) |

## License

[MIT](./LICENSE). Built on [OpenPipe ART](https://github.com/openpipe/art) (Apache-2.0).

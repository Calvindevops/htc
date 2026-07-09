# Hyperbolic Time Chamber (HTC)

**Measure how well an AI agent knows your codebase — and your company — then close the gap.**

Your agents start every session cold. Teams burn weeks hand-writing `CLAUDE.md`
files, re-explaining the same tribal knowledge, and guessing whether any of it
helped. HTC makes it measurable: **measure** what an agent doesn't know,
**generate** the context that fixes it, and **prove** the before/after.

```
MEASURE   htc goldens · htc eval     → a knowledge exam + Agent-Ready score
FIX       htc onboard · htc handbook → the context pack / handbook that closes gaps
ORIENT    htc studio                 → architecture diagrams + an audio-overview script
VALIDATE  htc study                  → prove the score predicts real task quality
```

## Quickstart (code-scope — works today, no setup beyond a key)

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...      # or any OpenAI-compatible endpoint; see Providers

htc goldens --root /path/to/repo         # generate a repo-specific knowledge exam
htc eval    --root /path/to/repo --agent-cmd 'claude -p'   # score YOUR agent
htc onboard --root /path/to/repo         # draft AGENTS.md from what it got wrong
htc eval    --root /path/to/repo --compare /path/to/repo/.htc/results.json   # prove the delta
```

![Agent-Ready](https://img.shields.io/badge/Agent--Ready-87%25-brightgreen)

## Company-scope (ingest more than code)

HTC ingests arbitrary local sources — code, docs, PDFs, DOCX/PPTX/XLSX, HTML,
and `.vtt/.srt` transcripts — into a persistent, cited memory, then generates
company knowledge from it:

```bash
pip install -e '.[ingest]'               # PDF/DOCX/PPTX/XLSX extractors (core stays light)
ollama pull nomic-embed-text             # recommended: semantic search via local Ollama (see docs/MEMORY.md)
pip install -e '.[embed]'                # optional offline fallback if you skip Ollama/cloud

htc handbook --root .                    # generate a structured Employee Handbook (draft)
htc goldens  --root . --scope business   # ask company questions, not just code
htc studio   --root . --kind diagram     # Mermaid architecture diagram from the memory
htc studio   --root . --kind podcast     # a 2-host audio-overview script
htc history  --root .                    # your Agent-Ready scores over time
```

Everything renders from the same cited memory layer — a local hybrid
BM25+semantic store by default (no external service), pluggable to
[gBrain](https://github.com/garrytan/gbrain), [Supermemory](https://supermemory.ai),
or a custom backend. See `docs/MEMORY.md`.

## Commands

| Command | What |
| --- | --- |
| `htc goldens` | generate a grounded knowledge exam (`--scope code\|business\|auto`) |
| `htc eval` | score an agent → Agent-Ready scorecard (builtin or `--agent-cmd`) |
| `htc onboard` | draft `AGENTS.md.htc-draft` from the gaps |
| `htc handbook` | generate a structured Employee Handbook from ingested sources |
| `htc studio` | render diagrams / mind-maps / an audio-overview script |
| `htc study` | run the correlation study that validates the score (`init/run/analyze`) |
| `htc history` | show your runs + Agent-Ready score trend |
| `htc twin` | read-only MCP server over your repo |

## How it works

- **Goldens** — HTC samples your sources (repo files weighted by git churn; docs
  via the ingestion layer) and generates grounded Q&A. Every question must cite a
  real source, must probe a decision / constraint / failure-mode / data-flow, and
  pure-lookup trivia is rejected. It tests understanding, not recall.
- **Eval** — an agent answers each golden using read-only, path-confined tools
  (it can't read the answer key). An LLM judge grades against the reference and
  whether the right source was cited → the **Agent-Ready score** (0–100) by
  category. Runs in parallel; `--agent-cmd 'claude -p'` measures your real agent.
- **Handbook / studio** — retrieve from the cited memory and generate onboarding
  artifacts (handbook, diagrams, audio-overview script) grounded in your sources.
- **Study** — an agent-ladder × task-bank design with blind human grading and a
  Spearman + bootstrap-CI verdict. This is what earns a "company-ready" claim —
  until you run it, treat the score as *knowledge coverage*, not a performance guarantee.

## Providers

| Env | Meaning | Default |
|-----|---------|---------|
| `ANTHROPIC_API_KEY` | default provider | — |
| `HTC_MODEL` / `HTC_JUDGE_MODEL` | agent-generation / judge model | `claude-sonnet-5` |
| `HTC_LLM_BASE_URL` + `HTC_LLM_API_KEY` | any OpenAI-compatible endpoint (GLM, DeepSeek, Groq, Nous, local) | — |
| `HTC_PROVIDER=claude-cli` | run generation/judging on the local Claude CLI (subscription-billed) | auto-detected |

See `docs/PROVIDERS.md` for worked configs.

## Sandbox, privacy, tracking

- `htc eval --agent-cmd '…' --sandbox` runs the agent in a Docker container with
  the repo mounted read-only (isolates untrusted agent commands from your host).
- **Telemetry is opt-in and off by default** — no phone-home. Error tracking
  (Sentry) and usage stats (PostHog) activate only when you set their env vars.
- Run history is local (`.htc/history/`) — your data, your machine.

## Roadmap

Company-scope ingestion, handbook, studio, and the study harness ship in this
release. Next: live connectors (Slack/CRM), the Training chamber (RL/GRPO
fine-tuning of an open model into a company-native agent — the `[rl]` extra), and
a hosted tier. Building in the open.

## Layout

| Path | What |
| --- | --- |
| `src/htc/adapters/` | `CompanyAdapter` protocol + `FilesystemAdapter` |
| `src/htc/goldens/` | sources → grounded golden Q&A (code + business scope) |
| `src/htc/evaluation/` | agent runner · LLM judge · Agent-Ready scorecard |
| `src/htc/onboard/` · `src/htc/handbook/` | context pack · Employee Handbook |
| `src/htc/world_model/` | ingestion · memory · studio renderers |
| `src/htc/study/` | correlation-study harness |
| `src/htc/llm.py` | provider-agnostic client |

## License

[MIT](./LICENSE). Optional training extra builds on [OpenPipe ART](https://github.com/openpipe/art) (Apache-2.0).

# HTC v0.2 — The Eval Wedge (Product Spec)

**One-liner:** Onboard AI agents to your company in an afternoon, not a quarter.

**The vision:** a company ingests everything it knows — code, docs, configs,
SOPs, schemas, tickets — into its own environment, and agents come out
*company-native on week one*: no months of hand context-engineering, no
re-teaching every new agent the same tribal knowledge. Like the namesake
chamber, HTC compresses onboarding time: measure what agents don't know,
generate the knowledge pack that fixes it, and (for teams that want it) train
a model that's natively fluent in the business.

## Positioning

HTC is a ladder with three rungs. v0.2 ships rungs 1 and 2 end-to-end; rung 3
(the RL chamber) stays behind the `[rl]` extra as the advanced tier. The
Agent-Ready score is the receipt for the whole ladder: "agents scored 34% on
your business cold; after the chamber, 89% — deployed in week one."

v0.2 scopes ingestion to what `FilesystemAdapter` reaches (any files in the
repo/directory: code, docs, configs). Multi-source connectors (tickets, wikis,
schemas via new adapters) are the v0.3 roadmap — the `CompanyAdapter.sources()`
contract already models them (`SourceKind = repo | docs | iac | schema | tickets`).

| Rung | Command | What it does | Cost to run |
|------|---------|-------------|-------------|
| 1. Measure | `htc goldens` + `htc eval` | Generate repo-specific golden Q&A, score any agent against them, print a scorecard | 1 API key, ~cents |
| 2. Fix | `htc onboard` | Turn the *failed* goldens into an agent onboarding doc (`AGENTS.md` draft); re-eval shows the delta | 1 API key, ~cents |
| 3. Train | `htc train` | GRPO/ART chamber that trains a company-native model against the twin | GPU / serverless |

The 5-minute magic moment: `pipx install htc && htc goldens --root . && htc eval --root .`
→ a scorecard that tells you something *true and surprising* about what agents
miss in your repo.

## Why this shape wins

- Adjacent tools trace agent runs (Arize) or provide generic eval infra
  (Agent-EvalKit). Nobody generates a **repo-specific knowledge exam** and
  nobody ships the measure→fix→train ladder in one tool.
- Rung 2 is the adoption engine: zero GPU, immediate visible payoff, produces a
  before/after number people screenshot and share.
- Rung 3 is the ceiling that makes the project credible as infrastructure
  (commercial validation: codebase-specific RL is sold today as a service).

## Architecture (v0.2 additions)

```
src/htc/
├── llm.py                  # provider-agnostic completion + tool-use client
│                           #   Anthropic (default) or any OpenAI-compatible base URL
├── goldens/
│   ├── __init__.py
│   └── generator.py        # repo → goldens.json (grounded, verifiable Q&A)
├── evaluation/
│   ├── __init__.py
│   ├── runner.py           # agent-under-test loop + LLM judge
│   └── scorecard.py        # terminal + markdown scorecard, badge line
└── onboard/
    ├── __init__.py
    └── writer.py           # failed goldens → AGENTS.md draft
```

### Goldens generation (`htc goldens`)

1. Walk the repo through the **twin** (path-confined, read-only) — never raw FS.
2. Sample candidate files, weighted by git churn when git history exists
   (frequently-changed files hold the knowledge that matters).
3. LLM generates Q&A items per batch. Every item must carry:
   - `question` — answerable only with real repo knowledge, not generic trivia
   - `answer` — short reference answer
   - `artifact` — file path (or graph node) a correct answer must cite
   - `category` — `architecture` | `config` | `behavior` | `ops`
   - `difficulty` — 1–3
4. Validation pass: an item is kept only if its `artifact` exists in the twin.
5. Output: `.htc/goldens.json`. Deterministic seed option for CI.

### Eval (`htc eval`)

Two agent-under-test modes:

- **builtin** (default): a minimal tool-use loop — the model gets the twin's
  MCP tools (`search_files`, `grep`, `read_file`, `query_graph`) and answers
  each golden. This measures *model + tools*, no external harness needed.
- **cmd**: `htc eval --agent-cmd 'claude -p'` — pipe each question to any CLI
  agent that explores the repo itself. This measures *your actual agent*.

Judging: LLM judge scores each transcript against the reference answer + artifact
(cited-the-right-file is part of the rubric). Verdicts: `correct` |
`partial` | `wrong`, with a one-line reason. Judge model defaults to the same
provider; overridable (`HTC_JUDGE_MODEL`).

### Scorecard

- Terminal table: per-category accuracy, overall **Agent-Ready score** (0–100).
- `.htc/scorecard.md` — sharable markdown with the same content.
- Badge line: shields.io URL embedding the score for READMEs.
- `--compare before.json` renders the delta after `htc onboard` (the screenshot).

### Onboard (`htc onboard`) — the context pack

The flagship rung: turn measured knowledge gaps into the onboarding pack that
makes any agent company-native.

1. Read the latest eval results; collect `wrong` + `partial` items (the
   gaps), plus all goldens (the knowledge inventory).
2. LLM drafts the **context pack**: an `AGENTS.md.htc-draft` — for each
   knowledge gap, the fact + the pointer (artifact path) an agent needs,
   organized by category (architecture / config / behavior / ops).
3. Never overwrite an existing `AGENTS.md` — always the `.htc-draft` suffix.
4. Print next step: review → merge into `AGENTS.md`/`CLAUDE.md` → `htc eval`
   again → `--compare` shows the time-compression delta (the screenshot).

## Config

| Env | Meaning | Default |
|-----|---------|---------|
| `ANTHROPIC_API_KEY` | default provider key | — |
| `HTC_MODEL` | agent-under-test model (builtin mode) | `claude-sonnet-5` |
| `HTC_JUDGE_MODEL` | judge model | `HTC_MODEL` |
| `HTC_LLM_BASE_URL` | OpenAI-compatible base URL (overrides Anthropic) | — |
| `HTC_LLM_API_KEY` | key for that base URL | — |

No config files required. One key, three commands.

## Non-goals for v0.2

- No hosted anything. No accounts. No telemetry.
- No writable twin (P2, per loop-architecture.md).
- RL chamber unchanged — `[rl]` extra, documented but not the front door.

## Quality bar (release gate for v0.2 tag)

- [ ] `pipx install .` on a clean machine → all three commands work with only
      `ANTHROPIC_API_KEY` set
- [ ] pytest suite green (unit: goldens validation, scorecard math, path
      safety; no network in tests)
- [ ] `ruff check` + `black --check` clean; CI workflow runs both + pytest
- [ ] README rewritten as the public front door (quickstart ≤ 10 lines)
- [ ] Internal spec (`docs/spec.md`) genericized — no private endpoints,
      entities, or absolute paths

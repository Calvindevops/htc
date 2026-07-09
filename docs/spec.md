# HTC — Specification

## What it is

A chamber that turns a generic agent into a **company-native operator**. Four
capabilities, built on a `CompanyAdapter` so the core stays company-agnostic:

1. **World-model** — ingest repos/docs/infra → knowledge graph + generated
   context **+** an executable twin agents can act against.
2. **Reinforcement chamber** — a student open-weights model (e.g.
   Qwen2.5-Coder-7B, LoRA) does rollouts vs the twin; reward = deterministic
   checks + LLM-judge ranking; GRPO updates weights.
3. **Self-improvement loops** — ongoing auto-research finds knowledge gaps,
   runs experiments vs the twin, and updates **both** the knowledge layer
   **and** the policy. (Past read-only auto-research: this adds reward-driven
   *practice*.)
4. **Knowledge layer** — graph + generated onboarding context, kept fresh by 3.

v0.2 ships the **eval wedge** end-to-end (measure → fix); see
`docs/product-spec-v0.2.md`. This document covers the full phased architecture.

## Capability → component map

| Capability | Module |
| --- | --- |
| Goldens (knowledge exam) | `goldens/generator.py` |
| Eval + scorecard | `evaluation/{runner,scorecard}.py` |
| Context pack | `onboard/writer.py` |
| Twin | `twin/{server,mcp_server}.py` |
| RL chamber | `chamber/{rollout,reward,scenarios,train}.py` (phased) |
| World-model | `world_model/{ingest,graph,wiki}.py` (phased) |
| Loops | `loops/{gap_finder,research_loop}.py` (phased) |
| Extension point | `adapters/base.py` (`CompanyAdapter`) |

## RL chamber design notes

- Built on [OpenPipe ART](https://github.com/openpipe/art) (GRPO); the rollout
  connects to the twin's MCP server over stdio.
- Training a 7B model with GRPO does not fit consumer GPUs (a 12 GB card holds
  policy + reference + optimizer + vLLM engine only for much smaller models) —
  the reference path is a serverless training backend, with local GPU used for
  inference/eval only.
- Judge is any strong LLM via the provider-agnostic client (`llm.py`).
- Reward = artifact-citation gate (deterministic) + judge ranking. The goldens
  pipeline doubles as the held-out eval set.
- **Success bar convention:** a trained student must beat its base model on
  held-out Q&A by a pre-committed margin (reference: ≥15 percentage points),
  fixed before training starts.

## Phases

- **P0 — scaffold (done):** `CompanyAdapter` + `FilesystemAdapter`, read-only
  twin MCP server, CLI.
- **v0.2 — eval wedge (this release):** goldens → eval → scorecard → context
  pack. One key, three commands, no GPU.
- **P1 — chamber MVP:** one task family (codebase/infra Q&A), student LoRA,
  beat-base gate on held-out goldens.
- **P2 — writable twin:** sandboxed mutating tasks (git-worktree reset
  semantics), broader task families.
- **P3 — loops:** gap-finder + research loop keep the knowledge layer and the
  policy improving on a schedule.

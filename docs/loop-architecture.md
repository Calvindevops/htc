# HTC loop architecture — the Odyssey Engine mapping

HTC's chamber is, structurally, the **Odyssey Engine** pattern from the autoresearch ecosystem:
a fusion of three loop styles. Naming them explicitly clarifies which HTC component plays which role,
and — more importantly — pins down which parts must stay **immutable** vs which are the **tunable program**.

The three loops (per [yibie/awesome-autoresearch](https://github.com/yibie/awesome-autoresearch),
[7-patterns taxonomy](https://zenn.dev/0xmamedai/articles/01237cd7c50282?locale=en)):

| Loop style | What it does | Type |
|---|---|---|
| **autoresearch** | metric-driven propose → train → measure → keep/discard | discovery (explore for improvements) |
| **gaggle-iterate** | checkpoint → verify → rollback | safety (never regress) |
| **ralph-loop** | run continuously until a standard is met | delivery (drive to completion) |

## Mapping onto HTC

```
ralph-loop  ─────────────────────────────────────────────────────────┐  (loops/research_loop.py)
  drives continuously until: trained student beats base by the margin   │   keep going until standard met
  and gap_finder reports no open gaps                                    │
        │                                                                │
        ▼                                                                │
  autoresearch (discovery) ── the RL chamber ──────────────┐            │
    chamber/scenarios.py   = program.md analog (TUNABLE)     │            │
    chamber/rollout.py     = the agent producing "mutations" │            │
    chamber/reward.py      = prepare.py analog (IMMUTABLE)   │            │
    chamber/train.py       = GRPO keep-if-beats-base         │            │
        │  keep the LoRA update only if it improves          │            │
        ▼                                                     │            │
  gaggle-iterate (safety) ── twin + eval ───────────────────┘            │
    twin/workspace.py  = git-worktree sandbox, reset per episode (checkpoint/rollback)
    eval/scorecard.py  = calculate_beat_comp (verify beats-base)         │
    P1 Q&A eval        = forgetting guardrail in every P2 loop (rollback if regressed)
                                                                          │
        └─────────────────────────────────────────────────────────────────┘
```

### autoresearch layer = the chamber (`chamber/`)

The RL chamber already *is* an autoresearch discovery loop. The non-negotiable discipline carries over
directly:

- **`chamber/reward.py` is the immutable evaluator** (the `prepare.py` / `val_bpb` analog). It is the
  deterministic artifact-citation gate + RULER ranking with the "fail-gate ranks lowest" rule. **Never tune
  the reward to make a policy look good** — that is reward-hacking, the exact failure the deterministic gate
  exists to block. If a fluent-but-wrong trajectory scores high, fix the *scenarios/rubric*, not the reward math.
- **`chamber/scenarios.py` + the rubric are the `program.md`** (tunable). They define *what to explore* — which
  Q&A, which task family, which constraints. This is where you steer the chamber, the same way you tune a
  skill-loop's `program.md`. Keep the metric out of it.
- **`chamber/train.py` is the mutating surface** (the LoRA policy = `train.py`). GRPO keeps the update only if
  it beats base — the keep/discard decision.

### gaggle-iterate layer = the twin + eval (`twin/`, `eval/`)

This is the safety loop that makes the discovery loop trustworthy:

- **`twin/workspace.py`** — the git-worktree sandbox is literally checkpoint/rollback: writable per episode,
  reset to clean state after. A mutating episode (P2 "infra change proposal") that fails its `run_check`
  rolls back; the worktree stays clean — the same git-keep/discard hygiene as autoresearch's auto-revert.
- **`eval/scorecard.py`** — `calculate_beat_comp` is the verify step: a trained student is only accepted if it
  beats base by the fixed margin (default ≥15pp, set before training).
- **Forgetting guardrail** — the P1 Q&A eval runs inside every P2 loop. A new LoRA that regresses P1 is rolled
  back even if it improved the new task. This is gaggle-iterate's "verify before commit" applied across task
  families.

### ralph-loop layer = the self-improvement loops (`loops/`)

The outer delivery loop that keeps the whole thing running unattended (P2):

- **`loops/gap_finder.py`** — finds the open work: graph holes + low-reward scenario clusters. Defines "the
  standard not yet met."
- **`loops/research_loop.py`** — runs continuously: gap → experiment vs twin → if knowledge gap, update
  wiki/graph; if skill gap, add to the training set and retrain the LoRA. Keeps going until gap_finder is dry
  and the scorecard margin holds. This is ralph-loop's "work until the job is done to a standard," wrapping the
  autoresearch discovery loop inside.

## Why naming this matters

1. **It tells you what you may never tune.** `reward.py` and the eval metric are immutable. Scenarios/rubric
   are the program. Mixing them is how RL loops start rewarding the wrong thing.
2. **It separates discovery from delivery.** The chamber *discovers* improvements; `research_loop` *delivers*
   to a standard. They are different loops with different stop conditions — don't collapse them.
3. **It is the beyond-Karpathy edge from the plan.** Plain autoresearch updates code against one metric. HTC's
   research_loop closes the loop onto **both** the knowledge layer (graph/wiki) *and* the policy (LoRA), with
   gaggle-iterate safety so it can run unattended without regressing. That combination is the Odyssey shape.

## Strategy configs (the engineer/creative/production analog)

Odyssey Engine ships three orientation presets. HTC's analog is per-**task-family** config (set in the
`CompanyAdapter`), not per-mood:

- **codebase/infra Q&A** (P1) — strict deterministic citation gate dominant; RULER as secondary signal.
- **infra-change-proposal** (P2) — `run_check` (tests/lint/typecheck in the worktree) is the gate; reward is
  pass-rate driven.
- future task families — each defines its own gate in `reward.py` + scenarios in the adapter, but the
  immutable-evaluator / tunable-program separation holds for all of them.

> Cross-reference: skill-mining loops are the same pattern at toy
> scale (immutable `evaluate.py` + tunable `program.md` + a propose/keep/discard loop). It's the cheapest
> place to build intuition for the discipline before applying it to the GPU-cost chamber here.

"""`htc` command-line entry point.

v0.2 ships the eval wedge: `goldens` (generate the repo knowledge exam),
`eval` (score an agent, print the Agent-Ready scorecard), and `onboard`
(draft the context pack from the gaps). `twin` boots the read-only twin.
`train` and `loop` (the RL chamber) land per docs/spec.md phases.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from . import history, telemetry
from .adapters.base import Source
from .adapters.filesystem import FilesystemAdapter
from .errors_tracking import capture_exception, init_error_tracking
from .llm import LLMError
from .sandbox import SandboxConfig, SandboxError
from .twin.server import list_tools, twin_server_params

HTC_DIR = ".htc"


def _htc_path(root: str, name: str) -> Path:
    return Path(root).expanduser().resolve() / HTC_DIR / name


def _safe_provider() -> str:
    """Best-effort provider name for telemetry buckets; never raises."""
    try:
        from .llm import _pick_provider

        return _pick_provider(None)
    except Exception:
        return "unknown"


def _cmd_twin(args: argparse.Namespace) -> int:
    adapter = FilesystemAdapter(args.root)
    params = twin_server_params(adapter, graph_path=args.graph)
    if args.list_tools:
        tools = asyncio.run(list_tools(params))
        print(f"twin[{adapter.name()}] tools: {', '.join(tools)}")
    else:
        print(f"twin ready for '{adapter.name()}' at {adapter.root}")
        print("  run with --list-tools to verify the MCP server boots.")
    return 0


def _cmd_goldens(args: argparse.Namespace) -> int:
    from .goldens import generate_goldens, save_goldens

    start = time.perf_counter()

    def on_batch(batch_index: int, running_total: int) -> None:
        print(f"  batch {batch_index}: {running_total} goldens so far", file=sys.stderr)

    print(f"generating ~{args.count} goldens for {args.root} ...")
    goldens = generate_goldens(
        args.root,
        count=args.count,
        seed=args.seed,
        model=args.model,
        balance=args.balance,
        on_batch=on_batch,
        scope=args.scope,
    )
    if not goldens:
        print("no goldens generated — is the repo empty, or did every item fail validation?")
        return 1
    out = Path(args.output) if args.output else _htc_path(args.root, "goldens.json")
    save_goldens(goldens, out)
    categories: dict[str, int] = {}
    for g in goldens:
        categories[g.category] = categories.get(g.category, 0) + 1
    breakdown = " · ".join(f"{n} {c}" for c, n in sorted(categories.items()))
    print(f"wrote {len(goldens)} goldens ({breakdown}) -> {out}")
    if len(goldens) < args.count:
        print(
            f"warning: generated {len(goldens)} goldens, fewer than the {args.count} requested",
            file=sys.stderr,
        )
    print(f"next: htc eval --root {args.root}")
    history.record_run(args.root, "goldens", {"count": len(goldens), "categories": categories})
    telemetry.track(
        "command_run",
        {
            "command": "goldens",
            "repo_size_bucket": telemetry.bucket_repo_size(len(goldens)),
            "provider": _safe_provider(),
            "duration_bucket": telemetry.bucket_duration(time.perf_counter() - start),
        },
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import render_compare, render_scorecard, scorecard_markdown
    from .evaluation.runner import ItemResult, load_results, run_eval, save_results
    from .goldens import load_goldens

    start = time.perf_counter()
    goldens_path = Path(args.goldens) if args.goldens else _htc_path(args.root, "goldens.json")
    if not goldens_path.is_file():
        print(f"no goldens at {goldens_path} — run `htc goldens --root {args.root}` first.")
        return 1
    goldens = load_goldens(goldens_path)

    def progress(index: int, total: int, item: ItemResult) -> None:
        print(f"  [{index}/{total}] {item.verdict:<7} {item.golden.question[:70]}")

    sandbox = None
    if args.sandbox:
        if not args.agent_cmd:
            print(
                "warning: --sandbox only applies to --agent-cmd mode; "
                "running unsandboxed builtin agent.",
                file=sys.stderr,
            )
        else:
            sandbox = SandboxConfig(
                image=args.sandbox_image,
                network=args.sandbox_network,
                env_passthrough=tuple(args.sandbox_env or ()),
            )

    print(f"evaluating {len(goldens)} goldens against {args.agent_cmd or 'builtin agent'} ...")
    try:
        result = run_eval(
            args.root,
            goldens,
            agent_cmd=args.agent_cmd,
            model=args.model,
            on_item=progress,
            concurrency=args.concurrency,
            sandbox=sandbox,
        )
    except SandboxError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    results_path = Path(args.output) if args.output else _htc_path(args.root, "results.json")
    save_results(result, results_path)
    md_path = _htc_path(args.root, "scorecard.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(scorecard_markdown(result))
    print(render_scorecard(result))
    print(f"\n  results -> {results_path}\n  scorecard -> {md_path}")
    if args.compare:
        before = load_results(args.compare)
        print(render_compare(before, result))
    history.record_run(args.root, "eval", {"score": result.score, "num_goldens": len(goldens)})
    telemetry.track(
        "eval_completed",
        {"score_bucket": telemetry.bucket_score(result.score), "num_goldens": len(goldens)},
    )
    telemetry.track(
        "command_run",
        {
            "command": "eval",
            "repo_size_bucket": telemetry.bucket_repo_size(len(goldens)),
            "provider": _safe_provider(),
            "duration_bucket": telemetry.bucket_duration(time.perf_counter() - start),
        },
    )
    return 0


def _cmd_onboard(args: argparse.Namespace) -> int:
    from .evaluation.runner import load_results
    from .onboard import write_context_pack

    start = time.perf_counter()
    results_path = Path(args.results) if args.results else _htc_path(args.root, "results.json")
    if not results_path.is_file():
        print(f"no eval results at {results_path} — run `htc eval --root {args.root}` first.")
        return 1
    result = load_results(results_path)
    print(f"drafting context pack from {results_path} (score {result.score}) ...")
    draft = write_context_pack(args.root, result, model=args.model)
    history.record_run(
        args.root, "onboard", {"score": result.score, "gaps_found": draft is not None}
    )
    telemetry.track(
        "command_run",
        {
            "command": "onboard",
            "provider": _safe_provider(),
            "duration_bucket": telemetry.bucket_duration(time.perf_counter() - start),
        },
    )
    if draft is None:
        print("no knowledge gaps — the agent scored perfectly. Nothing to onboard.")
        return 0
    print(f"context pack -> {draft}")
    print("next: review + merge into AGENTS.md / CLAUDE.md, then re-run:")
    print(f"  htc eval --root {args.root} --compare {results_path}")
    return 0


def _parse_sources(raw: list[str] | None) -> list[Source] | None:
    """Parse repeatable `--sources` values: 'path' or 'path:kind' (kind defaults
    to 'docs')."""
    if not raw:
        return None
    sources = []
    for item in raw:
        path, _, kind = item.rpartition(":")
        if not path:
            path, kind = item, "docs"
        sources.append(Source(path=path, kind=kind or "docs"))
    return sources


def _cmd_handbook(args: argparse.Namespace) -> int:
    from .handbook import DRAFT_NAME, generate_handbook

    start = time.perf_counter()
    sources = _parse_sources(args.sources)
    print(f"generating handbook for {args.root} ...")
    generate_handbook(args.root, sources=sources, model=args.model)
    draft = Path(args.root).expanduser().resolve() / DRAFT_NAME
    print(f"handbook -> {draft}")
    history.record_run(args.root, "handbook", {"num_sources": len(sources) if sources else 0})
    telemetry.track(
        "command_run",
        {
            "command": "handbook",
            "provider": _safe_provider(),
            "duration_bucket": telemetry.bucket_duration(time.perf_counter() - start),
        },
    )
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    entries = history.load_history(args.root)
    if not entries:
        print(f"no run history at {args.root} yet — run goldens/eval/onboard/handbook first.")
        return 0
    for entry in entries:
        print(f"  [{entry['index']}] {entry['kind']:<9} {entry['summary']}")
    trend = history.score_trend(args.root)
    if trend:
        print("\nscore trend: " + " -> ".join(str(s) for s in trend))
    return 0


def _not_yet(args: argparse.Namespace) -> int:
    print(f"`htc {args._cmd}` is not implemented yet (see docs/spec.md phases).")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="htc", description="Hyperbolic Time Chamber")
    sub = parser.add_subparsers(dest="_cmd", required=True)

    p_twin = sub.add_parser("twin", help="boot the read-only company twin")
    p_twin.add_argument("--root", required=True, help="path to the company repo")
    p_twin.add_argument("--graph", default=None, help="optional graphify graph.json")
    p_twin.add_argument("--list-tools", action="store_true", help="connect and list tools")
    p_twin.set_defaults(func=_cmd_twin)

    p_gold = sub.add_parser("goldens", help="generate the repo-specific knowledge exam")
    p_gold.add_argument("--root", required=True, help="path to the company repo")
    p_gold.add_argument("--count", type=int, default=20, help="target number of goldens")
    p_gold.add_argument("--seed", type=int, default=None, help="deterministic file sampling")
    p_gold.add_argument("--model", default=None, help="generation model override")
    p_gold.add_argument(
        "-o", "--output", default=None, help="output path (default .htc/goldens.json)"
    )
    p_gold.add_argument(
        "--balance",
        action="store_true",
        help="steer generation toward under-covered categories",
    )
    p_gold.add_argument(
        "--scope",
        default="code",
        choices=("code", "business", "auto"),
        help="generation prompt: code knowledge, business/process knowledge, or auto-detect",
    )
    p_gold.set_defaults(func=_cmd_goldens)

    p_eval = sub.add_parser("eval", help="score an agent against the goldens")
    p_eval.add_argument("--root", required=True, help="path to the company repo")
    p_eval.add_argument("--goldens", default=None, help="goldens path (default .htc/goldens.json)")
    p_eval.add_argument(
        "--agent-cmd", default=None, help="external agent command, e.g. 'claude -p'"
    )
    p_eval.add_argument("--model", default=None, help="builtin agent model override")
    p_eval.add_argument("--compare", default=None, help="earlier results.json to diff against")
    p_eval.add_argument(
        "--concurrency", type=int, default=4, help="parallel goldens in flight (default 4)"
    )
    p_eval.add_argument(
        "-o", "--output", default=None, help="results path (default .htc/results.json)"
    )
    p_eval.add_argument(
        "--sandbox",
        action="store_true",
        help="run --agent-cmd inside a read-only Docker container (no effect on builtin agent)",
    )
    p_eval.add_argument(
        "--sandbox-image", default="python:3.12-slim", help="Docker image for --sandbox"
    )
    p_eval.add_argument(
        "--sandbox-network",
        default="bridge",
        choices=("bridge", "none"),
        help="Docker --network for --sandbox: bridge (API calls) or none (repo-only agents)",
    )
    p_eval.add_argument(
        "--sandbox-env",
        action="append",
        default=None,
        help="host env var NAME to forward into the sandbox container (repeatable)",
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_onb = sub.add_parser("onboard", help="draft the context pack from eval gaps")
    p_onb.add_argument("--root", required=True, help="path to the company repo")
    p_onb.add_argument("--results", default=None, help="results path (default .htc/results.json)")
    p_onb.add_argument("--model", default=None, help="writer model override")
    p_onb.set_defaults(func=_cmd_onboard)

    p_hand = sub.add_parser("handbook", help="generate the structured onboarding handbook")
    p_hand.add_argument("--root", required=True, help="path to the company repo")
    p_hand.add_argument("--model", default=None, help="generation model override")
    p_hand.add_argument(
        "--sources",
        action="append",
        default=None,
        help="extra source to ingest, 'path' or 'path:kind' (repeatable; "
        "defaults to the repo via the filesystem adapter)",
    )
    p_hand.set_defaults(func=_cmd_handbook)

    p_hist = sub.add_parser("history", help="show run history and score trend")
    p_hist.add_argument("--root", required=True, help="path to the company repo")
    p_hist.set_defaults(func=_cmd_history)

    for name in ("ingest", "train", "loop"):
        p = sub.add_parser(name, help=f"{name} (not yet implemented)")
        p.set_defaults(func=_not_yet)

    args = parser.parse_args(argv)
    init_error_tracking()
    telemetry.ensure_preference()
    try:
        return args.func(args)
    except LLMError as err:
        print(f"error: {err}", file=sys.stderr)
        capture_exception(err)
        return 2
    except Exception as err:
        capture_exception(err)
        raise


if __name__ == "__main__":
    sys.exit(main())

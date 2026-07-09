"""`htc` command-line entry point.

v0.2 ships the eval wedge: `goldens` (generate the repo knowledge exam),
`eval` (score an agent, print the Agent-Ready scorecard), and `onboard`
(draft the context pack from the gaps). `twin` boots the read-only twin.
`study` runs the correlation-study harness (agent-ladder x task-bank, blind
grading) that validates whether the Agent-Ready score predicts real task
performance; grading itself is human-in-the-loop.
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


def _build_reranking_memory(root: str, sources: list[Source] | None, rerank_name: str):
    """Build memory and wrap it with a reranker when `--rerank` isn't "none";
    otherwise return `None` so the caller builds memory itself, unchanged
    from today's behavior. See HTC_RERANKER / `htc.world_model.rerank`."""
    if rerank_name == "none":
        return None
    from .world_model.build import build_memory
    from .world_model.rerank import RerankingMemoryStore, get_reranker

    root_path = Path(root).expanduser().resolve()
    store = build_memory(sources or FilesystemAdapter(str(root_path)).sources(), root_path)
    return RerankingMemoryStore(store, get_reranker(rerank_name))


def _cmd_handbook(args: argparse.Namespace) -> int:
    from .handbook import DRAFT_NAME, generate_handbook

    start = time.perf_counter()
    sources = _parse_sources(args.sources)
    print(f"generating handbook for {args.root} ...")
    memory = _build_reranking_memory(args.root, sources, args.rerank)
    generate_handbook(args.root, sources=sources, model=args.model, memory=memory)
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


def _cmd_studio(args: argparse.Namespace) -> int:
    from .world_model.render import generate_diagram, generate_podcast_script, render_audio

    start = time.perf_counter()
    sources = _parse_sources(args.sources)
    root_path = Path(args.root).expanduser().resolve()
    print(f"generating {args.kind} studio artifact for {args.root} ...")
    memory = _build_reranking_memory(args.root, sources, args.rerank)

    if args.kind == "podcast":
        script = generate_podcast_script(
            args.root, sources=sources, model=args.model, memory=memory
        )
        out = root_path / ".htc" / "studio" / "overview-script.md"
        print(f"podcast script -> {out}")
        audio_out = root_path / ".htc" / "studio" / "overview.mp3"
        audio_path = render_audio(script, audio_out)
        if audio_path:
            print(f"audio -> {audio_path}")
        else:
            print("script only (set HTC_TTS_* to render audio)")
    else:
        generate_diagram(
            args.root, sources=sources, model=args.model, kind=args.kind, memory=memory
        )
        out = root_path / ".htc" / "studio" / "architecture.mmd.md"
        print(f"{args.kind} diagram -> {out}")

    history.record_run(args.root, "studio", {"kind": args.kind})
    telemetry.track(
        "command_run",
        {
            "command": "studio",
            "provider": _safe_provider(),
            "duration_bucket": telemetry.bucket_duration(time.perf_counter() - start),
        },
    )
    return 0


def _cmd_wiki(args: argparse.Namespace) -> int:
    from .world_model.build import build_memory
    from .world_model.wiki import add_wiki_to_memory, build_wiki, write_wiki_files

    start = time.perf_counter()
    root_path = Path(args.root).expanduser().resolve()
    topics = [t.strip() for t in args.topics.split(",") if t.strip()] if args.topics else None
    print(f"building wiki for {args.root} ...")
    store = build_memory(FilesystemAdapter(str(root_path)).sources(), root_path)
    if args.rerank != "none":
        from .world_model.rerank import RerankingMemoryStore, get_reranker

        store = RerankingMemoryStore(store, get_reranker(args.rerank))
    pages = build_wiki(store, topics=topics, model=args.model)
    if not pages:
        print("no topics inferred and none provided — nothing to build.")
        return 1
    add_wiki_to_memory(pages, store)
    written = write_wiki_files(pages, root_path)
    for path in written:
        print(f"  wiki page -> {path}")
    print(f"{len(pages)} page(s) written and indexed into memory (kind=wiki, now searchable).")
    history.record_run(args.root, "wiki", {"num_pages": len(pages)})
    telemetry.track(
        "command_run",
        {
            "command": "wiki",
            "provider": _safe_provider(),
            "duration_bucket": telemetry.bucket_duration(time.perf_counter() - start),
        },
    )
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    from .world_model.graph import build_graph, graph_json_path
    from .world_model.ingest import ingest_sources

    start = time.perf_counter()
    root_path = Path(args.root).expanduser().resolve()
    sources = _parse_sources(args.sources) or FilesystemAdapter(str(root_path)).sources()
    print(f"building knowledge graph for {args.root} (zero LLM calls) ...")
    chunks = ingest_sources(sources, root=root_path).all_chunks()
    if not chunks:
        print("no chunks ingested — nothing to build a graph from.")
        return 1

    graph = build_graph(chunks, root_path)
    entities, relations = graph.entities(), graph.relations()
    print(f"{len(entities)} entities, {len(relations)} relations")
    for entity in graph.top_entities(10):
        print(f"  {entity.kind:<11} {entity.name} ({entity.mentions})")
    print(f"graph -> {graph_json_path(root_path)}")

    if args.mermaid:
        mmd_path = root_path / ".htc" / "graph" / "graph.mmd.md"
        mmd_path.write_text(graph.to_mermaid())
        print(f"mermaid -> {mmd_path}")

    history.record_run(
        args.root, "graph", {"num_entities": len(entities), "num_relations": len(relations)}
    )
    telemetry.track(
        "command_run",
        {
            "command": "graph",
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


def _study_path(root: str, name: str) -> Path:
    return Path(root).expanduser().resolve() / HTC_DIR / "study" / name


_BANK_TEMPLATE = """[
  {
    "id": "task-001",
    "prompt": "Describe the prompt exactly as a real teammate would give it.",
    "category": "example",
    "provenance": "the real event this task is drawn from, e.g. a JIRA ticket or PR"
  }
]
"""


def _cmd_study_init(args: argparse.Namespace) -> int:
    out = Path(args.output) if args.output else _study_path(args.root, "bank.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and not args.force:
        print(f"{out} already exists — use --force to overwrite.")
        return 1
    out.write_text(_BANK_TEMPLATE)
    print(f"task-bank template -> {out}")
    print("edit it with your real tasks (>= 8 recommended), then:")
    print(f"  htc study run --root {args.root} --bank {out} --agents <agents.json>")
    return 0


def _cmd_study_run(args: argparse.Namespace) -> int:
    import json

    from .study import AgentSpec, load_bank, make_grading_sheet, run_attempts, save_grading_sheet

    bank = load_bank(args.bank)
    agents_data = json.loads(Path(args.agents).expanduser().read_text())
    agents = [AgentSpec(**item) for item in agents_data]
    print(f"running {len(bank)} task(s) against {len(agents)} agent(s) ...")
    attempts = run_attempts(bank, agents, args.root)
    sheet = make_grading_sheet(attempts, bank, seed=args.seed)
    out = Path(args.output) if args.output else _study_path(args.root, "sheet.json")
    save_grading_sheet(sheet, out)
    print(f"{len(attempts)} attempt(s) -> {out}")
    print("next: have a human grade each blind_id 0-4 (see docs), then:")
    print(
        f"  htc study analyze --sheet {out} --grades <grader-scores.json> --scores <agent-scores.json>"
    )
    return 0


def _cmd_study_analyze(args: argparse.Namespace) -> int:
    import json

    from .study import ingest_grades, load_grading_sheet, study_verdict

    sheet = load_grading_sheet(args.sheet)
    grades = []
    for grades_path in args.grades:
        data = json.loads(Path(grades_path).expanduser().read_text())
        grades.extend(ingest_grades(sheet, data["scores"], data["grader_id"]))
    score_by_agent = json.loads(Path(args.scores).expanduser().read_text())
    verdict = study_verdict(score_by_agent, grades, n=args.n, seed=args.seed)
    print(json.dumps(verdict, indent=2))
    if verdict["passed"]:
        print(f"\nPASSED: rho={verdict['rho']} CI=({verdict['ci_lo']}, {verdict['ci_hi']})")
    else:
        print(f"\nNOT PASSED: rho={verdict['rho']} CI=({verdict['ci_lo']}, {verdict['ci_hi']})")
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
    p_hand.add_argument(
        "--rerank",
        default="none",
        choices=("none", "zerank", "cohere", "voyage", "local"),
        help="rerank retrieved chunks for precision before writing (default none; "
        "BYO key via env — see HTC_RERANKER)",
    )
    p_hand.set_defaults(func=_cmd_handbook)

    p_studio = sub.add_parser(
        "studio", help="render human-facing artifacts (diagram/mindmap/podcast) from memory"
    )
    p_studio.add_argument("--root", default=".", help="path to the company repo")
    p_studio.add_argument(
        "--kind",
        default="diagram",
        choices=("diagram", "mindmap", "podcast"),
        help="artifact to render (default diagram)",
    )
    p_studio.add_argument("--model", default=None, help="generation model override")
    p_studio.add_argument(
        "--sources",
        action="append",
        default=None,
        help="extra source to ingest, 'path' or 'path:kind' (repeatable; "
        "defaults to the repo via the filesystem adapter)",
    )
    p_studio.add_argument(
        "--rerank",
        default="none",
        choices=("none", "zerank", "cohere", "voyage", "local"),
        help="rerank retrieved chunks for precision before generating (default none; "
        "BYO key via env — see HTC_RERANKER)",
    )
    p_studio.set_defaults(func=_cmd_studio)

    p_wiki = sub.add_parser(
        "wiki", help="synthesize a grounded LLM-wiki from memory, indexed back into it"
    )
    p_wiki.add_argument("--root", default=".", help="path to the company repo")
    p_wiki.add_argument("--model", default=None, help="generation model override")
    p_wiki.add_argument(
        "--topics", default=None, help="comma-separated topics (default: inferred from memory)"
    )
    p_wiki.add_argument(
        "--rerank",
        default="none",
        choices=("none", "zerank", "cohere", "voyage", "local"),
        help="rerank retrieved chunks for precision before synthesizing (default none; "
        "BYO key via env — see HTC_RERANKER)",
    )
    p_wiki.set_defaults(func=_cmd_wiki)

    p_graph = sub.add_parser("graph", help="build the self-wiring knowledge graph (zero LLM calls)")
    p_graph.add_argument("--root", default=".", help="path to the company repo")
    p_graph.add_argument(
        "--sources",
        action="append",
        default=None,
        help="extra source to ingest, 'path' or 'path:kind' (repeatable; "
        "defaults to the repo via the filesystem adapter)",
    )
    p_graph.add_argument(
        "--mermaid",
        action="store_true",
        help="also write a Mermaid diagram to .htc/graph/graph.mmd.md",
    )
    p_graph.set_defaults(func=_cmd_graph)

    p_hist = sub.add_parser("history", help="show run history and score trend")
    p_hist.add_argument("--root", required=True, help="path to the company repo")
    p_hist.set_defaults(func=_cmd_history)

    p_study = sub.add_parser(
        "study",
        help="correlation study: validate Agent-Ready score against real task performance",
    )
    study_sub = p_study.add_subparsers(dest="_study_cmd", required=True)

    p_study_init = study_sub.add_parser("init", help="scaffold a task-bank template")
    p_study_init.add_argument("--root", required=True, help="path to the company repo")
    p_study_init.add_argument(
        "-o", "--output", default=None, help="output path (default .htc/study/bank.json)"
    )
    p_study_init.add_argument(
        "--force", action="store_true", help="overwrite an existing task bank"
    )
    p_study_init.set_defaults(func=_cmd_study_init)

    p_study_run = study_sub.add_parser(
        "run", help="run the agent ladder against the task bank, emit a blind grading sheet"
    )
    p_study_run.add_argument("--root", required=True, help="path to the company repo")
    p_study_run.add_argument("--bank", required=True, help="task-bank JSON (see `study init`)")
    p_study_run.add_argument(
        "--agents",
        required=True,
        help="JSON file: list of {id, label, agent_cmd} — the agent ladder to run "
        "(agent_cmd null/omitted for human rungs, which are skipped here)",
    )
    p_study_run.add_argument(
        "--seed", type=int, default=0, help="deterministic shuffle seed for the grading sheet"
    )
    p_study_run.add_argument(
        "-o", "--output", default=None, help="grading sheet path (default .htc/study/sheet.json)"
    )
    p_study_run.set_defaults(func=_cmd_study_run)

    p_study_analyze = study_sub.add_parser(
        "analyze", help="compute the Spearman correlation verdict from filled-in grades"
    )
    p_study_analyze.add_argument("--sheet", required=True, help="grading sheet from `study run`")
    p_study_analyze.add_argument(
        "--grades",
        required=True,
        action="append",
        help="grader score file, JSON {grader_id, scores: {blind_id: score}} (repeatable, "
        "one per human grader)",
    )
    p_study_analyze.add_argument(
        "--scores",
        required=True,
        help="JSON {agent_id: Agent-Ready score} from `htc eval` runs on each ladder rung",
    )
    p_study_analyze.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p_study_analyze.add_argument("--n", type=int, default=1000, help="bootstrap resamples")
    p_study_analyze.set_defaults(func=_cmd_study_analyze)

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

"""Run an agent against the goldens and judge every answer.

Two agent-under-test modes:

- builtin: a minimal tool-use loop — the model gets the twin's read-only tools
  (search_files / grep_content / read_file) and answers each golden. Measures
  model + tools with no external harness.
- cmd: pipe each question to any CLI agent (e.g. `claude -p`) that explores the
  repo itself. Measures your actual agent.

Judging is an LLM call per item with the reference answer + required artifact;
citing the right file is part of the rubric.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..adapters.base import EXCLUDED_DIRS
from ..adapters.filesystem import FilesystemAdapter
from ..goldens.generator import Golden
from ..llm import LLMError, ToolSpec, complete, extract_json, judge_model
from ..sandbox import SandboxConfig, run_in_sandbox

MAX_AGENT_TURNS = 8
MAX_TOOL_CHARS = 12_000
VERDICTS = ("correct", "partial", "wrong")

AGENT_SYSTEM = """You are being evaluated on how well you know a specific repository. \
Use the tools to investigate, then answer the question concisely and cite the \
file path(s) your answer is based on. If you cannot determine the answer, say so."""

JUDGE_SYSTEM = """You grade an AI agent's answer about a specific repository.

You get: the question, the reference answer, the artifact (file the correct
answer must be grounded in), and the agent's answer.

Verdicts:
- "correct": factually matches the reference answer's substance. Wording may differ.
- "partial": right direction but missing the key specific(s), or correct fact \
without grounding in the right artifact.
- "wrong": factually wrong, hallucinated, or no answer.

Reply with ONLY JSON: {"verdict": "correct"|"partial"|"wrong", "reason": "<one line>"}"""


@dataclass(frozen=True)
class ItemResult:
    golden: Golden
    agent_answer: str
    verdict: str
    reason: str


@dataclass(frozen=True)
class EvalResult:
    root: str
    agent: str
    items: list[ItemResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Agent-Ready score 0-100. correct=1, partial=0.5, wrong=0."""
        if not self.items:
            return 0.0
        points = sum(
            1.0 if item.verdict == "correct" else 0.5 if item.verdict == "partial" else 0.0
            for item in self.items
        )
        return round(100 * points / len(self.items), 1)

    def by_category(self) -> dict[str, float]:
        buckets: dict[str, list[ItemResult]] = {}
        for item in self.items:
            buckets.setdefault(item.golden.category, []).append(item)
        out: dict[str, float] = {}
        for category, items in sorted(buckets.items()):
            points = sum(
                1.0 if i.verdict == "correct" else 0.5 if i.verdict == "partial" else 0.0
                for i in items
            )
            out[category] = round(100 * points / len(items), 1)
        return out


def _twin_tools(root: Path) -> tuple[list[ToolSpec], dict[str, Callable[..., str]]]:
    """The twin's read-only tool surface, exposed in-process for the builtin agent."""
    adapter = FilesystemAdapter(str(root))
    repo = Path(adapter.workspace_spec().repo_path).resolve()

    def _safe(rel: str) -> Path:
        target = (repo / rel).resolve()
        if target != repo and repo not in target.parents:
            raise ValueError(f"path escapes repo root: {rel}")
        if set(target.relative_to(repo).parts) & EXCLUDED_DIRS:
            raise ValueError(f"path is inside an excluded directory: {rel}")
        return target

    def search_files(name: str) -> str:
        hits = [
            str(p.relative_to(repo))
            for p in repo.rglob("*")
            if p.is_file()
            and name.lower() in p.name.lower()
            and not (set(p.relative_to(repo).parts) & EXCLUDED_DIRS)
        ]
        return "\n".join(hits[:200]) or "(no matches)"

    def grep_content(pattern: str) -> str:
        try:
            out = subprocess.run(
                [
                    "grep",
                    "-rn",
                    "-I",
                    *(f"--exclude-dir={d}" for d in EXCLUDED_DIRS),
                    "-m",
                    "5",
                    "--",
                    pattern,
                    str(repo),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            return "(search timed out)"
        text = out.stdout.replace(str(repo) + "/", "")
        return text[:MAX_TOOL_CHARS] or "(no matches)"

    def read_file(path: str) -> str:
        target = _safe(path)
        if not target.is_file():
            return f"(not a file: {path})"
        text = target.read_text(errors="replace")
        return text[:MAX_TOOL_CHARS] + ("\n...[truncated]" if len(text) > MAX_TOOL_CHARS else "")

    specs = [
        ToolSpec(
            name="search_files",
            description="Find files whose name contains the given string (case-insensitive).",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        ToolSpec(
            name="grep_content",
            description="Search file contents for a pattern; returns matching lines with paths.",
            input_schema={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        ),
        ToolSpec(
            name="read_file",
            description="Read a file by repo-relative path.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]
    return specs, {
        "search_files": search_files,
        "grep_content": grep_content,
        "read_file": read_file,
    }


def _builtin_agent(root: Path, question: str, model: str | None) -> str:
    """Minimal tool-use loop over the twin tools; returns the final text answer."""
    specs, impls = _twin_tools(root)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    for _ in range(MAX_AGENT_TURNS):
        response = complete(AGENT_SYSTEM, messages, model=model, tools=specs)
        if not response.wants_tools:
            return response.text.strip()
        messages.append({"role": "assistant", "content": response.raw_content})
        results = []
        for call in response.tool_calls:
            impl = impls.get(call.name)
            try:
                output = impl(**call.arguments) if impl else f"(unknown tool {call.name})"
            except Exception as err:  # tool errors go back to the model, not up the stack
                output = f"(tool error: {err})"
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
        messages.append({"role": "user", "content": results})
    return response.text.strip() or "(agent ran out of turns without answering)"


def _cmd_agent(agent_cmd: str, root: Path, question: str) -> str:
    """Pipe the question to an external agent CLI run inside the repo."""
    try:
        out = subprocess.run(
            agent_cmd,
            shell=True,
            input=question,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(root),
        )
    except subprocess.TimeoutExpired:
        return "(agent command timed out)"
    answer = out.stdout.strip()
    return answer or f"(agent command produced no output; stderr: {out.stderr[:300]})"


def _judge(golden: Golden, answer: str) -> tuple[str, str]:
    prompt = (
        f"Question: {golden.question}\n"
        f"Reference answer: {golden.answer}\n"
        f"Required artifact: {golden.artifact}\n\n"
        f"Agent's answer:\n{answer[:4000]}"
    )
    response = complete(JUDGE_SYSTEM, [{"role": "user", "content": prompt}], model=judge_model())
    try:
        data = extract_json(response.text)
        if not isinstance(data, dict):
            return "wrong", "judge reply was not a JSON object"
        verdict = str(data.get("verdict", "")).lower()
        reason = str(data.get("reason", "")).strip()
    except LLMError:
        return "wrong", "judge reply was unparseable"
    if verdict not in VERDICTS:
        return "wrong", f"judge returned invalid verdict '{verdict}'"
    return verdict, reason


def _answer_and_judge(
    root_path: Path,
    golden: Golden,
    agent_cmd: str | None,
    model: str | None,
    sandbox: SandboxConfig | None,
) -> ItemResult:
    """One golden's full pipeline: agent answer -> judge. Raises LLMError on failure."""
    if agent_cmd and sandbox is not None:
        answer = run_in_sandbox(agent_cmd, root_path, golden.question, sandbox)
    elif agent_cmd:
        answer = _cmd_agent(agent_cmd, root_path, golden.question)
    else:
        answer = _builtin_agent(root_path, golden.question, model)
    verdict, reason = _judge(golden, answer)
    return ItemResult(golden=golden, agent_answer=answer, verdict=verdict, reason=reason)


def run_eval(
    root: str | Path,
    goldens: list[Golden],
    *,
    agent_cmd: str | None = None,
    model: str | None = None,
    on_item: Callable[[int, int, ItemResult], None] | None = None,
    concurrency: int = 4,
    sandbox: SandboxConfig | None = None,
) -> EvalResult:
    """Answer every golden with the chosen agent, judge each, return results.

    Goldens are farmed out to a thread pool (the work is I/O-bound HTTP calls
    to the provider). Result order need not match input order; a per-item
    provider failure is skipped, not fatal to the run.

    When `sandbox` is set and `agent_cmd` is set, the agent command runs
    inside a Docker container (see `htc.sandbox`) instead of directly on the
    host. `sandbox` has no effect on the builtin agent.
    """
    root_path = Path(root).expanduser().resolve()
    agent_label = agent_cmd or f"builtin:{model or 'default'}"
    items: list[ItemResult] = []
    skipped = 0
    total = len(goldens)
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_answer_and_judge, root_path, golden, agent_cmd, model, sandbox): index
            for index, golden in enumerate(goldens, start=1)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                item = future.result()
            except Exception as err:  # noqa: BLE001 - per-item guard, must not kill the run
                with print_lock:
                    skipped += 1
                    print(f"  [{index}/{total}] SKIPPED (provider error: {err})", file=sys.stderr)
                continue
            items.append(item)
            if on_item:
                with print_lock:
                    on_item(index, total, item)
    if skipped:
        print(f"  {skipped} item(s) skipped due to provider errors", file=sys.stderr)
    return EvalResult(root=str(root_path), agent=agent_label, items=items)


def save_results(result: EvalResult, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": result.root,
        "agent": result.agent,
        "score": result.score,
        "by_category": result.by_category(),
        "items": [asdict(item) for item in result.items],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def load_results(path: str | Path) -> EvalResult:
    data = json.loads(Path(path).expanduser().read_text())
    items = [
        ItemResult(
            golden=Golden(**item["golden"]),
            agent_answer=item["agent_answer"],
            verdict=item["verdict"],
            reason=item["reason"],
        )
        for item in data.get("items", [])
    ]
    return EvalResult(root=data.get("root", ""), agent=data.get("agent", ""), items=items)

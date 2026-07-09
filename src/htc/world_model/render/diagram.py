"""Diagram rendering: a Mermaid diagram grounded in world-model memory.

One of the two "output/studio" halves (the other is `podcast.py`): everything
here renders FROM the same memory the handbook reads from — no separate
knowledge source, no invented components.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...llm import complete
from ..memory import SearchResult
from ..retrieval import RetrievalPipeline

STUDIO_DIR = ".htc/studio"
DIAGRAM_FILENAME = "architecture.mmd.md"
SEARCH_K = 8
MAX_ATTEMPTS = 2

NO_DIAGRAM = "Could not generate a diagram from the ingested sources."

DIAGRAM_SYSTEM = """You generate a Mermaid diagram for a specific project. You \
are given retrieved source chunks — real file or document contents, each \
labeled with its source path.

Rules:
- Output ONLY a fenced Mermaid code block: ```mermaid ... ``` and nothing else \
(no prose before or after the fence).
- Every node and edge must reflect a real component, module, or data flow \
shown in the provided chunks. Never invent parts, services, or connections \
that aren't supported by the sources.
- Prefer a `flowchart` (or `graph`) diagram for "architecture", or a `mindmap` \
diagram when asked for a mindmap.
- Keep it readable: concise node labels, no more than ~20 nodes."""

_KIND_QUERIES: dict[str, str] = {
    "architecture": "architecture modules data flow components how system is put together",
    "mindmap": "project overview purpose main components concepts",
}

_MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_EDGE_RE = re.compile(r"-->|---|==>|-\.->|-\.-")


def _prompt(kind: str, results: list[SearchResult]) -> str:
    lines = [f"Generate a Mermaid {kind} diagram grounded in the following sources."]
    for result in results:
        lines.append(f"=== SOURCE: {result.chunk.source_path} ===\n{result.chunk.text}")
    return "\n\n".join(lines)


def _looks_like_diagram(body: str, kind: str) -> bool:
    if not body.strip():
        return False
    if kind == "mindmap":
        lines = [line for line in body.splitlines() if line.strip()]
        return len(lines) >= 2
    return bool(_EDGE_RE.search(body))


def _extract_fence(text: str, kind: str) -> str | None:
    match = _MERMAID_FENCE_RE.search(text)
    if not match:
        return None
    body = match.group(1).strip()
    if not _looks_like_diagram(body, kind):
        return None
    return match.group(0).strip()


def generate_diagram(
    root: str | Path,
    pipeline: RetrievalPipeline,
    *,
    model: str | None = None,
    kind: str = "architecture",
) -> str:
    """Retrieve the most relevant chunks for `kind` and ask the model to draw a
    Mermaid diagram grounded in them. Writes `<root>/.htc/studio/architecture.mmd.md`
    and returns the markdown body.

    Validates the reply contains a ```mermaid fence with at least one edge/node;
    retries once on an invalid reply before giving up with a clear note.
    """
    if kind not in _KIND_QUERIES:
        raise ValueError(f"unknown diagram kind '{kind}' (architecture | mindmap)")
    root_path = Path(root).expanduser().resolve()

    results = pipeline.retrieve(_KIND_QUERIES[kind], SEARCH_K)
    title = f"# {kind.title()} Diagram\n\n"
    if not results:
        markdown = title + NO_DIAGRAM + "\n"
    else:
        prompt = _prompt(kind, results)
        fence = None
        for _attempt in range(MAX_ATTEMPTS):
            response = complete(DIAGRAM_SYSTEM, [{"role": "user", "content": prompt}], model=model)
            fence = _extract_fence(response.text, kind)
            if fence:
                break
        markdown = title + (fence if fence else NO_DIAGRAM) + "\n"

    out = root_path / STUDIO_DIR / DIAGRAM_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    return markdown

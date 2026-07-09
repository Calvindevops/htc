"""Generate a structured onboarding handbook grounded in ingested memory.

Unlike `onboard/writer.py` (which drafts a gap-only context pack from eval
misses), this builds a full handbook from scratch: for each fixed section, it
retrieves the most relevant chunks from the world-model memory (Phase 2) and
asks the model to write that section grounded ONLY in what it retrieved,
citing source paths inline. Sections with no grounding say so rather than
inventing content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..adapters.base import Source
from ..adapters.filesystem import FilesystemAdapter
from ..llm import complete
from ..world_model.build import build_memory
from ..world_model.memory import MemoryStore, SearchResult
from ..world_model.query import retrieve_with_transform

DRAFT_NAME = "HANDBOOK.md.htc-draft"
SEARCH_K = 5
NO_GROUNDING = "Not enough information in the ingested sources."

HANDBOOK_SYSTEM = """You write one section of an employee/agent onboarding \
handbook for a specific project. You are given retrieved source chunks — real \
file or document contents, each labeled with its source path.

Rules:
- Be strictly factual: write ONLY what the provided chunks support. Never \
invent facts, tools, or behavior that isn't shown.
- Cite the source path inline for every claim, e.g. "... does X (see path/to/file)".
- No fluff, no generic advice, no filler — every sentence must teach a real \
fact about THIS project, aimed at a new team member or agent joining cold.
- If the chunks below do not contain enough information to write this section, \
reply with exactly this sentence and nothing else: Not enough information in \
the ingested sources.

Output ONLY the section body (markdown prose/lists). Do not repeat the section \
heading — the caller adds it."""


@dataclass(frozen=True)
class Section:
    """One handbook section: its heading, the memory query that grounds it,
    and the guidance line telling the model what the section should cover."""

    heading: str
    query: str
    guidance: str


SECTIONS: tuple[Section, ...] = (
    Section(
        heading="Overview",
        query="project overview purpose what this system does",
        guidance="Explain what this project is and what problem it solves.",
    ),
    Section(
        heading="Architecture",
        query="architecture modules data flow components how system is put together",
        guidance="Explain how the system is put together: modules, data flow, "
        "and how the pieces connect.",
    ),
    Section(
        heading="Configuration & Setup",
        query="configuration environment variables setup install dependencies",
        guidance="Explain how to configure and set up the project: dependencies, "
        "environment variables, install steps.",
    ),
    Section(
        heading="Operations",
        query="run deploy test build operations commands tooling",
        guidance="Explain how to run, test, build, and deploy the system day to day.",
    ),
    Section(
        heading="Conventions & House Style",
        query="conventions house style patterns code style testing conventions",
        guidance="Explain the coding conventions, patterns, and house style a "
        "new contributor must follow.",
    ),
    Section(
        heading="Current State / Notable Decisions",
        query="decisions tradeoffs current state known limitations design choices",
        guidance="Explain notable design decisions, tradeoffs, and the current "
        "state of the project.",
    ),
)


def _section_prompt(section: Section, results: list[SearchResult]) -> str:
    lines = [f"Write the '{section.heading}' section. {section.guidance}"]
    for result in results:
        lines.append(f"=== SOURCE: {result.chunk.source_path} ===\n{result.chunk.text}")
    return "\n\n".join(lines)


def _write_section(
    section: Section, store: MemoryStore, model: str | None, query_transform: str | None
) -> str:
    results = retrieve_with_transform(
        store, section.query, k=SEARCH_K, strategy=query_transform, model=model
    )
    if not results:
        return NO_GROUNDING
    prompt = _section_prompt(section, results)
    response = complete(HANDBOOK_SYSTEM, [{"role": "user", "content": prompt}], model=model)
    body = response.text.strip()
    if not body:
        raise RuntimeError(f"handbook model reply was empty for section '{section.heading}'")
    return body


def generate_handbook(
    root: str | Path,
    *,
    sources: list[Source] | None = None,
    model: str | None = None,
    memory: MemoryStore | None = None,
    query_transform: str | None = None,
) -> str:
    """Build (or reuse) the world-model memory over `root` and `sources`, then
    write a structured onboarding handbook, one section at a time, grounded in
    the most relevant retrieved chunks per section.

    `query_transform` (default: "none", see `htc.world_model.query`) opts
    into an LLM-driven retrieval-query transform per section; "none" makes no
    extra LLM call and retrieves exactly as before.

    Writes `<root>/HANDBOOK.md.htc-draft` (never touches an existing
    `HANDBOOK.md`) and returns the markdown body.
    """
    root_path = Path(root).expanduser().resolve()
    store = memory or build_memory(
        sources or FilesystemAdapter(str(root_path)).sources(), root_path
    )

    parts = [f"# {root_path.name} Handbook\n"]
    for section in SECTIONS:
        body = _write_section(section, store, model, query_transform)
        parts.append(f"## {section.heading}\n\n{body}\n")
    markdown = "\n".join(parts) + "\n"

    draft = root_path / DRAFT_NAME
    draft.write_text(markdown)
    return markdown

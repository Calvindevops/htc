"""LLM-wiki synthesis — the complement to raw memory.

Where the memory (Phase 2) stores raw ingested chunks, the wiki derives a
short set of topics/entities, retrieves the chunks that ground each one, and
asks the model to synthesize a concise, deduplicated, strictly grounded
knowledge page per topic — citing every source path it drew from. Pages are
written back into the SAME memory as `kind="wiki"` chunks (via
`add_wiki_to_memory`), so retrieval surfaces both the raw source material and
the synthesized page, and to `<root>/.htc/wiki/<slug>.md` for human browsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ...llm import complete, extract_json
from ..ingest.model import SourceChunk, chunk_id
from ..memory import MemoryStore, SearchResult

WIKI_DIR = ".htc/wiki"
SEARCH_K = 8
MAX_TOPICS = 8

UNKNOWN = "unknown"

WIKI_SYSTEM = """You write one page of an internal knowledge wiki for a \
specific project/company, synthesized from retrieved source chunks — real \
file or document contents, each labeled with its source path.

Rules:
- Be strictly factual and GROUNDED: write only what the provided chunks \
support. Never invent facts.
- SYNTHESIZE, don't copy: merge, deduplicate, and organize what the chunks \
say into one coherent page — don't just concatenate excerpts.
- Cite the source path inline for every claim, e.g. "... does X (see path/to/file)".
- If the chunks below do not ground this topic at all, reply with exactly \
this word and nothing else: unknown

Output ONLY the page body (markdown prose/lists). Do not repeat the title —
the caller adds it."""

TOPICS_SYSTEM = """You read source chunks from a project/company and propose \
a short list of distinct knowledge-wiki topics/entities worth their own page \
(e.g. "Authentication", "Deployment Pipeline", "Billing Model"). Reply with \
ONLY a JSON array of strings, nothing else. Propose at most 8 topics."""


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "page"


@dataclass(frozen=True)
class WikiPage:
    """One synthesized, grounded wiki page."""

    title: str
    body_md: str
    source_paths: list[str]


def _infer_topics(memory: MemoryStore, model: str | None) -> list[str]:
    """Ask the model to propose topics from a broad sample of the memory."""
    results = memory.search("overview architecture components purpose", k=SEARCH_K)
    if not results:
        return []
    lines = ["Propose wiki topics for this project, from the following sources."]
    for result in results:
        lines.append(f"=== SOURCE: {result.chunk.source_path} ===\n{result.chunk.text}")
    response = complete(
        TOPICS_SYSTEM, [{"role": "user", "content": "\n\n".join(lines)}], model=model
    )
    try:
        topics = extract_json(response.text)
    except Exception:
        return []
    if not isinstance(topics, list):
        return []
    return [str(topic).strip() for topic in topics[:MAX_TOPICS] if str(topic).strip()]


def _page_prompt(title: str, results: list[SearchResult]) -> str:
    lines = [f"Write the wiki page for '{title}'."]
    for result in results:
        lines.append(f"=== SOURCE: {result.chunk.source_path} ===\n{result.chunk.text}")
    return "\n\n".join(lines)


def _build_page(title: str, memory: MemoryStore, model: str | None) -> WikiPage:
    results = memory.search(title, k=SEARCH_K)
    if not results:
        return WikiPage(title=title, body_md=UNKNOWN, source_paths=[])
    response = complete(
        WIKI_SYSTEM, [{"role": "user", "content": _page_prompt(title, results)}], model=model
    )
    body = response.text.strip()
    if not body:
        raise RuntimeError(f"wiki model reply was empty for topic '{title}'")
    if body.lower() == UNKNOWN:
        return WikiPage(title=title, body_md=UNKNOWN, source_paths=[])
    source_paths = sorted({result.chunk.source_path for result in results})
    return WikiPage(title=title, body_md=body, source_paths=source_paths)


def build_wiki(
    memory: MemoryStore,
    topics: list[str] | None = None,
    model: str | None = None,
) -> list[WikiPage]:
    """Derive (or accept) topics, retrieve grounding chunks per topic, and ask
    the model to synthesize one concise, cited page per topic."""
    resolved_topics = topics if topics else _infer_topics(memory, model)
    return [_build_page(title, memory, model) for title in resolved_topics]


def add_wiki_to_memory(pages: list[WikiPage], memory: MemoryStore) -> None:
    """Store synthesized wiki pages BACK into `memory` as `kind="wiki"`
    chunks, so retrieval surfaces both raw source chunks and synthesized
    pages — the wiki complements the memory, it doesn't replace it."""
    chunks = []
    for page in pages:
        source_path = f"{WIKI_DIR}/{_slug(page.title)}.md"
        text = f"# {page.title}\n\n{page.body_md}"
        chunks.append(
            SourceChunk(
                id=chunk_id(source_path, 0),
                source_path=source_path,
                kind="wiki",
                text=text,
                start_char=0,
                end_char=len(text),
            )
        )
    if chunks:
        memory.add_chunks(chunks)


def write_wiki_files(pages: list[WikiPage], root: str | Path) -> list[Path]:
    """Write each page to `<root>/.htc/wiki/<slug>.md` for human browsing."""
    root_path = Path(root).expanduser().resolve()
    out_dir = root_path / WIKI_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page in pages:
        out = out_dir / f"{_slug(page.title)}.md"
        sources = (
            "\n".join(f"- {path}" for path in page.source_paths)
            if page.source_paths
            else "- (none — ungrounded)"
        )
        out.write_text(f"# {page.title}\n\n{page.body_md}\n\n## Sources\n\n{sources}\n")
        written.append(out)
    return written

"""Podcast script: a 2-host audio-overview script grounded in world-model
memory (the open-notebooklm/podcastfy pattern), plus an optional, env-gated
TTS render stub.

The script is the deliverable; audio is opt-in and never a hard dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...adapters.base import Source
from ...adapters.filesystem import FilesystemAdapter
from ...llm import complete
from ..build import build_memory
from ..memory import MemoryStore

STUDIO_DIR = ".htc/studio"
SCRIPT_FILENAME = "overview-script.md"
SEARCH_K = 8
QUERY = "project overview purpose architecture what this system does for a new team member"

NO_SCRIPT = "Not enough information in the ingested sources to write an overview script."

PODCAST_SYSTEM = """You write a 2-host audio-overview script (like a podcast \
"overview" episode) introducing a codebase/project to a new team member. You \
are given retrieved source chunks — real file or document contents, each \
labeled with its source path.

Rules:
- Two speakers only, labeled "Host A:" and "Host B:" at the start of each line.
- Conversational and natural — questions, reactions, brief back-and-forth —
not a lecture read aloud.
- Strictly grounded: only explain what the provided chunks support. Never
invent facts, tools, or behavior that isn't shown. It's fine to mention a
source path in passing if useful, but don't force citations into every line.
- Target a natural spoken length of roughly 2-4 minutes (roughly 300-600
words total).
- Output ONLY the script (speaker-labeled lines). No preamble, no headings."""


def _prompt(root_name: str, results) -> str:
    lines = [
        f"Write a 2-host audio-overview script introducing the project "
        f"'{root_name}' to a new team member."
    ]
    for result in results:
        lines.append(f"=== SOURCE: {result.chunk.source_path} ===\n{result.chunk.text}")
    return "\n\n".join(lines)


def generate_podcast_script(
    root: str | Path,
    *,
    sources: list[Source] | None = None,
    model: str | None = None,
    memory: MemoryStore | None = None,
) -> str:
    """Retrieve the most relevant chunks and ask the model to write a
    2-host audio-overview script grounded in them. Writes
    `<root>/.htc/studio/overview-script.md` and returns the script text.
    """
    root_path = Path(root).expanduser().resolve()
    store = memory or build_memory(
        sources or FilesystemAdapter(str(root_path)).sources(), root_path
    )

    results = store.search(QUERY, k=SEARCH_K)
    if not results:
        script = NO_SCRIPT
    else:
        prompt = _prompt(root_path.name, results)
        response = complete(PODCAST_SYSTEM, [{"role": "user", "content": prompt}], model=model)
        script = response.text.strip()
        if not script:
            raise RuntimeError("podcast model reply was empty")

    out = root_path / STUDIO_DIR / SCRIPT_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script + "\n")
    return script


def render_audio(script: str, out_path: str | Path) -> Path | None:
    """Render `script` to audio via a configured TTS provider.

    Only runs if `HTC_TTS_PROVIDER` (and its key, `HTC_TTS_API_KEY`) is set in
    the environment. HTC never hard-depends on any TTS library: this stub
    stays a no-op — returning `None` — until a provider is wired up, so
    callers should treat `None` as "script only (set HTC_TTS_* to render
    audio)".
    """
    provider = os.environ.get("HTC_TTS_PROVIDER")
    key = os.environ.get("HTC_TTS_API_KEY")
    if not provider or not key:
        return None
    # Extend here per provider (lazy-imported, so no TTS lib is a hard
    # dependency of HTC), e.g.:
    #   if provider == "elevenlabs":
    #       from elevenlabs import generate  # lazy import
    #       ...
    raise NotImplementedError(
        f"HTC_TTS_PROVIDER={provider!r} is configured but no TTS backend is "
        "wired up yet (script-only rendering works without this)."
    )

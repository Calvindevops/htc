"""Render/studio: human-facing Orientation artifacts rendered FROM the same
world-model memory the handbook reads from — the "output" half of ingest-in,
artifacts-out."""

from .diagram import NO_DIAGRAM, generate_diagram
from .podcast import NO_SCRIPT, generate_podcast_script, render_audio

__all__ = [
    "NO_DIAGRAM",
    "NO_SCRIPT",
    "generate_diagram",
    "generate_podcast_script",
    "render_audio",
]

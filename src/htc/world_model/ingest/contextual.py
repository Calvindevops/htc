"""Contextual Retrieval (Anthropic-style): before embedding a chunk, ask the
model for a short blurb that situates it within its whole source document,
and embed that blurb PREPENDED to the chunk — while the original chunk text
stays untouched for display/citation.

One `complete()` call per chunk, so this is opt-in (see
`build_memory(contextual=True)` / `HTC_CONTEXTUAL_RETRIEVAL=1`); a bad/empty
model reply just leaves the chunk unchanged (`embed_text` stays `None`),
never crashes ingestion.
"""

from __future__ import annotations

from dataclasses import replace

from ...llm import LLMError, complete
from .model import SourceChunk

_CONTEXT_SYSTEM = """You situate a chunk of text within its whole source \
document to improve search retrieval of that chunk. Given the whole document \
and one chunk taken from it, write a short, succinct context (1-2 sentences) \
that situates the chunk within the overall document.

Answer with ONLY the succinct context, nothing else."""

_USER_TEMPLATE = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>"""


def contextualize_chunks(
    chunks: list[SourceChunk], full_texts: dict[str, str], model: str | None = None
) -> list[SourceChunk]:
    """Return `chunks` with `embed_text` set to the model's context blurb
    prepended to the original text, for every chunk whose source has a full
    document text in `full_texts` and yields a usable context. Chunks with no
    matching full text, or a bad/empty model reply, come back unchanged
    (`embed_text` stays `None`, `text` is never touched)."""
    contextualized: list[SourceChunk] = []
    for chunk in chunks:
        document = full_texts.get(chunk.source_path)
        if not document:
            contextualized.append(chunk)
            continue
        context = _situate(document, chunk.text, model)
        if not context:
            contextualized.append(chunk)
            continue
        contextualized.append(replace(chunk, embed_text=f"{context}\n\n{chunk.text}"))
    return contextualized


def _situate(document: str, chunk_text: str, model: str | None) -> str | None:
    prompt = _USER_TEMPLATE.format(document=document, chunk=chunk_text)
    try:
        response = complete(
            _CONTEXT_SYSTEM, [{"role": "user", "content": prompt}], model=model, max_tokens=300
        )
    except LLMError:
        return None
    context = response.text.strip()
    return context or None

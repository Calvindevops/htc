"""Paragraph/sentence-aware chunker — splits ingested text into citable
`SourceChunk`s without ever cutting mid-word.

Splitting strategy, in order:
1. Blank-line paragraph boundaries.
2. Paragraphs over `max_chars` are regrouped by sentence boundary.
3. A single sentence still over `max_chars` is hard-wrapped on whitespace.

Offsets (`start_char`/`end_char`) are computed from spans into the original
text (never via re-searching for a substring), so they stay correct even
when the same line of text repeats elsewhere in the source.
"""

from __future__ import annotations

import re

from .model import SourceChunk, chunk_id

MAX_CHARS = 2_000

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Non-blank paragraph spans, split on blank lines."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Non-blank sentence spans within `text[start:end]`, in absolute offsets."""
    segment = text[start:end]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(segment):
        spans.append((start + cursor, start + match.start()))
        cursor = match.end()
    spans.append((start + cursor, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _word_wrap_spans(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Hard-wrap `text[start:end]` on whitespace so no piece exceeds `max_chars`."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + max_chars, end)
        if limit < end:
            back = text.rfind(" ", cursor, limit)
            if back > cursor:
                limit = back
        spans.append((cursor, limit))
        cursor = limit
        while cursor < end and text[cursor] == " ":
            cursor += 1
    return spans


def _regroup_by_sentence(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split an oversized paragraph span into <=max_chars spans on sentence
    boundaries, falling back to whitespace-wrapping any oversized sentence."""
    spans: list[tuple[int, int]] = []
    group_start: int | None = None
    group_end: int | None = None
    for s_start, s_end in _sentence_spans(text, start, end):
        if s_end - s_start > max_chars:
            if group_start is not None:
                spans.append((group_start, group_end))
                group_start = group_end = None
            spans.extend(_word_wrap_spans(text, s_start, s_end, max_chars))
            continue
        if group_start is None:
            group_start, group_end = s_start, s_end
        elif s_end - group_start > max_chars:
            spans.append((group_start, group_end))
            group_start, group_end = s_start, s_end
        else:
            group_end = s_end
    if group_start is not None:
        spans.append((group_start, group_end))
    return spans


def chunk_text(
    text: str, source_path: str, kind: str, max_chars: int = MAX_CHARS
) -> list[SourceChunk]:
    """Split `text` into coherent `SourceChunk`s no larger than `max_chars`.

    Paragraphs (blank-line delimited) that fit stay whole; oversized ones are
    regrouped by sentence boundary, and an oversized sentence is hard-wrapped
    on whitespace — never mid-word.
    """
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    for start, end in _paragraph_spans(text):
        if end - start <= max_chars:
            spans.append((start, end))
        else:
            spans.extend(_regroup_by_sentence(text, start, end, max_chars))

    chunks: list[SourceChunk] = []
    for start, end in spans:
        piece = text[start:end]
        if not piece.strip():
            continue
        chunks.append(
            SourceChunk(
                id=chunk_id(source_path, start),
                source_path=source_path,
                kind=kind,
                text=piece,
                start_char=start,
                end_char=end,
            )
        )
    return chunks

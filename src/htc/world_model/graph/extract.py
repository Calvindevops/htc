"""Deterministic entity + relation extraction — ZERO LLM calls.

Mirrors gBrain's "self-wiring knowledge graph" approach: cheap, fast,
seedless, regex/frequency-based heuristics over `SourceChunk` text. No
network calls, no randomness — the same input always yields the identical
graph (see `tests/test_graph.py::test_extraction_is_deterministic`).

Do NOT import `htc.llm` (or anything that calls an LLM) from this module.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import PurePosixPath

from ..ingest.model import SourceChunk
from .model import Entity, Relation

# --- Extraction patterns --------------------------------------------------

_DEF_RE = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_FUNCTION_RE = re.compile(r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)")
_CAMEL_RE = re.compile(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)+\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")  # mirrors the BM25 tokenizer in memory/local.py

_STOPWORDS = frozenset("""
    a an the and or but if then else for while with without to from of in on at
    by is are was were be been being this that these those it its as not no
    do does did doing have has had having will would shall should can could may
    might must so than too very s t just don now here there all any both each
    few more most other some such only own same you your yours we our ours they
    their them he she his her i my me
    """.split())

# --- Caps (keep the graph from exploding on a big repo) --------------------
# Every cap below is a top-N-by-mentions cutoff; when a cap trims the list,
# `_cap` prints how many entries were dropped so the cull is never silent.

MIN_SYMBOL_MENTIONS = 2  # generic camelCase/snake_case identifiers (not def/class/function)
MIN_TERM_MENTIONS = 2
MAX_SYMBOLS = 200
MAX_TERMS = 150
MAX_PROPER_NOUNS = 100
MAX_FILES = 300
MAX_MODULES = 150
MAX_RELATIONS = 4000


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def _entity_id(kind: str, name: str) -> str:
    return f"{kind}:{_slug(name)}"


def _module_of(source_path: str) -> str | None:
    parent = PurePosixPath(source_path).parent.as_posix()
    return None if parent in ("", ".") else parent


def _cap(items: list, limit: int, label: str) -> list:
    """Keep the top `limit` items (already sorted best-first); print what was
    dropped so a big repo's cull is visible, not silent."""
    if len(items) <= limit:
        return items
    dropped = len(items) - limit
    print(f"htc graph: capped {label} at {limit} (dropped {dropped} lower-salience entries)")
    return items[:limit]


def _tf_idf_terms(chunks: list[SourceChunk]) -> list[tuple[str, int]]:
    """Top salient unigrams/bigrams by tf * idf; returns (term, total mentions)."""
    n_chunks = len(chunks)
    term_total: Counter[str] = Counter()
    term_doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        tokens = [t for t in _TOKEN_RE.findall(chunk.text.lower()) if t not in _STOPWORDS]
        seen_in_chunk: set[str] = set()
        for token in tokens:
            term_total[token] += 1
            seen_in_chunk.add(token)
        for left, right in zip(tokens, tokens[1:]):
            bigram = f"{left} {right}"
            term_total[bigram] += 1
            seen_in_chunk.add(bigram)
        for term in seen_in_chunk:
            term_doc_freq[term] += 1

    scored: list[tuple[float, str]] = []
    for term, total in term_total.items():
        if total < MIN_TERM_MENTIONS:
            continue
        idf = math.log((n_chunks + 1) / (term_doc_freq[term] + 1)) + 1.0
        scored.append((total * idf, term))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [(term, term_total[term]) for _, term in scored]


def _chunk_symbol_names(text: str) -> tuple[set[str], set[str]]:
    """Return (all_identifier_names_in_chunk, definite_names) — the second
    from an explicit `def`/`class`/`function` declaration."""
    definite = {m.group(1) for m in _DEF_RE.finditer(text)} | {
        m.group(1) for m in _FUNCTION_RE.finditer(text)
    }
    all_names = (
        definite
        | {m.group(0) for m in _CAMEL_RE.finditer(text)}
        | {m.group(0) for m in _SNAKE_RE.finditer(text)}
    )
    return all_names, definite


def extract(chunks: list[SourceChunk]) -> tuple[list[Entity], list[Relation]]:
    """Extract entities + relations from `chunks`. Deterministic: sorts
    `chunks` by (source_path, start_char) first so extraction never depends
    on caller-supplied ordering."""
    ordered = sorted(chunks, key=lambda c: (c.source_path, c.start_char))

    symbol_counts: Counter[str] = Counter()
    definite_symbols: set[str] = set()
    counts_per_file: dict[str, Counter[str]] = {}
    proper_noun_counts: Counter[str] = Counter()
    file_chunk_counts: Counter[str] = Counter()
    # Per-chunk presence sets, reused below to build co-occurrence relations
    # without re-scanning chunk text.
    chunk_symbol_names: list[set[str]] = []
    chunk_proper_nouns: list[set[str]] = []
    chunk_symbol_home: dict[str, str] = {}  # symbol name -> first file that defines it

    for chunk in ordered:
        file_chunk_counts[chunk.source_path] += 1
        per_file = counts_per_file.setdefault(chunk.source_path, Counter())

        names_here, definite_here = _chunk_symbol_names(chunk.text)
        for name in names_here:
            symbol_counts[name] += 1
            per_file[name] += 1
        definite_symbols |= definite_here
        for name in definite_here:
            chunk_symbol_home.setdefault(name, chunk.source_path)
        chunk_symbol_names.append(names_here)

        nouns_here = {" ".join(m.group(0).split()) for m in _PROPER_NOUN_RE.finditer(chunk.text)}
        for noun in nouns_here:
            proper_noun_counts[noun] += 1
        chunk_proper_nouns.append(nouns_here)

    # --- symbol entities ---
    symbol_names = definite_symbols | {
        name for name, count in symbol_counts.items() if count >= MIN_SYMBOL_MENTIONS
    }
    symbols_sorted = sorted(symbol_names, key=lambda n: (-symbol_counts[n], n))
    symbols_sorted = _cap(symbols_sorted, MAX_SYMBOLS, "symbols")

    # --- file / module entities ---
    files_sorted = sorted(file_chunk_counts, key=lambda p: (-file_chunk_counts[p], p))
    files_sorted = _cap(files_sorted, MAX_FILES, "files")

    module_counts: Counter[str] = Counter()
    for path, count in file_chunk_counts.items():
        module = _module_of(path)
        if module:
            module_counts[module] += count
    modules_sorted = sorted(module_counts, key=lambda m: (-module_counts[m], m))
    modules_sorted = _cap(modules_sorted, MAX_MODULES, "modules")

    # --- proper nouns ---
    nouns_sorted = sorted(proper_noun_counts, key=lambda n: (-proper_noun_counts[n], n))
    nouns_sorted = _cap(nouns_sorted, MAX_PROPER_NOUNS, "proper nouns")

    # --- salient terms ---
    terms = _cap(_tf_idf_terms(ordered), MAX_TERMS, "terms")

    entities: list[Entity] = []
    entities.extend(
        Entity(id=_entity_id("symbol", n), name=n, kind="symbol", mentions=symbol_counts[n])
        for n in symbols_sorted
    )
    entities.extend(
        Entity(id=_entity_id("file", p), name=p, kind="file", mentions=file_chunk_counts[p])
        for p in files_sorted
    )
    entities.extend(
        Entity(id=_entity_id("module", m), name=m, kind="module", mentions=module_counts[m])
        for m in modules_sorted
    )
    entities.extend(
        Entity(
            id=_entity_id("proper_noun", n),
            name=n,
            kind="proper_noun",
            mentions=proper_noun_counts[n],
        )
        for n in nouns_sorted
    )
    entities.extend(
        Entity(id=_entity_id("term", term), name=term, kind="term", mentions=count)
        for term, count in terms
    )

    relations = _build_relations(
        ordered,
        entities,
        chunk_symbol_home,
        counts_per_file,
        chunk_symbol_names,
        chunk_proper_nouns,
    )
    return entities, relations


def _build_relations(
    chunks: list[SourceChunk],
    entities: list[Entity],
    symbol_home_file: dict[str, str],
    counts_per_file: dict[str, Counter[str]],
    chunk_symbol_names: list[set[str]],
    chunk_proper_nouns: list[set[str]],
) -> list[Relation]:
    symbol_entities = {e.name: e.id for e in entities if e.kind == "symbol"}
    file_entities = {e.name: e.id for e in entities if e.kind == "file"}
    proper_noun_entities = {e.name: e.id for e in entities if e.kind == "proper_noun"}

    relations: list[Relation] = []

    # `contains`: the file whose def/class/function first declared the symbol.
    for name, home in symbol_home_file.items():
        symbol_id = symbol_entities.get(name)
        file_id = file_entities.get(home)
        if symbol_id and file_id:
            weight = max(counts_per_file.get(home, Counter()).get(name, 1), 1)
            relations.append(Relation(file_id, symbol_id, "contains", weight=weight))

    # `references`: a symbol's name appearing in a file other than its home.
    for name, home in symbol_home_file.items():
        symbol_id = symbol_entities.get(name)
        if not symbol_id:
            continue
        for path, counts in counts_per_file.items():
            if path == home:
                continue
            weight = counts.get(name, 0)
            file_id = file_entities.get(path)
            if weight > 0 and file_id:
                relations.append(Relation(symbol_id, file_id, "references", weight=weight))

    # `co_occurs`: entities (symbols/files/proper nouns) sharing a chunk,
    # weighted by the number of chunks they co-occur in. Terms are excluded
    # here — nearly every chunk shares common terms, which would blow up the
    # pair count; terms still surface via `subgraph_for_query` name matching.
    co_occur_weights: dict[tuple[str, str], int] = {}
    for chunk, names_here, nouns_here in zip(chunks, chunk_symbol_names, chunk_proper_nouns):
        present: set[str] = set()
        file_id = file_entities.get(chunk.source_path)
        if file_id:
            present.add(file_id)
        present.update(symbol_entities[n] for n in names_here if n in symbol_entities)
        present.update(proper_noun_entities[n] for n in nouns_here if n in proper_noun_entities)
        ids = sorted(present)
        for idx, a in enumerate(ids):
            for b in ids[idx + 1 :]:
                key = (a, b)
                co_occur_weights[key] = co_occur_weights.get(key, 0) + 1

    co_occur_pairs = sorted(co_occur_weights, key=lambda k: (-co_occur_weights[k], k[0], k[1]))
    relations.extend(
        Relation(a, b, "co_occurs", weight=co_occur_weights[(a, b)]) for a, b in co_occur_pairs
    )

    relations.sort(key=lambda r: (-r.weight, r.kind, r.source_id, r.target_id))
    return _cap(relations, MAX_RELATIONS, "relations")

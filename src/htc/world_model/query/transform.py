"""Query-transformation strategies — gBrain's "expansion model": transform a
raw query into better retrieval queries BEFORE hitting the store. Each
strategy is one `complete()` call, parsed via `extract_json`; a bad/empty/
unparseable model reply always falls back to the original query, never
crashes retrieval, and never introduces unseeded randomness.
"""

from __future__ import annotations

from ...llm import LLMError, complete, extract_json

_EXPAND_SYSTEM = """You rewrite a search query into alternative phrasings to \
improve retrieval recall over a hybrid BM25+semantic index. Given the user's \
query, produce 2-3 alternative phrasings — synonyms, rephrasings, related \
terms a relevant document might use — PLUS the original query itself.

Reply with ONLY a JSON array of strings, no prose outside it, e.g.:
["original query", "alternative phrasing one", "alternative phrasing two"]
"""

_HYDE_SYSTEM = """You write a short hypothetical document that would perfectly \
answer the user's query (HyDE: Hypothetical Document Embeddings) — the kind \
of passage a real source would contain, written in a neutral, factual tone, \
even if you are not certain of the real answer. This text is embedded and \
used to retrieve real documents; it is never shown to the user.

Reply with ONLY a JSON object of this exact shape, no prose outside it:
{"hypothetical_document": "..."}
"""

_DECOMPOSE_SYSTEM = """You break a query into atomic sub-questions for \
retrieval. If the query is already a single, atomic question, return it \
unchanged as the only item. If it has multiple parts (e.g. "X and Y", \
"compare A vs B", "how does X work and why"), split it into separate, \
self-contained sub-questions.

Reply with ONLY a JSON array of strings, no prose outside it, e.g.:
["sub-question one", "sub-question two"]
"""

_MULTI_QUERY_SYSTEM = """You generate diverse reformulations of a search \
query to improve retrieval recall over a hybrid BM25+semantic index — each \
reformulation should approach the query from a different angle (different \
wording, different level of specificity, different implied intent), while \
preserving its meaning.

Reply with ONLY a JSON array of exactly {n} strings, no prose outside it."""


def _list_of_str(data: object) -> list[str] | None:
    """Return `data` as a non-empty list of non-empty strings, or `None` if
    it isn't a usable list (guards against non-list/empty model replies)."""
    if not isinstance(data, list) or not data:
        return None
    cleaned = [str(item).strip() for item in data if str(item).strip()]
    return cleaned or None


def _call_for_list(system: str, query: str, model: str | None) -> list[str] | None:
    try:
        response = complete(system, [{"role": "user", "content": query}], model=model)
        data = extract_json(response.text)
    except LLMError:
        return None
    return _list_of_str(data)


def expand(query: str, model: str | None = None) -> list[str]:
    """2-3 alternative phrasings / synonym-expanded variants of `query`,
    including the original. Falls back to `[query]` on a bad/empty reply."""
    variants = _call_for_list(_EXPAND_SYSTEM, query, model)
    return variants if variants else [query]


def hyde(query: str, model: str | None = None) -> str:
    """Generate a hypothetical answer/document for `query` (HyDE) — the text
    to embed/retrieve with INSTEAD of the bare query. Falls back to `query`
    itself on a bad/empty reply."""
    try:
        response = complete(_HYDE_SYSTEM, [{"role": "user", "content": query}], model=model)
        data = extract_json(response.text)
    except LLMError:
        return query
    if not isinstance(data, dict):
        return query
    document = str(data.get("hypothetical_document", "")).strip()
    return document if document else query


def decompose(query: str, model: str | None = None) -> list[str]:
    """Break a complex/multi-part query into atomic sub-questions. Falls back
    to `[query]` if it's already atomic or the reply is bad/empty."""
    sub_questions = _call_for_list(_DECOMPOSE_SYSTEM, query, model)
    return sub_questions if sub_questions else [query]


def multi_query(query: str, model: str | None = None, n: int = 3) -> list[str]:
    """`n` diverse reformulations of `query`. Falls back to `[query]` on a
    bad/empty reply."""
    system = _MULTI_QUERY_SYSTEM.format(n=n)
    variants = _call_for_list(system, query, model)
    return variants if variants else [query]

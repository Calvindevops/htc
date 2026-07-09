"""answer_question — the capstone of the memory system: gBrain's "gives you
the answer, not raw search results."

Retrieves across the whole memory (hybrid BM25+semantic, optionally reranked
for precision, optionally boosted/enriched by the knowledge graph), then asks
the model ONE question in ONE call: answer strictly from the retrieved
context, cite every source path used, and explicitly call out the GAPS —
what the context does not cover or only thinly supports. If retrieval finds
nothing, no LLM call is made at all; the caller gets an honest "don't know".
"""

from __future__ import annotations

from ...llm import LLMError, complete, extract_json
from ..memory.store import SearchResult
from ..retrieval import RetrievalPipeline
from .model import Answer

SEARCH_K = 8
CONFIDENCE_LEVELS = ("high", "medium", "low")

ANSWER_SYSTEM = """You answer a question about a project/company using ONLY \
the retrieved context below — real file/document contents, each labeled with \
its source path, plus (optionally) related knowledge-graph entities.

Rules:
- Be strictly GROUNDED: answer only from the provided context. Never invent \
or assume facts the context doesn't support.
- Cite the source path inline for every claim, e.g. "... does X (see path/to/file)".
- Then list GAPS: what the context does NOT answer, where it's thin, or \
where you're uncertain — be specific and honest. An empty list is fine if \
the context fully covers the question.
- Set confidence to "high" if the context clearly and fully answers the \
question, "medium" if it partially answers it, "low" if the context barely \
touches it or doesn't answer it at all.
- If the context does not answer the question at all, still reply with the \
same JSON shape: explain in answer_md that the memory has no relevant \
information, leave citations empty, describe what's missing in gaps, and \
set confidence to "low".

Reply with ONLY a JSON object, no prose outside it, of this exact shape:
{"answer_md": "...", "citations": ["path/one", "path/two"], "gaps": ["..."], "confidence": "high|medium|low"}
"""

ASSESS_SYSTEM = """You judge whether the retrieved context below is enough to fully \
answer a question, or whether another, more targeted search is needed first.

Rules:
- Set "sufficient" to true if the context clearly and fully answers the question.
- Set "sufficient" to false only if the context is missing something important that \
a different, more targeted search query could plausibly find.
- If insufficient, set "followup_query" to ONE focused search query (not a rephrasing \
of the original question) that targets the specific gap. If sufficient, or you can't \
think of a useful followup, set "followup_query" to null.

Reply with ONLY a JSON object, no prose outside it, of this exact shape:
{"sufficient": true|false, "followup_query": "..."|null}
"""


def _context_block(results: list[SearchResult], graph_entities: list) -> str:
    lines = [f"=== SOURCE: {r.chunk.source_path} ===\n{r.chunk.text}" for r in results]
    if graph_entities:
        names = ", ".join(f"{e.name} ({e.kind})" for e in graph_entities)
        lines.append(f"=== KNOWLEDGE GRAPH: entities related to the query ===\n{names}")
    return "\n\n".join(lines)


def _fallback(question: str, reason: str) -> Answer:
    return Answer(
        question=question,
        answer_md=f"The memory has no relevant information to answer this question ({reason}).",
        citations=[],
        gaps=[reason],
        confidence="low",
    )


def _assess_sufficiency(
    query: str,
    results: list[SearchResult],
    graph_entities: list,
    model: str | None,
) -> tuple[bool, str | None]:
    """One `complete()` call: is `results` sufficient to answer `query`, or
    should a followup search run first? A non-dict/unparseable reply is
    treated as sufficient (stop) so a bad reply never causes an infinite
    loop."""
    prompt = f"Question: {query}\n\n{_context_block(results, graph_entities)}"
    response = complete(ASSESS_SYSTEM, [{"role": "user", "content": prompt}], model=model)
    try:
        data = extract_json(response.text)
    except LLMError:
        return True, None
    if not isinstance(data, dict):
        return True, None
    sufficient = bool(data.get("sufficient", True))
    followup_query = data.get("followup_query")
    if not isinstance(followup_query, str) or not followup_query.strip():
        followup_query = None
    return sufficient, followup_query


def _merge_chunks(pool: list[SearchResult], new_results: list[SearchResult]) -> list[SearchResult]:
    """Append `new_results` to `pool`, deduped by chunk id (first occurrence
    wins), preserving `pool`'s existing order."""
    seen = {r.chunk.id for r in pool}
    merged = list(pool)
    for r in new_results:
        if r.chunk.id not in seen:
            merged.append(r)
            seen.add(r.chunk.id)
    return merged


def _iterate_retrieval(
    query: str,
    results: list[SearchResult],
    pipeline: RetrievalPipeline,
    *,
    model: str | None,
    k: int,
    max_rounds: int,
) -> list[SearchResult]:
    """Agentic retrieval loop: assess whether `results` sufficiently answers
    `query`; if not, retrieve for the assessor's followup query and merge the
    new chunks into the pool (deduped by chunk id). Stops when the assessor
    says sufficient, returns no followup, or `max_rounds` retrieval rounds
    (the initial one plus followups) have been used."""
    graph = pipeline.graph
    graph_entities = graph.subgraph_for_query(query, k=k) if graph is not None else []
    rounds_used = 1
    while rounds_used < max_rounds:
        sufficient, followup_query = _assess_sufficiency(query, results, graph_entities, model)
        if sufficient or not followup_query:
            break
        followup_results = pipeline.retrieve(followup_query, k)
        results = _merge_chunks(results, followup_results)
        rounds_used += 1
    return results


def answer_question(
    query: str,
    pipeline: RetrievalPipeline,
    *,
    model: str | None = None,
    k: int = SEARCH_K,
    iterative: bool = False,
    max_rounds: int = 3,
) -> Answer:
    """Retrieve via `pipeline` (hybrid + whatever rerank/graph/query-transform
    the pipeline was configured with), then ONE LLM call to synthesize a
    grounded, cited `Answer` with an explicit gap analysis. If retrieval is
    empty, returns a low-confidence "don't know" `Answer` with no LLM call at
    all.

    `iterative` (default: False, zero extra LLM calls, current behavior
    unchanged) opts into agentic/iterative retrieval: after the initial
    retrieval, the model assesses whether the accumulated context is
    sufficient; if not, it proposes a followup query, which is retrieved and
    merged into the pool (deduped by chunk id), up to `max_rounds` retrieval
    rounds total, before the final synthesis call below runs."""
    results = pipeline.retrieve(query, k)

    if iterative and results:
        results = _iterate_retrieval(
            query,
            results,
            pipeline,
            model=model,
            k=k,
            max_rounds=max_rounds,
        )

    if not results:
        return _fallback(query, "no relevant chunks were found in memory for this question")

    graph = pipeline.graph
    graph_entities = graph.subgraph_for_query(query, k=k) if graph is not None else []
    prompt = f"Question: {query}\n\n{_context_block(results, graph_entities)}"
    response = complete(ANSWER_SYSTEM, [{"role": "user", "content": prompt}], model=model)

    try:
        data = extract_json(response.text)
    except LLMError:
        return _fallback(query, "the model reply was unparseable")
    if not isinstance(data, dict):
        return _fallback(query, "the model reply was not a JSON object")

    answer_md = str(data.get("answer_md", "")).strip()
    if not answer_md:
        return _fallback(query, "the model returned an empty answer")

    citations_raw = data.get("citations", [])
    gaps_raw = data.get("gaps", [])
    citations = [str(c) for c in citations_raw] if isinstance(citations_raw, list) else []
    gaps = [str(g) for g in gaps_raw] if isinstance(gaps_raw, list) else []
    confidence = str(data.get("confidence", "low")).lower()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = "low"

    return Answer(
        question=query,
        answer_md=answer_md,
        citations=citations,
        gaps=gaps,
        confidence=confidence,
    )

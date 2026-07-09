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
from ..graph.graph import KnowledgeGraph
from ..memory.store import MemoryStore, SearchResult
from ..query import retrieve_with_transform
from ..rerank.base import Reranker
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


def answer_question(
    query: str,
    memory: MemoryStore,
    *,
    graph: KnowledgeGraph | None = None,
    reranker: Reranker | None = None,
    model: str | None = None,
    k: int = SEARCH_K,
    query_transform: str | None = None,
) -> Answer:
    """Retrieve across `memory` (hybrid + optional rerank + optional graph +
    optional query transformation), then ONE LLM call to synthesize a
    grounded, cited `Answer` with an explicit gap analysis. If retrieval is
    empty, returns a low-confidence "don't know" `Answer` with no LLM call at
    all.

    `query_transform` (default: "none", see `htc.world_model.query`) opts
    into an LLM-driven retrieval-query transform ("expand"/"hyde"/
    "decompose"/"multi") before the search above; "none" makes no extra LLM
    call and behaves exactly as before."""
    results = retrieve_with_transform(
        memory, query, k=k, strategy=query_transform, model=model, graph=graph, reranker=reranker
    )
    if not results:
        return _fallback(query, "no relevant chunks were found in memory for this question")

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

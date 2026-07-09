# Memory backends

HTC ingests sources into a persistent, cited memory (`MemoryStore`), then
retrieves from it for goldens, handbook, and studio generation. The backend
is pluggable — pick the one that fits, or bring your own.

```bash
export HTC_MEMORY_BACKEND=local          # default, zero-config
export HTC_MEMORY_BACKEND=gbrain         # external gBrain CLI
export HTC_MEMORY_BACKEND=supermemory    # hosted Supermemory API
export HTC_MEMORY_BACKEND=mypkg.mymod.MyStore   # custom
htc handbook --root .
```

Selection order: an explicit `backend` argument to `get_memory_store()`
(for programmatic callers), else the `HTC_MEMORY_BACKEND` env var, else
`"local"`.

## 1. `local` (default — hybrid BM25 + optional semantic)

No external service, no required ML dependency. Works offline out of the
box: retrieval is BM25 keyword scoring over ingested chunks, persisted to
`<root>/.htc/memory/chunks.jsonl`.

If you configure an OpenAI-compatible embeddings endpoint, retrieval becomes
**hybrid**: chunks and queries are embedded, cosine similarity ranks them
semantically, and the semantic ranking is fused with the BM25 ranking via
Reciprocal Rank Fusion (RRF, k=60). This catches relevant chunks that share
no keywords with the query. With no embedding endpoint configured, behavior
is unchanged — BM25 only.

| Var | Meaning |
|---|---|
| `HTC_EMBED_BASE_URL` | Base URL of an OpenAI-compatible `/embeddings` endpoint. |
| `HTC_EMBED_API_KEY` | API key for that endpoint. |
| `HTC_EMBED_MODEL` | Embedding model name. |

All three must be set for hybrid retrieval to activate. Embeddings are
computed once per chunk (on `add_chunks`) and persisted alongside the chunks
in a parallel `<root>/.htc/memory/embeddings.jsonl`, so they aren't
recomputed on later runs.

```bash
export HTC_EMBED_BASE_URL=https://api.openai.com/v1
export HTC_EMBED_API_KEY=sk-...
export HTC_EMBED_MODEL=text-embedding-3-small
```

## 2. `gbrain` (optional — external gBrain CLI)

Adapter over the maintainer's [gBrain](https://github.com/garrytan/gbrain)
CLI. Requires `gbrain` on `PATH`. Useful if you already run gBrain and want
HTC to read/write its memory instead of maintaining a separate store.

## 3. `supermemory` (optional — hosted or self-hosted Supermemory)

Adapter over the [Supermemory](https://supermemory.ai) API.

| Var | Meaning |
|---|---|
| `SUPERMEMORY_API_KEY` | required — get one at supermemory.ai |
| `SUPERMEMORY_BASE_URL` | optional — point at a self-hosted instance instead of the hosted API |

Network calls are best-effort: a failed add or search degrades gracefully
(logs a warning, skips the write / returns no results) rather than crashing
HTC.

## 4. Custom (bring your own)

Anything that isn't `local`, `gbrain`, or `supermemory` is treated as a
dotted class path (`module.submodule.ClassName`) and imported dynamically.
Your class must satisfy the `MemoryStore` protocol:

```python
class MyStore:
    def add_chunks(self, chunks: list[SourceChunk]) -> None: ...
    def search(self, query: str, k: int = 5) -> list[SearchResult]: ...
    def has_source(self, path: str) -> bool: ...
    def count(self) -> int: ...
```

Its constructor must accept either no arguments or a single `root` argument
(matching `LocalMemoryStore(root)`'s shape). Point HTC at it:

```bash
export HTC_MEMORY_BACKEND=mypkg.mymod.MyStore
htc handbook --root .
```

A bad module path, missing class, or a class that doesn't implement the
protocol all raise `MemoryBackendUnavailable` with a clear message rather
than failing silently.

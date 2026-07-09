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

## 1. `local` (default — hybrid BM25 + semantic, semantic on by default)

No external service required. Retrieval is BM25 keyword scoring over
ingested chunks, persisted to `<root>/.htc/memory/chunks.jsonl` — this part
always works offline with zero dependencies.

**Semantic search is the recommended default.** Install the `embed` extra:

```bash
pip install "htc[embed]"
```

With `fastembed` installed, chunks and queries are embedded locally on CPU
(`BAAI/bge-small-en-v1.5`, no network call, no configuration) and fused with
BM25 via Reciprocal Rank Fusion (RRF, k=60) — this catches relevant chunks
that share no keywords with the query. Zero config required: just install
the extra and retrieval becomes hybrid automatically.

If you'd rather use a remote embeddings endpoint (e.g. OpenAI, or your own),
set all three of these — they take precedence over the bundled local model:

| Var | Meaning |
|---|---|
| `HTC_EMBED_BASE_URL` | Base URL of an OpenAI-compatible `/embeddings` endpoint. |
| `HTC_EMBED_API_KEY` | API key for that endpoint. |
| `HTC_EMBED_MODEL` | Embedding model name. |

Precedence: (1) `HTC_EMBED_BASE_URL` + `HTC_EMBED_API_KEY` + `HTC_EMBED_MODEL`
if all three are set, (2) else the bundled `fastembed` model if installed,
(3) else BM25 only. Embeddings are computed once per chunk (on `add_chunks`)
and persisted alongside the chunks in a parallel
`<root>/.htc/memory/embeddings.jsonl`, so they aren't recomputed on later
runs.

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

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

**Semantic search is on by default.** The embedder is resolved in precedence
order, highest first:

1. **Cloud endpoint** — if `HTC_EMBED_BASE_URL` + `HTC_EMBED_API_KEY` +
   `HTC_EMBED_MODEL` are all set, that OpenAI-compatible `/embeddings`
   endpoint is used.
2. **Ollama (the recommended default, matching gBrain)** — if an Ollama
   server responds at `HTC_OLLAMA_URL` (default `http://localhost:11434`),
   it's used with model `nomic-embed-text` (override with `HTC_EMBED_MODEL`).
   Zero config beyond running Ollama:
   ```bash
   ollama pull nomic-embed-text
   ```
3. **`fastembed` (offline fallback)** — if neither of the above is available
   and the `embed` extra is installed (`pip install "htc[embed]"`), a small
   local CPU model (`BAAI/bge-small-en-v1.5`) embeds chunks and queries with
   no network call. This exists so HTC never hard-fails when Ollama isn't
   running and no cloud endpoint is set — it is NOT the recommended default,
   just the last-resort semantic option.
4. **BM25 only** — if none of the above is available.

Whichever embedder is chosen, its vectors are fused with BM25 via Reciprocal
Rank Fusion (RRF, k=60) — this catches relevant chunks that share no
keywords with the query. Embeddings are computed once per chunk (on
`add_chunks`) and persisted alongside the chunks in a parallel
`<root>/.htc/memory/embeddings.jsonl`, so they aren't recomputed on later
runs.

```bash
export HTC_EMBED_BASE_URL=https://api.openai.com/v1
export HTC_EMBED_API_KEY=sk-...
export HTC_EMBED_MODEL=text-embedding-3-small
```

### First-boot wizard

The first time HTC needs an embedder and finds no cloud endpoint, no
reachable Ollama, and no saved preference, it prints a short interactive
prompt (Supermemory-style) letting you pick: **(a)** Ollama local
[recommended], **(b)** a cloud endpoint, **(c)** `fastembed`, or **(d)** BM25
only. The answer is saved to `~/.htc/config.json` so you're only asked once.
Non-interactive runs (`HTC_EMBED_NONINTERACTIVE=1`, or no TTY — e.g. CI)
skip the prompt and silently fall through the precedence order above.

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

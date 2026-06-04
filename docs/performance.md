# Performance & scaling

How Engrava behaves as data grows, where the limits are, and the two levers that
matter most: the **vector backend** and **batched writes**. The numbers that
matter for *your* workload depend on corpus size, embedding dimension, query mix,
and hardware — measure on your own data rather than trusting a single headline
figure. This page explains *what* drives cost so you know what to measure.

For the dreaming *quality* benchmark (does consolidation help retrieval), see
[Benchmarks](benchmarks.md). For the hard platform constraints, see
[Known Limitations](known-limitations.md).

## Where the cost is

A query touches up to five signals; each scales differently:

| Signal | Cost driver | Scaling |
|---|---|---|
| **FTS5 / BM25** | SQLite's FTS5 inverted index | Sub-linear; scales well into large corpora. |
| **Vector** | The vector backend (see below) | **numpy: linear in #embeddings**; sqlite-vec: approximate, sub-linear. |
| **Recency** | A cheap per-candidate arithmetic decay | Negligible. |
| **Priority** | A per-candidate enum→multiplier lookup | Negligible. |
| **Graph** | 1-hop neighbour expansion over edges | Proportional to the fusion-pool size × average degree; **opt-in** (`graph_weight=0.0` makes zero graph queries). |

The dominant term at scale is almost always the **vector** signal, because the
default backend compares the query against every stored embedding.

## The brute-force ceiling (and how to pass it)

Without the `vec` extra, vector search is **brute-force cosine similarity in
Python**: every `search_similar` / `search_hybrid` query scans all embeddings.
This is simple and dependency-free, and works well up to roughly **100k
embeddings**. Past that, vector-query latency grows linearly and becomes the
bottleneck.

The fix is the **sqlite-vec** backend, which keeps an approximate-nearest-
neighbour `vec0` index for sub-linear vector queries. FTS5 scales independently
and usually needs no special handling.

> The ~100k figure is a rule of thumb, not a cliff — see
> [Known Limitations → sqlite-vec](known-limitations.md#sqlite-vec-pre-v1-status).
> Measure your own p95 query latency and switch when it stops meeting your budget.

## Switching to sqlite-vec (incl. migrating an existing database)

The migration is designed to be turnkey: your embeddings already live in the
`embedding` table, so switching backends only builds and backfills the ANN index
— you do **not** re-embed anything.

**1. Install the extra.**

```bash
pip install 'engrava[vec]'
```

**2. Set the backend in your config.**

```yaml
extensions:
  vector:
    backend: sqlite-vec      # default is "numpy"
    dimension: 384           # must match your embedding model
```

**3. Open the store with `from_config`.** On open, Engrava creates the `vec0`
virtual table and **backfills every existing embedding into it automatically**
(idempotent — safe to run repeatedly). From then on, new writes keep the index
in sync.

```python
from engrava import SqliteEngravaCore

# from_config wires the vector backend; the index is created and back-filled
# on open. A plain SqliteEngravaCore(conn) constructor stays on numpy.
async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    result = await store.search_similar(query_vector, top_k=5)
```

That's the whole migration — no manual re-index step, and no re-embedding,
because the vectors are reused from the existing `embedding` table.

**Important caveats.**

- **Use `from_config`.** Only the `from_config` path configures the vector
  backend. If you build the store directly with `SqliteEngravaCore(conn)`, it
  stays on the numpy backend regardless of the YAML.
- **Graceful fallback, not a hard error.** If the `sqlite-vec` package is missing
  or the extension can't load, Engrava logs a warning and **falls back to numpy**
  rather than crashing — so a "switch" that silently kept numpy usually means the
  extension didn't load.
- **macOS system SQLite blocks extensions.** The most common load failure is
  macOS's bundled SQLite, which disables extension loading. Install Python via
  Homebrew or pyenv (a full-featured SQLite build). See
  [Known Limitations → macOS](known-limitations.md#macos-sqlite-extension-loading).
- **Dimension must match.** The index is created for a fixed dimension; it must
  equal your embedding model's output. Mixing dimensions corrupts results (see
  [Embedding Dimension Consistency](known-limitations.md#embedding-dimension-consistency)).

## Write throughput and bulk ingest

By default each mutating call commits its own transaction. For a bulk load that
is the wrong granularity — one commit per row dominates wall-clock. Wrap the
batch in `suspend_auto_commit()`, which defers to a **single commit on success
and rolls the whole batch back on any error**:

```python
async def bulk_load(store, items):
    async with store.suspend_auto_commit():
        for item in items:
            await store.create_thought(item, deduplicate=True)
    return await store.count_thoughts()
```

- **`deduplicate=True`** collapses identical `content` into one thought (bumping
  `confirmation_count`) instead of inserting duplicate rows — cheaper storage and
  fewer embeddings to compute. (Note the persistence default is
  `deduplicate=False`; opt in per call.)
- **Keep each transaction short.** A long-running transaction blocks aiosqlite's
  background thread (see
  [Known Limitations → aiosqlite](known-limitations.md#aiosqlite-proxy-architecture)),
  so for very large imports, batch in chunks (e.g. a few thousand rows per
  `suspend_auto_commit()` block) rather than one giant transaction.
- **Embedding cost dominates a bulk load** when a provider is configured with
  `auto_embed=True`: each new thought is embedded on write. Pre-compute vectors
  and store them with `store_embedding(...)`, use a batching local provider, or
  import in chunks so the encoder isn't the bottleneck. See the
  [Embeddings guide](guides/embeddings.md).

A runnable end-to-end bulk-import example lives in the
[migration guide](guides/migrating-from-other-memory.md#bulk-import).

## Dreaming cost at scale

[Dreaming](dreaming.md) runs **off the hot path** — you invoke
`run_consolidation()` on your own cadence, so it never adds latency to CRUD or
search. Its own cost scales with the number of candidate thoughts and the
clustering algorithm:

- Run it **periodically**, not every turn (every N cycles, a cron job, or
  manually).
- `candidates_limit` caps how many thoughts are evaluated per pass — keep it
  bounded on large stores.
- Clustering has two backends via `extensions.dreaming.clustering_backend`
  (`"numpy"` default, or `"python"`); `numpy` is faster for the similarity math
  on larger candidate sets.
- The LPA clustering algorithm is `O(edges × iterations)`; the agglomerative
  algorithm operates over active thoughts — see [Dreaming](dreaming.md) for the
  algorithm tradeoffs.

## Checklist: scaling Engrava

1. **Past ~100k embeddings or missing your latency budget?** Switch to
   `sqlite-vec` (above).
2. **Bulk loading?** Batch writes with `suspend_auto_commit()` and consider
   `deduplicate=True`.
3. **Embedding is the bottleneck?** Use a batching provider or pre-compute
   vectors.
4. **Multi-tenant?** One database file per tenant via `EngravaManager` keeps each
   store smaller and independently lockable (see the
   [scoping section](guides/migrating-from-other-memory.md#filtering-scoping-and-multi-tenancy)).
5. **Dreaming heavy?** Cap `candidates_limit`, run it on a schedule, pick the
   right `clustering_backend`.

## See also

- [Known Limitations](known-limitations.md) — the brute-force ceiling, macOS, concurrency
- [Configuration](configuration.md) — the `extensions.vector` and dreaming knobs
- [Benchmarks](benchmarks.md) — the dreaming retrieval-quality benchmark
- [Embeddings](guides/embeddings.md) — provider choice and batching

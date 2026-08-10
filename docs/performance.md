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
| **Vector** | The vector backend (see below) | Linear in #embeddings for both backends; **sqlite-vec scans a compact `vec0` table with a much smaller constant factor** than the Python path. |
| **Recency** | A cheap per-candidate arithmetic decay | Negligible. |
| **Priority** | A per-candidate enum→multiplier lookup | Negligible. |
| **Graph** | 1-hop neighbour expansion over edges | Proportional to the fusion-pool size × average degree; **opt-in** (`graph_weight=0.0` makes zero graph queries). |

The dominant term at scale is almost always the **vector** signal, because both
backends compare the query against every stored embedding — the difference is how
efficiently they do it (see below).

## The brute-force ceiling (and how to pass it)

Without the `vec` extra, vector search is **brute-force cosine similarity in
Python**: every `search_similar` / `search_hybrid` query scans all embeddings.
This is simple and dependency-free, and works well up to roughly **100k
embeddings**. Past that, vector-query latency grows linearly and becomes the
bottleneck.

The fix is the **sqlite-vec** backend, which stores vectors in a dedicated,
compact `vec0` virtual table. In the pinned `sqlite-vec` 0.1.x line a `vec0`
query is still an **exhaustive k-nearest-neighbour scan** — not an approximate or
sub-linear index — but over a tightly packed, chunked columnar store, so it runs
with a far smaller constant factor (and lower memory overhead) than the Python
brute-force path. The practical effect is that the same corpus stays well under
your latency budget for much longer. FTS5 scales independently and usually needs
no special handling.

> The ~100k figure is a rule of thumb, not a cliff — see
> [Known Limitations → sqlite-vec](known-limitations.md#sqlite-vec-pre-v1-status).
> Measure your own p95 query latency and switch when it stops meeting your budget.

## Filtered search and the vector fast path

`search_hybrid()` / `recall()` with a `filters=` or `visibility=` argument scope
the ranked query (see [Scoped retrieval](search.md#scoped-retrieval)). There is a
performance consequence worth knowing.

**An active filter disables the `sqlite-vec` fast path.** With no filter, the
vector arm runs on the compiled `vec0` table. With a filter, the vector arm
instead runs an **exhaustive NumPy cosine over the rows that match the filter** —
because the filter is an arbitrary `metadata` predicate that `vec0` cannot apply
before it picks its `k` nearest neighbours. The filtered cost is:

- **~O(N_total)** for the scan that selects the eligible rows, **plus**
- **O(N_eligible × dims)** for the cosine over those eligible embeddings.

Note the first term: a selective filter shrinks the *cosine* work, but the
eligibility scan still visits the whole embedding-bearing population. Both the
filtered (NumPy) and unfiltered (`vec0`) paths are already **exhaustive** in the
pinned `sqlite-vec` 0.1.x line (see
[the brute-force ceiling](#the-brute-force-ceiling-and-how-to-pass-it)), so a
filter is a **constant-factor** slowdown, not a change in scaling.

**When it matters.** A *selective* filter (one project, one session, one user)
keeps the eligible set small — the filtered path stays acceptable, and can be
faster than scanning the whole `vec0` table when the filter is highly selective
(the eligibility scan still runs, so this is not guaranteed). The slow corner is
a **broad filter on a large
store** at the same time — for example `visibility=` admitting *public-or-mine*
where "mine" is most of the store, on a base approaching the brute-force ceiling.
If that is your workload, prefer **one store per tenant** via `EngravaManager`
(see [scoping & multi-tenancy](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy)):
each tenant's store is small, so the exhaustive scan stays small — and that is
real isolation, which a metadata filter is **not**.

## Switching to sqlite-vec (incl. migrating an existing database)

The migration is designed to be turnkey: your embeddings already live in the
`embedding` table, so switching backends only builds and backfills the `vec0`
vector table — you do **not** re-embed anything.

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

## SQLite tuning & hot-path indexes

Stores opened via `from_config` (and `EngravaManager`) come **pre-tuned** — you do
not configure any of this:

- **`synchronous=NORMAL`** — the documented-safe companion to WAL. It cuts an
  `fsync` from every commit (lower write latency) while staying durable across an
  application crash; only the most recent transactions are at risk on an OS crash
  or power loss. (See [Concurrency](concurrency.md#busy-timeout).)
- **`busy_timeout=5000`** — a second connection waits up to 5 s for a lock instead
  of failing immediately with `database is locked`.
- **Four hot-path indexes** back the equality filters and sort column that the
  common reads hit, turning what were full-table scans into index lookups:

  | Index | Column | Speeds up |
  |---|---|---|
  | `idx_edge_to_thought` | `edge(to_thought_id)` | `get_edges` (IN / BOTH) and the reflection-consolidation scan |
  | `idx_embedding_owner` | `embedding(owner_id)` | `get_embedding` lookups by thought |
  | `idx_thought_updated_cycle` | `thought(updated_cycle)` | `list_thoughts` recency ordering |
  | `idx_thought_type` | `thought(thought_type)` | `thought_type` equality filters |

The `0.4` schema also adds **valid-time indexes** so the
[bi-temporal](bitemporal.md) predicates are index-backed: `valid_from`, a
**partial** `valid_until` index (only non-`NULL` upper bounds are indexed — an open
upper bound is the common case and stays overhead-free), and a composite
`(valid_from, valid_until)` range index, on both the `thought` and `edge` tables.

The indexes are created idempotently (`CREATE INDEX IF NOT EXISTS`) and an existing
database gains them automatically through the additive `0.3 → 0.4` schema migration
— zero data loss, no manual step. See the [Upgrade Guide](upgrade.md#03---04).

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
   [scoping section](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy)).
5. **Dreaming heavy?** Cap `candidates_limit`, run it on a schedule, pick the
   right `clustering_backend`.

## See also

- [Known Limitations](known-limitations.md) — the brute-force ceiling, macOS, concurrency
- [Configuration](configuration.md) — the `extensions.vector` and dreaming knobs
- [Benchmarks](benchmarks.md) — the dreaming retrieval-quality benchmark
- [Embeddings](guides/embeddings.md) — provider choice and batching

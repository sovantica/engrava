# Known Limitations

This document covers platform-specific notes, constraints, and known issues.
For the consolidated threat and trust-boundary model, see [Security](security.md).

## Search is unscoped by default (multi-tenant caveat)

`search_hybrid()` / `recall()` / `search_similar()` span the **entire store** by
default — there is no implicit per-user or per-session boundary. If you keep
multiple tenants' thoughts in one database file, an unfiltered search can return
another tenant's memories.

Two supported ways to isolate:

- **A file per tenant** (the strongest boundary) — one store per tenant, managed
  via `EngravaManager`. Isolation is then the file boundary itself.
- **Scoped retrieval within one file** — pass `filters=` / `visibility=` to
  `search_hybrid()` or `recall()` to constrain results to a metadata scope
  (e.g. an owner or session key). The public `search_similar()` and
  `search_fts()` methods do not accept those metadata filters. See
  [Search](search.md#scoped-retrieval).

Choose file-per-tenant when tenants must never share a file; use scoped filters
for soft, in-file partitioning.

## Dreaming / consolidation: mechanism, not a proven retrieval lift

The built-in `DreamingExtension` performs no-LLM consolidation (promotion →
priority boost, association edges, reflections). With a fixed store,
configuration, cycle, embedding inputs, and deterministic custom signals, its
result is reproducible. Those are real
**mechanical** ranking effects, but Engrava makes **no claim that enabling
dreaming improves retrieval accuracy. The v0.5/v0.6-candidate frozen synthetic
snapshot measured aggregate recall@5 at `0.80` with Dreaming off and `0.70` with
it on, even though the separate curated release gates passed. Enable it for the
cognitive-hygiene mechanics it provides, not for an expected accuracy gain. See
[Dreaming](dreaming.md) and the current [benchmark evidence](benchmarks.md).

## Embedding-provider conformance is not checked at construction

`EmbeddingProviderProtocol` requires public `dimension` and `model_name`
members, but the protocol is **structural**: a store accepts any object passed
as `embedding_provider=` and only discovers a missing member when it reads one.
A provider that keeps the dimension privately (`self._dimension`) with no public
property therefore constructs cleanly and fails later — on the first vector
search that has to ask the provider — with `EmbeddingProviderContractError`
naming the class and the member. A store whose vector backend is `sqlite-vec`
takes the dimension from its dimension-typed `vec0` table instead, so that search
path does not ask the provider for it; `verify_embedding_model()` asks
regardless.

This is deliberate. Construction does not read the member, and a store that
never searches by vector — and never calls `verify_embedding_model()` — never
needs it, so failing at construction would reject stores that work today. Call
`verify_embedding_model()` after construction when you want the failure at
startup instead.

Note that the requirement is not new: `dimension` has been a required member of
the protocol since the first public release. See [Upgrade](upgrade.md#05---06).

## Validity intervals cannot be inverted

The bi-temporal `valid_from` / `valid_until` bounds on a `ThoughtRecord` or
`EdgeRecord` describe a forward interval. When **both** bounds are set, engrava
**rejects** an inverted interval (`valid_from` strictly after `valid_until`):

- **On write** — construction fails with a `ValidationError`; the store's update
  path re-validates the whole record, so a change that would invert a stored
  interval is refused; and `invalidate_thought()` / `invalidate_edge()` reject a
  `valid_until` earlier than the record's stored `valid_from`.
- **On read** — the invariant lives in the domain model, and the store
  reconstructs that model whenever it loads a row (`get_thought()`,
  `get_edges()`, and every ranked/query read). A row that became inverted **out
  of band** — for example one written by an older engrava build before this
  validation existed, or edited directly in the database file — therefore raises
  a `ValidationError` when it is read back, rather than silently returning a
  corrupt interval. This is a deliberate fail-loud choice for a data-integrity
  fault. If you are upgrading a database that may contain such rows, repair them
  (set the offending bound to `NULL`, or correct the order) before reading.

Bounds are compared as UTC-normalised instants, so differing offsets are
reconciled before the check. Two cases are **not** inversions and remain
accepted:

- **Equal bounds** (`valid_from == valid_until`) — a zero-length interval is a
  legitimate instantaneous fact.
- **An open bound** (`valid_from` or `valid_until` is `None`) — the interval is
  open on that side (see [The bi-temporal model](bitemporal.md)).

There is no way to store a fact "valid from June to March"; express an open-ended
fact with a `None` bound instead.

## macOS SQLite Extension Loading

macOS ships with a system SQLite that has extension loading disabled by default.
If you use the `vec` extra (`pip install engrava[vec]`), you may encounter:

```
sqlite3.OperationalError: not authorized
```

**Workaround:** Install Python via Homebrew or pyenv, which links against a
full-featured SQLite build:

```bash
brew install python@3.12
# or
pyenv install 3.12
```

## aiosqlite Proxy Architecture

engrava uses [aiosqlite](https://github.com/omnilib/aiosqlite) which runs
SQLite on a dedicated background thread and proxies calls via `asyncio`.
This has implications:

- **Connection objects** should not be shared across event loops.
- **Long-running transactions** block the background thread — keep transactions
  short.
- **WAL mode** is used by default for concurrent read access. Writes are
  serialized by SQLite's single-writer lock.

## sqlite-vec Pre-v1 Status

The [sqlite-vec](https://github.com/asg017/sqlite-vec) extension is pre-v1.
engrava pins `>=0.1.0,<0.2.0` to avoid breaking changes. When sqlite-vec
reaches 1.0, the pin will be relaxed.

Without the `vec` extra, engrava falls back to brute-force cosine similarity
search in Python. This works well for databases up to ~100k embeddings. For
larger collections, run `pip install 'engrava[vec]'` to use the compact compiled `vec0`
backend, but note that the pinned sqlite-vec 0.1.x line still performs an
**exhaustive linear KNN scan**. It reduces the constant factor and memory
overhead; it is not an approximate or sub-linear index. Measure your own p95
latency and see [Performance](performance.md#the-brute-force-ceiling-and-how-to-pass-it).

## FTS5 Availability

FTS5 is included in the standard SQLite build since version 3.9.0 (2015).
Most Python distributions include it. If FTS5 is not available, `search_fts()`
raises an error at schema creation time.

To verify FTS5 support:

```python
import sqlite3
conn = sqlite3.connect(":memory:")
conn.execute("CREATE VIRTUAL TABLE test USING fts5(content)")
conn.close()
print("FTS5 is available")
```

## Concurrent Write Safety

SQLite supports one writer at a time. With WAL mode, readers do not block
writers and vice versa. `aiosqlite` marshals every call onto one background
thread, so concurrent tasks' **statements** do not run at the same time.

That is statement-level serialisation, and it is not the same as operation-level
safety. Engrava's update methods (`update_thought`, `restore_thought`,
`upsert_by_hash`, `update_edge`, `update_action`) read the row, apply the change
in memory, then write — and another writer's whole update can land in that
window. Two writers editing the **same field** of the same row therefore lose one
of the two writes, silently and without an error. Editing different fields is
safe — an update writes only the columns it owns — **except** when one of the two
writers stamps a new `updated_cycle`: every thought update carries a version
guard on that column, so the other update then matches no row and is rejected in
full with `StaleDataError`, sharing no field with it or not.

**Only one store may write a given database file.** The locks that order
engrava's own operations live on the store instance, so a second store — in
another process, or a second connection in this one — is outside all of them.
WAL and `busy_timeout` keep the *file* intact under that topology; they do not
make the *operations* correct.

The full contract, the guarantees that do hold, and the idioms that close the
gap are in [Concurrency](concurrency.md). For multi-service setups via
`EngravaManager`, each service has its own database file with independent
locking — that is the supported way to run independent writers.

Separately: if the store ever quarantines its connection
(an internal safety response to an indeterminate transaction), quarantine
revokes admission synchronously: every new operation on the store or its journal
then fails fast with `ConnectionQuarantinedError`, so no write can flush an
orphaned transaction. It does not retract an operation already in flight — a
reader admitted just before quarantine may complete a possibly-stale read on the
pre-quarantine connection (never a commit). During quarantine the physical
connection close is a detached, best-effort cleanup; a permanently-hung close is
only a pending-task lifecycle nicety, not a safety concern.

## Embedding Dimension Consistency

All embeddings for a given database must use the same dimensionality. Mixing
dimensions (e.g., 384 and 768) is not supported and will cause search to
return incorrect results.

Engrava validates the stored model identity and raises
`EmbeddingModelMismatchError` rather than mixing incompatible vectors. A
deliberate CLI re-embed is available while restoring a snapshot into a
configured direct database or service. Pass `--config`: direct mode uses the
top-level `embeddings` provider, while service mode prefers its per-service
override and otherwise uses the top-level provider. Without a configured
provider, use `--skip-embeddings` or preserve the source vectors. See
[CLI restore](cli.md#restore).

## Query-vector dimension mismatch

Distinct from the DB-wide **Embedding Dimension Consistency** above (which is
about the *stored corpus* all sharing one dimension): this is about a single
**query** vector whose length does not match the store's embedding dimension.

`search_similar()` (and the vector arm of `search_hybrid()`) computes cosine
similarity between the query vector and stored embeddings, which is only defined
when both share a dimension. A wrong-length query vector is a caller-contract
violation, so it is **raised loudly** as
`VectorDimensionMismatchError` (carrying `expected` and `actual`) rather than
silently returning an empty result. The check is dimension-only and runs before
the degeneracy check, so a wrong-length all-zero vector is a dimension error, not
a [degenerate-vector degradation](observability.md#observability-signals).

> **Behaviour change — catch the typed error.** The wrong-dimension case now
> raises `VectorDimensionMismatchError` (a subclass of `EngravaError`, **not** of
> `ValueError`). Any caller that previously caught a plain `ValueError` around a
> vector search must catch `VectorDimensionMismatchError` (or `EngravaError`)
> instead.

## Archived thoughts and default retrieval

Archived thoughts (`lifecycle_status = ARCHIVED`) are **excluded from default
retrieval** on every ranked read — `search_hybrid()`, `recall()`, `search_fts()`,
and `search_similar()`. This is the retrieval side of
[Forgetting](memory-hygiene.md) — an **opt-in, off-by-default** hygiene loop — but
the exclusion applies to **any** archived row, whether or not that loop is enabled:
a forgotten (archived) thought stops surfacing without being deleted.

- **Behaviour change:** previously an `ARCHIVED` thought still appeared in search.
  It no longer does. This also affects the TTL `archive` strategy — a TTL-expired,
  archived thought now drops out of default search too.
- **Reversible:** `restore_thought()` re-activates a row, and `include_archived=True`
  re-admits archived rows for a single call without restoring them.
- **Not applied to counts/listing:** `count_thoughts()` and `list_thoughts()` still
  include archived rows (they are not ranked retrieval) — filter on
  `lifecycle_status` yourself if you need them excluded there.

## Deletion on a database that has not been migrated

Foreign-key `ON DELETE CASCADE` on the child tables (`edge`, `embedding`,
`action`) arrives with the **core-12** schema migration. A database carried
forward from an older engrava and never migrated is still below that version, so
nothing cascades. On such a database a deleted thought's **identifier** becomes
reachable again through the `vec0` vector arm — not immediately. The delete does
purge that thought's own vector from the index; what it cannot remove is the
`embedding` row behind it, and the reconcile that runs on the next
sqlite-vec-enabled open puts the vector back from that row. Everything below
describes the state **after** that reconcile.

**Three conditions must all hold.**

1. The database is **below core-12**. A freshly created store is not affected: a
   new database is bootstrapped at the head schema version, cascades included.
2. **A sqlite-vec backend is actually active on the store** — `vector_backend:
   sqlite-vec` *and* the extension loaded. If the load fails the store logs a
   warning and falls back to NumPy, which closes the gap; so does the default
   `vector_backend: numpy`. Either way the NumPy candidate query joins
   `thought`, and a row that is gone cannot be scored.
3. **The query reaches the `vec0` arm**, which it does whenever no *effective*
   metadata predicate applies. `search_similar()` takes no filter argument, so
   it is always on that arm. For `search_hybrid()` / `recall()`, `filters` and
   `visibility` are compiled together into one predicate, and a query carrying
   one is routed to the NumPy path even under sqlite-vec (the `vec0` table
   declares no metadata columns, so the predicate could only run as a
   post-`MATCH` join). What matters is whether that compilation produces
   anything:
   - `filters=None`, **an empty `MetadataFilter`**, and `visibility=None`
     compile to nothing. The query stays on the `vec0` arm and **is** affected —
     passing an empty filter is not protection.
   - a `MetadataFilter` holding at least one predicate, or **any**
     `VisibilityQueryFilter` at all, compiles to a predicate. That query takes
     the NumPy path, which joins `thought` and excludes the orphan on any schema
     version.

**What deletion does regardless of schema version.** The `thought` row is
removed and its `content` with it. Resolving the identifier — `get_thought()`,
or any read that hydrates an id into a record — returns `None`. The content does
not come back.

**What survives below core-12.** Nothing removes the thought's `embedding` row,
so it stays. Two lookups then read ownership as that row rather than as the
thought behind it:

- `sync_embeddings`, the reconcile that runs when a sqlite-vec-enabled
  connection opens, picks backfill candidates from the `embedding` table alone.
  The leftover row qualifies, so its vector is re-inserted into `embedding_vec`
  — including the vector that `delete_thought` had explicitly removed moments
  earlier.
- Vector search maps `embedding_vec` rowids back to ids through that same
  `embedding` table, and the post-search eligibility filter is an *exclusion*
  query over `thought`: a thought row that no longer exists produces no
  exclusion, so the id passes through.

So on such a database, after a delete that reported success **and after that
reconcile has run** — the delete itself leaves the index clean:

- the `vec0` arm returns the **deleted identifier** with a similarity score.
  From `search_similar()` that arm's output *is* the result; in
  `search_hybrid()` / `recall()` it is one input to fusion;
- a caller therefore learns that **a thought matching this query exists** in the
  index. Where the deletion answered an erasure request rather than being
  routine housekeeping, that existence signal is the disclosure that matters;
- **if the phantom reaches the returned window** it consumes one of the `top_k`
  slots. It need not: on a hybrid query, fusion and the final truncation to
  `top_k` can leave it outside the window altogether. And when it does land
  there, it costs you a live result only if at least one further live candidate
  would otherwise have qualified — the window is a truncation of whatever
  ranked, not a fixed-size budget the phantom takes a share of.

**What to do: migrate.** Run `engrava migrate` (or open the store through
`from_config()`, which calls `ensure_schema()`). The core-12 step recreates the
three child tables with `ON DELETE CASCADE` and purges any orphan rows that had
already accumulated. After it, deleting a thought takes its `embedding` row with
it and the reconcile has nothing left to backfill.

```bash
engrava --db engrava.db migrate
```

## Maximum Database Size

SQLite supports databases up to 281 TB (theoretical). In practice, engrava
has been tested with databases up to ~10 GB (millions of thoughts) without
issues. Performance depends on index coverage and query patterns.

## `HybridSearchResult.backends_used` Is an Open Set

`backends_used` is a `frozenset[str]` that may grow as new scoring signals
are added (e.g. `"priority"` was added in v0.3.0). Do **not** compare it
with exact equality (`== {"fts5", "vector"}`). Use subset checks instead:

```python
assert {"fts5"} <= result.backends_used  # preferred
```

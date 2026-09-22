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
  reconstructs that model whenever it loads a full record (`get_thought()`,
  `list_thoughts()`, `get_edges()` and `list_edges()`). A row that became
  inverted **out of band** — for example one written by an older engrava build
  before this validation existed, or edited directly in the database file —
  therefore raises a `ValidationError` when it is loaded through one of those
  calls, rather than silently returning a corrupt interval. This is a
  deliberate fail-loud choice for a data-integrity fault. Ranked search is the
  exception: `search_fts()`, `search_similar()`, `search_hybrid()` and
  `recall()` return ids and scores without rebuilding the model, so an inverted
  row can still appear in their results, and the `ValidationError` surfaces
  only when you load that record (for example with `get_thought()`). If you are
  upgrading a database that may contain such rows, repair them (set the
  offending bound to `NULL`, or correct the order) before reading.

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
- **Long-running SQL statements** block the background thread: it runs one call
  at a time, so a slow statement delays every other call on that connection. An
  open transaction alone does not — while a `suspend_auto_commit()` window sits
  idle, other tasks' reads still run. What the window does hold is the store's
  write lock, so a different task's write waits for it, for at most
  `write_lock_acquire_timeout_seconds`, and raises `WriteLockTimeoutError`
  after that — a long hold can make the waiting write fail instead of complete
  (see [Concurrency](concurrency.md)). Keep transactions short.
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
Most Python distributions include it. If FTS5 is not available, creating the
schema fails: `ensure_schema()` (and so opening a new store) raises a
`sqlite3.DatabaseError` and the store does not open. Separately, on a store that
is already open, `search_fts()` returns an empty list rather than raising when
the `thought_fts` table does not exist.

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

That is statement-level serialisation, and it is not by itself operation-level
safety. Engrava's update methods (`update_thought`, `restore_thought`,
`upsert_by_hash`, `update_edge`, `update_action`) read the row, apply the change
in memory, then write. Two genuinely concurrent tasks **sharing one store
instance** are serialised end to end by an in-process write lock, so one
task's own read-modify-write cannot be corrupted by another's landing mid-way
— see [Concurrency](concurrency.md#many-async-tasks-one-store). What survives
as a real gap is a competing write from **outside** that lock: a same-task
nested call reached through a caller-owned hook, or a write from a *second
store on the same database file* (a second connection, or a second process).
Every core row now carries a `revision` column that every guarded update
checks and increments atomically, so a competing write landing in either of
those gaps makes the guard match no row and raises `StaleDataError` — nothing
of the rejected update is written — rather than silently overwriting. Two
writers editing the same field, serialised through that guard, still resolve
last-write-wins, which is the correct outcome for two genuine edits; what the
guard removes is a write being torn or lost without any signal at all. See
[Optimistic concurrency](concurrency.md#optimistic-concurrency-and-staledataerror)
for the full contract.

**Only one store may write a given database file.** The locks that order
engrava's own operations live on the store instance, so a second store — in
another process, or a second connection in this one — is outside all of them.
WAL and `busy_timeout` keep the *file* intact under that topology, and the
`revision` guard above keeps a lost-update race from landing silently; neither
makes multiple concurrent writers on one file a supported topology.

The full contract, the guarantees that do hold, and the idioms that close the
gap are in [Concurrency](concurrency.md). For multi-service setups via
`EngravaManager`, each service has its own database file with independent
locking — that is the supported way to run independent writers.

Separately: if the store ever quarantines its connection
(an internal safety response to an indeterminate transaction), quarantine
revokes admission synchronously: any new operation that touches the database or
its journal then fails fast with `ConnectionQuarantinedError`, so no write can
flush an orphaned transaction. A call that never reaches the database can still
return, for example a search with an empty query text or a degenerate query
vector, which returns `[]`. Quarantine does not retract an operation already in
flight — a reader admitted just before quarantine may complete a possibly-stale
read on the pre-quarantine connection (never a commit). During quarantine the
physical connection close is a detached, best-effort cleanup; a permanently-hung
close is only a pending-task lifecycle nicety, not a safety concern.

## Embedding Dimension Consistency

All embeddings for a given database must use the same dimensionality. Mixing
dimensions (e.g., 384 and 768) is not supported. What happens depends on the
vector backend:

- With the sqlite-vec backend, storing a vector whose length differs from the
  backend's dimension raises `sqlite3.OperationalError`.
- With the NumPy backend, once a store holds vectors of two lengths, a search
  raises: `VectorDimensionMismatchError` for a query whose length differs from
  the first stored vector's, and a `ValueError` from NumPy for a query of that
  length.

Engrava also checks the stored model identity (name, dimension and document
prefix); a mismatch it finds is reported as `EmbeddingModelMismatchError`. The
model check runs on every embedding write and on every `verify_embedding_model()`
call on a store that has a provider — not only the first — so it does not stop
comparing after the first write.

A deliberate CLI re-embed is available while restoring a snapshot into a
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

**This used to be a documented non-guarantee: a deleted thought's identifier
could become reachable again through the `vec0` vector arm on a database
below the core-12 schema (the migration that adds `ON DELETE CASCADE` to
`edge`, `embedding`, and `action`). It no longer can.** A vector is now
treated as owned by a *live thought*, not by the presence of an `embedding`
row, and that rule is enforced in every place that could otherwise
resurrect one:

- **Deletion no longer depends on the cascade.** `delete_thought`, the TTL
  `delete` strategy, and hygiene GC each delete a thought's `edge`,
  `embedding`, and `action` rows explicitly, atomically with the parent
  delete that runs first — durable on every schema version, not only from
  core-12 onward.
- **Reconciliation only backfills a vector whose thought still exists.**
  `sync_embeddings` (the pass that runs on every sqlite-vec-enabled open)
  joins to `thought` before treating an `embedding` row as a valid backfill
  source, so a dangling row left behind by something older than this fix can
  no longer put a vector back.
- **The purge is the same join.** The sweep that removes orphaned
  `embedding_vec` rows (on reconcile, and in `engrava gc`) now considers a
  vector orphaned when its owning *thought* is gone, not only when its
  `embedding` row is gone.
- **Search resolves through a join that requires the thought row.** Both the
  `vec0` rowid-to-id resolution and the post-search eligibility check now
  positively confirm the thought exists (and is otherwise eligible) rather
  than only checking that it isn't *excluded* — an id that resolves to no
  thought row at all is dropped, not passed through by default.

**What deletion does, on any schema version.** The `thought` row is removed
and its `content` with it. Resolving the identifier — `get_thought()`, or
any read that hydrates an id into a record — returns `None`. The content
does not come back, and neither, now, does the identifier.

**Historical residue is not retroactively repaired.** If a database
accumulated dangling `embedding` rows *before* upgrading to this fix — every
delete made on a pre-core-12 schema by an older engrava build did, since
nothing removed them — those rows are still sitting there. They can no
longer be resurrected into a search result, but they are not cleaned up
until you migrate: `engrava migrate` runs the core-12 step, which recreates
the three child tables with `ON DELETE CASCADE` and purges the orphan rows
that had already accumulated.

```bash
engrava --db engrava.db migrate
```

**`engrava gc` refuses instead of running on a database below head.**
Physically deleting rows through an engine that does not understand the
schema it is deleting from is how this defect reached a user in the first
place, so `gc` — and every other built-in command that deletes user data —
now exits non-zero and names `engrava migrate` rather than proceeding on an
unmigrated database. A read-only command (`info`, `verify`, `export`,
`snapshot`, and a `query` that parses as `FIND`/`COUNT`/`SELECT`) is still
allowed to run against a behind schema — refusing an ordinary read because a
migration is pending would trade this defect for a worse one — but it warns
on stderr that the schema is behind rather than staying silent about it.

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

# Known Limitations

This document covers platform-specific notes, constraints, and known issues.

## Search is unscoped by default (multi-tenant caveat)

`search_hybrid()` / `recall()` / `search_similar()` span the **entire store** by
default — there is no implicit per-user or per-session boundary. If you keep
multiple tenants' thoughts in one database file, an unfiltered search can return
another tenant's memories.

Two supported ways to isolate:

- **A file per tenant** (the strongest boundary) — one store per tenant, managed
  via `EngravaManager`. Isolation is then the file boundary itself.
- **Scoped retrieval within one file** — pass `filters=` / `visibility=` to the
  ranked search methods to constrain results to a metadata scope (e.g. an owner
  or session key). See [Search](search.md#scoped-retrieval).

Choose file-per-tenant when tenants must never share a file; use scoped filters
for soft, in-file partitioning.

## Dreaming / consolidation: mechanism, not a proven retrieval lift

The `DreamingExtension` performs deterministic, no-LLM consolidation
(promotion → priority boost, association edges, reflections). Those are real
**mechanical** ranking effects, but Engrava makes **no claim that enabling
dreaming improves retrieval accuracy** on any benchmark — its measured effect on
our own runs was within noise. Enable it for the cognitive-hygiene mechanics it
provides, not for an expected accuracy gain. See [Dreaming](dreaming.md).

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
larger collections, install `engrava[vec]`.

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
writers and vice versa. If your application uses multiple async tasks that
write concurrently, `aiosqlite` serializes them automatically via its
background thread.

For multi-service setups via `EngravaManager`, each service has its own
database file with independent locking.

Under this single-writer contract, if the store ever quarantines its connection
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

The `restore --re-embed` flag validates model consistency and raises
`EmbeddingModelMismatchError` on mismatch.

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

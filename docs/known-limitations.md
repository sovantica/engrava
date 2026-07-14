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

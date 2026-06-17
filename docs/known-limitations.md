# Known Limitations

This document covers platform-specific notes, constraints, and known issues.

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

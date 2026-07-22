# Concurrency

Engrava is built on SQLite, so it inherits SQLite's concurrency model: **many
concurrent readers, one writer at a time.** This page explains what that means in
practice — within one process and across processes — and the specific behaviours
to know about (busy timeout, the journal's in-process lock, and per-service
isolation).

## WAL: many readers, one writer

File databases opened via `from_config` use **WAL** (write-ahead logging) mode.
Under WAL:

- **Readers don't block the writer and the writer doesn't block readers.** A
  read sees a consistent snapshot while a write is in progress.
- **There is still only one writer at a time.** Two writes are serialised; the
  second waits for the first to finish.

This is ideal for read-heavy agent-memory workloads: retrieval (the hot path) is
all reads and scales freely; writes are comparatively infrequent.

## Many async tasks, one store

**A single store instance safely serves many concurrent `asyncio` tasks.** You do
not need a connection pool or multiple stores for in-process concurrency:

- aiosqlite runs the actual SQLite calls on a dedicated background thread and
  marshals every query to it, so concurrent `await`s against one store are
  serialised onto that thread rather than racing.
- The store additionally guards order-sensitive operations (deduplication, the
  embedding-model check) with internal `asyncio.Lock`s.

What you must **not** do is share one store across **different event loops** — the
connection is bound to the loop it was created on. One store per loop; within
that loop, share it freely. (See
[Known Limitations](known-limitations.md#aiosqlite-proxy-architecture).)

## Busy timeout

When a connection can't immediately get the lock it needs (another writer holds
it), SQLite waits up to the **busy timeout** before giving up with
`database is locked`.

How the timeout is set depends on how the store opened its connection:

- **`from_config` and `EngravaManager`** (engrava owns the connection) open it
  with `PRAGMA busy_timeout=5000` **explicitly** — a second connection waits up to
  **5 s** for a lock instead of failing immediately. These paths also set
  `PRAGMA synchronous=NORMAL`, the documented-safe companion to WAL (durable
  across an application crash; only the most recent transactions are at risk on an
  OS crash or power loss).
- **The manual `SqliteEngravaCore(conn)` constructor** (you own the connection)
  changes none of these pragmas. The connection keeps whatever it was opened with;
  Python's `sqlite3`/`aiosqlite` default `busy_timeout` already happens to be
  **5000 ms**, and the default `synchronous` is `FULL`.

For workloads with more write contention you can raise the timeout on your own
connection before handing it to the store, or after `from_config` via the store's
connection:

```python
import aiosqlite
from engrava import SqliteEngravaCore

conn = await aiosqlite.connect("engrava.db")
conn.row_factory = aiosqlite.Row
await conn.execute("PRAGMA busy_timeout = 15000")  # wait up to 15s for a lock
store = SqliteEngravaCore(conn)
await store.ensure_schema()
```

A longer busy timeout trades latency-on-contention for fewer `database is locked`
errors; tune it to your write pattern.

## Multiple processes

WAL allows multiple **processes** to read concurrently, and one to write — but
heavy multi-process **writing** of the same database file is **out of scope** for
Engrava, for two reasons:

1. **SQLite is single-writer.** Multiple OS processes writing the same file
   contend on the database lock; the busy timeout only papers over light
   contention.
2. **The audit journal's lock is in-process only.** When journaling is enabled,
   appends are serialised by an `asyncio.Lock` keyed on the connection — which
   exists **only within one process**. A second process shares no such lock, so
   two processes journaling the same database can race the journal's
   monotonic `sequence_number`. The writer retries on the resulting
   `UNIQUE` collision up to **5 times**; if contention persists it raises:

   ```
   RuntimeError: Failed to append journal entry after 5 retries due to sequence contention
   ```

   This is the signal that you have more than one process writing a journaled
   database — which is unsupported.

If you need multiple independent writers, don't point them at the same file —
give each its own database (next section).

## Per-service isolation

`EngravaManager` runs **one database file per named service**, each with its own
connection and its own lock. This is the supported way to isolate writers (per
tenant, per worker, per logical partition):

```python
from engrava import EngravaManager, load_config

config = load_config("engrava.yaml")
async with EngravaManager.from_config(config.services) as mgr:
    store_a = await mgr.get_store("tenant_a")  # tenant_a.db
    store_b = await mgr.get_store("tenant_b")  # tenant_b.db
```

Because each service is a separate file, writes to `tenant_a` never contend with
writes to `tenant_b`, and each can be backed up or deleted independently. See the
[scoping section](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy)
for when to choose per-service isolation over in-store filtering.

## Summary

| Scenario | Supported? | Notes |
|---|---|---|
| Many async tasks, one store, one loop | ✅ | The normal case — share the store. |
| Many readers (WAL) | ✅ | Readers never block the writer. |
| One writer at a time | ✅ | SQLite serialises writes. |
| One store across multiple event loops | ❌ | Connection is loop-bound; one store per loop. |
| Many processes reading the same file | ✅ | WAL supports concurrent readers. |
| Many processes writing the same file | ❌ | Single-writer; journal lock is in-process — use `EngravaManager`. |

## See also

- [Deployment](deployment.md) — process model, files on disk, graceful shutdown
- [Known Limitations](known-limitations.md) — the aiosqlite proxy and write-safety notes
- [Audit Trail](audit-trail.md) — the journal whose lock is discussed above

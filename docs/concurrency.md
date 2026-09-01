# Concurrency

Engrava is built on SQLite, so it inherits SQLite's concurrency model: **many
concurrent readers, one writer at a time.** This page states what a single store
guarantees concurrent callers, what it deliberately does **not** guarantee, and
which writer topologies are supported.

## WAL: many readers, one writer

File databases opened via `from_config` use **WAL** (write-ahead logging) mode
**by default** — it is the `database.wal_mode` setting, and setting it to `false`
leaves the connection on SQLite's rollback journal, where none of the guarantees
in this section apply. Under WAL:

- **Readers don't block the writer and the writer doesn't block readers.** A
  read sees a consistent snapshot while a write is in progress.
- **There is still only one writer at a time.** Two writes are serialised; the
  second waits for the first to finish.

This is ideal for read-heavy agent-memory workloads: retrieval (the hot path) is
all reads and scales freely; writes are comparatively infrequent.

## Many async tasks, one store

Share one store instance across the `asyncio` tasks in your event loop. You do
not need a connection pool or multiple stores for in-process concurrency — but
the store does **not** make a read-modify-write operation atomic, and the rest of
this section is the precise line between the two.

### What is guaranteed

- **Statements do not run concurrently.** aiosqlite runs the actual SQLite calls
  on a dedicated background thread and marshals every call onto it, so two tasks'
  statements are serialised rather than executed at once — and SQLite applies
  each statement atomically, so no query observes a half-written row.
- **An edit writes only the columns it owns.** `update_thought` writes the fields
  the call gave a new value to, plus the `updated_at` stamp — not the whole
  record. A field another task changed since this call read the row is preserved
  rather than rolled back. That includes the fields engrava maintains for you:
  `access_count` / `last_accessed_at` from `record_access()`, and
  `confirmation_count`.

  **This holds only while nobody stamps a cycle.** Every update also carries a
  version guard on `updated_cycle`
  ([below](#optimistic-concurrency-and-staledataerror)). If the competing writer
  moved that column, your update matches no row and is **rejected in full** with
  `StaleDataError` — even though the two edits share no field. So two edits to
  different fields both survive when both are ordinary edits, and stop doing so
  the moment either one stamps a cycle.
- **A deduplication sighting counts.** `create_thought(deduplicate=True)` and
  `get_or_create()` bump `confirmation_count` **relative to what is stored**
  (`confirmation_count + 1`, evaluated by SQLite), so a bump made by another
  writer since the row was read is added to, never overwritten.
- **Deduplication is serialised against itself.** `create_thought(deduplicate=True)`,
  `get_or_create()` and `upsert_by_hash()` share an internal `asyncio.Lock`
  covering the whole "probe the content hash, then insert or bump" window, so two
  tasks on this instance calling **one of these three methods** cannot both race
  past the probe. The same window is *also* serialised across connections — see
  [Multiple stores, one database file](#multiple-stores-one-database-file). This
  is a narrower guarantee than "safe against concurrent activity on the
  instance" — see the next section for what a *different*, concurrent write can
  still do to this window.

### What is not guaranteed

- **A read-modify-write is not a critical section.** `update_thought` reads the
  row, applies your changes to it, then writes. aiosqlite serialises individual
  *statements*, not method bodies, so another task's entire update can land
  between that read and that write. If both edits name the **same** field, the
  one whose `UPDATE` runs last wins and the other is discarded — silently, with
  no error raised and nothing in the row to show it happened. (Edits to
  *different* fields survive each other, subject to the cycle-guard exception
  above.) The same window exists in `restore_thought` and
  `upsert_by_hash`, which share the thought write path, and in `update_edge` /
  `update_action` — those two carry no version guard at all, so they can never
  raise `StaleDataError` under any circumstance.
- **`StaleDataError` does not detect that.** See
  [Optimistic concurrency](#optimistic-concurrency-and-staledataerror) below for
  what the version guard actually rejects.
- **A state-machine check is made against the row this call read.**
  `update_thought(lifecycle_status=...)` and `update_action(status=...)` validate
  the transition against the record the call itself loaded. If another writer
  moves the row after that check passes, the write still lands — so two
  individually-legal transitions can compose into one the state machine forbids
  (an `ACTIVE -> DONE` edit landing on a row another task already archived leaves
  it `DONE`, and `ARCHIVED -> DONE` is not an allowed edge). Serialise lifecycle
  transitions on a row if that matters to you.
- **`suspend_auto_commit()` belongs to the store, not to the task that opened
  it.** The deferred-commit flag lives on the store instance, so a write issued
  by *any* task while the window is open joins the window's transaction. If the
  window rolls back, that unrelated write is rolled back with it, and the task
  that issued it is never told.
- **The dedup guard's own window has the same exposure as `suspend_auto_commit()`,
  in both directions.** `create_thought(deduplicate=True)`, `get_or_create()`
  and `upsert_by_hash()` hold a real transaction open across their
  probe-and-row window (see [Busy timeout](#busy-timeout)). `_dedup_lock` only
  keeps **another call to one of these three methods** from racing that
  window — it is not a general per-instance write lock. A concurrent,
  unrelated write from a different task on the same store instance
  (`record_access()`, `update_thought()`, a delete, ...) can still land while
  the window is open, in which case it joins that transaction: it is made
  durable a little earlier than the other task's own commit would have if
  this call's own write succeeds, and it is **rolled back along with this
  call's own work if this call fails** — the same "joins the window's
  transaction, and shares its fate" behaviour the bullet above describes for
  `suspend_auto_commit()`, and for the same reason: a connection has exactly
  one transaction, so once another task's write has joined this one there is
  no way for this call to tell its own work apart from the other task's when
  deciding what to commit or roll back. This is not new (every open
  transaction on this connection has always had this exposure, including
  before this window existed at all) and it is not fixed by clever bookkeeping
  inside the guard — an earlier version of this page said this window commits
  a concurrent task's write but never rolls it back, and that turned out to be
  false: closing the rollback side inside the guard only moved the defect
  somewhere else, three times, so the guard now defers every commit to
  whichever method actually did the write and keeps only a rollback for its
  own failure path. Closing this for real needs a task-reentrant lock around
  every write path on the instance, not only the three dedup methods. Engrava
  does not have that yet; until it does, the safe idioms below (one task owns
  a row, or partition writes so they never compete) apply here too.
- **One store belongs to one event loop.** Not because of the connection —
  aiosqlite creates each operation's future on the *calling* loop, so a plain
  read from a second loop works. It is the store's own synchronisation: the
  deduplication lock (and the audit journal's per-connection lock) are
  `asyncio.Lock`s, which bind to the first loop that has to **wait** on one and
  then raise `RuntimeError` for a waiter from any other loop. An uncontended
  lock never binds, so a store shared across loops can look healthy right up to
  the first time two callers actually contend. Use one store per loop; within
  that loop, share it freely. (See
  [Known Limitations](known-limitations.md#aiosqlite-proxy-architecture).)

### The safe idioms

- **Hold one store for the process lifetime** and share it across tasks — see
  [Deployment](deployment.md#one-store-per-process-opened-at-startup).
- **Let one task own a row**, or partition edits so that no two tasks write the
  same field of the same row. This needs no locking at all.
- **Or serialise the read-modify-write yourself** when tasks genuinely compete
  for one field:

  ```python
  # engrava does not make read-modify-write atomic; a caller-owned lock does.
  # edit_lock is one asyncio.Lock shared by every task that writes this way.
  async with edit_lock:
      thought = await store.get_thought(thought_id)
      if thought is not None:
          await store.update_thought(
              thought_id,
              confidence=min(1.0, thought.confidence + 0.1),
          )
  ```

  This closes the window **within one process only** — see
  [Multiple stores, one database file](#multiple-stores-one-database-file).
- **Drive `suspend_auto_commit()` from one task at a time**, with no other writer
  on that store instance for the duration. `bulk_store()` relies on the same
  contract.

## Optimistic concurrency and `StaleDataError`

`update_thought`, `restore_thought` and `upsert_by_hash` carry a version guard,
and it detects less than its name suggests. The guard compares the row's
`updated_cycle` against the value read at the start of the call — and **no
engrava operation advances `updated_cycle` on its own.** A cycle is stamped only
when a caller passes `updated_cycle=` to `update_thought` or `current_cycle=` to
`restore_thought`. `update_edge` and `update_action` carry no version guard at
all — their writes are keyed on the row id alone — so they never raise
`StaleDataError` under any circumstance.

**`StaleDataError` means the guarded `UPDATE` matched no row.** Two different
situations produce that, and the error does not tell them apart:

- the row's `updated_cycle` is no longer the value this call read — a competing
  writer stamped a cycle; or
- **the row no longer exists** — a competing writer deleted it between this
  call's read and its write. (`ThoughtNotFoundError` covers a row that was
  already missing when the call *started*, not one that vanished mid-call.)

The consequences are worth stating plainly:

- **An ordinary concurrent edit does not raise `StaleDataError`.** The row
  changed, the guard does not notice, and the write proceeds. The semantics are
  last-write-wins, per column.
- **A competing cycle stamp does raise it, whatever your edit touched**, and the
  rejected update then writes *nothing at all* — no field of it reaches storage.
  Recover by re-reading the record and recomputing the change; do not replay the
  original `changes`.

So `StaleDataError` is neither a general staleness check nor a reliable "someone
stamped a cycle" signal: it is "this write found no row to apply to". Do not read
its *absence* as proof nobody else wrote. If your application needs caller-level
staleness detection, keep your own version value in `metadata` and compare it
yourself, or serialise the edits with the lock idiom above.

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

**The content-hash dedup path is the one exception to "raw `database is
locked`."** `create_thought(deduplicate=True)`, `get_or_create()`,
`upsert_by_hash()` and `bulk_store(deduplicate=True)` open their probe-and-row
window with `BEGIN IMMEDIATE` instead of letting the probe's `SELECT` open an
implicit deferred transaction — so lock contention surfaces at the start of the
window rather than only once the eventual `INSERT`/`UPDATE` needs the lock,
which is where a deferred transaction would otherwise invite the classic
embedded-SQLite deadlock. The busy timeout above still applies to each
attempt; on top of it, the call retries a bounded number of times with a
short backoff and, only once that is exhausted, raises `WriteContentionError`
(a subclass of `EngravaError`) instead of leaking `sqlite3.OperationalError`.
Retrying the call outright is safe — nothing was read or written before the
lock was acquired.

For the single-row calls — `create_thought(deduplicate=True)`,
`get_or_create()` and `upsert_by_hash()` — the lock is released again as soon
as the row is written, before auto-embed or the `on_store` hook run, so a slow
or stalled embedding call cannot hold the file's write lock hostage.
**`bulk_store(deduplicate=True)` does not share this property, by design, and
this fix does not change that.** Its insert loop, and — when auto-embed is on
— the one batch embedding call it makes afterwards, run inside a single
`suspend_auto_commit` transaction (see
[Many async tasks, one store](#many-async-tasks-one-store)), so the write lock
taken for the first row's probe is held until the whole batch commits at the
end: across every row's insert *and* across the batch embedding call. This is
not new behaviour. Before `BEGIN IMMEDIATE` existed on this path, the first
`INSERT` in the loop already took SQLite's RESERVED lock implicitly and held
it for exactly as long, because `bulk_store` is documented as one
all-or-nothing transaction per batch. `BEGIN IMMEDIATE` only moves *when* the
lock is acquired — to before the first row's probe instead of at the first
row's insert — it does not change how long the lock is held.

One case falls back to the pre-`BEGIN IMMEDIATE` behaviour: a caller already
holding an open transaction on the connection through means this store does
not manage (a raw `BEGIN` issued directly against it, with nothing written
yet). This store cannot open a second `BEGIN` inside an already-open
transaction, so it cannot acquire the write lock up front in that case, and
falls back to whatever atomicity the caller's own transaction provides plus
the in-process lock. Nothing in `bulk_store` or `suspend_auto_commit` triggers
this — both are covered by the paragraph above — it can only arise from direct,
unmediated use of the underlying connection.

## Multiple stores, one database file

**Engrava does not currently support more than one store *writing* the same
database file** — whether those stores are in two processes or are two
connections inside one process.

That is a statement about engrava, not about SQLite. WAL (by default) and
`PRAGMA busy_timeout=5000` are configured, and they do exactly what they say: the
*file* tolerates several connections, readers do not block the writer, and SQLite
serialises the writers so the database itself is never corrupted. What they
cannot do is make engrava's own multi-statement operations correct across
connections, because every mechanism that orders those operations lives on a
single store object and stops at its boundary:

1. **An update can be silently discarded.** The read-modify-write window
   described above does not stop at the store — a second store's edit lands in it
   just the same, and the `updated_cycle` guard does not report it. Between two
   *processes* no in-process lock could close that window even in principle.
2. **`deduplicate=True`, `get_or_create()`, `upsert_by_hash()` and
   `bulk_store(deduplicate=True)` no longer duplicate across stores.** The
   in-process `asyncio.Lock` is still per store instance, but the
   probe-and-row window itself now opens with `BEGIN IMMEDIATE` (see
   [Busy timeout](#busy-timeout)) as soon as no transaction is already open on
   the connection — including the first row of a `bulk_store` batch, which
   covers the whole batch under that same lock even though `bulk_store` defers
   every row's own commit to the end. A second store reaching the same window
   while it is open cannot even start its own transaction, so the two can no
   longer both probe unlocked and both insert. Ordinary contention is waited
   out; only past a bounded number of retries does the second store raise
   `WriteContentionError` instead of getting a turn. The one case this does not
   cover — a caller holding a raw, unmediated transaction on the connection —
   is the same residual gap named under [Busy timeout](#busy-timeout).
3. **The audit journal's sequence numbers collide.** When journaling is enabled,
   appends are serialised by an `asyncio.Lock` keyed on **the connection**. A
   second store has a second connection and therefore a different lock — in
   another process, where no lock could be shared at all, but equally in this
   one. Two stores journaling the same database can race the journal's monotonic
   `sequence_number`. The writer retries on the resulting `UNIQUE` collision up to
   **5 times**; if contention persists it raises:

   ```
   RuntimeError: Failed to append journal entry after 5 retries due to sequence contention
   ```

   That error is the loud symptom of this topology. The first of the three
   (the silently-discarded update) has no symptom at all; the second is no
   longer a failure in the case this store manages — only the raw-transaction
   fallback described above still fails the same way, silently.

Several things *do* survive a second connection, and they are worth knowing so
the rule is not read as broader than it is: **reading** (any number of
processes may read one file under WAL); the **relative `confirmation_count`
bump**, evaluated by SQLite against the stored row rather than derived from a
prior read; and, as of the `BEGIN IMMEDIATE` window described above, **the
dedup probe-and-insert itself** — no longer racy, whether it lands as a fresh
insert or a confirmation bump.

Making the read-modify-write paths correct across connections needs a
transaction-level mechanism engrava does not have yet. It is deferred, not ruled
out; until it ships, treat **one writer per database file** as the contract. If
you need multiple independent writers, give each its own database (next section).

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
| Many async tasks, one store, one loop | ✅ | The normal case — share the store. The rows below qualify it. |
| Many readers (WAL) | ✅ | Readers never block the writer. |
| One writer at a time | ✅ | SQLite serialises writes. |
| Two tasks editing **different** fields of one row | ✅ | An update writes only the columns it owns — unless one of them stamps `updated_cycle` (next row). |
| Either task stamping `updated_cycle` | ❌ | The version guard rejects the other update in full, whatever field it touched. |
| Two tasks editing the **same** field of one row | ❌ | Last write wins, silently. `StaleDataError` does not fire; serialise it yourself. |
| Two tasks moving one row through its state machine | ❌ | Each transition is validated against the state its own call read. |
| `suspend_auto_commit()` with another writer on that store | ❌ | The window is store-wide; the other writer's rows roll back with it. |
| One store across multiple event loops | ❌ | The store's `asyncio.Lock`s bind to one loop; one store per loop. |
| Many processes reading the same file | ✅ | WAL supports concurrent readers. |
| Many stores/processes **writing** the same file | ❌ | Not supported — one writer per file; use `EngravaManager`. |

## See also

- [Deployment](deployment.md) — process model, files on disk, graceful shutdown
- [Known Limitations](known-limitations.md) — the aiosqlite proxy and write-safety notes
- [Error handling and recovery](error-handling.md) — what to do when a write fails
- [Audit Trail](audit-trail.md) — the journal whose lock is discussed above

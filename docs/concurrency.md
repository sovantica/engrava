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
not need a connection pool or multiple stores for in-process concurrency.
**Every guarded write path on a store instance shares one in-process,
task-reentrant write lock:** `create_thought` / `get_or_create` /
`upsert_by_hash` / `bulk_store` / `update_thought` / `restore_thought` /
`delete_thought` / `create_edge` / `update_edge` / `delete_edge` /
`create_action` / `update_action` / `store_embedding` / `record_access` /
`cleanup_expired` / `run_hygiene` / `suspend_auto_commit()` all take it before
touching a row. **Once locked**, each one's locked work runs inside one
continuous hold of the lock — one critical section **with respect to every
other task on this instance** — so a different task's call to any of these
blocks until the one in progress has committed (or rolled back) rather than
landing in the middle of it. That "once locked" qualifier is load-bearing for
four of them: `get_or_create` and `upsert_by_hash` take the lock *twice* on a
miss, with a real gap between (below); `create_thought` and `bulk_store` both
run the
[pre-insert seam](extension-hooks.md#1b-pre-insert-preparation-seam) — and,
for `create_thought`, its surrounding metadata/provenance validation too —
*before* the single locked window even opens, holding no lock this call
itself acquires while they do. `create_thought` validates the candidate,
runs the seam, and revalidates whatever it returns before it ever takes the
lock, then takes the lock once for the existence check and the insert or
confirmation-bump together. `bulk_store` runs the seam once per item, for
the whole batch, before it ever takes the lock; it then takes the
lock once, for the rest of the batch — per-item validation, the existence
check, and the insert or bump, item by item — so on both of these entry
points the seam (and, for `create_thought`, its surrounding validation) is
not covered by the same critical section that protects the write (see
[Extension hooks §1B.1](extension-hooks.md#1b1-contract) for the
per-entry-point seam invocation count).

**`get_or_create` and `upsert_by_hash` are the exception, on a miss.** Each
runs an exploratory probe under the lock; when that probe finds nothing, the
call releases `_write_lock`, `_dedup_lock`, and any `BEGIN IMMEDIATE` it
opened, runs the [pre-insert seam](extension-hooks.md#1b-pre-insert-preparation-seam)
(`prepare_thought_for_insert`) with **no lock this call itself took still
held** — a lock an *enclosing* caller took is a different matter, covered
below — and only then reacquires `_write_lock`, `_dedup_lock`, and a fresh
`BEGIN IMMEDIATE` for a decisive probe before it inserts. For a *different*
task, that gap is real: any other guarded write on this instance — a
second task's `get_or_create` / `upsert_by_hash` /
`create_thought(deduplicate=True)` / `bulk_store` targeting the same content,
or an unrelated `update_thought` / `delete_thought` / anything else in the
list above — can start and finish inside it, because nothing is held there to
stop it. The decisive probe exists to catch exactly that: it is re-run, keyed
on the seam's own output, so a write that landed in the gap turns this call's
outcome into a hit instead of a double insert (see
[Extension hooks §1B.1](extension-hooks.md#1b1-contract) for the full
per-entry-point invocation-count contract). On a stable hit neither method
ever opens this gap — the seam does not run, and the exploratory probe's own
critical section is the whole call.

**Nested inside the caller's own `suspend_auto_commit()`, none of that
applies to that caller's own task.** `_write_lock` is task-reentrant (below):
the outer window's hold of it never actually drops while this call's own
exploratory probe releases its nested acquisition, so `_write_lock` stays
held — blocking every *other* task — for the whole seam call. Whatever
transaction the outer window had already opened before this call began is
untouched by it too: the exploratory probe only opens its own `BEGIN
IMMEDIATE` when none was open yet, and only rolls one back on that same
condition, so an already-open outer transaction is never the one it closes.
`_dedup_lock` is not task-reentrant and is unaffected by any of this — it is
acquired and released fresh by this call regardless of nesting, and is the
one thing that genuinely comes off, for the duration of the seam call, in
this case. See [Extension hooks §1B.1](extension-hooks.md#1b1-contract) (the
nesting note) for the same fact stated from the seam-override side.

The lock is **task-reentrant, not call-reentrant**: the *same* task may freely
nest calls — notably a write issued from inside its own
`suspend_auto_commit()` window — but a *different* task genuinely waits for
whichever critical section it is trying to enter (a task is not waiting on
this lock at all while it sits in the gap described above — nothing holds the
lock there to wait for). This closes several races the rest of this page used
to document as permanent; the exact scope (what changed and what did not) is
stated precisely below rather than summarised here, because the two are easy
to conflate and the difference matters.

**What this does not do:** it does not make a read-modify-write your own code
performs *across two separate calls* atomic — `thought = await
store.get_thought(id)` followed later by `await store.update_thought(id,
field=thought.field + 1)` is still two independent critical sections with a gap
between them, exactly as before (see [the safe idioms](#the-safe-idioms)
below). It also does not reach across connections — a second store on the same
database file is unaffected (see
[Multiple stores, one database file](#multiple-stores-one-database-file)) — or
around a caller holding a raw, unmediated transaction directly against the
connection (see [Busy timeout](#busy-timeout)). **Nor does it reach a direct
call to `store.journal.append()`** — the public `JournalWriter` returned by
`store.journal` does not hold this lock itself (it only serialises its own
sequence-number allocation), so appending through it yourself, outside of a
write this store performs on your behalf, has the same exposure as the raw,
unmediated connection case: it can land inside another task's open window and
be rolled back with it. Let the store journal its own mutations; treat a
direct `append()` call as unmediated use of the connection.

### What is guaranteed

- **Statements do not run concurrently.** aiosqlite runs the actual SQLite calls
  on a dedicated background thread and marshals every call onto it, so two tasks'
  statements are serialised rather than executed at once — and SQLite applies
  each statement atomically, so no query observes a half-written row.
- **A guarded write's own existence check and write are now atomic across
  tasks, and so is its validation for most of these calls** — the exception
  is broader than one miss path: `get_or_create` / `upsert_by_hash` have
  their own real gap between the two probes on a miss, and `create_thought` /
  `bulk_store` validate the candidate and run the pre-insert seam before they
  ever take the lock, holding no lock this call itself acquires while they do
  (see [above](#many-async-tasks-one-store) for both). Neither reopens a race
  on *stored* data — that pre-lock work only validates and enriches the
  incoming candidate, it never reads a row this store already holds — but it
  means the atomicity below covers each call's locked work, not literally
  every line it executes. For every other guarded write, a
  different task's entire operation could, before this, land between one
  call's read and its write; now it cannot start until the first call's write
  lock is released. Concretely: a second task's read inside
  `update_thought` / `restore_thought` / `update_edge` / `update_action` always
  observes the *first* task's already-committed result, never a stale
  in-flight value; a competing cycle stamp that used to spuriously reject an
  unrelated edit — merely because the two calls' reads and writes happened to
  straddle each other — no longer does, because the second call's version guard
  is now captured against the post-first-call row; and a lifecycle transition
  is now validated against whatever the *most recently completed* call on this
  instance actually left behind, not a value another task's call might still be
  mid-write on.
- **An edit writes only the columns it owns.** `update_thought` writes the fields
  the call gave a new value to, plus the `updated_at` stamp — not the whole
  record. A field another task changed since this call read the row is preserved
  rather than rolled back. That includes the fields engrava maintains for you:
  `access_count` / `last_accessed_at` from `record_access()`, and
  `confirmation_count`.

  **This holds for the ordinary case — two genuinely concurrent tasks on this
  store — because the write lock above already serialises them end to end: by
  the time the second call does its own internal read, the first call's write
  has fully committed, so the second call's guard is captured against the
  post-first-call row and never finds it stale.** Every update also carries a
  `revision` guard now ([below](#optimistic-concurrency-and-staledataerror)),
  and the engine bumps `revision` on *every* guarded write, automatically — not
  only when a caller stamps something. What that guard actually catches is
  exactly the interleavings the write lock does **not** serialise: a same-task
  nested write reached through a caller-owned hook (the lock is
  task-reentrant, so it does not block this), and a second *store* on the same
  database file (a second process, or a second connection in this one). Either
  of those now makes the guarded write match no row and raise `StaleDataError`
  in full — even when the two edits share no column — where before, the first
  case only ever raised if the competing writer happened to stamp a cycle
  explicitly, and the second case never raised at all (see [Multiple stores,
  one database file](#multiple-stores-one-database-file)).
- **A deduplication sighting counts.** `create_thought(deduplicate=True)` and
  `get_or_create()` bump `confirmation_count` **relative to what is stored**
  (`confirmation_count + 1`, evaluated by SQLite), so a bump made by another
  writer since the row was read is added to, never overwritten.
- **Deduplication is serialised against itself, and now against every other
  guarded write too.** `create_thought(deduplicate=True)`, `get_or_create()`
  and `upsert_by_hash()` share an internal `asyncio.Lock` covering the whole
  "probe the content hash, then insert or bump" window, so two tasks on this
  instance calling **one of these three methods** cannot both race past the
  probe — and that same window is also held under `_write_lock`,
  so an *unrelated* guarded write from a different task cannot land inside it
  either (see the "riding along" exposure this closes, described under
  [Busy timeout](#busy-timeout)). The probe-and-insert window is *also*
  serialised across connections — see
  [Multiple stores, one database file](#multiple-stores-one-database-file).
- **`suspend_auto_commit()` genuinely excludes every other task's guarded
  write for its duration**, and survives nesting. See
  [the dedicated section below](#suspend_auto_commit-is-now-a-real-exclusive-window).

### What is not guaranteed

- **A caller-level read-modify-write across two separate calls is still not
  atomic.** The lock above covers one call's *own* internal read and write —
  it does not, and cannot, cover a gap between two calls your own code makes
  (`get_thought` then, later, `update_thought`). See
  [the safe idioms](#the-safe-idioms) for the caller-owned-lock pattern that
  closes that gap.
- **`get_or_create` / `upsert_by_hash` have this same kind of gap internally,
  on a miss.** Between their exploratory probe and their decisive probe,
  every lock and transaction the call itself opened is released while the
  pre-insert seam runs — see [above](#many-async-tasks-one-store) for what
  can land there and why the decisive probe exists.
- **Same-field edits still resolve last-write-wins.** Two tasks legitimately
  editing the same field is not a bug the lock removes — it removes the
  *silent corruption* (a call's own read or write being torn by another call
  landing mid-operation), not the ordinary fact that two edits to the same
  field can only leave one final value. Whichever call's critical section runs
  second determines the field's value, deterministically and without error.
- **`StaleDataError` does not detect "someone else wrote it," in general.** See
  [Optimistic concurrency](#optimistic-concurrency-and-staledataerror) below for
  what the `revision` guard actually rejects — it is a per-row write counter,
  not a semantic-conflict detector, and it says nothing about two sessions
  writing contradictory content to two *different* rows.
- **A same-*task* nested call is not blocked by its own lock.** A caller-owned
  hook or callback that itself issues a write, invoked synchronously from
  inside another operation's read-modify-write span (not a second `asyncio`
  task — a genuinely nested call on the *same* task), still runs freely: the
  lock is task-reentrant by design (see the re-entrancy requirement above), so
  it does not — and structurally cannot — protect against this shape. This is
  a narrow, same-task case; ordinary callers issuing a second independent
  operation from a second task are covered by the guarantee above.
- **A second store on the same database file is unaffected.** See
  [Multiple stores, one database file](#multiple-stores-one-database-file).
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

### `suspend_auto_commit()` is now a real, exclusive window

Before this, the deferred-commit flag lived on the store instance with
nothing to stop a *different* task's write from landing inside an open
`suspend_auto_commit()` window: it joined the window's transaction, and if the
window rolled back, that unrelated write rolled back with it — silently, with
the other task never told. `suspend_auto_commit()` now holds `_write_lock` for
its entire duration, so a different task's call to any guarded write path
cannot even begin until the window has closed. Concretely:

- **A different task's write now waits, and survives independently.** It
  cannot join the window's transaction — by the time it runs, the window has
  already committed or rolled back — so its own outcome no longer depends on
  the window's.
- **Nesting is supported and safe.** A nested `suspend_auto_commit()` call —
  reachable only from the *same* task, since the lock above blocks any other
  task before it could nest — shares the outer call's transaction. Only the
  **outermost** call commits on a clean exit or rolls back on an exception; an
  inner call's own exit does neither.
- **A write issued from inside your own window still works.** The lock is
  task-reentrant: the same task's own writes inside the window it opened do
  not block on themselves.

This governs **this store instance** only. A second store on the same
database file is unaffected — see
[Multiple stores, one database file](#multiple-stores-one-database-file) — and
the residual raw-transaction case described under
[Busy timeout](#busy-timeout) is still outside what any in-process lock can
reach.

### A deadlock this store cannot resolve raises, it does not hang

The in-process write lock is task-reentrant: the task that opened a
`suspend_auto_commit()` window may issue writes of its own inside it freely,
but a *different* task cannot reuse that grant. If the window's own task then
spawns a task — `asyncio.create_task`, `gather`, `wait_for`, or one spawned
inside an `on_store` hook or embedding provider callback — and **awaits it
before the window closes**, neither can make progress: the spawned task can
never get the lock the window's task holds, and the window's task can never
release it while still awaiting the spawned task. This is already out of the
documented contract above (drive every write inside an open window from the
one task that opened it), so it is a caller bug, not a case this store can
make safe by construction.

What it must not become is a silent, unattributable hang. Acquiring this lock
as a *different* task is bounded, and past the bound the acquiring call raises
`WriteLockTimeoutError` instead of waiting forever. Raising also ends the
deadlock itself: the spawned task's failure lets whatever was awaiting it
unwind, which frees the window's task to finish and release the lock.

**The same policy applies to the dedup lock's own, narrower failure mode: a
*same*-task re-entry.** The write lock above is task-reentrant by design (a
write nested inside your own `suspend_auto_commit()` window must proceed);
the in-process lock guarding the dedup probe-and-insert window
(`create_thought(deduplicate=True)`, `get_or_create`, `upsert_by_hash`) has
no such legitimate reentrant use, so a second acquisition by the same task
is always a bug rather than an intentional nesting to accommodate. The
concrete shape is `upsert_by_hash`'s hit branch calling the overridable
`update_thought` while still holding `_write_lock`, the dedup lock, and the
transaction the probe opened, when it opened one (see [Extension hooks
§1B.3](extension-hooks.md#1b3-a-pre-existing-restriction-update_thought-on-upsert_by_hashs-hit-branch)):
an `update_thought` override that calls back into
`create_thought(deduplicate=True)` / `get_or_create` / `upsert_by_hash` /
`bulk_store(deduplicate=True)` on that same task tries to acquire the dedup
lock a second time. Rather than blocking on itself
forever, that second acquisition raises `DedupLockReentryError` — detected
synchronously, with no bound to configure, since whether the current task
already owns the lock is known immediately.

**Say plainly what the bound is, since it is easy to state this in a way that
contradicts itself: it is a backstop that converts an unattributable hang
into an attributable error, and that backstop *can* fire on a legitimate
hold longer than the configured value — that is not a contradiction of its
purpose, it is why the value is configurable rather than fixed.** Sizing it
requires real care for exactly that reason. `bulk_store`'s batch embedding
call runs *inside* `suspend_auto_commit()`'s window (the batch's atomicity
spans insert + embed + commit as one unit), so this lock is held for that
whole call — a genuine network round trip to an embedding provider, which
can legitimately take minutes for a large batch. A bound sized for "ordinary
contention" would fire on this correct, documented usage; the default is
instead derived from the shipped default provider's own worst-case
retry-exhaustion time (three attempts at a 60s timeout each, plus backoff —
about 183 seconds) with headroom on top for larger batches and for other
providers this store cannot see the timeout/retry budget of. That headroom
lowers the odds of a false positive; it does not remove them, since this
store cannot know every caller's actual worst case.

Configure it to match your own embedding provider and batch sizes:

```python
store = SqliteEngravaCore(
    conn,
    embedding_provider=my_slow_provider,
    auto_embed=True,
    write_lock_acquire_timeout_seconds=900,  # if your batches legitimately run longer
)
```

or via `from_config(..., write_lock_acquire_timeout_seconds=...)`. Raise it
if your own provider's or batch's worst case is close to or above the
default — a legitimate hold longer than the configured value fails loudly
instead of completing, and that failure looks identical to the deadlock case
from the exception alone. Lower it only for a deployment that never drives
`bulk_store` with network-bound embedding under load.

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
- **`suspend_auto_commit()` no longer needs this discipline enforced by
  convention** — see the dedicated section above. It remains true that only
  the task that opened the window (or a write nested inside it) should be the
  one issuing writes inside it; a *different* task's write now simply waits
  rather than racing in, which is what `bulk_store()` relies on.

## Optimistic concurrency and `StaleDataError`

`update_thought`, `restore_thought`, `update_edge`, `update_action` and
`upsert_by_hash` all carry a `revision` guard. Every core row (`thought`,
`edge`, `action`) has a `revision` column, and every one of these guarded
writes both checks and increments it in the same atomic `UPDATE` — `revision
= revision + 1 WHERE id = ? AND revision = ?`. **No caller action arms this:**
unlike the `updated_cycle` guard this replaced (which nothing advanced on its
own, so an ordinary edit passed it silently), `revision` moves on *every*
guarded write to a row, automatically. `update_edge` and `update_action` carry
this guard for the first time — before this, neither could ever raise
`StaleDataError`.

**`StaleDataError` means the guarded `UPDATE` matched no row.** Two different
situations produce that, and the error does not tell them apart:

- the row's `revision` is no longer the value this call read — *any* other
  guarded write landed on this row in between, whatever field it touched; or
- **the row no longer exists** — a competing writer deleted it between this
  call's read and its write. (`ThoughtNotFoundError` / `ActionNotFoundError` /
  a `ValueError` for edges covers a row that was already missing when the call
  *started*, not one that vanished mid-call.)

The consequences are worth stating plainly:

- **An ordinary concurrent edit now raises `StaleDataError`, whatever field it
  touched**, the moment it lands on a row between this call's own read and
  write — because `revision` moved. The rejected update then writes *nothing
  at all* — no field of it reaches storage. Recover by re-reading the record
  and recomputing the change; do not replay the original `changes`.
- In the ordinary case of two genuinely concurrent tasks **on this store
  instance**, this almost never fires: the write lock ([Many async tasks, one
  store](#many-async-tasks-one-store)) already serialises them end to end, so
  the second call's own read happens strictly after the first call's write has
  committed, and its guard is captured against the post-first-call `revision`.
- What it *does* catch, for the first time, is exactly the two shapes the
  write lock cannot reach: a same-task nested write issued from inside a
  caller-owned hook (the lock is task-reentrant, so it does not block this —
  see the narrow shape noted under [Many async tasks, one
  store](#many-async-tasks-one-store)), and a write from a **second store on
  the same database file** — a second connection in this process, or a second
  process entirely. Both used to be silently lost (or, for the same-task case,
  spuriously rejected only if the competing write happened to stamp a cycle
  explicitly); both now raise `StaleDataError` before anything is written. See
  [Multiple stores, one database file](#multiple-stores-one-database-file) for
  the cross-connection case in full.

`StaleDataError` is still not a general semantic-conflict detector: it is "this
write found no row to apply to", nothing more. It says nothing about two
sessions writing contradictory content to two *different* rows, and an
`update_edge` / `update_action` call that changes nothing (a true no-op) issues
no `UPDATE` at all, so it cannot go stale — there is nothing for it to be stale
against. If your application needs to know it read the row a specific caller
last wrote — not merely that nothing else has touched it since — that
caller-visible check is not yet part of the public API (it is designed, and
scheduled for a later release, but not shipped).

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

At the top level, for the single-row calls — `create_thought(deduplicate=True)`,
`get_or_create()` and `upsert_by_hash()` — the lock is released again as soon
as the row is written, before auto-embed or the `on_store` hook run, so a slow
or stalled embedding call cannot hold the file's write lock hostage. Nested
inside the caller's own `suspend_auto_commit()` window, `_write_lock` stays
held through auto-embed and the hook too — see [Many async tasks, one
store](#many-async-tasks-one-store) for why.

**`upsert_by_hash()`'s hit branch is the one place a subclass override runs
inside this window.** Whether it is the exploratory probe's hit or the
decisive probe's hit (see [Many async tasks, one store](#many-async-tasks-one-store)),
the hit branch updates the matched row by calling the public, overridable
`update_thought` — and does so *before* the window closes, so `update_thought`
runs there with `_write_lock`, the in-process `_dedup_lock`, and the
transaction the probe opened, when it opened one, all still held. Neither hit route ever reaches the
pre-insert preparation seam. Which of the two routes predates the seam and
which one the seam introduced is stated once, in [Extension hooks
§1B.3](extension-hooks.md#1b3-a-pre-existing-restriction-update_thought-on-upsert_by_hashs-hit-branch) —
not repeated here. Either way it is not something a caller can route around:
an `update_thought` override reached this way must not call back into
`create_thought(deduplicate=True)` / `get_or_create` /
`upsert_by_hash` / `bulk_store(deduplicate=True)` on the same task
(`_dedup_lock` is not reentrant, so a second acquisition on
the same task raises `DedupLockReentryError` rather than deadlocking — see
below) and must not assume a second connection can see the update yet (the
transaction is not yet committed).

**The probe-and-row window used to have a second exposure, in-process: a
concurrent, unrelated write from a *different task on this same instance*
(`record_access()`, `update_thought()`, a delete, ...) could land while the
window was open — joining its transaction, made durable a little earlier than
that other task's own commit would have if this call's write succeeded, and
rolled back along with this call's own work if it failed.** That was the
"riding along" exposure: a connection has exactly one transaction, so once
another task's write had joined this one there was no way to tell the two
apart when deciding what to commit or roll back. An in-process
task-reentrant write lock closes it
in-process — the probe-and-row window now runs under the same `_write_lock`
every other guarded write does, so a different task's write can no longer even
start until the window has closed. It is not closed **across connections**: a
second store's write still can, which is exactly [Multiple stores, one
database file](#multiple-stores-one-database-file)'s subject.
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
all-or-nothing transaction per batch when it owns that transaction outright
(see the [API reference](api-reference.md) for the nested-caller case, where
a caught row error still commits the batch's successful prefix, and a caught
embedding failure — which only fires after every row is already inserted —
commits the whole batch instead; the lock is held for the same span either
way). `BEGIN IMMEDIATE` only moves *when* the
lock is acquired — to before the first row's probe instead of at the first
row's insert — it does not change how long the lock is held.

One case falls back to the pre-`BEGIN IMMEDIATE` behaviour: a caller already
holding an open transaction on the connection through means this store does
not manage (a raw `BEGIN` issued directly against it, with nothing written
yet). This store cannot open a second `BEGIN` inside an already-open
transaction, so it cannot acquire the cross-connection write lock up front in
that case, and falls back to whatever atomicity the caller's own transaction
provides. It still gets the in-process `_write_lock` — a
different task on *this* instance still cannot land inside the window — only
the cross-connection ordering is unavailable here. Nothing in `bulk_store` or
`suspend_auto_commit` triggers this — both are covered by the paragraph
above — it can only arise from direct, unmediated use of the underlying
connection.

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

1. **An update landing in the window is now rejected, not silently discarded.**
   The read-modify-write window described above does not stop at the store — a
   second store's edit can still land inside it, since no *in-process* lock
   could ever reach across a connection boundary — but the `revision` guard
   lives in the database itself, not in a lock, so it closes this window
   anyway: the first store's own guarded write reads `revision` fresh at write
   time, finds the second store's committed bump, matches no row, and raises
   `StaleDataError` instead of overwriting. This is a correctness improvement,
   not a support statement — multiple writers on one file remain outside what
   this store is built and tested for (below), and a caller still needs a
   retry idiom for the newly-typed rejection.
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

   That error is the loud symptom of this topology. The first of the three now
   raises `StaleDataError` (above) rather than staying silent; the second is no
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
async with await EngravaManager.from_config(config.services) as mgr:
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
| Two tasks editing **different** fields of one row | ✅ | An update writes only the columns it owns, and the write lock serialises the two calls end to end, so the second one's `revision` guard always matches. |
| A competing guarded write landing between one call's own read and write | ✅ (rejected) | `revision` bumps on every guarded write automatically; a write that lands in that window — a same-task nested call, or a second store on the file — makes the guard match no row, and the call raises `StaleDataError` rather than landing partially or silently. |
| Two tasks editing the **same** field of one row | ✅ | Still last write wins — that is the correct outcome for two genuine edits — but each call's own read and write can no longer be corrupted by the other landing mid-operation. |
| Two tasks moving one row through its state machine | ✅ | Each transition is validated against the row the *previous, now-completed* call actually left — a forbidden composite move is rejected instead of silently landing. |
| Two auto-embed provider calls for the same thought complete out of order | ✅ (rejected) | Each completion re-checks the thought's current `essence`/`content` against what it embedded before installing the vector; a completion for now-superseded content is dropped instead of overwriting a newer vector. |
| `suspend_auto_commit()` with another writer on that store | ✅ | The window now holds a task-reentrant lock for its duration; a different task's write waits instead of joining the window's transaction. |
| One store across multiple event loops | ❌ | The store's `asyncio.Lock`s bind to one loop; one store per loop. |
| Many processes reading the same file | ✅ | WAL supports concurrent readers. |
| Many stores/processes **writing** the same file | ❌ | Not supported — one writer per file; use `EngravaManager`. |

## See also

- [Deployment](deployment.md) — process model, files on disk, graceful shutdown
- [Known Limitations](known-limitations.md) — the aiosqlite proxy and write-safety notes
- [Error handling and recovery](error-handling.md) — what to do when a write fails
- [Audit Trail](audit-trail.md) — the journal whose lock is discussed above

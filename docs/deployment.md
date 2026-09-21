# Deployment

How to run Engrava in production: opening the store, the database files on disk,
multi-worker setups, and shutting down cleanly. Engrava is an embedded library —
there is no server to deploy; "deployment" means how your process opens and owns
the database.

For the concurrency model behind these recommendations, see
[Concurrency](concurrency.md). For backups, see
[Backup & Recovery](backup-and-recovery.md).

## One store per process, opened at startup

Open the store **once at process startup** and reuse it for the process's
lifetime. `from_config` opens and **owns** the connection (it applies the schema
and the right PRAGMAs), so use it as an async context manager that spans your
app's life:

```python
from engrava import SqliteEngravaCore


async def main() -> None:
    async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
        # Hold this store for the lifetime of the process / app.
        await run_app(store)
```

- **Do not open a new store per request.** Opening a store applies schema checks
  and PRAGMAs; doing it per request is wasteful and multiplies open handles to
  the same file.
- **Do not share one store across event loops.** The store holds `asyncio.Lock`s
  that bind to one loop the first time they have to wait, and reject a waiter
  from any other loop — see
  [Concurrency](concurrency.md#many-async-tasks-one-store). One store belongs to
  one running loop.
- **Share that one store across the tasks in the loop.** You do **not** need a
  pool of stores for in-process concurrency. A guarded write's own read and
  write are one critical section across tasks, so a genuinely concurrent
  task's whole operation can no longer land in the middle of another's — but
  two tasks editing the *same field* of the same row still leave only the
  later one's value in place. What this does not cover is a read-modify-write
  *your own code* spans across two separate calls. See
  [Concurrency](concurrency.md#many-async-tasks-one-store) for the exact
  guarantees and the idioms that close that gap.

## The database files on disk

In WAL mode (the default for file databases opened via `from_config`), SQLite
keeps **three** files side by side:

| File | Purpose |
|---|---|
| `engrava.db` | The main database. |
| `engrava.db-wal` | The write-ahead log — **uncommitted and recently-committed data lives here** until checkpointed. |
| `engrava.db-shm` | Shared-memory index for the WAL. |

Operational consequences:

- **Use a WAL-safe backup method** — copying only the `.db` file (or copying the
  three files non-atomically while writes continue) can capture inconsistent
  state. See [Backup & Recovery](backup-and-recovery.md) for the live-vs-stopped
  options.
- **Put them on a real local filesystem.** SQLite + WAL on networked filesystems
  (NFS, some container overlay mounts) can corrupt or fail locking. Use a local
  disk or a properly-configured volume.
- **Permissions.** The process needs read/write on the directory (SQLite creates
  and deletes `-wal`/`-shm`), not just the `.db` file. Lock the directory down to
  the service user.

## Containers

- **Mount a volume for the database directory**, not just the file — SQLite needs
  to create the `-wal`/`-shm` siblings next to the `.db`.
- Point `database.path` in your `engrava.yaml` at the mounted volume — that's the
  setting `from_config` reads. (`ENGRAVA_DB` is a **CLI-only** fallback for the
  `engrava --db` flag; it does **not** configure `from_config`, so application
  code should set `database.path`, not rely on `ENGRAVA_DB`.)
- One container instance = one writer. If you scale to multiple replicas, they
  must **not** all write the same database file — that is unsupported, not merely
  contended (see
  [multiple stores, one file](concurrency.md#multiple-stores-one-database-file)).
  Either run a single writer replica, or give each replica its own database via
  [`EngravaManager`](concurrency.md#per-service-isolation).

## Multiple workers

Engrava supports **one writing store per database file**. For multi-worker app
servers (Gunicorn/Uvicorn workers, etc.):

- **Reads scale freely** under WAL — many readers and one writer coexist, across
  processes as well as within one.
- **Route every write to one process.** Two workers writing one file is not a
  contention trade-off you can tune with `busy_timeout`; it silently loses updates
  and can duplicate deduplicated content. See
  [Concurrency → Multiple stores, one database file](concurrency.md#multiple-stores-one-database-file).
- **Per-tenant or per-worker isolation:** give each its own database file via
  [`EngravaManager`](concurrency.md#per-service-isolation) when you need
  independent writers.

## Graceful shutdown

Who closes the connection depends on how you opened the store — because the store
only closes a connection it **owns**:

- **`from_config` (owned connection).** `from_config` opens and owns the
  connection. Leaving the `async with` block closes it for you; equivalently, call
  `await store.close()`, which **closes and releases the owned connection
  cleanly**. (It does not issue an explicit WAL checkpoint — that is a
  backup/maintenance step, `PRAGMA wal_checkpoint(TRUNCATE)`, covered in
  [Backup & Recovery](backup-and-recovery.md#if-you-can-stop-or-quiesce-writers).)

  ```python
  async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
      ...
  # connection closed here

  # or, if you hold the store yourself:
  await store.close()
  ```

- **Manual `SqliteEngravaCore(conn)` (caller-managed connection).** The store does
  **not** own your connection, so `store.close()` is a **no-op** here — *you* must
  close the connection you created:

  ```python
  conn = await aiosqlite.connect("engrava.db")
  conn.row_factory = aiosqlite.Row
  store = SqliteEngravaCore(conn)
  try:
      ...
  except BaseException:
      # A failure above is what the caller needs to see; a close failure in
      # this cleanup is secondary, so it is reported rather than allowed to
      # replace it.
      try:
          await conn.close()
      except Exception as exc:  # noqa: BLE001 - never replace the real error
          print(f"warning: failed to close the database connection: {exc}")
      raise
  else:
      await conn.close()  # the caller owns and closes the connection
  ```

  A bare `async with aiosqlite.connect(...) as conn:` looks like a shortcut
  for this, but `aiosqlite.Connection.__aexit__` is an unconditional `await
  close()` — if the block above raised, a close failure there replaces the
  real error instead of the caller ever seeing it. Use the explicit
  `try`/`except`/`else` shape above whenever a close failure must never hide
  the original one.

Wire whichever applies into your framework's shutdown hook (e.g. FastAPI
`lifespan`, a signal handler) so an interrupted process still closes cleanly.

### If the worker never answers

`store.close()` waits for the background worker to finish whatever it was
doing before it can close the connection — including flushing buffered
access-tracking data first, when that feature is on. Those are two separate
waits, each bounded on its own by `close_timeout_seconds` (30 seconds by
default; tune it on `from_config()` / the manual constructor) rather than
open-ended — **not one shared budget for the call, and not the same
consequence if either one expires:**

- **The flush wait's bound expiring** is treated as an ordinary flush
  failure — the buffered access-tracking counts are best-effort telemetry
  that self-heals from a lost flush — so it quarantines nothing by itself.
  `close()` still goes on to attempt the physical close afterwards on an
  owned connection.
- **The physical-close wait's bound expiring** is the one with a lasting
  consequence: the store is left permanently unusable — every further
  operation on it, including a second `close()`, raises
  `ConnectionQuarantinedError` — because the worker's last operation never
  reported and the connection's true state can no longer be trusted.

A worker that never answers at all can therefore make one `close()` call on
an owned connection wait up to *twice* `close_timeout_seconds` (60 seconds
at the default) before returning: up to the full bound stuck in the flush
(no lasting effect on its own), then up to the full bound again stuck in the
physical close (the one that quarantines). Either way, a caller closing a
store whose worker has stopped responding still gets control back instead
of hanging forever — just not within a single `close_timeout_seconds`
window.

Bounding `close()` does not bound the **process**, and the reason is not the
worker thread. Measurement behind this bound tested that explanation and
ruled it out: a daemon and a non-daemon worker thread took the same ~20
seconds to exit, and by the time that residual delay is even observed the
worker thread has already finished. The delay instead lives inside the
interpreter's own async-runtime shutdown sequence, which runs *after* your
code — including a returned `close()` — has already handed back control, so
nothing about how `close()` waits (bounded or not) can shorten it. If a
clean, prompt process exit matters for your deployment, treat a
`ConnectionQuarantinedError` from `close()` as a signal to end the process
explicitly (rather than trusting the ordinary shutdown path) — that is a
choice about how you exit, not a fix to the underlying delay.

## See also

- [Concurrency](concurrency.md) — what one store guarantees, busy timeout, isolation
- [Backup & Recovery](backup-and-recovery.md) — WAL-safe backup and restore
- [Security](security.md) — file, network, extension, and tenant trust boundaries
- [Error handling and recovery](error-handling.md) — retry, repair, or replace
- [Configuration](configuration.md) — the YAML the deployment loads
- [Known Limitations](known-limitations.md) — filesystem and locking constraints

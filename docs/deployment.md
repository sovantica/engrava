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
  pool of stores for in-process concurrency. What the store does not do is make a
  read-modify-write atomic: two tasks editing the *same field* of the same row
  lose one of the two writes, silently, and a task that stamps `updated_cycle`
  gets a concurrent update rejected with `StaleDataError` whatever field it
  touched. See
  [Concurrency](concurrency.md#many-async-tasks-one-store) for the exact
  guarantees and the idioms that close the gap.

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
  ...
  await conn.close()  # the caller owns and closes the connection
  ```

  (Using `async with aiosqlite.connect(...) as conn:` handles this for you.)

Wire whichever applies into your framework's shutdown hook (e.g. FastAPI
`lifespan`, a signal handler) so an interrupted process still closes cleanly.

## See also

- [Concurrency](concurrency.md) — what one store guarantees, busy timeout, isolation
- [Backup & Recovery](backup-and-recovery.md) — WAL-safe backup and restore
- [Security](security.md) — file, network, extension, and tenant trust boundaries
- [Error handling and recovery](error-handling.md) — retry, repair, or replace
- [Configuration](configuration.md) — the YAML the deployment loads
- [Known Limitations](known-limitations.md) — filesystem and locking constraints

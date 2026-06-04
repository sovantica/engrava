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
- **Do not share one store across event loops.** The underlying connection is
  bound to the loop/thread that aiosqlite created it on — see
  [Known Limitations](known-limitations.md#aiosqlite-proxy-architecture). One
  store belongs to one running loop.
- **A single store safely serves many concurrent async tasks** within that one
  loop — see [Concurrency](concurrency.md). You do **not** need a pool of stores
  for in-process concurrency.

## The database files on disk

In WAL mode (the default for file databases opened via `from_config`), SQLite
keeps **three** files side by side:

| File | Purpose |
|---|---|
| `engrava.db` | The main database. |
| `engrava.db-wal` | The write-ahead log — **uncommitted and recently-committed data lives here** until checkpointed. |
| `engrava.db-shm` | Shared-memory index for the WAL. |

Operational consequences:

- **Back up all three together**, or use a WAL-safe method — copying only the
  `.db` file can lose data still in the `-wal`. See
  [Backup & Recovery](backup-and-recovery.md).
- **Put them on a real local filesystem.** SQLite + WAL on networked filesystems
  (NFS, some container overlay mounts) can corrupt or fail locking. Use a local
  disk or a properly-configured volume.
- **Permissions.** The process needs read/write on the directory (SQLite creates
  and deletes `-wal`/`-shm`), not just the `.db` file. Lock the directory down to
  the service user.

## Containers

- **Mount a volume for the database directory**, not just the file — SQLite needs
  to create the `-wal`/`-shm` siblings next to the `.db`.
- Point `database.path` (or `ENGRAVA_DB`) at the mounted volume.
- One container instance = one writer. If you scale to multiple replicas, they
  must **not** all write the same database file (see
  [multi-process](concurrency.md#multiple-processes)). Either run a single writer
  replica, or give each replica its own database via
  [`EngravaManager`](concurrency.md#per-service-isolation).

## Multiple workers

Engrava follows SQLite's single-writer model. For multi-worker app servers
(Gunicorn/Uvicorn workers, etc.):

- **Reads scale freely** under WAL — many readers and one writer coexist.
- **Concentrate writes.** Heavy write fan-out across many OS processes hitting the
  same file is out of scope; see [Concurrency → Multiple processes](concurrency.md#multiple-processes).
- **Per-tenant or per-worker isolation:** give each its own database file via
  [`EngravaManager`](concurrency.md#per-service-isolation) when you need
  independent writers.

## Graceful shutdown

If you opened the store via `from_config` as a context manager, leaving the
`async with` block closes the connection for you. If you constructed the store
some other way, close it explicitly on shutdown so the WAL is checkpointed and
handles are released:

```python
await store.close()
```

Wire this into your framework's shutdown hook (e.g. FastAPI `lifespan`, a signal
handler) so an interrupted process still closes cleanly.

## See also

- [Concurrency](concurrency.md) — the single-writer model, busy timeout, isolation
- [Backup & Recovery](backup-and-recovery.md) — WAL-safe backup and restore
- [Configuration](configuration.md) — the YAML the deployment loads
- [Known Limitations](known-limitations.md) — filesystem and locking constraints

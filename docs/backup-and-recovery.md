# Backup & Recovery

Two ways to back up an Engrava database, what each one covers, and how to restore
and verify. The most important thing to know up front: a **logical snapshot does
not include the audit journal**, and a **naive file copy in WAL mode can lose
data** — both are explained below.

## Two kinds of backup

| Method | What it captures | Portable across versions? |
|---|---|---|
| **Logical snapshot** (`engrava snapshot`) | Thoughts, edges, embeddings, and actions as JSONL records | Yes — it's data, not file format |
| **Physical file backup** | The exact database file(s) — *everything*, including the audit journal | Tied to the SQLite file format (very stable) |

Pick the logical snapshot for portability and selective restore; pick a physical
backup when you need a byte-exact copy (including the journal) or point-in-time
file recovery.

## Logical snapshot and restore

```bash
engrava --db engrava.db snapshot -o backup.jsonl   # export
engrava --db fresh.db   restore  -i backup.jsonl   # import into a fresh db
```

The snapshot is JSONL: a metadata header line, then one record per
thought / edge / embedding / action.

> **A snapshot does NOT include the audit journal.** The `journal_entry` table —
> the tamper-evident hash chain — is **not** exported by `engrava snapshot`, and
> therefore is **not** recreated by `restore`. A database restored from a snapshot
> starts with an **empty journal**: the data is intact, but its prior audit
> history is gone. If audit continuity matters, use a **physical file backup**
> (which copies the journal verbatim), not a logical snapshot. See
> [Audit Trail](audit-trail.md).

`restore` options worth knowing: `--clear` to wipe the target first,
`--skip-embeddings` / `--re-embed` to control embedding handling, and
`--service` for multi-service targets.

## Physical file backup (WAL-safe)

Engrava runs in **WAL mode**, where recently-written data lives in the `-wal`
file until it is checkpointed into the main `.db`. **Copying only `engrava.db`
while the database is in use can therefore miss data still in the WAL.** Use one
of these WAL-safe approaches instead:

**1. Checkpoint, then copy.** Force the WAL into the main file, then copy it:

```bash
sqlite3 engrava.db "PRAGMA wal_checkpoint(TRUNCATE);"
cp engrava.db engrava.db.bak
```

**2. Copy all three files together.** If you can't checkpoint (the app is
writing), copy `engrava.db`, `engrava.db-wal`, and `engrava.db-shm` as a set, so
the WAL travels with the database:

```bash
cp engrava.db engrava.db-wal engrava.db-shm /backup/
```

**3. `VACUUM INTO`** — write a clean, compacted copy without locking the source
for the whole copy:

```bash
sqlite3 engrava.db "VACUUM INTO 'engrava-backup.db';"
```

**4. The SQLite Online Backup API** — for a hot backup of a live database from
your own code (via the `sqlite3` backup API), if you need streaming/incremental
copies.

> Avoid a bare `cp engrava.db backup.db` on a database that is being written —
> that is the one method that can silently lose WAL-resident data.

## Restoring

- **From a snapshot:** `engrava --db <target> restore -i backup.jsonl`. Restore
  into a **fresh** database (optionally `--clear` an existing one). Remember the
  journal is not restored.
- **From a physical backup:** stop the process, put the backed-up file(s) in
  place, and start again. If you copied the three WAL files, restore all three
  together; if you checkpointed before copying, the single `.db` is sufficient.

### Verify a restore

After restoring, confirm the database is readable and the counts look right:

```bash
engrava --db restored.db info     # reports counts; confirms the schema is readable
```

For a snapshot restore you can compare `info` counts against the source. If you
rely on the audit journal and restored from a **physical** backup, also re-run
journal verification (see [Audit Trail](audit-trail.md)) to confirm the chain is
intact.

## Multi-service backups

With [`EngravaManager`](concurrency.md#per-service-isolation), each service is its
own database file under the shared data directory. Back them up the same way —
either snapshot each service (`snapshot --service <name>`) or take a WAL-safe
physical copy of each `<name>.db` (plus its `-wal`/`-shm`). Because services are
independent files, you can back up, restore, or delete one without touching the
others.

## See also

- [Audit Trail](audit-trail.md) — the journal that snapshots exclude
- [Concurrency](concurrency.md) — why WAL needs a WAL-safe backup
- [Data Lifecycle](data-lifecycle.md) — retention, erasure, and VACUUM
- [Upgrade Guide](upgrade.md) — backing up before an upgrade

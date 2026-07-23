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

`restore` options worth knowing (see the [CLI reference](cli.md#restore) for the
full list): `--clear` to wipe the target first, `--skip-embeddings` / `--re-embed`
to control embedding handling, and `--service` for multi-service targets.
When `--clear` encounters a persisted sqlite-vec index, restore drops the derived
table transactionally and the next configured open rebuilds it. Keep
`engrava[vec]` installed for that virtual-table reset.

### Embedding handling during restore

A normal restore imports the embedding rows carried by the snapshot.
`--skip-embeddings` discards those rows and restores text/graph/action data
without vectors.

`--re-embed` also discards the snapshot vectors, then generates new vectors for
every imported thought. The provider must come from the configuration passed via
`--config`. A top-level provider supports single-database restore and acts as the
fallback for configured services:

```yaml
database:
  path: ./fresh.db
embeddings:
  provider: sentence-transformer
  model: sentence-transformers/all-MiniLM-L6-v2
```

```bash
engrava --db fresh.db --config engrava.yaml restore -i backup.jsonl --clear --re-embed
```

A service can inherit that top-level provider or override it:

```yaml
embeddings:
  provider: ollama
  model: nomic-embed-text
services:
  data_dir: ./data
  default_service: main
  configs:
    main:
      embeddings:
        provider: sentence-transformer
        model: sentence-transformers/all-MiniLM-L6-v2
```

```bash
engrava --config engrava.yaml restore -i backup.jsonl --service main --clear --re-embed
```

`services.configs.<name>.embeddings` has precedence over the top-level fallback.
Without either provider, `--re-embed` fails before importing records. Retain the
snapshot embeddings or use `--skip-embeddings` when no provider is available.
The re-embed and its model/dimension/prefix identity update commit atomically.
Use a fresh target or `--clear`: a target with existing embeddings is rejected
without `--clear` so surviving vectors cannot be mislabeled as the new corpus.
The general `--clear` sqlite-vec reset described above also protects this path.
`--skip-embeddings` and `--re-embed` are mutually exclusive.

### Trust boundary

`restore` is designed for **snapshots you produced yourself** with `engrava
snapshot` — your own trusted backups. That is the supported input boundary.

Restore does not blindly trust the file. Every line must be a JSON object, and
each record targeting a known table (thought, edge, embedding, action) is
validated against a fixed, code-owned schema: its columns must be a known subset
of that table's columns, the required columns must be present and non-null, and
each value must match its column's type (an imported embedding vector must be
valid base64 — the base64 check applies only when embeddings are actually
imported, not under `--skip-embeddings` or `--re-embed`, which never read the
stored vectors). A record whose type is not one of those tables is skipped, so a newer snapshot restored
into an older build ignores record types it does not understand rather than
failing. The whole restore runs as **one transaction**: if any record is
malformed or carries a bad value, the transaction is rolled back and **no rows
are written** — not even a `--clear` that ran first, and not the records that
validated before the bad one. Column names never come from the file; the SQL
uses a fixed, built-in column set, so a hand-edited or corrupted key cannot
alter the statements that run.

What restore does **not** do is vouch for the *content* of a snapshot from
someone else. The thought text, metadata, and other values are restored
verbatim, so a snapshot obtained from an untrusted third party is untrusted data
in your database — treat it with the same caution as any external import.
Restore a snapshot you trust the origin of; if you must ingest an external one,
review it first. Restore also assumes the file is **not being modified while it
runs**; point it at a completed backup, not a snapshot that is still being
written.

## Physical file backup (WAL-safe)

Engrava runs in **WAL mode**, where recently-written data lives in the `-wal`
file until it is checkpointed into the main `.db`. A plain file copy is only safe
under specific conditions, so choose the method by whether the database is
**live** (being written) or **stopped**.

### If the database is live (writers running)

A file copy of a database under active writes is **not reliable** — the `.db` and
`-wal` change during the copy and can be captured inconsistently. Use a method
that produces an internally consistent copy *without* stopping writers:

**SQLite Online Backup API** — a hot, consistent backup driven from your own code
via Python's `sqlite3` backup API (`source.backup(dest)`). This is the
recommended way to back up a running database, and it supports incremental copies.

**`VACUUM INTO`** — writes a fresh, consistent, compacted copy of the database to
a new file. SQLite serialises it correctly against ongoing activity:

```bash
sqlite3 engrava.db "VACUUM INTO 'engrava-backup.db';"
```

Both produce a single clean `.db` you can store or move; neither requires copying
the `-wal`/`-shm` files.

### If you can stop or quiesce writers

When you can take the database offline (or guarantee no writes for the duration),
a file copy is safe — preferably after folding the WAL back into the main file:

**Checkpoint, then copy the single file:**

```bash
# with no writers active:
sqlite3 engrava.db "PRAGMA wal_checkpoint(TRUNCATE);"
cp engrava.db engrava.db.bak
```

**Or copy the file set** (`engrava.db` + `-wal` + `-shm`) **as one atomic unit** —
e.g. via a filesystem-level snapshot (LVM, ZFS, a cloud volume snapshot) that
captures all three at the same instant. A plain `cp` of the three files of a
*live* database is **not** atomic and can still be inconsistent; only do the
multi-file copy when writers are stopped or behind a consistent snapshot.

> **Do not** rely on a bare `cp engrava.db backup.db` — or even a non-atomic
> `cp engrava.db engrava.db-wal engrava.db-shm ...` — while the database is being
> written. For a live database use the Online Backup API or `VACUUM INTO`.

## Restoring

- **From a snapshot:** `engrava --db <target> restore -i backup.jsonl`. Restore
  into a **fresh** database (optionally `--clear` an existing one). Remember the
  journal is not restored. For `--re-embed`, pass `--config` with a top-level
  provider or a per-service provider override.
- **From a physical backup:** stop the process, put the backed-up file in place,
  and start again. A backup made with the Online Backup API, `VACUUM INTO`, or a
  checkpoint-then-copy is a single self-contained `.db`. If instead you captured a
  multi-file filesystem snapshot, restore `engrava.db`, `engrava.db-wal`, and
  `engrava.db-shm` together as the unit they were snapshotted in.

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

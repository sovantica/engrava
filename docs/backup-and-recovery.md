# Backup & Recovery

Two ways to back up an Engrava database, what each one covers, and how to restore
and verify. The most important thing to know up front: a **logical snapshot does
not include the audit journal**, and a **naive file copy in WAL mode can lose
data** — both are explained below.

## Two kinds of backup

| Method | What it captures | Portable across versions? |
|---|---|---|
| **Logical snapshot** (`engrava snapshot`) | Thoughts, edges, embeddings, and actions as JSONL records | Data, not file format — but restore validates every record against a fixed column set per table, so it is portable only to a build whose recognized tables accept the same columns |
| **Physical file backup** | The exact database file(s) — *everything*, including the audit journal | Tied to the SQLite file format (very stable) |

Pick the logical snapshot for selective restore across compatible schema
versions; pick a physical backup when you need the audit journal preserved or
point-in-time file recovery. A filesystem-level copy of the complete file set
can be byte-exact; `VACUUM INTO` and the Online Backup API instead produce a
consistent logical copy that is not guaranteed byte-identical to the source
(see below).

## Logical snapshot and restore

```bash
engrava --db engrava.db snapshot -o backup.jsonl   # export
engrava --db fresh.db   restore  -i backup.jsonl   # import into a fresh db
```

The snapshot is JSONL: a metadata header line, then one record per
thought / edge / embedding / action. The export reads the metadata header and
all four tables inside one read transaction opened before the first read
(`src/engrava/cli/main.py:1241`), so the snapshot is a consistent point-in-time
view rather than four independent scans that a concurrent writer could
interleave with.

> **A snapshot does NOT include the audit journal.** The `journal_entry` table —
> the tamper-evident hash chain — is **not** exported by `engrava snapshot`, and
> therefore is **not** recreated by `restore`. What that leaves in the
> *target's* journal depends on which of the three restore shapes you used:
>
> - **A fresh target** (the database file did not exist yet) starts with an
>   empty journal — there was no prior journal for it to have.
> - **`restore --clear`** empties `journal_entry` along with the four core
>   tables it wipes, so the journal ends empty too. Otherwise it would keep
>   describing thoughts the clear had just discarded.
> - **A restore without `--clear`** merges into an existing database — and if
>   that database's journal is non-empty, **it now refuses any record that
>   collides** with an existing row on a primary key or `UNIQUE` constraint,
>   rolling the whole restore back rather than writing anything from that
>   snapshot. Restoring a one-thought snapshot back into the journalled
>   database it came from failed like this:
>
>   ```text
>   Error: Restore refused: snapshot line 2 collides with an existing row (matching primary key or UNIQUE constraint), and the target's journal_entry table is not empty. Replacing that row would leave the audit trail describing data this merge discarded, while 'engrava verify' kept reporting the chain as valid. Re-run with --orphan-journal-entries to allow the merge and accept that gap, or with --clear to discard the journal along with the data.
>   ```
>
>   exit code `1`. This **journalled-merge collision gate** is conservative,
>   not precise — it refuses *any* uniqueness collision once a journal exists,
>   including one on a row the journal never described, rather than trying to
>   work out which collisions are actually dangerous. It never triggers when
>   the target's journal is empty, which is the ordinary case: journaling is
>   opt-in and the CLI never turns it on itself, so a merge restore into a
>   database that has never enabled it behaves exactly as it always has.
>
>   **`--orphan-journal-entries`** allows the merge anyway, and restores that
>   prior behaviour: the merged-in records are inserted directly and are not
>   themselves journalled, and a journal entry can be orphaned even when no
>   incoming ID collides with one the journal already describes. Within the
>   stock core schema, an incoming edge with a fresh `edge_id` but the same
>   `(from_thought_id, to_thought_id, edge_type)` triple as a journalled edge
>   replaces it through the table's own UNIQUE constraint — no ID collision
>   needed. And replacing a thought whose **own** ID does collide cascades the
>   delete, by foreign key, to that thought's edges, embeddings, and actions —
>   rows whose IDs never appeared in the snapshot. An action only has a
>   journal entry to orphan once it has been updated at least once:
>   `create_action` writes no journal entry, only `update_action` does, so a
>   freshly created action that was never updated cascades away with nothing
>   stale left behind. A database carrying an extension-installed or
>   user-defined trigger on these tables can open further routes: the
>   extension migration runner applies a migration's SQL verbatim, including
>   `CREATE TRIGGER` (see [Extensions](extensions.md#migration-files)), so a
>   trigger that deletes a row elsewhere in the schema can orphan its journal
>   entry on a restore insert that collides with nothing the target holds. In
>   every case the journal entries describing the earlier row stay behind
>   unchanged; `verify` still reports the chain as **valid**, even though
>   those entries no longer describe what the database now holds — confirmed
>   directly: restoring the collision above under `--orphan-journal-entries`
>   still leaves `engrava verify` reporting `Journal integrity OK — 1 entries
>   verified.`.
>
> If audit continuity matters, use a **physical file backup** (which copies
> the journal verbatim), not a logical snapshot. See
> [Audit Trail](audit-trail.md).

`restore` options worth knowing (see the [CLI reference](cli.md#restore) for the
full list): `--clear` to empty the target's four core tables and its journal
first (not `_metadata`, `extension_schema_versions`, or extension-owned
tables — see above), `--skip-embeddings` / `--re-embed`
to control embedding handling, `--orphan-journal-entries` to allow a merge that
would otherwise be refused by the collision gate above, and `--service` for
multi-service targets. When `--clear` encounters a persisted sqlite-vec index,
restore drops the derived table transactionally and the next configured open
rebuilds it. Keep `engrava[vec]` installed for that virtual-table reset.

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

Restore does not blindly trust the file. Every non-blank line must be a JSON
object — blank lines are skipped — and each record targeting a known table (thought, edge, embedding, action) is
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
a file copy is safe once it captures every committed change. Committed changes
can still sit in the `-wal` file, so either copy the main file only after a
checkpoint has folded the WAL back into it, or copy the whole file set (below).
Stopping writers alone does not guarantee the checkpoint completes: an existing
reader can still pin WAL frames, so check its result before copying rather than
copying unconditionally.

**Checkpoint, then copy the single file only if the checkpoint fully completed:**

```bash
# With no writers (and, ideally, no readers) active. The PRAGMA prints
# busy|log_frames|checkpointed_frames. busy = 0 means the checkpoint
# completed, so the main file alone holds every committed change. A non-zero
# busy means it did not complete (an open reader is a common cause), and the
# main file may then lack changes still in the WAL. `test -f` comes first
# because the sqlite3 CLI creates an empty database at a path that does not
# exist. The copy goes to a temporary name and replaces the backup only once
# complete, so a failed copy leaves an earlier backup intact. Each step runs
# only if the one before it succeeded.
test -f engrava.db &&
  [ "$(sqlite3 engrava.db 'PRAGMA wal_checkpoint(TRUNCATE);' | cut -d'|' -f1)" = "0" ] &&
  cp engrava.db engrava.db.bak.tmp &&
  mv engrava.db.bak.tmp engrava.db.bak ||
  { echo "no backup made: missing file, incomplete checkpoint, or failed copy" >&2; false; }
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

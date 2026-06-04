# CLI reference

Engrava ships an `engrava` command-line tool for inspecting, querying, and
maintaining a database without writing code. This page documents every command
and option.

```bash
engrava [GLOBAL OPTIONS] COMMAND [ARGS]...
```

## Global options

These apply to every command and go **before** the command name:

| Option | Values / type | Default | Description |
|---|---|---|---|
| `--db` | path | `./engrava.db` | Path to the SQLite database. Falls back to the `ENGRAVA_DB` env var, then the default. |
| `--config` | path | — | Path to `engrava.yaml`. Falls back to the `ENGRAVA_CONFIG` env var. |
| `--format` | `table` \| `json` \| `csv` | `table` | Output format for commands that print records. |
| `--verbose` | flag | off | Enable verbose output. |
| `--help` | flag | — | Show help and exit (works on the root and on every command). |

**Environment variables.** `ENGRAVA_DB` and `ENGRAVA_CONFIG` are CLI fallbacks for
`--db` and `--config` respectively; the explicit flag always wins
(`--db` > `ENGRAVA_DB` > `./engrava.db`).

```bash
export ENGRAVA_DB=/data/engrava.db
engrava info                       # uses /data/engrava.db
engrava --db other.db info         # flag overrides the env var
```

## Commands

| Command | Purpose |
|---|---|
| [`info`](#info) | Show a metrics snapshot for the database. |
| [`query`](#query) | Run a MindQL query. |
| [`snapshot`](#snapshot) | Export the whole database to a JSONL snapshot. |
| [`restore`](#restore) | Restore a database from a JSONL snapshot. |
| [`gc`](#gc) | Garbage-collect archived thoughts (and optionally expired ones). |
| [`migrate`](#migrate) | Run pending schema migrations. |
| [`export`](#export) | Export thoughts to a portable JSON file. |

### `info`

Shows a metrics snapshot (counts, etc.) for the current database. Takes no
command-specific options.

```bash
engrava --db engrava.db info
```

Use this after an upgrade or a restore to confirm the database is readable and
the counts look right.

### `query`

Executes a [MindQL](mindql.md) query and prints the results in the chosen
`--format`.

```bash
engrava query "MQL"
```

The `MQL` string is a positional argument. It accepts `FIND`, `COUNT`, `SELECT`,
or registered extension commands:

```bash
engrava query "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"
engrava query "COUNT thoughts WHERE priority = 'P1'"
engrava --format json query "SELECT thought_id, essence FROM thought LIMIT 5"
```

### `snapshot`

Exports the **entire** database to a JSONL snapshot (one record per line).

| Option | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output` | path | `<service>.snapshot.jsonl` (derived) | Output JSONL file path. |
| `--service` | name | default service | In multi-service mode, the service to snapshot. |

```bash
engrava --db engrava.db snapshot -o backup.jsonl
engrava snapshot --service tenant_a            # multi-service
```

> A snapshot exports `thought`, `edge`, `embedding`, and `action` records — but
> **not** the audit journal (`journal_entry`). See
> [Backup & Recovery](backup-and-recovery.md) for what this means and when to use
> a physical file backup instead.

### `restore`

Restores a database from a JSONL snapshot produced by `snapshot`.

| Option | Type | Default | Description |
|---|---|---|---|
| `-i`, `--input` | path | **required** | JSONL snapshot file to restore. |
| `--clear` | flag | off | Clear existing data before restoring. |
| `--skip-embeddings` | flag | off | Import without embedding records. |
| `--re-embed` | flag | off | Re-embed all thoughts via the target provider, ignoring source embeddings. |
| `--service` | name | default service | In multi-service mode, the service to restore into. |

`--skip-embeddings` and `--re-embed` are **mutually exclusive** — passing both
fails with:

```
Error: --re-embed and --skip-embeddings are mutually exclusive.
```

Use `--re-embed` when the target uses a different embedding model than the
snapshot (the embeddings would otherwise be incompatible — see
[Troubleshooting → EmbeddingModelMismatchError](troubleshooting.md#embeddingmodelmismatcherror-when-opening-an-existing-database)).
Use `--skip-embeddings` to import text only.

```bash
engrava --db fresh.db restore -i backup.jsonl
engrava --db fresh.db restore -i backup.jsonl --clear --re-embed
```

> Restore recreates thoughts, edges, embeddings, and actions, **not** the audit
> journal — a restored database starts with an empty journal.

### `gc`

Garbage-collects `ARCHIVED` thoughts and their orphaned edges. With `--expired`
it also runs the TTL expiry cleanup first.

| Option | Type | Default | Description |
|---|---|---|---|
| `--dry-run` | flag | off | Show what would be deleted without changing anything. |
| `--expired` | flag | off | Also run expiry cleanup (archive or delete per `ttl.strategy`) before collecting. |

```bash
engrava --db engrava.db gc                 # delete ARCHIVED thoughts + orphaned edges
engrava --db engrava.db gc --expired       # run expiry cleanup first (per strategy)
engrava --db engrava.db gc --expired --dry-run
```

The behaviour of `gc --expired` depends on `ttl.strategy`: with `delete` it
removes expired rows and then collects pre-existing archived rows; with the
default `archive` it archives the expired rows and stops (it does not collect
them in the same pass). See
[Data lifecycle → running cleanup](data-lifecycle.md#running-cleanup).

### `migrate`

Runs pending schema migrations (ensures the core tables exist and are
up to date). Takes no command-specific options. Safe to run after an upgrade.

```bash
engrava --db engrava.db migrate
```

### `export`

Exports thoughts to a portable JSON file (with edges and metadata). Unlike
`snapshot` (JSONL, whole-database, for backup/restore), `export` writes a single
indented JSON document and can be filtered by lifecycle status.

| Option | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output` | path | `<db>.export.json` (derived) | Output JSON file path. |
| `--status` | lifecycle status | all | Only export thoughts with this `lifecycle_status` (e.g. `ACTIVE`). |

```bash
engrava --db engrava.db export -o thoughts.json
engrava --db engrava.db export --status ACTIVE
```

## Journal verification

There is **no `engrava verify` command** in this version. To verify the
[audit journal](audit-trail.md)'s hash chain, use the Python API:

```python
result = await store.journal.verify_integrity()
print(result.is_valid)
```

## See also

- [MindQL](mindql.md) — the query language `engrava query` runs
- [Backup & Recovery](backup-and-recovery.md) — snapshot/restore vs physical backup
- [Data Lifecycle](data-lifecycle.md) — what `gc` and `gc --expired` do
- [Configuration](configuration.md) — the `engrava.yaml` that `--config` loads

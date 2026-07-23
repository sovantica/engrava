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
| `--verbose` | flag | off | Compatibility flag. It is accepted and stored in CLI configuration, but currently does not change logging or command output. |
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
| [`verify`](#verify) | Verify the audit journal's hash chain. |
| [`query`](#query) | Run a MindQL query. |
| [`snapshot`](#snapshot) | Export the whole database to a JSONL snapshot. |
| [`restore`](#restore) | Restore a database from a JSONL snapshot. |
| [`gc`](#gc) | Garbage-collect archived thoughts (and optionally expired ones). |
| [`migrate`](#migrate) | Run pending schema migrations. |
| [`export`](#export) | Export thoughts to a portable JSON file. |

## Extension discovery

The executable automatically discovers installed CLI extensions. Before Click
parses and dispatches the requested subcommand, every `engrava` invocation scans
the `engrava.cli` entry-point group and loads objects that provide Click
commands. This happens even when you invoke a built-in command such as `info` or
`--help`.

The built-in `query` command performs a second scan of
`engrava.extensions`. It loads each discovered manifest and registers its
`mindql_extensions` for that query. This query-time scan does not apply manifest
schema migrations.

Both scans import and may execute code from installed packages. Failed entry
points are skipped with a warning log. There is currently no global option,
environment variable, or YAML setting to disable CLI discovery. Use the CLI in
an environment containing only trusted packages; see
[Security and Trust Boundaries](security.md#extensions-hooks-and-migrations).

`--verbose` does not make these warnings or other logs more verbose in the
current release. The flag is retained for CLI compatibility but has no output or
logging effect.

## Service resolution

The `--service` option on `snapshot` and `restore` resolves the same way in both
commands:

| `--service` | Services config loaded? | Result |
|---|---|---|
| `--service NAME` (explicit) | either | Targets service **NAME**. Its database is found/created in the services `data_dir` if a config is loaded, otherwise in the **parent directory of `--db`** (i.e. `<parent-of-db>/NAME.db`). |
| omitted | yes | Falls back to `services.default_service`. |
| omitted | no | Operates on the single `--db` database (not service mode). |

In short: an explicit `--service` works even without a services config (using
`--db`'s directory as the data directory), while omitting it only enters
multi-service mode when a services config is present.

### `info`

Shows a metrics snapshot (counts, etc.) for the current database. Takes no
command-specific options.

```bash
engrava --db engrava.db info
```

Use this after an upgrade or a restore to confirm the database is readable and
the counts look right.

### `verify`

Verifies the [audit journal](audit-trail.md)'s hash chain. It walks every
recorded `journal_entry` in sequence order, recomputes each SHA-256 hash, and
checks the parent-hash linkage. Takes no command-specific options. The chain is
verified **regardless of whether journaling is currently enabled**, so a journal
recorded in an earlier session is still auditable.

```bash
engrava --db engrava.db verify
# Journal integrity OK — 128 entries verified.

engrava --db engrava.db --format json verify
# {"valid": true, "entries_checked": 128, ...}
```

The exit code is **`0`** when the chain verifies, **`1`** when it does not (the
text output names the first broken `sequence`, and the JSON output carries
`first_invalid_sequence` / `error_message`) or when the database is missing —
so it drops straight into a CI job, a pre-backup hook, or a monitoring check. An
empty or absent journal verifies as valid with `entries_checked: 0`.

Read the [Security model](audit-trail.md#security-model--guarantees) first: this
is a keyless in-file chain, so it detects accidental corruption and naive edits,
not a chain-aware actor who rewrites the whole `.db`.

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
engrava query "FIND thoughts WHERE valid_now"          # only currently-valid facts
```

The bi-temporal `valid_now`, `valid_at`, `valid_within`, and `valid_between`
predicates work here too — see [MindQL](mindql.md) for their full semantics.

### `snapshot`

Exports the **entire** database to a JSONL snapshot (one record per line).

| Option | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output` | path | derived (see below) | Output JSONL file path. |
| `--service` | name | see below | The service to snapshot (multi-service mode only). |

**Default output path** depends on the mode:

- **Single database:** `<db-stem>.snapshot.jsonl` next to the database — e.g.
  `--db engrava.db` → `engrava.snapshot.jsonl` (the `.db` suffix is replaced).
- **Multi-service:** `<data_dir>/<service>.snapshot.jsonl`.

**`--service`** resolves in three ways (see [Service resolution](#service-resolution)):

- **Explicit `--service NAME`** targets that service even with no services config
  — the service database is looked up/created in the data directory, which is the
  services config's `data_dir` if one is loaded, otherwise the **parent directory
  of `--db`**.
- **Omitted, with a services config loaded** → falls back to
  `services.default_service`.
- **Omitted, with no services config** → snapshots the single `--db` database.

```bash
engrava --db engrava.db snapshot -o backup.jsonl
engrava --db engrava.db snapshot               # -> engrava.snapshot.jsonl
engrava --db /data/engrava.db snapshot --service tenant_a   # -> /data/tenant_a.snapshot.jsonl
engrava --config engrava.yaml snapshot --service tenant_a   # data_dir from config
```

> A snapshot exports every column of the `thought`, `edge`, `embedding`, and
> `action` records — including the bi-temporal `valid_from` / `valid_until`
> fields — but **not** the audit journal (`journal_entry`). See
> [Backup & Recovery](backup-and-recovery.md) for what this means and when to use
> a physical file backup instead.

### `restore`

Restores a database from a JSONL snapshot produced by `snapshot`.

| Option | Type | Default | Description |
|---|---|---|---|
| `-i`, `--input` | path | **required** | JSONL snapshot file to restore. |
| `--clear` | flag | off | Clear existing data before restoring. |
| `--skip-embeddings` | flag | off | Import without embedding records. |
| `--re-embed` | flag | off | Re-embed all thoughts via a configured target service's provider, ignoring source embeddings. Requires config-backed service mode. |
| `--service` | name | see below | The service to restore into. |

`--service` resolves exactly as for [`snapshot`](#service-resolution): an explicit
`--service NAME` targets that service even without a services config (its database
resolves in the services `data_dir`, or the **parent directory of `--db`** when no
config is loaded); omitted with a services config falls back to
`services.default_service`; omitted with no services config restores into the
single `--db` database.

`--skip-embeddings` and `--re-embed` are **mutually exclusive** — passing both
fails with:

```
Error: --re-embed and --skip-embeddings are mutually exclusive.
```

Use `--re-embed` when a **configured service** uses a different embedding model
than the snapshot. Load an `engrava.yaml` whose `services` configuration
resolves the target service and sets
`services.configs.<name>.embeddings.provider`. The CLI does not forward the
top-level `embeddings` section as a service default. The restore opens that
service through `EngravaManager`, discards snapshot embedding rows, and
generates replacement vectors with the target provider.

The single-database `--db` branch constructs a bare store without an embedding
provider. Consequently, `engrava --db fresh.db restore ... --re-embed` is not a
supported re-embed path and fails before restoring snapshot records, even if a config
path is also present but does not resolve restore into service mode. An explicit
`--service` without a configured provider fails for the same reason. Use
`--skip-embeddings` to import text without vectors, or restore normally to retain
the snapshot vectors.

```bash
engrava --db fresh.db restore -i backup.jsonl
engrava --db fresh.db restore -i backup.jsonl --clear --skip-embeddings
engrava --config engrava.yaml restore -i backup.jsonl --service main --clear --re-embed
```

When `services.default_service` names the configured target, `--service main`
may be omitted. A normal restore into an attached configured service checks the
snapshot model metadata against the target provider; on mismatch, choose
`--re-embed` or `--skip-embeddings`. See
[Troubleshooting → EmbeddingModelMismatchError](troubleshooting.md#embeddingmodelmismatcherror-when-opening-an-existing-database).

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

Use [`engrava verify`](#verify) to verify the [audit journal](audit-trail.md)'s
hash chain from the shell (exit `0` = intact, `1` = broken or missing database).
The equivalent Python API is `store.verify_journal()` (the store-level
convenience) or `store.journal.verify_integrity()` (via the writer directly):

```python
result = await store.verify_journal()
print(result.valid)
```

## See also

- [MindQL](mindql.md) — the query language `engrava query` runs
- [Backup & Recovery](backup-and-recovery.md) — snapshot/restore vs physical backup
- [Data Lifecycle](data-lifecycle.md) — what `gc` and `gc --expired` do
- [Configuration](configuration.md) — the `engrava.yaml` that `--config` loads

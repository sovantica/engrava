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
| `--verbose` | flag | off | Emit DEBUG logs from Engrava modules to stderr for this invocation. Command data on stdout keeps its selected format. |
| `--no-extensions` | flag | off | Prevent loading both `engrava.cli` and `engrava.extensions` entry points. `ENGRAVA_DISABLE_EXTENSIONS=1` provides the same control. |
| `--help` | flag | — | Show help and exit (works on the root and on every command). |

**Environment variables.** `ENGRAVA_DB` and `ENGRAVA_CONFIG` are CLI fallbacks for
`--db` and `--config` respectively; the explicit flag always wins
(`--db` > `ENGRAVA_DB` > `./engrava.db`).
`ENGRAVA_DISABLE_EXTENSIONS=1` activates `--no-extensions` before command
resolution, including for root help and built-in commands.

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
| [`snapshot`](#snapshot) | Export thoughts, edges, embeddings, and actions to a JSONL snapshot (not the audit journal). |
| [`restore`](#restore) | Restore a database from a JSONL snapshot. |
| [`gc`](#gc) | Garbage-collect archived thoughts (and optionally expired ones). |
| [`migrate`](#migrate) | Run pending schema migrations. |
| [`export`](#export) | Export thoughts to a portable JSON file. |

## Extension discovery

The executable discovers installed CLI extensions lazily. Built-in commands are
resolved without scanning `engrava.cli`. Root help scans that entry-point group
so installed commands appear in the command list, and an otherwise unknown
command triggers the scan so an installed extension command can resolve.

The built-in `query` command performs a second scan of
`engrava.extensions`. It loads each discovered manifest and registers its
`mindql_extensions` for that query. This query-time scan does not apply manifest
schema migrations.

Both scans import and may execute code from installed packages. Failed entry
points are skipped with a warning log. Put `--no-extensions` before the command,
or set `ENGRAVA_DISABLE_EXTENSIONS=1`, to prevent **both** entry-point groups from
being scanned or loaded:

```bash
engrava --no-extensions --help
ENGRAVA_DISABLE_EXTENSIONS=1 engrava --db memory.db info
engrava --no-extensions --db memory.db query "SELECT thought_id FROM thought LIMIT 5"
```

This control is specific to automatic CLI entry-point loading. It does not
disable explicit manifest paths or library-side `manifests.discover` settings
used by an application-created store. Continue to install only trusted packages;
see
[Security and Trust Boundaries](security.md#extensions-hooks-and-migrations).

`--verbose` enables DEBUG logging for the `engrava` logger hierarchy for the
duration of the command. Logs go to stderr, leaving JSON/CSV/table command data
on stdout.

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

## Schema-version checks

Every built-in command now checks the database's schema version before it
acts, and the two kinds of command are held to different rules:

| Kind | Commands | On a schema below head | On a schema above head |
|---|---|---|---|
| Destructive | `gc`, `restore` | **Refuses**, exit `1`, names `engrava migrate` | **Refuses**, exit `1` |
| Read | `info`, `verify`, `export`, `snapshot`, a `query` that parses as `FIND`/`COUNT`/`SELECT` | Warns on stderr and runs anyway | **Refuses**, exit `1` |

A destructive command never deletes rows through an engine that does not
understand the schema it is deleting from — that gap is how a deleted thought
could come back on an unmigrated database (see
[Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated)).
A read command is allowed to attempt anyway, because refusing an ordinary read
over a pending migration would be a worse failure than the one this replaces —
but it is never silent about the gap. `query` classifies by the **parsed**
command, not by the fact that you typed `query`: a registered extension
command can write, so it is refused like a destructive one whenever the
schema is behind.

`migrate` is not in either row — running the pending migrations (or refusing
to, when the database is a populated pre-history schema `ensure_schema()`
cannot safely bootstrap, or is stamped above this build's head version) is
its entire job.

```bash
$ engrava --db old.db gc
Database schema is at version 11; this engrava build's head version is 20. Run 'engrava migrate' before running 'gc' on it.
$ echo $?
1
```

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

Exports thoughts, edges, embeddings, and actions to a JSONL snapshot (one
record per line) — **not** the audit journal; see below.

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
| `--re-embed` | flag | off | Re-embed all thoughts via the target provider, ignoring source embeddings. Requires `--config` with top-level or per-service embeddings. |
| `--orphan-journal-entries` | flag | off | Allow a merge restore (no `--clear`) into a target whose `journal_entry` table is non-empty to replace a colliding row anyway. Without it, such a collision is refused. |
| `--service` | name | see below | The service to restore into. |

For any `restore --clear`, an existing sqlite-vec table is dropped in the same
transaction as the canonical rows. The next sqlite-vec-enabled open recreates
and backfills it. Because SQLite must load the virtual-table module before it can
remove that table safely, install `engrava[vec]` before clearing a database that
already contains a persisted sqlite-vec index.

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

Use `--re-embed` when the target should use a different embedding model than the
snapshot. In single-database mode, the provider comes from the top-level
`embeddings` section of the file passed with `--config`. In service mode,
`services.configs.<name>.embeddings` takes precedence; when no override exists,
the same top-level `embeddings` section is the fallback. Restore discards source
embedding rows, generates replacement vectors with the resolved provider, and
atomically replaces the stored model, dimension, document-prefix fingerprint,
and query-prefix pairing. A target that already contains embeddings requires
`--clear`; without it, restore refuses to relabel vectors that are not part of
the snapshot. The sqlite-vec reset described above prevents stale index rows
from surviving that replacement.

If neither level declares a provider, restore fails before importing records and
names the missing configuration. An explicit `--service` without a config-backed
provider fails for the same reason. Use `--skip-embeddings` to import text
without vectors, or restore normally to retain the snapshot vectors.

```bash
engrava --db fresh.db restore -i backup.jsonl
engrava --db fresh.db restore -i backup.jsonl --clear --skip-embeddings
engrava --db fresh.db --config engrava.yaml restore -i backup.jsonl --clear --re-embed
engrava --config engrava.yaml restore -i backup.jsonl --service main --clear --re-embed
```

When `services.default_service` names the configured target, `--service main`
may be omitted. Every restore — regardless of `--re-embed` or
`--skip-embeddings` — first checks that the target's *own* pre-existing
`embedding` rows agree with its *stored* embedding model, or with each other
when it has none: neither flag imports the snapshot's vectors as-is, so
neither can excuse a target that is already inconsistent going in. A normal
restore (neither `--re-embed` nor `--skip-embeddings`) additionally checks the
`model_name` and `dimension` every `embedding` row in the snapshot declares
against that same reference before inserting it — never against a configured
provider, which a plain restore never resolves, and never against the
snapshot's metadata header, which is not proof of anything the rows do not
already say for themselves. On a mismatch, restore fails before committing
anything, naming both identities; choose `--re-embed` to regenerate vectors for
the target's own model, or `--skip-embeddings` to import without vectors — an
already-inconsistent target itself is not fixed by either flag and needs its
own repair (or `--clear`) first.
Restoring embeddings into a target that starts with neither a stored model nor
any embeddings of its own writes the snapshot's declared identity as the
target's lock — the same `_metadata` write embedding directly into it would
make, though restore trusts the snapshot's declaration rather than computing
and measuring a vector itself.

**This check cannot see the document or query prefix the corpus was built
with** (see [Embeddings guide → Asymmetric prefixes for instruction-tuned
models](guides/embeddings.md#asymmetric-prefixes-for-instruction-tuned-models))
— a snapshot carries neither, only the vectors and a declared model name and
dimension for each. A target that already had a prefix
fingerprint locked before this restore keeps it unchanged (restore never
touches it), but a target that locks fresh from the snapshot's own vectors
(see above) records no prefix at all, regardless of what the source corpus
actually used. If the source corpus was built with a non-empty document
prefix, confirm that out of band before pointing a prefix-aware provider at
a restored target — restore itself cannot tell you.

> Restore recreates thoughts, edges, embeddings, and actions, **not** the audit
> journal. A **fresh target** therefore starts with an empty journal, and
> **`--clear`** empties the journal along with the data it wipes. A restore
> **without `--clear`** merges into the target — and if that target's journal
> is non-empty, **it refuses any record that collides** with an existing row
> (primary key or `UNIQUE` constraint) and rolls the whole restore back,
> rather than letting the merge silently replace it. Pass
> **`--orphan-journal-entries`** to allow the merge anyway; that restores the
> pre-gate behaviour, where a merge can orphan journal entries even when no
> incoming ID collides with one the journal describes — a duplicate
> `(from_thought_id, to_thought_id, edge_type)` triple replaces an existing
> edge, and replacing a thought cascades to that thought's own edges,
> embeddings, and actions — and `verify` still reports the chain as **valid**
> even though it no longer matches the data. The gate never applies to an
> empty journal, which is the ordinary case since journaling is opt-in.
> See [Backup & Recovery](backup-and-recovery.md#logical-snapshot-and-restore)
> for the full breakdown.

### `gc`

Garbage-collects `ARCHIVED` thoughts together with every edge touching one on
either end — including edges whose other end is still live — their embeddings and
the actions sourced from them, then reconciles the vector index by removing every
`vec0` row no `embedding` row owns. With `--expired` it also runs the TTL expiry
cleanup first.

> **`gc` now refuses on a database that is not on the current schema.** If
> `--db` (or the resolved service database) is below or above this build's
> head schema version, `gc` deletes nothing and exits `1` naming
> `engrava migrate` — see [Schema-version checks](#schema-version-checks)
> above. If a `gc` that used to run cleanly now exits non-zero, that is this
> refusal, not a regression in what it collects: run `engrava migrate` first.

| Option | Type | Default | Description |
|---|---|---|---|
| `--dry-run` | flag | off | Show what would be deleted without changing anything. |
| `--expired` | flag | off | Also run expiry cleanup (archive or delete per `ttl.strategy`) before collecting. |

```bash
engrava --db engrava.db gc                 # delete ARCHIVED thoughts + their edges/embeddings/actions
engrava --db engrava.db gc --expired       # run expiry cleanup first (per strategy)
engrava --db engrava.db gc --expired --dry-run
```

The behaviour of `gc --expired` depends on `ttl.strategy`: with `delete` it
removes expired rows and then collects pre-existing archived rows; with the
default `archive` it archives the expired rows and stops **only if it archived at
least one** — collecting the rows it just archived would defeat the soft-retire.
With no expired rows to archive it falls through to collecting pre-existing
`ARCHIVED` rows. See
[Data lifecycle → running cleanup](data-lifecycle.md#running-cleanup).

> **`gc` refuses to delete on a `vec0`-indexed store without the vector extra.**
> If the database carries an `embedding_vec` table and `sqlite-vec` cannot be
> loaded — most commonly because `engrava[vec]` is not installed, though an
> unsupported build, an OS error or a SQLite error fail the same way — a pass that
> is about to physically delete stops **before deleting anything** and exits `1`
> with:
>
> ```text
> Error: This database has a sqlite-vec index, and collecting thoughts without removing their vectors would strand them in it. Install 'engrava[vec]' and retry.
> ```
>
> Removing the rows without removing their vectors would strand those vectors in
> an index nothing can then reach them through. Only *deleting* passes are
> refused: `--dry-run` is never refused, and neither is a run with nothing to
> delete. Read the archive strategy carefully, though — `gc --expired` under the
> default `ttl.strategy: archive` stops after archiving **only when it actually
> archived something**. With no expired rows to archive it falls through to the
> archived-collection pass, which *is* refused when there are archived rows to
> collect. Install the extra (`pip install 'engrava[vec]'`) and retry.

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

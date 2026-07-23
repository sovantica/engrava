# Upgrade Guide

Most users can run `pip install --upgrade engrava` safely. Database migration is
automatic on first connection, and the upgrade path is validated in CI before
minor releases.

## What Happens During Upgrade

- Core schema migration runs automatically on the first `ensure_schema()` call.
- Existing data is preserved; migrations are forward-only.
- Extension schema migrations are also applied when an installed extension
  declares them.

In practice, most applications do not need a separate migration step. If your
app already calls `ensure_schema()` during startup, that call performs the
upgrade.

## Rolling upgrades (multiple workers)

If several processes share one database file, whether you can do a **rolling**
upgrade (start new-version workers while old-version workers are still running)
depends on whether the new version changes the schema.

How migrations work: the core schema is versioned by SQLite's `PRAGMA
user_version`. On the first `ensure_schema()`, Engrava runs each pending
`vN → vN+1` step **inside a transaction** (forward-only). Most steps are
**additive** (new columns, tables, and indexes), but some rebuild a table in
place (create a new table, copy rows, drop the old, rename) — so the on-disk
shape of a table can change across a migration.

What that means for a rolling deploy:

- **Patch upgrades that don't change `user_version`** (e.g. `0.3.0 → 0.3.1`) make
  no schema change. Old and new workers can run side by side; roll them at will.
- **Minor upgrades that run migrations are not guaranteed to be
  backward-readable.** Once the first new-version worker calls `ensure_schema()`
  and a table is rebuilt, an old-version worker may no longer match the new
  on-disk shape. Do **not** run old and new workers concurrently across such an
  upgrade.

Recommended procedure for a schema-changing (minor) upgrade:

1. **Back up** the database (see [Before You Upgrade](#before-you-upgrade)).
2. **Quiesce writers** — stop the old workers (or take a brief maintenance
   window) so no old-version process writes during the migration.
3. **Run the migration once** — let a single new-version process call
   `ensure_schema()` (or run `engrava migrate`) to completion.
4. **Start the new workers** against the migrated database.

When you are unsure whether a target release changes the schema, treat it as
schema-changing and follow the quiesce procedure — it is always safe. The
[compatibility matrix](#compatibility-matrix) notes which listed upgrades change
the schema.

## Before You Upgrade

These steps are recommended, not required:

```bash
# Checkpoint the WAL first so the copy is complete, then back up.
sqlite3 my-data.db "PRAGMA wal_checkpoint(TRUNCATE);"
cp my-data.db my-data.db.bak
pip install --upgrade engrava
```

- Create a copy of the SQLite database file before the upgrade. In WAL mode a
  bare `cp` of just the `.db` can miss data still in the `-wal` file — checkpoint
  first (above), or copy `my-data.db` together with `my-data.db-wal` and
  `my-data.db-shm`. See [Backup & Recovery](backup-and-recovery.md) for all the
  WAL-safe options.
- Review [CHANGELOG.md](../CHANGELOG.md) for breaking changes and database notes.
- If you ship custom extensions, make sure their schema migrations are included
  in the version you are about to install.

## After You Upgrade

Use the CLI to confirm the upgraded database opens correctly:

```bash
engrava --db my-data.db info
engrava --db my-data.db migrate
```

- `engrava info` confirms the database is readable and reports current counts.
- `engrava migrate` is safe to run after upgrade; it re-checks that schema is up to date.
- `engrava gc` is optional if you want to remove archived or expired data after
  the upgrade. Note that `gc` deletes rows but does **not** shrink the database
  file — freed pages return to SQLite's free-list. To reclaim file size, run
  `VACUUM`. See [Data lifecycle → reclaiming disk space](data-lifecycle.md#reclaiming-disk-space).

## If Migration Fails

Migration errors should include the failing SQL or the extension responsible for
the failure.

Recommended recovery order:

1. Restore from your `.bak` copy.
2. Re-run the upgrade in a clean virtual environment.
3. Open an issue with the error message, `engrava info` output, and whether the
   failure happened in core schema migration or an extension migration.

When reporting the problem, redact file paths and application-specific content
if needed, but keep the SQL error and schema version details intact.

## Downgrade Policy

Downgrades are not supported for `0.x` releases. Migrations are forward-only.

If you must move data into an older version, use an export/import flow instead
of opening the upgraded database file directly:

```bash
engrava --db my-data.db snapshot -o backup.snapshot.jsonl
engrava --db new-old-version.db restore -i backup.snapshot.jsonl
```

> **Note:** a snapshot exports thoughts, edges, embeddings, and actions, but
> **not** the audit journal (`journal_entry`). A database restored from a
> snapshot starts with an empty journal. If you need the audit history preserved,
> take a physical file backup instead — see
> [Backup & Recovery](backup-and-recovery.md).

## Compatibility Matrix

| From | To | Supported | Notes |
|---|---|---|---|
| 0.2.0 | 0.2.2 | Yes | Patch-level upgrade, no dedicated new extension migration layer |
| 0.2.2 | 0.3.0 | Yes | Minor upgrade with extension migration tracking and upgrade CI coverage |
| 0.3.0 | 0.3.1 | Yes | Patch-level upgrade; no schema change (`user_version` unchanged) — safe to roll across workers |
| 0.3.x | 0.4.0 | Yes | **Schema-changing** minor upgrade — adds the valid-time columns (additive, zero data loss). Back up first and follow the [rolling-upgrades](#rolling-upgrades-multiple-workers) note |
| 0.4.x | 0.5.0 | Yes | **Schema-changing** minor upgrade (`user_version` 14 → 18), although the library API is drop-in. **Breaking for MCP-server users only:** the `engrava[mcp]` extra and the in-engrava `engrava-mcp` command are removed — the server moved to the standalone [`engrava-mcp`](https://github.com/sovantica/engrava-mcp) package (see the 0.4 → 0.5 note) |
| 0.5.0 | 0.6.0 | Yes | **Schema-changing** minor upgrade (`user_version` 18 → 20), with two additive columns. Default retrieval now excludes archived thoughts, and wrong-dimension query vectors raise a typed error. Back up, quiesce shared-store workers, migrate once, and review the [0.5 → 0.6 notes](#05---06) |

For any upgrade not listed, the rule of thumb is: **patch** upgrades within a
`0.x.*` line do not change the schema and are low-risk; **minor** upgrades
(`0.X` → `0.(X+1)`) may run schema migrations — back up first and read the
[rolling-upgrades](#rolling-upgrades-multiple-workers) note below.

## Version Notes

### 0.5 -> 0.6

Version 0.6 is a **schema-changing minor upgrade**. Do not roll it across old and
new workers sharing one database file. Back up the database, stop the 0.5
workers, let one 0.6 process run `ensure_schema()` (or `engrava migrate`) to
completion, and then start the 0.6 workers. The migration is automatic and
forward-only.

The core schema advances from `user_version = 18` to `user_version = 20` in two
additive steps:

| Step | Change | Existing-row behavior |
|---|---|---|
| 18 → 19 | Adds `edge.metadata_json` | Existing edges read back `metadata == {}`. |
| 19 → 20 | Adds nullable `thought.archived_at` | An older hygiene-archived row has no wall-clock timestamp and fails closed: it is not auto-GC-eligible while the wall-clock restore window is active. |

Neither step drops a table or rewrites user content. The columns are added
automatically with a neutral default or `NULL`. A physical pre-upgrade backup is
still required because downgrades and reverse migrations are unsupported.

Review these behavior changes before deployment:

- **Archived thoughts leave default retrieval.** `search_similar`, `search_fts`,
  `search_hybrid`, and `recall` now exclude `LifecycleStatus.ARCHIVED` rows by
  default. Pass `include_archived=True` for an archive-search call, or use
  `restore_thought(...)` to return a thought to `ACTIVE`. `list_thoughts` and
  `count_thoughts` retain their lifecycle-neutral behavior unless you filter
  them explicitly.
- **Query-vector dimensions fail loudly.** `search_similar` and the vector arm
  reject a vector whose length does not match the store dimension with
  `VectorDimensionMismatchError` (an `EngravaError` subclass). Code that caught
  the previous incidental `ValueError`, or relied on a wrong-dimension all-zero
  vector returning `[]`, must catch the typed error instead. Empty, all-zero, or
  non-finite vectors remain a graceful empty result and increment
  `vector_arm_degradation_count`.
- **Malformed FTS syntax gets one safe retry.** A failed expert `MATCH`
  expression is retried through bare-query normalization before the FTS arm is
  dropped; every failed first attempt increments `fts_match_failure_count`.
- **Recency has two explicit modes.** Existing `current_cycle` behavior remains,
  with an optional runtime `cycle_provider`. Callers without a cognitive cadence
  can select transaction-time recency with `recency_now`; supplying both explicit
  references raises `RecencyModeConflictError` rather than combining clocks.
- **Configuration validation is uniform.** Invalid values in supported config
  sections are rejected consistently whether the store is built from YAML or
  through the corresponding typed construction path. Fix invalid legacy values
  rather than relying on a path that previously skipped validation.
- **Enabled hygiene gains conservative wall-clock guards.** Memory Hygiene is
  still off by default. For a store that already enabled it in 0.5, omitted new
  fields resolve to a seven-day minimum inactivity age and a 30-day wall-clock
  restore window; archival also requires an active usage-history signal. Pass a
  fixed `now` when replaying a selection, and review the new defaults before the
  first 0.6 hygiene run. Existing hygiene archives without `archived_at` fail
  closed and are not auto-GC'd while the wall-clock window is active.

The new edge-metadata, derived-record, cycle-provider, and transaction-time
recency surfaces are additive. Existing stores do not enable automatic
derivation or a cycle provider merely by being migrated.

### 0.4 -> 0.5

The library upgrade is drop-in: `pip install --upgrade engrava` and your normal
startup. The schema migration runs automatically on first open as usual.

**Schema change (additive, zero data loss).** 0.5 steps the core schema from
`user_version = 14` to `user_version = 18` in four migrations:

| Step | Change | Existing-row behavior |
|---|---|---|
| 14 → 15 | Adds the composite `edge(edge_type, to_thought_id)` lookup index | Query-plan improvement only; no row changes. |
| 15 → 16 | Adds nullable `thought.action_outcome_score` and `idx_action_source_thought` | Existing thoughts have no action-outcome aggregate until linked terminal actions produce one. |
| 16 → 17 | Adds nullable `thought.provenance` plus session/actor JSON expression indexes | Existing thoughts have no captured provenance. |
| 17 → 18 | Adds `thought.pinned` and nullable `thought.archived_at_cycle` | Existing thoughts read as `pinned=False`; no row is treated as hygiene-archived. |

All four steps run automatically inside the migration transaction on first
`ensure_schema()`. They add columns or indexes without dropping or rewriting
user content.

**Breaking change for MCP-server users.** The Model Context Protocol server moved
out of `engrava` into its own package, **`engrava-mcp`**. Removed from `engrava`
in 0.5.0:

- the `engrava[mcp]` optional-dependency extra, and
- the in-engrava `engrava-mcp` console command.

**Migrate as follows:**

| Before (0.4) | After (0.5) |
|---|---|
| `pip install "engrava[mcp]"` | `pip install engrava-mcp` (or `uvx engrava-mcp`) |
| `engrava-mcp` (installed by engrava) | `engrava-mcp` (installed by the `engrava-mcp` package) |
| client `mcp.json`: `command: engrava-mcp` | client `mcp.json`: `command: uvx`, `args: ["engrava-mcp"]` |

- **Watch out:** `pip install "engrava[mcp]"` against engrava 0.5 **does not fail**
  — pip ignores the now-unknown extra and installs bare `engrava`, so you may think
  the server was installed when it was not. Install `engrava-mcp` instead.
- Update any pinned requirement strings (`engrava[mcp]>=...`) to depend on
  `engrava-mcp`, not just reinstall.
- **Store config is unchanged:** the server still reads `ENGRAVA_MCP_CONFIG`
  (an `engrava.yaml`) or `ENGRAVA_DB_PATH`, and `ENGRAVA_MCP_READ_ONLY` still gates
  the write tools.
- `engrava-mcp` depends on `engrava>=0.5`, so if an app on engrava 0.4.x and an
  `engrava-mcp` share one database file, upgrade the app to 0.5 too.
- Don't run the old in-engrava `engrava-mcp` and the new `engrava-mcp` package
  against the same store at once during the cutover.

See the [`engrava-mcp` package](https://github.com/sovantica/engrava-mcp) for the
full server documentation (install, client config, tools/resources/prompts,
read-only mode).

### 0.3 -> 0.4

Version 0.4 introduces a second time axis — **valid time** (`valid_from` /
`valid_until`), the period during which a fact is true in the world — alongside
the existing transaction time (`created_at`). See
[The Bi-temporal Model](bitemporal.md) for the full feature, the four query
predicates, and `invalidate`. From an upgrade standpoint, the change is
**additive and automatic**:

**The migration runs on first open, with zero data loss.** The first time a
0.4 process calls `ensure_schema()` (most apps already do this at startup), the
core schema steps forward from `user_version = 12` (the 0.3 schema) to
`user_version = 14` in **two additive steps** (12 → 13 adds the valid-time
columns and their indexes; 13 → 14 adds the hot-path indexes), each inside a
transaction. `pip install --upgrade engrava` plus your normal startup is all that
is required:

```bash
pip install --upgrade engrava
# your app's existing ensure_schema() call performs the migration on first open
```

What the migration does:

- **Adds two nullable columns** — `valid_from` and `valid_until` — to both the
  `thought` and `edge` tables, plus supporting indexes. Nothing is dropped or
  rewritten beyond adding columns; **no row is lost or modified in content**,
  and the row counts are unchanged.
- **Backfills existing thoughts conservatively.** A thought that has a recorded
  `created_at` gets `valid_from` backfilled from it (its valid-time lower bound
  starts where its transaction time started). `valid_until` is always left open
  (`NULL`).
- **Leaves legacy rows and all edges open-from.** A thought with no `created_at`
  (a legacy row) keeps `valid_from = NULL`. **Every existing edge** keeps both
  bounds `NULL` — the edge table has no calendar timestamp to source a date
  from, so the migration honestly leaves them open rather than fabricating one.
- **Adds four hot-path indexes.** A second additive step creates indexes that
  back the equality filters and the sort column hit on every common read
  (edges by their target thought, a thought's embedding by owner, listing
  thoughts in recency order, and filtering thoughts by type). This is a
  pure index addition — **no row is read, modified, or removed**, and the row
  counts are unchanged. The connection is also opened with `synchronous=NORMAL`
  and `busy_timeout=5000` (a PRAGMA-only change with no on-disk effect). Like
  the valid-time step, it runs automatically on first open with zero data loss.

**Structured (MindQL) queries are unchanged.** A query that uses no temporal
predicate behaves exactly as it did on 0.3. And because a `NULL` bound is treated
as an **open interval end** (−∞ / +∞), the open-from rows above still match
`valid_now` and `valid_at` queries — an un-dated fact is treated as "valid since
the beginning of time", not as "excluded". So adopting valid time is incremental:
you can start annotating new facts whenever you like, and the old ones keep
surfacing in temporal queries until you choose to bound them.

**Search behavior changes (no migration, but worth knowing).** Two 0.4 fixes to
keyword/full-text search are not schema changes but do change results:

- **Bare full-text queries now `OR`-match** instead of `AND`-matching, so a
  natural-language query that returned *nothing* on 0.3 (because no document
  contained *every* word) may now return results. This is the intended fix; if you
  relied on strict all-words matching, use uppercase `AND` or a quoted phrase
  explicitly. See [Keyword query syntax](search.md#keyword-query-syntax-fts).
- Stored embeddings are **not** re-computed by the upgrade — the full-content
  embedding fix and the `max_seq_length` fix take effect only when a thought is
  re-written (re-created, or its `essence`/`content` updated), at which point it is
  re-embedded with the corrected input. Existing vectors are untouched until then.

> **Honest note about edges.** Because the upgrade cannot invent a `valid_from`
> for an edge that never had a date, every edge migrated from 0.3 carries
> `valid_from = NULL`. That is the correct "open lower bound", so those edges
> still match `valid_now` / `valid_at`. They will **not** match `valid_between`
> (which requires real bounds on both ends) until you set their bounds
> explicitly. This is expected, not a defect.

**MCP server.** 0.4 shipped an optional Model Context Protocol server behind an
in-tree `engrava[mcp]` extra. As of 0.5 the server moved to its own package,
[`engrava-mcp`](https://github.com/sovantica/engrava-mcp) (`uvx engrava-mcp`); the
`engrava[mcp]` extra and the in-engrava `engrava-mcp` command are removed. See the
0.4 → 0.5 notes below for migration.

This is a schema-changing minor upgrade, so follow the
[rolling-upgrades](#rolling-upgrades-multiple-workers) procedure (back up,
quiesce writers, migrate once, start new workers) if you run multiple processes
against one database file.

### 0.3.0 -> 0.3.1

- Patch release: **no schema change** (`user_version` stays at its 0.3.0 value),
  so it is safe to roll across multiple workers without a quiesce.

### 0.2.2 -> 0.3.0

- Extension schema migration tracking is now part of the upgrade path.
- Upgrade-path CI validates the `0.2.2 -> main` flow before release.
- Release notes and `CHANGELOG.md` now carry a dedicated `Database Changes`
  section for schema-affecting releases.

### Dreaming Defaults

Future releases that change dreaming defaults should document them here. For
example, a benchmark-facing default such as `dreaming_cycles=1` belongs in this
guide once it becomes part of a shipped release.

## Release Communication Rule

Any release that changes schema behavior must include a `Database Changes`
section in [CHANGELOG.md](../CHANGELOG.md) and in GitHub release notes.

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

- **Patch upgrades are not supposed to change `user_version`** (e.g. `0.3.0 →
  0.3.1`). That is a rule this project follows when cutting a release, not a
  guarantee anything currently checks: no workflow, script, or test ties a
  schema-version bump to whether the release is a patch or a minor. Old and
  new workers are expected to run side by side on a patch upgrade on that
  basis. If you want certainty rather than a policy before rolling workers
  across a release, compare `PRAGMA user_version` before and after the
  upgrade yourself — that covers the recorded **core** schema version only.
  `ensure_schema()` also applies any pending **extension** schema migrations,
  tracked separately in `extension_schema_migrations`, which a `user_version`
  comparison does not reveal; check an installed extension's own migration
  history too before trusting mixed-version workers are safe.
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

  **If the upgraded install dropped the vector extra, `gc` will refuse rather
  than run.** A database that carries an `embedding_vec` table from an earlier
  run *with* `sqlite-vec` cannot be collected without it: a `gc` pass that is
  about to physically delete stops **before deleting anything** and exits `1`
  with `Install 'engrava[vec]' and retry`, because removing the rows without
  removing their vectors would strand those vectors in the index. The same
  refusal follows any other reason `sqlite-vec` fails to load — an unsupported
  build, an OS error, a SQLite error — not only a missing extra. Reinstall as
  `pip install 'engrava[vec]'` and retry. `--dry-run` is never refused, and
  neither is a run with nothing to delete; but `gc --expired` under the default
  `ttl.strategy: archive` stops after archiving only when it actually archived
  something, and with no expired rows falls through to the archived-collection
  pass, which *is* refused when there are archived rows to collect. See
  [CLI reference → `gc`](cli.md#gc).

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
| 0.5.0 | 0.6.0 | Yes | **Schema-changing** minor upgrade (`user_version` 18 → 20), with two additive columns. Default retrieval now excludes archived thoughts, and wrong-dimension query vectors raise a typed error. An edge `decay_multiplier` of `0.0` no longer reads back as `1.0`, and a later update no longer rewrites it to `1.0` — values a 0.5.x update already overwrote stay overwritten. Back up, quiesce shared-store workers, migrate once, and review the [0.5 → 0.6 notes](#05---06) |
| 0.6.x | 0.7.0 | Yes | No *database* schema change, but `EngravaMetrics.schema_version` moves `1 → 2` (see below). **Behaviour change:** when the resolved recency weight is `0.0` **and** a cognitive-cycle reference (`current_cycle`, explicit or via `cycle_provider`) is present, the query-less fallback path now treats recency as fully off instead of still decaying by cycle — which can change result order for stores with heterogeneous thought priorities. `recency_now` (transaction-time) callers are unaffected; that axis was already correct. Also in this release: two new `EngravaError` subclasses (`WriteContentionError`, `WriteLockTimeoutError`) can now come out of the dedup and guarded-write paths; `gc --dry-run` now names everything the real run deletes (edges, embeddings, and actions, not only orphaned edges); a deleted thought's vector can no longer resurface through search on a database that has not run the core-12 migration; and a corrupt or truncated database file now makes the CLI exit with an error instead of hanging. Review the [0.6 → 0.7 notes](#06---07) |

For any upgrade not listed, the rule of thumb is: **patch** upgrades within a
`0.x.*` line do not change the schema and are low-risk; **minor** upgrades
(`0.X` → `0.(X+1)`) may run schema migrations — back up first and read the
[rolling-upgrades](#rolling-upgrades-multiple-workers) note below.

## Version Notes

### 0.6 -> 0.7

**Breaking behaviour change: a resolved recency weight of zero now fully
disables cognitive-cycle recency, including on the query-less fallback path.**
No schema migration is involved.

**Who is affected.** Two conditions, both required: the resolved
`recency_weight` is `0.0`, **and** a cognitive-cycle reference is present — an
explicit `current_cycle`, or one resolved through a configured
`cycle_provider`. The resolved weight is `0.0` in three cases: an explicit
`recency_weight=0.0`; a directly constructed `SqliteEngravaCore` with no
`SearchConfig` at all, which resolves an *omitted* `recency_weight` to `0.0` —
those callers are affected without ever passing the argument, so do not rule
yourself out just because your code never mentions `recency_weight`; or a
supplied `SearchConfig` whose `default_recency_weight` is `0.0`. (A default
`SearchConfig()` and `from_config` with no override both resolve to `0.1`, not
`0.0` — only construction with no `SearchConfig` object at all defaults to
`0.0`, which is exactly why an omitted argument can mean "off" on one
construction path and "on" on another.) Within that population, the change
is on the *fallback* path — FTS inactive or skipped, and vector search
inactive, for that particular query. **Two separate things change there, on
two separate conditions, and they are not the same population:** result
*order* changes only on a store where another signal is also active
(commonly `priority_weight`, which defaults to `0.05`); the raw per-row
*score* `HybridSearchResult` reports changes regardless of whether any other
signal is active at all — see the executed, zero-other-signal case below.

**Who is not affected: `recency_now` callers, on both counts, executed.**
If you use `recency_now` (transaction-time recency) instead of
`current_cycle`, this fix changed nothing that reaches you — order or score.
The diff that produced this fix touches only the `current_cycle` branch;
the `transaction_now` branch is byte-for-byte the same function on both
sides. Executed directly against both branches with identical inputs: the
`transaction_now` path returns the same scores in the same order on 0.6.x
and 0.7. A resolved weight of `0.0` with no cognitive-cycle reference
present at all (no explicit `current_cycle`, no configured `cycle_provider`)
was likewise already a no-op on both revisions — the changed branch is never
reached because there is no reference for it to gate.

**What changed.** On 0.6.x, a resolved `recency_weight` of `0.0` correctly
kept `'recency'` out of `HybridSearchResult.backends_used`, but the fallback
path still computed a cycle-decayed score for each row — when a cognitive-cycle
reference was present — and used it as that row's score ahead of the sort: the
weight gated the label, not the number. On 0.7, the cognitive-cycle axis is
genuinely inert whenever the resolved weight is `0.0`: every row on this
path gets a flat `0.0` *recency contribution* to its score, matching the
transaction-time axis, which was already correct on 0.6.x. The row's
*total* score is this flat `0.0` only when nothing else contributes to it —
see the split below for what happens when another signal, such as priority,
is also active.

**Why this can reorder results, not just shrink a score.** The fallback path
adds every active signal's contribution and re-sorts. With the recency term no
longer contributing anything, whichever row has the higher priority boost can
now win outright — previously, a fresh-but-low-priority row could still beat a
stale-but-high-priority one on the strength of an undecayed recency score that
a resolved weight of `0.0` was supposed to have turned off.

Concrete example, on defaults (`current_cycle=100`, `recency_half_life=50`,
`priority_weight=0.05`), with the recency weight resolved to `0.0`:

| Row | 0.6.x score (leaked recency) | 0.7 score (fixed) |
|---|---|---|
| fresh, `Priority.P4` (`updated_cycle=100`) | `1.0` — **won** | `0.0` |
| stale, `Priority.P2` (`updated_cycle=0`) | `0.85` | `0.6` — **wins now** |

**The scores themselves change even with no other signal active at all —
order does not, but a reader who trusts the number is still affected.**
Executed directly (`priority_weight=0.0` too, so nothing else contributes):
the same two rows above, with every other weight at `0.0`, score `[1.0,
0.25]` on 0.6.x and `[0.0, 0.0]` on 0.7 — the raw, un-weighted cycle-decayed
value the fallback path used to return as each row's score, unconditionally,
whenever a cognitive-cycle reference was present. Order is unchanged (both
rows keep the same relative position — `1.0`/`0.25` and `0.0`/`0.0` both rank
the fresher row first), so a caller that only reads relative order is
unaffected here; a caller that reads, thresholds on, or logs the actual
`HybridSearchResult` score is affected even with every other signal off,
because `1.0` and `0.25` are not `0.0` and `0.0`.

**What to do.** If your recency weight resolves to `0.0` — by omitting
`recency_weight` on a directly constructed store, by passing it explicitly, or
through a zero `SearchConfig.default_recency_weight` — **and** you also supply
a cognitive-cycle reference (an explicit `current_cycle`, or a configured
`cycle_provider`): re-check result *order* after upgrading on any query that
has heterogeneous thought priorities, another signal active, and falls
through to the FTS-inactive/vector-inactive fallback path — it may now favor
priority where it previously favored an unintended recency leak. Separately,
and regardless of whether another signal is active: re-check any code that
reads, thresholds on, or logs the raw fallback-path score itself. On 0.7 the
*recency contribution* to that score is always exactly `0.0` where it used
to leak a cycle-decayed value — but the score's *total* only drops to `0.0`
when nothing else contributes to it (no other active signal, as above);
otherwise the total keeps whatever other contribution was already part of
it, unchanged, as the `0.85` → `0.6` stale row in the table shows: its
recency term went to `0.0`, its `0.6` priority contribution did not move.
If you actually want cycle-decayed ranking, give the resolved
`recency_weight` a positive value instead of relying on the previous
behavior. If you use `recency_now` instead of `current_cycle`, none of this
applies to you.

**Typed break: `EngravaMetrics.schema_version` moves from `Literal[1]` to
`Literal[2]`.** No *database* migration is involved — this is the metrics
snapshot value-object returned by `await store.metrics()`, not the SQLite
core schema (`PRAGMA user_version`), which is unaffected by this change.

**Why the version moved.** `EngravaMetrics` gained a new field, `measured:
bool`, so a caller can tell a genuinely empty, measured store apart from the
zero-filled placeholder `metrics()` returns when `MetricsConfig.enabled` is
`False`. Nothing else in the snapshot stated that either way: `storage.db_bytes`
reads `0` for a disabled store and for a measured in-memory one, and
`search_latency.sample_count` reads `0` for a disabled store and for a
measured store that has served no searches — see
[Observability → Configuration](observability.md#configuration) for the
measured numbers behind both. Adding a field is a shape change, so the version
moves with it.

**Who is affected.** This reaches more callers than the `Literal` type alone
suggests:

- Code that pattern-matches or asserts on `schema_version == 1` (or the
  `Literal[1]` annotation itself) breaks at type-check time or at runtime,
  whichever it does.
- Code that **persists** a serialized `EngravaMetrics` snapshot — to a file,
  a message queue, a metrics store's own history — and later deserializes it
  expecting exactly the old field set is affected too, even though it never
  wrote or checked `schema_version` itself: a snapshot written before this
  upgrade has no `measured` key, and a snapshot written after does.
- A script that only shells out to `engrava info` (or `engrava --format json
  info`) and parses its output is affected too, even though it never imports
  `engrava` or the `EngravaMetrics` type at all: the CLI builds that output
  from `asdict(metrics)`, so its JSON gains the `measured` key and its
  `schema_version` value moves to `2` right along with the library object.
- Code that only reads individual fields (`m.thoughts.total`, `m.storage.db_bytes`,
  etc.) without inspecting `schema_version` is unaffected.
- A **subclass of `EngravaMetrics`** that appends its own field (nothing in
  the public API prevents this — the class is not sealed) shifts that field
  one slot further out in its own positional order. A positional call built
  against the subclass's previous field count now silently binds its last
  argument to the new `measured` instead of the subclass's own field, with no
  error either way.
- A **keyword-based reconstructor or copy adapter** — code that rebuilds an
  `EngravaMetrics` from the six previously-named fields it knows about
  (`EngravaMetrics(schema_version=..., snapshot_timestamp=..., thoughts=...,
  edges=..., storage=..., search_latency=...)`) now silently produces
  `measured=False` on every rebuilt copy, including one built from a snapshot
  that was itself genuinely measured. Any such adapter that is meant to
  preserve a real measurement must be updated to carry `measured` through
  explicitly.
- A **whole-object validator** that checks `asdict(snapshot)` against an
  expected key set (e.g. `assert set(asdict(m)) == {...}`) now rejects every
  snapshot over the unexpected `measured` key — without persisting anything,
  constructing positionally, checking `schema_version`, or going through the
  CLI, so this is not the same population as any bullet above or below.

**What to do.** Update any `Literal[1]` annotation or `schema_version == 1`
check to `2`. If you persist snapshots, branch on `schema_version` (or on the
presence of the `measured` key) when reading old records back, and prefer
checking `measured` over trusting an all-zero snapshot as a real reading —
see [Observability → Configuration](observability.md#configuration). If you
parse `engrava --format json info` output in a script, update it the same
way: expect the new `measured` key and the `schema_version` value `2`. Also
review any consumer that constructs `EngravaMetrics` positionally (directly
or through a subclass), depends on its field order, rebuilds one from named
keywords, validates its key set, or unpacks `**asdict(snapshot)` into a
function typed for the old field set — that last case raises an
unexpected-keyword error immediately rather than persisting or silently
accepting anything, so it is not the same population as the bullets above.

**New exception types: `WriteContentionError` and `WriteLockTimeoutError`.**
Neither class existed on 0.6.0; both are new `EngravaError` subclasses,
exported from the package root. No schema change is involved.

**Who is affected.** Anyone whose exception handling names specific engrava
exception types around one of the guarded write paths under real
concurrency — **including a caller that also has a broader `EngravaError`
(or bare `except Exception`) handler in the same `try`**, since which
handler actually runs depends on clause **order**, not on which one is more
specific (see below). The two new types fire in different circumstances and
are not interchangeable:

- `WriteContentionError` comes only from the dedup probe-and-insert window:
  `create_thought(deduplicate=True)`, `get_or_create`, `upsert_by_hash`, and
  `bulk_store(deduplicate=True)` through an internal `create_thought` call —
  a caller branching on the error's `operation` field sees `"create_thought"`
  from `bulk_store` too, never `"bulk_store"`. `bulk_store`'s whole batch runs
  inside one transaction, so only its *first* row opens this window; every
  later row in the same batch shares the window the first row already holds
  rather than opening its own. It is raised only after `PRAGMA busy_timeout`
  has already waited out ordinary contention on a `BEGIN IMMEDIATE` attempt
  *and* the store's own bounded retries with backoff are exhausted — so it
  signals genuine, sustained cross-connection contention, not a transient
  collision. On 0.6.x the same contention could instead produce a duplicate
  row (two connections both probing the content hash, both missing, and both
  inserting) or leak a raw `sqlite3.OperationalError` — a class outside the
  `EngravaError` hierarchy — depending on timing.
- `WriteLockTimeoutError` is broader: it comes from the task-reentrant write
  lock now held across every guarded write path's read-validate-write-commit
  span, not only the four dedup methods above. It usually means a task was
  spawned and awaited from inside another task's own `suspend_auto_commit()`
  window — an out-of-contract deadlock this now ends by raising rather than
  hanging forever — but it can also mean the acquire-timeout bound is
  configured too small for a genuinely slow embedding provider, since
  `bulk_store`'s batch embedding call holds the lock for the whole,
  network-bound round trip. The two causes are not distinguishable from the
  exception alone.

**Who is not affected.** A store never used from more than one task or
connection at a time never exercises `WriteContentionError`'s path — that
one requires genuine cross-connection contention. `WriteLockTimeoutError` is
not fully excluded by single-task use: `write_lock_acquire_timeout_seconds`
takes no validation, and a non-positive configured value (`0`, or negative)
makes the write lock raise it immediately even with nothing else contending
— see the `flush_access_buffer()` reproduction later in this section for
measured numbers. A default or any positive timeout genuinely excludes a
single-task, single-connection caller; the gap is specifically a
non-positive configured timeout.

**The population that matters most here has two shapes, and the fix differs
by shape.** Before this release, sustained dedup contention surfaced as a
raw `sqlite3.OperationalError`, a class **outside** the `EngravaError`
hierarchy. A caller retrying on that specific type — `except
sqlite3.OperationalError: retry()` around one of the four methods above —
was catching and retrying it directly.

- **A caller with *only* that `sqlite3.OperationalError` handler, and no
  broader `except EngravaError` or bare `except Exception` anywhere in the
  same `try`:** the retry stops firing, because the store now retries
  internally first and raises the unrelated-hierarchy `WriteContentionError`
  instead, and nothing in this caller's `try` matches it. This population's
  contention failure now propagates uncaught.
- **A caller with *both* a broader handler — `except EngravaError` or a bare
  `except Exception` — and the `sqlite3.OperationalError` retry handler:**
  Python tries `except` clauses in the order they are written and runs the
  *first* one that matches — not the most specific one.
  `WriteContentionError` **is** both an `EngravaError` and an `Exception`,
  so whichever broader clause is written first intercepts it before the
  `sqlite3.OperationalError` clause below is ever reached. The retry does
  not raise and does not run uncaught — it silently never runs at all, and
  whatever the broader handler does runs in its place. **This is the case
  our own advice, followed literally, would still leave broken if
  incomplete:** adding a `WriteContentionError` handler after an existing
  broader one changes nothing, because the earlier, broader clause still
  matches first — regardless of whether that broader clause names
  `EngravaError` or is a bare `except Exception`.

**What changed.** The dedup window now opens with `BEGIN IMMEDIATE` instead
of an implicit deferred transaction, so cross-connection contention surfaces
at the window's start rather than mid-transaction or not at all; and every
guarded write path now serialises across tasks on one lock per store
instance, with a bounded wait rather than an unbounded one.

**What to do.** Add an `except WriteContentionError:` handler around the
four methods above to retry contention, and **place it before any broader
handler in the same `try` that could match it — a named `except
EngravaError`, or a bare `except Exception`.** Python runs the first
`except` clause that matches, not the most specific one, and
`WriteContentionError` is both an `EngravaError` and an `Exception`; either
one, written earlier, intercepts it and makes a later, more specific clause
dead code — there is no handler shape for which position stops mattering.
Separately: an existing `except sqlite3.OperationalError:` handler left
exactly where it is does not need reordering relative to a broader handler —
it no longer matches this failure at all (`WriteContentionError` is not a
`sqlite3.OperationalError`), so it now falls through to whatever broader
handler is present, in either order. Retrying on `WriteContentionError` is
safe outright — the transaction never started, so nothing was read or
written under it. A `WriteLockTimeoutError` from a store with a slow
embedding provider and large batches may just need a higher
`write_lock_acquire_timeout_seconds`; one from anywhere else is worth
checking for a task spawned and awaited from inside a `suspend_auto_commit()`
window on the same store instance — see
[Concurrency](concurrency.md#a-deadlock-this-store-cannot-resolve-raises-it-does-not-hang).

**`get_or_create()` and `upsert_by_hash()` no longer call `create_thought()`
on a miss — an override of it silently stops running on those two paths.**
No schema change. This is `47bd68e`, the same commit that added
`WriteContentionError` above — not `f2d2348`, which added
`WriteLockTimeoutError` and is unrelated to this change.

**Who is affected.** Anyone who subclasses `SqliteEngravaCore` and overrides
`create_thought()` — for validation, enrichment, or observability — and
calls `get_or_create()` or `upsert_by_hash()` on a subclass instance,
expecting the override to run on the insert (miss) branch of either. Before
this commit, both methods' miss branch called `self.create_thought(...)`, an
ordinary virtual call that reached an override on `self`. This commit
routes that branch through the same new internal insertion primitive
`create_thought()` itself now uses internally, called directly rather than
through `self.create_thought(...)`, so an override sitting on top of the
public method is no longer in the call path at all.

**Executed directly, with a subclass whose `create_thought()` override stamps
an extra metadata key on every thought before delegating to `super()`:**
calling the override directly, and calling it through `get_or_create()` /
`upsert_by_hash()` on a miss, all three stamped the key on `v0.6.0` and on
the commit immediately before this one. On this commit and the current
tree, the direct call still stamps it; the row inserted by `get_or_create()`
and the row inserted by `upsert_by_hash()` do not carry it — no error,
no warning, the call simply returns a record the override never touched.

**A direct `create_thought(deduplicate=True)` call is not simply unaffected
either — a miss there used to run the override twice, and now runs it
once.** Before this commit, `create_thought(deduplicate=True)`'s own miss
branch re-entered `self.create_thought(thought, deduplicate=False)` — the
same virtual call an override sits on top of — so a call arriving with
`deduplicate=True` reached the override once for the original call and once
more for that internal recursive re-dispatch. Executed directly, with the
same metadata-stamping override, recording each call's own `deduplicate`
argument: a single `create_thought(deduplicate=True)` miss produced
`[True, False]` on `v0.6.0` and on the commit immediately before this one —
the override ran twice, each time appending to a persisted counter reached
`2`. On this commit and the current tree it produced `[True]` — the override
runs once, and the same counter reaches `1`. Anyone whose override is not
idempotent (increments a counter, appends to a log, emits a metric) sees
that effect halve on this specific call shape; anyone whose override is
idempotent (sets a fixed value) sees no observable difference here, only on
the `get_or_create()` / `upsert_by_hash()` miss paths above.

**Who is not affected.** Anyone calling plain `create_thought()` — with
`deduplicate` omitted, `False`, or a hit under `deduplicate=True` — sees no
change: those paths call the override exactly as many times as before. A
subclass that does not override `create_thought()` has nothing to lose on
any path. A hit on `get_or_create()` was already routed through
`_increment_confirmation()`, not `create_thought()`, on every revision, so it
is unaffected by this change specifically — `upsert_by_hash()`'s hit branch
was never routed through `_increment_confirmation()` at all, on any
revision, and is covered separately below.

**What changed.** Serialising the dedup probe-and-insert window (the fix
this section documents) restructured `get_or_create()` and `upsert_by_hash()`
to hold the new cross-connection lock across the probe and the insert as one
span; the miss branch of both, and of `create_thought(deduplicate=True)`
itself, was rewritten to call the row-insertion step directly instead of
re-entering the public `create_thought()`, which has its own validation and
its own (differently scoped) locking. Nothing about this was aimed at
overrides — it is a consequence of which internal method each miss branch
now calls, not a documented design change to virtual dispatch.

**What to do — there is no public seam left that reaches these paths before
insertion; do not reach for `on_store` as one.** `on_store` looks like the
obvious replacement and is not one, though not for the simple reason it
looks like: it does **not** always run strictly after a durable commit.
`_insert_new_thought_row()` inserts the row and calls `self._maybe_commit()`
before `_finish_create_thought()` — which calls `on_store` — ever runs; that
`_maybe_commit()` is a real, immediate commit when nothing suspends it, but
a no-op inside a caller's own `suspend_auto_commit()` window, where it
defers to that window's own eventual commit or rollback. Executed directly,
on both miss paths, with a hook that raises: called with no surrounding
window, the row was committed and present (`stored=1`) despite the raise —
the simple "after commit" story holds there. Called inside
`suspend_auto_commit()`, with the hook's exception left to propagate all the
way out of the window uncaught, the window's own rollback undid the insert
along with everything else written in it, leaving **zero** stored thoughts
— not because `on_store` rejected anything itself, but because escaping the
window rolled back the whole window. Caught inside the window instead (so
the window's own body still exits cleanly), the row is committed exactly as
in the no-window case. **None of this makes `on_store` a working replacement
for a rejecting override**, because it cannot single out its own thought:
the only way it "prevents" a commit is by taking an entire caller-owned
transaction down with it, discarding whatever else that transaction held —
not a targeted rejection of one insert. Separately, and true regardless of
any window: an `on_store` hook that adds a metadata key returns a record
carrying that key, while the raw stored row never does — the enrichment
reached the caller, never the database, on either miss path, with or
without a window, because the row's own column values are already fixed by
`_insert_new_thought_row()` before `on_store` is ever called. **The one seam
that still runs before insertion, and is verified to work, is the argument
itself:** `get_or_create()` and `upsert_by_hash()` both take the candidate
`ThoughtRecord` as a plain input, the same object `create_thought()` would
otherwise have received — enrich it with `.evolve(...)` or validate and
raise on it in your own calling code *before* passing it to either method,
rather than inside an override on the store or inside `on_store`. Executed
directly: a `ThoughtRecord` enriched this way before the call carried that
enrichment in the raw stored row on both methods' miss branch, and raising
there prevents that one insert without touching anything else a surrounding
window might hold. This does not restore a single override point that
covers every insert path uniformly — it moves the responsibility to each
call site.

**A `upsert_by_hash()` call that changes nothing can still commit a
different, unrelated write that was left pending — no concurrency
required.** No schema change. This is `47bd68e`, the same commit as the
override-bypass fix documented above.

**Who is affected.** Anyone who journals or otherwise leaves a transaction
open after a caught failure (a rejected journal insert is the concrete
case, but any write that raises before its own `_maybe_commit()` runs
qualifies), on a store also used for `upsert_by_hash()` calls that turn out
to be a match with no field differences from the stored row — **provided
that call is not itself inside a `suspend_auto_commit()` window; see below
for what changes there.** `upsert_by_hash()`'s hit branch is **not**
`_increment_confirmation()` on any revision — that is `get_or_create()`'s
hit branch. `upsert_by_hash()` either forwards the fields that differ to
`update_thought()`, or, when nothing differs, does nothing to the row at
all. This commit added an unconditional `self._maybe_commit()` to that
no-change branch, and `_maybe_commit()` commits immediately outside any
suspended-commit window — so a call that touches no row still ends
whatever transaction is open on the connection, in that case specifically.

**Executed directly, without any concurrency:** journaling enabled, an
existing thought stored, then every future journal insert made to fail —
`create_thought()` on a new, unrelated thought raised, as expected, with its
row applied but not yet committed (`in_transaction` still `True`). Calling
`upsert_by_hash()` with a record that exactly matched the existing thought's
mutable fields (a true no-op) left the connection's transaction still open
on `v0.6.0` and on the commit immediately before this one; an explicit
`rollback()` issued afterward then undid the earlier, still-pending insert —
the unrelated thought was gone. On this commit and the current tree, that
same no-op `upsert_by_hash()` call closed the transaction on its own
(`in_transaction` became `False`); the same `rollback()` issued afterward
had nothing left to undo, and the earlier, previously-uncommitted thought
was there, committed, to stay.

**That "closes the transaction" result is itself conditional on there being
no surrounding `suspend_auto_commit()` window — inside one, the same
no-op call closes nothing.** `_maybe_commit()` is a no-op whenever a
`suspend_auto_commit()` window is open, on every revision this section
covers — the no-change branch's own commit is exactly as deferred as any
other guarded write's. Executed directly, on the current tree, repeating
the same setup (an unrelated write already raised and caught, its row
applied but uncommitted) inside `suspend_auto_commit()`: the no-op
`upsert_by_hash()` call left `in_transaction` still `True` — its own
`_maybe_commit()` did nothing — and when the surrounding window later
exited via an unrelated failure of its own, that window's own rollback
undid the earlier, still-pending insert along with everything else, exactly
as it would have without the no-op call ever running. Inside a window, the
no-op branch neither commits the pending insert early nor protects it; the
insert's fate is decided by however that window itself eventually exits.

**Who is not affected.** A store with no other write left pending on the
connection when a no-op `upsert_by_hash()` runs sees no behavioral
difference — its own commit (or no-op, inside a window) leaves nothing
outstanding either way. A `upsert_by_hash()` call that does change a field
routes through `update_thought()`, which already commits its own write (or
defers to the window, inside one) on every revision; this is specifically
about the branch where nothing changes.

**What changed.** The no-change branch of `upsert_by_hash()` returns the
existing record without writing to `thought` at all, but the probe ahead of
it still opens the same `BEGIN IMMEDIATE` window as every other dedup entry
point (it cannot know in advance that the match will need no change). This
commit closes that window with `self._maybe_commit()` on the no-change
branch specifically so the write lock is not held until some unrelated
later write happens to commit it — the same reasoning documented above for
`WriteContentionError`, the one exception `47bd68e` added
(`WriteLockTimeoutError` came from `f2d2348`). It does not check whether
anything else was already pending on the connection before it commits
(or, inside a window, before it declines to).

**What to do.** Outside a `suspend_auto_commit()` window: do not rely on a
caught failure leaving a transaction open for you to `rollback()` later if
a `upsert_by_hash()` call — even one you expect to be a no-op — can run on
the same connection in between; roll back (or otherwise resolve) a caught
failure immediately, before any other guarded call on the same store
instance has a chance to run. Inside your own `suspend_auto_commit()`
window, this specific interaction does not apply — a no-op `upsert_by_hash()`
there cannot commit anything early — but everything else in the window is
still one all-or-nothing unit: a caught failure you mean to discard must be
undone with your own `ROLLBACK TO` a caller-owned savepoint (see the
delete-atomicity section above), not left for the window's own eventual
exit to sort out, since a clean exit commits it and an unrelated failure
rolls back more than just the write you caught.

**A dedup hit's journal entry can no longer be lost while its confirmation
bump survives — no concurrency required.** No schema change. This is
`47bd68e`, the same commit that added `WriteContentionError` above — not
`f2d2348`, which added `WriteLockTimeoutError` and is unrelated to this
change.

**Who is affected.** Anyone journaling writes (`journal_enabled=True`) who
relies on the journal for a complete audit trail, and who deduplicates via
`create_thought(deduplicate=True)` or `get_or_create` against existing
content — with or without any concurrency at all; a single task on a single
connection reaches this. `_increment_confirmation()` (the dedup-hit branch)
used to commit the `confirmation_count` bump on its own, then append to the
journal afterward with no further commit of its own. If the connection
closed (or anything else ended the session) before some *later*, unrelated
write happened to flush that pending journal insert, the confirmation bump
was durable and its `UPDATE_THOUGHT` journal entry was not. Executed
directly: a dedup hit followed immediately by closing the connection left
`confirmation_count` bumped but only the original `INSERT_THOUGHT` entry in
`journal_entry` on 0.6.x-shaped code; the current code left both the bump
and its `UPDATE_THOUGHT` entry.

**Who is not affected.** Anyone not journaling (`journal_enabled=False`,
the default) — there is no journal entry to lose either way. A dedup hit
followed by any other guarded write on the same store before the connection
closes was also fine on 0.6.x: that later write's own commit flushed the
still-pending journal insert along with it.

**What changed.** `_increment_confirmation()` now performs the `UPDATE`, the
read-back, and the journal append, then commits once at the end — the same
single-commit-per-write-path rule this release applies elsewhere (see
`update_thought`, `upsert_by_hash`, and the other guarded write paths). The
row and its audit entry are durable together whenever `_increment_confirmation()`
itself decides when to commit. **It does not decide that inside a caller-owned
`suspend_auto_commit()` window, and the pair can still come apart there.** If
the journal append raises — a trigger rejecting the insert is the concrete
case — and the caller's own code catches that failure before the window's
body returns, the bump is never rolled back: `_increment_confirmation()`'s own
`_maybe_commit()` is a no-op inside the window, but the window's own clean
exit commits the transaction anyway, bump included, with no journal entry for
it. Executed directly, on both 0.6.x-shaped code and the current tree: a
dedup hit inside `suspend_auto_commit()`, with a trigger rejecting the
journal insert and the surrounding code catching that error before the block
exits, left `confirmation_count` bumped and committed with no `UPDATE_THOUGHT`
entry for it, on both revisions. This fix closes the single-connection,
no-`suspend_auto_commit` gap described above; it does not make the pair
atomic against a failure the caller catches inside its own deferred-commit
window.

**What to do.** If you journal and deduplicate, nothing to change — this is
a fix, not a new obligation. If you have tooling that reconciled a
0.6.x journal against `confirmation_count` values and treated a missing
`UPDATE_THOUGHT` entry after a dedup hit as evidence of corruption or
tampering, it was more likely this gap; that reconciliation logic no longer
needs to special-case it going forward. If you deduplicate inside your own
`suspend_auto_commit()` window, do not catch a failure from that call inside
the window if the bump and its journal entry must stay paired — let it
propagate so the window's own rollback undoes the bump too.

**`suspend_auto_commit()` now rolls back on cancellation, and a nested
window commits and rolls back only at the outermost exit.** No schema
change. These are two separate fixes to the same context manager, and the
first one reaches plain, non-nested callers — including an ordinary
`bulk_store()` call — not only nested usage.

**Who is affected.** Two groups:

- **Anyone whose `suspend_auto_commit()` block can be cancelled — nested or
  not.** This includes a single, non-nested `bulk_store()` call driven under
  `asyncio.wait_for`, a timeout wrapper, or a task group that cancels
  siblings on one member's failure.
- Code that opens a *nested* `suspend_auto_commit()` window on the same
  task — one batch method calling into another that also wraps its writes
  in `suspend_auto_commit()`, on the same store instance — and relies on the
  inner call's own exit to commit or roll back anything.

**Who is not affected.** A single, non-nested `suspend_auto_commit()` block
that runs to completion, or raises an ordinary `Exception`, sees no change:
it always committed on clean exit and rolled back on an ordinary exception,
and still does. **It is not unaffected under any `BaseException` that is not
an `Exception`** — cancellation is the practical case (see below), but
`SystemExit` and `KeyboardInterrupt` share the identical gap, for the
identical reason (`except Exception` on 0.6.x, `except BaseException` on
0.7). Executed directly: a single, non-nested block that ran one write and
then raised `SystemExit` left the write neither committed nor rolled back on
0.6.x-shaped code (`in_transaction` stuck `True`, the row visible only
within that still-open transaction); the identical block on the current tree
rolled the write back cleanly.

**What changed.** On 0.6.x, `suspend_auto_commit()` caught only `Exception`.
`asyncio.CancelledError` does not derive from `Exception`, so a cancellation
landing inside the block — nested or not — skipped both the `except` and the
`else` branch entirely: no rollback ran, `_skip_auto_commit` was still reset
to `False` in `finally`, and the transaction was left open (`in_transaction`
stuck `True`) rather than resolved either way. The `finally` reset was not
conditioned on a *clean* exit either: it ran whether the block below it
returned normally or raised anything at all — including an ordinary
exception the *caller's own code* went on to catch outside `suspend_auto_commit`
itself. So a nested inner call whose ordinary exception the outer body
catches still reset auto-commit for the rest of the outer block on its way
out, letting a subsequent outer write commit immediately and standalone,
ahead of anything that happened later — a cancellation included. Layered on
top of that, a clean inner exit made a separate mistake: its own `else`
branch committed the whole (still-open, outer-owned) transaction early,
regardless of nesting. On 0.7, the block catches `BaseException` around the
yielded body, and nesting is tracked by a depth counter rather than a flag:
only the **outermost** call's own exit from that body — clean, an ordinary
exception, or a cancellation of the body — decides whether to commit or roll
back, unconditionally and at any nesting depth; an inner call's own exit, of
any kind, only decrements the depth and touches no transaction state. **This
covers a cancellation of the block's own work, not a cancellation of the
commit that follows it.** The outermost call's own `commit()` runs in the
`else` branch, after the guarded body has already returned — outside the
`except BaseException` that protects the body — on both 0.6.x and 0.7 alike.
Executed directly: cancelling right as that `commit()` call is completing
leaves one committed row with `in_transaction` already `False`, on both
revisions; this method has never rolled back a cancellation that lands in
that specific window.

**What to do.** Do not carry over an assumption from 0.6.x about what a
nested batch left committed after an inner failure or an outer cancellation
— on 0.6.x this depended on exactly where in the nested sequence the failure
landed and on the *kind* of exit each inner call took, not only on whether
the whole batch ultimately raised. On 0.7, a nested `suspend_auto_commit()`
batch is atomic at the outermost boundary **only for a failure that actually
escapes the outermost call's own body** — propagates all the way out of
every nested level without being caught by intervening application code —
in which case nothing commits and the whole batch rolls back, regardless of
how deep the failure originated. **An inner exception or cancellation that
your own code catches before it reaches the outermost body does not roll
anything back**: the outermost body still exits cleanly and still commits
everything written so far, inner rows included. Executed directly: an inner
cancellation caught by the outer body's own `try`/`except`, with the outer
body then continuing to its own clean exit, left both the inner row and the
outer row committed. **This also does not cover a cancellation landing after
the body finishes, while the outermost call's own final `commit()` is in
flight** — that narrow window is unguarded on both 0.6.x and 0.7, and a
cancellation there can still leave the batch committed. Re-verify any code
that inspects or manually finishes a connection's transaction state, or that
assumes a caught inner failure implies a rollback, against this rule rather
than against 0.6.x's behavior.

**A `bulk_store()` call nested inside a caller's own `suspend_auto_commit()`
window silently drops automatic derivation.** No schema change. This is
`f2d2348`, the same commit as the depth-counter fix documented immediately
above — not the dedup journal-entry fix before that, which is `47bd68e` —
and a third, independent consequence of it.

**Who is affected.** Anyone with the derived-records seam enabled
(`DeriveGates(enabled=True)`) who calls `bulk_store()` from inside their own,
already-open `suspend_auto_commit()` window on the same store instance.
`bulk_store()` dispatches derivation itself, locally, right after its own
inner `suspend_auto_commit()` block exits — gated on `self._skip_auto_commit`
being `False`, meaning "durable now." On 0.6.x, *any* `suspend_auto_commit`
exit — inner or outer — reset that flag to `False` unconditionally, so by
the time `bulk_store()`'s own dispatch loop ran, the flag already read
`False` even though the caller's outer window was still open, and dispatch
proceeded. On 0.7, the flag is a depth counter that only reaches `0` when
the **outermost** window closes, so with an outer window still open,
`bulk_store()`'s dispatch loop correctly sees itself as "not yet durable"
and skips every dispatch in the loop — **and nothing ever retries them**:
there is no second pass when the outer window eventually closes. Executed
directly with a real structural-split producer: a single-thought
`bulk_store()` call nested inside an outer `suspend_auto_commit()` window
produced the source thought plus 2 derived children and 2 `DERIVED_FROM`
edges on 0.6.x-shaped code; on the current tree, only the source thought
persisted — no children, no edges, and no error of any kind.

**Who is not affected.** A `bulk_store()` call made directly, with no
`suspend_auto_commit()` window open anywhere in the same task's call stack,
dispatches derivation normally on both revisions — this is `bulk_store()`'s
ordinary, documented case. **"Not wrapped in an explicit
`suspend_auto_commit()` call" is not the same condition, and undercounts the
affected population**: `bulk_store()` always opens its own window
internally, and an `on_store` hook fires *inside* that window on each row —
so a hook that reacts to a stored thought by calling `bulk_store()` again is
nesting, exactly as if the caller had wrapped it explicitly, even though the
hook's own source never mentions `suspend_auto_commit()`. Executed directly,
with a real structural-split producer: a direct, non-nested `bulk_store()`
call for a multi-paragraph thought derived 3 children as expected; the
identical call issued from inside an `on_store` hook fired by an outer
`bulk_store()` call for a different thought derived 0. A store with the
derived-records seam disabled is unaffected either way, since dispatch is a
no-op regardless of nesting.

**What changed.** This is a side effect of the same depth-counter fix
documented above: correctly deferring "durable now" to the outermost
window's close, for a call path (`bulk_store`'s own post-exit dispatch
loop) that was never updated to defer *its* dispatch to that same outermost
close instead of its own.

**What to do.** Do not call `bulk_store()` from inside your own
`suspend_auto_commit()` window if you need its automatic derivation — call
it outside any such window, where its own inner window is the outermost one
and dispatch fires normally. If you already have code that does this
nested, its derived records are silently missing on 0.7 with no exception
raised; find every newly-created thought from such a call and drive
derivation for it explicitly via `derive_existing()`, the documented
backfill entry point.

**`gc --dry-run` understated what the real run deletes.** No schema change.

**Who is affected.** Anyone who ran `gc --dry-run` before an actual `gc` and
used its message to decide whether to proceed. **If you approved a `gc` run
on the strength of the old dry-run wording, you approved deleting more than
it showed you.**

**Who is not affected.** What `gc` actually deletes has not changed — only
the dry-run and `--help` text describing it.

**What changed.** The dry run and the command's own help text described `gc`
as removing archived thoughts and their *orphaned* edges. The real run has
always done more: it deletes every edge touching an archived thought
(orphaned or not), that thought's embeddings and its actions, then
reconciles the vector index. The dry run now says so — "Would delete N
archived thoughts, plus their edges, embeddings, and actions" — matching the
rest of the documentation. **The destructive run's own output did not
change and still does not say this**: run against a database with one
archived thought carrying an edge, an embedding, and an action, it prints
exactly `Collected 1 archived thoughts.` — nothing about the edge, the
embedding, or the action it also deleted. Only the dry run and the prose
documentation describe the full set; the real command's own output still
reports just the thought count.

**What to do.** Re-read [CLI reference → `gc`](cli.md#gc) before your next
collection if edges, embeddings, or actions on an archived thought matter to
you independently of the thought itself — the old dry run would not have
told you they were going.

**A deleted thought's vector can no longer resurface through search.** No
schema change from this fix itself, but it interacts with whether your
database has run the core-12 migration.

**Who is affected.** Anyone running a database that has not run `engrava
migrate` past core schema 12, with a `sqlite-vec` backend active. On such a
database, deleting a thought (via `delete_thought`, the TTL `delete`
strategy, or hygiene GC) purged the thought's own vector immediately, but
the reconcile pass that runs on the next sqlite-vec-enabled open could read
the still-present `embedding` row as proof the thought was live and put the
vector back — after which the deleted id could reappear as an ordinary
search-similar candidate. Separately, and regardless of schema version:
`gc` and `restore` now refuse outright and exit `1` on any database that is
not exactly on this build's head schema version — but the recovery action
differs by direction. **Below** head, the refusal names `engrava migrate`.
**Above** head — a database written by a newer engrava — `engrava migrate`
cannot help and refuses too (`SchemaVersionError`'s `"newer_than_head"`
reason); the refusal instead says to upgrade engrava itself before opening
that database. **If a `gc` that used to run cleanly on an unmigrated
database now exits non-zero, that is this refusal working as intended, not
a regression in what it collects.** Read commands (`info`, `verify`,
`export`, `snapshot`, a `query` that parses as `FIND`/`COUNT`/`SELECT`)
still run on a behind schema, but now warn on stderr rather than staying
silent about it; above head, a read command refuses exactly like a
destructive one.

**Who is not affected.** A database already migrated to core-12 or later
avoided the resurrection path only when the *deleting connection* also had
`PRAGMA foreign_keys=ON` — the cascade that removes the `embedding` row is a
per-connection setting, not a property of the schema file. `from_config()`
sets it explicitly, and so does an ordinary, transaction-free call to
`ensure_schema()`, including a no-op call against an already-current
schema — but `PRAGMA foreign_keys` is itself a no-op inside an open
transaction, so an `ensure_schema()` call made on a connection that already
has one open leaves enforcement exactly as it was, silently. A caller
who instead constructs `SqliteEngravaCore(conn)` directly and skips
`ensure_schema()` on that connection — plausible against a file a different
process already migrated — inherits SQLite's foreign-keys-off default, and
the old, cascade-reliant `delete_thought()` left the `embedding` row behind
there exactly as it would below core-12. Executed both ways at schema 20: a
thought and its embedding created and migrated on one connection, then
deleted with a bare `DELETE FROM thought` on a second connection that never
called `ensure_schema()`, left the embedding row in place (`remaining=1`,
`foreign_keys=0`); the identical delete on a connection that had called
`ensure_schema()` (or used `from_config()`) removed it (`remaining=0`,
`foreign_keys=1`). Independently of schema version or FK state: a store with
no `sqlite-vec` backend active was never able to serve the vector arm that
made the id reappear. Read commands against an already-current schema see no
new warning; destructive commands against one see no new refusal.

**What changed.** A vector is now owned by whichever thought it belongs to,
not by the mere presence of an `embedding` row, and that rule is enforced
everywhere a vector could otherwise be resurrected: reconciliation, the
vector-index purge, and search's own eligibility check. `delete_thought` (and
the TTL and hygiene delete paths) now delete `edge`, `embedding`, and
`action` rows explicitly before deleting the thought, on every schema
version, rather than depending on the core-12 `ON DELETE CASCADE`. Schema
version checks are also new: destructive commands refuse outside the head
version, read commands warn below it and refuse above it, and the new
`SchemaVersionError` (exported from the package root) is what `ensure_schema()`,
`from_config()`, `EngravaManager.get_store()`, and `engrava migrate` raise
when a database is a populated schema below the bootstrap floor, or stamped
above this build's head version.

**What to do.** Run `engrava migrate` if you have not already, on any
database below head that you plan to run `gc` or `restore` against. If the
refusal instead says the database is *newer* than this build's head
version, `engrava migrate` will not resolve it — that database was written
by a newer engrava, and the fix is to upgrade engrava, not to migrate. If a
database accumulated dangling `embedding` rows from deletions made *before*
this fix, under an older engrava build, on a schema still below core-12:
those rows can no longer be resurrected into a result, but they are not
removed until you migrate — `engrava migrate` purges them as part of the
core-12 step. See [Known Limitations → Deletion on a database that has not
been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated)
for the full mechanism.

**`ConnectionQuarantinedError` is now reachable from a plain `delete_thought()`
call, with the derived-records seam disabled.** No schema change. This is
`6e4ed41`, the same commit as the vector-ownership fix above, and a second,
independent consequence of it.

**Who is affected.** Anyone whose `delete_thought()` (or the TTL `delete`
strategy, or hygiene GC) hits **any failure the store cannot cleanly unwind**
while deleting a thought's `edge` / `embedding` / `action` rows — not only a
cancellation. Two distinct, independently-executed triggers both reach the
same outcome:

- **A cancellation landing during the savepoint's `RELEASE`.** aiosqlite's
  worker thread can complete that `RELEASE` on the connection even though
  the awaiting coroutine observes a cancellation instead of the release's
  success; the store's own unwind (`ROLLBACK TO` the savepoint) then fails
  because the savepoint it targets has already been consumed.
- **An ordinary, non-cancellation failure that also leaves the unwind unable
  to complete — no cancellation and no derivation involved at all.**
  Executed directly: an `ON DELETE` trigger that vetoes the edge delete with
  `RAISE(ABORT, ...)` makes the initial delete fail with `IntegrityError`. A
  trigger fires on table operations, not on a transaction-control
  statement, so it cannot itself deny the subsequent `ROLLBACK TO` — the
  mechanism that can is an authorizer callback (`set_authorizer`, on
  `sqlite3.Connection` and on `aiosqlite`'s wrapper of the same hook)
  refusing that statement. Executed directly: the denial surfaces as
  `DatabaseError("not authorized")`. Either way, the store cannot prove
  the deletes were undone. `delete_thought` raises the original
  `IntegrityError`, exactly as it should — and the *next* guarded call on
  that instance raises `ConnectionQuarantinedError`.

Both cases produce `ConnectionQuarantinedError` on the *next* guarded call,
not on the `delete_thought()` call itself (which raises whatever the
original failure was). Previously, `ConnectionQuarantinedError` was
reachable only through a failed derived-record compensation rollback — a
caller who never enabled that seam had a real reason to treat this error as
something that could not happen to them.

**Who is not affected.** A `delete_thought()` (or TTL delete, or hygiene GC)
call whose child-row savepoint unwind either never has to run (nothing
failed) or runs and completes cleanly never reaches this path, seam enabled
or not.

**What changed.** `delete_thought` (and the TTL delete strategy, and hygiene
GC) gained the explicit child-row deletion described above, wrapped in a
`SAVEPOINT` whose release sits inside the guarded region specifically so a
cancellation landing during that release can still be recognized as
"already released." When the store's own unwind attempt then fails for
**any** reason — a race with a cancellation, or an ordinary error that
happens to also defeat the `ROLLBACK TO` — it quarantines the connection
rather than guessing at a consistent state, exactly as the pre-existing
derived-record compensation path already did. The two paths (derivation
compensation, and this one) now share one outcome (`ConnectionQuarantinedError`
on the next guarded call) from independent triggers.

**What to do.** Treat `ConnectionQuarantinedError` as reachable from any
`delete_thought()`, TTL-delete, or hygiene-GC call whose child-row deletes
can fail to unwind cleanly — whether from a cancellation racing the
savepoint release, an installed trigger's delete veto paired with an
authorizer denying the follow-on `ROLLBACK TO`, or any other failure that
leaves the `ROLLBACK TO` itself unable to complete — not only from
derived-record-enabled stores. A trigger veto alone is not enough: it fails
the initial delete, but `ROLLBACK TO` still succeeds unless something else
— the authorizer, or another unwind failure — also blocks it. On catching
`ConnectionQuarantinedError`, replace the store instance — it is terminal
by design, the same as before this change.

**Child deletion is no longer atomic with the parent delete — anything that
prevents, diverts, or skips the parent delete after the children are already
gone commits their loss, or worse, lets the deletion the guard was meant to
block go through anyway — with one exception, covered later in this
section: a firing, unconditional `RAISE(ROLLBACK, ...)` trigger unwinds the
whole transaction instead, and preserves the children.** No schema change.
This is a further consequence of `6e4ed41`, the same commit as the
vector-ownership fix and the quarantine change above: splitting the
previously atomic, cascade-driven delete into an explicit children-delete
followed by a separate, unguarded parent delete opened a gap between the
two steps that anything landing in that gap falls into. The condition is
general along the timing and caller axes — it does not depend on which
trigger timing or which caller is involved. The conflict-resolution keyword
does matter: everything below holds for `ABORT`, `FAIL`, and `IGNORE`; a
firing, unconditional `RAISE(ROLLBACK, ...)` trigger is the one exception,
covered later in this section. The list below illustrates the keywords that
do lose the children, not an exhaustive population to check against.

**Who is affected.** Anyone with a `thought`-table trigger, or any other
mechanism, that can make the parent `DELETE FROM thought` fail, get skipped,
or silently not run — other than a firing, unconditional `RAISE(ROLLBACK,
...)` trigger, which is protected instead (see below) — on a database
already migrated to head schema, reached through `delete_thought()`, the
TTL `delete` cleanup strategy
(`cleanup_expired()`), or hygiene GC (`run_hygiene()` with `auto_gc_enabled`)
— every one of these calls `_delete_thought_children_explicit()` first,
which deletes the thought's `edge`, `embedding`, and `action` rows and
**releases its own savepoint before the parent delete runs at all**, folding
those three deletes into the still-open outer transaction with no savepoint
of their own left protecting them. **Executed, as illustrative forms of the
same general condition, all against a database already at head schema:**

- `BEFORE DELETE ON thought ... RAISE(ABORT, ...)` and `BEFORE DELETE ON
  thought ... RAISE(FAIL, ...)` both make `delete_thought()` raise
  `IntegrityError` with the children already gone and the transaction still
  open; the next ordinary write on the same connection — an unrelated
  `create_thought()` — commits that loss, and the thought itself survives.
- `AFTER DELETE ON thought ... RAISE(ABORT, ...)` reproduces the identical
  outcome: the trigger fires after the row delete but within the same
  statement, so `IntegrityError` propagates from the same `execute()` call,
  past the same unwound-nothing point, with the children already gone.
- `BEFORE DELETE ON thought ... RAISE(IGNORE)` raises nothing: the parent
  `DELETE` silently affects zero rows, `delete_thought()` returns `False`,
  and its own `_maybe_commit()` commits the already-applied child deletes
  immediately — durable before the call even returns, with no exception for
  any caller to notice.
- A `BEFORE DELETE ON thought WHEN EXISTS (...)` trigger, guarding against
  deleting a thought that still has an edge, is a distinct and worse case: by
  the time the parent delete runs and the trigger's `WHEN` clause is
  evaluated, `_delete_thought_children_explicit()` has already removed that
  edge, so the predicate is false and **the trigger never fires at all**.
  `delete_thought()` returns `True` and the whole delete — thought and all
  three children — commits normally. The guard is not merely bypassed on the
  children; it is bypassed on the thought too, silently, with a return value
  that looks like ordinary success.
- `cleanup_expired()` under the `delete` TTL strategy loses the same three
  rows the same way as `delete_thought()`, confirmed against the same
  `RAISE(IGNORE)` trigger.
- `run_hygiene()` with `auto_gc_enabled=True` reproduces it too: against a
  hygiene-archived thought past both restore windows, with the same
  `RAISE(IGNORE)` trigger installed, the GC pass reported `gc_count=0`
  (the parent delete affected zero rows) while the thought's edge,
  embedding, and action were already gone in the still-open transaction —
  an unrelated `create_thought()` afterward committed that loss.

None of these raise `ConnectionQuarantinedError`: `_delete_thought_children_explicit()`'s
own savepoint released cleanly in every case above, so the mechanism the
section above describes never engages, and nothing signals the loss (or the
bypassed guard) to the caller, the CLI, or a log.

**Who is not affected.** A database with nothing installed on the `thought`
table that can fail, skip, or no longer match the parent delete never
reaches any of this — `engrava` installs no such mechanism itself, only an
operator can. Beyond that, there is no trigger *timing* (`BEFORE` / `AFTER`)
that is safe by construction, and a `WHEN` predicate that reads a child row
is not made safe by *any* keyword, `RAISE(ROLLBACK, ...)` included — if the
predicate no longer matches once the children are gone, the trigger body
never runs, whichever keyword it names. Executed directly, with `BEFORE
DELETE ON thought WHEN EXISTS (SELECT 1 FROM edge WHERE ...) BEGIN SELECT
RAISE(ROLLBACK, ...); END`: on `v0.6.0` the predicate still sees the edge
(cascade has not run yet), the trigger fires, and everything is preserved;
on the current tree the edge is already gone by the time the predicate is
evaluated, the trigger never fires at all regardless of its raise keyword,
and `delete_thought()` returns `True`, committing the parent and all three
children.

**An *unconditional* `RAISE(ROLLBACK, ...)` trigger — no `WHEN`, or one that
still matches — is different: it does protect the children when it fires,
and it is the only keyword that does.** Executed directly, with a plain
`BEFORE DELETE ON thought BEGIN SELECT RAISE(ROLLBACK, ...); END` and no
predicate: the edge, embedding, and action rows were all present afterward
(`edges=1, embeddings=1, actions=1`) and the thought itself survived
(`thought=1`) — `RAISE(ROLLBACK, ...)` ends the whole transaction rather
than just the failing statement, and undoing the whole transaction undoes
the already-applied, already-released child deletes along with it. **The
cost is exactly "the whole transaction": an unrelated write made earlier in
the same window did not survive either** (`earlier_write=0`, where the same
setup with `RAISE(ABORT, ...)` instead left it at `1`), and a caller-owned
`SAVEPOINT` taken to protect against exactly this — see below — does not
survive it either: `ROLLBACK TO` that savepoint, attempted after the
trigger's own rollback had already run, itself raised `OperationalError:
no such savepoint`. A firing `RAISE(ROLLBACK, ...)` and the caller-owned
`SAVEPOINT` recipe below are not something to combine — the trigger's own
rollback removes the savepoint before the caller's `except` block gets to
use it.

**The one mechanism confirmed to protect the children on a trigger that
does *not* end the whole transaction, without losing unrelated work, is a
caller-owned rollback that encompasses the child deletes before anything
commits them.** A caller can take its own `SAVEPOINT` immediately before
calling `delete_thought()` (or triggering the TTL `delete` strategy, or
running hygiene GC), inside its own `suspend_auto_commit()` window, and
roll back to it on failure. Executed directly: inside `suspend_auto_commit()`,
after an unrelated `create_thought()` had already run, a caller-owned
`SAVEPOINT` taken just before `delete_thought()` on a thought guarded by a
plain `RAISE(ABORT, ...)` trigger — caught, then `ROLLBACK TO` that
savepoint — restored all three children and the thought; the outer window's
own clean exit afterward still committed the earlier, unrelated
`create_thought()`. The caller's savepoint predates the child helper's own
nested one, so rolling back to it undoes the child helper's already-released
deletes too, without touching anything written before the savepoint was
taken. **This depends on the trigger raising without itself ending the
whole transaction** — `ABORT` and `FAIL` qualify; a firing `RAISE(ROLLBACK,
...)` does not, for the reason above. **`RAISE(IGNORE)` does not qualify
either, and for a different reason: it does not raise at all.**
`delete_thought()` on an `IGNORE`-guarded thought returns `False` with no
exception, so the recipe's own `except` block — the only place it issues
`ROLLBACK TO` — never runs. Executed directly: following the recipe against
a plain `BEFORE DELETE ON thought BEGIN SELECT RAISE(IGNORE); END` trigger,
with no `WHEN` predicate involved, still left `edges=0, embeddings=0,
actions=0` — the savepoint was reached, taken, and then released as a
no-op success (the `else` branch, not the `except` one) — while the earlier
unrelated write survived at `1`, same as every case above. **It also only
helps when something actually raises for the caller to catch, and both
`IGNORE` and a defeated `WHEN` predicate share that gap even though they
fail for different reasons** — one because there is nothing to raise about
by design, the other because the predicate no longer matches. It does
nothing for the `WHEN`-predicate case above either: `delete_thought()`
returns `True` with no exception, so there is nothing to trigger a
`ROLLBACK TO`, and the savepoint is simply released along with the
deletion it did not know to stop. A caller that must handle `RAISE(IGNORE)`
correctly has to check the return value itself — `delete_thought()`
returning `False` is the only reliable signal, and reacting to it means
rolling back to the same caller-owned savepoint explicitly, from the
success path, not from a `except` block.

**The TTL count is not a substitute signal — measured, not assumed.**
`cleanup_expired()` returns `expired_count=len(expired_ids)`, the count of
rows its own candidate `SELECT` returned; nothing in the delete loop checks
`rowcount` against it, and the docstring calls it "Number of thoughts
processed", not "number deleted". Run directly: one thought made expired,
behind a plain `BEFORE DELETE ON thought BEGIN SELECT RAISE(IGNORE); END`
trigger with no `WHEN` predicate, still reported `expired_count == 1` — its
full expected value — while the thought row itself was still present
afterward, because the trigger silently discarded the parent delete. **On
the TTL path this count cannot detect a skipped deletion at all; a reader
relying on it to notice an `IGNORE`-guarded thought is unprotected.**
`_hygiene_gc()` is different and genuinely safer: it checks `if
cursor.rowcount <= 0: continue` before incrementing its own count, so a
skipped parent delete does make that count come back lower. Even there,
though, the count is not an early warning — `_delete_thought_children_explicit`
already ran and committed the edge, embedding, and action deletes inside the
same open transaction before the guarded parent `DELETE` is attempted, so by
the time the lower count reveals the skip, the children are already gone.

**What changed.** On `v0.6.0` and the commit immediately before this one,
`delete_thought()` relied on the core-12 `ON DELETE CASCADE` to remove the
children: the parent delete and its cascade were one atomic statement, so
anything that rejected, ignored, or failed to match the parent delete
rejected, ignored, or failed to match the cascade along with it — executed
directly against both revisions with the identical `BEFORE`/`AFTER`,
`ABORT`/`FAIL`/`IGNORE`/`ROLLBACK`, and `WHEN`-predicate forms above, the
edge, embedding, and action rows were untouched in every case. This commit
replaced that single atomic statement with the explicit children-then-parent
sequence, to close the pre-cascade resurrection gap the vector-ownership fix
above targets, and did not extend the children's savepoint to cover the
parent delete too — which is what leaves this gap open on exactly the
schema (head, cascade-capable) that no longer needed the explicit deletes
for correctness in the first place.

**Migrating does not resolve this.** The databases above were already at
head schema, with `ensure_schema()` run to completion, before anything on
the `thought` table ever fired — there is no pending migration step that
changes this sequencing, because none is needed for the schema to be
current.

**What to do.** Prefer moving a delete-time policy check to *before* calling
`delete_thought()` / triggering the TTL `delete` strategy / running hygiene
GC, instead of inside a `thought`-table trigger — this is the only form that
is not sensitive to trigger timing, keyword, or a `WHEN` predicate: the
check runs before any deletion has started at all, so there are no children
left to falsify a predicate or to be lost. If you already have a policy
check as an unconditional `thought`-table trigger and cannot move it: an
unconditional `RAISE(ROLLBACK, ...)` protects the children on its own (at
the cost of the whole transaction, unrelated work included — do not also
wrap the call in your own `SAVEPOINT`, since the trigger's own rollback
removes it before you can use it); `ABORT` and `FAIL` do not protect
anything by themselves, but wrapping the call in your own
`suspend_auto_commit()` window with a `SAVEPOINT` taken immediately before
it, rolled back to it in an `except` block, restores the children without
discarding unrelated work. **`RAISE(IGNORE)` needs different handling
again: it never raises, so the same recipe's `except` block never runs and
the savepoint is released as a no-op.** Check the call's own return value
(or count) instead, and roll back to the savepoint explicitly from the
success path when it indicates the delete did not happen. None of these
shapes rescues a `WHEN`-gated trigger that stops firing because its
predicate depended on a row the children delete already removed — there is
no keyword or caller-side rollback that helps when nothing raises in the
first place; only moving the check earlier closes that case.

**Connection cleanup on failure and cancellation is fixed — not only in the
CLI.** No schema change. This reaches every caller of the library, not just
the CLI: `SqliteEngravaCore.from_config()`, `EngravaManager.get_store()`,
`close()`, and `close_all()` all changed, and the fix is not limited to
corrupt-file opens.

**Who is affected.** The real axis is not "shutdown", and not cancellation
either — it is **whether a failure during construction or close reaches a
cleanup that runs to completion and preserves the original error, whether
or not a cancellation is involved** — with one limit: `_close_quietly`
itself only catches `Exception`, so a `CancelledError` landing during its
own `await conn.close()` still escapes and replaces the original error
(see "What changed" below). The four cases below all involve an ordinary
`Exception` cleanup failure and are unaffected by that limit:

- Anyone who has pointed (or might point) an engrava CLI command at a
  corrupt or truncated database file. This bug shipped in 0.5.0: `_open_db`
  opened the connection and ran its setup `PRAGMA`s with no exception
  handling at all, so a `DatabaseError` from a corrupt file raised while the
  code that would have closed the connection sat only on the success path,
  and aiosqlite's connection-worker thread is not a daemon, so the
  interpreter could not exit — the command printed nothing and never
  returned. A direct `from_config()` or `EngravaManager.get_store()` call
  did not hang on this: at `v0.6.0`, both already wrapped construction in
  `except Exception: await db.close(); raise`, so the same plain
  `DatabaseError` was already caught, the connection already closed, and the
  error already propagated.
- Anyone whose process can be cancelled while closing a store — **including
  one backed by a perfectly healthy database.** `close()` flushes the
  access-tracking buffer before closing the connection; a cancellation
  landing during that flush used to propagate immediately, skipping the
  connection close below it and leaking the same non-daemon worker thread as
  the corrupt-file case, on a database with nothing wrong with it.
  `EngravaManager.close_all()` had the matching gap across multiple stores: a
  cancellation while closing one store used to abandon the rest, worker
  threads and all, instead of finishing their closes first.
- **Anyone whose `from_config()` or `get_store()` call can be cancelled
  during construction — on a perfectly healthy database, with nothing to do
  with shutdown.** A timeout wrapped around either call (`asyncio.wait_for`,
  a cancelled parent task) can land after the connection opens but before
  construction finishes. Executed directly against both revisions: a
  cancellation raised mid-construction leaves the connection open — `close()`
  never reached — on 0.6.x-shaped code (`except Exception`, which does not
  match `asyncio.CancelledError`); the current code (`except BaseException`)
  reaches `close()` before re-raising the cancellation.
- **Anyone whose `from_config()` construction fails with an ordinary error
  whose own cleanup *also* fails — no cancellation anywhere in it.** On
  0.6.x, construction's `except Exception: await db.close(); raise` let a
  failure in `db.close()` itself replace the original error, because the
  bare `raise` is never reached when the statement before it raises. Current
  code routes that close through `_close_quietly`, which swallows (and logs)
  its own failure so the original error's `raise` is always reached.
  Executed directly: construction raising `ValueError` with cleanup raising
  `OSError` surfaced the `OSError` on 0.6.x-shaped code and the original
  `ValueError` on the current tree. A caller whose error handling branches on
  which exception type it catches, for this exact combination, sees a
  different type now — on a perfectly healthy database, with no cancellation
  involved at all.

**Who is not affected.** A process whose database file is healthy
(non-corrupt, non-truncated), whose store lifecycle — construction *and*
close, whether one store directly or several through `EngravaManager` — is
never reached by a cancellation, **and** whose cleanup never itself fails
during a construction error, sees no behavior change on that path.

**What changed.** Opening a connection (in the CLI, in `from_config()`, and
in `EngravaManager`'s service construction) and entering the block that
guarantees its close are now one step, so a failure while opening —
including the `sqlite3.DatabaseError` a corrupt or truncated file raises, or
an `asyncio.CancelledError` reaching that same span — always reaches the
close instead of leaking an unclosed connection. `close()` now defers a
cancellation arriving during the access-buffer flush until the connection is
actually closed, rather than letting it skip the close. `close_all()`
finishes closing every remaining store when one of them is cancelled,
re-raising the cancellation only afterward. Cleanup is also routed through a
shared `_close_quietly` helper that catches `Exception` and logs it rather
than raising, so an ordinary close failure during cleanup never replaces
the original error that triggered the cleanup — on any revision-vs-0.6.x
construction failure, cancelled or not. **This is not fixed for every
failure at that point.** `_close_quietly` only catches `Exception`; an
`asyncio.CancelledError` delivered to its own `await conn.close()` — the
same suspension point this change is protecting — is a `BaseException`,
not an `Exception`, so it still escapes, aborts the cleanup, and replaces
the original construction error. This is the same `except Exception` /
`BaseException` gap this change closes elsewhere, left open in the one
helper written to guard against it; the source is tracked separately, not
in scope for this note.

**What to do.** Nothing, unless a script wrapped CLI invocations in its own
timeout-and-kill logic to work around the old hang — that workaround is no
longer necessary, though it remains harmless. A script that inspected the
process's exit code will now see an immediate non-zero exit instead of an
indefinite hang. A library caller that wraps `from_config()`, `get_store()`,
or a store's `close()` / `close_all()` in a timeout, or whose surrounding
task can otherwise be cancelled during any of those calls, no longer needs
its own workaround for a leaked connection worker in the ordinary case —
construction and close alike now close the connection before the
cancellation propagates. **One window survives**: a cancellation delivered
to the cleanup's own `await conn.close()`, described above, still escapes
before that close completes, so the worker can leak there. The exposure is
narrowed, not removed; a caller that must be certain across that window
still needs its own guard. If your
error handling for a `from_config()` failure branches on the *type* of the
raised exception, re-check it against the original construction error, not
whatever a coincidentally-failing cleanup used to surface in its place.

**Three exception classes are now importable from the package root.**
`ReferentialIntegrityError`, `DuplicateEdgeError`, and `CoreMigrationError`
no longer require `from engrava.domain.exceptions import ...`; they import
directly from `engrava`, like every other exception in the reference table.
The troubleshooting entry documenting the old import failure is gone along
with it.

**A hook that calls back into dedup no longer deadlocks — a resolved defect,
in the reader's favour.** No schema change. This is `47bd68e`, the same
commit as the cross-connection dedup fixes above.

**Who is affected.** Anyone whose `on_store` hook (or anything else reached
during the insert pipeline of `create_thought(deduplicate=True)`,
`get_or_create()`, or `upsert_by_hash()`) itself awaits one of those same
three methods on the same store instance, from the same task — a hook that
records its own enrichment as a separately deduplicated thought is the
concrete shape. Before this commit, `_dedup_lock` — a plain, non-reentrant
`asyncio.Lock` — was held for the whole outer call, insert pipeline and
`on_store` included, because the miss branch of all three methods recursed
back into the public `create_thought()` from inside that lock's own `async
with` block. A same-task re-entry into any of the three then blocked on a
lock the same task already held and could never release first — an
unconditional hang, not a race.

**Executed directly, guarded by a bounded wait so the hang could be observed
rather than actually blocking forever:** a hook that, on a miss, awaits
`get_or_create()` against different, already-existing content (so the
inner call is a hit and does not recurse into the hook again) stalled past
the bound on every one of the three entry points — `create_thought
(deduplicate=True)`, `get_or_create()`, and `upsert_by_hash()` — on `v0.6.0`
and on the commit immediately before this one. The identical setup
completed normally, on all three entry points, on this commit and the
current tree.

**A spawned-and-awaited callback on a different task is also fixed by this
commit — measured, not assumed.** `_TaskReentrantLock` does not exist yet at
`v0.6.0` or at the commit immediately before this one — this whole
probe-and-insert mechanism, and its own residual-gap comment noting "Closing
it for real needs a task-reentrant lock", are themselves new in this commit
(`47bd68e`); `f2d2348` is what later replaces that comment with the class
itself. So at those two revisions there was no write lock standing between a
spawned task's `get_or_create()` call and the hang above — a hook that
spawns a task, and awaits it, hit `_dedup_lock` for the same reason the
same-task case did. Executed directly, guarded by the same bounded wait as
above: a hook that spawns a task to await `get_or_create()` against
different, already-existing content and then awaits that task stalled past
the bound on all three entry points — `create_thought(deduplicate=True)`,
`get_or_create()`, and `upsert_by_hash()` — on `v0.6.0` and on the commit
immediately before this one, and completed normally on all three, on this
commit and the current tree.

**Who is not affected.** A hook that never calls back into any of the three
dedup entry points on the same store instance never reached `_dedup_lock` a
second time, so it never hung *on that lock* on any revision, whether the
callback runs on the same task or one it spawns and awaits. **That is not
the same as "never hung" outright.** `_write_lock` is a separate,
later-introduced lock guarding every guarded write, not only the three dedup
methods, and a spawned task gets none of the same-task reentrancy that
closes the gap above: a hook that spawns a task to call plain
`create_thought()` — never touching `create_thought(deduplicate=True)`,
`get_or_create()`, or `upsert_by_hash()` — from inside an enclosing
`suspend_auto_commit()` window still raises `WriteLockTimeoutError` on the
current tree, the same residual mechanism the next paragraph describes.
Executed directly: that exact hook, armed inside an open
`suspend_auto_commit()` window with a 2-second
`write_lock_acquire_timeout_seconds`, raised `WriteLockTimeoutError` rather
than completing or hanging past the bound.

**A residual write-lock risk remains, but only under an enclosing
lock-holding window — a non-positive `write_lock_acquire_timeout_seconds`
is a separate, unvalidated risk, not this one.** Outside an enclosing
window, both `_dedup_lock` and `_write_lock` are released before `on_store`
runs (see "What changed" below), which is why the spawned-task shape above
now completes — provided the configured timeout is a default or a positive
value. `write_lock_acquire_timeout_seconds` takes no validation, and a
non-positive configured value (`0`, or negative) still makes the write lock
raise `WriteLockTimeoutError` immediately against a completely free lock,
with no enclosing window required — the same gap documented above and in
the `flush_access_buffer()` reproduction later in this section. The risk
below is conditional on the *calling* task already holding `_write_lock` open
across the whole call — the concrete case is an outer
`suspend_auto_commit()` window — because a task spawned and awaited from
inside that window cannot reuse its parent's task-reentrant grant, and
blocks on the lock its own parent task is still holding. Executed directly
on the current tree: the same spawn-and-await hook, driven from inside an
open `suspend_auto_commit()` window, raised `WriteLockTimeoutError` instead
of completing — the task-reentrant write-lock hazard documented elsewhere
in this file, not a new one, and reachable only with that enclosing window
in place.

**What changed.** Serialising the dedup probe-and-insert window narrowed
`_dedup_lock`'s scope to end where the probe-and-insert span ends, before
`_finish_create_thought()` — and therefore `on_store` — runs, rather than
wrapping the entire call. This was not aimed at reentrancy; it is a side
effect of the same restructuring documented above (the miss branch calling
the internal insertion step directly instead of recursing through the
public method), which happened to move `on_store` outside the lock's scope
entirely.

**What to do.** Nothing — this is a fix, not a new obligation. If you
previously worked around the hang (routing such a hook's dedup call through
a separate store instance, or avoiding the recursive call shape entirely),
the workaround is no longer necessary, though it remains harmless. If a test
or monitor was built around detecting this specific hang (a timeout that
was expected to fire), it will no longer fire for this reason.

**`flush_access_buffer()` can lose already-buffered access events on a
write-lock timeout.** No schema change. Introduced by `f2d2348` (the
read-modify-write critical-section commit above), not `47bd68e` (the
dedup-serialisation commit).

**Who is affected.** Anyone with access tracking enabled
(`access_tracking_enabled=True`) whose `flush_access_buffer()` call —
explicit, or the automatic one at a consolidation cycle boundary or on
`close()` — can run while a different task holds the store's write lock
long enough to exceed `write_lock_acquire_timeout_seconds`.
`flush_access_buffer()` drains the in-process access buffer *before*
acquiring the write lock it needs to apply the drained events, so a failed
acquisition has nowhere to put what it already took out of the buffer.

**Executed directly:** one access event buffered, a different task holding
the write lock open (via its own `suspend_auto_commit()` window) for longer
than a short configured `write_lock_acquire_timeout_seconds`, then
`flush_access_buffer()` called — it raised `WriteLockTimeoutError`, a retry
immediately afterward returned `0` (nothing left pending), and the raw
`access_count` on the target row never moved. Before this commit,
`flush_access_buffer()` acquired no lock at all; the identical setup, run
against that source, always drained and persisted the event.

**Who is not affected.** A store never used from more than one task
concurrently never contends for the write lock against another task — but
contention is not the only way to fail to acquire it.
`write_lock_acquire_timeout_seconds` takes no validation, and a non-positive
value (`0`, or negative) makes `_TaskReentrantLock.acquire()` raise
`WriteLockTimeoutError` immediately even against a completely free lock,
single task or not. Executed directly: one buffered event, no other task or
connection, a store constructed with `write_lock_acquire_timeout_seconds=0`
— `flush_access_buffer()` raised `WriteLockTimeoutError`, an immediate retry
returned `0`, and the persisted `access_count` stayed `0`; the identical
setup with a positive timeout (`0.001`s tested) drained and persisted
normally. This is a different failure from `flush_access_buffer()`'s own
return-value contract (what its count actually measures) — that is tracked
separately; this is about the buffered events themselves going missing, not
about what the returned number means.

**What changed.** The read-modify-write critical-section fix added
`async with self._write_lock:` around `flush_access_buffer()`'s batched
`UPDATE`, with the same bounded-wait, typed-timeout behaviour every other
guarded write path gained. It did not reorder the method to acquire that
lock *before* draining the buffer, so the drain and the acquisition remain
two separate steps with nothing between them to put the drained events back
if the second one fails.

**What to do.** The buffer's existing best-effort framing — a crash before
a flush undercounts, and self-heals as access continues — does not cover
this: a crash loses events that were never drained, but a `WriteLockTimeoutError`
here loses events that *were* drained and then had nowhere to go, with no
self-healing path back. If access counts must not silently under-count
beyond that existing tolerance, catch `WriteLockTimeoutError` around
`flush_access_buffer()` calls, and raise `write_lock_acquire_timeout_seconds`
if access tracking runs on a store that also holds long
`suspend_auto_commit()` windows from other tasks.

**This section covers what four large commits were found, by execution, to
change — not everything they touched.** `f2d2348`, `6e4ed41`, `47bd68e`, and
`d706f88` — the read-modify-write critical section, vector ownership,
cross-connection dedup serialisation, and connection cleanup on failure and
cancellation — each changed between roughly 1,000 and 2,500 lines apiece
(997 to 2,471 insertions, by commit), reworking transaction boundaries, lock
scope, dispatch, and cleanup behaviour that every guarded write path shares,
not one narrow code path apiece. The entries above are the effects twelve
rounds of executing this section's claims against real code have found and
confirmed; they are not a closed inventory of these four commits' effects.
A reader whose store is subclassed, hooked, wrapped in `suspend_auto_commit()`,
or driven from more than one task in a combination not covered above should
test that exact combination directly — treating its absence from this list
as a clean bill is not a conclusion this section supports.

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
- **Custom embedding providers must expose a public `dimension`.**
  `EmbeddingProviderProtocol` has always required it, but nothing checked it at
  construction and the 0.5 core read it at exactly one site, off the query path
  — so a provider that kept the value privately (`self._dimension`, no public property)
  worked on 0.5 for as long as nothing called `verify_embedding_model()`, which
  is that site. 0.6 reads it on **every** vector search, before any comparison —
  unless a `sqlite-vec` backend is configured, whose own `vec0` table declares
  the dimension and is consulted first. Such a provider now raises
  `EmbeddingProviderContractError` (an `EngravaError` subclass) naming the
  provider class and the missing member:

  ```text
  embedding provider 'MyProvider' does not expose the required
  EmbeddingProviderProtocol member 'dimension'. Add a public 'dimension'
  property to 'MyProvider' — a private attribute such as '_dimension' does not
  satisfy the protocol.
  ```

  The fix is to add the property:

  ```python
  @property
  def dimension(self) -> int:
      return self._dimension
  ```

  The check is **lazy**: constructing a store with such a provider still
  succeeds, and a store that never searches by vector — and never calls
  `verify_embedding_model()` — is unaffected. Call `verify_embedding_model()`
  after construction to fail at startup instead; it raises the same typed error.
  This is not a contract change — the protocol required the member before 0.6;
  only the point at which it is read, and the error you get, have changed.
- **An edge stored with `decay_multiplier = 0.0` read back as `1.0` on 0.5.x.**
  The column held the value you wrote, but the decode tested it for truthiness
  and substituted the `1.0` default, so a deliberate `0.0` never survived the
  read. Because `update_edge` rebuilds the whole record from that read and
  writes every column back, any later change to such an edge — a new `weight`,
  or an `invalidate_edge` call closing its valid-time interval — persisted `1.0`
  over the stored `0.0`, unless that same call set `decay_multiplier` itself.
  0.6 decodes the column on presence rather than truthiness, so `0.0` now
  round-trips as written. **Upgrading does not restore a value that was already
  overwritten**: where a 0.5.x update replaced a stored `0.0`, the database now
  holds `1.0`, and 0.6 reads that `1.0` back faithfully. If you set
  `decay_multiplier = 0.0` on any edge deliberately, check those edges after
  upgrading and set the value again where it now reads `1.0`.
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

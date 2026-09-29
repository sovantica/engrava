# Data lifecycle, retention & deletion

How a thought moves through its lifecycle, how time-to-live expiry works, and —
importantly for privacy and compliance — what it takes to **truly** erase data,
including the residue a naive delete leaves behind.

> **Compliance note.** This page describes the mechanics honestly so you can build
> a correct retention/erasure process. The default expiry strategy **archives**
> (does not erase), and a hard delete can still leave content in the audit
> journal and in backups. Read the [GDPR / hard deletion](#gdpr-and-hard-deletion)
> section before relying on TTL for "deletion".

## Lifecycle states

Every thought carries a `LifecycleStatus`. There are four states:

| State | Meaning |
|---|---|
| `CREATED` | Just created, not yet promoted into active use. |
| `ACTIVE` | In normal use — the default working state, included in queries. |
| `DONE` | Completed (e.g. a finished task) but retained. |
| `ARCHIVED` | Soft-retired and retained until garbage-collected. **Excluded from default ranked retrieval** (reversible via `restore_thought` / `include_archived`) — see the note below. |

You set the status on the `ThoughtRecord` you create, and update it over the
thought's life. Archiving is the soft-retire step: an `ARCHIVED` thought still
exists (and its content is still stored) until you garbage-collect it.

The ordinary lifecycle state machine is:

```text
CREATED -> ACTIVE -> DONE -> ARCHIVED
              \--------------> ARCHIVED
ARCHIVED --restore_thought()--> ACTIVE
```

The direct `ACTIVE -> ARCHIVED` transition is valid; a thought does not have to
pass through `DONE` before being manually archived. `restore_thought()` is the
canonical reverse path: it returns an archived thought to `ACTIVE` and clears
the hygiene archive stamps. Two maintenance mechanisms have dedicated archival
paths outside the ordinary `evolve()` sequence:

- TTL cleanup with the `archive` strategy archives any expired row, clears its
  `expires_at`, and clears the hygiene-specific archive stamps.
- Memory Hygiene may archive an eligible `ACTIVE` or `CREATED` row, clears its
  `expires_at`, and stamps `archived_at_cycle` plus `archived_at` for its restore
  windows.

Both remain reversible through `restore_thought()` until a hard-delete or
garbage-collection stage removes the row.

> **`ARCHIVED` is excluded from default retrieval, but still stored and counted.**
> Marking a thought `ARCHIVED` is a *retention* state — the row and its content
> stay in the database — but an archived thought is **dropped from default ranked
> retrieval**: `search_hybrid` / `recall` / `search_fts` / `search_similar` no
> longer return it unless you pass `include_archived=True` (and `restore_thought`
> re-activates it). This retrieval exclusion is the retrieval side of
> [Forgetting](memory-hygiene.md) — an **opt-in, off-by-default** hygiene loop —
> but the exclusion itself applies to **any** archived row (including one archived
> by the TTL `archive` strategy or manually), whether or not that loop is enabled.
> It is **still counted** by `count_thoughts()` /
> `list_thoughts()` (those are not ranked retrieval) — filter on `lifecycle_status`
> yourself to exclude it there — **provided it is not also expired**: both methods
> default to excluding expired rows regardless of lifecycle status, so an
> archived row whose `expires_at` has passed still needs `include_expired=True`
> to be counted or listed. **Expired** thoughts are likewise excluded from
> the four ranked retrieval methods listed above; those methods do not offer an
> `include_expired` escape hatch. `include_expired=True` applies only to
> `list_thoughts()` and `count_thoughts()`. A retired `REFLECTION` (one whose
> `lifecycle_status` is no longer `ACTIVE`) stays out of search by a
> type-specific *freshness floor* even under `include_archived=True`.

## Time-to-live (TTL) and expiry

A thought can carry an expiry time. Three ways to set it:

- **Per-thought, absolute:** set `ThoughtRecord.expires_at` to a timestamp.
- **Per-thought, relative at create time:** pass `expires_after_seconds=` to
  `create_thought(...)`, which computes `expires_at` for you.
- **A default for the whole store:** `ttl.default_ttl_seconds` in config applies a
  default TTL to new thoughts that don't set their own (see
  [Configuration → ttl](configuration.md#ttl)).

When more than one is available at creation, precedence is explicit and
deterministic: the `expires_after_seconds` argument wins, then the record's own
`expires_at`, then the configured store default. The default is used only when
neither per-call nor per-record expiry was supplied.

`expires_at` is stored in one canonical UTC form (`2026-07-01T00:00:00+00:00`),
whichever ISO-8601 form you pass: an offset-aware value is converted to UTC, and
a value with no offset is read as UTC. Expiry compares that stored value with the
current UTC time, so `2026-07-01 00:00:00`, `20260701T000000` and
`2026-07-01T02:00:00+02:00` all expire at the same instant. Upgrading a database
written by an earlier version converts its stored values once; the
[upgrade notes](upgrade.md#06---07) say which values it leaves as they are.

Expiry is **not** automatic on a timer. Expired thoughts remain until a cleanup
pass runs (see [running cleanup](#running-cleanup) below). By default, expired
thoughts are **excluded** from `count_thoughts(...)` and `list_thoughts(...)` —
pass `include_expired=True` to include them:

```python
live = await store.count_thoughts()  # excludes expired
everything = await store.count_thoughts(include_expired=True)
```

Ranked retrieval (`search_hybrid`, `recall`, `search_fts`, and
`search_similar`) always excludes rows whose `expires_at` has passed, even
before cleanup has archived or deleted them. `include_expired` belongs only to
the list/count APIs; it is not a ranked-search parameter.

## Archive vs. delete

What a cleanup pass *does* to an expired thought is governed by the store's TTL
strategy, set via `ttl.strategy` in config (see
[Configuration → ttl](configuration.md#ttl)):

| Strategy | Effect on an expired thought | Reversible? | Content erased? |
|---|---|---|---|
| `"archive"` (default) | Flips `lifecycle_status` to `ARCHIVED`; the row and its `content` stay in the database | Yes | **No** |
| `"delete"` | Removes the thought row from the `thought` table | No | From the live table, yes — but see [residue](#gdpr-and-hard-deletion) |

The default is **`archive`** — chosen so expiry is non-destructive and
auditable. This means **expiry alone does not erase anything** under the default
configuration. To make expiry actually remove rows, set `ttl.strategy: delete`.

## Running cleanup

Expiry is applied by an explicit cleanup pass — nothing happens on a timer.

**From Python:** `cleanup_expired()` returns a `CleanupResult`:

```python
result = await store.cleanup_expired()
print(result.expired_count)  # how many thoughts were expired
print(result.strategy_applied)  # "archive" or "delete" (per config)
print(result.timestamp)  # ISO-8601 time of the pass
```

You can also have the store run cleanup automatically via
`ttl.check_every_n_operations` (default `0` = manual only): the counter
advances on thought create/update calls, not on every store operation, and
triggers a cleanup pass once it reaches *N*.

**From the CLI:** `engrava gc --expired` runs the expiry cleanup per your TTL
strategy. What it does next depends on that strategy:

```bash
engrava gc --expired            # run expiry cleanup (per ttl.strategy)
engrava gc --expired --dry-run  # preview only, changes nothing
engrava gc                      # delete ARCHIVED thoughts + their edges/embeddings/actions
```

**`--dry-run` does not mirror the real run's stop condition under
`ttl.strategy: archive`.** A real `gc --expired` run stops after archiving
when it archived at least one row (below), but `--dry-run` always also
previews the archived-collection step when pre-existing `ARCHIVED` rows
exist — even on a preview where expired rows would have been archived and
the corresponding real run would have stopped there.

- **With `ttl.strategy: delete`:** the expired rows are deleted outright, and the
  same pass then garbage-collects any pre-existing `ARCHIVED` thoughts.
- **With `ttl.strategy: archive` (default):** the expired rows are *archived*
  (marked `ARCHIVED`), and the pass **stops there — but only if it archived at
  least one row.** (Collecting the rows it just archived would defeat the
  soft-retire.) With **no** expired rows to archive it falls through to collecting
  pre-existing `ARCHIVED` rows, exactly as a plain `engrava gc` would. So a
  repeated `gc --expired` under the archive strategy does eventually collect: it
  stops on the run that archives something and collects on the run that does not.

Plain `engrava gc` (no `--expired`) removes `ARCHIVED` thoughts together with
every edge touching one on either end — including edges whose other end is still
live — their embeddings and the actions sourced from them, then reconciles the
vector index by removing every `vec0` row no `embedding` row owns. This is how
archived data is finally deleted from the live table.

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

## GDPR and hard deletion

If you must erase a user's data (e.g. a GDPR erasure request), be aware that
**neither archiving nor a single delete is sufficient on its own**. The
procedure below addresses the following:

1. **Archive does not erase.** Under the default `ttl.strategy: archive`, an
   "expired" thought is only marked `ARCHIVED` — the row and its `content` remain
   in the database. Note that `engrava gc --expired` under the `archive` strategy
   *archives* the rows and stops **only if it archived at least one** — on that run
   it does not also delete archived rows. With no expired rows to archive the same
   command falls through and **does** collect pre-existing `ARCHIVED` rows, so do
   not read "it only archives" as a guarantee that nothing was deleted. To remove
   the row deliberately, run a **separate** `engrava gc`, or use
   `ttl.strategy: delete` so the row is deleted outright.
2. **The audit journal retains a content delta.** If the
   [audit journal](audit-trail.md) is enabled, the thought's content can live in
   that journal independently of how it is later deleted: the original
   `INSERT_THOUGHT` entry (and any `UPDATE_THOUGHT` entry) holds the content in
   its `delta` from whenever it was written through a journal-enabled store
   method, and a journal-enabled `delete_thought()` additionally records a
   `DELETE_THOUGHT` entry carrying the same content — so the data survives in
   `journal_entry` after the thought row is gone, in the insert/update entry if
   nothing else. The CLI's `engrava gc` (and `gc --expired`) construct their own
   connection without journaling, so a deletion made that way records no
   `DELETE_THOUGHT` entry at all even when `journal.enabled` is set in config —
   see [Audit Trail](audit-trail.md#what-gets-recorded) — but that does not
   remove the earlier insert or update entry either way. A true erasure must
   purge every journal entry that carries this content: the original insert
   (and any update), and the delete entry if one exists. Purging a non-tail
   entry breaks the retained chain's linkage and requires an explicit
   re-baseline if journal verification is retained; purging a self-consistent
   suffix, including every entry, does not by itself make local verification
   fail — see
   [Security → Audit journal threat model](security.md#audit-journal-threat-model).
3. **Backups.** Any snapshot or file backup taken before the deletion still
   contains the data. Erasure must extend to your backup retention.

A hard-erasure procedure therefore covers: delete (or archive-then-gc) the
thought rows → purge every `journal_entry` row that carries this thought's
content, if journaling was ever enabled while it existed → roll the deletion
through your backup retention. Don't treat
"the thought no longer appears in search" as "the data is gone." This list is
not exhaustive: it does not address SQLite WAL or freelist page residue (a
page holding the old content can persist in `-wal` or the freelist after the
row is gone, depending on your `secure_delete` setting), nor any export or
snapshot file made after the delete but before you rotate it out, nor
deleted-but-still-cached state on a remote embedding provider that received
the original text — evaluate each against your own threat model.

> **A database below the core-12 schema can carry a deleted thought's
> `embedding` row.** That schema has no `ON DELETE CASCADE` on `embedding`, so
> a delete made by an older engrava build could leave the row, vector
> included, behind. Reconciliation (the pass that runs when a sqlite-vec
> backend opens) backfills a vector only for an `embedding` row
> whose owner id matches a row in `thought`, and `delete_thought` issues its
> own delete for the thought's `embedding` row. The core-12 step of
> `engrava migrate` purges the dangling `embedding` rows that earlier deletes
> left behind. Full mechanism:
> [Known Limitations → Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated).

> **Memory-hygiene GC is a hard delete, not erasure.** The opt-in
> [Forgetting](memory-hygiene.md) garbage-collection stage removes archived rows
> exactly like the deletes above: it reclaims the live/queryable working set, but
> when journaling is on the content survives in the `DELETE_THOUGHT` journal entry.
> Treat it as cognitive cleanup, not guaranteed erasure.

## Reclaiming disk space

Deleting rows — whether via `ttl.strategy: delete`, `engrava gc`, or a hard
erasure — **does not shrink the database file** under SQLite's default
`auto_vacuum=NONE`, which engrava does not override. SQLite returns the freed
pages to an internal free-list and reuses them for future writes; the file
stays the same size on disk. A database externally configured with
`auto_vacuum=FULL` can instead truncate freed tail pages automatically at
commit — engrava neither sets nor requires a particular `auto_vacuum` mode.

To actually reclaim file size on the default configuration you must run
`VACUUM`, which rebuilds the database into a compact file. Plan for its cost:

- **Exclusive lock, with a WAL exception for existing readers.** `VACUUM`
  requires write access and blocks competing writers for its whole duration.
  Because engrava uses WAL mode by default, a reader that already has an
  established snapshot can continue running concurrently with it; a new
  connection still needs to wait its turn. Run it during a maintenance window.
- **Temporary space.** It writes a fresh copy before swapping, so it needs
  additional temporary disk space that depends on the database's contents and
  SQLite's own temporary-storage behavior — measure it for your database
  rather than assuming a fixed multiple.
- **Off-peak.** On a large database this can take a while; schedule it off-peak.

```sql
VACUUM;                 -- rebuild in place (exclusive lock against writers)
VACUUM INTO 'copy.db';  -- write a compacted copy without locking in place as long
```

Until you `VACUUM`, expect the file size to reflect the high-water mark, not the
live row count — this is normal SQLite behaviour, not a leak.

## See also

- [Configuration → ttl](configuration.md#ttl) — the strategy and default-TTL knobs
- [Forgetting (Memory Hygiene)](memory-hygiene.md) — the opt-in, rule-based
  archive-then-GC loop, its reproducibility conditions, and its restore windows
- [Audit Trail](audit-trail.md) — what the journal records (and its delta residue)
- [CLI](cli.md#gc) — the full `engrava gc` option reference
- [Known Limitations](known-limitations.md) — storage and concurrency constraints

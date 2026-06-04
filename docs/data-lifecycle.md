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
| `ARCHIVED` | Retired from active results; retained until garbage-collected. |

You set the status on the `ThoughtRecord` you create, and update it over the
thought's life. Archiving is the soft-retire step: an `ARCHIVED` thought still
exists (and its content is still stored) until you garbage-collect it.

## Time-to-live (TTL) and expiry

A thought can carry an expiry time. Two ways to set it:

- **Per-thought, absolute:** set `ThoughtRecord.expires_at` to a timestamp.
- **Per-thought, relative at create time:** pass `expires_after_seconds=` to
  `create_thought(...)`, which computes `expires_at` for you.
- **A default for the whole store:** `ttl.default_ttl_seconds` in config applies a
  default TTL to new thoughts that don't set their own (see
  [Configuration → ttl](configuration.md#ttl)).

Expiry is **not** automatic on a timer. Expired thoughts remain until a cleanup
pass runs (see [running cleanup](#running-cleanup) below). By default, expired
thoughts are **excluded** from `count_thoughts(...)` and `list_thoughts(...)` —
pass `include_expired=True` to include them:

```python
live = await store.count_thoughts()                      # excludes expired
everything = await store.count_thoughts(include_expired=True)
```

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
print(result.expired_count)       # how many thoughts were expired
print(result.strategy_applied)    # "archive" or "delete" (per config)
print(result.timestamp)           # ISO-8601 time of the pass
```

You can also have the store run cleanup automatically every *N* operations via
`ttl.check_every_n_operations` (default `0` = manual only).

**From the CLI:** `engrava gc --expired` runs the same expiry cleanup (archiving
or deleting per config), then garbage-collects archived thoughts:

```bash
engrava gc --expired            # cleanup expired (per strategy) + gc archived
engrava gc --expired --dry-run  # show what would happen, change nothing
engrava gc                      # gc archived thoughts (+ orphaned edges) only
```

Plain `engrava gc` removes `ARCHIVED` thoughts and their orphaned edges. This is
how archived data is finally deleted from the live table.

## GDPR and hard deletion

If you must erase a user's data (e.g. a GDPR erasure request), be aware that
**neither archiving nor a single delete is sufficient on its own**. Three places
can retain the content:

1. **Archive does not erase.** Under the default `ttl.strategy: archive`, an
   "expired" thought is only marked `ARCHIVED` — the row and its `content` remain
   in the database. You must additionally garbage-collect it (`engrava gc`, or
   `gc --expired` which does both) — or use `ttl.strategy: delete` — to remove the
   row.
2. **The audit journal retains a content delta.** If the
   [audit journal](audit-trail.md) is enabled, deleting a thought does **not**
   remove its content from the journal. The original `INSERT_THOUGHT` entry holds
   the content in its `delta`, and the `DELETE_THOUGHT` entry records the deletion
   delta too — so the data survives in `journal_entry` after the thought row is
   gone. A true erasure must also purge the relevant journal entries (and doing so
   breaks the hash chain from that point — re-baseline if you depend on
   verification).
3. **Backups.** Any snapshot or file backup taken before the deletion still
   contains the data. Erasure must extend to your backup retention.

A correct hard-erasure procedure therefore looks like: delete (or
archive-then-gc) the thought rows → purge the matching `journal_entry` rows if
journaling is on → roll the deletion through your backup retention. Don't treat
"the thought no longer appears in search" as "the data is gone."

## Reclaiming disk space

Deleting rows — whether via `ttl.strategy: delete`, `engrava gc`, or a hard
erasure — **does not shrink the database file**. SQLite returns the freed pages
to an internal free-list and reuses them for future writes; the file stays the
same size on disk.

To actually reclaim file size you must run `VACUUM`, which rebuilds the database
into a compact file. Plan for its cost:

- **Exclusive lock.** `VACUUM` takes an exclusive lock for its whole duration —
  no concurrent reads or writes. Run it during a maintenance window.
- **Temporary space.** It writes a fresh copy before swapping, so it needs
  roughly **2× the database size** in free disk (temp + final) transiently.
- **Off-peak.** On a large database this can take a while; schedule it off-peak.

```sql
VACUUM;                 -- rebuild in place (exclusive lock, ~2x temp space)
VACUUM INTO 'copy.db';  -- write a compacted copy without locking in place as long
```

Until you `VACUUM`, expect the file size to reflect the high-water mark, not the
live row count — this is normal SQLite behaviour, not a leak.

## See also

- [Configuration → ttl](configuration.md#ttl) — the strategy and default-TTL knobs
- [Audit Trail](audit-trail.md) — what the journal records (and its delta residue)
- [Known Limitations](known-limitations.md) — storage and concurrency constraints

# Audit Trail (hash-chain journal)

Engrava can record every change to your thought-graph in an append-only,
hash-linked **journal** — a tamper-evident audit trail. Each entry captures one
mutation (insert / update / delete of a thought or edge) as a before/after
delta, and is cryptographically chained to the previous entry with SHA-256.

> **Read the [Security model](#security-model--guarantees) before relying on this
> for compliance.** The chain detects accidental corruption and naive edits, but
> it is a *keyless* chain stored in the same database file — see the boundary
> below.

## Enabling the journal

Journaling is **off by default** (zero overhead when disabled — the
`journal_entry` table exists but is never written to). Turn it on either via
configuration or the constructor.

In `engrava.yaml`:

```yaml
database:
  path: "./engrava.db"

journal:
  enabled: true
```

```python
from engrava import SqliteEngravaCore

async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    assert store.journal is not None  # journaling is active
```

Or when constructing the store directly:

```python
import aiosqlite
from engrava import SqliteEngravaCore

async with aiosqlite.connect("engrava.db") as conn:
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, journal_enabled=True)
    await store.ensure_schema()
```

`store.journal` returns the `JournalWriter` when journaling is enabled, or
`None` when it is off — so a quick `if store.journal is not None:` guards any
journal-specific code.

## What gets recorded

When journaling is enabled, the store records a journal entry **automatically**
on every mutation of a thought or an edge — you do not call the journal
yourself. The recorded `mutation_type` values (the `MutationType` enum) are:

| `MutationType` | When |
|---|---|
| `INSERT_THOUGHT` | `create_thought()` |
| `UPDATE_THOUGHT` | `update_thought()` |
| `DELETE_THOUGHT` | `delete_thought()` (only when a row was actually deleted) |
| `INSERT_EDGE` | `create_edge()` |
| `UPDATE_EDGE` | `update_edge()` |
| `DELETE_EDGE` | `delete_edge()` (only when a row was actually deleted) |

Each entry's `delta` is a `{"before": ..., "after": ...}` dictionary: inserts
have `before: null`, deletes have `after: null`, and updates carry both sides.

> **Not recorded:** embeddings (`store_embedding`) and action records
> (`create_action`) are **not** written to the journal — the audit trail covers
> the thought-and-edge graph, not the embedding or action tables. This also
> matters for backups — see [Backup note](#backup--retention-note).

**TTL expiry is recorded.** `cleanup_expired()` (and the auto-cleanup it
triggers) goes through the same journaled paths, so expiry of a thought is
captured according to the configured TTL strategy:

- **archive** strategy → an `UPDATE_THOUGHT` entry (the thought's
  `lifecycle_status` flips to `ARCHIVED` and `expires_at` is cleared; the delta
  carries the before/after).
- **delete** strategy → a `DELETE_THOUGHT` entry (`after: null`).

(The separate `engrava gc` CLI command, which physically purges already-archived
rows, operates at the storage layer and is not journaled.)

## The `JournalEntry` schema

Each entry is an immutable `JournalEntry`:

| Field | Type | Meaning |
|---|---|---|
| `entry_id` | `str` | Stable UUID for this entry |
| `sequence_number` | `int` | Monotonic, gapless position in the chain (starts at 1) |
| `mutation_type` | `str` | One of the `MutationType` values above |
| `target_id` | `str \| None` | The affected `thought_id` / `edge_id` |
| `delta` | `dict` | `{"before": {...}, "after": {...}}` diff |
| `parent_hash` | `str \| None` | SHA-256 of the previous entry (`None` for the first entry) |
| `entry_hash` | `str` | SHA-256 of this entry's canonical content |
| `created_at` | `str` | ISO-8601 UTC timestamp |

The hash is computed over the canonical string
`"{sequence_number}|{mutation_type}|{target_id}|{json(delta, sort_keys)}|{parent_hash}"`
via `JournalWriter.compute_hash(...)` (a static method, exposed for callers who
want to recompute a hash independently).

## Querying history

Use `store.journal.get_entries(...)` to read the trail. All filters are
optional; results are ordered by `sequence_number` ascending.

```python
# Everything that ever happened to one thought:
history = await store.journal.get_entries(target_id="thought-001")
for entry in history:
    print(entry.sequence_number, entry.mutation_type, entry.created_at)

# Only deletions, since a timestamp, capped:
deletions = await store.journal.get_entries(
    mutation_type="DELETE_THOUGHT",
    since="2026-01-01T00:00:00+00:00",
    limit=500,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `target_id` | `None` | Filter by the affected entity ID |
| `mutation_type` | `None` | Filter by mutation type string |
| `since` | `None` | ISO-8601 lower bound on `created_at` (inclusive) |
| `limit` | `100` | Maximum entries returned |

## Verifying integrity

Verification walks the whole chain in order, recomputes every hash, and checks
the parent-hash linkage, returning a `JournalIntegrityResult`. There are three
ways to run it, all backed by the same walk:

| Entry point | Use it when |
|---|---|
| `store.verify_journal()` | You have a store and want a one-call check. **Verifies the on-disk chain even when journaling is currently disabled** (see below). |
| `store.journal.verify_integrity()` | You are already holding the `JournalWriter` (only available while journaling is enabled). |
| [`engrava verify`](cli.md#verify) | From the shell / CI / a pre-backup hook — exit `0` = intact, `1` = broken or missing database. |

```python
result = await store.verify_journal()
if result.valid:
    print(f"Chain OK — {result.entries_checked} entries verified.")
else:
    print(
        f"Tampering or corruption detected at sequence "
        f"{result.first_invalid_sequence}: {result.error_message}"
    )
```

| Field | Type | Meaning |
|---|---|---|
| `valid` | `bool` | `True` if every hash and link checks out |
| `entries_checked` | `int` | Number of entries verified |
| `first_invalid_sequence` | `int \| None` | Sequence of the first broken entry, or `None` |
| `error_message` | `str \| None` | Description of the first error, or `None` |

An empty journal verifies as `valid=True` with `entries_checked=0`.

**Verification is independent of the current `journal.enabled` state.** Entries
are recorded only while journaling is on, but once written they stay in the
`journal_entry` table. `store.verify_journal()` (and `engrava verify`) audit
whatever chain is on disk — so a database that had journaling enabled in an
earlier session and reopened with it **off** (`store.journal is None`) is still
fully auditable. (`store.journal.verify_integrity()` is unavailable in that case
because `store.journal` is `None`; use `store.verify_journal()`.)

**Run verification on a schedule** (e.g. before each backup, during incident
response, or as a periodic monitoring check) rather than only ad hoc — that is
what turns the chain from a passive structure into an active control.

### Verifying automatically on open

Set `journal.verify_on_open: true` to run the walk **once, when the store is
opened** through `SqliteEngravaCore.from_config(...)`. The check runs after the
schema is ensured and **raises `JournalIntegrityError`** (a subclass of
`EngravaError`, carrying `first_invalid_sequence` and `error_message`) instead of
returning a store when the chain does not verify — a fail-closed startup gate.

```yaml
journal:
  enabled: true
  verify_on_open: true   # refuse to open on a broken chain
```

```python
from engrava import JournalIntegrityError, SqliteEngravaCore

try:
    async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
        ...
except JournalIntegrityError as exc:
    # Startup aborted: the on-disk journal did not verify.
    print(f"broken at sequence {exc.first_invalid_sequence}: {exc.error_message}")
```

`verify_on_open` is **independent of `enabled`** (a recorded chain is checked
even if journaling is now off) and **defaults to `false`**. Leave it off unless
you want the gate: the walk is `O(entries)`, so on a large journal it adds a
one-time cost to every open. For periodic rather than on-open checking, call
`store.verify_journal()` / `engrava verify` on a schedule instead.

## Worked example

```python
import aiosqlite
import uuid
from engrava import (
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    Priority,
    LifecycleStatus,
)

async with aiosqlite.connect(":memory:") as conn:
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, journal_enabled=True)
    await store.ensure_schema()

    note = ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence="User prefers email over phone",
        content="Stated during onboarding call.",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="human",
    )
    await store.create_thought(note)
    await store.update_thought(note.thought_id, essence="User strongly prefers email")

    # Two entries were recorded automatically (INSERT_THOUGHT, UPDATE_THOUGHT).
    entries = await store.journal.get_entries(target_id=note.thought_id)
    assert [e.mutation_type for e in entries] == ["INSERT_THOUGHT", "UPDATE_THOUGHT"]

    # The chain verifies.
    result = await store.journal.verify_integrity()
    assert result.valid and result.entries_checked == 2
```

## Security model & guarantees

The journal is a **keyless** SHA-256 integrity chain stored **in the same
SQLite file** it protects. `verify_integrity()` recomputes each entry's hash
from that entry's own stored data — there is no secret key, HMAC, signature, or
external anchor.

**What it protects against (in scope):**

- **Accidental corruption** — bit-rot, a truncated file, a half-written row: the
  recomputed hash or the parent linkage will not match, and verification fails.
- **Naive tampering** — someone who edits, deletes, or reorders a journal row
  (or an audited record) *without* recomputing the rest of the chain: the break
  is detected at the first inconsistent entry.

**What it does NOT protect against (out of scope):**

- **A chain-aware actor with write access to the database file.** Because the
  chain is keyless and self-contained, anyone who can write to the `.db` can
  edit an entry **and** recompute every subsequent hash, producing a fully
  self-consistent chain that passes `verify_integrity()` with `valid=True`. The
  journal is **not** forgery-proof against an adversary (including the agent
  process itself) who controls the file.

If you need genuine, multi-party tamper-evidence, treat the in-file chain as one
layer and add at least one of:

- **Restrict write access** — store the `.db` on a volume only the trusted
  writer process can modify (OS file permissions / ownership).
- **Anchor the chain externally** — periodically export the latest
  `entry_hash` (the chain tail) to an append-only / WORM store, a signed log, or
  another system out of the writer's control. A later `verify_integrity()` plus
  a match against the externally-anchored tail hash detects a full-file rewrite.
- **Verify on a schedule** — run `verify_integrity()` from a separate monitored
  process so a detected mismatch raises an alert.

State this boundary plainly to stakeholders: Engrava's journal gives you
**integrity detection for accidental damage and unsophisticated edits**, not
cryptographic non-repudiation against a file-level adversary.

## Backup & retention note

The logical snapshot/restore path (`engrava snapshot` / `engrava restore`)
covers the thought / edge / embedding / action tables — it does **not** include
the `journal_entry` table. A snapshot is therefore **not** a backup of the audit
trail, and restoring from one starts a fresh chain. To preserve the journal,
back up the database file itself (see the upgrade/backup guidance), and note
that hard-deleting an audited thought still leaves its content in the journal's
`before`/`after` delta — relevant when handling erasure requests.

## See also

- The [Enabling the journal](#enabling-the-journal) section above is the
  canonical reference for the `journal.enabled` configuration flag; the general
  [Configuration](configuration.md) guide covers the rest of `engrava.yaml`.
- [API Reference](api-reference.md) — the broader public API (the journal
  classes `JournalWriter` / `JournalEntry` / `JournalIntegrityResult` and the
  `MutationType` enum are documented on this page).

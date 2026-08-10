# Audit Trail (hash-chain journal)

Engrava can record changes to your thought-graph in an append-only,
hash-linked **journal** — a **tamper-evident thought/edge journal** (not a
whole-database audit). Each entry captures one mutation — insert / update /
delete of a **thought** or **edge**, or an **action `status`/`verification_status`
transition** — as a before/after delta, cryptographically chained to the previous
entry with SHA-256. See [What gets recorded](#what-gets-recorded) for the exact
scope (embeddings and action *creation* are not covered).

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
on every mutation of a thought or edge — and on an action state-transition — you
do not call the journal yourself. The recorded `mutation_type` values are:

| `mutation_type` | When |
|---|---|
| `INSERT_THOUGHT` | `create_thought()` |
| `UPDATE_THOUGHT` | `update_thought()` |
| `DELETE_THOUGHT` | `delete_thought()` (only when a row was actually deleted) |
| `INSERT_EDGE` | `create_edge()` |
| `UPDATE_EDGE` | `update_edge()` |
| `DELETE_EDGE` | `delete_edge()` (only when a row was actually deleted) |
| `UPDATE_ACTION` | `update_action()` — only when an action's `status` / `verification_status` actually changes |

> The first six values are members of the `MutationType` enum; `UPDATE_ACTION` is
> a stored `mutation_type` **string** emitted by `update_action()` — the enum
> itself is not extended. `mutation_type` is a free-text column, so verification
> covers the entry regardless.

Each entry's `delta` is a `{"before": ..., "after": ...}` dictionary: inserts
have `before: null`, deletes have `after: null`, and updates carry both sides.

> **Recorded precisely — what the chain does and does not cover.** The journal
> covers **thought and edge mutations** (insert / update / delete) and **action
> state-transitions** (`update_action`). It does **not** record: embeddings
> (`store_embedding`), **action *creation*** (`create_action` — only an action's
> later transitions are journaled, not its initial insert), or read-derived
> access telemetry (`access_count` / `last_accessed_at`, which is regenerable
> and deliberately outside the chain). So this is a **tamper-evident
> thought/edge/action-transition journal**, not a whole-database audit — verify
> the parts it covers, and do not assume coverage of the embedding or
> action-creation tables. This also matters for backups — see
> [Backup note](#backup--retention-note).

**TTL expiry is recorded.** `cleanup_expired()` (and the auto-cleanup it
triggers) goes through the same journaled paths, so expiry of a thought is
captured according to the configured TTL strategy:

- **archive** strategy → an `UPDATE_THOUGHT` entry (the thought's
  `lifecycle_status` flips to `ARCHIVED` and `expires_at` is cleared; the delta
  carries the before/after).
- **delete** strategy → a `DELETE_THOUGHT` entry (`after: null`).

(The separate `engrava gc` CLI command, which physically purges already-archived
rows, operates at the storage layer and is not journaled.)

Deleting a thought also removes its incident edges, embeddings, and actions
through database cascades (and purges its vector-index row). When deletion goes
through a journaled store path, the only entry for that operation is
`DELETE_THOUGHT`: cascade-removed edges do not receive `DELETE_EDGE` entries,
and cascade-removed embeddings, actions, and vector-index rows receive no
separate journal entry. Embedding/vector mutations and action creation/deletion
are outside journal coverage generally; only action status and verification
transitions are covered. Call `delete_edge()` explicitly when an individually
journaled edge deletion is required.

**Below core schema 12 this cascade does not happen.** The `ON DELETE CASCADE` on
`edge`, `embedding` and `action` arrives with the core-12 migration, so on a database
carried forward from an older engrava and never migrated the thought's `embedding` row
outlives the delete. The delete does still purge that thought's own `vec0` vector, so
the identifier is **not** reachable straight afterwards; it returns once the reconcile
that runs on the next sqlite-vec-enabled open backfills the index from the surviving
`embedding` row. From then on it is an ordinary candidate on that arm whenever a
sqlite-vec backend is **active** on the store and the query carries no effective
metadata predicate — the arm *can* return it, subject to the same similarity threshold
and `top_k` window as any live row. Run `engrava migrate`. See
[Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated).

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

> **Entry ordering and content are tamper-evident; entry timestamps are not.**
> Read the canonical string above for what the chain covers: `sequence_number`
> and `parent_hash` bind each entry's position in the order, and `mutation_type`,
> `target_id`, and `delta` bind what it says happened — change any of them
> without recomputing the affected hashes and verification fails. `created_at`
> and `entry_id` are **not** in the hash preimage, so rewriting them changes no
> hash: a journal whose every `created_at` has been rewritten still verifies as
> `valid=True`. Treat `created_at` as an informative timestamp, not as evidence
> of when a mutation happened. This is a property of the current chain format,
> not a bug with a cheap fix — bringing `created_at` into the preimage changes
> the expected hash of every entry ever written, so covering it is a
> chain-format migration or verifier-versioning decision (keeping existing
> journals verifiable is part of the problem, not a detail after it), never a
> one-line change. See
> [Security model & guarantees](#security-model--guarantees).

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

> **`since=` is a convenience filter, not an audit boundary.** It compares
> `created_at >= since`, and `created_at` is outside the hash preimage (as is
> `entry_id` — see [the schema note above](#the-journalentry-schema)). Rewriting
> an entry's `created_at` **across** a window's lower bound can therefore remove
> it from **or add it to** that window while `verify_journal()` still reports
> `valid=True` — the two effects compound instead of one catching the other. A
> rewrite that stays inside one particular window leaves membership in that window
> unchanged, but it still changes the timestamp `get_entries` returns to the
> caller and may change membership under other lower bounds; because the deciding
> column is unprotected, no `since=` result is evidence. Use `since=` to narrow a
> report over a trusted journal; do not treat "nothing since T" as proof that
> nothing happened since T, nor an entry's presence in the window as proof that it
> happened after T. Chain-covered fields (`sequence_number`, `parent_hash`) are
> what a
> time-bounded claim has to be anchored on: record the
> `(sequence_number, entry_hash)` you saw at T and audit forward from that
> sequence number.

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

**What this check proves.** Verification proves that every journal row still on
disk has self-consistent hashed content and links to the preceding row that was
walked. It detects an in-row journal mutation, a changed parent link, or a
mid-chain deletion/reordering when the affected hashes have not been
recomputed. It does **not** provide any of these separate guarantees:

- **Chain length / tail completeness.** Removing a self-consistent suffix leaves
  a valid prefix, so `verify_journal()` cannot know that newer entries once
  existed. Detect this with an externally retained high-water mark and tail
  anchor, such as the expected `(sequence_number, entry_hash)` at a checkpoint.
- **Entry timestamps.** `created_at` is outside the hash preimage, so a chain
  whose timestamps have all been rewritten verifies exactly like an untouched
  one. What the walk proves is the order and the content of the entries it read,
  not when they were written.
- **Live-table reconciliation.** The verifier reads `journal_entry`; it does not
  compare the current thought, edge, embedding, or action tables with the latest
  journal deltas. A direct edit to a live thought or edge row is outside this
  check unless the edit also changes a journal row or breaks the SQLite file.

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

**What it protects against (in scope):** the **ordering and content** of the
entries still on disk — both are in the hash preimage, so neither can be changed
without breaking the chain unless the affected hashes are recomputed.

That preimage is the five fields joined by `|` with no escaping, an absent field
written as the empty string:
`"{sequence_number}|{mutation_type}|{target_id}|{json.dumps(delta, sort_keys=True)}|{parent_hash}"`.
Nothing in the encoding marks where one field ends and the next begins, so the
fields stay distinguishable because of what they can hold, not because the format
separates them. **For the entries the store emits** those grammars settle it:
`sequence_number` is decimal digits, `mutation_type` is one of seven fixed
literals, `parent_hash` is a SHA-256 hex digest (empty on the first entry), and
`delta` is always a JSON object dump — whose quoting rules a caller-chosen
`target_id` cannot imitate, so no identifier written through the store can move a
field boundary.

**That guarantee does not extend to arbitrary `JournalWriter.append()` calls.**
`JournalWriter` is exported, and `append()` does not enforce the `mutation_type`
vocabulary before hashing, so a caller driving it directly can write two distinct
entries with one preimage and therefore one hash — `mutation_type="INSERT_THOUGHT"`
with `target_id="x|"` collides with `mutation_type="INSERT_THOUGHT|x"` and no
`target_id`. Exploiting it takes in-process code execution against your own store,
which is already past the boundary this chain defends; treat the binding as a
property of the store's own field grammars, not of the encoding.

Verification also re-serialises the stored `delta` before hashing it
(`json.loads` in, `json.dumps(..., sort_keys=True)` out), so what the chain binds
is the delta's **decoded value**, not its stored bytes: rewriting the blob's key
order, whitespace, or escaping without changing the value it decodes to verifies
clean. Nothing the delta *says* can change that way — but do not read a matching
hash as proof that the stored bytes are the ones originally written.

- **Accidental corruption inside the retained journal rows** — changed hashed
  content or parent linkage makes verification fail.
- **Naive journal tampering** — someone who edits, deletes from the middle, or
  reorders a journal row *without* recomputing the affected chain: the break is
  detected at the first inconsistent entry.

**What it does NOT protect against (out of scope):**

- **Self-consistent tail removal.** The retained prefix still verifies. An
  external high-water mark or tail anchor is required to prove expected length.
- **Entry timestamps (`created_at`) and `entry_id`.** Neither is in the hash
  preimage. Rewriting every `created_at` in the journal — backdating the whole
  trail — leaves a chain that verifies as `valid=True`. A rewrite that moves an
  entry across a `get_entries(since=...)` lower bound also changes that window's
  contents, in either direction: downward hides the entry from the window,
  upward plants it inside one it did not belong to. The filter reads the same
  unprotected column, so a timestamp-based claim about the journal ("no
  deletions in the last 24 hours", or "this deletion happened during the
  incident") rests on data the chain does not protect. Anchor time claims on the
  chain-covered `(sequence_number, entry_hash)` pair recorded at a known point
  instead.
- **Direct edits to live records.** Verification does not replay the journal or
  reconcile thought/edge/action state against it.
- **A chain-aware actor with write access to the database file.** Because the
  chain is keyless and self-contained, anyone who can write to the `.db` can
  edit an entry **and** recompute every subsequent hash, producing a fully
  self-consistent chain that passes `store.verify_journal()` (or the enabled
  writer's `store.journal.verify_integrity()`) with `valid=True`. The
  journal is **not** forgery-proof against an adversary (including the agent
  process itself) who controls the file.

If you need genuine, multi-party tamper-evidence, treat the in-file chain as one
layer and add at least one of:

- **Restrict write access** — store the `.db` on a volume only the trusted
  writer process can modify (OS file permissions / ownership).
- **Anchor the chain externally** — periodically export the latest
  `(sequence_number, entry_hash)` to an append-only / WORM store, a signed log,
  or another system out of the writer's control. A later integrity walk plus a
  match against that checkpoint adds the length/tail expectation the in-file
  walk does not possess.
- **Verify on a schedule** — run `store.verify_journal()` from a separate
  monitored process so a detected mismatch raises an alert. This store-level
  entry point also works when journaling is currently disabled.

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

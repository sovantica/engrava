# Error Handling and Recovery

This guide describes the error and persistence contracts of Engrava v0.7. It is
for application code that must decide whether to fix an input, retry a remote
call, reconcile a partial write, or replace a store.

The central rule is: **classify the failure and establish what committed before
retrying.** An exception does not universally mean "nothing was written," and
Engrava does not make every write idempotent.

## Decision table

| Signal | State after the error | Action | Retry unchanged? |
|---|---|---|---|
| `ConfigError`, validation `ValueError`, invalid filter/recency/cycle, dimension or prefix mismatch | Rejected at a validation boundary in the usual path | Correct the configuration or arguments | No |
| `ThoughtNotFoundError`, `ActionNotFoundError`, `SourceThoughtNotFoundError`, `ReferentialIntegrityError` | Required data is absent | Refresh identifiers or create the missing parent deliberately | No |
| `DuplicateEdgeError` | The requested directed, typed relationship already exists | Treat the write as idempotent success or choose a different direction/type | No |
| `InvalidTransitionError`, `ReadOnlyViolationError` | Requested operation is not allowed in the current state/view | Change the requested transition or use a writable capability | No |
| `StaleDataError` | The guarded update matched no row — another guarded write bumped the row's `revision` in between, **or deleted the row**; **nothing** of the rejected update was written | Re-read, recompute the intended change, then issue a new update — and handle the row being gone | No, not without re-reading |
| Remote embedding timeout, network error, `408`, `409`, `425`, `429`, or selected `5xx` | Depends on the operation; a single thought/update may already be committed | Let the provider exhaust its bounded retry policy, then reconcile by thought ID | Not the whole write blindly |
| SQLite `OperationalError` containing `locked` or `busy` | The operation did not complete successfully; reconcile durable state when the write boundary is ambiguous | Reduce writer contention or increase `busy_timeout`; retry only an operation known to converge | Only with operation-specific proof |
| `WriteContentionError` | The call could not get the cross-connection write lock, after `busy_timeout` and, for the dedup calls, the store's own bounded retries; the failed attempt wrote nothing | Retry the call; if it keeps recurring, remove the contention (see SQLite lock contention below) | Yes, the call itself |
| `WriteLockTimeoutError` | A task could not get the store's in-process write lock within `write_lock_acquire_timeout_seconds`; the guarded write did not start | Find the cause first: a task spawned and awaited from inside another task's open `suspend_auto_commit()` window (drive writes from the one task that opened it), or a bound too small for a slow embedding provider (raise it) | Not until the cause is fixed |
| `DedupLockReentryError` | A task re-entered the dedup window it already holds and was refused instead of blocking on itself | Fix the caller: an `update_thought()` override reached from `upsert_by_hash()`'s hit branch must not call back into a dedup entry point | No |
| Rising `fts_match_failure_count` | Search retried with sanitized FTS syntax; the vector and other hybrid arms can still contribute | Inspect warning logs and offending queries | Engrava already retries the FTS arm once |
| Rising `vector_arm_degradation_count` | The vector arm returned no results for an empty, zero, or non-finite query vector | Fix query embedding generation | No; the same vector degrades again |
| `EmbeddingGenerationError` from single-item create/update | If this call owns its outermost transaction, the thought/update is committed; embedding is missing (create), or, only if the row already had one, left in place and now stale (update) — otherwise still none. Nested inside the caller's own `suspend_auto_commit()`, nothing is durable yet: the *outermost* window's exit decides, and rollback reverts to whatever existed when that outermost window opened, not just to before this call — an earlier write to the same thought inside the same window is undone too, and a thought created inside that window no longer exists at all after rollback | If this call owned its transaction: look up the thought and repair its embedding, do not recreate it. If it was nested and the outer window rolled back, do not assume based on which call you issued — look the thought up by id first: absent means nothing this window wrote for it survived, including an earlier create in the same window — create it; present means it reverted to whatever state existed before the window opened — reissue the write against that state (a blind create instead raises `ValueError` on the still-existing id) | Only when durable: retry embedding, not creation/update. Otherwise, look the thought up first and branch on what you find: create it if absent, or reissue the update if it is present — do not retry blindly based on which call you originally issued |
| `JournalIntegrityError`, `ExtensionMigrationError`, `SchemaVersionError`, `CoreMigrationError` | Store opening or migration was rejected; `CoreMigrationError` leaves the schema version at the last fully applied step | Stop writes, preserve the files, diagnose or restore. For `SchemaVersionError`, read `reason`: `newer_than_head` needs a newer Engrava; for the two sub-floor reasons, do not delete or re-initialise the file (see below) | No |
| `EngravaError` caught as the base class | Depends on the subclass; the base type says nothing about what committed | Handle the specific subclasses above first and put a base-class handler after them; treat anything that reaches it as unclassified | Not on the base type alone |
| `ConnectionQuarantinedError` | The store instance is terminally unusable | Close it, create a new store over a fresh connection, then reconcile durable state | Never on the same store |

## Caller, data, and configuration errors

These failures require a changed request, not backoff:

- `ConfigError` means configuration loading, provider resolution, or a bounded
  configuration value failed validation.
- `ValueError` is also used by some public write boundaries, including duplicate
  thought IDs and invalid metadata/provenance. Pydantic may raise its own
  validation error while constructing records.
- `InvalidFilterError` and `InvalidFilterPathError` reject an invalid compiled
  metadata or visibility filter before query execution.
- `RecencyModeConflictError`, `InvalidRecencyArgumentError`, and
  `CycleProviderError` identify an invalid recency request or provider result.
- `EmbeddingModelMismatchError`, `EmbeddingQueryPrefixMismatchError`, and
  `VectorDimensionMismatchError` protect the corpus from incompatible vectors.
  Restore the matching model/prefix/dimension or deliberately re-embed the
  corpus; repeating the same call cannot repair the mismatch.
- `EmbeddingProviderContractError` identifies a custom embedding provider that
  does not expose a required `EmbeddingProviderProtocol` member — today a public
  `dimension`. It names the provider class and the member; retrying cannot help,
  add the property. A provider that *has* the property and fails inside it does
  not raise this: its own exception propagates unchanged, and is classified by
  whatever raised it.
- `ThoughtNotFoundError`, `ActionNotFoundError`,
  `SourceThoughtNotFoundError`, and `ReferentialIntegrityError` identify a
  missing target or parent.
- `DuplicateEdgeError` identifies an existing `(from, to, type)` relationship;
  callers can safely use it as the idempotent-create signal without parsing
  SQLite error text.
- `InvalidTransitionError` rejects an illegal lifecycle/action transition, and
  `ReadOnlyViolationError` rejects a write through `ReadOnlyEngrava`.
- `StaleDataError` means a guarded update matched no row: another guarded write
  bumped the row's `revision` between your read and your write, or the row was
  deleted in that window. The error does not say which, and nothing of the
  rejected update was written. It is recoverable only through a new
  read-modify-write cycle: re-read the record, recompute the change, and do not
  replay stale `changes` without checking the newer record. Do not assume the
  record still exists, because a row deleted mid-call raises this too. The
  engine bumps `revision` on every guarded write, so a guarded update whose
  read-to-write window another writer's guarded write lands in raises this
  error, whatever field that other write touched, instead of silently
  overwriting that write. Between ordinary tasks sharing one store the write
  lock serialises guarded writes, so it almost never fires there; the shapes it
  does catch are a write from a second store on the same file and a same-task
  nested write from a caller-owned hook — see
  [Concurrency](concurrency.md#optimistic-concurrency-and-staledataerror).
- `WriteLockTimeoutError` and `DedupLockReentryError` are raised instead of a
  hang. The first means a task waited longer than
  `write_lock_acquire_timeout_seconds` for the store's write lock; the exception
  alone does not say whether a task was spawned and awaited from inside another
  task's open `suspend_auto_commit()` window (a caller bug: issue the window's
  writes from the one task that opened it) or the bound is too small for a
  slow embedding provider or large batch (raise it). The second means the same
  task tried to enter the dedup window it already holds, typically an
  `update_thought()` override reached from `upsert_by_hash()`'s hit branch that
  calls back into `create_thought(deduplicate=True)`, `get_or_create()`,
  `upsert_by_hash()` or `bulk_store(deduplicate=True)`. Repeating that call unchanged
  cannot help; see
  [Concurrency](concurrency.md#a-deadlock-this-store-cannot-resolve-raises-it-does-not-hang)
  and
  [Extension hooks](extension-hooks.md#1b3-a-pre-existing-restriction-update_thought-on-upsert_by_hashs-hit-branch).
- `DerivedRecordError` means a producer violated a derivation gate or collided
  with an unrelated identity. The source thought is already durable on the
  automatic post-store path; some earlier derived children may also be durable.

`JournalIntegrityError` and `ExtensionMigrationError` are operator failures,
not caller mistakes. Do not keep opening the same files in a write-capable
process. Preserve the database and WAL files, inspect the reported sequence or
migration, and follow [Backup & Recovery](backup-and-recovery.md).

`SchemaVersionError` and `CoreMigrationError` are the same kind of failure,
raised by `ensure_schema()` and so by `from_config()`,
`EngravaManager.get_store()`, and `engrava migrate`. `SchemaVersionError`
carries `current_version` and a `reason`:

- `newer_than_head`: the file was written by a newer Engrava. `engrava migrate`
  cannot help; upgrade Engrava before opening it.
- `populated_sub_floor` and `stale_shape_sub_floor`: the file is stamped below
  the bootstrap floor and this build cannot tell which older version produced
  it, so it refuses instead of stamping it current. Do not delete or
  re-initialise it — even when no core table holds a row, it may carry data
  outside the core tables — preserve it and follow Backup & Recovery.

`CoreMigrationError` means a migration step returned without leaving the
structure it targets (a column, table, index, foreign key, or trigger) in place.
The version is not stamped past the last fully applied step, so the next
`ensure_schema()` retries the remaining steps rather than treating a
half-migrated file as current; repeating the open without changing anything is
not a fix. Replace the database from a backup, or use a build that matches its
schema history.

`EngravaError` is the base of every typed error above except `ConfigError`,
which derives from `ValueError`. It is not raised directly and carries no
guarantee about what committed, so a handler for it belongs after the specific
ones: Python runs the first `except` clause that matches, and a base-class
clause written first hides the specific recovery below it. A raw
`sqlite3.OperationalError` is not an `EngravaError` either.

The complete exception surface is listed in the
[API Reference](api-reference.md#exceptions).

## Remote embedding failures

Provider behavior is deliberately provider-specific:

- `OpenAICompatibleProvider` retries transport timeouts/network errors and HTTP
  `408`, `409`, `425`, `429`, `500`, `502`, `503`, and `504`. It makes at most
  `max_attempts` attempts (default `3`), waiting
  `base_retry_delay_s * attempt_number` before each next attempt (default waits:
  1 second, then 2 seconds). Other non-success statuses, such as `400`, `401`,
  `403`, and `404`, fail immediately.
- `OllamaProvider` and `HuggingFaceProvider` do not add an Engrava retry loop.
  Their dependency/provider exceptions propagate.
- Local, callback, and third-party providers define their own failure behavior.

Once provider attempts are exhausted, auto-embedding never swallows the error.
With `require_embedding=true`, Engrava wraps it in
`EmbeddingGenerationError`; otherwise the provider's original exception is
re-raised. Strict mode changes the exception type, **not** the single-item
commit boundary.

For a search that only failed while generating its query vector, retrying the
search after a transient provider recovery does not replay a graph write. For a
write with auto-embedding, inspect the persistence boundary below before doing
anything else. See [Embeddings](guides/embeddings.md) for provider setup and
retry controls.

## Persistence boundaries

### Single thought creation

Outside `suspend_auto_commit()`, `create_thought()` (and therefore `remember()`)
commits the thought and its journal entry before auto-embedding, TTL cleanup,
`on_store`, and automatic derivation run. A later failure leaves the source
thought durable.

Consequences:

- An embedding-provider failure leaves a newly-created thought without an
  embedding. It remains available to direct/FTS retrieval but is absent from
  vector search.
- An `on_store` or derivation failure does not roll back the source. Depending on
  where derivation failed, earlier derived children may already be committed.
- Retrying `create_thought()` with the same ID raises `ValueError`; retrying
  `remember()` creates another random ID. Neither is a recovery strategy.

Use strict embedding mode when callers need the typed thought ID for repair:

```python
from engrava import EmbeddingGenerationError


async def create_with_partial_state(store, thought):
    try:
        return await store.create_thought(thought), True
    except EmbeddingGenerationError as exc:
        persisted = await store.get_thought(exc.thought_id)
        if persisted is None:
            raise RuntimeError("embedding failure did not leave the expected thought") from exc
        return persisted, False
```

The `False` result means "the thought exists but embedding repair is required,"
not "creation failed atomically."

If the application retains the provider, it can rebuild the same payload and
role-aware dispatch used by auto-embedding, then upsert the missing vector:

```python
from engrava import RoleAwareEmbeddingProvider, ThoughtNotFoundError


async def repair_embedding(store, provider, thought_id):
    thought = await store.get_thought(thought_id)
    if thought is None:
        raise ThoughtNotFoundError(thought_id)

    essence = thought.essence.strip()
    content = thought.content.strip()
    payload = thought.content if content.startswith(essence) else f"{thought.essence}\n{thought.content}"

    if isinstance(provider, RoleAwareEmbeddingProvider):
        vector = await provider.embed_document(payload)
    else:
        vector = await provider.embed(payload)

    return await store.store_embedding(
        thought_id,
        vector,
        model_name=provider.model_name,
    )
```

Use the same provider/model and role prefixes that built the corpus. The store's
model lock rejects an incompatible model or dimension.

### Thought updates

`update_thought()` commits the updated row and journal entry before re-embedding
when `essence` or `content` changed, provided this call owns its outermost
transaction. Nested inside the caller's own `suspend_auto_commit()` window,
nothing is durable yet on any path — the outermost window's exit decides,
rolling back an uncaught failure and everything it inserted. If the update
does commit and query generation then fails, the row contains the new text
but:

- a thought that had no embedding remains unembedded, and stays unfindable by
  vector search;
- a thought that had an embedding retains the old, now stale vector, and
  stays findable by vector search against that outdated content;
- dependent REFLECTION centroids may not yet have been rebound.

Re-read the thought and repair/recompute enrichment. Do not replay the update
unchanged and assume that doing so re-embeds: an update that no longer changes
`essence` or `content` does not enter the re-embedding branch.

### Bulk ingest

`bulk_store()` is different. Source rows, journal entries, and batch-generated
embeddings run inside one `suspend_auto_commit()` transaction. Called on its
own, any exception during that phase, including a provider exception or a task
cancellation, rolls the batch back and nothing from it is persisted. The
`require_embedding` option controls whether that provider failure is wrapped as
`EmbeddingGenerationError`; it does not change the rollback. The one window the
rollback does not cover is a cancellation that lands during the final commit,
described under transaction contexts below.

Nested inside a caller's own `suspend_auto_commit()` window, the batch shares
that outer transaction: a row error still aborts the batch's own inserts, but
only the outermost window's exit decides commit or rollback. If the caller
catches the propagated exception and the outer window then exits cleanly, the
batch's successful prefix commits along with the rest of the outer block's
work instead of being rolled back.

Automatic derivation runs only after the batch commits. A derivation failure can
therefore leave the complete source batch durable and zero or more derived
children durable. Reconcile sources by ID/content and use `derive_existing()` to
complete deterministic derivation rather than blindly replaying `bulk_store()`.

### Derived records

The source is durable before automatic derivation. Each derived child's row is
then committed as its own unit before its embedding and `DERIVED_FROM` edge are
completed. Re-running `derive_existing(source_id)` reuses deterministic children
and edges and fills missing enrichment, provided the producer obeys the
deterministic content contract. A failed child undoes only its own failing
step — the row and its journal entry, the edge and its journal entry, or an
embedding attempt — never an earlier child's work, and never a caller's other
pending writes when `derive_existing` runs inside the caller's own
`suspend_auto_commit()` window or a raw `BEGIN`. When a step's own unwind
cannot itself be trusted to have completed, it terminally quarantines the
connection; that is one of several causes, listed under
[Terminal connection quarantine](#terminal-connection-quarantine).

## Transaction context behavior

`suspend_auto_commit()` groups writes on one store connection. A normal exit
commits once; any exception escaping the block, including task cancellation,
causes a rollback of the outermost transaction and is re-raised.

```python
async def store_atomically(store, first, second, link):
    async with store.suspend_auto_commit():
        await store.create_thought(first)
        await store.create_thought(second)
        await store.create_edge(link)
```

Operational rules:

- Let an error escape the `async with` block. Catching it inside and then leaving
  normally tells the context manager to commit.
- `suspend_auto_commit()` holds a task-reentrant lock for its whole duration: a
  *different* task's write on the same store instance now waits for the window
  to close instead of joining its transaction (a wait that is bounded: past
  `write_lock_acquire_timeout_seconds` it raises `WriteLockTimeoutError`), and
  nesting on the *same* task is supported (only the outermost call commits or
  rolls back). Drive the
  window from one task at a time regardless — a second task's write is safe
  from corruption, but it still simply waits, so interleaving unrelated writes
  through a long-running window serialises them for no benefit. See
  [Concurrency](concurrency.md#suspend_auto_commit-is-now-a-real-exclusive-window)
  for the full contract.
- Automatic on-store derivation is skipped inside a caller-held transaction.
  After commit, invoke `derive_existing()` explicitly for sources that need it.
- The rollback branch catches `BaseException`, so `asyncio.CancelledError`
  (and `SystemExit` or `KeyboardInterrupt`) escaping the block body rolls the
  outermost transaction back like any other error. The exception to that rule is
  a cancellation that lands during the outermost window's own `commit()`: the
  commit runs after the body has returned, outside the rollback branch, so it
  may or may not have completed by the time the cancellation propagates. Treat
  the outcome as indeterminate: reconcile the affected IDs from a new store
  before repeating anything, and do not let a later commit decide it. See the
  `suspend_auto_commit()` entry in the
  [0.6 to 0.7 upgrade notes](upgrade.md#06---07) for the change.

For the supported writer topology and task boundaries, see
[Concurrency](concurrency.md).

## SQLite lock contention

SQLite permits many readers but only one writer. Stores opened through
`SqliteEngravaCore.from_config()` or `EngravaManager` set
`PRAGMA busy_timeout=5000`, so a competing connection waits for up to five
seconds before surfacing `database is locked`. A manually supplied connection
keeps its own pragma settings.

`create_thought(deduplicate=True)`, `get_or_create()`, `upsert_by_hash()`,
`bulk_store(deduplicate=True)` and the `revision`-guarded updates
(`update_thought`, `restore_thought`, `update_edge`, `update_action`) convert
contention that outlasts `busy_timeout` into `WriteContentionError`, carrying
`operation` and `attempts`; from `bulk_store` the `operation` reads
`"create_thought"`, the shared dedup path's label. The failed attempt wrote
nothing, so retrying that call outright is safe. Other write
paths can still surface the raw `OperationalError` from the table above.

Prefer removing contention over adding a generic retry loop:

1. Keep write transactions short.
2. Serialize writes per store/process.
3. Partition independent writers into separate database files with
   `EngravaManager`.
4. Increase `busy_timeout` only when the added tail latency is acceptable.

`busy_timeout` is a latency knob, not a correctness one. It decides how long a
connection waits for the file lock; it does nothing about two stores interleaving
inside one of engrava's read-modify-write operations, which is why only one store
may write a given file
([Concurrency](concurrency.md#multiple-stores-one-database-file)).

If an operation is known to be read-only and repeatable, bounded retry is
reasonable. This example deliberately retries journal verification, not an
arbitrary write:

```python
import asyncio
import sqlite3


async def verify_journal_with_lock_retry(store, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return await store.verify_journal()
        except sqlite3.OperationalError as exc:
            locked = "locked" in str(exc).lower() or "busy" in str(exc).lower()
            if not locked or attempt == attempts:
                raise
            await asyncio.sleep(0.1 * attempt)
    raise AssertionError("unreachable")
```

Do not wrap every Engrava method in this helper. After an ambiguous write error,
query the durable state from a healthy connection and choose a method whose
repetition is known to converge.

## Self-healing search and counters

Search has two intentional degradation paths:

- If the primary FTS5 `MATCH` raises `OperationalError`, Engrava increments
  `fts_match_failure_count`, logs a warning, sanitizes the original query into
  bare mode, and retries once. If that fallback also fails, the FTS arm returns
  no results; hybrid search can still use its other active arms.
- An empty, all-zero, `NaN`, or infinite query vector has no cosine direction.
  `search_similar()` increments `vector_arm_degradation_count` and returns `[]`.
  A wrong-length vector is not degraded: it raises
  `VectorDimensionMismatchError`.

The counters are cumulative per store instance and reset when a new store is
constructed. Alert on growth/deltas, not merely on a non-zero lifetime value:

```python
async def recall_with_degradation_flags(store, query):
    fts_before = store.fts_match_failure_count
    vector_before = store.vector_arm_degradation_count

    result = await store.recall(query)

    return (
        result,
        store.fts_match_failure_count > fts_before,
        store.vector_arm_degradation_count > vector_before,
    )
```

An unavailable FTS5 index returns an empty FTS arm. Failure to load the optional
`sqlite-vec` extension at store setup logs a warning and falls back to exhaustive
NumPy search; it does not increment `vector_arm_degradation_count`. Runtime
backend failures outside the two cases above are not covered by a blanket
self-healing guarantee.

See [Search](search.md#graceful-degradation) and
[Observability](observability.md#observability-signals) for the result-level and
metrics-level signals.

## Terminal connection quarantine

`ConnectionQuarantinedError` means Engrava could not prove that a rollback or
unwind it depends on completed, or that the connection's last operation ever
reported. The old connection may have an indeterminate open transaction, so the
store revokes all new operations and replaces its internal connection with a
failing proxy. This state never clears. Any of these triggers it:

- the savepoint unwind of a guarded write (`update_thought`,
  `restore_thought`, `update_edge`, `update_action`, a derived child's row
  insert or edge insert) failed;
- the savepoint unwind of a `delete_thought` failed, including a delete made
  through the TTL delete strategy or hygiene garbage collection — an ordinary
  error such as a trigger veto can cause this, not only a cancellation, and the
  derived-records seam need not be involved;
- `close()` gave up waiting on the physical close after
  `close_timeout_seconds` — see
  [Deployment](deployment.md#if-the-worker-never-answers).

The call that triggered the quarantine can surface its original error (or a
cancellation that outranks it); a `close()` whose bound expired raises
`ConnectionQuarantinedError` itself. Once quarantine is installed, the next
newly-admitted public operation raises `ConnectionQuarantinedError`.

Close the old store, open a new one, and inspect durable state before deciding
what to repeat:

```python
from engrava import ConnectionQuarantinedError, SqliteEngravaCore


async def replace_and_reconcile(store, config_path, source_id):
    try:
        source = await store.get_thought(source_id)
    except ConnectionQuarantinedError:
        await store.close()
        store = await SqliteEngravaCore.from_config(config_path)
        source = await store.get_thought(source_id)

    if source is not None:
        await store.derive_existing(source.thought_id)
    return store
```

The caller owns the returned replacement and should close it during normal
shutdown.

Do not retry against the quarantined instance, including through its journal.
Quarantine detaches the real connection and schedules its best-effort close even
when the store was built with the manual constructor. Any external reference to
that same connection must still be treated as revoked and discarded.

## Idempotency and convergence

Use these contracts narrowly:

| Operation | Repetition contract |
|---|---|
| `invalidate_thought(id, same_valid_until)` | Idempotent; converges to the same valid-time boundary |
| `invalidate_edge(id, same_valid_until)` | Idempotent; converges to the same valid-time boundary |
| `derive_existing(source_id)` | Idempotent only under the deterministic producer contract; reuses children/edges and completes missing enrichment |
| `upsert_by_hash(same_desired_record)` | State-convergent on the mutable fields, but a first-call post-commit enrichment failure still needs explicit reconciliation/derivation |
| `store_embedding(thought_id, same_vector, same_model)` with the default deterministic embedding ID | Upserts the same vector, but refreshes embedding metadata such as `created_at`; convergent, not byte-identical |
| `create_thought()` | Not idempotent; the same ID raises and a new ID creates another row |
| `remember()` | Not idempotent; each call creates a fresh ID unless content deduplication is requested |
| `create_thought(deduplicate=True)` / `get_or_create()` | Prevent duplicate content rows, but each hit increments `confirmation_count`; not observationally idempotent |
| `bulk_store()` | Atomic through source+embedding commit when it owns that transaction outright — nested inside a caller's own `suspend_auto_commit()`, a caught row error can still commit the batch's successful prefix, while a caught embedding failure (see [API reference](api-reference.md#exceptions)) commits every row the batch inserted instead, since that failure only fires after every row is already in — and, either way, not safe to replay blindly after a post-commit derivation failure |
| Retrieval calls | Do not mutate graph content, but can buffer access-frequency events when access tracking is enabled |

When an API is not listed as idempotent, assume that retry requires an
application-level operation key and a read-back/reconciliation step.

## Related documentation

- [API Reference](api-reference.md) — signatures, return types, and typed errors
- [Troubleshooting](troubleshooting.md) — symptom-to-fix entries
- [Concurrency](concurrency.md) — WAL, busy timeout, and writer topology
- [Embeddings](guides/embeddings.md) — providers, model locks, and retry settings
- [Observability](observability.md) — metrics and degradation signals
- [Backup & Recovery](backup-and-recovery.md) — WAL-safe backup and restore

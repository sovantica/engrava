# Forgetting

> **Mechanism: Memory Hygiene** — the built-in memory-hygiene loop makes no
> LLM calls. A configured `on_retrieve` or `decay_function` hook is async and
> can itself call an LLM or any other external service during a pass. See
> [Bounded, deterministic, previewable](#bounded-deterministic-previewable)
> for exactly what this loop does and does not reproduce across runs.

**Forgetting** is how an Engrava store lets cold, low-signal memories fade. It is
**opt-in** (off by default) and **reversible**: the default action **archives** a
thought — a soft-retire you can restore — and only a *separate*, independently
opted-in step ever garbage-collects an archived thought. The hygiene loop
mutates only when you call `run_hygiene()` explicitly, or when an enabled and
due hygiene policy runs as a convenience step inside `consolidate()` (see
[Running it](#running-it)); physical garbage-collection additionally requires
`auto_gc_enabled`.

## Dreaming + Forgetting — the two halves of memory maintenance

Forgetting is the **subtractive** half of a memory-maintenance cycle whose
**additive** half is [Dreaming](dreaming.md):

- **Dreaming** (consolidation) *promotes* — it keeps and strengthens what matters,
  links related memories with edges, and clusters them into higher-level
  reflections.
- **Forgetting** (memory hygiene) *demotes* — it lets memories that have gone cold
  and low-signal fade out of the active working set.

Together they mirror how biological memory maintains itself — reinforcing some
traces while letting others fade — so Forgetting can reduce the active working set
over time. The built-in scoring and default hooks make no LLM calls; a
configured `on_retrieve` or `decay_function` hook runs during the pass and may
itself call an LLM or any other external service. See
[Bounded, deterministic, previewable](#bounded-deterministic-previewable) for
what reproducibility this loop actually offers.

The whole capability is **OFF by default**: the hygiene *loop* does nothing until
you enable it — a store that never configures `hygiene_policy` never scores,
archives, or garbage-collects anything, so the loop itself changes no read or write
path. One thing is **not** gated on the policy, though: retrieval eligibility. Any
archived row — whether archived by hygiene, by the
[TTL `archive` strategy](data-lifecycle.md#archive-vs-delete), or manually — is
excluded from default retrieval (see
[Search](search.md#archived-thoughts-are-excluded-by-default)), so a store is
byte-identical to the pre-feature behaviour only when it holds no archived rows.
Following the same pattern as Dreaming (the concept "Dreaming" over the method
`consolidate()`), the public API keeps the mechanism's name: you invoke Forgetting
with `store.run_hygiene(...)` and configure it under `hygiene_policy`.

## What it does

Memory Hygiene runs **outside** the normal CRUD path. You invoke it explicitly
with `store.run_hygiene(current_cycle=N)` (or let it run at the end of a
[dreaming](dreaming.md) cycle — see [Running it](#running-it)).

```
  run_hygiene(current_cycle=N)
    │
  ┌─┴──────────────────────────────────────────────────────────┐
  │ score      each ACTIVE/CREATED thought gets a keep-score      │
  │            from the same signals dreaming uses, times a       │
  │            decay multiplier -> an eviction-score              │
  │ 1. archive thoughts below the eviction threshold (and not     │
  │    (Stage 1) protected) flip to ARCHIVED — lifecycle-reversible│
  │            via restore_thought(); archive stamps are set      │
  │ 2. gc      only when auto_gc_enabled: hygiene-archived         │
  │    (Stage 2) thoughts past the restore window are physically   │
  │            deleted (cascading edges/embeddings/actions, then  │
  │            purging the vector index)                           │
  └─┬──────────────────────────────────────────────────────────┘
    │
    ▼
  a smaller, higher-signal working set
```

The "score" step is a simplification: the candidate pool is unexpired
`ACTIVE`/`CREATED` rows, a keep-score is computed only once both run-level
gates below pass, and a candidate excluded by [protection](#protection--what-never-gets-forgotten)
or the [cold-start guards](#cold-start-safety) is never scored at all.

### The keep-score

For each candidate thought that reaches scoring — excluding one filtered out
by protection, the cold-start guards, or the run-level gates, as noted above
— hygiene computes a **keep-score** as a weighted combination of the active
scoring signals, normalised by the total active weight — the same library
[dreaming](dreaming.md#signals) uses — under hygiene's own weight vector:

| Signal | Default weight | Higher when… |
|---|---|---|
| `recency` | 0.30 | the thought was updated recently |
| `frequency` | 0.25 | the thought has been accessed often |
| `confirmation` | 0.20 | the same fact has been re-encountered |
| `confidence` | 0.15 | the thought carries a high confidence value |
| `staleness` | 0.10 | the thought has been active over a long span |

Which signals are active is decided once per run, over the whole candidate
pool, by the rules [dreaming](dreaming.md#signals) uses for its default
signals: `recency` and `staleness` are always active, and each other signal is
active only when some candidate has its data (`frequency` also needs access
tracking on). An inactive signal's weight is set to `0.0`, and each active
signal's weight is divided by the sum of the active weights. If no signal is
active, or that sum is zero, the keep-score selects nothing for archiving
that run. This keeps the
keep-score meaningful on sparse stores instead of dragging every score
toward a constant.

The keep-score is then multiplied by the
[`decay_function` hook](extension-hooks.md) to produce the **eviction-score**:

```
keep_score     = Σ active-signal weight · signal(thought)     (renormalised)
eviction_score = keep_score · decay_function(thought, elapsed_cycles)
archive(thought) ⇐ eviction_score < eviction_threshold  AND  not protected(thought)
```

The default `decay_function` returns `1.0` (no decay), so out of the box the
eviction-score *is* the keep-score. A custom hook can shape a decay curve; its
return is clamped into `[0.0, 1.0]`, and a non-finite value is treated as `1.0`,
so decay multiplies the keep-score by a factor between `0.0` and `1.0`. Because
a custom hook can deliberately
make an otherwise high-scoring thought archivable, treat it as part of the
retention policy and preview its effect with `dry_run` before enabling mutations.

The Memory Hygiene loop is the **only** place `decay_function` is consulted; it
is not part of search, ranking, or promotion.

## Protection — what never gets forgotten

A thought is **protected** (never auto-archived and never auto-GC'd) when:

- it is **pinned** (`ThoughtRecord.pinned = True`) — the durable, node-level
  never-forget marker; or
- its **priority** is listed in `protected_priorities` (default: `("P1",)`).

```python
from engrava import ThoughtRecord, ThoughtType, Priority, LifecycleStatus

keep_me = ThoughtRecord(
    thought_id="user-birthday",
    thought_type=ThoughtType.OBSERVATION,
    essence="User's birthday is 3 March",
    content="The user mentioned their birthday is 3 March.",
    priority=Priority.P2,
    lifecycle_status=LifecycleStatus.ACTIVE,
    created_cycle=0,
    updated_cycle=0,
    source="user",
    pinned=True,   # never auto-archived or auto-GC'd, regardless of score
)
```

`confidence` is **not** protection. A high-confidence but cold thought can still
be archived — a model-confidence estimate is not a user keep-decision.
`confidence` only *contributes* to the keep-score via its signal.

`protected_priorities` is a **default, not an invariant**: an operator who wants
more aggressive hygiene can set it to `()` so even top-priority thoughts are
eligible. Pinning is the invariant.

## Two stages: archive, then (optionally) GC

Stage 1 (archive) is the **default action** and is reversible for lifecycle
and the hygiene archive markers:

- A below-threshold, unprotected thought flips to `ARCHIVED` and its
  `archived_at_cycle` is set to the current cycle while `archived_at` receives
  the run's wall-clock instant. Its `expires_at` is cleared so it is no longer
  subject to [TTL](data-lifecycle.md). This dedicated path accepts eligible
  `ACTIVE` and `CREATED` rows; it does not require an ordinary lifecycle journey
  through `DONE`.
- **Restore** un-archives a thought: `store.restore_thought(thought_id)` transitions
  it back to `ACTIVE` (the `ARCHIVED → ACTIVE` lifecycle edge) and clears
  both `archived_at_cycle` and `archived_at`. It does not restore `expires_at`:
  an expiry the thought carried before archival stays cleared after restore.
  Restoring a thought that is not archived raises `InvalidTransitionError`.

Stage 2 (garbage collection) runs **only** when `auto_gc_enabled` is set (it is
**`false` by default**) — enabling hygiene never implicitly enables deletion:

- A thought is GC-eligible only when it was archived **by hygiene**
  (`archived_at_cycle` is set) *and* it has cleared **both enabled** restore
  windows:
  - a **cycle window** — `current_cycle - archived_at_cycle >=
    gc_min_archive_age_cycles` (default `gc_min_archive_age_cycles = 10` cycles).
    Setting `gc_min_archive_age_cycles: 0` makes this window always pass — the
    cycle gate is disabled, symmetric with the wall-clock case below; **and**
  - a **wall-clock window** — the thought has been archived for at least
    `gc_restore_window_seconds` of real time (`archived_at <= now -
    gc_restore_window_seconds`, default `2592000` = 30 days). This exists so a
    fast-cycling or bulk store cannot burn through the cycle window and delete a
    just-archived thought before there was any real-time chance to
    `restore_thought` it. Set `gc_restore_window_seconds: 0` to disable the
    wall-clock window (cycle-only, the pre-window behaviour).

  With both windows disabled (`0`), a hygiene-archived thought has no restore
  window before it becomes GC-eligible — opt into that only deliberately.
- GC excludes any row whose `archived_at_cycle` is `None` — ordinarily true of
  a thought archived through [TTL](data-lifecycle.md) or a manual lifecycle
  change. That marker is not proof of *this* archival's provenance, though: a
  raw `update_thought(lifecycle_status=ARCHIVED)` (bypassing `run_hygiene` and
  TTL cleanup, both of which refresh or clear it) leaves a prior hygiene
  archival's `archived_at_cycle` in place, so a thought re-archived that way
  can still be picked up by a later GC pass.
- A hygiene-archived row that predates the wall-clock `archived_at` column
  (so `archived_at` is `None`) is **never** GC'd while the wall-clock window is
  active — the irreversible stage **fails closed** rather than guess an age.
- Deletion runs an orphan-reflection sweep first (so no
  [REFLECTION](dreaming.md) is left summarising a cluster the delete would empty),
  then cascades to edges/embeddings/actions, then purges the vector index.

**Below core schema 12 there is no such cascade.** The `ON DELETE CASCADE` on
`edge`, `embedding` and `action` arrives with the core-12 migration.
`delete_thought` does not rely on it: it issues its own deletes for the thought's
rows in those three tables, in the same savepoint as the parent delete, which runs
first. See
[Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated)
for what `engrava migrate` cleans up on a database that already holds dangling
`embedding` rows.

> **GC is not erasure.** Garbage collection reclaims the live, queryable working
> set — it does **not** purge history. When the [hash-chain journal](audit-trail.md)
> is enabled, a GC delete is recorded as a `DELETE_THOUGHT` entry that keeps a full
> `before` snapshot, so the content survives in the append-only journal after the
> live row is gone.

> Garbage collection here is **cognitive hygiene, not compliance deletion**. It
> is best-effort, window-gated, and opt-in — it offers no deletion guarantee, legal
> hold, scheduled/enforced retention, or erasure receipt. For the honest deletion
> mechanics (and the residue a hard delete leaves in the audit journal and
> backups), see [Data lifecycle, retention & deletion](data-lifecycle.md).

## Bounded, deterministic, previewable

- **Bounded, for the archive and GC selections.** `max_evictions_per_run`
  (default `100`) caps **each stage** independently — at most that many
  archived, and at most that many GC'd, per run. The GC stage also runs an
  orphan-reflection sweep beforehand, so a synthesis never outlives its whole
  source cluster; that sweep is **not** capped by `max_evictions_per_run` and
  retires every qualifying orphan reflection it finds, counted separately
  from `archived_count`.
- **Deterministic for fixed inputs, for the windows `now` governs.** The
  injected (or, if omitted, wall-clock) `now` fixes the inactivity-age and
  restore-window boundaries for the run, and the same store + config + cycle +
  `now` selects the same set on those boundaries alone, when any configured
  custom hooks are deterministic too. Candidate collection is a separate
  matter: it pages through `list_thoughts()`, whose own default expiry filter
  reads real wall-clock time on every call regardless of the `now` passed to
  `run_hygiene` — so a thought's expiry crossing during a long-running pass can
  still change which candidates are collected, independent of `now`. When
  more candidates qualify than the cap allows, the archive stage keeps the
  lowest-scoring, then oldest, then lowest-id thoughts; the GC stage keeps the
  oldest-archived, then lowest-id thoughts.
- **Fail-safe on a blank slate.** A brand-new store archives nothing, but not
  because "no signal is active" — with the default weights, `recency` and
  `staleness` are active on presence alone (see above), so that condition is
  essentially unreachable. The actual guard is a separate **usage-signal gate**:
  without any usage-history signal (access counts, confirmations) anywhere in
  the candidate pool, cycle-recency alone cannot distinguish "cold" from
  "ingested early", so the pass archives nothing until at least one usage
  signal has data to work with.
- **Dry run.** With `dry_run: true`, `run_hygiene` computes and returns the set it
  *would* archive (with a per-thought reason). **Hygiene itself issues no
  archive or GC mutation and writes no journal entry for this run** — that is
  the whole guarantee. It still runs candidate collection and scoring to
  compute that set, so a configured `on_retrieve` or `decay_function` hook
  still executes; a hook that holds its own reference to the store can still
  write to it, or produce any other side effect, regardless of `dry_run` —
  the preview guarantee is about hygiene's own actions, not a hook's.

```python
result = await store.run_hygiene(current_cycle=1000)
print(result.archived_count, result.gc_count, result.candidates_evaluated)

# Preview mode (hygiene_policy.dry_run = true):
preview = await store.run_hygiene(current_cycle=1000)
for reason in preview.would_evict:
    print(reason.thought_id, reason.eviction_score, reason.signals)
```

## Cold-start safety

Two guards stop a fresh or bulk-imported store — where cycle-recency has no
history to work with and degenerates into ingest order — from archiving its
earliest-loaded rows. Both only ever *add* protection; neither can cause an
archival the keep-score alone would not.

- **Minimum inactivity age.** A thought is eligible for archival only once
  `now` is at least `min_inactivity_age_seconds` past its last contact
  (`last_accessed_at`, else `updated_at`, else `created_at`) — the row's own
  persisted timestamp, not when the store or the row was created. Default
  `604800` (7 days). A freshly created thought is protected until that
  persisted timestamp falls inside the window. A row restored from a snapshot
  or otherwise imported with the timestamps it already carried is protected
  only if one of those persisted timestamps is recent enough — an imported
  row carrying an old `created_at` (for example, a restored backup) is
  archival-eligible immediately, not protected by virtue of having just been
  imported. `min_inactivity_age_seconds: 0` disables the gate (the pre-gate
  behaviour); a row with no known last-contact time fails closed (protected).
- **Usage-signal access gate.** A run archives **nothing** unless at least one
  *usage-history* signal — `frequency` (reads), `confirmation` (reinforcements),
  or `action_outcome` — is active across the candidate pool. Without any evidence
  a thought was ever used, "cold" cannot be told apart from "merely ingested
  early", so cycle-recency must not drive eviction on its own. In practice this
  means Forgetting only has an effect once the store carries genuine usage data:
  enable [access tracking](concepts.md#access-tracking-and-usage-telemetry) (on by
  default when dreaming is enabled) or record confirmations. Implicit accesses
  are buffered; `consolidate()` flushes them before scoring, while a direct
  `run_hygiene()` does not. Call `flush_access_buffer()` first when a standalone
  hygiene pass must include the newest pending reads.

Together with the [all-flat fail-safe](#bounded-deterministic-previewable) above,
these make the first hygiene runs on a new store a safe no-op rather than a guess.

## Audit trail

Every archival and every garbage-collection is recorded in the
[hash-chain journal](audit-trail.md) when journaling is enabled, using the
existing mutation kinds (an archive is an `UPDATE_THOUGHT`; a GC-delete is a
`DELETE_THOUGHT` — **no new mutation type is introduced**). The forgetting
rationale rides in the entry's `delta` under a nested `eviction_reason`:

```json
{
  "mechanism": "hygiene",
  "keep_score": 0.03,
  "eviction_score": 0.03,
  "decay_multiplier": 1.0,
  "threshold": 0.20,
  "signals": { "recency": 0.02, "staleness": 0.0 }
}
```

Each recorded decision carries its own score, multiplier, threshold, and
per-signal values, and the chain still verifies with `verify_journal()` after
an archive and a GC — retained entries and the linkage between them can be
checked for tampering. It does not, on its own, prove the *set* of decisions
is complete: the recorded reason does not include the full candidate pool,
the effective weights, or the eviction-cap ordering for the run, and, as
[Audit journal threat model](security.md#audit-journal-threat-model) explains,
verification cannot detect the deletion of a self-consistent suffix of the
chain — so a local check cannot by itself prove that no later decision was
removed. A `dry_run` preview journals nothing, since nothing was mutated.

## Running it

**Directly** — bypasses the cadence and runs a pass immediately, provided a
hygiene policy is configured and enabled, and a cognitive cycle is
available: calling `run_hygiene()` with no `hygiene_policy` configured
raises `RuntimeError`; a configured but disabled policy (`enabled: false`)
returns an empty result without scoring anything, even on this explicit
call; and with neither an explicit `current_cycle` argument nor a
configured `cycle_provider`, it raises `ValueError` instead of inventing a
cycle. Pass a timezone-aware `now` when a replay or benchmark needs fixed
wall-clock boundaries:

```python
import asyncio
from datetime import UTC, datetime

from engrava import SqliteEngravaCore


async def run_a_hygiene_pass() -> None:
    async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
        result = await store.run_hygiene(
            current_cycle=1000,
            now=datetime(2026, 7, 23, tzinfo=UTC),
        )
        print(result.archived_count, result.gc_count)


asyncio.run(run_a_hygiene_pass())
```

**As a convenience at the end of a dreaming cycle** — when both `dreaming` and
`hygiene_policy` are enabled, `store.consolidate(current_cycle=N)` runs one
hygiene pass after promotion and the orphan sweep, but only when the cycle
satisfies the cadence (`current_cycle % check_every_n_cycles == 0`). An explicit
`run_hygiene` call always bypasses the cadence.

## Configuration

See [Configuration → `hygiene_policy`](configuration.md#hygiene_policy)
for the full YAML surface. A minimal enable:

```yaml
hygiene_policy:
  enabled: true                       # OFF by default
  eviction_threshold: 0.20            # archive below this eviction-score
  auto_gc_enabled: false              # keep GC off until you want physical deletion
  min_inactivity_age_seconds: 604800  # 7 days untouched before archivable; 0 disables
  gc_restore_window_seconds: 2592000  # 30-day real-time restore window before GC; 0 disables
  dry_run: true                       # preview first
```

## Related

- [Dreaming — memory consolidation](dreaming.md) — the additive counterpart in the
  "Dreaming + Forgetting" pair.
- [Data lifecycle, retention & deletion](data-lifecycle.md) — lifecycle states,
  TTL, and honest hard-deletion mechanics.
- [Audit trail](audit-trail.md) — the hash-chain journal that records evictions.
- [Observability → Observability signals](observability.md#observability-signals) —
  read-only counters for search-arm health (separate from the hygiene loop).
- [Extension hooks](extension-hooks.md) — the `decay_function` hook that Forgetting
  activates (its only call-site).

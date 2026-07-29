# Forgetting

> **Mechanism: Memory Hygiene** — a no-LLM memory-hygiene loop that is
> deterministic for a fixed store, configuration, cycle, and `now` when any
> configured custom hooks are deterministic too.

**Forgetting** is how an Engrava store lets cold, low-signal memories fade. It is
**opt-in** (off by default) and **reversible**: the default action **archives** a
thought — a soft-retire you can restore — and only a *separate*, independently
opted-in step ever garbage-collects an archived thought. The hygiene loop never
archives or removes anything implicitly.

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
over time. Both involve no LLM; Forgetting is deterministic when its cycle and
wall-clock reference are held fixed and any configured custom hooks are
deterministic too.

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
  │    (Stage 1) protected) flip to ARCHIVED — reversible, no data │
  │            loss; cycle and wall-clock archive stamps are set  │
  │ 2. gc      only when auto_gc_enabled: hygiene-archived         │
  │    (Stage 2) thoughts past the restore window are physically   │
  │            deleted (cascading edges/embeddings/actions, then  │
  │            purging the vector index)                           │
  └─┬──────────────────────────────────────────────────────────┘
    │
    ▼
  a smaller, higher-signal working set
```

### The keep-score

For each candidate thought, hygiene computes a **keep-score** as a weighted
average of the reusable scoring signals — the same library
[dreaming](dreaming.md#signals) uses — under hygiene's own weight vector:

| Signal | Default weight | Higher when… |
|---|---|---|
| `recency` | 0.30 | the thought was updated recently |
| `frequency` | 0.25 | the thought has been accessed often |
| `confirmation` | 0.20 | the same fact has been re-encountered |
| `confidence` | 0.15 | the thought carries a high confidence value |
| `staleness` | 0.10 | the thought has been active over a long span |

A signal that is **flat** across the whole candidate pool (for example
`frequency` on a store with no access history, or `confirmation` with no
deduplication) carries no ranking information, so it is dropped and its weight is
renormalised over the remaining active signals. This keeps the keep-score
meaningful on sparse stores instead of dragging every score toward a constant.

The keep-score is then multiplied by the
[`decay_function` hook](extension-hooks.md) to produce the **eviction-score**:

```
keep_score     = Σ active-signal weight · signal(thought)     (renormalised)
eviction_score = keep_score · decay_function(thought, elapsed_cycles)
archive(thought) ⇐ eviction_score < eviction_threshold  AND  not protected(thought)
```

The default `decay_function` returns `1.0` (no decay), so out of the box the
eviction-score *is* the keep-score. A custom hook can shape a decay curve; its
return is clamped into `[0.0, 1.0]`, and a non-finite value is treated as `1.0`.
Decay can therefore only ever *lower* a score toward archival; it cannot raise a
low keep-score back above the threshold. Because a custom hook can deliberately
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

Stage 1 (archive) is the **default action** and is fully reversible:

- A below-threshold, unprotected thought flips to `ARCHIVED` and its
  `archived_at_cycle` is set to the current cycle while `archived_at` receives
  the run's wall-clock instant. Its `expires_at` is cleared so it is no longer
  subject to [TTL](data-lifecycle.md). This dedicated path accepts eligible
  `ACTIVE` and `CREATED` rows; it does not require an ordinary lifecycle journey
  through `DONE`.
- **Restore** un-archives a thought: `store.restore_thought(thought_id)` transitions
  it back to `ACTIVE` (the `ARCHIVED → ACTIVE` lifecycle edge) and clears
  both `archived_at_cycle` and `archived_at`. No data was lost. Restoring a
  thought that is not archived raises `InvalidTransitionError`.

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
- A thought archived by any **other** path — [TTL](data-lifecycle.md) or a manual
  lifecycle change, where `archived_at_cycle` is `None` — is **never** auto-GC'd
  by hygiene. Hygiene only reaps what hygiene archived.
- A hygiene-archived row that predates the wall-clock `archived_at` column
  (so `archived_at` is `None`) is **never** GC'd while the wall-clock window is
  active — the irreversible stage **fails closed** rather than guess an age.
- Deletion runs an orphan-reflection sweep first (so no
  [REFLECTION](dreaming.md) is left summarising a cluster the delete would empty),
  then cascades to edges/embeddings/actions, then purges the vector index.

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

- **Bounded.** `max_evictions_per_run` (default `100`) caps **each stage**
  independently — at most that many archived, and at most that many GC'd, per run.
- **Deterministic for fixed inputs.** The same store + config + cycle + injected
  `now` selects the same set when any configured custom hooks are deterministic
  too. If `now` is omitted, `run_hygiene` reads UTC wall time once, so
  eligibility can change as the inactivity and restore windows advance. When
  more candidates qualify than the cap allows, the archive stage keeps the
  lowest-scoring, then oldest, then lowest-id thoughts; the GC stage keeps the
  oldest-archived, then lowest-id thoughts.
- **Fail-safe on a blank slate.** When *no* signal is active (a brand-new store
  with no access history, no confirmations, and a uniform cycle), the keep-score
  is uninformative, so the pass archives **nothing** rather than guessing.
- **Dry run.** With `dry_run: true`, `run_hygiene` computes and returns the set it
  *would* archive (with a per-thought reason) **without mutating anything and
  without journaling** — a safe preview before enabling for real.

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

- **Minimum inactivity age.** A thought is eligible for archival only after it has
  been untouched for at least `min_inactivity_age_seconds` of wall-clock time —
  measured from its last contact (`last_accessed_at`, else `updated_at`, else
  `created_at`). Default `604800` (7 days). A store younger than that archives
  nothing; a freshly created or just-imported thought is protected exactly like a
  pinned one until it has actually aged. `min_inactivity_age_seconds: 0` disables
  the gate (the pre-gate behaviour); a row with no known last-contact time fails
  closed (protected).
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

This makes every forgetting decision reconstructable and tamper-evident, and the
chain still verifies with `verify_journal()` after an archive and a GC. A
`dry_run` preview journals nothing, since nothing was mutated.

## Running it

**Directly** — runs immediately, ignoring the cadence. Pass a timezone-aware
`now` when a replay or benchmark needs fixed wall-clock boundaries:

```python
from datetime import UTC, datetime

async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    result = await store.run_hygiene(
        current_cycle=1000,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
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

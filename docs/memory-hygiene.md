# Memory Hygiene — Deterministic Forgetting

Engrava's **Memory Hygiene** loop is the subtractive counterpart to
[dreaming consolidation](dreaming.md): a deterministic, no-LLM pass that
**archives** cold, low-value thoughts and — as a separate, independently opted-in
step — **garbage-collects** them after a restore window. Where dreaming *promotes*
the memories worth keeping, hygiene *demotes* the ones that have gone cold, so a
long-running store does not accumulate unbounded low-signal thoughts.

The whole capability is **opt-in and OFF by default**. A store that never
configures `hygiene_policy` behaves exactly as it did before this feature existed
on every read and write path — nothing is ever archived or deleted implicitly.

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
  │            loss; their archived_at_cycle is stamped           │
  │ 2. gc      only when auto_gc_enabled: hygiene-archived         │
  │    (Stage 2) thoughts past the restore window are physically   │
  │            deleted (cascading edges/embeddings/vectors)       │
  └─┬──────────────────────────────────────────────────────────┘
    │
    ▼
  a smaller, higher-signal working set
```

### The keep-score

For each candidate thought, hygiene computes a **keep-score** as a weighted
average of the reusable scoring signals — the same library
[dreaming](dreaming.md#the-signals) uses — under hygiene's own weight vector:

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
Decay can therefore only ever *lower* a score toward archival — a misbehaving
hook can never cause a thought to be archived that the keep-score alone would
have kept.

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
  `archived_at_cycle` is set to the current cycle. Its `expires_at` is cleared so
  it is no longer subject to [TTL](data-lifecycle.md).
- **Restore** un-archives a thought: `store.restore_thought(thought_id)` transitions
  it back to `ACTIVE` (the `ARCHIVED → ACTIVE` lifecycle edge) and clears
  `archived_at_cycle` to `None`. No data was lost. Restoring a thought that is not
  archived raises `InvalidTransitionError`.

Stage 2 (garbage collection) runs **only** when `auto_gc_enabled` is set —
enabling hygiene never implicitly enables deletion:

- A thought is GC-eligible only when it was archived **by hygiene**
  (`archived_at_cycle` is set) *and* its restore window has elapsed
  (`current_cycle - archived_at_cycle >= gc_min_archive_age_cycles`).
- A thought archived by any **other** path — [TTL](data-lifecycle.md) or a manual
  lifecycle change, where `archived_at_cycle` is `None` — is **never** auto-GC'd
  by hygiene. Hygiene only reaps what hygiene archived.
- Deletion runs an orphan-reflection sweep first (so no
  [REFLECTION](dreaming.md) is left summarising a cluster the delete would empty),
  then cascades to edges/embeddings/actions, then purges the vector index.

> Garbage collection here is **cognitive hygiene, not compliance deletion**. It
> is best-effort, cycle-based, and opt-in — it offers no deletion guarantee, legal
> hold, scheduled/enforced retention, or erasure receipt. For the honest deletion
> mechanics (and the residue a hard delete leaves in the audit journal and
> backups), see [Data lifecycle, retention & deletion](data-lifecycle.md).

## Bounded, deterministic, previewable

- **Bounded.** `max_evictions_per_run` (default `100`) caps **each stage**
  independently — at most that many archived, and at most that many GC'd, per run.
- **Deterministic.** The same store + config + cycle always selects the same set.
  When more candidates qualify than the cap allows, the archive stage keeps the
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

**Directly** — runs immediately, ignoring the cadence:

```python
async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    result = await store.run_hygiene(current_cycle=1000)
```

**As a convenience at the end of a dreaming cycle** — when both `dreaming` and
`hygiene_policy` are enabled, `store.consolidate(current_cycle=N)` runs one
hygiene pass after promotion and the orphan sweep, but only when the cycle
satisfies the cadence (`current_cycle % check_every_n_cycles == 0`). An explicit
`run_hygiene` call always bypasses the cadence.

## Configuration

See [Configuration → `hygiene_policy`](configuration.md#memory-hygiene-hygiene_policy)
for the full YAML surface. A minimal enable:

```yaml
hygiene_policy:
  enabled: true            # OFF by default
  eviction_threshold: 0.20 # archive below this eviction-score
  auto_gc_enabled: false   # keep GC off until you want physical deletion
  dry_run: true            # preview first
```

## Related

- [Dreaming — memory consolidation](dreaming.md) — the additive counterpart.
- [Data lifecycle, retention & deletion](data-lifecycle.md) — lifecycle states,
  TTL, and honest hard-deletion mechanics.
- [Audit trail](audit-trail.md) — the hash-chain journal that records evictions.
- [Extension hooks](extension-hooks.md) — the `decay_function` hook hygiene
  activates.

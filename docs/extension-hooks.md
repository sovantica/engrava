# Extension hooks

Engrava exposes a **functional hook interface** that lets external code observe
and transform data flowing through the core pipeline:

| Interface | File | Semantic |
|-----------|------|----------|
| `EngravaHooksProtocol` | `domain/protocols/hooks.py` | **Transformation** — methods return modified data (custom scoring, MindQL extensions) |
| `DerivedRecordProducerProtocol` | `domain/protocols/derived_records.py` | **Derivation** — return N derived records from one stored thought; core persists each as an ordinary thought (see §1A) |

For higher-level extension patterns (embedding providers, custom MindQL commands)
see [extensions.md](extensions.md).

---

## 1. EngravaHooksProtocol

### 1.1 Contract

- All four data-flow methods are `async` and return a value.
- Hooks **must not raise** — unexpected exceptions will propagate to the caller.
- Hooks **must not have side effects** that modify shared state; return an
  enriched copy instead.
- Engrava is `frozen=True`-first — if you need to mutate a `ThoughtRecord`,
  return `thought.model_copy(update={...})`.

### 1.2 Available hooks

`on_store`, `on_retrieve`, and `decay_function` are invoked by the public engrava
package today. The remaining two are part of the protocol contract but are
**reserved** — core engrava does not call them (they exist for downstream
consumers and future use). Implement them if you want protocol conformance,
but do not expect core to invoke them.

| Method | When | Returns | Status in core |
|--------|------|---------|----------------|
| `on_store` | After a thought row is inserted — **not** necessarily durable yet. On a plain `create_thought`, the row has already committed by this point. Inside `bulk_store`, every item's `on_store` fires as the batch is built, *before* the batch's single commit — a row error later in the same batch rolls the transaction back, and `on_store` has already run for the rows that never persist | `ThoughtRecord` (enriched or unchanged) | **active** |
| `on_retrieve` | After a thought is loaded from storage — but only via `get_thought` and `list_thoughts`. `update_thought`, `restore_thought`, `invalidate_thought`, and the read-back inside `get_or_create` / `upsert_by_hash` all read a row back without calling it | `ThoughtRecord` (enriched or unchanged) | **active** |
| `decay_function(thought, elapsed_cycles)` | Per-candidate decay factor, multiplied into the hygiene eviction-score | `float` in `[0.0, 1.0]` | **active** — called for each candidate when an enabled `run_hygiene()` pass reaches archive scoring; it is never consulted in search, ranking, or promotion |
| `score_function(thought, context)` | Custom relevance score | `float` | reserved — not called by core |
| `mindql_extension_registry()` | Register custom MindQL verbs | `dict[str, MindQLExtension]` | reserved — the store itself never reads `ExtensionManifest.mindql_extensions` either; the only consumer of that field is the `engrava` CLI, which discovers verbs through the `engrava.extensions` entry-point group, not through a manifest passed to the constructor |

---

## 1A. Derived-records extension seam

Sometimes an extension needs to turn **one** stored thought into **several**
records — split a document into sections, distil an observation into atomic
facts, extract structured items. `on_store` cannot express this: it is
one-in / one-out. The derived-records seam fills that gap through a **separate,
optional capability protocol** — `EngravaHooksProtocol` is unchanged, so an
existing hooks class keeps working unchanged (byte-identical persisted results).

### 1A.1 Contract

Implement `derive_records` on your hooks object. Core detects the capability
with `isinstance(hooks, DerivedRecordProducerProtocol)` (it is
`@runtime_checkable`); if the method is absent, the seam is simply absent.

```python
async def derive_records(
    self, thought: ThoughtRecord, ctx: DeriveContext
) -> Sequence[DerivedRecord]: ...
```

- Called **only after the source thought is durable**, and only when the seam
  is enabled (`DeriveGates.enabled`). If `on_store` raises, derivation never
  runs.
- The producer describes **what** to derive; core owns **how** it is persisted.
  A `DerivedRecord` carries only producer-owned fields — a non-empty `content`,
  `thought_type`, `priority`, a `metadata` payload, and the
  `attach_provenance_edge` flag. Identity, the `essence`, timestamps, cycle, and
  lifecycle status are assigned by core (the `essence` is derived from
  `content`) and are **not representable** on the type, so there is nothing for a
  producer to forge.
- `DeriveContext` exposes only stable facts about the source
  (`source_thought_id`, `source_content_hash`, `cycle_at_derivation`) plus an
  informational `origin`. It exposes **no store handle**: a producer must not
  persist, query, or mutate anything itself, and must not spawn background
  tasks. Persistence is entirely core-controlled.
- For exact idempotency, make the output a deterministic function of the source.

### 1A.2 Persistence model

Derivation is **source-first, per-child, deferred, and non-atomic**. For each
returned record, in producer order, core runs the same lifecycle an ordinary
thought gets: insert → commit → auto-embed → (if requested) attach the single
`DERIVED_FROM` provenance edge (derived → source). A child's **row** commits as
its own durable unit; its enrichment (embedding, edge) completes afterward, so a
child can be durably present yet not-yet-enriched — a recoverable partial state,
not atomic enrichment. A crash mid-family leaves a **partial but regenerable**
result, recovered by re-running derivation (deterministic content-hash ids make a
re-run idempotent: existing children/edges are reused, missing ones filled).

The `DERIVED_FROM` edge records **content-level** provenance ("this content was
produced by deriving from the source"), so a child that reuses an existing row
is linked, not duplicated.

**When derivation fires.** Only on a **durably auto-committed** create: a normal
`create_thought` (dispatched inline right after its own commit) or a `bulk_store`
insert (dispatched after the batch commits, per genuinely-new record — dedup /
hash hits are excluded). A create issued **inside a caller-held
`suspend_auto_commit()` window does not auto-derive** — the caller owns that open
transaction and the source is not yet durable; trigger derivation with an
explicit re-run / backfill once your transaction has committed. This is
recoverability by explicit backfill, not automatic recovery.

### 1A.3 Gates

Configure the seam with `DeriveGates` (or the `derive:` YAML section):

| Gate | Default | Meaning |
|------|---------|---------|
| `enabled` | `False` | Master switch. When off, the persisted results (DB + journal) are byte-identical to a store without the seam. |
| `on_error` | `"log"` | `"log"` records a failure with ordinary logging and continues with the remaining children; `"raise"` re-raises after the source is durable, aborting the rest. |
| `max_derived_per_source` | `32` | Core reads at most this many + 1 items and rejects an over-cap (or lazy/unbounded) return before any child is written. |

Durability is decoupled from derivation: the source is **always** durable even
when a producer or a child fails. With `on_error="raise"` the caller may see an
error *while the source persists* (durability ≠ API success). `CancelledError`
always propagates, regardless of `on_error`.

### 1A.4 Example — a deterministic, dependency-free producer

`StructuralSplitProducer` (shipped in `engrava.extensions.structural_split`) is
a complete reference consumer: it splits a thought's content into paragraphs and
derives one linked child per paragraph, running purely on the stored text — no
model, no network, no external service.

```python
import aiosqlite
from engrava import DeriveGates, SqliteEngravaCore, StructuralSplitProducer

conn = await aiosqlite.connect("engrava.db")
conn.row_factory = aiosqlite.Row
store = SqliteEngravaCore(
    conn,
    hooks=StructuralSplitProducer(),
    derive_gates=DeriveGates(enabled=True),
)
await store.ensure_schema()
# Storing a multi-paragraph thought now also persists one derived child per
# paragraph, each carrying a DERIVED_FROM edge back to the source.
```

To write your own, subclass `DefaultEngravaHooks` and add `derive_records`:

```python
from collections.abc import Sequence

from engrava import DerivedRecord, DeriveContext, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.domain.protocols.hooks import DefaultEngravaHooks


class SentenceSplitter(DefaultEngravaHooks):
    async def derive_records(
        self, thought: ThoughtRecord, ctx: DeriveContext
    ) -> Sequence[DerivedRecord]:
        sentences = [s.strip() for s in thought.content.split(".") if s.strip()]
        return [
            DerivedRecord(
                content=s,
                thought_type=ThoughtType.OBSERVATION,
                priority=Priority.P3,
            )
            for s in sentences
        ]
```

`DerivedRecordProducerProtocol`, `DerivedRecord`, `DeriveContext`, and
`DeriveGates` are public API under the `X.Y.x` stability guarantee (no breaking
change within a patch series; breaking changes ship in a minor after a
deprecation window).

### 1A.5 Split modes (`StructuralSplitProducer`)

`StructuralSplitProducer` ships two deterministic, dependency-free split modes,
selected with `split_mode` (a `SplitMode` value):

| Mode | What it does |
|---|---|
| `SplitMode.PARAGRAPH` (default) | Splits on a blank-line (paragraph) boundary — the byte-identical original behaviour. |
| `SplitMode.FIXED_WINDOW` | Tiles the content into fixed-size windows, bounding chunk size for embedding robustness on long content with no dependence on natural boundaries. |

The complete constructor surface is:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `thought_type` | `ThoughtType` | `OBSERVATION` | Classification assigned to every derived child |
| `priority` | `Priority` | `P3` | Priority assigned to every derived child |
| `split_mode` | `SplitMode` | `PARAGRAPH` | Paragraph or fixed-window segmentation |
| `window_size` | `int` | `1000` | Fixed-window length in `window_unit`; must be `>= 1` |
| `window_unit` | `"char" \| "word"` | `"char"` | Count windows and overlap in characters or whitespace-delimited words |
| `window_overlap` | `int` | `0` | Units shared by consecutive windows; must satisfy `0 <= overlap < window_size` |
| `min_chars` | `int` | `0` | Minimum stripped source length required before splitting; must be `>= 0` |
| `boundary` | `re.Pattern[str]` | blank-line pattern | Custom paragraph boundary; ignored in fixed-window mode |
| `attach_edges` | `bool` | `True` | Attach one `DERIVED_FROM` edge from each child to the source |

Windows advance by `window_size - window_overlap` and fully cover the content (the
final window may be shorter). Every derived child records its `split_mode`,
`segment_index`, and source `char_start` / `char_end` in its `metadata`.
Regardless of mode, fewer than two resulting segments produces no children; a
single segment is not a structural split. `min_chars` is checked before
segmentation, while `attach_edges=False` changes provenance attachment only and
does not change child content or identity.

```python
from engrava import SplitMode, StructuralSplitProducer

# 200-word windows with a 20-word overlap.
producer = StructuralSplitProducer(
    split_mode=SplitMode.FIXED_WINDOW,
    window_size=200,
    window_unit="word",
    window_overlap=20,
)
```

Only `SplitMode.PARAGRAPH` and `SplitMode.FIXED_WINDOW` exist — a model-tokenizer
window is deliberately excluded, since it would couple the producer to an
embedding model.

### 1A.6 Backfilling an existing store (`derive_existing`)

`derive_records` fires automatically on a durable create. To run a producer over
thoughts that are **already stored** (for example after adding a producer to an
existing store), call `derive_existing`:

```python
result = await store.derive_existing(thought_id)
print(result.thought_id, result.created, result.reused, result.skipped)
```

| `DeriveResult` field | Meaning |
|---|---|
| `thought_id` | Source thought the backfill targeted |
| `created` | Children inserted by this run |
| `reused` | Content-addressed children that already existed |
| `skipped` | Child failures suppressed under `on_error="log"` |

- Returns a `DeriveResult` tallying children `created` / `reused` / `skipped`.
  Because derived-child identity is content-addressed, re-running is
  **idempotent** — already-present children are `reused`, not duplicated (a
  fully-derived source yields `created == 0`).
- Gated on a producer capability being present, honouring `DeriveGates.on_error`
  and `max_derived_per_source` — but **independent of `DeriveGates.enabled`**
  (that master switch governs only the automatic on-store trigger), so you can
  backfill once without committing to automatic derivation on every future write.
  With no producer registered it is a clean no-op.
- Raises `SourceThoughtNotFoundError` when `thought_id` does not exist (a
  precondition failure, distinct from the clean empty result returned for an
  ineligible — already-derived — source). Raises `DerivedRecordError` if the
  producer's return violates the seam's deterministic contract (over cap, or an
  identity collision) under `on_error="raise"`.
- A source that is itself a derived record (it carries an outgoing `DERIVED_FROM`
  edge) is never re-derived.
- Unlike the automatic on-store trigger, `derive_existing` does not defer inside
  a caller-held `suspend_auto_commit()` window (or a raw `BEGIN`): the source is
  already durable, so the children simply join that transaction. A failed child
  undoes only its own failing step (its row, its edge, or an embedding attempt)
  — never an earlier child's work, and never the caller's other pending writes
  in the same transaction, under either `on_error` policy. Who decides *when*
  that becomes durable differs by transaction kind: a `suspend_auto_commit()`
  window suppresses every write's own auto-commit, so nothing commits before
  the window's own single, final commit; a raw `BEGIN` does not, so a child's
  own successful step still commits the shared transaction — the caller's
  pending write included — as soon as that step succeeds.

`SplitMode`, `DeriveResult`, and `SourceThoughtNotFoundError` are public API under
the same `X.Y.x` stability guarantee.

---

## 1B. Pre-insert preparation seam

`on_store` (§1) runs *after* the row is inserted (on a plain top-level
`create_thought`, after it has already committed), so it is not a place for
pre-insert validation, and any enrichment it returns lands in the caller's
copy of the record, never in the stored row. For validation or persisted
enrichment that must run *before the decisive probe* for a duplicate and before
any row write, override `prepare_thought_for_insert` on your `SqliteEngravaCore` subclass —
a template method, like `_row_to_thought`, not a method on the hooks object.
It has no leading underscore: unlike `_row_to_thought`, this one is a public
override point, and a subclass's override makes the name part of that
subclass's own public surface — hence the public name.

### 1B.1 Contract

| Aspect | Behaviour |
|---|---|
| Default | Pass-through — returns the candidate unchanged |
| Signature | `async def prepare_thought_for_insert(self, thought: ThoughtRecord) -> ThoughtRecord` |
| Runs before | The **decisive** duplicate probe and any row write. `get_or_create` / `upsert_by_hash` run their own **exploratory** probe *before* this — see below; a stable hit there resolves the call and this seam never runs at all. |
| Store lock held while it runs | Never one **this call itself** acquires, on any entry point — `create_thought`, `get_or_create`, `upsert_by_hash` and `bulk_store` all release every lock and transaction they opened before calling this, and reacquire from scratch afterward. See the nesting note below. |
| Raising | Aborts the create — this call inserts no row and appends no journal entry |

**Nesting note.** "Never one this call itself acquires" is not "never any
lock at all". If you call `create_thought` / `get_or_create` / `upsert_by_hash`
/ `bulk_store` from *inside your own* `suspend_auto_commit()` window, that
window's `_write_lock` is still held while this seam runs — it is *your*
lock, acquired before you ever reached this call, not one the call took for
itself. This is not something a pre-insert override can close (a plain
in-process `asyncio.Lock` cannot distinguish "the caller's own reentrant
hold" from "a lock this call should release"), and it is not new: it applies
identically to a raw `create_thought` call inside your own
`suspend_auto_commit()` block, seam or no seam.

**Invocation count, by entry point:**

| Caller | Count |
|---|---|
| `create_thought` (either `deduplicate` value) | Once per call that passes metadata and provenance validation — including a dedup hit |
| `get_or_create` | Zero on a stable pre-existing hit; once when the call reaches the seam after a miss (a race that turns the miss into a hit still counts once) |
| `upsert_by_hash` | Same as `get_or_create` |
| `bulk_store` | Once per item, in input order until one call raises — run for the *whole batch*, holding no lock this call itself acquires (see the nesting note above), before `bulk_store` takes any lock for its insert transaction; not "through" a per-item `create_thought` call the way the other counts might suggest |
| `remember` | Through its own `create_thought` call, so the `create_thought` row applies |

The returned record is revalidated (metadata, provenance) before the decisive
probe or any row write — for `get_or_create` / `upsert_by_hash`, this
revalidation runs *after* their own exploratory probe already found nothing,
not before any probe at all — and — for those two — its
`content` supplies the hash used for the decisive probe that follows: an
override that changes `content` changes what counts as a duplicate for that
call.

**`bulk_store` is the one entry point where this seam does not see
pre-validated input.** Everywhere else, the candidate's `metadata` /
`provenance` are validated immediately before this call runs, so an override
can rely on that precondition. `bulk_store` runs this seam, batch-wide,
holding no lock this call itself acquires, over every item *before* its own
per-item validation — moving validation itself into that same batch-wide
pre-phase would have let a later item's ordinary (non-seam) validation
failure pre-empt an
earlier item's duplicate-id error or `on_store` call, which a bare loop of
`create_thought` calls never does (see `_bulk_store_inner`'s docstring for
the full reasoning). Restoring that per-item failure ordering means
`bulk_store`'s per-item validation now runs in its insert phase, after this
seam already ran for the whole batch — so an override reached through
`bulk_store` sees whatever was passed to `bulk_store`, unvalidated, and must
not assume otherwise.

### 1B.2 Migrating from a `create_thought` override

If your subclass currently overrides `create_thought` for validation or
persisted enrichment, move that logic into `prepare_thought_for_insert` and
let the inherited `create_thought` / `get_or_create` / `upsert_by_hash` /
`bulk_store` orchestration call it for you — every new row written through
`create_thought`, `get_or_create`, `upsert_by_hash`, `bulk_store` or `remember`
passes through it. Keep only one canonical implementation: retaining both the
old override and the new seam risks running your logic twice on a direct
`create_thought` call.

### 1B.3 A pre-existing restriction: `update_thought` on `upsert_by_hash`'s hit branch

`upsert_by_hash`'s hit branch — when the content-hash probe matches an
existing row — updates it, if a mutable field differs, by calling the public,
overridable `update_thought` while `_write_lock` and `_dedup_lock` are held.
**The locked `update_thought` shape predates the
pre-insert seam; the decisive-probe hit route does not.** Before the seam
was wired into `upsert_by_hash`, the method had a single probe — what is
now called the exploratory probe — whose hit branch already called
`update_thought`, when a field differed, under those two locks; that part is unchanged
behaviour. The decisive probe exists only because wiring in the seam split
the miss path into two phases, so that route is new; its hit branch reuses
the same `_upsert_matched_row` implementation as the exploratory probe's,
so it inherits the identical restriction on a route this seam introduced.
A hit on the exploratory probe costs zero calls to
`prepare_thought_for_insert`; a hit on the decisive probe comes after the seam
has already run once for the call (see the invocation-count table above). If
you override `update_thought`, know that your override is called while those
guards are held when it is reached this way: do not call back into
`create_thought(deduplicate=True)` / `get_or_create` / `upsert_by_hash` /
`bulk_store(deduplicate=True)` from inside it on the same task (see the
nesting note in §1B.1 — `_write_lock` is task-reentrant, so re-entering it is
free, but `_dedup_lock` has no legitimate reentrant use, and a same-task second
acquisition raises `DedupLockReentryError` — a "raise, don't hang" backstop
identical in spirit to `WriteLockTimeoutError`'s for a *different* task's
wait, converting what would otherwise be a silent hang on a lock this same
task already holds into an attributable error instead).

This restriction applies on **both** hit routes — the exploratory
probe's hit and the decisive probe's hit alike, since both resolve through
this same `update_thought` call, which is made when a mutable field differs
and while `_write_lock` and `_dedup_lock` are held.

---

## 2. Write your own hook in 20 lines

```python
from __future__ import annotations

from engrava.domain.protocols.hooks import DefaultEngravaHooks, ScoringContext
from engrava.domain.models.thought import ThoughtRecord


class RecencyBoostHooks(DefaultEngravaHooks):
    """Boosts score for recently updated thoughts."""

    async def score_function(
        self,
        thought: ThoughtRecord,
        context: ScoringContext,
    ) -> float:
        # Add a small recency bonus for thoughts updated in a recent cycle.
        # (updated_cycle is an int; updated_at is an ISO-8601 string, not a cycle.)
        if context.current_cycle > 0:
            age = context.current_cycle - thought.updated_cycle
            return max(0.0, 1.0 - age / 100)
        return 0.0


# Registration:
import aiosqlite
from engrava import SqliteEngravaCore

async def build_store(db_path: str) -> SqliteEngravaCore:
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, hooks=RecencyBoostHooks())
    await store.ensure_schema()
    return store
```

`DefaultEngravaHooks` is a no-op base class — override only the methods you
care about.

---

## 3. Custom MindQL verb

A custom MindQL command is an `MindQLExtension` whose `handler` is an async
callable. The executor invokes the handler with **two positional arguments** —
a `ReadOnlyAccessor` wrapping the store's connection and the parsed
extension-argument list — and expects a `list[dict[str, object]]` back. The
accessor's only capability is `execute()`, which runs a single `SELECT`
statement through the same guard as the `SELECT` passthrough command; see
[MindQL: Extension Commands](mindql.md#extension-commands) for the full
read-only contract:

```python
from __future__ import annotations

from engrava.domain.protocols.hooks import MindQLExtension
from engrava.mindql.executor import ReadOnlyAccessor


async def _recent_handler(
    db: ReadOnlyAccessor,
    args: list[str],  # noqa: ARG001 — this command takes no args
) -> list[dict[str, object]]:
    """Return the 100 most recently updated thoughts."""
    cursor = await db.execute(
        "SELECT thought_id, content FROM thought "
        "ORDER BY updated_cycle DESC LIMIT 100"
    )
    rows = await cursor.fetchall()
    return [{"thought_id": row["thought_id"], "content": row["content"]} for row in rows]


RECENT_COMMAND = MindQLExtension(
    command_name="RECENT",
    handler=_recent_handler,
    description="Return the most recently updated thoughts.",
    category="custom",
)
```

The command is registered by listing it in an extension's
`ExtensionManifest.mindql_extensions` (the discovery path), or by passing it
through `MindQLExtension`-keyed `extensions=` when constructing the executor:

```python
from engrava import MindQLExecutor, parse

executor = MindQLExecutor(conn, extensions={"RECENT": RECENT_COMMAND})
# parse() needs the registered verb names to recognise an extension command.
result = await executor.execute(parse("RECENT", known_extensions={"RECENT"}))
```

> Note: `EngravaHooksProtocol.mindql_extension_registry()` is **not** consulted
> by core engrava (see §1.2) — declare custom verbs via `ExtensionManifest`
> or the executor's `extensions=` argument as shown above.

---

## 4. Implementing a contract test

Add a contract test to verify your implementation satisfies the protocol:

```python
from engrava.domain.protocols.hooks import EngravaHooksProtocol


def test_my_hooks_satisfy_protocol() -> None:
    assert isinstance(RecencyBoostHooks(), EngravaHooksProtocol)
```

`EngravaHooksProtocol` is `@runtime_checkable`, so `isinstance` works without
meta-class magic.

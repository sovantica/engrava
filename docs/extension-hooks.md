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
| `on_store` | After a thought is persisted | `ThoughtRecord` (enriched or unchanged) | **active** |
| `on_retrieve` | After a thought is loaded from storage | `ThoughtRecord` (enriched or unchanged) | **active** |
| `decay_function(thought, elapsed_cycles)` | Per-candidate decay factor, multiplied into the hygiene eviction-score | `float` in `[0.0, 1.0]` | **active** — called for each candidate when an enabled `run_hygiene()` pass reaches archive scoring; it is never consulted in search, ranking, or promotion |
| `score_function(thought, context)` | Custom relevance score | `float` | reserved — not called by core |
| `mindql_extension_registry()` | Register custom MindQL verbs | `dict[str, MindQLExtension]` | reserved — core wires MindQL verbs via `ExtensionManifest.mindql_extensions`, not this hook |

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

`SplitMode`, `DeriveResult`, and `SourceThoughtNotFoundError` are public API under
the same `X.Y.x` stability guarantee.

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
the open `aiosqlite.Connection` and the parsed extension-argument list — and
expects a `list[dict[str, object]]` back:

```python
from __future__ import annotations

import aiosqlite

from engrava.domain.protocols.hooks import MindQLExtension


async def _recent_handler(
    db: aiosqlite.Connection,
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

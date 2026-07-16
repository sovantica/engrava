# Core Concepts

Engrava models an agent's memory as a **thought-graph**: typed *thoughts*
connected by typed *edges*, made searchable by *embeddings*, and refined over
time by *dreaming* into higher-order *reflections*. This page explains those
pieces as a mental model — what each is, why it exists, and when you'd create
it — before the how-to guides. Read it once and the rest of the docs will make
more sense.

> For a one-line definition of any term used here (essence, cycle, signal, gate,
> provenance, …), see the [Glossary](glossary.md).

```
                 ┌──────────────────────────────────────────┐
   OBSERVATION   │  "User prefers email over phone"         │  essence (prompt-facing)
   (a thought)   │  content: "Stated during onboarding..."  │  content (full text)
                 │  priority P2 · lifecycle ACTIVE           │
                 └───────────────┬──────────────────────────┘
                                 │ ASSOCIATED  (an edge: typed, weighted)
                 ┌───────────────▼──────────────────────────┐
   BELIEF        │  "This user is low-touch"                 │
                 └───────────────┬──────────────────────────┘
                                 │ CONSOLIDATED_FROM  (created by dreaming)
                 ┌───────────────▼──────────────────────────┐
   REFLECTION    │  cluster summary of related thoughts      │  (higher-order, system-made)
                 └──────────────────────────────────────────┘
```

## Thought

A **thought** (`ThoughtRecord`) is the unit of memory — one idea, fact,
observation, or message. Thoughts are *frozen* (immutable) value objects; you
don't mutate one in place, you `create_thought()` it and later
`update_thought()` to get a new version.

### `essence` vs `content` (two text fields, on purpose)

Every thought carries **two** texts, and the split is deliberate:

- **`essence`** — the compact, canonical, **prompt-facing** one-liner
  (1–200 characters, enforced). This is the text you inject into an LLM prompt
  when this memory is retrieved. Keep it short and self-contained.
- **`content`** — the **full** source text, retained for full-text search and
  provenance. It can be as long as you like.

> Why it matters: when you retrieve memories to build a prompt, you want the
> tight `essence`, not the whole `content`. Putting the same long text in both
> defeats the purpose. Think *headline* (`essence`) vs *article* (`content`).

### `valid_from` / `valid_until` (optional valid time)

A thought also carries two optional, nullable timestamps — `valid_from` and
`valid_until` — that record **when the fact is true in the world**, a separate
axis from when Engrava stored it (`created_at`) and from the [cycle](#cycle-the-agent-clock).
Both default to `None` (an open interval = "valid for all time"), so you can
ignore them entirely until you need point-in-time history. The same two fields
exist on an [edge](#edge). See [The Bi-temporal Model](bitemporal.md) for the full
semantics and the query predicates.

### Thought types

`ThoughtType` is a closed set — choose the one that fits what you're storing:

| Type | What it is | Who creates it |
|---|---|---|
| `OBSERVATION` | Something learned from the world (a user message, a fact) | you (ingest) |
| `BELIEF` | A held conclusion or stance derived from observations | you / your agent |
| `TASK` | Something to be done | you / your agent |
| `OUTPUT_DRAFT` | The agent's own outgoing content (a reply it produced) | your agent |
| `NOTE` | A free-form internal note | you / your agent |
| `REFLECTION` | A cluster summary produced by **dreaming** | the system (don't hand-create) |

There is no `INSIGHT`/`IDEA`/`GOAL` — the set is exactly the six above. Type is
not cosmetic: dreaming only clusters `OBSERVATION`s by default, and
`REFLECTION` is reserved for dreaming's output, so mis-typing changes downstream
behaviour.

### Priority

`Priority` is `P1` (highest) … `P4` (lowest). It is one of the signals that
hybrid search fuses into a ranking, so higher-priority thoughts surface more
readily. Set it to reflect how important a memory is to keep at hand.

### Lifecycle

A thought moves through a small state machine:

```
CREATED → ACTIVE → DONE → ARCHIVED
```

`LifecycleStatus` transitions are enforced (`evolve()` rejects illegal jumps).
Most thoughts you create will start `ACTIVE`. `ARCHIVED` is a **soft-retired**
retention state and a marker for garbage collection — an archived regular thought
is **not** automatically hidden from `search_hybrid` / `list_thoughts` /
`count_thoughts`; it stays searchable until you remove it with `engrava gc`. The
only rows search auto-excludes are **expired** thoughts and **retired
REFLECTIONs**. See [Data Lifecycle](data-lifecycle.md) for the full
retention and garbage-collection behavior.

## Edge

An **edge** (`EdgeRecord`) is a typed, weighted, directional link between two
thoughts — this is what makes Engrava a *graph*, not just a table. The
`EdgeType` set includes `ASSOCIATED`, `DEPENDS_ON`, `DERIVED_FROM`,
`MESSAGE_OF`, `BRIDGE`, `CONSOLIDATED_FROM`, and `CONTESTED_BY`. `weight` (0.0–1.0)
expresses how strong the relation is.

Create edges when a relationship between two memories is itself meaningful —
e.g. one thought supports, contradicts, or depends on another. Dreaming also
creates edges automatically (`ASSOCIATED` between consolidated thoughts, and
`CONSOLIDATED_FROM` from a reflection back to its sources).

## Embedding

An **embedding** is the vector representation of a thought that powers semantic
(meaning-based) search. Embeddings are optional: with no embedding provider
configured, search still works using the bundled lexical (FTS5/BM25) index, and
the vector signal is simply skipped. Configure a provider (and `auto_embed`) to
get semantic retrieval. See [Configuration](configuration.md) and the search
docs for the provider options.

## Reflection

A **reflection** is a `ThoughtType.REFLECTION` thought created by **dreaming**:
Engrava clusters semantically related thoughts and writes a higher-order summary
node, linked back to its members by `CONSOLIDATED_FROM` edges, with a centroid
embedding. Reflections are how a pile of individual observations becomes
fewer, more retrievable, higher-level memories over an agent's lifetime. You do
not create reflections by hand — dreaming makes them. See
[Dreaming](dreaming.md).

## Cycle (the agent clock)

A **cycle** is a *logical clock* — a monotonically increasing integer tick that
**you own and advance**. It is not wall-clock time and not a database row;
Engrava never increments or stores it for you. Typically one cycle = one agent
turn / interaction / scheduled pass.

Three fields use it:

- **`created_cycle`** / **`updated_cycle`** — optional on `ThoughtRecord`, both
  default to `0` (so callers that don't track cognitive cycles can omit them);
  when set, the model enforces `updated_cycle >= created_cycle`. They stamp
  *when, in your agent's logical time*, a thought appeared and last changed.
- **`current_cycle`** — the value you pass into `search_hybrid(...)` and
  `run_consolidation(...)` to tell Engrava "it is now tick N."

Why a cycle exists *alongside* timestamps: it gives recency and dreaming
deterministic, wall-clock-independent math. Search's recency signal and all of
dreaming's age/scheduling gates (`min_age_cycles`, `schedule_every_n_cycles`,
`recency_half_life`) are expressed in cycles, not seconds.

> **The trap to avoid.** Because Engrava does not advance the cycle for you,
> there are two distinct failure modes — and neither raises an error:
>
> - **Omitting it entirely** (`current_cycle=None`, the default in
>   `search_hybrid`) makes the recency signal **inactive** — it is dropped from
>   the ranking and its weight is redistributed to the other signals.
> - **Passing a constant** (e.g. always `current_cycle=0`, and never advancing
>   `created_cycle`/`updated_cycle`) keeps recency active but **useless**: a
>   thought's age is `current_cycle - updated_cycle`, so with everything frozen
>   at the same value every memory looks equally fresh and recency cannot
>   distinguish old from new. The same staleness also means dreaming's age gate
>   (`min_age_cycles`) never opens — `created_cycle`/`current_cycle` never grow,
>   so no thought ever ages enough to be promoted.
>
> **Do this instead:** keep a counter in your application, increment it once per
> turn, pass it as `current_cycle`, and use it for `created_cycle`/`updated_cycle`
> when building thoughts. On restart, recover it from
> [`max_cycle()`](#recovering-the-counter-max_cycle) so it stays monotonic across
> process restarts.

### The other recency axis: transaction time (`recency_now`)

The cognitive cycle is *one* of two recency axes. If your consumer has **no
natural cadence** — it never advances a cycle, so every write lands at cycle `0`
— cycle recency is degenerate (every row looks equally fresh). For that case,
rank by **transaction time** instead: pass `recency_now` (a caller-supplied
ISO-8601 instant) to `search_hybrid` / `recall`, and engrava ages each row by
its `updated_at` (falling back to `created_at`) in wall-clock seconds. This
expresses "recently *stored*" — the signal cycle recency cannot give a
single-cycle store.

- **You pick exactly one axis per query.** `current_cycle` selects cycle
  recency; `recency_now` selects transaction-time recency. An explicit
  `recency_now` **takes precedence over a passive `cycle_provider`**: when you
  pass `recency_now` and no explicit `current_cycle`, the provider is *not*
  consulted, so a provider-configured store can still opt into transaction
  recency. Supplying **both explicit** references
  (`current_cycle` *and* `recency_now`) raises `RecencyModeConflictError` — the
  two clocks are never silently combined.
- **The caller owns "now".** engrava's core reads no wall clock in ranking, so
  retrieval stays deterministic and replayable (same store + same `recency_now`
  → same ranking). A naive `recency_now` is interpreted as UTC (the host
  timezone is never consulted); a malformed `recency_now` (or a non-positive
  `recency_now_half_life`) raises `InvalidRecencyArgumentError`; a row with a
  missing or malformed `updated_at`/`created_at` is treated as maximally old.
- **The half-life is in seconds** here (`recency_now_half_life`, default 604800 =
  7 days), never cycles — the units never mix. See
  [Hybrid Search → Two recency axes](search.md#two-recency-axes) for a worked
  example.

This is a separately-typed second axis, not a replacement: a consumer with a
real cadence keeps using `current_cycle`; one without simply supplies
`recency_now` at query time. The three time axes engrava keeps distinct
(transaction time, valid time, cognitive cycle) are unchanged — recency ranking
can now read the transaction-time axis *as well as* the cognitive-cycle axis.

### Injecting the cycle once (the cycle provider)

Threading `current_cycle` through every `search_hybrid` / `recall` /
`consolidate` / `run_hygiene` call is repetitive when your application already
owns a cadence. As an **opt-in** convenience you can configure a **cycle
provider** once — on the constructor or `from_config` — and those read /
eligibility paths pull the cycle from it whenever you don't pass one explicitly:

```python
from engrava import SqliteEngravaCore, StaticCycleProvider

# Opt-in: wire a cycle source once (constructor shown; from_config takes the
# same keyword). The provider is a live runtime object, never part of config.
store = SqliteEngravaCore(conn, cycle_provider=StaticCycleProvider(0))
```

Resolution is deliberately simple, and an explicit argument always wins:

1. If you pass `current_cycle` (**including `0`**), that value is used — the
   check is "was an argument given", never truthiness, so an explicit `0` never
   silently falls through to the provider.
2. Otherwise, if a provider is configured, its value is pulled **and validated**
   — it must be a real, non-negative `int` (a `bool` is rejected). An invalid
   value raises `CycleProviderError`.
3. Otherwise the cycle stays `None` — exactly today's behaviour (recency off, no
   age-gating). **No provider configured = unchanged**: a store built without one
   behaves byte-for-byte as before.

> **Read-time only.** The provider feeds ranking and eligibility; it **never**
> stamps `created_cycle` / `updated_cycle` on writes. Write-side cycles stay your
> explicit choice on each `ThoughtRecord`.

Three reference providers ship: `StaticCycleProvider(value)` (a fixed value);
`CallableCycleProvider(fn)` (a thin adapter over your own zero-argument callable
— **its purity is your contract, not the adapter's**: whether it avoids
wall-clock reads and returns a genuine cognitive cycle depends entirely on your
`fn`); and `MaxCycleProvider` (below). Whatever a provider returns to be used as
a cognitive cycle **must not** be wall-clock-derived, nor conflate the three axes
Engrava keeps distinct (operation count, cognitive cycle, wall-time recency) —
Engrava validates the *value*, but that containment is **your** obligation.

> **The seam standardises injection; it does not manufacture a cadence.** It
> gives you one plug point instead of threading `current_cycle` everywhere. A
> consumer that already advances a cycle can now plug it in once; a consumer with
> **no** natural cycle still has nothing to supply — the seam enables the wiring,
> it does not invent a clock.

#### Recovering the counter: `max_cycle()`

`max_cycle()` is a read-only accessor returning the store's cognitive-cycle
**high-water mark** — the maximum across *all* cycle-bearing records
(`MAX(thought.updated_cycle)` unioned with `MAX(edge.created_cycle)`), or `0` on
an empty store. Use it to resume your counter after a restart:

```python
# Resume your counter after a restart from the store's high-water mark.
resume_from = await store.max_cycle()  # 0 on an empty store
```

It is unioned across thoughts *and* edges deliberately: an edge created at a
higher cycle than any thought would otherwise be missed, letting a resumed
counter go backwards. The `MaxCycleProvider` reference provider wraps this into a
provider — `await MaxCycleProvider.create(store)` snapshots the current mark, its
`current_cycle()` returns that **cached** (possibly stale) snapshot, and
`await provider.refresh()` re-reads the store when your cadence advances.

> **Chicken-and-egg (disclosed).** On a store where *every* write is stamped
> cycle `0` (a consumer that never advances the cycle), `max_cycle()` returns
> `0`. It helps a consumer that *does* advance cycles resume its counter; it
> cannot recover a cadence that was never expressed in the data.

## Provenance (where a memory came from)

Two distinct fields record origin, and they are easy to confuse:

- **`source`** — a free-form **string** identifier of the origin (e.g.
  `"human"`, `"ingest"`, your component name). Required, your choice.
- **`source_type`** — the **`KnowledgeSource` enum**: how the knowledge was
  obtained.

| `KnowledgeSource` | Set it when the memory came from… |
|---|---|
| `EXPERIENCE` | The agent's own experience / observed reality (the default) |
| `SEEDED_LLM` | Content seeded by an LLM up front |
| `DISTILLED_LLM` | Content distilled/derived by an LLM |
| `DREAMING` | Produced by consolidation — **the system sets this itself** on dream-created edges/reflections |

Provenance is not decoration: dreaming can filter on it (e.g. preferring
experience-based confirmations), so setting `source_type` honestly lets you tune
what consolidation trusts.

## Visibility (inner vs outer speech)

`ThoughtVisibility` marks whether a thought may surface in the agent's **outer
speech** (what it says) or stays **internal** (what it only thinks):

- **`private`** — never disclosed externally; internal memory only.
- **`selective`** — shared with trusted entities on request (the **default**).
- **`public`** — may appear in the agent's outer speech / output.

Engrava *stores* the level; **honouring it is your application's
responsibility** (Engrava won't stop you from reading a `private` thought — it
records the intent so your agent can respect it). Use it to keep a privacy
boundary between what the agent knows and what it's allowed to say.

## Reliability: `confidence` vs `confirmation_count`

A thought carries **two different** notions of how much to trust it, and they
feed dreaming as separate signals:

- **`confidence`** — a static `0.0–1.0` belief-strength **you assign** at
  creation (nullable; treated as `0.5` when unset). "How sure am I of this?"
- **`confirmation_count`** — a counter of how many times the thought has been
  **independently re-encountered / validated** over time. It grows via
  `deduplicate=True` on `create_thought` (identical content bumps the count) or
  your own logic. "How many times has reality re-confirmed this?"

Dreaming's `ConfidenceSignal` reads the first and `ConfirmationSignal` reads the
second, so they tune consolidation in different ways. (Relatedly,
`DreamingGates.allow_zero_confirmation` exists so single-write batch ingest —
where `confirmation_count` never grows — can still be consolidated.)

## Putting it together

```python
import uuid
from engrava import (
    ThoughtRecord,
    ThoughtType,
    Priority,
    LifecycleStatus,
    KnowledgeSource,
    ThoughtVisibility,
)

observation = ThoughtRecord(
    thought_id=str(uuid.uuid4()),
    thought_type=ThoughtType.OBSERVATION,  # learned from the world
    essence="User prefers email over phone",  # prompt-facing one-liner
    content="The user said during onboarding that email is the best way to reach them.",
    priority=Priority.P2,
    lifecycle_status=LifecycleStatus.ACTIVE,
    created_cycle=12,  # your agent's logical clock, this turn
    updated_cycle=12,
    source="onboarding-flow",  # free-form origin id
    source_type=KnowledgeSource.EXPERIENCE,  # how it was obtained
    confidence=0.9,  # how sure you are
    visibility=ThoughtVisibility.SELECTIVE,  # inner/outer-speech boundary
)
```

## Next

- [Quick Start](quickstart.md) — create, link, and search in five minutes.
- [Dreaming](dreaming.md) — how consolidation turns observations into reflections.
- [Hybrid Search](search.md) — how the signals (including recency/cycle and priority) fuse into a ranking.
- [The Bi-temporal Model](bitemporal.md) — the optional second time axis (valid time) and how it differs from the cycle.
- [API Reference](api-reference.md) — the exact fields, enums, and methods.

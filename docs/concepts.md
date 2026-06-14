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

- **`created_cycle`** / **`updated_cycle`** — required on every `ThoughtRecord`
  (the model enforces `updated_cycle >= created_cycle`). They stamp *when, in
  your agent's logical time*, a thought appeared and last changed.
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
> when building thoughts. On restart, recover it (e.g. from the maximum
> `created_cycle` you've stored) so it stays monotonic across process restarts.

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

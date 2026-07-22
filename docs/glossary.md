# Glossary

Short definitions of the terms Engrava uses, each linking to the page that
explains it in depth. New to Engrava? Read [Core Concepts](concepts.md) first —
this page is a quick reference, not a tutorial.

### Thought

The unit of memory — one idea, fact, observation, or message, stored as a frozen
(immutable) `ThoughtRecord`. You don't mutate a thought in place; you
`create_thought()` it and `update_thought()` to get a new version. See
[Core Concepts → Thought](concepts.md#thought).

### Essence

The compact, canonical, **prompt-facing** one-liner of a thought (1–200
characters, enforced) — the text you inject into an LLM prompt when the memory is
retrieved. Think *headline*. See
[Core Concepts → essence vs content](concepts.md#essence-vs-content-two-text-fields-on-purpose).

### Content

The **full** source text of a thought, retained for full-text search and
provenance — as long as you like. Think *article* (to the essence's *headline*).
See [Core Concepts → essence vs content](concepts.md#essence-vs-content-two-text-fields-on-purpose).

### Edge

A typed, weighted, directional link between two thoughts — what makes Engrava a
*graph* rather than a flat table. The `EdgeType` set is `ASSOCIATED`,
`DEPENDS_ON`, `DERIVED_FROM`, `MESSAGE_OF`, `BRIDGE`, `CONSOLIDATED_FROM`, and
`CONTESTED_BY`; `weight` (0.0–1.0) expresses how strong the relation is. See
[Core Concepts → Edge](concepts.md#edge).

### Embedding

The vector representation of a thought that powers semantic (meaning-based)
search. Embeddings are optional — without a provider, search falls back to the
lexical (FTS5) index and the vector signal is skipped. See the
[Embeddings guide](guides/embeddings.md).

### Reflection

A higher-order summary thought (`ThoughtType.REFLECTION`) created by **dreaming**:
Engrava clusters semantically related thoughts and writes a centroid-embedded
summary node, linked back to its members by `CONSOLIDATED_FROM` edges. You don't
create reflections by hand. See [Core Concepts → Reflection](concepts.md#reflection)
and [Dreaming](dreaming.md).

### Dreaming

The periodic, off-the-hot-path consolidation process you invoke with
`run_consolidation()`: it scores stored thoughts, **promotes** the important ones,
links related ones with edges, and clusters them into reflections. No LLM is
involved — it is purely structural. See [Dreaming](dreaming.md).

### Consolidation

Another name for what dreaming does in a single pass — evaluating candidates and
producing promotions, edges, and reflections via `run_consolidation()`. See
[Dreaming](dreaming.md).

### Forgetting

The **subtractive** counterpart to [dreaming](#dreaming) — the two halves of
memory maintenance. An **opt-in, reversible**, no-LLM loop (mechanism:
[Memory Hygiene](#memory-hygiene)) that lets cold, low-signal thoughts fade by
**archiving** them, and — as a *separately* opted-in step — garbage-collects
archived rows after cycle + wall-clock restore windows. OFF by default. See
[Forgetting (Memory Hygiene)](memory-hygiene.md).

### Memory Hygiene

The **mechanism** name for [Forgetting](#forgetting): the deterministic
`run_hygiene()` loop configured under `hygiene_policy`. "Forgetting" is the public
concept, "Memory Hygiene" is the mechanism, and `hygiene` is the API name — the
same concept-over-mechanism layering as Dreaming over `consolidate()`. See
[Forgetting (Memory Hygiene)](memory-hygiene.md).

### Promotion

The act, during consolidation, of marking an important thought by setting its
priority to **P1** so it surfaces more readily in search. Whether a candidate is
promoted depends on the [gates](#gate) and the `promote_threshold`. See
[Dreaming](dreaming.md).

### Cycle

A **logical clock** — a monotonically increasing integer tick that *you own and
advance* (typically one cycle per agent turn). It is not wall-clock time and not
a stored row; Engrava never increments it for you. It drives the recency signal
and dreaming's age gates. Leaving it at `None` makes recency inactive; freezing it
at a constant makes recency useless and stalls dreaming. See
[Core Concepts → Cycle](concepts.md#cycle-the-agent-clock).

### Valid time

The second of Engrava's two time axes: **when a fact is true in the world**, as
opposed to **transaction time** (when Engrava recorded it — `created_at` /
`updated_at`). Valid time is carried by two optional, nullable ISO-8601 fields,
`valid_from` and `valid_until`, on both `ThoughtRecord` and `EdgeRecord`. They
describe a half-open interval (`valid_until` is exclusive); a `None` bound means
*open* (±∞), so an un-annotated record is "valid for all time". Queried through
the `valid_now` / `valid_at` / `valid_within` / `valid_between` MindQL predicates.
See [The Bi-temporal Model](bitemporal.md).

### Transaction time

When Engrava *recorded or last changed* a fact — the `created_at` / `updated_at`
bookkeeping timestamps it sets automatically. It never moves backwards and you do
not manage it; contrast with [valid time](#valid-time) (the real-world axis you
set) and the [cycle](#cycle) (the logical agent clock). See
[The Bi-temporal Model](bitemporal.md).

### Signal

One scoring component that [hybrid search](#hybrid-search) computes for a
candidate and fuses into the final rank. Engrava has five: FTS5 keyword, vector
similarity, recency, priority, and graph. A signal whose prerequisite is missing
(e.g. no embeddings) is skipped rather than erroring. See [Search](search.md).

### Gate

A cheap boolean check in dreaming that a candidate must pass *before* it is scored
for promotion — e.g. `min_age_cycles` (the thought must be old enough) and the
confirmation gate. Gates filter out clearly ineligible thoughts. See
[Dreaming → Gates](dreaming.md#gates).

### Priority

A thought's importance level, `P1` (highest) to `P4` (lowest). It is one of the
hybrid-search signals, so higher-priority thoughts surface more readily; dreaming
**promotes** thoughts to `P1`. See [Core Concepts → Priority](concepts.md#priority).

### Lifecycle

The small state machine a thought moves through: `CREATED → ACTIVE → DONE →
ARCHIVED` (`LifecycleStatus`, with transitions enforced). `ARCHIVED` is a
soft-retired state — the row and its content remain until garbage-collected, but
an archived thought is **excluded from default ranked retrieval** (reversible via
`restore_thought` / `include_archived`). See
[Core Concepts → Lifecycle](concepts.md#lifecycle) and
[Data Lifecycle](data-lifecycle.md).

### Provenance

Where a memory came from, recorded in two fields: `source` (a free-form string id
you choose, e.g. `"onboarding-flow"`) and `source_type` (the `KnowledgeSource`
enum: `EXPERIENCE`, `SEEDED_LLM`, `DISTILLED_LLM`, `DREAMING`). Dreaming can
filter on provenance, so set it honestly. See
[Core Concepts → Provenance](concepts.md#provenance-where-a-memory-came-from).

### Confirmation

`confirmation_count` — a counter of how many times a thought has been
independently re-encountered or validated over time (grows via `deduplicate=True`
or your own logic). Distinct from `confidence`, the static belief-strength you
assign at creation. Dreaming reads them as separate signals. See
[Core Concepts → confidence vs confirmation_count](concepts.md#reliability-confidence-vs-confirmation_count).

### Visibility

`ThoughtVisibility` — whether a thought may surface in the agent's **outer
speech**: `private` (internal only), `selective` (shared on request — the
default), or `public` (may appear in output). Engrava stores the level;
**honouring it is your application's responsibility**. See
[Core Concepts → Visibility](concepts.md#visibility-inner-vs-outer-speech).

### Hybrid search

`search_hybrid()` — retrieval that fuses up to five [signals](#signal) (FTS5
keyword, vector, recency, priority, graph) into one ranked result, rather than
relying on vector similarity alone. See [Search](search.md).

### Graph signal

The fifth, **opt-in** hybrid-search signal: a 1-hop-weighted neighbour boost where
a candidate gains score if its graph neighbours also match the query. Disabled by
default (`default_graph_weight = 0.0`), so no graph queries run unless you enable
it. See [Search](search.md).

### Percept

In the agent loop, an incoming observation (e.g. a user message) stored as an
`OBSERVATION` thought, typically tagged with the `percept(...)` helper. It is what
the agent *takes in*. See [Building a memory-backed agent](guides/agent-memory.md).

### Utterance

In the agent loop, the agent's own outgoing reply, stored as an `OUTPUT_DRAFT`
thought. It is what the agent *produces*. See
[Building a memory-backed agent](guides/agent-memory.md).

## See also

- [Core Concepts](concepts.md) — the same ideas as a guided mental model
- [Search](search.md) — the signal model in depth
- [Dreaming](dreaming.md) — consolidation, gates, promotion, reflections

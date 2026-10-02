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
provenance. Think *article* (to the essence's *headline*).
See [Core Concepts → essence vs content](concepts.md#essence-vs-content-two-text-fields-on-purpose).

### Edge

A typed, weighted, directional link between two thoughts — what makes Engrava a
*graph* rather than a flat table. The `EdgeType` set is `ASSOCIATED`,
`DEPENDS_ON`, `DERIVED_FROM`, `MESSAGE_OF`, `BRIDGE`, `CONSOLIDATED_FROM`, and
`CONTESTED_BY`; `weight` (0.0–1.0) expresses how strong the relation is. See
[Core Concepts → Edge](concepts.md#edge).

### Embedding

The vector representation of a thought that powers semantic (meaning-based)
search. Embeddings are optional — without a provider and without an explicit
`query_vector`, vector retrieval is skipped and search falls back to the
lexical (FTS5) index; a caller-supplied `query_vector` still works with no
provider configured. See the [Embeddings guide](guides/embeddings.md).

### Reflection

A higher-order summary thought (`ThoughtType.REFLECTION`). Reflections that
**dreaming** creates are centroid-embedded and carry lineage edges: Engrava
clusters semantically related thoughts and writes a summary node, linked back
to its members by `CONSOLIDATED_FROM` edges. See
[Core Concepts → Reflection](concepts.md#reflection) and [Dreaming](dreaming.md).

### Dreaming

The periodic, off-the-hot-path consolidation process you invoke with
`run_consolidation()`: it scores eligible candidate thoughts and, when the
corresponding gates, thresholds, caps, embeddings, and feature settings
permit, may **promote** important ones, link related ones with edges, and
cluster them into reflections. The default scoring signals make no LLM
calls, and reflection content is built by a deterministic structural
function — but a custom signal you register (`DreamingExtension`'s
`custom_signals`) runs whatever code it contains, including a call to an
LLM. See [Dreaming](dreaming.md).

### Consolidation

Another name for what dreaming does in a single pass — evaluating candidates
and, conditionally, producing promotions, edges, and reflections via
`run_consolidation()`. See [Dreaming](dreaming.md).

### Forgetting

The **subtractive** counterpart to [dreaming](#dreaming) — the two halves of
memory maintenance. An **opt-in** loop (mechanism:
[Memory Hygiene](#memory-hygiene), built-in scoring makes no LLM calls) that
lets cold, low-signal thoughts fade by **archiving** them — the default
action, reversible via `restore_thought` — and, as a *separately* opted-in
step, garbage-collects archived rows after cycle + wall-clock restore
windows. Garbage collection physically deletes rows and is **not** reversible
the way archiving is. OFF by default. See
[Forgetting (Memory Hygiene)](memory-hygiene.md).

### Memory Hygiene

The **mechanism** name for [Forgetting](#forgetting): the `run_hygiene()` loop
configured under `hygiene_policy`, deterministic for a fixed store,
configuration, cycle, and `now` when any configured custom hooks are
deterministic too. "Forgetting" is the public concept, "Memory Hygiene" is the
mechanism, and `hygiene` is the API name — the same
concept-over-mechanism layering as Dreaming over `consolidate()`. See
[Forgetting (Memory Hygiene)](memory-hygiene.md).

### Promotion

The act, during consolidation, of marking an important thought by setting its
priority to **P1**. With a positive priority weight, that gives an
already-retrieved candidate the largest priority boost in hybrid search;
priority does not itself add a thought to the candidate set. Whether a
candidate is promoted depends on the [gates](#gate) and the
`promote_threshold`. See [Dreaming](dreaming.md).

### Cycle

A **logical clock** — a monotonically increasing integer tick that *you own and
advance* (typically one cycle per agent turn). It is not wall-clock time and not
a stored row; Engrava never increments it for you. It drives the recency signal
and dreaming's age gates. With no explicit recency reference and no configured
cycle provider, recency is inactive; an explicit `recency_now` instead selects
transaction-time recency. Freezing a cognitive cycle at a constant stops
cycle-based age and recency from advancing, and can stop a cadence-based
`run_if_due()` call from becoming due — but does not stop a direct
`run_consolidation()` call from executing and mutating memory. See
[Core Concepts → Cycle](concepts.md#cycle-the-agent-clock).

### Valid time

One of Engrava's three time axes: **when a fact is true in the world**, as
opposed to **transaction time** (when Engrava recorded it — `created_at` /
`updated_at`) and the consumer-owned cognitive [cycle](#cycle). Valid time is
carried by two optional, nullable ISO-8601 fields,
`valid_from` and `valid_until`, on both `ThoughtRecord` and `EdgeRecord`. They
describe a half-open interval (`valid_until` is exclusive); a `None` bound means
*open* (±∞) for the NULL-tolerant predicates `valid_now` / `valid_at` /
`valid_within`, so an un-annotated record is "valid for all time" under those.
`valid_between` is the exception: it requires a real (non-`None`) value on
both stored bounds and excludes a record where either is open. See
[The Bi-temporal Model](bitemporal.md).

### Transaction time

When Engrava *recorded or last changed* a fact — the `created_at` / `updated_at`
bookkeeping timestamps. Engrava supplies these when you omit them, but both are
caller-settable fields, and updating one explicitly is not checked against the
previous value; contrast with [valid time](#valid-time) (the real-world axis
you set) and the [cycle](#cycle) (the logical agent clock). See
[The Bi-temporal Model](bitemporal.md).

### Signal

One scoring component that [hybrid search](#hybrid-search) computes for a
candidate and fuses into the final rank. Engrava has five: FTS5 keyword, vector
similarity, recency, priority, and graph. A signal whose prerequisite is missing
(e.g. no embeddings) is skipped rather than erroring. See [Search](search.md).

### Gate

A cheap boolean check in dreaming that a candidate must pass to be **promoted**
— e.g. `min_age_cycles` (the thought must be old enough) and the confirmation
gate. Gates filter out clearly ineligible thoughts. See
[Dreaming → Gates](dreaming.md#gates).

### Priority

A thought's importance level, `P1` (highest) to `P4` (lowest). It is one of the
hybrid-search signals: when priority weighting is enabled and positive, a
higher-priority thought already in a hybrid-search candidate set scores higher;
dreaming **promotes** thoughts to `P1`. See
[Core Concepts → Priority](concepts.md#priority).

### Lifecycle

The small state machine a thought moves through (`LifecycleStatus`, with
transitions enforced): `CREATED → ACTIVE`, `ACTIVE → DONE`, `ACTIVE → ARCHIVED`
(direct, without passing through `DONE`), `DONE → ARCHIVED`, and the reverse
`ARCHIVED → ACTIVE`; a record can also be constructed with an initial
lifecycle value directly. `ARCHIVED` is a
soft-retired state — the row and its content remain until garbage-collected, but
an archived thought is **excluded from default ranked retrieval** (reversible via
`restore_thought` / `include_archived`). See
[Core Concepts → Lifecycle](concepts.md#lifecycle) and
[Data Lifecycle](data-lifecycle.md).

### Provenance

Where a memory came from. `source` is a free-form identifier and `source_type`
is the `KnowledgeSource` enum. The optional, typed `ProvenanceContext` records
caller-supplied write-time context such as session, actor, retrieval query, and
retrieved thought ids. These values are untrusted lineage hints, not identity,
authorization, or proof that a claim is true. Semantic evidence and the
hash-chain mutation journal are separate provenance layers. See
[Core Concepts → Provenance](concepts.md#provenance-where-a-memory-came-from) and
[Evidence and conflicts](evidence-and-conflicts.md#three-kinds-of-provenance).

### Conflict

A caller-detected tension between claims. Engrava lets incompatible thoughts
coexist and provides the directional `CONTESTED_BY` edge label, valid-time
bounds, metadata, and provenance needed to represent the tension. It does not
resolve entities, infer a contradiction, choose the true claim, or create a
clarification task. See [Evidence and conflicts](evidence-and-conflicts.md).

### Confirmation

`confirmation_count` — a counter of deduplication hash hits (grows via
`deduplicate=True`) or your own logic; Engrava does not establish that a hit is
an independent re-encounter. Distinct from `confidence`, a belief-strength you
assign that may be supplied at creation and changed later. Dreaming reads them
as separate signals. See
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
default (`default_graph_weight = 0.0`), so no graph ranking queries run unless you
enable it. Candidate-pool expansion over consolidation edges is a separate step
controlled by `graph_expansion_enabled`, which is on by default and reads those
edges only when a reflection ranks among the top candidates. See
[Search](search.md).

### Percept

In the agent loop, an incoming observation (e.g. a user message) stored as an
`OBSERVATION` thought, typically tagged with the `percept(...)` helper. It is what
the agent *takes in*. See [Building a memory-backed agent](guides/agent-memory.md).

### Utterance

In the agent loop, the agent's own outgoing reply, typically stored as an
`OUTPUT_DRAFT` thought. It is what the agent *produces*. See
[Building a memory-backed agent](guides/agent-memory.md).

## See also

- [Core Concepts](concepts.md) — the same ideas as a guided mental model
- [Search](search.md) — the signal model in depth
- [Dreaming](dreaming.md) — consolidation, gates, promotion, reflections
- [Evidence and conflicts](evidence-and-conflicts.md) — claims, lineage, and contested facts

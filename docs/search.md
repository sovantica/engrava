# Hybrid Search

engrava's `search_hybrid()` combines up to **five** scoring signals
into a single ranked result list.

## Signal Model

| # | Signal | Weight key | Default | Source |
|---|--------|------------|---------|--------|
| 1 | **FTS5 keyword** | `default_fts_weight` | `0.30` | BM25 full-text score (min-max normalized) |
| 2 | **Vector similarity** | `default_vector_weight` | `0.55` | Cosine similarity from embedding search |
| 3 | **Recency** | `default_recency_weight` | `0.10` | Exponential decay along one recency axis — cognitive-cycle (`current_cycle`) or transaction-time (`recency_now`); see [Two recency axes](#two-recency-axes) |
| 4 | **Priority** | `default_priority_weight` | `0.05` | Boost multiplier per priority level (P1–P4) |
| 5 | **Graph** | `default_graph_weight` | `0.00` | 1-hop-weighted neighbour boost (opt-in) |

Default weights sum to `1.0`. When a signal is unavailable (e.g. neither
recency reference nor a cycle provider can resolve recency, or no embeddings
can resolve the vector arm),
its weight is **redistributed proportionally** across active signals.

## Graceful Degradation

- FTS5 unavailable or empty query → FTS skipped.
- `query_vector` is `None` and no embedding provider → vector skipped.
- No explicit `current_cycle`, no configured cycle provider, and no
  `recency_now` → recency skipped.
- `priority_weight` is `0.0` → priority skipped.
- `graph_weight` is `0.0` → graph skipped, zero overhead.
- All signals disabled → fallback to `list_thoughts(LIMIT top_k)`.

The vector arm distinguishes two bad-query-vector cases:

- A **degenerate** `query_vector` (empty, all-zero, or non-finite) → the vector
  arm returns nothing and the read-only `vector_arm_degradation_count` counter is
  incremented (a bad query embedding, not an empty corpus). It does **not** raise.
- A **wrong-dimension** `query_vector` (its length differs from the store's
  embedding dimension) is a caller-contract violation and **raises**
  `VectorDimensionMismatchError` — it is not silently degraded. See
  [Observability signals](observability.md#observability-signals) and
  [Known Limitations](known-limitations.md#query-vector-dimension-mismatch).

Both health counters are read-only, monotonically increasing properties on one
store instance and reset when a new store is constructed:

```python
print(store.fts_match_failure_count)
print(store.vector_arm_degradation_count)
```

`fts_match_failure_count` increments before the one safe fallback retry whenever
the primary normalized FTS5 `MATCH` expression fails. The fallback may still
return valid matches; a non-zero count therefore means recovery occurred, not
that the FTS arm was necessarily lost. `vector_arm_degradation_count` increments
only for degenerate vectors. A dimension mismatch is not counted because it is
raised as a typed contract error.

> **v0.6.0 compatibility note.** A wrong-dimension query vector is rejected with
> `VectorDimensionMismatchError` instead of being confused with an empty
> neighbourhood or leaking a backend-specific shape error. Callers that accept
> externally supplied vectors should catch this error (or `EngravaError`) and
> correct the embedding/model configuration; retrying the same vector cannot
> produce a valid result.

## Archived thoughts are excluded by default

Every ranked read — `search_hybrid()`, `recall()`, `search_fts()`, and
`search_similar()` — drops **archived** thoughts (`lifecycle_status = ARCHIVED`)
from its default candidate set, the same eligibility class as expired rows and
retired reflections. This is the retrieval side of
[Forgetting](memory-hygiene.md) — an **opt-in, off-by-default** hygiene loop — but
the exclusion applies to any archived row, whether or not that loop is enabled: an
archived (forgotten) thought stops surfacing without being deleted.

The exclusion is **reversible**:

- `store.restore_thought(thought_id)` flips the row back to `ACTIVE`, so it is
  eligible again; and
- passing `include_archived=True` to any of the four methods re-admits archived
  rows for that one call, without restoring them.

> **Behaviour change.** Marking a thought `ARCHIVED` — including via the TTL
> `archive` strategy — now removes it from default retrieval; previously an
> archived thought still surfaced in search. It is still counted by
> `count_thoughts()` / `list_thoughts()` (those are not ranked retrieval); to
> exclude it there, filter on `lifecycle_status` yourself. The retired-reflection
> freshness floor is independent — a retired `REFLECTION` stays excluded even under
> `include_archived=True`. See [Data lifecycle](data-lifecycle.md#lifecycle-states)
> and [Known Limitations](known-limitations.md#archived-thoughts-and-default-retrieval).

## Two recency axes

Recency ranks along **two separately-typed axes**, and a query picks **exactly
one**. Passing neither explicit reference uses a configured cycle provider when
present; without one, recency remains off:

| Axis | Reference | Ages a row by | Half-life unit | For |
|---|---|---|---|---|
| **Cognitive-cycle** | `current_cycle` | `updated_cycle` vs the cycle | cycles (`recency_half_life`) | agents that own and advance a logical cycle |
| **Transaction-time** | `recency_now` | `updated_at` (→ `created_at`) vs the instant | seconds (`recency_now_half_life`) | "recently stored" — wall-time recency on any store |

Both use the same exponential half-life decay; only the clock differs. The
transaction-time axis makes "recently written" rankable even on a store where
every row shares one cognitive cycle (the common case for externally-written
memories) — the signal cycle-recency cannot express there.

`recency_now` is a caller-supplied ISO-8601 instant: engrava's core reads **no
wall clock** in ranking, so retrieval stays deterministic and replayable (same
store + same `recency_now` → same ranking). A naive value is interpreted as UTC
(the host timezone is never consulted); a malformed `recency_now` (or a
non-positive `recency_now_half_life`) raises `InvalidRecencyArgumentError`. A row
whose `updated_at`/`created_at` is missing or malformed is treated as maximally
old (the minimum recency score). An explicit `recency_now` **takes precedence
over a passive `cycle_provider`** (the provider is not consulted when
`recency_now` is supplied with no explicit `current_cycle`); supplying **both
explicit** references — `current_cycle` *and* `recency_now` — raises
`RecencyModeConflictError`.

```python
# Transaction-time recency: rank by how recently each memory was written,
# relative to a caller-supplied "now" (a 24-hour freshness half-life).
from datetime import UTC, datetime

results = await store.search_hybrid(
    "incident timeline",
    recency_now=datetime.now(UTC).isoformat(),  # the caller owns "now"
    recency_weight=0.4,
    recency_now_half_life=86400,  # seconds (1 day); default 604800 (7 days)
)
```

The units never mix: a wall-clock age is never subtracted from a cognitive
cycle. To use transaction-time recency on a store that has a cycle provider
configured, **pass `recency_now` while omitting an explicit `current_cycle`** —
the provider is passive and is not consulted, so the query ranks by transaction
time. (Omitting `recency_now` instead selects the provider's cognitive-cycle
recency.) You only hit `RecencyModeConflictError` if you explicitly pass *both*
`current_cycle` and `recency_now`.

## Keyword query syntax (FTS)

The keyword signal — and the `search_fts()` method that exposes it directly —
runs your text against an SQLite FTS5 index. engrava
normalises the query before handing it to FTS5, with two modes that switch
automatically on what you type:

**Bare queries are matched with `OR`.** A plain natural-language query like
`what was my sister doing` is treated as a bag of words joined with `OR`, so a
document matches when it shares **any** word. BM25's IDF weighting then ranks the
documents that share the most *distinctive* words first, so common function words
(`what`, `was`, `my`) carry little weight and need no stopword list or stemmer —
this works in any language. (Before this, the words were joined with FTS5's
implicit `AND`, so a question only matched documents containing *every* word and
relevant answers were missed.)

```python
# Bare query -> OR-matched: finds docs sharing any content word, best-ranked first
hits = await store.search_fts("what was my sister doing", top_k=10)
```

**Expert syntax is preserved unchanged.** If your query uses FTS5 operators, it is
passed through as written:

- **quoted phrases** — `"machine learning"` matches the exact phrase;
- **uppercase booleans** — `AND`, `OR`, `NOT` (must be uppercase) compose terms,
  e.g. `python AND NOT snake`;
- **prefix** — a trailing `*` does prefix matching, e.g. `neur*`;
- **column filters** — `essence:` and `content:` restrict a term to that column,
  e.g. `content:berlin`.

**Punctuation never raises.** Unsafe characters split a token into separate terms
rather than breaking the query: a contraction like `sister's` becomes `sister OR
s`, so it still matches a stored `sister's dog`. Pasting a URL or a timestamp is
safe too — only the real `essence:` / `content:` column filters are honoured, so
`http://example.com` and `12:30` are treated as ordinary search terms (they do
**not** become spurious `http:` / `12:` column filters). When a normalized
full-text expression is a genuinely malformed FTS5 query, engrava logs a warning,
increments the read-only `fts_match_failure_count` counter, and **retries once**
through the bare normalization (unsafe characters dropped, wildcards collapsed to
legal prefixes, any exposed `AND`/`OR`/`NOT` phrase-quoted so FTS5 cannot read it
as an operator), which is always a valid MATCH; the FTS arm returns that query's
matches (an empty set when the sanitized query matches nothing). See
[Observability signals](observability.md#observability-signals).

## Graph-Aware Ranking

The graph signal uses **1-hop-weighted neighbour boost**.  If a
candidate thought's graph neighbours also match the query, the
candidate receives a boost proportional to the neighbour's semantic
score and the connecting edge weight.

### Algorithm

```
For each candidate C in the fusion pool:
  neighbours = get_edges(C, direction="BOTH", limit=max_neighbors)
               ordered by edge.weight DESC (deterministic)
  For each (edge, neighbour):
    neighbour_base = max(fts_score[neighbour], vector_score[neighbour])
    boost[C] += edge.weight × neighbour_base × graph_edge_decay
final_score[C] += graph_weight × boost[C]
```

Key properties:

- **Only semantic scores propagate** — priority, recency, and graph
  scores are excluded from `neighbour_base` to prevent hub-cascade
  effects.
- **No new candidates** — graph signal re-ranks existing results, it
  does not add thoughts to the result set.
- **Deterministic** — neighbours are sorted by `edge.weight DESC`
  before the `max_neighbors` cap is applied.

### Configuration

```yaml
search:
  default_graph_weight: 0.0          # opt-in (0.0 = disabled)
  graph_edge_decay: 0.5              # decay factor for 1-hop distance
  max_neighbors_per_candidate: 5     # safety cap
```

Per-query override:

```python
result = await store.search_hybrid(
    "python async",
    graph_weight=0.1,
    graph_edge_decay=0.3,
)
```

### Performance

When `graph_weight=0.0` (default), no graph queries are executed and
there is zero performance impact.  When active, the implementation
uses a single batch SQL query bounded by
`O(top_k × max_neighbors_per_candidate)` rows.

### Observability

When the graph signal contributes to at least one candidate,
`"graph"` appears in `HybridSearchResult.backends_used`.

## Reflection-source candidate expansion

Graph expansion is separate from the weighted graph ranking signal above. It
is a bounded candidate-generation step over dreaming lineage, not a sixth
fusion signal:

1. After the available fusion signals have scored the current pool, engrava
   selects up to `graph_expansion_top_n` top-ranked `REFLECTION` candidates.
2. It follows each reflection's outgoing `CONSOLIDATED_FROM` edges, ordered by
   edge weight, and admits only source thoughts of type `OBSERVATION`.
3. Each admitted source receives
   `parent_score × graph_expansion_propagation_factor × edge.weight`; when the
   source is already present, the higher of its existing and propagated score
   wins.

Unlike graph-aware ranking, expansion **can add new candidates** and is not
controlled by `graph_weight`. It is enabled by default even though the weighted
graph signal defaults to `0.0`. Expansion re-applies expiry, archive, metadata,
and visibility eligibility, so it cannot bring back a source excluded by the
active query (unless `include_archived=True` explicitly re-admits archived
rows). Non-observation targets are ignored. Expansion runs before the final
reflection filter, so eligible source observations may still be admitted when
`include_reflections=False`; only the reflection rows are removed.

The store-level controls are:

```yaml
search:
  graph_expansion_enabled: true
  graph_expansion_top_n: 5
  graph_expansion_propagation_factor: 0.7
  graph_expansion_max_sources_per_reflection: 20
  graph_expansion_reflection_source_ceiling: 50
```

The per-reflection source cap keeps traversal bounded. A reflection with more
than `graph_expansion_reflection_source_ceiling` lineage sources is skipped
entirely, preventing a very large cluster from flooding the candidate pool.
These settings belong to `SearchConfig`; `search_hybrid()` has no per-query
expansion override. When expansion adds or improves at least one candidate,
`"graph_expansion"` appears in `HybridSearchResult.backends_used`.

## Per-Query Overrides

All weights can be overridden per call via keyword arguments:

```python
result = await store.search_hybrid(
    "quantum computing",
    query_vector=embedding,
    fts_weight=0.4,
    vector_weight=0.4,
    recency_weight=0.1,
    priority_weight=0.05,
    graph_weight=0.05,
    current_cycle=42,
)
```

## Scoped retrieval

`search_hybrid()` and `recall()` accept two optional, keyword-only filter
arguments that scope the ranked query to rows whose `metadata` satisfies a
typed predicate. Both default to `None` (no filtering, the candidate set is
unchanged).

```python
from engrava import FieldOp, FieldPredicate, MetadataFilter, VisibilityQueryFilter

# Equality / membership over your metadata keys (an AND of predicates):
result = await store.search_hybrid(
    "deployment runbook",
    top_k=10,
    filters=MetadataFilter(
        [
            FieldPredicate("$.project", FieldOp.EQ, "atlas"),
            FieldPredicate("$.env", FieldOp.IN, ["staging", "prod"]),
        ]
    ),
)

# "Public, or mine" — admit public rows plus rows this user owns:
result = await store.recall(
    "deployment runbook",
    top_k=10,
    visibility=VisibilityQueryFilter(allowed={"public"}, owner="u1"),
)
```

**Where the filter is applied.** The predicate runs **inside each search arm,
before that arm's limit** — so a narrow filter still returns up to `top_k`
matching rows and is never starved by out-of-filter candidates consuming the
ranking budget. This is the key advantage over over-fetching and post-filtering
(which can starve `top_k`) or a raw `json_extract` pre-filter (which loses the
hybrid ranking).

**Semantics:**

- `filters` is a `MetadataFilter` — an `AND` of `FieldPredicate(path, op, value)`
  over JSON paths (`$`, `$.key`, `$.key[0]`). Operators are `EQ` and `IN`.
- `EQ None` matches both a missing path and an explicit JSON `null`. An empty
  `IN` matches nothing. An empty `MetadataFilter` is a match-all no-op.
- Values compare under **SQLite value equality**: booleans are stored as
  integers, so `EQ True` matches a stored `1` (and `EQ False` matches `0`).
- `visibility` is a `VisibilityQueryFilter(allowed, owner)` reading
  `$.visibility` and `$.owner`. It composes with `filters` by `AND`; its `OR`
  is bounded and parenthesised, so an `owner` match can never escape the
  metadata `AND`.
- Filter objects validate **at construction** (bad path, unsupported operator,
  out-of-range integer, non-finite float, an all-empty `VisibilityQueryFilter`)
  and raise `InvalidFilterError` / `InvalidFilterPathError` — never mid-query.

> **A filter is a query refinement, not an isolation boundary.** It narrows what
> a ranked query *considers*; it performs no authentication, authorization,
> ownership validation, or write enforcement. The `visibility` filter reads only
> the metadata your application wrote, and a caller can omit it or supply any
> `owner`. **Do not use it to keep one tenant's data away from another** — for
> tenant isolation give each tenant its own store via `EngravaManager`
> (see [migrating-from-other-memory.md](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy)).

**Determinism.** Results are ordered by combined score descending, with ties
broken by canonical `thought_id` ascending — a stable total order independent of
the underlying scan order. (This tie-break is always applied, with or without a
filter.)

> **Note on BM25 scores.** The keyword arm's BM25 ranking uses store-global
> corpus statistics. A filter changes which rows are *eligible*, but does not
> recompute term frequencies over the filtered subset — relative scores are
> still computed against the whole corpus. This is the intended behaviour for a
> query refinement within one memory.

### De-fragmentation / collapse

`search_hybrid()` and `recall()` also accept an optional, keyword-only
`collapse_key` that **de-duplicates fragments of the same logical unit** in the
result. When many rows in your store describe the *same* thing — for example
several chunks of one conversation turn, or several versions of one document —
a plain ranked query can fill the top-`k` window with near-duplicates of one
unit and push other, distinct units out. `collapse_key` keeps only the single
best-ranked row per unit and lets deeper *distinct* units take the freed slots.

```python
# One best row per (caller-defined) unit; deeper distinct units backfill in.
result = await store.recall(
    "what did we decide about the rollout?",
    top_k=10,
    collapse_key="$.turn_id",            # a single metadata path
)

# Composite unit key — rows share a unit only if ALL components are equal:
result = await store.search_hybrid(
    "rollout decision",
    top_k=10,
    collapse_key=["$.session_id", "$.turn_index"],
)
```

**What it does.** Among the already-ranked candidates, the highest-ranked row of
each unit is kept; lower-ranked rows of that *same* unit are removed, and the
slots they vacate are backfilled by the next distinct units. So the prompt sees
one best row per unit plus more distinct units, instead of a pile of fragments
of one unit. `collapse_key=None` (the default) leaves the result exactly
unchanged.

**Semantics:**

- `collapse_key` is a single JSON path (`"$.turn_id"`) or an ordered sequence of
  paths forming a **composite key** (`["$.session_id", "$.turn_index"]`). Paths
  use the same grammar as `filters` (`$`, `$.key`, `$.key[0]`) and are validated
  **at call time** — a malformed path raises `InvalidFilterPathError`, never
  mid-query.
- A row whose key is **missing**, holds malformed metadata, or (for a composite
  key) is missing **any** component is treated as its **own** unit — it is never
  collapsed with another row. Distinct key-less rows all survive.
- The kept row per unit is the highest-ranked one under the same deterministic
  order used everywhere on this path (combined score descending, then canonical
  `thought_id` ascending), so the keeper and final order are stable across runs.
- To give backfill a deeper pool to draw from, each search arm's candidate
  budget is widened by a small, bounded factor **only while** `collapse_key` is
  set (configurable as `search.collapse_pool_factor`, default `4`); the
  `collapse_key=None` path is never widened. Because the keyword arm's scores
  are min-max normalized over the candidate set, this wider pool can rescale the
  normalized fusion scores and shift the relative order among units — so setting
  `collapse_key` is not score-neutral even though the collapse step itself never
  mutates a score. Only `collapse_key=None` is byte-identical to a plain query.

**Keeping more than one row per unit (`collapse_max_per_unit`).** By default
collapse keeps a *single* best row per unit. When a unit is genuinely a long,
multi-part thing — say one long turn split into several chunks whose useful
detail is spread across more than the top chunk — dropping every chunk but the
best one can discard the part you actually need. The optional, keyword-only
`collapse_max_per_unit` sets how many of a unit's highest-ranked rows may reach
the result, so a long unit can keep its deeper rows while the remaining slots
still backfill deeper *distinct* units:

```python
# Keep up to 2 rows of each unit; distinct deeper units still backfill.
result = await store.search_hybrid(
    "rollout decision",
    top_k=20,
    collapse_key="$.turn_id",
    collapse_max_per_unit=2,
)
```

- `None` (the default) keeps exactly one best row per unit — identical to
  passing only `collapse_key`. An integer `>= 1` keeps up to that many of a
  unit's highest-ranked rows; `1` is the single-best-row default.
- It only takes effect together with `collapse_key` (there is no unit to
  retain by otherwise), and is validated at call time — a value below `1` raises
  `InvalidFilterError`.
- It only relaxes the per-unit retention count. It never adds a row a search arm
  did not produce, never mutates a score, and never merges or drops a *distinct*
  unit as a side effect; the usual `top_k` truncation still applies, so distinct
  units beyond `top_k` are dropped exactly as they are without it. Key-less rows
  are unaffected — each is already its own unit.

> **Collapse is a presentation convenience, not a filter and not an isolation
> boundary.** It does not change which rows are *eligible* (use `filters` /
> `visibility` for that). The collapse step itself mutates no score — it only
> decides which of the already-eligible rows is shown per unit, dropping
> lower-ranked members of the same unit. (Setting `collapse_key` does widen the
> internal candidate pool, which can affect normalized fusion scores and order,
> as noted above.) It is only as meaningful as the unit metadata your
> application writes: if two genuinely different rows carry the same key they
> will be treated as one unit, and rows without the key are never merged.
> Eligibility (`filters` / `visibility`) always applies first, then collapse.

### Whole-turn assembly (a caller-side recipe)

Collapse and `collapse_max_per_unit` decide *which rows* reach your prompt. If you
also want the reader to see a unit's **contiguous** text — a whole conversation
turn in order, not only the chunks that happened to rank — assemble it yourself
from the results, using only the public API. engrava stores no intrinsic
chunk-sequence, so this works exactly to the extent your ingest wrote **(a)** a
stable unit key and **(b)** an orderable ordinal (e.g. a `chunk_index`) on each
chunk.

```python
from engrava.domain.models.filters import FieldOp, FieldPredicate, MetadataFilter

# 1. Retrieve as usual — collapse dedupes fragments; distinct units backfill.
result = await store.search_hybrid(
    "what did the assistant explain about retries?",
    top_k=20,
    collapse_key=["$.session_id", "$.turn_index"],
    collapse_max_per_unit=2,
)

# 2. For a result you want in full, read its unit-key value from its own metadata,
#    gather the unit's chunks with a metadata-filtered search, then order them by
#    your ordinal and concatenate.
async def assemble_unit(query: str, thought_id: str) -> str:
    seed = await store.get_thought(thought_id)
    assert seed is not None
    unit = await store.recall(
        query,                       # any query — the filter selects the unit
        top_k=100,                   # generous, to cover the unit's chunk count
        filters=MetadataFilter([
            FieldPredicate("$.session_id", FieldOp.EQ, seed.metadata["session_id"]),
            FieldPredicate("$.turn_index", FieldOp.EQ, seed.metadata["turn_index"]),
        ]),
    )
    chunks = [await store.get_thought(tid) for tid, _score in unit.results]
    ordered = sorted((c for c in chunks if c), key=lambda t: t.metadata["chunk_index"])
    return "\n".join(t.content for t in ordered)
```

- **No new API, no result-shape change.** This uses only `get_thought` and the
  metadata `filters=` you already have on `search_hybrid()`/`recall()` — the same
  surface you call to read a result's text. `search_hybrid()`'s result stays
  `(thought_id, score)` tuples.
- **The metadata predicate lives on the ranked search surface**
  (`search_hybrid`/`recall` `filters=`), not on the public `search_similar` or
  `search_fts` methods or on `list_thoughts` (whose filters cover the built-in
  columns + provenance, not arbitrary metadata), so gathering a unit's chunks
  is a filtered hybrid search with a generous `top_k`; the ranking is
  irrelevant because you re-order by your own ordinal.
- **Precondition (yours to guarantee).** Contiguous order is only as good as the
  metadata you wrote: without a stable, orderable ordinal on each chunk, siblings
  can be *fetched* but not meaningfully *ordered* — engrava has no chunk-sequence
  concept of its own. Write the ordinal at ingest alongside the unit key.
- **Where it runs.** Entirely on the results engrava already returned — prompt and
  context assembly is your application's concern, not the store's.

## Configuration Reference

See [configuration.md](configuration.md) for the full YAML reference
of `SearchConfig` fields.

## Querying reflections

After `DreamingExtension.run_consolidation()` runs its clustering phase,
`ThoughtType.REFLECTION` meta-thoughts exist in the store.  Three knobs
control how hybrid search handles them.

### `include_reflections` (default `True`)

When `False`, REFLECTION thoughts are **excluded** from `search_hybrid()`
results.  Useful when you want raw observations / insights without
higher-order aggregates:

```python
result = await store.search_hybrid(
    "machine learning",
    query_vector=embedding,
    include_reflections=False,
)
```

### `reflection_boost` (default `SearchConfig.reflection_boost = 1.0`)

When REFLECTIONs are included, their final score is multiplied by this
factor.  The default `1.0` leaves REFLECTIONs competing on equal footing;
raise it above `1.0` for a modest upranking so high-level abstractions
surface for broad queries without dominating narrow ones.

```python
# Stronger boost — reflections rank near the top for broad queries
result = await store.search_hybrid(
    "patterns in memory",
    query_vector=embedding,
    reflection_boost=1.5,
)

# Disable boost — reflections compete on equal footing
result = await store.search_hybrid(
    "specific fact",
    query_vector=embedding,
    reflection_boost=1.0,
)
```

Configure the default in YAML:

```yaml
search:
  reflection_boost: 1.0   # applies when reflection_boost not overridden per-call
```

### `reflection_topk_cap` (default `0.3`)

Caps how much of the final top-K window `REFLECTION` thoughts may occupy. After
all signals are applied, if reflections hold more than `top_k *
reflection_topk_cap` slots, the lowest-scoring excess reflections are evicted
and the freed slots are backfilled with the highest-scoring off-list
non-reflection candidates. Set it to `1.0` to disable the cap.

When the cap actually evicts, the result carries a programmatic signal and the
event is logged at `INFO`:

```python
result = await store.search_hybrid("patterns in memory", query_vector=embedding, top_k=10)
if result.reflections_evicted:
    # e.g. cap=0.3 * top_k=10 → 3 reflection slots; extra reflections were evicted
    print(f"{result.reflections_evicted} reflection(s) evicted by reflection_topk_cap")
```

`HybridSearchResult.reflections_evicted` is `0` on every query where the cap did
not evict (its default), so existing consumers are unaffected; a positive value
means the cap reshaped the top-K window.

### `search_reflections_only()`

Convenience helper that returns **only** REFLECTION thoughts, scored by
cosine similarity to the query vector (plus optional recency blend).
Designed for queries like "what themes exist in my memory?":

```python
result = await store.search_reflections_only(
    "recurring ideas about learning",
    query_vector=embedding,
    top_k=5,
    current_cycle=42,   # optional recency blend
)
for thought_id, score in result.results:
    ref = await store.get_thought(thought_id)
    print(ref.content)  # JSON with member_ids + keywords
```

Key difference from `search_hybrid(include_reflections=True)`:
`search_reflections_only()` fetches **all** REFLECTIONs directly from
the store (no pagination gap) and scores them purely by cosine similarity
to the query.  It does not compete against regular thoughts for result slots.

# Hybrid Search

engrava's `search_hybrid()` combines up to **five** scoring signals
into a single ranked result list.

## Signal Model

| # | Signal | Weight key | Default | Source |
|---|--------|------------|---------|--------|
| 1 | **FTS5 keyword** | `default_fts_weight` | `0.30` | BM25 full-text score (min-max normalized) |
| 2 | **Vector similarity** | `default_vector_weight` | `0.55` | Cosine similarity from embedding search |
| 3 | **Recency** | `default_recency_weight` | `0.10` | Exponential decay based on `current_cycle` |
| 4 | **Priority** | `default_priority_weight` | `0.05` | Boost multiplier per priority level (P1–P4) |
| 5 | **Graph** | `default_graph_weight` | `0.00` | 1-hop-weighted neighbour boost (opt-in) |

Default weights sum to `1.0`.  When a signal is unavailable (e.g. no
`current_cycle` → recency skipped, no embeddings → vector skipped),
its weight is **redistributed proportionally** across active signals.

## Graceful Degradation

- FTS5 unavailable or empty query → FTS skipped.
- `query_vector` is `None` and no embedding provider → vector skipped.
- `current_cycle` is `None` → recency skipped.
- `priority_weight` is `0.0` → priority skipped.
- `graph_weight` is `0.0` → graph skipped, zero overhead.
- All signals disabled → fallback to `list_thoughts(LIMIT top_k)`.

## Keyword query syntax (FTS)

The keyword signal — and the `search_fts()` method and the MCP `search_keywords`
tool that expose it directly — runs your text against an SQLite FTS5 index. engrava
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
**not** become spurious `http:` / `12:` column filters). A genuinely malformed
full-text expression is logged and degraded to zero FTS hits, so the rest of a
hybrid search still returns results.

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

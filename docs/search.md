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

## Configuration Reference

See [configuration.md](configuration.md) for the full YAML reference
of `SearchConfig` fields.

## Querying reflections

> **v0.4.0**

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

### `reflection_boost` (default `SearchConfig.reflection_boost = 1.2`)

When REFLECTIONs are included, their final score is multiplied by this
factor.  The default `1.2` gives a modest upranking so high-level
abstractions surface for broad queries without dominating narrow ones.

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
  reflection_boost: 1.2   # applies when reflection_boost not overridden per-call
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

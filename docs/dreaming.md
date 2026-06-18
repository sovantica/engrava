# Dreaming — Memory Consolidation

Engrava's **dreaming** extension provides periodic memory consolidation:
it evaluates stored thoughts, scores them against configurable signals,
and promotes the most important ones by setting their priority to **P1**.

Dreaming runs **outside** the normal CRUD path — the consumer decides
when to invoke `run_consolidation()` (after N cycles, in a cron job,
or manually).

## How memory consolidation works (the dreaming loop)

Think of a single memory's journey through an agent's lifetime. The first two
steps — ingest and confirmation — happen on the **normal write path** as you use
the store. The consolidation part is **manual**: when you call
`run_consolidation()`, that one call runs promotion → edge creation → reflection
clustering/creation → an orphan sweep, in order.

```
  ingest        you create an OBSERVATION ("user prefers email")   (write path)
    │
    ▼
  confirm       the same fact is re-encountered over time, so its  (write path)
    │           confirmation_count grows (e.g. via deduplicate=True)
    │
    ▼  run_consolidation(current_cycle=N)   ── manual ──
    │
  ┌─┴───────────────────────────────────────────────────────┐
  │ 1. promote   thoughts that pass the gates and clear       │
  │              promote_threshold are raised to priority P1  │
  │ 2. link      a promoted thought *may* gain ASSOCIATED      │
  │              edges to similar neighbours (when enabled)    │
  │ 3. reflect   related thoughts *may* be clustered into      │
  │              REFLECTION meta-thoughts (when enabled)       │
  │ 4. sweep     stale REFLECTIONs whose sources left the      │
  │              active set are retired                         │
  └─┬───────────────────────────────────────────────────────┘
    │
    ▼
  improved      later searches rank the P1 memory higher (priority
  retrieval     signal), follow any new edges (graph signal), and can
                surface a REFLECTION instead of many raw thoughts
```

Walking the journey:

1. **Ingest.** You store memories as thoughts (typically `OBSERVATION`s) on the
   normal write path. Dreaming does nothing yet.
2. **Confirm.** As the same knowledge recurs, its `confirmation_count` rises —
   automatically when you write with `deduplicate=True` (identical content
   collapses and bumps the count), or via your own logic. This is *evidence the
   memory matters*, and it feeds dreaming's confirmation signal. (Distinct from
   `confidence`, the static belief-strength you set — see
   [Core Concepts](concepts.md#reliability-confidence-vs-confirmation_count).)
3. **Promote.** When you run consolidation, each candidate must first pass the
   [gates](#gates) (e.g. old enough, enough confirmations) and then score above
   `promote_threshold` across the weighted [signals](#signals). Survivors are
   promoted to **P1**. (Both bars matter: a thought that passes the gates but
   scores low is *not* promoted — see
   [Troubleshooting](troubleshooting.md#dreaming-promotes-nothing-consolidation-is-inert).)
4. **Link.** A promoted thought *may* gain `ASSOCIATED` [edges](#edge-creation)
   to similar neighbours — when edge creation is enabled, the thought has a stored
   embedding, and qualifying neighbours (above `min_similarity`) are found. New
   edges persist the structure in the graph, idempotently (re-runs don't
   duplicate edges).
5. **Reflect.** Related thoughts *may* be clustered and summarised into
   [`REFLECTION`](#reflections-meta-consolidation) meta-thoughts — a centroid
   embedding plus `CONSOLIDATED_FROM` edges back to the members — when reflections
   are enabled and eligible clusters pass the clustering/quality gates. This turns
   a pile of observations into fewer, higher-level memories. (A REFLECTION whose
   source cluster later leaves the active set is automatically retired so a stale
   summary can't resurface.)
6. **Improved retrieval.** All of this changes future
   [hybrid search](search.md): the P1 memory ranks higher via the priority
   signal, any new edges feed the opt-in graph signal, and reflections let one
   high-level memory stand in for many raw ones.

The rest of this page is the knob-by-knob reference for each phase.

## Quick start

```python
from engrava.config import DreamingConfig, DreamingGates
from engrava.extensions.dreaming import DreamingExtension

config = DreamingConfig(
    enabled=True,
    promote_threshold=0.55,
    gates=DreamingGates(
        allow_zero_confirmation=True,
        min_age_cycles=1,
    ),
)
ext = DreamingExtension(config=config)

# After ingesting thoughts into `store`:
result = await ext.run_consolidation(store, current_cycle=1)
print(f"Promoted {result.promoted_count} thoughts")
```

## Gates

Before a thought is scored against the promotion threshold, it must
pass all active **gates**.  Gates are cheap boolean checks that filter
out clearly ineligible candidates.

| Gate | Field | Default | Description |
|------|-------|---------|-------------|
| Minimum age | `min_age_cycles` | `1` | `current_cycle - created_cycle` must be ≥ this value.  Prevents promoting thoughts that were just created in the same cycle. |
| Confirmation count | `min_confirmations` | `2` | `confirmation_count` must be ≥ this value.  **Bypassed** when `allow_zero_confirmation` is `True` (the default). |
| Max promoted | `max_promoted_per_run` | `20` | Cap on the number of promotions per consolidation run. |

### `allow_zero_confirmation`

When `True` (default), the confirmation gate is skipped entirely.
This is essential for **single-write batch-ingest** scenarios where
thoughts are stored once and never confirmed — without this flag,
no thought would ever pass the confirmation gate and dreaming would
be effectively dead.

Set to `False` only when your application explicitly tracks
confirmations and you want to require at least `min_confirmations`
experience-based validations before a thought is eligible for
promotion.

## Signals

Signals compute a score in `[0.0, 1.0]` for each candidate thought.
The weighted sum of all signal scores is compared against
`promote_threshold`.

| Signal | Weight | Description |
|--------|--------|-------------|
| `recency` | 0.25 | Exponential decay based on `updated_cycle` age. |
| `staleness` | 0.20 | Activity span (`updated_cycle - created_cycle`). |
| `confirmation` | 0.20 | Ratio of `confirmation_count` to max (5). |
| `confidence` | 0.15 | Thought's `confidence` field (default 0.5). |
| `frequency` | 0.20 | Ratio of `access_count` to max (10). |

Custom signals can be provided via `DreamingSignalProtocol`:

```python
class MySignal:
    def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
        return 0.42


ext = DreamingExtension(
    config=config,
    custom_signals={"my_signal": MySignal()},
)
```

## Priority signal in search

After dreaming promotes thoughts to **P1**, the hybrid search
`search_hybrid()` can use `priority` as a 4th scoring signal alongside
FTS5, vector similarity, and recency.

The priority signal maps each thought's `Priority` enum to a boost
multiplier:

| Priority | Default boost |
|----------|---------------|
| P1 | 1.0 |
| P2 | 0.6 |
| P3 | 0.3 |
| P4 | 0.0 |

The default priority weight is `0.05` (5% of the total score).
Configure it via `SearchConfig`:

```yaml
search:
  default_priority_weight: 0.05
  priority_boost_p1: 1.0
  priority_boost_p2: 0.6
  priority_boost_p3: 0.3
  priority_boost_p4: 0.0
```

To disable the priority signal entirely, set `default_priority_weight: 0.0`.

## Edge creation

When dreaming promotes a thought to **P1**, it can also create
**ASSOCIATED** edges connecting the promoted thought to its nearest
neighbours by embedding similarity.  This persists the dream's
structural knowledge in the graph so it survives application restarts.

Edges are created with `source=KnowledgeSource.DREAMING` for attribution.

### Configuration

```yaml
extensions:
  dreaming:
    edges:
      enabled: true          # create edges on promotion (default: true)
      top_k: 1               # max neighbours per promoted thought
      min_similarity: 0.7    # cosine threshold for edge creation
      edge_weight_factor: 0.5  # edge.weight = factor × similarity
```

### Idempotence

Re-running `run_consolidation()` on the same data does **not** create
duplicate edges.  Before creating an edge, the extension checks whether
the promoted thought already has any edge connecting it to the
candidate neighbour (regardless of type).

### Edge weight formula

```
edge.weight = edge_weight_factor × cosine_similarity
```

With the default `edge_weight_factor=0.5` and `min_similarity=0.7`,
practical edge weights range from `0.35` to `0.50`.

## Graph-aware search

After dream-created edges exist in the graph, `search_hybrid()` can
use them as a 5th scoring signal (**graph signal**).  The signal uses
**1-hop-weighted neighbour boost**: if a candidate thought's graph
neighbours also match the query, the candidate receives a boost
proportional to the neighbour's score and the connecting edge weight.

### Algorithm

```
For each candidate C in the fusion pool:
  neighbours = get_edges(C, direction="BOTH", limit=max_neighbors)
  For each (edge, neighbour):
    neighbour_base = max(fts[neighbour], vector[neighbour])
    boost[C] += edge.weight × neighbour_base × graph_edge_decay
final_score[C] += graph_weight × boost[C]
```

### Configuration

```yaml
search:
  default_graph_weight: 0.0     # opt-in (0.0 = disabled)
  graph_edge_decay: 0.5         # decay factor for 1-hop distance
  max_neighbors_per_candidate: 5  # safety cap
```

Per-query override:

```python
result = await store.search_hybrid(
    "python async",
    graph_weight=0.1,
    graph_edge_decay=0.3,
)
```

The graph signal is **opt-in** in v0.3.0 (`default_graph_weight=0.0`).
When the weight is `0.0`, no graph queries are made and there is zero
performance impact.

## Configuration reference

See [configuration.md](configuration.md) for the full YAML reference
of `DreamingGates`, `EdgeCreationConfig`, and `SearchConfig` fields.

## Reflections (meta-consolidation)

`run_consolidation()` runs a **third phase** after promotion and edge
creation: it clusters semantically related thoughts and creates
**`ThoughtType.REFLECTION`** meta-thoughts that aggregate each cluster.

### What is a REFLECTION?

A REFLECTION is a first-class `ThoughtRecord` that represents a
higher-order abstraction over a cluster of related thoughts:

| Field | Value |
|-------|-------|
| `thought_type` | `REFLECTION` |
| `embedding` | Centroid of member embeddings (mean, L2-normalised) |
| `content` | JSON: `{"member_ids": [...], "keywords": [...], "cluster_hash": "..."}` |
| `priority` | Maximum priority of any cluster member |
| `source` | `"dreaming:<cluster_hash>"` (hex-16) |
| `source_type` | `KnowledgeSource.DREAMING` |
| Edges | `CONSOLIDATED_FROM` → every cluster member |

**No LLM is involved** — content is purely structural (keyword frequency
counts from member text, centroid from member vectors). LLM-generated
prose summaries belong in downstream extension hooks, not in the
core graph layer.

> **Navigating the lineage.** The `CONSOLIDATED_FROM` edges are queryable
> through dedicated store helpers — `consolidated_member_ids(reflection_id)`,
> `consolidated_source_statuses(reflection_id)`, and the reverse
> `reflections_consolidated_from(source_id)`. Use them to walk from a REFLECTION
> to its sources and back (e.g. for provenance views or orphan detection)
> instead of querying the edge table directly. See
> [REFLECTION lineage](api-reference.md#reflection-lineage) in the API reference.

### How clustering works

Two algorithms are available via `DreamingGates.cluster_algorithm`:

**`"lpa"` (default) — Label Propagation Algorithm**
- Operates over the ASSOCIATED dream-edge graph built in phase 2.
- Deterministic via seeded PRNG (`seed=42` by default).
- O(E × iterations), no external dependencies.
- Works when the graph is dense enough to form communities.

**`"agglomerative"` — cosine-similarity single-linkage**
- Operates over **all active ACTIVE thoughts**, independent of graph edges.
- Intended for sparse-graph / first-run scenarios where LPA finds no clusters.
- Nodes whose cosine similarity ≥ `cluster_similarity_threshold` are merged
  via Union-Find.
- Use when you want clustering before dreams have built up a graph.

### Idempotence

Before creating a REFLECTION, the extension derives a 16-hex content-hash
from the sorted member IDs and checks whether any REFLECTION with
`source = "dreaming:<hash>"` already exists (exact SQL index lookup,
O(1), scales to any store size).  If found, the cluster is skipped.

Re-running `run_consolidation()` on unchanged data creates zero
duplicate REFLECTIONs.

### Configuration

```yaml
extensions:
  dreaming:
    gates:
      min_cluster_size: 3             # min members for a reflection to be created
      cluster_similarity_threshold: 0.7  # cosine threshold (agglomerative only)
      cluster_algorithm: lpa          # "lpa" or "agglomerative"
      enable_reflections: true        # set to false to skip phase 3 entirely
```

### `ConsolidationResult` fields

```python
result = await ext.run_consolidation(store, current_cycle=42)
print(result.promoted_count)  # thoughts promoted to P1
print(result.edges_created)  # ASSOCIATED edges created
print(result.reflections_created)  # new REFLECTION thoughts created
```

### Querying reflections

See [search.md](search.md) — "Querying reflections" section.

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
6. **How it shows up in retrieval.** All of this changes future
   [hybrid search](search.md) *mechanically*: the P1 memory ranks higher via the
   priority signal, any new edges feed the opt-in graph signal, and reflections
   let one high-level memory stand in for many raw ones. These are deterministic
   ranking effects — Engrava makes **no claim that dreaming improves retrieval
   accuracy** on any benchmark (see [Known Limitations](known-limitations.md)).

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

### From YAML config

If you enable dreaming in `engrava.yaml`
(`extensions.dreaming.enabled: true`) and build the store with
`SqliteEngravaCore.from_config(...)`, you do **not** need to construct a
`DreamingExtension` by hand — call `consolidate()` on the store directly:

```python
from engrava import SqliteEngravaCore

async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    result = await store.consolidate(current_cycle=1)
    print(f"Promoted {result.promoted_count} thoughts")
```

`store.consolidate(current_cycle=...)` first flushes any pending
access events (so the [`frequency`](#access-tracking-the-frequency-substrate)
signal sees the latest access counts this cycle), then runs the configured
extension. It raises `RuntimeError` if dreaming is not enabled/wired on the
store (built manually, or `dreaming.enabled` is `false`) — there is no
extension to run.

`schedule_every_n_cycles` does not start a background scheduler. Applications
with a cycle loop can call `DreamingExtension.run_if_due(store, current_cycle)`;
it runs only on positive cycle multiples while Dreaming is enabled and returns
`None` otherwise. `is_due(current_cycle)` exposes the same decision without a
write. Explicit `store.consolidate()` and `run_consolidation()` calls remain
unconditional.

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

## Eligibility filters and corpus caps

After the age/confirmation gates, promotion and REFLECTION members also pass
metadata-aware eligibility filters. They read caller-provided metadata and are
not identity or authorization controls.

| Field | Default | Effect |
|---|---|---|
| `eligible_perspectives` | `null` | Optional allow-list for `metadata.perspective`: `percept`, `utterance`, `thought`. |
| `self_filter_mode` | `any` | `self_only` / `external_only` use strict boolean `metadata.source.is_self`. |
| `min_source_confidence` | `low` | Minimum `metadata.source.confidence` in `low < medium < high` order. |
| `excluded_content_types` | `["code"]` | Reject matching declared `metadata.content_type` values. |
| `eligible_content_types` | `null` | Optional allow-list for declared content types. |

Missing perspective, `is_self`, or content-type annotations remain eligible for
backward compatibility. Once a non-empty `metadata` mapping exists, missing or
malformed source confidence is treated as `low`, so a higher configured
threshold rejects it. A record with absent or empty metadata bypasses every
metadata-driven filter, including the confidence threshold.

The remaining caps operate at different stages:

- `candidates_limit` (default `200`) bounds the ACTIVE promotion query and each
  type query used by agglomerative clustering;
- `max_promoted_per_run` (default `20`) limits writes in one promotion phase;
- `max_p1_fraction` (default `0.05`) limits the corpus-wide P1 population;
- `min_cluster_size` / `max_cluster_size` (defaults `3` / `200`) reject clusters
  that are too small or too broad after eligibility filtering;
- `clustering_min_new_candidates` (default `50`) skips repeat clustering when
  the eligible ACTIVE population has not grown enough. It does not skip signal
  scoring, promotion, or edge creation.

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
| `action_outcome` | 0.15 | Thought's `action_outcome_score` — the mean outcome value over its terminal linked actions (`None` ⇒ contributes `0.0`). |

A signal whose data source is flat across the whole candidate pool carries no
ranking information, so it is dropped and its weight is redistributed over the
active signals. `action_outcome` is therefore **inactive** — and its weight
falls out of the denominator — in any store where no candidate has a recorded
action outcome, so it never perturbs consolidation until actions are used. The
default weights sum to more than 1.0 for this reason: they are relative
priorities renormalised over the active set, not a probability distribution.

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

### Access tracking (the `frequency` substrate)

The `frequency` signal scores a thought by its `access_count`. For that to be
meaningful, reads have to be counted — otherwise `access_count` stays at 0 and
`frequency` contributes nothing.

When you build the store via `from_config` with dreaming enabled, retrieval
paths record access automatically, controlled by
`extensions.dreaming.access_tracking_enabled` (default **`true`**). To keep the
read path fast, access events are buffered and flushed at the cycle boundary
rather than written on every read; `store.consolidate()` flushes the buffer
before scoring so the current cycle sees up-to-date counts. Access tracking is
deliberately **not** journaled — the counts are regenerable telemetry, not part
of the tamper-evident chain (see [Audit Trail](audit-trail.md)).

Set `access_tracking_enabled: false` to leave `access_count` untouched (the
`frequency` signal then stays inactive and its weight is redistributed over the
remaining active signals, exactly as with any other inactive signal).

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

## Complete YAML surface

This example names every YAML-configurable `DreamingConfig`, `DreamingGates`,
and `EdgeCreationConfig` field with its default. It is a reference, not a
recommended production profile:

```yaml
extensions:
  dreaming:
    enabled: false
    schedule_every_n_cycles: 100   # used by is_due() / run_if_due()
    promote_threshold: 0.7
    candidates_limit: 200
    signals:
      recency: 0.25
      staleness: 0.20
      confirmation: 0.20
      confidence: 0.15
      frequency: 0.20
      action_outcome: 0.15

    clustering_backend: numpy      # numpy or python
    top_keyphrases_count: 3
    top_member_excerpts_count: 5
    member_excerpt_max_chars: 150
    max_p1_fraction: 0.05
    promote_targets: OBS_ONLY      # OBS_ONLY, REFL_ONLY, or ALL
    reflection_default_priority: P2

    eligible_perspectives: null    # optional list: percept, utterance, thought
    self_filter_mode: any          # any, self_only, external_only
    min_source_confidence: low     # low, medium, high
    excluded_content_types: [code]
    eligible_content_types: null

    boilerplate_threshold: 0.30
    boilerplate_min_corpus_size: 5
    boilerplate_min_keyphrases_per_refl: 1
    access_tracking_enabled: true

    gates:
      min_confirmations: 2
      min_age_cycles: 1
      max_promoted_per_run: 20
      allow_zero_confirmation: true

      min_cluster_size: 3
      cluster_similarity_threshold: 0.7
      cluster_algorithm: lpa       # lpa or agglomerative
      enable_reflections: true
      cold_start_clustering: false
      cluster_allowed_types: [OBSERVATION]
      clustering_min_new_candidates: 50
      max_cluster_size: 200        # null disables the upper bound

      cluster_quality_gating_enabled: true
      cluster_quality_persona_threshold: 0.75
      cluster_quality_cohesion_threshold: 0.40
      cluster_quality_external_homogeneity_threshold: 0.95
      cluster_quality_ne_consistency_threshold: 0.60
      cluster_quality_require_meaningful_keyphrases: true

    edges:
      enabled: true
      top_k: 1
      min_similarity: 0.7
      edge_weight_factor: 0.5
```

A partial `signals` mapping merges onto the six defaults. Unknown signal names
require a matching `custom_signals` implementation when constructing
`DreamingExtension`; the YAML-only `from_config()` path has no custom-signal
registry and rejects unknown names when it constructs the extension.

As described in [Configuration](configuration.md#loading-configuration), unknown
keys in the Dreaming, gates, and edge mappings raise `ConfigError`. See
[Configuration](configuration.md#dreaming) for field-by-field types and
validation ranges.

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
| `content` | Structural JSON schema v2 (fields below) |
| `priority` | `reflection_default_priority` (`P2` by default) |
| `source` | `"dreaming:<cluster_hash>"` (hex-16) |
| `source_type` | `KnowledgeSource.DREAMING` |
| Edges | `CONSOLIDATED_FROM` → every cluster member |

The v2 content payload preserves the legacy `member_ids`, `keywords`, and
`cluster_hash` fields and adds:

| Content field | Meaning |
|---|---|
| `type` / `version` | `"reflection"` / `2` schema discriminator |
| `member_count` | Number of eligible source members |
| `cluster_algorithm` | Algorithm used for the cluster |
| `created_at` | ISO-8601 payload build time |
| `top_keyphrases` | Bounded TF-IDF phrase/score objects after cross-cluster boilerplate filtering |
| `member_excerpts` | Bounded excerpts from priority/recency-ordered members |
| `temporal_span` | Member creation-time bounds and span |
| `named_entities` | Sorted unique structurally extracted entities |

**No LLM is involved**. Content, essence, centroid, metadata enrichments, and
valid-time extent are derived structurally from member records and vectors; the
payload's `created_at` records the current UTC build time. LLM-generated prose
summaries belong in downstream extensions, not in the core graph layer.

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
- Operates over bounded ACTIVE candidates whose thought types are listed in
  `cluster_allowed_types` (default: `OBSERVATION`), independent of graph edges.
- Intended for sparse-graph / first-run scenarios where LPA finds no clusters.
- Nodes whose cosine similarity ≥ `cluster_similarity_threshold` are merged
  via Union-Find.
- Use when you want clustering before dreams have built up a graph.

`clustering_backend` selects the similarity implementation for the
agglomerative path. `numpy` (default) uses vectorised float32 matrix operations
and chunks large candidate sets to bound peak matrix memory. `python` uses the
legacy pure-Python O(n²) loop and is intended only as a debugging escape hatch.
It does not change the LPA implementation.

**Cold-start fallback (`cold_start_clustering`, default `false`)**

The `"lpa"` path reads communities only from the ASSOCIATED dream-edge
graph, and those edges appear only after promotions create them. On a
fresh or sparse graph the edge set is empty, so LPA finds nothing and no
REFLECTIONs are produced no matter how many eligible OBSERVATIONs exist.

Set `cold_start_clustering: true` to opt into a fallback: when the LPA
path finds the edge graph empty, it falls back **within the same cycle**
to the cosine-similarity agglomerative clustering described above, run
over the eligible ACTIVE candidate pool. This lets dreaming form
clusters (and REFLECTIONs) before any dream edges exist.

- **Default is `false`** — the shipped `"lpa"` behaviour is unchanged, and
  the fallback never alters cluster output when edges are present.
- The fallback reuses the exact agglomerative path, so its clusters flow
  through the same `min_cluster_size` / `max_cluster_size` size gates and
  the content-quality gates — nothing bypasses them.
- The candidate pool is bounded by `candidates_limit`, so a large active
  set does not make the fallback unbounded.

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
      cold_start_clustering: false    # opt-in: LPA falls back to agglomerative
                                      #   clustering when the edge graph is empty
      cluster_allowed_types: [OBSERVATION]
      clustering_min_new_candidates: 50
      max_cluster_size: 200
```

### Cluster-quality gates

After clustering, metadata eligibility is applied again to the resolved
members. The cluster must still contain at least `min_cluster_size` eligible
members and must not exceed `max_cluster_size`.

With `cluster_quality_gating_enabled: true` (default), a cluster is rejected on
the first failed content-quality check:

- duplicate member content;
- persona-only share at or above `cluster_quality_persona_threshold` (`0.75`);
- contradictions flagged by the lightweight English token-pair heuristic;
- mean pairwise cosine below `cluster_quality_cohesion_threshold` (`0.40`);
- external-source fraction below
  `cluster_quality_external_homogeneity_threshold` (`0.95`);
- named-entity overlap below `cluster_quality_ne_consistency_threshold`
  (`0.60`); or
- no meaningful post-boilerplate keyphrase when
  `cluster_quality_require_meaningful_keyphrases` is true.

Disabling the master switch bypasses these content-quality checks, but not
cluster size, metadata eligibility, embedding availability, or idempotence.
The cross-cluster boilerplate filter is controlled separately by
`boilerplate_threshold`, `boilerplate_min_corpus_size`, and
`boilerplate_min_keyphrases_per_refl`.

### `ConsolidationResult` fields

| Field | Meaning |
|---|---|
| `candidates_evaluated` | Number of ACTIVE candidates in the bounded promotion pool |
| `promoted_count` / `promoted_ids` | Promotions written and their thought IDs |
| `skipped_gate_count` | Candidates rejected by age/confirmation gates |
| `scores` | Computed score for every promotion candidate |
| `edges_created` | New dream-created ASSOCIATED edges |
| `reflections_created` | New REFLECTION thoughts |
| `promotion_capped` | Whether the corpus-wide P1 fraction prevented a promotion |
| `p1_fraction_after` | P1 share after the run |
| `orphans_retired` | ACTIVE REFLECTIONs archived because all sources left the active set |
| `active_signal_weights` | Effective weights after flat-signal redistribution |
| `flat_signals` | Configured signal names that carried no ranking information this run |

### Querying reflections

See [search.md](search.md) — "Querying reflections" section.

# Architecture

engrava is a **thought-graph database** — a Python library for
storing, linking, searching, and evolving ideas.  It is SQLite-first
and designed to be embedded by larger cognitive systems.

## Layer Model

Imports flow **downward only**:

```
┌──────────────────────────────────────────────────┐
│  CLI / Consumer apps, scripts, benchmarks        │
├──────────────────────────────────────────────────┤
│  Extensions / Embeddings / MindQL                │
│  (dreaming, hooks, providers, query language)    │
├──────────────────────────────────────────────────┤
│  Infrastructure                                  │
│  SqliteEngravaCore, schema, migrations           │
├──────────────────────────────────────────────────┤
│  Domain                                          │
│  models, enums, protocols, exceptions            │
└──────────────────────────────────────────────────┘
```

- **Domain** (`src/engrava/domain/`) — stdlib + Pydantic only.
  Frozen models, `@runtime_checkable` Protocols, zero infra imports.
- **Infrastructure** (`src/engrava/infrastructure/`) — SQLite
  implementation of domain protocols.
- **Extensions** (`src/engrava/extensions/`) — optional capabilities
  (dreaming, hooks) that depend on domain + infrastructure.
- **MindQL** (`src/engrava/mindql/`) — read-only query language.
- **Embeddings** (`src/engrava/embeddings/`) — pluggable embedding
  providers.
- **CLI** (`src/engrava/cli/`) — Click-based command-line interface.

## Core Components

### SqliteEngravaCore

The primary store implementation.  Provides:

- Thought CRUD (create, read, update, list, search)
- Edge CRUD (create, read, update, delete, traverse)
- Embedding storage and vector similarity search
- Full-text search (FTS5)
- Hybrid search (5-signal fusion — see below)
- Schema management and migrations

### Hybrid Search (5-Signal Fusion)

`search_hybrid()` fuses five ranking signals:

```
final_score = w₁·FTS + w₂·Vector + w₃·Recency + w₄·Priority + w₅·Graph
```

| Signal | Default Weight | Description |
|--------|---------------|-------------|
| FTS5 | 0.30 | BM25 keyword match |
| Vector | 0.55 | Cosine similarity |
| Recency | 0.10 | Exponential time decay |
| Priority | 0.05 | P1–P4 boost |
| Graph | 0.00 (opt-in) | 1-hop neighbour boost |

Disabled signals have their weight redistributed proportionally.
See [search.md](../docs/search.md) for details.

### Dreaming Extension

Periodic memory consolidation that:

1. **Scores** active thoughts via configurable signals.
2. **Promotes** qualifying thoughts to P1 priority.
3. **Creates edges** (`ASSOCIATED`, `source=DREAMING`) between
   promoted thoughts and their nearest neighbours.
4. **Clusters + reflects** — groups semantically related thoughts
   and creates `ThoughtType.REFLECTION` meta-thoughts with centroid
   embeddings and `CONSOLIDATED_FROM` edges.

Dreaming is a **graph mutator and abstraction builder** — each
consolidation run can grow the thought graph with dream-discovered
connections *and* create higher-order REFLECTION thoughts that
aggregate clusters.  Both feed into hybrid search, closing the
dream → structure → retrieval loop.

See [dreaming.md](../docs/dreaming.md) for details.

### Extension System

New behaviors plug in via `EngravaHooksProtocol` and
`mindql_extension_registry`.  Application-level logic (planners,
reasoners) belongs in consumers, not in engrava core.

## Data Flow

```
   Ingest                    Dreaming                   Search
   ─────                    ────────                   ──────
   create_thought() ──┐     run_consolidation()        search_hybrid()
   store_embedding()  │      │                          │
                      ▼      ▼                          ▼
              ┌───────────────────┐            ┌─────────────────┐
              │    SQLite DB      │            │  5-signal fusion │
              │  thoughts table   │◄───────────│  FTS + Vec +    │
              │  edge table       │            │  Rec + Pri +    │
              │  embedding table  │            │  Graph          │
              └───────────────────┘            └─────────────────┘
                      ▲
                      │
              Dream: ASSOCIATED edges on promotion
              Dream: REFLECTION thoughts from clusters
```

## Upgrade Path (v0.2 → v0.3)

- **Additive only** — no breaking changes.
- New `SearchConfig` fields (`default_graph_weight`,
  `graph_edge_decay`, `max_neighbors_per_candidate`) default to
  backward-compatible values (`0.0`, `0.5`, `5`).
- New `DreamingConfig.edges` block defaults to `enabled=True`.
- Existing databases receive dream-created edges on the next
  consolidation run; no retroactive edge creation for historical data.

## Upgrade Path (v0.3 → v0.4)

- **Additive only** — no breaking changes.
- `run_consolidation()` gains a third phase (clustering + REFLECTION
  creation).  Opt-out via `DreamingGates.enable_reflections = False`.
- `ConsolidationResult.reflections_created` is a new field (default `0`).
- New `DreamingGates` fields: `min_cluster_size`, `cluster_similarity_threshold`,
  `cluster_algorithm`, `enable_reflections` — all backward-compatible defaults.
- New `SearchConfig.reflection_boost` field (default `1.2`).
- New `search_hybrid()` params: `include_reflections` (default `True`),
  `reflection_boost` (default `None` → uses config).
- New `search_reflections_only()` helper method.
- New `thought_exists_by_source()` utility method.
- Existing databases: no migration needed.  REFLECTIONs are created on
  the next consolidation run; no retroactive clustering.

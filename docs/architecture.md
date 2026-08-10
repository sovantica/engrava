# Architecture

engrava is **the memory database for AI agents** — a Python library for
storing, linking, searching, and evolving ideas.  It is SQLite-first
and designed to be embedded by larger cognitive systems.

> [Upgrade Guide](upgrade.md) is the authoritative source for schema versions,
> compatibility, and operator sequencing. The upgrade sections here summarize
> architectural impact only.

## Layer Model

Imports flow **downward only**:

```
┌──────────────────────────────────────────────────┐
│  CLI / Consumer apps, scripts, MCP server, …     │
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

The [Model Context Protocol](https://modelcontextprotocol.io) server ships as a
separate package, [`engrava-mcp`](https://github.com/sovantica/engrava-mcp). Like
the CLI it is a **top-layer API *consumer***, not part of engrava core: it wraps
the public async API over stdio so MCP clients (Claude Desktop, Cursor, …) can
use a store, and it depends on `engrava` rather than living inside it.

## Core Components

### SqliteEngravaCore

The primary store implementation.  Provides:

- Thought CRUD (create, read, update, list, search)
- Edge CRUD (create, read, update, delete, traverse)
- Embedding storage and vector similarity search
- Full-text search (FTS5)
- Hybrid search (5-signal fusion — see below)
- **Derived records** — an optional producer capability can emit bounded child
  thoughts after a source is durable; core owns child identity, persistence,
  and optional `DERIVED_FROM` provenance edges. Existing stores can be
  backfilled explicitly with `derive_existing()`.
- **Memory Hygiene** — an opt-in, no-LLM forgetting pass that reversibly
  archives cold records and can separately garbage-collect hygiene archives
  after both configured restore windows.
- **Bi-temporal valid time** — optional `valid_from` / `valid_until` bounds on
  thoughts *and* edges (a second time axis: *when a fact is true*, distinct from
  when it was recorded), queried via the four valid-time MindQL predicates, with
  `invalidate_thought` / `invalidate_edge` to close an interval without deleting.
  See [The Bi-temporal Model](../docs/bitemporal.md).
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

The extension surface is capability-based:

- `EngravaHooksProtocol` transforms stored/retrieved thoughts and supplies the
  active Memory Hygiene `decay_function`. `score_function` and
  `mindql_extension_registry()` remain reserved and are not called by core.
- `DerivedRecordProducerProtocol` is a separate optional capability for
  one-to-many derivation. The producer receives no store handle; persistence
  remains core-owned and bounded by `DeriveGates`.
- `ExtensionManifest` packages a hooks class, MindQL commands, and ordered SQL
  migrations. Manifests are loaded explicitly or through opt-in entry-point
  discovery; core does not consult `mindql_extension_registry()`.
- Embedding providers implement their own protocol and remain independent of
  lifecycle hooks and manifests.

Application-level logic such as planning and reasoning belongs in consumers,
not in engrava core. See [Extensions](extensions.md) and
[Extension hooks](extension-hooks.md).

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

With automatic derivation enabled, a committed source can produce ordinary
child thoughts after its own durable write; failures never make the source
disappear. Memory Hygiene runs as a separate maintenance path: it archives by
updating lifecycle state and records its reason in the journal, while optional
GC removes only rows carrying hygiene's own archive markers.

## Upgrade Path (v0.2 → v0.3)

- **Additive only** — no breaking changes.
- New `SearchConfig` fields (`default_graph_weight`,
  `graph_edge_decay`, `max_neighbors_per_candidate`) default to
  backward-compatible values (`0.0`, `0.5`, `5`).
- New `DreamingConfig.edges` block defaults to `enabled=True`.
- Existing databases receive dream-created edges on the next
  consolidation run; no retroactive edge creation for historical data.
- **REFLECTION clustering** (the third dreaming phase) also shipped in 0.3.0:
  `run_consolidation()` clusters semantically related thoughts and creates
  `ThoughtType.REFLECTION` meta-thoughts. Opt-out via
  `DreamingGates.enable_reflections = False`. New fields —
  `ConsolidationResult.reflections_created` (default `0`); `DreamingGates`
  `min_cluster_size` / `cluster_similarity_threshold` / `cluster_algorithm` /
  `enable_reflections`; `SearchConfig.reflection_boost` (default `1.0`);
  `search_hybrid()` `include_reflections` (default `True`) and `reflection_boost`
  (default `None` → uses config); the `search_reflections_only()` and
  `thought_exists_by_source()` helpers — all with backward-compatible defaults.
  REFLECTIONs are created on the next consolidation run; no retroactive clustering.

## Upgrade Path (v0.3 → v0.4)

- **Additive only** — no breaking changes; existing code is unaffected and a query
  that uses no temporal predicate behaves exactly as before.
- **Bi-temporal model.** Thoughts and edges gain two optional, nullable ISO-8601
  fields, `valid_from` and `valid_until` (an open bound = ±∞). Four opt-in MindQL
  `WHERE` predicates query *valid time* — `valid_now`, `valid_at`, `valid_within`,
  `valid_between` — and `invalidate_thought` / `invalidate_edge` close an interval
  without deleting. REFLECTIONs inherit their members' valid-time extent. The schema
  migration is **additive** (`user_version 12 → 14`, two steps), zero data loss; a
  legacy row keeps open (`NULL`) bounds and still matches point-in-time queries.
  See [The Bi-temporal Model](../docs/bitemporal.md).
- **MCP server.** A Model Context Protocol server over stdio is available as a
  separate package, [`engrava-mcp`](https://github.com/sovantica/engrava-mcp)
  (`uvx engrava-mcp`). It is a pure API *consumer* — plain `pip install engrava`
  stays dependency-light. (In 0.4 this shipped as an in-tree `engrava[mcp]` extra;
  it moved to its own package in 0.5 — see the Upgrade Guide.)
- **`execute_mindql`** — a store-level convenience that runs a parsed `MindQLQuery`
  against the store's own connection.
- Existing databases: the valid-time columns and indexes are added automatically on
  first open; no manual migration step.

## Upgrade Path (v0.4 → v0.5)

- The core schema advances from `user_version 14` to `18` through four additive
  migrations: the composite inbound edge index (15), action-outcome aggregate
  and lookup index (16), optional provenance plus session/actor indexes (17),
  and the Memory Hygiene `pinned` / `archived_at_cycle` fields (18).
- Memory Hygiene is default-off. Merely opening an upgraded database adds the
  columns but does not archive or delete records.
- The MCP server moved out of the library distribution into the standalone
  [`engrava-mcp`](https://github.com/sovantica/engrava-mcp) package. This is a
  packaging break for users of the former `engrava[mcp]` extra, not a core API
  dependency.
- Existing databases migrate automatically during `ensure_schema()` / factory
  open. Back up the database and quiesce other writers before the first open;
  see the [Upgrade Guide](upgrade.md).

## Upgrade Path (v0.5 → v0.6)

- The core schema advances from `user_version 18` to `20`: version 19 adds the
  generic `edge.metadata_json` JSON carrier, and version 20 adds
  `thought.archived_at` for Memory Hygiene's wall-clock restore window. Both are
  additive; existing edge metadata defaults to `{}`, and existing thoughts have
  no archive instant.
- Ranked retrieval now excludes `ARCHIVED` thoughts by default across FTS,
  vector, hybrid, and `recall`. Pass `include_archived=True` for an archive
  query or call `restore_thought()` to return a row to `ACTIVE`.
- Recency has two mutually exclusive references: cognitive-cycle
  `current_cycle` and caller-supplied transaction-time `recency_now`. A
  runtime-only `cycle_provider` can supply omitted cognitive cycles; it is never
  serialized to YAML and yields to an explicit `recency_now`.
- The derived-records seam adds `DerivedRecordProducerProtocol`, `DeriveGates`,
  `DeriveResult`, explicit `derive_existing()` backfill, and the deterministic
  `StructuralSplitProducer` with paragraph and fixed-window modes. It reuses
  ordinary thought/edge storage and requires no new schema table.
- Query vectors with the wrong dimension now raise the typed
  `VectorDimensionMismatchError`; degenerate vectors still degrade to an empty
  vector arm and increment `vector_arm_degradation_count`.
- Existing databases migrate automatically during first schema ensure. As with
  every schema-changing open, back up and quiesce other writers first; see the
  [Upgrade Guide](upgrade.md) for the operator sequence.

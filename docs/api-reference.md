# API Reference

Complete reference for engrava's public API.

## Core Store

### `SqliteEngravaCore`

The main persistence engine. All operations are async.

`SqliteEngravaCore` wraps an **already-open** `aiosqlite.Connection`. For a
config-driven, one-call setup that opens and owns the connection, use the
`from_config` factory instead.

```python
import aiosqlite
from engrava import SqliteEngravaCore

async with aiosqlite.connect(":memory:") as conn:
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()

# Or, config-driven (opens and owns the connection):
async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    ...  # schema already applied by from_config
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | `aiosqlite.Connection` | — | An already-open aiosqlite connection (set `row_factory = aiosqlite.Row`) |
| `hooks` | `EngravaHooksProtocol \| None` | `None` | Extension hooks (defaults to `DefaultEngravaHooks`) |
| `embedding_provider` | `EmbeddingProviderProtocol \| None` | `None` | Provider used when `auto_embed=True` |
| `auto_embed` | `bool` | `False` | Auto-embed thoughts on create/update |
| `require_embedding` | `bool` | `False` | When `True`, an auto-embed provider failure raises `EmbeddingGenerationError` instead of only logging a `WARNING` and re-raising the provider error. The thought is committed before embedding either way, so this governs how loudly a missing embedding is reported, not whether the row persists. No effect unless `auto_embed=True`. |
| `search_config` | `SearchConfig \| None` | `None` | Default hybrid-search weights, recency half-lives, graph expansion, reflection handling, and bounded candidate-pool settings |
| `journal_enabled` | `bool` | `False` | Record mutations in the hash-chain journal |
| `ttl_strategy` | `str` | `"archive"` | Expiry action: `"archive"` or `"delete"` |
| `ttl_check_every_n` | `int` | `0` | Automatic expiry-cleanup cadence in store operations; `0` disables automatic cleanup |
| `ttl_default_seconds` | `int \| None` | `None` | Default relative TTL for new thoughts |
| `metrics_config` | `MetricsConfig \| None` | `None` | Metrics enablement and latency-window settings |
| `manifests` | `Sequence[ExtensionManifest]` | `()` | Extension manifests whose schema migrations are applied by `ensure_schema()` |
| `access_tracking_enabled` | `bool` | `False` | Buffer retrieval access events for the dreaming frequency signal |
| `hygiene_policy` | `HygienePolicyConfig \| None` | `None` | Memory Hygiene policy; `None` means `run_hygiene()` is unavailable |
| `derive_gates` | `DeriveGates \| None` | `None` | Automatic derived-record production gates; defaults to disabled gates |
| `cycle_provider` | `CycleProvider \| None` | `None` | Runtime-only cognitive-cycle source used when a supported call omits `current_cycle` and no explicit `recency_now` selects transaction-time recency; explicit cycle values, including `0`, win |

> The `SqliteEngravaCore(db_path=...)` form does **not** exist — pass a
> connection, or use `await SqliteEngravaCore.from_config(path)`.

#### Factory

| Method | Returns | Description |
|--------|---------|-------------|
| `await SqliteEngravaCore.from_config(config_path, *, cycle_provider=None)` | `SqliteEngravaCore` | Build from a YAML config; opens + owns the connection and applies the schema. `cycle_provider` is a runtime object and is never read from YAML. Use the result as an async context manager. |

#### Schema

| Method | Description |
|--------|-------------|
| `await ensure_schema()` | Create tables if missing, run migrations |
| `await verify_journal()` | Verify the complete persisted hash chain and return `JournalIntegrityResult` |
| `await close()` | Close the database connection |

#### Thought CRUD

`create_thought` takes a single frozen `ThoughtRecord` object (build it, then
pass it) and returns the persisted record — it does **not** take field
keyword arguments and does **not** return a UUID string.

| Method | Returns | Description |
|--------|---------|-------------|
| `await remember(text, *, metadata=None, deduplicate=False)` | `ThoughtRecord` | Store a bare string as an `ACTIVE` P3 `NOTE` at cycle `0`; its first 200 characters become `essence`. Use an explicit `ThoughtRecord` when the caller owns cognitive cycles or needs other fields. |
| `await create_thought(thought, *, expires_after_seconds=None, deduplicate=False)` | `ThoughtRecord` | Persist a `ThoughtRecord`; returns the stored record. Raises `ValueError` if the ID already exists. |
| `await get_or_create(thought, *, expires_after_seconds=None)` | `tuple[ThoughtRecord, bool]` | Content-hash convenience over dedup: returns `(existing, False)` on a hash hit (confirmation bumped, like `deduplicate=True`) or `(new, True)` on a miss. The `bool` tells you whether it created, removing the check-then-create round trip. Does not alter a matched row's fields. |
| `await upsert_by_hash(thought, *, expires_after_seconds=None)` | `ThoughtRecord` | Update-on-match upsert: on a hash hit, overwrites the stored row's mutable fields (`essence`, `priority`, `metadata`, `visibility`, `lifecycle_status`, `source`, `confidence`, `source_type`, `thought_type`) from `thought` and returns it (**no** confirmation bump — distinct from `deduplicate=True`, which returns the stored record unchanged); on a miss, inserts. `content` is never rewritten (it is the hash key). |
| `await bulk_store(thoughts, *, deduplicate=False)` | `list[ThoughtRecord]` | Transactional batch insert: the whole list commits **once** and is all-or-nothing (any row error rolls the batch back). Order preserved. Under `auto_embed`, all thoughts are embedded in one batch provider call. `deduplicate` applies per row. |
| `await get_thought(thought_id)` | `ThoughtRecord \| None` | Retrieve by ID; `None` if not found |
| `await update_thought(thought_id, **changes)` | `ThoughtRecord` | Partial update: writes only the fields named in `changes` (plus the `updated_at` stamp), so columns another writer changed meanwhile are preserved; returns the row read back after the write. Raises `ThoughtNotFoundError` if the thought is missing when the call starts. Raises `StaleDataError` when the guarded write matches no row — a competing writer stamped a new `updated_cycle` (which rejects this update whatever field it touched) or deleted the row; an ordinary competing edit to the same field is overwritten silently instead — see [Concurrency](concurrency.md#optimistic-concurrency-and-staledataerror) |
| `await restore_thought(thought_id, *, current_cycle=None)` | `ThoughtRecord` | Un-archive: transition an `ARCHIVED` thought back to `ACTIVE`, clearing both hygiene markers (`archived_at_cycle` and `archived_at`) so an archive round-trips with no data loss. The reversible counterpart to the memory-hygiene / TTL / manual archive paths, journaled as an `UPDATE_THOUGHT`. Raises `ThoughtNotFoundError` if missing, `InvalidTransitionError` if the thought is not currently `ARCHIVED`, `StaleDataError` if the guarded write matches no row — a competing cycle stamp or a delete (see `update_thought` for what that guard does and does not catch). This is the **canonical** un-archive path; a raw `update_thought(lifecycle_status=...)` back to `ACTIVE` does not manage those markers. |
| `await list_thoughts(...)` | `list[ThoughtRecord]` | List with filters (keyword-only) |
| `await count_thoughts(...)` | `int` | Count with filters (keyword-only) |
| `await delete_thought(thought_id)` | `bool` | Hard delete; `True` if a row was removed. Deleting a thought also deletes every edge for which it is either endpoint, its embedding, and its linked actions. This is a physical cascade, unlike valid-time invalidation. **Below core schema 12 this cascade does not happen.** The `ON DELETE CASCADE` on `edge`, `embedding` and `action` arrives with the core-12 migration, so on a database carried forward from an older engrava and never migrated the thought's `embedding` row outlives the delete. The delete does still purge that thought's own `vec0` vector, so the identifier is **not** reachable straight afterwards; it returns once the reconcile that runs on the next sqlite-vec-enabled open backfills the index from the surviving `embedding` row. From then on it is an ordinary candidate on that arm whenever a sqlite-vec backend is **active** on the store and the query carries no effective metadata predicate — the arm *can* return it, subject to the same similarity threshold and `top_k` window as any live row. Run `engrava migrate`. See [Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated). |
| `await invalidate_thought(thought_id, valid_until)` | `ThoughtRecord` | Close the thought's *valid-time* interval at the given ISO-8601 instant — deterministic, idempotent, non-cascading, and **not a delete** (the row stays on file and remains retrievable for instants before `valid_until`). Raises `ThoughtNotFoundError` if missing. See [Bi-temporal Model](bitemporal.md#invalidate-vs-delete) |
| `await record_access(thought_id)` | `None` | Mark a thought as accessed — bumps `access_count` and sets `last_accessed_at`; raises `ThoughtNotFoundError` if missing. Drives the access-frequency dreaming signal. |

```python
import uuid
from engrava import ThoughtRecord, ThoughtType, Priority, LifecycleStatus

record = ThoughtRecord(
    thought_id=str(uuid.uuid4()),
    thought_type=ThoughtType.OBSERVATION,
    essence="Short summary (1-200 chars)",
    content="Full text content.",
    priority=Priority.P2,
    lifecycle_status=LifecycleStatus.ACTIVE,
    created_cycle=0,
    updated_cycle=0,
    source="human",
)
stored = await store.create_thought(record)
```

```python
from engrava import LifecycleStatus

# Archive, then restore — a lossless round-trip.
await store.update_thought(stored.thought_id, lifecycle_status=LifecycleStatus.ARCHIVED)
restored = await store.restore_thought(stored.thought_id, current_cycle=42)
assert restored.lifecycle_status is LifecycleStatus.ACTIVE
```

##### `create_thought` keyword-only options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expires_after_seconds` | `int \| None` | `None` | Relative TTL; overrides the store default |
| `deduplicate` | `bool` | `False` | Collapse identical `content` (SHA-256) into the existing thought, bumping `confirmation_count` |

##### `list_thoughts` / `count_thoughts` filters (keyword-only)

| Parameter | Type | Description |
|-----------|------|-------------|
| `thought_type` | `str \| None` | Filter by type |
| `lifecycle_status` | `str \| None` | Filter by status |
| `priority` | `str \| None` | Filter by priority |
| `include_expired` | `bool` | Include expired thoughts (default `False`) |
| `min_cycle` | `int \| None` | Minimum `updated_cycle`, inclusive (`list_thoughts` only) |
| `max_cycle` | `int \| None` | Maximum `updated_cycle`, inclusive (`list_thoughts` only) |
| `limit` | `int` | Max results (`list_thoughts` only; default `50`) |
| `offset` | `int` | Results to skip (`list_thoughts` only; default `0`) |

> `list_thoughts` also supports `visibility`, `exclude_visibility`, and
> `provenance_filter` (see
> [Provenance capture](#provenance-capture)).

##### Provenance capture

Attach optional, typed write-time **provenance** to a thought — the signals that
become irrecoverable once a synthesised thought exists (which session/actor
produced it, what query and instruction shaped it, which thoughts it was built
from). Set `ThoughtRecord.provenance` to a `ProvenanceContext` before
`create_thought`; every field is optional and bounded, and the default `None` is
byte-identical to not using provenance at all.

```python
from engrava.domain.models.provenance import ProvenanceContext

record = ThoughtRecord(
    # ... the usual fields ...
    provenance=ProvenanceContext(
        session_id="sess-42",                              # indexed identity hint
        actor_id="agent-a",                                # indexed identity hint
        retrieval_query="remote work trade-offs",          # ≤4096 chars
        instruction_context="summarise for a busy exec",   # ≤4096 chars
        retrieval_context_ids=["t-1", "t-2"],              # ≤128 ids, ≤256 chars each
    ),
)
await store.create_thought(record)
```

| `ProvenanceContext` field | Type | Notes |
|---|---|---|
| `session_id` | `str \| None` | Session handle. **Indexed** for lookup. ≤256 chars |
| `actor_id` | `str \| None` | Actor/agent handle. **Indexed**. ≤256 chars |
| `retrieval_query` | `str \| None` | Query text that retrieved the feeding context. ≤4096 chars |
| `instruction_context` | `str \| None` | Instruction / system-prompt fragment. ≤4096 chars |
| `retrieval_context_ids` | `list[str] \| None` | Source thought ids. Queryable, not indexed. ≤128 × ≤256 chars |

**Query by provenance.** `list_thoughts(provenance_filter=...)` takes a
`MetadataFilter` — an `AND` of `FieldPredicate`s over the `provenance` JSON
column. Predicates on `$.session_id` / `$.actor_id` use the provenance identity
index; rows whose `provenance` is `NULL` or malformed JSON never match a
non-empty filter.

```python
from engrava.domain.models.filters import FieldOp, FieldPredicate, MetadataFilter

mine = await store.list_thoughts(
    provenance_filter=MetadataFilter(
        [FieldPredicate("$.session_id", FieldOp.EQ, "sess-42")]
    ),
)
```

> **Untrusted hint — never identity, authentication, or authorization.**
> Provenance is descriptive-only: the engine grants it zero authority and
> consults it for no access, ranking, or consolidation decision. `actor_id` is
> **not** a tenant boundary — tenant isolation is the store's file boundary (one
> store per tenant). The engine never infers provenance; the caller passes it
> explicitly.

#### Edge CRUD

`create_edge` takes a single `EdgeRecord` object and returns the persisted
record. It raises `ReferentialIntegrityError` when an endpoint thought does
not exist and `DuplicateEdgeError` when the same directed, typed relationship
already exists.

| Method | Returns | Description |
|--------|---------|-------------|
| `await create_edge(edge)` | `EdgeRecord` | Persist an `EdgeRecord`; raises `ReferentialIntegrityError` on a missing endpoint or `DuplicateEdgeError` for an existing `(from, to, type)` relationship |
| `await get_edges(thought_id, *, direction='BOTH')` | `list[EdgeRecord]` | Edges for a thought (`direction` is `'IN'`/`'OUT'`/`'BOTH'`, keyword-only) |
| `await list_edges(*, edge_type=None, source=None, filters=None, limit=5000)` | `list[EdgeRecord]` | List edges with optional filters (`filters` is a typed `MetadataFilter` over the edge `metadata`; see the [`metadata` field](#metadata-field-edges) note) |
| `await update_edge(edge_id, **changes)` | `EdgeRecord` | Update edge fields |
| `await delete_edge(edge_id)` | `bool` | Hard delete; `True` if a row was removed |
| `await invalidate_edge(edge_id, valid_until)` | `EdgeRecord` | Close the edge's *valid-time* interval at the given ISO-8601 instant — deterministic, idempotent, and **not a delete** (the row stays on file). Invalidating a thought does **not** cascade to its edges; invalidate them separately. See [Bi-temporal Model](bitemporal.md#invalidate-vs-delete) |

The database permits at most one edge for each
`(from_thought_id, to_thought_id, edge_type)` tuple, independently of
`edge_id`; attempting to insert the same directed, typed relationship twice
raises `DuplicateEdgeError`. Reversing the endpoints or using a different edge
type produces a distinct edge. A duplicate caller-supplied `edge_id` for a
different relationship remains a separate integrity failure. A **hard** thought
delete cascades to edges where
the thought is either endpoint; `invalidate_thought()` does not remove or
invalidate any edge.

**Below core schema 12 this cascade does not happen.** The `ON DELETE CASCADE` on
`edge`, `embedding` and `action` arrives with the core-12 migration, so on a database
carried forward from an older engrava and never migrated the thought's `embedding` row
outlives the delete. The delete does still purge that thought's own `vec0` vector, so
the identifier is **not** reachable straight afterwards; it returns once the reconcile
that runs on the next sqlite-vec-enabled open backfills the index from the surviving
`embedding` row. From then on it is an ordinary candidate on that arm whenever a
sqlite-vec backend is **active** on the store and the query carries no effective
metadata predicate — the arm *can* return it, subject to the same similarity threshold
and `top_k` window as any live row. Run `engrava migrate`. See
[Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated).

```python
import uuid
from engrava import EdgeRecord, EdgeType

await store.create_edge(
    EdgeRecord(
        edge_id=str(uuid.uuid4()),
        from_thought_id=src_id,
        to_thought_id=dst_id,
        edge_type=EdgeType.ASSOCIATED,
        weight=0.8,
        created_cycle=0,
    )
)
```

#### REFLECTION lineage

Helpers for navigating the `CONSOLIDATED_FROM` graph that dreaming builds
between a REFLECTION and the source thoughts it summarises.

| Method | Returns | Description |
|--------|---------|-------------|
| `await consolidated_member_ids(reflection_id)` | `list[str]` | The thought IDs a REFLECTION was consolidated from |
| `await consolidated_source_statuses(reflection_id)` | `list[str]` | The lifecycle statuses of those source thoughts (e.g. to detect a fully-archived, orphaned cluster) |
| `await reflections_consolidated_from(source_id)` | `list[str]` | The REFLECTION IDs that consolidated a given source thought (the reverse direction) |
| `await thought_exists_by_source(*, source, thought_type_value)` | `bool` | Whether any thought exists with the given `source` and type — keyword-only |

```python
# Walk a REFLECTION down to its sources, and back from a source to its REFLECTIONs.
member_ids = await store.consolidated_member_ids(reflection_id)
for thought_id in member_ids:
    source = await store.get_thought(thought_id)
    if source is not None:
        print(source.essence)

# Detect an orphaned cluster — every source archived/gone:
statuses = await store.consolidated_source_statuses(reflection_id)
is_orphaned = bool(statuses) and all(s != "ACTIVE" for s in statuses)

# Reverse direction: which REFLECTIONs summarise this source?
parents = await store.reflections_consolidated_from(member_ids[0])

# Exact-source existence check (e.g. dreaming's idempotency guard — a REFLECTION's
# source is "dreaming:<cluster_hash>", so match the full value, not a prefix):
exists = await store.thought_exists_by_source(
    source="dreaming:abc123def4567890", thought_type_value="REFLECTION"
)
```

#### Embedding Operations

| Method | Returns | Description |
|--------|---------|-------------|
| `await store_embedding(thought_id, vector, *, model_name="all-MiniLM-L12-v2", embedding_id=None)` | `EmbeddingRecord` | Store an embedding vector (dimension derived from `len(vector)`) |
| `await get_embedding(thought_id)` | `EmbeddingRecord \| None` | Retrieve embedding |
| `await search_similar(query_vector, top_k=10, threshold=0.0, *, include_archived=False)` | `list[tuple[str, float]]` | Cosine similarity search → `(thought_id, score)`. Raises `VectorDimensionMismatchError` on a wrong-length query vector; a degenerate vector degrades to `[]` (see [Known Limitations](known-limitations.md#query-vector-dimension-mismatch)). |

#### Full-Text & Hybrid Search

`search_fts` and `search_similar` return `(thought_id, score)` tuples — fetch
the record with `get_thought` when you need its fields. `search_hybrid`
returns a single `HybridSearchResult` container.

| Method | Returns | Description |
|--------|---------|-------------|
| `await recall(query, *, top_k=10, current_cycle=None, recency_now=None, recency_now_half_life=None, filters=None, visibility=None, collapse_key=None, collapse_max_per_unit=None, include_archived=False)` | `HybridSearchResult` | Ergonomic shorthand over `search_hybrid` for the common retrieval case; supports either recency axis, scoped retrieval, collapse, and archived-row opt-in |
| `await search_fts(query, top_k=10, *, include_archived=False)` | `list[tuple[str, float]]` | FTS5/BM25 text search → `(thought_id, bm25_score)`; malformed expert expressions retry once through safe bare normalization |
| `await search_hybrid(query_text, query_vector=None, *, top_k=10, ...)` | `HybridSearchResult` | Combined FTS + vector + recency + priority + graph |
| `await search_reflections_only(query_text, query_vector=None, *, top_k=10, current_cycle=None)` | `HybridSearchResult` | Search restricted to active, unexpired REFLECTION thoughts, ranked by vector similarity and optional cycle recency. Expiry is evaluated against one UTC instant captured for the call; `expires_at <= now` is excluded. |

The complete `search_hybrid` signature is:

```python
async def search_hybrid(
    query_text: str,
    query_vector: list[float] | None = None,
    *,
    top_k: int = 10,
    fts_weight: float | None = None,
    vector_weight: float | None = None,
    recency_weight: float | None = None,
    recency_half_life: int | None = None,
    current_cycle: int | None = None,
    recency_now: str | None = None,
    recency_now_half_life: int | None = None,
    fts_top_k: int = 50,
    vector_top_k: int = 50,
    priority_weight: float | None = None,
    graph_weight: float | None = None,
    graph_edge_decay: float | None = None,
    include_reflections: bool = True,
    reflection_boost: float | None = None,
    filters: MetadataFilter | None = None,
    visibility: VisibilityQueryFilter | None = None,
    collapse_key: str | Sequence[str] | None = None,
    collapse_max_per_unit: int | None = None,
    include_archived: bool = False,
) -> HybridSearchResult: ...
```

`current_cycle` and `recency_now` are mutually exclusive when both are supplied
explicitly. `recency_now_half_life` is measured in seconds and is only valid
with `recency_now`; `recency_half_life` is measured in cognitive cycles.

> **Archived thoughts are excluded by default.** `recall`, `search_fts`,
> `search_hybrid`, and `search_similar` drop `ARCHIVED` thoughts from the default
> candidate set (the mechanism behind [Forgetting](memory-hygiene.md)). Pass
> `include_archived=True` to re-admit them for one call, or `restore_thought` to
> re-activate a row. `search_reflections_only` remains restricted to active
> reflections and has no archive opt-in. See
> [Search](search.md#archived-thoughts-are-excluded-by-default).

##### Scoped & de-fragmented retrieval (keyword-only)

Both `recall` and `search_hybrid` accept the same four opt-in keywords for
narrowing and de-duplicating ranked results. Every one defaults to `None` and
the `None` path is byte-identical to an unscoped query, so existing callers see
no change.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filters` | `MetadataFilter \| None` | Metadata scope — an `AND` of typed `FieldPredicate`s over the thought's `metadata`, applied *in-arm* before each arm's candidate limit so it never starves `top_k`. `None` (or an empty filter) leaves the candidate set unchanged. A query refinement, **not** a security boundary. |
| `visibility` | `VisibilityQueryFilter \| None` | Bounded `(visibility IN … [OR owner = …])` refinement over `$.visibility` / `$.owner` — the "public-or-mine" pattern. **A query filter, not access control:** it performs no authentication, authorization, or ownership enforcement; the caller can forge `owner`; it is bypassable and must not be used to protect tenant data (use one store per tenant for isolation). |
| `collapse_key` | `str \| Sequence[str] \| None` | Opt-in de-fragmentation unit key — a single metadata path (`"$.session_id"`) or an ordered sequence forming a composite key. When set, only the single best-ranked row per caller-defined unit reaches the result and the freed slots backfill deeper distinct units. A **presentation / de-dup convenience, not a filter and not isolation.** The collapse step mutates no score, but *setting* `collapse_key` widens the internal candidate pool, which can rescale the min-max-normalized keyword scores and shift order among units; only `collapse_key=None` leaves the path byte-identical. A missing/malformed key ⇒ the row is its own unit, never collapsed. |
| `collapse_max_per_unit` | `int \| None` | Opt-in intra-unit retention depth for `collapse_key`. `None` (default) keeps one best row per unit; an integer `>= 1` keeps up to that many highest-ranked members of a unit and lets the freed slots backfill deeper distinct units. Only effective together with `collapse_key`; a value `< 1` is rejected. |

```python
from engrava import MetadataFilter, VisibilityQueryFilter
from engrava import FieldOp, FieldPredicate

# Scope recall to one session's memories, keep only "public" or my own rows,
# and collapse per-turn fragments to one best row each.
result = await store.recall(
    "remote work trade-offs",
    top_k=10,
    filters=MetadataFilter([FieldPredicate("$.session_id", FieldOp.EQ, "sess-42")]),
    visibility=VisibilityQueryFilter(frozenset({"public"}), owner="alice"),
    collapse_key="$.session_turn",
    collapse_max_per_unit=2,   # keep up to 2 rows per turn instead of 1
)
```

#### Metrics & maintenance

| Method | Returns | Description |
|--------|---------|-------------|
| `await metrics()` | `EngravaMetrics` | Snapshot of thought/edge counts, storage, and search-latency percentiles (see [Observability](observability.md)) |
| `await max_cycle()` | `int` | Cognitive-cycle high-water mark across `thought.updated_cycle` and `edge.created_cycle`; returns `0` for an empty store |
| `await cleanup_expired(now=None, *, exclude_id=None)` | `CleanupResult` | Archive or delete thoughts past their `expires_at` |
| `await run_hygiene(*, current_cycle=None, now=None)` | `HygieneResult` | Run one [Forgetting / Memory Hygiene](memory-hygiene.md) pass: archive cold thoughts (reversibly) and, when `auto_gc_enabled`, GC ones past both restore windows. Raises `RuntimeError` if no policy was supplied; returns an empty result when the supplied policy is disabled. |
| `await derive_existing(thought_id)` | `DeriveResult` | Backfill: run the registered derived-records producer over an already-stored source thought (idempotent). Raises `SourceThoughtNotFoundError` if the id is absent. See [Backfilling existing thoughts](extension-hooks.md). |
| `await flush_access_buffer()` | `int` | Persist buffered retrieval-access deltas in one batch; returns the number of thoughts updated |
| `await retire_orphan_reflections()` | `int` | Archive active REFLECTIONs whose entire `CONSOLIDATED_FROM` source set is no longer active |
| `await consolidate(*, current_cycle=None)` | `ConsolidationResult` | Run the dreaming extension wired by `from_config`; raises `RuntimeError` when dreaming is not enabled and uses `cycle_provider` when no explicit cycle is supplied |
| `await verify_embedding_model()` | `None` | Raise `EmbeddingModelMismatchError` if the stored model lock disagrees with the configured provider |
| `async with store.suppress_access_tracking():` | context manager | Suppress implicit retrieval-access buffering inside the current async task; nestable and concurrency-safe, with no effect when access tracking is disabled. Used by internal maintenance and read-only views so their reads do not inflate the frequency signal. |
| `async with store.suspend_auto_commit():` | context manager | Defer per-call commits so a block of writes commits once (rolls back on error) — use for bulk ingest. The window belongs to the **store instance**, not to the task that opened it: any other task's write joins the same transaction and rolls back with it. Drive it from one task at a time |
| `await close()` | `None` | Close the owned connection (only when the store opened it via `from_config`) |

**Read-only health counters** (plain properties, not `await`, not in the metrics
snapshot): `store.fts_match_failure_count` and `store.vector_arm_degradation_count`
are monotonic `int` counters that surface silent, self-healing search-arm
degradations. See [Observability signals](observability.md#observability-signals).

```python
# Bulk ingest: one transaction instead of one commit per write.
async with store.suspend_auto_commit():
    for record in many_records:
        await store.create_thought(record)
# commit happens once on clean exit; any exception rolls the whole block back
```

### `ReadOnlyEngrava`

A composition wrapper that delegates reads to the wrapped store and raises
`ReadOnlyViolationError` on any write. Use it to hand a retrieval-only view of
shared memory to a component that should never mutate it — e.g. a sub-agent or
worker whose job is only to look things up. Delegated reads suppress access
tracking, so they do not stage deferred access-count mutations.

The guarantee is **behavioural, not structural**. `ReadOnlyEngrava` implements
the narrower `EngravaReadProtocol`, but also exposes signature-compatible write
methods whose only behaviour is to raise. Because `EngravaCoreProtocol` is a
runtime-checkable structural protocol, those blockers mean
`isinstance(ro, EngravaCoreProtocol)` is also `True`. Use
`EngravaReadProtocol` to type a read capability; do not use the structural core
check as evidence that a value is writable.

```python
from engrava import ReadOnlyEngrava

ro = ReadOnlyEngrava(store)
thought = await ro.get_thought("abc")  # OK
await ro.create_thought(...)           # Raises ReadOnlyViolationError
```

### `EngravaManager`

Multi-service database isolation. Each service owns a separate
`<data_dir>/<service_name>.db` file, with its own schema, connection, embedding
configuration, and journal. Service names accepted by `get_store()`,
`service_exists()`, and `delete_service()` must match
`^[a-z][a-z0-9_-]{0,62}$`: start with a lowercase
letter, then use lowercase letters, digits, `_`, or `-`, up to 63 characters.

| Method | Description |
|--------|-------------|
| `await get_store(service_name)` | Lazily create/open and initialize a service store on first access. The manager caches the store, so later calls for the same name return the same live instance until `close_all()` or deletion. |
| `service_exists(service_name)` | Validate the service name, then check whether its `.db` file exists without opening or creating it. Raises `ConfigError` for an invalid name. |
| `await list_services()` | Scan `data_dir` for `.db` files and return their stems in sorted order; returns `[]` when the directory does not exist. |
| `await delete_service(service_name)` | Close and evict a cached store, then delete its `.db`, `.db-wal`, and `.db-shm` files. Raises `FileNotFoundError` when the main database file does not exist. This permanently deletes that service's data. |
| `await close_all()` | Close all cached connections and clear the cache; idempotent. Database files remain on disk, and a later `get_store()` opens a fresh store instance. Async context-manager exit calls this method. |

```python
from pathlib import Path

from engrava import EngravaManager

async with EngravaManager(data_dir=Path("./data")) as mgr:
    store = await mgr.get_store("my-service")
    assert mgr.service_exists("my-service")
    print(await mgr.list_services())  # ["my-service"]

# The context manager has closed its cached connections, but the database
# remains. Delete is a separate, destructive operation:
async with EngravaManager(data_dir=Path("./data")) as mgr:
    await mgr.delete_service("my-service")
```

## Domain Models

All models are **frozen** Pydantic objects. For `ThoughtRecord`, use
`thought.evolve(...)` when changing lifecycle or temporal fields: it validates
lifecycle transitions, preserves `created_at` immutability, refreshes
`updated_at` unless supplied explicitly, and revalidates the complete record.
Likewise, `ActionRecord.evolve(...)` validates action-status transitions.

Pydantic's `model_copy(update={...})` does **not** validate its update payload.
Reserve it for trusted, already-validated changes to fields on which no domain
invariant depends (or use `model_copy()` without `update` for an unchanged
copy). Do not use it for `ThoughtRecord` lifecycle or timestamp changes.

### `ThoughtRecord`

| Field | Type | Description |
|-------|------|-------------|
| `thought_id` | `str` | UUID primary key |
| `thought_type` | `ThoughtType` | Classification |
| `essence` | `str` | Short summary (1-200 chars) |
| `content` | `str` | Full text |
| `priority` | `Priority` | P1-P4 |
| `lifecycle_status` | `LifecycleStatus` | State machine status |
| `created_cycle` | `int` | Creation cycle number |
| `updated_cycle` | `int` | Last update cycle |
| `source` | `str` | Origin identifier |
| `confidence` | `float \| None` | Confidence score |
| `embedding_ref` | `str \| None` | Embedding ID reference |
| `source_type` | `KnowledgeSource` | Knowledge provenance |
| `confirmation_count` | `int` | Experience confirmations |
| `consolidated_from` | `list[str] \| None` | Source thought IDs if consolidated |
| `visibility` | `ThoughtVisibility` | Access scope |
| `access_count` | `int` | Times explicitly accessed |
| `action_outcome_score` | `float \| None` | Mean outcome value (`[0.0, 1.0]`) over the thought's terminal linked actions; `None` when it has none. Maintained by the store when a linked action reaches a terminal state |
| `last_accessed_at` | `str \| None` | ISO-8601 datetime of last access |
| `created_at` | `str \| None` | ISO-8601 datetime when persisted |
| `updated_at` | `str \| None` | ISO-8601 datetime of last mutation |
| `expires_at` | `str \| None` | ISO-8601 datetime when the thought expires (TTL) |
| `valid_from` | `str \| None` | ISO-8601 start of the fact's real-world *valid time* (open lower bound when `None`); see [Bi-temporal Model](bitemporal.md) |
| `valid_until` | `str \| None` | ISO-8601 end of *valid time*, **exclusive** (open upper bound when `None`); see [Bi-temporal Model](bitemporal.md) |
| `metadata` | `dict[str, MetadataValue]` | Caller-supplied structured attributes (default `{}`) |
| `provenance` | `ProvenanceContext \| None` | Optional bounded write-time provenance; an untrusted query hint, never identity or authorization |
| `pinned` | `bool` | Hard keep-intent for Memory Hygiene; pinned thoughts are never auto-archived or auto-GC'd (default `False`) |
| `archived_at_cycle` | `int \| None` | Cognitive cycle stamped only by hygiene archival and cleared by `restore_thought`; gates cycle-based GC eligibility |
| `archived_at` | `str \| None` | UTC ISO-8601 instant stamped only by hygiene archival and cleared by `restore_thought`; gates wall-clock GC eligibility |

#### `metadata` field

`MetadataValue = str | int | float | bool | None | dict[str, MetadataValue]`
— leaf values must be JSON-serializable scalars; nested `dict` values are
accepted for structured namespaces (e.g.
`metadata["source"] = {"is_self": True, "confidence": "high"}`). Lists,
tuples, and other rich containers are rejected at the `create_thought` /
`update_thought` boundaries with a `ValueError`.

Conventional keys (recommended, **not enforced** — callers decide what
to populate):

| Key | Type | Purpose | Example |
|-----|------|---------|---------|
| `role` | `str` | Conversation role | `"user"`, `"assistant"`, `"system"` |
| `lang` | `str` | Content language (ISO 639-1) | `"en"`, `"pl"`, `"ja"` |
| `content_type` | `str` | Content category | `"natural_language"`, `"code"`, `"speech"` |
| `session_id` | `str` | External session reference | UUID, business ID |
| `turn_index` | `int` | Turn position within a session | `0`, `1`, `2` |
| `speaker` | `str` | Named speaker (multi-party) | `"Alice"`, `"Customer"` |

```python
await store.create_thought(
    ThoughtRecord(
        thought_id="...",
        thought_type=ThoughtType.OBSERVATION,
        essence="...",
        content="...",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="ingest",
        metadata={"role": "user", "lang": "en", "turn_index": 5},
    ),
)
```

**Size limits.** Serialized payloads above ~4 KiB emit a `WARNING` log;
payloads above ~64 KiB are rejected with `ValueError`.  If the data is
genuinely large (e.g. transcripts, structured documents), store it in
`content` or via an external reference rather than in `metadata`.

**Persistence.** Stored as a `metadata_json TEXT NOT NULL DEFAULT '{}'`
column on the `thought` table since `user_version = 11`.  JSON1
extension is recommended for filtering queries (`json_extract(metadata_json, '$.role')`).

### `EdgeRecord`

| Field | Type | Description |
|-------|------|-------------|
| `edge_id` | `str` | UUID primary key |
| `from_thought_id` | `str` | Source thought |
| `to_thought_id` | `str` | Target thought |
| `edge_type` | `EdgeType` | Relationship type |
| `weight` | `float` | Strength (0.0-1.0) |
| `created_cycle` | `int` | Creation cycle |
| `source` | `KnowledgeSource` | Provenance (default `EXPERIENCE`) |
| `decay_multiplier` | `float` | Decay rate multiplier (default `1.0`) |
| `valid_from` | `str \| None` | ISO-8601 start of the edge's real-world *valid time* (open lower bound when `None`); see [Bi-temporal Model](bitemporal.md) |
| `valid_until` | `str \| None` | ISO-8601 end of *valid time*, **exclusive** (open upper bound when `None`); see [Bi-temporal Model](bitemporal.md) |
| `metadata` | `dict[str, MetadataValue]` | Caller-supplied structured attributes (default `{}`); see the [`metadata` field](#metadata-field-edges) note below |

#### `metadata` field (edges)

Edges carry the **same** generic `metadata` field as thoughts —
`dict[str, MetadataValue]`, default `{}` — with the identical value domain
(`str | int | float | bool | None | dict[str, MetadataValue]`; lists and
non-finite floats such as `NaN` / `Infinity` are rejected) and the same
~4 KiB warn / ~64 KiB hard-reject size limits. It is validated at the
`create_edge` / `update_edge` boundaries with a `ValueError`.

Unlike thought metadata, edge metadata keys carry **no reserved meaning and
no conventional-key table** — the field is a purely generic carrier. Engrava
assigns no domain semantics to any key, applies no metadata-driven ranking,
enforces no vocabulary, and makes no compatibility guarantee about key names,
so do **not** treat any key as "well-known". Engrava's own edge creators
(dreaming, derived records) write an empty `{}`.

**Filtering.** `list_edges(filters=...)` accepts a typed `MetadataFilter` — an
`AND` of `FieldPredicate`s over the edge `metadata` — reusing the exact
machinery that backs thought-metadata filtering: operators `EQ` and `IN` only,
the restricted `$` / `$.key` / `$[0]` JSONPath grammar, and a 250-predicate
cap. An edge whose stored metadata is malformed JSON never matches a non-empty
filter. This is a query refinement, **not** a security boundary.

```python
from engrava import EdgeType
from engrava.domain.models.filters import FieldOp, FieldPredicate, MetadataFilter

supports = await store.list_edges(
    edge_type=EdgeType.ASSOCIATED,
    filters=MetadataFilter([FieldPredicate("$.subtype", FieldOp.EQ, "supports")]),
)
```

**Persistence.** Stored as a `metadata_json TEXT NOT NULL DEFAULT '{}'` column
on the `edge` table since `user_version = 19`.

### `EmbeddingRecord`

| Field | Type | Description |
|-------|------|-------------|
| `embedding_id` | `str` | UUID primary key |
| `owner_type` | `str` | Owning entity type (currently `"THOUGHT"`) |
| `owner_id` | `str` | Owning record ID (the thought ID) |
| `model_name` | `str` | Embedding model identifier |
| `dimension` | `int` | Vector dimensionality |
| `vector_blob` | `bytes` | Serialized vector (packed floats) |
| `created_at` | `str` | ISO-8601 creation timestamp |

### `ActionRecord`

Records an action the agent took (a tool call, a message, …), linked to the
thought that prompted it, with execution and verification state.

| Field | Type | Description |
|-------|------|-------------|
| `action_id` | `str` | UUID primary key |
| `source_thought_id` | `str` | The thought this action originated from |
| `action_type` | `ActionType` | Action classification |
| `intent` | `str` | Description of intent (min length 1) |
| `status` | `ActionStatus` | Current execution status |
| `verification_status` | `VerificationStatus` | Verification state |
| `raw_metrics_json` | `str \| None` | Optional ground-truth facts for verification |

**Store methods** (on `SqliteEngravaCore`):

| Method | Returns | Description |
|--------|---------|-------------|
| `await create_action(action)` | `ActionRecord` | Persist an `ActionRecord` |
| `await update_action(action_id, *, status=None, verification_status=None)` | `ActionRecord` | Advance a stored action's `status` and/or `verification_status`. Validates the transition when `status` changes (illegal jump raises `InvalidTransitionError`); a verification-only update is allowed even on a terminal action. Raises `ActionNotFoundError` when the id is unknown. A supplied value equal to the stored one is a no-op. |
| `await get_actions(thought_id)` | `list[ActionRecord]` | Actions linked to a thought |

`ActionStatus` is a state machine: `PLANNED → EXECUTING → CONFIRMED` / `FAILED`,
and `PLANNED → BLOCKED → PLANNED`. `can_transition_to(...)` / `evolve(...)`
enforce valid transitions on the in-memory record, and `update_action(...)`
enforces the same transitions when advancing the **stored** action (an illegal
change raises `InvalidTransitionError`).

When a linked action reaches a terminal state (`CONFIRMED` / `FAILED`), the
source thought's `action_outcome_score` — the mean outcome value over its
terminal actions, `None` when it has none — is recomputed. That score is also
read by the optional `action_outcome` dreaming signal and by Memory Hygiene's
usage-history gate and scoring path.

The per-action value is exact and depends on both terminal execution status and
verification:

| Action status | Verification status | Outcome value |
|---------------|---------------------|---------------|
| `FAILED` | Any | `0.0` |
| `CONFIRMED` | `CONFIRMED` | `1.0` |
| `CONFIRMED` | `FAILED` | `0.0` |
| `CONFIRMED` | `PENDING`, `PARTIAL`, or `UNVERIFIABLE` | `0.5` |
| `PLANNED`, `EXECUTING`, or `BLOCKED` | Any | Excluded (outcome undecided) |

`action_outcome_score` is the arithmetic mean of those values across terminal
actions only. It remains `None` when a thought has no terminal action.

```python
import uuid
from engrava import ActionRecord, ActionType, ActionStatus, VerificationStatus

action = ActionRecord(
    action_id=str(uuid.uuid4()),
    source_thought_id=prompting_thought_id,
    action_type=ActionType.TOOL_CALL,
    intent="search the web for flight prices",
    status=ActionStatus.PLANNED,
    verification_status=VerificationStatus.PENDING,
)
await store.create_action(action)

# advance the STORED action through its lifecycle (journaled; validates each
# transition). A terminal status recomputes the source thought's outcome score:
await store.update_action(action.action_id, status=ActionStatus.EXECUTING)
await store.update_action(action.action_id, status=ActionStatus.CONFIRMED)
# verification can still advance while the status stays terminal:
await store.update_action(
    action.action_id, verification_status=VerificationStatus.CONFIRMED
)

actions = await store.get_actions(prompting_thought_id)
```

### `HybridSearchResult`

A frozen container of ranked results plus backend diagnostics. It is **not** a
per-result object with score breakdowns.

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[tuple[str, float]]` | Ranked `(thought_id, combined_score)`, highest first |
| `backends_used` | `frozenset[str]` | Backends/signals available or applicable to the query (e.g. `"fts5"`, `"vector"`, `"graph_expansion"`); a backend may appear even when it returned no rows |
| `reflections_evicted` | `int` | Number of REFLECTION rows removed from the final top-K window by `reflection_topk_cap` and backfilled with non-reflections (default `0`) |

```python
result = await store.search_hybrid("query text", top_k=5)
for thought_id, score in result.results:
    record = await store.get_thought(thought_id)
    ...
```

### Forgetting (Memory Hygiene) types

Configured under `hygiene_policy` and returned by
[`run_hygiene`](#metrics--maintenance) — see [Memory Hygiene](memory-hygiene.md)
and [Configuration → `hygiene_policy`](configuration.md#hygiene_policy).

| Type | Kind | Key fields |
|------|------|------------|
| `HygienePolicyConfig` | config | `enabled`, `eviction_threshold`, `protected_priorities`, `signal_weights`, `auto_gc_enabled`, `gc_min_archive_age_cycles`, `gc_restore_window_seconds`, `min_inactivity_age_seconds`, `max_evictions_per_run`, `dry_run` |
| `HygieneResult` | result | `archived_count`, `gc_count`, `candidates_evaluated`, `dry_run`, `would_evict`, `flat_signals` |
| `EvictionReason` | audit record | `thought_id`, `keep_score`, `eviction_score`, `decay_multiplier`, `threshold`, `signals`, `mechanism` (always `"hygiene"`) |

## Metadata Helpers

Three exported helpers build the structured `metadata` dict that pins a
thought's origin (self vs external, source id, language). They are pure
functions — same arguments always return an equal dict — and you are free to
pass a literal dict instead; the helpers exist to remove typo-driven shape
mismatches at the call site.

| Helper | Signature | Use for |
|--------|-----------|---------|
| `percept` | `percept(*, is_self=False, source_id=None, label=None, confidence="high", lang="en")` | Input arriving from outside (user message, document) |
| `utterance` | `utterance(*, lang="en")` | The agent's own output sent to the world |
| `thought` | `thought(*, lang="en")` | The agent's internal cognition (reflection, plan) |

```python
from engrava import percept

metadata = percept(source_id="user-1", label="user")
# -> {'perspective': 'percept',
#     'source': {'is_self': False, 'confidence': 'high', 'id': 'user-1', 'label': 'user'},
#     'lang': 'en', 'content_type': 'natural_language'}
```

Pass the returned dict as `ThoughtRecord(..., metadata=...)`.

## Enums

All enums are `StrEnum` — JSON-serializable and stored as strings.

| Enum | Values |
|------|--------|
| `ThoughtType` | `TASK`, `OBSERVATION`, `BELIEF`, `REFLECTION`, `OUTPUT_DRAFT`, `NOTE` |
| `Priority` | `P1`, `P2`, `P3`, `P4` (P1 highest) |
| `LifecycleStatus` | `CREATED`, `ACTIVE`, `DONE`, `ARCHIVED` (forward state machine `CREATED → ACTIVE → DONE → ARCHIVED`, plus reversible `ARCHIVED → ACTIVE`) |
| `EdgeType` | `ASSOCIATED`, `DEPENDS_ON`, `DERIVED_FROM`, `MESSAGE_OF`, `BRIDGE`, `CONSOLIDATED_FROM`, `CONTESTED_BY` |
| `ActionType` | `CLI_OUTPUT`, `TOOL_CALL`, `MESSAGE`, `STATE_UPDATE` |
| `ActionStatus` | `PLANNED`, `EXECUTING`, `CONFIRMED`, `FAILED`, `BLOCKED` |
| `ThoughtVisibility` | member names `PRIVATE`, `SELECTIVE`, `PUBLIC` — **stored values are lowercase** (`"private"`, `"selective"`, `"public"`) |
| `KnowledgeSource` | `EXPERIENCE`, `SEEDED_LLM`, `DISTILLED_LLM`, `DREAMING` |
| `VerificationStatus` | `PENDING`, `CONFIRMED`, `PARTIAL`, `FAILED`, `UNVERIFIABLE` |
| `SplitMode` | `PARAGRAPH`, `FIXED_WINDOW` — the `StructuralSplitProducer` split strategy (see [Structural split modes](extension-hooks.md#1a5-split-modes-structuralsplitproducer)) |

## Exceptions

The table below defines the public types. For operational retry, repair, and
store-replacement guidance, see [Error handling and recovery](error-handling.md).

| Exception | Base | Description |
|-----------|------|-------------|
| `EngravaError` | `Exception` | Base for all engrava errors |
| `ThoughtNotFoundError` | `EngravaError` | Thought ID not found |
| `StaleDataError` | `EngravaError` | The guarded update matched no row: a competing writer stamped a new `updated_cycle`, or deleted the row, before this write. Nothing of the update was applied. Not a general staleness check — see [Concurrency](concurrency.md#optimistic-concurrency-and-staledataerror) |
| `InvalidTransitionError` | `EngravaError` | Invalid lifecycle state transition |
| `DuplicateEdgeError` | `EngravaError` | The directed `(from_thought_id, to_thought_id, edge_type)` relationship already exists |
| `ReadOnlyViolationError` | `EngravaError` | Write attempt on read-only store |
| `EmbeddingModelMismatchError` | `EngravaError` | Embedding model mismatch on restore |
| `EmbeddingProviderContractError` | `EngravaError` | A configured embedding provider does not expose `dimension`, a required `EmbeddingProviderProtocol` member. Carries `provider_class` / `member`. Raised where the core reads the member (vector search, `verify_embedding_model`), never at construction. A provider that has the property and fails inside it raises its own exception instead |
| `EmbeddingGenerationError` | `EngravaError` | Auto-embed failed under `require_embedding=True` (the thought is committed but unembedded); carries the failing `thought_id` |
| `ExtensionMigrationError` | `EngravaError` | Extension schema migration failed (e.g. attempted downgrade) |
| `ActionNotFoundError` | `EngravaError` | Action record ID not found; carries the failing `action_id` |
| `InvalidFilterError` | `EngravaError` | Metadata/visibility filter invalid at construction (bad value, empty filter, unsupported operator, or too many predicates) |
| `InvalidFilterPathError` | `EngravaError` | Filter path does not match the allowed JSONPath grammar (`$`, `$.key`, `$[0]` only) |
| `EmbeddingQueryPrefixMismatchError` | `EngravaError` | Active query prefix diverges from the corpus pairing, silently degrading ranking |
| `JournalIntegrityError` | `EngravaError` | On-open journal check (`journal.verify_on_open`) found a broken hash chain; carries first-broken diagnostics |
| `DerivedRecordError` | `EngravaError` | The derived-records seam rejected a producer result (over-cap return, or a derived identity colliding with its source); carries the failing `source_thought_id` |
| `SourceThoughtNotFoundError` | `EngravaError` | `derive_existing` targeted a source `thought_id` that does not exist; carries the failing `thought_id` |
| `VectorDimensionMismatchError` | `EngravaError` | A vector-search query vector's length differs from the store's embedding dimension; carries `expected` / `actual`. **Not** a `ValueError` — catch this typed error (or `EngravaError`). |
| `CycleProviderError` | `EngravaError` | A configured cycle provider returned a boolean, non-integer, or negative value |
| `RecencyModeConflictError` | `EngravaError` | A query explicitly supplied both `current_cycle` and `recency_now` |
| `InvalidRecencyArgumentError` | `EngravaError` | `recency_now` is malformed or `recency_now_half_life` is not positive |
| `ConnectionQuarantinedError` | `EngravaError` | A failed derived-record compensation left the connection potentially indeterminate; the store instance is terminal and must be replaced |
| `ConfigError` | `ValueError` | YAML or direct config construction violates a documented configuration invariant |

> `create_edge` raises `ReferentialIntegrityError` when an endpoint thought
> does not exist and `DuplicateEdgeError` when the relationship already exists.
> These exceptions are **not** re-exported from the top-level `engrava` package;
> import them from `engrava.domain.exceptions`.

## Protocols

### `EngravaReadProtocol`

The non-mutating capability shared by the writable store and
`ReadOnlyEngrava`: thought/edge/action/embedding reads, FTS/vector/hybrid
retrieval, `recall`, metrics, and `max_cycle`. Read-only retrieval suppresses
the concrete SQLite store's optional access-frequency instrumentation.
`ReadOnlyEngrava` declares and type-checks against this read-side protocol. Its
compatibility write methods are blocked and raise `ReadOnlyViolationError`.

### `EngravaCoreProtocol`

The writable abstract interface for an engrava implementation. It extends
`EngravaReadProtocol`, the separately exported read capability implemented by
both `SqliteEngravaCore` and `ReadOnlyEngrava`. Because runtime structural
checks inspect member names and `ReadOnlyEngrava` exposes rejecting write
methods, the wrapper also satisfies `isinstance(ro, EngravaCoreProtocol)`.
That result describes structural compatibility, not permission to write; use
the narrower read protocol for capability boundaries.

### `CycleProvider`

A runtime-checkable, synchronous pull protocol with one method:
`current_cycle() -> int`. It supplies a cognitive cycle only when a supported
call omits an explicit one and no explicit `recency_now` selects transaction-time
recency. It never advances a cycle, never stamps writes, and is never loaded from
YAML. The store validates each pulled value as a real, non-negative `int`.

| Implementation | Construction | Behavior |
|---|---|---|
| `StaticCycleProvider` | `StaticCycleProvider(value)` | Always returns one fixed value |
| `CallableCycleProvider` | `CallableCycleProvider(fn)` | Calls a zero-argument consumer function on every pull; purity is the consumer's responsibility |
| `MaxCycleProvider` | `await MaxCycleProvider.create(store)` | Caches `await store.max_cycle()`; `current_cycle()` may be stale until `await refresh()` |

### `EngravaHooksProtocol`

Extension hook interface — see [Extensions](extensions.md).

### `DerivedRecordProducerProtocol`

Optional capability: return `N` derived records from one stored thought. Core
persists each as an ordinary thought after the source is durable, through a
core-owned, guarded, per-child lifecycle. See
[Extension hooks §1A](extension-hooks.md#1a-derived-records-extension-seam).

```python
class DerivedRecordProducerProtocol(Protocol):
    async def derive_records(
        self, thought: ThoughtRecord, ctx: DeriveContext
    ) -> Sequence[DerivedRecord]: ...
```

Companion generic types (all public, `X.Y.x`-stable): `DerivedRecord`
(producer-owned non-empty `content` / `thought_type` / `priority` / `metadata` /
`attach_provenance_edge`; core derives the `essence` and owns identity),
`DeriveContext` (source id, content hash, cycle, informational `origin`; no
store handle), `DeriveGates`
(`enabled`, `on_error`, `max_derived_per_source`), and `DeriveResult` (the
`created` / `reused` / `skipped` tally returned by
[`derive_existing`](#metrics--maintenance)). Configure via the `derive:`
YAML section or the `derive_gates=` constructor argument. The shipped reference
producer is `StructuralSplitProducer` (see
[Structural split modes](extension-hooks.md)).

### `EmbeddingProviderProtocol`

```python
class EmbeddingProviderProtocol(Protocol):
    @property
    def dimension(self) -> int: ...
    @property
    def model_name(self) -> str: ...
    async def embed(self, text: str) -> list[float]: ...
```

## MindQL

See [MindQL](mindql.md) for the query language reference.

### `execute_mindql` (on the store)

`SqliteEngravaCore` exposes a convenience that runs a MindQL query directly
against the store's own connection, so you don't have to construct a
`MindQLExecutor` yourself. Like the executor, it takes a **parsed**
`MindQLQuery` — `parse()` the string first.

| Method | Returns | Description |
|--------|---------|-------------|
| `await execute_mindql(query, *, extensions=None)` | `MindQLResult` | Run a parsed `MindQLQuery` on the store's connection. `extensions` is an optional `dict[str, MindQLExtension]` registering extension commands. |

```python
from engrava import parse

result = await store.execute_mindql(
    parse("FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10")
)
print(result.rows)
```

### `MindQLExecutor`

`MindQLExecutor` runs against an open `aiosqlite.Connection` (the same
connection the store wraps), and `execute()` takes a **parsed** `MindQLQuery`
— parse the string first.

```python
from engrava import MindQLExecutor, MindQLResult, parse

executor = MindQLExecutor(conn)  # an aiosqlite.Connection, not a store
result: MindQLResult = await executor.execute(
    parse("FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 10")
)
print(result.rows)    # list[dict]
print(result.count)   # int | None (set for COUNT queries)
```

`MindQLResult` fields: `columns: list[str]`, `rows: list[dict]`,
`count: int | None`, `command: str`.

### `parse`

`parse()` returns a `MindQLQuery` plan (not a `MindQLCommand`). The grammar
requires a table name and an optional `WHERE` clause:
`FIND <table> [WHERE <field> <op> '<value>' [AND ...]] [LIMIT n]`.

```python
from engrava import parse, MindQLQuery

query: MindQLQuery = parse("FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 5")
print(query.command)     # MindQLCommand.FIND
print(query.table)       # "thought"
print(query.conditions)  # list of parsed conditions (field / operator / value)
print(query.limit)       # 5
```

See [MindQL](mindql.md) for the full grammar, operators, and per-table
filterable columns.

## Additional top-level exports

The sections above cover the primary store, records, protocols, and query
surface in depth. The remaining names exported directly from `engrava` are
listed here so the top-level package has a complete index; follow the linked
specialist guide for operational detail.

### Configuration objects and resolvers

| Export | Purpose |
|---|---|
| `EngravaConfig` | Root typed configuration returned by `load_config` |
| `SearchConfig` | Hybrid-search weights, half-lives, graph/reflection controls, and candidate-pool bounds |
| `DreamingConfig`, `DreamingGates` | Consolidation settings and eligibility gates |
| `EmbeddingConfig` | Embedding provider and model settings |
| `HygienePolicyConfig` | Forgetting / Memory Hygiene policy |
| `JournalConfig` | Journal enablement and verify-on-open settings |
| `MetricsConfig` | Metrics enablement and rolling latency-window size |
| `TTLConfig` | Expiry strategy, cadence, and default TTL |
| `ServiceConfig`, `ServicesConfig` | Per-service and multi-service manager configuration |
| `load_config(path)` | Parse and validate YAML into `EngravaConfig` |
| `resolve_embedding_provider(config)` | Construct the provider selected by an `EmbeddingConfig` |
| `resolve_hooks(hooks_class)` | Import and construct a dotted-path hooks class, or return default hooks for `None` |
| `resolve_manifests(manifest_paths, *, discover=False)` | Load explicit manifest paths and optionally append entry-point discovery results |

See [Configuration](configuration.md) for the fields, defaults, validation, and
YAML examples.

### Embedding and vector implementations

| Export | Purpose |
|---|---|
| `CallbackProvider` | Adapt caller-supplied embedding callbacks to `EmbeddingProviderProtocol` |
| `SentenceTransformerProvider` | Local sentence-transformers provider |
| `HuggingFaceProvider` | Hugging Face inference/provider adapter |
| `OllamaProvider` | Ollama embedding provider |
| `OpenAICompatibleProvider` | Provider for OpenAI-compatible embedding endpoints |
| `RoleAwareEmbeddingProvider` | Optional protocol for distinct query/document embedding paths |
| `SqliteVecSearchBackend` | sqlite-vec `vec0` backend used by configured stores |

See [Embeddings](guides/embeddings.md) and [Performance](performance.md) before
selecting a provider or vector backend.

### Dreaming and scoring types

| Export | Purpose |
|---|---|
| `DreamingExtension` | Consolidation implementation. `run_consolidation()` runs immediately; `is_due()` / `run_if_due()` apply the configured cycle cadence without starting a background scheduler. |
| `DreamingContext` | Per-run context supplied to dreaming signals |
| `DreamingSignalProtocol` | Contract implemented by promotion-scoring signals |
| `ActionOutcomeSignal`, `ConfidenceSignal`, `ConfirmationSignal` | Outcome, confidence, and confirmation scoring signals |
| `FrequencySignal`, `RecencySignal`, `StalenessSignal` | Access-frequency, cycle-recency, and staleness scoring signals |
| `ScoringContext` | Context type accepted by the reserved hook-level `score_function` |

See [Dreaming](dreaming.md) for activation, gates, signal semantics, and the
no-retrieval-accuracy-claim boundary.

### Journal, cleanup, and metrics value objects

| Export | Purpose |
|---|---|
| `JournalEntry` | One persisted hash-chain journal entry |
| `JournalWriter` | Low-level journal writer bound to an open connection; ordinary callers use `store.journal` / `verify_journal()` |
| `MutationType` | Journal mutation-kind enum |
| `CleanupStrategy` | TTL cleanup enum: `ARCHIVE` or `DELETE` |
| `ThoughtCounts`, `EdgeCounts` | Count components nested in `EngravaMetrics` |
| `StorageFootprint` | Database, WAL, vector-index, and total byte counts |
| `LatencyHistogram` | Rolling search-latency count and percentile summary |

### Discovery and parsing helpers

| Export | Purpose |
|---|---|
| `discover_manifests()` | Discover opt-in extension manifests from package entry points |
| `MindQLParseError` | Raised when `parse()` rejects invalid MindQL syntax |

### Deprecated compatibility aliases

`CoreThoughtRecord` is a compatibility alias of `ThoughtRecord`. The legacy
MindStore-era names below resolve lazily to their Engrava equivalents and emit a
`DeprecationWarning`; new code should not use them.

| Deprecated export | Use instead |
|---|---|
| `SqliteMindStoreCore` | `SqliteEngravaCore` |
| `MindStoreManager` | `EngravaManager` |
| `MindStoreConfig` | `EngravaConfig` |
| `MindStoreError` | `EngravaError` |
| `MindStoreCoreProtocol` | `EngravaCoreProtocol` |
| `MindStoreHooksProtocol` | `EngravaHooksProtocol` |
| `DefaultMindStoreHooks` | `DefaultEngravaHooks` |
| `ReadOnlyMindStore` | `ReadOnlyEngrava` |

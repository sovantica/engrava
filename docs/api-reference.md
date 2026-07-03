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

> The `SqliteEngravaCore(db_path=...)` form does **not** exist — pass a
> connection, or use `await SqliteEngravaCore.from_config(path)`.

#### Factory

| Method | Returns | Description |
|--------|---------|-------------|
| `await SqliteEngravaCore.from_config(config_path)` | `SqliteEngravaCore` | Build from a YAML config; opens + owns the connection and applies the schema. Use as an async context manager. |

#### Schema

| Method | Description |
|--------|-------------|
| `await ensure_schema()` | Create tables if missing, run migrations |
| `await close()` | Close the database connection |

#### Thought CRUD

`create_thought` takes a single frozen `ThoughtRecord` object (build it, then
pass it) and returns the persisted record — it does **not** take field
keyword arguments and does **not** return a UUID string.

| Method | Returns | Description |
|--------|---------|-------------|
| `await create_thought(thought, *, expires_after_seconds=None, deduplicate=False)` | `ThoughtRecord` | Persist a `ThoughtRecord`; returns the stored record. Raises `ValueError` if the ID already exists. |
| `await get_or_create(thought, *, expires_after_seconds=None)` | `tuple[ThoughtRecord, bool]` | Content-hash convenience over dedup: returns `(existing, False)` on a hash hit (confirmation bumped, like `deduplicate=True`) or `(new, True)` on a miss. The `bool` tells you whether it created, removing the check-then-create round trip. Does not alter a matched row's fields. |
| `await upsert_by_hash(thought, *, expires_after_seconds=None)` | `ThoughtRecord` | Update-on-match upsert: on a hash hit, overwrites the stored row's mutable fields (`essence`, `priority`, `metadata`, `visibility`, `lifecycle_status`, `source`, `confidence`, `source_type`, `thought_type`) from `thought` and returns it (**no** confirmation bump — distinct from `deduplicate=True`, which returns the stored record unchanged); on a miss, inserts. `content` is never rewritten (it is the hash key). |
| `await bulk_store(thoughts, *, deduplicate=False)` | `list[ThoughtRecord]` | Transactional batch insert: the whole list commits **once** and is all-or-nothing (any row error rolls the batch back). Order preserved. Under `auto_embed`, all thoughts are embedded in one batch provider call. `deduplicate` applies per row. |
| `await get_thought(thought_id)` | `ThoughtRecord \| None` | Retrieve by ID; `None` if not found |
| `await update_thought(thought_id, **changes)` | `ThoughtRecord` | Optimistic-concurrency update; raises `ThoughtNotFoundError` / `StaleDataError` |
| `await list_thoughts(...)` | `list[ThoughtRecord]` | List with filters (keyword-only) |
| `await count_thoughts(...)` | `int` | Count with filters (keyword-only) |
| `await delete_thought(thought_id)` | `bool` | Hard delete; `True` if a row was removed |
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
| `limit` | `int` | Max results (`list_thoughts` only; default `50`) |
| `offset` | `int` | Results to skip (`list_thoughts` only; default `0`) |

> `list_thoughts` also supports `min_cycle`, `max_cycle`, `visibility`, and
> `exclude_visibility`.

#### Edge CRUD

`create_edge` takes a single `EdgeRecord` object and returns the persisted
record. It raises `ReferentialIntegrityError` when an endpoint thought does
not exist.

| Method | Returns | Description |
|--------|---------|-------------|
| `await create_edge(edge)` | `EdgeRecord` | Persist an `EdgeRecord`; raises `ReferentialIntegrityError` on a missing endpoint |
| `await get_edges(thought_id, *, direction='BOTH')` | `list[EdgeRecord]` | Edges for a thought (`direction` is `'IN'`/`'OUT'`/`'BOTH'`, keyword-only) |
| `await list_edges(*, edge_type=None, source=None, limit=5000)` | `list[EdgeRecord]` | List edges with optional filters |
| `await update_edge(edge_id, **changes)` | `EdgeRecord` | Update edge fields |
| `await delete_edge(edge_id)` | `bool` | Hard delete; `True` if a row was removed |
| `await invalidate_edge(edge_id, valid_until)` | `EdgeRecord` | Close the edge's *valid-time* interval at the given ISO-8601 instant — deterministic, idempotent, and **not a delete** (the row stays on file). Invalidating a thought does **not** cascade to its edges; invalidate them separately. See [Bi-temporal Model](bitemporal.md#invalidate-vs-delete) |

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
| `await search_similar(query_vector, top_k=10, threshold=0.0)` | `list[tuple[str, float]]` | Cosine similarity search → `(thought_id, score)` |

#### Full-Text & Hybrid Search

`search_fts` and `search_similar` return `(thought_id, score)` tuples — fetch
the record with `get_thought` when you need its fields. `search_hybrid`
returns a single `HybridSearchResult` container.

| Method | Returns | Description |
|--------|---------|-------------|
| `await search_fts(query, top_k=10)` | `list[tuple[str, float]]` | FTS5/BM25 text search → `(thought_id, bm25_score)` |
| `await search_hybrid(query_text, query_vector=None, *, top_k=10, ...)` | `HybridSearchResult` | Combined FTS + vector + recency + priority + graph |
| `await search_reflections_only(query_text, *, top_k=10, ...)` | `HybridSearchResult` | Hybrid search restricted to REFLECTION thoughts |

`search_hybrid` keyword-only weight/limit overrides: `fts_weight`,
`vector_weight`, `recency_weight`, `recency_half_life`, `current_cycle`,
`fts_top_k`, `vector_top_k`, `priority_weight`, `graph_weight`,
`graph_edge_decay`, `include_reflections` (default `True`), `reflection_boost`.

#### Metrics & maintenance

| Method | Returns | Description |
|--------|---------|-------------|
| `await metrics()` | `EngravaMetrics` | Snapshot of thought/edge counts, storage, and search-latency percentiles (see [Observability](observability.md)) |
| `await cleanup_expired(now=None, *, exclude_id=None)` | `CleanupResult` | Archive or delete thoughts past their `expires_at` |
| `await verify_embedding_model()` | `None` | Raise `EmbeddingModelMismatchError` if the stored model lock disagrees with the configured provider |
| `async with store.suspend_auto_commit():` | context manager | Defer per-call commits so a block of writes commits once (rolls back on error) — use for bulk ingest |
| `await close()` | `None` | Close the owned connection (only when the store opened it via `from_config`) |

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
worker whose job is only to look things up.

```python
from engrava import ReadOnlyEngrava

ro = ReadOnlyEngrava(store)
thought = await ro.get_thought("abc")  # OK
await ro.create_thought(...)           # Raises ReadOnlyViolationError
```

### `EngravaManager`

Multi-service database isolation.

```python
from engrava import EngravaManager

async with EngravaManager(data_dir=Path("./data")) as mgr:
    store = await mgr.get_store("my-service")
    await mgr.list_services()   # -> ["my-service"]
    await mgr.delete_service("old-service")
```

## Domain Models

All models are **frozen** Pydantic objects. Use `model_copy(update={...})` to
create modified copies.

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
read by the optional `action_outcome` dreaming signal.

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

A frozen container of ranked results plus backend diagnostics. It has exactly
two fields — it is **not** a per-result object with score breakdowns.

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[tuple[str, float]]` | Ranked `(thought_id, combined_score)`, highest first |
| `backends_used` | `frozenset[str]` | Which signals contributed (e.g. `"fts5"`, `"vector"`, `"graph_expansion"`) |

```python
result = await store.search_hybrid("query text", top_k=5)
for thought_id, score in result.results:
    record = await store.get_thought(thought_id)
    ...
```

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
| `LifecycleStatus` | `CREATED`, `ACTIVE`, `DONE`, `ARCHIVED` (state machine `CREATED → ACTIVE → DONE → ARCHIVED`) |
| `EdgeType` | `ASSOCIATED`, `DEPENDS_ON`, `DERIVED_FROM`, `MESSAGE_OF`, `BRIDGE`, `CONSOLIDATED_FROM`, `CONTESTED_BY` |
| `ActionType` | `CLI_OUTPUT`, `TOOL_CALL`, `MESSAGE`, `STATE_UPDATE` |
| `ActionStatus` | `PLANNED`, `EXECUTING`, `CONFIRMED`, `FAILED`, `BLOCKED` |
| `ThoughtVisibility` | member names `PRIVATE`, `SELECTIVE`, `PUBLIC` — **stored values are lowercase** (`"private"`, `"selective"`, `"public"`) |
| `KnowledgeSource` | `EXPERIENCE`, `SEEDED_LLM`, `DISTILLED_LLM`, `DREAMING` |
| `VerificationStatus` | `PENDING`, `CONFIRMED`, `PARTIAL`, `FAILED`, `UNVERIFIABLE` |

## Exceptions

| Exception | Base | Description |
|-----------|------|-------------|
| `EngravaError` | `Exception` | Base for all engrava errors |
| `ThoughtNotFoundError` | `EngravaError` | Thought ID not found |
| `StaleDataError` | `EngravaError` | Concurrent modification detected |
| `InvalidTransitionError` | `EngravaError` | Invalid lifecycle state transition |
| `ReadOnlyViolationError` | `EngravaError` | Write attempt on read-only store |
| `EmbeddingModelMismatchError` | `EngravaError` | Embedding model mismatch on restore |
| `EmbeddingGenerationError` | `EngravaError` | Auto-embed failed under `require_embedding=True` (the thought is committed but unembedded); carries the failing `thought_id` |
| `ExtensionMigrationError` | `EngravaError` | Extension schema migration failed (e.g. attempted downgrade) |

> `create_edge` raises `ReferentialIntegrityError` when an endpoint thought
> does not exist. This exception is **not** re-exported from the top-level
> `engrava` package — catch it via
> `from engrava.domain.exceptions import ReferentialIntegrityError`.

## Protocols

### `EngravaCoreProtocol`

The abstract interface for any engrava implementation.

### `EngravaHooksProtocol`

Extension hook interface — see [Extensions](extensions.md).

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

# API Reference

Complete reference for engrava's public API.

## Core Store

### `SqliteEngravaCore`

The main persistence engine. All operations are async.

```python
from engrava import SqliteEngravaCore

store = SqliteEngravaCore(db_path=":memory:", hooks=None)
await store.ensure_schema()
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str \| Path` | — | SQLite database path (`:memory:` for in-memory) |
| `hooks` | `EngravaHooksProtocol \| None` | `None` | Extension hooks (defaults to `DefaultEngravaHooks`) |

#### Schema

| Method | Description |
|--------|-------------|
| `await ensure_schema()` | Create tables if missing, run migrations |
| `await close()` | Close the database connection |

#### Thought CRUD

| Method | Returns | Description |
|--------|---------|-------------|
| `await create_thought(...)` | `str` | Create a thought, returns UUID |
| `await get_thought(thought_id)` | `ThoughtRecord` | Retrieve by ID |
| `await update_thought(thought_id, **fields)` | `ThoughtRecord` | Update fields, returns updated record |
| `await list_thoughts(...)` | `list[ThoughtRecord]` | List with filters |
| `await delete_thought(thought_id)` | `None` | Hard delete |

##### `create_thought` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `thought_type` | `str` | Yes | `ThoughtType` enum value |
| `essence` | `str` | Yes | Short summary (1-200 chars) |
| `content` | `str` | Yes | Full text content |
| `priority` | `str` | Yes | `Priority` enum value |
| `source` | `str` | Yes | Origin identifier |
| `confidence` | `float \| None` | No | Confidence score (0.0-1.0) |
| `current_cycle` | `int` | No | Cycle number (default: 0) |

##### `list_thoughts` Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| `thought_type` | `str \| None` | Filter by type |
| `lifecycle_status` | `str \| None` | Filter by status |
| `priority` | `str \| None` | Filter by priority |
| `limit` | `int` | Max results (default: 100) |

#### Edge CRUD

| Method | Returns | Description |
|--------|---------|-------------|
| `await create_edge(from_id, to_id, edge_type, ...)` | `str` | Create edge, returns UUID |
| `await get_edge(edge_id)` | `EdgeRecord` | Retrieve by ID |
| `await get_edges(thought_id, direction)` | `list[EdgeRecord]` | Edges for a thought |
| `await delete_edge(edge_id)` | `None` | Hard delete |

#### Embedding Operations

| Method | Returns | Description |
|--------|---------|-------------|
| `await store_embedding(owner_id, vector, model, dim)` | `str` | Store embedding vector |
| `await get_embedding(owner_id)` | `EmbeddingRecord \| None` | Retrieve embedding |
| `await search_similar(vector, limit, threshold)` | `list` | Cosine similarity search |

#### Full-Text Search

| Method | Returns | Description |
|--------|---------|-------------|
| `await search_fts(query, limit)` | `list[ThoughtRecord]` | FTS5/BM25 text search |
| `await search_hybrid(query, vector, config)` | `list[HybridSearchResult]` | Combined search |

### `ReadOnlyEngrava`

Wrapper that raises `ReadOnlyViolationError` on any write operation.

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
| `consolidated_from` | `str \| None` | JSON source IDs |
| `visibility` | `ThoughtVisibility` | Access scope |
| `metadata` | `dict[str, MetadataValue]` | Caller-supplied structured attributes (default `{}`) |

#### `metadata` field

`MetadataValue = str | int | float | bool | None` — values must be flat
JSON-serializable scalars.  Nested containers (`dict`, `list`) are
rejected at `create_thought` / `update_thought` boundaries with a
`ValueError`.

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
| `source` | `str` | Origin |
| `decay_multiplier` | `float` | Decay rate multiplier |

### `EmbeddingRecord`

| Field | Type | Description |
|-------|------|-------------|
| `embedding_id` | `str` | UUID primary key |
| `owner_type` | `str` | `"thought"` or `"action"` |
| `owner_id` | `str` | Owning record ID |
| `model_name` | `str` | Embedding model identifier |
| `dimension` | `int` | Vector dimensionality |
| `vector_blob` | `bytes` | Raw numpy binary |

### `ActionRecord`

| Field | Type | Description |
|-------|------|-------------|
| `action_id` | `str` | UUID primary key |
| `source_thought_id` | `str` | Linked thought |
| `action_type` | `ActionType` | Action classification |
| `intent` | `str` | Description of intent |
| `status` | `ActionStatus` | Current status |
| `verification_status` | `VerificationStatus` | Verification state |

### `HybridSearchResult`

| Field | Type | Description |
|-------|------|-------------|
| `thought` | `ThoughtRecord` | The matched thought |
| `vector_score` | `float` | Vector similarity score |
| `fts_score` | `float` | FTS5/BM25 score |
| `recency_score` | `float` | Recency-based score |
| `combined_score` | `float` | Weighted composite |

## Enums

All enums are `StrEnum` — JSON-serializable and stored as strings.

| Enum | Values |
|------|--------|
| `ThoughtType` | `OBSERVATION`, `INSIGHT`, `BELIEF`, `GOAL`, `PLAN`, `MEMORY`, `HYPOTHESIS`, `EMOTION` |
| `Priority` | `P1`, `P2`, `P3`, `P4` |
| `LifecycleStatus` | `ACTIVE`, `COMPLETED`, `ARCHIVED`, `DORMANT` |
| `EdgeType` | `ASSOCIATION`, `CAUSATION`, `CONTRADICTION`, `REFINEMENT`, `TEMPORAL`, `SUPPORTS` |
| `ActionType` | `INTERNAL`, `EXTERNAL`, `QUERY`, `RESPONSE` |
| `ActionStatus` | `PENDING`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `ThoughtVisibility` | `PRIVATE`, `SELECTIVE`, `PUBLIC` |
| `KnowledgeSource` | `EXPERIENCE`, `SEEDED_LLM`, `DISTILLED_LLM` |
| `VerificationStatus` | `UNVERIFIED`, `VERIFIED`, `DISPUTED` |

## Exceptions

| Exception | Base | Description |
|-----------|------|-------------|
| `EngravaError` | `Exception` | Base for all engrava errors |
| `ThoughtNotFoundError` | `EngravaError` | Thought ID not found |
| `StaleDataError` | `EngravaError` | Concurrent modification detected |
| `InvalidTransitionError` | `EngravaError` | Invalid lifecycle state transition |
| `ReadOnlyViolationError` | `EngravaError` | Write attempt on read-only store |
| `EmbeddingModelMismatchError` | `EngravaError` | Embedding model mismatch on restore |

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

### `MindQLExecutor`

```python
from engrava import MindQLExecutor, MindQLResult

executor = MindQLExecutor(store)
result: MindQLResult = await executor.execute("FIND type=OBSERVATION LIMIT 10")
print(result.rows)       # list[dict]
print(result.row_count)  # int
```

### `parse`

```python
from engrava import parse, MindQLCommand

cmd: MindQLCommand = parse("FIND type=INSIGHT LIMIT 5")
print(cmd.verb)     # "FIND"
print(cmd.filters)  # {"type": "INSIGHT"}
print(cmd.limit)    # 5
```

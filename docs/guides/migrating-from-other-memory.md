# Migrating from another memory system

This guide helps you move an agent's memory from another store — a hosted
agent-memory service (mem0, Zep, …), a framework's built-in memory (LangChain,
…), or a plain vector database (Chroma, Qdrant, pgvector, …) — into Engrava.

It covers three things:

1. [Mapping concepts](#concept-mapping) from other systems onto Engrava's model.
2. [Porting your write/read calls](#porting-your-calls) with before/after snippets.
3. [Bulk-importing](#bulk-import) an existing corpus efficiently.

It ends with [filtering, scoping & multi-tenancy](#filtering-scoping-and-multi-tenancy)
— the one area where Engrava's defaults differ most from a hosted service, and
what to do about it.

Read [Core Concepts](../concepts.md) first if the terms *thought*, *edge*,
*cycle*, or *reflection* are unfamiliar, and [Positioning](../positioning.md)
to confirm Engrava is the right destination for your workload.

## Concept mapping

Other memory systems use different vocabulary for similar ideas. This table maps
common concepts onto Engrava:

| Concept elsewhere | Engrava equivalent | Notes |
|---|---|---|
| "Memory" / "record" / "document" | **`ThoughtRecord`** | The unit you store. Has `essence` (short) + `content` (full). |
| "Memory type" / "role" | **`thought_type`** (`OBSERVATION`, `BELIEF`, `TASK`, …) | A small fixed taxonomy; see [Core Concepts](../concepts.md). |
| Free-form metadata / `metadata={...}` | **`ThoughtRecord.metadata`** | An arbitrary JSON dict, persisted and round-tripped. |
| "User id" / "session id" / namespace | A key inside **`metadata`** (or `source`) | Engrava has no built-in tenant field — see [scoping](#filtering-scoping-and-multi-tenancy). |
| Relationship / link between memories | **`EdgeRecord`** (typed, weighted) | First-class graph; edges also feed ranking. |
| Embedding / vector | Stored on write only with `embedding_provider=...` **and** `auto_embed=True`; otherwise call `store_embedding(thought_id, vector)` yourself | See the [Embeddings guide](embeddings.md). |
| Vector / similarity search | **`search_similar(query_vector, …)`** | Needs a ready query vector. |
| Keyword / BM25 search | **`search_fts(query, …)`** | Returns `list[(thought_id, score)]`. |
| Hybrid search | **`search_hybrid(query_text, …)`** | Fuses FTS + vector + recency + priority + graph. |
| Automatic summarisation / fact extraction | *(none — by design)* | Engrava does no LLM-side extraction; see [Non-goals](../positioning.md#non-goals). |
| Decay / forgetting | TTL + lifecycle + the recency signal | See [Data lifecycle](../data-lifecycle.md) (TTL, archive-vs-delete, erasure) and the recency signal in [Search](../search.md). |
| Summaries of clusters | **`REFLECTION`** thoughts via [dreaming](../dreaming.md) | Structural (centroid + keywords), not LLM prose. |

## Porting your calls

The shapes below are illustrative fragments — they assume you already have a
`store` (see [Quick Start](../quickstart.md) for how to open one).

**Writing a memory.** Where another library takes a string and does extraction
for you, Engrava takes a fully-formed `ThoughtRecord` — you decide the type,
priority, and metadata:

```python
import uuid

from engrava import LifecycleStatus, Priority, ThoughtRecord, ThoughtType

# before (illustrative, another library):
#   memory.add("User prefers dark mode", user_id="u1")

# after (engrava):
await store.create_thought(
    ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence="User prefers dark mode",
        content="User prefers dark mode",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="chat",
        metadata={"user_id": "u1"},
    )
)
```

**Searching.** Where another library returns ranked memories from a single
`search`, pick the Engrava method that matches the signal you want; `search_hybrid`
is the closest analogue to a managed hybrid search:

```python
# before (illustrative):
#   hits = memory.search("what theme does the user like?", user_id="u1")

# after (engrava) — note: search is unscoped; filter by user yourself:
result = await store.search_hybrid("what theme does the user like?", top_k=10)
for thought_id, score in result.results:
    record = await store.get_thought(thought_id)
    if record is not None and record.metadata.get("user_id") == "u1":
        print(score, record.essence)
```

See [filtering, scoping & multi-tenancy](#filtering-scoping-and-multi-tenancy)
for why the post-filter is there and how to do it better.

## Bulk import

When migrating an existing corpus, insert under a single transaction instead of
committing once per row. The `suspend_auto_commit()` async context manager
defers the commit until the block exits — it **commits once on success and rolls
back the whole batch on any error**. Pair it with `deduplicate=True` so repeated
`content` collapses into one thought (bumping `confirmation_count`) instead of
inserting duplicate rows.

The following is a complete, runnable example (it uses an in-memory store and a
small fake export):

```python
import asyncio
import uuid

import aiosqlite

from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType

# Pretend this came from your previous memory system's export.
EXPORTED_MEMORIES = [
    {"text": "User prefers dark mode", "user": "u1"},
    {"text": "User is based in Berlin", "user": "u1"},
    {"text": "User prefers dark mode", "user": "u1"},  # a duplicate
    {"text": "Project deadline is Friday", "user": "u2"},
]


def to_thought(item: dict[str, str]) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence=item["text"][:200],
        content=item["text"],
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="import",
        metadata={"user_id": item["user"]},
    )


async def bulk_import(store, items: list[dict[str, str]]) -> int:
    # One transaction for the whole batch: commit on success, roll back on error.
    async with store.suspend_auto_commit():
        for item in items:
            # deduplicate=True collapses identical content into one thought.
            await store.create_thought(to_thought(item), deduplicate=True)
    return await store.count_thoughts()


async def main() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        total = await bulk_import(store, EXPORTED_MEMORIES)
        # 4 exported rows, one duplicate collapsed -> 3 stored.
        assert total == 3
        print(f"Imported {total} thoughts.")


if __name__ == "__main__":
    asyncio.run(main())
```

For large corpora, import in batches (e.g. a few thousand rows per
`suspend_auto_commit()` block) to keep each transaction short — long
transactions block the background SQLite thread (see
[Known Limitations](../known-limitations.md#aiosqlite-proxy-architecture)).
If you have embeddings configured, note that each new thought is embedded on
write (see the [Embeddings guide](embeddings.md)), so a bulk load pays the
embedding cost up front — pre-compute vectors or import in batches accordingly.
See the [Performance guide](../performance.md#write-throughput-and-bulk-ingest)
for the throughput levers in detail.

## Filtering, scoping & multi-tenancy

This is the most important difference from a hosted memory service. Engrava's
**`search_hybrid` / `search_similar` / `search_fts` take no scope or metadata
filter** — they rank across the entire store. There is no `user_id=` or
`session_id=` argument on the ranked path. You scope retrieval yourself, and
there are three patterns, with clear tradeoffs.

### Option A — over-fetch, then post-filter (simplest)

Ask for more results than you need, then drop the ones that don't match. Keep
the scope key in `metadata` when you write.

```python
# Want the top 5 for user "u1": over-fetch, then filter and trim.
result = await store.search_hybrid("dark mode", top_k=50)
scoped = []
for thought_id, score in result.results:
    record = await store.get_thought(thought_id)
    if record is not None and record.metadata.get("user_id") == "u1":
        scoped.append((thought_id, score))
    if len(scoped) >= 5:
        break
```

- **Pros:** no SQL, works with the high-level API, fine for modest stores.
- **Cons:** wasteful when one tenant is a small slice of a large store (you may
  over-fetch a lot, or miss matches if `top_k` is too small). Ranking is still
  computed over everything.

### Option B — one store per tenant (strongest isolation)

Give each tenant its own database file via
[`EngravaManager`](../api-reference.md). Each service has its own file and its
own lock, so retrieval is naturally scoped and tenants are physically isolated.

```python
from engrava import EngravaManager, load_config

config = load_config("engrava.yaml")
async with EngravaManager.from_config(config.services) as mgr:
    store_u1 = await mgr.get_store("u1")          # u1.db
    result = await store_u1.search_hybrid("dark mode", top_k=5)
```

- **Pros:** true isolation (separate files, separate locks, easy per-tenant
  backup/delete); search is scoped for free.
- **Cons:** not suitable for a very large number of tenants (one file each); no
  cross-tenant query. Best when tenants are coarse (a handful of services), not
  per-end-user at massive cardinality.

### Option C — pre-filter in raw SQL (scoped recall without over-fetch)

When you need keyword/metadata-scoped recall without over-fetching, query the
`thought` table directly. The Python `metadata` dict is persisted to a
`metadata_json` column you can index into with SQLite's `json_extract`:

```sql
-- thoughts for one user, most recent first
SELECT thought_id, essence
FROM thought
WHERE json_extract(metadata_json, '$.user_id') = :user_id
ORDER BY updated_cycle DESC
LIMIT 20;
```

Run it through the same connection you gave the store:

```python
cursor = await conn.execute(
    "SELECT thought_id, essence FROM thought "
    "WHERE json_extract(metadata_json, '$.user_id') = ? "
    "ORDER BY updated_cycle DESC LIMIT 20",
    ("u1",),
)
rows = await cursor.fetchall()
```

- **Pros:** exact scoping, no over-fetch; you can combine it with FTS by joining
  the `thought_fts` table.
- **Cons:** you drop below the high-level API to raw SQL against the schema, and
  this path does **not** apply the hybrid ranking signals (it is a filter, not a
  ranked search). Treat the schema as semi-stable and re-check it across upgrades.

### Choosing

| Situation | Use |
|---|---|
| Small/medium store, occasional scoping | **A** (over-fetch + post-filter) |
| A handful of coarse tenants needing real isolation | **B** (store per tenant) |
| Scoped recall over a large store, ranking not required | **C** (raw `json_extract`) |

> **Want a real filter on the ranked path?** Adding a scope/metadata argument to
> `search_*` is a public-API change under consideration, not a current feature.
> Until then, the patterns above are the supported approach.

## See also

- [Positioning](../positioning.md) — when Engrava fits, and its non-goals
- [Core Concepts](../concepts.md) — thoughts, edges, cycles, reflections
- [Recipes](../recipes/index.md) — short task-oriented snippets, incl. dedup
- [Known Limitations](../known-limitations.md) — concurrency and scale constraints

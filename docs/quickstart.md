# Quick Start

Get up and running with engrava in 5 minutes.

## Installation

```bash
pip install engrava
```

For vector search and the bundled walkthrough you also need a local
embedding encoder — install the `embeddings-local` extra:

```bash
pip install 'engrava[embeddings-local]'
```

The extra pulls `sentence-transformers` and `torch` and downloads a
small (~30-90 MB) encoder model on first use. The encoder is **not** a
language model: it turns text into a fixed-size vector. There are no
API keys, and there is no network traffic after the first download.
Engrava itself does not call any LLM at any time.

## Run the bundled walkthrough

The repository ships a single-file walkthrough that exercises the
end-to-end ingest → dream → query flow on a small demo dataset:

```bash
python examples/quickstart.py        # 5-minute end-to-end tour
```

`quickstart.py` boots an in-memory store, ingests a handful of
percepts (things the agent learned about the user) plus two
utterances (replies the agent already produced), runs one dreaming
consolidation cycle, and queries via hybrid search. The expected
top result for the shipped query is `My favorite color is teal.`.

### What is dreaming?

Dreaming is engrava's offline consolidation step. Between
interactions the engine reviews the thoughts that have proven
durable — confirmed and revisited over time — and groups related
memories into **REFLECTION** nodes: deterministic, structural
summaries that record which observations were grouped and the
keywords distilled from them. It uses no language model and
touches no network. Dreaming is deliberately conservative: a
brand-new store of one-off facts has nothing to consolidate yet;
REFLECTIONs emerge as memories accumulate and repeat over an
agent's lifetime.

### Self-anchored identity

Every thought carries structured metadata that pins its origin. The
package exposes three small helpers in `engrava.metadata`:

| Helper       | Use for                                              |
|--------------|------------------------------------------------------|
| `percept()`  | Input arriving from outside (user message, document) |
| `utterance()`| The agent's own output sent to the world             |
| `thought()`  | The agent's internal cognition (reflection, plan)    |

```python
from engrava import percept, utterance, thought

percept(source_id="user-42", label="user")
# -> {'perspective': 'percept', 'source': {'is_self': False, 'confidence': 'high', 'id': 'user-42', 'label': 'user'}, 'lang': 'en', 'content_type': 'natural_language'}

utterance()
# -> {'perspective': 'utterance', 'source': {'is_self': True, 'confidence': 'high'}, 'lang': 'en', 'content_type': 'natural_language'}
```

The helpers are pure functions: same arguments always return an equal
dictionary, and the returned value carries no shared state. Callers
who want a different shape are free to pass a literal dictionary
instead — the helpers exist to remove a class of typo-driven shape
mismatches at the call site.

### Seeing dreaming work

Dreaming's effect shows up on a store with accumulated, repeated
memories — not on a handful of one-off facts. To see it on a
representative workload, run the bundled synthetic benchmark:

```bash
python -m engrava.benchmarks.synthetic
```

It builds a multi-conversation corpus, runs consolidation, and
reports the REFLECTION coverage dreaming produces. See
[`benchmarks.md`](benchmarks.md) for how to read the numbers.

## Create a Store

```python
import asyncio
import aiosqlite
from engrava import SqliteEngravaCore

async def main() -> None:
    # SqliteEngravaCore wraps an open aiosqlite connection.
    # Use ":memory:" for experimentation, or a file path to persist.
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        print("Store ready!")

asyncio.run(main())
```

> The rest of this page assumes you are inside the `async with` block above,
> so `store` and `conn` are in scope. For a configuration-driven alternative,
> use `await SqliteEngravaCore.from_config("engrava.yaml")` (it opens and owns
> the connection for you).

## Add Thoughts

```python
import uuid
from engrava import ThoughtRecord, ThoughtType, Priority, LifecycleStatus

observation = ThoughtRecord(
    thought_id=str(uuid.uuid4()),
    thought_type=ThoughtType.OBSERVATION,
    essence="Python is great for AI agents",
    content="Python's async ecosystem and rich ML libraries make it ideal.",
    priority=Priority.P2,
    lifecycle_status=LifecycleStatus.ACTIVE,
    created_cycle=0,
    updated_cycle=0,
    source="human",
)
stored = await store.create_thought(observation)
print(f"Created thought: {stored.thought_id}")
```

> **About `created_cycle` / `updated_cycle`.** A *cycle* is a consumer-owned
> logical clock — Engrava never advances it for you. `0` is fine for this
> quickstart, but in a real long-running agent you should keep a counter and
> increment it once per turn, using it for these fields (and for `current_cycle`
> in search / consolidation). Otherwise recency can't tell old memories from new
> and dreaming's age gate never opens. See
> [Cycle (the agent clock)](concepts.md#cycle-the-agent-clock).

## Link Thoughts with Edges

```python
from engrava import EdgeRecord, EdgeType

belief = ThoughtRecord(
    thought_id=str(uuid.uuid4()),
    thought_type=ThoughtType.BELIEF,
    essence="SQLite provides zero-config persistence",
    content="WAL mode enables concurrent reads with single-writer safety.",
    priority=Priority.P2,
    lifecycle_status=LifecycleStatus.ACTIVE,
    created_cycle=0,
    updated_cycle=0,
    source="human",
)
await store.create_thought(belief)

edge = await store.create_edge(
    EdgeRecord(
        edge_id=str(uuid.uuid4()),
        from_thought_id=observation.thought_id,
        to_thought_id=belief.thought_id,
        edge_type=EdgeType.ASSOCIATED,
        weight=0.8,
        created_cycle=0,
    )
)
print(f"Linked thoughts via edge: {edge.edge_id}")
```

## Search

### Full-Text Search

```python
# search_fts returns (thought_id, bm25_score) tuples — fetch the record for fields.
for thought_id, score in await store.search_fts("Python AI", top_k=5):
    record = await store.get_thought(thought_id)
    if record is not None:
        print(f"  [{record.priority.value}] {record.essence}  (score={score:.3f})")
```

### Embedding Similarity Search

Use a real embedding provider so similarity is meaningful (this needs the
`embeddings-local` extra; see the [Embeddings guide](guides/embeddings.md) for
all provider options):

```python
from engrava import SentenceTransformerProvider

provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")

# Store an embedding for an existing thought
vector = await provider.embed(observation.content)
await store.store_embedding(
    observation.thought_id, vector, model_name=provider.model_name
)

# Search by similarity — returns (thought_id, score) tuples
for thought_id, score in await store.search_similar(vector, top_k=5):
    record = await store.get_thought(thought_id)
    if record is not None:
        print(f"  {record.essence}  (score: {score:.3f})")
```

> Tip: configure the provider on the store with `auto_embed=True` (or via
> `engrava.yaml`) and Engrava embeds thoughts on write — and embeds your query
> for you in `search_hybrid`. See the [Embeddings guide](guides/embeddings.md).

## Query with MindQL

```python
from engrava import MindQLExecutor, parse

# MindQLExecutor runs against an aiosqlite connection; parse the string first.
executor = MindQLExecutor(conn)

# Find observations
result = await executor.execute(
    parse("FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 10")
)
print(f"Found {len(result.rows)} thoughts")

# Count active thoughts
result = await executor.execute(
    parse("COUNT thoughts WHERE lifecycle_status = 'ACTIVE'")
)
print(f"Active thoughts: {result.count}")
```

## Use the CLI

```bash
# Database info
engrava --db my_thoughts.db info

# Run a MindQL query
engrava --db my_thoughts.db query "FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 5"

# Back up your data
engrava --db my_thoughts.db snapshot -o backup.jsonl

# Restore from backup
engrava --db my_thoughts.db restore -i backup.jsonl
```

## Next Steps

Build something next, then reach for the references:

- [Tutorial](tutorial.md) — build a small notes memory end to end (start here)
- [Recipes](recipes/index.md) — copy-paste snippets: store a turn, retrieve context, TTL, dedup, session scoping
- [Building a memory-backed agent](guides/agent-memory.md) — the full agent turn loop
- [Core Concepts](concepts.md) — the mental model (thought, edge, reflection, cycle)
- [Configuration](configuration.md) — YAML-based setup for production use
- [API Reference](api-reference.md) — full class and method reference
- [MindQL](mindql.md) — complete query language reference

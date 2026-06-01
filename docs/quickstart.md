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
from engrava import SqliteEngravaCore

async def main() -> None:
    # In-memory for experimentation, or provide a file path
    store = SqliteEngravaCore(":memory:")
    await store.ensure_schema()
    print("Store ready!")
    await store.close()

asyncio.run(main())
```

## Add Thoughts

```python
thought_id = await store.create_thought(
    thought_type="OBSERVATION",
    essence="Python is great for AI agents",
    content="Python's async ecosystem and rich ML libraries make it ideal.",
    priority="P2",
    source="human",
)
print(f"Created thought: {thought_id}")
```

## Link Thoughts with Edges

```python
insight_id = await store.create_thought(
    thought_type="INSIGHT",
    essence="SQLite provides zero-config persistence",
    content="WAL mode enables concurrent reads with single-writer safety.",
    priority="P2",
    source="human",
)

edge_id = await store.create_edge(
    from_thought_id=thought_id,
    to_thought_id=insight_id,
    edge_type="ASSOCIATION",
)
print(f"Linked thoughts via edge: {edge_id}")
```

## Search

### Full-Text Search

```python
results = await store.search_fts("Python AI", limit=5)
for thought in results:
    print(f"  [{thought.priority}] {thought.essence}")
```

### Embedding Similarity Search

```python
from engrava import CallbackProvider

# Use any embedding function
provider = CallbackProvider(
    callback=lambda text: [0.1] * 384,  # Replace with real embeddings
    dimension=384,
    model_name="my-model",
)

# Store embedding
vector = await provider.embed(thought.content)
await store.store_embedding(thought_id, vector, "my-model", 384)

# Search by similarity
similar = await store.search_similar(vector, limit=5)
for emb in similar:
    print(f"  {emb.owner_id} (score: {emb.score:.3f})")
```

## Query with MindQL

```python
from engrava import MindQLExecutor

executor = MindQLExecutor(store)

# Find observations
result = await executor.execute("FIND type=OBSERVATION LIMIT 10")
print(f"Found {len(result.rows)} thoughts")

# Count by status
result = await executor.execute("COUNT status=ACTIVE")
print(f"Active thoughts: {result.rows[0]['count']}")
```

## Use the CLI

```bash
# Database info
engrava --db my_thoughts.db info

# Run a MindQL query
engrava --db my_thoughts.db query "FIND type=INSIGHT LIMIT 5"

# Back up your data
engrava --db my_thoughts.db snapshot -o backup.jsonl

# Restore from backup
engrava --db my_thoughts.db restore -i backup.jsonl
```

## Next Steps

- [Configuration](configuration.md) — YAML-based setup for production use
- [Extensions](extensions.md) — Hook into the thought lifecycle
- [API Reference](api-reference.md) — Full class and method reference
- [MindQL](mindql.md) — Complete query language reference

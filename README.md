# engrava

> The memory database for AI agents.
>
> Graph memory, hybrid search, and a tamper-evident audit trail — one `pip install`, no server, no LLM.

[![CI](https://github.com/sovantica/engrava/actions/workflows/ci.yml/badge.svg)](https://github.com/sovantica/engrava/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/engrava.svg)](https://pypi.org/project/engrava/)
[![Python](https://img.shields.io/pypi/pyversions/engrava.svg)](https://pypi.org/project/engrava/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**engrava** is a standalone embedded database for AI agent memory. Built on
SQLite, it provides thought CRUD, edge-based knowledge graphs, embedding-based
similarity search, full-text search (FTS5/BM25), and a declarative extension
system — all in a single package with zero external service dependencies.

## Use Cases

- AI agent persistent memory
- Personal knowledge base
- Conversation storage with semantic search
- Research notes with associative linking
- Any application that needs a thought-graph with embeddings

## Quick Start

### Installation

```bash
pip install engrava
```

Optional extras:

```bash
pip install engrava[vec]               # sqlite-vec vector search backend
pip install engrava[embeddings-local]  # sentence-transformers embeddings
pip install engrava[embeddings-openai] # OpenAI-compatible embeddings
```

### Basic Usage

```python
import asyncio
from engrava import SqliteEngravaCore

async def main() -> None:
    store = SqliteEngravaCore(":memory:")
    await store.ensure_schema()

    # Create a thought
    thought_id = await store.create_thought(
        thought_type="OBSERVATION",
        essence="Python is great for AI",
        content="Python's async ecosystem makes it ideal for AI agents.",
        priority="P2",
        source="human",
    )

    # Retrieve it
    thought = await store.get_thought(thought_id)
    print(f"Stored: {thought.essence}")

    # Create an edge between thoughts
    second_id = await store.create_thought(
        thought_type="INSIGHT",
        essence="SQLite is underrated",
        content="SQLite provides ACID transactions with zero setup.",
        priority="P2",
        source="human",
    )
    await store.create_edge(thought_id, second_id, "ASSOCIATION")

    # Query with MindQL
    from engrava import MindQLExecutor
    executor = MindQLExecutor(store)
    result = await executor.execute("FIND type=OBSERVATION LIMIT 5")
    for row in result.rows:
        print(row)

    await store.close()

asyncio.run(main())
```

### Configuration-Driven Setup

```python
from engrava import load_config, SqliteEngravaCore

config = load_config("engrava.yaml")
store = SqliteEngravaCore(config.db_path)
await store.ensure_schema()
```

See [docs/configuration.md](docs/configuration.md) for the full YAML schema.

## Upgrading

Automatic schema migration runs on first connection. See the
[upgrade guide](docs/upgrade.md) for compatibility notes, backup guidance, and
troubleshooting steps.

## Features

### Thought CRUD

Create, read, update, and archive thoughts with full lifecycle management.
All models are frozen Pydantic objects — mutations happen via `evolve()`.

### Edge-Based Knowledge Graph

Link thoughts with typed, weighted edges. Supports association, causation,
contradiction, and custom edge types.

### Embedding Search

Store embeddings alongside thoughts and perform brute-force cosine similarity
search. Pluggable embedding providers:

| Provider | Extra | Backend |
|----------|-------|---------|
| `SentenceTransformerProvider` | `embeddings-local` | Local model via sentence-transformers |
| `OpenAICompatibleProvider` | `embeddings-openai` | Any OpenAI-compatible API |
| `OllamaProvider` | `embeddings-ollama` | Local Ollama server |
| `HuggingFaceProvider` | `embeddings-hf` | HuggingFace Inference API |
| `CallbackProvider` | *(built-in)* | Custom callable |

### Full-Text Search (FTS5)

SQLite FTS5 virtual table with BM25 ranking. Hybrid search combines vector
similarity, text relevance, and recency scoring.

### MindQL Query Language

Declarative query language for the thought-graph:

```
FIND type=OBSERVATION priority=P1 LIMIT 10
COUNT status=ACTIVE
SELECT thought_id, essence WHERE type=INSIGHT
```

Extensible with custom commands via the hook system.

### Extension System

Plug into the thought lifecycle via `EngravaHooksProtocol`:

```python
from engrava import EngravaHooksProtocol, ThoughtRecord, ScoringContext

class MyHooks(EngravaHooksProtocol):
    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        # Transform thoughts before persistence
        return thought

    async def score_function(
        self, thought: ThoughtRecord, context: ScoringContext
    ) -> float:
        # Custom relevance scoring
        return thought.confidence or 0.5
```

### Dreaming / Memory Consolidation

Built-in `DreamingExtension` for periodic memory consolidation — scores
thoughts via configurable signals, promotes high-value entries, and
(*v0.4.0*) creates **REFLECTION thoughts** by clustering semantically
related thoughts and computing centroid embeddings (no LLM required).

→ See [`docs/benchmarks.md`](docs/benchmarks.md) for reproducible
evidence (synthetic benchmark suite runnable in ~5 minutes).

### Multi-Service Isolation

Run multiple independent databases under one `EngravaManager`:

```python
from engrava import EngravaManager

async with EngravaManager(data_dir=Path("./data")) as mgr:
    agent_a = await mgr.get_store("agent-a")
    agent_b = await mgr.get_store("agent-b")
    # Completely isolated databases
```

## CLI

```bash
engrava --db mydata.db info          # Database stats
engrava --db mydata.db query "FIND type=OBSERVATION LIMIT 5"
engrava --db mydata.db snapshot -o backup.jsonl
engrava --db mydata.db restore -i backup.jsonl
engrava --db mydata.db gc            # Garbage-collect archived thoughts
engrava --db mydata.db migrate       # Ensure schema is up-to-date
engrava --db mydata.db export -o portable.json
```

`engrava info` now renders the same metrics snapshot contract exposed by
`await store.metrics()`.

## Architecture

- **SQLite** with WAL mode for concurrent reads
- **Frozen Pydantic models** — immutable domain objects
- **Async-first** — all I/O via `aiosqlite`
- **Hook-based extension** — zero monkey-patching
- **Template method pattern** — subclass `SqliteEngravaCore` for extended schemas
- **Zero external services** — everything runs locally in-process

## Documentation

- [Upgrade Guide](docs/upgrade.md) — compatibility matrix, backups, and troubleshooting
- [Quick Start](docs/quickstart.md) — 5-minute setup guide
- [Configuration](docs/configuration.md) — YAML config format and options
- [Extensions](docs/extensions.md) — Writing custom extensions and hooks
- [Observability](docs/observability.md) — Metrics snapshot API and event hooks
- [API Reference](docs/api-reference.md) — Full protocol and class reference
- [MindQL](docs/mindql.md) — Query language syntax and examples
- [Known Limitations](docs/known-limitations.md) — Platform notes and constraints

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/            # Lint
ruff format --check src/ tests/   # Format check
mypy --strict src/                # Type check
pytest --cov                      # Test with coverage
```

## License

MIT — see [LICENSE](LICENSE) for details.

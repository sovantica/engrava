# Engrava

> The memory database for AI agents.
>
> Graph memory, hybrid search, and a tamper-evident audit trail — one `pip install`, no server, no LLM.

[![CI](https://github.com/sovantica/engrava/actions/workflows/ci.yml/badge.svg)](https://github.com/sovantica/engrava/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/engrava.svg)](https://pypi.org/project/engrava/)
[![Python](https://img.shields.io/pypi/pyversions/engrava.svg)](https://pypi.org/project/engrava/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Engrava** is a standalone embedded database for AI agent memory. Built on
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
import uuid

import aiosqlite

from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType


async def main() -> None:
    # SqliteEngravaCore wraps an open aiosqlite connection.
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        # Build a ThoughtRecord and persist it with create_thought.
        observation = ThoughtRecord(
            thought_id=str(uuid.uuid4()),
            thought_type=ThoughtType.OBSERVATION,
            essence="Python is great for AI",
            content="Python's async ecosystem makes it ideal for AI agents.",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="human",
        )
        await store.create_thought(observation)

        # Retrieve it
        thought = await store.get_thought(observation.thought_id)
        if thought is not None:
            print(f"Stored: {thought.essence}")


asyncio.run(main())
```

From here, link thoughts with [typed edges](#edge-based-knowledge-graph),
query them with [MindQL](#mindql-query-language), or run the full
ingest → dream → search tour in the [Quick Start guide](docs/quickstart.md).

### Configuration-Driven Setup

```python
from engrava import SqliteEngravaCore

# from_config opens and OWNS the connection — use it as an async context manager.
async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    # The schema is already applied by from_config.
    thought = await store.get_thought("some-id")
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

Link thoughts with typed, weighted edges. Edge types include `ASSOCIATED`,
`DEPENDS_ON`, `DERIVED_FROM`, `CONSOLIDATED_FROM` (created by dreaming), and
`CONTESTED_BY`.

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
FIND thoughts WHERE thought_type = 'OBSERVATION' AND priority = 'P1' LIMIT 10
COUNT thoughts WHERE lifecycle_status = 'ACTIVE'
SELECT thought_id, essence FROM thought WHERE thought_type = 'BELIEF'
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
creates **REFLECTION thoughts** by clustering semantically related
thoughts and computing centroid embeddings (no LLM required). Available
since 0.3.0.

→ See [`docs/benchmarks.md`](docs/benchmarks.md) for reproducible
evidence (synthetic benchmark suite runnable in ~5 minutes).

### Tamper-Evident Audit Trail

Opt-in hash-chain **journal** that records every thought/edge mutation as a
SHA-256-linked, before/after entry — off by default, one config flag to enable.
Query history with `store.journal.get_entries(...)` and validate the chain with
`store.journal.verify_integrity()`.

→ See [`docs/audit-trail.md`](docs/audit-trail.md) for enabling, querying,
verification, and the security model (what "tamper-evident" does and does not
guarantee).

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

- [Core Concepts](docs/concepts.md) — the mental model (thought, edge, reflection, cycle, …) — start here
- [Positioning](docs/positioning.md) — when Engrava is (and isn't) the right tool, and how it compares
- [Quick Start](docs/quickstart.md) — 5-minute setup guide
- [Tutorial](docs/tutorial.md) — build a small notes memory end to end
- [Recipes](docs/recipes/index.md) — copy-paste snippets for common tasks (store a turn, retrieve context, TTL, dedup, …)
- [Building a memory-backed agent](docs/guides/agent-memory.md) — the end-to-end agent turn loop (ingest → retrieve → generate → consolidate)
- [Migrating from another memory system](docs/guides/migrating-from-other-memory.md) — concept mapping, porting calls, bulk import, and scoping/multi-tenancy
- [Embeddings](docs/guides/embeddings.md) — wiring a real embedding provider (local / OpenAI / Ollama / HuggingFace / custom)
- [Configuration](docs/configuration.md) — YAML config format and options
- [Upgrade Guide](docs/upgrade.md) — compatibility matrix, backups, and troubleshooting
- [Extensions](docs/extensions.md) — Writing custom extensions and hooks
- [Observability](docs/observability.md) — Metrics snapshot API
- [Audit Trail](docs/audit-trail.md) — Tamper-evident hash-chain journal (enabling, querying, verifying, security model)
- [API Reference](docs/api-reference.md) — Full protocol and class reference
- [MindQL](docs/mindql.md) — Query language syntax and examples
- [Troubleshooting](docs/troubleshooting.md) — symptom → cause → fix for common errors
- [FAQ](docs/faq.md) — quick answers (LLM/keys, embeddings-optional, scale, concurrency, backups, …)
- [Performance & Scaling](docs/performance.md) — the vector-backend switch, bulk-ingest, and dreaming cost at scale
- [Data Lifecycle & Retention](docs/data-lifecycle.md) — lifecycle states, TTL, archive-vs-delete, GDPR erasure, disk reclamation
- [Deployment](docs/deployment.md) — process model, database files on disk, containers, graceful shutdown
- [Concurrency](docs/concurrency.md) — the WAL single-writer model, busy timeout, and per-service isolation
- [Backup & Recovery](docs/backup-and-recovery.md) — WAL-safe backups, snapshot vs file copy, restore verification
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

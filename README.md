# Engrava

> The memory database for AI agents.
>
> Graph memory, hybrid search, and a tamper-evident thought/edge journal — one `pip install`, no server, no LLM.

[![CI](https://github.com/sovantica/engrava/actions/workflows/ci.yml/badge.svg)](https://github.com/sovantica/engrava/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/engrava.svg)](https://pypi.org/project/engrava/)
[![Python](https://img.shields.io/pypi/pyversions/engrava.svg)](https://pypi.org/project/engrava/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Engrava** is a standalone embedded database for AI agent memory. Built on
SQLite, it provides thought CRUD, edge-based knowledge graphs, embedding-based
similarity search, full-text search (FTS5/BM25), and a declarative extension
system — all in a single package with zero external service dependencies.

Benchmark results are published, and the runs are reproducible from a separate repository. Every
published run is Group A — `memory_pipeline_llms: []`, no language model anywhere in the memory
layer. That is a property of the architecture rather than a measurement, so it holds across releases.

- **[Benchmark results](https://engrava.ai/benchmarks/)** — the live table: every published row with the comparability segment it belongs to.
- **[engrava-benchmark](https://github.com/sovantica/engrava-benchmark)** — the runner. Clone it and reproduce a result against the package from PyPI. MIT.
- **[Discussions](https://github.com/sovantica/engrava/discussions)** — questions; reproduction questions belong in the Q&A category.

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
pip install 'engrava[vec]'                # sqlite-vec vector search backend
pip install 'engrava[embeddings-local]'   # sentence-transformers embeddings (local model)
pip install 'engrava[embeddings-openai]'  # OpenAI-compatible embeddings API
pip install 'engrava[embeddings-ollama]'  # Ollama local embeddings server
pip install 'engrava[embeddings-hf]'      # HuggingFace Inference API embeddings
```

Dreaming/consolidation and the knowledge graph need **no extra** — they are part
of the base install.

### Basic Usage

Store a memory and search for it in two calls — no IDs to generate, no record
to assemble:

```python
import asyncio

import aiosqlite

from engrava import SqliteEngravaCore


async def main() -> None:
    # SqliteEngravaCore wraps an open aiosqlite connection.
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        await store.remember("Python is great for AI agents")
        await store.remember("SQLite needs no server")

        result = await store.recall("what language is good for agents?")
        for thought_id, score in result.results:
            thought = await store.get_thought(thought_id)
            if thought is not None:
                print(f"{thought.essence}  (score: {score:.3f})")


asyncio.run(main())
```

`remember()` stores the text as a thought (generating its ID for you) and
returns the stored `ThoughtRecord`; `recall()` runs the same hybrid search as
`search_hybrid()` and returns the ranked results. For full control — setting
priority, thought type, metadata, or the cognitive cycle on a write — build a
`ThoughtRecord` yourself and call `create_thought()`.

From here, link thoughts with [typed edges](#edge-based-knowledge-graph),
query them with [MindQL](#mindql-query-language), or run the full
ingest → dream → search tour in the [Quick Start guide](https://github.com/sovantica/engrava/blob/main/docs/quickstart.md).

### Configuration-Driven Setup

```python
from engrava import SqliteEngravaCore

# from_config opens and OWNS the connection — use it as an async context manager.
async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
    # The schema is already applied by from_config.
    thought = await store.get_thought("some-id")
```

See [docs/configuration.md](https://github.com/sovantica/engrava/blob/main/docs/configuration.md) for the full YAML schema.

## Upgrading

Automatic schema migration runs on first connection. See the
[upgrade guide](https://github.com/sovantica/engrava/blob/main/docs/upgrade.md) for compatibility notes, backup guidance, and
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

Store embeddings alongside thoughts and search them with the built-in NumPy
cosine backend or the optional `sqlite-vec` backend (`pip install
'engrava[vec]'`). Pluggable embedding providers:

| Provider | Extra | Backend |
|----------|-------|---------|
| `SentenceTransformerProvider` | `embeddings-local` | Local model via sentence-transformers |
| `OpenAICompatibleProvider` | `embeddings-openai` | Any OpenAI-compatible API |
| `OllamaProvider` | `embeddings-ollama` | Local Ollama server |
| `HuggingFaceProvider` | `embeddings-hf` | HuggingFace Inference API |
| `CallbackProvider` | *(built-in)* | Custom callable |

### Full-Text Search (FTS5)

SQLite FTS5 virtual table with BM25 ranking. Hybrid search combines vector
similarity, text relevance, recency, priority, and graph connectivity. Signals
that cannot run for a query are skipped and the remaining weights are
redistributed.

### MindQL Query Language

Declarative query language for the thought-graph:

```
FIND thoughts WHERE thought_type = 'OBSERVATION' AND priority = 'P1' LIMIT 10
COUNT thoughts WHERE lifecycle_status = 'ACTIVE'
SELECT thought_id, essence FROM thought WHERE thought_type = 'BELIEF'
```

Extensible with custom commands through `MindQLExecutor` or an
`ExtensionManifest`; the lifecycle hook registry is reserved and is not
consulted by the core executor.

### Extension System

Plug into the thought lifecycle via `EngravaHooksProtocol`. Subclass
`DefaultEngravaHooks` when you only need selected active methods:

```python
from engrava import DefaultEngravaHooks, ThoughtRecord

class MyHooks(DefaultEngravaHooks):
    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        # Observe or enrich the object returned after persistence.
        return thought

    async def decay_function(
        self, thought: ThoughtRecord, elapsed_cycles: int
    ) -> float:
        # Supply a decay multiplier to an enabled Memory Hygiene pass.
        return 1.0
```

Core currently invokes `on_store`, `on_retrieve`, and `decay_function`.
`on_store` runs after the source thought is durable; changing its return value
does not rewrite the persisted row. `score_function` and
`mindql_extension_registry()` remain reserved protocol methods and are not
called by core.

### Dreaming / Memory Consolidation

Built-in `DreamingExtension` for periodic memory consolidation — scores
thoughts via configurable signals, promotes high-value entries, and
creates **REFLECTION thoughts** by clustering semantically related
thoughts and computing centroid embeddings (no LLM required). Available
since 0.3.0.

→ See [`docs/benchmarks.md`](https://github.com/sovantica/engrava/blob/main/docs/benchmarks.md) for reproducible
evidence (synthetic benchmark suite runnable in ~5 minutes).

### Forgetting / Memory Hygiene

The subtractive half of memory maintenance, paired with Dreaming: an **opt-in,
reversible**, no-LLM loop that **archives** cold, low-signal thoughts — and, as a
*separately* opted-in step, garbage-collects them only after both a cycle and a
wall-clock restore window. OFF by default; once enabled, archived thoughts drop out
of default retrieval and can be restored (`restore_thought` / `include_archived`).

→ See [`docs/memory-hygiene.md`](https://github.com/sovantica/engrava/blob/main/docs/memory-hygiene.md) for the
loop, protection, restore windows, and the honest deletion posture.

### Tamper-Evident Thought/Edge Journal

Opt-in hash-chain **journal** that records thought and edge mutations (plus
action `status`/`verification_status` transitions) as SHA-256-linked, before/after
entries — a tamper-evident thought/edge journal, **not** a whole-database audit
(embeddings and action *creation* are not covered). Off by default, one config flag to
enable. Query history with `store.journal.get_entries(...)` and validate the
chain with `store.verify_journal()`, which audits whatever chain is on disk
independent of the current `journal.enabled` state — the `store.journal` writer
handle does not exist while journaling is off, but the chain earlier sessions
wrote is still in `journal_entry` and still needs verifying.

→ See [`docs/audit-trail.md`](https://github.com/sovantica/engrava/blob/main/docs/audit-trail.md) for enabling, querying,
verification, and the security model (what "tamper-evident" does and does not
guarantee).

### Multi-Service Isolation

Run multiple independent databases under one `EngravaManager`:

```python
from pathlib import Path

from engrava import EngravaManager

async with EngravaManager(data_dir=Path("./data")) as mgr:
    agent_a = await mgr.get_store("agent-a")
    agent_b = await mgr.get_store("agent-b")
    # Completely isolated databases
```

### MCP Server

Want Engrava as a memory server for your agent? The MCP server ships as its own
package, **`engrava-mcp`** — a native stdio server (no HTTP shim) with read
tools, optional write tools, attachable `engrava://` resources, and guided
prompts, for any MCP client (Claude Desktop, Claude Code, Cursor, Windsurf,
VS Code):

```bash
uvx engrava-mcp        # or: pip install engrava-mcp
```

`engrava-mcp` pulls `engrava` in transitively, so installing it also gives you
the `import engrava` library. See the
[`engrava-mcp` package](https://github.com/sovantica/engrava-mcp) for install,
client configuration, the full tool/resource/prompt reference, and read-only
mode.

## CLI

```bash
engrava --db mydata.db info          # Database stats
engrava --db mydata.db query "FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 5"
engrava --db mydata.db snapshot -o backup.jsonl
engrava --db mydata.db restore -i backup.jsonl
engrava --db mydata.db gc            # Collect ARCHIVED thoughts + their edges/embeddings/actions
engrava --db mydata.db migrate       # Ensure schema is up-to-date
engrava --db mydata.db export -o portable.json
```

`gc` physically deletes `ARCHIVED` thoughts together with every edge touching one
on either end — including edges whose other end is still live — their embeddings
and the actions sourced from them, then reconciles the vector index by removing
every `vec0` row no `embedding` row owns; on a `vec0`-indexed store where
`sqlite-vec` cannot be loaded — most commonly because `engrava[vec]` is not
installed — a pass that is about to delete stops **before deleting anything** and
exits `1` rather than stranding those vectors in an index nothing can then reach
them through.

`engrava info` now renders the same metrics snapshot contract exposed by
`await store.metrics()`.

See the [CLI reference](https://github.com/sovantica/engrava/blob/main/docs/cli.md) for every command and option.

## Architecture

- **SQLite** with WAL mode for concurrent reads
- **Frozen Pydantic models** — immutable domain objects
- **Async-first** — all I/O via `aiosqlite`
- **Hook-based extension** — zero monkey-patching
- **Template method pattern** — subclass `SqliteEngravaCore` for extended schemas
- **Zero external services** — everything runs locally in-process

## Documentation

- [Documentation Index](https://github.com/sovantica/engrava/blob/main/docs/index.md) — choose a path by task: learn, build, operate, extend, or migrate
- [Core Concepts](https://github.com/sovantica/engrava/blob/main/docs/concepts.md) — the mental model (thought, edge, reflection, cycle, …) — start here
- [The Bi-temporal Model](https://github.com/sovantica/engrava/blob/main/docs/bitemporal.md) — the optional valid-time axis: query a fact as of any instant, `invalidate` without deleting
- [Positioning](https://github.com/sovantica/engrava/blob/main/docs/positioning.md) — when Engrava is (and isn't) the right tool, and how it compares
- [Quick Start](https://github.com/sovantica/engrava/blob/main/docs/quickstart.md) — 5-minute setup guide
- [Tutorial](https://github.com/sovantica/engrava/blob/main/docs/tutorial.md) — build a small notes memory end to end
- [Recipes](https://github.com/sovantica/engrava/blob/main/docs/recipes/index.md) — copy-paste snippets for common tasks (store a turn, retrieve context, TTL, dedup, …)
- [Building a memory-backed agent](https://github.com/sovantica/engrava/blob/main/docs/guides/agent-memory.md) — the end-to-end agent turn loop (ingest → retrieve → generate → consolidate)
- [Migrating from another memory system](https://github.com/sovantica/engrava/blob/main/docs/guides/migrating-from-other-memory.md) — concept mapping, porting calls, bulk import, and scoping/multi-tenancy
- [Embeddings](https://github.com/sovantica/engrava/blob/main/docs/guides/embeddings.md) — wiring a real embedding provider (local / OpenAI / Ollama / HuggingFace / custom)
- [MCP server (`engrava-mcp`)](https://github.com/sovantica/engrava-mcp) — expose a store to MCP clients (Claude Desktop, Claude Code, Cursor, Windsurf, VS Code) via the standalone server package: install, run, client config, tools/resources/prompts
- [Configuration](https://github.com/sovantica/engrava/blob/main/docs/configuration.md) — YAML config format and options
- [Upgrade Guide](https://github.com/sovantica/engrava/blob/main/docs/upgrade.md) — compatibility matrix, backups, and troubleshooting
- [Extensions](https://github.com/sovantica/engrava/blob/main/docs/extensions.md) — Writing custom extensions and hooks
- [Observability](https://github.com/sovantica/engrava/blob/main/docs/observability.md) — Metrics snapshot API
- [Audit Trail](https://github.com/sovantica/engrava/blob/main/docs/audit-trail.md) — Tamper-evident hash-chain journal (enabling, querying, verifying, security model)
- [Evidence and Conflicts](https://github.com/sovantica/engrava/blob/main/docs/evidence-and-conflicts.md) — model claims, provenance, contested facts, and caller-owned clarification
- [Security](https://github.com/sovantica/engrava/blob/main/docs/security.md) — trust boundaries, data egress, tenant isolation, encryption posture, and journal limits
- [Error Handling and Recovery](https://github.com/sovantica/engrava/blob/main/docs/error-handling.md) — decide when to retry, repair input, or replace a store
- [Benchmarks](https://github.com/sovantica/engrava/blob/main/docs/benchmarks.md) — release gates, versioned observed values, and LongMemEval
- [API Reference](https://github.com/sovantica/engrava/blob/main/docs/api-reference.md) — Full protocol and class reference
- [CLI Reference](https://github.com/sovantica/engrava/blob/main/docs/cli.md) — every `engrava` command and option
- [Glossary](https://github.com/sovantica/engrava/blob/main/docs/glossary.md) — quick definitions of every Engrava term
- [MindQL](https://github.com/sovantica/engrava/blob/main/docs/mindql.md) — Query language syntax and examples
- [Troubleshooting](https://github.com/sovantica/engrava/blob/main/docs/troubleshooting.md) — symptom → cause → fix for common errors
- [FAQ](https://github.com/sovantica/engrava/blob/main/docs/faq.md) — quick answers (LLM/keys, embeddings-optional, scale, concurrency, backups, …)
- [Performance & Scaling](https://github.com/sovantica/engrava/blob/main/docs/performance.md) — the vector-backend switch, bulk-ingest, and dreaming cost at scale
- [Data Lifecycle & Retention](https://github.com/sovantica/engrava/blob/main/docs/data-lifecycle.md) — lifecycle states, TTL, archive-vs-delete, GDPR erasure, disk reclamation
- [Deployment](https://github.com/sovantica/engrava/blob/main/docs/deployment.md) — process model, database files on disk, containers, graceful shutdown
- [Concurrency](https://github.com/sovantica/engrava/blob/main/docs/concurrency.md) — the WAL single-writer model, busy timeout, and per-service isolation
- [Backup & Recovery](https://github.com/sovantica/engrava/blob/main/docs/backup-and-recovery.md) — WAL-safe backups, snapshot vs file copy, restore verification
- [Known Limitations](https://github.com/sovantica/engrava/blob/main/docs/known-limitations.md) — Platform notes and constraints

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/            # Lint
ruff format --check src/ tests/   # Format check
mypy --strict src/                # Type check
pytest --cov                      # Test with coverage
```

## License

MIT — see [LICENSE](https://github.com/sovantica/engrava/blob/main/LICENSE) for details.

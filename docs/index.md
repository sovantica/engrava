# Engrava documentation

Engrava is an embedded, async Python memory database: one SQLite file, no
required server, and no LLM inside the core. These pages describe the v0.6.0
product line. When operating an older release, use the documentation at that
release's Git tag and read the [Upgrade Guide](upgrade.md) before opening its
database with a newer version.

## Requirements

- Python `>=3.11`; the maintained CI matrix covers Python 3.11, 3.12, and 3.13.
- A Python SQLite build with FTS5 support.
- No API key or network service for the base package.
- An optional local or remote embedding provider for semantic vector search;
  keyword, recency, priority, and graph signals work without one.

Start with `pip install engrava`. Platform and optional-backend constraints are
listed in [Known Limitations](known-limitations.md).

## Choose a path

| Goal | Start here | Continue with |
|---|---|---|
| Decide whether Engrava fits | [Positioning](positioning.md) | [Known Limitations](known-limitations.md) |
| Store and retrieve a first memory | [Quick Start](quickstart.md) | [Tutorial](tutorial.md), [Recipes](recipes/index.md) |
| Build an agent memory loop | [Building a memory-backed agent](guides/agent-memory.md) | [Core Concepts](concepts.md), [Search](search.md) |
| Model claims, evidence, and conflicts | [Evidence and conflicts](evidence-and-conflicts.md) | [Bi-temporal Model](bitemporal.md), [Audit Trail](audit-trail.md) |
| Configure embeddings | [Embeddings](guides/embeddings.md) | [Performance](performance.md) |
| Operate in production | [Deployment](deployment.md) | [Security](security.md), [Observability](observability.md), [Backup & Recovery](backup-and-recovery.md) |
| Handle failures | [Error handling and recovery](error-handling.md) | [Troubleshooting](troubleshooting.md), [Concurrency](concurrency.md) |
| Extend the store | [Extensions](extensions.md) | [Extension hooks](extension-hooks.md) |
| Migrate an existing memory system | [Migration guide](guides/migrating-from-other-memory.md) | [Scopes and isolation](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy) |

## Learn the model

- [Core Concepts](concepts.md) defines thoughts, edges, reflections, cycles,
  provenance, visibility, and reliability signals.
- [Glossary](glossary.md) is the compact definition index.
- [Bi-temporal Model](bitemporal.md) separates transaction time, valid time, and
  the consumer-owned cognitive cycle.
- [Evidence and conflicts](evidence-and-conflicts.md) explains what the graph can
  represent and what Engrava deliberately does not infer.
- [Data Lifecycle](data-lifecycle.md), [Dreaming](dreaming.md), and
  [Forgetting](memory-hygiene.md) cover memory maintenance.

## Build and retrieve

- [Quick Start](quickstart.md) and [Tutorial](tutorial.md) provide runnable
  introductions.
- [Recipes](recipes/index.md) contains focused copy-paste patterns.
- [Search](search.md) documents hybrid ranking, recency, scoping, collapse, and
  reflection participation.
- [Embeddings](guides/embeddings.md) covers local and remote providers, model
  identity, prefixes, and re-embedding.
- [MindQL](mindql.md) is the query-language guide.
- [CLI Reference](cli.md) covers operational commands.

## Operate

- [Deployment](deployment.md) defines connection ownership, process topology,
  containers, and graceful shutdown.
- [Concurrency](concurrency.md) explains SQLite WAL and the single-writer model.
- [Security](security.md) consolidates trust boundaries, data egress, tenant
  isolation, and journal limits.
- [Error handling and recovery](error-handling.md) maps failures to retry,
  repair, or store replacement.
- [Observability](observability.md) covers metrics, degradation counters, health
  checks, and alerts.
- [Backup & Recovery](backup-and-recovery.md), [Data Lifecycle](data-lifecycle.md),
  and [Upgrade Guide](upgrade.md) cover durability and release transitions.
- [Performance](performance.md) and [Benchmarks](benchmarks.md) separate workload
  tuning from product-quality evidence.

## Extend

- [Extensions](extensions.md) covers hooks, manifests, migrations, and custom
  MindQL commands.
- [Extension hooks](extension-hooks.md) defines lifecycle and derived-record
  capabilities, including structural splitting and backfill.
- [Architecture](architecture.md) describes the component and data-flow model.

## Reference

- [API Reference](api-reference.md) is authoritative for public classes,
  methods, models, exceptions, and top-level exports.
- [Configuration](configuration.md) is authoritative for YAML keys, defaults,
  validation, and runtime-only values.
- [Upgrade Guide](upgrade.md) is authoritative for schema transitions and
  operator sequencing. Architecture pages contain summaries only.
- [Known Limitations](known-limitations.md) is authoritative for supported
  boundaries and current constraints.
- [FAQ](faq.md) and [Troubleshooting](troubleshooting.md) provide short,
  task-oriented answers.

## Documentation versioning

The repository's default branch may describe the next release before its tag is
published. For reproducible operation, pair code and documentation from the same
Git tag. Observed benchmark values are versioned evidence, not timeless product
claims; see [Benchmarks](benchmarks.md).

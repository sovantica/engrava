# Positioning — what Engrava is (and isn't)

Engrava is a **standalone embedded database for AI-agent memory**. It is built on
SQLite and runs in-process: one `pip install`, no server, no LLM, no external
services. It gives an agent a durable thought-graph with hybrid retrieval
(full-text + vector + recency + priority + graph) and an optional tamper-evident
hash-chain journal.

This page explains **when Engrava is the right tool**, when it isn't, and how it
relates to the other memory options you might be choosing between.

## When Engrava is a good fit

- **You want memory you own and can inspect.** The whole store is one SQLite
  file. You can open it with any SQLite tool, back it up with a file copy
  ([with care around WAL](known-limitations.md#concurrent-write-safety)), and
  query it with SQL when the high-level API isn't enough.
- **You want retrieval, not just a vector index.** Engrava fuses FTS5/BM25,
  vector similarity, recency, priority, and a 1-hop graph signal into one ranked
  result. See [Search](search.md).
- **You want a graph, not a flat list.** Thoughts are connected by typed,
  weighted [edges](concepts.md), and the graph itself contributes to ranking.
- **You want it embedded.** No network hop, no service to operate, no separate
  process. It runs anywhere Python and SQLite run.
- **You want embeddings to be optional and pluggable.** Bring a local model, an
  OpenAI-compatible endpoint, Ollama, HuggingFace, or your own callback — or run
  with FTS-only and no embeddings at all. See the
  [Embeddings guide](guides/embeddings.md).
- **You want memory that maintains itself.** Engrava models both halves of memory
  maintenance: [Dreaming](dreaming.md) (consolidation) keeps and strengthens what
  matters, and [Forgetting](memory-hygiene.md) (opt-in, reversible memory hygiene)
  lets cold, low-signal memories fade rather than being kept indefinitely. The
  built-in mechanisms use no LLM; reproducibility requires fixed inputs and
  deterministic custom hooks/signals, and Forgetting additionally requires a
  fixed wall-clock `now`.
- **Small-to-medium corpora.** The default backend brute-forces vector search in
  Python and works well up to roughly 100k embeddings; beyond that, switch to
  the `sqlite-vec` backend. See
  [Known Limitations](known-limitations.md#sqlite-vec-pre-v1-status).

## When Engrava is *not* a good fit

- **You need a managed, horizontally-scaled vector service.** Engrava is a local
  embedded library, not a clustered database. One store is one SQLite file
  written by one process. If you need sharding, replication, or a multi-writer
  service across many machines, use a dedicated vector database.
- **You need more than one process writing the same store.** Only one store may
  write a database file; any number may read it. WAL lets readers and that one
  writer coexist, and one process can drive many async tasks against the store —
  but two writers on one file is unsupported, not merely contended. See
  [Concurrency](concurrency.md#multiple-stores-one-database-file).
- **You want the library to call an LLM for you.** Engrava does no LLM-side fact
  extraction, summarisation, or entity resolution (see [Non-goals](#non-goals)).
  It stores and retrieves what you give it; your agent decides what to write.
- **You need real per-tenant retrieval *isolation* out of the box.**
  `search_hybrid`/`recall` now take optional `filters=`/`visibility=` to *scope* a
  ranked query (project, session, visibility class) — but that is a query filter,
  **not** a security boundary. True cross-tenant isolation is a store per tenant
  (`EngravaManager`), not this filter; see the
  [migration guide's scoping section](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy).

## Non-goals

These are deliberate boundaries, not missing features:

- **No LLM-side intelligence.** Engrava core and built-in Dreaming do not call a
  language model. They do no fact extraction, no summarisation, no entity
  resolution, no semantic
  contradiction detection, no truth arbitration, and no automatic "memory
  writing" from raw text. Those belong in your agent (or a downstream
  extension), above the storage layer. The one consolidation feature that
  *does* synthesise — [dreaming](dreaming.md) — is purely structural
  (clustering + centroids + keyword counts), with **no LLM involved**. See
  [Evidence and conflicts](evidence-and-conflicts.md) for the explicit graph
  pattern available in core.
- **Retrieval is unscoped by default.** Queries rank across the whole store unless
  you scope them: `search_hybrid`/`recall` accept optional `filters=`/`visibility=`
  (a query filter, not access control); `search_similar`/`search_fts` take no filter
  argument. Real per-tenant isolation is a store per tenant (`EngravaManager`) —
  see the [scoping section](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy).
- **Not a distributed system.** No clustering, replication, or cross-machine
  consistency. One file, one writer.
- **Not an application framework.** Engrava is the memory layer. It does not
  provide an agent runtime, tool-calling, or prompt orchestration.

## How it compares

A rough orientation, not a feature scorecard. Evaluate the specifics against
your own workload.

| | Engrava | Hosted agent-memory services (e.g. mem0, Zep) | Framework memory (e.g. LangChain memory) | Standalone vector DBs (e.g. Chroma, Qdrant, pgvector) |
|---|---|---|---|---|
| **Deployment** | Embedded library, one SQLite file, in-process | Typically a hosted/managed service or self-hosted server | In-process, tied to the framework | Separate database/service (some have embedded modes) |
| **Retrieval model** | Hybrid: FTS + vector + recency + priority + graph, fused | Varies; often vector + recency with managed pipelines | Usually buffer/window or a vector-store wrapper | Primarily vector similarity (some add keyword/hybrid) |
| **Graph** | First-class typed/weighted edges that feed ranking | Some offer entity/graph memory | Generally no | Generally no |
| **LLM-side extraction** | None — you decide what to store | Often built in (auto fact-extraction/summarisation) | Sometimes, via chains | None |
| **External services** | None required | Usually yes | Depends on the chosen store | Usually a running service |
| **Audit trail** | Optional tamper-evident hash-chain journal | Varies | No | Generally no |
| **Best for** | Owning a local, inspectable, hybrid memory graph for an agent | Offloading memory ops to a managed pipeline | Quick memory inside an existing framework app | Large-scale pure vector retrieval |

If you are currently using one of these and want concept mappings and porting
snippets, see
[Migrating from another memory system](guides/migrating-from-other-memory.md).

## See also

- [Core Concepts](concepts.md) — the mental model behind thoughts, edges, and cycles
- [Search](search.md) — how the hybrid ranking actually works
- [Known Limitations](known-limitations.md) — the hard constraints in one place
- [Migrating from another memory system](guides/migrating-from-other-memory.md)

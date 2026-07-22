# FAQ

Short answers to the questions that come up most. For "something is broken" see
[Troubleshooting](troubleshooting.md); for "is this the right tool" see
[Positioning](positioning.md).

## Does Engrava call an LLM? Do I need an API key?

No. Engrava never calls a language model and needs no API key to run. It stores
and retrieves what your agent gives it; deciding *what* to remember (extraction,
summarisation) is your agent's job, above the storage layer. The one feature
that synthesises new thoughts — [dreaming](dreaming.md) — is purely structural
(clustering, centroids, keyword counts), with no LLM involved. See
[Non-goals](positioning.md#non-goals).

An API key is only relevant if **you** choose a remote embedding provider (e.g.
an OpenAI-compatible endpoint) — and that's for embeddings, not for any
Engrava-side reasoning. See the [Embeddings guide](guides/embeddings.md).

## Does it need network access or any running service?

No. Engrava is an embedded library built on SQLite — one `pip install`, runs
in-process, no server, no network. The only time network is involved is if you
configure a remote embedding provider yourself.

## Are embeddings required?

No. Without an embedding provider, search runs on FTS5/BM25 (keyword), priority,
and recency signals — semantic vector matching is simply skipped. Add a provider
(local or remote) when you want semantic retrieval. See the
[Embeddings guide](guides/embeddings.md). Note that storing on write only embeds
when you set both `embedding_provider=...` **and** `auto_embed=True`.

## How large a corpus can it handle?

The default vector backend brute-forces cosine similarity in Python, which works
well up to roughly **100k embeddings**. Beyond that, install the `sqlite-vec`
backend (`pip install engrava[vec]`, then `extensions.vector.backend:
sqlite-vec`) for indexed vector search. FTS5 scales well independently. SQLite
itself has been exercised here into the multi-GB / millions-of-thoughts range.
See [Known Limitations](known-limitations.md#sqlite-vec-pre-v1-status).

## Can multiple processes or tasks use the same store at once?

A single process can drive **many async tasks** against one store safely —
aiosqlite serialises them on its background thread, and WAL mode lets readers and
a single writer coexist. SQLite is **single-writer**, so heavy concurrent writes
from **multiple processes** are out of scope. For multi-tenant isolation, give
each tenant its own database file via `EngravaManager` (each has its own lock).
See [Known Limitations → Concurrent Write Safety](known-limitations.md#concurrent-write-safety)
and the [migration guide's scoping section](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy).

## How do I scope search to one user or session?

The `search_*` methods are **unscoped by default** — they take no `user_id` /
`session_id` filter and rank across the whole store. Scope it yourself with one
of three patterns: over-fetch + post-filter, one store per tenant via
`EngravaManager`, or a raw-SQL pre-filter on `metadata_json` with `json_extract`.
The tradeoffs are laid out in the
[scoping section](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy).

## When should I enable dreaming?

Enable [dreaming](dreaming.md) when memory **accumulates over time** and you want
the store to surface and link what matters: it promotes important thoughts to P1,
builds associative edges, and clusters related thoughts into
[`REFLECTION`](concepts.md) summaries. It is not useful on a tiny or write-once
store. Run it periodically (every N cycles, a cron
job, or manually) — never on the hot CRUD path. For single-write batch ingest,
keep `allow_zero_confirmation=True` or nothing will ever pass the confirmation
gate. See the agent loop's
[consolidation cadence](guides/agent-memory.md) pattern.

## What is a "cycle" and do I have to manage it?

A cycle is a **consumer-owned monotonic logical clock** — your agent's tick.
Engrava never advances or persists it for you; you pass `current_cycle` into
search and consolidation. It drives the recency signal and the dreaming age gate.
On restart, recover it from `max(created_cycle)` in the store.

Two ways to get it wrong have different effects: passing `current_cycle=None`
(the `search_hybrid` default) makes the recency signal **inactive** — it is
dropped from the ranking. Passing a **constant** (e.g. always `0`, never
advancing `created_cycle`/`updated_cycle`) keeps recency *active but useless* —
every thought's age collapses to the same value, so nothing looks more recent
than anything else, and the dreaming age gate (`min_age_cycles`) never opens.
Advance the cycle each turn. See
[Core Concepts → Cycle](concepts.md) and the related
[Troubleshooting entry](troubleshooting.md#dreaming-promotes-nothing-consolidation-is-inert).

## How do I back up the database safely?

Because Engrava uses WAL mode, a naive copy of just the `.db` file can miss
in-flight data in the `-wal` file. Use a WAL-safe approach — checkpoint then
copy, `VACUUM INTO`, or SQLite's backup API. Note that a logical snapshot does
**not** include the audit journal. See [Upgrade Guide](upgrade.md) for the
current backup guidance.

## Is the audit trail tamper-proof?

It is **tamper-evident**, not tamper-proof. The journal is a keyless in-file
SHA-256 hash chain: it reliably detects accidental corruption and naive edits or
truncation, but a write-capable actor who rewrites the whole file and recomputes
the chain is out of its threat model. Treat it as integrity evidence with OS
file permissions and periodic off-box verification, not as a cryptographic
guarantee against a privileged attacker. It is **off by default**
(`journal.enabled: false`). See [Audit Trail](audit-trail.md).

## Is Engrava production-ready?

Engrava is published on PyPI and maintained to a strict quality bar (typed,
linted, high test coverage). For production, the things to plan are the same as
for any embedded SQLite system: pick the right vector backend for your corpus
size, respect the single-writer model, set up WAL-safe backups, and (if you need
it) enable and monitor the audit trail. The [Known Limitations](known-limitations.md)
page is the honest list of constraints to design around.

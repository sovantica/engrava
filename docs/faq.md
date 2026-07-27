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

## Does Engrava encrypt the database file?

Not by itself. The core writes an ordinary SQLite database, so protect it with
filesystem permissions and, when at-rest encryption is required, an encrypted
disk or volume managed by your operating environment. A remote embedding
provider also receives the text it embeds; choose a local provider when that
egress is unacceptable. See [Security](security.md).

## Are embeddings required?

No. Without an embedding provider, search runs on FTS5/BM25 (keyword), priority,
and recency signals — semantic vector matching is simply skipped. Add a provider
(local or remote) when you want semantic retrieval. See the
[Embeddings guide](guides/embeddings.md). Note that storing on write only embeds
when you set both `embedding_provider=...` **and** `auto_embed=True`.

## How large a corpus can it handle?

The default vector backend brute-forces cosine similarity in Python, which works
well up to roughly **100k embeddings**. Beyond that, install the `sqlite-vec`
backend (`pip install 'engrava[vec]'`, then `extensions.vector.backend:
sqlite-vec`) for its compact compiled `vec0` storage and lower constant factor.
The pinned sqlite-vec 0.1.x backend still performs an **exhaustive linear KNN
scan**, not an approximate or sub-linear vector index, so measure your own p95
latency rather than treating 100k as a hard boundary. FTS5 scales well
independently. See [Performance](performance.md#the-brute-force-ceiling-and-how-to-pass-it)
and [Known Limitations](known-limitations.md#sqlite-vec-pre-v1-status).

## Can multiple processes or tasks use the same store at once?

**Tasks: yes, with limits.** Share one store across the tasks in your event
loop — aiosqlite serialises their statements on its background thread and WAL
lets readers and a single writer coexist. What engrava does not do is make a
read-modify-write atomic, so two tasks editing the *same field* of the same row
lose one of the two writes, silently. Editing *different* fields is safe — unless
one of the two passes `updated_cycle=`, which trips the version guard and gets
the other update rejected with `StaleDataError` even though they share no field.
Serialise the edits yourself when tasks genuinely compete for a row. See
[Concurrency](concurrency.md#many-async-tasks-one-store).

**Processes: no.** Only one store may *write* a given database file; any number
may read it. This is not a contention trade-off you can tune away — the locks
that order engrava's own operations stop at the store instance, so a second
writer loses updates and can duplicate deduplicated content. For multi-tenant or
multi-worker setups, give each writer its own database file via `EngravaManager`
(each has its own lock).

See [Concurrency](concurrency.md) for the full contract,
[Known Limitations → Concurrent Write Safety](known-limitations.md#concurrent-write-safety)
for the short version, and the
[migration guide's scoping section](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy)
for the isolation options.

## How do I scope search to one user or session?

Ranked retrieval is **unscoped by default**, but `search_hybrid(...)` and
`recall(...)` accept typed `filters=` and `visibility=` arguments that constrain
eligible rows before each ranking arm's limit. Put your user/session identifier
in thought metadata and query it with `filters=`; use `visibility=` for the
public-or-mine presentation pattern. These are query capabilities, not access
control: the caller can omit or forge them, so use one database per tenant via
`EngravaManager` when you need a security boundary. The direct low-level
`search_fts(...)` and `search_similar(...)` methods do not expose public metadata
filters. See [Scoped retrieval](search.md#scoped-retrieval) and the
[multi-tenancy tradeoffs](guides/migrating-from-other-memory.md#filtering-scoping--multi-tenancy).

## How does Engrava handle conflicting facts?

Engrava does not infer entities, detect semantic contradictions, choose a true
claim, or overwrite one side automatically. Conflicting thoughts can coexist.
The caller must establish that a conflict exists. Your application can attach
provenance and valid-time bounds, then record a directional `CONTESTED_BY` edge
under a documented convention and retrieve both sides for review or
clarification. Recording an explicit or rule-detected conflict needs no LLM;
open-ended contradiction detection may use an optional LLM above the store. See
[Evidence and conflicts](evidence-and-conflicts.md).

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
Engrava never advances or persists it for you; pass `current_cycle` explicitly or
configure a runtime `cycle_provider` for read/eligibility paths. It drives
cognitive-cycle recency and the dreaming age gate. On restart, recover its
high-water mark with `await store.max_cycle()`, which considers thought updates
and edge creation cycles rather than only thought creation.

Recency is inactive only when a query has no explicit `current_cycle`, no
explicit transaction-time `recency_now`, and no configured cycle provider. An
explicit `recency_now` suppresses a passive provider; supplying both explicit
references raises `RecencyModeConflictError`. Passing a **constant** cognitive
cycle (for example always `0`, while never advancing
`created_cycle`/`updated_cycle`) keeps that axis active but useless: every
thought's age collapses to the same value, and the dreaming age gate never opens.
Advance the cycle each turn when your application has a meaningful cadence; use
`recency_now` when it does not. See
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
SHA-256 hash chain: it detects changed retained journal rows and broken links
when the affected hashes are not recomputed. It cannot detect self-consistent
tail removal without an external high-water mark/tail anchor, and it does not
reconcile live thought or edge rows against journal history. It covers entry
**ordering and content**, not entry **timestamps**: neither `created_at` nor
`entry_id` is in the hash preimage, so a journal with every timestamp rewritten
still verifies. A rewrite across a `get_entries(since=...)` lower bound also
changes that window either way — downward hides an entry from it, upward plants
one inside it — so a time-bounded query is no safer than the timestamps it reads.
A write-capable
actor who rewrites the file and recomputes the chain is also out of scope. Treat
it as integrity evidence with OS file permissions and periodic off-box
checkpointing, not as a cryptographic guarantee against a privileged attacker.
It is **off by default** (`journal.enabled: false`). See
[Audit Trail](audit-trail.md).

## Is Engrava production-ready?

Engrava is published on PyPI and maintained to a strict quality bar (typed,
linted, high test coverage). For production, the things to plan are the same as
for any embedded SQLite system: pick the right vector backend for your corpus
size, respect the single-writer model, set up WAL-safe backups, and (if you need
it) enable and monitor the audit trail. The [Known Limitations](known-limitations.md)
page is the honest list of constraints to design around.

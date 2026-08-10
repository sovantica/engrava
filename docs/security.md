# Security and Trust Boundaries

Engrava is an embedded memory database, not an authentication gateway. It runs
inside the caller's Python process and stores data in SQLite. This page defines
the security boundary of Engrava v0.6 and the controls an application or
operator must provide around it.

For vulnerability reporting and disclosure timelines, see the repository
[Security Policy](../SECURITY.md). This guide covers deployment and data trust;
it does not replace that reporting policy.

## Default posture: local and embedded

The core library opens a local SQLite database and exposes an in-process Python
API. It does not start a network listener or require an external service. With
no embedding provider or custom extension configured, normal store operations
require no network access.

Network access can still enter the deployment through components selected by
the caller:

- remote or remotely addressed embedding providers;
- custom hooks, manifest factories, and MindQL handlers;
- application, service, or MCP wrappers around the library;
- dependency or model acquisition performed by an optional provider.

Treat those components as separate trust decisions. The fact that Engrava core
is embedded does not make arbitrary extensions or provider endpoints local.

## Data at rest

Engrava v0.6 uses ordinary SQLite through `aiosqlite`. It does **not** provide
built-in database encryption, encryption keys, or field-level encryption. The
database can contain thought text, metadata, provenance, actions, embeddings,
and audit-journal deltas. In WAL mode, recent data may also be present in the
adjacent `-wal` file; the `-shm` file belongs to the same operational file set.
Logical snapshots are JSONL data and are not encrypted by Engrava either.

Protect the entire database directory, not only the main `.db` file:

- run Engrava under a dedicated service identity and restrict directory access
  to that identity;
- set an appropriately restrictive process umask before the database or
  service directory is created;
- use an OS-, filesystem-, or volume-level encryption facility when stored
  memory requires encryption at rest;
- protect snapshot files, physical backups, temporary copies, and exported
  logs under the same policy as the live database;
- avoid network filesystems for a live WAL database; use a suitable local or
  properly configured persistent volume.

Engrava needs read/write access to the directory so SQLite can create and
manage the `-wal` and `-shm` files. Engrava does not configure encrypted
volumes or manage their keys. See [Deployment](deployment.md) for file-layout
and permission details.

## Embedding providers and data egress

Embedding generation sends the input text to the configured provider. That can
include newly stored thought content, re-embedded content, and search queries.
The resulting vectors are persisted locally and are sensitive derived data;
they should receive the same storage protection as the source text.

Provider boundaries in v0.6 are:

| Provider | Data path |
|---|---|
| `sentence-transformer` | Embedding inference is local after the model is available. Model-loading dependencies may acquire model files according to their own cache and network configuration. |
| `openai-compatible` | Sends text to the configured HTTP API; the default endpoint is OpenAI. |
| `huggingface` | Sends text to the Hugging Face Inference API. |
| `ollama` | Sends text to the configured Ollama endpoint; the default is loopback, but `base_url` can point elsewhere. |

Before enabling a remotely addressed provider:

- confirm the endpoint's retention, training, residency, and incident-response
  terms for the data you send;
- use TLS for non-loopback endpoints and do not assume a custom `base_url` is
  secure merely because the provider name is trusted;
- apply outbound network policy so the process can reach only approved hosts;
- avoid sending secrets or regulated content unless the endpoint is approved
  for it.

For supported remote providers, YAML accepts a whole-value environment
reference such as `api_key: "${EMBEDDING_API_KEY}"`. Prefer runtime secret
injection over a literal credential in a tracked configuration file. Engrava
resolves the value into process memory; it is not a secret manager and does not
rotate, scope, or revoke credentials. The built-in OpenAI-compatible provider's
HTTP error paths do not include the request authorization header, but application
logging and third-party clients remain part of the surrounding threat model.

## Tenant isolation and authorization

`filters=`, `visibility=`, thought `visibility`, metadata ownership fields, and
`ProvenanceContext.actor_id` are **not authorization controls**. They are
caller-supplied data or query refinements. A caller can omit a filter, forge
metadata, use another API, or access SQLite directly if it already has the
connection or file.

Engrava core performs no authentication, principal validation, row-level
authorization, or write-policy enforcement. For strong cross-tenant isolation,
use a separate database file per tenant or security domain, for example through
`EngravaManager`, and enforce the tenant-to-store mapping in trusted application
code before returning a store handle. Also isolate filesystem credentials and
backups where the threat model requires it. A database-per-tenant layout limits
accidental cross-querying; it does not protect against a process identity that
can read every tenant file.

If a shared corpus is required, authorization belongs in a trusted service
layer that mediates every read and write. Do not rely on a convention that each
caller will always pass the correct visibility filter.

## Extensions, hooks, and migrations

Extensions are trusted code. A configured hook class is dynamically imported
and instantiated in the Engrava process. Manifest factories, MindQL handlers,
CLI commands, and discovered entry points also execute Python from installed
packages. Hooks can inspect records passed to them, modify returned records,
perform I/O, and contact the network.

An extension manifest may declare SQL migrations. Engrava validates migration
shape, rejects transaction-control statements, applies each migration under a
savepoint, and records filenames and checksums to detect later historical
drift. A savepoint makes each migration file and its bookkeeping atomic; a
later-file failure does not roll back earlier files that were already committed.
Those checks provide consistency, but they do **not** establish publisher
identity or make the SQL safe to accept from an untrusted package. A migration
executes against the store connection and can change stored schema or data.

Use the extension boundary deliberately:

- install extensions only from a reviewed and trusted source;
- pin and verify package versions through the deployment's supply-chain
  controls;
- prefer explicit manifest paths for library-created stores;
- review new hook behavior and every pending migration before deployment;
- back up the database before applying core or extension migrations.

Discovery boundaries differ between the library and CLI:

- `SqliteEngravaCore` does not scan entry points by itself. Library discovery
  happens only when the caller invokes `discover_manifests()` or configuration
  sets `manifests.discover: true`.
- root `engrava --help` and resolution of an otherwise unknown CLI command scan
  `engrava.cli`; built-in commands resolve without that scan;
- `engrava query` additionally scans `engrava.extensions` to register MindQL
  commands from discovered manifests.

The CLI scans load Python entry-point objects even though they do not apply the
manifests' schema migrations. Use global `--no-extensions`, or set
`ENGRAVA_DISABLE_EXTENSIONS=1`, to prevent loading from **both** `engrava.cli`
and `engrava.extensions`, including for help, built-ins, and `query`. This
control does not override explicit manifest configuration used by library code.
Installing an extension package into an environment where discovery is enabled
grants that package code execution in the CLI process. Use a dedicated virtual
environment containing only reviewed packages, or use the Python API with an
explicit manifest allow-list when this trust is unacceptable.

See [Extension Hooks](extension-hooks.md), [Extensions](extensions.md), and the
[Upgrade Guide](upgrade.md) for the execution contracts.

## Snapshot and restore trust

`engrava restore` validates known record types, column sets, required values,
and value types, and applies the restore transactionally. This prevents a
malformed record from leaving a partial restore and keeps snapshot-controlled
column names out of generated SQL. It does not attest the origin or truth of the
content.

Use logical restore for snapshots whose origin and integrity you trust. Text and
metadata from an untrusted snapshot become untrusted stored content and may
later be returned to an agent, passed to an LLM, or sent to an embedding
provider. Review external snapshots as imports, including for instruction-like
or otherwise adversarial content. Restore only from a completed file that is no
longer being modified.

A logical snapshot excludes the audit journal, so restoring one starts with an
empty journal. A physical backup preserves the journal but is tied to the
database file and must be captured with a WAL-safe method. See [Backup &
Recovery](backup-and-recovery.md) for the supported boundary and procedures.

## Audit journal threat model

The optional journal is a keyless SHA-256 hash chain stored in the same SQLite
file as the records it describes. It can detect accidental corruption and
unsophisticated edits that do not recompute the chain. It is not a signature,
an HMAC, or cryptographic non-repudiation.

An actor who can rewrite the database can alter journal entries and recompute
all subsequent hashes. The journal also covers a defined mutation subset, not
the whole database: thought and edge mutations plus action status transitions;
embeddings, action creation, and read-derived access telemetry are outside the
chain.

Within the entries it does cover, the chain binds **ordering and content**
(`sequence_number`, `parent_hash`, `mutation_type`, `target_id`, `delta`) but
**not timestamps**: neither `created_at` nor `entry_id` is in the hash preimage.
Rewriting every `created_at` produces a fully backdated trail that still
verifies. Moving a timestamp across a `get_entries(since=...)` lower bound also
changes that window in either direction — downward removes an entry from an audit
window, upward inserts one into it — so a time-bounded audit query cannot
compensate: it reads the same unprotected column. Treat journal timestamps as
informative and anchor any time-bounded claim on a `(sequence_number,
entry_hash)` pair captured externally.

The preimage joins those five fields with `|` and no escaping, writing an absent
field as the empty string. Their separation therefore rests on what the fields
themselves can hold — a decimal `sequence_number`, one of seven fixed
`mutation_type` literals, a SHA-256 hex `parent_hash`, and a `delta` that is
always a JSON object dump whose quoting a caller-chosen `target_id` cannot
imitate — rather than on the encoding. Those grammars hold **for the entries the
store emits**; they are not enforced by `JournalWriter.append()`, which is
exported, so a caller writing entries through it directly can produce two
distinct entries with one preimage and one hash. That takes in-process code
execution against your own store, which this threat model already treats as past
the boundary — but read the binding as a property of the store's field grammars,
not as one the format guarantees. Verification also re-serialises the stored `delta`
before hashing it, so the chain binds that delta's decoded **value** and not its
stored bytes: a rewrite of the blob's key order, whitespace, or escaping that
preserves the value verifies clean. See
[Audit Trail → security model](audit-trail.md#security-model--guarantees).

For a stronger control:

- enable scheduled verification or `journal.verify_on_open` where its linear
  startup cost is acceptable;
- restrict database write access to the intended writer;
- periodically export the latest journal `entry_hash` to an append-only, WORM,
  signed, or independently controlled system;
- compare the verified local chain tail with that external anchor during audits
  and incident response.

Engrava v0.6 does not publish or manage an off-box anchor for you. See [Audit
Trail](audit-trail.md) for exact coverage and verification APIs.

## Backup, retention, and erasure

Security controls must follow all copies of the data. Use a WAL-safe physical
backup or the documented logical snapshot flow, encrypt backup storage where
required, restrict restore privileges, and test recovery using non-production
destinations.

Archiving is reversible retention, not erasure. A hard delete removes the live
row but audited content can remain in journal deltas, old copies can remain in
backups, and freed SQLite pages do not shrink the file until a rebuild such as
`VACUUM`. An erasure process must account for the live database, the journal,
snapshots and physical backups, replicas or exports, and the retention policy of
any remote provider that received the text. Purging journal entries breaks the
chain and requires an explicit re-baseline if journal verification is retained.

One residue is version-dependent and easy to miss: **on a database still below
core schema 12 a hard delete leaves the thought's `embedding` row behind**. The
delete does purge that thought's own `vec0` vector, so the index is clean
immediately afterwards — but the reconcile on the next sqlite-vec-enabled open
backfills it from the surviving `embedding` row, and from then on the deleted
**identifier** — not its content, which is gone — is returned by a vector query
that carries no effective metadata predicate on an active sqlite-vec backend.
That is an existence signal about a record an erasure request asked you to
remove, and checking the index straight after the delete will not reveal it.
`engrava migrate` closes it and purges the orphans that had already accumulated.
See
[Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated).

See [Data Lifecycle](data-lifecycle.md) for the complete retention and erasure
procedure.

## Service and MCP boundaries

Engrava core does not expose a network service and therefore does not implement
service authentication, transport encryption, request authorization, rate
limits, or internet-facing input policy. The MCP server is a separate package,
and any HTTP, RPC, worker, or agent-tool wrapper is outside the core trust
boundary.

The wrapper or deployment must authenticate callers, authorize every operation,
map principals to the correct store, protect transport, constrain request size
and concurrency, handle quotas and abuse, and avoid exposing raw database paths
or unrestricted store handles. Network exposure should be private by default
and enabled only with controls appropriate to the environment.

## Deployment checklist

- [ ] Use a dedicated runtime identity and restrict the full database directory.
- [ ] Place the database, WAL files, snapshots, and backups on approved encrypted
      storage when encryption at rest is required.
- [ ] Select local or approved remote embedding providers and restrict outbound
      destinations.
- [ ] Inject provider credentials at runtime; do not commit literal keys.
- [ ] Enforce authentication and authorization outside Engrava core.
- [ ] Use a separate database per tenant or security domain where isolation is
      required.
- [ ] Install only reviewed hooks and extensions; disable CLI entry-point loading
      with `--no-extensions` or `ENGRAVA_DISABLE_EXTENSIONS=1` when it is not needed.
- [ ] Back up before migrations and use a WAL-safe backup method.
- [ ] Treat restored content as trusted only when its source is trusted.
- [ ] Enable and verify the journal only with its keyless, partial-coverage
      threat model understood; anchor the chain externally when required.
- [ ] Include journal residue, backups, exports, and provider retention in
      erasure procedures.
- [ ] Apply transport, authorization, rate-limit, and abuse controls in every
      service or MCP layer that exposes Engrava.

## Related documentation

- [Deployment](deployment.md)
- [Backup & Recovery](backup-and-recovery.md)
- [Data Lifecycle](data-lifecycle.md)
- [Audit Trail](audit-trail.md)
- [Configuration](configuration.md)
- [Known Limitations](known-limitations.md)

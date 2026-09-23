"""Layer 5 of the documentation-example tests — the no-silent-gap census.

The compile layer (Layer 2) proves every fenced ``python`` block is valid
Python and names no phantom API, but compilation is a *floor*, not proof that
an example works. This module raises that floor to a hard guarantee: **every**
documentation block must be accounted for as exactly one of

* **(E) executed** — a self-contained script or a composable page-run that is
  actually run against the installed package by
  ``test_docs_examples_execute.py`` (its allowlists ``EXECUTABLE_BLOCKS`` /
  ``SYNC_EXECUTABLE_BLOCKS`` / ``CONCATENATED_PAGES`` are the source of truth
  here);
* **(B) behaviour-asserted** — a fragment (it assumes an existing
  ``store``/``conn``) whose behavioural claim is MIRRORED and asserted by a
  Layer-3 test in ``test_docs_examples_behavior.py``; it is listed in
  ``BEHAVIOUR_BLOCKS`` below; or
* **(C) compile-only** — a block that is not executed or behaviour-asserted, for
  reasons that differ in kind. Some genuinely cannot run standalone (a partial
  ``...`` snippet missing imports a sibling block on the same page supplies, a
  fragment needing a live external provider / network / on-disk artifact, an
  intentionally-invalid anti-pattern, or a duplicate whose behaviour is
  asserted elsewhere). Others could run cleanly on their own — a self-contained
  class or ``Protocol`` definition compiles and executes without error, it
  simply is never invoked within its own block — but running a definition
  nobody calls would assert nothing beyond compile, so it stays compile-only by
  choice, not by necessity. Not every ``CompileOnlyReason`` member even makes
  the "genuinely cannot run" claim: see its docstring in ``_md_blocks.py`` for
  the ones that do not. Each such block is registered in ``COMPILE_ONLY``
  **with an explicit reason**, so the compile-only status is a deliberate,
  auditable decision — never a silent gap.

The exhaustiveness test then asserts that the three registries partition the
discovered blocks: every block is classified, and no block is classified twice.
A future example that is added to the docs but is neither executed,
behaviour-asserted, nor registered as compile-only fails this suite — which is
the whole point: **no example may be silently un-covered beyond compile.**

Each registry entry binds a block by ``(markdown_path, anchor_substring)`` — the
same robust, line-number-independent anchoring the execute layer uses. An anchor
must appear in exactly one block *within its file*; the tests below enforce that,
so a moved/edited block surfaces as a loud failure (and a reminder to re-verify
the example) rather than drifting out of coverage.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from tests.docs._md_blocks import (
    REPO_ROOT,
    CodeBlock,
    CompileOnlyReason,
    all_python_blocks,
    extract_python_blocks,
)
from tests.docs.test_docs_examples_execute import (
    CONCATENATED_PAGES,
    EXECUTABLE_BLOCKS,
    FIXTURE_EXECUTED_BLOCKS,
    SYNC_EXECUTABLE_BLOCKS,
)

# (E) is derived from the execute layer's allowlists — see EXECUTABLE_BLOCKS,
# SYNC_EXECUTABLE_BLOCKS, CONCATENATED_PAGES, and FIXTURE_EXECUTED_BLOCKS in
# test_docs_examples_execute.py.
# Do not duplicate them.

# (B) Fragments whose behavioural claim is mirrored + asserted in
# test_docs_examples_behavior.py. Each pins a specific 0.5.0-era (or core) API
# claim: return shape, count, or value against the shipped code.
BEHAVIOUR_BLOCKS: tuple[tuple[str, str], ...] = (
    # api-reference.md
    ("docs/api-reference.md", "Short summary (1-200 chars)"),  # create_thought -> record
    ("docs/api-reference.md", "a lossless round-trip"),  # restore_thought
    ("docs/api-reference.md", 'retrieval_query="remote work trade-offs"'),  # ProvenanceContext
    ("docs/api-reference.md", "provenance_filter=MetadataFilter"),  # provenance filter
    (
        "docs/api-reference.md",
        "keep up to 2 rows per turn instead of 1",
    ),  # scoped + collapse recall
    (
        "docs/api-reference.md",
        "advance the STORED action through its lifecycle",
    ),  # action lifecycle
    ("docs/api-reference.md", 'search_hybrid("query text"'),  # hybrid result shape
    ("docs/api-reference.md", 'metadata = percept(source_id="user-1"'),  # percept() dict
    ("docs/api-reference.md", "store.execute_mindql"),  # execute_mindql
    ("docs/api-reference.md", "an aiosqlite.Connection, not a store"),  # MindQLExecutor find/count
    ("docs/api-reference.md", "query.command)     # MindQLCommand.FIND"),  # parse() fields
    # audit-trail.md
    ("docs/audit-trail.md", "Tampering or corruption detected at sequence"),  # verify_journal
    ("docs/audit-trail.md", "assert [e.mutation_type for e in entries]"),  # entries + verify
    # data-lifecycle.md
    ("docs/data-lifecycle.md", "excludes expired"),  # count_thoughts(include_expired)
    ("docs/data-lifecycle.md", "result.strategy_applied"),  # cleanup_expired result
    # dreaming.md
    ("docs/dreaming.md", "promote_threshold=0.55"),  # run_consolidation
    ("docs/dreaming.md", "store.consolidate(current_cycle=1)"),  # consolidate()
    ("docs/dreaming.md", "store.attach_dreaming_extension(ext)"),  # attach_dreaming_extension()
    # extension-hooks.md
    ("docs/extension-hooks.md", "class RecencyBoostHooks"),  # hooks protocol + score
    # extensions.md
    ("docs/extensions.md", "def mindql_extension_registry"),  # MyHooks protocol
    ("docs/extensions.md", "STATS_COMMAND = MindQLExtension"),  # custom command end-to-end
    ("docs/extensions.md", "candidates_limit=100"),  # DreamingExtension run
    ("docs/extensions.md", "class ImportanceSignal:"),  # custom signal
    # guides/embeddings.md
    ("docs/guides/embeddings.md", "query_prefix="),  # asymmetric role prefixes
    # guides/migrating-from-other-memory.md
    (
        "docs/guides/migrating-from-other-memory.md",
        "async def bulk_import_and_derive",
    ),  # bulk import derives nothing; derive_existing backfills it
    # memory-hygiene.md
    ("docs/memory-hygiene.md", "never auto-archived or auto-GC'd"),  # pinned
    ("docs/memory-hygiene.md", "preview.would_evict"),  # run_hygiene
    # mindql.md
    ("docs/mindql.md", "raw_sql="),  # raw SELECT
    ("docs/mindql.md", "EXPLAIN FIND thoughts"),  # EXPLAIN
    ("docs/mindql.md", "MindQLParseError as exc"),  # parse fields + error
    # observability.md
    ("docs/observability.md", "print(metrics.thoughts.total)"),  # metrics snapshot
    # quickstart.md
    ("docs/quickstart.md", 'percept(source_id="user-42"'),  # percept/utterance/thought
    # search.md
    ("docs/search.md", "priority_weight=0.05"),  # all weight params
    ("docs/search.md", "deployment runbook"),  # filters + visibility
    ("docs/search.md", "Composite unit key"),  # composite collapse
    ("docs/search.md", "Keep up to 2 rows of each unit"),  # collapse_max_per_unit
    # troubleshooting.md
    ("docs/troubleshooting.md", 'ThoughtType("BELIEF")'),  # enum member access
    ("docs/troubleshooting.md", "one endpoint is missing"),  # ReferentialIntegrityError
)

# (C) Blocks that are not executed or behaviour-asserted, each paired with a closed-
# vocabulary reason and an explicit free-text note. Each entry is ``(markdown_path,
# anchor_substring, reason, note)``, where ``reason`` is a ``CompileOnlyReason`` member
# (the count that matters -- see the tally asserted by
# ``test_compile_only_reasons_are_tallied`` below) and ``note`` is a required, non-
# empty free-text string saying what the member cannot: which exact API, which
# specific test mirrors the same claim on a runnable fixture. This is the auditable
# ledger of every deliberate compile-only decision: a block here is covered by the
# compile + phantom-API guards (Layer 2), and where a note cites a behaviour test,
# that test asserts the same API on a runnable mirror.
COMPILE_ONLY: tuple[tuple[str, str, CompileOnlyReason, str], ...] = (
    (
        "README.md",
        'from_config("engrava.yaml") as store:',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml; illustrative from_config wiring",
    ),
    (
        "README.md",
        "class MyHooks(DefaultEngravaHooks):",
        CompileOnlyReason.DEFINITION_ONLY,
        "class-definition-only hooks impl; conformance asserted in the hooks test",
    ),
    (
        "README.md",
        "EngravaManager(data_dir=Path(",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk data_dir and an unimported Path; illustrative multi-store",
    ),
    (
        "docs/api-reference.md",
        "schema already applied by from_config",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "illustrative setup with a `...` placeholder and from_config wiring",
    ),
    (
        "docs/api-reference.md",
        "from_thought_id=src_id",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined src_id/dst_id; edge creation asserted in the CRUD behaviour test",
    ),
    (
        "docs/api-reference.md",
        "consolidated_member_ids(reflection_id)",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined reflection_id; illustrative reflection-graph traversal",
    ),
    (
        "docs/api-reference.md",
        "for record in many_records",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined many_records; suspend_auto_commit runs in the migrating example",
    ),
    (
        "docs/api-reference.md",
        "Raises ReadOnlyViolationError",
        CompileOnlyReason.REQUIRES_SPECIALLY_CONFIGURED_STORE,
        "needs a store wrapped in ReadOnlyEngrava; a plain store does not raise here",
    ),
    (
        "docs/api-reference.md",
        "await mgr.list_services()",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk data_dir; illustrative manager list/delete usage",
    ),
    (
        "docs/api-reference.md",
        '"turn_index": 5',
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "ThoughtRecord/ThoughtType/Priority/LifecycleStatus unimported in this block",
    ),
    (
        "docs/api-reference.md",
        "class EmbeddingProviderProtocol(Protocol)",
        CompileOnlyReason.DEFINITION_ONLY,
        "Protocol-definition only",
    ),
    (
        "docs/upgrade.md",
        "def dimension(self) -> int:",
        CompileOnlyReason.DEFINITION_ONLY,
        (
            "method fragment; a conformant provider's search path is asserted in "
            "test_embedding_providers.TestProviderMissingRequiredMember"
        ),
    ),
    (
        "docs/api-reference.md",
        "class DerivedRecordProducerProtocol(Protocol)",
        CompileOnlyReason.DEFINITION_ONLY,
        "Protocol-definition only",
    ),
    (
        "docs/api-reference.md",
        "async def search_hybrid(",
        CompileOnlyReason.DEFINITION_ONLY,
        "bare search_hybrid signature (`...`); result shape asserted in the behaviour test",
    ),
    (
        "docs/error-handling.md",
        "async def create_with_partial_state",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "helper-function def taking a thought record; partial-persistence recovery illustrative",
    ),
    (
        "docs/error-handling.md",
        "async def repair_embedding",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "helper-function def assuming a store + provider; embedding repair illustrative",
    ),
    (
        "docs/error-handling.md",
        "async def store_atomically",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "helper-function def taking two thought records and an edge; illustrative batching",
    ),
    (
        "docs/error-handling.md",
        "async def replace_and_reconcile",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "helper-function def assuming a store; from_config reconnection illustrative",
    ),
    (
        "docs/extension-hooks.md",
        "-> Sequence[DerivedRecord]: ...",
        CompileOnlyReason.DEFINITION_ONLY,
        "bare derive_records signature fragment; conformance asserted in the seam tests",
    ),
    (
        "docs/extension-hooks.md",
        "hooks=StructuralSplitProducer(),",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database file; derived-records wiring illustrative",
    ),
    (
        "docs/extension-hooks.md",
        "class SentenceSplitter(DefaultEngravaHooks):",
        CompileOnlyReason.DEFINITION_ONLY,
        "class-definition-only producer; the shipped StructuralSplitProducer is behaviour-tested",
    ),
    (
        "docs/extension-hooks.md",
        "store.derive_existing(thought_id)",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "fragment assuming a store; derive_existing asserted in the derived-records backfill tests",
    ),
    (
        "docs/audit-trail.md",
        "journaling is active",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml; illustrative from_config wiring",
    ),
    (
        "docs/audit-trail.md",
        'connect("engrava.db") as conn:',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database file; journal wiring illustrative",
    ),
    (
        "docs/audit-trail.md",
        'target_id="thought-001"',
        CompileOnlyReason.REQUIRES_SPECIALLY_CONFIGURED_STORE,
        "needs a journal_enabled=True store; get_entries asserted in the journal test",
    ),
    (
        "docs/audit-trail.md",
        "JournalIntegrityError as exc",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "`...` placeholder; needs a corrupted on-disk journal; illustrative handling",
    ),
    (
        "docs/concurrency.md",
        "async with edit_lock:",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        (
            "fragment assuming a store and a caller-owned lock; the read-modify-write "
            "window it closes is asserted in the concurrency-contract tests"
        ),
    ),
    (
        "docs/concurrency.md",
        "PRAGMA busy_timeout",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database file; illustrative connection tuning",
    ),
    (
        "docs/concurrency.md",
        'get_store("tenant_a")',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml and per-service db files; illustrative",
    ),
    (
        "docs/concurrency.md",
        "write_lock_acquire_timeout_seconds=900",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        (
            "fragment assuming a store, connection, and a caller-defined slow "
            "embedding provider; illustrative constructor tuning, not runnable as-is"
        ),
    ),
    (
        "docs/configuration.md",
        'get_thought("abc")',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml; illustrative load_config wiring",
    ),
    (
        "docs/configuration.md",
        "resolve_embedding_provider(config.embeddings)",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml; illustrative provider resolution",
    ),
    (
        "docs/configuration.md",
        'get_store("main")',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml; illustrative manager wiring",
    ),
    (
        "docs/deployment.md",
        "Hold this store for the lifetime",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml and an undefined run_app; illustrative skeleton",
    ),
    (
        "docs/deployment.md",
        "connection closed here",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "contains a `...` placeholder; illustrative lifecycle/close",
    ),
    (
        "docs/deployment.md",
        "the caller owns and closes the connection",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database and contains `...`; illustrative lifecycle",
    ),
    (
        "docs/extension-hooks.md",
        "RECENT_COMMAND = MindQLExtension",
        CompileOnlyReason.DEFINITION_ONLY,
        "handler + command definition only; execution asserted via the STATS test",
    ),
    (
        "docs/extension-hooks.md",
        'extensions={"RECENT": RECENT_COMMAND}',
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "RECENT_COMMAND defined in a sibling block; execution asserted via the STATS test",
    ),
    (
        "docs/extension-hooks.md",
        "def test_my_hooks_satisfy_protocol",
        CompileOnlyReason.DEFINITION_ONLY,
        "illustrative unit-test snippet; conformance asserted in the hooks test",
    ),
    (
        "docs/extensions.md",
        "on_store / on_retrieve now run during CRUD",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database; invocation asserted via hooks/STATS tests",
    ),
    (
        "docs/extensions.md",
        'executor = MindQLExecutor(conn, extensions={"STATS": STATS_COMMAND})',
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        (
            "STATS_COMMAND defined in a sibling block; execution asserted end-to-end "
            "in the behaviour test"
        ),
    ),
    (
        "docs/extensions.md",
        "mindql_extensions=[],",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "references on-disk migration files and MyHooks; illustrative manifest",
    ),
    (
        "docs/extensions.md",
        "package_root override (test fixtures)",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "references on-disk migration files; illustrative manifest path options",
    ),
    (
        "docs/extensions.md",
        "are now applied",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database with on-disk migrations; illustrative",
    ),
    (
        "docs/extensions.md",
        "class ExtendedStore(SqliteEngravaCore)",
        CompileOnlyReason.DEFINITION_ONLY,
        "subclass-definition-only (overrides a private method); illustrative",
    ),
    (
        "docs/guides/agent-memory.md",
        'connect("agent-memory.db")',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database and undefined my_embed_fn; illustrative",
    ),
    (
        "docs/guides/agent-memory.md",
        "async def retrieve_context",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "helper-function def with undefined my_embed_fn; search_hybrid asserted elsewhere",
    ),
    (
        "docs/guides/agent-memory.md",
        "reply = await my_llm(prompt)",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "references an undefined my_llm; illustrative prompt assembly",
    ),
    (
        "docs/guides/agent-memory.md",
        'intent="answered user"',
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined percept_thought; action creation asserted in the action test",
    ),
    (
        "docs/guides/agent-memory.md",
        "async def store_utterance",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "helper-function def assuming a store; utterance metadata asserted in the metadata test",
    ),
    (
        "docs/guides/agent-memory.md",
        "cycle += 1",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "illustrative loop skeleton (undefined running)",
    ),
    (
        "docs/guides/agent-memory.md",
        "inside the loop, after advancing the cycle:",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined cycle/store; run_consolidation asserted in the dreaming tests",
    ),
    (
        "docs/guides/embeddings.md",
        'SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")\n# Abbreviated',
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "loads a real ST model (offline in CI) and opens a real db; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "provider wired from config, auto_embed honoured",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml; illustrative from_config wiring",
    ),
    (
        "docs/guides/embeddings.md",
        "async def strict_ingest",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        (
            "helper-function def taking a generic `provider: object`; no live "
            "provider required, but the block never supplies or invokes one"
        ),
    ),
    (
        "docs/guides/embeddings.md",
        "async def batch_ingest",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        (
            "helper-function def taking a generic `provider: object` (embed_batch); "
            "no live provider required, but the block never supplies or invokes one"
        ),
    ),
    (
        "docs/guides/embeddings.md",
        "batch_size=32,",
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "constructs a provider that loads a real model; illustrative config",
    ),
    (
        "docs/guides/embeddings.md",
        "point at any compatible API",
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "requires an OpenAI-compatible endpoint + key/network; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "base_retry_delay_s=0.5,",
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "requires an OpenAI-compatible endpoint; illustrative retry config",
    ),
    (
        "docs/guides/embeddings.md",
        "default Ollama address",
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "requires a running Ollama server; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "HuggingFaceProvider(",
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "requires an HF token + network; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "the length your callback returns",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined my_embed_fn; CallbackProvider asserted in the quickstart search test",
    ),
    (
        "docs/guides/embeddings.md",
        "the query text is embedded for you",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "assumes a configured provider + store; hybrid search asserted in behaviour tests",
    ),
    (
        "docs/guides/embeddings.md",
        "required — no auto-embed here",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "assumes a real provider + store; search_similar asserted in the quickstart test",
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        'get_store("u1")  # u1.db',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml + per-user db files; illustrative",
    ),
    (
        "docs/known-limitations.md",
        "result.backends_used  # preferred",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "single-line assertion fragment; backends_used asserted in the hybrid tests",
    ),
    (
        "docs/memory-hygiene.md",
        'from_config("engrava.yaml") as store:\n        result = await store.run_hygiene',
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml; run_hygiene asserted in the hygiene test",
    ),
    (
        "docs/observability.md",
        "engrava_search_p99_ms",
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "requires optional prometheus_client and a file-backed store; illustrative export",
    ),
    (
        "docs/performance.md",
        "the vector backend",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "requires an on-disk engrava.yaml + vector backend; illustrative",
    ),
    (
        "docs/performance.md",
        "async def bulk_load",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        (
            "helper-function def taking a collection of items; suspend_auto_commit "
            "runs in the migrating example"
        ),
    ),
    (
        "docs/quickstart.md",
        "WAL mode enables concurrent reads",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "fragment assuming a store; create_edge asserted in the CRUD behaviour test",
    ),
    (
        "docs/quickstart.md",
        "Store an embedding for an existing thought",
        CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE,
        "loads a real ST model (offline in CI); asserted in the quickstart search test",
    ),
    (
        "docs/recipes/index.md",
        "expire this thought one hour from now",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined transient_thought; cleanup_expired asserted in the data-lifecycle test",
    ),
    (
        "docs/recipes/index.md",
        "confirmation_count incremented, no new row",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined fact/same_fact; illustrative dedup",
    ),
    (
        "docs/recipes/index.md",
        "consolidation: promoted {result.promoted_count}",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "assumes a store/cycle; run_consolidation asserted in the dreaming tests",
    ),
    (
        "docs/recipes/index.md",
        "target_id=some_thought_id",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "fragment assuming a store; journal entries asserted in the journal test",
    ),
    (
        "docs/recipes/index.md",
        "source_thought_id=prompting_thought_id",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "undefined prompting_thought_id; action lifecycle asserted in the action test",
    ),
    (
        "docs/search.md",
        "include_reflections=False",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "assumes a store + embedding; reflection filtering illustrative",
    ),
    (
        "docs/search.md",
        "reflections rank near the top for broad queries",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "assumes a store + embedding; reflection_boost default in config-defaults test",
    ),
    (
        "docs/search.md",
        "reflections_evicted",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "assumes a store + embedding; illustrative reflection cap",
    ),
    (
        "docs/search.md",
        "search_reflections_only",
        CompileOnlyReason.UNDEFINED_DOMAIN_VALUE,
        "assumes a store + embedding with reflections present; illustrative",
    ),
    (
        "docs/troubleshooting.md",
        "row_factory = aiosqlite.Row  # required",
        CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT,
        "opens a real on-disk database; illustrative connection setup",
    ),
)


def _blocks_by_file() -> dict[str, list[CodeBlock]]:
    grouped: dict[str, list[CodeBlock]] = defaultdict(list)
    for block in all_python_blocks():
        grouped[block.rel].append(block)
    return grouped


def _unique_block(rel: str, anchor: str) -> CodeBlock:
    """Return the single block in ``rel`` whose body contains ``anchor``.

    Fails loudly when the anchor matches zero or more than one block — the
    signal that a registry entry has drifted from the docs and must be updated
    (and the example re-verified).
    """
    path = REPO_ROOT / rel
    matches = [b for b in extract_python_blocks(path) if anchor in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"anchor {anchor!r} matched {len(matches)} blocks in {rel} (want exactly 1); "
            f"update the registry in {__file__} and re-verify the example.",
        )
    return matches[0]


def _executable_locations_list() -> list[str]:
    """Every location the execute layer runs, in registry order, WITHOUT deduplication.

    A list, not a set: two ``EXECUTABLE_BLOCKS``/``SYNC_EXECUTABLE_BLOCKS``/
    ``CONCATENATED_PAGES``/``FIXTURE_EXECUTED_BLOCKS`` rows (or two overlapping
    ranges) that resolve to the same block must appear twice here, so a
    within-registry duplicate is visible to
    ``test_executable_registry_has_no_duplicate_locations`` instead of quietly
    collapsing the way ``_executable_locations()``'s ``set`` return would hide
    it.
    """
    locations: list[str] = []
    for rel, anchor in EXECUTABLE_BLOCKS:
        locations.append(_unique_block(rel, anchor).location)
    for rel, anchor in SYNC_EXECUTABLE_BLOCKS:
        locations.append(_unique_block(rel, anchor).location)
    for rel, first_anchor, last_anchor in CONCATENATED_PAGES:
        path = REPO_ROOT / rel
        blocks = extract_python_blocks(path)
        start = next(i for i, b in enumerate(blocks) if first_anchor in b.body)
        end = next(i for i, b in enumerate(blocks) if last_anchor in b.body)
        locations.extend(block.location for block in blocks[start : end + 1])
    for rel, anchor, _invoke in FIXTURE_EXECUTED_BLOCKS:
        locations.append(_unique_block(rel, anchor).location)
    return locations


def _executable_locations() -> set[str]:
    """Locations (``file:line``) of every block the execute layer runs."""
    return set(_executable_locations_list())


def _behaviour_locations_list() -> list[str]:
    """Every ``BEHAVIOUR_BLOCKS`` location, in registry order, WITHOUT deduplication."""
    return [_unique_block(rel, anchor).location for rel, anchor in BEHAVIOUR_BLOCKS]


def _behaviour_locations() -> set[str]:
    return set(_behaviour_locations_list())


def _compile_only_locations_list() -> list[str]:
    """Every ``COMPILE_ONLY`` location, in registry order, WITHOUT deduplication."""
    return [_unique_block(rel, anchor).location for rel, anchor, _reason, _note in COMPILE_ONLY]


def _compile_only_locations() -> set[str]:
    return set(_compile_only_locations_list())


def _duplicate_locations(locations: list[str]) -> list[str]:
    """Return every location appearing more than once in ``locations``, sorted.

    ``len(locations) != len(set(locations))`` tells you *that* a registry has a
    duplicate; this names *which* location it is, for a readable failure.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for location in locations:
        if location in seen:
            duplicates.add(location)
        seen.add(location)
    return sorted(duplicates)


# The exact per-reason tally over COMPILE_ONLY, as of this revision -- every
# CompileOnlyReason member appears, including the two currently at 0, so this is a
# complete accounting over the closed vocabulary rather than a sparse "only the
# reasons currently in use" snapshot. Update this constant deliberately whenever an
# entry is added, removed, or recategorised -- that is the point: a category quietly
# growing (or shrinking, or newly used) must fail the test below, not merely change a
# number in a captured ``-s`` report nobody reads.
_EXPECTED_COMPILE_ONLY_REASON_TALLY: dict[str, int] = {
    CompileOnlyReason.ASSUMES_STORE_OR_CONNECTION.value: 0,
    CompileOnlyReason.DEFINITION_ONLY.value: 10,
    CompileOnlyReason.HARNESS_SHAPE_MISMATCH.value: 0,
    CompileOnlyReason.NO_ASSERTABLE_CLAIM.value: 0,
    CompileOnlyReason.REQUIRES_LIVE_EXTERNAL_SERVICE.value: 8,
    CompileOnlyReason.REQUIRES_ON_DISK_ARTIFACT.value: 27,
    CompileOnlyReason.REQUIRES_SPECIALLY_CONFIGURED_STORE.value: 2,
    CompileOnlyReason.UNDEFINED_DOMAIN_VALUE.value: 35,
}


def _compile_only_reason_tally() -> dict[str, int]:
    """Per-``CompileOnlyReason`` counts across every ``COMPILE_ONLY`` entry.

    Exhaustive over the whole closed vocabulary: every ``CompileOnlyReason`` member is
    a key, starting at 0, so a member with no current entry is visibly absent (a 0 in
    the tally) rather than missing from the dict entirely.

    Also the runtime half of the closed-vocabulary guarantee, in two layers: this
    function itself raises ``AttributeError`` if an entry's ``reason`` is not a real
    ``CompileOnlyReason`` member (a plain string or ``None`` has no ``.value``
    attribute to key the tally by). The caller's own ``isinstance`` check in
    ``test_compile_only_reasons_are_tallied`` catches the same case first, failing
    with a readable ``AssertionError`` instead of ever reaching this function.
    """
    tally: dict[str, int] = dict.fromkeys((member.value for member in CompileOnlyReason), 0)
    for _rel, _anchor, reason, _note in COMPILE_ONLY:
        tally[reason.value] += 1
    return tally


def test_behaviour_registry_anchors_are_unique() -> None:
    """Every (E)/(B)/(C) anchor binds exactly one block (no drift, no ambiguity)."""
    # _unique_block fails on 0 or >1 matches; resolving all three registries here
    # turns any drift into one clear failure.
    _executable_locations()
    _behaviour_locations()
    _compile_only_locations()


def test_no_location_is_classified_twice() -> None:
    """A block is executed OR behaviour-asserted OR compile-only — never two."""
    executed = _executable_locations()
    behaviour = _behaviour_locations()
    compile_only = _compile_only_locations()

    overlaps = (executed & behaviour) | (executed & compile_only) | (behaviour & compile_only)
    assert not overlaps, (
        "these documentation blocks are classified in more than one coverage "
        f"registry (E/B/C must be disjoint): {sorted(overlaps)}"
    )


def test_executable_registry_has_no_duplicate_locations() -> None:
    """A block registered twice in EXECUTABLE_BLOCKS/CONCATENATED_PAGES must fail.

    ``_executable_locations()`` returns a ``set``, so two rows -- or two
    overlapping ranges -- resolving to the same fence collapse into one
    element, and the duplicate is invisible to
    ``test_no_location_is_classified_twice``, which only compares *across*
    registries, never within one. This compares the row-derived list against
    its own deduplication directly.
    """
    duplicates = _duplicate_locations(_executable_locations_list())
    assert not duplicates, (
        "the executed registry (EXECUTABLE_BLOCKS / CONCATENATED_PAGES) reaches "
        f"these blocks more than once: {duplicates}"
    )


def test_behaviour_registry_has_no_duplicate_locations() -> None:
    """A block registered twice in BEHAVIOUR_BLOCKS must fail, not collapse into a set."""
    duplicates = _duplicate_locations(_behaviour_locations_list())
    assert not duplicates, f"BEHAVIOUR_BLOCKS registers these blocks more than once: {duplicates}"


def test_compile_only_registry_has_no_duplicate_locations() -> None:
    """A block registered twice in COMPILE_ONLY must fail, not collapse into a set.

    Before this test, duplicating the first ``COMPILE_ONLY`` row verbatim left
    every existing exactly-once test green: ``_compile_only_locations()``
    returns a ``set``, so the two rows collapse into one location, and
    ``test_compile_only_reasons_are_tallied`` at the time only checked
    ``sum(tally.values()) == len(COMPILE_ONLY)`` -- true by construction of the
    tally (it sums to the row count no matter what the rows are), so it grew
    right along with the duplicate instead of catching it. That tautological
    check has since been removed in favour of comparing against a value fixed
    independently of the row count.
    """
    duplicates = _duplicate_locations(_compile_only_locations_list())
    assert not duplicates, f"COMPILE_ONLY registers these blocks more than once: {duplicates}"


def test_every_documentation_block_is_covered() -> None:
    """The no-silent-gap guarantee: every block is E, B, or C — exactly one.

    A newly added example that is neither executed, behaviour-asserted, nor
    registered as compile-only will appear in ``uncovered`` and fail here, so
    coverage can never silently regress below the compile floor.
    """
    executed = _executable_locations()
    behaviour = _behaviour_locations()
    compile_only = _compile_only_locations()
    covered = executed | behaviour | compile_only

    all_blocks = all_python_blocks()
    all_locations = {b.location for b in all_blocks}

    uncovered = sorted(all_locations - covered)
    assert not uncovered, (
        "these documentation code blocks are covered by nothing beyond compile — "
        "execute them (test_docs_examples_execute), behaviour-assert them "
        "(test_docs_examples_behavior), or register them in COMPILE_ONLY with a "
        f"reason: {uncovered}"
    )

    # Report the census (total blocks, #E, #B, #C) so the counts are visible in -s.
    print(  # noqa: T201 — intentional census summary for the -s report
        f"\nDoc-example census: total={len(all_locations)} "
        f"E(executed)={len(executed)} B(behaviour)={len(behaviour)} "
        f"C(compile-only)={len(compile_only)}"
    )

    # Belt-and-braces: with disjoint registries, the parts must sum to the whole.
    assert len(executed) + len(behaviour) + len(compile_only) == len(all_locations)


def test_compile_only_reasons_are_tallied() -> None:
    """Every ``COMPILE_ONLY`` entry cites one closed-vocabulary reason and one real note.

    Before ``CompileOnlyReason`` existed, 122 compile-only blocks carried 113 distinct
    free-text sentences -- 122 against 113 is nine duplicate uses, not a one-sentence-
    per-nine-blocks ratio -- so nobody could count how many blocks were exempt for
    which cause. This asserts each entry's third element really is a
    ``CompileOnlyReason`` member (not a stray string or ``None`` left by an incomplete
    edit), that its fourth element is an actual explicit free-text note (not ``""`` or
    ``None``, which passed unnoticed before this assertion existed), and that the
    per-cause tally exactly matches ``_EXPECTED_COMPILE_ONLY_REASON_TALLY``. That last
    check is deliberately an assertion against an independently pinned value, not a
    print and not a self-check: ``sum(tally.values()) == len(COMPILE_ONLY)`` would be
    true by construction (the tally is built by incrementing one bucket per row, so
    the sum can never be anything else while the loop completes) and was removed as a
    tautology that could not fail. A print in a ``-s`` report is read by nobody, so
    neither is the guarantee ``test_every_fenced_block_is_classified_exactly_once``
    gives the non-python fences unless a changed count actually fails a test against a
    value fixed independently of the count itself -- which this now does.
    """
    for rel, anchor, reason, note in COMPILE_ONLY:
        assert isinstance(reason, CompileOnlyReason), (
            f"{rel}:{anchor!r} does not cite a CompileOnlyReason member (got "
            f"{reason!r}); every COMPILE_ONLY entry must name exactly one closed-"
            f"vocabulary reason, in addition to its free-text note {note!r}."
        )
        assert isinstance(note, str), (
            f"{rel}:{anchor!r} has a non-string note (got {note!r}); every COMPILE_ONLY "
            f"entry's fourth element must be an explicit free-text string."
        )
        assert note.strip(), (
            f"{rel}:{anchor!r} has no explicit free-text note (got {note!r}); every "
            f"COMPILE_ONLY entry must say what its reason member cannot -- which "
            f"exact API, which specific test mirrors the same claim on a runnable "
            f"fixture."
        )

    tally = _compile_only_reason_tally()
    assert tally == _EXPECTED_COMPILE_ONLY_REASON_TALLY, (
        "the compile-only reason tally changed -- a cause grew, shrank, or a new one "
        f"appeared. Expected {_EXPECTED_COMPILE_ONLY_REASON_TALLY}, got {tally}. If "
        f"this is a deliberate addition/removal, update "
        f"_EXPECTED_COMPILE_ONLY_REASON_TALLY in {__file__} to match."
    )

    print(f"\nCompile-only reason tally (n={len(COMPILE_ONLY)}):")  # noqa: T201
    for reason_value, count in sorted(tally.items()):
        print(f"  compile-only reason[{reason_value}] = {count}")  # noqa: T201

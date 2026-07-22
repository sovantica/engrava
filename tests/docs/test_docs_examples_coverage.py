"""Layer 5 of the documentation-example tests — the no-silent-gap census.

The compile layer (Layer 2) proves every fenced ``python`` block is valid
Python and names no phantom API, but compilation is a *floor*, not proof that
an example works. This module raises that floor to a hard guarantee: **every**
documentation block must be accounted for as exactly one of

* **(E) executed** — a self-contained script or a composable page-run that is
  actually run against the installed package by
  ``test_docs_examples_execute.py`` (its allowlists ``EXECUTABLE_BLOCKS`` /
  ``CONCATENATED_PAGES`` are the source of truth here);
* **(B) behaviour-asserted** — a fragment (it assumes an existing
  ``store``/``conn``) whose behavioural claim is MIRRORED and asserted by a
  Layer-3 test in ``test_docs_examples_behavior.py``; it is listed in
  ``BEHAVIOUR_BLOCKS`` below; or
* **(C) compile-only** — a block that legitimately cannot be executed
  (a partial ``...`` snippet, a class/Protocol definition, a snippet needing a
  live external provider / network / on-disk config, an intentionally-invalid
  anti-pattern, or a duplicate whose behaviour is asserted elsewhere). Each such
  block is registered in ``COMPILE_ONLY`` **with an explicit reason**, so the
  compile-only status is a deliberate, auditable decision — never a silent gap.

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

from tests.docs._md_blocks import REPO_ROOT, CodeBlock, all_python_blocks, extract_python_blocks
from tests.docs.test_docs_examples_execute import CONCATENATED_PAGES, EXECUTABLE_BLOCKS

# (E) is derived from the execute layer's allowlists — see EXECUTABLE_BLOCKS and
# CONCATENATED_PAGES in test_docs_examples_execute.py. Do not duplicate them.

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
    ("docs/dreaming.md", "ASSOCIATED edges created"),  # result fields
    # extension-hooks.md
    ("docs/extension-hooks.md", "class RecencyBoostHooks"),  # hooks protocol + score
    # extensions.md
    ("docs/extensions.md", "def mindql_extension_registry"),  # MyHooks protocol
    ("docs/extensions.md", "STATS_COMMAND = MindQLExtension"),  # custom command end-to-end
    ("docs/extensions.md", "candidates_limit=100"),  # DreamingExtension run
    ("docs/extensions.md", "class ImportanceSignal:"),  # custom signal
    # guides/embeddings.md
    ("docs/guides/embeddings.md", "query_prefix="),  # asymmetric role prefixes
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

# (C) Blocks that legitimately cannot be executed, each paired with an explicit
# reason. Each entry is ``(markdown_path, anchor_substring, reason)``. This is the
# auditable ledger of every deliberate compile-only decision: a block here is
# covered by the compile + phantom-API guards (Layer 2), and where a reason cites a
# behaviour test, that test asserts the same API on a runnable mirror.
COMPILE_ONLY: tuple[tuple[str, str, str], ...] = (
    (
        "README.md",
        'from_config("engrava.yaml") as store:',
        "requires an on-disk engrava.yaml; illustrative from_config wiring",
    ),
    (
        "README.md",
        "class MyHooks(EngravaHooksProtocol):",
        "class-definition-only Protocol impl; conformance asserted in the hooks test",
    ),
    (
        "README.md",
        "EngravaManager(data_dir=Path(",
        "requires an on-disk data_dir and an unimported Path; illustrative multi-store",
    ),
    (
        "docs/api-reference.md",
        "schema already applied by from_config",
        "illustrative setup with a `...` placeholder and from_config wiring",
    ),
    (
        "docs/api-reference.md",
        "from_thought_id=src_id",
        "undefined src_id/dst_id; edge creation asserted in the CRUD behaviour test",
    ),
    (
        "docs/api-reference.md",
        "consolidated_member_ids(reflection_id)",
        "undefined reflection_id; illustrative reflection-graph traversal",
    ),
    (
        "docs/api-reference.md",
        "for record in many_records",
        "undefined many_records; suspend_auto_commit runs in the migrating example",
    ),
    (
        "docs/api-reference.md",
        "Raises ReadOnlyViolationError",
        "contains a `...` placeholder call; illustrative read-only wrapper",
    ),
    (
        "docs/api-reference.md",
        'list_services()   # -> ["my-service"]',
        "requires an on-disk data_dir and an unimported Path; illustrative manager usage",
    ),
    (
        "docs/api-reference.md",
        '"turn_index": 5',
        "placeholder field values (`...`); illustrative metadata usage",
    ),
    (
        "docs/api-reference.md",
        'FieldPredicate("$.subtype", FieldOp.EQ, "supports")',
        "undefined store; illustrative edge-metadata filter usage",
    ),
    (
        "docs/api-reference.md",
        "class EmbeddingProviderProtocol(Protocol)",
        "Protocol-definition only",
    ),
    (
        "docs/api-reference.md",
        "class DerivedRecordProducerProtocol(Protocol)",
        "Protocol-definition only",
    ),
    (
        "docs/extension-hooks.md",
        "-> Sequence[DerivedRecord]: ...",
        "bare derive_records signature fragment; conformance asserted in the seam tests",
    ),
    (
        "docs/extension-hooks.md",
        "hooks=StructuralSplitProducer(),",
        "opens a real on-disk database file; derived-records wiring illustrative",
    ),
    (
        "docs/extension-hooks.md",
        "class SentenceSplitter(DefaultEngravaHooks):",
        "class-definition-only producer; the shipped StructuralSplitProducer is behaviour-tested",
    ),
    (
        "docs/extension-hooks.md",
        'window_unit="word",',
        "StructuralSplitProducer FIXED_WINDOW constructor fragment; segmentation "
        "asserted in the structural-split tests",
    ),
    (
        "docs/extension-hooks.md",
        "store.derive_existing(thought_id)",
        "fragment assuming a store; derive_existing asserted in the derived-records backfill tests",
    ),
    (
        "docs/audit-trail.md",
        "journaling is active",
        "requires an on-disk engrava.yaml; illustrative from_config wiring",
    ),
    (
        "docs/audit-trail.md",
        'connect("engrava.db") as conn:',
        "opens a real on-disk database file; journal wiring illustrative",
    ),
    (
        "docs/audit-trail.md",
        'target_id="thought-001"',
        "fragment assuming a store; get_entries asserted in the journal test",
    ),
    (
        "docs/audit-trail.md",
        "JournalIntegrityError as exc",
        "`...` placeholder; needs a corrupted on-disk journal; illustrative handling",
    ),
    (
        "docs/cli.md",
        "print(result.valid)",
        "illustrative CLI snippet; verify_journal asserted in the audit-trail test",
    ),
    (
        "docs/concepts.md",
        "inner/outer-speech boundary",
        "illustrative ThoughtRecord field tour; construction asserted in the CRUD tests",
    ),
    (
        "docs/concepts.md",
        "cycle_provider=StaticCycleProvider(0)",
        "undefined conn; opt-in cycle-provider wiring (behaviour in cycle-provider tests)",
    ),
    (
        "docs/concepts.md",
        "resume_from = await store.max_cycle()",
        "fragment assuming a store; max_cycle recovery asserted in the cycle-provider tests",
    ),
    (
        "docs/concurrency.md",
        "PRAGMA busy_timeout",
        "opens a real on-disk database file; illustrative connection tuning",
    ),
    (
        "docs/concurrency.md",
        'get_store("tenant_a")',
        "requires an on-disk engrava.yaml and per-service db files; illustrative",
    ),
    (
        "docs/configuration.md",
        'get_thought("abc")',
        "requires an on-disk engrava.yaml; illustrative load_config wiring",
    ),
    (
        "docs/configuration.md",
        "resolve_embedding_provider(config.embeddings)",
        "requires an on-disk engrava.yaml; illustrative provider resolution",
    ),
    (
        "docs/configuration.md",
        'get_store("main")',
        "requires an on-disk engrava.yaml; illustrative manager wiring",
    ),
    (
        "docs/deployment.md",
        "Hold this store for the lifetime",
        "requires an on-disk engrava.yaml and an undefined run_app; illustrative skeleton",
    ),
    (
        "docs/deployment.md",
        "connection closed here",
        "contains a `...` placeholder; illustrative lifecycle/close",
    ),
    (
        "docs/deployment.md",
        "the caller owns and closes the connection",
        "opens a real on-disk database and contains `...`; illustrative lifecycle",
    ),
    (
        "docs/dreaming.md",
        "class MySignal:",
        "undefined config; custom-signal class (wiring asserted in the extensions test)",
    ),
    (
        "docs/dreaming.md",
        "graph_edge_decay=0.3",
        "fragment assuming a store; graph-weighted hybrid asserted in the hybrid tests",
    ),
    (
        "docs/extension-hooks.md",
        "RECENT_COMMAND = MindQLExtension",
        "handler + command definition only; execution asserted via the STATS test",
    ),
    (
        "docs/extension-hooks.md",
        'extensions={"RECENT": RECENT_COMMAND}',
        "undefined conn; execution asserted via the STATS behaviour test",
    ),
    (
        "docs/extension-hooks.md",
        "def test_my_hooks_satisfy_protocol",
        "illustrative unit-test snippet; conformance asserted in the hooks test",
    ),
    (
        "docs/extensions.md",
        "on_store / on_retrieve are now called automatically",
        "opens a real on-disk database; invocation asserted via hooks/STATS tests",
    ),
    (
        "docs/extensions.md",
        'executor = MindQLExecutor(conn, extensions={"STATS": STATS_COMMAND})',
        "undefined conn; STATS execution asserted end-to-end in the behaviour test",
    ),
    (
        "docs/extensions.md",
        "mindql_extensions=[],",
        "references on-disk migration files and MyHooks; illustrative manifest",
    ),
    (
        "docs/extensions.md",
        "package_root override (test fixtures)",
        "references on-disk migration files; illustrative manifest path options",
    ),
    (
        "docs/extensions.md",
        "are now applied",
        "opens a real on-disk database with on-disk migrations; illustrative",
    ),
    (
        "docs/extensions.md",
        "discover_manifests()",
        "undefined db; discovery scans installed packages",
    ),
    (
        "docs/extensions.md",
        "class ExtendedStore(SqliteEngravaCore)",
        "subclass-definition-only (overrides a private method); illustrative",
    ),
    (
        "docs/guides/agent-memory.md",
        'connect("agent-memory.db")',
        "opens a real on-disk database and undefined my_embed_fn; illustrative",
    ),
    (
        "docs/guides/agent-memory.md",
        "async def store_percept",
        "helper-function def assuming a store; percept metadata asserted in the metadata test",
    ),
    (
        "docs/guides/agent-memory.md",
        "async def retrieve_context",
        "helper-function def with undefined my_embed_fn; search_hybrid asserted elsewhere",
    ),
    (
        "docs/guides/agent-memory.md",
        "reply = await my_llm(prompt)",
        "references an undefined my_llm; illustrative prompt assembly",
    ),
    (
        "docs/guides/agent-memory.md",
        'intent="answered user"',
        "undefined percept_thought; action creation asserted in the action test",
    ),
    (
        "docs/guides/agent-memory.md",
        "async def store_utterance",
        "helper-function def assuming a store; utterance metadata asserted in the metadata test",
    ),
    (
        "docs/guides/agent-memory.md",
        "cycle += 1",
        "illustrative loop skeleton (undefined running)",
    ),
    (
        "docs/guides/agent-memory.md",
        "inside the loop, after advancing the cycle:",
        "undefined cycle/store; run_consolidation asserted in the dreaming tests",
    ),
    (
        "docs/guides/agent-memory.md",
        "ordered by updated_cycle desc",
        "fragment assuming a store; illustrative cycle bootstrap",
    ),
    (
        "docs/guides/embeddings.md",
        'SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")\nasync with',
        "loads a real ST model (offline in CI) and opens a real db; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "provider wired from config, auto_embed honoured",
        "requires an on-disk engrava.yaml; illustrative from_config wiring",
    ),
    (
        "docs/guides/embeddings.md",
        "async def strict_ingest",
        "requires a live embedding provider; illustrative failure handling",
    ),
    (
        "docs/guides/embeddings.md",
        "async def batch_ingest",
        "requires a live embedding provider (embed_batch); illustrative batch ingest",
    ),
    (
        "docs/guides/embeddings.md",
        "batch_size=32,",
        "constructs a provider that loads a real model; illustrative config",
    ),
    (
        "docs/guides/embeddings.md",
        "point at any compatible API",
        "requires an OpenAI-compatible endpoint + key/network; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "base_retry_delay_s=0.5,",
        "requires an OpenAI-compatible endpoint; illustrative retry config",
    ),
    (
        "docs/guides/embeddings.md",
        "default Ollama address",
        "requires a running Ollama server; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "HuggingFaceProvider(",
        "requires an HF token + network; illustrative",
    ),
    (
        "docs/guides/embeddings.md",
        "the length your callback returns",
        "undefined my_embed_fn; CallbackProvider asserted in the quickstart search test",
    ),
    (
        "docs/guides/embeddings.md",
        "the query text is embedded for you",
        "assumes a configured provider + store; hybrid search asserted in behaviour tests",
    ),
    (
        "docs/guides/embeddings.md",
        "required — no auto-embed here",
        "assumes a real provider + store; search_similar asserted in the quickstart test",
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        "memory.add(",
        "fragment assuming a store; create_thought asserted in the CRUD behaviour test",
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        "hits = memory.search(",
        "fragment assuming a store; filtered search asserted in the scoped-retrieval test",
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        "over-fetch, then filter and trim",
        "fragment assuming a store; illustrative post-filter of results",
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        'get_store("u1")  # u1.db',
        "requires an on-disk engrava.yaml + per-user db files; illustrative",
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        "json_extract(metadata_json",
        "fragment assuming a conn; illustrative raw-SQL escape hatch",
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        'allowed={"public"}, owner="u1"',
        "fragment assuming a store; filters + visibility asserted in the scoped test",
    ),
    (
        "docs/known-limitations.md",
        "FTS5 is available",
        "standalone stdlib sqlite3 capability probe; no engrava API",
    ),
    (
        "docs/known-limitations.md",
        "result.backends_used  # preferred",
        "single-line assertion fragment; backends_used asserted in the hybrid tests",
    ),
    (
        "docs/memory-hygiene.md",
        'from_config("engrava.yaml") as store:\n    result = await store.run_hygiene',
        "requires an on-disk engrava.yaml; run_hygiene asserted in the hygiene test",
    ),
    (
        "docs/mindql.md",
        "Active thoughts: {count_result.count}",
        "fragment assuming a conn; FIND/COUNT asserted in the MindQL behaviour test",
    ),
    (
        "docs/observability.md",
        "engrava_search_p99_ms",
        "requires optional prometheus_client and a file-backed store; illustrative export",
    ),
    (
        "docs/observability.md",
        "async def journal_ok",
        "helper-function def assuming a store; verify_integrity asserted in the journal test",
    ),
    (
        "docs/observability.md",
        "async def healthcheck",
        "helper-function def assuming a store; count_thoughts asserted elsewhere",
    ),
    (
        "docs/observability.md",
        'getLogger("engrava").setLevel',
        "illustrative logging configuration; no engrava API behaviour",
    ),
    (
        "docs/performance.md",
        "the vector backend",
        "requires an on-disk engrava.yaml + vector backend; illustrative",
    ),
    (
        "docs/performance.md",
        "async def bulk_load",
        "helper-function def assuming a store; suspend_auto_commit runs in the migrating example",
    ),
    (
        "docs/quickstart.md",
        'recall("what does the user prefer?")',
        "fragment assuming a store; remember/recall asserted in the quickstart test",
    ),
    (
        "docs/quickstart.md",
        "Python's async ecosystem and rich ML libraries",
        "fragment assuming a store; create_thought asserted in the CRUD test",
    ),
    (
        "docs/quickstart.md",
        "WAL mode enables concurrent reads",
        "fragment assuming a store; create_edge asserted in the CRUD behaviour test",
    ),
    (
        "docs/quickstart.md",
        "returns (thought_id, bm25_score) tuples",
        "fragment assuming a store; search_fts shape asserted in the quickstart search test",
    ),
    (
        "docs/quickstart.md",
        "Store an embedding for an existing thought",
        "loads a real ST model (offline in CI); asserted in the quickstart search test",
    ),
    (
        "docs/quickstart.md",
        "Found {len(result.rows)} thoughts",
        "fragment assuming a conn; FIND/COUNT asserted in the MindQL behaviour test",
    ),
    (
        "docs/recipes/index.md",
        "async def store_turn",
        "helper-function definition assuming an existing store",
    ),
    (
        "docs/recipes/index.md",
        "async def context_for",
        "helper-function definition assuming an existing store",
    ),
    (
        "docs/recipes/index.md",
        "async def search_in_session",
        "helper-function definition assuming an existing store",
    ),
    (
        "docs/recipes/index.md",
        "expire this thought one hour from now",
        "undefined transient_thought; cleanup_expired asserted in the data-lifecycle test",
    ),
    (
        "docs/recipes/index.md",
        "confirmation_count incremented, no new row",
        "undefined fact/same_fact; illustrative dedup",
    ),
    (
        "docs/recipes/index.md",
        "consolidation: promoted {result.promoted_count}",
        "assumes a store/cycle; run_consolidation asserted in the dreaming tests",
    ),
    (
        "docs/recipes/index.md",
        "target_id=some_thought_id",
        "fragment assuming a store; journal entries asserted in the journal test",
    ),
    (
        "docs/recipes/index.md",
        "PLANNED → EXECUTING → CONFIRMED / FAILED / BLOCKED",
        "undefined prompting_thought_id; action lifecycle asserted in the action test",
    ),
    (
        "docs/recipes/index.md",
        "recent = await store.list_thoughts(limit=1)        # ordered",
        "fragment assuming a store; illustrative cycle bootstrap",
    ),
    (
        "docs/search.md",
        "the caller owns",
        "fragment assuming a store; transaction-time recency asserted in "
        "test_transaction_recency.py",
    ),
    (
        "docs/search.md",
        "OR-matched",
        "fragment assuming a store; search_fts asserted in the quickstart search test",
    ),
    (
        "docs/search.md",
        "python async",
        "fragment assuming a store; graph-weighted hybrid asserted in the hybrid tests",
    ),
    (
        "docs/search.md",
        "async def assemble_unit",
        "helper-function def assuming a store; collapse/filter asserted in de-frag tests",
    ),
    (
        "docs/search.md",
        "include_reflections=False",
        "assumes a store + embedding; reflection filtering illustrative",
    ),
    (
        "docs/search.md",
        "reflections rank near the top for broad queries",
        "assumes a store + embedding; reflection_boost default in config-defaults test",
    ),
    (
        "docs/search.md",
        "reflections_evicted",
        "assumes a store + embedding; illustrative reflection cap",
    ),
    (
        "docs/search.md",
        "search_reflections_only",
        "assumes a store + embedding with reflections present; illustrative",
    ),
    (
        "docs/troubleshooting.md",
        "row_factory = aiosqlite.Row  # required",
        "opens a real on-disk database; illustrative connection setup",
    ),
    (
        "docs/troubleshooting.md",
        "['fts5', 'priority', 'recency']",
        "fragment assuming a store; backends_used asserted in the hybrid tests",
    ),
    (
        "docs/troubleshooting.md",
        "require an exact phrase",
        "fragment assuming a store; FTS operators illustrative",
    ),
    (
        "docs/troubleshooting.md",
        "lower it if nothing clears the bar",
        "fragment assuming a store; run_consolidation asserted in the dreaming tests",
    ),
    (
        "docs/troubleshooting.md",
        "ReferentialIntegrityError  # ImportError!",
        "intentionally-invalid import anti-pattern; must NOT be executed",
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


def _executable_locations() -> set[str]:
    """Locations (``file:line``) of every block the execute layer runs."""
    locations: set[str] = set()
    for rel, anchor in EXECUTABLE_BLOCKS:
        locations.add(_unique_block(rel, anchor).location)
    for rel, first_anchor, last_anchor in CONCATENATED_PAGES:
        path = REPO_ROOT / rel
        blocks = extract_python_blocks(path)
        start = next(i for i, b in enumerate(blocks) if first_anchor in b.body)
        end = next(i for i, b in enumerate(blocks) if last_anchor in b.body)
        for block in blocks[start : end + 1]:
            locations.add(block.location)
    return locations


def _behaviour_locations() -> set[str]:
    return {_unique_block(rel, anchor).location for rel, anchor in BEHAVIOUR_BLOCKS}


def _compile_only_locations() -> set[str]:
    locations: set[str] = set()
    for rel, anchor, _reason in COMPILE_ONLY:
        locations.add(_unique_block(rel, anchor).location)
    return locations


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

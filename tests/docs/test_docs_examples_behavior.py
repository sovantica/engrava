"""Layer 3 of the documentation-example tests — behaviour, not just execution.

This module re-implements the patterns shown in the documentation and asserts
that they not only run but produce the documented *behaviour* (correct return
shapes, counts, and values). It complements:

* ``test_docs_examples_compile.py`` — every block compiles + names no phantom API.
* ``test_docs_examples_execute.py`` — the self-contained scripts run end-to-end.

.. important::
   The code here MIRRORS the examples in ``README.md`` and ``docs/``. It is a
   deliberate copy (the doc fragments cannot be imported), so the two can drift
   apart. **If you edit any documentation example — or this file — verify that
   the documented snippet and these assertions still match the shipped API.**
   The phantom-API guard in ``test_docs_examples_compile.py`` catches the most
   common drift automatically, but value/shape assertions live here.

Each test names the doc + section it pins.
"""

from __future__ import annotations

import ast
import re
import uuid
from typing import TYPE_CHECKING, cast

import aiosqlite
import pytest

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    CallbackProvider,
    DefaultEngravaHooks,
    DeriveContext,
    DerivedRecord,
    DeriveGates,
    DreamingConfig,
    DreamingContext,
    DreamingExtension,
    DreamingGates,
    EdgeRecord,
    EdgeType,
    EmbeddingProviderContractError,
    EngravaHooksProtocol,
    FieldOp,
    FieldPredicate,
    HygienePolicyConfig,
    LifecycleStatus,
    MetadataFilter,
    MindQLCommand,
    MindQLExecutor,
    MindQLExtension,
    MindQLParseError,
    MindQLQuery,
    Priority,
    ProvenanceContext,
    ScoringContext,
    SentenceTransformerProvider,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
    VisibilityQueryFilter,
    parse,
    percept,
    thought,
    utterance,
)
from engrava.config import SearchConfig
from tests.docs._md_blocks import (
    REPO_ROOT,
    CodeBlock,
    extract_fenced_blocks,
    extract_python_blocks,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _observation(essence: str = "Python is great for AI") -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content="Python's async ecosystem makes it ideal for AI agents.",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="human",
    )


def _belief(essence: str = "SQLite is underrated") -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.BELIEF,
        essence=essence,
        content="SQLite provides ACID transactions with zero setup.",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="human",
    )


async def test_readme_and_quickstart_crud_and_edge() -> None:
    """README 'Basic Usage' + quickstart 'Add Thoughts' / 'Link Thoughts'.

    create_thought takes a ThoughtRecord and returns it; create_edge takes an
    EdgeRecord; get_thought returns the record or None.
    """
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        observation = _observation()
        stored = await store.create_thought(observation)
        assert stored.thought_id == observation.thought_id  # returns the record

        fetched = await store.get_thought(observation.thought_id)
        assert fetched is not None
        assert fetched.essence == "Python is great for AI"

        belief = _belief()
        await store.create_thought(belief)
        edge = await store.create_edge(
            EdgeRecord(
                edge_id=str(uuid.uuid4()),
                from_thought_id=observation.thought_id,
                to_thought_id=belief.thought_id,
                edge_type=EdgeType.ASSOCIATED,
                weight=0.8,
                created_cycle=0,
            ),
        )
        assert edge.edge_id
        edges = await store.get_edges(observation.thought_id, direction="BOTH")
        assert len(edges) == 1


async def test_quickstart_search_returns_tuples() -> None:
    """quickstart 'Full-Text Search' / 'Embedding Similarity Search'.

    search_fts and search_similar return (thought_id, score) tuples — not
    ThoughtRecord objects — and store_embedding takes keyword-only model_name.
    """
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        observation = _observation()
        await store.create_thought(observation)

        for thought_id, score in await store.search_fts("Python AI", top_k=5):
            assert isinstance(thought_id, str)
            assert isinstance(score, float)

        provider = CallbackProvider(
            callback=lambda _text: [0.1] * 384,
            dimension=384,
            model_name="my-model",
        )
        vector = await provider.embed(observation.content)
        await store.store_embedding(observation.thought_id, vector, model_name="my-model")
        results = await store.search_similar(vector, top_k=5)
        assert all(isinstance(tid, str) and isinstance(sc, float) for tid, sc in results)


async def test_quickstart_remember_and_recall() -> None:
    """quickstart 'Store and search a memory — the short way'.

    remember() stores text as a thought (generating its id) and returns the
    ThoughtRecord; recall() returns a HybridSearchResult whose ``results`` are
    (thought_id, score) tuples. Without a ``current_cycle``, recall() leaves
    recency off — the documented contract — so ``backends_used`` carries no
    recency marker.
    """
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        stored = await store.remember("User prefers concise answers")
        assert stored.thought_id  # generated for the caller
        assert stored.essence == "User prefers concise answers"
        await store.remember("User works in Berlin")

        result = await store.recall("what does the user prefer?")
        for thought_id, score in result.results:
            assert isinstance(thought_id, str)
            assert isinstance(score, float)
        # Documented: recency stays off until a current_cycle is supplied.
        assert "recency" not in result.backends_used


async def test_quickstart_mindql_find_and_count() -> None:
    """README + quickstart + mindql.md MindQL usage.

    MindQLExecutor runs against a connection; execute takes a parsed
    MindQLQuery; result.count carries the COUNT value.
    """
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.create_thought(_observation())
        await store.create_thought(_belief())

        executor = MindQLExecutor(conn)
        find = await executor.execute(
            parse("FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 10"),
        )
        assert len(find.rows) == 1

        count = await executor.execute(
            parse("COUNT thoughts WHERE lifecycle_status = 'ACTIVE'"),
        )
        assert count.count == 2


async def test_api_reference_hybrid_search_result_shape() -> None:
    """api-reference.md HybridSearchResult — results + backends_used only."""
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.create_thought(_observation())

        result = await store.search_hybrid("Python", top_k=5)
        assert isinstance(result.results, list)
        assert isinstance(result.backends_used, frozenset)
        for thought_id, score in result.results:
            assert isinstance(thought_id, str)
            assert isinstance(score, float)


async def test_api_reference_metrics_snapshot() -> None:
    """api-reference.md + observability.md store.metrics() snapshot."""
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.create_thought(_observation())
        await store.create_thought(_belief())

        metrics = await store.metrics()
        assert metrics.thoughts.total == 2
        assert isinstance(metrics.edges.by_type, dict)
        assert isinstance(metrics.search_latency.p95_ms, float)


async def test_extensions_custom_mindql_command_end_to_end() -> None:
    """extensions.md + extension-hooks.md custom MindQL command.

    A MindQLExtension handler is invoked as handler(conn, args) and the verb is
    recognised by parse() only when listed in known_extensions.
    """

    async def _handle_stats(
        db: aiosqlite.Connection,
        _args: list[str],
    ) -> list[dict[str, object]]:
        cursor = await db.execute(
            "SELECT thought_type, COUNT(*) AS n FROM thought GROUP BY thought_type",
        )
        rows = await cursor.fetchall()
        return [{row["thought_type"]: row["n"]} for row in rows]

    stats = MindQLExtension(
        command_name="STATS",
        handler=_handle_stats,
        description="Show thought statistics",
    )

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.create_thought(_observation())

        executor = MindQLExecutor(conn, extensions={"STATS": stats})
        result = await executor.execute(parse("STATS", known_extensions={"STATS"}))
        assert result.rows == [{"OBSERVATION": 1}]


def test_extensions_custom_signal_satisfies_protocol() -> None:
    """extensions.md custom DreamingSignalProtocol signal is callable (thought, ctx)."""

    class ImportanceSignal:
        def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
            if thought.priority == Priority.P1:
                return 1.0
            if thought.priority == Priority.P2:
                return 0.7
            return 0.3

    signal = ImportanceSignal()
    score = signal(_observation(), DreamingContext(current_cycle=1, total_thoughts=1))
    assert score == pytest.approx(0.7)

    # Wiring shown in the doc: signal registered with a matching weight.
    extension = DreamingExtension(
        config=DreamingConfig(enabled=True, signals={"importance": 0.3}),
        custom_signals={"importance": signal},
    )
    assert extension is not None


async def test_extensions_dreaming_run_consolidation() -> None:
    """extensions.md Dreaming Extension — run_consolidation returns a result."""
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.create_thought(_observation())

        dreaming = DreamingExtension(
            config=DreamingConfig(
                enabled=True,
                promote_threshold=0.6,
                gates=DreamingGates(
                    min_confirmations=2,
                    min_age_cycles=1,
                    max_promoted_per_run=20,
                    allow_zero_confirmation=True,
                ),
            ),
        )
        result = await dreaming.run_consolidation(store, current_cycle=42)
        assert hasattr(result, "promoted_count")


# ---------------------------------------------------------------------------
# 0.5.0 feature examples — behaviour mirrors of the fragment snippets that the
# execute layer cannot run (they assume an existing ``store``/``conn``). Each
# test MIRRORS a specific doc snippet and asserts the behaviour its prose
# claims, grounded in the shipped API.
# ---------------------------------------------------------------------------


async def _fresh_store(conn: aiosqlite.Connection, **kwargs: object) -> SqliteEngravaCore:
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn, **kwargs)  # type: ignore[arg-type]
    await store.ensure_schema()
    return store


async def test_api_reference_restore_thought_roundtrip() -> None:
    """api-reference.md 'Archive, then restore — a lossless round-trip'.

    ``update_thought(lifecycle_status=ARCHIVED)`` then ``restore_thought`` returns
    the record with ``lifecycle_status`` back at ``ACTIVE``.
    """
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        stored = await store.create_thought(_observation())

        await store.update_thought(stored.thought_id, lifecycle_status=LifecycleStatus.ARCHIVED)
        restored = await store.restore_thought(stored.thought_id, current_cycle=42)
        assert restored.lifecycle_status is LifecycleStatus.ACTIVE


async def test_api_reference_provenance_write_and_filter() -> None:
    """api-reference.md 'Provenance' — ProvenanceContext + provenance_filter.

    A thought written with a ``ProvenanceContext`` is retrievable by a
    ``provenance_filter`` predicate on ``$.session_id``; provenance is an
    untrusted, read-only hint (no new verb).
    """
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        tagged = _observation()
        tagged = ThoughtRecord(
            **{
                **tagged.__dict__,
                "provenance": ProvenanceContext(
                    session_id="sess-42",
                    actor_id="agent-a",
                    retrieval_query="remote work trade-offs",
                    instruction_context="summarise for a busy exec",
                    retrieval_context_ids=["t-1", "t-2"],
                ),
            }
        )
        await store.create_thought(tagged)
        await store.create_thought(_belief())  # no provenance -> must not match

        mine = await store.list_thoughts(
            provenance_filter=MetadataFilter(
                [FieldPredicate("$.session_id", FieldOp.EQ, "sess-42")]
            ),
        )
        assert [t.thought_id for t in mine] == [tagged.thought_id]


async def test_search_scoped_filters_and_visibility() -> None:
    """search.md + migrating 'scope with a metadata filter' / 'public, or mine'.

    ``filters=MetadataFilter([...])`` restricts a recall to matching rows;
    ``visibility=VisibilityQueryFilter({"public"}, owner=...)`` admits public
    rows plus rows the owner owns, and excludes others'.
    """

    def _row(essence: str, meta: dict[str, object]) -> ThoughtRecord:
        base = _observation(essence)
        return ThoughtRecord(**{**base.__dict__, "essence": essence, "metadata": meta})

    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        pub = await store.create_thought(
            _row("shared runbook", {"session_id": "s1", "visibility": "public"})
        )
        mine = await store.create_thought(
            _row("my runbook", {"session_id": "s1", "visibility": "private", "owner": "alice"})
        )
        theirs = await store.create_thought(
            _row("bob runbook", {"session_id": "s2", "visibility": "private", "owner": "bob"})
        )

        scoped = await store.recall(
            "runbook",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.session_id", FieldOp.EQ, "s1")]),
        )
        got = {tid for tid, _ in scoped.results}
        assert theirs.thought_id not in got  # filtered out by session
        assert got <= {pub.thought_id, mine.thought_id}

        vis = await store.recall(
            "runbook",
            top_k=10,
            visibility=VisibilityQueryFilter(allowed={"public"}, owner="alice"),
        )
        vis_ids = {tid for tid, _ in vis.results}
        assert pub.thought_id in vis_ids  # public admitted
        assert mine.thought_id in vis_ids  # owner's own row admitted
        assert theirs.thought_id not in vis_ids  # someone else's private row excluded


async def test_search_collapse_defragments_units() -> None:
    """search.md + api-reference collapse_key / collapse_max_per_unit de-frag.

    ``collapse_key`` keeps one best row per caller-defined unit (distinct units
    backfill); ``collapse_max_per_unit=2`` keeps up to two rows per unit. A
    composite key groups rows only when every component matches.
    """

    def _frag(turn: int, frag: int) -> ThoughtRecord:
        base = _observation("rollout decision")
        return ThoughtRecord(
            **{
                **base.__dict__,
                "thought_id": f"t{turn}-{frag}",
                "content": f"rollout decision turn {turn} fragment {frag}",
                "metadata": {"turn_id": f"turn-{turn}", "session_id": "s1", "turn_index": turn},
            }
        )

    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        for turn in range(3):
            for frag in range(2):
                await store.create_thought(_frag(turn, frag))

        one_each = await store.search_hybrid("rollout decision", top_k=10, collapse_key="$.turn_id")
        assert len(one_each.results) == 3  # one best row per turn unit

        two_each = await store.search_hybrid(
            "rollout decision", top_k=20, collapse_key="$.turn_id", collapse_max_per_unit=2
        )
        assert len(two_each.results) == 6  # up to two rows per unit

        composite = await store.search_hybrid(
            "rollout decision", top_k=10, collapse_key=["$.session_id", "$.turn_index"]
        )
        assert len(composite.results) == 3  # (session, turn) composite units


async def test_search_hybrid_accepts_all_weight_parameters() -> None:
    """search.md 'Tune the blend' — search_hybrid accepts every scoring weight."""
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        await store.create_thought(_observation())

        result = await store.search_hybrid(
            "Python",
            fts_weight=0.4,
            vector_weight=0.4,
            recency_weight=0.1,
            priority_weight=0.05,
            graph_weight=0.05,
            current_cycle=42,
        )
        assert isinstance(result.results, list)
        assert isinstance(result.backends_used, frozenset)


async def test_api_reference_action_lifecycle_and_outcome_score() -> None:
    """api-reference.md action lifecycle — update_action + action_outcome_score.

    A stored action advances PLANNED -> EXECUTING -> CONFIRMED; a terminal
    status recomputes the source thought's ``action_outcome_score``; a
    verification-only update is allowed on a terminal action.
    """
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        source = await store.create_thought(_observation())

        action = ActionRecord(
            action_id=str(uuid.uuid4()),
            source_thought_id=source.thought_id,
            action_type=ActionType.TOOL_CALL,
            intent="search the web for flight prices",
            status=ActionStatus.PLANNED,
            verification_status=VerificationStatus.PENDING,
        )
        await store.create_action(action)
        await store.update_action(action.action_id, status=ActionStatus.EXECUTING)
        await store.update_action(action.action_id, status=ActionStatus.CONFIRMED)
        await store.update_action(
            action.action_id, verification_status=VerificationStatus.CONFIRMED
        )

        actions = await store.get_actions(source.thought_id)
        assert len(actions) == 1
        assert actions[0].status is ActionStatus.CONFIRMED

        refreshed = await store.get_thought(source.thought_id)
        assert refreshed is not None
        # A terminal, confirmed action drives the source thought's outcome score.
        assert refreshed.action_outcome_score == pytest.approx(1.0)


async def test_metadata_helpers_return_documented_shapes() -> None:
    """quickstart + api-reference percept() / utterance() / thought() shapes.

    ``percept`` carries the external ``source`` block; ``utterance`` / ``thought``
    are self-anchored; all three tag ``lang`` and ``content_type``.
    """
    p = percept(source_id="user-42", label="user")
    assert p["perspective"] == "percept"
    assert p["source"] == {
        "is_self": False,
        "confidence": "high",
        "id": "user-42",
        "label": "user",
    }
    assert p["lang"] == "en"
    assert p["content_type"] == "natural_language"

    u = utterance()
    assert u["perspective"] == "utterance"
    assert u["source"] == {"is_self": True, "confidence": "high"}

    t = thought()
    assert t["perspective"] == "thought"
    assert t["source"] == {"is_self": True, "confidence": "high"}


async def test_store_execute_mindql_returns_rows() -> None:
    """api-reference.md store.execute_mindql — runs a parsed query on the store."""
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        await store.create_thought(_observation())

        result = await store.execute_mindql(
            parse("FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10")
        )
        assert isinstance(result.rows, list)
        assert len(result.rows) == 1


def test_mindql_parse_exposes_query_fields() -> None:
    """mindql.md + api-reference parse() -> MindQLQuery command/table/limit.

    ``parse`` returns a ``MindQLQuery`` whose ``command`` / ``table`` /
    ``conditions`` / ``limit`` describe the query; a bad query raises
    ``MindQLParseError``.
    """
    query = parse("FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 5")
    assert query.command is MindQLCommand.FIND
    assert query.table == "thought"
    assert query.limit == 5
    assert query.conditions  # at least the thought_type predicate

    with pytest.raises(MindQLParseError):
        parse("TOTALLY NOT MINDQL")


async def test_mindql_raw_sql_select_and_explain() -> None:
    """mindql.md raw SELECT + EXPLAIN.

    A ``MindQLQuery`` carrying ``command=SELECT`` runs its parameterised
    ``raw_sql``; ``EXPLAIN`` returns a single plan row exposing ``sql`` +
    ``params`` without touching data.
    """
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        await store.create_thought(_observation())  # P2

        executor = MindQLExecutor(conn)
        select = await executor.execute(
            MindQLQuery(
                command=MindQLCommand.SELECT,
                raw_sql="SELECT thought_id FROM thought WHERE priority = ?",
                select_params=("P2",),
            )
        )
        assert len(select.rows) == 1

        explained = await executor.execute(parse("EXPLAIN FIND thoughts WHERE priority = 'P1'"))
        plan = explained.rows[0]
        assert "sql" in plan
        assert "params" in plan
        assert plan["params"] == ["P1"]


async def test_audit_trail_verify_journal_and_entries() -> None:
    """audit-trail.md verify_journal + journal.get_entries.

    With journaling on, a create + update records INSERT/UPDATE entries; the
    hash-chain verifies (``verify_journal().valid`` and
    ``journal.verify_integrity()``), reporting ``entries_checked``.
    """
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn, journal_enabled=True)
        note = _observation("User prefers email over phone")
        await store.create_thought(note)
        await store.update_thought(note.thought_id, essence="User strongly prefers email")

        assert store.journal is not None
        entries = await store.journal.get_entries(target_id=note.thought_id)
        assert [e.mutation_type for e in entries] == ["INSERT_THOUGHT", "UPDATE_THOUGHT"]

        chain = await store.journal.verify_integrity()
        assert chain.valid
        assert chain.entries_checked == 2

        top = await store.verify_journal()
        assert top.valid
        assert top.entries_checked >= 2


async def test_data_lifecycle_count_and_cleanup_result() -> None:
    """data-lifecycle.md count_thoughts(include_expired) + cleanup_expired result.

    ``count_thoughts`` accepts ``include_expired``; ``cleanup_expired`` returns a
    ``CleanupResult`` with ``expired_count`` / ``strategy_applied`` / ``timestamp``
    (default strategy ``archive``).
    """
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        await store.create_thought(_observation())

        assert await store.count_thoughts() == 1
        assert await store.count_thoughts(include_expired=True) == 1

        result = await store.cleanup_expired()
        assert isinstance(result.expired_count, int)
        assert result.strategy_applied == "archive"
        assert result.timestamp  # ISO-8601 stamp of the pass


async def test_dreaming_consolidate_result_fields() -> None:
    """dreaming.md store.consolidate() + run_consolidation result fields.

    A from_config-less store has no dreaming wired, so ``consolidate`` raises;
    a ``DreamingExtension.run_consolidation`` returns a ConsolidationResult
    exposing promoted_count / edges_created / reflections_created.
    """
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        await store.create_thought(_observation())

        # consolidate() requires dreaming enabled via from_config.
        with pytest.raises(RuntimeError):
            await store.consolidate(current_cycle=1)

        ext = DreamingExtension(
            config=DreamingConfig(enabled=True, promote_threshold=0.55),
        )
        result = await ext.run_consolidation(store, current_cycle=42)
        assert isinstance(result.promoted_count, int)
        assert isinstance(result.edges_created, int)
        assert isinstance(result.reflections_created, int)


def test_hooks_protocol_conformance() -> None:
    """README + extensions + extension-hooks custom hooks satisfy the protocol.

    A ``DefaultEngravaHooks`` subclass and an ``EngravaHooksProtocol``
    implementation both satisfy the runtime-checkable protocol; a custom
    ``score_function`` returns the documented value.
    """

    class RecencyBoostHooks(DefaultEngravaHooks):
        async def score_function(self, thought: ThoughtRecord, context: ScoringContext) -> float:
            if context.current_cycle > 0:
                age = context.current_cycle - thought.updated_cycle
                return max(0.0, 1.0 - age / 100)
            return 0.0

    class MyHooks(EngravaHooksProtocol):
        async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
            return thought

        async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
            return thought

        async def score_function(self, thought: ThoughtRecord, context: ScoringContext) -> float:
            return thought.confidence or 0.5

        async def decay_function(self, thought: ThoughtRecord, elapsed_cycles: int) -> float:
            return 1.0

        def mindql_extension_registry(self) -> dict[str, MindQLExtension]:
            return {}

    assert isinstance(RecencyBoostHooks(), EngravaHooksProtocol)
    assert isinstance(MyHooks(), EngravaHooksProtocol)


async def test_extension_hooks_recency_score_function_value() -> None:
    """extension-hooks.md RecencyBoostHooks — the documented recency formula."""

    class RecencyBoostHooks(DefaultEngravaHooks):
        async def score_function(self, thought: ThoughtRecord, context: ScoringContext) -> float:
            if context.current_cycle > 0:
                age = context.current_cycle - thought.updated_cycle
                return max(0.0, 1.0 - age / 100)
            return 0.0

    hooks = RecencyBoostHooks()
    recent = ThoughtRecord(**{**_observation().__dict__, "updated_cycle": 95})
    score = await hooks.score_function(recent, ScoringContext(current_cycle=100))
    assert score == pytest.approx(1.0 - 5 / 100)


def test_embedding_role_prefixes_are_exposed() -> None:
    """guides/embeddings.md asymmetric role prefixes.

    ``SentenceTransformerProvider(query_prefix=..., document_prefix=...)`` stores
    and exposes the prefixes (no model load required to read them).
    """
    provider = SentenceTransformerProvider(
        model_name="intfloat/e5-base-v2",
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    assert provider.query_prefix == "query: "
    assert provider.document_prefix == "passage: "


async def test_memory_hygiene_pinned_and_run_hygiene() -> None:
    """memory-hygiene.md pinned thought + run_hygiene (default-off, dry-run preview).

    ``pinned=True`` is persisted; ``run_hygiene`` needs a configured policy
    (RuntimeError otherwise); a disabled policy forgets nothing; a ``dry_run``
    preview mutates nothing and returns per-thought EvictionReason previews with
    ``thought_id`` / ``eviction_score`` / ``signals``.
    """
    # No policy configured -> run_hygiene refuses.
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        with pytest.raises(RuntimeError):
            await store.run_hygiene(current_cycle=1000)

    # pinned persists.
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        pinned = ThoughtRecord(**{**_observation().__dict__, "pinned": True})
        await store.create_thought(pinned)
        fetched = await store.get_thought(pinned.thought_id)
        assert fetched is not None
        assert fetched.pinned is True

    # Disabled policy: master switch off -> nothing archived.
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn, hygiene_policy=HygienePolicyConfig(enabled=False))
        await store.create_thought(_observation())
        off = await store.run_hygiene(current_cycle=1000)
        assert off.archived_count == 0

    # Dry-run preview: mutates nothing, returns would-evict reasons.
    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(
            conn,
            hygiene_policy=HygienePolicyConfig(enabled=True, dry_run=True),
            access_tracking_enabled=True,
        )
        for cycle in range(5):
            await store.create_thought(
                ThoughtRecord(
                    **{**_observation().__dict__, "created_cycle": cycle, "updated_cycle": cycle}
                )
            )
        preview = await store.run_hygiene(current_cycle=1000)
        assert preview.dry_run is True
        assert preview.archived_count == 0  # dry run mutates nothing
        assert preview.candidates_evaluated == 5
        assert isinstance(preview.would_evict, list)
        for reason in preview.would_evict:
            assert isinstance(reason.thought_id, str)
            assert isinstance(reason.eviction_score, float)
            assert isinstance(reason.signals, dict)
        # Nothing was actually archived.
        assert await store.count_thoughts(lifecycle_status="ARCHIVED") == 0


async def test_troubleshooting_thought_type_member_access() -> None:
    """troubleshooting.md ThoughtType member access.

    ``ThoughtType.BELIEF`` and ``ThoughtType("BELIEF")`` name the same member.
    """
    assert ThoughtType("BELIEF") is ThoughtType.BELIEF


async def test_troubleshooting_referential_integrity_error() -> None:
    """troubleshooting.md ReferentialIntegrityError import + raise.

    The exception imports from ``engrava.domain.exceptions`` (not the top-level
    package) and ``create_edge`` raises it when an endpoint is missing.
    """
    from engrava.domain.exceptions import ReferentialIntegrityError

    async with aiosqlite.connect(":memory:") as conn:
        store = await _fresh_store(conn)
        with pytest.raises(ReferentialIntegrityError):
            await store.create_edge(
                EdgeRecord(
                    edge_id=str(uuid.uuid4()),
                    from_thought_id="does-not-exist",
                    to_thought_id="also-missing",
                    edge_type=EdgeType.ASSOCIATED,
                    weight=0.5,
                    created_cycle=0,
                )
            )


def _import_row(text: str) -> ThoughtRecord:
    """Build the record shape the migration guide's ``to_thought`` helper returns."""
    return ThoughtRecord(
        thought_id=str(uuid.uuid4()),
        thought_type=ThoughtType.OBSERVATION,
        essence=text[:200],
        content=text,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="import",
    )


class _OneChildPerSource(DefaultEngravaHooks):
    """A minimal derived-records producer: exactly one child per stored source.

    Counts its own invocations so a test can observe *whether the seam called
    the producer*, which a stored-row count cannot: derived children are
    content-addressed, so once a child exists, deriving it again writes nothing
    and looks exactly like not deriving it again.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> list[DerivedRecord]:
        """Return one derived child for the given source thought."""
        self.calls += 1
        return [
            DerivedRecord(
                content=f"derived from {thought.content}",
                thought_type=ThoughtType.OBSERVATION,
                priority=Priority.P3,
                attach_provenance_edge=True,
            )
        ]


_MIGRATION_GUIDE = "docs/guides/migrating-from-other-memory.md"


def _documented_bulk_import_and_derive() -> Callable[..., Awaitable[int]]:
    """Return the guide's own ``bulk_import_and_derive``, compiled from the page.

    The rest of this module mirrors its examples by hand, which is what its
    header warns can drift. Here the drift matters: the claim under test is that
    the page's recipe ends with derived records, so a mirror carrying its own
    backfill loop would stay green after the page lost one. Execute the page's
    function instead, together with the ``to_thought`` helper it calls.
    """
    path = REPO_ROOT / _MIGRATION_GUIDE
    blocks = extract_python_blocks(path)
    wanted = ("def to_thought(", "async def bulk_import_and_derive")
    namespace: dict[str, object] = {
        "uuid": uuid,
        "ThoughtRecord": ThoughtRecord,
        "ThoughtType": ThoughtType,
        "Priority": Priority,
        "LifecycleStatus": LifecycleStatus,
    }
    for anchor in wanted:
        matches = [b for b in blocks if anchor in b.body]
        assert len(matches) == 1, (
            f"Expected exactly one block in {_MIGRATION_GUIDE} containing "
            f"{anchor!r}, found {len(matches)}."
        )
        exec(compile(matches[0].body, matches[0].location, "exec"), namespace)  # noqa: S102 -- repo-authored documentation snippet
    return cast("Callable[..., Awaitable[int]]", namespace["bulk_import_and_derive"])


async def test_migration_guide_bulk_import_derives_only_after_backfill() -> None:
    """migrating-from-other-memory.md 'If you use the derived-records seam'.

    The bulk-import recipe wraps the whole import in ``suspend_auto_commit()``
    and passes ``deduplicate=True``. The guide states that neither path derives —
    a source that is not yet durable is not dispatched, and a dedup hit stores no
    new thought — and that ``derive_existing`` backfills them once the
    transaction has committed. Both halves are asserted here, because "no derived
    records" is equally satisfied by a producer that never runs at all: the same
    producer is therefore shown deriving on the ordinary write path first.
    """
    exported = ["User prefers dark mode", "User is based in Berlin", "User prefers dark mode"]
    control_total = 2  # the control source plus its one derived child
    after_import_total = 4  # control pair + 2 distinct imported sources, no children
    after_backfill_total = 6  # control pair + 2 sources + one child each
    # The control write, then one backfill per record the import returned — three,
    # because the dedup hit returned the existing source a second time. The extra
    # invocation adds no row: derived children are content-addressed, so a repeat
    # derivation of the same source converges on the child already stored.
    backfilled_calls = 4

    producer = _OneChildPerSource()
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            conn,
            hooks=producer,
            derive_gates=DeriveGates(enabled=True),
        )
        await store.ensure_schema()

        # Control: on the ordinary write path this producer does derive, so a
        # zero count below cannot be blamed on an inert seam.
        await store.create_thought(_import_row("a note written the ordinary way"))
        assert await store.count_thoughts() == control_total
        assert producer.calls == 1

        # The import half of the page's own recipe, run block by block so the
        # count below reports the state the page leaves behind.
        async with store.suspend_auto_commit():
            stored = [
                await store.create_thought(_import_row(text), deduplicate=True) for text in exported
            ]

        # 3 exported rows, one duplicate collapsed -> 2 sources, and no child for
        # either: the recipe as written derives nothing.
        assert await store.count_thoughts() == after_import_total
        assert producer.calls == 1  # the seam never reached the producer

        for thought in stored:
            await store.derive_existing(thought.thought_id)

        assert await store.count_thoughts() == after_backfill_total
        assert producer.calls == backfilled_calls

        # The second half of the guide's claim, pinned independently of the
        # suspended-commit window: a dedup hit on the ordinary write path stores
        # no new thought, so the seam never reaches the producer for it. The row
        # count cannot show this — derived children are content-addressed, so a
        # second derivation of the same source converges on the existing child.
        await store.create_thought(_import_row(exported[0]), deduplicate=True)
        assert await store.count_thoughts() == after_backfill_total
        assert producer.calls == backfilled_calls


async def test_migration_guide_backfill_recipe_runs_and_derives() -> None:
    """The guide's ``bulk_import_and_derive`` function, executed from the page.

    The sibling test above establishes *why* the recipe is needed; this one runs
    the recipe the page actually publishes, so deleting or weakening the page's
    backfill loop reds here — which a hand-written mirror of the same sequence
    could not do.

    Note what it deliberately does **not** pin: where the loop sits. The page
    prescribes running it after the import block, and this test asserts the
    outcome that placement produces, not the placement itself.
    """
    exported = [
        {"text": "User prefers dark mode", "user": "u1"},
        {"text": "User is based in Berlin", "user": "u1"},
        {"text": "User prefers dark mode", "user": "u1"},
    ]
    distinct_sources = 2
    sources_plus_children = 4

    bulk_import_and_derive = _documented_bulk_import_and_derive()
    producer = _OneChildPerSource()
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            conn,
            hooks=producer,
            derive_gates=DeriveGates(enabled=True),
        )
        await store.ensure_schema()

        total = await bulk_import_and_derive(store, exported)

        # Every distinct imported source ends up with its derived child, which is
        # the outcome the page promises and the plain recipe above does not reach.
        assert await store.count_thoughts() == sources_plus_children
        # ...and each of those children is attached to *its own* source by the
        # DERIVED_FROM edge the page names. Both endpoints, as a frozen set of
        # pairs: counting edges per source is satisfied by two edges leaving one
        # child, and says nothing about which child each one belongs to.
        cursor = await conn.execute(
            "SELECT from_thought_id, to_thought_id FROM edge WHERE edge_type = ?",
            (EdgeType.DERIVED_FROM.value,),
        )
        edges = {(str(row[0]), str(row[1])) for row in await cursor.fetchall()}
        rows = await store.list_thoughts(limit=sources_plus_children)
        sources = {t.thought_id: t.content for t in rows if not t.content.startswith("derived ")}
        children = {t.content: t.thought_id for t in rows if t.content.startswith("derived ")}
        assert len(children) == len(sources)
        assert edges == {
            (children[f"derived from {content}"], source_id)
            for source_id, content in sources.items()
        }
        assert total == sources_plus_children
        # Exactly one derivation per distinct source: the page deduplicates the
        # ids before backfilling, so a repeated row does not re-derive.
        assert producer.calls == distinct_sources


_TUTORIAL = "docs/tutorial.md"


def _tutorial_block(anchor: str) -> CodeBlock:
    """Return the single block of docs/tutorial.md containing ``anchor``."""
    matches = [b for b in extract_python_blocks(REPO_ROOT / _TUTORIAL) if anchor in b.body]
    assert len(matches) == 1, (
        f"Expected exactly one block in {_TUTORIAL} containing {anchor!r}, found {len(matches)}."
    )
    return matches[0]


def _tutorial_pieces() -> dict[str, object]:
    """Compile the tutorial's own helpers from the page, in document order."""
    namespace: dict[str, object] = {}
    for anchor in (
        "def embed(text: str) -> list[float]:",
        "NOTES = [",
        "async def link(",
        "async def search(",
    ):
        block = _tutorial_block(anchor)
        exec(compile(block.body, block.location, "exec"), namespace)  # noqa: S102 -- repo-authored documentation snippet
    return namespace


def _tutorial_linked_pair() -> tuple[int, int]:
    """Return the note indices the tutorial's ``main()`` links together.

    Read out of the page rather than copied, so a page that links a different
    pair — or stops linking — is exercised as it is written.
    """
    module = ast.parse(_tutorial_block("asyncio.run(main())").body)
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "link"
        ):
            indices = [
                arg.slice.value
                for arg in node.args
                if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant)
            ]
            assert len(indices) == 2, f"Expected two linked notes in {_TUTORIAL}."
            return (int(indices[0]), int(indices[1]))
    pytest.fail(f"{_TUTORIAL}'s main() no longer links two notes.")


def _tutorial_transcript() -> str:
    """Return the ``text`` block docs/tutorial.md publishes as its own output."""
    blocks = extract_fenced_blocks(REPO_ROOT / _TUTORIAL, "text")
    matches = [b for b in blocks if b.body.lstrip().startswith("Query: ")]
    assert len(matches) == 1, f"Expected one transcript block in {_TUTORIAL}."
    return matches[0].body


def _tutorial_query() -> str:
    """Return the query the page's published transcript shows it running."""
    return _tutorial_transcript().lstrip().splitlines()[0].split("'")[1]


def _tutorial_ablated_score() -> str:
    """Return the tied score the tutorial names for its vector-ablated run."""
    raw = (REPO_ROOT / _TUTORIAL).read_text(encoding="utf-8")
    # Collapse the wrapping so the sentence matches however Markdown broke it.
    prose = " ".join(raw.split())
    match = re.search(r"one and the same score \(`([0-9.]+)`, to three decimals\)", prose)
    assert match is not None, (
        f"{_TUTORIAL} no longer states the score its ablation ties at; this test "
        f"reads that number from the page rather than holding its own copy."
    )
    return match.group(1)


def _documented_transcript_body() -> str:
    """Return the transcript's search section — everything before the stored count."""
    return _tutorial_transcript().split("\n\nStored")[0].strip()


def _ingest_from_page() -> object:
    """Compile the tutorial's ``ingest`` with the names its block relies on."""
    namespace: dict[str, object] = {
        "uuid": uuid,
        "ThoughtRecord": ThoughtRecord,
        "ThoughtType": ThoughtType,
        "Priority": Priority,
        "LifecycleStatus": LifecycleStatus,
    }
    block = _tutorial_block("async def ingest(")
    exec(compile(block.body, block.location, "exec"), namespace)  # noqa: S102 -- repo-authored documentation snippet
    return namespace["ingest"]


async def test_tutorial_hash_vectors_are_what_separate_the_lower_notes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """tutorial.md: "run the same query with vector_weight=0.0 and those three tie".

    The page explains its own ranking by saying the arbitrary hash-vector scores
    are what separates everything below the top hit. That is a causal claim about
    one arm of a fused ranking, and the published transcript cannot show it — the
    transcript also carries a priority signal, which could just as well have
    produced the order. So run the ablation the page names.

    The scenario is the page's: its ``embed``, its ``NOTES``, its ``ingest``, its
    ``link`` between the pair its ``main()`` names, its ``search``, the dimension
    its ``embed`` returns, and the query its transcript shows. Before ablating
    anything, the store is driven through the page's own ``search`` and the
    printed result is required to equal the transcript the page publishes — so
    the thing being ablated is demonstrably the page's scenario, not a lookalike.
    A page-only change to its corpus, its priorities, its linking or its search
    call then fails here rather than leaving this paragraph quietly false.

    The lower scores are compared **exactly**, not rounded: "tie" is the claim,
    and three distinct scores that round alike would satisfy a rounded one.
    """
    pieces = _tutorial_pieces()
    embed = cast("Callable[[str], list[float]]", pieces["embed"])
    notes = cast("list[str]", pieces["NOTES"])
    link = cast("Callable[..., Awaitable[None]]", pieces["link"])
    search = cast("Callable[..., Awaitable[None]]", pieces["search"])
    ingest = cast("Callable[..., Awaitable[list[ThoughtRecord]]]", _ingest_from_page())
    first, second = _tutorial_linked_pair()
    documented = _documented_transcript_body()

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        provider = CallbackProvider(
            callback=embed,
            dimension=len(embed("probe")),
            model_name="tutorial",
        )
        store = SqliteEngravaCore(conn, embedding_provider=provider, auto_embed=True)
        await store.ensure_schema()
        stored = await ingest(store, notes)
        await link(store, stored[first], stored[second])

        capsys.readouterr()
        await search(store, _tutorial_query(), len(notes))
        printed = capsys.readouterr().out.strip()

        ablated = await store.search_hybrid(
            _tutorial_query(),
            top_k=len(notes),
            current_cycle=len(notes),
            vector_weight=0.0,
        )

    # Precondition: this store reproduces the page. Without it the ablation below
    # would be a statement about some other corpus that resembles the tutorial's.
    assert printed == documented, (
        f"This scenario no longer reproduces {_TUTORIAL}'s published output, so "
        f"the ablation below would say nothing about that page.\n"
        f"--- documented ---\n{documented}\n--- produced ---\n{printed}"
    )

    scores = [score for _, score in ablated.results]
    assert len(scores) == len(notes)
    # Everything below the top hit collapses to a single identical score once the
    # arbitrary vector contribution is removed, so nothing but that arm was
    # ordering them in the output the page publishes.
    assert len(set(scores[1:])) == 1
    assert scores[0] > scores[1]
    # The page prints that number; pin it, or the prose could name any value.
    assert f"{scores[1]:.3f}" == _tutorial_ablated_score()


def test_upgrade_page_publishes_the_error_the_code_actually_raises() -> None:
    """upgrade.md prints the provider-contract error verbatim, so pin it to the code.

    The page shows the exact text a reader will see, which is the most checkable
    kind of documentation and the easiest to leave behind: the wording lives in
    the exception, and nothing else stops the two drifting. Rebuild the message
    from the class and compare, ignoring only how Markdown wrapped it.
    """
    blocks = extract_fenced_blocks(REPO_ROOT / "docs/upgrade.md", "text")
    matches = [b for b in blocks if "EmbeddingProviderProtocol member" in b.body]
    assert len(matches) == 1, (
        f"Expected exactly one published EmbeddingProviderContractError message in "
        f"docs/upgrade.md, found {len(matches)}."
    )

    published = " ".join(matches[0].body.split())
    raised = str(EmbeddingProviderContractError(provider_class="MyProvider", member="dimension"))

    assert published == raised


async def _rank_with_optional_edge(
    *,
    edge: bool,
    graph_weight: float | None,
    with_config: bool,
) -> tuple[list[str], set[str]]:
    """Rank a small corpus, optionally linking two of its notes.

    ``with_config`` selects which default the store resolves the graph weight
    from: a store built without a ``SearchConfig`` falls back to a literal in the
    core, while one built with the default ``SearchConfig()`` reads
    ``default_graph_weight``. The page names the latter; both must hold.

    Returns:
        The ranked contents and the signals the search reports using.
    """
    texts = [
        "Coffee tastes better with freshly ground beans.",
        "Buy oat milk and coffee beans on the way home.",
        "The espresso machine descaling is overdue.",
    ]
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, search_config=SearchConfig() if with_config else None)
        await store.ensure_schema()
        stored = [
            await store.create_thought(
                ThoughtRecord(
                    thought_id=str(uuid.uuid4()),
                    thought_type=ThoughtType.OBSERVATION,
                    essence=text[:200],
                    content=text,
                    priority=Priority.P3,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    created_cycle=index,
                    updated_cycle=index,
                    source="notes",
                )
            )
            for index, text in enumerate(texts)
        ]
        if edge:
            await store.create_edge(
                EdgeRecord(
                    edge_id=str(uuid.uuid4()),
                    from_thought_id=stored[2].thought_id,
                    to_thought_id=stored[1].thought_id,
                    edge_type=EdgeType.ASSOCIATED,
                    weight=0.9,
                    created_cycle=0,
                )
            )
        kwargs = {} if graph_weight is None else {"graph_weight": graph_weight}
        result = await store.search_hybrid(
            "coffee beans espresso",
            top_k=3,
            current_cycle=3,
            **kwargs,  # type: ignore[arg-type]
        )
        by_id = {t.thought_id: t.content for t in stored}
        return [by_id[thought_id] for thought_id, _ in result.results], set(result.backends_used)


async def test_migration_guide_edges_do_not_feed_ranking_at_defaults() -> None:
    """migrating-from-other-memory.md: letting edges feed ranking is opt-in.

    The concept table used to say edges "also feed ranking"; it now says the
    graph signal is opt-in at ``default_graph_weight = 0.0``. That is a claim
    about ranked output, so pin it as a frozen order: the same corpus and query,
    with and without an ``ASSOCIATED`` edge, rank identically at defaults — and
    the search reports no graph signal at all.

    Both stores are exercised, because the two resolve the default from
    different places: one from ``SearchConfig.default_graph_weight``, which the
    page names, and one from the core's config-less fallback.

    The opted-in half is asserted in the same test rather than left to a
    mutation, because it is the page's own point. Without it, "the edge changed
    nothing" would be equally satisfied by a corpus in which nothing could ever
    change.
    """
    # The page names this value; assert it, then show what it does.
    assert SearchConfig().default_graph_weight == 0.0
    # A seam that never runs would also leave the recipe complete; pin the value
    # the page relies on for that claim as well.
    assert DeriveGates().enabled is False

    # The frozen default order, written down rather than compared to itself: two
    # rankings that agree prove nothing if a change moves both.
    frozen_default = [
        "The espresso machine descaling is overdue.",
        "Coffee tastes better with freshly ground beans.",
        "Buy oat milk and coffee beans on the way home.",
    ]

    for with_config in (False, True):
        without_edge, signals_without = await _rank_with_optional_edge(
            edge=False, graph_weight=None, with_config=with_config
        )
        with_edge, signals_with = await _rank_with_optional_edge(
            edge=True, graph_weight=None, with_config=with_config
        )
        assert with_edge == without_edge
        assert "graph" not in signals_without
        assert "graph" not in signals_with
        if not with_config:
            # The store the page's own defaults produce. (With a SearchConfig the
            # recency arm also activates — a separate, out-of-scope inconsistency
            # between that default and the config-less one — so its order is only
            # required to be edge-independent, not to equal this.)
            assert without_edge == frozen_default

        # Opting in turns the signal on, for the very same store and edge — so
        # the absences above are the default weight's doing.
        _, opted_in_signals = await _rank_with_optional_edge(
            edge=True, graph_weight=0.4, with_config=with_config
        )
        assert "graph" in opted_in_signals

    # ...and on the config-less path the opted-in signal actually moves the order.
    baseline, _ = await _rank_with_optional_edge(edge=False, graph_weight=None, with_config=False)
    opted_in, _ = await _rank_with_optional_edge(edge=True, graph_weight=0.4, with_config=False)
    assert opted_in != baseline

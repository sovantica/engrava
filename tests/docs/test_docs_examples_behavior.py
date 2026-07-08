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

import uuid

import aiosqlite
import pytest

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    CallbackProvider,
    DefaultEngravaHooks,
    DreamingConfig,
    DreamingContext,
    DreamingExtension,
    DreamingGates,
    EdgeRecord,
    EdgeType,
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

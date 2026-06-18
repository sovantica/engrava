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
    CallbackProvider,
    DreamingConfig,
    DreamingContext,
    DreamingExtension,
    DreamingGates,
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    MindQLExecutor,
    MindQLExtension,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    parse,
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

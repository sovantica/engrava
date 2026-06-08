"""Unit tests for the MCP read-tool implementations.

Each tool implementation is exercised directly against a seeded
in-memory store (see ``conftest.store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)
from engrava.mcp.server import (
    DEFAULT_TOP_K,
    StoreNotReadyError,
    StoreProvider,
    UnsupportedQueryError,
    get_thought_impl,
    list_memory_impl,
    memory_stats_impl,
    query_memory_impl,
    search_keywords_impl,
    search_memory_impl,
)
from engrava.mindql.parser import MindQLParseError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class TestGetThought:
    """Tests for the ``get_thought`` tool."""

    async def test_returns_serialised_thought(self, store: SqliteEngravaCore) -> None:
        result = await get_thought_impl(store, "thought-alpha")
        assert result["found"] is True
        thought = result["thought"]
        assert thought is not None
        assert thought["thought_id"] == "thought-alpha"
        assert thought["essence"] == "Coffee brewing notes"
        # The payload must be JSON-friendly (the enum serialised to its value).
        assert thought["lifecycle_status"] == "ACTIVE"

    async def test_missing_thought_reports_not_found(self, store: SqliteEngravaCore) -> None:
        result = await get_thought_impl(store, "does-not-exist")
        assert result == {"found": False, "thought": None}


class TestSearchMemory:
    """Tests for the ``search_memory`` tool."""

    async def test_returns_results_and_backends(self, store: SqliteEngravaCore) -> None:
        result = await search_memory_impl(store, "coffee")

        assert [entry["thought_id"] for entry in result["results"]] == ["thought-alpha"]
        assert all(isinstance(entry["score"], float) for entry in result["results"])
        # No embedding provider / vector backend was configured, so the
        # vector backend must not appear in the diagnostics.
        assert "vector" not in result["backends_used"]
        assert "fts5" in result["backends_used"]

    async def test_respects_top_k(self, store: SqliteEngravaCore) -> None:
        result = await search_memory_impl(store, "notes", top_k=1)
        assert len(result["results"]) <= 1

    async def test_exclude_reflections_flag_is_passed(self, store: SqliteEngravaCore) -> None:
        result = await search_memory_impl(store, "tea", include_reflections=False)
        assert [entry["thought_id"] for entry in result["results"]] == ["thought-beta"]


class TestSearchKeywords:
    """Tests for the ``search_keywords`` tool."""

    async def test_returns_ranked_matches(self, store: SqliteEngravaCore) -> None:
        result = await search_keywords_impl(store, "tea")
        assert [entry["thought_id"] for entry in result["results"]] == ["thought-beta"]
        assert all(isinstance(entry["score"], float) for entry in result["results"])

    async def test_no_match_returns_empty(self, store: SqliteEngravaCore) -> None:
        result = await search_keywords_impl(store, "spaceship")
        assert result["results"] == []


class TestQueryMemory:
    """Tests for the ``query_memory`` MindQL tool (FIND only)."""

    async def test_find_returns_rows(self, store: SqliteEngravaCore) -> None:
        result = await query_memory_impl(
            store,
            "FIND thoughts WHERE lifecycle_status = 'ACTIVE'",
        )
        ids = {row["thought_id"] for row in result["rows"]}
        assert ids == {"thought-alpha", "thought-beta"}
        assert "thought_id" in result["columns"]

    async def test_find_with_explicit_limit_override(self, store: SqliteEngravaCore) -> None:
        result = await query_memory_impl(
            store,
            "FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 5",
            limit=1,
        )
        assert len(result["rows"]) == 1

    async def test_select_is_rejected(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(UnsupportedQueryError) as excinfo:
            await query_memory_impl(store, "SELECT * FROM thought")
        assert excinfo.value.command == "SELECT"

    async def test_count_is_rejected(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(UnsupportedQueryError) as excinfo:
            await query_memory_impl(store, "COUNT thoughts")
        assert excinfo.value.command == "COUNT"

    async def test_malformed_query_raises_parse_error(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(MindQLParseError):
            await query_memory_impl(store, "")


class TestMemoryStats:
    """Tests for the ``memory_stats`` tool."""

    async def test_reports_counts(self, store: SqliteEngravaCore) -> None:
        result = await memory_stats_impl(store)
        assert result["thought_count"] == 2
        assert result["metrics"]["thoughts"]["total"] == 2
        assert result["metrics"]["edges"]["total"] == 0
        assert result["metrics"]["thoughts"]["by_status"]["ACTIVE"] == 2
        assert isinstance(result["metrics"]["storage_total_bytes"], int)


class TestStoreProvider:
    """Tests for the ``StoreProvider`` lifecycle holder."""

    def test_require_without_store_raises(self) -> None:
        provider = StoreProvider()
        with pytest.raises(StoreNotReadyError):
            provider.require()

    def test_set_then_require(self, store: SqliteEngravaCore) -> None:
        provider = StoreProvider()
        provider.set(store)
        assert provider.require() is store

    def test_clear_resets(self, store: SqliteEngravaCore) -> None:
        provider = StoreProvider()
        provider.set(store)
        provider.clear()
        with pytest.raises(StoreNotReadyError):
            provider.require()


class TestSearchMemoryFilters:
    """Tests for the optional filters on the ``search_memory`` tool.

    The shared ``store`` fixture seeds two ``BELIEF`` thoughts, both
    ``ACTIVE``: ``thought-alpha`` at ``P2`` and ``thought-beta`` at ``P1``.
    A query for "notes" ranks both (each essence ends in "notes"), which
    lets a single filter drop exactly one ranked hit.
    """

    async def test_unfiltered_call_is_unchanged(self, store: SqliteEngravaCore) -> None:
        # The unfiltered response must keep its original shape exactly: a
        # results list and backends_used, and crucially no ``filtered``
        # block (that key only appears once a filter is supplied).
        result = await search_memory_impl(store, "notes")
        assert set(result) == {"results", "backends_used"}
        assert "filtered" not in result
        assert {entry["thought_id"] for entry in result["results"]} == {
            "thought-alpha",
            "thought-beta",
        }

    async def test_priority_filter_keeps_only_matching(self, store: SqliteEngravaCore) -> None:
        result = await search_memory_impl(store, "notes", priority=Priority.P1)
        assert [entry["thought_id"] for entry in result["results"]] == ["thought-beta"]

    async def test_lifecycle_filter_keeps_only_matching(self, store: SqliteEngravaCore) -> None:
        # Both seeded thoughts are ACTIVE, so an ACTIVE filter keeps both
        # while a DONE filter keeps none.
        active = await search_memory_impl(store, "notes", lifecycle_status=LifecycleStatus.ACTIVE)
        assert {entry["thought_id"] for entry in active["results"]} == {
            "thought-alpha",
            "thought-beta",
        }
        done = await search_memory_impl(store, "notes", lifecycle_status=LifecycleStatus.DONE)
        assert done["results"] == []

    async def test_thought_type_filter_keeps_only_matching(self, store: SqliteEngravaCore) -> None:
        # Both seeded thoughts are BELIEF; a TASK filter matches nothing.
        belief = await search_memory_impl(store, "notes", thought_type=ThoughtType.BELIEF)
        assert {entry["thought_id"] for entry in belief["results"]} == {
            "thought-alpha",
            "thought-beta",
        }
        task = await search_memory_impl(store, "notes", thought_type=ThoughtType.TASK)
        assert task["results"] == []

    async def test_combined_filters_apply_as_and(self, store: SqliteEngravaCore) -> None:
        # P1 AND ACTIVE matches only beta; P1 AND DONE matches nothing.
        match = await search_memory_impl(
            store,
            "notes",
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.ACTIVE,
        )
        assert [entry["thought_id"] for entry in match["results"]] == ["thought-beta"]
        miss = await search_memory_impl(
            store,
            "notes",
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.DONE,
        )
        assert miss["results"] == []

    async def test_ranking_honesty_reports_dropped_hits(self, store: SqliteEngravaCore) -> None:
        # Filtering "notes" to P1 drops alpha (P2). The response must say so
        # truthfully rather than silently returning a short list.
        result = await search_memory_impl(store, "notes", priority=Priority.P1)
        filtered = result["filtered"]
        assert filtered["criteria"] == {"priority": "P1"}
        assert filtered["scanned"] == 2
        assert filtered["matched"] == 1
        assert filtered["dropped"] == 1
        # matched must equal the number of returned results — no padding.
        assert filtered["matched"] == len(result["results"])

    async def test_ranking_honesty_when_filter_removes_all(self, store: SqliteEngravaCore) -> None:
        # A filter that matches nothing returns an empty list, but the
        # counts make clear hits *were* ranked and then dropped (so an empty
        # result is not mistaken for "the query ranked nothing").
        result = await search_memory_impl(store, "notes", thought_type=ThoughtType.TASK)
        assert result["results"] == []
        assert result["filtered"]["scanned"] == 2
        assert result["filtered"]["matched"] == 0
        assert result["filtered"]["dropped"] == 2

    async def test_filtered_results_preserve_score_and_order(
        self, store: SqliteEngravaCore
    ) -> None:
        # Scores are carried through from the ranker, never fabricated, and
        # the surviving hits keep their ranked order.
        unfiltered = await search_memory_impl(store, "notes")
        ranked_order = [entry["thought_id"] for entry in unfiltered["results"]]
        scores = {entry["thought_id"]: entry["score"] for entry in unfiltered["results"]}

        filtered = await search_memory_impl(store, "notes", lifecycle_status=LifecycleStatus.ACTIVE)
        kept_order = [entry["thought_id"] for entry in filtered["results"]]
        # Order is the unfiltered order restricted to the survivors.
        assert kept_order == [tid for tid in ranked_order if tid in set(kept_order)]
        for entry in filtered["results"]:
            assert entry["score"] == scores[entry["thought_id"]]


@pytest.fixture
async def varied_store() -> AsyncIterator[SqliteEngravaCore]:
    """Yield a store seeded with thoughts spanning the filter matrix.

    Five thoughts vary by type, lifecycle status, priority, and
    ``updated_cycle`` so the ``list_memory`` filters and pagination can be
    exercised independently.

    Yields:
        A ``SqliteEngravaCore`` seeded with five varied thoughts.

    """
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys=ON")
    backend = SqliteEngravaCore(connection)
    await backend.ensure_schema()

    seeds = [
        ("note-1", ThoughtType.NOTE, LifecycleStatus.CREATED, Priority.P3, 1),
        ("task-1", ThoughtType.TASK, LifecycleStatus.ACTIVE, Priority.P1, 2),
        ("task-2", ThoughtType.TASK, LifecycleStatus.ACTIVE, Priority.P2, 3),
        ("belief-1", ThoughtType.BELIEF, LifecycleStatus.ACTIVE, Priority.P1, 4),
        ("note-2", ThoughtType.NOTE, LifecycleStatus.CREATED, Priority.P4, 5),
    ]
    for thought_id, thought_type, status, priority, cycle in seeds:
        await backend.create_thought(
            CoreThoughtRecord(
                thought_id=thought_id,
                thought_type=thought_type,
                essence=f"Essence for {thought_id}",
                content=f"Content body for {thought_id}.",
                priority=priority,
                lifecycle_status=status,
                created_cycle=cycle,
                updated_cycle=cycle,
                source="test",
            )
        )

    try:
        yield backend
    finally:
        await connection.close()


class TestListMemory:
    """Tests for the deterministic ``list_memory`` browse tool."""

    async def test_unfiltered_lists_all_newest_first(self, varied_store: SqliteEngravaCore) -> None:
        result = await list_memory_impl(varied_store)
        ids = [thought["thought_id"] for thought in result["thoughts"]]
        # list_thoughts orders by descending updated_cycle (newest first).
        assert ids == ["note-2", "belief-1", "task-2", "task-1", "note-1"]
        assert result["count"] == len(ids)
        # Browse results never carry a relevance score.
        assert all("score" not in thought for thought in result["thoughts"])

    async def test_filter_by_thought_type(self, varied_store: SqliteEngravaCore) -> None:
        result = await list_memory_impl(varied_store, thought_type=ThoughtType.TASK)
        ids = {thought["thought_id"] for thought in result["thoughts"]}
        assert ids == {"task-1", "task-2"}

    async def test_filter_by_lifecycle_status(self, varied_store: SqliteEngravaCore) -> None:
        result = await list_memory_impl(varied_store, lifecycle_status=LifecycleStatus.CREATED)
        ids = {thought["thought_id"] for thought in result["thoughts"]}
        assert ids == {"note-1", "note-2"}

    async def test_filter_by_priority(self, varied_store: SqliteEngravaCore) -> None:
        result = await list_memory_impl(varied_store, priority=Priority.P1)
        ids = {thought["thought_id"] for thought in result["thoughts"]}
        assert ids == {"task-1", "belief-1"}

    async def test_combined_filters_apply_as_and(self, varied_store: SqliteEngravaCore) -> None:
        result = await list_memory_impl(
            varied_store,
            thought_type=ThoughtType.TASK,
            priority=Priority.P1,
        )
        ids = [thought["thought_id"] for thought in result["thoughts"]]
        assert ids == ["task-1"]

    async def test_cycle_range_filters(self, varied_store: SqliteEngravaCore) -> None:
        result = await list_memory_impl(varied_store, min_cycle=2, max_cycle=4)
        ids = {thought["thought_id"] for thought in result["thoughts"]}
        assert ids == {"task-1", "task-2", "belief-1"}

    async def test_pagination_limit_and_offset(self, varied_store: SqliteEngravaCore) -> None:
        # Newest-first order: note-2, belief-1, task-2, task-1, note-1.
        first = await list_memory_impl(varied_store, limit=2, offset=0)
        assert [t["thought_id"] for t in first["thoughts"]] == ["note-2", "belief-1"]
        assert first["count"] == 2
        assert first["limit"] == 2
        assert first["offset"] == 0

        second = await list_memory_impl(varied_store, limit=2, offset=2)
        assert [t["thought_id"] for t in second["thoughts"]] == ["task-2", "task-1"]
        assert second["offset"] == 2

        third = await list_memory_impl(varied_store, limit=2, offset=4)
        assert [t["thought_id"] for t in third["thoughts"]] == ["note-1"]
        assert third["count"] == 1

    async def test_offset_past_end_is_empty(self, varied_store: SqliteEngravaCore) -> None:
        result = await list_memory_impl(varied_store, offset=100)
        assert result["thoughts"] == []
        assert result["count"] == 0


class TestQueryMemoryLimit:
    """Tests that ``query_memory`` paginates by ``limit`` (MindQL has no OFFSET)."""

    async def test_limit_argument_caps_rows(self, store: SqliteEngravaCore) -> None:
        # Both seeded thoughts are ACTIVE; an explicit limit caps the rows.
        result = await query_memory_impl(
            store,
            "FIND thoughts WHERE lifecycle_status = 'ACTIVE'",
            limit=1,
        )
        assert len(result["rows"]) == 1

    async def test_limit_argument_overrides_clause(self, store: SqliteEngravaCore) -> None:
        # The limit argument wins over a larger LIMIT clause in the query.
        result = await query_memory_impl(
            store,
            "FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10",
            limit=1,
        )
        assert len(result["rows"]) == 1


def test_default_top_k_is_ten() -> None:
    assert DEFAULT_TOP_K == 10

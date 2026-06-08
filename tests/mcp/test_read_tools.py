"""Unit tests for the MCP read-tool implementations.

Each tool implementation is exercised directly against a seeded
in-memory store (see ``conftest.store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from engrava.mcp.server import (
    DEFAULT_TOP_K,
    StoreNotReadyError,
    StoreProvider,
    UnsupportedQueryError,
    get_thought_impl,
    memory_stats_impl,
    query_memory_impl,
    search_keywords_impl,
    search_memory_impl,
)
from engrava.mindql.parser import MindQLParseError

if TYPE_CHECKING:
    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore


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


def test_default_top_k_is_ten() -> None:
    assert DEFAULT_TOP_K == 10

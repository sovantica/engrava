"""Tests for the deterministic ``invalidate`` valid-time primitives.

Covers :meth:`SqliteEngravaCore.invalidate_thought` and
:meth:`SqliteEngravaCore.invalidate_edge`: each closes a record's valid-time
interval by stamping ``valid_until``, is idempotent, and never cascades or
deletes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.domain.exceptions import ThoughtNotFoundError
from engrava.mindql import executor as executor_module
from engrava.mindql.parser import parse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_T_JAN = "2025-01-01T00:00:00+00:00"
_T_MID = "2025-03-01T00:00:00+00:00"
_T_CLOSE = "2025-04-01T00:00:00+00:00"


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """In-memory store with the core schema applied."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    core = SqliteEngravaCore(conn)
    await core.ensure_schema()
    yield core
    await conn.close()


def _mk_thought(thought_id: str) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content="content",
        priority=Priority.P1,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=1,
        updated_cycle=1,
        source="test",
        valid_from=_T_JAN,
    )


class TestInvalidateThought:
    """``invalidate_thought`` closes a thought's valid-time interval."""

    async def test_sets_valid_until(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_mk_thought("t1"))
        updated = await store.invalidate_thought("t1", _T_CLOSE)
        assert updated.valid_until == _T_CLOSE
        # Persisted, not just returned.
        reloaded = await store.get_thought("t1")
        assert reloaded is not None
        assert reloaded.valid_until == _T_CLOSE

    async def test_is_not_a_delete(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_mk_thought("t1"))
        await store.invalidate_thought("t1", _T_CLOSE)
        # The row is still present and retrievable.
        assert await store.get_thought("t1") is not None

    async def test_idempotent(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_mk_thought("t1"))
        first = await store.invalidate_thought("t1", _T_CLOSE)
        second = await store.invalidate_thought("t1", _T_CLOSE)
        assert first.valid_until == second.valid_until == _T_CLOSE

    async def test_drops_out_of_valid_now_after_close(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_mk_thought("t1"))
        # Before invalidation: visible at MID.
        token = executor_module.mindql_now.set(_T_MID)
        try:
            before = await store.execute_mindql(parse("FIND thoughts WHERE valid_now"))
            assert {row["thought_id"] for row in before.rows} == {"t1"}
            # Close the interval at MID, then query "now" after the close.
            await store.invalidate_thought("t1", _T_MID)
            executor_module.mindql_now.set(_T_CLOSE)
            after = await store.execute_mindql(parse("FIND thoughts WHERE valid_now"))
            assert after.rows == []
        finally:
            executor_module.mindql_now.reset(token)

    async def test_does_not_modify_edges(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_mk_thought("t1"))
        await store.create_thought(_mk_thought("t2"))
        await store.create_edge(
            EdgeRecord(
                edge_id="e1",
                from_thought_id="t1",
                to_thought_id="t2",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=1,
                valid_from=_T_JAN,
            )
        )
        await store.invalidate_thought("t1", _T_CLOSE)
        # The edge's valid-time interval is untouched — no cascade.
        edges = await store.get_edges("t1")
        assert len(edges) == 1
        assert edges[0].valid_until is None

    async def test_missing_thought_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ThoughtNotFoundError):
            await store.invalidate_thought("ghost", _T_CLOSE)

    async def test_non_iso_valid_until_raises(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_mk_thought("t1"))
        with pytest.raises(ValueError, match="ISO-8601"):
            await store.invalidate_thought("t1", "not-a-date")


class TestInvalidateEdge:
    """``invalidate_edge`` closes an edge's valid-time interval."""

    async def _seed_edge(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_mk_thought("t1"))
        await store.create_thought(_mk_thought("t2"))
        await store.create_edge(
            EdgeRecord(
                edge_id="e1",
                from_thought_id="t1",
                to_thought_id="t2",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=1,
                valid_from=_T_JAN,
            )
        )

    async def test_sets_valid_until(self, store: SqliteEngravaCore) -> None:
        await self._seed_edge(store)
        updated = await store.invalidate_edge("e1", _T_CLOSE)
        assert updated.valid_until == _T_CLOSE
        edges = await store.get_edges("t1")
        assert edges[0].valid_until == _T_CLOSE

    async def test_is_not_a_delete(self, store: SqliteEngravaCore) -> None:
        await self._seed_edge(store)
        await store.invalidate_edge("e1", _T_CLOSE)
        assert len(await store.get_edges("t1")) == 1

    async def test_idempotent(self, store: SqliteEngravaCore) -> None:
        await self._seed_edge(store)
        first = await store.invalidate_edge("e1", _T_CLOSE)
        second = await store.invalidate_edge("e1", _T_CLOSE)
        assert first.valid_until == second.valid_until == _T_CLOSE

    async def test_drops_out_of_valid_now_after_close(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await self._seed_edge(store)
        token = executor_module.mindql_now.set(_T_MID)
        try:
            before = await store.execute_mindql(parse("FIND edges WHERE valid_now"))
            assert {row["edge_id"] for row in before.rows} == {"e1"}
            await store.invalidate_edge("e1", _T_MID)
            executor_module.mindql_now.set(_T_CLOSE)
            after = await store.execute_mindql(parse("FIND edges WHERE valid_now"))
            assert after.rows == []
        finally:
            executor_module.mindql_now.reset(token)

    async def test_missing_edge_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ValueError, match="Edge not found"):
            await store.invalidate_edge("ghost", _T_CLOSE)

    async def test_non_iso_valid_until_raises(self, store: SqliteEngravaCore) -> None:
        await self._seed_edge(store)
        with pytest.raises(ValueError, match="ISO-8601"):
            await store.invalidate_edge("e1", "not-a-date")

"""Integration tests for ReadOnlyEngrava.

Verifies that:
1. All read operations are properly delegated to the inner Engrava store.
2. Every write operation raises ReadOnlyViolationError unconditionally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ReadOnlyEngrava,
    ReadOnlyViolationError,
    SqliteEngravaCore,
    ThoughtType,
    ThoughtVisibility,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite database with core schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def inner_store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """SqliteEngravaCore backed by the test database."""
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


@pytest.fixture
def ro_store(inner_store: SqliteEngravaCore) -> ReadOnlyEngrava:
    """ReadOnlyEngrava wrapping the inner store."""
    return ReadOnlyEngrava(inner_store)


def _make_thought(
    thought_id: str = "t-001",
    *,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    priority: Priority = Priority.P2,
    visibility: ThoughtVisibility = ThoughtVisibility.SELECTIVE,
) -> CoreThoughtRecord:
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence="Test thought",
        content="Test content",
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=visibility,
    )


# ---------------------------------------------------------------------------
# Read operations — must delegate to inner store
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaRead:
    """Read operations are properly delegated to the inner store."""

    async def test_get_thought_existing(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        result = await ro_store.get_thought("t-001")
        assert result is not None
        assert result.thought_id == "t-001"
        assert result.essence == "Test thought"

    async def test_get_thought_missing_returns_none(self, ro_store: ReadOnlyEngrava) -> None:
        result = await ro_store.get_thought("nonexistent")
        assert result is None

    async def test_list_thoughts_returns_all(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.create_thought(_make_thought("t-002"))
        results = await ro_store.list_thoughts(limit=10)
        assert len(results) == 2

    async def test_list_thoughts_empty_store(self, ro_store: ReadOnlyEngrava) -> None:
        results = await ro_store.list_thoughts()
        assert results == []

    async def test_list_thoughts_filter_priority(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-p2", priority=Priority.P2))
        await inner_store.create_thought(_make_thought("t-p1", priority=Priority.P1))
        p1_results = await ro_store.list_thoughts(priority="P1")
        assert len(p1_results) == 1
        assert p1_results[0].thought_id == "t-p1"

    async def test_list_thoughts_filter_lifecycle_status(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(
            _make_thought("t-active", lifecycle_status=LifecycleStatus.ACTIVE)
        )
        await inner_store.create_thought(
            _make_thought("t-created", lifecycle_status=LifecycleStatus.CREATED)
        )
        active = await ro_store.list_thoughts(lifecycle_status="ACTIVE")
        assert len(active) == 1
        assert active[0].thought_id == "t-active"

    async def test_list_thoughts_filter_visibility(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(
            _make_thought("t-public", visibility=ThoughtVisibility.PUBLIC)
        )
        await inner_store.create_thought(
            _make_thought("t-private", visibility=ThoughtVisibility.PRIVATE)
        )
        public = await ro_store.list_thoughts(visibility="public")
        assert len(public) == 1
        assert public[0].thought_id == "t-public"

    async def test_list_thoughts_limit_and_offset(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        for i in range(5):
            await inner_store.create_thought(_make_thought(f"t-{i:03d}"))
        page_1 = await ro_store.list_thoughts(limit=2, offset=0)
        page_2 = await ro_store.list_thoughts(limit=2, offset=2)
        assert len(page_1) == 2
        assert len(page_2) == 2
        assert {t.thought_id for t in page_1}.isdisjoint({t.thought_id for t in page_2})

    async def test_search_similar_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [1.0, 0.0, 0.0])
        results = await ro_store.search_similar([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0] == "t-001"
        assert results[0][1] > 0.99

    async def test_search_similar_empty_store(self, ro_store: ReadOnlyEngrava) -> None:
        results = await ro_store.search_similar([1.0, 0.0, 0.0])
        assert results == []

    async def test_search_similar_with_threshold_filters(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [1.0, 0.0, 0.0])
        # Orthogonal vector → similarity ~0.0, below threshold
        results = await ro_store.search_similar([0.0, 1.0, 0.0], threshold=0.99)
        assert results == []

    async def test_search_similar_top_k(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        for i in range(5):
            await inner_store.create_thought(_make_thought(f"t-{i:03d}"))
            await inner_store.store_embedding(f"t-{i:03d}", [1.0, float(i) * 0.1, 0.0])
        results = await ro_store.search_similar([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    async def test_get_edges_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-a"))
        await inner_store.create_thought(_make_thought("t-b"))
        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.9,
            created_cycle=0,
        )
        await inner_store.create_edge(edge)
        edges = await ro_store.get_edges("t-a")
        assert len(edges) == 1
        assert edges[0].edge_id == "e-001"

    async def test_get_edges_no_edges(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        edges = await ro_store.get_edges("t-001")
        assert edges == []

    async def test_get_edges_direction_in(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-a"))
        await inner_store.create_thought(_make_thought("t-b"))
        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.DEPENDS_ON,
            weight=1.0,
            created_cycle=0,
        )
        await inner_store.create_edge(edge)
        # t-b is the target, so direction="IN" from t-b's perspective
        edges = await ro_store.get_edges("t-b", direction="IN")
        assert len(edges) == 1
        assert edges[0].from_thought_id == "t-a"

    async def test_get_embedding_delegates(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.store_embedding("t-001", [0.1, 0.2, 0.3])
        emb = await ro_store.get_embedding("t-001")
        assert emb is not None
        assert emb.dimension == 3
        assert emb.owner_id == "t-001"

    async def test_get_embedding_missing_returns_none(self, ro_store: ReadOnlyEngrava) -> None:
        result = await ro_store.get_embedding("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Write operations — must raise ReadOnlyViolationError
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaWriteBlocked:
    """All write operations unconditionally raise ReadOnlyViolationError."""

    async def test_create_thought_raises(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.create_thought(_make_thought())

    async def test_create_thought_error_names_method(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError, match="create_thought"):
            await ro_store.create_thought(_make_thought())

    async def test_update_thought_raises(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.update_thought("t-001", essence="new essence")

    async def test_update_thought_error_names_method(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError, match="update_thought"):
            await ro_store.update_thought("t-001", essence="new")

    async def test_delete_thought_raises(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.delete_thought("t-001")

    async def test_delete_thought_error_names_method(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError, match="delete_thought"):
            await ro_store.delete_thought("t-001")

    async def test_create_edge_raises(self, ro_store: ReadOnlyEngrava) -> None:
        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.ASSOCIATED,
            weight=1.0,
            created_cycle=0,
        )
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.create_edge(edge)

    async def test_create_edge_error_names_method(self, ro_store: ReadOnlyEngrava) -> None:
        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-a",
            to_thought_id="t-b",
            edge_type=EdgeType.ASSOCIATED,
            weight=1.0,
            created_cycle=0,
        )
        with pytest.raises(ReadOnlyViolationError, match="create_edge"):
            await ro_store.create_edge(edge)

    async def test_delete_edge_raises(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.delete_edge("e-001")

    async def test_delete_edge_error_names_method(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError, match="delete_edge"):
            await ro_store.delete_edge("e-001")

    async def test_update_edge_raises(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.update_edge("e-001", weight=0.5)

    async def test_update_edge_error_names_method(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError, match="update_edge"):
            await ro_store.update_edge("e-001", weight=0.5)

    async def test_store_embedding_raises(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.store_embedding("t-001", [0.1, 0.2, 0.3])

    async def test_store_embedding_error_names_method(self, ro_store: ReadOnlyEngrava) -> None:
        with pytest.raises(ReadOnlyViolationError, match="store_embedding"):
            await ro_store.store_embedding("t-001", [0.1, 0.2])

    async def test_store_embedding_model_kwarg_still_raises(
        self, ro_store: ReadOnlyEngrava
    ) -> None:
        """Keyword arguments are accepted but the error is still raised."""
        with pytest.raises(ReadOnlyViolationError):
            await ro_store.store_embedding("t-001", [0.1, 0.2], model_name="custom-model")


# ---------------------------------------------------------------------------
# Wrapping behaviour — inner store isolation
# ---------------------------------------------------------------------------


class TestReadOnlyEngravaIsolation:
    """ReadOnlyEngrava is a thin wrapper; mutations via inner store are visible."""

    async def test_mutation_via_inner_visible_through_ro(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        await inner_store.update_thought("t-001", essence="Updated essence")
        result = await ro_store.get_thought("t-001")
        assert result is not None
        assert result.essence == "Updated essence"

    async def test_deletion_via_inner_visible_through_ro(
        self,
        inner_store: SqliteEngravaCore,
        ro_store: ReadOnlyEngrava,
    ) -> None:
        await inner_store.create_thought(_make_thought("t-001"))
        assert await ro_store.get_thought("t-001") is not None
        await inner_store.delete_thought("t-001")
        assert await ro_store.get_thought("t-001") is None

    async def test_multiple_ro_wrappers_share_same_inner(
        self,
        inner_store: SqliteEngravaCore,
    ) -> None:
        ro1 = ReadOnlyEngrava(inner_store)
        ro2 = ReadOnlyEngrava(inner_store)
        await inner_store.create_thought(_make_thought("t-001"))
        assert await ro1.get_thought("t-001") is not None
        assert await ro2.get_thought("t-001") is not None

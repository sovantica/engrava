"""Tests for hook wiring in SqliteEngravaCore.

Verifies that hooks are dispatched at the correct lifecycle points:
on_store after create, on_retrieve after get/list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    DefaultEngravaHooks,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.domain.protocols.hooks import ScoringContext

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Tracking hooks
# ---------------------------------------------------------------------------


class TrackingHooks(DefaultEngravaHooks):
    """Hooks that record calls for test assertions."""

    def __init__(self) -> None:
        self.on_store_calls: list[str] = []
        self.on_retrieve_calls: list[str] = []

    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Record on_store call.

        Args:
            thought: The thought record.

        Returns:
            The same thought.

        """
        self.on_store_calls.append(thought.thought_id)
        return thought

    async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Record on_retrieve call.

        Args:
            thought: The thought record.

        Returns:
            The same thought.

        """
        self.on_retrieve_calls.append(thought.thought_id)
        return thought


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_thought(
    thought_id: str = "t-001",
    priority: Priority = Priority.P2,
) -> ThoughtRecord:
    """Create a test thought."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence="Test thought",
        content="Test thought content",
        priority=priority,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
    )


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """Create an in-memory SQLite database."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHookWiring:
    """Verify hooks are dispatched at correct lifecycle points."""

    async def test_on_store_called_after_create(self, db: aiosqlite.Connection) -> None:
        hooks = TrackingHooks()
        store = SqliteEngravaCore(db, hooks=hooks)
        await store.ensure_schema()

        await store.create_thought(_make_thought())
        assert hooks.on_store_calls == ["t-001"]

    async def test_on_retrieve_called_after_get(self, db: aiosqlite.Connection) -> None:
        hooks = TrackingHooks()
        store = SqliteEngravaCore(db, hooks=hooks)
        await store.ensure_schema()

        await store.create_thought(_make_thought())
        hooks.on_retrieve_calls.clear()

        await store.get_thought("t-001")
        assert hooks.on_retrieve_calls == ["t-001"]

    async def test_on_retrieve_called_for_each_in_list(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        hooks = TrackingHooks()
        store = SqliteEngravaCore(db, hooks=hooks)
        await store.ensure_schema()

        for i in range(3):
            await store.create_thought(_make_thought(thought_id=f"t-{i:03d}"))
        hooks.on_retrieve_calls.clear()

        await store.list_thoughts(limit=10)
        assert len(hooks.on_retrieve_calls) == 3

    async def test_get_nonexistent_does_not_call_hook(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        hooks = TrackingHooks()
        store = SqliteEngravaCore(db, hooks=hooks)
        await store.ensure_schema()

        result = await store.get_thought("nonexistent")
        assert result is None
        assert hooks.on_retrieve_calls == []

    async def test_default_hooks_used_when_none(self, db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(db)
        await store.ensure_schema()

        # Should work with default no-op hooks
        created = await store.create_thought(_make_thought())
        assert created.thought_id == "t-001"

        fetched = await store.get_thought("t-001")
        assert fetched is not None

    async def test_hooks_attribute_is_accessible(self, db: aiosqlite.Connection) -> None:
        hooks = TrackingHooks()
        store = SqliteEngravaCore(db, hooks=hooks)
        assert store._hooks is hooks


class TestDefaultHooksScoring:
    """Verify DefaultEngravaHooks scoring behavior."""

    async def test_priority_based_score(self) -> None:
        hooks = DefaultEngravaHooks()
        ctx = ScoringContext(current_cycle=10)
        t = _make_thought(priority=Priority.P1)
        score = await hooks.score_function(t, ctx)
        assert score == 4.0

    async def test_decay_returns_one(self) -> None:
        hooks = DefaultEngravaHooks()
        t = _make_thought()
        decay = await hooks.decay_function(t, elapsed_cycles=100)
        assert decay == 1.0

    async def test_empty_extension_registry(self) -> None:
        hooks = DefaultEngravaHooks()
        assert hooks.mindql_extension_registry() == {}

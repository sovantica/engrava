"""A row write and its journal entry recover together, not separately.

``update_thought``, ``restore_thought``, ``update_edge`` and ``update_action``
each write a row, then append the journal entry describing it, and only then
decide whether to commit. The journal append runs inside the write-readback
savepoint (``SqliteEngravaCore._write_readback_savepoint``):
``JournalWriter.append`` awaits a chain-tail read before its own ``INSERT``
(see ``journal_writer.py``), and a failure or a cancellation landing in that
await unwinds the row write with it.

This is a different window from the one ``test_update_readback_atomicity.py``
pins: that file's failures happen *inside* the savepoint (a read-back that
raises before the savepoint releases). This file's failures happen in the
journal append that follows the row write and its read-back, before the
append's own ``INSERT`` lands.

The ``test_failed_append_leaves_no_durable_row_change`` and
``test_cancelled_append_leaves_no_durable_row_change`` tests commit an
unrelated write after the failed append, close the store, and read the
original row through a separate connection. The
``test_outermost_suspend_auto_commit_rollback_is_unchanged`` tests run the
update inside a caller-owned outermost ``suspend_auto_commit()`` window and
check that no transaction is left open and the row is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionRecord,
    ActionStatus,
    ActionType,
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _open(db_path: Path, *, journal_enabled: bool = True) -> SqliteEngravaCore:
    """Open a fresh, independent connection on ``db_path`` and build a store on it."""
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, journal_enabled=journal_enabled)
    await store.ensure_schema()
    return store


def _thought(thought_id: str = "t-1", *, essence: str = "essence") -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=f"content of {thought_id}",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.75,
    )


def _archived_thought(thought_id: str = "t-1") -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content=f"content of {thought_id}",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ARCHIVED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.75,
        archived_at_cycle=0,
    )


def _edge(edge_id: str = "e-1", *, weight: float = 0.5) -> EdgeRecord:
    return EdgeRecord(
        edge_id=edge_id,
        from_thought_id="t-1",
        to_thought_id="t-2",
        edge_type=EdgeType.ASSOCIATED,
        weight=weight,
        created_cycle=0,
        source=KnowledgeSource.EXPERIENCE,
        decay_multiplier=1.0,
    )


def _action(action_id: str = "a-1") -> ActionRecord:
    return ActionRecord(
        action_id=action_id,
        source_thought_id="t-1",
        action_type=ActionType.CLI_OUTPUT,
        intent="do the thing",
        status=ActionStatus.PLANNED,
        verification_status=VerificationStatus.PENDING,
    )


async def _raw_row(db_path: Path, table: str, key: str, value: str) -> aiosqlite.Row | None:
    """Read a row straight from storage, on a brand-new independent connection."""
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        cursor = await conn.execute(f"SELECT * FROM {table} WHERE {key} = ?", (value,))  # noqa: S608 -- test literals
        return await cursor.fetchone()
    finally:
        await conn.close()


def _fail_journal_append_once(store: SqliteEngravaCore, exc: BaseException) -> None:
    """Make the journal's own chain-tail read raise/cancel on its next call.

    Targets ``JournalWriter._get_latest_entry_state`` -- the ``await`` that
    ``JournalWriter.append`` performs *before* its own ``INSERT``. This is
    exactly the window a transient failure or a cancellation lands in: by
    the time it fires, the guarded call's own row write has already landed
    on the connection and its confirming read-back has already run.
    """
    journal = store._journal
    assert journal is not None
    original = journal._get_latest_entry_state
    fired = False

    async def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal fired
        if not fired:
            fired = True
            raise exc
        return await original(*args, **kwargs)

    journal._get_latest_entry_state = wrapper  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 1. update_thought
# ---------------------------------------------------------------------------


class TestUpdateThoughtJournalAppendFailure:
    async def test_failed_append_leaves_no_durable_row_change(self, tmp_path: Path) -> None:
        db_path = tmp_path / "update-thought-fail.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1", essence="original essence"))

            _fail_journal_append_once(store, RuntimeError("forced journal append failure"))
            with pytest.raises(RuntimeError, match="forced journal append failure"):
                await store.update_thought("t-1", essence="should never land")

            # An unrelated write on the same connection, committed normally --
            # it must not carry the failed update's row write along with it.
            await store.create_thought(_thought("t-2", essence="unrelated"))
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "thought", "thought_id", "t-1")
        assert row is not None
        assert row["essence"] == "original essence"

        reopened = await _open(db_path)
        try:
            entries = await reopened._journal.get_entries(  # type: ignore[union-attr]
                target_id="t-1", mutation_type="UPDATE_THOUGHT"
            )
            assert entries == []
            result = await reopened.verify_journal()
            assert result.valid is True
        finally:
            await reopened._db.close()

    async def test_cancelled_append_leaves_no_durable_row_change(self, tmp_path: Path) -> None:
        db_path = tmp_path / "update-thought-cancel.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1", essence="original essence"))

            _fail_journal_append_once(store, asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                await store.update_thought("t-1", essence="should never land")

            await store.create_thought(_thought("t-2", essence="unrelated"))
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "thought", "thought_id", "t-1")
        assert row is not None
        assert row["essence"] == "original essence"

        reopened = await _open(db_path)
        try:
            entries = await reopened._journal.get_entries(  # type: ignore[union-attr]
                target_id="t-1", mutation_type="UPDATE_THOUGHT"
            )
            assert entries == []
        finally:
            await reopened._db.close()

    async def test_outermost_suspend_auto_commit_rollback_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Inside an outermost window a failed append leaves no open transaction or row change."""
        db_path = tmp_path / "update-thought-outermost.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1", essence="original essence"))

            _fail_journal_append_once(store, asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                async with store.suspend_auto_commit():
                    await store.update_thought("t-1", essence="should never land")

            assert store._db.in_transaction is False
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "thought", "thought_id", "t-1")
        assert row is not None
        assert row["essence"] == "original essence"


# ---------------------------------------------------------------------------
# 2. restore_thought
# ---------------------------------------------------------------------------


class TestRestoreThoughtJournalAppendFailure:
    async def test_failed_append_leaves_no_durable_row_change(self, tmp_path: Path) -> None:
        db_path = tmp_path / "restore-thought-fail.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_archived_thought("t-1"))

            _fail_journal_append_once(store, RuntimeError("forced journal append failure"))
            with pytest.raises(RuntimeError, match="forced journal append failure"):
                await store.restore_thought("t-1")

            await store.create_thought(_thought("t-2", essence="unrelated"))
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "thought", "thought_id", "t-1")
        assert row is not None
        assert row["lifecycle_status"] == "ARCHIVED"

        reopened = await _open(db_path)
        try:
            entries = await reopened._journal.get_entries(  # type: ignore[union-attr]
                target_id="t-1", mutation_type="UPDATE_THOUGHT"
            )
            assert entries == []
            result = await reopened.verify_journal()
            assert result.valid is True
        finally:
            await reopened._db.close()

    async def test_outermost_suspend_auto_commit_rollback_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "restore-thought-outermost.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_archived_thought("t-1"))

            _fail_journal_append_once(store, asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                async with store.suspend_auto_commit():
                    await store.restore_thought("t-1")

            assert store._db.in_transaction is False
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "thought", "thought_id", "t-1")
        assert row is not None
        assert row["lifecycle_status"] == "ARCHIVED"


# ---------------------------------------------------------------------------
# 3. update_edge
# ---------------------------------------------------------------------------


class TestUpdateEdgeJournalAppendFailure:
    async def test_failed_append_leaves_no_durable_row_change(self, tmp_path: Path) -> None:
        db_path = tmp_path / "update-edge-fail.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1"))
            await store.create_thought(_thought("t-2"))
            await store.create_edge(_edge("e-1", weight=0.5))

            _fail_journal_append_once(store, RuntimeError("forced journal append failure"))
            with pytest.raises(RuntimeError, match="forced journal append failure"):
                await store.update_edge("e-1", weight=0.9)

            await store.create_thought(_thought("t-3", essence="unrelated"))
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "edge", "edge_id", "e-1")
        assert row is not None
        assert row["weight"] == 0.5

        reopened = await _open(db_path)
        try:
            entries = await reopened._journal.get_entries(  # type: ignore[union-attr]
                target_id="e-1", mutation_type="UPDATE_EDGE"
            )
            assert entries == []
            result = await reopened.verify_journal()
            assert result.valid is True
        finally:
            await reopened._db.close()

    async def test_outermost_suspend_auto_commit_rollback_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "update-edge-outermost.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1"))
            await store.create_thought(_thought("t-2"))
            await store.create_edge(_edge("e-1", weight=0.5))

            _fail_journal_append_once(store, asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                async with store.suspend_auto_commit():
                    await store.update_edge("e-1", weight=0.9)

            assert store._db.in_transaction is False
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "edge", "edge_id", "e-1")
        assert row is not None
        assert row["weight"] == 0.5


# ---------------------------------------------------------------------------
# 4. update_action
# ---------------------------------------------------------------------------


class TestUpdateActionJournalAppendFailure:
    async def test_failed_append_leaves_no_durable_row_change(self, tmp_path: Path) -> None:
        db_path = tmp_path / "update-action-fail.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1"))
            await store.create_action(_action("a-1"))

            _fail_journal_append_once(store, RuntimeError("forced journal append failure"))
            with pytest.raises(RuntimeError, match="forced journal append failure"):
                await store.update_action("a-1", status=ActionStatus.EXECUTING)

            await store.create_thought(_thought("t-2", essence="unrelated"))
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "action", "action_id", "a-1")
        assert row is not None
        assert row["status"] == "PLANNED"

        reopened = await _open(db_path)
        try:
            entries = await reopened._journal.get_entries(  # type: ignore[union-attr]
                target_id="a-1", mutation_type="UPDATE_ACTION"
            )
            assert entries == []
            result = await reopened.verify_journal()
            assert result.valid is True
        finally:
            await reopened._db.close()

    async def test_outermost_suspend_auto_commit_rollback_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "update-action-outermost.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1"))
            await store.create_action(_action("a-1"))

            _fail_journal_append_once(store, asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                async with store.suspend_auto_commit():
                    await store.update_action("a-1", status=ActionStatus.EXECUTING)

            assert store._db.in_transaction is False
        finally:
            await store._db.close()

        row = await _raw_row(db_path, "action", "action_id", "a-1")
        assert row is not None
        assert row["status"] == "PLANNED"


# ---------------------------------------------------------------------------
# 5. The chain still verifies after a normal write following a failed one.
# ---------------------------------------------------------------------------


class TestChainStillVerifiesAfterAFailedAppend:
    async def test_chain_verifies_after_a_failed_update_thought_append(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "chain-after-failure.db"
        store = await _open(db_path)
        try:
            await store.create_thought(_thought("t-1", essence="original essence"))

            _fail_journal_append_once(store, RuntimeError("forced journal append failure"))
            with pytest.raises(RuntimeError, match="forced journal append failure"):
                await store.update_thought("t-1", essence="should never land")

            await store.update_thought("t-1", essence="a real, later change")
            result = await store.verify_journal()
            assert result.valid is True
        finally:
            await store._db.close()

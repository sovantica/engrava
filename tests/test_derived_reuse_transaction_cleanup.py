"""A reused derived child or provenance edge closes its own open transaction.

``_insert_derived_row`` and ``_insert_derived_edge`` each run their insert
inside their own ``_write_readback_savepoint`` unit, opening ``BEGIN
IMMEDIATE`` when no transaction is already open. On a ``UNIQUE``-conflict
reuse the insert aborts and the unit still exits cleanly (a plain
``RELEASE``, never an unwind), but the transaction the call opened for
itself used to stay open after that clean exit -- holding SQLite's write
lock until some unrelated later commit happened to flush it. These tests pin
the fixed behaviour: on a reuse, a transaction a call opened for itself is
rolled back (never committed -- nothing was written); a transaction a caller
already held when the call started is left exactly as it was.

Every database here is file-backed WAL (never ``:memory:``): a few tests
need a genuinely independent second connection to probe whether SQLite's
write lock is actually held, and every test in this module carries the
``integration`` marker for that reason.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    DefaultEngravaHooks,
    DerivedRecord,
    DeriveGates,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from engrava.domain.models.thought import ThoughtRecord
    from engrava.domain.protocols.derived_records import DeriveContext

pytestmark = pytest.mark.integration


@contextlib.asynccontextmanager
async def _connection(path: str) -> AsyncIterator[aiosqlite.Connection]:
    """A WAL-mode connection that is always closed, including on a failed assertion."""
    conn = await aiosqlite.connect(path)
    try:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        yield conn
    finally:
        await conn.close()


@contextlib.asynccontextmanager
async def _reopened(path: str) -> AsyncIterator[SqliteEngravaCore]:
    """A fresh, separate connection + store over the same on-disk database.

    Used both to prove durability (what a genuinely independent reader sees
    after a commit) and non-visibility (what it does *not* see while a
    writer's transaction is still open) -- the writing connection's own view
    would include its uncommitted work either way.
    """
    conn = await aiosqlite.connect(path)
    try:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        yield store
    finally:
        await conn.close()


def _source(thought_id: str, content: str) -> CoreThoughtRecord:
    """Build a realistic source thought."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.NOTE,
        essence="source essence",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
    )


class _SharedChildProducer(DefaultEngravaHooks):
    """Always derive one fixed-content record, so every source's child collides."""

    def __init__(self, content: str, *, attach_edge: bool) -> None:
        self._content = content
        self._attach_edge = attach_edge
        self.derive_calls = 0

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        self.derive_calls += 1
        return [
            DerivedRecord(
                content=self._content,
                thought_type=ThoughtType.OBSERVATION,
                priority=Priority.P3,
                attach_provenance_edge=self._attach_edge,
            )
        ]


async def _open_store(
    conn: aiosqlite.Connection,
    producer: DefaultEngravaHooks,
    *,
    gates_enabled: bool = False,
) -> SqliteEngravaCore:
    store = SqliteEngravaCore(
        conn,
        hooks=producer,
        journal_enabled=True,
        derive_gates=DeriveGates(enabled=gates_enabled),
    )
    await store.ensure_schema()
    return store


def _second_write_succeeds(path: str) -> bool:
    """Try a write from an independent, short-timeout sqlite3 connection.

    Succeeds only when nothing else holds SQLite's write lock on the file --
    proving the first connection's own transaction (if any) has genuinely
    ended, not merely that this process believes it has.
    """
    conn = sqlite3.connect(path, timeout=1.0)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS probe_other(x)")
        conn.execute("INSERT INTO probe_other VALUES (1)")
        conn.commit()
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()
    return True


# ---------------------------------------------------------------------------
# No caller transaction: the reuse exit must close what it opened itself.
# ---------------------------------------------------------------------------


async def test_rerun_with_edge_reused_closes_self_opened_transaction(tmp_path: Path) -> None:
    """``derive_existing`` re-run, child and edge both reused: no transaction left open."""
    db_path = tmp_path / "rerun-edge.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=True)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer)
        source = await store.create_thought(_source("src-1", "source body"))

        first = await store.derive_existing(source.thought_id)
        assert first.created == 1
        assert first.reused == 0

        second = await store.derive_existing(source.thought_id)
        assert (second.created, second.reused, second.skipped) == (0, 1, 0)
        assert store._db.in_transaction is False
        assert _second_write_succeeds(str(db_path))


async def test_rerun_without_edge_reused_closes_self_opened_transaction(tmp_path: Path) -> None:
    """The same re-run with ``attach_provenance_edge=False``: only the child reuses."""
    db_path = tmp_path / "rerun-no-edge.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=False)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer)
        source = await store.create_thought(_source("src-1", "source body"))

        first = await store.derive_existing(source.thought_id)
        assert first.created == 1
        assert first.reused == 0

        second = await store.derive_existing(source.thought_id)
        assert (second.created, second.reused, second.skipped) == (0, 1, 0)
        assert store._db.in_transaction is False
        assert _second_write_succeeds(str(db_path))


async def test_create_thought_second_source_reuses_child_closes_self_opened_transaction(
    tmp_path: Path,
) -> None:
    """A second source whose producer yields an already-stored child (no edge), gates enabled."""
    db_path = tmp_path / "on-store-no-edge.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=False)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer, gates_enabled=True)
        await store.create_thought(_source("src-1", "first source"))
        await store.create_thought(_source("src-2", "second source"))

        assert store._db.in_transaction is False
        assert _second_write_succeeds(str(db_path))


# ---------------------------------------------------------------------------
# Inside an empty suspend_auto_commit window: the self-opened transaction is
# rolled back even though commits are suppressed; the window itself is left
# untouched for its own later write and exit.
# ---------------------------------------------------------------------------


async def test_empty_window_rerun_with_edge_leaves_no_transaction_open(tmp_path: Path) -> None:
    """Reuse of both halves inside an empty window leaves no transaction open.

    Also pins that the window's own later write stays invisible to a fresh
    reader until the window exits, becomes durable after, and that a
    ``create_thought`` issued inside the window still does not auto-dispatch
    derivation.
    """
    db_path = tmp_path / "window-edge.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=True)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer, gates_enabled=True)
        source = await store.create_thought(_source("src-1", "source body"))
        assert producer.derive_calls == 1
        assert store._db.in_transaction is False  # clean setup, nothing leaked yet

        async with store.suspend_auto_commit():
            # Clean entry: the window itself has written nothing yet.
            assert store._db.in_transaction is False

            second = await store.derive_existing(source.thought_id)
            assert (second.created, second.reused, second.skipped) == (0, 1, 0)
            # RED on the unfixed code: the reuse's own self-opened empty
            # transaction leaks, leaving this True.
            assert store._db.in_transaction is False

            await store.create_thought(_source("later-in-window", "later body"))
            # The window's own write still has not auto-dispatched derivation
            # for the shared child (the source is not yet durable).
            assert producer.derive_calls == 2
            # The window still owns this write: it is pending, not committed.
            assert store._db.in_transaction is True

            # Checked only now, after the write -- this actually proves the
            # write is still pending (not merely "absent because we looked
            # too early"): a fresh reader still does not see it while the
            # window is still open.
            async with _reopened(str(db_path)) as mid_window_reader:
                assert await mid_window_reader.get_thought("later-in-window") is None

        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("later-in-window") is not None


async def test_empty_window_rerun_without_edge_leaves_no_transaction_open(
    tmp_path: Path,
) -> None:
    """Reuse of the child only (no edge) inside an empty window leaves nothing open."""
    db_path = tmp_path / "window-no-edge.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=False)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer, gates_enabled=True)
        source = await store.create_thought(_source("src-1", "source body"))
        assert store._db.in_transaction is False

        async with store.suspend_auto_commit():
            assert store._db.in_transaction is False  # clean entry

            result = await store.derive_existing(source.thought_id)
            assert (result.created, result.reused, result.skipped) == (0, 1, 0)
            # RED on the unfixed code.
            assert store._db.in_transaction is False

        assert store._db.in_transaction is False


# ---------------------------------------------------------------------------
# Rollback, not commit: a spy on the connection shows exactly what the
# self-opened reuse exit does to end its transaction.
# ---------------------------------------------------------------------------


async def test_rerun_reuse_rolls_back_not_commits_child_and_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both the child's and the edge's self-opened reuse exits roll back, never commit."""
    db_path = tmp_path / "rollback-spy-edge.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=True)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer)
        source = await store.create_thought(_source("src-1", "source body"))
        first = await store.derive_existing(source.thought_id)
        assert first.created == 1

        rollback_calls = 0
        commit_calls = 0
        real_rollback = store._db.rollback
        real_commit = store._db.commit

        async def _spy_rollback() -> None:
            nonlocal rollback_calls
            rollback_calls += 1
            await real_rollback()

        async def _spy_commit() -> None:
            nonlocal commit_calls
            commit_calls += 1
            await real_commit()

        monkeypatch.setattr(store._db, "rollback", _spy_rollback)
        monkeypatch.setattr(store._db, "commit", _spy_commit)

        second = await store.derive_existing(source.thought_id)
        assert (second.created, second.reused, second.skipped) == (0, 1, 0)
        # One rollback for the child's self-opened transaction, one for the
        # edge's (fixed, separately, after the child's own has already closed).
        assert rollback_calls == 2
        assert commit_calls == 0
        assert store._db.in_transaction is False


async def test_rerun_reuse_rolls_back_not_commits_child_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no edge in play, exactly one rollback -- the child's own -- happens."""
    db_path = tmp_path / "rollback-spy-child.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=False)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer)
        source = await store.create_thought(_source("src-1", "source body"))
        first = await store.derive_existing(source.thought_id)
        assert first.created == 1

        rollback_calls = 0
        commit_calls = 0
        real_rollback = store._db.rollback
        real_commit = store._db.commit

        async def _spy_rollback() -> None:
            nonlocal rollback_calls
            rollback_calls += 1
            await real_rollback()

        async def _spy_commit() -> None:
            nonlocal commit_calls
            commit_calls += 1
            await real_commit()

        monkeypatch.setattr(store._db, "rollback", _spy_rollback)
        monkeypatch.setattr(store._db, "commit", _spy_commit)

        second = await store.derive_existing(source.thought_id)
        assert (second.created, second.reused, second.skipped) == (0, 1, 0)
        assert rollback_calls == 1
        assert commit_calls == 0
        assert store._db.in_transaction is False


# ---------------------------------------------------------------------------
# A transaction the caller already held is never touched by the reuse exit --
# these pin behaviour that must hold both before and after the fix.
# ---------------------------------------------------------------------------


async def test_window_with_pending_write_reuse_leaves_transaction_open_then_commits_at_exit(
    tmp_path: Path,
) -> None:
    """A reuse inside a window that already holds the caller's own pending write.

    The caller's transaction (and its pending write) is left exactly as it
    was; the window's own exit still commits everything, caller write
    included.
    """
    db_path = tmp_path / "window-pending.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=True)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer)
        source = await store.create_thought(_source("src-1", "source body"))
        first = await store.derive_existing(source.thought_id)
        assert first.created == 1

        async with store.suspend_auto_commit():
            await store.create_thought(_source("caller-write", "caller body"))
            assert store._db.in_transaction is True

            second = await store.derive_existing(source.thought_id)
            assert (second.created, second.reused, second.skipped) == (0, 1, 0)
            # The caller's own transaction is untouched by the reuse exit.
            assert store._db.in_transaction is True

            async with _reopened(str(db_path)) as mid_window_reader:
                assert await mid_window_reader.get_thought("caller-write") is None

        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("caller-write") is not None


async def test_raw_begin_with_pending_write_reuse_leaves_transaction_open_uncommitted(
    tmp_path: Path,
) -> None:
    """A reuse under a caller-held raw ``BEGIN`` with a pending write.

    The caller's transaction is left open and the pending row uncommitted --
    a fresh reader does not see it until the caller's own explicit commit.
    """
    db_path = tmp_path / "raw-begin-pending.db"
    producer = _SharedChildProducer("shared derived child", attach_edge=True)
    async with _connection(str(db_path)) as conn:
        store = await _open_store(conn, producer)
        source = await store.create_thought(_source("src-1", "source body"))
        first = await store.derive_existing(source.thought_id)
        assert first.created == 1

        await store._db.execute("BEGIN")
        await store._db.execute(
            store._CORE_INSERT_SQL,
            store._thought_to_core_params(_source("caller-raw", "caller raw body")),
        )
        assert store._db.in_transaction is True

        second = await store.derive_existing(source.thought_id)
        assert (second.created, second.reused, second.skipped) == (0, 1, 0)
        assert store._db.in_transaction is True
        assert await store._get_thought_row("caller-raw") is not None

        async with _reopened(str(db_path)) as mid_reader:
            assert await mid_reader.get_thought("caller-raw") is None

        await store._db.commit()
        assert store._db.in_transaction is False

    async with _reopened(str(db_path)) as fresh:
        assert await fresh.get_thought("caller-raw") is not None

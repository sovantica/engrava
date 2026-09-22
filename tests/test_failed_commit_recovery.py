"""A failed runtime ``COMMIT`` ends its own transaction, one way or another.

Both ``suspend_auto_commit``'s clean-exit commit and the plain ``_maybe_commit``
call used to hand the ``COMMIT`` straight to the driver with no handling of the
commit call itself failing. ``COMMIT`` can fail on its own account -- most
concretely, a concurrent connection still holding a read lock when
``busy_timeout`` expires, reported as ``SQLITE_BUSY`` -- while the write
transaction stays open on the connection: SQLite does not roll a transaction
back just because its own ``COMMIT`` failed. Left alone, a later, unrelated
write on the *same* connection would then commit the failed operation's
changes right alongside its own -- reproduced below, on a real file-backed
database, with nothing more exotic than one blocking reader.

The fix (``SqliteEngravaCore._commit_or_recover``, shared by both call sites)
attempts a rollback of the now-known-bad transaction when the commit itself
fails, and quarantines the connection via the existing
``_quarantine_connection`` mechanism when that rollback also cannot be
trusted -- mirroring ``_write_readback_savepoint``'s own unwind-failure
handling for the same reason: a caller must never be able to reach a commit
that could flush an indeterminate transaction.

Every test in this module was confirmed failing against the pre-fix code
(``suspend_auto_commit``'s commit outside the ``except``, ``_maybe_commit``'s
bare ``await self._db.commit()``) before this file was written: the two
real-contention tests reproduced the durability leak this closes, and the
monkeypatched ones hung or propagated the wrong exception.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import ConnectionQuarantinedError, SqliteEngravaCore
from tests.test_partial_field_updates import _thought

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures + helpers -- real cross-connection contention
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_path(tmp_path: Path) -> str:
    """A real on-disk database file in rollback-journal mode (not WAL).

    Non-WAL is required: only in rollback-journal mode can a concurrent
    reader's own lock make the *writer's* ``COMMIT`` itself fail with
    ``SQLITE_BUSY`` once ``busy_timeout`` elapses. In WAL mode a reader never
    blocks a writer's commit this way, so the defect this module pins would
    not be reachable through it.
    """
    path = str(tmp_path / "failed-commit.db")
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    bootstrap = SqliteEngravaCore(conn)
    await bootstrap.ensure_schema()
    await conn.close()
    return path


async def _open_store(
    db_path: str, *, busy_timeout_ms: int
) -> tuple[aiosqlite.Connection, SqliteEngravaCore]:
    """Open a fresh connection + store against *db_path* with a given busy timeout.

    Deliberately small ``busy_timeout_ms`` so a real busy failure surfaces in
    milliseconds instead of production's default seconds.
    """
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    return conn, SqliteEngravaCore(conn)


async def _open_blocking_reader(db_path: str) -> aiosqlite.Connection:
    """Open a second connection holding a read transaction (a SHARED lock).

    A small write transaction stays buffered in the writer's page cache and
    only needs to escalate to an ``EXCLUSIVE`` lock at ``COMMIT`` time -- which
    this reader's own ``SHARED`` lock blocks until the writer's
    ``busy_timeout`` gives up, exactly the precondition the work item names.
    """
    reader = await aiosqlite.connect(db_path)
    await reader.execute("BEGIN")
    cursor = await reader.execute("SELECT thought_id FROM thought")
    await cursor.fetchall()
    return reader


async def _stored_thought_ids(db_path: str) -> set[str]:
    """Read back every ``thought_id`` from a fresh, independent connection."""
    conn = await aiosqlite.connect(db_path)
    try:
        cursor = await conn.execute("SELECT thought_id FROM thought")
        return {row[0] for row in await cursor.fetchall()}
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 1. A busy COMMIT does not become durable via a later, unrelated write
# ---------------------------------------------------------------------------


class TestABusyCommitDoesNotSurviveViaALaterWrite:
    """The defect this closes: a failed commit's write riding a later one."""

    async def test_standalone_write_busy_commit_is_not_published_later(self, db_path: str) -> None:
        """The plain ``_maybe_commit`` path (a single ``create_thought``)."""
        reader = await _open_blocking_reader(db_path)
        conn, store = await _open_store(db_path, busy_timeout_ms=200)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                await store.create_thought(_thought("t-should-not-land"))

            # The failed commit must not leave the transaction open for a
            # later, unrelated commit on this same connection to inherit.
            assert store._db.in_transaction is False

            # Remove the reader and perform an unrelated write.
            await reader.close()
            await store.create_thought(_thought("t-unrelated"))
        finally:
            # ``reader.close()`` is idempotent (a no-op once already closed),
            # so closing it here unconditionally -- even on the success path,
            # where it was already closed above -- never double-fails, and it
            # guarantees the reader's own connection worker thread is always
            # torn down, including when an assertion above fails first.
            await reader.close()
            await conn.close()

        # Reopen independently: only the unrelated write is durable.
        assert await _stored_thought_ids(db_path) == {"t-unrelated"}

    async def test_outermost_suspend_auto_commit_busy_commit_is_not_published_later(
        self, db_path: str
    ) -> None:
        """The outermost ``suspend_auto_commit`` / ``bulk_store`` path."""
        reader = await _open_blocking_reader(db_path)
        conn, store = await _open_store(db_path, busy_timeout_ms=200)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                await store.bulk_store([_thought("t-batch-1"), _thought("t-batch-2")])

            assert store._db.in_transaction is False

            await reader.close()
            await store.create_thought(_thought("t-unrelated"))
        finally:
            await reader.close()
            await conn.close()

        assert await _stored_thought_ids(db_path) == {"t-unrelated"}


# ---------------------------------------------------------------------------
# 2. A commit failure rolls back rather than leaving the transaction open
# ---------------------------------------------------------------------------


async def _in_memory_store() -> SqliteEngravaCore:
    """A schema-bootstrapped store over its own ``:memory:`` connection.

    Callers that do **not** expect this scenario to quarantine the connection
    must close ``store._db`` themselves once done (see the two fixture-free
    ``db``/``store`` helpers elsewhere in this suite for the pattern) --
    otherwise aiosqlite's non-daemon connection worker thread is never told to
    stop and the test process hangs at interpreter shutdown. A test that
    *does* quarantine the connection does not need to: ``_quarantine_connection``
    already schedules the real connection's own close, matching the existing
    quarantine tests elsewhere in this suite (e.g.
    ``tests/test_referential_integrity.py``), which likewise never call
    ``close()`` themselves after quarantining.
    """
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
    await store.ensure_schema()
    return store


class TestFailedCommitRollsBackTheTransaction:
    """A commit failure is unwound, not merely re-raised over an open transaction."""

    async def test_maybe_commit_path_rolls_back_on_commit_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = await _in_memory_store()
        commit_error = sqlite3.OperationalError("database is locked")

        async def failing_commit() -> None:
            raise commit_error

        monkeypatch.setattr(store._db, "commit", failing_commit)

        try:
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                await store.create_thought(_thought("t-1"))

            assert excinfo.value is commit_error
            assert store._db.in_transaction is False
            assert store._connection_quarantined is False
        finally:
            await store._db.close()

    async def test_suspend_auto_commit_path_rolls_back_on_commit_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = await _in_memory_store()
        commit_error = sqlite3.OperationalError("database is locked")

        async def failing_commit() -> None:
            raise commit_error

        async def _write_then_break_commit() -> None:
            async with store.suspend_auto_commit():
                await store.create_thought(_thought("t-1"))
                monkeypatch.setattr(store._db, "commit", failing_commit)

        try:
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                await _write_then_break_commit()

            assert excinfo.value is commit_error
            assert store._db.in_transaction is False
            assert store._connection_quarantined is False
        finally:
            await store._db.close()


# ---------------------------------------------------------------------------
# 3. A failed rollback after a failed commit quarantines the connection
# ---------------------------------------------------------------------------


class TestFailedRollbackAfterFailedCommitQuarantines:
    """When the compensating rollback itself cannot be trusted either.

    Neither test here closes the store's connection explicitly: quarantining
    already schedules the real connection's own close (see
    ``_in_memory_store``'s docstring).
    """

    async def test_quarantines_and_chains_both_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = await _in_memory_store()
        commit_error = sqlite3.OperationalError("database is locked")
        rollback_error = sqlite3.OperationalError("disk I/O error")

        async def failing_commit() -> None:
            raise commit_error

        async def failing_rollback() -> None:
            raise rollback_error

        monkeypatch.setattr(store._db, "commit", failing_commit)
        monkeypatch.setattr(store._db, "rollback", failing_rollback)

        with pytest.raises(sqlite3.OperationalError) as excinfo:
            await store.create_thought(_thought("t-1"))

        assert excinfo.value is commit_error
        assert excinfo.value.__cause__ is rollback_error
        assert store._connection_quarantined is True

    async def test_quarantined_connection_refuses_further_writes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = await _in_memory_store()
        commit_error_message = "database is locked"
        rollback_error_message = "disk I/O error"

        async def failing_commit() -> None:
            raise sqlite3.OperationalError(commit_error_message)

        async def failing_rollback() -> None:
            raise sqlite3.OperationalError(rollback_error_message)

        monkeypatch.setattr(store._db, "commit", failing_commit)
        monkeypatch.setattr(store._db, "rollback", failing_rollback)

        with pytest.raises(sqlite3.OperationalError):
            await store.create_thought(_thought("t-1"))

        with pytest.raises(ConnectionQuarantinedError):
            await store.create_thought(_thought("t-unrelated-after-quarantine"))


# ---------------------------------------------------------------------------
# 4. Cancellation during the commit, or during its recovery
# ---------------------------------------------------------------------------


class TestCancellationDuringCommit:
    """Cancellation is caught alongside every other exception, and unwound."""

    async def test_cancellation_during_commit_still_rolls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = await _in_memory_store()

        async def cancelling_commit() -> None:
            raise asyncio.CancelledError

        monkeypatch.setattr(store._db, "commit", cancelling_commit)

        try:
            with pytest.raises(asyncio.CancelledError):
                await store.create_thought(_thought("t-1"))

            assert store._db.in_transaction is False
            assert store._connection_quarantined is False
        finally:
            await store._db.close()

    async def test_cancellation_during_the_rollback_wins_over_the_commit_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancellation raised by the unwind itself outranks the error it unwinds.

        Mirrors ``_write_readback_savepoint``'s own rule for the identical
        shape: the commit failed for an ordinary reason, but the rollback
        attempting to recover from it is what actually gets cancelled -- the
        caller must see that cancellation, not the original commit error, and
        the connection must still be quarantined since the transaction's real
        state was never confirmed. Does not close the connection explicitly --
        quarantining already schedules that (see ``_in_memory_store``'s
        docstring).
        """
        store = await _in_memory_store()
        commit_error_message = "database is locked"

        async def failing_commit() -> None:
            raise sqlite3.OperationalError(commit_error_message)

        async def cancelling_rollback() -> None:
            raise asyncio.CancelledError

        monkeypatch.setattr(store._db, "commit", failing_commit)
        monkeypatch.setattr(store._db, "rollback", cancelling_rollback)

        with pytest.raises(asyncio.CancelledError):
            await store.create_thought(_thought("t-1"))

        assert store._connection_quarantined is True

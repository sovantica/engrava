"""A cancellation between an executed ``BEGIN``/``SAVEPOINT`` and its ``await`` leaves nothing open.

``delete_thought`` and ``delete_edge`` each open their own ``BEGIN
IMMEDIATE`` before reading the row their delete depends on.
``_write_readback_savepoint`` opens its own transaction (when one is not
already open) and then a ``SAVEPOINT``, for every guarded write in this
module. All three statements are awaited individually, and a cancellation
can be delivered to the awaiting coroutine *after* SQLite has already
executed the statement on aiosqlite's worker thread but *before* that
``await`` itself returns control here -- the worker thread is a real OS
thread, unaffected by asyncio cancelling the ``Future`` wrapping it. Code
written as ``if opened_transaction: await self._db.execute("BEGIN
IMMEDIATE")`` *outside* the ``try`` whose ``except`` rolls a self-opened
transaction back cannot see that cancellation at all: the statement ran, a
real transaction now exists, and nothing ever closes it.

Each test below wraps the connection's own ``execute`` so that the specific
statement under test runs for real -- SQLite genuinely executes it -- and
then raises ``asyncio.CancelledError`` in the awaiting coroutine, exactly
reproducing the race described above without needing real concurrency or
timing to land it. Every test then asserts the same three things: the
``CancelledError`` propagates unchanged, a transaction this call opened is
not left open (``in_transaction`` is ``False``), and the connection was
never quarantined -- a plain rollback was enough, because nothing had
written anything yet.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from tests.test_partial_field_updates import _edge, _thought

if TYPE_CHECKING:
    from collections.abc import Callable


async def _in_memory_store() -> SqliteEngravaCore:
    """A schema-bootstrapped store over its own ``:memory:`` connection.

    Mirrors ``tests/test_failed_commit_recovery.py``'s own helper of the
    same name and the same caller obligation: a test that does not
    quarantine the connection must close ``store._db`` itself, or
    aiosqlite's non-daemon worker thread outlives the test.
    """
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(db, embedding_provider=None, auto_embed=False)
    await store.ensure_schema()
    return store


def _cancel_after_real_execute(conn: aiosqlite.Connection, match_sql: str) -> Callable[[], bool]:
    """Make the *first* ``conn.execute(match_sql)`` run for real, then raise ``CancelledError``.

    Every other statement (including a second call with the same SQL, since
    a rollback issues a bare ``ROLLBACK`` with no risk of colliding with
    ``match_sql``) passes through to the real ``execute`` untouched.

    Args:
        conn: The connection whose ``execute`` is wrapped.
        match_sql: The exact SQL text to intercept once.

    Returns:
        A zero-argument callable that reports whether the interception has
        already fired, so a test can confirm the statement it meant to
        target was actually reached.

    """
    original_execute = conn.execute
    fired = False

    async def wrapper(sql: str, *args: object, **kwargs: object) -> object:
        nonlocal fired
        if not fired and sql == match_sql:
            fired = True
            await original_execute(sql, *args, **kwargs)
            raise asyncio.CancelledError
        return await original_execute(sql, *args, **kwargs)

    conn.execute = wrapper  # type: ignore[method-assign, assignment]
    return lambda: fired


class TestDeleteThoughtCancelledDuringItsOwnBegin:
    """``delete_thought`` opens ``BEGIN IMMEDIATE`` itself, before reading anything."""

    async def test_cancelled_after_begin_executes_rolls_back(self) -> None:
        store = await _in_memory_store()
        await store.create_thought(_thought("t-1"))

        has_fired = _cancel_after_real_execute(store._db, "BEGIN IMMEDIATE")

        try:
            with pytest.raises(asyncio.CancelledError):
                await store.delete_thought("t-1")

            assert has_fired(), "the interception never actually reached BEGIN IMMEDIATE"
            assert store._db.in_transaction is False, (
                "a transaction this call opened was left open after cancellation"
            )
            assert store._connection_quarantined is False
        finally:
            await store._db.close()


class TestDeleteEdgeCancelledDuringItsOwnBegin:
    """``delete_edge`` opens ``BEGIN IMMEDIATE`` itself, before reading the before-image."""

    async def test_cancelled_after_begin_executes_rolls_back(self) -> None:
        store = await _in_memory_store()
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge("e-1"))

        has_fired = _cancel_after_real_execute(store._db, "BEGIN IMMEDIATE")

        try:
            with pytest.raises(asyncio.CancelledError):
                await store.delete_edge("e-1")

            assert has_fired(), "the interception never actually reached BEGIN IMMEDIATE"
            assert store._db.in_transaction is False, (
                "a transaction this call opened was left open after cancellation"
            )
            assert store._connection_quarantined is False
        finally:
            await store._db.close()


class TestWriteReadbackSavepointCancelledDuringItsOwnEntry:
    """``_write_readback_savepoint`` opens its own transaction, then a ``SAVEPOINT``.

    ``update_thought`` is the vehicle here (rather than ``delete_thought`` /
    ``delete_edge``) specifically because it never opens a transaction of
    its own before calling ``_write_readback_savepoint`` — a fresh,
    in-memory store guarantees ``_write_readback_savepoint`` itself is the
    one sampling ``opened_transaction`` as ``True`` and issuing the ``BEGIN``
    under test, isolating this shared helper's own entry-cancellation
    handling from either delete path's separate one above.
    """

    async def test_cancelled_after_begin_executes_rolls_back(self) -> None:
        store = await _in_memory_store()
        await store.create_thought(_thought("t-1"))

        has_fired = _cancel_after_real_execute(store._db, "BEGIN DEFERRED")

        try:
            with pytest.raises(asyncio.CancelledError):
                await store.update_thought("t-1", essence="new essence")

            assert has_fired(), "the interception never actually reached BEGIN DEFERRED"
            assert store._db.in_transaction is False, (
                "a transaction this call opened was left open after cancellation"
            )
            assert store._connection_quarantined is False
        finally:
            await store._db.close()

    async def test_cancelled_after_savepoint_executes_rolls_back(self) -> None:
        store = await _in_memory_store()
        await store.create_thought(_thought("t-1"))

        has_fired = _cancel_after_real_execute(store._db, "SAVEPOINT update_thought_readback")

        try:
            with pytest.raises(asyncio.CancelledError):
                await store.update_thought("t-1", essence="new essence")

            assert has_fired(), (
                "the interception never actually reached SAVEPOINT update_thought_readback"
            )
            assert store._db.in_transaction is False, (
                "a transaction this call opened was left open after cancellation, even "
                "though this call cannot prove the SAVEPOINT itself was ever created"
            )
            assert store._connection_quarantined is False
        finally:
            await store._db.close()

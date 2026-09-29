"""Failure atomicity of the write-then-confirming-read-back sequence.

``update_thought``, ``restore_thought``, ``update_edge`` and ``update_action``
all write a row, then read it back to confirm and report what actually landed,
and only then decide whether to commit. A read-back failure -- the row
vanished, the row mapper rejected a stored value -- must not leave the write
pending in the connection's transaction, where a later, unrelated commit on
the same connection would publish a mutation whose own operation had
reported failure. Each write + read-back therefore runs in a ``SAVEPOINT``
(``SqliteEngravaCore._write_readback_savepoint``), mirroring the
``_delete_thought_atomic`` / ``_delete_thought_children_explicit`` pattern
already in this module, so a read-back failure unwinds only this operation's
own write -- never a caller-owned transaction already in progress.

``update_edge`` / ``update_action`` also capture their ``UPDATE``'s cursor and
reject a zero-row match immediately, before the read-back runs. Without that
check, a row deleted after the initial read and re-created under the same
identifier before the read-back would be reported as though this call had
updated it, when it had written nothing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionStatus,
    SqliteEngravaCore,
    StaleDataError,
)
from tests.test_partial_field_updates import (
    _action,
    _edge,
    _interleave_after_statement,
    _interleave_once,
    _row,
    _thought,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite with the head schema (journal table included)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    bootstrap = SqliteEngravaCore(conn, journal_enabled=True)
    await bootstrap.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with journaling off — the plain write path."""
    return SqliteEngravaCore(db)


@pytest.fixture
async def journaling_store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with journaling on — lets a test assert no phantom entry was written."""
    return SqliteEngravaCore(db, journal_enabled=True)


def _fail_once(store: SqliteEngravaCore, method_name: str, exc: BaseException) -> None:
    """Replace ``method_name`` so its first call raises ``exc`` instead of running.

    Stands in for a read-back failure (the row is missing, SQLite errors, or
    the row mapper rejects a value) without needing to actually corrupt a
    stored value: the write that precedes the read-back has already landed in
    storage by the time this fires, which is what lets these tests assert on
    the row directly. Only the *first* call raises; a later, legitimate call
    (e.g. a subsequent, unrelated operation reusing the same read-back
    method) runs normally.
    """
    original = getattr(store, method_name)
    fired = False

    async def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal fired
        if not fired:
            fired = True
            raise exc
        return await original(*args, **kwargs)

    setattr(store, method_name, wrapper)


# ---------------------------------------------------------------------------
# 1. A read-back failure leaves no pending mutation from that operation
# ---------------------------------------------------------------------------


class TestReadBackFailureLeavesNoPendingWrite:
    """The operation's own write does not survive a read-back it cannot confirm."""

    async def test_update_thought_readback_failure_leaves_no_pending_write(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The essence the failed call tried to write is not on the row afterward."""
        await store.create_thought(_thought("t-1", essence="original essence"))

        _fail_once(store, "_read_back_thought", RuntimeError("row mapper rejected a value"))

        with pytest.raises(RuntimeError, match="row mapper rejected"):
            await store.update_thought("t-1", essence="new essence")

        row = await _row(db, "thought", "thought_id", "t-1")
        assert row["essence"] == "original essence"
        # Nothing is left open on the connection for a later commit to inherit.
        assert store._db.in_transaction is False

    async def test_update_edge_readback_failure_leaves_no_pending_write(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The same protection applies to the edge write path."""
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge("e-1", weight=0.5))

        _fail_once(store, "_read_back_edge", RuntimeError("forced read-back failure"))

        with pytest.raises(RuntimeError, match="forced read-back failure"):
            await store.update_edge("e-1", weight=0.9)

        row = await _row(db, "edge", "edge_id", "e-1")
        assert row["weight"] == 0.5
        assert store._db.in_transaction is False


# ---------------------------------------------------------------------------
# 2. A later, unrelated commit cannot publish the failed operation's write
# ---------------------------------------------------------------------------


class TestALaterCommitCannotPublishAFailedWrite:
    async def test_a_later_unrelated_update_does_not_carry_the_failed_write_with_it(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The defect this closes: a failed op's write riding along on a later commit."""
        await store.create_thought(_thought("t-1", essence="original essence"))
        await store.create_thought(_thought("t-2", essence="unrelated"))

        _fail_once(store, "_read_back_thought", RuntimeError("forced read-back failure"))

        with pytest.raises(RuntimeError, match="forced read-back failure"):
            await store.update_thought("t-1", essence="should never land")

        # An unrelated write on the same connection, committed normally.
        await store.update_thought("t-2", essence="unrelated, changed")

        # It does not publish the earlier, failed operation's write.
        row_t1 = await _row(db, "thought", "thought_id", "t-1")
        assert row_t1["essence"] == "original essence"
        row_t2 = await _row(db, "thought", "thought_id", "t-2")
        assert row_t2["essence"] == "unrelated, changed"


# ---------------------------------------------------------------------------
# 3. A caller-owned transaction survives a failed operation intact
# ---------------------------------------------------------------------------


class TestCallerOwnedTransactionSurvivesAFailedOperation:
    async def test_the_callers_earlier_write_survives_a_sibling_calls_failure(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A bare connection-wide rollback would be the wrong instrument here.

        The caller opens its own transaction with ``suspend_auto_commit``,
        writes ``t-2``, then a *different* call on the same connection --
        ``update_thought`` on ``t-1`` -- fails its read-back. Catching that
        failure (as any reasonable caller would, to keep going) and letting
        the window close cleanly must still publish the caller's own earlier
        write; only the failed call's own write may be gone.
        """
        await store.create_thought(_thought("t-1", essence="original essence"))
        await store.create_thought(_thought("t-2", essence="second original"))

        async with store.suspend_auto_commit():
            await store.update_thought("t-2", essence="caller work survives")

            _fail_once(store, "_read_back_thought", RuntimeError("forced read-back failure"))
            with pytest.raises(RuntimeError, match="forced read-back failure"):
                await store.update_thought("t-1", essence="should not survive")

        row_t2 = await _row(db, "thought", "thought_id", "t-2")
        assert row_t2["essence"] == "caller work survives"
        row_t1 = await _row(db, "thought", "thought_id", "t-1")
        assert row_t1["essence"] == "original essence"


# ---------------------------------------------------------------------------
# 4. An UPDATE matching no row is rejected even if a row reappears by read-back
# ---------------------------------------------------------------------------


class TestZeroRowUpdateIsRejectedDespiteRecreation:
    """The UPDATE's own rowcount decides this, not what the read-back finds."""

    async def test_update_edge_rejects_when_the_row_reappears_before_the_readback(
        self,
        journaling_store: SqliteEngravaCore,
    ) -> None:
        """The row is absent when the UPDATE runs; a same-id row exists again after.

        A delete-and-recreate *after* the UPDATE cannot be caught by that
        UPDATE's own rowcount (already 1 by then). The sequence that matters
        is the other one: absent *when the UPDATE runs* (rowcount 0), then
        recreated before the read-back. Without the rowcount check, the
        read-back finds the recreated row and reports it as though this call
        had updated it.

        The zero-row match raises ``StaleDataError``: a row deleted then
        recreated under the same id is the "matched no row" case that error
        names uniformly across ``update_thought`` / ``update_edge`` /
        ``update_action``. The test pins that the call raises and writes no
        journal entry, so the recreated row is never reported as this call's
        own result.
        """
        store = journaling_store
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2"))
        await store.create_edge(_edge("e-1", weight=0.5))

        async def _delete() -> None:
            await store._db.execute("DELETE FROM edge WHERE edge_id = ?", ("e-1",))

        _interleave_once(store, "_get_edge_row", _delete)

        async def _recreate() -> None:
            await store._db.execute(
                "INSERT INTO edge "
                "(edge_id, from_thought_id, to_thought_id, edge_type, weight, "
                " created_cycle, source, decay_multiplier, valid_from, valid_until, "
                " metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("e-1", "t-1", "t-2", "ASSOCIATED", 0.42, 0, "EXPERIENCE", 1.0, None, None, "{}"),
            )

        _interleave_after_statement(store, "UPDATE edge SET", _recreate)

        with pytest.raises(StaleDataError):
            await store.update_edge("e-1", weight=0.9)

        assert store._journal is not None
        entries = await store._journal.get_entries(target_id="e-1", mutation_type="UPDATE_EDGE")
        assert entries == []

    async def test_update_action_rejects_when_the_row_reappears_before_the_readback(
        self,
        journaling_store: SqliteEngravaCore,
    ) -> None:
        """The action counterpart of the edge case above.

        Hooks ``_get_action_row`` -- the raw-row read ``update_action`` uses
        internally to reach the ``revision`` -- rather than ``_get_action``
        (the domain-mapped read): ``update_action`` does not call
        ``_get_action`` at all, so hooking it would leave the delete below never firing and the
        recreate below colliding with the still-live row instead of
        reproducing the intended race.
        """
        store = journaling_store
        await store.create_thought(_thought("t-1"))
        await store.create_action(_action("a-1"))

        async def _delete() -> None:
            await store._db.execute("DELETE FROM action WHERE action_id = ?", ("a-1",))

        _interleave_once(store, "_get_action_row", _delete)

        async def _recreate() -> None:
            await store._db.execute(
                "INSERT INTO action "
                "(action_id, source_thought_id, action_type, intent, "
                " status, verification_status, raw_metrics_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("a-1", "t-1", "CLI_OUTPUT", "do the thing", "PLANNED", "PENDING", None),
            )

        _interleave_after_statement(store, "UPDATE action SET", _recreate)

        with pytest.raises(StaleDataError):
            await store.update_action("a-1", status=ActionStatus.EXECUTING)

        assert store._journal is not None
        entries = await store._journal.get_entries(target_id="a-1", mutation_type="UPDATE_ACTION")
        assert entries == []

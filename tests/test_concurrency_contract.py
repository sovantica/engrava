"""The concurrency contract the documentation states — guarantees and non-guarantees.

``docs/concurrency.md`` describes what one store promises concurrent callers and,
just as importantly, what it does **not** promise. A promise nobody tests drifts;
a *withheld* promise drifts just as quietly, because nothing fails when the docs
keep claiming safety the code stopped providing. Every claim on that page that
can be exercised in-process is pinned here, in both directions:

* **Guarantees** — an edit writes only the columns it owns, so two edits to
  different fields both survive; a confirmation bump is relative, so it survives
  a second connection.
* **Non-guarantees** — the read-modify-write inside ``update_thought`` is not a
  critical section, so a competing edit to the same field is silently discarded;
  ``StaleDataError`` fires only when a competing writer moved ``updated_cycle``,
  and the edge/action write paths carry no version guard at all; a state-machine
  check is made against the row the call read, so two legal transitions can
  compose into a forbidden one; a ``suspend_auto_commit`` window belongs to the
  store instance, not to the task that opened it; and nothing that orders
  operations inside one store reaches a second store on the same file.

The interleavings are deterministic. The single-store cases reuse the one-shot
seam from ``test_partial_field_updates`` to run the competing write at the exact
point between an operation's read and its write; the transaction-window case is
driven by :class:`asyncio.Event` handshakes. No test depends on wall-clock timing
or on how the event loop happens to schedule.
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
    StaleDataError,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
)
from tests.test_partial_field_updates import _interleave_once

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _thought(
    thought_id: str = "t-1",
    *,
    essence: str = "essence",
    content: str = "the stored content",
) -> ThoughtRecord:
    """Build a minimal ACTIVE thought.

    ``content`` is a parameter here (unlike the partial-update suite's builder,
    which derives it from the id) because the deduplication cases need two
    records that differ in id and agree byte for byte in ``content``.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.75,
    )


def _edge(edge_id: str = "e-1", *, weight: float = 0.5) -> EdgeRecord:
    """Build an edge between ``t-1`` and ``t-2``."""
    return EdgeRecord(
        edge_id=edge_id,
        from_thought_id="t-1",
        to_thought_id="t-2",
        edge_type=EdgeType.ASSOCIATED,
        weight=weight,
        created_cycle=0,
        source=KnowledgeSource.EXPERIENCE,
    )


def _action(action_id: str = "a-1") -> ActionRecord:
    """Build a PLANNED / PENDING action rooted at ``t-1``."""
    return ActionRecord(
        action_id=action_id,
        source_thought_id="t-1",
        action_type=ActionType.CLI_OUTPUT,
        intent="do the thing",
        status=ActionStatus.PLANNED,
        verification_status=VerificationStatus.PENDING,
    )


async def _row(db: aiosqlite.Connection, thought_id: str) -> aiosqlite.Row:
    """Read a thought row straight from storage — never through a return value."""
    cursor = await db.execute("SELECT * FROM thought WHERE thought_id = ?", (thought_id,))
    row = await cursor.fetchone()
    assert row is not None
    return row


async def _thought_ids(db: aiosqlite.Connection) -> list[str]:
    """Every stored thought id, ordered, read straight from storage."""
    cursor = await db.execute("SELECT thought_id FROM thought ORDER BY thought_id")
    return [row["thought_id"] for row in await cursor.fetchall()]


async def _connect(path: str) -> aiosqlite.Connection:
    """Open a connection configured the way ``from_config`` configures one."""
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA busy_timeout = 5000")
    await conn.execute("PRAGMA foreign_keys = ON")
    return conn


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite with the head schema applied."""
    conn = await _connect(":memory:")
    bootstrap = SqliteEngravaCore(conn)
    await bootstrap.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """The store under test — one connection, many tasks."""
    return SqliteEngravaCore(db)


@pytest.fixture
async def two_stores(
    tmp_path: Path,
) -> AsyncIterator[tuple[SqliteEngravaCore, SqliteEngravaCore, aiosqlite.Connection]]:
    """Two stores over **one** on-disk database file, each with its own connection.

    The in-process stand-in for two processes: separate connections are what
    makes the topology, and a second process differs only in that no in-process
    lock could even in principle be shared. WAL and the busy timeout are set on
    both, exactly as ``from_config`` sets them, so the file-level story is the
    supported one and only engrava's own ordering is under test.
    """
    path = str(tmp_path / "shared.db")
    conn_a = await _connect(path)
    conn_b = await _connect(path)
    store_a = SqliteEngravaCore(conn_a)
    store_b = SqliteEngravaCore(conn_b)
    await store_a.ensure_schema()
    await store_b.ensure_schema()
    yield store_a, store_b, conn_a
    await conn_a.close()
    await conn_b.close()


# ---------------------------------------------------------------------------
# One store, many tasks
# ---------------------------------------------------------------------------


class TestOneStoreManyTasks:
    """What sharing a single store between concurrent tasks does and does not buy."""

    async def test_interleaved_edits_to_different_fields_both_survive(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Two edits that touch different columns do not overwrite each other.

        The guarantee the docs state: an update writes only the columns the
        operation gave a new value to, so a competing edit landing between this
        one's read and its write is preserved as long as the two edits do not
        name the same field. The guarantee has an exception, pinned by
        ``test_a_competing_cycle_stamp_rejects_an_edit_to_a_different_field``
        below: a competing writer who moves ``updated_cycle`` trips the version
        guard and this update is rejected outright, different field or not.
        """
        await store.create_thought(_thought())
        landed: list[str] = []

        async def _competing_edit() -> None:
            await store.update_thought("t-1", priority=Priority.P1)
            landed.append((await _row(db, "t-1"))["priority"])

        _interleave_once(store, "_get_thought_row", _competing_edit)

        await store.update_thought("t-1", essence="mine")

        # Precondition: the competing edit really reached storage first.
        assert landed == [Priority.P1.value]
        row = await _row(db, "t-1")
        assert row["essence"] == "mine"
        assert row["priority"] == Priority.P1.value

    async def test_interleaved_edits_to_one_field_keep_only_the_later_write(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A competing edit to the same field is discarded, silently.

        The non-guarantee. ``update_thought`` reads the row, evolves it in
        memory, then writes — and aiosqlite serialises *statements*, not method
        bodies, so a second task's whole update can land in that window. The
        write that issues its UPDATE last wins and the other is gone, with no
        error and nothing in the row to show it ever happened.
        """
        await store.create_thought(_thought())
        landed: list[str] = []

        async def _competing_edit() -> None:
            await store.update_thought("t-1", essence="from the other task")
            landed.append((await _row(db, "t-1"))["essence"])

        _interleave_once(store, "_get_thought_row", _competing_edit)

        returned = await store.update_thought("t-1", essence="from this task")

        # Precondition: the competing edit really reached storage first, so the
        # assertion below is about it being overwritten, not about it never
        # having happened.
        assert landed == ["from the other task"]
        row = await _row(db, "t-1")
        assert row["essence"] == "from this task"
        assert returned.essence == "from this task"

    async def test_a_competing_cycle_stamp_rejects_an_edit_to_a_different_field(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The version guard is on every update, so it rejects unrelated edits too.

        This is the limit of the "different fields both survive" guarantee. The
        competing writer here touches **only** ``updated_cycle``; this call
        touches only ``essence``. They share no column — and the edit is still
        rejected, because the guard is part of every update's ``WHERE`` clause,
        not something that fires only when the two edits collide. Nothing of the
        rejected update reaches storage.
        """
        await store.create_thought(_thought())

        async def _competing_cycle_stamp() -> None:
            await store.update_thought("t-1", updated_cycle=7)

        _interleave_once(store, "_get_thought_row", _competing_cycle_stamp)

        with pytest.raises(StaleDataError) as exc_info:
            await store.update_thought("t-1", essence="mine")

        row = await _row(db, "t-1")
        assert row["essence"] == "essence"
        assert row["updated_cycle"] == 7
        assert exc_info.value.entity_type == "ThoughtRecord"
        assert exc_info.value.entity_id == "t-1"
        assert exc_info.value.expected_version == 0

    async def test_a_row_deleted_in_the_window_also_raises_stale_data_error(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A vanished row raises ``StaleDataError``, not ``ThoughtNotFoundError``.

        ``StaleDataError`` is raised on ``rowcount == 0``, and a guarded UPDATE
        matches no row for two distinct reasons: the cycle moved, or the row is
        gone. So the error does **not** mean "somebody stamped a cycle" — that
        is why the documentation states the condition as "the guarded update
        matched no row" rather than naming only the cycle. ``update_thought``
        raising ``ThoughtNotFoundError`` for a missing row is true only of the
        read it does *before* the write.
        """
        await store.create_thought(_thought())
        await store.create_thought(_thought("t-keep", content="a row nobody touches"))
        deleted: list[bool] = []

        async def _competing_delete() -> None:
            deleted.append(await store.delete_thought("t-1"))

        _interleave_once(store, "_get_thought_row", _competing_delete)

        with pytest.raises(StaleDataError) as exc_info:
            await store.update_thought("t-1", essence="mine")

        # Precondition: the row really was deleted inside the window.
        assert deleted == [True]
        assert await _thought_ids(db) == ["t-keep"]
        assert exc_info.value.entity_id == "t-1"
        assert exc_info.value.expected_version == 0

    async def test_upsert_by_hash_overwrites_a_competing_edit_without_raising(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The hash probe and the in-place update are not one atomic step.

        ``upsert_by_hash`` documents ``StaleDataError`` for a row modified
        between its probe and its update. It inherits ``update_thought``'s
        guard, so an ordinary competing edit passes straight through it: the
        upsert's value overwrites the competing one and no error is raised.
        """
        await store.create_thought(_thought(content="shared content"))
        landed: list[str] = []

        async def _competing_edit() -> None:
            await store.update_thought("t-1", essence="from the other task")
            landed.append((await _row(db, "t-1"))["essence"])

        _interleave_once(store, "_get_thought_row", _competing_edit)

        await store.upsert_by_hash(
            _thought("t-unused", essence="from the upsert", content="shared content"),
        )

        # Precondition: the competing edit really reached storage first.
        assert landed == ["from the other task"]
        row = await _row(db, "t-1")
        assert row["essence"] == "from the upsert"
        assert await _thought_ids(db) == ["t-1"]


class TestEdgeAndActionUpdatesCarryNoVersionGuard:
    """The other update paths have the window too, and not even the cycle guard."""

    async def test_interleaved_edge_edits_to_one_field_keep_only_the_later_write(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """``update_edge`` is keyed on the id alone, so it cannot reject anything.

        Thought updates at least carry an ``updated_cycle`` guard, narrow as it
        is. The edge write path carries none, so a competing edit is discarded
        with nothing that could ever have flagged it.
        """
        await store.create_thought(_thought("t-1"))
        await store.create_thought(_thought("t-2", content="the other end"))
        await store.create_edge(_edge())
        landed: list[float] = []

        async def _competing_edit() -> None:
            await store.update_edge("e-1", weight=0.9)
            cursor = await db.execute("SELECT weight FROM edge WHERE edge_id = 'e-1'")
            row = await cursor.fetchone()
            assert row is not None
            landed.append(row["weight"])

        _interleave_once(store, "_get_edge_row", _competing_edit)

        await store.update_edge("e-1", weight=0.1)

        # Precondition: the competing edit really reached storage first.
        assert landed == [0.9]
        cursor = await db.execute("SELECT weight FROM edge WHERE edge_id = 'e-1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row["weight"] == 0.1


class TestStateMachineChecksUseTheStateThisCallRead:
    """A lifecycle check that already passed is not re-checked against storage."""

    async def test_interleaved_lifecycle_moves_can_produce_a_forbidden_state(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Two legal transitions compose into one the state machine forbids.

        ``update_thought`` validates the transition against the record *it*
        read. A competing writer moving the row in between does not invalidate
        that check, so ``ACTIVE -> DONE`` can land on a row that is already
        ``ARCHIVED`` — and ``ARCHIVED -> DONE`` is not an allowed edge.
        """
        await store.create_thought(_thought())
        landed: list[str] = []

        async def _competing_archive() -> None:
            await store.update_thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED)
            landed.append((await _row(db, "t-1"))["lifecycle_status"])

        _interleave_once(store, "_get_thought_row", _competing_archive)

        await store.update_thought("t-1", lifecycle_status=LifecycleStatus.DONE)

        # Precondition: the row really was ARCHIVED when the second write landed.
        assert landed == [LifecycleStatus.ARCHIVED.value]
        row = await _row(db, "t-1")
        assert row["lifecycle_status"] == LifecycleStatus.DONE.value
        # ...and that edge is not one the state machine would have allowed.
        assert not LifecycleStatus.ARCHIVED.can_transition_to(LifecycleStatus.DONE)

    async def test_interleaved_action_moves_can_produce_a_forbidden_state(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The same hole in the action state machine.

        ``PLANNED -> BLOCKED`` is validated against the read state, so it lands
        on a row another writer already moved to ``EXECUTING`` — and
        ``EXECUTING -> BLOCKED`` is not an allowed edge.
        """
        await store.create_thought(_thought())
        await store.create_action(_action())
        landed: list[str] = []

        async def _competing_start() -> None:
            await store.update_action("a-1", status=ActionStatus.EXECUTING)
            cursor = await db.execute("SELECT status FROM action WHERE action_id = 'a-1'")
            row = await cursor.fetchone()
            assert row is not None
            landed.append(row["status"])

        _interleave_once(store, "_get_action", _competing_start)

        await store.update_action("a-1", status=ActionStatus.BLOCKED)

        # Precondition: the row really was EXECUTING when the second write landed.
        assert landed == [ActionStatus.EXECUTING.value]
        cursor = await db.execute("SELECT status FROM action WHERE action_id = 'a-1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row["status"] == ActionStatus.BLOCKED.value
        # ...and that edge is not one the state machine would have allowed.
        assert not ActionStatus.EXECUTING.can_transition_to(ActionStatus.BLOCKED)


class TestOneStorePerEventLoop:
    """What actually breaks when a store is driven from a second event loop."""

    def test_a_second_loop_breaks_the_stores_own_lock_but_not_its_connection(
        self,
        tmp_path: Path,
    ) -> None:
        """The connection is not loop-bound; the store's ``asyncio.Lock`` is.

        The reason to keep one store per loop is the store's own
        synchronisation, not aiosqlite: aiosqlite creates each operation's
        future on the *calling* loop, so a plain read from a second loop works.
        The deduplication lock is an :class:`asyncio.Lock`, which binds to the
        first loop that has to **wait** on it and rejects a waiter from any
        other loop thereafter.

        The uncontended fast path never binds, which is what makes this
        dangerous in practice: a store shared across loops looks healthy until
        two callers first contend. Both halves are asserted here so the
        documented reason cannot quietly become as wrong as the one it replaced.

        Provoking the rejection is also a way to strand a task, so the test
        asserts its own cleanliness. The rejection is raised by one of two
        concurrent calls, and a ``gather`` whose child raises does not cancel
        that child's siblings — so the other call can still be running when the
        caller resumes and the loops are closed. Two assertions rule that out:
        neither loop holds a task afterwards, and storage holds exactly the rows
        the calls that ran to completion inserted (``a-1`` and ``a-2`` from the
        first loop, ``b-1`` from the second). Both discriminate — with a plain
        ``gather`` in ``_open_and_contend`` the first reports a pending
        ``create_thought`` and the second reports ``b-1`` missing, because that
        call is still short of its INSERT when the assertions read storage.
        """
        db_path = str(tmp_path / "loops.db")
        opened: dict[str, object] = {}

        async def _open_and_contend(prefix: str) -> None:
            if "store" not in opened:
                conn = await _connect(db_path)
                store = SqliteEngravaCore(conn)
                await store.ensure_schema()
                opened["conn"] = conn
                opened["store"] = store
            store = opened["store"]
            assert isinstance(store, SqliteEngravaCore)
            # Two dedup writes at once: the lock is held across an await, so the
            # second one must wait — which is what binds it to this loop.
            #
            # ``return_exceptions=True`` is load-bearing here, not defensive
            # habit. A plain ``gather`` completes the moment one child raises
            # and does **not** cancel its siblings, so the caller resumes — and
            # the test closes the loop — with the other ``create_thought`` still
            # running inside the dedup critical section, holding that lock and
            # sharing the connection. Collecting every outcome makes both
            # children finish before the failure is re-raised; the assertions
            # below pin that nothing is left running and that storage holds what
            # the completed calls wrote.
            outcomes = await asyncio.gather(
                store.create_thought(
                    _thought(f"{prefix}-1", content=f"{prefix} one"), deduplicate=True
                ),
                store.create_thought(
                    _thought(f"{prefix}-2", content=f"{prefix} two"), deduplicate=True
                ),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

        first_loop = asyncio.new_event_loop()
        second_loop = asyncio.new_event_loop()
        try:
            first_loop.run_until_complete(_open_and_contend("a"))
            store = opened["store"]
            assert isinstance(store, SqliteEngravaCore)

            # The connection itself crosses loops without complaint.
            assert second_loop.run_until_complete(store.get_thought("a-1")) is not None

            # The store's lock does not.
            with pytest.raises(RuntimeError, match="bound to a different event loop"):
                second_loop.run_until_complete(_open_and_contend("b"))

            # Nothing is left running. Neither loop holds a task, so closing
            # them below cannot destroy a coroutine that still holds the dedup
            # lock or has an operation in flight on the shared connection.
            assert asyncio.all_tasks(second_loop) == set()
            assert asyncio.all_tasks(first_loop) == set()

            # ...and storage is in a stated condition rather than whatever a
            # task abandoned in flight would have left. Both calls on the first
            # loop committed; on the second only the one that took the
            # uncontended fast path did, since the other was rejected before it
            # wrote anything.
            conn = opened["conn"]
            assert isinstance(conn, aiosqlite.Connection)
            assert first_loop.run_until_complete(_thought_ids(conn)) == ["a-1", "a-2", "b-1"]
        finally:
            conn = opened.get("conn")
            if conn is not None:
                assert isinstance(conn, aiosqlite.Connection)
                first_loop.run_until_complete(conn.close())
            first_loop.close()
            second_loop.close()


# ---------------------------------------------------------------------------
# suspend_auto_commit belongs to the store, not to the task
# ---------------------------------------------------------------------------


class TestSuspendAutoCommitIsStoreWide:
    """An open transaction window captures every writer on that store instance."""

    async def test_another_tasks_write_is_rolled_back_with_the_window(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A second task's unrelated write joins the window and dies with it.

        The deferred-commit flag lives on the store instance, so a write issued
        by any task while the window is open is part of the window's
        transaction. When the window rolls back it takes that write with it —
        the other task saw no error and has no way to learn its write is gone.
        The row committed before the window survives, which is what separates
        "the rollback discarded the open transaction" from "the rollback
        discarded everything".
        """
        await store.create_thought(_thought("t-committed", content="committed earlier"))
        window_open = asyncio.Event()
        other_written = asyncio.Event()

        async def _window() -> None:
            async with store.suspend_auto_commit():
                await store.create_thought(_thought("t-window", content="the window's own row"))
                window_open.set()
                await other_written.wait()
                msg = "the window's own work failed"
                raise RuntimeError(msg)

        async def _unrelated_writer() -> None:
            await window_open.wait()
            await store.create_thought(_thought("t-other", content="an unrelated row"))
            other_written.set()

        with pytest.raises(RuntimeError, match="the window's own work failed"):
            await asyncio.gather(_window(), _unrelated_writer())

        assert await _thought_ids(db) == ["t-committed"]


# ---------------------------------------------------------------------------
# Two stores, one database file
# ---------------------------------------------------------------------------


class TestTwoStoresOneFile:
    """Nothing that orders operations inside one store reaches a second store."""

    async def test_an_edit_from_a_second_store_is_discarded_without_raising(
        self,
        two_stores: tuple[SqliteEngravaCore, SqliteEngravaCore, aiosqlite.Connection],
    ) -> None:
        """The lost update crosses the connection boundary unchanged.

        Same window as the single-store case, but now the competing write comes
        from a different connection — the shape a second process has. Between two
        processes no in-process lock could close it even in principle, which is
        why multiple stores writing one file is unsupported rather than merely
        discouraged.
        """
        store_a, store_b, conn_a = two_stores
        await store_a.create_thought(_thought())
        landed: list[str] = []

        async def _edit_from_the_second_store() -> None:
            await store_b.update_thought("t-1", essence="from the second store")
            landed.append((await _row(conn_a, "t-1"))["essence"])

        _interleave_once(store_a, "_get_thought_row", _edit_from_the_second_store)

        await store_a.update_thought("t-1", essence="from the first store")

        # Precondition: the second store's edit really committed, and the first
        # store's connection could see it.
        assert landed == ["from the second store"]
        row = await _row(conn_a, "t-1")
        assert row["essence"] == "from the first store"

    async def test_deduplication_still_inserts_a_duplicate_across_stores(
        self,
        two_stores: tuple[SqliteEngravaCore, SqliteEngravaCore, aiosqlite.Connection],
    ) -> None:
        """``deduplicate=True`` is enforced by a lock that stops at the store.

        Both stores probe the content hash, both miss, and both insert — so the
        database ends up holding two rows for byte-identical content, neither
        of them confirmation-counted, which is exactly what the flag was asked
        to prevent.
        """
        store_a, store_b, conn_a = two_stores

        async def _insert_from_the_second_store() -> None:
            await store_b.create_thought(
                _thought("t-b", content="identical content"),
                deduplicate=True,
            )

        _interleave_once(store_a, "_get_thought_by_content_hash", _insert_from_the_second_store)

        await store_a.create_thought(
            _thought("t-a", content="identical content"),
            deduplicate=True,
        )

        assert await _thought_ids(conn_a) == ["t-a", "t-b"]
        for thought_id in ("t-a", "t-b"):
            row = await _row(conn_a, thought_id)
            assert row["content"] == "identical content"
            assert row["confirmation_count"] == 0

    async def test_confirmation_counting_survives_a_second_store(
        self,
        two_stores: tuple[SqliteEngravaCore, SqliteEngravaCore, aiosqlite.Connection],
    ) -> None:
        """A dedup hit counts relative to storage, so neither sighting is lost.

        The guarantee that does cross the connection boundary: the bump is
        ``confirmation_count = confirmation_count + 1`` evaluated by SQLite, not
        an absolute value derived from a read. Two stores confirming the same
        content therefore both count.
        """
        store_a, store_b, conn_a = two_stores
        await store_a.create_thought(_thought(content="identical content"))

        async def _confirm_from_the_second_store() -> None:
            await store_b.create_thought(
                _thought("t-b", content="identical content"),
                deduplicate=True,
            )

        _interleave_once(store_a, "_get_thought_by_content_hash", _confirm_from_the_second_store)

        await store_a.create_thought(
            _thought("t-a", content="identical content"),
            deduplicate=True,
        )

        assert await _thought_ids(conn_a) == ["t-1"]
        row = await _row(conn_a, "t-1")
        assert row["confirmation_count"] == 2

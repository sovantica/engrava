"""The concurrency contract the documentation states — guarantees and non-guarantees.

``docs/concurrency.md`` describes what one store promises concurrent callers and,
just as importantly, what it does **not** promise. A promise nobody tests drifts;
a *withheld* promise drifts just as quietly, because nothing fails when the docs
keep claiming safety the code stopped providing. Every claim on that page that
can be exercised in-process is pinned here, in both directions:

* **Guarantees** — an edit writes only the columns it owns, so two *sequential*
  edits to different fields both survive; a confirmation bump is relative, so
  it survives a second connection; the in-process task-reentrant write lock
  (see ``TestInProcessCriticalSection`` below) makes the read, the validation,
  and the write of ``update_thought`` / ``restore_thought`` / ``update_edge`` /
  ``update_action`` one critical section **across genuinely concurrent tasks**
  on this instance; ``suspend_auto_commit()`` genuinely excludes every other
  task's guarded write for its duration; and, as of the ``revision`` guard
  (stage 3 — this module's newest layer), **every guarded update's row-version
  check is enforced in the database itself**, so it now also catches a
  competing write from a *different store on the same file* — including a
  second process, per ``TestTwoStoresOneFile`` below — not only a different
  task on this instance. ``update_edge`` and ``update_action`` carry this
  guard for the first time; before this stage neither could ever raise a
  staleness error.
* **Non-guarantees** — same-field edits still resolve last-write-wins when
  nothing else moved the row's ``revision`` in between (the correct outcome
  for two genuine, sequential edits); and a same-*task* nested call reached
  through a caller-owned hook, not a second ``asyncio`` task, is still
  unaffected by the task-scoped write lock — but the ``revision`` guard now
  catches it where the old ``updated_cycle`` guard could not, because
  ``revision`` moves on *every* guarded write to a row, whatever field it
  touched, with no caller action required to arm it. ``TestOneStoreManyTasks``
  below still uses the single-task ``_interleave_once`` stand-in for that
  same-task shape; what used to land silently (or land in a state the domain
  model forbids) now raises ``StaleDataError`` before anything is written.
* **The dedup probe-and-insert window (pre-existing, stage 1)** — the
  content-hash window (``create_thought(deduplicate=True)``, ``get_or_create``,
  ``upsert_by_hash``) orders across a second store on the same file: it opens
  with ``BEGIN IMMEDIATE``, so a second store reaching the same window while it
  is open cannot even start its own transaction. It waits out ordinary
  contention and, only past a bounded number of retries, raises
  ``WriteContentionError`` instead of racing the probe. This is independent of
  the ``revision`` guard — it orders the probe itself, not a later guarded
  update.

Three interleaving techniques are used, deliberately kept apart:

* ``TestOneStoreManyTasks`` / ``TestEdgeAndActionUpdatesCarryARevisionGuard`` /
  ``TestStateMachineChecksUseTheStateThisCallRead`` reuse the one-shot seam from
  ``test_partial_field_updates`` (``_interleave_once``) to run a competing
  operation **inline, on the same task**, at the exact point between an
  operation's read and its write. This is a same-task TOCTOU stand-in, not a
  second ``asyncio`` task — the write lock's task-reentrancy deliberately
  does not block it (that is what makes a write issued from inside the
  caller's own ``suspend_auto_commit`` complete instead of deadlocking) — but
  the ``revision`` guard now rejects it, where before this stage it could only
  ever land (silently, or in a forbidden composite state).
* ``TestInProcessCriticalSection`` and ``TestSuspendAutoCommitIsStoreWide`` use
  genuinely separate ``asyncio.Task`` objects, paused at a precise point via
  :class:`asyncio.Event` handshakes — these are the cases the task-reentrant
  write lock actually changes.
* ``TestTwoStoresOneFile`` uses two independent store instances sharing one
  database file to exercise the guard **across connections** — the shape a
  second process has. This is the class this stage exists for: before it, a
  second store's edit was lost with no error at all; now it is rejected.

Between the tasks of one test, ordering comes from :class:`asyncio.Event`
handshakes and from the FIFO ordering ``asyncio`` guarantees for
``call_soon``/``Event.set()`` callbacks, apart from the four tests below, which
depend on a time bound:

* ``test_a_child_task_awaited_inside_the_window_raises_instead_of_hanging``
  sets the write lock's acquire bound to 0.05 s and expects it to expire.
* ``test_a_slow_legitimate_batch_embed_does_not_time_out_a_waiting_writer``
  simulates a 0.2 s embedding call and passes only if the writer waiting behind
  it acquires the lock inside the 2 s bound the test sets.
* ``test_deduplication_across_stores_raises_instead_of_duplicating`` and
  ``test_confirmation_counting_across_stores_now_raises_instead_of_racing``
  set the second store's SQLite busy timeout to 20 ms and expect its attempts
  to run out.

``TestWriteLockAcquireBoundDefault`` pins the production default of the write
lock's acquire bound, which the first two tests override.
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
    DeriveGates,
    EdgeRecord,
    EdgeType,
    InvalidTransitionError,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    StaleDataError,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
    WriteContentionError,
    WriteLockTimeoutError,
)
from engrava.infrastructure.sqlite.engrava_core import _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS
from tests.test_derived_records_seam import ListProducer, _child, _source
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


def _pause_after_first_call(
    store: SqliteEngravaCore,
    method_name: str,
) -> tuple[asyncio.Event, asyncio.Event]:
    """Patch ``method_name`` so its *first* call pauses after returning.

    Unlike ``_interleave_once`` (which runs the competing operation inline, on
    the calling task, and is the right tool for the same-task TOCTOU stand-in
    used elsewhere in this module), this parks the **calling task itself** —
    still holding whatever lock it acquired to get this far — right after the
    wrapped method's first call returns, until a second, genuinely separate
    task releases it. That makes it possible to launch a real concurrent
    ``asyncio.Task`` while the first task is provably still inside its own
    critical section, and to observe what that second task can and cannot do
    while it waits.

    Args:
        store: The store whose method is patched (in place, on the instance).
        method_name: Name of the (async, no-side-effect-on-args) method to
            pause after its first invocation. Typically a read such as
            ``_get_thought_row``.

    Returns:
        A ``(paused, resume)`` pair: ``paused`` is set once the first call has
        returned its result and is about to block; the caller must ``.set()``
        ``resume`` to let it continue.

    """
    original = getattr(store, method_name)
    paused = asyncio.Event()
    resume = asyncio.Event()
    fired = False

    async def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal fired
        result = await original(*args, **kwargs)
        if not fired:
            fired = True
            paused.set()
            await resume.wait()
        return result

    setattr(store, method_name, wrapper)
    return paused, resume


def _write_lock_is_held(store: SqliteEngravaCore) -> bool | None:
    """Return whether ``store._write_lock`` is held, or ``None`` if it does not exist.

    ``None`` on a store that has no ``_write_lock`` attribute, so a
    caller can skip the check gracefully instead of raising ``AttributeError``
    partway through an ``asyncio.Event`` handshake — which would strand the
    paused task forever, waiting on a ``resume`` event nothing would go on to
    set.
    """
    write_lock = getattr(store, "_write_lock", None)
    if write_lock is None:
        return None
    return bool(write_lock._lock.locked())


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

    async def test_interleaved_edits_to_different_fields_now_reject_the_second(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Different columns no longer save an interleaved edit from rejection.

        Before the ``revision`` guard, an update writing only the columns it
        owns meant a competing edit to a *different* field survived even when
        interleaved this tightly. Now ``revision`` moves on **every** guarded
        write, whatever field it touched, so this call's guard — captured
        against the row it read, before the competing edit landed — no longer
        matches. The whole update is rejected, not merged: even the essence
        field this call owns is not written.
        """
        await store.create_thought(_thought())
        landed: list[str] = []

        async def _competing_edit() -> None:
            await store.update_thought("t-1", priority=Priority.P1)
            landed.append((await _row(db, "t-1"))["priority"])

        _interleave_once(store, "_get_thought_row", _competing_edit)

        with pytest.raises(StaleDataError):
            await store.update_thought("t-1", essence="mine")

        # Precondition: the competing edit really reached storage first.
        assert landed == [Priority.P1.value]
        row = await _row(db, "t-1")
        assert row["essence"] == "essence"
        assert row["priority"] == Priority.P1.value

    async def test_interleaved_edits_to_one_field_now_reject_the_second(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A competing edit to the same field is caught, not silently discarded.

        Before the ``revision`` guard, ``update_thought`` read the row, evolved
        it in memory, then wrote — and aiosqlite serialises *statements*, not
        method bodies, so a second task's whole update could land in that
        window and simply be overwritten, with nothing in the row to show it
        ever happened. Now the second call's guard no longer matches once the
        first has bumped ``revision``, so it raises ``StaleDataError`` instead
        of silently winning.
        """
        await store.create_thought(_thought())
        landed: list[str] = []

        async def _competing_edit() -> None:
            await store.update_thought("t-1", essence="from the other task")
            landed.append((await _row(db, "t-1"))["essence"])

        _interleave_once(store, "_get_thought_row", _competing_edit)

        with pytest.raises(StaleDataError):
            await store.update_thought("t-1", essence="from this task")

        # Precondition: the competing edit really reached storage first, so the
        # assertion below is about it surviving unclobbered, not about it never
        # having happened.
        assert landed == ["from the other task"]
        row = await _row(db, "t-1")
        assert row["essence"] == "from the other task"

    async def test_a_competing_cycle_stamp_rejects_an_edit_to_a_different_field(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The revision guard is on every update, so it rejects unrelated edits too.

        This is one instance of the general rule pinned by the two tests
        above: the competing writer here touches **only** ``updated_cycle``;
        this call touches only ``essence``. They share no column — and the
        edit is still rejected, because ``revision`` bumps on every guarded
        write regardless of which columns it touches. Nothing of the rejected
        update reaches storage.
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

    async def test_upsert_by_hash_now_raises_on_a_competing_edit(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The hash probe and the in-place update are not one atomic step.

        ``upsert_by_hash`` documents ``StaleDataError`` for a row modified
        between its probe and its update. It inherits ``update_thought``'s
        guard, and that guard now enforces: the competing edit bumps
        ``revision`` between the probe and the upsert's own guarded write, so
        the upsert's write matches no row and raises — the competing edit's
        value survives untouched rather than being silently overwritten.
        """
        await store.create_thought(_thought(content="shared content"))
        landed: list[str] = []

        async def _competing_edit() -> None:
            await store.update_thought("t-1", essence="from the other task")
            landed.append((await _row(db, "t-1"))["essence"])

        _interleave_once(store, "_get_thought_row", _competing_edit)

        with pytest.raises(StaleDataError):
            await store.upsert_by_hash(
                _thought("t-unused", essence="from the upsert", content="shared content"),
            )

        # Precondition: the competing edit really reached storage first.
        assert landed == ["from the other task"]
        row = await _row(db, "t-1")
        assert row["essence"] == "from the other task"
        assert await _thought_ids(db) == ["t-1"]


class TestEdgeAndActionUpdatesCarryARevisionGuard:
    """The other update paths now have the same guard update_thought does.

    Before this stage, ``update_edge`` and ``update_action`` were keyed on id
    alone and could never raise a staleness error — a competing edit was
    discarded with nothing that could ever have flagged it. They now carry the
    same ``revision`` guard :class:`update_thought` does.
    """

    async def test_interleaved_edge_edits_to_one_field_now_reject_the_second(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """``update_edge``'s guard rejects a competing edit instead of losing it."""
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

        with pytest.raises(StaleDataError):
            await store.update_edge("e-1", weight=0.1)

        # Precondition: the competing edit really reached storage first.
        assert landed == [0.9]
        cursor = await db.execute("SELECT weight FROM edge WHERE edge_id = 'e-1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row["weight"] == 0.9


class TestStateMachineChecksUseTheStateThisCallRead:
    """A lifecycle check that already passed is not re-checked against storage.

    Before the ``revision`` guard, two legal transitions read against a stale
    snapshot could compose into a state the machine forbids — this class used
    to demonstrate exactly that. Now the guarded write's ``revision`` check
    catches the second call before its already-validated transition can land,
    so the forbidden composite state can no longer be reached this way.
    """

    async def test_interleaved_lifecycle_moves_now_reject_before_landing(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The second, stale-validated write is rejected, not merely mis-validated.

        ``update_thought`` still validates the transition against the record
        *it* read — a competing writer moving the row in between does not
        invalidate that in-memory check. What is new is the guarded write
        itself: ``ACTIVE -> DONE``, validated against the pre-interleave
        ``ACTIVE`` snapshot, would have landed on a row already moved to
        ``ARCHIVED`` (and ``ARCHIVED -> DONE`` is not an allowed edge) — but
        the guarded ``UPDATE`` now matches no row, since ``revision`` moved
        when the competing archive landed, so nothing is written and
        ``StaleDataError`` is raised instead.
        """
        await store.create_thought(_thought())
        landed: list[str] = []

        async def _competing_archive() -> None:
            await store.update_thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED)
            landed.append((await _row(db, "t-1"))["lifecycle_status"])

        _interleave_once(store, "_get_thought_row", _competing_archive)

        with pytest.raises(StaleDataError):
            await store.update_thought("t-1", lifecycle_status=LifecycleStatus.DONE)

        # Precondition: the row really was ARCHIVED when the second write landed.
        assert landed == [LifecycleStatus.ARCHIVED.value]
        row = await _row(db, "t-1")
        assert row["lifecycle_status"] == LifecycleStatus.ARCHIVED.value
        # ...and that edge is not one the state machine would have allowed,
        # which is exactly why the rejection matters here.
        assert not LifecycleStatus.ARCHIVED.can_transition_to(LifecycleStatus.DONE)

    async def test_interleaved_action_moves_now_reject_before_landing(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The same fix in the action state machine.

        ``PLANNED -> BLOCKED`` is validated against the read state, which
        would have landed on a row another writer already moved to
        ``EXECUTING`` (``EXECUTING -> BLOCKED`` is not an allowed edge) — but
        ``update_action``'s own ``revision`` guard now rejects the write
        first.
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

        _interleave_once(store, "_get_action_row", _competing_start)

        with pytest.raises(StaleDataError):
            await store.update_action("a-1", status=ActionStatus.BLOCKED)

        # Precondition: the row really was EXECUTING when the second write landed.
        assert landed == [ActionStatus.EXECUTING.value]
        cursor = await db.execute("SELECT status FROM action WHERE action_id = 'a-1'")
        row = await cursor.fetchone()
        assert row is not None
        assert row["status"] == ActionStatus.EXECUTING.value
        # ...and that edge is not one the state machine would have allowed,
        # which is exactly why the rejection matters here.
        assert not ActionStatus.EXECUTING.can_transition_to(ActionStatus.BLOCKED)


# ---------------------------------------------------------------------------
# An in-process task-reentrant lock across two real tasks
# ---------------------------------------------------------------------------


class TestInProcessCriticalSection:
    """A task-reentrant ``_write_lock`` makes the read-validate-write span atomic.

    Every test here launches a **second, genuine ``asyncio.Task``** while the
    first is paused (via ``_pause_after_first_call``) between its own read and
    its own write — not the same-task ``_interleave_once`` stand-in used
    elsewhere in this module. Before this fix, nothing stopped the second
    task's own read from racing ahead of the first task's not-yet-committed
    write; now the second task cannot even start its own read until the first
    task's entire critical section — read, validate, write, read-back, journal,
    commit — has completed, because both go through the same
    ``SqliteEngravaCore._write_lock``.
    """

    async def test_a_second_tasks_read_now_sees_the_firsts_committed_write(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Two tasks editing the **same field**: the second's read is never stale.

        The discriminating fact is not the row's *final* contents — last write
        wins is the correct outcome for two genuine, independent edits, fix or
        no fix — it is what task B's own internal read (inside its own
        ``update_thought`` call) observed. Before this fix, nothing stopped
        B's read from racing ahead of task A's not-yet-committed write, so B
        could read the *pre-A* essence even though A's write was already
        underway. Now B's call cannot even begin its own read until A's write
        lock is released, which happens only once A's write is durable — so
        B's read always sees A's committed essence, never the stale one.

        Every ``_get_thought_row`` call's essence is recorded, in order, to
        make that observable. The order is deterministic, not a scheduling
        gamble: task creation via ``asyncio.ensure_future`` schedules the new
        task's first step with ``call_soon``, and ``resume.set()`` schedules
        the paused task's continuation the same way — both calls happen
        inside ``_task_b``, in that fixed order, so ``asyncio``'s FIFO
        ``call_soon`` ordering guarantees B's task starts running (up to its
        own first suspension point, inside its own read) before A's
        continuation runs.
        """
        await store.create_thought(_thought())
        reads: list[str] = []
        paused = asyncio.Event()
        resume = asyncio.Event()
        original_get_thought_row = store._get_thought_row
        fired = False

        async def _tracking_get_thought_row(thought_id: str) -> object:
            nonlocal fired
            row = await original_get_thought_row(thought_id)
            if row is not None:
                reads.append(row["essence"])
            if not fired:
                fired = True
                paused.set()
                await resume.wait()
            return row

        setattr(store, "_get_thought_row", _tracking_get_thought_row)  # noqa: B010 -- mypy rejects a direct method-assign here

        async def _task_a() -> ThoughtRecord:
            return await store.update_thought("t-1", essence="from A")

        async def _task_b() -> ThoughtRecord:
            await paused.wait()
            # A is parked right after its own read, still holding the write
            # lock for the rest of its critical section (`_write_lock_is_held`
            # returns None when the store has no `_write_lock`; the check is
            # then skipped rather than raised, which would strand `_task_a` on
            # `resume.wait()` -- see its docstring).
            held = _write_lock_is_held(store)
            if held is not None:
                assert held, (
                    "task A must still hold _write_lock while paused between "
                    "its own read and its own write"
                )
            b_task = asyncio.ensure_future(store.update_thought("t-1", essence="from B"))
            resume.set()
            return await b_task

        a_result, b_result = await asyncio.gather(_task_a(), _task_b())

        # reads[0] is A's own initial read (the original, pre-edit essence);
        # reads[1] is B's own initial read -- the one this test is about.
        assert reads[0] == "essence"
        assert reads[1] == "from A", (
            "task B's own read must see task A's already-committed essence, "
            f"not a stale value -- full read order was {reads!r}"
        )
        assert a_result.essence == "from A"
        assert b_result.essence == "from B"
        row = await _row(db, "t-1")
        assert row["essence"] == "from B"

    async def test_a_committed_cycle_stamp_no_longer_spuriously_rejects_a_pending_edit(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Row 1 of the summary table: either task stamping ``updated_cycle``.

        Before this fix, task B's guard was captured from a read taken before
        task A's cycle stamp committed, so B's unrelated edit was rejected in
        full with ``StaleDataError`` the moment A's stamp landed — even though
        B never touched the cycle. Now B's read cannot happen until A's stamp
        is already durable, so B's guard is captured against the *post-A*
        value and matches when B writes.
        """
        await store.create_thought(_thought())
        paused, resume = _pause_after_first_call(store, "_get_thought_row")

        async def _task_a() -> ThoughtRecord:
            return await store.update_thought("t-1", updated_cycle=5)

        async def _task_b() -> ThoughtRecord:
            await paused.wait()
            held = _write_lock_is_held(store)
            if held is not None:
                assert held
            b_task = asyncio.ensure_future(store.update_thought("t-1", essence="from B"))
            resume.set()
            return await b_task

        a_result, b_result = await asyncio.gather(_task_a(), _task_b())

        assert a_result.updated_cycle == 5
        assert b_result.essence == "from B"
        # B's own guard picked up A's already-committed cycle stamp.
        assert b_result.updated_cycle == 5
        row = await _row(db, "t-1")
        assert row["updated_cycle"] == 5
        assert row["essence"] == "from B"

    async def test_a_committed_lifecycle_move_is_what_the_next_transition_validates_against(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Row 3 of the summary table: two tasks moving one row's state machine.

        Before this fix, task B's ``ACTIVE -> DONE`` move was validated
        against the ``ACTIVE`` row B itself read before task A's
        ``ACTIVE -> ARCHIVED`` move committed, so both writes landed and the
        row ended up ``DONE`` — a transition (``ARCHIVED -> DONE``) the state
        machine does not allow as a single step. Now B's read cannot happen
        until A's move is durable, so B's transition is validated against the
        row A actually left behind (``ARCHIVED``) and is correctly rejected.
        """
        await store.create_thought(_thought())
        paused, resume = _pause_after_first_call(store, "_get_thought_row")

        async def _task_a() -> ThoughtRecord:
            return await store.update_thought("t-1", lifecycle_status=LifecycleStatus.ARCHIVED)

        async def _task_b() -> ThoughtRecord:
            await paused.wait()
            held = _write_lock_is_held(store)
            if held is not None:
                assert held
            b_task = asyncio.ensure_future(
                store.update_thought("t-1", lifecycle_status=LifecycleStatus.DONE)
            )
            resume.set()
            return await b_task

        outcomes = await asyncio.gather(_task_a(), _task_b(), return_exceptions=True)
        a_outcome, b_outcome = outcomes
        assert not isinstance(a_outcome, BaseException)
        assert isinstance(b_outcome, InvalidTransitionError)

        # A's legal move landed; B's illegal composite move never reached storage.
        row = await _row(db, "t-1")
        assert row["lifecycle_status"] == LifecycleStatus.ARCHIVED.value
        assert not LifecycleStatus.ARCHIVED.can_transition_to(LifecycleStatus.DONE)

    async def test_two_tasks_editing_different_fields_still_survive_each_other(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Requirement 3: same-field safety must not become a bottleneck.

        Two genuinely concurrent tasks editing **different** fields of the
        same row must both still land — the lock serialises the two critical
        sections (one runs, then the other), it does not reject or drop
        either edit. This is the guard against "buying same-field safety by
        serialising everything into a queue that breaks unrelated edits": it
        does not — different-field edits still both succeed, exactly as
        documented, and the serialisation is just ordering, not loss.
        """
        await store.create_thought(_thought())
        paused, resume = _pause_after_first_call(store, "_get_thought_row")

        async def _task_a() -> ThoughtRecord:
            return await store.update_thought("t-1", essence="from A")

        async def _task_b() -> ThoughtRecord:
            await paused.wait()
            b_task = asyncio.ensure_future(store.update_thought("t-1", priority=Priority.P1))
            resume.set()
            return await b_task

        a_result, b_result = await asyncio.gather(_task_a(), _task_b())

        assert a_result.essence == "from A"
        assert b_result.priority == Priority.P1
        row = await _row(db, "t-1")
        # Both edits landed: neither serialised call clobbered the other's column.
        assert row["essence"] == "from A"
        assert row["priority"] == Priority.P1.value

    async def test_a_write_issued_inside_suspend_auto_commit_completes(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Re-entrancy: a write inside the caller's own window does not deadlock.

        The risk the spec calls out explicitly: getting re-entrancy wrong
        "converts a silent data loss into a hang". A plain (non-reentrant)
        lock around ``suspend_auto_commit`` would deadlock the moment the
        caller issues a write of its own from inside the window, because that
        write also acquires ``_write_lock``. The lock is task-reentrant, so
        this must complete instead.
        """
        async with store.suspend_auto_commit():
            created = await store.create_thought(_thought("t-inside", content="written inside"))
            updated = await store.update_thought("t-inside", essence="edited inside the window")

        assert created.thought_id == "t-inside"
        assert updated.essence == "edited inside the window"
        row = await _row(db, "t-inside")
        assert row["essence"] == "edited inside the window"

    async def test_a_child_task_awaited_inside_the_window_raises_instead_of_hanging(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Contract violation: spawning and awaiting a writer from inside the window.

        Re-entrancy is keyed on the *task*, not the call: a task the window's
        own task spawns (``asyncio.create_task``, ``gather``, ``wait_for``, an
        ``on_store`` hook or embedding provider callback that does the same)
        is a different task, so it cannot reuse the window's re-entrant grant.
        If the window's own task then awaits that child before its window
        closes, neither can make progress: the child can never get the lock
        the parent holds, and the parent can never release it while still
        awaiting the child. This is already out of the documented contract —
        drive every write inside an open ``suspend_auto_commit`` window from
        the one task that opened it — so the failure mode this test pins is
        not "this becomes supported", it is "violating it now raises a typed,
        catchable error instead of hanging the process forever with nothing
        in any log to explain why".

        ``_acquire_timeout_seconds`` is set short here purely so the test
        does not wait out the real, generous production bound; the mechanism
        is identical either way — a real deadlock never resolves on its own,
        so any positive bound eventually fires.
        """
        store._write_lock._acquire_timeout_seconds = 0.05

        async def _window() -> None:
            async with store.suspend_auto_commit():
                await store.create_thought(_thought("t-window", content="the window's own row"))
                # A task the window's own task spawns and then awaits before
                # the window closes -- the exact deadlock shape. The child
                # can never acquire `_write_lock` (the parent holds it and
                # is not the same task), and the parent cannot finish (and
                # release it) while still awaiting the child.
                child = asyncio.create_task(
                    store.create_thought(_thought("t-child", content="from a spawned task"))
                )
                await child

        with pytest.raises(WriteLockTimeoutError):
            await _window()

        # The deadlock ending in a raise, not a hang, is also what lets the
        # window's own task resume and roll back cleanly: nothing from this
        # attempt is left durable, and the store is still usable afterward.
        assert await _thought_ids(db) == []
        recovered = await store.create_thought(_thought("t-after", content="store still works"))
        assert recovered.thought_id == "t-after"

    async def test_a_slow_legitimate_batch_embed_does_not_time_out_a_waiting_writer(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The acquire bound must survive real, network-bound embedding latency.

        ``bulk_store``'s batch embedding call runs *inside*
        ``suspend_auto_commit``'s window, which holds ``_write_lock`` for its
        whole duration — so the lock's acquire bound has to clear that
        legitimate, genuinely slow round trip, not just "ordinary" contention.
        This is the case the fix exists to protect: an unrelated task's
        write must still succeed after waiting
        out a slow-but-real embedding call, never time out on it. The
        embedding provider below is deliberately slower than the store's
        configured bound would be *if it were sized wrong* (a plain
        ``asyncio.sleep`` standing in for real network latency); the bound
        configured here is sized to clear it, exactly as the derivation in
        ``_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS`` intends for the real default.

        The final assertions alone would not discriminate: with ``_write_lock``
        removed entirely, the unrelated writer would run immediately (never
        blocked), finish long before the batch, and every assertion on the
        final state would still pass — the lock could be doing nothing and
        this test would not notice. ``order`` records the real sequence
        instead: ``_db.commit`` is patched to log the moment the batch's
        window actually closes (which is the only thing that releases
        ``_write_lock`` here), and ``_get_thought_row`` is patched to log the
        writer's own existence-check read for ``t-other`` specifically. The
        assertion below requires the commit to precede that read, which is
        only possible if the writer was genuinely blocked until the batch's
        window closed.
        """

        class _SlowProvider:
            """A provider whose batch call is slow but eventually succeeds."""

            dimension = 3
            model_name = "slow-test-model"

            def __init__(self, embed_started: asyncio.Event, delay_seconds: float) -> None:
                self._embed_started = embed_started
                self._delay_seconds = delay_seconds

            async def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                self._embed_started.set()
                await asyncio.sleep(self._delay_seconds)
                return [[0.1, 0.2, 0.3] for _ in texts]

        embed_started = asyncio.Event()
        embed_delay_seconds = 0.2
        slow_store = SqliteEngravaCore(
            db,
            embedding_provider=_SlowProvider(embed_started, embed_delay_seconds),
            auto_embed=True,
            # Comfortably above the artificial delay above, standing in for a
            # bound correctly sized to the provider's real worst case -- the
            # point under test is that clearing it lets the waiting writer
            # through rather than timing out.
            write_lock_acquire_timeout_seconds=embed_delay_seconds * 10,
        )

        order: list[str] = []
        original_commit = slow_store._db.commit

        async def _tracking_commit() -> None:
            order.append("bulk-commit")
            await original_commit()

        setattr(slow_store._db, "commit", _tracking_commit)  # noqa: B010 -- mypy rejects a direct method-assign here

        original_get_thought_row = slow_store._get_thought_row

        async def _tracking_get_thought_row(thought_id: str) -> object:
            if thought_id == "t-other":
                order.append("writer-read")
            return await original_get_thought_row(thought_id)

        setattr(slow_store, "_get_thought_row", _tracking_get_thought_row)  # noqa: B010 -- mypy rejects a direct method-assign here

        async def _bulk() -> list[ThoughtRecord]:
            return await slow_store.bulk_store([_thought("t-bulk-1", content="batch content")])

        async def _unrelated_writer() -> ThoughtRecord:
            await embed_started.wait()
            # The batch embedding call is now in flight, holding `_write_lock`
            # for the whole `suspend_auto_commit` window. This call must wait
            # out the artificial delay above and still succeed.
            return await slow_store.create_thought(
                _thought("t-other", content="unrelated content"),
            )

        bulk_result, writer_result = await asyncio.gather(_bulk(), _unrelated_writer())

        assert bulk_result[0].thought_id == "t-bulk-1"
        assert writer_result.thought_id == "t-other"
        assert set(await _thought_ids(db)) == {"t-bulk-1", "t-other"}
        # The writer's own create_thought call (auto_embed=True on this
        # store too) contributes further commits of its own after its read,
        # so the exact count is not asserted -- only that at least one
        # commit (the batch's) precedes the writer's read, and that the
        # read itself is not the very first event.
        assert "bulk-commit" in order, f"expected a batch commit to be observed -- got {order!r}"
        assert "writer-read" in order, (
            f"expected the writer's own read to be observed -- got {order!r}"
        )
        assert order.index("writer-read") > order.index("bulk-commit"), (
            "the unrelated writer's own read must be provably ordered after "
            f"the batch's window released the lock -- got {order!r}"
        )

    async def test_nested_suspend_auto_commit_inner_exit_does_not_commit_early(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Nesting failure 1: a clean inner exit must not commit the outer window.

        ``_skip_auto_commit`` used to be a plain ``bool``: a nested
        ``suspend_auto_commit``'s ``else`` branch committed whenever
        ``self._db.in_transaction`` was true, with no regard for whether an
        *enclosing* window was still open. So the inner block's own clean exit
        committed the outer window's transaction early — durably, before the
        outer block had decided anything. Only the outermost call may commit;
        this reproduces the bug by raising *after* the inner block exits and
        checking that the eventual rollback still discards everything,
        including what the inner block wrote.
        """

        async def _run() -> None:
            async with store.suspend_auto_commit():
                await store.create_thought(_thought("t-outer", content="outer, before nesting"))
                async with store.suspend_auto_commit():
                    await store.create_thought(_thought("t-inner", content="inner"))
                # The inner block has already exited cleanly here. If its exit
                # committed early (the bug), the two rows above are already
                # durable and the rollback below cannot undo them.
                msg = "abort after the nested block"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="abort after the nested block"):
            await _run()

        assert await _thought_ids(db) == []

    async def test_nested_suspend_auto_commit_finally_does_not_resume_autocommit(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Nesting failure 2: an inner exit must not resume per-call autocommit.

        The companion bug in the same ``bool``: the inner block's ``finally``
        cleared ``_skip_auto_commit`` outright on exit, so per-call autocommit
        resumed for the rest of the *outer* block even though that outer
        window was still open. A write issued after the nested block then
        committed on its own, immune to the outer window's eventual rollback.
        This reproduces it with a write placed *after* the nested block:
        under the bug it survives on its own; fixed, it is still inside the
        outer window and rolls back with everything else.
        """

        async def _run() -> None:
            async with store.suspend_auto_commit():
                await store.create_thought(_thought("t-outer-1", content="before nesting"))
                async with store.suspend_auto_commit():
                    await store.create_thought(_thought("t-inner", content="inner"))
                # If the inner block's `finally` resumed per-call autocommit
                # (the bug), this write commits on its own, right here.
                await store.create_thought(_thought("t-outer-2", content="after nesting"))
                msg = "abort the whole outer window"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="abort the whole outer window"):
            await _run()

        assert await _thought_ids(db) == []

    async def test_dedup_probe_rollback_no_longer_discards_another_tasks_write(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Investigation: does ``_write_lock`` restore the withdrawn cross-task guarantee?

        ``_serialize_dedup_probe``'s docstring names an unclosed gap directly:
        "a genuinely concurrent, unrelated task's write riding along with this
        call's own commit ... [rolled] back along with this call's own work if
        this call fails" — and states closing it "needs a task-reentrant lock
        around every write path on the instance, which does not exist yet."
        This test exercises exactly that: task A opens the dedup
        probe-and-insert window (``create_thought(deduplicate=True)``) and is
        driven to fail *inside* it — by a natural id collision on the insert,
        not a monkeypatch — forcing ``_serialize_dedup_probe``'s own rollback.
        Task B's plain, unrelated write is proven not to ride along: it cannot
        even start until A's whole window (including A's own rollback) has
        closed, so it survives regardless of A's failure.
        """
        # A pre-existing row at the id A's dedup insert will collide on, with
        # different content so the hash probe misses and the collision is
        # only discovered once `_insert_new_thought_row` runs, inside the
        # already-open `BEGIN IMMEDIATE` window.
        await store.create_thought(_thought("t-a", content="pre-existing, different content"))
        paused, resume = _pause_after_first_call(store, "_get_thought_by_content_hash")

        async def _task_a() -> None:
            with pytest.raises(ValueError, match="Thought already exists"):
                await store.create_thought(
                    _thought("t-a", content="dedup content that misses the probe"),
                    deduplicate=True,
                )

        async def _task_b() -> ThoughtRecord:
            await paused.wait()
            held = _write_lock_is_held(store)
            if held is not None:
                assert held, "task A must still hold _write_lock inside the dedup window"
            b_task = asyncio.ensure_future(
                store.create_thought(_thought("t-b", content="unrelated content"))
            )
            resume.set()
            return await b_task

        _, b_result = await asyncio.gather(_task_a(), _task_b())

        assert b_result.thought_id == "t-b"
        # A's failed dedup insert contributed nothing; B's unrelated write is
        # unaffected by A's rollback — both the pre-existing row and B's row
        # are present, and nothing was discarded that should not have been.
        assert set(await _thought_ids(db)) == {"t-a", "t-b"}

    async def test_cleanup_expired_read_is_now_inside_its_own_critical_section(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """The candidate SELECT now runs under ``_write_lock``, not before it.

        Before this fix, ``cleanup_expired`` read the candidate ids and only
        *then* acquired ``_write_lock`` for the writes that act on them. A
        concurrent task extending a thought's ``expires_at`` — or a suspended
        transaction exposing a value it then rolls back — could land in that
        gap, and this call would still archive or delete a row that was no
        longer actually expired by the time it acted: caller data loss, not
        merely a stale read. Moving the read inside the same lock acquisition
        as the writes closes the gap; this pins that the read itself now runs
        while the lock is held, not just the writes that follow it.
        """
        await store.create_thought(
            _thought("t-expired", content="already expired"),
            expires_after_seconds=-100,
        )
        select_prefix = "SELECT thought_id FROM thought WHERE expires_at"
        original_execute = store._db.execute
        observed_lock_state: list[bool | None] = []

        async def _tracking_execute(sql: str, *args: object, **kwargs: object) -> object:
            if sql.strip().startswith(select_prefix):
                observed_lock_state.append(_write_lock_is_held(store))
            return await original_execute(sql, *args, **kwargs)

        setattr(store._db, "execute", _tracking_execute)  # noqa: B010 -- mypy rejects a direct method-assign here

        result = await store.cleanup_expired()

        assert result.expired_count == 1
        assert observed_lock_state == [True], (
            "the candidate SELECT must run while _write_lock is held, not before "
            f"it is acquired -- observed {observed_lock_state!r}"
        )

    async def test_derived_child_compensation_stays_under_the_same_lock(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """A failed derived-child insert's unwind must not release the lock first.

        ``_insert_derived_row`` wraps its insert and journal append in its own
        ``_write_readback_savepoint`` unit, and that whole call — attempt and,
        on failure, the unit's own ``ROLLBACK TO`` / ``RELEASE`` unwind — runs
        inside one ``async with self._write_lock:`` acquisition. If the unwind
        instead ran under a *fresh* acquisition taken after the failing
        attempt's own lock had already released, a different, waiting task
        could acquire the lock in that gap, join the still-open transaction,
        and have its own successful write discarded once the unwind finally
        ran — the exact cross-task exposure the write lock exists to close.
        This pins that a waiting task's write still cannot land in that gap.
        """
        producer = ListProducer([_child("derived content")])
        store_with_producer = SqliteEngravaCore(
            db,
            hooks=producer,
            derive_gates=DeriveGates(enabled=False, on_error="raise"),
            journal_enabled=True,
        )
        await store_with_producer.create_thought(_source("src-1"), deduplicate=False)

        paused = asyncio.Event()
        resume = asyncio.Event()
        assert store_with_producer.journal is not None
        original_append = store_with_producer.journal.append
        call_count = 0

        async def _failing_append(
            mutation_type: str, target_id: str | None, delta: dict[str, object]
        ) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                paused.set()
                await resume.wait()
                msg = "forced journal failure for the derived child's own insert"
                raise RuntimeError(msg)
            return await original_append(mutation_type, target_id, delta)

        setattr(store_with_producer.journal, "append", _failing_append)  # noqa: B010 -- mypy rejects a direct method-assign here

        async def _derive() -> None:
            with pytest.raises(RuntimeError, match="forced journal failure"):
                await store_with_producer.derive_existing("src-1")

        async def _unrelated_writer() -> ThoughtRecord:
            await paused.wait()
            held = _write_lock_is_held(store_with_producer)
            if held is not None:
                assert held, (
                    "the derived child's insert unit must still hold _write_lock "
                    "while it unwinds, when a different task tries to acquire it"
                )
            writer_task = asyncio.ensure_future(
                store_with_producer.create_thought(
                    _thought("t-other", content="unrelated content"),
                )
            )
            resume.set()
            return await writer_task

        _, writer_result = await asyncio.gather(_derive(), _unrelated_writer())

        assert writer_result.thought_id == "t-other"
        # The failed derived child never landed (its insert was unwound); the
        # source and the unrelated writer's row are both present and
        # unaffected by that unwind.
        assert set(await _thought_ids(db)) == {"src-1", "t-other"}

    async def test_derived_child_embed_failure_rolls_back_and_writer_survives(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """An embedding failure *after* the row is written must not corrupt a writer.

        A realistic post-DML failure here (a vec0 write failing right after the
        ``embedding`` table row is already written) unwinds through
        ``store_embedding``'s own ``_write_readback_savepoint`` unit -- no
        compensation of any kind runs in ``_persist_derived_child`` itself for
        the embed step (see its docstring). ``_persist_derived_child`` still
        holds one continuous ``_write_lock`` acquisition spanning the whole
        embed step, and ``store_embedding``'s own acquisition nests inside it
        as a free re-entrant no-op (:class:`_TaskReentrantLock`), so that unit's
        unwind runs under the *same* lock hold the failing write did, without
        any special-casing: a waiting writer cannot land in the gap between the
        failing write and its unwind.

        This pins both required outcomes: the writer's row survives, and the
        connection is left with no open transaction (the unwind actually ran
        to completion rather than being skipped or raced).
        """

        class _WorkingEmbeddingProvider:
            """A provider whose embed call always succeeds."""

            dimension = 3
            model_name = "test-model"

            async def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        producer = ListProducer([_child("derived content")])
        store_with_producer = SqliteEngravaCore(
            db,
            hooks=producer,
            derive_gates=DeriveGates(enabled=False, on_error="raise"),
            auto_embed=True,
            embedding_provider=_WorkingEmbeddingProvider(),
        )
        await store_with_producer.create_thought(_source("src-1"), deduplicate=False)

        paused = asyncio.Event()
        resume = asyncio.Event()
        original_execute = store_with_producer._db.execute
        embed_insert_count = 0

        async def _failing_execute(sql: str, *args: object, **kwargs: object) -> object:
            nonlocal embed_insert_count
            if sql.strip().upper().startswith("INSERT INTO EMBEDDING"):
                embed_insert_count += 1
                if embed_insert_count == 1:
                    # Let the derived child's own embedding row actually get
                    # written -- the realistic failure point is *after* that,
                    # not instead of it (e.g. the vec0 write that follows it).
                    await original_execute(sql, *args, **kwargs)
                    paused.set()
                    await resume.wait()
                    msg = "forced post-write embedding failure (simulating a vec0 failure)"
                    raise RuntimeError(msg)
            return await original_execute(sql, *args, **kwargs)

        setattr(store_with_producer._db, "execute", _failing_execute)  # noqa: B010 -- mypy rejects a direct method-assign here

        async def _derive() -> None:
            with pytest.raises(RuntimeError, match="forced post-write embedding failure"):
                await store_with_producer.derive_existing("src-1")

        async def _unrelated_writer() -> ThoughtRecord:
            await paused.wait()
            held = _write_lock_is_held(store_with_producer)
            if held is not None:
                assert held, (
                    "the embed phase's store_embedding unit must still hold "
                    "_write_lock while it unwinds, when a different task tries "
                    "to acquire it"
                )
            writer_task = asyncio.ensure_future(
                store_with_producer.create_thought(
                    _thought("t-other", content="unrelated content"),
                )
            )
            resume.set()
            return await writer_task

        _, writer_result = await asyncio.gather(_derive(), _unrelated_writer())

        assert writer_result.thought_id == "t-other"
        # The unwind actually completed -- not skipped by a timeout on a fresh
        # acquisition, and not raced by the writer's own commit closing the
        # transaction first.
        assert not db.in_transaction
        # Per-child transaction isolation: the derived child's *row*
        # already committed as its own durable unit in the insert phase --
        # that is unaffected by a later embed-phase failure and correctly
        # survives. What store_embedding's own unit unwinds is only its own
        # not-yet-committed work: the embedding row never lands. The source
        # and the unrelated writer's row are both present.
        thought_ids = set(await _thought_ids(db))
        remaining = thought_ids - {"src-1", "t-other"}
        assert len(remaining) == 1, (
            f"expected exactly the derived child's own row left over -- got {thought_ids!r}"
        )
        child_id = next(iter(remaining))
        embedding_cursor = await db.execute(
            "SELECT 1 FROM embedding WHERE owner_id = ?", (child_id,)
        )
        assert await embedding_cursor.fetchone() is None, (
            "the derived child's embedding must not have survived the unwound embed phase"
        )


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
    """A second task's write now waits for the window instead of joining it.

    An in-process task-reentrant write lock closes the exposure this class
    used to document: ``suspend_auto_commit`` now holds
    :attr:`SqliteEngravaCore._write_lock`
    — a task-reentrant lock — for its whole duration, so a *different* task's
    guarded write can no longer land inside the window's transaction. It
    blocks until the window closes, then runs as its own, independent write.
    """

    async def test_another_tasks_write_now_waits_and_survives_the_window(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """An unrelated write waits out the window and survives its rollback.

        Before this fix, the deferred-commit flag lived on the store instance
        with nothing to stop a second task's write from landing inside the
        open transaction — it joined the window and was rolled back with it,
        and the task that issued it was never told. Now ``suspend_auto_commit``
        holds the task-reentrant write lock for its whole duration:
        ``_unrelated_writer`` below cannot even start its own read until
        ``_window``'s task has released it, which only happens once the
        window has already rolled back.

        The ordering is asserted directly, not inferred from the final state:
        signalling ``window_open`` and then immediately raising does not, on
        its own, prove the unrelated writer's operation started *after* the
        rollback rather than being queued and simply resolved later with the
        same end result either way. ``order`` records the real sequence:
        ``store._db.rollback`` is patched to log when the window's own
        rollback actually runs, and ``_get_thought_row`` is patched to log
        the unrelated writer's own existence-check read (the first such read
        after ``window_open`` fires — the window's own row was already
        written before that point). The assertion below requires the
        rollback to precede that read, which is only possible if the writer
        was genuinely blocked until the window closed.
        """
        await store.create_thought(_thought("t-committed", content="committed earlier"))
        window_open = asyncio.Event()
        order: list[str] = []

        original_rollback = store._db.rollback

        async def _tracking_rollback() -> None:
            order.append("window-rollback")
            await original_rollback()

        setattr(store._db, "rollback", _tracking_rollback)  # noqa: B010 -- mypy rejects a direct method-assign here

        original_get_thought_row = store._get_thought_row
        writer_read_logged = False

        async def _tracking_get_thought_row(thought_id: str) -> object:
            nonlocal writer_read_logged
            if window_open.is_set() and not writer_read_logged:
                writer_read_logged = True
                order.append("writer-read")
            return await original_get_thought_row(thought_id)

        setattr(store, "_get_thought_row", _tracking_get_thought_row)  # noqa: B010 -- mypy rejects a direct method-assign here

        async def _window() -> None:
            async with store.suspend_auto_commit():
                await store.create_thought(_thought("t-window", content="the window's own row"))
                window_open.set()
                msg = "the window's own work failed"
                raise RuntimeError(msg)

        async def _unrelated_writer() -> None:
            await window_open.wait()
            # This call blocks on `_write_lock` until `_window`'s task releases
            # it -- which happens only after the window has rolled back. A
            # `gather` without `return_exceptions=True` would not wait for this
            # task once `_window` raises (it does not cancel siblings), so the
            # outcomes are collected explicitly below instead.
            await store.create_thought(_thought("t-other", content="an unrelated row"))

        outcomes = await asyncio.gather(_window(), _unrelated_writer(), return_exceptions=True)
        window_outcome, writer_outcome = outcomes
        assert isinstance(window_outcome, RuntimeError)
        assert str(window_outcome) == "the window's own work failed"
        assert writer_outcome is None

        # The discriminating assertion: the writer's own read happened only
        # after the window's rollback actually ran -- not merely that the
        # final row set looks right, which a queued-then-resolved rollback
        # could also produce.
        assert order == ["window-rollback", "writer-read"], (
            f"the unrelated writer's read must be provably ordered after the "
            f"window's rollback, not merely consistent with it -- got {order!r}"
        )

        # The window's own row rolled back with it; the unrelated write, which
        # had to wait for the window to close, was never part of that
        # transaction and survives on its own.
        assert set(await _thought_ids(db)) == {"t-committed", "t-other"}

    async def test_cancellation_inside_the_window_releases_the_lock(
        self,
        store: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        """Cancelling a task mid-window rolls back and releases the RESERVED lock.

        ``suspend_auto_commit`` catches ``except BaseException``, not only
        ``except Exception``. ``asyncio.CancelledError`` derives from
        ``BaseException``, so a cancellation landing inside the window (a
        ``bulk_store`` call whose caller times out or is torn down, for
        instance) must still roll the window back: skipping the rollback
        would leave ``db.in_transaction`` ``True`` and the RESERVED lock
        stranded, blocking every other writer on the connection until
        something else eventually committed, rolled back, or closed it.

        Driven by an ``asyncio.Event`` handshake rather than a sleep: the
        window signals once its own row is written and it is parked, so the
        cancellation lands deterministically mid-window instead of hoping a
        delay was long enough.
        """
        window_started = asyncio.Event()

        async def _window() -> None:
            async with store.suspend_auto_commit():
                await store.create_thought(_thought("t-cancelled", content="never durable"))
                window_started.set()
                await asyncio.Event().wait()  # never set; only cancellation ends this

        task = asyncio.create_task(_window())
        await window_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not db.in_transaction, (
            "the RESERVED lock was left stranded after cancellation inside suspend_auto_commit"
        )
        # Rolled back, not merely abandoned: the window's own row must not
        # have survived the cancellation either.
        assert await _thought_ids(db) == []

        # The store must still be usable -- a stranded lock would make this
        # hang (waiting on itself) or raise "cannot start a transaction
        # within a transaction".
        created = await store.create_thought(
            _thought("t-after", content="proves the store still works"),
        )
        assert created.thought_id == "t-after"
        assert await _thought_ids(db) == ["t-after"]


# ---------------------------------------------------------------------------
# Two stores, one database file
# ---------------------------------------------------------------------------


class TestTwoStoresOneFile:
    """The ``revision`` guard is the first thing in this module to order stores.

    Before stage 3, nothing ordered a guarded update across stores except the
    dedup probe window (the two tests further down): a lost update crossed
    the connection boundary unchanged, silently, which is why multiple stores
    writing one file stayed unsupported. The ``revision`` guard lives in the
    database rather than in any in-process lock, so it is the first mechanism
    in this module that reaches across connections for the ordinary
    read-modify-write paths too — this class's first test is the direct
    reproduction this stage exists to fix.
    """

    async def test_an_edit_from_a_second_store_is_now_caught_not_discarded(
        self,
        two_stores: tuple[SqliteEngravaCore, SqliteEngravaCore, aiosqlite.Connection],
    ) -> None:
        """The lost update across the connection boundary is now a raised error.

        Same window as the single-store case, but the competing write comes
        from a different connection — the shape a second process has. Before
        the ``revision`` guard, no in-process lock could close this even in
        principle; the guard closes it anyway, because it lives in the
        database itself: store A's guarded write reads ``revision`` fresh at
        write time and finds store B's committed bump, so it matches no row
        and raises ``StaleDataError`` instead of silently overwriting.
        """
        store_a, store_b, conn_a = two_stores
        await store_a.create_thought(_thought())
        landed: list[str] = []

        async def _edit_from_the_second_store() -> None:
            await store_b.update_thought("t-1", essence="from the second store")
            landed.append((await _row(conn_a, "t-1"))["essence"])

        _interleave_once(store_a, "_get_thought_row", _edit_from_the_second_store)

        with pytest.raises(StaleDataError):
            await store_a.update_thought("t-1", essence="from the first store")

        # Precondition: the second store's edit really committed, and the first
        # store's connection could see it.
        assert landed == ["from the second store"]
        row = await _row(conn_a, "t-1")
        assert row["essence"] == "from the second store"

    async def test_deduplication_across_stores_raises_instead_of_duplicating(
        self,
        two_stores: tuple[SqliteEngravaCore, SqliteEngravaCore, aiosqlite.Connection],
    ) -> None:
        """``deduplicate=True`` serialises across stores instead of racing.

        The first store's probe-and-insert window holds the write lock
        (``BEGIN IMMEDIATE``) for its duration; a second store reaching the
        same window while it is open cannot even start its own transaction.
        Forced — by this test's interleave — to stay open for the whole
        nested call, the first store's window outlasts every retry the second
        store makes, so the second store raises ``WriteContentionError``. That
        exception then unwinds out of the interleaved call and rolls the first
        store's own attempt back too: neither row lands, rather than one
        succeeding partially.
        """
        store_a, store_b, conn_a = two_stores
        # A short busy_timeout on the second store: its every attempt is
        # doomed for as long as the first store's window stays open, which by
        # construction is the whole interleaved call, so a longer timeout
        # would only slow the test down, not change the outcome.
        await store_b._db.execute("PRAGMA busy_timeout = 20")

        async def _insert_from_the_second_store() -> None:
            await store_b.create_thought(
                _thought("t-b", content="identical content"),
                deduplicate=True,
            )

        _interleave_once(store_a, "_get_thought_by_content_hash", _insert_from_the_second_store)

        with pytest.raises(WriteContentionError):
            await store_a.create_thought(
                _thought("t-a", content="identical content"),
                deduplicate=True,
            )

        # Neither insert landed: the second store never got past
        # ``BEGIN IMMEDIATE``, and the first store's own transaction was
        # rolled back once the interleaved call raised into it.
        assert await _thought_ids(conn_a) == []

    async def test_confirmation_counting_across_stores_now_raises_instead_of_racing(
        self,
        two_stores: tuple[SqliteEngravaCore, SqliteEngravaCore, aiosqlite.Connection],
    ) -> None:
        """A confirmation racing a second store's open window fails loudly.

        The first store's window holds the write lock for its duration, so
        the second store cannot even start counting while it is open; it
        raises ``WriteContentionError``, which unwinds the first store's own
        attempt too rather than leaving it half-applied.
        """
        store_a, store_b, conn_a = two_stores
        await store_a.create_thought(_thought(content="identical content"))
        # See the previous test for why this needs a short busy_timeout.
        await store_b._db.execute("PRAGMA busy_timeout = 20")

        async def _confirm_from_the_second_store() -> None:
            await store_b.create_thought(
                _thought("t-b", content="identical content"),
                deduplicate=True,
            )

        _interleave_once(store_a, "_get_thought_by_content_hash", _confirm_from_the_second_store)

        with pytest.raises(WriteContentionError):
            await store_a.create_thought(
                _thought("t-a", content="identical content"),
                deduplicate=True,
            )

        # Neither confirmation landed: the only row is the original insert,
        # untouched.
        assert await _thought_ids(conn_a) == ["t-1"]
        row = await _row(conn_a, "t-1")
        assert row["confirmation_count"] == 0


# ---------------------------------------------------------------------------
# The production default of the write lock's acquire bound
# ---------------------------------------------------------------------------


class TestWriteLockAcquireBoundDefault:
    """The write lock's acquire bound defaults to 600 seconds, however the store is built.

    The slow-embedding test and the deadlock test above override this bound so
    that they finish quickly, so neither would notice the default drifting.
    ``docs/api-reference.md`` states the figure: ``600.0`` in the
    ``write_lock_acquire_timeout_seconds`` row of the constructor table and in
    the ``from_config`` signature. It is sized to clear a slow but legitimate
    hold -- ``bulk_store``'s batch embedding call runs inside
    ``suspend_auto_commit``'s window with the lock held.

    The two store tests read the bound off the lock the built store actually
    holds, not off a declared default: a signature can keep ``600.0`` while
    forwarding something else to the constructor.
    """

    def test_the_documented_default_is_600_seconds(self) -> None:
        """The constant the constructor and ``from_config`` default to is ``600.0``."""
        assert _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS == 600.0

    async def test_a_store_built_with_no_override_holds_the_default_bound(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        """``SqliteEngravaCore(conn)`` gives its write lock the default bound."""
        store = SqliteEngravaCore(db)

        assert store._write_lock._acquire_timeout_seconds == _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS

    async def test_from_config_with_no_override_holds_the_default_bound(
        self,
        tmp_path: Path,
    ) -> None:
        """``from_config`` with no override gives the store's own write lock the default bound."""
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(f"database:\n  path: {tmp_path / 'store.db'}\n", encoding="utf-8")

        store = await SqliteEngravaCore.from_config(config_path)
        try:
            assert store._write_lock._acquire_timeout_seconds == _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS
        finally:
            await store.close()

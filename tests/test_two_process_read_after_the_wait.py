"""A concurrent delete journals what the holder left, not what it read before the wait.

``delete_thought`` and ``delete_edge`` each read a row (a before-image for
the journal, and, for ``delete_thought``, the embedding rowid the vector
purge needs) before doing their own guarded write. Both open with ``BEGIN
IMMEDIATE``, so that read can now be preceded by a real wait for another
process's write lock -- and until this fix, the read still ran *before* that
wait, not after it. A concurrent delete's journaled before-image should be
the row actually deleted; at the unfixed revision, the before-image is read
before the wait too, so it carries whatever the row looked like *before* the
other process's edit landed, not the row the delete actually removed. There
is no ``revision`` guard here to preserve either way -- a delete has nothing
to compare a stale read against, only a row to remove -- so this is a pure
correctness fix, unlike the update paths (see
``tests/test_begin_immediate_contention_is_typed.py`` for those, which keep
their documented fail-fast-under-contention contract instead).

**The rendezvous is hooked to the read itself, not timed.** An earlier
version of this test had the holder hold the lock for a fixed duration and
then cross-checked ``time.monotonic()`` readings from both processes to
confirm the contender's call had actually started before that hold ended --
a validity check bolted onto a timing-based design, not a guarantee. This
version removes the guesswork instead of merely checking for it: the
contender's own store has its private before-image read
(``_get_thought_row`` for ``delete_thought``, ``_get_edge_row`` for
``delete_edge``) wrapped so that the first call for the target row sets a
``read_done`` event, the instant that specific read actually returns. The
holder takes the lock, makes its edit, signals ``holding``, then waits
(bounded to ``_READ_DONE_TIMEOUT_SECONDS``) on ``read_done`` before
committing regardless of whether it arrived -- and records which happened.

What the two revisions do under this protocol:

* At the unfixed revision, the contender's before-image read runs *before*
  its own ``BEGIN IMMEDIATE`` -- nothing blocks it, so it completes near
  instantly, almost always while the holder is still waiting out its own
  bound. ``read_done`` arrives well within ``_READ_DONE_TIMEOUT_SECONDS``,
  and the before-image the contender captured is the seed value, not the
  holder's edit -- RED. The one way the unfixed revision escapes RED is a
  contender stalled for longer than the bound between signalling and its
  read: it then reads the holder's edit and passes. That is a missed
  detection on a pathologically stalled run, never a false failure of the
  fixed revision.
* At the fixed revision, the contender's ``BEGIN IMMEDIATE`` cannot succeed
  until the holder's own commit releases the write lock, and that same read
  cannot run until ``BEGIN IMMEDIATE`` has succeeded. ``read_done`` can
  therefore *never* arrive before the holder commits: not "is unlikely to",
  structurally cannot, since the read it guards is provably downstream of
  the lock release. The holder's bounded wait always elapses in full, the
  before-image the contender then captures is the holder's own edit, and
  the holder's own recorded "did it arrive" answer is always
  ``False`` -- GREEN, and the fact of that ``False`` is itself part of what
  the test below pins, not an incidental side effect.

This module races real ``multiprocessing.Process`` workers against one
on-disk database file, reusing ``tests/test_two_process_write_busy_wait.py``'s
own deterministic rendezvous (an explicit "holding" / "about to call" signal
pair rather than a fixed sleep, so a slow contender can never be mistaken for
one that raced ahead), bounded waits, and ``_reap`` forced-stop cleanup -- see
that module's own docstring for why a real process, not an in-process second
connection, is what exercises this shape.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import sqlite3
import time
from typing import TYPE_CHECKING, TypedDict

import aiosqlite
import pytest

import engrava

if TYPE_CHECKING:
    from pathlib import Path

    from engrava import SqliteEngravaCore

#: Bound on the holder's own wait for ``read_done`` before it commits
#: regardless of whether it arrived -- see :func:`_holder`. On the fixed
#: revision this always elapses in full: the contender's own ``BEGIN
#: IMMEDIATE`` cannot even begin its read until this commit releases the
#: write lock, so ``read_done`` cannot arrive first -- see the module
#: docstring.
_READ_DONE_TIMEOUT_SECONDS = 1.0

#: The contender's own ``PRAGMA busy_timeout``, comfortably above
#: ``_READ_DONE_TIMEOUT_SECONDS`` so a correctly-waiting contender always
#: has margin left.
_CONTENDER_BUSY_TIMEOUT_MS = 5000

#: Floor on a successful contender's elapsed time -- comfortably below
#: ``_READ_DONE_TIMEOUT_SECONDS`` to absorb scheduling jitter, but well
#: above the near-instant failure a busy-wait regression would produce, so
#: the two shapes cannot be confused.
_MIN_WAIT_SECONDS = _READ_DONE_TIMEOUT_SECONDS * 0.5

#: Bound on waiting for the contender's "about to call" signal.
_ABOUT_TO_CALL_TIMEOUT_SECONDS = 15.0

#: Bound on every ``Process.join`` / ``Queue.get`` below that is not itself
#: one of the rendezvous signals above.
_JOIN_TIMEOUT_SECONDS = _READ_DONE_TIMEOUT_SECONDS + 20.0

#: Grace period given to a process after ``terminate()`` (and again after
#: ``kill()``) before :func:`_reap` gives up waiting for it to actually exit.
_FORCE_STOP_GRACE_SECONDS = 5.0

_DELETE_THOUGHT_ID = "seed-thought-to-delete"
_DELETE_EDGE_ID = "seed-edge-to-delete"
_EDGE_FROM = "seed-thought-edge-from"
_EDGE_TO = "seed-thought-edge-to"

_SEED_ESSENCE = "essence written when the row was seeded"
_HOLDER_ESSENCE = "essence written by the concurrent holder"

_SEED_WEIGHT = 0.25
_HOLDER_DECAY = 0.42

#: The two delete operations this module exercises.
_DELETE_OPERATIONS = ("delete_thought", "delete_edge")

#: Recorded once, in the parent process, so a contender's own import can be
#: compared against it.
_PARENT_ENGRAVA_FILE = engrava.__file__


class _ContenderResult(TypedDict):
    """What a contender process reports back through its result queue."""

    kind: str
    """``"ok"`` or ``"error"``."""

    op: str
    """The operation name this result is for."""

    error: str | None
    """``"<module>.<type>: <message>"`` when ``kind`` is ``"error"``, else ``None``."""

    elapsed: float | None
    """Seconds from just before the guarded call to just after it settled.

    ``None`` only when the contender failed before reaching the call at all,
    which every assertion below treats as a hard failure regardless.
    """

    engrava_file: str | None
    """This process's own ``engrava.__file__``, or ``None`` if never reached."""


def _reap(process: multiprocessing.process.BaseProcess, timeout: float) -> None:
    """Join ``process``, terminating then killing it if the bounded join times out.

    Identical in shape to ``test_two_process_write_busy_wait.py``'s own
    ``_reap`` -- see that module for the full rationale. Repeated here rather
    than imported: each cross-process test module in this suite is
    self-contained.

    Args:
        process: The child process to join.
        timeout: How long to wait for a clean exit before forcing one.

    Raises:
        AssertionError: Whenever ``process`` was still alive after
            ``timeout`` and had to be terminated or killed.

    """
    process.join(timeout)
    if not process.is_alive():
        return
    process.terminate()
    process.join(_FORCE_STOP_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(_FORCE_STOP_GRACE_SECONDS)
    msg = (
        f"process {process.pid} did not exit within {timeout}s and had to be "
        "forced to stop -- treating this as a failure rather than continuing"
    )
    raise AssertionError(msg)


def _holder(  # noqa: PLR0917 - multiprocessing.Process passes `args` positionally
    path: str,
    op: str,
    holding: multiprocessing.synchronize.Event,
    read_done: multiprocessing.synchronize.Event,
    read_done_arrived: multiprocessing.sharedctypes.Synchronized[int],
    read_done_timeout_seconds: float,
) -> None:
    """Take the write lock with a real edit to the contended row, and hold it.

    A raw ``sqlite3`` connection, deliberately outside the engrava core: this
    process plays the role of "some other writer", not a second engrava
    instance -- exactly like ``test_two_process_write_busy_wait.py``'s own
    holder, except this one's ``UPDATE`` targets the specific row (and
    column) the contender will read and act on, so the contender's read can
    actually observe whether it happened before or after this hold.

    Signals ``holding`` once the write lock and the edit are both in place,
    then waits (bounded by ``read_done_timeout_seconds``) for ``read_done``
    -- set by the contender's own hooked read, not by a fixed sleep -- and
    commits once that wait ends, whether ``read_done`` arrived or not. See
    the module docstring for why, on the fixed revision, this bound always
    elapses in full: the contender's read this signal guards cannot run
    before this very commit releases the write lock it is waiting on, so
    there is nothing racy about always using the whole budget.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation the contender will run; selects which row and
            column this holder edits.
        holding: Set once the write lock is held and the edit has been made.
        read_done: Waited on (bounded) before committing; set by the
            contender's own hooked read -- see :func:`_hook_first_read`.
        read_done_arrived: Written ``1`` or ``0`` once the bounded wait on
            ``read_done`` ends, recording which of the two it was -- read by
            the test itself. It records only whether the holder observed
            ``read_done`` before its bounded wait ended (see the module docstring).
        read_done_timeout_seconds: Bound on waiting for ``read_done``.

    """
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("BEGIN IMMEDIATE")
    if op == "delete_thought":
        conn.execute(
            "UPDATE thought SET essence = ?, revision = revision + 1 WHERE thought_id = ?",
            (_HOLDER_ESSENCE, _DELETE_THOUGHT_ID),
        )
    elif op == "delete_edge":
        conn.execute(
            "UPDATE edge SET decay_multiplier = ?, revision = revision + 1 WHERE edge_id = ?",
            (_HOLDER_DECAY, _DELETE_EDGE_ID),
        )
    else:
        msg = f"unknown op: {op}"
        raise AssertionError(msg)
    holding.set()
    arrived = read_done.wait(read_done_timeout_seconds)
    read_done_arrived.value = 1 if arrived else 0
    conn.execute("COMMIT")
    conn.close()


def _contender_worker(  # noqa: PLR0917 - multiprocessing.Process passes `args` positionally
    path: str,
    op: str,
    busy_timeout_ms: int,
    about_to_call: multiprocessing.synchronize.Event,
    read_done: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.Queue[_ContenderResult],
) -> None:
    """Process entry point: connect fresh, then run one guarded delete.

    Mirrors ``test_two_process_write_busy_wait.py``'s own
    ``_contender_worker`` -- see that module for why the outcome is reported
    through a queue rather than the process exit code.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation to run; one of :data:`_DELETE_OPERATIONS`.
        busy_timeout_ms: The value this process sets for its own
            ``PRAGMA busy_timeout`` before running ``op``.
        about_to_call: Set just before the guarded call is made.
        read_done: Set by the hooked read -- see :func:`_hook_first_read`.
        result_queue: Queue the single :class:`_ContenderResult` is put on.

    """
    info: dict[str, object] = {}
    try:
        coro = _contender_async(path, op, busy_timeout_ms, about_to_call, read_done, info)
        result = asyncio.run(coro)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        t0 = info.get("t0")
        elapsed = time.monotonic() - t0 if isinstance(t0, float) else None
        engrava_file = info.get("engrava_file")
        result_queue.put(
            {
                "kind": "error",
                "op": op,
                "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
                "elapsed": elapsed,
                "engrava_file": engrava_file if isinstance(engrava_file, str) else None,
            }
        )
    else:
        result_queue.put(result)


def _hook_first_read(
    store: SqliteEngravaCore, op: str, read_done: multiprocessing.synchronize.Event
) -> None:
    """Patch the store's own before-image read to signal ``read_done`` once it returns.

    ``delete_thought`` reads the row it is about to delete via
    ``_get_thought_row``, ``delete_edge`` via ``_get_edge_row`` -- see each
    method's own docstring for exactly which call this targets: the one that
    determines the journaled before-image, immediately after that method's
    own ``BEGIN IMMEDIATE``. ``read_done`` is set only *after* the wrapped
    read has actually returned, not before calling it, so the signal
    reflects a read that genuinely executed on the connection, not merely
    one that was about to.

    Args:
        store: The contender's own store, patched in place before its
            guarded call runs.
        op: ``"delete_thought"`` or ``"delete_edge"`` -- selects which
            private read method and target row id this hooks.
        read_done: Set on the first call whose id matches the target row.

    Raises:
        AssertionError: If ``op`` is neither of the two supported values.

    """
    if op == "delete_thought":
        original_thought = store._get_thought_row

        async def _hooked_thought_row(thought_id: str) -> aiosqlite.Row | None:
            row = await original_thought(thought_id)
            if thought_id == _DELETE_THOUGHT_ID and not read_done.is_set():
                read_done.set()
            return row

        store._get_thought_row = _hooked_thought_row  # type: ignore[method-assign]
    elif op == "delete_edge":
        original_edge = store._get_edge_row

        async def _hooked_edge_row(edge_id: str) -> aiosqlite.Row | None:
            row = await original_edge(edge_id)
            if edge_id == _DELETE_EDGE_ID and not read_done.is_set():
                read_done.set()
            return row

        store._get_edge_row = _hooked_edge_row  # type: ignore[method-assign]
    else:
        msg = f"unknown op: {op}"
        raise AssertionError(msg)


async def _contender_async(  # noqa: PLR0917 - mirrors _contender_worker's own parameter list
    path: str,
    op: str,
    busy_timeout_ms: int,
    about_to_call: multiprocessing.synchronize.Event,
    read_done: multiprocessing.synchronize.Event,
    info: dict[str, object],
) -> _ContenderResult:
    """Connect with journaling enabled, then run the delete named by ``op``.

    Journaling is on (unlike ``test_two_process_write_busy_wait.py``'s
    contender) because this module asserts on the journal's before-image --
    its whole point.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation to run.
        busy_timeout_ms: The value to set for ``PRAGMA busy_timeout``.
        about_to_call: Set immediately before the guarded call.
        read_done: Passed to :func:`_hook_first_read` for this store.
        info: Mutable carrier for ``engrava_file`` and ``t0``, read by the
            caller regardless of whether this coroutine raises.

    Returns:
        A :class:`_ContenderResult` with ``kind="ok"``.

    Raises:
        BaseException: Whatever the guarded operation itself raised.

    """
    import engrava
    from engrava import SqliteEngravaCore

    info["engrava_file"] = engrava.__file__

    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    store = SqliteEngravaCore(conn, journal_enabled=True)
    _hook_first_read(store, op, read_done)

    info["t0"] = time.monotonic()
    about_to_call.set()
    try:
        if op == "delete_thought":
            deleted = await store.delete_thought(_DELETE_THOUGHT_ID)
            if not deleted:
                msg = "delete_thought reported no row deleted"
                raise AssertionError(msg)
        elif op == "delete_edge":
            deleted = await store.delete_edge(_DELETE_EDGE_ID)
            if not deleted:
                msg = "delete_edge reported no row deleted"
                raise AssertionError(msg)
        else:
            msg = f"unknown op: {op}"
            raise AssertionError(msg)
    finally:
        await conn.close()

    t0 = info["t0"]
    assert isinstance(t0, float)
    return {
        "kind": "ok",
        "op": op,
        "error": None,
        "elapsed": time.monotonic() - t0,
        "engrava_file": engrava.__file__,
    }


async def _bootstrap_and_seed(path: str) -> None:
    """Create the schema and seed every row a parametrized case needs.

    Args:
        path: Path to the (not yet existing) SQLite database file.

    """
    from engrava import (
        CoreThoughtRecord,
        EdgeRecord,
        EdgeType,
        KnowledgeSource,
        LifecycleStatus,
        Priority,
        SqliteEngravaCore,
        ThoughtType,
    )

    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(conn, journal_enabled=True)
    await store.ensure_schema()

    def _seed_thought(thought_id: str) -> CoreThoughtRecord:
        return CoreThoughtRecord(
            thought_id=thought_id,
            thought_type=ThoughtType.NOTE,
            essence=_SEED_ESSENCE,
            content="seed content",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=0,
            updated_cycle=0,
            source="test-suite",
        )

    for thought_id in (_DELETE_THOUGHT_ID, _EDGE_FROM, _EDGE_TO):
        await store.create_thought(_seed_thought(thought_id))

    await store.create_edge(
        EdgeRecord(
            edge_id=_DELETE_EDGE_ID,
            from_thought_id=_EDGE_FROM,
            to_thought_id=_EDGE_TO,
            edge_type=EdgeType.DEPENDS_ON,
            weight=_SEED_WEIGHT,
            created_cycle=0,
            source=KnowledgeSource.EXPERIENCE,
        )
    )

    await conn.close()


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """A real on-disk, schema-bootstrapped, WAL-enabled, pre-seeded database file."""
    path = str(tmp_path / "two_process_read_after_the_wait.db")
    asyncio.run(_bootstrap_and_seed(path))
    return path


def _run_holder_and_contender(db_path: str, op: str) -> tuple[_ContenderResult, bool]:
    """Run the holder/contender rendezvous once for ``op``.

    Args:
        db_path: Path to the shared SQLite database file.
        op: Which operation to run.

    Returns:
        The contender's own reported outcome, and whether the holder's own
        bounded wait for ``read_done`` saw it arrive before committing --
        whether the holder observed ``read_done`` before its bounded
        wait ended (see the module docstring).

    """
    ctx = multiprocessing.get_context("spawn")
    holding = ctx.Event()
    about_to_call = ctx.Event()
    read_done = ctx.Event()
    read_done_arrived = ctx.Value("b", 0)
    holder = ctx.Process(
        target=_holder,
        args=(
            db_path,
            op,
            holding,
            read_done,
            read_done_arrived,
            _READ_DONE_TIMEOUT_SECONDS,
        ),
    )
    holder.start()
    try:
        assert holding.wait(_JOIN_TIMEOUT_SECONDS), (
            "holder process never signalled it holds the lock"
        )

        result_queue: multiprocessing.Queue[_ContenderResult] = ctx.Queue()
        contender = ctx.Process(
            target=_contender_worker,
            args=(db_path, op, _CONTENDER_BUSY_TIMEOUT_MS, about_to_call, read_done, result_queue),
        )
        contender.start()
        try:
            assert about_to_call.wait(_ABOUT_TO_CALL_TIMEOUT_SECONDS), (
                "contender never signalled it was about to make its call"
            )
            result = result_queue.get(timeout=_JOIN_TIMEOUT_SECONDS)
        finally:
            _reap(contender, _JOIN_TIMEOUT_SECONDS)
        assert contender.exitcode == 0, f"contender process exited with {contender.exitcode}"
    finally:
        _reap(holder, _JOIN_TIMEOUT_SECONDS)
    assert holder.exitcode == 0, f"holder process exited with {holder.exitcode}"

    assert result["engrava_file"] == _PARENT_ENGRAVA_FILE, (
        f"contender imported engrava from {result['engrava_file']!r}, "
        f"the parent test process imported it from {_PARENT_ENGRAVA_FILE!r}"
    )
    return result, bool(read_done_arrived.value)


@pytest.mark.parametrize("op", _DELETE_OPERATIONS)
def test_delete_journals_the_row_the_holder_left_not_a_stale_snapshot(
    db_path: str, op: str
) -> None:
    """A concurrent delete's journaled before-image is the row actually deleted.

    The holder changes the row's ``essence`` (edge: ``decay_multiplier``),
    signals that it holds the lock, then waits (bounded by
    ``_READ_DONE_TIMEOUT_SECONDS``) for the contender's own before-image read
    to fire ``read_done`` -- hooked directly onto ``_get_thought_row`` /
    ``_get_edge_row``, see :func:`_hook_first_read` -- before committing
    regardless of whether it arrived. The contender deletes the same row.

    **This pins two things, not one.** First, the durable delete and its
    journaled before-image: at the unfixed revision, the before-image is
    read *before* the contender's own ``BEGIN IMMEDIATE`` waits out the
    holder, so it carries the seed value, not the holder's edit; after the
    fix, that read happens only once the contender's own lock wait is over,
    so it carries the holder's edit. Second -- and this is what makes the
    first assertion trustworthy rather than assumed -- whether the holder's
    own wait for ``read_done`` actually saw it arrive before committing.
    Structurally, on the fixed revision it cannot: the read that sets
    ``read_done`` is downstream of the very commit the holder is waiting to
    make, so ``read_done_arrived`` is always ``False`` there. At the unfixed
    revision it is ``True`` on any run where the contender is not stalled past
    the bound before its read, because nothing blocks that read. This replaces an earlier version of
    this test that instead cross-checked ``time.monotonic()`` readings from
    both processes to *infer* the rendezvous was valid; hooking the read
    itself proves it directly instead.
    """
    result, read_done_arrived = _run_holder_and_contender(db_path, op)

    assert result["kind"] == "ok", f"{op} failed under contention: {result['error']}"

    assert not read_done_arrived, (
        "the holder's own bounded wait saw read_done arrive before it committed -- "
        "the contender's before-image read ran while the holder still held the "
        "lock; a fixed revision cannot produce this, because its read waits for "
        "the write lock"
    )

    elapsed = result["elapsed"]
    assert elapsed is not None
    assert elapsed >= _MIN_WAIT_SECONDS, (
        f"{op} returned after only {elapsed:.3f}s -- too fast to have waited out "
        f"the holder's {_READ_DONE_TIMEOUT_SECONDS}s bound; the holder may not "
        "have actually held the lock"
    )

    conn = sqlite3.connect(db_path)
    try:
        if op == "delete_thought":
            row = conn.execute(
                "SELECT 1 FROM thought WHERE thought_id = ?", (_DELETE_THOUGHT_ID,)
            ).fetchone()
            assert row is None, "delete_thought's delete is not durable"

            journal_row = conn.execute(
                "SELECT delta FROM journal_entry "
                "WHERE mutation_type = 'DELETE_THOUGHT' AND target_id = ?",
                (_DELETE_THOUGHT_ID,),
            ).fetchone()
            assert journal_row is not None, "no DELETE_THOUGHT journal entry was recorded"
            delta = json.loads(journal_row[0])
            before_essence = delta["before"]["essence"]
            assert before_essence == _HOLDER_ESSENCE, (
                f"journaled before-image carries essence={before_essence!r} -- the seed "
                f"value read before the wait, not {_HOLDER_ESSENCE!r}, the holder's "
                "committed essence that a read after the wait would have seen"
            )
        elif op == "delete_edge":
            row = conn.execute(
                "SELECT 1 FROM edge WHERE edge_id = ?", (_DELETE_EDGE_ID,)
            ).fetchone()
            assert row is None, "delete_edge's delete is not durable"

            journal_row = conn.execute(
                "SELECT delta FROM journal_entry "
                "WHERE mutation_type = 'DELETE_EDGE' AND target_id = ?",
                (_DELETE_EDGE_ID,),
            ).fetchone()
            assert journal_row is not None, "no DELETE_EDGE journal entry was recorded"
            delta = json.loads(journal_row[0])
            before_decay = delta["before"]["decay_multiplier"]
            assert before_decay == _HOLDER_DECAY, (
                f"journaled before-image carries decay_multiplier={before_decay!r} -- the "
                f"seed value read before the wait, not {_HOLDER_DECAY!r}, the holder's "
                "committed value that a read after the wait would have seen"
            )
        else:
            msg = f"unknown op: {op}"
            raise AssertionError(msg)
    finally:
        conn.close()

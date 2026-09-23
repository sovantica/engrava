"""A contending write waits out ``PRAGMA busy_timeout`` instead of failing at once.

Every store method that opens its own transaction in order to write does so
with ``BEGIN IMMEDIATE``, not a deferred ``BEGIN``. A deferred transaction's
first read takes a WAL snapshot; a write later in the same transaction must
upgrade that snapshot, and SQLite refuses the upgrade while another
connection holds the write lock -- returning ``SQLITE_BUSY`` *without ever
invoking the busy handler*. That made a perfectly ordinary write fail at once
with a raw "database is locked" instead of waiting out
``PRAGMA busy_timeout`` like every other write on the connection.
``BEGIN IMMEDIATE`` takes the write lock up front, through the busy handler,
before anything in the guarded body gets to read. Not every write path reads
before it writes, though -- see
:func:`test_write_unit_waits_for_a_busy_lock`'s own docstring for exactly
which of the five operations exercised here were actually affected.
``update_thought`` is deliberately not one of them: a later workstream
reverted its own unit to a deferred ``BEGIN`` on purpose, restoring a
documented fail-fast-under-contention contract these five never had — see
``tests/test_begin_immediate_contention_is_typed.py`` for that path's own,
now-opposite pinning.

This module races real ``multiprocessing.Process`` workers (not asyncio
tasks or a second connection in the same process, which shares the
aiosqlite thread model differently and does not exercise the reported
scenario) against one on-disk database file. The rendezvous between the two
processes is deliberately explicit rather than time-based, mirroring
``tests/test_two_process_dedup.py``'s own bounded rendezvous: the holder
takes the write lock with its own ``BEGIN IMMEDIATE`` and signals once it
holds it; the contender connects, sets its pragmas, and signals just before
it makes its guarded call; only *then* does the holder start counting its
hold -- otherwise a slow contender (process spawn, import, connect) could
reach its call only after a fixed-duration hold had already elapsed and the
lock had already been released, passing or failing for the wrong reason.
Every wait on every one of these signals is bounded, and a child still alive
after its bounded join is forced to stop rather than left to outlive the
test.

The contender always imports ``engrava`` fresh in its own process (a
``multiprocessing`` *spawn* child does not inherit the parent's already-
imported modules the way a ``fork`` child would); this module confirms that
import resolves to the same copy of the package the test itself is running
against, rather than a stray installed copy on the child's default path.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3
import time
from typing import TYPE_CHECKING, TypedDict

import aiosqlite
import pytest

import engrava

if TYPE_CHECKING:
    from pathlib import Path

#: How long the holder process keeps the write lock, once it starts counting
#: (see the module docstring for why that is not the same instant it
#: acquires the lock). Long enough that a contender which merely got lucky
#: with scheduling could not be mistaken for one that actually waited out
#: contention; short enough to keep this module's total run time reasonable
#: across five parametrized operations.
_HOLD_SECONDS = 1.0

#: The contender's own ``PRAGMA busy_timeout``, comfortably above
#: ``_HOLD_SECONDS`` so a correctly-waiting contender always has margin left.
_CONTENDER_BUSY_TIMEOUT_MS = 5000

#: Floor on a successful contender's elapsed time. Comfortably below
#: ``_HOLD_SECONDS`` to absorb scheduling jitter, but an order of magnitude
#: above the "fails at once" signature this fix closes (observed at ~2 ms
#: against the unfixed tree -- see the workstream's report for the captured
#: transcript), so the two shapes cannot be confused.
_MIN_WAIT_SECONDS = _HOLD_SECONDS * 0.5

#: Ceiling on the negative control's elapsed time: with ``busy_timeout=0``
#: the contender must fail long before the holder would ever release the
#: lock, or the control has not actually forced the "no waiting" case.
_IMMEDIATE_FAILURE_MAX_SECONDS = _HOLD_SECONDS * 0.5

#: Bound on waiting for the contender's "about to call" signal, and (as a
#: safety ceiling, not the expected duration) on the holder's own wait for
#: that same signal. Generous relative to process spawn + import overhead
#: under ``spawn``, which is the whole reason this signal exists rather than
#: a fixed pre-call sleep.
_ABOUT_TO_CALL_TIMEOUT_SECONDS = 15.0

#: Safety ceiling on the negative control's holder waiting for the
#: contender's "I have an outcome" signal, in place of a fixed hold. A
#: correct negative-control run resolves in a few milliseconds; this bound
#: only guards against the contender hanging before it ever reports.
_RELEASE_SAFETY_CEILING_SECONDS = 15.0

#: Bound on every ``Process.join`` / ``Queue.get`` below that is not itself
#: one of the rendezvous signals above. Generous relative to
#: ``_HOLD_SECONDS`` so a healthy run never approaches it, but never
#: unbounded -- a hang here must fail the test, not wedge the suite.
_JOIN_TIMEOUT_SECONDS = _HOLD_SECONDS + 20.0

#: Grace period given to a process after ``terminate()`` (and again after
#: ``kill()``) before :func:`_reap` gives up waiting for it to actually exit.
_FORCE_STOP_GRACE_SECONDS = 5.0

# Seed rows every parametrized operation can draw on. One shared seed set
# (rather than a fixture branching per operation) keeps every operation's
# setup identical -- the fix under test does not special-case any of them,
# and neither does this test's own seeding.
_SEED_THOUGHT_A = "seed-thought-a"
_SEED_THOUGHT_B = "seed-thought-b"
_SEED_THOUGHT_TO_DELETE = "seed-thought-to-delete"
_SEED_THOUGHT_ACTION_SOURCE = "seed-thought-action-source"
_SEED_EDGE = "seed-edge"

_NEW_THOUGHT_ID = "contender-created-thought"
_NEW_EDGE_ID = "contender-created-edge"
_NEW_ACTION_ID = "contender-created-action"

#: Every operation the workstream's acceptance criteria names.
#: ``update_thought`` was here too until a later workstream reverted its own
#: unit to a deferred ``BEGIN`` on purpose, restoring a documented
#: fail-fast-under-contention contract — see the module docstring.
_OPERATIONS = (
    "create_thought",
    "delete_thought",
    "create_edge",
    "delete_edge",
    "create_action",
)

#: Recorded once, in the parent process, so a contender's own import can be
#: compared against it -- the portable form of "the child imported the
#: package under test," independent of any one host's install layout.
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

    ``None`` only when the contender failed before reaching the call at all
    (an import or connection failure), which every assertion below treats as
    a hard failure regardless.
    """

    engrava_file: str | None
    """This process's own ``engrava.__file__``, or ``None`` if never reached."""


def _reap(process: multiprocessing.process.BaseProcess, timeout: float) -> None:
    """Join ``process``, terminating then killing it if the bounded join times out.

    A child still alive after ``timeout`` is sent ``terminate()``, given a
    short grace period to exit, sent ``kill()`` if it still has not, and
    joined once more with that same bounded grace period -- then this
    raises, whether or not that last join actually saw it exit. This does
    **not** guarantee the child is gone by the time it raises: both of its
    own joins are themselves bounded, not unbounded waits for exit. What it
    does guarantee is that reaching this point is never accepted silently --
    a caller that only asserted ``process.exitcode == 0`` afterwards would
    already fail on a genuinely forced exit (a negative signal number, never
    ``0``), but raising here names the actual problem -- a hang that needed
    forcing -- instead of leaving a generic exit-code mismatch to stand in
    for it.

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
    holding: multiprocessing.synchronize.Event,
    about_to_call: multiprocessing.synchronize.Event,
    release_after_contender: multiprocessing.synchronize.Event | None,
    rendezvous_failed: multiprocessing.synchronize.Event,
    hold_seconds: float,
    about_to_call_timeout_seconds: float,
    release_timeout_seconds: float,
) -> None:
    """Take the write lock with ``BEGIN IMMEDIATE`` and hold it deterministically.

    A raw ``sqlite3`` connection, deliberately outside the engrava core:
    this process plays the role of "some other writer," not a second engrava
    instance.

    Signals ``holding`` the instant the write lock is actually held, then
    waits (bounded) on ``about_to_call`` *before* starting to hold for
    real -- the lock is taken well before the contender has even spawned,
    imported and connected, so starting a fixed hold clock immediately upon
    acquiring the lock could let a slow contender reach its call only after
    the lock was already released, making the test's outcome depend on
    scheduling rather than on the fix. If that bounded wait times out, this
    sets ``rendezvous_failed`` and releases the lock at once, **without**
    falling through to the normal hold -- a timed-out wait must never be
    treated as if the rendezvous happened, or the lock could already be gone
    by the time a slow contender (still within its own, separately bounded
    window) actually signals and makes its call, and the parent would accept
    that run's timing as if the hold had genuinely covered it.

    Once ``about_to_call`` fires, this holds either for a fixed
    ``hold_seconds`` (when ``release_after_contender`` is ``None``) or until
    ``release_after_contender`` fires, bounded by ``release_timeout_seconds``
    (the negative control: the lock must never be released so close to the
    contender's attempt that a correct failure could race a release and pass
    for the wrong reason). That second wait times out the same way: also a
    ``rendezvous_failed`` methodology failure, not silently accepted as "the
    contender must have finished."

    Args:
        path: Path to the shared SQLite database file.
        holding: Set once the write lock is held.
        about_to_call: Waited on (bounded) before the hold begins; set by the
            contender just before it makes its guarded call.
        release_after_contender: When given, waited on (bounded) instead of
            sleeping ``hold_seconds`` -- set by the contender once its call
            has settled, success or failure alike.
        rendezvous_failed: Set by this process when either bounded wait below
            times out, so the parent can fail the test on that specific
            cause instead of trusting a timing result the rendezvous never
            actually guaranteed.
        hold_seconds: Fixed hold duration, used only when
            ``release_after_contender`` is ``None``.
        about_to_call_timeout_seconds: Bound on waiting for ``about_to_call``.
        release_timeout_seconds: Bound on waiting for
            ``release_after_contender``.

    """
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("CREATE TABLE IF NOT EXISTS holder_marker(x)")
    conn.execute("INSERT INTO holder_marker VALUES (1)")
    holding.set()
    if not about_to_call.wait(about_to_call_timeout_seconds):
        rendezvous_failed.set()
    elif release_after_contender is not None:
        if not release_after_contender.wait(release_timeout_seconds):
            rendezvous_failed.set()
    else:
        time.sleep(hold_seconds)
    conn.execute("COMMIT")
    conn.close()


def _contender_worker(  # noqa: PLR0917 - multiprocessing.Process passes `args` positionally
    path: str,
    op: str,
    busy_timeout_ms: int,
    about_to_call: multiprocessing.synchronize.Event,
    release_after_contender: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.Queue[_ContenderResult],
) -> None:
    """Process entry point: connect fresh, then run one guarded operation.

    Runs in its own process with its own event loop and its own SQLite
    connection -- the two boundaries an in-process lock cannot see across.
    The outcome (or a failure) is reported back through ``result_queue``
    rather than the process exit code, so the parent can tell a clean
    success apart from a raised exception, and can still read the
    ``engrava.__file__`` this process resolved even when the guarded call
    itself failed.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation to run; one of :data:`_OPERATIONS`.
        busy_timeout_ms: The value this process sets for its own
            ``PRAGMA busy_timeout`` before running ``op``.
        about_to_call: Set just before the guarded call is made -- see
            :func:`_holder` for why the holder's own hold timer waits on it.
        release_after_contender: Set once the guarded call has settled,
            success or failure alike -- see :func:`_holder` for the one
            caller (the negative control) that waits on it.
        result_queue: Queue the single :class:`_ContenderResult` is put on.

    """
    info: dict[str, object] = {}
    try:
        coro = _contender_async(
            path, op, busy_timeout_ms, about_to_call, release_after_contender, info
        )
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


async def _contender_async(  # noqa: PLR0917 - called positionally, mirroring _contender_worker
    path: str,
    op: str,
    busy_timeout_ms: int,
    about_to_call: multiprocessing.synchronize.Event,
    release_after_contender: multiprocessing.synchronize.Event,
    info: dict[str, object],
) -> _ContenderResult:
    """Connect, then run the operation named by ``op``.

    ``info["engrava_file"]`` and ``info["t0"]`` are recorded before
    ``about_to_call`` is signalled, so :func:`_contender_worker` can still
    read them if the call raises. ``release_after_contender`` is set in its
    own outermost ``finally``, independent of closing the connection: a
    caller (the negative control's holder) waiting on it must be released
    however the guarded call *and* the close each end, including a close
    that itself raises -- which is also why the close gets its own, inner
    ``finally`` rather than sharing one statement with the event: an
    unclosed ``aiosqlite`` connection hangs the interpreter at exit, but a
    failure closing it must not by itself suppress the release signal.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation to run; one of :data:`_OPERATIONS`.
        busy_timeout_ms: The value to set for ``PRAGMA busy_timeout``.
        about_to_call: Set immediately before the guarded call, once the
            connection and its pragmas are already in place.
        release_after_contender: Set once the guarded call has settled.
        info: Mutable carrier for ``engrava_file`` and ``t0``, read by the
            caller regardless of whether this coroutine raises.

    Returns:
        A :class:`_ContenderResult` with ``kind="ok"``.

    Raises:
        BaseException: Whatever the guarded operation itself raised (e.g. an
            ``aiosqlite.OperationalError`` for "database is locked").

    """
    import engrava
    from engrava import (
        ActionRecord,
        ActionStatus,
        ActionType,
        CoreThoughtRecord,
        EdgeRecord,
        EdgeType,
        KnowledgeSource,
        LifecycleStatus,
        Priority,
        SqliteEngravaCore,
        ThoughtType,
        VerificationStatus,
    )

    info["engrava_file"] = engrava.__file__

    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    store = SqliteEngravaCore(conn)

    info["t0"] = time.monotonic()
    about_to_call.set()
    try:
        try:
            if op == "create_thought":
                await store.create_thought(
                    CoreThoughtRecord(
                        thought_id=_NEW_THOUGHT_ID,
                        thought_type=ThoughtType.NOTE,
                        essence="a note created under contention",
                        content="contender create_thought",
                        priority=Priority.P2,
                        lifecycle_status=LifecycleStatus.CREATED,
                        created_cycle=0,
                        updated_cycle=0,
                        source="test-suite",
                    )
                )
            elif op == "delete_thought":
                deleted = await store.delete_thought(_SEED_THOUGHT_TO_DELETE)
                if not deleted:
                    msg = "delete_thought reported no row deleted"
                    raise AssertionError(msg)
            elif op == "create_edge":
                await store.create_edge(
                    EdgeRecord(
                        edge_id=_NEW_EDGE_ID,
                        from_thought_id=_SEED_THOUGHT_A,
                        to_thought_id=_SEED_THOUGHT_B,
                        edge_type=EdgeType.DEPENDS_ON,
                        weight=0.5,
                        created_cycle=0,
                        source=KnowledgeSource.EXPERIENCE,
                    )
                )
            elif op == "delete_edge":
                deleted = await store.delete_edge(_SEED_EDGE)
                if not deleted:
                    msg = "delete_edge reported no row deleted"
                    raise AssertionError(msg)
            elif op == "create_action":
                await store.create_action(
                    ActionRecord(
                        action_id=_NEW_ACTION_ID,
                        source_thought_id=_SEED_THOUGHT_ACTION_SOURCE,
                        action_type=ActionType.CLI_OUTPUT,
                        intent="contender create_action",
                        status=ActionStatus.PLANNED,
                        verification_status=VerificationStatus.PENDING,
                    )
                )
            else:
                msg = f"unknown op: {op}"
                raise AssertionError(msg)
        finally:
            await conn.close()
    finally:
        release_after_contender.set()

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
    """Create the schema and seed every row a parametrized operation needs.

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
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()

    def _seed_thought(thought_id: str) -> CoreThoughtRecord:
        return CoreThoughtRecord(
            thought_id=thought_id,
            thought_type=ThoughtType.NOTE,
            essence="a seeded note",
            content="seed content",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=0,
            updated_cycle=0,
            source="test-suite",
        )

    for thought_id in (
        _SEED_THOUGHT_A,
        _SEED_THOUGHT_B,
        _SEED_THOUGHT_TO_DELETE,
        _SEED_THOUGHT_ACTION_SOURCE,
    ):
        await store.create_thought(_seed_thought(thought_id))

    await store.create_edge(
        EdgeRecord(
            edge_id=_SEED_EDGE,
            from_thought_id=_SEED_THOUGHT_A,
            to_thought_id=_SEED_THOUGHT_B,
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=0,
            source=KnowledgeSource.EXPERIENCE,
        )
    )

    await conn.close()


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """A real on-disk, schema-bootstrapped, WAL-enabled, pre-seeded database file."""
    path = str(tmp_path / "two_process_write_busy_wait.db")
    asyncio.run(_bootstrap_and_seed(path))
    return path


def _assert_durable(path: str, op: str) -> None:
    """Verify ``op``'s write is visible through a brand-new connection.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation was run; one of :data:`_OPERATIONS`.

    """
    conn = sqlite3.connect(path)
    try:
        if op == "create_thought":
            row = conn.execute(
                "SELECT content FROM thought WHERE thought_id = ?", (_NEW_THOUGHT_ID,)
            ).fetchone()
            assert row is not None, "create_thought's row is not durable"
            assert row[0] == "contender create_thought"
        elif op == "delete_thought":
            row = conn.execute(
                "SELECT 1 FROM thought WHERE thought_id = ?", (_SEED_THOUGHT_TO_DELETE,)
            ).fetchone()
            assert row is None, "delete_thought's delete is not durable"
        elif op == "create_edge":
            row = conn.execute("SELECT 1 FROM edge WHERE edge_id = ?", (_NEW_EDGE_ID,)).fetchone()
            assert row is not None, "create_edge's row is not durable"
        elif op == "delete_edge":
            row = conn.execute("SELECT 1 FROM edge WHERE edge_id = ?", (_SEED_EDGE,)).fetchone()
            assert row is None, "delete_edge's delete is not durable"
        elif op == "create_action":
            row = conn.execute(
                "SELECT 1 FROM action WHERE action_id = ?", (_NEW_ACTION_ID,)
            ).fetchone()
            assert row is not None, "create_action's row is not durable"
        else:
            msg = f"unknown op: {op}"
            raise AssertionError(msg)
    finally:
        conn.close()


@pytest.mark.parametrize("op", _OPERATIONS)
def test_write_unit_waits_for_a_busy_lock(db_path: str, op: str) -> None:
    """A contending write waits out the holder's lock instead of failing at once.

    Before the fix, ``create_thought`` and ``delete_thought`` failed at once
    under contention: each one's write path reads before it writes, inside
    the transaction that ``_write_readback_savepoint`` (``create_thought``
    via ``_insert_new_thought_row``'s FTS5 sync trigger reading its own
    config on insert) or ``_delete_thought_atomic`` (its own existence-check
    ``SELECT``) opened with a deferred ``BEGIN``. That read took a WAL
    snapshot the write then had to upgrade, and SQLite refused the upgrade
    while the holder process held the lock: ``SQLITE_BUSY`` at once, without
    the busy handler ever running -- confirmed by running this test against
    the pre-fix tree, where these two failed in ~2 ms. ``create_edge``,
    ``delete_edge`` and ``create_action`` already waited and succeeded even
    before the fix: nothing reads inside their unit before its write, so a
    deferred ``BEGIN`` never had a snapshot to upgrade there in the first
    place. With every one of these five write-opening units now using
    ``BEGIN IMMEDIATE``, the write lock is taken up front, through the busy
    handler -- so every operation parametrized here waits out the hold and
    succeeds.

    ``update_thought`` originally belonged to the first group (its own
    content-changing update fires the same FTS5 sync trigger
    ``create_thought``'s insert does) and, for one workstream, waited here
    too. A later, narrower-scoped workstream reverted ``update_thought``'s
    own unit to a deferred ``BEGIN`` on purpose: measured against
    engrava-validation's full multiprocess suite, having it wait here turned
    a *documented* fail-fast-under-contention contract (a caller retries a
    typed ``WriteContentionError``) into a wait that could read a revision
    before another process's disjoint-column edit and reject it as falsely
    stale. ``update_thought`` is intentionally absent from
    :data:`_OPERATIONS` now; see
    ``tests/test_begin_immediate_contention_is_typed.py`` for its own,
    opposite pinning — contention fails fast there, typed, not this
    module's "waits and succeeds".
    """
    ctx = multiprocessing.get_context("spawn")
    holding = ctx.Event()
    about_to_call = ctx.Event()
    rendezvous_failed = ctx.Event()
    holder = ctx.Process(
        target=_holder,
        args=(
            db_path,
            holding,
            about_to_call,
            None,
            rendezvous_failed,
            _HOLD_SECONDS,
            _ABOUT_TO_CALL_TIMEOUT_SECONDS,
            _RELEASE_SAFETY_CEILING_SECONDS,
        ),
    )
    holder.start()
    try:
        assert holding.wait(_JOIN_TIMEOUT_SECONDS), (
            "holder process never signalled it holds the lock"
        )

        release_after_contender = ctx.Event()
        result_queue: multiprocessing.Queue[_ContenderResult] = ctx.Queue()
        contender = ctx.Process(
            target=_contender_worker,
            args=(
                db_path,
                op,
                _CONTENDER_BUSY_TIMEOUT_MS,
                about_to_call,
                release_after_contender,
                result_queue,
            ),
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
    assert not rendezvous_failed.is_set(), (
        "holder's own bounded wait for the rendezvous timed out -- the lock may "
        "already have been released before the contender made its call, so this "
        "run's timing proves nothing"
    )

    assert result["engrava_file"] == _PARENT_ENGRAVA_FILE, (
        f"contender imported engrava from {result['engrava_file']!r}, "
        f"the parent test process imported it from {_PARENT_ENGRAVA_FILE!r}"
    )

    assert result["kind"] == "ok", (
        f"{op} failed under contention: {result['error']} after {result['elapsed']}s"
    )
    elapsed = result["elapsed"]
    assert elapsed is not None
    assert elapsed >= _MIN_WAIT_SECONDS, (
        f"{op} returned after only {elapsed:.3f}s -- too fast to have waited out "
        f"the {_HOLD_SECONDS}s hold; this is the pre-fix 'fails at once' signature"
    )

    _assert_durable(db_path, op)


def test_write_unit_with_zero_busy_timeout_fails_at_once(db_path: str) -> None:
    """A negative control: with ``busy_timeout=0`` the contender still fails at once.

    Neither the pre-fix deferred ``BEGIN`` nor the fixed ``BEGIN IMMEDIATE``
    waits when the busy timeout itself is zero, so this outcome is the same
    on both sides of the fix. Its purpose is methodological: it proves the
    success and elapsed-time assertions in
    :func:`test_write_unit_waits_for_a_busy_lock` are actually driven by
    ``PRAGMA busy_timeout`` waiting out a real, held lock, and not by some
    other reason that test could pass for -- the holder never truly taking
    the lock, or a bounded retry hidden somewhere in the write path.

    The holder here does not sleep a fixed duration: it holds until the
    contender itself reports an outcome (bounded by a safety ceiling), so a
    correct near-instant failure can never race a release -- see
    :func:`_holder`.
    """
    ctx = multiprocessing.get_context("spawn")
    holding = ctx.Event()
    about_to_call = ctx.Event()
    release_after_contender = ctx.Event()
    rendezvous_failed = ctx.Event()
    holder = ctx.Process(
        target=_holder,
        args=(
            db_path,
            holding,
            about_to_call,
            release_after_contender,
            rendezvous_failed,
            _HOLD_SECONDS,
            _ABOUT_TO_CALL_TIMEOUT_SECONDS,
            _RELEASE_SAFETY_CEILING_SECONDS,
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
            args=(
                db_path,
                "create_thought",
                0,
                about_to_call,
                release_after_contender,
                result_queue,
            ),
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
    assert not rendezvous_failed.is_set(), (
        "holder's own bounded wait for the rendezvous timed out -- the lock's "
        "state at the moment of the contender's call is not known, so this "
        "run's negative-control result proves nothing"
    )

    assert result["engrava_file"] == _PARENT_ENGRAVA_FILE, (
        f"contender imported engrava from {result['engrava_file']!r}, "
        f"the parent test process imported it from {_PARENT_ENGRAVA_FILE!r}"
    )

    assert result["kind"] == "error", (
        "create_thought succeeded with busy_timeout=0 while the holder still "
        "held the write lock -- the holder never actually took it"
    )
    error = str(result["error"])
    assert "locked" in error.lower() or "busy" in error.lower(), (
        f"unexpected failure shape for a busy timeout of zero: {error}"
    )
    elapsed = result["elapsed"]
    assert elapsed is not None
    assert elapsed < _IMMEDIATE_FAILURE_MAX_SECONDS, (
        f"create_thought took {elapsed:.3f}s to fail with busy_timeout=0 -- "
        f"expected an immediate failure well under the holder's {_HOLD_SECONDS}s hold"
    )

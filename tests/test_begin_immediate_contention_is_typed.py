"""Contention on an update path's own guarded write is typed, never a stale race.

``update_thought``, ``restore_thought``, ``update_edge`` and ``update_action``
each open their unit with a deferred ``BEGIN`` and read the row before
opening it -- see ``tests/test_two_process_write_busy_wait.py`` for why:
these four carry a documented contract a caller retries on
``WriteContentionError``, which reading under the write lock (as a
since-reverted workstream tried) would have silently turned into a wait that
could reject a disjoint-column edit as falsely stale instead.

**The four do not all fail the same way, and this module is deliberately
built so that difference cannot leak into a false pass.** ``update_thought``
has an extra, narrower mechanism on top: a content-changing update fires the
FTS5 sync trigger, which reads its own config as part of the same ``UPDATE``
statement; that makes the deferred transaction's write-lock upgrade fail at
once, ``SQLITE_BUSY`` without the busy handler ever running (confirmed
against this tree: consistently within a few milliseconds, independent of
``PRAGMA busy_timeout``) -- see
``tests/test_two_process_write_busy_wait.py``'s own docstring, which
documents the identical mechanism for ``create_thought``. ``restore_thought``,
``update_edge`` and ``update_action`` have no such trigger: their guarded
``UPDATE`` goes through the *ordinary* busy handler like any other write, and
waits up to ``PRAGMA busy_timeout`` for the lock -- confirmed empirically
(a direct probe against this tree, not inferred): given a hold long enough to
outlast that wait, all three resolve only once ``busy_timeout`` itself is
exhausted, not sooner. Left with a *generous* ``busy_timeout``, that wait
could run long enough for the holder to release and commit first, at which
point these three would re-discover a genuinely changed ``revision`` and
raise ``StaleDataError`` -- correctly, not falsely, but not the typed,
promptly-retryable shape this module pins. This module closes that
ambiguity by giving the contender a ``busy_timeout`` **shorter than the
holder's own hold**, so every one of the four is guaranteed to settle -- by
whichever mechanism actually applies to it -- while the holder still holds
the lock: never in the race window where a real ``StaleDataError`` could
occur instead.

This module races real ``multiprocessing.Process`` workers against one
on-disk database file, reusing ``tests/test_two_process_write_busy_wait.py``'s
own deterministic rendezvous and ``_reap`` forced-stop cleanup -- see that
module's own docstring for why a real process, not an in-process second
connection, is what exercises this shape.

**The holder releases only once the contender has reported an outcome, not
after a fixed sleep -- on the path where that reporting actually happens.**
A fixed hold cannot actually guarantee the contender's call has *started*
contending before the holder gives up the lock: on a slow or
scheduling-delayed run, the contender could still be mid-spawn when a fixed
sleep elapses, so the holder would release, and commit, before the guarded
call ever reached the statement this module means to contend on -- passing
the assertions below for the wrong reason, or flaking under load depending
on exactly how late the guarded call started. Mirroring engrava-validation's
own C7(b) probe, the holder instead waits (bounded by a generous safety
ceiling, since a correct run resolves in milliseconds) for the contender to
signal that its call has settled -- success or failure alike -- and only
then commits. On that path, the guarded call is then provably made, and
settled, while the holder still holds the lock, not merely likely to have
been. The bound is a safety ceiling, not a second guarantee, though: if the
contender never signals at all (a crash, a hang, or a delay past the
ceiling), the holder commits anyway, without ever learning the outcome --
that path is caught downstream instead, via ``rendezvous_failed``, which the
test asserts was never set, rather than trusted as if it proved the same
thing the reporting path does.
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

#: The contender's own ``PRAGMA busy_timeout`` -- short on purpose (see the
#: module docstring): this forces even the three operations whose contention
#: goes through the *ordinary* busy handler (``restore_thought``,
#: ``update_edge``, ``update_action``) to give up and raise
#: ``WriteContentionError`` promptly, rather than tying up this test with a
#: long wait -- the holder now releases only once the contender reports an
#: outcome, so nothing here depends on outrunning a fixed hold any more.
_CONTENDER_BUSY_TIMEOUT_MS = 200

#: Ceiling on the contending call's elapsed time -- a comfortable multiple
#: of ``_CONTENDER_BUSY_TIMEOUT_MS`` so a busy-handler wait that legitimately
#: uses its whole budget still passes, while still bounding this to a small,
#: fixed fraction of a second: a call that instead hung, or that somehow
#: still waited for the holder to release, cannot be mistaken for either
#: settling shape this module pins.
_MAX_FAIL_SECONDS = (_CONTENDER_BUSY_TIMEOUT_MS / 1000) * 5

#: Bound on waiting for the contender's "about to call" signal, and (as a
#: safety ceiling) on the holder's own wait for that same signal.
_ABOUT_TO_CALL_TIMEOUT_SECONDS = 15.0

#: Safety ceiling on the holder's own wait for the contender's "I have an
#: outcome" signal, in place of a fixed hold -- see the module docstring for
#: why a fixed hold cannot actually guarantee the ordering this module
#: needs. A correct run resolves in well under a second; this bound only
#: guards against the contender hanging before it ever reports.
_RELEASE_SAFETY_CEILING_SECONDS = 15.0

#: Bound on every ``Process.join`` / ``Queue.get`` below that is not itself
#: one of the rendezvous signals above.
_JOIN_TIMEOUT_SECONDS = _RELEASE_SAFETY_CEILING_SECONDS + 20.0

#: Grace period given to a process after ``terminate()`` (and again after
#: ``kill()``) before :func:`_reap` gives up waiting for it to actually exit.
_FORCE_STOP_GRACE_SECONDS = 5.0

_THOUGHT_ID = "seed-thought-to-update"
_ARCHIVED_THOUGHT_ID = "seed-thought-to-restore"
_EDGE_ID = "seed-edge-to-update"
_EDGE_FROM = "seed-thought-edge-from"
_EDGE_TO = "seed-thought-edge-to"
_ACTION_ID = "seed-action-to-update"
_ACTION_SOURCE_THOUGHT = "seed-thought-action-source"

_SEED_ESSENCE = "essence written when the row was seeded"
_HOLDER_ESSENCE = "essence written by the concurrent holder"
_CONTENDER_CONTENT = "content written by the retried update"

_SEED_WEIGHT = 0.25
_HOLDER_DECAY = 0.42
_CONTENDER_WEIGHT = 0.9

_SEED_VERIFICATION_STATUS = "PENDING"
_HOLDER_VERIFICATION_STATUS = "PARTIAL"
_CONTENDER_ACTION_STATUS = "EXECUTING"

#: The four update paths named in the workstream's acceptance criteria.
_OPERATIONS = ("update_thought", "restore_thought", "update_edge", "update_action")

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
    about_to_call: multiprocessing.synchronize.Event,
    release_after_contender: multiprocessing.synchronize.Event,
    rendezvous_failed: multiprocessing.synchronize.Event,
    about_to_call_timeout_seconds: float,
    release_timeout_seconds: float,
) -> None:
    """Take the write lock with a real edit to a disjoint column, and hold it.

    A raw ``sqlite3`` connection, deliberately outside the engrava core --
    this process plays the role of "some other writer". Mirrors
    ``tests/test_two_process_read_after_the_wait.py``'s own holder: the
    ``UPDATE`` targets a column the contender's own call never touches, so a
    later retry landing both edits is a meaningful check, not a coincidence.

    Waits (bounded) on ``about_to_call``, then (also bounded, by a generous
    safety ceiling rather than a fixed sleep -- see the module docstring for
    why) on ``release_after_contender``, before committing either way. When
    the contender does signal within both bounds, this process has not
    released the lock before the contender's own call settled, so the
    guarded call is provably made -- and settled -- while this process still
    held it. When either bounded wait instead times out, this process
    commits anyway, with no idea what the contender's outcome was --
    ``rendezvous_failed`` records that instead of leaving it silent, so the
    caller can fail the run on that specific cause rather than trust a
    "provably made" outcome this path never actually delivers.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation the contender will run; selects which row and
            column this holder edits.
        holding: Set once the write lock is held and the edit has been made.
        about_to_call: Waited on (bounded) before the hold begins; set by the
            contender just before it makes its guarded call.
        release_after_contender: Waited on (bounded), instead of a fixed
            sleep, once ``about_to_call`` fires; set by the contender once
            its guarded call has settled, success or failure alike.
        rendezvous_failed: Set when either bounded wait above times out, so
            the parent can fail the test on that specific cause instead of
            trusting a timing result the rendezvous never actually
            guaranteed.
        about_to_call_timeout_seconds: Bound on waiting for ``about_to_call``.
        release_timeout_seconds: Bound on waiting for
            ``release_after_contender``.

    """
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("BEGIN IMMEDIATE")
    if op in ("update_thought", "restore_thought"):
        target_id = _THOUGHT_ID if op == "update_thought" else _ARCHIVED_THOUGHT_ID
        conn.execute(
            "UPDATE thought SET essence = ?, revision = revision + 1 WHERE thought_id = ?",
            (_HOLDER_ESSENCE, target_id),
        )
    elif op == "update_edge":
        conn.execute(
            "UPDATE edge SET decay_multiplier = ?, revision = revision + 1 WHERE edge_id = ?",
            (_HOLDER_DECAY, _EDGE_ID),
        )
    elif op == "update_action":
        conn.execute(
            "UPDATE action SET verification_status = ?, revision = revision + 1 "
            "WHERE action_id = ?",
            (_HOLDER_VERIFICATION_STATUS, _ACTION_ID),
        )
    else:
        msg = f"unknown op: {op}"
        raise AssertionError(msg)
    holding.set()
    about_to_call_ok = about_to_call.wait(about_to_call_timeout_seconds)
    release_ok = about_to_call_ok and release_after_contender.wait(release_timeout_seconds)
    if not about_to_call_ok or not release_ok:
        rendezvous_failed.set()
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
    """Process entry point: connect fresh, then make one guarded call.

    Mirrors ``test_two_process_write_busy_wait.py``'s own
    ``_contender_worker`` -- see that module for why the outcome is reported
    through a queue rather than the process exit code.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation to run; one of :data:`_OPERATIONS`.
        busy_timeout_ms: The value this process sets for its own
            ``PRAGMA busy_timeout`` before running ``op``.
        about_to_call: Set just before the guarded call is made.
        release_after_contender: Set once the guarded call has settled,
            success or failure alike -- see :func:`_holder`, the one waiter.
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


async def _call(op: str, store: object) -> None:
    """Make the one guarded call ``op`` names.

    Args:
        op: Which operation to run.
        store: A :class:`~engrava.SqliteEngravaCore` (typed ``object`` here
            only so this helper can be shared between the child-process
            import and the parent-process retry without importing engrava
            at module scope).

    """
    from engrava import ActionStatus, SqliteEngravaCore

    assert isinstance(store, SqliteEngravaCore)
    if op == "update_thought":
        await store.update_thought(_THOUGHT_ID, content=_CONTENDER_CONTENT)
    elif op == "restore_thought":
        await store.restore_thought(_ARCHIVED_THOUGHT_ID)
    elif op == "update_edge":
        await store.update_edge(_EDGE_ID, weight=_CONTENDER_WEIGHT)
    elif op == "update_action":
        await store.update_action(_ACTION_ID, status=ActionStatus.EXECUTING)
    else:
        msg = f"unknown op: {op}"
        raise AssertionError(msg)


async def _contender_async(  # noqa: PLR0917 - mirrors _contender_worker's own parameter list
    path: str,
    op: str,
    busy_timeout_ms: int,
    about_to_call: multiprocessing.synchronize.Event,
    release_after_contender: multiprocessing.synchronize.Event,
    info: dict[str, object],
) -> _ContenderResult:
    """Connect, then make the one guarded call named by ``op``.

    ``release_after_contender`` is set in its own outermost ``finally``,
    independent of closing the connection: the holder waiting on it must be
    released however the guarded call *and* the close each end, including a
    close that itself raises -- which is also why the close gets its own,
    inner ``finally`` rather than sharing one statement with the event: a
    failure closing it must not by itself suppress the release signal.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation to run.
        busy_timeout_ms: The value to set for ``PRAGMA busy_timeout``.
        about_to_call: Set immediately before the guarded call.
        release_after_contender: Set once the guarded call has settled,
            success or failure alike -- see :func:`_holder` for the waiter.
        info: Mutable carrier for ``engrava_file`` and ``t0``, read by the
            caller regardless of whether this coroutine raises.

    Returns:
        A :class:`_ContenderResult` with ``kind="ok"``.

    Raises:
        BaseException: Whatever the guarded operation itself raised (e.g.
            ``WriteContentionError`` while the holder still holds the lock).

    """
    import engrava
    from engrava import SqliteEngravaCore

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
            await _call(op, store)
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
    """Create the schema and seed every row a parametrized case needs.

    Args:
        path: Path to the (not yet existing) SQLite database file.

    """
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

    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()

    def _seed_thought(thought_id: str, **overrides: object) -> CoreThoughtRecord:
        params: dict[str, object] = {
            "thought_id": thought_id,
            "thought_type": ThoughtType.NOTE,
            "essence": _SEED_ESSENCE,
            "content": "seed content",
            "priority": Priority.P2,
            "lifecycle_status": LifecycleStatus.CREATED,
            "created_cycle": 0,
            "updated_cycle": 0,
            "source": "test-suite",
        }
        params.update(overrides)
        return CoreThoughtRecord(**params)  # type: ignore[arg-type]

    for thought_id in (_THOUGHT_ID, _EDGE_FROM, _EDGE_TO, _ACTION_SOURCE_THOUGHT):
        await store.create_thought(_seed_thought(thought_id))
    await store.create_thought(
        _seed_thought(_ARCHIVED_THOUGHT_ID, lifecycle_status=LifecycleStatus.ARCHIVED)
    )

    await store.create_edge(
        EdgeRecord(
            edge_id=_EDGE_ID,
            from_thought_id=_EDGE_FROM,
            to_thought_id=_EDGE_TO,
            edge_type=EdgeType.ASSOCIATED,
            weight=_SEED_WEIGHT,
            created_cycle=0,
            source=KnowledgeSource.EXPERIENCE,
        )
    )
    await store.create_action(
        ActionRecord(
            action_id=_ACTION_ID,
            source_thought_id=_ACTION_SOURCE_THOUGHT,
            action_type=ActionType.CLI_OUTPUT,
            intent="seed action for the fail-fast-then-retry test",
            status=ActionStatus.PLANNED,
            verification_status=VerificationStatus(_SEED_VERIFICATION_STATUS),
        )
    )

    await conn.close()


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """A real on-disk, schema-bootstrapped, WAL-enabled, pre-seeded database file."""
    path = str(tmp_path / "begin_immediate_contention_is_typed.db")
    asyncio.run(_bootstrap_and_seed(path))
    return path


@pytest.mark.parametrize("op", _OPERATIONS)
def test_contention_fails_fast_typed_then_a_retry_lands_both_edits(db_path: str, op: str) -> None:
    """Contention on the guarded write settles typed, while the holder still holds the lock.

    The holder changes a column disjoint from what the contender writes (and
    bumps ``revision``), then holds the write lock until the contender itself
    reports that its one guarded call has settled -- success or failure alike
    -- bounded by the generous ``_RELEASE_SAFETY_CEILING_SECONDS`` safety
    ceiling rather than a fixed sleep, mirroring engrava-validation's own
    C7(b). **This describes the ordinary path, not a guarantee that holds on
    every path.** When the contender does report in time, the holder commits
    exactly once that outcome is known, so the guarded call is guaranteed to
    have settled *while the lock was still held*, not merely likely to have
    -- closing the flake a fixed-sleep hold would otherwise leave open, and
    removing any dependence on the two waits happening to race a particular
    way (see the module docstring for why that matters: two of the four
    mechanisms this pins would otherwise be timing-dependent in exactly the
    same way). But the safety ceiling is exactly that: if the contender never
    signals -- a crash, a hang, or scheduling delay past
    ``_ABOUT_TO_CALL_TIMEOUT_SECONDS`` or ``_RELEASE_SAFETY_CEILING_SECONDS``
    -- the holder's own bounded wait times out and it commits anyway, with no
    idea what the contender's outcome was or whether the guarded call ever
    even ran under the lock. The holder does know that its own wait timed
    out, and records it in ``rendezvous_failed``, naming which wait it was.
    The assertion below on ``rendezvous_failed`` is what turns that
    silent commit into a failed test rather than a run that trusts an outcome
    the timeout path never actually guaranteed.

    Because the call is guaranteed to settle under the lock, it must settle
    within ``_MAX_FAIL_SECONDS`` -- well under any plausible hold -- as
    ``WriteContentionError``, never ``StaleDataError`` and never a raw driver
    error. At the unfixed revision (each of these four opening its own
    ``BEGIN IMMEDIATE`` *before* it reads, instead of a deferred ``BEGIN``),
    that ``BEGIN IMMEDIATE`` is itself what contends for the holder's lock;
    with this test's short ``_CONTENDER_BUSY_TIMEOUT_MS`` it exhausts that
    budget and raises a raw, untyped ``sqlite3.OperationalError`` instead --
    never invoking this call's own guarded ``UPDATE`` (and its typed
    conversion) at all. Confirmed by running this test against that
    revision.

    Once the holder has released (this test joins it before retrying), a
    plain retry -- an ordinary call, no stale state to inherit -- succeeds,
    and the durable row carries both the holder's edit and the retry's own.
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
            op,
            holding,
            about_to_call,
            release_after_contender,
            rendezvous_failed,
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
        "holder's own bounded wait for the rendezvous timed out -- either the "
        "contender never signalled it was about to call, or it never reported its "
        "outcome within the safety ceiling -- so this run's outcome proves nothing"
    )

    assert result["engrava_file"] == _PARENT_ENGRAVA_FILE, (
        f"contender imported engrava from {result['engrava_file']!r}, "
        f"the parent test process imported it from {_PARENT_ENGRAVA_FILE!r}"
    )

    assert result["kind"] == "error", (
        f"{op} succeeded under contention -- expected it to fail fast as "
        "WriteContentionError while the holder still held the lock"
    )
    error = str(result["error"])
    assert "WriteContentionError" in error, (
        f"{op} failed under contention with {error!r}, not the typed "
        "WriteContentionError this pins -- at the unfixed revision (this unit's own "
        "BEGIN IMMEDIATE contending for the holder's lock) this is a raw, untyped "
        "sqlite3.OperationalError instead"
    )
    elapsed = result["elapsed"]
    assert elapsed is not None
    assert elapsed <= _MAX_FAIL_SECONDS, (
        f"{op} took {elapsed:.3f}s to raise WriteContentionError -- expected it to "
        f"settle by {_CONTENDER_BUSY_TIMEOUT_MS}ms of its own busy_timeout at the "
        "latest, well before the holder could ever release (the holder waits for "
        "this very call to settle before it commits); taking this long means it "
        "somehow outlasted its own busy_timeout, which risks racing a real "
        "StaleDataError instead of a typed, prompt failure"
    )

    # The holder has already committed and exited (reaped above): retry on
    # an ordinary, fresh connection -- no stale snapshot to inherit.
    asyncio.run(_retry_and_assert_both_edits_land(db_path, op))


async def _retry_and_assert_both_edits_land(path: str, op: str) -> None:
    """Retry ``op`` once on a fresh connection and assert both edits are durable.

    Args:
        path: Path to the shared SQLite database file.
        op: Which operation to retry.

    """
    from engrava import SqliteEngravaCore

    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(conn)
    try:
        await _call(op, store)
    finally:
        await conn.close()

    verify_conn = sqlite3.connect(path)
    try:
        if op == "update_thought":
            row = verify_conn.execute(
                "SELECT essence, content, revision FROM thought WHERE thought_id = ?",
                (_THOUGHT_ID,),
            ).fetchone()
            assert row is not None, "update_thought's row is not durable"
            essence, content, revision = row
            assert essence == _HOLDER_ESSENCE, (
                f"the holder's essence did not survive: got {essence!r}"
            )
            assert content == _CONTENDER_CONTENT, (
                f"the retry's content did not survive: got {content!r}"
            )
            assert revision == 2, f"expected two guarded writes to land, got revision={revision}"
        elif op == "restore_thought":
            row = verify_conn.execute(
                "SELECT essence, lifecycle_status, revision FROM thought WHERE thought_id = ?",
                (_ARCHIVED_THOUGHT_ID,),
            ).fetchone()
            assert row is not None, "restore_thought's row is not durable"
            essence, lifecycle_status, revision = row
            assert essence == _HOLDER_ESSENCE, (
                f"the holder's essence did not survive: got {essence!r}"
            )
            assert lifecycle_status == "ACTIVE", (
                f"the retry's restore did not land: got lifecycle_status={lifecycle_status!r}"
            )
            assert revision == 2, f"expected two guarded writes to land, got revision={revision}"
        elif op == "update_edge":
            row = verify_conn.execute(
                "SELECT decay_multiplier, weight, revision FROM edge WHERE edge_id = ?",
                (_EDGE_ID,),
            ).fetchone()
            assert row is not None, "update_edge's row is not durable"
            decay, weight, revision = row
            assert decay == _HOLDER_DECAY, (
                f"the holder's decay_multiplier did not survive: got {decay!r}"
            )
            assert weight == _CONTENDER_WEIGHT, (
                f"the retry's weight did not survive: got {weight!r}"
            )
            assert revision == 2, f"expected two guarded writes to land, got revision={revision}"
        elif op == "update_action":
            row = verify_conn.execute(
                "SELECT verification_status, status, revision FROM action WHERE action_id = ?",
                (_ACTION_ID,),
            ).fetchone()
            assert row is not None, "update_action's row is not durable"
            verification_status, status, revision = row
            assert verification_status == _HOLDER_VERIFICATION_STATUS, (
                f"the holder's verification_status did not survive: got {verification_status!r}"
            )
            assert status == _CONTENDER_ACTION_STATUS, (
                f"the retry's status did not survive: got {status!r}"
            )
            assert revision == 2, f"expected two guarded writes to land, got revision={revision}"
        else:
            msg = f"unknown op: {op}"
            raise AssertionError(msg)
    finally:
        verify_conn.close()

"""Two real OS processes racing identical content must not duplicate a memory.

``_dedup_lock`` (an ``asyncio.Lock``) is constructed per store *instance*, so it
protects concurrent tasks sharing one connection but nothing beyond it. Two
separate processes — each with its own connection to the same database file —
could both probe ``content_hash``, both find nothing, and both insert unless
something serialises the window across connections: a duplicate memory that
then competes with its twin in retrieval and skews the dreaming frequency
signal. ``BEGIN IMMEDIATE`` around the probe-and-insert window is what
serialises them.

This module races real ``multiprocessing.Process`` workers (not asyncio tasks
in one process) against one on-disk database file to exercise exactly the
boundary the in-process lock cannot reach. This file asserts one row, one
winner.

**Why a second, bounded barrier sits inside the probe (not just before the
call).** A rendezvous only *before* ``get_or_create``/``bulk_store`` would
leave the interleaving of the probe-and-insert windows to host scheduling, so
a run could converge on one row without the workers ever having raced.
``_RacingStore`` below adds a second rendezvous *inside* the probe itself, so
a worker waits there, for at most a fixed time, for its siblings before
deciding hit-or-miss.

That second barrier cannot be a plain (unbounded) one, or the store would
deadlock the whole test: with ``BEGIN IMMEDIATE`` in place, only one worker at
a time can even reach the probe — everyone else is still parked inside
``_begin_dedup_write_lock``, waiting for the write lock. An unbounded barrier
would have that one worker wait forever for siblings that can never arrive
until it finishes and releases the lock. Giving the barrier a short timeout
gives it two outcomes:

* If all ten workers call ``wait()`` within the first arrival's timeout, the
  barrier is satisfied, and they are held at the same instant right before the
  insert-or-bump decision, then released together.
* If the write lock lets only the lock's current holder reach the probe at a
  given moment, its wait times out (``BrokenBarrierError``, caught and
  ignored), it proceeds alone, and the next worker takes its turn once the
  transaction closes.

**Why a timeout there is not, by itself, a pass/fail signal — and what is.**
A ``BrokenBarrierError`` on *every* worker is the *expected*, correct signature
of a serialised race: the write lock makes it structurally impossible for two
workers to ever be inside the probe at once, so full attendance can never
happen there, timeout or not. That means the probe barrier's own outcome
cannot tell a correctly-serialised race apart from a race that never actually
happened — which is the false green to guard against: if the OS scheduler
fails to run the ten worker processes anywhere near simultaneously (a starved
host), the workers that arrive late reach the probe after earlier ones have
already broken the barrier and inserted, so the run can converge on one row
whether or not the cross-connection lock is present — a pass that proves
nothing.

The precondition this test actually needs is not "the probe barrier
succeeded" but "the ten workers were given a fair chance to race at all" —
and *that* is measurable independently of the mechanism under test. Every
worker records a monotonic timestamp the instant it is released from the
first, unbounded, pre-call barrier (before it does anything the lock could
possibly affect), and the test asserts these timestamps land within
``_FAIR_RELEASE_SPREAD_MAX_SECONDS`` of each other. A spread inside that
budget means the scheduler handed all ten workers their turn close enough
together that the probe barrier's ``_PROBE_SYNC_TIMEOUT_SECONDS`` window had a
genuine chance to gather them; a spread outside it means the host could not
deliver a fair race this run, and the test fails loudly with that stated as
the reason — an environment problem, not a verdict on the locking — rather than
silently reporting whatever row count a staggered, non-racing run happened to
produce.

**What this precondition proves, and what it does not.** It proves the OS
scheduler gave all ten worker *processes* a fair, near-simultaneous chance to
*start* their guarded call — which closes off the failure mode in which a
whole worker takes arbitrarily long just to fork, connect and reach the
starting line, so only stragglers ever probe. It does **not** prove every
worker actually reached the probe: after the timestamp is recorded, each
worker still runs a short, constant stretch of pure-Python, no-I/O code
(metadata validation, an uncontended ``asyncio.Lock`` acquire, an
``in_transaction`` check) before its first real ``await`` — the call that
opens ``BEGIN IMMEDIATE`` and is where the mechanism under test actually
serialises them. Nothing forces the OS to schedule a *process* fairly across
that stretch either; a kernel scheduler under severe enough pressure could in
principle preempt a worker mid-instruction there, same as anywhere else in
the program, and this test has no way to see that if it happens. This residual
gap is real but categorically smaller than the one this precondition closes:
it is a few microseconds of synchronous Python on every run, not a
process-startup cost that scales with host load. Treat a pass here as "the
locking holds under a fair race on this host", not as a formal proof that
OS-level process scheduling was fair down to the instruction.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
    ThoughtVisibility,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Number of processes racing the same content.
_FAN_OUT = 10

#: Bound on how long a worker waits, inside the content-hash probe, for its
#: siblings to reach the same point (see the module docstring). Long enough
#: for sibling processes to actually land the rendezvous on a loaded CI box;
#: short enough that the store's one-at-a-time serialisation only adds
#: this once per race (the first waiter's timeout breaks the barrier for
#: everyone after it — see ``threading.Barrier``).
_PROBE_SYNC_TIMEOUT_SECONDS = 2.0

#: Maximum spread, in seconds, allowed between the earliest and the latest
#: worker's release from the pre-call barrier (see the module docstring's
#: "Why a timeout there is not, by itself, a pass/fail signal" section). This
#: is the fair-race precondition: comfortably smaller than
#: ``_PROBE_SYNC_TIMEOUT_SECONDS`` so that satisfying it implies the probe
#: barrier's window had a genuine chance to gather the workers, and generous
#: relative to plain OS-scheduling jitter for waking ten already-forked,
#: already-connected processes (which is sub-millisecond in practice on this
#: host). A run whose spread exceeds this fails outright rather than silently
#: reporting a row count that a staggered, non-racing run happened to produce.
_FAIR_RELEASE_SPREAD_MAX_SECONDS = 1.0

_RACE_CONTENT = "Two independent sessions observed the same fact about the user."
_BULK_RACE_CONTENT = "Two independent ingest batches observed the same fact."


def _thought(thought_id: str, *, content: str = _RACE_CONTENT) -> CoreThoughtRecord:
    """Build a ``CoreThoughtRecord`` with shared content for the race."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="Race probe",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


class _RacingStore(SqliteEngravaCore):
    """A store whose probe waits, for at most a fixed time, for the other workers.

    Overrides the one method every dedup entry point (``create_thought``,
    ``get_or_create``, ``upsert_by_hash``, and — through ``create_thought`` —
    ``bulk_store``) calls to read the existing row, and adds a bounded
    rendezvous immediately after it returns: a worker waits here, for at most
    a timeout, for its siblings to wait here too *before* acting on what the
    probe saw. See the module docstring for why the rendezvous
    must be bounded rather than a plain barrier.
    """

    def __init__(
        self,
        connection: aiosqlite.Connection,
        probe_barrier: multiprocessing.synchronize.Barrier,
    ) -> None:
        super().__init__(connection)
        self._probe_barrier = probe_barrier

    async def _get_thought_by_content_hash(self, content_hash: str) -> CoreThoughtRecord | None:
        result = await super()._get_thought_by_content_hash(content_hash)
        loop = asyncio.get_running_loop()
        with contextlib.suppress(threading.BrokenBarrierError):
            await loop.run_in_executor(
                None,
                self._probe_barrier.wait,
                _PROBE_SYNC_TIMEOUT_SECONDS,
            )
        return result


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """A real on-disk, schema-bootstrapped, WAL-enabled database file.

    A plain synchronous fixture (bootstrapping via ``asyncio.run`` internally)
    so it composes cleanly with the synchronous, ``multiprocessing``-driven
    test below rather than depending on pytest-asyncio's fixture-injection
    behaviour for a sync test.
    """
    path = str(tmp_path / "two_process_dedup.db")

    async def _bootstrap() -> None:
        conn = await aiosqlite.connect(path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await conn.close()

    asyncio.run(_bootstrap())
    return path


def _race_worker(
    db_path: str,
    worker_index: int,
    barrier: multiprocessing.synchronize.Barrier,
    probe_barrier: multiprocessing.synchronize.Barrier,
    result_queue: multiprocessing.Queue[tuple[str, str, float | None]],
) -> None:
    """Process entry point: open a fresh connection, then race ``get_or_create``.

    Runs in its own process with its own event loop and its own SQLite
    connection to *db_path* — the two boundaries the in-process
    ``asyncio.Lock`` cannot see across. Every worker connects and reaches the
    barrier before any of them calls ``get_or_create``, so the actual
    probe-and-insert windows start as close to simultaneously as the OS
    scheduler allows; ``probe_barrier`` (see ``_RacingStore``) adds a bounded
    wait at the probe. The outcome (or a failure) is
    reported back through ``result_queue`` rather than via the process exit
    code, so the parent can tell a clean "one winner, rest converged" run apart
    from a worker that raised. ``timing`` is mutated by ``_race_worker_async``
    the instant the pre-call barrier releases this worker, before anything the
    lock could affect runs — read here regardless of whether the guarded call
    went on to succeed or raise, so the parent can check the fair-race
    precondition (see the module docstring) even for a worker that errored.
    """
    timing: dict[str, float] = {}
    try:
        thought_id = asyncio.run(
            _race_worker_async(db_path, worker_index, barrier, probe_barrier, timing),
        )
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        result_queue.put(("error", f"{type(exc).__name__}: {exc}", timing.get("release_time")))
    else:
        result_queue.put(("ok", thought_id, timing["release_time"]))


async def _race_worker_async(
    db_path: str,
    worker_index: int,
    barrier: multiprocessing.synchronize.Barrier,
    probe_barrier: multiprocessing.synchronize.Barrier,
    timing: dict[str, float],
) -> str:
    """Connect, synchronise with siblings, then run the guarded call."""
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute("PRAGMA busy_timeout=5000")
    store = _RacingStore(conn, probe_barrier)
    await store._probe_fts()  # query-strategy cache is per instance

    # ``Barrier.wait`` is a blocking call; running it in the default executor
    # keeps this coroutine well-behaved even though it is the only thing this
    # worker's event loop is doing.
    await asyncio.get_running_loop().run_in_executor(None, barrier.wait)
    # Recorded immediately on release, before the guarded call -- the
    # fair-race precondition this timestamp feeds is about the scheduler, not
    # about anything the lock under test could influence.
    timing["release_time"] = time.monotonic()

    record, _created = await store.get_or_create(_thought(f"t-race-{worker_index}"))
    await conn.close()
    return record.thought_id


def test_two_processes_racing_get_or_create_converge_on_one_row(db_path: str) -> None:
    """``_FAN_OUT`` processes racing identical content leave exactly one row.

    Every worker calls ``get_or_create`` with byte-identical ``content``, and
    ``_RacingStore`` makes each worker wait at the probe, for at most a
    bounded time, for its siblings (see the module docstring). If the
    probe-and-insert window were only serialised in-process, workers that
    probe before any insert lands would each observe "no match" and each
    insert. With ``BEGIN IMMEDIATE`` around the window, SQLite itself serialises the
    competing transactions, so only the first insert lands and every later
    worker's probe observes it and bumps ``confirmation_count`` instead.
    """
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(_FAN_OUT)
    probe_barrier = ctx.Barrier(_FAN_OUT)
    result_queue: multiprocessing.Queue[tuple[str, str, float | None]] = ctx.Queue()
    processes = [
        ctx.Process(target=_race_worker, args=(db_path, i, barrier, probe_barrier, result_queue))
        for i in range(_FAN_OUT)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0, (
            f"worker process {process.pid} did not exit cleanly (exitcode={process.exitcode})"
        )

    outcomes = [result_queue.get(timeout=5) for _ in processes]
    errors = [payload for kind, payload, _release_time in outcomes if kind == "error"]
    assert not errors, f"worker(s) raised: {errors}"

    # Fair-race precondition (see the module docstring): a probe-barrier
    # timeout is expected and proves nothing by itself, so the thing actually
    # checked here is that the OS scheduler gave every worker its turn close
    # enough together for that window to matter. Checked before trusting the
    # row-count assertion below, not after.
    release_times_raw = [release_time for _kind, _payload, release_time in outcomes]
    assert all(t is not None for t in release_times_raw), (
        "a worker never reached the pre-call barrier (see the error above)"
    )
    release_times: list[float] = [t for t in release_times_raw if t is not None]
    spread = max(release_times) - min(release_times)
    assert spread <= _FAIR_RELEASE_SPREAD_MAX_SECONDS, (
        f"workers were released {spread:.3f}s apart from the pre-call barrier "
        f"(limit {_FAIR_RELEASE_SPREAD_MAX_SECONDS}s) -- the host could not "
        "schedule them close enough together for this race to prove anything "
        "about the locking. This is an environment problem, not a defect -- "
        "rerun on a less loaded host."
    )

    thought_ids = {payload for kind, payload, _release_time in outcomes if kind == "ok"}
    assert len(thought_ids) == 1, (
        f"workers converged on {len(thought_ids)} distinct thought ids "
        f"(expected exactly 1): {thought_ids}"
    )

    conn = sqlite3.connect(db_path)
    try:
        # Every worker submits its own candidate ``thought_id`` for the same
        # ``content``, so a bug that let more than one insert land would show
        # up as more than one *row* even where (as here) the returned ids
        # happened to agree — this is the count that actually discriminates,
        # not the id-agreement check above. This test only ever races one
        # piece of content, so a plain table-wide count is precise.
        total_rows = conn.execute("SELECT COUNT(*) FROM thought").fetchone()[0]
        confirmation_count = conn.execute(
            "SELECT confirmation_count FROM thought WHERE thought_id = ?",
            (next(iter(thought_ids)),),
        ).fetchone()[0]
    finally:
        conn.close()

    assert total_rows == 1
    # The one surviving row saw every other worker as a confirmation.
    assert confirmation_count == _FAN_OUT - 1


def _bulk_race_worker(
    db_path: str,
    worker_index: int,
    barrier: multiprocessing.synchronize.Barrier,
    probe_barrier: multiprocessing.synchronize.Barrier,
    result_queue: multiprocessing.Queue[tuple[str, str, float | None]],
) -> None:
    """Process entry point: open a fresh connection, then race ``bulk_store``.

    Mirrors ``_race_worker`` exactly, except the guarded call is
    ``bulk_store(deduplicate=True)``, which runs its insert loop inside one
    ``suspend_auto_commit`` window. See ``_race_worker`` for why ``timing`` is
    read regardless of success or failure.
    """
    timing: dict[str, float] = {}
    try:
        thought_id = asyncio.run(
            _bulk_race_worker_async(db_path, worker_index, barrier, probe_barrier, timing),
        )
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        result_queue.put(("error", f"{type(exc).__name__}: {exc}", timing.get("release_time")))
    else:
        result_queue.put(("ok", thought_id, timing["release_time"]))


async def _bulk_race_worker_async(
    db_path: str,
    worker_index: int,
    barrier: multiprocessing.synchronize.Barrier,
    probe_barrier: multiprocessing.synchronize.Barrier,
    timing: dict[str, float],
) -> str:
    """Connect, synchronise with siblings, then run the guarded bulk call."""
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute("PRAGMA busy_timeout=5000")
    store = _RacingStore(conn, probe_barrier)
    await store._probe_fts()

    await asyncio.get_running_loop().run_in_executor(None, barrier.wait)
    timing["release_time"] = time.monotonic()

    results = await store.bulk_store(
        [_thought(f"t-bulk-race-{worker_index}", content=_BULK_RACE_CONTENT)],
        deduplicate=True,
    )
    await conn.close()
    return results[0].thought_id


def test_two_processes_racing_bulk_store_dedup_converge_on_one_row(db_path: str) -> None:
    """``_FAN_OUT`` processes racing identical content via ``bulk_store`` insert one row.

    Same shape as ``test_two_processes_racing_get_or_create_converge_on_one_row``,
    but through ``bulk_store(deduplicate=True)``, whose insert loop runs inside
    one ``suspend_auto_commit`` window. The probe-and-insert window takes the
    same cross-connection write lock there too (it does not consult
    ``_skip_auto_commit``), so every worker's single-row batch is serialised
    and only one insert lands.
    """
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(_FAN_OUT)
    probe_barrier = ctx.Barrier(_FAN_OUT)
    result_queue: multiprocessing.Queue[tuple[str, str, float | None]] = ctx.Queue()
    processes = [
        ctx.Process(
            target=_bulk_race_worker,
            args=(db_path, i, barrier, probe_barrier, result_queue),
        )
        for i in range(_FAN_OUT)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0, (
            f"worker process {process.pid} did not exit cleanly (exitcode={process.exitcode})"
        )

    outcomes = [result_queue.get(timeout=5) for _ in processes]
    errors = [payload for kind, payload, _release_time in outcomes if kind == "error"]
    assert not errors, f"worker(s) raised: {errors}"

    # Fair-race precondition -- see test_two_processes_racing_get_or_create_
    # converge_on_one_row and the module docstring for why this is checked
    # before the row-count assertion means anything.
    release_times_raw = [release_time for _kind, _payload, release_time in outcomes]
    assert all(t is not None for t in release_times_raw), (
        "a worker never reached the pre-call barrier (see the error above)"
    )
    release_times: list[float] = [t for t in release_times_raw if t is not None]
    spread = max(release_times) - min(release_times)
    assert spread <= _FAIR_RELEASE_SPREAD_MAX_SECONDS, (
        f"workers were released {spread:.3f}s apart from the pre-call barrier "
        f"(limit {_FAIR_RELEASE_SPREAD_MAX_SECONDS}s) -- the host could not "
        "schedule them close enough together for this race to prove anything "
        "about the locking. This is an environment problem, not a defect -- "
        "rerun on a less loaded host."
    )

    conn = sqlite3.connect(db_path)
    try:
        total_rows = conn.execute("SELECT COUNT(*) FROM thought").fetchone()[0]
    finally:
        conn.close()

    assert total_rows == 1, (
        f"bulk_store(deduplicate=True) left {total_rows} rows across {_FAN_OUT} "
        "racing processes (expected exactly 1)"
    )

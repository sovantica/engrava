"""Tests for the connect helper that waits out a failed-connect worker thread.

aiosqlite 0.22's worker thread can still be mid-shutdown when a failed
connect's exception reaches the caller; if the caller's event loop closes
before the thread notices, the thread raises unhandled. The tests below
delay the worker so that this timing is very likely, and check the helper's
wait.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, Any

import aiosqlite
import aiosqlite.core as aiosqlite_core
import pytest

from engrava.infrastructure.sqlite import aiosqlite_connect
from engrava.infrastructure.sqlite.aiosqlite_connect import connect

if TYPE_CHECKING:
    from pathlib import Path
    from typing import NoReturn

#: How long the worker's stop-item ``get()`` call is delayed. A caller that
#: does not wait for the thread normally lets ``asyncio.run()`` close its loop
#: well within this time; the helper's wait is longer.
_RACE_DELAY_SECONDS = 0.3


def _aiosqlite_matches_the_known_race() -> bool:
    """Whether this aiosqlite build has the internals the race depends on."""
    return hasattr(aiosqlite_core, "SimpleQueue") and hasattr(aiosqlite.Connection, "stop")


def _delayed_second_get_queue_class() -> type:
    """Build a queue class whose second ``get()`` call sleeps before returning.

    aiosqlite's worker thread calls ``get()`` once for the connector, and
    again for the stop item a failed connect enqueues. Delaying the second
    call makes the worker pick up that stop item late. Built on demand, after
    the skip check, because it subclasses an aiosqlite internal.
    """

    class _DelayedSecondGetQueue(aiosqlite_core.SimpleQueue):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._get_calls = 0

        def get(
            self,
            *args: Any,  # noqa: ANN401 -- matches queue.SimpleQueue.get's own signature
            **kwargs: Any,  # noqa: ANN401 -- matches queue.SimpleQueue.get's own signature
        ) -> Any:  # noqa: ANN401 -- matches queue.SimpleQueue.get's own signature
            self._get_calls += 1
            if self._get_calls == 2:
                time.sleep(_RACE_DELAY_SECONDS)
            return super().get(*args, **kwargs)

    return _DelayedSecondGetQueue


def _run_forced_connect_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cancel_during_wait: bool = False,
) -> tuple[list[threading.ExceptHookArgs], list[BaseException], threading.Thread]:
    """Connect to a directory with the worker delayed, and report what happened.

    With *cancel_during_wait*, the connect task is cancelled once the helper
    has entered its wait for the delayed worker.
    Returns the thread exceptions captured during the run, the exception(s)
    the connect call raised, and the worker thread after a bounded join.
    """
    monkeypatch.setattr(aiosqlite_core, "SimpleQueue", _delayed_second_get_queue_class())

    created_connections: list[aiosqlite.Connection] = []
    original_connect = aiosqlite.connect

    def _spy_connect(
        *args: Any,  # noqa: ANN401 -- matches aiosqlite.connect's own signature
        **kwargs: Any,  # noqa: ANN401 -- matches aiosqlite.connect's own signature
    ) -> aiosqlite.Connection:
        conn = original_connect(*args, **kwargs)
        created_connections.append(conn)
        return conn

    monkeypatch.setattr(aiosqlite, "connect", _spy_connect)

    captured_thread_exceptions: list[threading.ExceptHookArgs] = []
    raised: list[BaseException] = []
    original_hook = threading.excepthook
    threading.excepthook = captured_thread_exceptions.append

    original_wait = aiosqlite_connect._wait_for_worker_stop
    wait_entered = asyncio.Event()

    async def _spy_wait(thread: threading.Thread | None) -> None:
        wait_entered.set()
        await original_wait(thread)

    monkeypatch.setattr(aiosqlite_connect, "_wait_for_worker_stop", _spy_wait)

    async def _main() -> None:
        task = asyncio.ensure_future(connect(str(tmp_path)))
        if cancel_during_wait:
            # Bounded, so a helper that never enters its wait fails the test
            # instead of hanging it.
            await asyncio.wait_for(wait_entered.wait(), timeout=_RACE_DELAY_SECONDS + 2.0)
            task.cancel()
        try:
            await task
        except BaseException as exc:  # noqa: BLE001 -- captured for the assertions below
            raised.append(exc)

    try:
        asyncio.run(_main())
        assert created_connections, "aiosqlite.connect was never called"
        worker = created_connections[-1]._thread
        # Join while our hook is still installed, so a worker exception raised
        # during this wait is captured as well.
        worker.join(timeout=_RACE_DELAY_SECONDS + 2.0)
    finally:
        threading.excepthook = original_hook

    return captured_thread_exceptions, raised, worker


def test_failed_connect_waits_for_the_worker_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the worker delayed, a failed connect leaves no unhandled thread exception.

    Connecting to a directory fails inside the worker thread. The helper keeps
    the loop running until the delayed worker has stopped.
    """
    if not _aiosqlite_matches_the_known_race():
        pytest.skip("aiosqlite internals do not match the known 0.22 connect-failure race")

    thread_exceptions, raised, worker = _run_forced_connect_race(tmp_path, monkeypatch)

    assert not worker.is_alive(), "worker thread did not stop within the join timeout"
    assert not thread_exceptions, f"worker thread raised unhandled: {thread_exceptions}"
    assert len(raised) == 1
    assert isinstance(raised[0], sqlite3.OperationalError)


def test_a_cancellation_during_the_wait_still_lets_the_worker_stop_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancellation that lands inside the helper's wait does not cut it short.

    The caller still receives the cancellation, but only after the delayed
    worker has stopped, so the worker never posts to a closed loop.
    """
    if not _aiosqlite_matches_the_known_race():
        pytest.skip("aiosqlite internals do not match the known 0.22 connect-failure race")

    thread_exceptions, raised, worker = _run_forced_connect_race(
        tmp_path, monkeypatch, cancel_during_wait=True
    )

    assert not worker.is_alive(), "worker thread did not stop within the join timeout"
    assert not thread_exceptions, f"worker thread raised unhandled: {thread_exceptions}"
    assert len(raised) == 1
    assert isinstance(raised[0], asyncio.CancelledError)


def test_an_older_aiosqlite_connection_is_not_waited_for() -> None:
    """A connection that is itself the worker thread (aiosqlite < 0.22) has nothing to wait for."""
    assert aiosqlite_connect._stopping_worker_thread(threading.Thread(target=lambda: None)) is None


async def test_connect_returns_a_working_connection_on_success(tmp_path: Path) -> None:
    """The helper's connection behaves like a normal, usable aiosqlite connection."""
    db_path = tmp_path / "ok.db"
    conn = await connect(str(db_path))
    try:
        cursor = await conn.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row == (1,)
    finally:
        await conn.close()


async def test_connect_reraises_the_identical_exception_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed connect re-raises the exact object the underlying await raised.

    Not merely an equal or same-typed exception: the same object, as a bare
    ``raise`` (and nothing else) produces. A connection stand-in with no
    ``_thread`` at all also checks the no-thread-found path takes it straight
    to that re-raise.
    """
    sentinel = RuntimeError("boom")

    class _FailingAwaitable:
        def __await__(self) -> NoReturn:
            raise sentinel

    monkeypatch.setattr(aiosqlite, "connect", lambda *_a, **_kw: _FailingAwaitable())

    with pytest.raises(RuntimeError) as exc_info:
        await connect("ignored")

    assert exc_info.value is sentinel

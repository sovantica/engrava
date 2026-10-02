"""One ``EngravaManager`` caller's cancellation must not break another caller.

``get_store`` shares one in-flight creation between the call that creates a
store (its *creator*) and every concurrent call waiting on the same name: a
second ``get_store``, ``delete_service`` and ``close_all``. Cancelling any one
of those calls must affect that call alone. A waiter's cancellation must not
reach the shared creation, and a creator's cancellation must not reach its
waiters, which start the creation over (``get_store``), check again
(``delete_service``) or treat it as having produced no store (``close_all``).

Each test holds a creation at a gate it controls, so the interleaving is
fixed by the test rather than by timing:

- ``ensure_schema``: the creation is inside ``_create_store`` with its
  connection open;
- ``aiosqlite.connect``: the creation is inside ``_create_store`` with nothing
  open yet;
- the manager's own lock: ``_create_store`` has returned and the store is not
  yet published.

Every wait on a task or event is bounded. A caller left waiting on a
creation that nobody resolves fails its test instead of stalling the suite.
``asyncio.wait_for`` bounds most of them. A ``close_all`` task needs
:func:`_close_all_outcome` and the ``start_close_all`` fixture instead,
because ``close_all`` keeps waiting on a creation after it is cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import time
from typing import TYPE_CHECKING

import pytest

from engrava import (
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.infrastructure import service_manager as service_manager_module
from engrava.infrastructure.service_manager import EngravaManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from pathlib import Path

    import aiosqlite

#: Upper bound for any single wait in these tests. Every scenario finishes in
#: milliseconds when it is correct, so reaching this means a caller was left
#: waiting on a creation nobody will ever resolve.
_BOUND_SECONDS = 5.0

#: What asyncio logs when a future or task is finalized with an outcome nobody
#: observed.
_UNOBSERVED_OUTCOME_PHRASES = (
    "exception was never retrieved",
    "was destroyed but it is pending",
)

_SERVICE = "shared"


class _Gate:
    """Hold the first ``hold`` calls through :meth:`pass_through` until released.

    ``entered[i]`` is set when call ``i`` reaches the gate. Calls after the
    first ``hold`` run straight through, so a waiter's retry of an abandoned
    creation is not held unless the test asks for that.
    """

    def __init__(self, *, hold: int = 1) -> None:
        self.entered = [asyncio.Event() for _ in range(hold)]
        self.release = asyncio.Event()
        self.calls = 0

    async def pass_through(self) -> None:
        index = self.calls
        self.calls += 1
        if index < len(self.entered):
            self.entered[index].set()
            await self.release.wait()


def _gate_schema_setup(monkeypatch: pytest.MonkeyPatch, *, hold: int = 1) -> _Gate:
    """Gate ``ensure_schema``, which runs inside ``_create_store`` after the connection opens."""
    gate = _Gate(hold=hold)
    real_ensure_schema = SqliteEngravaCore.ensure_schema

    async def _gated_ensure_schema(store: SqliteEngravaCore) -> None:
        await gate.pass_through()
        await real_ensure_schema(store)

    monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _gated_ensure_schema)
    return gate


def _gate_connect(monkeypatch: pytest.MonkeyPatch) -> _Gate:
    """Gate the manager's ``aiosqlite.connect``, before any database file exists."""
    gate = _Gate()
    real_connect = service_manager_module.aiosqlite.connect

    async def _gated_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
        await gate.pass_through()
        return await real_connect(*args, **kwargs)

    monkeypatch.setattr(service_manager_module.aiosqlite, "connect", _gated_connect)
    return gate


def _gate_store_close(monkeypatch: pytest.MonkeyPatch) -> _Gate:
    """Gate ``SqliteEngravaCore.close``, holding the first store close until released.

    A close cancelled while it is held still closes the store before the
    cancellation propagates, which is what the real ``close`` does too.
    """
    gate = _Gate()
    real_close = SqliteEngravaCore.close

    async def _gated_close(store: SqliteEngravaCore) -> None:
        try:
            await gate.pass_through()
        finally:
            await real_close(store)

    monkeypatch.setattr(SqliteEngravaCore, "close", _gated_close)
    return gate


def _signal_create_store_return(
    mgr: EngravaManager, monkeypatch: pytest.MonkeyPatch
) -> asyncio.Event:
    """Return an event that is set once ``mgr._create_store`` has returned a store.

    The creator runs on from that return, without suspending, until it has to
    wait for the manager's lock to publish the store. So when a test holding
    that lock wakes on this event, the creator is already parked on the lock.
    """
    returned = asyncio.Event()
    real_create_store = mgr._create_store

    async def _create_store(service_name: str, *, migrate: bool = True) -> SqliteEngravaCore:
        store = await real_create_store(service_name, migrate=migrate)
        returned.set()
        return store

    monkeypatch.setattr(mgr, "_create_store", _create_store)
    return returned


async def _let_other_tasks_run() -> None:
    """Yield to the event loop for enough turns that every runnable task reaches its next await.

    These are scheduler yields only, so there is no wall-clock race: a gated
    creation stays gated however many turns this takes, and a caller that
    finds the shared creation parks on it within its first turn.
    """
    for _ in range(10):
        await asyncio.sleep(0)


async def _close_all_outcome(closer: asyncio.Task[None]) -> None:
    """Return or raise what a ``close_all`` task ends with, failing rather than hanging.

    ``asyncio.wait_for`` cannot bound this wait. On expiry it cancels the task
    and then waits for the task to finish, and ``close_all`` deliberately keeps
    waiting on a creation from its snapshot after it is cancelled. So the task
    is waited on without being cancelled. The task comes from the
    ``start_close_all`` fixture, which releases a task still stuck when the
    test ends.
    """
    done, _ = await asyncio.wait({closer}, timeout=_BOUND_SECONDS)
    if closer not in done:
        pytest.fail("close_all never finished: it was left waiting on a creation nobody resolved")
    await closer


async def _worker_exited(conn: aiosqlite.Connection) -> bool:
    """Return whether *conn*'s aiosqlite worker thread has exited (it is not a daemon)."""
    deadline = time.monotonic() + 2.0
    while conn._thread.is_alive() and time.monotonic() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(0.01)
    return not conn._thread.is_alive()


async def _assert_store_works(store: SqliteEngravaCore) -> None:
    """Write a thought through *store* and read it back."""
    await store.create_thought(
        ThoughtRecord(
            thought_id="t-probe",
            essence="probe",
            content="written through the store a caller got back",
            thought_type=ThoughtType.OBSERVATION,
            source="test",
            lifecycle_status=LifecycleStatus.ACTIVE,
            priority=Priority.P2,
            created_cycle=1,
            updated_cycle=1,
        )
    )
    fetched = await store.get_thought("t-probe")
    assert fetched is not None
    assert fetched.essence == "probe"


@pytest.fixture(autouse=True)
def _no_unobserved_asyncio_outcome(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Fail a scenario that leaves a future or task outcome nobody observed.

    asyncio reports one when the object is finalized, which can be well after
    the scenario ends. Garbage is collected before the test, so an earlier
    test's leftovers are not reported here, and again after the test body is
    gone, so this test's own leftovers are.
    """
    gc.collect()
    caplog.set_level(logging.WARNING, logger="asyncio")
    yield
    gc.collect()
    reported = [
        record.getMessage()
        for when in ("call", "teardown")
        for record in caplog.get_records(when)
        if record.name == "asyncio"
        and any(phrase in record.getMessage() for phrase in _UNOBSERVED_OUTCOME_PHRASES)
    ]
    assert reported == []


@pytest.fixture
async def opened(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[aiosqlite.Connection]]:
    """Record every connection the manager opens, in order.

    Afterwards, stop any worker thread still running, so a failing scenario
    cannot leave a non-daemon thread behind that blocks interpreter exit.
    """
    connections: list[aiosqlite.Connection] = []
    real_connect = service_manager_module.aiosqlite.connect

    async def _recording_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
        conn = await real_connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(service_manager_module.aiosqlite, "connect", _recording_connect)
    yield connections
    for conn in connections:
        if conn._thread.is_alive():
            stopped = conn.stop()
            if stopped is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stopped, timeout=_BOUND_SECONDS)
            conn._thread.join(timeout=_BOUND_SECONDS)


@pytest.fixture
async def start_close_all() -> AsyncIterator[Callable[[EngravaManager], asyncio.Task[None]]]:
    """Start ``mgr.close_all()`` as a task; afterwards, leave none of them stuck.

    A scenario that fails early can leave ``close_all`` waiting on a creation
    nobody will resolve. ``close_all`` keeps waiting after it is cancelled, so
    the event loop's shutdown would then wait for it forever. Afterwards,
    every creation such a task can still be waiting on is cancelled directly,
    which ``close_all`` counts as a creation that produced no store, and the
    task is given the bound to finish.
    """
    started: list[tuple[EngravaManager, asyncio.Task[None]]] = []

    def _start(mgr: EngravaManager) -> asyncio.Task[None]:
        closer = asyncio.create_task(mgr.close_all())
        started.append((mgr, closer))
        return closer

    yield _start
    for mgr, closer in started:
        if not closer.done():
            for creating in mgr._creating.values():
                creating.cancel()
            await asyncio.wait({closer}, timeout=_BOUND_SECONDS)


class TestCancellationStaysWithTheCancelledCaller:
    """Callers sharing one creation, one of them cancelled (or the creation failing)."""

    async def test_cancelled_get_store_waiter_leaves_the_creator_and_other_waiters_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[aiosqlite.Connection]
    ) -> None:
        """A cancelled waiter must not cancel the creation it was waiting on.

        Awaited unshielded, the waiter's cancellation cancelled the shared
        future itself. The creator's own ``get_store`` then raised
        ``InvalidStateError`` when it published the store it had just built,
        and every other waiter raised a ``CancelledError`` nobody asked for.
        """
        gate = _gate_schema_setup(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        cancelled_waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        other_waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        await _let_other_tasks_run()
        assert not cancelled_waiter.done()
        assert not other_waiter.done()

        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled_waiter, _BOUND_SECONDS)

        gate.release.set()
        store = await asyncio.wait_for(creator, _BOUND_SECONDS)
        assert await asyncio.wait_for(other_waiter, _BOUND_SECONDS) is store
        assert mgr._stores == {_SERVICE: store}
        assert mgr._creating == {}
        assert len(opened) == 1
        await _assert_store_works(store)
        await asyncio.wait_for(mgr.close_all(), _BOUND_SECONDS)

    async def test_cancelled_creator_hands_the_creation_to_its_waiter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[aiosqlite.Connection]
    ) -> None:
        """A waiter nobody cancelled must not inherit its creator's cancellation.

        The creator resolves the shared future with ``_CreationAbandonedError``,
        not with its own ``CancelledError``, so the waiter starts the creation
        over and gets a working store from its own retry.
        """
        gate = _gate_schema_setup(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        await _let_other_tasks_run()
        assert not waiter.done()

        creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creator, _BOUND_SECONDS)

        store = await asyncio.wait_for(waiter, _BOUND_SECONDS)
        assert mgr._stores == {_SERVICE: store}
        assert mgr._creating == {}
        assert len(opened) == 2
        assert store._db is opened[1], "the waiter's store must come from its own retry"
        assert await _worker_exited(opened[0]), "the abandoned creation's connection must be closed"
        await _assert_store_works(store)
        await asyncio.wait_for(mgr.close_all(), _BOUND_SECONDS)

    async def test_retry_is_one_new_creation_per_abandoned_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[aiosqlite.Connection]
    ) -> None:
        """A retrying waiter that becomes the creator owns its own cancellation.

        The waiter that took over an abandoned creation is cancelled in turn:
        it raises its own ``CancelledError`` rather than retrying again, and
        the waiter behind it retries once and succeeds. That makes three
        connections, one per creation, and no more.
        """
        gate = _gate_schema_setup(monkeypatch, hold=2)
        mgr = EngravaManager(data_dir=tmp_path / "services")

        first_creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        second_creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await _let_other_tasks_run()

        first_creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first_creator, _BOUND_SECONDS)
        # The waiter has retried and is now creating the store itself.
        await asyncio.wait_for(gate.entered[1].wait(), _BOUND_SECONDS)

        last_waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        await _let_other_tasks_run()
        assert not last_waiter.done()

        second_creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(second_creator, _BOUND_SECONDS)

        store = await asyncio.wait_for(last_waiter, _BOUND_SECONDS)
        assert mgr._stores == {_SERVICE: store}
        assert mgr._creating == {}
        assert len(opened) == 3
        assert store._db is opened[2]
        assert await _worker_exited(opened[0])
        assert await _worker_exited(opened[1])
        await asyncio.wait_for(mgr.close_all(), _BOUND_SECONDS)

    async def test_cancelled_delete_service_leaves_the_creation_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[aiosqlite.Connection]
    ) -> None:
        """Cancelling ``delete_service`` must not cancel the creation it waits on."""
        gate = _gate_schema_setup(monkeypatch)
        data_dir = tmp_path / "services"
        mgr = EngravaManager(data_dir=data_dir)

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        deleter = asyncio.create_task(mgr.delete_service(_SERVICE))
        other_waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        await _let_other_tasks_run()
        assert not deleter.done()
        assert not other_waiter.done()

        deleter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(deleter, _BOUND_SECONDS)

        gate.release.set()
        store = await asyncio.wait_for(creator, _BOUND_SECONDS)
        assert await asyncio.wait_for(other_waiter, _BOUND_SECONDS) is store
        assert mgr._stores == {_SERVICE: store}
        assert mgr._creating == {}
        assert (data_dir / f"{_SERVICE}.db").exists(), "the cancelled delete must not have run"
        assert len(opened) == 1
        await _assert_store_works(store)
        await asyncio.wait_for(mgr.close_all(), _BOUND_SECONDS)

    @pytest.mark.parametrize(
        "database_file_written",
        [True, False],
        ids=["cancelled-with-connection-open", "cancelled-before-connecting"],
    )
    async def test_cancelled_creator_leaves_delete_service_its_documented_outcome(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        database_file_written: bool,
    ) -> None:
        """``delete_service`` must not raise its creator's ``CancelledError``.

        It checks again once the creation it waited on is over. Then it
        deletes the database file if the abandoned creation got far enough to
        write it, and raises ``FileNotFoundError`` otherwise.
        """
        gate = (
            _gate_schema_setup(monkeypatch) if database_file_written else _gate_connect(monkeypatch)
        )
        data_dir = tmp_path / "services"
        db_path = data_dir / f"{_SERVICE}.db"
        mgr = EngravaManager(data_dir=data_dir)

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        assert db_path.exists() is database_file_written
        deleter = asyncio.create_task(mgr.delete_service(_SERVICE))
        await _let_other_tasks_run()
        assert not deleter.done()

        creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creator, _BOUND_SECONDS)

        if database_file_written:
            assert await asyncio.wait_for(deleter, _BOUND_SECONDS) is None
            for suffix in (".db", ".db-wal", ".db-shm"):
                assert not db_path.with_suffix(suffix).exists()
        else:
            with pytest.raises(FileNotFoundError):
                await asyncio.wait_for(deleter, _BOUND_SECONDS)
        assert mgr._stores == {}
        assert mgr._creating == {}
        assert len(opened) == int(database_file_written)
        for conn in opened:
            assert await _worker_exited(conn)

    async def test_cancelled_close_all_still_closes_the_creation_it_waited_on(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        start_close_all: Callable[[EngravaManager], asyncio.Task[None]],
    ) -> None:
        """A cancelled ``close_all`` finishes with the in-flight creation before re-raising.

        Its cancellation does not cancel the shared creation: the creator's
        ``get_store`` returns the store it built, and ``close_all`` closes that
        store before it re-raises.
        """
        gate = _gate_schema_setup(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        closer = start_close_all(mgr)
        await _let_other_tasks_run()
        assert not closer.done()

        closer.cancel()
        await _let_other_tasks_run()
        assert not closer.done(), "close_all must keep waiting for the creation it was cancelled on"

        finished: list[str] = []
        creator.add_done_callback(lambda _: finished.append("get_store"))
        closer.add_done_callback(lambda _: finished.append("close_all"))
        gate.release.set()
        store = await asyncio.wait_for(creator, _BOUND_SECONDS)
        with pytest.raises(asyncio.CancelledError):
            await _close_all_outcome(closer)

        assert finished == ["get_store", "close_all"]
        assert mgr._stores == {}
        assert mgr._creating == {}
        assert opened == [store._db]
        assert await _worker_exited(store._db), "close_all must close the store it waited for"

    async def test_cancelled_creator_does_not_cancel_close_all_or_a_waiter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        start_close_all: Callable[[EngravaManager], asyncio.Task[None]],
    ) -> None:
        """``close_all`` treats an abandoned creation as one that produced no store.

        The waiter's retry is a new creation that starts after ``close_all``
        took its snapshot, so ``close_all`` leaves it alone, as its docstring
        says.
        """
        gate = _gate_schema_setup(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        closer = start_close_all(mgr)
        waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        await _let_other_tasks_run()
        assert not closer.done()
        assert not waiter.done()

        creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creator, _BOUND_SECONDS)

        await _close_all_outcome(closer)
        store = await asyncio.wait_for(waiter, _BOUND_SECONDS)
        assert mgr._stores == {_SERVICE: store}
        assert mgr._creating == {}
        assert len(opened) == 2
        assert store._db is opened[1]
        assert await _worker_exited(opened[0])
        await _assert_store_works(store)
        await asyncio.wait_for(mgr.close_all(), _BOUND_SECONDS)

    @pytest.mark.parametrize("cancelled_while", ["creating", "waiting-to-publish"])
    async def test_creator_cancelled_under_close_all_alone_leaves_nothing_behind(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        start_close_all: Callable[[EngravaManager], asyncio.Task[None]],
        cancelled_while: str,
    ) -> None:
        """With ``close_all`` as its only waiter, an abandoned creation leaves nothing behind."""
        gate = _gate_schema_setup(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")
        create_store_returned = _signal_create_store_return(mgr, monkeypatch)

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        closer = start_close_all(mgr)
        await _let_other_tasks_run()
        assert not closer.done()

        if cancelled_while == "creating":
            creator.cancel()
        else:
            async with mgr._lock:
                gate.release.set()
                await asyncio.wait_for(create_store_returned.wait(), _BOUND_SECONDS)
                assert not creator.done()
                creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creator, _BOUND_SECONDS)

        await _close_all_outcome(closer)
        assert mgr._stores == {}
        assert mgr._creating == {}
        assert len(opened) == 1
        assert await _worker_exited(opened[0])

    async def test_creator_cancelled_before_publishing_hands_over_and_closes_its_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[aiosqlite.Connection]
    ) -> None:
        """A creator cancelled between building its store and publishing it cleans up.

        The creator waits for the manager's lock to publish. A cancellation
        during that wait closes the store it had built and resolves the
        creation as abandoned, so the waiter starts the creation over.
        """
        gate = _gate_schema_setup(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")
        create_store_returned = _signal_create_store_return(mgr, monkeypatch)

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        await _let_other_tasks_run()
        assert not waiter.done()

        async with mgr._lock:
            gate.release.set()
            await asyncio.wait_for(create_store_returned.wait(), _BOUND_SECONDS)
            assert not creator.done()
            creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creator, _BOUND_SECONDS)

        store = await asyncio.wait_for(waiter, _BOUND_SECONDS)
        assert mgr._stores == {_SERVICE: store}
        assert mgr._creating == {}
        assert len(opened) == 2
        assert store._db is opened[1], "the waiter's store must come from its own retry"
        assert await _worker_exited(opened[0]), "the unpublished store must be closed"
        await _assert_store_works(store)
        await asyncio.wait_for(mgr.close_all(), _BOUND_SECONDS)

    @pytest.mark.parametrize("unpublished_close", ["completes", "is-cancelled-again"])
    async def test_close_all_returns_only_once_the_abandoned_store_is_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        start_close_all: Callable[[EngravaManager], asyncio.Task[None]],
        unpublished_close: str,
    ) -> None:
        """``close_all`` must not return while the store it waited for is still open.

        A creator cancelled while waiting for the lock to publish has built a
        store, and it must close that store. If it resolved the shared
        creation before that close finished, ``close_all`` would take the
        creation as having produced no store and return with the connection
        still open. This holds that close and checks that ``close_all`` is
        still waiting meanwhile. It checks the same again when the close is
        cancelled in turn: the creation must still be resolved once the close
        is over, or ``close_all`` would wait on it forever.
        """
        gate = _gate_schema_setup(monkeypatch)
        close_gate = _gate_store_close(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")
        create_store_returned = _signal_create_store_return(mgr, monkeypatch)

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        closer = start_close_all(mgr)
        await _let_other_tasks_run()
        assert not closer.done()

        async with mgr._lock:
            gate.release.set()
            await asyncio.wait_for(create_store_returned.wait(), _BOUND_SECONDS)
            assert not creator.done()
            creator.cancel()
        await asyncio.wait_for(close_gate.entered[0].wait(), _BOUND_SECONDS)
        conn = opened[0]
        assert conn._connection is not None, "the unpublished store's close is being held"

        done, _ = await asyncio.wait({closer}, timeout=0.1)
        assert not done, "close_all returned while the store it waited for was still open"

        closed_when_close_all_finished: list[bool] = []
        closer.add_done_callback(
            lambda _: closed_when_close_all_finished.append(conn._connection is None)
        )
        if unpublished_close == "completes":
            close_gate.release.set()
        else:
            creator.cancel()
        await _close_all_outcome(closer)

        assert closed_when_close_all_finished == [True]
        assert mgr._creating == {}
        assert mgr._stores == {}
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creator, _BOUND_SECONDS)
        assert len(opened) == 1
        assert await _worker_exited(conn)

    @pytest.mark.parametrize("cancelled_while", ["creating", "waiting-to-publish"])
    async def test_cancelled_creator_with_no_waiter_leaves_nothing_behind(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        cancelled_while: str,
    ) -> None:
        """A creation abandoned with nobody waiting still resolves and unregisters.

        Nobody retrieves the outcome it leaves on the shared future, so the
        creator marks it retrieved itself. This file's autouse fixture fails
        the test if asyncio reports it as never retrieved.
        """
        gate = _gate_schema_setup(monkeypatch)
        mgr = EngravaManager(data_dir=tmp_path / "services")
        create_store_returned = _signal_create_store_return(mgr, monkeypatch)

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        if cancelled_while == "creating":
            creator.cancel()
        else:
            async with mgr._lock:
                gate.release.set()
                await asyncio.wait_for(create_store_returned.wait(), _BOUND_SECONDS)
                assert not creator.done()
                creator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creator, _BOUND_SECONDS)

        assert mgr._stores == {}
        assert mgr._creating == {}
        assert len(opened) == 1
        assert await _worker_exited(opened[0])

    async def test_ordinary_creation_failure_still_reaches_every_waiter_unchanged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        start_close_all: Callable[[EngravaManager], asyncio.Task[None]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Only a cancelled creation is retried. An ordinary failure is every waiter's failure."""
        failure = RuntimeError("schema setup failed")
        gate = _Gate()

        async def _failing_ensure_schema(store: SqliteEngravaCore) -> None:
            await gate.pass_through()
            raise failure

        monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _failing_ensure_schema)
        mgr = EngravaManager(data_dir=tmp_path / "services")

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        waiter = asyncio.create_task(mgr.get_store(_SERVICE))
        closer = start_close_all(mgr)
        await _let_other_tasks_run()
        assert not waiter.done()
        assert not closer.done()

        gate.release.set()
        with pytest.raises(RuntimeError) as creator_error:
            await asyncio.wait_for(creator, _BOUND_SECONDS)
        with pytest.raises(RuntimeError) as waiter_error:
            await asyncio.wait_for(waiter, _BOUND_SECONDS)
        assert creator_error.value is failure
        assert waiter_error.value is failure
        await _close_all_outcome(closer)
        assert any(
            record.name == service_manager_module.__name__
            and record.getMessage() == f"Error awaiting in-flight creation of service {_SERVICE!r}"
            and record.exc_info is not None
            and record.exc_info[1] is failure
            for record in caplog.records
        )
        assert mgr._stores == {}
        assert mgr._creating == {}
        assert len(opened) == 1, "an ordinary failure must not be retried"
        assert await _worker_exited(opened[0])

    @pytest.mark.parametrize("creator_outcome", ["created", "cancelled"])
    async def test_creator_resolves_the_shared_future_only_if_nothing_else_has(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opened: list[aiosqlite.Connection],
        start_close_all: Callable[[EngravaManager], asyncio.Task[None]],
        creator_outcome: str,
    ) -> None:
        """An already-settled shared future breaks neither its creator nor ``close_all``.

        Nothing in the manager settles that future except its creator,
        because every waiter awaits it through ``asyncio.shield``. This test
        reaches into the manager and cancels the future directly, the only
        way to get there, to pin the guards for the day something else does.
        The creator must not raise ``InvalidStateError``. A ``close_all``
        already waiting on the creation must treat it as having produced no
        store, must not mistake the creation's cancellation for its own, and
        must still close every cached store.
        """
        mgr = EngravaManager(data_dir=tmp_path / "services")
        cached = await asyncio.wait_for(mgr.get_store("cached"), _BOUND_SECONDS)
        gate = _gate_schema_setup(monkeypatch)

        creator = asyncio.create_task(mgr.get_store(_SERVICE))
        await asyncio.wait_for(gate.entered[0].wait(), _BOUND_SECONDS)
        closer = start_close_all(mgr)
        await _let_other_tasks_run()
        assert not closer.done()

        mgr._creating[_SERVICE].cancel()
        await _close_all_outcome(closer)
        assert await _worker_exited(cached._db), "close_all must still close the cached store"

        if creator_outcome == "created":
            gate.release.set()
            store = await asyncio.wait_for(creator, _BOUND_SECONDS)
            assert mgr._stores == {_SERVICE: store}
            await asyncio.wait_for(mgr.close_all(), _BOUND_SECONDS)
        else:
            creator.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(creator, _BOUND_SECONDS)
            assert mgr._stores == {}
        assert mgr._creating == {}
        assert len(opened) == 2
        assert await _worker_exited(opened[1])

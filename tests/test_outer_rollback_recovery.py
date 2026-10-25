"""A failed outer rollback retires admission before another write can commit it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ConnectionQuarantinedError,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

pytestmark = pytest.mark.integration


def _thought(identifier: str) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=identifier,
        thought_type=ThoughtType.OBSERVATION,
        priority=Priority.P3,
        essence=identifier,
        content=f"Outer rollback recovery record {identifier}",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source="rollback-recovery-test",
    )


@pytest.fixture
async def file_store(
    tmp_path: Path,
) -> AsyncIterator[tuple[Path, aiosqlite.Connection, SqliteEngravaCore]]:
    path = tmp_path / "outer-rollback.sqlite3"
    db = await aiosqlite.connect(path)
    db.row_factory = aiosqlite.Row
    close = db.close
    core = SqliteEngravaCore(db, journal_enabled=True, close_timeout_seconds=0.02)
    try:
        await core.ensure_schema()
        await core.create_thought(_thought("sentinel"))
        yield path, db, core
    finally:
        task = core._quarantine_close_task
        if task is None:
            await close()
        else:
            # Reuse the one physical close. A failed/cancelled close is drained
            # completely before a sequential cleanup attempt on the raw handle.
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            if task.cancelled() or task.exception() is not None:
                await close()


async def _durable_ids(path: Path) -> set[str]:
    async with aiosqlite.connect(path) as reader:
        cursor = await reader.execute("SELECT thought_id FROM thought")
        return {row[0] for row in await cursor.fetchall()}


async def _failure_from_window(
    core: SqliteEngravaCore, primary: BaseException
) -> BaseException | None:
    try:
        async with core.suspend_auto_commit():
            await core.create_thought(_thought("failed"))
            raise primary  # noqa: TRY301 -- capture the real window unwind for precedence
    except BaseException as exc:  # noqa: BLE001 -- retain exact error/cancellation objects
        return exc
    return None


async def _later_write(core: SqliteEngravaCore) -> BaseException | None:
    try:
        await core.create_thought(_thought("later"))
    except BaseException as exc:  # noqa: BLE001 -- inspect real write outcome before durable state
        return exc
    return None


def _assert_window_closed(core: SqliteEngravaCore) -> None:
    assert core._skip_auto_commit_depth == 0, "window depth leaked"
    assert core._current_auto_commit_window.get() is None, "window task marker leaked"
    assert core._open_auto_commit_windows == set(), "window identities leaked"
    assert core._write_lock._owner is None, "window write lock remained owned"


@pytest.mark.parametrize("body_cancelled", [False, True], ids=["body-error", "body-cancellation"])
@pytest.mark.parametrize(
    "rollback_cancelled", [False, True], ids=["rollback-error", "rollback-cancellation"]
)
async def test_failed_outer_rollback_cannot_publish_failed_work(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
    body_cancelled: bool,
    rollback_cancelled: bool,
) -> None:
    path, db, core = file_store
    primary = (
        asyncio.CancelledError("body cancelled") if body_cancelled else ValueError("body failed")
    )
    cleanup = (
        asyncio.CancelledError("rollback cancelled")
        if rollback_cancelled
        else RuntimeError("rollback failed")
    )
    original = db.rollback

    async def failed_rollback() -> None:
        assert db.in_transaction
        raise cleanup

    monkeypatch.setattr(db, "rollback", failed_rollback)
    observed = await _failure_from_window(core, primary)
    monkeypatch.setattr(db, "rollback", original)
    later = await _later_write(core)
    # State first: the broken implementation actually publishes both failed
    # and unrelated rows; an exception-only assertion would hide that damage.
    assert await _durable_ids(path) == {"sentinel"}, "failed window was published by a later write"
    assert isinstance(later, ConnectionQuarantinedError)
    assert core._connection_quarantined
    _assert_window_closed(core)
    assert observed is (cleanup if rollback_cancelled else primary), "rollback precedence changed"
    if not rollback_cancelled:
        assert observed.__cause__ is cleanup, "body failure lost its rollback cause"
    with pytest.raises(ConnectionQuarantinedError):
        await core.get_thought("sentinel")
    with pytest.raises(ConnectionQuarantinedError):
        await core.verify_journal()


@pytest.mark.parametrize("rollback_fails", [False, True], ids=["recovered", "quarantined"])
async def test_real_task_cancel_during_body_preserves_cancellation(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
    rollback_fails: bool,
) -> None:
    path, db, core = file_store
    entered = asyncio.Event()
    release = asyncio.Event()
    source: list[asyncio.CancelledError] = []
    cleanup = RuntimeError("rollback failed")
    original = db.rollback

    async def rollback() -> None:
        if rollback_fails:
            raise cleanup
        await original()

    async def worker() -> None:
        try:
            async with core.suspend_auto_commit():
                await core.create_thought(_thought("failed"))
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError as exc:
                    source.append(exc)
                    raise
        finally:
            _assert_window_closed(core)

    monkeypatch.setattr(db, "rollback", rollback)
    task = asyncio.create_task(worker())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            task.cancel("body task cancelled")
            observed: asyncio.CancelledError | None = None
            try:
                await task
            except asyncio.CancelledError as exc:
                observed = exc
        monkeypatch.setattr(db, "rollback", original)
        later = await _later_write(core)
        expected = {"sentinel"} if rollback_fails else {"sentinel", "later"}
        assert await _durable_ids(path) == expected
        assert task.cancelled()
        assert task.cancelling() == 1
        assert len(source) == 1
        assert observed is source[0]
        assert observed.__cause__ is (cleanup if rollback_fails else None)
        assert isinstance(later, ConnectionQuarantinedError) if rollback_fails else later is None
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("body_cancelled", [False, True], ids=["body-error", "body-cancellation"])
async def test_real_task_cancel_during_rollback_outranks_body(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
    body_cancelled: bool,
) -> None:
    path, db, core = file_store
    entered = asyncio.Event()
    release = asyncio.Event()
    cancellations: list[asyncio.CancelledError] = []
    original = db.rollback
    primary = (
        asyncio.CancelledError("body cancellation") if body_cancelled else ValueError("body error")
    )

    async def rollback() -> None:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError as exc:
            cancellations.append(exc)
            raise

    async def worker() -> None:
        observed = await _failure_from_window(core, primary)
        _assert_window_closed(core)
        assert observed is not None
        raise observed

    monkeypatch.setattr(db, "rollback", rollback)
    task = asyncio.create_task(worker())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            task.cancel("rollback task cancelled")
            observed: asyncio.CancelledError | None = None
            try:
                await task
            except asyncio.CancelledError as exc:
                observed = exc
        monkeypatch.setattr(db, "rollback", original)
        later = await _later_write(core)
        assert await _durable_ids(path) == {"sentinel"}
        assert isinstance(later, ConnectionQuarantinedError)
        assert task.cancelled()
        assert task.cancelling() == 1
        assert len(cancellations) == 1
        assert observed is cancellations[0]
        assert observed is not primary
        assert observed.args == ("rollback task cancelled",)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_writer_already_queued_on_failed_window_is_refused(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, db, core = file_store
    in_rollback = asyncio.Event()
    peer_attempted = asyncio.Event()
    release = asyncio.Event()
    primary = ValueError("outer failure")
    cleanup = RuntimeError("rollback failure")
    original_rollback = db.rollback
    acquire = core._write_lock.acquire

    async def rollback() -> None:
        in_rollback.set()
        await release.wait()
        raise cleanup

    async def watch_acquire() -> None:
        task = asyncio.current_task()
        if task is not None and task.get_name() == "queued-writer":
            peer_attempted.set()
        await acquire()

    async def owner() -> BaseException | None:
        result = await _failure_from_window(core, primary)
        _assert_window_closed(core)
        return result

    monkeypatch.setattr(db, "rollback", rollback)
    # Observe the existing acquisition boundary; supply neither a lock nor a
    # scheduling delay. The real core must keep the second writer excluded.
    monkeypatch.setattr(core._write_lock, "acquire", watch_acquire)
    tasks: list[asyncio.Task[BaseException | None]] = []
    try:
        async with asyncio.timeout(5):
            first = asyncio.create_task(owner())
            tasks.append(first)
            await in_rollback.wait()
            peer = asyncio.create_task(_later_write(core), name="queued-writer")
            tasks.append(peer)
            await peer_attempted.wait()
            release.set()
            source, later = await asyncio.gather(first, peer)
        monkeypatch.setattr(db, "rollback", original_rollback)
        assert await _durable_ids(path) == {"sentinel"}, "queued writer published the failed window"
        assert isinstance(later, ConnectionQuarantinedError)
        assert source is primary
        assert source.__cause__ is cleanup
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("rollback_fails", [False, True], ids=["recovered", "quarantined"])
async def test_outer_window_retains_ownership_of_callers_existing_transaction(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
    rollback_fails: bool,
) -> None:
    path, db, core = file_store
    await db.execute(
        "UPDATE thought SET essence = 'pending caller edit' WHERE thought_id = 'sentinel'"
    )
    assert db.in_transaction
    primary = ValueError("outer body failed")
    cleanup = RuntimeError("rollback failed")
    original = db.rollback

    async def rollback() -> None:
        raise cleanup

    if rollback_fails:
        monkeypatch.setattr(db, "rollback", rollback)
    observed = await _failure_from_window(core, primary)
    monkeypatch.setattr(db, "rollback", original)
    later = await _later_write(core)
    expected = {"sentinel"} if rollback_fails else {"sentinel", "later"}
    assert await _durable_ids(path) == expected
    async with aiosqlite.connect(path) as reader:
        cursor = await reader.execute("SELECT essence FROM thought WHERE thought_id = 'sentinel'")
        assert (await cursor.fetchone())[0] == "sentinel", (
            "outer rollback left caller edits pending"
        )
    assert observed is primary
    assert isinstance(later, ConnectionQuarantinedError) if rollback_fails else later is None
    _assert_window_closed(core)


@pytest.mark.parametrize("body_fails", [False, True], ids=["clean", "body-error"])
async def test_empty_windows_do_not_issue_commit_or_rollback(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
    body_fails: bool,
) -> None:
    path, db, core = file_store
    calls: list[str] = []
    commit, rollback = db.commit, db.rollback
    primary = ValueError("empty body failed")

    async def watch_commit() -> None:
        calls.append("commit")
        await commit()

    async def watch_rollback() -> None:
        calls.append("rollback")
        await rollback()

    monkeypatch.setattr(db, "commit", watch_commit)
    monkeypatch.setattr(db, "rollback", watch_rollback)
    observed = None
    try:
        async with core.suspend_auto_commit():
            if body_fails:
                raise primary
    except ValueError as exc:
        observed = exc
    assert await _durable_ids(path) == {"sentinel"}
    assert calls == [], "empty window issued transaction cleanup"
    assert observed is (primary if body_fails else None)
    assert not db.in_transaction
    assert not core._connection_quarantined
    _assert_window_closed(core)


async def test_caught_inner_failure_commits_prefix_without_an_inner_rollback(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, db, core = file_store
    rollbacks: list[bool] = []
    rollback = db.rollback

    async def watch_rollback() -> None:
        rollbacks.append(True)
        await rollback()

    monkeypatch.setattr(db, "rollback", watch_rollback)
    async with core.suspend_auto_commit():
        outer_marker = core._current_auto_commit_window.get()
        await core.create_thought(_thought("before"))

        async def inner_failure() -> None:
            async with core.suspend_auto_commit():
                await core.create_thought(_thought("inner-prefix"))
                message = "caught inner error"
                raise ValueError(message)

        with pytest.raises(ValueError, match="caught inner error"):
            await inner_failure()
        assert core._current_auto_commit_window.get() is outer_marker
        assert core._skip_auto_commit_depth == 1
        assert len(core._open_auto_commit_windows) == 1
        assert db.in_transaction
        await core.create_thought(_thought("after"))
    assert await _durable_ids(path) == {"sentinel", "before", "inner-prefix", "after"}
    assert rollbacks == []
    assert (await core.verify_journal()).valid
    _assert_window_closed(core)


async def test_escaping_inner_failure_rolls_back_entire_outer_window(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
) -> None:
    path, db, core = file_store
    primary = ValueError("escaping inner failure")
    observed = None
    try:
        async with core.suspend_auto_commit():
            await core.create_thought(_thought("before"))
            async with core.suspend_auto_commit():
                await core.create_thought(_thought("inner-prefix"))
                raise primary
    except ValueError as exc:
        observed = exc
    assert await _durable_ids(path) == {"sentinel"}
    assert observed is primary
    assert not core._connection_quarantined
    assert not db.in_transaction
    _assert_window_closed(core)
    await core.create_thought(_thought("later"))
    assert await _durable_ids(path) == {"sentinel", "later"}


@pytest.fixture
async def owned_file_store(
    tmp_path: Path,
) -> AsyncIterator[tuple[Path, aiosqlite.Connection, SqliteEngravaCore]]:
    path = tmp_path / "owned.sqlite3"
    config = tmp_path / "engrava.yaml"
    config.write_text(f"database:\n  path: {path}\n  wal_mode: false\n", encoding="utf-8")
    core = await SqliteEngravaCore.from_config(config, close_timeout_seconds=0.02)
    db = core._db
    close = db.close
    try:
        await core.create_thought(_thought("sentinel"))
        yield path, db, core
    finally:
        task = core._quarantine_close_task
        if task is None:
            await close()
        else:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            if task.cancelled() or task.exception() is not None:
                await close()


async def test_hung_quarantine_close_revokes_admission_and_shares_owned_close(
    owned_file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, db, core = owned_file_store
    entered, release = asyncio.Event(), asyncio.Event()
    original_close = db.close
    calls: list[bool] = []

    async def delayed_close() -> None:
        calls.append(True)
        entered.set()
        await release.wait()
        await original_close()

    async def failed_rollback() -> None:
        message = "rollback failed"
        raise RuntimeError(message)

    monkeypatch.setattr(db, "close", delayed_close)
    monkeypatch.setattr(db, "rollback", failed_rollback)
    physical = None
    try:
        async with asyncio.timeout(5):
            observed = await _failure_from_window(core, ValueError("body failed"))
            await entered.wait()
        physical = core._quarantine_close_task
        later = await _later_write(core)
        assert await _durable_ids(path) == {"sentinel"}
        assert isinstance(later, ConnectionQuarantinedError)
        assert isinstance(observed, ValueError)
        assert physical is not None
        assert not physical.done(), "quarantine waited for the hung physical close"
        async with asyncio.timeout(5):
            outcomes = await asyncio.gather(core.close(), core.close(), return_exceptions=True)
        assert core._quarantine_close_task is physical, "shared physical close was replaced"
        assert all(isinstance(outcome, ConnectionQuarantinedError) for outcome in outcomes)
        assert not physical.cancelled()
        assert calls == [True]
        release.set()
        await asyncio.wait_for(asyncio.shield(physical), 5)
        await core.close()
        assert calls == [True]
    finally:
        release.set()
        if physical is not None:
            await asyncio.wait_for(asyncio.gather(physical, return_exceptions=True), 5)


async def test_already_quarantined_outer_unwind_never_reads_terminal_proxy(
    owned_file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, db, core = owned_file_store
    release = asyncio.Event()
    original_close = db.close
    primary = ValueError("body failed after close timed out")
    observed = None

    async def delayed_close() -> None:
        await release.wait()
        await original_close()

    monkeypatch.setattr(db, "close", delayed_close)
    try:
        async with asyncio.timeout(5):
            try:
                async with core.suspend_auto_commit():
                    await core.create_thought(_thought("failed"))
                    with pytest.raises(ConnectionQuarantinedError):
                        await core.close()
                    raise primary  # noqa: TRY301 -- exercise an already-quarantined real window
            except BaseException as exc:  # noqa: BLE001 -- distinguish original error from proxy failure
                observed = exc
        later = await _later_write(core)
        assert await _durable_ids(path) == {"sentinel"}
        assert isinstance(later, ConnectionQuarantinedError)
        assert observed is primary, "quarantined unwind replaced the original failure"
        _assert_window_closed(core)
    finally:
        release.set()


async def test_detached_close_failure_is_consumed_without_reopening_admission(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    path, db, core = file_store
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    reports: list[dict[str, object]] = []
    finished = asyncio.Event()
    close_error = RuntimeError("physical close failed")

    def capture(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        reports.append(context)

    async def failed_close() -> None:
        raise close_error

    async def failed_rollback() -> None:
        message = "rollback failed"
        raise RuntimeError(message)

    monkeypatch.setattr(db, "close", failed_close)
    monkeypatch.setattr(db, "rollback", failed_rollback)
    loop.set_exception_handler(capture)
    try:
        await _failure_from_window(core, ValueError("body failed"))
        task = core._quarantine_close_task
        assert task is not None
        task.add_done_callback(lambda _done: finished.set())
        async with asyncio.timeout(5):
            await finished.wait()
        later = await _later_write(core)
        assert await _durable_ids(path) == {"sentinel"}
        assert isinstance(later, ConnectionQuarantinedError)
        # Neither await the task nor read its result here: that would consume
        # the failure ourselves and let a missing production consumer pass.
        reference = weakref.ref(task)
        core._quarantine_close_task = None
        del task
        gc.collect()
        checkpoint = asyncio.Event()
        loop.call_soon(checkpoint.set)
        await checkpoint.wait()
        assert reports == [], "detached physical-close failure was not consumed"
        assert reference() is None
    finally:
        loop.set_exception_handler(original_handler)
        # The injected close task is finished before this fixture releases the
        # real handle; a second concurrent aiosqlite close is never started.


async def test_manual_close_keeps_the_callers_healthy_connection_usable(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
) -> None:
    path, db, core = file_store
    await core.close()
    await core.create_thought(_thought("later"))
    assert await _durable_ids(path) == {"sentinel", "later"}
    assert not db.in_transaction


class _Unrenderable:
    def __repr__(self) -> str:
        message = "exception representation failed"
        raise RuntimeError(message)


class _UnrenderableBodyError(_Unrenderable, ValueError):
    pass


class _UnrenderableRollbackError(_Unrenderable, RuntimeError):
    pass


class _UnrenderableCancellation(_Unrenderable, asyncio.CancelledError):
    pass


@pytest.mark.parametrize("opaque", ["body", "rollback"])
@pytest.mark.parametrize("body_cancelled", [False, True], ids=["body-error", "body-cancellation"])
@pytest.mark.parametrize(
    "rollback_cancelled", [False, True], ids=["rollback-error", "rollback-cancellation"]
)
async def test_exception_representation_cannot_prevent_retirement(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
    opaque: str,
    body_cancelled: bool,
    rollback_cancelled: bool,
) -> None:
    path, db, core = file_store
    body_type = _UnrenderableBodyError if opaque == "body" else ValueError
    rollback_type = _UnrenderableRollbackError if opaque == "rollback" else RuntimeError
    body_cancel_type = _UnrenderableCancellation if opaque == "body" else asyncio.CancelledError
    rollback_cancel_type = (
        _UnrenderableCancellation if opaque == "rollback" else asyncio.CancelledError
    )
    primary = body_cancel_type("body cancelled") if body_cancelled else body_type("body failed")
    cleanup = (
        rollback_cancel_type("rollback cancelled")
        if rollback_cancelled
        else rollback_type("rollback failed")
    )
    original = db.rollback

    async def failed_rollback() -> None:
        assert db.in_transaction
        raise cleanup

    monkeypatch.setattr(db, "rollback", failed_rollback)
    observed = await _failure_from_window(core, primary)
    monkeypatch.setattr(db, "rollback", original)
    later = await _later_write(core)
    assert await _durable_ids(path) == {"sentinel"}, (
        "exception formatting left failed writes admissible"
    )
    assert isinstance(later, ConnectionQuarantinedError)
    assert core._connection_quarantined
    _assert_window_closed(core)
    assert observed is (cleanup if rollback_cancelled else primary), (
        "opaque exception precedence changed"
    )
    if not rollback_cancelled:
        assert observed.__cause__ is cleanup, "opaque exception cause was lost"
    with pytest.raises(ConnectionQuarantinedError):
        await core.get_thought("sentinel")
    with pytest.raises(ConnectionQuarantinedError):
        await core.verify_journal()


async def _cancel_when_entered(
    task: asyncio.Task[None], entered: asyncio.Event, message: object
) -> BaseException | None:
    async with asyncio.timeout(5):
        await entered.wait()
        task.cancel(message)
        try:
            await task
        except BaseException as exc:  # noqa: BLE001 -- inspect state before opaque cancellation identity
            return exc
    return None


@pytest.mark.parametrize("phase", ["body", "rollback"])
async def test_real_task_cancel_with_opaque_message_still_retires_connection(
    file_store: tuple[Path, aiosqlite.Connection, SqliteEngravaCore],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    path, db, core = file_store
    entered, release = asyncio.Event(), asyncio.Event()
    message = _Unrenderable()
    cancellations: list[asyncio.CancelledError] = []
    primary = ValueError("body failed")
    cleanup = RuntimeError("rollback failed")
    original = db.rollback

    async def await_cancellation() -> None:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError as exc:
            cancellations.append(exc)
            raise

    async def failed_rollback() -> None:
        if phase == "rollback":
            await await_cancellation()
        raise cleanup

    async def worker() -> None:
        try:
            async with core.suspend_auto_commit():
                await core.create_thought(_thought("failed"))
                if phase == "body":
                    await await_cancellation()
                raise primary
        finally:
            _assert_window_closed(core)

    monkeypatch.setattr(db, "rollback", failed_rollback)
    task = asyncio.create_task(worker())
    try:
        observed = await _cancel_when_entered(task, entered, message)
        monkeypatch.setattr(db, "rollback", original)
        later = await _later_write(core)
        assert await _durable_ids(path) == {"sentinel"}, (
            "opaque task cancellation left failed writes admissible"
        )
        assert isinstance(later, ConnectionQuarantinedError)
        assert task.cancelled()
        assert task.cancelling() == 1
        assert len(cancellations) == 1
        assert observed is cancellations[0]
        assert observed.args[0] is message
        assert observed.__cause__ is (cleanup if phase == "body" else None)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

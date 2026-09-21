"""In-process regression tests for the examples' cleanup-close ordering.

Each of ``agent_loop.py``, ``notes_memory.py`` and ``quickstart.py`` used a
bare ``async with aiosqlite.connect(":memory:") as conn:`` --
``aiosqlite.Connection.__aexit__`` is an unconditional ``await close()`` and
cannot distinguish a cleanup close (something in the body already raised)
from a success-path one, so a failure in that close would replace whatever
the body actually raised. These tests import each example as a module (the
subprocess-driven tests in ``test_quickstart_runs.py`` cannot observe this --
a monkeypatched ``aiosqlite.connect`` only reaches code running in the same
process) and force both a body failure and a close failure to prove the
body's own exception is what actually propagates.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import aiosqlite

requires_local_embeddings = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (engrava[embeddings-local] extra)",
)


def _install_failing_close_spy(module: object, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch *module*'s ``aiosqlite.connect`` so every returned close fails.

    Returns a mutable counter dict so the caller can assert the close was
    actually attempted (a regression that drops the cleanup close entirely
    would also let the original exception escape untouched, so that alone
    would not be enough to trust). Goes through ``monkeypatch`` so the
    patch cannot leak into another test via the shared ``aiosqlite``
    module.
    """
    close_calls = {"n": 0}
    real_connect = module.aiosqlite.connect  # type: ignore[attr-defined]

    def _spy_connect(*args: object, **kwargs: object) -> object:
        # ``aiosqlite.connect()`` itself is synchronous -- it returns a
        # ``Connection`` proxy immediately without opening anything, so this
        # spy must be a plain sync function too: an ``async def`` spy would
        # support only one of ``await``/``async with`` and fail for the
        # wrong reason.
        conn = real_connect(*args, **kwargs)
        real_close = conn.close

        async def _close_blows_up() -> None:
            close_calls["n"] += 1
            await real_close()
            msg = "close blew up during cleanup"
            raise RuntimeError(msg)

        conn.close = _close_blows_up
        return conn

    monkeypatch.setattr(module.aiosqlite, "connect", _spy_connect)  # type: ignore[attr-defined]
    return close_calls


def _install_close_that_never_releases_spy(
    module: object, monkeypatch: pytest.MonkeyPatch
) -> list[aiosqlite.Connection]:
    """Patch *module*'s ``aiosqlite.connect`` so the returned close never runs the real one.

    ``_install_failing_close_spy`` above always calls the real ``close()``
    first and only raises on top of it, so the connection is already
    released before the simulated failure -- that spy cannot tell a
    helper that reports-and-swallows an ordinary close failure apart from
    one that would also paper over a close that raises *before* doing any
    of its own cleanup, which is the failure mode that actually leaks: the
    connection and its non-daemon worker thread stay alive. This spy never
    touches the real close at all. Returns the opened connections so the
    caller can inspect that leaked state directly.
    """
    opened: list[aiosqlite.Connection] = []
    real_connect = module.aiosqlite.connect  # type: ignore[attr-defined]

    def _spy_connect(*args: object, **kwargs: object) -> object:
        conn = real_connect(*args, **kwargs)
        opened.append(conn)

        async def _close_raises_before_releasing_anything() -> None:
            msg = "close blew up before releasing anything"
            raise RuntimeError(msg)

        conn.close = _close_raises_before_releasing_anything
        return conn

    monkeypatch.setattr(module.aiosqlite, "connect", _spy_connect)  # type: ignore[attr-defined]
    return opened


def _assert_leaked(opened: list[aiosqlite.Connection]) -> None:
    """Assert the observable leak a close that never releases anything causes.

    ``_close_quietly`` can only report a close failure and let the body's
    error propagate -- it cannot force a release that a broken ``close()``
    never performed. So the honest assertion here is that the connection
    and its worker thread are *still alive*, not that they were somehow
    cleaned up anyway.
    """
    assert opened, "the connect() spy never observed a connection being opened"
    conn = opened[0]
    assert conn._connection is not None, (
        "the connection was released even though the close spy never "
        "touched the real close -- the thread-alive assertion below "
        "would then be trivially true, so this pins the precondition "
        "instead"
    )
    assert conn._thread.is_alive(), (
        "the worker thread has already stopped even though the close spy "
        "raised before doing any cleanup -- the point of this test is "
        "that a close failing this way is *not* silently turned into a "
        "released connection; if this ever goes false, something started "
        "forcing a release on a close that raised before performing one, "
        "and this test's assumptions need re-checking"
    )


async def _finalize_leaked(opened: list[aiosqlite.Connection]) -> None:
    """Stop the worker thread(s) the spy above left running, and check it worked.

    Bypasses the deliberately broken patched ``close`` and stops each
    thread directly. Neither ``wait_for(..., timeout=5)`` nor
    ``join(timeout=5)`` on their own prove the thread actually stopped --
    both simply return once the timeout elapses either way -- so this
    asserts it did rather than silently leaving a wedged thread to
    outlive the test.
    """
    for leaked in opened:
        if leaked._thread.is_alive():
            stopped = leaked.stop()
            if stopped is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stopped, timeout=5)
            leaked._thread.join(timeout=5)
        assert not leaked._thread.is_alive(), (
            "the worker thread did not stop within the 5s join timeout -- "
            "it would otherwise outlive this test"
        )


async def test_agent_loop_body_failure_survives_a_failing_cleanup_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``agent_loop.main()`` must not let a close failure replace the body's."""
    import examples.agent_loop as agent_loop_module
    from engrava import SqliteEngravaCore

    close_calls = _install_failing_close_spy(agent_loop_module, monkeypatch)

    async def _ensure_schema_blows_up(self: SqliteEngravaCore) -> None:
        msg = "original body failure"
        raise ValueError(msg)

    monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _ensure_schema_blows_up)

    with pytest.raises(ValueError, match="original body failure"):
        await agent_loop_module.main()

    assert close_calls["n"] == 1


async def test_agent_loop_body_failure_survives_a_close_that_never_releases_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``agent_loop.main()`` must not let a close that never releases anything win either."""
    import examples.agent_loop as agent_loop_module
    from engrava import SqliteEngravaCore

    opened = _install_close_that_never_releases_spy(agent_loop_module, monkeypatch)

    async def _ensure_schema_blows_up(self: SqliteEngravaCore) -> None:
        msg = "original body failure"
        raise ValueError(msg)

    monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _ensure_schema_blows_up)

    try:
        with pytest.raises(ValueError, match="original body failure"):
            await agent_loop_module.main()
        _assert_leaked(opened)
    finally:
        await _finalize_leaked(opened)


async def test_notes_memory_body_failure_survives_a_failing_cleanup_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``notes_memory.main()`` must not let a close failure replace the body's."""
    import examples.notes_memory as notes_memory_module
    from engrava import SqliteEngravaCore

    close_calls = _install_failing_close_spy(notes_memory_module, monkeypatch)

    async def _ensure_schema_blows_up(self: SqliteEngravaCore) -> None:
        msg = "original body failure"
        raise ValueError(msg)

    monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _ensure_schema_blows_up)

    with pytest.raises(ValueError, match="original body failure"):
        await notes_memory_module.main()

    assert close_calls["n"] == 1


async def test_notes_memory_body_failure_survives_a_close_that_never_releases_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``notes_memory.main()`` must not let a close that never releases anything win either."""
    import examples.notes_memory as notes_memory_module
    from engrava import SqliteEngravaCore

    opened = _install_close_that_never_releases_spy(notes_memory_module, monkeypatch)

    async def _ensure_schema_blows_up(self: SqliteEngravaCore) -> None:
        msg = "original body failure"
        raise ValueError(msg)

    monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _ensure_schema_blows_up)

    try:
        with pytest.raises(ValueError, match="original body failure"):
            await notes_memory_module.main()
        _assert_leaked(opened)
    finally:
        await _finalize_leaked(opened)


@requires_local_embeddings
async def test_quickstart_body_failure_survives_a_failing_cleanup_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``quickstart.main()`` must not let a close failure replace the body's.

    The real ``SentenceTransformerProvider`` loads a local model on
    construction, which this test has no need to pay for -- the failure
    is forced before any embedding ever happens -- so the deferred import
    inside ``main()`` is patched to a lightweight stand-in first.
    """
    import engrava.embeddings.sentence_transformer as st_module
    import examples.quickstart as quickstart_module
    from engrava import SqliteEngravaCore

    class _FakeProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(st_module, "SentenceTransformerProvider", _FakeProvider)

    close_calls = _install_failing_close_spy(quickstart_module, monkeypatch)

    async def _ensure_schema_blows_up(self: SqliteEngravaCore) -> None:
        msg = "original body failure"
        raise ValueError(msg)

    monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _ensure_schema_blows_up)

    with pytest.raises(ValueError, match="original body failure"):
        await quickstart_module.main()

    assert close_calls["n"] == 1


@requires_local_embeddings
async def test_quickstart_body_failure_survives_a_close_that_never_releases_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``quickstart.main()`` must not let a close that never releases anything win either.

    See ``test_quickstart_body_failure_survives_a_failing_cleanup_close``
    above for why the deferred ``SentenceTransformerProvider`` import is
    patched to a lightweight stand-in first.
    """
    import engrava.embeddings.sentence_transformer as st_module
    import examples.quickstart as quickstart_module
    from engrava import SqliteEngravaCore

    class _FakeProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(st_module, "SentenceTransformerProvider", _FakeProvider)

    opened = _install_close_that_never_releases_spy(quickstart_module, monkeypatch)

    async def _ensure_schema_blows_up(self: SqliteEngravaCore) -> None:
        msg = "original body failure"
        raise ValueError(msg)

    monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _ensure_schema_blows_up)

    try:
        with pytest.raises(ValueError, match="original body failure"):
            await quickstart_module.main()
        _assert_leaked(opened)
    finally:
        await _finalize_leaked(opened)

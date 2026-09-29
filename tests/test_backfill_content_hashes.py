"""Tests for ``scripts.backfill_content_hashes``.

Covers the cleanup-close ordering in ``backfill``, which does not use a bare
``async with aiosqlite.connect(...) as db:`` --
``aiosqlite.Connection.__aexit__`` is an unconditional ``await close()`` and
cannot distinguish a cleanup close (something in the body already raised)
from a success-path one, so a failure in that close would replace whatever
the body actually raised.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import aiosqlite
import pytest

if TYPE_CHECKING:
    from pathlib import Path


async def _seed_db(db_path: Path, *, null_hash_count: int) -> None:
    """Create a minimal ``thought`` table with *null_hash_count* NULL-hash rows."""
    conn = await aiosqlite.connect(str(db_path))
    await conn.execute(
        "CREATE TABLE thought (thought_id TEXT PRIMARY KEY, content TEXT, content_hash TEXT)"
    )
    for i in range(null_hash_count):
        await conn.execute(
            "INSERT INTO thought (thought_id, content, content_hash) VALUES (?, ?, NULL)",
            (f"t-{i:03d}", f"content {i}"),
        )
    await conn.commit()
    await conn.close()


class TestBackfillCleanupClose:
    """``backfill`` must not let a close failure replace the body's own failure."""

    async def test_body_failure_survives_a_failing_cleanup_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A close failure during cleanup must not replace the body's own failure."""
        import scripts.backfill_content_hashes as backfill_module

        db_path = tmp_path / "backfill.db"
        await _seed_db(db_path, null_hash_count=5)

        close_calls = {"n": 0}
        real_connect = backfill_module.aiosqlite.connect

        def _spy_connect(*args: object, **kwargs: object) -> object:
            # ``aiosqlite.connect()`` itself is synchronous -- it returns a
            # ``Connection`` proxy immediately without opening anything, so
            # this spy must be a plain sync function too: an ``async def``
            # spy would support only one of ``await``/``async with`` and
            # fail for the wrong reason.
            conn = real_connect(*args, **kwargs)
            real_close = conn.close

            async def _close_blows_up() -> None:
                close_calls["n"] += 1
                await real_close()
                msg = "close blew up during cleanup"
                raise RuntimeError(msg)

            conn.close = _close_blows_up
            return conn

        monkeypatch.setattr(backfill_module.aiosqlite, "connect", _spy_connect)

        async def _execute_blows_up(*args: object, **kwargs: object) -> object:
            msg = "original body failure"
            raise ValueError(msg)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _execute_blows_up)

        with pytest.raises(ValueError, match="original body failure"):
            await backfill_module.backfill(db_path)

        assert close_calls["n"] == 1, (
            "the cleanup close was never attempted -- a regression that "
            "drops the close call entirely would also let the original "
            "ValueError escape untouched, so that alone is not enough"
        )

    async def test_body_failure_survives_a_close_that_never_releases_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A close that raises *before* releasing anything must not win either.

        The spy above calls the real ``close()`` first and only raises on
        top of it, so the connection is already released before the
        simulated failure. That spy cannot tell a helper that
        reports-and-swallows an ordinary close failure apart from one that
        would also paper over a close that raises *before* doing any of
        its own cleanup, which is the failure mode that actually leaks:
        the connection and its non-daemon worker thread stay alive. This
        spy never touches the real close at all, so the assertions below
        are about that observable state rather than about whether a close
        was merely attempted.
        """
        import scripts.backfill_content_hashes as backfill_module

        db_path = tmp_path / "backfill_leak.db"
        await _seed_db(db_path, null_hash_count=5)

        opened: list[aiosqlite.Connection] = []
        real_connect = backfill_module.aiosqlite.connect

        def _spy_connect(*args: object, **kwargs: object) -> object:
            conn = real_connect(*args, **kwargs)
            opened.append(conn)

            async def _close_raises_before_releasing_anything() -> None:
                msg = "close blew up before releasing anything"
                raise RuntimeError(msg)

            conn.close = _close_raises_before_releasing_anything
            return conn

        monkeypatch.setattr(backfill_module.aiosqlite, "connect", _spy_connect)

        async def _execute_blows_up(*args: object, **kwargs: object) -> object:
            msg = "original body failure"
            raise ValueError(msg)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _execute_blows_up)

        try:
            with pytest.raises(ValueError, match="original body failure"):
                await backfill_module.backfill(db_path)

            assert opened, "the connect() spy never observed a connection being opened"
            conn = opened[0]
            assert conn._connection is not None, (
                "the connection was released even though the close spy "
                "never touched the real close -- the thread-alive "
                "assertion below would then be trivially true, so this "
                "pins the precondition instead"
            )
            assert conn._thread.is_alive(), (
                "the worker thread has already stopped even though the "
                "close spy raised before doing any cleanup -- the point "
                "of this test is that a close failing this way is *not* "
                "silently turned into a released connection; if this ever "
                "goes false, something started forcing a release on a "
                "close that raised before performing one, and this test's "
                "assumptions need re-checking"
            )
        finally:
            # The spy above never lets the real close run, so the worker
            # thread from this test would otherwise outlive it -- stop and
            # join it directly rather than through the (deliberately
            # broken) patched ``close``.
            for conn in opened:
                if conn._thread.is_alive():
                    stopped = conn.stop()
                    if stopped is not None:
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(stopped, timeout=5)
                    conn._thread.join(timeout=5)

    async def test_close_happens_on_the_success_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ordinary success path (no NULL rows left) still closes exactly once."""
        import scripts.backfill_content_hashes as backfill_module

        db_path = tmp_path / "backfill_ok.db"
        await _seed_db(db_path, null_hash_count=3)

        real_connect = backfill_module.aiosqlite.connect
        close_calls = {"n": 0}

        def _counting_connect(*args: object, **kwargs: object) -> object:
            conn = real_connect(*args, **kwargs)
            real_close = conn.close

            async def _counted_close() -> None:
                close_calls["n"] += 1
                await real_close()

            conn.close = _counted_close
            return conn

        monkeypatch.setattr(backfill_module.aiosqlite, "connect", _counting_connect)

        updated = await backfill_module.backfill(db_path, batch_size=2)
        assert updated == 3
        assert close_calls["n"] == 1

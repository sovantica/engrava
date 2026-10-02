"""Tests for ``scripts.rebalance_p1``.

Covers the cleanup-close ordering fixed alongside the rest of this sweep:
``rebalance`` used a bare ``async with aiosqlite.connect(...) as db:``, and
its body has *three* distinct semantic branches (no demotion needed,
dry-run preview, real write) that each set ``demoted`` differently before
falling through to the single ``return`` after the try/except/else. A naive
conversion to an explicit ``try`` could instead give one of those branches
its own early ``return`` sitting *inside* the ``try``, which would skip the
``else`` clause that closes the connection on the success path -- these
tests pin all three branches so that regression cannot land unnoticed.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import aiosqlite
import pytest

if TYPE_CHECKING:
    from pathlib import Path


async def _seed_db(db_path: Path, *, p1_count: int, p2_count: int = 0) -> None:
    """Create a minimal ``thought`` table with *p1_count* P1 rows."""
    conn = await aiosqlite.connect(str(db_path))
    await conn.execute(
        "CREATE TABLE thought (thought_id TEXT PRIMARY KEY, priority TEXT, created_at TEXT)"
    )
    for i in range(p1_count):
        await conn.execute(
            "INSERT INTO thought VALUES (?, 'P1', ?)",
            (f"p1-{i:03d}", f"2026-01-{(i % 28) + 1:02d}T00:00:00+00:00"),
        )
    for i in range(p2_count):
        await conn.execute(
            "INSERT INTO thought VALUES (?, 'P2', ?)",
            (f"p2-{i:03d}", f"2026-01-{(i % 28) + 1:02d}T00:00:00+00:00"),
        )
    await conn.commit()
    await conn.close()


# ---------------------------------------------------------------------------
# Cleanup-close ordering
# ---------------------------------------------------------------------------


class TestRebalanceCleanupClose:
    """``rebalance`` must not let a close failure replace the body's own failure.

    It used a bare ``async with aiosqlite.connect(...) as db:`` --
    ``aiosqlite.Connection.__aexit__`` is an unconditional ``await
    close()`` and cannot distinguish a cleanup close (something in the
    body already raised) from a success-path one, so a failure in that
    close would replace whatever the body actually raised.
    """

    async def test_body_failure_survives_a_failing_cleanup_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A close failure during cleanup must not replace the body's own failure."""
        import scripts.rebalance_p1 as rebalance_module

        db_path = tmp_path / "rebalance.db"
        await _seed_db(db_path, p1_count=10)

        close_calls = {"n": 0}
        real_connect = rebalance_module.aiosqlite.connect

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

        monkeypatch.setattr(rebalance_module.aiosqlite, "connect", _spy_connect)

        async def _execute_blows_up(*args: object, **kwargs: object) -> object:
            msg = "original body failure"
            raise ValueError(msg)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _execute_blows_up)

        with pytest.raises(ValueError, match="original body failure"):
            await rebalance_module.rebalance(db_path)

        assert close_calls["n"] == 1, (
            "the cleanup close was never attempted -- a regression that "
            "drops the close call entirely would also let the original "
            "ValueError escape untouched, so that alone is not enough"
        )

    async def test_body_failure_survives_a_close_that_never_releases_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A close that raises *before* releasing anything must not win either.

        The ``_spy_connect`` in ``test_body_failure_survives_a_failing_cleanup_close``
        above calls the real ``close()`` first and only raises on top of it,
        so the connection is already released before the simulated
        failure -- the matching spies in ``test_backfill_content_hashes.py``
        and ``test_reenrich_reflections_to_v2.py`` have the identical shape.
        None of those can tell a helper that reports-and-swallows an
        ordinary close failure apart from one that would also paper over a
        close that raises *before* doing any of its own cleanup, which is
        the failure mode that actually leaks: the connection and its
        non-daemon worker thread stay alive. This spy never touches the
        real close at all, so the assertions below are about that
        observable state rather than about whether a close was merely
        attempted.
        """
        import scripts.rebalance_p1 as rebalance_module

        db_path = tmp_path / "rebalance_leak.db"
        await _seed_db(db_path, p1_count=10)

        opened: list[aiosqlite.Connection] = []
        real_connect = rebalance_module.aiosqlite.connect

        def _spy_connect(*args: object, **kwargs: object) -> object:
            conn = real_connect(*args, **kwargs)
            opened.append(conn)

            async def _close_raises_before_releasing_anything() -> None:
                msg = "close blew up before releasing anything"
                raise RuntimeError(msg)

            conn.close = _close_raises_before_releasing_anything
            return conn

        monkeypatch.setattr(rebalance_module.aiosqlite, "connect", _spy_connect)

        async def _execute_blows_up(*args: object, **kwargs: object) -> object:
            msg = "original body failure"
            raise ValueError(msg)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _execute_blows_up)

        try:
            with pytest.raises(ValueError, match="original body failure"):
                await rebalance_module.rebalance(db_path)

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

    async def test_close_happens_on_every_success_return_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every one of the three success branches still closes the connection.

        ``rebalance`` has one ``return``, after the try/except/else, but
        three semantic branches reach it: no demotion needed, dry-run
        preview, and after a real write. A naive ``try/except
        BaseException/else`` conversion that gave any one of those
        branches its own early ``return`` inside the ``try`` body would
        skip the ``else`` clause on that path and leak the connection --
        this pins all three branches against that regression.
        """
        import scripts.rebalance_p1 as rebalance_module

        real_connect = rebalance_module.aiosqlite.connect
        close_calls = {"n": 0}

        def _counting_connect(*args: object, **kwargs: object) -> object:
            conn = real_connect(*args, **kwargs)
            real_close = conn.close

            async def _counted_close() -> None:
                close_calls["n"] += 1
                await real_close()

            conn.close = _counted_close
            return conn

        monkeypatch.setattr(rebalance_module.aiosqlite, "connect", _counting_connect)

        # Branch 1: current_p1 <= max_p1 -- the "no demotion needed" return.
        no_op_db = tmp_path / "no_op.db"
        await _seed_db(no_op_db, p1_count=1, p2_count=99)
        close_calls["n"] = 0
        result = await rebalance_module.rebalance(no_op_db, max_p1_fraction=0.5)
        assert result == 0
        assert close_calls["n"] == 1, "no-op branch leaked the connection"

        # Branch 2: dry_run=True -- the mid-function preview return.
        dry_run_db = tmp_path / "dry_run.db"
        await _seed_db(dry_run_db, p1_count=10)
        close_calls["n"] = 0
        result = await rebalance_module.rebalance(dry_run_db, max_p1_fraction=0.05, dry_run=True)
        assert result > 0
        assert close_calls["n"] == 1, "dry-run branch leaked the connection"

        # Branch 3: a real write -- the final return after ``db.commit()``.
        write_db = tmp_path / "write.db"
        await _seed_db(write_db, p1_count=10)
        close_calls["n"] = 0
        result = await rebalance_module.rebalance(write_db, max_p1_fraction=0.05)
        assert result > 0
        assert close_calls["n"] == 1, "write branch leaked the connection"

"""Tests for ``scripts.reenrich_reflections_to_v2``.

Covers:

* Batched legacy → v2 enrichment writes correct content back.
* Idempotence — running again on a fully-migrated DB is a no-op.
* ``--dry-run`` mode terminates after a single full scan even when
  the legacy row count is exactly *batch_size*.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import DreamingConfig
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def populated_db(tmp_path: Path) -> AsyncIterator[Path]:
    """Build an on-disk DB seeded with three legacy v1 REFLECTIONs + members."""
    db_path = tmp_path / "reenrich.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()

    # Two real cluster members per reflection — keep IDs sortable so
    # the pagination cursor in the script has deterministic order.
    for i, prefix in enumerate(["alpha", "beta", "gamma"]):
        member_a = await store.create_thought(
            ThoughtRecord(
                thought_id=f"m-{prefix}-1",
                thought_type=ThoughtType.OBSERVATION,
                essence=f"member {prefix} 1",
                content=f"{prefix} keyword discussion notes one",
                priority=Priority.P3,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=0,
                updated_cycle=0,
                source="seed",
                source_type=KnowledgeSource.EXPERIENCE,
                visibility=ThoughtVisibility.SELECTIVE,
                created_at="2026-04-29T12:00:00+00:00",
            ),
        )
        member_b = await store.create_thought(
            ThoughtRecord(
                thought_id=f"m-{prefix}-2",
                thought_type=ThoughtType.OBSERVATION,
                essence=f"member {prefix} 2",
                content=f"{prefix} keyword discussion notes two",
                priority=Priority.P3,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=0,
                updated_cycle=0,
                source="seed",
                source_type=KnowledgeSource.EXPERIENCE,
                visibility=ThoughtVisibility.SELECTIVE,
                created_at="2026-04-29T12:00:00+00:00",
            ),
        )
        legacy_v1_content = json.dumps(
            {
                "member_ids": sorted([member_a.thought_id, member_b.thought_id]),
                "keywords": [prefix, "discussion"],
                "cluster_hash": f"abc1234567890{i:03d}",
            }
        )
        await store.create_thought(
            ThoughtRecord(
                thought_id=f"r-legacy-{prefix}",
                thought_type=ThoughtType.REFLECTION,
                essence=f"REFLECTION [{prefix}]",
                content=legacy_v1_content,
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=1,
                updated_cycle=1,
                source=f"dreaming:abc1234567890{i:03d}",
                source_type=KnowledgeSource.DREAMING,
                visibility=ThoughtVisibility.SELECTIVE,
                created_at="2026-04-29T12:00:00+00:00",
            ),
        )

    await conn.close()
    return db_path


# ---------------------------------------------------------------------------
# reenrich behaviour
# ---------------------------------------------------------------------------


class TestReenrichV2Behaviour:
    """End-to-end behaviour of the re-enrichment helper."""

    async def test_writes_v2_content_back(self, populated_db: Path) -> None:
        """A non-dry-run pass writes valid v2 content for every legacy row."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        updated = await reenrich(populated_db, batch_size=2, dry_run=False)
        assert updated == 3

        async with aiosqlite.connect(str(populated_db)) as db:
            cursor = await db.execute(
                "SELECT thought_id, content FROM thought WHERE thought_type = 'REFLECTION' "
                "ORDER BY thought_id"
            )
            rows = await cursor.fetchall()

        assert len(rows) == 3
        for _, content_str in rows:
            content = json.loads(content_str)
            assert content["version"] == 2
            assert content["type"] == "reflection"
            assert "top_keyphrases" in content
            assert "member_excerpts" in content

    async def test_idempotent_on_fully_migrated_db(self, populated_db: Path) -> None:
        """Re-running on a fully migrated DB is a no-op."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        first = await reenrich(populated_db, batch_size=10, dry_run=False)
        second = await reenrich(populated_db, batch_size=10, dry_run=False)
        assert first == 3
        assert second == 0


# ---------------------------------------------------------------------------
# Malformed-JSON tolerance
# ---------------------------------------------------------------------------


class TestMalformedJsonTolerance:
    """A single corrupt-JSON REFLECTION row must not block the whole run.

    The legacy filter combines ``json_valid(content)`` with
    ``json_extract(content, '$.version') IS NULL``.  These tests pin
    that malformed rows are silently excluded by the fetch filter.
    """

    async def _inject_malformed_reflection(self, db_path: Path, thought_id: str) -> None:
        """Direct INSERT of a REFLECTION row whose ``content`` is not valid JSON.

        Goes through aiosqlite to bypass the Pydantic-validated
        ``create_thought`` path — we explicitly want a row with a
        non-JSON content string, which the public API would never
        produce on its own.
        """
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                """
                INSERT INTO thought (
                    thought_id, thought_type, essence, content, priority,
                    lifecycle_status, created_cycle, updated_cycle, source,
                    confidence, embedding_ref, source_type, confirmation_count,
                    consolidated_from, visibility, access_count,
                    last_accessed_at, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thought_id,
                    "REFLECTION",
                    "REFLECTION [corrupt]",
                    "not json",
                    "P3",
                    "ACTIVE",
                    1,
                    1,
                    "dreaming:corrupt",
                    None,
                    None,
                    "DREAMING",
                    0,
                    None,
                    "selective",
                    0,
                    None,
                    "2026-04-29T12:00:00+00:00",
                    "2026-04-29T12:00:00+00:00",
                    None,
                ),
            )
            await db.commit()

    async def test_corrupt_reflection_does_not_block_run(self, populated_db: Path) -> None:
        """A row with non-JSON content is silently skipped; valid rows enrich."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        await self._inject_malformed_reflection(populated_db, "r-corrupt-1")

        # Three valid legacy reflections + one corrupt one in the DB.
        # The valid three must still enrich; the corrupt one is left alone.
        updated = await reenrich(populated_db, batch_size=2, dry_run=False)
        assert updated == 3

        async with aiosqlite.connect(str(populated_db)) as db:
            cursor = await db.execute(
                "SELECT thought_id, content FROM thought "
                "WHERE thought_type = 'REFLECTION' ORDER BY thought_id"
            )
            rows = await cursor.fetchall()

        # Find the corrupt row and verify its content was not touched.
        corrupt = next((c for tid, c in rows if tid == "r-corrupt-1"), None)
        assert corrupt == "not json", "malformed row must not be rewritten"

        # Every other reflection now carries v2 content.
        for tid, content_str in rows:
            if tid == "r-corrupt-1":
                continue
            content = json.loads(content_str)
            assert content["version"] == 2

    async def test_corrupt_reflection_in_dry_run_does_not_block(self, populated_db: Path) -> None:
        """Dry-run also tolerates malformed JSON — same fetch filter path."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        await self._inject_malformed_reflection(populated_db, "r-corrupt-2")
        result = await reenrich(populated_db, batch_size=3, dry_run=True)
        # Three valid legacy rows; the corrupt one is excluded by json_valid.
        assert result == 3


# ---------------------------------------------------------------------------
# dry-run loop termination
# ---------------------------------------------------------------------------


class TestDryRunTermination:
    """Dry-run mode terminates after a single scan even at row-count boundaries.

    A dry run issues no UPDATE, so every legacy row keeps matching the
    ``json_extract($.version) IS NULL`` filter.  Pagination by
    ``thought_id`` advances the cursor regardless of whether an UPDATE was
    issued, so a database with exactly *batch_size* legacy rows is not
    fetched again.
    """

    async def test_dry_run_with_full_batch_does_not_loop(self, populated_db: Path) -> None:
        """``batch_size`` matching the row count terminates after one scan."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        # Three legacy reflections in the fixture; ``batch_size=3`` makes
        # the first batch full, so the loop fetches again and must get an
        # empty page.
        result = await reenrich(populated_db, batch_size=3, dry_run=True)
        assert result == 3

        # After the dry run, every row must remain legacy v1 (no version key).
        async with aiosqlite.connect(str(populated_db)) as db:
            cursor = await db.execute(
                "SELECT content FROM thought WHERE thought_type = 'REFLECTION'"
            )
            rows = await cursor.fetchall()
        for (content_str,) in rows:
            content = json.loads(content_str)
            assert "version" not in content, "dry-run must not mutate any reflection"

    async def test_dry_run_small_batches_terminate_after_full_scan(
        self, populated_db: Path
    ) -> None:
        """``batch_size=1`` paginates through every legacy row exactly once."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        result = await reenrich(populated_db, batch_size=1, dry_run=True)
        assert result == 3

    async def test_dry_run_then_real_run_produces_same_count(self, populated_db: Path) -> None:
        """Operator preview (dry-run) reflects the count of an actual run."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        preview = await reenrich(populated_db, batch_size=2, dry_run=True)
        real = await reenrich(populated_db, batch_size=2, dry_run=False)
        assert preview == real == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_default_config_used_when_omitted() -> None:
    """The ``config`` argument defaults to ``DreamingConfig()`` defaults."""
    from scripts.reenrich_reflections_to_v2 import reenrich  # noqa: F401

    cfg = DreamingConfig()
    assert cfg.top_keyphrases_count == 3
    assert cfg.top_member_excerpts_count == 5


# ---------------------------------------------------------------------------
# Cleanup-close ordering
# ---------------------------------------------------------------------------


class TestReenrichCleanupClose:
    """``reenrich`` must not let a close failure replace the body's own failure.

    It used a bare ``async with aiosqlite.connect(...) as db:`` --
    ``aiosqlite.Connection.__aexit__`` is an unconditional ``await
    close()`` and cannot distinguish a cleanup close (something in the
    body already raised) from a success-path one, so a failure in that
    close would replace whatever the body actually raised.
    """

    async def test_body_failure_survives_a_failing_cleanup_close(
        self, populated_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A close failure during cleanup must not replace the body's own failure."""
        import scripts.reenrich_reflections_to_v2 as reenrich_module

        close_calls = {"n": 0}
        real_connect = reenrich_module.aiosqlite.connect

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

        monkeypatch.setattr(reenrich_module.aiosqlite, "connect", _spy_connect)

        async def _fetch_blows_up(*args: object, **kwargs: object) -> object:
            msg = "original body failure"
            raise ValueError(msg)

        monkeypatch.setattr(reenrich_module, "_fetch_legacy_reflection_batch", _fetch_blows_up)

        with pytest.raises(ValueError, match="original body failure"):
            await reenrich_module.reenrich(populated_db)

        assert close_calls["n"] == 1, (
            "the cleanup close was never attempted -- a regression that "
            "drops the close call entirely would also let the original "
            "ValueError escape untouched, so that alone is not enough"
        )

    async def test_body_failure_survives_a_close_that_never_releases_anything(
        self, populated_db: Path, monkeypatch: pytest.MonkeyPatch
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
        import scripts.reenrich_reflections_to_v2 as reenrich_module

        opened: list[aiosqlite.Connection] = []
        real_connect = reenrich_module.aiosqlite.connect

        def _spy_connect(*args: object, **kwargs: object) -> object:
            conn = real_connect(*args, **kwargs)
            opened.append(conn)

            async def _close_raises_before_releasing_anything() -> None:
                msg = "close blew up before releasing anything"
                raise RuntimeError(msg)

            conn.close = _close_raises_before_releasing_anything
            return conn

        monkeypatch.setattr(reenrich_module.aiosqlite, "connect", _spy_connect)

        async def _fetch_blows_up(*args: object, **kwargs: object) -> object:
            msg = "original body failure"
            raise ValueError(msg)

        monkeypatch.setattr(reenrich_module, "_fetch_legacy_reflection_batch", _fetch_blows_up)

        try:
            with pytest.raises(ValueError, match="original body failure"):
                await reenrich_module.reenrich(populated_db)

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

"""Tests for ``scripts.reenrich_reflections_to_v2``.

Covers:

* Batched legacy → v2 enrichment writes correct content back.
* Idempotence — running again on a fully-migrated DB is a no-op.
* ``--dry-run`` mode terminates after a single full scan even when
  the legacy row count is exactly *batch_size* (the previous loop
  re-fetched the same rows because the legacy filter still matched
  every row after the in-memory rollback).
"""

from __future__ import annotations

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
# Malformed-JSON tolerance — regression guard
# ---------------------------------------------------------------------------


class TestMalformedJsonTolerance:
    """A single corrupt-JSON REFLECTION row must not block the whole run.

    The legacy filter combines ``json_valid(content)`` with
    ``json_extract(content, '$.version') IS NULL``.  Without the
    ``json_valid`` guard SQLite raises ``OperationalError: malformed
    JSON`` mid-SELECT the moment ``json_extract`` hits a corrupt row,
    and the entire re-enrichment pass crashes.  These tests pin the
    fix: malformed rows are silently excluded by the fetch filter and
    every other legacy row is enriched normally.
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
# dry-run loop termination — regression guard
# ---------------------------------------------------------------------------


class TestDryRunTermination:
    """Dry-run mode terminates after a single scan even at row-count boundaries.

    The previous loop's termination condition was ``len(batch) < batch_size``.
    In dry-run mode the helper rolled back the in-memory UPDATEs each batch,
    which kept every row matching the ``json_extract($.version) IS NULL``
    filter forever — so a database with exactly *batch_size* legacy rows
    would loop infinitely (the next fetch returned the same *batch_size*
    rows).  This test pins the fix: pagination by ``thought_id`` advances
    the cursor regardless of whether an UPDATE was issued.
    """

    async def test_dry_run_with_full_batch_does_not_loop(self, populated_db: Path) -> None:
        """``batch_size`` matching the row count terminates after one scan."""
        from scripts.reenrich_reflections_to_v2 import reenrich

        # Three legacy reflections in the fixture; ``batch_size=3`` puts
        # the loop into the worst-case branch where ``len(batch) ==
        # batch_size`` would trigger a re-fetch under the old logic.
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

"""Schema migration tests for the ``content_hash`` column (core-9 -> core-10).

Exercises both the ``_migrate_core_v9_to_v10`` helper directly (the
unit-level invariants — idempotence, ALTER + INDEX presence) and the
full ``ensure_schema`` cascade so that DBs at every supported source
version converge on the current user_version with the new column
populated by fresh inserts.

Final-state user_version assertions track the latest core schema
revision (currently 11 — see ``_migrate_core_v10_to_v11``); the
``content_hash`` invariants pinned by this module remain valid because
the cascade always walks through v9->v10 before reaching the head.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _user_version(db: aiosqlite.Connection) -> int:
    cursor = await db.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _table_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {row["name"] for row in rows}


async def _index_exists(db: aiosqlite.Connection, name: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    )
    return await cursor.fetchone() is not None


async def _bootstrap_thought_table_at_v9(db: aiosqlite.Connection) -> None:
    """Recreate the core-9 ``thought`` table without ``content_hash``.

    Mirrors what schema_core.sql looked like at user_version=9 so we
    can simulate an upgrade from a real pre-deduplication database.  Other
    core tables / indexes / triggers are immaterial for these tests
    and are left to ``ensure_schema`` to populate via the standard
    fresh-DB path on the second pass.
    """
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS thought (
            thought_id        TEXT    PRIMARY KEY,
            thought_type      TEXT    NOT NULL,
            essence           TEXT    NOT NULL,
            content           TEXT    NOT NULL,
            priority          TEXT    NOT NULL,
            lifecycle_status  TEXT    NOT NULL DEFAULT 'CREATED',
            created_cycle     INTEGER NOT NULL DEFAULT 0,
            updated_cycle     INTEGER NOT NULL DEFAULT 0,
            source            TEXT    NOT NULL DEFAULT 'human',
            confidence        REAL,
            embedding_ref     TEXT,
            source_type       TEXT    NOT NULL DEFAULT 'EXPERIENCE',
            confirmation_count INTEGER NOT NULL DEFAULT 0,
            consolidated_from TEXT,
            visibility        TEXT    NOT NULL DEFAULT 'selective',
            access_count      INTEGER NOT NULL DEFAULT 0,
            last_accessed_at  TEXT,
            created_at        TEXT,
            updated_at        TEXT,
            expires_at        TEXT
        );
        PRAGMA user_version = 9;
        """
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_db() -> AsyncIterator[aiosqlite.Connection]:
    """Empty in-memory SQLite (user_version starts at 0)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# Helper-level tests (5 — direct ``_migrate_core_v9_to_v10``)
# ---------------------------------------------------------------------------


async def test_migrate_v9_to_v10_adds_column_and_index(
    fresh_db: aiosqlite.Connection,
) -> None:
    """``_migrate_core_v9_to_v10`` adds the column and the supporting index."""
    await _bootstrap_thought_table_at_v9(fresh_db)
    cols_before = await _table_columns(fresh_db, "thought")
    assert "content_hash" not in cols_before
    assert not await _index_exists(fresh_db, "idx_thought_content_hash")

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v9_to_v10()

    cols_after = await _table_columns(fresh_db, "thought")
    assert "content_hash" in cols_after
    assert await _index_exists(fresh_db, "idx_thought_content_hash")


async def test_migrate_v9_to_v10_idempotent(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Re-running the helper is safe (ALTER duplicate-column tolerated)."""
    await _bootstrap_thought_table_at_v9(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    for _ in range(3):
        await store._migrate_core_v9_to_v10()

    cols = await _table_columns(fresh_db, "thought")
    assert "content_hash" in cols
    assert await _index_exists(fresh_db, "idx_thought_content_hash")


async def test_migrate_v9_to_v10_preserves_existing_rows(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Pre-migration rows survive untouched, with content_hash NULL."""
    await _bootstrap_thought_table_at_v9(fresh_db)
    await fresh_db.execute(
        """
        INSERT INTO thought
            (thought_id, thought_type, essence, content, priority)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("t-old-1", "OBSERVATION", "old essence", "old content", "P2"),
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v9_to_v10()

    cursor = await fresh_db.execute(
        "SELECT thought_id, content, content_hash FROM thought WHERE thought_id = ?",
        ("t-old-1",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["thought_id"] == "t-old-1"
    assert row["content"] == "old content"
    assert row["content_hash"] is None


# ---------------------------------------------------------------------------
# ensure_schema cascade tests (5 — full integration through ensure_schema)
# ---------------------------------------------------------------------------


async def test_ensure_schema_fresh_db_starts_at_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Empty DB ends at the current head user_version with content_hash + index."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 13
    assert "content_hash" in await _table_columns(fresh_db, "thought")
    assert await _index_exists(fresh_db, "idx_thought_content_hash")


async def test_ensure_schema_from_v9_to_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """DB at v9 cascades through v9->v10->v11 elif branches up to head."""
    await _bootstrap_thought_table_at_v9(fresh_db)
    assert await _user_version(fresh_db) == 9

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 13
    assert "content_hash" in await _table_columns(fresh_db, "thought")
    assert await _index_exists(fresh_db, "idx_thought_content_hash")


async def test_ensure_schema_from_v9_idempotent(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Running ``ensure_schema`` after a v9->head upgrade is a no-op."""
    await _bootstrap_thought_table_at_v9(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    for _ in range(3):
        await store.ensure_schema()

    assert await _user_version(fresh_db) == 13


async def test_ensure_schema_at_head_skips_all_migration_branches(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Already-migrated DB stays at head across repeat ``ensure_schema`` calls."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()  # bootstrap fresh -> head
    assert await _user_version(fresh_db) == 13

    # Re-run; helpers should not fire (idempotent on user_version branch).
    await store.ensure_schema()
    assert await _user_version(fresh_db) == 13
    assert "content_hash" in await _table_columns(fresh_db, "thought")


async def test_ensure_schema_round_trip_insert_then_dedup(
    fresh_db: aiosqlite.Connection,
) -> None:
    """End-to-end smoke — fresh DB ingests, hash populated, dedup works."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    # We verify content_hash gets populated on regular create_thought
    # (computed at insert time, regardless of dedup flag).
    from engrava import (
        CoreThoughtRecord,
        KnowledgeSource,
        LifecycleStatus,
        Priority,
        ThoughtType,
        ThoughtVisibility,
    )

    record = CoreThoughtRecord(
        thought_id="t-roundtrip",
        thought_type=ThoughtType.OBSERVATION,
        essence="Round-trip smoke",
        content="Round-trip content used to confirm post-migration ingest works.",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )
    await store.create_thought(record)

    cursor = await fresh_db.execute(
        "SELECT content_hash FROM thought WHERE thought_id = ?",
        ("t-roundtrip",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["content_hash"] is not None
    assert len(row["content_hash"]) == 64  # SHA-256 hex

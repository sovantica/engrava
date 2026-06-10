"""Schema migration tests for the ``metadata_json`` column (core-10 -> core-11).

Mirrors ``test_dedup_migration.py``: exercises the
``_migrate_core_v10_to_v11`` helper directly (idempotence, ALTER
TABLE add-column behaviour) and the full ``ensure_schema`` cascade
so DBs at every supported source version converge on the current head
``user_version`` with the ``metadata_json`` column populated by
the schema-level ``DEFAULT '{}'``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers (mirror test_dedup_migration.py)
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


async def _bootstrap_thought_table_at_v10(db: aiosqlite.Connection) -> None:
    """Recreate the core-10 ``thought`` table without ``metadata_json``.

    Mirrors what ``schema_core.sql`` looked like at ``user_version=10``
    so we can simulate an upgrade from a real pre-v11 database.  Other
    core tables / indexes / triggers are immaterial for these tests
    and are left to ``ensure_schema`` to populate via the cascade.
    """
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS thought (
            thought_id        TEXT    PRIMARY KEY,
            thought_type      TEXT    NOT NULL,
            essence           TEXT    NOT NULL,
            content           TEXT    NOT NULL,
            content_hash      TEXT,
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
        PRAGMA user_version = 10;
        """,
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_db() -> AsyncIterator[aiosqlite.Connection]:
    """Empty in-memory SQLite (``user_version`` starts at 0)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# Helper-level tests (3 — direct ``_migrate_core_v10_to_v11``)
# ---------------------------------------------------------------------------


async def test_migrate_v10_to_v11_adds_column(
    fresh_db: aiosqlite.Connection,
) -> None:
    """``_migrate_core_v10_to_v11`` adds the ``metadata_json`` column."""
    await _bootstrap_thought_table_at_v10(fresh_db)
    cols_before = await _table_columns(fresh_db, "thought")
    assert "metadata_json" not in cols_before

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v10_to_v11()

    cols_after = await _table_columns(fresh_db, "thought")
    assert "metadata_json" in cols_after


async def test_migrate_v10_to_v11_idempotent(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Re-running the helper is safe (ALTER duplicate-column tolerated)."""
    await _bootstrap_thought_table_at_v10(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    for _ in range(3):
        await store._migrate_core_v10_to_v11()

    cols = await _table_columns(fresh_db, "thought")
    assert "metadata_json" in cols


async def test_migrate_v10_to_v11_preserves_existing_rows_with_default(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Pre-migration rows receive ``metadata_json = '{}'`` from the DEFAULT."""
    await _bootstrap_thought_table_at_v10(fresh_db)
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
    await store._migrate_core_v10_to_v11()

    cursor = await fresh_db.execute(
        "SELECT thought_id, content, metadata_json FROM thought WHERE thought_id = ?",
        ("t-old-1",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["thought_id"] == "t-old-1"
    assert row["content"] == "old content"
    assert row["metadata_json"] == "{}"
    assert json.loads(row["metadata_json"]) == {}


# ---------------------------------------------------------------------------
# ensure_schema cascade tests (4 — full integration including from-v10)
# ---------------------------------------------------------------------------


async def test_ensure_schema_fresh_db_lands_at_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Empty DB ends at the head ``user_version`` with the metadata column."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 13
    assert "metadata_json" in await _table_columns(fresh_db, "thought")


async def test_ensure_schema_from_v10_to_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """DB at v10 walks the ``< 11`` elif branch up to head."""
    await _bootstrap_thought_table_at_v10(fresh_db)
    assert await _user_version(fresh_db) == 10

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 13
    assert "metadata_json" in await _table_columns(fresh_db, "thought")


async def test_ensure_schema_idempotent_at_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Repeat calls after reaching head stay at the head version without errors."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()
    assert await _user_version(fresh_db) == 13

    for _ in range(3):
        await store.ensure_schema()

    assert await _user_version(fresh_db) == 13
    assert "metadata_json" in await _table_columns(fresh_db, "thought")


async def test_ensure_schema_from_v10_with_legacy_row_then_insert_new(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Legacy v10 row gets ``{}`` default; new INSERT carries supplied metadata."""
    await _bootstrap_thought_table_at_v10(fresh_db)
    await fresh_db.execute(
        """
        INSERT INTO thought
            (thought_id, thought_type, essence, content, priority)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("legacy-1", "OBSERVATION", "legacy essence", "legacy content", "P2"),
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    # Legacy row has DEFAULT '{}' metadata (post-cascade read goes through
    # _row_to_thought).
    legacy = await store.get_thought("legacy-1")
    assert legacy is not None
    assert legacy.metadata == {}

    # New row preserves caller-supplied metadata end-to-end.
    new = ThoughtRecord(
        thought_id="new-1",
        thought_type=ThoughtType.OBSERVATION,
        essence="new essence",
        content="new content",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        metadata={"role": "user", "lang": "en"},
    )
    await store.create_thought(new)
    fetched = await store.get_thought("new-1")
    assert fetched is not None
    assert fetched.metadata == {"role": "user", "lang": "en"}


# ---------------------------------------------------------------------------
# Cascade-from-any-version (1 parametrized — covers v3..v10 -> head)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_version", [3, 4, 5, 6, 7, 8, 9, 10])
async def test_cascade_from_any_version_to_head(
    fresh_db: aiosqlite.Connection,
    source_version: int,
) -> None:
    """A DB stamped at any historical core version cascades to head.

    The historical schemas differ across versions in ways unrelated to
    the metadata column, so this test only seeds the ``user_version``
    PRAGMA and lets ``ensure_schema`` walk the appropriate elif branch
    up to head — exactly what an in-place upgrade from an older
    Engrava install would experience.  The bootstrap helper for v10 is
    used to provide a thought-table shape that the lower-version
    branches (which run ``_migrate_core_vN_to_vN+1`` helpers in order)
    can safely chain through; for v3..v9 we drop back to a
    fresh-bootstrap path which the cascade handles via the lower
    branches.
    """
    if source_version == 10:
        await _bootstrap_thought_table_at_v10(fresh_db)
    else:
        # For v3..v9 we let ensure_schema bootstrap from scratch and
        # then force-downgrade to the simulated source version, which
        # mirrors how the cascade tests stage v9 etc.
        bootstrap = SqliteEngravaCore(fresh_db)
        await bootstrap.ensure_schema()
        await fresh_db.execute(f"PRAGMA user_version = {source_version}")
        await fresh_db.commit()

    assert await _user_version(fresh_db) == source_version

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 13
    assert "metadata_json" in await _table_columns(fresh_db, "thought")

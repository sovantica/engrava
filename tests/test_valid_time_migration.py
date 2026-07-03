"""Schema migration tests for the valid-time columns (core-12 -> core-13).

Exercises ``_migrate_core_v12_to_v13`` directly (idempotence, ALTER
add-column behaviour, asymmetric backfill, index creation) and the full
``ensure_schema`` cascade so a database stamped at any supported source
version converges on the head ``user_version`` with the ``valid_from`` /
``valid_until`` columns and their indexes on both the ``thought`` and
``edge`` tables.

The feature introduces a second time axis ("valid time" — when a fact is
true in the world) alongside the existing transaction time
(``created_at``). Thought rows are backfilled from ``created_at``; edge
rows are deliberately left ``NULL`` because the edge table has no
calendar timestamp to source from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.domain.enums import EdgeType, LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import EdgeRecord, ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_VALID_THOUGHT_INDEXES = (
    "idx_thought_valid_from",
    "idx_thought_valid_until",
    "idx_thought_valid_range",
)
_VALID_EDGE_INDEXES = (
    "idx_edge_valid_from",
    "idx_edge_valid_until",
    "idx_edge_valid_range",
)
_ALL_VALID_INDEXES = _VALID_THOUGHT_INDEXES + _VALID_EDGE_INDEXES


# ---------------------------------------------------------------------------
# Helpers (mirror test_metadata_migration.py / test_dedup_migration.py)
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


async def _table_info(db: aiosqlite.Connection, table: str) -> list[tuple[str, str]]:
    """Return ``(name, declared_type)`` pairs for ``table`` in cid order."""
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return [(row["name"], row["type"]) for row in rows]


async def _index_names(db: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
        (table,),
    )
    rows = await cursor.fetchall()
    return {row["name"] for row in rows if row["name"] is not None}


async def _index_exists(db: aiosqlite.Connection, name: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    )
    return await cursor.fetchone() is not None


async def _row_count(db: aiosqlite.Connection, table: str) -> int:
    cursor = await db.execute(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608
    row = await cursor.fetchone()
    assert row is not None
    return int(row["n"])


async def _bootstrap_core_at_v12(db: aiosqlite.Connection) -> None:
    """Recreate a faithful core-12 ``thought`` + ``edge`` schema.

    Mirrors what ``schema_core.sql`` looked like at ``user_version=12``
    (referential integrity present, but no valid-time axis). The
    pre-valid-time indexes a real v12 install carries on these two tables
    (``idx_edge_type_from``, ``idx_thought_content_hash``,
    ``idx_thought_expires``) are recreated too, so that after the upgrade a
    migrated database is structurally identical to a freshly bootstrapped
    one. The valid-time columns and their six indexes are deliberately
    absent — that is precisely the surface the upgrade re-adds.
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
            expires_at        TEXT,
            metadata_json     TEXT    NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS edge (
            edge_id           TEXT PRIMARY KEY,
            from_thought_id   TEXT NOT NULL,
            to_thought_id     TEXT NOT NULL,
            edge_type         TEXT NOT NULL,
            weight            REAL NOT NULL DEFAULT 0.5,
            created_cycle     INTEGER NOT NULL DEFAULT 0,
            source            TEXT NOT NULL DEFAULT 'EXPERIENCE',
            decay_multiplier  REAL NOT NULL DEFAULT 1.0,
            UNIQUE(from_thought_id, to_thought_id, edge_type),
            FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE,
            FOREIGN KEY (to_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_thought_expires ON thought(expires_at)
            WHERE expires_at IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash);
        CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id);
        PRAGMA user_version = 12;
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
# Helper-level tests (direct ``_migrate_core_v12_to_v13``)
# ---------------------------------------------------------------------------


async def test_migrate_v12_to_v13_adds_columns_to_both_tables(
    fresh_db: aiosqlite.Connection,
) -> None:
    """The migration adds valid-time columns to ``thought`` and ``edge``."""
    await _bootstrap_core_at_v12(fresh_db)
    assert "valid_from" not in await _table_columns(fresh_db, "thought")
    assert "valid_from" not in await _table_columns(fresh_db, "edge")

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v12_to_v13()

    thought_cols = await _table_columns(fresh_db, "thought")
    edge_cols = await _table_columns(fresh_db, "edge")
    assert {"valid_from", "valid_until"} <= thought_cols
    assert {"valid_from", "valid_until"} <= edge_cols


async def test_migrate_v12_to_v13_creates_all_six_indexes(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Three partial indexes per table are created (six total)."""
    await _bootstrap_core_at_v12(fresh_db)
    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v12_to_v13()

    for index_name in _ALL_VALID_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_migrate_v12_to_v13_idempotent(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Re-running the helper is safe: no duplicate columns or indexes, no error."""
    await _bootstrap_core_at_v12(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    for _ in range(3):
        await store._migrate_core_v12_to_v13()

    # Exactly one occurrence of each valid-time column per table.
    for table in ("thought", "edge"):
        info = await _table_info(fresh_db, table)
        names = [name for name, _ in info]
        assert names.count("valid_from") == 1, table
        assert names.count("valid_until") == 1, table
    # All six indexes present exactly once (sqlite_master keys by name).
    for index_name in _ALL_VALID_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_migrate_v12_to_v13_backfills_thought_from_created_at(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A thought with non-NULL ``created_at`` gets ``valid_from == created_at``."""
    await _bootstrap_core_at_v12(fresh_db)
    await fresh_db.execute(
        """
        INSERT INTO thought
            (thought_id, thought_type, essence, content, priority, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("t-dated", "OBSERVATION", "e", "c", "P2", "2026-01-02T03:04:05+00:00"),
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v12_to_v13()

    cursor = await fresh_db.execute(
        "SELECT created_at, valid_from, valid_until FROM thought WHERE thought_id = ?",
        ("t-dated",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["valid_from"] == row["created_at"]
    assert row["valid_from"] == "2026-01-02T03:04:05+00:00"
    # valid_until is always left open by the backfill.
    assert row["valid_until"] is None


async def test_migrate_v12_to_v13_does_not_fabricate_for_null_created_at(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A thought with NULL ``created_at`` keeps ``valid_from`` NULL (no fabrication)."""
    await _bootstrap_core_at_v12(fresh_db)
    await fresh_db.execute(
        """
        INSERT INTO thought (thought_id, thought_type, essence, content, priority)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("t-legacy", "OBSERVATION", "e", "c", "P2"),
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v12_to_v13()

    cursor = await fresh_db.execute(
        "SELECT created_at, valid_from FROM thought WHERE thought_id = ?",
        ("t-legacy",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["created_at"] is None
    assert row["valid_from"] is None


async def test_migrate_v12_to_v13_leaves_all_edges_null(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Edges are intentionally not backfilled — both valid-time fields stay NULL."""
    await _bootstrap_core_at_v12(fresh_db)
    await fresh_db.execute(
        """
        INSERT INTO thought (thought_id, thought_type, essence, content, priority)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("t-1", "OBSERVATION", "e", "c", "P2"),
    )
    await fresh_db.execute(
        """
        INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type, created_cycle)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("e-1", "t-1", "t-1", "ASSOCIATED", 7),
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v12_to_v13()

    cursor = await fresh_db.execute(
        "SELECT valid_from, valid_until FROM edge WHERE edge_id = ?",
        ("e-1",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["valid_from"] is None
    assert row["valid_until"] is None


async def test_migrate_v12_to_v13_tolerates_absent_edge_table(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A thought-only partial bootstrap (no ``edge`` table) migrates cleanly.

    Some databases carry only the ``thought`` table at this point — the
    ``edge`` table is created lazily. The migration must skip the edge
    column-adds and edge indexes rather than raising ``no such table``,
    while still upgrading ``thought`` fully.
    """
    await fresh_db.executescript(
        """
        CREATE TABLE thought (
            thought_id    TEXT PRIMARY KEY,
            thought_type  TEXT NOT NULL,
            essence       TEXT NOT NULL,
            content       TEXT NOT NULL,
            priority      TEXT NOT NULL,
            created_at    TEXT
        );
        PRAGMA user_version = 12;
        """,
    )
    await fresh_db.commit()
    assert "edge" not in {
        row["name"]
        for row in await (
            await fresh_db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ).fetchall()
    }

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v12_to_v13()  # must not raise

    thought_cols = await _table_columns(fresh_db, "thought")
    assert {"valid_from", "valid_until"} <= thought_cols
    for index_name in _VALID_THOUGHT_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name
    # No edge indexes were created (the table is absent).
    for index_name in _VALID_EDGE_INDEXES:
        assert not await _index_exists(fresh_db, index_name), index_name


async def test_migrate_v12_to_v13_preserves_row_counts(
    fresh_db: aiosqlite.Connection,
) -> None:
    """The additive migration changes no row counts in either table."""
    await _bootstrap_core_at_v12(fresh_db)
    await fresh_db.executemany(
        """
        INSERT INTO thought (thought_id, thought_type, essence, content, priority, created_at)
        VALUES (?, 'OBSERVATION', 'e', 'c', 'P2', ?)
        """,
        [("t-1", "2026-01-01T00:00:00+00:00"), ("t-2", None), ("t-3", "2026-02-02T00:00:00+00:00")],
    )
    await fresh_db.execute(
        """
        INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type)
        VALUES (?, 't-1', 't-2', 'ASSOCIATED'), (?, 't-2', 't-3', 'CONSOLIDATED_FROM')
        """,
        ("e-1", "e-2"),
    )
    await fresh_db.commit()
    thoughts_before = await _row_count(fresh_db, "thought")
    edges_before = await _row_count(fresh_db, "edge")

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v12_to_v13()

    assert await _row_count(fresh_db, "thought") == thoughts_before == 3
    assert await _row_count(fresh_db, "edge") == edges_before == 2


# ---------------------------------------------------------------------------
# ensure_schema cascade tests
# ---------------------------------------------------------------------------


async def test_ensure_schema_fresh_db_lands_at_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """An empty DB bootstraps straight to v13 with all valid-time columns."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 17
    assert {"valid_from", "valid_until"} <= await _table_columns(fresh_db, "thought")
    assert {"valid_from", "valid_until"} <= await _table_columns(fresh_db, "edge")
    for index_name in _ALL_VALID_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_ensure_schema_from_v12_to_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A v12 DB walks the ``< 13`` branch up to head."""
    await _bootstrap_core_at_v12(fresh_db)
    assert await _user_version(fresh_db) == 12

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 17
    assert {"valid_from", "valid_until"} <= await _table_columns(fresh_db, "thought")
    assert {"valid_from", "valid_until"} <= await _table_columns(fresh_db, "edge")


async def test_ensure_schema_idempotent_at_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Repeated ``ensure_schema`` calls stay at v13 without error."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()
    assert await _user_version(fresh_db) == 17

    for _ in range(3):
        await store.ensure_schema()

    assert await _user_version(fresh_db) == 17


async def test_ensure_schema_from_v12_backfills_and_preserves_counts(
    fresh_db: aiosqlite.Connection,
) -> None:
    """End-to-end cascade: dated thought backfilled, legacy + edges NULL, counts kept."""
    await _bootstrap_core_at_v12(fresh_db)
    await fresh_db.execute(
        """
        INSERT INTO thought (thought_id, thought_type, essence, content, priority, created_at)
        VALUES (?, 'OBSERVATION', 'e', 'c', 'P2', ?)
        """,
        ("t-dated", "2026-03-04T05:06:07+00:00"),
    )
    await fresh_db.execute(
        """
        INSERT INTO thought (thought_id, thought_type, essence, content, priority)
        VALUES (?, 'OBSERVATION', 'e', 'c', 'P2')
        """,
        ("t-legacy",),
    )
    await fresh_db.execute(
        """
        INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type)
        VALUES (?, 't-dated', 't-legacy', 'ASSOCIATED')
        """,
        ("e-1",),
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 17
    assert await _row_count(fresh_db, "thought") == 2
    assert await _row_count(fresh_db, "edge") == 1

    dated = await fresh_db.execute("SELECT valid_from FROM thought WHERE thought_id = 't-dated'")
    dated_row = await dated.fetchone()
    assert dated_row is not None
    assert dated_row["valid_from"] == "2026-03-04T05:06:07+00:00"

    legacy = await fresh_db.execute("SELECT valid_from FROM thought WHERE thought_id = 't-legacy'")
    legacy_row = await legacy.fetchone()
    assert legacy_row is not None
    assert legacy_row["valid_from"] is None

    edge = await fresh_db.execute("SELECT valid_from, valid_until FROM edge WHERE edge_id = 'e-1'")
    edge_row = await edge.fetchone()
    assert edge_row is not None
    assert edge_row["valid_from"] is None
    assert edge_row["valid_until"] is None


@pytest.mark.parametrize("source_version", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
async def test_cascade_from_any_version_to_head(
    fresh_db: aiosqlite.Connection,
    source_version: int,
) -> None:
    """A DB stamped at any historical core version cascades to head v15.

    Only the ``user_version`` PRAGMA is seeded; ``ensure_schema`` walks
    the matching elif branch up to head, exactly as an in-place upgrade
    from an older install would.
    """
    bootstrap = SqliteEngravaCore(fresh_db)
    await bootstrap.ensure_schema()
    await fresh_db.execute(f"PRAGMA user_version = {source_version}")
    await fresh_db.commit()
    assert await _user_version(fresh_db) == source_version

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == 17
    assert {"valid_from", "valid_until"} <= await _table_columns(fresh_db, "thought")
    assert {"valid_from", "valid_until"} <= await _table_columns(fresh_db, "edge")


# ---------------------------------------------------------------------------
# Fresh-head == migrated-head structural equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["thought", "edge"])
async def test_fresh_equals_migrated_schema(table: str) -> None:
    """A fresh-bootstrap head DB is structurally identical to a migrated head DB.

    Compares ``PRAGMA table_info`` (column name + declared type) and the
    set of indexes for ``table`` between a database that ran the
    fresh-create DDL and one upgraded in place from v12.
    """
    fresh = await aiosqlite.connect(":memory:")
    fresh.row_factory = aiosqlite.Row
    migrated = await aiosqlite.connect(":memory:")
    migrated.row_factory = aiosqlite.Row
    try:
        await SqliteEngravaCore(fresh).ensure_schema()

        await _bootstrap_core_at_v12(migrated)
        await SqliteEngravaCore(migrated).ensure_schema()

        assert await _user_version(fresh) == await _user_version(migrated) == 17
        assert await _table_info(fresh, table) == await _table_info(migrated, table)
        assert await _index_names(fresh, table) == await _index_names(migrated, table)
    finally:
        await fresh.close()
        await migrated.close()


# ---------------------------------------------------------------------------
# Round-trip through the public CRUD path
# ---------------------------------------------------------------------------


async def test_thought_round_trip_preserves_valid_time(
    fresh_db: aiosqlite.Connection,
) -> None:
    """``create_thought`` then read preserves set valid-time; unset round-trips None."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    with_valid = ThoughtRecord(
        thought_id="t-valid",
        thought_type=ThoughtType.OBSERVATION,
        essence="e",
        content="c",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2026-12-31T00:00:00+00:00",
    )
    await store.create_thought(with_valid)
    fetched = await store.get_thought("t-valid")
    assert fetched is not None
    assert fetched.valid_from == "2026-01-01T00:00:00+00:00"
    assert fetched.valid_until == "2026-12-31T00:00:00+00:00"

    without_valid = ThoughtRecord(
        thought_id="t-none",
        thought_type=ThoughtType.OBSERVATION,
        essence="e",
        content="c",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )
    await store.create_thought(without_valid)
    fetched_none = await store.get_thought("t-none")
    assert fetched_none is not None
    assert fetched_none.valid_from is None
    assert fetched_none.valid_until is None


async def test_edge_round_trip_preserves_valid_time(
    fresh_db: aiosqlite.Connection,
) -> None:
    """``create_edge`` then read preserves set valid-time; unset round-trips None."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    parent = ThoughtRecord(
        thought_id="t-parent",
        thought_type=ThoughtType.OBSERVATION,
        essence="e",
        content="c",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )
    await store.create_thought(parent)

    edge_with_valid = EdgeRecord(
        edge_id="e-valid",
        from_thought_id="t-parent",
        to_thought_id="t-parent",
        edge_type=EdgeType.ASSOCIATED,
        weight=0.5,
        created_cycle=1,
        valid_from="2026-05-01T00:00:00+00:00",
        valid_until="2026-06-01T00:00:00+00:00",
    )
    await store.create_edge(edge_with_valid)
    edges = await store.get_edges("t-parent")
    by_id = {edge.edge_id: edge for edge in edges}
    assert by_id["e-valid"].valid_from == "2026-05-01T00:00:00+00:00"
    assert by_id["e-valid"].valid_until == "2026-06-01T00:00:00+00:00"

    edge_without_valid = EdgeRecord(
        edge_id="e-none",
        from_thought_id="t-parent",
        to_thought_id="t-parent",
        edge_type=EdgeType.CONSOLIDATED_FROM,
        weight=0.5,
        created_cycle=1,
    )
    await store.create_edge(edge_without_valid)
    edges_after = {e.edge_id: e for e in await store.get_edges("t-parent")}
    assert edges_after["e-none"].valid_from is None
    assert edges_after["e-none"].valid_until is None


async def test_existing_list_unaffected_by_additive_columns(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A non-temporal listing returns the same rows regardless of valid-time."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    ids = ("t-a", "t-b", "t-c")
    for index, thought_id in enumerate(ids):
        await store.create_thought(
            ThoughtRecord(
                thought_id=thought_id,
                thought_type=ThoughtType.OBSERVATION,
                essence="e",
                content="c",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.CREATED,
                created_cycle=index,
                updated_cycle=index,
                source="test",
                valid_from="2026-01-01T00:00:00+00:00" if index == 0 else None,
            )
        )

    for thought_id in ids:
        fetched = await store.get_thought(thought_id)
        assert fetched is not None
        assert fetched.thought_id == thought_id

"""Schema migration tests for the hot-path indexes (core-13 -> core-14).

Exercises ``_migrate_core_v13_to_v14`` directly (idempotence, table-absence
tolerance, index creation) and the full ``ensure_schema`` cascade so a
database stamped at v13 converges on the head ``user_version`` with the four
new hot-path indexes on ``thought``, ``edge`` and ``embedding``.

The migration is purely additive: it creates indexes that back the equality
filters and the sort column hit on every common read, without touching any
row or column. A freshly bootstrapped database must carry exactly the same
indexes as one upgraded in place from v13.

Also verifies the connection-init PRAGMA tuning (``synchronous=NORMAL`` and
``busy_timeout=5000``) on both connection-init paths
(``SqliteEngravaCore.from_config`` and ``ServiceManager``), and that the new
equality / sort indexes are actually chosen by the query planner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# The four indexes introduced by the v13 -> v14 migration, grouped by the
# table they live on.
_NEW_THOUGHT_INDEXES = (
    "idx_thought_updated_cycle",
    "idx_thought_type",
)
_NEW_EDGE_INDEXES = ("idx_edge_to_thought",)
_NEW_EMBEDDING_INDEXES = ("idx_embedding_owner",)
_ALL_NEW_INDEXES = _NEW_THOUGHT_INDEXES + _NEW_EDGE_INDEXES + _NEW_EMBEDDING_INDEXES

_HEAD_VERSION = 19


# ---------------------------------------------------------------------------
# Helpers (mirror test_valid_time_migration.py)
# ---------------------------------------------------------------------------


async def _user_version(db: aiosqlite.Connection) -> int:
    cursor = await db.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


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


async def _table_names(db: aiosqlite.Connection) -> set[str]:
    cursor = await db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    rows = await cursor.fetchall()
    return {row["name"] for row in rows}


async def _bootstrap_core_at_v13(db: aiosqlite.Connection) -> None:
    """Recreate a faithful core-13 ``thought`` + ``edge`` + ``embedding`` schema.

    Mirrors what ``schema_core.sql`` looked like at ``user_version=13``
    (valid-time axis present, but without the four hot-path indexes the
    v14 upgrade adds). The pre-v14 indexes a real v13 install carries on
    these tables are recreated too, so that after the upgrade a migrated
    database is structurally identical to a freshly bootstrapped one. The
    four new indexes are deliberately absent — that is precisely the
    surface the upgrade re-adds.
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
            metadata_json     TEXT    NOT NULL DEFAULT '{}',
            valid_from        TEXT,
            valid_until       TEXT
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
            valid_from        TEXT,
            valid_until       TEXT,
            UNIQUE(from_thought_id, to_thought_id, edge_type),
            FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE,
            FOREIGN KEY (to_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS embedding (
            embedding_id TEXT PRIMARY KEY,
            owner_type   TEXT    NOT NULL,
            owner_id     TEXT    NOT NULL,
            model_name   TEXT    NOT NULL,
            dimension    INTEGER NOT NULL,
            vector_blob  BLOB    NOT NULL,
            created_at   TEXT    NOT NULL,
            FOREIGN KEY (owner_id) REFERENCES thought(thought_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_thought_expires ON thought(expires_at)
            WHERE expires_at IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash);
        CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id);
        CREATE INDEX IF NOT EXISTS idx_thought_valid_from ON thought(valid_from);
        CREATE INDEX IF NOT EXISTS idx_thought_valid_until ON thought(valid_until)
            WHERE valid_until IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_thought_valid_range
            ON thought(valid_from, valid_until);
        CREATE INDEX IF NOT EXISTS idx_edge_valid_from ON edge(valid_from);
        CREATE INDEX IF NOT EXISTS idx_edge_valid_until ON edge(valid_until)
            WHERE valid_until IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_edge_valid_range ON edge(valid_from, valid_until);
        PRAGMA user_version = 13;
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
# Helper-level tests (direct ``_migrate_core_v13_to_v14``)
# ---------------------------------------------------------------------------


async def test_v13_base_lacks_the_new_indexes(fresh_db: aiosqlite.Connection) -> None:
    """Guard: the v13 base fixture genuinely omits the four new indexes.

    This is the pre-fix structural assertion — a v13 database has none of
    the hot-path indexes, which is exactly the gap the migration closes.
    """
    await _bootstrap_core_at_v13(fresh_db)
    for index_name in _ALL_NEW_INDEXES:
        assert not await _index_exists(fresh_db, index_name), index_name


async def test_migrate_v13_to_v14_creates_all_four_indexes(
    fresh_db: aiosqlite.Connection,
) -> None:
    """The migration creates all four hot-path indexes."""
    await _bootstrap_core_at_v13(fresh_db)
    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v13_to_v14()

    for index_name in _ALL_NEW_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_migrate_v13_to_v14_idempotent(fresh_db: aiosqlite.Connection) -> None:
    """Re-running the helper is safe: no duplicate indexes, no error."""
    await _bootstrap_core_at_v13(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    for _ in range(3):
        await store._migrate_core_v13_to_v14()

    for index_name in _ALL_NEW_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_migrate_v13_to_v14_tolerates_absent_edge_and_embedding(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A thought-only partial bootstrap migrates cleanly.

    Some databases carry only the ``thought`` table at this point — the
    ``edge`` and ``embedding`` tables are created lazily. The migration
    must skip the edge / embedding indexes rather than raising ``no such
    table``, while still indexing ``thought`` fully.
    """
    await fresh_db.executescript(
        """
        CREATE TABLE thought (
            thought_id    TEXT PRIMARY KEY,
            thought_type  TEXT NOT NULL,
            essence       TEXT NOT NULL,
            content       TEXT NOT NULL,
            priority      TEXT NOT NULL,
            updated_cycle INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT
        );
        PRAGMA user_version = 13;
        """,
    )
    await fresh_db.commit()
    tables = await _table_names(fresh_db)
    assert "edge" not in tables
    assert "embedding" not in tables

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v13_to_v14()  # must not raise

    for index_name in _NEW_THOUGHT_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name
    for index_name in _NEW_EDGE_INDEXES + _NEW_EMBEDDING_INDEXES:
        assert not await _index_exists(fresh_db, index_name), index_name


async def test_migrate_v13_to_v14_tolerates_absent_indexed_column(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A minimal legacy ``thought`` table missing ``updated_cycle`` migrates cleanly.

    A hand-rolled or very old schema may carry a ``thought`` table that
    predates a now-indexed column. The migration must skip that single
    index (guarded by ``_column_exists``) rather than raising ``no such
    column``, while still creating the indexes for the columns present.
    """
    await fresh_db.executescript(
        """
        CREATE TABLE thought (
            thought_id    TEXT PRIMARY KEY,
            thought_type  TEXT NOT NULL,
            essence       TEXT NOT NULL,
            content       TEXT NOT NULL,
            priority      TEXT NOT NULL
        );
        PRAGMA user_version = 13;
        """,
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v13_to_v14()  # must not raise

    # thought_type is present, so its index is created.
    assert await _index_exists(fresh_db, "idx_thought_type")
    # updated_cycle is absent, so its index is skipped (not fabricated).
    assert not await _index_exists(fresh_db, "idx_thought_updated_cycle")


# ---------------------------------------------------------------------------
# ensure_schema cascade tests
# ---------------------------------------------------------------------------


async def test_ensure_schema_fresh_db_lands_at_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """An empty DB bootstraps straight to head with all four new indexes."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    for index_name in _ALL_NEW_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_ensure_schema_from_v13_to_head(fresh_db: aiosqlite.Connection) -> None:
    """A v13 DB walks the ``< 14`` branch up to head and gains the indexes."""
    await _bootstrap_core_at_v13(fresh_db)
    assert await _user_version(fresh_db) == 13

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    for index_name in _ALL_NEW_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_ensure_schema_from_empty_v13_base_to_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """An *empty* (no rows) v13 base also lands at head with the indexes."""
    await _bootstrap_core_at_v13(fresh_db)
    assert await _row_count(fresh_db, "thought") == 0

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    for index_name in _ALL_NEW_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_ensure_schema_idempotent_at_head(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Repeated ``ensure_schema`` calls stay at head without error."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()
    assert await _user_version(fresh_db) == _HEAD_VERSION

    for _ in range(3):
        await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    for index_name in _ALL_NEW_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


@pytest.mark.parametrize("source_version", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
async def test_cascade_from_any_version_to_head(
    fresh_db: aiosqlite.Connection,
    source_version: int,
) -> None:
    """A DB stamped at any historical core version cascades to the head version.

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

    assert await _user_version(fresh_db) == _HEAD_VERSION
    for index_name in _ALL_NEW_INDEXES:
        assert await _index_exists(fresh_db, index_name), index_name


async def test_ensure_schema_from_v13_preserves_row_counts(
    fresh_db: aiosqlite.Connection,
) -> None:
    """The additive migration changes no row counts (zero data loss)."""
    await _bootstrap_core_at_v13(fresh_db)
    await fresh_db.executemany(
        """
        INSERT INTO thought (thought_id, thought_type, essence, content, priority)
        VALUES (?, 'OBSERVATION', 'e', 'c', 'P2')
        """,
        [("t-1",), ("t-2",), ("t-3",)],
    )
    await fresh_db.execute(
        """
        INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type)
        VALUES (?, 't-1', 't-2', 'ASSOCIATED'), (?, 't-2', 't-3', 'CONSOLIDATED_FROM')
        """,
        ("e-1", "e-2"),
    )
    await fresh_db.execute(
        """
        INSERT INTO embedding
            (embedding_id, owner_type, owner_id, model_name, dimension,
             vector_blob, created_at)
        VALUES (?, 'THOUGHT', 't-1', 'm', 3, ?, '2026-01-01T00:00:00+00:00')
        """,
        ("emb-1", b"\x00\x01\x02"),
    )
    await fresh_db.commit()
    thoughts_before = await _row_count(fresh_db, "thought")
    edges_before = await _row_count(fresh_db, "edge")
    embeddings_before = await _row_count(fresh_db, "embedding")

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    assert await _row_count(fresh_db, "thought") == thoughts_before == 3
    assert await _row_count(fresh_db, "edge") == edges_before == 2
    assert await _row_count(fresh_db, "embedding") == embeddings_before == 1


# ---------------------------------------------------------------------------
# Fresh-head == migrated-head structural equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["thought", "edge", "embedding"])
async def test_fresh_equals_migrated_schema(table: str) -> None:
    """A fresh-bootstrap head DB has the same index set as a migrated head DB.

    Compares the set of indexes for ``table`` between a database that ran
    the fresh-create DDL and one upgraded in place from v13. The migration
    must leave the on-disk index surface identical to a fresh bootstrap.
    """
    fresh = await aiosqlite.connect(":memory:")
    fresh.row_factory = aiosqlite.Row
    migrated = await aiosqlite.connect(":memory:")
    migrated.row_factory = aiosqlite.Row
    try:
        await SqliteEngravaCore(fresh).ensure_schema()

        await _bootstrap_core_at_v13(migrated)
        await SqliteEngravaCore(migrated).ensure_schema()

        assert await _user_version(fresh) == await _user_version(migrated) == _HEAD_VERSION
        assert await _index_names(fresh, table) == await _index_names(migrated, table)
    finally:
        await fresh.close()
        await migrated.close()


# ---------------------------------------------------------------------------
# PRAGMA tuning at connection init (both paths)
# ---------------------------------------------------------------------------


async def _read_pragma(db: aiosqlite.Connection, pragma: str) -> int:
    cursor = await db.execute(f"PRAGMA {pragma}")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Write a minimal YAML config file pointing at a temp database file."""
    db_file = tmp_path / "engrava.db"
    config_file = tmp_path / "engrava.yaml"
    config_file.write_text(
        f"database:\n  path: {db_file}\n  wal_mode: true\n",
        encoding="utf-8",
    )
    return config_file


async def test_from_config_sets_synchronous_normal_and_busy_timeout(
    config_path: Path,
) -> None:
    """``from_config`` tunes ``synchronous=NORMAL`` and ``busy_timeout=5000``."""
    async with await SqliteEngravaCore.from_config(config_path) as store:
        # synchronous=NORMAL reports as 1.
        assert await _read_pragma(store._db, "synchronous") == 1
        assert await _read_pragma(store._db, "busy_timeout") == 5000


async def test_service_manager_sets_synchronous_normal_and_busy_timeout(
    tmp_path: Path,
) -> None:
    """``EngravaManager`` tunes ``synchronous=NORMAL`` and ``busy_timeout=5000``."""
    from engrava.infrastructure.service_manager import EngravaManager

    async with EngravaManager(data_dir=tmp_path) as manager:
        store = await manager.get_store("svc-pragma")
        assert await _read_pragma(store._db, "synchronous") == 1
        assert await _read_pragma(store._db, "busy_timeout") == 5000


# ---------------------------------------------------------------------------
# Query-plan tests — the planner actually uses the new indexes
# ---------------------------------------------------------------------------


def _scanned_object(plan_rows: list[aiosqlite.Row]) -> set[str]:
    """Return the exact table tokens a query plan reports a SCAN over.

    The plan detail reads e.g. ``SCAN thought`` or ``SEARCH edge USING
    INDEX idx_edge_to_thought (...)``. We parse the token immediately
    after ``SCAN`` so a substring such as ``thought`` does not spuriously
    match ``thought_fts`` — the exact object name is asserted, not a
    substring (the planner reports the precise table token here).
    """
    scanned: set[str] = set()
    for row in plan_rows:
        detail = str(row["detail"])
        if detail.startswith("SCAN "):
            scanned.add(detail.split()[1])
    return scanned


def _indexes_used(plan_rows: list[aiosqlite.Row]) -> set[str]:
    """Return the set of index names the plan reports ``USING INDEX``."""
    used: set[str] = set()
    for row in plan_rows:
        detail = str(row["detail"])
        marker = "USING INDEX "
        if marker in detail:
            after = detail.split(marker, 1)[1]
            used.add(after.split()[0])
    return used


async def _explain(
    db: aiosqlite.Connection, sql: str, params: tuple[object, ...]
) -> list[aiosqlite.Row]:
    cursor = await db.execute(f"EXPLAIN QUERY PLAN {sql}", params)
    return list(await cursor.fetchall())


@pytest.fixture
async def populated_store(
    fresh_db: aiosqlite.Connection,
) -> SqliteEngravaCore:
    """A head-version store with enough rows that the planner prefers indexes."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    row_count = 200
    await fresh_db.executemany(
        """
        INSERT INTO thought
            (thought_id, thought_type, essence, content, priority, updated_cycle)
        VALUES (?, ?, 'e', 'c', 'P2', ?)
        """,
        [(f"t-{i}", "REFLECTION" if i % 2 else "OBSERVATION", i) for i in range(row_count)],
    )
    await fresh_db.executemany(
        """
        INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type)
        VALUES (?, ?, ?, 'ASSOCIATED')
        """,
        [(f"e-{i}", f"t-{i}", f"t-{(i + 1) % row_count}") for i in range(row_count)],
    )
    await fresh_db.executemany(
        """
        INSERT INTO embedding
            (embedding_id, owner_type, owner_id, model_name, dimension,
             vector_blob, created_at)
        VALUES (?, 'THOUGHT', ?, 'm', 3, ?, '2026-01-01T00:00:00+00:00')
        """,
        [(f"emb-{i}", f"t-{i}", b"\x00\x01\x02") for i in range(row_count)],
    )
    await fresh_db.commit()
    # ANALYZE so the planner has table/index statistics to choose from.
    await fresh_db.execute("ANALYZE")
    await fresh_db.commit()
    return store


async def test_get_edges_query_uses_to_thought_index(
    populated_store: SqliteEngravaCore,
) -> None:
    """``WHERE to_thought_id = ?`` searches via ``idx_edge_to_thought``, not a scan."""
    db = populated_store._db
    plan = await _explain(db, "SELECT * FROM edge WHERE to_thought_id = ?", ("t-5",))

    assert "edge" not in _scanned_object(plan)
    assert "idx_edge_to_thought" in _indexes_used(plan)


async def test_get_embedding_query_uses_owner_index(
    populated_store: SqliteEngravaCore,
) -> None:
    """``WHERE owner_id = ?`` searches via ``idx_embedding_owner``, not a scan."""
    db = populated_store._db
    plan = await _explain(
        db,
        "SELECT * FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
        ("t-5",),
    )

    assert "embedding" not in _scanned_object(plan)
    assert "idx_embedding_owner" in _indexes_used(plan)


async def test_list_thoughts_order_uses_updated_cycle_index(
    populated_store: SqliteEngravaCore,
) -> None:
    """``ORDER BY updated_cycle`` is satisfied by ``idx_thought_updated_cycle``.

    With the index present the planner can read rows in ``updated_cycle``
    order directly instead of doing a full scan plus a sort step, so the
    plan uses the index and reports no ``USE TEMP B-TREE FOR ORDER BY``.
    """
    db = populated_store._db
    plan = await _explain(
        db,
        "SELECT * FROM thought ORDER BY updated_cycle DESC LIMIT ? OFFSET ?",
        (10, 0),
    )

    assert "idx_thought_updated_cycle" in _indexes_used(plan)
    # The index ordering removes the need for an explicit sort pass.
    details = [str(row["detail"]) for row in plan]
    assert not any("USE TEMP B-TREE FOR ORDER BY" in d for d in details)


async def test_thought_type_filter_uses_type_index(
    populated_store: SqliteEngravaCore,
) -> None:
    """``WHERE thought_type = ?`` searches via ``idx_thought_type``, not a scan.

    The exact table token after SCAN/SEARCH is asserted so the FTS shadow
    table ``thought_fts`` is never mistaken for ``thought``.
    """
    db = populated_store._db
    plan = await _explain(
        db,
        "SELECT * FROM thought WHERE thought_type = ?",
        ("REFLECTION",),
    )

    assert "thought" not in _scanned_object(plan)
    assert "idx_thought_type" in _indexes_used(plan)

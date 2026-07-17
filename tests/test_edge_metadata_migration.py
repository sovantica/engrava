"""Schema migration tests for the edge ``metadata_json`` column (core-18 -> core-19).

Mirrors ``test_metadata_migration.py`` (the thought-side ``metadata_json``
migration) and ``test_valid_time_migration.py``. Exercises
``_migrate_core_v18_to_v19`` directly (idempotence, ALTER add-column, the
postcondition) and the full ``ensure_schema`` cascade so a database stamped
at any supported source version converges on the head ``user_version`` with
the ``edge.metadata_json`` column populated by the schema-level
``DEFAULT '{}'``.

The **seed-at-exactly-v18** case is the load-bearing one: it is the actual
prior shipped version, and it matches no *lower* ladder arm — only the new
``< 19`` arm rescues it. An "every arm reaches v19" test (the parametrized
cascade below) passes even while a real v18 base is stranded, so the exact-v18
seed is asserted explicitly.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_HEAD_VERSION = 19


# ---------------------------------------------------------------------------
# Helpers (mirror test_metadata_migration.py / test_valid_time_migration.py)
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


async def _full_column_info(
    db: aiosqlite.Connection,
    table: str,
) -> list[tuple[str, str, int, str | None, int]]:
    """Return the full per-column definition for ``table`` in cid order.

    Each tuple is ``(name, type, notnull, dflt_value, pk)`` — strictly stronger
    than :func:`_table_info`, so a divergence in nullability, default value or
    primary-key membership between two schemas is caught, not just a rename or a
    declared-type change.
    """
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return [
        (row["name"], row["type"], int(row["notnull"]), row["dflt_value"], int(row["pk"]))
        for row in rows
    ]


async def _foreign_keys(db: aiosqlite.Connection, table: str) -> list[tuple[object, ...]]:
    """Return the foreign-key set of ``table``, normalised to be order-independent.

    ``PRAGMA foreign_key_list`` enumerates each constraint with an ``(id, seq)``
    pair that is a mere ordering artefact; the semantic definition is
    ``(referenced_table, from_column, to_column, on_update, on_delete, match)``.
    Comparing the sorted set of those tuples asserts two schemas carry the same
    foreign keys regardless of declaration order.
    """
    cursor = await db.execute(f"PRAGMA foreign_key_list({table})")
    rows = await cursor.fetchall()
    return sorted(
        (
            row["table"],
            row["from"],
            row["to"],
            row["on_update"],
            row["on_delete"],
            row["match"],
        )
        for row in rows
    )


async def _index_names(db: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
        (table,),
    )
    rows = await cursor.fetchall()
    return {row["name"] for row in rows if row["name"] is not None}


async def _row_count(db: aiosqlite.Connection, table: str) -> int:
    cursor = await db.execute(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608
    row = await cursor.fetchone()
    assert row is not None
    return int(row["n"])


# The faithful core-18 ``thought`` DDL, shared by the full v18 bootstrap and the
# thought-only (no ``edge`` table) partial bootstrap so both stay byte-identical.
# The ``thought`` table is unchanged across v18 -> v19, so this is its full v18
# column set (a single source of truth avoids the two bootstraps drifting apart).
_CORE_V18_THOUGHT_DDL = """
    CREATE TABLE thought (
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
        valid_until       TEXT,
        action_outcome_score REAL,
        provenance        TEXT,
        pinned            INTEGER NOT NULL DEFAULT 0,
        archived_at_cycle INTEGER
    );
"""


async def _bootstrap_core_at_v18(db: aiosqlite.Connection) -> None:
    """Recreate a faithful core-18 ``thought`` + ``edge`` schema.

    A real v18 install differs from head (v19) in exactly one way: the
    ``edge`` table has **no** ``metadata_json`` column. The ``thought`` table
    is unchanged across v18 -> v19, so it carries its full v18 column set.
    The ``edge`` table carries the FK + CASCADE (present since v12), the
    valid-time columns (v13), and the pre-existing edge indexes — everything a
    v18 edge has *except* the column the v18 -> v19 upgrade re-adds.
    """
    await db.executescript(
        _CORE_V18_THOUGHT_DDL
        + """
        CREATE TABLE edge (
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
        -- Every edge index a real v18 install carries (added by the v12..v15
        -- migrations, none of which run when the upgrade starts at 18).
        CREATE INDEX idx_edge_type_from ON edge(edge_type, from_thought_id);
        CREATE INDEX idx_edge_valid_from ON edge(valid_from);
        CREATE INDEX idx_edge_valid_until ON edge(valid_until) WHERE valid_until IS NOT NULL;
        CREATE INDEX idx_edge_valid_range ON edge(valid_from, valid_until);
        CREATE INDEX idx_edge_to_thought ON edge(to_thought_id);
        CREATE INDEX idx_edge_type_to ON edge(edge_type, to_thought_id);
        PRAGMA user_version = 18;
        """,
    )
    await db.commit()


async def _bootstrap_thought_only_at_v18(db: aiosqlite.Connection) -> None:
    """Recreate a core-18 base carrying ONLY the ``thought`` table (no ``edge``).

    Models a partial bootstrap stamped at v18 with the thought table present but
    the ``edge`` table never created. The v18 -> v19 edge migration must advance
    the version WITHOUT raising — its ``_table_exists('edge')`` guard skips the
    absent table, matching the precedent of every earlier edge migration. The
    thought table is faithful to v18 so the base DDL can later be re-applied to
    create the missing ``edge`` table (the self-healing path).
    """
    await db.executescript(
        _CORE_V18_THOUGHT_DDL + "        PRAGMA user_version = 18;\n",
    )
    await db.commit()


async def _bootstrap_core_at_v11_no_fk(db: aiosqlite.Connection) -> None:
    """Recreate a faithful core-11 ``thought`` + ``edge`` schema (no FK on edge).

    v11 predates the FK-recreation migration (v11 -> v12,
    ``_recreate_edge_with_fk``). Seeding here forces the cascade to actually
    run the frozen 8-column FK recreation, then re-add valid-time (v13) and,
    finally, ``metadata_json`` (v19) — proving the frozen DDL and the new
    column co-exist through the ordered upgrade.
    """
    await db.executescript(
        """
        CREATE TABLE thought (
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
        CREATE TABLE edge (
            edge_id           TEXT PRIMARY KEY,
            from_thought_id   TEXT NOT NULL,
            to_thought_id     TEXT NOT NULL,
            edge_type         TEXT NOT NULL,
            weight            REAL NOT NULL DEFAULT 0.5,
            created_cycle     INTEGER NOT NULL DEFAULT 0,
            source            TEXT NOT NULL DEFAULT 'EXPERIENCE',
            decay_multiplier  REAL NOT NULL DEFAULT 1.0,
            UNIQUE(from_thought_id, to_thought_id, edge_type)
        );
        PRAGMA user_version = 11;
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
# Helper-level tests (direct ``_migrate_core_v18_to_v19``)
# ---------------------------------------------------------------------------


async def test_migrate_v18_to_v19_adds_column(fresh_db: aiosqlite.Connection) -> None:
    """``_migrate_core_v18_to_v19`` adds the ``metadata_json`` column to ``edge``."""
    await _bootstrap_core_at_v18(fresh_db)
    assert "metadata_json" not in await _table_columns(fresh_db, "edge")

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v18_to_v19()

    assert "metadata_json" in await _table_columns(fresh_db, "edge")


async def test_migrate_v18_to_v19_appends_column_last(fresh_db: aiosqlite.Connection) -> None:
    """The new column is appended last (ALTER can only append)."""
    await _bootstrap_core_at_v18(fresh_db)
    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v18_to_v19()

    names = [name for name, _ in await _table_info(fresh_db, "edge")]
    assert names[-1] == "metadata_json"


async def test_migrate_v18_to_v19_idempotent(fresh_db: aiosqlite.Connection) -> None:
    """Re-running the helper is safe (ALTER duplicate-column tolerated)."""
    await _bootstrap_core_at_v18(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    for _ in range(3):
        await store._migrate_core_v18_to_v19()

    names = [name for name, _ in await _table_info(fresh_db, "edge")]
    assert names.count("metadata_json") == 1


async def test_migrate_v18_to_v19_preserves_existing_rows_with_default(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Pre-migration edges receive ``metadata_json = '{}'`` from the DEFAULT."""
    await _bootstrap_core_at_v18(fresh_db)
    await fresh_db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
        "VALUES ('t-a', 'OBSERVATION', 'e', 'c', 'P2')",
    )
    await fresh_db.execute(
        "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type) "
        "VALUES ('e-old', 't-a', 't-a', 'ASSOCIATED')",
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store._migrate_core_v18_to_v19()

    cursor = await fresh_db.execute(
        "SELECT metadata_json FROM edge WHERE edge_id = 'e-old'",
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["metadata_json"] == "{}"
    assert json.loads(row["metadata_json"]) == {}


# ---------------------------------------------------------------------------
# ensure_schema cascade tests
# ---------------------------------------------------------------------------


async def test_ensure_schema_fresh_db_lands_at_head(fresh_db: aiosqlite.Connection) -> None:
    """An empty DB bootstraps straight to v19 with the edge metadata column."""
    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    assert "metadata_json" in await _table_columns(fresh_db, "edge")


async def test_ensure_schema_from_exactly_v18_reaches_v19_with_column(
    fresh_db: aiosqlite.Connection,
) -> None:
    """SHIP-BLOCKER: a DB at *exactly* v18 reaches v19 WITH the column.

    A base at exactly 18 matches no *lower* ladder arm; only the new ``< 19``
    arm rescues it. Without that arm this database would be stranded — the
    column would never be added and the first edge write would raise
    ``OperationalError: no column named metadata_json``.
    """
    await _bootstrap_core_at_v18(fresh_db)
    assert await _user_version(fresh_db) == 18
    assert "metadata_json" not in await _table_columns(fresh_db, "edge")

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    assert "metadata_json" in await _table_columns(fresh_db, "edge")


async def test_ensure_schema_from_exactly_v18_preloaded_edges_read_empty(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A pre-loaded v18 base: existing edges read back ``metadata = {}`` post-upgrade."""
    await _bootstrap_core_at_v18(fresh_db)
    await fresh_db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority, "
        "lifecycle_status) VALUES ('t-a', 'OBSERVATION', 'e', 'c', 'P2', 'ACTIVE')",
    )
    await fresh_db.execute(
        "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type, created_cycle) "
        "VALUES ('e-legacy', 't-a', 't-a', 'ASSOCIATED', 3)",
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    edges = {e.edge_id: e for e in await store.get_edges("t-a")}
    assert edges["e-legacy"].metadata == {}


async def test_ensure_schema_from_exactly_v18_is_idempotent(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Re-running ensure_schema after the v18 -> v19 upgrade stays at v19."""
    await _bootstrap_core_at_v18(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    for _ in range(3):
        await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    assert "metadata_json" in await _table_columns(fresh_db, "edge")


async def test_ensure_schema_before_fk_recreation_v11_reaches_v19(
    fresh_db: aiosqlite.Connection,
) -> None:
    """A v11 base (before the frozen FK recreation) still reaches v19 with the column.

    The cascade runs ``_recreate_edge_with_fk`` (the frozen 8-column DDL) at
    v11 -> v12, then re-adds valid-time (v13) and ``metadata_json`` (v19). The
    frozen DDL is untouched by this feature; the ordered upgrade re-adds the
    column on top.
    """
    await _bootstrap_core_at_v11_no_fk(fresh_db)
    await fresh_db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
        "VALUES ('t-a', 'OBSERVATION', 'e', 'c', 'P2')",
    )
    await fresh_db.execute(
        "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type) "
        "VALUES ('e-11', 't-a', 't-a', 'ASSOCIATED')",
    )
    await fresh_db.commit()

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    assert "metadata_json" in await _table_columns(fresh_db, "edge")
    cursor = await fresh_db.execute("SELECT metadata_json FROM edge WHERE edge_id = 'e-11'")
    row = await cursor.fetchone()
    assert row is not None
    assert row["metadata_json"] == "{}"

    # The frozen FK-recreation (v11 -> v12) must still have installed the edge
    # foreign keys under the new migration ladder, and the later v18 -> v19 add-
    # column must not have disturbed them. Assert they are present and identical
    # to a fresh-bootstrap edge — proving the frozen FK path ran correctly.
    migrated_fks = await _foreign_keys(fresh_db, "edge")
    assert migrated_fks  # non-empty: the FK recreation actually ran
    fresh = await aiosqlite.connect(":memory:")
    fresh.row_factory = aiosqlite.Row
    try:
        await SqliteEngravaCore(fresh).ensure_schema()
        assert migrated_fks == await _foreign_keys(fresh, "edge")
    finally:
        await fresh.close()


async def test_ensure_schema_interrupt_reentry_converges(
    fresh_db: aiosqlite.Connection,
) -> None:
    """An interrupted upgrade (column added, version not yet bumped) re-runs cleanly.

    Simulates a crash between the ``ALTER`` (durable in autocommit) and the
    ``PRAGMA user_version`` bump: the column exists but the version is still
    18. A re-entry hits the ``< 19`` arm, the guarded ALTER no-ops, and the
    version is bumped — converging on v19 with the column exactly once.
    """
    await _bootstrap_core_at_v18(fresh_db)
    store = SqliteEngravaCore(fresh_db)

    # Column added, but user_version deliberately left at 18 (the interrupt).
    await store._migrate_core_v18_to_v19()
    assert "metadata_json" in await _table_columns(fresh_db, "edge")
    assert await _user_version(fresh_db) == 18

    # Re-entry: the < 19 arm runs the guarded (now no-op) ALTER and bumps.
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    names = [name for name, _ in await _table_info(fresh_db, "edge")]
    assert names.count("metadata_json") == 1


async def test_ensure_schema_v18_thought_only_stamps_v19_then_self_heals(
    fresh_db: aiosqlite.Connection,
) -> None:
    """Pins the precedent-consistent edge-absent migration path + self-healing.

    A thought-only v18 partial bootstrap (no ``edge`` table) must reach v19
    WITHOUT raising: ``_migrate_core_v18_to_v19`` guards its ``ALTER`` behind
    ``_table_exists('edge')`` and returns, letting the ladder stamp the version
    — exactly as every earlier edge migration (v12->v13 ... v17->v18) guards its
    edge work with ``_table_exists`` and still advances. This early return is an
    intentional, precedent-consistent choice (NOT a hole); this test pins it so
    a future change cannot silently turn the guard into a hard failure.

    Self-healing: the ``edge`` table is only ever created from nothing by the
    base DDL (``schema_core.sql``), which at v19 already carries
    ``metadata_json``. Re-applying that base DDL creates the previously-absent
    edge table WITH the column, so no database can reach a state with an
    ``edge`` table lacking ``metadata_json``.
    """
    await _bootstrap_thought_only_at_v18(fresh_db)
    assert await _user_version(fresh_db) == 18
    assert await _table_columns(fresh_db, "edge") == set()  # no edge table at all

    store = SqliteEngravaCore(fresh_db)
    # Must not raise despite the absent edge table (precedent-consistent guard).
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    # The edge-absent guard advanced the version without creating the table.
    assert await _table_columns(fresh_db, "edge") == set()

    # Self-heal: create the edge table via the normal base-DDL path (the only
    # thing that creates an edge table from nothing). It carries the column.
    schema_sql = (
        resources.files("engrava.infrastructure.sqlite")
        .joinpath("schema_core.sql")
        .read_text(encoding="utf-8")
    )
    await fresh_db.executescript(schema_sql)

    assert "metadata_json" in await _table_columns(fresh_db, "edge")


@pytest.mark.parametrize("source_version", list(range(3, 19)))
async def test_cascade_from_any_version_to_head(
    fresh_db: aiosqlite.Connection,
    source_version: int,
) -> None:
    """A DB stamped at any historical core version cascades to head v19.

    Complements (does not replace) the exact-v18 seed above: this seeds the
    head schema then force-stamps an older ``user_version``, so it proves
    "every lower arm reaches v19" but — because the head schema already has
    the column — it cannot catch the v18 stranding on its own.
    """
    bootstrap = SqliteEngravaCore(fresh_db)
    await bootstrap.ensure_schema()
    await fresh_db.execute(f"PRAGMA user_version = {source_version}")
    await fresh_db.commit()
    assert await _user_version(fresh_db) == source_version

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _user_version(fresh_db) == _HEAD_VERSION
    assert "metadata_json" in await _table_columns(fresh_db, "edge")


# ---------------------------------------------------------------------------
# Fresh-head == migrated-head structural equivalence
# ---------------------------------------------------------------------------


async def test_fresh_equals_migrated_edge_schema() -> None:
    """A fresh-bootstrap head ``edge`` is structurally identical to a migrated one.

    Compares the **full** per-column definition (name, type, notnull,
    dflt_value, pk), the foreign-key set, and the index set between a database
    that ran the fresh-create DDL and one upgraded in place from v18. Beyond the
    column-order-parity guarantee (metadata_json appended last in both), this
    catches any divergence in nullability, default value, primary-key membership
    or the FK/CASCADE definition between the two paths.
    """
    fresh = await aiosqlite.connect(":memory:")
    fresh.row_factory = aiosqlite.Row
    migrated = await aiosqlite.connect(":memory:")
    migrated.row_factory = aiosqlite.Row
    try:
        await SqliteEngravaCore(fresh).ensure_schema()

        await _bootstrap_core_at_v18(migrated)
        await SqliteEngravaCore(migrated).ensure_schema()

        assert await _user_version(fresh) == await _user_version(migrated) == _HEAD_VERSION
        # Full per-column definition: a nullability / default / PK divergence is
        # caught, not just a rename or a declared-type change.
        assert await _full_column_info(fresh, "edge") == await _full_column_info(migrated, "edge")
        # The foreign-key set (from/to columns + ON DELETE CASCADE) is identical.
        assert await _foreign_keys(fresh, "edge") == await _foreign_keys(migrated, "edge")
        assert await _index_names(fresh, "edge") == await _index_names(migrated, "edge")
    finally:
        await fresh.close()
        await migrated.close()


async def test_migration_preserves_edge_row_count(fresh_db: aiosqlite.Connection) -> None:
    """The additive migration changes no edge row count."""
    await _bootstrap_core_at_v18(fresh_db)
    await fresh_db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority) "
        "VALUES ('t-a', 'OBSERVATION', 'e', 'c', 'P2'), ('t-b', 'OBSERVATION', 'e', 'c', 'P2')",
    )
    await fresh_db.execute(
        "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type) "
        "VALUES ('e-1', 't-a', 't-b', 'ASSOCIATED'), ('e-2', 't-b', 't-a', 'CONSOLIDATED_FROM')",
    )
    await fresh_db.commit()
    before = await _row_count(fresh_db, "edge")

    store = SqliteEngravaCore(fresh_db)
    await store.ensure_schema()

    assert await _row_count(fresh_db, "edge") == before == 2

"""Integration tests for extension schema migration loading via SqliteEngravaCore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava.domain.exceptions import ExtensionMigrationError
from engrava.domain.manifest import ExtensionManifest
from engrava.domain.protocols.hooks import DefaultEngravaHooks
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fresh_core(
    db: aiosqlite.Connection,
    manifests: list[ExtensionManifest] | None = None,
) -> SqliteEngravaCore:
    store = SqliteEngravaCore(db, manifests=manifests or [])
    await store.ensure_schema()
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# Core schema v9 tests (no extension manifests)
# ---------------------------------------------------------------------------


class TestCoreSchemaV9:
    async def test_extension_schema_versions_table_exists_after_ensure_schema(
        self, db: aiosqlite.Connection
    ) -> None:
        await _fresh_core(db)
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extension_schema_versions'"
        )
        assert await cursor.fetchone() is not None

    async def test_schema_version_is_head(self, db: aiosqlite.Connection) -> None:
        await _fresh_core(db)
        cursor = await db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert int(row[0]) == 17

    async def test_no_manifests_leaves_versions_table_empty(self, db: aiosqlite.Connection) -> None:
        await _fresh_core(db)
        cursor = await db.execute("SELECT COUNT(*) FROM extension_schema_versions")
        assert (await cursor.fetchone())[0] == 0


# ---------------------------------------------------------------------------
# Extension migration integration
# ---------------------------------------------------------------------------


class TestExtensionMigrationIntegration:
    async def test_fresh_db_applies_all_three_migrations(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        mdir = tmp_path / "m"
        mdir.mkdir()
        (mdir / "001_create_items.sql").write_text(
            "CREATE TABLE plugin_items (id TEXT PRIMARY KEY, label TEXT);",
            encoding="utf-8",
        )
        (mdir / "002_add_status.sql").write_text(
            "ALTER TABLE plugin_items ADD COLUMN status TEXT DEFAULT 'active';",
            encoding="utf-8",
        )
        (mdir / "003_create_log.sql").write_text(
            "CREATE TABLE plugin_log (entry_id TEXT PRIMARY KEY, msg TEXT);",
            encoding="utf-8",
        )

        manifest = ExtensionManifest(
            name="my-test-plugin",
            version="1.0.0",
            hooks_class=DefaultEngravaHooks,
            schema_migrations=[
                mdir / "001_create_items.sql",
                mdir / "002_add_status.sql",
                mdir / "003_create_log.sql",
            ],
        )
        await _fresh_core(db, manifests=[manifest])

        # All tables must exist.
        for table in ("plugin_items", "plugin_log"):
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            assert await cursor.fetchone() is not None, f"Table {table!r} missing"

        # extension_schema_versions must record version=3 and correct metadata.
        cursor = await db.execute(
            "SELECT version, extension_version FROM extension_schema_versions "
            "WHERE extension_name = 'my-test-plugin'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 3
        assert row[1] == "1.0.0"

    async def test_incremental_reload_applies_only_new_migration(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        mdir = tmp_path / "m"
        mdir.mkdir()
        (mdir / "001_init.sql").write_text(
            "CREATE TABLE incr_t (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (mdir / "002_extend.sql").write_text(
            "CREATE TABLE incr_t2 (id TEXT PRIMARY KEY);", encoding="utf-8"
        )

        manifest_v1 = ExtensionManifest(
            name="incr-plugin",
            version="1.0.0",
            hooks_class=DefaultEngravaHooks,
            schema_migrations=[mdir / "001_init.sql"],
        )
        await _fresh_core(db, manifests=[manifest_v1])

        # Simulate reopening the store with a newer manifest.
        store2 = SqliteEngravaCore(
            db,
            manifests=[
                ExtensionManifest(
                    name="incr-plugin",
                    version="2.0.0",
                    hooks_class=DefaultEngravaHooks,
                    schema_migrations=[
                        mdir / "001_init.sql",
                        mdir / "002_extend.sql",
                    ],
                )
            ],
        )
        await store2.ensure_schema()

        cursor = await db.execute(
            "SELECT version, extension_version FROM extension_schema_versions "
            "WHERE extension_name = 'incr-plugin'"
        )
        row = await cursor.fetchone()
        assert row[0] == 2
        assert row[1] == "2.0.0"

        # New table must exist.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='incr_t2'"
        )
        assert await cursor.fetchone() is not None

    async def test_broken_migration_raises_and_version_unchanged(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        mdir = tmp_path / "m"
        mdir.mkdir()
        (mdir / "001_bad.sql").write_text("NOT SQL AT ALL !!!;", encoding="utf-8")
        manifest = ExtensionManifest(
            name="broken-plugin",
            version="1.0.0",
            hooks_class=DefaultEngravaHooks,
            schema_migrations=[mdir / "001_bad.sql"],
        )

        store = SqliteEngravaCore(db, manifests=[manifest])
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await store.ensure_schema()

        assert exc_info.value.extension_name == "broken-plugin"
        assert exc_info.value.migration_file == "001_bad.sql"

        # Version must NOT have been written.
        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'broken-plugin'"
        )
        assert await cursor.fetchone() is None

    async def test_store_without_manifests_skips_extension_migration(
        self, db: aiosqlite.Connection
    ) -> None:
        """Empty manifests sequence must not raise and versions table stays empty."""
        store = SqliteEngravaCore(db, manifests=[])
        await store.ensure_schema()  # must not raise

        cursor = await db.execute("SELECT COUNT(*) FROM extension_schema_versions")
        assert (await cursor.fetchone())[0] == 0

    async def test_two_manifests_tracked_independently(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "001_a.sql").write_text(
            "CREATE TABLE two_a (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (dir_b / "001_b.sql").write_text(
            "CREATE TABLE two_b (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (dir_b / "002_b.sql").write_text(
            "CREATE TABLE two_b2 (id TEXT PRIMARY KEY);", encoding="utf-8"
        )

        manifests = [
            ExtensionManifest(
                name="ext-a",
                version="1.0.0",
                hooks_class=DefaultEngravaHooks,
                schema_migrations=[dir_a / "001_a.sql"],
            ),
            ExtensionManifest(
                name="ext-b",
                version="2.0.0",
                hooks_class=DefaultEngravaHooks,
                schema_migrations=[dir_b / "001_b.sql", dir_b / "002_b.sql"],
            ),
        ]
        await _fresh_core(db, manifests=manifests)

        cursor = await db.execute(
            "SELECT extension_name, version FROM extension_schema_versions ORDER BY extension_name"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "ext-a"
        assert rows[0][1] == 1
        assert rows[1][0] == "ext-b"
        assert rows[1][1] == 2


# ---------------------------------------------------------------------------
# Upgrade path — existing v8 database gets v9 migration
# ---------------------------------------------------------------------------


class TestUpgradeFromV8:
    async def test_existing_v8_db_upgraded_to_v9(self, tmp_path: Path) -> None:
        db_path = tmp_path / "upgrade.db"

        # Create a v8 database without extension_schema_versions.
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        try:
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()
            # Manually downgrade to v8 by resetting PRAGMA + dropping the table.
            await conn.execute("DROP TABLE IF EXISTS extension_schema_versions")
            await conn.execute("PRAGMA user_version = 8")
            await conn.commit()
        finally:
            await conn.close()

        # Re-open the database and run ensure_schema again (simulates upgrade).
        conn2 = await aiosqlite.connect(str(db_path))
        conn2.row_factory = aiosqlite.Row
        await conn2.execute("PRAGMA foreign_keys = ON")
        try:
            store2 = SqliteEngravaCore(conn2)
            await store2.ensure_schema()

            cursor = await conn2.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert int(row[0]) == 17

            cursor = await conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='extension_schema_versions'"
            )
            assert await cursor.fetchone() is not None
        finally:
            await conn2.close()

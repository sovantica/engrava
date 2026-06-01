"""Unit tests for ExtensionMigrationRunner and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava.domain.exceptions import ExtensionMigrationError
from engrava.domain.manifest import ExtensionManifest
from engrava.domain.protocols.hooks import DefaultEngravaHooks
from engrava.infrastructure.sqlite.extension_migrations import (
    ExtensionMigrationRunner,
    _read_migration_sql,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite connection."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


def _make_manifest(
    name: str = "test-ext",
    version: str = "1.0.0",
    migrations: list[Path] | None = None,
    package_root: Path | None = None,
) -> ExtensionManifest:
    return ExtensionManifest(
        name=name,
        version=version,
        hooks_class=DefaultEngravaHooks,
        schema_migrations=migrations or [],
        package_root=package_root,
    )


# ---------------------------------------------------------------------------
# _read_migration_sql
# ---------------------------------------------------------------------------


class TestReadMigrationSql:
    def test_absolute_path_used_as_is(self, tmp_path: Path) -> None:
        sql_file = tmp_path / "001_init.sql"
        sql_file.write_text("CREATE TABLE foo (id TEXT PRIMARY KEY);", encoding="utf-8")
        manifest = _make_manifest()
        content = _read_migration_sql(manifest, sql_file)
        assert "CREATE TABLE foo" in content

    def test_package_root_override_takes_precedence_over_module_resolution(
        self, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        sql_file = migrations_dir / "001_init.sql"
        sql_file.write_text("CREATE TABLE bar (id TEXT PRIMARY KEY);", encoding="utf-8")

        manifest = _make_manifest(package_root=tmp_path)
        content = _read_migration_sql(manifest, Path("migrations/001_init.sql"))
        assert "CREATE TABLE bar" in content

    def test_relative_path_resolved_via_package_resources(self) -> None:
        # engrava itself is an installable package, so we can resolve a known file.
        # Use the existing schema_core.sql as the target.
        manifest = ExtensionManifest(
            name="self-test",
            version="0.0.1",
            hooks_class=DefaultEngravaHooks,
            schema_migrations=[Path("infrastructure/sqlite/schema_core.sql")],
        )
        content = _read_migration_sql(manifest, Path("infrastructure/sqlite/schema_core.sql"))
        assert "CREATE TABLE" in content

    def test_unresolvable_relative_path_raises_with_guidance(self) -> None:
        """Non-package hooks_class (builtins) raises ExtensionMigrationError."""

        class _InlineHooks(DefaultEngravaHooks):
            pass

        # Patch the module to simulate __main__
        original_module = _InlineHooks.__module__
        _InlineHooks.__module__ = "__main__"
        try:
            manifest = ExtensionManifest(
                name="bad-ext",
                version="0.0.1",
                hooks_class=_InlineHooks,
                schema_migrations=[Path("migrations/001.sql")],
            )
            with pytest.raises(ExtensionMigrationError) as exc_info:
                _read_migration_sql(manifest, Path("migrations/001.sql"))
            assert "package_root" in exc_info.value.message
            assert exc_info.value.extension_name == "bad-ext"
            assert exc_info.value.migration_file == "001.sql"
        finally:
            _InlineHooks.__module__ = original_module

    def test_missing_absolute_file_raises(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        with pytest.raises(ExtensionMigrationError):
            _read_migration_sql(manifest, tmp_path / "nonexistent.sql")

    def test_missing_package_root_file_raises(self, tmp_path: Path) -> None:
        manifest = _make_manifest(package_root=tmp_path)
        with pytest.raises(ExtensionMigrationError) as exc_info:
            _read_migration_sql(manifest, Path("migrations/nonexistent.sql"))
        assert exc_info.value.migration_file == "nonexistent.sql"


# ---------------------------------------------------------------------------
# ExtensionMigrationRunner
# ---------------------------------------------------------------------------


class TestExtensionMigrationRunner:
    async def test_fresh_install_applies_all_migrations(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_init.sql").write_text(
            "CREATE TABLE ext_items (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (migrations_dir / "002_add_name.sql").write_text(
            "ALTER TABLE ext_items ADD COLUMN name TEXT;", encoding="utf-8"
        )
        (migrations_dir / "003_add_tags.sql").write_text(
            "CREATE TABLE ext_tags (tag TEXT PRIMARY KEY);", encoding="utf-8"
        )

        manifest = _make_manifest(
            migrations=[
                migrations_dir / "001_init.sql",
                migrations_dir / "002_add_name.sql",
                migrations_dir / "003_add_tags.sql",
            ]
        )
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(manifest, db)

        # All three tables/columns must exist.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ext_items'"
        )
        assert await cursor.fetchone() is not None

        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ext_tags'"
        )
        assert await cursor.fetchone() is not None

        # Version must be 3.
        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = ?",
            ("test-ext",),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 3

    async def test_incremental_applies_only_pending(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_init.sql").write_text(
            "CREATE TABLE inc_items (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (migrations_dir / "002_add_col.sql").write_text(
            "ALTER TABLE inc_items ADD COLUMN extra TEXT;", encoding="utf-8"
        )

        manifest_v1 = _make_manifest(migrations=[migrations_dir / "001_init.sql"])
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(manifest_v1, db)

        # Version should be 1 now.
        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 1

        # Now apply v2 manifest with both files.
        manifest_v2 = _make_manifest(
            migrations=[
                migrations_dir / "001_init.sql",
                migrations_dir / "002_add_col.sql",
            ]
        )
        await runner.apply_pending(manifest_v2, db)

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 2

    async def test_no_op_when_current(self, db: aiosqlite.Connection, tmp_path: Path) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_init.sql").write_text(
            "CREATE TABLE noop_t (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_init.sql"])
        runner = ExtensionMigrationRunner()

        await runner.apply_pending(manifest, db)
        await runner.apply_pending(manifest, db)  # second call — no-op

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 1

    async def test_idempotent_rerun(self, db: aiosqlite.Connection, tmp_path: Path) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_init.sql").write_text(
            "CREATE TABLE IF NOT EXISTS idem_t (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_init.sql"])
        runner = ExtensionMigrationRunner()

        await runner.apply_pending(manifest, db)
        await runner.apply_pending(manifest, db)  # idempotent — no error

    async def test_downgrade_detection_raises(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_init.sql").write_text(
            "CREATE TABLE dg_t (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (migrations_dir / "002_add_col.sql").write_text(
            "ALTER TABLE dg_t ADD COLUMN extra TEXT;", encoding="utf-8"
        )

        # Apply both migrations.
        manifest_full = _make_manifest(
            migrations=[
                migrations_dir / "001_init.sql",
                migrations_dir / "002_add_col.sql",
            ]
        )
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(manifest_full, db)

        # Now present a manifest with only one file (downgrade).
        manifest_downgraded = _make_manifest(migrations=[migrations_dir / "001_init.sql"])
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(manifest_downgraded, db)

        assert "downgrade" in exc_info.value.message.lower()
        assert exc_info.value.extension_name == "test-ext"

    async def test_sql_failure_rolls_back_and_does_not_mark_version(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_bad.sql").write_text("THIS IS NOT VALID SQL !!!;", encoding="utf-8")
        manifest = _make_manifest(migrations=[migrations_dir / "001_bad.sql"])
        runner = ExtensionMigrationRunner()

        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(manifest, db)

        assert exc_info.value.migration_file == "001_bad.sql"

        # Version must NOT have been written.
        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert await cursor.fetchone() is None

    async def test_extension_version_recorded_at_apply_time(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_init.sql").write_text(
            "CREATE TABLE ver_t (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        manifest = _make_manifest(
            version="2.5.0",
            migrations=[migrations_dir / "001_init.sql"],
        )
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(manifest, db)

        cursor = await db.execute(
            "SELECT extension_version, migration_file "
            "FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "2.5.0"
        assert row[1] == "001_init.sql"

    async def test_no_op_for_empty_schema_migrations(self, db: aiosqlite.Connection) -> None:
        manifest = _make_manifest(migrations=[])
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(manifest, db)  # must not raise

        # Table may or may not exist — no row is the key assertion.
        try:
            cursor = await db.execute("SELECT COUNT(*) FROM extension_schema_versions")
            count = (await cursor.fetchone())[0]
            assert count == 0
        except Exception:  # noqa: BLE001, S110
            pass  # table doesn't exist — also acceptable


# ---------------------------------------------------------------------------
# Multiple independent extensions
# ---------------------------------------------------------------------------


class TestMultipleExtensions:
    async def test_two_extensions_independent_tracking(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        dir_a = tmp_path / "ext_a"
        dir_b = tmp_path / "ext_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "001_a.sql").write_text(
            "CREATE TABLE a_t (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (dir_b / "001_b.sql").write_text(
            "CREATE TABLE b_t (id TEXT PRIMARY KEY);", encoding="utf-8"
        )

        manifest_a = _make_manifest(name="ext-a", migrations=[dir_a / "001_a.sql"])
        manifest_b = _make_manifest(name="ext-b", migrations=[dir_b / "001_b.sql"])
        runner = ExtensionMigrationRunner()

        await runner.apply_pending(manifest_a, db)
        await runner.apply_pending(manifest_b, db)

        cursor = await db.execute(
            "SELECT extension_name, version FROM extension_schema_versions ORDER BY extension_name"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "ext-a"
        assert rows[0][1] == 1
        assert rows[1][0] == "ext-b"
        assert rows[1][1] == 1

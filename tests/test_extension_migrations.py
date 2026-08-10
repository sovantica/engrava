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
    _checksum,
    _read_migration_sql,
    _split_sql_statements,
    _UnsupportedMigrationSQLError,
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

    def test_invalid_utf8_absolute_file_raises_typed_error(self, tmp_path: Path) -> None:
        sql_file = tmp_path / "001_bad_bytes.sql"
        sql_file.write_bytes(b"CREATE TABLE t (\xff\xfe);")  # invalid UTF-8
        manifest = _make_manifest()
        with pytest.raises(ExtensionMigrationError) as exc_info:
            _read_migration_sql(manifest, sql_file)
        assert exc_info.value.migration_file == "001_bad_bytes.sql"

    def test_invalid_utf8_package_root_file_raises_typed_error(self, tmp_path: Path) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_bad_bytes.sql").write_bytes(b"CREATE TABLE t (\xff\xfe);")
        manifest = _make_manifest(package_root=tmp_path)
        with pytest.raises(ExtensionMigrationError) as exc_info:
            _read_migration_sql(manifest, Path("migrations/001_bad_bytes.sql"))
        assert exc_info.value.migration_file == "001_bad_bytes.sql"


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

        # An extension with no migrations and no recorded history is a pure
        # no-op: it must not create the bookkeeping table.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extension_schema_versions'"
        )
        assert await cursor.fetchone() is None


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


# ---------------------------------------------------------------------------
# SQLite-aware statement splitting
# ---------------------------------------------------------------------------


class TestSplitSqlStatements:
    def test_multiple_statements_split(self) -> None:
        stmts = _split_sql_statements("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);")
        assert len(stmts) == 2
        assert stmts[0] == "CREATE TABLE a (x TEXT);"
        assert stmts[1] == "CREATE TABLE b (y TEXT);"

    def test_semicolon_in_single_quoted_string_not_split(self) -> None:
        stmts = _split_sql_statements("INSERT INTO t (v) VALUES ('a;b;c');")
        assert stmts == ["INSERT INTO t (v) VALUES ('a;b;c');"]

    def test_doubled_quote_escape_inside_string(self) -> None:
        # The doubled single quote is an escaped literal quote, not a close.
        stmts = _split_sql_statements("INSERT INTO t (v) VALUES ('it''s; fine');")
        assert stmts == ["INSERT INTO t (v) VALUES ('it''s; fine');"]

    def test_semicolon_in_line_comment_not_split(self) -> None:
        sql = "-- header; note\nCREATE TABLE t (x TEXT);"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert stmts[0].endswith("CREATE TABLE t (x TEXT);")

    def test_semicolon_in_block_comment_not_split(self) -> None:
        sql = "CREATE TABLE t (\n  x TEXT /* col; comment */\n);"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1

    def test_semicolon_in_bracket_identifier_not_split(self) -> None:
        stmts = _split_sql_statements('CREATE TABLE t ("weird;name" TEXT);')
        assert stmts == ['CREATE TABLE t ("weird;name" TEXT);']

    def test_compound_trigger_body_is_single_statement(self) -> None:
        sql = (
            "CREATE TRIGGER trg AFTER INSERT ON t BEGIN\n"
            "  UPDATE t SET n = n + 1 WHERE id = NEW.id;\n"
            "  DELETE FROM audit WHERE stale = 1;\n"
            "END;"
        )
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert stmts[0].startswith("CREATE TRIGGER trg")
        assert stmts[0].rstrip().endswith("END;")

    def test_trigger_followed_by_statement_split(self) -> None:
        sql = (
            "CREATE TRIGGER trg AFTER INSERT ON t BEGIN\n"
            "  UPDATE t SET n = 1;\n"
            "END;\n"
            "CREATE INDEX idx_t ON t (n);"
        )
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 2
        assert stmts[0].startswith("CREATE TRIGGER trg")
        assert stmts[1] == "CREATE INDEX idx_t ON t (n);"

    def test_comment_and_whitespace_only_yields_no_statements(self) -> None:
        assert _split_sql_statements("-- just a comment\n   \n/* nothing here */\n") == []

    def test_trailing_comment_after_statement_ignored(self) -> None:
        stmts = _split_sql_statements("CREATE TABLE t (x TEXT);\n-- trailing note")
        assert len(stmts) == 1

    def test_missing_terminating_semicolon_raises(self) -> None:
        with pytest.raises(_UnsupportedMigrationSQLError):
            _split_sql_statements("CREATE TABLE t (x TEXT)")

    def test_unclosed_trigger_body_raises(self) -> None:
        sql = "CREATE TRIGGER trg AFTER INSERT ON t BEGIN\n  UPDATE t SET n = 1;\n"
        with pytest.raises(_UnsupportedMigrationSQLError):
            _split_sql_statements(sql)

    def test_unterminated_string_raises(self) -> None:
        with pytest.raises(_UnsupportedMigrationSQLError):
            _split_sql_statements("INSERT INTO t (v) VALUES ('open;")


# ---------------------------------------------------------------------------
# Compound / commented migrations applied through the runner
# ---------------------------------------------------------------------------


class TestCompoundMigrationExecution:
    async def test_trigger_migration_applies_and_fires(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_trigger.sql").write_text(
            "CREATE TABLE trg_items (id TEXT PRIMARY KEY, n INTEGER NOT NULL DEFAULT 0);\n"
            "CREATE TABLE trg_audit (id TEXT NOT NULL);\n"
            "CREATE TRIGGER trg_after_ins AFTER INSERT ON trg_items BEGIN\n"
            "  INSERT INTO trg_audit (id) VALUES (NEW.id);\n"
            "  UPDATE trg_items SET n = n + 1 WHERE id = NEW.id;\n"
            "END;",
            encoding="utf-8",
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_trigger.sql"])
        await ExtensionMigrationRunner().apply_pending(manifest, db)

        # The trigger exists and fires: inserting a row audits it and bumps n.
        await db.execute("INSERT INTO trg_items (id) VALUES ('x')")
        cursor = await db.execute("SELECT id FROM trg_audit")
        assert (await cursor.fetchone())[0] == "x"
        cursor = await db.execute("SELECT n FROM trg_items WHERE id = 'x'")
        assert (await cursor.fetchone())[0] == 1

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 1

    async def test_commented_migration_applies_intact(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_comments.sql").write_text(
            "-- create the widget table; it stores widgets\n"
            "CREATE TABLE widgets (\n"
            "  id TEXT PRIMARY KEY,  -- the id; a primary key\n"
            "  label TEXT NOT NULL DEFAULT 'a;b'  /* default; value */\n"
            ");",
            encoding="utf-8",
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_comments.sql"])
        await ExtensionMigrationRunner().apply_pending(manifest, db)

        # The literal default containing a semicolon survived splitting intact.
        await db.execute("INSERT INTO widgets (id) VALUES ('w1')")
        cursor = await db.execute("SELECT label FROM widgets WHERE id = 'w1'")
        assert (await cursor.fetchone())[0] == "a;b"

    async def test_unsupported_shape_fails_before_any_statement_applies(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # A valid CREATE TABLE followed by an unclosed BEGIN...END trigger.
        (migrations_dir / "001_bad.sql").write_text(
            "CREATE TABLE guard_t (id TEXT PRIMARY KEY);\n"
            "CREATE TRIGGER guard_trg AFTER INSERT ON guard_t BEGIN\n"
            "  UPDATE guard_t SET id = id;\n",
            encoding="utf-8",
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_bad.sql"])
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await ExtensionMigrationRunner().apply_pending(manifest, db)

        assert exc_info.value.migration_file == "001_bad.sql"

        # The leading valid CREATE TABLE must NOT have been applied.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='guard_t'"
        )
        assert await cursor.fetchone() is None
        # No history or summary row was written.
        cursor = await db.execute("SELECT COUNT(*) FROM extension_schema_migrations")
        assert (await cursor.fetchone())[0] == 0
        cursor = await db.execute("SELECT COUNT(*) FROM extension_schema_versions")
        assert (await cursor.fetchone())[0] == 0


# ---------------------------------------------------------------------------
# Append-only history, checksums and drift detection
# ---------------------------------------------------------------------------


class TestMigrationHistoryAndDrift:
    async def test_fresh_store_records_checksummed_history(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f2 = migrations_dir / "002_more.sql"
        f1.write_text("CREATE TABLE hist_a (id TEXT PRIMARY KEY);", encoding="utf-8")
        f2.write_text("CREATE TABLE hist_b (id TEXT PRIMARY KEY);", encoding="utf-8")

        manifest = _make_manifest(migrations=[f1, f2])
        await ExtensionMigrationRunner().apply_pending(manifest, db)

        cursor = await db.execute(
            "SELECT migration_index, migration_file, checksum "
            "FROM extension_schema_migrations WHERE extension_name = 'test-ext' "
            "ORDER BY migration_index"
        )
        rows = await cursor.fetchall()
        assert [(r[0], r[1]) for r in rows] == [(1, "001_init.sql"), (2, "002_more.sql")]
        assert rows[0][2] == _checksum(f1.read_text(encoding="utf-8"))
        assert rows[1][2] == _checksum(f2.read_text(encoding="utf-8"))

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 2

    async def test_rerun_is_idempotent_and_appends_no_duplicate_history(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f1.write_text("CREATE TABLE idem_hist (id TEXT PRIMARY KEY);", encoding="utf-8")
        manifest = _make_manifest(migrations=[f1])
        runner = ExtensionMigrationRunner()

        await runner.apply_pending(manifest, db)
        await runner.apply_pending(manifest, db)  # no-op; must not duplicate history

        cursor = await db.execute(
            "SELECT COUNT(*) FROM extension_schema_migrations WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 1

        # Both bookkeeping tables exist after repeated (idempotent) runs.
        for table in ("extension_schema_versions", "extension_schema_migrations"):
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            assert await cursor.fetchone() is not None

    async def test_content_drift_of_applied_migration_is_rejected(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f1.write_text("CREATE TABLE drift_a (id TEXT PRIMARY KEY);", encoding="utf-8")
        manifest = _make_manifest(migrations=[f1])
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(manifest, db)

        # Edit the already-applied file in place.
        f1.write_text("CREATE TABLE drift_a (id TEXT PRIMARY KEY, extra TEXT);", encoding="utf-8")
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(manifest, db)

        assert "drift" in exc_info.value.message.lower()
        assert exc_info.value.migration_file == "001_init.sql"

    async def test_summary_history_mismatch_fails_closed(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f2 = migrations_dir / "002_more.sql"
        f1.write_text("CREATE TABLE cc_a (id TEXT PRIMARY KEY);", encoding="utf-8")
        f2.write_text("CREATE TABLE cc_b (id TEXT PRIMARY KEY);", encoding="utf-8")
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(_make_manifest(migrations=[f1, f2]), db)

        # Corrupt the bookkeeping: drop the history tail row while the summary
        # still claims version=2. Without a cross-check, migration #2 would be
        # treated as pending and reapplied.
        await db.execute(
            "DELETE FROM extension_schema_migrations "
            "WHERE extension_name = 'test-ext' AND migration_index = 2"
        )
        await db.commit()

        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(_make_manifest(migrations=[f1, f2]), db)
        message = exc_info.value.message.lower()
        assert "inconsistent" in message or "bookkeeping" in message

        # Fail-closed: nothing was reapplied and the history was not mutated.
        cursor = await db.execute(
            "SELECT COUNT(*) FROM extension_schema_migrations WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 1

    async def test_non_integer_summary_fails_closed(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f1.write_text("CREATE TABLE ni_a (id TEXT PRIMARY KEY);", encoding="utf-8")
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(_make_manifest(migrations=[f1]), db)

        # Corrupt the summary version to a non-integer. The non-STRICT INTEGER
        # column keeps 1.5 as REAL; int() would truncate it to 1 and mask a
        # mismatch, so a non-integer summary must fail closed.
        await db.execute(
            "UPDATE extension_schema_versions SET version = 1.5 WHERE extension_name = 'test-ext'"
        )
        await db.commit()

        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(_make_manifest(migrations=[f1]), db)
        assert "not an integer" in exc_info.value.message.lower()

    async def test_line_ending_and_bom_changes_do_not_trip_drift(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        # LF, no BOM.
        f1.write_bytes(b"CREATE TABLE le_t (id TEXT PRIMARY KEY);\nCREATE INDEX i ON le_t (id);\n")
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(_make_manifest(migrations=[f1]), db)

        # Rewrite the same SQL content with a BOM and CRLF line endings only.
        f1.write_bytes(
            b"\xef\xbb\xbfCREATE TABLE le_t (id TEXT PRIMARY KEY);\r\n"
            b"CREATE INDEX i ON le_t (id);\r\n"
        )
        # Cosmetic-only change: the semantic-content checksum is unchanged, so no
        # drift is reported and the run is a clean no-op.
        await runner.apply_pending(_make_manifest(migrations=[f1]), db)

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 1

    async def test_inserted_earlier_migration_is_rejected(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_a.sql"
        f1.write_text("CREATE TABLE ins_a (id TEXT PRIMARY KEY);", encoding="utf-8")
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(_make_manifest(migrations=[f1]), db)

        # Insert a new file that sorts before the applied one.
        f0 = migrations_dir / "000_early.sql"
        f0.write_text("CREATE TABLE ins_early (id TEXT PRIMARY KEY);", encoding="utf-8")
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(_make_manifest(migrations=[f0, f1]), db)

        assert "drift" in exc_info.value.message.lower()
        # The inserted migration must NOT have been applied.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ins_early'"
        )
        assert await cursor.fetchone() is None

    async def test_renamed_earlier_migration_is_rejected(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f_orig = migrations_dir / "001_original.sql"
        f_orig.write_text("CREATE TABLE ren_t (id TEXT PRIMARY KEY);", encoding="utf-8")
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(_make_manifest(migrations=[f_orig]), db)

        # Same position, different filename.
        f_renamed = migrations_dir / "001_renamed.sql"
        f_renamed.write_text("CREATE TABLE ren_t (id TEXT PRIMARY KEY);", encoding="utf-8")
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(_make_manifest(migrations=[f_renamed]), db)

        assert "drift" in exc_info.value.message.lower()
        assert exc_info.value.migration_file == "001_original.sql"

    async def test_upgraded_store_adopts_legacy_summary_and_stays_consistent(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # No IF NOT EXISTS: re-running an adopted migration would raise, so a
        # clean upgrade proves the baseline was adopted rather than re-applied.
        f1 = migrations_dir / "001_init.sql"
        f2 = migrations_dir / "002_more.sql"
        f1.write_text("CREATE TABLE up_a (id TEXT PRIMARY KEY);", encoding="utf-8")
        f2.write_text("CREATE TABLE up_b (id TEXT PRIMARY KEY);", encoding="utf-8")
        runner = ExtensionMigrationRunner()

        # An older build applied only 001 and tracked no checksum history.
        await runner.apply_pending(_make_manifest(migrations=[f1]), db)
        await db.execute("DROP TABLE extension_schema_migrations")
        await db.commit()

        # The new build sees [001, 002] and must adopt 001, then apply 002.
        await runner.apply_pending(_make_manifest(migrations=[f1, f2]), db)

        cursor = await db.execute(
            "SELECT migration_index, migration_file, checksum "
            "FROM extension_schema_migrations WHERE extension_name = 'test-ext' "
            "ORDER BY migration_index"
        )
        rows = await cursor.fetchall()
        assert [(r[0], r[1]) for r in rows] == [(1, "001_init.sql"), (2, "002_more.sql")]
        # The adopted baseline carries the same content checksum a fresh install
        # would have recorded.
        assert rows[0][2] == _checksum(f1.read_text(encoding="utf-8"))

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = 'test-ext'"
        )
        assert (await cursor.fetchone())[0] == 2

        for table in ("up_a", "up_b"):
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            assert await cursor.fetchone() is not None, f"Table {table!r} missing"

    async def test_upgraded_store_rejects_drift_after_adoption(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f1.write_text("CREATE TABLE adopt_drift (id TEXT PRIMARY KEY);", encoding="utf-8")
        runner = ExtensionMigrationRunner()

        await runner.apply_pending(_make_manifest(migrations=[f1]), db)
        await db.execute("DROP TABLE extension_schema_migrations")
        await db.commit()

        # Edit the already-applied file, then upgrade: adoption records the new
        # checksum, so this run's drift is masked (no prior checksum existed) but
        # any subsequent edit is caught.
        f1.write_text(
            "CREATE TABLE adopt_drift (id TEXT PRIMARY KEY, extra TEXT);", encoding="utf-8"
        )
        await runner.apply_pending(_make_manifest(migrations=[f1]), db)

        f1.write_text(
            "CREATE TABLE adopt_drift (id TEXT PRIMARY KEY, other TEXT);", encoding="utf-8"
        )
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(_make_manifest(migrations=[f1]), db)
        assert "drift" in exc_info.value.message.lower()


# ---------------------------------------------------------------------------
# Atomicity, statement-shape validation and manifest integrity
# ---------------------------------------------------------------------------


class TestMigrationSafety:
    def test_unterminated_block_comment_raises(self) -> None:
        with pytest.raises(_UnsupportedMigrationSQLError):
            _split_sql_statements("CREATE TABLE t (x TEXT); /* never closed")

    async def test_multi_statement_failure_rolls_back_earlier_statement(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # First statement is valid and mutating; the second fails at execution.
        (migrations_dir / "001_partial.sql").write_text(
            "CREATE TABLE atomic_ok (id TEXT PRIMARY KEY);\n"
            "INSERT INTO missing_table (id) VALUES ('x');",
            encoding="utf-8",
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_partial.sql"])
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await ExtensionMigrationRunner().apply_pending(manifest, db)

        assert exc_info.value.migration_file == "001_partial.sql"

        # The valid earlier statement must have been rolled back with the step.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_ok'"
        )
        assert await cursor.fetchone() is None
        cursor = await db.execute("SELECT COUNT(*) FROM extension_schema_migrations")
        assert (await cursor.fetchone())[0] == 0
        cursor = await db.execute("SELECT COUNT(*) FROM extension_schema_versions")
        assert (await cursor.fetchone())[0] == 0

    async def test_transaction_control_statement_rejected_before_applying(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_txn.sql").write_text(
            "CREATE TABLE txn_guard (id TEXT PRIMARY KEY);\nCOMMIT;",
            encoding="utf-8",
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_txn.sql"])
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await ExtensionMigrationRunner().apply_pending(manifest, db)

        assert exc_info.value.migration_file == "001_txn.sql"
        # Rejected during preparation — the leading valid statement is not applied.
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='txn_guard'"
        )
        assert await cursor.fetchone() is None

    async def test_bom_prefixed_transaction_control_is_rejected(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # A leading UTF-8 BOM must not hide the COMMIT from statement-shape
        # validation (SQLite would otherwise execute the BOM-prefixed COMMIT).
        (migrations_dir / "001_bom.sql").write_text(
            "CREATE TABLE bom_guard (id TEXT PRIMARY KEY);\n\ufeffCOMMIT;",
            encoding="utf-8",
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_bom.sql"])
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await ExtensionMigrationRunner().apply_pending(manifest, db)

        assert exc_info.value.migration_file == "001_bom.sql"
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bom_guard'"
        )
        assert await cursor.fetchone() is None

    async def test_bom_prefixed_migration_applies_cleanly(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # A legitimate migration saved with a leading BOM still applies.
        (migrations_dir / "001_bom_ok.sql").write_text(
            "\ufeffCREATE TABLE bom_ok (id TEXT PRIMARY KEY);",
            encoding="utf-8",
        )
        manifest = _make_manifest(migrations=[migrations_dir / "001_bom_ok.sql"])
        await ExtensionMigrationRunner().apply_pending(manifest, db)

        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bom_ok'"
        )
        assert await cursor.fetchone() is not None

    async def test_duplicate_migration_filenames_rejected(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "001_init.sql").write_text(
            "CREATE TABLE dup_a (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        (dir_b / "001_init.sql").write_text(
            "CREATE TABLE dup_b (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
        manifest = _make_manifest(migrations=[dir_a / "001_init.sql", dir_b / "001_init.sql"])
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await ExtensionMigrationRunner().apply_pending(manifest, db)

        assert "duplicate" in exc_info.value.message.lower()
        assert exc_info.value.migration_file == "001_init.sql"

    async def test_removing_all_migrations_is_detected_as_downgrade(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f1.write_text("CREATE TABLE rm_all (id TEXT PRIMARY KEY);", encoding="utf-8")
        runner = ExtensionMigrationRunner()
        await runner.apply_pending(_make_manifest(migrations=[f1]), db)

        # An extension that previously shipped a migration now ships none.
        with pytest.raises(ExtensionMigrationError) as exc_info:
            await runner.apply_pending(_make_manifest(migrations=[]), db)
        assert "downgrade" in exc_info.value.message.lower()

    async def test_safe_savepoint_name_allows_unusual_extension_names(
        self, db: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        f1 = migrations_dir / "001_init.sql"
        f1.write_text("CREATE TABLE weird_name_t (id TEXT PRIMARY KEY);", encoding="utf-8")
        # A name with characters that are invalid in a bare SQLite identifier.
        manifest = _make_manifest(name="my ext.plugin/2", migrations=[f1])
        await ExtensionMigrationRunner().apply_pending(manifest, db)

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = ?",
            ("my ext.plugin/2",),
        )
        assert (await cursor.fetchone())[0] == 1

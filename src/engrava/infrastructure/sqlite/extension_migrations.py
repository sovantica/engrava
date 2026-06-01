"""Extension schema migration runner.

Applies SQL migration scripts declared in ``ExtensionManifest.schema_migrations``,
tracking per-extension schema versions in the ``extension_schema_versions`` table.
"""

from __future__ import annotations

import importlib.resources
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import aiosqlite

    from engrava.domain.manifest import ExtensionManifest

from engrava.domain.exceptions import ExtensionMigrationError

# Matches semicolons that are NOT inside single-quoted or double-quoted strings.
# This covers standard SQL DDL/DML; does not handle $$ dollar-quoting (PostgreSQL).
_STATEMENT_SPLIT_RE = re.compile(
    r"""(?x)
    (?:                         # skip over string literals
        '(?:[^'\\]|\\.)*'       # single-quoted literal
      | "(?:[^"\\]|\\.)*"       # double-quoted identifier
    )
    | (;)                       # capture bare semicolons (group 1)
    """
)


def _split_sql_statements(sql: str) -> list[str]:
    """Split *sql* into individual statements on bare semicolons.

    Semicolons inside string literals (single- or double-quoted) are not
    treated as statement terminators.  Empty strings produced by trailing
    semicolons or blank lines are discarded.

    Args:
        sql: SQL text that may contain multiple ``';'``-separated statements.

    Returns:
        Non-empty list of individual SQL statement strings (without trailing
        semicolons).

    """
    parts: list[str] = []
    pos = 0
    for m in _STATEMENT_SPLIT_RE.finditer(sql):
        if m.group(1) is not None:  # bare semicolon
            stmt = sql[pos : m.start()].strip()
            if stmt:
                parts.append(stmt)
            pos = m.end()
    # tail after last semicolon (or entire string if no semicolons)
    tail = sql[pos:].strip()
    if tail:
        parts.append(tail)
    return parts


_SQL_ENSURE_VERSIONS_TABLE = """\
CREATE TABLE IF NOT EXISTS extension_schema_versions (
    extension_name    TEXT PRIMARY KEY,
    version           INTEGER NOT NULL DEFAULT 0,
    applied_at        REAL NOT NULL,
    migration_file    TEXT,
    extension_version TEXT
)"""

_SQL_UPSERT_VERSION = """\
INSERT INTO extension_schema_versions
    (extension_name, version, applied_at, migration_file, extension_version)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(extension_name) DO UPDATE SET
    version           = excluded.version,
    applied_at        = excluded.applied_at,
    migration_file    = excluded.migration_file,
    extension_version = excluded.extension_version"""


def _read_migration_sql(manifest: ExtensionManifest, path: Path) -> str:
    """Resolve *path* and return its SQL content.

    Resolution order:

    1. Absolute path → used as-is (developer / CI escape hatch).
    2. ``manifest.package_root`` is set → joined with ``package_root``.
    3. Default → resolved via ``importlib.resources.files`` for the
       top-level package that contains ``manifest.hooks_class``.

    Args:
        manifest: Extension manifest that declares the migration.
        path: Path to the migration SQL file (may be relative).

    Returns:
        SQL text content of the migration file.

    Raises:
        ExtensionMigrationError: If the path cannot be resolved (hooks class
            lives in a non-installable module) or the file cannot be read.

    """
    if path.is_absolute():
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Cannot read migration file {str(path)!r}: {exc}"
            raise ExtensionMigrationError(manifest.name, msg, migration_file=path.name) from exc

    if manifest.package_root is not None:
        resolved = manifest.package_root / path
        try:
            return resolved.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Cannot read migration file {str(resolved)!r}: {exc}"
            raise ExtensionMigrationError(manifest.name, msg, migration_file=path.name) from exc

    top_level = manifest.hooks_class.__module__.partition(".")[0]
    if top_level in ("__main__", "", "builtins"):
        msg = (
            f"Cannot resolve relative migration path {str(path)!r}: "
            f"manifest for {manifest.name!r} has hooks_class in module "
            f"{manifest.hooks_class.__module__!r}, which is not an installable "
            f"package.  Provide an absolute Path or set manifest.package_root "
            f"to the directory that contains your migration files."
        )
        raise ExtensionMigrationError(manifest.name, msg, migration_file=path.name)

    try:
        traversable = importlib.resources.files(top_level).joinpath(str(path))
        return traversable.read_text(encoding="utf-8")
    except Exception as exc:
        msg = f"Cannot read migration file {str(path)!r} from package {top_level!r}: {exc}"
        raise ExtensionMigrationError(manifest.name, msg, migration_file=path.name) from exc


class ExtensionMigrationRunner:
    """Applies pending SQL migration files declared in an ``ExtensionManifest``.

    Migrations are executed in filename-sort order (convention:
    ``NNN_slug.sql``, e.g. ``001_initial.sql``, ``002_add_tags.sql``).
    The per-extension current schema version is stored in the
    ``extension_schema_versions`` table.  After each successful migration
    the version counter is advanced; failures leave the counter unchanged.

    The ``extension_schema_versions`` table is created on first use if it
    does not already exist (idempotent DDL).

    Typical usage::

        runner = ExtensionMigrationRunner()
        await runner.apply_pending(manifest, db)

    """

    async def apply_pending(
        self,
        manifest: ExtensionManifest,
        db: aiosqlite.Connection,
    ) -> None:
        """Apply any pending migrations for *manifest* against *db*.

        Migrations are identified by sorting ``manifest.schema_migrations``
        by filename.  The runner compares the count of already-applied
        migrations (stored version) with the total declared and executes
        only the new ones.

        Args:
            manifest: Extension manifest with ``schema_migrations`` paths.
            db: An open aiosqlite connection (core schema already applied).

        Raises:
            ExtensionMigrationError: On SQL execution failure (the savepoint
                is rolled back so the database is left unchanged), downgrade
                detection, or unresolvable relative path.

        """
        if not manifest.schema_migrations:
            return

        # Ensure tracking table exists — idempotent.
        await db.executescript(_SQL_ENSURE_VERSIONS_TABLE)

        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = ?",
            (manifest.name,),
        )
        row = await cursor.fetchone()
        current_version: int = int(row[0]) if row else 0

        sorted_files = sorted(manifest.schema_migrations, key=lambda p: p.name)
        total = len(sorted_files)

        if current_version > total:
            msg = (
                f"Extension downgrade detected: database records version "
                f"{current_version} but manifest declares only {total} "
                f"migration file(s).  Revert to an extension release that "
                f"provides at least {current_version} migration file(s), or "
                f"manually clear the extension_schema_versions row for "
                f"{manifest.name!r}."
            )
            raise ExtensionMigrationError(manifest.name, msg)

        pending = sorted_files[current_version:]
        for new_version, migration_path in enumerate(pending, start=current_version + 1):
            basename = migration_path.name
            sql = _read_migration_sql(manifest, migration_path)
            statements = _split_sql_statements(sql)
            savepoint = f"ext_mig_{manifest.name.replace('-', '_')}_{new_version}"

            # Use a savepoint so that both the migration SQL *and* the version
            # write are applied atomically.  SQLite DDL is fully transactional,
            # so a ROLLBACK undoes CREATE TABLE / ALTER TABLE statements too.
            await db.execute(f"SAVEPOINT {savepoint}")
            try:
                for stmt in statements:
                    await db.execute(stmt)
                await db.execute(
                    _SQL_UPSERT_VERSION,
                    (manifest.name, new_version, time.time(), basename, manifest.version),
                )
                await db.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception as exc:
                await db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                await db.execute(f"RELEASE SAVEPOINT {savepoint}")
                msg = f"SQL execution failed: {exc}"
                raise ExtensionMigrationError(manifest.name, msg, migration_file=basename) from exc

            await db.commit()

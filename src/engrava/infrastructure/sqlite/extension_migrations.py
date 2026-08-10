"""Extension schema migration runner.

Applies SQL migration scripts declared in ``ExtensionManifest.schema_migrations``.

Every applied migration is recorded in the append-only
``extension_schema_migrations`` history table, keyed by extension name and
1-based ordinal index, storing the migration filename together with a content
checksum.  Pending work is derived from that history, and the recorded
filename/checksum of every already-applied migration is re-verified against the
current manifest on each run so that a renamed, reordered, inserted, or edited
earlier migration is rejected as historical drift *before* any migration runs.

The single-row-per-extension ``extension_schema_versions`` summary table is kept
up to date as a convenience lookup for the latest applied migration.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.resources
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import aiosqlite

    from engrava.domain.manifest import ExtensionManifest

from engrava.domain.exceptions import ExtensionMigrationError

_CHECKSUM_ALGORITHM = "sha256"

# Leading keywords of statements that manage transaction or savepoint state.
# Extension migrations are pure schema/data changes; the runner owns the
# transaction, so a migration that opens/commits one would subvert the
# per-migration savepoint that guarantees atomic application.  These are
# rejected during preparation, before any statement runs.
_TRANSACTION_CONTROL_KEYWORDS = frozenset(
    {"BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE"}
)


def _checksum(sql: str) -> str:
    """Return a stable content checksum for a migration file's SQL text.

    The digest is taken over the *decoded* SQL string returned by
    :func:`_read_migration_sql` (UTF-8 with any leading BOM stripped), not over
    the raw file bytes.  This is deliberately a **semantic-content** fingerprint,
    not a byte-exact file identity: a BOM toggle or a CRLF/LF/CR line-ending
    change does not alter the checksum.  That tolerates cosmetic cross-platform
    differences (for example a Windows checkout with ``core.autocrlf``) while
    still detecting any change to the SQL content itself.

    Args:
        sql: The full decoded SQL text of a migration file.

    Returns:
        An algorithm-prefixed hex digest (``"sha256:<hex>"``) usable for exact
        string comparison against a previously recorded checksum.

    """
    digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
    return f"{_CHECKSUM_ALGORITHM}:{digest}"


class _UnsupportedMigrationSQLError(ValueError):
    """Raised when a migration file cannot be split into terminated statements.

    Signals a structurally unsupported migration shape (an unclosed
    ``BEGIN...END`` block, an unterminated string literal, an unclosed block
    comment, or a final statement missing its terminating semicolon).  The
    runner translates this into a typed :class:`ExtensionMigrationError` before
    any statement of the offending file is applied.
    """


def _consume_quoted(sql: str, start: int, quote: str) -> tuple[int, bool]:
    """Return the index just past a quoted string or identifier.

    SQLite escapes a closing quote by doubling it (``''``, ``""`` or backticks);
    backslashes are *not* escape characters.  An unterminated quote consumes to
    the end of the input.

    Args:
        sql: Full SQL text being scanned.
        start: Index of the opening quote character in *sql*.
        quote: The quote character (``'``, ``"`` or a backtick).

    Returns:
        ``(next_index, terminated)`` where ``next_index`` is the index
        immediately after the closing quote (or ``len(sql)`` when the quote is
        never closed) and ``terminated`` is ``True`` only when a closing quote
        was found.

    """
    n = len(sql)
    i = start + 1
    while i < n:
        if sql[i] == quote:
            if i + 1 < n and sql[i + 1] == quote:
                i += 2  # doubled quote — an escaped literal quote, not a close
                continue
            return (i + 1, True)
        i += 1
    return (n, False)


def _scan_non_statement_region(sql: str, i: int) -> tuple[int, bool, bool] | None:
    """Scan a comment or quoted region starting at *i*, if one begins there.

    Recognises ``--`` line comments, ``/* */`` block comments, single-quoted
    strings, double-quoted / backtick identifiers and ``[...]`` bracket
    identifiers — the regions whose contents (including any semicolons) must not
    be treated as statement text or boundaries.

    Args:
        sql: Full SQL text being scanned.
        i: Current scan index into *sql*.

    Returns:
        ``(next_index, is_content, terminated)`` when a region begins at *i* —
        where ``next_index`` is the first index past the region, ``is_content``
        is ``True`` for quoted literals/identifiers (real statement content) and
        ``False`` for comments, and ``terminated`` is ``True`` unless the region
        runs to end-of-input without its closing delimiter.  A ``--`` line
        comment is always terminated (end-of-input closes it).  ``None`` when no
        such region begins at *i*.

    """
    n = len(sql)
    ch = sql[i]
    if ch == "-" and i + 1 < n and sql[i + 1] == "-":  # line comment
        newline = sql.find("\n", i + 2)
        return (n, False, True) if newline == -1 else (newline + 1, False, True)
    if ch == "/" and i + 1 < n and sql[i + 1] == "*":  # block comment
        close = sql.find("*/", i + 2)
        return (n, False, False) if close == -1 else (close + 2, False, True)
    if ch in ("'", '"', "`"):  # string literal or quoted identifier
        next_index, terminated = _consume_quoted(sql, i, ch)
        return (next_index, True, terminated)
    if ch == "[":  # bracket-quoted identifier (no nesting/escape in SQLite)
        close = sql.find("]", i + 1)
        return (n, True, False) if close == -1 else (close + 1, True, True)
    return None


def _leading_keyword(statement: str) -> str:
    """Return the uppercased leading SQL keyword of *statement*.

    Leading whitespace and comments are skipped so the first executable token is
    returned.  An empty string is returned when the statement begins with a
    non-keyword token (for example a quoted identifier).

    Args:
        statement: A single SQL statement, possibly comment-prefixed.

    Returns:
        The first keyword token uppercased, or ``""`` when none is present.

    """
    n = len(statement)
    i = 0
    while i < n:
        ch = statement[i]
        if ch.isspace() or ch == "\ufeff":  # whitespace or a stray BOM
            i += 1
            continue
        region = _scan_non_statement_region(statement, i)
        if region is not None and not region[1]:  # a comment region — skip it
            i = region[0]
            continue
        break
    end = i
    while end < n and (statement[end].isalpha() or statement[end] == "_"):
        end += 1
    return statement[i:end].upper()


def _split_sql_statements(sql: str) -> list[str]:
    """Split *sql* into individual, complete SQLite statements.

    The splitter is SQLite-aware: it skips semicolons inside single-quoted
    strings, double-quoted / backtick / bracket identifiers, ``--`` line
    comments and ``/* */`` block comments, and treats a compound
    ``BEGIN ... END`` trigger body as a single statement (its internal
    semicolons are not statement terminators).  Statement boundaries are
    confirmed with :func:`sqlite3.complete_statement` so that trigger bodies are
    handled exactly as SQLite parses them.

    Every statement must be terminated by a semicolon.  Any trailing content
    that is not so terminated — an unclosed ``BEGIN...END`` block, an
    unterminated string literal, an unclosed block comment mid-statement, or a
    final statement missing its ``;`` — is an unsupported shape and is rejected
    before any statement is returned.

    Args:
        sql: SQL text that may contain multiple ``';'``-separated statements.

    Returns:
        Non-empty list of individual SQL statement strings (each including its
        terminating semicolon).  Whitespace- and comment-only fragments are
        discarded.

    Raises:
        _UnsupportedMigrationSQLError: If the text ends with an unterminated or
            otherwise unsupported statement.

    """
    statements: list[str] = []
    stmt_start = 0
    content_since_emit = False
    i = 0
    n = len(sql)

    while i < n:
        region = _scan_non_statement_region(sql, i)
        if region is not None:
            next_index, is_content, terminated = region
            if not terminated:
                kind = "string literal or quoted identifier" if is_content else "block comment"
                msg = f"unsupported migration SQL: unterminated {kind}."
                raise _UnsupportedMigrationSQLError(msg)
            content_since_emit = content_since_emit or is_content
            i = next_index
            continue

        ch = sql[i]
        if ch == ";":
            candidate = sql[stmt_start : i + 1]
            if sqlite3.complete_statement(candidate):
                if content_since_emit:
                    statements.append(candidate.strip())
                stmt_start = i + 1
                content_since_emit = False
            # Otherwise the semicolon closes a statement inside a compound
            # ``BEGIN...END`` body — keep scanning for the real boundary.
            i += 1
            continue

        if not ch.isspace():
            content_since_emit = True
        i += 1

    if content_since_emit:
        remainder = sql[stmt_start:].strip()
        msg = (
            "unsupported migration SQL: the file ends with a statement that is "
            "not terminated by a semicolon (an unclosed 'BEGIN...END' block or a "
            f"missing ';'): {remainder[:120]!r}"
        )
        raise _UnsupportedMigrationSQLError(msg)

    return statements


_SQL_ENSURE_VERSIONS_TABLE = """\
CREATE TABLE IF NOT EXISTS extension_schema_versions (
    extension_name    TEXT PRIMARY KEY,
    version           INTEGER NOT NULL DEFAULT 0,
    applied_at        REAL NOT NULL,
    migration_file    TEXT,
    extension_version TEXT
)"""

_SQL_ENSURE_HISTORY_TABLE = """\
CREATE TABLE IF NOT EXISTS extension_schema_migrations (
    extension_name    TEXT NOT NULL,
    migration_index   INTEGER NOT NULL,
    migration_file    TEXT NOT NULL,
    checksum          TEXT NOT NULL,
    applied_at        REAL NOT NULL,
    extension_version TEXT,
    PRIMARY KEY (extension_name, migration_index)
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

_SQL_INSERT_HISTORY = """\
INSERT INTO extension_schema_migrations
    (extension_name, migration_index, migration_file, checksum, applied_at, extension_version)
VALUES (?, ?, ?, ?, ?, ?)"""


@dataclass(frozen=True)
class _AppliedMigration:
    """A single recorded entry from the append-only migration history.

    Attributes:
        index: 1-based ordinal position of the migration in filename-sort order.
        migration_file: Basename recorded when the migration was applied.
        checksum: Algorithm-prefixed content checksum recorded at apply time.

    """

    index: int
    migration_file: str
    checksum: str


@dataclass(frozen=True)
class _PreparedMigration:
    """A pending migration whose SQL has been read, checksummed and split.

    Attributes:
        index: 1-based ordinal position assigned to this migration.
        basename: Basename of the migration file.
        checksum: Content checksum to record on successful application.
        statements: Individual, terminated SQL statements to execute in order.

    """

    index: int
    basename: str
    checksum: str
    statements: list[str]


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

    The file is decoded as UTF-8 with a leading byte-order mark stripped
    (``utf-8-sig``), so a BOM cannot prefix — and thereby hide the leading
    keyword of — the first statement.

    Returns:
        SQL text content of the migration file.

    Raises:
        ExtensionMigrationError: If the path cannot be resolved (hooks class
            lives in a non-installable module), the file cannot be read, or its
            content is not valid UTF-8.

    """
    if path.is_absolute():
        try:
            return path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            msg = f"Cannot read migration file {str(path)!r}: {exc}"
            raise ExtensionMigrationError(manifest.name, msg, migration_file=path.name) from exc

    if manifest.package_root is not None:
        resolved = manifest.package_root / path
        try:
            return resolved.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
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
        return traversable.read_text(encoding="utf-8-sig")
    except Exception as exc:
        msg = f"Cannot read migration file {str(path)!r} from package {top_level!r}: {exc}"
        raise ExtensionMigrationError(manifest.name, msg, migration_file=path.name) from exc


class ExtensionMigrationRunner:
    """Applies pending SQL migration files declared in an ``ExtensionManifest``.

    Migrations are executed in filename-sort order (convention:
    ``NNN_slug.sql``, e.g. ``001_initial.sql``, ``002_add_tags.sql``).

    Each successfully applied migration is appended to the immutable
    ``extension_schema_migrations`` history table (extension name, 1-based
    ordinal index, filename, content checksum).  Pending work is the suffix of
    the manifest's migration list beyond the recorded history.  Before applying
    anything, the runner re-verifies that every already-applied migration still
    matches its recorded filename and checksum, rejecting historical drift with
    a typed :class:`ExtensionMigrationError`.

    A database written by an older engrava build tracks only the
    ``extension_schema_versions`` summary row.  On first run under this build the
    runner adopts that recorded version as the history baseline, computing the
    current checksums of the already-applied files (no prior checksum exists to
    verify against — drift on those files becomes detectable only from this
    point forward).

    Both the history table and the summary table are created on first use if
    absent (idempotent DDL), so the bookkeeping schema upgrade is self-applying
    and safe to re-run.

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

        Args:
            manifest: Extension manifest with ``schema_migrations`` paths.
            db: An open aiosqlite connection (core schema already applied).

        Raises:
            ExtensionMigrationError: On historical drift (a recorded migration's
                filename or checksum no longer matches the manifest), corrupt or
                duplicate migration filenames, a summary/history bookkeeping
                mismatch, downgrade detection, an unsupported/unterminated or
                transaction-control migration SQL shape, SQL execution failure
                (the savepoint is rolled back so the database is left unchanged),
                or an unresolvable relative path.

        """
        if not manifest.schema_migrations:
            # An extension that ships no migrations manages nothing here — but a
            # database that already recorded applied migrations for it is a
            # downgrade (every migration file removed) that must not pass
            # silently.  Never create bookkeeping tables just to no-op.
            recorded = await self._recorded_migration_count(db, manifest.name)
            if recorded > 0:
                raise ExtensionMigrationError(manifest.name, _downgrade_message(recorded, 0))
            return

        sorted_files = sorted(manifest.schema_migrations, key=lambda p: p.name)
        total = len(sorted_files)
        self._reject_duplicate_filenames(manifest, sorted_files)

        # Ensure both bookkeeping tables exist — idempotent DDL.
        await db.execute(_SQL_ENSURE_VERSIONS_TABLE)
        await db.execute(_SQL_ENSURE_HISTORY_TABLE)

        applied = await self._load_history(db, manifest.name)
        if applied:
            await self._verify_summary_matches_history(db, manifest, len(applied))
        else:
            applied = await self._adopt_legacy_history(db, manifest, sorted_files, total)

        current_version = len(applied)
        if current_version > total:
            raise ExtensionMigrationError(manifest.name, _downgrade_message(current_version, total))

        self._verify_no_drift(manifest, sorted_files, applied)

        prepared = self._prepare_pending(manifest, sorted_files, current_version)

        for step in prepared:
            savepoint = self._savepoint_name(manifest.name, str(step.index))
            applied_at = time.time()

            # A savepoint keeps the migration SQL, the history append and the
            # summary upsert atomic.  SQLite DDL is transactional, so a ROLLBACK
            # undoes CREATE TABLE / ALTER TABLE statements too.
            await db.execute(f"SAVEPOINT {savepoint}")
            try:
                for stmt in step.statements:
                    await db.execute(stmt)
                await db.execute(
                    _SQL_INSERT_HISTORY,
                    (
                        manifest.name,
                        step.index,
                        step.basename,
                        step.checksum,
                        applied_at,
                        manifest.version,
                    ),
                )
                await db.execute(
                    _SQL_UPSERT_VERSION,
                    (manifest.name, step.index, applied_at, step.basename, manifest.version),
                )
                await db.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception as exc:
                # Roll the failed step back; suppress any secondary cleanup error
                # so the typed migration error is what surfaces to the caller.
                with contextlib.suppress(Exception):
                    await db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    await db.execute(f"RELEASE SAVEPOINT {savepoint}")
                msg = f"SQL execution failed: {exc}"
                raise ExtensionMigrationError(
                    manifest.name, msg, migration_file=step.basename
                ) from exc

            await db.commit()

    @staticmethod
    def _savepoint_name(extension_name: str, suffix: str) -> str:
        """Build a safe savepoint identifier scoped to an extension and step.

        Every character of *extension_name* outside ASCII ``[A-Za-z0-9_]`` is
        replaced with ``_`` so an unusual extension name cannot produce invalid
        or injectable savepoint SQL.  The savepoint is scoped to a single
        ``apply_pending`` call and released before returning, so the lossy
        mapping cannot collide with another extension's live savepoint.

        Args:
            extension_name: Name of the extension being migrated.
            suffix: Step discriminator (the ordinal index or an operation tag).

        Returns:
            A savepoint identifier composed only of safe identifier characters.

        """
        safe = "".join(
            c if c == "_" or (c.isascii() and c.isalnum()) else "_" for c in extension_name
        )
        return f"ext_mig_{safe}_{suffix}"

    @staticmethod
    def _reject_duplicate_filenames(
        manifest: ExtensionManifest,
        sorted_files: list[Path],
    ) -> None:
        """Reject a manifest that declares two migrations with the same basename.

        Migration identity and ordering are keyed by filename; duplicate
        basenames (from different directories) would make identity ambiguous.

        Args:
            manifest: Extension manifest under migration.
            sorted_files: Migration paths in filename-sort order.

        Raises:
            ExtensionMigrationError: If any basename occurs more than once.

        """
        seen: set[str] = set()
        for path in sorted_files:
            if path.name in seen:
                msg = (
                    f"Duplicate migration filename {path.name!r}: migration "
                    f"identity is keyed by filename, so basenames must be unique."
                )
                raise ExtensionMigrationError(manifest.name, msg, migration_file=path.name)
            seen.add(path.name)

    @staticmethod
    async def _recorded_migration_count(
        db: aiosqlite.Connection,
        extension_name: str,
    ) -> int:
        """Return the count of migrations already recorded for an extension.

        Reads whichever bookkeeping tables exist — the append-only history and
        the legacy summary — and returns the larger recorded count so that a
        downgrade is detectable regardless of which bookkeeping generation wrote
        the database.

        Args:
            db: Open connection (bookkeeping tables may or may not exist).
            extension_name: Name of the extension to inspect.

        Returns:
            The highest recorded applied-migration count, or ``0`` when nothing
            has been recorded for the extension.

        """
        count = 0
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('extension_schema_migrations', 'extension_schema_versions')"
        )
        existing = {str(row[0]) for row in await cursor.fetchall()}

        if "extension_schema_migrations" in existing:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM extension_schema_migrations WHERE extension_name = ?",
                (extension_name,),
            )
            row = await cursor.fetchone()
            count = max(count, int(row[0]) if row else 0)

        if "extension_schema_versions" in existing:
            cursor = await db.execute(
                "SELECT version FROM extension_schema_versions WHERE extension_name = ?",
                (extension_name,),
            )
            row = await cursor.fetchone()
            count = max(count, int(row[0]) if row else 0)

        return count

    async def _load_history(
        self,
        db: aiosqlite.Connection,
        extension_name: str,
    ) -> list[_AppliedMigration]:
        """Load the append-only applied-migration history for an extension.

        Args:
            db: Open connection with the bookkeeping tables present.
            extension_name: Name of the extension whose history to load.

        Returns:
            Recorded migrations ordered by ascending ordinal index.

        """
        cursor = await db.execute(
            "SELECT migration_index, migration_file, checksum "
            "FROM extension_schema_migrations "
            "WHERE extension_name = ? ORDER BY migration_index",
            (extension_name,),
        )
        rows = await cursor.fetchall()
        return [
            _AppliedMigration(index=int(row[0]), migration_file=str(row[1]), checksum=str(row[2]))
            for row in rows
        ]

    async def _verify_summary_matches_history(
        self,
        db: aiosqlite.Connection,
        manifest: ExtensionManifest,
        history_length: int,
    ) -> None:
        """Cross-check the legacy summary version against the history length.

        The runner writes the append-only history row and the single-row summary
        together in one savepoint, so the summary version must equal the number
        of recorded history rows.  A mismatch means the bookkeeping is corrupt —
        a lost or truncated history tail row while the summary still claims a
        higher version would otherwise let the missing migration silently become
        "pending" and be reapplied (and an edited copy of that file would evade
        checksum comparison).  Fail closed rather than trust the shorter history.

        Args:
            db: Open connection with the bookkeeping tables present.
            manifest: Extension manifest under migration.
            history_length: Number of contiguous rows in the applied history.

        Raises:
            ExtensionMigrationError: If the summary version (``0`` when no summary
                row exists) does not equal *history_length*, or the stored summary
                value is not an integer.

        """
        cursor = await db.execute(
            "SELECT version FROM extension_schema_versions WHERE extension_name = ?",
            (manifest.name,),
        )
        row = await cursor.fetchone()
        stored = row[0] if row is not None else 0
        # Fail closed on a non-integer summary. ``extension_schema_versions`` is a
        # non-STRICT ``INTEGER`` column, so a corrupted/tampered bookkeeping row can
        # hold a ``REAL`` or text value; ``int(stored)`` would silently truncate
        # ``1.5`` to ``1`` (masking a mismatch) or raise an untyped error. ``bool``
        # is rejected explicitly since it is an ``int`` subclass.
        if isinstance(stored, bool) or not isinstance(stored, int):
            msg = (
                f"Inconsistent migration bookkeeping: the summary version {stored!r} "
                f"is not an integer.  The bookkeeping is corrupt; refusing to proceed."
            )
            raise ExtensionMigrationError(manifest.name, msg)
        summary_version = stored
        if summary_version != history_length:
            msg = (
                f"Inconsistent migration bookkeeping: the summary records "
                f"{summary_version} applied migration(s) but the append-only history "
                f"holds {history_length}.  The bookkeeping is corrupt; refusing to "
                f"proceed so a missing migration is not silently reapplied."
            )
            raise ExtensionMigrationError(manifest.name, msg)

    async def _adopt_legacy_history(
        self,
        db: aiosqlite.Connection,
        manifest: ExtensionManifest,
        sorted_files: list[Path],
        total: int,
    ) -> list[_AppliedMigration]:
        """Backfill history from a pre-checksum ``extension_schema_versions`` row.

        A database written before checksum history existed records only a latest
        version number.  This adopts the first ``version`` migration files as the
        immutable baseline, recording their *current* checksums (there is no
        prior checksum to verify against).

        Trust boundary — this is an explicit, understood limitation, not a silent
        gap: the already-applied files are **trusted** at adoption because no
        historical checksum exists to retro-verify them against.  Drift detection
        for those pre-existing migrations is therefore **prospective** — it
        catches any change made *after* this upgrade point, but cannot detect an
        edit that predates the adoption.  Migrations applied under this build
        (and every later one) are checksum-verified from the moment they run.

        Args:
            db: Open connection with the bookkeeping tables present.
            manifest: Extension manifest under migration.
            sorted_files: Migration paths in filename-sort order.
            total: Number of migration files the manifest declares.

        Returns:
            The adopted baseline as history entries, or an empty list when there
            is no legacy summary row to adopt.

        Raises:
            ExtensionMigrationError: If the recorded legacy version exceeds the
                number of declared migration files (downgrade), or a baseline
                file cannot be read.

        """
        cursor = await db.execute(
            "SELECT version, applied_at, extension_version "
            "FROM extension_schema_versions WHERE extension_name = ?",
            (manifest.name,),
        )
        row = await cursor.fetchone()
        if row is None:
            return []
        legacy_version = int(row[0])
        if legacy_version <= 0:
            return []
        if legacy_version > total:
            raise ExtensionMigrationError(manifest.name, _downgrade_message(legacy_version, total))

        legacy_applied_at = float(row[1])
        legacy_extension_version = None if row[2] is None else str(row[2])

        # Read + checksum every baseline file first; a read failure raises before
        # any write, keeping the backfill atomic.
        baseline = [
            _AppliedMigration(
                index=index,
                migration_file=sorted_files[index - 1].name,
                checksum=_checksum(_read_migration_sql(manifest, sorted_files[index - 1])),
            )
            for index in range(1, legacy_version + 1)
        ]

        savepoint = self._savepoint_name(manifest.name, "adopt")
        await db.execute(f"SAVEPOINT {savepoint}")
        try:
            for entry in baseline:
                await db.execute(
                    _SQL_INSERT_HISTORY,
                    (
                        manifest.name,
                        entry.index,
                        entry.migration_file,
                        entry.checksum,
                        legacy_applied_at,
                        legacy_extension_version,
                    ),
                )
            await db.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception as exc:
            with contextlib.suppress(Exception):
                await db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                await db.execute(f"RELEASE SAVEPOINT {savepoint}")
            msg = f"Failed to adopt legacy migration history: {exc}"
            raise ExtensionMigrationError(manifest.name, msg) from exc
        await db.commit()
        return baseline

    def _verify_no_drift(
        self,
        manifest: ExtensionManifest,
        sorted_files: list[Path],
        applied: list[_AppliedMigration],
    ) -> None:
        """Reject historical drift before any pending migration is applied.

        Confirms the recorded history is a contiguous ``1..N`` prefix and that
        each already-applied migration still matches the manifest file at its
        ordinal position by both filename and content checksum.

        Args:
            manifest: Extension manifest under migration.
            sorted_files: Migration paths in filename-sort order.
            applied: Recorded history (already known to satisfy ``len <= total``).

        Raises:
            ExtensionMigrationError: On non-contiguous history, a filename
                mismatch (a reordered/renamed/inserted earlier migration), or a
                content checksum mismatch (an edited earlier migration).

        """
        for position, entry in enumerate(applied, start=1):
            if entry.index != position:
                msg = (
                    f"Corrupt migration history: expected contiguous indices but "
                    f"found index {entry.index} at position {position}."
                )
                raise ExtensionMigrationError(manifest.name, msg)

            current_path = sorted_files[entry.index - 1]
            if current_path.name != entry.migration_file:
                msg = (
                    f"Historical migration drift: migration #{entry.index} was "
                    f"applied as {entry.migration_file!r} but the manifest now "
                    f"provides {current_path.name!r} at that position.  Applied "
                    f"migrations are immutable; do not rename, reorder, or insert "
                    f"earlier migration files."
                )
                raise ExtensionMigrationError(
                    manifest.name, msg, migration_file=entry.migration_file
                )

            current_checksum = _checksum(_read_migration_sql(manifest, current_path))
            if current_checksum != entry.checksum:
                msg = (
                    f"Historical migration drift: migration #{entry.index} "
                    f"({entry.migration_file!r}) content changed since it was "
                    f"applied (checksum mismatch).  Applied migrations are "
                    f"immutable; add a new migration file instead of editing an "
                    f"applied one."
                )
                raise ExtensionMigrationError(
                    manifest.name, msg, migration_file=entry.migration_file
                )

    def _prepare_pending(
        self,
        manifest: ExtensionManifest,
        sorted_files: list[Path],
        current_version: int,
    ) -> list[_PreparedMigration]:
        """Read, checksum, split and validate every pending migration up front.

        Splitting and validating each pending file before any is applied
        guarantees that an unsupported SQL shape fails before a single statement
        runs.  Transaction-control statements are rejected here because they
        would subvert the per-migration savepoint that provides atomicity.

        Args:
            manifest: Extension manifest under migration.
            sorted_files: Migration paths in filename-sort order.
            current_version: Count of already-applied migrations.

        Returns:
            Prepared pending migrations in application order.

        Raises:
            ExtensionMigrationError: If a pending file cannot be read, contains
                an unsupported/unterminated statement shape, or contains a
                transaction-control statement.

        """
        prepared: list[_PreparedMigration] = []
        for index, path in enumerate(sorted_files[current_version:], start=current_version + 1):
            sql = _read_migration_sql(manifest, path)
            try:
                statements = _split_sql_statements(sql)
            except _UnsupportedMigrationSQLError as exc:
                raise ExtensionMigrationError(
                    manifest.name, str(exc), migration_file=path.name
                ) from exc
            self._reject_transaction_control(manifest, path.name, statements)
            prepared.append(
                _PreparedMigration(
                    index=index,
                    basename=path.name,
                    checksum=_checksum(sql),
                    statements=statements,
                )
            )
        return prepared

    @staticmethod
    def _reject_transaction_control(
        manifest: ExtensionManifest,
        basename: str,
        statements: list[str],
    ) -> None:
        """Reject transaction- or savepoint-control statements in a migration.

        The runner wraps each migration in a savepoint, so a statement that
        opens or ends a transaction/savepoint would break atomic application.

        Args:
            manifest: Extension manifest under migration.
            basename: Basename of the migration file being validated.
            statements: Split statements of the migration.

        Raises:
            ExtensionMigrationError: If any statement's leading keyword is a
                transaction- or savepoint-control verb.

        """
        for statement in statements:
            keyword = _leading_keyword(statement)
            if keyword in _TRANSACTION_CONTROL_KEYWORDS:
                msg = (
                    f"unsupported migration SQL: transaction-control statement "
                    f"{keyword!r} is not permitted in an extension migration; the "
                    f"runner manages the transaction."
                )
                raise ExtensionMigrationError(manifest.name, msg, migration_file=basename)


def _downgrade_message(current_version: int, total: int) -> str:
    """Build the downgrade-detection error message.

    Args:
        current_version: Number of migrations recorded as applied.
        total: Number of migration files the manifest declares.

    Returns:
        A human-readable explanation with recovery guidance.

    """
    return (
        f"Extension downgrade detected: the database records {current_version} "
        f"applied migration(s) but the manifest declares only {total} migration "
        f"file(s).  Revert to an extension release that provides at least "
        f"{current_version} migration file(s), or clear the recorded history for "
        f"this extension."
    )

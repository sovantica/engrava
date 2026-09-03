"""CLI smoke tests for ``engrava`` command.

Tests all subcommands against an in-memory (temp file) SQLite database.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
from click.testing import CliRunner

from engrava.cli.config import EngravaCLIConfig
from engrava.cli.main import cli

# Literal SQL, never interpolated: a read-back that assembles its own query
# cannot be trusted to disagree with the schema the command wrote to.
_CORE_ID_QUERIES: tuple[tuple[str, str], ...] = (
    ("thought", "SELECT thought_id FROM thought"),
    ("edge", "SELECT edge_id FROM edge"),
    ("embedding", "SELECT embedding_id FROM embedding"),
)


def _missing_snapshot(tmp_path: Path) -> Path:
    """Build a ``--input`` path that is not there — the likeliest typo."""
    return tmp_path / "no-such-snapshot.jsonl"


def _directory_snapshot(tmp_path: Path) -> Path:
    """Build a ``--input`` path naming a directory — the backup folder, not the file."""
    directory = tmp_path / "backups"
    directory.mkdir()
    return directory


def _binary_snapshot(tmp_path: Path) -> Path:
    """Build a ``--input`` path that opens but holds no UTF-8 text — e.g. a database."""
    binary = tmp_path / "looks-like-a-snapshot.jsonl"
    binary.write_bytes(b"\xff\xfe\x00\x01not text at all\n")
    return binary


#: The unusable ``--input`` paths reachable from the command line, declared once
#: so the tests that assert the message and the test that asserts the error type
#: cannot come to disagree about what they are feeding the command. A read that
#: fails after a successful open is not reachable this way and is covered where
#: the iterator itself is tested.
_UNUSABLE_SNAPSHOTS: tuple[Callable[[Path], Path], ...] = (
    _missing_snapshot,
    _directory_snapshot,
    _binary_snapshot,
)


def _stored_core_ids(db_path: Path) -> dict[str, set[str]]:
    """Read the stored core-row ids back from the database file itself.

    The CLI owns and closes its own connection, so what it left behind is read
    over an independent one rather than taken from the command's report.

    Args:
        db_path: Path to the database the CLI operated on.

    Returns:
        Every stored id, per core table.

    """
    conn = sqlite3.connect(db_path)
    try:
        return {
            table: {str(row[0]) for row in conn.execute(query)} for table, query in _CORE_ID_QUERIES
        }
    finally:
        conn.close()


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a temporary DB path."""
    return tmp_path / "test.db"


@pytest.fixture
def populated_db(db_path: Path) -> Path:
    """Create a DB with schema and sample data, return its path."""
    import asyncio

    import aiosqlite

    from engrava import (
        EdgeRecord,
        EdgeType,
        LifecycleStatus,
        Priority,
        SqliteEngravaCore,
        ThoughtRecord,
        ThoughtType,
    )

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        for i in range(3):
            t = ThoughtRecord(
                thought_id=f"thought-{i:03d}",
                essence=f"Test thought {i}",
                content=f"Test thought number {i}",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE if i < 2 else LifecycleStatus.ARCHIVED,
                priority=Priority.P2,
                created_cycle=i + 1,
                updated_cycle=i + 1,
            )
            await store.create_thought(t)
            await store.store_embedding(f"thought-{i:03d}", [float(i)] * 16)

        edge = EdgeRecord(
            edge_id="edge-001",
            from_thought_id="thought-000",
            to_thought_id="thought-001",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.9,
            created_cycle=1,
        )
        await store.create_edge(edge)
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())
    return db_path


class TestGlobalControls:
    """Tests for extension isolation and verbose logging controls."""

    def test_no_extensions_skips_cli_discovery_for_help(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _unexpected_discovery() -> list[object]:
            msg = "CLI extension discovery must stay disabled"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "engrava.cli.main._discover_extension_commands",
            _unexpected_discovery,
        )

        result = runner.invoke(cli, ["--no-extensions", "--help"])

        assert result.exit_code == 0
        assert "--no-extensions" in result.output

    def test_disable_extensions_environment_variable_skips_cli_discovery(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _unexpected_discovery() -> list[object]:
            msg = "CLI extension discovery must stay disabled"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "engrava.cli.main._discover_extension_commands",
            _unexpected_discovery,
        )

        result = runner.invoke(
            cli,
            ["--help"],
            env={"ENGRAVA_DISABLE_EXTENSIONS": "1"},
        )

        assert result.exit_code == 0

    def test_no_extensions_skips_mindql_discovery(
        self,
        runner: CliRunner,
        populated_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _unexpected_discovery() -> dict[str, object]:
            msg = "MindQL extension discovery must stay disabled"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "engrava.cli.main._load_mindql_extensions",
            _unexpected_discovery,
        )

        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--no-extensions",
                "query",
                "SELECT thought_id FROM thought LIMIT 1",
            ],
        )

        assert result.exit_code == 0
        assert "thought-000" in result.output

    def test_verbose_emits_debug_logging(
        self,
        runner: CliRunner,
        populated_db: Path,
    ) -> None:
        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        capture_handler = _CaptureHandler()
        package_logger = logging.getLogger("engrava")
        package_logger.addHandler(capture_handler)
        try:
            result = runner.invoke(
                cli,
                ["--db", str(populated_db), "--verbose", "info"],
            )
        finally:
            package_logger.removeHandler(capture_handler)

        assert result.exit_code == 0
        assert any(
            record.levelno == logging.DEBUG and record.getMessage() == "Verbose logging enabled"
            for record in records
        )


class TestInfo:
    """Tests for ``engrava info``."""

    def test_info_table_format(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "info"])
        assert result.exit_code == 0
        assert "Thoughts: 3" in result.output
        assert "Edges: 1" in result.output

    def test_info_json_format(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "--format", "json", "info"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["thoughts"]["total"] == 3
        assert data["edges"]["total"] == 1
        assert data["schema_version"] == 1
        assert data["search_latency"]["sample_count"] == 0

    def test_info_missing_db(self, runner: CliRunner, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.db"
        result = runner.invoke(cli, ["--db", str(missing), "info"])
        assert result.exit_code != 0


class TestQuery:
    """Tests for ``engrava query``."""

    def test_query_select(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "query", "SELECT thought_id FROM thought"],
        )
        assert result.exit_code == 0
        assert "thought-000" in result.output

    def test_query_json(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--format",
                "json",
                "query",
                "SELECT thought_id, content FROM thought LIMIT 1",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert "thought_id" in data[0]

    def test_query_rejects_non_select_sql(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "query", "DELETE FROM thought"],
        )
        assert result.exit_code != 0

    def test_query_csv(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--format",
                "csv",
                "query",
                "SELECT thought_id FROM thought LIMIT 2",
            ],
        )
        assert result.exit_code == 0
        assert "thought_id" in result.output
        assert "thought-000" in result.output

    def test_query_mindql_find(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "query", "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"],
        )
        assert result.exit_code == 0
        assert "thought-000" in result.output

    def test_query_mindql_count(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "--db",
                str(populated_db),
                "--format",
                "json",
                "query",
                "COUNT thoughts",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["count"] == 3


class TestSnapshot:
    """Tests for ``engrava snapshot``."""

    def test_snapshot_creates_file(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "snap.jsonl"
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "snapshot", "-o", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3  # 3 thoughts minimum
        for line in lines:
            record = json.loads(line)
            assert "_type" in record

    def test_snapshot_default_path(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "snapshot"])
        assert result.exit_code == 0
        default_out = populated_db.with_suffix(".snapshot.jsonl")
        assert default_out.exists()

    def test_snapshot_invalid_service_name_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
    ) -> None:
        """An invalid --service value is a clean CLI error, never a traceback."""
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "snapshot", "--service", "../escape"],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        # It was handled as a ClickException (clean exit), not an uncaught error.
        assert isinstance(result.exception, SystemExit)

    @pytest.mark.parametrize("bad_service", ["", "   "])
    def test_snapshot_empty_service_name_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        bad_service: str,
    ) -> None:
        """An explicit empty/whitespace --service is validated, not routed away.

        Such a value is falsy, so it must be rejected on `is not None` rather
        than a truthiness check that would fall through to the single-database
        path.
        """
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "snapshot", "--service", bad_service],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        assert isinstance(result.exception, SystemExit)


class TestRestore:
    """Tests for ``engrava restore``."""

    def test_restore_roundtrip(self, runner: CliRunner, populated_db: Path, tmp_path: Path) -> None:
        snap = tmp_path / "snap.jsonl"
        runner.invoke(cli, ["--db", str(populated_db), "snapshot", "-o", str(snap)])

        new_db = tmp_path / "restored.db"
        result = runner.invoke(
            cli,
            ["--db", str(new_db), "restore", "-i", str(snap)],
        )
        assert result.exit_code == 0
        assert "Restored" in result.output

        # Verify restored data
        check = runner.invoke(
            cli,
            ["--db", str(new_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 3

    def test_restore_with_clear(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        snap = tmp_path / "snap.jsonl"
        runner.invoke(cli, ["--db", str(populated_db), "snapshot", "-o", str(snap)])

        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "restore", "-i", str(snap), "--clear"],
        )
        assert result.exit_code == 0

    def test_restore_invalid_service_name_is_distinct_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """An invalid --service value is a clean, distinct error on restore.

        It must not be mislabelled as an embedding-provider initialisation
        failure, and must not print a traceback. Validation happens before any
        snapshot file is read, so the input path need not exist.
        """
        snap = tmp_path / "missing.jsonl"
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(snap), "--service", "../escape"],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        assert "embedding provider" not in result.output
        assert isinstance(result.exception, SystemExit)

    @pytest.mark.parametrize("bad_service", ["", "   "])
    def test_restore_empty_service_name_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
        bad_service: str,
    ) -> None:
        """An explicit empty/whitespace --service on restore is rejected cleanly.

        Such a value is falsy; validating on `is not None` gives it the same
        clean ClickException as any other malformed name instead of silently
        falling through to the single-database path.
        """
        snap = tmp_path / "missing.jsonl"
        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(snap), "--service", bad_service],
        )
        assert result.exit_code != 0
        assert "Invalid --service value" in result.output
        assert "embedding provider" not in result.output
        assert isinstance(result.exception, SystemExit)

    def test_restore_missing_input_file_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """A ``-i`` path that is not there is a clean CLI error, never a traceback.

        Mistyping the snapshot path is the likeliest way to get this command
        wrong, so it is held to the same standard as an invalid ``--service``:
        a message naming the path and the option, not an ``OSError`` the user
        has to read a stack trace to understand.
        """
        missing = _missing_snapshot(tmp_path)

        result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(missing)])

        assert result.exit_code != 0
        assert str(missing) in result.output
        assert "--input" in result.output
        # Handled as a ClickException (clean exit), not an uncaught OSError.
        assert isinstance(result.exception, SystemExit)

    def test_restore_missing_input_file_in_service_mode_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """The ``--service`` restore path rejects the same bad ``-i`` as cleanly.

        ``restore`` reaches its snapshot through two entry points — the
        single-database path and the service path — and the second one opens a
        service store first. Asserted here in its own right so neither can be
        clean only because the other is.
        """
        missing = _missing_snapshot(tmp_path)

        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(missing), "--service", "svc"],
        )

        assert result.exit_code != 0
        assert str(missing) in result.output
        assert "--input" in result.output
        assert isinstance(result.exception, SystemExit)

    def test_restore_input_directory_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """A ``-i`` path naming a directory is rejected the same clean way.

        Passing the backup *folder* instead of the file inside it fails at the
        same open, with a different ``OSError`` — so it is the same defect
        unless the guard covers the whole class, and it earns the same
        actionable message rather than only a tidy exit code.
        """
        directory = _directory_snapshot(tmp_path)

        result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(directory)])

        assert result.exit_code != 0
        assert str(directory) in result.output
        assert "--input" in result.output
        assert isinstance(result.exception, SystemExit)

    def test_restore_non_utf8_input_is_clean_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        """A ``-i`` path that is not UTF-8 text is a clean CLI error too.

        Pointing ``restore`` at the database instead of at the snapshot opens
        successfully and then fails mid-read while decoding, which is a
        different failure from an unopenable path and needs its own guard.
        """
        binary = _binary_snapshot(tmp_path)

        result = runner.invoke(cli, ["--db", str(db_path), "restore", "-i", str(binary)])

        assert result.exit_code != 0
        assert str(binary) in result.output
        assert "UTF-8" in result.output
        assert "--input" in result.output
        assert isinstance(result.exception, SystemExit)

    @pytest.mark.parametrize(
        "service_args",
        [[], ["--service", "svc"]],
        ids=["single-db", "service"],
    )
    @pytest.mark.parametrize("make_input", _UNUSABLE_SNAPSHOTS)
    def test_restore_unusable_input_raises_the_typed_cli_error(
        self,
        runner: CliRunner,
        db_path: Path,
        tmp_path: Path,
        make_input: Callable[[Path], Path],
        service_args: list[str],
    ) -> None:
        """The guard raises ``ClickException`` itself, on either restore path.

        The tests above assert what the user sees, and Click renders every
        error it handles the same way — a usage error, an explicit ``sys.exit``
        and an abort all reach ``CliRunner`` as ``SystemExit``. Running with
        ``standalone_mode=False`` stops Click catching the exception, so the
        type is pinned here: that is what makes this a typed boundary rather
        than a tidy exit code.
        """
        bad_input = make_input(tmp_path)

        result = runner.invoke(
            cli,
            ["--db", str(db_path), "restore", "-i", str(bad_input), *service_args],
            standalone_mode=False,
        )

        assert isinstance(result.exception, click.ClickException), result.exception
        assert str(bad_input) in str(result.exception)

    def test_restore_missing_input_with_clear_keeps_every_stored_row(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        """``--clear`` must not empty the database when the snapshot is unusable.

        ``--clear`` deletes every core row inside the restore transaction,
        before the snapshot is opened. A failure at the open is therefore only
        harmless because the transaction rolls back, so the rows are re-read
        from SQLite here rather than trusted to the command's exit.
        """
        missing = tmp_path / "no-such-snapshot.jsonl"
        before = _stored_core_ids(populated_db)
        assert all(before.values()), f"corpus precondition failed: {before}"

        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "restore", "-i", str(missing), "--clear"],
        )

        assert _stored_core_ids(populated_db) == before
        # Only once the stored rows are settled does the reported failure matter.
        assert result.exit_code != 0
        assert isinstance(result.exception, SystemExit)


class TestGc:
    """Tests for ``engrava gc``."""

    def test_gc_removes_archived(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "gc"])
        assert result.exit_code == 0
        assert "Collected 1" in result.output

        # Verify only 2 remain
        check = runner.invoke(
            cli,
            ["--db", str(populated_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 2

    def test_gc_dry_run(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "gc", "--dry-run"])
        assert result.exit_code == 0
        # The dry run is read immediately before the destructive run, so it must
        # name every row category the real run deletes, not just the thoughts
        # its count covers.
        assert "Would delete 1 archived thoughts" in result.output
        assert "edges, embeddings, and actions" in result.output
        assert "orphaned edges" not in result.output

        # Verify nothing actually deleted
        check = runner.invoke(
            cli,
            ["--db", str(populated_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 3

    def test_gc_help_names_full_blast_radius(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["gc", "--help"])
        assert result.exit_code == 0
        assert "edges, embeddings, and actions" in result.output
        assert "orphaned edges" not in result.output

    def test_gc_nothing_to_collect(self, runner: CliRunner, populated_db: Path) -> None:
        # First gc removes the archived one
        runner.invoke(cli, ["--db", str(populated_db), "gc"])
        # Second gc should find nothing
        result = runner.invoke(cli, ["--db", str(populated_db), "gc"])
        assert result.exit_code == 0
        assert "No archived" in result.output


class TestMigrate:
    """Tests for ``engrava migrate``."""

    def test_migrate_creates_schema(self, runner: CliRunner, tmp_path: Path) -> None:
        new_db = tmp_path / "fresh.db"
        result = runner.invoke(cli, ["--db", str(new_db), "migrate"])
        assert result.exit_code == 0
        assert "Schema up to date" in result.output
        assert new_db.exists()


class TestExport:
    """Tests for ``engrava export``."""

    def test_export_all(self, runner: CliRunner, populated_db: Path, tmp_path: Path) -> None:
        out = tmp_path / "export.json"
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "export", "-o", str(out)],
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["format"] == "engrava-export"
        assert data["version"] == "0.1.0"
        assert len(data["thoughts"]) == 3
        assert len(data["edges"]) == 1

    def test_export_with_status_filter(
        self,
        runner: CliRunner,
        populated_db: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "active.json"
        result = runner.invoke(
            cli,
            ["--db", str(populated_db), "export", "-o", str(out), "--status", "ACTIVE"],
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["thoughts"]) == 2

    def test_export_default_path(self, runner: CliRunner, populated_db: Path) -> None:
        result = runner.invoke(cli, ["--db", str(populated_db), "export"])
        assert result.exit_code == 0
        default_out = populated_db.with_suffix(".export.json")
        assert default_out.exists()


# ------------------------------------------------------------------
# Corrupt-database hang guard
# ------------------------------------------------------------------
#
# ``sqlite3.DatabaseError: file is not a database`` raised from the first
# ``PRAGMA`` against a corrupt/truncated file used to leave the aiosqlite
# connection open. aiosqlite's connection worker thread is not a daemon and
# stops only when ``Connection.close()`` sends it the shutdown sentinel, so a
# leaked connection blocks ``threading._shutdown`` and the process never
# exits — it prints a traceback (or nothing, depending on buffering) and then
# hangs forever rather than returning any exit code.
#
# The hang happens at interpreter shutdown, which an in-process
# ``CliRunner`` invocation never reaches, so these run the real CLI as a
# subprocess with a hard wall-clock timeout: a regression here fails these
# tests promptly instead of wedging the whole suite.

# Long enough that a fixed, promptly-erroring command never gets close on a
# loaded CI host; far short of "wedge the test worker" if the fix regresses.
_CLI_SUBPROCESS_TIMEOUT_S = 20.0

# A command that returns fast enough for a live process to still be a
# meaningful "promptly" — as opposed to merely "before the hard cap fired".
_PROMPT_CEILING_S = 10.0


def _run_python_subprocess(argv: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    """Run ``python *argv*`` as a real, separate process with a hard timeout.

    Uses this test file's own ``src`` on ``PYTHONPATH`` rather than whatever
    ``sys.path`` the test runner happened to start with, so the subprocess
    always exercises the same worktree's code the test itself was collected
    from — a shared, editable-installed ``engrava`` elsewhere on the host
    must not shadow it.

    Args:
        argv: Full argv after the interpreter, e.g.
            ``["-m", "engrava.cli.main", "--db", str(db_path), "info"]`` or
            ``["-c", some_source, "--db", str(db_path), "info"]``.

    Returns:
        The completed process and the wall-clock seconds it took.

    Raises:
        Failed test: via ``pytest.fail``, if the process does not exit
            within :data:`_CLI_SUBPROCESS_TIMEOUT_S` — the hang this guards
            against — instead of letting ``subprocess.TimeoutExpired``
            propagate as an error or, worse, blocking forever.

    """
    repo_src = str(Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": repo_src}
    start = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, our own source
            [sys.executable, *argv],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=_CLI_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"`python {' '.join(argv)}` did not exit within "
            f"{_CLI_SUBPROCESS_TIMEOUT_S:.0f}s — this is the hang this test "
            "guards against, not a slow environment."
        )
    return completed, time.monotonic() - start


def _run_engrava_subprocess(args: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    """Run ``python -m engrava.cli.main *args*`` as a real, separate process.

    Args:
        args: Full CLI argv after the interpreter and ``-m`` module name,
            e.g. ``["--db", str(db_path), "info"]``.

    Returns:
        The completed process and the wall-clock seconds it took.

    """
    return _run_python_subprocess(["-m", "engrava.cli.main", *args])


def _assert_completed_fails_fast(
    completed: subprocess.CompletedProcess[str], elapsed: float
) -> None:
    """Assert a completed subprocess exited non-zero, promptly, not via the hang."""
    assert completed.returncode != 0, (
        f"expected a non-zero exit, got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    assert completed.returncode != 124, "124 is the timeout(1) sentinel — this hung"
    assert elapsed < _PROMPT_CEILING_S, (
        f"took {elapsed:.1f}s — expected a prompt failure, not one that only "
        "beat the hard subprocess timeout"
    )


def _assert_fails_fast_and_names_the_problem(args: list[str]) -> None:
    """Assert *args* exits non-zero, promptly, with the failure named."""
    completed, elapsed = _run_engrava_subprocess(args)
    _assert_completed_fails_fast(completed, elapsed)
    combined = (completed.stdout + completed.stderr).lower()
    assert "database" in combined, (
        f"expected the failure to name the database problem\nstdout={completed.stdout!r}\n"
        f"stderr={completed.stderr!r}"
    )


def _assert_succeeds(args: list[str]) -> None:
    """Known-good control: the same command against a valid database still works."""
    completed, _elapsed = _run_engrava_subprocess(args)
    assert completed.returncode == 0, (
        f"expected exit 0, got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )


@pytest.fixture
def corrupt_db(tmp_path: Path) -> Path:
    """A file named like a database that is not one — text, truncated, whatever.

    Reproduces the reported shape: a plain text file where a SQLite database
    is expected, which opens fine (the file exists) but fails the first real
    read against it.
    """
    path = tmp_path / "corrupt.sqlite"
    path.write_text("this is not a sqlite database, just some text\n", encoding="utf-8")
    return path


@pytest.fixture
def valid_snapshot(runner: CliRunner, populated_db: Path, tmp_path: Path) -> Path:
    """A real JSONL snapshot of ``populated_db`` — a valid ``restore`` input."""
    snap = tmp_path / "valid-snapshot.jsonl"
    result = runner.invoke(cli, ["--db", str(populated_db), "snapshot", "-o", str(snap)])
    assert result.exit_code == 0
    return snap


class TestCorruptDatabaseExitsInsteadOfHanging:
    """Every built-in command that opens the database file must fail fast.

    One test per affected command, each with the known-good control the
    acceptance criteria require: the same command against ``populated_db``
    (or an equivalent valid target) still works. Covers all eight built-in
    commands found to route through a connection-opening call in
    ``cli/main.py`` — the originally reported four (``info``, ``gc``,
    ``migrate``, ``snapshot``) plus ``verify``, ``query``, ``export``, and
    the single-database branch of ``restore``, which shared the exact same
    unclosed-connection mechanism. The two commands that only ever go
    through :class:`~engrava.infrastructure.service_manager.EngravaManager`
    (the ``--service`` branch of ``snapshot``/``restore``) are not covered
    here — that path already closes on its own error, independently of this
    fix.
    """

    def test_info(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "info"])
        _assert_succeeds(["--db", str(populated_db), "info"])

    def test_verify(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "verify"])
        _assert_succeeds(["--db", str(populated_db), "verify"])

    def test_query(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "query", "COUNT thoughts"]
        )
        _assert_succeeds(["--db", str(populated_db), "query", "COUNT thoughts"])

    def test_gc(self, corrupt_db: Path, populated_db: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "gc", "--dry-run"])
        _assert_succeeds(["--db", str(populated_db), "gc", "--dry-run"])

    def test_migrate(self, corrupt_db: Path, tmp_path: Path) -> None:
        _assert_fails_fast_and_names_the_problem(["--db", str(corrupt_db), "migrate"])
        # migrate is the one built-in whose target need not exist yet — a
        # fresh path is its own "known good" control (see
        # TestMigrate.test_migrate_creates_schema above).
        _assert_succeeds(["--db", str(tmp_path / "fresh-for-migrate.db"), "migrate"])

    def test_snapshot(self, corrupt_db: Path, populated_db: Path, tmp_path: Path) -> None:
        bad_out = tmp_path / "bad.snapshot.jsonl"
        good_out = tmp_path / "good.snapshot.jsonl"
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "snapshot", "-o", str(bad_out)]
        )
        _assert_succeeds(["--db", str(populated_db), "snapshot", "-o", str(good_out)])

    def test_export(self, corrupt_db: Path, populated_db: Path, tmp_path: Path) -> None:
        bad_out = tmp_path / "bad.export.json"
        good_out = tmp_path / "good.export.json"
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "export", "-o", str(bad_out)]
        )
        _assert_succeeds(["--db", str(populated_db), "export", "-o", str(good_out)])

    def test_restore(self, corrupt_db: Path, valid_snapshot: Path, tmp_path: Path) -> None:
        _assert_fails_fast_and_names_the_problem(
            ["--db", str(corrupt_db), "restore", "-i", str(valid_snapshot)]
        )
        good_target = tmp_path / "restored-for-control.db"
        _assert_succeeds(["--db", str(good_target), "restore", "-i", str(valid_snapshot)])


# A failure injected into ``SqliteEngravaCore.ensure_schema`` before ``cli()``
# ever runs. Unlike ``corrupt_db``, this never touches ``_open_db`` at all —
# ``aiosqlite.connect()`` against a target that does not exist yet always
# succeeds, and the file it creates is a valid, empty SQLite database, so
# every ``PRAGMA`` in ``_open_db`` passes. The only way to fail *after* the
# connection is open and *before* the block that is supposed to protect it
# is to fail inside whatever runs in between — here, ``ensure_schema()``.
_ENSURE_SCHEMA_FAULT_INJECTION = """
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore


async def _boom(self, *args, **kwargs):
    raise RuntimeError("injected ensure_schema failure for a bootstrap-window test")


SqliteEngravaCore.ensure_schema = _boom

from engrava.cli.main import cli

cli()
"""


class TestRestoreBootstrapWindowClosesOnFailure:
    """A failure inside ``ensure_schema()`` -- not inside ``_open_db`` -- must
    also close the connection promptly.

    ``restore`` against a target that does not pre-exist calls
    ``store = SqliteEngravaCore(conn)`` and then ``await
    store.ensure_schema()`` to bootstrap it. Before ``_opened_db`` wrapped
    the whole body in one step, both of those ran ahead of the ``try`` that
    was supposed to guarantee the close, so a failure there leaked the
    connection exactly like the corrupt-file case -- just through a
    different call, with a target file that is itself perfectly valid. The
    ``corrupt_db`` tests above are blind to this: they all fail inside
    ``_open_db``, which was already closing correctly before this round of
    fixes even started.
    """

    def test_restore_bootstrap_failure_is_not_a_hang(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh-target-that-fails-to-bootstrap.db"
        assert not target.exists()  # pre_existing must be False to reach ensure_schema()
        missing_input = tmp_path / "never-read.jsonl"  # ensure_schema() fails first

        completed, elapsed = _run_python_subprocess(
            [
                "-c",
                _ENSURE_SCHEMA_FAULT_INJECTION,
                "--db",
                str(target),
                "restore",
                "-i",
                str(missing_input),
            ]
        )
        _assert_completed_fails_fast(completed, elapsed)
        combined = completed.stdout + completed.stderr
        assert "injected ensure_schema failure" in combined, (
            f"expected the injected failure to surface, not something else\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )


# A failure injected into ``aiosqlite.Connection.close()`` itself, on a
# command that otherwise succeeds against a perfectly valid database. Proves
# ``_opened_db``'s success path does not route a close failure through
# ``_close_quietly`` -- which logs and swallows by design, correct only when
# an exception is already in flight.
_CLOSE_FAILS_ON_SUCCESS_INJECTION = """
import aiosqlite

_real_close = aiosqlite.Connection.close


async def _close_blows_up(self):
    await _real_close(self)
    raise RuntimeError("injected close failure on the success path")


aiosqlite.Connection.close = _close_blows_up

from engrava.cli.main import cli

cli()
"""


class TestSuccessPathCloseFailureIsNotSwallowed:
    """A close() failure on an otherwise-successful command must surface.

    ``_opened_db``'s cleanup used to be an unconditional
    ``finally: await _close_quietly(conn)`` -- including on the success
    path, where a close failure is not secondary to anything, it is the
    only error there is. ``_close_quietly`` logs and swallows by design
    (correct for the exception-in-flight case), so the unconditional call
    turned a genuine close failure into a command that printed its normal
    success output and exited 0 -- a worse outcome than the original hang,
    which was at least visible.
    """

    def test_close_failure_after_a_successful_command_is_not_silent(
        self, populated_db: Path
    ) -> None:
        completed, elapsed = _run_python_subprocess(
            [
                "-c",
                _CLOSE_FAILS_ON_SUCCESS_INJECTION,
                "--db",
                str(populated_db),
                "info",
            ]
        )
        _assert_completed_fails_fast(completed, elapsed)
        combined = completed.stdout + completed.stderr
        assert "injected close failure" in combined, (
            f"expected the close failure to surface, not be swallowed\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )

    def test_close_success_control_still_exits_zero(self, populated_db: Path) -> None:
        """Known-good control: an ordinary close on the success path still exits 0."""
        _assert_succeeds(["--db", str(populated_db), "info"])


class _FormatThatComparesAsAnother(str):
    """Reads as its real text; hashes and compares as ``json``."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash("json")

    def __eq__(self, other: object) -> bool:
        return other == "json"

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class TestCliConfigKeepsWhatItResolved:
    """The CLI config is a public entry point reached straight from the command line.

    ``output_format`` is admitted by a membership test and then selects a
    renderer by equality — both run the value's own methods — and the paths are
    built by joining text that a subclass could render differently from the
    text that was checked.
    """

    def test_the_resolved_format_is_an_exact_str(self) -> None:
        config = EngravaCLIConfig.resolve(output_format=_FormatThatComparesAsAnother("table"))
        assert type(config.output_format) is str
        assert config.output_format == "table"

    def test_an_unrecognised_format_still_falls_back_to_table(self) -> None:
        assert EngravaCLIConfig.resolve(output_format="nonsense").output_format == "table"

    def test_every_legitimate_format_survives(self) -> None:
        for fmt in ("json", "table", "csv"):
            assert EngravaCLIConfig.resolve(output_format=fmt).output_format == fmt

    def test_the_resolved_path_is_built_from_owned_text(self, tmp_path: Path) -> None:
        class _PathTextThatRendersDifferently(str):
            __slots__ = ()

            def __str__(self) -> str:
                return "/etc/escaped.db"

        declared = str(tmp_path / "real.db")
        config = EngravaCLIConfig.resolve(db_path=_PathTextThatRendersDifferently(declared))
        assert str(config.db_path) == declared

    def test_direct_construction_owns_its_fields(self, tmp_path: Path) -> None:
        config = EngravaCLIConfig(
            db_path=tmp_path / "t.db",
            output_format=_FormatThatComparesAsAnother("csv"),  # type: ignore[arg-type]  # a str subclass is what is under test
        )
        assert type(config.output_format) is str
        assert config.output_format == "csv"


class TestCliConfigChoosesTheSourceItWasGiven:
    """An explicitly supplied path is used, whatever the value says about itself.

    The resolution order is written with ``or``, which asks each candidate
    whether it is truthy — a method the value may define. A string subclass
    that answers ``False`` erases the ``--db`` the user typed and silently
    substitutes the environment's value or the built-in default.
    """

    def test_an_explicit_path_is_not_suppressed_by_the_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        class _PathThatDeniesItself(str):
            __slots__ = ()

            def __bool__(self) -> bool:
                return False

        monkeypatch.setenv("ENGRAVA_DB", str(tmp_path / "from-env.db"))
        supplied = str(tmp_path / "explicit.db")

        config = EngravaCLIConfig.resolve(db_path=_PathThatDeniesItself(supplied))

        assert str(config.db_path) == supplied

    def test_an_explicit_config_path_is_not_suppressed_by_the_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        class _PathThatDeniesItself(str):
            __slots__ = ()

            def __bool__(self) -> bool:
                return False

        monkeypatch.setenv("ENGRAVA_CONFIG", str(tmp_path / "from-env.yaml"))
        supplied = str(tmp_path / "explicit.yaml")

        config = EngravaCLIConfig.resolve(config_path=_PathThatDeniesItself(supplied))

        assert config.config_path is not None
        assert str(config.config_path) == supplied

    def test_the_environment_is_still_used_when_nothing_is_supplied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("ENGRAVA_DB", str(tmp_path / "from-env.db"))
        assert str(EngravaCLIConfig.resolve().db_path) == str(tmp_path / "from-env.db")

    def test_an_empty_string_still_falls_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Genuine emptiness keeps its old meaning; only the lie is closed."""
        monkeypatch.setenv("ENGRAVA_DB", str(tmp_path / "from-env.db"))
        assert str(EngravaCLIConfig.resolve(db_path="").db_path) == str(tmp_path / "from-env.db")

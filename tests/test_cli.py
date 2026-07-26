"""CLI smoke tests for ``engrava`` command.

Tests all subcommands against an in-memory (temp file) SQLite database.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from click.testing import CliRunner

from engrava.cli.config import EngravaCLIConfig
from engrava.cli.main import cli


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
        assert "Would delete 1" in result.output

        # Verify nothing actually deleted
        check = runner.invoke(
            cli,
            ["--db", str(populated_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 3

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

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
            raise AssertionError("CLI extension discovery must stay disabled")

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
            raise AssertionError("CLI extension discovery must stay disabled")

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
            raise AssertionError("MindQL extension discovery must stay disabled")

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

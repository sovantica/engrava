"""Journal integrity verification exposure tests.

Covers the store-level convenience, the ``engrava verify`` CLI command, and
the opt-in on-open integrity gate that wrap the existing hash-chain walk
(``JournalWriter.verify_integrity``):

- ``SqliteEngravaCore.verify_journal()`` (clean chain, tamper detection,
  empty chain, and the journal-disabled-but-entries-exist path)
- ``engrava verify`` CLI (exit codes + messages, text and JSON)
- ``journal.verify_on_open`` flag (fail-closed on a broken chain; no walk when
  off) and its config parsing
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from click.testing import CliRunner

from engrava import (
    JournalConfig,
    JournalIntegrityError,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.cli.main import cli
from engrava.config import ConfigError, _parse_journal

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thought(thought_id: str = "t-001", essence: str = "Test thought") -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence=essence,
        content="Full content",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


async def _open(db_path: Path, *, journal_enabled: bool) -> SqliteEngravaCore:
    """Open a fresh connection on ``db_path`` and build a store on it."""
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, journal_enabled=journal_enabled)
    await store.ensure_schema()
    return store


async def _seed_chain(db_path: Path) -> None:
    """Write a well-formed, multi-entry chain to ``db_path`` and close."""
    store = await _open(db_path, journal_enabled=True)
    try:
        await store.create_thought(_make_thought("t-001"))
        await store.update_thought("t-001", essence="Updated essence")
        await store.create_thought(_make_thought("t-002"))
        await store.delete_thought("t-002")
        await store._db.commit()
    finally:
        await store._db.close()


async def _tamper(db_path: Path, sequence_number: int) -> None:
    """Mutate one persisted ``journal_entry`` row in place (no rehash)."""
    conn = await aiosqlite.connect(str(db_path))
    try:
        await conn.execute(
            "UPDATE journal_entry SET delta = ? WHERE sequence_number = ?",
            (json.dumps({"before": None, "after": {"tampered": True}}), sequence_number),
        )
        await conn.commit()
    finally:
        await conn.close()


def _write_config(
    tmp_path: Path,
    db_path: Path,
    *,
    enabled: bool,
    verify_on_open: bool,
) -> Path:
    cfg_file = tmp_path / "engrava.yaml"
    cfg_file.write_text(
        "database:\n"
        f"  path: {db_path}\n"
        "journal:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        f"  verify_on_open: {'true' if verify_on_open else 'false'}\n",
        encoding="utf-8",
    )
    return cfg_file


# ---------------------------------------------------------------------------
# store.verify_journal() — clean, tampered, empty
# ---------------------------------------------------------------------------


class TestVerifyJournal:
    """SqliteEngravaCore.verify_journal() reuses the chain walk."""

    async def test_clean_chain_is_valid(self, tmp_path: Path) -> None:
        db_path = tmp_path / "clean.db"
        await _seed_chain(db_path)

        store = await _open(db_path, journal_enabled=True)
        try:
            result = await store.verify_journal()
            assert result.valid is True
            assert result.entries_checked == 4
            assert result.first_invalid_sequence is None
        finally:
            await store._db.close()

    async def test_tamper_detected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tampered.db"
        await _seed_chain(db_path)
        await _tamper(db_path, sequence_number=2)

        store = await _open(db_path, journal_enabled=True)
        try:
            result = await store.verify_journal()
            assert result.valid is False
            assert result.first_invalid_sequence == 2
            assert result.error_message is not None
        finally:
            await store._db.close()

    async def test_empty_journal_is_valid(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        store = await _open(db_path, journal_enabled=True)
        try:
            result = await store.verify_journal()
            assert result.valid is True
            assert result.entries_checked == 0
        finally:
            await store._db.close()

    async def test_no_journal_table_writes_is_valid(self, tmp_path: Path) -> None:
        """A store that never recorded anything still verifies (empty chain)."""
        db_path = tmp_path / "never.db"
        store = await _open(db_path, journal_enabled=False)
        try:
            await store.create_thought(_make_thought())
            await store._db.commit()
            result = await store.verify_journal()
            assert result.valid is True
            assert result.entries_checked == 0
        finally:
            await store._db.close()


# ---------------------------------------------------------------------------
# verify_journal() when journaling is currently disabled but entries exist
# ---------------------------------------------------------------------------


class TestVerifyJournalDisabledButEntriesExist:
    """Recorded entries stay auditable after journaling is turned off."""

    async def test_clean_chain_verifies_with_journal_off(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen-clean.db"
        await _seed_chain(db_path)

        # Reopen with journaling DISABLED — store.journal is None.
        store = await _open(db_path, journal_enabled=False)
        try:
            assert store.journal is None
            result = await store.verify_journal()
            assert result.valid is True
            assert result.entries_checked == 4
        finally:
            await store._db.close()

    async def test_tamper_detected_with_journal_off(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen-tampered.db"
        await _seed_chain(db_path)
        await _tamper(db_path, sequence_number=3)

        store = await _open(db_path, journal_enabled=False)
        try:
            assert store.journal is None
            result = await store.verify_journal()
            assert result.valid is False
            assert result.first_invalid_sequence == 3
        finally:
            await store._db.close()


# ---------------------------------------------------------------------------
# engrava verify CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


class TestVerifyCli:
    """``engrava verify`` command.

    These tests are synchronous because ``CliRunner.invoke`` drives the
    command through ``asyncio.run`` internally; the chain is therefore seeded
    with an explicit ``asyncio.run`` rather than an ``await``.
    """

    def test_clean_chain_exit_zero(self, runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "cli-clean.db"
        asyncio.run(_seed_chain(db_path))

        result = runner.invoke(cli, ["--db", str(db_path), "verify"])
        assert result.exit_code == 0
        assert "Journal integrity OK" in result.output
        assert "4 entries" in result.output

    def test_tampered_chain_exit_one(self, runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "cli-tampered.db"
        asyncio.run(_seed_chain(db_path))
        asyncio.run(_tamper(db_path, sequence_number=2))

        result = runner.invoke(cli, ["--db", str(db_path), "verify"])
        assert result.exit_code == 1
        assert "FAILED" in result.output
        assert "sequence 2" in result.output

    def test_json_output(self, runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "cli-json.db"
        asyncio.run(_seed_chain(db_path))

        result = runner.invoke(cli, ["--db", str(db_path), "--format", "json", "verify"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["valid"] is True
        assert payload["entries_checked"] == 4

    def test_json_output_tampered(self, runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "cli-json-bad.db"
        asyncio.run(_seed_chain(db_path))
        asyncio.run(_tamper(db_path, sequence_number=1))

        result = runner.invoke(cli, ["--db", str(db_path), "--format", "json", "verify"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["valid"] is False
        assert payload["first_invalid_sequence"] == 1

    def test_missing_db_exit_one(self, runner: CliRunner, tmp_path: Path) -> None:
        missing = tmp_path / "nope.db"
        result = runner.invoke(cli, ["--db", str(missing), "verify"])
        assert result.exit_code != 0
        assert "Database not found" in result.output


# ---------------------------------------------------------------------------
# on-open integrity gate (journal.verify_on_open)
# ---------------------------------------------------------------------------


class TestVerifyOnOpen:
    """``journal.verify_on_open`` fail-closed startup gate."""

    async def test_flag_on_tampered_chain_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "onopen-bad.db"
        await _seed_chain(db_path)
        await _tamper(db_path, sequence_number=2)

        cfg_file = _write_config(tmp_path, db_path, enabled=True, verify_on_open=True)

        with pytest.raises(JournalIntegrityError) as excinfo:
            await SqliteEngravaCore.from_config(cfg_file)
        assert excinfo.value.first_invalid_sequence == 2
        assert excinfo.value.error_message is not None

    async def test_flag_on_clean_chain_opens(self, tmp_path: Path) -> None:
        db_path = tmp_path / "onopen-clean.db"
        await _seed_chain(db_path)

        cfg_file = _write_config(tmp_path, db_path, enabled=True, verify_on_open=True)

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            fetched = await store.get_thought("t-001")
            assert fetched is not None

    async def test_flag_off_does_not_raise_on_tampered_chain(self, tmp_path: Path) -> None:
        """Default (off) opens fine even on a broken chain — no walk runs."""
        db_path = tmp_path / "onopen-off.db"
        await _seed_chain(db_path)
        await _tamper(db_path, sequence_number=2)

        cfg_file = _write_config(tmp_path, db_path, enabled=True, verify_on_open=False)

        # Opens without raising despite the tampered chain.
        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            # A subsequent explicit verification still detects the break.
            result = await store.verify_journal()
            assert result.valid is False
            assert result.first_invalid_sequence == 2

    async def test_flag_on_verifies_even_when_journaling_disabled(self, tmp_path: Path) -> None:
        """verify_on_open is independent of enabled — checks the on-disk chain."""
        db_path = tmp_path / "onopen-disabled.db"
        await _seed_chain(db_path)
        await _tamper(db_path, sequence_number=1)

        cfg_file = _write_config(tmp_path, db_path, enabled=False, verify_on_open=True)

        with pytest.raises(JournalIntegrityError):
            await SqliteEngravaCore.from_config(cfg_file)


# ---------------------------------------------------------------------------
# Config parsing for verify_on_open
# ---------------------------------------------------------------------------


class TestVerifyOnOpenConfig:
    """JournalConfig.verify_on_open parsing."""

    def test_default_false(self) -> None:
        assert JournalConfig().verify_on_open is False

    def test_parse_absent_defaults_false(self) -> None:
        cfg = _parse_journal({"enabled": True})
        assert cfg.verify_on_open is False

    def test_parse_true(self) -> None:
        cfg = _parse_journal({"enabled": True, "verify_on_open": True})
        assert cfg.enabled is True
        assert cfg.verify_on_open is True

    def test_parse_false_explicit(self) -> None:
        cfg = _parse_journal({"verify_on_open": False})
        assert cfg.verify_on_open is False

    def test_parse_non_bool_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"verify_on_open.*must be a boolean"):
            _parse_journal({"verify_on_open": "yes"})

    def test_frozen(self) -> None:
        cfg = JournalConfig(verify_on_open=True)
        with pytest.raises(AttributeError):
            cfg.verify_on_open = False  # type: ignore[misc]

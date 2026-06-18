"""CognitiveJournal (hash-chain audit log) tests.

Covers:
- JournalEntry / JournalIntegrityResult value objects
- MutationType enum
- JournalWriter (append, verify_integrity, get_entries)
- Hash-chain correctness (SHA-256)
- SqliteEngravaCore integration (create/update/delete thought + edge)
- Config parsing (journal section)
- Edge cases (empty journal, journal disabled, concurrent appends)
- Schema migration core-5 → core-6
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    EdgeRecord,
    EdgeType,
    JournalConfig,
    JournalEntry,
    JournalIntegrityResult,
    JournalWriter,
    KnowledgeSource,
    LifecycleStatus,
    MutationType,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.config import ConfigError, _parse_journal
from engrava.domain.protocols.hooks import DefaultEngravaHooks

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite database with core-6 schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn, journal_enabled=True)
    await store.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with journal enabled."""
    s = SqliteEngravaCore(db, journal_enabled=True)
    await s._probe_fts()
    return s


@pytest.fixture
async def store_no_journal(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with journal disabled (default)."""
    s = SqliteEngravaCore(db, journal_enabled=False)
    await s._probe_fts()
    return s


@pytest.fixture
async def writer(db: aiosqlite.Connection) -> JournalWriter:
    """Standalone JournalWriter for unit tests."""
    return JournalWriter(db)


def _make_thought(
    thought_id: str = "t-001",
    thought_type: ThoughtType = ThoughtType.TASK,
    essence: str = "Test thought",
    content: str = "Full content",
    priority: Priority = Priority.P2,
    lifecycle_status: LifecycleStatus = LifecycleStatus.CREATED,
    created_cycle: int = 0,
    updated_cycle: int = 0,
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


def _make_edge(
    edge_id: str = "e-001",
    from_id: str = "t-001",
    to_id: str = "t-002",
) -> EdgeRecord:
    return EdgeRecord(
        edge_id=edge_id,
        from_thought_id=from_id,
        to_thought_id=to_id,
        edge_type=EdgeType.ASSOCIATED,
        weight=0.5,
        created_cycle=0,
        source=KnowledgeSource.EXPERIENCE,
    )


# ---------------------------------------------------------------------------
# MutationType enum
# ---------------------------------------------------------------------------


class TestMutationType:
    """MutationType StrEnum tests."""

    def test_all_six_values(self) -> None:
        assert len(MutationType) == 6

    def test_values_are_strings(self) -> None:
        for mt in MutationType:
            assert isinstance(mt, str)
            assert mt == mt.value

    def test_construction_from_string(self) -> None:
        assert MutationType("INSERT_THOUGHT") is MutationType.INSERT_THOUGHT


# ---------------------------------------------------------------------------
# JournalEntry value object
# ---------------------------------------------------------------------------


class TestJournalEntry:
    """JournalEntry frozen dataclass tests."""

    def test_frozen(self) -> None:
        entry = JournalEntry(
            entry_id="e-1",
            sequence_number=1,
            mutation_type="INSERT_THOUGHT",
            target_id="t-001",
            delta={"before": None, "after": {}},
            parent_hash=None,
            entry_hash="abc",
            created_at="2026-01-01T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            entry.entry_id = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        entry = JournalEntry(
            entry_id="e-1",
            sequence_number=1,
            mutation_type="INSERT_THOUGHT",
            target_id=None,
            delta={"before": None, "after": {}},
            parent_hash=None,
            entry_hash="abc",
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert entry.sequence_number == 1
        assert entry.target_id is None
        assert entry.parent_hash is None


# ---------------------------------------------------------------------------
# JournalIntegrityResult value object
# ---------------------------------------------------------------------------


class TestJournalIntegrityResult:
    """JournalIntegrityResult tests."""

    def test_valid_result(self) -> None:
        result = JournalIntegrityResult(valid=True, entries_checked=10)
        assert result.valid is True
        assert result.first_invalid_sequence is None
        assert result.error_message is None

    def test_invalid_result(self) -> None:
        result = JournalIntegrityResult(
            valid=False,
            entries_checked=5,
            first_invalid_sequence=3,
            error_message="Hash mismatch",
        )
        assert result.valid is False
        assert result.first_invalid_sequence == 3


# ---------------------------------------------------------------------------
# JournalWriter — hash computation
# ---------------------------------------------------------------------------


class TestJournalWriterHash:
    """JournalWriter.compute_hash() unit tests."""

    def test_deterministic(self) -> None:
        h1 = JournalWriter.compute_hash(1, "INSERT_THOUGHT", "t-001", {"a": 1}, None)
        h2 = JournalWriter.compute_hash(1, "INSERT_THOUGHT", "t-001", {"a": 1}, None)
        assert h1 == h2

    def test_different_sequence_different_hash(self) -> None:
        h1 = JournalWriter.compute_hash(1, "INSERT_THOUGHT", "t-001", {}, None)
        h2 = JournalWriter.compute_hash(2, "INSERT_THOUGHT", "t-001", {}, None)
        assert h1 != h2

    def test_different_type_different_hash(self) -> None:
        h1 = JournalWriter.compute_hash(1, "INSERT_THOUGHT", "t-001", {}, None)
        h2 = JournalWriter.compute_hash(1, "DELETE_THOUGHT", "t-001", {}, None)
        assert h1 != h2

    def test_none_target_id(self) -> None:
        h = JournalWriter.compute_hash(1, "INSERT_THOUGHT", None, {}, None)
        canonical = "1|INSERT_THOUGHT||{}|"
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert h == expected

    def test_sort_keys_in_delta(self) -> None:
        delta = {"z": 1, "a": 2}
        h = JournalWriter.compute_hash(1, "INSERT_THOUGHT", "t-1", delta, None)
        canonical = f"1|INSERT_THOUGHT|t-1|{json.dumps(delta, sort_keys=True)}|"
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert h == expected

    def test_parent_hash_included(self) -> None:
        parent = "abc123"
        h1 = JournalWriter.compute_hash(2, "UPDATE_THOUGHT", "t-1", {}, parent)
        h2 = JournalWriter.compute_hash(2, "UPDATE_THOUGHT", "t-1", {}, None)
        assert h1 != h2


# ---------------------------------------------------------------------------
# JournalWriter — append + chain
# ---------------------------------------------------------------------------


class TestJournalWriterAppend:
    """JournalWriter.append() tests."""

    async def test_first_entry_has_no_parent(self, writer: JournalWriter) -> None:
        entry = await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        assert entry.sequence_number == 1
        assert entry.parent_hash is None
        assert entry.entry_hash != ""
        assert entry.target_id == "t-001"

    async def test_second_entry_links_to_first(self, writer: JournalWriter) -> None:
        e1 = await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        e2 = await writer.append("UPDATE_THOUGHT", "t-001", {"before": {}, "after": {"x": 1}})
        assert e2.sequence_number == 2
        assert e2.parent_hash == e1.entry_hash

    async def test_sequence_monotonic(self, writer: JournalWriter) -> None:
        entries = []
        for i in range(5):
            e = await writer.append("INSERT_THOUGHT", f"t-{i}", {"before": None, "after": {}})
            entries.append(e)
        seqs = [e.sequence_number for e in entries]
        assert seqs == [1, 2, 3, 4, 5]

    async def test_entry_hash_matches_recomputed(self, writer: JournalWriter) -> None:
        entry = await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {"a": 1}})
        recomputed = JournalWriter.compute_hash(
            entry.sequence_number,
            entry.mutation_type,
            entry.target_id,
            entry.delta,
            entry.parent_hash,
        )
        assert entry.entry_hash == recomputed

    async def test_chain_hash_integrity(self, writer: JournalWriter) -> None:
        """Build a 3-entry chain and verify each link."""
        e1 = await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        e2 = await writer.append("UPDATE_THOUGHT", "t-001", {"before": {}, "after": {"x": 1}})
        e3 = await writer.append("DELETE_THOUGHT", "t-001", {"before": {"x": 1}, "after": None})

        assert e1.parent_hash is None
        assert e2.parent_hash == e1.entry_hash
        assert e3.parent_hash == e2.entry_hash


# ---------------------------------------------------------------------------
# JournalWriter — verify_integrity
# ---------------------------------------------------------------------------


class TestJournalWriterVerify:
    """JournalWriter.verify_integrity() tests."""

    async def test_empty_journal_is_valid(self, writer: JournalWriter) -> None:
        result = await writer.verify_integrity()
        assert result.valid is True
        assert result.entries_checked == 0

    async def test_single_entry_valid(self, writer: JournalWriter) -> None:
        await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        result = await writer.verify_integrity()
        assert result.valid is True
        assert result.entries_checked == 1

    async def test_chain_valid(self, writer: JournalWriter) -> None:
        for i in range(10):
            await writer.append("INSERT_THOUGHT", f"t-{i}", {"before": None, "after": {}})
        result = await writer.verify_integrity()
        assert result.valid is True
        assert result.entries_checked == 10

    async def test_tampered_hash_detected(
        self,
        writer: JournalWriter,
        db: aiosqlite.Connection,
    ) -> None:
        await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        await writer.append("INSERT_THOUGHT", "t-002", {"before": None, "after": {}})

        # Tamper with the first entry's hash.
        await db.execute(
            "UPDATE journal_entry SET entry_hash = 'tampered' WHERE sequence_number = 1"
        )
        await db.commit()

        # Create fresh writer to re-read from DB.
        fresh = JournalWriter(db)
        result = await fresh.verify_integrity()
        assert result.valid is False
        assert result.first_invalid_sequence == 1
        assert "Hash mismatch" in (result.error_message or "")

    async def test_tampered_parent_hash_detected(
        self,
        writer: JournalWriter,
        db: aiosqlite.Connection,
    ) -> None:
        await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        await writer.append("INSERT_THOUGHT", "t-002", {"before": None, "after": {}})

        # Tamper with the second entry's parent_hash.
        await db.execute("UPDATE journal_entry SET parent_hash = 'wrong' WHERE sequence_number = 2")
        await db.commit()

        fresh = JournalWriter(db)
        result = await fresh.verify_integrity()
        assert result.valid is False
        assert result.first_invalid_sequence == 2
        assert "Parent hash mismatch" in (result.error_message or "")


# ---------------------------------------------------------------------------
# JournalWriter — get_entries (query filters)
# ---------------------------------------------------------------------------


class TestJournalWriterQuery:
    """JournalWriter.get_entries() tests."""

    async def test_get_all(self, writer: JournalWriter) -> None:
        for i in range(3):
            await writer.append("INSERT_THOUGHT", f"t-{i}", {"before": None, "after": {}})
        entries = await writer.get_entries()
        assert len(entries) == 3
        assert all(isinstance(e, JournalEntry) for e in entries)

    async def test_filter_by_target_id(self, writer: JournalWriter) -> None:
        await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        await writer.append("INSERT_THOUGHT", "t-002", {"before": None, "after": {}})
        await writer.append("UPDATE_THOUGHT", "t-001", {"before": {}, "after": {"x": 1}})

        entries = await writer.get_entries(target_id="t-001")
        assert len(entries) == 2
        assert all(e.target_id == "t-001" for e in entries)

    async def test_filter_by_mutation_type(self, writer: JournalWriter) -> None:
        await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        await writer.append("UPDATE_THOUGHT", "t-001", {"before": {}, "after": {"x": 1}})
        await writer.append("DELETE_THOUGHT", "t-001", {"before": {"x": 1}, "after": None})

        entries = await writer.get_entries(mutation_type="UPDATE_THOUGHT")
        assert len(entries) == 1
        assert entries[0].mutation_type == "UPDATE_THOUGHT"

    async def test_filter_by_since(self, writer: JournalWriter) -> None:
        await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        entries = await writer.get_entries(since="2020-01-01T00:00:00")
        assert len(entries) == 1

        entries = await writer.get_entries(since="2099-01-01T00:00:00")
        assert len(entries) == 0

    async def test_limit(self, writer: JournalWriter) -> None:
        for i in range(10):
            await writer.append("INSERT_THOUGHT", f"t-{i}", {"before": None, "after": {}})
        entries = await writer.get_entries(limit=3)
        assert len(entries) == 3
        assert entries[0].sequence_number == 1  # ordered ASC

    async def test_combined_filters(self, writer: JournalWriter) -> None:
        await writer.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        await writer.append("INSERT_EDGE", "e-001", {"before": None, "after": {}})
        await writer.append("UPDATE_THOUGHT", "t-001", {"before": {}, "after": {"x": 1}})

        entries = await writer.get_entries(target_id="t-001", mutation_type="INSERT_THOUGHT")
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# JournalWriter — lazy initialization from existing DB
# ---------------------------------------------------------------------------


class TestJournalWriterRestart:
    """Verify that a fresh JournalWriter picks up from existing chain."""

    async def test_resume_after_restart(self, db: aiosqlite.Connection) -> None:
        w1 = JournalWriter(db)
        e1 = await w1.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}})
        await db.commit()

        # Simulate a restart — new writer on same DB.
        w2 = JournalWriter(db)
        e2 = await w2.append("INSERT_THOUGHT", "t-002", {"before": None, "after": {}})
        assert e2.sequence_number == 2
        assert e2.parent_hash == e1.entry_hash

        result = await w2.verify_integrity()
        assert result.valid is True
        assert result.entries_checked == 2

    async def test_concurrent_appends_from_multiple_writers(self, db: aiosqlite.Connection) -> None:
        w1 = JournalWriter(db)
        w2 = JournalWriter(db)

        entries = await asyncio.gather(
            w1.append("INSERT_THOUGHT", "t-001", {"before": None, "after": {}}),
            w2.append("INSERT_THOUGHT", "t-002", {"before": None, "after": {}}),
        )

        assert sorted(entry.sequence_number for entry in entries) == [1, 2]

        fresh = JournalWriter(db)
        result = await fresh.verify_integrity()
        assert result.valid is True
        assert result.entries_checked == 2


# ---------------------------------------------------------------------------
# SqliteEngravaCore integration — journal recording
# ---------------------------------------------------------------------------


class TestStoreJournalIntegration:
    """Verify that CRUD operations record journal entries atomically."""

    async def test_create_thought_records_journal(self, store: SqliteEngravaCore) -> None:
        thought = _make_thought()
        await store.create_thought(thought)

        assert store.journal is not None
        entries = await store.journal.get_entries()
        assert len(entries) == 1
        assert entries[0].mutation_type == "INSERT_THOUGHT"
        assert entries[0].target_id == "t-001"
        assert entries[0].delta["before"] is None
        assert entries[0].delta["after"]["thought_id"] == "t-001"

    async def test_update_thought_records_journal(self, store: SqliteEngravaCore) -> None:
        thought = _make_thought()
        await store.create_thought(thought)
        await store.update_thought("t-001", essence="Updated essence")

        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="UPDATE_THOUGHT")
        assert len(entries) == 1
        assert entries[0].delta["before"]["essence"] == "Test thought"
        assert entries[0].delta["after"]["essence"] == "Updated essence"

    async def test_delete_thought_records_journal(self, store: SqliteEngravaCore) -> None:
        thought = _make_thought()
        await store.create_thought(thought)
        deleted = await store.delete_thought("t-001")
        assert deleted is True

        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="DELETE_THOUGHT")
        assert len(entries) == 1
        assert entries[0].delta["before"]["thought_id"] == "t-001"
        assert entries[0].delta["after"] is None

    async def test_delete_nonexistent_thought_no_journal(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        deleted = await store.delete_thought("nonexistent")
        assert deleted is False

        assert store.journal is not None
        entries = await store.journal.get_entries()
        assert len(entries) == 0

    async def test_create_edge_records_journal(self, store: SqliteEngravaCore) -> None:
        # Create thoughts first.
        await store.create_thought(_make_thought("t-001"))
        await store.create_thought(_make_thought("t-002"))
        edge = _make_edge()
        await store.create_edge(edge)

        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="INSERT_EDGE")
        assert len(entries) == 1
        assert entries[0].target_id == "e-001"

    async def test_delete_edge_records_journal(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        await store.create_thought(_make_thought("t-002"))
        edge = _make_edge()
        await store.create_edge(edge)
        deleted = await store.delete_edge("e-001")
        assert deleted is True

        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="DELETE_EDGE")
        assert len(entries) == 1
        assert entries[0].delta["before"] is not None
        assert entries[0].delta["after"] is None

    async def test_update_edge_records_journal(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_thought("t-001"))
        await store.create_thought(_make_thought("t-002"))
        edge = _make_edge()
        await store.create_edge(edge)

        updated = await store.update_edge("e-001", weight=0.9, decay_multiplier=0.7)
        assert updated.weight == 0.9
        assert updated.decay_multiplier == 0.7

        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="UPDATE_EDGE")
        assert len(entries) == 1
        assert entries[0].delta["before"]["weight"] == 0.5
        assert entries[0].delta["after"]["weight"] == 0.9

    async def test_delete_nonexistent_edge_no_journal(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        deleted = await store.delete_edge("nonexistent")
        assert deleted is False

        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="DELETE_EDGE")
        assert len(entries) == 0

    async def test_full_lifecycle_chain_integrity(self, store: SqliteEngravaCore) -> None:
        """Create, update, delete — verify chain is valid."""
        thought = _make_thought()
        await store.create_thought(thought)
        await store.update_thought("t-001", essence="Updated")
        await store.delete_thought("t-001")

        assert store.journal is not None
        result = await store.journal.verify_integrity()
        assert result.valid is True
        assert result.entries_checked == 3


# ---------------------------------------------------------------------------
# Journal disabled (backward compat)
# ---------------------------------------------------------------------------


class TestJournalDisabled:
    """Verify zero overhead when journal is disabled."""

    async def test_journal_property_is_none(
        self,
        store_no_journal: SqliteEngravaCore,
    ) -> None:
        assert store_no_journal.journal is None

    async def test_crud_works_without_journal(
        self,
        store_no_journal: SqliteEngravaCore,
        db: aiosqlite.Connection,
    ) -> None:
        thought = _make_thought()
        created = await store_no_journal.create_thought(thought)
        assert created.thought_id == "t-001"

        await store_no_journal.update_thought("t-001", essence="Updated")
        deleted = await store_no_journal.delete_thought("t-001")
        assert deleted is True

        # Journal table exists (schema v6) but should be empty.
        cursor = await db.execute("SELECT COUNT(*) FROM journal_entry")
        row = await cursor.fetchone()
        assert row[0] == 0


class MutatingRetrieveHooks(DefaultEngravaHooks):
    """Test hook that mutates loaded thought records."""

    async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
        return thought.evolve(essence="HOOKED")


class TestJournalUsesStoredState:
    """Journal deltas must reflect persisted state, not retrieval hook output."""

    async def test_delete_thought_uses_raw_db_state(self, db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(db, hooks=MutatingRetrieveHooks(), journal_enabled=True)
        await store.ensure_schema()

        await store.create_thought(_make_thought(essence="ORIGINAL"))
        await store.delete_thought("t-001")

        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="DELETE_THOUGHT")
        assert entries[0].delta["before"]["essence"] == "ORIGINAL"

    async def test_update_thought_uses_raw_db_state(self, db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(db, hooks=MutatingRetrieveHooks(), journal_enabled=True)
        await store.ensure_schema()

        await store.create_thought(_make_thought(essence="ORIGINAL"))
        updated = await store.update_thought("t-001", content="updated-content")

        assert updated.essence == "ORIGINAL"
        assert store.journal is not None
        entries = await store.journal.get_entries(mutation_type="UPDATE_THOUGHT")
        assert entries[0].delta["before"]["essence"] == "ORIGINAL"
        assert entries[0].delta["after"]["essence"] == "ORIGINAL"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestJournalConfig:
    """JournalConfig and _parse_journal() tests."""

    def test_default_disabled(self) -> None:
        cfg = JournalConfig()
        assert cfg.enabled is False

    def test_enabled(self) -> None:
        cfg = JournalConfig(enabled=True)
        assert cfg.enabled is True

    def test_frozen(self) -> None:
        cfg = JournalConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]

    def test_parse_none(self) -> None:
        cfg = _parse_journal(None)
        assert cfg.enabled is False

    def test_parse_enabled(self) -> None:
        cfg = _parse_journal({"enabled": True})
        assert cfg.enabled is True

    def test_parse_disabled_explicit(self) -> None:
        cfg = _parse_journal({"enabled": False})
        assert cfg.enabled is False

    def test_parse_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            _parse_journal("invalid")

    def test_parse_invalid_enabled_type(self) -> None:
        with pytest.raises(ConfigError, match="must be a boolean"):
            _parse_journal({"enabled": "yes"})


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """Core-5 → core-6 migration tests."""

    async def test_fresh_schema_has_journal_table(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()

            cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='journal_entry'"
            )
            assert await cursor.fetchone() is not None

            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert row[0] == 14
        finally:
            await conn.close()

    async def test_migrate_from_v5(self) -> None:
        """Simulate a v5 database and ensure migration to v6."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            # Manually create v5 schema (just the pragma + metadata table).
            await conn.execute("PRAGMA user_version = 5")
            await conn.execute(
                "CREATE TABLE thought ("
                "  thought_id TEXT PRIMARY KEY, thought_type TEXT NOT NULL,"
                "  essence TEXT NOT NULL, content TEXT NOT NULL,"
                "  priority TEXT NOT NULL, lifecycle_status TEXT NOT NULL DEFAULT 'CREATED',"
                "  created_cycle INTEGER NOT NULL DEFAULT 0,"
                "  updated_cycle INTEGER NOT NULL DEFAULT 0,"
                "  source TEXT NOT NULL DEFAULT 'human',"
                "  confidence REAL, embedding_ref TEXT,"
                "  source_type TEXT NOT NULL DEFAULT 'EXPERIENCE',"
                "  confirmation_count INTEGER NOT NULL DEFAULT 0,"
                "  consolidated_from TEXT,"
                "  visibility TEXT NOT NULL DEFAULT 'selective',"
                "  access_count INTEGER NOT NULL DEFAULT 0,"
                "  last_accessed_at TEXT, created_at TEXT, updated_at TEXT"
                ")"
            )
            await conn.execute("CREATE TABLE _metadata (key TEXT PRIMARY KEY, value TEXT)")
            await conn.commit()

            store = SqliteEngravaCore(conn)
            await store.ensure_schema()

            # journal_entry table should now exist.
            cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='journal_entry'"
            )
            assert await cursor.fetchone() is not None

            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert row[0] == 14
        finally:
            await conn.close()

    async def test_migration_idempotent(self) -> None:
        """Running ensure_schema twice should not fail."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()
            await store.ensure_schema()

            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert row[0] == 14
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# Edge case: many appends
# ---------------------------------------------------------------------------


class TestJournalStress:
    """Stress / edge-case tests for journal."""

    async def test_many_appends_chain_valid(self, writer: JournalWriter) -> None:
        """100-entry chain should verify cleanly."""
        for i in range(100):
            await writer.append("INSERT_THOUGHT", f"t-{i}", {"before": None, "after": {"n": i}})
        result = await writer.verify_integrity()
        assert result.valid is True
        assert result.entries_checked == 100

    async def test_large_delta(self, writer: JournalWriter) -> None:
        """Verify hashing works with large deltas."""
        big_delta: dict[str, object] = {
            "before": None,
            "after": {"data": "x" * 10_000},
        }
        entry = await writer.append("INSERT_THOUGHT", "t-big", big_delta)
        assert len(entry.entry_hash) == 64  # SHA-256 hex digest

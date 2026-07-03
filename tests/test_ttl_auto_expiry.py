"""TTL / Auto-Expiry tests.

Covers:
- ThoughtRecord.expires_at field (None by default)
- CleanupResult / CleanupStrategy value objects
- TTLConfig parsing from YAML config
- Schema migration core-6 → core-7 (expires_at column + partial index)
- create_thought() with expires_after_seconds (relative TTL)
- create_thought() with default_ttl_seconds (store-level default)
- cleanup_expired() — archive strategy
- cleanup_expired() — delete strategy
- cleanup_expired() — journal integration
- Search exclusion: search_fts, search_similar, list_thoughts
- list_thoughts(include_expired=True) opt-in
- Auto-cleanup cadence (check_every_n_operations)
- ReadOnlyEngrava.cleanup_expired() blocked
- Edge cases: None TTL, past expires_at on create, concurrent cleanup
- Backward compat: no ttl config = identical behavior
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CleanupResult,
    CleanupStrategy,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ReadOnlyEngrava,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    ThoughtVisibility,
    TTLConfig,
)
from engrava.config import ConfigError, _parse_ttl
from engrava.domain.exceptions import ReadOnlyViolationError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """In-memory SQLite database with core-7 schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def store(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with default TTL config (archive, no auto-check)."""
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


@pytest.fixture
async def store_delete(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with delete strategy."""
    s = SqliteEngravaCore(db, ttl_strategy="delete")
    await s._probe_fts()
    return s


@pytest.fixture
async def store_default_ttl(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with a default TTL of 60 seconds."""
    s = SqliteEngravaCore(db, ttl_default_seconds=60)
    await s._probe_fts()
    return s


@pytest.fixture
async def store_auto(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with auto-cleanup every 2 operations."""
    s = SqliteEngravaCore(db, ttl_check_every_n=2)
    await s._probe_fts()
    return s


@pytest.fixture
async def store_journal(db: aiosqlite.Connection) -> SqliteEngravaCore:
    """Store with journaling + archive strategy."""
    s = SqliteEngravaCore(db, journal_enabled=True)
    await s._probe_fts()
    return s


def _make_thought(
    thought_id: str = "t-001",
    expires_at: str | None = None,
    lifecycle_status: LifecycleStatus = LifecycleStatus.CREATED,
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence="Test thought",
        content="Full content",
        priority=Priority.P2,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        expires_at=expires_at,
    )


def _past_ts(seconds: int = 3600) -> str:
    """Return an ISO-8601 timestamp ``seconds`` in the past."""
    return (datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=seconds)).isoformat()


def _future_ts(seconds: int = 3600) -> str:
    """Return an ISO-8601 timestamp ``seconds`` in the future."""
    return (datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=seconds)).isoformat()


# =====================================================================
# 1. Value Objects
# =====================================================================


class TestCleanupStrategy:
    """CleanupStrategy enum tests."""

    def test_values(self) -> None:
        assert CleanupStrategy.ARCHIVE == "archive"
        assert CleanupStrategy.DELETE == "delete"

    def test_from_string(self) -> None:
        assert CleanupStrategy("archive") is CleanupStrategy.ARCHIVE
        assert CleanupStrategy("delete") is CleanupStrategy.DELETE


class TestCleanupResult:
    """CleanupResult frozen dataclass tests."""

    def test_creation(self) -> None:
        result = CleanupResult(
            expired_count=5,
            strategy_applied="archive",
            timestamp="2026-04-12T10:00:00+00:00",
        )
        assert result.expired_count == 5
        assert result.strategy_applied == "archive"
        assert result.timestamp == "2026-04-12T10:00:00+00:00"

    def test_frozen(self) -> None:
        result = CleanupResult(expired_count=0, strategy_applied="delete", timestamp="t")
        with pytest.raises(AttributeError):
            result.expired_count = 1  # type: ignore[misc]


class TestTTLConfig:
    """TTLConfig frozen dataclass tests."""

    def test_defaults(self) -> None:
        cfg = TTLConfig()
        assert cfg.strategy == "archive"
        assert cfg.check_every_n_operations == 0
        assert cfg.default_ttl_seconds is None

    def test_frozen(self) -> None:
        cfg = TTLConfig()
        with pytest.raises(AttributeError):
            cfg.strategy = "delete"  # type: ignore[misc]


# =====================================================================
# 2. ThoughtRecord.expires_at
# =====================================================================


class TestThoughtRecordExpiresAt:
    """ThoughtRecord expires_at field tests."""

    def test_default_none(self) -> None:
        t = _make_thought()
        assert t.expires_at is None

    def test_explicit_value(self) -> None:
        ts = _future_ts()
        t = _make_thought(expires_at=ts)
        assert t.expires_at == ts

    def test_evolve_expires_at(self) -> None:
        t = _make_thought()
        ts = _future_ts()
        t2 = t.evolve(expires_at=ts)
        assert t.expires_at is None
        assert t2.expires_at == ts


# =====================================================================
# 3. Config parsing (_parse_ttl)
# =====================================================================


class TestParseTTL:
    """_parse_ttl config parser tests."""

    def test_none_returns_defaults(self) -> None:
        cfg = _parse_ttl(None)
        assert cfg.strategy == "archive"
        assert cfg.check_every_n_operations == 0
        assert cfg.default_ttl_seconds is None

    def test_empty_dict_returns_defaults(self) -> None:
        cfg = _parse_ttl({})
        assert cfg.strategy == "archive"

    def test_valid_config(self) -> None:
        cfg = _parse_ttl(
            {
                "strategy": "delete",
                "check_every_n_operations": 50,
                "default_ttl_seconds": 3600,
            }
        )
        assert cfg.strategy == "delete"
        assert cfg.check_every_n_operations == 50
        assert cfg.default_ttl_seconds == 3600

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ConfigError, match="strategy"):
            _parse_ttl({"strategy": "purge"})

    def test_negative_check_every(self) -> None:
        with pytest.raises(ConfigError, match="check_every_n_operations"):
            _parse_ttl({"check_every_n_operations": -1})

    def test_zero_default_ttl(self) -> None:
        with pytest.raises(ConfigError, match="default_ttl_seconds"):
            _parse_ttl({"default_ttl_seconds": 0})

    def test_negative_default_ttl(self) -> None:
        with pytest.raises(ConfigError, match="default_ttl_seconds"):
            _parse_ttl({"default_ttl_seconds": -10})


# =====================================================================
# 4. Schema migration
# =====================================================================


class TestSchemaMigration:
    """Schema core-7 migration tests."""

    async def test_fresh_schema_version_current(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 17

    async def test_expires_at_column_exists(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(thought)")
        cols = [r["name"] for r in await cursor.fetchall()]
        assert "expires_at" in cols

    async def test_partial_index_exists(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_thought_expires'"
        )
        row = await cursor.fetchone()
        assert row is not None

    async def test_migration_from_v6_idempotent(self) -> None:
        """Migrating from core-6 to core-7 should be idempotent."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        # First run: creates schema up to core-7.
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        # Second run: should not raise.
        store2 = SqliteEngravaCore(conn)
        await store2.ensure_schema()

        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 17
        await conn.close()


# =====================================================================
# 5. create_thought() with TTL
# =====================================================================


class TestCreateThoughtTTL:
    """create_thought with expires_after_seconds / default_ttl."""

    async def test_no_ttl_by_default(self, store: SqliteEngravaCore) -> None:
        t = _make_thought()
        created = await store.create_thought(t)
        assert created.expires_at is None

    async def test_explicit_expires_after_seconds(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        t = _make_thought()
        created = await store.create_thought(t, expires_after_seconds=600)
        assert created.expires_at is not None
        # Should be roughly 600 seconds from now.
        exp = datetime.datetime.fromisoformat(created.expires_at)
        now = datetime.datetime.now(datetime.UTC)
        diff = (exp - now).total_seconds()
        assert 590 < diff < 610

    async def test_default_ttl_applied(
        self,
        store_default_ttl: SqliteEngravaCore,
    ) -> None:
        t = _make_thought()
        created = await store_default_ttl.create_thought(t)
        assert created.expires_at is not None
        exp = datetime.datetime.fromisoformat(created.expires_at)
        now = datetime.datetime.now(datetime.UTC)
        diff = (exp - now).total_seconds()
        assert 50 < diff < 70

    async def test_explicit_overrides_default(
        self,
        store_default_ttl: SqliteEngravaCore,
    ) -> None:
        t = _make_thought()
        created = await store_default_ttl.create_thought(t, expires_after_seconds=3600)
        assert created.expires_at is not None
        exp = datetime.datetime.fromisoformat(created.expires_at)
        now = datetime.datetime.now(datetime.UTC)
        diff = (exp - now).total_seconds()
        assert 3590 < diff < 3610

    async def test_explicit_expires_at_on_thought_preserved(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """If thought already has expires_at, it should be kept unless overridden."""
        ts = _future_ts(seconds=7200)
        t = _make_thought(expires_at=ts)
        created = await store.create_thought(t)
        assert created.expires_at == ts

    async def test_expires_after_overrides_thought_field(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """expires_after_seconds overrides thought.expires_at."""
        ts = _future_ts(seconds=7200)
        t = _make_thought(expires_at=ts)
        created = await store.create_thought(t, expires_after_seconds=60)
        assert created.expires_at is not None
        assert created.expires_at != ts
        exp = datetime.datetime.fromisoformat(created.expires_at)
        now = datetime.datetime.now(datetime.UTC)
        diff = (exp - now).total_seconds()
        assert 50 < diff < 70

    async def test_persisted_in_db(self, store: SqliteEngravaCore) -> None:
        t = _make_thought()
        created = await store.create_thought(t, expires_after_seconds=300)
        fetched = await store.get_thought(created.thought_id)
        assert fetched is not None
        assert fetched.expires_at == created.expires_at


# =====================================================================
# 6. cleanup_expired() — archive strategy (default)
# =====================================================================


class TestCleanupArchive:
    """cleanup_expired with archive strategy."""

    async def test_archive_expired_thought(self, store: SqliteEngravaCore) -> None:
        past = _past_ts()
        t = _make_thought(expires_at=past)
        await store.create_thought(t)

        result = await store.cleanup_expired()

        assert result.expired_count == 1
        assert result.strategy_applied == "archive"

        fetched = await store.get_thought("t-001")
        assert fetched is not None
        assert fetched.lifecycle_status == LifecycleStatus.ARCHIVED
        assert fetched.expires_at is None

    async def test_archive_skips_non_expired(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        future = _future_ts()
        t = _make_thought(expires_at=future)
        await store.create_thought(t)

        result = await store.cleanup_expired()
        assert result.expired_count == 0

        fetched = await store.get_thought("t-001")
        assert fetched is not None
        assert fetched.lifecycle_status == LifecycleStatus.CREATED

    async def test_archive_skips_no_ttl(self, store: SqliteEngravaCore) -> None:
        t = _make_thought()
        await store.create_thought(t)

        result = await store.cleanup_expired()
        assert result.expired_count == 0

    async def test_archive_multiple(self, store: SqliteEngravaCore) -> None:
        past = _past_ts()
        for i in range(3):
            await store.create_thought(_make_thought(thought_id=f"t-{i:03d}", expires_at=past))

        result = await store.cleanup_expired()
        assert result.expired_count == 3

    async def test_archive_with_explicit_now(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        future = _future_ts(seconds=100)
        t = _make_thought(expires_at=future)
        await store.create_thought(t)

        # Pass a "now" far in the future.
        far_future = _future_ts(seconds=200)
        result = await store.cleanup_expired(now=far_future)
        assert result.expired_count == 1


# =====================================================================
# 7. cleanup_expired() — delete strategy
# =====================================================================


class TestCleanupDelete:
    """cleanup_expired with delete strategy."""

    async def test_delete_expired_thought(
        self,
        store_delete: SqliteEngravaCore,
    ) -> None:
        past = _past_ts()
        t = _make_thought(expires_at=past)
        await store_delete.create_thought(t)

        result = await store_delete.cleanup_expired()

        assert result.expired_count == 1
        assert result.strategy_applied == "delete"

        fetched = await store_delete.get_thought("t-001")
        assert fetched is None

    async def test_delete_skips_non_expired(
        self,
        store_delete: SqliteEngravaCore,
    ) -> None:
        future = _future_ts()
        t = _make_thought(expires_at=future)
        await store_delete.create_thought(t)

        result = await store_delete.cleanup_expired()
        assert result.expired_count == 0

        fetched = await store_delete.get_thought("t-001")
        assert fetched is not None


# =====================================================================
# 8. cleanup_expired() — journal integration
# =====================================================================


class TestCleanupJournal:
    """cleanup_expired records entries in journal when enabled."""

    async def test_archive_journals_update(
        self,
        store_journal: SqliteEngravaCore,
    ) -> None:
        past = _past_ts()
        t = _make_thought(expires_at=past)
        await store_journal.create_thought(t)

        result = await store_journal.cleanup_expired()
        assert result.expired_count == 1

        writer = store_journal.journal
        assert writer is not None
        entries = await writer.get_entries()
        # At least 2 entries: create_thought + cleanup update
        update_entries = [e for e in entries if e.mutation_type == "UPDATE_THOUGHT"]
        assert len(update_entries) >= 1
        assert update_entries[-1].target_id == "t-001"

    async def test_delete_journals_delete(self, db: aiosqlite.Connection) -> None:
        store = SqliteEngravaCore(
            db,
            journal_enabled=True,
            ttl_strategy="delete",
        )
        await store._probe_fts()

        past = _past_ts()
        t = _make_thought(expires_at=past)
        await store.create_thought(t)

        await store.cleanup_expired()

        writer = store.journal
        assert writer is not None
        entries = await writer.get_entries()
        delete_entries = [e for e in entries if e.mutation_type == "DELETE_THOUGHT"]
        assert len(delete_entries) >= 1
        assert delete_entries[-1].target_id == "t-001"


# =====================================================================
# 9. Search exclusion
# =====================================================================


class TestSearchExclusion:
    """Expired thoughts are excluded from search methods by default."""

    async def test_list_thoughts_excludes_expired(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought(thought_id="t-active"))
        await store.create_thought(
            _make_thought(thought_id="t-expired", expires_at=_past_ts()),
        )

        results = await store.list_thoughts()
        ids = [t.thought_id for t in results]
        assert "t-active" in ids
        assert "t-expired" not in ids

    async def test_list_thoughts_include_expired(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make_thought(thought_id="t-active"))
        await store.create_thought(
            _make_thought(thought_id="t-expired", expires_at=_past_ts()),
        )

        results = await store.list_thoughts(include_expired=True)
        ids = [t.thought_id for t in results]
        assert "t-active" in ids
        assert "t-expired" in ids

    async def test_search_fts_excludes_expired(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(
            _make_thought(thought_id="t-active"),
        )
        await store.create_thought(
            _make_thought(thought_id="t-expired", expires_at=_past_ts()),
        )

        results = await store.search_fts("Test thought")
        ids = [r[0] for r in results]
        assert "t-active" in ids
        assert "t-expired" not in ids

    async def test_search_fts_includes_future_expiry(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(
            _make_thought(thought_id="t-future", expires_at=_future_ts()),
        )

        results = await store.search_fts("Test thought")
        ids = [r[0] for r in results]
        assert "t-future" in ids


# =====================================================================
# 10. Auto-cleanup cadence
# =====================================================================


class TestAutoCleanup:
    """Auto-cleanup triggered every N operations."""

    async def test_auto_cleanup_after_n_ops(
        self,
        store_auto: SqliteEngravaCore,
    ) -> None:
        # Create an expired thought (this counts as op 1).
        past = _past_ts()
        await store_auto.create_thought(_make_thought(thought_id="t-exp", expires_at=past))

        # Op 2 hits cadence (check_every_n=2) → auto-cleanup runs.
        await store_auto.create_thought(_make_thought(thought_id="t-001"))
        fetched = await store_auto.get_thought("t-exp")
        assert fetched is not None
        assert fetched.lifecycle_status == LifecycleStatus.ARCHIVED

    async def test_no_auto_cleanup_when_disabled(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        past = _past_ts()
        await store.create_thought(_make_thought(thought_id="t-exp", expires_at=past))

        # Many operations, but auto-cleanup is off (check_every_n = 0).
        for i in range(10):
            await store.create_thought(_make_thought(thought_id=f"t-{i:03d}"))

        fetched = await store.get_thought("t-exp")
        assert fetched is not None
        assert fetched.lifecycle_status == LifecycleStatus.CREATED


# =====================================================================
# 11. ReadOnlyEngrava blocks cleanup_expired
# =====================================================================


class TestReadOnlyBlocked:
    """ReadOnlyEngrava must block cleanup_expired."""

    async def test_cleanup_blocked(self, store: SqliteEngravaCore) -> None:
        ro = ReadOnlyEngrava(store)
        with pytest.raises(ReadOnlyViolationError):
            await ro.cleanup_expired()

    async def test_create_thought_blocked(self, store: SqliteEngravaCore) -> None:
        ro = ReadOnlyEngrava(store)
        t = _make_thought()
        with pytest.raises(ReadOnlyViolationError):
            await ro.create_thought(t, expires_after_seconds=60)


# =====================================================================
# 12. Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge cases for TTL logic."""

    async def test_cleanup_empty_store(self, store: SqliteEngravaCore) -> None:
        result = await store.cleanup_expired()
        assert result.expired_count == 0
        assert result.strategy_applied == "archive"

    async def test_past_expires_at_on_create(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Creating a thought with past expires_at is allowed."""
        past = _past_ts()
        t = _make_thought(expires_at=past)
        created = await store.create_thought(t)
        assert created.expires_at == past

    async def test_cleanup_result_timestamp(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        result = await store.cleanup_expired()
        # Timestamp should be a valid ISO-8601 string.
        datetime.datetime.fromisoformat(result.timestamp)

    async def test_update_thought_triggers_auto_cleanup(
        self,
        store_auto: SqliteEngravaCore,
    ) -> None:
        """update_thought also increments the operation counter."""
        past = _past_ts()
        await store_auto.create_thought(_make_thought(thought_id="t-exp", expires_at=past))
        t = _make_thought(thought_id="t-target")
        await store_auto.create_thought(t)

        # 2nd op: update_thought hits cadence.
        await store_auto.update_thought("t-target", content="updated")

        fetched = await store_auto.get_thought("t-exp")
        assert fetched is not None
        assert fetched.lifecycle_status == LifecycleStatus.ARCHIVED


# =====================================================================
# 13. Backward compatibility
# =====================================================================


class TestBackwardCompat:
    """Without TTL config, behavior should be identical to pre-TTL."""

    async def test_no_ttl_config_defaults(self) -> None:
        cfg = _parse_ttl(None)
        assert cfg.strategy == "archive"
        assert cfg.check_every_n_operations == 0
        assert cfg.default_ttl_seconds is None

    async def test_store_works_without_ttl_params(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        store = SqliteEngravaCore(db)
        await store._probe_fts()
        t = _make_thought()
        created = await store.create_thought(t)
        assert created.expires_at is None
        fetched = await store.get_thought("t-001")
        assert fetched is not None
        assert fetched.expires_at is None


# =====================================================================
# 14. Integration scenario
# =====================================================================


class TestIntegration:
    """End-to-end TTL scenario: create → expire → cleanup → verify."""

    async def test_full_lifecycle_archive(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Create with short TTL.
        t = _make_thought(thought_id="t-session")
        created = await store.create_thought(t, expires_after_seconds=1)
        assert created.expires_at is not None

        # Not yet expired: use now = 0.5s from now.
        now_plus_half = (
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(milliseconds=500)
        ).isoformat()
        result = await store.cleanup_expired(now=now_plus_half)
        assert result.expired_count == 0

        # Expired: use now = 2s from now.
        now_plus_two = (
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=2)
        ).isoformat()
        result = await store.cleanup_expired(now=now_plus_two)
        assert result.expired_count == 1

        fetched = await store.get_thought("t-session")
        assert fetched is not None
        assert fetched.lifecycle_status == LifecycleStatus.ARCHIVED
        assert fetched.expires_at is None

    async def test_full_lifecycle_delete(
        self,
        store_delete: SqliteEngravaCore,
    ) -> None:
        t = _make_thought(thought_id="t-temp")
        created = await store_delete.create_thought(t, expires_after_seconds=1)
        assert created.expires_at is not None

        now_plus_two = (
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=2)
        ).isoformat()
        result = await store_delete.cleanup_expired(now=now_plus_two)
        assert result.expired_count == 1

        fetched = await store_delete.get_thought("t-temp")
        assert fetched is None


# =====================================================================
# 15. Quality fixes — timezone normalization
# =====================================================================


class TestTimezoneNormalization:
    """Timestamps with non-UTC offsets are normalized to UTC."""

    def test_positive_offset_normalized(self) -> None:
        """Timestamp with +02:00 offset is stored as UTC."""
        t = _make_thought(expires_at="2026-04-12T15:00:00+02:00")
        assert t.expires_at is not None
        assert t.expires_at == "2026-04-12T13:00:00+00:00"

    def test_negative_offset_normalized(self) -> None:
        t = _make_thought(expires_at="2026-04-12T10:00:00-05:00")
        assert t.expires_at is not None
        assert t.expires_at == "2026-04-12T15:00:00+00:00"

    def test_utc_offset_unchanged(self) -> None:
        t = _make_thought(expires_at="2026-04-12T13:00:00+00:00")
        assert t.expires_at == "2026-04-12T13:00:00+00:00"

    def test_naive_timestamp_preserved(self) -> None:
        """Naive timestamps (no timezone info) are left as-is."""
        t = _make_thought(expires_at="2026-04-12T13:00:00")
        assert t.expires_at == "2026-04-12T13:00:00"

    def test_created_at_normalized(self) -> None:
        """created_at also benefits from normalization."""
        t = ThoughtRecord(
            thought_id="t-tz",
            thought_type=ThoughtType.TASK,
            essence="TZ test",
            content="TZ content",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=0,
            updated_cycle=0,
            source="test",
            created_at="2026-01-01T12:00:00+03:00",
        )
        assert t.created_at == "2026-01-01T09:00:00+00:00"

    async def test_offset_expires_at_compared_correctly(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Offset timestamps should be normalized before storage so SQL TEXT comparison works."""
        # 1 hour in the past expressed with +02:00 offset.
        past_offset = (
            (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1))
            .astimezone(datetime.timezone(datetime.timedelta(hours=2)))
            .isoformat()
        )
        t = _make_thought(expires_at=past_offset)
        await store.create_thought(t)

        result = await store.cleanup_expired()
        assert result.expired_count == 1


# =====================================================================
# 16. Quality fixes — exclude_id in cleanup_expired
# =====================================================================


class TestCleanupExcludeId:
    """cleanup_expired(exclude_id=...) skips the given thought."""

    async def test_exclude_id_skips_target(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        past = _past_ts()
        await store.create_thought(_make_thought(thought_id="t-exp1", expires_at=past))
        await store.create_thought(_make_thought(thought_id="t-exp2", expires_at=past))

        result = await store.cleanup_expired(exclude_id="t-exp1")
        assert result.expired_count == 1

        # t-exp1 should still be CREATED (skipped).
        t1 = await store.get_thought("t-exp1")
        assert t1 is not None
        assert t1.lifecycle_status == LifecycleStatus.CREATED

        # t-exp2 should be ARCHIVED (cleaned up).
        t2 = await store.get_thought("t-exp2")
        assert t2 is not None
        assert t2.lifecycle_status == LifecycleStatus.ARCHIVED

    async def test_auto_cleanup_excludes_current_thought(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        """Auto-cleanup after create_thought should not archive the just-created thought."""
        store = SqliteEngravaCore(db, ttl_check_every_n=1, ttl_default_seconds=1)
        await store._probe_fts()

        # Create a thought with 1s TTL that immediately expires.
        past = _past_ts(seconds=10)
        t = _make_thought(thought_id="t-new", expires_at=past)
        created = await store.create_thought(t)

        # Even though thought is already expired and auto-cleanup fires,
        # the just-created thought should NOT be cleaned up yet.
        fetched = await store.get_thought("t-new")
        assert fetched is not None
        assert fetched.lifecycle_status == created.lifecycle_status


# =====================================================================
# 17. Quality fixes — sqlite-vec search filters expired
# =====================================================================


class TestSqliteVecExpiredFilter:
    """_filter_expired_results properly filters expired thought IDs."""

    async def test_filter_removes_expired(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        past = _past_ts()
        await store.create_thought(_make_thought(thought_id="t-exp", expires_at=past))
        await store.create_thought(_make_thought(thought_id="t-ok"))

        # Simulate results that would come from sqlite-vec.
        results = [("t-exp", 0.9), ("t-ok", 0.8)]
        filtered = await store._filter_expired_results(results)

        ids = [r[0] for r in filtered]
        assert "t-exp" not in ids
        assert "t-ok" in ids

    async def test_filter_empty_input(self, store: SqliteEngravaCore) -> None:
        filtered = await store._filter_expired_results([])
        assert filtered == []

    async def test_filter_keeps_non_expired(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        future = _future_ts()
        await store.create_thought(_make_thought(thought_id="t-future", expires_at=future))
        await store.create_thought(_make_thought(thought_id="t-ok"))

        results = [("t-future", 0.9), ("t-ok", 0.8)]
        filtered = await store._filter_expired_results(results)
        assert len(filtered) == 2

"""Engrava completeness tests.

Covers hybrid search, access tracking, datetime timestamps,
evolve() auto-updated_at, created_at immutability, FrequencySignal,
and core-3→4 migration.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    FrequencySignal,
    HybridSearchResult,
    InvalidTransitionError,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtNotFoundError,
    ThoughtType,
)
from engrava.extensions.dreaming_signals import DreamingContext
from engrava.infrastructure.sqlite.engrava_core import _normalize_min_max

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: object) -> AsyncIterator[aiosqlite.Connection]:
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
    s = SqliteEngravaCore(db)
    await s._probe_fts()
    return s


def _make(
    thought_id: str = "t-001",
    *,
    access_count: int = 0,
    created_at: str | None = None,
    updated_at: str | None = None,
    last_accessed_at: str | None = None,
    lifecycle_status: LifecycleStatus = LifecycleStatus.CREATED,
    essence: str = "Test thought",
    content: str = "Test thought content",
) -> CoreThoughtRecord:
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence=essence,
        content=content,
        priority=Priority.P2,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        access_count=access_count,
        created_at=created_at,
        updated_at=updated_at,
        last_accessed_at=last_accessed_at,
    )


# ===================================================================
# ThoughtRecord — new fields + validators
# ===================================================================


class TestThoughtTimestampFields:
    """Tests for access_count, created_at, updated_at, last_accessed_at."""

    def test_defaults(self) -> None:
        t = _make()
        assert t.access_count == 0
        assert t.created_at is None
        assert t.updated_at is None
        assert t.last_accessed_at is None

    def test_explicit_timestamps(self) -> None:
        now = datetime.datetime.now(datetime.UTC).isoformat()
        t = _make(created_at=now, updated_at=now, last_accessed_at=now)
        assert t.created_at == now
        assert t.updated_at == now
        assert t.last_accessed_at == now

    def test_iso8601_validator_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            _make(created_at="not-a-date")

    def test_iso8601_validator_rejects_updated_at_garbage(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            _make(updated_at="bad")

    def test_iso8601_validator_rejects_last_accessed_at_garbage(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            _make(last_accessed_at="nope")

    def test_iso8601_accepts_various_formats(self) -> None:
        t = _make(created_at="2025-01-01T00:00:00+00:00")
        assert t.created_at is not None
        t2 = _make(created_at="2025-01-01T12:30:00")
        assert t2.created_at is not None

    def test_access_count_ge_zero(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal"):
            _make(access_count=-1)

    def test_access_count_positive(self) -> None:
        t = _make(access_count=42)
        assert t.access_count == 42


# ===================================================================
# ThoughtRecord.evolve() — auto updated_at + created_at immutability
# ===================================================================


class TestEvolveTimestamps:
    """Tests for evolve() auto-setting updated_at and guarding created_at."""

    def test_evolve_auto_sets_updated_at(self) -> None:
        t = _make()
        before = datetime.datetime.now(datetime.UTC)
        t2 = t.evolve(essence="Changed")
        after = datetime.datetime.now(datetime.UTC)

        assert t2.updated_at is not None
        ts = datetime.datetime.fromisoformat(t2.updated_at)
        assert before <= ts <= after

    def test_evolve_explicit_updated_at_respected(self) -> None:
        t = _make()
        explicit = "2099-12-31T23:59:59+00:00"
        t2 = t.evolve(updated_at=explicit)
        assert t2.updated_at == explicit

    def test_evolve_created_at_immutable(self) -> None:
        now = datetime.datetime.now(datetime.UTC).isoformat()
        t = _make(created_at=now)
        with pytest.raises(ValueError, match="created_at is immutable"):
            t.evolve(created_at="2099-01-01T00:00:00")

    def test_evolve_created_at_same_value_ok(self) -> None:
        now = datetime.datetime.now(datetime.UTC).isoformat()
        t = _make(created_at=now)
        t2 = t.evolve(created_at=now)
        assert t2.created_at == now

    def test_evolve_created_at_from_none_ok(self) -> None:
        t = _make()  # created_at=None
        now = datetime.datetime.now(datetime.UTC).isoformat()
        t2 = t.evolve(created_at=now)
        assert t2.created_at == now

    def test_evolve_lifecycle_still_validated(self) -> None:
        t = _make(lifecycle_status=LifecycleStatus.ARCHIVED)
        with pytest.raises(InvalidTransitionError):
            t.evolve(lifecycle_status=LifecycleStatus.CREATED)


# ===================================================================
# HybridSearchResult value object
# ===================================================================


class TestHybridSearchResult:
    def test_defaults(self) -> None:
        r = HybridSearchResult()
        assert r.results == []
        assert r.backends_used == frozenset()

    def test_with_data(self) -> None:
        r = HybridSearchResult(
            results=[("t1", 0.9), ("t2", 0.5)],
            backends_used=frozenset({"fts5", "vector"}),
        )
        assert len(r.results) == 2
        assert "fts5" in r.backends_used
        assert "vector" in r.backends_used

    def test_frozen(self) -> None:
        r = HybridSearchResult()
        with pytest.raises(AttributeError):
            r.results = []  # type: ignore[misc]


# ===================================================================
# _normalize_min_max helper
# ===================================================================


class TestNormalizeMinMax:
    def test_empty(self) -> None:
        assert _normalize_min_max([]) == []

    def test_single_item(self) -> None:
        result = _normalize_min_max([("t1", 5.0)])
        assert result == [("t1", 1.0)]

    def test_two_items(self) -> None:
        result = _normalize_min_max([("t1", 10.0), ("t2", 20.0)])
        assert result[0] == ("t1", 0.0)
        assert result[1] == ("t2", 1.0)

    def test_three_items(self) -> None:
        result = _normalize_min_max([("a", 0.0), ("b", 5.0), ("c", 10.0)])
        assert result[0] == ("a", 0.0)
        assert result[1] == ("b", 0.5)
        assert result[2] == ("c", 1.0)

    def test_all_same_score(self) -> None:
        result = _normalize_min_max([("t1", 7.0), ("t2", 7.0), ("t3", 7.0)])
        assert all(s == 1.0 for _, s in result)


# ===================================================================
# search_hybrid — integration
# ===================================================================


class TestSearchHybrid:
    async def test_no_fts_no_vec_returns_empty(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        r = await store.search_hybrid("query", [0.1, 0.2, 0.3])
        assert r.results == []
        # Both backends are *available* even though neither returned results.
        assert "fts5" in r.backends_used
        assert "vector" in r.backends_used

    async def test_fts_only_graceful(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """When no embeddings, only FTS5 contributes."""
        await store.create_thought(
            _make("t-fts1", essence="Quantum computing", content="Quantum entanglement research"),
        )
        r = await store.search_hybrid("quantum", [0.0, 0.0, 0.0])
        assert len(r.results) >= 1
        assert "fts5" in r.backends_used
        # vector is always available (numpy fallback)
        assert "vector" in r.backends_used

    async def test_both_backends_used(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Both FTS5 and vector should contribute when data available."""
        t = await store.create_thought(
            _make("t-both", essence="Machine learning", content="Neural network training"),
        )
        # Store embedding for this thought
        import numpy as np

        vec = np.random.default_rng(42).random(384).astype(np.float32)
        await store.store_embedding(t.thought_id, vec.tolist(), model_name="test-model")

        r = await store.search_hybrid(
            "machine learning",
            vec.tolist(),
        )
        assert "fts5" in r.backends_used
        assert "vector" in r.backends_used
        assert len(r.results) >= 1

    async def test_top_k_limits_results(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        for i in range(5):
            await store.create_thought(
                _make(
                    f"t-lim-{i}",
                    essence=f"Topic alpha {i}",
                    content=f"Content alpha detail {i}",
                ),
            )
        r = await store.search_hybrid("alpha", [0.0], top_k=2)
        assert len(r.results) <= 2

    async def test_weighted_fusion_scores(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """All scores should be non-negative after fusion."""
        await store.create_thought(
            _make("t-w1", essence="Important task", content="Priority task details"),
        )
        r = await store.search_hybrid("important", [0.0])
        for _, score in r.results:
            assert score >= 0.0


# ===================================================================
# record_access
# ===================================================================


class TestRecordAccess:
    async def test_increments_access_count(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        t = await store.create_thought(_make("t-acc"))
        assert t.access_count == 0

        await store.record_access("t-acc")
        t2 = await store.get_thought("t-acc")
        assert t2 is not None
        assert t2.access_count == 1

        await store.record_access("t-acc")
        t3 = await store.get_thought("t-acc")
        assert t3 is not None
        assert t3.access_count == 2

    async def test_sets_last_accessed_at(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        await store.create_thought(_make("t-la"))
        t1 = await store.get_thought("t-la")
        assert t1 is not None
        assert t1.last_accessed_at is None

        before = datetime.datetime.now(datetime.UTC)
        await store.record_access("t-la")
        after = datetime.datetime.now(datetime.UTC)

        t2 = await store.get_thought("t-la")
        assert t2 is not None
        assert t2.last_accessed_at is not None
        ts = datetime.datetime.fromisoformat(t2.last_accessed_at)
        assert before <= ts <= after

    async def test_raises_on_missing_thought(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        with pytest.raises(ThoughtNotFoundError):
            await store.record_access("nonexistent-id")


# ===================================================================
# create_thought auto-timestamps
# ===================================================================


class TestCreateThoughtTimestamps:
    async def test_auto_sets_created_at_and_updated_at(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        before = datetime.datetime.now(datetime.UTC)
        t = await store.create_thought(_make("t-ts"))
        after = datetime.datetime.now(datetime.UTC)

        assert t.created_at is not None
        assert t.updated_at is not None

        ts_created = datetime.datetime.fromisoformat(t.created_at)
        ts_updated = datetime.datetime.fromisoformat(t.updated_at)
        assert before <= ts_created <= after
        assert before <= ts_updated <= after

    async def test_explicit_timestamps_preserved(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        explicit = "2025-06-15T12:00:00+00:00"
        t = await store.create_thought(
            _make("t-exp", created_at=explicit, updated_at=explicit),
        )
        assert t.created_at == explicit
        assert t.updated_at == explicit

    async def test_roundtrip_preserves_timestamps(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        t = await store.create_thought(_make("t-rt"))
        t2 = await store.get_thought("t-rt")
        assert t2 is not None
        assert t.created_at == t2.created_at
        assert t.updated_at == t2.updated_at


# ===================================================================
# FrequencySignal
# ===================================================================


class TestFrequencySignal:
    def test_zero_accesses(self) -> None:
        sig = FrequencySignal()
        ctx = DreamingContext(current_cycle=10, total_thoughts=5)
        t = _make(access_count=0)
        assert sig(t, ctx) == 0.0

    def test_half_max(self) -> None:
        sig = FrequencySignal(max_accesses=10)
        ctx = DreamingContext(current_cycle=10, total_thoughts=5)
        t = _make(access_count=5)
        assert sig(t, ctx) == pytest.approx(0.5)

    def test_at_max(self) -> None:
        sig = FrequencySignal(max_accesses=10)
        ctx = DreamingContext(current_cycle=10, total_thoughts=5)
        t = _make(access_count=10)
        assert sig(t, ctx) == pytest.approx(1.0)

    def test_above_max_clamped(self) -> None:
        sig = FrequencySignal(max_accesses=10)
        ctx = DreamingContext(current_cycle=10, total_thoughts=5)
        t = _make(access_count=20)
        assert sig(t, ctx) == 1.0

    def test_custom_max_accesses(self) -> None:
        sig = FrequencySignal(max_accesses=100)
        ctx = DreamingContext(current_cycle=10, total_thoughts=5)
        t = _make(access_count=50)
        assert sig(t, ctx) == pytest.approx(0.5)

    def test_max_accesses_floor_at_one(self) -> None:
        sig = FrequencySignal(max_accesses=0)
        ctx = DreamingContext(current_cycle=10, total_thoughts=5)
        t = _make(access_count=1)
        assert sig(t, ctx) == 1.0


# ===================================================================
# Migration core-3 → core-4
# ===================================================================


class TestMigrationCoreV3ToV4:
    async def test_migration_adds_columns(self) -> None:
        """Migration from core-3 should add all 4 new columns."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        cursor = await conn.execute("PRAGMA table_info(thought)")
        cols = {row[1] for row in await cursor.fetchall()}
        assert "access_count" in cols
        assert "last_accessed_at" in cols
        assert "created_at" in cols
        assert "updated_at" in cols

        await conn.close()

    async def test_migration_idempotent(self) -> None:
        """Running ensure_schema twice should not error."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.ensure_schema()  # Second call should be safe

        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 14

        await conn.close()

    async def test_migration_backfills_timestamps(self) -> None:
        """Existing rows without timestamps should get backfilled."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        # Create a thought (auto-timestamps)
        t = await store.create_thought(_make("t-bf"))
        assert t.created_at is not None
        assert t.updated_at is not None

        # Simulate a pre-migration row by nulling timestamps
        await conn.execute(
            "UPDATE thought SET created_at = NULL, updated_at = NULL WHERE thought_id = 't-bf'",
        )
        await conn.commit()

        # Re-run migration
        await store._migrate_core_v3_to_v4()
        await conn.commit()

        t2 = await store.get_thought("t-bf")
        assert t2 is not None
        assert t2.created_at is not None
        assert t2.updated_at is not None

        await conn.close()

    async def test_migration_backfills_partial_null_updated_at(self) -> None:
        """Row with created_at set but updated_at NULL gets backfilled."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        await store.create_thought(_make("t-partial"))

        # Simulate partial state: created_at present, updated_at NULL.
        await conn.execute(
            "UPDATE thought SET updated_at = NULL WHERE thought_id = 't-partial'",
        )
        await conn.commit()

        await store._migrate_core_v3_to_v4()
        await conn.commit()

        t = await store.get_thought("t-partial")
        assert t is not None
        assert t.created_at is not None
        assert t.updated_at is not None

        await conn.close()


class TestSearchHybridBackendsUsed:
    """backends_used reflects availability, not just non-empty results."""

    async def test_vector_always_in_backends(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Vector backend should always appear (numpy fallback)."""
        await store.create_thought(_make("t-hb"))
        result = await store.search_hybrid(
            query_text="anything",
            query_vector=[0.1] * 384,
        )
        assert "vector" in result.backends_used

    async def test_fts5_in_backends_when_available_no_results(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """FTS5 should appear even when query matches nothing."""
        await store.create_thought(_make("t-hb2"))
        result = await store.search_hybrid(
            query_text="zzzznonexistent",
            query_vector=[0.1] * 384,
        )
        # FTS5 is available in test DB (ensure_schema creates it)
        assert "fts5" in result.backends_used
        assert "vector" in result.backends_used

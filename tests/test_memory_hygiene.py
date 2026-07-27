"""Tests for the deterministic Memory Hygiene forgetting loop.

Covers the opt-in, no-LLM forgetting pass that archives cold/low-value thoughts
and, separately opt-in, garbage-collects them after a restore window:

* config: ``HygienePolicyConfig`` defaults, validation, and YAML parsing;
* the ``core-18`` migration (``pinned`` / ``archived_at_cycle`` columns) and the
  ``core-20`` migration (``archived_at`` column) plus the ThoughtRecord round trip
  of those fields;
* default-OFF ⇒ no behavioural change (byte-identical write/read paths);
* keep-score + eviction rule with active-signal redistribution;
* protection: pinned is never touched, ``P1`` is protected by default,
  ``confidence`` is not protection;
* archive is reversible and restore clears ``archived_at_cycle``;
* deterministic capped selection (same store + config + cycle ⇒ same set);
* the all-flat-signals fail-safe (zero evictions);
* the decay clamp (a hook returning >1 / <0 / NaN / inf never over-evicts);
* two-stage GC: only hygiene-archived rows reaped, both restore windows enforced
  (cycle + wall-clock, BOTH required; legacy NULL fails closed; monotone-safe),
  orphan-reflection sweep before delete, vec0 purge, hash-chain still valid;
* ``dry_run`` mutates nothing;
* the journal ``eviction_reason`` with no new mutation type;
* the ``consolidate()`` convenience invocation + cadence.
"""

from __future__ import annotations

import datetime
import importlib.util
import math
from typing import TYPE_CHECKING
from unittest import mock

import aiosqlite
import pytest

from engrava import (
    DreamingConfig,
    HygienePolicyConfig,
    InvalidTransitionError,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtNotFoundError,
    ThoughtRecord,
    ThoughtType,
)
from engrava.config import ConfigError, _parse_hygiene
from engrava.domain.enums import EdgeType, KnowledgeSource
from engrava.domain.models.edge import EdgeRecord
from engrava.infrastructure.sqlite import engrava_core
from engrava.infrastructure.sqlite.engrava_core import (
    _clamp_decay,
    _ensure_utc,
    _hygiene_inactive_enough,
    _hygiene_protected,
)
from engrava.infrastructure.sqlite.hygiene import (
    USAGE_HISTORY_SIGNALS,
    EvictionReason,
    HygieneResult,
    compute_active_hygiene_weights,
    has_active_usage_signal,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """A schema-bootstrapped in-memory store (journaling off, no hygiene policy)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(conn)
    await s.ensure_schema()
    yield s
    await conn.close()


async def _make_store(
    policy: HygienePolicyConfig | None,
    *,
    journal_enabled: bool = False,
    ttl_strategy: str = "archive",
) -> SqliteEngravaCore:
    """Build a bootstrapped in-memory store carrying the given hygiene policy."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(
        conn,
        hygiene_policy=policy,
        journal_enabled=journal_enabled,
        ttl_strategy=ttl_strategy,
    )
    await s.ensure_schema()
    return s


# A wall-clock instant far enough in the past that any thought stamped with it
# is always older than the default minimum-inactivity window (7 days) — used to
# build a "genuinely aged" store for the eviction path.
_LONG_AGO = "2000-01-01T00:00:00+00:00"

# An expiry no wall clock this suite runs under will reach. A TTL must be
# unreached for the row to stay in the hygiene candidate pool at all
# (``list_thoughts`` excludes expired rows against the real clock), so a test
# pinning TTL behaviour resolves the expiry only through an injected sweep
# instant — never by the machine's date.
_FAR_FUTURE = "2999-01-01T00:00:00+00:00"
_AFTER_FAR_FUTURE = "2999-01-02T00:00:00+00:00"

# A wall-clock instant tests pin so every wall-clock boundary (the
# minimum-inactivity age and the GC restore window) is deterministic.
_NOW = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
_WEEK_SECONDS = 604800
_MONTH_SECONDS = 2592000  # 30 days — the default wall-clock restore window.


def _iso_before(now: datetime.datetime, seconds: float) -> str:
    """ISO-8601 timestamp ``seconds`` before ``now`` (UTC)."""
    return (now - datetime.timedelta(seconds=seconds)).isoformat()


def _thought(
    thought_id: str,
    *,
    priority: Priority = Priority.P3,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    created_cycle: int = 0,
    updated_cycle: int = 0,
    confidence: float | None = None,
    confirmation_count: int = 0,
    access_count: int = 0,
    action_outcome_score: float | None = None,
    pinned: bool = False,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    created_at: str | None = None,
    updated_at: str | None = None,
    last_accessed_at: str | None = None,
    expires_at: str | None = None,
) -> ThoughtRecord:
    """Build a thought with hygiene-relevant fields controllable per test.

    The transaction-time fields drive the minimum-inactivity-age gate through a
    COALESCE ladder — ``last_accessed_at`` → ``updated_at`` → ``created_at`` —
    in which the first one present decides the row's age. ``create_thought``
    stamps **both** ``created_at`` and ``updated_at`` to *now* when they are
    ``None``, so ``created_at=_LONG_AGO`` on its own still yields a *young* row:
    the stamped ``updated_at`` outranks it. Ageing a row past the gate means
    setting ``updated_at`` (or ``last_accessed_at``) as well — which is what
    :func:`_evictable_thought` does.
    ``action_outcome_score`` is the usage-history signal used to satisfy the
    run-level access-gate without perturbing the keep-score (it is not a
    configured hygiene keep-signal). ``expires_at`` puts the thought under TTL,
    which hygiene archival must clear.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=f"essence {thought_id}",
        content=f"content of {thought_id}",
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confidence=confidence,
        confirmation_count=confirmation_count,
        access_count=access_count,
        action_outcome_score=action_outcome_score,
        pinned=pinned,
        created_at=created_at,
        updated_at=updated_at,
        last_accessed_at=last_accessed_at,
        expires_at=expires_at,
    )


def _evictable_thought(thought_id: str, *, expires_at: str | None = None) -> ThoughtRecord:
    """A thought that an *enabled* policy archives at a far-future cycle.

    Every gate other than the ``enabled`` master switch is deliberately
    satisfied, so a test built on this row isolates that switch:

    * **minimum-inactivity-age gate** — ``created_at`` / ``updated_at`` stamped
      ``_LONG_AGO``, decades past the default 7-day window. Aged by an absolute
      timestamp rather than an injected offset, so the row is archivable under
      *any* wall clock — which is what lets a caller that cannot inject ``now``
      (``consolidate()``) stay deterministic.
    * **run-level access-gate** — ``action_outcome_score`` supplies the
      usage-history signal the pool needs.
    * **keep-score** — ``updated_cycle=0`` against a far-future cycle makes the
      row maximally cold, so any threshold above 0 evicts it.
    """
    return _thought(
        thought_id,
        updated_cycle=0,
        action_outcome_score=0.0,
        created_at=_LONG_AGO,
        updated_at=_LONG_AGO,
        expires_at=expires_at,
    )


def _forgetful_policy(**overrides: object) -> HygienePolicyConfig:
    """An enabled hygiene policy with the minimum-inactivity-age gate disabled.

    The two cold-start guards (the per-thought minimum-inactivity-age gate and
    the run-level access-gate) were added after most of the scoring / archive /
    GC scenarios below were written. Setting ``min_inactivity_age_seconds=0``
    turns off the *age* gate so those tests stay focused on the behaviour they
    actually exercise on a freshly-created pool. The *access-gate* is not a
    config knob, so each such test still seeds one usage-history signal in its
    pool (an ``action_outcome_score`` thought) for the run to proceed.
    """
    params: dict[str, object] = {"enabled": True, "min_inactivity_age_seconds": 0}
    params.update(overrides)
    return HygienePolicyConfig(**params)  # type: ignore[arg-type]


async def _raw_lifecycle(store: SqliteEngravaCore, thought_id: str) -> str | None:
    cursor = await store._db.execute(
        "SELECT lifecycle_status FROM thought WHERE thought_id = ?", (thought_id,)
    )
    row = await cursor.fetchone()
    return None if row is None else str(row["lifecycle_status"])


async def _raw_archived_at_cycle(store: SqliteEngravaCore, thought_id: str) -> int | None:
    cursor = await store._db.execute(
        "SELECT archived_at_cycle FROM thought WHERE thought_id = ?", (thought_id,)
    )
    row = await cursor.fetchone()
    if row is None or row["archived_at_cycle"] is None:
        return None
    return int(row["archived_at_cycle"])


async def _raw_archived_at(store: SqliteEngravaCore, thought_id: str) -> str | None:
    cursor = await store._db.execute(
        "SELECT archived_at FROM thought WHERE thought_id = ?", (thought_id,)
    )
    row = await cursor.fetchone()
    if row is None or row["archived_at"] is None:
        return None
    return str(row["archived_at"])


async def _raw_expires_at(store: SqliteEngravaCore, thought_id: str) -> str | None:
    cursor = await store._db.execute(
        "SELECT expires_at FROM thought WHERE thought_id = ?", (thought_id,)
    )
    row = await cursor.fetchone()
    if row is None or row["expires_at"] is None:
        return None
    return str(row["expires_at"])


# ---------------------------------------------------------------------------
# HygienePolicyConfig — defaults + validation
# ---------------------------------------------------------------------------


class TestHygienePolicyConfigDefaults:
    def test_defaults_match_spec(self) -> None:
        """Every documented default is present and OFF by default."""
        cfg = HygienePolicyConfig()
        assert cfg.enabled is False
        assert cfg.eviction_threshold == pytest.approx(0.20)
        assert cfg.protected_priorities == ("P1",)
        assert cfg.check_every_n_cycles == 1
        assert cfg.max_evictions_per_run == 100
        assert cfg.auto_gc_enabled is False
        assert cfg.gc_min_archive_age_cycles == 10
        assert cfg.dry_run is False
        assert cfg.min_inactivity_age_seconds == 604800  # 7 days
        assert cfg.gc_restore_window_seconds == 2592000  # 30 days

    def test_default_signal_weights(self) -> None:
        """The keep-weight vector matches the ratified defaults and sums to 1.0."""
        cfg = HygienePolicyConfig()
        assert cfg.signal_weights == {
            "recency": 0.30,
            "frequency": 0.25,
            "confirmation": 0.20,
            "confidence": 0.15,
            "staleness": 0.10,
        }
        assert sum(cfg.signal_weights.values()) == pytest.approx(1.0)

    def test_frozen(self) -> None:
        """The config is frozen — fields cannot be reassigned."""
        cfg = HygienePolicyConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.enabled = True  # type: ignore[misc]


class TestHygienePolicyConfigValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_threshold_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="eviction_threshold"):
            HygienePolicyConfig(eviction_threshold=bad)

    def test_threshold_bool_rejected(self) -> None:
        """``True`` must not impersonate ``1.0`` for the threshold."""
        with pytest.raises(ConfigError, match="eviction_threshold"):
            HygienePolicyConfig(eviction_threshold=True)  # type: ignore[arg-type]

    def test_check_every_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="check_every_n_cycles"):
            HygienePolicyConfig(check_every_n_cycles=0)

    def test_max_evictions_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_evictions_per_run"):
            HygienePolicyConfig(max_evictions_per_run=0)

    def test_gc_min_age_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="gc_min_archive_age_cycles"):
            HygienePolicyConfig(gc_min_archive_age_cycles=-1)

    def test_min_inactivity_age_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_inactivity_age_seconds"):
            HygienePolicyConfig(min_inactivity_age_seconds=-1)

    def test_min_inactivity_age_zero_allowed(self) -> None:
        """``0`` is the explicit gate-disabled value and must construct cleanly."""
        assert HygienePolicyConfig(min_inactivity_age_seconds=0).min_inactivity_age_seconds == 0

    def test_gc_restore_window_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="gc_restore_window_seconds"):
            HygienePolicyConfig(gc_restore_window_seconds=-1)

    def test_gc_restore_window_zero_allowed(self) -> None:
        """``0`` is the explicit wall-clock-window-disabled value (cycle-only)."""
        cfg = HygienePolicyConfig(gc_restore_window_seconds=0)
        assert cfg.gc_restore_window_seconds == 0

    def test_protected_priorities_non_string_raises(self) -> None:
        with pytest.raises(ConfigError, match="protected_priorities"):
            HygienePolicyConfig(protected_priorities=(1,))  # type: ignore[arg-type]

    def test_signal_weight_non_numeric_raises(self) -> None:
        with pytest.raises(ConfigError, match="signal_weights"):
            HygienePolicyConfig(signal_weights={"recency": "high"})  # type: ignore[dict-item]

    def test_empty_protected_priorities_allowed(self) -> None:
        """An operator may disable priority-based protection entirely."""
        cfg = HygienePolicyConfig(protected_priorities=())
        assert cfg.protected_priorities == ()


class TestHygieneYamlParsing:
    def test_absent_section_is_none(self) -> None:
        assert _parse_hygiene(None) is None

    def test_full_section_parses(self) -> None:
        cfg = _parse_hygiene(
            {
                "enabled": True,
                "eviction_threshold": 0.15,
                "protected_priorities": ["P1", "P2"],
                "signal_weights": {"recency": 0.5},
                "check_every_n_cycles": 3,
                "max_evictions_per_run": 50,
                "auto_gc_enabled": True,
                "gc_min_archive_age_cycles": 20,
                "dry_run": True,
                "min_inactivity_age_seconds": 3600,
                "gc_restore_window_seconds": 86400,
            }
        )
        assert cfg is not None
        assert cfg.enabled is True
        assert cfg.eviction_threshold == pytest.approx(0.15)
        assert cfg.protected_priorities == ("P1", "P2")
        # Partial signal override merges onto defaults (does not zero the rest).
        assert cfg.signal_weights["recency"] == pytest.approx(0.5)
        assert cfg.signal_weights["frequency"] == pytest.approx(0.25)
        assert cfg.auto_gc_enabled is True
        assert cfg.gc_min_archive_age_cycles == 20
        assert cfg.dry_run is True
        assert cfg.min_inactivity_age_seconds == 3600
        assert cfg.gc_restore_window_seconds == 86400

    def test_min_inactivity_age_defaults_when_absent(self) -> None:
        """An omitted ``min_inactivity_age_seconds`` falls back to the 7-day default."""
        cfg = _parse_hygiene({"enabled": True})
        assert cfg is not None
        assert cfg.min_inactivity_age_seconds == 604800

    def test_min_inactivity_age_loader_parity_and_round_trip(self) -> None:
        """The loader mirrors direct construction: same rejection, same round-trip."""
        # Loader parity: the YAML path rejects the same out-of-range value that
        # direct construction does (both name the field), and a bool is not a
        # valid integer (``True`` must not impersonate ``1``).
        with pytest.raises(ConfigError, match="min_inactivity_age_seconds"):
            _parse_hygiene({"min_inactivity_age_seconds": -1})
        with pytest.raises(ConfigError, match="min_inactivity_age_seconds"):
            _parse_hygiene({"min_inactivity_age_seconds": True})
        # Round-trip: an accepted value survives the YAML → dataclass parse.
        cfg = _parse_hygiene({"min_inactivity_age_seconds": 42})
        assert cfg is not None
        assert cfg.min_inactivity_age_seconds == 42

    def test_gc_restore_window_defaults_when_absent(self) -> None:
        """An omitted ``gc_restore_window_seconds`` falls back to the 30-day default."""
        cfg = _parse_hygiene({"enabled": True})
        assert cfg is not None
        assert cfg.gc_restore_window_seconds == 2592000

    def test_gc_restore_window_loader_parity_and_round_trip(self) -> None:
        """The loader mirrors direct construction: same rejection, same round-trip."""
        # Loader parity: the YAML path rejects the same out-of-range value that
        # direct construction does (both name the field), and a bool is not a
        # valid integer (``True`` must not impersonate ``1``).
        with pytest.raises(ConfigError, match="gc_restore_window_seconds"):
            _parse_hygiene({"gc_restore_window_seconds": -1})
        with pytest.raises(ConfigError, match="gc_restore_window_seconds"):
            _parse_hygiene({"gc_restore_window_seconds": True})
        # Round-trip: an accepted value survives the YAML → dataclass parse.
        cfg = _parse_hygiene({"gc_restore_window_seconds": 0})
        assert cfg is not None
        assert cfg.gc_restore_window_seconds == 0

    def test_non_mapping_raises(self) -> None:
        with pytest.raises(ConfigError, match="hygiene_policy"):
            _parse_hygiene([1, 2, 3])

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"enabled": "yes"}, "enabled"),
            ({"eviction_threshold": 2.0}, "eviction_threshold"),
            ({"eviction_threshold": True}, "eviction_threshold"),
            ({"check_every_n_cycles": 0}, "check_every_n_cycles"),
            ({"max_evictions_per_run": -1}, "max_evictions_per_run"),
            ({"gc_min_archive_age_cycles": -1}, "gc_min_archive_age_cycles"),
            ({"min_inactivity_age_seconds": -1}, "min_inactivity_age_seconds"),
            ({"min_inactivity_age_seconds": True}, "min_inactivity_age_seconds"),
            ({"min_inactivity_age_seconds": 1.5}, "min_inactivity_age_seconds"),
            ({"gc_restore_window_seconds": -1}, "gc_restore_window_seconds"),
            ({"gc_restore_window_seconds": True}, "gc_restore_window_seconds"),
            ({"gc_restore_window_seconds": 1.5}, "gc_restore_window_seconds"),
            ({"auto_gc_enabled": "no"}, "auto_gc_enabled"),
            ({"dry_run": 1}, "dry_run"),
            ({"protected_priorities": "P1"}, "protected_priorities"),
            ({"protected_priorities": [1]}, "protected_priorities"),
            ({"signal_weights": [1]}, "signal_weights"),
            ({"signal_weights": {"recency": "x"}}, "signal_weights"),
        ],
    )
    def test_invalid_values_raise(self, payload: dict[str, object], match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            _parse_hygiene(payload)


# ---------------------------------------------------------------------------
# core-18 migration + ThoughtRecord round trip
# ---------------------------------------------------------------------------


class TestCore18Migration:
    async def test_fresh_schema_has_columns_and_head_version(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            cursor = await conn.execute("PRAGMA user_version")
            assert (await cursor.fetchone())[0] == 20
            cursor = await conn.execute("PRAGMA table_info(thought)")
            cols = {row["name"] for row in await cursor.fetchall()}
            assert "pinned" in cols
            assert "archived_at_cycle" in cols
            assert "archived_at" in cols
        finally:
            await conn.close()

    async def test_v17_db_migrates_and_existing_rows_default(self) -> None:
        """An existing v17 DB migrates; existing rows read back as unpinned/NULL."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            # Bootstrap a fresh head DB, then stamp it back to v17 with a row
            # that predates the hygiene columns.
            boot = SqliteEngravaCore(conn)
            await boot.ensure_schema()
            await conn.execute("DROP TABLE thought")
            await conn.executescript(
                """
                CREATE TABLE thought (
                    thought_id        TEXT    PRIMARY KEY,
                    thought_type      TEXT    NOT NULL,
                    essence           TEXT    NOT NULL,
                    content           TEXT    NOT NULL,
                    content_hash      TEXT,
                    priority          TEXT    NOT NULL,
                    lifecycle_status  TEXT    NOT NULL DEFAULT 'CREATED',
                    created_cycle     INTEGER NOT NULL DEFAULT 0,
                    updated_cycle     INTEGER NOT NULL DEFAULT 0,
                    source            TEXT    NOT NULL DEFAULT 'human',
                    confidence        REAL,
                    embedding_ref     TEXT,
                    source_type       TEXT    NOT NULL DEFAULT 'EXPERIENCE',
                    confirmation_count INTEGER NOT NULL DEFAULT 0,
                    consolidated_from TEXT,
                    visibility        TEXT    NOT NULL DEFAULT 'selective',
                    access_count      INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at  TEXT,
                    created_at        TEXT,
                    updated_at        TEXT,
                    expires_at        TEXT,
                    metadata_json     TEXT    NOT NULL DEFAULT '{}',
                    valid_from        TEXT,
                    valid_until       TEXT,
                    action_outcome_score REAL,
                    provenance        TEXT
                );
                INSERT INTO thought (thought_id, thought_type, essence, content, priority,
                                     lifecycle_status, updated_cycle)
                VALUES ('legacy', 'OBSERVATION', 'e', 'c', 'P2', 'ACTIVE', 0);
                PRAGMA user_version = 17;
                """
            )
            await conn.commit()

            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            cursor = await conn.execute("PRAGMA user_version")
            assert (await cursor.fetchone())[0] == 20

            fetched = await s.get_thought("legacy")
            assert fetched is not None
            assert fetched.pinned is False
            assert fetched.archived_at_cycle is None
            assert fetched.archived_at is None
        finally:
            await conn.close()

    async def test_migrate_helper_idempotent(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            for _ in range(3):
                await s._migrate_core_v17_to_v18()
            cursor = await conn.execute("PRAGMA table_info(thought)")
            names = [row["name"] for row in await cursor.fetchall()]
            assert names.count("pinned") == 1
            assert names.count("archived_at_cycle") == 1
        finally:
            await conn.close()

    async def test_thought_round_trips_both_fields(self, store: SqliteEngravaCore) -> None:
        """A pinned + hygiene-archived thought survives a write/read round trip."""
        await store.create_thought(_thought("t1", pinned=True))
        fetched = await store.get_thought("t1")
        assert fetched is not None
        assert fetched.pinned is True
        assert fetched.archived_at_cycle is None

        updated = fetched.evolve(archived_at_cycle=7)
        await store.update_thought("t1", archived_at_cycle=7)
        refetched = await store.get_thought("t1")
        assert refetched is not None
        assert refetched.archived_at_cycle == 7
        assert updated.archived_at_cycle == 7


# ---------------------------------------------------------------------------
# core-20 migration + ThoughtRecord round trip (thought.archived_at)
# ---------------------------------------------------------------------------


class TestArchivedAtColumnMigration:
    """The core-20 ``thought.archived_at`` column: migration + model round trip."""

    async def test_fresh_schema_has_archived_at_and_head_version(self) -> None:
        """A fresh bootstrap lands at head v20 with the ``archived_at`` column."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            cursor = await conn.execute("PRAGMA user_version")
            assert (await cursor.fetchone())[0] == 20
            cursor = await conn.execute("PRAGMA table_info(thought)")
            cols = {row["name"] for row in await cursor.fetchall()}
            assert "archived_at" in cols
        finally:
            await conn.close()

    async def test_v19_db_migrates_and_existing_rows_default(self) -> None:
        """A v19 DB (pre-column) upgrades cleanly; existing rows read ``archived_at`` NULL."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            # Bootstrap a fresh head DB, then reshape ``thought`` back to its v19
            # form (``archived_at_cycle`` present, ``archived_at`` absent) with a
            # pre-column row, and stamp the version back to 19.
            boot = SqliteEngravaCore(conn)
            await boot.ensure_schema()
            await conn.execute("DROP TABLE thought")
            await conn.executescript(
                """
                CREATE TABLE thought (
                    thought_id        TEXT    PRIMARY KEY,
                    thought_type      TEXT    NOT NULL,
                    essence           TEXT    NOT NULL,
                    content           TEXT    NOT NULL,
                    content_hash      TEXT,
                    priority          TEXT    NOT NULL,
                    lifecycle_status  TEXT    NOT NULL DEFAULT 'CREATED',
                    created_cycle     INTEGER NOT NULL DEFAULT 0,
                    updated_cycle     INTEGER NOT NULL DEFAULT 0,
                    source            TEXT    NOT NULL DEFAULT 'human',
                    confidence        REAL,
                    embedding_ref     TEXT,
                    source_type       TEXT    NOT NULL DEFAULT 'EXPERIENCE',
                    confirmation_count INTEGER NOT NULL DEFAULT 0,
                    consolidated_from TEXT,
                    visibility        TEXT    NOT NULL DEFAULT 'selective',
                    access_count      INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at  TEXT,
                    created_at        TEXT,
                    updated_at        TEXT,
                    expires_at        TEXT,
                    metadata_json     TEXT    NOT NULL DEFAULT '{}',
                    valid_from        TEXT,
                    valid_until       TEXT,
                    action_outcome_score REAL,
                    provenance        TEXT,
                    pinned            INTEGER NOT NULL DEFAULT 0,
                    archived_at_cycle INTEGER
                );
                INSERT INTO thought (thought_id, thought_type, essence, content, priority,
                                     lifecycle_status, updated_cycle)
                VALUES ('legacy', 'OBSERVATION', 'e', 'c', 'P2', 'ACTIVE', 0);
                PRAGMA user_version = 19;
                """
            )
            await conn.commit()
            info = await conn.execute("PRAGMA table_info(thought)")
            existing_cols = {row["name"] for row in await info.fetchall()}
            assert "archived_at" not in existing_cols

            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            cursor = await conn.execute("PRAGMA user_version")
            assert (await cursor.fetchone())[0] == 20

            fetched = await s.get_thought("legacy")
            assert fetched is not None
            assert fetched.archived_at is None
        finally:
            await conn.close()

    async def test_migrate_helper_idempotent(self) -> None:
        """Running the v19->v20 migration repeatedly adds the column exactly once."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            for _ in range(3):
                await s._migrate_core_v19_to_v20()
            cursor = await conn.execute("PRAGMA table_info(thought)")
            names = [row["name"] for row in await cursor.fetchall()]
            assert names.count("archived_at") == 1
        finally:
            await conn.close()

    async def test_thought_round_trips_archived_at(self, store: SqliteEngravaCore) -> None:
        """An ``archived_at`` value survives a write/read round trip, UTC-normalised."""
        await store.create_thought(_thought("t1"))
        fetched = await store.get_thought("t1")
        assert fetched is not None
        assert fetched.archived_at is None

        await store.update_thought("t1", archived_at="2026-01-01T12:00:00+00:00")
        refetched = await store.get_thought("t1")
        assert refetched is not None
        assert refetched.archived_at == "2026-01-01T12:00:00+00:00"

        # An offset timestamp is normalised to UTC on the way in (so the GC
        # lexicographic compare stays correct).
        await store.update_thought("t1", archived_at="2026-01-01T14:00:00+02:00")
        renorm = await store.get_thought("t1")
        assert renorm is not None
        assert renorm.archived_at == "2026-01-01T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Default-OFF ⇒ no behavioural change
# ---------------------------------------------------------------------------


class TestDefaultOff:
    async def test_no_policy_run_hygiene_raises(self, store: SqliteEngravaCore) -> None:
        """A store built without a hygiene policy cannot run the pass."""
        with pytest.raises(RuntimeError, match="hygiene policy"):
            await store.run_hygiene(current_cycle=1)

    async def test_write_read_path_identical_without_hygiene(
        self, store: SqliteEngravaCore
    ) -> None:
        """Creating + reading a thought is unchanged with no hygiene policy."""
        await store.create_thought(_thought("t1", priority=Priority.P2))
        fetched = await store.get_thought("t1")
        assert fetched is not None
        assert fetched.lifecycle_status is LifecycleStatus.ACTIVE
        assert fetched.pinned is False
        assert fetched.archived_at_cycle is None

    async def test_disabled_policy_consolidate_does_not_forget(self) -> None:
        """With a policy present but disabled, consolidate() never archives.

        The pool is the archivable one built by :func:`_evictable_thought`, and
        the cadence is satisfied (``1000 % 1 == 0``), so ``enabled`` is the only
        variable — flipping it on at the end archives the same row on the same
        cycle, which is what proves nothing else was quietly protecting it.
        """
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        from engrava.extensions.dreaming import DreamingExtension

        s = SqliteEngravaCore(
            conn,
            hygiene_policy=HygienePolicyConfig(
                enabled=False, eviction_threshold=1.0, check_every_n_cycles=1
            ),
        )
        s._dreaming_extension = DreamingExtension(config=DreamingConfig(enabled=True))
        await s.ensure_schema()
        try:
            await s.create_thought(_evictable_thought("cold"))

            # Spy (``wraps`` — the real pass still runs) so the rejection is
            # observable on the path itself: the pass is never *invoked*, not
            # merely inert once invoked.
            with mock.patch.object(s, "run_hygiene", wraps=s.run_hygiene) as pass_spy:
                await s.consolidate(current_cycle=1000)

                assert await _raw_lifecycle(s, "cold") == "ACTIVE"
                assert await _raw_archived_at_cycle(s, "cold") is None
                assert await _raw_archived_at(s, "cold") is None
                assert pass_spy.call_count == 0
                # The cadence gate carries its own ``enabled`` term — the second,
                # independent guard on this path.
                assert s._hygiene_due(1000) is False

                # Flip the single variable: same store, same pool, same cycle.
                s._hygiene_policy = HygienePolicyConfig(
                    enabled=True, eviction_threshold=1.0, check_every_n_cycles=1
                )
                await s.consolidate(current_cycle=1000)
                assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
                assert await _raw_archived_at_cycle(s, "cold") == 1000
                assert pass_spy.call_count == 1
        finally:
            await conn.close()

    async def test_disabled_policy_direct_run_hygiene_is_a_noop(self) -> None:
        """``enabled`` is a hard master switch — a direct call also never forgets.

        ``run_hygiene`` bypasses the cadence gate, so the master switch is the
        *only* guard here: the thought is aged past the default 7-day
        minimum-inactivity window, carries the usage signal the access-gate
        needs, and scores under the threshold. Flipping ``enabled`` on at the end
        archives it, so none of those gates is doing the work.
        """
        policy = HygienePolicyConfig(enabled=False, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_evictable_thought("cold"))

            result = await s.run_hygiene(current_cycle=1000, now=_NOW)

            assert await _raw_lifecycle(s, "cold") == "ACTIVE"
            assert await _raw_archived_at_cycle(s, "cold") is None
            assert await _raw_archived_at(s, "cold") is None
            assert result.archived_count == 0
            assert result.gc_count == 0

            s._hygiene_policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
            enabled_run = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            assert enabled_run.archived_count == 1
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Keep-score + eviction rule (+ redistribution)
# ---------------------------------------------------------------------------


class TestKeepScoreAndEviction:
    async def test_cold_low_value_thought_archived(self) -> None:
        policy = _forgetful_policy(eviction_threshold=0.5)
        s = await _make_store(policy)
        try:
            # Old, never-accessed, low confidence -> low keep-score. A usage
            # signal (action_outcome) opens the access-gate without changing the
            # keep-score.
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_recently_accessed_thought_survives(self) -> None:
        """A thought that scores high on recency stays regardless of the threshold."""
        policy = _forgetful_policy(eviction_threshold=0.5)
        s = await _make_store(policy)
        try:
            # updated_cycle == current_cycle -> recency ~= 1.0.
            await s.create_thought(_thought("warm", updated_cycle=1000))
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=1000)
            assert await _raw_lifecycle(s, "warm") == "ACTIVE"
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            assert result.archived_count == 1
        finally:
            await s._db.close()

    async def test_created_lifecycle_is_a_candidate(self) -> None:
        """CREATED thoughts are in the candidate set (not only ACTIVE)."""
        policy = _forgetful_policy(eviction_threshold=0.5)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought(
                    "created",
                    lifecycle_status=LifecycleStatus.CREATED,
                    updated_cycle=0,
                    action_outcome_score=0.0,
                )
            )
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "created") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_terminal_thought_not_reprocessed(self) -> None:
        """An already-ARCHIVED thought is outside the candidate set."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("done", lifecycle_status=LifecycleStatus.ARCHIVED, updated_cycle=0)
            )
            result = await s.run_hygiene(current_cycle=1000)
            assert result.candidates_evaluated == 0
            assert result.archived_count == 0
        finally:
            await s._db.close()

    async def test_candidate_scan_spans_multiple_pages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every candidate is scored even when the pool spans several scan pages.

        The per-run cap bounds the *archived* set, not the *considered* set, so a
        thought beyond the first scan page must still be evaluated (and archived
        if cold).
        """
        # One-row pages force the full ACTIVE pool to be walked page by page.
        monkeypatch.setattr(engrava_core, "_ORPHAN_SWEEP_PAGE_SIZE", 1)
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            for i in range(5):
                await s.create_thought(_thought(f"t{i}", updated_cycle=0, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.candidates_evaluated == 5
            assert result.archived_count == 5
        finally:
            await s._db.close()

    async def test_redistribution_reports_flat_signals(self) -> None:
        """Signals with no data source are flat and reported in the result."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=0.5)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("t", updated_cycle=0))
            result = await s.run_hygiene(current_cycle=100)
            # No access / confirmations / confidence in the pool -> those signals
            # are flat; recency + staleness (cycle-based) remain active.
            assert set(result.flat_signals) == {"confidence", "confirmation", "frequency"}
        finally:
            await s._db.close()

    def test_compute_active_weights_renormalises(self) -> None:
        """Active weights renormalise to 1.0 over the active signals."""
        thoughts = [_thought("t", updated_cycle=0)]
        weights, flat = compute_active_hygiene_weights(
            HygienePolicyConfig().signal_weights,
            thoughts,
            current_cycle=10,
            access_tracking_enabled=False,
        )
        active_sum = sum(w for w in weights.values() if w > 0)
        assert active_sum == pytest.approx(1.0)
        assert "frequency" in flat  # access tracking off -> flat


# ---------------------------------------------------------------------------
# Protection: pinned, P1, confidence-is-not-protection
# ---------------------------------------------------------------------------


class TestProtection:
    async def test_pinned_never_archived_even_below_threshold(self) -> None:
        # Cold-start gates open (age gate off + a usage signal in the pool) so the
        # pin is the *sole* reason the row is not archived, not a gate short-circuit.
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("pin", pinned=True, updated_cycle=0, action_outcome_score=0.0)
            )
            result = await s.run_hygiene(current_cycle=10_000)
            assert result.archived_count == 0
            assert await _raw_lifecycle(s, "pin") == "ACTIVE"
        finally:
            await s._db.close()

    async def test_p1_protected_by_default(self) -> None:
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("p1", priority=Priority.P1, updated_cycle=0))
            await s.create_thought(
                _thought("p3", priority=Priority.P3, updated_cycle=0, action_outcome_score=0.0)
            )
            await s.run_hygiene(current_cycle=10_000)
            assert await _raw_lifecycle(s, "p1") == "ACTIVE"
            assert await _raw_lifecycle(s, "p3") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_confidence_is_not_protection(self) -> None:
        """A high-confidence but cold thought is NOT protected by confidence."""
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought(
                    "confident",
                    confidence=1.0,
                    priority=Priority.P3,
                    updated_cycle=0,
                    action_outcome_score=0.0,
                )
            )
            result = await s.run_hygiene(current_cycle=10_000)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "confident") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_protected_priorities_configurable_off(self) -> None:
        """Setting protected_priorities to () lets P1 be archived."""
        policy = _forgetful_policy(eviction_threshold=1.0, protected_priorities=())
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("p1", priority=Priority.P1, updated_cycle=0, action_outcome_score=0.0)
            )
            result = await s.run_hygiene(current_cycle=10_000)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "p1") == "ARCHIVED"
        finally:
            await s._db.close()

    def test_hygiene_protected_helper(self) -> None:
        policy = HygienePolicyConfig(protected_priorities=("P1",))
        assert _hygiene_protected(_thought("a", pinned=True, priority=Priority.P3), policy)
        assert _hygiene_protected(_thought("b", priority=Priority.P1), policy)
        assert not _hygiene_protected(_thought("c", priority=Priority.P3, confidence=1.0), policy)

    async def test_archive_write_time_recheck_skips_now_protected(self) -> None:
        """The **pin** is what skips a now-protected thought at the archive write (TOCTOU).

        Both rows go straight to the archive stage — selection already skips
        protected thoughts, so only the write-time re-check is under test — and
        both are aged through ``updated_at``, not ``created_at`` alone. Ageing by
        ``created_at`` alone would leave this test unable to observe the pin:
        ``create_thought`` stamps a ``None`` ``updated_at`` to *now*, and
        ``updated_at`` outranks ``created_at`` in the COALESCE ladder
        (``last_accessed_at`` → ``updated_at`` → ``created_at``), so the row stays
        inside the inactivity window and the **minimum-inactivity-age** re-check
        performs the skip the pin is supposed to perform.

        With ``updated_at`` at ``_LONG_AGO`` the age gate cannot be what skips
        either row — it passes ``unpinned``, and for ``pinned_late`` it is never
        reached at all (below). Both rows are ``ACTIVE`` and ``P3`` is outside the
        default ``protected_priorities``, so ``pinned`` is the only difference
        left between them.

        **What is covered, and what is not — a documented limit.**
        ``_hygiene_protected`` is the first term of the write-time ``or`` chain,
        so ``pinned_late`` is skipped by the **Python re-check** and the age
        predicate is never evaluated for it (it would pass if it were reached,
        which is not the same as being what this test observes). The row
        therefore never reaches the predicate-guarded ``UPDATE``, so that
        statement's ``pinned = 0`` clause is **not** independently observable
        here — delete the clause and this test, and this whole file, stay green.
        Reaching it needs a row that clears the Python re-check and is only then
        found pinned — a genuine check-to-write race, which this file does not
        set up. The machinery for one already exists: ``_interleave_once`` in
        ``tests/test_partial_field_updates.py`` fires an intruder immediately
        after a wrapped store read returns, which is exactly where a pin would
        have to land for the re-check to act on an unpinned snapshot while the
        ``UPDATE`` meets the pinned row.

        ``unpinned`` is the control: the same call must leave it ``ARCHIVED``,
        which is what proves the stage ran. A stage that did nothing at all would
        satisfy "``pinned_late`` is still ``ACTIVE``" just as well.
        """
        from engrava.infrastructure.sqlite.hygiene import EvictionReason

        policy = HygienePolicyConfig(enabled=True)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought(
                    "pinned_late",
                    pinned=True,
                    updated_cycle=0,
                    created_at=_LONG_AGO,
                    updated_at=_LONG_AGO,
                )
            )
            await s.create_thought(
                _thought(
                    "unpinned",
                    updated_cycle=0,
                    created_at=_LONG_AGO,
                    updated_at=_LONG_AGO,
                )
            )
            reasons = [
                EvictionReason(
                    thought_id=thought_id,
                    keep_score=0.0,
                    eviction_score=0.0,
                    decay_multiplier=1.0,
                    threshold=0.2,
                )
                for thought_id in ("pinned_late", "unpinned")
            ]
            now = datetime.datetime.now(datetime.UTC)
            archived = await s._hygiene_archive(reasons, policy=policy, current_cycle=1000, now=now)
            assert await _raw_lifecycle(s, "pinned_late") == "ACTIVE"
            assert await _raw_lifecycle(s, "unpinned") == "ARCHIVED"
            assert archived == 1
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Archive reversibility + restore clears archived_at_cycle
# ---------------------------------------------------------------------------


class TestArchiveReversible:
    async def test_archive_sets_archived_at_cycle_and_clears_expires(self) -> None:
        """Archival stamps both hygiene markers and drops the row's TTL.

        Both thoughts are created *under TTL* — otherwise ``expires_at`` is
        already ``NULL`` before the pass and the clear cannot be observed. The
        pinned row is the non-target: the clear must reach the archived row and
        only the archived row, so a blanket TTL wipe fails this test.
        """
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("cold", updated_cycle=0, action_outcome_score=0.0, expires_at=_FAR_FUTURE)
            )
            await s.create_thought(_thought("keep", pinned=True, expires_at=_FAR_FUTURE))
            assert await _raw_expires_at(s, "cold") == _FAR_FUTURE  # under TTL before the pass

            await s.run_hygiene(current_cycle=42, now=_NOW)

            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            assert await _raw_archived_at_cycle(s, "cold") == 42
            assert await _raw_archived_at(s, "cold") == _NOW.isoformat()
            # No longer subject to TTL — see the delete-strategy test below for
            # what the surviving TTL would have cost.
            assert await _raw_expires_at(s, "cold") is None
            # The non-target keeps its lifecycle, its TTL and its NULL markers.
            assert await _raw_lifecycle(s, "keep") == "ACTIVE"
            assert await _raw_expires_at(s, "keep") == _FAR_FUTURE
            assert await _raw_archived_at_cycle(s, "keep") is None
        finally:
            await s._db.close()

    async def test_archived_row_survives_a_later_ttl_delete_sweep(self) -> None:
        """The cleared TTL is what keeps a reversibly-archived row out of the sweep.

        ``cleanup_expired`` has no lifecycle filter, so a hygiene-archived row
        that kept its ``expires_at`` would be **physically deleted** under
        ``ttl_strategy="delete"`` — bypassing both GC restore windows and the
        ``auto_gc_enabled`` switch. The pinned control row proves the sweep is
        live and really does delete what is still under TTL.
        """
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy, ttl_strategy="delete")
        try:
            await s.create_thought(_evictable_thought("cold", expires_at=_FAR_FUTURE))
            # Pinned, so hygiene never archives it: it keeps its TTL and is the
            # non-target the sweep is expected to reap.
            await s.create_thought(_thought("under_ttl", pinned=True, expires_at=_FAR_FUTURE))

            await s.run_hygiene(current_cycle=42, now=_NOW)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            # The non-target came through archival untouched, TTL included —
            # so its later deletion is the sweep's doing, not hygiene's.
            assert await _raw_lifecycle(s, "under_ttl") == "ACTIVE"
            assert await _raw_expires_at(s, "under_ttl") == _FAR_FUTURE

            result = await s.cleanup_expired(now=_AFTER_FAR_FUTURE)

            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"  # not deleted
            assert await _raw_lifecycle(s, "under_ttl") is None  # the sweep is live
            assert result.expired_count == 1
            assert result.strategy_applied == "delete"
        finally:
            await s._db.close()

    async def test_restore_clears_archived_at_cycle(self) -> None:
        """restore_thought un-archives a hygiene-archived thought and clears the stamp."""
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=42)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"

            restored = await s.restore_thought("cold")

            assert restored.lifecycle_status is LifecycleStatus.ACTIVE
            assert restored.archived_at_cycle is None
            assert await _raw_lifecycle(s, "cold") == "ACTIVE"
            assert await _raw_archived_at_cycle(s, "cold") is None
        finally:
            await s._db.close()

    async def test_archived_thought_transitions_to_active_via_state_machine(self) -> None:
        """The ARCHIVED -> ACTIVE edge exists, so restore is not a bypass.

        Regression: ARCHIVED was terminal (zero outbound transitions), so the
        'reversible archive' contract only held by writing the raw string value
        and skipping the transition check. The edge now exists, so evolving with
        the ACTIVE *enum* (which runs the state-machine check) no longer raises.
        """
        assert LifecycleStatus.ARCHIVED.can_transition_to(LifecycleStatus.ACTIVE)
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=42)
            updated = await s.update_thought("cold", lifecycle_status=LifecycleStatus.ACTIVE)
            assert updated.lifecycle_status is LifecycleStatus.ACTIVE
        finally:
            await s._db.close()

    async def test_restore_round_trip_preserves_content(self) -> None:
        """archive -> restore round-trips with no data loss."""
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            original = _thought("cold", updated_cycle=0, action_outcome_score=0.0)
            await s.create_thought(original)
            await s.run_hygiene(current_cycle=42)
            restored = await s.restore_thought("cold")
            assert restored.essence == original.essence
            assert restored.content == original.content
            assert restored.metadata == original.metadata
        finally:
            await s._db.close()

    async def test_restore_non_archived_raises(self) -> None:
        """Restoring a thought that is not ARCHIVED raises InvalidTransitionError."""
        s = await _make_store(None)
        try:
            await s.create_thought(_thought("active", updated_cycle=0))
            with pytest.raises(InvalidTransitionError):
                await s.restore_thought("active")
        finally:
            await s._db.close()

    async def test_restore_missing_raises(self) -> None:
        """Restoring an unknown thought raises ThoughtNotFoundError."""
        s = await _make_store(None)
        try:
            with pytest.raises(ThoughtNotFoundError):
                await s.restore_thought("ghost")
        finally:
            await s._db.close()

    async def test_restore_is_journaled_and_chain_verifies(self) -> None:
        """The restore writes exactly one UPDATE_THOUGHT entry; the chain verifies."""
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=42)

            async def _journal_count() -> int:
                cur = await s._db.execute("SELECT COUNT(*) FROM journal_entry")
                row = await cur.fetchone()
                assert row is not None
                return int(row[0])

            before = await _journal_count()
            await s.restore_thought("cold")

            # The restore actually journaled (one appended entry, not a no-op)...
            assert await _journal_count() == before + 1
            cur = await s._db.execute(
                "SELECT mutation_type FROM journal_entry ORDER BY sequence_number DESC LIMIT 1"
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "UPDATE_THOUGHT"
            # ...and the chain still verifies after the restore.
            result = await s.verify_journal()
            assert result.valid is True
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Deterministic capped selection
# ---------------------------------------------------------------------------


class TestDeterministicCappedSelection:
    async def test_same_inputs_same_eviction_set(self) -> None:
        """Two stores with identical content + config + cycle evict the same ids."""
        ids_seen: list[list[str]] = []
        for _ in range(2):
            policy = _forgetful_policy(
                eviction_threshold=0.5, max_evictions_per_run=3, dry_run=True
            )
            s = await _make_store(policy)
            try:
                for i in range(10):
                    await s.create_thought(
                        _thought(f"t{i:02d}", updated_cycle=0, action_outcome_score=0.0)
                    )
                result = await s.run_hygiene(current_cycle=1000)
                ids_seen.append([r.thought_id for r in result.would_evict])
            finally:
                await s._db.close()
        assert ids_seen[0] == ids_seen[1]

    async def test_cap_enforced_per_stage(self) -> None:
        """No more than max_evictions_per_run thoughts are archived per run."""
        policy = _forgetful_policy(eviction_threshold=0.9, max_evictions_per_run=2)
        s = await _make_store(policy)
        try:
            for i in range(5):
                await s.create_thought(_thought(f"t{i}", updated_cycle=0, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 2
        finally:
            await s._db.close()

    async def test_cap_selects_the_coldest_candidates(self) -> None:
        """Under the cap the *coldest* thoughts are archived, not an arbitrary set."""
        policy = _forgetful_policy(eviction_threshold=1.0, max_evictions_per_run=2, dry_run=True)
        s = await _make_store(policy)
        try:
            # Different recency -> different eviction-score; the two oldest
            # (lowest keep-score) must be the ones selected under a cap of 2.
            for tid, updated in (("warm", 1000), ("mid", 500), ("cold1", 0), ("cold2", 10)):
                await s.create_thought(
                    _thought(tid, updated_cycle=updated, action_outcome_score=0.0)
                )
            result = await s.run_hygiene(current_cycle=1000)
            assert sorted(r.thought_id for r in result.would_evict) == ["cold1", "cold2"]
        finally:
            await s._db.close()

    async def test_tiebreak_order_is_stable(self) -> None:
        """Under equal scores the id tiebreak makes the selection deterministic."""
        # All ten thoughts share updated_cycle and score identically -> the
        # eviction_score + updated_cycle tie is broken by thought_id ASC.
        policy = _forgetful_policy(eviction_threshold=0.9, max_evictions_per_run=3, dry_run=True)
        s = await _make_store(policy)
        try:
            for i in range(10):
                await s.create_thought(
                    _thought(f"id-{i:02d}", updated_cycle=0, action_outcome_score=0.0)
                )
            result = await s.run_hygiene(current_cycle=1000)
            selected = [r.thought_id for r in result.would_evict]
            assert selected == ["id-00", "id-01", "id-02"]
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# All-flat fail-safe
# ---------------------------------------------------------------------------


class TestAllFlatFailSafe:
    async def test_all_flat_signals_archive_nothing(self) -> None:
        """When no signal is active the pass archives nothing (fail-safe)."""
        # current_cycle=None-equivalent: pass cycle 0 with brand-new thoughts so
        # recency/staleness carry no span, and clear the cycle-based signals by
        # setting only ``frequency`` as the sole weight with access tracking off.
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=1.0,
            signal_weights={"frequency": 1.0},
        )
        s = await _make_store(policy)
        try:
            # access_tracking is off (no dreaming) -> frequency is the only
            # configured signal and it is flat -> empty active set. The usage
            # signal opens the access-gate so the empty-active-set fail-safe (not
            # the access-gate) is the sole reason nothing is archived.
            await s.create_thought(_thought("t", updated_cycle=0, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 0
            assert await _raw_lifecycle(s, "t") == "ACTIVE"
            assert result.flat_signals == ["frequency"]
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Decay clamp
# ---------------------------------------------------------------------------


class _DecayHooks:
    """Hooks whose ``decay_function`` returns a fixed (possibly out-of-range) value."""

    def __init__(self, value: float) -> None:
        self._value = value

    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        return thought

    async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
        return thought

    async def score_function(self, thought: ThoughtRecord, context: object) -> float:
        return 1.0

    async def decay_function(self, thought: ThoughtRecord, elapsed_cycles: int) -> float:
        return self._value

    def mindql_extension_registry(self) -> dict[str, object]:
        return {}


class TestDecayClamp:
    def test_clamp_helper(self) -> None:
        assert _clamp_decay(0.5) == pytest.approx(0.5)
        assert _clamp_decay(-1.0) == pytest.approx(0.0)
        assert _clamp_decay(2.0) == pytest.approx(1.0)
        assert _clamp_decay(math.nan) == pytest.approx(1.0)
        assert _clamp_decay(math.inf) == pytest.approx(1.0)
        assert _clamp_decay(-math.inf) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad_decay", [math.nan, math.inf, -math.inf, 5.0])
    async def test_nonfinite_or_high_decay_never_over_evicts(self, bad_decay: float) -> None:
        """A decay hook returning >1 / NaN / inf clamps to 1.0 (never extra evict).

        A thought whose keep-score is above the threshold must stay even when the
        hook misbehaves: the clamp caps decay at 1.0, so eviction_score never
        exceeds keep_score and a would-be-kept thought is never archived.
        """
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        policy = _forgetful_policy(eviction_threshold=0.5)
        s = SqliteEngravaCore(conn, hooks=_DecayHooks(bad_decay), hygiene_policy=policy)
        await s.ensure_schema()
        try:
            # A warm thought (recency ~1.0) has keep_score above threshold. The
            # usage signal opens the access-gate so the decay clamp is what keeps
            # the row, not a gate short-circuit.
            await s.create_thought(_thought("warm", updated_cycle=1000, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 0
            assert await _raw_lifecycle(s, "warm") == "ACTIVE"
        finally:
            await conn.close()

    async def test_negative_decay_clamps_to_zero_not_below(self) -> None:
        """A negative decay clamps to 0.0 — eviction_score never goes negative."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        policy = _forgetful_policy(eviction_threshold=0.5)
        s = SqliteEngravaCore(conn, hooks=_DecayHooks(-2.0), hygiene_policy=policy)
        await s.ensure_schema()
        try:
            # Decay 0 -> eviction_score 0 < threshold -> archived (but not an
            # error, and the score is a clean 0.0 in the reason).
            await s.create_thought(_thought("warm", updated_cycle=1000, action_outcome_score=0.0))
            policy_dry = _forgetful_policy(eviction_threshold=0.5, dry_run=True)
            s._hygiene_policy = policy_dry
            result = await s.run_hygiene(current_cycle=1000)
            assert len(result.would_evict) == 1
            assert result.would_evict[0].decay_multiplier == pytest.approx(0.0)
            assert result.would_evict[0].eviction_score == pytest.approx(0.0)
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# Stage 2 — GC
# ---------------------------------------------------------------------------


class TestGarbageCollection:
    async def test_gc_disabled_by_default_only_archives(self) -> None:
        """Enabling hygiene never implicitly enables deletion.

        The row is archived on the first pass and then left to age past **both**
        restore windows before the second pass, so ``auto_gc_enabled`` is the
        only thing still standing between it and a permanent delete — flipping
        that switch on at the end reaps it, which is what proves the windows had
        really elapsed rather than the test having stopped short of them.
        """
        policy = _forgetful_policy(eviction_threshold=1.0, auto_gc_enabled=False)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            # Pinned, never archived: the non-target the GC flip must not touch.
            await s.create_thought(_thought("keep", pinned=True, updated_cycle=0))
            archive_run = await s.run_hygiene(current_cycle=5, now=_NOW)

            # Cycle window elapsed (1000 - 5 >= the default 10) and wall-clock
            # window elapsed (31 days >= the default 30).
            later = _NOW + datetime.timedelta(seconds=_MONTH_SECONDS + 86400)
            gc_run = await s.run_hygiene(current_cycle=1000, now=later)

            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"  # archived, not deleted
            assert await _raw_archived_at_cycle(s, "cold") == 5
            assert archive_run.archived_count == 1
            assert gc_run.gc_count == 0

            # Flip the single variable: same store, same cycle, same ``now``.
            s._hygiene_policy = _forgetful_policy(eviction_threshold=1.0, auto_gc_enabled=True)
            enabled_run = await s.run_hygiene(current_cycle=1000, now=later)
            assert await _raw_lifecycle(s, "cold") is None  # physically gone
            assert await _raw_lifecycle(s, "keep") == "ACTIVE"  # non-target survives
            assert enabled_run.gc_count == 1
        finally:
            await s._db.close()

    async def test_gc_respects_restore_window(self) -> None:
        """A just-archived thought is not GC'd until the cycle window elapses.

        The wall-clock window is disabled (``gc_restore_window_seconds=0``) so
        this exercises the cycle window in isolation.
        """
        policy = _forgetful_policy(
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
            gc_restore_window_seconds=0,
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            r1 = await s.run_hygiene(current_cycle=5)  # archives at cycle 5
            assert r1.archived_count == 1
            assert r1.gc_count == 0
            assert await s.get_thought("cold") is not None  # window 5-5=0 < 10

            r2 = await s.run_hygiene(current_cycle=20)  # 20-5=15 >= 10 -> GC
            assert r2.gc_count == 1
            assert await s.get_thought("cold") is None
        finally:
            await s._db.close()

    async def test_gc_only_reaps_hygiene_archived_rows(self) -> None:
        """A TTL/manually-archived row (archived_at_cycle NULL) is never auto-GC'd."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,  # archive nothing new
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        s = await _make_store(policy)
        try:
            # Manually archive without setting archived_at_cycle (TTL/manual path).
            await s.create_thought(_thought("manual", updated_cycle=0))
            await s.update_thought("manual", lifecycle_status=LifecycleStatus.ARCHIVED)
            assert await _raw_archived_at_cycle(s, "manual") is None

            result = await s.run_hygiene(current_cycle=1000)
            assert result.gc_count == 0
            assert await s.get_thought("manual") is not None  # untouched by hygiene GC
        finally:
            await s._db.close()

    async def test_gc_cap_not_starved_by_protected_archived(self) -> None:
        """A protected hygiene-archived row does not consume a GC cap slot."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,  # archive nothing new
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=1,
            gc_restore_window_seconds=0,
            # One slot: a pinned, older-archived row must not consume it and
            # starve the younger eligible row (protection is excluded in SQL).
            max_evictions_per_run=1,
        )
        s = await _make_store(policy)
        try:
            for tid, arch_cycle, is_pinned in (
                ("old_pinned", 5, True),
                ("young", 6, False),
            ):
                await s.create_thought(
                    _thought(tid, lifecycle_status=LifecycleStatus.ARCHIVED, pinned=is_pinned)
                )
                await s._db.execute(
                    "UPDATE thought SET archived_at_cycle = ? WHERE thought_id = ?",
                    (arch_cycle, tid),
                )
            result = await s.run_hygiene(current_cycle=1000)
            assert result.gc_count == 1  # young reaped, not starved by the protected row
            assert await s.get_thought("young") is None
            assert await s.get_thought("old_pinned") is not None
        finally:
            await s._db.close()

    async def test_gc_never_reaps_pinned(self) -> None:
        """A pinned thought is never GC'd even if somehow archived past the window."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        s = await _make_store(policy)
        try:
            # Pinned + archived-by-hygiene bookkeeping present, window elapsed.
            await s.create_thought(_thought("pin", pinned=True, updated_cycle=0))
            await s.update_thought(
                "pin",
                lifecycle_status=LifecycleStatus.ARCHIVED,
                archived_at_cycle=0,
            )
            result = await s.run_hygiene(current_cycle=1000)
            assert result.gc_count == 0
            assert await s.get_thought("pin") is not None
        finally:
            await s._db.close()

    async def test_gc_orphan_sweep_runs_before_delete(self) -> None:
        """The orphan-reflection sweep retires danglers before the GC delete."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,  # archive nothing new via score
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        s = await _make_store(policy)
        try:
            # A source OBSERVATION hygiene-archived past the window (GC target).
            await s.create_thought(_thought("src", updated_cycle=0))
            await s.update_thought(
                "src",
                lifecycle_status=LifecycleStatus.ARCHIVED,
                archived_at_cycle=0,
            )
            # A REFLECTION consolidated only from that (now non-live) source.
            await s.create_thought(
                _thought("refl", thought_type=ThoughtType.REFLECTION, updated_cycle=0)
            )
            await s.create_edge(
                EdgeRecord(
                    edge_id="e1",
                    from_thought_id="refl",
                    to_thought_id="src",
                    edge_type=EdgeType.CONSOLIDATED_FROM,
                    weight=0.5,
                    created_cycle=0,
                    source=KnowledgeSource.EXPERIENCE,
                )
            )
            result = await s.run_hygiene(current_cycle=1000)
            # The source is GC'd; the orphan REFLECTION was retired before the
            # delete, so no dangling CONSOLIDATED_FROM edge is left.
            assert result.gc_count == 1
            assert await s.get_thought("src") is None
            assert await _raw_lifecycle(s, "refl") == "ARCHIVED"
            cursor = await s._db.execute("SELECT COUNT(*) AS n FROM edge")
            assert (await cursor.fetchone())["n"] == 0
        finally:
            await s._db.close()

    @pytest.mark.skipif(
        importlib.util.find_spec("sqlite_vec") is None,
        reason="sqlite-vec package not installed",
    )
    async def test_gc_purges_vec0_vector(self, tmp_path: Path) -> None:
        """The vec0 vector for a GC'd thought is purged (no ghost row)."""
        db_path = tmp_path / "hygiene_vec.db"
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        s = SqliteEngravaCore(conn, hygiene_policy=policy)
        s._owns_connection = True
        await s.ensure_schema()
        await s._configure_vector_backend(backend_name="sqlite-vec", embedding_dimension=3)
        try:
            await s.create_thought(_thought("vt", updated_cycle=0))
            await s.update_thought(
                "vt",
                lifecycle_status=LifecycleStatus.ARCHIVED,
                archived_at_cycle=0,
            )
            await s.store_embedding(thought_id="vt", vector=[0.1, 0.2, 0.3], model_name="m")

            vec_rowid = await s._embedding_rowid_for_thought("vt")
            assert vec_rowid is not None
            cursor = await conn.execute(
                "SELECT COUNT(*) AS n FROM embedding_vec WHERE rowid = ?", (vec_rowid,)
            )
            assert (await cursor.fetchone())["n"] == 1

            result = await s.run_hygiene(current_cycle=1000)
            assert result.gc_count == 1
            assert await s.get_thought("vt") is None
            # The vec0 vector must be gone — no ghost, no reserved slot.
            cursor = await conn.execute(
                "SELECT COUNT(*) AS n FROM embedding_vec WHERE rowid = ?", (vec_rowid,)
            )
            assert (await cursor.fetchone())["n"] == 0
        finally:
            await s.close()

    async def test_journal_valid_after_archive_and_gc(self) -> None:
        """The hash-chain still verifies after an archive + GC in the same store."""
        policy = _forgetful_policy(
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
            gc_restore_window_seconds=0,
        )
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=5)  # archive
            await s.run_hygiene(current_cycle=20)  # GC
            integrity = await s.verify_journal()
            assert integrity.valid is True
            assert integrity.entries_checked >= 3
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------


class TestDryRun:
    async def test_dry_run_mutates_nothing(self) -> None:
        policy = _forgetful_policy(eviction_threshold=1.0, dry_run=True)
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=1000)
            # Nothing archived, nothing GC'd.
            assert result.archived_count == 0
            assert result.gc_count == 0
            assert result.dry_run is True
            # Store unchanged.
            assert await _raw_lifecycle(s, "cold") == "ACTIVE"
            assert await _raw_archived_at_cycle(s, "cold") is None
            # Would-evict returned with a per-thought reason.
            assert len(result.would_evict) == 1
            reason = result.would_evict[0]
            assert reason.thought_id == "cold"
            assert reason.mechanism == "hygiene"
            assert 0.0 <= reason.eviction_score <= 1.0
            # Journal untouched (nothing mutated -> no entries).
            entries = await s.journal.get_entries()
            hygiene_entries = [
                e for e in entries if isinstance(e.delta, dict) and "eviction_reason" in e.delta
            ]
            assert hygiene_entries == []
        finally:
            await s._db.close()

    async def test_dry_run_reason_carries_signal_breakdown(self) -> None:
        policy = _forgetful_policy(eviction_threshold=1.0, dry_run=True)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            result = await s.run_hygiene(current_cycle=100)
            reason = result.would_evict[0]
            # recency + staleness are the active signals this run.
            assert "recency" in reason.signals
            assert "staleness" in reason.signals
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Journal eviction_reason + no new mutation type
# ---------------------------------------------------------------------------


class TestJournalEvictionReason:
    async def test_archive_uses_update_thought_with_nested_reason(self) -> None:
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=7)
            entries = await s.journal.get_entries()
            archive_entries = [
                e
                for e in entries
                if e.mutation_type == "UPDATE_THOUGHT"
                and isinstance(e.delta, dict)
                and "eviction_reason" in e.delta
            ]
            assert len(archive_entries) == 1
            reason = archive_entries[0].delta["eviction_reason"]
            assert reason["mechanism"] == "hygiene"
            assert "keep_score" in reason
            assert "eviction_score" in reason
            assert "decay_multiplier" in reason
            assert "threshold" in reason
            assert "signals" in reason
        finally:
            await s._db.close()

    async def test_gc_uses_delete_thought_with_reason(self) -> None:
        policy = _forgetful_policy(
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=5)  # archive
            await s.run_hygiene(current_cycle=6)  # GC (window 0)
            entries = await s.journal.get_entries()
            gc_entries = [
                e
                for e in entries
                if e.mutation_type == "DELETE_THOUGHT"
                and isinstance(e.delta, dict)
                and "eviction_reason" in e.delta
            ]
            assert len(gc_entries) == 1
            assert gc_entries[0].delta["eviction_reason"]["mechanism"] == "hygiene"
        finally:
            await s._db.close()

    async def test_no_new_mutation_type_introduced(self) -> None:
        """Every hygiene journal entry uses an existing mutation-type value."""
        from engrava.domain.models.mutation_type import MutationType

        policy = _forgetful_policy(
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
        )
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=5)
            await s.run_hygiene(current_cycle=6)
            entries = await s.journal.get_entries()
            valid = {m.value for m in MutationType}
            for e in entries:
                assert e.mutation_type in valid
        finally:
            await s._db.close()

    def test_eviction_reason_to_delta_shape(self) -> None:
        reason = EvictionReason(
            thought_id="t1",
            keep_score=0.1,
            eviction_score=0.08,
            decay_multiplier=0.8,
            threshold=0.2,
            signals={"recency": 0.1},
        )
        delta = reason.to_delta()
        assert delta == {
            "mechanism": "hygiene",
            "keep_score": 0.1,
            "eviction_score": 0.08,
            "decay_multiplier": 0.8,
            "threshold": 0.2,
            "signals": {"recency": 0.1},
        }


# ---------------------------------------------------------------------------
# consolidate() convenience invocation + cadence
# ---------------------------------------------------------------------------


class TestConsolidateInvocation:
    async def test_consolidate_runs_hygiene_on_cadence(self) -> None:
        """With dreaming + hygiene enabled, consolidate() forgets on-cadence."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        from engrava.extensions.dreaming import DreamingExtension

        policy = _forgetful_policy(eviction_threshold=1.0, check_every_n_cycles=1)
        s = SqliteEngravaCore(conn, hygiene_policy=policy)
        s._dreaming_extension = DreamingExtension(config=DreamingConfig(enabled=True))
        await s.ensure_schema()
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.consolidate(current_cycle=1000)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
        finally:
            await conn.close()

    async def test_consolidate_skips_hygiene_off_cadence(self) -> None:
        """The **cadence** is what stops ``consolidate()`` forgetting on an off-cadence cycle.

        The row comes from :func:`_evictable_thought`, so every other gate is
        already satisfied — including the minimum-inactivity-age gate, which this
        policy leaves at its 7-day default. Clearing that gate needs
        ``updated_at``, not ``created_at`` alone: ``create_thought`` stamps a
        ``None`` ``updated_at`` to *now* and ``updated_at`` outranks
        ``created_at`` in the COALESCE ladder, so a row aged only by
        ``created_at`` sits inside the window and it is the **age** gate, not the
        cadence, that keeps it ``ACTIVE`` on cycle 7.

        The second, on-cadence ``consolidate()`` is the control: the same row
        under the same policy comes out ``ARCHIVED`` on cycle 10, which rules out
        a dead pass — or a row nothing would ever evict — as the reason cycle 7
        left it alone. Its limit is worth naming too: the control runs a *later*
        cycle, where the row is strictly colder, so it establishes evictability
        at cycle 10 and not at cycle 7. The cadence predicate itself is pinned
        directly, by the ``_hygiene_due`` assertions below.
        """
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        from engrava.extensions.dreaming import DreamingExtension

        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=5)
        s = SqliteEngravaCore(conn, hygiene_policy=policy)
        s._dreaming_extension = DreamingExtension(config=DreamingConfig(enabled=True))
        await s.ensure_schema()
        try:
            await s.create_thought(_evictable_thought("cold"))
            # cycle 7 % 5 != 0 -> hygiene skipped.
            await s.consolidate(current_cycle=7)
            assert await _raw_lifecycle(s, "cold") == "ACTIVE"
            # cycle 10 % 5 == 0 -> the same row, same policy, now archived.
            await s.consolidate(current_cycle=10)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            assert s._hygiene_due(7) is False
            assert s._hygiene_due(10) is True
        finally:
            await conn.close()

    async def test_run_hygiene_bypasses_cadence(self) -> None:
        """An explicit run_hygiene ignores check_every_n_cycles."""
        policy = _forgetful_policy(eviction_threshold=1.0, check_every_n_cycles=100)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            # cycle 7 is not a multiple of 100, but the direct call runs anyway.
            result = await s.run_hygiene(current_cycle=7)
            assert result.archived_count == 1
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# HygieneResult value object
# ---------------------------------------------------------------------------


class TestHygieneResult:
    def test_defaults(self) -> None:
        result = HygieneResult()
        assert result.archived_count == 0
        assert result.gc_count == 0
        assert result.candidates_evaluated == 0
        assert result.dry_run is False
        assert result.would_evict == []
        assert result.flat_signals == []


# ---------------------------------------------------------------------------
# from_config end-to-end wiring
# ---------------------------------------------------------------------------


class TestFromConfigWiring:
    async def test_yaml_hygiene_policy_wires_into_store(self, tmp_path: Path) -> None:
        """A ``hygiene_policy`` YAML section wires the loop onto the store."""
        db_path = tmp_path / "hygiene.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"""
            database:
              path: "{db_path}"
            hygiene_policy:
              enabled: true
              eviction_threshold: 0.5
              min_inactivity_age_seconds: 0
            """,
            encoding="utf-8",
        )
        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            assert store._hygiene_policy is not None
            assert store._hygiene_policy.enabled is True
            assert store._hygiene_policy.eviction_threshold == pytest.approx(0.5)
            assert store._hygiene_policy.min_inactivity_age_seconds == 0
            await store.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            result = await store.run_hygiene(current_cycle=1000)
            assert result.archived_count == 1

    async def test_absent_section_leaves_store_without_policy(self, tmp_path: Path) -> None:
        """No ``hygiene_policy`` section ⇒ the store has no policy and cannot run."""
        db_path = tmp_path / "no_hygiene.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f'database:\n  path: "{db_path}"\n',
            encoding="utf-8",
        )
        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            assert store._hygiene_policy is None
            with pytest.raises(RuntimeError, match="hygiene policy"):
                await store.run_hygiene(current_cycle=1)


# ---------------------------------------------------------------------------
# Cold-start guard — minimum-inactivity-age gate (per-thought)
# ---------------------------------------------------------------------------


class TestInactiveEnoughHelper:
    """Unit tests for the ``_hygiene_inactive_enough`` COALESCE-ladder predicate."""

    def test_gate_disabled_passes_everything(self) -> None:
        """``min_inactivity_age_seconds == 0`` disables the gate (all-NULL included)."""
        policy = HygienePolicyConfig(min_inactivity_age_seconds=0)
        assert _hygiene_inactive_enough(_thought("t"), policy, _NOW) is True

    def test_all_null_timestamps_fail_closed(self) -> None:
        """No last-contact time ⇒ indeterminate age ⇒ protected (fail closed)."""
        policy = HygienePolicyConfig(min_inactivity_age_seconds=_WEEK_SECONDS)
        t = _thought("t")
        assert t.last_accessed_at is None
        assert t.updated_at is None
        assert t.created_at is None
        assert _hygiene_inactive_enough(t, policy, _NOW) is False

    def test_coalesce_prefers_recent_last_accessed_over_old_created(self) -> None:
        """(a) A recent read protects a row with an otherwise-ancient creation time."""
        policy = HygienePolicyConfig(min_inactivity_age_seconds=_WEEK_SECONDS)
        t = _thought(
            "t",
            created_at=_LONG_AGO,
            last_accessed_at=_iso_before(_NOW, 60),  # read a minute ago
        )
        assert _hygiene_inactive_enough(t, policy, _NOW) is False

    def test_coalesce_falls_back_to_updated_at_when_last_accessed_null(self) -> None:
        """(b) ``last_accessed_at`` NULL ⇒ ``updated_at`` decides, not ``created_at``."""
        policy = HygienePolicyConfig(min_inactivity_age_seconds=_WEEK_SECONDS)
        # updated recently over an ancient creation time -> protected via updated_at.
        t = _thought(
            "t",
            created_at=_LONG_AGO,
            updated_at=_iso_before(_NOW, 60),
        )
        assert _hygiene_inactive_enough(t, policy, _NOW) is False

    def test_coalesce_falls_back_to_created_at_when_others_null(self) -> None:
        """(c) Both ``last_accessed_at`` and ``updated_at`` NULL ⇒ ``created_at`` decides."""
        policy = HygienePolicyConfig(min_inactivity_age_seconds=_WEEK_SECONDS)
        t = _thought("t", created_at=_LONG_AGO)
        assert t.updated_at is None
        assert _hygiene_inactive_enough(t, policy, _NOW) is True

    def test_boundary_just_over_is_eligible_just_under_is_protected(self) -> None:
        """The ``>=`` boundary is exact and deterministic under an injected ``now``."""
        policy = HygienePolicyConfig(min_inactivity_age_seconds=_WEEK_SECONDS)
        just_over = _thought("over", last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS + 1))
        just_under = _thought("under", last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS - 1))
        assert _hygiene_inactive_enough(just_over, policy, _NOW) is True
        assert _hygiene_inactive_enough(just_under, policy, _NOW) is False


class TestEnsureUtc:
    """The ``now`` normaliser that keeps the age gate correct across offsets."""

    def test_naive_is_interpreted_as_utc(self) -> None:
        naive = datetime.datetime(2026, 1, 1, 12, 0, 0)  # noqa: DTZ001 - deliberately naive input
        normalised = _ensure_utc(naive)
        assert normalised.tzinfo == datetime.UTC
        assert normalised == datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

    def test_aware_non_utc_is_converted_to_utc(self) -> None:
        plus_two = datetime.timezone(datetime.timedelta(hours=2))
        aware = datetime.datetime(2026, 1, 1, 14, 0, 0, tzinfo=plus_two)  # 12:00 UTC
        normalised = _ensure_utc(aware)
        assert normalised.utcoffset() == datetime.timedelta(0)
        # Same instant, re-expressed in UTC.
        assert normalised == datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

    def test_utc_is_unchanged(self) -> None:
        assert _ensure_utc(_NOW) == _NOW


class TestUsageSignalHelper:
    """Unit tests for the access-gate predicate ``has_active_usage_signal``."""

    def test_usage_history_signal_names(self) -> None:
        assert USAGE_HISTORY_SIGNALS == ("frequency", "confirmation", "action_outcome")

    def test_no_usage_data_is_inactive(self) -> None:
        assert (
            has_active_usage_signal(
                [_thought("t")], current_cycle=10, access_tracking_enabled=False
            )
            is False
        )

    def test_action_outcome_activates_gate(self) -> None:
        assert (
            has_active_usage_signal(
                [_thought("t", action_outcome_score=0.0)],
                current_cycle=10,
                access_tracking_enabled=False,
            )
            is True
        )

    def test_confirmation_activates_gate(self) -> None:
        assert (
            has_active_usage_signal(
                [_thought("t", confirmation_count=1)],
                current_cycle=10,
                access_tracking_enabled=False,
            )
            is True
        )

    def test_frequency_needs_access_tracking_enabled(self) -> None:
        """``access_count`` only counts as a usage signal when tracking is on."""
        pool = [_thought("t", access_count=3)]
        assert (
            has_active_usage_signal(pool, current_cycle=10, access_tracking_enabled=False) is False
        )
        assert has_active_usage_signal(pool, current_cycle=10, access_tracking_enabled=True) is True


class TestMinimumInactivityAgeGate:
    """The per-thought age gate via ``run_hygiene`` (the E35 cold-start fix)."""

    async def test_fresh_store_archives_nothing_then_gate_off_reveals_footgun(self) -> None:
        """A fresh bulk-ingested store archives nothing; disabling the gate archives
        the earliest-ingested rows (the pre-gate footgun), proving the gate.
        """
        # Fresh store: created_at/updated_at stamped ~now, last_accessed_at NULL.
        # A usage signal on every row keeps the *access-gate* open so the *age*
        # gate is the sole variable under test.
        s = await _make_store(
            HygienePolicyConfig(enabled=True, eviction_threshold=1.0, max_evictions_per_run=2)
        )
        try:
            for i in range(5):
                await s.create_thought(
                    _thought(f"t{i:02d}", updated_cycle=i * 100, action_outcome_score=0.0)
                )
            # Gate ON (default 7 days): every row is seconds old -> nothing archived.
            on = await s.run_hygiene(current_cycle=1000)
            assert on.archived_count == 0

            # Gate OFF: the earliest-ingested (coldest-cycle) rows are archived
            # again under the cap — exactly the E35 artifact the gate suppresses.
            s._hygiene_policy = HygienePolicyConfig(
                enabled=True,
                eviction_threshold=1.0,
                max_evictions_per_run=2,
                min_inactivity_age_seconds=0,
            )
            off = await s.run_hygiene(current_cycle=1000)
            assert off.archived_count == 2
            assert await _raw_lifecycle(s, "t00") == "ARCHIVED"
            assert await _raw_lifecycle(s, "t01") == "ARCHIVED"
            assert await _raw_lifecycle(s, "t04") == "ACTIVE"
        finally:
            await s._db.close()

    async def test_aged_store_with_usage_signal_still_forgets(self) -> None:
        """A genuinely aged, used store is not over-protected — cold rows archive."""
        s = await _make_store(HygienePolicyConfig(enabled=True, eviction_threshold=1.0))
        try:
            # Backdated last-contact (> 7 days) + a usage signal -> both guards open.
            await s.create_thought(
                _thought(
                    "cold",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS * 4),
                    action_outcome_score=0.0,
                )
            )
            result = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_boundary_via_injected_now(self) -> None:
        """A row 1s past the window archives; a row 1s inside it is protected."""
        s = await _make_store(
            HygienePolicyConfig(
                enabled=True,
                eviction_threshold=1.0,
                min_inactivity_age_seconds=_WEEK_SECONDS,
            )
        )
        try:
            await s.create_thought(
                _thought(
                    "over",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS + 1),
                    action_outcome_score=0.0,
                )
            )
            await s.create_thought(
                _thought(
                    "under",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS - 1),
                    action_outcome_score=0.0,
                )
            )
            result = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "over") == "ARCHIVED"
            assert await _raw_lifecycle(s, "under") == "ACTIVE"
        finally:
            await s._db.close()

    async def test_write_time_recheck_protects_row_read_between_select_and_archive(
        self,
    ) -> None:
        """TOCTOU: a row read (``last_accessed_at`` bumped) after selection is skipped
        at the archive write, while an untouched aged row is archived — pinning the
        write-time re-check as load-bearing (drop it and both would archive).
        """
        policy = HygienePolicyConfig(enabled=True, min_inactivity_age_seconds=_WEEK_SECONDS)
        s = await _make_store(policy)
        try:
            for tid in ("read_late", "stays_cold"):
                await s.create_thought(
                    _thought(
                        tid,
                        updated_cycle=0,
                        last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS * 4),
                    )
                )
            # Simulate a read landing between selection and archival: bump the one
            # row's last_accessed_at back to ~now (inside the inactivity window).
            await s._db.execute(
                "UPDATE thought SET last_accessed_at = ? WHERE thought_id = ?",
                (_NOW.isoformat(), "read_late"),
            )

            def _reason(tid: str) -> EvictionReason:
                return EvictionReason(
                    thought_id=tid,
                    keep_score=0.0,
                    eviction_score=0.0,
                    decay_multiplier=1.0,
                    threshold=policy.eviction_threshold,
                )

            archived = await s._hygiene_archive(
                [_reason("read_late"), _reason("stays_cold")],
                policy=policy,
                current_cycle=1,
                now=_NOW,
            )
            assert archived == 1
            assert await _raw_lifecycle(s, "read_late") == "ACTIVE"
            assert await _raw_lifecycle(s, "stays_cold") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_injected_now_normalised_from_non_utc_offset(self) -> None:
        """An aware non-UTC injected ``now`` yields the same decision as its UTC
        equivalent — the age gate and the write-time SQL cutoff both normalise it.
        """
        plus_five = datetime.timezone(datetime.timedelta(hours=5))
        now_plus_five = _NOW.astimezone(plus_five)  # same instant as _NOW, +05:00
        s = await _make_store(
            HygienePolicyConfig(
                enabled=True,
                eviction_threshold=1.0,
                min_inactivity_age_seconds=_WEEK_SECONDS,
            )
        )
        try:
            await s.create_thought(
                _thought(
                    "aged",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS * 3),
                    action_outcome_score=0.0,
                )
            )
            await s.create_thought(
                _thought(
                    "young",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, 3600),
                    action_outcome_score=0.0,
                )
            )
            result = await s.run_hygiene(current_cycle=1000, now=now_plus_five)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "aged") == "ARCHIVED"
            assert await _raw_lifecycle(s, "young") == "ACTIVE"
        finally:
            await s._db.close()

    async def test_write_time_sql_guard_closes_the_stale_refetch_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The atomic UPDATE re-asserts the inactivity cutoff, so a row that is
        actually fresh in the DB is not archived even when the archive path's
        re-fetched copy still looks aged — the window the Python re-check alone
        cannot see (drop the SQL guard and this row is wrongly archived).
        """
        policy = HygienePolicyConfig(enabled=True, min_inactivity_age_seconds=_WEEK_SECONDS)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought(
                    "raced",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, _WEEK_SECONDS * 4),
                )
            )
            # Capture the (aged) row the archive path will re-fetch, then make the
            # *actual* DB row fresh (as a read landing after that fetch would).
            stale_row = await s._get_thought_row("raced")
            assert stale_row is not None
            await s._db.execute(
                "UPDATE thought SET last_accessed_at = ? WHERE thought_id = ?",
                (_NOW.isoformat(), "raced"),
            )

            async def _stale_fetch(thought_id: str) -> object:
                return stale_row

            monkeypatch.setattr(s, "_get_thought_row", _stale_fetch)

            reason = EvictionReason(
                thought_id="raced",
                keep_score=0.0,
                eviction_score=0.0,
                decay_multiplier=1.0,
                threshold=policy.eviction_threshold,
            )
            archived = await s._hygiene_archive([reason], policy=policy, current_cycle=1, now=_NOW)
            # Python re-check passes (stale row is aged); the SQL WHERE excludes the
            # actually-fresh row -> nothing archived.
            assert archived == 0
            assert await _raw_lifecycle(s, "raced") == "ACTIVE"
        finally:
            await s._db.close()

    @pytest.mark.parametrize("threshold", [0.2, 0.5, 1.0])
    @pytest.mark.parametrize("gate_seconds", [_WEEK_SECONDS, _WEEK_SECONDS * 8])
    async def test_monotone_safe_archive_set_is_subset_of_gate_off_set(
        self, threshold: float, gate_seconds: int
    ) -> None:
        """The gated archive set is a subset of the ``gate=0`` set when the cap does not bind.

        The gate only ever *protects* (skips) a candidate — it never lowers a
        keep-score or raises the cap — so with a non-binding ``max_evictions_per_run``
        (the default 100 here, 4 rows) the guarded set is a strict subset of the
        gate-off set and omits exactly the young rows. (Under a *binding* cap a freed
        slot may be backfilled by another independently-eligible aged row — see
        ``test_binding_cap_backfills_only_eligible_aged_rows``.) Compared on a mixed
        store via a non-mutating ``dry_run`` (same store, two policies).
        """
        s = await _make_store(
            HygienePolicyConfig(
                enabled=True,
                eviction_threshold=threshold,
                dry_run=True,
                min_inactivity_age_seconds=0,  # the genuine gate-off baseline
            )
        )
        try:
            young_ids = {"young0", "young1"}
            # A mix of young (1 day) and aged (> the gate window) rows, all cold by
            # cycle, with a usage signal so the access-gate never hides the effect.
            for tid, seconds in (
                ("young0", 86400),
                ("young1", 86400),
                ("aged0", gate_seconds * 2),
                ("aged1", gate_seconds * 3),
            ):
                await s.create_thought(
                    _thought(
                        tid,
                        updated_cycle=0,
                        last_accessed_at=_iso_before(_NOW, seconds),
                        action_outcome_score=0.0,
                    )
                )

            gate_off = await s.run_hygiene(current_cycle=1000, now=_NOW)
            off_ids = {r.thought_id for r in gate_off.would_evict}

            s._hygiene_policy = HygienePolicyConfig(
                enabled=True,
                eviction_threshold=threshold,
                dry_run=True,
                min_inactivity_age_seconds=gate_seconds,
            )
            gate_on = await s.run_hygiene(current_cycle=1000, now=_NOW)
            on_ids = {r.thought_id for r in gate_on.would_evict}

            assert on_ids <= off_ids  # monotone-safe: the gate only removes rows
            # And it removes exactly the young ones (the discriminating direction).
            assert on_ids.isdisjoint(young_ids)
        finally:
            await s._db.close()

    async def test_binding_cap_backfills_only_eligible_aged_rows(self) -> None:
        """Under a binding cap the gate may archive an aged row absent from the
        gate-off set — a benign rate-limit reshuffle, never a young/ineligible row.

        The age gate protects a candidate *before* ``max_evictions_per_run`` is
        applied, so protecting a high-ranked young row can free a slot another
        independently-eligible aged row fills. That backfilled row is not a subset
        of the gate-off set, yet it is genuinely eligible (aged past the window,
        below threshold, unprotected) and would be archived in a later run anyway.
        The invariant that actually holds is per-candidate: no young, protected, or
        ineligible row is ever archived.
        """
        gate = _WEEK_SECONDS
        # Both rows share every keep-score input (same cycle, no keep-score-weighted
        # usage signal), so the archive order is the (score, updated_cycle,
        # thought_id) tiebreak — "a_young" sorts ahead of "b_aged". ``action_outcome``
        # (not a keep-score signal) only opens the access-gate.
        s = await _make_store(
            HygienePolicyConfig(
                enabled=True,
                eviction_threshold=1.0,
                max_evictions_per_run=1,  # binding: two eligible rows, one slot
                dry_run=True,
                min_inactivity_age_seconds=0,  # genuine gate-off baseline
            )
        )
        try:
            await s.create_thought(
                _thought(
                    "a_young",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, 86400),  # 1 day -> young
                    action_outcome_score=0.0,
                )
            )
            await s.create_thought(
                _thought(
                    "b_aged",
                    updated_cycle=0,
                    last_accessed_at=_iso_before(_NOW, gate * 3),  # aged past window
                    action_outcome_score=0.0,
                )
            )

            gate_off = await s.run_hygiene(current_cycle=1000, now=_NOW)
            off_ids = {r.thought_id for r in gate_off.would_evict}
            # The single cap slot goes to the tiebreak-first row, "a_young".
            assert off_ids == {"a_young"}

            s._hygiene_policy = HygienePolicyConfig(
                enabled=True,
                eviction_threshold=1.0,
                max_evictions_per_run=1,
                dry_run=True,
                min_inactivity_age_seconds=gate,
            )
            gate_on = await s.run_hygiene(current_cycle=1000, now=_NOW)
            on_ids = {r.thought_id for r in gate_on.would_evict}

            # The backfill: "b_aged" is archived under the gate though it is NOT in
            # the gate-off set -> not a strict subset ...
            assert on_ids == {"b_aged"}
            assert not on_ids <= off_ids
            # ... but every archived row is aged + eligible, no young row is archived,
            # and the cap is still respected.
            assert "a_young" not in on_ids
            assert len(on_ids) <= 1
        finally:
            await s._db.close()


class TestAccessGate:
    """The run-level access-gate (D4): no usage signal ⇒ archive nothing."""

    async def test_aged_never_queried_store_archives_nothing(self) -> None:
        """An aged-but-never-used store archives nothing (recency-of-cycle alone
        must not drive eviction); adding one usage signal lets it proceed.
        """
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            # Aged (backdated created/updated), never read, no confirmations or
            # outcomes -> the age gate passes but the access-gate must block.
            for i in range(3):
                await s.create_thought(
                    _thought(
                        f"aged{i}",
                        updated_cycle=0,
                        created_at=_LONG_AGO,
                        updated_at=_LONG_AGO,
                    )
                )
            blocked = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert blocked.archived_count == 0
            assert await _raw_lifecycle(s, "aged0") == "ACTIVE"

            # Add a single usage signal (one applied-action outcome) -> the run
            # proceeds and archives the genuinely-cold rows.
            await s.create_thought(
                _thought(
                    "aged_used",
                    updated_cycle=0,
                    created_at=_LONG_AGO,
                    updated_at=_LONG_AGO,
                    action_outcome_score=0.0,
                )
            )
            proceeded = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert proceeded.archived_count >= 1
            assert await _raw_lifecycle(s, "aged0") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_access_gate_builds_the_full_result_contract(self) -> None:
        """When the access-gate fires, the run is a safe no-op that still reports
        candidates and flat signals (only the archive set is empty).
        """
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("aged", updated_cycle=0, created_at=_LONG_AGO, updated_at=_LONG_AGO)
            )
            result = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert result.archived_count == 0
            assert result.candidates_evaluated == 1
            # Usage/confidence signals with no data are still reported as flat.
            assert "frequency" in result.flat_signals
            assert "confirmation" in result.flat_signals
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Stage 2 GC — wall-clock restore window (required in addition to the cycle window)
# ---------------------------------------------------------------------------


async def _archive_row_raw(
    store: SqliteEngravaCore,
    thought_id: str,
    *,
    archived_at_cycle: int | None,
    archived_at: str | None,
    pinned: bool = False,
    priority: Priority = Priority.P3,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
) -> None:
    """Insert an ARCHIVED thought and stamp both hygiene markers directly.

    Stamping via raw SQL lets a test set the cycle window
    (``archived_at_cycle``) and the wall-clock window (``archived_at``)
    independently, so the two-window conjunction can be exercised in isolation.
    ``archived_at=None`` reproduces a hygiene-archived row that predates the
    ``archived_at`` column (the legacy fail-closed case).
    """
    await store.create_thought(
        _thought(
            thought_id,
            lifecycle_status=LifecycleStatus.ARCHIVED,
            pinned=pinned,
            priority=priority,
            thought_type=thought_type,
        )
    )
    await store._db.execute(
        "UPDATE thought SET archived_at_cycle = ?, archived_at = ? WHERE thought_id = ?",
        (archived_at_cycle, archived_at, thought_id),
    )


class TestGarbageCollectionWallClockWindow:
    """The wall-clock restore window: GC requires BOTH windows before permanent delete."""

    async def test_wall_clock_window_protects_then_zero_reveals_footgun(self) -> None:
        """A recently-archived row survives the elapsed cycle window; disabling the
        wall-clock window (``gc_restore_window_seconds=0``) GC's it — the
        discriminating revert that reproduces the pre-window footgun.
        """
        policy = _forgetful_policy(
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
            gc_restore_window_seconds=_MONTH_SECONDS,
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            r1 = await s.run_hygiene(current_cycle=5, now=_NOW)  # archives, stamps archived_at
            assert r1.archived_count == 1
            assert await _raw_archived_at(s, "cold") == _NOW.isoformat()

            # Cycle window elapsed (100 - 5 >= 10) but only one hour of wall clock
            # has passed (< 30 days) -> protected by the wall-clock window.
            r2 = await s.run_hygiene(current_cycle=100, now=_NOW + datetime.timedelta(hours=1))
            assert r2.gc_count == 0
            assert await s.get_thought("cold") is not None

            # Disable the wall-clock window -> cycle-only -> the row is now GC'd
            # (the pre-window behaviour the wall-clock window guards against).
            s._hygiene_policy = _forgetful_policy(
                eviction_threshold=1.0,
                auto_gc_enabled=True,
                gc_min_archive_age_cycles=10,
                gc_restore_window_seconds=0,
            )
            r3 = await s.run_hygiene(current_cycle=100, now=_NOW + datetime.timedelta(hours=1))
            assert r3.gc_count == 1
            assert await s.get_thought("cold") is None
        finally:
            await s._db.close()

    async def test_both_windows_required(self) -> None:
        """Only a row past BOTH windows is GC'd; each window alone protects."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,  # archive nothing new; GC the pre-stamped rows
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
            gc_restore_window_seconds=_MONTH_SECONDS,
        )
        s = await _make_store(policy)
        try:
            recent = _iso_before(_NOW, 60)  # 1 minute ago -> wall-clock NOT elapsed
            old = _iso_before(_NOW, _MONTH_SECONDS * 2)  # wall-clock elapsed
            # cycle elapsed at current_cycle=100: archived_at_cycle=5 (100-5>=10);
            # cycle NOT elapsed: archived_at_cycle=95 (100-95 < 10).
            await _archive_row_raw(s, "cycle_only", archived_at_cycle=5, archived_at=recent)
            await _archive_row_raw(s, "wallclock_only", archived_at_cycle=95, archived_at=old)
            await _archive_row_raw(s, "both", archived_at_cycle=5, archived_at=old)

            result = await s.run_hygiene(current_cycle=100, now=_NOW)

            assert result.gc_count == 1
            assert await s.get_thought("cycle_only") is not None  # wall-clock protects
            assert await s.get_thought("wallclock_only") is not None  # cycle protects
            assert await s.get_thought("both") is None  # both elapsed -> reaped
        finally:
            await s._db.close()

    async def test_boundary_via_injected_now(self) -> None:
        """The wall-clock boundary is exact and inclusive (``<=``), under an injected ``now``.

        A row archived exactly at ``now - gc_restore_window_seconds`` is eligible
        (the ``<=`` cutoff is inclusive — changing it to ``<`` would protect the
        ``exact`` row and fail this test); one second later is protected, one
        second earlier is eligible.
        """
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
            gc_restore_window_seconds=_MONTH_SECONDS,
        )
        s = await _make_store(policy)
        try:
            # All rows are past the cycle window (archived_at_cycle=0, cycle=1000),
            # so only the wall-clock boundary decides.
            await _archive_row_raw(
                s, "over", archived_at_cycle=0, archived_at=_iso_before(_NOW, _MONTH_SECONDS + 1)
            )
            await _archive_row_raw(
                s, "exact", archived_at_cycle=0, archived_at=_iso_before(_NOW, _MONTH_SECONDS)
            )
            await _archive_row_raw(
                s, "under", archived_at_cycle=0, archived_at=_iso_before(_NOW, _MONTH_SECONDS - 1)
            )
            result = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert result.gc_count == 2
            assert await s.get_thought("over") is None  # just over the window -> reaped
            assert await s.get_thought("exact") is None  # exactly at the cutoff -> reaped (<=)
            assert await s.get_thought("under") is not None  # just under -> protected
        finally:
            await s._db.close()

    async def test_legacy_null_archived_at_never_gc(self) -> None:
        """A hygiene-archived row with ``archived_at`` NULL (pre-column) fails closed.

        Even with the cycle window and ``gc_restore_window_seconds`` both
        satisfied it is never GC-eligible, while a stamped-old sibling in the same
        run *is* reaped — proving the NULL exclusion is the reason, not a
        misconfigured window.
        """
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=1,  # active window; even 1s makes stamped rows eligible
        )
        s = await _make_store(policy)
        try:
            await _archive_row_raw(s, "legacy", archived_at_cycle=0, archived_at=None)
            await _archive_row_raw(
                s, "stamped", archived_at_cycle=0, archived_at=_iso_before(_NOW, 3600)
            )
            result = await s.run_hygiene(current_cycle=1000, now=_NOW)
            assert result.gc_count == 1
            assert await s.get_thought("legacy") is not None  # fail closed: never reaped
            assert await s.get_thought("stamped") is None  # window satisfied -> reaped
        finally:
            await s._db.close()

    async def test_restore_clears_archived_at_and_rearchive_restamps(self) -> None:
        """archive stamps ``archived_at``; restore clears it; re-archival restamps fresh.

        A restored row transitions back to ACTIVE with both hygiene markers NULL,
        so it leaves the GC candidate set entirely (GC only touches ARCHIVED rows
        with a non-NULL ``archived_at_cycle``) — it is structurally never GC'd
        until it is re-archived, which stamps a fresh ``archived_at``.
        """
        policy = _forgetful_policy(
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=_MONTH_SECONDS,
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0, action_outcome_score=0.0))
            await s.run_hygiene(current_cycle=5, now=_NOW)
            assert await _raw_archived_at(s, "cold") == _NOW.isoformat()

            restored = await s.restore_thought("cold")
            assert restored.archived_at is None
            assert await _raw_archived_at(s, "cold") is None
            assert await _raw_archived_at_cycle(s, "cold") is None
            assert await _raw_lifecycle(s, "cold") == "ACTIVE"

            # Re-archival on a later pass stamps a fresh, later ``archived_at``
            # (not the stale original), so the wall-clock window restarts.
            later = _NOW + datetime.timedelta(days=1)
            await s.run_hygiene(current_cycle=1001, now=later)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            assert await _raw_archived_at(s, "cold") == later.isoformat()
        finally:
            await s._db.close()

    async def test_ttl_manual_archived_never_gc_with_window_active(self) -> None:
        """A TTL/manually-archived row (both markers NULL) is never GC'd (window active)."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=_MONTH_SECONDS,
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("manual", updated_cycle=0))
            await s.update_thought("manual", lifecycle_status=LifecycleStatus.ARCHIVED)
            assert await _raw_archived_at_cycle(s, "manual") is None
            assert await _raw_archived_at(s, "manual") is None

            far_future = _NOW + datetime.timedelta(days=999)
            result = await s.run_hygiene(current_cycle=1000, now=far_future)
            assert result.gc_count == 0
            assert await s.get_thought("manual") is not None
        finally:
            await s._db.close()

    async def test_archived_at_stamped_only_on_hygiene_path(self) -> None:
        """Only the hygiene archive path stamps ``archived_at``; TTL/manual leaves NULL."""
        policy = _forgetful_policy(eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("hygiene", updated_cycle=0, action_outcome_score=0.0))
            await s.create_thought(_thought("manual", updated_cycle=0, action_outcome_score=0.0))
            # Manually archive one row before the pass; hygiene archives the other.
            await s.update_thought("manual", lifecycle_status=LifecycleStatus.ARCHIVED)

            await s.run_hygiene(current_cycle=42, now=_NOW)

            assert await _raw_archived_at(s, "hygiene") == _NOW.isoformat()
            assert await _raw_archived_at(s, "manual") is None
        finally:
            await s._db.close()

    async def test_delete_scope_preserved_with_journal(self) -> None:
        """A GC under the wall-clock window keeps the full delete scope + journal.

        Orphan-reflection sweep runs first, the FK cascade drops the edge, and a
        ``DELETE_THOUGHT`` journal entry with a full ``before`` snapshot is
        appended — the GC-is-not-erasure property (content survives in history).
        """
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,  # archive nothing new via score
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=_MONTH_SECONDS,
        )
        s = await _make_store(policy, journal_enabled=True)
        try:
            # A source OBSERVATION hygiene-archived past BOTH windows (GC target).
            await _archive_row_raw(
                s, "src", archived_at_cycle=0, archived_at=_iso_before(_NOW, _MONTH_SECONDS * 2)
            )
            # A REFLECTION consolidated only from that (now non-live) source.
            await s.create_thought(
                _thought("refl", thought_type=ThoughtType.REFLECTION, updated_cycle=0)
            )
            await s.create_edge(
                EdgeRecord(
                    edge_id="e1",
                    from_thought_id="refl",
                    to_thought_id="src",
                    edge_type=EdgeType.CONSOLIDATED_FROM,
                    weight=0.5,
                    created_cycle=0,
                    source=KnowledgeSource.EXPERIENCE,
                )
            )
            result = await s.run_hygiene(current_cycle=1000, now=_NOW)

            assert result.gc_count == 1
            assert await s.get_thought("src") is None
            # Orphan sweep ran before the delete -> the REFLECTION is retired ...
            assert await _raw_lifecycle(s, "refl") == "ARCHIVED"
            # ... and the FK cascade dropped the dangling edge.
            cursor = await s._db.execute("SELECT COUNT(*) AS n FROM edge")
            assert (await cursor.fetchone())["n"] == 0

            # GC != erasure: the DELETE_THOUGHT entry keeps a full ``before`` snapshot.
            entries = await s.journal.get_entries()
            gc_entries = [
                e
                for e in entries
                if e.mutation_type == "DELETE_THOUGHT"
                and isinstance(e.delta, dict)
                and e.delta.get("before") is not None
                and e.delta["before"].get("thought_id") == "src"
            ]
            assert len(gc_entries) == 1
            assert gc_entries[0].delta["before"]["content"] == "content of src"
            assert (await s.verify_journal()).valid is True
        finally:
            await s._db.close()

    async def test_monotone_gc_set_is_subset_of_cycle_only_set_when_cap_not_binding(self) -> None:
        """With a non-binding cap the wall-clock GC set is a subset of the cycle-only set.

        Requiring an additional window can only ever *shrink* the eligible pool, so
        with a non-binding ``max_evictions_per_run`` (the default 100 here, 4 rows)
        the window-active store reaps a strict subset, omitting the recently-archived
        row and the legacy NULL-stamped row. (Under a *binding* cap a freed slot may
        be filled by another genuinely-eligible aged row — see
        ``test_binding_cap_reaps_only_rows_past_both_windows``.) Compared on two
        identical stores (GC deletes, so a single store cannot show both outcomes).
        """
        seeds = {
            "old1": _iso_before(_NOW, _MONTH_SECONDS * 2),  # both windows
            "old2": _iso_before(_NOW, _MONTH_SECONDS + 100),  # both windows
            "recent": _iso_before(_NOW, 60),  # cycle only (wall-clock too new)
            "legacy": None,  # cycle only (no wall-clock stamp -> fail closed when active)
        }

        async def _run(window_seconds: int) -> set[str]:
            policy = HygienePolicyConfig(
                enabled=True,
                eviction_threshold=0.0,
                auto_gc_enabled=True,
                gc_min_archive_age_cycles=10,
                gc_restore_window_seconds=window_seconds,
            )
            s = await _make_store(policy)
            try:
                for tid, archived_at in seeds.items():
                    await _archive_row_raw(s, tid, archived_at_cycle=0, archived_at=archived_at)
                await s.run_hygiene(current_cycle=1000, now=_NOW)
                return {tid for tid in seeds if await s.get_thought(tid) is None}
            finally:
                await s._db.close()

        cycle_only = await _run(0)
        wall_clock = await _run(_MONTH_SECONDS)

        assert wall_clock <= cycle_only  # monotone-safe: the window only removes rows
        assert wall_clock == {"old1", "old2"}
        # The discriminating direction: the extra window omits exactly the young
        # and the unstampable-legacy rows the cycle-only path would have reaped.
        assert cycle_only == {"old1", "old2", "recent", "legacy"}

    async def test_binding_cap_reaps_only_rows_past_both_windows(self) -> None:
        """Under a binding cap the window may reap a *different* row, never an unsafe one.

        The wall-clock predicate filters candidates *before* the ``ORDER BY … LIMIT``
        cap, so protecting a high-ordered young row can free the single slot for a
        genuinely-eligible aged row that the cycle-only path (which spent the slot on
        the young row) did not reach. The gated set is then **not** a subset of the
        cycle-only set — yet every reaped row is past BOTH windows, and the young
        row is never reaped under the window. This mirrors the archive-stage
        binding-cap reshuffle and pins the per-candidate safety invariant.
        """
        # Both rows share ``archived_at_cycle`` so the delete order is the
        # ``thought_id ASC`` tiebreak: ``a_recent`` sorts before ``b_aged``.
        recent = _iso_before(_NOW, 60)  # wall-clock NOT elapsed
        aged = _iso_before(_NOW, _MONTH_SECONDS * 2)  # wall-clock elapsed

        async def _run(window_seconds: int) -> set[str]:
            policy = HygienePolicyConfig(
                enabled=True,
                eviction_threshold=0.0,
                auto_gc_enabled=True,
                gc_min_archive_age_cycles=0,
                gc_restore_window_seconds=window_seconds,
                max_evictions_per_run=1,  # binding: two cycle-eligible rows, one slot
            )
            s = await _make_store(policy)
            try:
                await _archive_row_raw(s, "a_recent", archived_at_cycle=0, archived_at=recent)
                await _archive_row_raw(s, "b_aged", archived_at_cycle=0, archived_at=aged)
                await s.run_hygiene(current_cycle=1000, now=_NOW)
                return {tid for tid in ("a_recent", "b_aged") if await s.get_thought(tid) is None}
            finally:
                await s._db.close()

        cycle_only = await _run(0)
        wall_clock = await _run(_MONTH_SECONDS)

        # Cycle-only spends the one slot on the tiebreak-first row ...
        assert cycle_only == {"a_recent"}
        # ... while the window filters that young row out and reaps the aged one:
        # not a subset (disjoint), but a safe reshuffle.
        assert wall_clock == {"b_aged"}
        assert not wall_clock <= cycle_only
        # Per-candidate safety: the young (wall-clock-failing) row is never reaped
        # under the window, and the reaped row is genuinely past both windows.
        assert "a_recent" not in wall_clock

    async def test_ttl_re_archival_clears_stale_hygiene_markers_end_to_end(self) -> None:
        """End-to-end reproduction of the stale-marker GC-bypass the fix closes.

        Literally walks the reachable sequence rather than hand-assigning markers:
        hygiene archives a cold row (stamping both markers) -> a low-level
        ``update_thought(lifecycle_status=ACTIVE)`` un-archives it and *leaves* the
        markers -> the row later expires and TTL re-archives it. Because a TTL
        archival is not a hygiene archival it clears both markers, so
        ``archived_at_cycle`` is NULL and the irreversible GC stage cannot reap the
        freshly re-archived row on the earlier, already-elapsed restore windows.
        Without the fix the stale markers would make GC delete it immediately.
        """
        policy = _forgetful_policy(
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
            gc_restore_window_seconds=_MONTH_SECONDS,
        )
        s = await _make_store(policy)
        try:
            # A genuinely cold + aged row with a usage signal so hygiene archives it.
            await s.create_thought(
                _thought(
                    "t",
                    updated_cycle=0,
                    created_at=_LONG_AGO,
                    updated_at=_LONG_AGO,
                    action_outcome_score=0.0,
                )
            )

            # 1) Hygiene archives it, stamping both markers.
            r1 = await s.run_hygiene(current_cycle=500, now=_NOW)
            assert r1.archived_count == 1
            assert await _raw_archived_at_cycle(s, "t") == 500
            assert await _raw_archived_at(s, "t") == _NOW.isoformat()

            # 2) A raw low-level un-archive leaves the (now stale) markers behind.
            await s.update_thought("t", lifecycle_status=LifecycleStatus.ACTIVE)
            assert await _raw_lifecycle(s, "t") == "ACTIVE"
            assert await _raw_archived_at_cycle(s, "t") == 500
            assert await _raw_archived_at(s, "t") == _NOW.isoformat()

            # 3) The row later expires and TTL re-archives it -> markers cleared.
            await s.update_thought("t", expires_at=_iso_before(_NOW, 60))
            await s.cleanup_expired(now=_NOW.isoformat())
            assert await _raw_lifecycle(s, "t") == "ARCHIVED"
            assert await _raw_archived_at_cycle(s, "t") is None
            assert await _raw_archived_at(s, "t") is None

            # 4) GC far past both stale windows does NOT reap it (no hygiene marker).
            r2 = await s.run_hygiene(current_cycle=5000, now=_NOW + datetime.timedelta(days=90))
            assert r2.gc_count == 0
            assert await s.get_thought("t") is not None
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# The protection list the policy holds is the one it validated
# ---------------------------------------------------------------------------


class _LyingPriorities(tuple):  # type: ignore[type-arg]  # a bare built-in base is exactly what the adversary subclasses
    """A protection list that denies protecting anything.

    ``protected_priorities`` is consulted twice on the way to a deletion: once
    as a truth value, to decide whether the ``priority NOT IN (...)`` guard is
    added to the SQL, and once with ``in``, on the row that came back. A
    container answering both for itself un-protects every priority it holds
    while still reporting the right entries to anything that iterates it.
    """

    __slots__ = ()

    def __contains__(self, item: object) -> bool:
        del item
        return False

    def __bool__(self) -> bool:
        return False


class TestProtectedPrioritiesAreOwnedByThePolicy:
    """A pinned-by-priority thought survives whatever its policy's list claims."""

    def test_the_stored_list_is_a_plain_tuple_of_plain_strings(self) -> None:
        """Construction keeps the decoded list, not the caller's container."""

        class _Priority(str):
            __slots__ = ()

        policy = HygienePolicyConfig(protected_priorities=_LyingPriorities([_Priority("P1")]))
        assert type(policy.protected_priorities) is tuple
        assert [type(entry) for entry in policy.protected_priorities] == [str]
        assert policy.protected_priorities == ("P1",)
        assert "P1" in policy.protected_priorities
        assert bool(policy.protected_priorities) is True

    async def test_gc_never_reaps_a_protected_priority(self) -> None:
        """A ``P1`` row archived past both windows is still not deleted."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
            protected_priorities=_LyingPriorities(("P1",)),
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("keep", priority=Priority.P1))
            await s.update_thought(
                "keep",
                lifecycle_status=LifecycleStatus.ARCHIVED,
                archived_at_cycle=0,
            )
            result = await s.run_hygiene(current_cycle=1000)

            # State first: the row is what matters, and a run that deleted it
            # and then reported ``gc_count == 0`` would satisfy the count alone.
            assert await s.get_thought("keep") is not None
            assert await _raw_lifecycle(s, "keep") == "ARCHIVED"
            assert result.gc_count == 0
        finally:
            await s._db.close()

    async def test_archival_never_touches_a_protected_priority(self) -> None:
        """The archive stage skips a ``P1`` row for the same reason GC does."""
        policy = _forgetful_policy(
            eviction_threshold=1.0,
            protected_priorities=_LyingPriorities(("P1",)),
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("keep", priority=Priority.P1, created_at=_LONG_AGO, updated_at=_LONG_AGO)
            )
            await s.create_thought(
                _thought(
                    "drop",
                    action_outcome_score=0.9,
                    created_at=_LONG_AGO,
                    updated_at=_LONG_AGO,
                )
            )
            result = await s.run_hygiene(current_cycle=10)

            # Both halves: the protected row survives and the unprotected one
            # does not, so the run is shown to have done real work either way.
            assert await _raw_lifecycle(s, "keep") == "ACTIVE"
            assert await _raw_archived_at_cycle(s, "keep") is None
            assert await _raw_lifecycle(s, "drop") == "ARCHIVED"
            assert result.archived_count == 1
        finally:
            await s._db.close()

    async def test_an_unprotected_priority_is_still_collected(self) -> None:
        """The protection is a filter, not an off switch — GC still reaps."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=0.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
            gc_restore_window_seconds=0,
            protected_priorities=("P1",),
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("drop", priority=Priority.P3))
            await s.update_thought(
                "drop",
                lifecycle_status=LifecycleStatus.ARCHIVED,
                archived_at_cycle=0,
            )
            result = await s.run_hygiene(current_cycle=1000)

            assert await s.get_thought("drop") is None
            assert result.gc_count == 1
        finally:
            await s._db.close()

"""Tests for the deterministic Memory Hygiene forgetting loop.

Covers the opt-in, no-LLM forgetting pass that archives cold/low-value thoughts
and, separately opt-in, garbage-collects them after a restore window:

* config: ``HygienePolicyConfig`` defaults, validation, and YAML parsing;
* the ``core-18`` migration (``pinned`` / ``archived_at_cycle`` columns) and the
  ThoughtRecord round trip of both fields;
* default-OFF ⇒ no behavioural change (byte-identical write/read paths);
* keep-score + eviction rule with active-signal redistribution;
* protection: pinned is never touched, ``P1`` is protected by default,
  ``confidence`` is not protection;
* archive is reversible and restore clears ``archived_at_cycle``;
* deterministic capped selection (same store + config + cycle ⇒ same set);
* the all-flat-signals fail-safe (zero evictions);
* the decay clamp (a hook returning >1 / <0 / NaN / inf never over-evicts);
* two-stage GC: only hygiene-archived rows reaped, restore window enforced,
  orphan-reflection sweep before delete, vec0 purge, hash-chain still valid;
* ``dry_run`` mutates nothing;
* the journal ``eviction_reason`` with no new mutation type;
* the ``consolidate()`` convenience invocation + cadence.
"""

from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING

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
from engrava.infrastructure.sqlite.engrava_core import _clamp_decay, _hygiene_protected
from engrava.infrastructure.sqlite.hygiene import (
    EvictionReason,
    HygieneResult,
    compute_active_hygiene_weights,
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
) -> SqliteEngravaCore:
    """Build a bootstrapped in-memory store carrying the given hygiene policy."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(conn, hygiene_policy=policy, journal_enabled=journal_enabled)
    await s.ensure_schema()
    return s


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
    pinned: bool = False,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
) -> ThoughtRecord:
    """Build a thought with hygiene-relevant fields controllable per test."""
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
        pinned=pinned,
    )


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
        with pytest.raises(TypeError, match="eviction_threshold"):
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

    def test_protected_priorities_non_string_raises(self) -> None:
        with pytest.raises(TypeError, match="protected_priorities"):
            HygienePolicyConfig(protected_priorities=(1,))  # type: ignore[arg-type]

    def test_signal_weight_non_numeric_raises(self) -> None:
        with pytest.raises(TypeError, match="signal_weights"):
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
            assert (await cursor.fetchone())[0] == 18
            cursor = await conn.execute("PRAGMA table_info(thought)")
            cols = {row["name"] for row in await cursor.fetchall()}
            assert "pinned" in cols
            assert "archived_at_cycle" in cols
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
            assert (await cursor.fetchone())[0] == 18

            fetched = await s.get_thought("legacy")
            assert fetched is not None
            assert fetched.pinned is False
            assert fetched.archived_at_cycle is None
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
        """With a policy present but disabled, consolidate() never archives."""
        policy = HygienePolicyConfig(enabled=False, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            # A disabled policy makes ``_hygiene_due`` false regardless of cycle.
            assert s._hygiene_due(1) is False
            assert await _raw_lifecycle(s, "cold") == "ACTIVE"
        finally:
            await s._db.close()

    async def test_disabled_policy_direct_run_hygiene_is_a_noop(self) -> None:
        """``enabled`` is a hard master switch — a direct call also never forgets."""
        policy = HygienePolicyConfig(enabled=False, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 0
            assert result.gc_count == 0
            assert await _raw_lifecycle(s, "cold") == "ACTIVE"
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Keep-score + eviction rule (+ redistribution)
# ---------------------------------------------------------------------------


class TestKeepScoreAndEviction:
    async def test_cold_low_value_thought_archived(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=0.5)
        s = await _make_store(policy)
        try:
            # Old, never-accessed, low confidence -> low keep-score.
            await s.create_thought(_thought("cold", updated_cycle=0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_recently_accessed_thought_survives(self) -> None:
        """A thought that scores high on recency stays regardless of the threshold."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=0.5)
        s = await _make_store(policy)
        try:
            # updated_cycle == current_cycle -> recency ~= 1.0.
            await s.create_thought(_thought("warm", updated_cycle=1000))
            await s.create_thought(_thought("cold", updated_cycle=0))
            result = await s.run_hygiene(current_cycle=1000)
            assert await _raw_lifecycle(s, "warm") == "ACTIVE"
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            assert result.archived_count == 1
        finally:
            await s._db.close()

    async def test_created_lifecycle_is_a_candidate(self) -> None:
        """CREATED thoughts are in the candidate set (not only ACTIVE)."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=0.5)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("created", lifecycle_status=LifecycleStatus.CREATED, updated_cycle=0)
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            for i in range(5):
                await s.create_thought(_thought(f"t{i}", updated_cycle=0))
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("pin", pinned=True, updated_cycle=0))
            result = await s.run_hygiene(current_cycle=10_000)
            assert result.archived_count == 0
            assert await _raw_lifecycle(s, "pin") == "ACTIVE"
        finally:
            await s._db.close()

    async def test_p1_protected_by_default(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("p1", priority=Priority.P1, updated_cycle=0))
            await s.create_thought(_thought("p3", priority=Priority.P3, updated_cycle=0))
            await s.run_hygiene(current_cycle=10_000)
            assert await _raw_lifecycle(s, "p1") == "ACTIVE"
            assert await _raw_lifecycle(s, "p3") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_confidence_is_not_protection(self) -> None:
        """A high-confidence but cold thought is NOT protected by confidence."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(
                _thought("confident", confidence=1.0, priority=Priority.P3, updated_cycle=0)
            )
            result = await s.run_hygiene(current_cycle=10_000)
            assert result.archived_count == 1
            assert await _raw_lifecycle(s, "confident") == "ARCHIVED"
        finally:
            await s._db.close()

    async def test_protected_priorities_configurable_off(self) -> None:
        """Setting protected_priorities to () lets P1 be archived."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, protected_priorities=())
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("p1", priority=Priority.P1, updated_cycle=0))
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
        """A thought protected after selection is skipped at the archive write (TOCTOU)."""
        from engrava.infrastructure.sqlite.hygiene import EvictionReason

        policy = HygienePolicyConfig(enabled=True)
        s = await _make_store(policy)
        try:
            # Feed a protected thought straight to the archive stage (selection
            # already skips protected) to exercise the write-time re-check.
            await s.create_thought(_thought("pinned_late", pinned=True, updated_cycle=0))
            reason = EvictionReason(
                thought_id="pinned_late",
                keep_score=0.0,
                eviction_score=0.0,
                decay_multiplier=1.0,
                threshold=0.2,
            )
            archived = await s._hygiene_archive([reason], policy=policy, current_cycle=1000)
            assert archived == 0
            assert await _raw_lifecycle(s, "pinned_late") == "ACTIVE"
        finally:
            await s._db.close()


# ---------------------------------------------------------------------------
# Archive reversibility + restore clears archived_at_cycle
# ---------------------------------------------------------------------------


class TestArchiveReversible:
    async def test_archive_sets_archived_at_cycle_and_clears_expires(self) -> None:
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            await s.run_hygiene(current_cycle=42)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
            assert await _raw_archived_at_cycle(s, "cold") == 42
        finally:
            await s._db.close()

    async def test_restore_clears_archived_at_cycle(self) -> None:
        """restore_thought un-archives a hygiene-archived thought and clears the stamp."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            await s.run_hygiene(current_cycle=42)
            updated = await s.update_thought("cold", lifecycle_status=LifecycleStatus.ACTIVE)
            assert updated.lifecycle_status is LifecycleStatus.ACTIVE
        finally:
            await s._db.close()

    async def test_restore_round_trip_preserves_content(self) -> None:
        """archive -> restore round-trips with no data loss."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy)
        try:
            original = _thought("cold", updated_cycle=0)
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
            policy = HygienePolicyConfig(
                enabled=True, eviction_threshold=0.5, max_evictions_per_run=3, dry_run=True
            )
            s = await _make_store(policy)
            try:
                for i in range(10):
                    await s.create_thought(_thought(f"t{i:02d}", updated_cycle=0))
                result = await s.run_hygiene(current_cycle=1000)
                ids_seen.append([r.thought_id for r in result.would_evict])
            finally:
                await s._db.close()
        assert ids_seen[0] == ids_seen[1]

    async def test_cap_enforced_per_stage(self) -> None:
        """No more than max_evictions_per_run thoughts are archived per run."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=0.9, max_evictions_per_run=2)
        s = await _make_store(policy)
        try:
            for i in range(5):
                await s.create_thought(_thought(f"t{i}", updated_cycle=0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 2
        finally:
            await s._db.close()

    async def test_cap_selects_the_coldest_candidates(self) -> None:
        """Under the cap the *coldest* thoughts are archived, not an arbitrary set."""
        policy = HygienePolicyConfig(
            enabled=True, eviction_threshold=1.0, max_evictions_per_run=2, dry_run=True
        )
        s = await _make_store(policy)
        try:
            # Different recency -> different eviction-score; the two oldest
            # (lowest keep-score) must be the ones selected under a cap of 2.
            for tid, updated in (("warm", 1000), ("mid", 500), ("cold1", 0), ("cold2", 10)):
                await s.create_thought(_thought(tid, updated_cycle=updated))
            result = await s.run_hygiene(current_cycle=1000)
            assert sorted(r.thought_id for r in result.would_evict) == ["cold1", "cold2"]
        finally:
            await s._db.close()

    async def test_tiebreak_order_is_stable(self) -> None:
        """Under equal scores the id tiebreak makes the selection deterministic."""
        # All ten thoughts share updated_cycle and score identically -> the
        # eviction_score + updated_cycle tie is broken by thought_id ASC.
        policy = HygienePolicyConfig(
            enabled=True, eviction_threshold=0.9, max_evictions_per_run=3, dry_run=True
        )
        s = await _make_store(policy)
        try:
            for i in range(10):
                await s.create_thought(_thought(f"id-{i:02d}", updated_cycle=0))
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
            # configured signal and it is flat -> empty active set.
            await s.create_thought(_thought("t", updated_cycle=0))
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=0.5)
        s = SqliteEngravaCore(conn, hooks=_DecayHooks(bad_decay), hygiene_policy=policy)
        await s.ensure_schema()
        try:
            # A warm thought (recency ~1.0) has keep_score above threshold.
            await s.create_thought(_thought("warm", updated_cycle=1000))
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=0.5)
        s = SqliteEngravaCore(conn, hooks=_DecayHooks(-2.0), hygiene_policy=policy)
        await s.ensure_schema()
        try:
            # Decay 0 -> eviction_score 0 < threshold -> archived (but not an
            # error, and the score is a clean 0.0 in the reason).
            await s.create_thought(_thought("warm", updated_cycle=1000))
            policy_dry = HygienePolicyConfig(enabled=True, eviction_threshold=0.5, dry_run=True)
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, auto_gc_enabled=False)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            result = await s.run_hygiene(current_cycle=1000)
            assert result.archived_count == 1
            assert result.gc_count == 0
            assert await s.get_thought("cold") is not None  # still present, archived
        finally:
            await s._db.close()

    async def test_gc_respects_restore_window(self) -> None:
        """A just-archived thought is not GC'd until the window elapses."""
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
        )
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=10,
        )
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, dry_run=True)
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, dry_run=True)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0)
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
        )
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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

        policy = HygienePolicyConfig(
            enabled=True,
            eviction_threshold=1.0,
            auto_gc_enabled=True,
            gc_min_archive_age_cycles=0,
        )
        s = await _make_store(policy, journal_enabled=True)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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

        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=1)
        s = SqliteEngravaCore(conn, hygiene_policy=policy)
        s._dreaming_extension = DreamingExtension(config=DreamingConfig(enabled=True))
        await s.ensure_schema()
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            await s.consolidate(current_cycle=1000)
            assert await _raw_lifecycle(s, "cold") == "ARCHIVED"
        finally:
            await conn.close()

    async def test_consolidate_skips_hygiene_off_cadence(self) -> None:
        """When the cycle misses the cadence, consolidate() does not forget."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        from engrava.extensions.dreaming import DreamingExtension

        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=5)
        s = SqliteEngravaCore(conn, hygiene_policy=policy)
        s._dreaming_extension = DreamingExtension(config=DreamingConfig(enabled=True))
        await s.ensure_schema()
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
            # cycle 7 % 5 != 0 -> hygiene skipped.
            await s.consolidate(current_cycle=7)
            assert await _raw_lifecycle(s, "cold") == "ACTIVE"
            assert s._hygiene_due(7) is False
            assert s._hygiene_due(10) is True
        finally:
            await conn.close()

    async def test_run_hygiene_bypasses_cadence(self) -> None:
        """An explicit run_hygiene ignores check_every_n_cycles."""
        policy = HygienePolicyConfig(enabled=True, eviction_threshold=1.0, check_every_n_cycles=100)
        s = await _make_store(policy)
        try:
            await s.create_thought(_thought("cold", updated_cycle=0))
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
            """,
            encoding="utf-8",
        )
        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            assert store._hygiene_policy is not None
            assert store._hygiene_policy.enabled is True
            assert store._hygiene_policy.eviction_threshold == pytest.approx(0.5)
            await store.create_thought(_thought("cold", updated_cycle=0))
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

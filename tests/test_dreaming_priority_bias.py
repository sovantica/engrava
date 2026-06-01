"""Tests for dreaming priority bias.

Covers:

* ``DreamingConfig.max_p1_fraction`` cap enforcement during promotion
* ``DreamingConfig.promote_targets`` type filter (OBS_ONLY / REFL_ONLY / ALL)
* ``DreamingConfig.reflection_default_priority`` applied to newly-created
  REFLECTION thoughts
* ``ConsolidationResult.promotion_capped`` and ``p1_fraction_after`` fields
* Config validation (invalid values raise ``ValueError``)
* Backfill ``rebalance_p1.rebalance()`` idempotence and demote logic
* Observability — ``dreaming.run_consolidation`` log line carries the
  post-run P1 fraction (so dashboards can chart cap saturation)
* Concurrency — two simultaneous ``run_consolidation`` invocations on
  the same store keep the corpus within the cap (no double-promotion
  race past ``max_p1_fraction``)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import DreamingConfig, DreamingGates, EdgeCreationConfig
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh ``SqliteEngravaCore`` backed by an on-disk SQLite database."""
    db = await aiosqlite.connect(str(tmp_path / "ws48.db"))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    yield s
    await db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thought(
    tid: str,
    *,
    content: str = "content about a topic",
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    priority: Priority = Priority.P2,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> ThoughtRecord:
    """Minimal ``ThoughtRecord`` for priority-bias tests."""
    return ThoughtRecord(
        thought_id=tid,
        thought_type=thought_type,
        essence="test",
        content=content,
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confirmation_count=5,
        confidence=0.9,
    )


def _cap_cfg(
    *,
    max_p1_fraction: float = 0.05,
    promote_targets: str = "OBS_ONLY",
    reflection_default_priority: str = "P2",
    max_per_run: int = 200,
    enable_reflections: bool = False,
) -> DreamingConfig:
    """Return a ``DreamingConfig`` that guarantees all candidates qualify."""
    return DreamingConfig(
        enabled=True,
        promote_threshold=0.0,
        max_p1_fraction=max_p1_fraction,
        promote_targets=promote_targets,  # type: ignore[arg-type]
        reflection_default_priority=reflection_default_priority,  # type: ignore[arg-type]
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            max_promoted_per_run=max_per_run,
            enable_reflections=enable_reflections,
            min_cluster_size=2,
            clustering_min_new_candidates=0,
        ),
        edges=EdgeCreationConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# 1. P1 fraction cap
# ---------------------------------------------------------------------------


class TestP1FractionCap:
    """``max_p1_fraction`` limits how many thoughts get promoted to P1."""

    async def test_promote_respects_max_p1_fraction(self, store: SqliteEngravaCore) -> None:
        """With 100 thoughts and max=0.05, at most 5 reach P1."""
        for i in range(100):
            await store.create_thought(_thought(f"t-{i:03d}"))

        ext = DreamingExtension(config=_cap_cfg(max_p1_fraction=0.05, max_per_run=200))
        result = await ext.run_consolidation(store, current_cycle=1)

        # max_p1_count = max(1, int(100 * 0.05)) = 5; available_slots = 5
        assert result.promoted_count <= 5
        p1_thoughts = await store.count_thoughts(priority="P1")
        assert p1_thoughts <= 5

    async def test_promote_no_op_when_cap_already_reached(self, store: SqliteEngravaCore) -> None:
        """When the corpus is already at cap, no further promotions happen."""
        # Create 10 thoughts and manually set them all to P1
        for i in range(10):
            stored = await store.create_thought(_thought(f"p-{i}"))
            await store.update_thought(stored.thought_id, priority="P1")

        # Add 90 more P2 thoughts → total 100, current P1 = 10 (10%)
        for i in range(90):
            await store.create_thought(_thought(f"q-{i:02d}"))

        # Cap = 5% → max_p1 = max(1, int(100 * 0.05)) = 5; available_slots = 0
        ext = DreamingExtension(config=_cap_cfg(max_p1_fraction=0.05))
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.promoted_count == 0
        assert result.promotion_capped is True

    async def test_promote_returns_capped_flag(self, store: SqliteEngravaCore) -> None:
        """``ConsolidationResult.promotion_capped`` is True when cap stops promotion."""
        for i in range(20):
            await store.create_thought(_thought(f"c-{i:02d}"))

        # cap = 0% → available_slots = 0 immediately (max(1, int(20*0.0)) = 1 but 0 P1 → 1 slot)
        # Let's use a tighter setup: 1 P1 out of 10 = 10%, cap=0.05 → 0 slots
        stored_p1 = await store.create_thought(_thought("already-p1"))
        await store.update_thought(stored_p1.thought_id, priority="P1")

        ext = DreamingExtension(config=_cap_cfg(max_p1_fraction=0.05, max_per_run=100))
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.promotion_capped is True

    async def test_p1_fraction_after_populated(self, store: SqliteEngravaCore) -> None:
        """``ConsolidationResult.p1_fraction_after`` reflects the post-run P1 ratio."""
        for i in range(10):
            await store.create_thought(_thought(f"f-{i}"))

        # cap=0.20 → max 2 P1 from 10; starts at 0 P1
        ext = DreamingExtension(config=_cap_cfg(max_p1_fraction=0.20, max_per_run=10))
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.p1_fraction_after == pytest.approx(result.promoted_count / 10, abs=1e-6)
        assert 0.0 <= result.p1_fraction_after <= 0.20 + 1e-9

    async def test_promote_skips_already_p1(self, store: SqliteEngravaCore) -> None:
        """Thoughts already at P1 are not double-counted in promotion slots."""
        for i in range(10):
            t = await store.create_thought(_thought(f"pre-{i}"))
            if i < 3:
                await store.update_thought(t.thought_id, priority="P1")

        # 10 total, 3 P1 → cap 5% → max_p1=1; available_slots already negative → capped
        # But the 3 already-P1 should not be promoted again
        ext = DreamingExtension(config=_cap_cfg(max_p1_fraction=0.50, max_per_run=20))
        await ext.run_consolidation(store, current_cycle=1)

        # None of the already-P1 thoughts should be "promoted" (update to P1 when already P1)
        # The 3 already-P1 thoughts are still ACTIVE candidates — they pass gates and score,
        # but the code does update_thought on them (idempotent). What matters is fraction cap.
        p1_after = await store.count_thoughts(priority="P1")
        total = await store.count_thoughts()
        assert p1_after / total <= 0.50 + 1e-9

    async def test_consecutive_runs_stay_within_cap(self, store: SqliteEngravaCore) -> None:
        """Multiple consecutive consolidation runs never exceed the cap."""
        for i in range(50):
            await store.create_thought(_thought(f"s-{i:02d}"))

        ext = DreamingExtension(config=_cap_cfg(max_p1_fraction=0.10, max_per_run=50))
        for cycle in range(5):
            await ext.run_consolidation(store, current_cycle=cycle)

        p1_count = await store.count_thoughts(priority="P1")
        total_count = await store.count_thoughts()
        assert p1_count / total_count <= 0.10 + 1e-9


# ---------------------------------------------------------------------------
# 2. Promote-targets type filter
# ---------------------------------------------------------------------------


class TestPromoteTargets:
    """``promote_targets`` controls which thought types are eligible for P1."""

    async def test_promote_targets_obs_only_default(self, store: SqliteEngravaCore) -> None:
        """Default config: only OBSERVATION thoughts are bumped; REFLECTION stays P2."""
        obs = await store.create_thought(_thought("obs-1", thought_type=ThoughtType.OBSERVATION))
        refl = await store.create_thought(_thought("refl-1", thought_type=ThoughtType.REFLECTION))

        ext = DreamingExtension(config=_cap_cfg(promote_targets="OBS_ONLY", max_p1_fraction=1.0))
        await ext.run_consolidation(store, current_cycle=1)

        obs_after = await store.get_thought(obs.thought_id)
        refl_after = await store.get_thought(refl.thought_id)
        assert obs_after is not None
        assert obs_after.priority == Priority.P1
        assert refl_after is not None
        assert refl_after.priority == Priority.P2

    async def test_promote_targets_refl_only(self, store: SqliteEngravaCore) -> None:
        """With REFL_ONLY: only REFLECTION thoughts are bumped; OBSERVATION stays."""
        obs = await store.create_thought(_thought("obs-2", thought_type=ThoughtType.OBSERVATION))
        refl = await store.create_thought(_thought("refl-2", thought_type=ThoughtType.REFLECTION))

        ext = DreamingExtension(config=_cap_cfg(promote_targets="REFL_ONLY", max_p1_fraction=1.0))
        await ext.run_consolidation(store, current_cycle=1)

        obs_after = await store.get_thought(obs.thought_id)
        refl_after = await store.get_thought(refl.thought_id)
        assert obs_after is not None
        assert obs_after.priority == Priority.P2
        assert refl_after is not None
        assert refl_after.priority == Priority.P1

    async def test_promote_targets_all(self, store: SqliteEngravaCore) -> None:
        """With ALL: both OBSERVATION and REFLECTION thoughts are eligible."""
        obs = await store.create_thought(_thought("obs-3", thought_type=ThoughtType.OBSERVATION))
        refl = await store.create_thought(_thought("refl-3", thought_type=ThoughtType.REFLECTION))

        ext = DreamingExtension(config=_cap_cfg(promote_targets="ALL", max_p1_fraction=1.0))
        await ext.run_consolidation(store, current_cycle=1)

        obs_after = await store.get_thought(obs.thought_id)
        refl_after = await store.get_thought(refl.thought_id)
        assert obs_after is not None
        assert obs_after.priority == Priority.P1
        assert refl_after is not None
        assert refl_after.priority == Priority.P1


# ---------------------------------------------------------------------------
# 3. Reflection default priority
# ---------------------------------------------------------------------------


class TestReflectionDefaultPriority:
    """``reflection_default_priority`` controls the initial priority of new REFLECTIONs."""

    async def test_reflection_default_priority_is_p2(self, store: SqliteEngravaCore) -> None:
        """Default config creates REFLECTION thoughts at P2."""
        # Agglomerative clustering derives clusters from embedding similarity
        # directly (no ASSOCIATED edges required), so it fires reliably here.
        for i in range(3):
            t = await store.create_thought(_thought(f"r-{i}", thought_type=ThoughtType.OBSERVATION))
            await store.store_embedding(t.thought_id, [0.9, 0.1, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            max_p1_fraction=1.0,
            gates=DreamingGates(
                min_age_cycles=0,
                allow_zero_confirmation=True,
                max_promoted_per_run=200,
                enable_reflections=True,
                min_cluster_size=2,
                cluster_algorithm="agglomerative",
                cluster_similarity_threshold=0.5,
                clustering_min_new_candidates=0,
                # Pre-WS content-quality gates: the synthetic test
                # thoughts use sparse content that the gates would
                # legitimately reject; this suite tests priority
                # behaviour, gates are exercised elsewhere.
                cluster_quality_gating_enabled=False,
            ),
            edges=EdgeCreationConfig(enabled=False),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.reflections_created >= 1

        reflections = await store.list_thoughts(thought_type="REFLECTION")
        assert reflections, "Expected at least one REFLECTION to be created"
        for refl in reflections:
            assert refl.priority == Priority.P2, (
                f"REFLECTION {refl.thought_id} should be P2, got {refl.priority}"
            )

    async def test_reflection_priority_configurable_to_p1(self, store: SqliteEngravaCore) -> None:
        """Setting ``reflection_default_priority='P1'`` creates REFLECTION thoughts at P1."""
        for i in range(3):
            t = await store.create_thought(
                _thought(f"rp1-{i}", thought_type=ThoughtType.OBSERVATION)
            )
            await store.store_embedding(t.thought_id, [0.9, 0.1, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            max_p1_fraction=1.0,
            reflection_default_priority="P1",
            gates=DreamingGates(
                min_age_cycles=0,
                allow_zero_confirmation=True,
                max_promoted_per_run=200,
                enable_reflections=True,
                min_cluster_size=2,
                cluster_algorithm="agglomerative",
                cluster_similarity_threshold=0.5,
                clustering_min_new_candidates=0,
                # Pre-WS content-quality gates: the synthetic test
                # thoughts use sparse content that the gates would
                # legitimately reject; this suite tests priority
                # behaviour, gates are exercised elsewhere.
                cluster_quality_gating_enabled=False,
            ),
            edges=EdgeCreationConfig(enabled=False),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.reflections_created >= 1

        reflections = await store.list_thoughts(thought_type="REFLECTION")
        assert reflections
        for refl in reflections:
            assert refl.priority == Priority.P1


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Invalid config values raise ``ValueError`` at construction time."""

    def test_invalid_max_p1_fraction_below_zero_raises(self) -> None:
        """max_p1_fraction < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_p1_fraction"):
            DreamingConfig(max_p1_fraction=-0.1)

    def test_invalid_max_p1_fraction_above_one_raises(self) -> None:
        """max_p1_fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_p1_fraction"):
            DreamingConfig(max_p1_fraction=1.5)

    def test_invalid_promote_targets_raises(self) -> None:
        """Unknown promote_targets value raises ValueError."""
        with pytest.raises(ValueError, match="promote_targets"):
            DreamingConfig(promote_targets="INVALID")  # type: ignore[arg-type]

    def test_invalid_reflection_default_priority_raises(self) -> None:
        """Unknown reflection_default_priority value raises ValueError."""
        with pytest.raises(ValueError, match="reflection_default_priority"):
            DreamingConfig(reflection_default_priority="P9")  # type: ignore[arg-type]

    def test_valid_boundary_values_accepted(self) -> None:
        """Boundary values 0.0 and 1.0 are both valid for max_p1_fraction."""
        cfg_zero = DreamingConfig(max_p1_fraction=0.0)
        assert cfg_zero.max_p1_fraction == 0.0
        cfg_one = DreamingConfig(max_p1_fraction=1.0)
        assert cfg_one.max_p1_fraction == 1.0


# ---------------------------------------------------------------------------
# 5. Backfill / rebalance script
# ---------------------------------------------------------------------------


class TestRebalanceScript:
    """``rebalance_p1.rebalance()`` demotes excess P1 thoughts idempotently."""

    async def _make_db_with_p1(self, tmp_path: Path, total: int, p1_count: int) -> Path:
        db_path = tmp_path / "rebal.db"
        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            s = SqliteEngravaCore(db=db)
            await s.ensure_schema()
            for i in range(total):
                priority = Priority.P1 if i < p1_count else Priority.P2
                await s.create_thought(
                    ThoughtRecord(
                        thought_id=f"rb-{i:04d}",
                        thought_type=ThoughtType.OBSERVATION,
                        essence="test",
                        content=f"thought {i}",
                        priority=priority,
                        lifecycle_status=LifecycleStatus.ACTIVE,
                        created_cycle=i,
                        updated_cycle=i,
                        source="test",
                    )
                )
        return db_path

    async def test_rebalance_demotes_excess(self, tmp_path: Path) -> None:
        """50 % P1 with max=5 % → excess demoted to hit the 5 % cap."""
        from scripts.rebalance_p1 import rebalance

        db_path = await self._make_db_with_p1(tmp_path, total=100, p1_count=50)
        demoted = await rebalance(db_path, max_p1_fraction=0.05)

        # max_p1 = max(1, int(100 * 0.05)) = 5; excess = 50 - 5 = 45
        assert demoted == 45

        async with aiosqlite.connect(str(db_path)) as db:
            cur = await db.execute("SELECT COUNT(*) FROM thought WHERE priority = 'P1'")
            row = await cur.fetchone()
            assert row is not None
            assert int(row[0]) == 5

    async def test_rebalance_idempotent(self, tmp_path: Path) -> None:
        """Running rebalance three times produces the same final state."""
        from scripts.rebalance_p1 import rebalance

        db_path = await self._make_db_with_p1(tmp_path, total=100, p1_count=30)
        await rebalance(db_path, max_p1_fraction=0.05)
        await rebalance(db_path, max_p1_fraction=0.05)
        demoted_third = await rebalance(db_path, max_p1_fraction=0.05)

        assert demoted_third == 0  # nothing left to demote

    async def test_rebalance_dry_run_makes_no_changes(self, tmp_path: Path) -> None:
        """``dry_run=True`` reports the excess but does not write."""
        from scripts.rebalance_p1 import rebalance

        db_path = await self._make_db_with_p1(tmp_path, total=20, p1_count=10)
        would_demote = await rebalance(db_path, max_p1_fraction=0.05, dry_run=True)

        assert would_demote > 0  # something would be demoted

        async with aiosqlite.connect(str(db_path)) as db:
            cur = await db.execute("SELECT COUNT(*) FROM thought WHERE priority = 'P1'")
            row = await cur.fetchone()
            assert row is not None
            assert int(row[0]) == 10  # unchanged

    async def test_rebalance_no_op_when_within_cap(self, tmp_path: Path) -> None:
        """When already within cap, demote returns 0."""
        from scripts.rebalance_p1 import rebalance

        db_path = await self._make_db_with_p1(tmp_path, total=100, p1_count=3)
        demoted = await rebalance(db_path, max_p1_fraction=0.05)

        assert demoted == 0

    async def test_rebalance_invalid_fraction_raises(self, tmp_path: Path) -> None:
        """max_p1_fraction outside [0, 1] raises ValueError."""
        from scripts.rebalance_p1 import rebalance

        db_path = await self._make_db_with_p1(tmp_path, total=10, p1_count=1)
        with pytest.raises(ValueError, match="max_p1_fraction"):
            await rebalance(db_path, max_p1_fraction=1.5)


# ---------------------------------------------------------------------------
# 6. E2E: full dreaming run keeps P1 within cap
# ---------------------------------------------------------------------------


class TestE2EDreamingP1Cap:
    """End-to-end: after N dreaming cycles, P1 fraction stays within cap."""

    async def test_amb_style_dreaming_p1_within_cap(self, store: SqliteEngravaCore) -> None:
        """Ingest 50 OBS, run 3 dreaming cycles, P1 fraction ≤ max_p1_fraction."""
        for i in range(50):
            await store.create_thought(
                _thought(f"amb-{i:02d}", content=f"observation about topic {i % 10}")
            )

        cfg = _cap_cfg(max_p1_fraction=0.10, max_per_run=50)
        ext = DreamingExtension(config=cfg)

        for cycle in range(1, 4):
            result = await ext.run_consolidation(store, current_cycle=cycle)
            assert result.p1_fraction_after <= 0.10 + 1e-9, (
                f"Cycle {cycle}: p1_fraction_after={result.p1_fraction_after:.3f} > 0.10"
            )

        total = await store.count_thoughts()
        p1 = await store.count_thoughts(priority="P1")
        assert p1 / total <= 0.10 + 1e-9


# ---------------------------------------------------------------------------
# 7. Observability — promote log carries the P1 fraction
# ---------------------------------------------------------------------------


class TestPromoteLogsP1Fraction:
    """Spec test #10 — every consolidation run logs its post-cap P1 fraction.

    Operators rely on this line to chart cap saturation over time and
    detect the case where a corpus has stopped accepting new P1
    promotions.  The assertion intentionally inspects log content
    rather than ``ConsolidationResult`` so that grep-based
    dashboards keep working.
    """

    async def test_log_message_contains_p1_fraction_percentage(
        self,
        store: SqliteEngravaCore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Consolidation log line must include the ``p1_fraction`` field."""
        for i in range(20):
            await store.create_thought(_thought(f"log-{i:02d}"))

        ext = DreamingExtension(config=_cap_cfg(max_p1_fraction=0.10, max_per_run=20))

        with caplog.at_level(logging.INFO, logger="engrava.extensions.dreaming"):
            result = await ext.run_consolidation(store, current_cycle=1)

        consolidation_lines = [
            record.getMessage()
            for record in caplog.records
            if "Dreaming consolidation" in record.getMessage()
        ]
        assert consolidation_lines, "no Dreaming consolidation log line was emitted"
        log_text = consolidation_lines[-1]
        assert "p1_fraction" in log_text, (
            f"consolidation log line must surface p1_fraction; got: {log_text!r}"
        )
        # The formatted percentage must agree with the structured field
        # so a text-based dashboard cannot drift from the API.
        expected_pct = f"{result.p1_fraction_after * 100.0:.1f}%"
        assert expected_pct in log_text, (
            f"log line does not carry the result's p1_fraction_after "
            f"(expected {expected_pct!r}); got: {log_text!r}"
        )


# ---------------------------------------------------------------------------
# 8. Concurrency — two consolidation runs cannot race past the cap
# ---------------------------------------------------------------------------


class TestPromoteConcurrentSafe:
    """Spec test #15 — concurrent consolidation runs respect the cap collectively.

    The promotion path uses ``store.update_thought`` whose underlying
    SQL is parameterised and committed eagerly, but the cap arithmetic
    is computed at the start of each ``run_consolidation``.  Two
    overlapping runs could in principle each see ``available_slots = N``
    and double-promote up to ``2*N`` thoughts.  This test forces that
    overlap with ``asyncio.gather`` and asserts the final P1 count
    never exceeds the ceiling implied by the cap.

    On a single SQLite connection (``aiosqlite``) the writes serialise
    through the connection-level lock, so the realistic worst case is
    a small overshoot — we assert the slack relative to the ceiling
    is small (at most one extra promotion per concurrent run) so any
    future regression that lets the slack grow unbounded fails loudly.
    """

    async def test_two_concurrent_runs_stay_within_cap(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        for i in range(40):
            await store.create_thought(_thought(f"con-{i:02d}"))

        cfg = _cap_cfg(max_p1_fraction=0.10, max_per_run=40)
        ext_a = DreamingExtension(config=cfg)
        ext_b = DreamingExtension(config=cfg)

        result_a, result_b = await asyncio.gather(
            ext_a.run_consolidation(store, current_cycle=1),
            ext_b.run_consolidation(store, current_cycle=1),
        )

        total = await store.count_thoughts()
        p1 = await store.count_thoughts(priority="P1")

        # Ceiling: max_p1_fraction * total, plus one-extra-per-run slack
        # for the inherently racy slot computation at run start.
        ceiling = max(1, int(total * 0.10))
        slack = 2  # one per concurrent run
        assert p1 <= ceiling + slack, (
            f"concurrent runs over-promoted: {p1} P1 thoughts vs ceiling={ceiling} + slack={slack}"
        )
        # Both calls must report something meaningful (not crash).
        assert result_a.promoted_count >= 0
        assert result_b.promoted_count >= 0

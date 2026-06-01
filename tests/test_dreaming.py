"""Tests for dreaming signals and DreamingExtension."""

from __future__ import annotations

import math

import pytest

from engrava.config import DreamingConfig, DreamingGates
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import ConsolidationResult, DreamingExtension
from engrava.extensions.dreaming_signals import (
    ConfidenceSignal,
    ConfirmationSignal,
    DreamingContext,
    DreamingSignalProtocol,
    RecencySignal,
    StalenessSignal,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_thought(
    thought_id: str = "t1",
    *,
    created_cycle: int = 0,
    updated_cycle: int = 0,
    confirmation_count: int = 0,
    confidence: float | None = None,
    lifecycle_status: str = "ACTIVE",
    priority: str = "P2",
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="test thought",
        content="test content for dreaming",
        priority=Priority(priority),
        lifecycle_status=LifecycleStatus(lifecycle_status),
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confirmation_count=confirmation_count,
        confidence=confidence,
    )


# ------------------------------------------------------------------
# Signal tests
# ------------------------------------------------------------------


class TestRecencySignal:
    def test_recent_thought_high_score(self) -> None:
        sig = RecencySignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(updated_cycle=99)
        assert sig(t, ctx) == pytest.approx(math.exp(-0.01 * 1))

    def test_old_thought_low_score(self) -> None:
        sig = RecencySignal()
        ctx = DreamingContext(current_cycle=500, total_thoughts=50)
        t = _make_thought(updated_cycle=0)
        assert sig(t, ctx) == pytest.approx(math.exp(-0.01 * 500))

    def test_same_cycle_score_is_one(self) -> None:
        sig = RecencySignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(updated_cycle=100)
        assert sig(t, ctx) == pytest.approx(1.0)

    def test_custom_decay_rate(self) -> None:
        sig = RecencySignal(decay_rate=0.1)
        ctx = DreamingContext(current_cycle=10, total_thoughts=50)
        t = _make_thought(updated_cycle=0)
        assert sig(t, ctx) == pytest.approx(math.exp(-1.0))


class TestStalenessSignal:
    def test_zero_span(self) -> None:
        sig = StalenessSignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(created_cycle=0, updated_cycle=0)
        assert sig(t, ctx) == 0.0

    def test_half_span(self) -> None:
        sig = StalenessSignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(created_cycle=0, updated_cycle=50)
        assert sig(t, ctx) == 0.5

    def test_full_span_clamps_to_one(self) -> None:
        sig = StalenessSignal()
        ctx = DreamingContext(current_cycle=200, total_thoughts=50)
        t = _make_thought(created_cycle=0, updated_cycle=200)
        assert sig(t, ctx) == 1.0

    def test_custom_max_span(self) -> None:
        sig = StalenessSignal(max_span=50)
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(created_cycle=0, updated_cycle=50)
        assert sig(t, ctx) == 1.0


class TestConfirmationSignal:
    def test_zero_confirmations(self) -> None:
        sig = ConfirmationSignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(confirmation_count=0)
        assert sig(t, ctx) == 0.0

    def test_partial_confirmations(self) -> None:
        sig = ConfirmationSignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(confirmation_count=3)
        assert sig(t, ctx) == pytest.approx(0.6)

    def test_max_confirmations_clamps(self) -> None:
        sig = ConfirmationSignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(confirmation_count=10)
        assert sig(t, ctx) == 1.0


class TestConfidenceSignal:
    def test_with_confidence(self) -> None:
        sig = ConfidenceSignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(confidence=0.8)
        assert sig(t, ctx) == 0.8

    def test_none_confidence_defaults(self) -> None:
        sig = ConfidenceSignal()
        ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        t = _make_thought(confidence=None)
        assert sig(t, ctx) == 0.5


class TestDreamingSignalProtocol:
    def test_runtime_checkable(self) -> None:
        class MySignal:
            def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
                return 0.42

        assert isinstance(MySignal(), DreamingSignalProtocol)


# ------------------------------------------------------------------
# DreamingExtension tests
# ------------------------------------------------------------------


class TestDreamingExtension:
    def test_init_with_defaults(self) -> None:
        cfg = DreamingConfig(enabled=True)
        ext = DreamingExtension(config=cfg)
        assert ext.config.enabled is True
        assert len(ext._signals) == 5

    def test_unknown_signal_raises(self) -> None:
        cfg = DreamingConfig(
            enabled=True,
            signals={"unknown_signal": 1.0},
        )
        with pytest.raises(ValueError, match=r"Unknown dreaming signal.*unknown_signal"):
            DreamingExtension(config=cfg)

    def test_custom_signal_overrides_default(self) -> None:
        class ConstantSignal:
            def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
                return 0.99

        cfg = DreamingConfig(
            enabled=True,
            signals={"recency": 1.0},
        )
        ext = DreamingExtension(
            config=cfg,
            custom_signals={"recency": ConstantSignal()},
        )
        t = _make_thought()
        ctx = DreamingContext(current_cycle=1000, total_thoughts=1)
        score = ext._compute_score(t, ctx)
        assert score == pytest.approx(0.99)

    def test_custom_signal_extends_registry(self) -> None:
        class EmotionalSignal:
            def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
                return 0.7

        cfg = DreamingConfig(
            enabled=True,
            signals={"emotional_weight": 1.0},
        )
        ext = DreamingExtension(
            config=cfg,
            custom_signals={"emotional_weight": EmotionalSignal()},
        )
        t = _make_thought()
        ctx = DreamingContext(current_cycle=100, total_thoughts=1)
        score = ext._compute_score(t, ctx)
        assert score == pytest.approx(0.7)


class TestConsolidationResult:
    def test_frozen(self) -> None:
        result = ConsolidationResult(
            candidates_evaluated=10,
            promoted_count=2,
            promoted_ids=["t1", "t2"],
        )
        with pytest.raises(AttributeError):
            result.promoted_count = 5  # type: ignore[misc]


# ------------------------------------------------------------------
# Integration: run_consolidation with real SqliteEngravaCore
# ------------------------------------------------------------------


class TestRunConsolidation:
    async def test_promotes_qualifying_thoughts(self) -> None:
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            # Create thoughts: one qualifying, one not
            qualifying = _make_thought(
                thought_id="t-qualify",
                created_cycle=0,
                updated_cycle=80,
                confirmation_count=5,
                confidence=0.9,
            )
            not_qualifying = _make_thought(
                thought_id="t-young",
                created_cycle=95,
                updated_cycle=95,
                confirmation_count=0,
                confidence=0.2,
            )
            await store.create_thought(qualifying)
            await store.create_thought(not_qualifying)

            # Activate both
            await store.update_thought("t-qualify", lifecycle_status="ACTIVE")
            await store.update_thought("t-young", lifecycle_status="ACTIVE")

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.4,
                gates=DreamingGates(
                    min_confirmations=2,
                    min_age_cycles=10,
                    max_promoted_per_run=20,
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=100)

            assert result.candidates_evaluated == 2
            assert "t-qualify" in result.promoted_ids
            assert "t-young" not in result.promoted_ids
            assert result.skipped_gate_count >= 1

            # Verify promotion side effect
            promoted = await store.get_thought("t-qualify")
            assert promoted is not None
            assert promoted.priority == Priority.P1
        finally:
            await db.close()

    async def test_max_promoted_per_run_cap(self) -> None:
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            # Create 5 qualifying thoughts
            for i in range(5):
                t = _make_thought(
                    thought_id=f"t-{i}",
                    created_cycle=0,
                    updated_cycle=80,
                    confirmation_count=5,
                    confidence=0.9,
                )
                await store.create_thought(t)
                await store.update_thought(f"t-{i}", lifecycle_status="ACTIVE")

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.3,
                gates=DreamingGates(
                    min_confirmations=2,
                    min_age_cycles=5,
                    max_promoted_per_run=2,  # Cap at 2
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=100)

            assert result.promoted_count <= 2
        finally:
            await db.close()

    async def test_no_candidates_returns_empty(self) -> None:
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            cfg = DreamingConfig(enabled=True)
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=100)

            assert result.candidates_evaluated == 0
            assert result.promoted_count == 0
            assert result.promoted_ids == []
        finally:
            await db.close()

    async def test_scores_recorded_in_result(self) -> None:
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            t = _make_thought(
                thought_id="t-scored",
                created_cycle=0,
                updated_cycle=50,
                confirmation_count=3,
                confidence=0.7,
            )
            await store.create_thought(t)
            await store.update_thought("t-scored", lifecycle_status="ACTIVE")

            cfg = DreamingConfig(enabled=True, promote_threshold=0.99)
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=100)

            assert "t-scored" in result.scores
            assert 0.0 <= result.scores["t-scored"] <= 1.0
        finally:
            await db.close()


# ------------------------------------------------------------------
# Gate bypass tests
# ------------------------------------------------------------------


class TestGateBypass:
    """Verify that ``allow_zero_confirmation`` lets fresh thoughts promote."""

    async def test_zero_confirmation_allows_promotion(self) -> None:
        """Thoughts with 0 confirmations pass gates when flag is True."""
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            t = _make_thought(
                thought_id="t-zero-conf",
                created_cycle=0,
                updated_cycle=50,
                confirmation_count=0,
                confidence=0.9,
            )
            await store.create_thought(t)
            await store.update_thought("t-zero-conf", lifecycle_status="ACTIVE")

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.3,
                gates=DreamingGates(
                    min_confirmations=2,
                    min_age_cycles=1,
                    max_promoted_per_run=20,
                    allow_zero_confirmation=True,
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=10)

            assert "t-zero-conf" in result.promoted_ids
            assert result.skipped_gate_count == 0
        finally:
            await db.close()

    async def test_min_age_one_cycle_default(self) -> None:
        """Default min_age_cycles=1 means cycle-0 thoughts are not eligible at cycle 0."""
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            t = _make_thought(
                thought_id="t-too-young",
                created_cycle=0,
                updated_cycle=0,
                confirmation_count=5,
                confidence=0.9,
            )
            await store.create_thought(t)
            await store.update_thought("t-too-young", lifecycle_status="ACTIVE")

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.1,
                gates=DreamingGates(),  # default min_age_cycles=1
            )
            ext = DreamingExtension(config=cfg)
            # current_cycle=0, created_cycle=0 → age=0 < 1 → gate fails
            result = await ext.run_consolidation(store, current_cycle=0)
            assert "t-too-young" not in result.promoted_ids
            assert result.skipped_gate_count == 1

            # current_cycle=1 → age=1 >= 1 → gate passes
            result2 = await ext.run_consolidation(store, current_cycle=1)
            assert "t-too-young" in result2.promoted_ids
        finally:
            await db.close()

    async def test_legacy_config_still_works(self) -> None:
        """Config without allow_zero_confirmation defaults to True (backward compat)."""
        gates = DreamingGates(min_confirmations=2, min_age_cycles=5)
        assert gates.allow_zero_confirmation is True

        gates_strict = DreamingGates(
            min_confirmations=2,
            min_age_cycles=5,
            allow_zero_confirmation=False,
        )
        assert gates_strict.allow_zero_confirmation is False


class TestSingleWriteScenario:
    """Simulate batch-ingest with no confirmation, verify promotion works."""

    async def test_batch_ingest_100_chunks_promotes(self) -> None:
        """100 freshly ingested thoughts with zero confirmation → promoted > 0."""
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            for i in range(100):
                t = _make_thought(
                    thought_id=f"batch-{i}",
                    created_cycle=0,
                    updated_cycle=0,
                    confirmation_count=0,
                    confidence=0.7,
                )
                await store.create_thought(t)
                await store.update_thought(f"batch-{i}", lifecycle_status="ACTIVE")

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.3,
                gates=DreamingGates(
                    allow_zero_confirmation=True,
                    min_age_cycles=1,
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=1)

            assert result.candidates_evaluated == 100
            assert result.promoted_count > 0
        finally:
            await db.close()

    async def test_non_zero_promoted_count_after_single_dream_cycle(self) -> None:
        """Single dream cycle on fresh batch produces non-zero promoted count."""
        import aiosqlite

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            for i in range(10):
                t = _make_thought(
                    thought_id=f"fresh-{i}",
                    created_cycle=0,
                    updated_cycle=0,
                    confirmation_count=0,
                    confidence=0.8,
                )
                await store.create_thought(t)
                await store.update_thought(f"fresh-{i}", lifecycle_status="ACTIVE")

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.2,
                gates=DreamingGates(
                    allow_zero_confirmation=True,
                    min_age_cycles=1,
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=1)

            assert result.promoted_count > 0
            # Verify side effect: promoted thoughts have P1 priority
            for tid in result.promoted_ids:
                promoted = await store.get_thought(tid)
                assert promoted is not None
                assert promoted.priority == Priority.P1
        finally:
            await db.close()

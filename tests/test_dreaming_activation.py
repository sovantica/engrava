"""Dreaming activation correctness — reachable scoring + live access substrate.

Covers the activation fixes: active-signal weight redistribution (promotion is
arithmetically reachable under defaults again), the batched access substrate
(feeds the ``frequency`` signal without per-read writes), config wiring
(``from_config`` builds + runs dreaming, partial ``signals`` merges), and the
default-off byte-identity guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import DreamingConfig, DreamingGates, _parse_dreaming
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# Cycle at which recency == staleness == 1.0: a thought created at cycle 0 and
# last updated at cycle 100 has age 0 (recency 1.0) and span 100 (staleness 1.0)
# when scored at cycle 100.
_CYCLE = 100


def _obs(
    thought_id: str,
    *,
    confirmation_count: int = 0,
    confidence: float | None = None,
    access_count: int = 0,
    created_cycle: int = 0,
    updated_cycle: int = _CYCLE,
) -> ThoughtRecord:
    """A recent + mature ACTIVE OBSERVATION (recency = staleness = 1.0 at _CYCLE)."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"Essence {thought_id}",
        content=f"Content of {thought_id} about apples and oranges and pears.",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confirmation_count=confirmation_count,
        confidence=confidence,
        access_count=access_count,
    )


def _activation_cfg(
    *,
    promote_threshold: float = 0.7,
    access_tracking_enabled: bool = True,
    max_p1_fraction: float = 0.5,
) -> DreamingConfig:
    """Dreaming config with the real 0.7 threshold and open promotion gates.

    The promotion *gates* (age / confirmation) are opened so the test isolates
    the *score* reachability; the score threshold stays at its shipped 0.7.
    """
    return DreamingConfig(
        enabled=True,
        promote_threshold=promote_threshold,
        max_p1_fraction=max_p1_fraction,
        access_tracking_enabled=access_tracking_enabled,
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            enable_reflections=False,  # isolate promotion from clustering
        ),
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    db = await aiosqlite.connect(str(tmp_path / "activation.db"))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    yield s
    await db.close()


# ---------------------------------------------------------------------------
# _compute_active_weights — redistribution unit (the reachability mechanism)
# ---------------------------------------------------------------------------


class TestActiveWeightRedistribution:
    """The score is a weighted average over the signals active for the run."""

    def test_all_default_signals_flat_redistribute_to_recency_staleness(self) -> None:
        ext = DreamingExtension(config=_activation_cfg(access_tracking_enabled=False))
        candidates = [_obs("a"), _obs("b")]  # unconfirmed, no confidence, no access
        weights, flat = ext._compute_active_weights(candidates, current_cycle=_CYCLE)

        # confirmation / confidence / frequency carry no data → flat.
        assert set(flat) == {"confirmation", "confidence", "frequency"}
        # Their weight redistributes onto recency (.25) + staleness (.20).
        assert weights["recency"] == pytest.approx(0.25 / 0.45)
        assert weights["staleness"] == pytest.approx(0.20 / 0.45)
        assert weights["confirmation"] == 0.0
        assert weights["frequency"] == 0.0
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_degenerate_no_active_signal_yields_zero_weights(self) -> None:
        ext = DreamingExtension(config=_activation_cfg(access_tracking_enabled=False))
        # current_cycle None → recency + staleness inactive; nothing else has data.
        weights, flat = ext._compute_active_weights([_obs("a")], current_cycle=None)  # type: ignore[arg-type]
        assert all(w == 0.0 for w in weights.values())
        assert "recency" in flat and "staleness" in flat

    def test_confirmation_becomes_active_when_a_candidate_is_confirmed(self) -> None:
        """Candidate-set dependence (intended): a confirmed peer activates the signal."""
        ext = DreamingExtension(config=_activation_cfg(access_tracking_enabled=False))
        weights, flat = ext._compute_active_weights(
            [_obs("a"), _obs("b", confirmation_count=5)],
            current_cycle=_CYCLE,
        )
        assert "confirmation" not in flat  # now active (one candidate has data)
        assert weights["confirmation"] == pytest.approx(0.20 / 0.65)


# ---------------------------------------------------------------------------
# Reachability — promotion is possible under defaults again (the core fix)
# ---------------------------------------------------------------------------


class TestPromotionReachable:
    async def test_recent_mature_observations_promote_under_default_threshold(
        self, store: SqliteEngravaCore
    ) -> None:
        """A recent + mature OBSERVATION promotes at the shipped 0.7 threshold.

        On the pre-fix score (dead frequency + confirmation dragging the max to
        0.525) this promotes zero; after active-signal redistribution recency +
        staleness alone reach 1.0 > 0.7.
        """
        ext = DreamingExtension(config=_activation_cfg())
        for i in range(4):
            await store.create_thought(_obs(f"obs-{i}"))

        result = await ext.run_consolidation(store, current_cycle=_CYCLE)

        assert result.promoted_count >= 1
        # Redistribution recorded on the result; the dead signals are flat.
        assert result.active_signal_weights["recency"] > 0.0
        assert "frequency" in result.flat_signals

    async def test_candidate_set_dependence_end_to_end(
        self, store: SqliteEngravaCore
    ) -> None:
        """A confirmed peer can demote an unconfirmed thought below the gate.

        With confirmation flat, an unconfirmed recent thought scores ~1.0 and
        promotes. Add a confirmed peer → confirmation activates → the same
        unconfirmed thought scores ~0.692 < 0.70 and no longer promotes. This
        pins the intended pool-relative behaviour.
        """
        ext = DreamingExtension(config=_activation_cfg())
        await store.create_thought(_obs("solo"))
        alone = await ext.run_consolidation(store, current_cycle=_CYCLE)
        assert "solo" in alone.promoted_ids

        # Fresh store with the same unconfirmed thought + a confirmed peer.
        db2 = await aiosqlite.connect(":memory:")
        db2.row_factory = aiosqlite.Row
        store2 = SqliteEngravaCore(db=db2)
        await store2.ensure_schema()
        await store2.create_thought(_obs("unconf"))
        await store2.create_thought(_obs("conf", confirmation_count=5))
        mixed = await ext.run_consolidation(store2, current_cycle=_CYCLE)
        assert "unconf" not in mixed.promoted_ids  # confirmation now discriminates
        assert "confirmation" not in mixed.flat_signals
        await db2.close()

    async def test_population_p1_cap_bounds_repeated_cycles(
        self, store: SqliteEngravaCore
    ) -> None:
        """Repeated cycles do not push the P1 population past max_p1_fraction."""
        ext = DreamingExtension(config=_activation_cfg(max_p1_fraction=0.25))
        for i in range(8):
            await store.create_thought(_obs(f"obs-{i}"))
        await ext.run_consolidation(store, current_cycle=_CYCLE)
        await ext.run_consolidation(store, current_cycle=_CYCLE + 1)
        await ext.run_consolidation(store, current_cycle=_CYCLE + 2)
        p1 = await store.count_thoughts(priority="P1")
        total = await store.count_thoughts()
        assert p1 <= max(1, int(total * 0.25))


# ---------------------------------------------------------------------------
# Access substrate — batched, no per-read write, feeds frequency
# ---------------------------------------------------------------------------


class TestAccessSubstrate:
    async def test_retrieval_buffers_then_flush_increments_no_hot_write(
        self, store: SqliteEngravaCore
    ) -> None:
        store._access_tracking_enabled = True  # noqa: SLF001 -- simulate from_config wiring
        await store.create_thought(_obs("t"))

        await store.get_thought("t")
        await store.get_thought("t")
        # No DB write on the read path yet — access_count still 0.
        row = await store.get_thought("t")  # a third buffered access
        assert row is not None and row.access_count == 0
        assert len(store._access_buffer) == 1  # noqa: SLF001

        updated = await store.flush_access_buffer()
        assert updated == 1
        after = await store.get_thought("t")  # buffers again, but DB already flushed
        assert after is not None and after.access_count == 3  # noqa: PLR2004

    async def test_default_off_no_access_tracking_byte_identical(
        self, store: SqliteEngravaCore
    ) -> None:
        """Default store (tracking off) never buffers or writes access counts."""
        assert store._access_tracking_enabled is False  # noqa: SLF001
        await store.create_thought(_obs("t"))
        await store.get_thought("t")
        assert len(store._access_buffer) == 0  # noqa: SLF001
        assert await store.flush_access_buffer() == 0
        row = await store.get_thought("t")
        assert row is not None and row.access_count == 0


# ---------------------------------------------------------------------------
# Config wiring — from_config activation + signals merge + new fields
# ---------------------------------------------------------------------------


class TestConfigActivation:
    def test_partial_signals_merge_keeps_other_defaults(self) -> None:
        """Overriding one signal weight must not zero the other four."""
        cfg = _parse_dreaming({"enabled": True, "signals": {"recency": 0.5}})
        assert cfg is not None
        assert cfg.signals["recency"] == 0.5  # noqa: PLR2004
        # The other four keep their defaults, not dropped.
        assert set(cfg.signals) == {
            "recency",
            "staleness",
            "confirmation",
            "confidence",
            "frequency",
        }
        assert cfg.signals["staleness"] == 0.20  # noqa: PLR2004

    def test_new_dreaming_fields_parse_from_yaml(self) -> None:
        cfg = _parse_dreaming(
            {
                "enabled": True,
                "access_tracking_enabled": False,
                "self_filter_mode": "self_only",
                "min_source_confidence": "high",
                "boilerplate_threshold": 0.5,
                "eligible_content_types": ["note", "fact"],
            }
        )
        assert cfg is not None
        assert cfg.access_tracking_enabled is False
        assert cfg.self_filter_mode == "self_only"
        assert cfg.min_source_confidence == "high"
        assert cfg.boilerplate_threshold == 0.5  # noqa: PLR2004
        assert cfg.eligible_content_types == frozenset({"note", "fact"})

    def test_access_tracking_rejects_non_bool(self) -> None:
        from engrava.config import ConfigError

        with pytest.raises(ConfigError):
            _parse_dreaming({"enabled": True, "access_tracking_enabled": "yes"})

    async def test_from_config_wires_and_runs_dreaming(self, tmp_path: Path) -> None:
        """A YAML-only user activates dreaming end-to-end via store.consolidate."""
        db_path = tmp_path / "yaml.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            "database:\n"
            f"  path: {db_path}\n"
            "extensions:\n"
            "  dreaming:\n"
            "    enabled: true\n"
            "    enable_reflections: false\n",
            encoding="utf-8",
        )
        store = await SqliteEngravaCore.from_config(cfg_file)
        try:
            assert store._dreaming_extension is not None  # noqa: SLF001 -- wired
            assert store._access_tracking_enabled is True  # noqa: SLF001
            for i in range(4):
                await store.create_thought(_obs(f"obs-{i}"))
            result = await store.consolidate(current_cycle=_CYCLE)
            assert result.promoted_count >= 1
        finally:
            await store.close()

    async def test_consolidate_without_dreaming_raises(self, store: SqliteEngravaCore) -> None:
        """A manually-built store has no wired extension → consolidate() raises."""
        with pytest.raises(RuntimeError, match="dreaming"):
            await store.consolidate(current_cycle=1)

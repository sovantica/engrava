"""Tests for dreaming early-stop clustering guard.

Covers:
- First run always executes clustering (no prior count stored)
- Second run with no new candidates skips clustering
- Second run with new_count < threshold skips clustering
- Run after threshold is met executes clustering and resets the counter
- threshold=0 disables the guard (always cluster)
- Configurable threshold is respected
- _last_clustering_candidate_count tracks state correctly
- count_thoughts on SqliteEngravaCore returns correct totals
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import DreamingConfig, DreamingGates, EdgeCreationConfig
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(thought_id: str, *, created_cycle: int = 0) -> ThoughtRecord:
    """Minimal OBSERVATION thought for lifecycle tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"obs-{thought_id}",
        content="content",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=created_cycle,
        updated_cycle=created_cycle,
        source="test",
    )


def _make_reflection(thought_id: str) -> ThoughtRecord:
    """Minimal REFLECTION thought for lifecycle tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.REFLECTION,
        essence=f"ref-{thought_id}",
        content='{"member_ids":[],"keywords":[],"cluster_hash":"h0"}',
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="dreaming:h0",
        source_type=KnowledgeSource.DREAMING,
    )


def _early_stop_cfg(*, min_new: int = 50) -> DreamingConfig:
    """DreamingConfig with agglomerative clustering and early-stop enabled."""
    return DreamingConfig(
        enabled=True,
        promote_threshold=0.0,
        candidates_limit=1000,
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            min_cluster_size=2,
            cluster_algorithm="agglomerative",
            enable_reflections=True,
            cluster_allowed_types=("OBSERVATION",),
            clustering_min_new_candidates=min_new,
        ),
        edges=EdgeCreationConfig(enabled=False),
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh in-memory store for each test."""
    db = await aiosqlite.connect(str(tmp_path / "test.db"))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    yield s
    await db.close()


# ---------------------------------------------------------------------------
# count_thoughts — unit tests
# ---------------------------------------------------------------------------


class TestCountThoughts:
    async def test_empty_store_returns_zero(self, store: SqliteEngravaCore) -> None:
        assert await store.count_thoughts() == 0

    async def test_counts_all_thoughts(self, store: SqliteEngravaCore) -> None:
        for i in range(5):
            await store.create_thought(_make_obs(f"t-{i}"))
        assert await store.count_thoughts() == 5

    async def test_filters_by_lifecycle_status(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_obs("active-1"))
        t = await store.create_thought(_make_obs("to-archive"))
        await store.update_thought(t.thought_id, lifecycle_status="ARCHIVED")
        assert await store.count_thoughts(lifecycle_status="ACTIVE") == 1
        assert await store.count_thoughts(lifecycle_status="ARCHIVED") == 1

    async def test_filters_by_thought_type(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_obs("obs-1"))
        await store.create_thought(_make_reflection("ref-1"))
        assert await store.count_thoughts(thought_type="OBSERVATION") == 1
        assert await store.count_thoughts(thought_type="REFLECTION") == 1

    async def test_combined_filters(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_make_obs("obs-active"))
        ref = await store.create_thought(_make_reflection("ref-active"))
        await store.update_thought(ref.thought_id, lifecycle_status="ARCHIVED")
        count = await store.count_thoughts(lifecycle_status="ACTIVE", thought_type="OBSERVATION")
        assert count == 1


# ---------------------------------------------------------------------------
# Early-stop guard — unit / integration
# ---------------------------------------------------------------------------


class TestEarlyStopGuard:
    async def test_first_run_always_clusters(self, store: SqliteEngravaCore) -> None:
        """First consolidation run has no prior count — guard is bypassed."""
        for i in range(5):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=50))
        assert ext._last_clustering_candidate_count is None

        build_clusters_called: list[bool] = []
        original = ext._build_clusters

        async def spy(*args: object, **kwargs: object) -> list[frozenset[str]]:
            build_clusters_called.append(True)
            return await original(*args, **kwargs)

        ext._build_clusters = spy  # type: ignore[method-assign]
        await ext.run_consolidation(store, current_cycle=1)

        assert build_clusters_called, "_build_clusters must be called on first run"
        assert ext._last_clustering_candidate_count == 5

    async def test_second_run_no_new_candidates_skips_clustering(
        self, store: SqliteEngravaCore
    ) -> None:
        """When new_count == 0, clustering is skipped on subsequent runs."""
        for i in range(5):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=1))
        # Simulate prior run having seen all 5 candidates.
        ext._last_clustering_candidate_count = 5

        build_clusters_called: list[bool] = []
        original = ext._build_clusters

        async def spy(*args: object, **kwargs: object) -> list[frozenset[str]]:
            build_clusters_called.append(True)
            return await original(*args, **kwargs)

        ext._build_clusters = spy  # type: ignore[method-assign]
        result = await ext.run_consolidation(store, current_cycle=2)

        assert not build_clusters_called, "_build_clusters must NOT be called"
        assert result.reflections_created == 0

    async def test_new_count_below_threshold_skips_clustering(
        self, store: SqliteEngravaCore
    ) -> None:
        """new_count=30 < threshold=50 → clustering skipped."""
        for i in range(80):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=50))
        # Simulate previous run having seen 50 of the 80 thoughts.
        ext._last_clustering_candidate_count = 50

        build_clusters_called: list[bool] = []
        original = ext._build_clusters

        async def spy(*args: object, **kwargs: object) -> list[frozenset[str]]:
            build_clusters_called.append(True)
            return await original(*args, **kwargs)

        ext._build_clusters = spy  # type: ignore[method-assign]
        await ext.run_consolidation(store, current_cycle=3)

        assert not build_clusters_called

    async def test_new_count_meets_threshold_triggers_clustering(
        self, store: SqliteEngravaCore
    ) -> None:
        """new_count == threshold → clustering executes and counter updates."""
        for i in range(100):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=50))
        ext._last_clustering_candidate_count = 50  # delta = 50 == min_new → should run

        build_clusters_called: list[bool] = []
        original = ext._build_clusters

        async def spy(*args: object, **kwargs: object) -> list[frozenset[str]]:
            build_clusters_called.append(True)
            return await original(*args, **kwargs)

        ext._build_clusters = spy  # type: ignore[method-assign]
        await ext.run_consolidation(store, current_cycle=4)

        assert build_clusters_called
        assert ext._last_clustering_candidate_count == 100

    async def test_threshold_zero_disables_guard(self, store: SqliteEngravaCore) -> None:
        """clustering_min_new_candidates=0 disables early-stop entirely."""
        for i in range(5):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=0))
        ext._last_clustering_candidate_count = 5  # no new candidates

        build_clusters_called: list[bool] = []
        original = ext._build_clusters

        async def spy(*args: object, **kwargs: object) -> list[frozenset[str]]:
            build_clusters_called.append(True)
            return await original(*args, **kwargs)

        ext._build_clusters = spy  # type: ignore[method-assign]
        await ext.run_consolidation(store, current_cycle=5)

        assert build_clusters_called, "Guard must be disabled when min_new=0"

    async def test_counter_not_updated_when_guard_fires(self, store: SqliteEngravaCore) -> None:
        """When early-stop fires, _last_clustering_candidate_count stays unchanged."""
        for i in range(5):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=50))
        ext._last_clustering_candidate_count = 4  # delta = 1 < 50

        await ext.run_consolidation(store, current_cycle=6)

        # Counter must NOT be updated — we want the next run to accumulate.
        assert ext._last_clustering_candidate_count == 4

    async def test_counter_updates_after_clustering_runs(self, store: SqliteEngravaCore) -> None:
        """After clustering runs, counter is set to the current candidate count."""
        for i in range(60):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=50))
        assert ext._last_clustering_candidate_count is None

        await ext.run_consolidation(store, current_cycle=1)

        assert ext._last_clustering_candidate_count == 60

    async def test_reflections_counted_correctly_when_guard_fires(
        self, store: SqliteEngravaCore
    ) -> None:
        """ConsolidationResult.reflections_created == 0 when clustering is skipped."""
        for i in range(5):
            await store.create_thought(_make_obs(f"obs-{i}"))

        ext = DreamingExtension(config=_early_stop_cfg(min_new=50))
        ext._last_clustering_candidate_count = 5  # no new candidates

        result = await ext.run_consolidation(store, current_cycle=2)

        assert result.reflections_created == 0


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestEarlyStopConfig:
    def test_default_threshold_is_50(self) -> None:
        """Default clustering_min_new_candidates is 50."""
        from engrava.config import DreamingGates

        gates = DreamingGates()
        assert gates.clustering_min_new_candidates == 50

    def test_custom_threshold(self) -> None:
        """Operator can set a custom threshold."""
        from engrava.config import DreamingGates

        gates = DreamingGates(clustering_min_new_candidates=100)
        assert gates.clustering_min_new_candidates == 100

    def test_zero_disables_guard(self) -> None:
        """threshold=0 is a valid opt-out value."""
        from engrava.config import DreamingGates

        gates = DreamingGates(clustering_min_new_candidates=0)
        assert gates.clustering_min_new_candidates == 0

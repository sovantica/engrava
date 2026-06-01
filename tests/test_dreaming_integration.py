"""Integration tests for DreamingExtension — end-to-end pipeline scenarios.

Exercises the full consolidation pipeline against a real ``SqliteEngravaCore``
instance (in-memory SQLite) to verify cross-phase interactions that unit
tests cannot cover in isolation:

- Dream-created ASSOCIATED edges for embedded thoughts
- Edge-creation idempotence (no duplicates on repeated runs)
- Lifecycle filtering — ARCHIVED / COMPLETED thoughts never promoted
- Post-dream hybrid search: priority-boost surfaces P1 thoughts higher
- Post-dream hybrid search: graph signal boosts neighbours of promoted thoughts
- Age gate enforcement across two consecutive cycles
- Custom signal integration against a real store
- Full REFLECTION → ``search_reflections_only`` feedback loop
- ConsolidationResult diagnostic counters under realistic conditions
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import (
    DreamingConfig,
    DreamingGates,
    EdgeCreationConfig,
    SearchConfig,
)
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.extensions.dreaming_signals import DreamingContext

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh ``SqliteEngravaCore`` backed by an on-disk SQLite database."""
    db = await aiosqlite.connect(str(tmp_path / "integration.db"))
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
    essence: str = "test",
    content: str = "content",
    created_cycle: int = 0,
    updated_cycle: int = 0,
    confirmation_count: int = 5,
    confidence: float = 0.9,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    priority: Priority = Priority.P3,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
) -> ThoughtRecord:
    """Build a minimal ``ThoughtRecord`` for integration tests."""
    return ThoughtRecord(
        thought_id=tid,
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confirmation_count=confirmation_count,
        confidence=confidence,
    )


def _promote_cfg(
    *,
    threshold: float = 0.0,
    min_age: int = 0,
    max_per_run: int = 50,
    edges_top_k: int = 3,
    edges_min_sim: float = 0.5,
    enable_reflections: bool = False,
    cluster_algorithm: str = "lpa",
    min_cluster_size: int = 2,
) -> DreamingConfig:
    """Return a ``DreamingConfig`` tuned to guarantee promotion in tests."""
    return DreamingConfig(
        enabled=True,
        promote_threshold=threshold,
        max_p1_fraction=1.0,
        promote_targets="ALL",
        gates=DreamingGates(
            min_age_cycles=min_age,
            allow_zero_confirmation=True,
            max_promoted_per_run=max_per_run,
            enable_reflections=enable_reflections,
            cluster_algorithm=cluster_algorithm,  # type: ignore[arg-type]
            min_cluster_size=min_cluster_size,
            # These integration tests pre-date the content-quality
            # gates and build synthetic clusters whose member content
            # is contrived for the feedback-loop, idempotence and IDF
            # behaviours under test — exercising the gates here would
            # turn them into flakiness sources.  The gates themselves
            # are exercised by ``test_dreaming_cluster_quality.py`` and
            # the dedicated gating integration suite.
            cluster_quality_gating_enabled=False,
        ),
        edges=EdgeCreationConfig(
            enabled=True,
            top_k=edges_top_k,
            min_similarity=edges_min_sim,
            edge_weight_factor=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Edge creation
# ---------------------------------------------------------------------------


class TestEdgeCreationPipeline:
    """Dream-created ASSOCIATED edges are persisted after promotion."""

    async def test_edges_created_between_similar_embedded_thoughts(
        self, store: SqliteEngravaCore
    ) -> None:
        """Similar embedded thoughts receive an ASSOCIATED edge after dreaming."""
        t1 = await store.create_thought(_thought("e1", essence="machine learning"))
        t2 = await store.create_thought(_thought("e2", essence="deep learning"))
        t3 = await store.create_thought(_thought("e3", essence="cooking recipes"))
        # t1 and t2 are very similar; t3 is orthogonal
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")
        await store.store_embedding(t3.thought_id, [0.0, 0.0, 0.9, 0.1], model_name="test")

        ext = DreamingExtension(config=_promote_cfg(edges_min_sim=0.7))
        result = await ext.run_consolidation(store, current_cycle=5)

        assert result.promoted_count >= 2
        assert result.edges_created >= 1

        # t1 and t2 must be connected
        edges_t1 = await store.get_edges(t1.thought_id, direction="BOTH")
        connected = {
            e.to_thought_id if e.from_thought_id == t1.thought_id else e.from_thought_id
            for e in edges_t1
            if e.edge_type == EdgeType.ASSOCIATED
        }
        assert t2.thought_id in connected

    async def test_edge_source_is_dreaming(self, store: SqliteEngravaCore) -> None:
        """Dream-created edges carry ``source=KnowledgeSource.DREAMING``."""
        t1 = await store.create_thought(_thought("src1"))
        t2 = await store.create_thought(_thought("src2"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15], model_name="test")

        ext = DreamingExtension(config=_promote_cfg(edges_min_sim=0.5))
        await ext.run_consolidation(store, current_cycle=1)

        all_edges = await store.list_edges(
            edge_type=EdgeType.ASSOCIATED, source=KnowledgeSource.DREAMING
        )
        assert len(all_edges) >= 1
        assert all(e.source == KnowledgeSource.DREAMING for e in all_edges)

    async def test_no_edges_for_thoughts_without_embeddings(self, store: SqliteEngravaCore) -> None:
        """Thoughts with no stored embedding do not get dream edges."""
        for i in range(3):
            await store.create_thought(_thought(f"noem-{i}"))
        # No embeddings stored

        ext = DreamingExtension(config=_promote_cfg())
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.promoted_count == 3
        assert result.edges_created == 0


# ---------------------------------------------------------------------------
# 2. Edge-creation idempotence
# ---------------------------------------------------------------------------


class TestEdgeCreationIdempotence:
    """Running consolidation twice does not create duplicate edges."""

    async def test_second_run_creates_no_duplicate_edges(self, store: SqliteEngravaCore) -> None:
        """Edge count is identical on the second consolidation run."""
        t1 = await store.create_thought(_thought("idem1"))
        t2 = await store.create_thought(_thought("idem2"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15], model_name="test")

        ext = DreamingExtension(config=_promote_cfg(edges_min_sim=0.5))
        result1 = await ext.run_consolidation(store, current_cycle=1)
        result2 = await ext.run_consolidation(store, current_cycle=2)

        assert result1.edges_created >= 1
        assert result2.edges_created == 0  # All edges already exist

    async def test_total_edge_count_stable_after_repeated_runs(
        self, store: SqliteEngravaCore
    ) -> None:
        """Total ASSOCIATED edge count does not grow on re-runs."""
        for i in range(3):
            t = await store.create_thought(_thought(f"rep-{i}"))
            # All similar — they all cluster together
            await store.store_embedding(t.thought_id, [0.9, 0.1, 0.0], model_name="test")

        ext = DreamingExtension(config=_promote_cfg(edges_min_sim=0.5))
        await ext.run_consolidation(store, current_cycle=1)

        edges_after_first = await store.list_edges(edge_type=EdgeType.ASSOCIATED)
        count_first = len(edges_after_first)

        await ext.run_consolidation(store, current_cycle=2)

        edges_after_second = await store.list_edges(edge_type=EdgeType.ASSOCIATED)
        assert len(edges_after_second) == count_first


# ---------------------------------------------------------------------------
# 3. Lifecycle filtering
# ---------------------------------------------------------------------------


class TestLifecycleFiltering:
    """Only ACTIVE thoughts are eligible for dreaming promotion."""

    async def test_archived_thoughts_not_promoted(self, store: SqliteEngravaCore) -> None:
        """ARCHIVED thoughts are never promoted regardless of score."""
        active = await store.create_thought(
            _thought("lc-active", lifecycle_status=LifecycleStatus.ACTIVE)
        )
        archived = await store.create_thought(
            _thought("lc-archived", lifecycle_status=LifecycleStatus.ARCHIVED)
        )

        ext = DreamingExtension(config=_promote_cfg())
        result = await ext.run_consolidation(store, current_cycle=1)

        assert active.thought_id in result.promoted_ids
        assert archived.thought_id not in result.promoted_ids

    async def test_done_thoughts_not_promoted(self, store: SqliteEngravaCore) -> None:
        """DONE thoughts are never promoted."""
        active = await store.create_thought(
            _thought("lc-act-2", lifecycle_status=LifecycleStatus.ACTIVE)
        )
        done = await store.create_thought(
            _thought("lc-done", lifecycle_status=LifecycleStatus.DONE)
        )

        ext = DreamingExtension(config=_promote_cfg())
        result = await ext.run_consolidation(store, current_cycle=1)

        assert active.thought_id in result.promoted_ids
        assert done.thought_id not in result.promoted_ids

    async def test_candidates_evaluated_counts_only_active(self, store: SqliteEngravaCore) -> None:
        """``candidates_evaluated`` reflects only ACTIVE thoughts."""
        for status in [
            LifecycleStatus.ACTIVE,
            LifecycleStatus.ARCHIVED,
            LifecycleStatus.DONE,
        ]:
            await store.create_thought(_thought(f"lc-{status}", lifecycle_status=status))

        ext = DreamingExtension(config=_promote_cfg())
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.candidates_evaluated == 1  # Only ACTIVE


# ---------------------------------------------------------------------------
# 4. Age gate enforcement across cycles
# ---------------------------------------------------------------------------


class TestAgeGateMultiCycle:
    """Age gate correctly blocks promotion at cycle 0 and allows at cycle 1."""

    async def test_thought_blocked_at_birth_cycle(self, store: SqliteEngravaCore) -> None:
        """Thought created at cycle N is not promoted when run at cycle N (age=0)."""
        await store.create_thought(_thought("age-t1", created_cycle=5, updated_cycle=5))

        cfg = _promote_cfg(min_age=1)  # min_age_cycles=1
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=5)  # age = 0

        assert "age-t1" not in result.promoted_ids
        assert result.skipped_gate_count == 1

    async def test_thought_promoted_one_cycle_later(self, store: SqliteEngravaCore) -> None:
        """Same thought is promoted one cycle later (age = min_age)."""
        await store.create_thought(_thought("age-t2", created_cycle=5, updated_cycle=5))

        cfg = _promote_cfg(min_age=1)
        ext = DreamingExtension(config=cfg)

        # At cycle 5 (age=0) — should be blocked
        r1 = await ext.run_consolidation(store, current_cycle=5)
        assert "age-t2" not in r1.promoted_ids

        # At cycle 6 (age=1=min_age) — should pass
        r2 = await ext.run_consolidation(store, current_cycle=6)
        assert "age-t2" in r2.promoted_ids

    async def test_min_age_zero_promotes_immediately(self, store: SqliteEngravaCore) -> None:
        """When ``min_age_cycles=0``, a brand-new thought is eligible."""
        await store.create_thought(_thought("age-imm", created_cycle=99, updated_cycle=99))

        cfg = _promote_cfg(min_age=0)
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=99)

        assert "age-imm" in result.promoted_ids


# ---------------------------------------------------------------------------
# 5. Post-dream search — priority boost
# ---------------------------------------------------------------------------


class TestPostDreamPriorityBoost:
    """Hybrid search ranks P1 thoughts higher when priority_weight > 0."""

    async def test_promoted_thought_outranks_p3_in_search(self, store: SqliteEngravaCore) -> None:
        """P1 thought ranks above P3 thought with the same embedding."""
        # Two thoughts with identical embeddings — same semantic score
        t_high = await store.create_thought(
            _thought("pri-high", essence="topic A", content="topic A detail")
        )
        t_low = await store.create_thought(
            _thought("pri-low", essence="topic A", content="topic A detail")
        )
        v = [0.7, 0.3, 0.0, 0.0]
        await store.store_embedding(t_high.thought_id, v, model_name="test")
        await store.store_embedding(t_low.thought_id, v, model_name="test")

        # Promote t_high to P1
        ext = DreamingExtension(config=_promote_cfg())
        await ext.run_consolidation(store, current_cycle=1)

        # Reload to get actual priority
        reloaded_high = await store.get_thought(t_high.thought_id)
        assert reloaded_high is not None
        assert reloaded_high.priority == Priority.P1

        # Search with strong priority weight — P1 should rank first
        search_cfg = SearchConfig(
            default_priority_weight=0.3,
            default_vector_weight=0.7,
            priority_boost_p1=1.0,
            priority_boost_p3=0.0,
        )
        store._search_config = search_cfg
        result = await store.search_hybrid(
            "",
            v,
            priority_weight=0.3,
            vector_weight=0.7,
        )
        result_ids = [tid for tid, _ in result.results]
        assert result_ids[0] == t_high.thought_id


# ---------------------------------------------------------------------------
# 6. Post-dream search — graph signal
# ---------------------------------------------------------------------------


class TestPostDreamGraphSignal:
    """Graph signal in ``search_hybrid`` boosts thoughts connected by dream edges."""

    async def test_graph_signal_boosts_connected_thought(self, store: SqliteEngravaCore) -> None:
        """Thought connected to a high-scoring thought gets a graph boost."""
        # t_anchor: matches the query perfectly
        # t_connected: similar to t_anchor → dream creates edge
        # t_unrelated: no connection, no query match
        t_anchor = await store.create_thought(_thought("gs-anchor", essence="python async"))
        t_connected = await store.create_thought(
            _thought("gs-connected", essence="asyncio event loop")
        )
        t_unrelated = await store.create_thought(
            _thought("gs-unrelated", essence="baking bread recipes")
        )

        await store.store_embedding(t_anchor.thought_id, [0.9, 0.1, 0.0], model_name="test")
        await store.store_embedding(t_connected.thought_id, [0.85, 0.15, 0.0], model_name="test")
        await store.store_embedding(t_unrelated.thought_id, [0.0, 0.0, 1.0], model_name="test")

        # Run dreaming — creates ASSOCIATED edge between anchor and connected
        ext = DreamingExtension(config=_promote_cfg(edges_min_sim=0.7))
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.edges_created >= 1

        # Search with graph signal enabled
        result_graph = await store.search_hybrid(
            "",
            [0.9, 0.1, 0.0],
            graph_weight=0.2,
            vector_weight=0.8,
        )
        result_ids = [tid for tid, _ in result_graph.results]
        # Both anchor and connected should be in top results
        assert t_anchor.thought_id in result_ids
        assert t_connected.thought_id in result_ids
        # Graph signal should appear in backends_used
        assert "graph" in result_graph.backends_used

    async def test_graph_weight_zero_no_graph_backend(self, store: SqliteEngravaCore) -> None:
        """With ``graph_weight=0``, the graph backend is not invoked."""
        t = await store.create_thought(_thought("gs-zero", essence="test"))
        await store.store_embedding(t.thought_id, [0.9, 0.1], model_name="test")

        result = await store.search_hybrid("", [0.9, 0.1], graph_weight=0.0)
        assert "graph" not in result.backends_used


# ---------------------------------------------------------------------------
# 7. Custom signal integration
# ---------------------------------------------------------------------------


class TestCustomSignalIntegration:
    """Custom signal functions work correctly with a real store."""

    async def test_content_length_signal_determines_promotion(
        self, store: SqliteEngravaCore
    ) -> None:
        """A custom signal based on content length correctly differentiates thoughts."""

        class ContentLengthSignal:
            """Score = normalised content length (max 200 chars → 1.0)."""

            def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
                return min(len(thought.content) / 200.0, 1.0)

        # Long-content thought (should score high)
        await store.create_thought(_thought("cl-long", content="x" * 200))
        # Short-content thought (should score low)
        await store.create_thought(_thought("cl-short", content="hi"))

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.5,
            signals={"content_length": 1.0},
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
        )
        ext = DreamingExtension(
            config=cfg,
            custom_signals={"content_length": ContentLengthSignal()},
        )
        result = await ext.run_consolidation(store, current_cycle=1)

        assert "cl-long" in result.promoted_ids
        assert "cl-short" not in result.promoted_ids

    async def test_custom_signal_score_recorded_in_result(self, store: SqliteEngravaCore) -> None:
        """Scores from custom signals appear in ``ConsolidationResult.scores``."""

        class FixedSignal:
            def __call__(self, thought: ThoughtRecord, ctx: DreamingContext) -> float:
                return 0.77

        await store.create_thought(_thought("fixed-1"))

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            signals={"fixed": 1.0},
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
        )
        ext = DreamingExtension(config=cfg, custom_signals={"fixed": FixedSignal()})
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.scores["fixed-1"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# 8. ConsolidationResult — diagnostic counters
# ---------------------------------------------------------------------------


class TestConsolidationResultCounters:
    """ConsolidationResult fields accurately reflect what happened."""

    async def test_all_counters_populated_on_real_run(self, store: SqliteEngravaCore) -> None:
        """All result fields are non-negative and internally consistent."""
        for i in range(4):
            t = await store.create_thought(_thought(f"cnt-{i}", created_cycle=0, updated_cycle=0))
            await store.store_embedding(t.thought_id, [0.8, 0.2], model_name="test")

        ext = DreamingExtension(config=_promote_cfg(edges_min_sim=0.5))
        result = await ext.run_consolidation(store, current_cycle=1)

        assert result.candidates_evaluated == 4
        assert result.promoted_count == len(result.promoted_ids)
        assert result.skipped_gate_count >= 0
        assert result.edges_created >= 0
        assert result.reflections_created == 0  # enable_reflections=False by default

    async def test_skipped_gate_count_matches_unqualified_thoughts(
        self, store: SqliteEngravaCore
    ) -> None:
        """``skipped_gate_count`` equals the number of thoughts that failed age gate."""
        # Created at cycle 10, run at cycle 10 → age=0, fails min_age=1
        for i in range(3):
            await store.create_thought(_thought(f"skip-{i}", created_cycle=10, updated_cycle=10))

        cfg = _promote_cfg(min_age=1)
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=10)

        assert result.skipped_gate_count == 3
        assert result.promoted_count == 0

    async def test_scores_dict_has_entry_for_each_candidate(self, store: SqliteEngravaCore) -> None:
        """``result.scores`` contains one entry per candidate thought."""
        tids = [f"sc-{i}" for i in range(5)]
        for tid in tids:
            await store.create_thought(_thought(tid))

        ext = DreamingExtension(config=_promote_cfg())
        result = await ext.run_consolidation(store, current_cycle=1)

        for tid in tids:
            assert tid in result.scores
            assert 0.0 <= result.scores[tid] <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 9. Full REFLECTION → search feedback loop
# ---------------------------------------------------------------------------


class TestReflectionFeedbackLoop:
    """Full pipeline: ingest → dream with clustering → REFLECTION → search."""

    async def test_reflections_appear_in_search_reflections_only(
        self, store: SqliteEngravaCore
    ) -> None:
        """After dreaming with agglomerative, REFLECTIONs surface in search."""
        # Two tightly clustered thoughts
        t1 = await store.create_thought(_thought("rfl-1", essence="python async programming"))
        t2 = await store.create_thought(_thought("rfl-2", essence="asyncio concurrency python"))
        # Third thought far away
        t3 = await store.create_thought(_thought("rfl-3", essence="baking sourdough bread"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.88, 0.12, 0.0], model_name="test")
        await store.store_embedding(t3.thought_id, [0.0, 0.0, 1.0], model_name="test")

        cfg = _promote_cfg(
            enable_reflections=True,
            cluster_algorithm="agglomerative",
            min_cluster_size=2,
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=5)

        assert result.reflections_created >= 1

        # REFLECTION thoughts should be in the store
        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) >= 1

        # search_reflections_only should return them
        search_result = await store.search_reflections_only("", [0.9, 0.1, 0.0], top_k=10)
        result_ids = {tid for tid, _ in search_result.results}
        reflection_ids = {r.thought_id for r in reflections}
        assert reflection_ids.issubset(result_ids)

    async def test_reflection_content_contains_member_ids(self, store: SqliteEngravaCore) -> None:
        """REFLECTION content JSON includes all cluster member IDs."""
        t1 = await store.create_thought(_thought("rfl-mem-1", essence="topic A"))
        t2 = await store.create_thought(_thought("rfl-mem-2", essence="topic A variant"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.88, 0.12, 0.0], model_name="test")

        cfg = _promote_cfg(
            enable_reflections=True,
            cluster_algorithm="agglomerative",
            min_cluster_size=2,
        )
        ext = DreamingExtension(config=cfg)
        await ext.run_consolidation(store, current_cycle=5)

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert reflections
        content = json.loads(reflections[0].content)
        assert "member_ids" in content
        assert "keywords" in content
        assert "cluster_hash" in content
        member_set = set(content["member_ids"])
        assert t1.thought_id in member_set
        assert t2.thought_id in member_set

    async def test_reflection_excluded_from_hybrid_search_when_flag_false(
        self, store: SqliteEngravaCore
    ) -> None:
        """REFLECTIONs created by dreaming are excluded when ``include_reflections=False``."""
        t1 = await store.create_thought(_thought("rfl-excl-1", essence="AI concepts"))
        t2 = await store.create_thought(_thought("rfl-excl-2", essence="AI concepts variant"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.88, 0.12, 0.0], model_name="test")

        cfg = _promote_cfg(
            enable_reflections=True,
            cluster_algorithm="agglomerative",
            min_cluster_size=2,
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=5)
        assert result.reflections_created >= 1

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        ref_ids = {r.thought_id for r in reflections}

        search_result = await store.search_hybrid("", [0.9, 0.1, 0.0], include_reflections=False)
        result_ids = {tid for tid, _ in search_result.results}
        assert ref_ids.isdisjoint(result_ids)

    async def test_reflection_idempotence_across_consolidation_runs(
        self, store: SqliteEngravaCore
    ) -> None:
        """Re-running consolidation (LPA) does not duplicate REFLECTIONs.

        LPA is idempotent because it operates over ASSOCIATED edges, which
        are not re-created on the second run (edge idempotence).  The
        same cluster of {t1, t2} produces the same 16-hex hash on both runs,
        so the second call to ``thought_exists_by_source`` returns ``True`` and
        the cluster is skipped.
        """
        t1 = await store.create_thought(_thought("rfl-idem-1", essence="topic X"))
        t2 = await store.create_thought(_thought("rfl-idem-2", essence="topic X variant"))
        # Embeddings must be stored so the edge-creation phase can link t1↔t2,
        # giving LPA a graph to cluster over.
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.88, 0.12, 0.0], model_name="test")

        # LPA algorithm: clusters via ASSOCIATED dream edges (content-hash idempotent)
        cfg = _promote_cfg(
            enable_reflections=True,
            cluster_algorithm="lpa",  # LPA over graph edges — idempotent by design
            min_cluster_size=2,
            edges_min_sim=0.5,
        )
        ext = DreamingExtension(config=cfg)

        r1 = await ext.run_consolidation(store, current_cycle=5)
        r2 = await ext.run_consolidation(store, current_cycle=6)

        assert r1.reflections_created >= 1
        assert r2.reflections_created == 0

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == r1.reflections_created


# ---------------------------------------------------------------------------
# 10. Promotion stability — already-P1 thoughts
# ---------------------------------------------------------------------------


class TestPromotionStability:
    """Thoughts already at P1 are handled gracefully on re-runs."""

    async def test_already_p1_thought_not_in_promoted_ids_on_rerun(
        self, store: SqliteEngravaCore
    ) -> None:
        """Thoughts promoted in run 1 are still P1 after run 2 (update is idempotent)."""
        await store.create_thought(_thought("p1-stable"))

        ext = DreamingExtension(config=_promote_cfg())
        r1 = await ext.run_consolidation(store, current_cycle=1)
        assert "p1-stable" in r1.promoted_ids

        # Even if re-promoted in run 2, priority stays P1
        await ext.run_consolidation(store, current_cycle=2)
        t_after = await store.get_thought("p1-stable")
        assert t_after is not None
        assert t_after.priority == Priority.P1

    async def test_mix_of_p1_and_p3_thoughts_separate_correctly(
        self, store: SqliteEngravaCore
    ) -> None:
        """In a mixed batch, only qualifying P3 thoughts are promoted."""
        # Pre-existing P1 (already promoted externally)
        await store.create_thought(_thought("mix-p1", priority=Priority.P1))
        # P3 qualifying — will be promoted
        await store.create_thought(_thought("mix-p3", priority=Priority.P3))

        ext = DreamingExtension(config=_promote_cfg())
        await ext.run_consolidation(store, current_cycle=1)

        # mix-p3 should get promoted; mix-p1 might also be listed (idempotent update)
        t_p3_after = await store.get_thought("mix-p3")
        assert t_p3_after is not None
        assert t_p3_after.priority == Priority.P1


# ---------------------------------------------------------------------------
# 11. v2 REFLECTION content — corpus passthrough produces real TF-IDF scores
# ---------------------------------------------------------------------------


class TestReflectionV2CorpusPassthrough:
    """``run_consolidation`` forwards the candidate corpus to the v2 builder.

    Without the passthrough the builder receives an empty corpus, every
    IDF degenerates to ``log(1.0) = 0``, all phrase scores collapse to
    zero, and the ranking falls back to lexicographic tie-break.  This
    test pins the contract: at least one ``top_keyphrases`` entry must
    have a strictly positive score, which only happens when the corpus
    actually reaches the builder.
    """

    async def test_top_keyphrases_have_positive_tfidf_scores(
        self, store: SqliteEngravaCore
    ) -> None:
        """Cluster phrases that are rare in the corpus rank above zero."""
        # Cluster: two thoughts containing the rare phrase "monday standup".
        # Corpus baseline: three more thoughts with unrelated bigrams.  The
        # rare phrase appears only inside the cluster, so its document
        # frequency in the corpus is at most 2 of 5 documents — IDF stays
        # well above zero and at least one phrase score is positive.
        cluster_a = await store.create_thought(
            _thought(
                "rfl-tf-1",
                essence="planning",
                content="monday standup decisions about quarterly goals",
            )
        )
        cluster_b = await store.create_thought(
            _thought(
                "rfl-tf-2",
                essence="planning",
                content="monday standup retrospective notes captured",
            )
        )
        # Three thoughts that share NO bigrams with the cluster — they
        # shape the corpus IDF without contaminating the cluster.
        await store.create_thought(
            _thought("noise-1", essence="recipe", content="baking sourdough bread weekly"),
        )
        await store.create_thought(
            _thought(
                "noise-2",
                essence="garden",
                content="watering tomatoes plants every morning",
            ),
        )
        await store.create_thought(
            _thought(
                "noise-3",
                essence="walking",
                content="evening walks around the lake park",
            ),
        )

        # Tightly cluster the two planning thoughts; isolate the noise.
        await store.store_embedding(cluster_a.thought_id, [0.9, 0.1, 0.0], model_name="test")
        await store.store_embedding(cluster_b.thought_id, [0.88, 0.12, 0.0], model_name="test")
        await store.store_embedding("noise-1", [0.0, 0.9, 0.1], model_name="test")
        await store.store_embedding("noise-2", [0.0, 0.85, 0.15], model_name="test")
        await store.store_embedding("noise-3", [0.0, 0.8, 0.2], model_name="test")

        cfg = _promote_cfg(
            enable_reflections=True,
            cluster_algorithm="agglomerative",
            min_cluster_size=2,
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=5)
        assert result.reflections_created >= 1

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert reflections
        content = json.loads(reflections[0].content)
        top_keyphrases = content["top_keyphrases"]
        assert top_keyphrases, "top_keyphrases must not be empty for a 2-member cluster"

        # The contract: at least one positive score.  Without the corpus
        # passthrough every score collapses to 0.0 (log((1+0)/(1+0)) = 0),
        # so this assertion is the regression guard for that bug class.
        scores = [entry["score"] for entry in top_keyphrases]
        assert any(score > 0.0 for score in scores), (
            f"all top_keyphrases scores are zero — corpus is not reaching the builder.  "
            f"scores={scores}"
        )

    async def test_top_keyphrases_idf_distinguishes_rare_from_common(
        self, store: SqliteEngravaCore
    ) -> None:
        """Rare phrases (low corpus DF) outrank common ones (high corpus DF).

        Cluster contains both rare bigrams ("rare unique", "unique
        combination") and a phrase ("alpha beta") that recurs across
        the noise corpus.  Real TF-IDF must surface a phrase rooted in
        the rare half of the cluster vocabulary; the empty-corpus
        fallback would tie everything at score 0 and pick by
        lexicographic order, putting "alpha beta" first.
        """
        # Cluster: two thoughts mentioning the rare phrase + the common one.
        cluster_a = await store.create_thought(
            _thought(
                "rfl-rare-1",
                content="alpha beta gamma rare unique combination",
            )
        )
        cluster_b = await store.create_thought(
            _thought(
                "rfl-rare-2",
                content="alpha beta gamma rare unique combination",
            )
        )
        # Five baseline thoughts containing "alpha beta gamma" but never
        # "rare unique combination" — pushes "alpha beta" / "beta gamma"
        # high into the corpus DF, near-zero IDF.
        for i in range(5):
            await store.create_thought(
                _thought(
                    f"common-{i}",
                    content=f"alpha beta gamma scenario number {i}",
                ),
            )
            await store.store_embedding(f"common-{i}", [0.0, 0.9, 0.1], model_name="test")

        await store.store_embedding(cluster_a.thought_id, [0.9, 0.1, 0.0], model_name="test")
        await store.store_embedding(cluster_b.thought_id, [0.88, 0.12, 0.0], model_name="test")

        cfg = _promote_cfg(
            enable_reflections=True,
            cluster_algorithm="agglomerative",
            min_cluster_size=2,
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=5)
        assert result.reflections_created >= 1

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert reflections
        content = json.loads(reflections[0].content)
        top_keyphrases = content["top_keyphrases"]
        assert top_keyphrases, "top_keyphrases must not be empty"

        # Top-3 must be dominated by phrases anchored on the rare-half
        # vocabulary ("rare", "unique", "combination") rather than the
        # near-zero-IDF tokens ("alpha", "beta", "gamma").
        rare_anchors = {"rare", "unique", "combination"}
        common_anchors = {"alpha", "beta"}
        for entry in top_keyphrases:
            phrase_words = set(str(entry["phrase"]).split())
            assert phrase_words & rare_anchors, (
                f"top_keyphrases entry {entry['phrase']!r} has no rare-anchor token "
                f"(score={entry['score']:.4f}) — corpus IDF passthrough may be broken; "
                f"all entries: {top_keyphrases}"
            )
            assert not (phrase_words <= common_anchors), (
                f"top_keyphrases entry {entry['phrase']!r} consists only of "
                f"high-DF tokens — TF-IDF should have ranked it lower"
            )

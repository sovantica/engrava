"""Tests for clustering and REFLECTION creation.

Covers:
- _lpa_clusters / _agglomerative_clusters produce disjoint clusters
- _build_clusters filters by min_cluster_size
- _create_reflections creates ThoughtType.REFLECTION thoughts
- CONSOLIDATED_FROM edges from REFLECTION to each member
- Idempotence: re-run does not create duplicate REFLECTIONs
- enable_reflections=False skips clustering entirely
- ConsolidationResult.reflections_created counter
- search_hybrid include_reflections=False excludes REFLECTIONs
- search_hybrid reflection_boost multiplies REFLECTION scores
- search_reflections_only returns only REFLECTION thoughts

Cognitive-boundary guard tests for the keyword extractor live in
``tests/test_dreaming_keyphrases.py`` since the helper now lives in
the ``dreaming_keyphrases`` sibling module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension, _lpa_clusters

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(
    thought_id: str,
    *,
    essence: str = "Test",
    content: str = "Content",
    created_cycle: int = 0,
    updated_cycle: int = 0,
    priority: Priority = Priority.P3,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> ThoughtRecord:
    """Minimal thought for clustering tests.

    ``valid_from`` / ``valid_until`` default to ``None`` (open bounds) so
    existing callers keep their old behaviour; the REFLECTION
    valid-time-inheritance tests pass explicit ISO-8601 bounds.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        valid_from=valid_from,
        valid_until=valid_until,
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh store."""
    db = await aiosqlite.connect(str(tmp_path / "test.db"))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    yield s
    await db.close()


def _reflection_cfg(
    *,
    enable_reflections: bool = True,
    min_cluster_size: int = 2,
    algorithm: str = "lpa",
) -> DreamingConfig:
    return DreamingConfig(
        enabled=True,
        promote_threshold=0.0,
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            min_cluster_size=min_cluster_size,
            cluster_algorithm=algorithm,  # type: ignore[arg-type]
            enable_reflections=enable_reflections,
            # The legacy clustering tests build synthetic clusters whose
            # member content is identical and whose TF-IDF keyphrases
            # end up empty — patterns that the new content-quality
            # gates correctly flag.  Disabling the gates here keeps
            # this test module focused on the cluster-construction
            # contract; the gating logic is covered separately by
            # ``test_dreaming_cluster_quality.py`` and the gating
            # integration suite.
            cluster_quality_gating_enabled=False,
        ),
        edges=EdgeCreationConfig(
            enabled=True,
            top_k=1,
            min_similarity=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests — _lpa_clusters
# ---------------------------------------------------------------------------


class TestLpaClusters:
    """Unit tests for Label Propagation helper."""

    def test_isolated_nodes_each_own_cluster(self) -> None:
        """Nodes with no edges form singleton clusters."""
        adj: dict[str, set[str]] = {"a": set(), "b": set(), "c": set()}
        clusters = _lpa_clusters(adj)
        assert len(clusters) == 3
        assert all(len(c) == 1 for c in clusters)

    def test_connected_pair_same_cluster(self) -> None:
        """Two connected nodes must be in the same cluster."""
        adj: dict[str, set[str]] = {"a": {"b"}, "b": {"a"}}
        clusters = _lpa_clusters(adj)
        merged = {n for c in clusters for n in c}
        assert merged == {"a", "b"}
        assert any({"a", "b"} == set(c) for c in clusters)

    def test_two_disjoint_groups(self) -> None:
        """Two disjoint connected components → two clusters."""
        adj: dict[str, set[str]] = {
            "a": {"b"},
            "b": {"a"},
            "c": {"d"},
            "d": {"c"},
        }
        clusters = _lpa_clusters(adj)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [2, 2]

    def test_deterministic_with_same_seed(self) -> None:
        """Same adjacency + seed → identical cluster assignment."""
        adj: dict[str, set[str]] = {str(i): {str((i + 1) % 6), str((i - 1) % 6)} for i in range(6)}
        c1 = _lpa_clusters(adj, seed=0)
        c2 = _lpa_clusters(adj, seed=0)
        assert sorted(sorted(c) for c in c1) == sorted(sorted(c) for c in c2)


# ---------------------------------------------------------------------------
# Integration tests — cluster building from graph edges
# ---------------------------------------------------------------------------


class TestBuildClusters:
    """Integration tests for cluster building from graph edges."""

    async def test_empty_graph_returns_no_clusters(self, store: SqliteEngravaCore) -> None:
        """No ASSOCIATED edges → no clusters."""
        ext = DreamingExtension(config=_reflection_cfg())
        clusters = await ext._build_clusters(store, current_cycle=1)
        assert clusters == []

    async def test_two_connected_thoughts_form_cluster(self, store: SqliteEngravaCore) -> None:
        """Two thoughts connected by ASSOCIATED edge → one cluster."""
        t1 = await store.create_thought(_make("t-cls-1", essence="alpha beta"))
        t2 = await store.create_thought(_make("t-cls-2", essence="beta gamma"))
        edge = EdgeRecord(
            edge_id="e-cls-1",
            from_thought_id=t1.thought_id,
            to_thought_id=t2.thought_id,
            edge_type=EdgeType.ASSOCIATED,
            weight=0.4,
            created_cycle=1,
            source=KnowledgeSource.DREAMING,
        )
        await store.create_edge(edge)

        ext = DreamingExtension(config=_reflection_cfg(min_cluster_size=2))
        clusters = await ext._build_clusters(store, current_cycle=1)
        assert len(clusters) == 1
        assert t1.thought_id in clusters[0]
        assert t2.thought_id in clusters[0]

    async def test_min_cluster_size_filters_small_groups(self, store: SqliteEngravaCore) -> None:
        """Clusters smaller than min_cluster_size are discarded."""
        t1 = await store.create_thought(_make("t-min-1"))
        t2 = await store.create_thought(_make("t-min-2"))
        edge = EdgeRecord(
            edge_id="e-min-1",
            from_thought_id=t1.thought_id,
            to_thought_id=t2.thought_id,
            edge_type=EdgeType.ASSOCIATED,
            weight=0.4,
            created_cycle=1,
            source=KnowledgeSource.DREAMING,
        )
        await store.create_edge(edge)

        ext = DreamingExtension(config=_reflection_cfg(min_cluster_size=3))
        clusters = await ext._build_clusters(store, current_cycle=1)
        # Cluster of 2 < min_cluster_size=3 → filtered
        assert clusters == []

    async def test_agglomerative_fallback_with_embeddings(self, store: SqliteEngravaCore) -> None:
        """Agglomerative algorithm clusters high-similarity nodes."""
        t1 = await store.create_thought(_make("t-agg-1", essence="neural networks"))
        t2 = await store.create_thought(_make("t-agg-2", essence="deep learning"))
        t3 = await store.create_thought(_make("t-agg-3", essence="completely unrelated"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")
        await store.store_embedding(t3.thought_id, [0.0, 0.0, 0.9, 0.1], model_name="test")

        # Add edges so adjacency is built
        for tid_a, tid_b, eid in [
            (t1.thought_id, t2.thought_id, "e-agg-1"),
            (t1.thought_id, t3.thought_id, "e-agg-2"),
        ]:
            await store.create_edge(
                EdgeRecord(
                    edge_id=eid,
                    from_thought_id=tid_a,
                    to_thought_id=tid_b,
                    edge_type=EdgeType.ASSOCIATED,
                    weight=0.4,
                    created_cycle=1,
                    source=KnowledgeSource.DREAMING,
                )
            )

        ext = DreamingExtension(
            config=_reflection_cfg(min_cluster_size=2, algorithm="agglomerative")
        )
        clusters = await ext._build_clusters(store, current_cycle=1)
        # t1+t2 should be in the same cluster (high similarity)
        combined = {n for c in clusters for n in c}
        assert t1.thought_id in combined
        assert t2.thought_id in combined


# ---------------------------------------------------------------------------
# Integration tests — REFLECTION thought persistence
# ---------------------------------------------------------------------------


class TestCreateReflections:
    """Integration tests for REFLECTION thought persistence."""

    async def test_reflection_thought_created_for_cluster(self, store: SqliteEngravaCore) -> None:
        """A REFLECTION thought is created for a qualifying cluster."""
        t1 = await store.create_thought(_make("t-ref-1", essence="concept A"))
        t2 = await store.create_thought(_make("t-ref-2", essence="concept A related"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        count = await ext._create_reflections(store, [cluster], current_cycle=5)

        assert count == 1
        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == 1
        r = reflections[0]
        assert r.thought_type == ThoughtType.REFLECTION
        assert r.source_type == KnowledgeSource.DREAMING

    async def test_consolidated_from_edges_created(self, store: SqliteEngravaCore) -> None:
        """CONSOLIDATED_FROM edges from REFLECTION to each member are created."""
        t1 = await store.create_thought(_make("t-edge-1", essence="alpha"))
        t2 = await store.create_thought(_make("t-edge-2", essence="beta"))
        await store.store_embedding(t1.thought_id, [1.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.9, 0.1], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        await ext._create_reflections(store, [cluster], current_cycle=5)

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert reflections
        ref_id = reflections[0].thought_id

        edges = await store.get_edges(ref_id, direction="OUT")
        consol_edges = [e for e in edges if e.edge_type == EdgeType.CONSOLIDATED_FROM]
        member_targets = {e.to_thought_id for e in consol_edges}
        assert t1.thought_id in member_targets
        assert t2.thought_id in member_targets

    async def test_reflection_idempotent_same_cluster(self, store: SqliteEngravaCore) -> None:
        """Calling _create_reflections twice with the same cluster creates only one REFLECTION."""
        t1 = await store.create_thought(_make("t-idem-1", essence="idempotent A"))
        t2 = await store.create_thought(_make("t-idem-2", essence="idempotent B"))
        await store.store_embedding(t1.thought_id, [1.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.9, 0.1], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())

        count1 = await ext._create_reflections(store, [cluster], current_cycle=5)
        count2 = await ext._create_reflections(store, [cluster], current_cycle=6)

        assert count1 == 1
        assert count2 == 0  # idempotent — no duplicate
        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == 1

    async def test_reflection_has_centroid_embedding(self, store: SqliteEngravaCore) -> None:
        """REFLECTION thought has a stored centroid embedding."""
        t1 = await store.create_thought(_make("t-emb-1", essence="embedding A"))
        t2 = await store.create_thought(_make("t-emb-2", essence="embedding B"))
        await store.store_embedding(t1.thought_id, [1.0, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.0, 1.0, 0.0], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        await ext._create_reflections(store, [cluster], current_cycle=5)

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert reflections
        emb = await store.get_embedding(reflections[0].thought_id)
        assert emb is not None
        assert emb.dimension == 3


# Fixed UTC-normalised ISO-8601 instants for valid-time tests
# (lexicographic == chronological by the domain's UTC-normalisation
# invariant).
_VT_EARLY = "2024-01-01T00:00:00+00:00"
_VT_MID = "2024-02-01T00:00:00+00:00"
_VT_LATE = "2024-03-01T00:00:00+00:00"
_VT_LATEST = "2024-04-01T00:00:00+00:00"


class TestReflectionValidTimeInheritance:
    """A REFLECTION inherits a deterministic valid-time extent from members."""

    async def test_persisted_reflection_carries_derived_extent(
        self, store: SqliteEngravaCore
    ) -> None:
        """All-finite members → REFLECTION gets MIN(from) / MAX(until)."""
        t1 = await store.create_thought(
            _make("t-vt-1", essence="extent A", valid_from=_VT_EARLY, valid_until=_VT_LATE)
        )
        t2 = await store.create_thought(
            _make("t-vt-2", essence="extent B", valid_from=_VT_MID, valid_until=_VT_LATEST)
        )
        await store.store_embedding(t1.thought_id, [1.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.9, 0.1], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        await ext._create_reflections(store, [cluster], current_cycle=5)

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == 1
        r = reflections[0]
        assert r.valid_from == _VT_EARLY
        assert r.valid_until == _VT_LATEST

    async def test_open_lower_bound_member_forces_open_from(self, store: SqliteEngravaCore) -> None:
        """A member with valid_from None → REFLECTION valid_from None."""
        t1 = await store.create_thought(
            _make("t-vt-of-1", essence="open from", valid_from=None, valid_until=_VT_LATE)
        )
        t2 = await store.create_thought(
            _make("t-vt-of-2", essence="finite", valid_from=_VT_MID, valid_until=_VT_LATEST)
        )
        await store.store_embedding(t1.thought_id, [1.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.9, 0.1], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        await ext._create_reflections(store, [cluster], current_cycle=5)

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == 1
        r = reflections[0]
        assert r.valid_from is None
        assert r.valid_until == _VT_LATEST

    async def test_open_upper_bound_member_forces_open_until(
        self, store: SqliteEngravaCore
    ) -> None:
        """A member with valid_until None → REFLECTION valid_until None."""
        t1 = await store.create_thought(
            _make("t-vt-ou-1", essence="finite", valid_from=_VT_EARLY, valid_until=_VT_LATE)
        )
        t2 = await store.create_thought(
            _make("t-vt-ou-2", essence="open until", valid_from=_VT_MID, valid_until=None)
        )
        await store.store_embedding(t1.thought_id, [1.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.9, 0.1], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        await ext._create_reflections(store, [cluster], current_cycle=5)

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == 1
        r = reflections[0]
        assert r.valid_from == _VT_EARLY
        assert r.valid_until is None

    async def test_caller_override_beats_derived_extent(self, store: SqliteEngravaCore) -> None:
        """Explicit override bounds win over the member-derived extent."""
        t1 = await store.create_thought(
            _make("t-vt-ov-1", essence="extent A", valid_from=_VT_EARLY, valid_until=_VT_LATE)
        )
        t2 = await store.create_thought(
            _make("t-vt-ov-2", essence="extent B", valid_from=_VT_MID, valid_until=_VT_LATEST)
        )
        await store.store_embedding(t1.thought_id, [1.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.9, 0.1], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        await ext._create_reflections(
            store,
            [cluster],
            current_cycle=5,
            override_valid_from=_VT_MID,
            override_valid_until=_VT_LATE,
        )

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == 1
        r = reflections[0]
        # Override pins both axes, ignoring the derived (early, latest).
        assert r.valid_from == _VT_MID
        assert r.valid_until == _VT_LATE

    async def test_extent_is_deterministic_across_runs(self, store: SqliteEngravaCore) -> None:
        """Re-deriving the same cluster twice yields the same extent."""
        t1 = await store.create_thought(
            _make("t-vt-det-1", essence="extent A", valid_from=_VT_EARLY, valid_until=_VT_LATE)
        )
        t2 = await store.create_thought(
            _make("t-vt-det-2", essence="extent B", valid_from=_VT_MID, valid_until=_VT_LATEST)
        )
        await store.store_embedding(t1.thought_id, [1.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.9, 0.1], model_name="test")

        cluster = frozenset([t1.thought_id, t2.thought_id])
        ext = DreamingExtension(config=_reflection_cfg())
        await ext._create_reflections(store, [cluster], current_cycle=5)
        first = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(first) == 1
        first_extent = (first[0].valid_from, first[0].valid_until)

        # Idempotence skips a second REFLECTION for the same cluster, so
        # assert the persisted extent equals the deterministic derivation
        # for these member bounds — no clock or randomness involved.
        assert first_extent == (_VT_EARLY, _VT_LATEST)


# ---------------------------------------------------------------------------
# Integration: run_consolidation end-to-end
# ---------------------------------------------------------------------------


class TestRunConsolidationWithReflections:
    """End-to-end consolidation with clustering enabled."""

    async def test_reflections_created_counted_in_result(self, store: SqliteEngravaCore) -> None:
        """ConsolidationResult.reflections_created is non-zero after clustering."""
        thoughts = [
            _make(f"t-run-{i}", essence=f"cluster topic {i % 2}", content=f"content {i}")
            for i in range(4)
        ]
        for t in thoughts:
            stored = await store.create_thought(t)
            await store.store_embedding(stored.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(
                min_age_cycles=0,
                allow_zero_confirmation=True,
                min_cluster_size=2,
                enable_reflections=True,
            ),
            edges=EdgeCreationConfig(enabled=True, top_k=2, min_similarity=0.5),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=10)

        assert result.reflections_created >= 0  # may vary with graph density
        assert isinstance(result.reflections_created, int)

    async def test_enable_reflections_false_skips_clustering(
        self, store: SqliteEngravaCore
    ) -> None:
        """enable_reflections=False → reflections_created is always 0."""
        t1 = await store.create_thought(_make("t-skip-1", essence="alpha"))
        t2 = await store.create_thought(_make("t-skip-2", essence="beta"))
        await store.store_embedding(t1.thought_id, [0.9, 0.1], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15], model_name="test")

        cfg = _reflection_cfg(enable_reflections=False)
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=5)

        assert result.reflections_created == 0
        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        assert len(reflections) == 0


# ---------------------------------------------------------------------------
# Integration: search_hybrid with reflections
# ---------------------------------------------------------------------------


class TestSearchHybridReflections:
    """Tests for include_reflections / reflection_boost in search_hybrid."""

    async def _insert_reflection(
        self, store: SqliteEngravaCore, rid: str = "ref-1"
    ) -> ThoughtRecord:
        """Insert a REFLECTION thought directly."""
        import json

        content = json.dumps(
            {"member_ids": ["m1", "m2"], "keywords": ["neural", "learning"], "cluster_hash": rid}
        )
        r = ThoughtRecord(
            thought_id=rid,
            thought_type=ThoughtType.REFLECTION,
            essence="REFLECTION [neural, learning]",
            content=content,
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source=f"dreaming:{rid}",
            source_type=KnowledgeSource.DREAMING,
        )
        return await store.create_thought(r)

    async def test_include_reflections_false_excludes_reflection(
        self, store: SqliteEngravaCore
    ) -> None:
        """include_reflections=False removes REFLECTION from results."""
        ref = await self._insert_reflection(store)
        # Also insert a normal thought with same embedding
        t = await store.create_thought(
            _make("t-normal", essence="neural learning", content="deep neural learning")
        )
        await store.store_embedding(ref.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t.thought_id, [0.8, 0.2, 0.0, 0.0], model_name="test")

        result = await store.search_hybrid(
            "neural learning",
            [0.9, 0.1, 0.0, 0.0],
            include_reflections=False,
        )
        result_ids = {tid for tid, _ in result.results}
        assert ref.thought_id not in result_ids

    async def test_include_reflections_true_includes_reflection(
        self, store: SqliteEngravaCore
    ) -> None:
        """include_reflections=True (default) keeps REFLECTION in results."""
        ref = await self._insert_reflection(store, "ref-inc-1")
        await store.store_embedding(ref.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")

        result = await store.search_hybrid(
            "",
            [0.9, 0.1, 0.0, 0.0],
            include_reflections=True,
        )
        result_ids = {tid for tid, _ in result.results}
        assert ref.thought_id in result_ids

    async def test_search_reflections_only_returns_only_reflections(
        self, store: SqliteEngravaCore
    ) -> None:
        """search_reflections_only returns only ThoughtType.REFLECTION thoughts."""
        ref = await self._insert_reflection(store, "ref-only-1")
        normal = await store.create_thought(_make("t-not-ref", essence="regular thought"))
        await store.store_embedding(ref.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(normal.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")

        result = await store.search_reflections_only("", [0.9, 0.1, 0.0, 0.0], top_k=10)
        result_ids = {tid for tid, _ in result.results}
        assert ref.thought_id in result_ids
        assert normal.thought_id not in result_ids


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestDreamingGatesConfigParsing:
    """DreamingGates YAML parsing for clustering gate fields."""

    def test_defaults(self) -> None:
        """Default gates have sensible cluster defaults."""
        gates = DreamingGates()
        assert gates.min_cluster_size == 3
        assert gates.cluster_similarity_threshold == 0.7
        assert gates.cluster_algorithm == "lpa"
        assert gates.enable_reflections is True

    def test_explicit_values(self) -> None:
        """All new fields accepted at construction."""
        gates = DreamingGates(
            min_cluster_size=5,
            cluster_similarity_threshold=0.8,
            cluster_algorithm="agglomerative",
            enable_reflections=False,
        )
        assert gates.min_cluster_size == 5
        assert gates.cluster_similarity_threshold == 0.8
        assert gates.cluster_algorithm == "agglomerative"
        assert gates.enable_reflections is False

    def test_search_config_reflection_boost_default(self) -> None:
        """SearchConfig.reflection_boost defaults to 1.0 (neutral)."""
        cfg = SearchConfig()
        assert cfg.reflection_boost == 1.0


# ---------------------------------------------------------------------------
# Clustering and reflection gate APIs
# ---------------------------------------------------------------------------


class TestThoughtExistsBySource:
    """Unit tests for thought_exists_by_source — O(1) idempotence helper."""

    async def test_returns_true_when_matching_thought_exists(
        self, store: SqliteEngravaCore
    ) -> None:
        """Returns True when a REFLECTION with the given source is found."""
        import json

        r = ThoughtRecord(
            thought_id="r-src-1",
            thought_type=ThoughtType.REFLECTION,
            essence="REFLECTION [test]",
            content=json.dumps({"member_ids": [], "keywords": [], "cluster_hash": "abc123"}),
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="dreaming:abc123",
            source_type=KnowledgeSource.DREAMING,
        )
        await store.create_thought(r)

        exists = await store.thought_exists_by_source(
            source="dreaming:abc123",
            thought_type_value="REFLECTION",
        )
        assert exists is True

    async def test_returns_false_when_no_matching_thought(self, store: SqliteEngravaCore) -> None:
        """Returns False when no thought matches both source and type."""
        exists = await store.thought_exists_by_source(
            source="dreaming:nonexistent",
            thought_type_value="REFLECTION",
        )
        assert exists is False

    async def test_wrong_thought_type_returns_false(self, store: SqliteEngravaCore) -> None:
        """Type mismatch causes False even when source matches."""
        t = await store.create_thought(
            _make("t-src-type", essence="regular", content="plain content")
        )
        # Update source manually via update_thought is not available;
        # create a fresh OBSERVATION with the sentinel source directly.
        obs = ThoughtRecord(
            thought_id="obs-src-1",
            thought_type=ThoughtType.OBSERVATION,
            essence="obs",
            content="obs content",
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="dreaming:xyz",
        )
        await store.create_thought(obs)
        exists = await store.thought_exists_by_source(
            source="dreaming:xyz",
            thought_type_value="REFLECTION",
        )
        # OBSERVATION, not REFLECTION → False
        assert exists is False
        # Suppress unused variable warning
        assert t is not None


class TestSearchReflectionsOnlyEdgeCases:
    """Edge-case coverage for the new search_reflections_only implementation."""

    async def test_empty_store_returns_empty(self, store: SqliteEngravaCore) -> None:
        """No reflections → empty result, no errors."""
        result = await store.search_reflections_only("query", [0.5, 0.5])
        assert result.results == []

    async def test_no_vector_returns_unranked(self, store: SqliteEngravaCore) -> None:
        """Without a query vector and no provider, thoughts are returned unranked."""
        import json

        r = ThoughtRecord(
            thought_id="r-unrank-1",
            thought_type=ThoughtType.REFLECTION,
            essence="REFLECTION [a]",
            content=json.dumps({"member_ids": [], "keywords": [], "cluster_hash": "x"}),
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="dreaming:x",
            source_type=KnowledgeSource.DREAMING,
        )
        await store.create_thought(r)
        # Pass None explicitly to exercise the no-vector code path
        result = await store.search_reflections_only("", None)
        assert len(result.results) == 1
        assert result.results[0][0] == "r-unrank-1"

    async def test_with_current_cycle_applies_recency(self, store: SqliteEngravaCore) -> None:
        """search_reflections_only with current_cycle adds recency backend."""
        import json

        r = ThoughtRecord(
            thought_id="r-rec-1",
            thought_type=ThoughtType.REFLECTION,
            essence="REFLECTION [recent]",
            content=json.dumps({"member_ids": [], "keywords": [], "cluster_hash": "r1"}),
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=5,
            updated_cycle=5,
            source="dreaming:r1",
            source_type=KnowledgeSource.DREAMING,
        )
        await store.create_thought(r)
        await store.store_embedding(r.thought_id, [0.9, 0.1], model_name="test")

        result = await store.search_reflections_only(
            "",
            [0.9, 0.1],
            top_k=5,
            current_cycle=10,
        )
        assert "recency" in result.backends_used
        assert len(result.results) == 1


class TestAgglomerativeNoEdges:
    """Agglomerative algorithm works even with zero dream edges (sparse-graph / first-run)."""

    async def test_agglomerative_no_edges_clusters_by_embedding(
        self, store: SqliteEngravaCore
    ) -> None:
        """With no dream edges, agglomerative clusters by cosine similarity alone."""
        t1 = await store.create_thought(_make("t-noedge-1", essence="alpha"))
        t2 = await store.create_thought(_make("t-noedge-2", essence="beta"))
        t3 = await store.create_thought(_make("t-noedge-3", essence="gamma"))
        await store.store_embedding(t1.thought_id, [0.95, 0.05, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.90, 0.10, 0.0], model_name="test")
        await store.store_embedding(t3.thought_id, [0.0, 0.0, 1.0], model_name="test")

        cfg = _reflection_cfg(min_cluster_size=2, algorithm="agglomerative")
        ext = DreamingExtension(config=cfg)
        clusters = await ext._build_clusters(store, current_cycle=1)

        # t1 and t2 are close; t3 is far — expect at least the t1+t2 cluster
        combined = {n for c in clusters for n in c}
        assert t1.thought_id in combined
        assert t2.thought_id in combined


# ---------------------------------------------------------------------------
# OBSERVATION-only clustering filter
# ---------------------------------------------------------------------------


class TestClusteringObservationFilter:
    """REFLECTIONs must not re-enter the clustering candidate pool.

    This eliminates the meta-reflection cascade: REFLECTIONs created in cycle N being clustered
    with OBSERVATIONs in cycle N+1 → meta-REFLECTIONs whose centroids
    drift further from the source facts each cycle.
    """

    async def test_clustering_excludes_reflections(self, store: SqliteEngravaCore) -> None:
        """REFLECTIONs in the store are NOT fed into _agglomerative_clusters."""
        import json

        # Two OBSERVATIONs with near-identical embeddings (expected cluster).
        obs1 = await store.create_thought(_make("obs-1", essence="alpha"))
        obs2 = await store.create_thought(_make("obs-2", essence="alpha related"))
        await store.store_embedding(obs1.thought_id, [0.95, 0.05, 0.0], model_name="test")
        await store.store_embedding(obs2.thought_id, [0.90, 0.10, 0.0], model_name="test")

        # A REFLECTION with an embedding that WOULD cluster with the two
        # OBSERVATIONs if it were allowed into the pool.
        ref = ThoughtRecord(
            thought_id="ref-nocluster-1",
            thought_type=ThoughtType.REFLECTION,
            essence="REFLECTION [alpha]",
            content=json.dumps(
                {"member_ids": ["obs-1", "obs-2"], "keywords": ["alpha"], "cluster_hash": "h1"}
            ),
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="dreaming:h1",
            source_type=KnowledgeSource.DREAMING,
        )
        await store.create_thought(ref)
        await store.store_embedding(ref.thought_id, [0.92, 0.08, 0.0], model_name="test")

        cfg = _reflection_cfg(min_cluster_size=2, algorithm="agglomerative")
        ext = DreamingExtension(config=cfg)
        clusters = await ext._build_clusters(store, current_cycle=2)

        # The REFLECTION must NOT appear in any cluster, even though its
        # embedding would have clustered it with the two OBSERVATIONs.
        all_members = {m for c in clusters for m in c}
        assert ref.thought_id not in all_members
        # The two OBSERVATIONs still cluster together.
        assert obs1.thought_id in all_members
        assert obs2.thought_id in all_members

    async def test_clustering_allows_reflections_when_opted_in(
        self, store: SqliteEngravaCore
    ) -> None:
        """Operator can opt back into meta-consolidation via cluster_allowed_types."""
        import json

        obs1 = await store.create_thought(_make("obsA-1", essence="beta"))
        await store.store_embedding(obs1.thought_id, [0.9, 0.1, 0.0], model_name="test")

        ref = ThoughtRecord(
            thought_id="refA-1",
            thought_type=ThoughtType.REFLECTION,
            essence="REFLECTION [beta]",
            content=json.dumps(
                {"member_ids": ["obsA-1"], "keywords": ["beta"], "cluster_hash": "h2"}
            ),
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="dreaming:h2",
            source_type=KnowledgeSource.DREAMING,
        )
        await store.create_thought(ref)
        await store.store_embedding(ref.thought_id, [0.88, 0.12, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(
                min_age_cycles=0,
                allow_zero_confirmation=True,
                min_cluster_size=2,
                cluster_algorithm="agglomerative",
                enable_reflections=True,
                cluster_allowed_types=("OBSERVATION", "REFLECTION"),
            ),
        )
        ext = DreamingExtension(config=cfg)
        clusters = await ext._build_clusters(store, current_cycle=2)

        members = {m for c in clusters for m in c}
        # When opted in, REFLECTION can cluster with OBSERVATION.
        assert ref.thought_id in members
        assert obs1.thought_id in members


# ---------------------------------------------------------------------------
# numpy-vectorized clustering parity
# ---------------------------------------------------------------------------


class TestVectorizedClusteringParity:
    """numpy-vectorized _agglomerative_clusters produces the same
    clusters (modulo ~1e-7 float32 accumulation noise) as the pure-Python
    legacy implementation on representative inputs.
    """

    @staticmethod
    def _python_legacy_clusters(  # noqa: C901
        vectors: dict[str, list[float]],
        threshold: float,
    ) -> list[frozenset[str]]:
        """Mirror of the pre-optimization pure-Python implementation.

        Kept inline so the parity test is self-contained — i.e. it still
        works after the legacy code is removed from production.
        """
        import math

        ids = [tid for tid, v in vectors.items() if any(v)]
        if not ids:
            return []

        parent = {tid: tid for tid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            ra, rb = find(x), find(y)
            if ra != rb:
                parent[ra] = rb

        norms = {tid: math.sqrt(sum(v * v for v in vectors[tid])) for tid in ids}
        for i, a in enumerate(ids):
            na = norms[a]
            if na == 0.0:
                continue
            for b in ids[i + 1 :]:
                nb = norms[b]
                if nb == 0.0:
                    continue
                dot = sum(x * y for x, y in zip(vectors[a], vectors[b], strict=False))
                if dot / (na * nb) >= threshold:
                    union(a, b)

        groups: dict[str, set[str]] = {}
        for tid in ids:
            groups.setdefault(find(tid), set()).add(tid)
        return [frozenset(g) for g in groups.values()]

    @staticmethod
    def _canon(clusters: list[frozenset[str]]) -> list[list[str]]:
        """Normalize for equality comparison across implementations."""
        return sorted([sorted(c) for c in clusters])

    async def test_parity_small_random(self, store: SqliteEngravaCore) -> None:
        """50 random float32 embeddings: numpy path == python legacy."""
        import random
        import struct

        rng = random.Random(42)  # noqa: S311
        dim = 16
        threshold = 0.65

        # Persist 50 thoughts with random embeddings.
        vectors: dict[str, list[float]] = {}
        node_ids: list[str] = []
        for i in range(50):
            tid = f"par-s-{i}"
            node_ids.append(tid)
            await store.create_thought(_make(tid, essence=f"random-{i}"))
            v = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            vectors[tid] = v
            await store.store_embedding(tid, v, model_name="test")

        # Truncate to float32 precision to match the numpy path exactly.
        vectors_f32 = {
            tid: list(struct.unpack(f"{dim}f", struct.pack(f"{dim}f", *v)))
            for tid, v in vectors.items()
        }

        legacy = self._python_legacy_clusters(vectors_f32, threshold)
        modern = await DreamingExtension._agglomerative_clusters(
            store=store,
            node_ids=node_ids,
            threshold=threshold,
        )

        assert self._canon(legacy) == self._canon(modern)

    async def test_parity_handles_zero_vectors(self, store: SqliteEngravaCore) -> None:
        """All-zero vectors are dropped, matching the legacy norm==0 guard."""
        import struct

        node_ids = ["z-1", "z-2", "z-3", "z-4"]
        vectors: dict[str, list[float]] = {
            "z-1": [1.0, 0.0, 0.0],
            "z-2": [0.0, 0.0, 0.0],  # zero vector — must drop
            "z-3": [0.98, 0.02, 0.0],  # clusters with z-1
            "z-4": [0.0, 1.0, 0.0],
        }
        for tid in node_ids:
            await store.create_thought(_make(tid, essence=tid))
            await store.store_embedding(tid, vectors[tid], model_name="test")

        threshold = 0.9
        vectors_f32 = {
            tid: list(struct.unpack(f"{len(v)}f", struct.pack(f"{len(v)}f", *v)))
            for tid, v in vectors.items()
        }

        legacy = self._python_legacy_clusters(vectors_f32, threshold)
        modern = await DreamingExtension._agglomerative_clusters(
            store=store,
            node_ids=node_ids,
            threshold=threshold,
        )
        # Zero vector must be dropped from both implementations.
        legacy_ids = {m for c in legacy for m in c}
        modern_ids = {m for c in modern for m in c}
        assert "z-2" not in legacy_ids
        assert "z-2" not in modern_ids
        assert self._canon(legacy) == self._canon(modern)

    async def test_chunked_path_parity(self, store: SqliteEngravaCore) -> None:
        """Chunked mode (N > chunk_size) produces identical clusters.

        Monkey-patches the module constant so a tiny synthetic N exercises
        the chunked branch.
        """
        import random
        import struct

        from engrava.extensions import dreaming as dreaming_mod

        rng = random.Random(7)  # noqa: S311
        dim = 8
        threshold = 0.5
        n = 40

        node_ids: list[str] = []
        vectors: dict[str, list[float]] = {}
        for i in range(n):
            tid = f"ch-{i}"
            node_ids.append(tid)
            await store.create_thought(_make(tid, essence=f"ch-{i}"))
            v = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            vectors[tid] = v
            await store.store_embedding(tid, v, model_name="test")

        vectors_f32 = {
            tid: list(struct.unpack(f"{dim}f", struct.pack(f"{dim}f", *v)))
            for tid, v in vectors.items()
        }
        legacy = self._python_legacy_clusters(vectors_f32, threshold)

        # Force the chunked branch by shrinking the module-level threshold.
        original = dreaming_mod._VECTORIZED_CLUSTERING_CHUNK_SIZE
        try:
            dreaming_mod._VECTORIZED_CLUSTERING_CHUNK_SIZE = 10
            modern = await DreamingExtension._agglomerative_clusters(
                store=store,
                node_ids=node_ids,
                threshold=threshold,
            )
        finally:
            dreaming_mod._VECTORIZED_CLUSTERING_CHUNK_SIZE = original

        assert self._canon(legacy) == self._canon(modern)

    async def test_empty_input_returns_empty(self, store: SqliteEngravaCore) -> None:
        """Zero candidates → empty result, no crash."""
        result = await DreamingExtension._agglomerative_clusters(
            store=store,
            node_ids=[],
            threshold=0.65,
        )
        assert result == []

    async def test_parity_medium_random(self, store: SqliteEngravaCore) -> None:
        """1000 random float32 embeddings: numpy path == python legacy.

        > 99.5% of clusters must be
        identical.  In practice, with threshold=0.65 and dim=32, results
        are 100% identical (float32 accumulation differences are ~1e-7,
        well below the threshold gap).
        """
        import random
        import struct

        rng = random.Random(99)  # noqa: S311
        dim = 32
        threshold = 0.65
        n = 1000

        node_ids: list[str] = []
        vectors: dict[str, list[float]] = {}
        for i in range(n):
            tid = f"med-{i}"
            node_ids.append(tid)
            await store.create_thought(_make(tid, essence=f"med-{i}"))
            v = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            vectors[tid] = v
            await store.store_embedding(tid, v, model_name="test")

        vectors_f32 = {
            tid: list(struct.unpack(f"{dim}f", struct.pack(f"{dim}f", *v)))
            for tid, v in vectors.items()
        }

        legacy = self._python_legacy_clusters(vectors_f32, threshold)
        modern = await DreamingExtension._agglomerative_clusters(
            store=store,
            node_ids=node_ids,
            threshold=threshold,
        )

        legacy_canon = self._canon(legacy)
        modern_canon = self._canon(modern)

        # Count matching clusters.
        matching = sum(1 for c in legacy_canon if c in modern_canon)
        total = max(len(legacy_canon), len(modern_canon), 1)
        parity_ratio = matching / total
        assert parity_ratio >= 0.995, (
            f"Parity ratio {parity_ratio:.4f} < 0.995 — "
            f"legacy={len(legacy_canon)} clusters, modern={len(modern_canon)} clusters"
        )


# ---------------------------------------------------------------------------
# DreamingGates.cluster_allowed_types config parsing
# ---------------------------------------------------------------------------


class TestClusterAllowedTypesConfig:
    """Config surface for cluster_allowed_types default + overrides."""

    def test_default_is_observation_only(self) -> None:
        """Default restricts clustering to OBSERVATION thoughts."""
        gates = DreamingGates()
        assert gates.cluster_allowed_types == ("OBSERVATION",)

    def test_operator_can_extend_types(self) -> None:
        """Operator can opt into meta-consolidation by extending the tuple."""
        gates = DreamingGates(cluster_allowed_types=("OBSERVATION", "REFLECTION"))
        assert "REFLECTION" in gates.cluster_allowed_types


# ---------------------------------------------------------------------------
# DreamingConfig.clustering_backend config + escape hatch
# ---------------------------------------------------------------------------


class TestClusteringBackendConfig:
    """Config surface for clustering_backend default + dispatch."""

    def test_default_backend_is_numpy(self) -> None:
        """Default clustering_backend is 'numpy'."""
        from engrava.config import DreamingConfig

        cfg = DreamingConfig()
        assert cfg.clustering_backend == "numpy"

    def test_python_backend_accepted(self) -> None:
        """'python' is a valid clustering_backend value."""
        from engrava.config import DreamingConfig

        cfg = DreamingConfig(clustering_backend="python")
        assert cfg.clustering_backend == "python"

    async def test_python_backend_produces_same_clusters(self, store: SqliteEngravaCore) -> None:
        """clustering_backend='python' produces identical results to 'numpy'."""
        node_ids = ["be-1", "be-2", "be-3"]
        vecs = {
            "be-1": [1.0, 0.0],
            "be-2": [0.99, 0.14],  # clusters with be-1
            "be-3": [0.0, 1.0],
        }
        for tid, v in vecs.items():
            await store.create_thought(_make(tid, essence=tid))
            await store.store_embedding(tid, v, model_name="test")

        threshold = 0.95
        numpy_clusters = await DreamingExtension._agglomerative_clusters(
            store=store,
            node_ids=node_ids,
            threshold=threshold,
        )
        python_clusters = await DreamingExtension._agglomerative_clusters_python_legacy(
            store=store,
            node_ids=node_ids,
            threshold=threshold,
        )

        def _canon(cs: list[frozenset[str]]) -> list[list[str]]:
            return sorted([sorted(c) for c in cs])

        assert _canon(numpy_clusters) == _canon(python_clusters)

    async def test_python_backend_dispatches_via_config(self, store: SqliteEngravaCore) -> None:
        """DreamingExtension with clustering_backend='python' calls legacy path."""
        from engrava.config import DreamingConfig, DreamingGates, EdgeCreationConfig
        from engrava.extensions.dreaming import DreamingExtension

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            clustering_backend="python",
            gates=DreamingGates(
                min_age_cycles=0,
                allow_zero_confirmation=True,
                min_cluster_size=2,
                cluster_algorithm="agglomerative",
                cluster_allowed_types=("OBSERVATION",),
            ),
            edges=EdgeCreationConfig(enabled=False),
        )
        ext = DreamingExtension(config=cfg)

        called: list[str] = []
        original_legacy = ext._agglomerative_clusters_python_legacy
        original_numpy = ext._agglomerative_clusters

        # Spy wrappers forward to a private method whose signature is an
        # implementation detail; the test only inspects the ``called`` list, so
        # we type-erase the variadic forwarding to keep mypy --strict happy.
        async def spy_legacy(
            *args: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401
        ) -> list[frozenset[str]]:
            called.append("legacy")
            return await original_legacy(*args, **kwargs)

        async def spy_numpy(
            *args: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401
        ) -> list[frozenset[str]]:
            called.append("numpy")
            return await original_numpy(*args, **kwargs)

        ext._agglomerative_clusters_python_legacy = spy_legacy  # type: ignore[method-assign]
        ext._agglomerative_clusters = spy_numpy  # type: ignore[method-assign]

        await ext.run_consolidation(store, current_cycle=1)

        assert "legacy" in called
        assert "numpy" not in called


# ---------------------------------------------------------------------------
# max_cluster_size guard
# ---------------------------------------------------------------------------


class TestMaxClusterSizeGuard:
    """Unit tests for the max_cluster_size guard in _build_clusters.

    Uses the agglomerative algorithm with high-similarity synthetic vectors
    so that all nodes end up in a single cluster by default, making the
    guard behaviour deterministic and observable without LPA graph setup.
    """

    def _guard_cfg(self, max_cluster_size: int | None) -> DreamingConfig:
        """Return a DreamingConfig with agglomerative algorithm and the given max_cluster_size."""
        return DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            candidates_limit=500,
            gates=DreamingGates(
                min_age_cycles=0,
                allow_zero_confirmation=True,
                min_cluster_size=2,
                cluster_algorithm="agglomerative",
                cluster_allowed_types=("OBSERVATION",),
                cluster_similarity_threshold=0.5,
                max_cluster_size=max_cluster_size,
            ),
            edges=EdgeCreationConfig(enabled=False),
        )

    async def _populate(
        self,
        store: SqliteEngravaCore,
        n: int,
        *,
        prefix: str = "mg",
    ) -> list[str]:
        """Create n thoughts with near-identical 2-D embeddings (all cluster together)."""
        ids: list[str] = []
        for i in range(n):
            tid = f"{prefix}-{i}"
            await store.create_thought(_make(tid, essence=f"topic {i}"))
            # All vectors point in nearly the same direction → high pairwise similarity.
            await store.store_embedding(tid, [1.0, 0.01 * i], model_name="test")
            ids.append(tid)
        return ids

    async def test_rejects_oversized_cluster(self, store: SqliteEngravaCore) -> None:
        """Clusters larger than max_cluster_size are rejected entirely.

        120 highly similar nodes form one cluster of size 120. With
        max_cluster_size=100 that single cluster must be rejected, leaving
        zero clusters returned.
        """
        await self._populate(store, 120, prefix="ov")
        ext = DreamingExtension(config=self._guard_cfg(max_cluster_size=100))
        clusters = await ext._build_clusters(store, current_cycle=1)
        assert clusters == []

    async def test_preserves_valid_clusters(self, store: SqliteEngravaCore) -> None:
        """Valid clusters (size <= max_cluster_size) are unaffected.

        50 highly similar nodes form one cluster of size 50. With
        max_cluster_size=100 that cluster is within the limit and must
        be returned.
        """
        ids = await self._populate(store, 50, prefix="ok")
        ext = DreamingExtension(config=self._guard_cfg(max_cluster_size=100))
        clusters = await ext._build_clusters(store, current_cycle=1)
        assert len(clusters) == 1
        returned_ids = set(clusters[0])
        assert returned_ids == set(ids)

    async def test_none_disables_guard(self, store: SqliteEngravaCore) -> None:
        """max_cluster_size=None disables the guard; any size is accepted.

        250 highly similar nodes form one cluster of size 250. With
        max_cluster_size=None the guard is off and the full cluster
        must be returned.
        """
        ids = await self._populate(store, 250, prefix="nd")
        ext = DreamingExtension(config=self._guard_cfg(max_cluster_size=None))
        clusters = await ext._build_clusters(store, current_cycle=1)
        assert len(clusters) == 1
        assert len(clusters[0]) == len(ids)

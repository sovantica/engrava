"""Tests for dreaming edge creation and graph-aware search.

Covers:
- Edge creation on promotion (ASSOCIATED + KnowledgeSource.DREAMING)
- Peer-similarity edge creation with configurable top-k
- Idempotence (re-run dream ≠ duplicate edges)
- Graph growth per dream cycle
- Graph-aware hybrid search signal (1-hop-weighted boost)
- Graph signal disabled → backward-compat ordering
- EdgeCreationConfig validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    SearchConfig,
    SqliteEngravaCore,
)
from engrava.config import (
    ConfigError,
    DreamingConfig,
    DreamingGates,
    EdgeCreationConfig,
    _parse_dreaming,
    _parse_edge_creation,
    _parse_search,
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
from engrava.extensions.dreaming import DreamingExtension

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


async def _open_store(db_path: Path) -> tuple[aiosqlite.Connection, SqliteEngravaCore]:
    """Open a store backed by the given database path."""
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(db=db)
    await store.ensure_schema()
    return (db, store)


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
) -> ThoughtRecord:
    """Minimal thought for edge tests."""
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
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh store with embeddings."""
    db = await aiosqlite.connect(str(tmp_path / "test.db"))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    yield s
    await db.close()


# ---------------------------------------------------------------------------
# TestEdgeCreationOnPromotion
# ---------------------------------------------------------------------------


class TestEdgeCreationOnPromotion:
    """Edge creation during dreaming consolidation."""

    async def test_promotion_creates_associated_edge_to_top_similar_peer(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A promoted thought links to its top similar peer."""
        t1 = await store.create_thought(
            _make("t-promo-1", essence="async python", content="async/await patterns"),
        )
        t2 = await store.create_thought(
            _make("t-neighbour-1", essence="python coroutines", content="coroutine semantics"),
        )
        # Store similar embeddings
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(
                enabled=True,
                top_k=1,
                min_similarity=0.5,
                edge_weight_factor=0.5,
            ),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=10)

        assert result.promoted_count >= 1
        assert result.edges_created >= 1

        # Check the edge exists
        edges = await store.get_edges(t1.thought_id, direction="BOTH")
        associated = [e for e in edges if e.edge_type == EdgeType.ASSOCIATED]
        assert len(associated) >= 1

    async def test_non_duplicate_integrity_error_is_not_swallowed(
        self,
        store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real integrity failure during dream edge creation propagates.

        Only a ``DuplicateEdgeError`` is a benign "already exists" skip. A
        CHECK / trigger / other integrity failure must surface rather than being
        logged away — otherwise the run reports edges it never persisted.
        """
        t1 = await store.create_thought(
            _make("t-c1-1", essence="async python", content="async/await patterns"),
        )
        t2 = await store.create_thought(
            _make("t-c1-2", essence="python coroutines", content="coroutine semantics"),
        )
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(
                enabled=True,
                top_k=1,
                min_similarity=0.5,
                edge_weight_factor=0.5,
            ),
        )
        ext = DreamingExtension(config=cfg)

        async def _raise_check_violation(_edge: EdgeRecord) -> EdgeRecord:
            msg = "CHECK constraint failed: edge_weight_range"
            raise aiosqlite.IntegrityError(msg)

        monkeypatch.setattr(store, "create_edge", _raise_check_violation)

        with pytest.raises(aiosqlite.IntegrityError, match="CHECK constraint failed"):
            await ext.run_consolidation(store, current_cycle=10)

    async def test_edge_weight_equals_half_cosine(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Dream-created edge weight equals factor multiplied by cosine score."""
        t1 = await store.create_thought(
            _make("t-weight-1", essence="weighted edge A", content="content A"),
        )
        t2 = await store.create_thought(
            _make("t-weight-2", essence="weighted edge B", content="content B"),
        )
        await store.store_embedding(t1.thought_id, [1.0, 0.0, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.8, 0.6, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(
                enabled=True,
                top_k=1,
                min_similarity=0.5,
                edge_weight_factor=0.5,
            ),
        )
        ext = DreamingExtension(config=cfg)
        await ext.run_consolidation(store, current_cycle=10)

        edges = await store.get_edges(t1.thought_id, direction="BOTH")
        associated = [e for e in edges if e.edge_type == EdgeType.ASSOCIATED]
        assert associated
        assert associated[0].weight == pytest.approx(0.4)

    async def test_edge_source_is_knowledge_source_dreaming(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Dream-created edges use KnowledgeSource.DREAMING."""
        t1 = await store.create_thought(
            _make("t-src-1", essence="ML training", content="training neural nets"),
        )
        t2 = await store.create_thought(
            _make("t-src-2", essence="deep learning", content="neural network training"),
        )
        await store.store_embedding(t1.thought_id, [0.8, 0.2, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.75, 0.25, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(enabled=True, top_k=1, min_similarity=0.5),
        )
        ext = DreamingExtension(config=cfg)
        await ext.run_consolidation(store, current_cycle=10)

        edges = await store.get_edges(t1.thought_id, direction="BOTH")
        dreaming_edges = [e for e in edges if e.source == KnowledgeSource.DREAMING]
        assert len(dreaming_edges) >= 1

    async def test_top_k_creates_multiple_peer_edges(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """top_k=2 creates up to two peer-similarity edges per thought."""
        t1 = await store.create_thought(
            _make("t-mp-1", essence="hub thought", content="connects to many"),
        )
        t2 = await store.create_thought(
            _make("t-mp-2", essence="neighbour A", content="related content A"),
        )
        t3 = await store.create_thought(
            _make("t-mp-3", essence="neighbour B", content="related content B"),
        )
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")
        await store.store_embedding(t3.thought_id, [0.8, 0.2, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(
                enabled=True,
                top_k=2,
                min_similarity=0.5,
            ),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=10)

        assert result.edges_created >= 2

    async def test_no_edge_when_disabled(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """edges.enabled=False → no edges created."""
        t1 = await store.create_thought(
            _make("t-dis-1", essence="topic A", content="content A"),
        )
        t2 = await store.create_thought(
            _make("t-dis-2", essence="topic A also", content="content A too"),
        )
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(enabled=False),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=10)

        assert result.edges_created == 0

    async def test_no_edge_without_embedding(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Promoted thought with no embedding → no edge created."""
        await store.create_thought(
            _make("t-noemb-1", essence="no vector", content="unembedded"),
        )

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(enabled=True, top_k=1, min_similarity=0.5),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=10)

        assert result.edges_created == 0


# ---------------------------------------------------------------------------
# TestEdgeIdempotence
# ---------------------------------------------------------------------------


class TestEdgeIdempotence:
    """Re-running dream on the same batch must not create duplicate edges."""

    async def test_rerun_dream_no_duplicate_edges(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Second consolidation run doesn't duplicate existing edges."""
        t1 = await store.create_thought(
            _make("t-idem-1", essence="idempotent A", content="content A"),
        )
        t2 = await store.create_thought(
            _make("t-idem-2", essence="idempotent B", content="content B"),
        )
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(enabled=True, top_k=1, min_similarity=0.5),
        )
        ext = DreamingExtension(config=cfg)

        await ext.run_consolidation(store, current_cycle=10)
        edges_after_first = await store.get_edges(t1.thought_id, direction="BOTH")

        r2 = await ext.run_consolidation(store, current_cycle=20)
        edges_after_second = await store.get_edges(t1.thought_id, direction="BOTH")

        # No new edges on second run (existing edge skipped)
        assert len(edges_after_second) == len(edges_after_first)
        assert r2.edges_created == 0

    async def test_edges_persist_after_reopen(self, tmp_path: Path) -> None:
        """Dream-created edges are committed and survive connection reopen."""
        db_path = tmp_path / "persist.db"
        db, store = await _open_store(db_path)
        try:
            t1 = await store.create_thought(
                _make("t-persist-1", essence="persistent A", content="content A"),
            )
            t2 = await store.create_thought(
                _make("t-persist-2", essence="persistent B", content="content B"),
            )
            await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
            await store.store_embedding(t2.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="test")

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.0,
                gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
                edges=EdgeCreationConfig(enabled=True, top_k=1, min_similarity=0.5),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=10)

            assert result.edges_created >= 1
        finally:
            await db.close()

        reopened_db, reopened_store = await _open_store(db_path)
        try:
            edges = await reopened_store.get_edges("t-persist-1", direction="BOTH")
            assert any(edge.edge_type == EdgeType.ASSOCIATED for edge in edges)
        finally:
            await reopened_db.close()


# ---------------------------------------------------------------------------
# TestGraphGrowth
# ---------------------------------------------------------------------------


class TestGraphGrowth:
    """Verifies dream actually grows the graph."""

    async def test_edge_count_grows_per_cycle(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """After dreaming, edge_count > initial (was 0)."""
        thoughts = []
        for i in range(5):
            t = await store.create_thought(
                _make(f"t-grow-{i}", essence=f"topic {i}", content=f"content {i}"),
            )
            vec = [0.0] * 4
            vec[i % 4] = 0.9
            vec[(i + 1) % 4] = 0.1
            await store.store_embedding(t.thought_id, vec, model_name="test")
            thoughts.append(t)

        cfg = DreamingConfig(
            enabled=True,
            promote_threshold=0.0,
            gates=DreamingGates(min_age_cycles=0, allow_zero_confirmation=True),
            edges=EdgeCreationConfig(enabled=True, top_k=1, min_similarity=0.1),
        )
        ext = DreamingExtension(config=cfg)
        result = await ext.run_consolidation(store, current_cycle=10)

        assert result.edges_created > 0
        assert result.promoted_count > 0


# ---------------------------------------------------------------------------
# TestGraphAwareSearch
# ---------------------------------------------------------------------------


class TestGraphAwareSearch:
    """Graph-aware 1-hop-weighted boost in search_hybrid."""

    async def test_1hop_weighted_boosts_connected_thought(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """A thought with a connected high-scoring neighbour ranks higher."""
        t_connected = await store.create_thought(
            _make("t-conn", essence="python async", content="async programming"),
        )
        t_isolated = await store.create_thought(
            _make("t-iso", essence="python async", content="async programming"),
        )
        t_neighbour = await store.create_thought(
            _make("t-neigh", essence="coroutine patterns", content="coroutine semantics"),
        )

        # All have similar embeddings
        await store.store_embedding(t_connected.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="t")
        await store.store_embedding(t_isolated.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="t")
        await store.store_embedding(t_neighbour.thought_id, [0.85, 0.15, 0.0, 0.0], model_name="t")

        # Create edge: t_connected ↔ t_neighbour
        edge = EdgeRecord(
            edge_id="e-test-1",
            from_thought_id=t_connected.thought_id,
            to_thought_id=t_neighbour.thought_id,
            edge_type=EdgeType.ASSOCIATED,
            weight=0.8,
            created_cycle=1,
            source=KnowledgeSource.DREAMING,
        )
        await store.create_edge(edge)

        result = await store.search_hybrid(
            "",
            [0.9, 0.1, 0.0, 0.0],
            graph_weight=0.3,
            priority_weight=0.0,
        )

        # t_connected should rank above t_isolated due to graph boost
        ids = [tid for tid, _score in result.results]
        assert ids.index("t-conn") < ids.index("t-iso")
        assert "graph" in result.backends_used

    async def test_graph_signal_disabled_matches_pre_p1_ordering(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """graph_weight=0.0 → identical ordering to pre-graph-signal baseline."""
        t1 = await store.create_thought(
            _make("t-gd-1", essence="first", content="content first"),
        )
        t2 = await store.create_thought(
            _make("t-gd-2", essence="second", content="content second"),
        )
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")
        await store.store_embedding(t2.thought_id, [0.5, 0.5, 0.0, 0.0], model_name="test")

        # Create edge that would boost t2 if graph was active
        edge = EdgeRecord(
            edge_id="e-gd-1",
            from_thought_id=t2.thought_id,
            to_thought_id=t1.thought_id,
            edge_type=EdgeType.ASSOCIATED,
            weight=0.9,
            created_cycle=1,
        )
        await store.create_edge(edge)

        result = await store.search_hybrid(
            "",
            [0.9, 0.1, 0.0, 0.0],
            graph_weight=0.0,
            priority_weight=0.0,
        )

        # t1 closer to query vector → should still be first
        assert result.results[0][0] == "t-gd-1"
        assert "graph" not in result.backends_used

    async def test_graph_weight_config_validation(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Negative graph_weight raises ValueError."""
        with pytest.raises(ValueError, match="graph_weight must be non-negative"):
            await store.search_hybrid(
                "test",
                None,
                graph_weight=-0.1,
                priority_weight=0.0,
            )

    async def test_graph_no_boost_when_no_edges(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Graph active but no edges → 'graph' not in backends_used."""
        t1 = await store.create_thought(
            _make("t-ne-1", essence="lonely", content="no edges here"),
        )
        await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0, 0.0], model_name="test")

        result = await store.search_hybrid(
            "",
            [0.9, 0.1, 0.0, 0.0],
            graph_weight=0.1,
            priority_weight=0.0,
        )

        assert "graph" not in result.backends_used


# ---------------------------------------------------------------------------
# TestEdgeCreationConfig
# ---------------------------------------------------------------------------


class TestEdgeCreationConfig:
    """Config parsing for edge creation."""

    def test_defaults(self) -> None:
        """Default EdgeCreationConfig values."""
        cfg = EdgeCreationConfig()
        assert cfg.enabled is True
        assert cfg.top_k == 1
        assert cfg.min_similarity == 0.7
        assert cfg.edge_weight_factor == 0.5

    def test_frozen(self) -> None:
        """EdgeCreationConfig is immutable."""
        cfg = EdgeCreationConfig()
        with pytest.raises(AttributeError):
            cfg.top_k = 5  # type: ignore[misc]

    def test_dreaming_config_includes_edges(self) -> None:
        """DreamingConfig has edges field with default."""
        cfg = DreamingConfig()
        assert cfg.edges == EdgeCreationConfig()

    def test_parse_edge_creation_valid(self) -> None:
        """Valid YAML edge section parses correctly."""
        raw = {
            "enabled": True,
            "top_k": 3,
            "min_similarity": 0.6,
            "edge_weight_factor": 0.8,
        }
        cfg = _parse_edge_creation(raw)
        assert cfg.top_k == 3
        assert cfg.min_similarity == 0.6
        assert cfg.edge_weight_factor == 0.8

    def test_parse_edge_creation_bad_type(self) -> None:
        """Non-dict raises ConfigError."""
        with pytest.raises(ConfigError):
            _parse_edge_creation("not a dict")

    def test_parse_edge_creation_bad_top_k(self) -> None:
        """top_k < 1 raises ConfigError."""
        with pytest.raises(ConfigError, match="top_k"):
            _parse_edge_creation({"top_k": 0})

    def test_parse_edge_creation_bad_similarity(self) -> None:
        """min_similarity > 1.0 raises ConfigError."""
        with pytest.raises(ConfigError, match="min_similarity"):
            _parse_edge_creation({"min_similarity": 1.5})

    def test_parse_dreaming_with_edges(self) -> None:
        """Dreaming section with edges block parses correctly."""
        raw = {
            "enabled": True,
            "edges": {
                "enabled": True,
                "top_k": 2,
                "min_similarity": 0.6,
            },
        }
        cfg = _parse_dreaming(raw)
        assert cfg is not None
        assert cfg.edges.top_k == 2
        assert cfg.edges.min_similarity == 0.6


# ---------------------------------------------------------------------------
# TestSearchConfigGraph
# ---------------------------------------------------------------------------


class TestSearchConfigGraph:
    """SearchConfig graph-related fields."""

    def test_graph_defaults(self) -> None:
        """Default SearchConfig has graph_weight=0.0 (disabled by default)."""
        cfg = SearchConfig()
        assert cfg.default_graph_weight == 0.0
        assert cfg.graph_edge_decay == 0.5
        assert cfg.max_neighbors_per_candidate == 5

    def test_parse_search_with_graph(self) -> None:
        """YAML search section with graph fields parses correctly."""
        raw = {
            "default_graph_weight": 0.1,
            "graph_edge_decay": 0.3,
            "max_neighbors_per_candidate": 10,
        }
        cfg = _parse_search(raw)
        assert cfg.default_graph_weight == 0.1
        assert cfg.graph_edge_decay == 0.3
        assert cfg.max_neighbors_per_candidate == 10

    def test_parse_search_bad_max_neighbors(self) -> None:
        """Non-positive max_neighbors_per_candidate raises ConfigError."""
        with pytest.raises(ConfigError, match="max_neighbors_per_candidate"):
            _parse_search({"max_neighbors_per_candidate": 0})

    def test_total_weights_sum_to_one(self) -> None:
        """Default weights sum to 1.0 (graph=0.0 disabled by default)."""
        cfg = SearchConfig()
        total = (
            cfg.default_fts_weight
            + cfg.default_vector_weight
            + cfg.default_recency_weight
            + cfg.default_priority_weight
            + cfg.default_graph_weight
        )
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestKnowledgeSourceDreaming
# ---------------------------------------------------------------------------


class TestKnowledgeSourceDreaming:
    """KnowledgeSource.DREAMING enum value."""

    def test_dreaming_value_exists(self) -> None:
        """DREAMING is a valid KnowledgeSource."""
        assert KnowledgeSource.DREAMING == "DREAMING"

    def test_dreaming_in_edge(self) -> None:
        """EdgeRecord accepts source=DREAMING."""
        edge = EdgeRecord(
            edge_id="e-ks-1",
            from_thought_id="t1",
            to_thought_id="t2",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=1,
            source=KnowledgeSource.DREAMING,
        )
        assert edge.source == KnowledgeSource.DREAMING

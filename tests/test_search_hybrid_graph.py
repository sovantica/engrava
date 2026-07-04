"""Tests for candidate expansion via CONSOLIDATED_FROM.

Covers:
- _expand_via_consolidated_from pulls source OBSs from top REFLECTIONs
- max_sources_per_reflection cap limits sources pulled per REFLECTION
- propagation_factor scales the propagated score correctly
- no-op when top-N candidates contain no REFLECTIONs
- interaction with existing _load_graph_signal (both active)
- giant-cluster-aware guard: skips REFLECTIONs above source_ceiling
- backends_used reports "graph_expansion" when expansion is active
- _parse_search wires new graph_expansion_* fields from YAML
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SearchConfig, SqliteEngravaCore
from engrava.config import ConfigError, _parse_search
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(
    thought_id: str,
    *,
    essence: str = "obs",
    content: str = "content",
    created_cycle: int = 0,
) -> ThoughtRecord:
    """Minimal OBSERVATION thought."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=created_cycle,
        updated_cycle=created_cycle,
        source="test",
    )


def _reflection(
    thought_id: str,
    *,
    essence: str = "reflection",
    content: str = "summary",
    created_cycle: int = 0,
) -> ThoughtRecord:
    """Minimal REFLECTION thought."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.REFLECTION,
        essence=essence,
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=created_cycle,
        updated_cycle=created_cycle,
        source="test",
    )


def _edge(
    from_id: str,
    to_id: str,
    *,
    edge_type: EdgeType = EdgeType.CONSOLIDATED_FROM,
    weight: float = 1.0,
) -> EdgeRecord:
    """Minimal edge."""
    return EdgeRecord(
        edge_id=str(uuid.uuid4()),
        from_thought_id=from_id,
        to_thought_id=to_id,
        edge_type=edge_type,
        weight=weight,
        created_cycle=0,
        source=KnowledgeSource.DREAMING,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh in-file store (avoids :memory: WAL conflicts)."""
    conn = await aiosqlite.connect(str(tmp_path / "test.db"))
    conn.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(conn)
    await s.ensure_schema()
    yield s
    await conn.close()


@pytest.fixture
async def store_with_cfg(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Store with explicit SearchConfig (expansion enabled, defaults)."""
    conn = await aiosqlite.connect(str(tmp_path / "test_cfg.db"))
    conn.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(conn, search_config=SearchConfig())
    await s.ensure_schema()
    yield s
    await conn.close()


# ---------------------------------------------------------------------------
# _expand_via_consolidated_from — unit tests
# ---------------------------------------------------------------------------


class TestExpandViaConsolidatedFrom:
    """Direct tests of SqliteEngravaCore._expand_via_consolidated_from."""

    async def test_expansion_pulls_source_obs(self, store: SqliteEngravaCore) -> None:
        """REFLECTION in combined pulls its source OBSs even when they are absent.

        Setup: REFLECTION r1 → sources [o1, o2, o3] via CONSOLIDATED_FROM.
        r1 is in combined with score 0.8; o1/o2/o3 are NOT in combined.
        After expansion all three OBSs must appear with propagated scores.
        """
        r1 = await store.create_thought(_reflection("r1", essence="summary"))
        o1 = await store.create_thought(_obs("o1", essence="fact one"))
        o2 = await store.create_thought(_obs("o2", essence="fact two"))
        o3 = await store.create_thought(_obs("o3", essence="fact three"))

        for obs_id in (o1.thought_id, o2.thought_id, o3.thought_id):
            await store.create_edge(_edge(r1.thought_id, obs_id, weight=0.9))

        combined: dict[str, float] = {r1.thought_id: 0.8}
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )

        for obs_id in (o1.thought_id, o2.thought_id, o3.thought_id):
            assert obs_id in combined, f"{obs_id} not expanded into combined"
            assert combined[obs_id] > 0.0

    async def test_expansion_respects_max_per_reflection(self, store: SqliteEngravaCore) -> None:
        """Only the top-N sources by edge weight are pulled.

        REFLECTION r1 has 10 source OBSs. With max_sources=3 only the
        3 highest-weight edges must produce entries in combined.
        """
        r1 = await store.create_thought(_reflection("r1-max"))
        obs_ids: list[str] = []
        for i in range(10):
            obs = await store.create_thought(_obs(f"o-max-{i}"))
            obs_ids.append(obs.thought_id)
            # Weight decreases with index so first 3 are highest.
            await store.create_edge(_edge(r1.thought_id, obs.thought_id, weight=1.0 - i * 0.05))

        combined: dict[str, float] = {r1.thought_id: 1.0}
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=3,
            reflection_source_ceiling=50,
        )

        expanded = [tid for tid in obs_ids if tid in combined]
        assert len(expanded) == 3
        # The 3 highest-weight (first 3 by insertion, largest weight) must be included.
        for i in range(3):
            assert obs_ids[i] in combined

    async def test_expansion_respects_propagation_factor(self, store: SqliteEngravaCore) -> None:
        """Propagated score = parent_score * propagation_factor * edge_weight."""
        r1 = await store.create_thought(_reflection("r1-factor"))
        o1 = await store.create_thought(_obs("o1-factor"))
        await store.create_edge(_edge(r1.thought_id, o1.thought_id, weight=0.8))

        parent_score = 1.0
        factor = 0.5
        combined: dict[str, float] = {r1.thought_id: parent_score}
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=factor,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )

        expected = parent_score * factor * 0.8
        assert abs(combined[o1.thought_id] - expected) < 1e-9

    async def test_expansion_noop_no_reflections(self, store: SqliteEngravaCore) -> None:
        """combined unchanged when top-N contains no REFLECTIONs."""
        o1 = await store.create_thought(_obs("o-noop-1"))
        o2 = await store.create_thought(_obs("o-noop-2"))

        combined: dict[str, float] = {
            o1.thought_id: 0.9,
            o2.thought_id: 0.7,
        }
        original_keys = set(combined.keys())
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )

        assert set(combined.keys()) == original_keys

    async def test_giant_cluster_guard_skips_large_reflection(
        self, store: SqliteEngravaCore
    ) -> None:
        """REFLECTIONs with sources > ceiling are not expanded.

        REFLECTION r_big has 10 sources; ceiling=5 → skipped.
        REFLECTION r_small has 3 sources; ceiling=5 → expanded.
        Only r_small's sources appear after expansion.
        """
        r_big = await store.create_thought(_reflection("r-big"))
        r_small = await store.create_thought(_reflection("r-small"))

        big_obs: list[str] = []
        for i in range(10):
            o = await store.create_thought(_obs(f"o-big-{i}"))
            big_obs.append(o.thought_id)
            await store.create_edge(_edge(r_big.thought_id, o.thought_id))

        small_obs: list[str] = []
        for i in range(3):
            o = await store.create_thought(_obs(f"o-small-{i}"))
            small_obs.append(o.thought_id)
            await store.create_edge(_edge(r_small.thought_id, o.thought_id))

        combined: dict[str, float] = {
            r_big.thought_id: 0.95,
            r_small.thought_id: 0.80,
        }
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=5,
        )

        # big reflection's sources must NOT appear
        for tid in big_obs:
            assert tid not in combined, f"{tid} (big cluster) should have been guarded"
        # small reflection's sources MUST appear
        for tid in small_obs:
            assert tid in combined, f"{tid} (small cluster) should have been expanded"

    async def test_existing_score_preserved_when_higher(self, store: SqliteEngravaCore) -> None:
        """When source OBS is already in combined with a higher score, it is kept."""
        r1 = await store.create_thought(_reflection("r1-keep"))
        o1 = await store.create_thought(_obs("o1-keep"))
        await store.create_edge(_edge(r1.thought_id, o1.thought_id, weight=0.5))

        # OBS already has a high score; propagated = 0.6 * 0.7 * 0.5 = 0.21 < 0.9
        combined: dict[str, float] = {r1.thought_id: 0.6, o1.thought_id: 0.9}
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )

        assert combined[o1.thought_id] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# search_hybrid integration — expansion via backends_used + end-to-end
# ---------------------------------------------------------------------------


class TestSearchHybridExpansion:
    """Integration tests for candidate expansion wired into search_hybrid."""

    async def test_backends_used_contains_graph_expansion(
        self, store_with_cfg: SqliteEngravaCore
    ) -> None:
        """search_hybrid reports 'graph_expansion' in backends_used when active."""
        await store_with_cfg.create_thought(_obs("t-bu-1", essence="alpha beta"))
        # REFLECTION must match the query to land in FTS combined scores.
        r1 = await store_with_cfg.create_thought(_reflection("r-bu-1", essence="alpha summary"))
        o1 = await store_with_cfg.create_thought(_obs("o-bu-1", essence="fact"))
        await store_with_cfg.create_edge(_edge(r1.thought_id, o1.thought_id))

        result = await store_with_cfg.search_hybrid(
            query_text="alpha",
            top_k=10,
        )
        assert "graph_expansion" in result.backends_used

    async def test_expansion_disabled_via_config(self, tmp_path: Path) -> None:
        """graph_expansion_enabled=False removes 'graph_expansion' from backends_used."""
        conn = await aiosqlite.connect(str(tmp_path / "disabled.db"))
        conn.row_factory = aiosqlite.Row
        cfg = SearchConfig(graph_expansion_enabled=False)
        s = SqliteEngravaCore(conn, search_config=cfg)
        await s.ensure_schema()

        await s.create_thought(_obs("t-dis-1", essence="topic"))

        result = await s.search_hybrid(query_text="topic", top_k=5)
        assert "graph_expansion" not in result.backends_used
        await conn.close()

    async def test_expansion_combines_with_graph_signal(self, tmp_path: Path) -> None:
        """Both graph signal and expansion are active and do not conflict.

        graph_weight > 0 activates _load_graph_signal; expansion adds
        sources. Both should appear in backends_used.
        """
        conn = await aiosqlite.connect(str(tmp_path / "both.db"))
        conn.row_factory = aiosqlite.Row
        cfg = SearchConfig(default_graph_weight=0.1, graph_expansion_enabled=True)
        s = SqliteEngravaCore(conn, search_config=cfg)
        await s.ensure_schema()

        r1 = await s.create_thought(_reflection("r-both", essence="central summary"))
        o1 = await s.create_thought(_obs("o-both-1", essence="detail a"))
        o2 = await s.create_thought(_obs("o-both-2", essence="detail b"))
        # CONSOLIDATED_FROM: reflection → sources
        await s.create_edge(_edge(r1.thought_id, o1.thought_id))
        await s.create_edge(_edge(r1.thought_id, o2.thought_id))
        # ASSOCIATED: so graph signal also fires
        await s.create_edge(_edge(r1.thought_id, o1.thought_id, edge_type=EdgeType.ASSOCIATED))

        result = await s.search_hybrid(
            query_text="central",
            top_k=20,
        )
        # Both backends should be recorded
        assert "graph_expansion" in result.backends_used
        await conn.close()


# ---------------------------------------------------------------------------
# _parse_search — new graph_expansion_* fields
# ---------------------------------------------------------------------------


class TestParseSearchGraphExpansion:
    """Config parsing tests for the new graph_expansion_* fields."""

    def test_defaults_match_searchconfig(self) -> None:
        """Parsing None returns SearchConfig with correct expansion defaults."""
        cfg = _parse_search(None)
        assert cfg.graph_expansion_enabled is True
        assert cfg.graph_expansion_top_n == 5
        assert cfg.graph_expansion_propagation_factor == pytest.approx(0.7)
        assert cfg.graph_expansion_max_sources_per_reflection == 20
        assert cfg.graph_expansion_reflection_source_ceiling == 50

    def test_parse_explicit_values(self) -> None:
        """Explicit YAML values are wired correctly."""
        raw = {
            "graph_expansion_enabled": False,
            "graph_expansion_top_n": 10,
            "graph_expansion_propagation_factor": 0.5,
            "graph_expansion_max_sources_per_reflection": 30,
            "graph_expansion_reflection_source_ceiling": 100,
        }
        cfg = _parse_search(raw)
        assert cfg.graph_expansion_enabled is False
        assert cfg.graph_expansion_top_n == 10
        assert cfg.graph_expansion_propagation_factor == pytest.approx(0.5)
        assert cfg.graph_expansion_max_sources_per_reflection == 30
        assert cfg.graph_expansion_reflection_source_ceiling == 100

    def test_invalid_top_n_raises(self) -> None:
        """Non-positive graph_expansion_top_n raises ConfigError."""
        with pytest.raises(ConfigError):
            _parse_search({"graph_expansion_top_n": 0})

    def test_invalid_max_sources_raises(self) -> None:
        """Non-positive graph_expansion_max_sources_per_reflection raises ConfigError."""
        with pytest.raises(ConfigError):
            _parse_search({"graph_expansion_max_sources_per_reflection": -1})

    def test_invalid_ceiling_raises(self) -> None:
        """Non-positive graph_expansion_reflection_source_ceiling raises ConfigError."""
        with pytest.raises(ConfigError):
            _parse_search({"graph_expansion_reflection_source_ceiling": 0})

    def test_invalid_enabled_type_raises(self) -> None:
        """Non-boolean graph_expansion_enabled raises ConfigError."""
        with pytest.raises(ConfigError):
            _parse_search({"graph_expansion_enabled": "yes"})

    def test_collapse_pool_factor_default(self) -> None:
        """collapse_pool_factor defaults to 4 when unspecified."""
        assert _parse_search(None).collapse_pool_factor == 4

    def test_collapse_pool_factor_explicit(self) -> None:
        """An explicit collapse_pool_factor is wired through."""
        assert _parse_search({"collapse_pool_factor": 6}).collapse_pool_factor == 6

    def test_invalid_collapse_pool_factor_raises(self) -> None:
        """A non-positive collapse_pool_factor raises ConfigError."""
        with pytest.raises(ConfigError, match=r"collapse_pool_factor.*positive integer"):
            _parse_search({"collapse_pool_factor": 0})


# ---------------------------------------------------------------------------
# Regression tests for the 4 gaps found in review (2026-04-23)
# ---------------------------------------------------------------------------


class TestExpansionGapFixes:
    """Tests that validate each of the 4 review gaps are now correctly fixed."""

    # --- Gap 1: top-N REFLECTION ranking must respect combined score ---

    async def test_top_n_respects_score_order(self, store: SqliteEngravaCore) -> None:
        """The expansion seeds must be the highest-scored REFLECTIONs, not SQL-order ones.

        Setup:
          - r_low  (REFLECTION, score 0.3) — added to DB first
          - r_high (REFLECTION, score 0.9) — added to DB second
          - obs_low  is a source of r_low  only
          - obs_high is a source of r_high only
        With expansion_top_n=1, only the highest-scored REFLECTION (r_high)
        must be used as a seed.  obs_high must appear; obs_low must NOT.
        """
        r_low = await store.create_thought(_reflection("r-order-low", essence="low"))
        r_high = await store.create_thought(_reflection("r-order-high", essence="high"))
        obs_low = await store.create_thought(_obs("obs-order-low"))
        obs_high = await store.create_thought(_obs("obs-order-high"))

        await store.create_edge(_edge(r_low.thought_id, obs_low.thought_id, weight=0.9))
        await store.create_edge(_edge(r_high.thought_id, obs_high.thought_id, weight=0.9))

        # r_low was inserted first, so it would win under raw SQL IN (...) order.
        # r_high has a higher score — it must win under correct score-ranked logic.
        combined: dict[str, float] = {
            r_low.thought_id: 0.3,
            r_high.thought_id: 0.9,
        }
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=1,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )

        assert obs_high.thought_id in combined, "High-scored REFLECTION seed was not expanded"
        assert obs_low.thought_id not in combined, (
            "Low-scored REFLECTION should NOT have been seeded"
        )

    # --- Gap 2: only OBSERVATION targets must be pulled into combined ---

    async def test_non_observation_targets_not_expanded(self, store: SqliteEngravaCore) -> None:
        """CONSOLIDATED_FROM targets that are not OBSERVATION are skipped.

        A REFLECTION can have CONSOLIDATED_FROM edges to TASK or other
        REFLECTION thoughts (legacy / edge case).  Only OBSERVATION-type
        targets must be added to combined.
        """
        r_parent = await store.create_thought(_reflection("r-types", essence="parent"))
        obs_child = await store.create_thought(_obs("obs-types", essence="fact"))
        task_child = await store.create_thought(
            ThoughtRecord(
                thought_id="task-types",
                thought_type=ThoughtType.TASK,
                essence="a task",
                content="do something",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=0,
                updated_cycle=0,
                source="test",
            )
        )
        refl_child = await store.create_thought(_reflection("refl-types-child", essence="meta"))

        await store.create_edge(_edge(r_parent.thought_id, obs_child.thought_id, weight=0.9))
        await store.create_edge(_edge(r_parent.thought_id, task_child.thought_id, weight=0.8))
        await store.create_edge(_edge(r_parent.thought_id, refl_child.thought_id, weight=0.7))

        combined: dict[str, float] = {r_parent.thought_id: 1.0}
        await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )

        assert obs_child.thought_id in combined, "OBSERVATION child must be expanded"
        assert task_child.thought_id not in combined, "TASK child must NOT be expanded"
        assert refl_child.thought_id not in combined, "REFLECTION child must NOT be expanded"

    # --- Gap 3: backends_used is false-positive when no expansion occurs ---

    async def test_backends_used_absent_when_no_obs_expanded(self, tmp_path: Path) -> None:
        """'graph_expansion' must NOT appear in backends_used when no OBS was added.

        Case: there are REFLECTIONs but none has CONSOLIDATED_FROM edges.
        Expansion runs, finds no healthy sources, adds nothing → backend
        must NOT be reported.
        """
        conn = await aiosqlite.connect(str(tmp_path / "noop_bu.db"))
        conn.row_factory = aiosqlite.Row
        s = SqliteEngravaCore(conn, search_config=SearchConfig())
        await s.ensure_schema()

        # Add a REFLECTION with no edges — nothing to expand.
        await s.create_thought(_reflection("r-noop-bu", essence="lonely summary"))
        await s.create_thought(_obs("o-noop-bu", essence="isolated fact"))

        result = await s.search_hybrid(query_text="lonely", top_k=5)
        assert "graph_expansion" not in result.backends_used
        await conn.close()

    async def test_backends_used_present_only_when_expansion_fires(
        self, store: SqliteEngravaCore
    ) -> None:
        """_expand_via_consolidated_from returns 0 → no backend entry; > 0 → entry added."""
        r1 = await store.create_thought(_reflection("r-bu-fire"))
        o1 = await store.create_thought(_obs("o-bu-fire"))
        await store.create_edge(_edge(r1.thought_id, o1.thought_id))

        # Empty combined — should be 0 (early exit guard).
        added_empty = await store._expand_via_consolidated_from(
            combined={},
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )
        assert added_empty == 0

        # Combined with REFLECTION seed — should return > 0.
        combined: dict[str, float] = {r1.thought_id: 0.9}
        added = await store._expand_via_consolidated_from(
            combined=combined,
            expansion_top_n=5,
            propagation_factor=0.7,
            max_sources_per_reflection=20,
            reflection_source_ceiling=50,
        )
        assert added > 0

    # --- Gap 4: schema migration v7 → v8 creates the edge index ---

    async def test_schema_v8_creates_edge_index(self, tmp_path: Path) -> None:
        """ensure_schema on a fresh DB produces idx_edge_type_from in sqlite_master."""
        conn = await aiosqlite.connect(str(tmp_path / "idx_test.db"))
        conn.row_factory = aiosqlite.Row
        s = SqliteEngravaCore(conn)
        await s.ensure_schema()

        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_edge_type_from'"
        )
        row = await cursor.fetchone()
        await conn.close()

        assert row is not None, "idx_edge_type_from not found in sqlite_master after ensure_schema"

    async def test_schema_migration_v7_to_head(self, tmp_path: Path) -> None:
        """ensure_schema on an existing v7 DB cascades to head and adds the index."""
        conn = await aiosqlite.connect(str(tmp_path / "migrate_test.db"))
        conn.row_factory = aiosqlite.Row
        # Bootstrap as v7 (no expansion index).
        s = SqliteEngravaCore(conn)
        await s.ensure_schema()
        # Force-downgrade user_version to simulate an existing v7 database.
        await conn.execute("PRAGMA user_version = 7")
        # Drop the index to simulate v7 state.
        await conn.execute("DROP INDEX IF EXISTS idx_edge_type_from")
        await conn.commit()

        # Re-run ensure_schema — cascades through v7->head and recreates the index.
        s2 = SqliteEngravaCore(conn)
        await s2.ensure_schema()

        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_edge_type_from'"
        )
        row = await cursor.fetchone()
        cursor2 = await conn.execute("PRAGMA user_version")
        version_row = await cursor2.fetchone()
        await conn.close()

        assert row is not None, "idx_edge_type_from missing after v7->head migration"
        assert int(version_row[0]) == 18

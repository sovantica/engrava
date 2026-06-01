"""Tests for reflection_topk_cap + SearchConfig default retune.

Covers:
- SearchConfig defaults: reflection_boost=1.0, default_graph_weight=0.0, reflection_topk_cap=0.3
- Cap enforcement: excess REFLECTIONs evicted, replaced by off-list non-REFLECTIONs
- No-op when REFLECTION count is within cap
- Cap=1.0 disables enforcement entirely
- Warning emitted when off-list non-REFLECTIONs are insufficient to fill evicted slots
- _parse_search wires reflection_topk_cap from YAML
- Zero REFLECTION-only top-K: OBSERVATIONs always present when cap < 1.0 and sources exist
"""

from __future__ import annotations

import logging
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
    """Store with explicit SearchConfig using test-friendly defaults."""
    conn = await aiosqlite.connect(str(tmp_path / "cap_test.db"))
    conn.row_factory = aiosqlite.Row
    # Disable graph expansion to isolate cap logic; graph signal off.
    cfg = SearchConfig(
        graph_expansion_enabled=False,
        default_graph_weight=0.0,
        reflection_topk_cap=0.3,
    )
    s = SqliteEngravaCore(conn, search_config=cfg)
    await s.ensure_schema()
    yield s
    await conn.close()


@pytest.fixture
async def store_cap1(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Store with reflection_topk_cap=1.0 (cap disabled)."""
    conn = await aiosqlite.connect(str(tmp_path / "cap1_test.db"))
    conn.row_factory = aiosqlite.Row
    cfg = SearchConfig(
        graph_expansion_enabled=False,
        default_graph_weight=0.0,
        reflection_topk_cap=1.0,
    )
    s = SqliteEngravaCore(conn, search_config=cfg)
    await s.ensure_schema()
    yield s
    await conn.close()


# ---------------------------------------------------------------------------
# SearchConfig defaults
# ---------------------------------------------------------------------------


class TestSearchConfigDefaults:
    """Verify production SearchConfig defaults."""

    def test_reflection_boost_default_is_neutral(self) -> None:
        """reflection_boost default is 1.0 (neutral — no passive multiplier)."""
        assert SearchConfig().reflection_boost == pytest.approx(1.0)

    def test_default_graph_weight_disabled(self) -> None:
        """default_graph_weight default is 0.0 (graph signal off)."""
        assert SearchConfig().default_graph_weight == pytest.approx(0.0)

    def test_reflection_topk_cap_default(self) -> None:
        """reflection_topk_cap default is 0.3."""
        assert SearchConfig().reflection_topk_cap == pytest.approx(0.3)

    def test_parse_search_none_matches_defaults(self) -> None:
        """_parse_search(None) returns SearchConfig with new defaults."""
        cfg = _parse_search(None)
        assert cfg.reflection_boost == pytest.approx(1.0)
        assert cfg.default_graph_weight == pytest.approx(0.0)
        assert cfg.reflection_topk_cap == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# _parse_search — reflection_topk_cap YAML wiring
# ---------------------------------------------------------------------------


class TestParseSearchReflectionCap:
    """Config parsing tests for the new reflection_topk_cap field."""

    def test_parse_explicit_cap(self) -> None:
        """Explicit reflection_topk_cap is wired through."""
        cfg = _parse_search({"reflection_topk_cap": 0.5})
        assert cfg.reflection_topk_cap == pytest.approx(0.5)

    def test_parse_cap_zero(self) -> None:
        """reflection_topk_cap=0.0 means zero REFLECTION slots allowed."""
        cfg = _parse_search({"reflection_topk_cap": 0.0})
        assert cfg.reflection_topk_cap == pytest.approx(0.0)

    def test_parse_cap_one(self) -> None:
        """reflection_topk_cap=1.0 disables cap (allow all)."""
        cfg = _parse_search({"reflection_topk_cap": 1.0})
        assert cfg.reflection_topk_cap == pytest.approx(1.0)

    def test_parse_negative_cap_raises(self) -> None:
        """Negative reflection_topk_cap raises ConfigError."""
        with pytest.raises(ConfigError):
            _parse_search({"reflection_topk_cap": -0.1})


# ---------------------------------------------------------------------------
# TestReflectionCapInvariant — integration tests via search_hybrid
# ---------------------------------------------------------------------------


class TestReflectionCapInvariant:
    """Invariant: REFLECTIONs may not occupy more than cap * top_k result slots."""

    async def _populate_flood(
        self,
        store: SqliteEngravaCore,
        *,
        n_reflections: int,
        n_obs: int,
        reflection_score_base: float = 0.9,
        obs_score_base: float = 0.5,
    ) -> tuple[list[str], list[str]]:
        """Create N REFLECTIONs with FTS-visible essence and N OBSs.

        Returns (reflection_ids, obs_ids).
        """
        reflection_ids: list[str] = []
        for i in range(n_reflections):
            r = await store.create_thought(
                _reflection(
                    str(uuid.uuid4()),
                    essence=f"summary reflection topic alpha {i}",
                    content=f"reflection content {i}",
                )
            )
            reflection_ids.append(r.thought_id)

        obs_ids: list[str] = []
        for i in range(n_obs):
            o = await store.create_thought(
                _obs(
                    str(uuid.uuid4()),
                    essence=f"observation topic alpha {i}",
                    content=f"observation content {i}",
                )
            )
            obs_ids.append(o.thought_id)

        _ = reflection_score_base, obs_score_base  # scores set via FTS ranking
        return reflection_ids, obs_ids

    async def test_cap_enforced_evicts_excess_reflections(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """With cap=0.3 and top_k=10, at most 3 REFLECTIONs may appear in results.

        Setup: 8 REFLECTIONs and 4 OBSs all matching the query.
        Without cap the top-10 would be REFLECTION-heavy.
        With cap=0.3 the final list must contain ≤ 3 REFLECTIONs.
        """
        reflection_ids, _obs_ids = await self._populate_flood(
            store,
            n_reflections=8,
            n_obs=8,
        )

        result = await store.search_hybrid(
            query_text="topic alpha",
            top_k=10,
        )

        result_ids = {tid for tid, _ in result.results}
        refs_in_result = [tid for tid in result_ids if tid in set(reflection_ids)]
        # cap=0.3 * top_k=10 → max 3 slots
        assert len(refs_in_result) <= 3, (
            f"Expected ≤3 REFLECTIONs in top-10 with cap=0.3, got {len(refs_in_result)}"
        )

    async def test_cap_not_triggered_when_reflections_within_limit(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """When REFLECTION count ≤ cap * top_k, results are unchanged.

        Setup: 2 REFLECTIONs and 8 OBSs; top_k=10, cap=0.3 → 3 slots allowed.
        2 REFLECTIONs < 3 allowed → no eviction; both REFLECTIONs remain.
        """
        reflection_ids, _obs_ids = await self._populate_flood(
            store,
            n_reflections=2,
            n_obs=8,
        )

        result = await store.search_hybrid(
            query_text="topic alpha",
            top_k=10,
        )

        result_ids = {tid for tid, _ in result.results}
        refs_in_result = [tid for tid in result_ids if tid in set(reflection_ids)]
        # Both REFLECTIONs should still be present (under cap)
        assert len(refs_in_result) == 2

    async def test_cap_one_disables_enforcement(
        self,
        store_cap1: SqliteEngravaCore,
    ) -> None:
        """cap=1.0 allows all REFLECTIONs; no eviction occurs."""
        reflection_ids, _ = await self._populate_flood(
            store_cap1,
            n_reflections=8,
            n_obs=2,
        )

        result = await store_cap1.search_hybrid(
            query_text="topic alpha",
            top_k=10,
        )

        result_ids = {tid for tid, _ in result.results}
        refs_in_result = [tid for tid in result_ids if tid in set(reflection_ids)]
        # No cap → REFLECTIONs are not artificially limited
        assert len(refs_in_result) > 3, "cap=1.0 should allow more than 3 REFLECTIONs in top-10"

    async def test_cap_respected_result_length(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Final result list does not exceed top_k even after cap enforcement."""
        await self._populate_flood(store, n_reflections=8, n_obs=8)

        result = await store.search_hybrid(
            query_text="topic alpha",
            top_k=10,
        )

        assert len(result.results) <= 10

    async def test_warning_when_insufficient_obs_for_fill(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning is logged when off-list OBSs are fewer than excess REFLECTIONs.

        Setup: 8 REFLECTIONs and 0 OBSs; top_k=10, cap=0.3.
        No OBSs available to fill evicted slots → partial enforcement warning.
        """
        conn = await aiosqlite.connect(str(tmp_path / "warn_test.db"))
        conn.row_factory = aiosqlite.Row
        cfg = SearchConfig(
            graph_expansion_enabled=False,
            default_graph_weight=0.0,
            reflection_topk_cap=0.3,
        )
        s = SqliteEngravaCore(conn, search_config=cfg)
        await s.ensure_schema()

        for i in range(8):
            await s.create_thought(
                _reflection(
                    str(uuid.uuid4()),
                    essence=f"only reflections topic beta {i}",
                )
            )

        with caplog.at_level(logging.WARNING):
            await s.search_hybrid(query_text="topic beta", top_k=10)

        await conn.close()

        assert any(
            "reflection_topk_cap" in record.message and "partial enforcement" in record.message
            for record in caplog.records
        ), "Expected partial enforcement warning when no off-list OBSs available"

    async def test_zero_reflection_only_top_k_when_obs_present(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """With cap=0.3, at least one OBS appears in top-5 result when OBSs exist.

        This validates the 'zero REFLECTION-only top-20' acceptance criterion
        when scaled down to top-5.
        """
        _, obs_ids = await self._populate_flood(
            store,
            n_reflections=8,
            n_obs=5,
        )

        result = await store.search_hybrid(
            query_text="topic alpha",
            top_k=5,
        )

        result_ids = {tid for tid, _ in result.results}
        obs_in_result = [tid for tid in result_ids if tid in set(obs_ids)]
        assert len(obs_in_result) >= 1, (
            "Expected at least one OBSERVATION in top-5 with cap=0.3 and 5 OBSs in store"
        )

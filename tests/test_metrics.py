"""Tests for the metrics snapshot API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from engrava import (
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    MetricsConfig,
    Priority,
    SearchConfig,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.config import _parse_config
from engrava.domain.models.metrics import EngravaMetrics
from engrava.infrastructure.sqlite.engrava_core import _LatencyRingBuffer

if TYPE_CHECKING:
    from pathlib import Path


def _thought(
    thought_id: str,
    *,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    essence: str = "alpha",
    content: str = "alpha content",
    status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=Priority.P2,
        lifecycle_status=status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


@pytest.fixture
async def metrics_store(tmp_path: Path) -> SqliteEngravaCore:
    db_path = tmp_path / "metrics.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(
        conn,
        search_config=SearchConfig(default_graph_weight=0.1),
        metrics_config=MetricsConfig(window_size=100),
    )
    await store.ensure_schema()
    try:
        yield store
    finally:
        await conn.close()


class TestLatencyRingBuffer:
    async def test_percentiles_sane_for_uniform_range(self) -> None:
        buf = _LatencyRingBuffer(window_size=1000)
        for sample in range(1, 101):
            await buf.record(float(sample))

        snapshot = await buf.snapshot()
        assert snapshot.sample_count == 100
        assert 45.0 <= snapshot.p50_ms <= 55.0
        assert 92.0 <= snapshot.p95_ms <= 98.0
        assert 96.0 <= snapshot.p99_ms <= 100.0
        assert snapshot.min_ms == 1.0
        assert snapshot.max_ms == 100.0


class TestMetricsSnapshot:
    async def test_defaults_schema(self) -> None:
        snapshot = EngravaMetrics()
        assert snapshot.schema_version == 1
        assert snapshot.thoughts.total == 0
        assert snapshot.edges.total == 0
        assert snapshot.search_latency.sample_count == 0

    async def test_parse_metrics_config(self) -> None:
        config = _parse_config(
            {
                "database": {"path": "./test.db"},
                "metrics": {"window_size": 321, "enabled": False},
            }
        )
        assert config.metrics.window_size == 321
        assert not config.metrics.enabled

    async def test_metrics_counts_and_storage(self, metrics_store: SqliteEngravaCore) -> None:
        await metrics_store.create_thought(_thought("t-1", essence="alpha one"))
        await metrics_store.create_thought(_thought("t-2", essence="alpha two"))
        await metrics_store.create_thought(
            _thought(
                "r-1",
                thought_type=ThoughtType.REFLECTION,
                essence="alpha reflection",
                content="cluster summary",
            )
        )
        await metrics_store.create_edge(
            EdgeRecord(
                edge_id="e-1",
                from_thought_id="r-1",
                to_thought_id="t-1",
                edge_type=EdgeType.CONSOLIDATED_FROM,
                weight=1.0,
                created_cycle=0,
                source=KnowledgeSource.DREAMING,
            )
        )

        snapshot = await metrics_store.metrics()
        assert snapshot.thoughts.total == 3
        assert snapshot.thoughts.by_type["OBSERVATION"] == 2
        assert snapshot.thoughts.by_type["REFLECTION"] == 1
        assert snapshot.edges.total == 1
        assert snapshot.edges.by_type["CONSOLIDATED_FROM"] == 1
        assert snapshot.storage.db_bytes > 0
        assert snapshot.storage.total_bytes >= snapshot.storage.db_bytes

    async def test_public_search_methods_record_latency(
        self,
        metrics_store: SqliteEngravaCore,
    ) -> None:
        await metrics_store.create_thought(_thought("t-search", essence="alpha term"))
        await metrics_store.create_thought(
            _thought(
                "r-search",
                thought_type=ThoughtType.REFLECTION,
                essence="alpha summary",
                content="alpha cluster",
            )
        )
        await metrics_store.create_edge(
            EdgeRecord(
                edge_id="e-search",
                from_thought_id="r-search",
                to_thought_id="t-search",
                edge_type=EdgeType.CONSOLIDATED_FROM,
                weight=1.0,
                created_cycle=0,
                source=KnowledgeSource.DREAMING,
            )
        )
        await metrics_store.store_embedding("t-search", [1.0, 0.0, 0.0, 0.0], model_name="test")
        await metrics_store.store_embedding("r-search", [1.0, 0.0, 0.0, 0.0], model_name="test")

        await metrics_store.search_fts("alpha")
        await metrics_store.search_similar([1.0, 0.0, 0.0, 0.0])
        await metrics_store.search_hybrid("alpha", [1.0, 0.0, 0.0, 0.0])
        await metrics_store.search_reflections_only("alpha", [1.0, 0.0, 0.0, 0.0])

        snapshot = await metrics_store.metrics()
        assert snapshot.search_latency.sample_count == 4
        assert snapshot.search_latency.p50_ms >= 0.0
        assert snapshot.search_latency.p99_ms >= snapshot.search_latency.p50_ms

    async def test_metrics_opt_out_returns_zero_without_queries(self) -> None:
        mock_conn = AsyncMock()
        store = SqliteEngravaCore(mock_conn, metrics_config=MetricsConfig(enabled=False))

        snapshot = await store.metrics()
        assert snapshot.thoughts.total == 0
        assert snapshot.edges.total == 0
        assert snapshot.search_latency.sample_count == 0
        mock_conn.execute.assert_not_called()

    async def test_metrics_concurrent_safe(self, metrics_store: SqliteEngravaCore) -> None:
        await metrics_store.create_thought(_thought("t-conc", essence="alpha concurrent"))
        await metrics_store.store_embedding("t-conc", [1.0, 0.0, 0.0, 0.0], model_name="test")

        tasks = [metrics_store.search_hybrid("alpha", [1.0, 0.0, 0.0, 0.0]) for _ in range(10)]
        tasks.append(metrics_store.metrics())
        results = await asyncio.gather(*tasks)

        concurrent_snapshot = results[-1]
        assert isinstance(concurrent_snapshot, EngravaMetrics)

        final_snapshot = await metrics_store.metrics()
        assert final_snapshot.search_latency.sample_count == 10
        assert final_snapshot.search_latency.p95_ms >= 0.0


async def test_snapshot_perf(metrics_store: SqliteEngravaCore) -> None:
    import time

    await metrics_store.create_thought(_thought("t-perf"))

    start = time.perf_counter()
    await metrics_store.metrics()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 10.0

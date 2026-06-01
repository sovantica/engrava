"""Regression tests for graph_weight=0.0 default.

Covers:
- SearchConfig.default_graph_weight default is 0.0 (graph signal disabled)
- _parse_search(None) fallback uses 0.0
- graph_weight=0.0 produces no graph-signal contribution in search results
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SearchConfig, SqliteEngravaCore
from engrava.config import _parse_search
from engrava.domain.enums import (
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models.thought import ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(thought_id: str, *, essence: str = "obs", content: str = "content") -> ThoughtRecord:
    """Minimal OBSERVATION thought."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """In-memory store with graph_weight disabled (default)."""
    conn = await aiosqlite.connect(str(tmp_path / "test.db"))
    conn.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(conn, search_config=SearchConfig())
    await s.ensure_schema()
    yield s
    await conn.close()


# ---------------------------------------------------------------------------
# Unit — SearchConfig default
# ---------------------------------------------------------------------------


class TestGraphWeightDefault:
    """SearchConfig.default_graph_weight must default to 0.0."""

    def test_default_graph_weight_is_zero(self) -> None:
        """default_graph_weight default is 0.0 — graph signal disabled.

        Empirical evidence: graph_weight=0.3 hotfix from 2026-04-22 caused
        -8 pp AMB PersonaMem regression (Chi² p=0.045). Reverted empirically.
        """
        assert SearchConfig().default_graph_weight == pytest.approx(0.0)

    def test_parse_search_none_graph_weight_is_zero(self) -> None:
        """_parse_search(None) fallback for default_graph_weight is 0.0."""
        cfg = _parse_search(None)
        assert cfg.default_graph_weight == pytest.approx(0.0)

    def test_parse_search_explicit_zero_accepted(self) -> None:
        """Explicit default_graph_weight=0.0 in YAML is accepted."""
        cfg = _parse_search({"default_graph_weight": 0.0})
        assert cfg.default_graph_weight == pytest.approx(0.0)

    def test_parse_search_explicit_nonzero_preserved(self) -> None:
        """Explicit non-zero default_graph_weight from YAML is honoured (opt-in)."""
        cfg = _parse_search({"default_graph_weight": 0.2})
        assert cfg.default_graph_weight == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Integration — graph_weight=0.0 produces no score inflation
# ---------------------------------------------------------------------------


class TestGraphWeightZeroNoInflation:
    """With graph_weight=0.0, graph signal must not affect search scores."""

    @pytest.mark.asyncio
    async def test_search_with_graph_weight_zero_returns_results(
        self, store: SqliteEngravaCore
    ) -> None:
        """search_hybrid succeeds and returns results when graph_weight=0.0."""
        t1 = _obs(
            str(uuid.uuid4()), essence="cat cafe visit", content="I went to a cat cafe today."
        )
        await store.create_thought(t1)

        result = await store.search_hybrid(
            query_text="cat cafe",
            query_vector=None,
            top_k=5,
        )

        assert len(result.results) >= 1
        assert t1.thought_id in {tid for tid, _ in result.results}

    @pytest.mark.asyncio
    async def test_search_backends_used_no_graph_signal(self, store: SqliteEngravaCore) -> None:
        """With default config (graph_weight=0.0), 'graph' is absent from backends_used."""
        t1 = _obs(str(uuid.uuid4()), essence="morning run", content="Ran 5km this morning.")
        await store.create_thought(t1)

        result = await store.search_hybrid(
            query_text="morning run",
            query_vector=None,
            top_k=5,
        )

        assert "graph" not in result.backends_used, (
            "Expected no graph signal with default config, "
            f"got backends_used={result.backends_used!r}"
        )

"""Tests for the per-arm candidate-pool floor in ``search_hybrid`` (and ``recall``).

``fts_top_k`` / ``vector_top_k`` cap each arm's candidate pool before fusion.
Without a floor, a ``top_k`` above those pools (default 50 each) silently
returns fewer results than actually match, both directly through
``search_hybrid`` and through ``recall``, and the same under-fill reaches the
de-fragmentation (``collapse_key``) backfill, which only has the arms'
fetched candidates to draw from.

Each arm's effective pool is ``max(fts_top_k, top_k)`` / ``max(vector_top_k,
top_k)``, applied **before** any collapse-pool widening. For ``top_k`` at or
below the default pool (the common case) both ``max()`` calls are a no-op,
so this is byte-identical to the prior behaviour there.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.vector_sqlite_vec import SqliteVecSearchBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# Skip sqlite-vec-only tests when the optional extension is absent, matching
# the guard in tests/test_sqlite_vec.py -- without it, a real backend build
# fails on correct code too, which is not the behaviour under test.
sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)


def _make(
    thought_id: str,
    *,
    essence: str = "essence",
    content: str = "content",
    updated_cycle: int = 0,
    priority: Priority = Priority.P2,
    metadata: dict[str, object] | None = None,
) -> ThoughtRecord:
    """Minimal thought for search tests, mirroring the existing search fixtures."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=updated_cycle,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        metadata=metadata or {},
    )


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """Plain in-memory store, no vector backend, no search config."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(conn)
    await s.ensure_schema()
    try:
        yield s
    finally:
        await conn.close()


async def _build_backend_store(
    tmp_path: Path, *, backend: str, dimension: int
) -> SqliteEngravaCore:
    """Construct a store with a real (file-backed) connection and the given backend.

    ``backend`` is ``"numpy"`` (brute-force fallback) or ``"sqlite-vec"``. Mirrors
    ``TestSqliteVecRealConnection._build_store`` in ``tests/test_sqlite_vec.py``: a
    real file-backed connection, not ``:memory:``, since backend configuration goes
    through the same schema-bootstrap + ``_configure_vector_backend`` path either
    way, and lets a caller pick the backend explicitly instead of relying on
    whichever the plain ``store`` fixture happens to default to.
    """
    db_path = tmp_path / f"{backend}.db"
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    s = SqliteEngravaCore(db)
    s._owns_connection = True
    await s.ensure_schema()
    await s._configure_vector_backend(backend_name=backend, embedding_dimension=dimension)
    return s


# ---------------------------------------------------------------------------
# The floor: an FTS-only top_k above the default pool returns a full page.
# ---------------------------------------------------------------------------


class TestFloorFillsAFullPage:
    """``top_k`` above the per-arm pool no longer caps the result below it."""

    async def test_top_k_above_default_pool_returns_full_page(
        self, store: SqliteEngravaCore
    ) -> None:
        """120 FTS-only matches, top_k=100: the result reaches 100, not 50."""
        for i in range(120):
            await store.create_thought(_make(f"t-{i:03d}", content="keyword content"))

        result = await store.search_hybrid(query_text="keyword", top_k=100)

        assert len(result.results) == 100

    async def test_recall_inherits_the_same_floor(self, store: SqliteEngravaCore) -> None:
        """``recall`` reaches the floor through ``search_hybrid`` too."""
        for i in range(120):
            await store.create_thought(_make(f"t-{i:03d}", content="keyword content"))

        result = await store.recall("keyword", top_k=100)

        assert len(result.results) == 100

    async def test_top_k_at_default_pool_unchanged(self, store: SqliteEngravaCore) -> None:
        """At top_k <= 50 (the default pool), the floor is a no-op: still capped by top_k."""
        for i in range(120):
            await store.create_thought(_make(f"t-{i:03d}", content="keyword content"))

        result = await store.search_hybrid(query_text="keyword", top_k=50)

        assert len(result.results) == 50

    async def test_both_arms_reach_top_k_above_default_pool(self, store: SqliteEngravaCore) -> None:
        """With both arms active, the fused result also reaches top_k above 50."""
        query_vector = [1.0, 0.0, 0.0, 0.0]
        for i in range(110):
            tid = f"t-{i:03d}"
            await store.create_thought(_make(tid, content="keyword content"))
            await store.store_embedding(
                thought_id=tid, vector=query_vector, model_name="test-fixture-model"
            )

        result = await store.search_hybrid(
            query_text="keyword",
            query_vector=query_vector,
            top_k=100,
        )

        assert "fts5" in result.backends_used
        assert "vector" in result.backends_used
        assert len(result.results) == 100


# ---------------------------------------------------------------------------
# The floor is applied before collapse widening, so backfill still gets the
# same bounded headroom beyond top_k that it always did.
# ---------------------------------------------------------------------------


class TestFloorAppliedBeforeCollapseWidening:
    """The floor must raise the pool *before* the collapse-factor multiply.

    If the floor were applied after that multiply instead, a ``top_k`` large
    enough to dominate ``default_pool * collapse_pool_factor`` collapses the
    widening to zero headroom (``effective == top_k`` exactly), starving
    collapse backfill of the deeper distinct-unit pool it needs.
    """

    async def test_collapse_backfill_gets_the_floored_headroom(
        self, store: SqliteEngravaCore
    ) -> None:
        """A dominant duplicate unit does not starve backfill when top_k is large.

        50 rows share one collapse unit and rank above 260 rows that are each
        their own unit. With the floor applied before the x4 collapse
        widening, the pool (``max(50, 250) * 4 = 1000``) comfortably covers
        the whole 310-row corpus, so collapse (50 dup rows -> 1 survivor)
        still leaves 261 distinct units -- enough to fill top_k=250. Applying
        the floor after the widening instead would leave the pool at exactly
        ``max(50 * 4, 250) = 250`` -- no headroom beyond top_k at all -- and
        the result would fall short of a full page.
        """
        for i in range(50):
            await store.create_thought(
                _make(
                    f"a-dup-{i:03d}",
                    content="keyword keyword keyword keyword keyword",
                    metadata={"unit": "dup"},
                )
            )
        for i in range(260):
            await store.create_thought(
                _make(
                    f"b-live-{i:03d}",
                    content="keyword",
                    metadata={"unit": f"live-{i:03d}"},
                )
            )

        result = await store.search_hybrid(
            query_text="keyword",
            top_k=250,
            collapse_key="$.unit",
        )

        assert len(result.results) == 250


class TestVectorOnlyCollapseFloorAppliedBeforeWidening:
    """The floor's vector-arm placement matters with no FTS hits at all.

    Mirrors ``test_collapse_backfill_gets_the_floored_headroom`` above, but
    through the vector arm alone: ``query_text=""`` disables FTS outright (no
    FTS hits), so the only candidates come from ``search_similar`` on the
    numpy brute-force path (no sqlite-vec needed). The FTS collapse test does
    not exercise this -- moving *only* the vector arm's floor to after the
    collapse-factor multiply leaves every earlier test in this module green,
    because none of them collapse on the vector arm with no FTS hits.
    """

    async def test_vector_only_collapse_backfill_gets_the_floored_headroom(
        self, store: SqliteEngravaCore
    ) -> None:
        """50 near-identical embeddings share one unit, ranked above 260 others.

        Same shape as the FTS version: with the vector floor applied before
        the x4 collapse widening, the pool (``max(50, 250) * 4 = 1000``) covers
        the whole 310-row corpus, so collapse (50 dup rows -> 1 survivor)
        leaves 261 distinct units -- enough to fill ``top_k=250``. Moving only
        the vector floor to after the widening leaves the pool at exactly
        ``max(50 * 4, 250) = 250``, no headroom beyond ``top_k`` at all.
        """
        for i in range(50):
            tid = f"a-dup-{i:03d}"
            await store.create_thought(_make(tid, metadata={"unit": "dup"}))
            await store.store_embedding(
                thought_id=tid, vector=[1.0, 0.0], model_name="test-fixture-model"
            )
        for i in range(260):
            tid = f"b-live-{i:03d}"
            await store.create_thought(_make(tid, metadata={"unit": f"live-{i:03d}"}))
            await store.store_embedding(
                thought_id=tid, vector=[0.6, 0.8], model_name="test-fixture-model"
            )

        result = await store.search_hybrid(
            query_text="",
            query_vector=[1.0, 0.0],
            top_k=250,
            collapse_key="$.unit",
            vector_weight=1.0,
            priority_weight=0.0,
        )

        assert "fts5" not in result.backends_used
        assert "vector" in result.backends_used
        assert len(result.results) == 250


# ---------------------------------------------------------------------------
# A vector-only top_k above the default pool reaches a full page too, on
# either backend the store can select -- the floor is backend-independent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    ["numpy", pytest.param("sqlite-vec", marks=sqlite_vec_required)],
)
class TestVectorOnlyFloorFillsAFullPage:
    """120 vector-only matches, top_k=100: the result reaches 100, not 50.

    ``query_text=""`` disables the FTS arm outright, so only the vector
    floor is exercised. Runs on the numpy brute-force path (always, no
    extension needed) and, when available, on the real sqlite-vec backend
    too -- the floor lives in ``search_hybrid``, above either backend, so it
    must not depend on which one is selected.
    """

    async def test_top_k_above_default_pool_returns_full_page(
        self, tmp_path: Path, backend: str
    ) -> None:
        store = await _build_backend_store(tmp_path, backend=backend, dimension=2)
        try:
            for i in range(120):
                tid = f"v-{i:03d}"
                await store.create_thought(_make(tid))
                await store.store_embedding(
                    thought_id=tid, vector=[0.9, 0.436], model_name="test-fixture-model"
                )

            result = await store.search_hybrid(
                query_text="",
                query_vector=[1.0, 0.0],
                top_k=100,
                vector_weight=1.0,
                priority_weight=0.0,
            )

            assert "fts5" not in result.backends_used
            assert "vector" in result.backends_used
            assert len(result.results) == 100
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Small top_k (at or below the default pool) is unchanged: the floor's
# max() must use the *default*, not top_k itself, as the pool.
# ---------------------------------------------------------------------------


class TestSmallTopKUnchanged:
    """At top_k below the default pool, the arm must still fetch the full pool.

    A candidate ranked near the bottom of the FTS pool by keyword score, but
    boosted to the top by recency, only reaches the final result if the arm
    fetched it at all -- which requires the *default* pool (50), not top_k
    (10), to bound the fetch.
    """

    async def test_recency_boosted_candidate_survives_from_the_full_default_pool(
        self, store: SqliteEngravaCore
    ) -> None:
        for i in range(49):
            await store.create_thought(
                _make(
                    f"filler-{i:03d}",
                    content="keyword keyword keyword keyword keyword",
                    updated_cycle=0,
                )
            )
        await store.create_thought(_make("target", content="keyword", updated_cycle=1000))

        result = await store.search_hybrid(
            query_text="keyword",
            top_k=10,
            current_cycle=1000,
            recency_half_life=50,
            fts_weight=0.1,
            vector_weight=0.0,
            recency_weight=0.9,
            priority_weight=0.0,
            graph_weight=0.0,
        )

        assert len(result.results) == 10
        result_ids = {tid for tid, _ in result.results}
        assert "target" in result_ids


# ---------------------------------------------------------------------------
# A caller-supplied fts_top_k above both the default and top_k is honoured
# unchanged -- the floor only ever raises, never lowers, the pool.
# ---------------------------------------------------------------------------


class TestExplicitPoolIsHonoured:
    async def test_explicit_fts_top_k_above_default_passes_through(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        real_search_fts = SqliteEngravaCore.search_fts

        async def spy_search_fts(
            self: SqliteEngravaCore,
            query: str,
            top_k: int = 10,
            *,
            include_archived: bool = False,
            _filter_clause: tuple[str, list[object]] | None = None,
        ) -> list[tuple[str, float]]:
            calls.append(top_k)
            return await real_search_fts(
                self,
                query,
                top_k,
                include_archived=include_archived,
                _filter_clause=_filter_clause,
            )

        monkeypatch.setattr(SqliteEngravaCore, "search_fts", spy_search_fts)

        await store.create_thought(_make("t-1", content="keyword content"))
        await store.search_hybrid(query_text="keyword", top_k=10, fts_top_k=200)

        assert calls == [200]


# ---------------------------------------------------------------------------
# The vec0 double-widening (search_similar's own over-fetch factor, under
# _VEC0_OVERFETCH_CAP) is untouched: at top_k=10 the request is identical to
# before the floor existed, and at top_k=100 the change is exactly the
# floor's effect, nothing else.
# ---------------------------------------------------------------------------


@sqlite_vec_required
class TestVec0WideningsAreUntouched:
    async def test_vec0_request_size_at_top_k_10_and_100(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At top_k=10 the request is unchanged; at top_k=100 only the floor moves it.

        Default ``vector_top_k`` (50) and ``vec0_overfetch_factor`` (4), no
        ``SearchConfig``:

        - top_k=10: effective pool stays 50 (the floor is a no-op there), so
          the vec0 request is ``min(50 * 4, 500) = 200`` -- identical to
          before this WS.
        - top_k=100: the floor raises the pool to 100, so the request is
          ``min(100 * 4, 500) = 400`` -- exactly what the floor predicts, with
          no other change to the over-fetch formula or its cap.
        """
        calls: list[int] = []
        real_search = SqliteVecSearchBackend.search

        async def spy_search(
            self: SqliteVecSearchBackend,
            db: aiosqlite.Connection,
            query_vector: list[float],
            top_k: int = 10,
            threshold: float = 0.0,
        ) -> list[tuple[str, float]]:
            calls.append(top_k)
            return await real_search(self, db, query_vector, top_k, threshold)

        monkeypatch.setattr(SqliteVecSearchBackend, "search", spy_search)

        store = await _build_backend_store(tmp_path, backend="sqlite-vec", dimension=2)
        try:
            await store.search_hybrid(
                query_text="",
                query_vector=[1.0, 0.0],
                top_k=10,
                vector_weight=1.0,
                priority_weight=0.0,
            )
            await store.search_hybrid(
                query_text="",
                query_vector=[1.0, 0.0],
                top_k=100,
                vector_weight=1.0,
                priority_weight=0.0,
            )
        finally:
            await store.close()

        assert calls == [200, 400]

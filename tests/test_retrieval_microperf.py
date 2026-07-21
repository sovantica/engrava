"""Read-path micro-performance regression tests.

These lock in a bundle of behaviour-preserving read-path optimisations:

* ``search_reflections_only`` batch-fetches every REFLECTION embedding in a
  single ``... IN (…)`` query instead of one round trip per REFLECTION.
* The numpy cosine arm of ``search_similar`` decodes all vector blobs with a
  single ``np.frombuffer`` instead of a per-row ``struct.unpack`` loop.
* An ``edge(edge_type, to_thought_id)`` composite index backs the
  edge-type-scoped inbound lookup so it seeks on both columns.

The load-bearing property throughout is that **results do not move**: the same
ranked ids and the same scores come out before and after. Each optimisation is
paired with an identity assertion (batch decode == per-row decode, ranking is
stable, the index changes only the query plan).
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import aiosqlite
import numpy as np
import pytest

from engrava import SearchConfig, SqliteEngravaCore
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import (
    _format_fts_bare_fragment,
    _normalize_fts_query,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

_DIM = 8


def _unit_vector(seed: int, dim: int = _DIM) -> list[float]:
    """Deterministic pseudo-random unit-ish vector for a given seed."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.uniform(-1.0, 1.0, size=dim)]


def _thought(
    thought_id: str,
    *,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    essence: str = "essence",
    content: str = "content",
) -> ThoughtRecord:
    """Minimal ACTIVE thought."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
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
    """Store with graph expansion off so scoring is easy to reason about."""
    conn = await aiosqlite.connect(str(tmp_path / "microperf.db"))
    conn.row_factory = aiosqlite.Row
    cfg = SearchConfig(graph_expansion_enabled=False, default_graph_weight=0.0)
    s = SqliteEngravaCore(conn, search_config=cfg)
    await s.ensure_schema()
    yield s
    await conn.close()


async def _seed_mixed_store(store: SqliteEngravaCore) -> tuple[list[str], list[str]]:
    """Create a mix of OBSERVATIONs + REFLECTIONs, each with an embedding.

    Returns ``(observation_ids, reflection_ids)``.
    """
    obs_ids: list[str] = []
    for i in range(6):
        t = await store.create_thought(_thought(f"obs-{i}", essence=f"alpha topic observation {i}"))
        await store.store_embedding(t.thought_id, _unit_vector(i), model_name="test")
        obs_ids.append(t.thought_id)

    reflection_ids: list[str] = []
    for i in range(5):
        r = await store.create_thought(
            _thought(
                f"ref-{i}",
                thought_type=ThoughtType.REFLECTION,
                essence=f"alpha topic reflection {i}",
            )
        )
        await store.store_embedding(r.thought_id, _unit_vector(100 + i), model_name="test")
        reflection_ids.append(r.thought_id)

    return obs_ids, reflection_ids


# ---------------------------------------------------------------------------
# Item 2 — numpy-arm batch decode is bit-identical to per-row struct.unpack
# ---------------------------------------------------------------------------


class TestNumpyBatchDecode:
    """The ``np.frombuffer`` batch decode must equal the old per-row decode."""

    def test_frombuffer_equals_struct_unpack_bit_for_bit(self) -> None:
        """Decoded matrices are exactly equal for the same stored blobs."""
        vectors = [_unit_vector(seed, dim=_DIM) for seed in range(20)]
        blobs = [struct.pack(f"{_DIM}f", *v) for v in vectors]

        # Old path: per-row struct.unpack into a float64 matrix.
        reference = np.asarray(
            [list(struct.unpack(f"{_DIM}f", b)) for b in blobs], dtype=np.float64
        )
        # New path: one frombuffer (native float32, matching the native
        # ``struct.pack`` storage) over the joined buffer, widened to float64.
        batched = (
            np.frombuffer(b"".join(blobs), dtype=np.float32)
            .reshape(len(blobs), _DIM)
            .astype(np.float64)
        )

        assert np.array_equal(reference, batched)

    async def test_search_similar_ranking_stable(self, store: SqliteEngravaCore) -> None:
        """``search_similar`` returns a deterministic, repeatable ranking.

        The batch-decode arm feeds the same float64 matrix as the per-row loop,
        so scoring is identical run to run and matches an independent reference
        cosine computed straight from the stored vectors.
        """
        obs_ids, reflection_ids = await _seed_mixed_store(store)
        all_ids = obs_ids + reflection_ids
        query = _unit_vector(7)

        result_a = await store.search_similar(query, top_k=len(all_ids))
        result_b = await store.search_similar(query, top_k=len(all_ids))
        assert result_a == result_b  # deterministic

        # Independent reference cosine over the same vectors. Stored vectors are
        # persisted through float32 (``struct.pack("f")``), so the reference
        # must round-trip through float32 too to match the decoded values.
        q = np.asarray(query, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        expected: list[tuple[str, float]] = []
        seed_by_id = {f"obs-{i}": i for i in range(len(obs_ids))}
        seed_by_id.update({f"ref-{i}": 100 + i for i in range(len(reflection_ids))})
        for tid in all_ids:
            v = np.asarray(_unit_vector(seed_by_id[tid]), dtype=np.float32).astype(np.float64)
            score = float(v @ q / (np.linalg.norm(v) * qn))
            # ``search_similar`` drops rows below the default ``threshold=0.0``.
            if score >= 0.0:
                expected.append((tid, score))
        expected.sort(key=lambda x: (-x[1], x[0]))

        assert [tid for tid, _ in result_a] == [tid for tid, _ in expected]
        for (_, got), (_, ref) in zip(result_a, expected, strict=True):
            assert got == pytest.approx(ref, abs=1e-9)

    async def test_search_similar_tolerates_short_blob(self, store: SqliteEngravaCore) -> None:
        """A truncated blob falls back to the per-row path without a hard error.

        The batch fast path is only taken when every blob has the expected
        length; a short/corrupt row abandons it for the original per-row decode,
        which raises exactly as before (``struct.error``) — the point is that
        the fast path never silently changes that behaviour.
        """
        t = await store.create_thought(_thought("short-1", essence="alpha"))
        await store.store_embedding(t.thought_id, _unit_vector(1), model_name="test")
        # Corrupt the stored blob to fewer bytes than ``dimension`` implies.
        await store._db.execute(
            "UPDATE embedding SET vector_blob = ? WHERE owner_id = ?",
            (b"\x00\x00\x00\x00", t.thought_id),
        )
        await store._db.commit()

        with pytest.raises(struct.error):
            await store.search_similar(_unit_vector(1), top_k=5)


# ---------------------------------------------------------------------------
# Item 1 — search_reflections_only batches its embedding fetch (O(1) queries)
# ---------------------------------------------------------------------------


class TestReflectionsOnlyBatchFetch:
    """``search_reflections_only`` must not issue one embedding query per row."""

    async def test_embedding_fetch_is_o1(self, store: SqliteEngravaCore) -> None:
        """Embedding fetch count is independent of the REFLECTION count.

        The per-id ``get_embedding`` loop issued one ``FROM embedding`` query
        per REFLECTION; the batched fetch issues a single ``IN (…)`` query
        regardless of how many REFLECTIONs exist.
        """
        _obs_ids, _reflection_ids = await _seed_mixed_store(store)

        embedding_queries = 0
        original_execute = store._db.execute

        def _counting_execute(sql: str, *args: object, **kwargs: object) -> object:
            nonlocal embedding_queries
            if "FROM embedding" in sql:
                embedding_queries += 1
            return original_execute(sql, *args, **kwargs)

        store._db.execute = _counting_execute  # type: ignore[method-assign]
        try:
            result = await store.search_reflections_only(
                "alpha topic", query_vector=_unit_vector(7), top_k=10
            )
        finally:
            store._db.execute = original_execute  # type: ignore[method-assign]

        # 5 REFLECTIONs are scored; the batched fetch must be a single query
        # (well under the naive one-per-reflection count of 5).
        assert embedding_queries == 1
        assert len(result.results) == 5

    async def test_ranking_matches_reference_cosine(self, store: SqliteEngravaCore) -> None:
        """Batched fetch produces the same ranking as an independent cosine."""
        _obs_ids, reflection_ids = await _seed_mixed_store(store)
        query = _unit_vector(7)

        result = await store.search_reflections_only("alpha topic", query_vector=query, top_k=10)

        # Stored vectors round-trip through float32; match that in the reference.
        q = np.asarray(query, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        expected: list[tuple[str, float]] = []
        for i, tid in enumerate(reflection_ids):
            v = np.asarray(_unit_vector(100 + i), dtype=np.float32).astype(np.float64)
            vn = float(np.linalg.norm(v))
            expected.append((tid, float(v @ q) / (qn * vn)))
        expected.sort(key=lambda x: x[1], reverse=True)

        assert [tid for tid, _ in result.results] == [tid for tid, _ in expected]
        for (_, got), (_, ref) in zip(result.results, expected, strict=True):
            assert got == pytest.approx(ref, abs=1e-9)

    async def test_matches_get_embedding_under_duplicate_owner(
        self, store: SqliteEngravaCore
    ) -> None:
        """Under duplicate owner rows, the batch keeps the same row as ``get_embedding``.

        The default ``embedding_id`` is deterministic (one row per owner), so
        this is only reachable by explicitly writing distinct ``embedding_id``
        values for one owner. In that pathological case ``get_embedding`` returns
        the lowest-``rowid`` row; the batch fetch must agree so scoring never
        diverges.
        """
        r = await store.create_thought(
            _thought("dup", thought_type=ThoughtType.REFLECTION, essence="alpha dup")
        )
        await store.store_embedding(
            r.thought_id, _unit_vector(1), model_name="test", embedding_id="dup-A"
        )
        await store.store_embedding(
            r.thought_id, _unit_vector(2), model_name="test", embedding_id="dup-B"
        )

        single = await store.get_embedding(r.thought_id)
        assert single is not None
        batched = await store._batch_fetch_embedding_blobs([r.thought_id])
        dim, blob = batched[r.thought_id]
        assert dim == single.dimension
        assert blob == single.vector_blob

    async def test_missing_embedding_scored_zero(self, store: SqliteEngravaCore) -> None:
        """A REFLECTION with no embedding row is scored 0.0, as before.

        The batched fetch omits ids with no embedding row; the scoring loop
        maps those to 0.0 exactly as the per-row ``emb is None`` branch did.
        """
        r_scored = await store.create_thought(
            _thought("r-scored", thought_type=ThoughtType.REFLECTION, essence="alpha yes")
        )
        await store.store_embedding(r_scored.thought_id, _unit_vector(3), model_name="test")
        r_bare = await store.create_thought(
            _thought("r-bare", thought_type=ThoughtType.REFLECTION, essence="alpha none")
        )

        result = await store.search_reflections_only(
            "alpha", query_vector=_unit_vector(3), top_k=10
        )
        by_id = dict(result.results)

        assert by_id[r_bare.thought_id] == pytest.approx(0.0)
        assert by_id[r_scored.thought_id] > 0.0


# ---------------------------------------------------------------------------
# Byte-identical ranking regression — the primary guard
# ---------------------------------------------------------------------------


class TestRankingRegression:
    """The full retrieval surface produces stable, deterministic rankings."""

    async def test_recall_and_hybrid_deterministic(self, store: SqliteEngravaCore) -> None:
        """``recall`` / ``search_hybrid`` / ``search_reflections_only`` repeat exactly.

        Runs each entry point twice on the same seeded store and asserts the
        ranked ids and scores are byte-identical. The perf changes must not
        introduce any run-to-run drift.
        """
        await _seed_mixed_store(store)
        query = _unit_vector(7)

        recall_a = await store.recall("alpha topic", top_k=8)
        recall_b = await store.recall("alpha topic", top_k=8)
        assert recall_a.results == recall_b.results

        hybrid_a = await store.search_hybrid("alpha topic", query_vector=query, top_k=8)
        hybrid_b = await store.search_hybrid("alpha topic", query_vector=query, top_k=8)
        assert hybrid_a.results == hybrid_b.results

        refs_a = await store.search_reflections_only("alpha", query_vector=query, top_k=8)
        refs_b = await store.search_reflections_only("alpha", query_vector=query, top_k=8)
        assert refs_a.results == refs_b.results


# ---------------------------------------------------------------------------
# Item 3 — composite (edge_type, to_thought_id) index + idempotent migration
# ---------------------------------------------------------------------------


class TestCompositeEdgeIndex:
    """The inbound edge-type-scoped lookup seeks the composite index."""

    async def _seed_edges(self, store: SqliteEngravaCore) -> None:
        parent = await store.create_thought(
            _thought("parent", thought_type=ThoughtType.REFLECTION, essence="parent")
        )
        for i in range(6):
            child = await store.create_thought(_thought(f"child-{i}", essence=f"child {i}"))
            await store.create_edge(
                EdgeRecord(
                    edge_id=f"edge-{i}",
                    from_thought_id=parent.thought_id,
                    to_thought_id=child.thought_id,
                    edge_type=EdgeType.CONSOLIDATED_FROM,
                    weight=1.0,
                    created_cycle=0,
                    source=KnowledgeSource.DREAMING,
                )
            )

    async def test_inbound_edge_type_query_uses_composite_index(
        self, store: SqliteEngravaCore
    ) -> None:
        """``to_thought_id = ? AND edge_type = ?`` seeks ``idx_edge_type_to``.

        This is the ``reflections_consolidated_from`` scan shape. Before the
        composite index it could only seek ``to_thought_id`` (single-column
        ``idx_edge_to_thought``) and test ``edge_type`` as a residual; the
        composite lets one seek satisfy both predicates.
        """
        await self._seed_edges(store)
        sql = (
            "SELECT DISTINCT t.thought_id AS thought_id FROM edge e "
            "JOIN thought t ON e.from_thought_id = t.thought_id "
            "WHERE e.to_thought_id = ? AND e.edge_type = 'CONSOLIDATED_FROM' "
            "AND t.thought_type = 'REFLECTION'"
        )
        cursor = await store._db.execute("EXPLAIN QUERY PLAN " + sql, ("child-0",))
        plan = " ".join(str(row["detail"]) for row in await cursor.fetchall())

        # The composite index is used (the perf guarantee) and there is no full
        # table scan (the robust invariant). The exact operator-order substring
        # in the plan is SQLite-planner-formatting-dependent, so it is not
        # asserted; index *existence* is separately guarded by
        # test_index_present_after_bootstrap.
        assert "idx_edge_type_to" in plan
        assert "SCAN edge" not in plan  # no full table scan

    async def test_graph_signal_query_avoids_full_scan(self, store: SqliteEngravaCore) -> None:
        """The graph-signal edge query never full-scans the ``to_thought_id`` side.

        ``_load_graph_signal`` filters ``from_thought_id IN (…) OR
        to_thought_id IN (…)``; the ``to_thought_id`` arm is served by
        ``idx_edge_to_thought`` (a MULTI-INDEX OR), so there is no full scan.
        """
        await self._seed_edges(store)
        sql = (
            "SELECT from_thought_id, to_thought_id, weight FROM edge "
            "WHERE from_thought_id IN (?, ?) OR to_thought_id IN (?, ?) "
            "ORDER BY weight DESC"
        )
        cursor = await store._db.execute(
            "EXPLAIN QUERY PLAN " + sql, ("parent", "child-0", "parent", "child-0")
        )
        plan = " ".join(str(row["detail"]) for row in await cursor.fetchall())

        assert "SCAN edge" not in plan
        assert "idx_edge_to_thought" in plan

    async def test_index_present_after_bootstrap(self, store: SqliteEngravaCore) -> None:
        """A fresh bootstrap carries the composite index."""
        cursor = await store._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_edge_type_to'"
        )
        assert await cursor.fetchone() is not None

    async def test_migration_idempotent(self, tmp_path: Path) -> None:
        """Running ``ensure_schema`` twice is safe and leaves the index in place."""
        conn = await aiosqlite.connect(str(tmp_path / "idem.db"))
        conn.row_factory = aiosqlite.Row
        try:
            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            await s.ensure_schema()  # must not raise

            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert row is not None
            assert int(row[0]) == 20

            cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_edge_type_to'"
            )
            assert await cursor.fetchone() is not None
        finally:
            await conn.close()

    async def test_reflections_consolidated_from_correct(self, store: SqliteEngravaCore) -> None:
        """The optimised query still returns the right rows (results unchanged)."""
        await self._seed_edges(store)
        # ``child-0`` was consolidated by the REFLECTION ``parent``.
        parents = await store.reflections_consolidated_from("child-0")
        assert parents == ["parent"]


# ---------------------------------------------------------------------------
# Item 6 — hyphen preservation in the FTS bare-fragment formatter
# ---------------------------------------------------------------------------


class TestHyphenPreservation:
    """A hyphenated term survives the FTS sanitiser intact."""

    def test_fragment_formatter_quotes_hyphenated_term(self) -> None:
        """``_format_fts_bare_fragment`` keeps the hyphen and quotes the term."""
        assert _format_fts_bare_fragment("well-being", in_bare_query=True) == '"well-being"'

    def test_fragment_formatter_preserves_hyphen_with_prefix(self) -> None:
        """A trailing ``*`` prefix marker is preserved alongside the hyphen."""
        assert _format_fts_bare_fragment("co-work*", in_bare_query=True) == '"co-work"*'

    def test_normalize_keeps_hyphenated_term_whole(self) -> None:
        """The full normaliser does not split a hyphenated term into pieces."""
        assert _normalize_fts_query("well-being") == '"well-being"'
        assert _normalize_fts_query("state-of-the-art system") == ('"state-of-the-art" OR system')

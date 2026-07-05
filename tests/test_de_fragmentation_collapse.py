"""Integration + unit tests for de-fragmentation collapse-by-unit + backfill.

Exercises ``search_hybrid`` / ``recall`` with ``collapse_key=`` against a real
SQLite store, plus unit coverage of the pure collapse/normalize helpers.

Covers, per the acceptance criteria:

* API surface + ``collapse_key=None`` candidate/score/order parity.
* Recall-neutrality (never drop a distinct unit, never surface a non-candidate).
* Best-per-unit keeper + distinct-unit backfill into freed slots.
* NULL / partial-composite / malformed-metadata key ⇒ own unit, no abort.
* Collapse x ``reflection_topk_cap`` ordering (single backfill source).
* Determinism / canonical keeper invariant to scan order.
* Bounded candidate-pool widening + the ``None`` path being unaffected.
* Composition with ``filters=`` / ``visibility=``.
* Path validation raised at argument time.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import SearchConfig
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.exceptions import InvalidFilterPathError
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.filters import (
    FieldOp,
    FieldPredicate,
    MetadataFilter,
    VisibilityQueryFilter,
)
from engrava.domain.models.thought import MetadataValue, ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import (
    _collapse_ranked_by_unit,
    _normalize_collapse_key,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thought(
    thought_id: str,
    *,
    essence: str,
    content: str | None = None,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    metadata: dict[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Build a minimal thought with the given metadata."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=essence,
        content=content if content is not None else essence,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        source="test",
        metadata=metadata or {},
    )


def _edge(from_id: str, to_id: str, *, weight: float = 1.0) -> EdgeRecord:
    """Build a CONSOLIDATED_FROM edge (REFLECTION -> source OBSERVATION)."""
    return EdgeRecord(
        edge_id=str(uuid.uuid4()),
        from_thought_id=from_id,
        to_thought_id=to_id,
        edge_type=EdgeType.CONSOLIDATED_FROM,
        weight=weight,
        created_cycle=0,
        source=KnowledgeSource.DREAMING,
    )


def _ids(result: list[tuple[str, float]]) -> list[str]:
    """Extract the ordered thought IDs from a result list."""
    return [tid for tid, _ in result]


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh in-file store with the default SearchConfig."""
    conn = await aiosqlite.connect(str(tmp_path / "collapse.db"))
    conn.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(conn, search_config=SearchConfig())
    await s.ensure_schema()
    yield s
    await conn.close()


# ---------------------------------------------------------------------------
# Pure-helper unit tests (collapse + normalize)
# ---------------------------------------------------------------------------


class TestNormalizeCollapseKey:
    """A1 / A10 — single-str normalization + argument-time path validation."""

    def test_single_str_becomes_one_element_tuple(self) -> None:
        """A bare path string normalizes to a one-element composite key."""
        assert _normalize_collapse_key("$.session_turn") == ("$.session_turn",)

    def test_sequence_preserves_order(self) -> None:
        """A sequence of paths is kept in caller order."""
        assert _normalize_collapse_key(["$.session_id", "$.turn"]) == (
            "$.session_id",
            "$.turn",
        )

    def test_bad_path_raises_invalid_filter_path_error(self) -> None:
        """A malformed path is rejected with the typed error at argument time."""
        with pytest.raises(InvalidFilterPathError):
            _normalize_collapse_key("not-a-path")

    def test_bad_path_in_composite_raises(self) -> None:
        """One bad component of a composite key rejects the whole key."""
        with pytest.raises(InvalidFilterPathError):
            _normalize_collapse_key(["$.ok", "bad path"])

    def test_empty_sequence_rejected(self) -> None:
        """An empty composite key has no grouping identity and is rejected."""
        with pytest.raises(InvalidFilterPathError):
            _normalize_collapse_key([])


class TestCollapseRankedByUnit:
    """A3 / A5 / A7 — keeper selection, key-less pass-through, order preserved."""

    def test_keeps_first_per_unit_drops_later_same_unit(self) -> None:
        """The first (highest-ranked) member per unit is the keeper."""
        ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        keys: dict[str, tuple[object, ...] | None] = {
            "a": ("u1",),
            "b": ("u1",),  # same unit as a -> dropped
            "c": ("u2",),
            "d": ("u2",),  # same unit as c -> dropped
        }
        assert _collapse_ranked_by_unit(ranked, keys) == [("a", 0.9), ("c", 0.7)]

    def test_keyless_rows_each_their_own_unit(self) -> None:
        """None-keyed rows always pass through; never collapsed together."""
        ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        keys: dict[str, tuple[object, ...] | None] = {"a": None, "b": None, "c": None}
        assert _collapse_ranked_by_unit(ranked, keys) == ranked

    def test_missing_id_treated_as_keyless(self) -> None:
        """An id absent from the key map is treated as key-less (own unit)."""
        ranked = [("a", 0.9), ("b", 0.8)]
        keys: dict[str, tuple[object, ...] | None] = {"a": ("u1",)}
        # 'b' missing -> own unit, survives.
        assert _collapse_ranked_by_unit(ranked, keys) == ranked

    def test_d8_order_preserved(self) -> None:
        """Survivor order follows the input D8 order; no re-sort is introduced."""
        ranked = [("z", 0.9), ("y", 0.8), ("x", 0.7)]
        keys: dict[str, tuple[object, ...] | None] = {
            "z": ("u1",),
            "y": ("u2",),
            "x": ("u1",),  # dropped
        }
        out = _collapse_ranked_by_unit(ranked, keys)
        assert out == [("z", 0.9), ("y", 0.8)]


# ---------------------------------------------------------------------------
# API surface + None-path parity (A1)
# ---------------------------------------------------------------------------


class TestApiSurface:
    """collapse_key exists on both entry points; None preserves the path."""

    async def test_search_hybrid_accepts_collapse_key(self, store: SqliteEngravaCore) -> None:
        """The keyword-only param exists and a single str path is accepted."""
        await store.create_thought(_thought("t1", essence="alpha apple", metadata={"unit": "u1"}))
        result = await store.search_hybrid("alpha", collapse_key="$.unit")
        assert _ids(result.results) == ["t1"]

    async def test_recall_delegates_collapse_key(self, store: SqliteEngravaCore) -> None:
        """recall forwards collapse_key= to search_hybrid."""
        # Two fragments of one unit + one distinct unit.
        await store.create_thought(_thought("f1", essence="beta banana", metadata={"unit": "u1"}))
        await store.create_thought(_thought("f2", essence="beta banana", metadata={"unit": "u1"}))
        await store.create_thought(_thought("d1", essence="beta banana", metadata={"unit": "u2"}))
        result = await store.recall("beta", collapse_key="$.unit")
        returned = set(_ids(result.results))
        # At most one of the u1 fragments survives; the distinct unit is kept.
        assert "d1" in returned
        assert len(returned & {"f1", "f2"}) == 1

    async def test_none_parity_candidate_score_order(self, store: SqliteEngravaCore) -> None:
        """collapse_key=None is byte-identical to omitting it (candidate/score/order)."""
        for i in range(5):
            await store.create_thought(
                _thought(f"t{i}", essence=f"gamma grape {i}", metadata={"unit": "u1"})
            )
        baseline = await store.search_hybrid("gamma", top_k=10)
        explicit_none = await store.search_hybrid("gamma", top_k=10, collapse_key=None)
        assert baseline.results == explicit_none.results

    async def test_path_validation_at_argument_time(self, store: SqliteEngravaCore) -> None:
        """A bad collapse path raises before any row is touched (A10)."""
        await store.create_thought(_thought("t1", essence="delta date"))
        with pytest.raises(InvalidFilterPathError):
            await store.search_hybrid("delta", collapse_key="bogus[path")


# ---------------------------------------------------------------------------
# Recall-neutrality -- criterion A2
# ---------------------------------------------------------------------------


class TestRecallNeutrality:
    """Collapse only removes lower-ranked members of the SAME unit."""

    async def test_no_distinct_unit_dropped_and_no_phantom_row(
        self, store: SqliteEngravaCore
    ) -> None:
        """Every distinct unit survives; nothing the arms did not produce appears."""
        # Three distinct units, each with two fragments (same essence -> tie scores).
        produced: set[str] = set()
        for unit in ("u1", "u2", "u3"):
            for frag in ("a", "b"):
                tid = f"{unit}-{frag}"
                produced.add(tid)
                await store.create_thought(
                    _thought(tid, essence="epsilon keyword", metadata={"unit": unit})
                )
        result = await store.search_hybrid("epsilon", top_k=10, collapse_key="$.unit")
        returned = _ids(result.results)
        # Exactly one survivor per distinct unit.
        units_seen = {tid.split("-")[0] for tid in returned}
        assert units_seen == {"u1", "u2", "u3"}
        assert len(returned) == 3
        # No phantom: every returned id was actually produced.
        assert set(returned) <= produced

    async def test_scores_unchanged_by_collapse(self, store: SqliteEngravaCore) -> None:
        """The surviving keeper keeps the exact score it had without collapse."""
        await store.create_thought(_thought("d1", essence="zeta one", metadata={"unit": "u1"}))
        await store.create_thought(_thought("d2", essence="zeta two", metadata={"unit": "u2"}))
        plain = dict((await store.search_hybrid("zeta", top_k=10)).results)
        collapsed = dict(
            (await store.search_hybrid("zeta", top_k=10, collapse_key="$.unit")).results
        )
        for tid, score in collapsed.items():
            assert plain[tid] == score


# ---------------------------------------------------------------------------
# Best-per-unit + backfill -- criterion A3
# ---------------------------------------------------------------------------


class TestBestPerUnitBackfill:
    """A fragment-dominated window backfills deeper distinct units."""

    async def test_fragment_window_backfills_distinct_units(self, store: SqliteEngravaCore) -> None:
        """top_k slots dominated by one unit's fragments are freed for distinct units."""
        # Unit u1 has many strong fragments (high BM25 via repeated token);
        # several distinct single-fragment units sit deeper.
        for i in range(8):
            await store.create_thought(
                _thought(
                    f"u1-{i}",
                    essence="theta theta theta keyword",  # strong, same unit
                    metadata={"unit": "u1"},
                )
            )
        for u in range(2, 6):  # u2..u5 distinct, single fragment, weaker
            await store.create_thought(
                _thought(f"u{u}", essence="theta keyword", metadata={"unit": f"u{u}"})
            )
        result = await store.search_hybrid("theta keyword", top_k=5, collapse_key="$.unit")
        returned = _ids(result.results)
        units_seen = [tid.split("-")[0] for tid in returned]
        # Exactly one u1 survivor; the rest are distinct backfilled units.
        assert units_seen.count("u1") == 1
        assert set(units_seen) == {"u1", "u2", "u3", "u4", "u5"}
        assert len(returned) == 5

    async def test_keeper_is_highest_ranked_member(self, store: SqliteEngravaCore) -> None:
        """The retained member of a unit is its highest-D8-ranked fragment.

        Both fragments share the same essence (equal BM25); the keeper is made
        unambiguous via the priority signal (P1 boosts ``strong`` above
        ``weak``), so the survivor is the higher-scored member, not an artefact
        of BM25 length normalization.
        """
        strong = _thought("strong", essence="iota word", metadata={"unit": "u1"})
        weak = _thought("weak", essence="iota word", metadata={"unit": "u1"})
        await store.create_thought(
            ThoughtRecord(
                thought_id=strong.thought_id,
                thought_type=strong.thought_type,
                essence=strong.essence,
                content=strong.content,
                priority=Priority.P1,
                lifecycle_status=strong.lifecycle_status,
                source=strong.source,
                metadata=dict(strong.metadata),
            )
        )
        await store.create_thought(
            ThoughtRecord(
                thought_id=weak.thought_id,
                thought_type=weak.thought_type,
                essence=weak.essence,
                content=weak.content,
                priority=Priority.P4,
                lifecycle_status=weak.lifecycle_status,
                source=weak.source,
                metadata=dict(weak.metadata),
            )
        )
        result = await store.search_hybrid(
            "iota word", top_k=10, priority_weight=0.5, collapse_key="$.unit"
        )
        assert _ids(result.results) == ["strong"]

    async def test_final_length_is_min_topk_and_distinct_units(
        self, store: SqliteEngravaCore
    ) -> None:
        """final length = min(top_k, #distinct units present)."""
        for frag in range(6):
            await store.create_thought(
                _thought(f"u1-{frag}", essence="kappa word", metadata={"unit": "u1"})
            )
        await store.create_thought(_thought("u2", essence="kappa word", metadata={"unit": "u2"}))
        result = await store.search_hybrid("kappa", top_k=5, collapse_key="$.unit")
        # Only 2 distinct units exist, so final has 2 rows even though top_k=5.
        assert len(result.results) == 2


# ---------------------------------------------------------------------------
# NULL / composite / malformed key semantics (A5)
# ---------------------------------------------------------------------------


class TestKeylessSemantics:
    """A key-less row is its own unit; the query never aborts."""

    async def test_two_keyless_rows_both_survive(self, store: SqliteEngravaCore) -> None:
        """Two distinct rows missing the unit key are NOT collapsed together."""
        await store.create_thought(_thought("k1", essence="lambda word", metadata={}))
        await store.create_thought(_thought("k2", essence="lambda word", metadata={}))
        result = await store.search_hybrid("lambda", top_k=10, collapse_key="$.unit")
        assert set(_ids(result.results)) == {"k1", "k2"}

    async def test_partial_composite_key_is_own_unit(self, store: SqliteEngravaCore) -> None:
        """A composite row missing one component is key-less (own unit)."""
        # Both rows share session_id but only one has turn -> partial = own unit.
        await store.create_thought(
            _thought("p1", essence="mu word", metadata={"session_id": "s", "turn": 1})
        )
        await store.create_thought(
            _thought("p2", essence="mu word", metadata={"session_id": "s"})  # no turn
        )
        await store.create_thought(
            _thought("p3", essence="mu word", metadata={"session_id": "s"})  # no turn
        )
        result = await store.search_hybrid("mu", top_k=10, collapse_key=["$.session_id", "$.turn"])
        # p2 and p3 both lack 'turn' -> each its own unit -> both survive.
        assert set(_ids(result.results)) == {"p1", "p2", "p3"}

    async def test_full_composite_key_collapses(self, store: SqliteEngravaCore) -> None:
        """Two rows with all composite components equal share a unit."""
        await store.create_thought(
            _thought("c1", essence="nu nu word", metadata={"session_id": "s", "turn": 1})
        )
        await store.create_thought(
            _thought("c2", essence="nu word", metadata={"session_id": "s", "turn": 1})
        )
        await store.create_thought(
            _thought("c3", essence="nu word", metadata={"session_id": "s", "turn": 2})
        )
        result = await store.search_hybrid("nu", top_k=10, collapse_key=["$.session_id", "$.turn"])
        returned = set(_ids(result.results))
        # (s,1) collapses to exactly one of {c1, c2}; (s,2)=c3 is a distinct unit.
        assert "c3" in returned
        assert len(returned & {"c1", "c2"}) == 1
        assert len(returned) == 2

    async def test_malformed_metadata_is_own_unit_no_abort(self, store: SqliteEngravaCore) -> None:
        """A malformed metadata_json row collapses to a NULL key; query runs."""
        await store.create_thought(_thought("good", essence="xi word", metadata={"unit": "u1"}))
        await store.create_thought(_thought("bad1", essence="xi word", metadata={"unit": "u1"}))
        await store.create_thought(_thought("bad2", essence="xi word", metadata={"unit": "u1"}))
        # Corrupt the two 'bad' rows' metadata_json directly.
        for tid in ("bad1", "bad2"):
            await store._db.execute(
                "UPDATE thought SET metadata_json = ? WHERE thought_id = ?",
                ("{not valid json", tid),
            )
        await store._db.commit()
        result = await store.search_hybrid("xi", top_k=10, collapse_key="$.unit")
        returned = set(_ids(result.results))
        # 'good' survives; both malformed rows are key-less -> own units -> both survive.
        assert returned == {"good", "bad1", "bad2"}

    async def test_object_valued_key_collapses_by_json_text(self, store: SqliteEngravaCore) -> None:
        """A key path pointing at a JSON object/array value collapses by its TEXT.

        ``json_extract`` of an object/array path returns its JSON text (a
        ``str``), which stays hashable as a unit-key component — so two rows
        with the same object value share a unit and a row with a different
        object value is distinct, all without error.
        """
        # Two rows share an identical nested-object value; one differs.
        await store.create_thought(
            _thought("o1", essence="psi word", metadata={"unit": {"s": "a", "t": 1}})
        )
        await store.create_thought(
            _thought("o2", essence="psi word", metadata={"unit": {"s": "a", "t": 1}})
        )
        await store.create_thought(
            _thought("o3", essence="psi word", metadata={"unit": {"s": "b", "t": 2}})
        )
        # An array-valued key, injected as valid JSON to exercise the array path.
        await store.create_thought(_thought("arr1", essence="psi word", metadata={}))
        await store.create_thought(_thought("arr2", essence="psi word", metadata={}))
        for tid in ("arr1", "arr2"):
            await store._db.execute(
                "UPDATE thought SET metadata_json = ? WHERE thought_id = ?",
                ('{"unit": [1, 2, 3]}', tid),
            )
        await store._db.commit()

        result = await store.search_hybrid("psi", top_k=10, collapse_key="$.unit")
        returned = set(_ids(result.results))
        # {s:a,t:1} collapses to one of {o1,o2}; {s:b,t:2}=o3 is distinct;
        # [1,2,3] collapses arr1/arr2 to one survivor.
        assert "o3" in returned
        assert len(returned & {"o1", "o2"}) == 1
        assert len(returned & {"arr1", "arr2"}) == 1


# ---------------------------------------------------------------------------
# Stage ordering: collapse x reflection_topk_cap (A6)
# ---------------------------------------------------------------------------


class TestCollapseReflectionCapInteraction:
    """Collapse runs before the cap; the cap backfills from the collapsed pool."""

    async def test_cap_backfills_from_collapsed_off_list(self, store: SqliteEngravaCore) -> None:
        """No unit is double-counted across collapse + cap; cap still enforced."""
        # Many REFLECTION fragments of one unit (would flood top_k without collapse),
        # plus distinct OBSERVATION units to backfill into the freed slots.
        for i in range(6):
            await store.create_thought(
                _thought(
                    f"refl-{i}",
                    essence="omicron omicron summary",
                    thought_type=ThoughtType.REFLECTION,
                    metadata={"unit": "r1"},
                )
            )
        for u in range(1, 6):
            await store.create_thought(
                _thought(f"obs{u}", essence="omicron summary", metadata={"unit": f"o{u}"})
            )
        # cap = 0.3 with top_k=5 -> at most 1 reflection slot.
        result = await store.search_hybrid(
            "omicron summary",
            top_k=5,
            collapse_key="$.unit",
        )
        returned = _ids(result.results)
        refl_in_final = [tid for tid in returned if tid.startswith("refl-")]
        obs_in_final = [tid for tid in returned if tid.startswith("obs")]
        # The window is filled via backfill, not starved by collapse + cap.
        assert len(returned) == 5
        # At most one reflection (collapse leaves one r1 keeper; cap allows <=1).
        assert len(refl_in_final) <= 1
        # Backfill sources are DISTINCT observation units — never the collapsed-off
        # r1 fragments (only the single r1 keeper can appear as a reflection).
        assert len(obs_in_final) == len(set(obs_in_final))
        assert len(refl_in_final) + len(obs_in_final) == 5
        # No duplicate ids (no double-count across the two backfill stages).
        assert len(returned) == len(set(returned))


# ---------------------------------------------------------------------------
# Determinism -- criterion A7
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Equal-score unit members yield a canonical, scan-order-invariant keeper."""

    async def test_equal_score_canonical_keeper_stable(self, store: SqliteEngravaCore) -> None:
        """Tied fragments of a unit collapse to the canonical (id-asc) keeper, stably."""
        ids = [f"frag-{i:02d}" for i in range(10)]
        # Insert in shuffled order so physical scan order != id order.
        for tid in [ids[7], ids[2], ids[9], ids[0], *ids]:
            try:
                await store.create_thought(
                    _thought(tid, essence="pi-word", metadata={"unit": "u1"})
                )
            except ValueError:
                continue
        first = await store.search_hybrid("pi-word", top_k=10, collapse_key="$.unit")
        second = await store.search_hybrid("pi-word", top_k=10, collapse_key="$.unit")
        assert _ids(first.results) == _ids(second.results)
        # One survivor, and it is the canonical lowest id (all scores equal).
        assert _ids(first.results) == ["frag-00"]


# ---------------------------------------------------------------------------
# Bounded pool widening (A8)
# ---------------------------------------------------------------------------


class TestPoolWidening:
    """Widening is bounded and confined to the collapse path.

    Uses the exhaustive vector arm with hand-crafted embeddings so the rank of
    each candidate is fully controlled (no BM25 length-normalization surprises)
    and the only variable is the per-arm ``vector_top_k`` budget.
    """

    @staticmethod
    async def _seed(s: SqliteEngravaCore) -> None:
        """3 strong same-unit fragments (rank 1-3) + 1 deeper distinct unit (rank 4)."""
        # Cosine-to-query [1,0,0,0]: u1 frags ~ {1.0, 0.99, 0.98}; deep ~ 0.90.
        frags = {
            "u1-0": [1.0, 0.0, 0.0, 0.0],
            "u1-1": [0.99, 0.14, 0.0, 0.0],
            "u1-2": [0.98, 0.20, 0.0, 0.0],
        }
        for tid, vec in frags.items():
            await s.create_thought(_thought(tid, essence=f"row {tid}", metadata={"unit": "u1"}))
            await s.store_embedding(tid, vec)
        await s.create_thought(_thought("deep", essence="row deep", metadata={"unit": "u2"}))
        await s.store_embedding("deep", [0.90, 0.44, 0.0, 0.0])

    async def test_widening_enables_deeper_distinct_unit_backfill(self, tmp_path: Path) -> None:
        """A distinct unit past the un-widened budget is backfilled once widened.

        ``vector_top_k=2`` un-widened sees only the 2 strongest u1 fragments;
        the ``x4`` widening (=> 8) pulls the rank-4 ``deep`` row into the pool,
        and collapse then surfaces it as a distinct unit.
        """
        conn = await aiosqlite.connect(str(tmp_path / "widen.db"))
        conn.row_factory = aiosqlite.Row
        s = SqliteEngravaCore(conn, search_config=SearchConfig(collapse_pool_factor=4))
        await s.ensure_schema()
        try:
            await self._seed(s)
            result = await s.search_hybrid(
                "no-fts-match",
                query_vector=[1.0, 0.0, 0.0, 0.0],
                top_k=5,
                fts_top_k=2,
                vector_top_k=2,
                collapse_key="$.unit",
            )
            assert "deep" in set(_ids(result.results))
        finally:
            await conn.close()

    async def test_none_path_pool_not_widened(self, tmp_path: Path) -> None:
        """Without collapse_key the arm budget is NOT widened (deep row stays out).

        Same store and budget; with ``collapse_key=None`` the budget stays at
        ``vector_top_k=2``, so the rank-4 ``deep`` row never enters the pool.
        """
        conn = await aiosqlite.connect(str(tmp_path / "nowiden.db"))
        conn.row_factory = aiosqlite.Row
        s = SqliteEngravaCore(conn, search_config=SearchConfig(collapse_pool_factor=4))
        await s.ensure_schema()
        try:
            await self._seed(s)
            result = await s.search_hybrid(
                "no-fts-match",
                query_vector=[1.0, 0.0, 0.0, 0.0],
                top_k=5,
                fts_top_k=2,
                vector_top_k=2,
                collapse_key=None,
            )
            assert "deep" not in set(_ids(result.results))
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# Composition with metadata filters / visibility -- criterion A9
# ---------------------------------------------------------------------------


class TestCompositionWithFilters:
    """Eligibility (filters/visibility) runs first, then collapse."""

    async def test_collapse_with_metadata_filter(self, store: SqliteEngravaCore) -> None:
        """Out-of-filter rows never reach collapse; in-filter unit collapses."""
        # In-filter: two fragments of one unit + one distinct unit.
        await store.create_thought(
            _thought("in-a", essence="tau word", metadata={"project": "x", "unit": "u1"})
        )
        await store.create_thought(
            _thought("in-b", essence="tau word", metadata={"project": "x", "unit": "u1"})
        )
        await store.create_thought(
            _thought("in-c", essence="tau word", metadata={"project": "x", "unit": "u2"})
        )
        # Out-of-filter row sharing the unit must never appear.
        await store.create_thought(
            _thought("out", essence="tau word", metadata={"project": "y", "unit": "u1"})
        )
        result = await store.search_hybrid(
            "tau",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
            collapse_key="$.unit",
        )
        returned = set(_ids(result.results))
        assert "out" not in returned
        assert "in-c" in returned
        assert len(returned & {"in-a", "in-b"}) == 1

    async def test_collapse_with_visibility(self, store: SqliteEngravaCore) -> None:
        """collapse_key composes with a visibility filter (eligibility first)."""
        await store.create_thought(
            _thought(
                "pub-a",
                essence="upsilon word",
                metadata={"visibility": "public", "unit": "u1"},
            )
        )
        await store.create_thought(
            _thought(
                "pub-b",
                essence="upsilon word",
                metadata={"visibility": "public", "unit": "u1"},
            )
        )
        await store.create_thought(
            _thought(
                "priv",
                essence="upsilon word",
                metadata={"visibility": "private", "unit": "u2"},
            )
        )
        result = await store.search_hybrid(
            "upsilon",
            top_k=10,
            visibility=VisibilityQueryFilter({"public"}),
            collapse_key="$.unit",
        )
        returned = set(_ids(result.results))
        assert "priv" not in returned
        assert len(returned & {"pub-a", "pub-b"}) == 1


# ---------------------------------------------------------------------------
# Expansion-path interaction (collapse sees the expanded candidate set)
# ---------------------------------------------------------------------------


class TestCollapseAfterExpansion:
    """Collapse runs after CONSOLIDATED_FROM expansion (A4 locus)."""

    async def test_expanded_sources_collapse_by_unit(self, store: SqliteEngravaCore) -> None:
        """Two expansion-pulled sources of the same unit collapse to one."""
        refl = _thought(
            "refl",
            essence="phi summary cluster",
            thought_type=ThoughtType.REFLECTION,
            metadata={"unit": "ru"},
        )
        await store.create_thought(refl)
        # Two sources share a unit; only one should survive collapse.
        await store.create_thought(
            _thought("src-a", essence="unrelated source text a", metadata={"unit": "su"})
        )
        await store.create_thought(
            _thought("src-b", essence="unrelated source text b", metadata={"unit": "su"})
        )
        await store.create_edge(_edge("refl", "src-a", weight=0.9))
        await store.create_edge(_edge("refl", "src-b", weight=0.9))
        result = await store.search_hybrid("phi", top_k=10, collapse_key="$.unit")
        returned = _ids(result.results)
        assert "graph_expansion" in result.backends_used
        assert len([tid for tid in returned if tid.startswith("src-")]) == 1

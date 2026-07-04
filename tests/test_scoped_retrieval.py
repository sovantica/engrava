"""Integration tests for scoped / metadata-filtered ranked retrieval.

Exercises ``search_hybrid`` / ``recall`` with ``filters=`` / ``visibility=``
against a real SQLite store. Covers the eligibility invariant (including the
CONSOLIDATED_FROM expansion-injection path), no-starvation, determinism,
typed semantics, and visibility composition / precedence.
"""

from __future__ import annotations

import importlib.util
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
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.filters import (
    FieldOp,
    FieldPredicate,
    MetadataFilter,
    VisibilityQueryFilter,
)
from engrava.domain.models.thought import MetadataValue, ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# A small fixed embedding dimension for hand-crafted, deterministic vectors.
_DIM = 4


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
    """Fresh in-file store with expansion enabled (default SearchConfig)."""
    conn = await aiosqlite.connect(str(tmp_path / "scoped.db"))
    conn.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(conn, search_config=SearchConfig())
    await s.ensure_schema()
    yield s
    await conn.close()


# ---------------------------------------------------------------------------
# API surface + parity for the unfiltered path
# ---------------------------------------------------------------------------


class TestApiSurface:
    """filters=/visibility= exist and default-None preserves the pipeline."""

    async def test_search_hybrid_accepts_filters_and_visibility(
        self, store: SqliteEngravaCore
    ) -> None:
        """Both keyword-only params exist and accept the filter objects."""
        await store.create_thought(_thought("t1", essence="alpha apple", metadata={"project": "x"}))
        result = await store.search_hybrid(
            "alpha",
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
            visibility=None,
        )
        assert _ids(result.results) == ["t1"]

    async def test_recall_delegates_filters(self, store: SqliteEngravaCore) -> None:
        """recall forwards filters= to search_hybrid."""
        await store.create_thought(
            _thought("keep", essence="beta banana", metadata={"project": "x"})
        )
        await store.create_thought(
            _thought("drop", essence="beta banana", metadata={"project": "y"})
        )
        result = await store.recall(
            "beta",
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert _ids(result.results) == ["keep"]

    async def test_none_none_parity_candidate_membership(self, store: SqliteEngravaCore) -> None:
        """filters=None/visibility=None returns the same candidate set as omitting them."""
        for i in range(5):
            await store.create_thought(_thought(f"t{i}", essence=f"gamma grape {i}"))
        baseline = await store.search_hybrid("gamma", top_k=10)
        explicit_none = await store.search_hybrid("gamma", top_k=10, filters=None, visibility=None)
        assert set(_ids(baseline.results)) == set(_ids(explicit_none.results))


# ---------------------------------------------------------------------------
# Eligibility invariant (FTS arm + vector arm + consolidation-expansion path)
# ---------------------------------------------------------------------------


class TestEligibilityInvariant:
    """No out-of-filter row is ever returned via any candidate-producing path."""

    async def test_fts_arm_excludes_out_of_filter_rows(self, store: SqliteEngravaCore) -> None:
        """An out-of-filter row matching the query text is never returned (FTS arm)."""
        await store.create_thought(_thought("in", essence="delta donut", metadata={"project": "x"}))
        await store.create_thought(
            _thought("out", essence="delta donut", metadata={"project": "y"})
        )
        result = await store.search_hybrid(
            "delta",
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert _ids(result.results) == ["in"]

    async def test_vector_arm_excludes_out_of_filter_rows(self, store: SqliteEngravaCore) -> None:
        """An out-of-filter row that would rank highly by cosine is excluded (vector arm).

        The exhaustive numpy vector arm is exercised by supplying an explicit
        query vector with no FTS-matching text, so only the vector arm
        contributes candidates.
        """
        # Two rows with the SAME embedding: one in-filter, one out-of-filter.
        for tid, project in (("vin", "x"), ("vout", "y")):
            await store.create_thought(
                _thought(tid, essence=f"vector row {tid}", metadata={"project": project})
            )
            await store.store_embedding(tid, [1.0, 0.0, 0.0, 0.0])
        result = await store.search_hybrid(
            "zzz-no-fts-match",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert _ids(result.results) == ["vin"]

    async def test_expansion_path_excludes_out_of_filter_observation(
        self, store: SqliteEngravaCore
    ) -> None:
        """A CONSOLIDATED_FROM-pulled OBSERVATION outside the filter is never injected.

        Construct a REFLECTION that ranks into top_k and whose source
        OBSERVATIONs split across the filter. The out-of-filter source would
        otherwise be pulled into the result set by the expansion stage; the
        re-applied predicate must keep it out, while the in-filter source is
        still expanded in.
        """
        # The REFLECTION matches the query text and carries the in-filter tag
        # so it survives the arm's WHERE and seeds expansion.
        refl = _thought(
            "refl",
            essence="epsilon summary cluster",
            thought_type=ThoughtType.REFLECTION,
            metadata={"project": "x"},
        )
        await store.create_thought(refl)
        # Sources do NOT match the query text — they can only enter via expansion.
        src_in = _thought("src-in", essence="unrelated factual text in", metadata={"project": "x"})
        src_out = _thought(
            "src-out", essence="unrelated factual text out", metadata={"project": "y"}
        )
        await store.create_thought(src_in)
        await store.create_thought(src_out)
        await store.create_edge(_edge("refl", "src-in", weight=0.9))
        await store.create_edge(_edge("refl", "src-out", weight=0.9))

        result = await store.search_hybrid(
            "epsilon",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        returned = set(_ids(result.results))
        assert "src-out" not in returned, "out-of-filter expansion source leaked"
        # Sanity: the in-filter source IS expanded in (so the test proves the
        # predicate, not merely that expansion was disabled).
        assert "src-in" in returned
        assert "graph_expansion" in result.backends_used

    async def test_expansion_injection_direct_and_via_expansion(
        self, store: SqliteEngravaCore
    ) -> None:
        """A single out-of-filter row that could enter both directly and via expansion stays out.

        ``shared`` matches the query text (direct arm candidate) AND is a
        CONSOLIDATED_FROM source of a matching REFLECTION (expansion
        candidate). It is out-of-filter, so neither path may surface it.
        """
        await store.create_thought(
            _thought(
                "refl",
                essence="zeta summary",
                thought_type=ThoughtType.REFLECTION,
                metadata={"project": "x"},
            )
        )
        await store.create_thought(
            _thought("shared", essence="zeta shared donut", metadata={"project": "y"})
        )
        await store.create_edge(_edge("refl", "shared", weight=0.9))

        result = await store.search_hybrid(
            "zeta",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert "shared" not in set(_ids(result.results))


# ---------------------------------------------------------------------------
# Fallback arm eligibility (query-less / unindexable, no vector)
# ---------------------------------------------------------------------------


class TestFallbackArmEligibility:
    """The query-less / unindexable fallback arm honours filters=/visibility=.

    When neither the FTS arm nor the vector arm is active — an empty or
    unindexable ``query_text`` with no ``query_vector`` and no embedding
    provider (e.g. a recency-only scoped browse) — ``search_hybrid`` falls back
    to a plain lifecycle-ordered scan. That fallback must apply the same
    ``filters=`` / ``visibility=`` eligibility predicate as the main arms: the
    docstring promises an out-of-filter row *never* enters the result set, with
    no exemption for the fallback path.
    """

    async def test_fallback_path_applies_filters(self, store: SqliteEngravaCore) -> None:
        """An empty-query recency-only call still excludes out-of-filter rows."""
        await store.create_thought(_thought("in", essence="one", metadata={"project": "x"}))
        await store.create_thought(_thought("out", essence="two", metadata={"project": "y"}))
        # Empty query + no provider -> neither FTS nor vector arm active -> fallback.
        result = await store.search_hybrid(
            "",
            top_k=10,
            current_cycle=1,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert _ids(result.results) == ["in"]

    async def test_fallback_path_applies_visibility(self, store: SqliteEngravaCore) -> None:
        """The fallback honours the public-or-mine visibility group too."""
        await store.create_thought(_thought("pub", essence="a", metadata={"visibility": "public"}))
        await store.create_thought(
            _thought("mine", essence="b", metadata={"visibility": "private", "owner": "alice"})
        )
        await store.create_thought(
            _thought("theirs", essence="c", metadata={"visibility": "private", "owner": "bob"})
        )
        result = await store.search_hybrid(
            "",
            top_k=10,
            current_cycle=1,
            visibility=VisibilityQueryFilter({"public"}, owner="alice"),
        )
        assert set(_ids(result.results)) == {"pub", "mine"}


# ---------------------------------------------------------------------------
# No starvation / refill
# ---------------------------------------------------------------------------


class TestNoStarvation:
    """A narrow filter returns up to top_k in-filter rows, never starved."""

    async def test_narrow_filter_not_starved_by_out_of_filter_candidates(
        self, store: SqliteEngravaCore
    ) -> None:
        """top_k is honoured within the filter even when out-of-filter rows are plentiful.

        Many out-of-filter rows share the query term; a small set of in-filter
        rows must all be returned (up to top_k), not displaced from the arm
        budget by the out-of-filter majority.
        """
        for i in range(40):
            await store.create_thought(
                _thought(f"out{i}", essence="eta keyword filler", metadata={"project": "y"})
            )
        for i in range(3):
            await store.create_thought(
                _thought(f"in{i}", essence="eta keyword target", metadata={"project": "x"})
            )
        result = await store.search_hybrid(
            "eta",
            top_k=3,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert set(_ids(result.results)) == {"in0", "in1", "in2"}

    async def test_returns_min_top_k_and_matches(self, store: SqliteEngravaCore) -> None:
        """A filter narrower than top_k returns exactly the matching rows."""
        await store.create_thought(
            _thought("only", essence="theta keyword", metadata={"project": "x"})
        )
        for i in range(10):
            await store.create_thought(
                _thought(f"o{i}", essence="theta keyword", metadata={"project": "y"})
            )
        result = await store.search_hybrid(
            "theta",
            top_k=5,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert _ids(result.results) == ["only"]


# ---------------------------------------------------------------------------
# Determinism (canonical thought_id tie-break)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Equal-score sets return a stable canonical order across runs."""

    async def test_equal_score_stable_canonical_order(self, store: SqliteEngravaCore) -> None:
        """Rows with identical FTS scores order by thought_id ascending, stably.

        All rows share the exact same single-token essence, so BM25 assigns
        them equal scores; the only tie-break is canonical thought_id.
        """
        ids = [f"id-{i:02d}" for i in range(12)]
        # Insert in a shuffled order so physical scan order != id order.
        for tid in [ids[7], ids[0], ids[11], ids[3], ids[5], *ids]:
            try:
                await store.create_thought(_thought(tid, essence="iotaword"))
            except ValueError:
                # Duplicate id from the shuffled prefix — already inserted.
                continue

        first = await store.search_hybrid("iotaword", top_k=12)
        second = await store.search_hybrid("iotaword", top_k=12)
        assert _ids(first.results) == _ids(second.results)
        # The returned ids must be in ascending canonical order (equal scores).
        returned = _ids(first.results)
        assert returned == sorted(returned)

    async def test_determinism_holds_with_filter(self, store: SqliteEngravaCore) -> None:
        """The canonical tie-break also applies on the filtered path."""
        for i in range(8):
            await store.create_thought(
                _thought(f"k-{i:02d}", essence="kappaword", metadata={"project": "x"})
            )
        flt = MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")])
        a = await store.search_hybrid("kappaword", top_k=8, filters=flt)
        b = await store.search_hybrid("kappaword", top_k=8, filters=flt)
        assert _ids(a.results) == _ids(b.results)
        assert _ids(a.results) == sorted(_ids(a.results))


# ---------------------------------------------------------------------------
# Typed semantics
# ---------------------------------------------------------------------------


class TestTypedSemantics:
    """EQ None, large IN, bool/int aliasing, malformed JSON."""

    async def test_eq_none_matches_missing_and_json_null(self, store: SqliteEngravaCore) -> None:
        """EQ None matches both a missing key and an explicit JSON null."""
        await store.create_thought(_thought("missing", essence="lambda word", metadata={}))
        await store.create_thought(
            _thought("explicit-null", essence="lambda word", metadata={"team": None})
        )
        await store.create_thought(
            _thought("has-value", essence="lambda word", metadata={"team": "blue"})
        )
        result = await store.search_hybrid(
            "lambda",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.team", FieldOp.EQ, None)]),
        )
        assert set(_ids(result.results)) == {"missing", "explicit-null"}

    async def test_large_in_well_past_999(self, store: SqliteEngravaCore) -> None:
        """A large IN collection (well past SQLite's 999 bind-variable limit) works.

        Bound as a single JSON-array parameter via json_each, so 2000 elements
        cost one parameter, not 2000 placeholders.
        """
        await store.create_thought(_thought("hit", essence="mu word", metadata={"n": 1500}))
        await store.create_thought(_thought("miss", essence="mu word", metadata={"n": 99999}))
        big_in = list(range(2000))  # 0..1999 includes 1500, excludes 99999
        result = await store.search_hybrid(
            "mu",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.n", FieldOp.IN, big_in)]),
        )
        assert _ids(result.results) == ["hit"]

    async def test_bool_int_aliasing(self, store: SqliteEngravaCore) -> None:
        """EQ True aliases EQ 1 under SQLite value equality (documented).

        A row written with metadata ``{"flag": True}`` stores ``flag`` as the
        JSON integer ``1``; ``EQ True`` and ``EQ 1`` both match it.
        """
        await store.create_thought(_thought("flagged", essence="nu word", metadata={"flag": True}))
        await store.create_thought(
            _thought("unflagged", essence="nu word", metadata={"flag": False})
        )
        by_true = await store.search_hybrid(
            "nu",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.flag", FieldOp.EQ, True)]),
        )
        by_one = await store.search_hybrid(
            "nu",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.flag", FieldOp.EQ, 1)]),
        )
        assert _ids(by_true.results) == ["flagged"]
        assert _ids(by_one.results) == ["flagged"]

    async def test_malformed_json_row_does_not_match_and_does_not_abort(
        self, store: SqliteEngravaCore
    ) -> None:
        """A row holding invalid JSON in metadata_json is non-matching; the query still runs.

        ``metadata_json`` is library-written, so the malformed value is
        injected via raw SQL to exercise the json_valid guard.
        """
        await store.create_thought(_thought("good", essence="xi word", metadata={"project": "x"}))
        await store.create_thought(_thought("bad", essence="xi word", metadata={"project": "x"}))
        # Corrupt the 'bad' row's metadata_json directly.
        await store._db.execute(
            "UPDATE thought SET metadata_json = ? WHERE thought_id = ?",
            ("{not valid json", "bad"),
        )
        await store._db.commit()

        # EQ None would wrongly match a malformed row if the guard wrapped only
        # the extracted value; it must not.
        none_result = await store.search_hybrid(
            "xi",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, None)]),
        )
        assert "bad" not in set(_ids(none_result.results))

        # And a positive filter returns the good row without aborting.
        eq_result = await store.search_hybrid(
            "xi",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
        )
        assert _ids(eq_result.results) == ["good"]


# ---------------------------------------------------------------------------
# Visibility composition + precedence
# ---------------------------------------------------------------------------


class TestVisibilityComposition:
    """public-or-mine + the AND-cannot-be-escaped precedence guarantee."""

    async def test_public_or_mine_returns_both(self, store: SqliteEngravaCore) -> None:
        """visibility=(allowed={public}, owner=alice) returns public rows AND alice's rows."""
        await store.create_thought(
            _thought("pub", essence="omicron word", metadata={"visibility": "public"})
        )
        await store.create_thought(
            _thought(
                "mine", essence="omicron word", metadata={"visibility": "private", "owner": "alice"}
            )
        )
        await store.create_thought(
            _thought(
                "theirs",
                essence="omicron word",
                metadata={"visibility": "private", "owner": "bob"},
            )
        )
        result = await store.search_hybrid(
            "omicron",
            top_k=10,
            visibility=VisibilityQueryFilter({"public"}, owner="alice"),
        )
        assert set(_ids(result.results)) == {"pub", "mine"}

    async def test_precedence_owner_match_outside_filters_not_returned(
        self, store: SqliteEngravaCore
    ) -> None:
        """The visibility OR cannot escape the metadata AND.

        A row owned by alice but outside ``filters=`` must NOT be returned:
        the effective predicate is ``P_filters AND ( visibility-group )``, so
        an owner match cannot bypass the metadata filter.
        """
        # In-filter + owned by alice -> returned.
        await store.create_thought(
            _thought(
                "in-mine",
                essence="pi word",
                metadata={"project": "x", "visibility": "private", "owner": "alice"},
            )
        )
        # Owned by alice but OUT of filter -> must be excluded by the AND.
        await store.create_thought(
            _thought(
                "out-mine",
                essence="pi word",
                metadata={"project": "y", "visibility": "private", "owner": "alice"},
            )
        )
        result = await store.search_hybrid(
            "pi",
            top_k=10,
            filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
            visibility=VisibilityQueryFilter({"public"}, owner="alice"),
        )
        returned = set(_ids(result.results))
        assert "out-mine" not in returned, "owner match escaped the metadata AND"
        assert returned == {"in-mine"}

    async def test_visibility_in_uses_json_each_for_many_values(
        self, store: SqliteEngravaCore
    ) -> None:
        """A large allowed-set binds via json_each (single parameter)."""
        await store.create_thought(
            _thought("v5", essence="rho word", metadata={"visibility": "level-5"})
        )
        await store.create_thought(
            _thought("vx", essence="rho word", metadata={"visibility": "excluded"})
        )
        allowed = {f"level-{i}" for i in range(1500)}
        result = await store.search_hybrid(
            "rho",
            top_k=10,
            visibility=VisibilityQueryFilter(allowed),
        )
        assert _ids(result.results) == ["v5"]


# ---------------------------------------------------------------------------
# Vector arm with the sqlite-vec backend: a filter must bypass the
# LIMIT-before-cosine KNN path (which cannot pre-filter on metadata).
# ---------------------------------------------------------------------------


sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)


@sqlite_vec_required
class TestVectorArmFilterBypassesKnnBackend:
    """With a vec0 backend configured, a filter forces the exhaustive numpy path."""

    async def _build_vec_store(self, tmp_path: Path, *, dimension: int) -> SqliteEngravaCore:
        """Build a store with a live sqlite-vec backend (mirrors from_config)."""
        db = await aiosqlite.connect(str(tmp_path / "vec.db"))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        store = SqliteEngravaCore(db, search_config=SearchConfig())
        store._owns_connection = True
        await store.ensure_schema()
        await store._configure_vector_backend(
            backend_name="sqlite-vec",
            embedding_dimension=dimension,
        )
        return store

    async def test_filtered_vector_search_excludes_out_of_filter_rows(self, tmp_path: Path) -> None:
        """An out-of-filter row is excluded even though a vec0 KNN backend is active.

        The vec0 ``MATCH`` query applies a ``LIMIT`` before any metadata
        predicate could run, so a filtered query must bypass it and use the
        exhaustive numpy arm. Two rows share the top embedding; only the
        in-filter one may be returned.
        """
        store = await self._build_vec_store(tmp_path, dimension=_DIM)
        try:
            assert store._vector_backend is not None  # backend really is live
            for tid, project in (("vin", "x"), ("vout", "y")):
                await store.create_thought(
                    _thought(tid, essence=f"sigma {tid}", metadata={"project": project})
                )
                await store.store_embedding(tid, [1.0, 0.0, 0.0, 0.0])
            result = await store.search_hybrid(
                "no-fts-here",
                query_vector=[1.0, 0.0, 0.0, 0.0],
                filters=MetadataFilter([FieldPredicate("$.project", FieldOp.EQ, "x")]),
            )
            assert _ids(result.results) == ["vin"]
        finally:
            await store.close()

    async def test_unfiltered_vector_search_uses_backend(self, tmp_path: Path) -> None:
        """Without a filter the vec0 backend still serves the query (no bypass)."""
        store = await self._build_vec_store(tmp_path, dimension=_DIM)
        try:
            for tid, vec in (("near", [1.0, 0.0, 0.0, 0.0]), ("far", [-1.0, 0.0, 0.0, 0.0])):
                await store.create_thought(_thought(tid, essence=f"tau {tid}"))
                await store.store_embedding(tid, vec)
            result = await store.search_similar([1.0, 0.0, 0.0, 0.0], top_k=2)
            assert result[0][0] == "near"
        finally:
            await store.close()

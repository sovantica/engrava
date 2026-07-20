"""Cross-backend ranking-parity invariant for the two vector search backends.

``search_similar`` has two interchangeable vector arms:

* the **numpy** brute-force arm (the default), which filters eligible rows in
  the SQL ``WHERE`` *before* computing cosine and applying top-k; and
* the **sqlite-vec** ``vec0`` arm, which applies its ``k``/``LIMIT`` *first*
  and only then drops expired thoughts and retired REFLECTIONs via a
  post-``MATCH`` live-row filter.

Because numpy is the default, a vec0-only ranking bug is invisible to every
numpy-default test. This module pins the invariant directly: the same query
run through *both* backends over an identical stored corpus must return the
same ranked ``thought_id`` order and near-identical scores, and must do so
*after* the live-row filter (the two arms apply that filter at different
stages, so raw-KNN parity would not prove it).

The score tolerance is a small ``atol`` rather than a bitwise equality on
purpose: the two cosine paths differ numerically (vec0 computes in float32;
the numpy arm widens the same stored float32 bytes to float64), so the scores
are close but not bit-identical. A bitwise assertion would be flaky.

Skips cleanly when ``sqlite-vec`` is absent, mirroring ``test_sqlite_vec.py``.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import aiosqlite
import numpy as np
import pytest

from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.vector_sqlite_vec import SqliteVecSearchBackend
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from pathlib import Path

# Skip the real-extension parity tests when sqlite-vec is absent, but never let
# them silently pass when it is installed — mirrors ``test_sqlite_vec.py``.
sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)

_PARITY_MODEL = "test-fixture-model"
_DIMENSION = 3
#: Measured max vec0-vs-numpy score divergence on this corpus is ~1.2e-8
#: (float32 cosine distance vs float64 dot/norm). ``1e-6`` keeps a wide margin
#: while staying tight enough to catch a real conversion regression. Never
#: bitwise — the two cosine paths are numerically distinct.
_SCORE_ATOL = 1e-6


_Kind = Literal["live", "expired", "retired"]


@dataclass(frozen=True)
class _CorpusRow:
    """A single seeded thought: its id, stored embedding, and liveness kind."""

    thought_id: str
    vector: tuple[float, ...]
    kind: _Kind


#: Query on the ``x = y`` diagonal so ``t-live-a`` ([1,0,0]) and ``t-live-b``
#: ([0,1,0]) are *exactly* equidistant — a deliberate cosine tie.
_QUERY: tuple[float, float, float] = (1.0, 1.0, 0.0)

#: Deterministic corpus. Insertion order is deliberate: the two non-live rows
#: are inserted first (lowest rowids, so a naive scan surfaces them at the very
#: top), and ``t-live-b`` is inserted before ``t-live-a`` so physical scan order
#: differs from canonical id order. The non-live rows sit at/above the top of
#: the raw KNN (``t-exp`` is cosine-identical to the best live row), so the
#: live-row filter is load-bearing: unfiltered, they would corrupt the ranking.
_CORPUS: tuple[_CorpusRow, ...] = (
    _CorpusRow("t-exp", (1.0, 1.0, 0.0), "expired"),  # cosine 1.0 — would top the list
    _CorpusRow("t-ret", (0.95, 1.0, 0.0), "retired"),  # retired REFLECTION, near top
    _CorpusRow("t-live-c", (1.0, 1.0, 0.0), "live"),  # cosine 1.0 — the top live row
    _CorpusRow("t-live-d", (1.0, 0.5, 0.0), "live"),  # near neighbour
    _CorpusRow("t-live-b", (0.0, 1.0, 0.0), "live"),  # tie partner (inserted before a)
    _CorpusRow("t-live-a", (1.0, 0.0, 0.0), "live"),  # tie partner
    _CorpusRow("t-live-e", (0.1, 0.0, 1.0), "live"),  # far but above threshold
)

#: The live ranking both backends must agree on (score DESC, ties id ASC).
_EXPECTED_LIVE_ORDER: tuple[str, ...] = (
    "t-live-c",
    "t-live-d",
    "t-live-a",
    "t-live-b",
    "t-live-e",
)
#: The exact-tie pair, in the canonical id-ascending order the tie resolves to.
_TIE_IDS: tuple[str, str] = ("t-live-a", "t-live-b")
#: Rows the live-row filter must exclude on *both* backends.
_NON_LIVE_IDS: frozenset[str] = frozenset({"t-exp", "t-ret"})
#: Search depth = the number of live rows, so a correct run returns them all.
_TOP_K = sum(1 for row in _CORPUS if row.kind == "live")


def _past_iso() -> str:
    """Return an ISO-8601 UTC timestamp one day in the past (already expired)."""
    return (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).isoformat()


def _cosine(query: tuple[float, ...], vector: tuple[float, ...]) -> float:
    """Cosine similarity in float64, mirroring the numpy arm's arithmetic."""
    q = np.asarray(query, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q)))


def _assert_ranking_parity(
    numpy_results: list[tuple[str, float]],
    vec_results: list[tuple[str, float]],
    *,
    atol: float,
) -> None:
    """Assert both backends return the same ranked ids and near-equal scores.

    Order is checked first so a genuine *ranking* divergence surfaces as the
    "ranking diverged" failure; scores are then compared within ``atol``
    (never bitwise).
    """
    numpy_order = [tid for tid, _ in numpy_results]
    vec_order = [tid for tid, _ in vec_results]
    assert vec_order == numpy_order, f"ranking diverged: vec0={vec_order} numpy={numpy_order}"
    for (nid, nscore), (vid, vscore) in zip(numpy_results, vec_results, strict=True):
        assert vid == nid
        assert abs(vscore - nscore) <= atol, (
            f"score parity broke at {nid}: numpy={nscore} vec0={vscore} atol={atol}"
        )


async def _make_thought(store: SqliteEngravaCore, row: _CorpusRow) -> None:
    """Create the thought backing a corpus row with the right liveness state."""
    if row.kind == "expired":
        thought_type = ThoughtType.OBSERVATION
        lifecycle = LifecycleStatus.CREATED
        expires_at: str | None = _past_iso()
    elif row.kind == "retired":
        # A retired REFLECTION: archived once its cluster left the active set.
        thought_type = ThoughtType.REFLECTION
        lifecycle = LifecycleStatus.ARCHIVED
        expires_at = None
    else:
        thought_type = ThoughtType.OBSERVATION
        lifecycle = LifecycleStatus.CREATED
        expires_at = None
    await store.create_thought(
        ThoughtRecord(
            thought_id=row.thought_id,
            thought_type=thought_type,
            essence=f"essence {row.thought_id}",
            content=f"content {row.thought_id}",
            priority=Priority.P3,
            lifecycle_status=lifecycle,
            created_cycle=0,
            updated_cycle=0,
            source="test",
            expires_at=expires_at,
        )
    )


async def _build_store(tmp_path: Path, *, backend: str) -> SqliteEngravaCore:
    """Construct a store with a real connection and the given vector backend."""
    db = await aiosqlite.connect(str(tmp_path / f"{backend}.db"))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(db)
    store._owns_connection = True
    await store.ensure_schema()
    await store._configure_vector_backend(
        backend_name=backend,
        embedding_dimension=_DIMENSION,
    )
    if backend == "sqlite-vec":
        # Fail loudly if the extension failed to *load*. The package-presence
        # skip marker (``importlib.util.find_spec``) only proves sqlite-vec is
        # installed; ``_configure_sqlite_vec_vector_backend`` still silently
        # degrades to the numpy fallback (``_vector_backend -> None``) when the
        # extension cannot load on this connection. Without this guard a load
        # failure would turn the whole comparison into a vacuous numpy-vs-numpy
        # pass, defeating the invariant.
        assert isinstance(store._vector_backend, SqliteVecSearchBackend), (
            "sqlite-vec backend degraded to the numpy fallback — parity would be vacuous"
        )
    return store


async def _collect(tmp_path: Path, backend: str) -> list[tuple[str, float]]:
    """Seed the shared corpus into ``backend`` and return the ranked query hits."""
    store = await _build_store(tmp_path, backend=backend)
    try:
        for row in _CORPUS:
            await _make_thought(store, row)
            await store.store_embedding(
                thought_id=row.thought_id,
                vector=list(row.vector),
                model_name=_PARITY_MODEL,
            )
        return await store.search_similar(list(_QUERY), top_k=_TOP_K)
    finally:
        await store.close()


def test_corpus_exercises_ties_and_live_row_filter() -> None:
    """Meta-test: the corpus really exercises ties + the live-row-filter interaction.

    Runs without sqlite-vec — it only reasons over the known vectors — so the
    guarantees the parity tests rely on are pinned even on a numpy-only host.
    """
    live = [row for row in _CORPUS if row.kind == "live"]
    non_live = [row for row in _CORPUS if row.kind != "live"]

    # (0) Pin the exact shape so a future edit cannot quietly shrink the corpus
    #     back to a trivially-separated set (no tie, no top non-live neighbour)
    #     that any backend would agree on — which would make the parity tests
    #     pass without actually discriminating between the backends.
    assert len(_CORPUS) == 7
    assert len(live) == 5
    assert len(non_live) == 2
    assert {row.thought_id for row in non_live} == set(_NON_LIVE_IDS)
    # Every corpus id is distinct (a duplicate would silently collapse a row).
    assert len({row.thought_id for row in _CORPUS}) == len(_CORPUS)

    # (1) An exact score tie among live rows makes the ranked-order assertion
    #     non-trivial: a tie must be broken deterministically and identically
    #     by both backends (score DESC, then id ASC). Pin the *specific* pair,
    #     not merely "some tie exists".
    tie_scores = {
        row.thought_id: _cosine(_QUERY, row.vector) for row in live if row.thought_id in _TIE_IDS
    }
    assert set(tie_scores) == set(_TIE_IDS)  # both tie ids are live rows
    assert len(set(tie_scores.values())) == 1  # and they are exactly equidistant

    # (2) At least one non-live row is a *top* raw neighbour (cosine >= the best
    #     live row), so the live-row filter is load-bearing: unfiltered, a
    #     non-live row would occupy the top of the ranking.
    live_scores = [_cosine(_QUERY, row.vector) for row in live]
    best_live = max(live_scores)
    assert any(_cosine(_QUERY, row.vector) >= best_live for row in non_live)

    # (3) Both non-live species the filter must drop are present: an expired
    #     thought and a retired (non-ACTIVE) REFLECTION.
    assert {row.kind for row in non_live} == {"expired", "retired"}


@sqlite_vec_required
class TestVec0NumpyRankingParity:
    """The vec0 and numpy arms must rank an identical corpus identically."""

    async def test_backends_agree_on_live_ranking(self, tmp_path: Path) -> None:
        """Same query, same stored embeddings, both backends -> same ranking."""
        numpy_results = await _collect(tmp_path, "numpy")
        vec_results = await _collect(tmp_path, "sqlite-vec")

        # Identical ranked order and scores within a tight (non-bitwise) atol.
        _assert_ranking_parity(numpy_results, vec_results, atol=_SCORE_ATOL)

        # Lock the actual value, not just backend-equality: catches a bug that
        # would corrupt the ranking identically on both arms.
        assert [tid for tid, _ in numpy_results] == list(_EXPECTED_LIVE_ORDER)

        # Parity holds *after* the live-row filter. Both backends dropped the
        # expired thought and the retired REFLECTION even though they apply the
        # filter at different stages (numpy: SQL WHERE before cosine; vec0:
        # post-KNN-LIMIT join) — asserted on the returned rows, not raw KNN.
        numpy_ids = {tid for tid, _ in numpy_results}
        vec_ids = {tid for tid, _ in vec_results}
        assert _NON_LIVE_IDS.isdisjoint(numpy_ids)
        assert _NON_LIVE_IDS.isdisjoint(vec_ids)

        # The exact-tie pair resolves id-ascending on both backends.
        assert [tid for tid, _ in numpy_results if tid in _TIE_IDS] == list(_TIE_IDS)
        assert [tid for tid, _ in vec_results if tid in _TIE_IDS] == list(_TIE_IDS)

    async def test_cosine_conversion_bug_breaks_parity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discriminating power: a vec0-only cosine-conversion bug fails parity.

        Injects the classic "forgot the ``1 - distance`` conversion" mistake
        into the vec0 arm only — cosine *distance* is handed back as the score,
        so ``search_similar`` re-sorts the ranking into reverse order. The numpy
        arm is untouched, exactly the vec0-only regression this invariant exists
        to catch.
        """
        numpy_results = await _collect(tmp_path, "numpy")

        original_search = SqliteVecSearchBackend.search

        async def _distance_as_score(
            backend: SqliteVecSearchBackend,
            db: aiosqlite.Connection,
            query_vector: list[float],
            top_k: int = 10,
            threshold: float = 0.0,
        ) -> list[tuple[str, float]]:
            real = await original_search(backend, db, query_vector, top_k, threshold)
            return [(tid, 1.0 - score) for tid, score in real]

        monkeypatch.setattr(SqliteVecSearchBackend, "search", _distance_as_score)
        vec_results = await _collect(tmp_path, "sqlite-vec")

        with pytest.raises(AssertionError, match="ranking diverged"):
            _assert_ranking_parity(numpy_results, vec_results, atol=_SCORE_ATOL)

    async def test_disabled_live_row_filter_breaks_parity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discriminating power: dropping the vec0 live-row filter fails parity.

        With the vec0 post-KNN filter neutered, the expired thought and retired
        REFLECTION leak into the vec0 ranking while the numpy arm still excludes
        them via its SQL ``WHERE`` — the precise after-filter divergence the
        parity invariant must catch.
        """
        numpy_results = await _collect(tmp_path, "numpy")

        async def _no_filter(
            store: SqliteEngravaCore,
            results: list[tuple[str, float]],
        ) -> list[tuple[str, float]]:
            return results

        monkeypatch.setattr(SqliteEngravaCore, "_filter_expired_results", _no_filter)
        vec_results = await _collect(tmp_path, "sqlite-vec")

        with pytest.raises(AssertionError, match="ranking diverged"):
            _assert_ranking_parity(numpy_results, vec_results, atol=_SCORE_ATOL)

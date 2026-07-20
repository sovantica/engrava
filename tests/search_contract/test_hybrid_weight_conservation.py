"""Weight-conservation invariant for hybrid-search fusion.

``search_hybrid`` blends up to five signals — FTS, vector, recency, priority,
graph — each with a configured weight. When a signal is inactive for a query
(no query text, no vector, recency reference absent, or a zeroed weight), its
weight must be redistributed across the *active* signals rather than dropped, so
the fused scores stay on a stable, comparable scale no matter which arms fired.
:meth:`SqliteEngravaCore._redistribute_hybrid_weights` performs that
redistribution.

:class:`TestWeightConservationTable` enumerates all ``2**5`` active/inactive
combinations and pins two properties over every one:

* **Conservation** — the effective weights of the active signals sum to exactly
  ``1`` (within a float tolerance, never asserted bitwise), so no query silently
  loses or gains total signal mass.
* **Non-degeneracy** — disabling an arm never zeroes the total signal: for any
  combination with at least one active arm the effective weights sum to ``1``;
  only the all-off combination yields a zero total, which is the single case
  ``search_hybrid`` routes to its query-less ``list_thoughts`` fallback.

:class:`TestFallbackOnlyWhenAllOff` pins that routing behaviour end-to-end, and
:class:`TestConservationDiscriminatingPower` shows the conservation assertion has
teeth: an un-normalized redistribution breaks the sum-to-one property.
"""

from __future__ import annotations

import itertools
import math
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# Distinct, non-normalized positive weights (their raw sum is 28, not 1), so the
# redistribution is doing real work: any correct effective-weight sum of 1 can
# only come from dividing by the active mass, never from the inputs already
# summing to 1.
_FTS_W = 2.0
_VEC_W = 3.0
_REC_W = 5.0
_PRI_W = 7.0
_GRA_W = 11.0

# Backend/score conservation is asserted within a tolerance, never bitwise:
# proportional redistribution is exact in rationals but the float divisions
# (e.g. 2/18 + 5/18 + 11/18) can land a unit in the last place off 1.0.
_ATOL = 1e-9

_ARM_NAMES = ("fts", "vector", "recency", "priority", "graph")


def _redistribute(active: tuple[bool, bool, bool, bool, bool]) -> tuple[float, ...]:
    """Run the real redistribution for one active/inactive combination.

    Args:
        active: The ``(fts, vector, recency, priority, graph)`` active flags.

    Returns:
        The 5-tuple of effective weights.
    """
    fts_a, vec_a, rec_a, pri_a, gra_a = active
    return SqliteEngravaCore._redistribute_hybrid_weights(
        fts_active=fts_a,
        vector_active=vec_a,
        recency_active=rec_a,
        priority_active=pri_a,
        graph_active=gra_a,
        fts_weight=_FTS_W,
        vector_weight=_VEC_W,
        recency_weight=_REC_W,
        priority_weight=_PRI_W,
        graph_weight=_GRA_W,
    )


_ALL_COMBOS: tuple[tuple[bool, bool, bool, bool, bool], ...] = tuple(
    itertools.product((False, True), repeat=5)
)


class TestWeightConservationTable:
    """Every one of the 2**5 active/inactive combinations conserves weight mass."""

    def test_enumeration_is_complete(self) -> None:
        """The table is exactly the 32 combinations, none missing or duplicated."""
        assert len(_ALL_COMBOS) == 2**5
        assert len(set(_ALL_COMBOS)) == 2**5

    @pytest.mark.parametrize("active", _ALL_COMBOS)
    def test_active_weights_sum_to_one(
        self,
        active: tuple[bool, bool, bool, bool, bool],
    ) -> None:
        """Active effective weights sum to 1 (all-off is the single 0 exception)."""
        effective = _redistribute(active)
        total = math.fsum(effective)
        if any(active):
            assert math.isclose(total, 1.0, abs_tol=_ATOL), (
                f"effective weights sum to {total!r} != 1 for "
                f"{dict(zip(_ARM_NAMES, active, strict=True))}"
            )
        else:
            assert effective == (0.0, 0.0, 0.0, 0.0, 0.0)
            assert total == 0.0

    @pytest.mark.parametrize("active", _ALL_COMBOS)
    def test_inactive_arms_get_zero_active_arms_get_positive(
        self,
        active: tuple[bool, bool, bool, bool, bool],
    ) -> None:
        """An inactive arm contributes exactly 0; an active arm a positive share."""
        effective = _redistribute(active)
        for is_active, weight, name in zip(active, effective, _ARM_NAMES, strict=True):
            if is_active:
                assert weight > 0.0, f"active {name} arm must carry positive weight for {active}"
            else:
                assert weight == 0.0, f"inactive {name} arm must carry zero weight for {active}"

    @pytest.mark.parametrize("active", _ALL_COMBOS)
    def test_disabling_an_arm_never_zeroes_total_signal(
        self,
        active: tuple[bool, bool, bool, bool, bool],
    ) -> None:
        """Total signal is non-zero for any combo with >=1 active arm; zero only when all off."""
        total = math.fsum(_redistribute(active))
        if any(active):
            assert total > 0.0, f"disabling arms zeroed the total signal for {active}"
        else:
            assert total == 0.0

    def test_single_active_arm_absorbs_all_weight(self) -> None:
        """A lone active arm receives the entire unit weight (1.0), others zero."""
        for index in range(5):
            active = tuple(i == index for i in range(5))
            effective = _redistribute(active)  # type: ignore[arg-type]
            assert effective[index] == pytest.approx(1.0, abs=_ATOL)
            assert math.fsum(effective) == pytest.approx(1.0, abs=_ATOL)


# ---------------------------------------------------------------------------
# End-to-end: all-off is the only combination that falls back to list_thoughts
# ---------------------------------------------------------------------------


def _thought(thought_id: str, content: str) -> ThoughtRecord:
    """Build a stored thought for the fallback-routing tests.

    Args:
        thought_id: Stable identifier.
        content: FTS-indexed text.

    Returns:
        A fully populated :class:`ThoughtRecord`.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=content,
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


@pytest.fixture
async def fts_only_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return an FTS-only store (no embedding provider, so the vector arm is off).

    Yields:
        A :class:`SqliteEngravaCore` with two indexed thoughts.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    await store.create_thought(_thought("t-1", "alpha beta gamma keyword"))
    await store.create_thought(_thought("t-2", "delta epsilon zeta keyword"))
    yield store
    await conn.close()


class TestFallbackOnlyWhenAllOff:
    """``search_hybrid`` routes to the query-less fallback only when every arm is off."""

    async def test_all_off_falls_back(self, fts_only_store: SqliteEngravaCore) -> None:
        """No query text, no vector, all non-lexical weights zero -> fallback, no arms used."""
        result = await fts_only_store.search_hybrid(
            "",
            None,
            top_k=10,
            recency_weight=0.0,
            priority_weight=0.0,
            graph_weight=0.0,
        )
        assert "fts5" not in result.backends_used
        assert "vector" not in result.backends_used
        # The fallback still returns the stored rows (the list_thoughts window).
        assert {tid for tid, _ in result.results} == {"t-1", "t-2"}

    async def test_one_active_arm_does_not_fall_back(
        self,
        fts_only_store: SqliteEngravaCore,
    ) -> None:
        """A live FTS arm keeps hybrid on the fusion path, not the fallback."""
        result = await fts_only_store.search_hybrid("keyword", None, top_k=10)
        assert "fts5" in result.backends_used
        assert {tid for tid, _ in result.results} == {"t-1", "t-2"}


# ---------------------------------------------------------------------------
# Discriminating power — an un-normalized redistribution breaks conservation
# ---------------------------------------------------------------------------


class TestConservationDiscriminatingPower:
    """The sum-to-one assertion is only meaningful if breaking normalization fails it."""

    def test_unnormalized_redistribution_breaks_the_sum(self) -> None:
        """A redistribution that skips the divide-by-active-mass no longer sums to 1.

        The real function normalizes each active weight by the active mass, so a
        two-arm combo sums to 1. A buggy variant that returns the raw weights of
        the active arms (the pre-normalization value) sums to their raw total
        (here ``2 + 3 = 5``), so the conservation assertion the invariant relies
        on fails — proving that assertion discriminates a correct redistribution
        from a broken one rather than passing vacuously.
        """

        def _buggy_identity(active: tuple[bool, bool, bool, bool, bool]) -> tuple[float, ...]:
            weights = (_FTS_W, _VEC_W, _REC_W, _PRI_W, _GRA_W)
            return tuple(w if a else 0.0 for a, w in zip(active, weights, strict=True))

        fts_and_vector = (True, True, False, False, False)

        # The real function conserves mass to 1...
        assert math.isclose(math.fsum(_redistribute(fts_and_vector)), 1.0, abs_tol=_ATOL)
        # ...the un-normalized variant does not, so the invariant would catch it.
        broken_total = math.fsum(_buggy_identity(fts_and_vector))
        assert not math.isclose(broken_total, 1.0, abs_tol=_ATOL)
        assert broken_total == pytest.approx(_FTS_W + _VEC_W)

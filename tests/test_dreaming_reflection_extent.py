"""Unit tests for the REFLECTION valid-time extent helper.

Covers :func:`derive_reflection_extent` — the pure interval-arithmetic
fold that gives a REFLECTION the valid-time extent it inherits from its
cluster members at creation time:

- all members finite → MIN(valid_from) / MAX(valid_until)
- any open lower bound (valid_from is None) → derived valid_from None
- any open upper bound (valid_until is None) → derived valid_until None
- the two axes are independent
- single member → that member's bounds verbatim
- empty cluster → fully unbounded (None, None)
- determinism: identical inputs → identical output, no clock / randomness
"""

from __future__ import annotations

from engrava.extensions.dreaming_reflection_extent import (
    EMPTY_EXTENT,
    derive_reflection_extent,
)

# Fixed UTC-normalised ISO-8601 instants (lexicographic == chronological).
_T0 = "2024-01-01T00:00:00+00:00"
_T1 = "2024-02-01T00:00:00+00:00"
_T2 = "2024-03-01T00:00:00+00:00"
_T3 = "2024-04-01T00:00:00+00:00"


class TestAllBoundsFinite:
    """Every member carries finite bounds → MIN/MAX cover."""

    def test_min_from_and_max_until(self) -> None:
        """valid_from = MIN of lowers; valid_until = MAX of uppers."""
        result = derive_reflection_extent(
            [
                (_T0, _T2),
                (_T1, _T3),
            ]
        )
        assert result == (_T0, _T3)

    def test_ordering_independent_of_input_order(self) -> None:
        """Same members in a different order yield the same extent."""
        forward = derive_reflection_extent([(_T0, _T2), (_T1, _T3)])
        reversed_ = derive_reflection_extent([(_T1, _T3), (_T0, _T2)])
        assert forward == reversed_ == (_T0, _T3)

    def test_three_members_pick_extremes(self) -> None:
        """The MIN lower and MAX upper are picked across all members."""
        result = derive_reflection_extent(
            [
                (_T1, _T1),
                (_T0, _T2),
                (_T2, _T3),
            ]
        )
        assert result == (_T0, _T3)


class TestOpenLowerBound:
    """Any member with valid_from None forces an open lower bound."""

    def test_one_open_from_makes_derived_from_none(self) -> None:
        """A single None valid_from → derived valid_from is None."""
        result = derive_reflection_extent(
            [
                (None, _T2),
                (_T1, _T3),
            ]
        )
        assert result == (None, _T3)

    def test_open_from_does_not_affect_until(self) -> None:
        """Open lower bound leaves the upper bound MAX intact."""
        valid_from, valid_until = derive_reflection_extent(
            [
                (None, _T1),
                (_T0, _T3),
            ]
        )
        assert valid_from is None
        assert valid_until == _T3


class TestOpenUpperBound:
    """Any member with valid_until None forces an open upper bound."""

    def test_one_open_until_makes_derived_until_none(self) -> None:
        """A single None valid_until → derived valid_until is None."""
        result = derive_reflection_extent(
            [
                (_T0, _T2),
                (_T1, None),
            ]
        )
        assert result == (_T0, None)

    def test_open_until_does_not_affect_from(self) -> None:
        """Open upper bound leaves the lower bound MIN intact."""
        valid_from, valid_until = derive_reflection_extent(
            [
                (_T0, None),
                (_T1, _T3),
            ]
        )
        assert valid_from == _T0
        assert valid_until is None


class TestIndependentAxes:
    """The two bounds are derived independently of each other."""

    def test_open_on_both_axes_from_different_members(self) -> None:
        """One member opens the lower bound, another the upper bound."""
        result = derive_reflection_extent(
            [
                (None, _T2),
                (_T1, None),
            ]
        )
        assert result == (None, None)

    def test_single_member_open_on_both(self) -> None:
        """A single fully-open member yields a fully-open extent."""
        result = derive_reflection_extent([(None, None)])
        assert result == (None, None)


class TestSingleMember:
    """A single member's bounds pass through verbatim."""

    def test_single_finite_member(self) -> None:
        """One finite member → its own bounds."""
        result = derive_reflection_extent([(_T0, _T3)])
        assert result == (_T0, _T3)

    def test_single_open_lower(self) -> None:
        """One member with open lower bound → open lower bound."""
        result = derive_reflection_extent([(None, _T3)])
        assert result == (None, _T3)


class TestEmptyCluster:
    """The degenerate empty-cluster case is fully unbounded."""

    def test_empty_iterable_yields_empty_extent(self) -> None:
        """No members → (None, None), the fully-unbounded interval."""
        result = derive_reflection_extent([])
        assert result == (None, None)
        assert result == EMPTY_EXTENT

    def test_empty_generator_yields_empty_extent(self) -> None:
        """An exhausted generator is handled like any empty iterable."""
        result = derive_reflection_extent(b for b in ())
        assert result == EMPTY_EXTENT


class TestDeterminism:
    """The fold is pure: no clock, no randomness, stable output."""

    def test_repeated_calls_identical(self) -> None:
        """Identical inputs produce identical output across calls."""
        members = [(_T0, _T2), (None, _T3), (_T1, _T1)]
        first = derive_reflection_extent(members)
        second = derive_reflection_extent(members)
        assert first == second

    def test_accepts_an_iterable_not_only_a_list(self) -> None:
        """The helper consumes any iterable, e.g. a generator expression."""
        gen = ((vf, vu) for vf, vu in [(_T0, _T2), (_T1, _T3)])
        assert derive_reflection_extent(gen) == (_T0, _T3)

"""Derive a REFLECTION's valid-time extent from its cluster members.

Public surface is :func:`derive_reflection_extent`, a pure deterministic
function the dreaming extension calls once per cluster on its way to
creating the corresponding REFLECTION thought.  It folds the members'
nullable ISO-8601 ``valid_from`` / ``valid_until`` bounds into a single
interval the REFLECTION inherits at creation time.

The reduction is deliberately interval-arithmetic over the *open* bounds
the bi-temporal model assigns to ``None``:

* ``valid_from is None`` means an **open lower bound** (negative
  infinity).  The minimum of negative infinity and any finite instant is
  negative infinity, so a single member with ``valid_from is None``
  forces the REFLECTION's ``valid_from`` to ``None``.
* ``valid_until is None`` means an **open upper bound** (positive
  infinity).  The maximum of positive infinity and any finite instant is
  positive infinity, so a single member with ``valid_until is None``
  forces the REFLECTION's ``valid_until`` to ``None``.

When every member carries a finite bound, the REFLECTION inherits the
``MIN`` of the lower bounds and the ``MAX`` of the upper bounds — the
tightest interval that fully covers every member's validity window.

ISO-8601 strings are stored UTC-normalised by the domain layer, so a
plain lexicographic string comparison is equivalent to chronological
comparison; this module relies on that invariant and performs no parsing.

This module is LLM-free, clock-free and stateless: identical inputs
produce identical output, and the extent is computed once at creation
time only (member changes after the fact do not re-derive it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The empty-cluster extent.  A REFLECTION derived from no members has no
#: information to bound either side, so both ends are left open
#: (``None`` / ``None``) — the fully-unbounded interval.  Exposed as a
#: named constant so callers and tests share one definition of the
#: degenerate case rather than hard-coding ``(None, None)``.
EMPTY_EXTENT: tuple[None, None] = (None, None)


def derive_reflection_extent(
    member_bounds: Iterable[tuple[str | None, str | None]],
) -> tuple[str | None, str | None]:
    """Fold member valid-time bounds into the REFLECTION's inherited extent.

    Treats ``None`` bounds as open interval ends (``valid_from is None``
    = negative infinity, ``valid_until is None`` = positive infinity) and
    reduces the members to the tightest interval that covers them all:

    * ``valid_from`` = the ``MIN`` of every member's ``valid_from``,
      **unless any** member has ``valid_from is None`` — in which case the
      result is ``None`` (the minimum with negative infinity is negative
      infinity).
    * ``valid_until`` = the ``MAX`` of every member's ``valid_until``,
      **only when every** member has a non-``None`` ``valid_until``; if
      **any** member has ``valid_until is None`` the result is ``None``
      (the maximum with positive infinity is positive infinity).

    The two axes are independent: a member may pin one bound open and the
    other finite.

    ISO-8601 inputs are assumed UTC-normalised (the engrava domain layer
    guarantees this on write), so ``min`` / ``max`` over the raw strings
    is chronologically correct without any parsing.

    Args:
        member_bounds: Iterable of ``(valid_from, valid_until)`` pairs,
            one per cluster member.  Each element of a pair is an
            ISO-8601 string or ``None``.

    Returns:
        The derived ``(valid_from, valid_until)`` pair for the
        REFLECTION.  An empty iterable yields :data:`EMPTY_EXTENT`
        (``(None, None)``) — a REFLECTION with no members is fully
        unbounded on both ends.

    Examples:
        >>> derive_reflection_extent(
        ...     [
        ...         ("2024-01-01T00:00:00+00:00", "2024-03-01T00:00:00+00:00"),
        ...         ("2024-02-01T00:00:00+00:00", "2024-04-01T00:00:00+00:00"),
        ...     ]
        ... )
        ('2024-01-01T00:00:00+00:00', '2024-04-01T00:00:00+00:00')
        >>> derive_reflection_extent(
        ...     [(None, "2024-03-01T00:00:00+00:00")]
        ... )
        (None, '2024-03-01T00:00:00+00:00')
        >>> derive_reflection_extent([])
        (None, None)

    """
    lower_bounds: list[str] = []
    upper_bounds: list[str] = []
    open_lower = False
    open_upper = False
    saw_member = False

    for valid_from, valid_until in member_bounds:
        saw_member = True
        if valid_from is None:
            open_lower = True
        else:
            lower_bounds.append(valid_from)
        if valid_until is None:
            open_upper = True
        else:
            upper_bounds.append(valid_until)

    if not saw_member:
        return EMPTY_EXTENT

    derived_from = None if open_lower else min(lower_bounds)
    derived_until = None if open_upper else max(upper_bounds)
    return derived_from, derived_until

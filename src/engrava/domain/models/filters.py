"""Metadata and visibility filters for the ranked retrieval path.

These value objects let a caller scope a ranked query (``search_hybrid`` /
``recall``) to rows whose ``metadata_json`` satisfies a typed predicate —
**without** dropping to raw SQL (which loses hybrid ranking) or
over-fetching and post-filtering (which starves ``top_k``).

Scope (binding): this is a **query capability, not a security boundary.**
The filter narrows what a ranked query *considers*; it performs no
authentication, authorization, ownership validation, or write enforcement.
For cross-agent / cross-tenant isolation use a store per tenant
(:class:`~engrava.infrastructure.service_manager.EngravaManager`); for
shared-corpus access control use the commercial RBAC tier.

Two parameters, by design:

* :class:`MetadataFilter` — an ``AND``-conjunction of
  :class:`FieldPredicate` (operators :attr:`FieldOp.EQ` and
  :attr:`FieldOp.IN`). An empty ``MetadataFilter`` is a match-all no-op,
  equivalent to passing ``None``.
* :class:`VisibilityQueryFilter` — the bounded ``(visibility IN … [OR
  owner = …])`` shape. Its all-empty form is *rejected*, because an
  all-empty visibility predicate matches nothing — a silent trap.

The effective predicate is their conjunction
``P_filters AND ( P_visibility )`` with the visibility ``OR`` always
parenthesised, and the whole per-row predicate guarded by
``CASE WHEN json_valid(metadata_json) THEN (...) ELSE 0 END`` so a row
holding malformed JSON is non-matching and never aborts the query.

Value semantics are **SQLite value equality**, not strict JSON-type
matching: SQLite's JSON1 stores booleans as integers, so ``EQ True``
aliases ``EQ 1`` and ``EQ False`` aliases ``EQ 0``. This is documented
behaviour, chosen to avoid a per-row ``json_type`` cost on the
exhaustive vector arm.
"""

from __future__ import annotations

import enum
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from engrava.domain.exceptions import InvalidFilterError, InvalidFilterPathError

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Allowed leaf-value domain for an ``EQ`` value and each ``IN`` element.
#: Mirrors the JSON scalar domain; values bind as SQL parameters and
#: compare under SQLite value equality.
FilterScalar = str | int | float | bool | None

#: JSONPath grammar accepted by :class:`FieldPredicate`. Restricts paths to
#: a ``$`` root followed by dot-identifiers (``.key``) and bracketed array
#: indices (``[0]``). Anything else is rejected with
#: :class:`~engrava.domain.exceptions.InvalidFilterPathError`. The path is
#: *also* bound as a SQL parameter (defence in depth), so this grammar is
#: a contract check, not the sole injection guard.
_PATH_RE: Final = re.compile(r"^\$(\.[A-Za-z0-9_]+|\[[0-9]+\])*$")

#: SQLite stores integers in a signed 64-bit cell. A Python ``int`` outside
#: this range cannot bind and would fail at execution time, so it is
#: rejected at construction to preserve the "never mid-query" contract.
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1

#: Upper bound on the total number of field predicates in one effective
#: predicate. ``json_each`` removes the per-element bind-variable cost of
#: ``IN``, but it does not lift SQLite's statement-variable / expression
#: limits, so a pathologically large conjunction is rejected at
#: construction against this documented maximum (never mid-query). Well
#: above any realistic hand-written filter.
MAX_PREDICATE_COUNT: Final = 250


def _validate_scalar(value: FilterScalar, *, context: str) -> None:
    """Validate that ``value`` is in the allowed scalar domain.

    ``bool`` is intentionally accepted (it aliases ``int`` under SQLite
    value equality — documented in the module docstring). ``int`` is
    range-checked against SQLite's signed 64-bit cell; ``float`` is
    rejected when non-finite (``NaN``/``±inf`` cannot round-trip JSON or
    bind meaningfully).

    Args:
        value: The candidate scalar.
        context: Human-readable origin for the error message (e.g.
            ``"EQ value"`` or ``"IN element"``).

    Raises:
        InvalidFilterError: If ``value`` is out of the allowed domain.

    """
    # Order matters: ``bool`` is a subclass of ``int``, so test it first to
    # avoid range-checking ``True``/``False`` as integers (harmless, but the
    # message would mislead).
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        if not (_INT64_MIN <= value <= _INT64_MAX):
            msg = (
                f"{context} integer {value} is outside SQLite's signed "
                f"64-bit range [{_INT64_MIN}, {_INT64_MAX}]"
            )
            raise InvalidFilterError(msg)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{context} float must be finite, got {value!r}"
            raise InvalidFilterError(msg)
        return
    if isinstance(value, str):
        return
    msg = f"{context} must be one of str, int, float, bool, or None; got {type(value).__name__}"
    raise InvalidFilterError(msg)


def _validate_path(path: object) -> None:
    """Validate a JSONPath against the restricted grammar.

    Uses ``fullmatch`` rather than ``match`` so a trailing newline (which
    ``$`` would otherwise admit) is rejected — the grammar must match the
    *entire* string. A non-string path is also rejected as a typed error
    (never a bare ``TypeError``), preserving the "construction-time only"
    contract.

    Args:
        path: The candidate path (must be a ``str``).

    Raises:
        InvalidFilterPathError: If ``path`` is not a string, or does not
            match the grammar exactly.

    """
    if not isinstance(path, str) or not _PATH_RE.fullmatch(path):
        raise InvalidFilterPathError(path if isinstance(path, str) else repr(path))


class FieldOp(enum.Enum):
    """Operator for a :class:`FieldPredicate`.

    Only equality and membership are expressible in this version.
    Negation/inequality (``!=``, ``NOT IN``) and ranges
    (``>``/``<``/``BETWEEN``) are intentionally not supported; attempting
    any other operator raises
    :class:`~engrava.domain.exceptions.InvalidFilterError` at construction.

    Attributes:
        EQ: Equality against a single scalar. ``EQ None`` matches both a
            missing path and a JSON ``null`` (compiles to ``IS NULL``).
        IN: Membership in a collection of scalars. An empty collection
            matches nothing.

    """

    EQ = "EQ"
    IN = "IN"


@dataclass(frozen=True)
class FieldPredicate:
    """A single typed predicate over one ``metadata_json`` path.

    Construction validates the path grammar and the value domain so an
    invalid predicate can never reach query compilation.

    Args:
        path: JSONPath of the restricted shape ``$``, ``$.key`` or
            ``$[0]`` (dot-identifiers and bracketed indices only).
        op: The operator (:attr:`FieldOp.EQ` or :attr:`FieldOp.IN`).
        value: For :attr:`FieldOp.EQ`, a single scalar
            (``str | int | float | bool | None``). For :attr:`FieldOp.IN`,
            a collection of such scalars (an empty collection matches
            nothing). Collections are normalised to a tuple so the
            predicate stays hashable/frozen.

    Raises:
        InvalidFilterPathError: If ``path`` violates the grammar.
        InvalidFilterError: If ``op`` is unsupported, or any value is
            outside the allowed scalar domain (non-finite float,
            out-of-range int, unsupported type).

    Examples:
        >>> FieldPredicate("$.session_id", FieldOp.EQ, "s-1").op.value
        'EQ'
        >>> FieldPredicate("$.role", FieldOp.IN, ("user", "system")).value
        ('user', 'system')

    """

    path: str
    op: FieldOp
    value: FilterScalar | tuple[FilterScalar, ...]

    def __init__(
        self,
        path: str,
        op: FieldOp,
        value: FilterScalar | Iterable[FilterScalar],
    ) -> None:
        """Validate and freeze the predicate (see class docstring)."""
        _validate_path(path)
        if not isinstance(op, FieldOp):
            msg = (
                f"unsupported filter operator {op!r}: only FieldOp.EQ and FieldOp.IN are supported"
            )
            raise InvalidFilterError(msg)

        normalized: FilterScalar | tuple[FilterScalar, ...]
        if op is FieldOp.IN:
            if isinstance(value, str | bytes) or not isinstance(value, Iterable):
                msg = "FieldOp.IN value must be a collection of scalars"
                raise InvalidFilterError(msg)
            elements = tuple(value)
            for element in elements:
                _validate_scalar(element, context="IN element")
            normalized = elements
        else:
            if isinstance(value, Iterable) and not isinstance(value, str | bytes):
                msg = "FieldOp.EQ value must be a single scalar, not a collection"
                raise InvalidFilterError(msg)
            _validate_scalar(value, context="EQ value")  # type: ignore[arg-type]
            normalized = value  # type: ignore[assignment]

        # Frozen dataclass: bypass the blocked setattr to store fields.
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "value", normalized)

    def compile(self, *, column: str) -> tuple[str, list[object]]:
        """Compile this predicate to an inner SQL fragment and its params.

        The fragment references ``column`` (the JSON-bearing column, e.g.
        ``t.metadata_json``) and is *not* itself ``json_valid``-guarded —
        the caller wraps the whole effective predicate once (see
        :func:`compile_effective_predicate`).

        Args:
            column: SQL column expression holding the JSON document.

        Returns:
            A ``(sql_fragment, params)`` pair. ``params`` are positional,
            in fragment order.

        """
        if self.op is FieldOp.IN:
            elements = self.value if isinstance(self.value, tuple) else ()
            # Empty IN matches nothing. Use a contradiction rather than an
            # empty ``IN ()`` (a syntax error in SQLite).
            if not elements:
                return ("0 = 1", [])
            non_null = [e for e in elements if e is not None]
            has_null = len(non_null) != len(elements)
            terms: list[str] = []
            params: list[object] = []
            if non_null:
                # Bind the whole collection as ONE JSON-array parameter via
                # json_each — no per-element bind-variable cost. ``column`` is
                # a fixed internal literal (never user input); the path and
                # values are bound parameters.
                terms.append(
                    f"json_extract({column}, ?) IN (SELECT value FROM json_each(?))"  # noqa: S608
                )
                params.append(self.path)
                params.append(json.dumps(non_null))
            if has_null:
                # A NULL member folds into the IS NULL branch (json_extract
                # returns NULL for both a missing path and a JSON null).
                terms.append(f"json_extract({column}, ?) IS NULL")
                params.append(self.path)
            fragment = "(" + " OR ".join(terms) + ")"
            return (fragment, params)

        # EQ
        if self.value is None:
            # SQL ``NULL = NULL`` is NULL, so EQ None must compile to IS NULL.
            return (f"json_extract({column}, ?) IS NULL", [self.path])
        return (f"json_extract({column}, ?) = ?", [self.path, self.value])


@dataclass(frozen=True)
class MetadataFilter:
    """An ``AND``-conjunction of :class:`FieldPredicate` over ``metadata_json``.

    An empty ``MetadataFilter`` (zero predicates) is a **match-all no-op**,
    equivalent to passing ``None`` — ``AND`` of nothing is conventionally
    ``TRUE``, and the common "start empty, append from request params"
    pattern expects an empty filter to *widen*, not narrow. (Deliberately
    the opposite of :class:`VisibilityQueryFilter`, whose all-empty form is
    rejected.)

    This is a query refinement, **not** a security boundary (see module
    docstring).

    Args:
        predicates: The field predicates to ``AND`` together. Normalised to
            a tuple so the filter stays hashable/frozen.

    Raises:
        InvalidFilterError: If the predicate count exceeds
            :data:`MAX_PREDICATE_COUNT`.

    Examples:
        >>> f = MetadataFilter(
        ...     [FieldPredicate("$.session_id", FieldOp.EQ, "s-1")]
        ... )
        >>> f.is_empty()
        False
        >>> MetadataFilter([]).is_empty()
        True

    """

    predicates: tuple[FieldPredicate, ...]

    def __init__(self, predicates: Sequence[FieldPredicate] = ()) -> None:
        """Validate the elements + predicate count and freeze the filter."""
        try:
            normalized = tuple(predicates)
        except TypeError as exc:
            # A non-iterable ``predicates`` (e.g. None) must surface as the
            # typed construction error, never a bare TypeError.
            msg = (
                "MetadataFilter predicates must be an iterable of "
                f"FieldPredicate; got {predicates!r}"
            )
            raise InvalidFilterError(msg) from exc
        for predicate in normalized:
            if not isinstance(predicate, FieldPredicate):
                msg = (
                    "MetadataFilter predicates must be FieldPredicate "
                    f"instances; got {type(predicate).__name__}"
                )
                raise InvalidFilterError(msg)
        if len(normalized) > MAX_PREDICATE_COUNT:
            msg = (
                f"MetadataFilter has {len(normalized)} predicates, exceeding "
                f"the maximum of {MAX_PREDICATE_COUNT}"
            )
            raise InvalidFilterError(msg)
        object.__setattr__(self, "predicates", normalized)

    def is_empty(self) -> bool:
        """Return ``True`` when the filter holds no predicates (match-all).

        Returns:
            ``True`` when this filter is the match-all no-op.

        """
        return len(self.predicates) == 0

    def compile(self, *, column: str) -> tuple[str, list[object]]:
        """Compile the conjunction to an inner SQL fragment and its params.

        Args:
            column: SQL column expression holding the JSON document.

        Returns:
            A ``(sql_fragment, params)`` pair. The fragment is ``"1 = 1"``
            (match-all) when the filter is empty.

        """
        if self.is_empty():
            return ("1 = 1", [])
        fragments: list[str] = []
        params: list[object] = []
        for predicate in self.predicates:
            fragment, predicate_params = predicate.compile(column=column)
            fragments.append(fragment)
            params.extend(predicate_params)
        return ("(" + " AND ".join(fragments) + ")", params)


@dataclass(frozen=True)
class VisibilityQueryFilter:
    """Bounded ``(visibility IN … [OR owner = …])`` query refinement.

    Reads ``$.visibility`` and ``$.owner`` from ``metadata_json`` and
    compiles to a parenthesised group so its ``OR`` can never bind looser
    than the surrounding ``AND``.

    **This is a query filter, NOT access control.** It performs no
    authentication, authorization, ownership validation, or write
    enforcement. The caller supplies (and can forge) ``owner`` and the
    stored ``visibility``; the filter is bypassable by passing
    ``visibility=None``, by using another API, or by issuing raw SQL. It
    **must not** be used to protect tenant data — for cross-tenant
    isolation use a store per tenant
    (:class:`~engrava.infrastructure.service_manager.EngravaManager`); for
    shared-corpus access control use the commercial RBAC tier. ``visibility``
    intent is only as trustworthy as the metadata the application writes;
    this filter neither stamps nor enforces it.

    Args:
        allowed: Visibility values to admit (matched against
            ``$.visibility``). Normalised to a ``frozenset``.
        owner: When provided, rows whose ``$.owner`` equals this value are
            also admitted (the ``OR`` branch) — the "public-or-mine"
            pattern. When ``None`` the ``OR`` branch is omitted.

    Raises:
        InvalidFilterError: If the all-empty form
            (``allowed=frozenset(), owner=None``) is used — it would match
            nothing, and a loud error beats silently returning zero rows.

    Examples:
        >>> v = VisibilityQueryFilter(frozenset({"public"}), owner="alice")
        >>> v.owner
        'alice'

    """

    allowed: frozenset[str]
    owner: str | None

    def __init__(self, allowed: Iterable[str], owner: str | None = None) -> None:
        """Validate types + non-emptiness and freeze the filter.

        Raises:
            InvalidFilterError: If ``allowed`` is a bare ``str`` (a common
                mistake that would silently iterate characters), if any
                ``allowed`` member is not a ``str``, if ``owner`` is neither
                ``str`` nor ``None``, or if the filter is all-empty.

        """
        if isinstance(allowed, str | bytes):
            msg = "VisibilityQueryFilter.allowed must be a collection of strings, not a bare string"
            raise InvalidFilterError(msg)
        try:
            normalized = frozenset(allowed)
        except TypeError as exc:
            # A non-iterable ``allowed`` (e.g. None) or an unhashable member
            # (e.g. a list) must surface as the typed construction error.
            msg = (
                "VisibilityQueryFilter.allowed must be a collection of "
                f"hashable strings; got {allowed!r}"
            )
            raise InvalidFilterError(msg) from exc
        for value in normalized:
            if not isinstance(value, str):
                msg = (
                    f"VisibilityQueryFilter.allowed members must be str; got {type(value).__name__}"
                )
                raise InvalidFilterError(msg)
        if owner is not None and not isinstance(owner, str):
            msg = f"VisibilityQueryFilter.owner must be str or None; got {type(owner).__name__}"
            raise InvalidFilterError(msg)
        if not normalized and owner is None:
            msg = (
                "VisibilityQueryFilter is empty (allowed=frozenset(), "
                "owner=None): an all-empty visibility filter matches nothing"
            )
            raise InvalidFilterError(msg)
        object.__setattr__(self, "allowed", normalized)
        object.__setattr__(self, "owner", owner)

    def compile(self, *, column: str) -> tuple[str, list[object]]:
        """Compile to a parenthesised SQL group and its params.

        Args:
            column: SQL column expression holding the JSON document.

        Returns:
            A ``(sql_fragment, params)`` pair. The fragment is always
            wrapped in parentheses so its ``OR`` is precedence-safe.

        """
        terms: list[str] = []
        params: list[object] = []
        if self.allowed:
            # Sorted for a deterministic, stable bound JSON-array param.
            # ``column`` is a fixed internal literal; ``'$.visibility'`` is a
            # constant path; the array is a bound parameter.
            terms.append(
                f"json_extract({column}, '$.visibility') "  # noqa: S608
                "IN (SELECT value FROM json_each(?))"
            )
            params.append(json.dumps(sorted(self.allowed)))
        if self.owner is not None:
            terms.append(f"json_extract({column}, '$.owner') = ?")
            params.append(self.owner)
        return ("(" + " OR ".join(terms) + ")", params)


def compile_effective_predicate(
    filters: MetadataFilter | None,
    visibility: VisibilityQueryFilter | None,
    *,
    column: str,
) -> tuple[str, list[object]] | None:
    """Compile ``filters`` and ``visibility`` into one guarded SQL predicate.

    Produces the per-row predicate
    ``CASE WHEN json_valid(<column>) THEN ( P_filters AND ( P_visibility ) )
    ELSE 0 END`` — the ``json_valid`` guard wraps the **whole** predicate
    (including the visibility group), so a row holding malformed JSON is
    non-matching for every predicate and never aborts the query.

    Args:
        filters: Optional metadata conjunction. ``None`` or an empty filter
            contributes ``TRUE``.
        visibility: Optional bounded visibility group. ``None`` contributes
            ``TRUE``.
        column: SQL column expression holding the JSON document (e.g.
            ``t.metadata_json``).

    Returns:
        A ``(sql_fragment, params)`` pair ready to ``AND`` into a ``WHERE``
        clause, or ``None`` when neither argument constrains anything (so
        the caller can preserve the exact unfiltered query path).

    """
    has_filters = filters is not None and not filters.is_empty()
    has_visibility = visibility is not None
    if not has_filters and not has_visibility:
        return None

    conjuncts: list[str] = []
    params: list[object] = []
    if has_filters and filters is not None:
        fragment, filter_params = filters.compile(column=column)
        conjuncts.append(fragment)
        params.extend(filter_params)
    if has_visibility and visibility is not None:
        fragment, visibility_params = visibility.compile(column=column)
        conjuncts.append(fragment)
        params.extend(visibility_params)

    inner = " AND ".join(conjuncts)
    guarded = f"CASE WHEN json_valid({column}) THEN ({inner}) ELSE 0 END"
    return (guarded, params)

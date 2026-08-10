"""Strict configuration decoders shared across engrava.

A single source of truth for the value-level checks the configuration layer
applies. Both construction paths — direct dataclass construction
(``__post_init__``) and the YAML loader — call these helpers, so equivalent
invalid values are rejected identically regardless of how a config object is
built. Domain value objects that mirror a config section (for example
``DeriveGates``) also use them, which is why the decoders live in this
dependency-free module rather than in :mod:`engrava.config` (that module imports
those value objects, so hosting the decoders there would be circular).

Every numeric decoder rejects ``bool`` explicitly (Python ``bool`` is an ``int``
subclass) and every float decoder rejects non-finite values (``NaN`` / ``inf``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import DataclassInstance

_T = TypeVar("_T")


class ConfigError(ValueError):
    """Raised when engrava configuration is invalid.

    Subclasses :class:`ValueError` because an invalid configuration value is a
    value-domain error. Every validation failure — direct dataclass construction
    or the YAML loader — raises this one type, so equivalent invalid values are
    rejected identically. Catching ``ValueError`` (the historical
    direct-construction behaviour) still catches it.

    Attributes:
        message: Human-readable error description.

    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def require_mapping(value: object, label: str) -> dict[str, Any]:
    """Return ``value`` when it is a mapping, else raise :class:`ConfigError`.

    The mapping returned is a plain ``dict`` this module owns, with its keys and
    values re-owned one level deep. A ``dict`` subclass may answer
    ``__getitem__`` / ``__iter__`` / ``__contains__`` for itself, so a mapping
    validated once can present different entries to the code that reads it
    afterwards; and its keys and values are as subclassable as any other value.

    Args:
        value: Candidate value (typically a raw YAML node or a config field).
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated entries as a plain ``dict``.

    Raises:
        ConfigError: When ``value`` is not a ``dict``.

    """
    if not isinstance(value, dict):
        msg = f"'{label}' must be a mapping"
        raise ConfigError(msg)
    return {_own_scalar(key): _own_scalar(entry) for key, entry in dict.items(value)}


def require_bool(value: object, label: str) -> bool:
    """Return ``value`` when it is a ``bool``, else raise :class:`ConfigError`.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated boolean.

    Raises:
        ConfigError: When ``value`` is not a ``bool``.

    """
    if not isinstance(value, bool):
        msg = f"'{label}' must be a boolean"
        raise ConfigError(msg)
    return value


def own_str(value: str) -> str:
    """Return an exact ``str`` carrying the same text as *value*.

    ``str.__str__`` is resolved on the built-in type rather than on the
    instance, so a ``str`` subclass cannot intercept it: it reads the real text
    buffer and hands back an exact ``str``. For a value that already *is* an
    exact ``str`` this is the identity and costs nothing.

    ``isinstance(value, str)`` establishes that a value *has* real text; it says
    nothing about what the value will answer later. A subclass may return
    anything at all from ``__format__``, ``__str__``, ``__contains__`` or
    ``rsplit``, so a check performed on the caller's object constrains only that
    object's behaviour at that instant. Re-owning the text closes the gap by
    construction: what was validated and what is used become the same value.

    Args:
        value: An already type-checked string.

    Returns:
        The same text as an exact ``str``.

    """
    return str.__str__(value)


def own_int(value: int) -> int:
    """Return an exact ``int`` carrying the same value as *value*.

    The integer counterpart of :func:`own_str`: ``int.__index__`` is resolved on
    the built-in type, reads the real machine value, and hands back an exact
    ``int``, so an ``int`` subclass that answers ``__format__`` with arbitrary
    text cannot reach a formatted string through a validated field. For a value
    that already *is* an exact ``int`` this is the identity.

    Args:
        value: An already type-checked integer.

    Returns:
        The same numeric value as an exact ``int``.

    """
    return int.__index__(value)


def own_float(value: float) -> float:
    """Return an exact ``float`` carrying the same value as *value*.

    ``float.__float__`` is resolved on the built-in type and reads the real
    IEEE double, where the ``float(value)`` constructor dispatches to
    ``type(value).__float__`` and therefore returns whatever a subclass decides.
    An ``int`` is handed back through :func:`own_int` instead — the numeric
    tower means an integer is an acceptable value for a float field, and
    converting it here would change what the caller configured.

    Args:
        value: An already type-checked real number.

    Returns:
        The same numeric value as an exact ``float`` (or an exact ``int`` when
        an integer was supplied).

    """
    if isinstance(value, int):
        return own_int(value)
    return float.__float__(value)


def _own_scalar(value: object) -> Any:  # noqa: ANN401 -- mirrors the caller's own type
    """Return an exact-built-in copy of a scalar, or *value* itself.

    Anything this does not recognise — a ``Path``, a nested config object, a
    mapping inside a raw YAML node — is passed through untouched, so the
    caller's own validation still sees exactly what it was given and decides
    whether to reject it. This function never raises and never recurses.

    Args:
        value: Any value.

    Returns:
        An exact ``bool`` / ``int`` / ``float`` / ``str`` when *value* is one
        of those, otherwise *value* unchanged.

    """
    # ``bool`` is final in CPython ("type 'bool' is not an acceptable base
    # type"), so a bool is always exactly a bool and there is nothing to own.
    # It is tested before ``int`` because ``bool`` is an ``int`` subclass.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return own_int(value)
    if isinstance(value, float):
        return own_float(value)
    if isinstance(value, str):
        return own_str(value)
    return value


#: How to read the real contents of each container type without asking the
#: instance. Every one of these is resolved on the built-in, so it reads the
#: real storage and cannot be intercepted, cannot run caller code, and cannot
#: fail to terminate — where ``iter(value)`` / ``value.items()`` would do all
#: three at the caller's discretion.
_UNBOUND_ITER: tuple[tuple[type, Any], ...] = (
    (tuple, tuple.__iter__),
    (list, list.__iter__),
    (frozenset, frozenset.__iter__),
    (set, set.__iter__),
)


def _owned_entries(value: object, read: Any) -> list[Any] | None:  # noqa: ANN401 -- built-in method descriptor
    """Return every entry of a container, owned, or ``None`` if one cannot be.

    A container is rebuilt only when *all* of its entries decode to exact
    built-ins: the rebuilt ``set`` or ``dict`` key is then a built-in whose
    ``__hash__`` is the built-in one, so reconstruction runs no caller code of
    its own. A container holding anything else is left exactly as it was, for
    the validator that follows to accept or reject on its own terms.

    Args:
        value: The container to read.
        read: The unbound iterator for that container's built-in type.

    Returns:
        The owned entries, or ``None`` when at least one entry is not an
        ownable scalar.

    """
    owned: list[Any] = []
    for entry in read(value):
        candidate = _own_scalar(entry)
        if not _is_exact_builtin_scalar(candidate):
            return None
        owned.append(candidate)
    return owned


def _is_exact_builtin_scalar(value: object) -> bool:
    """Return whether *value* is exactly a built-in scalar, subclasses excluded."""
    return type(value) in (bool, int, float, str)


def _own_mapping(value: dict[Any, Any]) -> Any:  # noqa: ANN401 -- mirrors the caller's own type
    """Return a plain ``dict`` of owned keys, or *value* when a key is not ownable.

    ``dict.items`` is the unbound built-in, so it reads the real hash table
    rather than whatever a subclass wants ``items()`` to report. Values are
    owned where they are scalars and passed through otherwise — a mapping of
    nested config objects keeps them — but a key that cannot be owned leaves
    the whole mapping alone, because rebuilding would have to hash it.

    Args:
        value: The mapping to own.

    Returns:
        A plain ``dict``, or *value* unchanged.

    """
    owned: dict[Any, Any] = {}
    for key, entry in dict.items(value):
        owned_key = _own_scalar(key)
        if not _is_exact_builtin_scalar(owned_key):
            return value
        owned[owned_key] = _own_scalar(entry)
    return owned


def _own_value(value: object) -> Any:  # noqa: ANN401 -- mirrors the caller's own type
    """Return an exact-built-in copy of a scalar or a one-level container.

    Every dispatch this *can* avoid, it avoids: containers are read through the
    unbound built-in iterators over their real storage, entries are decoded
    through the unbound scalar conversions, and a container is only rebuilt
    when every entry (or, for a mapping, every key) has decoded to an exact
    built-in whose ``__hash__`` is the built-in one. So an overridden
    ``__iter__`` / ``items`` / ``__hash__`` is never reached from here, and a
    container cannot present entries it does not hold.

    It is **not** claimed that no caller code runs at all — see
    :func:`own_config_fields` for why that is not achievable, and for the
    weaker property that is.

    Nothing recurses either. Configuration containers are one level deep by
    construction, so a self-referential container has no path to unbounded
    recursion.

    Args:
        value: Any value.

    Returns:
        An owned copy, or *value* unchanged when its type is not one this
        module knows how to own, or when one of its entries is not.

    """
    if isinstance(value, (bool, int, float, str)):
        return _own_scalar(value)
    if isinstance(value, dict):
        return _own_mapping(value)
    for container, read in _UNBOUND_ITER:
        if isinstance(value, container):
            entries = _owned_entries(value, read)
            return value if entries is None else container(entries)
    return value


def _read_container(value: object) -> tuple[Any, ...]:
    """Return a container's real entries, read through the unbound built-in.

    ``tuple(value)`` would iterate with ``type(value).__iter__``, which a
    subclass may define to raise, to yield different entries on each pass, or
    never to stop. Reading the built-in's own iterator over the real storage
    admits none of those.

    Args:
        value: A ``list`` / ``tuple`` / ``set`` / ``frozenset`` instance.

    Returns:
        The container's entries, exactly once.

    """
    for container, read in _UNBOUND_ITER:
        if isinstance(value, container):
            return tuple(read(value))
    # Unreachable for the four types the caller has already narrowed to; kept
    # as a total fallback rather than an assertion that could fire in a release.
    return tuple(cast("Iterable[Any]", value))


def forbid_subclassing(cls: type[_T]) -> type[_T]:
    """Make *cls* refuse to be subclassed, at class-definition time.

    A configuration value's entire job is to report the settings it was
    validated as holding. Subclassing one is therefore not a subtype relation
    but a way to make it answer twice — and the sharpest form of that cannot be
    survived by any amount of care at the reading end. A **data descriptor**
    installed over a field returns benign values while the object is decoded
    and validated, then different ones for ever after; nothing raises, so no
    fail-closed guard engages, and the write-back that would dislodge it goes
    through the descriptor's own ``__set__`` and is discarded.

    **What this closes, and where it stops.** For a caller that reaches engrava
    only by passing values across its API, declaring a subclass and putting the
    descriptor in its class body is the way in, and that is the way this
    refuses — at class-definition time, so the failure names the offending
    definition rather than surfacing later at some use of an instance, and
    ``type(name, (Config,), {})`` is blocked exactly as a ``class`` statement
    is. It equally removes the hostile-metaclass case, which also has to be
    declared on a subclass.

    What it does **not** close is assignment straight onto this package's own
    class object (``HygienePolicyConfig.protected_priorities = descriptor``),
    which needs no subclass. That is deliberate and is not defended anywhere:
    code able to rewrite engrava's class objects is already executing in the
    process and could as easily replace the hygiene pass itself, or
    ``os.unlink``. It is the absence of a boundary rather than a hole in one.

    Nor is this sufficient on its own even within the scope it does cover:
    ``__class__`` can be reassigned to a look-alike that was never a subclass.
    That is why every boundary accepting a configuration object also requires
    the exact type — see :func:`require_exact_type`. Neither check subsumes the
    other.

    Args:
        cls: The configuration class to close.

    Returns:
        *cls*, unchanged apart from the added hook.

    """

    def __init_subclass__(subclass: type, /, **kwargs: object) -> None:  # noqa: N807 -- this *is* the dunder hook
        del subclass, kwargs
        # Names this module's own class, never the offending one: reading the
        # subclass's name is one more caller-controlled lookup.
        msg = (
            f"{cls.__name__} may not be subclassed. A configuration value "
            f"reports the settings it was validated as holding; a subclass can "
            f"report something else afterwards. Compose one instead."
        )
        raise TypeError(msg)

    cls.__init_subclass__ = classmethod(__init_subclass__)  # type: ignore[assignment]  # installing the hook is the point
    return cls


def require_exact_type(value: object, expected: type[_T], label: str) -> _T:
    """Return *value* when its type is exactly *expected*, else raise.

    ``isinstance`` is the wrong test at this boundary and cannot be made into
    the right one. A configuration object is a **value**: what it is for is to
    report the settings it was validated as holding. A subclass is genuinely an
    instance and passes ``isinstance``, and is also free to define
    ``__getattribute__`` — so it can report one set of settings while it is
    being validated and another every time it is read afterwards. Owning the
    *fields* does not help when the object holding them is the caller's.

    :func:`forbid_subclassing` closes the subclass route at its source, but not
    this one: ``__class__`` can be reassigned to a look-alike that was never a
    subclass, and only a check on the exact type at the point of acceptance
    sees that. The two are complementary, not redundant.

    Args:
        value: Candidate configuration object.
        expected: The exact class required.
        label: Fully-qualified field label used in the error message.

    Returns:
        *value*, unchanged.

    Raises:
        ConfigError: When ``type(value)`` is not exactly *expected*.

    """
    if type(value) is not expected:
        # ``expected`` is this package's own class, so naming it in the message
        # runs no caller-controlled code; ``value`` is not named for the usual
        # reason.
        msg = f"'{label}' must be exactly a {expected.__name__}"
        raise ConfigError(msg)
    return value


def require_exact_type_or_none(value: object, expected: type[_T], label: str) -> _T | None:
    """Return *value* when it is ``None`` or exactly *expected*, else raise.

    Args:
        value: Candidate configuration object, or ``None``.
        expected: The exact class required when a value is supplied.
        label: Fully-qualified field label used in the error message.

    Returns:
        *value*, unchanged.

    Raises:
        ConfigError: When *value* is neither ``None`` nor exactly *expected*.

    """
    if value is None:
        return None
    return require_exact_type(value, expected, label)


def own_config_fields(instance: DataclassInstance) -> None:
    """Replace every field of a config dataclass with a value this module owns.

    Called first in a configuration object's ``__post_init__``, so that every
    check that follows runs against an owned value and every field the object
    retains is the value that was checked. Without it each field would have to
    remember to store what its decoder returned, and the audit of "did this one
    remember?" is exactly the audit that has failed repeatedly: a value is
    inspected, a clean copy is built, the copy is dropped, and the caller's
    object is what a later ``__format__`` / ``__contains__`` / ``rsplit`` /
    ``__eq__`` answers for.

    Fields whose value this module cannot own — a ``Path``, a nested config
    object — are left untouched. A nested config owns its own fields in its own
    ``__post_init__``; a ``Path`` carries no validated property for a subclass
    to contradict, since any path is a legal path.

    **What is and is not guaranteed.** Reading an arbitrary object's state in
    Python cannot be done with zero dispatch — a metaclass, a data descriptor
    and even ``isinstance`` all consult something the caller can define — so
    "this never runs caller code" is not a property anything here could hold,
    and it is not claimed. What is held is **fail-closed on what escapes**: any
    exception leaving the sweep becomes the same :class:`ConfigError` the
    validation behind it would have raised, so caller code that *raises* can
    stop a configuration being built but cannot leave by an untyped door.

    That is a statement about exceptions, and deliberately not about
    acceptances, because it cannot be one. Caller code that returns quietly is
    not detectable from here: a data descriptor assigned straight onto one of
    this package's own config classes answers benign values while the object is
    decoded and validated and different ones ever after, raising nothing, and
    this sweep will accept it. :func:`forbid_subclassing` removes the route to
    that for any caller which only passes values across the API; nothing here
    or anywhere else defends a process whose own class objects have been
    rewritten, and nothing should — such code can replace the hygiene pass or
    ``os.unlink`` just as easily.

    Caller code which never returns is outside all of it: a metaclass that
    loops forever hangs the constructor. Nothing is accepted and nothing
    retained, so it is a denial of the caller's own process by the caller's own
    class.

    Args:
        instance: A dataclass instance, mid-``__post_init__``.

    Raises:
        ConfigError: If reading or decoding any field raises at all.

    """
    try:
        _own_config_fields_unguarded(instance)
    except BaseException as exc:
        # A module-owned literal: naming the value or its type here would be
        # one more read of exactly the thing that has just proved hostile.
        # ``BaseException`` rather than ``Exception`` because the guarantee is
        # about what escapes, not about which exceptions are polite: a
        # descriptor raising ``KeyboardInterrupt`` would otherwise leave by a
        # door this claims is shut. The cost is that an interrupt arriving
        # during the microseconds a config spends decoding surfaces as a
        # ``ConfigError`` chained from it, which is a trade taken knowingly.
        msg = "Configuration could not be decoded"
        raise ConfigError(msg) from exc


def _own_config_fields_unguarded(instance: DataclassInstance) -> None:
    """Do the work of :func:`own_config_fields`, without the fail-closed guard.

    Args:
        instance: A dataclass instance, mid-``__post_init__``.

    """
    # The field list is read off the *class*: ``dataclasses.fields(instance)``
    # fetches ``__dataclass_fields__`` with ``getattr`` on the instance, which
    # would run a subclass's ``__getattribute__`` before the sweep has done
    # anything at all. Field values are then read the same way, for the same
    # reason. A hostile metaclass or data descriptor is a level below both, and
    # is not defended against here - it is caught by the guard in the caller,
    # which turns it into a refusal.
    for field in dataclasses.fields(type(instance)):
        current = object.__getattribute__(instance, field.name)
        owned = _own_value(current)
        if owned is not current:
            object.__setattr__(instance, field.name, owned)


def require_int(value: object, label: str) -> int:
    """Return *value* as an exact ``int``, with no bound imposed.

    For a number whose *range* already carries meaning at the point of use — a
    cadence where anything below one means "off", say — where imposing a bound
    here would change what the caller is allowed to say. What it does do is
    make the comparison at the point of use run against a number rather than
    against a method: an ``int`` subclass answering ``<`` for itself can make
    "off" read as "on".

    ``bool`` is rejected as it is by every other numeric decoder here.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated value as an exact ``int``.

    Raises:
        ConfigError: When ``value`` is a ``bool`` or is not an ``int``.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"'{label}' must be an integer"
        raise ConfigError(msg)
    return own_int(value)


def require_int_or_none(value: object, label: str) -> int | None:
    """Return *value* as an exact ``int``, or ``None``.

    Args:
        value: Candidate value, or ``None``.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated value as an exact ``int``, or ``None``.

    Raises:
        ConfigError: When *value* is neither ``None`` nor an ``int``.

    """
    if value is None:
        return None
    return require_int(value, label)


def require_positive_int(value: object, label: str) -> int:
    """Return ``value`` when it is an integer ``>= 1``, else raise.

    ``bool`` is a subclass of ``int`` in Python; it is rejected explicitly so
    ``True`` / ``False`` cannot impersonate ``1`` / ``0``.

    The value returned is an exact ``int`` this module owns (see
    :func:`own_int`), never the caller's object — callers must use the
    returned value, not the one they passed in.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated positive integer, as an exact ``int``.

    Raises:
        ConfigError: When ``value`` is a ``bool``, is not an ``int``, or is ``< 1``.

    """
    msg = f"'{label}' must be a positive integer"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(msg)
    # Own *before* comparing: ``<`` is ``type(value).__lt__``, so a subclass
    # that answers it for itself would clear a range check the stored value
    # does not actually satisfy.
    owned = own_int(value)
    if owned < 1:
        raise ConfigError(msg)
    return owned


def require_nonneg_int(value: object, label: str) -> int:
    """Return ``value`` when it is an integer ``>= 0``, else raise.

    ``bool`` is rejected explicitly (see :func:`require_positive_int`). As
    there, the returned value is an exact ``int`` this module owns.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated non-negative integer, as an exact ``int``.

    Raises:
        ConfigError: When ``value`` is a ``bool``, is not an ``int``, or is ``< 0``.

    """
    msg = f"'{label}' must be a non-negative integer"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(msg)
    owned = own_int(value)
    if owned < 0:
        raise ConfigError(msg)
    return owned


def require_unit_float(value: object, label: str) -> float:
    """Return ``value`` as a finite float in ``[0.0, 1.0]``, else raise.

    Integers are accepted (numeric tower) and coerced to ``float``; ``bool`` is
    rejected explicitly so ``True`` / ``False`` cannot impersonate ``1.0`` /
    ``0.0`` and silently disable a gate, and non-finite values (``NaN`` / ``inf``)
    are rejected.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated value as a ``float``.

    Raises:
        ConfigError: When ``value`` is a ``bool``, is not numeric, is non-finite,
            or falls outside ``[0.0, 1.0]``.

    """
    msg = f"'{label}' must be a float in [0.0, 1.0]"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(msg)
    owned = own_float(value)
    if not is_finite(owned) or not 0.0 <= owned <= 1.0:
        raise ConfigError(msg)
    return owned


def require_nonneg_float(value: object, label: str) -> float:
    """Return ``value`` as a finite non-negative float, else raise.

    Integers are accepted (numeric tower) and coerced to ``float``; ``bool`` is
    rejected explicitly and non-finite values (``NaN`` / ``inf``) are rejected —
    ``NaN < 0.0`` is ``False``, so without the finiteness guard a ``NaN`` would
    slip through a bare lower-bound check.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated value as a ``float``.

    Raises:
        ConfigError: When ``value`` is a ``bool``, is not numeric, is non-finite,
            or is negative.

    """
    msg = f"'{label}' must be a non-negative number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(msg)
    owned = own_float(value)
    if not is_finite(owned) or owned < 0.0:
        raise ConfigError(msg)
    return owned


def require_finite_number(value: object, label: str) -> float:
    """Return ``value`` as a finite float of any magnitude/sign, else raise.

    Used for unconstrained numeric fields (for example signal weights, which are
    relative priorities with no fixed range). ``bool`` is rejected explicitly and
    non-finite values (``NaN`` / ``inf``, and an integer too large to represent as
    a finite float) raise :class:`ConfigError`.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated value as a ``float``.

    Raises:
        ConfigError: When ``value`` is a ``bool``, is not numeric, or is non-finite.

    """
    msg = f"'{label}' must be a finite number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(msg)
    owned = own_float(value)
    if not is_finite(owned):
        raise ConfigError(msg)
    return owned


def is_finite(value: float) -> bool:
    """Return whether ``value`` is a finite real number.

    ``math.isfinite`` raises :class:`OverflowError` for an integer too large to
    convert to a float (for example ``10**1000``, or a *sum* of individually finite
    integer weights that overflows the float range); such a value is not a finite
    float, so the overflow is normalised to ``False`` rather than propagated as a
    raw ``OverflowError`` past the config decoders.

    Args:
        value: A real number (``int`` or ``float``).

    Returns:
        ``True`` when ``value`` is finite and float-representable, else ``False``.

    """
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def require_str_collection(value: object, label: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of strings, rejecting a bare string.

    A Python ``str`` (and ``bytes``) is iterable character-by-character, so a bare
    string passed where a list of strings is expected would be silently accepted
    and iterate into single characters. Reject ``str`` / ``bytes`` explicitly and
    require a non-string iterable whose every entry is a ``str``. An empty
    collection is valid (it iterates zero times).

    Both the container and its entries are re-owned. The caller's collection is
    drained **once** into a plain ``tuple``, so a container whose ``__iter__``
    yields different values on a second pass cannot present one set of entries
    to this check and another to the code that consumes the config; every entry
    is then re-read through :func:`own_str`, so a ``str`` subclass cannot pass
    the ``isinstance`` check here and emit something else from ``__format__`` /
    ``__str__`` later. Callers must store the returned value: a container kept
    from the caller can still lie from ``__contains__`` / ``__bool__``, which is
    exactly how a membership gate gets bypassed.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated entries as a plain tuple of exact ``str`` values.

    Raises:
        ConfigError: When ``value`` is a ``str`` / ``bytes``, is not a
            list/tuple/set/frozenset, or holds a non-string entry.

    """
    if isinstance(value, (str, bytes)):
        msg = f"'{label}' must be a list of strings, not a single string"
        raise ConfigError(msg)
    if not isinstance(value, (list, tuple, set, frozenset)):
        msg = f"'{label}' must be a list of strings"
        raise ConfigError(msg)
    entries = _read_container(value)
    for entry in entries:
        if not isinstance(entry, str):
            msg = f"'{label}' entries must be strings"
            raise ConfigError(msg)
    return tuple(own_str(entry) for entry in entries)

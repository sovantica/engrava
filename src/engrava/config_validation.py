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

import math
from typing import Any


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

    Args:
        value: Candidate value (typically a raw YAML node or a config field).
        label: Fully-qualified field label used in the error message.

    Returns:
        The value narrowed to ``dict``.

    Raises:
        ConfigError: When ``value`` is not a ``dict``.

    """
    if not isinstance(value, dict):
        msg = f"'{label}' must be a mapping"
        raise ConfigError(msg)
    return value


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


def require_positive_int(value: object, label: str) -> int:
    """Return ``value`` when it is an integer ``>= 1``, else raise.

    ``bool`` is a subclass of ``int`` in Python; it is rejected explicitly so
    ``True`` / ``False`` cannot impersonate ``1`` / ``0``.

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated positive integer.

    Raises:
        ConfigError: When ``value`` is a ``bool``, is not an ``int``, or is ``< 1``.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"'{label}' must be a positive integer"
        raise ConfigError(msg)
    return value


def require_nonneg_int(value: object, label: str) -> int:
    """Return ``value`` when it is an integer ``>= 0``, else raise.

    ``bool`` is rejected explicitly (see :func:`require_positive_int`).

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated non-negative integer.

    Raises:
        ConfigError: When ``value`` is a ``bool``, is not an ``int``, or is ``< 0``.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"'{label}' must be a non-negative integer"
        raise ConfigError(msg)
    return value


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
    if not is_finite(value) or not 0.0 <= value <= 1.0:
        raise ConfigError(msg)
    return float(value)


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
    if not is_finite(value) or value < 0.0:
        raise ConfigError(msg)
    return float(value)


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
    if not is_finite(value):
        raise ConfigError(msg)
    return float(value)


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

    Args:
        value: Candidate value.
        label: Fully-qualified field label used in the error message.

    Returns:
        The validated entries as a tuple of strings.

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
    for entry in value:
        if not isinstance(entry, str):
            msg = f"'{label}' entries must be strings"
            raise ConfigError(msg)
    return tuple(value)

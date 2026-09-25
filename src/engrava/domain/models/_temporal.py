"""Shared temporal-field validation for domain models.

Provides a single source of truth for ISO-8601 timestamp validation and
UTC normalisation, used by every record that stores nullable timestamp
columns (transaction-time, access-time, and valid-time fields). Keeping
the logic here avoids divergent copies drifting across models.
"""

from __future__ import annotations

import datetime


def validate_iso8601_nullable(value: str | None) -> str | None:
    """Validate an ISO-8601 timestamp and return it in the canonical UTC form.

    Every accepted value comes back as ``parse_iso8601_to_utc(value).isoformat()``:
    ``T`` separator, extended format, ``+00:00`` offset, and microseconds only
    when they are non-zero. That is exactly what the store writes for its own
    timestamps (``datetime.now(UTC).isoformat()``), so an instant compares equal
    to engrava's own stamp of it, and lexicographic SQLite TEXT comparison of
    two canonical values orders them by instant. A timezone-aware value is
    converted to UTC; a naive value is read as UTC, the same convention
    :func:`parse_iso8601_to_utc` uses, so a naive value, a space separator,
    basic format and a week date all become the same canonical string.

    Args:
        value: Timestamp string or ``None``.

    Returns:
        The canonical UTC form of ``value``, or ``None`` when the input was
        ``None``.

    Raises:
        ValueError: If ``value`` is a string that is not valid ISO-8601, or
            one whose instant has no UTC form within the supported ``datetime``
            range (see :func:`canonical_timestamp`).

    Examples:
        >>> validate_iso8601_nullable("2026-04-12 15:00:00")
        '2026-04-12T15:00:00+00:00'
        >>> validate_iso8601_nullable("2026-04-12T15:00:00.5+02:00")
        '2026-04-12T13:00:00.500000+00:00'

    """
    if value is None:
        return value
    return canonical_timestamp(value)


def canonical_timestamp(value: str) -> str:
    """Return a required ISO-8601 timestamp in the canonical UTC form.

    The non-nullable form of :func:`validate_iso8601_nullable`, for a timestamp
    passed as an argument rather than stored in a nullable field — for example
    a caller's ``now`` or ``since``, which is compared as TEXT against stored
    canonical values and so has to be in the same form.

    Args:
        value: An ISO-8601 timestamp string, naive (read as UTC) or aware.

    Returns:
        ``parse_iso8601_to_utc(value).isoformat()``.

    Raises:
        ValueError: If ``value`` is not valid ISO-8601, or if it is but its
            instant has no UTC form within the supported ``datetime`` range —
            an aware value at the range limits, such as
            ``0001-01-01T00:00:00+01:00``, whose UTC conversion falls below
            year 1 (or above year 9999).

    Examples:
        >>> canonical_timestamp("20260412T150000")
        '2026-04-12T15:00:00+00:00'

    """
    try:
        instant = parse_iso8601_to_utc(value)
    except ValueError as exc:
        msg = f"Must be ISO-8601 timestamp, got {value!r}"
        raise ValueError(msg) from exc
    except OverflowError as exc:
        raise _no_utc_form(value) from exc
    return instant.isoformat()


def _no_utc_form(value: str) -> ValueError:
    """Build the error for a valid ISO-8601 value whose UTC conversion overflows.

    Args:
        value: The timestamp whose UTC form falls outside the ``datetime`` range.

    Returns:
        The ``ValueError`` to raise (the caller chains the ``OverflowError``).

    """
    return ValueError(f"Timestamp {value!r} has no UTC form within the supported datetime range")


def canonical_timestamp_or_none(value: object) -> str | None:
    """Return the canonical UTC form of an already-stored value, or ``None``.

    For a value that was written before, or around, the validator — a row read
    back from an older database, or a record in a snapshot file — where a value
    that cannot be normalised is left as it is rather than rejected. A value
    rewritten through this is byte-identical to what a fresh write of the same
    instant stores.

    Args:
        value: The stored value. Normally ``str``; a ``TEXT`` column can still
            hold a ``BLOB`` or number written directly.

    Returns:
        The canonical string, or ``None`` when ``value`` is not a string or
        does not name an ISO-8601 instant with a UTC form (an unparseable
        string, or an aware value so close to the ``datetime`` range limits
        that its UTC conversion does not exist).

    """
    if not isinstance(value, str):
        return None
    try:
        return canonical_timestamp(value)
    except ValueError:
        # canonical_timestamp raises ValueError for both cases above.
        return None


def parse_iso8601_to_utc(value: str) -> datetime.datetime:
    """Parse an ISO-8601 string into a UTC-normalised aware ``datetime``.

    Naive inputs are interpreted as UTC so that any two parsed instants are
    directly comparable regardless of the original offset (or the absence of
    one) — comparing a naive and an aware ``datetime`` would otherwise raise
    ``TypeError``. :func:`validate_iso8601_nullable` is built on this: it
    returns the ``isoformat()`` of the instant parsed here, the canonical
    string form the store keeps.

    Args:
        value: An ISO-8601 timestamp string (already format-validated by the
            time it reaches this helper).

    Returns:
        The parsed instant as a timezone-aware ``datetime`` in UTC.

    Raises:
        ValueError: If ``value`` is not a valid ISO-8601 timestamp.
        OverflowError: If ``value`` is valid ISO-8601 but its UTC conversion
            falls outside the ``datetime`` range (an aware value at the range
            limits, such as ``0001-01-01T00:00:00+01:00``). Callers that take
            such a value from outside turn this into their own typed error.

    Examples:
        >>> parse_iso8601_to_utc("2026-04-12T15:00:00+02:00").isoformat()
        '2026-04-12T13:00:00+00:00'

    """
    parsed = datetime.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC)


def validate_interval_ordering(valid_from: str | None, valid_until: str | None) -> None:
    """Reject an inverted closed valid-time interval.

    When both bounds are present the interval must not run backwards:
    ``valid_from`` must be at or before ``valid_until``, compared as
    UTC-normalised instants (via :func:`parse_iso8601_to_utc`) rather than as
    raw strings — so differing offsets and naive/aware mixes normalise before
    the comparison. An equal pair is permitted: a zero-length interval is a
    legitimate instantaneous fact. A ``None`` on either bound denotes an open
    interval and is always accepted.

    This is the single source of truth for the ordering invariant, shared by
    the ``ThoughtRecord`` / ``EdgeRecord`` model validators and the store's
    invalidate mutation path.

    Args:
        valid_from: Start of the validity interval, or ``None`` (open lower
            bound).
        valid_until: End of the validity interval, or ``None`` (open upper
            bound).

    Raises:
        ValueError: If both bounds are present and ``valid_from`` is strictly
            after ``valid_until``, or if a bound has no UTC form within the
            supported ``datetime`` range. A record's own bounds are canonical
            by the time they reach this, so the second case comes only from a
            stored bound the invalidate path reads back as it is.

    Examples:
        >>> validate_interval_ordering("2026-01-01T00:00:00", None)  # open bound
        >>> validate_interval_ordering(
        ...     "2026-01-01T00:00:00", "2026-01-01T00:00:00"
        ... )  # equal instants -> accepted

    """
    if valid_from is None or valid_until is None:
        return
    if _bound_instant(valid_from) > _bound_instant(valid_until):
        msg = (
            f"valid_from ({valid_from!r}) must not be after valid_until "
            f"({valid_until!r}): an inverted validity interval is rejected"
        )
        raise ValueError(msg)


def _bound_instant(value: str) -> datetime.datetime:
    """Parse a valid-time bound, reporting a missing UTC form as ``ValueError``.

    Args:
        value: The bound, as given or as stored.

    Returns:
        The bound as a UTC-normalised instant.

    Raises:
        ValueError: If ``value`` is not valid ISO-8601, or has no UTC form
            within the supported ``datetime`` range.

    """
    try:
        return parse_iso8601_to_utc(value)
    except OverflowError as exc:
        raise _no_utc_form(value) from exc

"""Shared temporal-field validation for domain models.

Provides a single source of truth for ISO-8601 timestamp validation and
UTC normalisation, used by every record that stores nullable timestamp
columns (transaction-time, access-time, and valid-time fields). Keeping
the logic here avoids divergent copies drifting across models.
"""

from __future__ import annotations

import datetime


def validate_iso8601_nullable(value: str | None) -> str | None:
    """Validate ISO-8601 format and normalise to UTC when not ``None``.

    Timezone-aware timestamps are converted to UTC so that SQLite TEXT
    comparisons (lexicographic) produce correct results regardless of the
    original offset. Naive timestamps are accepted and returned unchanged.

    Args:
        value: Timestamp string or ``None``.

    Returns:
        The validated (and UTC-normalised) string, or ``None`` when the
        input was ``None``.

    Raises:
        ValueError: If ``value`` is a string that is not valid ISO-8601.

    """
    if value is None:
        return value
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"Must be ISO-8601 timestamp, got {value!r}"
        raise ValueError(msg) from exc
    # Normalise timezone-aware timestamps to UTC for safe TEXT ordering.
    if parsed.tzinfo is not None:
        return parsed.astimezone(datetime.UTC).isoformat()
    return value


def parse_iso8601_to_utc(value: str) -> datetime.datetime:
    """Parse an ISO-8601 string into a UTC-normalised aware ``datetime``.

    Naive inputs are interpreted as UTC so that any two parsed instants are
    directly comparable regardless of the original offset (or the absence of
    one) — comparing a naive and an aware ``datetime`` would otherwise raise
    ``TypeError``. This mirrors :func:`validate_iso8601_nullable`, which
    normalises timezone-aware strings to UTC for lexicographic TEXT ordering,
    but returns a ``datetime`` for instant-level comparison rather than a
    string.

    Args:
        value: An ISO-8601 timestamp string (already format-validated by the
            time it reaches this helper).

    Returns:
        The parsed instant as a timezone-aware ``datetime`` in UTC.

    Raises:
        ValueError: If ``value`` is not a valid ISO-8601 timestamp.

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
            after ``valid_until``.

    Examples:
        >>> validate_interval_ordering("2026-01-01T00:00:00", None)  # open bound
        >>> validate_interval_ordering(
        ...     "2026-01-01T00:00:00", "2026-01-01T00:00:00"
        ... )  # equal instants -> accepted

    """
    if valid_from is None or valid_until is None:
        return
    if parse_iso8601_to_utc(valid_from) > parse_iso8601_to_utc(valid_until):
        msg = (
            f"valid_from ({valid_from!r}) must not be after valid_until "
            f"({valid_until!r}): an inverted validity interval is rejected"
        )
        raise ValueError(msg)

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

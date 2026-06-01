"""TTL value objects for thought auto-expiry.

Immutable data structures representing cleanup results and
expiry strategy configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique


@unique
class CleanupStrategy(StrEnum):
    """Strategy applied when cleaning up expired thoughts.

    Examples:
        >>> CleanupStrategy.ARCHIVE
        <CleanupStrategy.ARCHIVE: 'archive'>
        >>> CleanupStrategy("delete")
        <CleanupStrategy.DELETE: 'delete'>

    """

    ARCHIVE = "archive"
    DELETE = "delete"


@dataclass(frozen=True)
class CleanupResult:
    """Result of a ``cleanup_expired()`` operation.

    Attributes:
        expired_count: Number of thoughts processed.
        strategy_applied: The cleanup strategy that was used.
        timestamp: ISO-8601 UTC timestamp when cleanup was performed.

    Examples:
        >>> result = CleanupResult(
        ...     expired_count=5,
        ...     strategy_applied="archive",
        ...     timestamp="2026-04-12T10:00:00+00:00",
        ... )
        >>> result.expired_count
        5

    """

    expired_count: int
    strategy_applied: str
    timestamp: str

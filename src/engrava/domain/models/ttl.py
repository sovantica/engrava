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
        expired_count: Number of thoughts processed (archived or deleted,
            per ``strategy_applied``). A pinned thought past its TTL is never
            counted here — it is counted in ``pinned_kept_count`` instead.
        strategy_applied: The cleanup strategy that was used.
        timestamp: ISO-8601 UTC timestamp when cleanup was performed.
        pinned_kept_count: Number of past-TTL thoughts left untouched because
            ``pinned`` is set. A priority in the hygiene policy's
            ``protected_priorities`` (default ``P1``) does **not** count here:
            priority protection is a Memory Hygiene concept, and a TTL is the
            row's own explicit lifetime — only ``pinned`` exempts a row from
            expiry. Defaults to ``0`` so existing keyword construction is
            unaffected.

    Examples:
        >>> result = CleanupResult(
        ...     expired_count=5,
        ...     strategy_applied="archive",
        ...     timestamp="2026-04-12T10:00:00+00:00",
        ... )
        >>> result.expired_count
        5
        >>> result.pinned_kept_count
        0

    """

    expired_count: int
    strategy_applied: str
    timestamp: str
    pinned_kept_count: int = 0

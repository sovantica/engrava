"""Dreaming signal definitions and protocol.

Signals compute a normalized ``[0.0, 1.0]`` score for a thought during
memory consolidation.  The five default signals operate **exclusively**
on ``CoreThoughtRecord`` fields — no external dependencies.

Consumers can register custom signal functions (implementing
``DreamingSignalProtocol``) to score extended fields (e.g.
``emotional_charge``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engrava.domain.models.thought import ThoughtRecord

# ------------------------------------------------------------------
# Context & Protocol
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DreamingContext:
    """Context provided to signal functions during consolidation.

    Attributes:
        current_cycle: The cognitive cycle number for this consolidation run.
        total_thoughts: Total number of candidate thoughts being evaluated.

    Examples:
        >>> ctx = DreamingContext(current_cycle=500, total_thoughts=120)
        >>> ctx.current_cycle
        500

    """

    current_cycle: int
    total_thoughts: int


@runtime_checkable
class DreamingSignalProtocol(Protocol):
    """Single scoring signal for dreaming consolidation.

    Implementations compute a normalized score in ``[0.0, 1.0]`` from
    a thought and its context.  Higher values indicate a stronger
    signal for promotion.

    Examples:
        >>> class MySignal:
        ...     def __call__(self, thought, ctx):
        ...         return 0.5
        >>> isinstance(MySignal(), DreamingSignalProtocol)
        True

    """

    def __call__(
        self,
        thought: ThoughtRecord,
        ctx: DreamingContext,
    ) -> float:
        """Compute a signal score for the given thought.

        Args:
            thought: The thought record to score.
            ctx: Consolidation context.

        Returns:
            A float in ``[0.0, 1.0]``.

        """
        ...


# ------------------------------------------------------------------
# Default signals (core fields only)
# ------------------------------------------------------------------


class RecencySignal:
    """Score based on how recently the thought was updated.

    Uses exponential decay: ``score = exp(-decay_rate * age_cycles)``.
    More recent thoughts score higher.

    Args:
        decay_rate: Exponential decay constant (default 0.01).

    Examples:
        >>> sig = RecencySignal()
        >>> ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        >>> from engrava.domain.models.thought import ThoughtRecord
        >>> t = ThoughtRecord(
        ...     thought_id="t1", thought_type="TASK", essence="test",
        ...     content="test content", priority="P2",
        ...     lifecycle_status="ACTIVE", created_cycle=0,
        ...     updated_cycle=90, source="test",
        ... )
        >>> 0.9 < sig(t, ctx) < 1.0
        True

    """

    def __init__(self, decay_rate: float = 0.01) -> None:
        self._decay_rate = decay_rate

    def __call__(
        self,
        thought: ThoughtRecord,
        ctx: DreamingContext,
    ) -> float:
        """Compute recency score via exponential decay.

        Args:
            thought: The thought record.
            ctx: Consolidation context with current_cycle.

        Returns:
            Float in ``[0.0, 1.0]``.

        """
        age = max(0, ctx.current_cycle - thought.updated_cycle)
        return math.exp(-self._decay_rate * age)


class StalenessSignal:
    """Score based on activity span (updated_cycle - created_cycle).

    Thoughts that have been updated many times over their lifetime
    score higher — a proxy for sustained relevance.

    Args:
        max_span: Cycle span that maps to a score of 1.0 (default 100).

    Examples:
        >>> sig = StalenessSignal()
        >>> ctx = DreamingContext(current_cycle=200, total_thoughts=50)
        >>> from engrava.domain.models.thought import ThoughtRecord
        >>> t = ThoughtRecord(
        ...     thought_id="t1", thought_type="TASK", essence="test",
        ...     content="test content", priority="P2",
        ...     lifecycle_status="ACTIVE", created_cycle=0,
        ...     updated_cycle=50, source="test",
        ... )
        >>> sig(t, ctx)
        0.5

    """

    def __init__(self, max_span: int = 100) -> None:
        self._max_span = max(1, max_span)

    def __call__(
        self,
        thought: ThoughtRecord,
        ctx: DreamingContext,  # noqa: ARG002
    ) -> float:
        """Compute staleness score from lifecycle span.

        Args:
            thought: The thought record.
            ctx: Consolidation context (unused).

        Returns:
            Float in ``[0.0, 1.0]``.

        """
        span = thought.updated_cycle - thought.created_cycle
        return min(1.0, span / self._max_span)


class ConfirmationSignal:
    """Score based on ``confirmation_count``.

    More confirmations indicate higher confidence in thought quality.

    Args:
        max_count: Confirmation count that maps to 1.0 (default 5).

    Examples:
        >>> sig = ConfirmationSignal()
        >>> ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        >>> from engrava.domain.models.thought import ThoughtRecord
        >>> t = ThoughtRecord(
        ...     thought_id="t1", thought_type="TASK", essence="test",
        ...     content="test content", priority="P2",
        ...     lifecycle_status="ACTIVE", created_cycle=0,
        ...     updated_cycle=0, source="test", confirmation_count=3,
        ... )
        >>> sig(t, ctx)
        0.6

    """

    def __init__(self, max_count: int = 5) -> None:
        self._max_count = max(1, max_count)

    def __call__(
        self,
        thought: ThoughtRecord,
        ctx: DreamingContext,  # noqa: ARG002
    ) -> float:
        """Compute confirmation score.

        Args:
            thought: The thought record.
            ctx: Consolidation context (unused).

        Returns:
            Float in ``[0.0, 1.0]``.

        """
        return min(1.0, thought.confirmation_count / self._max_count)


class ConfidenceSignal:
    """Score based on the ``confidence`` field (nullable → default 0.5).

    Examples:
        >>> sig = ConfidenceSignal()
        >>> ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        >>> from engrava.domain.models.thought import ThoughtRecord
        >>> t = ThoughtRecord(
        ...     thought_id="t1", thought_type="TASK", essence="test",
        ...     content="test content", priority="P2",
        ...     lifecycle_status="ACTIVE", created_cycle=0,
        ...     updated_cycle=0, source="test", confidence=0.8,
        ... )
        >>> sig(t, ctx)
        0.8

    """

    def __call__(
        self,
        thought: ThoughtRecord,
        ctx: DreamingContext,  # noqa: ARG002
    ) -> float:
        """Return the thought's confidence (or 0.5 if null).

        Args:
            thought: The thought record.
            ctx: Consolidation context (unused).

        Returns:
            Float in ``[0.0, 1.0]``.

        """
        return thought.confidence if thought.confidence is not None else 0.5


class FrequencySignal:
    """Score based on how often a thought has been explicitly accessed.

    Uses the ``access_count`` field.
    More accesses indicate higher relevance for promotion.

    Args:
        max_accesses: Access count that maps to a score of 1.0 (default 10).

    Examples:
        >>> sig = FrequencySignal()
        >>> ctx = DreamingContext(current_cycle=100, total_thoughts=50)
        >>> from engrava.domain.models.thought import ThoughtRecord
        >>> t = ThoughtRecord(
        ...     thought_id="t1", thought_type="TASK", essence="test",
        ...     content="test content", priority="P2",
        ...     lifecycle_status="ACTIVE", created_cycle=0,
        ...     updated_cycle=0, source="test", access_count=5,
        ... )
        >>> sig(t, ctx)
        0.5

    """

    def __init__(self, max_accesses: int = 10) -> None:
        self._max_accesses = max(1, max_accesses)

    def __call__(
        self,
        thought: ThoughtRecord,
        ctx: DreamingContext,  # noqa: ARG002
    ) -> float:
        """Compute frequency score from access count.

        Args:
            thought: The thought record.
            ctx: Consolidation context (unused).

        Returns:
            Float in ``[0.0, 1.0]``.

        """
        return min(1.0, thought.access_count / self._max_accesses)


# ------------------------------------------------------------------
# Default signal registry
# ------------------------------------------------------------------

DEFAULT_SIGNALS: dict[str, type] = {
    "recency": RecencySignal,
    "staleness": StalenessSignal,
    "confirmation": ConfirmationSignal,
    "confidence": ConfidenceSignal,
    "frequency": FrequencySignal,
}
"""Registry mapping signal names to their default factory classes."""

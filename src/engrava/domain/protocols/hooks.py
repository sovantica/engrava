"""Engrava hooks — extension hook points for engrava.

Defines the contract for extensions that plug into core Engrava
lifecycle operations. ``engrava`` calls these hooks; consumer
packages provide real implementations.

The ``DefaultEngravaHooks`` class supplies no-op implementations
so that engrava can operate stand-alone without any extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from engrava.domain.models.thought import ThoughtRecord


@dataclass(frozen=True)
class ScoringContext:
    """Context passed to the ``score_function`` hook.

    Attributes:
        query_vector: Optional embedding vector for vector-based scoring.
        query_text: Optional raw text query.
        current_cycle: The cycle number at the time of scoring.

    """

    query_vector: list[float] | None = None
    query_text: str | None = None
    current_cycle: int = 0


@dataclass(frozen=True)
class MindQLExtension:
    """Registration entry for a custom MindQL command.

    Attributes:
        command_name: The MindQL command verb (e.g. ``"CUSTOM"``).
        handler: Async callable that executes the command.
        description: Human-readable description for help output.
        category: Grouping category for help/listing. Defaults to ``"extension"``.

    """

    command_name: str
    handler: Callable[..., Awaitable[list[dict[str, object]]]]
    description: str
    category: str = "extension"


@runtime_checkable
class EngravaHooksProtocol(Protocol):
    """Extension hooks called by engrava during lifecycle operations.

    ``engrava`` invokes these hooks at well-defined points.
    All hooks are ``async`` to support I/O-bound extensions.
    """

    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Enrich a thought after persistence.

        Args:
            thought: The just-persisted thought record.

        Returns:
            The (possibly enriched) thought record.

        """
        ...

    async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Enrich a thought after loading from storage.

        Args:
            thought: The thought record as loaded from storage.

        Returns:
            The (possibly enriched) thought record.

        """
        ...

    async def score_function(
        self,
        thought: ThoughtRecord,
        context: ScoringContext,
    ) -> float:
        """Compute a custom relevance score for a thought.

        Args:
            thought: The thought to score.
            context: Query-time context for scoring.

        Returns:
            A relevance score (higher = more relevant).

        """
        ...

    async def decay_function(
        self,
        thought: ThoughtRecord,
        elapsed_cycles: int,
    ) -> float:
        """Compute a decay factor for a thought over elapsed cycles.

        Args:
            thought: The thought to compute decay for.
            elapsed_cycles: Number of cycles since last access / creation.

        Returns:
            A decay multiplier in ``[0.0, 1.0]`` (1.0 = no decay).

        """
        ...

    def mindql_extension_registry(self) -> dict[str, MindQLExtension]:
        """Return custom MindQL commands registered by this extension.

        Returns:
            Mapping of command name to ``MindQLExtension`` entry.

        """
        ...


class DefaultEngravaHooks:
    """No-op default implementation of :class:`EngravaHooksProtocol`.

    Allows engrava to function without any extension installed.
    Every hook is a pass-through or returns a neutral value.
    """

    async def on_store(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Pass-through — return the thought unchanged.

        Args:
            thought: The just-persisted thought record.

        Returns:
            The same thought record, unmodified.

        """
        return thought

    async def on_retrieve(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Pass-through — return the thought unchanged.

        Args:
            thought: The thought record as loaded from storage.

        Returns:
            The same thought record, unmodified.

        """
        return thought

    async def score_function(
        self,
        thought: ThoughtRecord,
        context: ScoringContext,  # noqa: ARG002
    ) -> float:
        """Return priority-based default score (P1=4.0, P2=3.0, P3=2.0, P4=1.0).

        Args:
            thought: The thought to score.
            context: Query-time context (unused in default).

        Returns:
            Float score derived from priority (higher = more relevant).

        """
        _priority_scores = {"P1": 4.0, "P2": 3.0, "P3": 2.0, "P4": 1.0}
        return _priority_scores.get(thought.priority.value, 0.0)

    async def decay_function(
        self,
        thought: ThoughtRecord,  # noqa: ARG002
        elapsed_cycles: int,  # noqa: ARG002
    ) -> float:
        """Return ``1.0`` — no decay.

        Args:
            thought: The thought (unused in default).
            elapsed_cycles: Cycles since last access (unused in default).

        Returns:
            ``1.0`` (no decay).

        """
        return 1.0

    def mindql_extension_registry(self) -> dict[str, MindQLExtension]:
        """Return empty registry — no extensions.

        Returns:
            Empty dict.

        """
        return {}

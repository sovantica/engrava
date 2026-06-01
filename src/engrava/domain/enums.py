"""Core enumerations for the thought-graph domain.

All enums use StrEnum for human-readable serialization and JSON compatibility.
"""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class ThoughtType(StrEnum):
    """Classification of thought content.

    Examples:
        >>> ThoughtType.TASK
        <ThoughtType.TASK: 'TASK'>
        >>> ThoughtType("OBSERVATION")
        <ThoughtType.OBSERVATION: 'OBSERVATION'>

    """

    TASK = "TASK"
    OBSERVATION = "OBSERVATION"
    BELIEF = "BELIEF"
    REFLECTION = "REFLECTION"
    OUTPUT_DRAFT = "OUTPUT_DRAFT"
    NOTE = "NOTE"


@unique
class Priority(StrEnum):
    """Thought priority levels with natural ordering (P1 highest, P4 lowest).

    Examples:
        >>> Priority.P1 < Priority.P4
        True
        >>> Priority.P2 < Priority.P1
        False

    """

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"

    def __lt__(self, other: object) -> bool:
        """Compare priorities by urgency (P1 > P2 > P3 > P4).

        Args:
            other: Another Priority to compare against.

        Returns:
            True if self is higher priority (lower number) than other.

        """
        if not isinstance(other, Priority):
            return NotImplemented
        order = list(Priority)
        return order.index(self) < order.index(other)

    def __le__(self, other: object) -> bool:
        """Compare priorities for less-than-or-equal.

        Args:
            other: Another Priority to compare against.

        Returns:
            True if self is higher or equal priority to other.

        """
        if not isinstance(other, Priority):
            return NotImplemented
        return self == other or self.__lt__(other)

    def __gt__(self, other: object) -> bool:
        """Compare priorities for greater-than.

        Args:
            other: Another Priority to compare against.

        Returns:
            True if self is lower priority (higher number) than other.

        """
        if not isinstance(other, Priority):
            return NotImplemented
        return not self.__le__(other)

    def __ge__(self, other: object) -> bool:
        """Compare priorities for greater-than-or-equal.

        Args:
            other: Another Priority to compare against.

        Returns:
            True if self is lower or equal priority to other.

        """
        if not isinstance(other, Priority):
            return NotImplemented
        return not self.__lt__(other)


@unique
class LifecycleStatus(StrEnum):
    """Thought lifecycle states with enforced transitions (state machine).

    CREATED -> ACTIVE -> DONE -> ARCHIVED.

    Examples:
        >>> LifecycleStatus.CREATED.allowed_transitions()
        frozenset({<LifecycleStatus.ACTIVE: 'ACTIVE'>})
        >>> LifecycleStatus.ACTIVE.can_transition_to(LifecycleStatus.DONE)
        True
        >>> LifecycleStatus.ARCHIVED.can_transition_to(LifecycleStatus.ACTIVE)
        False

    """

    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    DONE = "DONE"
    ARCHIVED = "ARCHIVED"

    def allowed_transitions(self) -> frozenset[LifecycleStatus]:
        """Return the set of valid target states from this state.

        Returns:
            Frozenset of LifecycleStatus values reachable from this state.

        """
        transitions: dict[LifecycleStatus, frozenset[LifecycleStatus]] = {
            LifecycleStatus.CREATED: frozenset({LifecycleStatus.ACTIVE}),
            LifecycleStatus.ACTIVE: frozenset({LifecycleStatus.DONE, LifecycleStatus.ARCHIVED}),
            LifecycleStatus.DONE: frozenset({LifecycleStatus.ARCHIVED}),
            LifecycleStatus.ARCHIVED: frozenset(),
        }
        return transitions[self]

    def can_transition_to(self, target: LifecycleStatus) -> bool:
        """Check if transition to target state is allowed.

        Args:
            target: The desired target state.

        Returns:
            True if the transition is valid per the state machine.

        """
        return target in self.allowed_transitions()


@unique
class EdgeType(StrEnum):
    """Typed relations between thoughts.

    Examples:
        >>> EdgeType.ASSOCIATED
        <EdgeType.ASSOCIATED: 'ASSOCIATED'>

    """

    ASSOCIATED = "ASSOCIATED"
    DEPENDS_ON = "DEPENDS_ON"
    DERIVED_FROM = "DERIVED_FROM"
    MESSAGE_OF = "MESSAGE_OF"
    BRIDGE = "BRIDGE"
    CONSOLIDATED_FROM = "CONSOLIDATED_FROM"
    CONTESTED_BY = "CONTESTED_BY"


@unique
class KnowledgeSource(StrEnum):
    """Origin of a thought or edge in the knowledge graph.

    Examples:
        >>> KnowledgeSource.EXPERIENCE
        <KnowledgeSource.EXPERIENCE: 'EXPERIENCE'>

    """

    EXPERIENCE = "EXPERIENCE"
    SEEDED_LLM = "SEEDED_LLM"
    DISTILLED_LLM = "DISTILLED_LLM"
    DREAMING = "DREAMING"


@unique
class ActionType(StrEnum):
    """Classification of actions originating from thoughts.

    Examples:
        >>> ActionType.CLI_OUTPUT
        <ActionType.CLI_OUTPUT: 'CLI_OUTPUT'>

    """

    CLI_OUTPUT = "CLI_OUTPUT"
    TOOL_CALL = "TOOL_CALL"
    MESSAGE = "MESSAGE"
    STATE_UPDATE = "STATE_UPDATE"


@unique
class ActionStatus(StrEnum):
    """Action execution status with state machine transitions.

    Examples:
        >>> ActionStatus.PLANNED.allowed_transitions()
        frozenset({<ActionStatus.EXECUTING: 'EXECUTING'>, <ActionStatus.BLOCKED: 'BLOCKED'>})

    """

    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"

    def allowed_transitions(self) -> frozenset[ActionStatus]:
        """Return the set of valid target states from this state.

        Returns:
            Frozenset of ActionStatus values reachable from this state.

        """
        transitions: dict[ActionStatus, frozenset[ActionStatus]] = {
            ActionStatus.PLANNED: frozenset({ActionStatus.EXECUTING, ActionStatus.BLOCKED}),
            ActionStatus.EXECUTING: frozenset({ActionStatus.CONFIRMED, ActionStatus.FAILED}),
            ActionStatus.CONFIRMED: frozenset(),
            ActionStatus.FAILED: frozenset(),
            ActionStatus.BLOCKED: frozenset({ActionStatus.PLANNED}),
        }
        return transitions[self]

    def can_transition_to(self, target: ActionStatus) -> bool:
        """Check if transition to target state is allowed.

        Args:
            target: The desired target state.

        Returns:
            True if the transition is valid per the state machine.

        """
        return target in self.allowed_transitions()


@unique
class VerificationStatus(StrEnum):
    """Action verification status.

    Examples:
        >>> VerificationStatus.PENDING
        <VerificationStatus.PENDING: 'PENDING'>

    """

    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNVERIFIABLE = "UNVERIFIABLE"


@unique
class ThoughtVisibility(StrEnum):
    """Visibility level of a thought for inner/outer speech boundary.

    Controls whether a thought may be shared externally:
    - PRIVATE: never disclosed to external entities.
    - SELECTIVE: shared with trusted entities on request.
    - PUBLIC: may appear in outer speech.

    Examples:
        >>> ThoughtVisibility.PRIVATE
        <ThoughtVisibility.PRIVATE: 'private'>
        >>> ThoughtVisibility.PUBLIC
        <ThoughtVisibility.PUBLIC: 'public'>

    """

    PRIVATE = "private"
    SELECTIVE = "selective"
    PUBLIC = "public"

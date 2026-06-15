"""ThoughtRecord — core thought-graph node.

Core domain entity for persisted cognitive content.  Immutable (frozen)
— use ``evolve()`` to create modified copies.

This is the **core** ThoughtRecord with generic graph fields only.
Extension-specific fields can be added by subclassing.
"""

from __future__ import annotations

import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeAliasType

from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.exceptions import InvalidTransitionError
from engrava.domain.models._temporal import validate_iso8601_nullable

#: Allowed value types for ``ThoughtRecord.metadata`` entries.
#:
#: Leaf values must be JSON-serialisable scalars (``str``, ``int``,
#: ``float``, ``bool``, ``None``) so the column stays queryable via
#: SQLite's JSON1 ``json_extract`` without secondary parsing.  Nested
#: ``dict[str, MetadataValue]`` values are also accepted (structured
#: namespaces), which lets callers express the ``ThoughtSource`` structure
#: (e.g. ``metadata["source"] = {"is_self": bool, ...}``).
#:
#: Lists, tuples, sets, and custom objects remain rejected at write time
#: by the persistence-layer validator (caller wanting a string array
#: should encode it as a stringified JSON value or use a nested dict
#: keyed by index).
#:
#: Defined via :class:`typing_extensions.TypeAliasType` so the recursive
#: self-reference resolves correctly under Pydantic v2's runtime type
#: introspection (a plain ``TypeAlias = ... | dict[str, "MetadataValue"]``
#: triggers infinite recursion in Pydantic's schema builder on 3.11+).
MetadataValue = TypeAliasType(
    "MetadataValue",
    "str | int | float | bool | None | dict[str, MetadataValue]",
)


class ThoughtRecord(BaseModel):
    """Core persisted unit for thought-graph content.

    All fields are validated on construction.  The model is frozen — use
    ``evolve()`` to create a new instance with modified fields.

    Args:
        thought_id: Stable UUID identity.
        thought_type: Classification of thought content.
        essence: Compact canonical text used in prompts (1-200 chars).
        content: Full stored content.
        priority: Urgency level (P1 highest).
        lifecycle_status: Current state in the lifecycle state machine.
        created_cycle: First cycle this thought appeared.
        updated_cycle: Last cycle this thought was mutated.
        source: Origin of the thought (e.g., 'human', 'internal').
        confidence: Optional reliability estimate (0.0-1.0).
        embedding_ref: Optional reference to an embedding row.
        source_type: Provenance (EXPERIENCE, SEEDED_LLM, DISTILLED_LLM).
        confirmation_count: Number of experience-based confirmations.
        consolidated_from: JSON list of source thought IDs if consolidated.
        visibility: Inner/outer speech visibility (private, selective, public).
        access_count: Number of times this thought has been explicitly accessed.
        last_accessed_at: ISO-8601 datetime of last explicit access (nullable).
        created_at: ISO-8601 datetime when the thought was persisted (nullable
            for thoughts created before timestamp tracking was added).
            This is transaction time — when the fact was *recorded*.
        updated_at: ISO-8601 datetime of last mutation (nullable for legacy).
        valid_from: ISO-8601 datetime marking the start of the interval
            during which the fact is true in the world (valid time, the
            second time axis). ``None`` means an open lower bound — the
            fact is treated as valid from the beginning of time.
        valid_until: ISO-8601 datetime marking the end of the interval
            during which the fact is true in the world (valid time).
            ``None`` means an open upper bound — the fact has no known
            end and is treated as currently valid.
        metadata: Extensible structured attributes (e.g. ``role``, ``lang``,
            ``content_type``, ``session_id``, ``turn_index``, ``speaker``).
            Leaf values must be scalars (``str``, ``int``, ``float``,
            ``bool`` or ``None``); nested ``dict[str, MetadataValue]``
            values are accepted for structured namespaces (e.g.
            ``metadata["source"] = {"is_self": True, "confidence":
            "high", ...}``).  Lists and other rich containers are
            rejected at write time.  Defaults to an empty dict.

    Examples:
        >>> thought = ThoughtRecord(
        ...     thought_id="550e8400-e29b-41d4-a716-446655440000",
        ...     thought_type=ThoughtType.TASK,
        ...     essence="Analyze remote work trade-offs",
        ...     content="Evaluate pros and cons of remote work...",
        ...     priority=Priority.P2,
        ...     lifecycle_status=LifecycleStatus.CREATED,
        ...     created_cycle=0,
        ...     updated_cycle=0,
        ...     source="human",
        ... )
        >>> thought.is_active()
        False

    """

    model_config = ConfigDict(frozen=True)

    thought_id: str
    thought_type: ThoughtType
    essence: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    priority: Priority
    lifecycle_status: LifecycleStatus
    created_cycle: int = Field(default=0, ge=0)
    updated_cycle: int = Field(default=0, ge=0)
    source: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding_ref: str | None = None
    source_type: KnowledgeSource = KnowledgeSource.EXPERIENCE
    confirmation_count: int = Field(default=0, ge=0)
    consolidated_from: list[str] | None = None
    visibility: ThoughtVisibility = ThoughtVisibility.SELECTIVE
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    expires_at: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_cycle_ordering(self) -> Self:
        """Ensure updated_cycle >= created_cycle."""
        if self.updated_cycle < self.created_cycle:
            msg = (
                f"updated_cycle ({self.updated_cycle}) must be >= "
                f"created_cycle ({self.created_cycle})"
            )
            raise ValueError(msg)
        return self

    @field_validator("thought_id")
    @classmethod
    def _validate_thought_id_not_empty(cls, v: str) -> str:
        """Validate that thought_id is not empty or whitespace."""
        if not v.strip():
            msg = "thought_id must not be empty or whitespace"
            raise ValueError(msg)
        return v

    @field_validator(
        "created_at",
        "updated_at",
        "last_accessed_at",
        "expires_at",
        "valid_from",
        "valid_until",
    )
    @classmethod
    def _validate_iso8601_nullable(cls, v: str | None) -> str | None:
        """Validate ISO-8601 format and normalize to UTC when not None.

        Timezone-aware timestamps are converted to UTC so that SQLite
        TEXT comparisons (lexicographic) produce correct results
        regardless of the original offset.

        Args:
            v: Timestamp string or None.

        Returns:
            The validated (and UTC-normalized) string, or None.

        Raises:
            ValueError: If string is not valid ISO-8601.

        """
        return validate_iso8601_nullable(v)

    def is_active(self) -> bool:
        """Check if the thought is in ACTIVE lifecycle status.

        Returns:
            True if lifecycle_status is ACTIVE.

        """
        return self.lifecycle_status == LifecycleStatus.ACTIVE

    def is_archivable(self) -> bool:
        """Check if the thought can transition to ARCHIVED.

        Returns:
            True if current status allows transition to ARCHIVED.

        """
        return self.lifecycle_status.can_transition_to(LifecycleStatus.ARCHIVED)

    def can_transition_to(self, target: LifecycleStatus) -> bool:
        """Check if transition to target lifecycle status is allowed.

        Args:
            target: The desired target lifecycle status.

        Returns:
            True if the transition is valid per the state machine.

        """
        return self.lifecycle_status.can_transition_to(target)

    def evolve(self, **changes: object) -> Self:
        """Create a new ThoughtRecord with specified fields changed.

        Validates lifecycle transitions when lifecycle_status is changed.
        Automatically sets ``updated_at`` to the current UTC time unless
        the caller provides an explicit value.  ``created_at`` is
        immutable — attempting to change it raises ``ValueError``.

        Args:
            **changes: Fields to override in the new instance.

        Returns:
            A new ThoughtRecord with the specified changes applied.

        Raises:
            InvalidTransitionError: If lifecycle_status change is invalid.
            ValueError: If caller attempts to change ``created_at``.

        Examples:
            >>> new_thought = thought.evolve(
            ...     lifecycle_status=LifecycleStatus.ACTIVE, updated_cycle=1
            ... )

        """
        if "created_at" in changes and self.created_at is not None:
            new_val = changes["created_at"]
            if new_val != self.created_at:
                msg = (
                    "created_at is immutable once set "
                    f"(current={self.created_at!r}, attempted={new_val!r})"
                )
                raise ValueError(msg)
        if "lifecycle_status" in changes:
            target = changes["lifecycle_status"]
            if isinstance(target, LifecycleStatus) and not self.can_transition_to(target):
                raise InvalidTransitionError(
                    entity_type="LifecycleStatus",
                    current_state=self.lifecycle_status.value,
                    target_state=target.value,
                )
        if "updated_at" not in changes:
            changes["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        current_data = self.model_dump()
        current_data.update(changes)
        return type(self).model_validate(current_data)

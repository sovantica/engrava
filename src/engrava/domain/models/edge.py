"""EdgeRecord — typed relation between thoughts.

Represents a lightweight directional edge in the thought graph.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engrava.domain.enums import EdgeType, KnowledgeSource
from engrava.domain.models._temporal import (
    validate_interval_ordering,
    validate_iso8601_nullable,
)


class EdgeRecord(BaseModel):
    """Lightweight typed relation between two thoughts.

    Args:
        edge_id: Stable UUID identity.
        from_thought_id: Source thought UUID.
        to_thought_id: Target thought UUID.
        edge_type: Classification of the relationship.
        weight: Relation strength (0.0-1.0).
        created_cycle: Cycle when this edge was created.
        source: Provenance of the edge (EXPERIENCE, SEEDED_LLM, DISTILLED_LLM).
        decay_multiplier: Multiplier for accelerated decay (1.0 normal).
        valid_from: ISO-8601 datetime marking the start of the interval
            during which the relation is true in the world (valid time).
            ``None`` means an open lower bound — the relation is treated
            as valid from the beginning of time.
        valid_until: ISO-8601 datetime marking the end of the interval
            during which the relation is true in the world (valid time).
            ``None`` means an open upper bound — the relation has no
            known end and is treated as currently valid.

    Examples:
        >>> edge = EdgeRecord(
        ...     edge_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        ...     from_thought_id="thought-001",
        ...     to_thought_id="thought-002",
        ...     edge_type=EdgeType.ASSOCIATED,
        ...     weight=0.8,
        ...     created_cycle=1,
        ... )

    """

    model_config = ConfigDict(frozen=True)

    edge_id: str
    from_thought_id: str
    to_thought_id: str
    edge_type: EdgeType
    weight: float = Field(ge=0.0, le=1.0)
    created_cycle: int = Field(ge=0)
    source: KnowledgeSource = KnowledgeSource.EXPERIENCE
    decay_multiplier: float = Field(default=1.0, ge=0.0)
    valid_from: str | None = None
    valid_until: str | None = None

    @field_validator("edge_id", "from_thought_id", "to_thought_id")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        """Validate that ID fields are not empty or whitespace."""
        if not v.strip():
            msg = "ID field must not be empty or whitespace"
            raise ValueError(msg)
        return v

    @field_validator("valid_from", "valid_until")
    @classmethod
    def _validate_iso8601_nullable(cls, v: str | None) -> str | None:
        """Validate ISO-8601 format and normalize to UTC when not None.

        Uses the shared timestamp validator so edge valid-time fields
        normalise timezone-aware values to UTC exactly like the thought
        record's timestamp columns, keeping SQLite TEXT ordering correct.

        Args:
            v: Timestamp string or None.

        Returns:
            The validated (and UTC-normalized) string, or None.

        Raises:
            ValueError: If string is not valid ISO-8601.

        """
        return validate_iso8601_nullable(v)

    @model_validator(mode="after")
    def _validate_valid_interval(self) -> Self:
        """Reject an inverted valid-time interval.

        When both ``valid_from`` and ``valid_until`` are set, the validity
        interval must not run backwards. The bounds are compared as
        UTC-normalised instants (not raw strings), so equal instants across
        differing offsets are accepted (a zero-length interval is a legitimate
        instantaneous relation) while a strictly inverted pair is rejected. A
        ``None`` on either bound is an open interval and is always accepted.

        Raises:
            ValueError: If ``valid_from`` is strictly after ``valid_until``.

        """
        validate_interval_ordering(self.valid_from, self.valid_until)
        return self

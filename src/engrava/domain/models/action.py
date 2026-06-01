"""ActionRecord — persisted external or internal action attempt."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engrava.domain.enums import ActionStatus, ActionType, VerificationStatus
from engrava.domain.exceptions import InvalidTransitionError


class ActionRecord(BaseModel):
    """Persisted external or internal action attempt.

    Args:
        action_id: Stable UUID identity.
        source_thought_id: UUID of the originating thought.
        action_type: Classification of the action.
        intent: Intended effect description.
        status: Current execution status.
        verification_status: Current verification status.
        raw_metrics_json: Optional ground-truth facts for verification.

    Examples:
        >>> action = ActionRecord(
        ...     action_id="act-001",
        ...     source_thought_id="thought-001",
        ...     action_type=ActionType.CLI_OUTPUT,
        ...     intent="Display analysis results",
        ...     status=ActionStatus.PLANNED,
        ...     verification_status=VerificationStatus.PENDING,
        ... )

    """

    model_config = ConfigDict(frozen=True)

    action_id: str
    source_thought_id: str
    action_type: ActionType
    intent: str = Field(min_length=1)
    status: ActionStatus
    verification_status: VerificationStatus
    raw_metrics_json: str | None = None

    @field_validator("action_id", "source_thought_id")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        """Validate that ID fields are not empty or whitespace."""
        if not v.strip():
            msg = "ID field must not be empty or whitespace"
            raise ValueError(msg)
        return v

    def can_transition_to(self, target: ActionStatus) -> bool:
        """Check if transition to target action status is allowed.

        Args:
            target: The desired target action status.

        Returns:
            True if the transition is valid per the state machine.

        """
        return self.status.can_transition_to(target)

    def evolve(self, **changes: object) -> ActionRecord:
        """Create a new ActionRecord with specified fields changed.

        Validates status transitions when status is changed.

        Args:
            **changes: Fields to override in the new instance.

        Returns:
            A new ActionRecord with the specified changes applied.

        Raises:
            InvalidTransitionError: If status change is invalid.

        """
        if "status" in changes:
            target = changes["status"]
            if isinstance(target, ActionStatus) and not self.can_transition_to(target):
                raise InvalidTransitionError(
                    entity_type="ActionStatus",
                    current_state=self.status.value,
                    target_state=target.value,
                )
        current_data = self.model_dump()
        current_data.update(changes)
        return ActionRecord.model_validate(current_data)

"""Unit tests for the valid-time fields on the core domain records.

Covers the nullable ISO-8601 ``valid_from`` / ``valid_until`` fields on
both :class:`ThoughtRecord` and :class:`EdgeRecord`: ``None`` and valid
ISO values are accepted, malformed strings are rejected, and
timezone-aware values are normalised to UTC. The normalisation logic is
shared via :mod:`engrava.domain.models._temporal`, so both records are
exercised in parallel to guard against drift.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engrava.domain.enums import EdgeType, LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import EdgeRecord, ThoughtRecord
from engrava.domain.models._temporal import validate_iso8601_nullable


def _make_thought(**overrides: object) -> ThoughtRecord:
    base: dict[str, object] = {
        "thought_id": "t-1",
        "thought_type": ThoughtType.OBSERVATION,
        "essence": "e",
        "content": "c",
        "priority": Priority.P2,
        "lifecycle_status": LifecycleStatus.CREATED,
        "created_cycle": 0,
        "updated_cycle": 0,
        "source": "test",
    }
    base.update(overrides)
    return ThoughtRecord(**base)  # type: ignore[arg-type]


def _make_edge(**overrides: object) -> EdgeRecord:
    base: dict[str, object] = {
        "edge_id": "e-1",
        "from_thought_id": "t-1",
        "to_thought_id": "t-2",
        "edge_type": EdgeType.ASSOCIATED,
        "weight": 0.5,
        "created_cycle": 0,
    }
    base.update(overrides)
    return EdgeRecord(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


class TestValidateIso8601Nullable:
    """Direct tests for the shared validation helper."""

    def test_none_passes_through(self) -> None:
        assert validate_iso8601_nullable(None) is None

    def test_naive_returned_unchanged(self) -> None:
        assert validate_iso8601_nullable("2026-01-01T00:00:00") == "2026-01-01T00:00:00"

    def test_positive_offset_normalised_to_utc(self) -> None:
        assert validate_iso8601_nullable("2026-04-12T15:00:00+02:00") == "2026-04-12T13:00:00+00:00"

    def test_malformed_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Must be ISO-8601 timestamp"):
            validate_iso8601_nullable("not-a-timestamp")


# ---------------------------------------------------------------------------
# ThoughtRecord
# ---------------------------------------------------------------------------


class TestThoughtValidTime:
    def test_defaults_to_none(self) -> None:
        thought = _make_thought()
        assert thought.valid_from is None
        assert thought.valid_until is None

    def test_valid_iso_accepted(self) -> None:
        thought = _make_thought(
            valid_from="2026-01-01T00:00:00",
            valid_until="2026-12-31T00:00:00",
        )
        assert thought.valid_from == "2026-01-01T00:00:00"
        assert thought.valid_until == "2026-12-31T00:00:00"

    def test_tz_aware_normalised_to_utc(self) -> None:
        thought = _make_thought(
            valid_from="2026-04-12T15:00:00+02:00",
            valid_until="2026-04-12T10:00:00-05:00",
        )
        assert thought.valid_from == "2026-04-12T13:00:00+00:00"
        assert thought.valid_until == "2026-04-12T15:00:00+00:00"

    @pytest.mark.parametrize("field", ["valid_from", "valid_until"])
    def test_malformed_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match="Must be ISO-8601 timestamp"):
            _make_thought(**{field: "garbage"})


# ---------------------------------------------------------------------------
# EdgeRecord
# ---------------------------------------------------------------------------


class TestEdgeValidTime:
    def test_defaults_to_none(self) -> None:
        edge = _make_edge()
        assert edge.valid_from is None
        assert edge.valid_until is None

    def test_valid_iso_accepted(self) -> None:
        edge = _make_edge(
            valid_from="2026-01-01T00:00:00",
            valid_until="2026-12-31T00:00:00",
        )
        assert edge.valid_from == "2026-01-01T00:00:00"
        assert edge.valid_until == "2026-12-31T00:00:00"

    def test_tz_aware_normalised_to_utc(self) -> None:
        edge = _make_edge(
            valid_from="2026-04-12T15:00:00+02:00",
            valid_until="2026-04-12T10:00:00-05:00",
        )
        assert edge.valid_from == "2026-04-12T13:00:00+00:00"
        assert edge.valid_until == "2026-04-12T15:00:00+00:00"

    @pytest.mark.parametrize("field", ["valid_from", "valid_until"])
    def test_malformed_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match="Must be ISO-8601 timestamp"):
            _make_edge(**{field: "garbage"})

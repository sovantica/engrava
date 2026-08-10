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
from engrava.domain.models._temporal import (
    validate_interval_ordering,
    validate_iso8601_nullable,
)


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


class TestValidateIntervalOrdering:
    """Direct tests for the shared interval-ordering invariant."""

    def test_open_lower_bound_accepted(self) -> None:
        assert validate_interval_ordering(None, "2026-01-01T00:00:00") is None

    def test_open_upper_bound_accepted(self) -> None:
        assert validate_interval_ordering("2026-01-01T00:00:00", None) is None

    def test_both_open_accepted(self) -> None:
        assert validate_interval_ordering(None, None) is None

    def test_ordered_interval_accepted(self) -> None:
        assert validate_interval_ordering("2026-01-01T00:00:00", "2026-12-31T00:00:00") is None

    def test_equal_instant_accepted(self) -> None:
        assert validate_interval_ordering("2026-01-01T00:00:00", "2026-01-01T00:00:00") is None

    def test_equal_across_offsets_accepted(self) -> None:
        # 12:00+02:00 and 05:00-05:00 are the same UTC instant (10:00Z).
        assert (
            validate_interval_ordering("2026-06-01T12:00:00+02:00", "2026-06-01T05:00:00-05:00")
            is None
        )

    def test_naive_equals_aware_accepted(self) -> None:
        # Same instant, one naive (interpreted as UTC) and one UTC-aware. A raw
        # string comparison would order these differently and wrongly reject
        # the pair; instant comparison treats them as equal.
        assert (
            validate_interval_ordering("2026-06-01T10:00:00+00:00", "2026-06-01T10:00:00") is None
        )

    def test_inverted_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="inverted validity interval"):
            validate_interval_ordering("2026-06-01T00:00:00", "2026-03-01T00:00:00")

    def test_inverted_across_offsets_rejected(self) -> None:
        # until 01:00+05:00 == 2026-05-31T20:00Z, which precedes from 00:00Z.
        with pytest.raises(ValueError, match="inverted validity interval"):
            validate_interval_ordering("2026-06-01T00:00:00+00:00", "2026-06-01T01:00:00+05:00")

    def test_microsecond_ordered_accepted(self) -> None:
        # One microsecond apart, in order — accepted.
        assert (
            validate_interval_ordering("2026-01-01T00:00:00.000000", "2026-01-01T00:00:00.000001")
            is None
        )

    def test_microsecond_inverted_rejected(self) -> None:
        # One microsecond apart, inverted — rejected at sub-second precision.
        with pytest.raises(ValueError, match="inverted validity interval"):
            validate_interval_ordering("2026-01-01T00:00:00.000001", "2026-01-01T00:00:00.000000")

    def test_microsecond_equal_across_offsets_accepted(self) -> None:
        # 12:00:00.500000+02:00 and 10:00:00.500000+00:00 are the same instant.
        assert (
            validate_interval_ordering(
                "2026-06-01T12:00:00.500000+02:00", "2026-06-01T10:00:00.500000+00:00"
            )
            is None
        )


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

    def test_inverted_interval_rejected(self) -> None:
        # AC-1: valid_from strictly after valid_until is rejected.
        with pytest.raises(ValidationError, match="inverted validity interval"):
            _make_thought(
                valid_from="2026-06-01T00:00:00",
                valid_until="2026-03-01T00:00:00",
            )

    def test_equal_instant_accepted(self) -> None:
        # AC-2: a zero-length interval is a legitimate instantaneous fact.
        thought = _make_thought(
            valid_from="2026-01-01T00:00:00",
            valid_until="2026-01-01T00:00:00",
        )
        assert thought.valid_from == thought.valid_until == "2026-01-01T00:00:00"

    def test_equal_across_offsets_accepted(self) -> None:
        # AC-2: equal instants expressed with differing offsets normalise equal.
        thought = _make_thought(
            valid_from="2026-06-01T12:00:00+02:00",
            valid_until="2026-06-01T05:00:00-05:00",
        )
        assert thought.valid_from == thought.valid_until == "2026-06-01T10:00:00+00:00"

    def test_naive_equals_aware_accepted(self) -> None:
        # AC-2: same instant, one naive and one UTC-aware. Instant comparison
        # accepts it; a raw-string comparison would wrongly reject it.
        thought = _make_thought(
            valid_from="2026-06-01T10:00:00+00:00",
            valid_until="2026-06-01T10:00:00",
        )
        assert thought.valid_until == "2026-06-01T10:00:00"

    @pytest.mark.parametrize(
        ("valid_from", "valid_until"),
        [
            ("2026-01-01T00:00:00", None),
            (None, "2026-01-01T00:00:00"),
            (None, None),
        ],
    )
    def test_open_bounds_accepted(self, valid_from: str | None, valid_until: str | None) -> None:
        # AC-3: a NULL on either bound preserves the open interval.
        thought = _make_thought(valid_from=valid_from, valid_until=valid_until)
        assert thought.valid_from == valid_from
        assert thought.valid_until == valid_until


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

    def test_inverted_interval_rejected(self) -> None:
        # AC-1: valid_from strictly after valid_until is rejected.
        with pytest.raises(ValidationError, match="inverted validity interval"):
            _make_edge(
                valid_from="2026-06-01T00:00:00",
                valid_until="2026-03-01T00:00:00",
            )

    def test_equal_instant_accepted(self) -> None:
        # AC-2: a zero-length interval is a legitimate instantaneous relation.
        edge = _make_edge(
            valid_from="2026-01-01T00:00:00",
            valid_until="2026-01-01T00:00:00",
        )
        assert edge.valid_from == edge.valid_until == "2026-01-01T00:00:00"

    def test_equal_across_offsets_accepted(self) -> None:
        # AC-2: equal instants expressed with differing offsets normalise equal.
        edge = _make_edge(
            valid_from="2026-06-01T12:00:00+02:00",
            valid_until="2026-06-01T05:00:00-05:00",
        )
        assert edge.valid_from == edge.valid_until == "2026-06-01T10:00:00+00:00"

    def test_naive_equals_aware_accepted(self) -> None:
        # AC-2: same instant, one naive and one UTC-aware. Instant comparison
        # accepts it; a raw-string comparison would wrongly reject it.
        edge = _make_edge(
            valid_from="2026-06-01T10:00:00+00:00",
            valid_until="2026-06-01T10:00:00",
        )
        assert edge.valid_until == "2026-06-01T10:00:00"

    @pytest.mark.parametrize(
        ("valid_from", "valid_until"),
        [
            ("2026-01-01T00:00:00", None),
            (None, "2026-01-01T00:00:00"),
            (None, None),
        ],
    )
    def test_open_bounds_accepted(self, valid_from: str | None, valid_until: str | None) -> None:
        # AC-3: a NULL on either bound preserves the open interval.
        edge = _make_edge(valid_from=valid_from, valid_until=valid_until)
        assert edge.valid_from == valid_from
        assert edge.valid_until == valid_until

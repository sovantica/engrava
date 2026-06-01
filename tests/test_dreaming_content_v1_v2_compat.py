"""Backward-compat tests — v1 reader handles v2 content gracefully.

The dreaming extension now emits structural REFLECTION content schema
v2, but legacy code paths and external consumers built against v1
must keep working.  These tests pin three properties:

* A v1-only reader (i.e. code that reads only the three legacy fields
  ``member_ids``, ``keywords``, ``cluster_hash``) keeps working when
  handed a v2 dict — extra fields are JSON noise from its perspective
  and ``json.loads`` does not raise on them.
* The v2 reader (``parse_reflection_content``) round-trips legacy v1
  content (no ``version`` field) without error.
* JSON serialisation round-trips both schemas byte-for-byte; nothing
  in v2 needs ``ensure_ascii=False`` for ASCII-only fixtures, but
  the unicode round-trip is exercised separately.
"""

from __future__ import annotations

import datetime
import json

import pytest

from engrava.config import DreamingConfig
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming_reflection_content import (
    build_reflection_content_v2,
    parse_reflection_content,
)

_FIXED_NOW = datetime.datetime(2026, 4, 30, 10, 0, 0, tzinfo=datetime.UTC)


def _thought(*, thought_id: str, content: str) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        created_at="2026-04-29T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Legacy v1 → v1 reader (no version field)
# ---------------------------------------------------------------------------


class TestLegacyV1Roundtrip:
    """Legacy v1 content (3 fields, no version key) round-trips through dispatch."""

    def test_dispatch_returns_v1_content_verbatim(self) -> None:
        legacy = {
            "member_ids": ["t-001", "t-002"],
            "keywords": ["alpha", "beta"],
            "cluster_hash": "abc1234567890def",
        }
        parsed = parse_reflection_content(legacy)
        assert parsed == legacy

    def test_v1_json_serialisation_roundtrip(self) -> None:
        legacy = {
            "member_ids": ["t-001"],
            "keywords": ["alpha"],
            "cluster_hash": "deadbeef",
        }
        encoded = json.dumps(legacy, sort_keys=True)
        decoded = json.loads(encoded)
        assert decoded == legacy
        assert "version" not in decoded


# ---------------------------------------------------------------------------
# v1 reader sees a v2 dict — extra fields harmless
# ---------------------------------------------------------------------------


class TestV1ReaderOnV2Content:
    """A reader that consumes only the 3 legacy fields ignores v2 enrichments."""

    def test_v1_reader_finds_three_legacy_fields(self) -> None:
        cluster = [_thought(thought_id="t-001", content="alpha beta gamma")]
        v2 = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        # Simulate a v1-only reader that only knows about legacy fields.
        legacy_view = {k: v2[k] for k in ("member_ids", "keywords", "cluster_hash")}
        assert legacy_view["member_ids"] == ["t-001"]
        assert isinstance(legacy_view["keywords"], list)
        assert isinstance(legacy_view["cluster_hash"], str)

    def test_v1_reader_does_not_crash_on_v2_extra_fields(self) -> None:
        """``json.loads`` does not raise on extra fields, so v1 reader is safe."""
        cluster = [_thought(thought_id="t-001", content="alpha beta gamma")]
        v2 = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        encoded = json.dumps(v2, sort_keys=True, ensure_ascii=False)
        decoded = json.loads(encoded)
        # All v2 enrichment fields land in the dict; v1 reader simply
        # ignores them without raising.
        for extra in ("type", "version", "top_keyphrases", "member_excerpts"):
            assert extra in decoded


# ---------------------------------------------------------------------------
# v2 reader on v2 content
# ---------------------------------------------------------------------------


class TestV2ReaderOnV2Content:
    """The v2 dispatch path returns the dict verbatim and JSON round-trips."""

    def test_v2_passthrough_via_parse(self) -> None:
        cluster = [_thought(thought_id="t-001", content="alpha beta")]
        v2 = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        assert parse_reflection_content(v2) is v2

    def test_v2_json_roundtrip_byte_identical(self) -> None:
        cluster = [_thought(thought_id="t-001", content="alpha beta")]
        v2 = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        encoded = json.dumps(v2, sort_keys=True, ensure_ascii=False)
        decoded = json.loads(encoded)
        re_encoded = json.dumps(decoded, sort_keys=True, ensure_ascii=False)
        assert encoded == re_encoded


# ---------------------------------------------------------------------------
# Future / unknown versions raise explicitly
# ---------------------------------------------------------------------------


class TestUnsupportedVersion:
    def test_unknown_version_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported reflection content version"):
            parse_reflection_content({"version": 99, "type": "reflection"})

    def test_v1_field_set_with_extra_version_3_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported reflection content version"):
            parse_reflection_content(
                {
                    "version": 3,
                    "member_ids": ["t-001"],
                    "keywords": [],
                    "cluster_hash": "x",
                },
            )

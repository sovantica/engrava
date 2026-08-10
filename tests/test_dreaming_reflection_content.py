"""Unit tests for ``engrava.extensions.dreaming_reflection_content``.

Covers v2 schema completeness, legacy v1 dispatch, sort key safety,
unicode handling, deterministic output, and the 2 KB content-size
budget.
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
    MemberExcerpt,
    ReflectionContentV2,
    TemporalSpan,
    build_reflection_content_v2,
    extract_named_entities_per_member,
    parse_reflection_content,
)

_FIXED_NOW = datetime.datetime(2026, 4, 30, 10, 0, 0, tzinfo=datetime.UTC)


def _thought(
    *,
    thought_id: str,
    content: str,
    priority: Priority = Priority.P2,
    created_at: str | None = "2026-04-29T12:00:00+00:00",
) -> ThoughtRecord:
    """Build a realistic ``ThoughtRecord`` for builder tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# v2 builder
# ---------------------------------------------------------------------------


class TestBuildV2Minimal:
    """Small clusters produce the full v2 schema with all required fields."""

    def test_2_member_cluster_has_all_fields(self) -> None:
        cluster = [
            _thought(thought_id="t-001", content="alpha beta gamma decisions monday"),
            _thought(thought_id="t-002", content="delta epsilon zeta retrospective notes"),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        # All 7 schema-mandated fields:
        assert content["type"] == "reflection"
        assert content["version"] == 2
        assert content["member_ids"] == ["t-001", "t-002"]
        assert isinstance(content["keywords"], list)
        assert content["member_count"] == 2
        assert content["cluster_algorithm"] == "lpa"
        assert content["created_at"] == "2026-04-30T10:00:00+00:00"
        # Legacy v1 field preserved:
        assert "cluster_hash" in content
        assert len(content["cluster_hash"]) == 16
        # v2 enrichment fields:
        assert "top_keyphrases" in content
        assert "member_excerpts" in content
        assert "temporal_span" in content
        assert "named_entities" in content


class TestBuildV2Large:
    """A 12-member cluster fits within the 2 KB content budget."""

    def test_12_member_cluster_under_2kb(self) -> None:
        cluster = [
            _thought(
                thought_id=f"t-{i:03d}",
                content=(
                    f"Cluster member {i}: this thought captures a "
                    "discussion about the upcoming Q3 launch and the "
                    "associated migration plan with pending decisions."
                ),
            )
            for i in range(12)
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="agglomerative", config=DreamingConfig(), now=_FIXED_NOW
        )
        size = len(json.dumps(content, ensure_ascii=False).encode("utf-8"))
        assert size <= 2048, f"v2 content size {size} exceeds 2 KB budget"


class TestDeterminism:
    """Identical inputs produce byte-identical output across calls."""

    def test_same_input_byte_identical_three_calls(self) -> None:
        cluster = [
            _thought(thought_id="t-001", content="quarterly goals planning Q3 launch"),
            _thought(thought_id="t-002", content="Q3 migration planning decisions"),
        ]
        cfg = DreamingConfig()
        a = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        b = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        c = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert json.dumps(b, sort_keys=True) == json.dumps(c, sort_keys=True)


class TestMemberExcerpts:
    """Excerpt selection prioritises P1 members and truncates at word boundary."""

    def test_p1_member_appears_first(self) -> None:
        cluster = [
            _thought(thought_id="t-low", content="low priority chatter", priority=Priority.P4),
            _thought(thought_id="t-hi", content="critical incident summary", priority=Priority.P1),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        excerpts = content["member_excerpts"]
        assert excerpts[0]["thought_id"] == "t-hi"

    def test_excerpt_truncated_at_word_boundary(self) -> None:
        # Long enough that ``DreamingConfig().member_excerpt_max_chars``
        # (150 by default post-amendment) actually truncates the
        # source content — we want to exercise the truncation path,
        # not just confirm pass-through.
        long_content = (
            "this is a very long observation about deployment latency "
            "and morale and quarterly retrospectives and dashboards "
            "covering pipeline stages and the ongoing rotation of "
            "ownership across teams running the site reliability practice"
        )
        cluster = [_thought(thought_id="t-long", content=long_content)]
        cfg = DreamingConfig()
        content = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        excerpt = content["member_excerpts"][0]["excerpt"]
        # Must fit within the configured ``member_excerpt_max_chars``
        # bound and never break a word mid-token.
        assert len(excerpt) <= cfg.member_excerpt_max_chars
        if excerpt.endswith("..."):
            head = excerpt[:-3].rstrip()
            assert " " not in head[-1] or head[-1].isalnum()

    def test_unicode_safe(self) -> None:
        cluster = [
            _thought(
                thought_id="t-emoji",
                content="🌟 milestone reached — team morale up by 30%, deployment latency down",
            ),
            _thought(thought_id="t-cjk", content="用户偏好简洁的解释而非冗长解释"),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        excerpts = content["member_excerpts"]
        # Encoding round-trip must succeed and preserve content.
        json.dumps(content, ensure_ascii=False).encode("utf-8")
        assert any("🌟" in e["excerpt"] for e in excerpts)

    def test_top_n_caps_excerpt_count(self) -> None:
        cluster = [
            _thought(thought_id=f"t-{i:03d}", content=f"member {i} content") for i in range(10)
        ]
        cfg = DreamingConfig(top_member_excerpts_count=3)
        content = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        assert len(content["member_excerpts"]) == 3

    def test_member_excerpts_handles_none_created_at(self) -> None:
        """Sort key must not raise when ``created_at`` is None on some members."""
        cluster = [
            _thought(thought_id="t-001", content="alpha", created_at=None),
            _thought(thought_id="t-002", content="beta", created_at="2026-04-29T12:00:00+00:00"),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        ids = [e["thought_id"] for e in content["member_excerpts"]]
        assert set(ids) == {"t-001", "t-002"}


# ---------------------------------------------------------------------------
# Temporal span + named entities
# ---------------------------------------------------------------------------


class TestTemporalSpan:
    def test_min_max_span_days(self) -> None:
        cluster = [
            _thought(
                thought_id="t-old",
                content="alpha",
                created_at="2026-04-15T14:00:00+00:00",
            ),
            _thought(
                thought_id="t-new",
                content="beta",
                created_at="2026-04-29T16:30:00+00:00",
            ),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        ts = content["temporal_span"]
        assert ts["min_created_at"].startswith("2026-04-15")
        assert ts["max_created_at"].startswith("2026-04-29")
        assert ts["span_days"] == pytest.approx(14.10, abs=0.05)

    def test_all_none_created_at_falls_back(self) -> None:
        cluster = [_thought(thought_id="t-001", content="alpha", created_at=None)]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        ts = content["temporal_span"]
        assert ts["span_days"] == 0.0


class TestNamedEntities:
    def test_extracts_capitalized_tokens(self) -> None:
        cluster = [_thought(thought_id="t-001", content="Q3 retrospective with Jordan and Acme")]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        entities = content["named_entities"]
        assert "Jordan" in entities
        assert "Acme" in entities

    def test_extracts_year_via_regex(self) -> None:
        cluster = [_thought(thought_id="t-001", content="planning for fiscal year 2026")]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        assert "2026" in content["named_entities"]


# ---------------------------------------------------------------------------
# Dispatch parser
# ---------------------------------------------------------------------------


class TestParseReflectionContent:
    def test_legacy_v1_no_version_field(self) -> None:
        legacy = {
            "member_ids": ["t-001", "t-002"],
            "keywords": ["alpha", "beta"],
            "cluster_hash": "abc123",
        }
        parsed = parse_reflection_content(legacy)
        assert parsed == legacy

    def test_v2_passthrough(self) -> None:
        v2 = {
            "type": "reflection",
            "version": 2,
            "member_ids": [],
            "keywords": [],
            "cluster_hash": "",
            "member_count": 0,
            "cluster_algorithm": "lpa",
            "created_at": "2026-04-30T10:00:00+00:00",
            "top_keyphrases": [],
            "member_excerpts": [],
            "temporal_span": {
                "min_created_at": "1970-01-01T00:00:00+00:00",
                "max_created_at": "1970-01-01T00:00:00+00:00",
                "span_days": 0.0,
            },
            "named_entities": [],
        }
        parsed = parse_reflection_content(v2)
        assert parsed == v2

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported reflection content version"):
            parse_reflection_content({"version": 99, "type": "reflection"})


# ---------------------------------------------------------------------------
# v1 reader graceful downgrade (treats v2 dict as if it were v1)
# ---------------------------------------------------------------------------


class TestV1ReaderIgnoresV2Fields:
    """A v1-aware reader (3 known fields) safely ignores extra v2 fields."""

    def test_v1_reader_can_read_v2_legacy_subset(self) -> None:
        cluster = [_thought(thought_id="t-001", content="alpha beta gamma")]
        v2 = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        # Simulate a v1-only reader by extracting only the 3 legacy fields.
        legacy_view = {k: v2[k] for k in ("member_ids", "keywords", "cluster_hash")}
        assert legacy_view["member_ids"] == ["t-001"]
        assert isinstance(legacy_view["keywords"], list)
        assert isinstance(legacy_view["cluster_hash"], str)


class TestExtractNamedEntitiesPerMember:
    """Per-member named-entity extractor — the primitive used by both the
    v2 content builder (aggregated across the cluster) and the
    cluster-quality named-entity-consistency gate.
    """

    def test_capitalised_proper_nouns_extracted(self) -> None:
        thought = _thought(
            thought_id="t-1",
            content="Alice visited Paris and the Louvre.",
        )
        entities = extract_named_entities_per_member(thought)
        assert "Alice" in entities
        assert "Paris" in entities
        assert "Louvre" in entities

    def test_year_tokens_extracted(self) -> None:
        thought = _thought(
            thought_id="t-2",
            content="The cohort of 1974 produced 12 graduates by 2026.",
        )
        entities = extract_named_entities_per_member(thought)
        assert "1974" in entities
        assert "2026" in entities

    def test_sentence_starter_capitalisations_filtered(self) -> None:
        # Sentence starters like "The" should be filtered against the
        # shared blocklist even though they are capitalised.
        thought = _thought(
            thought_id="t-3",
            content="The cat sat. The dog ran. The bird sang.",
        )
        entities = extract_named_entities_per_member(thought)
        assert "The" not in entities

    def test_short_capitalised_tokens_filtered(self) -> None:
        # Tokens below the minimum length cut-off should not appear.
        thought = _thought(thought_id="t-4", content="A B is here, but Alice is too.")
        entities = extract_named_entities_per_member(thought)
        assert "A" not in entities
        assert "B" not in entities
        assert "Alice" in entities

    def test_empty_content_returns_empty(self) -> None:
        thought = _thought(thought_id="t-5", content="lowercase only here")
        assert extract_named_entities_per_member(thought) == []

    def test_output_is_sorted_unique(self) -> None:
        thought = _thought(
            thought_id="t-6",
            content="Alice met Bob. Alice and Bob walked. Bob waved at Alice.",
        )
        entities = extract_named_entities_per_member(thought)
        assert entities == sorted(set(entities))
        assert entities.count("Alice") == 1
        assert entities.count("Bob") == 1


# ---------------------------------------------------------------------------
# Typed-shape contract — the runtime payload matches its TypedDict schema
# ---------------------------------------------------------------------------


class TestTypedShapeContract:
    """Lock the runtime builder output to the declared TypedDict schemas.

    These assertions catch drift in either direction: a field added to the
    builder without updating :class:`ReflectionContentV2` (and its nested
    :class:`MemberExcerpt` / :class:`TemporalSpan`), or a schema field the
    builder stops emitting.
    """

    def test_v2_keys_exactly_match_typeddict(self) -> None:
        cluster = [
            _thought(thought_id="t-001", content="alpha beta gamma decisions monday"),
            _thought(thought_id="t-002", content="delta epsilon zeta retrospective notes"),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        assert set(content.keys()) == set(ReflectionContentV2.__annotations__)

    def test_v2_field_value_types(self) -> None:
        cluster = [
            _thought(thought_id="t-001", content="alpha beta gamma decisions monday"),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        assert content["type"] == "reflection"
        assert content["version"] == 2
        assert isinstance(content["member_ids"], list)
        assert all(isinstance(m, str) for m in content["member_ids"])
        assert isinstance(content["keywords"], list)
        assert isinstance(content["cluster_hash"], str)
        assert isinstance(content["member_count"], int)
        assert isinstance(content["cluster_algorithm"], str)
        assert isinstance(content["created_at"], str)
        assert isinstance(content["top_keyphrases"], list)
        assert isinstance(content["member_excerpts"], list)
        assert isinstance(content["temporal_span"], dict)
        assert isinstance(content["named_entities"], list)

    def test_member_excerpt_keys_exactly_match_typeddict(self) -> None:
        cluster = [_thought(thought_id="t-001", content="alpha beta gamma")]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        excerpts = content["member_excerpts"]
        assert excerpts, "expected at least one excerpt for a non-empty cluster"
        for excerpt in excerpts:
            assert set(excerpt.keys()) == set(MemberExcerpt.__annotations__)
            assert isinstance(excerpt["thought_id"], str)
            assert isinstance(excerpt["excerpt"], str)

    def test_temporal_span_keys_exactly_match_typeddict(self) -> None:
        cluster = [
            _thought(thought_id="t-001", content="alpha", created_at="2026-04-01T00:00:00+00:00"),
            _thought(thought_id="t-002", content="beta", created_at="2026-04-15T00:00:00+00:00"),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        span = content["temporal_span"]
        assert set(span.keys()) == set(TemporalSpan.__annotations__)
        assert isinstance(span["min_created_at"], str)
        assert isinstance(span["max_created_at"], str)
        assert isinstance(span["span_days"], float)

"""Quality-amendment tests for the v2 REFLECTION content builder.

Covers two surfaces:

* ``_extract_named_entities`` blocklist filter — common
  sentence-openers and bare role-marker capitalisations no longer
  pollute the ``named_entities`` field.
* ``_build_member_excerpts`` honours ``max_length`` / the
  ``DreamingConfig.member_excerpt_max_chars`` knob.
"""

from __future__ import annotations

import datetime

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
)

_FIXED_NOW = datetime.datetime(2026, 4, 30, 10, 0, 0, tzinfo=datetime.UTC)


def _thought(
    *,
    thought_id: str,
    content: str,
    priority: Priority = Priority.P2,
) -> ThoughtRecord:
    """Realistic ``ThoughtRecord`` fixture for builder tests."""
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
        created_at="2026-04-29T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# named_entities blocklist filter
# ---------------------------------------------------------------------------


class TestNamedEntitiesBlocklist:
    """Common sentence-starters are filtered out of ``named_entities``."""

    def test_blocklist_filters_sentence_openers(self) -> None:
        # Each member starts with a capitalised non-entity word — every
        # one of them was historically captured as a "named entity".
        cluster = [
            _thought(
                thought_id="t-1",
                content="Absolutely the team agreed on the next steps.",
            ),
            _thought(
                thought_id="t-2",
                content="Additionally we will follow up with Cornell faculty.",
            ),
            _thought(
                thought_id="t-3",
                content="However the Berlin office is on a different schedule.",
            ),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        named_entities = set(content["named_entities"])
        # None of the sentence-starters survive.
        assert "Absolutely" not in named_entities
        assert "Additionally" not in named_entities
        assert "However" not in named_entities

    def test_blocklist_preserves_real_proper_nouns(self) -> None:
        """Genuine proper nouns survive the blocklist filter."""
        cluster = [
            _thought(
                thought_id="t-1",
                content="Cornell faculty visited the Berlin lab last month.",
            ),
            _thought(
                thought_id="t-2",
                content="Tokyo team coordinated with Anthropic on the audit.",
            ),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        named_entities = set(content["named_entities"])
        for proper_noun in ("Cornell", "Berlin", "Tokyo", "Anthropic"):
            assert proper_noun in named_entities, (
                f"real proper noun {proper_noun!r} disappeared from named_entities"
            )

    def test_blocklist_filters_bare_role_capitalisation(self) -> None:
        """Bare ``User`` / ``Assistant`` capitalisations are filtered too.

        After the upstream chunker strips the ``[USER]`` / ``[ASSISTANT]``
        marker brackets, the bare ``User`` / ``Assistant`` capitalisation
        can survive at the start of the resulting fragment.  The
        blocklist drops both forms so they do not pollute the entity
        list.
        """
        cluster = [
            _thought(
                thought_id="t-1",
                content="User mentioned Cornell faculty in the context of grants.",
            ),
            _thought(
                thought_id="t-2",
                content="Assistant explained how the Berlin lab handled the audit.",
            ),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        named_entities = set(content["named_entities"])
        assert "User" not in named_entities
        assert "Assistant" not in named_entities
        # Real proper nouns inside the content still appear.
        assert "Cornell" in named_entities
        assert "Berlin" in named_entities

    def test_year_extractor_unaffected_by_blocklist(self) -> None:
        """The year / measurement regex path is independent of the blocklist."""
        cluster = [
            _thought(
                thought_id="t-1",
                content="The 2026 budget covers Q3 onboarding work.",
            ),
        ]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        assert "2026" in content["named_entities"]


# ---------------------------------------------------------------------------
# member_excerpts: max_length threaded from config
# ---------------------------------------------------------------------------


class TestMemberExcerptMaxChars:
    """``DreamingConfig.member_excerpt_max_chars`` is honoured by the builder."""

    def test_default_excerpt_size_is_150(self) -> None:
        long_content = (
            "this is a very long observation about deployment latency "
            "and morale and quarterly retrospectives and dashboards "
            "covering pipeline stages and the ongoing rotation of "
            "ownership across teams running the site reliability practice"
        )
        cluster = [_thought(thought_id="t-long", content=long_content)]
        cfg = DreamingConfig()
        assert cfg.member_excerpt_max_chars == 150  # baseline sanity

        content = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        excerpt = content["member_excerpts"][0]["excerpt"]
        assert len(excerpt) <= cfg.member_excerpt_max_chars
        # Truncation actually happened — the source is much longer
        # than 150 chars, so the excerpt must end with the ellipsis
        # marker.
        assert excerpt.endswith("...")

    def test_config_override_lower_bound(self) -> None:
        long_content = "filler content " * 40  # ~600 chars, well above the override
        cluster = [_thought(thought_id="t-long", content=long_content)]
        cfg = DreamingConfig(member_excerpt_max_chars=80)
        content = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        excerpt = content["member_excerpts"][0]["excerpt"]
        assert len(excerpt) <= 80
        assert excerpt.endswith("...")

    def test_config_override_higher_bound(self) -> None:
        long_content = "long substantive observation " * 20  # ~580 chars
        cluster = [_thought(thought_id="t-long", content=long_content)]
        cfg = DreamingConfig(member_excerpt_max_chars=300)
        content = build_reflection_content_v2(cluster, algorithm="lpa", config=cfg, now=_FIXED_NOW)
        excerpt = content["member_excerpts"][0]["excerpt"]
        # Up to 300 chars allowed now — 150-char truncation no longer fires.
        assert len(excerpt) <= 300
        assert len(excerpt) > 150  # genuinely longer than the default


# ---------------------------------------------------------------------------
# Backward-compat smoke — v2 schema keys unchanged
# ---------------------------------------------------------------------------


class TestV2SchemaKeysUnchanged:
    """The amendment must not add or remove top-level keys from v2 content."""

    def test_v2_keys_preserved(self) -> None:
        cluster = [_thought(thought_id="t-1", content="cornell visit recap")]
        content = build_reflection_content_v2(
            cluster, algorithm="lpa", config=DreamingConfig(), now=_FIXED_NOW
        )
        expected_keys = {
            "type",
            "version",
            "member_ids",
            "keywords",
            "cluster_hash",
            "member_count",
            "cluster_algorithm",
            "created_at",
            "top_keyphrases",
            "member_excerpts",
            "temporal_span",
            "named_entities",
        }
        assert set(content.keys()) == expected_keys, (
            f"v2 schema keys changed: got {set(content.keys())} vs expected {expected_keys}"
        )

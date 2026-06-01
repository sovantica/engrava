"""Unit tests for the cluster quality gate helpers.

Each gate is a pure deterministic function operating on resolved cluster
members (``list[ThoughtRecord]`` plus, for the cohesion gate, member
embedding vectors).  These tests pin down happy paths, edge cases
(empty / single-member clusters), threshold boundaries, and defensive
handling of malformed input.
"""

from __future__ import annotations

import pytest

from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import MetadataValue, ThoughtRecord
from engrava.extensions.dreaming_cluster_quality import (
    cluster_cohesion_score,
    has_consistent_entities,
    has_contradictory_members,
    has_duplicate_content_members,
    has_meaningful_keyphrases,
    is_external_source_homogeneous,
    is_low_cohesion,
    is_persona_only_cluster,
)


def _thought(
    thought_id: str,
    content: str,
    *,
    metadata: dict[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Build a minimal OBSERVATION ThoughtRecord for gate tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=content[:60] or "essence",
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        metadata=metadata or {},
    )


def _kp(phrase: object, score: float) -> dict[str, float | str]:
    """Build a keyphrase dict matching the v2 content builder shape."""
    return {"phrase": phrase, "score": score}  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Gate 1 — duplicate-member content
# ---------------------------------------------------------------------------


class TestHasDuplicateContentMembers:
    def test_byte_identical_pair_rejected(self) -> None:
        cluster = [_thought("a", "same text"), _thought("b", "same text")]
        is_dup, count = has_duplicate_content_members(cluster)
        assert is_dup is True
        assert count == 2

    def test_three_identical_members_counted(self) -> None:
        cluster = [
            _thought("a", "x"),
            _thought("b", "x"),
            _thought("c", "x"),
        ]
        is_dup, count = has_duplicate_content_members(cluster)
        assert is_dup is True
        assert count == 3

    def test_no_duplicates_pass(self) -> None:
        cluster = [_thought("a", "alpha"), _thought("b", "beta")]
        is_dup, count = has_duplicate_content_members(cluster)
        assert is_dup is False
        assert count == 0

    def test_single_member_pass(self) -> None:
        is_dup, count = has_duplicate_content_members([_thought("a", "lonely")])
        assert is_dup is False
        assert count == 0

    def test_empty_cluster_pass(self) -> None:
        is_dup, count = has_duplicate_content_members([])
        assert is_dup is False
        assert count == 0

    def test_whitespace_difference_is_not_a_duplicate(self) -> None:
        # The gate detects byte-identical content; a trailing space makes
        # the two members distinct (this is the very dedup-escape the gate
        # exists to catch when content IS identical — here it is not).
        cluster = [_thought("a", "text"), _thought("b", "text ")]
        is_dup, count = has_duplicate_content_members(cluster)
        assert is_dup is False
        assert count == 0


# ---------------------------------------------------------------------------
# Gate 2 — persona-only cluster
# ---------------------------------------------------------------------------


class TestIsPersonaOnlyCluster:
    def test_all_persona_descriptions_rejected(self) -> None:
        cluster = [
            _thought("a", "Alex Martinez is a graphic design student."),
            _thought("b", "Alex was born in 1974 and is a designer."),
            _thought("c", "Embracing their creative side, Alex is an artist."),
        ]
        is_persona, ratio = is_persona_only_cluster(cluster)
        assert is_persona is True
        assert ratio == pytest.approx(1.0)

    def test_conversational_members_not_persona(self) -> None:
        cluster = [
            _thought("a", "[USER] I love painting on weekends"),
            _thought("b", "[ASSISTANT] That sounds wonderful"),
        ]
        is_persona, ratio = is_persona_only_cluster(cluster)
        assert is_persona is False
        assert ratio == pytest.approx(0.0)

    def test_mixed_persona_and_conversation_below_threshold(self) -> None:
        cluster = [
            _thought("a", "Alex is a designer."),  # persona
            _thought("b", "[USER] What do you think of my logo?"),  # conversational
            _thought("c", "[USER] I prefer minimalist design"),  # conversational
        ]
        # 1/3 persona < 0.75 threshold
        is_persona, ratio = is_persona_only_cluster(cluster)
        assert is_persona is False
        assert ratio == pytest.approx(1 / 3)

    def test_persona_indicator_with_conversation_marker_not_counted(self) -> None:
        # A conversational turn that happens to contain "is a" must not be
        # counted as persona — the conversation marker disqualifies it.
        cluster = [
            _thought("a", "[USER] My brother is a doctor"),
            _thought("b", "[USER] He works long hours"),
        ]
        is_persona, ratio = is_persona_only_cluster(cluster)
        assert is_persona is False
        assert ratio == pytest.approx(0.0)

    def test_threshold_boundary_exactly_75_percent(self) -> None:
        cluster = [
            _thought("a", "Alice is a teacher."),  # persona
            _thought("b", "Bob is a nurse."),  # persona
            _thought("c", "Carol is an artist."),  # persona
            _thought("d", "[USER] just a normal turn"),  # conversational
        ]
        # 3/4 = 0.75 == threshold → flagged (>=)
        is_persona, ratio = is_persona_only_cluster(cluster)
        assert is_persona is True
        assert ratio == pytest.approx(0.75)

    def test_custom_threshold_respected(self) -> None:
        cluster = [
            _thought("a", "Alice is a teacher."),  # persona
            _thought("b", "[USER] hello there"),  # conversational
        ]
        # 1/2 = 0.5 — passes default 0.75, fails a 0.4 threshold
        assert is_persona_only_cluster(cluster)[0] is False
        assert is_persona_only_cluster(cluster, persona_threshold=0.4)[0] is True

    def test_empty_cluster_pass(self) -> None:
        is_persona, ratio = is_persona_only_cluster([])
        assert is_persona is False
        assert ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gate 3 — contradictory members
# ---------------------------------------------------------------------------


class TestHasContradictoryMembers:
    def test_stopped_vs_created_rejected(self) -> None:
        cluster = [
            _thought("a", "I stopped making study videos last year"),
            _thought("b", "I created a study video, got positive feedback"),
        ]
        is_contra, reasons = has_contradictory_members(cluster)
        assert is_contra is True
        assert reasons
        assert any("stopped" in r and "created" in r for r in reasons)

    def test_love_vs_hate_rejected(self) -> None:
        cluster = [
            _thought("a", "I hate running in the cold"),
            _thought("b", "I love running marathons"),
        ]
        is_contra, reasons = has_contradictory_members(cluster)
        assert is_contra is True
        assert any("hate" in r and "love" in r for r in reasons)

    def test_no_contradiction_pass(self) -> None:
        cluster = [
            _thought("a", "I enjoy hiking in the mountains"),
            _thought("b", "I started a new pottery class"),
        ]
        is_contra, reasons = has_contradictory_members(cluster)
        assert is_contra is False
        assert reasons == []

    def test_contradiction_within_single_member_still_flags(self) -> None:
        # Conservative behaviour: a single member containing both sides
        # ("couldn't ... managed") trips the gate.
        cluster = [
            _thought("a", "At first I couldn't finish it, but later I managed"),
        ]
        is_contra, _reasons = has_contradictory_members(cluster)
        assert is_contra is True

    def test_empty_cluster_pass(self) -> None:
        is_contra, reasons = has_contradictory_members([])
        assert is_contra is False
        assert reasons == []

    def test_contraction_token_matched(self) -> None:
        cluster = [
            _thought("a", "I can't swim"),
            _thought("b", "I can ride a bike"),
        ]
        is_contra, _reasons = has_contradictory_members(cluster)
        assert is_contra is True


# ---------------------------------------------------------------------------
# Gate 4 — low cohesion
# ---------------------------------------------------------------------------


class TestClusterCohesion:
    def test_identical_unit_vectors_full_cohesion(self) -> None:
        vec = [1.0, 0.0, 0.0]
        assert cluster_cohesion_score([vec, vec, vec]) == pytest.approx(1.0)

    def test_orthogonal_vectors_zero_cohesion(self) -> None:
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        assert cluster_cohesion_score(vectors) == pytest.approx(0.0)

    def test_single_vector_trivially_cohesive(self) -> None:
        assert cluster_cohesion_score([[1.0, 0.0]]) == pytest.approx(1.0)

    def test_empty_trivially_cohesive(self) -> None:
        assert cluster_cohesion_score([]) == pytest.approx(1.0)

    def test_three_member_mean_pairwise(self) -> None:
        # Two members identical (cos 1.0), third orthogonal (cos 0.0 to
        # each).  Pairs: (a,b)=1.0, (a,c)=0.0, (b,c)=0.0 → mean = 1/3.
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        assert cluster_cohesion_score(vectors) == pytest.approx(1 / 3)


class TestIsLowCohesion:
    def test_high_cohesion_pass(self) -> None:
        vec = [1.0, 0.0, 0.0]
        is_loose, score = is_low_cohesion([vec, vec])
        assert is_loose is False
        assert score == pytest.approx(1.0)

    def test_low_cohesion_rejected(self) -> None:
        vectors = [[1.0, 0.0], [0.0, 1.0]]  # cosine 0.0
        is_loose, score = is_low_cohesion(vectors)
        assert is_loose is True
        assert score == pytest.approx(0.0)

    def test_threshold_boundary_not_strict_below(self) -> None:
        # Construct a cluster whose mean pairwise cosine is exactly 0.40.
        # Two members: cos(a, b) = 0.40 → mean = 0.40, which is NOT
        # strictly below the 0.40 threshold → passes.
        import math

        theta = math.acos(0.40)
        vectors = [
            [1.0, 0.0],
            [math.cos(theta), math.sin(theta)],
        ]
        is_loose, score = is_low_cohesion(vectors, cohesion_threshold=0.40)
        assert score == pytest.approx(0.40)
        assert is_loose is False

    def test_custom_threshold_respected(self) -> None:
        vectors = [[1.0, 0.0], [0.0, 1.0]]  # cosine 0.0
        # Passes a threshold of 0.0 (not strictly below), fails 0.1.
        assert is_low_cohesion(vectors, cohesion_threshold=0.0)[0] is False
        assert is_low_cohesion(vectors, cohesion_threshold=0.1)[0] is True

    def test_single_member_passes(self) -> None:
        is_loose, score = is_low_cohesion([[1.0, 0.0]])
        assert is_loose is False
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gate 5 — external-source homogeneity
# ---------------------------------------------------------------------------


class TestIsExternalSourceHomogeneous:
    def test_all_external_pass(self) -> None:
        cluster = [
            _thought("a", "x", metadata={"source": {"is_self": False}}),
            _thought("b", "y", metadata={"source": {"is_self": False}}),
        ]
        is_homog, frac = is_external_source_homogeneous(cluster)
        assert is_homog is True
        assert frac == pytest.approx(1.0)

    def test_one_self_authored_below_threshold_rejected(self) -> None:
        # 1 self / 3 total → external fraction 2/3 ≈ 0.667 < 0.95.
        cluster = [
            _thought("a", "x", metadata={"source": {"is_self": False}}),
            _thought("b", "y", metadata={"source": {"is_self": False}}),
            _thought("c", "z", metadata={"source": {"is_self": True}}),
        ]
        is_homog, frac = is_external_source_homogeneous(cluster)
        assert is_homog is False
        assert frac == pytest.approx(2 / 3)

    def test_missing_source_treated_as_external(self) -> None:
        # Safe-fallback: legacy data without metadata.source passes.
        cluster = [_thought("a", "x"), _thought("b", "y")]
        is_homog, frac = is_external_source_homogeneous(cluster)
        assert is_homog is True
        assert frac == pytest.approx(1.0)

    def test_missing_is_self_treated_as_external(self) -> None:
        cluster = [
            _thought("a", "x", metadata={"source": {"confidence": "low"}}),
            _thought("b", "y", metadata={"source": {"is_self": False}}),
        ]
        is_homog, frac = is_external_source_homogeneous(cluster)
        assert is_homog is True
        assert frac == pytest.approx(1.0)

    def test_malformed_source_treated_as_external(self) -> None:
        cluster = [
            _thought("a", "x", metadata={"source": "not-a-dict"}),
            _thought("b", "y", metadata={"source": {"is_self": False}}),
        ]
        is_homog, frac = is_external_source_homogeneous(cluster)
        assert is_homog is True
        assert frac == pytest.approx(1.0)

    def test_all_self_authored_rejected(self) -> None:
        cluster = [
            _thought("a", "x", metadata={"source": {"is_self": True}}),
            _thought("b", "y", metadata={"source": {"is_self": True}}),
        ]
        is_homog, frac = is_external_source_homogeneous(cluster)
        assert is_homog is False
        assert frac == pytest.approx(0.0)

    def test_empty_cluster_vacuously_homogeneous(self) -> None:
        is_homog, frac = is_external_source_homogeneous([])
        assert is_homog is True
        assert frac == pytest.approx(1.0)

    def test_custom_threshold_respected(self) -> None:
        # 1 self / 2 total → external fraction 0.5.
        cluster = [
            _thought("a", "x", metadata={"source": {"is_self": False}}),
            _thought("b", "y", metadata={"source": {"is_self": True}}),
        ]
        assert is_external_source_homogeneous(cluster)[0] is False
        assert is_external_source_homogeneous(cluster, min_external_fraction=0.5)[0] is True


# ---------------------------------------------------------------------------
# Gate 6 — named-entity consistency
# ---------------------------------------------------------------------------


class TestHasConsistentEntities:
    def test_shared_entities_pass(self) -> None:
        cluster = [
            _thought("a", "Alice visited Paris in summer"),
            _thought("b", "Alice loved the Louvre in Paris"),
        ]
        is_consistent, ratio = has_consistent_entities(cluster)
        assert is_consistent is True
        assert ratio == pytest.approx(1.0)

    def test_disjoint_entities_rejected(self) -> None:
        cluster = [
            _thought("a", "Alice does pottery"),
            _thought("b", "Bob does painting"),
            _thought("c", "Carol does sculpture"),
        ]
        is_consistent, ratio = has_consistent_entities(cluster)
        assert is_consistent is False
        # Only the first member "scores" itself; nobody else overlaps
        # with Alice -> ratio = 1/3.
        assert ratio == pytest.approx(1 / 3)

    def test_single_member_vacuously_consistent(self) -> None:
        is_consistent, ratio = has_consistent_entities(
            [_thought("a", "Alice visited Paris")],
        )
        assert is_consistent is True
        assert ratio == pytest.approx(1.0)

    def test_empty_cluster_vacuously_consistent(self) -> None:
        is_consistent, ratio = has_consistent_entities([])
        assert is_consistent is True
        assert ratio == pytest.approx(1.0)

    def test_first_member_has_no_entities_passes(self) -> None:
        # Lowercase only — no capitalised tokens, no year/measurement.
        cluster = [
            _thought("a", "lowercase prose with no proper nouns"),
            _thought("b", "Alice writes more lowercase prose"),
        ]
        is_consistent, ratio = has_consistent_entities(cluster)
        assert is_consistent is True
        assert ratio == pytest.approx(1.0)

    def test_partial_overlap_below_threshold(self) -> None:
        # First member entities {Alice, Paris}; only 2/5 members overlap
        # with Alice -> ratio = 3/5 = 0.60.  With min_shared_ratio=0.75
        # the cluster is rejected.
        cluster = [
            _thought("a", "Alice visited Paris last summer"),
            _thought("b", "Alice loved the museum"),
            _thought("c", "Alice took the train home"),
            _thought("d", "Bob was elsewhere entirely"),
            _thought("e", "Carol stayed in town"),
        ]
        # Default 0.60 -> exactly at threshold (>=) -> passes.
        is_consistent, ratio = has_consistent_entities(cluster)
        assert ratio == pytest.approx(0.60)
        assert is_consistent is True
        # Tighter 0.75 -> rejected.
        is_strict_consistent, _ = has_consistent_entities(
            cluster,
            min_shared_ratio=0.75,
        )
        assert is_strict_consistent is False

    def test_year_token_counts_as_entity(self) -> None:
        # Year tokens flow through the per-member extractor, so two
        # members sharing "1974" overlap even without a proper noun in
        # common.
        cluster = [
            _thought("a", "Born in 1974, started sculpture early"),
            _thought("b", "The 1974 cohort produced many artists"),
        ]
        is_consistent, ratio = has_consistent_entities(cluster)
        assert is_consistent is True
        assert ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gate 8 — meaningful keyphrases
# ---------------------------------------------------------------------------


class TestHasMeaningfulKeyphrases:
    def test_at_least_one_specific_phrase_passes(self) -> None:
        keyphrases = [
            _kp("these moments", 0.5),
            _kp("piano practice", 0.4),
        ]
        assert has_meaningful_keyphrases(keyphrases) is True

    def test_all_generic_rejected(self) -> None:
        keyphrases = [
            _kp("these moments", 0.5),
            _kp("specific projects", 0.4),
            _kp("their preferences", 0.3),
        ]
        assert has_meaningful_keyphrases(keyphrases) is False

    def test_empty_keyphrases_rejected(self) -> None:
        assert has_meaningful_keyphrases([]) is False

    def test_generic_leader_case_insensitive(self) -> None:
        assert has_meaningful_keyphrases([_kp("These Moments", 0.5)]) is False
        assert has_meaningful_keyphrases([_kp("THESE moments", 0.5)]) is False

    def test_generic_leader_alone_is_not_generic(self) -> None:
        # The pattern requires a generic leader FOLLOWED BY another word.
        # A single bare word "these" is not matched → counts as meaningful
        # (degenerate, but the gate is about determiner-noun phrases).
        assert has_meaningful_keyphrases([_kp("these", 0.5)]) is True

    def test_non_string_phrase_treated_as_generic(self) -> None:
        assert has_meaningful_keyphrases([_kp(42, 0.5)]) is False
        # ...but a meaningful string alongside it still passes.
        assert has_meaningful_keyphrases([_kp(42, 0.5), _kp("violin lesson", 0.4)]) is True

    def test_leading_whitespace_tolerated(self) -> None:
        assert has_meaningful_keyphrases([_kp("  these moments", 0.5)]) is False

"""Unit tests for the cross-cluster boilerplate filter helpers.

Covers :func:`compute_cluster_phrase_frequency` and
:func:`is_boilerplate_phrase` from
``engrava.extensions.dreaming_keyphrases``.  The helpers are
deterministic and operate on plain Python dicts, so the tests stay
purely synchronous and do not touch SQLite.

Multi-language coverage is included because the filter is statistical
and language-agnostic by design — any phrase that floods the corpus,
regardless of language, becomes a boilerplate candidate.
"""

from __future__ import annotations

import pytest

from engrava.domain.models import ThoughtRecord
from engrava.extensions.dreaming_keyphrases import (
    compute_cluster_phrase_frequency,
    is_boilerplate_phrase,
)


def _kp(phrase: object, score: float) -> dict[str, float | str]:
    """Build a keyphrase dict matching ``top_keyphrases_tfidf`` shape.

    The helper accepts an ``object`` ``phrase`` argument so the defensive
    "non-string phrase silently skipped" test case can pass in ``int``
    or ``None`` without tripping mypy on every callsite.
    """
    return {"phrase": phrase, "score": score}  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# compute_cluster_phrase_frequency
# ---------------------------------------------------------------------------


class TestComputeClusterPhraseFrequency:
    """Document-frequency tallying across clusters."""

    def test_counts_unique_clusters_per_phrase(self) -> None:
        clusters = [
            [_kp("wonderful", 0.5), _kp("piano", 0.4)],
            [_kp("wonderful", 0.6), _kp("running", 0.4)],
            [_kp("wonderful", 0.5), _kp("cooking", 0.4)],
        ]
        result = compute_cluster_phrase_frequency(clusters)
        assert result == {
            "wonderful": 3,
            "piano": 1,
            "running": 1,
            "cooking": 1,
        }

    def test_duplicate_phrase_in_same_cluster_counts_once(self) -> None:
        """A cluster contributes at most ``1`` per phrase regardless of repeats."""
        clusters = [
            [
                _kp("wonderful", 0.5),
                _kp("Wonderful", 0.4),
                _kp("WONDERFUL", 0.3),
            ],
            [_kp("wonderful", 0.5)],
        ]
        result = compute_cluster_phrase_frequency(clusters)
        assert result["wonderful"] == 2

    def test_empty_cluster_list_returns_empty_mapping(self) -> None:
        assert compute_cluster_phrase_frequency([]) == {}

    def test_empty_inner_clusters_handled(self) -> None:
        assert compute_cluster_phrase_frequency([[], [], []]) == {}

    def test_phrase_normalised_lowercase(self) -> None:
        clusters = [
            [_kp("Wonderful", 0.5)],
            [_kp("WONDERFUL", 0.5)],
            [_kp("wonderful", 0.5)],
        ]
        result = compute_cluster_phrase_frequency(clusters)
        assert result == {"wonderful": 3}

    def test_non_string_phrase_silently_skipped(self) -> None:
        """Defensive: malformed entries with non-string phrases do not crash."""
        clusters = [
            [_kp(42, 0.5), _kp("valid", 0.4)],
            [_kp(None, 0.5), _kp("valid", 0.4)],
        ]
        result = compute_cluster_phrase_frequency(clusters)
        assert result == {"valid": 2}


# ---------------------------------------------------------------------------
# is_boilerplate_phrase
# ---------------------------------------------------------------------------


class TestIsBoilerplatePhrase:
    """Boilerplate threshold semantics."""

    def test_above_threshold_flagged(self) -> None:
        df = {"wonderful": 5}
        assert is_boilerplate_phrase(
            "wonderful",
            df,
            total_clusters=10,
            threshold=0.30,
            min_corpus_size=5,
        )

    def test_at_threshold_not_flagged(self) -> None:
        """Strictly greater than — equality is not boilerplate."""
        df = {"wonderful": 3}
        # 3 / 10 = 0.30, threshold = 0.30 -> not strictly greater -> False
        assert not is_boilerplate_phrase(
            "wonderful",
            df,
            total_clusters=10,
            threshold=0.30,
            min_corpus_size=5,
        )

    def test_below_threshold_not_flagged(self) -> None:
        df = {"piano": 2}
        assert not is_boilerplate_phrase(
            "piano",
            df,
            total_clusters=10,
            threshold=0.30,
            min_corpus_size=5,
        )

    def test_missing_phrase_not_flagged(self) -> None:
        """A phrase absent from the document-frequency mapping returns ``False``."""
        assert not is_boilerplate_phrase(
            "phantom",
            {"other": 99},
            total_clusters=10,
            threshold=0.30,
            min_corpus_size=5,
        )

    def test_case_insensitive_lookup(self) -> None:
        df = {"wonderful": 5}
        for variant in ("Wonderful", "WONDERFUL", "wONDERFUL"):
            assert is_boilerplate_phrase(
                variant,
                df,
                total_clusters=10,
                threshold=0.30,
                min_corpus_size=5,
            ), variant

    def test_small_corpus_bypasses_filter(self) -> None:
        """``total_clusters < min_corpus_size`` -> filter inactive."""
        df = {"wonderful": 5}
        assert not is_boilerplate_phrase(
            "wonderful",
            df,
            total_clusters=4,
            threshold=0.30,
            min_corpus_size=5,
        )

    def test_min_corpus_size_boundary(self) -> None:
        """At exactly ``min_corpus_size`` the filter engages."""
        df = {"wonderful": 5}
        assert is_boilerplate_phrase(
            "wonderful",
            df,
            total_clusters=5,
            threshold=0.30,
            min_corpus_size=5,
        )

    def test_threshold_one_disables_filter(self) -> None:
        """``threshold=1.0`` is impossible to exceed -> filter disabled."""
        df = {"wonderful": 10}
        assert not is_boilerplate_phrase(
            "wonderful",
            df,
            total_clusters=10,
            threshold=1.0,
            min_corpus_size=5,
        )


# ---------------------------------------------------------------------------
# Multi-language scenarios
# ---------------------------------------------------------------------------


class TestMultiLanguageBoilerplateDetection:
    """The filter is statistical, not lexical — language must not matter."""

    @pytest.mark.parametrize(
        ("flooded_phrase", "control_phrase"),
        [
            ("to wspaniale", "fotografia"),  # Polish (ASCII-only for repo)
            ("subarashii desu", "piano"),  # Japanese romaji
            ("c'est incroyable", "saxophone"),  # French
            ("das ist wunderbar", "klavier"),  # German
        ],
    )
    def test_floods_detected_regardless_of_language(
        self,
        flooded_phrase: str,
        control_phrase: str,
    ) -> None:
        clusters = [
            [_kp(flooded_phrase, 0.5)],
            [_kp(flooded_phrase, 0.5)],
            [_kp(flooded_phrase, 0.5)],
            [_kp(flooded_phrase, 0.5)],
            [_kp(flooded_phrase, 0.5)],
            [_kp(control_phrase, 0.6)],
        ]
        df = compute_cluster_phrase_frequency(clusters)
        assert is_boilerplate_phrase(
            flooded_phrase,
            df,
            total_clusters=len(clusters),
            threshold=0.30,
            min_corpus_size=5,
        )
        assert not is_boilerplate_phrase(
            control_phrase,
            df,
            total_clusters=len(clusters),
            threshold=0.30,
            min_corpus_size=5,
        )


# ---------------------------------------------------------------------------
# build_reflection_content_v2 integration
# ---------------------------------------------------------------------------


class TestBuildReflectionContentV2BoilerplateFilter:
    """End-to-end: the v2 content builder drops boilerplate when given the DF."""

    @staticmethod
    def _make_member(thought_id: str, content: str) -> object:
        """Build a ThoughtRecord member with the minimum fields the builder reads."""
        from engrava.domain.enums import (
            LifecycleStatus,
            Priority,
            ThoughtType,
        )
        from engrava.domain.models import ThoughtRecord

        return ThoughtRecord(
            thought_id=thought_id,
            thought_type=ThoughtType.OBSERVATION,
            essence=content[:60],
            content=content,
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="test",
        )

    def test_backward_compat_when_filter_args_omitted(self) -> None:
        """No ``cluster_phrase_df`` / ``total_clusters`` -> raw keyphrases preserved."""
        import datetime

        from engrava.config import DreamingConfig
        from engrava.extensions.dreaming_reflection_content import (
            build_reflection_content_v2,
        )

        cluster = [
            self._make_member("t-1", "wonderful piano practice today"),
            self._make_member("t-2", "wonderful piano session this morning"),
        ]
        result = build_reflection_content_v2(
            cluster,  # type: ignore[arg-type]
            algorithm="agglomerative",
            config=DreamingConfig(),
            corpus=["wonderful piano practice", "wonderful piano session"],
            now=datetime.datetime(2026, 5, 12, tzinfo=datetime.UTC),
        )
        assert result["top_keyphrases"], "without filter args the raw TF-IDF list must come through"

    def test_filter_drops_corpus_wide_boilerplate(self) -> None:
        """Phrase present in too many clusters is removed from top_keyphrases."""
        import datetime

        from engrava.config import DreamingConfig
        from engrava.extensions.dreaming_reflection_content import (
            build_reflection_content_v2,
        )

        cluster = [
            self._make_member("t-1", "wonderful piano practice"),
            self._make_member("t-2", "wonderful piano session"),
        ]
        # Synthetic DF: "wonderful piano" appears in 8/10 clusters -> boilerplate.
        cluster_phrase_df = {"wonderful piano": 8, "piano practice": 1}
        result = build_reflection_content_v2(
            cluster,  # type: ignore[arg-type]
            algorithm="agglomerative",
            config=DreamingConfig(boilerplate_min_keyphrases_per_refl=0),
            corpus=["wonderful piano practice", "wonderful piano session"],
            cluster_phrase_df=cluster_phrase_df,
            total_clusters=10,
            now=datetime.datetime(2026, 5, 12, tzinfo=datetime.UTC),
        )
        surviving = {kp["phrase"] for kp in result["top_keyphrases"]}
        assert "wonderful piano" not in surviving

    def test_fallback_when_filter_strips_below_minimum(self) -> None:
        """Filter that would empty the list is bypassed (raw kept)."""
        import datetime

        from engrava.config import DreamingConfig
        from engrava.extensions.dreaming_reflection_content import (
            build_reflection_content_v2,
        )

        cluster = [
            self._make_member("t-1", "alpha beta gamma delta epsilon"),
        ]
        # Every phrase in the cluster is "boilerplate" per the synthetic DF.
        cluster_phrase_df = {
            "alpha beta": 9,
            "beta gamma": 9,
            "gamma delta": 9,
            "delta epsilon": 9,
            "alpha beta gamma": 9,
            "beta gamma delta": 9,
            "gamma delta epsilon": 9,
        }
        result = build_reflection_content_v2(
            cluster,  # type: ignore[arg-type]
            algorithm="agglomerative",
            config=DreamingConfig(boilerplate_min_keyphrases_per_refl=1),
            corpus=["alpha beta gamma delta epsilon"],
            cluster_phrase_df=cluster_phrase_df,
            total_clusters=10,
            now=datetime.datetime(2026, 5, 12, tzinfo=datetime.UTC),
        )
        # Fallback engaged -> top_keyphrases is the raw list, NOT empty.
        assert result["top_keyphrases"], "fallback must preserve at least one keyphrase"

    def test_filter_disabled_when_only_one_kwarg_supplied(self) -> None:
        """Supplying ``cluster_phrase_df`` without ``total_clusters`` keeps raw list.

        Asserts on the raw-keyphrase set produced without the filter so
        the test is independent of any particular TF-IDF tie-break.
        """
        import datetime

        from engrava.config import DreamingConfig
        from engrava.extensions.dreaming_reflection_content import (
            build_reflection_content_v2,
        )

        cluster = [
            self._make_member("t-1", "wonderful piano practice"),
            self._make_member("t-2", "wonderful piano session"),
        ]
        config = DreamingConfig()
        corpus = ["wonderful piano practice"]
        now = datetime.datetime(2026, 5, 12, tzinfo=datetime.UTC)

        baseline = build_reflection_content_v2(
            cluster,  # type: ignore[arg-type]
            algorithm="agglomerative",
            config=config,
            corpus=corpus,
            now=now,
        )
        with_partial_kwargs = build_reflection_content_v2(
            cluster,  # type: ignore[arg-type]
            algorithm="agglomerative",
            config=config,
            corpus=corpus,
            cluster_phrase_df={"wonderful piano": 8},
            # total_clusters intentionally omitted -> filter disabled
            now=now,
        )
        assert with_partial_kwargs["top_keyphrases"] == baseline["top_keyphrases"]


# ---------------------------------------------------------------------------
# End-to-end: dreaming consolidation pre-pass + persisted REFLECTION content
# ---------------------------------------------------------------------------


class TestDreamingPrepassFiltersPersistedKeyphrases:
    """Real ``DreamingExtension.run_consolidation`` strips boilerplate.

    The unit tests above exercise the builder with a synthetic
    ``cluster_phrase_df``.  This case wires up real clusters, lets the
    consolidation pipeline run its pre-pass + main loop end-to-end, and
    asserts on the JSON ``top_keyphrases`` payload persisted to the
    REFLECTION thought — closing the gap between the builder unit tests
    and the orchestration that ships in production.
    """

    @staticmethod
    async def _seed_cluster(
        store: object,
        members: list[ThoughtRecord],
        *,
        embedding_dim: int = 8,
    ) -> None:
        unit_vector = [0.0] * embedding_dim
        unit_vector[0] = 1.0
        for thought in members:
            await store.create_thought(thought)  # type: ignore[attr-defined]
            await store.store_embedding(  # type: ignore[attr-defined]
                thought.thought_id,
                unit_vector,
                model_name="test-embed",
            )

    @staticmethod
    def _make_member(thought_id: str, content: str) -> ThoughtRecord:
        from engrava.domain.enums import (
            LifecycleStatus,
            Priority,
            ThoughtType,
        )
        from engrava.domain.models import ThoughtRecord as _ThoughtRecord

        return _ThoughtRecord(
            thought_id=thought_id,
            thought_type=ThoughtType.OBSERVATION,
            essence=content[:60],
            content=content,
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="test",
        )

    async def test_prepass_strips_corpus_wide_phrase_from_persisted_reflection(
        self,
    ) -> None:
        """End-to-end: a phrase flooding every cluster vanishes from
        the persisted ``REFLECTION.content.top_keyphrases``.

        Compares two consolidation runs on the same corpus:

        1. Baseline run with ``boilerplate_threshold=1.0`` (filter
           disabled) — the flooding trigram ``"flooded marker shared"``
           reaches every cluster's ``top_keyphrases`` because every
           member content embeds it verbatim alongside a per-cluster
           distinct token.
        2. Active run with the default ``boilerplate_threshold=0.30``
           and ``boilerplate_min_corpus_size=5`` — the pre-pass sees
           the trigram (and the two component bigrams) in every
           cluster's raw keyphrase list, so the in-builder filter
           drops them from every persisted REFLECTION.

        The two-run shape pins down the contract:
        the flooded phrase IS reachable in ``top_keyphrases`` without
        the filter (so the assertion would catch a regression that
        silently broke the wiring) AND is dropped once the filter
        engages.
        """
        import json

        import aiosqlite

        from engrava.config import DreamingConfig, DreamingGates
        from engrava.domain.enums import ThoughtType
        from engrava.extensions.dreaming import DreamingExtension
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        # Eight clusters of two members each.  The shared trigram
        # ``"flooded marker shared"`` is identical across all
        # clusters; each cluster has one extra distinguishing token
        # so the per-cluster TF-IDF picks up the shared trigram plus
        # cluster-specific phrases.
        flood = "flooded marker shared"
        distinct_terms = [
            "violin",
            "marathon",
            "ceramic",
            "barista",
            "chess",
            "watercolour",
            "kayaking",
            "beekeeping",
        ]

        def _seed_corpus(store: SqliteEngravaCore) -> list[None]:
            return []  # placeholder for type; loop below does the seeding

        async def _seed(store: SqliteEngravaCore) -> None:
            for cluster_idx, distinct in enumerate(distinct_terms):
                axis_vector = [0.0] * 8
                axis_vector[cluster_idx] = 1.0
                for member_idx in range(2):
                    thought = self._make_member(
                        f"c{cluster_idx}-m{member_idx}",
                        content=f"{flood} {distinct}",
                    )
                    await store.create_thought(thought)
                    await store.store_embedding(
                        thought.thought_id,
                        axis_vector,
                        model_name="test-embed",
                    )

        gates = DreamingGates(
            min_confirmations=0,
            min_age_cycles=0,
            max_promoted_per_run=20,
            min_cluster_size=2,
            cluster_algorithm="agglomerative",
            clustering_min_new_candidates=0,
            cluster_similarity_threshold=0.5,
            # This test pins the boilerplate filter's effect on the
            # persisted REFLECTION content — the gate suite handles
            # the orthogonal content-quality checks.
            cluster_quality_gating_enabled=False,
        )

        offending = {
            "flooded marker",
            "marker shared",
            "flooded marker shared",
        }

        async def _collect_keyphrases(
            *,
            boilerplate_threshold: float,
        ) -> tuple[int, list[set[str]]]:
            db = await aiosqlite.connect(":memory:")
            db.row_factory = aiosqlite.Row
            try:
                store = SqliteEngravaCore(db)
                await store.ensure_schema()
                await _seed(store)

                cfg = DreamingConfig(
                    enabled=True,
                    clustering_backend="numpy",
                    boilerplate_threshold=boilerplate_threshold,
                    boilerplate_min_corpus_size=5,
                    boilerplate_min_keyphrases_per_refl=0,
                    gates=gates,
                )
                ext = DreamingExtension(config=cfg)
                result = await ext.run_consolidation(store, current_cycle=10)
                reflections = await store.list_thoughts(
                    thought_type=ThoughtType.REFLECTION,
                )
                per_refl = [
                    {
                        str(kp.get("phrase", "")).lower()
                        for kp in json.loads(refl.content).get("top_keyphrases", [])
                    }
                    for refl in reflections
                ]
                return result.reflections_created, per_refl
            finally:
                await db.close()

        # --- Baseline: filter disabled, flooded phrase visible somewhere ---
        baseline_count, baseline_keyphrases = await _collect_keyphrases(
            boilerplate_threshold=1.0,
        )
        assert baseline_count >= 5, (
            f"expected at least 5 REFLECTIONs in baseline, got {baseline_count}"
        )
        leaked_baseline = [kps & offending for kps in baseline_keyphrases if kps & offending]
        assert leaked_baseline, (
            "baseline must surface the flooded phrase somewhere so the "
            "active-run assertion below is meaningful — got "
            f"keyphrases: {baseline_keyphrases}"
        )

        # --- Active: filter engaged, flooded phrase nowhere ---
        active_count, active_keyphrases = await _collect_keyphrases(
            boilerplate_threshold=0.30,
        )
        assert active_count >= 5
        for kps in active_keyphrases:
            leaked = kps & offending
            assert not leaked, (
                f"filter failed to remove boilerplate from a persisted "
                f"REFLECTION — top_keyphrases still contains {leaked} "
                f"(full set: {kps})"
            )

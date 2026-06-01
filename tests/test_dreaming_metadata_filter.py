"""Unit tests for ``_is_eligible_for_dreaming`` and ``DreamingConfig`` filter fields.

Each test focuses on a single filter axis from the self-anchored metadata
schema (perspective / source.is_self / source.confidence / content_type)
and verifies that the helper honours the corresponding ``DreamingConfig``
opt-in semantics.  Backward-compatibility (legacy thoughts without
metadata, default config) is covered explicitly.

Domain-level only — no SQLite involved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from engrava.config import DreamingConfig
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import MetadataValue, ThoughtRecord
from engrava.extensions.dreaming import _is_eligible_for_dreaming

if TYPE_CHECKING:
    from collections.abc import Mapping


def _make_thought(
    thought_id: str = "t-1",
    *,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Build a minimal ThoughtRecord that carries arbitrary metadata."""
    metadata_value: dict[str, MetadataValue] = dict(metadata) if metadata is not None else {}
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content="content",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        metadata=metadata_value,
    )


# ---------------------------------------------------------------------------
# DreamingConfig new fields — defaults + validation
# ---------------------------------------------------------------------------


class TestDreamingConfigFilterFields:
    """New filter fields land with backward-compatible defaults."""

    def test_defaults_preserve_legacy_behaviour(self) -> None:
        config = DreamingConfig()
        assert config.eligible_perspectives is None
        assert config.self_filter_mode == "any"
        assert config.min_source_confidence == "low"
        assert config.excluded_content_types == frozenset({"code"})
        assert config.eligible_content_types is None

    def test_self_filter_mode_validates_literal(self) -> None:
        with pytest.raises(ValueError, match="self_filter_mode"):
            DreamingConfig(self_filter_mode="invalid")  # type: ignore[arg-type]

    def test_min_source_confidence_validates_literal(self) -> None:
        with pytest.raises(ValueError, match="min_source_confidence"):
            DreamingConfig(min_source_confidence="best")  # type: ignore[arg-type]

    def test_eligible_perspectives_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError, match="eligible_perspectives"):
            DreamingConfig(
                eligible_perspectives=frozenset({"percept", "system"}),  # type: ignore[arg-type]
            )

    def test_eligible_perspectives_accepts_full_enum(self) -> None:
        config = DreamingConfig(
            eligible_perspectives=frozenset({"percept", "utterance", "thought"}),
        )
        assert config.eligible_perspectives == frozenset(
            {"percept", "utterance", "thought"},
        )


# ---------------------------------------------------------------------------
# Helper — defaults / legacy thoughts
# ---------------------------------------------------------------------------


class TestEligibilityDefaults:
    """Default config preserves the full dreaming pool."""

    def test_default_config_accepts_richly_annotated_thought(self) -> None:
        thought = _make_thought(
            metadata={
                "perspective": "utterance",
                "source": {"is_self": True, "confidence": "high"},
            },
        )
        assert _is_eligible_for_dreaming(thought, DreamingConfig()) is True

    def test_legacy_thought_without_metadata_passes(self) -> None:
        """Legacy stores carry an empty metadata dict — still eligible."""
        config = DreamingConfig(
            self_filter_mode="external_only",
            eligible_perspectives=frozenset({"percept"}),
            min_source_confidence="high",
        )
        legacy = _make_thought(metadata=None)
        assert _is_eligible_for_dreaming(legacy, config) is True


# ---------------------------------------------------------------------------
# self_filter_mode axis
# ---------------------------------------------------------------------------


class TestSelfFilterMode:
    """``self_filter_mode`` operates on ``metadata['source']['is_self']``."""

    def test_external_only_rejects_self_authored(self) -> None:
        config = DreamingConfig(self_filter_mode="external_only")
        own = _make_thought(
            metadata={"source": {"is_self": True, "confidence": "high"}},
        )
        external = _make_thought(
            metadata={"source": {"is_self": False, "confidence": "high"}},
        )
        assert _is_eligible_for_dreaming(own, config) is False
        assert _is_eligible_for_dreaming(external, config) is True

    def test_self_only_rejects_external(self) -> None:
        config = DreamingConfig(self_filter_mode="self_only")
        own = _make_thought(
            metadata={"source": {"is_self": True}},
        )
        external = _make_thought(
            metadata={"source": {"is_self": False}},
        )
        assert _is_eligible_for_dreaming(own, config) is True
        assert _is_eligible_for_dreaming(external, config) is False

    def test_is_self_missing_treated_as_eligible(self) -> None:
        """Backward compat: thoughts without is_self pass the filter."""
        config = DreamingConfig(self_filter_mode="external_only")
        thought = _make_thought(metadata={"source": {"confidence": "high"}})
        assert _is_eligible_for_dreaming(thought, config) is True

    def test_role_hint_does_not_influence_decision(self) -> None:
        """``role_hint`` is debug-only and must not influence the filter."""
        config = DreamingConfig(self_filter_mode="external_only")
        thought = _make_thought(
            metadata={
                "source": {"is_self": False, "role_hint": "assistant"},
            },
        )
        assert _is_eligible_for_dreaming(thought, config) is True

    @pytest.mark.parametrize(
        "is_self_value",
        ["true", "false", "True", "False", 0, 1, "yes", ""],
    )
    def test_non_bool_is_self_treated_as_missing(
        self,
        is_self_value: object,
    ) -> None:
        """``is_self`` is contractually a strict ``bool``.

        Non-boolean values (strings like ``"false"``, ints, empty
        strings, etc.) must NOT be interpreted by truthiness — a string
        ``"false"`` would otherwise leak through as ``True`` and break
        ``external_only`` callers in subtle ways.  Such payloads are
        treated as unclassified, exactly like a thought that never
        carried the annotation, and therefore remain eligible under
        both modes per the backward-compat policy.
        """
        ext_config = DreamingConfig(self_filter_mode="external_only")
        self_config = DreamingConfig(self_filter_mode="self_only")
        thought = _make_thought(
            metadata={"source": {"is_self": is_self_value}},  # type: ignore[dict-item]
        )
        assert _is_eligible_for_dreaming(thought, ext_config) is True
        assert _is_eligible_for_dreaming(thought, self_config) is True

    def test_bool_true_is_not_confused_with_int_one(self) -> None:
        """``True is True`` succeeds; ``1 is True`` does not — verified.

        ``bool`` is a subclass of ``int`` in Python, but identity
        comparisons (``is True`` / ``is False``) intentionally
        distinguish them so the filter does not honour ``1`` as a
        synonym of ``True`` (or ``0`` as a synonym of ``False``).
        """
        ext_config = DreamingConfig(self_filter_mode="external_only")
        # Genuine bool -> filter fires.
        bool_self = _make_thought(metadata={"source": {"is_self": True}})
        assert _is_eligible_for_dreaming(bool_self, ext_config) is False
        # int 1 -> treated as missing, eligible.
        int_one = _make_thought(metadata={"source": {"is_self": 1}})
        assert _is_eligible_for_dreaming(int_one, ext_config) is True


# ---------------------------------------------------------------------------
# perspective axis
# ---------------------------------------------------------------------------


class TestPerspectiveFilter:
    """``eligible_perspectives`` is a positive list opt-in."""

    def test_percept_only_excludes_utterance_and_thought(self) -> None:
        config = DreamingConfig(eligible_perspectives=frozenset({"percept"}))
        percept = _make_thought(
            metadata={"perspective": "percept", "source": {"is_self": False}},
        )
        utterance = _make_thought(
            metadata={"perspective": "utterance", "source": {"is_self": True}},
        )
        internal = _make_thought(
            metadata={"perspective": "thought", "source": {"is_self": True}},
        )
        assert _is_eligible_for_dreaming(percept, config) is True
        assert _is_eligible_for_dreaming(utterance, config) is False
        assert _is_eligible_for_dreaming(internal, config) is False

    def test_perspective_missing_treated_as_eligible(self) -> None:
        """No perspective annotation -> backward-compat eligible."""
        config = DreamingConfig(eligible_perspectives=frozenset({"percept"}))
        thought = _make_thought(metadata={"source": {"is_self": False}})
        assert _is_eligible_for_dreaming(thought, config) is True


# ---------------------------------------------------------------------------
# min_source_confidence axis
# ---------------------------------------------------------------------------


class TestMinSourceConfidence:
    """Confidence ranking ``low < medium < high``."""

    def test_high_threshold_rejects_lower_levels(self) -> None:
        config = DreamingConfig(min_source_confidence="high")
        high = _make_thought(
            metadata={"source": {"is_self": False, "confidence": "high"}},
        )
        medium = _make_thought(
            metadata={"source": {"is_self": False, "confidence": "medium"}},
        )
        low = _make_thought(
            metadata={"source": {"is_self": False, "confidence": "low"}},
        )
        assert _is_eligible_for_dreaming(high, config) is True
        assert _is_eligible_for_dreaming(medium, config) is False
        assert _is_eligible_for_dreaming(low, config) is False

    def test_medium_threshold_admits_medium_rejects_low(self) -> None:
        config = DreamingConfig(min_source_confidence="medium")
        medium = _make_thought(
            metadata={"source": {"confidence": "medium"}},
        )
        low = _make_thought(metadata={"source": {"confidence": "low"}})
        assert _is_eligible_for_dreaming(medium, config) is True
        assert _is_eligible_for_dreaming(low, config) is False

    def test_missing_confidence_treated_as_low(self) -> None:
        """Schema default — a caller without confidence ranks as ``"low"``."""
        config = DreamingConfig(min_source_confidence="high")
        thought = _make_thought(metadata={"source": {"is_self": False}})
        assert _is_eligible_for_dreaming(thought, config) is False


# ---------------------------------------------------------------------------
# content_type axes
# ---------------------------------------------------------------------------


class TestContentTypeFilters:
    """``excluded_content_types`` + optional ``eligible_content_types``."""

    def test_default_excludes_code(self) -> None:
        config = DreamingConfig()
        code = _make_thought(metadata={"content_type": "code"})
        text = _make_thought(metadata={"content_type": "natural_language"})
        assert _is_eligible_for_dreaming(code, config) is False
        assert _is_eligible_for_dreaming(text, config) is True

    def test_custom_excluded_set(self) -> None:
        config = DreamingConfig(
            excluded_content_types=frozenset({"code", "json", "binary"}),
        )
        for ctype in ("code", "json", "binary"):
            thought = _make_thought(metadata={"content_type": ctype})
            assert _is_eligible_for_dreaming(thought, config) is False
        ok = _make_thought(metadata={"content_type": "natural_language"})
        assert _is_eligible_for_dreaming(ok, config) is True

    def test_positive_list_restricts_pool(self) -> None:
        config = DreamingConfig(
            eligible_content_types=frozenset({"natural_language"}),
        )
        nl = _make_thought(metadata={"content_type": "natural_language"})
        speech = _make_thought(metadata={"content_type": "speech"})
        assert _is_eligible_for_dreaming(nl, config) is True
        assert _is_eligible_for_dreaming(speech, config) is False

    def test_missing_content_type_eligible_under_positive_list(self) -> None:
        """Caller omitted ``content_type`` — backward-compat eligible.

        Matches the docstring contract on
        ``DreamingConfig.eligible_content_types`` — a thought that
        declares no ``content_type`` is unaffected by the positive
        list, mirroring the missing-key semantics of every other
        filter axis and the documented backward-compat policy.
        Callers that require an explicit annotation must enforce
        that at ingest time rather than expecting the positive
        filter to fail-close on unannotated records.
        """
        config = DreamingConfig(
            eligible_content_types=frozenset({"natural_language"}),
        )
        thought = _make_thought(metadata={"source": {"is_self": False}})
        assert _is_eligible_for_dreaming(thought, config) is True


# ---------------------------------------------------------------------------
# Combined / robustness
# ---------------------------------------------------------------------------


class TestCombinedAndRobustness:
    """Cross-axis combinations + malformed metadata defensiveness."""

    def test_all_axes_must_pass_simultaneously(self) -> None:
        """Each axis is AND-combined."""
        config = DreamingConfig(
            eligible_perspectives=frozenset({"percept"}),
            self_filter_mode="external_only",
            min_source_confidence="high",
        )
        passes = _make_thought(
            metadata={
                "perspective": "percept",
                "source": {"is_self": False, "confidence": "high"},
                "content_type": "natural_language",
            },
        )
        fails_perspective = _make_thought(
            metadata={
                "perspective": "utterance",
                "source": {"is_self": False, "confidence": "high"},
            },
        )
        fails_self = _make_thought(
            metadata={
                "perspective": "percept",
                "source": {"is_self": True, "confidence": "high"},
            },
        )
        fails_confidence = _make_thought(
            metadata={
                "perspective": "percept",
                "source": {"is_self": False, "confidence": "low"},
            },
        )
        assert _is_eligible_for_dreaming(passes, config) is True
        assert _is_eligible_for_dreaming(fails_perspective, config) is False
        assert _is_eligible_for_dreaming(fails_self, config) is False
        assert _is_eligible_for_dreaming(fails_confidence, config) is False

    def test_non_dict_source_is_tolerated(self) -> None:
        """Defensive: ``source`` set to a non-dict scalar is treated as missing."""
        config = DreamingConfig(self_filter_mode="external_only")
        thought = _make_thought(metadata={"source": "user_string"})
        # Falls through is_self check (no boolean to compare) -> eligible.
        assert _is_eligible_for_dreaming(thought, config) is True

    def test_unknown_confidence_string_treated_as_low(self) -> None:
        """Defensive: typo in confidence string ranks as 0 (low)."""
        config = DreamingConfig(min_source_confidence="medium")
        thought = _make_thought(
            metadata={"source": {"confidence": "lowish"}},
        )
        assert _is_eligible_for_dreaming(thought, config) is False


# ---------------------------------------------------------------------------
# Integration: filter applied in _apply_promotions + _create_reflections
# ---------------------------------------------------------------------------


def _make_promotion_candidate(
    thought_id: str,
    *,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Build a ThoughtRecord that satisfies the existing promotion gates.

    Old enough, well-confirmed, ACTIVE — so the only remaining decision
    on the promotion path is the metadata filter.
    """
    metadata_value: dict[str, MetadataValue] = dict(metadata) if metadata is not None else {}
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content="content for dreaming",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=80,
        source="test",
        confirmation_count=5,
        confidence=0.9,
        metadata=metadata_value,
    )


class TestPromotionFilterIntegration:
    """End-to-end: ``_apply_promotions`` skips metadata-filtered thoughts."""

    async def test_external_only_promotes_only_external_thoughts(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        import aiosqlite

        from engrava.config import DreamingGates
        from engrava.extensions.dreaming import DreamingExtension
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            external = _make_promotion_candidate(
                "t-external",
                metadata={"source": {"is_self": False, "confidence": "high"}},
            )
            self_authored = _make_promotion_candidate(
                "t-self",
                metadata={"source": {"is_self": True, "confidence": "high"}},
            )
            await store.create_thought(external)
            await store.create_thought(self_authored)

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.0,
                self_filter_mode="external_only",
                # Decouple this assertion from the priority fraction cap.
                max_p1_fraction=1.0,
                gates=DreamingGates(
                    min_confirmations=2,
                    min_age_cycles=10,
                    max_promoted_per_run=20,
                    enable_reflections=False,
                ),
            )
            ext = DreamingExtension(config=cfg)
            caplog.set_level(logging.DEBUG, logger="engrava.extensions.dreaming")
            result = await ext.run_consolidation(store, current_cycle=100)

            assert "t-external" in result.promoted_ids
            assert "t-self" not in result.promoted_ids
            assert any(
                "rejected by metadata filter" in record.getMessage() for record in caplog.records
            ), "expected debug log emitted by metadata filter counter"
        finally:
            await db.close()

    async def test_default_config_does_not_change_promotion_behaviour(
        self,
    ) -> None:
        """Backward compat: default filter knobs leave the legacy decision intact."""
        import aiosqlite

        from engrava.config import DreamingGates
        from engrava.extensions.dreaming import DreamingExtension
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()
            # Two well-qualified candidates with no metadata at all.
            for tid in ("t-a", "t-b"):
                await store.create_thought(_make_promotion_candidate(tid))

            cfg = DreamingConfig(
                enabled=True,
                promote_threshold=0.0,
                # Raise the P1 fraction cap so both candidates can promote.
                # Default 5 % would let only one through with two total
                # thoughts — that limit belongs to the priority-bias logic, not this test.
                max_p1_fraction=1.0,
                gates=DreamingGates(
                    min_confirmations=2,
                    min_age_cycles=10,
                    max_promoted_per_run=20,
                    enable_reflections=False,
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=100)
            assert set(result.promoted_ids) == {"t-a", "t-b"}
        finally:
            await db.close()


def _make_clusterable_thought(
    thought_id: str,
    *,
    content: str,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Build a thought ready to be clustered (ACTIVE OBSERVATION)."""
    metadata_value: dict[str, MetadataValue] = dict(metadata) if metadata is not None else {}
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
        metadata=metadata_value,
    )


class TestReflectionFilterIntegration:
    """End-to-end: ``_create_reflections`` honours the C2a contract."""

    @staticmethod
    async def _seed_cluster(
        store: object,  # SqliteEngravaCore — opaque to avoid import cycle in fixture
        members: list[ThoughtRecord],
        *,
        embedding_dim: int = 8,
    ) -> None:
        """Persist thoughts + identical embeddings so they form a single tight cluster."""
        unit_vector = [0.0] * embedding_dim
        unit_vector[0] = 1.0
        for thought in members:
            await store.create_thought(thought)  # type: ignore[attr-defined]
            await store.store_embedding(  # type: ignore[attr-defined]
                thought.thought_id,
                unit_vector,
                model_name="test-embed",
            )

    async def test_cluster_filter_drops_ineligible_members_from_lineage(
        self,
    ) -> None:
        """Only eligible members appear in ``CONSOLIDATED_FROM`` edges (C2a)."""
        import aiosqlite

        from engrava.config import DreamingGates
        from engrava.domain.enums import EdgeType
        from engrava.extensions.dreaming import DreamingExtension
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            members = [
                _make_clusterable_thought(
                    f"e-{i}",
                    content=f"external thought {i}",
                    metadata={"source": {"is_self": False}},
                )
                for i in range(3)
            ] + [
                _make_clusterable_thought(
                    "s-1",
                    content="self utterance",
                    metadata={"source": {"is_self": True}},
                ),
            ]
            await self._seed_cluster(store, members)

            cfg = DreamingConfig(
                enabled=True,
                self_filter_mode="external_only",
                clustering_backend="numpy",
                gates=DreamingGates(
                    min_confirmations=0,
                    min_age_cycles=0,
                    max_promoted_per_run=20,
                    min_cluster_size=2,
                    cluster_algorithm="agglomerative",
                    clustering_min_new_candidates=0,
                    cluster_similarity_threshold=0.5,
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=10)

            assert result.reflections_created == 1, (
                f"expected one REFLECTION, got {result.reflections_created}"
            )

            reflections = await store.list_thoughts(
                thought_type=ThoughtType.REFLECTION,
            )
            assert len(reflections) == 1
            refl_id = reflections[0].thought_id

            edges = await store.list_edges(edge_type=EdgeType.CONSOLIDATED_FROM)
            consolidated_targets = {
                edge.to_thought_id for edge in edges if edge.from_thought_id == refl_id
            }
            assert consolidated_targets == {"e-0", "e-1", "e-2"}, (
                f"CONSOLIDATED_FROM lineage should exclude self-authored member; "
                f"got {consolidated_targets}"
            )
        finally:
            await db.close()

    async def test_cluster_dropped_when_eligible_subset_too_small(
        self,
    ) -> None:
        """Cluster drops entirely when eligible members fall under min_cluster_size."""
        import aiosqlite

        from engrava.config import DreamingGates
        from engrava.extensions.dreaming import DreamingExtension
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            # 3 self-authored + 1 external; min_cluster_size=2 leaves only 1
            # eligible member -> cluster dropped, zero REFLECTIONs.
            members = [
                _make_clusterable_thought(
                    f"s-{i}",
                    content=f"self thought {i}",
                    metadata={"source": {"is_self": True}},
                )
                for i in range(3)
            ] + [
                _make_clusterable_thought(
                    "e-1",
                    content="external thought",
                    metadata={"source": {"is_self": False}},
                ),
            ]
            await self._seed_cluster(store, members)

            cfg = DreamingConfig(
                enabled=True,
                self_filter_mode="external_only",
                clustering_backend="numpy",
                gates=DreamingGates(
                    min_confirmations=0,
                    min_age_cycles=0,
                    max_promoted_per_run=20,
                    min_cluster_size=2,
                    cluster_algorithm="agglomerative",
                    clustering_min_new_candidates=0,
                    cluster_similarity_threshold=0.5,
                ),
            )
            ext = DreamingExtension(config=cfg)
            result = await ext.run_consolidation(store, current_cycle=10)

            assert result.reflections_created == 0
            reflections = await store.list_thoughts(
                thought_type=ThoughtType.REFLECTION,
            )
            assert reflections == []
        finally:
            await db.close()

    async def test_filter_aware_idempotence_hash(
        self,
    ) -> None:
        """Same cluster + different filter -> legitimately distinct REFLECTIONs (C2a)."""
        import aiosqlite

        from engrava.config import DreamingGates
        from engrava.extensions.dreaming import DreamingExtension
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()

            members = [
                _make_clusterable_thought(
                    f"m-{i}",
                    content=f"thought {i}",
                    metadata={"source": {"is_self": i % 2 == 0}},
                )
                for i in range(4)
            ]
            await self._seed_cluster(store, members)

            gates = DreamingGates(
                min_confirmations=0,
                min_age_cycles=0,
                max_promoted_per_run=20,
                min_cluster_size=2,
                cluster_algorithm="agglomerative",
                clustering_min_new_candidates=0,
                cluster_similarity_threshold=0.5,
                # The two-member synthetic clusters here exist to
                # exercise the eligibility filter's idempotence hash;
                # their content is too sparse for the content-quality
                # gates, which are tested in their own suite.
                cluster_quality_gating_enabled=False,
            )

            # First run: external_only filter -> REFLECTION from {m-1, m-3}.
            ext_external = DreamingExtension(
                config=DreamingConfig(
                    enabled=True,
                    self_filter_mode="external_only",
                    clustering_backend="numpy",
                    gates=gates,
                ),
            )
            r1 = await ext_external.run_consolidation(store, current_cycle=10)
            assert r1.reflections_created == 1

            # Second run: self_only filter -> REFLECTION from {m-0, m-2}.
            # Filter-aware hash means this is a legitimately distinct cluster
            # signature, so a brand-new REFLECTION must materialise rather
            # than being idempotence-skipped against the first one.
            ext_self = DreamingExtension(
                config=DreamingConfig(
                    enabled=True,
                    self_filter_mode="self_only",
                    clustering_backend="numpy",
                    gates=gates,
                ),
            )
            r2 = await ext_self.run_consolidation(store, current_cycle=11)
            assert r2.reflections_created == 1

            reflections = await store.list_thoughts(
                thought_type=ThoughtType.REFLECTION,
            )
            assert len(reflections) == 2, (
                "filter-aware hash should produce two distinct REFLECTIONs"
            )
            sources = sorted(r.source for r in reflections)
            # Sources are formed as `dreaming:<cluster_hash>` — distinct hashes
            # demonstrate the filter-aware idempotence contract.
            assert sources[0] != sources[1]
        finally:
            await db.close()

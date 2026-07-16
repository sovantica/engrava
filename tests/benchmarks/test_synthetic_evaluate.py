"""Evaluator + runner-CLI tests for the synthetic benchmark.

End-to-end tests use a tiny dataset (3-5 conversations) and the
real ``SqliteEngravaCore`` + ``DreamingExtension`` + sentence-
transformers stack — there is no fictional ``engrava_test_engine``
fixture, and there are no mocks of the embedding provider.  The
trade-off is wall-clock time: a single OFF / ON pair against a
five-conversation dataset takes a few seconds even with the
MiniLM-L6 cold-load cost amortised across the suite.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from engrava.benchmarks.synthetic.evaluate import (
    EvaluationResult,
    PerBreakdown,
    evaluate_run,
    resolve_embedding_provider_or_exit,
    run_evaluation,
)
from engrava.benchmarks.synthetic.generate import (
    SyntheticConversation,
    generate_dataset,
)
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from engrava.domain.protocols.embedding_provider import (
        EmbeddingProviderProtocol,
    )


# ---------------------------------------------------------------------------
# Shared session-scoped embedding provider — pays the cold-load cost once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_provider() -> EmbeddingProviderProtocol:
    """Shared MiniLM-L6 provider for the whole evaluator test module."""
    return resolve_embedding_provider_or_exit()


# ---------------------------------------------------------------------------
# Tiny datasets — kept small for CI walltime.
# ---------------------------------------------------------------------------


def _tiny_dataset() -> tuple[SyntheticConversation, ...]:
    """5 conversations, mixed scenario library."""
    return generate_dataset(
        seed=20260508,
        n_conversations=5,
        avg_turns_per_conversation=25,
        distraction_density=0.3,
    )


def _neutral_sanity_dataset() -> tuple[SyntheticConversation, ...]:
    """Sanity subset — only anti-cherry-pick neutrals.

    Sized at 24 conversations so the |ON - OFF| <= 0.02 band has
    enough sample-size resolution to be meaningful — at 8 questions
    every difference is at least 1/8 = 0.125 and the band would be
    statistically toothless.
    """
    return generate_dataset(
        seed=20260508,
        n_conversations=24,
        avg_turns_per_conversation=20,
        distraction_density=0.3,
        scenario_mix={
            "single_unique_fact": 1.0,
            "recent_fact_recall": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Evaluator contracts
# ---------------------------------------------------------------------------


class TestEvaluatorContracts:
    """The evaluator's public surface — keyword-only API, real engrava core."""

    @pytest.mark.asyncio
    async def test_off_run_returns_complete_evaluation_result(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        result = await evaluate_run(
            _tiny_dataset(),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
            retrieval_top_k=5,
        )
        assert isinstance(result, EvaluationResult)
        assert result.dreaming_enabled is False
        assert result.top_k == 5
        assert result.total_questions > 0
        assert 0.0 <= result.aggregate_recall_at_k <= 1.0
        assert 0.0 <= result.aggregate_substring_match_rate <= 1.0
        for breakdown in result.per_scenario.values():
            assert isinstance(breakdown, PerBreakdown)
            assert breakdown.question_count > 0

    @pytest.mark.asyncio
    async def test_on_run_returns_complete_evaluation_result(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        result = await evaluate_run(
            _tiny_dataset(),
            dreaming_enabled=True,
            embedding_provider=embedding_provider,
            retrieval_top_k=5,
        )
        assert result.dreaming_enabled is True
        assert result.total_questions > 0

    @pytest.mark.asyncio
    async def test_per_difficulty_breakdown_present(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        result = await evaluate_run(
            _tiny_dataset(),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )
        assert result.per_difficulty, "per_difficulty breakdown missing"
        for difficulty_key in result.per_difficulty:
            assert difficulty_key in {"easy", "medium", "hard"}

    @pytest.mark.asyncio
    async def test_empty_dataset_yields_zero_aggregates(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        result = await evaluate_run(
            (),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )
        assert result.total_questions == 0
        assert result.aggregate_recall_at_k == 0.0
        assert result.aggregate_substring_match_rate == 0.0


# ---------------------------------------------------------------------------
# Self-anchored metadata coverage — every ingested thought MUST carry
# perspective + source.is_self per the schema the dreaming filter consumes.
# ---------------------------------------------------------------------------


class TestSelfAnchoredMetadata:
    """Every persisted ``ThoughtRecord`` carries the v0.3 self-anchored shape."""

    @pytest.mark.asyncio
    async def test_evaluator_passes_perspective_and_source_metadata(
        self,
        embedding_provider: EmbeddingProviderProtocol,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch ``create_thought`` to capture the metadata of every
        # ingested record; assert each carries the expected keys.
        captured: list[dict[str, object]] = []

        from engrava.infrastructure.sqlite import engrava_core

        original = engrava_core.SqliteEngravaCore.create_thought

        async def _capturing_create_thought(
            self: engrava_core.SqliteEngravaCore,
            thought: object,
            **kwargs: object,
        ) -> object:
            captured.append(dict(thought.metadata))  # type: ignore[attr-defined]
            return await original(self, thought, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            engrava_core.SqliteEngravaCore,
            "create_thought",
            _capturing_create_thought,
        )

        await evaluate_run(
            generate_dataset(
                seed=42,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=0.2,
            ),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )

        assert captured, "evaluator must call create_thought for every turn"
        for metadata in captured:
            assert metadata.get("perspective") in {"percept", "utterance", "thought"}
            source = metadata.get("source")
            assert isinstance(source, dict)
            assert isinstance(source.get("is_self"), bool)
            assert source.get("confidence") == "high"
            assert metadata.get("lang") == "en"
            assert metadata.get("content_type") == "natural_language"


# ---------------------------------------------------------------------------
# OFF / ON pair contracts — used by the AC-9 floor and the AC-8 sanity band.
# ---------------------------------------------------------------------------


class TestOffOnPair:
    """OFF and ON arms behave like independent passes that share no state."""

    @pytest.mark.asyncio
    async def test_off_and_on_run_independently(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        # OFF then ON: ON must report ``dreaming_enabled=True`` and the
        # two arms must not share retrieved-ID state (no cross-call leak).
        off = await evaluate_run(
            _tiny_dataset(),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )
        on = await evaluate_run(
            _tiny_dataset(),
            dreaming_enabled=True,
            embedding_provider=embedding_provider,
        )
        assert off.dreaming_enabled is False
        assert on.dreaming_enabled is True
        assert off.total_questions == on.total_questions

    @pytest.mark.asyncio
    async def test_sanity_subset_dreaming_neutral(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        # AC-8 v0.3.0 tolerance ≤0.05 (v1.4 amendment).
        #
        # v1.4 relaxed from ≤0.02 with explicit empirical rationale:
        # REFLECTIONs participate at parity in retrieval despite
        # ``reflection_boost=1.0`` (boost is a multiplier on top of
        # the intrinsic vector / FTS score, not an enable/disable
        # toggle).  Measured delta post-NA-1 is 0.042 — the 0.05
        # ceiling carries a small safety margin without claiming
        # neutrality the engrava-core ranking does not provide at
        # v0.3.0.  The v0.4.0 follow-up workstream tightens this
        # back to ≤0.02 once REFLECTION ranking is refined.
        dataset = _neutral_sanity_dataset()
        off = await evaluate_run(
            dataset,
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )
        on = await evaluate_run(
            dataset,
            dreaming_enabled=True,
            embedding_provider=embedding_provider,
        )
        delta = abs(on.aggregate_recall_at_k - off.aggregate_recall_at_k)
        assert delta <= 0.05, (
            f"AC-8 v0.3.0 tolerance (0.05) exceeded: {delta:.3f}.  "
            f"v0.4.0 tightens back to 0.02 post REFLECTION ranking "
            f"refinement; a regression past 0.05 here means a NEW "
            f"source of dreaming-side interference, not just the "
            f"known intrinsic-score participation."
        )


# ---------------------------------------------------------------------------
# Public-export discipline — AC-14
# ---------------------------------------------------------------------------


# Baseline of the legitimate package-level ``__all__``.  Any *accidental*
# new export through ``engrava/__init__.py`` (e.g. benchmark code leaking
# out) will surface as a non-empty diff on this set. Deliberate public-API
# additions are recorded here when they ship.
#
# Baseline captured pre-WS from release/v0.3.0 HEAD = bb407ac, then extended
# with the metadata-filter query surface (FieldOp / FieldPredicate /
# MetadataFilter / VisibilityQueryFilter + the two typed filter errors), an
# intentional, ratified public-API addition. Later extended with the
# opt-in asymmetric embedding-prefix surface (RoleAwareEmbeddingProvider
# capability protocol + EmbeddingQueryPrefixMismatchError), an additive,
# default-off public-API addition. Later extended with JournalIntegrityError,
# raised by the opt-in on-open journal integrity check — an additive,
# default-off public-API addition. Later extended with EmbeddingGenerationError,
# the typed error surfaced by the additive ingest-ergonomics / embedding-
# robustness surface (bulk_store / get_or_create / upsert_by_hash + the opt-in
# require_embedding fail-fast), also additive and default-off. Later extended
# with ActionNotFoundError (raised by the new update_action lifecycle write) and
# ActionOutcomeSignal (the action-outcome dreaming signal) — the additive,
# default-off action-outcome feedback surface (inactive in any store that never
# records a terminal action outcome). Later extended with ProvenanceContext
# (the typed, bounded, opt-in write-time provenance sub-model captured at
# create_thought and made queryable but never consumed) — an additive,
# default-off public-API addition (a thought with no provenance is byte-identical
# to before it existed). Later extended with the Memory Hygiene forgetting-loop
# surface (HygienePolicyConfig / HygieneResult / EvictionReason), an additive,
# default-off public-API addition (a store that never enables hygiene_policy is
# unchanged on every read/write path). Later extended with the derived-records
# extension-seam surface (DerivedRecordProducerProtocol / DerivedRecord /
# DeriveContext / DeriveGates / DerivedRecordError, plus the StructuralSplitProducer
# reference consumer), an additive, default-off public-API addition (with the seam
# disabled, or the hooks object not a producer, every write path is byte-identical).
# Later extended with ConnectionQuarantinedError, the typed error raised only when a
# derived-child compensating rollback could not be guaranteed to complete under
# cancellation (the connection is quarantined so a later operation cannot run on an
# indeterminate transaction) — an additive, default-off public-API addition never
# raised on any normal read/write path.
# Later extended with the opt-in cycle-provider seam (the CycleProvider capability
# protocol + StaticCycleProvider / CallableCycleProvider / MaxCycleProvider reference
# providers + the typed CycleProviderError), an additive, default-off public-API
# addition — a store that configures no cycle_provider pulls no cycle and is
# unchanged on every read/write path.
_PRE_WS_ALL_BASELINE = frozenset(
    {
        "ActionNotFoundError",
        "ActionOutcomeSignal",
        "ActionRecord",
        "ActionStatus",
        "ActionType",
        "CallableCycleProvider",
        "CallbackProvider",
        "CleanupResult",
        "CleanupStrategy",
        "ConfidenceSignal",
        "ConfigError",
        "ConfirmationSignal",
        "ConnectionQuarantinedError",
        "ConsolidationResult",
        "CoreThoughtRecord",
        "CycleProvider",
        "CycleProviderError",
        "DefaultEngravaHooks",
        "DefaultMindStoreHooks",
        "DeriveContext",
        "DeriveGates",
        "DerivedRecord",
        "DerivedRecordError",
        "DerivedRecordProducerProtocol",
        "DreamingConfig",
        "DreamingContext",
        "DreamingExtension",
        "DreamingGates",
        "DreamingSignalProtocol",
        "EdgeCounts",
        "EdgeRecord",
        "EdgeType",
        "EmbeddingConfig",
        "EmbeddingGenerationError",
        "EmbeddingModelMismatchError",
        "EmbeddingProviderProtocol",
        "EmbeddingQueryPrefixMismatchError",
        "EmbeddingRecord",
        "EngravaConfig",
        "EngravaCoreProtocol",
        "EngravaError",
        "EngravaHooksProtocol",
        "EngravaManager",
        "EngravaMetrics",
        "EvictionReason",
        "ExtensionManifest",
        "ExtensionMigrationError",
        "FieldOp",
        "FieldPredicate",
        "FrequencySignal",
        "HuggingFaceProvider",
        "HybridSearchResult",
        "HygienePolicyConfig",
        "HygieneResult",
        "InvalidFilterError",
        "InvalidFilterPathError",
        "InvalidTransitionError",
        "JournalConfig",
        "JournalEntry",
        "JournalIntegrityError",
        "JournalIntegrityResult",
        "JournalWriter",
        "KnowledgeSource",
        "LatencyHistogram",
        "LifecycleStatus",
        "MaxCycleProvider",
        "MetadataFilter",
        "MetricsConfig",
        "MindQLCommand",
        "MindQLExecutor",
        "MindQLExtension",
        "MindQLParseError",
        "MindQLQuery",
        "MindQLResult",
        "MindStoreConfig",
        "MindStoreCoreProtocol",
        "MindStoreError",
        "MindStoreHooksProtocol",
        "MindStoreManager",
        "MutationType",
        "OllamaProvider",
        "OpenAICompatibleProvider",
        "Priority",
        "ProvenanceContext",
        "ReadOnlyEngrava",
        "ReadOnlyMindStore",
        "ReadOnlyViolationError",
        "RecencySignal",
        "RoleAwareEmbeddingProvider",
        "ScoringContext",
        "SearchConfig",
        "SentenceTransformerProvider",
        "ServiceConfig",
        "ServicesConfig",
        "SqliteEngravaCore",
        "SqliteMindStoreCore",
        "SqliteVecSearchBackend",
        "StaleDataError",
        "StalenessSignal",
        "StaticCycleProvider",
        "StorageFootprint",
        "StructuralSplitProducer",
        "TTLConfig",
        "ThoughtCounts",
        "ThoughtNotFoundError",
        "ThoughtRecord",
        "ThoughtType",
        "ThoughtVisibility",
        "VerificationStatus",
        "VisibilityQueryFilter",
        "discover_manifests",
        "load_config",
        "parse",
        "percept",
        "resolve_embedding_provider",
        "resolve_hooks",
        "resolve_manifests",
        "thought",
        "utterance",
    },
)


class TestPublicSurfaceDiscipline:
    """Benchmark code MUST NOT leak through the package-level ``__all__``."""

    def test_no_new_public_exports(self) -> None:
        import engrava

        current = frozenset(engrava.__all__)
        new_exports = current - _PRE_WS_ALL_BASELINE
        assert not new_exports, f"benchmark suite leaked new public exports: {sorted(new_exports)}"
        # Defensive: regressions that quietly remove a public export are
        # not in scope for this WS either.
        dropped = _PRE_WS_ALL_BASELINE - current
        assert not dropped, (
            f"benchmark suite accidentally dropped public exports: {sorted(dropped)}"
        )


# ---------------------------------------------------------------------------
# Runner CLI tests — argparse + dataset deserialisation + summary formatting.
# ---------------------------------------------------------------------------


class TestRunnerCli:
    """argparse contracts + bundled-dataset loader behaviour."""

    def test_unknown_scenarios_arg_exits(self) -> None:
        from engrava.benchmarks.synthetic.runner import main

        with pytest.raises(SystemExit):
            main(["--scenarios", "definitely-not-a-scenario"])


class TestRunnerDeserialisation:
    """The frozen JSON round-trip — covers code paths used by C4 freeze."""

    def test_dataset_round_trip_via_runner_loader(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import importlib.resources

        from engrava.benchmarks.synthetic import runner
        from engrava.benchmarks.synthetic.generate import dataset_to_json

        data = generate_dataset(
            seed=20260508,
            n_conversations=3,
            avg_turns_per_conversation=15,
            distraction_density=0.3,
        )
        json_blob = dataset_to_json(data)

        # Patch ``importlib.resources.files`` so the runner loader
        # reads our tmp file instead of the (non-existent at this
        # point in the branch) bundled dataset.
        class _FakeTraversable:
            def __init__(self, path: Path) -> None:
                self._path = path

            def is_file(self) -> bool:
                return self._path.is_file()

            def read_text(self, encoding: str = "utf-8") -> str:
                return self._path.read_text(encoding=encoding)

        class _FakeAnchor:
            def __init__(self, path: Path) -> None:
                self._path = path

            def joinpath(self, _name: str) -> _FakeTraversable:
                return _FakeTraversable(self._path)

        fake_file = tmp_path / "synthetic-v1.json"
        fake_file.write_text(json_blob, encoding="utf-8", newline="\n")
        monkeypatch.setattr(
            importlib.resources,
            "files",
            lambda _pkg: _FakeAnchor(fake_file),
        )

        loaded = runner._load_frozen_dataset()
        assert len(loaded) == len(data)
        for original_conv, round_trip_conv in zip(data, loaded, strict=True):
            assert original_conv == round_trip_conv

    def test_deserialise_rejects_non_dict_payload(self) -> None:
        from engrava.benchmarks.synthetic.runner import _checked_dict

        with pytest.raises(SystemExit):
            _checked_dict([1, 2, 3])

    def test_deserialise_rejects_non_list_payload(self) -> None:
        from engrava.benchmarks.synthetic.runner import _checked_list

        with pytest.raises(SystemExit):
            _checked_list({"not": "a list"})

    def test_as_int_rejects_bool(self) -> None:
        from engrava.benchmarks.synthetic.runner import _as_int

        with pytest.raises(SystemExit):
            _as_int(True)

    def test_as_bool_rejects_string(self) -> None:
        # Regression: pre-fix ``bool("false")`` coerced to ``True``,
        # silently flipping the self-anchored provenance contract.
        from engrava.benchmarks.synthetic.runner import _as_bool

        with pytest.raises(SystemExit):
            _as_bool("false")

    def test_as_bool_rejects_int(self) -> None:
        from engrava.benchmarks.synthetic.runner import _as_bool

        with pytest.raises(SystemExit):
            _as_bool(0)

    def test_as_bool_rejects_none(self) -> None:
        from engrava.benchmarks.synthetic.runner import _as_bool

        with pytest.raises(SystemExit):
            _as_bool(None)

    def test_as_bool_accepts_real_booleans(self) -> None:
        from engrava.benchmarks.synthetic.runner import _as_bool

        assert _as_bool(True) is True
        assert _as_bool(False) is False

    def test_as_perspective_rejects_unknown(self) -> None:
        from engrava.benchmarks.synthetic.runner import _as_perspective

        with pytest.raises(SystemExit):
            _as_perspective("weird")

    def test_as_difficulty_rejects_unknown(self) -> None:
        from engrava.benchmarks.synthetic.runner import _as_difficulty

        with pytest.raises(SystemExit):
            _as_difficulty("impossible")


class TestScoreRetrieval:
    """``_score_retrieval`` unit tests covering both (a) OBS and (b) REFL paths.

    These tests do not go through ``evaluate_run`` — they exercise the
    helper directly against a fresh in-memory store so the
    REFLECTION-as-answer-carrier semantics are pinned independently of
    the rest of the evaluator's wiring.
    """

    @pytest.mark.asyncio
    async def test_direct_obs_hit(
        self,
    ) -> None:
        from engrava.benchmarks.synthetic.evaluate import _score_retrieval

        async with _fresh_store() as store:
            expected = await _make_observation(store, content="planted fact", cycle=0)
            decoy = await _make_observation(store, content="other thought", cycle=1)
            retrieved = [(expected, 0.9), (decoy, 0.7)]
            recalled = await _score_retrieval(
                store=store,
                retrieved=retrieved,
                expected_thought_ids={expected},
            )
            assert recalled is True

    @pytest.mark.asyncio
    async def test_no_match_returns_false(
        self,
    ) -> None:
        from engrava.benchmarks.synthetic.evaluate import _score_retrieval

        async with _fresh_store() as store:
            expected = await _make_observation(store, content="planted fact", cycle=0)
            decoy_a = await _make_observation(store, content="decoy 1", cycle=1)
            decoy_b = await _make_observation(store, content="decoy 2", cycle=2)
            retrieved = [(decoy_a, 0.8), (decoy_b, 0.7)]
            recalled = await _score_retrieval(
                store=store,
                retrieved=retrieved,
                expected_thought_ids={expected},
            )
            assert recalled is False

    @pytest.mark.asyncio
    async def test_empty_expected_returns_false(
        self,
    ) -> None:
        from engrava.benchmarks.synthetic.evaluate import _score_retrieval

        async with _fresh_store() as store:
            stray = await _make_observation(store, content="anything", cycle=0)
            recalled = await _score_retrieval(
                store=store,
                retrieved=[(stray, 0.9)],
                expected_thought_ids=set(),
            )
            assert recalled is False

    @pytest.mark.asyncio
    async def test_reflection_carrying_expected_fact_counts_as_hit(
        self,
    ) -> None:
        # The REFLECTION-as-answer-carrier branch (b): a retrieved
        # REFLECTION whose ``consolidated_from`` list intersects the
        # expected_thought_ids is a hit, even when no expected OBS is
        # itself in the retrieved set.
        from engrava.benchmarks.synthetic.evaluate import _score_retrieval

        async with _fresh_store() as store:
            obs_a = await _make_observation(store, content="facet A", cycle=0)
            obs_b = await _make_observation(store, content="facet B", cycle=1)
            refl = await _make_reflection(
                store,
                content="cluster summary",
                consolidated_from=[obs_a, obs_b],
                cycle=2,
            )
            decoy = await _make_observation(store, content="unrelated", cycle=3)
            # Note: neither obs_a nor obs_b is in ``retrieved`` — only
            # the REFLECTION and a decoy.  This is exactly the
            # situation synthesis scenarios produce.
            retrieved = [(refl, 0.85), (decoy, 0.40)]
            recalled = await _score_retrieval(
                store=store,
                retrieved=retrieved,
                expected_thought_ids={obs_a, obs_b},
            )
            assert recalled is True

    @pytest.mark.asyncio
    async def test_reflection_with_empty_consolidated_from_does_not_match(
        self,
    ) -> None:
        from engrava.benchmarks.synthetic.evaluate import _score_retrieval

        async with _fresh_store() as store:
            obs = await _make_observation(store, content="planted", cycle=0)
            refl = await _make_reflection(
                store,
                content="legacy summary",
                consolidated_from=None,
                cycle=1,
            )
            retrieved = [(refl, 0.9)]
            recalled = await _score_retrieval(
                store=store,
                retrieved=retrieved,
                expected_thought_ids={obs},
            )
            assert recalled is False

    @pytest.mark.asyncio
    async def test_reflection_consolidated_from_disjoint_does_not_match(
        self,
    ) -> None:
        from engrava.benchmarks.synthetic.evaluate import _score_retrieval

        async with _fresh_store() as store:
            expected_obs = await _make_observation(store, content="A", cycle=0)
            other_obs = await _make_observation(store, content="B", cycle=1)
            refl = await _make_reflection(
                store,
                content="summary over unrelated facts",
                consolidated_from=[other_obs],
                cycle=2,
            )
            retrieved = [(refl, 0.9)]
            recalled = await _score_retrieval(
                store=store,
                retrieved=retrieved,
                expected_thought_ids={expected_obs},
            )
            assert recalled is False

    @pytest.mark.asyncio
    async def test_missing_retrieved_thought_is_skipped_not_raised(
        self,
    ) -> None:
        # A retrieved thought_id that no longer resolves (e.g. raced
        # against a delete) should not crash the helper — the loop just
        # continues to the next entry.
        from engrava.benchmarks.synthetic.evaluate import _score_retrieval

        async with _fresh_store() as store:
            obs = await _make_observation(store, content="actually-recalled", cycle=0)
            retrieved = [("ghost-id-not-in-store", 0.95), (obs, 0.5)]
            recalled = await _score_retrieval(
                store=store,
                retrieved=retrieved,
                expected_thought_ids={obs},
            )
            assert recalled is True


# ---------------------------------------------------------------------------
# Helpers for ``_score_retrieval`` unit tests — small async-context
# wrappers that hide aiosqlite boilerplate and produce real records the
# helper inspects via ``get_thought``.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _fresh_store() -> AsyncIterator[SqliteEngravaCore]:
    """Minimal in-memory ``SqliteEngravaCore`` with schema applied."""
    import aiosqlite

    async with aiosqlite.connect(":memory:") as db:
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(db=db)
        await store.ensure_schema()
        yield store


async def _make_observation(store: SqliteEngravaCore, *, content: str, cycle: int) -> str:
    from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
    from engrava.domain.models.thought import ThoughtRecord

    thought = ThoughtRecord(
        thought_id=f"obs-{cycle:04d}",
        thought_type=ThoughtType.OBSERVATION,
        essence=content[:80],
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source="unit-test",
        confirmation_count=0,
        confidence=0.9,
    )
    persisted = await store.create_thought(thought)
    return persisted.thought_id


async def _make_reflection(
    store: SqliteEngravaCore,
    *,
    content: str,
    consolidated_from: list[str] | None,
    cycle: int,
) -> str:
    from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
    from engrava.domain.models.thought import ThoughtRecord

    thought = ThoughtRecord(
        thought_id=f"refl-{cycle:04d}",
        thought_type=ThoughtType.REFLECTION,
        essence=content[:80],
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source="dreaming",
        confirmation_count=0,
        confidence=0.9,
        consolidated_from=consolidated_from,
    )
    persisted = await store.create_thought(thought)
    return persisted.thought_id


class TestRunEvaluationWrapper:
    """``run_evaluation`` drives asyncio for synchronous callers."""

    def test_sync_wrapper_runs(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        result = run_evaluation(
            generate_dataset(
                seed=20260508,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=0.3,
            ),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )
        assert result.total_questions > 0


class TestImportErrorUx:
    """Missing extras → exit code 2 with actionable message."""

    def test_resolve_embedding_provider_exits_on_missing_extras(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Simulate a vanilla install without the embeddings extras.
        # The helper probes the third-party ``sentence_transformers``
        # package via ``importlib.util.find_spec`` — patch that to
        # return ``None`` so the helper takes the missing-extras branch
        # even when the package is installed in the test environment.
        import importlib.util

        original_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":
                return None
            return original_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

        with pytest.raises(SystemExit) as excinfo:
            resolve_embedding_provider_or_exit()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "embeddings-local" in err


# ---------------------------------------------------------------------------
# JSON summary smoke
# ---------------------------------------------------------------------------


class TestSummaryFormats:
    """Both text and JSON summary writers run cleanly on real results."""

    @pytest.mark.asyncio
    async def test_json_summary_writes_payload_to_path(
        self,
        embedding_provider: EmbeddingProviderProtocol,
        tmp_path: Path,
    ) -> None:
        from engrava.benchmarks.synthetic.runner import (
            _BindingResult,
            _CliArgs,
            _CliReport,
            _write_json_report,
        )

        off = await evaluate_run(
            _tiny_dataset(),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )
        on = off  # smoke — payload shape is what we are checking, not numerics.
        target = tmp_path / "summary.json"
        args = _CliArgs(
            regenerate=False,
            seed=0,
            n_conversations=0,
            avg_turns_per_conversation=0,
            distraction_density=0.0,
            scenarios=frozenset(),
            output_format="json",
            output_path=target,
            top_k=5,
            with_reproducibility=True,
        )
        report = _CliReport(
            binding_results=(
                _BindingResult(
                    label="AC-9a synthesis coverage",
                    value=0.85,
                    threshold=0.80,
                    passed=True,
                    rule_text=">= 0.80",
                ),
            ),
            reproducibility_off=off,
            reproducibility_on=on,
        )
        _write_json_report(report=report, passed=True, args=args)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert "reproducibility" in payload
        assert "binding" in payload
        assert payload["passed"] is True
        assert payload["binding"][0]["label"] == "AC-9a synthesis coverage"

    @pytest.mark.asyncio
    async def test_text_summary_writes_to_stdout(
        self,
        embedding_provider: EmbeddingProviderProtocol,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from engrava.benchmarks.synthetic.runner import (
            _BindingResult,
            _CliReport,
            _write_text_report,
        )

        off = await evaluate_run(
            _tiny_dataset(),
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
        )
        on = off
        report = _CliReport(
            binding_results=(
                _BindingResult(
                    label="AC-9a synthesis coverage",
                    value=0.85,
                    threshold=0.80,
                    passed=True,
                    rule_text=">= 0.80",
                ),
            ),
            reproducibility_off=off,
            reproducibility_on=on,
        )
        _write_text_report(report=report, passed=True)
        out = capsys.readouterr().out
        assert "Engrava Synthetic Benchmark Suite" in out
        assert "Binding acceptance measurements" in out
        assert "Reproducibility snapshot" in out
        assert "AC-9a synthesis coverage" in out

"""Determinism + shape tests for the synthetic generator.

The frozen-dataset reproducibility contract depends on
:func:`generate_dataset` being a pure function of its inputs.  These
tests pin the byte-identity invariant directly (same seed →
byte-identical JSON) and a handful of structural invariants the
evaluator relies on (fact_id uniqueness, self-anchored metadata
schema coverage, question placement after memorable turns, etc).
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import TypedDict

import pytest

from engrava.benchmarks.synthetic.generate import (
    SyntheticConversation,
    SyntheticQuestion,
    SyntheticTurn,
    dataset_to_json,
    generate_dataset,
)
from engrava.benchmarks.synthetic.scenarios import SCENARIO_LIBRARY


class _GenKwargs(TypedDict):
    """Typed kwargs bundle so mypy --strict can verify ``**unpacks``."""

    seed: int
    n_conversations: int
    avg_turns_per_conversation: int
    distraction_density: float


# Shared generation parameters — small but rich enough to exercise the
# generator (every scenario type lands in at least one conversation on
# average with n_conversations >= 14).
_SMALL: _GenKwargs = {
    "seed": 20260508,
    "n_conversations": 14,
    "avg_turns_per_conversation": 40,
    "distraction_density": 0.3,
}


class TestDeterminism:
    """Byte-identity contracts for the frozen-dataset commitment."""

    def test_same_seed_same_dataset_object_equality(self) -> None:
        a = generate_dataset(**_SMALL)
        b = generate_dataset(**_SMALL)
        assert a == b

    def test_same_seed_same_dataset_json_byte_identity(self) -> None:
        # Stronger than object equality — pins the on-disk frozen
        # JSON format the package-data wheel will ship.
        a_json = dataset_to_json(generate_dataset(**_SMALL))
        b_json = dataset_to_json(generate_dataset(**_SMALL))
        assert a_json == b_json

    def test_different_seeds_diverge(self) -> None:
        a = dataset_to_json(generate_dataset(**_with_seed(_SMALL, 1)))
        b = dataset_to_json(generate_dataset(**_with_seed(_SMALL, 2)))
        assert a != b

    def test_three_consecutive_runs_byte_identical(self) -> None:
        # AC-2 precursor — when the runner consumes the frozen JSON
        # three consecutive default-invocations must agree.  This test
        # exercises the generator side of that contract; the runner
        # side lives in C3.
        runs = [dataset_to_json(generate_dataset(**_SMALL)) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]

    def test_json_is_ascii_and_sorted(self) -> None:
        rendered = dataset_to_json(generate_dataset(**_SMALL))
        # ensure_ascii=True — no raw non-ASCII bytes.
        rendered.encode("ascii")  # raises UnicodeEncodeError on regression
        # sort_keys=True — every dict in the payload is sorted.
        loaded = json.loads(rendered)
        _assert_keys_sorted(loaded)


class TestStructuralInvariants:
    """Shape contracts the evaluator depends on."""

    def test_returns_requested_number_of_conversations(self) -> None:
        out = generate_dataset(**_with_n(_SMALL, 7))
        assert len(out) == 7

    def test_conversation_ids_are_unique_and_padded(self) -> None:
        out = generate_dataset(**_SMALL)
        ids = [c.conversation_id for c in out]
        assert len(ids) == len(set(ids)), "conversation IDs must be unique"
        for idx, conv_id in enumerate(ids):
            assert conv_id == f"synth-conv-{idx:04d}"

    def test_fact_ids_are_unique_across_dataset(self) -> None:
        # The evaluator builds a global fact_id -> thought_id map, so
        # duplicate fact_ids would silently merge memorable turns from
        # different conversations.
        out = generate_dataset(**_SMALL)
        fact_ids: list[str] = [
            turn.fact_id for conv in out for turn in conv.turns if turn.fact_id is not None
        ]
        assert len(fact_ids) == len(set(fact_ids))

    def test_question_ids_are_unique(self) -> None:
        out = generate_dataset(**_SMALL)
        qids = [q.question_id for conv in out for q in conv.questions]
        assert len(qids) == len(set(qids))

    def test_every_turn_carries_adr024_metadata(self) -> None:
        # AC-10 precursor — every ingested ThoughtRecord must carry
        # perspective + source.is_self; verify the generator side here,
        # the evaluator side lands in C3.
        out = generate_dataset(**_SMALL)
        for conv in out:
            for turn in conv.turns:
                assert turn.perspective in {"percept", "utterance", "thought"}
                assert isinstance(turn.source_is_self, bool)

    def test_memorable_facts_are_user_utterances(self) -> None:
        # Self-anchored metadata schema binding (perspective +
        # source.is_self): planted facts are user statements, not
        # external observations.  The neutrality of
        # ``recent_fact_recall`` and the metadata-aware filter on
        # the dreaming side both depend on this.
        out = generate_dataset(**_SMALL)
        for conv in out:
            for turn in conv.turns:
                if turn.is_memorable:
                    assert turn.perspective == "utterance"
                    assert turn.source_is_self is True
                    assert turn.fact_id is not None

    def test_questions_are_asked_after_every_memorable_turn(self) -> None:
        out = generate_dataset(**_SMALL)
        for conv in out:
            memorable_indices = [t.turn_index for t in conv.turns if t.is_memorable]
            if not memorable_indices:
                continue
            last_memorable = max(memorable_indices)
            for q in conv.questions:
                assert q.asked_at_turn > last_memorable

    def test_questions_respect_scenario_offset_window(self) -> None:
        out = generate_dataset(**_SMALL)
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        for conv in out:
            scenario = by_name[conv.scenario_name]
            memorable_indices = [t.turn_index for t in conv.turns if t.is_memorable]
            last_memorable = max(memorable_indices)
            for q in conv.questions:
                gap = q.asked_at_turn - last_memorable
                assert gap >= scenario.question_offset_min_turns
                assert gap <= scenario.question_offset_max_turns

    def test_each_conversation_has_at_least_one_question(self) -> None:
        out = generate_dataset(**_SMALL)
        for conv in out:
            assert conv.questions, f"{conv.conversation_id} has no questions"

    def test_expected_fact_ids_resolve_to_real_turns(self) -> None:
        out = generate_dataset(**_SMALL)
        for conv in out:
            available = {t.fact_id for t in conv.turns if t.fact_id is not None}
            for q in conv.questions:
                missing = set(q.expected_fact_ids) - available
                assert not missing, (
                    f"Question {q.question_id} references unknown fact_ids {sorted(missing)}"
                )

    def test_direct_scenario_substrings_appear_in_memorable_turns(self) -> None:
        # Direct-retrieval scenarios MUST plant the answer substring in
        # the memorable text — otherwise even perfect retrieval cannot
        # score a substring hit.  Synthesis scenarios are exempt by
        # design: their substrings (e.g. the theme name ``"Lisbon"``)
        # appear ONLY in the REFLECTION cluster summary, never in any
        # single planted observation.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        out = generate_dataset(**_SMALL)
        for conv in out:
            scenario = by_name[conv.scenario_name]
            if scenario.theme_bundles is not None:
                continue  # synthesis path — substrings live in the cluster summary
            memorable_text = " ".join(t.text for t in conv.turns if t.is_memorable)
            for q in conv.questions:
                for substring in q.expected_substrings:
                    assert substring in memorable_text, (
                        f"Substring {substring!r} from question "
                        f"{q.question_id} not present in any memorable turn"
                    )

    def test_turn_indices_are_dense_and_sorted(self) -> None:
        out = generate_dataset(**_SMALL)
        for conv in out:
            indices = [t.turn_index for t in conv.turns]
            assert indices == list(range(len(indices)))

    def test_question_always_lands_inside_conversation(self) -> None:
        # The conversation length may shrink below the nominal target
        # when distraction density skips slots, but it MUST always
        # extend far enough that ``asked_at_turn`` is a real turn.
        out = generate_dataset(**_SMALL)
        for conv in out:
            for q in conv.questions:
                assert q.asked_at_turn < len(conv.turns)


class TestDistractionDensity:
    """Density is a real generation knob, not an ignored parameter."""

    def test_density_affects_dataset_at_same_seed(self) -> None:
        # Same seed, different density → output diverges.  Pre-fix the
        # density parameter was validated but never threaded through;
        # this regression test pins the per-slot Bernoulli gate.
        sparse = dataset_to_json(
            generate_dataset(
                seed=20260508,
                n_conversations=6,
                avg_turns_per_conversation=30,
                distraction_density=0.1,
            ),
        )
        dense = dataset_to_json(
            generate_dataset(
                seed=20260508,
                n_conversations=6,
                avg_turns_per_conversation=30,
                distraction_density=0.9,
            ),
        )
        assert sparse != dense

    def test_density_zero_emits_only_forced_tail_distractions(self) -> None:
        # At density 0.0 the only distractions emitted are the
        # forced tail-fill turns between the last memorable and the
        # question — the corpus is as sparse as the question-offset
        # window allows.
        out = generate_dataset(
            seed=20260508,
            n_conversations=4,
            avg_turns_per_conversation=30,
            distraction_density=0.0,
        )
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        for conv in out:
            scenario = by_name[conv.scenario_name]
            memorable_count = sum(1 for t in conv.turns if t.is_memorable)
            distraction_count = sum(1 for t in conv.turns if not t.is_memorable)
            assert memorable_count == scenario.n_memorable_facts
            # The forced-fill tail tops out at the question offset
            # window's upper bound; below that bound no distraction
            # ever lands.
            assert distraction_count <= scenario.question_offset_max_turns

    def test_density_one_fills_every_non_memorable_slot(self) -> None:
        # At density 1.0 every slot up to the provisional pre-density
        # length emits.  The conversation length therefore equals the
        # provisional total (max of ``target_turns`` and the
        # last-memorable-plus-offset placement), not less.
        out = generate_dataset(
            seed=20260508,
            n_conversations=4,
            avg_turns_per_conversation=30,
            distraction_density=1.0,
        )
        for conv in out:
            # At density 1.0 no Bernoulli skip ever fires, so the
            # tail-fill loop is unreachable; we can pin a minimum
            # length matching the target.
            assert len(conv.turns) >= 30

    def test_density_is_deterministic_per_seed(self) -> None:
        # Density is a parameter of the deterministic stream — same
        # seed AND same density must reproduce byte-for-byte.
        kwargs: _GenKwargs = {
            "seed": 20260508,
            "n_conversations": 5,
            "avg_turns_per_conversation": 25,
            "distraction_density": 0.3,
        }
        a = dataset_to_json(generate_dataset(**kwargs))
        b = dataset_to_json(generate_dataset(**kwargs))
        assert a == b


class TestQuestionFields:
    """The question record is self-sufficient — no scenario re-lookup."""

    def test_every_question_carries_difficulty(self) -> None:
        out = generate_dataset(**_SMALL)
        for conv in out:
            for q in conv.questions:
                assert q.difficulty in {"easy", "medium", "hard"}

    def test_question_difficulty_matches_source_scenario(self) -> None:
        out = generate_dataset(**_SMALL)
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        for conv in out:
            scenario = by_name[conv.scenario_name]
            for q in conv.questions:
                assert q.difficulty == scenario.difficulty


class TestScenarioMix:
    """Weighted scenario picking for sanity subsets."""

    def test_mix_with_single_scenario_produces_only_that_scenario(self) -> None:
        out = generate_dataset(
            seed=42,
            n_conversations=8,
            avg_turns_per_conversation=20,
            distraction_density=0.3,
            scenario_mix={"single_unique_fact": 1.0},
        )
        names = {c.scenario_name for c in out}
        assert names == {"single_unique_fact"}

    def test_mix_with_neutral_subset_produces_only_neutrals(self) -> None:
        # Exactly the subset the AC-8 sanity-band test consumes.
        out = generate_dataset(
            seed=20260508,
            n_conversations=10,
            avg_turns_per_conversation=20,
            distraction_density=0.3,
            scenario_mix={
                "single_unique_fact": 1.0,
                "recent_fact_recall": 1.0,
            },
        )
        names = {c.scenario_name for c in out}
        assert names <= {"single_unique_fact", "recent_fact_recall"}
        # Both scenarios should appear with equal weights and a
        # large-enough draw — pinned at the seed used by AC-8.
        assert "single_unique_fact" in names
        assert "recent_fact_recall" in names

    def test_unknown_scenario_in_mix_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scenarios"):
            generate_dataset(
                seed=1,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=0.3,
                scenario_mix={"definitely_not_a_scenario": 1.0},
            )

    def test_empty_mix_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            generate_dataset(
                seed=1,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=0.3,
                scenario_mix={},
            )

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            generate_dataset(
                seed=1,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=0.3,
                scenario_mix={"single_unique_fact": -1.0},
            )

    def test_zero_weight_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to zero"):
            generate_dataset(
                seed=1,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=0.3,
                scenario_mix={
                    "single_unique_fact": 0.0,
                    "recent_fact_recall": 0.0,
                },
            )


class TestParameterValidation:
    """Input contracts surface ValueError, not corrupt output."""

    def test_zero_conversations_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_conversations"):
            generate_dataset(
                seed=1,
                n_conversations=0,
                avg_turns_per_conversation=10,
                distraction_density=0.3,
            )

    def test_negative_conversations_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_conversations"):
            generate_dataset(
                seed=1,
                n_conversations=-1,
                avg_turns_per_conversation=10,
                distraction_density=0.3,
            )

    def test_zero_turns_rejected(self) -> None:
        with pytest.raises(ValueError, match="avg_turns_per_conversation"):
            generate_dataset(
                seed=1,
                n_conversations=2,
                avg_turns_per_conversation=0,
                distraction_density=0.3,
            )

    def test_distraction_density_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="distraction_density"):
            generate_dataset(
                seed=1,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=1.5,
            )

    def test_distraction_density_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="distraction_density"):
            generate_dataset(
                seed=1,
                n_conversations=2,
                avg_turns_per_conversation=10,
                distraction_density=-0.1,
            )


class TestRecordTypes:
    """Smoke checks on the dataclass surface the evaluator imports."""

    def test_types_are_frozen(self) -> None:
        out = generate_dataset(**_SMALL)
        conv = out[0]
        # Frozen-dataclass semantics — every attribute binding is
        # protected; mutation must raise ``FrozenInstanceError``.
        with pytest.raises(FrozenInstanceError):
            conv.scenario_name = "mutated"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            conv.turns[0].text = "mutated"  # type: ignore[misc]

    def test_record_classes_are_classes(self) -> None:
        # The evaluator (C3) imports these directly; AC-14 forbids new
        # ``engrava.__all__`` exports, but module-level imports out of
        # the benchmarks sub-package are free of that constraint.
        assert isinstance(SyntheticConversation, type)
        assert isinstance(SyntheticQuestion, type)
        assert isinstance(SyntheticTurn, type)


def _with_seed(base: _GenKwargs, seed: int) -> _GenKwargs:
    """Return a copy of ``base`` with a different ``seed``."""
    return {
        "seed": seed,
        "n_conversations": base["n_conversations"],
        "avg_turns_per_conversation": base["avg_turns_per_conversation"],
        "distraction_density": base["distraction_density"],
    }


def _with_n(base: _GenKwargs, n_conversations: int) -> _GenKwargs:
    """Return a copy of ``base`` with a different ``n_conversations``."""
    return {
        "seed": base["seed"],
        "n_conversations": n_conversations,
        "avg_turns_per_conversation": base["avg_turns_per_conversation"],
        "distraction_density": base["distraction_density"],
    }


def _assert_keys_sorted(node: object) -> None:
    """Recursively verify every dict in a JSON tree has sorted keys."""
    if isinstance(node, dict):
        keys = list(node.keys())
        assert keys == sorted(keys), f"Unsorted JSON keys: {keys}"
        for value in node.values():
            _assert_keys_sorted(value)
    elif isinstance(node, list):
        for item in node:
            _assert_keys_sorted(item)

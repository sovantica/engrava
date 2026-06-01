"""Pre-registration tests for the synthetic scenario library.

These tests pin the *shape* of ``SCENARIO_LIBRARY`` so the
anti-cherry-pick scenarios (``single_unique_fact``,
``recent_fact_recall``) cannot be removed or quietly re-labelled
between releases.  The scenario library MUST contain those
two neutrals and the library size MUST be 7 at v0.3.0; the
evaluator's sanity-band assertion is meaningless without the
library shape these tests enforce.
"""

from __future__ import annotations

import re

import pytest

from engrava.benchmarks.synthetic.scenarios import (
    SCENARIO_LIBRARY,
    Scenario,
)

# Pre-registered v0.3.0 library — count and identifying names.  Bumping the
# count requires a new WS that explicitly documents the new scenarios in
# CHANGELOG; this test catches accidental edits.
_EXPECTED_LIBRARY_SIZE = 9
_REQUIRED_NEUTRAL_NAMES = frozenset({"single_unique_fact", "recent_fact_recall"})
_REQUIRED_NEUTRAL_OR_MINOR_NAMES = frozenset(
    {
        "long_recall_simple",
        "multi_fact_recall",
        "contradiction_resolution",
        "distraction_heavy",
    },
)
_REQUIRED_GAIN_NAMES = frozenset(
    {
        "thematic_cluster",
        "abstract_theme_recall",
        "repeated_paraphrase_compression",
    },
)
_ALL_REQUIRED_NAMES = (
    _REQUIRED_NEUTRAL_NAMES | _REQUIRED_NEUTRAL_OR_MINOR_NAMES | _REQUIRED_GAIN_NAMES
)


class TestScenarioLibraryShape:
    """Library-level invariants enforced via git history."""

    def test_library_size_is_nine(self) -> None:
        assert len(SCENARIO_LIBRARY) == _EXPECTED_LIBRARY_SIZE

    def test_required_neutral_scenarios_present(self) -> None:
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        assert _REQUIRED_NEUTRAL_NAMES.issubset(by_name.keys())

    def test_required_neutral_or_minor_scenarios_present(self) -> None:
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        assert _REQUIRED_NEUTRAL_OR_MINOR_NAMES.issubset(by_name.keys())

    def test_required_gain_scenarios_present(self) -> None:
        # Synthesis subset — answer carried by REFLECTION cluster summary.
        # If any of these go missing the AC-9a binding ≥5pp gain band
        # cannot be enforced and the benchmark loses its primary signal.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        assert _REQUIRED_GAIN_NAMES.issubset(by_name.keys())

    def test_no_unexpected_scenarios(self) -> None:
        # Defence-in-depth: if a future change adds a scenario without
        # bumping ``_EXPECTED_LIBRARY_SIZE`` the count test fails first;
        # this assertion fails second with a clearer message.
        actual = {s.name: s for s in SCENARIO_LIBRARY}.keys()
        assert set(actual) == _ALL_REQUIRED_NAMES, (
            f"Scenario library drift: unexpected={set(actual) - _ALL_REQUIRED_NAMES}, "
            f"missing={_ALL_REQUIRED_NAMES - set(actual)}"
        )

    def test_neutrals_carry_neutral_effect_flag(self) -> None:
        # Anti-cherry-pick guard: the flag is what the evaluator's
        # sanity band keys on; flipping it would silently disable the
        # AC-8 protection.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        for neutral_name in _REQUIRED_NEUTRAL_NAMES:
            scenario = by_name[neutral_name]
            assert scenario.expected_dreaming_effect == "neutral", (
                f"Scenario {neutral_name!r} lost its 'neutral' "
                f"expected_dreaming_effect flag — anti-cherry-pick "
                f"protection compromised."
            )

    def test_neutral_or_minor_scenarios_carry_correct_flag(self) -> None:
        # Direct-retrieval subset (AC-9b) — FTS/vector finds these on
        # their own; dreaming must not degrade them and the flag
        # documents that intent.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        for name in _REQUIRED_NEUTRAL_OR_MINOR_NAMES:
            scenario = by_name[name]
            assert scenario.expected_dreaming_effect == "neutral_or_minor", (
                f"Scenario {name!r} must be flagged as neutral_or_minor "
                f"(direct-retrieval subset, AC-9b ±2pp neutrality)."
            )

    def test_gain_scenarios_carry_correct_flag(self) -> None:
        # Synthesis subset (AC-9a) — answer ONLY in REFLECTION summary.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        for name in _REQUIRED_GAIN_NAMES:
            scenario = by_name[name]
            assert scenario.expected_dreaming_effect == "gain", (
                f"Scenario {name!r} must be flagged as gain (synthesis "
                f"subset, AC-9a ≥5pp binding floor)."
            )

    def test_scenario_names_are_unique(self) -> None:
        names = [s.name for s in SCENARIO_LIBRARY]
        assert len(names) == len(set(names)), f"Duplicate scenario names: {names}"


def _is_direct_path(scenario: Scenario) -> bool:
    """True iff the scenario uses the template-based direct path."""
    return scenario.theme_bundles is None


def _is_synthesis_path(scenario: Scenario) -> bool:
    """True iff the scenario uses the ``ThemeBundle``-based synthesis path."""
    return scenario.theme_bundles is not None


class TestDirectPathCompleteness:
    """Direct-retrieval (template-based) scenarios MUST be runnable."""

    def test_every_direct_scenario_has_at_least_one_memorable_template(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            if _is_direct_path(scenario):
                assert scenario.memorable_templates, (
                    f"Scenario {scenario.name!r} has no memorable_templates"
                )

    def test_every_direct_scenario_has_at_least_one_question_template(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            if _is_direct_path(scenario):
                assert scenario.question_templates, (
                    f"Scenario {scenario.name!r} has no question_templates"
                )

    def test_answer_substring_templates_match_question_count(self) -> None:
        # The generator emits one question per ``question_templates``
        # entry and pulls the matching substring tuple by index; a
        # mismatch would crash at runtime.
        for scenario in SCENARIO_LIBRARY:
            if not _is_direct_path(scenario):
                continue
            assert len(scenario.answer_substring_templates) == len(
                scenario.question_templates,
            ), (
                f"Scenario {scenario.name!r}: answer_substring_templates "
                f"length ({len(scenario.answer_substring_templates)}) "
                f"does not match question_templates length "
                f"({len(scenario.question_templates)})"
            )

    def test_every_scenario_has_distractions(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            assert scenario.distraction_templates, (
                f"Scenario {scenario.name!r} has no distraction_templates"
            )

    def test_direct_n_memorable_facts_matches_template_count(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            if not _is_direct_path(scenario):
                continue
            assert scenario.n_memorable_facts == len(scenario.memorable_templates), (
                f"Scenario {scenario.name!r}: n_memorable_facts="
                f"{scenario.n_memorable_facts} does not match "
                f"memorable_templates count ({len(scenario.memorable_templates)})"
            )

    def test_question_offset_window_is_non_empty(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            assert scenario.question_offset_min_turns <= scenario.question_offset_max_turns, (
                f"Scenario {scenario.name!r}: question_offset window is "
                f"inverted (min={scenario.question_offset_min_turns} > "
                f"max={scenario.question_offset_max_turns})"
            )

    def test_recent_fact_recall_question_offset_is_short(self) -> None:
        # The neutrality contract for recent_fact_recall depends on
        # recency dominating retrieval; if anyone widens the offset
        # window it stops being a recency-bias sanity check.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        scenario = by_name["recent_fact_recall"]
        assert scenario.question_offset_max_turns <= 10, (
            "recent_fact_recall must keep its question close to the "
            "memorable turn — widening the window breaks the recency "
            "neutrality contract."
        )

    def test_slot_placeholders_have_vocabulary_entries(self) -> None:
        # Catch typos like "{colour}" in templates without a "colour"
        # entry in ``slot_vocabulary``; the generator would crash
        # otherwise.
        for scenario in SCENARIO_LIBRARY:
            if not _is_direct_path(scenario):
                continue
            referenced = _extract_slot_names(scenario)
            vocab_keys = set(scenario.slot_vocabulary.keys())
            missing = referenced - vocab_keys
            assert not missing, (
                f"Scenario {scenario.name!r} references slots "
                f"{sorted(missing)} that are not in slot_vocabulary"
            )

    def test_slot_vocabulary_is_non_empty(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            for slot_name, candidates in scenario.slot_vocabulary.items():
                assert candidates, (
                    f"Scenario {scenario.name!r} slot {slot_name!r} has no candidate values"
                )

    def test_slot_vocabulary_is_immutable(self) -> None:
        # Pre-registered library must be effectively immutable; a test
        # or runtime caller that tries to mutate
        # ``SCENARIO_LIBRARY[i].slot_vocabulary`` MUST get a TypeError
        # rather than silently corrupting the corpus on a future run.
        for scenario in SCENARIO_LIBRARY:
            with pytest.raises(TypeError):
                scenario.slot_vocabulary["__injected__"] = ("nope",)  # type: ignore[index]


class TestSynthesisPathCompleteness:
    """Synthesis (``ThemeBundle``-based) scenarios MUST be runnable.

    These contracts pin the invariants the AC-9a binding gain depends
    on: bundles exist, facets are present, the theme name does NOT
    leak into any single facet (the abstract-theme scenario would
    collapse to a direct-retrieval question otherwise), and the
    expected substring is reachable via the REFLECTION cluster
    summary that dreaming materialises.
    """

    def test_synthesis_scenarios_have_bundles(self) -> None:
        synthesis_names = {"abstract_theme_recall", "repeated_paraphrase_compression"}
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        for name in synthesis_names:
            scenario = by_name[name]
            assert scenario.theme_bundles, (
                f"Scenario {name!r} is on the synthesis path but ships no theme_bundles."
            )

    def test_each_bundle_has_at_least_eight_facets(self) -> None:
        # The abstract-theme scenario needs enough facets so the cluster
        # quality gates do not collapse the cluster to a small-noise
        # outlier; 8 is the minimum that survives the cluster quality gates with
        # comfortable margin.
        for scenario in SCENARIO_LIBRARY:
            if scenario.theme_bundles is None:
                continue
            for bundle in scenario.theme_bundles:
                assert len(bundle.facets) >= 8, (
                    f"Scenario {scenario.name!r} bundle "
                    f"{bundle.theme_id!r}: facet count "
                    f"{len(bundle.facets)} is below the minimum 8."
                )

    def test_bundle_facets_are_lexically_distinct(self) -> None:
        # Byte-identical facets collapse to a single cluster member
        # under deduplication and let the duplicate_members gate
        # legitimately reject the cluster — both calibration killers
        # observed during v1.1 attempts.
        for scenario in SCENARIO_LIBRARY:
            if scenario.theme_bundles is None:
                continue
            for bundle in scenario.theme_bundles:
                unique = set(bundle.facets)
                assert len(unique) == len(bundle.facets), (
                    f"Bundle {bundle.theme_id!r} in scenario "
                    f"{scenario.name!r} has duplicate facets."
                )

    def test_abstract_theme_recall_facets_do_not_leak_theme(self) -> None:
        # The whole point of abstract_theme_recall is that the theme
        # name (in expected_substrings) NEVER appears verbatim in any
        # facet — otherwise the question becomes a direct lookup and
        # measurement of dreaming gain is invalidated.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        scenario = by_name["abstract_theme_recall"]
        assert scenario.theme_bundles is not None
        for bundle in scenario.theme_bundles:
            for theme_term in bundle.expected_substrings:
                lowered_term = theme_term.lower()
                for facet in bundle.facets:
                    assert lowered_term not in facet.lower(), (
                        f"Bundle {bundle.theme_id!r}: facet "
                        f"{facet!r} leaks theme term {theme_term!r} — "
                        f"the answer must exist ONLY in the cluster "
                        f"summary, never in a single planted fact."
                    )

    def test_paraphrase_bundle_substring_present_in_at_least_one_facet(self) -> None:
        # repeated_paraphrase_compression deliberately uses lexically
        # diverse paraphrases — over-requiring the substring to appear
        # everywhere would contradict the scenario's whole point.  But
        # the substring MUST appear in at least one facet so the
        # cluster summary has a chance of surfacing it; a substring
        # that appears in zero facets is dead by construction.
        by_name = {s.name: s for s in SCENARIO_LIBRARY}
        scenario = by_name["repeated_paraphrase_compression"]
        assert scenario.theme_bundles is not None
        for bundle in scenario.theme_bundles:
            for substring in bundle.expected_substrings:
                lowered_sub = substring.lower()
                hits = sum(1 for facet in bundle.facets if lowered_sub in facet.lower())
                assert hits >= 1, (
                    f"Bundle {bundle.theme_id!r}: substring "
                    f"{substring!r} appears in zero of "
                    f"{len(bundle.facets)} paraphrases — the cluster "
                    f"summary has no source for the substring."
                )

    def test_bundle_question_text_is_non_empty(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            if scenario.theme_bundles is None:
                continue
            for bundle in scenario.theme_bundles:
                assert bundle.question_text.strip(), (
                    f"Bundle {bundle.theme_id!r} in scenario "
                    f"{scenario.name!r} has empty question_text."
                )

    def test_bundle_expected_substrings_are_non_empty(self) -> None:
        for scenario in SCENARIO_LIBRARY:
            if scenario.theme_bundles is None:
                continue
            for bundle in scenario.theme_bundles:
                assert bundle.expected_substrings, (
                    f"Bundle {bundle.theme_id!r} in scenario "
                    f"{scenario.name!r} has empty expected_substrings."
                )


def _extract_slot_names(scenario: Scenario) -> set[str]:
    """Collect every ``{slot}`` placeholder used in a scenario's templates."""
    names: set[str] = set()
    template_pools = (
        scenario.memorable_templates,
        scenario.question_templates,
        scenario.distraction_templates,
    )
    for pool in template_pools:
        for template in pool:
            names.update(re.findall(r"\{(\w+)\}", template))
    for substring_tuple in scenario.answer_substring_templates:
        for template in substring_tuple:
            names.update(re.findall(r"\{(\w+)\}", template))
    return names

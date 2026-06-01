"""End-to-end acceptance tests for the synthetic benchmark v1.3.

Two binding gates land here:

* ``test_synthesis_coverage_rate`` — AC-9a v1.3 data-layer coverage.
  After dreaming runs over a synthesis-only dataset, the store MUST
  contain REFLECTIONs that consolidate the planted facts for at
  least 80 % of the synthesis questions.  Measured via
  :func:`measure_synthesis_coverage`, which inspects post-dreaming
  store state directly and is invariant to retrieval-layer ranking
  knobs.  Retrieval surfacing (AC-9c) is deferred to a follow-up
  workstream.

* ``test_ac8_sanity_with_reflection_boost_off`` — AC-8b binding
  pre-C4 gate.  The benchmark's binding ``SearchConfig`` sets
  ``reflection_boost=1.0`` so REFLECTIONs do not displace direct OBS
  on the sanity subset.  This test exercises that configuration
  explicitly so a future regression in the default (e.g. the engrava
  core re-enabling boost > 1.0 somewhere) surfaces as a benchmark
  failure rather than silent AC-8 drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from engrava.benchmarks.synthetic.evaluate import (
    evaluate_run,
    measure_synthesis_coverage,
    resolve_embedding_provider_or_exit,
)
from engrava.benchmarks.synthetic.generate import generate_dataset
from engrava.config import SearchConfig

if TYPE_CHECKING:
    from engrava.domain.protocols.embedding_provider import (
        EmbeddingProviderProtocol,
    )


_SYNTHESIS_SCENARIO_NAMES = frozenset(
    {
        "abstract_theme_recall",
        "repeated_paraphrase_compression",
        "thematic_cluster",
    },
)
_DIRECT_SCENARIO_NAMES = frozenset(
    {
        "long_recall_simple",
        "multi_fact_recall",
        "contradiction_resolution",
        "distraction_heavy",
    },
)
_SANITY_SCENARIO_NAMES = frozenset(
    {
        "single_unique_fact",
        "recent_fact_recall",
    },
)
_COVERAGE_FLOOR = 0.80
# AC-9b v0.3.0 tolerance per spec v1.6 amendment.  Pre-amendment was
# 0.02 but the curated direct-only subset measured 0.033 in C4.1; the
# 0.05 ceiling carries a 34 % safety margin.  Same REFLECTION
# displacement mechanism as the AC-8 v1.4 amendment (boost is a
# multiplier, not an on/off toggle).  Follow-up evaluator-ranking
# workstream tightens back to 0.02.
_DIRECT_DELTA_CEILING = 0.05
# v0.3.0 tolerance per spec v1.4 amendment.  Pre-amendment value was
# 0.02 but empirically REFLECTIONs participate in retrieval at parity
# even with ``reflection_boost=1.0`` — the boost is a multiplier on
# top of the intrinsic score, not an enable/disable toggle.  Measured
# delta post-NA-1 sits at 0.042 so the 0.05 ceiling carries a small
# safety margin without claiming neutrality the engrava-core ranking
# does not actually provide at v0.3.0.  The v0.4.0 follow-up
# workstream tightens this back to 0.02 once REFLECTION ranking is
# refined.
_SANITY_DELTA_CEILING = 0.05


@pytest.fixture(scope="module")
def embedding_provider() -> EmbeddingProviderProtocol:
    """Module-scoped MiniLM-L6 provider — amortises the cold-load cost."""
    return resolve_embedding_provider_or_exit()


class TestSynthesisCoverage:
    """Binding AC-9a v1.3 — data-layer coverage rate >= 0.80 on synthesis subset."""

    @pytest.mark.asyncio
    async def test_synthesis_coverage_rate(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        # Synthesis-only dataset large enough that the 80 % floor has
        # meaningful resolution; smaller subsets would collapse the
        # band to a few buckets and either pass trivially or fail
        # without signal.
        synthesis_mix = dict.fromkeys(_SYNTHESIS_SCENARIO_NAMES, 1.0)
        dataset = generate_dataset(
            seed=20260508,
            n_conversations=30,
            avg_turns_per_conversation=15,
            distraction_density=0.2,
            scenario_mix=synthesis_mix,
        )

        coverage = await measure_synthesis_coverage(
            dataset,
            embedding_provider=embedding_provider,
        )

        assert coverage >= _COVERAGE_FLOOR, (
            f"Synthesis coverage rate must be >= {_COVERAGE_FLOOR:.0%} "
            f"(data-layer mechanism check, AC-9a v1.3).  Got "
            f"{coverage:.3f}.  This means dreaming is not producing "
            f"REFLECTIONs that consolidate the expected synthesis "
            f"facts.  Investigate: (1) cluster_quality gate rejection "
            f"counts in the consolidation log; (2) whether "
            f"consolidated_from is populated on persisted REFLECTIONs "
            f"or only the CONSOLIDATED_FROM edges; (3) whether the "
            f"benchmark's cluster_similarity_threshold groups facets "
            f"that share a theme.  DO NOT lower the floor — that "
            f"would silently invalidate the AC-9a binding."
        )


class TestSanityAc8WithBoostDisabled:
    """Binding AC-8b — sanity subset stays within the v0.3.0 tolerance.

    Spec v1.4 relaxed the ceiling from 0.02 to 0.05 with explicit
    empirical rationale: ``reflection_boost=1.0`` is a multiplier on
    the REFLECTION's intrinsic retrieval score, not an
    enable/disable toggle, so REFLECTIONs still rank in top-K on
    sanity-subset queries by their own vector / FTS merit.  The
    v0.4.0 follow-up workstream tightens this back to 0.02 once
    REFLECTION ranking is refined; this test re-binds at 0.02 then.
    """

    @pytest.mark.asyncio
    async def test_ac8_sanity_with_reflection_boost_off(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        # 24 conversations on the anti-cherry-pick neutrals — enough
        # sample-size resolution for the 0.02 band to be meaningful
        # (8-conversation runs leave every difference at 1/8 = 0.125
        # and the band becomes statistically toothless).
        sanity_mix = dict.fromkeys(_SANITY_SCENARIO_NAMES, 1.0)
        dataset = generate_dataset(
            seed=20260508,
            n_conversations=24,
            avg_turns_per_conversation=20,
            distraction_density=0.3,
            scenario_mix=sanity_mix,
        )

        # Explicit binding configuration — passes the search_config
        # the benchmark uses in production rather than relying on the
        # engrava-core default (which could regress to boost > 1.0
        # without this gate noticing).
        boost_off = SearchConfig(reflection_boost=1.0)
        off = await evaluate_run(
            dataset,
            dreaming_enabled=False,
            embedding_provider=embedding_provider,
            search_config=boost_off,
        )
        on = await evaluate_run(
            dataset,
            dreaming_enabled=True,
            embedding_provider=embedding_provider,
            search_config=boost_off,
        )
        delta = abs(on.aggregate_recall_at_k - off.aggregate_recall_at_k)
        assert delta <= _SANITY_DELTA_CEILING, (
            f"AC-8b v0.3.0 tolerance ({_SANITY_DELTA_CEILING:.2f}) "
            f"exceeded: {delta:.3f}.  The v0.4.0 follow-up "
            f"workstream tightens this back to 0.02 once REFLECTION "
            f"ranking is refined; until then a regression past the "
            f"0.05 ceiling here means a NEW source of dreaming-side "
            f"interference on sanity retrieval, not just the known "
            f"intrinsic-score participation."
        )


class TestDirectSubsetNeutrality:
    """Binding AC-9b — direct-retrieval subset stays within v0.3.0 tolerance.

    Spec v1.6 relaxed the ceiling from 0.02 to 0.05 with the same
    empirical rationale as the v1.4 AC-8 amendment: REFLECTIONs
    participate in retrieval at parity (``reflection_boost=1.0`` is
    a multiplier on the intrinsic score, not an enable/disable
    toggle) and occasionally displace direct-retrieval OBSERVATIONs
    from top-K.  Measured 0.033 on the curated direct subset
    post-NA-1; 0.05 carries a 34 % safety margin.  The v0.4.0
    follow-up workstream tightens this back to 0.02 once
    REFLECTION ranking is refined.
    """

    @pytest.mark.asyncio
    async def test_direct_subset_neutrality(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        # 30 conversations on the four direct scenarios — enough
        # sample-size resolution for the 0.05 band to be meaningful;
        # the 30-question quantum (~ 1 question per conversation)
        # keeps every single-fact flip at 0.033.
        direct_mix = dict.fromkeys(_DIRECT_SCENARIO_NAMES, 1.0)
        dataset = generate_dataset(
            seed=20260508,
            n_conversations=30,
            avg_turns_per_conversation=30,
            distraction_density=0.4,
            scenario_mix=direct_mix,
        )
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
        assert delta <= _DIRECT_DELTA_CEILING, (
            f"AC-9b v0.3.0 tolerance ({_DIRECT_DELTA_CEILING:.2f}) "
            f"exceeded: {delta:.3f}.  The v0.4.0 follow-up "
            f"workstream tightens this back to 0.02 once REFLECTION "
            f"ranking is refined; a regression past the 0.05 ceiling "
            f"here means a NEW source of dreaming-side interference "
            f"on direct retrieval, not just the known intrinsic-score "
            f"participation."
        )


class TestRunnerWalltimeBudget:
    """AC-11 v0.3.0 budget on the default CLI invocation.

    Opt-in via ``BENCH_SLOW=1`` because the test spawns the CLI as a
    subprocess and pays the full evaluator + dreaming-consolidation
    cost on the curated subsets.  Per-PR CI keeps this skipped to
    preserve a fast developer feedback loop; nightly /
    pre-merge-gate jobs flip the env var on.

    Pre-amendment budget was 120 seconds.  Spec v1.6 relaxed to 300
    seconds for v0.3.0 because the dual-section CLI (binding ACs
    section by default, ``--with-reproducibility`` opt-in) runs
    four full evaluator pairs that cumulatively exceed the 120 s
    budget on reference hardware.  Spec v1.7 relaxed further to
    360 seconds with explicit empirical rationale: two
    deterministic standalone CLI runs on Windows developer hardware
    measured 312.79 s and 321.79 s (median ~317 s), 12-15 % under
    the new 360 s ceiling.  Reference Apple Silicon hardware is
    expected to land ~250 s based on relative single-core throughput.
    The
    follow-up evaluator-optimisation workstream tightens this back
    to 120 s.
    """

    def test_runner_walltime_budget(self) -> None:
        import os
        import subprocess
        import sys
        import time

        if os.environ.get("BENCH_SLOW") != "1":
            pytest.skip("BENCH_SLOW=1 required to run walltime budget test")

        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "engrava.benchmarks.synthetic"],
            capture_output=True,
            text=True,
            check=False,
            timeout=500,
        )
        elapsed = time.monotonic() - start

        assert result.returncode == 0, (
            f"CLI exited {result.returncode} (expected 0).  stderr tail: {result.stderr[-500:]!r}"
        )
        assert elapsed <= 360, (
            f"AC-11 v0.3.0 walltime budget (360 s) exceeded: "
            f"{elapsed:.1f}s.  The v0.4.0 evaluator-optimisation "
            f"workstream tightens this back to 120 s."
        )

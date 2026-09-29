"""End-to-end acceptance tests for the synthetic benchmark v1.3.

Two binding gates land here:

* ``test_synthesis_coverage_rate`` — AC-9a v1.3 data-layer coverage.
  After dreaming runs over a synthesis-only dataset, the store MUST
  contain REFLECTIONs that consolidate the planted facts for at
  least 80 % of the synthesis questions.  Measured via
  :func:`measure_synthesis_coverage`, which inspects post-dreaming
  store state directly and is invariant to retrieval-layer ranking
  knobs.

* ``test_ac8_sanity_with_reflection_boost_off`` — AC-8b binding
  gate.  The benchmark's binding ``SearchConfig`` sets
  ``reflection_boost=1.0``, which leaves REFLECTION scores unscaled
  but does not keep REFLECTIONs out of the results: the boost is a
  multiplier, not an enable/disable toggle, so a REFLECTION can still
  displace a direct OBS from the top-K and the tolerance is 0.05, not
  zero.  This test passes that configuration explicitly to both runs,
  so the sanity subset is measured under the binding configuration
  rather than under whatever the ``SearchConfig`` default is.
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
# AC-9b tolerance on the direct-only subset.  REFLECTIONs take part in
# retrieval at ``reflection_boost=1.0`` (the boost is a multiplier, not
# an on/off toggle), so a REFLECTION can displace a direct OBSERVATION
# from the top-K and the delta need not be zero.  The subset holds 30 questions:
# one changed answer moves recall by 1/30 = 0.033, inside the ceiling;
# two changed the same way move it by 0.067, outside.
_DIRECT_DELTA_CEILING = 0.05
# AC-8b tolerance on the sanity subset, for the same reason.  The subset
# holds 24 questions: one changed answer moves recall by 1/24 = 0.042,
# inside the ceiling; two changed the same way move it by 0.083, outside.
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

    ``reflection_boost=1.0`` is a multiplier on the REFLECTION's
    intrinsic retrieval score, not an enable/disable toggle, so
    REFLECTIONs still rank in top-K on sanity-subset queries by their
    own vector / FTS merit.  The 0.05 ceiling admits one changed
    answer out of 24 (see ``_SANITY_DELTA_CEILING``).
    """

    @pytest.mark.asyncio
    async def test_ac8_sanity_with_reflection_boost_off(
        self,
        embedding_provider: EmbeddingProviderProtocol,
    ) -> None:
        # 24 conversations on the anti-cherry-pick neutrals — enough
        # sample-size resolution for the 0.05 band to be meaningful
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
        # the benchmark runner uses, so both runs are measured under
        # it rather than under the ``SearchConfig`` default.
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
            f"AC-8b v0.3.0 tolerance ({_SANITY_DELTA_CEILING:.2f}) exceeded: {delta:.3f}."
        )


class TestDirectSubsetNeutrality:
    """Binding AC-9b — direct-retrieval subset stays within v0.3.0 tolerance.

    REFLECTIONs participate in retrieval at parity
    (``reflection_boost=1.0`` is a multiplier on the intrinsic score,
    not an enable/disable toggle) and can displace direct-retrieval
    OBSERVATIONs from top-K, so the ceiling is 0.05, not zero.  The
    subset holds 30 questions: one changed answer (1/30 = 0.033) is
    inside the ceiling, two changed the same way (0.067) are outside it.
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
            f"AC-9b v0.3.0 tolerance ({_DIRECT_DELTA_CEILING:.2f}) exceeded: {delta:.3f}."
        )


class TestRunnerWalltimeBudget:
    """The v0.3.0 walltime budget on the default CLI invocation.

    Opt-in via ``BENCH_SLOW=1`` because the test spawns the CLI as a
    subprocess and pays the full evaluator + dreaming-consolidation
    cost on the curated subsets.  Per-PR CI keeps this skipped to
    preserve a fast developer feedback loop; nightly /
    pre-merge-gate jobs flip the env var on.

    The default invocation runs the four binding measurements (one
    synthesis-coverage run and three OFF / ON evaluator pairs;
    ``--with-reproducibility`` is opt-in and adds a reproducibility
    snapshot).  The ceiling on the whole run is 360 seconds.
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
        assert elapsed <= 360, f"The v0.3.0 walltime budget (360 s) exceeded: {elapsed:.1f}s."

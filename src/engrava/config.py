"""Configuration loader for engrava.

Provides declarative YAML-based configuration and a convenience factory
method for creating fully wired ``SqliteEngravaCore`` instances.

Usage::

    async with SqliteEngravaCore.from_config("engrava.yaml") as store:
        thought = await store.get_thought("abc")

The manual constructor ``SqliteEngravaCore(db, hooks=...)`` is **not**
affected — callers who manage their own connections continue to work
without changes.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from engrava.domain.protocols.derived_records import DeriveGates
from engrava.domain.protocols.hooks import DefaultEngravaHooks

if TYPE_CHECKING:
    from engrava.domain.manifest import ExtensionManifest
    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol
    from engrava.domain.protocols.hooks import EngravaHooksProtocol

logger = logging.getLogger(__name__)

_DEFAULT_DREAMING_SIGNALS: dict[str, float] = {
    "recency": 0.25,
    "staleness": 0.20,
    "confirmation": 0.20,
    "confidence": 0.15,
    "frequency": 0.20,
    "action_outcome": 0.15,
}
"""Default dreaming signal weights (CoreThoughtRecord fields only).

The six weights intentionally sum to more than 1.0. The scoring path
renormalises over the signals that are *active* for a given run (an inactive
signal — one whose data source is flat across the candidate pool — is dropped
from the denominator), so the configured map is a set of relative priorities,
not a probability distribution. In particular ``action_outcome`` is inactive in
an action-free store, so the remaining five renormalise exactly as before this
signal existed.
"""

_DEFAULT_HYGIENE_SIGNALS: dict[str, float] = {
    "recency": 0.30,
    "frequency": 0.25,
    "confirmation": 0.20,
    "confidence": 0.15,
    "staleness": 0.10,
}
"""Default keep-score signal weights for the Memory Hygiene loop.

These five weights sum to ``1.0`` and are deliberately **distinct** from the
dreaming promotion weights: hygiene carries its own weight vector and threshold
so a change to one loop's tuning never silently perturbs the other, even though
both read the same signal library. The weights are relative priorities, not a
probability distribution — the keep-score path renormalises over the signals
that are *active* for a given run (an inactive signal, one whose data source is
flat across the candidate pool, is dropped from the denominator), mirroring the
active-signal redistribution the dreaming scorer uses. A high keep-score marks a
thought as worth retaining; a low keep-score (times the decay multiplier) is
what drives an archive.
"""


# ------------------------------------------------------------------
# Value objects
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DreamingGates:
    """Gate thresholds for dreaming consolidation.

    Attributes:
        min_confirmations: Minimum confirmation count before promotion.
        min_age_cycles: Minimum age (current_cycle - created_cycle).
        max_promoted_per_run: Cap on promotions per consolidation run.
        allow_zero_confirmation: When ``True``, bypass the
            ``min_confirmations`` gate so that freshly ingested
            thoughts (with zero confirmations) are eligible for
            promotion.  Defaults to ``True`` so that single-write
            batch-ingest scenarios work out of the box.
        min_cluster_size: Minimum number of thoughts in a cluster for
            a REFLECTION to be created.
        cluster_similarity_threshold: Minimum cosine similarity for the
            agglomerative fallback algorithm.
        cluster_algorithm: Clustering algorithm: ``"lpa"`` (Label
            Propagation, default) or ``"agglomerative"`` (cosine-based
            single-linkage fallback for sparse graphs).
        enable_reflections: When ``False``, skip the clustering + REFLECTION
            creation phase entirely.
        cold_start_clustering: When ``True``, the ``"lpa"`` clustering path
            falls back to cosine-similarity agglomerative clustering within
            the same cycle whenever the ASSOCIATED edge graph is empty.
            This lets dreaming form clusters (and therefore REFLECTIONs) on a
            fresh or sparse graph, before any dream edges exist. Defaults to
            ``False`` so the shipped ``"lpa"`` behaviour is unchanged — the
            fallback is strictly opt-in and never alters cluster output when
            edges are present.
        cluster_allowed_types: Thought types eligible to enter the
            agglomerative clustering candidate pool. Defaults to
            ``("OBSERVATION",)`` so REFLECTIONs created in earlier
            cycles are NOT re-clustered into meta-reflections
            (fixes meta-cascade pathology and cuts the O(n²) cost
            of cluster buildup across cycles by ~30-45%). Operators
            who explicitly want meta-consolidation can pass
            e.g. ``("OBSERVATION", "REFLECTION")``.
        clustering_min_new_candidates: Minimum number of new eligible
            candidate thoughts (vs. the previous consolidation run)
            required to trigger the clustering phase. When the delta
            falls below this threshold the clustering and REFLECTION
            creation phases are skipped entirely — signals, promotion,
            and edge creation still execute. Defaults to ``50``.
            Set to ``0`` to disable the early-stop guard (always
            cluster). The guard is bypassed on the first run since
            there is no previous count to compare against.
        max_cluster_size: Maximum number of source thoughts allowed in a
            single cluster before a REFLECTION is created from it.
            Clusters exceeding this limit are rejected entirely — their
            member thoughts remain as ungrouped OBSERVATIONs with no
            REFLECTION above them. This prevents single-link chaining
            pathology where one monstrous cluster (e.g. 5 000+ sources)
            forms and produces an overly generic REFLECTION centroid
            that floods retrieval results.
            Defaults to ``200`` — rejects only extreme outliers while
            leaving typical healthy clusters (2-50 members) intact.
            Set to ``None`` to disable the guard entirely.
        cluster_quality_gating_enabled: Master switch for the
            content-quality gates that run after the size guard and
            before REFLECTION materialisation.  When ``True`` (default)
            the dreaming consolidation loop calls every gate from
            ``engrava.extensions.dreaming_cluster_quality`` on each
            resolved cluster and skips the cluster when any gate fails.
            Flip to ``False`` to recover the pre-gating behaviour for
            ablation testing — every cluster surviving the size guard
            then becomes a REFLECTION as before.
        cluster_quality_persona_threshold: Persona-only-cluster
            threshold passed to ``is_persona_only_cluster``.  A cluster
            is flagged when the fraction of members detected as
            system-side persona descriptions reaches this value;
            defaults to ``0.75``.
        cluster_quality_cohesion_threshold: Cohesion threshold passed
            to ``is_low_cohesion``.  A cluster is flagged when the
            mean pairwise cosine of its member embeddings is strictly
            below this value; defaults to ``0.40`` (calibrated on
            sentence-transformer embeddings whose member cosines
            cluster around 0.3-0.7).
        cluster_quality_external_homogeneity_threshold: External-source
            homogeneity threshold passed to
            ``is_external_source_homogeneous``.  A cluster is flagged
            when the fraction of members with ``is_self != True`` on
            ``metadata.source`` drops below this value; defaults to
            ``0.95`` so the gate stays a belt-and-suspenders pass under
            normal operation (the upstream eligibility filter already
            keeps the pool external).
        cluster_quality_ne_consistency_threshold: Named-entity
            consistency threshold passed to ``has_consistent_entities``.
            A cluster is flagged when the fraction of members sharing a
            named entity with the first member drops below this value;
            defaults to ``0.60``.
        cluster_quality_require_meaningful_keyphrases: When ``True``
            (default) the post-content gate ``has_meaningful_keyphrases``
            rejects a cluster whose ``top_keyphrases`` list is empty or
            entirely composed of generic determiner-noun phrases
            (``"these moments"``, ``"specific projects"``).  Flip to
            ``False`` to skip this gate without disabling the others.

    Examples:
        >>> gates = DreamingGates(min_confirmations=3)
        >>> gates.min_confirmations
        3

    """

    min_confirmations: int = 2
    min_age_cycles: int = 1
    max_promoted_per_run: int = 20
    allow_zero_confirmation: bool = True
    min_cluster_size: int = 3
    cluster_similarity_threshold: float = 0.7
    cluster_algorithm: Literal["lpa", "agglomerative"] = "lpa"
    enable_reflections: bool = True
    cold_start_clustering: bool = False
    cluster_allowed_types: tuple[str, ...] = ("OBSERVATION",)
    clustering_min_new_candidates: int = 50
    max_cluster_size: int | None = 200
    cluster_quality_gating_enabled: bool = True
    cluster_quality_persona_threshold: float = 0.75
    cluster_quality_cohesion_threshold: float = 0.40
    cluster_quality_external_homogeneity_threshold: float = 0.95
    cluster_quality_ne_consistency_threshold: float = 0.60
    cluster_quality_require_meaningful_keyphrases: bool = True

    def __post_init__(self) -> None:
        """Validate field invariants on construction.

        The YAML loader (:func:`_parse_gates`) performs the same range
        check before reaching this dataclass — the duplication is
        intentional so a direct ``DreamingGates(...)`` call from
        Python code raises the same ``ValueError`` that a malformed
        YAML would, instead of silently accepting an out-of-range
        threshold that would later disable a gate (e.g. persona ratio
        ``> 1.0`` is unreachable so the gate never fires).

        Raises:
            ValueError: When ``cluster_similarity_threshold`` or any
                ``cluster_quality_*_threshold`` falls outside
                ``[0.0, 1.0]``.

        """
        for field_name in (
            "cluster_similarity_threshold",
            "cluster_quality_persona_threshold",
            "cluster_quality_cohesion_threshold",
            "cluster_quality_external_homogeneity_threshold",
            "cluster_quality_ne_consistency_threshold",
        ):
            value = getattr(self, field_name)
            # ``bool`` is a subclass of ``int`` in Python; reject it explicitly so
            # ``True``/``False`` cannot impersonate ``1.0``/``0.0`` and silently
            # disable a gate.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"DreamingGates.{field_name} must be a float in [0.0, 1.0]; got {value!r}"
                raise TypeError(msg)
            if not 0.0 <= value <= 1.0:
                msg = f"DreamingGates.{field_name} must be a float in [0.0, 1.0]; got {value!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class EdgeCreationConfig:
    """Configuration for dream-created edges.

    Controls whether dreaming consolidation creates ``ASSOCIATED``
    edges between promoted thoughts and their nearest neighbours.

    Attributes:
        enabled: Whether edge creation is active during consolidation.
        top_k: Maximum neighbours to link per promoted thought.
        min_similarity: Minimum cosine similarity to create an edge.
        edge_weight_factor: Factor applied to similarity for edge weight
            (``edge.weight = edge_weight_factor * similarity``).

    Examples:
        >>> cfg = EdgeCreationConfig(top_k=3)
        >>> cfg.top_k
        3

    """

    enabled: bool = True
    top_k: int = 1
    min_similarity: float = 0.7
    edge_weight_factor: float = 0.5


@dataclass(frozen=True)
class DreamingConfig:
    """Configuration for the dreaming memory-consolidation extension.

    Default signals operate **exclusively** on ``CoreThoughtRecord`` fields.
    Custom signal functions can operate on extended fields via the
    ``DreamingSignalProtocol``.

    Attributes:
        enabled: Whether dreaming consolidation is active.
        schedule_every_n_cycles: Consolidation cadence.
        promote_threshold: Weighted-score cutoff for promotion.
        signals: Mapping of signal name to weight. Relative priorities, not
            a probability distribution — the scoring path renormalises over
            the signals active for each run, so the map need not sum to 1.0
            (the shipped defaults sum to 1.15).
        gates: Gate thresholds for candidate filtering.
        candidates_limit: Maximum thoughts per consolidation pass.
        edges: Configuration for dream-created edges.
        clustering_backend: Similarity-computation backend for
            ``_agglomerative_clusters``. ``"numpy"`` (default) uses
            vectorised matmul for ~1000x speedup over the pure-Python
            loop. ``"python"`` falls back to the legacy
            O(N^2) loop -- useful as a debugging escape hatch if a
            numerical discrepancy is ever suspected.
        top_keyphrases_count: Number of TF-IDF + n-gram phrases to
            embed in the structural REFLECTION content's
            ``top_keyphrases`` field.  Defaults to 3; raise to widen
            the surface area for downstream readers (LLM judges,
            semantic search) at the cost of a few hundred bytes per
            REFLECTION.
        top_member_excerpts_count: Number of cluster members whose
            content prefix is embedded in the REFLECTION content's
            ``member_excerpts`` field.  Defaults to 5; ordering is
            P1-priority-first with recency tie-breaks.
        member_excerpt_max_chars: Hard upper bound on the length of
            every entry in ``member_excerpts`` (including the
            trailing ellipsis when the source content is truncated).
            Defaults to 150 — long enough to fit a substantive
            sentence-aligned fragment after the upstream chunker
            stops cutting overlap mid-word, while keeping the total
            REFLECTION content size well within the 2 KB-per-row
            budget that the downstream consumers count on.  Raise to
            embed more semantic surface; lower to shrink the trace
            footprint on very dense corpora.
        max_p1_fraction: Maximum fraction of total corpus thoughts
            allowed at priority ``P1`` at any point after a
            consolidation run.  Defaults to ``0.05`` (5 %).

            Empirical motivation: a store with dreaming enabled
            accumulated 29.9 % P1 thoughts, giving
            those thoughts a systematic 67 % ranking boost over P2
            entries in hybrid-search fusion — even though the majority
            were structural REFLECTION thoughts whose content did not
            warrant high priority.  Capping at 5 % prevents this bias
            while still allowing a meaningful set of genuinely
            high-signal observations to be surfaced by the ranking
            layer.

            Set to ``1.0`` to disable the cap entirely (legacy
            behaviour).
        promote_targets: Which thought types are eligible for priority
            promotion during a consolidation run.

            * ``"OBS_ONLY"`` (default) — only ``OBSERVATION`` thoughts
              are candidates for ``P1`` promotion; REFLECTION thoughts
              start at ``reflection_default_priority`` and are not
              auto-bumped.
            * ``"REFL_ONLY"`` — only ``REFLECTION`` thoughts are
              eligible.
            * ``"ALL"`` — both types are eligible.
        reflection_default_priority: Priority assigned to newly-created
            ``REFLECTION`` thoughts.  Defaults to ``"P2"``.

            Earlier REFLECTION thoughts were created with
            the highest priority of their cluster members (effectively
            ``P1`` in most stores).  The new default ``P2`` removes the
            automatic ranking bias; reflections that prove valuable can
            still be promoted via the normal promotion pipeline when
            ``promote_targets`` includes ``"REFL_ONLY"`` or ``"ALL"``.
        eligible_perspectives: When set, only thoughts whose
            ``metadata["perspective"]`` falls in this frozenset are
            eligible for promotion and clustering.  Members are
            restricted to the self-anchored enum ``"percept"``,
            ``"utterance"`` and ``"thought"``.  ``None`` (default)
            disables the filter — every thought passes regardless of
            its perspective annotation, preserving backward
            compatibility with stores written before the metadata
            schema landed.  Thoughts that simply omit the
            ``perspective`` key are always eligible (legacy callers
            without structured attributes).
        self_filter_mode: Restricts eligibility based on whether the
            thought was emitted by the agent itself.  ``"any"``
            (default) applies no filter; ``"self_only"`` keeps only
            thoughts whose ``metadata["source"]["is_self"]`` is
            ``True``; ``"external_only"`` keeps only thoughts whose
            ``is_self`` is ``False``.  Thoughts without an
            ``is_self`` annotation remain eligible (legacy
            backward-compat — caller did not classify).
        min_source_confidence: Minimum acceptable value of
            ``metadata["source"]["confidence"]`` for a thought to be
            eligible.  Confidence is ranked ``"low" < "medium" <
            "high"``; the default ``"low"`` accepts every level
            including thoughts with no confidence annotation
            (treated as ``"low"``).  Set to ``"high"`` for strict
            deployments where uncertain attribution must be excluded
            from dreaming.
        excluded_content_types: Negative filter on
            ``metadata["content_type"]`` — any thought whose content
            type matches an entry here is dropped from the dreaming
            pool.  Defaults to ``frozenset({"code"})`` because code
            fragments cluster poorly under cosine similarity.
            Operators can add ``"json"``, ``"binary"`` or domain-
            specific types as needed.  Thoughts without a
            ``content_type`` annotation are unaffected.
        boilerplate_threshold: Cluster-share ratio strictly above
            which a keyphrase is flagged as corpus-wide boilerplate
            and dropped from a REFLECTION's ``top_keyphrases``.  The
            ratio is computed against every cluster in the same
            dreaming run, so the filter learns what counts as
            boilerplate for the live deployment rather than relying
            on a hardcoded blocklist.  Defaults to ``0.30`` (phrases
            present in more than 30 % of clusters are dropped); lower
            for more aggressive filtering, raise to ``1.0`` to
            disable the filter entirely.
        boilerplate_min_corpus_size: Minimum number of clusters
            scanned in a dreaming run before the boilerplate filter
            engages.  Smaller corpora have too little signal for the
            cluster-frequency statistic to be meaningful, so the
            filter is skipped and every keyphrase is preserved.
            Defaults to ``5``.
        boilerplate_min_keyphrases_per_refl: Fallback guard — if the
            boilerplate filter would strip the keyphrase list below
            this size, the raw unfiltered list is kept instead.
            Prevents the filter from emptying ``top_keyphrases`` on
            REFLECTIONs whose every phrase happens to look like
            boilerplate.  Defaults to ``1``.
        eligible_content_types: Optional positive filter on
            ``metadata["content_type"]`` — when set, a thought that
            declares a ``content_type`` is eligible only if that type
            is listed here.  ``None`` (default) disables the positive
            filter, so any content type not in
            ``excluded_content_types`` is accepted.

            Backward-compatibility caveat: thoughts that omit
            ``content_type`` entirely remain eligible regardless of
            this list, matching the missing-key semantics documented
            for the other filter axes — legacy stores without
            structured attributes must not be punished by new
            filters.  Use ``excluded_content_types`` to keep
            specific bad shapes out; if you also need to require an
            explicit ``content_type`` annotation, supply it at ingest
            time rather than expecting this filter to fail-close on
            unannotated records.
        access_tracking_enabled: When ``True`` (default), retrieval paths
            buffer an access event for every thought a caller actually
            retrieves (search / recall / reflection search results and an
            explicit ``get_thought``), and the buffered counts are flushed
            in a single batched update at each consolidation-cycle boundary
            (and on an explicit flush or store close).  This feeds the
            ``frequency`` signal with data.  The read path never issues a
            per-result database write — events accumulate in a bounded
            in-process buffer, so the retrieval hot path stays read-only.
            Set to ``False`` for throughput deployments that do not want the
            buffering overhead; the ``frequency`` signal is then structurally
            flat and its weight redistributes onto the other active signals.
            Access counts are high-volume regenerable telemetry — they are
            **not** written to the hash-chain journal, and a crash before a
            flush undercounts (acceptable; it self-heals as access continues).

    Examples:
        >>> cfg = DreamingConfig(enabled=True, promote_threshold=0.6)
        >>> cfg.promote_threshold
        0.6

    """

    enabled: bool = False
    schedule_every_n_cycles: int = 100
    promote_threshold: float = 0.7
    signals: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_DREAMING_SIGNALS),
    )
    gates: DreamingGates = field(default_factory=DreamingGates)
    candidates_limit: int = 200
    edges: EdgeCreationConfig = field(default_factory=EdgeCreationConfig)
    clustering_backend: Literal["python", "numpy"] = "numpy"
    top_keyphrases_count: int = 3
    top_member_excerpts_count: int = 5
    member_excerpt_max_chars: int = 150
    max_p1_fraction: float = 0.05
    promote_targets: Literal["OBS_ONLY", "REFL_ONLY", "ALL"] = "OBS_ONLY"
    reflection_default_priority: Literal["P1", "P2", "P3"] = "P2"
    eligible_perspectives: frozenset[Literal["percept", "utterance", "thought"]] | None = None
    self_filter_mode: Literal["any", "self_only", "external_only"] = "any"
    min_source_confidence: Literal["high", "medium", "low"] = "low"
    excluded_content_types: frozenset[str] = frozenset({"code"})
    eligible_content_types: frozenset[str] | None = None
    boilerplate_threshold: float = 0.30
    boilerplate_min_corpus_size: int = 5
    boilerplate_min_keyphrases_per_refl: int = 1
    access_tracking_enabled: bool = True

    def __post_init__(self) -> None:  # noqa: C901 -- each branch validates a distinct field; splitting would obscure validation locality
        """Validate field values on construction.

        Raises:
            ValueError: If ``max_p1_fraction`` is outside ``[0.0, 1.0]``,
                ``promote_targets`` is not a recognised literal,
                ``reflection_default_priority`` is not a recognised literal,
                ``self_filter_mode`` is not ``any``/``self_only``/``external_only``,
                ``min_source_confidence`` is not ``high``/``medium``/``low``,
                or ``eligible_perspectives`` contains a value outside the
                ``percept``/``utterance``/``thought`` enum.

        """
        # Signal weights are relative priorities, not a probability
        # distribution: the scoring path renormalises over the signals active
        # for each run (see DreamingExtension._compute_active_weights), so the
        # configured map need not sum to exactly 1.0 — the six shipped defaults
        # deliberately sum to 1.15. The warning therefore flags only a sum far
        # enough from the ~1.0 scale that a typo (a near-zero or an inflated
        # map) is the likely cause, not a legitimately weighted set.
        weight_sum = sum(self.signals.values())
        if not 0.5 <= weight_sum <= 1.5:  # noqa: PLR2004
            logger.warning(
                "Dreaming signal weights sum to %.3f (expected roughly 1.0); "
                "scoring may behave unexpectedly",
                weight_sum,
            )
        if not 0.0 <= self.max_p1_fraction <= 1.0:
            msg = "DreamingConfig.max_p1_fraction must be in [0.0, 1.0]"
            raise ValueError(msg)
        if self.promote_targets not in ("OBS_ONLY", "REFL_ONLY", "ALL"):
            msg = "DreamingConfig.promote_targets must be 'OBS_ONLY', 'REFL_ONLY', or 'ALL'"
            raise ValueError(msg)
        if self.reflection_default_priority not in ("P1", "P2", "P3"):
            msg = "DreamingConfig.reflection_default_priority must be 'P1', 'P2', or 'P3'"
            raise ValueError(msg)
        if self.self_filter_mode not in ("any", "self_only", "external_only"):
            msg = "DreamingConfig.self_filter_mode must be 'any', 'self_only', or 'external_only'"
            raise ValueError(msg)
        if self.min_source_confidence not in ("high", "medium", "low"):
            msg = "DreamingConfig.min_source_confidence must be 'high', 'medium', or 'low'"
            raise ValueError(msg)
        if self.eligible_perspectives is not None:
            allowed_perspectives = {"percept", "utterance", "thought"}
            invalid = set(self.eligible_perspectives) - allowed_perspectives
            if invalid:
                msg = (
                    f"DreamingConfig.eligible_perspectives contains unsupported "
                    f"value(s) {sorted(invalid)!r}; allowed: "
                    f"'percept', 'utterance', 'thought'"
                )
                raise ValueError(msg)
        if not 0.0 <= self.boilerplate_threshold <= 1.0:
            msg = "DreamingConfig.boilerplate_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)
        if self.boilerplate_min_corpus_size < 1:
            msg = "DreamingConfig.boilerplate_min_corpus_size must be >= 1"
            raise ValueError(msg)
        if self.boilerplate_min_keyphrases_per_refl < 0:
            msg = "DreamingConfig.boilerplate_min_keyphrases_per_refl must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class HygienePolicyConfig:
    """Configuration for the deterministic Memory Hygiene forgetting loop.

    Memory Hygiene is the subtractive counterpart to dreaming consolidation: a
    deterministic, no-LLM, **opt-in** pass that archives cold/low-value thoughts
    (and, separately opt-in, garbage-collects them after a restore window). It
    reuses the dreaming signal library to compute a per-thought *keep-score*,
    multiplies it by the ``decay_function`` hook, and archives thoughts whose
    resulting *eviction-score* falls below ``eviction_threshold`` — unless the
    thought is protected.

    The whole capability is **default-OFF** (``enabled=False``): a store that
    never enables it behaves exactly as before on every read/write path. The
    default action is a **reversible archive**; physical deletion (GC) is a
    second, independently opted-in stage (``auto_gc_enabled``) gated behind a
    restore window.

    Every default is chosen to fail *safe* (keep) rather than aggressive
    (evict): a low threshold, top-priority protection on by default, a bounded
    per-run eviction cap, a restore window before any deletion, a dry-run
    preview mode, and a non-finite-decay fallback that can only ever keep.

    Attributes:
        enabled: Whether the hygiene pass runs. Default ``False`` — the whole
            capability is inert until explicitly turned on, so an existing store
            is unaffected.
        eviction_threshold: Eviction-score cutoff (in ``[0.0, 1.0]``). A thought
            is a candidate for archival when its ``eviction_score`` (keep-score
            times decay) is strictly below this value and it is not protected.
            Default ``0.20`` — a deliberately *low* bar so only clearly cold and
            low-value thoughts fall beneath it.
        protected_priorities: Priorities that are never auto-archived or
            auto-GC'd regardless of score. Default ``("P1",)`` — the top tier,
            where wrongly forgetting is the high-cost error. Config-tunable; set
            to ``()`` for more aggressive hygiene. This is a *default*, not an
            invariant (pinning is the invariant — see ``pinned`` on the thought
            model).
        signal_weights: Keep-score signal weights (relative priorities, not a
            probability distribution). Defaults to the hygiene weight vector
            (``recency 0.30, frequency 0.25, confirmation 0.20, confidence 0.15,
            staleness 0.10``). The keep-score renormalises over the signals
            *active* for a run, so a partial override merges onto the defaults.
        check_every_n_cycles: Cadence gate for the *convenience* invocation from
            ``consolidate()`` — the pass runs there only when
            ``current_cycle % check_every_n_cycles == 0``. An explicit
            ``run_hygiene`` call bypasses the cadence entirely. Default ``1``
            (every cycle). Must be ``>= 1``.
        max_evictions_per_run: Upper bound on the number of thoughts each stage
            may act on per run (at most this many archived, and at most this
            many GC'd). Bounds blast radius and runtime. When more candidates
            qualify than the cap allows, the selected set is deterministic and
            stable. Default ``100``. Must be ``>= 1``.
        auto_gc_enabled: Whether the second (physical-delete) stage runs.
            Default ``False`` — enabling hygiene must never implicitly enable
            deletion. When ``False`` the pass only ever archives (fully
            reversible).
        gc_min_archive_age_cycles: The restore window, in cycles. A
            hygiene-archived thought is GC-eligible only once
            ``current_cycle - archived_at_cycle >= gc_min_archive_age_cycles``.
            Computed from the explicit ``archived_at_cycle`` column, so a thought
            archived by another path (TTL / manual, ``archived_at_cycle`` is
            ``None``) is never auto-GC'd. Default ``10``. Must be ``>= 0``.
        dry_run: When ``True`` the pass computes and returns the would-evict set
            (with per-thought eviction reasons) **without mutating anything and
            without journaling** — a safe preview before enabling for real.
            Default ``False``.

    Examples:
        >>> cfg = HygienePolicyConfig(enabled=True, eviction_threshold=0.15)
        >>> cfg.eviction_threshold
        0.15
        >>> HygienePolicyConfig().enabled
        False

    """

    enabled: bool = False
    eviction_threshold: float = 0.20
    protected_priorities: tuple[str, ...] = ("P1",)
    signal_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_HYGIENE_SIGNALS),
    )
    check_every_n_cycles: int = 1
    max_evictions_per_run: int = 100
    auto_gc_enabled: bool = False
    gc_min_archive_age_cycles: int = 10
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate field invariants on construction.

        The YAML loader (:func:`_parse_hygiene`) performs the same checks
        before reaching this dataclass — the duplication is intentional so a
        direct ``HygienePolicyConfig(...)`` call from Python raises the same
        errors a malformed YAML would, instead of silently accepting an
        out-of-range value that would later mis-drive eviction.

        Raises:
            TypeError: When ``eviction_threshold`` is not a real number, or a
                ``protected_priorities`` / ``signal_weights`` entry has the
                wrong type.
            ValueError: When ``eviction_threshold`` is outside ``[0.0, 1.0]``,
                ``check_every_n_cycles`` or ``max_evictions_per_run`` is ``< 1``,
                or ``gc_min_archive_age_cycles`` is ``< 0``.

        """
        # ``bool`` is an ``int`` subclass; reject it explicitly so ``True`` /
        # ``False`` cannot impersonate ``1.0`` / ``0.0`` for the threshold.
        if isinstance(self.eviction_threshold, bool) or not isinstance(
            self.eviction_threshold,
            (int, float),
        ):
            msg = "HygienePolicyConfig.eviction_threshold must be a float in [0.0, 1.0]"
            raise TypeError(msg)
        if not 0.0 <= self.eviction_threshold <= 1.0:
            msg = "HygienePolicyConfig.eviction_threshold must be a float in [0.0, 1.0]"
            raise ValueError(msg)
        self._validate_collections()
        if self.check_every_n_cycles < 1:
            msg = "HygienePolicyConfig.check_every_n_cycles must be >= 1"
            raise ValueError(msg)
        if self.max_evictions_per_run < 1:
            msg = "HygienePolicyConfig.max_evictions_per_run must be >= 1"
            raise ValueError(msg)
        if self.gc_min_archive_age_cycles < 0:
            msg = "HygienePolicyConfig.gc_min_archive_age_cycles must be >= 0"
            raise ValueError(msg)

    def _validate_collections(self) -> None:
        """Validate the ``protected_priorities`` and ``signal_weights`` entries.

        Split out of :meth:`__post_init__` to keep each method's branch count
        within the linter's complexity budget while preserving the same
        direct-construction guards the YAML loader also applies.

        Raises:
            TypeError: When a ``protected_priorities`` entry is not a string, or
                a ``signal_weights`` key is not a string / value is not numeric.

        """
        for priority in self.protected_priorities:
            if not isinstance(priority, str):
                msg = "HygienePolicyConfig.protected_priorities entries must be strings"
                raise TypeError(msg)
        for name, weight in self.signal_weights.items():
            if not isinstance(name, str):
                msg = "HygienePolicyConfig.signal_weights keys must be strings"
                raise TypeError(msg)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                msg = f"HygienePolicyConfig.signal_weights[{name!r}] must be numeric"
                raise TypeError(msg)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the built-in embedding provider.

    Attributes:
        provider: Provider identifier (``"sentence-transformer"``,
            ``"openai-compatible"``, ``"ollama"``, ``"huggingface"``),
            or ``None`` for no built-in provider.
        model: Model name passed to the provider constructor.
        auto_embed: Automatically embed on ``create_thought``/``update_thought``.
        require_embedding: When ``False`` (default), an auto-embed provider
            failure logs a ``WARNING`` naming the thought and re-raises the
            provider's own exception — byte-identical to the pre-existing
            behaviour. When ``True``, that failure is normalised into a typed
            :class:`~engrava.domain.exceptions.EmbeddingGenerationError`, the
            explicit fail-fast an operator opts into (the thought is still
            persisted, since auto-embed runs after the commit; the error
            surfaces that it is unembedded). Only takes effect when
            ``auto_embed`` is enabled.
        device: Compute device for local providers (``"cpu"``, ``"cuda"``).
        batch_size: Batch encoding size for local providers.
        base_url: Base URL for remote providers.
        api_key: API key for remote providers (supports ``${ENV_VAR}`` syntax).
        query_prefix: Optional instruction prefix prepended to a search query
            before embedding (e.g. ``"query: "`` for an E5 model). Applies
            only to the asymmetric-capable providers (``sentence-transformer``,
            ``ollama``, ``huggingface``); the symmetric ``openai-compatible``
            provider ignores prefixing entirely. Empty/``None`` by default —
            an empty prefix is a literal passthrough, byte-identical to no
            prefixing.
        document_prefix: Optional instruction prefix prepended to a stored
            document before embedding (e.g. ``"passage: "``). Same provider
            scope and passthrough guarantee as ``query_prefix``. Changing this
            on an existing store changes every stored vector and requires a
            deliberate re-embed — the store raises rather than silently
            re-embedding.

    Examples:
        >>> cfg = EmbeddingConfig(provider="sentence-transformer")
        >>> cfg.auto_embed
        False

    """

    provider: str | None = None
    model: str | None = None
    auto_embed: bool = False
    require_embedding: bool = False
    device: str = "cpu"
    batch_size: int = 32
    base_url: str | None = None
    api_key: str | None = None
    query_prefix: str | None = None
    document_prefix: str | None = None


@dataclass(frozen=True)
class SearchConfig:
    """Default weights and parameters for hybrid search.

    These values are used when the caller does not override them in
    the ``search_hybrid()`` call.

    Attributes:
        default_fts_weight: Default FTS5 score weight.
        default_vector_weight: Default vector similarity weight.
        default_recency_weight: Default recency signal weight.
        default_priority_weight: Default priority signal weight.
        default_graph_weight: Default graph-neighbour signal weight.
        recency_half_life: Cycles for the recency score to halve.
        priority_boost_p1: Score multiplier for P1 thoughts.
        priority_boost_p2: Score multiplier for P2 thoughts.
        priority_boost_p3: Score multiplier for P3 thoughts.
        priority_boost_p4: Score multiplier for P4 thoughts.
        graph_edge_decay: Decay factor for 1-hop neighbour boost.
        max_neighbors_per_candidate: Safety cap on neighbours per candidate.
        reflection_boost: Score multiplier applied to REFLECTION thoughts
            retrieved by ``search_hybrid()``.
        graph_expansion_enabled: When ``True``, candidate pool is expanded
            by traversing ``CONSOLIDATED_FROM`` edges from top-N REFLECTIONs
            in the result set. Pulled source OBSERVATIONs receive a
            propagated score ``parent_score * propagation_factor * edge_weight``

        graph_expansion_top_n: Number of top-ranked REFLECTIONs to use as
            expansion seeds per query. Defaults to ``5``.
        graph_expansion_propagation_factor: Multiplier applied to the parent
            REFLECTION score when computing the propagated OBS score.
            Defaults to ``0.7`` (below 1.0 — ensures sources never outrank
            the REFLECTION itself).
        graph_expansion_max_sources_per_reflection: Safety cap on the number
            of source OBSERVATIONs pulled per REFLECTION. Sources are ordered
            by descending ``edge.weight``; only the top N are added.
            Defaults to ``20``.
        graph_expansion_reflection_source_ceiling: REFLECTIONs with more
            than this many ``CONSOLIDATED_FROM`` sources are skipped during
            expansion. Guards against single-link chaining pathology where
            a giant cluster (e.g. 5 000+ sources) causes random noise to
            flood the candidate pool. Defaults to ``50``. Works in concert
            with ``DreamingGates.max_cluster_size``.
        reflection_topk_cap: Maximum fraction of the final top-K result
            that may be occupied by REFLECTION thoughts. After all signals
            are applied, if the number of REFLECTIONs in the top-K window
            exceeds ``top_k * reflection_topk_cap``, the lowest-scoring
            excess REFLECTIONs are evicted and replaced by the highest-
            scoring off-list non-REFLECTION candidates. Set to ``1.0`` to
            disable the cap. Defaults to ``0.3``.
        collapse_pool_factor: Bounded multiplier applied to each arm's
            candidate budget (``fts_top_k`` / ``vector_top_k``) **only when**
            a ``collapse_key`` is passed to ``search_hybrid()`` / ``recall()``.
            De-fragmentation backfill can only draw from candidates the arms
            produced; under heavy same-unit fragmentation the per-arm budgets
            may be dominated by fragments of few units, leaving fewer than
            ``top_k`` distinct units after collapse. Widening each arm by this
            small, bounded factor gives collapse a deeper pool to backfill
            distinct units from — never an unbounded over-fetch. Has no effect
            on the ``collapse_key=None`` path. Must be ``>= 1``. Defaults
            to ``4``.
        vec0_overfetch_factor: Bounded multiplier applied to ``top_k`` when the
            sqlite-vec (``vec0``) vector backend serves ``search_similar()``.
            ``vec0`` applies its ``k``/``LIMIT`` before expired thoughts and
            retired REFLECTIONs can be filtered out, so fetching exactly
            ``top_k`` would under-fill whenever nearby neighbours are non-live.
            The arm over-fetches ``top_k * vec0_overfetch_factor`` (capped by an
            absolute internal bound), applies the live-row filter, then trims to
            ``top_k`` — giving the filter a deeper pool to survive from. This is
            best-effort: an extreme store where nearly all nearest neighbours are
            non-live can still under-fill. No effect on the numpy backend, which
            filters eligibility before top-k. Must be ``>= 1``. Defaults to ``4``.

    Examples:
        >>> cfg = SearchConfig()
        >>> total = (
        ...     cfg.default_fts_weight
        ...     + cfg.default_vector_weight
        ...     + cfg.default_recency_weight
        ...     + cfg.default_priority_weight
        ...     + cfg.default_graph_weight
        ... )
        >>> total
            1.0

    """

    default_fts_weight: float = 0.3
    default_vector_weight: float = 0.55
    default_recency_weight: float = 0.1
    default_priority_weight: float = 0.05
    default_graph_weight: float = 0.0
    recency_half_life: int = 50
    priority_boost_p1: float = 1.0
    priority_boost_p2: float = 0.6
    priority_boost_p3: float = 0.3
    priority_boost_p4: float = 0.0
    graph_edge_decay: float = 0.5
    max_neighbors_per_candidate: int = 5
    reflection_boost: float = 1.0
    graph_expansion_enabled: bool = True
    graph_expansion_top_n: int = 5
    graph_expansion_propagation_factor: float = 0.7
    graph_expansion_max_sources_per_reflection: int = 20
    graph_expansion_reflection_source_ceiling: int = 50
    reflection_topk_cap: float = 0.3
    collapse_pool_factor: int = 4
    vec0_overfetch_factor: int = 4


@dataclass(frozen=True)
class ServiceConfig:
    """Per-service configuration within a multi-service setup.

    Attributes:
        embeddings: Optional embedding-provider configuration override.

    Examples:
        >>> svc = ServiceConfig(embeddings=EmbeddingConfig(provider="ollama"))
        >>> svc.embeddings.provider
        'ollama'

    """

    embeddings: EmbeddingConfig | None = None


@dataclass(frozen=True)
class ServicesConfig:
    """Multi-service mode configuration.

    Each service gets its own SQLite database file under ``data_dir``,
    with independent schema, embeddings, FTS5 index, and WAL.

    Attributes:
        data_dir: Directory containing per-service ``<name>.db`` files.
        default_service: Name of the default service for CLI commands.
        configs: Per-service configuration overrides keyed by service name.

    Examples:
        >>> cfg = ServicesConfig(data_dir=Path("./data"))
        >>> cfg.default_service
        'main'

    """

    data_dir: Path
    default_service: str = "main"
    configs: dict[str, ServiceConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class JournalConfig:
    """Configuration for the hash-chain audit journal.

    When ``enabled`` is ``True``, every mutation (INSERT/UPDATE/DELETE)
    on thoughts and edges is recorded as a hash-linked journal entry.
    When ``False`` (default), the journal table exists but is never
    written to — zero runtime overhead.

    Attributes:
        enabled: Whether journal recording is active.
        verify_on_open: When ``True``, opening a store via
            ``SqliteEngravaCore.from_config`` re-walks the persisted hash
            chain after the schema is ensured and raises
            ``JournalIntegrityError`` if it does not verify. Independent of
            ``enabled`` — a chain recorded in an earlier session is still
            checked. Default ``False`` (the open path is unchanged, so the
            walk cost — which grows with the journal size — is never paid
            unless opted in).

    Examples:
        >>> cfg = JournalConfig(enabled=True)
        >>> cfg.enabled
        True

    """

    enabled: bool = False
    verify_on_open: bool = False


@dataclass(frozen=True)
class TTLConfig:
    """TTL / auto-expiry configuration.

    Attributes:
        strategy: Cleanup strategy — ``"archive"`` or ``"delete"``.
        check_every_n_operations: Run auto-cleanup every *N* store
            operations (0 = manual only).
        default_ttl_seconds: Default TTL applied to new thoughts that
            have no explicit ``expires_at``.  ``None`` = no default.

    Examples:
        >>> TTLConfig()
        TTLConfig(strategy='archive', check_every_n_operations=0, default_ttl_seconds=None)

    """

    strategy: str = "archive"
    check_every_n_operations: int = 0
    default_ttl_seconds: int | None = None


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for the metrics snapshot API.

    Attributes:
        window_size: Search-latency rolling-window size.
        enabled: When ``False``, ``store.metrics()`` returns a zero-filled
            snapshot without issuing SQL queries.

    """

    window_size: int = 1000
    enabled: bool = True


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for ingest-layer behaviours.

    .. note::
        This config object is **declared by the caller layer**.
        ``engrava`` core itself has no ingest pipeline that reads this
        field — ``SqliteEngravaCore.create_thought`` always defaults to
        ``deduplicate=False`` so existing callers keep their behaviour.
        Ingest-side code that constructs ``CoreThoughtRecord`` instances
        (benchmark adapters, bulk import scripts, downstream products)
        is responsible for reading ``config.ingest.deduplication_enabled``
        and forwarding it as
        ``store.create_thought(record, deduplicate=...)``.

    Attributes:
        deduplication_enabled: When ``True``, ingest pipelines should
            pass ``deduplicate=True`` to
            ``SqliteEngravaCore.create_thought`` so that thoughts with
            identical SHA-256 ``content_hash`` collapse into a single
            record with bumped ``confirmation_count`` instead of
            producing duplicates.  Defaults to ``True`` because
            empirical AMB-PersonaMem benchmarks showed ~38.5% duplicate
            observation thoughts in real ingest traffic; the
            persistence-layer flag stays ``False`` by default for
            Liskov stability, but the recommended config posture is on.
            Set to ``False`` only for use cases where repeated identical
            content is itself meaningful and ``confirmation_count``
            semantics are not acceptable.

    """

    deduplication_enabled: bool = True


@dataclass(frozen=True)
class EngravaConfig:
    """Parsed configuration for ``SqliteEngravaCore``.

    Attributes:
        database_path: Path to the SQLite database file.
        wal_mode: Enable WAL journal mode for concurrent reads.
        hooks_class: Dotted import path to a ``EngravaHooksProtocol`` class.
        vector_backend: ``"numpy"`` (default brute-force) or ``"sqlite-vec"``
            (compact ``vec0`` vector table — faster brute-force KNN, not ANN).
        embedding_dimension: Dimension of embedding vectors (e.g. 384 for MiniLM).
        dreaming: Optional dreaming-consolidation configuration.
        hygiene_policy: Optional Memory Hygiene (deterministic forgetting)
            configuration. ``None`` (default) or ``enabled=False`` leaves every
            existing read/write path unchanged — the forgetting loop never runs.
        embeddings: Optional embedding-provider configuration.
        search: Hybrid search default weights.
        services: Optional multi-service configuration.
        journal: Journal audit-log configuration.
        ttl: TTL / auto-expiry configuration.
        metrics: Basic observability snapshot configuration.
        ingest: Ingest-layer configuration (content-hash deduplication).
        derive: Derived-records extension-seam gates. ``enabled=False``
            (default) leaves every write path byte-identical to a store without
            the seam; enabling it lets a hooks object that implements
            ``DerivedRecordProducerProtocol`` persist derived records after a
            source store.
        extension_manifest_paths: Dotted import paths to
            ``ExtensionManifest`` objects to load on startup.  Each
            entry must use the form ``"module.path:ATTRIBUTE"``.  Loaded
            manifests are passed to ``SqliteEngravaCore`` so that any
            ``schema_migrations`` they declare are applied automatically.
        extension_discover: When ``True``, the ``engrava.extensions``
            entry-point group is scanned for additional manifests at
            startup.  Discovered manifests are appended after any
            explicit ``extension_manifest_paths``.

    Examples:
        >>> cfg = EngravaConfig(database_path=Path("./test.db"))
        >>> cfg.vector_backend
        'numpy'

    """

    database_path: Path
    wal_mode: bool = True
    hooks_class: str | None = None
    vector_backend: str = "numpy"
    embedding_dimension: int = 384
    dreaming: DreamingConfig | None = None
    hygiene_policy: HygienePolicyConfig | None = None
    embeddings: EmbeddingConfig | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    services: ServicesConfig | None = None
    journal: JournalConfig = field(default_factory=JournalConfig)
    ttl: TTLConfig = field(default_factory=TTLConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    derive: DeriveGates = field(default_factory=DeriveGates)
    extension_manifest_paths: list[str] = field(default_factory=list)
    extension_discover: bool = False


# ------------------------------------------------------------------
# YAML loader
# ------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when engrava configuration is invalid.

    Attributes:
        message: Human-readable error description.

    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def load_config(path: str | Path) -> EngravaConfig:
    """Load and validate an ``engrava.yaml`` configuration file.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        A validated ``EngravaConfig`` instance.

    Raises:
        ConfigError: If the file is missing, unparseable, or contains
            invalid values.

    Examples:
        >>> config = load_config("engrava.yaml")  # doctest: +SKIP
        >>> config.database_path
        PosixPath('./engrava.db')

    """
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise ConfigError(msg)

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Failed to parse YAML: {exc}"
        raise ConfigError(msg) from exc

    if not isinstance(raw, dict):
        msg = "Config must be a YAML mapping (dict), got " + type(raw).__name__
        raise ConfigError(msg)

    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> EngravaConfig:
    """Parse a raw YAML dict into a validated ``EngravaConfig``.

    Args:
        raw: Parsed YAML dictionary.

    Returns:
        Validated ``EngravaConfig``.

    Raises:
        ConfigError: On missing or invalid fields.

    """
    db_section = raw.get("database", {})
    if not isinstance(db_section, dict):
        msg = "'database' must be a mapping"
        raise ConfigError(msg)

    db_path = db_section.get("path")
    if not db_path:
        msg = "'database.path' is required"
        raise ConfigError(msg)

    wal_mode = db_section.get("wal_mode", True)
    if not isinstance(wal_mode, bool):
        msg = "'database.wal_mode' must be a boolean"
        raise ConfigError(msg)

    # Extensions section
    ext_section = raw.get("extensions", {})
    if not isinstance(ext_section, dict):
        msg = "'extensions' must be a mapping"
        raise ConfigError(msg)

    vector_cfg = ext_section.get("vector", {})
    vector_backend = "numpy"
    embedding_dimension = 384
    if isinstance(vector_cfg, dict):
        vector_backend = vector_cfg.get("backend", "numpy")
        raw_dim = vector_cfg.get("dimension", 384)
        if not isinstance(raw_dim, int) or raw_dim < 1:
            msg = f"'extensions.vector.dimension' must be a positive integer, got {raw_dim!r}"
            raise ConfigError(msg)
        embedding_dimension = raw_dim
    if vector_backend not in {"numpy", "sqlite-vec"}:
        msg = f"'extensions.vector.backend' must be 'numpy' or 'sqlite-vec', got {vector_backend!r}"
        raise ConfigError(msg)

    dreaming_cfg = _parse_dreaming(ext_section.get("dreaming"))

    # Memory Hygiene (deterministic forgetting) section.
    hygiene_cfg = _parse_hygiene(raw.get("hygiene_policy"))

    # Embeddings section
    embeddings_cfg = _parse_embeddings(raw.get("embeddings"))

    # Search section
    search_cfg = _parse_search(raw.get("search"))

    # Hooks section
    hooks_section = raw.get("hooks", {})
    hooks_class: str | None = None
    if isinstance(hooks_section, dict):
        hooks_class = hooks_section.get("class")

    # Services section
    services_cfg = _parse_services(raw.get("services"))

    # Journal section
    journal_cfg = _parse_journal(raw.get("journal"))

    # TTL section
    ttl_cfg = _parse_ttl(raw.get("ttl"))

    # Metrics section
    metrics_cfg = _parse_metrics(raw.get("metrics"))

    # Ingest section (content-hash deduplication)
    ingest_cfg = _parse_ingest(raw.get("ingest"))

    # Derived-records extension-seam section
    derive_cfg = _parse_derive(raw.get("derive"))

    # Manifests section
    ext_manifest_paths, ext_discover = _parse_manifests(raw.get("manifests"))

    return EngravaConfig(
        database_path=Path(db_path),
        wal_mode=wal_mode,
        hooks_class=hooks_class,
        vector_backend=vector_backend,
        embedding_dimension=embedding_dimension,
        dreaming=dreaming_cfg,
        hygiene_policy=hygiene_cfg,
        embeddings=embeddings_cfg,
        search=search_cfg,
        services=services_cfg,
        journal=journal_cfg,
        ttl=ttl_cfg,
        metrics=metrics_cfg,
        ingest=ingest_cfg,
        derive=derive_cfg,
        extension_manifest_paths=ext_manifest_paths,
        extension_discover=ext_discover,
    )


# ------------------------------------------------------------------
# Metrics config parser
# ------------------------------------------------------------------


def _parse_metrics(raw: Any) -> MetricsConfig:  # noqa: ANN401
    """Parse the ``metrics:`` YAML section."""
    if raw is None:
        return MetricsConfig()
    if not isinstance(raw, dict):
        msg = "'metrics' must be a mapping"
        raise ConfigError(msg)

    window_size = raw.get("window_size", 1000)
    if not isinstance(window_size, int) or window_size < 1:
        msg = "'metrics.window_size' must be a positive integer"
        raise ConfigError(msg)

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        msg = "'metrics.enabled' must be a boolean"
        raise ConfigError(msg)

    return MetricsConfig(window_size=window_size, enabled=enabled)


# ------------------------------------------------------------------
# Ingest config parser (content-hash deduplication)
# ------------------------------------------------------------------


def _parse_ingest(raw: Any) -> IngestConfig:  # noqa: ANN401
    """Parse the ``ingest:`` YAML section (content-hash deduplication)."""
    if raw is None:
        return IngestConfig()
    if not isinstance(raw, dict):
        msg = "'ingest' must be a mapping"
        raise ConfigError(msg)

    deduplication_enabled = raw.get("deduplication_enabled", True)
    if not isinstance(deduplication_enabled, bool):
        msg = "'ingest.deduplication_enabled' must be a boolean"
        raise ConfigError(msg)

    return IngestConfig(deduplication_enabled=deduplication_enabled)


# ------------------------------------------------------------------
# Derived-records extension-seam config parser
# ------------------------------------------------------------------


def _parse_derive(raw: Any) -> DeriveGates:  # noqa: ANN401
    """Parse the ``derive:`` YAML section into :class:`DeriveGates`.

    Args:
        raw: Raw YAML value for the ``derive`` section (dict or None).

    Returns:
        A validated :class:`DeriveGates` (the disabled default when the section
        is absent).

    Raises:
        ConfigError: On invalid field types or values.

    """
    if raw is None:
        return DeriveGates()
    if not isinstance(raw, dict):
        msg = "'derive' must be a mapping"
        raise ConfigError(msg)

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        msg = "'derive.enabled' must be a boolean"
        raise ConfigError(msg)

    on_error = raw.get("on_error", "log")
    if on_error not in ("raise", "log"):
        msg = "'derive.on_error' must be 'raise' or 'log'"
        raise ConfigError(msg)

    max_derived = raw.get("max_derived_per_source", 32)
    if isinstance(max_derived, bool) or not isinstance(max_derived, int) or max_derived < 1:
        msg = "'derive.max_derived_per_source' must be a positive integer"
        raise ConfigError(msg)

    return DeriveGates(
        enabled=enabled,
        on_error=on_error,
        max_derived_per_source=max_derived,
    )


# ------------------------------------------------------------------
# Services config parser
# ------------------------------------------------------------------

_SERVICE_NAME_RE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
"""Compiled pattern for valid service names (lowercase, alphanumeric, hyphens, underscores)."""


def _validate_service_name(name: str) -> None:
    """Validate that a service name is well-formed.

    Args:
        name: Candidate service name.

    Raises:
        ConfigError: If the name does not match the allowed pattern.

    """
    if not _SERVICE_NAME_RE_PATTERN.match(name):
        msg = (
            f"Invalid service name {name!r}: must match {_SERVICE_NAME_RE_PATTERN.pattern} "
            f"(lowercase alphanumeric, hyphens, underscores, max 63 chars)"
        )
        raise ConfigError(msg)


def _parse_service_config(name: str, raw: Any) -> ServiceConfig:  # noqa: ANN401
    """Parse a single per-service configuration block.

    Args:
        name: Service name (for error messages).
        raw: Raw YAML value for the service section (dict or None).

    Returns:
        Parsed ``ServiceConfig``.

    Raises:
        ConfigError: On invalid field types.

    """
    if raw is None:
        return ServiceConfig()
    if not isinstance(raw, dict):
        msg = f"'services.configs.{name}' must be a mapping"
        raise ConfigError(msg)

    embeddings_cfg = _parse_embeddings(raw.get("embeddings"))
    return ServiceConfig(embeddings=embeddings_cfg)


def _parse_services(raw: Any) -> ServicesConfig | None:  # noqa: ANN401
    """Parse the ``services:`` YAML section.

    Args:
        raw: Raw YAML value for the services section (dict or None).

    Returns:
        Parsed ``ServicesConfig``, or ``None`` for single-service mode.

    Raises:
        ConfigError: On invalid field types or values.

    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        msg = "'services' must be a mapping"
        raise ConfigError(msg)

    data_dir = raw.get("data_dir")
    if not data_dir:
        msg = "'services.data_dir' is required"
        raise ConfigError(msg)
    if not isinstance(data_dir, str):
        msg = "'services.data_dir' must be a string"
        raise ConfigError(msg)

    default_service = raw.get("default_service", "main")
    if not isinstance(default_service, str):
        msg = "'services.default_service' must be a string"
        raise ConfigError(msg)
    _validate_service_name(default_service)

    configs_raw = raw.get("configs", {})
    if not isinstance(configs_raw, dict):
        msg = "'services.configs' must be a mapping"
        raise ConfigError(msg)

    configs: dict[str, ServiceConfig] = {}
    for svc_name, svc_raw in configs_raw.items():
        if not isinstance(svc_name, str):
            msg = f"Service name must be a string, got {type(svc_name).__name__}"
            raise ConfigError(msg)
        _validate_service_name(svc_name)
        configs[svc_name] = _parse_service_config(svc_name, svc_raw)

    return ServicesConfig(
        data_dir=Path(data_dir),
        default_service=default_service,
        configs=configs,
    )


def _parse_dreaming(raw: Any) -> DreamingConfig | None:  # noqa: ANN401, C901, PLR0912, PLR0915
    """Parse the ``extensions.dreaming`` section.

    Args:
        raw: Raw YAML value for the dreaming section (dict or None).

    Returns:
        Parsed ``DreamingConfig``, or ``None`` if not specified.

    Raises:
        ConfigError: On invalid field types or values.

    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        msg = "'extensions.dreaming' must be a mapping"
        raise ConfigError(msg)

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        msg = "'extensions.dreaming.enabled' must be a boolean"
        raise ConfigError(msg)

    schedule = raw.get("schedule_every_n_cycles", 100)
    if not isinstance(schedule, int) or schedule < 1:
        msg = "'extensions.dreaming.schedule_every_n_cycles' must be a positive integer"
        raise ConfigError(msg)

    threshold = raw.get("promote_threshold", 0.7)
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        msg = "'extensions.dreaming.promote_threshold' must be a float in [0.0, 1.0]"
        raise ConfigError(msg)

    signals = raw.get("signals")
    if signals is not None:
        if not isinstance(signals, dict):
            msg = "'extensions.dreaming.signals' must be a mapping of name→weight"
            raise ConfigError(msg)
        for name, weight in signals.items():
            if not isinstance(name, str):
                msg = f"Signal name must be a string, got {type(name).__name__}"
                raise ConfigError(msg)
            if not isinstance(weight, (int, float)):
                msg = f"Signal weight for {name!r} must be numeric"
                raise ConfigError(msg)

    gates_raw = raw.get("gates")
    gates = _parse_gates(gates_raw) if gates_raw is not None else DreamingGates()

    edges_raw = raw.get("edges")
    edges = _parse_edge_creation(edges_raw) if edges_raw is not None else EdgeCreationConfig()

    candidates_limit = raw.get("candidates_limit", 200)
    if not isinstance(candidates_limit, int) or candidates_limit < 1:
        msg = "'extensions.dreaming.candidates_limit' must be a positive integer"
        raise ConfigError(msg)

    clustering_backend_raw = raw.get("clustering_backend", "numpy")
    if clustering_backend_raw not in ("python", "numpy"):
        msg = "'extensions.dreaming.clustering_backend' must be 'python' or 'numpy'"
        raise ConfigError(msg)
    clustering_backend: Literal["python", "numpy"] = clustering_backend_raw

    top_keyphrases_count = raw.get("top_keyphrases_count", 3)
    if not isinstance(top_keyphrases_count, int) or top_keyphrases_count < 1:
        msg = "'extensions.dreaming.top_keyphrases_count' must be a positive integer"
        raise ConfigError(msg)

    top_member_excerpts_count = raw.get("top_member_excerpts_count", 5)
    if not isinstance(top_member_excerpts_count, int) or top_member_excerpts_count < 1:
        msg = "'extensions.dreaming.top_member_excerpts_count' must be a positive integer"
        raise ConfigError(msg)

    member_excerpt_max_chars = raw.get("member_excerpt_max_chars", 150)
    if not isinstance(member_excerpt_max_chars, int) or member_excerpt_max_chars < 1:
        msg = "'extensions.dreaming.member_excerpt_max_chars' must be a positive integer"
        raise ConfigError(msg)

    max_p1_fraction = raw.get("max_p1_fraction", 0.05)
    if not isinstance(max_p1_fraction, (int, float)) or not 0.0 <= max_p1_fraction <= 1.0:
        msg = "'extensions.dreaming.max_p1_fraction' must be a float in [0.0, 1.0]"
        raise ConfigError(msg)

    promote_targets_raw = raw.get("promote_targets", "OBS_ONLY")
    if promote_targets_raw not in ("OBS_ONLY", "REFL_ONLY", "ALL"):
        msg = "'extensions.dreaming.promote_targets' must be 'OBS_ONLY', 'REFL_ONLY', or 'ALL'"
        raise ConfigError(msg)
    promote_targets: Literal["OBS_ONLY", "REFL_ONLY", "ALL"] = promote_targets_raw

    reflection_default_priority_raw = raw.get("reflection_default_priority", "P2")
    if reflection_default_priority_raw not in ("P1", "P2", "P3"):
        msg = "'extensions.dreaming.reflection_default_priority' must be 'P1', 'P2', or 'P3'"
        raise ConfigError(msg)
    reflection_default_priority: Literal["P1", "P2", "P3"] = reflection_default_priority_raw

    # A partial ``signals:`` mapping MERGES onto the defaults so overriding one
    # weight does not silently zero the other four. An absent section keeps the
    # full default set.
    merged_signals = dict(_DEFAULT_DREAMING_SIGNALS)
    if signals is not None:
        merged_signals.update(signals)

    eligible_perspectives = _parse_eligible_perspectives(raw.get("eligible_perspectives"))
    self_filter_mode = _parse_self_filter_mode(raw.get("self_filter_mode"))
    min_source_confidence = _parse_min_source_confidence(raw.get("min_source_confidence"))
    excluded_content_types = _parse_content_type_set(
        raw.get("excluded_content_types"),
        "excluded_content_types",
        default=frozenset({"code"}),
    )
    # ``default`` is non-None, so the parser never returns None on this call;
    # narrow for the non-optional ``excluded_content_types`` field.
    if excluded_content_types is None:  # pragma: no cover -- default is non-None
        excluded_content_types = frozenset({"code"})
    eligible_content_types = _parse_content_type_set(
        raw.get("eligible_content_types"),
        "eligible_content_types",
        default=None,
    )
    boilerplate_threshold = _parse_dreaming_unit_float(raw, "boilerplate_threshold", 0.30)
    boilerplate_min_corpus_size = _parse_dreaming_positive_int(
        raw, "boilerplate_min_corpus_size", 5
    )
    boilerplate_min_keyphrases_per_refl = _parse_dreaming_nonneg_int(
        raw, "boilerplate_min_keyphrases_per_refl", 1
    )

    access_tracking_enabled = raw.get("access_tracking_enabled", True)
    if not isinstance(access_tracking_enabled, bool):
        msg = "'extensions.dreaming.access_tracking_enabled' must be a boolean"
        raise ConfigError(msg)

    return DreamingConfig(
        enabled=enabled,
        schedule_every_n_cycles=schedule,
        promote_threshold=float(threshold),
        signals=merged_signals,
        gates=gates,
        candidates_limit=candidates_limit,
        edges=edges,
        clustering_backend=clustering_backend,
        top_keyphrases_count=top_keyphrases_count,
        top_member_excerpts_count=top_member_excerpts_count,
        member_excerpt_max_chars=member_excerpt_max_chars,
        max_p1_fraction=float(max_p1_fraction),
        promote_targets=promote_targets,
        reflection_default_priority=reflection_default_priority,
        eligible_perspectives=eligible_perspectives,
        self_filter_mode=self_filter_mode,
        min_source_confidence=min_source_confidence,
        excluded_content_types=excluded_content_types,
        eligible_content_types=eligible_content_types,
        boilerplate_threshold=boilerplate_threshold,
        boilerplate_min_corpus_size=boilerplate_min_corpus_size,
        boilerplate_min_keyphrases_per_refl=boilerplate_min_keyphrases_per_refl,
        access_tracking_enabled=access_tracking_enabled,
    )


def _parse_eligible_perspectives(
    raw: object,
) -> frozenset[Literal["percept", "utterance", "thought"]] | None:
    """Parse ``extensions.dreaming.eligible_perspectives``.

    Args:
        raw: Raw YAML value — a list/set of perspective strings, or ``None``.

    Returns:
        Frozenset of validated perspective literals, or ``None`` when the
        filter is disabled (key absent / explicit ``null``).

    Raises:
        ConfigError: When the value is not a list of the allowed literals.

    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple, set, frozenset)):
        msg = "'extensions.dreaming.eligible_perspectives' must be a list of strings"
        raise ConfigError(msg)
    allowed = {"percept", "utterance", "thought"}
    values: set[Literal["percept", "utterance", "thought"]] = set()
    for entry in raw:
        if entry not in allowed:
            msg = (
                "'extensions.dreaming.eligible_perspectives' entries must be "
                "'percept', 'utterance', or 'thought'"
            )
            raise ConfigError(msg)
        values.add(entry)
    return frozenset(values)


def _parse_self_filter_mode(raw: object) -> Literal["any", "self_only", "external_only"]:
    """Parse ``extensions.dreaming.self_filter_mode`` (defaults to ``"any"``)."""
    if raw is None:
        return "any"
    if raw not in ("any", "self_only", "external_only"):
        msg = (
            "'extensions.dreaming.self_filter_mode' must be 'any', 'self_only', or 'external_only'"
        )
        raise ConfigError(msg)
    return raw


def _parse_min_source_confidence(raw: object) -> Literal["high", "medium", "low"]:
    """Parse ``extensions.dreaming.min_source_confidence`` (defaults to ``"low"``)."""
    if raw is None:
        return "low"
    if raw not in ("high", "medium", "low"):
        msg = "'extensions.dreaming.min_source_confidence' must be 'high', 'medium', or 'low'"
        raise ConfigError(msg)
    return raw


def _parse_content_type_set(
    raw: object,
    key: str,
    *,
    default: frozenset[str] | None,
) -> frozenset[str] | None:
    """Parse a content-type string set for the dreaming filters.

    Args:
        raw: Raw YAML value — a list of content-type strings, or ``None``.
        key: Field name (for error messages).
        default: Value to return when the key is absent (``None`` or a
            frozenset).  An explicit empty list yields an empty frozenset,
            which is distinct from ``None`` for the positive-filter axis.

    Returns:
        Frozenset of content-type strings, or the default when absent.

    Raises:
        ConfigError: When the value is not a list of strings.

    """
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple, set, frozenset)):
        msg = f"'extensions.dreaming.{key}' must be a list of strings"
        raise ConfigError(msg)
    values: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            msg = f"'extensions.dreaming.{key}' entries must be strings"
            raise ConfigError(msg)
        values.add(entry)
    return frozenset(values)


def _parse_dreaming_unit_float(raw: dict[str, Any], key: str, default: float) -> float:
    """Parse a ``[0.0, 1.0]`` float from the ``dreaming`` mapping."""
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        msg = f"'extensions.dreaming.{key}' must be a float in [0.0, 1.0]"
        raise ConfigError(msg)
    return float(value)


def _parse_dreaming_positive_int(raw: dict[str, Any], key: str, default: int) -> int:
    """Parse a ``>= 1`` integer from the ``dreaming`` mapping."""
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"'extensions.dreaming.{key}' must be a positive integer"
        raise ConfigError(msg)
    return value


def _parse_dreaming_nonneg_int(raw: dict[str, Any], key: str, default: int) -> int:
    """Parse a ``>= 0`` integer from the ``dreaming`` mapping."""
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"'extensions.dreaming.{key}' must be a non-negative integer"
        raise ConfigError(msg)
    return value


def _parse_hygiene(raw: Any) -> HygienePolicyConfig | None:  # noqa: ANN401
    """Parse the ``hygiene_policy:`` YAML section (deterministic forgetting).

    Uses the same defensive machinery as :func:`_parse_dreaming` (bool-vs-number
    guards, range checks, partial-override merge for the weight map), so a
    malformed config is rejected with a typed :class:`ConfigError` at load time
    rather than mis-driving eviction later.

    Args:
        raw: Raw YAML value for the ``hygiene_policy`` section (dict or None).

    Returns:
        Parsed :class:`HygienePolicyConfig`, or ``None`` when the section is
        absent (the forgetting loop is then entirely inert).

    Raises:
        ConfigError: On invalid field types or values.

    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        msg = "'hygiene_policy' must be a mapping"
        raise ConfigError(msg)

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        msg = "'hygiene_policy.enabled' must be a boolean"
        raise ConfigError(msg)

    eviction_threshold = raw.get("eviction_threshold", 0.20)
    if (
        isinstance(eviction_threshold, bool)
        or not isinstance(eviction_threshold, (int, float))
        or not 0.0 <= eviction_threshold <= 1.0
    ):
        msg = "'hygiene_policy.eviction_threshold' must be a float in [0.0, 1.0]"
        raise ConfigError(msg)

    protected_priorities = _parse_protected_priorities(raw.get("protected_priorities"))

    signal_weights = _parse_hygiene_signal_weights(raw.get("signal_weights"))

    check_every_n_cycles = raw.get("check_every_n_cycles", 1)
    if (
        isinstance(check_every_n_cycles, bool)
        or not isinstance(check_every_n_cycles, int)
        or check_every_n_cycles < 1
    ):
        msg = "'hygiene_policy.check_every_n_cycles' must be a positive integer"
        raise ConfigError(msg)

    max_evictions_per_run = raw.get("max_evictions_per_run", 100)
    if (
        isinstance(max_evictions_per_run, bool)
        or not isinstance(max_evictions_per_run, int)
        or max_evictions_per_run < 1
    ):
        msg = "'hygiene_policy.max_evictions_per_run' must be a positive integer"
        raise ConfigError(msg)

    auto_gc_enabled = raw.get("auto_gc_enabled", False)
    if not isinstance(auto_gc_enabled, bool):
        msg = "'hygiene_policy.auto_gc_enabled' must be a boolean"
        raise ConfigError(msg)

    gc_min_archive_age_cycles = raw.get("gc_min_archive_age_cycles", 10)
    if (
        isinstance(gc_min_archive_age_cycles, bool)
        or not isinstance(gc_min_archive_age_cycles, int)
        or gc_min_archive_age_cycles < 0
    ):
        msg = "'hygiene_policy.gc_min_archive_age_cycles' must be a non-negative integer"
        raise ConfigError(msg)

    dry_run = raw.get("dry_run", False)
    if not isinstance(dry_run, bool):
        msg = "'hygiene_policy.dry_run' must be a boolean"
        raise ConfigError(msg)

    return HygienePolicyConfig(
        enabled=enabled,
        eviction_threshold=float(eviction_threshold),
        protected_priorities=protected_priorities,
        signal_weights=signal_weights,
        check_every_n_cycles=check_every_n_cycles,
        max_evictions_per_run=max_evictions_per_run,
        auto_gc_enabled=auto_gc_enabled,
        gc_min_archive_age_cycles=gc_min_archive_age_cycles,
        dry_run=dry_run,
    )


def _parse_protected_priorities(raw: object) -> tuple[str, ...]:
    """Parse ``hygiene_policy.protected_priorities`` (defaults to ``("P1",)``).

    Args:
        raw: Raw YAML value — a list of priority strings, an explicit empty
            list (meaning *no* priority is protected), or ``None`` (absent →
            the ``("P1",)`` default).

    Returns:
        Tuple of validated priority strings.

    Raises:
        ConfigError: When the value is not a list of strings.

    """
    if raw is None:
        return ("P1",)
    if not isinstance(raw, (list, tuple)):
        msg = "'hygiene_policy.protected_priorities' must be a list of priority strings"
        raise ConfigError(msg)
    values: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            msg = "'hygiene_policy.protected_priorities' entries must be strings"
            raise ConfigError(msg)
        values.append(entry)
    return tuple(values)


def _parse_hygiene_signal_weights(raw: object) -> dict[str, float]:
    """Parse ``hygiene_policy.signal_weights`` merging onto the hygiene defaults.

    A partial mapping MERGES onto the defaults so overriding one weight does not
    silently zero the others; an absent section keeps the full default vector.

    Args:
        raw: Raw YAML value — a mapping of signal name to weight, or ``None``.

    Returns:
        The merged weight mapping.

    Raises:
        ConfigError: When the value is not a mapping of string to number.

    """
    merged = dict(_DEFAULT_HYGIENE_SIGNALS)
    if raw is None:
        return merged
    if not isinstance(raw, dict):
        msg = "'hygiene_policy.signal_weights' must be a mapping of name→weight"
        raise ConfigError(msg)
    for name, weight in raw.items():
        if not isinstance(name, str):
            msg = f"Signal name must be a string, got {type(name).__name__}"
            raise ConfigError(msg)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            msg = f"'hygiene_policy.signal_weights[{name!r}]' must be numeric"
            raise ConfigError(msg)
        merged[name] = float(weight)
    return merged


def _parse_edge_creation(raw: object) -> EdgeCreationConfig:
    """Parse the ``extensions.dreaming.edges`` section.

    Args:
        raw: Raw YAML value for the edge-creation section.

    Returns:
        Parsed ``EdgeCreationConfig``.

    Raises:
        ConfigError: On invalid field types or values.

    """
    if not isinstance(raw, dict):
        msg = "'extensions.dreaming.edges' must be a mapping"
        raise ConfigError(msg)

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        msg = "'edges.enabled' must be a boolean"
        raise ConfigError(msg)

    top_k = raw.get("top_k", 1)
    if not isinstance(top_k, int) or top_k < 1:
        msg = "'edges.top_k' must be a positive integer"
        raise ConfigError(msg)

    min_sim = raw.get("min_similarity", 0.7)
    if not isinstance(min_sim, (int, float)) or not 0.0 <= min_sim <= 1.0:
        msg = "'edges.min_similarity' must be a float in [0.0, 1.0]"
        raise ConfigError(msg)

    weight_factor = raw.get("edge_weight_factor", 0.5)
    if not isinstance(weight_factor, (int, float)) or weight_factor < 0.0:
        msg = "'edges.edge_weight_factor' must be a non-negative number"
        raise ConfigError(msg)

    return EdgeCreationConfig(
        enabled=enabled,
        top_k=top_k,
        min_similarity=float(min_sim),
        edge_weight_factor=float(weight_factor),
    )


def _parse_gates(raw: Any) -> DreamingGates:  # noqa: ANN401, C901, PLR0912, PLR0915
    """Parse the ``extensions.dreaming.gates`` section.

    Args:
        raw: Raw YAML value for the gates section.

    Returns:
        Parsed ``DreamingGates``.

    Raises:
        ConfigError: On invalid field types.

    """
    if not isinstance(raw, dict):
        msg = "'extensions.dreaming.gates' must be a mapping"
        raise ConfigError(msg)

    min_conf = raw.get("min_confirmations", 2)
    if not isinstance(min_conf, int) or min_conf < 0:
        msg = "'gates.min_confirmations' must be a non-negative integer"
        raise ConfigError(msg)

    min_age = raw.get("min_age_cycles", 1)
    if not isinstance(min_age, int) or min_age < 0:
        msg = "'gates.min_age_cycles' must be a non-negative integer"
        raise ConfigError(msg)

    max_promoted = raw.get("max_promoted_per_run", 20)
    if not isinstance(max_promoted, int) or max_promoted < 1:
        msg = "'gates.max_promoted_per_run' must be a positive integer"
        raise ConfigError(msg)

    allow_zero = raw.get("allow_zero_confirmation", True)
    if not isinstance(allow_zero, bool):
        msg = "'gates.allow_zero_confirmation' must be a boolean"
        raise ConfigError(msg)

    min_cluster_size = raw.get("min_cluster_size", 3)
    if not isinstance(min_cluster_size, int) or min_cluster_size < 1:
        msg = "'gates.min_cluster_size' must be a positive integer"
        raise ConfigError(msg)

    cluster_threshold = raw.get("cluster_similarity_threshold", 0.7)
    if not isinstance(cluster_threshold, (int, float)) or not 0.0 <= cluster_threshold <= 1.0:
        msg = "'gates.cluster_similarity_threshold' must be a float in [0.0, 1.0]"
        raise ConfigError(msg)

    cluster_algorithm = raw.get("cluster_algorithm", "lpa")
    if cluster_algorithm not in ("lpa", "agglomerative"):
        msg = "'gates.cluster_algorithm' must be 'lpa' or 'agglomerative'"
        raise ConfigError(msg)

    enable_reflections = raw.get("enable_reflections", True)
    if not isinstance(enable_reflections, bool):
        msg = "'gates.enable_reflections' must be a boolean"
        raise ConfigError(msg)

    cold_start_clustering = raw.get("cold_start_clustering", False)
    if not isinstance(cold_start_clustering, bool):
        msg = "'gates.cold_start_clustering' must be a boolean"
        raise ConfigError(msg)

    _min_cluster_size_bound = 2
    max_cluster_size = raw.get("max_cluster_size", 200)
    if max_cluster_size is not None and (
        not isinstance(max_cluster_size, int) or max_cluster_size < _min_cluster_size_bound
    ):
        msg = "'gates.max_cluster_size' must be None or an integer >= 2"
        raise ConfigError(msg)

    cluster_quality_gating_enabled = raw.get("cluster_quality_gating_enabled", True)
    if not isinstance(cluster_quality_gating_enabled, bool):
        msg = "'gates.cluster_quality_gating_enabled' must be a boolean"
        raise ConfigError(msg)

    cluster_quality_persona_threshold = _parse_unit_float(
        raw, "cluster_quality_persona_threshold", 0.75
    )
    cluster_quality_cohesion_threshold = _parse_unit_float(
        raw, "cluster_quality_cohesion_threshold", 0.40
    )
    cluster_quality_external_homogeneity_threshold = _parse_unit_float(
        raw, "cluster_quality_external_homogeneity_threshold", 0.95
    )
    cluster_quality_ne_consistency_threshold = _parse_unit_float(
        raw, "cluster_quality_ne_consistency_threshold", 0.60
    )

    cluster_quality_require_meaningful_keyphrases = raw.get(
        "cluster_quality_require_meaningful_keyphrases", True
    )
    if not isinstance(cluster_quality_require_meaningful_keyphrases, bool):
        msg = "'gates.cluster_quality_require_meaningful_keyphrases' must be a boolean"
        raise ConfigError(msg)

    cluster_allowed_types_raw = raw.get("cluster_allowed_types", ("OBSERVATION",))
    if not isinstance(cluster_allowed_types_raw, (list, tuple)):
        msg = "'gates.cluster_allowed_types' must be a list of thought-type strings"
        raise ConfigError(msg)
    cluster_allowed_types_list: list[str] = []
    for entry in cluster_allowed_types_raw:
        if not isinstance(entry, str):
            msg = "'gates.cluster_allowed_types' entries must be strings"
            raise ConfigError(msg)
        cluster_allowed_types_list.append(entry)
    if not cluster_allowed_types_list:
        msg = "'gates.cluster_allowed_types' must list at least one thought type"
        raise ConfigError(msg)
    cluster_allowed_types = tuple(cluster_allowed_types_list)

    clustering_min_new_candidates = raw.get("clustering_min_new_candidates", 50)
    if (
        isinstance(clustering_min_new_candidates, bool)
        or not isinstance(clustering_min_new_candidates, int)
        or clustering_min_new_candidates < 0
    ):
        msg = "'gates.clustering_min_new_candidates' must be a non-negative integer"
        raise ConfigError(msg)

    return DreamingGates(
        min_confirmations=min_conf,
        min_age_cycles=min_age,
        max_promoted_per_run=max_promoted,
        allow_zero_confirmation=allow_zero,
        min_cluster_size=min_cluster_size,
        cluster_similarity_threshold=float(cluster_threshold),
        cluster_algorithm=cluster_algorithm,
        enable_reflections=enable_reflections,
        cold_start_clustering=cold_start_clustering,
        max_cluster_size=max_cluster_size,
        cluster_quality_gating_enabled=cluster_quality_gating_enabled,
        cluster_quality_persona_threshold=cluster_quality_persona_threshold,
        cluster_quality_cohesion_threshold=cluster_quality_cohesion_threshold,
        cluster_quality_external_homogeneity_threshold=(
            cluster_quality_external_homogeneity_threshold
        ),
        cluster_quality_ne_consistency_threshold=cluster_quality_ne_consistency_threshold,
        cluster_quality_require_meaningful_keyphrases=(
            cluster_quality_require_meaningful_keyphrases
        ),
        cluster_allowed_types=cluster_allowed_types,
        clustering_min_new_candidates=clustering_min_new_candidates,
    )


def _parse_unit_float(raw: dict[str, Any], key: str, default: float) -> float:
    """Parse a ``[0.0, 1.0]`` threshold from the ``gates`` mapping.

    Args:
        raw: Raw ``gates`` mapping from YAML.
        key: Field name to read.
        default: Default value when the key is absent.

    Returns:
        Validated float value in ``[0.0, 1.0]``.

    Raises:
        ConfigError: When the value is not a number in ``[0.0, 1.0]``.

    """
    value = raw.get(key, default)
    # ``bool`` is a subclass of ``int``; reject it explicitly so YAML ``true``
    # / ``false`` cannot impersonate ``1.0`` / ``0.0`` and silently disable a
    # gate.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        msg = f"'gates.{key}' must be a float in [0.0, 1.0]"
        raise ConfigError(msg)
    return float(value)


# ------------------------------------------------------------------
# Search config parser
# ------------------------------------------------------------------


def _parse_nonneg_float(raw: dict[str, Any], key: str, default: float, section: str) -> float:
    """Extract and validate a non-negative float from a raw YAML dict.

    Args:
        raw: Parsed YAML mapping.
        key: Key to look up.
        default: Default when key is absent.
        section: Config section name for error messages.

    Returns:
        Validated float value.

    Raises:
        ConfigError: If the value is not numeric or is negative.

    """
    val = raw.get(key, default)
    if not isinstance(val, (int, float)) or val < 0.0:
        msg = f"'{section}.{key}' must be a non-negative number"
        raise ConfigError(msg)
    return float(val)


def _parse_positive_int(raw: dict[str, Any], key: str, default: int, section: str) -> int:
    """Extract and validate a positive integer from a raw YAML dict.

    Args:
        raw: Parsed YAML mapping.
        key: Key to look up.
        default: Default when key is absent.
        section: Config section name for error messages.

    Returns:
        Validated integer value (``>= 1``).

    Raises:
        ConfigError: If the value is not an integer or is less than 1.

    """
    val = raw.get(key, default)
    if not isinstance(val, int) or isinstance(val, bool) or val < 1:
        msg = f"'{section}.{key}' must be a positive integer"
        raise ConfigError(msg)
    return val


def _parse_search(raw: Any) -> SearchConfig:  # noqa: ANN401
    """Parse the ``search:`` YAML section.

    Args:
        raw: Raw YAML value for the search section (dict or None).

    Returns:
        Parsed ``SearchConfig`` (defaults when not specified).

    Raises:
        ConfigError: On invalid field types or values.

    """
    if raw is None:
        return SearchConfig()
    if not isinstance(raw, dict):
        msg = "'search' must be a mapping"
        raise ConfigError(msg)

    fts_w = _parse_nonneg_float(raw, "default_fts_weight", 0.3, "search")
    vec_w = _parse_nonneg_float(raw, "default_vector_weight", 0.55, "search")
    rec_w = _parse_nonneg_float(raw, "default_recency_weight", 0.1, "search")
    pri_w = _parse_nonneg_float(raw, "default_priority_weight", 0.05, "search")

    half_life = raw.get("recency_half_life", 50)
    if not isinstance(half_life, int) or half_life < 1:
        msg = "'search.recency_half_life' must be a positive integer"
        raise ConfigError(msg)

    boost_p1 = _parse_nonneg_float(raw, "priority_boost_p1", 1.0, "search")
    boost_p2 = _parse_nonneg_float(raw, "priority_boost_p2", 0.6, "search")
    boost_p3 = _parse_nonneg_float(raw, "priority_boost_p3", 0.3, "search")
    boost_p4 = _parse_nonneg_float(raw, "priority_boost_p4", 0.0, "search")

    graph_w = _parse_nonneg_float(raw, "default_graph_weight", 0.0, "search")
    graph_decay = _parse_nonneg_float(raw, "graph_edge_decay", 0.5, "search")
    max_neighbors = raw.get("max_neighbors_per_candidate", 5)
    if not isinstance(max_neighbors, int) or max_neighbors < 1:
        msg = "'search.max_neighbors_per_candidate' must be a positive integer"
        raise ConfigError(msg)

    reflection_boost = _parse_nonneg_float(raw, "reflection_boost", 1.0, "search")

    reflection_topk_cap = _parse_nonneg_float(raw, "reflection_topk_cap", 0.3, "search")

    expansion_enabled = raw.get("graph_expansion_enabled", True)
    if not isinstance(expansion_enabled, bool):
        msg = "'search.graph_expansion_enabled' must be a boolean"
        raise ConfigError(msg)

    expansion_top_n = raw.get("graph_expansion_top_n", 5)
    if not isinstance(expansion_top_n, int) or expansion_top_n < 1:
        msg = "'search.graph_expansion_top_n' must be a positive integer"
        raise ConfigError(msg)

    expansion_factor = _parse_nonneg_float(raw, "graph_expansion_propagation_factor", 0.7, "search")

    expansion_max_sources = raw.get("graph_expansion_max_sources_per_reflection", 20)
    if not isinstance(expansion_max_sources, int) or expansion_max_sources < 1:
        msg = "'search.graph_expansion_max_sources_per_reflection' must be a positive integer"
        raise ConfigError(msg)

    expansion_ceiling = raw.get("graph_expansion_reflection_source_ceiling", 50)
    if not isinstance(expansion_ceiling, int) or expansion_ceiling < 1:
        msg = "'search.graph_expansion_reflection_source_ceiling' must be a positive integer"
        raise ConfigError(msg)

    collapse_pool_factor = _parse_positive_int(raw, "collapse_pool_factor", 4, "search")
    vec0_overfetch_factor = _parse_positive_int(raw, "vec0_overfetch_factor", 4, "search")

    return SearchConfig(
        default_fts_weight=fts_w,
        default_vector_weight=vec_w,
        default_recency_weight=rec_w,
        default_priority_weight=pri_w,
        default_graph_weight=graph_w,
        recency_half_life=half_life,
        priority_boost_p1=boost_p1,
        priority_boost_p2=boost_p2,
        priority_boost_p3=boost_p3,
        priority_boost_p4=boost_p4,
        graph_edge_decay=graph_decay,
        max_neighbors_per_candidate=max_neighbors,
        reflection_boost=reflection_boost,
        graph_expansion_enabled=expansion_enabled,
        graph_expansion_top_n=expansion_top_n,
        graph_expansion_propagation_factor=expansion_factor,
        graph_expansion_max_sources_per_reflection=expansion_max_sources,
        graph_expansion_reflection_source_ceiling=expansion_ceiling,
        reflection_topk_cap=reflection_topk_cap,
        collapse_pool_factor=collapse_pool_factor,
        vec0_overfetch_factor=vec0_overfetch_factor,
    )


# ------------------------------------------------------------------
# Embeddings config parser
# ------------------------------------------------------------------

_VALID_PROVIDERS = frozenset(
    {
        "sentence-transformer",
        "openai-compatible",
        "ollama",
        "huggingface",
    }
)
"""Valid built-in embedding provider identifiers."""


def _resolve_env_var(value: str) -> str:
    """Resolve ``${ENV_VAR}`` patterns in a string.

    Only a single ``${...}`` wrapping the entire value is supported
    (no inline interpolation).  Returns the value unchanged when it
    does not match the pattern.

    Args:
        value: Raw config string, possibly ``${ENV_VAR}``.

    Returns:
        Resolved value from the environment, or the original string.

    Raises:
        ConfigError: If the referenced environment variable is not set.

    """
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        resolved = os.environ.get(env_name)
        if resolved is None:
            msg = f"Environment variable '{env_name}' is not set (referenced in embeddings config)"
            raise ConfigError(msg)
        return resolved
    return value


def _parse_embeddings(raw: Any) -> EmbeddingConfig | None:  # noqa: ANN401, C901, PLR0912
    """Parse the ``embeddings:`` YAML section.

    Args:
        raw: Raw YAML value for the embeddings section (dict or None).

    Returns:
        Parsed ``EmbeddingConfig``, or ``None`` if not specified.

    Raises:
        ConfigError: On invalid field types or values.

    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        msg = "'embeddings' must be a mapping"
        raise ConfigError(msg)

    provider = raw.get("provider")
    if provider is not None and provider not in _VALID_PROVIDERS:
        msg = (
            f"'embeddings.provider' must be one of {sorted(_VALID_PROVIDERS)} "
            f"or null, got {provider!r}"
        )
        raise ConfigError(msg)

    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        msg = "'embeddings.model' must be a string"
        raise ConfigError(msg)

    auto_embed = raw.get("auto_embed", False)
    if not isinstance(auto_embed, bool):
        msg = "'embeddings.auto_embed' must be a boolean"
        raise ConfigError(msg)

    require_embedding = raw.get("require_embedding", False)
    if not isinstance(require_embedding, bool):
        msg = "'embeddings.require_embedding' must be a boolean"
        raise ConfigError(msg)

    device = raw.get("device", "cpu")
    if not isinstance(device, str):
        msg = "'embeddings.device' must be a string"
        raise ConfigError(msg)

    batch_size = raw.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size < 1:
        msg = "'embeddings.batch_size' must be a positive integer"
        raise ConfigError(msg)

    base_url = raw.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        msg = "'embeddings.base_url' must be a string"
        raise ConfigError(msg)

    api_key_raw = raw.get("api_key")
    api_key: str | None = None
    if api_key_raw is not None:
        if not isinstance(api_key_raw, str):
            msg = "'embeddings.api_key' must be a string"
            raise ConfigError(msg)
        api_key = _resolve_env_var(api_key_raw)

    query_prefix = raw.get("query_prefix")
    if query_prefix is not None and not isinstance(query_prefix, str):
        msg = "'embeddings.query_prefix' must be a string"
        raise ConfigError(msg)

    document_prefix = raw.get("document_prefix")
    if document_prefix is not None and not isinstance(document_prefix, str):
        msg = "'embeddings.document_prefix' must be a string"
        raise ConfigError(msg)

    return EmbeddingConfig(
        provider=provider,
        model=model,
        auto_embed=auto_embed,
        require_embedding=require_embedding,
        device=device,
        batch_size=batch_size,
        base_url=base_url,
        api_key=api_key,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )


def resolve_embedding_provider(
    config: EmbeddingConfig | None,
) -> EmbeddingProviderProtocol | None:
    """Create an embedding provider instance from config.

    Args:
        config: Parsed embedding configuration, or ``None``.

    Returns:
        An ``EmbeddingProviderProtocol`` instance, or ``None``.

    Raises:
        ConfigError: If the provider cannot be instantiated.

    """
    if config is None or config.provider is None:
        return None

    provider_name = config.provider

    if provider_name == "sentence-transformer":
        try:
            from engrava.embeddings.sentence_transformer import (  # noqa: PLC0415
                SentenceTransformerProvider,
            )
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for 'sentence-transformer' provider. "
                "Install with: pip install engrava[embeddings-local]"
            )
            raise ConfigError(msg) from exc
        return SentenceTransformerProvider(
            model_name=config.model or "all-MiniLM-L12-v2",
            device=config.device,
            batch_size=config.batch_size,
            query_prefix=config.query_prefix or "",
            document_prefix=config.document_prefix or "",
        )

    if provider_name == "openai-compatible":
        from engrava.embeddings.openai_compatible import (  # noqa: PLC0415
            OpenAICompatibleProvider,
        )

        return OpenAICompatibleProvider(
            model_name=config.model or "text-embedding-3-small",
            base_url=config.base_url or "https://api.openai.com/v1",
            api_key=config.api_key,
        )

    if provider_name == "ollama":
        from engrava.embeddings.ollama import OllamaProvider  # noqa: PLC0415

        return OllamaProvider(
            model_name=config.model or "nomic-embed-text",
            base_url=config.base_url or "http://localhost:11434",
            query_prefix=config.query_prefix or "",
            document_prefix=config.document_prefix or "",
        )

    if provider_name == "huggingface":
        try:
            from engrava.embeddings.huggingface import (  # noqa: PLC0415
                HuggingFaceProvider,
            )
        except ImportError as exc:
            msg = (
                "huggingface_hub is required for 'huggingface' provider. "
                "Install with: pip install engrava[embeddings-hf]"
            )
            raise ConfigError(msg) from exc
        return HuggingFaceProvider(
            model_name=config.model or "sentence-transformers/all-MiniLM-L12-v2",
            api_key=config.api_key,
            query_prefix=config.query_prefix or "",
            document_prefix=config.document_prefix or "",
        )

    msg = f"Unknown embedding provider: {provider_name!r}"
    raise ConfigError(msg)


# ------------------------------------------------------------------
# Journal config parser
# ------------------------------------------------------------------


def _parse_journal(raw: Any) -> JournalConfig:  # noqa: ANN401
    """Parse the ``journal:`` YAML section.

    Args:
        raw: Raw YAML value for the journal section (dict or None).

    Returns:
        Parsed ``JournalConfig`` (defaults when not specified).

    Raises:
        ConfigError: On invalid field types.

    """
    if raw is None:
        return JournalConfig()
    if not isinstance(raw, dict):
        msg = "'journal' must be a mapping"
        raise ConfigError(msg)

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        msg = "'journal.enabled' must be a boolean"
        raise ConfigError(msg)

    verify_on_open = raw.get("verify_on_open", False)
    if not isinstance(verify_on_open, bool):
        msg = "'journal.verify_on_open' must be a boolean"
        raise ConfigError(msg)

    return JournalConfig(enabled=enabled, verify_on_open=verify_on_open)


# ------------------------------------------------------------------
# TTL config parser
# ------------------------------------------------------------------


def _parse_ttl(raw: Any) -> TTLConfig:  # noqa: ANN401
    """Parse the ``ttl:`` YAML section.

    Args:
        raw: Raw YAML value for the ttl section (dict or None).

    Returns:
        Parsed ``TTLConfig`` (defaults when not specified).

    Raises:
        ConfigError: On invalid field types or values.

    """
    if raw is None:
        return TTLConfig()
    if not isinstance(raw, dict):
        msg = "'ttl' must be a mapping"
        raise ConfigError(msg)

    strategy = raw.get("strategy", "archive")
    if strategy not in {"archive", "delete"}:
        msg = f"'ttl.strategy' must be 'archive' or 'delete', got {strategy!r}"
        raise ConfigError(msg)

    check_every = raw.get("check_every_n_operations", 0)
    if not isinstance(check_every, int) or check_every < 0:
        msg = "'ttl.check_every_n_operations' must be a non-negative integer"
        raise ConfigError(msg)

    default_ttl = raw.get("default_ttl_seconds")
    if default_ttl is not None and (not isinstance(default_ttl, int) or default_ttl < 1):
        msg = "'ttl.default_ttl_seconds' must be a positive integer or null"
        raise ConfigError(msg)

    return TTLConfig(
        strategy=strategy,
        check_every_n_operations=check_every,
        default_ttl_seconds=default_ttl,
    )


# ------------------------------------------------------------------
# Hook resolver
# ------------------------------------------------------------------


def resolve_hooks(hooks_class: str | None) -> EngravaHooksProtocol:
    """Dynamically import and instantiate a hooks class from a dotted path.

    Args:
        hooks_class: Dotted import path (e.g. ``"my_pkg.MyHooks"``),
            or ``None`` for the default no-op hooks.

    Returns:
        An instance of ``EngravaHooksProtocol``.

    Raises:
        ConfigError: If the class cannot be imported or instantiated.

    Examples:
        >>> hooks = resolve_hooks(None)
        >>> isinstance(hooks, DefaultEngravaHooks)
        True

    """
    if hooks_class is None:
        return DefaultEngravaHooks()

    try:
        module_path, class_name = hooks_class.rsplit(".", maxsplit=1)
    except ValueError:
        msg = f"hooks.class must be a dotted path (got {hooks_class!r})"
        raise ConfigError(msg) from None

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        msg = f"Cannot import hooks module {module_path!r}: {exc}"
        raise ConfigError(msg) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        msg = f"Class {class_name!r} not found in module {module_path!r}"
        raise ConfigError(msg)

    try:
        return cls()  # type: ignore[no-any-return]
    except Exception as exc:
        msg = f"Cannot instantiate hooks class {hooks_class!r}: {exc}"
        raise ConfigError(msg) from exc


# ------------------------------------------------------------------
# Manifest config parser + resolver
# ------------------------------------------------------------------

_MANIFEST_PATH_RE_PATTERN = re.compile(r"^[\w.]+:[\w]+$")
"""Pattern for manifest dotted paths: ``module.path:ATTRIBUTE``."""


def _parse_manifests(raw: Any) -> tuple[list[str], bool]:  # noqa: ANN401
    """Parse the ``manifests:`` YAML section.

    Accepted forms::

        # List of dotted paths
        manifests:
          - "my_plugin.manifest:MANIFEST"

        # Auto-discover via entry points
        manifests:
          discover: true

        # Both
        manifests:
          discover: true
          paths:
            - "my_plugin.manifest:MANIFEST"

    Args:
        raw: Raw YAML value for the ``manifests`` key (list, dict, or None).

    Returns:
        Tuple of ``(manifest_paths, discover_flag)``.

    Raises:
        ConfigError: On invalid format or malformed dotted paths.

    """
    if raw is None:
        return [], False

    # Short form: a plain list of dotted paths.
    if isinstance(raw, list):
        paths = _validate_manifest_paths(raw)
        return paths, False

    if not isinstance(raw, dict):
        msg = (
            "'manifests' must be a list of dotted paths or a mapping "
            "with optional 'discover' and 'paths' keys"
        )
        raise ConfigError(msg)

    discover = raw.get("discover", False)
    if not isinstance(discover, bool):
        msg = "'manifests.discover' must be a boolean"
        raise ConfigError(msg)

    raw_paths = raw.get("paths", [])
    if not isinstance(raw_paths, list):
        msg = "'manifests.paths' must be a list of dotted paths"
        raise ConfigError(msg)

    paths = _validate_manifest_paths(raw_paths)
    return paths, discover


def _validate_manifest_paths(raw_paths: list[Any]) -> list[str]:
    """Validate and return a list of manifest dotted-path strings.

    Args:
        raw_paths: Raw list from YAML.

    Returns:
        Validated list of strings matching ``module.path:ATTRIBUTE``.

    Raises:
        ConfigError: If any entry is not a valid dotted path.

    """
    result: list[str] = []
    for entry in raw_paths:
        if not isinstance(entry, str):
            msg = f"Manifest path must be a string, got {type(entry).__name__}"
            raise ConfigError(msg)
        if not _MANIFEST_PATH_RE_PATTERN.match(entry):
            msg = (
                f"Invalid manifest path {entry!r}: must be in the form "
                f"'module.path:ATTRIBUTE' (e.g. 'my_plugin.manifest:MANIFEST')"
            )
            raise ConfigError(msg)
        result.append(entry)
    return result


def resolve_manifests(
    manifest_paths: list[str],
    *,
    discover: bool = False,
) -> list[ExtensionManifest]:
    """Load ``ExtensionManifest`` objects from dotted import paths.

    Optionally appends manifests discovered via the ``engrava.extensions``
    entry-point group when *discover* is ``True``.

    Args:
        manifest_paths: List of ``"module.path:ATTRIBUTE"`` strings.  Each
            attribute must be an ``ExtensionManifest`` instance.
        discover: When ``True``, the ``engrava.extensions`` entry-point
            group is scanned and any found manifests are appended.

    Returns:
        Ordered list of ``ExtensionManifest`` instances.

    Raises:
        ConfigError: If a path cannot be imported or the referenced
            attribute is not an ``ExtensionManifest``.

    Examples:
        >>> resolve_manifests([])
        []

    """
    from engrava.domain.manifest import ExtensionManifest  # noqa: PLC0415

    result: list[ExtensionManifest] = []

    for path in manifest_paths:
        try:
            module_path, attr_name = path.rsplit(":", maxsplit=1)
        except ValueError:
            msg = f"Invalid manifest path {path!r}: missing ':' separator"
            raise ConfigError(msg) from None

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            msg = f"Cannot import manifest module {module_path!r}: {exc}"
            raise ConfigError(msg) from exc

        obj = getattr(module, attr_name, None)
        if obj is None:
            msg = f"Attribute {attr_name!r} not found in module {module_path!r}"
            raise ConfigError(msg)

        if callable(obj) and not isinstance(obj, ExtensionManifest):
            try:
                obj = obj()
            except Exception as exc:
                msg = f"Cannot call manifest factory {path!r}: {exc}"
                raise ConfigError(msg) from exc

        if not isinstance(obj, ExtensionManifest):
            msg = f"Manifest {path!r} is {type(obj).__name__!r}, expected ExtensionManifest"
            raise ConfigError(msg)

        result.append(obj)

    if discover:
        from engrava.extensions.discovery import discover_manifests  # noqa: PLC0415

        result.extend(discover_manifests())

    return result

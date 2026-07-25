"""DreamingExtension — periodic memory consolidation.

A standalone module for promoting short-term thoughts to long-term
memory based on configurable scoring signals and gate thresholds.

**Not** built into core CRUD — the consumer decides when to invoke
``run_consolidation()`` directly or calls ``run_if_due()`` from its cycle loop
to honor ``schedule_every_n_cycles``.

Default signals operate **exclusively** on ``CoreThoughtRecord`` fields.
Custom signals can be provided via ``DreamingSignalProtocol`` to score
extension-specific fields.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from engrava.config import DreamingConfig
from engrava.domain.dreaming import (
    CENTROID_MODEL_NAME,
    DEFAULT_SIGNALS,
    ConsolidationResult,
    DreamingContext,
    DreamingSignalProtocol,
    compute_centroid,
    default_signal_active,
)
from engrava.domain.exceptions import DuplicateEdgeError
from engrava.domain.protocols.dreaming import DreamingStoreProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from engrava.domain.models.thought import ThoughtRecord

logger = logging.getLogger(__name__)


# Row-chunk size for the vectorized agglomerative clustering.
# N <= this threshold -> single-shot N x N matmul; N > threshold -> chunked
# processing in blocks of this size. At 10 000 the peak float32 similarity
# matrix is ~400 MB, well within a workstation budget. Exposed as a module
# constant so tests can monkey-patch it to exercise the chunked path on
# small synthetic inputs.
_VECTORIZED_CLUSTERING_CHUNK_SIZE = 10_000


# ------------------------------------------------------------------
# Extension
# ------------------------------------------------------------------


class DreamingExtension:
    """Periodic memory consolidation with configurable scoring and gates.

    Signals compute per-thought scores; gates enforce minimum thresholds.
    Thoughts that pass gates **and** exceed the ``promote_threshold``
    weighted score are promoted by setting their priority to ``P1``.

    Args:
        config: Parsed ``DreamingConfig`` with weights, gates, and schedule.
        custom_signals: Optional overrides / additions for signal functions.
            Keys matching a default signal name replace the default; new
            keys extend the signal set.

    Raises:
        ValueError: If a signal name from config is unknown and not
            provided in ``custom_signals``.

    Examples:
        >>> from engrava.config import DreamingConfig
        >>> ext = DreamingExtension(config=DreamingConfig(enabled=True))

    """

    def __init__(
        self,
        config: DreamingConfig,
        custom_signals: dict[str, DreamingSignalProtocol] | None = None,
    ) -> None:
        self._config = config
        self._signals = self._build_signal_map(custom_signals or {})
        # Tracks the ACTIVE OBSERVATION candidate count from the last
        # consolidation run that executed clustering.  Used by the
        # early-stop guard to skip clustering when too few
        # new candidates have appeared since the previous run.
        self._last_clustering_candidate_count: int | None = None

    @property
    def config(self) -> DreamingConfig:
        """Return the dreaming configuration.

        Returns:
            The ``DreamingConfig`` instance.

        """
        return self._config

    def is_due(self, current_cycle: int) -> bool:
        """Return whether configured cycle cadence permits consolidation.

        The helper does not own a background scheduler. Consumers call it from
        their own cycle loop, or use :meth:`run_if_due`. Cycle ``0`` is treated
        as initialization rather than the first scheduled run.

        Args:
            current_cycle: Non-negative cognitive cycle number.

        Returns:
            ``True`` when Dreaming is enabled and the cycle is a positive
            multiple of ``schedule_every_n_cycles``.

        Raises:
            ValueError: If ``current_cycle`` is negative or not an integer.

        """
        # A single value-domain contract: the cycle must be a non-negative,
        # non-bool integer. Combining the type and range guards keeps the public
        # ValueError contract (documented above and asserted by the tests) rather
        # than splitting into a TypeError for the type case.
        is_nonneg_int = (
            not isinstance(current_cycle, bool)
            and isinstance(current_cycle, int)
            and current_cycle >= 0
        )
        if not is_nonneg_int:
            msg = "current_cycle must be a non-negative integer"
            raise ValueError(msg)
        return (
            self._config.enabled
            and current_cycle > 0
            and current_cycle % self._config.schedule_every_n_cycles == 0
        )

    async def run_if_due(
        self,
        store: DreamingStoreProtocol,
        current_cycle: int,
    ) -> ConsolidationResult | None:
        """Run consolidation only when the configured cadence is due.

        This is the schedule-aware convenience path. Explicit callers that
        intentionally need an immediate pass can continue to call
        :meth:`run_consolidation`, which remains unconditional.

        Args:
            store: Store to consolidate when the cadence is due.
            current_cycle: Current cognitive cycle number.

        Returns:
            A consolidation result when a pass ran, otherwise ``None``.

        """
        if not self.is_due(current_cycle):
            return None
        return await self.run_consolidation(store, current_cycle)

    def _build_signal_map(
        self,
        custom: dict[str, DreamingSignalProtocol],
    ) -> dict[str, tuple[DreamingSignalProtocol, float]]:
        """Resolve signal names to ``(callable, weight)`` pairs.

        Custom signals override defaults with the same name.  Unknown
        names raise ``ValueError``.

        Args:
            custom: Custom signal overrides keyed by name.

        Returns:
            Mapping of signal name to ``(signal_callable, weight)``.

        Raises:
            ValueError: For unknown signal names not in defaults or custom.

        """
        signal_map: dict[str, tuple[DreamingSignalProtocol, float]] = {}
        for name, weight in self._config.signals.items():
            if name in custom:
                signal_map[name] = (custom[name], weight)
            elif name in DEFAULT_SIGNALS:
                signal_map[name] = (DEFAULT_SIGNALS[name](), weight)
            else:
                msg = (
                    f"Unknown dreaming signal: {name!r}. "
                    f"Available defaults: {sorted(DEFAULT_SIGNALS)}. "
                    f"Register custom signals via custom_signals parameter."
                )
                raise ValueError(msg)
        return signal_map

    async def run_consolidation(
        self,
        store: DreamingStoreProtocol,
        current_cycle: int,
    ) -> ConsolidationResult:
        """Execute one consolidation pass over candidate thoughts.

        Fetches ``ACTIVE`` thoughts, scores each via configured signals,
        applies gate thresholds, and promotes qualifying thoughts by
        setting their priority to ``P1``.  Two additional constraints
        the configured promotion gates are enforced:

        * **P1 fraction cap** — at most ``config.max_p1_fraction`` of
          the total corpus may be ``P1`` after this run.  Once the
          cap is reached, promotion stops regardless of how many
          candidates still qualify by score.
        * **Promote-targets filter** — only thought types listed in
          ``config.promote_targets`` are eligible for promotion.
          Default is ``"OBS_ONLY"`` (OBSERVATION thoughts only).

        Args:
            store: A store implementing the Dreaming capability protocol.
            current_cycle: Current cognitive cycle number.

        Returns:
            ``ConsolidationResult`` with diagnostic counts and promoted IDs.

        """
        candidates = await store.list_thoughts(
            lifecycle_status="ACTIVE",
            limit=self._config.candidates_limit,
        )
        ctx = DreamingContext(
            current_cycle=current_cycle,
            total_thoughts=len(candidates),
        )

        # --- Reachable scoring: active-signal weight redistribution ---
        # Computed ONCE over the candidate pool (pool-relative, not
        # per-thought). Flat signals (no data source this run) contribute 0
        # and their weight redistributes onto the active signals, mirroring
        # the hybrid-search precedent ``_redistribute_hybrid_weights``.
        active_weights, flat_signals = self._compute_active_weights(
            candidates,
            current_cycle=current_cycle,
        )
        if flat_signals:
            logger.info(
                "Dreaming scoring: %d signal(s) structurally flat this run (no data "
                "source) — weight redistributed onto active signals: %s",
                len(flat_signals),
                flat_signals,
            )

        # --- compute available P1 slots under fraction cap ---
        # Population-level cap: ``current_p1_count`` is the existing P1
        # population in the store, so ``available_slots`` already accounts for
        # every P1 promoted by prior runs — repeated cycles cannot push the P1
        # population past ``max_p1_fraction`` of the total.
        total_thoughts = await store.count_thoughts()
        current_p1_count = await store.count_thoughts(priority="P1")
        max_p1_count = max(1, int(total_thoughts * self._config.max_p1_fraction))
        available_slots = max(0, max_p1_count - current_p1_count)

        if available_slots == 0:
            logger.info(
                "dreaming.promote: P1 fraction cap reached (%d/%d = %.1f%%), "
                "no promotion slots available",
                current_p1_count,
                total_thoughts,
                current_p1_count * 100.0 / total_thoughts if total_thoughts else 0.0,
            )

        promoted, scores, skipped_gate, promotion_capped = await self._apply_promotions(
            store=store,
            candidates=candidates,
            ctx=ctx,
            current_cycle=current_cycle,
            available_slots=available_slots,
            promote_type_filter=self._build_promote_type_filter(),
            active_weights=active_weights,
        )

        p1_fraction_after = (
            (current_p1_count + len(promoted)) / total_thoughts if total_thoughts > 0 else 0.0
        )

        # --- Edge creation for promoted thoughts ---
        edges_created = 0
        if promoted and self._config.edges.enabled:
            edges_created = await self._create_edges_for_promoted(
                store=store,
                promoted_ids=promoted,
                current_cycle=current_cycle,
            )

        # --- Clustering + REFLECTION creation ---
        reflections_created = 0
        if self._config.gates.enable_reflections:
            clusters = await self._run_clustering_if_needed(store, current_cycle)
            if clusters:
                reflections_created = await self._create_reflections(
                    store=store,
                    clusters=clusters,
                    current_cycle=current_cycle,
                    candidate_corpus=[t.content for t in candidates],
                )

        # --- Orphan REFLECTION sweep ---
        # Retire any pre-existing ACTIVE REFLECTION whose entire source
        # cluster has left the active set. REFLECTIONs created above keep
        # live sources, so they are naturally safe from the sweep.
        orphans_retired = await self._sweep_orphan_reflections(store)

        logger.info(
            "Dreaming consolidation: %d candidates, %d promoted "
            "(capped=%s, p1_fraction=%.1f%%), "
            "%d skipped gates, %d edges created, %d reflections created, "
            "%d orphan reflections retired",
            len(candidates),
            len(promoted),
            promotion_capped,
            p1_fraction_after * 100.0,
            skipped_gate,
            edges_created,
            reflections_created,
            orphans_retired,
        )

        return ConsolidationResult(
            candidates_evaluated=len(candidates),
            promoted_count=len(promoted),
            promoted_ids=promoted,
            skipped_gate_count=skipped_gate,
            scores=scores,
            edges_created=edges_created,
            reflections_created=reflections_created,
            promotion_capped=promotion_capped,
            p1_fraction_after=p1_fraction_after,
            orphans_retired=orphans_retired,
            active_signal_weights=active_weights,
            flat_signals=flat_signals,
        )

    async def _sweep_orphan_reflections(self, store: DreamingStoreProtocol) -> int:
        """Retire REFLECTIONs whose entire source cluster has left ACTIVE.

        Thin delegation to the store-owned ``retire_orphan_reflections``
        capability, which is the single shared implementation also used by the
        Memory Hygiene GC stage. This wrapper preserves the consolidation
        call-site without binding Dreaming to a concrete backend.

        Args:
            store: The store to sweep.

        Returns:
            Number of REFLECTIONs retired during this sweep.

        """
        return await store.retire_orphan_reflections()

    def _compute_active_weights(
        self,
        candidates: list[ThoughtRecord],
        *,
        current_cycle: int,
    ) -> tuple[dict[str, float], list[str]]:
        """Compute the redistributed per-signal weights for this run.

        The promotion score is a **weighted average over the signals active
        for the run**. A signal is active when its data source yields a
        non-default value for at least one candidate in the pool (see
        :func:`~engrava.domain.dreaming.default_signal_active`);
        an inactive signal contributes the same constant to every candidate
        and so carries no ranking information. This mirrors the hybrid-search
        precedent ``_redistribute_hybrid_weights``: the configured weights of
        the inactive signals are dropped and the remainder renormalised over
        the active set, so flat signals fall out of the denominator instead of
        dragging every score toward a constant.

        Custom signals (registered via ``custom_signals``) have no
        introspectable data source, so they are always treated as active — the
        operator opted into them deliberately.

        **Degenerate guard.** When no signal is active the returned weights are
        all zero, so every score is ``0.0`` and nothing promotes — the exact
        analogue of the precedent's ``active_weight == 0 -> zeros`` branch.

        **Pool-relative, once per run.** Activeness is decided over the whole
        candidate pool a single time; it is never recomputed per thought (a
        per-thought active set would let one thought's data presence
        over-promote thoughts that lack it).

        Args:
            candidates: The candidate pool for this consolidation run.
            current_cycle: The run's cycle number (drives the cycle-based
                ``recency`` / ``staleness`` activeness).

        Returns:
            A ``(weights, flat_signals)`` pair. ``weights`` maps every
            configured signal name to its effective (renormalised) weight —
            ``0.0`` for inactive signals; the active entries sum to ``1.0``
            unless no signal is active (all-zero). ``flat_signals`` is the
            sorted list of configured signals found inactive this run.

        """
        active_names: list[str] = []
        flat_signals: list[str] = []
        for name in self._signals:
            if name in DEFAULT_SIGNALS:
                is_active = default_signal_active(
                    name,
                    candidates,
                    current_cycle=current_cycle,
                    access_tracking_enabled=self._config.access_tracking_enabled,
                )
            else:
                # Custom signal — no introspectable data source; treat as active.
                is_active = True
            if is_active:
                active_names.append(name)
            else:
                flat_signals.append(name)

        active_weight = sum(self._signals[name][1] for name in active_names)
        if active_weight == 0.0:
            # No active signal (or all active weights zero) -> nothing promotes.
            weights = dict.fromkeys(self._signals, 0.0)
            return weights, sorted(flat_signals)

        weights = {
            name: (self._signals[name][1] / active_weight if name in active_names else 0.0)
            for name in self._signals
        }
        return weights, sorted(flat_signals)

    async def _apply_promotions(
        self,
        store: DreamingStoreProtocol,
        candidates: list[ThoughtRecord],
        ctx: DreamingContext,
        current_cycle: int,
        available_slots: int,
        promote_type_filter: frozenset[object],
        active_weights: dict[str, float],
    ) -> tuple[list[str], dict[str, float], int, bool]:
        """Score candidates and promote qualifying thoughts to P1.

        Separated from ``run_consolidation`` to keep per-method complexity
        within bounds.

        Args:
            store: The store to write promotion updates to.
            candidates: Active thoughts to evaluate.
            ctx: Scoring context for this pass.
            current_cycle: Current cognitive cycle (for gate checks).
            available_slots: Maximum number of new P1 promotions allowed.
            promote_type_filter: Frozenset of ``ThoughtType`` values eligible
                for promotion (``promote_targets``).
            active_weights: Redistributed per-signal weights for this run (see
                :meth:`_compute_active_weights`), applied by
                :meth:`_compute_score`.

        Returns:
            Tuple of ``(promoted_ids, scores, skipped_gate_count, promotion_capped)``.

        """
        promoted: list[str] = []
        scores: dict[str, float] = {}
        skipped_gate = 0
        promotion_capped = False
        filtered_metadata = 0

        for thought in candidates:
            score = self._compute_score(thought, ctx, active_weights)
            scores[thought.thought_id] = score

            if not self._passes_gates(thought, current_cycle):
                skipped_gate += 1
                continue

            # Metadata-aware eligibility filter — a separate exit from gate
            # checks so the observability counter can distinguish gate
            # skips (threshold/cycle) from metadata skips (perspective /
            # is_self / confidence / content_type).
            if not _is_eligible_for_dreaming(thought, self._config):
                filtered_metadata += 1
                continue

            if score > self._config.promote_threshold:
                if len(promoted) >= self._config.gates.max_promoted_per_run:
                    break
                # Skip thoughts already at P1 — they are counted in
                # ``current_p1_count`` and must not consume a new slot.
                if thought.priority.value == "P1":
                    continue
                # P1 fraction cap — continue scoring but stop promoting
                if len(promoted) >= available_slots:
                    promotion_capped = True
                    continue
                # promote_targets type filter
                if thought.thought_type not in promote_type_filter:
                    continue
                await store.update_thought(thought.thought_id, priority="P1")
                promoted.append(thought.thought_id)

        if filtered_metadata > 0:
            logger.debug(
                "dreaming filter: %d/%d candidates rejected by metadata filter (cycle %d)",
                filtered_metadata,
                len(candidates),
                current_cycle,
            )

        return promoted, scores, skipped_gate, promotion_capped

    def _compute_score(
        self,
        thought: ThoughtRecord,
        ctx: DreamingContext,
        active_weights: dict[str, float] | None = None,
    ) -> float:
        """Compute the promotion score as a weighted average over the signals.

        When ``active_weights`` is provided (the consolidation path always
        passes it), the score is the weighted average over the signals **active
        for the run** — inactive signals contribute ``0.0`` and their weight
        has already been redistributed onto the active ones by
        :meth:`_compute_active_weights`, so the effective weights sum to
        ``1.0`` (all-zero when no signal is active, giving a ``0.0`` score that
        promotes nothing). This is the reachable default scoring: a flat signal
        no longer drags every score toward a constant.

        When ``active_weights`` is ``None`` (direct callers / unit tests) the
        raw configured weights are used unchanged — the historical weighted
        sum. **This compatibility path reproduces the pre-fix, arithmetically
        unreachable scoring** (a structurally-flat signal still consumes its
        weight); it exists only for direct/legacy callers. The consolidation
        path MUST pass the redistributed ``active_weights`` from
        :meth:`_compute_active_weights` — it is the only production caller and
        always does. Do not add a new production caller that omits them.

        Args:
            thought: The thought to score.
            ctx: Consolidation context.
            active_weights: Effective per-signal weights for this run, or
                ``None`` to use the raw configured weights.

        Returns:
            Weighted score (typically in ``[0.0, 1.0]``).

        """
        total = 0.0
        for name, (signal_fn, raw_weight) in self._signals.items():
            weight = raw_weight if active_weights is None else active_weights.get(name, 0.0)
            if weight == 0.0:
                continue
            total += signal_fn(thought, ctx) * weight
        return total

    def _passes_gates(self, thought: ThoughtRecord, current_cycle: int) -> bool:
        """Check whether a thought meets the minimum gate thresholds.

        When ``allow_zero_confirmation`` is ``True`` the confirmation
        gate is bypassed, allowing freshly ingested thoughts (with
        zero confirmations) to be eligible for promotion.  The
        minimum-age gate is always enforced.

        Args:
            thought: The thought to check.
            current_cycle: Current cycle number.

        Returns:
            ``True`` if all active gates are satisfied.

        """
        gates = self._config.gates
        age = current_cycle - thought.created_cycle
        if age < gates.min_age_cycles:
            return False
        if gates.allow_zero_confirmation:
            return True
        return thought.confirmation_count >= gates.min_confirmations

    def _build_promote_type_filter(self) -> frozenset[object]:
        """Return the set of thought types eligible for priority promotion.

        Determined by ``DreamingConfig.promote_targets``.

        Returns:
            Frozenset of ``ThoughtType`` values that may be promoted.

        """
        from engrava.domain.enums import ThoughtType  # noqa: PLC0415

        target = self._config.promote_targets
        if target == "OBS_ONLY":
            return frozenset({ThoughtType.OBSERVATION})
        if target == "REFL_ONLY":
            return frozenset({ThoughtType.REFLECTION})
        # "ALL"
        return frozenset({ThoughtType.OBSERVATION, ThoughtType.REFLECTION})

    async def _create_edges_for_promoted(
        self,
        store: DreamingStoreProtocol,
        promoted_ids: list[str],
        current_cycle: int,
    ) -> int:
        """Create ``ASSOCIATED`` edges from promoted thoughts to neighbours.

        For each promoted thought that has a stored embedding, find the
        ``top_k`` most similar thoughts (above ``min_similarity``) and
        create an ``ASSOCIATED`` edge with ``source=DREAMING``.

        Idempotent: skips edges that already exist (``UNIQUE``
        constraint on ``(from_thought_id, to_thought_id, edge_type)``).
        Also skips if an edge between the two thoughts already exists
        with any type to avoid overriding user-created edges.

        Args:
            store: The store instance to create edges in.
            promoted_ids: IDs of thoughts just promoted.
            current_cycle: Current cognitive cycle.

        Returns:
            Number of edges successfully created.

        """
        import struct  # noqa: PLC0415

        from engrava.domain.enums import EdgeType, KnowledgeSource  # noqa: PLC0415
        from engrava.domain.models.edge import EdgeRecord  # noqa: PLC0415

        edge_cfg = self._config.edges
        created = 0

        async with store.suspend_auto_commit():
            for thought_id in promoted_ids:
                embedding = await store.get_embedding(thought_id)
                if embedding is None:
                    continue

                vector = list(
                    struct.unpack(f"{embedding.dimension}f", embedding.vector_blob),
                )

                # Intentionally at the default archived-exclusion: a forgotten
                # (archived) thought must not accrue new dreaming edges, so we do
                # NOT pass include_archived=True here. Only live neighbours are
                # eligible edge endpoints.
                neighbours = await store.search_similar(
                    vector,
                    top_k=edge_cfg.top_k + 1,
                    threshold=edge_cfg.min_similarity,
                )

                existing_edges = await store.get_edges(thought_id, direction="BOTH")
                connected_ids = {
                    e.to_thought_id if e.from_thought_id == thought_id else e.from_thought_id
                    for e in existing_edges
                }

                for neighbour_id, similarity in neighbours:
                    if neighbour_id == thought_id:
                        continue
                    if neighbour_id in connected_ids:
                        continue
                    if created >= edge_cfg.top_k * len(promoted_ids):
                        break

                    edge = EdgeRecord(
                        edge_id=str(uuid.uuid4()),
                        from_thought_id=thought_id,
                        to_thought_id=neighbour_id,
                        edge_type=EdgeType.ASSOCIATED,
                        weight=min(edge_cfg.edge_weight_factor * similarity, 1.0),
                        created_cycle=current_cycle,
                        source=KnowledgeSource.DREAMING,
                    )
                    try:
                        await store.create_edge(edge)
                        created += 1
                    except DuplicateEdgeError:
                        # Only an exact directed-endpoint + type duplicate is a
                        # benign skip. Any other integrity failure (CHECK,
                        # trigger, FK) propagates rather than being silenced.
                        logger.debug(
                            "Edge %s→%s already exists, skipping",
                            thought_id,
                            neighbour_id,
                        )

        return created

    # ------------------------------------------------------------------
    # Clustering / Early-stop guard
    # ------------------------------------------------------------------

    async def _run_clustering_if_needed(
        self,
        store: DreamingStoreProtocol,
        current_cycle: int,
    ) -> list[frozenset[str]]:
        """Apply the early-stop guard before delegating to ``_build_clusters``.

        Counts the current number of eligible (ACTIVE, OBSERVATION)
        candidates and compares with the count from the previous run.
        When the delta is below ``DreamingGates.clustering_min_new_candidates``
        the clustering phase is skipped entirely and an empty list is
        returned — signals, promotion, and edge creation are unaffected.

        The guard is bypassed on the very first run (no previous count
        is stored yet) and when ``clustering_min_new_candidates`` is
        ``0`` (opt-out).

        Args:
            store: The store instance to read candidate counts from.
            current_cycle: Current cognitive cycle.

        Returns:
            Clusters from ``_build_clusters``, or an empty list if the
            early-stop guard fires.

        """
        min_new = self._config.gates.clustering_min_new_candidates

        # Count eligible candidates using the same type filter as _build_clusters.
        allowed_types = self._config.gates.cluster_allowed_types
        current_count = 0
        for ttype in allowed_types:
            current_count += await store.count_thoughts(
                lifecycle_status="ACTIVE",
                thought_type=ttype,
            )

        if min_new > 0 and self._last_clustering_candidate_count is not None:
            new_count = current_count - self._last_clustering_candidate_count
            if new_count < min_new:
                logger.info(
                    "Skipping clustering in cycle %d: only %d new candidates "
                    "(< %d threshold). Signals and promotion still executed.",
                    current_cycle,
                    new_count,
                    min_new,
                )
                return []

        self._last_clustering_candidate_count = current_count
        return await self._build_clusters(store, current_cycle)

    async def _build_clusters(
        self,
        store: DreamingStoreProtocol,
        current_cycle: int,
    ) -> list[frozenset[str]]:
        """Build thought clusters using ASSOCIATED edges in the graph.

        Runs Label Propagation (LPA) over the ASSOCIATED edge graph.
        Falls back to cosine-similarity agglomerative clustering when
        the graph is explicitly configured to use it. When
        ``cold_start_clustering`` is enabled and the LPA edge graph is
        empty, the same agglomerative path runs within this cycle so
        clustering still succeeds on a fresh or sparse graph.

        Args:
            store: The store instance to read edges and embeddings from.
            current_cycle: Current cognitive cycle (reserved for
                future cycle-scoped clustering; unused at present).

        Returns:
            List of disjoint clusters, each a frozen set of thought IDs.
            Only clusters with at least ``min_cluster_size`` members are
            returned.

        """
        logger.debug("_build_clusters called at cycle %d", current_cycle)
        gates = self._config.gates
        min_size = gates.min_cluster_size
        algorithm = gates.cluster_algorithm

        from engrava.domain.enums import EdgeType, KnowledgeSource  # noqa: PLC0415

        if algorithm == "agglomerative":
            # Graph-independent path: cluster ACTIVE candidates of the
            # allowed types (default: OBSERVATION only). See
            # ``_cold_start_agglomerative_clusters`` for the pool build.
            clusters = await self._cold_start_agglomerative_clusters(store)
        else:
            # LPA path: operates over the dream-created ASSOCIATED edge graph.
            all_edges = await store.list_edges(
                edge_type=EdgeType.ASSOCIATED,
                source=KnowledgeSource.DREAMING,
            )

            adjacency: dict[str, set[str]] = {}
            for edge in all_edges:
                a, b = edge.from_thought_id, edge.to_thought_id
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)

            if not adjacency:
                if gates.cold_start_clustering:
                    logger.info(
                        "Dreaming clustering: LPA edge graph empty at cycle %d; "
                        "falling back to cold-start agglomerative clustering.",
                        current_cycle,
                    )
                    clusters = await self._cold_start_agglomerative_clusters(store)
                else:
                    return []
            else:
                clusters = _lpa_clusters(adjacency)

        # Reject oversized clusters (single-link chaining guard).
        # Clusters exceeding max_cluster_size are dropped entirely — their
        # member thoughts remain as ungrouped OBSERVATIONs with no REFLECTION.
        max_size = gates.max_cluster_size
        if max_size is not None:
            oversized = [c for c in clusters if len(c) > max_size]
            if oversized:
                logger.info(
                    "Rejected %d cluster(s) exceeding max_cluster_size=%d at cycle %d (sizes: %s)",
                    len(oversized),
                    max_size,
                    current_cycle,
                    sorted((len(c) for c in oversized), reverse=True)[:5],
                )
                clusters = [c for c in clusters if len(c) <= max_size]

        return [c for c in clusters if len(c) >= min_size]

    async def _cold_start_agglomerative_clusters(
        self,
        store: DreamingStoreProtocol,
    ) -> list[frozenset[str]]:
        """Cluster the eligible candidate pool with agglomerative clustering.

        Builds the candidate pool from ``ACTIVE`` thoughts of the configured
        ``cluster_allowed_types`` (default: ``OBSERVATION`` only) and runs the
        cosine-similarity agglomerative backend over it. This path is
        graph-independent, so it produces clusters even when no dream
        ``ASSOCIATED`` edges exist yet — the shared mechanism behind both the
        explicit ``cluster_algorithm="agglomerative"`` mode and the opt-in
        ``cold_start_clustering`` fallback for the ``"lpa"`` path.

        Excluding ``REFLECTION`` from the default pool prevents the
        meta-reflection cascade where REFLECTIONs created in cycle N get
        re-clustered into even more abstract meta-REFLECTIONs in cycle N+1.

        The candidate query is bounded by ``candidates_limit`` so a large
        active set cannot make the O(n²) similarity matrix unbounded.

        Args:
            store: The store instance to read candidates and embeddings from.

        Returns:
            List of disjoint clusters, each a frozen set of thought IDs. An
            empty list when no eligible candidates have stored embeddings.

        """
        gates = self._config.gates
        allowed_types = gates.cluster_allowed_types
        if len(allowed_types) == 1:
            candidates = await store.list_thoughts(
                lifecycle_status="ACTIVE",
                thought_type=allowed_types[0],
                limit=self._config.candidates_limit,
            )
        else:
            # Union over multiple types (rare path; keeps behaviour
            # correct if operator opts into meta-consolidation).
            candidates = []
            for ttype in allowed_types:
                candidates.extend(
                    await store.list_thoughts(
                        lifecycle_status="ACTIVE",
                        thought_type=ttype,
                        limit=self._config.candidates_limit,
                    )
                )
        cluster_fn = (
            self._agglomerative_clusters
            if self._config.clustering_backend == "numpy"
            else self._agglomerative_clusters_python_legacy
        )
        return await cluster_fn(
            store=store,
            node_ids=[t.thought_id for t in candidates],
            threshold=gates.cluster_similarity_threshold,
        )

    @staticmethod
    async def _agglomerative_clusters(  # noqa: C901
        store: DreamingStoreProtocol,
        node_ids: list[str],
        threshold: float,
    ) -> list[frozenset[str]]:
        """Cosine-similarity agglomerative clustering (single-linkage).

        Groups thoughts whose cosine similarity exceeds ``threshold``.
        Only thoughts with stored embeddings are considered; others are
        dropped silently.

        **Implementation:** similarity matrix computed with
        numpy matmul. For N candidates with embedding dim D, memory
        footprint of the N x N float32 similarity matrix is ``4 * N**2``
        bytes. For N greater than ``_VECTORIZED_CLUSTERING_CHUNK_SIZE``
        a chunked mode processes rows in blocks to bound peak RAM.
        Mathematically identical to the legacy pure-Python loop (modulo
        ~1e-7 float32 accumulation noise — safely below the 0.65-0.70
        cluster threshold).

        Args:
            store: Store to read embeddings from.
            node_ids: Candidate thought IDs to cluster.
            threshold: Minimum cosine similarity to link two thoughts.

        Returns:
            List of clusters as frozen sets of thought IDs.

        """
        import struct  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        # --- Load embeddings -------------------------------------------
        vectors: dict[str, list[float]] = {}
        for tid in node_ids:
            emb = await store.get_embedding(tid)
            if emb is None:
                continue
            vec = list(struct.unpack(f"{emb.dimension}f", emb.vector_blob))
            # Drop all-zero vectors (parity with the legacy path which
            # skipped them via the ``norm == 0.0`` guard).
            if any(vec):
                vectors[tid] = vec

        ids = list(vectors)
        if not ids:
            return []

        # --- Union-Find ------------------------------------------------
        parent: dict[str, str] = {tid: tid for tid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            ra, rb = find(x), find(y)
            if ra != rb:
                parent[ra] = rb

        # --- Build similarity matrix (numpy, vectorized) --------------
        mat = np.asarray(
            [vectors[tid] for tid in ids],
            dtype=np.float32,
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        mat_n = mat / norms

        n = len(ids)
        chunk = _VECTORIZED_CLUSTERING_CHUNK_SIZE

        if n <= chunk:
            # Single-shot matmul — N x N sim matrix fits in RAM.
            sim = mat_n @ mat_n.T
            # Upper triangle (k=1 excludes the diagonal / self-pairs).
            mask = np.triu(sim >= threshold, k=1)
            pair_rows, pair_cols = np.where(mask)
            for i, j in zip(pair_rows.tolist(), pair_cols.tolist(), strict=False):
                union(ids[i], ids[j])
        else:
            # Chunked mode: process rows in blocks to bound memory.
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                block = mat_n[start:end]
                sim_block = block @ mat_n.T
                mask = sim_block >= threshold
                local_rows, cols = np.where(mask)
                for lr, c in zip(local_rows.tolist(), cols.tolist(), strict=False):
                    global_i = start + lr
                    # Skip diagonal + lower triangle to preserve the
                    # "upper-triangle once" semantics of the single-shot
                    # path. (Duplicates would be idempotent under union
                    # but wasted work.)
                    if c <= global_i:
                        continue
                    union(ids[global_i], ids[c])

        # --- Group by root --------------------------------------------
        groups: dict[str, set[str]] = {}
        for tid in ids:
            root = find(tid)
            groups.setdefault(root, set()).add(tid)

        return [frozenset(members) for members in groups.values()]

    @staticmethod
    async def _agglomerative_clusters_python_legacy(  # noqa: C901
        store: DreamingStoreProtocol,
        node_ids: list[str],
        threshold: float,
    ) -> list[frozenset[str]]:
        """Pure-Python O(N²) cosine-similarity clustering (escape hatch).

        Mathematically equivalent to ``_agglomerative_clusters`` but
        uses a nested Python loop instead of numpy matmul.  Provided
        as a debugging escape hatch via
        ``DreamingConfig.clustering_backend = "python"`` — useful if
        a float32 numerical discrepancy between backends is ever
        suspected.

        **Do not use in production** -- this path is ~1000x slower
        for N > 1 000.

        Args:
            store: Store to read embeddings from.
            node_ids: Candidate thought IDs to cluster.
            threshold: Minimum cosine similarity to link two thoughts.

        Returns:
            List of clusters as frozen sets of thought IDs.

        """
        import math  # noqa: PLC0415
        import struct  # noqa: PLC0415

        # --- Load embeddings ------------------------------------------
        vectors: dict[str, list[float]] = {}
        for tid in node_ids:
            emb = await store.get_embedding(tid)
            if emb is None:
                continue
            vec = list(struct.unpack(f"{emb.dimension}f", emb.vector_blob))
            if any(vec):
                vectors[tid] = vec

        ids = list(vectors)
        if not ids:
            return []

        # --- Union-Find -----------------------------------------------
        parent: dict[str, str] = {tid: tid for tid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            ra, rb = find(x), find(y)
            if ra != rb:
                parent[ra] = rb

        # --- Pure-Python nested loop ----------------------------------
        norms = {tid: math.sqrt(sum(v * v for v in vectors[tid])) for tid in ids}
        for i, a in enumerate(ids):
            na = norms[a]
            if na == 0.0:
                continue
            for b in ids[i + 1 :]:
                nb = norms[b]
                if nb == 0.0:
                    continue
                dot = sum(x * y for x, y in zip(vectors[a], vectors[b], strict=False))
                sim_val = dot / (na * nb)
                if sim_val >= threshold:
                    union(a, b)

        # --- Group by root --------------------------------------------
        groups: dict[str, set[str]] = {}
        for tid in ids:
            groups.setdefault(find(tid), set()).add(tid)

        return [frozenset(members) for members in groups.values()]

    # ------------------------------------------------------------------
    # REFLECTION creation
    # ------------------------------------------------------------------

    async def _create_reflections(  # noqa: C901, PLR0912, PLR0915
        self,
        store: DreamingStoreProtocol,
        clusters: list[frozenset[str]],
        current_cycle: int,
        candidate_corpus: list[str] | None = None,
        *,
        override_valid_from: str | None = None,
        override_valid_until: str | None = None,
    ) -> int:
        """Create REFLECTION thoughts from clustered thought sets.

        For each cluster:

        1. Compute centroid embedding (mean of member vectors).
        2. Build structured content (top-N keywords + member IDs).
        3. Derive an idempotence hash from sorted member IDs.
        4. Skip if a REFLECTION with the same hash already exists.
        5. Derive the REFLECTION's valid-time extent from its members
           (see ``derive_reflection_extent``), unless the caller pins an
           explicit override.
        6. Persist the REFLECTION thought, centroid embedding, and
           ``CONSOLIDATED_FROM`` edges to each cluster member.

        Args:
            store: Store to read embeddings from and write to.
            clusters: Disjoint clusters returned by ``_build_clusters()``.
            current_cycle: Cycle used for created/updated_cycle fields.
            candidate_corpus: Content strings of the parent candidate
                set the clusters were drawn from.  Forwarded to the v2
                content builder as the IDF document set so
                ``top_keyphrases`` get meaningful TF-IDF scores.  When
                omitted (legacy callers) the builder degenerates to a
                tie-broken-by-phrase ordering — that is the documented
                empty-corpus behaviour, but production callsites should
                always supply the corpus to keep the enrichment
                substantive.
            override_valid_from: When non-``None``, every REFLECTION
                created in this call takes this ISO-8601 ``valid_from``
                instead of the value derived from its members.  ``None``
                (the default) means "derive from members" — it does
                **not** force an open lower bound; an open lower bound is
                only produced when the derivation itself yields ``None``.
            override_valid_until: When non-``None``, every REFLECTION
                created in this call takes this ISO-8601 ``valid_until``
                instead of the derived value.  ``None`` (the default)
                means "derive from members", mirroring
                ``override_valid_from``.

        Returns:
            Number of new REFLECTION thoughts persisted.

        """
        import hashlib  # noqa: PLC0415
        import json  # noqa: PLC0415
        import struct  # noqa: PLC0415

        from engrava.domain.enums import (  # noqa: PLC0415
            EdgeType,
            KnowledgeSource,
            LifecycleStatus,
            Priority,
            ThoughtType,
        )
        from engrava.domain.models.edge import EdgeRecord  # noqa: PLC0415
        from engrava.domain.models.thought import ThoughtRecord  # noqa: PLC0415
        from engrava.extensions.dreaming_cluster_quality import (  # noqa: PLC0415
            has_consistent_entities,
            has_contradictory_members,
            has_duplicate_content_members,
            has_meaningful_keyphrases,
            is_external_source_homogeneous,
            is_low_cohesion,
            is_persona_only_cluster,
        )
        from engrava.extensions.dreaming_reflection_content import (  # noqa: PLC0415
            build_reflection_content_v2,
        )
        from engrava.extensions.dreaming_reflection_extent import (  # noqa: PLC0415
            derive_reflection_extent,
        )

        # Resolve cluster algorithm at the callsite — the dreaming
        # extension does not retain it as instance state, but the v2
        # content builder needs it for the ``cluster_algorithm``
        # field.
        algorithm = self._config.gates.cluster_algorithm

        created = 0
        # Per-gate rejection counters — populated by the cluster quality
        # gating loop below and emitted via a single summary ``INFO`` log
        # line once the consolidation pass is over (zero counts stay
        # silent).
        gating_rejections: dict[str, int] = {
            "duplicate_members": 0,
            "persona_only": 0,
            "contradictory_members": 0,
            "low_cohesion": 0,
            "external_source_mixed": 0,
            "named_entity_inconsistent": 0,
            "generic_keyphrases": 0,
        }

        # ----------------------------------------------------------
        # PRE-PASS — compute cross-cluster phrase document frequency
        # ----------------------------------------------------------
        #
        # The pre-pass enumerates every cluster's raw TF-IDF
        # keyphrases ahead of the main loop so the boilerplate filter
        # has a corpus-wide phrase-frequency picture before any
        # REFLECTION is materialised.  The map produced here is fed
        # to ``build_reflection_content_v2`` as the new
        # ``cluster_phrase_df`` / ``total_clusters`` pair — the
        # in-builder filter is the actual decision point, the
        # pre-pass merely collects the statistics.  Clusters that
        # would be dropped by the eligibility filter on the main
        # loop are still counted here intentionally: the document-
        # frequency signal is more robust when computed over every
        # cluster the run considered, not just the survivors.
        from engrava.extensions.dreaming_keyphrases import (  # noqa: PLC0415 -- local import keeps the top-of-file import surface stable
            compute_cluster_phrase_frequency,
            top_keyphrases_tfidf,
        )

        prepass_corpus: list[str] = []
        prepass_cluster_thoughts: list[list[ThoughtRecord]] = []
        for cluster in clusters:
            cluster_members: list[ThoughtRecord] = []
            for tid in sorted(cluster):
                try:
                    thought = await store.get_thought(tid)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "pre-pass: could not retrieve thought %s for "
                        "phrase-frequency tally; skipping",
                        tid,
                    )
                    continue
                if thought is not None:
                    cluster_members.append(thought)
                    prepass_corpus.append(thought.content)
            prepass_cluster_thoughts.append(cluster_members)

        raw_keyphrases_per_cluster: list[list[dict[str, float | str]]] = [
            top_keyphrases_tfidf(
                members,
                corpus=prepass_corpus,
                top_n=self._config.top_keyphrases_count,
            )
            if members
            else []
            for members in prepass_cluster_thoughts
        ]
        cluster_phrase_df = compute_cluster_phrase_frequency(
            raw_keyphrases_per_cluster,
        )
        total_clusters = len(clusters)

        async with store.suspend_auto_commit():
            for cluster in clusters:
                raw_member_ids = sorted(cluster)

                # --- Resolve cluster members (thoughts + embeddings keyed by id) ---
                resolved_thoughts: dict[str, ThoughtRecord] = {}
                resolved_vectors: dict[str, list[float]] = {}
                for tid in raw_member_ids:
                    emb = await store.get_embedding(tid)
                    if emb is not None:
                        vec = list(struct.unpack(f"{emb.dimension}f", emb.vector_blob))
                        resolved_vectors[tid] = vec
                    try:
                        thought = await store.get_thought(tid)
                        if thought is not None:
                            resolved_thoughts[tid] = thought
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not retrieve thought %s for clustering", tid)

                # --- Metadata-aware eligibility filter (Aggressive C2a) ---
                # Filtered subset feeds the cluster hash, the centroid
                # embedding, the structured content payload and the
                # CONSOLIDATED_FROM lineage edges.  A cluster with too few
                # eligible members is dropped entirely — the threshold
                # parameter lives on DreamingGates.
                eligible_member_ids = [
                    tid
                    for tid in raw_member_ids
                    if tid in resolved_thoughts
                    and _is_eligible_for_dreaming(
                        resolved_thoughts[tid],
                        self._config,
                    )
                ]
                if len(eligible_member_ids) < self._config.gates.min_cluster_size:
                    continue

                # From here onward only the filtered subset participates.
                member_ids = eligible_member_ids
                member_thoughts: list[ThoughtRecord] = [
                    resolved_thoughts[tid] for tid in member_ids
                ]
                member_vectors: list[list[float]] = [
                    resolved_vectors[tid] for tid in member_ids if tid in resolved_vectors
                ]
                max_priority: Priority = Priority.P4
                for thought in member_thoughts:
                    max_priority = min(max_priority, thought.priority)

                # --- Cluster quality gating ---
                # Six content-quality gates run on the resolved member
                # subset before idempotence + REFLECTION materialisation,
                # skipping clusters that look like dedup escapes, persona
                # dumps, contradictory mixes, loose-cohesion bags,
                # source-mixed clusters, or named-entity-inconsistent
                # groupings.  The seventh gate (meaningful keyphrases)
                # needs the built content payload and fires below.  The
                # whole block is bypassed when the operator flips
                # ``cluster_quality_gating_enabled`` to ``False`` for
                # ablation testing.  First failure rejects the cluster
                # and increments the per-gate counter for the summary
                # log line emitted at the end of the consolidation pass.
                if self._config.gates.cluster_quality_gating_enabled:
                    is_dup, dup_count = has_duplicate_content_members(member_thoughts)
                    if is_dup:
                        gating_rejections["duplicate_members"] += 1
                        logger.info(
                            "Skipping cluster %s: %d duplicate-content member(s)",
                            cluster,
                            dup_count,
                        )
                        continue

                    is_persona, persona_ratio = is_persona_only_cluster(
                        member_thoughts,
                        persona_threshold=(self._config.gates.cluster_quality_persona_threshold),
                    )
                    if is_persona:
                        gating_rejections["persona_only"] += 1
                        logger.info(
                            "Skipping cluster %s: persona-only ratio %.2f",
                            cluster,
                            persona_ratio,
                        )
                        continue

                    is_contradictory, contradiction_reasons = has_contradictory_members(
                        member_thoughts,
                    )
                    if is_contradictory:
                        gating_rejections["contradictory_members"] += 1
                        logger.info(
                            "Skipping cluster %s: contradictory members %s",
                            cluster,
                            contradiction_reasons,
                        )
                        continue

                    is_loose, cohesion_score = is_low_cohesion(
                        member_vectors,
                        cohesion_threshold=(self._config.gates.cluster_quality_cohesion_threshold),
                    )
                    if is_loose:
                        gating_rejections["low_cohesion"] += 1
                        logger.info(
                            "Skipping cluster %s: cohesion %.3f below threshold",
                            cluster,
                            cohesion_score,
                        )
                        continue

                    is_homogeneous, external_fraction = is_external_source_homogeneous(
                        member_thoughts,
                        min_external_fraction=(
                            self._config.gates.cluster_quality_external_homogeneity_threshold
                        ),
                    )
                    if not is_homogeneous:
                        gating_rejections["external_source_mixed"] += 1
                        logger.info(
                            "Skipping cluster %s: external-source fraction %.2f below threshold",
                            cluster,
                            external_fraction,
                        )
                        continue

                    is_consistent, ne_ratio = has_consistent_entities(
                        member_thoughts,
                        min_shared_ratio=(
                            self._config.gates.cluster_quality_ne_consistency_threshold
                        ),
                    )
                    if not is_consistent:
                        gating_rejections["named_entity_inconsistent"] += 1
                        logger.info(
                            "Skipping cluster %s: named-entity overlap %.2f below threshold",
                            cluster,
                            ne_ratio,
                        )
                        continue

                # --- Idempotence check (content-hash of filtered member IDs) ---
                # Filter-aware hash: same cluster scanned under different
                # eligibility configuration legitimately yields a new
                # REFLECTION, matching the C2a contract that filtered
                # members alone form the synthesis.
                cluster_hash = hashlib.sha256(
                    json.dumps(member_ids).encode(),
                ).hexdigest()[:16]

                if await store.thought_exists_by_source(
                    source=f"dreaming:{cluster_hash}",
                    thought_type_value="REFLECTION",
                ):
                    logger.debug(
                        "Skipping reflection for cluster %s (already exists)",
                        cluster_hash,
                    )
                    continue

                if not member_vectors:
                    continue

                # Centroid = L2-normalized mean of member vectors. Shared with
                # the re-bind path so creation and refresh cannot diverge.
                centroid = compute_centroid(member_vectors)

                # --- Build structured content (schema v2 — additive over legacy v1) ---
                content_obj = build_reflection_content_v2(
                    member_thoughts,
                    algorithm=algorithm,
                    config=self._config,
                    corpus=candidate_corpus,
                    cluster_phrase_df=cluster_phrase_df,
                    total_clusters=total_clusters,
                )
                # Post-build gate — runs once the cross-cluster TF-IDF
                # boilerplate filter has had its say on ``top_keyphrases``,
                # so any cluster left with an entirely generic
                # determiner-noun keyphrase list is rejected here.  Same
                # master switch as the pre-build gates, plus a dedicated
                # ``cluster_quality_require_meaningful_keyphrases`` opt-out
                # for the rare deployment that wants the other gates but
                # not this one.
                if (
                    self._config.gates.cluster_quality_gating_enabled
                    and self._config.gates.cluster_quality_require_meaningful_keyphrases
                    and not has_meaningful_keyphrases(content_obj["top_keyphrases"])
                ):
                    gating_rejections["generic_keyphrases"] += 1
                    logger.info(
                        "Skipping cluster %s: top_keyphrases entirely generic",
                        cluster,
                    )
                    continue
                content_str = json.dumps(content_obj, ensure_ascii=False)
                keywords = content_obj["keywords"]
                essence = f"REFLECTION [{', '.join(keywords[:3])}]"[:200]

                # --- Derive valid-time extent inherited from members ---
                # ``member_thoughts`` is already resolved above (it drives
                # the centroid + content build), so the bounds are read
                # from records already in hand — no extra store round-trip.
                # An explicit caller override wins over the derived value
                # on each axis independently.
                derived_valid_from, derived_valid_until = derive_reflection_extent(
                    (t.valid_from, t.valid_until) for t in member_thoughts
                )
                reflection_valid_from = (
                    override_valid_from if override_valid_from is not None else derived_valid_from
                )
                reflection_valid_until = (
                    override_valid_until
                    if override_valid_until is not None
                    else derived_valid_until
                )

                # --- Create REFLECTION thought ---
                reflection_id = str(uuid.uuid4())
                reflection = ThoughtRecord(
                    thought_id=reflection_id,
                    thought_type=ThoughtType.REFLECTION,
                    essence=essence,
                    content=content_str,
                    priority=Priority(self._config.reflection_default_priority),
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    created_cycle=current_cycle,
                    updated_cycle=current_cycle,
                    source=f"dreaming:{cluster_hash}",
                    source_type=KnowledgeSource.DREAMING,
                    valid_from=reflection_valid_from,
                    valid_until=reflection_valid_until,
                )
                try:
                    await store.create_thought(reflection)
                except Exception:  # noqa: BLE001
                    logger.debug("Could not create reflection thought %s", reflection_id)
                    continue

                # --- Store centroid embedding ---
                try:
                    await store.store_embedding(
                        reflection_id,
                        centroid,
                        model_name=CENTROID_MODEL_NAME,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("Could not store centroid embedding for %s", reflection_id)

                # --- Create CONSOLIDATED_FROM edges: REFLECTION → each member ---
                for member_id in member_ids:
                    edge = EdgeRecord(
                        edge_id=str(uuid.uuid4()),
                        from_thought_id=reflection_id,
                        to_thought_id=member_id,
                        edge_type=EdgeType.CONSOLIDATED_FROM,
                        weight=1.0,
                        created_cycle=current_cycle,
                        source=KnowledgeSource.DREAMING,
                    )
                    try:
                        await store.create_edge(edge)
                    except DuplicateEdgeError:
                        # Only an exact duplicate is a benign skip. A real
                        # integrity failure (CHECK, trigger, FK) propagates so a
                        # REFLECTION is never counted as created while its
                        # lineage edges are silently missing.
                        logger.debug(
                            "CONSOLIDATED_FROM edge %s→%s already exists",
                            reflection_id,
                            member_id,
                        )

                created += 1
                logger.debug(
                    "Created REFLECTION %s for cluster of %d members",
                    reflection_id,
                    len(member_ids),
                )

        total_rejected = sum(gating_rejections.values())
        if total_rejected > 0:
            logger.info(
                "Cluster quality gating rejected %d cluster(s): %s",
                total_rejected,
                {gate: count for gate, count in gating_rejections.items() if count > 0},
            )

        return created


# ------------------------------------------------------------------
# Module-level helpers (no LLM)
# ------------------------------------------------------------------


#: Ordinal ranking of the confidence levels recognised on
#: ``metadata["source"]["confidence"]``.  Used by
#: :func:`_is_eligible_for_dreaming` to compare a thought's annotated
#: confidence against the configured ``min_source_confidence`` threshold.
#: Higher integer = stronger evidence the caller had on ingest.
_CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def _is_eligible_for_dreaming(  # noqa: PLR0911 -- one return per filter axis reads clearer than nested guards
    thought: ThoughtRecord,
    config: DreamingConfig,
) -> bool:
    """Return ``True`` if the thought passes the metadata-aware filters.

    The filter operates on the self-anchored attributes carried under
    ``ThoughtRecord.metadata``:

    * ``metadata["perspective"]`` — caller-reported cognitive axis,
      restricted to ``"percept"`` / ``"utterance"`` / ``"thought"``.
    * ``metadata["source"]["is_self"]`` — agent-vs-world asymmetry.
    * ``metadata["source"]["confidence"]`` — caller certainty,
      ranked ``"low" < "medium" < "high"``.
    * ``metadata["content_type"]`` — orthogonal modality filter.

    Each axis is honoured only when the relevant ``DreamingConfig``
    field opts in (sentinel ``None`` / ``"any"``).  Thoughts that omit
    an annotation pass the corresponding axis to preserve backward
    compatibility with legacy stores written before the metadata
    schema landed; this matches the contract documented on the
    ``DreamingConfig`` fields.  Type-/source-level filters such as
    ``ThoughtType.OBSERVATION`` and ``KnowledgeSource.EXPERIENCE``
    remain the caller's responsibility — they are intentionally
    orthogonal to this helper.

    Args:
        thought: Candidate thought being considered for promotion or
            REFLECTION clustering.
        config: Dreaming configuration carrying the filter knobs.

    Returns:
        ``True`` when the thought clears every configured filter axis,
        ``False`` otherwise.

    """
    metadata = thought.metadata or {}

    # Backward-compat policy: thoughts with NO structured
    # metadata at all (legacy callers that pre-date the schema) bypass
    # every metadata-driven filter.  Once any structured attribute is
    # present the per-axis filters apply, with missing-key semantics
    # documented per field on ``DreamingConfig``.
    if not metadata:
        return True

    source_raw = metadata.get("source")
    source: Mapping[str, object] = source_raw if isinstance(source_raw, dict) else {}

    # Perspective filter (positive list — caller opts in via non-None).
    if config.eligible_perspectives is not None:
        perspective = metadata.get("perspective")
        if perspective is not None and perspective not in config.eligible_perspectives:
            return False

    # Self-filter mode (boolean axis on source.is_self).
    #
    # Identity comparisons (``is True`` / ``is False``) are used here on
    # purpose: ``is_self`` is contractually a strict ``bool`` and any
    # other shape (string ``"false"``, int ``0``, missing key, malformed
    # payload) is treated as an unclassified caller and falls through to
    # the backward-compat "eligible" branch, exactly like a thought that
    # never carried the annotation in the first place.  A plain truthiness
    # check would let ``"false"`` masquerade as ``True``, silently breaking
    # ``external_only`` deployments.
    if config.self_filter_mode != "any":
        is_self = source.get("is_self")
        if is_self is True and config.self_filter_mode == "external_only":
            return False
        if is_self is False and config.self_filter_mode == "self_only":
            return False

    # Source confidence threshold (ordinal ranking).
    confidence = source.get("confidence", "low")
    confidence_rank = _CONFIDENCE_RANK.get(confidence, 0) if isinstance(confidence, str) else 0
    if confidence_rank < _CONFIDENCE_RANK[config.min_source_confidence]:
        return False

    # Content type — negative filter first, then optional positive list.
    ctype = metadata.get("content_type")
    if ctype is not None and ctype in config.excluded_content_types:
        return False
    return not (
        config.eligible_content_types is not None
        and ctype is not None
        and ctype not in config.eligible_content_types
    )


def _lpa_clusters(
    adjacency: dict[str, set[str]],
    *,
    max_iter: int = 20,
    seed: int = 42,
) -> list[frozenset[str]]:
    """Label Propagation Algorithm over an adjacency graph.

    Deterministic via seeded random for shuffling order.  Each node
    takes the most common label among its neighbours; ties broken by
    smallest label string.  Convergence detected when no label changes.

    Args:
        adjacency: Mapping of node_id → set of neighbour node_ids.
        max_iter: Maximum number of iterations before early stop.
        seed: Random seed for deterministic shuffling.

    Returns:
        List of clusters as frozen sets of node IDs.

    """
    import random  # noqa: PLC0415

    nodes = list(adjacency)
    labels: dict[str, str] = {n: n for n in nodes}
    rng = random.Random(seed)  # noqa: S311

    for _ in range(max_iter):
        rng.shuffle(nodes)
        changed = False
        for node in nodes:
            neighbours = adjacency.get(node, set())
            if not neighbours:
                continue
            # Count labels among neighbours
            counts: dict[str, int] = {}
            for nb in neighbours:
                lbl = labels[nb]
                counts[lbl] = counts.get(lbl, 0) + 1
            max_count = max(counts.values())
            # Among labels with max count, pick lexicographically smallest
            best = min(lbl for lbl, cnt in counts.items() if cnt == max_count)
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break

    # Group by label
    groups: dict[str, set[str]] = {}
    for node, lbl in labels.items():
        groups.setdefault(lbl, set()).add(node)

    return [frozenset(members) for members in groups.values()]

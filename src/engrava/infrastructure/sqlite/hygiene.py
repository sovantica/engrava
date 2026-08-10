"""Memory Hygiene forgetting-loop scoring and result types.

The deterministic, no-LLM forgetting loop is the subtractive counterpart to
dreaming consolidation. This module holds its **pure** pieces — the keep-score
computation with active-signal weight redistribution, the per-thought eviction
reason, and the run-result value object — while the loop's orchestration (the
two-stage archive then garbage-collect, journaling, and database access) lives
on :class:`~engrava.infrastructure.sqlite.engrava_core.SqliteEngravaCore`.

The keep-score reuses the inward dreaming signal library
(:mod:`engrava.domain.dreaming`) and the same active-signal
redistribution the dreaming scorer uses (a signal whose data source is flat
across the candidate pool is dropped and its weight renormalised over the active
set), but carries the hygiene weight vector and threshold so the two loops tune
independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from engrava.domain.dreaming import (
    DEFAULT_SIGNALS,
    DreamingContext,
    default_signal_active,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from engrava.domain.models.thought import ThoughtRecord


@dataclass(frozen=True)
class EvictionReason:
    """Why the hygiene loop selected a thought for archival.

    A structured, deterministic record of the keep-score computation behind a
    single archival decision. It rides in the journal entry's ``delta`` (nested
    under ``eviction_reason``) so a forgotten thought is reconstructable and
    tamper-evident, and it is returned per-thought in a ``dry_run`` preview.

    Attributes:
        thought_id: The archived (or would-be-archived) thought's id.
        mechanism: Always ``"hygiene"`` — distinguishes a hygiene archival from
            a TTL/manual one when auditing the journal.
        keep_score: The renormalised weighted keep-score in ``[0.0, 1.0]``.
        eviction_score: ``keep_score * decay_multiplier`` in ``[0.0, 1.0]`` — the
            value compared against ``threshold``.
        decay_multiplier: The clamped ``decay_function`` result in ``[0.0, 1.0]``
            (``1.0`` for the default no-op hook, and for a non-finite custom
            return — the fail-safe direction).
        threshold: The ``eviction_threshold`` the score fell below.
        signals: Per-signal raw scores (only the signals active this run) that
            fed the keep-score, for audit.

    Examples:
        >>> reason = EvictionReason(
        ...     thought_id="t1",
        ...     keep_score=0.1,
        ...     eviction_score=0.1,
        ...     decay_multiplier=1.0,
        ...     threshold=0.2,
        ...     signals={"recency": 0.1},
        ... )
        >>> reason.mechanism
        'hygiene'

    """

    thought_id: str
    keep_score: float
    eviction_score: float
    decay_multiplier: float
    threshold: float
    signals: dict[str, float] = field(default_factory=dict)
    mechanism: str = "hygiene"

    def to_delta(self) -> dict[str, object]:
        """Render this reason as the nested ``eviction_reason`` journal payload.

        Returns:
            A JSON-serialisable mapping with the mechanism, both scores, the
            decay multiplier, the threshold, and the per-signal breakdown — the
            exact shape embedded under ``delta.eviction_reason``.

        """
        return {
            "mechanism": self.mechanism,
            "keep_score": self.keep_score,
            "eviction_score": self.eviction_score,
            "decay_multiplier": self.decay_multiplier,
            "threshold": self.threshold,
            "signals": dict(self.signals),
        }


@dataclass(frozen=True)
class HygieneResult:
    """Outcome of a single Memory Hygiene run.

    Attributes:
        archived_count: Number of thoughts archived this run (``0`` under
            ``dry_run``, which mutates nothing).
        gc_count: Number of archived thoughts physically garbage-collected this
            run (``0`` unless ``auto_gc_enabled``, and ``0`` under ``dry_run``).
        candidates_evaluated: Number of ACTIVE/CREATED thoughts scored this run.
        dry_run: Whether this was a preview — when ``True`` nothing was mutated
            and nothing was journaled; ``would_evict`` carries the preview.
        would_evict: Under ``dry_run``, the per-thought eviction reasons for the
            thoughts that *would* be archived (empty on a real run — the
            information is in the journal instead). Ordered by the deterministic
            archive selection order.
        flat_signals: Names of configured keep-signals that were inactive this
            run (their data source was flat across the candidate pool), so their
            weight was redistributed onto the active signals. Sorted.

    Examples:
        >>> result = HygieneResult(archived_count=3, gc_count=0)
        >>> result.archived_count
        3
        >>> HygieneResult().dry_run
        False

    """

    archived_count: int = 0
    gc_count: int = 0
    candidates_evaluated: int = 0
    dry_run: bool = False
    would_evict: list[EvictionReason] = field(default_factory=list)
    flat_signals: list[str] = field(default_factory=list)


def compute_active_hygiene_weights(
    signal_weights: Mapping[str, float],
    candidates: list[ThoughtRecord],
    *,
    current_cycle: int,
    access_tracking_enabled: bool,
) -> tuple[dict[str, float], list[str]]:
    """Redistribute the keep-score weights over the signals active this run.

    A signal is *active* when its data source yields a non-default value for at
    least one candidate in the pool (see
    :func:`~engrava.domain.dreaming.default_signal_active`); an
    inactive (structurally flat) signal contributes the same constant to every
    candidate and so carries no ranking information. Its configured weight is
    dropped and the remainder renormalised over the active set, mirroring the
    dreaming scorer's ``_compute_active_weights`` — so the keep-score is
    meaningful on sparse stores instead of being dragged toward a constant.

    Only default signals participate: the hygiene weight vector is validated to
    hold recognised signal names, so an unknown name is treated as inactive
    (weight redistributed away) rather than raising — the fail-safe direction.

    **Pool-relative, once per run.** Activeness is decided over the whole
    candidate pool a single time; it is never recomputed per thought.

    Args:
        signal_weights: The configured hygiene keep-score weights.
        candidates: The candidate pool for this run.
        current_cycle: The run's cycle number (drives the cycle-based
            ``recency`` / ``staleness`` activeness).
        access_tracking_enabled: Whether access tracking is on (the
            ``frequency`` signal can only be active when it is).

    Returns:
        A ``(weights, flat_signals)`` pair. ``weights`` maps every configured
        signal name to its effective (renormalised) weight — ``0.0`` for
        inactive signals; the active entries sum to ``1.0`` unless the active
        set is empty (all-zero). ``flat_signals`` is the sorted list of
        configured signals found inactive this run.

    """
    active_names: list[str] = []
    flat_signals: list[str] = []
    for name in signal_weights:
        is_active = name in DEFAULT_SIGNALS and default_signal_active(
            name,
            candidates,
            current_cycle=current_cycle,
            access_tracking_enabled=access_tracking_enabled,
        )
        if is_active:
            active_names.append(name)
        else:
            flat_signals.append(name)

    active_weight = sum(signal_weights[name] for name in active_names)
    if active_weight == 0.0:
        # No active signal (or all active weights zero): keep-score is
        # undefined, so every effective weight is zero. The caller reads the
        # empty active set as the all-flat fail-safe and archives nothing.
        weights = dict.fromkeys(signal_weights, 0.0)
        return weights, sorted(flat_signals)

    weights = {
        name: (signal_weights[name] / active_weight if name in active_names else 0.0)
        for name in signal_weights
    }
    return weights, sorted(flat_signals)


USAGE_HISTORY_SIGNALS: tuple[str, ...] = ("frequency", "confirmation", "action_outcome")
"""The usage-history keep-signals that gate a hygiene run (the access-gate).

``frequency`` (reads), ``confirmation`` (reinforcements), and ``action_outcome``
(applied-outcome aggregates) are the only signals grounded in a thought actually
being *used*. Cycle-based signals (``recency`` / ``staleness``) and ``confidence``
are deliberately excluded: they are present on any store, so on a bulk import with
no usage history they would let ingest order alone drive eviction. Membership is
independent of the configured ``signal_weights`` — the access-gate asks whether the
pool holds *any* usage evidence, not whether a signal is weighted into the score.
"""


def has_active_usage_signal(
    candidates: list[ThoughtRecord],
    *,
    current_cycle: int,
    access_tracking_enabled: bool,
) -> bool:
    """Report whether any usage-history signal is active across the candidate pool.

    The run-level *access-gate*: a hygiene run may archive nothing unless at least
    one of the usage-history signals (:data:`USAGE_HISTORY_SIGNALS` —
    ``frequency`` / ``confirmation`` / ``action_outcome``) has a data source in the
    pool. Without any usage evidence a "cold" thought cannot be distinguished from
    one merely ingested early, so cycle-recency alone must not drive eviction. The
    per-signal activeness test is delegated to the shared
    :func:`~engrava.domain.dreaming.default_signal_active` predicate
    (the same one :func:`compute_active_hygiene_weights` uses) so the gate stays in
    lock-step with the scorer rather than re-deriving the test.

    Args:
        candidates: The candidate pool for this run.
        current_cycle: The run's cycle number, passed through to the shared
            predicate. The usage signals are not cycle-based, so it does not affect
            their result — it is threaded only to honour the predicate's signature.
        access_tracking_enabled: Whether access tracking is on. The ``frequency``
            signal can only be active when it is.

    Returns:
        ``True`` when at least one usage-history signal is active this run.

    """
    return any(
        default_signal_active(
            name,
            candidates,
            current_cycle=current_cycle,
            access_tracking_enabled=access_tracking_enabled,
        )
        for name in USAGE_HISTORY_SIGNALS
    )


def compute_keep_score(
    thought: ThoughtRecord,
    ctx: DreamingContext,
    active_weights: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """Compute a thought's keep-score as a weighted average of active signals.

    The score is ``Σ active-signal weight · signal(thought)`` over the signals
    with a non-zero effective weight (inactive signals have already been
    redistributed to ``0.0`` by :func:`compute_active_hygiene_weights`, so the
    effective weights sum to ``1.0`` and the score lands in ``[0.0, 1.0]``).

    Args:
        thought: The thought to score.
        ctx: The scoring context (current cycle, candidate-pool size).
        active_weights: The redistributed per-signal weights for this run.

    Returns:
        A ``(keep_score, per_signal)`` pair. ``per_signal`` maps each
        contributing signal name to its raw ``[0.0, 1.0]`` score (only signals
        with a non-zero effective weight), for the audit breakdown.

    """
    total = 0.0
    per_signal: dict[str, float] = {}
    for name, weight in active_weights.items():
        if weight == 0.0:
            continue
        signal_cls = DEFAULT_SIGNALS.get(name)
        if signal_cls is None:  # pragma: no cover - unknown names carry zero weight
            continue
        raw = signal_cls()(thought, ctx)
        per_signal[name] = raw
        total += raw * weight
    return total, per_signal

"""Derived-records extension seam — public protocol and generic types.

An extension can produce **N derived records** from a single stored thought
through this seam: implement :class:`DerivedRecordProducerProtocol` on the
hooks object and Engrava persists each returned :class:`DerivedRecord` as an
ordinary thought, after the source thought is durable, through a core-owned,
guarded, per-child lifecycle. The source-store control flow is never altered
and no store handle is ever handed to the producer — the producer describes
*what* to derive; core owns *how* it is persisted (identity, timestamps,
cycle, lifecycle status, and the single provenance edge).

Stability
---------
:class:`DerivedRecordProducerProtocol`, :class:`DerivedRecord`,
:class:`DeriveContext`, and :class:`DeriveGates` are public API and follow the
same ``X.Y.x`` compatibility guarantee as the rest of ``engrava``: no breaking
change (removed field, renamed field, tightened type, or changed default) lands
within a patch series. Any breaking change ships in a minor release and is
preceded by at least one minor of deprecation warnings. Producers may rely on
the field set documented here staying additive within ``0.6.x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engrava.domain.enums import Priority, ThoughtType
    from engrava.domain.models.thought import MetadataValue, ThoughtRecord


@dataclass(frozen=True)
class DerivedRecord:
    """A single record a producer derives from a stored source thought.

    A ``DerivedRecord`` carries **only producer-owned content fields**. The
    system-managed fields of the resulting thought — its content-hash identity
    (``thought_id``), timestamps, cognitive cycle, lifecycle status, and
    valid-time bounds — are **not part of this type**. They are structurally
    unrepresentable here, so there is nothing for a producer to set or forge:
    core assigns them at persist time (the identity is derived deterministically
    from ``content``, exactly like an ordinary thought's content hash), which is
    what makes re-running derivation idempotent.

    Attributes:
        content: Full derived content. Must be non-empty. Its deterministic
            content hash becomes the persisted thought's identity, so two derived
            records with byte-identical content collapse to a single stored
            thought. Core also derives the thought's compact ``essence`` from
            this content at persist time.
        thought_type: Classification of the derived thought.
        priority: Priority level assigned to the derived thought.
        metadata: Extensible structured attribute payload copied verbatim onto
            the persisted thought's ``metadata``. Same leaf-value rules as
            :class:`~engrava.domain.models.thought.ThoughtRecord.metadata`
            (scalars or nested string-keyed maps). Defaults to empty.
        attach_provenance_edge: When ``True`` (default) core attaches the single
            ``DERIVED_FROM`` edge from the derived thought to its source,
            recording content-level provenance. Set ``False`` to persist the
            derived thought without a provenance edge.

    Examples:
        >>> from engrava.domain.enums import Priority, ThoughtType
        >>> record = DerivedRecord(
        ...     content="First paragraph.",
        ...     thought_type=ThoughtType.OBSERVATION,
        ...     priority=Priority.P3,
        ... )
        >>> record.attach_provenance_edge
        True

    """

    content: str
    thought_type: ThoughtType
    priority: Priority
    metadata: dict[str, MetadataValue] = field(default_factory=dict)
    attach_provenance_edge: bool = True

    def __post_init__(self) -> None:
        """Validate that ``content`` is non-empty.

        Core derives the persisted thought's ``essence`` from ``content``; an
        empty content would produce an invalid (empty) essence, so it is
        rejected at construction.

        Raises:
            ValueError: When ``content`` is empty.

        """
        if not self.content:
            msg = "DerivedRecord.content must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class DeriveContext:
    """Stable, read-only context passed to :meth:`derive_records`.

    Exposes only immutable facts about the committed source thought plus a
    core-owned informational label. It deliberately carries **no store handle**:
    a producer cannot persist, query, or mutate anything through this context —
    persistence is entirely core-controlled.

    Attributes:
        source_thought_id: Identity of the committed source thought.
        source_content_hash: SHA-256 hex digest of the source thought's content
            (the same hash Engrava uses for content deduplication).
        cycle_at_derivation: The cognitive cycle observed on the source thought
            when derivation runs. Informational; core assigns the derived
            thoughts' own cycle from it.
        origin: Core-owned, purely informational label naming the write
            operation that triggered derivation (e.g. ``"create_thought"`` or
            ``"bulk_store"``). It is **never** used for recursion control or
            authorization — the recursion guard is a separate core mechanism.

    Examples:
        >>> ctx = DeriveContext(
        ...     source_thought_id="t-1",
        ...     source_content_hash="ab12",
        ...     cycle_at_derivation=0,
        ...     origin="create_thought",
        ... )
        >>> ctx.origin
        'create_thought'

    """

    source_thought_id: str
    source_content_hash: str
    cycle_at_derivation: int
    origin: str


@dataclass(frozen=True)
class DeriveGates:
    """Gates controlling the derived-records extension seam.

    The seam is inert unless ``enabled`` is ``True`` **and** the configured
    hooks object implements :class:`DerivedRecordProducerProtocol`. With
    ``enabled=False`` (the default) the source-store path produces byte-identical
    persisted results (DB + journal) to a store without the seam.

    Durability contract: derivation fires only on a **durably auto-committed**
    create — a normal ``create_thought`` (dispatched inline after its commit) or a
    ``bulk_store`` insert (dispatched after the batch commits). A create issued
    inside a caller-held ``suspend_auto_commit`` window does **not** auto-derive,
    because the caller owns that open transaction and the source is not yet
    durable; such a caller triggers derivation via an explicit re-run / backfill
    once its transaction has committed (recoverability, not automatic recovery). A
    dedup / hash hit never derives.

    Attributes:
        enabled: Master switch. ``False`` (default) ⇒ derivation never runs and
            the persisted results are byte-identical to a store without the seam.
        on_error: Failure policy for producer / child-persistence errors.
            ``"log"`` (default) records the failure with ordinary application
            logging and continues (best-effort — a failed child is left for a
            later re-run to fill); ``"raise"`` re-raises after the source is
            durable, aborting the remaining children. This is ordinary logging
            only — never a telemetry, benchmark, or audit surface.
        max_derived_per_source: Upper bound on how many derived records core
            consumes from a single source's returned sequence. Core reads at
            most ``max_derived_per_source + 1`` items and rejects an over-cap
            return (per ``on_error``) *before* any child is written, so a lazy
            or unbounded sequence cannot flood the store. Must be ``>= 1``.

    Examples:
        >>> gates = DeriveGates(enabled=True, max_derived_per_source=8)
        >>> gates.on_error
        'log'
        >>> DeriveGates().enabled
        False

    """

    enabled: bool = False
    on_error: Literal["raise", "log"] = "log"
    max_derived_per_source: int = 32

    def __post_init__(self) -> None:
        """Validate field invariants on construction.

        Enforces the same contract the YAML loader
        (:func:`~engrava.config._parse_derive`) applies, so a direct
        ``DeriveGates(...)`` call rejects the same malformed values a config file
        would — ``enabled`` must be a strict ``bool`` and, in particular, a
        ``bool`` cannot masquerade as an ``int`` for the cap (``bool`` is an
        ``int`` subclass).

        Raises:
            TypeError: When ``enabled`` is not a ``bool``, or
                ``max_derived_per_source`` is a ``bool`` or not an ``int``.
            ValueError: When ``on_error`` is not ``"raise"``/``"log"`` or
                ``max_derived_per_source`` is ``< 1``.

        """
        if not isinstance(self.enabled, bool):
            msg = "DeriveGates.enabled must be a bool"
            raise TypeError(msg)
        if self.on_error not in ("raise", "log"):
            msg = "DeriveGates.on_error must be 'raise' or 'log'"
            raise ValueError(msg)
        if isinstance(self.max_derived_per_source, bool) or not isinstance(
            self.max_derived_per_source,
            int,
        ):
            msg = "DeriveGates.max_derived_per_source must be an int"
            raise TypeError(msg)
        if self.max_derived_per_source < 1:
            msg = "DeriveGates.max_derived_per_source must be >= 1"
            raise ValueError(msg)


@runtime_checkable
class DerivedRecordProducerProtocol(Protocol):
    """Optional capability: derive N records from one stored source thought.

    A hooks object opts into the seam simply by implementing this method;
    Engrava detects the capability with ``isinstance(hooks,
    DerivedRecordProducerProtocol)``. It is a **separate** capability protocol —
    :class:`~engrava.domain.protocols.hooks.EngravaHooksProtocol` is unchanged,
    so an existing hooks implementation that does not derive records keeps
    working unchanged (byte-identical persisted results).

    Stability: this protocol is public API under the ``X.Y.x`` guarantee (see
    the module docstring).
    """

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,
    ) -> Sequence[DerivedRecord]:
        """Derive records from a committed source thought.

        Called by core **after** the source thought is durable and only when
        the seam is enabled. The producer must be pure with respect to the
        store — it may inspect ``thought`` and ``ctx`` and compute records, but
        it MUST NOT resolve or use any store handle (globals, service locator,
        or config), and MUST NOT spawn detached/background tasks. Persistence of
        the returned records is entirely core-controlled.

        For exact idempotency across re-runs the output should be a
        deterministic function of ``thought`` (a non-deterministic producer
        still converges but may add records over successive runs).

        Args:
            thought: The committed source thought (the record that was
                persisted, identical to the input of ``on_store``).
            ctx: Stable derivation context. Exposes no store handle.

        Returns:
            The derived records, in producer order. An empty sequence means
            "nothing to derive". Core consumes at most
            ``max_derived_per_source + 1`` items.

        """
        ...

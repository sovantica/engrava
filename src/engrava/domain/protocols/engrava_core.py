"""EngravaCoreProtocol — core persistence interface.

Defines the abstract interface for thought-graph CRUD, edge management,
embedding storage, and similarity search.  This is the **core** protocol
— no cognitive-layer-specific methods belong here (free-tier boundary).

The read (non-mutating) half lives in
:class:`~engrava.domain.protocols.engrava_read.EngravaReadProtocol`, which this
protocol extends; the methods declared *here* are the writable capabilities
(persist, update, delete, and the derived-records write entry point).  A full
store satisfies both by construction.  A read-only view
(:class:`~engrava.infrastructure.read_only_store.ReadOnlyEngrava`) delegates the
``EngravaReadProtocol`` reads and typed-rejects every write; because it must
still expose the write method *names* in order to reject them, it also
structurally satisfies ``EngravaCoreProtocol`` — so its read-only guarantee is
**behavioural** (a write call raises ``ReadOnlyViolationError``), not a
structural exclusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from engrava.domain.protocols.engrava_read import EngravaReadProtocol

if TYPE_CHECKING:
    from engrava.domain.enums import (
        ActionStatus,
        VerificationStatus,
    )
    from engrava.domain.models.action import ActionRecord
    from engrava.domain.models.edge import EdgeRecord
    from engrava.domain.models.embedding import EmbeddingRecord
    from engrava.domain.models.thought import MetadataValue, ThoughtRecord
    from engrava.domain.models.ttl import CleanupResult
    from engrava.domain.protocols.derived_records import DeriveResult


@runtime_checkable
class EngravaCoreProtocol(EngravaReadProtocol, Protocol):
    """Abstract persistence interface for core thought-graph data.

    Implementations must provide async CRUD for thoughts, edges,
    and embedding-based search.  Extends
    :class:`~engrava.domain.protocols.engrava_read.EngravaReadProtocol` with the
    writable capabilities; the read surface is inherited unchanged.
    """

    async def create_thought(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
        deduplicate: bool = False,
    ) -> ThoughtRecord:
        """Persist a new thought record.

        Args:
            thought: The thought record to create.
            expires_after_seconds: Optional TTL in seconds.  When provided,
                ``expires_at`` is computed as *now + seconds*.  Takes
                precedence over the store's ``default_ttl_seconds``.
            deduplicate: When ``True`` and an existing thought with the
                same SHA-256 ``content_hash`` already exists,
                implementations should increment its
                ``confirmation_count`` and return the existing record
                rather than inserting a duplicate.  Default ``False``
                preserves create-on-every-call semantics so existing
                callers are unaffected.  Implementations that cannot
                support deduplication MUST raise on ``deduplicate=True``
                rather than silently ignoring it.

        Returns:
            The persisted thought record (or the existing record with a
            bumped ``confirmation_count`` when deduplication hits).

        Raises:
            ValueError: If a thought with the same ID already exists.

        """
        ...

    async def get_or_create(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
    ) -> tuple[ThoughtRecord, bool]:
        """Fetch an existing thought by content hash, or create it.

        A convenience over content-hash deduplication that returns whether it
        created, eliminating the caller's check-then-create round trip. On a
        hash hit the existing record is returned with ``created=False`` (its
        ``confirmation_count`` bumped, identical to
        ``create_thought(deduplicate=True)``); on a miss a new row is inserted
        and returned with ``created=True``. The matched row's mutable fields are
        not altered from ``thought`` — use :meth:`upsert_by_hash` for that.

        Args:
            thought: The candidate thought (its ``content`` supplies the hash).
            expires_after_seconds: Optional relative TTL applied only on create.

        Returns:
            A ``(record, created)`` tuple; ``created`` is ``True`` on insert.

        """
        ...

    async def upsert_by_hash(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
    ) -> ThoughtRecord:
        """Insert a thought, or update the matching row's mutable fields in place.

        Update-on-match content-hash upsert, distinct from
        ``create_thought(deduplicate=True)``: on a hash hit the stored row's
        mutable fields (``essence``, ``priority``, ``metadata``, ``visibility``,
        ``lifecycle_status``, ``source``, ``confidence``, ``source_type``,
        ``thought_type``) are overwritten from ``thought`` and the updated
        record returned — ``confirmation_count`` is *not* bumped. ``content`` is
        never rewritten (it is the hash key). A miss delegates to
        :meth:`create_thought`.

        Args:
            thought: The desired thought state (its ``content`` supplies the
                hash; its mutable fields are copied onto a matched row).
            expires_after_seconds: Optional relative TTL applied only on insert.

        Returns:
            The freshly inserted row on a miss, or the updated existing row on a
            hit.

        """
        ...

    async def bulk_store(
        self,
        thoughts: list[ThoughtRecord],
        *,
        deduplicate: bool = False,
    ) -> list[ThoughtRecord]:
        """Persist many thoughts in one all-or-nothing transaction.

        Batch analogue of :meth:`create_thought`: the whole loop commits once
        (not per row) and is transactional — if any row raises, the entire batch
        is rolled back and nothing is persisted. The returned list is in input
        order. When auto-embed is active, all inserted thoughts are embedded in
        a single batch provider call, producing vectors byte-identical to
        per-thought embedding.

        Args:
            thoughts: The thoughts to persist, in order (empty list ⇒ ``[]``).
            deduplicate: Applied per row like ``create_thought(deduplicate=True)``.

        Returns:
            The persisted records in input order.

        """
        ...

    async def remember(
        self,
        text: str,
        *,
        metadata: dict[str, MetadataValue] | None = None,
        deduplicate: bool = False,
    ) -> ThoughtRecord:
        """Store a string as a thought with one call.

        Ergonomic shorthand over :meth:`create_thought` for the common
        case of persisting a bare string: implementations build a
        :class:`ThoughtRecord` (deriving ``essence`` from the opening of
        ``text``) and delegate to ``create_thought``.

        The thought is created at the store's default cognitive cycle
        (cycle ``0``); callers that track cycles should build a
        :class:`ThoughtRecord` explicitly and use ``create_thought``.

        Args:
            text: The content to remember.  Becomes the thought's
                ``content``; the opening is also used as its ``essence``.
            metadata: Optional structured attributes (e.g. ``speaker``,
                ``lang``, ``session_id``).  Defaults to an empty mapping.
            deduplicate: When ``True`` and a thought with byte-identical
                ``content`` already exists, its ``confirmation_count`` is
                incremented and the existing record is returned instead of
                inserting a duplicate (delegated to
                ``create_thought(deduplicate=True)``).

        Returns:
            The persisted thought record (or the existing record with a
            bumped ``confirmation_count`` when deduplication hits).

        """
        ...

    async def update_thought(self, thought_id: str, **changes: object) -> ThoughtRecord:
        """Update a thought with optimistic concurrency.

        Args:
            thought_id: UUID of the thought to update.
            **changes: Fields to update.

        Returns:
            The updated thought record.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.
            StaleDataError: If the row was modified by another writer.

        """
        ...

    async def restore_thought(
        self, thought_id: str, *, current_cycle: int | None = None
    ) -> ThoughtRecord:
        """Restore an archived thought to ``ACTIVE``, clearing its archive stamp.

        Args:
            thought_id: UUID of the archived thought to restore.
            current_cycle: Optional cycle to stamp as the new ``updated_cycle``.

        Returns:
            The restored thought record (``lifecycle_status`` is ``ACTIVE``).

        Raises:
            ThoughtNotFoundError: If the thought does not exist.
            InvalidTransitionError: If the thought is not currently ``ARCHIVED``.
            StaleDataError: If the row was modified by another writer.

        """
        ...

    async def cleanup_expired(
        self,
        now: str | None = None,
        *,
        exclude_id: str | None = None,
    ) -> CleanupResult:
        """Remove or archive thoughts whose ``expires_at`` is in the past.

        Args:
            now: Optional ISO-8601 timestamp for deterministic testing.
            exclude_id: Optional thought ID to skip during cleanup.

        Returns:
            A ``CleanupResult`` with count and strategy applied.

        """
        ...

    async def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought by its ID.

        Args:
            thought_id: UUID of the thought to delete.

        Returns:
            True if the thought was deleted, False if not found.

        """
        ...

    async def record_access(self, thought_id: str) -> None:
        """Record an explicit access to a thought.

        Increments ``access_count`` by 1 and sets ``last_accessed_at``
        to the current UTC time.  This is an **explicit** operation —
        ``get_thought()`` and ``list_thoughts()`` do NOT auto-increment.
        The consumer decides when an access is semantically meaningful.

        Args:
            thought_id: UUID of the thought to mark as accessed.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.

        """
        ...

    async def create_edge(self, edge: EdgeRecord) -> EdgeRecord:
        """Persist a new edge record.

        Args:
            edge: The edge record to create.

        Returns:
            The persisted edge record.

        """
        ...

    async def update_edge(self, edge_id: str, **changes: object) -> EdgeRecord:
        """Update an edge by its ID.

        Args:
            edge_id: UUID of the edge to update.
            **changes: Fields to update.

        Returns:
            The updated edge record.

        Raises:
            ValueError: If the edge does not exist.

        """
        ...

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by its ID.

        Args:
            edge_id: UUID of the edge to delete.

        Returns:
            True if the edge was deleted, False if not found.

        """
        ...

    async def store_embedding(
        self,
        thought_id: str,
        vector: list[float],
        *,
        model_name: str = "all-MiniLM-L12-v2",
        embedding_id: str | None = None,
    ) -> EmbeddingRecord:
        """Store an embedding vector for a thought.

        Args:
            thought_id: UUID of the thought that owns this embedding.
            vector: The embedding vector.
            model_name: Name of the embedding model.
            embedding_id: Optional explicit ID; generated if omitted.

        Returns:
            The persisted EmbeddingRecord.

        """
        ...

    async def create_action(self, action: ActionRecord) -> ActionRecord:
        """Persist a new action record.

        Args:
            action: The action record to create.

        Returns:
            The persisted action record.

        """
        ...

    async def update_action(
        self,
        action_id: str,
        *,
        status: ActionStatus | None = None,
        verification_status: VerificationStatus | None = None,
    ) -> ActionRecord:
        """Advance a stored action's status and/or verification status.

        Args:
            action_id: UUID of the action to update.
            status: New status, or ``None`` to leave the status unchanged.
            verification_status: New verification status, or ``None`` to
                leave it unchanged.

        Returns:
            The updated (or, for a no-op, the unchanged) action record.

        Raises:
            ActionNotFoundError: If the action does not exist.
            InvalidTransitionError: If a real ``status`` change is illegal.

        """
        ...

    async def derive_existing(self, thought_id: str) -> DeriveResult:
        """Run the registered derived-records producer over a stored thought.

        The explicit backfill counterpart of the automatic on-store derived-
        records trigger: for an already-stored source thought, invoke the
        configured producer capability and persist every returned child through
        the **same** core-owned per-child lifecycle the on-store path uses
        (content-addressed identity, per-child commit, auto-embed, conflict-as-
        reuse, and the single ``DERIVED_FROM`` provenance edge). It exists so a
        base built *before* a producer was installed can be enriched
        retroactively, and because backfilled children share the on-store path's
        exact identity they **converge** with it (deduped, idempotent).

        Gating: this runs whenever a producer capability is present, honouring
        ``DeriveGates.on_error`` and ``max_derived_per_source``, but
        **independently of** ``DeriveGates.enabled`` (the master switch that
        governs only the automatic on-store trigger) — so an existing base can
        be backfilled without committing to automatic derivation on every future
        write. When no producer capability is registered it is a clean no-op.

        Idempotent by construction: because derived-child identity is content-
        addressed and insertion is conflict-as-reuse, re-running it converges
        (already-present children are reused, missing ones filled). A source that
        is itself a derived record is never re-derived (a clean, empty result).

        Stability: additive public method under the ``X.Y.x`` guarantee — it is a
        write entry point, so it belongs to the writable core, never a read-only
        view.

        Args:
            thought_id: The already-stored source thought to derive from.

        Returns:
            A :class:`~engrava.domain.protocols.derived_records.DeriveResult`
            tallying children created / reused / skipped for this run.

        Raises:
            SourceThoughtNotFoundError: If ``thought_id`` does not exist.
            DerivedRecordError: If the producer's return violates the seam's
                deterministic contract (over cap, or an identity collision) and
                ``DeriveGates.on_error="raise"``.

        """
        ...

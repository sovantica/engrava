"""ReadOnlyEngrava — write-blocking wrapper for EngravaCoreProtocol.

Wraps any ``EngravaCoreProtocol`` implementation and delegates all read
operations while raising ``ReadOnlyViolationError`` on any write attempt.
Used in processing layers that must be strictly read-only (e.g., in
read-only contexts where write access must be prevented).

Design:
    Composition over inheritance — receives ``EngravaCoreProtocol`` in
    the constructor and forwards 6 read methods.  Write methods (10) raise
    ``ReadOnlyViolationError`` unconditionally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from engrava.domain.exceptions import ReadOnlyViolationError

if TYPE_CHECKING:
    from engrava.domain.enums import ActionStatus, VerificationStatus
    from engrava.domain.models.action import ActionRecord
    from engrava.domain.models.edge import EdgeRecord
    from engrava.domain.models.embedding import EmbeddingRecord
    from engrava.domain.models.metrics import EngravaMetrics
    from engrava.domain.models.thought import ThoughtRecord
    from engrava.domain.protocols.engrava_core import EngravaCoreProtocol


class ReadOnlyEngrava:
    """Write-blocking wrapper that enforces read-only access to an Engrava store.

    Delegates all read operations to the wrapped ``EngravaCoreProtocol``
    implementation.  Any call to a write method raises
    ``ReadOnlyViolationError``.

    Args:
        inner: The Engrava implementation to wrap.

    Examples:
        >>> ro_store = ReadOnlyEngrava(sqlite_store)
        >>> thought = await ro_store.get_thought("abc-123")  # OK
        >>> await ro_store.create_thought(thought)  # raises ReadOnlyViolationError

    """

    def __init__(self, inner: EngravaCoreProtocol) -> None:
        self._inner = inner

    # ── Read operations (delegated) ──────────────────────────────────

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Retrieve a thought by its ID.

        Args:
            thought_id: UUID of the thought to retrieve.

        Returns:
            The thought record, or None if not found.

        """
        return await self._inner.get_thought(thought_id)

    async def list_thoughts(
        self,
        *,
        priority: str | None = None,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        min_cycle: int | None = None,
        max_cycle: int | None = None,
        visibility: str | None = None,
        exclude_visibility: str | None = None,
        include_expired: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThoughtRecord]:
        """List thoughts matching the given filters.

        Args:
            priority: Filter by priority level.
            lifecycle_status: Filter by lifecycle status.
            thought_type: Filter by thought type.
            min_cycle: Minimum updated_cycle (inclusive).
            max_cycle: Maximum updated_cycle (inclusive).
            visibility: Include only thoughts with this visibility.
            exclude_visibility: Exclude thoughts with this visibility.
            include_expired: If True, include expired thoughts.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of matching thought records.

        """
        return await self._inner.list_thoughts(
            priority=priority,
            lifecycle_status=lifecycle_status,
            thought_type=thought_type,
            min_cycle=min_cycle,
            max_cycle=max_cycle,
            visibility=visibility,
            exclude_visibility=exclude_visibility,
            include_expired=include_expired,
            limit=limit,
            offset=offset,
        )

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search for thoughts similar to the query vector.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity score.

        Returns:
            List of (thought_id, similarity_score) tuples, sorted descending.

        """
        return await self._inner.search_similar(
            query_vector,
            top_k=top_k,
            threshold=threshold,
        )

    async def get_edges(
        self,
        thought_id: str,
        *,
        direction: str = "BOTH",
    ) -> list[EdgeRecord]:
        """Retrieve edges connected to a thought.

        Args:
            thought_id: UUID of the thought.
            direction: 'IN', 'OUT', or 'BOTH'.

        Returns:
            List of matching edge records.

        """
        return await self._inner.get_edges(thought_id, direction=direction)

    async def get_embedding(self, thought_id: str) -> EmbeddingRecord | None:
        """Retrieve the embedding for a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The EmbeddingRecord, or None if not found.

        """
        return await self._inner.get_embedding(thought_id)

    async def get_actions(self, thought_id: str) -> list[ActionRecord]:
        """Retrieve actions linked to a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            List of action records.

        """
        return await self._inner.get_actions(thought_id)

    async def metrics(self) -> EngravaMetrics:
        """Return a point-in-time metrics snapshot from the wrapped store."""
        return await self._inner.metrics()

    async def max_cycle(self) -> int:
        """Return the wrapped store's cognitive-cycle high-water mark.

        A read-only recovery accessor — ``MAX(thought.updated_cycle)`` unioned
        with ``MAX(edge.created_cycle)``, or ``0`` on an empty store — delegated
        verbatim to the wrapped store.

        Returns:
            The maximum cognitive cycle stored, or ``0`` when the store holds no
            cycle-bearing records.

        """
        return await self._inner.max_cycle()

    # ── Write operations (blocked) ───────────────────────────────────

    async def create_thought(
        self,
        _thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,  # noqa: ARG002
        deduplicate: bool = False,  # noqa: ARG002 -- signature parity with EngravaCoreProtocol
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Accepts every keyword the live ``SqliteEngravaCore.create_thought``
        accepts (currently ``expires_after_seconds`` and ``deduplicate``)
        so callers passing them through a read-only adapter still get a
        coherent ``ReadOnlyViolationError`` instead of an opaque
        ``TypeError`` from argparse-style signature mismatch.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "create_thought"
        raise ReadOnlyViolationError(msg)

    async def get_or_create(
        self,
        _thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,  # noqa: ARG002 -- signature parity
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Accepts the same keywords as ``SqliteEngravaCore.get_or_create`` so a
        pass-through call raises a coherent ``ReadOnlyViolationError`` rather
        than an opaque ``TypeError``.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "get_or_create"
        raise ReadOnlyViolationError(msg)

    async def upsert_by_hash(
        self,
        _thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,  # noqa: ARG002 -- signature parity
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Accepts the same keywords as ``SqliteEngravaCore.upsert_by_hash`` so a
        pass-through call raises a coherent ``ReadOnlyViolationError`` rather
        than an opaque ``TypeError``.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "upsert_by_hash"
        raise ReadOnlyViolationError(msg)

    async def bulk_store(
        self,
        _thoughts: list[ThoughtRecord],
        *,
        deduplicate: bool = False,  # noqa: ARG002 -- signature parity
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Accepts the same keywords as ``SqliteEngravaCore.bulk_store`` so a
        pass-through call raises a coherent ``ReadOnlyViolationError`` rather
        than an opaque ``TypeError``.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "bulk_store"
        raise ReadOnlyViolationError(msg)

    async def update_thought(self, _thought_id: str, **_changes: object) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "update_thought"
        raise ReadOnlyViolationError(msg)

    async def delete_thought(self, _thought_id: str) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "delete_thought"
        raise ReadOnlyViolationError(msg)

    async def create_edge(self, _edge: EdgeRecord) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "create_edge"
        raise ReadOnlyViolationError(msg)

    async def delete_edge(self, _edge_id: str) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "delete_edge"
        raise ReadOnlyViolationError(msg)

    async def update_edge(self, _edge_id: str, **_changes: object) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "update_edge"
        raise ReadOnlyViolationError(msg)

    async def store_embedding(
        self,
        _thought_id: str,
        _vector: list[float],
        *,
        model_name: str = "all-MiniLM-L12-v2",  # noqa: ARG002
        embedding_id: str | None = None,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "store_embedding"
        raise ReadOnlyViolationError(msg)

    async def create_action(self, _action: ActionRecord) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "create_action"
        raise ReadOnlyViolationError(msg)

    async def update_action(
        self,
        _action_id: str,
        *,
        status: ActionStatus | None = None,  # noqa: ARG002
        verification_status: VerificationStatus | None = None,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "update_action"
        raise ReadOnlyViolationError(msg)

    async def cleanup_expired(
        self,
        now: str | None = None,  # noqa: ARG002
        *,
        exclude_id: str | None = None,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "cleanup_expired"
        raise ReadOnlyViolationError(msg)

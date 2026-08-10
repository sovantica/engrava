"""ReadOnlyEngrava — write-blocking view over an Engrava store.

Wraps an Engrava store and re-exposes its read surface while typed-rejecting
every write.  The view **implements**
:class:`~engrava.domain.protocols.engrava_read.EngravaReadProtocol` for reads
(so ``isinstance(view, EngravaReadProtocol)`` holds and the delegated read set is
type-checked against the protocol, never able to silently drift), and it defines
a real, signature-compatible method for every writable capability whose sole
behaviour is to raise
:class:`~engrava.domain.exceptions.ReadOnlyViolationError`.

The read-only guarantee is therefore **behavioural, not structural**: because the
write blockers are real methods, the view structurally presents all core method
names too (``isinstance(view, EngravaCoreProtocol)`` is ``True`` — a
``runtime_checkable`` protocol matches on member *names*), so the write methods
must exist precisely so a write call binds and then raises the typed error
instead of an ``AttributeError``.  What makes the view read-only is that every
write *raises* at call time, not that the names are absent.

Access-tracking policy:
    Some retrieval methods on a concrete backend (e.g. ``get_thought`` and the
    hybrid-search paths on ``SqliteEngravaCore``) buffer a deferred, best-effort
    access-frequency write — a mutation that a later flush would apply.  A
    read-only view must not cause *any* mutation, even a deferred one, so **every
    read delegated through this view runs inside the wrapped store's
    ``suppress_access_tracking`` block**.  That capability is a hard requirement:
    the constructor rejects a store that cannot make its reads side-effect free,
    so the guarantee is enforced rather than best-effort.  Reads never feed the
    access-frequency signal or stage an ``access_count`` write; explicit access
    recording remains available only through the blocked :meth:`record_access`
    write path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Protocol, runtime_checkable

from engrava.domain.exceptions import ReadOnlyViolationError
from engrava.domain.protocols.engrava_read import EngravaReadProtocol

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Sequence

    from engrava.domain.enums import (
        ActionStatus,
        EdgeType,
        KnowledgeSource,
        VerificationStatus,
    )
    from engrava.domain.models.action import ActionRecord
    from engrava.domain.models.edge import EdgeRecord
    from engrava.domain.models.embedding import EmbeddingRecord
    from engrava.domain.models.filters import MetadataFilter, VisibilityQueryFilter
    from engrava.domain.models.metrics import EngravaMetrics
    from engrava.domain.models.search import HybridSearchResult
    from engrava.domain.models.thought import MetadataValue, ThoughtRecord


@runtime_checkable
class _SuppressibleReadStore(EngravaReadProtocol, Protocol):
    """A read store that can also make its reads side-effect free.

    The backend contract ``ReadOnlyEngrava`` requires: the full read surface of
    :class:`~engrava.domain.protocols.engrava_read.EngravaReadProtocol` **plus** a
    ``suppress_access_tracking`` async context manager that keeps reads inside its
    block out of any deferred access-frequency write.  A store that cannot offer
    this cannot be wrapped read-only, because a read through the view could
    otherwise stage a mutation.
    """

    def suppress_access_tracking(self) -> contextlib.AbstractAsyncContextManager[None]:
        """Return a context manager that suppresses access buffering for its block."""
        ...


class ReadOnlyEngrava(EngravaReadProtocol):
    """Write-blocking view that enforces read-only access to an Engrava store.

    Delegates every read operation to the wrapped store and raises
    :class:`~engrava.domain.exceptions.ReadOnlyViolationError` on every write.
    Implements
    :class:`~engrava.domain.protocols.engrava_read.EngravaReadProtocol` for reads
    (verified at type-check time; ``isinstance(view, EngravaReadProtocol)`` holds
    at runtime) and typed-rejects every write at call time — the read-only
    guarantee is behavioural (writes raise), not structural exclusion of the write
    method names (see the module docstring).

    Reads are delegated inside the wrapped store's ``suppress_access_tracking``
    block, so a read through this view never stages an access-frequency write.
    The constructor requires a store that offers that capability.

    Args:
        inner: The Engrava store to wrap.  Must expose the
            :class:`~engrava.domain.protocols.engrava_read.EngravaReadProtocol`
            read surface *and* ``suppress_access_tracking`` (every core store
            does); otherwise a read-only guarantee cannot be enforced.

    Raises:
        TypeError: If ``inner`` cannot make its reads side-effect free (no
            ``suppress_access_tracking`` capability).

    Examples:
        >>> ro_store = ReadOnlyEngrava(sqlite_store)
        >>> thought = await ro_store.get_thought("abc-123")  # OK
        >>> await ro_store.create_thought(thought)  # raises ReadOnlyViolationError

    """

    def __init__(self, inner: _SuppressibleReadStore) -> None:
        if not isinstance(inner, _SuppressibleReadStore):
            msg = (
                "ReadOnlyEngrava requires a store whose reads can be made "
                "side-effect free via a suppress_access_tracking() context "
                f"manager; {type(inner).__name__} does not provide one, so a "
                "read-only guarantee cannot be enforced over it."
            )
            raise TypeError(msg)
        self._inner = inner

    def _reading(self) -> contextlib.AbstractAsyncContextManager[None]:
        """Return the read-only delegation context for a single read.

        Suppresses the wrapped store's access tracking for the duration of the
        delegated call, so no read through this view feeds the access-frequency
        signal or stages a deferred access-count write.

        Returns:
            An async context manager to wrap the delegated read call.

        """
        return self._inner.suppress_access_tracking()

    # ── Read operations (delegated, access tracking suppressed) ──────

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        current_cycle: int | None = None,
        recency_now: str | None = None,
        recency_now_half_life: int | None = None,
        filters: MetadataFilter | None = None,
        visibility: VisibilityQueryFilter | None = None,
        collapse_key: str | Sequence[str] | None = None,
        collapse_max_per_unit: int | None = None,
        include_archived: bool = False,
    ) -> HybridSearchResult:
        """Retrieve thoughts relevant to a query (delegated read).

        Args:
            query: Natural-language text to search for.
            top_k: Maximum number of results to return.
            current_cycle: Optional cognitive-cycle recency reference.
            recency_now: Optional transaction-time recency instant (ISO-8601).
            recency_now_half_life: Optional transaction-time half-life override.
            filters: Optional metadata filter forwarded verbatim.
            visibility: Optional bounded visibility query filter.
            collapse_key: Optional de-fragmentation unit key.
            collapse_max_per_unit: Optional intra-unit retention depth.
            include_archived: When ``True``, re-admit archived thoughts.

        Returns:
            A ``HybridSearchResult`` with the ranked matches and contributing
            backends.

        """
        async with self._reading():
            return await self._inner.recall(
                query,
                top_k=top_k,
                current_cycle=current_cycle,
                recency_now=recency_now,
                recency_now_half_life=recency_now_half_life,
                filters=filters,
                visibility=visibility,
                collapse_key=collapse_key,
                collapse_max_per_unit=collapse_max_per_unit,
                include_archived=include_archived,
            )

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Retrieve a thought by its ID.

        Args:
            thought_id: UUID of the thought to retrieve.

        Returns:
            The thought record, or None if not found.

        """
        async with self._reading():
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
        provenance_filter: MetadataFilter | None = None,
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
            provenance_filter: Optional typed filter over the ``provenance``
                JSON column, forwarded verbatim.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of matching thought records.

        """
        async with self._reading():
            return await self._inner.list_thoughts(
                priority=priority,
                lifecycle_status=lifecycle_status,
                thought_type=thought_type,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
                visibility=visibility,
                exclude_visibility=exclude_visibility,
                include_expired=include_expired,
                provenance_filter=provenance_filter,
                limit=limit,
                offset=offset,
            )

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
        *,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Search for thoughts similar to the query vector.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity score.
            include_archived: When ``False`` (default) archived thoughts are
                excluded; when ``True`` they are re-admitted for this call.
                Forwarded verbatim to the wrapped store.

        Returns:
            List of (thought_id, similarity_score) tuples, sorted descending.

        """
        async with self._reading():
            return await self._inner.search_similar(
                query_vector,
                top_k=top_k,
                threshold=threshold,
                include_archived=include_archived,
            )

    async def search_fts(
        self,
        query: str,
        top_k: int = 10,
        *,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Full-text search using SQLite FTS5 with BM25 ranking.

        Args:
            query: Search query string (FTS5 syntax supported).
            top_k: Maximum number of results.
            include_archived: When ``True``, re-admit archived thoughts.

        Returns:
            List of ``(thought_id, bm25_score)`` tuples sorted by relevance;
            empty when the FTS5 index is unavailable.

        """
        async with self._reading():
            return await self._inner.search_fts(
                query,
                top_k,
                include_archived=include_archived,
            )

    async def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        *,
        top_k: int = 10,
        fts_weight: float | None = None,
        vector_weight: float | None = None,
        recency_weight: float | None = None,
        recency_half_life: int | None = None,
        current_cycle: int | None = None,
        recency_now: str | None = None,
        recency_now_half_life: int | None = None,
        fts_top_k: int = 50,
        vector_top_k: int = 50,
        filters: MetadataFilter | None = None,
        visibility: VisibilityQueryFilter | None = None,
        collapse_key: str | Sequence[str] | None = None,
        collapse_max_per_unit: int | None = None,
        include_archived: bool = False,
    ) -> HybridSearchResult:
        """Hybrid search combining FTS5 keyword + vector cosine similarity.

        Args:
            query_text: Text query for FTS5 keyword search.
            query_vector: Optional embedding vector for similarity search.
            top_k: Maximum number of final merged results.
            fts_weight: Optional FTS5 fusion-weight override.
            vector_weight: Optional vector fusion-weight override.
            recency_weight: Optional recency fusion-weight override.
            recency_half_life: Optional cognitive-cycle recency half-life.
            current_cycle: Optional cognitive-cycle recency reference.
            recency_now: Optional transaction-time recency instant (ISO-8601).
            recency_now_half_life: Optional transaction-time half-life override.
            fts_top_k: Max candidates from FTS5 before fusion.
            vector_top_k: Max candidates from vector search before fusion.
            filters: Optional metadata filter forwarded verbatim.
            visibility: Optional bounded visibility query filter.
            collapse_key: Optional de-fragmentation unit key.
            collapse_max_per_unit: Optional intra-unit retention depth.
            include_archived: When ``True``, re-admit archived thoughts.

        Returns:
            A ``HybridSearchResult`` with ranked results and contributing
            backends.

        """
        async with self._reading():
            return await self._inner.search_hybrid(
                query_text,
                query_vector,
                top_k=top_k,
                fts_weight=fts_weight,
                vector_weight=vector_weight,
                recency_weight=recency_weight,
                recency_half_life=recency_half_life,
                current_cycle=current_cycle,
                recency_now=recency_now,
                recency_now_half_life=recency_now_half_life,
                fts_top_k=fts_top_k,
                vector_top_k=vector_top_k,
                filters=filters,
                visibility=visibility,
                collapse_key=collapse_key,
                collapse_max_per_unit=collapse_max_per_unit,
                include_archived=include_archived,
            )

    async def metrics(self) -> EngravaMetrics:
        """Return a point-in-time metrics snapshot from the wrapped store."""
        async with self._reading():
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
        async with self._reading():
            return await self._inner.max_cycle()

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
        async with self._reading():
            return await self._inner.get_edges(thought_id, direction=direction)

    async def list_edges(
        self,
        *,
        edge_type: EdgeType | None = None,
        source: KnowledgeSource | None = None,
        filters: MetadataFilter | None = None,
        limit: int = 5000,
    ) -> list[EdgeRecord]:
        """List edges matching optional filters.

        Args:
            edge_type: If given, restrict to this edge type.
            source: If given, restrict to this knowledge source.
            filters: Optional typed ``AND`` filter over the edge
                ``metadata_json`` column, forwarded verbatim.
            limit: Maximum number of edges to return.

        Returns:
            List of matching edge records, ordered by ``created_cycle`` DESC.

        """
        async with self._reading():
            return await self._inner.list_edges(
                edge_type=edge_type,
                source=source,
                filters=filters,
                limit=limit,
            )

    async def get_embedding(self, thought_id: str) -> EmbeddingRecord | None:
        """Retrieve the embedding for a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The EmbeddingRecord, or None if not found.

        """
        async with self._reading():
            return await self._inner.get_embedding(thought_id)

    async def get_actions(self, thought_id: str) -> list[ActionRecord]:
        """Retrieve actions linked to a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            List of action records.

        """
        async with self._reading():
            return await self._inner.get_actions(thought_id)

    # ── Write operations (blocked) ───────────────────────────────────
    #
    # Each blocker keeps the wrapped store's real parameter names (not
    # underscore-renamed) so both positional and keyword calls bind and then
    # raise ``ReadOnlyViolationError`` — a keyword call like ``remember(text=...)``
    # must not fail at argument binding with ``TypeError`` before reaching the
    # body.  The intentionally-unused arguments are silenced per-parameter with an
    # ARG002 lint suppression.

    async def create_thought(
        self,
        thought: ThoughtRecord,  # noqa: ARG002
        *,
        expires_after_seconds: int | None = None,  # noqa: ARG002
        deduplicate: bool = False,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Accepts every keyword the live ``SqliteEngravaCore.create_thought``
        accepts (currently ``expires_after_seconds`` and ``deduplicate``) so
        callers passing them through a read-only view — positionally or by
        keyword — still get a coherent ``ReadOnlyViolationError`` instead of an
        opaque ``TypeError`` from a signature mismatch.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "create_thought"
        raise ReadOnlyViolationError(msg)

    async def get_or_create(
        self,
        thought: ThoughtRecord,  # noqa: ARG002
        *,
        expires_after_seconds: int | None = None,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "get_or_create"
        raise ReadOnlyViolationError(msg)

    async def upsert_by_hash(
        self,
        thought: ThoughtRecord,  # noqa: ARG002
        *,
        expires_after_seconds: int | None = None,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "upsert_by_hash"
        raise ReadOnlyViolationError(msg)

    async def bulk_store(
        self,
        thoughts: list[ThoughtRecord],  # noqa: ARG002
        *,
        deduplicate: bool = False,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "bulk_store"
        raise ReadOnlyViolationError(msg)

    async def remember(
        self,
        text: str,  # noqa: ARG002
        *,
        metadata: dict[str, MetadataValue] | None = None,  # noqa: ARG002
        deduplicate: bool = False,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        ``remember`` writes a new thought, so it is rejected; the wrapped store's
        keywords are accepted so a positional or keyword pass-through raises a
        coherent ``ReadOnlyViolationError`` rather than an opaque ``TypeError``.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "remember"
        raise ReadOnlyViolationError(msg)

    async def update_thought(self, thought_id: str, **changes: object) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "update_thought"
        raise ReadOnlyViolationError(msg)

    async def restore_thought(
        self,
        thought_id: str,  # noqa: ARG002
        *,
        current_cycle: int | None = None,  # noqa: ARG002
    ) -> NoReturn:
        """Blocked: read-only contexts do not allow writes.

        Restoring an archived thought is a lifecycle mutation; the wrapped
        store's keywords are accepted so a pass-through call raises a coherent
        ``ReadOnlyViolationError`` rather than an opaque ``TypeError``.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "restore_thought"
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

    async def delete_thought(self, thought_id: str) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "delete_thought"
        raise ReadOnlyViolationError(msg)

    async def record_access(self, thought_id: str) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        Recording an access mutates ``access_count`` / ``last_accessed_at``, so
        it is a write even though it targets telemetry rather than content.
        Delegated reads suppress the wrapped store's *implicit* access tracking;
        this *explicit* entry point is likewise blocked.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "record_access"
        raise ReadOnlyViolationError(msg)

    async def create_edge(self, edge: EdgeRecord) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "create_edge"
        raise ReadOnlyViolationError(msg)

    async def update_edge(self, edge_id: str, **changes: object) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "update_edge"
        raise ReadOnlyViolationError(msg)

    async def delete_edge(self, edge_id: str) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "delete_edge"
        raise ReadOnlyViolationError(msg)

    async def store_embedding(
        self,
        thought_id: str,  # noqa: ARG002
        vector: list[float],  # noqa: ARG002
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

    async def create_action(self, action: ActionRecord) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "create_action"
        raise ReadOnlyViolationError(msg)

    async def update_action(
        self,
        action_id: str,  # noqa: ARG002
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

    async def derive_existing(self, thought_id: str) -> NoReturn:  # noqa: ARG002
        """Blocked: read-only contexts do not allow writes.

        ``derive_existing`` is a write entry point (it persists derived child
        records), so it is blocked even though it reads a source thought first.

        Raises:
            ReadOnlyViolationError: Always.

        """
        msg = "derive_existing"
        raise ReadOnlyViolationError(msg)

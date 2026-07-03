"""EngravaCoreProtocol — core persistence interface.

Defines the abstract interface for thought-graph CRUD, edge management,
embedding storage, and similarity search.  This is the **core** protocol
— no cognitive-layer-specific methods belong here (free-tier boundary).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engrava.domain.enums import ActionStatus, VerificationStatus
    from engrava.domain.models.action import ActionRecord
    from engrava.domain.models.edge import EdgeRecord
    from engrava.domain.models.embedding import EmbeddingRecord
    from engrava.domain.models.filters import MetadataFilter, VisibilityQueryFilter
    from engrava.domain.models.metrics import EngravaMetrics
    from engrava.domain.models.search import HybridSearchResult
    from engrava.domain.models.thought import MetadataValue, ThoughtRecord
    from engrava.domain.models.ttl import CleanupResult


@runtime_checkable
class EngravaCoreProtocol(Protocol):
    """Abstract persistence interface for core thought-graph data.

    Implementations must provide async CRUD for thoughts, edges,
    and embedding-based search.
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

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        current_cycle: int | None = None,
        filters: MetadataFilter | None = None,
        visibility: VisibilityQueryFilter | None = None,
        collapse_key: str | Sequence[str] | None = None,
    ) -> HybridSearchResult:
        """Retrieve thoughts relevant to a query with one call.

        Ergonomic shorthand over :meth:`search_hybrid` for the common
        retrieval case: implementations delegate to ``search_hybrid`` with
        the query text and the given ``top_k``/``current_cycle``.

        Args:
            query: Natural-language text to search for.
            top_k: Maximum number of results to return.
            current_cycle: Current cognitive cycle.  When provided, the
                recency signal is blended into ranking; when ``None``,
                recency is skipped.
            filters: Optional metadata filter (an ``AND`` of typed field
                predicates over ``metadata``); delegated to
                ``search_hybrid``. ``None`` leaves the candidate set
                unchanged.
            visibility: Optional bounded visibility query filter; delegated
                to ``search_hybrid``. A query refinement, not access
                control — it enforces nothing and is bypassable.
            collapse_key: Optional de-fragmentation unit key (a single
                metadata path or an ordered sequence forming a composite key);
                delegated to ``search_hybrid``. Keeps one best-ranked row per
                caller-defined unit and backfills deeper distinct units. A
                presentation / de-dup convenience, not a filter and not
                isolation. The collapse step mutates no score, but *setting*
                ``collapse_key`` widens the internal candidate pool, which can
                rescale normalized fusion scores and shift order among units;
                only ``collapse_key=None`` leaves the result path unchanged.

        Returns:
            A ``HybridSearchResult`` with the ranked matches and the set of
            backends that contributed.

        """
        ...

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Retrieve a thought by its ID.

        Args:
            thought_id: UUID of the thought to retrieve.

        Returns:
            The thought record, or None if not found.

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
        ...

    async def search_fts(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Full-text search using SQLite FTS5 with BM25 ranking.

        Searches ``essence`` and ``content`` fields for keyword matches.
        Best for proper names, identifiers, and codes where embedding
        similarity is weak (e.g. ``"Kowalski"``, ``"REQ-FUNC*"``).

        Args:
            query: Search query string.  FTS5 syntax is supported
                (e.g. ``"projekt AND Alpha"``, ``"REQ-FUNC*"``).
            top_k: Maximum number of results.

        Returns:
            List of ``(thought_id, bm25_score)`` tuples sorted by
            relevance (higher score = more relevant).  Returns an empty
            list when the FTS5 index is not available (backward compat
            with databases that have not been migrated).

        """
        ...

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
        fts_top_k: int = 50,
        vector_top_k: int = 50,
        filters: MetadataFilter | None = None,
        visibility: VisibilityQueryFilter | None = None,
        collapse_key: str | Sequence[str] | None = None,
    ) -> HybridSearchResult:
        """Hybrid search combining FTS5 keyword + vector cosine similarity.

        Calls ``search_fts()`` and ``search_similar()`` independently,
        normalizes scores to ``[0, 1]``, applies weighted fusion,
        deduplicates by ``thought_id``, and returns merged results.

        Score normalization:
            - BM25 (FTS5): min-max normalization within result set.
            - Cosine (vector): already in ``[0, 1]``, used as-is.
            - Recency: exponential decay from ``updated_cycle``.

        Graceful degradation:
            - If FTS5 is unavailable, returns pure vector results.
            - If no embeddings exist, returns pure FTS5 results.
            - If ``query_vector`` is ``None`` and no embedding provider
              is configured, vector search is skipped.
            - If ``current_cycle`` is ``None``, recency is skipped.
            - Disabled component weights are redistributed
              proportionally to active components.

        Args:
            query_text: Text query for FTS5 keyword search.
            query_vector: Embedding vector for similarity search.
                When ``None`` and an embedding provider is configured,
                the query text is auto-embedded.
            top_k: Maximum number of final merged results.
            fts_weight: Optional FTS5 fusion-weight override.
            vector_weight: Optional vector fusion-weight override.
            recency_weight: Optional recency fusion-weight override.
            recency_half_life: Optional recency half-life override.
            current_cycle: Current cycle number for recency calculation.
            fts_top_k: Max candidates from FTS5 before fusion.
            vector_top_k: Max candidates from vector search before fusion.
            filters: Optional metadata filter (an ``AND`` of typed field
                predicates over ``metadata``), applied in-arm before each
                arm's limit so it never starves ``top_k``. ``None`` (or an
                empty filter) leaves the candidate set unchanged.
            visibility: Optional bounded visibility query filter for the
                "public-or-mine" pattern. A query refinement, **not** access
                control — it performs no authentication, authorization, or
                ownership enforcement; the caller can forge ``owner``; it is
                bypassable; it must not be used to protect tenant data.
            collapse_key: Optional de-fragmentation unit key — a single
                metadata path or an ordered sequence of paths forming a
                composite key. When set, among the already-ranked candidates
                only the single best-ranked row per caller-defined unit reaches
                the result and the freed slots are backfilled by deeper
                distinct units. A **presentation / de-dup convenience, not a
                filter and not isolation**: it does not change which rows are
                *eligible*, and the collapse step itself mutates no score (it
                only drops lower-ranked same-unit members). *Setting*
                ``collapse_key`` does, however, widen the internal candidate
                pool for deeper backfill, which — because the keyword arm is
                min-max normalized over the candidate set — can rescale
                normalized fusion scores and shift order among units; only
                ``collapse_key=None`` leaves the candidate, score, and order
                path byte-identical to the unfiltered query. It is only as
                meaningful as the unit metadata the application writes (a
                missing or malformed key ⇒ the row is its own unit, never
                collapsed).

        Returns:
            ``HybridSearchResult`` with ranked results and the set of
            backends that contributed (e.g. ``{"fts5", "vector"}``).

        """
        ...

    async def metrics(self) -> EngravaMetrics:
        """Return a point-in-time snapshot of store health and workload metrics."""
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

    async def get_embedding(self, thought_id: str) -> EmbeddingRecord | None:
        """Retrieve the embedding for a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The EmbeddingRecord, or None if not found.

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

    async def get_actions(self, thought_id: str) -> list[ActionRecord]:
        """Retrieve actions linked to a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            List of action records.

        """
        ...

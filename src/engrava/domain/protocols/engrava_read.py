"""EngravaReadProtocol — the read-only capability of the core store.

Isolates the **non-mutating** half of :class:`EngravaCoreProtocol`: thought and
edge retrieval, embedding lookup, similarity / full-text / hybrid search, and
health metrics.  Splitting the read surface into its own runtime-checkable
protocol lets a read-only view (``ReadOnlyEngrava``) declare — and be mechanically
verified against — exactly the capabilities it forwards, so the view can never
silently drift out of parity with the store it wraps.

Membership rule:
    A method belongs here only if it is free of persistent mutation *as a
    contract*.  Retrieval methods that a concrete backend couples to an
    access-frequency signal (a deferred, best-effort telemetry write) still
    qualify — that side effect is optional store instrumentation, not part of the
    read contract — but a read-only view is responsible for suppressing it (see
    ``ReadOnlyEngrava``).  Any method that mutates the thought graph as its
    purpose belongs to the writable :class:`EngravaCoreProtocol` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engrava.domain.enums import EdgeType, KnowledgeSource
    from engrava.domain.models.action import ActionRecord
    from engrava.domain.models.edge import EdgeRecord
    from engrava.domain.models.embedding import EmbeddingRecord
    from engrava.domain.models.filters import MetadataFilter, VisibilityQueryFilter
    from engrava.domain.models.metrics import EngravaMetrics
    from engrava.domain.models.search import HybridSearchResult
    from engrava.domain.models.thought import ThoughtRecord


@runtime_checkable
class EngravaReadProtocol(Protocol):
    """Read-only persistence interface for core thought-graph data.

    The non-mutating half of :class:`EngravaCoreProtocol`.  Implementations
    provide async retrieval for thoughts, edges, embeddings, and the
    similarity / full-text / hybrid search paths, plus a metrics snapshot and
    the cognitive-cycle high-water mark.  A store that satisfies the full
    :class:`EngravaCoreProtocol` satisfies this protocol by construction.  A
    write-blocking view (``ReadOnlyEngrava``) implements this protocol for its
    reads and typed-rejects every write at call time; because those rejecting
    write methods are still real members, such a view *structurally* presents the
    writable names too, so its read-only nature is behavioural (writes raise), not
    the absence of the write methods.
    """

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
        """Retrieve thoughts relevant to a query with one call.

        Ergonomic shorthand over :meth:`search_hybrid` for the common
        retrieval case: implementations delegate to ``search_hybrid`` with
        the query text and the given ``top_k`` and recency reference.

        Args:
            query: Natural-language text to search for.
            top_k: Maximum number of results to return.
            current_cycle: Current cognitive cycle (cognitive-cycle recency).
                When provided, the recency signal is blended into ranking; when
                ``None``, cycle recency is skipped. Mutually exclusive with
                ``recency_now``: both **explicit** ⇒ ``RecencyModeConflictError``.
            recency_now: Optional caller-supplied "now" instant (ISO-8601)
                selecting transaction-time recency (age by ``updated_at`` /
                ``created_at`` in wall-clock seconds); takes precedence over a
                passive ``cycle_provider``. The core reads no host clock —
                omitting it leaves the axis off.
            recency_now_half_life: Optional transaction-time half-life override,
                in seconds; consulted only with ``recency_now``.
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
            collapse_max_per_unit: Optional intra-unit retention depth for
                ``collapse_key``; delegated to ``search_hybrid``. ``None``
                (default) keeps one best row per unit; an integer ``>= 1`` keeps
                up to that many highest-ranked members of a unit and backfills
                the freed slots with deeper distinct units. Only effective with
                ``collapse_key``; rejected when ``< 1``.
            include_archived: When ``False`` (default) archived thoughts are
                excluded from every retrieval path; delegated to
                ``search_hybrid``. When ``True`` archived rows are re-admitted for
                this call without restoring them.

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
            provenance_filter: Optional typed ``AND`` filter over the
                ``provenance`` JSON column (read-only; reuses the metadata
                filter machinery). ``None`` leaves the result unchanged. A
                ``session_id`` / ``actor_id`` predicate uses the provenance
                identity index. Provenance is an untrusted hint, not a security
                boundary.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of matching thought records.

        """
        ...

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
            include_archived: When ``False`` (default) archived thoughts
                (``lifecycle_status = 'ARCHIVED'``) are excluded; when ``True``
                they are re-admitted for this call without restoring them.

        Returns:
            List of (thought_id, similarity_score) tuples, sorted descending.

        """
        ...

    async def search_fts(
        self,
        query: str,
        top_k: int = 10,
        *,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Full-text search using SQLite FTS5 with BM25 ranking.

        Searches ``essence`` and ``content`` fields for keyword matches.
        Best for proper names, identifiers, and codes where embedding
        similarity is weak (e.g. ``"Kowalski"``, ``"REQ-FUNC*"``).

        Args:
            query: Search query string.  FTS5 syntax is supported
                (e.g. ``"projekt AND Alpha"``, ``"REQ-FUNC*"``).
            top_k: Maximum number of results.
            include_archived: When ``False`` (default) archived thoughts
                (``lifecycle_status = 'ARCHIVED'``) are excluded from matches;
                when ``True`` they are re-admitted for this call without
                restoring them.

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
            - If neither recency reference (``current_cycle`` or
              ``recency_now``) is present, recency is skipped.
            - Disabled component weights are redistributed
              proportionally to active components.

        Recency has two separately-typed axes; a query selects **exactly one**
        reference — the cognitive ``current_cycle`` or the transaction-time
        ``recency_now``. An explicit ``recency_now`` takes precedence over a
        passive ``cycle_provider``; supplying both **explicit** references raises
        ``RecencyModeConflictError``. The core reads no host clock: a missing
        ``recency_now`` simply leaves the transaction axis off.

        Args:
            query_text: Text query for FTS5 keyword search.
            query_vector: Embedding vector for similarity search.
                When ``None`` and an embedding provider is configured,
                the query text is auto-embedded.
            top_k: Maximum number of final merged results.
            fts_weight: Optional FTS5 fusion-weight override.
            vector_weight: Optional vector fusion-weight override.
            recency_weight: Optional recency fusion-weight override.
            recency_half_life: Optional cognitive-cycle recency half-life
                override (in cycles).
            current_cycle: Current cycle number for cognitive-cycle recency.
                Mutually exclusive with ``recency_now``.
            recency_now: Optional caller-supplied "now" instant (ISO-8601)
                selecting transaction-time recency (age by ``updated_at`` /
                ``created_at`` in wall-clock seconds); takes precedence over a
                passive ``cycle_provider``. Parsed / UTC-normalised via the
                shared temporal helper (naive as UTC; host tz never consulted);
                malformed ⇒ ``InvalidRecencyArgumentError``.
            recency_now_half_life: Optional transaction-time half-life override,
                in seconds (default 604800 = 7 days); consulted only with
                ``recency_now``.
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
            collapse_max_per_unit: Optional intra-unit retention depth for
                ``collapse_key`` — how many rows of one unit may reach the
                result. ``None`` (default) keeps one best row per unit; an
                integer ``>= 1`` keeps up to that many highest-ranked members of
                a unit and lets the freed slots backfill deeper distinct units.
                Only takes effect together with ``collapse_key``, only relaxes
                the intra-unit count (never adds a non-candidate row, mutates a
                score, or drops a distinct unit as a side effect), and is
                rejected when ``< 1``.
            include_archived: When ``False`` (default) archived thoughts
                (``lifecycle_status = 'ARCHIVED'``) are excluded from every
                candidate path — the FTS and vector arms, the query-less
                fallback, and the ``CONSOLIDATED_FROM`` graph expansion. When
                ``True`` archived rows are re-admitted across all of those paths
                for this call without restoring them. The independent
                retired-REFLECTION freshness floor is unaffected either way.

        Returns:
            ``HybridSearchResult`` with ranked results and the set of
            backends that contributed (e.g. ``{"fts5", "vector"}``).

        """
        ...

    async def metrics(self) -> EngravaMetrics:
        """Return a point-in-time snapshot of store health and workload metrics."""
        ...

    async def max_cycle(self) -> int:
        """Return the store's cognitive-cycle high-water mark.

        The maximum cognitive cycle across **every** cycle-bearing record —
        ``MAX(thought.updated_cycle)`` unioned with ``MAX(edge.created_cycle)``
        — i.e. the true store high-water mark, not merely the thought maximum
        (an edge created at a higher cycle than any thought still counts).

        This is a read-only recovery accessor: a consumer that advances its own
        cognitive cycle can resume its counter from this value across process
        restarts. On an empty store — or one where every record is stamped
        cycle ``0`` — it returns ``0``.

        Returns:
            The maximum cognitive cycle stored, or ``0`` when the store holds no
            cycle-bearing records.

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
                ``metadata_json`` column (reuses the metadata-filter machinery,
                EQ / IN + ``$``/``$.key``/``$[0]`` JSONPath only). ``None`` leaves
                the result unchanged. Edges with malformed ``metadata_json`` never
                match a non-empty filter. A query refinement, not a security
                boundary.
            limit: Maximum number of edges to return.

        Returns:
            List of matching edge records, ordered by ``created_cycle`` DESC.

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

    async def get_actions(self, thought_id: str) -> list[ActionRecord]:
        """Retrieve actions linked to a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            List of action records.

        """
        ...

"""Search result value objects for engrava.

Contains structured result types returned by composite search methods
(e.g. hybrid search).  These value objects carry diagnostic metadata
alongside the ranked results so consumers can introspect which backends
were available for the query.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HybridSearchResult:
    """Result of a hybrid FTS5 + vector search operation.

    Carries the fused ranked results together with diagnostic metadata
    indicating which search backends were available for the query.

    Attributes:
        results: Ranked ``(thought_id, combined_score)`` tuples,
            sorted descending by score.
        backends_used: Names of backends that were *available* for this
            query (e.g. ``{"fts5", "vector"}``).  A backend appears
            even if it returned zero results; absence means the backend
            was unavailable or not applicable.

    Examples:
        >>> r = HybridSearchResult(
        ...     results=[("t1", 0.85), ("t2", 0.60)],
        ...     backends_used={"fts5", "vector"},
        ... )
        >>> len(r.results)
        2
        >>> "fts5" in r.backends_used
        True

    """

    results: list[tuple[str, float]] = field(default_factory=list)
    backends_used: frozenset[str] = field(default_factory=frozenset)

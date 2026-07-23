"""Backend-independent capability protocols for Dreaming consolidation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from engrava.domain.dreaming import ConsolidationResult
    from engrava.domain.enums import EdgeType, KnowledgeSource
    from engrava.domain.models.edge import EdgeRecord
    from engrava.domain.models.embedding import EmbeddingRecord
    from engrava.domain.models.thought import ThoughtRecord


@runtime_checkable
class DreamingStoreProtocol(Protocol):
    """The fourteen persistence capabilities required by Dreaming.

    The protocol deliberately models only operations used by consolidation.
    It lets Dreaming run against compatible stores without depending on a
    concrete backend or inheriting the much broader core persistence surface.
    """

    async def count_thoughts(
        self,
        *,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        priority: str | None = None,
    ) -> int:
        """Count thoughts matching Dreaming's candidate filters."""
        ...

    async def create_edge(self, edge: EdgeRecord) -> EdgeRecord:
        """Persist an edge created during consolidation."""
        ...

    async def create_thought(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Persist a reflection created during consolidation."""
        ...

    async def get_edges(
        self,
        thought_id: str,
        *,
        direction: str = "BOTH",
    ) -> list[EdgeRecord]:
        """Retrieve edges connected to a thought."""
        ...

    async def get_embedding(self, thought_id: str) -> EmbeddingRecord | None:
        """Retrieve a thought embedding when present."""
        ...

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Retrieve a thought by identifier."""
        ...

    async def list_edges(
        self,
        *,
        edge_type: EdgeType | None = None,
        source: KnowledgeSource | None = None,
        limit: int = 5000,
    ) -> list[EdgeRecord]:
        """List edges used to build the consolidation graph."""
        ...

    async def list_thoughts(
        self,
        *,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        limit: int = 50,
    ) -> list[ThoughtRecord]:
        """List bounded candidate thoughts for consolidation."""
        ...

    async def retire_orphan_reflections(self) -> int:
        """Retire reflections whose source set is no longer active."""
        ...

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Find live thoughts similar to an embedding vector."""
        ...

    async def store_embedding(
        self,
        thought_id: str,
        vector: list[float],
        *,
        model_name: str = "all-MiniLM-L12-v2",
    ) -> EmbeddingRecord:
        """Persist a reflection centroid embedding."""
        ...

    def suspend_auto_commit(self) -> AbstractAsyncContextManager[None]:
        """Open a store-owned transaction window for grouped writes."""
        ...

    async def thought_exists_by_source(
        self,
        *,
        source: str,
        thought_type_value: str,
    ) -> bool:
        """Check reflection idempotence by deterministic source value."""
        ...

    async def update_thought(self, thought_id: str, **changes: object) -> ThoughtRecord:
        """Update a thought promoted during consolidation."""
        ...


@runtime_checkable
class DreamingConsolidatorProtocol(Protocol):
    """A backend-independent consolidator installable on a store facade."""

    async def run_consolidation(
        self,
        store: DreamingStoreProtocol,
        current_cycle: int,
    ) -> ConsolidationResult:
        """Run one consolidation pass against the supplied store capabilities."""
        ...

"""Core domain models — frozen Pydantic entities."""

from engrava.domain.models.action import ActionRecord
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.embedding import EmbeddingRecord
from engrava.domain.models.filters import (
    FieldOp,
    FieldPredicate,
    FilterScalar,
    MetadataFilter,
    VisibilityQueryFilter,
)
from engrava.domain.models.metrics import (
    EdgeCounts,
    EngravaMetrics,
    LatencyHistogram,
    StorageFootprint,
    ThoughtCounts,
)
from engrava.domain.models.provenance import ProvenanceContext
from engrava.domain.models.search import HybridSearchResult
from engrava.domain.models.thought import MetadataValue, ThoughtRecord
from engrava.domain.models.ttl import CleanupResult, CleanupStrategy

__all__ = [
    "ActionRecord",
    "CleanupResult",
    "CleanupStrategy",
    "EdgeCounts",
    "EdgeRecord",
    "EmbeddingRecord",
    "EngravaMetrics",
    "FieldOp",
    "FieldPredicate",
    "FilterScalar",
    "HybridSearchResult",
    "LatencyHistogram",
    "MetadataFilter",
    "MetadataValue",
    "ProvenanceContext",
    "StorageFootprint",
    "ThoughtCounts",
    "ThoughtRecord",
    "VisibilityQueryFilter",
]

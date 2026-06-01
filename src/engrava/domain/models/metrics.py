"""Engrava metrics snapshot models.

Thought / edge counts, storage, search latency histograms — plain
runtime statistics value-objects consumed by callers that read the
store's `metrics_snapshot()` output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class LatencyHistogram:
    """Rolling-window search latency snapshot."""

    sample_count: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0


@dataclass(frozen=True)
class ThoughtCounts:
    """Thought counts by type and lifecycle status."""

    by_type: dict[str, int] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)
    total: int = 0


@dataclass(frozen=True)
class EdgeCounts:
    """Edge counts keyed by edge type."""

    by_type: dict[str, int] = field(default_factory=dict)
    total: int = 0


@dataclass(frozen=True)
class StorageFootprint:
    """On-disk storage footprint for the main database."""

    db_bytes: int = 0
    wal_bytes: int = 0
    vec_index_bytes: int = 0
    total_bytes: int = 0


@dataclass(frozen=True)
class EngravaMetrics:
    """Point-in-time snapshot of store health and workload metrics."""

    schema_version: Literal[1] = 1
    snapshot_timestamp: float = 0.0
    thoughts: ThoughtCounts = field(default_factory=ThoughtCounts)
    edges: EdgeCounts = field(default_factory=EdgeCounts)
    storage: StorageFootprint = field(default_factory=StorageFootprint)
    search_latency: LatencyHistogram = field(default_factory=LatencyHistogram)

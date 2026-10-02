"""Engrava metrics snapshot models.

Thought / edge counts, storage, search latency histograms — plain
runtime statistics value-objects consumed by callers that read the
store's `metrics()` output.
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
    """Point-in-time snapshot of store health and workload metrics.

    ``schema_version``, ``snapshot_timestamp``, and ``measured`` describe the
    snapshot itself; ``thoughts``, ``edges``, ``storage``, and
    ``search_latency`` are what was (or, when ``measured`` is ``False``, was
    not) measured.

    Args:
        schema_version: Snapshot shape version; bumped on any field addition.
        snapshot_timestamp: Unix timestamp the snapshot was taken at.
        thoughts: Thought counts by type and lifecycle status.
        edges: Edge counts by type.
        storage: On-disk storage footprint for the main database.
        search_latency: Rolling-window search latency snapshot.
        measured: Whether this snapshot was produced by actually querying the
            store, rather than being a zero-filled placeholder (e.g. the
            early return `MetricsConfig(enabled=False)` takes). Defaults to
            ``False`` so any construction path this class does not already
            know about — including a bare ``EngravaMetrics()`` and a future
            early return nobody remembers to update — reports "not measured"
            rather than fabricating a confident zero. Only the one code path
            that runs the aggregate queries sets this to ``True``, and it
            must do so explicitly.

            This describes the *snapshot as a whole*, not that every field is
            populated. A store with metrics enabled that has genuinely served
            no searches yet is measured (``measured=True``), and its
            `search_latency.sample_count` is honestly ``0`` — that emptiness
            is already carried by the field itself; a measured, empty
            database can still have a nonzero `storage` footprint and a real
            `snapshot_timestamp` too. Treating ``measured`` as "and every
            sub-field is nonzero" would misreport a real, empty-but-measured
            store as unmeasured, which is the same lie this flag exists to
            prevent, pointed the other way.

            Deliberately declared **last**, not grouped with
            ``schema_version``/``snapshot_timestamp`` above despite
            describing the same thing: this field was added after the other
            six shipped, and a dataclass's field order is also its
            positional-constructor order. Inserting it earlier would
            silently repurpose whatever a caller's existing positional
            argument in that slot meant — for instance a six-argument
            positional call's third argument, previously ``thoughts``,
            would silently become ``measured`` instead, producing a
            wrong-but-valid object with no error at all. Appending it keeps
            every *direct* ``EngravaMetrics(...)`` positional call meaning
            what it always meant.

            This guarantee covers direct construction only, not a subclass.
            A frozen subclass that appends its own field (e.g.
            ``cached: bool = False`` on a hypothetical ``CachedMetrics``)
            gets that field placed after ``measured`` in *its* positional
            order, so a positional call built against the subclass's
            previous (pre-``measured``) field count has its last argument
            silently rebound from the subclass's own field to ``measured``
            instead. No position for a new base-class field would avoid
            this for a subclass that appends its own — inserting anywhere
            else would additionally break direct base-class callers too, so
            appended-last remains the least-bad choice. Do not move it
            earlier to "tidy up" the grouping, and do not treat this
            docstring as claiming subclass positional safety it cannot
            provide.

    """

    schema_version: Literal[2] = 2
    snapshot_timestamp: float = 0.0
    thoughts: ThoughtCounts = field(default_factory=ThoughtCounts)
    edges: EdgeCounts = field(default_factory=EdgeCounts)
    storage: StorageFootprint = field(default_factory=StorageFootprint)
    search_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    measured: bool = False

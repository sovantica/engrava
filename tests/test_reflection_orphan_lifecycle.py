"""Orphan REFLECTION lifecycle tests — retire a synthesis once its cluster leaves ACTIVE.

A REFLECTION is a derived synthesis of a live cluster. When every thought it
was consolidated from is no longer ACTIVE, the REFLECTION must transition
ACTIVE -> ARCHIVED during the consolidation pass so ordinary ``gc`` reclaims
it (cascading its centroid embedding and CONSOLIDATED_FROM edges). A
partially-archived cluster keeps its REFLECTION; a REFLECTION with no sources
is never retired by the all-non-ACTIVE rule firing over an empty set.

All assertions read lifecycle state back through raw SQLite (not the public
ORM) where the contract is about the persisted row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.config import (
    DreamingConfig,
    DreamingGates,
    EdgeCreationConfig,
)
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.infrastructure.sqlite import engrava_core

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh ``SqliteEngravaCore`` backed by an on-disk SQLite database."""
    db = await aiosqlite.connect(str(tmp_path / "orphan.db"))
    db.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(db=db)
    await s.ensure_schema()
    yield s
    await db.close()


def _obs(tid: str, *, essence: str = "topic", content: str = "content") -> ThoughtRecord:
    """Build a minimal ACTIVE OBSERVATION thought."""
    return ThoughtRecord(
        thought_id=tid,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confirmation_count=5,
        confidence=0.9,
    )


def _reflection_cfg() -> DreamingConfig:
    """Config that creates REFLECTIONs from a 2-member agglomerative cluster."""
    return DreamingConfig(
        enabled=True,
        promote_threshold=0.0,
        max_p1_fraction=1.0,
        promote_targets="ALL",
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            max_promoted_per_run=50,
            enable_reflections=True,
            cluster_algorithm="agglomerative",
            min_cluster_size=2,
            cluster_quality_gating_enabled=False,
        ),
        edges=EdgeCreationConfig(
            enabled=True,
            top_k=3,
            min_similarity=0.5,
            edge_weight_factor=0.5,
        ),
    )


async def _seed_reflection(store: SqliteEngravaCore) -> tuple[str, list[str]]:
    """Create two clustered OBSERVATIONs and dream a REFLECTION over them.

    Returns the REFLECTION id and the list of its source ids.
    """
    t1 = await store.create_thought(_obs("o1", essence="python async programming"))
    t2 = await store.create_thought(_obs("o2", essence="asyncio concurrency python"))
    await store.store_embedding(t1.thought_id, [0.9, 0.1, 0.0], model_name="test")
    await store.store_embedding(t2.thought_id, [0.88, 0.12, 0.0], model_name="test")

    ext = DreamingExtension(config=_reflection_cfg())
    result = await ext.run_consolidation(store, current_cycle=1)
    assert result.reflections_created >= 1

    reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
    rid = reflections[0].thought_id
    sources = await store.consolidated_member_ids(rid)
    assert set(sources) == {t1.thought_id, t2.thought_id}
    return rid, sources


async def _raw_lifecycle(store: SqliteEngravaCore, thought_id: str) -> str | None:
    """Read ``lifecycle_status`` directly from the SQLite row."""
    cursor = await store._db.execute(
        "SELECT lifecycle_status FROM thought WHERE thought_id = ?",
        (thought_id,),
    )
    row = await cursor.fetchone()
    return None if row is None else str(row["lifecycle_status"])


async def _raw_thought_count(store: SqliteEngravaCore, thought_id: str) -> int:
    """Raw COUNT(*) of thought rows with the given id."""
    cursor = await store._db.execute(
        "SELECT COUNT(*) AS n FROM thought WHERE thought_id = ?",
        (thought_id,),
    )
    row = await cursor.fetchone()
    return 0 if row is None else int(row["n"])


async def _raw_embedding_count(store: SqliteEngravaCore, owner_id: str) -> int:
    """Raw COUNT(*) of embedding rows owned by the given thought."""
    cursor = await store._db.execute(
        "SELECT COUNT(*) AS n FROM embedding WHERE owner_id = ?",
        (owner_id,),
    )
    row = await cursor.fetchone()
    return 0 if row is None else int(row["n"])


async def _raw_outgoing_edge_count(store: SqliteEngravaCore, from_id: str) -> int:
    """Raw COUNT(*) of edges leaving the given thought."""
    cursor = await store._db.execute(
        "SELECT COUNT(*) AS n FROM edge WHERE from_thought_id = ?",
        (from_id,),
    )
    row = await cursor.fetchone()
    return 0 if row is None else int(row["n"])


# ---------------------------------------------------------------------------
# Orphan retire (100% sources non-ACTIVE)
# ---------------------------------------------------------------------------


class TestOrphanRetire:
    """A REFLECTION whose every source left ACTIVE is retired ACTIVE -> ARCHIVED."""

    async def test_all_sources_archived_retires_reflection(self, store: SqliteEngravaCore) -> None:
        """All sources ARCHIVED -> REFLECTION transitions to ARCHIVED (raw read-back)."""
        rid, sources = await _seed_reflection(store)
        assert await _raw_lifecycle(store, rid) == "ACTIVE"

        for sid in sources:
            await store.update_thought(sid, lifecycle_status=LifecycleStatus.ARCHIVED)

        ext = DreamingExtension(config=_reflection_cfg())
        result = await ext.run_consolidation(store, current_cycle=2)

        assert result.orphans_retired == 1
        assert await _raw_lifecycle(store, rid) == "ARCHIVED"

    async def test_all_sources_done_retires_reflection(self, store: SqliteEngravaCore) -> None:
        """Sources transitioned ACTIVE->DONE (non-ACTIVE) also orphan the REFLECTION."""
        rid, sources = await _seed_reflection(store)
        for sid in sources:
            await store.update_thought(sid, lifecycle_status=LifecycleStatus.DONE)

        ext = DreamingExtension(config=_reflection_cfg())
        result = await ext.run_consolidation(store, current_cycle=2)

        assert result.orphans_retired == 1
        assert await _raw_lifecycle(store, rid) == "ARCHIVED"

    async def test_retired_reflection_not_recalled(self, store: SqliteEngravaCore) -> None:
        """After retire, the orphan does not surface in search_similar / search_hybrid."""
        rid, sources = await _seed_reflection(store)
        for sid in sources:
            await store.update_thought(sid, lifecycle_status=LifecycleStatus.ARCHIVED)
        ext = DreamingExtension(config=_reflection_cfg())
        await ext.run_consolidation(store, current_cycle=2)

        # Query on the REFLECTION's own topic vector — pre-retire it would rank.
        sim = await store.search_similar([0.9, 0.1, 0.0], top_k=10)
        assert rid not in {tid for tid, _ in sim}

        hybrid = await store.search_hybrid("", [0.9, 0.1, 0.0], top_k=10)
        assert rid not in {tid for tid, _ in hybrid.results}

        only = await store.search_reflections_only("", [0.9, 0.1, 0.0], top_k=10)
        assert rid not in {tid for tid, _ in only.results}

    async def test_gc_cascades_retired_orphan(self, store: SqliteEngravaCore) -> None:
        """After retire + gc, the orphan row, its centroid, and edges are all gone."""
        rid, sources = await _seed_reflection(store)
        for sid in sources:
            await store.update_thought(sid, lifecycle_status=LifecycleStatus.ARCHIVED)
        ext = DreamingExtension(config=_reflection_cfg())
        await ext.run_consolidation(store, current_cycle=2)
        assert await _raw_lifecycle(store, rid) == "ARCHIVED"

        # Emulate ordinary gc: delete edges/embeddings/thought for ARCHIVED rows.
        await store._db.execute(
            "DELETE FROM edge WHERE from_thought_id IN "
            "(SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED') "
            "OR to_thought_id IN "
            "(SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED')"
        )
        await store._db.execute(
            "DELETE FROM embedding WHERE owner_id IN "
            "(SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED')"
        )
        await store._db.execute("DELETE FROM thought WHERE lifecycle_status = 'ARCHIVED'")
        await store._db.commit()

        assert await _raw_thought_count(store, rid) == 0
        assert await _raw_embedding_count(store, rid) == 0
        assert await _raw_outgoing_edge_count(store, rid) == 0


# ---------------------------------------------------------------------------
# Partial-archived must NOT retire (guard against over-eager GC)
# ---------------------------------------------------------------------------


class TestPartialArchivedKept:
    """A REFLECTION with any still-ACTIVE source is kept."""

    async def test_one_active_source_keeps_reflection(self, store: SqliteEngravaCore) -> None:
        """Archiving only one of two sources leaves the REFLECTION ACTIVE."""
        rid, sources = await _seed_reflection(store)
        await store.update_thought(sources[0], lifecycle_status=LifecycleStatus.ARCHIVED)
        # sources[1] stays ACTIVE.

        ext = DreamingExtension(config=_reflection_cfg())
        result = await ext.run_consolidation(store, current_cycle=2)

        assert result.orphans_retired == 0
        assert await _raw_lifecycle(store, rid) == "ACTIVE"

    async def test_zero_source_reflection_not_retired(self, store: SqliteEngravaCore) -> None:
        """A REFLECTION with no CONSOLIDATED_FROM edges is never retired (>=1 guard)."""
        # Defensive/legacy shape: a REFLECTION with zero source edges.
        reflection = ThoughtRecord(
            thought_id="orphan-noedge",
            thought_type=ThoughtType.REFLECTION,
            essence="REFLECTION [x]",
            content="{}",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="dreaming:deadbeef",
            source_type=KnowledgeSource.DREAMING,
        )
        await store.create_thought(reflection)

        ext = DreamingExtension(config=_reflection_cfg())
        result = await ext.run_consolidation(store, current_cycle=2)

        assert result.orphans_retired == 0
        assert await _raw_lifecycle(store, "orphan-noedge") == "ACTIVE"

    async def test_extra_active_source_keeps_reflection(self, store: SqliteEngravaCore) -> None:
        """Adding a third still-ACTIVE source edge leaves the REFLECTION ACTIVE."""
        rid, _sources = await _seed_reflection(store)
        # Add a brand-new ACTIVE source edge but leave both originals ACTIVE.
        extra = await store.create_thought(_obs("o3", essence="python async extra"))
        await store.store_embedding(extra.thought_id, [0.87, 0.13, 0.0], model_name="test")
        await store.create_edge(
            EdgeRecord(
                edge_id="e-extra",
                from_thought_id=rid,
                to_thought_id=extra.thought_id,
                edge_type=EdgeType.CONSOLIDATED_FROM,
                weight=1.0,
                created_cycle=2,
                source=KnowledgeSource.DREAMING,
            )
        )
        ext = DreamingExtension(config=_reflection_cfg())
        result = await ext.run_consolidation(store, current_cycle=3)
        assert result.orphans_retired == 0
        assert await _raw_lifecycle(store, rid) == "ACTIVE"


# ---------------------------------------------------------------------------
# Idempotence / determinism
# ---------------------------------------------------------------------------


class TestSweepIdempotence:
    """The orphan sweep retires each orphan once and is a no-op afterwards."""

    async def test_second_pass_is_noop(self, store: SqliteEngravaCore) -> None:
        """Running consolidation twice retires the orphan once, then 0 on re-run."""
        rid, sources = await _seed_reflection(store)
        for sid in sources:
            await store.update_thought(sid, lifecycle_status=LifecycleStatus.ARCHIVED)

        ext = DreamingExtension(config=_reflection_cfg())
        first = await ext.run_consolidation(store, current_cycle=2)
        second = await ext.run_consolidation(store, current_cycle=3)

        assert first.orphans_retired == 1
        assert second.orphans_retired == 0
        assert await _raw_lifecycle(store, rid) == "ARCHIVED"

    async def test_source_statuses_helper_is_deterministic(self, store: SqliteEngravaCore) -> None:
        """``consolidated_source_statuses`` reports each source exactly once."""
        rid, sources = await _seed_reflection(store)
        statuses = await store.consolidated_source_statuses(rid)
        assert sorted(statuses) == ["ACTIVE", "ACTIVE"]

        await store.update_thought(sources[0], lifecycle_status=LifecycleStatus.ARCHIVED)
        statuses_after = await store.consolidated_source_statuses(rid)
        assert sorted(statuses_after) == ["ACTIVE", "ARCHIVED"]


# ---------------------------------------------------------------------------
# Full-coverage sweep — orphans beyond the first page are still retired
# ---------------------------------------------------------------------------


def _reflection_record(rid: str, *, updated_cycle: int) -> ThoughtRecord:
    """Build a minimal ACTIVE REFLECTION at a fixed ``updated_cycle``."""
    return ThoughtRecord(
        thought_id=rid,
        thought_type=ThoughtType.REFLECTION,
        essence=f"REFLECTION [{rid}]",
        content="{}",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=updated_cycle,
        updated_cycle=updated_cycle,
        source=f"dreaming:{rid}",
        source_type=KnowledgeSource.DREAMING,
    )


async def _add_source(
    store: SqliteEngravaCore,
    reflection_id: str,
    source_id: str,
    *,
    lifecycle: LifecycleStatus,
) -> None:
    """Create an OBSERVATION at ``lifecycle`` and a CONSOLIDATED_FROM edge to it."""
    await store.create_thought(_obs(source_id, essence=f"src {source_id}"))
    if lifecycle is not LifecycleStatus.ACTIVE:
        await store.update_thought(source_id, lifecycle_status=lifecycle)
    await store.create_edge(
        EdgeRecord(
            edge_id=f"e-{reflection_id}-{source_id}",
            from_thought_id=reflection_id,
            to_thought_id=source_id,
            edge_type=EdgeType.CONSOLIDATED_FROM,
            weight=1.0,
            created_cycle=0,
            source=KnowledgeSource.DREAMING,
        )
    )


class TestSweepCoversAllActiveReflections:
    """The sweep retires orphans regardless of how many ACTIVE REFLECTIONs exist.

    Regression: the sweep previously fetched only one capped page ordered by
    ``updated_cycle DESC``. An orphan with a low ``updated_cycle`` (old,
    untouched) fell beyond that page and was never retired, violating the
    "for each ACTIVE REFLECTION" contract.
    """

    async def test_orphan_beyond_first_page_is_retired(
        self,
        store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A low-``updated_cycle`` orphan on a later page is still retired.

        Builds two ACTIVE REFLECTIONs with a single-row page size:

        * ``r-fresh`` (``updated_cycle=10``) keeps a still-ACTIVE source -> the
          first page under ``DESC`` ordering -> must stay ACTIVE.
        * ``r-orphan`` (``updated_cycle=1``) has only an ARCHIVED source -> a
          later page -> must transition ACTIVE -> ARCHIVED.

        With ``candidates_limit=1`` and a single-row sweep page, the old capped
        sweep only ever saw ``r-fresh`` and left ``r-orphan`` ACTIVE forever;
        the paginated full-coverage sweep walks every page and retires it.
        """
        # Force single-row pagination so two REFLECTIONs span two pages. The
        # sweep is store-owned (retire_orphan_reflections), so the page-size
        # constant lives on the core module.
        monkeypatch.setattr(engrava_core, "_ORPHAN_SWEEP_PAGE_SIZE", 1)

        # Most-recently-updated REFLECTION with a live source (first DESC page).
        await store.create_thought(_reflection_record("r-fresh", updated_cycle=10))
        await _add_source(store, "r-fresh", "src-live", lifecycle=LifecycleStatus.ACTIVE)

        # Older REFLECTION whose only source is archived -> a full orphan.
        await store.create_thought(_reflection_record("r-orphan", updated_cycle=1))
        await _add_source(store, "r-orphan", "src-dead", lifecycle=LifecycleStatus.ARCHIVED)

        ext = DreamingExtension(
            config=DreamingConfig(
                enabled=True,
                promote_threshold=0.0,
                max_p1_fraction=1.0,
                promote_targets="ALL",
                candidates_limit=1,
                gates=DreamingGates(
                    min_age_cycles=0,
                    allow_zero_confirmation=True,
                    max_promoted_per_run=50,
                    enable_reflections=True,
                ),
            )
        )
        result = await ext.run_consolidation(store, current_cycle=11)

        # Raw read-back: the orphan retired, the one with a live source stayed.
        assert result.orphans_retired == 1
        assert await _raw_lifecycle(store, "r-orphan") == "ARCHIVED"
        assert await _raw_lifecycle(store, "r-fresh") == "ACTIVE"

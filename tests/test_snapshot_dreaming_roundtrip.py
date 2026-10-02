"""A real Dreaming consolidation survives snapshot + restore, byte for byte.

Companion to ``test_snapshot_records.py``'s
``TestRestoreExemptsCentroidRowsFromTheIdentityInvariant``, which builds a
REFLECTION and its centroid embedding directly to exercise restore's
identity check in isolation. This file instead runs the real ``dreaming``
extension's consolidation pipeline -- only the two member thoughts'
embeddings are supplied by hand, under one fixed provider model name, to
keep the run offline and deterministic -- so the REFLECTION thought, its
``CONSOLIDATED_FROM`` lineage, and its centroid vector are exactly what
production code produces, not a hand-built stand-in.

Two scenarios, matching the two points restore's centroid-identity fix
touches:

* Restoring a real-Dreaming snapshot into a fresh target exercises the
  per-row exemption on every *incoming* embedding row.
* Merging an unrelated snapshot into an already-healthy, centroid-bearing
  target exercises the pre-loop exemption on the target's own *existing*
  rows -- which runs whether or not ``--skip-embeddings`` is set.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from click.testing import CliRunner

from engrava import LifecycleStatus, Priority, SqliteEngravaCore, ThoughtRecord, ThoughtType
from engrava.cli.main import cli
from engrava.config import DreamingConfig, DreamingGates, EdgeCreationConfig
from engrava.domain.dreaming import CENTROID_MODEL_NAME
from engrava.extensions.dreaming import DreamingExtension

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click test runner."""
    return CliRunner()


def _minimal_thought_data(thought_id: str) -> dict[str, object]:
    """Return a thought ``data`` mapping with exactly the required columns."""
    return {
        "thought_id": thought_id,
        "thought_type": "OBSERVATION",
        "essence": "essence",
        "content": "content",
        "priority": "P2",
    }


def _thought_line(data: dict[str, object]) -> str:
    """Serialise a thought data record to a snapshot line."""
    return json.dumps({"_type": "thought", "data": data})


async def _dump_table(db_path: Path, table: str) -> list[dict[str, object]]:
    """Return every row of a table as ordered dicts, sorted by primary key."""
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        cursor = await conn.execute(f"SELECT * FROM {table}")  # noqa: S608
        rows = [dict(row) for row in await cursor.fetchall()]
    finally:
        await conn.close()
    return sorted(rows, key=lambda row: str(next(iter(row.values()))))


async def _metadata_map(db_path: Path) -> dict[str, str]:
    """Return every ``_metadata`` row as a ``{key: value}`` dict."""
    conn = await aiosqlite.connect(str(db_path))
    try:
        cursor = await conn.execute("SELECT key, value FROM _metadata")
        rows = await cursor.fetchall()
    finally:
        await conn.close()
    return {str(key): str(value) for key, value in rows}


async def _ranked_ids(db_path: Path, query_vector: list[float]) -> list[str]:
    """Return the ranked thought ids ``search_hybrid`` returns for a query.

    Every fusion weight but ``vector_weight`` is pinned to zero so the
    ranking depends only on the stored vectors -- deterministic and free of
    any wall-clock-based recency signal -- which is what makes comparing the
    ranking from two separately opened connections (source vs. restored
    target) a meaningful equivalence check rather than a coincidence.
    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        search_result = await store.search_hybrid(
            "",
            query_vector,
            top_k=10,
            fts_weight=0.0,
            vector_weight=1.0,
            priority_weight=0.0,
            graph_weight=0.0,
        )
        return [thought_id for thought_id, _ in search_result.results]
    finally:
        await conn.close()


# The real-Dreaming precedent (``test_dreaming_integration.py::
# TestReflectionFeedbackLoop``): agglomerative clustering needs no prior
# ASSOCIATED edges, so ``edges`` stays disabled here -- the only edges a run
# creates are the ``CONSOLIDATED_FROM`` lineage from ``_create_reflections``,
# which is gated solely by ``gates.enable_reflections``, never by ``edges``.
_DREAMING_CONFIG = DreamingConfig(
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
    edges=EdgeCreationConfig(enabled=False),
)


async def _real_dreaming_store(db_path: Path, *, provider_model: str) -> str:
    """Build a database via a real Dreaming consolidation pass.

    Two clustered thoughts get hand-supplied embeddings under one fixed
    provider model name. Running consolidation with reflections enabled
    then exercises the real write path in ``extensions/dreaming.py`` end to
    end: the REFLECTION thought, its ``CONSOLIDATED_FROM`` lineage edges to
    both members, and its centroid embedding under ``CENTROID_MODEL_NAME``
    are all produced by that code -- nothing here hand-builds any of them.

    Args:
        db_path: Path to create the database at.
        provider_model: The provider model name the two member thoughts are
            embedded under.

    Returns:
        The ``thought_id`` of the REFLECTION consolidation created.

    """
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    try:
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        members = (
            ("member-1", "quarterly roadmap planning", [0.9, 0.1, 0.0]),
            ("member-2", "quarterly roadmap follow-up", [0.88, 0.12, 0.0]),
        )
        for thought_id, essence, vector in members:
            await store.create_thought(
                ThoughtRecord(
                    thought_id=thought_id,
                    essence=essence,
                    content=f"{essence} in detail",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P3,
                    created_cycle=1,
                    updated_cycle=1,
                )
            )
            await store.store_embedding(thought_id, vector, model_name=provider_model)

        ext = DreamingExtension(config=_DREAMING_CONFIG)
        result = await ext.run_consolidation(store, current_cycle=5)
        if result.reflections_created < 1:
            msg = "setup: real Dreaming consolidation produced no REFLECTION"
            raise AssertionError(msg)

        reflections = await store.list_thoughts(thought_type=ThoughtType.REFLECTION)
        if len(reflections) != 1:
            msg = f"setup: expected exactly one REFLECTION, got {len(reflections)}"
            raise AssertionError(msg)
        return reflections[0].thought_id
    finally:
        await conn.close()


class TestRealDreamingSnapshotRestoreRoundTrip:
    """Scenario A: restore into a fresh target exercises the incoming-row
    exemption in ``_check_embedding_row_before_insert`` -- every embedding
    row in the snapshot, provider vectors and centroid alike, is an
    *incoming* row from restore's point of view.
    """

    def test_restoring_a_real_dreaming_snapshot_into_a_fresh_target_matches_the_source(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.db"
        reflection_id = asyncio.run(_real_dreaming_store(source, provider_model="model-real-dream"))

        snap = tmp_path / "snap.jsonl"
        snap_result = runner.invoke(cli, ["--db", str(source), "snapshot", "-o", str(snap)])
        assert snap_result.exit_code == 0, snap_result.output

        target = tmp_path / "fresh.db"
        result = runner.invoke(cli, ["--db", str(target), "restore", "-i", str(snap)])
        assert result.exit_code == 0, result.output

        # Thought rows: byte-identical, the REFLECTION included.
        source_thoughts = asyncio.run(_dump_table(source, "thought"))
        target_thoughts = asyncio.run(_dump_table(target, "thought"))
        assert target_thoughts == source_thoughts
        assert reflection_id in {row["thought_id"] for row in target_thoughts}

        # Embedding rows: ids AND vectors identical, centroid included.
        source_embeddings = asyncio.run(_dump_table(source, "embedding"))
        target_embeddings = asyncio.run(_dump_table(target, "embedding"))
        assert target_embeddings == source_embeddings
        assert len(source_embeddings) == 3  # two provider rows + one centroid
        assert {row["model_name"] for row in target_embeddings} == {
            "model-real-dream",
            CENTROID_MODEL_NAME,
        }

        # CONSOLIDATED_FROM lineage: non-empty, identical to the source.
        source_edges = asyncio.run(_dump_table(source, "edge"))
        target_edges = asyncio.run(_dump_table(target, "edge"))
        assert source_edges
        assert target_edges == source_edges
        assert {row["edge_type"] for row in source_edges} == {"CONSOLIDATED_FROM"}
        assert {row["from_thought_id"] for row in source_edges} == {reflection_id}

        # The identity lock is the provider model, never the centroid sentinel.
        metadata = asyncio.run(_metadata_map(target))
        assert metadata["embedding_model_name"] == "model-real-dream"
        assert metadata["embedding_dimension"] == "3"

        # The same query returns the same ranked ids from source and target,
        # the reflection among them.
        query_vector = [0.9, 0.1, 0.0]
        source_ranked = asyncio.run(_ranked_ids(source, query_vector))
        target_ranked = asyncio.run(_ranked_ids(target, query_vector))
        assert source_ranked == target_ranked
        assert reflection_id in target_ranked


class TestRealDreamingMergeRestorePreservesExistingReflectionRows:
    """Scenario B: merging an unrelated snapshot with ``--skip-embeddings``
    into an already-healthy, centroid-bearing target exercises the pre-loop
    exemption in ``_initial_embedding_state`` / ``_existing_embedding_identities``
    -- the scan that runs whether or not ``--skip-embeddings`` is set.
    """

    def test_merging_an_unrelated_snapshot_leaves_pre_existing_rows_unchanged(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        target = tmp_path / "target.db"
        reflection_id = asyncio.run(_real_dreaming_store(target, provider_model="model-real-dream"))

        # Dump the pre-existing application records by primary key BEFORE
        # the restore. A whole-table comparison would be the wrong tool here
        # (the restored unrelated thought is new, so the tables cannot stay
        # equal), and comparing the raw database file or an internal counter
        # would not be comparing the application records at all -- so each
        # pre-existing row is read back by the same key it was dumped under.
        before_thoughts = {
            row["thought_id"]: row for row in asyncio.run(_dump_table(target, "thought"))
        }
        before_embeddings = {
            row["embedding_id"]: row for row in asyncio.run(_dump_table(target, "embedding"))
        }
        before_edges = {row["edge_id"]: row for row in asyncio.run(_dump_table(target, "edge"))}
        assert before_thoughts
        assert before_embeddings
        assert before_edges

        snap = tmp_path / "snap.jsonl"
        snap.write_text(
            _thought_line(_minimal_thought_data("t-unrelated")) + "\n", encoding="utf-8"
        )

        result = runner.invoke(
            cli, ["--db", str(target), "restore", "-i", str(snap), "--skip-embeddings"]
        )
        assert result.exit_code == 0, result.output

        after_thoughts = {
            row["thought_id"]: row for row in asyncio.run(_dump_table(target, "thought"))
        }
        after_embeddings = {
            row["embedding_id"]: row for row in asyncio.run(_dump_table(target, "embedding"))
        }
        after_edges = {row["edge_id"]: row for row in asyncio.run(_dump_table(target, "edge"))}

        # The unrelated thought was restored -- and is new, which is exactly
        # why this is not a whole-table comparison.
        assert "t-unrelated" in after_thoughts
        assert "t-unrelated" not in before_thoughts

        # Every pre-existing record -- REFLECTION, member thoughts, centroid
        # embedding, provider embeddings, CONSOLIDATED_FROM edges -- is
        # unchanged, read back by the same primary key.
        for thought_id, row in before_thoughts.items():
            assert after_thoughts[thought_id] == row
        for embedding_id, row in before_embeddings.items():
            assert after_embeddings[embedding_id] == row
        for edge_id, row in before_edges.items():
            assert after_edges[edge_id] == row

        # --skip-embeddings restores no vectors of its own, and the merge
        # must not have disturbed the target's existing embedding rows either.
        assert len(after_embeddings) == len(before_embeddings)

        assert reflection_id in after_thoughts
        assert after_thoughts[reflection_id]["thought_type"] == "REFLECTION"

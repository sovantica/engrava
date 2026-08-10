"""Domain + serde tests for ``EdgeRecord.metadata``.

Mirrors ``test_thought_metadata.py`` for the edge record: the generic
``metadata`` field behaves identically on both core records (one metadata
contract), so these tests intentionally track the thought-metadata suite.

Covers:

* default-empty + frozen invariants on the Pydantic model;
* INSERT + UPDATE serde round trips through ``SqliteEngravaCore``;
* the shared ``_validate_metadata`` shape / size / finite-float rules on the
  edge write paths;
* unicode preservation via ``ensure_ascii=False``;
* the last-writer-wins UPDATE contract — an unrelated-field update preserves
  stored metadata, and a metadata update replaces it wholesale;
* typed ``MetadataFilter`` composition on ``list_edges``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import aiosqlite
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from engrava import SqliteEngravaCore
from engrava.cli.main import cli
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
)
from engrava.domain.models import EdgeRecord, MetadataValue, ThoughtRecord
from engrava.domain.models.filters import FieldOp, FieldPredicate, MetadataFilter
from engrava.infrastructure.read_only_store import ReadOnlyEngrava

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_edge(
    edge_id: str = "e-1",
    *,
    from_thought_id: str = "t-1",
    to_thought_id: str = "t-2",
    edge_type: EdgeType = EdgeType.ASSOCIATED,
    source: KnowledgeSource = KnowledgeSource.EXPERIENCE,
    created_cycle: int = 0,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> EdgeRecord:
    """Build a minimal EdgeRecord for direct domain-level checks."""
    metadata_value: dict[str, MetadataValue] = dict(metadata) if metadata is not None else {}
    return EdgeRecord(
        edge_id=edge_id,
        from_thought_id=from_thought_id,
        to_thought_id=to_thought_id,
        edge_type=edge_type,
        weight=0.5,
        created_cycle=created_cycle,
        source=source,
        metadata=metadata_value,
    )


def _make_thought(thought_id: str) -> ThoughtRecord:
    """Build a minimal ACTIVE ThoughtRecord (edge endpoints must exist)."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content="content",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """In-memory ``SqliteEngravaCore`` with the head schema + five thoughts.

    ``t-1`` .. ``t-5`` are pre-persisted so edges between them satisfy the
    ``ON DELETE CASCADE`` foreign keys on both endpoints. Multiple distinct
    endpoints let the filter tests create several edges without colliding on
    the ``UNIQUE(from_thought_id, to_thought_id, edge_type)`` constraint.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    core = SqliteEngravaCore(conn)
    await core.ensure_schema()
    for i in range(1, 6):
        await core.create_thought(_make_thought(f"t-{i}"))
    try:
        yield core
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Domain-model invariants (pure Pydantic — no SQLite)
# ---------------------------------------------------------------------------


class TestEdgeMetadataDomain:
    """Pure-Pydantic invariants — no SQLite involved."""

    def test_default_metadata_is_empty_dict(self) -> None:
        edge = _make_edge()
        assert edge.metadata == {}

    def test_dumped_edge_always_includes_metadata_key(self) -> None:
        """A dumped ``EdgeRecord`` now always carries ``"metadata": {}``."""
        dumped = _make_edge().model_dump()
        assert "metadata" in dumped
        assert dumped["metadata"] == {}

    def test_frozen_metadata_assignment_raises(self) -> None:
        """Frozen Pydantic model rejects attribute reassignment."""
        edge = _make_edge(metadata={"subtype": "supports"})
        with pytest.raises(ValidationError):
            edge.metadata = {"subtype": "refutes"}  # type: ignore[misc]

    def test_accepts_nested_namespace(self) -> None:
        edge = _make_edge(metadata={"origin": {"is_self": True, "confidence": "high"}})
        assert edge.metadata == {"origin": {"is_self": True, "confidence": "high"}}

    def test_rejects_list_value_at_construction(self) -> None:
        """The recursive ``MetadataValue`` alias forbids lists on edges too."""
        with pytest.raises(ValidationError):
            EdgeRecord(
                edge_id="e-bad",
                from_thought_id="t-1",
                to_thought_id="t-2",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=0,
                metadata=cast("dict[str, MetadataValue]", {"tags": ["a", "b"]}),
            )


# ---------------------------------------------------------------------------
# Persistence round trips (INSERT / UPDATE / READ through SqliteEngravaCore)
# ---------------------------------------------------------------------------


class TestEdgeMetadataPersistence:
    """End-to-end INSERT/UPDATE/READ round trips through ``create_edge``."""

    async def _fetch(self, store: SqliteEngravaCore, edge_id: str) -> EdgeRecord:
        edges = {e.edge_id: e for e in await store.get_edges("t-1")}
        return edges[edge_id]

    async def test_default_empty_metadata_round_trip(self, store: SqliteEngravaCore) -> None:
        await store.create_edge(_make_edge("e-default"))
        fetched = await self._fetch(store, "e-default")
        assert fetched.metadata == {}

    async def test_metadata_round_trip_mixed_types(self, store: SqliteEngravaCore) -> None:
        meta: dict[str, MetadataValue] = {
            "subtype": "supports",
            "sequence": 5,
            "weight_hint": 0.85,
            "is_primary": True,
            "note": None,
        }
        await store.create_edge(_make_edge("e-mixed", metadata=meta))
        fetched = await self._fetch(store, "e-mixed")
        assert fetched.metadata == meta

    async def test_metadata_bool_preserved_distinct_from_int(
        self, store: SqliteEngravaCore
    ) -> None:
        """``bool`` ⊂ ``int`` in Python — JSON keeps them distinguishable."""
        await store.create_edge(
            _make_edge("e-bool", metadata={"is_primary": True, "is_last": False, "sequence": 1}),
        )
        fetched = await self._fetch(store, "e-bool")
        assert fetched.metadata["is_primary"] is True
        assert fetched.metadata["is_last"] is False
        assert fetched.metadata["sequence"] == 1
        assert not isinstance(fetched.metadata["sequence"], bool)

    async def test_metadata_unicode_preserved(self, store: SqliteEngravaCore) -> None:
        """``ensure_ascii=False`` preserves non-ASCII attribute values."""
        await store.create_edge(
            _make_edge("e-unicode", metadata={"label": "Łukasz", "city": "Kraków"}),
        )
        fetched = await self._fetch(store, "e-unicode")
        assert fetched.metadata["label"] == "Łukasz"
        assert fetched.metadata["city"] == "Kraków"

    async def test_nested_namespace_round_trip(self, store: SqliteEngravaCore) -> None:
        meta: dict[str, MetadataValue] = {"origin": {"is_self": True, "confidence": "high"}}
        await store.create_edge(_make_edge("e-nested", metadata=meta))
        fetched = await self._fetch(store, "e-nested")
        assert fetched.metadata == meta

    async def test_update_metadata_persists(self, store: SqliteEngravaCore) -> None:
        """UPDATE writes modified metadata (read+write patched as one unit)."""
        await store.create_edge(_make_edge("e-upd", metadata={"subtype": "supports"}))
        await store.update_edge("e-upd", metadata={"subtype": "refutes"})
        fetched = await self._fetch(store, "e-upd")
        assert fetched.metadata == {"subtype": "refutes"}

    async def test_update_unrelated_field_preserves_metadata(
        self, store: SqliteEngravaCore
    ) -> None:
        """CRITICAL: updating a non-metadata field must not wipe stored metadata.

        This is the atomic-unit coupling guard: because ``_row_to_edge`` (read)
        and the ``update_edge`` UPDATE (write) are patched together,
        ``update_edge``'s merge sees the real stored metadata and preserves it.
        A read-side regression would silently zero it here.
        """
        await store.create_edge(_make_edge("e-keep", metadata={"subtype": "supports", "seq": 2}))
        await store.update_edge("e-keep", weight=0.9)
        fetched = await self._fetch(store, "e-keep")
        assert fetched.metadata == {"subtype": "supports", "seq": 2}
        assert fetched.weight == 0.9

    async def test_update_metadata_is_last_writer_wins_wholesale(
        self, store: SqliteEngravaCore
    ) -> None:
        """Regression: ``update_edge`` replaces metadata wholesale (no key merge).

        Pins the current last-writer-wins semantics — the second update's
        ``metadata`` fully replaces the first's; keys are not deep-merged and
        there is no optimistic-concurrency guard. (Documented as a tracked
        follow-up, not changed by this feature.)
        """
        await store.create_edge(_make_edge("e-lww", metadata={"a": 1}))
        await store.update_edge("e-lww", metadata={"b": 2})
        fetched = await self._fetch(store, "e-lww")
        assert fetched.metadata == {"b": 2}

    async def test_derived_edge_writes_empty_metadata(self, store: SqliteEngravaCore) -> None:
        """The derived-records path writes an empty ``{}`` metadata mapping (no populate)."""
        await store._insert_derived_edge("t-1", "t-2", cycle=3)
        edges = await store.get_edges("t-1", direction="OUT")
        derived = [e for e in edges if e.edge_type is EdgeType.DERIVED_FROM]
        assert len(derived) == 1
        assert derived[0].metadata == {}

    async def test_derived_edge_binds_empty_literal_not_record_metadata(
        self,
        store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The derived INSERT persists the literal ``'{}'`` regardless of the record.

        ``_insert_derived_edge`` binds the empty-object literal for the
        ``metadata_json`` column instead of serializing the ``EdgeRecord`` it
        builds, so a future change to that internal record cannot smuggle
        unvalidated metadata through this validation-free provenance path.
        Forcing the internally constructed record to carry a non-finite value —
        which ``_validate_metadata`` rejects on the caller paths — proves the
        derived INSERT ignores it and stores ``'{}'`` verbatim (the old
        ``json.dumps(edge.metadata, ...)`` binding would have stored
        ``{"smuggled": NaN}`` instead).
        """

        def _smuggle_metadata(**kwargs: object) -> EdgeRecord:
            return EdgeRecord(**{**kwargs, "metadata": {"smuggled": float("nan")}})

        monkeypatch.setattr(
            "engrava.infrastructure.sqlite.engrava_core.EdgeRecord",
            _smuggle_metadata,
        )

        await store._insert_derived_edge("t-1", "t-2", cycle=3)

        cursor = await store._db.execute(
            "SELECT metadata_json FROM edge WHERE edge_type = ?",
            (EdgeType.DERIVED_FROM.value,),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["metadata_json"] == "{}"


# ---------------------------------------------------------------------------
# Validation on the edge write paths (shared ``_validate_metadata``)
# ---------------------------------------------------------------------------


class TestEdgeMetadataValidation:
    """The edge write paths enforce the shared metadata contract."""

    async def test_create_edge_rejects_list_value(self, store: SqliteEngravaCore) -> None:
        bad = EdgeRecord(
            edge_id="e-list",
            from_thought_id="t-1",
            to_thought_id="t-2",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=0,
            metadata={"ns": {"deep": "ok"}},
        ).model_copy(update={"metadata": cast("dict[str, MetadataValue]", {"tags": ["a"]})})
        with pytest.raises(ValueError, match="not allowed"):
            await store.create_edge(bad)

    @pytest.mark.parametrize("bad_float", [float("nan"), float("inf"), float("-inf")])
    async def test_create_edge_rejects_non_finite_float(
        self, store: SqliteEngravaCore, bad_float: float
    ) -> None:
        """CRITICAL: NaN / ±Infinity are rejected (else the row is unfilterable)."""
        bad = _make_edge("e-nan", metadata={"score": bad_float})
        with pytest.raises(ValueError, match="finite number"):
            await store.create_edge(bad)

    async def test_create_edge_rejects_oversized_metadata(self, store: SqliteEngravaCore) -> None:
        meta: dict[str, MetadataValue] = {"huge": "x" * 70_000}
        with pytest.raises(ValueError, match="exceeds maximum"):
            await store.create_edge(_make_edge("e-big", metadata=meta))

    async def test_create_edge_accepts_metadata_just_under_limit(
        self, store: SqliteEngravaCore
    ) -> None:
        """A payload just under the 64 KiB hard limit is accepted and round-trips."""
        meta: dict[str, MetadataValue] = {"blob": "x" * 60_000}
        await store.create_edge(_make_edge("e-boundary", metadata=meta))
        edges = {e.edge_id: e for e in await store.get_edges("t-1")}
        assert edges["e-boundary"].metadata == meta

    async def test_update_edge_rejects_non_finite_float(self, store: SqliteEngravaCore) -> None:
        """Validation parity: ``update_edge`` enforces the same finite-float rule."""
        await store.create_edge(_make_edge("e-updnan", metadata={"score": 1.0}))
        with pytest.raises(ValueError, match="finite number"):
            await store.update_edge("e-updnan", metadata={"score": float("nan")})

    async def test_update_edge_rejects_oversized_metadata(self, store: SqliteEngravaCore) -> None:
        await store.create_edge(_make_edge("e-updbig", metadata={"role": "primary"}))
        with pytest.raises(ValueError, match="exceeds maximum"):
            await store.update_edge("e-updbig", metadata={"huge": "x" * 70_000})


# ---------------------------------------------------------------------------
# Typed MetadataFilter composition on list_edges
# ---------------------------------------------------------------------------


async def _seed_filter_edges(store: SqliteEngravaCore) -> None:
    """Four edges from ``t-1`` with distinct endpoints (UNIQUE-safe)."""
    await store.create_edge(
        _make_edge("e1", to_thought_id="t-2", metadata={"subtype": "supports", "seq": 1}),
    )
    await store.create_edge(
        _make_edge("e2", to_thought_id="t-3", metadata={"subtype": "refutes", "seq": 2}),
    )
    await store.create_edge(
        _make_edge(
            "e3",
            to_thought_id="t-4",
            edge_type=EdgeType.CONSOLIDATED_FROM,
            source=KnowledgeSource.DREAMING,
            metadata={"subtype": "supports", "seq": 3},
        ),
    )
    await store.create_edge(_make_edge("e4", to_thought_id="t-5", metadata={}))


def _ids(edges: list[EdgeRecord]) -> list[str]:
    return sorted(edge.edge_id for edge in edges)


class TestEdgeMetadataFilter:
    """``list_edges(filters=...)`` reuses the shipped MetadataFilter machinery."""

    async def test_eq_predicate(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        f = MetadataFilter([FieldPredicate("$.subtype", FieldOp.EQ, "supports")])
        assert _ids(await store.list_edges(filters=f)) == ["e1", "e3"]

    async def test_in_predicate(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        f = MetadataFilter([FieldPredicate("$.seq", FieldOp.IN, (1, 3))])
        assert _ids(await store.list_edges(filters=f)) == ["e1", "e3"]

    async def test_absent_path_is_non_match(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        f = MetadataFilter([FieldPredicate("$.missing", FieldOp.EQ, "x")])
        assert await store.list_edges(filters=f) == []

    async def test_eq_none_matches_missing_path(self, store: SqliteEngravaCore) -> None:
        """``EQ None`` matches rows missing the path (and the empty-metadata row)."""
        await _seed_filter_edges(store)
        f = MetadataFilter([FieldPredicate("$.subtype", FieldOp.EQ, None)])
        # e4 has empty metadata (no $.subtype) -> matches the IS NULL branch.
        assert "e4" in _ids(await store.list_edges(filters=f))

    async def test_none_filter_leaves_result_unchanged(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        assert len(await store.list_edges(filters=None)) == 4
        assert len(await store.list_edges()) == 4

    async def test_empty_filter_is_match_all(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        assert len(await store.list_edges(filters=MetadataFilter([]))) == 4

    async def test_conjunction_with_edge_type(self, store: SqliteEngravaCore) -> None:
        """The metadata predicate is AND-conjoined with existing predicates."""
        await _seed_filter_edges(store)
        f = MetadataFilter([FieldPredicate("$.subtype", FieldOp.EQ, "supports")])
        # e1 (ASSOCIATED, supports) matches; e3 (CONSOLIDATED_FROM, supports) does not.
        assert _ids(await store.list_edges(edge_type=EdgeType.ASSOCIATED, filters=f)) == ["e1"]

    async def test_conjunction_with_source(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        f = MetadataFilter([FieldPredicate("$.subtype", FieldOp.EQ, "supports")])
        got = await store.list_edges(source=KnowledgeSource.DREAMING, filters=f)
        assert _ids(got) == ["e3"]

    async def test_limit_applies_with_filter(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        f = MetadataFilter([FieldPredicate("$.subtype", FieldOp.EQ, "supports")])
        assert len(await store.list_edges(filters=f, limit=1)) == 1

    async def test_malformed_metadata_row_is_non_match_not_error(
        self, store: SqliteEngravaCore
    ) -> None:
        """A row with malformed ``metadata_json`` is non-matching, never an error.

        Malformed JSON cannot arise via the API (validate + json.dumps always
        produce valid JSON); this corrupts the column directly to exercise the
        ``json_valid`` guard on the filter path.
        """
        await store.create_edge(_make_edge("e-ok", to_thought_id="t-2", metadata={"k": "v"}))
        await store._db.execute(
            "UPDATE edge SET metadata_json = '{bad json' WHERE edge_id = 'e-ok'",
        )
        await store._db.commit()
        f = MetadataFilter([FieldPredicate("$.k", FieldOp.EQ, "v")])
        assert await store.list_edges(filters=f) == []

    async def test_large_valid_metadata_is_filterable(self, store: SqliteEngravaCore) -> None:
        """A large-but-valid metadata payload (near the 64 KiB cap) still filters."""
        big = "y" * 60_000
        await store.create_edge(
            _make_edge("e-big", to_thought_id="t-2", metadata={"tag": "hit", "blob": big}),
        )
        await store.create_edge(
            _make_edge("e-small", to_thought_id="t-3", metadata={"tag": "miss"})
        )
        f = MetadataFilter([FieldPredicate("$.tag", FieldOp.EQ, "hit")])
        assert _ids(await store.list_edges(filters=f)) == ["e-big"]

    async def test_read_only_store_delegates_filter(self, store: SqliteEngravaCore) -> None:
        await _seed_filter_edges(store)
        ro = ReadOnlyEngrava(store)
        f = MetadataFilter([FieldPredicate("$.subtype", FieldOp.EQ, "supports")])
        assert _ids(await ro.list_edges(filters=f)) == ["e1", "e3"]


# ---------------------------------------------------------------------------
# Column-driven snapshot round trip (CLI export -> import)
# ---------------------------------------------------------------------------


async def _build_edge_db(path: Path, metadata: dict[str, MetadataValue]) -> None:
    conn = await aiosqlite.connect(str(path))
    conn.row_factory = aiosqlite.Row
    core = SqliteEngravaCore(conn)
    await core.ensure_schema()
    await core.create_thought(_make_thought("t-1"))
    await core.create_thought(_make_thought("t-2"))
    await core.create_edge(_make_edge("edge-meta", metadata=metadata))
    await conn.commit()
    await conn.close()


async def _read_edge_metadata(path: Path, edge_id: str) -> dict[str, MetadataValue]:
    conn = await aiosqlite.connect(str(path))
    conn.row_factory = aiosqlite.Row
    core = SqliteEngravaCore(conn)
    await core.ensure_schema()
    edges = {e.edge_id: e for e in await core.get_edges("t-1")}
    await conn.close()
    return edges[edge_id].metadata


class TestEdgeMetadataSnapshotRoundTrip:
    """A v19 edge's generic metadata survives a CLI snapshot -> restore.

    The snapshot export/import is dynamic-column, so the ``metadata_json``
    column is carried automatically once it exists — this pins that it is not
    silently dropped.
    """

    def test_snapshot_restore_preserves_edge_metadata(self, tmp_path: Path) -> None:
        meta: dict[str, MetadataValue] = {
            "subtype": "supports",
            "sequence": 7,
            "city": "Kraków",
            "origin": {"is_self": True},
        }
        src = tmp_path / "src.db"
        asyncio.run(_build_edge_db(src, meta))

        runner = CliRunner()
        snap = tmp_path / "snap.jsonl"
        exported = runner.invoke(cli, ["--db", str(src), "snapshot", "-o", str(snap)])
        assert exported.exit_code == 0, exported.output

        restored = tmp_path / "restored.db"
        imported = runner.invoke(cli, ["--db", str(restored), "restore", "-i", str(snap)])
        assert imported.exit_code == 0, imported.output

        assert asyncio.run(_read_edge_metadata(restored, "edge-meta")) == meta

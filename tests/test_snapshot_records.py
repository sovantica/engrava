"""Tests for the snapshot-restore input boundary.

Covers the typed record model and fixed-SQL builder in
``engrava.cli.snapshot_records`` and the streaming, validate-before-write
restore path in ``engrava.cli.main``:

* fixed SQL never derives an identifier from snapshot data;
* a non-object record, non-object ``data``, unknown column, and missing
  required column each raise a typed CLI error before any write;
* a valid snapshot round-trips byte-for-byte across every column;
* restore streams the snapshot and re-embeds in bounded batches, so memory is
  bounded by the batch rather than the snapshot size.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import struct
from typing import TYPE_CHECKING

import aiosqlite
import click
import pytest
from click.testing import CliRunner

from engrava.cli import main as cli_main
from engrava.cli.main import (
    _assert_embedding_model_match,
    _import_records_to_db,
    _iter_snapshot_lines,
    cli,
)
from engrava.cli.snapshot_records import (
    ColumnKind,
    CoreTable,
    IgnoredRecord,
    InvalidColumnValueError,
    InvalidEmbeddingBlobError,
    MalformedMetadataError,
    MalformedRecordDataError,
    MalformedSnapshotRecordError,
    MetadataRecord,
    MissingRequiredColumnError,
    TableRecord,
    UnknownSnapshotColumnError,
    parse_snapshot_record,
    table_spec,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click test runner."""
    return CliRunner()


def _minimal_thought_data(thought_id: str = "t-1") -> dict[str, object]:
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


def _embedding_data(vector: bytes, owner_id: str = "t-1") -> dict[str, object]:
    """Return an embedding ``data`` mapping with a base64-encoded blob."""
    return {
        "embedding_id": "e-1",
        "owner_type": "thought",
        "owner_id": owner_id,
        "model_name": "m",
        "dimension": 4,
        "vector_blob": base64.b64encode(vector).decode("ascii"),
        "created_at": "2026-01-01T00:00:00+00:00",
    }


async def _fresh_schema_conn(db_path: Path) -> aiosqlite.Connection:
    """Create a database with the core schema applied and return its connection."""
    from engrava import SqliteEngravaCore

    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    await conn.commit()
    return conn


async def _live_columns(conn: aiosqlite.Connection, table: str) -> list[aiosqlite.Row]:
    """Return ``PRAGMA table_info`` rows for a table."""
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    return list(await cursor.fetchall())


# ---------------------------------------------------------------------------
# Schema-parity guard: the hand-written specs must mirror the live schema.
# ---------------------------------------------------------------------------


class TestSchemaParity:
    """The immutable specs must not drift from ``schema_core.sql``."""

    def test_specs_match_live_schema(self, tmp_path: Path) -> None:
        declared_kind = {
            "TEXT": ColumnKind.TEXT,
            "INTEGER": ColumnKind.INTEGER,
            "REAL": ColumnKind.REAL,
            "BLOB": ColumnKind.BLOB,
        }

        async def _check() -> None:
            conn = await _fresh_schema_conn(tmp_path / "schema.db")
            try:
                for table in CoreTable:
                    rows = await _live_columns(conn, table.value)
                    live_columns = tuple(row["name"] for row in rows)
                    live_required = frozenset(
                        row["name"]
                        for row in rows
                        # Required = primary key, or NOT NULL without a default.
                        if row["pk"] > 0 or (row["notnull"] == 1 and row["dflt_value"] is None)
                    )
                    live_kinds = {row["name"]: declared_kind[row["type"]] for row in rows}
                    spec = table_spec(table)
                    assert spec.columns == live_columns, table.value
                    assert spec.required == live_required, table.value
                    assert dict(spec.column_types) == live_kinds, table.value
            finally:
                await conn.close()

        asyncio.run(_check())


# ---------------------------------------------------------------------------
# Fixed SQL: identifiers come only from the spec, never from snapshot data.
# ---------------------------------------------------------------------------


class TestFixedSql:
    """``TableSpec.build_insert`` emits fixed, allow-listed SQL."""

    def test_identifiers_only_from_spec_in_canonical_order(self) -> None:
        spec = table_spec(CoreTable.THOUGHT)
        # Supply the columns out of order and as a partial subset.
        data: dict[str, object] = {
            "content": "c",
            "thought_id": "t-1",
            "priority": "P2",
            "essence": "e",
            "thought_type": "OBSERVATION",
        }
        sql, values = spec.build_insert(data)
        # Column list is the spec's canonical order, not the input order.
        assert sql == (
            "INSERT OR REPLACE INTO thought "
            "(thought_id, thought_type, essence, content, priority) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        assert values == ("t-1", "OBSERVATION", "e", "c", "P2")

    def test_full_column_insert_binds_every_column(self) -> None:
        spec = table_spec(CoreTable.EDGE)
        data: dict[str, object] = dict.fromkeys(spec.columns, "x")
        sql, values = spec.build_insert(data)
        column_list = ", ".join(spec.columns)
        placeholders = ", ".join("?" for _ in spec.columns)
        expected = f"INSERT OR REPLACE INTO edge ({column_list}) VALUES ({placeholders})"  # noqa: S608
        assert sql == expected
        assert len(values) == len(spec.columns)

    def test_injection_shaped_key_is_rejected_not_interpolated(self) -> None:
        # A key crafted to escape the identifier list must be rejected as an
        # unknown column and never reach the SQL text.
        malicious = "thought_id) VALUES ('x'); DROP TABLE thought;--"
        data = _minimal_thought_data()
        data[malicious] = "boom"
        line = _thought_line(data)
        with pytest.raises(UnknownSnapshotColumnError) as exc:
            parse_snapshot_record(line, line_number=7)
        assert malicious in exc.value.columns
        assert exc.value.line_number == 7


# ---------------------------------------------------------------------------
# Typed validation: each malformed shape raises its typed error.
# ---------------------------------------------------------------------------


class TestParseValidation:
    """``parse_snapshot_record`` rejects malformed records with typed errors."""

    @pytest.mark.parametrize("raw", ["[1, 2, 3]", "42", '"a string"', "null", "{not json"])
    def test_non_object_record_rejected(self, raw: str) -> None:
        with pytest.raises(MalformedSnapshotRecordError) as exc:
            parse_snapshot_record(raw, line_number=3)
        assert exc.value.line_number == 3

    @pytest.mark.parametrize("data", [42, "text", [1, 2], None])
    def test_non_object_data_rejected(self, data: object) -> None:
        line = json.dumps({"_type": "thought", "data": data})
        with pytest.raises(MalformedRecordDataError) as exc:
            parse_snapshot_record(line, line_number=4)
        assert exc.value.table is CoreTable.THOUGHT
        assert exc.value.line_number == 4

    def test_missing_data_member_rejected(self) -> None:
        line = json.dumps({"_type": "edge"})
        with pytest.raises(MalformedRecordDataError):
            parse_snapshot_record(line, line_number=1)

    def test_unknown_column_rejected(self) -> None:
        data = _minimal_thought_data()
        data["surprise"] = "value"
        with pytest.raises(UnknownSnapshotColumnError) as exc:
            parse_snapshot_record(_thought_line(data), line_number=9)
        assert "surprise" in exc.value.columns

    def test_missing_required_column_rejected(self) -> None:
        data = _minimal_thought_data()
        del data["priority"]
        with pytest.raises(MissingRequiredColumnError) as exc:
            parse_snapshot_record(_thought_line(data), line_number=2)
        assert "priority" in exc.value.columns


class TestParseHappyPath:
    """``parse_snapshot_record`` accepts and classifies well-formed lines."""

    def test_valid_thought_returns_table_record(self) -> None:
        data = _minimal_thought_data("t-42")
        record = parse_snapshot_record(_thought_line(data), line_number=1)
        assert isinstance(record, TableRecord)
        assert record.spec.table is CoreTable.THOUGHT
        assert record.data["thought_id"] == "t-42"

    def test_metadata_returns_metadata_record(self) -> None:
        line = json.dumps(
            {"_type": "metadata", "schema_version": 20, "embedding_model_name": "model-x"}
        )
        record = parse_snapshot_record(line, line_number=1)
        assert isinstance(record, MetadataRecord)
        assert record.embedding_model_name == "model-x"

    def test_metadata_without_model_name(self) -> None:
        record = parse_snapshot_record(json.dumps({"_type": "metadata"}), line_number=1)
        assert isinstance(record, MetadataRecord)
        assert record.embedding_model_name is None

    def test_legacy_table_key_supported(self) -> None:
        data = _minimal_thought_data()
        line = json.dumps({"table": "thought", "data": data})
        record = parse_snapshot_record(line, line_number=1)
        assert isinstance(record, TableRecord)
        assert record.spec.table is CoreTable.THOUGHT

    def test_unknown_type_is_ignored(self) -> None:
        line = json.dumps({"_type": "widget", "data": {"anything": 1}})
        record = parse_snapshot_record(line, line_number=1)
        assert isinstance(record, IgnoredRecord)
        assert record.record_type == "widget"

    def test_embedding_blob_decoded_at_insert(self) -> None:
        vector = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
        line = json.dumps({"_type": "embedding", "data": _embedding_data(vector)})
        record = parse_snapshot_record(line, line_number=1)
        assert isinstance(record, TableRecord)
        # Parsing keeps the base64 transport string; decoding is deferred.
        assert record.data["vector_blob"] == base64.b64encode(vector).decode("ascii")
        _sql, values = record.to_insert()
        assert vector in values


class TestValueValidation:
    """Value-level input hardening: scalars, metadata, and blob encoding."""

    @pytest.mark.parametrize("bad_value", [[1, 2, 3], {"nested": "object"}])
    def test_non_scalar_column_value_rejected(self, bad_value: object) -> None:
        data = _minimal_thought_data()
        data["metadata_json"] = bad_value
        with pytest.raises(InvalidColumnValueError) as exc:
            parse_snapshot_record(_thought_line(data), line_number=5)
        assert exc.value.column == "metadata_json"
        assert exc.value.line_number == 5

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("confidence", 1),  # REAL accepts an int
            ("confidence", 1.5),  # REAL accepts a float
            ("confidence", None),  # nullable column accepts null
            ("content_hash", "abc"),  # TEXT accepts a string
            ("created_cycle", 7),  # INTEGER accepts an int
            ("pinned", 1),  # INTEGER accepts the 0/1 flag as an int
        ],
    )
    def test_well_typed_column_values_accepted(self, column: str, value: object) -> None:
        data = _minimal_thought_data()
        data[column] = value
        record = parse_snapshot_record(_thought_line(data), line_number=1)
        assert isinstance(record, TableRecord)

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("confidence", "text"),  # REAL rejects a string
            ("essence", 5),  # TEXT rejects a number  (note: essence is required)
            ("created_cycle", "seven"),  # INTEGER rejects a string
            ("created_cycle", 1.5),  # INTEGER rejects a float
            ("pinned", True),  # INTEGER rejects a JSON bool (not a number)
            ("confidence", True),  # REAL rejects a JSON bool
            ("metadata_json", [1, 2, 3]),  # any column rejects a non-scalar
        ],
    )
    def test_mistyped_column_value_rejected(self, column: str, value: object) -> None:
        data = _minimal_thought_data()
        data[column] = value
        with pytest.raises(InvalidColumnValueError) as exc:
            parse_snapshot_record(_thought_line(data), line_number=6)
        assert exc.value.column == column
        assert exc.value.line_number == 6

    @pytest.mark.parametrize("null_required", ["thought_id", "priority"])
    def test_null_required_column_rejected(self, null_required: str) -> None:
        # SQLite allows null in a TEXT primary key, so a null required column
        # must be rejected explicitly rather than silently inserted.
        data = _minimal_thought_data()
        data[null_required] = None
        with pytest.raises(MissingRequiredColumnError) as exc:
            parse_snapshot_record(_thought_line(data), line_number=3)
        assert null_required in exc.value.columns

    def test_non_string_metadata_model_rejected(self) -> None:
        line = json.dumps({"_type": "metadata", "embedding_model_name": 123})
        with pytest.raises(MalformedMetadataError) as exc:
            parse_snapshot_record(line, line_number=1)
        assert exc.value.line_number == 1

    def test_invalid_base64_blob_raises_typed_error(self) -> None:
        data = _embedding_data(b"")
        data["vector_blob"] = "not valid base64 !!!"
        line = json.dumps({"_type": "embedding", "data": data})
        record = parse_snapshot_record(line, line_number=8)
        assert isinstance(record, TableRecord)
        with pytest.raises(InvalidEmbeddingBlobError) as exc:
            record.to_insert()
        assert exc.value.line_number == 8

    def test_non_string_blob_rejected_at_parse(self) -> None:
        # A numeric blob must not slip into the BLOB column via type affinity;
        # validation rejects it (BLOB expects the base64 transport string).
        data = _embedding_data(b"")
        data["vector_blob"] = 12345
        line = json.dumps({"_type": "embedding", "data": data})
        with pytest.raises(InvalidColumnValueError) as exc:
            parse_snapshot_record(line, line_number=2)
        assert exc.value.column == "vector_blob"

    def test_to_insert_guards_non_string_blob(self) -> None:
        # Defence in depth: even a hand-built record with a non-string blob is
        # rejected by ``to_insert`` rather than binding a non-blob value.
        record = TableRecord(
            spec=table_spec(CoreTable.EMBEDDING),
            data={**_embedding_data(b""), "vector_blob": 12345},
            line_number=4,
        )
        with pytest.raises(InvalidEmbeddingBlobError):
            record.to_insert()


# ---------------------------------------------------------------------------
# Streaming line iterator.
# ---------------------------------------------------------------------------


class TestStreamingIterator:
    """``_iter_snapshot_lines`` streams lazily with correct line numbers."""

    def test_yields_lazily_with_line_numbers(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.jsonl"
        snap.write_text("first\n\n  \nthird\n", encoding="utf-8")
        gen = _iter_snapshot_lines(snap)
        assert inspect.isgenerator(gen)  # streaming, not a materialised list
        assert list(gen) == [(1, "first"), (4, "third")]


# ---------------------------------------------------------------------------
# Integration: validate-before-write and round-trip identity.
# ---------------------------------------------------------------------------


def _seed_db(db_path: Path) -> None:
    """Populate a database with thoughts, an edge, embeddings, and an action."""
    from engrava import (
        EdgeRecord,
        EdgeType,
        LifecycleStatus,
        Priority,
        SqliteEngravaCore,
        ThoughtRecord,
        ThoughtType,
    )

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        for i in range(3):
            thought = ThoughtRecord(
                thought_id=f"thought-{i:03d}",
                essence=f"essence {i}",
                content=f"content {i}",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=i + 1,
                updated_cycle=i + 1,
            )
            await store.create_thought(thought)
            await store.store_embedding(f"thought-{i:03d}", [float(i)] * 16)
        await store.create_edge(
            EdgeRecord(
                edge_id="edge-001",
                from_thought_id="thought-000",
                to_thought_id="thought-001",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.9,
                created_cycle=1,
            )
        )
        # An action row so the round-trip identity check exercises every table.
        await conn.execute(
            "INSERT INTO action "
            "(action_id, source_thought_id, action_type, intent, status, "
            "verification_status, raw_metrics_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "action-001",
                "thought-000",
                "WRITE",
                "persist a note",
                "PLANNED",
                "PENDING",
                json.dumps({"latency_ms": 12}),
            ),
        )
        # Exercise non-default values in late-added columns so the round-trip
        # identity check covers them explicitly.
        await conn.execute(
            "UPDATE thought SET pinned = 1, metadata_json = ?, provenance = ? "
            "WHERE thought_id = 'thought-000'",
            (json.dumps({"tag": "keep"}), json.dumps({"session_id": "s-1"})),
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())


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


class TestRestoreInputBoundary:
    """The CLI restore command enforces the input boundary before any write."""

    def _snapshot(self, runner: CliRunner, source: Path, out: Path) -> None:
        result = runner.invoke(cli, ["--db", str(source), "snapshot", "-o", str(out)])
        assert result.exit_code == 0

    def test_roundtrip_is_byte_identical(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "source.db"
        _seed_db(source)
        snap = tmp_path / "snap.jsonl"
        self._snapshot(runner, source, snap)

        target = tmp_path / "restored.db"
        result = runner.invoke(cli, ["--db", str(target), "restore", "-i", str(snap)])
        assert result.exit_code == 0

        for table in ("thought", "edge", "embedding", "action"):
            source_rows = asyncio.run(_dump_table(source, table))
            assert source_rows, f"seed left {table} empty"  # every table exercised
            assert source_rows == asyncio.run(_dump_table(target, table)), table

    @pytest.mark.parametrize(
        ("bad_line", "expected"),
        [
            ("[1, 2, 3]", "not a valid JSON object"),
            (json.dumps({"_type": "thought", "data": 5}), "non-object"),
            (
                json.dumps({"_type": "thought", "data": {**_minimal_thought_data(), "x": 1}}),
                "unknown column",
            ),
            (
                json.dumps(
                    {
                        "_type": "thought",
                        "data": {
                            k: v for k, v in _minimal_thought_data().items() if k != "priority"
                        },
                    }
                ),
                "missing required column",
            ),
        ],
    )
    def test_malformed_line_rejected_before_any_write(
        self, runner: CliRunner, tmp_path: Path, bad_line: str, expected: str
    ) -> None:
        # A valid record precedes the malformed one; nothing must be written.
        good_line = _thought_line(_minimal_thought_data("thought-ok"))
        snap = tmp_path / "bad.jsonl"
        snap.write_text(
            json.dumps({"_type": "metadata", "schema_version": 20})
            + "\n"
            + good_line
            + "\n"
            + bad_line
            + "\n",
            encoding="utf-8",
        )

        target = tmp_path / "target.db"
        result = runner.invoke(cli, ["--db", str(target), "restore", "-i", str(snap)])
        assert result.exit_code != 0
        assert expected in result.output
        # The preceding valid record must not have been written.
        assert asyncio.run(_dump_table(target, "thought")) == []

    def test_clear_rolled_back_on_structural_error(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "source.db"
        _seed_db(source)

        bad_snap = tmp_path / "bad.jsonl"
        bad_snap.write_text(
            json.dumps({"_type": "metadata", "schema_version": 20}) + "\n[1, 2, 3]\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            cli, ["--db", str(source), "restore", "-i", str(bad_snap), "--clear"]
        )
        assert result.exit_code != 0
        # The whole transaction (including --clear) is rolled back.
        assert len(asyncio.run(_dump_table(source, "thought"))) == 3

    def test_bad_value_at_record_n_persists_nothing(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Valid thoughts precede an embedding with an invalid base64 blob; the
        # whole restore must roll back so no partial rows persist.
        bad_embedding = _embedding_data(b"")
        bad_embedding["vector_blob"] = "not valid base64 !!!"
        snap = tmp_path / "bad.jsonl"
        snap.write_text(
            json.dumps({"_type": "metadata", "schema_version": 20})
            + "\n"
            + _thought_line(_minimal_thought_data("thought-a"))
            + "\n"
            + _thought_line(_minimal_thought_data("thought-b"))
            + "\n"
            + json.dumps({"_type": "embedding", "data": bad_embedding})
            + "\n",
            encoding="utf-8",
        )
        target = tmp_path / "target.db"
        result = runner.invoke(cli, ["--db", str(target), "restore", "-i", str(snap)])
        assert result.exit_code != 0
        assert "base64" in result.output
        # The two valid thoughts inserted before the bad record are rolled back.
        assert asyncio.run(_dump_table(target, "thought")) == []

    def test_clear_rolled_back_on_value_error(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "source.db"
        _seed_db(source)
        bad_embedding = _embedding_data(b"")
        bad_embedding["vector_blob"] = "not valid base64 !!!"
        snap = tmp_path / "bad.jsonl"
        snap.write_text(
            json.dumps({"_type": "metadata", "schema_version": 20})
            + "\n"
            + json.dumps({"_type": "embedding", "data": bad_embedding})
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(cli, ["--db", str(source), "restore", "-i", str(snap), "--clear"])
        assert result.exit_code != 0
        # --clear + inserts are one transaction; a value error rolls all of it back.
        assert len(asyncio.run(_dump_table(source, "thought"))) == 3

    def test_skip_embeddings_ignores_bad_base64(self, runner: CliRunner, tmp_path: Path) -> None:
        bad_embedding = _embedding_data(b"")
        bad_embedding["vector_blob"] = "not valid base64 !!!"
        snap = tmp_path / "snap.jsonl"
        snap.write_text(
            json.dumps({"_type": "metadata", "schema_version": 20})
            + "\n"
            + _thought_line(_minimal_thought_data("thought-a"))
            + "\n"
            + json.dumps({"_type": "embedding", "data": bad_embedding})
            + "\n",
            encoding="utf-8",
        )
        target = tmp_path / "target.db"
        result = runner.invoke(
            cli, ["--db", str(target), "restore", "-i", str(snap), "--skip-embeddings"]
        )
        # The skipped embedding is never decoded, so its bad blob is irrelevant.
        assert result.exit_code == 0
        assert len(asyncio.run(_dump_table(target, "thought"))) == 1
        assert asyncio.run(_dump_table(target, "embedding")) == []

    def test_older_snapshot_missing_columns_gets_db_defaults(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # A snapshot from an older schema carries only the required columns; the
        # restored row must fall back to the live schema's column defaults.
        snap = tmp_path / "old.jsonl"
        snap.write_text(
            json.dumps({"_type": "metadata", "schema_version": 1})
            + "\n"
            + _thought_line(_minimal_thought_data("thought-old"))
            + "\n",
            encoding="utf-8",
        )
        target = tmp_path / "target.db"
        result = runner.invoke(cli, ["--db", str(target), "restore", "-i", str(snap)])
        assert result.exit_code == 0

        rows = asyncio.run(_dump_table(target, "thought"))
        assert len(rows) == 1
        row = rows[0]
        assert row["pinned"] == 0  # NOT NULL DEFAULT 0
        assert row["metadata_json"] == "{}"  # NOT NULL DEFAULT '{}'
        assert row["lifecycle_status"] == "CREATED"  # NOT NULL DEFAULT 'CREATED'
        assert row["archived_at"] is None

    def test_legacy_table_key_snapshot_restores(self, runner: CliRunner, tmp_path: Path) -> None:
        snap = tmp_path / "legacy.jsonl"
        snap.write_text(
            json.dumps({"table": "thought", "data": _minimal_thought_data("thought-legacy")})
            + "\n",
            encoding="utf-8",
        )
        target = tmp_path / "target.db"
        result = runner.invoke(cli, ["--db", str(target), "restore", "-i", str(snap)])
        assert result.exit_code == 0
        rows = asyncio.run(_dump_table(target, "thought"))
        assert [row["thought_id"] for row in rows] == ["thought-legacy"]

    def test_skip_embeddings_imports_thoughts_only(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "source.db"
        _seed_db(source)
        snap = tmp_path / "snap.jsonl"
        self._snapshot(runner, source, snap)

        target = tmp_path / "target.db"
        result = runner.invoke(
            cli, ["--db", str(target), "restore", "-i", str(snap), "--skip-embeddings"]
        )
        assert result.exit_code == 0
        assert len(asyncio.run(_dump_table(target, "thought"))) == 3
        assert asyncio.run(_dump_table(target, "embedding")) == []


class _FakeProvider:
    """A minimal embedding provider stand-in (methods are never called here)."""

    model_name = "fake-model"
    dimension = 4


class TestModelMismatchValidation:
    """The metadata model-compatibility check gates restore."""

    def test_mismatch_raises(self) -> None:
        with pytest.raises(click.ClickException, match="model mismatch"):
            _assert_embedding_model_match(MetadataRecord("other-model"), _FakeProvider())

    def test_matching_model_passes(self) -> None:
        _assert_embedding_model_match(MetadataRecord("fake-model"), _FakeProvider())

    def test_absent_model_passes(self) -> None:
        _assert_embedding_model_match(MetadataRecord(None), _FakeProvider())


# ---------------------------------------------------------------------------
# Streaming re-embed: memory is bounded by the batch, not the snapshot.
# ---------------------------------------------------------------------------


class TestBatchedReembed:
    """Re-embedding flushes IDs in bounded batches during the streaming pass."""

    def test_reembed_ids_flushed_in_bounded_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thought_count = 5
        snap = tmp_path / "snap.jsonl"
        lines = [
            json.dumps(
                {"_type": "metadata", "schema_version": 20, "embedding_model_name": "fake-model"}
            )
        ]
        lines.extend(
            _thought_line(_minimal_thought_data(f"thought-{i:03d}")) for i in range(thought_count)
        )
        snap.write_text("\n".join(lines) + "\n", encoding="utf-8")

        observed_batches: list[int] = []

        async def _fake_reembed(_conn: object, thought_ids: list[str], _provider: object) -> int:
            observed_batches.append(len(thought_ids))
            return len(thought_ids)

        monkeypatch.setattr(cli_main, "_REEMBED_BATCH_SIZE", 2)
        monkeypatch.setattr(cli_main, "_reembed_thoughts", _fake_reembed)

        async def _run() -> int:
            conn = await _fresh_schema_conn(tmp_path / "target.db")
            try:
                return await _import_records_to_db(
                    conn,
                    snap,
                    re_embed=True,
                    embedding_provider=_FakeProvider(),  # type: ignore[arg-type]
                )
            finally:
                await conn.close()

        total = asyncio.run(_run())

        # No batch exceeds the (patched) cap, so peak retained IDs are bounded by
        # the batch size rather than the total thought count.
        assert observed_batches == [2, 2, 1]
        assert max(observed_batches) <= 2
        # 5 thoughts inserted + 5 re-embeddings reported.
        assert total == thought_count * 2

    def test_re_embed_ignores_bad_base64(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Under --re-embed the source embedding is skipped, so its bad blob is
        # never decoded and the restore succeeds.
        bad_embedding = _embedding_data(b"")
        bad_embedding["vector_blob"] = "not valid base64 !!!"
        snap = tmp_path / "snap.jsonl"
        snap.write_text(
            json.dumps(
                {"_type": "metadata", "schema_version": 20, "embedding_model_name": "fake-model"}
            )
            + "\n"
            + _thought_line(_minimal_thought_data("thought-a"))
            + "\n"
            + json.dumps({"_type": "embedding", "data": bad_embedding})
            + "\n",
            encoding="utf-8",
        )

        async def _fake_reembed(_conn: object, thought_ids: list[str], _provider: object) -> int:
            return len(thought_ids)

        monkeypatch.setattr(cli_main, "_reembed_thoughts", _fake_reembed)
        target = tmp_path / "target.db"

        async def _run() -> int:
            conn = await _fresh_schema_conn(target)
            try:
                return await _import_records_to_db(
                    conn,
                    snap,
                    re_embed=True,
                    embedding_provider=_FakeProvider(),  # type: ignore[arg-type]
                )
            finally:
                await conn.close()

        asyncio.run(_run())
        assert len(asyncio.run(_dump_table(target, "thought"))) == 1
        assert asyncio.run(_dump_table(target, "embedding")) == []

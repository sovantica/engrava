"""Typed snapshot-record model and input boundary for ``engrava restore``.

A logical snapshot (``engrava snapshot``) is a JSONL file whose lines are
untyped JSON produced by ``SELECT *`` over the core tables. On restore that
JSON crosses a trust boundary into ``INSERT`` statements, so it must be
validated against a fixed, code-owned schema before any write.

This module owns that boundary:

* :class:`CoreTable` -- the immutable allow-list of restorable table names.
* :class:`TableSpec` -- the immutable allowed-column set (and required-column
  subset) for one table, plus a builder that emits **fixed SQL whose column
  identifiers come only from the spec**, never from snapshot data.
* :func:`parse_snapshot_record` -- parses one snapshot line into a typed
  :data:`SnapshotRecord`. Every line must be a JSON object; **schema validation
  then applies only to recognized core-table records** (a non-object ``data``
  member, an unknown column, a missing or null required column, or a non-scalar
  value each raise a typed :class:`SnapshotRestoreError`). A metadata header and
  a record whose ``_type`` is not a core table are returned as their own typed
  variants -- the latter is skipped on restore for forward compatibility with
  future record types -- and neither is ever used to derive a SQL identifier;
  only :class:`TableRecord` values are inserted.

The column sets mirror ``schema_core.sql`` and are guarded against drift by a
parity test that compares them to a freshly created schema.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, TypeAlias, cast

import click

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: A JSON scalar value as it arrives from a snapshot record's ``data`` mapping.
SnapshotScalar: TypeAlias = str | int | float | bool | None
#: A value ready to bind to a SQLite parameter (scalars plus decoded blobs).
SnapshotBindValue: TypeAlias = str | int | float | bool | bytes | None

#: JSON metadata header ``_type``. Metadata lines carry no table data.
METADATA_TYPE = "metadata"

_TYPE_KEY = "_type"
#: Backward-compat key: old snapshots used ``{"table": ..., "data": ...}``.
_LEGACY_TYPE_KEY = "table"
_DATA_KEY = "data"
_EMBEDDING_MODEL_KEY = "embedding_model_name"
_VECTOR_BLOB_COLUMN = "vector_blob"


@unique
class CoreTable(StrEnum):
    """Immutable allow-list of core tables a snapshot may restore into.

    A record whose ``_type`` is not one of these members is ignored by restore
    (forward-compatible with extension record types); it is never used to derive
    a SQL identifier.
    """

    THOUGHT = "thought"
    EDGE = "edge"
    EMBEDDING = "embedding"
    ACTION = "action"


@unique
class ColumnKind(StrEnum):
    """The SQLite storage class a snapshot value must match before it is bound.

    Values mirror the declared column affinities in ``schema_core.sql``. This is
    a bounded per-column type check, not a full emulation of SQLite affinity: a
    value whose Python type does not match the column's kind is rejected before
    any write, so a mistyped snapshot value cannot be persisted into a
    non-``STRICT`` table and break later reads.
    """

    TEXT = "TEXT"
    INTEGER = "INTEGER"
    REAL = "REAL"
    BLOB = "BLOB"


def _value_matches_kind(value: object, kind: ColumnKind) -> bool:
    """Return whether a non-null scalar is acceptable for a column kind.

    Args:
        value: The record's value for the column (never ``None`` here).
        kind: The column's declared storage kind.

    Returns:
        ``True`` if the value's Python type is acceptable for the kind. ``BLOB``
        expects the base64 transport string. ``bool`` is **rejected** for numeric
        kinds: although Python's ``bool`` is an ``int`` subclass, a JSON boolean
        is not a number, and an Engrava-produced snapshot always serialises
        integer/real columns (including the 0/1 ``pinned`` flag) as numbers — so a
        ``true``/``false`` for a numeric column is malformed input, not a 0/1.

    """
    if kind in (ColumnKind.TEXT, ColumnKind.BLOB):
        return isinstance(value, str)
    if kind is ColumnKind.INTEGER:
        return type(value) is int  # reject bool (an int subclass) and float
    return type(value) in (int, float)  # REAL accepts int or float, rejects bool


class SnapshotRestoreError(click.ClickException):
    """Base class for snapshot-restore input-boundary violations.

    A :class:`click.ClickException` so the CLI reports it as a typed error and
    exits non-zero rather than raising an opaque traceback.
    """


class MalformedSnapshotRecordError(SnapshotRestoreError):
    """Raised when a snapshot line is not a valid JSON object.

    Args:
        line_number: 1-based line number of the offending record.

    """

    def __init__(self, line_number: int) -> None:
        self.line_number = line_number
        super().__init__(f"Snapshot line {line_number} is not a valid JSON object.")


class MalformedRecordDataError(SnapshotRestoreError):
    """Raised when a core-table record's ``data`` member is not an object.

    Args:
        table: The core table the record targets.
        line_number: 1-based line number of the offending record.

    """

    def __init__(self, table: CoreTable, line_number: int) -> None:
        self.table = table
        self.line_number = line_number
        super().__init__(
            f"Snapshot line {line_number} for table {table.value!r} "
            f"has a non-object {_DATA_KEY!r} member."
        )


class UnknownSnapshotColumnError(SnapshotRestoreError):
    """Raised when a record carries a column outside the table's allowed set.

    Args:
        table: The core table the record targets.
        columns: The offending column names, sorted.
        line_number: 1-based line number of the offending record.

    """

    def __init__(self, table: CoreTable, columns: Sequence[str], line_number: int) -> None:
        self.table = table
        self.columns = tuple(columns)
        self.line_number = line_number
        super().__init__(
            f"Snapshot line {line_number} for table {table.value!r} "
            f"contains unknown column(s): {', '.join(columns)}."
        )


class MissingRequiredColumnError(SnapshotRestoreError):
    """Raised when a record omits a column the table requires.

    Args:
        table: The core table the record targets.
        columns: The missing required column names, sorted.
        line_number: 1-based line number of the offending record.

    """

    def __init__(self, table: CoreTable, columns: Sequence[str], line_number: int) -> None:
        self.table = table
        self.columns = tuple(columns)
        self.line_number = line_number
        super().__init__(
            f"Snapshot line {line_number} for table {table.value!r} "
            f"is missing required column(s): {', '.join(columns)}."
        )


class InvalidColumnValueError(SnapshotRestoreError):
    """Raised when a column value has the wrong type for its column.

    A value must be a JSON scalar whose Python type matches the column's kind
    (text, integer, real, or a base64 string for a blob). A nested array/object,
    or a scalar of the wrong kind (a string for a numeric column), is rejected.

    Args:
        table: The core table the record targets.
        column: The offending column name.
        line_number: 1-based line number of the offending record.

    """

    def __init__(self, table: CoreTable, column: str, line_number: int) -> None:
        self.table = table
        self.column = column
        self.line_number = line_number
        super().__init__(
            f"Snapshot line {line_number} for table {table.value!r} "
            f"has an invalid value for column {column!r}."
        )


class MalformedMetadataError(SnapshotRestoreError):
    """Raised when a metadata header carries a non-string embedding model name.

    Args:
        line_number: 1-based line number of the offending record.

    """

    def __init__(self, line_number: int) -> None:
        self.line_number = line_number
        super().__init__(
            f"Snapshot line {line_number} metadata has a non-string {_EMBEDDING_MODEL_KEY!r}."
        )


class InvalidEmbeddingBlobError(SnapshotRestoreError):
    """Raised when an embedding record's ``vector_blob`` is not a base64 string.

    Args:
        line_number: 1-based line number of the offending record.

    """

    def __init__(self, line_number: int) -> None:
        self.line_number = line_number
        super().__init__(
            f"Snapshot line {line_number} has a {_VECTOR_BLOB_COLUMN!r} "
            f"that is not a valid base64 string."
        )


@dataclass(frozen=True, slots=True)
class TableSpec:
    """The immutable restore contract for one core table.

    Attributes:
        table: The table this spec governs.
        columns: The allowed columns, in canonical schema order. These are the
            only identifiers ever emitted into restore SQL.
        required: The subset of :attr:`columns` a record must supply (the
            primary key plus every ``NOT NULL`` column without a default).

    """

    table: CoreTable
    columns: tuple[str, ...]
    required: frozenset[str]
    column_types: Mapping[str, ColumnKind]

    def validate(self, data: Mapping[str, object], *, line_number: int) -> None:
        """Validate a record's columns and values against this spec.

        Args:
            data: The record's ``data`` mapping (column name to value).
            line_number: 1-based line number, for error context.

        Raises:
            UnknownSnapshotColumnError: If ``data`` has a column not in
                :attr:`columns`.
            MissingRequiredColumnError: If ``data`` omits a :attr:`required`
                column or supplies it as ``null`` (a required column, including
                the primary key, must carry a real value).
            InvalidColumnValueError: If a value's type does not match its
                column's :class:`ColumnKind`.

        """
        allowed = set(self.columns)
        unknown = sorted(key for key in data if key not in allowed)
        if unknown:
            raise UnknownSnapshotColumnError(self.table, unknown, line_number)
        # A required column must be present AND non-null. SQLite permits a null
        # in a non-integer primary key, so an explicit check is what keeps a
        # ``"thought_id": null`` record from inserting a keyless row.
        missing = sorted(column for column in self.required if data.get(column) is None)
        if missing:
            raise MissingRequiredColumnError(self.table, missing, line_number)
        for column, value in data.items():
            # A null is acceptable here for any nullable column; required
            # columns were already proven non-null above.
            if value is not None and not _value_matches_kind(value, self.column_types[column]):
                raise InvalidColumnValueError(self.table, column, line_number)

    def build_insert(
        self, data: Mapping[str, SnapshotBindValue]
    ) -> tuple[str, tuple[SnapshotBindValue, ...]]:
        """Build a fixed ``INSERT`` statement and its aligned value tuple.

        The column identifiers are taken exclusively from :attr:`columns`
        (filtered to those present in ``data``), so no identifier is ever
        derived from snapshot data. The record's values are bound as parameters.

        Args:
            data: A validated, bind-ready record mapping (columns are a subset of
                :attr:`columns`).

        Returns:
            A ``(sql, values)`` pair ready for ``execute``.

        """
        present = tuple(column for column in self.columns if column in data)
        columns_sql = ", ".join(present)
        placeholders = ", ".join("?" for _ in present)
        # Only the table name and allow-listed column constants reach the SQL
        # text; every value travels as a bound parameter, so this is not an
        # injection surface despite the f-string.
        sql = (
            f"INSERT OR REPLACE INTO {self.table.value} "  # noqa: S608
            f"({columns_sql}) VALUES ({placeholders})"
        )
        values = tuple(data[column] for column in present)
        return sql, values


# Short aliases keep the per-column ``(name, kind)`` tables readable below.
_TEXT = ColumnKind.TEXT
_INT = ColumnKind.INTEGER
_REAL = ColumnKind.REAL
_BLOB = ColumnKind.BLOB


def _make_spec(
    table: CoreTable,
    columns: tuple[tuple[str, ColumnKind], ...],
    required: frozenset[str],
) -> TableSpec:
    """Build a :class:`TableSpec` from ``(column, kind)`` pairs.

    Args:
        table: The table the spec governs.
        columns: The allowed columns with their storage kinds, in schema order.
        required: The required column names (primary key plus non-defaulted
            ``NOT NULL`` columns).

    Returns:
        The immutable spec, with the column name tuple and type map derived from
        ``columns``.

    """
    return TableSpec(
        table=table,
        columns=tuple(name for name, _ in columns),
        required=required,
        column_types=dict(columns),
    )


# Columns mirror ``schema_core.sql`` in canonical order, each paired with its
# declared storage kind. The parity test in ``tests/test_snapshot_records.py``
# fails if a schema change drifts from these names, kinds, or required sets.
_TABLE_SPECS: dict[CoreTable, TableSpec] = {
    CoreTable.THOUGHT: _make_spec(
        CoreTable.THOUGHT,
        (
            ("thought_id", _TEXT),
            ("thought_type", _TEXT),
            ("essence", _TEXT),
            ("content", _TEXT),
            ("content_hash", _TEXT),
            ("priority", _TEXT),
            ("lifecycle_status", _TEXT),
            ("created_cycle", _INT),
            ("updated_cycle", _INT),
            ("source", _TEXT),
            ("confidence", _REAL),
            ("embedding_ref", _TEXT),
            ("source_type", _TEXT),
            ("confirmation_count", _INT),
            ("consolidated_from", _TEXT),
            ("visibility", _TEXT),
            ("access_count", _INT),
            ("last_accessed_at", _TEXT),
            ("created_at", _TEXT),
            ("updated_at", _TEXT),
            ("expires_at", _TEXT),
            ("metadata_json", _TEXT),
            ("valid_from", _TEXT),
            ("valid_until", _TEXT),
            ("action_outcome_score", _REAL),
            ("provenance", _TEXT),
            ("pinned", _INT),
            ("archived_at_cycle", _INT),
            ("archived_at", _TEXT),
        ),
        frozenset({"thought_id", "thought_type", "essence", "content", "priority"}),
    ),
    CoreTable.EDGE: _make_spec(
        CoreTable.EDGE,
        (
            ("edge_id", _TEXT),
            ("from_thought_id", _TEXT),
            ("to_thought_id", _TEXT),
            ("edge_type", _TEXT),
            ("weight", _REAL),
            ("created_cycle", _INT),
            ("source", _TEXT),
            ("decay_multiplier", _REAL),
            ("valid_from", _TEXT),
            ("valid_until", _TEXT),
            ("metadata_json", _TEXT),
        ),
        frozenset({"edge_id", "from_thought_id", "to_thought_id", "edge_type"}),
    ),
    CoreTable.EMBEDDING: _make_spec(
        CoreTable.EMBEDDING,
        (
            ("embedding_id", _TEXT),
            ("owner_type", _TEXT),
            ("owner_id", _TEXT),
            ("model_name", _TEXT),
            ("dimension", _INT),
            ("vector_blob", _BLOB),
            ("created_at", _TEXT),
        ),
        frozenset(
            {
                "embedding_id",
                "owner_type",
                "owner_id",
                "model_name",
                "dimension",
                "vector_blob",
                "created_at",
            }
        ),
    ),
    CoreTable.ACTION: _make_spec(
        CoreTable.ACTION,
        (
            ("action_id", _TEXT),
            ("source_thought_id", _TEXT),
            ("action_type", _TEXT),
            ("intent", _TEXT),
            ("status", _TEXT),
            ("verification_status", _TEXT),
            ("raw_metrics_json", _TEXT),
        ),
        frozenset({"action_id", "source_thought_id", "action_type", "intent"}),
    ),
}


def table_spec(table: CoreTable) -> TableSpec:
    """Return the immutable :class:`TableSpec` for a core table.

    Args:
        table: The core table to look up.

    Returns:
        The table's restore spec.

    """
    return _TABLE_SPECS[table]


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    """A snapshot metadata header line.

    Attributes:
        embedding_model_name: The embedding model the snapshot was produced
            with, or ``None`` if the snapshot did not record one.

    """

    embedding_model_name: str | None


@dataclass(frozen=True, slots=True)
class TableRecord:
    """A validated core-table data record ready to insert.

    Attributes:
        spec: The governing table spec.
        data: The validated column mapping. Keys are a subset of the spec's
            allowed columns, every required column is present, and every value is
            a JSON scalar (an ``embedding`` record's ``vector_blob`` is still the
            base64 transport string until :meth:`to_insert` decodes it).
        line_number: 1-based line number, retained for insert-time error context.

    """

    spec: TableSpec
    data: Mapping[str, SnapshotScalar]
    line_number: int

    def to_insert(self) -> tuple[str, tuple[SnapshotBindValue, ...]]:
        """Return fixed SQL and bind values, decoding any transport encoding.

        The base64 ``vector_blob`` of an ``embedding`` record is decoded to bytes
        here -- at insert time -- so a record that is skipped (``--skip-embeddings``
        or ``--re-embed``) never pays the decode and a corrupt blob does not fail
        an import that would not have stored it.

        Returns:
            A ``(sql, values)`` pair ready for ``execute``.

        Raises:
            InvalidEmbeddingBlobError: If an ``embedding`` record's ``vector_blob``
                is not valid base64.

        """
        values: Mapping[str, SnapshotBindValue] = self.data
        if self.spec.table is CoreTable.EMBEDDING:
            values = _decode_embedding_blob(self.data, line_number=self.line_number)
        return self.spec.build_insert(values)


@dataclass(frozen=True, slots=True)
class IgnoredRecord:
    """A well-formed record whose ``_type`` is not a restorable core table.

    Restore skips these lines (forward-compatible with extension record types).

    Attributes:
        record_type: The record's declared type.

    """

    record_type: str


#: A parsed snapshot line: metadata header, restorable record, or ignored line.
SnapshotRecord: TypeAlias = MetadataRecord | TableRecord | IgnoredRecord


def _decode_embedding_blob(
    data: Mapping[str, SnapshotScalar], *, line_number: int
) -> dict[str, SnapshotBindValue]:
    """Return a bind-ready copy of an embedding record with its blob decoded.

    ``vector_blob`` must be a base64 string. A non-string value (for example a
    number that would otherwise be bound straight into the ``BLOB`` column under
    SQLite's loose type affinity) is rejected here rather than silently stored.

    Args:
        data: The validated embedding column mapping.
        line_number: 1-based line number, for error context.

    Returns:
        A new mapping with ``vector_blob`` decoded from base64 to bytes.

    Raises:
        InvalidEmbeddingBlobError: If ``vector_blob`` is not a base64 string.

    """
    result: dict[str, SnapshotBindValue] = dict(data)
    blob = result.get(_VECTOR_BLOB_COLUMN)
    if not isinstance(blob, str):
        raise InvalidEmbeddingBlobError(line_number)
    try:
        result[_VECTOR_BLOB_COLUMN] = base64.b64decode(blob, validate=True)
    except ValueError as exc:  # binascii.Error is a ValueError subclass
        raise InvalidEmbeddingBlobError(line_number) from exc
    return result


def parse_snapshot_record(raw_line: str, *, line_number: int) -> SnapshotRecord:
    """Parse and validate one snapshot line into a typed record.

    Supports both the current ``{_type, data}`` form and the legacy
    ``{table, data}`` form. A core-table record is validated against its
    :class:`TableSpec` before it is returned, so a returned :class:`TableRecord`
    is always safe to insert via fixed SQL. A metadata header and a
    non-core-table record are returned as their own typed variants and are never
    inserted.

    Args:
        raw_line: A non-empty, stripped snapshot line.
        line_number: 1-based line number, for error context.

    Returns:
        A :data:`SnapshotRecord`: metadata header, validated table record, or an
        ignored non-core record.

    Raises:
        MalformedSnapshotRecordError: If the line is not a JSON object.
        MalformedMetadataError: If a metadata header has a non-string model name.
        MalformedRecordDataError: If a core record's ``data`` is not an object.
        UnknownSnapshotColumnError: If a core record has an unknown column.
        MissingRequiredColumnError: If a core record omits or nulls a required
            column.
        InvalidColumnValueError: If a core record has a non-scalar column value.

    """
    try:
        decoded: object = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise MalformedSnapshotRecordError(line_number) from exc
    if not isinstance(decoded, dict):
        raise MalformedSnapshotRecordError(line_number)
    record: dict[str, object] = decoded

    type_value = record.get(_TYPE_KEY) or record.get(_LEGACY_TYPE_KEY) or ""
    record_type = type_value if isinstance(type_value, str) else ""

    if record_type == METADATA_TYPE:
        model = record.get(_EMBEDDING_MODEL_KEY)
        if model is not None and not isinstance(model, str):
            raise MalformedMetadataError(line_number)
        return MetadataRecord(model)

    try:
        table = CoreTable(record_type)
    except ValueError:
        return IgnoredRecord(record_type)

    spec = _TABLE_SPECS[table]
    data = record.get(_DATA_KEY)
    if not isinstance(data, dict):
        raise MalformedRecordDataError(table, line_number)
    data_map: dict[str, object] = data
    spec.validate(data_map, line_number=line_number)
    # ``validate`` proved every value is a scalar, so this view is sound.
    scalar_data = cast("Mapping[str, SnapshotScalar]", data_map)
    return TableRecord(spec=spec, data=scalar_data, line_number=line_number)

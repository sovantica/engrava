"""Every accepted timestamp is stored in one canonical UTC form.

The store compares timestamp columns as TEXT: ``expires_at`` against
``datetime.now(UTC).isoformat()`` for expiry, and the valid-time columns against
MindQL literals. That comparison is only correct when every stored value has
the one shape engrava writes for its own timestamps (``T`` separator, extended
format, ``+00:00`` offset). This module drives each input shape
``datetime.fromisoformat`` accepts through the public write path and checks,
per shape:

* the stored value is that canonical form of the same instant;
* ``recall``, ``search_fts``, ``search_similar``, ``search_hybrid``,
  ``list_thoughts`` and ``count_thoughts`` include or exclude the row by its
  real instant, on both vector backends;
* ``cleanup_expired`` under ``ttl.strategy: delete`` takes exactly the rows
  whose instant has passed.

The clock is fixed so the "past" and "future" instants of each shape can sit on
the same UTC day as "now" -- the case where a space separator (0x20) sorts
before ``T`` (0x54) and a still-live row reads as already expired.
"""

from __future__ import annotations

import datetime
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from pydantic import ValidationError

from engrava import MindQLExecutor, parse
from engrava.domain.enums import EdgeType, LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import EdgeRecord, ThoughtRecord
from engrava.domain.models._temporal import (
    canonical_timestamp,
    canonical_timestamp_or_none,
    validate_iso8601_nullable,
)
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore
from engrava.infrastructure.sqlite.vector_sqlite_vec import SqliteVecSearchBackend

if TYPE_CHECKING:
    from pathlib import Path

_REAL_DATETIME = datetime.datetime

#: The fixed "now" every store read in this module sees. It carries
#: microseconds, as a real ``datetime.now(UTC).isoformat()`` almost always does.
_NOW = _REAL_DATETIME(2026, 9, 25, 12, 0, 0, 500000, tzinfo=datetime.UTC)

_DIMENSION = 3
_VECTOR = [1.0, 0.0, 0.0]
_MODEL = "test-fixture-model"
_TERM = "zebracorn"


class _FixedClock(_REAL_DATETIME):
    """A ``datetime`` whose wall-clock readers return :data:`_NOW`."""

    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> _FixedClock:
        instant = _NOW if tz is None else _NOW.astimezone(tz)
        return cls(
            instant.year,
            instant.month,
            instant.day,
            instant.hour,
            instant.minute,
            instant.second,
            instant.microsecond,
            tzinfo=None if tz is None else instant.tzinfo,
        )

    @classmethod
    def utcnow(cls) -> _FixedClock:
        return cls.now(datetime.UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class _Shape:
    """One accepted input shape, as a past and a future instant around ``_NOW``.

    The expected canonical strings are written out literally rather than
    computed, so the oracle does not share code with the implementation.
    """

    label: str
    past: str
    past_canonical: str
    future: str
    future_canonical: str


_SHAPES: tuple[_Shape, ...] = (
    _Shape(
        "naive extended",
        "2026-09-25T09:30:00",
        "2026-09-25T09:30:00+00:00",
        "2026-09-25T14:30:00",
        "2026-09-25T14:30:00+00:00",
    ),
    _Shape(
        "naive extended, microseconds",
        "2026-09-25T09:30:00.250000",
        "2026-09-25T09:30:00.250000+00:00",
        "2026-09-25T14:30:00.250000",
        "2026-09-25T14:30:00.250000+00:00",
    ),
    _Shape(
        "naive space separator",
        "2026-09-25 09:30:00",
        "2026-09-25T09:30:00+00:00",
        "2026-09-25 14:30:00",
        "2026-09-25T14:30:00+00:00",
    ),
    _Shape(
        "naive space separator, short fraction",
        "2026-09-25 09:30:00.25",
        "2026-09-25T09:30:00.250000+00:00",
        "2026-09-25 14:30:00.25",
        "2026-09-25T14:30:00.250000+00:00",
    ),
    _Shape(
        "naive basic format",
        "20260925T093000",
        "2026-09-25T09:30:00+00:00",
        "20260925T143000",
        "2026-09-25T14:30:00+00:00",
    ),
    _Shape(
        "naive basic format, microseconds",
        "20260925T093000.250000",
        "2026-09-25T09:30:00.250000+00:00",
        "20260925T143000.250000",
        "2026-09-25T14:30:00.250000+00:00",
    ),
    _Shape(
        "week date with time",
        "2026-W39-5T09:30:00",
        "2026-09-25T09:30:00+00:00",
        "2026-W39-5T14:30:00",
        "2026-09-25T14:30:00+00:00",
    ),
    _Shape(
        "week date",
        "2026-W39-1",
        "2026-09-21T00:00:00+00:00",
        "2026-W40-1",
        "2026-09-28T00:00:00+00:00",
    ),
    _Shape(
        "date only",
        "2026-09-24",
        "2026-09-24T00:00:00+00:00",
        "2026-09-26",
        "2026-09-26T00:00:00+00:00",
    ),
    _Shape(
        "aware non-UTC offset",
        "2026-09-25T15:00:00+05:30",
        "2026-09-25T09:30:00+00:00",
        "2026-09-25T20:00:00+05:30",
        "2026-09-25T14:30:00+00:00",
    ),
    _Shape(
        "aware non-UTC offset, microseconds",
        "2026-09-25T04:30:00.250000-05:00",
        "2026-09-25T09:30:00.250000+00:00",
        "2026-09-25T09:30:00.250000-05:00",
        "2026-09-25T14:30:00.250000+00:00",
    ),
    _Shape(
        "aware UTC, Z suffix",
        "2026-09-25T09:30:00Z",
        "2026-09-25T09:30:00+00:00",
        "2026-09-25T14:30:00Z",
        "2026-09-25T14:30:00+00:00",
    ),
    _Shape(
        "aware UTC, canonical",
        "2026-09-25T09:30:00+00:00",
        "2026-09-25T09:30:00+00:00",
        "2026-09-25T14:30:00.250000+00:00",
        "2026-09-25T14:30:00.250000+00:00",
    ),
    # The same second as "now": a canonical value without microseconds must
    # still order correctly against a "now" that has them ('+' sorts before '.').
    _Shape(
        "same second as now, without microseconds",
        "2026-09-25T12:00:00",
        "2026-09-25T12:00:00+00:00",
        "2026-09-25T12:00:01",
        "2026-09-25T12:00:01+00:00",
    ),
    _Shape(
        "same second as now, with microseconds",
        "2026-09-25T12:00:00.250000",
        "2026-09-25T12:00:00.250000+00:00",
        "2026-09-25T12:00:00.750000",
        "2026-09-25T12:00:00.750000+00:00",
    ),
)

_BACKENDS = (
    pytest.param("numpy", id="numpy"),
    pytest.param(
        "sqlite-vec",
        id="sqlite-vec",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("sqlite_vec") is None,
            reason="sqlite-vec package not installed",
        ),
    ),
)

#: Every ThoughtRecord / EdgeRecord field the shared timestamp validator covers.
_THOUGHT_TIMESTAMP_FIELDS = (
    "created_at",
    "updated_at",
    "last_accessed_at",
    "expires_at",
    "valid_from",
    "valid_until",
    "archived_at",
)
_EDGE_TIMESTAMP_FIELDS = ("valid_from", "valid_until")


def _thought(thought_id: str, **fields: object) -> ThoughtRecord:
    base: dict[str, object] = {
        "thought_id": thought_id,
        "thought_type": ThoughtType.OBSERVATION,
        "essence": f"{_TERM} {thought_id}",
        "content": f"{_TERM} marker content for {thought_id}",
        "priority": Priority.P2,
        "lifecycle_status": LifecycleStatus.ACTIVE,
        "created_cycle": 0,
        "updated_cycle": 0,
        "source": "test",
    }
    base.update(fields)
    return ThoughtRecord.model_validate(base)


async def _build_store(tmp_path: Path, backend: str) -> SqliteEngravaCore:
    db = await aiosqlite.connect(str(tmp_path / f"{backend}.db"))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(db, ttl_strategy="delete")
    store._owns_connection = True
    await store.ensure_schema()
    await store._configure_vector_backend(backend_name=backend, embedding_dimension=_DIMENSION)
    if backend == "sqlite-vec":
        # A backend that silently degraded to numpy would make this arm vacuous.
        assert isinstance(store._vector_backend, SqliteVecSearchBackend)
    return store


async def _raw_column(store: SqliteEngravaCore, table: str, column: str, row_id: str) -> object:
    key = "thought_id" if table == "thought" else "edge_id"
    cursor = await store._db.execute(
        f"SELECT {column} FROM {table} WHERE {key} = ?",  # noqa: S608 - fixed test identifiers
        (row_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    return row[0]


@pytest.fixture
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every ``datetime.now`` read to :data:`_NOW` for the test's duration."""
    monkeypatch.setattr(datetime, "datetime", _FixedClock)
    assert datetime.datetime.now(datetime.UTC).isoformat() == "2026-09-25T12:00:00.500000+00:00"


@pytest.mark.usefixtures("fixed_clock")
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=[shape.label for shape in _SHAPES])
async def test_expiry_follows_the_real_instant_for_every_input_shape(
    tmp_path: Path, backend: str, shape: _Shape
) -> None:
    """Store, retrieval and cleanup all agree with the instant, whatever its shape."""
    store = await _build_store(tmp_path, backend)
    try:
        for thought_id, expires_at in (("past", shape.past), ("future", shape.future)):
            await store.create_thought(_thought(thought_id, expires_at=expires_at))
            await store.store_embedding(thought_id, _VECTOR, model_name=_MODEL)

        hybrid = await store.search_hybrid(_TERM, query_vector=_VECTOR, top_k=10)
        recalled = await store.recall(_TERM, top_k=10)
        observed_before_cleanup = {
            "stored past": await _raw_column(store, "thought", "expires_at", "past"),
            "stored future": await _raw_column(store, "thought", "expires_at", "future"),
            "recall": sorted(tid for tid, _ in recalled.results),
            "search_fts": sorted(tid for tid, _ in await store.search_fts(_TERM, top_k=10)),
            "search_similar": sorted(
                tid for tid, _ in await store.search_similar(_VECTOR, top_k=10)
            ),
            "search_hybrid": sorted(tid for tid, _ in hybrid.results),
            "list_thoughts": sorted(t.thought_id for t in await store.list_thoughts()),
            "count_thoughts": await store.count_thoughts(),
            "list_thoughts(include_expired)": sorted(
                t.thought_id for t in await store.list_thoughts(include_expired=True)
            ),
        }
        cleanup = await store.cleanup_expired()
        cursor = await store._db.execute("SELECT thought_id FROM thought ORDER BY thought_id")
        remaining = [row[0] for row in await cursor.fetchall()]

        # State first: what is stored and what every read surface returns.
        assert observed_before_cleanup == {
            "stored past": shape.past_canonical,
            "stored future": shape.future_canonical,
            "recall": ["future"],
            "search_fts": ["future"],
            "search_similar": ["future"],
            "search_hybrid": ["future"],
            "list_thoughts": ["future"],
            "count_thoughts": 1,
            "list_thoughts(include_expired)": ["future", "past"],
        }
        # ttl.strategy: delete -- exactly the row whose instant has passed goes.
        assert remaining == ["future"]
        assert cleanup.expired_count == 1
    finally:
        await store.close()


@pytest.mark.parametrize("shape", _SHAPES, ids=[shape.label for shape in _SHAPES])
def test_validator_returns_the_canonical_utc_form(shape: _Shape) -> None:
    assert validate_iso8601_nullable(shape.past) == shape.past_canonical
    assert validate_iso8601_nullable(shape.future) == shape.future_canonical


@pytest.mark.parametrize("shape", _SHAPES, ids=[shape.label for shape in _SHAPES])
def test_canonical_form_is_a_fixed_point(shape: _Shape) -> None:
    """Re-validating a stored value never changes it again."""
    assert validate_iso8601_nullable(shape.past_canonical) == shape.past_canonical
    assert validate_iso8601_nullable(shape.future_canonical) == shape.future_canonical


def test_canonical_form_matches_what_engrava_writes_for_its_own_timestamps() -> None:
    """The store stamps ``datetime.now(UTC).isoformat()``; a value of the same
    instant passed in any shape must compare equal to that string."""
    own = _REAL_DATETIME(2026, 9, 25, 12, 0, 0, 500000, tzinfo=datetime.UTC).isoformat()
    whole_second = _REAL_DATETIME(2026, 9, 25, 12, 0, 0, tzinfo=datetime.UTC).isoformat()
    assert validate_iso8601_nullable("2026-09-25 12:00:00.5") == own
    assert validate_iso8601_nullable("2026-09-25T14:00:00.500000+02:00") == own
    assert validate_iso8601_nullable("20260925T120000") == whole_second


@pytest.mark.parametrize(
    "value",
    ["not-a-timestamp", "2026-13-01T00:00:00", "2026-09-25T25:00:00", "", "2026/09/25"],
)
def test_unparseable_input_is_still_rejected_with_the_same_message(value: str) -> None:
    with pytest.raises(ValueError, match=r"^Must be ISO-8601 timestamp, got ") as excinfo:
        validate_iso8601_nullable(value)
    assert str(excinfo.value) == f"Must be ISO-8601 timestamp, got {value!r}"


@pytest.mark.parametrize("field", _THOUGHT_TIMESTAMP_FIELDS)
def test_every_thought_timestamp_field_is_normalised(field: str) -> None:
    record = _thought("t-1", **{field: "2026-09-25 09:30:00"})
    assert getattr(record, field) == "2026-09-25T09:30:00+00:00"


@pytest.mark.parametrize("field", _EDGE_TIMESTAMP_FIELDS)
def test_every_edge_timestamp_field_is_normalised(field: str) -> None:
    edge = EdgeRecord.model_validate(
        {
            "edge_id": "e-1",
            "from_thought_id": "a",
            "to_thought_id": "b",
            "edge_type": EdgeType.ASSOCIATED,
            "weight": 0.5,
            "created_cycle": 0,
            field: "20260925T093000",
        }
    )
    assert getattr(edge, field) == "2026-09-25T09:30:00+00:00"


async def test_every_timestamp_column_is_stored_canonical(tmp_path: Path) -> None:
    """The write path persists the canonical form in every validated column."""
    store = await _build_store(tmp_path, "numpy")
    try:
        naive = {
            "created_at": "2026-09-20 08:00:00",
            "updated_at": "20260921T080000",
            "last_accessed_at": "2026-W39-2T08:00:00",
            "expires_at": "2027-01-01 00:00:00.5",
            "valid_from": "2026-01-01T00:00:00",
            "valid_until": "2026-07-01T02:00:00+02:00",
            "archived_at": "2026-09-23",
        }
        await store.create_thought(_thought("a", **naive))
        await store.create_thought(_thought("b"))
        await store.create_edge(
            EdgeRecord(
                edge_id="e-1",
                from_thought_id="a",
                to_thought_id="b",
                edge_type=EdgeType.ASSOCIATED,
                weight=0.5,
                created_cycle=0,
                valid_from="2026-01-01 00:00:00",
                valid_until="20260701T000000",
            )
        )
        stored_thought = {field: await _raw_column(store, "thought", field, "a") for field in naive}
        stored_edge = {
            field: await _raw_column(store, "edge", field, "e-1")
            for field in _EDGE_TIMESTAMP_FIELDS
        }
        assert stored_thought == {
            "created_at": "2026-09-20T08:00:00+00:00",
            "updated_at": "2026-09-21T08:00:00+00:00",
            "last_accessed_at": "2026-09-22T08:00:00+00:00",
            "expires_at": "2027-01-01T00:00:00.500000+00:00",
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_until": "2026-07-01T00:00:00+00:00",
            "archived_at": "2026-09-23T00:00:00+00:00",
        }
        assert stored_edge == {
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_until": "2026-07-01T00:00:00+00:00",
        }
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        # Exact lower bound, offset-less: valid_from is inclusive.
        ("valid_at '2026-01-01T00:00:00'", ["fact"]),
        ("valid_at '2026-01-01 00:00:00'", ["fact"]),
        # Exact upper bound, offset-less: valid_until is exclusive.
        ("valid_at '2026-07-01T00:00:00'", []),
        ("valid_at '20260701T000000'", []),
        # A window whose lower bound sits on the fact's exclusive upper bound.
        ("valid_within '2026-07-01T00:00:00' '2026-12-01T00:00:00'", []),
        ("valid_within '2026-06-01 00:00:00' '2026-12-01 00:00:00'", ["fact"]),
        # Fully contained, with the containing range's bounds on the fact's own.
        ("valid_between '2026-01-01T00:00:00' '2026-07-01T00:00:00'", ["fact"]),
        ("valid_between '2026-01-01 00:00:01' '2026-07-01 00:00:00'", []),
    ],
)
async def test_an_offset_less_mindql_literal_behaves_as_its_utc_instant(
    predicate: str, expected: list[str]
) -> None:
    """MindQL literals go through the same validator: naive means UTC."""
    conn = await aiosqlite.connect(":memory:")
    try:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await store.create_thought(
            _thought(
                "fact",
                valid_from="2026-01-01T00:00:00+00:00",
                valid_until="2026-07-01T00:00:00+00:00",
            )
        )
        result = await MindQLExecutor(conn).execute(parse(f"FIND thoughts WHERE {predicate}"))
        assert sorted(str(row["thought_id"]) for row in result.rows) == expected
    finally:
        await conn.close()


async def test_update_accepts_created_at_restated_in_the_form_first_passed(
    tmp_path: Path,
) -> None:
    """A caller re-sending its own naive ``created_at`` is not changing it."""
    store = await _build_store(tmp_path, "numpy")
    try:
        await store.create_thought(_thought("a", created_at="2026-01-02 03:04:05"))
        updated = await store.update_thought("a", created_at="2026-01-02 03:04:05", essence="new")
        assert await _raw_column(store, "thought", "created_at", "a") == (
            "2026-01-02T03:04:05+00:00"
        )
        assert updated.essence == "new"
    finally:
        await store.close()


#: Valid ISO-8601, but converting either to UTC leaves the ``datetime`` range.
_NO_UTC_FORM = ("0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00")


@pytest.mark.parametrize("value", _NO_UTC_FORM)
def test_a_timestamp_with_no_utc_form_is_a_value_error(value: str) -> None:
    with pytest.raises(ValueError, match=r"has no UTC form within the supported datetime range"):
        canonical_timestamp(value)
    # A stored value of this kind is left as it is, not rejected.
    assert canonical_timestamp_or_none(value) is None


@pytest.mark.parametrize("value", _NO_UTC_FORM)
def test_a_record_with_no_utc_form_timestamp_is_a_validation_error(value: str) -> None:
    with pytest.raises(ValidationError, match=r"has no UTC form"):
        _thought("t-1", expires_at=value)
    with pytest.raises(ValidationError, match=r"has no UTC form"):
        EdgeRecord.model_validate(
            {
                "edge_id": "e-1",
                "from_thought_id": "a",
                "to_thought_id": "b",
                "edge_type": EdgeType.ASSOCIATED,
                "weight": 0.5,
                "created_cycle": 0,
                "valid_from": value,
            }
        )

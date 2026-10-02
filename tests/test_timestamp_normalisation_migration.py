"""The v20 -> v21 upgrade rewrites stored timestamps into the canonical UTC form.

Before 0.7.0 the shared timestamp validator stored a naive value exactly as the
caller wrote it, so a v20 database can hold ``2026-01-02 03:04:05``,
``20260102T030405`` or ``2026-W01-5`` in a column the store compares as TEXT.
The upgrade step rewrites every such value, once, into the form the validator
now produces (``T`` separator, extended format, ``+00:00`` offset) and leaves
the instant unchanged.

What these tests pin, on a real-shape v20 database with values planted directly
in the file:

* every column the validator covers ends up canonical -- all seven on
  ``thought`` and both on ``edge`` -- so a migration that skips any one column
  fails here;
* canonical values and NULLs are left byte-identical; a value without the
  canonical shape that cannot be parsed is left untouched and only counted in
  a log line that never carries the value; a value that already has the
  canonical shape is not read back at all, so even an impossible date in that
  shape is neither rewritten nor counted;
* no row is added or lost, ``revision`` stays at its default, and the migrated
  schema still equals a freshly bootstrapped v21 one;
* the rewrite is part of the step's transaction: a failure at the version
  stamp leaves every planted value exactly as it was.
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import random
import re
import sqlite3
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from pydantic import ValidationError

from engrava import SqliteEngravaCore
from engrava.domain.exceptions import CoreMigrationError
from engrava.domain.models import EdgeRecord, ThoughtRecord
from engrava.infrastructure.sqlite import engrava_core as _core
from tests.test_migration_upgrade_chains import (
    _bootstrap_core_at_version,
    _capture_schema_shape,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_HEAD_VERSION = 21

_THOUGHT_COLUMNS = (
    "created_at",
    "updated_at",
    "last_accessed_at",
    "expires_at",
    "valid_from",
    "valid_until",
    "archived_at",
)
_EDGE_COLUMNS = ("valid_from", "valid_until")
_MIGRATED_COLUMNS: tuple[tuple[str, str], ...] = (
    *(("thought", column) for column in _THOUGHT_COLUMNS),
    *(("edge", column) for column in _EDGE_COLUMNS),
)

_CANONICAL_SHAPE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{6})?\+00:00"
)


def _is_canonical(value: object) -> bool:
    """Return whether ``value`` is exactly what ``datetime.isoformat()`` writes in UTC.

    Independent of the implementation: the fixed shape, and the value being its
    own ``isoformat()`` (so ``.000000`` and impossible dates do not pass).
    """
    if not isinstance(value, str) or _CANONICAL_SHAPE.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value


# Every value below is planted with raw SQL. The expectations are literals so
# the oracle shares no code with the implementation.
_THOUGHT_ROWS: dict[str, dict[str, object]] = {
    # Every migrated column non-canonical, each in a different accepted shape.
    "t-mixed": {
        "created_at": "2026-01-02 03:04:05",
        "updated_at": "20260102T030405.5",
        "last_accessed_at": "2026-W01-5T03:04:05",
        "expires_at": "2099-12-31T23:00:00-02:00",
        "valid_from": "2026-01-01",
        "valid_until": "2026-07-01T00:00:00Z",
        "archived_at": "2026-09-25T12:00:00.000000+00:00",
    },
    # Already canonical: must come out byte-identical.
    "t-canonical": {
        "created_at": "2026-01-02T03:04:05+00:00",
        "updated_at": "2026-01-02T03:04:05.500000+00:00",
        "last_accessed_at": "2026-01-02T03:04:05+00:00",
        "expires_at": "2100-01-01T01:00:00+00:00",
        "valid_from": "2026-01-01T00:00:00+00:00",
        "valid_until": "2026-07-01T00:00:00+00:00",
        "archived_at": "2026-09-25T12:00:00+00:00",
    },
    # NULL stays NULL.
    "t-null": dict.fromkeys(_THOUGHT_COLUMNS),
    # Cannot be normalised: left untouched, counted.
    "t-bad": {
        "created_at": "yesterday",
        "updated_at": "0001-01-01T00:00:00+01:00",  # parses, but has no UTC instant
        "last_accessed_at": b"\x00\xff",
        "expires_at": "2026-13-45 00:00:00",
        "valid_from": None,
        "valid_until": None,
        "archived_at": "2026-09-25T25:00:00",
    },
}

_EDGE_ROWS: dict[str, dict[str, object]] = {
    "e-mixed": {"valid_from": "2026-01-01 00:00:00", "valid_until": "20260701T000000"},
    "e-canonical": {
        "valid_from": "2026-01-01T00:00:00+00:00",
        "valid_until": "2026-07-01T00:00:00.250000+00:00",
    },
    "e-bad": {"valid_from": "garbage", "valid_until": None},
}

_EXPECTED_THOUGHTS: dict[str, dict[str, object]] = {
    "t-mixed": {
        "created_at": "2026-01-02T03:04:05+00:00",
        "updated_at": "2026-01-02T03:04:05.500000+00:00",
        "last_accessed_at": "2026-01-02T03:04:05+00:00",
        "expires_at": "2100-01-01T01:00:00+00:00",
        "valid_from": "2026-01-01T00:00:00+00:00",
        "valid_until": "2026-07-01T00:00:00+00:00",
        "archived_at": "2026-09-25T12:00:00+00:00",
    },
    "t-canonical": dict(_THOUGHT_ROWS["t-canonical"]),
    "t-null": dict.fromkeys(_THOUGHT_COLUMNS),
    "t-bad": dict(_THOUGHT_ROWS["t-bad"]),
}

_EXPECTED_EDGES: dict[str, dict[str, object]] = {
    "e-mixed": {
        "valid_from": "2026-01-01T00:00:00+00:00",
        "valid_until": "2026-07-01T00:00:00+00:00",
    },
    "e-canonical": dict(_EDGE_ROWS["e-canonical"]),
    "e-bad": dict(_EDGE_ROWS["e-bad"]),
}

#: The planted values that cannot be normalised (non-NULL entries of the bad rows).
_UNPARSEABLE = [
    value
    for row in (_THOUGHT_ROWS["t-bad"], _EDGE_ROWS["e-bad"])
    for value in row.values()
    if value is not None
]


@pytest.fixture
async def v20_db() -> AsyncIterator[aiosqlite.Connection]:
    """A real-shape v20 database (the schema 0.6.0 shipped), still empty."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await _bootstrap_core_at_version(conn, 20)
    yield conn
    await conn.close()


async def _plant_thought(
    db: aiosqlite.Connection, thought_id: str, values: dict[str, object]
) -> None:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    await db.execute(
        "INSERT INTO thought (thought_id, thought_type, essence, content, priority, "  # noqa: S608 - fixed test identifiers
        f"lifecycle_status, {columns}) VALUES (?, 'OBSERVATION', ?, ?, 'P2', 'ACTIVE', "
        f"{placeholders})",
        (thought_id, f"essence {thought_id}", f"content {thought_id}", *values.values()),
    )


#: Distinct endpoints per planted edge: ``edge`` is unique on (from, to, type).
_EDGE_ENDPOINTS = {
    "e-mixed": ("t-canonical", "t-mixed"),
    "e-canonical": ("t-mixed", "t-canonical"),
    "e-bad": ("t-null", "t-bad"),
    "e-1": ("t-canonical", "t-mixed"),
}


async def _plant_edge(db: aiosqlite.Connection, edge_id: str, values: dict[str, object]) -> None:
    from_id, to_id = _EDGE_ENDPOINTS[edge_id]
    await db.execute(
        "INSERT INTO edge (edge_id, from_thought_id, to_thought_id, edge_type, "
        "valid_from, valid_until) VALUES (?, ?, ?, 'ASSOCIATED', ?, ?)",
        (edge_id, from_id, to_id, values["valid_from"], values["valid_until"]),
    )


async def _plant_everything(db: aiosqlite.Connection) -> None:
    for thought_id, values in _THOUGHT_ROWS.items():
        await _plant_thought(db, thought_id, values)
    for edge_id, values in _EDGE_ROWS.items():
        await _plant_edge(db, edge_id, values)
    await db.commit()


async def _read_rows(
    db: aiosqlite.Connection, table: str, key: str, columns: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    cursor = await db.execute(
        f"SELECT {key}, {', '.join(columns)} FROM {table} ORDER BY {key}"  # noqa: S608 - fixed test identifiers
    )
    return {
        str(row[0]): dict(zip(columns, row[1:], strict=True)) for row in await cursor.fetchall()
    }


async def _row_counts(db: aiosqlite.Connection) -> dict[str, int]:
    counts = {}
    for table in ("thought", "edge", "embedding", "action", "journal_entry"):
        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 - fixed names
        row = await cursor.fetchone()
        assert row is not None
        counts[table] = int(row[0])
    return counts


async def _user_version(db: aiosqlite.Connection) -> int:
    cursor = await db.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _non_canonical_values(db: aiosqlite.Connection) -> list[tuple[str, str, object]]:
    found: list[tuple[str, str, object]] = []
    for table, column in _MIGRATED_COLUMNS:
        cursor = await db.execute(
            f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"  # noqa: S608 - fixed names
        )
        found.extend(
            (table, column, row[0]) for row in await cursor.fetchall() if not _is_canonical(row[0])
        )
    return found


def test_the_oracle_discriminates() -> None:
    """The canonical check itself rejects every planted non-canonical shape."""
    for value in _THOUGHT_ROWS["t-mixed"].values():
        assert not _is_canonical(value), value
    for value in _THOUGHT_ROWS["t-canonical"].values():
        assert _is_canonical(value), value
    for value in _UNPARSEABLE:
        assert not _is_canonical(value), value


async def test_every_migrated_column_is_canonical_after_the_upgrade(
    v20_db: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    await _plant_everything(v20_db)
    # Non-vacuous: before the upgrade, every migrated column holds at least one
    # parseable, non-canonical value.
    before = await _non_canonical_values(v20_db)
    assert {(table, column) for table, column, value in before if value not in _UNPARSEABLE} == (
        set(_MIGRATED_COLUMNS)
    )
    counts_before = await _row_counts(v20_db)

    with caplog.at_level(logging.WARNING, logger="engrava"):
        await SqliteEngravaCore(v20_db).ensure_schema()

    assert await _user_version(v20_db) == _HEAD_VERSION
    assert await _row_counts(v20_db) == counts_before
    assert await _read_rows(v20_db, "thought", "thought_id", _THOUGHT_COLUMNS) == (
        _EXPECTED_THOUGHTS
    )
    assert await _read_rows(v20_db, "edge", "edge_id", _EDGE_COLUMNS) == _EXPECTED_EDGES
    # Only the values that cannot be normalised are left non-canonical.
    assert sorted(repr(value) for _, _, value in await _non_canonical_values(v20_db)) == sorted(
        repr(value) for value in _UNPARSEABLE
    )
    # The rewrite does not count as an edit: revision stays at its default.
    cursor = await v20_db.execute("SELECT DISTINCT revision FROM thought")
    assert [row[0] for row in await cursor.fetchall()] == [0]

    # The unparseable values are counted in one log line, never quoted.
    counted = [
        record
        for record in caplog.records
        if re.search(rf"\b{len(_UNPARSEABLE)} stored timestamp value", record.getMessage())
    ]
    assert len(counted) == 1
    for value in _UNPARSEABLE:
        text = value.decode("latin-1") if isinstance(value, bytes) else value
        assert text not in caplog.text


@pytest.mark.parametrize(("table", "column"), _MIGRATED_COLUMNS)
async def test_each_migrated_column_is_normalised_on_its_own(
    v20_db: aiosqlite.Connection, table: str, column: str
) -> None:
    """A non-canonical value in exactly one column is enough to be rewritten."""
    await _plant_thought(v20_db, "t-canonical", dict(_THOUGHT_ROWS["t-canonical"]))
    await _plant_thought(v20_db, "t-mixed", dict(_THOUGHT_ROWS["t-canonical"]))
    planted = "2026-01-02 03:04:05"
    if table == "thought":
        key, row_id = "thought_id", "t-mixed"
    else:
        await _plant_edge(v20_db, "e-1", {"valid_from": None, "valid_until": None})
        key, row_id = "edge_id", "e-1"
    update = f"UPDATE {table} SET {column} = ? WHERE {key} = ?"  # noqa: S608 - fixed names
    await v20_db.execute(update, (planted, row_id))
    await v20_db.commit()

    await SqliteEngravaCore(v20_db).ensure_schema()

    cursor = await v20_db.execute(
        f"SELECT {column} FROM {table} WHERE {key} = ?",  # noqa: S608 - fixed names
        (row_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "2026-01-02T03:04:05+00:00"
    assert await _non_canonical_values(v20_db) == []


async def test_the_migrated_schema_still_equals_a_fresh_v21_schema(
    v20_db: aiosqlite.Connection,
) -> None:
    await _plant_everything(v20_db)
    await SqliteEngravaCore(v20_db).ensure_schema()

    fresh = await aiosqlite.connect(":memory:")
    try:
        fresh.row_factory = aiosqlite.Row
        await SqliteEngravaCore(fresh).ensure_schema()
        assert await _capture_schema_shape(v20_db) == await _capture_schema_shape(fresh)
    finally:
        await fresh.close()


async def test_a_thought_only_database_is_normalised_without_an_edge_table(
    v20_db: aiosqlite.Connection,
) -> None:
    """A partial bootstrap with no ``edge`` table still migrates its thoughts."""
    await v20_db.execute("DROP TABLE edge")
    await _plant_thought(v20_db, "t-mixed", dict(_THOUGHT_ROWS["t-mixed"]))
    await v20_db.commit()

    await SqliteEngravaCore(v20_db).ensure_schema()

    assert await _user_version(v20_db) == _HEAD_VERSION
    assert await _read_rows(v20_db, "thought", "thought_id", _THOUGHT_COLUMNS) == {
        "t-mixed": _EXPECTED_THOUGHTS["t-mixed"]
    }


async def test_the_rewrite_rolls_back_with_its_step(
    v20_db: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure at the v21 stamp leaves every planted value as it was."""
    await _plant_everything(v20_db)
    before = await _read_rows(v20_db, "thought", "thought_id", _THOUGHT_COLUMNS)
    edges_before = await _read_rows(v20_db, "edge", "edge_id", _EDGE_COLUMNS)

    real_execute = v20_db.execute

    async def _fail_the_v21_stamp(
        sql: str, parameters: tuple[object, ...] | None = None
    ) -> aiosqlite.Cursor:
        if sql.strip() == f"PRAGMA user_version = {_HEAD_VERSION}":
            message = "injected failure at the v21 stamp"
            raise sqlite3.OperationalError(message)
        return await real_execute(sql, parameters)

    monkeypatch.setattr(v20_db, "execute", _fail_the_v21_stamp)
    store = SqliteEngravaCore(v20_db)
    with pytest.raises(sqlite3.OperationalError, match="injected failure at the v21 stamp"):
        await store.ensure_schema()
    monkeypatch.undo()

    assert await _user_version(v20_db) == _HEAD_VERSION - 1
    assert await _read_rows(v20_db, "thought", "thought_id", _THOUGHT_COLUMNS) == before
    assert await _read_rows(v20_db, "edge", "edge_id", _EDGE_COLUMNS) == edges_before

    # The retry applies the whole step, rewrite included.
    await store.ensure_schema()
    assert await _user_version(v20_db) == _HEAD_VERSION
    assert await _read_rows(v20_db, "thought", "thought_id", _THOUGHT_COLUMNS) == (
        _EXPECTED_THOUGHTS
    )


# ---------------------------------------------------------------------------
# The step's own mechanics: its column list, its GLOB, its paging, its check
# ---------------------------------------------------------------------------


def _validated_timestamp_fields(
    model: type[ThoughtRecord | EdgeRecord], base: dict[str, object]
) -> set[str]:
    """Discover, by behaviour, which fields the model rewrites into canonical form."""
    found: set[str] = set()
    for name in model.model_fields:
        try:
            record = model.model_validate({**base, name: "2026-01-02 03:04:05"})
        except ValidationError:
            continue
        if getattr(record, name) == "2026-01-02T03:04:05+00:00":
            found.add(name)
    return found


def test_the_step_covers_exactly_the_columns_the_validator_covers() -> None:
    thought_base: dict[str, object] = {
        "thought_id": "t",
        "thought_type": "OBSERVATION",
        "essence": "e",
        "content": "c",
        "priority": "P2",
        "lifecycle_status": "ACTIVE",
        "created_cycle": 0,
        "updated_cycle": 0,
        "source": "s",
    }
    edge_base: dict[str, object] = {
        "edge_id": "e",
        "from_thought_id": "a",
        "to_thought_id": "b",
        "edge_type": "ASSOCIATED",
        "weight": 0.5,
        "created_cycle": 0,
    }
    assert dict(_core._CANONICAL_TIMESTAMP_COLUMNS) == {
        "thought": tuple(_THOUGHT_COLUMNS),
        "edge": tuple(_EDGE_COLUMNS),
    }
    assert set(_THOUGHT_COLUMNS) == _validated_timestamp_fields(ThoughtRecord, thought_base)
    assert set(_EDGE_COLUMNS) == _validated_timestamp_fields(EdgeRecord, edge_base)


def _instants() -> list[datetime.datetime]:
    """A fixed, seeded sample of UTC instants, range limits included."""
    rng = random.Random(20260925)  # noqa: S311 - a reproducible sample, not a secret
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
    instants = [
        datetime.datetime(1, 1, 1, tzinfo=datetime.UTC),
        datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=datetime.UTC),
        datetime.datetime(2026, 9, 25, 12, 0, 0, tzinfo=datetime.UTC),
        datetime.datetime(2026, 9, 25, 12, 0, 0, 1, tzinfo=datetime.UTC),
    ]
    for _ in range(150):
        microseconds = rng.choice((0, rng.randrange(1, 1_000_000)))
        offset = datetime.timedelta(
            seconds=rng.randrange(-(10**10), 10**10), microseconds=microseconds
        )
        instants.append(epoch + offset)
    return instants


def _renderings(instant: datetime.datetime) -> list[str]:
    """Non-canonical spellings of ``instant`` that ``fromisoformat`` accepts."""
    naive = instant.replace(tzinfo=None)
    basic = (
        f"{naive.year:04d}{naive.month:02d}{naive.day:02d}"
        f"T{naive.hour:02d}{naive.minute:02d}{naive.second:02d}"
    )
    spellings = [
        naive.isoformat(),
        str(naive),
        naive.isoformat() + "Z",
        basic + (f".{naive.microsecond:06d}" if naive.microsecond else ""),
        naive.isoformat().replace("T", "t"),
    ]
    if not naive.microsecond:
        spellings.append(naive.isoformat(timespec="microseconds") + "+00:00")
    # The +05:30 local time of a range-limit instant does not exist.
    with contextlib.suppress(OverflowError):
        india = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        spellings.append(instant.astimezone(india).isoformat())
    return spellings


async def test_every_spelling_of_a_sampled_instant_is_rewritten_and_nothing_else(
    v20_db: aiosqlite.Connection,
) -> None:
    """Oracle sweep: canonical in, untouched; any other spelling in, canonical out.

    ``total_changes`` counts the rows the step actually rewrote, so a GLOB
    that failed to recognise a canonical value (and rewrote the whole store)
    is caught as surely as one that let a non-canonical value through.
    """
    expected: dict[str, str] = {}
    rewrites_needed = 0
    for index, instant in enumerate(_instants()):
        canonical = instant.isoformat()
        assert _is_canonical(canonical), canonical
        spellings = [canonical, *_renderings(instant)]
        for number, spelling in enumerate(spellings):
            thought_id = f"t-{index}-{number}"
            await _plant_thought(v20_db, thought_id, {"expires_at": spelling})
            expected[thought_id] = canonical
            rewrites_needed += spelling != canonical
    await v20_db.commit()
    changes_before = v20_db.total_changes

    await SqliteEngravaCore(v20_db).ensure_schema()

    cursor = await v20_db.execute("SELECT thought_id, expires_at FROM thought")
    assert {str(row[0]): row[1] for row in await cursor.fetchall()} == expected
    assert v20_db.total_changes - changes_before == rewrites_needed


async def test_paging_continues_past_a_page_of_values_it_cannot_rewrite(
    v20_db: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full page of untouched values must not end the scan early."""
    monkeypatch.setattr(_core, "_TIMESTAMP_NORMALISATION_BATCH_SIZE", 3)
    planted = [
        "bad-1", "bad-2", "bad-3", "2026-01-02 03:04:05", "bad-4",
        "20260102T030405", "2026-01-02T03:04:05+00:00", "2026-01-02", "bad-5",
    ]  # fmt: skip
    for number, value in enumerate(planted):
        await _plant_thought(v20_db, f"t-{number:02d}", {"expires_at": value})
    await v20_db.commit()

    await SqliteEngravaCore(v20_db).ensure_schema()

    cursor = await v20_db.execute("SELECT expires_at FROM thought ORDER BY thought_id")
    assert [row[0] for row in await cursor.fetchall()] == [
        "bad-1", "bad-2", "bad-3", "2026-01-02T03:04:05+00:00", "bad-4",
        "2026-01-02T03:04:05+00:00", "2026-01-02T03:04:05+00:00",
        "2026-01-02T00:00:00+00:00", "bad-5",
    ]  # fmt: skip


async def test_a_rewrite_that_leaves_a_value_non_canonical_fails_the_step(
    v20_db: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step's postcondition refuses to stamp v21 over a rewrite that did not take."""
    await _plant_everything(v20_db)
    before = await _read_rows(v20_db, "thought", "thought_id", _THOUGHT_COLUMNS)

    def _broken_formatter(value: object) -> str | None:
        return value if isinstance(value, str) else None

    monkeypatch.setattr(_core, "canonical_timestamp_or_none", _broken_formatter)
    with pytest.raises(CoreMigrationError, match=r"thought\.created_at still holds"):
        await SqliteEngravaCore(v20_db).ensure_schema()

    assert await _user_version(v20_db) == _HEAD_VERSION - 1
    assert await _read_rows(v20_db, "thought", "thought_id", _THOUGHT_COLUMNS) == before


async def test_a_value_in_the_canonical_shape_is_not_read_back_or_counted(
    v20_db: aiosqlite.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """Only values without the canonical shape are read back and counted.

    The shape is checked in SQL, so a large canonical store is not re-read in
    Python -- and so a value that has the shape but names an impossible date is
    left as it is and is not part of the count, while a value without the shape
    that cannot be read is.
    """
    impossible_in_shape = "2026-02-30T12:00:00+00:00"
    unreadable_out_of_shape = "2026-02-30 12:00:00"
    await _plant_thought(v20_db, "t-in-shape", {"expires_at": impossible_in_shape})
    await _plant_thought(v20_db, "t-out-of-shape", {"expires_at": unreadable_out_of_shape})
    await v20_db.commit()

    with caplog.at_level(logging.WARNING, logger="engrava"):
        await SqliteEngravaCore(v20_db).ensure_schema()

    assert await _user_version(v20_db) == _HEAD_VERSION
    assert await _read_rows(v20_db, "thought", "thought_id", ("expires_at",)) == {
        "t-in-shape": {"expires_at": impossible_in_shape},
        "t-out-of-shape": {"expires_at": unreadable_out_of_shape},
    }
    expected_warning = (
        "Upgrading the core schema to version 21 left 1 stored timestamp value(s) "
        "unchanged because they cannot be read as an ISO-8601 instant "
        "(thought.expires_at: 1)"
    )
    assert [
        record.getMessage()
        for record in caplog.records
        if "stored timestamp value" in record.getMessage()
    ] == [expected_warning]

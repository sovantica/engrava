"""Timestamp strings a caller passes in are compared by instant, whatever their form.

The stored timestamp columns hold one canonical UTC form. A timestamp that
arrives through a public parameter, or inside a snapshot being restored, is
compared with those columns as TEXT or written into them, so it has to be put in
the same form first -- otherwise a space separator, basic format, a week date or
an offset makes the comparison or the stored value wrong.

Covered here, one per entry point:

* ``cleanup_expired(now=...)`` takes exactly the rows expired at that instant;
* ``journal.get_entries(since=...)`` filters by instant;
* ``invalidate_thought`` / ``invalidate_edge`` / ``update_thought`` /
  ``update_edge`` store the canonical form;
* ``recency_now`` already parses to an instant (pinned, not changed);
* ``engrava restore`` writes the canonical form of a snapshot's timestamps and
  leaves a value it cannot read as it was.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from click.testing import CliRunner

from engrava.cli.main import cli
from engrava.cli.snapshot_records import (
    CoreTable,
    TableRecord,
    parse_snapshot_record,
    table_spec,
)
from engrava.domain.enums import EdgeType
from engrava.domain.models import EdgeRecord, ThoughtRecord
from engrava.domain.models.edge import EDGE_TIMESTAMP_FIELDS
from engrava.domain.models.thought import THOUGHT_TIMESTAMP_FIELDS
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore
from tests.test_timestamp_canonical_form import _FixedClock, _raw_column, _thought

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


#: One instant, 2026-09-25T12:00:00Z, spelled every way ``fromisoformat`` accepts.
_NOON_SPELLINGS = (
    pytest.param("2026-09-25T12:00:00", id="naive extended"),
    pytest.param("2026-09-25 12:00:00", id="naive space separator"),
    pytest.param("20260925T120000", id="naive basic format"),
    pytest.param("2026-W39-5T12:00:00", id="week date"),
    pytest.param("2026-09-25T17:30:00+05:30", id="aware non-UTC offset"),
    pytest.param("2026-09-25T12:00:00Z", id="aware UTC, Z suffix"),
    pytest.param("2026-09-25T12:00:00+00:00", id="canonical"),
)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """A journaled file store with the ``delete`` TTL strategy."""
    conn = await aiosqlite.connect(str(tmp_path / "store.db"))
    conn.row_factory = aiosqlite.Row
    core = SqliteEngravaCore(conn, ttl_strategy="delete", journal_enabled=True)
    core._owns_connection = True
    await core.ensure_schema()
    yield core
    await core.close()


async def _thought_ids(store: SqliteEngravaCore) -> list[str]:
    cursor = await store._db.execute("SELECT thought_id FROM thought ORDER BY thought_id")
    return [str(row[0]) for row in await cursor.fetchall()]


# ---------------------------------------------------------------------------
# cleanup_expired: the "now" argument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("now", _NOON_SPELLINGS)
async def test_cleanup_expired_takes_exactly_what_expired_by_the_given_instant(
    store: SqliteEngravaCore, now: str
) -> None:
    await store.create_thought(_thought("expired", expires_at="2026-09-25T09:30:00+00:00"))
    await store.create_thought(_thought("live", expires_at="2026-09-25T14:30:00+00:00"))

    result = await store.cleanup_expired(now=now)

    assert await _thought_ids(store) == ["live"]
    assert result.expired_count == 1
    assert result.timestamp == "2026-09-25T12:00:00+00:00"


async def test_cleanup_expired_rejects_a_now_that_is_not_a_timestamp(
    store: SqliteEngravaCore,
) -> None:
    await store.create_thought(_thought("expired", expires_at="2026-09-25T09:30:00+00:00"))

    with pytest.raises(ValueError, match=r"^Must be ISO-8601 timestamp, got 'zzz'$"):
        await store.cleanup_expired(now="zzz")

    assert await _thought_ids(store) == ["expired"]


#: Valid ISO-8601, but converting either to UTC leaves the ``datetime`` range.
_NO_UTC_FORM = (
    pytest.param("0001-01-01T00:00:00+01:00", id="below year 1 in UTC"),
    pytest.param("9999-12-31T23:59:59-01:00", id="above year 9999 in UTC"),
)


async def _thought_rows(store: SqliteEngravaCore) -> list[tuple[object, ...]]:
    cursor = await store._db.execute("SELECT * FROM thought ORDER BY thought_id")
    return [tuple(row) for row in await cursor.fetchall()]


@pytest.mark.parametrize("now", _NO_UTC_FORM)
async def test_cleanup_expired_rejects_a_now_with_no_utc_form(
    store: SqliteEngravaCore, now: str
) -> None:
    await store.create_thought(_thought("expired", expires_at="2026-09-25T09:30:00+00:00"))
    rows_before = await _thought_rows(store)

    with pytest.raises(ValueError, match=r"has no UTC form within the supported datetime range"):
        await store.cleanup_expired(now=now)

    assert await _thought_rows(store) == rows_before
    assert await _thought_ids(store) == ["expired"]


# ---------------------------------------------------------------------------
# journal.get_entries: the "since" argument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("2026-09-25T11:00:00", "2026-09-25T13:00:00", id="naive extended"),
        pytest.param("2026-09-25 11:00:00", "2026-09-25 13:00:00", id="naive space separator"),
        pytest.param("20260925T110000", "20260925T130000", id="naive basic format"),
        pytest.param("2026-W39-5T11:00:00", "2026-W39-5T13:00:00", id="week date"),
        pytest.param(
            "2026-09-25T16:30:00+05:30", "2026-09-25T18:30:00+05:30", id="aware non-UTC offset"
        ),
    ],
)
async def test_journal_since_filters_by_instant(
    store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch, before: str, after: str
) -> None:
    # Every journal entry below is stamped 2026-09-25T12:00:00.500000+00:00.
    monkeypatch.setattr(datetime, "datetime", _FixedClock)
    await store.create_thought(_thought("a"))
    await store.create_thought(_thought("b"))
    journal = store.journal
    assert journal is not None

    since_before = await journal.get_entries(since=before)
    since_after = await journal.get_entries(since=after)

    assert [entry.target_id for entry in since_before] == ["a", "b"]
    assert since_after == []


async def test_journal_since_rejects_a_value_that_is_not_a_timestamp(
    store: SqliteEngravaCore,
) -> None:
    await store.create_thought(_thought("a"))
    journal = store.journal
    assert journal is not None

    with pytest.raises(ValueError, match=r"^Must be ISO-8601 timestamp, got 'zzz'$"):
        await journal.get_entries(since="zzz")


@pytest.mark.parametrize("since", _NO_UTC_FORM)
async def test_journal_since_rejects_a_value_with_no_utc_form(
    store: SqliteEngravaCore, since: str
) -> None:
    await store.create_thought(_thought("a"))
    journal = store.journal
    assert journal is not None

    with pytest.raises(ValueError, match=r"has no UTC form within the supported datetime range"):
        await journal.get_entries(since=since)


# ---------------------------------------------------------------------------
# Write paths that take a timestamp argument
# ---------------------------------------------------------------------------


async def test_every_timestamp_write_argument_is_stored_canonical(
    store: SqliteEngravaCore,
) -> None:
    await store.create_thought(_thought("a", valid_from="2026-01-01T00:00:00+00:00"))
    await store.create_thought(_thought("b", valid_from="2026-01-01T00:00:00+00:00"))
    await store.create_thought(_thought("c"))
    await store.create_edge(
        EdgeRecord(
            edge_id="e-1",
            from_thought_id="a",
            to_thought_id="b",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=0,
        )
    )
    await store.create_edge(
        EdgeRecord(
            edge_id="e-2",
            from_thought_id="b",
            to_thought_id="c",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=0,
        )
    )

    await store.invalidate_thought("a", valid_until="2026-07-01 00:00:00")
    await store.invalidate_edge("e-1", valid_until="20260701T000000")
    await store.update_thought("c", expires_at="2099-W01-1T00:00:00")
    await store.update_edge("e-2", valid_from="2026-01-01 00:00:00.5")

    assert {
        "invalidate_thought": await _raw_column(store, "thought", "valid_until", "a"),
        "invalidate_edge": await _raw_column(store, "edge", "valid_until", "e-1"),
        "update_thought": await _raw_column(store, "thought", "expires_at", "c"),
        "update_edge": await _raw_column(store, "edge", "valid_from", "e-2"),
    } == {
        "invalidate_thought": "2026-07-01T00:00:00+00:00",
        "invalidate_edge": "2026-07-01T00:00:00+00:00",
        "update_thought": "2098-12-29T00:00:00+00:00",
        "update_edge": "2026-01-01T00:00:00.500000+00:00",
    }


# ---------------------------------------------------------------------------
# recency_now: already an instant (pinned so it stays that way)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recency_now", _NOON_SPELLINGS)
async def test_recency_now_is_read_as_an_instant(
    store: SqliteEngravaCore, recency_now: str
) -> None:
    await store.create_thought(_thought("old", updated_at="2026-09-18T12:00:00+00:00"))
    await store.create_thought(_thought("new", updated_at="2026-09-25T11:00:00+00:00"))

    canonical = await store.search_hybrid(
        "",
        recency_now="2026-09-25T12:00:00+00:00",
        recency_weight=1.0,
        recency_now_half_life=86400,
    )
    spelled = await store.search_hybrid(
        "", recency_now=recency_now, recency_weight=1.0, recency_now_half_life=86400
    )

    assert [tid for tid, _ in canonical.results] == ["new", "old"]
    assert spelled.results == canonical.results


# ---------------------------------------------------------------------------
# engrava restore
# ---------------------------------------------------------------------------


def _write_snapshot(path: Path) -> None:
    thought = {
        "thought_id": "t-naive",
        "thought_type": "OBSERVATION",
        "essence": "essence",
        "content": "content",
        "priority": "P2",
        "created_at": "2026-01-02 03:04:05",
        "updated_at": "20260102T030405",
        "last_accessed_at": "2026-W01-5T03:04:05",
        "expires_at": "2099-12-31 23:00:00.5",
        "valid_from": "2026-01-01",
        "valid_until": "2099-07-01T02:00:00+02:00",
        "archived_at": "2026-09-25T12:00:00.000000+00:00",
    }
    other = {
        "thought_id": "t-other",
        "thought_type": "OBSERVATION",
        "essence": "other",
        "content": "other",
        "priority": "P2",
        # Cannot be read as an instant: restored exactly as it was.
        "created_at": "not a timestamp",
    }
    edge = {
        "edge_id": "e-naive",
        "from_thought_id": "t-naive",
        "to_thought_id": "t-other",
        "edge_type": "ASSOCIATED",
        "valid_from": "2026-01-01 00:00:00",
        "valid_until": "20990701T000000",
    }
    lines = [
        {"_type": "metadata", "schema_version": 20},
        {"_type": "thought", "data": thought},
        {"_type": "thought", "data": other},
        {"_type": "edge", "data": edge},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def test_restore_writes_the_canonical_form_of_snapshot_timestamps(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.jsonl"
    _write_snapshot(snapshot)
    target = tmp_path / "target.db"

    result = CliRunner().invoke(cli, ["--db", str(target), "restore", "-i", str(snapshot)])

    with sqlite3.connect(target) as conn:
        thoughts = {
            row[0]: row[1:]
            for row in conn.execute(
                "SELECT thought_id, created_at, updated_at, last_accessed_at, expires_at, "
                "valid_from, valid_until, archived_at FROM thought"
            )
        }
        edges = {
            row[0]: row[1:]
            for row in conn.execute("SELECT edge_id, valid_from, valid_until FROM edge")
        }
    assert thoughts == {
        "t-naive": (
            "2026-01-02T03:04:05+00:00",
            "2026-01-02T03:04:05+00:00",
            "2026-01-02T03:04:05+00:00",
            "2099-12-31T23:00:00.500000+00:00",
            "2026-01-01T00:00:00+00:00",
            "2099-07-01T00:00:00+00:00",
            "2026-09-25T12:00:00+00:00",
        ),
        "t-other": ("not a timestamp", None, None, None, None, None, None),
    }
    assert edges == {"e-naive": ("2026-01-01T00:00:00+00:00", "2099-07-01T00:00:00+00:00")}
    assert result.exit_code == 0, result.output


def test_restore_canonicalises_exactly_the_columns_the_model_does() -> None:
    """Restore's timestamp columns come from the model, and exist in the snapshot spec."""
    thought_spec = table_spec(CoreTable.THOUGHT)
    edge_spec = table_spec(CoreTable.EDGE)
    assert thought_spec.timestamp_columns == THOUGHT_TIMESTAMP_FIELDS
    assert edge_spec.timestamp_columns == EDGE_TIMESTAMP_FIELDS
    assert set(THOUGHT_TIMESTAMP_FIELDS) <= set(thought_spec.columns)
    assert set(EDGE_TIMESTAMP_FIELDS) <= set(edge_spec.columns)
    assert table_spec(CoreTable.EMBEDDING).timestamp_columns == ()
    assert table_spec(CoreTable.ACTION).timestamp_columns == ()
    # The model's constant is the list its validator actually runs on.
    validated = ThoughtRecord.model_validate(
        {
            "thought_id": "t",
            "thought_type": "OBSERVATION",
            "essence": "e",
            "content": "c",
            "priority": "P2",
            "lifecycle_status": "ACTIVE",
            "created_cycle": 0,
            "updated_cycle": 0,
            "source": "s",
            **dict.fromkeys(THOUGHT_TIMESTAMP_FIELDS, "2026-01-02 03:04:05"),
        }
    )
    assert {getattr(validated, name) for name in THOUGHT_TIMESTAMP_FIELDS} == {
        "2026-01-02T03:04:05+00:00"
    }


def test_a_canonical_snapshot_record_is_passed_through_unchanged() -> None:
    data = {
        "thought_id": "t",
        "thought_type": "OBSERVATION",
        "essence": "e",
        "content": "c",
        "priority": "P2",
        "created_at": "2026-01-02T03:04:05+00:00",
        "expires_at": None,
    }
    record = parse_snapshot_record(json.dumps({"_type": "thought", "data": data}), line_number=1)
    assert isinstance(record, TableRecord)
    assert record.data == data


@pytest.mark.parametrize("planted", _NO_UTC_FORM)
async def test_invalidate_against_a_stored_bound_with_no_utc_form_is_a_value_error(
    store: SqliteEngravaCore, planted: str
) -> None:
    """The upgrade leaves such a stored ``valid_from`` untouched; invalidating
    against it is a ``ValueError`` like any other unusable bound, and writes
    nothing."""
    await store.create_thought(_thought("a"))
    await store.create_thought(_thought("b"))
    await store.create_edge(
        EdgeRecord(
            edge_id="e-1",
            from_thought_id="a",
            to_thought_id="b",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=0,
        )
    )
    await store._db.execute("UPDATE thought SET valid_from = ? WHERE thought_id = 'a'", (planted,))
    await store._db.execute("UPDATE edge SET valid_from = ? WHERE edge_id = 'e-1'", (planted,))
    await store._db.commit()
    thoughts_before = await _thought_rows(store)
    edges_cursor = await store._db.execute("SELECT * FROM edge ORDER BY edge_id")
    edges_before = [tuple(row) for row in await edges_cursor.fetchall()]

    with pytest.raises(ValueError, match=r"has no UTC form within the supported datetime range"):
        await store.invalidate_thought("a", valid_until="2026-07-01T00:00:00+00:00")
    with pytest.raises(ValueError, match=r"has no UTC form within the supported datetime range"):
        await store.invalidate_edge("e-1", valid_until="2026-07-01T00:00:00+00:00")

    edges_cursor = await store._db.execute("SELECT * FROM edge ORDER BY edge_id")
    assert [tuple(row) for row in await edges_cursor.fetchall()] == edges_before
    assert await _thought_rows(store) == thoughts_before

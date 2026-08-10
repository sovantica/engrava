"""Store-level enforcement of the valid-time interval-ordering invariant.

The ``valid_from <= valid_until`` invariant (see
:mod:`engrava.domain.models._temporal`) lives in the domain model, so it is
enforced wherever a record crosses the model boundary — not only at explicit
construction. These tests pin the two non-obvious store paths:

* **Read path** — ``get_thought`` / ``get_edges`` reconstruct domain models from
  raw rows (``_row_to_thought`` / ``_row_to_edge``), so a row that became
  inverted *out of band* (written by an older build before this validation
  existed, or edited directly in the database file) fails loudly with a
  :class:`pydantic.ValidationError` on read rather than surfacing a corrupt
  interval.
* **Update path** — ``update_thought`` / ``update_edge`` re-validate the whole
  record via ``model_validate``, so any field change that would invert a stored
  interval is rejected before the row is written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest
from pydantic import ValidationError

from engrava import (
    EdgeRecord,
    EdgeType,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_T_JAN = "2025-01-01T00:00:00+00:00"
_T_APR = "2025-04-01T00:00:00+00:00"
_T_DEC = "2025-12-01T00:00:00+00:00"
_T_BEFORE = "2024-06-01T00:00:00+00:00"  # precedes _T_JAN

# Bounds for a row corrupted directly in the database (from > until).
_INVERTED_FROM = "2026-06-01T00:00:00+00:00"
_INVERTED_UNTIL = "2026-03-01T00:00:00+00:00"


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """In-memory store with the core schema applied."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    core = SqliteEngravaCore(conn)
    await core.ensure_schema()
    yield core
    await conn.close()


def _mk_thought(
    thought_id: str,
    *,
    valid_from: str | None = _T_JAN,
    valid_until: str | None = None,
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content="content",
        priority=Priority.P1,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=1,
        updated_cycle=1,
        source="test",
        valid_from=valid_from,
        valid_until=valid_until,
    )


async def _seed_edge(
    store: SqliteEngravaCore,
    *,
    valid_from: str | None = _T_JAN,
    valid_until: str | None = None,
) -> None:
    await store.create_thought(_mk_thought("t1"))
    await store.create_thought(_mk_thought("t2"))
    await store.create_edge(
        EdgeRecord(
            edge_id="e1",
            from_thought_id="t1",
            to_thought_id="t2",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.5,
            created_cycle=1,
            valid_from=valid_from,
            valid_until=valid_until,
        )
    )


class TestReadPathEnforcement:
    """A stored inverted interval fails loudly on read reconstruction."""

    async def test_thought_read_rejects_inverted_row(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_mk_thought("t1"))
        # Corrupt the row out of band, bypassing model validation — this
        # simulates a row written before the invariant was enforced.
        await store._db.execute(
            "UPDATE thought SET valid_from = ?, valid_until = ? WHERE thought_id = ?",
            (_INVERTED_FROM, _INVERTED_UNTIL, "t1"),
        )
        await store._db.commit()
        with pytest.raises(ValidationError, match="inverted validity interval"):
            await store.get_thought("t1")

    async def test_edge_read_rejects_inverted_row(self, store: SqliteEngravaCore) -> None:
        await _seed_edge(store)
        await store._db.execute(
            "UPDATE edge SET valid_from = ?, valid_until = ? WHERE edge_id = ?",
            (_INVERTED_FROM, _INVERTED_UNTIL, "e1"),
        )
        await store._db.commit()
        with pytest.raises(ValidationError, match="inverted validity interval"):
            await store.get_edges("t1")


class TestUpdatePathEnforcement:
    """The generic update path re-validates and rejects an inverting change."""

    async def test_update_thought_backdated_valid_until_rejected(
        self, store: SqliteEngravaCore
    ) -> None:
        await store.create_thought(_mk_thought("t1"))  # valid_from == _T_JAN
        with pytest.raises(ValidationError, match="inverted validity interval"):
            await store.update_thought("t1", valid_until=_T_BEFORE)

    async def test_update_edge_backdated_valid_until_rejected(
        self, store: SqliteEngravaCore
    ) -> None:
        await _seed_edge(store)  # valid_from == _T_JAN
        with pytest.raises(ValidationError, match="inverted validity interval"):
            await store.update_edge("e1", valid_until=_T_BEFORE)

    async def test_update_thought_inverting_valid_from_rejected(
        self, store: SqliteEngravaCore
    ) -> None:
        await store.create_thought(_mk_thought("t1", valid_from=_T_JAN, valid_until=_T_APR))
        # Moving valid_from past the stored valid_until inverts the interval.
        with pytest.raises(ValidationError, match="inverted validity interval"):
            await store.update_thought("t1", valid_from=_T_DEC)

    async def test_update_edge_inverting_valid_from_rejected(
        self, store: SqliteEngravaCore
    ) -> None:
        await _seed_edge(store, valid_until=_T_APR)
        with pytest.raises(ValidationError, match="inverted validity interval"):
            await store.update_edge("e1", valid_from=_T_DEC)

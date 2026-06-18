"""Shared fixtures for the MCP server tests.

Builds a real in-memory ``SqliteEngravaCore`` (no vector backend, so
hybrid search degrades to its lexical backend) seeded with a couple of
thoughts.  Tools are exercised directly against this store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

#: Identifiers of the thoughts seeded by the ``store`` fixture.
SEEDED_THOUGHT_IDS = ("thought-alpha", "thought-beta")


def make_thought(
    thought_id: str,
    *,
    essence: str,
    content: str,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    priority: Priority = Priority.P2,
) -> CoreThoughtRecord:
    """Build a core thought record for seeding.

    Args:
        thought_id: Stable identifier for the thought.
        essence: Compact canonical text.
        content: Full stored content.
        lifecycle_status: Lifecycle state to persist.
        priority: Priority level.

    Returns:
        A constructed ``CoreThoughtRecord``.

    """
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.BELIEF,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """Yield a seeded in-memory store with no vector backend.

    Yields:
        A ``SqliteEngravaCore`` containing two active thoughts.

    """
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys=ON")
    backend = SqliteEngravaCore(connection)
    await backend.ensure_schema()

    await backend.create_thought(
        make_thought(
            "thought-alpha",
            essence="Coffee brewing notes",
            content="Pour-over coffee extracts best between 90 and 96 degrees.",
        )
    )
    await backend.create_thought(
        make_thought(
            "thought-beta",
            essence="Tea steeping notes",
            content="Green tea steeps best below boiling to avoid bitterness.",
            priority=Priority.P1,
        )
    )

    try:
        yield backend
    finally:
        await connection.close()

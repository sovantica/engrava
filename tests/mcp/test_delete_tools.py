"""Unit tests for the MCP delete-tool implementations.

Each tool implementation is exercised directly against a seeded
in-memory store (see ``conftest.store``).  The repeated-delete cases are
the evidence that the delete tools are genuinely idempotent: a delete of
an absent identifier returns ``{"deleted": False}`` without raising, so a
client may safely retry a delete that appeared to fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engrava import EdgeType
from engrava.domain.models.edge import EdgeRecord
from engrava.mcp.server import (
    delete_edge_impl,
    delete_thought_impl,
    get_thought_impl,
    link_thoughts_impl,
)

if TYPE_CHECKING:
    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore


class TestDeleteThought:
    """Tests for the ``delete_thought`` tool."""

    async def test_removes_existing_thought(self, store: SqliteEngravaCore) -> None:
        result = await delete_thought_impl(store, "thought-alpha")
        assert result == {"deleted": True}

        # The thought is gone: a read-back reports it missing.
        read_back = await get_thought_impl(store, "thought-alpha")
        assert read_back["found"] is False
        assert read_back["thought"] is None

    async def test_repeated_delete_is_idempotent(self, store: SqliteEngravaCore) -> None:
        # First delete removes the thought; the second converges on the same
        # end state (already gone) and reports it without raising. This is the
        # evidence behind the destructive-but-idempotent annotation.
        first = await delete_thought_impl(store, "thought-alpha")
        assert first == {"deleted": True}

        second = await delete_thought_impl(store, "thought-alpha")
        assert second == {"deleted": False}

    async def test_unknown_id_is_not_an_error(self, store: SqliteEngravaCore) -> None:
        result = await delete_thought_impl(store, "never-existed")
        assert result == {"deleted": False}


class TestDeleteEdge:
    """Tests for the ``delete_edge`` tool."""

    async def test_removes_existing_edge(self, store: SqliteEngravaCore) -> None:
        created = await link_thoughts_impl(
            store,
            "thought-alpha",
            "thought-beta",
            EdgeType.ASSOCIATED,
        )
        edge_id = created["edge"]["edge_id"]

        result = await delete_edge_impl(store, edge_id)
        assert result == {"deleted": True}

        # The edge is gone: the source thought has no outgoing edges left.
        edges = await store.get_edges("thought-alpha", direction="OUT")
        assert edges == []

    async def test_repeated_delete_is_idempotent(self, store: SqliteEngravaCore) -> None:
        edge_id = "edge-to-remove"
        await store.create_edge(
            EdgeRecord(
                edge_id=edge_id,
                from_thought_id="thought-alpha",
                to_thought_id="thought-beta",
                edge_type=EdgeType.ASSOCIATED,
                weight=1.0,
                created_cycle=0,
            )
        )

        first = await delete_edge_impl(store, edge_id)
        assert first == {"deleted": True}

        # The repeated delete converges on the same end state without raising.
        second = await delete_edge_impl(store, edge_id)
        assert second == {"deleted": False}

    async def test_unknown_id_is_not_an_error(self, store: SqliteEngravaCore) -> None:
        result = await delete_edge_impl(store, "never-existed")
        assert result == {"deleted": False}

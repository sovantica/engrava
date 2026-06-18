"""Unit tests for the MCP write-tool implementations.

Each tool implementation is exercised directly against a seeded
in-memory store (see ``conftest.store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    EdgeType,
    LifecycleStatus,
    Priority,
    ThoughtNotFoundError,
    ThoughtType,
)
from engrava.domain.exceptions import ReferentialIntegrityError
from engrava.mcp.server import (
    get_thought_impl,
    link_thoughts_impl,
    store_thought_impl,
    update_thought_impl,
)

if TYPE_CHECKING:
    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore


class TestStoreThought:
    """Tests for the ``store_thought`` tool."""

    async def test_creates_and_reads_back(self, store: SqliteEngravaCore) -> None:
        result = await store_thought_impl(
            store,
            essence="Sourdough starter notes",
            content="Feed the starter twice daily at a 1:1:1 ratio.",
        )
        new_id = result["thought"]["thought_id"]
        assert isinstance(new_id, str)
        assert new_id

        read_back = await get_thought_impl(store, new_id)
        assert read_back["found"] is True
        thought = read_back["thought"]
        assert thought is not None
        assert thought["essence"] == "Sourdough starter notes"
        # New thoughts start in the CREATED lifecycle state.
        assert thought["lifecycle_status"] == "CREATED"

    async def test_generates_unique_ids(self, store: SqliteEngravaCore) -> None:
        first = await store_thought_impl(store, essence="One", content="First body.")
        second = await store_thought_impl(store, essence="Two", content="Second body.")
        assert first["thought"]["thought_id"] != second["thought"]["thought_id"]

    async def test_honours_supplied_id_and_fields(self, store: SqliteEngravaCore) -> None:
        result = await store_thought_impl(
            store,
            essence="Explicit identity",
            content="A thought with a caller-chosen id.",
            thought_type=ThoughtType.TASK,
            priority=Priority.P1,
            thought_id="thought-explicit",
        )
        thought = result["thought"]
        assert thought["thought_id"] == "thought-explicit"
        assert thought["thought_type"] == "TASK"
        assert thought["priority"] == "P1"

    async def test_deduplicate_collapses_on_content_hash(self, store: SqliteEngravaCore) -> None:
        content = "Identical body used to trigger content-hash dedup."
        first = await store_thought_impl(
            store, essence="Dedup A", content=content, deduplicate=True
        )
        second = await store_thought_impl(
            store, essence="Dedup B", content=content, deduplicate=True
        )
        assert first["thought"]["thought_id"] == second["thought"]["thought_id"]


class TestUpdateThought:
    """Tests for the ``update_thought`` tool."""

    async def test_changes_supplied_field(self, store: SqliteEngravaCore) -> None:
        result = await update_thought_impl(
            store,
            "thought-alpha",
            essence="Updated coffee notes",
        )
        assert result["thought"]["essence"] == "Updated coffee notes"

        read_back = await get_thought_impl(store, "thought-alpha")
        thought = read_back["thought"]
        assert thought is not None
        assert thought["essence"] == "Updated coffee notes"

    async def test_changes_lifecycle_status(self, store: SqliteEngravaCore) -> None:
        # The seeded thought is ACTIVE; ACTIVE -> DONE is a valid transition.
        result = await update_thought_impl(
            store,
            "thought-alpha",
            lifecycle_status=LifecycleStatus.DONE,
        )
        assert result["thought"]["lifecycle_status"] == "DONE"

    async def test_changes_content_and_confidence(self, store: SqliteEngravaCore) -> None:
        await update_thought_impl(
            store,
            "thought-alpha",
            content="Rewritten brewing guidance.",
            confidence=0.75,
        )
        read_back = await get_thought_impl(store, "thought-alpha")
        thought = read_back["thought"]
        assert thought is not None
        assert thought["content"] == "Rewritten brewing guidance."
        assert thought["confidence"] == 0.75

    async def test_omitted_fields_are_untouched(self, store: SqliteEngravaCore) -> None:
        before = await get_thought_impl(store, "thought-beta")
        before_thought = before["thought"]
        assert before_thought is not None
        original_content = before_thought["content"]

        await update_thought_impl(store, "thought-beta", priority=Priority.P3)

        after = await get_thought_impl(store, "thought-beta")
        after_thought = after["thought"]
        assert after_thought is not None
        assert after_thought["content"] == original_content
        assert after_thought["priority"] == "P3"

    async def test_missing_thought_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ThoughtNotFoundError):
            await update_thought_impl(store, "does-not-exist", essence="x")


class TestLinkThoughts:
    """Tests for the ``link_thoughts`` tool."""

    async def test_creates_edge_between_existing_thoughts(self, store: SqliteEngravaCore) -> None:
        result = await link_thoughts_impl(
            store,
            "thought-alpha",
            "thought-beta",
            EdgeType.ASSOCIATED,
            weight=0.5,
        )
        edge = result["edge"]
        assert edge["from_thought_id"] == "thought-alpha"
        assert edge["to_thought_id"] == "thought-beta"
        assert edge["edge_type"] == "ASSOCIATED"
        assert edge["weight"] == 0.5

        edges = await store.get_edges("thought-alpha", direction="OUT")
        assert any(
            e.to_thought_id == "thought-beta" and e.edge_type is EdgeType.ASSOCIATED for e in edges
        )

    async def test_generates_edge_id_when_omitted(self, store: SqliteEngravaCore) -> None:
        result = await link_thoughts_impl(
            store,
            "thought-alpha",
            "thought-beta",
            EdgeType.DEPENDS_ON,
        )
        edge_id = result["edge"]["edge_id"]
        assert isinstance(edge_id, str)
        assert edge_id

    async def test_default_weight_applied(self, store: SqliteEngravaCore) -> None:
        result = await link_thoughts_impl(
            store,
            "thought-beta",
            "thought-alpha",
            EdgeType.DERIVED_FROM,
        )
        assert result["edge"]["weight"] == 1.0

    async def test_missing_endpoint_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ReferentialIntegrityError):
            await link_thoughts_impl(
                store,
                "thought-alpha",
                "ghost-thought",
                EdgeType.ASSOCIATED,
            )

    async def test_duplicate_link_is_rejected_not_idempotent(
        self, store: SqliteEngravaCore
    ) -> None:
        # An edge is unique per (from, to, type). Linking the same pair with
        # the same type twice must be rejected rather than silently ignored or
        # converged — this is why link_thoughts is annotated idempotentHint=False.
        await link_thoughts_impl(
            store,
            "thought-alpha",
            "thought-beta",
            EdgeType.ASSOCIATED,
        )
        with pytest.raises(aiosqlite.IntegrityError):
            await link_thoughts_impl(
                store,
                "thought-alpha",
                "thought-beta",
                EdgeType.ASSOCIATED,
            )

        # The failed retry left exactly one edge — the write did not converge.
        edges = await store.get_edges("thought-alpha", direction="OUT")
        matching = [
            e
            for e in edges
            if e.to_thought_id == "thought-beta" and e.edge_type is EdgeType.ASSOCIATED
        ]
        assert len(matching) == 1

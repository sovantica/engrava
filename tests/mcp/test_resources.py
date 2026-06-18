"""End-to-end tests for the MCP read-only resources.

Exercises the resources through the in-memory MCP client transport (so
registration, URI-template binding, and JSON serialisation all run for
real), mirroring the tool tests in :mod:`tests.mcp.test_server`.

Resources are reads by definition, so they are advertised in both the
default and the read-only deployment; the read-only cases below assert
that independence directly.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import aiosqlite
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava import (
    CoreThoughtRecord,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)
from engrava.mcp import build_server
from engrava.mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR
from engrava.mcp.server import READ_ONLY_ENV_VAR

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

#: Static resource URIs the server must advertise via ``list_resources``.
STATIC_RESOURCE_URIS = frozenset({"engrava://stats", "engrava://recent"})
#: Templated resource URI advertised via ``list_resource_templates``.
THOUGHT_TEMPLATE_URI = "engrava://thought/{thought_id}"


async def _seed_two_thoughts(path: Path) -> None:
    """Create a database file with two thoughts updated in a known order.

    The second thought carries the larger ``updated_cycle`` so it is the
    most recent — ``list_thoughts`` orders by descending ``updated_cycle``,
    so ``engrava://recent`` must return it first.

    Args:
        path: Filesystem path for the new database.

    """
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(connection)
    await store.ensure_schema()
    await store.create_thought(
        CoreThoughtRecord(
            thought_id="older-thought",
            thought_type=ThoughtType.BELIEF,
            essence="Older note",
            content="The earlier of the two seeded thoughts.",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=1,
            updated_cycle=1,
            source="test",
        )
    )
    await store.create_thought(
        CoreThoughtRecord(
            thought_id="newer-thought",
            thought_type=ThoughtType.BELIEF,
            essence="Newer note",
            content="The later of the two seeded thoughts.",
            priority=Priority.P1,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=2,
            updated_cycle=2,
            source="test",
        )
    )
    await connection.close()


def _decode_single(result: object) -> dict[str, object]:
    """Parse the single JSON text payload of a ``read_resource`` result.

    Args:
        result: The ``ReadResourceResult`` returned by ``read_resource``.

    Returns:
        The decoded JSON object carried by the result's sole content
        block.

    """
    contents = result.contents  # type: ignore[attr-defined]
    assert len(contents) == 1
    block = contents[0]
    assert block.mimeType == "application/json"
    decoded = json.loads(block.text)
    assert isinstance(decoded, dict)
    return decoded


class TestResourceListing:
    """List resources and templates through a connected client."""

    async def test_static_resources_are_listed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "list.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_resources()

        assert {str(resource.uri) for resource in listed.resources} == STATIC_RESOURCE_URIS

    async def test_thought_template_is_listed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "templates.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_resource_templates()

        templates = {template.uriTemplate for template in listed.resourceTemplates}
        assert THOUGHT_TEMPLATE_URI in templates


class TestResourceReads:
    """Read each resource through a connected client."""

    async def test_thought_resource_returns_seeded_thought(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "thought.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.read_resource("engrava://thought/newer-thought")

        payload = _decode_single(result)
        assert payload["found"] is True
        thought = payload["thought"]
        assert isinstance(thought, dict)
        assert thought["thought_id"] == "newer-thought"
        assert thought["essence"] == "Newer note"

    async def test_thought_resource_unknown_id_is_graceful(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "missing.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        # An unknown identifier must not raise over the transport; it
        # returns a not-found payload, mirroring the get_thought tool.
        async with connect_client(server) as client:
            result = await client.read_resource("engrava://thought/no-such-id")

        payload = _decode_single(result)
        assert payload == {"found": False, "thought": None}

    async def test_recent_resource_orders_newest_first(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "recent.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.read_resource("engrava://recent")

        payload = _decode_single(result)
        thoughts = payload["thoughts"]
        assert isinstance(thoughts, list)
        ids = [thought["thought_id"] for thought in thoughts]
        # list_thoughts orders by descending updated_cycle, so the newer
        # thought comes first.
        assert ids == ["newer-thought", "older-thought"]
        assert payload["limit"] == 10

    async def test_stats_resource_matches_memory_stats_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "stats.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            resource_result = await client.read_resource("engrava://stats")
            tool_result = await client.call_tool("memory_stats", {})

        resource_payload = _decode_single(resource_result)
        assert tool_result.structuredContent is not None
        # The resource and the memory_stats tool share memory_stats_impl,
        # so they must agree field-for-field (no duplicate stats logic).
        assert resource_payload == tool_result.structuredContent
        assert resource_payload["thought_count"] == 2


class TestResourcesInReadOnlyMode:
    """Resources are reads, so they survive the write-tool gate."""

    async def test_resources_listed_in_read_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ro_list.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            static = await client.list_resources()
            templates = await client.list_resource_templates()

        # Read-only mode hides the write tools but must not hide resources.
        assert {str(resource.uri) for resource in static.resources} == STATIC_RESOURCE_URIS
        assert THOUGHT_TEMPLATE_URI in {
            template.uriTemplate for template in templates.resourceTemplates
        }

    async def test_resources_readable_in_read_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "ro_read.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            stats = await client.read_resource("engrava://stats")
            recent = await client.read_resource("engrava://recent")
            thought = await client.read_resource("engrava://thought/newer-thought")

        assert _decode_single(stats)["thought_count"] == 2
        assert len(_decode_single(recent)["thoughts"]) == 2
        assert _decode_single(thought)["found"] is True

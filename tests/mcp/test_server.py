"""End-to-end and store-resolution tests for the MCP server.

Exercises the server through the in-memory MCP client transport (so the
lifespan, tool registration, and JSON serialisation all run for real) and
covers store resolution from environment variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava import (
    CoreThoughtRecord,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)
from engrava.mcp import build_server
from engrava.mcp.config import (
    CONFIG_ENV_VAR,
    DB_PATH_ENV_VAR,
    ResolvedStore,
    StoreResolutionError,
    resolve_store,
)
from engrava.mcp.server import READ_ONLY_ENV_VAR

if TYPE_CHECKING:
    from pathlib import Path

READ_TOOL_NAMES = frozenset(
    {"get_thought", "search_memory", "search_keywords", "query_memory", "memory_stats"}
)
WRITE_TOOL_NAMES = frozenset(
    {"store_thought", "update_thought", "link_thoughts", "delete_thought", "delete_edge"}
)
#: The subset of write tools that remove data and therefore carry
#: ``destructiveHint=True``.
DESTRUCTIVE_TOOL_NAMES = frozenset({"delete_thought", "delete_edge"})
EXPECTED_TOOL_NAMES = READ_TOOL_NAMES | WRITE_TOOL_NAMES


async def _seed_database(path: Path) -> None:
    """Create a database file with a single active thought.

    Args:
        path: Filesystem path for the new database.

    """
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(connection)
    await store.ensure_schema()
    await store.create_thought(
        CoreThoughtRecord(
            thought_id="seeded-1",
            thought_type=ThoughtType.BELIEF,
            essence="Persisted note",
            content="A note that survives a fresh connection.",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="test",
        )
    )
    await connection.close()


class TestServerEndToEnd:
    """Drive the server through a connected in-memory client."""

    async def test_lists_read_and_write_tools_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "tools.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_tools()

        read_only_by_name: dict[str, bool | None] = {}
        idempotent_by_name: dict[str, bool | None] = {}
        destructive_by_name: dict[str, bool | None] = {}
        for tool in listed.tools:
            # Every tool must carry an annotation block.
            assert tool.annotations is not None
            read_only_by_name[tool.name] = tool.annotations.readOnlyHint
            idempotent_by_name[tool.name] = tool.annotations.idempotentHint
            destructive_by_name[tool.name] = tool.annotations.destructiveHint

        assert set(read_only_by_name) == EXPECTED_TOOL_NAMES
        # The read tools are read-only and the write tools are not.
        assert all(read_only_by_name[name] for name in READ_TOOL_NAMES)
        assert all(read_only_by_name[name] is False for name in WRITE_TOOL_NAMES)

        # Idempotency hints must match the real store semantics a client
        # would rely on for safe retries:
        #   - update_thought converges on the same end state  -> idempotent
        #   - store_thought creates a fresh node each call    -> NOT idempotent
        #   - link_thoughts rejects a duplicate (from,to,type) -> NOT idempotent
        #   - delete_* of an absent id is a no-op, same end state -> idempotent
        assert idempotent_by_name["update_thought"] is True
        assert idempotent_by_name["store_thought"] is False
        assert idempotent_by_name["link_thoughts"] is False
        assert idempotent_by_name["delete_thought"] is True
        assert idempotent_by_name["delete_edge"] is True

        # Only the delete tools remove data, so only they are destructive.
        assert all(destructive_by_name[name] is True for name in DESTRUCTIVE_TOOL_NAMES)
        assert all(
            destructive_by_name[name] is False for name in WRITE_TOOL_NAMES - DESTRUCTIVE_TOOL_NAMES
        )

    async def test_read_only_mode_hides_write_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ro.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_tools()

        assert {tool.name for tool in listed.tools} == READ_TOOL_NAMES

    async def test_write_tools_round_trip_over_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "writes.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            created = await client.call_tool(
                "store_thought",
                {"essence": "Live note", "content": "Stored over the transport."},
            )
            assert created.isError is False
            assert created.structuredContent is not None
            first_id = created.structuredContent["thought"]["thought_id"]

            second = await client.call_tool(
                "store_thought",
                {"essence": "Second note", "content": "Another stored note."},
            )
            assert second.structuredContent is not None
            second_id = second.structuredContent["thought"]["thought_id"]

            updated = await client.call_tool(
                "update_thought",
                {"thought_id": first_id, "essence": "Edited note"},
            )
            assert updated.isError is False
            assert updated.structuredContent is not None
            assert updated.structuredContent["thought"]["essence"] == "Edited note"

            linked = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": first_id,
                    "to_thought_id": second_id,
                    "edge_type": "ASSOCIATED",
                },
            )
            assert linked.isError is False
            assert linked.structuredContent is not None
            assert linked.structuredContent["edge"]["from_thought_id"] == first_id

            fetched = await client.call_tool("get_thought", {"thought_id": first_id})

        assert fetched.structuredContent is not None
        assert fetched.structuredContent["found"] is True
        assert fetched.structuredContent["thought"]["essence"] == "Edited note"

    async def test_delete_tools_round_trip_over_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "deletes.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            first = await client.call_tool(
                "store_thought",
                {"essence": "From note", "content": "Source thought."},
            )
            assert first.structuredContent is not None
            first_id = first.structuredContent["thought"]["thought_id"]

            second = await client.call_tool(
                "store_thought",
                {"essence": "To note", "content": "Target thought."},
            )
            assert second.structuredContent is not None
            second_id = second.structuredContent["thought"]["thought_id"]

            linked = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": first_id,
                    "to_thought_id": second_id,
                    "edge_type": "ASSOCIATED",
                },
            )
            assert linked.structuredContent is not None
            edge_id = linked.structuredContent["edge"]["edge_id"]

            deleted_edge = await client.call_tool("delete_edge", {"edge_id": edge_id})
            assert deleted_edge.isError is False
            assert deleted_edge.structuredContent is not None
            assert deleted_edge.structuredContent["deleted"] is True

            deleted_thought = await client.call_tool("delete_thought", {"thought_id": first_id})
            assert deleted_thought.isError is False
            assert deleted_thought.structuredContent is not None
            assert deleted_thought.structuredContent["deleted"] is True

            # Deleting the same thought again converges on the same end state
            # (already gone) and reports it without erroring.
            again = await client.call_tool("delete_thought", {"thought_id": first_id})
            assert again.isError is False
            assert again.structuredContent is not None
            assert again.structuredContent["deleted"] is False

            fetched = await client.call_tool("get_thought", {"thought_id": first_id})

        assert fetched.structuredContent is not None
        assert fetched.structuredContent["found"] is False

    async def test_get_thought_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "seeded.db"
        await _seed_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.call_tool("get_thought", {"thought_id": "seeded-1"})

        assert result.isError is False
        assert result.structuredContent is not None
        assert result.structuredContent["found"] is True
        assert result.structuredContent["thought"]["thought_id"] == "seeded-1"

    async def test_query_memory_rejects_select_over_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "reject.db"
        await _seed_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "SELECT * FROM thought"},
            )

        assert result.isError is True
        assert "FIND" in result.content[0].text  # type: ignore[union-attr]

    async def test_memory_stats_reports_seeded_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "stats.db"
        await _seed_database(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.call_tool("memory_stats", {})

        assert result.structuredContent is not None
        assert result.structuredContent["thought_count"] == 1


class TestStoreResolution:
    """Tests for environment-driven store resolution."""

    async def test_db_path_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "resolve.db"
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        resolved = await resolve_store()
        assert isinstance(resolved, ResolvedStore)
        try:
            assert await resolved.store.count_thoughts() == 0
        finally:
            await resolved.aclose()
        assert db_path.exists()

    async def test_config_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "from_config.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(
            f"database:\n  path: {db_path.as_posix()}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)

        resolved = await resolve_store()
        try:
            assert await resolved.store.count_thoughts() == 0
        finally:
            await resolved.aclose()

    async def test_config_takes_priority_over_db_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_db = tmp_path / "config_priority.db"
        config_path = tmp_path / "priority.yaml"
        config_path.write_text(
            f"database:\n  path: {config_db.as_posix()}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ignored.db"))

        resolved = await resolve_store()
        await resolved.aclose()
        # The config path's database is the one that gets created.
        assert config_db.exists()
        assert not (tmp_path / "ignored.db").exists()

    async def test_no_configuration_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        with pytest.raises(StoreResolutionError):
            await resolve_store()

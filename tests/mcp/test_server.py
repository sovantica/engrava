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

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_TOOL_NAMES = frozenset(
    {"get_thought", "search_memory", "search_keywords", "query_memory", "memory_stats"}
)


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

    async def test_lists_exactly_five_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "tools.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_tools()

        assert {tool.name for tool in listed.tools} == EXPECTED_TOOL_NAMES
        assert all(
            tool.annotations is not None and tool.annotations.readOnlyHint for tool in listed.tools
        )

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

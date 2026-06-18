"""End-to-end tests for the MCP guided retrieval prompts.

Exercises the prompts through the in-memory MCP client transport (so
registration, argument-schema derivation, and message rendering all run
for real), mirroring the tool tests in :mod:`tests.mcp.test_server` and
the resource tests in :mod:`tests.mcp.test_resources`.

Prompts are read-oriented, so they are advertised in both the default and
the read-only deployment; the read-only cases below assert that
independence directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
from mcp.shared.memory import create_connected_server_and_client_session as connect_client
from mcp.types import TextContent

from engrava import (
    CoreThoughtRecord,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
)
from engrava.mcp import build_server
from engrava.mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR
from engrava.mcp.server import (
    DEFAULT_SUMMARY_LIMIT,
    READ_ONLY_ENV_VAR,
    _find_related_prompt,
    _reflect_on_topic_prompt,
    _summarize_recent_prompt,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

#: Names every prompt the server must advertise via ``list_prompts``.
EXPECTED_PROMPT_NAMES = frozenset({"summarize_recent_memory", "find_related", "reflect_on_topic"})


async def _seed_two_thoughts(path: Path) -> None:
    """Create a database file with two thoughts updated in a known order.

    The second thought carries the larger ``updated_cycle`` so it is the
    most recent, which lets the ``summarize_recent_memory`` prompt embed a
    deterministic newest-first snapshot.

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


def _single_text(result: object) -> str:
    """Return the text of a ``get_prompt`` result's single user message.

    Args:
        result: The ``GetPromptResult`` returned by ``get_prompt``.

    Returns:
        The text carried by the result's sole message, which must be a
        ``user``-role text block.

    """
    messages = result.messages  # type: ignore[attr-defined]
    assert len(messages) == 1
    message = messages[0]
    assert message.role == "user"
    content = message.content
    assert isinstance(content, TextContent)
    assert content.text
    return content.text


class TestPromptListing:
    """List prompts and their declared arguments through a client."""

    async def test_all_prompts_are_listed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "list.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_prompts()

        assert {prompt.name for prompt in listed.prompts} == EXPECTED_PROMPT_NAMES

    async def test_prompt_arguments_are_declared(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "args.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_prompts()

        required_by_name: dict[str, dict[str, bool]] = {}
        for prompt in listed.prompts:
            required_by_name[prompt.name] = {
                argument.name: bool(argument.required) for argument in (prompt.arguments or [])
            }

        # topic is required for both topic prompts; limit is optional.
        assert required_by_name["find_related"] == {"topic": True}
        assert required_by_name["reflect_on_topic"] == {"topic": True}
        assert required_by_name["summarize_recent_memory"] == {"limit": False}


class TestPromptRendering:
    """Render each prompt through a connected client."""

    async def test_find_related_reflects_topic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "related.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.get_prompt("find_related", {"topic": "pour-over coffee"})

        text = _single_text(result)
        assert "pour-over coffee" in text
        # The prompt must steer the model toward the search tool.
        assert "search_memory" in text

    async def test_reflect_on_topic_reflects_topic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "reflect.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.get_prompt("reflect_on_topic", {"topic": "green tea"})

        text = _single_text(result)
        assert "green tea" in text
        assert "search_memory" in text

    async def test_summarize_recent_uses_default_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "summary_default.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            result = await client.get_prompt("summarize_recent_memory", {})

        text = _single_text(result)
        # With no limit supplied the prompt falls back to the default.
        assert f"{DEFAULT_SUMMARY_LIMIT} most recently stored" in text
        assert "engrava://recent" in text
        # The embedded snapshot is read-only data drawn from the store.
        assert "newer-thought" in text

    async def test_summarize_recent_honours_explicit_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "summary_limit.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        server = build_server()
        async with connect_client(server) as client:
            # Arguments arrive as strings over the wire; FastMCP coerces
            # the declared ``int`` parameter.
            result = await client.get_prompt("summarize_recent_memory", {"limit": "2"})

        text = _single_text(result)
        assert "2 most recently stored" in text
        # limit=2 covers both seeded thoughts, so both appear in the snapshot.
        assert "newer-thought" in text
        assert "older-thought" in text


class TestPromptsInReadOnlyMode:
    """Prompts are reads, so they survive the write-tool gate."""

    async def test_prompts_listed_in_read_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "ro_list.db"))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            listed = await client.list_prompts()

        # Read-only mode hides the write tools but must not hide prompts.
        assert {prompt.name for prompt in listed.prompts} == EXPECTED_PROMPT_NAMES

    async def test_prompts_gettable_in_read_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "ro_get.db"
        await _seed_two_thoughts(db_path)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")

        server = build_server()
        async with connect_client(server) as client:
            summary = await client.get_prompt("summarize_recent_memory", {})
            related = await client.get_prompt("find_related", {"topic": "tea"})
            reflect = await client.get_prompt("reflect_on_topic", {"topic": "tea"})

        # Each prompt renders normally even with writes disabled.
        assert "newer-thought" in _single_text(summary)
        assert "tea" in _single_text(related)
        assert "tea" in _single_text(reflect)


class TestPromptTextBuilders:
    """Unit-cover the pure prompt-text builders directly.

    The end-to-end cases above always seed thoughts, so the builders'
    empty-store path is exercised here without standing up a server.
    """

    def test_summarize_recent_handles_empty_store(self) -> None:
        text = _summarize_recent_prompt(7, {"thoughts": [], "limit": 7})
        assert "7 most recently stored" in text
        assert "no thoughts to summarise" in text

    def test_summarize_recent_embeds_thoughts(self) -> None:
        recent = {"thoughts": [{"thought_id": "abc", "essence": "Note"}], "limit": 1}
        text = _summarize_recent_prompt(1, recent)
        assert "1 most recent thoughts" in text
        assert "abc" in text

    def test_find_related_builder_includes_topic_and_tool(self) -> None:
        text = _find_related_prompt("databases")
        assert "databases" in text
        assert "search_memory" in text

    def test_reflect_builder_includes_topic_and_tool(self) -> None:
        text = _reflect_on_topic_prompt("databases")
        assert "databases" in text
        assert "search_memory" in text

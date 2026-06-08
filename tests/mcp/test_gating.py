"""Tests for read-only gating of the MCP write tools.

The write tools must not be *registered* when the server is in read-only
mode, so a read-only deployment never advertises them to clients.  These
tests build a server under each mode and inspect the registered tool list,
and exercise the truthy parsing of :data:`READ_ONLY_ENV_VAR` directly.
"""

from __future__ import annotations

import pytest

from engrava.mcp.server import (
    READ_ONLY_ENV_VAR,
    _read_only_enabled,
    build_server,
)

#: Tool names that are always registered.
READ_TOOL_NAMES = frozenset(
    {
        "get_thought",
        "search_memory",
        "search_keywords",
        "query_memory",
        "memory_stats",
    }
)

#: Tool names that are gated behind write access.
WRITE_TOOL_NAMES = frozenset(
    {
        "store_thought",
        "update_thought",
        "link_thoughts",
    }
)


async def _registered_tool_names() -> set[str]:
    """Build a server and return the set of its registered tool names.

    Returns:
        The names of every tool the freshly built server advertises.

    """
    server = build_server()
    return {tool.name for tool in await server.list_tools()}


class TestRegistrationGating:
    """Tests for which tools register under each mode."""

    async def test_unset_registers_all_eight_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)
        names = await _registered_tool_names()
        assert names == READ_TOOL_NAMES | WRITE_TOOL_NAMES

    async def test_read_only_hides_write_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(READ_ONLY_ENV_VAR, "true")
        names = await _registered_tool_names()
        # Read tools stay; write tools are absent entirely.
        assert names >= READ_TOOL_NAMES
        assert names.isdisjoint(WRITE_TOOL_NAMES)
        assert names == READ_TOOL_NAMES


class TestTruthyParsing:
    """Tests for :func:`_read_only_enabled` truthy parsing."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes", " true "])
    def test_truthy_values_enable_read_only(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(READ_ONLY_ENV_VAR, value)
        assert _read_only_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  ", "maybe"])
    def test_falsy_values_keep_writes(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(READ_ONLY_ENV_VAR, value)
        assert _read_only_enabled() is False

    def test_unset_keeps_writes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)
        assert _read_only_enabled() is False

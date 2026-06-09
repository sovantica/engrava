"""Tests for the MCP tool error contract.

When a tool hits a known failure condition, the client must receive a
clean, typed, actionable error — a message with a helpful hint and
``isError`` set — rather than a raw Python traceback or an internal class
name.  These tests drive the real tool boundary through the in-process MCP
client transport (so FastMCP's error wrapping runs for real) and assert on
the message the client actually sees.

The conditions covered are:

* the store is not yet available (a misconfigured deployment),
* ``query_memory`` receives a non-``FIND`` command (``SELECT`` / ``COUNT``),
* ``query_memory`` receives a malformed ``FIND``,
* ``update_thought`` names a thought that does not exist,
* ``link_thoughts`` names an endpoint that does not exist.

Two cross-cutting properties are asserted in addition to per-condition
hints: the ``FIND``-only guard on ``query_memory`` is preserved (a
``SELECT`` is still rejected and the message never invites raw SQL), and no
error message leaks a filesystem path, a stack frame, or an internal symbol
name.

The server and client are built inside each test (rather than via a
yielding fixture) so the in-process transport's task-bound cancel scopes
enter and exit within the same task — the pattern the end-to-end server
tests already use.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava.mcp.server import SERVER_NAME, StoreProvider, register_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp import ClientSession

    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

#: Substrings that would indicate a leaked traceback or internal symbol.
#: Error messages shown to a client must contain none of them.
_LEAK_MARKERS = (
    "Traceback",
    'File "',
    "StoreNotReadyError",
    "UnsupportedQueryError",
    "MindQLParseError",
    "ThoughtNotFoundError",
    "ReferentialIntegrityError",
    "SqliteEngravaCore",
    "lifespan",
)

#: Phrases that would wrongly suggest raw SQL is runnable over the wire.
#: The ``FIND``-only rejection message must contain none of them.
_SQL_INVITATIONS = (
    "use select",
    "run select",
    "raw sql",
    "arbitrary sql",
    "try select",
    "select is",
    "select instead",
)


@asynccontextmanager
async def _client_for(store: SqliteEngravaCore) -> AsyncIterator[ClientSession]:
    """Open a connected client whose tools query the given store.

    Builds a server, points a :class:`StoreProvider` at ``store``, registers
    the tools, and connects the in-process client so the real tool boundary
    (and FastMCP's error wrapping) runs end to end.

    Args:
        store: The seeded store the tools should query.

    Yields:
        A connected client session wired to ``store``.

    """
    server: FastMCP = FastMCP(SERVER_NAME)
    provider = StoreProvider()
    provider.set(store)
    register_tools(server, provider)
    async with connect_client(server) as client:
        yield client


@asynccontextmanager
async def _store_less_client() -> AsyncIterator[ClientSession]:
    """Open a connected client whose provider never received a store.

    Registering the tools against an unpopulated :class:`StoreProvider`
    reproduces a deployment whose store has not been configured: the first
    tool call hits the store-not-ready condition at the real boundary.

    Yields:
        A connected client session backed by a store-less provider.

    """
    server: FastMCP = FastMCP(SERVER_NAME)
    register_tools(server, StoreProvider())
    async with connect_client(server) as client:
        yield client


def _error_text(content: object) -> str:
    """Extract the text of a tool error result's first content block.

    Args:
        content: The ``content`` sequence of a ``CallToolResult``.

    Returns:
        The ``text`` attribute of the first content block.

    """
    assert isinstance(content, list)
    assert content, "an error result must carry a content block"
    text = content[0].text  # type: ignore[union-attr]
    assert isinstance(text, str)
    return text


def _assert_no_leak(text: str) -> None:
    """Assert an error message leaks no path, stack frame, or symbol name.

    Args:
        text: The client-facing error message to inspect.

    """
    for marker in _LEAK_MARKERS:
        assert marker not in text, f"error message leaked {marker!r}: {text!r}"
    # No forward-slash path segment ...
    assert not re.search(r"/\w", text), f"error message leaked a '/' path: {text!r}"
    # ... and no backslash path segment.
    assert "\\" not in text, f"error message leaked a '\\' path: {text!r}"


class TestStoreNotReady:
    """The store-not-ready condition surfaces an actionable config hint."""

    async def test_reports_missing_store_with_env_var_hint(self) -> None:
        async with _store_less_client() as client:
            result = await client.call_tool("memory_stats", {})

        assert result.isError is True
        text = _error_text(result.content)
        # Actionable: it names the two documented configuration env vars ...
        assert "ENGRAVA_DB_PATH" in text
        assert "ENGRAVA_MCP_CONFIG" in text
        # ... and leaks no path, stack frame, or internal symbol.
        _assert_no_leak(text)


class TestUnsupportedQuery:
    """A non-``FIND`` ``query_memory`` is rejected with the FIND contract."""

    async def test_select_is_rejected_and_message_states_find_only(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "SELECT thought_id FROM thought"},
            )

        assert result.isError is True
        text = _error_text(result.content)
        # The guard still rejects the query and states the FIND-only contract.
        assert "FIND" in text
        assert "only FIND" in text
        _assert_no_leak(text)

    async def test_count_is_rejected_with_find_example(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "COUNT thoughts"})

        assert result.isError is True
        text = _error_text(result.content)
        assert "FIND" in text
        # A valid FIND example is offered to get the caller back on track.
        assert "FIND thoughts WHERE" in text
        _assert_no_leak(text)


class TestGuardPreservation:
    """The FIND-only guard must reject SELECT without ever inviting SQL."""

    async def test_select_rejection_does_not_invite_sql(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "SELECT * FROM thought WHERE 1=1"},
            )

        # The rejection itself is intact: a SELECT still fails.
        assert result.isError is True
        text = _error_text(result.content)
        lowered = text.lower()

        # The message asserts the FIND-only contract ...
        assert "find" in lowered
        assert "only find" in lowered
        # ... and must NOT suggest that raw SQL / SELECT is runnable.
        for invite in _SQL_INVITATIONS:
            assert invite not in lowered, f"message invited SQL via {invite!r}: {text!r}"
        # The only mention of SELECT permitted is echoing the rejected verb.
        # Stripping that quoted echo, no bare "SELECT" remains — so the
        # message never presents SELECT as a usable command.
        assert "SELECT" not in text.replace("'SELECT'", "")

    async def test_extension_command_is_also_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # A made-up verb parses as an unknown command and is rejected too —
        # the surface stays restricted to FIND, not just "not SELECT".
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "DROP thoughts"})

        assert result.isError is True
        text = _error_text(result.content)
        lowered = text.lower()
        assert "only find" in lowered
        _assert_no_leak(text)
        # The message must NOT leak the parser's full command set. For an
        # unrecognised verb the raw parser error reads "Expected FIND, COUNT,
        # SELECT, or extension command" — naming COUNT / SELECT / extension
        # would advertise commands the MCP surface deliberately hides. The
        # input verb ("DROP") is not echoed, so none of these may appear.
        assert "COUNT" not in text
        assert "SELECT" not in text
        assert "extension" not in lowered


class TestMalformedFind:
    """A malformed query is rejected with a FIND-only message + example.

    The message is deliberately generic (FIND-only + a valid example) and
    does NOT echo the parser's raw text, because the parser names the full
    MindQL command set for an unrecognised verb — which the MCP surface must
    not advertise. The trade-off (a malformed FIND loses the precise parse
    detail) is accepted to keep the over-the-wire surface FIND-only.
    """

    async def test_incomplete_find_reports_find_only_and_example(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool("query_memory", {"query": "FIND"})

        assert result.isError is True
        text = _error_text(result.content)
        # FIND-only contract is stated, with a valid FIND example to copy ...
        assert "only find" in text.lower()
        assert "FIND thoughts WHERE" in text
        # ... and the parser's command set is never leaked.
        assert "COUNT" not in text
        assert "extension" not in text.lower()
        _assert_no_leak(text)

    async def test_unknown_table_reports_problem(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "FIND widgets WHERE x = '1'"},
            )

        assert result.isError is True
        text = _error_text(result.content)
        assert "FIND thoughts WHERE" in text
        _assert_no_leak(text)


class TestUpdateMissingThought:
    """Updating an absent thought names the missing identifier."""

    async def test_missing_thought_reports_id_with_hint(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "update_thought",
                {"thought_id": "ghost-thought", "essence": "x"},
            )

        assert result.isError is True
        text = _error_text(result.content)
        # The offending id is echoed so the caller knows which one is wrong.
        assert "ghost-thought" in text
        # An actionable next step is offered.
        assert "search_memory" in text or "list_memory" in text
        _assert_no_leak(text)


class TestLinkMissingEndpoint:
    """Linking to an absent endpoint names the missing identifier."""

    async def test_missing_endpoint_reports_id_with_hint(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        async with _client_for(store) as client:
            result = await client.call_tool(
                "link_thoughts",
                {
                    "from_thought_id": "thought-alpha",
                    "to_thought_id": "ghost-endpoint",
                    "edge_type": "ASSOCIATED",
                },
            )

        assert result.isError is True
        text = _error_text(result.content)
        # The dangling endpoint id is echoed ...
        assert "ghost-endpoint" in text
        # ... and the message leaks no path, stack frame, or class name.
        _assert_no_leak(text)


class TestSuccessPathUnchanged:
    """Mapping errors must not alter what a successful tool call returns."""

    async def test_valid_find_still_succeeds(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # The seeded store has two ACTIVE thoughts; a valid FIND returns rows
        # with no error, confirming the wrapper is presentation-only.
        async with _client_for(store) as client:
            result = await client.call_tool(
                "query_memory",
                {"query": "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"},
            )

        assert result.isError is False
        assert result.structuredContent is not None
        assert "thought_id" in result.structuredContent["columns"]
        assert len(result.structuredContent["rows"]) == 2

    async def test_valid_keyword_search_still_succeeds(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # A well-formed call through a wrapped read tool returns its normal
        # payload unchanged — the error wrapper adds nothing on the happy path.
        async with _client_for(store) as client:
            result = await client.call_tool("search_keywords", {"query": "coffee"})

        assert result.isError is False
        assert result.structuredContent is not None
        assert "results" in result.structuredContent

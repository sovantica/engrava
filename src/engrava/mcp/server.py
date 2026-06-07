"""FastMCP server exposing engrava's read API as agent tools.

This module builds a Model Context Protocol server that wraps the public
async read API of :class:`~engrava.SqliteEngravaCore`.  It is an *API
consumer*, not an engrava extension: it registers no hooks, manifests, or
MindQL extension commands.  Think of it as a sibling of the command-line
interface that speaks MCP over stdio.

Five read-only tools are exposed:

``get_thought``
    Fetch a single thought by identifier.
``search_memory``
    Hybrid (lexical + vector + recency) ranked search.
``search_keywords``
    Pure full-text BM25 keyword search.
``query_memory``
    Structured ``FIND`` queries in the MindQL query language.  Only the
    ``FIND`` command is accepted; raw-SQL passthrough and every other
    command are rejected.
``memory_stats``
    Aggregate counts and store-health metrics.

The active store is supplied to tool calls through a :class:`StoreProvider`
that the server's lifespan populates on startup and clears on shutdown.
Each tool delegates to a module-level implementation function that takes an
explicit store argument, which keeps the query logic unit-testable without a
running server.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from engrava.mcp.config import ResolvedStore, resolve_store
from engrava.mindql.parser import MindQLCommand, MindQLQuery, parse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

#: Server name advertised to MCP clients.
SERVER_NAME = "engrava"

#: Default number of results returned by search tools.
DEFAULT_TOP_K = 10

_READ_ONLY = ToolAnnotations(readOnlyHint=True)


class StoreNotReadyError(RuntimeError):
    """Raised when a tool is invoked before a store has been provided.

    This indicates a lifecycle bug — tools should only run while the
    server lifespan is active.
    """


class UnsupportedQueryError(ValueError):
    """Raised when ``query_memory`` receives a non-``FIND`` command.

    The MCP read surface deliberately accepts only the MindQL ``FIND``
    command.  Raw-SQL passthrough (``SELECT``), aggregate ``COUNT``, and
    extension commands are rejected so the tool cannot be used to run
    arbitrary statements against the database.

    Args:
        command: The rejected command verb.

    """

    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__(
            f"query_memory accepts only FIND queries; received {command!r}. "
            "Use the FIND command, for example: "
            "FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10"
        )


class StoreProvider:
    """Holds the active store for the lifetime of a running server.

    The server lifespan calls :meth:`set` on startup and :meth:`clear`
    on shutdown.  Registered tools call :meth:`require` to obtain the
    store, which raises if the server is not currently serving.
    """

    def __init__(self) -> None:
        self._store: SqliteEngravaCore | None = None

    def set(self, store: SqliteEngravaCore) -> None:
        """Record the active store.

        Args:
            store: The store that tools should query.

        """
        self._store = store

    def clear(self) -> None:
        """Forget the active store after shutdown."""
        self._store = None

    def require(self) -> SqliteEngravaCore:
        """Return the active store.

        Returns:
            The store recorded by the lifespan.

        Raises:
            StoreNotReadyError: If no store is currently active.

        """
        if self._store is None:
            msg = "No active engrava store; the server lifespan is not running."
            raise StoreNotReadyError(msg)
        return self._store


async def get_thought_impl(store: SqliteEngravaCore, thought_id: str) -> dict[str, Any]:
    """Fetch a single thought by identifier.

    Args:
        store: The store to query.
        thought_id: Identifier of the thought to retrieve.

    Returns:
        A dict with a ``found`` flag and a ``thought`` entry.  ``thought``
        is the JSON-serialisable thought when it exists, otherwise
        ``None``.

    """
    thought = await store.get_thought(thought_id)
    if thought is None:
        return {"found": False, "thought": None}
    return {"found": True, "thought": thought.model_dump(mode="json")}


async def search_memory_impl(
    store: SqliteEngravaCore,
    query_text: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    include_reflections: bool = True,
) -> dict[str, Any]:
    """Run a hybrid ranked search over stored memory.

    Args:
        store: The store to query.
        query_text: Natural-language query text.
        top_k: Maximum number of ranked results to return.
        include_reflections: Whether consolidated reflection thoughts may
            appear in the results.

    Returns:
        A dict with a ``results`` list of ``{"thought_id", "score"}``
        entries and a ``backends_used`` list naming the search backends
        that were available for the query.

    """
    result = await store.search_hybrid(
        query_text,
        top_k=top_k,
        include_reflections=include_reflections,
    )
    return {
        "results": [
            {"thought_id": thought_id, "score": score} for thought_id, score in result.results
        ],
        "backends_used": sorted(result.backends_used),
    }


async def search_keywords_impl(
    store: SqliteEngravaCore,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run a full-text BM25 keyword search over stored memory.

    Args:
        store: The store to query.
        query: Full-text query string (supports ``AND``, ``OR``, ``NOT``
            and prefix ``*`` operators).
        top_k: Maximum number of ranked results to return.

    Returns:
        A dict with a ``results`` list of ``{"thought_id", "score"}``
        entries ordered by descending relevance.

    """
    matches = await store.search_fts(query, top_k=top_k)
    return {
        "results": [{"thought_id": thought_id, "score": score} for thought_id, score in matches],
    }


async def query_memory_impl(
    store: SqliteEngravaCore,
    query: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a MindQL ``FIND`` query over stored memory.

    Only the ``FIND`` command is accepted.  The grammar is
    ``FIND <table> WHERE <field> <op> '<value>' [LIMIT n]``.

    Args:
        store: The store to query.
        query: A MindQL ``FIND`` query string.
        limit: Optional row cap.  When provided, it overrides any
            ``LIMIT`` clause present in ``query``.

    Returns:
        A dict with the result ``columns`` and matching ``rows``.

    Raises:
        UnsupportedQueryError: If the query is not a ``FIND`` command.
        MindQLParseError: If the query is malformed.

    """
    parsed = parse(query)
    if parsed.command is not MindQLCommand.FIND:
        raise UnsupportedQueryError(parsed.command.value)

    effective = parsed if limit is None else _with_limit(parsed, limit)

    # Execute via the public store-level entry point. The store owns the
    # connection; this consumer must not reach into it. The FIND-only guard
    # above is intentionally kept here (a consumer exposure policy), and no
    # ``extensions`` map is passed — both keep the over-the-wire surface
    # restricted to FIND.
    result = await store.execute_mindql(effective)
    return {"columns": result.columns, "rows": result.rows}


async def memory_stats_impl(store: SqliteEngravaCore) -> dict[str, Any]:
    """Return aggregate counts and store-health metrics.

    Args:
        store: The store to inspect.

    Returns:
        A dict with the live ``thought_count`` plus a ``metrics`` block
        carrying thought/edge counts and a storage-byte total.

    """
    thought_count = await store.count_thoughts()
    metrics = await store.metrics()
    return {
        "thought_count": thought_count,
        "metrics": {
            "thoughts": {
                "total": metrics.thoughts.total,
                "by_type": metrics.thoughts.by_type,
                "by_status": metrics.thoughts.by_status,
            },
            "edges": {
                "total": metrics.edges.total,
                "by_type": metrics.edges.by_type,
            },
            "storage_total_bytes": metrics.storage.total_bytes,
        },
    }


def _with_limit(parsed: MindQLQuery, limit: int) -> MindQLQuery:
    """Return a copy of a parsed query with its ``limit`` replaced.

    Args:
        parsed: The parsed ``MindQLQuery``.
        limit: The row cap to apply.

    Returns:
        A new ``MindQLQuery`` identical to ``parsed`` but with ``limit``
        set to the supplied value.

    """
    return replace(parsed, limit=limit)


def build_server() -> FastMCP:
    """Build the engrava MCP server with all read tools registered.

    The returned server resolves its store from the environment when its
    lifespan starts and releases the connection when the lifespan ends.

    Returns:
        A configured :class:`FastMCP` server ready to ``run()``.

    """
    provider = StoreProvider()

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
        resolved: ResolvedStore = await resolve_store()
        provider.set(resolved.store)
        try:
            yield
        finally:
            provider.clear()
            # Shield the connection teardown so it runs to completion even
            # when the surrounding server task is being cancelled (as it is
            # on stdio EOF).  Without the shield the database worker thread
            # can outlive the event loop and raise on a late callback.
            with anyio.CancelScope(shield=True):
                await resolved.aclose()

    server: FastMCP = FastMCP(
        SERVER_NAME,
        instructions=(
            "Read-only access to an engrava agent-memory store: fetch "
            "thoughts, run hybrid and keyword search, run structured "
            "MindQL FIND queries, and read store statistics."
        ),
        lifespan=lifespan,
    )
    register_tools(server, provider)
    return server


def register_tools(server: FastMCP, provider: StoreProvider) -> None:
    """Register the five read tools on a server.

    Args:
        server: The server to register tools on.
        provider: Supplies the active store to each tool at call time.

    """

    @server.tool(
        name="get_thought",
        description="Fetch a single thought by its identifier.",
        annotations=_READ_ONLY,
    )
    async def get_thought(thought_id: str) -> dict[str, Any]:
        return await get_thought_impl(provider.require(), thought_id)

    @server.tool(
        name="search_memory",
        description=(
            "Hybrid ranked search (lexical + vector + recency) over stored "
            "memory. Returns ranked thought identifiers with scores and the "
            "search backends that were available."
        ),
        annotations=_READ_ONLY,
    )
    async def search_memory(
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        include_reflections: bool = True,
    ) -> dict[str, Any]:
        return await search_memory_impl(
            provider.require(),
            query_text,
            top_k=top_k,
            include_reflections=include_reflections,
        )

    @server.tool(
        name="search_keywords",
        description=(
            "Full-text BM25 keyword search over stored memory. Returns ranked "
            "thought identifiers with scores."
        ),
        annotations=_READ_ONLY,
    )
    async def search_keywords(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        return await search_keywords_impl(provider.require(), query, top_k=top_k)

    @server.tool(
        name="query_memory",
        description=(
            "Run a structured MindQL FIND query over stored memory, e.g. "
            "\"FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10\". "
            "Only the FIND command is supported."
        ),
        annotations=_READ_ONLY,
    )
    async def query_memory(query: str, limit: int | None = None) -> dict[str, Any]:
        return await query_memory_impl(provider.require(), query, limit=limit)

    @server.tool(
        name="memory_stats",
        description=(
            "Return aggregate statistics about the memory store: thought and "
            "edge counts and total storage size."
        ),
        annotations=_READ_ONLY,
    )
    async def memory_stats() -> dict[str, Any]:
        return await memory_stats_impl(provider.require())


def main() -> None:
    """Run the engrava MCP server over stdio.

    Builds the server and serves it on the stdio transport (the FastMCP
    default).  This is the console-script and ``python -m engrava.mcp``
    entry point.
    """
    build_server().run()

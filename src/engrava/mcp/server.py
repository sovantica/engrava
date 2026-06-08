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

Three write tools complete the surface:

``store_thought``
    Create a new thought node.
``update_thought``
    Mutate selected fields of an existing thought.
``link_thoughts``
    Create a typed edge between two existing thoughts.

The write tools are gated by the :data:`READ_ONLY_ENV_VAR` environment
variable.  When it is set to a truthy value the write tools are not
registered at all, so a read-only deployment never advertises them to
clients.  The read tools are always available.

The active store is supplied to tool calls through a :class:`StoreProvider`
that the server's lifespan populates on startup and clears on shutdown.
Each tool delegates to a module-level implementation function that takes an
explicit store argument, which keeps the query and mutation logic
unit-testable without a running server.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from engrava.domain.enums import EdgeType, LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.mcp.config import ResolvedStore, resolve_store
from engrava.mindql.parser import MindQLCommand, MindQLQuery, parse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

#: Server name advertised to MCP clients.
SERVER_NAME = "engrava"

#: Default number of results returned by search tools.
DEFAULT_TOP_K = 10

#: Default edge weight when a caller does not supply one.
DEFAULT_EDGE_WEIGHT = 1.0

#: Cycle counter assigned to thoughts and edges created through the MCP
#: write surface.  This API consumer has no notion of a cognitive cycle
#: clock, so new records start at the origin cycle.
INITIAL_CYCLE = 0

#: Environment variable that, when truthy, suppresses registration of the
#: write tools so the server exposes a read-only surface.
READ_ONLY_ENV_VAR = "ENGRAVA_MCP_READ_ONLY"

#: Values that enable read-only mode (compared case-insensitively after
#: stripping surrounding whitespace).  Any other value — including unset
#: or empty — leaves the full read and write surface enabled.
READ_ONLY_TRUTHY_VALUES = frozenset({"1", "true", "yes"})

_READ_ONLY = ToolAnnotations(readOnlyHint=True)

#: Annotation for a non-idempotent, non-destructive write.  Covers both
#: creating a new thought node (repeating the call creates another node)
#: and creating a typed edge (an edge is unique per source/target/type, so
#: repeating an identical link is rejected rather than converging) — neither
#: is safe for a client to blindly retry.
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)

#: Annotation for an idempotent, non-destructive write (updating a thought —
#: repeating with the same arguments converges on the same end state).
_WRITE_IDEMPOTENT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


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


async def store_thought_impl(
    store: SqliteEngravaCore,
    essence: str,
    content: str,
    *,
    thought_type: ThoughtType = ThoughtType.NOTE,
    priority: Priority = Priority.P3,
    source: str = "agent",
    confidence: float | None = None,
    thought_id: str | None = None,
    deduplicate: bool = False,
) -> dict[str, Any]:
    """Create a new thought node in the store.

    A :class:`~engrava.ThoughtRecord` is constructed from the supplied
    fields and persisted.  The remaining record fields take their model
    defaults.  New thoughts start in the ``CREATED`` lifecycle state at
    the origin cycle.

    Args:
        store: The store to write to.
        essence: Compact canonical text used in prompts (1-200 chars).
        content: Full stored content (non-empty).
        thought_type: Classification of the thought content.
        priority: Urgency level (``P1`` highest).
        source: Origin label for the thought (e.g. ``"agent"``, ``"human"``).
        confidence: Optional reliability estimate in ``[0.0, 1.0]``.
        thought_id: Optional caller-supplied identifier.  When omitted a
            fresh UUID4 is generated.
        deduplicate: When ``True``, an existing thought whose content hash
            matches has its confirmation count incremented and is returned
            instead of inserting a duplicate.

    Returns:
        A dict with a ``thought`` entry carrying the persisted thought's
        ``thought_id``, ``essence``, ``thought_type``, ``priority`` and
        ``lifecycle_status``.  When deduplication collapses onto an
        existing record, its identifier is returned.

    """
    record = ThoughtRecord(
        thought_id=thought_id if thought_id is not None else str(uuid.uuid4()),
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=priority,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=INITIAL_CYCLE,
        updated_cycle=INITIAL_CYCLE,
        source=source,
        confidence=confidence,
    )
    created = await store.create_thought(record, deduplicate=deduplicate)
    return {
        "thought": {
            "thought_id": created.thought_id,
            "essence": created.essence,
            "thought_type": created.thought_type.value,
            "priority": created.priority.value,
            "lifecycle_status": created.lifecycle_status.value,
        }
    }


async def update_thought_impl(
    store: SqliteEngravaCore,
    thought_id: str,
    *,
    essence: str | None = None,
    content: str | None = None,
    priority: Priority | None = None,
    lifecycle_status: LifecycleStatus | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Update selected fields of an existing thought.

    Only the fields the caller supplies are changed; every omitted
    argument leaves its stored value untouched.  Field changes are
    applied with the store's optimistic-concurrency guard.

    Args:
        store: The store to write to.
        thought_id: Identifier of the thought to update.
        essence: New compact canonical text, if changing.
        content: New full content, if changing.
        priority: New urgency level, if changing.
        lifecycle_status: New lifecycle state, if changing.  The store
            validates that the transition is allowed.
        confidence: New reliability estimate in ``[0.0, 1.0]``, if changing.

    Returns:
        A dict with a ``thought`` entry carrying the updated thought's
        ``thought_id``, ``essence``, ``priority`` and ``lifecycle_status``.

    Raises:
        ThoughtNotFoundError: If no thought has the given identifier.
        StaleDataError: If the thought changed concurrently.
        InvalidTransitionError: If a lifecycle change is not permitted.

    """
    changes: dict[str, object] = {}
    if essence is not None:
        changes["essence"] = essence
    if content is not None:
        changes["content"] = content
    if priority is not None:
        changes["priority"] = priority
    if lifecycle_status is not None:
        changes["lifecycle_status"] = lifecycle_status
    if confidence is not None:
        changes["confidence"] = confidence

    updated = await store.update_thought(thought_id, **changes)
    return {
        "thought": {
            "thought_id": updated.thought_id,
            "essence": updated.essence,
            "priority": updated.priority.value,
            "lifecycle_status": updated.lifecycle_status.value,
        }
    }


async def link_thoughts_impl(
    store: SqliteEngravaCore,
    from_thought_id: str,
    to_thought_id: str,
    edge_type: EdgeType,
    *,
    weight: float = DEFAULT_EDGE_WEIGHT,
    edge_id: str | None = None,
) -> dict[str, Any]:
    """Create a typed edge between two existing thoughts.

    An :class:`~engrava.EdgeRecord` is constructed from the supplied
    endpoints and persisted.  Both endpoints must already exist.

    Args:
        store: The store to write to.
        from_thought_id: Identifier of the source thought.
        to_thought_id: Identifier of the target thought.
        edge_type: Classification of the relationship.
        weight: Relation strength in ``[0.0, 1.0]``.
        edge_id: Optional caller-supplied identifier.  When omitted a
            fresh UUID4 is generated.

    Returns:
        A dict with an ``edge`` entry carrying the persisted edge's
        ``edge_id``, ``from_thought_id``, ``to_thought_id``, ``edge_type``
        and ``weight``.

    Raises:
        ReferentialIntegrityError: If either endpoint does not exist.
        IntegrityError: If an edge with the same source, target and type
            already exists.  Edges are unique per ``(from, to, type)``, so
            this write is not idempotent — repeating an identical link is
            rejected rather than ignored.

    """
    record = EdgeRecord(
        edge_id=edge_id if edge_id is not None else str(uuid.uuid4()),
        from_thought_id=from_thought_id,
        to_thought_id=to_thought_id,
        edge_type=edge_type,
        weight=weight,
        created_cycle=INITIAL_CYCLE,
    )
    created = await store.create_edge(record)
    return {
        "edge": {
            "edge_id": created.edge_id,
            "from_thought_id": created.from_thought_id,
            "to_thought_id": created.to_thought_id,
            "edge_type": created.edge_type.value,
            "weight": created.weight,
        }
    }


def _read_only_enabled() -> bool:
    """Report whether the server should expose a read-only surface.

    Reads :data:`READ_ONLY_ENV_VAR` and compares it against
    :data:`READ_ONLY_TRUTHY_VALUES` after stripping surrounding whitespace
    and lower-casing.  An unset or empty value is treated as not
    read-only.

    Returns:
        ``True`` when the environment requests a read-only surface,
        otherwise ``False``.

    """
    raw = os.environ.get(READ_ONLY_ENV_VAR, "")
    return raw.strip().lower() in READ_ONLY_TRUTHY_VALUES


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
    """Build the engrava MCP server with its tools registered.

    The returned server resolves its store from the environment when its
    lifespan starts and releases the connection when the lifespan ends.
    The read tools are always registered; the write tools are registered
    unless :func:`_read_only_enabled` reports a read-only deployment.

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
            "Access to an engrava agent-memory store: fetch thoughts, run "
            "hybrid and keyword search, run structured MindQL FIND queries, "
            "and read store statistics. Unless the server is started in "
            "read-only mode, you can also store new thoughts, update existing "
            "thoughts, and link thoughts with typed edges."
        ),
        lifespan=lifespan,
    )
    register_tools(server, provider)
    return server


def register_tools(server: FastMCP, provider: StoreProvider) -> None:
    """Register the MCP tools on a server.

    The five read tools are always registered.  The three write tools are
    registered only when the server is not in read-only mode (see
    :func:`_read_only_enabled`); in read-only mode they are never
    advertised to clients.

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

    if _read_only_enabled():
        return

    @server.tool(
        name="store_thought",
        description=(
            "Create a new thought node. Provide its essence (short canonical "
            "text) and full content; optionally set the thought type, "
            "priority, source, and confidence. Returns the created thought's "
            "identifier and key fields."
        ),
        annotations=_WRITE,
    )
    async def store_thought(
        essence: str,
        content: str,
        thought_type: ThoughtType = ThoughtType.NOTE,
        priority: Priority = Priority.P3,
        source: str = "agent",
        *,
        confidence: float | None = None,
        thought_id: str | None = None,
        deduplicate: bool = False,
    ) -> dict[str, Any]:
        return await store_thought_impl(
            provider.require(),
            essence,
            content,
            thought_type=thought_type,
            priority=priority,
            source=source,
            confidence=confidence,
            thought_id=thought_id,
            deduplicate=deduplicate,
        )

    @server.tool(
        name="update_thought",
        description=(
            "Update fields of an existing thought by identifier. Only the "
            "fields you supply change; omit the rest. Can change essence, "
            "content, priority, lifecycle status, and confidence."
        ),
        annotations=_WRITE_IDEMPOTENT,
    )
    async def update_thought(
        thought_id: str,
        essence: str | None = None,
        content: str | None = None,
        priority: Priority | None = None,
        lifecycle_status: LifecycleStatus | None = None,
        *,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        return await update_thought_impl(
            provider.require(),
            thought_id,
            essence=essence,
            content=content,
            priority=priority,
            lifecycle_status=lifecycle_status,
            confidence=confidence,
        )

    @server.tool(
        name="link_thoughts",
        description=(
            "Create a typed edge between two existing thoughts, identified by "
            "their identifiers. Choose the edge type and optionally a weight "
            "in [0.0, 1.0]. Both endpoints must already exist. An edge is "
            "unique per (source, target, type): linking the same pair with the "
            "same type twice is rejected rather than ignored."
        ),
        annotations=_WRITE,
    )
    async def link_thoughts(
        from_thought_id: str,
        to_thought_id: str,
        edge_type: EdgeType,
        weight: float = DEFAULT_EDGE_WEIGHT,
        *,
        edge_id: str | None = None,
    ) -> dict[str, Any]:
        return await link_thoughts_impl(
            provider.require(),
            from_thought_id,
            to_thought_id,
            edge_type,
            weight=weight,
            edge_id=edge_id,
        )


def main() -> None:
    """Run the engrava MCP server over stdio.

    Builds the server and serves it on the stdio transport (the FastMCP
    default).  This is the console-script and ``python -m engrava.mcp``
    entry point.
    """
    build_server().run()

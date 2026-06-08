"""FastMCP server exposing engrava's read API as agent tools.

This module builds a Model Context Protocol server that wraps the public
async read API of :class:`~engrava.SqliteEngravaCore`.  It is an *API
consumer*, not an engrava extension: it registers no hooks, manifests, or
MindQL extension commands.  Think of it as a sibling of the command-line
interface that speaks MCP over stdio.

Six read-only tools are exposed:

``get_thought``
    Fetch a single thought by identifier.
``search_memory``
    Hybrid (lexical + vector + recency) ranked search.  Optional
    ``thought_type`` / ``lifecycle_status`` / ``priority`` filters narrow
    the ranked hits *after* ranking (the hybrid ranker cannot filter), so
    a filtered call may return fewer than ``top_k`` results and reports
    how many ranked hits it dropped.
``search_keywords``
    Pure full-text BM25 keyword search.
``list_memory``
    Deterministic, unranked browse over stored thoughts with the full
    filter matrix (``thought_type``, ``lifecycle_status``, ``priority``,
    updated-cycle range) and ``limit`` / ``offset`` pagination.  Returns
    thoughts newest-first with no score — the clean home for "list memory
    by structured field", complementing the ranked ``search_memory``.
``query_memory``
    Structured ``FIND`` queries in the MindQL query language.  Only the
    ``FIND`` command is accepted; raw-SQL passthrough and every other
    command are rejected.  Accepts an optional ``limit`` (the MindQL
    grammar has no ``OFFSET``, so this tool paginates by ``limit`` only).
``memory_stats``
    Aggregate counts and store-health metrics.

Five write tools complete the surface:

``store_thought``
    Create a new thought node.
``update_thought``
    Mutate selected fields of an existing thought.
``link_thoughts``
    Create a typed edge between two existing thoughts.
``delete_thought``
    Remove a thought by identifier.
``delete_edge``
    Remove an edge by identifier.

The write tools are gated by the :data:`READ_ONLY_ENV_VAR` environment
variable.  When it is set to a truthy value the write tools are not
registered at all, so a read-only deployment never advertises them to
clients.  The read tools are always available.

Three read-only *resources* round out the surface.  Where tools are
*invoked*, resources are addressable ``engrava://`` URIs that clients
surface as attachable context:

``engrava://thought/{thought_id}``
    A single thought as a JSON document.  Reading an unknown identifier
    yields a graceful not-found payload rather than an error.
``engrava://stats``
    Store-health counts and size, identical to the ``memory_stats`` tool
    (both share :func:`memory_stats_impl`).
``engrava://recent``
    The most-recently-updated thoughts as a JSON document.

Resources are reads by definition, so — unlike the write tools — they are
*not* gated by :data:`READ_ONLY_ENV_VAR`; they are advertised in both the
default and read-only deployments.

Three *prompts* complete the surface.  Prompts are parameterised templates
that a client surfaces as slash-commands or buttons; each one renders a
ready-to-send instruction that guides the assistant to gather context with
the read tools and resources above.  They are templates only — they open no
write path and call no store method:

``summarize_recent_memory``
    Summarise the most recently stored thoughts.  Takes an optional
    ``limit`` (how many recent thoughts to consider).
``find_related``
    Find and synthesise thoughts related to a required ``topic``.
``reflect_on_topic``
    Reflect over what memory holds about a required ``topic``.

Prompts are read-oriented, so — like the resources — they are *not* gated by
:data:`READ_ONLY_ENV_VAR` and are advertised in both deployments.

The active store is supplied to tool and resource calls through a
:class:`StoreProvider` that the server's lifespan populates on startup and
clears on shutdown.  Each tool delegates to a module-level implementation
function that takes an explicit store argument, which keeps the query and
mutation logic unit-testable without a running server.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from engrava.domain.enums import EdgeType, LifecycleStatus, Priority, ThoughtType
from engrava.domain.exceptions import ReferentialIntegrityError, ThoughtNotFoundError
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.mcp.config import ResolvedStore, resolve_store
from engrava.mindql.parser import MindQLCommand, MindQLParseError, MindQLQuery, parse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

#: Server name advertised to MCP clients.
SERVER_NAME = "engrava"

#: Default number of results returned by search tools.
DEFAULT_TOP_K = 10

#: Default number of thoughts returned by the ``engrava://recent`` resource.
DEFAULT_RECENT_LIMIT = 10

#: Default page size for the ``list_memory`` browse tool.  Matches the
#: store's own ``list_thoughts`` default so an unpaged listing behaves the
#: same whether driven through MCP or the core API directly.
DEFAULT_LIST_LIMIT = 50

#: Default number of recent thoughts the ``summarize_recent_memory`` prompt
#: asks the assistant to consider when the caller omits ``limit``.  Kept
#: small so the summary stays focused on the latest activity.
DEFAULT_SUMMARY_LIMIT = 5

#: MIME type advertised for every ``engrava://`` resource.  Resource
#: handlers return a JSON document as text, so clients receive a stable,
#: machine-parseable content type.
RESOURCE_MIME_TYPE = "application/json"

#: Default edge weight when a caller does not supply one.
DEFAULT_EDGE_WEIGHT = 1.0

#: Cycle counter assigned to thoughts and edges created through the MCP
#: write surface.  This API consumer has no notion of a cognitive cycle
#: clock, so new records start at the origin cycle.
INITIAL_CYCLE = 0

#: A valid MindQL ``FIND`` query, embedded verbatim in the actionable hints
#: that ``query_memory`` returns when a caller sends a malformed or
#: unsupported query.  Showing one correct example is the fastest way to get
#: a client back onto the supported path; it deliberately demonstrates only
#: the ``FIND`` command, never raw SQL.
FIND_QUERY_EXAMPLE = "FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10"

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

#: Annotation for a destructive but idempotent write (deleting a thought or
#: edge).  It is marked idempotent because deleting an already-absent
#: identifier is a no-op that returns ``deleted=False`` and leaves the same
#: end state — the record is gone either way — so a client may safely retry a
#: delete that appeared to fail.
_WRITE_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True)


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
            f"Use the FIND command, for example: {FIND_QUERY_EXAMPLE}"
        )


@asynccontextmanager
async def _tool_errors() -> AsyncIterator[None]:
    """Translate known typed failures into clean, actionable MCP errors.

    Wraps the body of a tool handler so that the typed exceptions raised by
    the store, the MindQL parser, and this module's own consumer-policy
    guard surface to the client as a :class:`ToolError` carrying a curated,
    agent-facing message instead of an internal exception.  FastMCP reports
    a :class:`ToolError` to the client with ``isError`` set and the message
    as text, so the client receives an actionable hint rather than a raw
    traceback or an internal class name.

    This is *presentation only*: it adds no new capability and relaxes no
    guard.  Each branch re-raises an existing failure with a better message;
    the ``UnsupportedQueryError`` branch in particular preserves the
    ``FIND``-only contract verbatim and never suggests that raw SQL is
    runnable over the wire.  Conditions this module does not recognise are
    left to propagate unchanged.

    The messages name only the documented configuration *environment
    variables* (never a filesystem path), carry no stack frames, and expose
    no internal symbol names, so a misuse reply leaks nothing about the
    deployment.

    Yields:
        ``None``; the caller runs the guarded tool body inside the ``with``.

    Raises:
        ToolError: With an actionable message when a recognised typed
            failure occurs while the body runs.

    """
    try:
        yield
    except StoreNotReadyError as exc:
        msg = (
            "The engrava memory store is not available yet. Start the server "
            "with a store configured: set ENGRAVA_DB_PATH to a database file, "
            "or point ENGRAVA_MCP_CONFIG at an engrava.yaml that names one."
        )
        raise ToolError(msg) from exc
    except UnsupportedQueryError as exc:
        # The exception text already states the FIND-only contract and shows
        # a valid FIND example; echoing it keeps the guard's wording intact
        # and never invites raw SQL.
        raise ToolError(str(exc)) from exc
    except MindQLParseError as exc:
        msg = (
            f"That query could not be parsed: {exc}. query_memory runs MindQL "
            f"FIND queries; for example: {FIND_QUERY_EXAMPLE}"
        )
        raise ToolError(msg) from exc
    except ThoughtNotFoundError as exc:
        msg = (
            f"No thought exists with id {exc.thought_id!r}. Check the "
            "identifier, or use search_memory or list_memory to find it."
        )
        raise ToolError(msg) from exc
    except ReferentialIntegrityError as exc:
        msg = (
            f"Cannot link thoughts: no thought exists with id "
            f"{exc.referenced_id!r}. Create that thought first, or correct "
            "the identifier."
        )
        raise ToolError(msg) from exc


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


def _filter_criteria(
    *,
    thought_type: ThoughtType | None,
    lifecycle_status: LifecycleStatus | None,
    priority: Priority | None,
) -> dict[str, str]:
    """Collect the active thought filters as a JSON-friendly mapping.

    Only the filters the caller actually supplied appear in the result;
    each enum is reduced to its string value so the mapping serialises
    cleanly into a tool response.

    Args:
        thought_type: Thought-type filter, or ``None`` if not filtering.
        lifecycle_status: Lifecycle-status filter, or ``None``.
        priority: Priority filter, or ``None``.

    Returns:
        A dict mapping each supplied filter's field name to its string
        value.  Empty when no filter was supplied.

    """
    criteria: dict[str, str] = {}
    if thought_type is not None:
        criteria["thought_type"] = thought_type.value
    if lifecycle_status is not None:
        criteria["lifecycle_status"] = lifecycle_status.value
    if priority is not None:
        criteria["priority"] = priority.value
    return criteria


def _thought_matches(
    thought: ThoughtRecord,
    *,
    thought_type: ThoughtType | None,
    lifecycle_status: LifecycleStatus | None,
    priority: Priority | None,
) -> bool:
    """Report whether a thought satisfies every supplied filter.

    A ``None`` filter is not applied, so a thought matches when it equals
    each filter that *was* supplied (logical AND).  With no filters
    supplied this trivially returns ``True``.

    Args:
        thought: The thought record to test.
        thought_type: Required thought type, or ``None`` to ignore.
        lifecycle_status: Required lifecycle state, or ``None`` to ignore.
        priority: Required priority level, or ``None`` to ignore.

    Returns:
        ``True`` when the thought matches every supplied filter.

    """
    if thought_type is not None and thought.thought_type is not thought_type:
        return False
    if lifecycle_status is not None and thought.lifecycle_status is not lifecycle_status:
        return False
    return not (priority is not None and thought.priority is not priority)


async def search_memory_impl(
    store: SqliteEngravaCore,
    query_text: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    include_reflections: bool = True,
    thought_type: ThoughtType | None = None,
    lifecycle_status: LifecycleStatus | None = None,
    priority: Priority | None = None,
) -> dict[str, Any]:
    """Run a hybrid ranked search over stored memory.

    The hybrid ranker itself does not filter by type, status, or
    priority, so any of those filters are applied *after* ranking: the
    ranked hits are fetched and the ones that do not match every supplied
    filter are dropped.  Ranking order is preserved and scores are never
    altered or fabricated — a filtered response simply carries fewer
    entries than ``top_k`` and reports how many were dropped (see the
    ``filtered`` block below) so the caller is never misled into reading
    an empty or short list as "nothing was found".

    Args:
        store: The store to query.
        query_text: Natural-language query text.
        top_k: Maximum number of ranked results to consider.  Filters are
            applied to this ranked window, so the returned list may be
            shorter when filters drop hits.
        include_reflections: Whether consolidated reflection thoughts may
            appear in the results.
        thought_type: When set, keep only hits of this type.
        lifecycle_status: When set, keep only hits in this lifecycle state.
        priority: When set, keep only hits at this priority level.

    Returns:
        A dict with a ``results`` list of ``{"thought_id", "score"}``
        entries (ranking order preserved) and a ``backends_used`` list
        naming the search backends that were available for the query.
        When at least one filter is supplied, a ``filtered`` block is
        added carrying the active ``criteria`` and the ``scanned`` /
        ``matched`` / ``dropped`` counts over the ranked window, so a
        short or empty list is never mistaken for "no hits ranked".

    """
    result = await store.search_hybrid(
        query_text,
        top_k=top_k,
        include_reflections=include_reflections,
    )
    backends_used = sorted(result.backends_used)

    criteria = _filter_criteria(
        thought_type=thought_type,
        lifecycle_status=lifecycle_status,
        priority=priority,
    )
    if not criteria:
        # Unfiltered path: byte-for-byte the original response shape.
        return {
            "results": [
                {"thought_id": thought_id, "score": score} for thought_id, score in result.results
            ],
            "backends_used": backends_used,
        }

    kept: list[dict[str, Any]] = []
    for thought_id, score in result.results:
        thought = await store.get_thought(thought_id)
        if thought is not None and _thought_matches(
            thought,
            thought_type=thought_type,
            lifecycle_status=lifecycle_status,
            priority=priority,
        ):
            kept.append({"thought_id": thought_id, "score": score})

    scanned = len(result.results)
    return {
        "results": kept,
        "backends_used": backends_used,
        "filtered": {
            "criteria": criteria,
            "scanned": scanned,
            "matched": len(kept),
            "dropped": scanned - len(kept),
        },
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


async def recent_thoughts_impl(
    store: SqliteEngravaCore,
    *,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> dict[str, Any]:
    """Return the most-recently-updated thoughts.

    Wraps the public :meth:`~engrava.SqliteEngravaCore.list_thoughts`,
    which orders by descending ``updated_cycle`` — so the first entry is
    the thought touched most recently.

    Args:
        store: The store to query.
        limit: Maximum number of thoughts to return, newest first.

    Returns:
        A dict with a ``thoughts`` list of JSON-serialisable thoughts
        (newest first) and the ``limit`` that was applied.

    """
    thoughts = await store.list_thoughts(limit=limit)
    return {
        "thoughts": [thought.model_dump(mode="json") for thought in thoughts],
        "limit": limit,
    }


async def list_memory_impl(
    store: SqliteEngravaCore,
    *,
    thought_type: ThoughtType | None = None,
    lifecycle_status: LifecycleStatus | None = None,
    priority: Priority | None = None,
    min_cycle: int | None = None,
    max_cycle: int | None = None,
    include_expired: bool = False,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """List thoughts deterministically with filters and pagination.

    A direct pass-through to the public
    :meth:`~engrava.SqliteEngravaCore.list_thoughts`, which orders by
    descending ``updated_cycle`` (newest first) and applies every filter
    server-side.  Unlike :func:`search_memory_impl` this is a plain
    browse: there is no relevance ranking and therefore no score.  It is
    the right tool when a caller wants an exhaustive, paginated slice of
    memory narrowed by structured fields rather than the best matches for
    a query.

    Args:
        store: The store to query.
        thought_type: When set, keep only thoughts of this type.
        lifecycle_status: When set, keep only thoughts in this state.
        priority: When set, keep only thoughts at this priority level.
        min_cycle: Inclusive lower bound on ``updated_cycle``.
        max_cycle: Inclusive upper bound on ``updated_cycle``.
        include_expired: When ``True``, expired thoughts are included.
            Defaults to ``False`` so expired thoughts stay hidden.
        limit: Maximum number of thoughts to return (page size).
        offset: Number of leading thoughts to skip (page start).

    Returns:
        A dict with a ``thoughts`` list of JSON-serialisable thoughts
        (newest first), the ``count`` of thoughts on this page, and the
        ``limit`` / ``offset`` that were applied so the caller can drive
        pagination.

    """
    thoughts = await store.list_thoughts(
        thought_type=thought_type.value if thought_type is not None else None,
        lifecycle_status=lifecycle_status.value if lifecycle_status is not None else None,
        priority=priority.value if priority is not None else None,
        min_cycle=min_cycle,
        max_cycle=max_cycle,
        include_expired=include_expired,
        limit=limit,
        offset=offset,
    )
    serialised = [thought.model_dump(mode="json") for thought in thoughts]
    return {
        "thoughts": serialised,
        "count": len(serialised),
        "limit": limit,
        "offset": offset,
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


async def delete_thought_impl(store: SqliteEngravaCore, thought_id: str) -> dict[str, Any]:
    """Delete a thought by identifier.

    Deleting an identifier that is not present is a no-op rather than an
    error: the call simply reports that nothing was removed.

    Args:
        store: The store to write to.
        thought_id: Identifier of the thought to delete.

    Returns:
        A dict with a ``deleted`` flag: ``True`` when a thought was
        removed, ``False`` when no thought had the given identifier.

    """
    deleted = await store.delete_thought(thought_id)
    return {"deleted": deleted}


async def delete_edge_impl(store: SqliteEngravaCore, edge_id: str) -> dict[str, Any]:
    """Delete an edge by identifier.

    Deleting an identifier that is not present is a no-op rather than an
    error: the call simply reports that nothing was removed.

    Args:
        store: The store to write to.
        edge_id: Identifier of the edge to delete.

    Returns:
        A dict with a ``deleted`` flag: ``True`` when an edge was removed,
        ``False`` when no edge had the given identifier.

    """
    deleted = await store.delete_edge(edge_id)
    return {"deleted": deleted}


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


def _summarize_recent_prompt(limit: int, recent: dict[str, Any]) -> str:
    """Build the ``summarize_recent_memory`` prompt text.

    The text embeds the recent thoughts already gathered from the store so
    the assistant can summarise them directly, while still naming the read
    tools and resources it can use to widen the picture.  Embedding is
    read-only: ``recent`` is the output of :func:`recent_thoughts_impl`.

    Args:
        limit: Number of recent thoughts the summary should cover.
        recent: The payload returned by :func:`recent_thoughts_impl`,
            carrying a ``thoughts`` list newest-first.

    Returns:
        A ready-to-send instruction asking for a concise summary of the
        most recent stored memory.

    """
    thoughts = recent.get("thoughts", [])
    if thoughts:
        snapshot = json.dumps(thoughts, indent=2)
        data_section = (
            f"Here are the {len(thoughts)} most recent thoughts "
            f"(newest first), as JSON:\n\n{snapshot}\n\n"
        )
    else:
        data_section = "The store currently holds no thoughts to summarise.\n\n"
    return (
        f"Summarise the {limit} most recently stored memories in this "
        "engrava store.\n\n"
        f"{data_section}"
        "If you need more detail or want to confirm the latest activity, "
        "read the `engrava://recent` resource or call the `memory_stats` "
        "tool; use `get_thought` to expand any single thought by its "
        "identifier. Produce a concise summary that highlights the main "
        "themes, any recurring topics, and anything that looks important "
        "or unresolved. Keep it brief — a short paragraph or a few bullet "
        "points."
    )


def _find_related_prompt(topic: str) -> str:
    """Build the ``find_related`` prompt text.

    Args:
        topic: The subject to find related thoughts about.

    Returns:
        A ready-to-send instruction asking the assistant to gather and
        synthesise thoughts related to ``topic`` using ``search_memory``.

    """
    return (
        f"Find and synthesise what this engrava memory store holds about "
        f"{topic!r}.\n\n"
        f"Use the `search_memory` tool with a query for {topic!r} (it ranks "
        "results by lexical, vector, and recency signals); you can also try "
        "`search_keywords` for an exact-term pass. Expand the most relevant "
        "hits with `get_thought` to read their full content. Then synthesise "
        "the findings into a short, organised summary of what is known about "
        f"{topic!r}, grouping related points and noting any gaps or "
        "contradictions."
    )


def _reflect_on_topic_prompt(topic: str) -> str:
    """Build the ``reflect_on_topic`` prompt text.

    Args:
        topic: The subject to reflect on.

    Returns:
        A ready-to-send instruction that scaffolds a structured reflection
        over what the store holds about ``topic``.

    """
    return (
        f"Reflect on what this engrava memory store holds about {topic!r}.\n\n"
        f"First gather the relevant memories: call `search_memory` for "
        f"{topic!r} and read the strongest hits in full with `get_thought`. "
        "Then reflect rather than merely listing: structure your response "
        "around (1) what is well established about the topic, (2) open "
        "questions or gaps in what is stored, and (3) any tensions or "
        "contradictions between thoughts. Close with one or two concrete "
        "follow-ups worth recording. Ground every observation in the "
        "retrieved thoughts."
    )


def build_server() -> FastMCP:
    """Build the engrava MCP server with its tools registered.

    The returned server resolves its store from the environment when its
    lifespan starts and releases the connection when the lifespan ends.
    The read tools, the resources, and the prompts are always registered;
    the write tools are registered unless :func:`_read_only_enabled`
    reports a read-only deployment.

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
            "hybrid and keyword search, list thoughts with structured filters "
            "and pagination, run structured MindQL FIND queries, and read "
            "store statistics. Hybrid search (search_memory) can also be "
            "narrowed by thought type, lifecycle status, or priority, but it "
            "filters after ranking; for an exhaustive unranked listing by "
            "those fields use list_memory. Unless the server is started in "
            "read-only mode, you can also store new thoughts, update existing "
            "thoughts, link thoughts with typed edges, and delete thoughts or "
            "edges. Read-only resources are also available as attachable "
            "context: a single thought (engrava://thought/{thought_id}), store "
            "statistics (engrava://stats), and the most recent thoughts "
            "(engrava://recent). Guided prompts scaffold common retrieval "
            "workflows: summarize_recent_memory, find_related, and "
            "reflect_on_topic."
        ),
        lifespan=lifespan,
    )
    register_resources(server, provider)
    register_prompts(server, provider)
    register_tools(server, provider)
    return server


def register_resources(server: FastMCP, provider: StoreProvider) -> None:
    """Register the read-only MCP resources on a server.

    Three resources are registered.  They are reads by definition, so —
    unlike the write tools — they are *not* gated by the read-only
    environment flag and are advertised in every deployment:

    ``engrava://thought/{thought_id}``
        A single thought as a JSON document.  An unknown identifier
        yields a graceful not-found payload rather than an error.
    ``engrava://stats``
        Store-health counts and size.  Shares :func:`memory_stats_impl`
        with the ``memory_stats`` tool, so the two agree by construction.
    ``engrava://recent``
        The most-recently-updated thoughts as a JSON document.

    Each handler returns a JSON string with the ``application/json`` MIME
    type, so clients receive a stable, machine-parseable payload.

    Args:
        server: The server to register resources on.
        provider: Supplies the active store to each resource at read time.

    """

    @server.resource(
        "engrava://thought/{thought_id}",
        name="thought",
        title="Thought",
        description="A single thought by its identifier, as a JSON document.",
        mime_type=RESOURCE_MIME_TYPE,
    )
    async def thought_resource(thought_id: str) -> str:
        payload = await get_thought_impl(provider.require(), thought_id)
        return json.dumps(payload)

    @server.resource(
        "engrava://stats",
        name="stats",
        title="Store statistics",
        description="Aggregate thought and edge counts and total storage size.",
        mime_type=RESOURCE_MIME_TYPE,
    )
    async def stats_resource() -> str:
        payload = await memory_stats_impl(provider.require())
        return json.dumps(payload)

    @server.resource(
        "engrava://recent",
        name="recent",
        title="Recent thoughts",
        description="The most-recently-updated thoughts, newest first, as a JSON document.",
        mime_type=RESOURCE_MIME_TYPE,
    )
    async def recent_resource() -> str:
        payload = await recent_thoughts_impl(provider.require())
        return json.dumps(payload)


def register_prompts(server: FastMCP, provider: StoreProvider) -> None:
    """Register the guided retrieval prompts on a server.

    Three prompts are registered.  They are parameterised templates that a
    client surfaces as slash-commands or buttons; each renders a
    ready-to-send instruction guiding the assistant to gather context with
    the read tools and resources before answering.  Prompts are
    read-oriented, so — like the resources and unlike the write tools —
    they are *not* gated by the read-only environment flag and are
    advertised in every deployment:

    ``summarize_recent_memory``
        Summarise the most recent thoughts.  Takes an optional ``limit``;
        this is the one prompt that reads the store, embedding the recent
        thoughts (read-only) so the assistant can summarise them inline.
    ``find_related``
        Find and synthesise thoughts related to a required ``topic``.
    ``reflect_on_topic``
        Reflect over what memory holds about a required ``topic``.

    Args:
        server: The server to register prompts on.
        provider: Supplies the active store to ``summarize_recent_memory``
            at render time; the topic prompts are pure templates and do not
            use it.

    """

    @server.prompt(
        name="summarize_recent_memory",
        title="Summarise recent memory",
        description=(
            "Summarise the most recently stored thoughts. Optionally set "
            "how many recent thoughts to consider."
        ),
    )
    async def summarize_recent_memory(limit: int = DEFAULT_SUMMARY_LIMIT) -> str:
        recent = await recent_thoughts_impl(provider.require(), limit=limit)
        return _summarize_recent_prompt(limit, recent)

    @server.prompt(
        name="find_related",
        title="Find related thoughts",
        description="Find and synthesise stored thoughts related to a topic.",
    )
    def find_related(topic: str) -> str:
        return _find_related_prompt(topic)

    @server.prompt(
        name="reflect_on_topic",
        title="Reflect on a topic",
        description="Reflect on what stored memory holds about a topic.",
    )
    def reflect_on_topic(topic: str) -> str:
        return _reflect_on_topic_prompt(topic)


# C901: the mccabe count is inflated by the nested ``@server.tool`` handler
# definitions (one trivial delegating wrapper per tool), not by branching logic
# — this function has a single branch, the read-only guard. Splitting the flat
# registration list would hurt readability, so the complexity cap is waived here
# deliberately.
def register_tools(server: FastMCP, provider: StoreProvider) -> None:  # noqa: C901
    """Register the MCP tools on a server.

    The six read tools (``get_thought``, ``search_memory``,
    ``search_keywords``, ``list_memory``, ``query_memory``,
    ``memory_stats``) are always registered.  The five write tools are
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
        async with _tool_errors():
            return await get_thought_impl(provider.require(), thought_id)

    @server.tool(
        name="search_memory",
        description=(
            "Hybrid ranked search (lexical + vector + recency) over stored "
            "memory. Returns ranked thought identifiers with scores and the "
            "search backends that were available. Optionally narrow the "
            "ranked hits by thought type, lifecycle status, or priority; "
            "these filters are applied after ranking, so a filtered call may "
            "return fewer than top_k results and reports how many ranked hits "
            "were dropped. For an exhaustive, unranked, paginated listing by "
            "those same fields, use list_memory instead."
        ),
        annotations=_READ_ONLY,
    )
    async def search_memory(
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        include_reflections: bool = True,
        thought_type: ThoughtType | None = None,
        lifecycle_status: LifecycleStatus | None = None,
        priority: Priority | None = None,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await search_memory_impl(
                provider.require(),
                query_text,
                top_k=top_k,
                include_reflections=include_reflections,
                thought_type=thought_type,
                lifecycle_status=lifecycle_status,
                priority=priority,
            )

    @server.tool(
        name="list_memory",
        description=(
            "List stored thoughts deterministically with optional filters and "
            "pagination. Unlike search_memory this does no relevance ranking "
            "and returns no scores: it is a plain browse over memory, ordered "
            "newest first. Filter by thought type, lifecycle status, priority, "
            "and an updated-cycle range; page through results with limit and "
            "offset. Use this to enumerate memory by structured fields; use "
            "search_memory when you want the best matches for a query."
        ),
        annotations=_READ_ONLY,
    )
    async def list_memory(
        thought_type: ThoughtType | None = None,
        lifecycle_status: LifecycleStatus | None = None,
        priority: Priority | None = None,
        *,
        min_cycle: int | None = None,
        max_cycle: int | None = None,
        include_expired: bool = False,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        async with _tool_errors():
            return await list_memory_impl(
                provider.require(),
                thought_type=thought_type,
                lifecycle_status=lifecycle_status,
                priority=priority,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
                include_expired=include_expired,
                limit=limit,
                offset=offset,
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
        async with _tool_errors():
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
        async with _tool_errors():
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
        async with _tool_errors():
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
        async with _tool_errors():
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
        async with _tool_errors():
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
        async with _tool_errors():
            return await link_thoughts_impl(
                provider.require(),
                from_thought_id,
                to_thought_id,
                edge_type,
                weight=weight,
                edge_id=edge_id,
            )

    @server.tool(
        name="delete_thought",
        description=(
            "Delete a thought by its identifier. Use this to remove a memory "
            "that is wrong or no longer wanted. Returns whether a thought was "
            "removed; deleting an identifier that does not exist is not an "
            "error and simply reports that nothing was removed."
        ),
        annotations=_WRITE_DESTRUCTIVE,
    )
    async def delete_thought(thought_id: str) -> dict[str, Any]:
        async with _tool_errors():
            return await delete_thought_impl(provider.require(), thought_id)

    @server.tool(
        name="delete_edge",
        description=(
            "Delete an edge between two thoughts by its identifier. Use this to "
            "remove a relationship that is wrong or no longer wanted. Returns "
            "whether an edge was removed; deleting an identifier that does not "
            "exist is not an error and simply reports that nothing was removed."
        ),
        annotations=_WRITE_DESTRUCTIVE,
    )
    async def delete_edge(edge_id: str) -> dict[str, Any]:
        async with _tool_errors():
            return await delete_edge_impl(provider.require(), edge_id)


def main() -> None:
    """Run the engrava MCP server over stdio.

    Builds the server and serves it on the stdio transport (the FastMCP
    default).  This is the console-script and ``python -m engrava.mcp``
    entry point.
    """
    build_server().run()

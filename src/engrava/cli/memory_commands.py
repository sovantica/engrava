"""One-shot memory verbs: ``remember``, ``recall``, ``link``.

Every other built-in (``info``, ``verify``, ``query``, ``snapshot``,
``restore``, ``gc``, ``migrate``, ``export``) inspects or maintains a
database that something else already wrote to. Nothing lets a shell store or
search a thought in one call — every non-Python integration has had to shell
into ``python -c ...`` to reach ``create_thought`` / ``recall`` /
``create_edge`` directly. These three commands close that gap.

They are deliberately thin wrappers over the public library surface:

* ``remember`` builds a :class:`~engrava.domain.models.thought.ThoughtRecord`
  explicitly and calls ``create_thought()`` — **not** the library's own
  ``remember()`` helper, which takes only text, metadata, and a dedup flag and
  always produces a ``NOTE`` / ``P3`` thought. ``--type`` / ``--priority``
  cannot ride that helper.
* ``recall`` calls the library ``recall()`` directly and prints its ranked
  results.
* ``link`` builds an :class:`~engrava.domain.models.edge.EdgeRecord` and
  calls ``create_edge()`` — there is no public ``link()`` to call instead.

All three resolve their database through
:func:`engrava.cli.store_resolution.resolve_store_target`, the shared
precedence used across the memory verbs (see that module's docstring), so a
``--config`` file's embedding/search/journal configuration is honoured the
same way a direct library ``recall()`` call would honour it — unlike a bare
``aiosqlite`` connection, which has no embedding provider at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, NoReturn

import click

from engrava import (
    EdgeRecord,
    EdgeType,
    FieldOp,
    FieldPredicate,
    InvalidFilterPathError,
    KnowledgeSource,
    LifecycleStatus,
    MetadataFilter,
    Priority,
    ReferentialIntegrityError,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.cli.exception_reporting import _describe_exception, _frame_only_stack
from engrava.cli.main import _opened_db, _run, cli
from engrava.cli.store_resolution import ResolvedStore, resolve_store_target
from engrava.config_validation import ConfigError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

    from engrava.cli.config import EngravaCLIConfig

logger = logging.getLogger(__name__)

# Plain version strings, not URLs — the CLI has no schema registry to
# publish a URL against, and a bare identifier is enough for a caller to
# branch on across a release.
_REMEMBER_SCHEMA = "engrava.cli.remember.v1"
_RECALL_SCHEMA = "engrava.cli.recall.v1"
_LINK_SCHEMA = "engrava.cli.link.v1"
_ERROR_SCHEMA = "engrava.cli.error.v1"

# The one kind (and exit code) a previously-unclassified exception maps to.
# See ``_error_boundary`` for why this is deliberately not one of the
# specific kinds (``invalid_config``, ``database_not_found``, ...) or their
# exit codes (2/3/4).
_UNEXPECTED_ERROR_KIND = "unexpected_error"
_UNEXPECTED_ERROR_EXIT_CODE = 1

# Shared, single-source fallback text for the two known-error adapters below
# (an exact `ConfigError` / `ReferentialIntegrityError` whose fields fail
# validation, and every subclass of either). One named constant per message,
# used by both the "malformed field" and the "subclass" branch, so the two
# call sites cannot drift apart the way a hand-copied literal has before.
# Says the detail is *omitted*, not that a read was attempted and failed --
# the subclass branch makes no read attempt at all, so "omitted" is the
# claim that stays true in both cases.
_CONFIG_ERROR_DETAILS_OMITTED_MESSAGE = (
    "The configuration file is invalid. Its details are omitted rather than shown here."
)
_MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE = (
    "link: FROM or TO does not reference an existing thought. Which one is "
    "omitted rather than shown here."
)

_VALID_THOUGHT_TYPES = tuple(t.value for t in ThoughtType)
_VALID_PRIORITIES = tuple(p.value for p in Priority)
_VALID_EDGE_TYPES = tuple(t.value for t in EdgeType)


class _CliError(Exception):
    """A memory-verb failure already classified by kind, message and exit code.

    Raised by :func:`_fail` instead of writing output and exiting on the
    spot. The actual write is deferred to :func:`_error_boundary`, the one
    place that performs it -- see that function's docstring for why the
    write has to happen there, after every ``async with`` block still open
    at the point of failure has finished unwinding, rather than here,
    before it.
    """

    def __init__(self, *, as_json: bool, kind: str, message: str, code: int) -> None:
        super().__init__(message)
        self.as_json = as_json
        self.kind = kind
        self.message = message
        self.code = code


def _emit_and_exit(*, as_json: bool, kind: str, message: str, code: int) -> NoReturn:
    r"""Write the one, terminal line of output for a command's failure, and exit.

    This is the *only* place that writes a memory-verb's failure output --
    :func:`_fail` no longer does, precisely so that nothing else in the
    command can write after it (see :func:`_error_boundary`, the only
    caller). ``message`` is always an *exact* ``str`` instance by the time
    it reaches here: ``_fail``'s callers pass literal text or the shared
    fixed-fallback constants, the boundary's own generic branch builds its
    fallback through :func:`_describe_exception` (which normalizes both
    halves with :func:`~engrava.config_validation.own_str` before
    returning), and the two known-error adapters in
    :func:`_resolve_for_command` / ``link`` now check ``type(...) is str``
    on a field before ever using it, falling back to a fixed literal
    otherwise -- so nothing here can be a ``str`` *subclass* whose own
    ``__format__`` might misbehave downstream, even though ``json.dumps``
    would have accepted one either way.

    ``ensure_ascii=True``: a message can embed an arbitrary caller-supplied
    value (a ``--db`` path, a ``--meta`` token, ...), and with
    ``ensure_ascii=False`` a value containing U+0085/U+2028/U+2029 --
    Unicode line separators that are not JSON control characters and so
    survive literally -- would come through as a literal line break inside
    the JSON object. That breaks the documented reading recipe as soon as a
    consumer's split treats those the same as ``\\n`` (``str.splitlines()``
    does; splitting strictly on ``\\n`` does not -- see ``docs/cli.md``).
    Escaping non-ASCII here removes the hazard from the one line this CLI
    controls end to end, at the cost of a stored thought's own non-ASCII
    *content* never appearing in an error message in the first place (it
    doesn't -- these messages only ever carry option values and exception
    text, never thought content).

    Args:
        as_json: Whether the invoking command was given ``--json``.
        kind: Short machine-readable error identifier, carried in the JSON
            object's ``error`` field.
        message: Human-readable message. Contains the offending value when
            available, and for an enum-validation failure, every valid member.
        code: Process exit code (``2`` usage/validation, ``3`` absent
            database on a read, ``4`` a missing referenced thought, ``1`` an
            unanticipated failure).

    """
    if as_json:
        payload = {"schema": _ERROR_SCHEMA, "error": kind, "message": message}
        click.echo(json.dumps(payload, ensure_ascii=True), err=True)
    else:
        click.echo(message, err=True)
    sys.exit(code)


def _fail(*, as_json: bool, kind: str, message: str, code: int) -> NoReturn:
    """Classify a command failure. The actual exit happens later, at :func:`_error_boundary`.

    Args:
        as_json: Whether the invoking command was given ``--json``.
        kind: Short machine-readable error identifier (e.g.
            ``"malformed_meta"``), carried in the JSON object's ``error`` field.
        message: Human-readable message. Contains the offending value when
            available, and for an enum-validation failure, every valid member.
        code: Process exit code (``2`` usage/validation, ``3`` absent database
            on a read, ``4`` a missing referenced thought, ``1`` an
            unanticipated failure :func:`_error_boundary` converted).

    Raises:
        _CliError: Always -- carrying ``as_json``/``kind``/``message``/``code``
            for :func:`_error_boundary` to act on once this propagates there.

    """
    raise _CliError(as_json=as_json, kind=kind, message=message, code=code)


class _ResolvedDatabasePath:
    """Single-slot mutable box for the database path a command body resolves.

    :func:`_error_boundary`'s generic branch needs to name the resolved
    store's path when an unclassified exception (a corrupt ``--db`` file, an
    uninitialised database, ...) reaches it, but that path is only known once
    :func:`_resolve_for_command` returns -- itself running *inside* the
    ``with`` block, since a malformed ``--config`` is one of the failures the
    boundary exists to convert with its own specific kind, never the generic
    one. A command body sets ``.path`` immediately after resolution succeeds;
    a failure raised before that point leaves it at the default ``None``, and
    the generic branch falls back to a path-free message rather than naming a
    database nothing has resolved yet. This happens today, not just
    hypothetically: :func:`_resolve_for_command` only gives ``--config``
    itself the specific ``invalid_config`` treatment for an exact
    :class:`~engrava.config_validation.ConfigError`; a directory or
    non-UTF-8 ``--config`` path raises ``IsADirectoryError`` /
    ``UnicodeDecodeError`` instead, before resolution ever returns, and
    ``remember`` reports it exactly this way -- ``"remember: unexpected
    IsADirectoryError: [Errno 21] Is a directory: '...'"``, with no database
    path in the message.

    Deliberately not a ``dataclass``: ``tests/test_config_validation_parity.py``
    discovers every dataclass under ``engrava`` by walking the package and
    requires each one to be classified as configuration or not. This is
    neither -- it is per-invocation mutable state with a single field that
    changes after construction, not a settings object -- so a plain class
    with ``__slots__`` avoids tripping a gate built for an entirely different
    question.
    """

    __slots__ = ("path",)

    def __init__(self) -> None:
        self.path: Path | None = None


@contextmanager
def _error_boundary(*, as_json: bool, command: str) -> Iterator[_ResolvedDatabasePath]:
    """Guard a memory-verb body -- the one seam it passes through on the way out.

    Wraps a command's *entire* body — synchronous validation, store
    resolution, and the ``_run(coro)`` call together — so that any
    ``Exception`` escaping any of it becomes the documented
    ``engrava.cli.error.v1`` object (or its plain-text equivalent under
    non-``--json``), never an interpreter traceback, regardless of which
    library underneath raised it or what it raised.

    This replaces enumerating exception types one ``except`` clause at a
    time. Two review rounds each found more types escaping that enumeration
    — a malformed ``--filter`` path, a directory or corrupt file given as
    ``--db``, an unreadable or non-UTF-8 ``--config``, an uninitialised
    database — because "everything a library this CLI touches might raise"
    is not a set these functions can enumerate correctly, and ``mypy`` gives
    no help either: nothing in ``resolve_hooks()`` or ``aiosqlite.connect()``'s
    signature says what they raise. Catching ``Exception`` once, here, closes
    the *class* of defect instead of chasing its latest instance.

    **This is also the only place that writes a failure's output, and it
    does so only after everything it wraps has finished unwinding.** A
    third review round found that the previous shape -- ``_fail`` itself
    calling ``click.echo`` and ``sys.exit()`` at the point of detection --
    let cleanup still running *underneath* that call corrupt the output
    that had already been written: a database's ``close()`` failing during
    the ``async with`` unwind either printed a stray line after the
    documented final JSON object, or (worse, on a ``--config`` store, whose
    ``finally: await store.close()`` unconditionally ran on the way out)
    replaced an already-decided ``SystemExit(4)`` outright, so the boundary
    caught the close failure instead and reported a second, contradicting
    error object at exit ``1``. Enumerating more cleanup paths to special-case
    is the same mistake this boundary already replaced once for exception
    *types*; the actual fix is structural: :func:`_fail` (and a directly
    raised :class:`ReferentialIntegrityError` conversion, and this boundary's
    own generic-exception branch) no longer write or exit themselves, they
    only raise :class:`_CliError` -- an ordinary ``Exception`` -- and let it
    propagate through every enclosing ``async with``/``finally`` exactly like
    any other exception would, so those blocks' own cleanup (and cleanup's
    own failure, logged rather than raised -- see ``_opened_full_store``)
    always completes *before* this ``except`` clause below ever runs. The
    write-and-exit in :func:`_emit_and_exit` is therefore always the last
    thing a failing invocation does, by construction, not by checking that it
    happened to be.

    **A kind this module already classifies specifically never reaches the
    generic branch below -- not even for a subclass.** An *exact*
    ``ConfigError`` instance in :func:`_resolve_for_command`, an *exact*
    ``ReferentialIntegrityError`` instance in ``link``'s body, and every
    validation check that calls :func:`_fail` directly (``empty_text``,
    ``malformed_meta``, ``invalid_top_k``, ...) all raise ``_CliError``
    with their own specific kind and exit code already decided; the
    ``except _CliError`` clause below emits exactly that, unchanged. Both
    ``ConfigError`` and ``ReferentialIntegrityError`` are public library
    classes, so a third-party *subclass* is realistic, and reading a
    subclass-overridable attribute or calling an overridden ``__str__``
    unprotected -- which the specific formatting for each would otherwise
    do -- is not safe. Each call site checks ``type(exc) is ...``: a
    *subclass* keeps the same documented kind and exit code -- the class
    hierarchy is trustworthy even when its attributes are not -- but its
    message is a fixed literal instead, with nothing read off it at all: a
    later review round found that even the hardened, generic
    :func:`_describe_exception` was not safe enough for this case, since it
    still includes ``str(exc)``, which the subclass fully controls, and a
    subclass built to return a believable-looking fabricated diagnosis
    would have that fabrication reported as if this CLI had produced it. An
    *exact* instance is not thereby safe to format directly either -- its
    own fields can still hold a hostile or unexpected value (a still later
    review round demonstrated this for both classes) -- so each field is
    read once and validated (an exact ``str`` type, and, for
    ``ReferentialIntegrityError``, a real column name whose value matches
    this invocation's own endpoint) before it is used anywhere; a field
    that fails validation falls back to the same fixed literal the subclass
    branch uses, with the documented kind and exit code unchanged either
    way. Only an exception that neither of those two classes (exact or
    subclass), nor any other narrower catch, recognised ever reaches the
    generic branch below, and only that one gets the generic kind.
    ``SystemExit`` itself still passes through this boundary
    untouched, same as before ``_CliError`` existed: it does not subclass
    ``Exception``, so a ``sys.exit()`` a click internal or a genuinely
    intentional early exit raises is never caught here.

    **Why a generic kind and exit code, not one of the specific ones.**
    Reusing ``database_not_found`` (exit ``3``) or ``invalid_config`` (exit
    ``2``) for something this boundary did not specifically recognise would
    tell a scripted consumer something false — those exit codes already carry
    a narrower, documented meaning ("the resolved database does not exist",
    "the config file does not parse"), and code that branches on them (say,
    retrying ``3`` after creating the missing database) would retry into a
    failure retrying cannot fix. ``exit 1`` and ``kind: "unexpected_error"``
    make "this was not one of the documented failure shapes" observable
    rather than silently indistinguishable from one that was. ``1`` also
    matches this CLI's own existing convention for an undifferentiated
    failure (``info`` / ``verify`` / ``query`` already exit ``1`` on their own
    unclassified refusals), rather than colliding with ``2`` / ``3`` / ``4``,
    which these three verbs already define narrowly.

    **What the message contains.** ``f"{command}: unexpected
    {_describe_exception(exc)}"`` — the exception's own type name and text,
    never a traceback. :func:`_describe_exception` guards both halves of
    that description independently: an exception whose own ``__str__``
    raises, and one whose *type's metaclass* raises on a plain ``__name__``
    read. That guarding converts an ordinary exception only, though: a
    ``KeyboardInterrupt`` or ``SystemExit`` raised while obtaining either
    half propagates instead, so building this very message can itself end
    in an abort rather than the error object described here. For exactly
    the cases this boundary exists to catch,
    that text is already the actionable part:
    ``sqlite3.DatabaseError: file is not a database``, ``IsADirectoryError:
    [Errno 21] Is a directory: '...'``, and
    ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0:
    invalid start byte`` each already name the offending path or byte,
    courtesy of the standard library's own exception messages — this
    boundary does not need to know the specific shape to say something
    useful about it.

    **The underlying exception is not discarded, just not shown by
    default, and not fully.** :func:`_describe_exception`'s type-name-and-text
    summary plus the caught exception's own stack -- filename, line number
    and function name per frame, built by :func:`_frame_only_stack` (see its
    docstring for why that reads the traceback through the built-in
    descriptor rather than the plain, interceptable ``exc.__traceback__``
    attribute) -- are logged at ``DEBUG`` through this module's
    logger, which is invisible unless the invocation carried ``--verbose``
    — ``_configure_verbose_logging`` in ``main.py`` attaches a ``DEBUG``
    handler to the ``engrava`` logger hierarchy for exactly the lifetime of
    one invocation. A developer chasing a real defect reruns with
    ``--verbose`` and gets that stack; a script parsing ``--json`` output
    never sees one by default. This deliberately trades diagnostic detail
    for safety: it is not a full exception-chain rendering. It never calls
    the exception's own formatter a second time, and it never visits a
    ``__cause__``, ``__context__``, attached notes, or exception-group
    children -- an earlier shape passed ``exc_info=True`` to ``logger.debug``
    instead, which made CPython's own traceback formatter render the
    exception a second time to build that text; that formatter wraps its own
    rendering in a bare ``except``, so a real OS ``SIGINT`` arriving during
    that second render was swallowed before it could reach this module's
    guards, and the command still exited ``1`` with an ordinary error object
    instead of aborting (verified by sending a real signal). Reading frame
    metadata instead touches no exception-controlled code at all: nothing
    here calls ``__str__``, ``__format__``, or any other overridable method,
    and no local variable or source line is inspected. This debug call is
    unconditionally cheap when ``--verbose`` was not given: it is wrapped in
    its own ``logger.isEnabledFor(logging.DEBUG)`` check, so neither the
    stack nor the formatted message is ever built in that case -- which also
    means it cannot itself raise ``MemoryError`` under the exhaustion this
    boundary is trying to survive unless ``--verbose`` asked it to do the one
    thing that does allocate.

    **This does not make a programming error in this CLI's own code harder
    to find.** Catching ``Exception`` this broadly risks turning a genuine
    bug here into "just another unexpected_error" that stops getting looked
    at. Two things keep that from happening: the exit code is always
    non-zero and distinct from every success path, so a test or a script
    still fails loudly even without reading the message; and the traceback
    is one flag away rather than requiring a code change to surface — a
    developer is never worse off than "reproduce with the same input, add
    ``--verbose``". What this boundary deliberately does *not* do is
    re-raise or print a traceback by default: a CLI that sometimes
    tracebacks and sometimes doesn't, depending on which exception type
    happened to already be enumerated, is the exact defect this replaces.

    **What is not, and cannot be, guaranteed.** A ``MemoryError`` raised
    while memory is genuinely exhausted is still caught and still routed
    through this same path, but converting it reliably is not promised:
    the debug-log gate above avoids the *avoidable* allocation, yet
    ``_emit_and_exit`` still has to build and print a message, and under
    real exhaustion even that can fail. This boundary narrows the failure
    surface as far as ordinary code can; it does not repeal the fact that
    nothing in a CPython process is exception-safe against running out of
    memory at an arbitrary point.

    Args:
        as_json: Whether the invoking command was given ``--json``.
        command: The command name (``"remember"`` / ``"recall"`` /
            ``"link"``), named in the fallback message.

    Yields:
        A :class:`_ResolvedDatabasePath` box, empty until the command body
        sets its ``.path`` right after :func:`_resolve_for_command` returns.
        The generic branch below reads it back to name the database an
        unclassified failure (a corrupt ``--db`` file, an uninitialised
        database, ...) happened against, the same way the command's own
        "Database not found" / "Created database" messages already do.

    """
    database = _ResolvedDatabasePath()
    try:
        yield database
    except _CliError as exc:
        _emit_and_exit(as_json=exc.as_json, kind=exc.kind, message=exc.message, code=exc.code)
    except Exception as exc:  # noqa: BLE001 -- deliberate: the CLI's one catch-all boundary
        # A plain `type(exc).__name__` argument here (as opposed to `exc`
        # itself, passed positionally so its formatting is deferred to
        # `getMessage()` and so only happens under `--verbose`) would be
        # evaluated eagerly at this call site regardless of whether DEBUG
        # logging is enabled -- `Logger.debug`'s own `isEnabledFor` check
        # runs *after* Python has already evaluated every argument passed to
        # it. Computing the description once, through `_describe_exception`,
        # keeps that read behind the same guard the message below uses,
        # rather than reading `exc` a second time, unprotected, through the
        # logging arguments themselves.
        description = _describe_exception(exc)
        if logger.isEnabledFor(logging.DEBUG):
            # Deliberately not `logger.debug(..., exc_info=True)`: that made
            # Python's own traceback formatter render `exc` a second time to
            # build the traceback text, outside `_describe_exception`'s
            # guards entirely -- and that formatter wraps its own rendering
            # in a bare `except`, so a real OS `SIGINT` arriving during that
            # second render was swallowed before it could reach this
            # module's guards, leaving this command to still emit an
            # `unexpected_error` object at exit `1` instead of aborting
            # (verified with a real signal). `_frame_only_stack` instead
            # reads only frame metadata -- filename, line number, function
            # name -- for each frame, through the built-in traceback
            # descriptor rather than a plain, interceptable attribute read
            # (see that function's docstring): it never calls `exc`'s own
            # `__str__` or any other overridable method, and it never visits
            # a `__cause__`, `__context__`, attached notes, or
            # exception-group children, so nothing here can re-invoke
            # attacker-controlled formatting. That is a deliberate trade:
            # this stack carries less detail than a full exception-chain
            # traceback would -- no chained-exception text, no source lines,
            # no locals -- in exchange for never executing exception
            # formatting a second time. The `isEnabledFor` check keeps this
            # whole block, not just the log call, cheap when `--verbose` was
            # not given -- the same property `_error_boundary`'s own
            # docstring documents. `_opened_full_store`'s own cleanup-failure
            # warning shares this same helper rather than a second copy.
            logger.debug(
                "Unexpected %s in %r; caught exception's stack (file:line in "
                "function, not a full exception-chain rendering):\n%s",
                description,
                command,
                _frame_only_stack(exc),
            )
        message = (
            f"{command}: {database.path}: unexpected {description}"
            if database.path is not None
            else f"{command}: unexpected {description}"
        )
        _emit_and_exit(
            as_json=as_json,
            kind=_UNEXPECTED_ERROR_KIND,
            message=message,
            code=_UNEXPECTED_ERROR_EXIT_CODE,
        )


def _parse_kv_pairs(pairs: tuple[str, ...], *, option_name: str, as_json: bool) -> dict[str, str]:
    """Parse repeated ``key=value`` options into a flat string-valued dict.

    Args:
        pairs: Raw ``key=value`` strings as given on the command line.
        option_name: The option's own name (e.g. ``"--meta"``), named in a
            failure message so the user knows which flag was malformed.
        as_json: Forwarded to :func:`_fail` for the error-object shape.

    Returns:
        The parsed mapping, later keys overriding earlier ones on a repeated
        key exactly like a shell environment assignment would.

    """
    result: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            _fail(
                as_json=as_json,
                kind=f"malformed_{option_name.lstrip('-')}",
                message=f"Malformed {option_name} {raw!r}: expected key=value.",
                code=2,
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            _fail(
                as_json=as_json,
                kind=f"malformed_{option_name.lstrip('-')}",
                message=f"Malformed {option_name} {raw!r}: key must not be empty.",
                code=2,
            )
        result[key] = value
    return result


@asynccontextmanager
async def _opened_full_store(
    resolved: ResolvedStore,
    cfg: EngravaCLIConfig,
    *,
    create: bool,
) -> AsyncIterator[SqliteEngravaCore]:
    """Open the resolved target as a fully configured store.

    Dispatches on ``resolved.source``: a ``config`` target is built via
    :meth:`SqliteEngravaCore.from_config` (embeddings, search weights, and
    journal settings all apply); ``db_flag`` and ``default`` both open bare,
    through the same ``_opened_db`` helper ``info`` / ``verify`` / ``query`` /
    ``gc`` already use, since ``resolved.db_path`` is ``cfg.db_path`` by
    construction on those two tiers.

    Directories are created here, not by the caller, and only when
    ``create=True`` — a read command (``recall``) must not leave a new empty
    directory behind on a path it is about to refuse.

    Args:
        resolved: The target chosen by :func:`resolve_store_target`.
        cfg: The CLI config, forwarded to ``_opened_db`` for the bare tiers.
        create: Whether this call may create the database (and its parent
            directory) if absent. The caller is responsible for the
            absent-database *read* refusal — this context manager never
            refuses, it only avoids creating on the caller's behalf.

    Yields:
        An open, fully configured store. Closed on exit regardless of how the
        block completes.

    """
    if create:
        resolved.db_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved.source == "config":
        assert resolved.config_path is not None  # noqa: S101 -- tier invariant, not user input
        store = await SqliteEngravaCore.from_config(resolved.config_path)
        try:
            yield store
        except BaseException:
            # The body already raised (or was cancelled) -- including a
            # _CliError already carrying a decided kind/message/exit code
            # (see memory_commands._fail). That is what the caller needs to
            # see, so a close failure here is secondary: log and swallow it
            # rather than letting it replace the real error. An unconditional
            # `finally: await store.close()` cannot make that distinction --
            # a close exception raised inside a `finally` silently replaces
            # whatever was already propagating, which is exactly how a
            # deliberate exit 4 (`missing_thought`) turned into an
            # undocumented second error object at exit 1 during review. Same
            # split _opened_db makes for the bare tier above, and the same
            # convention `engrava.infrastructure.sqlite.engrava_core`'s own
            # `_close_quietly` uses for a bare connection -- not reused
            # directly here since that helper is typed for an
            # ``aiosqlite.Connection``, not a ``SqliteEngravaCore``, and
            # ``SqliteEngravaCore.close()`` already documents handling its
            # own cancellation-safety internally, so this does not need the
            # same explicit shield-and-redrain ceremony.
            #
            # Deliberately not `logger.warning(..., exc_info=True)`: a review
            # round proved that renders more than this close exception. This
            # `except BaseException` block is already handling the original,
            # still-propagating exception, so Python sets it as this close
            # exception's own `__context__` -- and `exc_info=True` asks the
            # standard library's traceback formatter to walk that whole
            # chain: the close exception's own text, the original
            # exception's text a *second* time (`_error_boundary`'s
            # `_describe_exception` already reads it once, downstream, to
            # build the final message), and any `__cause__`/`__context__`
            # or exception-group children attached to *either* -- all
            # through their own overridable formatters. Measured with a
            # secondary close failure whose original exception carried an
            # exception group with one child: one call the fixed code
            # below makes of the close exception's own formatter, not zero:
            # a later review round found that dropping it entirely (as an
            # earlier shape of this fix did, alongside the original
            # exception a second time, the group, and the child, all of
            # which stay at zero) was itself a diagnostic regression -- a
            # frame-only stack says *where* closing failed, never *why*, so
            # an ordinary `PermissionError`, a full disk, or a locked file
            # were all indistinguishable. `_describe_exception` is called
            # here exactly once, on the close exception, which is the same
            # single, non-absorbing attempt it always made for the original
            # exception downstream -- not the forbidden second render: a
            # `KeyboardInterrupt`/`SystemExit` raised while describing the
            # close exception still escapes instead of being caught by a
            # bare `except` the way the standard library's own formatter
            # used to. The same formatter also wraps its own rendering in a
            # bare `except`, so a real OS `SIGINT` (or a formatter raising
            # `SystemExit`) arriving during that render was swallowed there
            # instead of reaching this module's guards, leaving the command
            # to still exit `1` as an ordinary `unexpected_error` object
            # instead of aborting (both verified live). `_frame_only_stack`
            # gives the same "where" detail `_error_boundary`'s own debug
            # log uses, through the same helper, without touching the
            # original exception, the group, or the child.
            try:
                await store.close()
            except Exception as close_exc:  # noqa: BLE001 -- deliberately logged, not raised
                logger.warning(
                    "Error closing store during cleanup: %s; stack (file:line "
                    "in function, not a full exception-chain rendering):\n%s",
                    _describe_exception(close_exc),
                    _frame_only_stack(close_exc),
                )
                # A later round found a real OS SIGINT delivered here -- after
                # the warning above was already logged -- absorbed instead of
                # aborting. `asyncio.run()`'s own SIGINT handler (see
                # `asyncio.runners.Runner`) does not raise anything into this
                # coroutine: on the first Ctrl-C it only calls the main
                # task's `cancel()`, which *requests* a `CancelledError` but
                # only actually throws one in at this coroutine's *next*
                # suspension point. Nothing above this line suspends --
                # `_describe_exception`, `_frame_only_stack`, and
                # `logger.warning` are all synchronous -- so a synchronous
                # function simply cannot observe a pending cancellation at
                # all; without an `await` here, the `raise` below would
                # re-raise the original error object and the request to
                # cancel would be silently dropped once this coroutine
                # finishes. This `await asyncio.sleep(0)` is a real
                # suspension point purely to give that pending cancellation
                # somewhere to be delivered -- it does not sleep in the timer
                # sense, it just returns control to the event loop for one
                # iteration, which is exactly when `Task.__step` checks for
                # and throws in a cancellation that was requested while this
                # coroutine was running synchronously.
                await asyncio.sleep(0)
            raise
        else:
            # The body succeeded. A close failure here is not secondary to
            # anything -- it is the only error there is, so it must
            # propagate normally rather than being logged and swallowed.
            await store.close()
    else:
        async with _opened_db(cfg) as conn:
            store = SqliteEngravaCore(conn)
            if create:
                # A bare connection never migrates on its own -- info /
                # verify / query / gc rely on that (they refuse or warn
                # instead, see the schema-version gate above them in
                # main.py) because they only ever act on a database that
                # supposedly already exists. remember / link are different:
                # they are allowed to create the database, and a freshly
                # created file has no `thought` table at all until this
                # runs. Idempotent on an already-migrated database.
                await store.ensure_schema()
            yield store


def _resolve_for_command(
    ctx: click.Context, *, as_json: bool
) -> tuple[EngravaCLIConfig, ResolvedStore]:
    """Resolve the database target for one memory-verb invocation.

    Reads ``db_explicit`` off ``ctx.obj``, set by the root ``cli()`` group,
    and delegates the precedence decision to :func:`resolve_store_target`.

    A ``--config`` file only ever reaches ``resolve_store_target`` when
    ``db_explicit`` is ``False`` (an explicit ``--db`` wins outright and never
    causes the file to be read at all — see that function's own docstring).
    When it is in play, though, a file the caller named explicitly is worth
    surfacing if it is missing or malformed, rather than silently falling
    through to the CLI's own default database: :func:`resolve_store_target`
    raises :class:`~engrava.config_validation.ConfigError` for exactly that
    case, and this is the one place that turns it into a clean, ``--json``-aware
    CLI failure instead of a traceback.

    **Only an exact ``ConfigError`` instance gets its ``message`` field read
    at all -- a subclass keeps the same kind and exit code regardless.**
    ``ConfigError`` is a public library class (see its own docstring), so a
    third-party subclass is realistic, not theoretical -- and a subclass can
    override anything this function would otherwise read off it before
    ``str(exc)`` is even called. A review round built one whose accessor
    raised and found it silently downgraded to ``unexpected_error`` (losing
    the specific diagnosis this branch exists to give and the exit code
    ``docs/cli.md`` promises categorically for it), and another whose
    accessor raised ``SystemExit`` and found it escaped with no error object
    at all. An earlier fix checked ``type(exc) is ConfigError`` before
    reading anything off it and re-raised a subclass otherwise, sending it
    through :func:`_error_boundary`'s generic branch instead -- which kept
    the failure path from corrupting, but also silently downgraded the
    subclass to ``unexpected_error``, exit ``1``, breaking the documented
    promise that a ``--config`` file this function actually reads (no
    explicit ``--db``), if invalid, is exit ``2``. The class
    hierarchy is trustworthy even when a subclass's own attributes are
    not: a ``ConfigError`` subclass genuinely *is* a configuration failure,
    so it keeps ``invalid_config`` / exit ``2`` either way. Only *how the
    message is built* differs -- a subclass gets a fixed literal message
    instead, with nothing read off it at all. A later review round found
    that even the hardened, generic :func:`_describe_exception` was not
    enough here: it still includes ``str(exc)``, which the subclass fully
    controls, and a subclass built to return a believable-looking
    fabricated diagnosis would have that fabrication reported as if this
    CLI had produced it. Trading the subclass's detail for a fixed, honest
    message closes that the way normalizing the read never could.

    **An exact instance is not automatically safe either -- its own fields
    can still be hostile.** ``type(exc) is ConfigError`` rules out an
    overridden ``__str__`` on the exception itself, but ``.message`` is
    still whatever value was assigned to it, and nothing stops that value
    from being something other than a plain ``str`` (a review round found
    exactly this: an exact ``ConfigError`` carrying a ``str`` subclass whose
    own formatting misbehaved). So this function reads ``.message`` once
    and checks ``type(message) is str`` *before* using it anywhere -- a
    type check never calls the value's own methods, unlike ``str(...)`` or
    an f-string interpolation. A message that passes is used directly and
    unchanged; one that does not falls back to the same fixed message the
    subclass branch uses, with nothing further read off it. Either way the
    kind stays ``invalid_config`` and the exit code stays ``2``.

    Args:
        ctx: The current Click context (a subcommand's, so ``ctx.obj`` is the
            group's shared object).
        as_json: Forwarded to :func:`_fail` for the error-object shape.

    Returns:
        The CLI config and the resolved store target.

    """
    cfg: EngravaCLIConfig = ctx.obj["config"]
    db_explicit: bool = ctx.obj.get("db_explicit", False)
    try:
        resolved = resolve_store_target(cfg, db_explicit=db_explicit)
    except ConfigError as exc:
        if type(exc) is ConfigError:
            # Read once, check the type before using it anywhere -- see the
            # docstring's "not automatically safe either" paragraph. A plain
            # type check never calls the value's own formatting methods.
            message = exc.message
            if type(message) is str:
                _fail(as_json=as_json, kind="invalid_config", message=message, code=2)
            _fail(
                as_json=as_json,
                kind="invalid_config",
                message=_CONFIG_ERROR_DETAILS_OMITTED_MESSAGE,
                code=2,
            )
        # A subclass may override __str__ (or anything else) to raise or
        # return something misleading, but the class hierarchy is still
        # trustworthy -- it genuinely is a configuration failure, so it
        # keeps the same kind and exit code. The message is a fixed literal
        # instead, with nothing read off the subclass at all -- not even
        # through the hardened describer -- because a believable-looking
        # fabricated detail (a review round produced exactly that) would be
        # worse than no detail.
        _fail(
            as_json=as_json,
            kind="invalid_config",
            message=_CONFIG_ERROR_DETAILS_OMITTED_MESSAGE,
            code=2,
        )
    return cfg, resolved


def _report_resolution(cfg: EngravaCLIConfig, resolved: ResolvedStore) -> None:
    """Echo the resolution decision to stderr when ``--verbose`` is set."""
    if cfg.verbose:
        click.echo(resolved.describe(), err=True)


# ------------------------------------------------------------------
# remember
# ------------------------------------------------------------------


@cli.command()
@click.argument("text")
@click.option(
    "--type",
    "thought_type",
    type=click.Choice(_VALID_THOUGHT_TYPES),
    default=ThoughtType.NOTE.value,
    help="Thought type.",
)
@click.option(
    "--priority",
    type=click.Choice(_VALID_PRIORITIES),
    default=Priority.P3.value,
    help="Thought priority.",
)
@click.option(
    "--meta",
    "meta_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Metadata key=value pair (repeatable).",
)
@click.option(
    "--dedup",
    is_flag=True,
    help="Increment confirmation_count and return the existing thought on identical content.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON object instead of a bare id.")
@click.pass_context
def remember(
    ctx: click.Context,
    text: str,
    thought_type: str,
    priority: str,
    meta_pairs: tuple[str, ...],
    *,
    dedup: bool,
    as_json: bool,
) -> None:
    """Store TEXT as a thought and print its id.

    TEXT may be ``-`` to read the content from stdin instead of the command
    line. Built over ``create_thought()`` with an explicitly constructed
    ``ThoughtRecord`` — not the library's ``remember()`` shorthand, which
    always produces a ``NOTE`` / ``P3`` thought and so cannot honour
    ``--type`` / ``--priority``.

    Creates the resolved database if it does not already exist, printing the
    path to stderr when it does.

    The whole body below runs under :func:`_error_boundary`: a validation
    failure this function checks for itself keeps its own specific ``error``
    kind and exit code, but any *other* exception — a corrupt database, an
    unreadable ``--config`` — becomes the documented error object with a
    generic kind instead of a traceback.
    """
    with _error_boundary(as_json=as_json, command="remember") as boundary_database:
        if text == "-":
            text = sys.stdin.read()
        if not text.strip():
            _fail(
                as_json=as_json,
                kind="empty_text",
                message=f"remember: TEXT must not be empty, got {text!r}.",
                code=2,
            )

        metadata = _parse_kv_pairs(meta_pairs, option_name="--meta", as_json=as_json)
        cfg, resolved = _resolve_for_command(ctx, as_json=as_json)
        boundary_database.path = resolved.db_path

        async def _remember() -> None:
            pre_existing = resolved.db_path.exists()
            async with _opened_full_store(resolved, cfg, create=True) as store:
                if not pre_existing:
                    click.echo(f"Created database: {resolved.db_path}", err=True)
                _report_resolution(cfg, resolved)

                thought = ThoughtRecord(
                    thought_id=str(uuid.uuid4()),
                    thought_type=ThoughtType(thought_type),
                    essence=text[:200],
                    content=text,
                    priority=Priority(priority),
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    source="engrava-cli-remember",
                    source_type=KnowledgeSource.EXPERIENCE,
                    metadata={**metadata},
                )
                created = await store.create_thought(thought, deduplicate=dedup)
                deduplicated = dedup and created.thought_id != thought.thought_id

                if as_json:
                    payload = {
                        "schema": _REMEMBER_SCHEMA,
                        "thought_id": created.thought_id,
                        "deduplicated": deduplicated,
                    }
                    click.echo(json.dumps(payload, ensure_ascii=False))
                else:
                    click.echo(created.thought_id)

        _run(_remember())


# ------------------------------------------------------------------
# recall
# ------------------------------------------------------------------


@cli.command()
@click.argument("query")
@click.option("--top-k", type=int, default=10, show_default=True, help="Maximum results to return.")
@click.option(
    "--filter",
    "filter_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Metadata equality filter, KEY=VALUE. Repeatable: different keys are "
        "AND-combined; a repeated key keeps its last value."
    ),
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit a JSON object instead of formatted rows."
)
@click.pass_context
def recall(
    ctx: click.Context,
    query: str,
    top_k: int,
    filter_pairs: tuple[str, ...],
    *,
    as_json: bool,
) -> None:
    """Search for thoughts relevant to QUERY and print ranked results.

    Calls the library ``recall()`` directly, so a configured embedding
    provider, hybrid-search weights, and journal settings apply exactly as
    they would to a direct library call — unlike a bare connection, which has
    no embedding provider at all. ``query`` is structural MindQL, not this:
    it has no ranking and no embedding provider of its own.

    Exits ``3`` naming the resolved path when the database does not exist,
    rather than silently reporting zero hits.

    The whole body below runs under :func:`_error_boundary`: a validation
    failure this function checks for itself — including a malformed
    ``--filter`` key rejected by :class:`~engrava.domain.models.filters.FieldPredicate`
    itself — keeps its own specific ``error`` kind and exit code, but any
    *other* exception — a corrupt database, an unreadable ``--config`` —
    becomes the documented error object with a generic kind instead of a
    traceback.
    """
    with _error_boundary(as_json=as_json, command="recall") as boundary_database:
        if top_k < 1:
            _fail(
                as_json=as_json,
                kind="invalid_top_k",
                message=f"recall: --top-k must be a positive integer, got {top_k}.",
                code=2,
            )

        filters = _parse_kv_pairs(filter_pairs, option_name="--filter", as_json=as_json)
        cfg, resolved = _resolve_for_command(ctx, as_json=as_json)
        boundary_database.path = resolved.db_path

        if not resolved.db_path.exists():
            _fail(
                as_json=as_json,
                kind="database_not_found",
                message=f"Database not found: {resolved.db_path}",
                code=3,
            )

        # _parse_kv_pairs has already confirmed every raw token here contains
        # "=" and a non-empty key, so recovering a key's own raw token by
        # re-splitting is safe without repeating that validation. Built from
        # filter_pairs (not filters.items()) in the same left-to-right order
        # filters itself was built in, so a repeated key maps to its last
        # (winning) occurrence -- the same one whose value ended up in
        # filters, matching _parse_kv_pairs's own override rule.
        raw_filter_tokens = {raw.partition("=")[0].strip(): raw for raw in filter_pairs}

        predicates: list[FieldPredicate] = []
        for key, value in filters.items():
            try:
                predicates.append(FieldPredicate(f"$.{key}", FieldOp.EQ, value))
            except InvalidFilterPathError:
                _fail(
                    as_json=as_json,
                    kind="malformed_filter",
                    message=(
                        f"Malformed --filter {raw_filter_tokens[key]!r}: not a valid filter path."
                    ),
                    code=2,
                )
        metadata_filter = MetadataFilter(predicates) if predicates else None

        async def _recall() -> None:
            async with _opened_full_store(resolved, cfg, create=False) as store:
                _report_resolution(cfg, resolved)

                result = await store.recall(query, top_k=top_k, filters=metadata_filter)
                rows = []
                for thought_id, score in result.results:
                    thought = await store.get_thought(thought_id)
                    rows.append(
                        {
                            "thought_id": thought_id,
                            "score": score,
                            "essence": thought.essence if thought is not None else None,
                        }
                    )

                if as_json:
                    payload = {
                        "schema": _RECALL_SCHEMA,
                        "query": query,
                        "top_k": top_k,
                        "backends_used": sorted(result.backends_used),
                        "results": rows,
                    }
                    click.echo(json.dumps(payload, ensure_ascii=False))
                else:
                    from engrava.cli.main import _format_rows  # noqa: PLC0415

                    click.echo(
                        _format_rows(
                            rows, cfg.output_format, columns=["thought_id", "score", "essence"]
                        )
                    )

        _run(_recall())


# ------------------------------------------------------------------
# link
# ------------------------------------------------------------------


@cli.command()
@click.argument("from_id", metavar="FROM")
@click.argument("to_id", metavar="TO")
@click.option("--type", "edge_type_str", required=True, metavar="EDGE_TYPE", help="Edge type.")
@click.option(
    "--weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Relation strength, 0.0-1.0.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON object instead of a bare id.")
@click.pass_context
def link(
    ctx: click.Context,
    from_id: str,
    to_id: str,
    edge_type_str: str,
    weight: float,
    *,
    as_json: bool,
) -> None:
    """Create a typed edge FROM one thought TO another, and print its id.

    Builds an ``EdgeRecord`` and calls the public ``create_edge()`` — there is
    no public ``link()`` to call instead. Creates the resolved database if it
    does not already exist, printing the path to stderr when it does.

    The whole body below runs under :func:`_error_boundary`: a validation
    failure this function checks for itself keeps its own specific ``error``
    kind and exit code, but any *other* exception — a corrupt database, an
    unreadable ``--config`` — becomes the documented error object with a
    generic kind instead of a traceback.
    """
    with _error_boundary(as_json=as_json, command="link") as boundary_database:
        if edge_type_str not in _VALID_EDGE_TYPES:
            _fail(
                as_json=as_json,
                kind="invalid_edge_type",
                message=(
                    f"Invalid edge type {edge_type_str!r}; valid values: "
                    f"{', '.join(_VALID_EDGE_TYPES)}"
                ),
                code=2,
            )

        # Checked here, before the database is resolved or opened -- EdgeRecord's
        # own Pydantic field validation would catch the same out-of-range value,
        # but only once construction actually runs, deep inside `_link()` below
        # and after `_opened_full_store` has already created the database (and
        # its parent directories) for a `link` that was never going to succeed.
        # An unhandled ValidationError there would, before this boundary
        # existed, also have been a raw traceback rather than the documented
        # exit ``2`` -- it is now checked explicitly anyway, so it keeps its
        # own specific kind rather than falling through to the generic one.
        if not 0.0 <= weight <= 1.0:
            _fail(
                as_json=as_json,
                kind="invalid_weight",
                message=f"link: --weight must be between 0.0 and 1.0, got {weight}.",
                code=2,
            )

        cfg, resolved = _resolve_for_command(ctx, as_json=as_json)
        boundary_database.path = resolved.db_path

        async def _link() -> None:
            pre_existing = resolved.db_path.exists()
            async with _opened_full_store(resolved, cfg, create=True) as store:
                if not pre_existing:
                    click.echo(f"Created database: {resolved.db_path}", err=True)
                _report_resolution(cfg, resolved)

                edge = EdgeRecord(
                    edge_id=str(uuid.uuid4()),
                    from_thought_id=from_id,
                    to_thought_id=to_id,
                    edge_type=EdgeType(edge_type_str),
                    weight=weight,
                    created_cycle=0,
                )
                try:
                    created = await store.create_edge(edge)
                except ReferentialIntegrityError as exc:
                    # ReferentialIntegrityError is a public library class (see
                    # engrava.domain.exceptions), so a third-party subclass is
                    # realistic, not theoretical -- and .column /
                    # .referenced_id are as subclass-overridable as any other
                    # attribute. A review round found a subclass whose
                    # accessor raised silently downgraded to
                    # unexpected_error (losing the exit code docs/cli.md
                    # promises categorically for a missing FROM/TO), and one
                    # raising SystemExit escaped with no error object at
                    # all. The class hierarchy is still trustworthy even
                    # when a subclass's own attributes are not: a
                    # ReferentialIntegrityError subclass genuinely *is* a
                    # missing-reference failure, so it keeps
                    # missing_thought / exit 4 either way -- a subclass gets
                    # a fixed literal message, with nothing read off it at
                    # all -- not even through the hardened describer --
                    # because a believable-looking fabricated column or id
                    # (a review round produced exactly that) would be worse
                    # than not naming which endpoint is missing.
                    #
                    # An *exact* instance is not automatically safe either --
                    # .column and .referenced_id are still whatever values
                    # were assigned to them, exact-type or not (a review
                    # round found a hostile value in exactly this shape).
                    # So this reads both once and requires each rule below in
                    # order, never comparing or interpolating an unvalidated
                    # value:
                    #   1. both .column and .referenced_id are exact `str`
                    #      instances (`type(x) is str`, never `isinstance`,
                    #      and checked before anything else touches them);
                    #   2. .column is one of the two real column names this
                    #      command could actually violate;
                    #   3. .referenced_id equals *this invocation's own*
                    #      value for that endpoint (FROM_ID for
                    #      from_thought_id, TO_ID for to_thought_id) -- a
                    #      value store.create_edge() was never given cannot
                    #      legitimately be the one it rejected.
                    # The rendered message then names the endpoint from this
                    # invocation's own FROM_ID/TO_ID, not from the exception
                    # -- by rule 3 they are equal, but only the invocation's
                    # copy is ever interpolated. This confirms an id this
                    # command was actually asked to link is the one that is
                    # missing; it does not prove which endpoint is absent
                    # from the database, and a `str` equal to FROM_ID/TO_ID
                    # can still be a lie about *why* the write failed.
                    # Any rule failing falls back to the same fixed message
                    # the subclass branch uses.
                    if type(exc) is ReferentialIntegrityError:
                        column = exc.column
                        referenced_id = exc.referenced_id
                        if (
                            type(column) is str
                            and type(referenced_id) is str
                            and column in ("from_thought_id", "to_thought_id")
                        ):
                            endpoint_value = from_id if column == "from_thought_id" else to_id
                            if referenced_id == endpoint_value:
                                _fail(
                                    as_json=as_json,
                                    kind="missing_thought",
                                    message=(
                                        f"link: {column} {endpoint_value!r} "
                                        "does not reference an existing thought."
                                    ),
                                    code=4,
                                )
                        _fail(
                            as_json=as_json,
                            kind="missing_thought",
                            message=_MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE,
                            code=4,
                        )
                    _fail(
                        as_json=as_json,
                        kind="missing_thought",
                        message=_MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE,
                        code=4,
                    )

                if as_json:
                    payload = {
                        "schema": _LINK_SCHEMA,
                        "edge_id": created.edge_id,
                        "from_thought_id": created.from_thought_id,
                        "to_thought_id": created.to_thought_id,
                        "edge_type": created.edge_type.value,
                        "weight": created.weight,
                    }
                    click.echo(json.dumps(payload, ensure_ascii=False))
                else:
                    click.echo(created.edge_id)

        _run(_link())

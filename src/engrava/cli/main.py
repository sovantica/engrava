"""``engrava`` — CLI entry point for engrava.

Provides sub-commands: info, verify, query, snapshot, restore, gc,
migrate, export.

Usage::

    engrava --db ./my.db info
    engrava --db ./my.db verify
    engrava query "SELECT * FROM thought WHERE lifecycle_status = 'ACTIVE'"
    engrava snapshot -o backup.jsonl
    engrava restore -i backup.jsonl
    engrava gc
    engrava migrate
    engrava export -o thoughts.json
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from engrava.cli.config import EngravaCLIConfig
from engrava.cli.exception_reporting import _describe_exception, _frame_only_stack
from engrava.cli.snapshot_records import (
    CoreTable,
    MetadataRecord,
    TableRecord,
    parse_snapshot_record,
)
from engrava.config import (
    EmbeddingConfig,
    ServicesConfig,
    resolve_embedding_provider,
)
from engrava.config_validation import ConfigError
from engrava.domain.protocols.hooks import MindQLExtension
from engrava.infrastructure.sqlite.engrava_core import (
    CORE_SCHEMA_HEAD_VERSION,
    SqliteEngravaCore,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
    from typing import TextIO

    import aiosqlite

    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol

logger = logging.getLogger(__name__)

# Re-embedding thought IDs are flushed in batches of this size so restore memory
# stays bounded by the batch rather than by the total number of thoughts.
_REEMBED_BATCH_SIZE = 128

# sqlite3.IntegrityError.sqlite_errorcode values the journalled-merge collision
# gate (see `_import_records_to_db`) treats as a refused collision, rather than
# an unrelated integrity failure it must let propagate unchanged (a foreign-key
# violation, SQLITE_CONSTRAINT_FOREIGNKEY = 787, is one such unrelated case).
# The exception's *message* is not used to tell these apart: SQLite's own text
# reads "UNIQUE constraint failed" even for a primary-key violation, since a
# ``PRIMARY KEY`` is implemented as a ``UNIQUE`` index internally.
_SQLITE_CONSTRAINT_PRIMARYKEY = 1555
_SQLITE_CONSTRAINT_UNIQUE = 2067
_JOURNAL_GATE_CONSTRAINT_CODES = frozenset(
    {_SQLITE_CONSTRAINT_PRIMARYKEY, _SQLITE_CONSTRAINT_UNIQUE}
)

_DISABLE_EXTENSIONS_META_KEY = "engrava_disable_extensions"
_FALSE_ENV_FLAG_VALUES = frozenset({"", "0", "false", "no", "off"})

# Core tables in dependency order (thought first, dependents after). Typed as
# CoreTable so every table identifier that reaches SQL comes from the enum.
_CORE_TABLES: tuple[CoreTable, ...] = (
    CoreTable.THOUGHT,
    CoreTable.EDGE,
    CoreTable.EMBEDDING,
    CoreTable.ACTION,
)

# Reverse order for safe deletion (dependents first).
_CORE_TABLES_DELETE_ORDER: tuple[CoreTable, ...] = (
    CoreTable.ACTION,
    CoreTable.EMBEDDING,
    CoreTable.EDGE,
    CoreTable.THOUGHT,
)

# The journal is deliberately not a CoreTable member: that enum is the
# allow-list of tables a *snapshot* may contain (see
# ``engrava.cli.snapshot_records.CoreTable``), and a snapshot never carries
# journal rows. But ``restore --clear`` still has to remove ``journal_entry``
# alongside the four core tables above -- leaving it in place would let the
# cleared store's data and its append-only journal describe two different
# histories, and ``verify_journal()`` would keep reporting that mismatched
# chain as valid. It is deleted by its own fixed literal below rather than
# through this enum, since the enum exists to keep every *snapshot-derived*
# SQL identifier off the trust boundary -- a concern that does not apply to a
# name that is never read from snapshot input.

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _close_quietly(conn: Any) -> None:  # noqa: ANN401
    """Close *conn*, logging rather than raising if the close itself fails.

    Cleanup code that closes a connection while another exception -- or a
    cancellation -- is already propagating must not let a failure in the
    close itself replace what the caller actually needs to see: a bare
    ``raise`` after an unconditional ``await conn.close()`` only re-raises
    the original error when that close *succeeds*. If the close itself
    raises, its exception becomes the one that propagates and the original
    -- a ``sqlite3.DatabaseError``, a ``ClickException``, an
    ``asyncio.CancelledError`` -- is lost. A close failure is real
    information, but it belongs logged underneath the original error, not
    raised in front of it. Every place in this module that closes a
    connection during cleanup (as opposed to on the ordinary success path,
    where a close failure is the only thing to report) goes through this
    rather than re-deriving the same try/except. The infrastructure layer
    has the same rule under the same name in
    :mod:`engrava.infrastructure.sqlite.engrava_core` -- not shared as one
    function across the CLI/infrastructure boundary, but copied rather
    than re-derived.

    ``await conn.close()`` is itself a suspension point, so a bare
    ``try/except Exception`` around it has the identical gap this whole
    helper exists to close: a cancellation arriving while the close is
    in flight is a ``BaseException``, skips that handler, and can leave
    the close abandoned mid-way with aiosqlite's non-daemon worker thread
    still alive. The close is run as its own task and shielded so that
    cancelling *this* coroutine does not also cancel the close itself;
    the shield alone would not be enough, though, since it only stops the
    cancellation from reaching the close, not from being re-thrown into
    this coroutine before the close finishes running. So on cancellation
    this explicitly awaits the same task again -- now cancellation-proof,
    since a second throw only happens on an explicit second
    ``cancel()`` -- to hold this coroutine (and so whatever awaits it,
    keeping the event loop alive) open until the real close has actually
    completed, before letting the cancellation propagate.

    **The warning below no longer passes ``exc_info=True``.** A later
    verification round found this function reached by the memory verbs'
    own bare/default store tier (``remember`` / ``recall`` / ``link`` with
    no ``--config``, via ``_opened_db`` above) -- not just the other
    built-ins this module already owned -- and that ``exc_info=True`` had
    the same live defect here that a previous round had already fixed at
    the ``--config``-tier cleanup site in
    :mod:`engrava.cli.memory_commands`: it asks the standard library's
    traceback formatter to render the close exception through its own
    overridable ``__str__``, and that formatter wraps its own rendering in
    a bare ``except``, so a real OS ``SIGINT`` -- and separately, a
    formatter raising ``SystemExit`` -- arriving during that render was
    absorbed there instead of propagating, both verified live against this
    exact function. Because the close here runs as a separately scheduled,
    shielded task (see above) rather than inline inside the caller's
    ``except`` block, ``exc_info=True``'s chain-walk had nothing to walk
    beyond the close exception itself -- no implicit ``__context__`` links
    it to whatever this coroutine was cleaning up after, unlike the
    ``--config`` tier's inline ``await store.close()`` -- so the defect
    here was one unwanted render, not the multi-exception cascade found
    there. ``_frame_only_stack`` and ``_describe_exception``, shared with
    that module through :mod:`engrava.cli.exception_reporting` (the two
    modules import each other and neither can define these at module level
    without a cycle -- see that module's own docstring), replace it:
    the frame-only stack gives "where" without touching the close
    exception's own formatting at all, and the description gives "why" --
    an ordinary ``PermissionError``, a full disk, a locked file -- through
    the same single, guarded, non-absorbing read
    :func:`~engrava.cli.exception_reporting._describe_exception` always
    performs, once, never a second render of anything.

    Args:
        conn: The aiosqlite connection to close.

    """
    try:
        close_task = asyncio.ensure_future(conn.close())
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        try:
            await close_task
        except Exception as close_exc:  # noqa: BLE001
            logger.warning(
                "Error closing connection during cleanup: %s; stack (file:line "
                "in function, not a full exception-chain rendering):\n%s",
                _describe_exception(close_exc),
                _frame_only_stack(close_exc),
            )
        raise
    except Exception as close_exc:  # noqa: BLE001
        logger.warning(
            "Error closing connection during cleanup: %s; stack (file:line "
            "in function, not a full exception-chain rendering):\n%s",
            _describe_exception(close_exc),
            _frame_only_stack(close_exc),
        )
        # A later round found a real OS SIGINT delivered here -- after the
        # warning above was already logged -- absorbed instead of aborting.
        # `asyncio.run()`'s own SIGINT handler (see `asyncio.runners.Runner`)
        # does not raise anything into this coroutine: on the first Ctrl-C it
        # only calls the main task's `cancel()`, which *requests* a
        # `CancelledError` but only actually throws one in at this
        # coroutine's *next* suspension point. Nothing above this line
        # suspends -- `_describe_exception`, `_frame_only_stack`, and
        # `logger.warning` are all synchronous -- so a synchronous function
        # simply cannot observe a pending cancellation at all; without an
        # `await` here, this function would return normally, the caller's
        # `raise` would re-raise the original error object, and the request
        # to cancel would be silently dropped once this task finishes. This
        # `await asyncio.sleep(0)` is a real suspension point purely to give
        # that pending cancellation somewhere to be delivered -- it does not
        # sleep in the timer sense, it just returns control to the event
        # loop for one iteration, which is exactly when `Task.__step` checks
        # for and throws in a cancellation that was requested while this
        # coroutine was running synchronously.
        await asyncio.sleep(0)


async def _open_db(cfg: EngravaCLIConfig) -> Any:  # noqa: ANN401
    """Open an aiosqlite connection with WAL + row_factory.

    ``aiosqlite.connect()`` succeeding only means the background worker
    thread started — a corrupt or truncated file is not discovered until
    the first ``PRAGMA`` actually runs against it, below. If that fails,
    the connection is closed before the error propagates: aiosqlite's
    connection worker thread is not a daemon and only stops when
    ``close()`` sends it the shutdown sentinel, so an open-but-never-closed
    connection left behind by a raised exception here would hang interpreter
    shutdown indefinitely instead of exiting on the error. Every command
    reaches this through :func:`_opened_db` rather than calling it directly,
    so this is the one place that has to get the connect-time failure right.

    Args:
        cfg: Resolved CLI config with db_path.

    Returns:
        An open aiosqlite Connection.

    Raises:
        sqlite3.DatabaseError: If the file is not a valid SQLite database
            (connection is closed first).

    """
    import aiosqlite  # noqa: PLC0415

    conn = await aiosqlite.connect(str(cfg.db_path))
    try:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")
    except BaseException:
        await _close_quietly(conn)
        raise
    return conn


@asynccontextmanager
async def _opened_db(cfg: EngravaCLIConfig) -> AsyncIterator[Any]:
    """Open a connection and guarantee it closes, however the block exits.

    Acquiring the connection and entering the protected block are one
    syntactic step at the call site (``async with _opened_db(cfg) as conn:``),
    so no statement — a store constructor, a schema-version-gate check,
    ``ensure_schema()`` — can sit between a successful open and the
    guarantee that closes it. A hand-written ``try/finally`` at the call
    site cannot make that promise: it only protects what is written after
    it, and a failure in a statement placed before it (by oversight, or by
    a later edit) leaks the connection exactly as an absent ``finally``
    would. This closes on normal return, on any raised exception — a
    ``sqlite3.DatabaseError`` from a corrupt file, a ``ClickException``, a
    ``SystemExit`` from ``sys.exit()`` — and on cancellation.

    **What a close failure itself does differs by which of those it is.**
    If the command body already raised (or was cancelled), that is what
    the caller needs to see, so a failure in this closing call is logged
    and swallowed rather than replacing it. If the body succeeded, a
    close failure is not secondary to anything — it is the only error
    there is, so it propagates normally. A single unconditional
    ``finally: await conn.close()`` cannot draw that distinction: it
    would let a genuine close failure on the success path be silently
    swallowed, so the command would print success and exit ``0``.

    Yields:
        The open aiosqlite connection from :func:`_open_db`.

    """
    conn = await _open_db(cfg)
    try:
        yield conn
    except BaseException:
        # The command body raised (or was cancelled) -- that is what the
        # caller needs to see, so a close failure here is secondary and
        # goes through ``_close_quietly`` rather than replacing it.
        await _close_quietly(conn)
        raise
    else:
        # The command body succeeded. A close failure here is not secondary
        # to anything -- it is the *only* error there is, so it must
        # propagate normally rather than being logged and swallowed by
        # ``_close_quietly``. An unconditional ``finally: await
        # _close_quietly(conn)`` would silently turn a genuine close
        # failure into a command that prints success and exits 0.
        await conn.close()


# ------------------------------------------------------------------
# Schema-version gate
# ------------------------------------------------------------------
#
# ``ensure_schema`` is reached from only three built-in command names —
# ``migrate``, ``restore``, and ``snapshot`` in service mode — so every other
# built-in (``info``, ``verify``, ``query``, ``export``, plain ``snapshot``,
# and ``gc``) opens through ``_open_db`` and never learns whether the
# database it is about to act on is even at the version it understands. This
# gate classifies every built-in command as destructive or read and checks it
# against the database's stamped ``user_version`` accordingly: a destructive
# command refuses on anything but a head schema, a read command warns and
# proceeds on a behind schema but still refuses above head, and ``query``
# classifies by its parsed command rather than by the CLI command name.


async def _read_schema_version(conn: Any) -> int:  # noqa: ANN401
    """Read a connection's stamped ``user_version`` (0 if never set).

    A plain ``PRAGMA`` read — it never migrates the database, which is the
    point: the gate's own check must not itself be the implicit migration no
    command is meant to perform.

    Args:
        conn: An open connection (aiosqlite or the ``sqlite3`` it wraps).

    Returns:
        The stamped ``user_version``.

    """
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


def _behind_schema_warning(version: int, *, command: str) -> str:
    """Build the stderr warning for a read-classified command on a behind schema."""
    return (
        f"Warning: database schema is at version {version}, behind this "
        f"engrava build's head version ({CORE_SCHEMA_HEAD_VERSION}). "
        f"'{command}' will run against the schema as stored — run "
        "'engrava migrate' to bring it current."
    )


def _behind_schema_refusal(version: int, *, command: str) -> str:
    """Build the refusal message for a destructive command on a behind schema."""
    return (
        f"Database schema is at version {version}; this engrava build's head "
        f"version is {CORE_SCHEMA_HEAD_VERSION}. Run 'engrava migrate' before "
        f"running '{command}' on it."
    )


def _ahead_schema_refusal(version: int, *, command: str) -> str:
    """Build the refusal message for any command on a newer-than-head schema."""
    return (
        f"Database schema is at version {version}, newer than this engrava "
        f"build's head version ({CORE_SCHEMA_HEAD_VERSION}). Refusing to run "
        f"'{command}' — upgrade engrava before opening this database."
    )


def _apply_read_schema_gate_for_version(version: int, *, command: str) -> None:
    """Apply the schema-version gate for a read-classified built-in command.

    Warns and proceeds on a behind schema (refusing would trade this defect
    for a worse one — a pending migration blocking an ordinary read); refuses
    unconditionally on a schema newer than this build's head, which it cannot
    understand at all.

    Args:
        version: The database's stamped ``user_version``.
        command: The command name, named in the message.

    """
    if version > CORE_SCHEMA_HEAD_VERSION:
        click.echo(_ahead_schema_refusal(version, command=command), err=True)
        sys.exit(1)
    if version < CORE_SCHEMA_HEAD_VERSION:
        click.echo(_behind_schema_warning(version, command=command), err=True)


def _apply_destructive_schema_gate_for_version(version: int, *, command: str) -> None:
    """Apply the schema-version gate for a destructive built-in command.

    Refuses on any schema that is not exactly head — below head because a
    destructive operation must not delete rows through an engine that does
    not understand the schema it is deleting from (the "gc never migrates"
    defect reached a user through exactly this gap), and above head because
    this build cannot understand it either.

    Args:
        version: The database's stamped ``user_version``.
        command: The command name, named in the message.

    """
    if version > CORE_SCHEMA_HEAD_VERSION:
        click.echo(_ahead_schema_refusal(version, command=command), err=True)
        sys.exit(1)
    if version < CORE_SCHEMA_HEAD_VERSION:
        click.echo(_behind_schema_refusal(version, command=command), err=True)
        sys.exit(1)


async def _apply_read_schema_gate(conn: Any, *, command: str) -> None:  # noqa: ANN401
    """Read-then-gate convenience wrapper — see :func:`_apply_read_schema_gate_for_version`."""
    _apply_read_schema_gate_for_version(await _read_schema_version(conn), command=command)


async def _apply_destructive_schema_gate(conn: Any, *, command: str) -> None:  # noqa: ANN401
    """Read-then-gate convenience wrapper.

    See :func:`_apply_destructive_schema_gate_for_version`.
    """
    _apply_destructive_schema_gate_for_version(await _read_schema_version(conn), command=command)


def _run(coro: Any) -> Any:  # noqa: ANN401
    """Run an async coroutine from sync CLI context.

    Args:
        coro: Awaitable to execute.

    Returns:
        The coroutine result.

    """
    return asyncio.run(coro)


def _format_rows(
    rows: Sequence[Mapping[str, object]],
    fmt: str,
    *,
    columns: list[str] | None = None,
) -> str:
    """Format a list of row-dicts for display.

    Args:
        rows: List of row dictionaries.
        fmt: Output format (json/table/csv).
        columns: Optional column order for table/csv output.

    Returns:
        Formatted string.

    """
    if fmt == "json":
        return json.dumps(rows, indent=2, default=str, ensure_ascii=False)

    if not rows:
        return "(no rows)"

    cols = columns or list(rows[0].keys())

    if fmt == "csv":
        import csv  # noqa: PLC0415
        import io  # noqa: PLC0415

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue().rstrip()

    # table format
    col_widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            val = str(row.get(c, ""))
            col_widths[c] = max(col_widths[c], len(val))

    header = "  ".join(c.ljust(col_widths[c]) for c in cols)
    sep = "  ".join("-" * col_widths[c] for c in cols)
    lines = [header, sep]
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in cols)
        lines.append(line)
    return "\n".join(lines)


def _load_mindql_extensions() -> dict[str, MindQLExtension]:
    """Discover MindQL extensions from installed entry points.

    Scans ``engrava.extensions`` entry point group for extension
    manifests that provide MindQL extension commands.

    Returns:
        Mapping of command name to ``MindQLExtension``.

    """
    registry: dict[str, MindQLExtension] = {}
    eps = entry_points(group="engrava.extensions")

    for ep in eps:
        try:
            manifest = ep.load()
            # Manifest may be an ExtensionManifest instance or callable
            if callable(manifest) and not hasattr(manifest, "mindql_extensions"):
                manifest = manifest()
            for ext in getattr(manifest, "mindql_extensions", []):
                registry[ext.command_name] = ext
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load extension %s", ep.name, exc_info=True)

    return registry


def _discover_extension_commands() -> list[click.Command]:
    """Discover CLI commands from installed extension entry points.

    Scans ``engrava.cli`` entry point group for click commands
    or groups registered by extension packages.

    Returns:
        List of click commands to add to the main CLI group.

    """
    commands: list[click.Command] = []
    eps = entry_points(group="engrava.cli")

    for ep in eps:
        try:
            cmd = ep.load()
            if isinstance(cmd, click.Command):
                commands.append(cmd)
            elif callable(cmd):
                result = cmd()
                if isinstance(result, list):
                    commands.extend(item for item in result if isinstance(item, click.Command))
                elif isinstance(result, click.Command):
                    commands.append(result)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load CLI extension %s", ep.name, exc_info=True)

    return commands


class _ExtensionAwareGroup(click.Group):
    """Click group that discovers extension commands only when needed.

    Built-in commands resolve without importing third-party entry points. Global
    help discovers extensions so they remain visible by default, while the
    ``--no-extensions`` control suppresses discovery and hides commands loaded by
    an earlier in-process invocation.
    """

    _extension_commands_loaded = False
    _extension_command_names: frozenset[str] = frozenset()

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Capture the disable control before eager options such as help run."""
        env_value = os.environ.get("ENGRAVA_DISABLE_EXTENSIONS")
        disabled_by_env = (
            env_value is not None and env_value.strip().lower() not in _FALSE_ENV_FLAG_VALUES
        )
        ctx.meta[_DISABLE_EXTENSIONS_META_KEY] = "--no-extensions" in args or disabled_by_env
        return super().parse_args(ctx, args)

    @staticmethod
    def _extensions_disabled(ctx: click.Context) -> bool:
        """Return whether extension loading is disabled for this invocation."""
        return bool(
            ctx.meta.get(_DISABLE_EXTENSIONS_META_KEY, False)
            or ctx.params.get("disable_extensions", False)
        )

    def _register_extension_commands(self) -> None:
        """Discover and register installed extension commands once."""
        if self._extension_commands_loaded:
            return

        self._extension_commands_loaded = True
        loaded_names: set[str] = set()
        for command in _discover_extension_commands():
            command_name = command.name
            if command_name is None:
                logger.warning("Ignoring unnamed CLI extension command")
                continue
            if command_name in self.commands:
                logger.warning(
                    "Ignoring CLI extension command %r because the name is already registered",
                    command_name,
                )
                continue
            self.add_command(command)
            loaded_names.add(command_name)
        self._extension_command_names = frozenset(loaded_names)

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List built-ins and, unless disabled, discovered extension commands."""
        disabled = self._extensions_disabled(ctx)
        if not disabled:
            self._register_extension_commands()

        command_names = super().list_commands(ctx)
        if not disabled:
            return command_names
        return [name for name in command_names if name not in self._extension_command_names]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Resolve built-ins first and load entry points only for unknown commands."""
        disabled = self._extensions_disabled(ctx)
        command = super().get_command(ctx, cmd_name)
        if command is not None and (not disabled or cmd_name not in self._extension_command_names):
            return command
        if disabled:
            return None

        self._register_extension_commands()
        return super().get_command(ctx, cmd_name)


def _configure_verbose_logging(ctx: click.Context) -> None:
    """Emit DEBUG logs from Engrava for the lifetime of one CLI invocation.

    Args:
        ctx: Root Click context used to restore logger state on close.

    """
    package_logger = logging.getLogger("engrava")
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    package_logger.addHandler(handler)
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False

    def _restore_logging() -> None:
        package_logger.removeHandler(handler)
        handler.close()
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate

    ctx.call_on_close(_restore_logging)


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

#: Subcommands that read ``ctx.obj["services_config"]`` / ``["default_embeddings"]``
#: — see the ``cli()`` group callback below, which loads a ``--config`` file
#: only when the invoked subcommand is one of these two.
_SERVICES_CONFIG_COMMANDS = frozenset({"snapshot", "restore"})


def _reject_option_shaped_value(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Refuse a value that looks like another option instead of silently taking it.

    ``--db`` and ``--config`` are plain string options: like ``argparse``,
    Click's parser treats whatever token immediately follows one as its
    value -- even when that token itself starts with ``-`` and spells out
    another known flag. ``engrava --db --json remember "x"`` therefore
    parsed as ``--db`` bound to the literal string ``"--json"``: it created
    a database called ``--json``, stored the thought, and exited ``0``
    without the caller's requested ``--json`` output ever taking effect --
    a silent write to the wrong place that defeats the very flag meant to
    make the outcome machine-readable.

    A previous revision of the documentation called this "ordinary
    argparse behaviour" and left it alone. That claim was false: running
    the equivalent ``argparse`` program on the same value rejects it with
    "expected one argument" rather than accepting it. This closes the gap
    by rejecting any ``--db`` / ``--config`` value starting with ``-`` --
    the whole shape of the ambiguity, not just the ``--json`` instance of
    it -- as a Click-level usage error, before either option's value ever
    reaches a memory-verb's own resolution logic. A caller who genuinely
    needs such a path can disambiguate it the usual shell way, by
    prefixing it (``./--json``).

    Args:
        ctx: The option's Click context, forwarded to ``BadParameter`` for
            its usage-message formatting.
        param: The option being validated (``--db`` or ``--config``).
        value: The parsed value, or ``None`` when the option was omitted.

    Returns:
        ``value`` unchanged, when it does not look like another option.

    Raises:
        click.BadParameter: ``value`` starts with ``-``.

    """
    if value is not None and value.startswith("-"):
        message = (
            f"{value!r} looks like an option, not a path. "
            f"Prefix it (e.g. './{value}') if this is really the intended path."
        )
        raise click.BadParameter(message, ctx=ctx, param=param)
    return value


@click.group(cls=_ExtensionAwareGroup)
@click.option(
    "--db",
    "db_path",
    default=None,
    callback=_reject_option_shaped_value,
    help="Path to SQLite database.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table", "csv"]),
    default="table",
    help="Output format.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.option(
    "--no-extensions",
    "disable_extensions",
    is_flag=True,
    envvar="ENGRAVA_DISABLE_EXTENSIONS",
    help="Disable installed CLI and MindQL extension entry points.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    callback=_reject_option_shaped_value,
    help="Path to engrava.yaml (also ENGRAVA_CONFIG env).",
)
@click.pass_context
def cli(
    ctx: click.Context,
    db_path: str | None,
    output_format: str,
    *,
    verbose: bool,
    disable_extensions: bool,
    config_path: str | None,
) -> None:
    """Engrava — standalone thought-graph CLI.

    Manage an engrava SQLite database: inspect, query, snapshot,
    restore, garbage-collect, migrate, and export.
    """
    ctx.ensure_object(dict)
    cfg = EngravaCLIConfig.resolve(
        db_path=db_path,
        output_format=output_format,
        verbose=verbose,
        config_path=config_path,
        disable_extensions=disable_extensions,
    )
    ctx.obj["config"] = cfg

    if cfg.verbose:
        _configure_verbose_logging(ctx)
        logger.debug("Verbose logging enabled")

    # Pre-load services config for --service default resolution. Gated on the
    # two commands that actually read ``services_config`` / ``default_embeddings``
    # off ``ctx.obj`` (``snapshot`` and ``restore`` — see their own bodies
    # below): every other command, including the memory verbs (they resolve
    # their own database through engrava.cli.store_resolution and never touch
    # either value), used to pay for this load — and its failure — anyway,
    # since a group callback runs before Click even knows which subcommand's
    # options to parse. That made an explicit --db unable to save any command
    # from a broken --config: this callback raised before a subcommand's own
    # precedence logic ever ran, so --db's documented "explicit always wins"
    # was true for the *chosen database* but false for whether the command
    # ran at all. Scoping the load to the two commands that need it restores
    # that precedence for everything else, while a broken --config given to
    # snapshot/restore themselves is still reported here, once, as a clean
    # CLI error instead of a traceback.
    services_cfg = None
    default_embeddings = None
    if ctx.invoked_subcommand in _SERVICES_CONFIG_COMMANDS and cfg.config_path is not None:
        from engrava.config import load_config  # noqa: PLC0415

        try:
            ms_config = load_config(cfg.config_path)
        except ConfigError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        services_cfg = ms_config.services
        default_embeddings = ms_config.embeddings
    ctx.obj["services_config"] = services_cfg
    ctx.obj["default_embeddings"] = default_embeddings

    # Whether --db (or ENGRAVA_DB) was actually supplied, as opposed to
    # cfg.db_path holding the CLI's own hardcoded default. EngravaCLIConfig
    # folds "explicit" and "defaulted" into one value once resolved, so the
    # memory verbs' shared store-resolution helper (see
    # engrava.cli.store_resolution) needs this computed the same way here,
    # from the raw option, mirroring EngravaCLIConfig.resolve's own
    # truthiness check rather than re-deriving a different one.
    ctx.obj["db_explicit"] = bool(db_path) or bool(os.environ.get("ENGRAVA_DB"))


# ------------------------------------------------------------------
# info
# ------------------------------------------------------------------


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show a metrics snapshot for the current database."""
    cfg: EngravaCLIConfig = ctx.obj["config"]

    async def _info() -> None:
        if not cfg.db_path.exists():
            click.echo(f"Database not found: {cfg.db_path}")
            sys.exit(1)

        async with _opened_db(cfg) as conn:
            await _apply_read_schema_gate(conn, command="info")
            store = SqliteEngravaCore(conn)
            metrics = await store.metrics()
            stats: dict[str, Any] = {
                "db_path": str(cfg.db_path.resolve()),
                **asdict(metrics),
            }

            if cfg.output_format == "json":
                click.echo(json.dumps(stats, indent=2))
            else:
                click.echo(f"Database: {stats['db_path']}")
                click.echo(f"Schema version: {stats['schema_version']}")
                click.echo(
                    f"Thoughts: {stats['thoughts']['total']} ({stats['thoughts']['by_type']})"
                )
                click.echo(f"Edges: {stats['edges']['total']} ({stats['edges']['by_type']})")
                click.echo(f"Storage: {stats['storage']['total_bytes']} bytes")
                click.echo(
                    "Search latency: "
                    f"n={stats['search_latency']['sample_count']} "
                    f"p50={stats['search_latency']['p50_ms']:.1f}ms "
                    f"p95={stats['search_latency']['p95_ms']:.1f}ms "
                    f"p99={stats['search_latency']['p99_ms']:.1f}ms"
                )

    _run(_info())


# ------------------------------------------------------------------
# verify
# ------------------------------------------------------------------


@cli.command()
@click.pass_context
def verify(ctx: click.Context) -> None:
    """Verify the audit journal's hash chain for the current database.

    Walks every recorded ``journal_entry`` in sequence order, recomputes
    each SHA-256 hash, and checks the parent-hash linkage. The chain is
    verified regardless of whether journaling is currently enabled, so a
    journal recorded in an earlier session is still auditable.

    Exit code is ``0`` when the chain verifies and ``1`` when it does not
    (or when the database is missing).
    """
    cfg: EngravaCLIConfig = ctx.obj["config"]

    async def _verify() -> None:
        if not cfg.db_path.exists():
            click.echo(f"Database not found: {cfg.db_path}")
            sys.exit(1)

        async with _opened_db(cfg) as conn:
            await _apply_read_schema_gate(conn, command="verify")
            store = SqliteEngravaCore(conn)
            result = await store.verify_journal()

            if cfg.output_format == "json":
                click.echo(json.dumps(asdict(result), indent=2))
            elif result.valid:
                click.echo(f"Journal integrity OK — {result.entries_checked} entries verified.")
            else:
                click.echo(
                    f"Journal integrity FAILED at sequence {result.first_invalid_sequence}: "
                    f"{result.error_message} "
                    f"({result.entries_checked} entries checked)."
                )

            if not result.valid:
                sys.exit(1)

    _run(_verify())


# ------------------------------------------------------------------
# query
# ------------------------------------------------------------------


@cli.command()
@click.argument("mql")
@click.pass_context
def query(ctx: click.Context, mql: str) -> None:
    """Execute a MindQL query and display results.

    Accepts FIND, COUNT, SELECT, or registered extension commands.

    Examples::

        engrava query "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"
        engrava query "COUNT thoughts WHERE priority = 'P1'"
        engrava query "SELECT thought_id, essence FROM thought LIMIT 5"
    """
    cfg: EngravaCLIConfig = ctx.obj["config"]

    async def _query() -> None:
        if not cfg.db_path.exists():
            click.echo(f"Database not found: {cfg.db_path}")
            sys.exit(1)

        from engrava.mindql.executor import MindQLExecutor  # noqa: PLC0415
        from engrava.mindql.parser import (  # noqa: PLC0415
            MindQLCommand,
            MindQLParseError,
            parse,
        )

        async with _opened_db(cfg) as conn:
            try:
                # Gather extension commands from loaded extensions
                extensions = _load_mindql_extensions() if cfg.extensions_enabled else {}
                known_names = set(extensions.keys())

                try:
                    parsed = parse(mql, known_extensions=known_names)
                except MindQLParseError as exc:
                    click.echo(f"Parse error: {exc}", err=True)
                    sys.exit(1)

                # The schema-version gate classifies the *parsed* command, not
                # the CLI command name — FIND/COUNT/SELECT are reads (warn and
                # attempt on a behind schema); EXTENSION can write (there is
                # today no read-only accessor for an extension handler to run
                # under, so it is refused on a behind schema like any other
                # destructive operation). Every classification also refuses a
                # newer-than-head schema outright.
                schema_version = await _read_schema_version(conn)
                if parsed.command is MindQLCommand.EXTENSION:
                    _apply_destructive_schema_gate_for_version(schema_version, command="query")
                else:
                    _apply_read_schema_gate_for_version(schema_version, command="query")

                executor = MindQLExecutor(conn, extensions=extensions)
                result = await executor.execute(parsed)
                click.echo(_format_rows(result.rows, cfg.output_format, columns=result.columns))
            except MindQLParseError as exc:
                click.echo(f"Query error: {exc}", err=True)
                sys.exit(1)

    _run(_query())


# ------------------------------------------------------------------
# snapshot (export to JSONL)
# ------------------------------------------------------------------


async def _export_db_to_jsonl(conn: Any, out: Path) -> int:  # noqa: ANN401
    """Export all core tables from a connection to a JSONL file.

    Writes metadata header, then thought/edge/embedding/action records.

    Args:
        conn: Open aiosqlite connection.
        out: Output file path.

    Returns:
        Total number of records exported.

    """
    total = 0

    # Write metadata header.
    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    schema_version = int(row[0]) if row else 0

    # Read embedding model lock if present.
    model_name: str | None = None
    dimension: int | None = None
    try:
        cursor = await conn.execute(
            "SELECT value FROM _metadata WHERE key = 'embedding_model_name'"
        )
        mrow = await cursor.fetchone()
        if mrow:
            model_name = mrow[0]
        cursor = await conn.execute("SELECT value FROM _metadata WHERE key = 'embedding_dimension'")
        drow = await cursor.fetchone()
        if drow:
            dimension = int(drow[0])
    except Exception:  # noqa: BLE001
        logger.debug("_metadata table not available for snapshot headers")

    with out.open("w", encoding="utf-8") as f:
        meta_record: dict[str, Any] = {
            "_type": "metadata",
            "schema_version": schema_version,
        }
        if model_name is not None:
            meta_record["embedding_model_name"] = model_name
        if dimension is not None:
            meta_record["embedding_dimension"] = dimension
        f.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
        total += 1

        _select_all_sql = {
            CoreTable.THOUGHT: "SELECT * FROM thought",
            CoreTable.EDGE: "SELECT * FROM edge",
            CoreTable.EMBEDDING: "SELECT * FROM embedding",
            CoreTable.ACTION: "SELECT * FROM action",
        }
        for table in _CORE_TABLES:
            cursor = await conn.execute(_select_all_sql[table])
            keys = [desc[0] for desc in cursor.description] if cursor.description else []
            async for row in cursor:
                record: dict[str, Any] = {}
                for i, key in enumerate(keys):
                    val = row[i]
                    if isinstance(val, bytes):
                        import base64  # noqa: PLC0415

                        val = base64.b64encode(val).decode("ascii")
                    record[key] = val
                line = json.dumps(
                    {"_type": table.value, "data": record},
                    default=str,
                    ensure_ascii=False,
                )
                f.write(line + "\n")
                total += 1

    return total


@cli.command()
@click.option("-o", "--output", "output_path", default=None, help="Output JSONL file path.")
@click.option(
    "--service",
    "service_name",
    default=None,
    help="Service name (multi-service mode).",
)
@click.pass_context
def snapshot(ctx: click.Context, output_path: str | None, service_name: str | None) -> None:
    """Export the entire database to a JSONL snapshot file.

    Each line is a JSON object with ``{_type, ...}`` (metadata header)
    or ``{_type, data}`` (thought/edge/embedding/action records).

    In multi-service mode, use ``--service`` to target a specific service.
    """
    cfg: EngravaCLIConfig = ctx.obj["config"]
    services_cfg: ServicesConfig | None = ctx.obj.get("services_config")

    # Resolve default service from config if --service not given.
    effective_service = service_name
    if effective_service is None and services_cfg is not None:
        effective_service = services_cfg.default_service

    # Validate any resolved service name — an explicit --service (including an
    # empty string, which is falsy) or a config default — up front so a malformed
    # value is a clean ClickException rather than a silent fall-through to the
    # single-database path or a later traceback.
    if effective_service is not None:
        _require_valid_cli_service_name(effective_service)

    async def _snapshot() -> None:
        if effective_service:
            from engrava.infrastructure.service_manager import (  # noqa: PLC0415
                EngravaManager,
            )

            data_dir = services_cfg.data_dir if services_cfg else cfg.db_path.parent
            manager = EngravaManager(
                data_dir=data_dir,
                services_config=services_cfg,
            )
            if not manager.service_exists(effective_service):
                click.echo(
                    f"Service {effective_service!r} not found "
                    f"(no database at {manager._service_db_path(effective_service)}).",  # noqa: SLF001
                    err=True,
                )
                sys.exit(1)
            try:
                # snapshot is read-classified, so a behind target is warned
                # about and attempted rather than refused — but attempting
                # must not itself migrate the service database, which calling
                # manager.get_store() on a behind target would. peek_schema_version()
                # reads the stamped version without migrating it.
                existing_version = await manager.peek_schema_version(effective_service)
                if existing_version is not None:
                    _apply_read_schema_gate_for_version(existing_version, command="snapshot")
                # service_exists() above already confirmed this service has a
                # database, so migrate=False opens it exactly as stored —
                # never a silent implicit migration on a behind target.
                store = await manager.get_store(effective_service, migrate=False)
                db = store._db  # noqa: SLF001
                out = (
                    Path(output_path)
                    if output_path
                    else data_dir / f"{effective_service}.snapshot.jsonl"
                )
                total = await _export_db_to_jsonl(db, out)
                click.echo(f"Exported {total} records from service {effective_service!r} to {out}")
            finally:
                await manager.close_all()
        else:
            if not cfg.db_path.exists():
                click.echo(f"Database not found: {cfg.db_path}")
                sys.exit(1)

            async with _opened_db(cfg) as conn:
                await _apply_read_schema_gate(conn, command="snapshot")
                out = (
                    Path(output_path) if output_path else cfg.db_path.with_suffix(".snapshot.jsonl")
                )
                total = await _export_db_to_jsonl(conn, out)
                click.echo(f"Exported {total} records to {out}")

    _run(_snapshot())


# ------------------------------------------------------------------
# restore (import from JSONL snapshot)
# ------------------------------------------------------------------


def _unreadable_snapshot_error(input_path: Path, exc: OSError) -> click.ClickException:
    """Describe an OS-level snapshot read failure as an actionable CLI error.

    ``--input`` is a user-typed path, so the ordinary outcomes of a typo — the
    file is not there, the path names a directory, the file is not readable —
    are user errors, not defects, and belong in a message rather than in an
    interpreter traceback. Shared by the open and the read so both report the
    same way.

    Args:
        input_path: Path to the JSONL snapshot file.
        exc: The failure the operating system reported.

    Returns:
        The exception to raise at the CLI boundary.

    """
    detail = exc.strerror or type(exc).__name__
    msg = (
        f"Cannot read snapshot file '{input_path}': {detail}. "
        "Pass an existing JSONL snapshot to --input (-i) — "
        "'engrava snapshot -o <file>' writes one."
    )
    return click.ClickException(msg)


def _open_snapshot(input_path: Path) -> TextIO:
    """Open a snapshot file for reading, or fail with a clean CLI error.

    Args:
        input_path: Path to the JSONL snapshot file.

    Returns:
        The opened text handle, which the caller closes.

    Raises:
        click.ClickException: If the path cannot be opened for reading.

    """
    try:
        return input_path.open(encoding="utf-8")
    except OSError as exc:
        raise _unreadable_snapshot_error(input_path, exc) from exc


def _iter_snapshot_lines(input_path: Path) -> Iterator[tuple[int, str]]:
    """Stream a snapshot file, yielding non-empty ``(line_number, line)`` pairs.

    Streaming keeps restore memory bounded by a single line rather than the
    whole snapshot. Lines are stripped and blank lines are skipped; line numbers
    are 1-based and count every physical line for accurate error context.

    This is the only place a restore opens the ``--input`` path, so both restore
    modes — single-database and ``--service`` — surface an unusable path as the
    same clean error.

    Args:
        input_path: Path to the JSONL snapshot file.

    Yields:
        ``(line_number, stripped_line)`` for each non-empty line.

    Raises:
        click.ClickException: If the path cannot be opened or read, or if its
            bytes are not UTF-8 text (the shape of pointing ``--input`` at a
            database or another binary file rather than at a snapshot). Reading
            is guarded as well as opening: a path that opens can still fail
            part-way through, and that too is a message rather than a traceback.

    """
    with _open_snapshot(input_path) as handle:
        try:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if stripped:
                    yield line_number, stripped
        except UnicodeDecodeError as exc:
            msg = (
                f"Snapshot file '{input_path}' is not UTF-8 text: {exc.reason}. "
                "--input (-i) expects a JSONL snapshot written by "
                "'engrava snapshot', not a database or other binary file."
            )
            raise click.ClickException(msg) from exc
        except OSError as exc:
            raise _unreadable_snapshot_error(input_path, exc) from exc


def _format_embedding_identity(identity: tuple[str, int]) -> str:
    """Render a declared ``(model_name, dimension)`` pair for an error message.

    Args:
        identity: The identity to render.

    Returns:
        A human-readable ``'<model>' at dimension <n>`` fragment.

    """
    model_name, dimension = identity
    return f"{model_name!r} at dimension {dimension}"


def _track_embedding_identity(
    identity: tuple[str, int],
    reference: tuple[str, int] | None,
    reference_label: str,
    *,
    subject_label: str,
) -> tuple[str, int]:
    """Compare one declared embedding identity against the running reference.

    The reference is whatever every embedding row seen so far in this restore
    agrees on. The first identity ever seen establishes it silently; every
    later one must match exactly, or restore fails before committing anything.

    Args:
        identity: The ``(model_name, dimension)`` pair just observed.
        reference: The identity established so far, or ``None`` if this is
            the first one seen.
        reference_label: A phrase describing where ``reference`` came from
            (the target's stored model, its existing rows, or an earlier
            snapshot row), for the mismatch message.
        subject_label: A phrase describing where ``identity`` came from, for
            the mismatch message.

    Returns:
        ``reference`` unchanged when it already matched, or ``identity`` when
        no reference was established yet.

    Raises:
        click.ClickException: If ``identity`` differs from an already
            established ``reference``.

    """
    if reference is None:
        return identity
    if identity == reference:
        return reference
    msg = (
        f"Embedding model mismatch: {reference_label} declares "
        f"{_format_embedding_identity(reference)}, but {subject_label} declares "
        f"{_format_embedding_identity(identity)}. Use --re-embed to regenerate "
        "embeddings for the target's model, or --skip-embeddings to skip "
        "importing vectors."
    )
    raise click.ClickException(msg)


async def _read_embedding_lock(conn: aiosqlite.Connection) -> tuple[str, int] | None:
    """Read the target's stored embedding-model lock, if any.

    Args:
        conn: Restore connection with an active transaction.

    Returns:
        The ``(model_name, dimension)`` recorded in ``_metadata``, or ``None``
        when the target has never locked a model.

    Raises:
        click.ClickException: If the target has a stored
            ``embedding_dimension`` that is not a valid integer -- the
            target's ``_metadata`` is corrupt and restore cannot verify
            embedding identity against it.

    """
    cursor = await conn.execute("SELECT value FROM _metadata WHERE key = 'embedding_model_name'")
    row = await cursor.fetchone()
    if row is None:
        return None
    model_name = str(row[0])
    dim_cursor = await conn.execute("SELECT value FROM _metadata WHERE key = 'embedding_dimension'")
    dim_row = await dim_cursor.fetchone()
    if dim_row is None:
        return model_name, 0
    try:
        dimension = int(dim_row[0])
    except ValueError as exc:
        msg = (
            "The target's stored embedding_dimension "
            f"({dim_row[0]!r}) is not a valid integer; its _metadata is "
            "corrupt. Repair the target's _metadata directly, or restore "
            "with --clear to reset it, before restoring again."
        )
        raise click.ClickException(msg) from exc
    return model_name, dimension


async def _existing_embedding_identities(conn: aiosqlite.Connection) -> list[tuple[str, int]]:
    """Return every distinct declared identity already in the target's ``embedding`` table.

    Args:
        conn: Restore connection with an active transaction.

    Returns:
        Distinct ``(model_name, dimension)`` pairs currently stored, in no
        particular order.

    """
    cursor = await conn.execute("SELECT DISTINCT model_name, dimension FROM embedding")
    rows = await cursor.fetchall()
    return [(str(row[0]), int(row[1])) for row in rows]


async def _write_embedding_lock(conn: aiosqlite.Connection, identity: tuple[str, int]) -> None:
    """Adopt a restored corpus's declared identity as the target's new lock.

    Used only when the target began this restore with neither a stored model
    nor any embedding rows of its own. Restore inserts ``embedding`` rows via
    fixed SQL directly (never through ``store_embedding()``), so nothing else
    would ever lock a target that starts this way -- it would otherwise end
    the restore holding vectors under no declared model at all.

    Writes only ``embedding_model_name`` and ``embedding_dimension``. A
    snapshot carries neither a document-prefix fingerprint nor a query
    prefix, so this cannot -- and does not -- set them (see the restore
    entry in ``docs/cli.md`` for that limit).

    Args:
        conn: Restore connection with an active transaction.
        identity: The ``(model_name, dimension)`` every embedding row just
            inserted was checked to declare.

    """
    model_name, dimension = identity
    await conn.execute(
        "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
        ("embedding_model_name", model_name),
    )
    await conn.execute(
        "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
        ("embedding_dimension", str(dimension)),
    )


async def _finalize_embedding_identity(
    conn: aiosqlite.Connection,
    *,
    may_adopt_identity: bool,
    identity_reference: tuple[str, int] | None,
) -> None:
    """Adopt the restored corpus's declared identity as the target's lock, if eligible.

    ``may_adopt_identity`` is only ever ``True`` when the target began the
    restore with neither a stored model nor any embedding rows (see
    ``_initial_embedding_state``), so this already implies the identity
    check ran; ``identity_reference`` is ``None`` only when no embedding row
    was ever inserted, in which case there is nothing to adopt.

    Args:
        conn: Restore connection with an active transaction.
        may_adopt_identity: Whether the target started this restore eligible
            for adoption.
        identity_reference: The identity every inserted embedding row was
            checked to declare, or ``None`` if none were inserted.

    """
    if may_adopt_identity and identity_reference is not None:
        await _write_embedding_lock(conn, identity_reference)


async def _initial_embedding_state(
    conn: aiosqlite.Connection,
) -> tuple[tuple[str, int] | None, str, bool]:
    """Establish the identity every embedding in the target must share, pre-restore.

    Runs unconditionally, regardless of ``--skip-embeddings`` or ``--re-embed``:
    those flags only decide whether the snapshot's own vectors are imported as
    incoming rows, never whether the target's pre-existing state is internally
    consistent. Reads the target's stored embedding-model lock, if any, and
    every distinct identity already declared by its own ``embedding`` rows,
    and asserts the two agree before any snapshot row is even parsed -- a
    merge restore into an already-inconsistent target must not report success
    (criterion: the check covers rows already in the target, not only
    incoming ones), no matter which import flags are given.

    A target with a lock and no vectors, and a target with vectors and no
    lock, are different states: the former's reference is its lock and never
    changes; the latter's reference comes from its own rows, and neither is
    later written back as a fresh lock (see ``_stream_insert``) -- only a
    target that starts with **neither** a lock nor any embeddings gets its
    incoming identity adopted.

    Args:
        conn: Restore connection with an active transaction.

    Returns:
        A ``(reference, reference_label, may_adopt)`` triple. ``reference``
        is the identity every embedding row -- existing and about to be
        inserted -- must share, or ``None`` when the target starts with
        neither a lock nor any embedding rows. ``may_adopt`` is ``True``
        only for that last case: the target started with neither, so it is
        eligible to have the snapshot's identity written as its new lock
        once every row has been checked to agree on it.

    Raises:
        click.ClickException: If the target's own existing embedding rows
            disagree with its stored lock, or with each other when there is
            no lock.

    """
    stored_lock = await _read_embedding_lock(conn)
    existing_identities = await _existing_embedding_identities(conn)

    if stored_lock is not None:
        reference_label = "the target's stored embedding model"
    elif existing_identities:
        reference_label = "the target's existing embedding rows"
    else:
        reference_label = "an earlier row in this snapshot"

    reference = stored_lock
    for existing_identity in existing_identities:
        reference = _track_embedding_identity(
            existing_identity,
            reference,
            reference_label,
            subject_label="one of the target's existing embedding rows",
        )
    may_adopt = stored_lock is None and not existing_identities
    return reference, reference_label, may_adopt


async def _check_embedding_row_before_insert(
    record: TableRecord,
    identity_reference: tuple[str, int] | None,
    identity_reference_label: str,
) -> tuple[str, int]:
    """Validate one about-to-be-inserted embedding row and update the reference.

    Called only for an ``embedding``-table record that will actually be
    inserted (the caller has already excluded ``--skip-embeddings`` /
    ``--re-embed``), so this is where both the structural check (dimension
    vs. blob) and the cross-row identity check happen.

    Args:
        record: A validated ``embedding``-table record about to be inserted.
        identity_reference: The identity established so far, or ``None``.
        identity_reference_label: A phrase describing where
            ``identity_reference`` came from, for a mismatch message.

    Returns:
        The identity every embedding row must now agree on.

    Raises:
        click.ClickException: If the row is structurally invalid, or its
            declared identity differs from ``identity_reference``.

    """
    row_identity = await _assert_embedding_row_structurally_valid(record)
    return _track_embedding_identity(
        row_identity,
        identity_reference,
        identity_reference_label,
        subject_label="a row in the snapshot",
    )


async def _assert_embedding_row_structurally_valid(record: TableRecord) -> tuple[str, int]:
    """Validate one embedding row's declared identity against its own bytes.

    ``EmbeddingRecord`` already validates that ``dimension`` matches the
    decoded ``vector_blob`` length; restore had never imported it, so a row
    claiming, say, dimension 384 backed by 16 bytes was accepted as-is. This
    reuses that same validator rather than re-implementing it.

    Args:
        record: A validated ``embedding``-table record about to be inserted.

    Returns:
        The row's declared ``(model_name, dimension)`` identity.

    Raises:
        click.ClickException: If the row's ``vector_blob`` is not valid
            base64 (surfaced by ``to_insert()``), or if the decoded record
            fails an ``EmbeddingRecord`` constraint -- including a
            ``dimension`` that does not match the decoded blob length.

    """
    from pydantic import ValidationError  # noqa: PLC0415

    from engrava.domain.models.embedding import EmbeddingRecord  # noqa: PLC0415

    _sql, values = record.to_insert()
    (
        embedding_id,
        owner_type,
        owner_id,
        model_name,
        dimension,
        vector_blob,
        created_at,
    ) = values
    if not (
        isinstance(embedding_id, str)
        and isinstance(owner_type, str)
        and isinstance(owner_id, str)
        and isinstance(model_name, str)
        and isinstance(dimension, int)
        and isinstance(vector_blob, bytes)
        and isinstance(created_at, str)
    ):
        # Unreachable in practice: TableSpec.validate() already enforced these
        # exact types for every required `embedding` column before a
        # TableRecord could exist.
        msg = f"Snapshot line {record.line_number} has a malformed embedding record."
        raise click.ClickException(msg)
    try:
        embedding = EmbeddingRecord(
            embedding_id=embedding_id,
            owner_type=owner_type,
            owner_id=owner_id,
            model_name=model_name,
            dimension=dimension,
            vector_blob=vector_blob,
            created_at=created_at,
        )
    except ValidationError as exc:
        msg = f"Snapshot line {record.line_number} has an invalid embedding record: {exc}"
        raise click.ClickException(msg) from exc
    return embedding.model_name, embedding.dimension


async def _reembed_thoughts(
    conn: aiosqlite.Connection,
    thought_ids: list[str],
    embedding_provider: EmbeddingProviderProtocol,
) -> int:
    """Re-embed a batch of imported thoughts via the target provider.

    Args:
        conn: Open aiosqlite connection.
        thought_ids: IDs of thoughts to re-embed (a single bounded batch).
        embedding_provider: Async embedding provider.

    Returns:
        Number of embeddings created.

    """
    import datetime  # noqa: PLC0415
    import struct  # noqa: PLC0415
    import uuid as _uuid  # noqa: PLC0415

    from engrava.infrastructure.sqlite.engrava_core import (  # noqa: PLC0415
        _embed_document,
    )

    count = 0
    for tid in thought_ids:
        cursor = await conn.execute(
            "SELECT essence, content FROM thought WHERE thought_id = ?", (tid,)
        )
        row = await cursor.fetchone()
        if not row:
            continue
        text = f"{row[0]}\n{row[1]}"
        vector = await _embed_document(embedding_provider, text)
        if len(vector) != embedding_provider.dimension:
            msg = (
                f"Embedding provider {embedding_provider.model_name!r} returned "
                f"{len(vector)} dimensions; expected {embedding_provider.dimension}."
            )
            raise click.ClickException(msg)
        blob = struct.pack(f"<{len(vector)}f", *vector)
        now = datetime.datetime.now(tz=datetime.UTC).isoformat()
        await conn.execute(
            "INSERT OR REPLACE INTO embedding "
            "(embedding_id, owner_type, owner_id, "
            "model_name, dimension, vector_blob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(_uuid.uuid4()),
                "THOUGHT",
                tid,
                embedding_provider.model_name,
                embedding_provider.dimension,
                blob,
                now,
            ),
        )
        count += 1
    return count


async def _replace_embedding_model_metadata(
    conn: aiosqlite.Connection,
    embedding_provider: EmbeddingProviderProtocol | None,
) -> None:
    """Replace the corpus identity after a successful transactional re-embed.

    Args:
        conn: Restore connection with an active transaction.
        embedding_provider: Provider that generated every imported vector, or
            ``None`` when the restored corpus has no vectors and must remain
            unlocked.

    """
    from engrava.infrastructure.sqlite.engrava_core import (  # noqa: PLC0415
        _METADATA_DOCUMENT_PREFIX_FINGERPRINT,
        _METADATA_QUERY_PREFIX,
        _document_prefix_fingerprint,
        _role_prefixes,
    )

    await conn.execute(
        "DELETE FROM _metadata WHERE key IN (?, ?, ?, ?)",
        (
            "embedding_model_name",
            "embedding_dimension",
            _METADATA_DOCUMENT_PREFIX_FINGERPRINT,
            _METADATA_QUERY_PREFIX,
        ),
    )
    if embedding_provider is None:
        return

    query_prefix, document_prefix = _role_prefixes(embedding_provider)
    document_fingerprint = _document_prefix_fingerprint(document_prefix)
    metadata = [
        ("embedding_model_name", embedding_provider.model_name),
        ("embedding_dimension", str(embedding_provider.dimension)),
    ]
    if document_fingerprint is not None:
        metadata.append((_METADATA_DOCUMENT_PREFIX_FINGERPRINT, document_fingerprint))
    if query_prefix:
        metadata.append((_METADATA_QUERY_PREFIX, query_prefix))
    await conn.executemany(
        "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
        metadata,
    )


async def _assert_reembed_target_is_safe(
    conn: aiosqlite.Connection,
    *,
    clear: bool,
) -> None:
    """Reject relabelling embeddings that are not part of this restore.

    Args:
        conn: Restore connection with an active transaction.
        clear: Whether restore will clear all existing core records first.

    Raises:
        click.ClickException: If existing embeddings would survive the restore.

    """
    if clear:
        return
    cursor = await conn.execute("SELECT COUNT(*) FROM embedding")
    row = await cursor.fetchone()
    existing_count = int(row[0]) if row is not None else 0
    if existing_count:
        msg = (
            "--re-embed requires an empty embedding target or --clear; "
            f"the target already contains {existing_count} embedding record(s)."
        )
        raise click.ClickException(msg)


async def _has_persisted_vector_index(conn: aiosqlite.Connection) -> bool:
    """Report whether the database carries a persisted vec0 index.

    Asked of ``sqlite_master``, which is an ordinary table: every statement
    naming ``embedding_vec`` itself fails with ``no such module: vec0`` unless
    sqlite-vec has been loaded into the connection first, so this is the only
    question about the index that can be answered before loading it.

    Args:
        conn: An open connection to the database.

    Returns:
        ``True`` when an ``embedding_vec`` table exists.

    """
    cursor = await conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding_vec'"
    )
    return await cursor.fetchone() is not None


async def _reset_sqlite_vec_index_for_restore(conn: aiosqlite.Connection) -> None:
    """Drop a persisted vec0 index before replacing its canonical vectors.

    The ``embedding`` table is the source of truth. A vec0 table may retain old
    rows across ``--clear`` and SQLite may then reuse their rowids, making a
    missing-row-only startup sync mistake stale vectors for current ones.
    Dropping the derived table inside the restore transaction lets the next
    sqlite-vec-enabled open recreate it at the new dimension and backfill it.

    Args:
        conn: Restore connection with an active transaction.

    Raises:
        click.ClickException: If a persisted vec0 table exists but the
            sqlite-vec module cannot be loaded to remove it safely.

    """
    if not await _has_persisted_vector_index(conn):
        return

    from engrava.infrastructure.sqlite.vector_sqlite_vec import load_sqlite_vec  # noqa: PLC0415

    if not await load_sqlite_vec(conn):
        msg = (
            "Restore found an existing sqlite-vec index but could not load "
            "sqlite-vec to rebuild it safely. Install 'engrava[vec]' and retry."
        )
        raise click.ClickException(msg)
    await conn.execute("DROP TABLE embedding_vec")


async def _insert_record(
    conn: aiosqlite.Connection,
    record: TableRecord,
    *,
    plain_insert: bool,
) -> None:
    """Insert one validated snapshot record via fixed, allow-listed SQL.

    The statement's column identifiers come only from the record's
    :class:`~engrava.cli.snapshot_records.TableSpec`; no identifier is derived
    from snapshot data, and every value travels as a bound parameter.

    Args:
        conn: Open aiosqlite connection.
        record: A validated core-table record.
        plain_insert: When ``True``, insert with an ordinary ``INSERT`` so a
            colliding primary key or ``UNIQUE`` constraint raises instead of
            silently replacing (see the journalled-merge collision gate in
            :func:`_import_records_to_db`).

    Raises:
        sqlite3.IntegrityError: If ``plain_insert`` is set and the record
            collides with an existing row. The caller (:func:`_stream_insert`)
            translates this into a ``click.ClickException``.

    """
    sql, values = record.to_insert(plain_insert=plain_insert)
    await conn.execute(sql, values)


def _reembed_id(
    record: TableRecord,
    *,
    re_embed: bool,
    embedding_provider: EmbeddingProviderProtocol | None,
) -> str | None:
    """Return the thought ID to re-embed for a record, or ``None``.

    Args:
        record: A validated core-table record about to be inserted.
        re_embed: Whether re-embedding is requested.
        embedding_provider: Target embedding provider (or ``None``).

    Returns:
        The thought's ID when re-embedding applies to it, otherwise ``None``.

    """
    if not (re_embed and embedding_provider is not None and record.spec.table is CoreTable.THOUGHT):
        return None
    tid = record.data["thought_id"]  # required + non-null, validated at parse
    return tid if isinstance(tid, str) else None


async def _insert_record_under_gate(
    conn: aiosqlite.Connection,
    record: TableRecord,
    *,
    plain_insert: bool,
) -> None:
    """Insert one record, translating a gate-relevant collision into a clean error.

    Thin wrapper around :func:`_insert_record` that exists only to keep the
    ``try``/``except`` out of :func:`_stream_insert`'s already-long loop body.

    Args:
        conn: Open aiosqlite connection.
        record: A validated core-table record.
        plain_insert: See :func:`_insert_record`.

    Raises:
        click.ClickException: If ``plain_insert`` is set and the record
            collides on a primary key or ``UNIQUE`` constraint.

    """
    try:
        await _insert_record(conn, record, plain_insert=plain_insert)
    except sqlite3.IntegrityError as exc:
        if exc.sqlite_errorcode in _JOURNAL_GATE_CONSTRAINT_CODES:
            raise _journal_gate_collision_error(record.line_number) from exc
        raise


def _journal_gate_collision_error(line_number: int) -> click.ClickException:
    """Describe a refused collision under the journalled-merge collision gate.

    Args:
        line_number: 1-based snapshot line number of the record that collided.

    Returns:
        A ``click.ClickException`` naming the gate and its override.

    """
    msg = (
        f"Restore refused: snapshot line {line_number} collides with an existing row "
        "(matching primary key or UNIQUE constraint), and the target's journal_entry "
        "table is not empty. Replacing that row would leave the audit trail describing "
        "data this merge discarded, while 'engrava verify' kept reporting the chain as "
        "valid. Re-run with --orphan-journal-entries to allow the merge and accept that "
        "gap, or with --clear to discard the journal along with the data."
    )
    return click.ClickException(msg)


async def _stream_insert(
    conn: aiosqlite.Connection,
    input_path: Path,
    *,
    skip_embeddings: bool,
    re_embed: bool,
    embedding_provider: EmbeddingProviderProtocol | None,
    plain_insert: bool,
) -> int:
    """Stream a snapshot once, validating and inserting each record in order.

    Each record is fully validated -- structure and values -- immediately before
    it is inserted, so a bad record raises before its own write. Re-embedding IDs
    are flushed in bounded batches; peak memory is one line plus one batch.

    Every ``embedding`` row already in the target, at the very start of this
    restore, must declare the same ``model_name``/``dimension`` identity as
    the target's stored embedding-model lock (or, for a target with no lock,
    as each other) -- checked unconditionally, regardless of
    ``skip_embeddings`` or ``re_embed``, because those flags only decide
    whether *incoming* vectors are imported, never whether the target's own
    pre-existing rows are internally consistent.

    Unless ``skip_embeddings`` or ``re_embed`` is set, every incoming
    ``embedding`` row about to be inserted must also declare that same
    identity. This is checked against the snapshot's ``embedding`` rows and
    the target's own data, never against ``embedding_provider`` or the
    snapshot's metadata header: a plain restore never resolves a provider,
    and the header is not proof of anything the rows do not already say for
    themselves. A target that starts with neither a lock nor any embeddings
    adopts the snapshot's declared identity as its new lock once every row
    has been checked to agree on it.

    Args:
        conn: Open aiosqlite connection (inside the caller's transaction).
        input_path: Path to the JSONL snapshot file.
        skip_embeddings: Skip embedding records during import.
        re_embed: Re-embed thoughts via the embedding provider after insert.
        embedding_provider: ``EmbeddingProviderProtocol`` for re-embedding.
        plain_insert: When ``True``, every record is written with an ordinary
            ``INSERT`` instead of ``INSERT OR REPLACE`` (the journalled-merge
            collision gate computed once by :func:`_import_records_to_db`), so
            a colliding primary key or ``UNIQUE`` constraint is refused rather
            than silently replacing (or cascade-deleting) the existing row.

    Returns:
        Total number of records written (inserts plus re-embeddings).

    Raises:
        click.ClickException: On a malformed record, an invalid value, an
            embedding identity mismatch without an override flag, a
            journalled-merge collision refused under ``plain_insert`` (see
            above), or the target's own existing rows already disagreeing
            among themselves or with its lock.

    """
    check_incoming = not re_embed and not skip_embeddings
    total = 0
    reembedded = 0
    reembed_batch: list[str] = []

    (
        identity_reference,
        identity_reference_label,
        may_adopt_identity,
    ) = await _initial_embedding_state(conn)

    for line_number, line in _iter_snapshot_lines(input_path):
        record = parse_snapshot_record(line, line_number=line_number)
        if isinstance(record, MetadataRecord):
            continue
        if not isinstance(record, TableRecord):
            continue
        if record.spec.table is CoreTable.EMBEDDING:
            if not check_incoming:
                continue
            identity_reference = await _check_embedding_row_before_insert(
                record, identity_reference, identity_reference_label
            )

        await _insert_record_under_gate(conn, record, plain_insert=plain_insert)
        total += 1

        tid = _reembed_id(record, re_embed=re_embed, embedding_provider=embedding_provider)
        if tid is not None and embedding_provider is not None:
            reembed_batch.append(tid)
            if len(reembed_batch) >= _REEMBED_BATCH_SIZE:
                batch_count = await _reembed_thoughts(conn, reembed_batch, embedding_provider)
                total += batch_count
                reembedded += batch_count
                reembed_batch.clear()

    if reembed_batch and embedding_provider is not None:
        batch_count = await _reembed_thoughts(conn, reembed_batch, embedding_provider)
        total += batch_count
        reembedded += batch_count
    if re_embed and embedding_provider is not None:
        await _replace_embedding_model_metadata(
            conn,
            embedding_provider if reembedded else None,
        )
    await _finalize_embedding_identity(
        conn, may_adopt_identity=may_adopt_identity, identity_reference=identity_reference
    )
    return total


@dataclass(frozen=True, slots=True)
class RestoreImportResult:
    """Outcome of importing snapshot records into one database connection.

    Attributes:
        total_records: Total records inserted from the snapshot (the whole
            historical return value of :func:`_import_records_to_db`).
        journal_entries_cleared: Rows removed from ``journal_entry`` by
            ``--clear``. Always ``0`` when ``clear`` is not set, since a
            restore without ``--clear`` never touches the journal.

    """

    total_records: int
    journal_entries_cleared: int


async def _journal_entry_has_rows(conn: aiosqlite.Connection) -> bool:
    """Report whether the target's ``journal_entry`` table currently holds a row.

    Scopes the journalled-merge collision gate in :func:`_import_records_to_db`:
    journalling is opt-in and the CLI never enables it itself, so the
    overwhelmingly common restore target has an empty ``journal_entry`` and
    this returns ``False``, leaving the merge behaviour exactly as it always
    was.

    Args:
        conn: Restore connection with an active transaction.

    Returns:
        ``True`` when at least one row exists in ``journal_entry``.

    """
    cursor = await conn.execute("SELECT 1 FROM journal_entry LIMIT 1")
    return await cursor.fetchone() is not None


async def _import_records_to_db(
    conn: aiosqlite.Connection,
    input_path: Path,
    *,
    clear: bool = False,
    skip_embeddings: bool = False,
    re_embed: bool = False,
    embedding_provider: EmbeddingProviderProtocol | None = None,
    orphan_journal_entries: bool = False,
) -> RestoreImportResult:
    """Import JSONL records into a database connection atomically.

    The whole restore runs in a **single transaction over a single streaming
    pass**: each record is fully validated -- structure and values, including
    base64 decoding of an embedding blob -- immediately before it is inserted,
    and any failure (a malformed record, an embedding-model mismatch, or a bad
    value) rolls the transaction back so nothing is ever committed from an
    invalid snapshot. "Reject before any write" therefore holds as "nothing
    persists", including the optional ``clear``. The file is read exactly once
    and peak memory is one line plus one re-embed batch.

    ``clear`` also empties ``journal_entry``. Without that, a cleared store's
    data and its existing journal would describe two different histories --
    the journal would keep authenticating thoughts the clear just removed --
    and ``verify_journal()`` would keep reporting that mismatched chain as
    valid.

    A restore without ``--clear`` never writes to ``journal_entry`` itself,
    but that is not the same as leaving the journal *consistent*. Without the
    gate described below, every record is inserted with ``INSERT OR
    REPLACE``, and an incoming thought, edge, or action whose id matches one
    the journal already describes replaces it outright. But an id match is
    not the only way a journalled row is orphaned this way, and framing it as
    the only way is exactly the false conservatism the shipped documentation
    used to carry: an incoming edge with a brand-new ``edge_id`` still
    replaces a journalled edge if it repeats that table's composite
    ``UNIQUE(from_thought_id, to_thought_id, edge_type)`` (schema_core.sql),
    with no id ever colliding, and replacing a journalled thought cascades an
    ``ON DELETE CASCADE`` foreign-key delete onto *that thought's own* edges,
    embeddings, and actions -- rows whose ids never appeared in the incoming
    snapshot at all. Either way, the journal entries describing what was just
    removed are left behind unchanged, and ``verify_journal()`` keeps reporting that
    mismatched chain as valid, because the chain itself stays internally
    self-consistent; it simply no longer matches what is stored.

    The **journalled-merge collision gate** closes this for every case above,
    because it does not try to enumerate them: when this restore has no
    ``--clear``, no ``orphan_journal_entries`` override, and the target's
    ``journal_entry`` table is non-empty, every incoming record is instead
    written with a plain ``INSERT``, so SQLite itself refuses any uniqueness
    violation -- primary key or ``UNIQUE``, including the composite one above
    -- with a ``click.ClickException`` instead of silently replacing (or
    cascade-deleting) the row. Nothing is deleted first, so the cascade path
    is not merely caught, it never fires. The gate is conservative, not
    precise: it refuses *any* such collision once a journal exists, including
    one on a row the journal never described, and ``orphan_journal_entries``
    exists for a caller who has weighed that and wants the merge anyway --
    which restores exactly the unconditional ``INSERT OR REPLACE`` behaviour
    described above. The gate never applies when ``journal_entry`` is empty,
    which is the overwhelmingly common case since journalling is opt-in and
    the CLI never enables it itself -- that restore's merge behaviour is
    unchanged.

    Args:
        conn: Open aiosqlite connection with schema applied.
        input_path: Path to the JSONL snapshot file.
        clear: Delete existing data before import.
        skip_embeddings: Skip embedding records during import.
        re_embed: Re-embed thoughts via the embedding provider after import.
        embedding_provider: ``EmbeddingProviderProtocol`` for re-embedding.
        orphan_journal_entries: Opt out of the journalled-merge collision gate
            described above, restoring the unconditional ``INSERT OR REPLACE``
            merge even when the target's journal is non-empty.

    Returns:
        The total records imported and how many journal entries ``--clear``
        discarded (zero when ``clear`` is not set).

    Raises:
        click.ClickException: On a malformed snapshot record, an invalid value,
            an embedding-model mismatch without an override flag, or a
            collision refused by the journalled-merge collision gate. The
            transaction is rolled back before the error propagates.

    """
    total = 0
    journal_entries_cleared = 0
    committed = False
    # Open the transaction explicitly so atomicity holds regardless of the
    # connection's isolation configuration (it does not depend on the driver's
    # implicit-transaction default).
    await conn.execute("BEGIN")
    try:
        if re_embed:
            await _assert_reembed_target_is_safe(conn, clear=clear)
        if clear or re_embed:
            await _reset_sqlite_vec_index_for_restore(conn)
        if clear:
            for table in _CORE_TABLES_DELETE_ORDER:
                await conn.execute(f"DELETE FROM {table.value}")  # noqa: S608
            # Fixed literal, never interpolated -- see the module comment
            # above `_CORE_TABLES_DELETE_ORDER` for why `journal_entry` is
            # cleared this way instead of through that enum. `rowcount` on a
            # bare `DELETE FROM` (no `WHERE`) reports the exact number of
            # rows removed.
            journal_cursor = await conn.execute("DELETE FROM journal_entry")
            journal_entries_cleared = journal_cursor.rowcount
        # The journalled-merge collision gate's three-part condition,
        # evaluated once, before the insert loop below. `clear` short-circuits
        # the query entirely -- a `--clear` restore has already emptied
        # `journal_entry` above, so the query would return `False` anyway.
        plain_insert = (
            not clear and not orphan_journal_entries and await _journal_entry_has_rows(conn)
        )
        total = await _stream_insert(
            conn,
            input_path,
            skip_embeddings=skip_embeddings,
            re_embed=re_embed,
            embedding_provider=embedding_provider,
            plain_insert=plain_insert,
        )
        await conn.commit()
        committed = True
    finally:
        if not committed:
            # Any validation or insert failure discards the whole restore.
            await conn.rollback()
    return RestoreImportResult(
        total_records=total,
        journal_entries_cleared=journal_entries_cleared,
    )


def _require_valid_cli_service_name(service_name: str) -> None:
    """Reject a malformed ``--service`` value with a distinct CLI error.

    ``EngravaManager.service_exists`` / ``get_store`` validate the service name
    and raise :class:`ConfigError` for a malformed value (e.g. a path-escape
    attempt like ``../escape``). Surfacing that as a plain ``ClickException``
    keeps it a clean, user-facing message — never a traceback — and keeps it
    distinct from an embedding-provider initialisation failure.

    This is a rejection gate only, so it discards the validated name the
    validator hands back. That is safe precisely here and nowhere else: the
    value is a command-line argument, which ``click`` always supplies as an
    exact ``str``, and ``EngravaManager`` re-validates and re-owns the name at
    its own boundary before it ever addresses a file with it.

    Args:
        service_name: The ``--service`` value supplied on the command line.

    Raises:
        click.ClickException: If ``service_name`` is not a valid service name.

    """
    from engrava.config import _validate_service_name  # noqa: PLC0415

    try:
        _validate_service_name(service_name)
    except ConfigError as exc:
        msg = f"Invalid --service value: {exc}"
        raise click.ClickException(msg) from exc


async def _restore_service_snapshot(
    *,
    effective_service: str,
    input_path: str,
    clear: bool,
    skip_embeddings: bool,
    re_embed: bool,
    orphan_journal_entries: bool,
    cfg: EngravaCLIConfig,
    services_cfg: ServicesConfig | None,
    default_embeddings: EmbeddingConfig | None,
) -> None:
    """Restore a JSONL snapshot into a named service database.

    Raises:
        click.ClickException: On an invalid service name, an embedding-provider
            initialisation failure, or a missing ``--re-embed`` provider.

    """
    from engrava.infrastructure.service_manager import EngravaManager  # noqa: PLC0415

    # The service name was validated up front by the command (see restore()), so
    # ``effective_service`` is a well-formed, non-empty name here.
    data_dir = services_cfg.data_dir if services_cfg else cfg.db_path.parent
    manager = EngravaManager(
        data_dir=data_dir,
        default_embeddings=default_embeddings if re_embed else None,
        services_config=services_cfg,
    )
    try:
        # restore is destructive (an existing target can be cleared, and is
        # always rewritten), and no CLI command migrates a database
        # implicitly. An existing service target must already be at head; a
        # target that does not exist yet is a fresh service, which
        # get_store() below still bootstraps to head as it always has.
        existing_version = await manager.peek_schema_version(effective_service)
        if existing_version is not None:
            _apply_destructive_schema_gate_for_version(existing_version, command="restore")
        try:
            store = await manager.get_store(effective_service)
        except ConfigError as exc:
            msg = (
                f"Cannot initialize service {effective_service!r} from the configured "
                f"embedding provider: {exc}"
            )
            raise click.ClickException(msg) from exc
        emb_provider = None
        if re_embed:
            emb_provider = store._embedding_provider  # noqa: SLF001
            if emb_provider is None:
                msg = (
                    "--re-embed requires an embedding provider, but none is "
                    f"configured for service {effective_service!r}. "
                    "Set 'embeddings.provider' in engrava.yaml or "
                    "use --skip-embeddings instead."
                )
                raise click.ClickException(msg)
        result = await _import_records_to_db(
            store._db,  # noqa: SLF001
            Path(input_path),
            clear=clear,
            skip_embeddings=skip_embeddings,
            re_embed=re_embed,
            embedding_provider=emb_provider,
            orphan_journal_entries=orphan_journal_entries,
        )
        click.echo(
            f"Restored {result.total_records} records to service {effective_service!r} "
            f"from {input_path}"
        )
        if clear:
            click.echo(f"Discarded {result.journal_entries_cleared} journal entries")
    finally:
        await manager.close_all()


async def _restore_single_db(
    *,
    input_path: str,
    clear: bool,
    skip_embeddings: bool,
    re_embed: bool,
    orphan_journal_entries: bool,
    cfg: EngravaCLIConfig,
    default_embeddings: EmbeddingConfig | None,
) -> None:
    """Restore a JSONL snapshot into the single-file database.

    Raises:
        click.ClickException: On an embedding-provider initialisation failure or
            a missing ``--re-embed`` provider.

    """
    from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore  # noqa: PLC0415

    emb_provider = None
    if re_embed:
        try:
            emb_provider = resolve_embedding_provider(default_embeddings)
        except ConfigError as exc:
            msg = f"Cannot initialize the configured embedding provider: {exc}"
            raise click.ClickException(msg) from exc
        if emb_provider is None:
            msg = (
                "--re-embed requires an embedding provider, but none is configured. "
                "Set top-level 'embeddings.provider' in engrava.yaml and pass "
                "--config, or use --skip-embeddings instead."
            )
            raise click.ClickException(msg)

    # restore is destructive, and no CLI command migrates a database
    # implicitly. Checked *before* connecting opens (and therefore creates)
    # the file, so a target that does not exist yet is unambiguously a fresh
    # restore rather than "behind" — there is nothing to be behind.
    pre_existing = cfg.db_path.exists()

    # _opened_db closes the connection on any exit from this block — a
    # corrupt existing target, a schema-gate refusal, or a failure inside
    # ensure_schema() while bootstrapping a fresh one — because opening the
    # connection and entering the protected block are the same step. Neither
    # the constructor below nor ensure_schema() can leak by sitting in front
    # of a try that starts later.
    async with _opened_db(cfg) as conn:
        store = SqliteEngravaCore(conn)
        if pre_existing:
            await _apply_destructive_schema_gate(conn, command="restore")
            # Schema confirmed at head above; ensure_schema() here would be a
            # no-op, so it is skipped entirely rather than called for its side
            # effect of none — no command migrates implicitly.
        else:
            await store.ensure_schema()

        result = await _import_records_to_db(
            conn,
            Path(input_path),
            clear=clear,
            skip_embeddings=skip_embeddings,
            re_embed=re_embed,
            embedding_provider=emb_provider,
            orphan_journal_entries=orphan_journal_entries,
        )
        click.echo(f"Restored {result.total_records} records from {input_path}")
        if clear:
            click.echo(f"Discarded {result.journal_entries_cleared} journal entries")


@cli.command()
@click.option("-i", "--input", "input_path", required=True, help="JSONL snapshot file to restore.")
@click.option("--clear", is_flag=True, help="Clear existing data before restore.")
@click.option(
    "--skip-embeddings",
    is_flag=True,
    help="Skip embedding records during import.",
)
@click.option(
    "--re-embed",
    is_flag=True,
    help="Re-embed all thoughts via the target provider (ignores source embeddings).",
)
@click.option(
    "--orphan-journal-entries",
    is_flag=True,
    help=(
        "Allow a merge restore (no --clear) to replace rows in a target whose "
        "journal_entry table is not empty, even though the journal will then "
        "describe data the merge discarded. Without this flag, such a "
        "collision is refused."
    ),
)
@click.option(
    "--service",
    "service_name",
    default=None,
    help="Service name (multi-service mode).",
)
@click.pass_context
def restore(
    ctx: click.Context,
    input_path: str,
    *,
    clear: bool,
    skip_embeddings: bool,
    re_embed: bool,
    orphan_journal_entries: bool,
    service_name: str | None,
) -> None:
    """Restore database from a JSONL snapshot file.

    Supports model-mismatch handling via ``--re-embed`` (re-generate
    embeddings) or ``--skip-embeddings`` (import without vectors).

    A merge restore (no ``--clear``) into a target whose ``journal_entry``
    table is not empty refuses any record that collides with an existing row,
    to keep the audit trail from silently describing data the merge replaced
    or removed. Pass ``--orphan-journal-entries`` to allow that merge anyway.
    """
    cfg: EngravaCLIConfig = ctx.obj["config"]
    services_cfg: ServicesConfig | None = ctx.obj.get("services_config")
    default_embeddings: EmbeddingConfig | None = ctx.obj.get("default_embeddings")

    if re_embed and skip_embeddings:
        click.echo("Error: --re-embed and --skip-embeddings are mutually exclusive.", err=True)
        sys.exit(1)

    # Resolve default service from config if --service not given.
    effective_service = service_name
    if effective_service is None and services_cfg is not None:
        effective_service = services_cfg.default_service

    # Validate any resolved service name — an explicit --service (including an
    # empty string, which is falsy) or a config default — up front so a malformed
    # value is a clean ClickException rather than a silent fall-through to the
    # single-database path or a mislabelled embedding-provider error.
    if effective_service is not None:
        _require_valid_cli_service_name(effective_service)

    async def _restore() -> None:
        if effective_service:
            await _restore_service_snapshot(
                effective_service=effective_service,
                input_path=input_path,
                clear=clear,
                skip_embeddings=skip_embeddings,
                re_embed=re_embed,
                orphan_journal_entries=orphan_journal_entries,
                cfg=cfg,
                services_cfg=services_cfg,
                default_embeddings=default_embeddings,
            )
        else:
            await _restore_single_db(
                input_path=input_path,
                clear=clear,
                skip_embeddings=skip_embeddings,
                re_embed=re_embed,
                orphan_journal_entries=orphan_journal_entries,
                cfg=cfg,
                default_embeddings=default_embeddings,
            )

    _run(_restore())


# ------------------------------------------------------------------
# gc (garbage-collect archived/soft-deleted thoughts)
# ------------------------------------------------------------------


async def _prepare_vector_index_purge(conn: aiosqlite.Connection) -> bool:
    """Load sqlite-vec when the store has an index the collection must maintain.

    ``embedding_vec`` is a vec0 virtual table that no foreign key reaches, so
    deleting a thought — by any route — leaves its vector behind unless the
    vector is removed explicitly, and removing it needs the module loaded on
    this connection, which the CLI's plain connection helper does not do.

    Called by each pass that is about to physically delete, immediately before
    it does, because the failure has to be refused while the store is still
    whole: continuing would strand the vectors of rows that are already gone,
    which is the state the purge exists to prevent. A pass that deletes nothing
    — a dry run, an archiving expiry sweep, a collection with nothing to
    collect — cannot strand a vector and is never refused for want of the
    module. Loading twice on one connection is harmless, so the two passes ask
    independently rather than sharing a resolution neither of them owns.

    Args:
        conn: The command's open connection.

    Returns:
        ``True`` when a persisted index exists and is now reachable, so the
        collection paths must purge it; ``False`` when there is none.

    Raises:
        click.ClickException: If a persisted vec0 index exists but sqlite-vec
            cannot be loaded to maintain it.

    """
    if not await _has_persisted_vector_index(conn):
        return False

    from engrava.infrastructure.sqlite.vector_sqlite_vec import load_sqlite_vec  # noqa: PLC0415

    if not await load_sqlite_vec(conn):
        msg = (
            "This database has a sqlite-vec index, and collecting thoughts "
            "without removing their vectors would strand them in it. "
            "Install 'engrava[vec]' and retry."
        )
        raise click.ClickException(msg)
    return True


async def _reconcile_vector_index(conn: aiosqlite.Connection) -> None:
    """Bring the vec0 index back into agreement with the rows behind it.

    Called after a pass has deleted, so the vectors whose ``embedding`` row
    went with those rows are **among** what this removes — it is the index's
    own reconciliation, not a targeted delete, so a vector some earlier writer
    left behind goes with them provided its rowid is still unowned. One that
    SQLite has since handed to a new embedding is not, and no reconciliation
    phrased this way can tell.

    Phrased as "no ``embedding`` row owns it" rather than as a list of rowids
    on purpose: it is the predicate the index is reconciled by everywhere else,
    it needs no second reading of which rows a sweep actually took, and it
    cannot remove the vector of a thought that is still stored.

    Args:
        conn: The command's connection, with sqlite-vec already loaded by
            :func:`_prepare_vector_index_purge`.

    """
    from engrava.infrastructure.sqlite.vector_sqlite_vec import (  # noqa: PLC0415
        purge_orphan_vectors,
    )

    await purge_orphan_vectors(conn)


async def _gc_expired(
    conn: aiosqlite.Connection,
    cfg: EngravaCLIConfig,
    *,
    dry_run: bool,
) -> bool:
    """Cleanup expired TTL thoughts.

    Under the ``delete`` strategy the sweep physically removes thoughts, so it
    owes their vectors the same purge the collection pass owes its own. The
    sweep and that purge run in **one** transaction: the core commits per
    operation by default, and a purge that failed after that commit would leave
    behind exactly the stranded vectors it exists to remove — and would hand a
    concurrent writer a window in which to reuse a freed ``embedding`` rowid,
    after which the stale vector is indistinguishable from a live one.

    Returns True when the caller should skip the subsequent archived-GC
    (i.e. archive strategy was used and thoughts were moved).
    """
    from engrava.config import TTLConfig, load_config  # noqa: PLC0415
    from engrava.domain.models.ttl import CleanupStrategy  # noqa: PLC0415
    from engrava.infrastructure.sqlite.engrava_core import (  # noqa: PLC0415
        SqliteEngravaCore,
    )

    ttl_cfg = TTLConfig()
    if cfg.config_path and cfg.config_path.exists():
        ms_cfg = load_config(cfg.config_path)
        ttl_cfg = ms_cfg.ttl

    store = SqliteEngravaCore(db=conn, ttl_strategy=ttl_cfg.strategy)

    cursor = await conn.execute(
        "SELECT COUNT(*) FROM thought WHERE expires_at IS NOT NULL AND expires_at <= ?",
        (datetime.datetime.now(datetime.UTC).isoformat(),),
    )
    row = await cursor.fetchone()
    exp_count = row[0] if row else 0

    if exp_count == 0:
        click.echo("No expired thoughts to cleanup.")
        return False

    if dry_run:
        click.echo(f"Would {ttl_cfg.strategy} {exp_count} expired thoughts.")
        return False

    purge_vectors = False
    if ttl_cfg.strategy == CleanupStrategy.DELETE:
        purge_vectors = await _prepare_vector_index_purge(conn)

    async with store.suspend_auto_commit():
        result = await store.cleanup_expired()
        if purge_vectors and result.expired_count > 0:
            await _reconcile_vector_index(conn)
    click.echo(
        f"Cleaned up {result.expired_count} expired thoughts (strategy: {result.strategy_applied})."
    )
    return result.strategy_applied == CleanupStrategy.ARCHIVE and result.expired_count > 0


async def _gc_archived(
    conn: aiosqlite.Connection,
    *,
    dry_run: bool,
    quiet: bool,
) -> None:
    """Physically delete all ARCHIVED thoughts, their edges, embeddings, actions and vectors.

    Every statement below — the child deletes, the parent delete and the vector
    purge — runs in the one transaction this function's ``commit`` closes, so a
    failed purge takes the deletes with it rather than stranding the vectors of
    rows that are already gone.
    """
    cursor = await conn.execute("SELECT COUNT(*) FROM thought WHERE lifecycle_status = 'ARCHIVED'")
    row = await cursor.fetchone()
    archived_count = row[0] if row else 0

    if archived_count == 0:
        if not quiet:
            click.echo("No archived thoughts to collect.")
        return

    if dry_run:
        click.echo(
            f"Would delete {archived_count} archived thoughts, "
            "plus their edges, embeddings, and actions."
        )
        return

    purge_vectors = await _prepare_vector_index_purge(conn)

    await conn.execute(
        "DELETE FROM edge WHERE from_thought_id IN "
        "(SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED') "
        "OR to_thought_id IN "
        "(SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED')"
    )
    await conn.execute(
        "DELETE FROM embedding WHERE owner_id IN "
        "(SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED')"
    )
    await conn.execute(
        "DELETE FROM action WHERE source_thought_id IN "
        "(SELECT thought_id FROM thought WHERE lifecycle_status = 'ARCHIVED')"
    )
    cursor = await conn.execute("DELETE FROM thought WHERE lifecycle_status = 'ARCHIVED'")
    collected = cursor.rowcount
    if purge_vectors:
        await _reconcile_vector_index(conn)
    await conn.commit()
    click.echo(f"Collected {collected} archived thoughts.")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without acting.")
@click.option(
    "--expired",
    is_flag=True,
    help="Also cleanup expired TTL thoughts (archive or delete per config).",
)
@click.pass_context
def gc(ctx: click.Context, *, dry_run: bool, expired: bool) -> None:
    """Garbage-collect archived thoughts, their edges, embeddings, and actions.

    With ``--expired``, also clean up expired TTL thoughts first (archived
    or deleted per the configured ``ttl.strategy``).
    """
    cfg: EngravaCLIConfig = ctx.obj["config"]

    async def _gc() -> None:
        if not cfg.db_path.exists():
            click.echo(f"Database not found: {cfg.db_path}")
            sys.exit(1)

        async with _opened_db(cfg) as conn:
            # gc is destructive and never migrates the schema itself (see
            # _gc_archived's own docstring below on the explicit child
            # deletes that limit today's damage) — so it refuses outright on
            # anything but a head schema rather than deleting rows through an
            # engine that does not understand the schema it is deleting from.
            # A database maintained only through gc never migrates, so it
            # must never be allowed to run destructive deletes there either.
            await _apply_destructive_schema_gate(conn, command="gc")
            if expired:
                skip_archived_gc = await _gc_expired(conn, cfg, dry_run=dry_run)
                if skip_archived_gc:
                    return
            await _gc_archived(conn, dry_run=dry_run, quiet=expired)

    _run(_gc())


# ------------------------------------------------------------------
# migrate
# ------------------------------------------------------------------


@cli.command()
@click.pass_context
def migrate(ctx: click.Context) -> None:
    """Run pending schema migrations (ensure core tables exist)."""
    cfg: EngravaCLIConfig = ctx.obj["config"]

    async def _migrate() -> None:
        from engrava.domain.exceptions import SchemaVersionError  # noqa: PLC0415
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore  # noqa: PLC0415

        # _opened_db closes the connection on any exit from this block —
        # a corrupt existing target, a store-construction failure, or a
        # failure inside ensure_schema() alike — because opening the
        # connection and entering the protected block are the same step.
        # migrate is the one built-in whose target may not exist yet —
        # aiosqlite.connect() creates the file, matching today's behaviour
        # of bootstrapping a fresh database.
        async with _opened_db(cfg) as conn:
            store = SqliteEngravaCore(conn)
            try:
                # migrate is the one built-in that calls ensure_schema()
                # unconditionally rather than through the schema-version
                # gate — that is its entire job. ensure_schema() itself
                # still refuses a populated sub-floor database or one
                # stamped above this build's head version
                # (SchemaVersionError) rather than mislabelling or silently
                # opening either; caught here so that refusal reads as a
                # clean message, not a traceback.
                await store.ensure_schema()
                await conn.commit()
            except SchemaVersionError as exc:
                click.echo(str(exc), err=True)
                sys.exit(1)
        click.echo(f"Schema up to date: {cfg.db_path}")

    _run(_migrate())


# ------------------------------------------------------------------
# export (portable JSON with thought details)
# ------------------------------------------------------------------


@cli.command(name="export")
@click.option("-o", "--output", "output_path", default=None, help="Output JSON file path.")
@click.option("--status", "status_filter", default=None, help="Filter by lifecycle_status.")
@click.pass_context
def export_cmd(ctx: click.Context, output_path: str | None, status_filter: str | None) -> None:
    """Export thoughts to a portable JSON format with edges and metadata."""
    cfg: EngravaCLIConfig = ctx.obj["config"]

    async def _export() -> None:
        if not cfg.db_path.exists():
            click.echo(f"Database not found: {cfg.db_path}")
            sys.exit(1)

        async with _opened_db(cfg) as conn:
            await _apply_read_schema_gate(conn, command="export")
            # Fetch thoughts
            if status_filter:
                cursor = await conn.execute(
                    "SELECT * FROM thought WHERE lifecycle_status = ?", (status_filter,)
                )
            else:
                cursor = await conn.execute("SELECT * FROM thought")
            keys = [desc[0] for desc in cursor.description] if cursor.description else []
            thoughts = [dict(zip(keys, row, strict=True)) for row in await cursor.fetchall()]

            # Fetch edges
            cursor = await conn.execute("SELECT * FROM edge")
            edge_keys = [desc[0] for desc in cursor.description] if cursor.description else []
            edges = [dict(zip(edge_keys, row, strict=True)) for row in await cursor.fetchall()]

            export_data = {
                "format": "engrava-export",
                "version": "0.1.0",
                "thoughts": thoughts,
                "edges": edges,
                "stats": {
                    "thought_count": len(thoughts),
                    "edge_count": len(edges),
                },
            }

            out = Path(output_path) if output_path else cfg.db_path.with_suffix(".export.json")
            out.write_text(
                json.dumps(export_data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            click.echo(f"Exported {len(thoughts)} thoughts, {len(edges)} edges to {out}")

    _run(_export())


# ------------------------------------------------------------------
# One-shot memory verbs (remember / recall / link)
# ------------------------------------------------------------------
#
# Defined in their own module — engrava.cli.memory_commands — rather than
# inline here, and registered on `cli` purely by importing it: each command
# is declared there with `@cli.command()` against the very `cli` group
# object this module defines above, so the import's only observable effect
# is that import-time decoration running. Kept as a separate module so the
# three new verbs, their shared store-resolution helper, and this file's own
# long-standing commands stay in different files -- this module already
# carries unrelated in-flight changes on sibling branches.
from engrava.cli import memory_commands as _memory_commands  # noqa: E402, F401

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``engrava`` command.

    Extension CLI commands are discovered lazily by the root group after global
    options have been parsed, so ``--no-extensions`` can prevent all entry-point
    loading.
    """
    cli()


if __name__ == "__main__":
    # Running this file directly (``python -m engrava.cli.main``, or
    # ``python path/to/main.py``) executes it as ``__main__`` — a module
    # object distinct from ``engrava.cli.main`` even though they share this
    # file's code. ``engrava.cli.memory_commands`` (imported above via the
    # dotted path) decorates the ``cli`` Group belonging to *that* import,
    # registering remember / recall / link on it — not on this run's
    # ``__main__.cli``, which therefore never gains the three new commands.
    # Re-entering through the dotted import's own ``main()`` runs the
    # canonical, fully-decorated module instead of this half-decorated one.
    from engrava.cli.main import main as _canonical_main

    _canonical_main()

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
import sys
from dataclasses import asdict
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from engrava.cli.config import EngravaCLIConfig
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
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import aiosqlite

    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol

logger = logging.getLogger(__name__)

# Re-embedding thought IDs are flushed in batches of this size so restore memory
# stays bounded by the batch rather than by the total number of thoughts.
_REEMBED_BATCH_SIZE = 128

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

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _open_db(cfg: EngravaCLIConfig) -> Any:  # noqa: ANN401
    """Open an aiosqlite connection with WAL + row_factory.

    Args:
        cfg: Resolved CLI config with db_path.

    Returns:
        An open aiosqlite Connection.

    Raises:
        click.ClickException: If the database file does not exist (for read commands).

    """
    import aiosqlite  # noqa: PLC0415

    conn = await aiosqlite.connect(str(cfg.db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    return conn


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


@click.group(cls=_ExtensionAwareGroup)
@click.option("--db", "db_path", default=None, help="Path to SQLite database.")
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

    # Pre-load services config for --service default resolution.
    services_cfg = None
    default_embeddings = None
    if cfg.config_path and cfg.config_path.exists():
        from engrava.config import load_config  # noqa: PLC0415

        ms_config = load_config(cfg.config_path)
        services_cfg = ms_config.services
        default_embeddings = ms_config.embeddings
    ctx.obj["services_config"] = services_cfg
    ctx.obj["default_embeddings"] = default_embeddings


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

        conn = await _open_db(cfg)
        try:
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
        finally:
            await conn.close()

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

        conn = await _open_db(cfg)
        try:
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
        finally:
            await conn.close()

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

        conn = await _open_db(cfg)
        try:
            from engrava.mindql.executor import MindQLExecutor  # noqa: PLC0415
            from engrava.mindql.parser import MindQLParseError, parse  # noqa: PLC0415

            # Gather extension commands from loaded extensions
            extensions = _load_mindql_extensions() if cfg.extensions_enabled else {}
            known_names = set(extensions.keys())

            try:
                parsed = parse(mql, known_extensions=known_names)
            except MindQLParseError as exc:
                click.echo(f"Parse error: {exc}", err=True)
                sys.exit(1)

            executor = MindQLExecutor(conn, extensions=extensions)
            result = await executor.execute(parsed)
            click.echo(_format_rows(result.rows, cfg.output_format, columns=result.columns))
        except MindQLParseError as exc:
            click.echo(f"Query error: {exc}", err=True)
            sys.exit(1)
        finally:
            await conn.close()

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
                store = await manager.get_store(effective_service)
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

            conn = await _open_db(cfg)
            try:
                out = (
                    Path(output_path) if output_path else cfg.db_path.with_suffix(".snapshot.jsonl")
                )
                total = await _export_db_to_jsonl(conn, out)
                click.echo(f"Exported {total} records to {out}")
            finally:
                await conn.close()

    _run(_snapshot())


# ------------------------------------------------------------------
# restore (import from JSONL snapshot)
# ------------------------------------------------------------------


def _iter_snapshot_lines(input_path: Path) -> Iterator[tuple[int, str]]:
    """Stream a snapshot file, yielding non-empty ``(line_number, line)`` pairs.

    Streaming keeps restore memory bounded by a single line rather than the
    whole snapshot. Lines are stripped and blank lines are skipped; line numbers
    are 1-based and count every physical line for accurate error context.

    Args:
        input_path: Path to the JSONL snapshot file.

    Yields:
        ``(line_number, stripped_line)`` for each non-empty line.

    """
    with input_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if stripped:
                yield line_number, stripped


def _assert_embedding_model_match(
    record: MetadataRecord,
    embedding_provider: EmbeddingProviderProtocol,
) -> None:
    """Reject a snapshot whose embedding model differs from the target's.

    Args:
        record: The snapshot metadata header.
        embedding_provider: The target embedding provider.

    Raises:
        click.ClickException: On model mismatch without an override flag.

    """
    source_model = record.embedding_model_name
    if source_model is not None and source_model != embedding_provider.model_name:
        msg = (
            f"Embedding model mismatch: snapshot has '{source_model}', "
            f"target has '{embedding_provider.model_name}'. "
            f"Use --re-embed to re-generate or --skip-embeddings to skip."
        )
        raise click.ClickException(msg)


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
    cursor = await conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding_vec'"
    )
    if await cursor.fetchone() is None:
        return

    from engrava.extensions.vector_sqlite_vec import load_sqlite_vec  # noqa: PLC0415

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
) -> None:
    """Insert one validated snapshot record via fixed, allow-listed SQL.

    The statement's column identifiers come only from the record's
    :class:`~engrava.cli.snapshot_records.TableSpec`; no identifier is derived
    from snapshot data, and every value travels as a bound parameter.

    Args:
        conn: Open aiosqlite connection.
        record: A validated core-table record.

    """
    sql, values = record.to_insert()
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


async def _stream_insert(
    conn: aiosqlite.Connection,
    input_path: Path,
    *,
    skip_embeddings: bool,
    re_embed: bool,
    embedding_provider: EmbeddingProviderProtocol | None,
) -> int:
    """Stream a snapshot once, validating and inserting each record in order.

    Each record is fully validated -- structure and values -- immediately before
    it is inserted, so a bad record raises before its own write. Re-embedding IDs
    are flushed in bounded batches; peak memory is one line plus one batch.

    Args:
        conn: Open aiosqlite connection (inside the caller's transaction).
        input_path: Path to the JSONL snapshot file.
        skip_embeddings: Skip embedding records during import.
        re_embed: Re-embed thoughts via the embedding provider after insert.
        embedding_provider: ``EmbeddingProviderProtocol`` for re-embedding.

    Returns:
        Total number of records written (inserts plus re-embeddings).

    Raises:
        click.ClickException: On a malformed record, an invalid value, or an
            embedding-model mismatch without an override flag.

    """
    check_model = embedding_provider is not None and not re_embed and not skip_embeddings
    total = 0
    reembedded = 0
    reembed_batch: list[str] = []
    for line_number, line in _iter_snapshot_lines(input_path):
        record = parse_snapshot_record(line, line_number=line_number)
        if isinstance(record, MetadataRecord):
            if check_model and embedding_provider is not None:
                _assert_embedding_model_match(record, embedding_provider)
            continue
        if not isinstance(record, TableRecord):
            continue
        if record.spec.table is CoreTable.EMBEDDING and (skip_embeddings or re_embed):
            continue

        await _insert_record(conn, record)
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
    return total


async def _import_records_to_db(
    conn: aiosqlite.Connection,
    input_path: Path,
    *,
    clear: bool = False,
    skip_embeddings: bool = False,
    re_embed: bool = False,
    embedding_provider: EmbeddingProviderProtocol | None = None,
) -> int:
    """Import JSONL records into a database connection atomically.

    The whole restore runs in a **single transaction over a single streaming
    pass**: each record is fully validated -- structure and values, including
    base64 decoding of an embedding blob -- immediately before it is inserted,
    and any failure (a malformed record, an embedding-model mismatch, or a bad
    value) rolls the transaction back so nothing is ever committed from an
    invalid snapshot. "Reject before any write" therefore holds as "nothing
    persists", including the optional ``clear``. The file is read exactly once
    and peak memory is one line plus one re-embed batch.

    Args:
        conn: Open aiosqlite connection with schema applied.
        input_path: Path to the JSONL snapshot file.
        clear: Delete existing data before import.
        skip_embeddings: Skip embedding records during import.
        re_embed: Re-embed thoughts via the embedding provider after import.
        embedding_provider: ``EmbeddingProviderProtocol`` for re-embedding.

    Returns:
        Total number of records imported.

    Raises:
        click.ClickException: On a malformed snapshot record, an invalid value,
            or an embedding-model mismatch without an override flag. The
            transaction is rolled back before the error propagates.

    """
    total = 0
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
        total = await _stream_insert(
            conn,
            input_path,
            skip_embeddings=skip_embeddings,
            re_embed=re_embed,
            embedding_provider=embedding_provider,
        )
        await conn.commit()
        committed = True
    finally:
        if not committed:
            # Any validation or insert failure discards the whole restore.
            await conn.rollback()
    return total


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
        total = await _import_records_to_db(
            store._db,  # noqa: SLF001
            Path(input_path),
            clear=clear,
            skip_embeddings=skip_embeddings,
            re_embed=re_embed,
            embedding_provider=emb_provider,
        )
        click.echo(f"Restored {total} records to service {effective_service!r} from {input_path}")
    finally:
        await manager.close_all()


async def _restore_single_db(
    *,
    input_path: str,
    clear: bool,
    skip_embeddings: bool,
    re_embed: bool,
    cfg: EngravaCLIConfig,
    default_embeddings: EmbeddingConfig | None,
) -> None:
    """Restore a JSONL snapshot into the single-file database.

    Raises:
        click.ClickException: On an embedding-provider initialisation failure or
            a missing ``--re-embed`` provider.

    """
    import aiosqlite as _aiosqlite  # noqa: PLC0415

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

    conn = await _aiosqlite.connect(str(cfg.db_path))
    conn.row_factory = _aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")

    store = SqliteEngravaCore(conn)
    await store.ensure_schema()

    try:
        total = await _import_records_to_db(
            conn,
            Path(input_path),
            clear=clear,
            skip_embeddings=skip_embeddings,
            re_embed=re_embed,
            embedding_provider=emb_provider,
        )
        click.echo(f"Restored {total} records from {input_path}")
    finally:
        await conn.close()


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
    service_name: str | None,
) -> None:
    """Restore database from a JSONL snapshot file.

    Supports model-mismatch handling via ``--re-embed`` (re-generate
    embeddings) or ``--skip-embeddings`` (import without vectors).
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
                cfg=cfg,
                default_embeddings=default_embeddings,
            )

    _run(_restore())


# ------------------------------------------------------------------
# gc (garbage-collect archived/soft-deleted thoughts)
# ------------------------------------------------------------------


async def _gc_expired(
    conn: aiosqlite.Connection,
    cfg: EngravaCLIConfig,
    *,
    dry_run: bool,
) -> bool:
    """Cleanup expired TTL thoughts.

    Returns True when the caller should skip the subsequent archived-GC
    (i.e. archive strategy was used and thoughts were moved).
    """
    from engrava.config import TTLConfig, load_config  # noqa: PLC0415
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

    result = await store.cleanup_expired()
    await conn.commit()
    click.echo(
        f"Cleaned up {result.expired_count} expired thoughts (strategy: {result.strategy_applied})."
    )
    return result.strategy_applied == "archive" and result.expired_count > 0


async def _gc_archived(
    conn: aiosqlite.Connection,
    *,
    dry_run: bool,
    quiet: bool,
) -> None:
    """Physically delete all ARCHIVED thoughts and their orphaned edges."""
    cursor = await conn.execute("SELECT COUNT(*) FROM thought WHERE lifecycle_status = 'ARCHIVED'")
    row = await cursor.fetchone()
    archived_count = row[0] if row else 0

    if archived_count == 0:
        if not quiet:
            click.echo("No archived thoughts to collect.")
        return

    if dry_run:
        click.echo(f"Would delete {archived_count} archived thoughts and orphaned edges.")
        return

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
    await conn.commit()
    click.echo(f"Collected {cursor.rowcount} archived thoughts.")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without acting.")
@click.option(
    "--expired",
    is_flag=True,
    help="Also cleanup expired TTL thoughts (archive or delete per config).",
)
@click.pass_context
def gc(ctx: click.Context, *, dry_run: bool, expired: bool) -> None:
    """Garbage-collect archived thoughts and their orphaned edges.

    With ``--expired``, also clean up expired TTL thoughts first (archived
    or deleted per the configured ``ttl.strategy``).
    """
    cfg: EngravaCLIConfig = ctx.obj["config"]

    async def _gc() -> None:
        if not cfg.db_path.exists():
            click.echo(f"Database not found: {cfg.db_path}")
            sys.exit(1)

        conn = await _open_db(cfg)
        try:
            if expired:
                skip_archived_gc = await _gc_expired(conn, cfg, dry_run=dry_run)
                if skip_archived_gc:
                    return
            await _gc_archived(conn, dry_run=dry_run, quiet=expired)
        finally:
            await conn.close()

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
        import aiosqlite  # noqa: PLC0415, I001
        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore  # noqa: PLC0415

        conn = await aiosqlite.connect(str(cfg.db_path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        store = SqliteEngravaCore(conn)
        await store.ensure_schema()
        await conn.commit()
        await conn.close()
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

        conn = await _open_db(cfg)
        try:
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
        finally:
            await conn.close()

    _run(_export())


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
    main()

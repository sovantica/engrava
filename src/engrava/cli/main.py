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
import sys
from dataclasses import asdict
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from engrava.cli.config import EngravaCLIConfig
from engrava.config import ServicesConfig
from engrava.domain.protocols.hooks import MindQLExtension
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Core tables in dependency order (thought first, dependents after).
_CORE_TABLES: tuple[str, ...] = ("thought", "edge", "embedding", "action")

# Reverse order for safe deletion (dependents first).
_CORE_TABLES_DELETE_ORDER: tuple[str, ...] = ("action", "embedding", "edge", "thought")

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
    rows: list[dict[str, Any]],
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


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------


@click.group()
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
    )
    ctx.obj["config"] = cfg

    # Pre-load services config for --service default resolution.
    services_cfg = None
    if cfg.config_path and cfg.config_path.exists():
        from engrava.config import load_config  # noqa: PLC0415

        ms_config = load_config(cfg.config_path)
        services_cfg = ms_config.services
    ctx.obj["services_config"] = services_cfg


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
            extensions = _load_mindql_extensions()
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
            "thought": "SELECT * FROM thought",
            "edge": "SELECT * FROM edge",
            "embedding": "SELECT * FROM embedding",
            "action": "SELECT * FROM action",
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
                    {"_type": table, "data": record},
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

# Valid _type values for data records (excludes "metadata" header).
_VALID_IMPORT_TYPES = frozenset(_CORE_TABLES)

# Backward compat: old snapshots used {"table": "...", "data": {...}}.
_LEGACY_FORMAT = "table"


def _parse_snapshot_line(raw_line: str) -> tuple[str, dict[str, Any]]:
    """Parse a single JSONL line into (record_type, data).

    Supports both new ``{_type, data}`` and legacy ``{table, data}``
    formats.

    Args:
        raw_line: Stripped JSON line.

    Returns:
        Tuple of (record type, data dict).  For metadata records the
        data dict is the full record.

    """
    record: dict[str, Any] = json.loads(raw_line)
    record_type: str = record.get("_type") or record.get(_LEGACY_FORMAT, "")
    if record_type == "metadata":
        return record_type, record
    data: dict[str, Any] = record.get("data", {})
    return record_type, data


async def _check_model_mismatch(
    lines: list[str],
    embedding_provider: Any,  # noqa: ANN401
    *,
    re_embed: bool,
    skip_embeddings: bool,
) -> None:
    """Validate embedding model compatibility before import.

    Args:
        lines: Raw JSONL lines from the snapshot.
        embedding_provider: Target embedding provider (or None).
        re_embed: Whether re-embedding is requested.
        skip_embeddings: Whether embeddings are skipped.

    Raises:
        click.ClickException: On model mismatch without override flags.

    """
    if embedding_provider is None or re_embed or skip_embeddings:
        return

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        record_type, data = _parse_snapshot_line(stripped)
        if record_type == "metadata":
            source_model = data.get("embedding_model_name")
            if source_model is not None and source_model != embedding_provider.model_name:
                msg = (
                    f"Embedding model mismatch: snapshot has '{source_model}', "
                    f"target has '{embedding_provider.model_name}'. "
                    f"Use --re-embed to re-generate or --skip-embeddings to skip."
                )
                raise click.ClickException(msg)
            break


async def _reembed_thoughts(
    conn: Any,  # noqa: ANN401
    thought_ids: list[str],
    embedding_provider: Any,  # noqa: ANN401
) -> int:
    """Re-embed imported thoughts via the target provider.

    Args:
        conn: Open aiosqlite connection.
        thought_ids: IDs of thoughts to re-embed.
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
        blob = struct.pack(f"<{len(vector)}f", *vector)
        now = datetime.datetime.now(tz=datetime.UTC).isoformat()
        await conn.execute(
            "INSERT OR REPLACE INTO embedding "
            "(embedding_id, owner_type, owner_id, "
            "model_name, dimension, vector_blob, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(_uuid.uuid4()),
                "thought",
                tid,
                embedding_provider.model_name,
                embedding_provider.dimension,
                blob,
                now,
            ),
        )
        count += 1
    return count


async def _insert_record(
    conn: Any,  # noqa: ANN401
    record_type: str,
    data: dict[str, Any],
) -> None:
    """Insert a single record into the database.

    Args:
        conn: Open aiosqlite connection.
        record_type: Table name (thought/edge/embedding/action).
        data: Column name-value mapping.

    """
    # Re-encode base64 blobs for embedding table.
    if record_type == "embedding" and "vector_blob" in data:
        import base64  # noqa: PLC0415

        raw = data["vector_blob"]
        if isinstance(raw, str):
            data["vector_blob"] = base64.b64decode(raw)

    cols = list(data.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    values = list(data.values())
    await conn.execute(
        f"INSERT OR REPLACE INTO {record_type} ({col_names}) VALUES ({placeholders})",  # noqa: S608
        values,
    )


async def _import_records_to_db(
    conn: Any,  # noqa: ANN401
    input_path: Path,
    *,
    clear: bool = False,
    skip_embeddings: bool = False,
    re_embed: bool = False,
    embedding_provider: Any = None,  # noqa: ANN401
) -> int:
    """Import JSONL records into a database connection.

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
        click.ClickException: On embedding model mismatch without flags.

    """
    if clear:
        for table in _CORE_TABLES_DELETE_ORDER:
            await conn.execute(f"DELETE FROM {table}")  # noqa: S608

    raw_text = input_path.read_text(encoding="utf-8")  # noqa: ASYNC240
    raw_lines = raw_text.splitlines()

    await _check_model_mismatch(
        raw_lines, embedding_provider, re_embed=re_embed, skip_embeddings=skip_embeddings
    )

    total = 0
    thought_ids_for_reembed: list[str] = []

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        record_type, data = _parse_snapshot_line(stripped)

        if record_type == "metadata" or record_type not in _VALID_IMPORT_TYPES:
            continue
        if record_type == "embedding" and (skip_embeddings or re_embed):
            continue

        if record_type == "thought" and re_embed:
            tid = data.get("thought_id")
            if tid:
                thought_ids_for_reembed.append(tid)

        await _insert_record(conn, record_type, data)
        total += 1

    # Re-embed pass.
    if re_embed and embedding_provider is not None and thought_ids_for_reembed:
        total += await _reembed_thoughts(conn, thought_ids_for_reembed, embedding_provider)

    await conn.commit()
    return total


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

    if re_embed and skip_embeddings:
        click.echo("Error: --re-embed and --skip-embeddings are mutually exclusive.", err=True)
        sys.exit(1)

    # Resolve default service from config if --service not given.
    effective_service = service_name
    if effective_service is None and services_cfg is not None:
        effective_service = services_cfg.default_service

    async def _restore() -> None:
        import aiosqlite as _aiosqlite  # noqa: PLC0415

        from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore  # noqa: PLC0415

        emb_provider = None

        if effective_service:
            from engrava.infrastructure.service_manager import (  # noqa: PLC0415
                EngravaManager,
            )

            data_dir = services_cfg.data_dir if services_cfg else cfg.db_path.parent
            manager = EngravaManager(
                data_dir=data_dir,
                services_config=services_cfg,
            )
            try:
                store = await manager.get_store(effective_service)
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
                click.echo(
                    f"Restored {total} records to service {effective_service!r} from {input_path}"
                )
            finally:
                await manager.close_all()
        else:
            conn = await _aiosqlite.connect(str(cfg.db_path))
            conn.row_factory = _aiosqlite.Row
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA foreign_keys = ON")

            store = SqliteEngravaCore(conn)
            await store.ensure_schema()

            if re_embed:
                emb_provider = store._embedding_provider  # noqa: SLF001
                if emb_provider is None:
                    await conn.close()
                    msg = (
                        "--re-embed requires an embedding provider, but none is "
                        "configured. Set 'embeddings.provider' in engrava.yaml "
                        "or use --skip-embeddings instead."
                    )
                    raise click.ClickException(msg)

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

    Discovers and registers extension CLI commands before invoking
    the main CLI group.
    """
    for cmd in _discover_extension_commands():
        cli.add_command(cmd)
    cli()


if __name__ == "__main__":
    main()

"""SqliteEngravaCore — core SQLite thought-graph persistence.

Provides async CRUD for thoughts, edges, actions, and brute-force
embedding similarity search.  All SQL uses parameterized queries.

The ``_row_to_thought`` method is an overridable template method —
subclasses can override it to produce richer model types while
reusing all core SQL logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import hashlib
import json
import logging
import math
import re
import struct
import uuid as _uuid
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Self

import aiosqlite
import numpy as np

from engrava.domain.enums import (
    ActionStatus,
    ActionType,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
    VerificationStatus,
)
from engrava.domain.exceptions import (
    EmbeddingModelMismatchError,
    ReferentialIntegrityError,
    StaleDataError,
    ThoughtNotFoundError,
)
from engrava.domain.models._temporal import validate_iso8601_nullable
from engrava.domain.models.action import ActionRecord
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.embedding import EmbeddingRecord
from engrava.domain.models.thought import MetadataValue, ThoughtRecord
from engrava.domain.models.ttl import CleanupResult, CleanupStrategy
from engrava.domain.protocols.hooks import DefaultEngravaHooks, EngravaHooksProtocol
from engrava.infrastructure.sqlite.centroid import CENTROID_MODEL_NAME, compute_centroid
from engrava.infrastructure.sqlite.journal_writer import JournalWriter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from engrava.config import MetricsConfig, SearchConfig
    from engrava.domain.manifest import ExtensionManifest
    from engrava.domain.models.metrics import EngravaMetrics, LatencyHistogram
    from engrava.domain.models.search import HybridSearchResult
    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol
    from engrava.domain.protocols.hooks import MindQLExtension
    from engrava.extensions.vector_sqlite_vec import SqliteVecSearchBackend
    from engrava.mindql.executor import MindQLResult
    from engrava.mindql.parser import MindQLQuery

logger = logging.getLogger(__name__)

_FTS_FIELD_FILTER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:.+")
_FTS_UNSAFE_CHAR_RE = re.compile(r"[^\w\-*]")
_SUPPRESS_SEARCH_METRICS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "engrava_suppress_search_metrics",
    default=False,
)


class _LatencyRingBuffer:
    """Fixed-size async-safe ring buffer of recent search latencies."""

    def __init__(self, window_size: int = 1000) -> None:
        self._window_size = window_size
        self._buf: list[float] = []
        self._lock = asyncio.Lock()

    async def record(self, latency_ms: float) -> None:
        """Append a latency sample, evicting the oldest entry if needed."""
        async with self._lock:
            if len(self._buf) >= self._window_size:
                self._buf.pop(0)
            self._buf.append(latency_ms)

    async def snapshot(self) -> LatencyHistogram:
        """Return percentile statistics for the current window."""
        from engrava.domain.models.metrics import LatencyHistogram  # noqa: PLC0415

        async with self._lock:
            samples = list(self._buf)

        if not samples:
            return LatencyHistogram()

        arr = np.asarray(samples, dtype=np.float64)
        return LatencyHistogram(
            sample_count=len(samples),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            mean_ms=float(np.mean(arr)),
        )


class SqliteEngravaCore:
    """Core SQLite persistence backend for thought-graph CRUD.

    Uses manual SQL with parameterized queries — no ORM.

    Supports transaction-aware commit control: when ``_skip_auto_commit``
    is ``True`` (set via :meth:`suspend_auto_commit`), individual methods
    skip ``db.commit()`` so the caller manages the commit boundary.

    Subclasses can override ``_row_to_thought`` to produce extended model
    types (template method pattern).

    Args:
        db: An open aiosqlite connection (WAL mode, FK enabled).
        hooks: Optional extension hooks; defaults to ``DefaultEngravaHooks``.
        embedding_provider: Optional async embedding provider for auto-embed.
        auto_embed: Whether to auto-embed on ``create_thought``/``update_thought``.
        search_config: Optional default hybrid-search weights from config.
        journal_enabled: Whether to record mutations in the hash-chain
            journal.  Defaults to ``False``.
        ttl_strategy: Cleanup strategy for expired thoughts.
            ``"archive"`` (default) or ``"delete"``.
        ttl_check_every_n: Auto-cleanup cadence.  ``0`` disables.
        ttl_default_seconds: Default TTL for new thoughts.  ``None``
            means thoughts do not expire unless explicitly set.
        manifests: Extension manifests whose ``schema_migrations`` will be
            applied after core schema bootstrap.  Pass an empty
            sequence (default) to skip extension migrations entirely.

    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        hooks: EngravaHooksProtocol | None = None,
        *,
        embedding_provider: EmbeddingProviderProtocol | None = None,
        auto_embed: bool = False,
        search_config: SearchConfig | None = None,
        journal_enabled: bool = False,
        ttl_strategy: str = "archive",
        ttl_check_every_n: int = 0,
        ttl_default_seconds: int | None = None,
        metrics_config: MetricsConfig | None = None,
        manifests: Sequence[ExtensionManifest] = (),
    ) -> None:
        self._db = db
        self._hooks: EngravaHooksProtocol = hooks or DefaultEngravaHooks()
        self._skip_auto_commit: bool = False
        self._fts_available: bool = False
        self._fts_probed: bool = False
        self._vector_backend: SqliteVecSearchBackend | None = None
        self._owns_connection: bool = False
        self._embedding_provider: EmbeddingProviderProtocol | None = embedding_provider
        self._auto_embed: bool = auto_embed and embedding_provider is not None
        self._embedding_model_verified: bool = False
        self._search_config: SearchConfig | None = search_config
        self._journal_enabled: bool = journal_enabled
        self._journal: JournalWriter | None = JournalWriter(db) if journal_enabled else None
        self._ttl_strategy: str = ttl_strategy
        self._ttl_check_every_n: int = ttl_check_every_n
        self._ttl_default_seconds: int | None = ttl_default_seconds
        if metrics_config is not None:
            self._metrics_config = metrics_config
        else:
            from engrava.config import MetricsConfig  # noqa: PLC0415

            self._metrics_config = MetricsConfig()
        self._latency_buffer = _LatencyRingBuffer(self._metrics_config.window_size)
        self._operation_count: int = 0
        self._manifests: tuple[ExtensionManifest, ...] = tuple(manifests)
        # Serialises the dedup ``check existing -> INSERT or UPDATE``
        # sequence so concurrent ``create_thought(deduplicate=True)`` calls
        # converge on a single row even though aiosqlite does not expose
        # row-level locking.  Acquired only on the dedup branch — the
        # legacy ``deduplicate=False`` path stays lock-free.
        self._dedup_lock: asyncio.Lock = asyncio.Lock()

    @property
    def journal(self) -> JournalWriter | None:
        """Return the ``JournalWriter`` if journaling is enabled, else ``None``.

        Returns:
            The active journal writer, or ``None``.

        """
        return self._journal

    async def _record_search_latency(self, latency_ms: float) -> None:
        """Record a completed public-search latency when metrics are enabled."""
        if self._metrics_config.enabled and not _SUPPRESS_SEARCH_METRICS.get():
            await self._latency_buffer.record(latency_ms)

    async def _main_db_path(self) -> Path | None:
        """Resolve the main SQLite file path from the active connection."""
        cursor = await self._db.execute("PRAGMA database_list")
        rows = await cursor.fetchall()
        for row in rows:
            if str(row[1]) == "main" and row[2]:
                return Path(str(row[2]))
        return None

    async def _storage_footprint(self) -> tuple[int, int, int, int]:
        """Return ``(db_bytes, wal_bytes, vec_index_bytes, total_bytes)``."""
        db_path = await self._main_db_path()
        if db_path is None:
            return (0, 0, 0, 0)

        db_bytes = db_path.stat().st_size if db_path.exists() else 0
        wal_path = Path(f"{db_path}-wal")
        wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0  # noqa: ASYNC240
        vec_index_bytes = 0
        total_bytes = db_bytes + wal_bytes + vec_index_bytes
        return (db_bytes, wal_bytes, vec_index_bytes, total_bytes)

    async def metrics(self) -> EngravaMetrics:
        """Return a point-in-time snapshot of store health and workload metrics."""
        import time  # noqa: PLC0415

        from engrava.domain.models.metrics import (  # noqa: PLC0415
            EdgeCounts,
            EngravaMetrics,
            StorageFootprint,
            ThoughtCounts,
        )

        snapshot_ts = time.time()
        if not self._metrics_config.enabled:
            return EngravaMetrics(snapshot_timestamp=snapshot_ts)

        thought_by_type_cursor = await self._db.execute(
            "SELECT thought_type, COUNT(*) FROM thought GROUP BY thought_type"
        )
        thought_by_type_rows = await thought_by_type_cursor.fetchall()
        thought_by_type = {str(row[0]): int(row[1]) for row in thought_by_type_rows}

        thought_by_status_cursor = await self._db.execute(
            "SELECT lifecycle_status, COUNT(*) FROM thought GROUP BY lifecycle_status"
        )
        thought_by_status_rows = await thought_by_status_cursor.fetchall()
        thought_by_status = {str(row[0]): int(row[1]) for row in thought_by_status_rows}

        thought_total_cursor = await self._db.execute("SELECT COUNT(*) FROM thought")
        thought_total_row = await thought_total_cursor.fetchone()
        thought_total = int(thought_total_row[0]) if thought_total_row is not None else 0

        edge_by_type_cursor = await self._db.execute(
            "SELECT edge_type, COUNT(*) FROM edge GROUP BY edge_type"
        )
        edge_by_type_rows = await edge_by_type_cursor.fetchall()
        edge_by_type = {str(row[0]): int(row[1]) for row in edge_by_type_rows}

        edge_total_cursor = await self._db.execute("SELECT COUNT(*) FROM edge")
        edge_total_row = await edge_total_cursor.fetchone()
        edge_total = int(edge_total_row[0]) if edge_total_row is not None else 0

        db_bytes, wal_bytes, vec_index_bytes, total_bytes = await self._storage_footprint()
        latency_snapshot = await self._latency_buffer.snapshot()

        return EngravaMetrics(
            snapshot_timestamp=snapshot_ts,
            thoughts=ThoughtCounts(
                by_type=thought_by_type,
                by_status=thought_by_status,
                total=thought_total,
            ),
            edges=EdgeCounts(by_type=edge_by_type, total=edge_total),
            storage=StorageFootprint(
                db_bytes=db_bytes,
                wal_bytes=wal_bytes,
                vec_index_bytes=vec_index_bytes,
                total_bytes=total_bytes,
            ),
            search_latency=latency_snapshot,
        )

    # ------------------------------------------------------------------
    # Factory + async context manager
    # ------------------------------------------------------------------

    @classmethod
    async def from_config(
        cls,
        config_path: str | Path,
    ) -> Self:
        """Create a fully configured instance from a YAML config file.

        The returned instance **owns** the database connection and should
        be used as an async context manager to ensure proper cleanup::

            async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
                thought = await store.get_thought("abc")

        The manual constructor ``SqliteEngravaCore(db, hooks=...)``
        still works unchanged — the caller owns the connection in that case.

        Args:
            config_path: Filesystem path to ``engrava.yaml``.

        Returns:
            A configured ``SqliteEngravaCore`` with schema applied.

        Raises:
            ConfigError: If the config file is invalid.

        """
        from engrava.config import load_config, resolve_hooks  # noqa: PLC0415

        config = load_config(config_path)
        db = await aiosqlite.connect(str(config.database_path))
        try:
            if config.wal_mode:
                await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            db.row_factory = aiosqlite.Row

            hooks = resolve_hooks(config.hooks_class)

            # Resolve embedding provider from config.
            from engrava.config import (  # noqa: PLC0415
                resolve_embedding_provider,
                resolve_manifests,
            )

            emb_provider = resolve_embedding_provider(config.embeddings)
            auto_embed = config.embeddings.auto_embed if config.embeddings else False

            manifests = resolve_manifests(
                config.extension_manifest_paths,
                discover=config.extension_discover,
            )

            store = cls(
                db,
                hooks=hooks,
                embedding_provider=emb_provider,
                auto_embed=auto_embed,
                search_config=config.search,
                journal_enabled=config.journal.enabled,
                ttl_strategy=config.ttl.strategy,
                ttl_check_every_n=config.ttl.check_every_n_operations,
                ttl_default_seconds=config.ttl.default_ttl_seconds,
                metrics_config=config.metrics,
                manifests=manifests,
            )
            store._owns_connection = True
            await store.ensure_schema()

            await store._configure_vector_backend(
                backend_name=config.vector_backend,
                embedding_dimension=config.embedding_dimension,
            )
        except Exception:
            await db.close()
            raise

        return store

    async def close(self) -> None:
        """Close the database connection if owned by this instance.

        No-op when the connection is caller-managed (i.e. created via
        the manual constructor).
        """
        if self._owns_connection:
            await self._db.close()

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        Returns:
            This store instance.

        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit the async context manager and close if owned.

        Args:
            *exc: Exception info (type, value, traceback).

        """
        await self.close()

    async def _configure_vector_backend(
        self,
        *,
        backend_name: str,
        embedding_dimension: int,
    ) -> None:
        """Configure the requested vector backend with graceful fallback.

        Args:
            backend_name: Backend identifier from config.
            embedding_dimension: Expected embedding vector dimension.

        """
        backend_handlers = {
            "numpy": self._configure_numpy_vector_backend,
            "sqlite-vec": self._configure_sqlite_vec_vector_backend,
        }
        await backend_handlers[backend_name](embedding_dimension)

    async def _configure_numpy_vector_backend(self, embedding_dimension: int) -> None:
        """Use the default numpy brute-force vector search backend.

        Args:
            embedding_dimension: Expected embedding vector dimension.

        """
        del embedding_dimension
        self._vector_backend = None

    async def _configure_sqlite_vec_vector_backend(self, embedding_dimension: int) -> None:
        """Configure the sqlite-vec backend when the extension is available.

        Falls back to numpy when the extension cannot be loaded.

        Args:
            embedding_dimension: Expected embedding vector dimension.

        """
        from engrava.extensions.vector_sqlite_vec import (  # noqa: PLC0415
            SqliteVecSearchBackend,
            load_sqlite_vec,
        )

        loaded = await load_sqlite_vec(self._db)
        if not loaded:
            logger.warning("sqlite-vec requested but unavailable — using numpy fallback")
            self._vector_backend = None
            return

        backend = SqliteVecSearchBackend(embedding_dimension)
        await backend.ensure_index(self._db)
        await backend.sync_embeddings(self._db)
        self._vector_backend = backend

    # ------------------------------------------------------------------
    # Schema bootstrap (standalone usage)
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Create core tables if they don't already exist.

        Applies the full ``schema_core.sql`` (including FTS5 virtual
        table and sync triggers) only when the database has not already
        been bootstrapped to schema version 3+.  Databases at older
        versions are upgraded incrementally up to the current version (13).

        After core schema creation or upgrade, probes for the ``thought_fts``
        table and then runs any pending extension schema migrations for each
        manifest supplied via the ``manifests`` constructor parameter.
        """
        cursor = await self._db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current_version = int(row[0]) if row else 0

        if current_version < 2:  # noqa: PLR2004
            schema_sql = (
                resources.files("engrava.infrastructure.sqlite")
                .joinpath("schema_core.sql")
                .read_text(encoding="utf-8")
            )
            await self._db.executescript(schema_sql)
        elif current_version < 3:  # noqa: PLR2004
            await self._rebuild_fts_index()
            await self._migrate_core_v3_to_v4()
            await self._migrate_core_v4_to_v5()
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 4:  # noqa: PLR2004
            await self._migrate_core_v3_to_v4()
            await self._migrate_core_v4_to_v5()
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 5:  # noqa: PLR2004
            await self._migrate_core_v4_to_v5()
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 6:  # noqa: PLR2004
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 7:  # noqa: PLR2004
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 8:  # noqa: PLR2004
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 9:  # noqa: PLR2004
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 10:  # noqa: PLR2004
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 11:  # noqa: PLR2004
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 12:  # noqa: PLR2004
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()
        elif current_version < 13:  # noqa: PLR2004
            await self._migrate_core_v12_to_v13()
            await self._db.execute("PRAGMA user_version = 13")
            await self._db.commit()

        # Ensure referential integrity is enforced for the lifetime of this
        # connection. SQLite ships with foreign_keys=OFF by default, so any
        # caller that constructs SqliteEngravaCore directly (rather than via
        # the from_config factory) would otherwise miss FK enforcement even
        # though the schema declares the constraints (core-12).
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._probe_fts()

        # Apply extension schema migrations.
        if self._manifests:
            from engrava.infrastructure.sqlite.extension_migrations import (  # noqa: PLC0415
                ExtensionMigrationRunner,
            )

            runner = ExtensionMigrationRunner()
            for _manifest in self._manifests:
                await runner.apply_pending(_manifest, self._db)

    async def _probe_fts(self) -> None:
        """Detect whether the ``thought_fts`` FTS5 table exists.

        Sets ``_fts_available`` to ``True`` when the virtual table is
        present, ``False`` otherwise.  Called once during schema bootstrap
        to avoid repeated introspection on every ``search_fts()`` call.
        """
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='thought_fts'"
        )
        self._fts_available = (await cursor.fetchone()) is not None
        self._fts_probed = True

    async def _rebuild_fts_index(self) -> None:
        """Rebuild the FTS5 index with the hyphen-aware tokenizer.

        This upgrade path is used for existing core schema version 2
        databases whose original FTS5 configuration treated ``-`` as an
        operator, breaking prefix searches for identifier-like terms.
        """
        await self._db.execute("DROP TRIGGER IF EXISTS thought_fts_insert")
        await self._db.execute("DROP TRIGGER IF EXISTS thought_fts_delete")
        await self._db.execute("DROP TRIGGER IF EXISTS thought_fts_update")
        await self._db.execute("DROP TABLE IF EXISTS thought_fts")
        await self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS thought_fts USING fts5("
            "  essence, content,"
            "  tokenize = \"unicode61 tokenchars '-_'\","
            "  content='thought', content_rowid='rowid'"
            ")"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS thought_fts_insert "
            "AFTER INSERT ON thought BEGIN "
            "  INSERT INTO thought_fts(rowid, essence, content) "
            "  VALUES (new.rowid, new.essence, new.content); "
            "END"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS thought_fts_delete "
            "AFTER DELETE ON thought BEGIN "
            "  INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
            "  VALUES ('delete', old.rowid, old.essence, old.content); "
            "END"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS thought_fts_update "
            "AFTER UPDATE OF essence, content ON thought BEGIN "
            "  INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
            "  VALUES ('delete', old.rowid, old.essence, old.content); "
            "  INSERT INTO thought_fts(rowid, essence, content) "
            "  VALUES (new.rowid, new.essence, new.content); "
            "END"
        )
        await self._db.execute(
            "INSERT OR IGNORE INTO thought_fts(rowid, essence, content) "
            "SELECT rowid, essence, content FROM thought"
        )

    async def _migrate_core_v3_to_v4(self) -> None:
        """Add access tracking and datetime timestamp columns (core-4).

        Idempotent — safe to run on a database that already has the columns.
        Backfills ``created_at`` and ``updated_at`` with the current UTC
        time for existing rows that lack timestamps.
        """
        from sqlite3 import OperationalError  # noqa: PLC0415

        alter_statements = [
            "ALTER TABLE thought ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE thought ADD COLUMN last_accessed_at TEXT",
            "ALTER TABLE thought ADD COLUMN created_at TEXT",
            "ALTER TABLE thought ADD COLUMN updated_at TEXT",
        ]
        for stmt in alter_statements:
            with contextlib.suppress(OperationalError):
                await self._db.execute(stmt)

        now = datetime.datetime.now(datetime.UTC).isoformat()
        await self._db.execute(
            "UPDATE thought SET created_at = ?, updated_at = ? WHERE created_at IS NULL",
            (now, now),
        )
        await self._db.execute(
            "UPDATE thought SET updated_at = ? WHERE updated_at IS NULL",
            (now,),
        )

    async def _migrate_core_v4_to_v5(self) -> None:
        """Add the ``_metadata`` key/value table (core-5).

        Idempotent — uses ``CREATE TABLE IF NOT EXISTS``.
        """
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS _metadata (key TEXT PRIMARY KEY, value TEXT)"
        )

    async def _migrate_core_v5_to_v6(self) -> None:
        """Add the ``journal_entry`` table and indexes (core-6).

        Idempotent — uses ``CREATE TABLE IF NOT EXISTS`` and
        ``CREATE INDEX IF NOT EXISTS``.
        """
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS journal_entry ("
            "  entry_id         TEXT PRIMARY KEY,"
            "  sequence_number  INTEGER NOT NULL UNIQUE,"
            "  mutation_type    TEXT NOT NULL,"
            "  target_id        TEXT,"
            "  delta            TEXT NOT NULL,"
            "  parent_hash      TEXT,"
            "  entry_hash       TEXT NOT NULL,"
            "  created_at       TEXT NOT NULL"
            ")"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_target "
            "ON journal_entry(target_id, sequence_number)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entry(mutation_type)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_seq ON journal_entry(sequence_number)"
        )

    async def _migrate_core_v6_to_v7(self) -> None:
        """Add ``expires_at`` column and partial index (core-7).

        Idempotent — silently skips when the column already exists.
        """
        with contextlib.suppress(Exception):  # Column may already exist.
            await self._db.execute("ALTER TABLE thought ADD COLUMN expires_at TEXT")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_expires "
            "ON thought(expires_at) WHERE expires_at IS NOT NULL"
        )

    async def _migrate_core_v7_to_v8(self) -> None:
        """Add composite edge index for candidate expansion queries (core-8).

        Supports ``_expand_via_consolidated_from`` which queries::

            SELECT ... FROM edge
            WHERE edge_type = 'CONSOLIDATED_FROM'
            AND from_thought_id IN (...)

        Without this index SQLite performs a full table scan on the edge
        table, which breaks the p95 < 50 ms latency requirement at scale.
        The same index also accelerates the existing ``_load_graph_signal``
        COUNT query backing the giant-cluster guard.

        Idempotent — uses ``CREATE INDEX IF NOT EXISTS``.  Silently skips
        when the ``edge`` table does not yet exist (partial-schema test
        environments or future migrations that reorder DDL).
        """
        with contextlib.suppress(Exception):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id)"
            )

    async def _migrate_core_v8_to_v9(self) -> None:
        """Add extension_schema_versions table (core-9).

        Tracks which SQL migration files have been applied for each
        installed extension.  The table is created with
        ``CREATE TABLE IF NOT EXISTS``, so this migration is fully
        idempotent.
        """
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS extension_schema_versions (
                extension_name    TEXT PRIMARY KEY,
                version           INTEGER NOT NULL DEFAULT 0,
                applied_at        REAL NOT NULL,
                migration_file    TEXT,
                extension_version TEXT
            )
            """
        )

    async def _migrate_core_v9_to_v10(self) -> None:
        """Add ``content_hash`` column + index to ``thought`` table (core-10).

        Adds a nullable ``content_hash TEXT`` column and the
        ``idx_thought_content_hash`` index used by opt-in ingest
        deduplication (``create_thought(..., deduplicate=True)``).

        Idempotent: ``ALTER TABLE ADD COLUMN`` is wrapped in
        duplicate-column tolerance and ``CREATE INDEX`` uses
        ``IF NOT EXISTS``, so re-running the migration after a partial
        crash converges on the fully-applied state.

        Existing rows are left with ``content_hash = NULL`` until the
        bundled backfill utility populates them; new ``create_thought``
        calls compute the hash at insert time regardless of the
        ``deduplicate`` flag.
        """
        try:
            await self._db.execute("ALTER TABLE thought ADD COLUMN content_hash TEXT")
        except aiosqlite.OperationalError as exc:
            # SQLite emits "duplicate column name: content_hash" when the
            # column already exists; that is the idempotent re-run signal.
            if "duplicate column" not in str(exc).lower():
                raise

        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash)"
        )

    async def _migrate_core_v10_to_v11(self) -> None:
        """Add ``metadata_json`` column to ``thought`` table (core-11).

        Adds a NOT NULL ``metadata_json TEXT`` column with default
        ``'{}'`` to support structured metadata (role, lang,
        content_type, session_id, ...).  Existing rows get the
        empty-dict default — no data migration required.

        Idempotent: ``ALTER TABLE ADD COLUMN`` is wrapped in
        duplicate-column tolerance (matches ``_migrate_core_v9_to_v10``
        precedent), so re-running the migration after a partial crash
        converges on the fully-applied state.
        """
        try:
            await self._db.execute(
                "ALTER TABLE thought ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
            )
        except aiosqlite.OperationalError as exc:
            # SQLite emits "duplicate column name: metadata_json" when
            # the column already exists; that is the idempotent re-run
            # signal.
            if "duplicate column" not in str(exc).lower():
                raise

    async def _migrate_core_v11_to_v12(self) -> None:
        """Add referential integrity (FK + ON DELETE CASCADE) to child tables.

        SQLite does not support ``ALTER TABLE ADD CONSTRAINT`` so the
        FK clauses on ``edge``, ``embedding`` and ``action`` are
        introduced via the recreate-table pattern: build a new table
        with the FK declaration, copy rows over, drop the old table,
        rename the new one, and rebuild any indexes that the schema
        declared on it.

        ``PRAGMA foreign_keys=OFF`` is a documented no-op while a
        transaction is open. The dispatch chain in ``ensure_schema``
        may arrive here with prior migration writes still in an open
        implicit transaction; this helper therefore commits any pending
        work *before* toggling the pragma, runs the recreate steps,
        commits the recreations, and only then re-enables enforcement.
        Without this, the ladder path (older schema → … → v12) leaves
        FK enforcement on during the swap and the recreated tables
        fail their first ``INSERT … SELECT *`` if any unpurged orphan
        remains.

        Pre-existing orphan rows in any of the three child tables are
        purged before the constraint is enabled. Orphans are already
        invalid against the documented invariant (``ON DELETE CASCADE``
        on the parent) and the documented contract on
        ``delete_thought`` — keeping them would block enabling the FK.
        The purge is unconditional on the FK column (no
        ``owner_type='THOUGHT'`` gate on the embedding side) because
        the FK does not look at ``owner_type``; a stray lowercase or
        non-THOUGHT owner that does not resolve to a thought would
        otherwise survive the purge and break the recreate.

        Idempotent: the helper detects whether each table already
        carries the FK declaration via ``PRAGMA foreign_key_list`` and
        skips the recreation when the constraint is already present.
        Re-running ``ensure_schema`` on a freshly migrated database is
        therefore a no-op for this step. A partial earlier run that
        upgraded some tables and not others resumes per-table on the
        next call.
        """
        edge_exists = await self._table_exists("edge")
        embedding_exists = await self._table_exists("embedding")
        action_exists = await self._table_exists("action")
        edge_done = edge_exists and await self._fk_present("edge", "from_thought_id")
        embedding_done = embedding_exists and await self._fk_present("embedding", "owner_id")
        action_done = action_exists and await self._fk_present("action", "source_thought_id")
        # Tables absent from a partial bootstrap (only `thought` present) are
        # treated as "nothing to migrate" — fresh installs receive the FK
        # directly from ``schema_core.sql``.
        if (
            (edge_done or not edge_exists)
            and (embedding_done or not embedding_exists)
            and (action_done or not action_exists)
        ):
            return

        # Close any implicit transaction opened by prior migration steps
        # so PRAGMA foreign_keys=OFF actually takes effect (the pragma
        # is silently ignored while a transaction is open).
        await self._db.commit()
        await self._db.execute("PRAGMA foreign_keys=OFF")
        try:
            await self._purge_orphan_children()
            if edge_exists and not edge_done:
                await self._recreate_edge_with_fk()
            if embedding_exists and not embedding_done:
                await self._recreate_embedding_with_fk()
            if action_exists and not action_done:
                await self._recreate_action_with_fk()
            # Commit the recreate steps before re-enabling FK so the
            # ON pragma also lands outside a transaction.
            await self._db.commit()
        finally:
            await self._db.execute("PRAGMA foreign_keys=ON")

    async def _migrate_core_v12_to_v13(self) -> None:
        """Add nullable valid-time columns + indexes to thought and edge (core-13).

        Introduces a second time axis ("valid time") alongside the
        existing transaction time. ``created_at`` records *when a fact was
        stored*; ``valid_from`` / ``valid_until`` record *when a fact is
        true in the world*. Both new columns are nullable ISO-8601 TEXT.

        Backfill is intentionally asymmetric:

        * ``thought.valid_from`` is seeded from ``created_at`` for rows
          that have a transaction timestamp, giving existing thoughts a
          sensible default lower bound. ``valid_until`` is left ``NULL``
          (open upper bound). Rows whose ``created_at`` is ``NULL``
          (pre-timestamp legacy rows) keep ``valid_from = NULL`` — no
          date is fabricated.
        * ``edge`` rows are **not** backfilled. The edge table has no
          ``created_at`` column; its only temporal field is
          ``created_cycle``, which is an internal cognitive-cycle counter,
          not a calendar timestamp. Synthesising a valid-time date from a
          cycle number would invent information, so edges keep both
          valid-time fields ``NULL`` (an open lower bound).

        Idempotent: each ``ALTER TABLE ... ADD COLUMN`` is wrapped in
        ``contextlib.suppress(Exception)`` so a re-run after the column
        already exists is a no-op, and every index uses
        ``CREATE INDEX IF NOT EXISTS``. Re-running the migration leaves
        the schema unchanged.
        """
        # Only touch tables that exist. A partial bootstrap may carry just
        # ``thought`` (the ``edge`` table is created lazily); operating on an
        # absent ``edge`` would raise ``no such table``. ``thought`` is always
        # present at this point. This mirrors the table-existence guards used
        # by the earlier edge-touching migrations and ``_purge_orphan_children``.
        tables = ["thought"]
        if await self._table_exists("edge"):
            tables.append("edge")

        for table in tables:
            for column in ("valid_from", "valid_until"):
                with contextlib.suppress(Exception):  # Column may already exist.
                    await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

        # Asymmetric backfill — thought only, sourced from transaction time.
        # Rows with NULL created_at (legacy, pre-timestamp) keep NULL
        # valid_from; no calendar date is fabricated for them.
        await self._db.execute(
            "UPDATE thought SET valid_from = created_at "
            "WHERE created_at IS NOT NULL AND valid_from IS NULL"
        )
        # Edge has no created_at; created_cycle is internal cognitive time,
        # not calendar time, so edges are deliberately left with NULL
        # valid_from / valid_until (an open lower bound).

        for table in tables:
            await self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_valid_from ON {table}(valid_from)"
            )
            await self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_valid_until "
                f"ON {table}(valid_until) WHERE valid_until IS NOT NULL"
            )
            await self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_valid_range "
                f"ON {table}(valid_from, valid_until)"
            )

    async def _fk_present(self, table: str, column: str) -> bool:
        """Return ``True`` when ``table`` carries an FK on ``column``."""
        cursor = await self._db.execute(f"PRAGMA foreign_key_list({table})")
        rows = await cursor.fetchall()
        return any(row["from"] == column for row in rows)

    async def _table_exists(self, table: str) -> bool:
        """Return ``True`` when ``table`` is registered in ``sqlite_master``."""
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        )
        return await cursor.fetchone() is not None

    async def _purge_orphan_children(self) -> None:
        """Delete orphan rows whose parent thought no longer exists.

        Runs before FK enablement so the constraint can be added
        without rejecting existing-but-invalid data. Counts of removed
        rows are intentionally not surfaced: orphans are bugs by
        definition and the migration runs once in pre-publish. Tables
        absent from a partial bootstrap are skipped.
        """
        if await self._table_exists("edge"):
            await self._db.execute(
                "DELETE FROM edge "
                "WHERE from_thought_id NOT IN (SELECT thought_id FROM thought) "
                "   OR to_thought_id   NOT IN (SELECT thought_id FROM thought)",
            )
        if await self._table_exists("embedding"):
            # Unconditional on owner_id: the FK does not branch on
            # owner_type, so any owner_id that fails to resolve to a
            # thought is an orphan against the new constraint
            # regardless of the (case-variant) owner_type value.
            await self._db.execute(
                "DELETE FROM embedding WHERE owner_id NOT IN (SELECT thought_id FROM thought)",
            )
        if await self._table_exists("action"):
            await self._db.execute(
                "DELETE FROM action "
                "WHERE source_thought_id NOT IN (SELECT thought_id FROM thought)",
            )

    async def _recreate_edge_with_fk(self) -> None:
        """Recreate ``edge`` with FK + CASCADE on both endpoints."""
        await self._db.execute(
            "CREATE TABLE edge_new ("
            "  edge_id           TEXT PRIMARY KEY,"
            "  from_thought_id   TEXT NOT NULL,"
            "  to_thought_id     TEXT NOT NULL,"
            "  edge_type         TEXT NOT NULL,"
            "  weight            REAL NOT NULL DEFAULT 0.5,"
            "  created_cycle     INTEGER NOT NULL DEFAULT 0,"
            "  source            TEXT NOT NULL DEFAULT 'EXPERIENCE',"
            "  decay_multiplier  REAL NOT NULL DEFAULT 1.0,"
            "  UNIQUE(from_thought_id, to_thought_id, edge_type),"
            "  FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE,"
            "  FOREIGN KEY (to_thought_id)   REFERENCES thought(thought_id) ON DELETE CASCADE"
            ")",
        )
        await self._db.execute(
            "INSERT INTO edge_new SELECT "
            "  edge_id, from_thought_id, to_thought_id, edge_type, weight, "
            "  created_cycle, source, decay_multiplier "
            "FROM edge",
        )
        await self._db.execute("DROP TABLE edge")
        await self._db.execute("ALTER TABLE edge_new RENAME TO edge")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id)",
        )

    async def _recreate_embedding_with_fk(self) -> None:
        """Recreate ``embedding`` with FK + CASCADE on ``owner_id``.

        The polymorphic ``owner_type`` column is preserved for forward
        compatibility, but every persisted embedding currently uses
        ``owner_type='THOUGHT'`` so the FK targets ``thought`` directly.
        """
        await self._db.execute(
            "CREATE TABLE embedding_new ("
            "  embedding_id TEXT PRIMARY KEY,"
            "  owner_type   TEXT    NOT NULL,"
            "  owner_id     TEXT    NOT NULL,"
            "  model_name   TEXT    NOT NULL,"
            "  dimension    INTEGER NOT NULL,"
            "  vector_blob  BLOB    NOT NULL,"
            "  created_at   TEXT    NOT NULL,"
            "  FOREIGN KEY (owner_id) REFERENCES thought(thought_id) ON DELETE CASCADE"
            ")",
        )
        await self._db.execute("INSERT INTO embedding_new SELECT * FROM embedding")
        await self._db.execute("DROP TABLE embedding")
        await self._db.execute("ALTER TABLE embedding_new RENAME TO embedding")

    async def _identify_orphan_endpoint(self, edge: EdgeRecord) -> tuple[str, str]:
        """Return the offending ``(column, referenced_id)`` pair after a FK reject.

        SQLite reports a generic "FOREIGN KEY constraint failed" without
        naming the column or value. The endpoints are checked in left-
        to-right order (``from_thought_id`` first); when both endpoints
        are missing the function reports the first one — sufficient
        signal for callers, and consistent with how SQLite itself
        reports a single constraint violation per row.
        """
        if not await self._thought_exists(edge.from_thought_id):
            return "from_thought_id", edge.from_thought_id
        return "to_thought_id", edge.to_thought_id

    async def _thought_exists(self, thought_id: str) -> bool:
        """Return ``True`` when a thought with ``thought_id`` is persisted."""
        cursor = await self._db.execute(
            "SELECT 1 FROM thought WHERE thought_id = ? LIMIT 1",
            (thought_id,),
        )
        return await cursor.fetchone() is not None

    async def _recreate_action_with_fk(self) -> None:
        """Recreate ``action`` with FK + CASCADE on ``source_thought_id``."""
        await self._db.execute(
            "CREATE TABLE action_new ("
            "  action_id           TEXT PRIMARY KEY,"
            "  source_thought_id   TEXT NOT NULL,"
            "  action_type         TEXT NOT NULL,"
            "  intent              TEXT NOT NULL,"
            "  status              TEXT NOT NULL DEFAULT 'PLANNED',"
            "  verification_status TEXT NOT NULL DEFAULT 'PENDING',"
            "  raw_metrics_json    TEXT,"
            "  FOREIGN KEY (source_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE"
            ")",
        )
        await self._db.execute("INSERT INTO action_new SELECT * FROM action")
        await self._db.execute("DROP TABLE action")
        await self._db.execute("ALTER TABLE action_new RENAME TO action")

    # ------------------------------------------------------------------
    # Embedding model immutability
    # ------------------------------------------------------------------

    async def _ensure_embedding_model_lock(self, model_name: str, dimension: int) -> None:
        """Lazy-lock the embedding model on first ``store_embedding()``.

        On first call (no ``embedding_model_name`` in ``_metadata``), writes
        the model name and dimension.  On subsequent calls, verifies the
        provider matches the stored values.

        Args:
            model_name: Model identifier from the current provider.
            dimension: Vector dimensionality from the current provider.

        Raises:
            EmbeddingModelMismatchError: When the configured model differs
                from the one stored in ``_metadata``.

        """
        if self._embedding_model_verified:
            return

        # Ensure _metadata table exists (idempotent).
        await self._migrate_core_v4_to_v5()

        cursor = await self._db.execute(
            "SELECT value FROM _metadata WHERE key = 'embedding_model_name'"
        )
        row = await cursor.fetchone()

        if row is None:
            # First embedding — lock the model.
            await self._db.execute(
                "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                ("embedding_model_name", model_name),
            )
            await self._db.execute(
                "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                ("embedding_dimension", str(dimension)),
            )
            await self._maybe_commit()
        else:
            stored_model = row["value"]
            dim_cursor = await self._db.execute(
                "SELECT value FROM _metadata WHERE key = 'embedding_dimension'"
            )
            dim_row = await dim_cursor.fetchone()
            stored_dimension = int(dim_row["value"]) if dim_row else 0

            if stored_model != model_name or stored_dimension != dimension:
                raise EmbeddingModelMismatchError(
                    stored_model=stored_model,
                    configured_model=model_name,
                    stored_dimension=stored_dimension,
                    configured_dimension=dimension,
                )

        self._embedding_model_verified = True

    async def verify_embedding_model(self) -> None:
        """Explicit eager check for embedding model compatibility.

        Callers that want fail-fast behaviour at startup can invoke this
        after construction.  When no ``embedding_provider`` is set, this
        is a no-op.

        Raises:
            EmbeddingModelMismatchError: When the configured model differs
                from the one stored in ``_metadata``.

        """
        if self._embedding_provider is None:
            return
        await self._ensure_embedding_model_lock(
            self._embedding_provider.model_name,
            self._embedding_provider.dimension,
        )

    # ------------------------------------------------------------------
    # Transaction control
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def suspend_auto_commit(self) -> AsyncIterator[None]:
        """Context manager that disables per-method auto-commit.

        Yields:
            None — the store operates in deferred-commit mode.

        """
        self._skip_auto_commit = True
        try:
            yield
        except Exception:
            await self._db.rollback()
            raise
        else:
            await self._db.commit()
        finally:
            self._skip_auto_commit = False

    async def _maybe_commit(self) -> None:
        """Commit if auto-commit is not suspended."""
        if not self._skip_auto_commit:
            await self._db.commit()

    # ------------------------------------------------------------------
    # Row -> Domain mappers (template methods — override in subclasses)
    # ------------------------------------------------------------------

    def _row_to_thought(self, row: aiosqlite.Row) -> ThoughtRecord:
        """Map a SQLite row to a ThoughtRecord.

        Override in subclasses to produce extended model types.

        Args:
            row: A row from the thought table.

        Returns:
            A ThoughtRecord domain model.

        """
        keys = row.keys()
        source_type_raw = row["source_type"] if "source_type" in keys else None
        confirmation_raw = row["confirmation_count"] if "confirmation_count" in keys else 0
        consolidated_raw = row["consolidated_from"] if "consolidated_from" in keys else None
        visibility_raw = row["visibility"] if "visibility" in keys else "selective"
        access_count_raw = row["access_count"] if "access_count" in keys else 0
        last_accessed_at_raw = row["last_accessed_at"] if "last_accessed_at" in keys else None
        created_at_raw = row["created_at"] if "created_at" in keys else None
        updated_at_raw = row["updated_at"] if "updated_at" in keys else None
        expires_at_raw = row["expires_at"] if "expires_at" in keys else None
        valid_from_raw = row["valid_from"] if "valid_from" in keys else None
        valid_until_raw = row["valid_until"] if "valid_until" in keys else None
        metadata_json_raw = row["metadata_json"] if "metadata_json" in keys else "{}"
        metadata_decoded: dict[str, MetadataValue] = (
            json.loads(metadata_json_raw) if metadata_json_raw else {}
        )
        return ThoughtRecord(
            thought_id=row["thought_id"],
            thought_type=ThoughtType(row["thought_type"]),
            essence=row["essence"],
            content=row["content"],
            priority=Priority(row["priority"]),
            lifecycle_status=LifecycleStatus(row["lifecycle_status"]),
            created_cycle=row["created_cycle"],
            updated_cycle=row["updated_cycle"],
            source=row["source"],
            confidence=row["confidence"],
            embedding_ref=row["embedding_ref"],
            source_type=(
                KnowledgeSource(source_type_raw) if source_type_raw else KnowledgeSource.EXPERIENCE
            ),
            confirmation_count=int(confirmation_raw) if confirmation_raw else 0,
            consolidated_from=_decode_consolidated(consolidated_raw),
            visibility=(
                ThoughtVisibility(visibility_raw) if visibility_raw else ThoughtVisibility.SELECTIVE
            ),
            access_count=int(access_count_raw) if access_count_raw else 0,
            last_accessed_at=last_accessed_at_raw,
            created_at=created_at_raw,
            updated_at=updated_at_raw,
            expires_at=expires_at_raw,
            valid_from=valid_from_raw,
            valid_until=valid_until_raw,
            metadata=metadata_decoded,
        )

    async def _get_thought_row(self, thought_id: str) -> aiosqlite.Row | None:
        """Fetch a raw thought row without applying retrieval hooks.

        Args:
            thought_id: UUID of the thought.

        Returns:
            Raw SQLite row, or ``None`` if not found.

        """
        cursor = await self._db.execute("SELECT * FROM thought WHERE thought_id = ?", (thought_id,))
        return await cursor.fetchone()

    async def _get_edge_row(self, edge_id: str) -> aiosqlite.Row | None:
        """Fetch a raw edge row without applying transformations.

        Args:
            edge_id: UUID of the edge.

        Returns:
            Raw SQLite row, or ``None`` if not found.

        """
        cursor = await self._db.execute("SELECT * FROM edge WHERE edge_id = ?", (edge_id,))
        return await cursor.fetchone()

    # ------------------------------------------------------------------
    # ThoughtRecord CRUD
    # ------------------------------------------------------------------

    _CORE_INSERT_SQL = (
        "INSERT INTO thought "
        "(thought_id, thought_type, essence, content, content_hash, priority, "
        " lifecycle_status, created_cycle, updated_cycle, source, "
        " confidence, embedding_ref, source_type, confirmation_count, "
        " consolidated_from, visibility, access_count, "
        " last_accessed_at, created_at, updated_at, expires_at, "
        " valid_from, valid_until, "
        " metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def _thought_to_core_params(self, thought: ThoughtRecord) -> tuple[object, ...]:
        """Extract core SQL parameters from a ThoughtRecord.

        Computes ``content_hash`` deterministically from
        ``thought.content`` (SHA-256 of the UTF-8 bytes, no normalization),
        so duplicate detection is always based on byte-exact content.

        Serializes ``thought.metadata`` with ``ensure_ascii=False`` so
        non-ASCII attribute values (speaker names, language strings,
        ...) survive a write/read round trip byte-exact.

        Args:
            thought: The thought record.

        Returns:
            Tuple of parameter values for ``_CORE_INSERT_SQL``.

        """
        return (
            thought.thought_id,
            thought.thought_type.value,
            thought.essence,
            thought.content,
            _compute_content_hash(thought.content),
            thought.priority.value,
            thought.lifecycle_status.value,
            thought.created_cycle,
            thought.updated_cycle,
            thought.source,
            thought.confidence,
            thought.embedding_ref,
            thought.source_type.value,
            thought.confirmation_count,
            _encode_consolidated(thought.consolidated_from),
            thought.visibility.value,
            thought.access_count,
            thought.last_accessed_at,
            thought.created_at,
            thought.updated_at,
            thought.expires_at,
            thought.valid_from,
            thought.valid_until,
            json.dumps(thought.metadata, ensure_ascii=False),
        )

    _CORE_UPDATE_SQL = (
        "UPDATE thought SET "
        " thought_type = ?, essence = ?, content = ?, priority = ?,"
        " lifecycle_status = ?, created_cycle = ?, updated_cycle = ?,"
        " source = ?, confidence = ?, embedding_ref = ?,"
        " source_type = ?, confirmation_count = ?,"
        " consolidated_from = ?, visibility = ?,"
        " access_count = ?, last_accessed_at = ?,"
        " created_at = ?, updated_at = ?, expires_at = ?,"
        " valid_from = ?, valid_until = ?,"
        " metadata_json = ? "
        "WHERE thought_id = ? AND updated_cycle = ?"
    )

    def _thought_to_core_update_params(
        self,
        updated: ThoughtRecord,
        thought_id: str,
        expected_cycle: int,
    ) -> tuple[object, ...]:
        """Extract core SQL parameters for UPDATE from a ThoughtRecord.

        Mirrors :py:meth:`_thought_to_core_params` for the SET column
        ordering, including the trailing ``metadata_json`` value, so a
        round trip through ``update_thought`` preserves caller-supplied
        metadata instead of silently reverting it to the column default.

        Args:
            updated: The updated thought record.
            thought_id: UUID of the thought to update.
            expected_cycle: The OCC version guard.

        Returns:
            Tuple of parameter values for ``_CORE_UPDATE_SQL`` — SET
            columns first (in declaration order), then the WHERE
            ``thought_id`` and ``expected_cycle`` guard.

        """
        return (
            updated.thought_type.value,
            updated.essence,
            updated.content,
            updated.priority.value,
            updated.lifecycle_status.value,
            updated.created_cycle,
            updated.updated_cycle,
            updated.source,
            updated.confidence,
            updated.embedding_ref,
            updated.source_type.value,
            updated.confirmation_count,
            _encode_consolidated(updated.consolidated_from),
            updated.visibility.value,
            updated.access_count,
            updated.last_accessed_at,
            updated.created_at,
            updated.updated_at,
            updated.expires_at,
            updated.valid_from,
            updated.valid_until,
            json.dumps(updated.metadata, ensure_ascii=False),
            thought_id,
            expected_cycle,
        )

    async def _get_thought_by_content_hash(
        self,
        content_hash: str,
    ) -> ThoughtRecord | None:
        """Return the first thought whose ``content_hash`` matches.

        Used by opt-in ingest deduplication to detect existing rows
        with identical content.  Pre-core-10 thoughts whose hash has
        not been backfilled have ``content_hash IS NULL`` and are
        therefore not eligible for deduplication.

        Args:
            content_hash: Lowercase hex SHA-256 digest of the candidate
                thought's content.

        Returns:
            The matching ``ThoughtRecord`` or ``None`` if no row
            matches.  When more than one row shares the hash (only
            possible for older data ingested before this fix
            was deployed) the first match in B-tree order is returned.

        """
        cursor = await self._db.execute(
            "SELECT * FROM thought WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_thought(row)

    async def _increment_confirmation(
        self,
        existing: ThoughtRecord,
    ) -> ThoughtRecord:
        """Bump ``confirmation_count`` + ``updated_at`` for an existing thought.

        Implements the dedup-hit branch of ``create_thought``.  Persists
        the bump in SQLite, refreshes ``updated_at`` to the current UTC
        time, and returns a rebuilt frozen ``ThoughtRecord`` reflecting
        the new state (Pydantic ``model_copy`` because the model is
        ``frozen=True`` and forbids in-place mutation).

        Args:
            existing: The thought already in the database.

        Returns:
            A new ``ThoughtRecord`` instance with
            ``confirmation_count = existing.confirmation_count + 1``
            and a refreshed ``updated_at``.

        """
        new_count = existing.confirmation_count + 1
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()

        await self._db.execute(
            "UPDATE thought SET confirmation_count = ?, updated_at = ? WHERE thought_id = ?",
            (new_count, now_iso, existing.thought_id),
        )
        await self._maybe_commit()

        if self._journal is not None:
            after = existing.model_copy(
                update={
                    "confirmation_count": new_count,
                    "updated_at": now_iso,
                },
            )
            await self._journal.append(
                mutation_type="UPDATE_THOUGHT",
                target_id=existing.thought_id,
                delta={
                    "before": existing.model_dump(mode="json"),
                    "after": after.model_dump(mode="json"),
                },
            )
            return after

        return existing.model_copy(
            update={
                "confirmation_count": new_count,
                "updated_at": now_iso,
            },
        )

    async def _create_thought_with_dedup(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None,
    ) -> ThoughtRecord:
        """Lock-protected dedup branch of ``create_thought``.

        Acquires ``self._dedup_lock`` for the entire ``check existing
        → INSERT or UPDATE`` window so concurrent calls with identical
        ``content`` never race past the existence probe.  When the
        content has not been seen the call delegates back to
        ``create_thought(..., deduplicate=False)`` so the regular
        insert / journal / auto-embed pipeline runs unchanged; when it
        has been seen ``confirmation_count`` is bumped and the existing
        record returned without any additional INSERT.
        """
        async with self._dedup_lock:
            existing = await self._get_thought_by_content_hash(
                _compute_content_hash(thought.content),
            )
            if existing is not None:
                return await self._increment_confirmation(existing)
            return await self.create_thought(
                thought,
                expires_after_seconds=expires_after_seconds,
                deduplicate=False,
            )

    async def create_thought(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
        deduplicate: bool = False,
    ) -> ThoughtRecord:
        """Persist a new thought record.

        Automatically sets ``created_at`` and ``updated_at`` to the
        current UTC time when they are ``None``.  When
        ``expires_after_seconds`` is given (or a default TTL is
        configured), ``expires_at`` is computed as an absolute ISO-8601
        timestamp.

        Content-hash deduplication is opt-in via ``deduplicate=True``:
        when an existing thought with the same SHA-256 hash of
        ``content`` is found, its ``confirmation_count`` is incremented
        and the existing record is returned instead of creating a
        duplicate.  The default ``deduplicate=False`` preserves the
        legacy create-on-every-call behaviour.

        Args:
            thought: The thought record to create.
            expires_after_seconds: Optional relative TTL in seconds.
                Overrides the store-level default when provided.
            deduplicate: When True, check for an existing thought with
                identical ``content`` (via the indexed SHA-256
                ``content_hash`` column).  If a match is found, that
                thought's ``confirmation_count`` is incremented, its
                ``updated_at`` is refreshed, and the updated record is
                returned without inserting a new row.  When False
                (default) a new thought is always inserted, exactly
                matching the pre-deduplication behaviour.

        Returns:
            The persisted thought record (with timestamps populated).
            When ``deduplicate=True`` collides with an existing thought,
            returns the existing record with bumped
            ``confirmation_count`` and ``updated_at``.

        Raises:
            ValueError: If a thought with the same ID already exists, or
                if ``thought.metadata`` violates the metadata-shape or
                size invariants enforced by :func:`_validate_metadata`.

        """
        _validate_metadata(thought.metadata)

        if deduplicate:
            return await self._create_thought_with_dedup(
                thought,
                expires_after_seconds=expires_after_seconds,
            )

        existing_row = await self._get_thought_row(thought.thought_id)
        if existing_row is not None:
            msg = f"Thought already exists: {thought.thought_id}"
            raise ValueError(msg)

        now = datetime.datetime.now(datetime.UTC)
        now_iso = now.isoformat()
        updates: dict[str, object] = {}
        if thought.created_at is None:
            updates["created_at"] = now_iso
        if thought.updated_at is None:
            updates["updated_at"] = now_iso

        # Resolve expiry: explicit param > thought field > store default.
        if expires_after_seconds is not None:
            updates["expires_at"] = (
                now + datetime.timedelta(seconds=expires_after_seconds)
            ).isoformat()
        elif thought.expires_at is None and self._ttl_default_seconds is not None:
            updates["expires_at"] = (
                now + datetime.timedelta(seconds=self._ttl_default_seconds)
            ).isoformat()

        if updates:
            thought = type(thought).model_validate(
                {**thought.model_dump(), **updates},
            )

        await self._db.execute(self._CORE_INSERT_SQL, self._thought_to_core_params(thought))

        if self._journal is not None:
            await self._journal.append(
                mutation_type="INSERT_THOUGHT",
                target_id=thought.thought_id,
                delta={"before": None, "after": thought.model_dump(mode="json")},
            )

        await self._maybe_commit()

        # Auto-embed when a provider is configured and auto_embed is on.
        if self._auto_embed and self._embedding_provider is not None:
            await self._auto_embed_thought(thought)

        await self._maybe_auto_cleanup(exclude_id=thought.thought_id)
        return await self._hooks.on_store(thought)

    async def cleanup_expired(
        self,
        now: str | None = None,
        *,
        exclude_id: str | None = None,
    ) -> CleanupResult:
        """Remove or archive thoughts whose ``expires_at`` is in the past.

        The strategy used (``archive`` or ``delete``) is determined by
        the store's ``ttl_strategy`` setting.

        * **archive**: Sets ``lifecycle_status`` to ``ARCHIVED`` and clears
          ``expires_at`` so the thought is no longer subject to TTL.
        * **delete**: Physically deletes the expired thought rows (cascading
          to edges, embeddings, and actions via ON DELETE CASCADE).

        Mutations are recorded in the journal when journaling is enabled.

        Args:
            now: Optional ISO-8601 UTC timestamp to use as "current time".
                Defaults to ``datetime.now(UTC).isoformat()`` when omitted.
                Useful for deterministic testing.
            exclude_id: Optional thought ID to skip during cleanup.
                Used by auto-cleanup to protect a just-written thought.

        Returns:
            A ``CleanupResult`` with the count of processed thoughts, the
            strategy that was applied, and a UTC timestamp.

        """
        if now is None:
            now = datetime.datetime.now(datetime.UTC).isoformat()

        cursor = await self._db.execute(
            "SELECT thought_id FROM thought WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        expired_rows = await cursor.fetchall()
        expired_ids = [row["thought_id"] for row in expired_rows if row["thought_id"] != exclude_id]

        strategy = CleanupStrategy(self._ttl_strategy)

        for tid in expired_ids:
            if strategy is CleanupStrategy.ARCHIVE:
                before_row = await self._get_thought_row(tid) if self._journal is not None else None
                await self._db.execute(
                    "UPDATE thought SET lifecycle_status = ?, expires_at = NULL "
                    "WHERE thought_id = ?",
                    (LifecycleStatus.ARCHIVED.value, tid),
                )
                if self._journal is not None and before_row is not None:
                    before = self._row_to_thought(before_row)
                    after = before.evolve(
                        lifecycle_status=LifecycleStatus.ARCHIVED.value,
                        expires_at=None,
                    )
                    await self._journal.append(
                        mutation_type="UPDATE_THOUGHT",
                        target_id=tid,
                        delta={
                            "before": before.model_dump(mode="json"),
                            "after": after.model_dump(mode="json"),
                        },
                    )
            else:
                # DELETE strategy.
                before_row = await self._get_thought_row(tid) if self._journal is not None else None
                await self._db.execute(
                    "DELETE FROM thought WHERE thought_id = ?",
                    (tid,),
                )
                if self._journal is not None and before_row is not None:
                    await self._journal.append(
                        mutation_type="DELETE_THOUGHT",
                        target_id=tid,
                        delta={
                            "before": self._row_to_thought(before_row).model_dump(
                                mode="json",
                            ),
                            "after": None,
                        },
                    )

        if expired_ids:
            await self._maybe_commit()

        return CleanupResult(
            expired_count=len(expired_ids),
            strategy_applied=strategy.value,
            timestamp=now,
        )

    async def _maybe_auto_cleanup(self, *, exclude_id: str | None = None) -> None:
        """Run auto-cleanup of expired thoughts if cadence threshold is met.

        Args:
            exclude_id: Optional thought ID to exclude from cleanup.
                Prevents archiving/deleting a thought that was just
                created or updated in the current operation.

        """
        if self._ttl_check_every_n < 1:
            return
        self._operation_count += 1
        if self._operation_count >= self._ttl_check_every_n:
            self._operation_count = 0
            await self.cleanup_expired(exclude_id=exclude_id)

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Retrieve a thought by its ID, or None if not found.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The thought record, or None if not found.

        """
        row = await self._get_thought_row(thought_id)
        if row is None:
            return None
        return await self._hooks.on_retrieve(self._row_to_thought(row))

    async def update_thought(self, thought_id: str, **changes: object) -> ThoughtRecord:
        """Update a thought with optimistic concurrency.

        Uses ``updated_cycle`` as a version guard.

        Args:
            thought_id: UUID of the thought to update.
            **changes: Fields to update.

        Returns:
            The updated thought record.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.
            StaleDataError: If the row was modified since it was read.
            ValueError: If the post-``evolve`` metadata violates the
                metadata-shape or size invariants enforced by
                :func:`_validate_metadata`.

        """
        current_row = await self._get_thought_row(thought_id)
        if current_row is None:
            raise ThoughtNotFoundError(thought_id)

        current = self._row_to_thought(current_row)

        expected_cycle = current.updated_cycle
        updated = current.evolve(**changes)

        _validate_metadata(updated.metadata)

        cursor = await self._db.execute(
            self._CORE_UPDATE_SQL,
            self._thought_to_core_update_params(updated, thought_id, expected_cycle),
        )
        if cursor.rowcount == 0:
            raise StaleDataError(
                entity_type="ThoughtRecord",
                entity_id=thought_id,
                expected_version=expected_cycle,
            )

        if self._journal is not None:
            await self._journal.append(
                mutation_type="UPDATE_THOUGHT",
                target_id=thought_id,
                delta={
                    "before": current.model_dump(mode="json"),
                    "after": updated.model_dump(mode="json"),
                },
            )

        await self._maybe_commit()

        # Re-embed when essence or content changed.
        if (
            self._auto_embed
            and self._embedding_provider is not None
            and (updated.essence != current.essence or updated.content != current.content)
        ):
            await self._auto_embed_thought(updated)
            # The member's vector moved, so any REFLECTION that summarizes it
            # must re-bind to the current cluster instead of scoring on a
            # frozen centroid. Strictly on the essence/content path — a
            # metadata-only edit never reaches here, so it cannot re-bind.
            await self._rebind_consolidated_reflections(thought_id)

        await self._maybe_auto_cleanup(exclude_id=thought_id)
        return updated

    async def invalidate_thought(
        self,
        thought_id: str,
        valid_until: str,
    ) -> ThoughtRecord:
        """Close a thought's valid-time interval at the given instant.

        Sets the thought's ``valid_until`` to ``valid_until``, marking the
        end of the window during which the fact is considered true in the
        world. This is a deterministic, valid-time-only operation:

        * It is **not** a delete — the row and all of its history remain
          stored and retrievable; only the valid-time upper bound changes.
        * It performs **no** similarity search, automatic invalidation, or
          model inference of any kind.
        * It does **not** cascade to the thought's edges — invalidating a
          thought leaves every connected edge's valid-time interval
          untouched.
        * It is **idempotent**: invalidating with the same ``valid_until``
          twice converges to the same stored value.

        Args:
            thought_id: UUID of the thought to invalidate.
            valid_until: ISO-8601 instant at which the fact stops being
                valid. Stored as the thought's ``valid_until`` bound.

        Returns:
            The updated thought record.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.
            StaleDataError: If the row was modified since it was read.
            ValueError: If ``valid_until`` is not a valid ISO-8601 timestamp.

        """
        normalized = validate_iso8601_nullable(valid_until)
        return await self.update_thought(thought_id, valid_until=normalized)

    async def list_thoughts(
        self,
        *,
        priority: str | None = None,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        min_cycle: int | None = None,
        max_cycle: int | None = None,
        visibility: str | None = None,
        exclude_visibility: str | None = None,
        include_expired: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThoughtRecord]:
        """List thoughts matching the given filters.

        Args:
            priority: Filter by priority level.
            lifecycle_status: Filter by lifecycle status.
            thought_type: Filter by thought type.
            min_cycle: Minimum updated_cycle (inclusive).
            max_cycle: Maximum updated_cycle (inclusive).
            visibility: Include only thoughts with this visibility.
            exclude_visibility: Exclude thoughts with this visibility.
            include_expired: If True, include expired thoughts. Defaults to False.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of matching thought records.

        """
        clauses: list[str] = []
        params: list[object] = []

        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.datetime.now(datetime.UTC).isoformat())

        if priority is not None:
            clauses.append("priority = ?")
            params.append(priority)
        if lifecycle_status is not None:
            clauses.append("lifecycle_status = ?")
            params.append(lifecycle_status)
        if thought_type is not None:
            clauses.append("thought_type = ?")
            params.append(thought_type)
        if min_cycle is not None:
            clauses.append("updated_cycle >= ?")
            params.append(min_cycle)
        if max_cycle is not None:
            clauses.append("updated_cycle <= ?")
            params.append(max_cycle)
        if visibility is not None:
            clauses.append("visibility = ?")
            params.append(visibility)
        if exclude_visibility is not None:
            clauses.append("visibility != ?")
            params.append(exclude_visibility)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM thought{where} ORDER BY updated_cycle DESC LIMIT ? OFFSET ?"  # noqa: S608
        params.extend([limit, offset])

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        thoughts = [self._row_to_thought(r) for r in rows]
        return [await self._hooks.on_retrieve(t) for t in thoughts]

    async def count_thoughts(
        self,
        *,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        priority: str | None = None,
        include_expired: bool = False,
    ) -> int:
        """Count thoughts matching the given filters.

        A lightweight alternative to ``list_thoughts`` when only the
        total count is needed (e.g. for the early-stop clustering
        guard in ``DreamingExtension``).

        Args:
            lifecycle_status: Filter by lifecycle status.
            thought_type: Filter by thought type.
            priority: Filter by priority level (e.g. ``"P1"``).
            include_expired: If True, include expired thoughts. Defaults to False.

        Returns:
            Number of thoughts matching the filters.

        """
        clauses: list[str] = []
        params: list[object] = []

        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.datetime.now(datetime.UTC).isoformat())

        if lifecycle_status is not None:
            clauses.append("lifecycle_status = ?")
            params.append(lifecycle_status)
        if thought_type is not None:
            clauses.append("thought_type = ?")
            params.append(thought_type)
        if priority is not None:
            clauses.append("priority = ?")
            params.append(priority)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT COUNT(*) FROM thought{where}"  # noqa: S608

        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought by its ID.

        Args:
            thought_id: UUID of the thought to delete.

        Returns:
            True if the thought was deleted, False if not found.

        """
        before_row = await self._get_thought_row(thought_id) if self._journal is not None else None

        cursor = await self._db.execute("DELETE FROM thought WHERE thought_id = ?", (thought_id,))
        deleted = cursor.rowcount > 0

        if deleted and self._journal is not None and before_row is not None:
            await self._journal.append(
                mutation_type="DELETE_THOUGHT",
                target_id=thought_id,
                delta={
                    "before": self._row_to_thought(before_row).model_dump(mode="json"),
                    "after": None,
                },
            )

        await self._maybe_commit()
        return deleted

    # ------------------------------------------------------------------
    # EdgeRecord CRUD
    # ------------------------------------------------------------------

    async def create_edge(self, edge: EdgeRecord) -> EdgeRecord:
        """Persist a new edge record.

        The schema (core-12+) enforces an ``ON DELETE CASCADE`` foreign
        key on both edge endpoints. Inserting an edge whose endpoints
        do not resolve to existing thoughts raises
        :class:`ReferentialIntegrityError` — the raw
        ``sqlite3.IntegrityError`` is intentionally not surfaced.

        Args:
            edge: The edge record to create.

        Returns:
            The persisted edge record.

        Raises:
            ReferentialIntegrityError: When ``from_thought_id`` or
                ``to_thought_id`` does not match any persisted thought.

        """
        try:
            await self._db.execute(
                "INSERT INTO edge "
                "(edge_id, from_thought_id, to_thought_id, edge_type, weight, "
                " created_cycle, source, decay_multiplier, valid_from, valid_until) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edge.edge_id,
                    edge.from_thought_id,
                    edge.to_thought_id,
                    edge.edge_type.value,
                    edge.weight,
                    edge.created_cycle,
                    edge.source.value,
                    edge.decay_multiplier,
                    edge.valid_from,
                    edge.valid_until,
                ),
            )
        except aiosqlite.IntegrityError as exc:
            if "FOREIGN KEY" not in str(exc).upper():
                # Surface UNIQUE / NOT NULL violations unchanged — only
                # FK violations get the domain wrapper.
                raise
            column, referenced = await self._identify_orphan_endpoint(edge)
            raise ReferentialIntegrityError(
                entity_type="edge",
                column=column,
                referenced_id=referenced,
            ) from exc

        if self._journal is not None:
            await self._journal.append(
                mutation_type="INSERT_EDGE",
                target_id=edge.edge_id,
                delta={"before": None, "after": edge.model_dump(mode="json")},
            )

        await self._maybe_commit()
        return edge

    async def update_edge(self, edge_id: str, **changes: object) -> EdgeRecord:
        """Update an edge by its ID.

        Args:
            edge_id: UUID of the edge to update.
            **changes: Fields to update.

        Returns:
            The updated edge record.

        Raises:
            ValueError: If the edge does not exist.

        """
        current_row = await self._get_edge_row(edge_id)
        if current_row is None:
            msg = f"Edge not found: {edge_id}"
            raise ValueError(msg)

        current = _row_to_edge(current_row)
        updated = type(current).model_validate({**current.model_dump(mode="json"), **changes})

        await self._db.execute(
            "UPDATE edge SET from_thought_id = ?, to_thought_id = ?, edge_type = ?, "
            "weight = ?, created_cycle = ?, source = ?, decay_multiplier = ?, "
            "valid_from = ?, valid_until = ? "
            "WHERE edge_id = ?",
            (
                updated.from_thought_id,
                updated.to_thought_id,
                updated.edge_type.value,
                updated.weight,
                updated.created_cycle,
                updated.source.value,
                updated.decay_multiplier,
                updated.valid_from,
                updated.valid_until,
                edge_id,
            ),
        )

        if self._journal is not None:
            await self._journal.append(
                mutation_type="UPDATE_EDGE",
                target_id=edge_id,
                delta={
                    "before": current.model_dump(mode="json"),
                    "after": updated.model_dump(mode="json"),
                },
            )

        await self._maybe_commit()
        return updated

    async def invalidate_edge(
        self,
        edge_id: str,
        valid_until: str,
    ) -> EdgeRecord:
        """Close an edge's valid-time interval at the given instant.

        Sets the edge's ``valid_until`` to ``valid_until``, marking the end
        of the window during which the relation is considered true in the
        world. Like :meth:`invalidate_thought`, this is a deterministic,
        valid-time-only operation:

        * It is **not** a delete — the edge row remains stored and
          retrievable; only the valid-time upper bound changes.
        * It performs **no** similarity search, automatic invalidation, or
          model inference of any kind.
        * It is **idempotent**: invalidating with the same ``valid_until``
          twice converges to the same stored value.

        Args:
            edge_id: UUID of the edge to invalidate.
            valid_until: ISO-8601 instant at which the relation stops being
                valid. Stored as the edge's ``valid_until`` bound.

        Returns:
            The updated edge record.

        Raises:
            ValueError: If the edge does not exist, or ``valid_until`` is not
                a valid ISO-8601 timestamp.

        """
        normalized = validate_iso8601_nullable(valid_until)
        return await self.update_edge(edge_id, valid_until=normalized)

    async def get_edges(
        self,
        thought_id: str,
        *,
        direction: str = "BOTH",
    ) -> list[EdgeRecord]:
        """Retrieve edges connected to a thought.

        Args:
            thought_id: UUID of the thought.
            direction: 'IN', 'OUT', or 'BOTH'.

        Returns:
            List of matching edge records.

        """
        if direction == "OUT":
            sql = "SELECT * FROM edge WHERE from_thought_id = ?"
            params: tuple[str, ...] = (thought_id,)
        elif direction == "IN":
            sql = "SELECT * FROM edge WHERE to_thought_id = ?"
            params = (thought_id,)
        else:
            sql = "SELECT * FROM edge WHERE from_thought_id = ? OR to_thought_id = ?"
            params = (thought_id, thought_id)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by its ID.

        Args:
            edge_id: UUID of the edge to delete.

        Returns:
            True if the edge was deleted, False if not found.

        """
        before_row = await self._get_edge_row(edge_id) if self._journal is not None else None

        cursor = await self._db.execute("DELETE FROM edge WHERE edge_id = ?", (edge_id,))
        deleted = cursor.rowcount > 0

        if deleted and self._journal is not None and before_row is not None:
            await self._journal.append(
                mutation_type="DELETE_EDGE",
                target_id=edge_id,
                delta={
                    "before": dict(before_row),
                    "after": None,
                },
            )

        await self._maybe_commit()
        return deleted

    async def list_edges(
        self,
        *,
        edge_type: EdgeType | None = None,
        source: KnowledgeSource | None = None,
        limit: int = 5000,
    ) -> list[EdgeRecord]:
        """List edges matching optional filters.

        Args:
            edge_type: If given, restrict to this edge type.
            source: If given, restrict to this knowledge source.
            limit: Maximum number of edges to return.

        Returns:
            List of matching edge records, ordered by ``created_cycle`` DESC.

        """
        clauses: list[str] = []
        params: list[object] = []

        if edge_type is not None:
            clauses.append("edge_type = ?")
            params.append(str(edge_type))
        if source is not None:
            clauses.append("source = ?")
            params.append(str(source))

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM edge{where} ORDER BY created_cycle DESC LIMIT ?"  # noqa: S608
        params.append(limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def thought_exists_by_source(
        self,
        *,
        source: str,
        thought_type_value: str,
    ) -> bool:
        """Check whether any thought with an exact source and thought_type exists.

        Performs a single O(1) index lookup — safe to call per cluster in
        ``_create_reflections`` regardless of store size.

        Args:
            source: Exact ``source`` field value to match.
            thought_type_value: ``thought_type`` enum string (e.g.
                ``"REFLECTION"``).

        Returns:
            ``True`` if at least one matching thought exists.

        Examples:
            >>> exists = await store.thought_exists_by_source(
            ...     source="dreaming:abc123",
            ...     thought_type_value="REFLECTION",
            ... )  # doctest: +SKIP

        """
        cursor = await self._db.execute(
            "SELECT thought_id FROM thought WHERE thought_type = ? AND source = ? LIMIT 1",
            (thought_type_value, source),
        )
        return await cursor.fetchone() is not None

    async def consolidated_source_statuses(self, reflection_id: str) -> list[str]:
        """Return the lifecycle statuses of a REFLECTION's source thoughts.

        Resolves the ``CONSOLIDATED_FROM`` edges leaving ``reflection_id``
        and returns the ``lifecycle_status`` of each source thought, using a
        single indexed join rather than a per-edge lookup. The result is the
        liveness picture an orphan sweep needs: a REFLECTION is orphaned when
        this list is non-empty and contains no ``ACTIVE`` entry.

        Args:
            reflection_id: UUID of the REFLECTION whose sources to inspect.

        Returns:
            One lifecycle-status string per resolvable source thought, in no
            particular order. Empty when the REFLECTION has no
            ``CONSOLIDATED_FROM`` edges (or none resolve to a live thought
            row).

        """
        cursor = await self._db.execute(
            "SELECT t.lifecycle_status AS lifecycle_status "
            "FROM edge e "
            "JOIN thought t ON e.to_thought_id = t.thought_id "
            "WHERE e.from_thought_id = ? AND e.edge_type = 'CONSOLIDATED_FROM'",
            (reflection_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["lifecycle_status"]) for row in rows]

    async def reflections_consolidated_from(self, source_id: str) -> list[str]:
        """Return REFLECTION ids that were consolidated from a source thought.

        Resolves the inbound ``CONSOLIDATED_FROM`` edges of ``source_id`` and
        keeps only the parents whose ``thought_type`` is ``REFLECTION``. Used
        by the re-bind path to find the syntheses that must be refreshed when
        a member's embedding changes.

        Args:
            source_id: UUID of the source thought.

        Returns:
            Distinct REFLECTION ids that consolidated ``source_id`` as a
            member. Empty when the source belongs to no REFLECTION.

        """
        cursor = await self._db.execute(
            "SELECT DISTINCT t.thought_id AS thought_id "
            "FROM edge e "
            "JOIN thought t ON e.from_thought_id = t.thought_id "
            "WHERE e.to_thought_id = ? AND e.edge_type = 'CONSOLIDATED_FROM' "
            "AND t.thought_type = 'REFLECTION'",
            (source_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["thought_id"]) for row in rows]

    async def consolidated_member_ids(self, reflection_id: str) -> list[str]:
        """Return the source-thought ids a REFLECTION was consolidated from.

        Args:
            reflection_id: UUID of the REFLECTION.

        Returns:
            The ``to_thought_id`` of each ``CONSOLIDATED_FROM`` edge leaving
            ``reflection_id``. Empty when the REFLECTION has no such edges.

        """
        cursor = await self._db.execute(
            "SELECT to_thought_id FROM edge "
            "WHERE from_thought_id = ? AND edge_type = 'CONSOLIDATED_FROM'",
            (reflection_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["to_thought_id"]) for row in rows]

    # ------------------------------------------------------------------
    # Auto-embed helper
    # ------------------------------------------------------------------

    async def _auto_embed_thought(self, thought: ThoughtRecord) -> None:
        """Generate and store an embedding for a thought via the provider.

        Combines ``essence`` and ``content`` into a single text payload,
        embeds it via the configured provider, and persists the vector.

        Args:
            thought: The thought to embed.

        """
        provider = self._embedding_provider
        if provider is None:
            return  # pragma: no cover
        text = f"{thought.essence}\n{thought.content}"

        vector = await provider.embed(text)

        await self.store_embedding(
            thought.thought_id,
            vector,
            model_name=provider.model_name,
        )

    async def _rebind_consolidated_reflections(self, source_id: str) -> int:
        """Recompute the centroids of REFLECTIONs that summarize a source.

        Called after a source thought is re-embedded (essence/content
        evolve). A REFLECTION is a synthesis bound to the live state of its
        cluster, so when a member's vector changes the REFLECTION's centroid
        must be recomputed from the current member vectors rather than stay
        frozen at its creation-time value. The recompute reuses the same
        deterministic L2-normalized mean as REFLECTION creation
        (:func:`compute_centroid`) and overwrites the centroid in place via
        the ``store_embedding`` upsert — no schema change, no model call.

        This is intentionally *not* invoked on metadata-only edits: it is
        called only from the essence/content re-embed branch of
        :meth:`update_thought`, so metadata/priority churn leaves dependent
        REFLECTION centroids untouched.

        Args:
            source_id: UUID of the source thought that was just re-embedded.

        Returns:
            Number of REFLECTION centroids recomputed.

        """
        reflection_ids = await self.reflections_consolidated_from(source_id)
        if not reflection_ids:
            return 0

        rebound = 0
        for reflection_id in reflection_ids:
            member_ids = await self.consolidated_member_ids(reflection_id)
            member_vectors: list[list[float]] = []
            for member_id in member_ids:
                embedding = await self.get_embedding(member_id)
                if embedding is None:
                    continue
                member_vectors.append(
                    list(struct.unpack(f"{embedding.dimension}f", embedding.vector_blob)),
                )
            if not member_vectors:
                continue
            centroid = compute_centroid(member_vectors)
            await self.store_embedding(
                reflection_id,
                centroid,
                model_name=CENTROID_MODEL_NAME,
            )
            rebound += 1
        return rebound

    # ------------------------------------------------------------------
    # EmbeddingRecord CRUD
    # ------------------------------------------------------------------

    async def store_embedding(
        self,
        thought_id: str,
        vector: list[float],
        *,
        model_name: str = "all-MiniLM-L12-v2",
        embedding_id: str | None = None,
    ) -> EmbeddingRecord:
        """Persist an embedding vector for a thought.

        On first call, locks the embedding model in ``_metadata``.
        Subsequent calls verify the model matches the stored one.

        Args:
            thought_id: UUID of the thought that owns this embedding.
            vector: Embedding vector as a list of floats.
            model_name: Embedding model identifier.
            embedding_id: Optional explicit ID; generated if omitted.

        Returns:
            The persisted EmbeddingRecord.

        Raises:
            EmbeddingModelMismatchError: When model_name does not match
                the model already stored in ``_metadata``.

        """
        await self._ensure_embedding_model_lock(model_name, len(vector))
        eid = embedding_id or f"emb-{_uuid.uuid5(_uuid.NAMESPACE_URL, thought_id)}"
        dimension = len(vector)
        blob = struct.pack(f"{dimension}f", *vector)
        created_at = datetime.datetime.now(datetime.UTC).isoformat()

        cursor = await self._db.execute(
            "SELECT rowid FROM embedding WHERE embedding_id = ?",
            (eid,),
        )
        existing_row = await cursor.fetchone()

        rowid: int
        if existing_row is None:
            await self._db.execute(
                "INSERT INTO embedding "
                "(embedding_id, owner_type, owner_id, model_name, "
                "dimension, vector_blob, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (eid, "THOUGHT", thought_id, model_name, dimension, blob, created_at),
            )
            cursor = await self._db.execute(
                "SELECT rowid FROM embedding WHERE embedding_id = ?",
                (eid,),
            )
            inserted_row = await cursor.fetchone()
            if inserted_row is None:
                msg = f"Embedding row missing after insert: {eid}"
                raise RuntimeError(msg)
            rowid = int(inserted_row["rowid"])
        else:
            rowid = int(existing_row["rowid"])
            await self._db.execute(
                "UPDATE embedding SET "
                "owner_type = ?, owner_id = ?, model_name = ?, dimension = ?, "
                "vector_blob = ?, created_at = ? "
                "WHERE embedding_id = ?",
                ("THOUGHT", thought_id, model_name, dimension, blob, created_at, eid),
            )

        # Keep the vec0 vector table in sync when a vector backend is active.
        if self._vector_backend is not None:
            await self._vector_backend.upsert_embedding(
                self._db,
                rowid=rowid,
                vector=vector,
            )

        await self._maybe_commit()
        return EmbeddingRecord(
            embedding_id=eid,
            owner_type="THOUGHT",
            owner_id=thought_id,
            model_name=model_name,
            dimension=dimension,
            vector_blob=blob,
            created_at=created_at,
        )

    async def get_embedding(self, thought_id: str) -> EmbeddingRecord | None:
        """Retrieve the embedding for a thought, or None if not found.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The EmbeddingRecord, or None if not found.

        """
        cursor = await self._db.execute(
            "SELECT * FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
            (thought_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_embedding(row)

    # ------------------------------------------------------------------
    # Embedding similarity search (brute-force cosine)
    # ------------------------------------------------------------------

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Cosine similarity search — delegates to sqlite-vec if available.

        When a ``SqliteVecSearchBackend`` is configured (via
        ``from_config`` with ``vector_backend: "sqlite-vec"``), the
        ``vec0`` vector table serves the query.  Otherwise falls back to
        brute-force numpy cosine similarity.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity score.

        Returns:
            List of ``(thought_id, similarity_score)`` sorted descending.

        """
        import time as _time  # noqa: PLC0415

        _t_start = _time.perf_counter()
        if self._vector_backend is not None:
            results = await self._vector_backend.search(
                self._db,
                query_vector,
                top_k,
                threshold,
            )
            filtered = await self._filter_expired_results(results)
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return filtered
        results = await self._search_similar_numpy(query_vector, top_k, threshold)
        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
        return results

    async def _filter_expired_results(
        self,
        results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Remove expired thoughts and retired REFLECTIONs from results.

        Used as a post-filter for search backends (e.g. sqlite-vec)
        that cannot natively exclude expired rows in their queries. It also
        applies the REFLECTION freshness floor: a retired REFLECTION (an
        orphan archived once its cluster left the active set) must not
        over-recall on its now-stale centroid, so any REFLECTION whose
        ``lifecycle_status`` is not ``ACTIVE`` is dropped. Other thought
        types are gated on expiry only, exactly as before.

        Args:
            results: List of ``(thought_id, similarity_score)`` pairs.

        Returns:
            Filtered list with expired thoughts and retired REFLECTIONs
            removed.

        """
        if not results:
            return results
        now = datetime.datetime.now(datetime.UTC).isoformat()
        ids = [r[0] for r in results]
        placeholders = ",".join("?" * len(ids))
        cursor = await self._db.execute(
            f"SELECT thought_id FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders}) "
            f"AND ((expires_at IS NOT NULL AND expires_at <= ?) "
            f"OR (thought_type = 'REFLECTION' AND lifecycle_status != 'ACTIVE'))",
            [*ids, now],
        )
        excluded_ids = {row["thought_id"] for row in await cursor.fetchall()}
        if not excluded_ids:
            return results
        return [(tid, score) for tid, score in results if tid not in excluded_ids]

    async def _search_similar_numpy(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Brute-force cosine similarity search (numpy-batched).

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity score.

        Returns:
            List of ``(thought_id, similarity_score)`` sorted descending.

        """
        cursor = await self._db.execute(
            "SELECT e.owner_id, e.dimension, e.vector_blob "
            "FROM embedding e "
            "JOIN thought t ON e.owner_id = t.thought_id "
            "WHERE e.owner_type = 'THOUGHT' "
            "AND (t.expires_at IS NULL OR t.expires_at > ?) "
            # Freshness floor: a retired REFLECTION (an orphan archived once
            # its cluster left the active set) must not over-recall on its
            # now-stale centroid. Only REFLECTIONs are gated on lifecycle
            # here; other thought types keep their existing recall behaviour.
            "AND NOT (t.thought_type = 'REFLECTION' AND t.lifecycle_status != 'ACTIVE')",
            (datetime.datetime.now(datetime.UTC).isoformat(),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        query_arr = np.asarray(query_vector, dtype=np.float64)
        q_norm = float(np.linalg.norm(query_arr))
        if q_norm == 0.0:
            return []

        owner_ids: list[str] = []
        vectors: list[list[float]] = []
        for row in rows:
            dimension: int = row["dimension"]
            blob: bytes = row["vector_blob"]
            vectors.append(list(struct.unpack(f"{dimension}f", blob)))
            owner_ids.append(row["owner_id"])

        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        dot_products = matrix @ query_arr
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        scores = np.where(norms > 0.0, dot_products / (safe_norms * q_norm), 0.0)

        results: list[tuple[str, float]] = [
            (owner_ids[i], float(scores[i]))
            for i in range(len(owner_ids))
            if float(scores[i]) >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Full-text search (FTS5 + BM25)
    # ------------------------------------------------------------------

    async def search_fts(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Full-text search via SQLite FTS5 with BM25 ranking.

        Returns an empty list when the FTS5 index is unavailable
        (backward compat for databases that predate the migration).

        Args:
            query: FTS5 query string (supports ``AND``, ``OR``,
                ``NOT``, prefix ``*``, column filters, etc.).
            top_k: Maximum number of results.

        Returns:
            List of ``(thought_id, bm25_score)`` sorted by relevance
            (higher = more relevant).

        """
        import time as _time  # noqa: PLC0415

        _t_start = _time.perf_counter()
        if not query or not query.strip():
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return []

        if not self._fts_probed:
            await self._probe_fts()

        if not self._fts_available:
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return []

        normalized_query = _normalize_fts_query(query)

        # bm25() returns negative values; negate so higher = more relevant.
        sql = (
            "SELECT t.thought_id, -bm25(thought_fts) AS score "
            "FROM thought_fts "
            "JOIN thought t ON t.rowid = thought_fts.rowid "
            "WHERE thought_fts MATCH ? "
            "AND (t.expires_at IS NULL OR t.expires_at > ?) "
            # Freshness floor: retired REFLECTIONs are excluded so a stale
            # synthesis does not out-rank fresh relevant thoughts.
            "AND NOT (t.thought_type = 'REFLECTION' AND t.lifecycle_status != 'ACTIVE') "
            "ORDER BY score DESC "
            "LIMIT ?"
        )
        cursor = await self._db.execute(
            sql,
            (normalized_query, datetime.datetime.now(datetime.UTC).isoformat(), top_k),
        )
        rows = await cursor.fetchall()
        results = [(row["thought_id"], float(row["score"])) for row in rows]
        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
        return results

    # ------------------------------------------------------------------
    # MindQL execution
    # ------------------------------------------------------------------

    async def execute_mindql(
        self,
        query: MindQLQuery,
        *,
        extensions: dict[str, MindQLExtension] | None = None,
    ) -> MindQLResult:
        """Execute an already-parsed MindQL query against this store's connection.

        This is the store-level entry point for the MindQL execution
        contract: it lets a caller whose connection is owned by this store
        run MindQL without reaching into store internals. Parse the query
        first with :func:`engrava.mindql.parse`.

        This method performs **no command-level policy filtering** — it will
        execute whatever command the parsed query carries (``FIND``,
        ``COUNT``, ``SELECT``, or a registered extension command). Callers
        that need to restrict the command set (for example an
        over-the-wire consumer exposing ``FIND`` only) **must** validate
        ``query.command`` *before* calling this method.

        Args:
            query: A parsed MindQL query.
            extensions: Optional registered MindQL extension commands. When
                omitted, no extension commands are available. Callers that
                expose extension commands supply their own map (the store
                holds no extension-command registry of its own).

        Returns:
            The ``MindQLResult`` produced by the executor, carrying
            ``columns``, ``rows``, ``count``, and the executed ``command``.

        Raises:
            MindQLParseError: If the executor rejects the query at
                execution time (for example a ``SELECT`` whose raw SQL is
                not a ``SELECT`` statement, or a ``FIND`` referencing an
                invalid column).

        """
        from engrava.mindql.executor import MindQLExecutor  # noqa: PLC0415

        executor = MindQLExecutor(self._db, extensions=extensions or {})
        return await executor.execute(query)

    # ------------------------------------------------------------------
    # Hybrid search (FTS5 + vector + recency fusion)
    # ------------------------------------------------------------------

    def _resolve_hybrid_defaults(
        self,
        *,
        fts_weight: float | None,
        vector_weight: float | None,
        recency_weight: float | None,
        recency_half_life: int | None,
        priority_weight: float | None = None,
        graph_weight: float | None = None,
    ) -> tuple[float, float, float, int, float, float]:
        """Resolve per-call hybrid settings against configured defaults.

        Args:
            fts_weight: Optional per-call FTS weight override.
            vector_weight: Optional per-call vector weight override.
            recency_weight: Optional per-call recency weight override.
            recency_half_life: Optional per-call recency half-life override.
            priority_weight: Optional per-call priority weight override.
            graph_weight: Optional per-call graph signal weight override.

        Returns:
            Resolved ``(fts_weight, vector_weight, recency_weight,
            recency_half_life, priority_weight, graph_weight)``.

        Raises:
            ValueError: If any weight is negative or half-life is invalid.

        """
        search_config = self._search_config

        resolved_fts_weight = (
            fts_weight
            if fts_weight is not None
            else (search_config.default_fts_weight if search_config is not None else 0.3)
        )
        resolved_vector_weight = (
            vector_weight
            if vector_weight is not None
            else (search_config.default_vector_weight if search_config is not None else 0.55)
        )
        resolved_recency_weight = (
            recency_weight
            if recency_weight is not None
            else (search_config.default_recency_weight if search_config is not None else 0.0)
        )
        resolved_recency_half_life = (
            recency_half_life
            if recency_half_life is not None
            else (search_config.recency_half_life if search_config is not None else 50)
        )
        resolved_priority_weight = (
            priority_weight
            if priority_weight is not None
            else (search_config.default_priority_weight if search_config is not None else 0.05)
        )
        resolved_graph_weight = (
            graph_weight
            if graph_weight is not None
            else (search_config.default_graph_weight if search_config is not None else 0.0)
        )

        if resolved_fts_weight < 0.0:
            msg = "fts_weight must be non-negative"
            raise ValueError(msg)
        if resolved_vector_weight < 0.0:
            msg = "vector_weight must be non-negative"
            raise ValueError(msg)
        if resolved_recency_weight < 0.0:
            msg = "recency_weight must be non-negative"
            raise ValueError(msg)
        if resolved_recency_half_life < 1:
            msg = "recency_half_life must be a positive integer"
            raise ValueError(msg)
        if resolved_priority_weight < 0.0:
            msg = "priority_weight must be non-negative"
            raise ValueError(msg)
        if resolved_graph_weight < 0.0:
            msg = "graph_weight must be non-negative"
            raise ValueError(msg)

        return (
            resolved_fts_weight,
            resolved_vector_weight,
            resolved_recency_weight,
            resolved_recency_half_life,
            resolved_priority_weight,
            resolved_graph_weight,
        )

    @staticmethod
    def _redistribute_hybrid_weights(
        *,
        fts_active: bool,
        vector_active: bool,
        recency_active: bool,
        priority_active: bool = False,
        graph_active: bool = False,
        fts_weight: float,
        vector_weight: float,
        recency_weight: float,
        priority_weight: float = 0.0,
        graph_weight: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        """Redistribute disabled-signal weights across active components."""
        active_weight = 0.0
        if fts_active:
            active_weight += fts_weight
        if vector_active:
            active_weight += vector_weight
        if recency_active:
            active_weight += recency_weight
        if priority_active:
            active_weight += priority_weight
        if graph_active:
            active_weight += graph_weight

        if active_weight == 0.0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            (fts_weight / active_weight) if fts_active else 0.0,
            (vector_weight / active_weight) if vector_active else 0.0,
            (recency_weight / active_weight) if recency_active else 0.0,
            (priority_weight / active_weight) if priority_active else 0.0,
            (graph_weight / active_weight) if graph_active else 0.0,
        )

    async def _load_recency_scores(
        self,
        *,
        thought_ids: set[str],
        current_cycle: int,
        recency_half_life: int,
    ) -> dict[str, float]:
        """Load recency scores for a candidate set of thought IDs."""
        if not thought_ids:
            return {}

        decay_rate = math.log(2) / recency_half_life
        placeholders = ", ".join("?" for _ in thought_ids)
        sql = (
            f"SELECT thought_id, updated_cycle FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders})"
        )
        cursor = await self._db.execute(sql, list(thought_ids))
        rows = await cursor.fetchall()

        scores: dict[str, float] = {}
        for row in rows:
            thought_id = row["thought_id"]
            updated_cycle = int(row["updated_cycle"])
            age = max(current_cycle - updated_cycle, 0)
            scores[thought_id] = math.exp(-decay_rate * age)
        return scores

    async def _load_priority_scores(
        self,
        *,
        thought_ids: set[str],
    ) -> dict[str, float]:
        """Load priority-boost scores for a candidate set of thought IDs.

        Maps each thought's ``priority`` enum value to a boost
        multiplier defined in ``SearchConfig``.  Thoughts whose
        priority is ``NULL`` or unknown receive a neutral score of
        ``0.0``.

        Args:
            thought_ids: Candidate thought IDs to score.

        Returns:
            Mapping of ``thought_id`` → priority boost score in
            ``[0.0, 1.0]``.

        """
        if not thought_ids:
            return {}

        search_config = self._search_config
        boost_map: dict[str, float] = {
            Priority.P1: search_config.priority_boost_p1 if search_config else 1.0,
            Priority.P2: search_config.priority_boost_p2 if search_config else 0.6,
            Priority.P3: search_config.priority_boost_p3 if search_config else 0.3,
            Priority.P4: search_config.priority_boost_p4 if search_config else 0.0,
        }

        placeholders = ", ".join("?" for _ in thought_ids)
        sql = (
            f"SELECT thought_id, priority FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders})"
        )
        cursor = await self._db.execute(sql, list(thought_ids))
        rows = await cursor.fetchall()

        scores: dict[str, float] = {}
        for row in rows:
            thought_id = row["thought_id"]
            priority_val = row["priority"]
            scores[thought_id] = boost_map.get(priority_val, 0.0)
        return scores

    async def _load_graph_signal(
        self,
        *,
        candidate_scores: dict[str, float],
        graph_edge_decay: float,
        max_neighbors: int,
    ) -> dict[str, float]:
        """Compute 1-hop-weighted graph boost for candidate thoughts.

        For each candidate, look up its 1-hop neighbours via edges.
        If a neighbour is also in the candidate pool, propagate its
        semantic base score (``max(fts, vector)``) weighted by
        ``edge.weight * graph_edge_decay``.

        Only content signals (FTS and vector) propagate
        through the graph.  Priority, recency, and graph scores are
        excluded to prevent hub-cascade effects.

        Args:
            candidate_scores: Semantic-only base scores
                (``max(fts, vector)`` per thought) for each candidate.
            graph_edge_decay: Decay factor applied to neighbour boost.
            max_neighbors: Maximum neighbours to consider per candidate.

        Returns:
            Mapping of thought_id to graph boost value.

        """
        if not candidate_scores:
            return {}

        all_ids = list(candidate_scores.keys())
        # Batch query — fetch all edges touching any candidate
        chunk_size = 450
        edge_rows: list[dict[str, object]] = []
        for i in range(0, len(all_ids), chunk_size):
            chunk = all_ids[i : i + chunk_size]
            placeholders = ", ".join("?" for _ in chunk)
            sql = (
                f"SELECT from_thought_id, to_thought_id, weight "  # noqa: S608
                f"FROM edge WHERE from_thought_id IN ({placeholders}) "
                f"OR to_thought_id IN ({placeholders}) "
                f"ORDER BY weight DESC"
            )
            params = [*chunk, *chunk]
            cursor = await self._db.execute(sql, params)
            rows = await cursor.fetchall()
            edge_rows.extend(
                {"from": r["from_thought_id"], "to": r["to_thought_id"], "weight": r["weight"]}
                for r in rows
            )

        # Build adjacency: candidate → list of (neighbour_id, edge_weight)
        adjacency: dict[str, list[tuple[str, float]]] = {}
        candidate_set = set(all_ids)
        for edge in edge_rows:
            from_id = str(edge["from"])
            to_id = str(edge["to"])
            w = float(edge["weight"])  # type: ignore[arg-type]
            if from_id in candidate_set:
                adjacency.setdefault(from_id, []).append((to_id, w))
            if to_id in candidate_set:
                adjacency.setdefault(to_id, []).append((from_id, w))

        # Compute boost per candidate (prefer highest-weight neighbours)
        boosts: dict[str, float] = {}
        for cid in all_ids:
            neighbors = sorted(
                adjacency.get(cid, []),
                key=lambda x: x[1],
                reverse=True,
            )[:max_neighbors]
            boost = 0.0
            for neighbour_id, edge_weight in neighbors:
                neighbour_base = candidate_scores.get(neighbour_id, 0.0)
                boost += edge_weight * neighbour_base * graph_edge_decay
            if boost > 0.0:
                boosts[cid] = boost

        return boosts

    async def _find_healthy_reflection_seeds(
        self,
        *,
        combined: dict[str, float],
        expansion_top_n: int,
        reflection_source_ceiling: int,
    ) -> list[str]:
        """Return top-ranked REFLECTION IDs from ``combined`` that pass the ceiling guard.

        Preserves score order from ``combined`` (SQL ``IN (…)`` does not guarantee
        row order, so the result is re-ranked against the original scores).
        REFLECTIONs with more ``CONSOLIDATED_FROM`` sources than
        ``reflection_source_ceiling`` are excluded to guard against giant-cluster
        pathology.

        Args:
            combined: Current score mapping (thought_id → score).
            expansion_top_n: Maximum number of healthy REFLECTION seeds to return.
            reflection_source_ceiling: Skip REFLECTIONs with strictly more
                ``CONSOLIDATED_FROM`` sources than this threshold.

        Returns:
            Ordered list of healthy REFLECTION IDs (best-scored first),
            up to ``expansion_top_n`` entries. Empty list when none qualify.

        """
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        window = ranked[: expansion_top_n * 5]
        candidate_ids = [tid for tid, _ in window]

        placeholders = ", ".join("?" for _ in candidate_ids)
        cursor = await self._db.execute(
            f"SELECT thought_id, thought_type FROM thought"  # noqa: S608
            f" WHERE thought_id IN ({placeholders})",
            candidate_ids,
        )
        rows = await cursor.fetchall()
        type_lookup: dict[str, str] = {str(r["thought_id"]): str(r["thought_type"]) for r in rows}
        reflection_candidates = [tid for tid, _ in window if type_lookup.get(tid) == "REFLECTION"][
            :expansion_top_n
        ]

        if not reflection_candidates:
            return []

        rc_placeholders = ", ".join("?" for _ in reflection_candidates)
        cursor = await self._db.execute(
            f"SELECT from_thought_id, COUNT(*) AS n FROM edge"  # noqa: S608
            f" WHERE edge_type = 'CONSOLIDATED_FROM'"
            f" AND from_thought_id IN ({rc_placeholders})"
            f" GROUP BY from_thought_id",
            reflection_candidates,
        )
        count_rows = await cursor.fetchall()
        source_counts: dict[str, int] = {str(r["from_thought_id"]): int(r["n"]) for r in count_rows}
        healthy = [
            rid
            for rid in reflection_candidates
            if source_counts.get(rid, 0) <= reflection_source_ceiling
        ]
        skipped = len(reflection_candidates) - len(healthy)
        if skipped > 0:
            logger.info(
                "expansion guard: skipped %d reflection(s) exceeding "
                "source_ceiling=%d (counts: %s)",
                skipped,
                reflection_source_ceiling,
                {rid: source_counts[rid] for rid in reflection_candidates if rid not in healthy},
            )
        return healthy

    async def _expand_via_consolidated_from(  # noqa: C901
        self,
        *,
        combined: dict[str, float],
        expansion_top_n: int,
        propagation_factor: float,
        max_sources_per_reflection: int,
        reflection_source_ceiling: int,
        expansion_sources: dict[str, str] | None = None,
    ) -> int:
        """Expand candidate pool by pulling source OBSERVATIONs from top REFLECTIONs.

        Identifies the top-ranked REFLECTIONs in ``combined`` (preserving their
        score-rank), traverses their outgoing ``CONSOLIDATED_FROM`` edges, and
        adds each **OBSERVATION**-type source to ``combined`` with::

            propagated_score = parent_score * propagation_factor * edge_weight

        If a source is already in ``combined`` the higher score wins.
        Non-OBSERVATION targets (REFLECTION, TASK, …) are silently skipped —
        only factual observations should be pulled through.
        REFLECTIONs with more than ``reflection_source_ceiling`` sources
        are skipped to guard against single-link chaining pathology
        (giant clusters whose centroid is too generic to be useful as an
        expansion seed).

        Args:
            combined: Current score mapping (thought_id → score). Modified
                in place.
            expansion_top_n: How many top-ranked REFLECTIONs to use as seeds.
            propagation_factor: Scalar applied to parent score during
                propagation (< 1.0 keeps sources below the REFLECTION).
            max_sources_per_reflection: At most this many source OBSs are
                pulled per REFLECTION, ordered by descending edge weight.
            reflection_source_ceiling: REFLECTIONs with strictly more sources
                than this value are skipped entirely.
            expansion_sources: Optional output mapping populated with
                ``source_id -> parent_reflection_id`` for candidates that
                were newly introduced by graph expansion.

        Returns:
            Number of new or updated entries written into ``combined``.
            Zero means no expansion occurred (no-op).

        """
        if not combined:
            return 0

        healthy = await self._find_healthy_reflection_seeds(
            combined=combined,
            expansion_top_n=expansion_top_n,
            reflection_source_ceiling=reflection_source_ceiling,
        )
        if not healthy:
            return 0

        # Traverse CONSOLIDATED_FROM edges (top sources by weight).
        h_placeholders = ", ".join("?" for _ in healthy)
        cursor = await self._db.execute(
            f"SELECT from_thought_id, to_thought_id, weight FROM edge"  # noqa: S608
            f" WHERE edge_type = 'CONSOLIDATED_FROM'"
            f" AND from_thought_id IN ({h_placeholders})"
            f" ORDER BY from_thought_id, weight DESC",
            healthy,
        )
        edge_rows = await cursor.fetchall()

        obs_ids = await self._filter_observation_ids([str(r["to_thought_id"]) for r in edge_rows])
        if not obs_ids:
            return 0

        # Group by parent and respect per-reflection cap; skip non-OBSERVATION targets.
        per_reflection: dict[str, list[tuple[str, float]]] = {}
        for row in edge_rows:
            parent_id = str(row["from_thought_id"])
            source_id = str(row["to_thought_id"])
            if source_id not in obs_ids:
                continue
            w = float(row["weight"])
            bucket = per_reflection.setdefault(parent_id, [])
            if len(bucket) < max_sources_per_reflection:
                bucket.append((source_id, w))

        # Propagate scores into combined; count newly written entries.
        added = 0
        for parent_id, sources in per_reflection.items():
            parent_score = combined.get(parent_id, 0.0)
            for source_id, edge_weight in sources:
                propagated = parent_score * propagation_factor * edge_weight
                existing = combined.get(source_id, 0.0)
                was_present = source_id in combined
                if propagated > existing:
                    combined[source_id] = propagated
                    added += 1
                    if expansion_sources is not None and not was_present:
                        expansion_sources[source_id] = parent_id

        return added

    async def _filter_observation_ids(self, candidate_ids: list[str]) -> frozenset[str]:
        """Return the subset of ``candidate_ids`` whose thought_type is OBSERVATION.

        Used by ``_expand_via_consolidated_from`` to strip non-factual
        targets (TASK, REFLECTION, …) from the expansion pool before
        propagating scores.

        Args:
            candidate_ids: Unfiltered list of target thought IDs.

        Returns:
            Frozenset containing only IDs of OBSERVATION-type thoughts.
            Empty frozenset when ``candidate_ids`` is empty.

        """
        unique = list(dict.fromkeys(candidate_ids))  # deduplicate, preserve insertion order
        if not unique:
            return frozenset()
        placeholders = ", ".join("?" for _ in unique)
        cursor = await self._db.execute(
            f"SELECT thought_id FROM thought"  # noqa: S608
            f" WHERE thought_type = 'OBSERVATION'"
            f" AND thought_id IN ({placeholders})",
            unique,
        )
        rows = await cursor.fetchall()
        return frozenset(str(r["thought_id"]) for r in rows)

    async def _fallback_hybrid_results(
        self,
        *,
        top_k: int,
        current_cycle: int | None,
        recency_half_life: int,
    ) -> list[tuple[str, float]]:
        """Fallback results when neither FTS nor vector search is usable."""
        thoughts = await self.list_thoughts(limit=top_k)
        # Apply the REFLECTION freshness floor consistently with the FTS and
        # vector paths: a retired REFLECTION must not surface here either.
        thoughts = [
            thought
            for thought in thoughts
            if not (
                thought.thought_type == ThoughtType.REFLECTION
                and thought.lifecycle_status != LifecycleStatus.ACTIVE
            )
        ]
        if current_cycle is None:
            return [(thought.thought_id, 0.0) for thought in thoughts]

        decay_rate = math.log(2) / recency_half_life
        ranked = [
            (
                thought.thought_id,
                math.exp(-decay_rate * max(current_cycle - thought.updated_cycle, 0)),
            )
            for thought in thoughts
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    async def _resolve_hybrid_state(
        self,
        *,
        query_text: str,
        query_vector: list[float] | None,
        current_cycle: int | None,
        recency_weight: float,
    ) -> tuple[bool, list[float] | None, bool]:
        """Resolve active hybrid-search components for the current query."""
        if not self._fts_probed:
            await self._probe_fts()

        fts_active = bool(self._fts_available and query_text and query_text.strip())

        effective_vector = query_vector
        if effective_vector is None and self._embedding_provider is not None and query_text.strip():
            effective_vector = await self._embedding_provider.embed(query_text)

        recency_active = current_cycle is not None and recency_weight > 0.0
        return (fts_active, effective_vector, recency_active)

    async def search_hybrid(  # noqa: C901, PLR0912, PLR0915
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        *,
        top_k: int = 10,
        fts_weight: float | None = None,
        vector_weight: float | None = None,
        recency_weight: float | None = None,
        recency_half_life: int | None = None,
        current_cycle: int | None = None,
        fts_top_k: int = 50,
        vector_top_k: int = 50,
        priority_weight: float | None = None,
        graph_weight: float | None = None,
        graph_edge_decay: float | None = None,
        include_reflections: bool = True,
        reflection_boost: float | None = None,
    ) -> HybridSearchResult:
        """Hybrid search combining FTS5 + vector + recency + priority + graph signals.

        Calls ``search_fts()`` and ``search_similar()`` independently,
        normalizes BM25 scores to ``[0, 1]`` via min-max, computes
        exponential recency decay, applies priority boost, then adds
        1-hop-weighted graph boost, and returns merged results
        sorted by combined score.

        Graceful degradation:
            - If FTS5 unavailable or ``query_text`` empty → FTS skipped.
            - If ``query_vector`` is ``None`` and no provider → vector skipped.
            - If ``current_cycle`` is ``None`` → recency skipped.
            - If ``priority_weight`` is ``0.0`` → priority skipped.
            - If ``graph_weight`` is ``0.0`` → graph skipped.
            - Disabled weights redistributed proportionally to active signals.
            - If all signals disabled → fallback to ``list_thoughts(LIMIT top_k)``.

        Args:
            query_text: Text query for FTS5 keyword search.
            query_vector: Embedding vector for similarity search.
                When ``None`` and an embedding provider is configured,
                the query text is auto-embedded.
            top_k: Maximum number of final merged results.
            fts_weight: Optional FTS5 fusion-weight override.
            vector_weight: Optional vector fusion-weight override.
            recency_weight: Optional recency fusion-weight override.
            recency_half_life: Optional recency half-life override.
            current_cycle: Current cycle number for recency calculation.
            fts_top_k: Max candidates from FTS5 before fusion.
            vector_top_k: Max candidates from vector search before fusion.
            priority_weight: Optional priority fusion-weight override.
            graph_weight: Optional graph signal fusion-weight override.
            graph_edge_decay: Optional graph edge decay override.
            include_reflections: When ``False``, REFLECTION thoughts are
                excluded from results.
            reflection_boost: Multiplier applied to REFLECTION thought
                scores. ``None`` uses the value from ``SearchConfig``
                (default ``1.2``).

        Returns:
            ``HybridSearchResult`` with ranked results and diagnostics.

        """
        import time as _time  # noqa: PLC0415

        from engrava.domain.models.search import HybridSearchResult  # noqa: PLC0415

        _t_start = _time.perf_counter()

        backends_used: set[str] = set()
        (
            resolved_fts_weight,
            resolved_vector_weight,
            resolved_recency_weight,
            resolved_recency_half_life,
            resolved_priority_weight,
            resolved_graph_weight,
        ) = self._resolve_hybrid_defaults(
            fts_weight=fts_weight,
            vector_weight=vector_weight,
            recency_weight=recency_weight,
            recency_half_life=recency_half_life,
            priority_weight=priority_weight,
            graph_weight=graph_weight,
        )

        resolved_graph_edge_decay = (
            graph_edge_decay
            if graph_edge_decay is not None
            else (self._search_config.graph_edge_decay if self._search_config is not None else 0.5)
        )
        resolved_max_neighbors = (
            self._search_config.max_neighbors_per_candidate
            if self._search_config is not None
            else 5
        )

        # --- Determine active signals and redistribute weights ---
        fts_active, effective_vector, recency_active = await self._resolve_hybrid_state(
            query_text=query_text,
            query_vector=query_vector,
            current_cycle=current_cycle,
            recency_weight=resolved_recency_weight,
        )
        vector_active = effective_vector is not None
        priority_active = resolved_priority_weight > 0.0
        graph_active = resolved_graph_weight > 0.0

        (
            eff_fts_w,
            eff_vec_w,
            eff_rec_w,
            eff_pri_w,
            eff_gra_w,
        ) = self._redistribute_hybrid_weights(
            fts_active=fts_active,
            vector_active=vector_active,
            recency_active=recency_active,
            priority_active=priority_active,
            graph_active=graph_active,
            fts_weight=resolved_fts_weight,
            vector_weight=resolved_vector_weight,
            recency_weight=resolved_recency_weight,
            priority_weight=resolved_priority_weight,
            graph_weight=resolved_graph_weight,
        )

        if not fts_active and not vector_active:
            if recency_active:
                backends_used.add("recency")
            fallback = await self._fallback_hybrid_results(
                top_k=top_k,
                current_cycle=current_cycle,
                recency_half_life=resolved_recency_half_life,
            )
            if priority_active and fallback:
                backends_used.add("priority")
                priority_scores = await self._load_priority_scores(
                    thought_ids={tid for tid, _ in fallback},
                )
                fallback = [
                    (tid, score + priority_scores.get(tid, 0.0) * eff_pri_w)
                    for tid, score in fallback
                ]
                fallback.sort(key=lambda x: x[1], reverse=True)
            # --- REFLECTION filter in fallback path ---
            if not include_reflections and fallback:
                fb_ids = [tid for tid, _ in fallback]
                placeholders = ", ".join("?" for _ in fb_ids)
                cursor = await self._db.execute(
                    f"SELECT thought_id FROM thought"  # noqa: S608
                    f" WHERE thought_type = 'REFLECTION'"
                    f" AND thought_id IN ({placeholders})",
                    fb_ids,
                )
                rows = await cursor.fetchall()
                ref_set = {str(r["thought_id"]) for r in rows}
                fallback = [(tid, s) for tid, s in fallback if tid not in ref_set]

            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)

            return HybridSearchResult(
                results=fallback,
                backends_used=frozenset(backends_used),
            )

        token = _SUPPRESS_SEARCH_METRICS.set(True)
        try:
            # --- Gather FTS results ---
            if fts_active:
                backends_used.add("fts5")
                fts_results = await self.search_fts(query_text, top_k=fts_top_k)
            else:
                fts_results = []

            # --- Gather vector results ---
            vec_results: list[tuple[str, float]] = []
            if effective_vector is not None:
                vec_results = await self.search_similar(effective_vector, top_k=vector_top_k)
                backends_used.add("vector")
        finally:
            _SUPPRESS_SEARCH_METRICS.reset(token)

        # --- Fuse scores ---
        fts_normalized = _normalize_min_max(fts_results)

        # Semantic-only base scores for graph signal (max(fts, vector))
        semantic_base: dict[str, float] = {}
        for tid, score in fts_normalized:
            semantic_base[tid] = max(semantic_base.get(tid, 0.0), score)
        for tid, score in vec_results:
            semantic_base[tid] = max(semantic_base.get(tid, 0.0), score)

        combined: dict[str, float] = {}
        for tid, score in fts_normalized:
            combined[tid] = combined.get(tid, 0.0) + score * eff_fts_w
        for tid, score in vec_results:
            combined[tid] = combined.get(tid, 0.0) + score * eff_vec_w

        # --- Recency signal ---
        if recency_active:
            backends_used.add("recency")
            recency_scores = await self._load_recency_scores(
                thought_ids=set(combined.keys()),
                current_cycle=current_cycle if current_cycle is not None else 0,
                recency_half_life=resolved_recency_half_life,
            )
            for thought_id, recency_score in recency_scores.items():
                combined[thought_id] = combined.get(thought_id, 0.0) + recency_score * eff_rec_w

        # --- Priority signal ---
        if priority_active and combined:
            backends_used.add("priority")
            priority_scores = await self._load_priority_scores(
                thought_ids=set(combined.keys()),
            )
            for thought_id, priority_score in priority_scores.items():
                combined[thought_id] = combined.get(thought_id, 0.0) + priority_score * eff_pri_w

        # --- Graph signal (1-hop-weighted, semantic base only) ---
        if graph_active and combined:
            graph_boosts = await self._load_graph_signal(
                candidate_scores=semantic_base,
                graph_edge_decay=resolved_graph_edge_decay,
                max_neighbors=resolved_max_neighbors,
            )
            if graph_boosts:
                backends_used.add("graph")
                for thought_id, graph_boost in graph_boosts.items():
                    combined[thought_id] = combined.get(thought_id, 0.0) + graph_boost * eff_gra_w

        # --- Candidate expansion via CONSOLIDATED_FROM ---
        expansion_cfg = self._search_config
        expansion_enabled = (
            expansion_cfg.graph_expansion_enabled if expansion_cfg is not None else True
        )
        if expansion_enabled and combined:
            _added = await self._expand_via_consolidated_from(
                combined=combined,
                expansion_top_n=(
                    expansion_cfg.graph_expansion_top_n if expansion_cfg is not None else 5
                ),
                propagation_factor=(
                    expansion_cfg.graph_expansion_propagation_factor
                    if expansion_cfg is not None
                    else 0.7
                ),
                max_sources_per_reflection=(
                    expansion_cfg.graph_expansion_max_sources_per_reflection
                    if expansion_cfg is not None
                    else 20
                ),
                reflection_source_ceiling=(
                    expansion_cfg.graph_expansion_reflection_source_ceiling
                    if expansion_cfg is not None
                    else 50
                ),
                expansion_sources=None,
            )
            if _added > 0:
                backends_used.add("graph_expansion")

        # --- REFLECTION filter + boost + top-K cap ---
        resolved_reflection_boost = (
            reflection_boost
            if reflection_boost is not None
            else (self._search_config.reflection_boost if self._search_config is not None else 1.0)
        )
        resolved_reflection_topk_cap = (
            self._search_config.reflection_topk_cap if self._search_config is not None else 0.3
        )
        _needs_reflection_ids = (
            not include_reflections
            or resolved_reflection_boost != 1.0
            or resolved_reflection_topk_cap < 1.0
        )
        reflection_ids: set[str] = set()
        if _needs_reflection_ids and combined:
            candidate_ids = list(combined)
            placeholders = ", ".join("?" for _ in candidate_ids)
            cursor = await self._db.execute(
                f"SELECT thought_id FROM thought"  # noqa: S608
                f" WHERE thought_type = 'REFLECTION'"
                f" AND thought_id IN ({placeholders})",
                candidate_ids,
            )
            rows = await cursor.fetchall()
            reflection_ids = {str(r["thought_id"]) for r in rows}

        if not include_reflections:
            for rid in reflection_ids:
                combined.pop(rid, None)
        elif resolved_reflection_boost != 1.0 and reflection_ids:
            for rid in reflection_ids:
                if rid in combined:
                    combined[rid] = combined[rid] * resolved_reflection_boost

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        final = ranked[:top_k]

        # --- reflection_topk_cap enforcement ---
        if include_reflections and resolved_reflection_topk_cap < 1.0 and reflection_ids:
            _max_ref_slots = max(0, int(top_k * resolved_reflection_topk_cap))
            _ref_in_final = [
                (i, tid, s) for i, (tid, s) in enumerate(final) if tid in reflection_ids
            ]
            if len(_ref_in_final) > _max_ref_slots:
                import logging as _logging_mod  # noqa: PLC0415

                _excess = len(_ref_in_final) - _max_ref_slots
                _to_evict = {
                    tid for _, tid, _ in sorted(_ref_in_final, key=lambda x: x[2])[:_excess]
                }
                _off_list_obs = [(tid, s) for tid, s in ranked[top_k:] if tid not in reflection_ids]
                if len(_off_list_obs) < _excess:
                    _logging_mod.getLogger(__name__).warning(
                        "reflection_topk_cap: %d excess REFLECTION(s) to evict but only %d "
                        "off-list non-REFLECTION candidates available — partial enforcement",
                        _excess,
                        len(_off_list_obs),
                    )
                _fill = _off_list_obs[:_excess]
                _kept = [(tid, s) for tid, s in final if tid not in _to_evict]
                final = sorted(_kept + _fill, key=lambda x: x[1], reverse=True)[:top_k]

        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)

        return HybridSearchResult(
            results=final,
            backends_used=frozenset(backends_used),
        )

    async def search_reflections_only(
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        *,
        top_k: int = 10,
        current_cycle: int | None = None,
    ) -> HybridSearchResult:
        """Return only REFLECTION thoughts ranked by cosine similarity.

        Directly fetches all ``ThoughtType.REFLECTION`` thoughts and
        scores them against the query vector — guarantees completeness
        regardless of how many regular thoughts are in the store (no
        pagination gap like an over-fetch approach would have).

        When ``current_cycle`` is provided, a recency blend is applied
        alongside cosine similarity using the configured
        ``default_recency_weight``.

        When no query vector is available and no embedding provider is
        configured, all REFLECTION thoughts are returned unranked.

        Args:
            query_text: Text used for auto-embedding when no
                ``query_vector`` is supplied and a provider is configured.
            query_vector: Embedding vector for cosine similarity ranking.
            top_k: Maximum number of results to return.
            current_cycle: Current cycle for optional recency blending.

        Returns:
            ``HybridSearchResult`` containing only REFLECTION thoughts,
            sorted by cosine similarity (and optionally recency) descending.

        """
        import math  # noqa: PLC0415
        import struct  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        from engrava.domain.models.search import HybridSearchResult  # noqa: PLC0415

        _t_start = _time.perf_counter()

        # Resolve effective query vector (auto-embed if provider available)
        effective_vector = query_vector
        if effective_vector is None and self._embedding_provider is not None and query_text.strip():
            effective_vector = await self._embedding_provider.embed(query_text)

        # Fetch all REFLECTION thought IDs directly — complete, no pagination
        # gap. Retired REFLECTIONs (lifecycle != ACTIVE) are excluded by the
        # same freshness floor the similarity/hybrid paths apply.
        cursor = await self._db.execute(
            "SELECT thought_id FROM thought "
            "WHERE thought_type = 'REFLECTION' AND lifecycle_status = 'ACTIVE'"
        )
        rows = await cursor.fetchall()
        reflection_ids = [str(r["thought_id"]) for r in rows]

        if not reflection_ids:
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return HybridSearchResult(results=[], backends_used=frozenset())

        if effective_vector is None:
            # No scoring available — return unranked, capped at top_k
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return HybridSearchResult(
                results=[(rid, 0.0) for rid in reflection_ids[:top_k]],
                backends_used=frozenset(),
            )

        # Score each REFLECTION by cosine similarity to the query vector
        q_norm = math.sqrt(sum(x * x for x in effective_vector))
        if q_norm == 0.0:
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return HybridSearchResult(
                results=[(rid, 0.0) for rid in reflection_ids[:top_k]],
                backends_used=frozenset({"vector"}),
            )

        backends_used_set: set[str] = {"vector"}
        scores: list[tuple[str, float]] = []
        for rid in reflection_ids:
            emb = await self.get_embedding(rid)
            if emb is None:
                scores.append((rid, 0.0))
                continue
            vec = list(struct.unpack(f"{emb.dimension}f", emb.vector_blob))
            v_norm = math.sqrt(sum(x * x for x in vec))
            if v_norm == 0.0:
                scores.append((rid, 0.0))
                continue
            dot = sum(a * b for a, b in zip(effective_vector, vec, strict=False))
            scores.append((rid, dot / (q_norm * v_norm)))

        # Optional recency blend when current_cycle is provided
        if current_cycle is not None:
            backends_used_set.add("recency")
            search_config = self._search_config
            recency_weight = search_config.default_recency_weight if search_config else 0.1
            recency_half_life = search_config.recency_half_life if search_config else 50
            if recency_weight > 0.0:
                recency_scores = await self._load_recency_scores(
                    thought_ids={rid for rid, _ in scores},
                    current_cycle=current_cycle,
                    recency_half_life=recency_half_life,
                )
                vec_w = 1.0 - recency_weight
                scores = [
                    (rid, sim * vec_w + recency_scores.get(rid, 0.0) * recency_weight)
                    for rid, sim in scores
                ]

        scores.sort(key=lambda x: x[1], reverse=True)
        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
        return HybridSearchResult(
            results=scores[:top_k],
            backends_used=frozenset(backends_used_set),
        )

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    async def record_access(self, thought_id: str) -> None:
        """Record an explicit access to a thought.

        Increments ``access_count`` by 1 and sets ``last_accessed_at``
        to the current UTC time.

        Args:
            thought_id: UUID of the thought to mark as accessed.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.

        """
        now = datetime.datetime.now(datetime.UTC).isoformat()
        cursor = await self._db.execute(
            "UPDATE thought SET access_count = access_count + 1, "
            "last_accessed_at = ? WHERE thought_id = ?",
            (now, thought_id),
        )
        if cursor.rowcount == 0:
            raise ThoughtNotFoundError(thought_id)
        await self._maybe_commit()

    # ------------------------------------------------------------------
    # ActionRecord CRUD
    # ------------------------------------------------------------------

    async def create_action(self, action: ActionRecord) -> ActionRecord:
        """Persist a new action record.

        Args:
            action: The action record to create.

        Returns:
            The persisted action record.

        """
        await self._db.execute(
            "INSERT INTO action "
            "(action_id, source_thought_id, action_type, intent, "
            " status, verification_status, raw_metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                action.action_id,
                action.source_thought_id,
                action.action_type.value,
                action.intent,
                action.status.value,
                action.verification_status.value,
                action.raw_metrics_json,
            ),
        )
        await self._maybe_commit()
        return action

    async def get_actions(self, thought_id: str) -> list[ActionRecord]:
        """Retrieve actions linked to a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            List of action records.

        """
        cursor = await self._db.execute(
            "SELECT * FROM action WHERE source_thought_id = ?", (thought_id,)
        )
        rows = await cursor.fetchall()
        return [_row_to_action(r) for r in rows]


# ------------------------------------------------------------------
# Row -> Domain mapper functions (private, module-level)
# ------------------------------------------------------------------


def _row_to_edge(row: aiosqlite.Row) -> EdgeRecord:
    """Map a SQLite row to an EdgeRecord domain model.

    Args:
        row: A row from the edge table.

    Returns:
        An EdgeRecord domain model.

    """
    keys = row.keys()
    source_raw = row["source"] if "source" in keys else None
    decay_raw = row["decay_multiplier"] if "decay_multiplier" in keys else 1.0
    valid_from_raw = row["valid_from"] if "valid_from" in keys else None
    valid_until_raw = row["valid_until"] if "valid_until" in keys else None
    return EdgeRecord(
        edge_id=row["edge_id"],
        from_thought_id=row["from_thought_id"],
        to_thought_id=row["to_thought_id"],
        edge_type=EdgeType(row["edge_type"]),
        weight=row["weight"],
        created_cycle=row["created_cycle"],
        source=KnowledgeSource(source_raw) if source_raw else KnowledgeSource.EXPERIENCE,
        decay_multiplier=float(decay_raw) if decay_raw else 1.0,
        valid_from=valid_from_raw,
        valid_until=valid_until_raw,
    )


def _normalize_fts_query(query: str) -> str:
    """Normalize user-facing FTS queries to SQLite-compatible syntax.

    SQLite FTS5 treats hyphens as operators in bare tokens, which breaks
    intuitive identifier-style prefix queries such as ``REQ-FUNC*``.
    This normalizer preserves the public API contract by rewriting those
    simple tokens to the accepted form ``"REQ-FUNC"*``.  It also strips
    trailing natural-language punctuation like ``?`` and ``,`` from bare
    tokens so user questions can be passed directly into FTS5.
    """
    parts = query.split()
    normalized_parts = [_normalize_fts_token(part) for part in parts]
    return " ".join(part for part in normalized_parts if part)


def _strip_fts_boundary_punctuation(raw: str) -> str:
    """Strip unsupported leading and trailing punctuation from a bare token."""
    while raw and not (raw[0].isalnum() or raw[0] in {"_", '"'}):
        raw = raw[1:]

    while raw and not (raw[-1].isalnum() or raw[-1] in {"_", "*"}):
        raw = raw[:-1]

    return raw


def _sanitize_fts_bare_token(raw: str) -> str:
    """Remove unsupported FTS punctuation from an unquoted bare token."""
    stripped = _strip_fts_boundary_punctuation(raw)
    return _FTS_UNSAFE_CHAR_RE.sub("", stripped)


def _normalize_fts_token(token: str) -> str:
    """Normalize a single FTS token if it contains a hyphenated identifier."""
    if not token or '"' in token:
        return token
    if token in {"AND", "OR", "NOT"}:
        return token

    leading = ""
    trailing = ""
    raw = token
    while raw.startswith("("):
        leading += "("
        raw = raw[1:]
    while raw.endswith(")"):
        trailing = ")" + trailing
        raw = raw[:-1]

    if _FTS_FIELD_FILTER_RE.match(raw):
        normalized = f"{leading}{raw}{trailing}"
    else:
        raw = _sanitize_fts_bare_token(raw)
        if not raw:
            return ""

        suffix = ""
        if raw.endswith("*"):
            raw = raw[:-1]
            suffix = "*"

        normalized = (
            f'{leading}"{raw}"{suffix}{trailing}'
            if "-" in raw
            else f"{leading}{raw}{suffix}{trailing}"
        )

    return normalized


def _row_to_action(row: aiosqlite.Row) -> ActionRecord:
    """Map a SQLite row to an ActionRecord domain model.

    Args:
        row: A row from the action table.

    Returns:
        An ActionRecord domain model.

    """
    return ActionRecord(
        action_id=row["action_id"],
        source_thought_id=row["source_thought_id"],
        action_type=ActionType(row["action_type"]),
        intent=row["intent"],
        status=ActionStatus(row["status"]),
        verification_status=VerificationStatus(row["verification_status"]),
        raw_metrics_json=row["raw_metrics_json"],
    )


def _row_to_embedding(row: aiosqlite.Row) -> EmbeddingRecord:
    """Map a SQLite row to an EmbeddingRecord domain model.

    Args:
        row: A row from the embedding table.

    Returns:
        An EmbeddingRecord domain model.

    """
    return EmbeddingRecord(
        embedding_id=row["embedding_id"],
        owner_type=row["owner_type"],
        owner_id=row["owner_id"],
        model_name=row["model_name"],
        dimension=row["dimension"],
        vector_blob=row["vector_blob"],
        created_at=row["created_at"],
    )


def _encode_consolidated(value: list[str] | None) -> str | None:
    """Encode consolidated_from list as JSON string for storage.

    Args:
        value: List of source thought IDs, or None.

    Returns:
        JSON string, or None.

    """
    if value is None:
        return None
    return json.dumps(value)


def _decode_consolidated(raw: str | None) -> list[str] | None:
    """Decode consolidated_from JSON string from storage.

    Args:
        raw: JSON string from database, or None.

    Returns:
        List of source thought IDs, or None.

    """
    if raw is None:
        return None
    result: list[str] = json.loads(raw)
    return result


def _compute_content_hash(content: str) -> str:
    """Compute the SHA-256 hex digest of *content* for ingest deduplication.

    The hash is computed over the UTF-8 encoded bytes of *content* with
    no normalization (no whitespace, casing, or unicode-form folding) so
    that "exact same content" is the well-defined semantic, and any
    deliberate formatting difference is treated as a distinct thought.

    Args:
        content: The thought content string.

    Returns:
        Lowercase hex digest of ``sha256(content.encode("utf-8"))``.

    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Metadata validation
# ----------------------------------------------------------------------

#: Soft warning threshold (bytes) for serialized ``ThoughtRecord.metadata``.
#:
#: Crossing this size emits a ``logger.warning`` to flag callers that may
#: be smuggling structured payloads through metadata where the ``content``
#: field would be the appropriate place.
_METADATA_WARN_BYTES = 4 * 1024

#: Hard rejection threshold (bytes) for serialized ``ThoughtRecord.metadata``.
#:
#: Crossing this size raises ``ValueError`` outright.  SQLite TEXT can
#: store much larger values, but query latency on the JSON1 ``json_extract``
#: paths used by downstream filtering degrades meaningfully past this
#: scale, so the limit is set well below SQLite's hard cap.
_METADATA_REJECT_BYTES = 64 * 1024


def _validate_metadata_value(value: MetadataValue, key_path: str) -> None:
    """Recursively validate a metadata value's structure.

    Walks nested ``dict[str, MetadataValue]`` namespaces and rejects
    anything that is not a scalar leaf (``str``, ``int``, ``float``,
    ``bool``, ``None``) or another mapping with string keys.  The
    ``key_path`` argument is dotted so error messages point at the
    offending location inside nested namespaces (e.g.
    ``source.tags``).

    Args:
        value: Candidate value at the current key path.
        key_path: Dot-joined key path from the root for error messages.

    Raises:
        ValueError: If a non-string key, a non-scalar leaf, or a
            list/tuple/set/custom container is encountered at any depth.

    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                msg = f"metadata key at {key_path} must be str, got {type(nested_key).__name__}"
                raise ValueError(msg)  # noqa: TRY004
            _validate_metadata_value(nested_value, f"{key_path}.{nested_key}")
        return
    # Lists, tuples, sets, custom objects -> reject.
    msg = (
        f"metadata value at {key_path} type {type(value).__name__} not allowed; "
        f"allowed: str, int, float, bool, None, dict[str, MetadataValue]"
    )
    raise ValueError(msg)


def _validate_metadata(metadata: dict[str, MetadataValue]) -> None:
    """Validate metadata dict structure and serialized size.

    Caller-supplied metadata must be a ``dict`` keyed by ``str`` with
    leaf values restricted to ``str | int | float | bool | None`` so the
    column can be queried directly via SQLite's JSON1 functions without
    secondary parsing.  Nested ``dict[str, MetadataValue]`` values are
    accepted (structured namespaces per ``ThoughtSource``).
    Lists, tuples, sets and custom objects are rejected at every depth.

    Note that ``bool`` is a subclass of ``int`` in Python — both are
    accepted as scalar values, and the deserialized round trip preserves
    the original type because :func:`json.dumps` and :func:`json.loads`
    distinguish them.

    Size rules:

    * Serialized size > 4 KiB  -> ``logger.warning`` (soft signal).
    * Serialized size > 64 KiB -> :class:`ValueError` (hard rejection).

    Args:
        metadata: Caller-supplied attributes to validate.

    Raises:
        ValueError: If the structure or size invariants are violated.

    """
    # ValueError is the contractual surface for every metadata-validation
    # failure — callers (and tests) catch a single exception type for both
    # shape and size violations.  TRY004 is silenced to preserve that API.
    if not isinstance(metadata, dict):
        msg = f"metadata must be dict, got {type(metadata).__name__}"
        raise ValueError(msg)  # noqa: TRY004
    for key, value in metadata.items():
        if not isinstance(key, str):
            msg = f"metadata key must be str, got {type(key).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        _validate_metadata_value(value, key)
    serialized = json.dumps(metadata, ensure_ascii=False)
    size_bytes = len(serialized.encode("utf-8"))
    if size_bytes > _METADATA_REJECT_BYTES:
        msg = (
            f"metadata serialized size {size_bytes} bytes exceeds maximum "
            f"{_METADATA_REJECT_BYTES} bytes; consider storing large "
            "payloads as `content` or external references"
        )
        raise ValueError(msg)
    if size_bytes > _METADATA_WARN_BYTES:
        logger.warning(
            "metadata size %d bytes exceeds soft limit %d bytes — consider "
            "whether structured data should be in `content` field instead",
            size_bytes,
            _METADATA_WARN_BYTES,
        )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (numpy-accelerated).

    Returns 0.0 for zero-magnitude vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0], or 0.0 if either vector has zero norm.

    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _normalize_min_max(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize scores to ``[0, 1]`` via min-max scaling.

    When all scores are identical (``hi == lo``), every score becomes
    ``1.0`` to avoid division by zero.

    Args:
        results: ``(thought_id, raw_score)`` pairs.

    Returns:
        ``(thought_id, normalized_score)`` pairs.

    """
    if not results:
        return []
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(tid, 1.0) for tid, _ in results]
    return [(tid, (s - lo) / (hi - lo)) for tid, s in results]

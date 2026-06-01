"""EngravaManager — per-service database isolation.

Manages a pool of ``SqliteEngravaCore`` instances, one per named
service.  Each service gets its own SQLite file under ``data_dir``,
with independent schema, embedding model, FTS5 index, and WAL journal.

Usage::

    manager = EngravaManager(data_dir=Path("./data/engrava"))
    store = await manager.get_store("default")
    thought = await store.get_thought("abc")
    await manager.close_all()

Or as an async context manager::

    async with EngravaManager(data_dir=Path("./data")) as mgr:
        store = await mgr.get_store("default")
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Self

import aiosqlite

from engrava.config import (
    ServicesConfig,
    resolve_embedding_provider,
)
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from pathlib import Path

    from engrava.config import EmbeddingConfig, SearchConfig

logger = logging.getLogger(__name__)


class EngravaManager:
    """Manages per-service ``SqliteEngravaCore`` instances.

    Each service is a separate SQLite database file (``<name>.db``)
    under ``data_dir``.  Stores are lazily initialized on first
    ``get_store()`` call and cached for subsequent access.

    Args:
        data_dir: Directory for per-service database files.
        default_embeddings: Fallback embedding config for services
            without explicit overrides.
        default_search: Default hybrid-search weights.
        wal_mode: Enable WAL journal mode.
        vector_backend: Vector backend name (``"numpy"`` or ``"sqlite-vec"``).
        embedding_dimension: Default embedding vector dimension.
        services_config: Optional ``ServicesConfig`` for per-service overrides.

    """

    def __init__(
        self,
        data_dir: Path,
        *,
        default_embeddings: EmbeddingConfig | None = None,
        default_search: SearchConfig | None = None,
        wal_mode: bool = True,
        vector_backend: str = "numpy",
        embedding_dimension: int = 384,
        services_config: ServicesConfig | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._default_embeddings = default_embeddings
        self._default_search = default_search
        self._wal_mode = wal_mode
        self._vector_backend = vector_backend
        self._embedding_dimension = embedding_dimension
        self._services_config = services_config
        self._stores: dict[str, SqliteEngravaCore] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        Returns:
            This manager instance.

        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit the async context manager and close all stores.

        Args:
            *exc: Exception info (type, value, traceback).

        """
        await self.close_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_store(self, service_name: str) -> SqliteEngravaCore:
        """Get or create the store for a named service.

        The store is lazily initialized: the database file and schema are
        created on the first call.  Subsequent calls return the cached
        instance.

        Args:
            service_name: Unique service identifier.  Must match
                ``^[a-z][a-z0-9_-]{0,62}$``.

        Returns:
            A fully initialized ``SqliteEngravaCore`` instance
            with its own database, schema, and embedding provider.

        Raises:
            ConfigError: If the service name is invalid.

        """
        from engrava.config import _validate_service_name  # noqa: PLC0415

        _validate_service_name(service_name)

        if service_name in self._stores:
            return self._stores[service_name]

        async with self._lock:
            # Double-check after acquiring the lock.
            if service_name in self._stores:
                return self._stores[service_name]
            store = await self._create_store(service_name)
            self._stores[service_name] = store
            return store

    def service_exists(self, service_name: str) -> bool:
        """Check whether a service database file exists on disk.

        Does **not** create or open the database.

        Args:
            service_name: Service identifier.

        Returns:
            ``True`` if ``<data_dir>/<service_name>.db`` exists.

        """
        return self._service_db_path(service_name).exists()

    async def list_services(self) -> list[str]:
        """List all service names with existing database files.

        Scans ``data_dir`` for ``*.db`` files and returns their stems
        as service names.

        Returns:
            Sorted list of service names found on disk.

        """
        if not self._data_dir.exists():
            return []
        return sorted(p.stem for p in self._data_dir.iterdir() if p.suffix == ".db" and p.is_file())

    async def delete_service(self, service_name: str) -> None:
        """Delete a service's database and remove from cache.

        Closes the store's connection (if open) and removes the
        ``<name>.db`` file plus any WAL/SHM journals.

        Args:
            service_name: Name of the service to delete.

        Raises:
            ConfigError: If the service name is invalid.
            FileNotFoundError: If the database file does not exist.

        """
        from engrava.config import _validate_service_name  # noqa: PLC0415

        _validate_service_name(service_name)

        # Close cached store if open.
        if service_name in self._stores:
            store = self._stores.pop(service_name)
            store._owns_connection = True  # noqa: SLF001
            await store.close()

        db_path = self._service_db_path(service_name)
        if not db_path.exists():
            msg = f"Service database not found: {db_path}"
            raise FileNotFoundError(msg)

        db_path.unlink()
        # Clean up WAL/SHM journal files.
        for suffix in (".db-wal", ".db-shm"):
            journal = db_path.with_suffix(suffix)
            if journal.exists():
                journal.unlink()

        logger.info("Deleted service %r database: %s", service_name, db_path)

    async def close_all(self) -> None:
        """Close all cached store connections.

        Safe to call multiple times.  After this call, cached stores
        are cleared and ``get_store()`` will create fresh connections.
        """
        for name, store in self._stores.items():
            try:
                store._owns_connection = True  # noqa: SLF001
                await store.close()
            except Exception:  # noqa: BLE001
                logger.warning("Error closing store %r", name, exc_info=True)
        self._stores.clear()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def from_config(cls, config: ServicesConfig, **kwargs: object) -> EngravaManager:
        """Create a manager from a ``ServicesConfig``.

        Args:
            config: Parsed services configuration.
            **kwargs: Additional keyword arguments forwarded to the
                constructor (``default_embeddings``, ``wal_mode``, etc.).

        Returns:
            A new ``EngravaManager`` instance.

        """
        return cls(
            data_dir=config.data_dir,
            services_config=config,
            **kwargs,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _service_db_path(self, service_name: str) -> Path:
        """Compute the database file path for a service.

        Args:
            service_name: Service identifier.

        Returns:
            Path to ``<data_dir>/<service_name>.db``.

        """
        return self._data_dir / f"{service_name}.db"

    def _resolve_embedding_config(self, service_name: str) -> EmbeddingConfig | None:
        """Resolve the embedding config for a service.

        Per-service overrides take precedence over the manager-level
        default.

        Args:
            service_name: Service identifier.

        Returns:
            Merged ``EmbeddingConfig``, or ``None``.

        """
        if self._services_config and service_name in self._services_config.configs:
            svc_cfg = self._services_config.configs[service_name]
            if svc_cfg.embeddings is not None:
                return svc_cfg.embeddings
        return self._default_embeddings

    async def _create_store(self, service_name: str) -> SqliteEngravaCore:
        """Create and initialize a new store for a service.

        Creates the data directory and database file if needed,
        applies the schema, and configures the embedding provider.

        Args:
            service_name: Service identifier.

        Returns:
            Fully initialized ``SqliteEngravaCore``.

        """
        self._data_dir.mkdir(parents=True, exist_ok=True)

        db_path = self._service_db_path(service_name)
        db = await aiosqlite.connect(str(db_path))
        try:
            if self._wal_mode:
                await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            db.row_factory = aiosqlite.Row

            emb_config = self._resolve_embedding_config(service_name)
            emb_provider = resolve_embedding_provider(emb_config)
            auto_embed = emb_config.auto_embed if emb_config else False

            store = SqliteEngravaCore(
                db,
                embedding_provider=emb_provider,
                auto_embed=auto_embed,
                search_config=self._default_search,
            )
            store._owns_connection = True  # noqa: SLF001
            await store.ensure_schema()

            await store._configure_vector_backend(  # noqa: SLF001
                backend_name=self._vector_backend,
                embedding_dimension=self._embedding_dimension,
            )
        except Exception:
            await db.close()
            raise

        logger.info("Initialized service %r: %s", service_name, db_path)
        return store

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
import contextlib
import logging
from typing import TYPE_CHECKING, Self

import aiosqlite

from engrava.config import (
    EmbeddingConfig,
    SearchConfig,
    ServicesConfig,
    resolve_embedding_provider,
)
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore, _close_quietly

if TYPE_CHECKING:
    from pathlib import Path


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
        from engrava.config_validation import (  # noqa: PLC0415 -- deferred to avoid a config import cycle
            require_exact_type_or_none,
            require_positive_int,
        )

        self._data_dir = data_dir
        # The manager retains these and reads them itself (``configs`` decides
        # which embedding config a service gets), so it requires the exact
        # class at its own boundary rather than relying on the store's.
        self._default_embeddings = require_exact_type_or_none(
            default_embeddings, EmbeddingConfig, "EngravaManager.default_embeddings"
        )
        self._default_search = require_exact_type_or_none(
            default_search, SearchConfig, "EngravaManager.default_search"
        )
        self._wal_mode = wal_mode
        self._vector_backend = vector_backend
        # Also a raw constructor argument: it reaches the vector index
        # declaration, so it is decoded here as well as at that boundary.
        self._embedding_dimension = require_positive_int(
            embedding_dimension, "EngravaManager.embedding_dimension"
        )
        self._services_config = require_exact_type_or_none(
            services_config, ServicesConfig, "EngravaManager.services_config"
        )
        self._stores: dict[str, SqliteEngravaCore] = {}
        # Tracks a creation in flight for a name that is not in ``_stores``
        # yet. ``self._lock`` guards both dicts, but only for the bookkeeping
        # steps (checking/registering/resolving an entry) -- never across the
        # slow I/O of ``_create_store`` or ``store.close()`` -- so creating
        # two *different* services stays genuinely concurrent, while
        # ``get_store``, ``delete_service`` and ``close_all`` can all see and
        # wait on a same-named creation that is still in flight.
        self._creating: dict[str, asyncio.Future[SqliteEngravaCore]] = {}
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

    async def get_store(self, service_name: str, *, migrate: bool = True) -> SqliteEngravaCore:
        """Get or create the store for a named service.

        The store is lazily initialized: the database file and schema are
        created on the first call.  Subsequent calls return the cached
        instance.

        Args:
            service_name: Unique service identifier.  Must match
                ``^[a-z][a-z0-9_-]{0,62}$``.
            migrate: When ``True`` (the default), a new store calls
                ``ensure_schema()`` as part of construction — a fresh service
                is bootstrapped and an existing behind one is brought
                current. Pass ``False`` to skip that call and open the
                database exactly as stored; used by a CLI command's
                schema-version gate for a read against an existing,
                previously-initialized service, where a warn-and-attempt on a
                behind schema must not itself perform an implicit migration.
                Ignored for a service whose store is already cached — the
                cached instance's schema state was decided by whichever call
                created it, not by this one.

        Returns:
            A fully initialized ``SqliteEngravaCore`` instance
            with its own database, schema, and embedding provider.

        Raises:
            ConfigError: If the service name is invalid.

        """
        from engrava.config import _validate_service_name  # noqa: PLC0415

        # Everything below addresses the service by the *validated* name, never
        # by the caller's object: it keys the store cache and, through
        # :meth:`_service_db_path`, names a file on disk.
        name = _validate_service_name(service_name)

        if name in self._stores:
            return self._stores[name]

        async with self._lock:
            # Double-check after acquiring the lock.
            if name in self._stores:
                return self._stores[name]
            # A concurrent call for this exact name is already creating it:
            # share its result instead of racing a second connection for the
            # same file. A call for a *different* name never reaches this
            # branch, so it is free to create concurrently -- the lock above
            # is only held long enough to check/register, not for the I/O
            # below.
            owns_creation = name not in self._creating
            if owns_creation:
                self._creating[name] = asyncio.get_running_loop().create_future()
            pending = self._creating[name]

        if not owns_creation:
            return await pending

        try:
            store = await self._create_store(name, migrate=migrate)
        except BaseException as exc:
            async with self._lock:
                del self._creating[name]
            pending.set_exception(exc)
            # Mark it retrieved on this side too: this coroutine re-raises
            # its own copy below regardless of whether another waiter ever
            # awaits ``pending``, and an exception set on a future that no
            # one retrieves is logged by asyncio's own future finalizer as
            # an unhandled error when it is garbage-collected.
            pending.exception()
            raise
        else:
            async with self._lock:
                self._stores[name] = store
                del self._creating[name]
            pending.set_result(store)
            return store

    async def peek_schema_version(self, service_name: str) -> int | None:
        """Read a service database's stamped ``user_version`` without migrating it.

        Returns ``None`` when the service has no database file yet — there is
        nothing to peek, and a fresh service is always bootstrapped to head by
        the ordinary :meth:`get_store` path. A plain ``PRAGMA`` read on a
        throwaway connection, never routed through ``ensure_schema``, so a
        caller applying its own schema-version gate can decide whether to
        warn, refuse, or proceed *before* anything about the database's schema
        changes.

        Args:
            service_name: Service identifier.

        Returns:
            The stamped ``user_version``, or ``None`` if the service database
            does not exist.

        Raises:
            ConfigError: If the service name is invalid.

        """
        from engrava.config import _validate_service_name  # noqa: PLC0415

        name = _validate_service_name(service_name)
        db_path = self._service_db_path(name)
        if not db_path.exists():
            return None
        conn = await aiosqlite.connect(str(db_path))
        try:
            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
        except BaseException:
            # Not ``except Exception``: a cancellation while the read is in
            # flight must still close ``conn``. The read failed (or was
            # cancelled) -- that is what the caller needs to see, so a
            # failure in this closing call is secondary and goes through
            # ``_close_quietly`` rather than replacing it -- see that
            # function's docstring.
            await _close_quietly(conn)
            raise
        else:
            # The read succeeded. A close failure here is not secondary to
            # anything -- it is the only error there is, so it must
            # propagate normally rather than being logged and swallowed by
            # ``_close_quietly``. An unconditional
            # ``finally: await _close_quietly(conn)`` would silently turn a
            # genuine close failure into a successful-looking read.
            await conn.close()
        return int(row[0]) if row else 0

    def service_exists(self, service_name: str) -> bool:
        """Check whether a service database file exists on disk.

        Does **not** create or open the database.

        Args:
            service_name: Service identifier.

        Returns:
            ``True`` if ``<data_dir>/<service_name>.db`` exists.

        Raises:
            ConfigError: If the service name is invalid.

        """
        from engrava.config import _validate_service_name  # noqa: PLC0415

        return self._service_db_path(_validate_service_name(service_name)).exists()

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

        The existence-check-then-unlink sequence runs under the same lock
        :meth:`get_store` uses to register a creation in flight, so this
        cannot unlink a database file out from under a same-named
        ``_create_store`` that is still writing its schema: if one is in
        flight when this is called, this method waits for it to land (or
        fail) before checking again, rather than treating "not yet cached"
        as "does not exist".

        Args:
            service_name: Name of the service to delete.

        Raises:
            ConfigError: If the service name is invalid.
            FileNotFoundError: If the database file does not exist.

        """
        from engrava.config import _validate_service_name  # noqa: PLC0415

        # The validated name is what addresses the file this method unlinks.
        # Building the path from the caller's object instead would let a ``str``
        # subclass name one thing to the pattern check and another to the
        # filesystem, deleting a file outside ``data_dir``.
        name = _validate_service_name(service_name)

        while True:
            async with self._lock:
                pending = self._creating.get(name)
                if pending is None:
                    # No creation for this name is in flight, so the cache
                    # and the filesystem are consistent with each other
                    # right now -- close (if cached) and unlink atomically
                    # with respect to a same-named ``get_store`` call, which
                    # cannot register a new creation until this lock is
                    # released.
                    if name in self._stores:
                        store = self._stores.pop(name)
                        store._owns_connection = True  # noqa: SLF001
                        await store.close()

                    db_path = self._service_db_path(name)
                    if not db_path.exists():
                        msg = f"Service database not found: {db_path}"
                        raise FileNotFoundError(msg)

                    db_path.unlink()
                    # Clean up WAL/SHM journal files.
                    for suffix in (".db-wal", ".db-shm"):
                        journal = db_path.with_suffix(suffix)
                        if journal.exists():
                            journal.unlink()

                    logger.info("Deleted service %r database: %s", name, db_path)
                    return

            # A creation for this exact name is still in flight. Wait for it
            # outside the lock -- ``get_store`` needs the lock itself to
            # record the result -- then loop back and recheck under a fresh
            # acquisition. A failed creation never reached the cache or the
            # filesystem, so its exception (already surfaced to its own
            # caller, deliberately not re-logged here) is nothing new to
            # report here -- this is only a synchronization wait, not a
            # second attempt at the work.
            with contextlib.suppress(Exception):
                await pending

    async def close_all(self) -> None:
        """Close all cached store connections.

        Safe to call multiple times.  After this call, every store that was
        either already cached or already being created when this method was
        called has been closed and dropped from the cache; ``get_store()``
        will create a fresh connection for those names afterward. A
        creation for a *different*, previously-unseen name that starts
        after this method has taken its snapshot is unaffected -- it is not
        this call's responsibility and is left running, exactly like any
        other ``get_store()`` call that overlaps a call to this method for
        an unrelated name.

        The store list is snapshotted under the lock and then closed
        outside it: holding the lock across every ``await store.close()``
        would serialize unrelated closes for no reason and turn one slow or
        hanging close into a bottleneck for all the others. What the
        snapshot captures is two things, not just the cache: the stores
        already in ``_stores``, and the creations already registered in
        ``_creating`` -- a creation still in flight when this method starts
        is waited for here before its store is closed, so a connection that
        was mid-creation at the moment of the call can never survive past
        this method returning. Iterating a snapshot list rather than the
        live cache also means a concurrent ``get_store()`` for another name
        inserting into ``_stores`` mid-loop cannot raise
        ``RuntimeError: dictionary changed size during iteration``.

        A cancellation while closing one store must not abandon the rest:
        ``store.close()`` completes the real, physical close before it
        re-raises a ``CancelledError`` from an in-flight cancellation (see
        its own docstring), so catching that here and continuing to the
        next store never skips a close that hasn't actually happened yet
        -- it only avoids abandoning every store *after* the one that was
        in flight when the cancellation landed. The cancellation itself is
        never swallowed: it is re-raised once every store has had its
        close attempt, not discarded and not left to interrupt the loop
        early.
        """
        async with self._lock:
            stores_to_close = list(self._stores.items())
            pending_creations = list(self._creating.items())

        pending_cancellation: asyncio.CancelledError | None = None

        for name, creating in pending_creations:
            try:
                store = await creating
            except asyncio.CancelledError as exc:
                pending_cancellation = exc
                continue
            except Exception:  # noqa: BLE001 -- one failed creation must not abort waiting for the rest
                # The creation failed and already reported that failure to
                # its own ``get_store()`` caller -- it never reached the
                # cache, so there is nothing here to close. Logged here too
                # (unlike delete_service's wait, which is a plain
                # synchronization point) because close_all is a shutdown
                # path where a swallowed failure would otherwise be
                # invisible.
                logger.warning(
                    "Error awaiting in-flight creation of service %r", name, exc_info=True
                )
                continue
            stores_to_close.append((name, store))

        for name, store in stores_to_close:
            try:
                store._owns_connection = True  # noqa: SLF001
                await store.close()
            except asyncio.CancelledError as exc:
                pending_cancellation = exc
            except Exception:  # noqa: BLE001 -- one store's close failure must not abort closing the rest
                logger.warning("Error closing store %r", name, exc_info=True)
            finally:
                async with self._lock:
                    # Only drop the entry this call actually closed. A name
                    # popped by a concurrent ``delete_service()`` is already
                    # gone; a name re-created after this method's snapshot
                    # belongs to that later creation, not to this close.
                    if self._stores.get(name) is store:
                        del self._stores[name]

        if pending_cancellation is not None:
            raise pending_cancellation

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
            **kwargs,  # type: ignore[arg-type]  # forwarded verbatim to the constructor, which types them
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _service_db_path(self, service_name: str) -> Path:
        """Compute the database file path for a service.

        The interpolation below runs ``type(service_name).__format__``, so the
        text that lands in the path is whatever that method returns. Every
        caller therefore passes a name already returned by
        ``_validate_service_name`` — an exact ``str`` — and never the object it
        received from its own caller.

        Args:
            service_name: Service identifier, already validated and owned by
                :func:`~engrava.config._validate_service_name`.

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

    async def _create_store(self, service_name: str, *, migrate: bool = True) -> SqliteEngravaCore:
        """Create and initialize a new store for a service.

        Creates the data directory and database file if needed, applies the
        schema (unless ``migrate`` is ``False``), and configures the
        embedding provider.

        Args:
            service_name: Service identifier.
            migrate: Forwarded from :meth:`get_store` — see its docstring.

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
            # synchronous=NORMAL is the documented-safe companion to WAL: the
            # database stays durable across an application crash and is only at
            # risk of losing the most recent transactions on an OS crash or
            # power loss, which is the standard recommendation for WAL.
            await db.execute("PRAGMA synchronous=NORMAL")
            # busy_timeout makes a second connection wait (up to 5s) for a lock
            # instead of failing immediately with SQLITE_BUSY.
            await db.execute("PRAGMA busy_timeout=5000")
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
            if migrate:
                await store.ensure_schema()

            await store._configure_vector_backend(  # noqa: SLF001
                backend_name=self._vector_backend,
                embedding_dimension=self._embedding_dimension,
            )
        except BaseException:
            # Not ``except Exception``: ``asyncio.CancelledError`` derives
            # from ``BaseException``, and a cancellation during any await
            # above (most likely at shutdown, exactly when things get
            # cancelled) must close ``db`` exactly like an ordinary failure
            # does — otherwise it leaks aiosqlite's non-daemon connection
            # worker thread just as an uncaught ``DatabaseError`` would.
            # Routed through ``_close_quietly`` rather than a direct
            # ``await db.close()`` so a failure in the close itself cannot
            # replace this exception -- see that function's docstring.
            await _close_quietly(db)
            raise

        logger.info("Initialized service %r: %s", service_name, db_path)
        return store

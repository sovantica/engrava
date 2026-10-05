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
    EngravaConfig,
    SearchConfig,
    ServicesConfig,
    resolve_embedding_provider,
)
from engrava.config_validation import ConfigError
from engrava.infrastructure.sqlite.aiosqlite_connect import connect
from engrava.infrastructure.sqlite.engrava_core import (
    _CLOSE_TIMEOUT_SECONDS,
    _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS,
    SqliteEngravaCore,
    _close_quietly,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol


logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel type for ``EngravaManager.__init__``'s ``default_embeddings`` default.

    Distinguishes "the caller did not pass ``default_embeddings`` at all" from
    an explicit ``None``, which means "no embedding provider for any service
    without its own override" and must *not* fall back to
    ``base_config.embeddings`` — see ``_resolve_embedding_config``. Module-private:
    no caller outside this module ever constructs or names an instance of it.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


class _CreationAbandonedError(Exception):
    """A store creation ended with no store because its creating call was cancelled.

    A cancelled creator resolves the future it shares with its waiters with
    this, not with its own ``CancelledError``. It re-raises that cancellation
    to its own caller only, so no waiter raises a cancellation nobody asked it
    to. The exception never leaves this module. A waiting ``get_store`` starts
    the creation over, ``delete_service`` checks again, and ``close_all``
    counts it as a creation that left nothing to close.
    """


class EngravaManager:
    """Manages per-service ``SqliteEngravaCore`` instances.

    Each service is a separate SQLite database file (``<name>.db``)
    under ``data_dir``.  Stores are lazily initialized on first
    ``get_store()`` call and cached for subsequent access.

    Args:
        data_dir: Directory for per-service database files.
        default_embeddings: Fallback embedding config for a service without
            its own override (see ``services_config``). Left unset (the
            default), it falls back to ``base_config.embeddings`` when
            ``base_config`` is set, and to ``None`` otherwise — the same
            fallback whether the manager is constructed directly or through
            :meth:`from_config`. Pass an explicit value, including ``None``,
            to override that fallback: an explicit ``None`` means no provider
            for any service without its own override even when
            ``base_config`` carries its own ``embeddings`` — how the CLI's
            ``restore`` suppresses embedding regeneration without
            ``--re-embed``.
        default_search: Default hybrid-search weights. Ignored when
            ``base_config`` is set, which carries its own ``search`` weights.
        wal_mode: Enable WAL journal mode. Ignored when ``base_config`` is
            set, which carries its own ``wal_mode``.
        vector_backend: Vector backend name (``"numpy"`` or ``"sqlite-vec"``).
            Ignored when ``base_config`` is set, which carries its own
            ``vector_backend``.
        embedding_dimension: Default embedding vector dimension. Ignored when
            ``base_config`` is set, which carries its own
            ``embedding_dimension``.
        services_config: Optional ``ServicesConfig`` for per-service overrides.
        base_config: Optional full ``EngravaConfig`` to apply to every store
            this manager builds — the journal, hygiene policy, TTL, metrics,
            extension manifests, access tracking, derive gates, dreaming,
            hooks and ``require_embedding``, none of which reach a store
            built without it. When set, each per-service store is built the
            same way :meth:`SqliteEngravaCore.from_config` builds one, except
            that the database path is this service's own file (never
            ``base_config.database_path``) and the embedding config is
            resolved per service (see ``_resolve_embedding_config``) rather
            than taken from ``base_config.embeddings`` directly. ``None``
            (the default) keeps a service's store exactly as before —
            PRAGMAs, embeddings and the vector backend only.
        embedding_provider_factory: Optional factory called with a service
            name to supply that service's embedding provider object
            directly, instead of one resolved from configuration. Takes
            precedence over the resolved embedding configuration when it
            returns a provider; returning ``None`` falls back to the
            resolved embedding configuration, the same on both the
            ``base_config`` path and the plain per-service path.

    """

    def __init__(
        self,
        data_dir: Path,
        *,
        default_embeddings: EmbeddingConfig | _Unset | None = _UNSET,
        default_search: SearchConfig | None = None,
        wal_mode: bool = True,
        vector_backend: str = "numpy",
        embedding_dimension: int = 384,
        services_config: ServicesConfig | None = None,
        base_config: EngravaConfig | None = None,
        embedding_provider_factory: Callable[[str], EmbeddingProviderProtocol | None] | None = None,
    ) -> None:
        from engrava.config_validation import (  # noqa: PLC0415 -- deferred to avoid a config import cycle
            require_exact_type_or_none,
            require_positive_int,
        )

        self._data_dir = data_dir
        # The manager retains these and reads them itself (``configs`` decides
        # which embedding config a service gets), so it requires the exact
        # class at its own boundary rather than relying on the store's. The
        # sentinel default is never owned through that boundary: it is not a
        # config object, and ``_resolve_embedding_config`` checks for it with
        # ``isinstance`` before this value could ever reach ``base_config``'s
        # own embeddings.
        self._default_embeddings: EmbeddingConfig | _Unset | None = (
            default_embeddings
            if isinstance(default_embeddings, _Unset)
            else require_exact_type_or_none(
                default_embeddings, EmbeddingConfig, "EngravaManager.default_embeddings"
            )
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
        # The full configuration a per-service store is built from when set
        # (see ``_create_configured_store``). ``None`` keeps every existing
        # caller's behaviour byte-for-byte: ``_create_store`` only takes that
        # branch when this is set.
        self._base_config = require_exact_type_or_none(
            base_config, EngravaConfig, "EngravaManager.base_config"
        )
        # A live callable, not a configuration value — stored verbatim, like
        # ``SqliteEngravaCore``'s own ``cycle_provider``, rather than owned
        # through a config-object boundary.
        self._embedding_provider_factory = embedding_provider_factory
        self._stores: dict[str, SqliteEngravaCore] = {}
        # Tracks a creation in flight for a name that is not in ``_stores``
        # yet. ``self._lock`` guards the bookkeeping on both dicts
        # (checking, registering and publishing an entry), never the slow
        # I/O of ``_create_store`` or ``close_all``'s closes, so creating two
        # *different* services stays genuinely concurrent, while
        # ``get_store``, ``delete_service`` and ``close_all`` can all see and
        # wait on a same-named creation that is still in flight. Only
        # ``delete_service`` holds it across a close and an unlink, so that
        # no creation of that name can register while its file is removed.
        #
        # Only the call that registered an entry (its creator) removes it or
        # resolves its future. It does both together, on every way out of
        # the creation: under the lock when it publishes the store, and
        # without waiting for the lock when the creation ends without one
        # (see ``_abandon_creation``). Every other caller awaits the future
        # through ``asyncio.shield``, so a waiter's cancellation never
        # cancels it. A creator that is cancelled resolves it with
        # ``_CreationAbandonedError``, not with its own ``CancelledError``.
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

        Concurrent calls for the same name share one creation: the first
        call creates the store and the others wait for it. Cancelling a call
        affects that call alone. A cancelled waiter stops waiting, and the
        creation carries on for everyone else. If the creating call is
        cancelled before it has published a store, that creation produces
        no store: the creating call raises its own ``CancelledError``, and
        each waiting call starts over, becoming the creator itself unless
        another call has already started a new creation. An ordinary failure
        of the creation is raised to every call waiting on it.

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

        while True:
            if name in self._stores:
                return self._stores[name]

            async with self._lock:
                # Double-check after acquiring the lock.
                if name in self._stores:
                    return self._stores[name]
                # A concurrent call for this exact name is already creating
                # it: share its result instead of racing a second connection
                # for the same file. A call for a *different* name never
                # reaches this branch, so it is free to create concurrently
                # -- the lock above is only held long enough to
                # check/register, not for the I/O below.
                pending = self._creating.get(name)
                if pending is None:
                    pending = asyncio.get_running_loop().create_future()
                    self._creating[name] = pending
                    break

            try:
                # Shielded: cancelling this call ends this call's wait and
                # nothing else. The creation it shares with its creator and
                # every other waiter carries on.
                return await asyncio.shield(pending)
            except _CreationAbandonedError:
                # The creating call was cancelled before it had a store, and
                # it has already unregistered that creation. Start over:
                # return a store cached in the meantime, wait on a creation
                # registered in the meantime, or register one and create it.
                # Each pass waits on a different creation, so this loop
                # cannot spin on one abandoned creation.
                continue

        return await self._create_registered_store(name, pending, migrate=migrate)

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
        conn = await connect(str(db_path))
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

        Cancelling this call ends only its own wait. The creation it was
        waiting on carries on for its other callers. A creation abandoned
        because its creating call was cancelled is one more outcome here,
        and this method checks again. A same-named ``get_store()`` call that
        was waiting on that abandoned creation starts a new one, which this
        method treats like any other ``get_store()`` call for that name: it
        waits for that creation if it is registered by the time this method
        checks again, and otherwise the retry runs after this method's check
        and is not this method's concern. This method does not chase a
        retry, because one that has not registered yet cannot be seen
        without a race.

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
            # second attempt at the work. An abandoned creation's
            # ``_CreationAbandonedError`` is such an exception too. The wait
            # is shielded so that cancelling this call never cancels the
            # creation it waits on.
            with contextlib.suppress(Exception):
                await asyncio.shield(pending)

    async def close_all(self) -> None:
        """Close all cached store connections.

        Safe to call multiple times.  After this call, every store that was
        either already cached or already being created when this method took
        its snapshot has had its close attempted and has been dropped from
        the cache; ``get_store()`` will create a fresh connection for those
        names afterward. A close that fails is logged, not raised, and does
        not stop the others. A creation that starts after the snapshot is
        unaffected, whether its name is a different one or one of those same
        names -- it is not this call's responsibility and is left running,
        exactly like any other ``get_store()`` call that overlaps a call to
        this method.

        That includes a waiting ``get_store()`` call's retry. A creation
        whose creating call is cancelled before it publishes a store is
        abandoned. That call finishes its close attempt on whatever the
        creation opened before it tells its waiters the creation is over, so
        there is nothing left here to close. Its
        cancellation belongs to that call alone: this method neither
        re-raises it nor raises a ``CancelledError`` of its own for it. A
        ``get_store()`` call that was waiting on the abandoned creation then
        starts a new one, after the abandoned one and so after this
        method's snapshot. This method does not chase that retry, because
        one that has not registered yet cannot be seen without a race.

        The store list is snapshotted under the lock and then closed
        outside it: holding the lock across every ``await store.close()``
        would serialize unrelated closes for no reason and turn one slow or
        hanging close into a bottleneck for all the others. What the
        snapshot captures is two things, not just the cache: the stores
        already in ``_stores``, and the creations already registered in
        ``_creating`` -- a creation still in flight when this method starts
        is waited for here, and its store's close attempted, before this
        method returns. Iterating a snapshot list rather than the
        live cache also means a concurrent ``get_store()`` for another name
        inserting into ``_stores`` mid-loop cannot raise
        ``RuntimeError: dictionary changed size during iteration``.

        A cancellation while closing one store must not abandon the rest:
        ``store.close()`` waits for its physical close, up to its own bound,
        before it re-raises a ``CancelledError`` from an in-flight
        cancellation (see its own docstring). The loop catches a
        ``CancelledError`` from ``store.close()``, keeps the last one it
        caught, and raises it after the loop.

        A cancellation while this method waits on a creation from its
        snapshot is handled the same way. It keeps waiting until that
        creation finishes, attempts the close of the store it produced and
        drops it from the cache, and re-raises the cancellation at the end.
        Stopping early would skip the close of a store this method is
        responsible for.
        """
        async with self._lock:
            stores_to_close = list(self._stores.items())
            pending_creations = list(self._creating.items())

        pending_cancellation: asyncio.CancelledError | None = None

        for name, creating in pending_creations:
            cancelled_while_waiting = await self._wait_out_creation(creating)
            if cancelled_while_waiting is not None:
                pending_cancellation = cancelled_while_waiting
            try:
                store = creating.result()
            except (_CreationAbandonedError, asyncio.CancelledError):
                # The creating call was cancelled before it had a store:
                # nothing to close. That cancellation is the creating call's
                # own and is raised to that call's caller, not here.
                # ``result()`` does not suspend, so a ``CancelledError``
                # from it is never this call's own either. It could only
                # mean the creation's future was cancelled, which nothing in
                # this class does, and it is handled the same way.
                continue
            except Exception:  # one failed creation must not abort waiting for the rest
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
            except Exception:  # one store's close failure must not abort closing the rest
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
    async def from_config(
        cls, config: ServicesConfig | EngravaConfig, **kwargs: object
    ) -> EngravaManager:
        """Create a manager from a ``ServicesConfig`` or a full ``EngravaConfig``.

        Args:
            config: Either a ``ServicesConfig`` (today's behaviour, unchanged:
                only per-service embedding overrides reach a built store), or
                a full ``EngravaConfig`` whose ``services`` is set. The latter
                is passed through as this manager's ``base_config``, so every
                store it builds gets the same wiring
                :meth:`SqliteEngravaCore.from_config` gives one — see
                ``base_config`` on the constructor.
            **kwargs: Additional keyword arguments forwarded to the
                constructor (``default_embeddings``, ``wal_mode``, etc.).

        Returns:
            A new ``EngravaManager`` instance.

        Raises:
            ConfigError: If ``config`` is an ``EngravaConfig`` whose
                ``services`` is ``None`` — there is no ``data_dir`` to build
                per-service stores under.

        """
        if isinstance(config, EngravaConfig):
            if config.services is None:
                msg = (
                    "EngravaConfig.services must be set to build an EngravaManager "
                    "from a full EngravaConfig"
                )
                raise ConfigError(msg)
            # No ``default_embeddings`` seeding needed here: left out of
            # ``**kwargs``, the constructor's sentinel default makes
            # ``_resolve_embedding_config`` fall back to
            # ``base_config.embeddings`` itself — the same ``config`` passed
            # below. An explicit ``default_embeddings`` in ``**kwargs`` —
            # including ``None`` — still wins, forwarded verbatim like any
            # other constructor keyword this method passes through.
            return cls(
                data_dir=config.services.data_dir,
                services_config=config.services,
                base_config=config,
                **kwargs,  # type: ignore[arg-type]  # forwarded verbatim to the constructor, which types them
            )
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

        Resolution order:

        1. the service's own override in ``services_config``, when present;
        2. an explicitly passed ``default_embeddings`` — including an
           explicit ``None``, which means no provider for this service and
           wins even when ``base_config`` carries its own ``embeddings``
           (the CLI's ``restore`` relies on this to suppress embedding
           regeneration when ``--re-embed`` is not requested);
        3. when ``default_embeddings`` was left unset and ``base_config`` is
           set, ``base_config.embeddings``;
        4. otherwise ``None``.

        This order is the same whether the manager was built directly or
        through :meth:`from_config` — the sentinel default on
        ``default_embeddings`` is what makes step 3 possible without
        ``from_config`` having to seed anything itself.

        Args:
            service_name: Service identifier.

        Returns:
            Merged ``EmbeddingConfig``, or ``None``.

        """
        if self._services_config and service_name in self._services_config.configs:
            svc_cfg = self._services_config.configs[service_name]
            if svc_cfg.embeddings is not None:
                return svc_cfg.embeddings
        if isinstance(self._default_embeddings, _Unset):
            return self._base_config.embeddings if self._base_config is not None else None
        return self._default_embeddings

    async def _create_store(self, service_name: str, *, migrate: bool = True) -> SqliteEngravaCore:
        """Create and initialize a new store for a service.

        Creates the data directory and database file if needed, applies the
        schema (unless ``migrate`` is ``False``), and configures the
        embedding provider. When ``base_config`` is set on this manager, the
        store is instead built with that configuration's full wiring — see
        :meth:`_create_configured_store`; everything below this point is the
        unchanged, PRAGMAs-and-embeddings-only path a manager without
        ``base_config`` has always used.

        Args:
            service_name: Service identifier.
            migrate: Forwarded from :meth:`get_store` — see its docstring.

        Returns:
            Fully initialized ``SqliteEngravaCore``.

        """
        self._data_dir.mkdir(parents=True, exist_ok=True)

        db_path = self._service_db_path(service_name)

        if self._base_config is not None:
            store = await self._create_configured_store(service_name, db_path, migrate=migrate)
            logger.info("Initialized service %r: %s", service_name, db_path)
            return store

        db = await connect(str(db_path))
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
            # A set factory takes precedence over the resolved configuration,
            # but only when it actually returns a provider: a factory that
            # returns None falls back to resolving one from emb_config, the
            # same as an unset factory -- mirroring how
            # SqliteEngravaCore._build_configured_store treats its own
            # embedding_provider argument.
            emb_provider = (
                self._embedding_provider_factory(service_name)
                if self._embedding_provider_factory is not None
                else None
            )
            if emb_provider is None:
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

    async def _create_configured_store(
        self, service_name: str, db_path: Path, *, migrate: bool
    ) -> SqliteEngravaCore:
        """Build a per-service store with the manager's full ``base_config`` wiring.

        Delegates to :meth:`SqliteEngravaCore._build_configured_store` — the
        same internal builder :meth:`SqliteEngravaCore.from_config` uses — so
        a manager-built service gets the journal, hygiene policy, TTL, hooks,
        dreaming and ``require_embedding`` settings ``base_config`` carries,
        not just the PRAGMAs and the vector backend :meth:`_create_store`'s
        other branch configures. Only called when ``self._base_config`` is
        set; the ``None`` guard below exists purely so the type checker can
        narrow it, since that is already guaranteed by the only caller.

        Args:
            service_name: Service identifier, used to resolve a per-service
                ``embeddings`` override and, when set, to call
                ``embedding_provider_factory``.
            db_path: This service's own database file. ``base_config``'s own
                ``database_path`` is never read — the manager always
                allocates one file per service.
            migrate: Forwarded from :meth:`get_store` — see its docstring and
                :meth:`SqliteEngravaCore._build_configured_store`'s.

        Returns:
            Fully initialized ``SqliteEngravaCore``.

        """
        base_config = self._base_config
        if base_config is None:
            msg = "_create_configured_store requires EngravaManager.base_config to be set"
            raise ConfigError(msg)  # pragma: no cover -- guarded by the only caller

        emb_config = self._resolve_embedding_config(service_name)
        embedding_provider = (
            self._embedding_provider_factory(service_name)
            if self._embedding_provider_factory is not None
            else None
        )
        return await SqliteEngravaCore._build_configured_store(  # noqa: SLF001
            base_config,
            db_path,
            migrate=migrate,
            embeddings=emb_config,
            embedding_provider=embedding_provider,
            cycle_provider=None,
            write_lock_acquire_timeout_seconds=_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS,
            close_timeout_seconds=_CLOSE_TIMEOUT_SECONDS,
        )

    async def _create_registered_store(
        self,
        name: str,
        pending: asyncio.Future[SqliteEngravaCore],
        *,
        migrate: bool,
    ) -> SqliteEngravaCore:
        """Create the store this call registered as ``pending``, then settle that registration.

        Every way out of this method removes the ``_creating`` entry and
        resolves ``pending`` exactly once: with the store once it is
        published, with the failure if creating or publishing it failed, and
        with :class:`_CreationAbandonedError` if this call was cancelled
        first. When the creation ends without publishing a store, its
        cleanup -- a close attempt on whatever it opened -- finishes before
        ``pending`` is resolved. A close that itself fails is logged, as
        ``close_all`` logs its own close failures, and is not raised in
        place of the original exception.

        Args:
            name: The validated service name ``pending`` is registered under.
            pending: The future this call registered in ``_creating``.
            migrate: Forwarded from :meth:`get_store` -- see its docstring.

        Returns:
            The created, published store.

        """
        try:
            store = await self._create_store(name, migrate=migrate)
        except BaseException as exc:
            # ``_create_store`` has already made its close attempt on its own
            # connection, through ``_close_quietly``, before anything
            # propagates out of it, a cancellation included. So ``pending``
            # is resolved only after that attempt. ``_close_quietly``'s
            # docstring covers how it logs a failed close and the one
            # repeated-cancellation window it leaves.
            self._abandon_creation(name, pending, exc)
            raise

        # Waiting for the lock is the one suspension point between
        # ``_create_store`` returning and ``pending`` being resolved, so it is
        # the one place a cancellation can land in between. The wait is
        # wrapped rather than avoided. Whatever interrupts it first makes the
        # close attempt on the unpublished store and only then unregisters
        # the creation and resolves ``pending``, so no waiter -- ``close_all``
        # in particular -- learns the creation is over before that attempt
        # has finished. The ``finally`` settles the creation even if that
        # close is itself interrupted. Once the lock is held, publishing is
        # synchronous dict work that nothing can interrupt.
        try:
            await self._lock.acquire()
        except BaseException as exc:
            try:
                await self._close_unpublished_store(name, store)
            finally:
                self._abandon_creation(name, pending, exc)
            raise
        try:
            self._stores[name] = store
            del self._creating[name]
            if not pending.done():
                pending.set_result(store)
        finally:
            self._lock.release()
        return store

    def _abandon_creation(
        self,
        name: str,
        pending: asyncio.Future[SqliteEngravaCore],
        exc: BaseException,
    ) -> None:
        """Unregister a creation that produced no store, and resolve its future with why.

        A cancellation belongs to the creator, which re-raises it to its own
        caller. Its waiters get :class:`_CreationAbandonedError` instead.
        Any other exception reaches them unchanged. The creator calls this
        only after its close attempt on whatever the creation opened has
        finished.

        This method is synchronous on purpose. Nothing in it can suspend, so
        no cancellation can separate unregistering the creation from
        resolving its future, or prevent either. For the same reason it does
        not wait for ``self._lock``, since that wait would be a suspension
        point. It does not need the lock either. The lock keeps one holder's
        check-and-update of the two dicts from interleaving with another's,
        and a step with no await in it cannot interleave with anything. The
        one holder that suspends under the lock, ``delete_service`` across
        its close and unlink, has already found no creation of its name
        registered and keeps any from registering, so it has no entry for
        this method to remove.

        Args:
            name: The validated service name ``pending`` is registered under.
            pending: The future the creator registered in ``_creating``.
            exc: What ended the creation.

        """
        del self._creating[name]
        if pending.done():
            return
        if isinstance(exc, asyncio.CancelledError):
            pending.set_exception(_CreationAbandonedError())
        else:
            pending.set_exception(exc)
        # Mark it retrieved on this side too: the creator re-raises its own
        # exception whether or not any waiter ever awaits ``pending``, and an
        # exception set on a future that no one retrieves is logged by
        # asyncio's own future finalizer as an unhandled error when it is
        # garbage-collected.
        pending.exception()

    @staticmethod
    async def _wait_out_creation(
        creating: asyncio.Future[SqliteEngravaCore],
    ) -> asyncio.CancelledError | None:
        """Wait until *creating* is done, however often the calling task is cancelled meanwhile.

        The wait is shielded, so cancelling the caller never cancels the
        creation. The caller's own cancellation is recorded, and the wait
        goes on. The loop ends as soon as the creation is done, whatever its
        outcome, so it never spins on a finished future. The caller reads
        that outcome from the future itself.

        Args:
            creating: A creation's shared future, from ``_creating``.

        Returns:
            The last cancellation of the caller seen while waiting, for the
            caller to re-raise once its own work is done, or ``None``.

        """
        cancelled: asyncio.CancelledError | None = None
        while not creating.done():
            try:
                with contextlib.suppress(Exception):  # the outcome, read by the caller
                    await asyncio.shield(creating)
            except asyncio.CancelledError as exc:
                # Nothing in this class cancels a creation's future, but if
                # something did, the ``CancelledError`` that ends this wait
                # would be the creation's, not the caller's.
                if not creating.cancelled():
                    cancelled = exc
        return cancelled

    @staticmethod
    async def _close_unpublished_store(name: str, store: SqliteEngravaCore) -> None:
        """Close a store its creator built but never published.

        This runs while the exception that stopped the publication is
        already propagating, and that exception is the one to report. An
        ordinary failure of this close is therefore logged, not raised in its
        place. A further cancellation does not cancel the physical close:
        ``store.close()`` waits for it, up to its own bound, before it
        re-raises the cancellation (see its docstring).

        Args:
            name: The service the store was built for.
            store: The unpublished store.

        """
        try:
            await store.close()
        except Exception:  # the exception that stopped the publication is what propagates
            logger.warning("Error closing unpublished store for service %r", name, exc_info=True)

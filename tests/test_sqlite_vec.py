"""Tests for engrava.extensions.vector_sqlite_vec."""

from __future__ import annotations

import importlib.util
import sqlite3
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiosqlite
import pytest

from engrava import EmbeddingProviderContractError
from engrava.config import ConfigError, EngravaConfig
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.vector_sqlite_vec import (
    SqliteVecSearchBackend,
    _load_sqlite_vec_sync,
    load_sqlite_vec,
)
from engrava.infrastructure.service_manager import EngravaManager
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

# Skip the real-extension integration tests when sqlite-vec is absent, but
# never let them silently pass when it is installed and broken.
sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)

# ------------------------------------------------------------------
# SqliteVecSearchBackend unit tests (mocked DB)
# ------------------------------------------------------------------


class TestSqliteVecSearchBackend:
    def test_dimension_property(self) -> None:
        backend = SqliteVecSearchBackend(dimension=384)
        assert backend.dimension == 384

    async def test_ensure_index(self) -> None:
        db = AsyncMock()
        backend = SqliteVecSearchBackend(dimension=128)
        await backend.ensure_index(db)
        db.execute.assert_called_once()
        call_sql = db.execute.call_args[0][0]
        assert "embedding_vec" in call_sql
        assert "float[128]" in call_sql
        assert "distance_metric=cosine" in call_sql
        db.commit.assert_awaited_once()

    async def test_sync_embeddings_empty(self) -> None:
        cursor = AsyncMock()
        cursor.fetchall.return_value = []
        # The reconcile prune reuses the same mocked cursor; give it a real
        # ``rowcount`` (0 orphans) so the "any work?" check is a plain int.
        cursor.rowcount = 0
        db = AsyncMock()
        db.execute.return_value = cursor

        backend = SqliteVecSearchBackend(dimension=4)
        count = await backend.sync_embeddings(db)
        assert count == 0

    async def test_sync_embeddings_backfills(self) -> None:
        vec = [0.1, 0.2, 0.3, 0.4]
        blob = struct.pack("4f", *vec)
        row = {"rowid": 1, "dimension": 4, "vector_blob": blob}

        cursor_select = AsyncMock()
        cursor_select.fetchall.return_value = [row]
        # The reconcile prune reuses the same mocked cursor; 0 orphans pruned.
        cursor_select.rowcount = 0

        db = AsyncMock()
        db.execute.return_value = cursor_select

        backend = SqliteVecSearchBackend(dimension=4)
        count = await backend.sync_embeddings(db)
        assert count == 1
        db.commit.assert_awaited_once()

    async def test_search_empty(self) -> None:
        cursor = AsyncMock()
        cursor.fetchall.return_value = []
        db = AsyncMock()
        db.execute.return_value = cursor

        backend = SqliteVecSearchBackend(dimension=4)
        results = await backend.search(db, [0.1, 0.2, 0.3, 0.4])
        assert results == []

    async def test_search_returns_results(self) -> None:
        vec_cursor = AsyncMock()
        vec_cursor.fetchall.return_value = [
            {"rowid": 1, "distance": 0.5},
            {"rowid": 2, "distance": 1.0},
        ]
        id_cursor = AsyncMock()
        id_cursor.fetchall.return_value = [
            {"rowid": 1, "owner_id": "t-aaa"},
            {"rowid": 2, "owner_id": "t-bbb"},
        ]

        call_count = 0

        async def mock_execute(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return vec_cursor
            return id_cursor

        db = AsyncMock()
        db.execute = mock_execute

        backend = SqliteVecSearchBackend(dimension=4)
        results = await backend.search(db, [0.1, 0.2, 0.3, 0.4], top_k=10)
        assert len(results) == 2
        # First result should have higher similarity (smaller distance)
        assert results[0][0] == "t-aaa"
        assert results[0][1] > results[1][1]

    async def test_search_threshold_filters(self) -> None:
        vec_cursor = AsyncMock()
        # cosine distance=0.8 → similarity = 1 - 0.8 = 0.2, below threshold 0.5
        vec_cursor.fetchall.return_value = [
            {"rowid": 1, "distance": 0.8},
        ]
        id_cursor = AsyncMock()
        id_cursor.fetchall.return_value = [
            {"rowid": 1, "owner_id": "t-aaa"},
        ]

        call_count = 0

        async def mock_execute(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return vec_cursor
            return id_cursor

        db = AsyncMock()
        db.execute = mock_execute

        backend = SqliteVecSearchBackend(dimension=4)
        results = await backend.search(db, [0.1, 0.2, 0.3, 0.4], threshold=0.5)
        assert results == []


# ------------------------------------------------------------------
# load_sqlite_vec
# ------------------------------------------------------------------


class TestLoadSqliteVec:
    async def test_import_error_returns_false(self) -> None:
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            db = AsyncMock()
            result = await load_sqlite_vec(db)
            assert result is False
            # Import fails before any worker-thread dispatch.
            db._execute.assert_not_called()

    async def test_programming_error_returns_false(self) -> None:
        """A same-thread ProgrammingError degrades to numpy, not a crash."""
        mock_vec = MagicMock()
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            db = AsyncMock()
            db._execute.side_effect = sqlite3.ProgrammingError("wrong thread")
            result = await load_sqlite_vec(db)
            assert result is False

    async def test_os_error_returns_false(self) -> None:
        mock_vec = MagicMock()
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            db = AsyncMock()
            db._execute.side_effect = OSError("cannot open shared object")
            result = await load_sqlite_vec(db)
            assert result is False

    async def test_successful_load_runs_on_worker_thread(self) -> None:
        """A successful load dispatches the sync loader via ``_execute``.

        ``_execute`` is aiosqlite's worker-thread execution primitive, so
        routing the load through it guarantees the extension is loaded on
        the thread that owns the connection.
        """
        mock_vec = MagicMock()
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            db = AsyncMock()
            result = await load_sqlite_vec(db)
            assert result is True
            db._execute.assert_awaited_once()
            # First positional arg is the sync loader, second is the raw conn.
            call_args = db._execute.await_args
            assert call_args.args[0] is _load_sqlite_vec_sync
            assert call_args.args[1] is db._conn

    def test_load_sync_helper_invokes_extension_api(self) -> None:
        """The sync helper enables loading, calls sqlite_vec.load, then disables."""
        mock_vec = MagicMock()
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            raw_conn = MagicMock()
            _load_sqlite_vec_sync(raw_conn)
            assert raw_conn.enable_load_extension.call_args_list == [
                call(True),
                call(False),
            ]
            mock_vec.load.assert_called_once_with(raw_conn)

    def test_load_sync_helper_disables_extension_loading_on_load_failure(self) -> None:
        """When sqlite_vec.load raises, extension loading is still re-disabled.

        Guards the ``finally`` invariant: once ``enable_load_extension(True)``
        has succeeded the connection must never be left with extension loading
        enabled, even if the load itself fails. The exception still propagates
        so the caller can fall back to numpy.
        """
        mock_vec = MagicMock()
        mock_vec.load.side_effect = sqlite3.OperationalError("load failed")
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            raw_conn = MagicMock()
            with pytest.raises(sqlite3.OperationalError, match="load failed"):
                _load_sqlite_vec_sync(raw_conn)
            # enable(True) then enable(False) — the finally ran despite the raise.
            assert raw_conn.enable_load_extension.call_args_list == [
                call(True),
                call(False),
            ]
            mock_vec.load.assert_called_once_with(raw_conn)


# ------------------------------------------------------------------
# upsert_embedding unit tests
# ------------------------------------------------------------------


class TestUpsertEmbedding:
    async def test_upsert_embedding_calls_delete_and_insert(self) -> None:
        db = AsyncMock()
        backend = SqliteVecSearchBackend(dimension=3)
        await backend.upsert_embedding(db, rowid=42, vector=[0.1, 0.2, 0.3])

        # Expect two execute calls: DELETE + INSERT
        assert db.execute.await_count == 2
        delete_sql = db.execute.call_args_list[0][0][0]
        assert "DELETE FROM embedding_vec" in delete_sql
        insert_sql = db.execute.call_args_list[1][0][0]
        assert "INSERT INTO embedding_vec" in insert_sql

    async def test_upsert_passes_correct_rowid(self) -> None:
        db = AsyncMock()
        backend = SqliteVecSearchBackend(dimension=2)
        await backend.upsert_embedding(db, rowid=99, vector=[1.0, 2.0])

        insert_args = db.execute.call_args_list[1][0][1]
        assert insert_args[0] == 99
        assert "[1.0,2.0]" in insert_args[1]


# ------------------------------------------------------------------
# search cosine similarity conversion
# ------------------------------------------------------------------


class TestSearchCosineConversion:
    async def test_cosine_similarity_formula(self) -> None:
        """Verify similarity = 1 - distance (cosine metric)."""
        vec_cursor = AsyncMock()
        vec_cursor.fetchall.return_value = [
            {"rowid": 1, "distance": 0.2},
        ]
        id_cursor = AsyncMock()
        id_cursor.fetchall.return_value = [
            {"rowid": 1, "owner_id": "t-cos"},
        ]

        call_count = 0

        async def mock_execute(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return vec_cursor
            return id_cursor

        db = AsyncMock()
        db.execute = mock_execute

        backend = SqliteVecSearchBackend(dimension=4)
        results = await backend.search(db, [0.1, 0.2, 0.3, 0.4])
        assert len(results) == 1
        assert results[0][0] == "t-cos"
        # distance=0.2 → similarity = 1 - 0.2 = 0.8
        assert abs(results[0][1] - 0.8) < 1e-9


# ------------------------------------------------------------------
# Config embedding_dimension tests
# ------------------------------------------------------------------


class TestConfigEmbeddingDimension:
    def test_default_dimension(self) -> None:
        from engrava.config import EngravaConfig

        cfg = EngravaConfig(database_path=Path("test.db"))
        assert cfg.embedding_dimension == 384

    def test_custom_dimension_from_yaml(self, tmp_path: Path) -> None:
        from engrava.config import load_config

        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            "database:\n  path: test.db\nextensions:\n  vector:\n    dimension: 768\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.embedding_dimension == 768

    def test_invalid_dimension_raises(self, tmp_path: Path) -> None:
        from engrava.config import ConfigError, load_config

        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            "database:\n  path: test.db\nextensions:\n  vector:\n    dimension: -1\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="positive integer"):
            load_config(cfg_file)


# ------------------------------------------------------------------
# Real sqlite-vec extension integration (requires the package + a
# real aiosqlite connection — exercises the worker-thread extension
# load that mocked connections cannot cover).
# ------------------------------------------------------------------


_PARITY_MODEL = "test-fixture-model"


async def _make_thought(
    store: SqliteEngravaCore,
    thought_id: str,
    *,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    lifecycle_status: LifecycleStatus = LifecycleStatus.CREATED,
    expires_at: str | None = None,
) -> None:
    """Create a minimal thought so embeddings satisfy the FK to ``thought``.

    The optional ``thought_type`` / ``lifecycle_status`` / ``expires_at`` let a
    test seed non-live rows (expired TTL, or a retired REFLECTION) so the
    live-row post-filter has something to drop.
    """
    thought = ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=f"essence {thought_id}",
        content=f"content {thought_id}",
        priority=Priority.P3,
        lifecycle_status=lifecycle_status,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        expires_at=expires_at,
    )
    await store.create_thought(thought)


def _past_iso() -> str:
    """Return an ISO-8601 UTC timestamp safely in the past (already expired)."""
    import datetime as _dt

    return (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=1)).isoformat()


@sqlite_vec_required
class TestSqliteVecRealConnection:
    """End-to-end tests against a real loaded sqlite-vec extension."""

    async def _build_store(
        self,
        tmp_path: Path,
        *,
        backend: str,
        dimension: int,
        search_config: object | None = None,
        db_name: str | None = None,
        ttl_strategy: str = "archive",
    ) -> SqliteEngravaCore:
        """Construct a store with a real connection and the given backend.

        Mirrors what ``from_config`` does internally (schema bootstrap then
        ``_configure_vector_backend``) but lets the test pick the backend.
        ``search_config`` threads a custom :class:`SearchConfig` (e.g. a
        ``vec0_overfetch_factor`` override); ``db_name`` disambiguates the file
        when a single test builds two stores of the same backend;
        ``ttl_strategy`` selects the cleanup strategy so a test can exercise
        the physical-delete TTL path against a real vec0 index.
        """
        from engrava.config import SearchConfig

        db_path = tmp_path / f"{db_name or backend}.db"
        db = await aiosqlite.connect(str(db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        cfg = search_config if isinstance(search_config, SearchConfig) else None
        store = SqliteEngravaCore(db, search_config=cfg, ttl_strategy=ttl_strategy)
        store._owns_connection = True
        await store.ensure_schema()
        await store._configure_vector_backend(
            backend_name=backend,
            embedding_dimension=dimension,
        )
        return store

    async def test_sqlite_vec_backend_constructs_and_searches(self, tmp_path: Path) -> None:
        """A real sqlite-vec backend loads, indexes, and returns ranked results.

        This is the regression guard for the worker-thread load: on the
        pre-fix code construction raised ``sqlite3.ProgrammingError`` here.
        """
        store = await self._build_store(tmp_path, backend="sqlite-vec", dimension=3)
        try:
            # Construction must yield a *live* backend, never the numpy fallback.
            assert isinstance(store._vector_backend, SqliteVecSearchBackend)

            await _make_thought(store, "t-x")
            await _make_thought(store, "t-y")
            await store.store_embedding(
                thought_id="t-x", vector=[1.0, 0.0, 0.0], model_name=_PARITY_MODEL
            )
            await store.store_embedding(
                thought_id="t-y", vector=[0.0, 1.0, 0.0], model_name=_PARITY_MODEL
            )

            results = await store.search_similar([1.0, 0.0, 0.0], top_k=2)
            assert results, "sqlite-vec search returned no results"
            # Nearest neighbour to [1,0,0] is t-x.
            assert results[0][0] == "t-x"
        finally:
            await store.close()

    async def test_sqlite_vec_search_after_construction_uses_worker_thread(
        self, tmp_path: Path
    ) -> None:
        """Index ops after the load succeed (load + queries share one thread).

        ``ensure_index`` / ``upsert_embedding`` / ``search`` all run through
        ``await db.execute`` on aiosqlite's worker thread.  If the extension
        had been loaded on a different thread, these would fail; that they
        succeed confirms load and queries share the worker thread.
        """
        store = await self._build_store(tmp_path, backend="sqlite-vec", dimension=2)
        try:
            await _make_thought(store, "t-near")
            await _make_thought(store, "t-far")
            await store.store_embedding(
                thought_id="t-near", vector=[0.9, 0.1], model_name=_PARITY_MODEL
            )
            await store.store_embedding(
                thought_id="t-far", vector=[-1.0, 0.0], model_name=_PARITY_MODEL
            )

            results = await store.search_similar([1.0, 0.0], top_k=2)
            ids = [r[0] for r in results]
            assert "t-near" in ids
            assert ids[0] == "t-near"
        finally:
            await store.close()

    async def test_sqlite_vec_matches_numpy_backend(self, tmp_path: Path) -> None:
        """Accuracy parity: sqlite-vec returns the same ranking as numpy.

        Over a small deterministic fixture the ANN index is exact, so the
        sqlite-vec backend must retrieve the same ids in the same order as
        the brute-force numpy backend, and with matching cosine scores.
        """
        dimension = 3
        fixture: list[tuple[str, list[float]]] = [
            ("t-1", [1.0, 0.0, 0.0]),
            ("t-2", [0.0, 1.0, 0.0]),
            ("t-3", [0.0, 0.0, 1.0]),
            ("t-4", [0.8, 0.2, 0.0]),
            ("t-5", [0.1, 0.9, 0.1]),
        ]
        query = [0.9, 0.1, 0.0]

        async def collect(backend: str) -> list[tuple[str, float]]:
            store = await self._build_store(tmp_path, backend=backend, dimension=dimension)
            try:
                for tid, vec in fixture:
                    await _make_thought(store, tid)
                    await store.store_embedding(
                        thought_id=tid, vector=vec, model_name=_PARITY_MODEL
                    )
                return await store.search_similar(query, top_k=len(fixture))
            finally:
                await store.close()

        numpy_results = await collect("numpy")
        vec_results = await collect("sqlite-vec")

        # Same ids in the same order.
        assert [r[0] for r in vec_results] == [r[0] for r in numpy_results]
        # Same cosine scores (within float tolerance).
        for (vid, vscore), (nid, nscore) in zip(vec_results, numpy_results, strict=True):
            assert vid == nid
            assert abs(vscore - nscore) < 1e-5

    async def test_sqlite_vec_tie_break_matches_numpy(self, tmp_path: Path) -> None:
        """Score-tied rows resolve to the same canonical order on both backends.

        The vec0 backend's own ``.search()`` sorts by score only; determinism on
        a cosine tie relies on ``search_similar`` re-sorting via
        ``_sort_scored_descending`` (score DESC, then thought_id ASC). Two rows
        equidistant from the query must return in the same id order from vec0 and
        numpy — and, for the tie, id-ascending.
        """
        dimension = 3
        # Query on the x=y diagonal: [1,0,0] and [0,1,0] are exactly equidistant.
        query = [1.0, 1.0, 0.0]
        fixture: list[tuple[str, list[float]]] = [
            ("tie-b", [0.0, 1.0, 0.0]),  # inserted b-before-a so scan order != id order
            ("tie-a", [1.0, 0.0, 0.0]),
            ("far", [0.0, 0.0, 1.0]),
        ]

        async def collect(backend: str) -> list[tuple[str, float]]:
            store = await self._build_store(tmp_path, backend=backend, dimension=dimension)
            try:
                for tid, vec in fixture:
                    await _make_thought(store, tid)
                    await store.store_embedding(
                        thought_id=tid, vector=vec, model_name=_PARITY_MODEL
                    )
                return await store.search_similar(query, top_k=len(fixture))
            finally:
                await store.close()

        numpy_results = await collect("numpy")
        vec_results = await collect("sqlite-vec")

        assert [r[0] for r in vec_results] == [r[0] for r in numpy_results]
        # The tie resolves id-ascending (tie-a before tie-b) on both backends.
        assert [r[0] for r in numpy_results if r[0].startswith("tie-")] == ["tie-a", "tie-b"]


async def _vec_rowids(store: SqliteEngravaCore) -> set[int]:
    """Return the set of rowids currently present in ``embedding_vec``."""
    cursor = await store._db.execute("SELECT rowid FROM embedding_vec")
    return {int(row["rowid"]) for row in await cursor.fetchall()}


async def _embedding_rowid(store: SqliteEngravaCore, thought_id: str) -> int | None:
    """Return the ``embedding`` rowid for a thought, or ``None`` if absent."""
    cursor = await store._db.execute(
        "SELECT rowid FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
        (thought_id,),
    )
    row = await cursor.fetchone()
    return int(row["rowid"]) if row is not None else None


# ------------------------------------------------------------------
# R4 — vec0 delete leaves no ghost vector
# ------------------------------------------------------------------


class _PrivateDimensionProvider:
    """A provider that keeps its dimension private, violating the protocol.

    ``EmbeddingProviderProtocol`` requires a public ``dimension``; this shape
    (the value held as ``self._dimension`` with no property) is the natural way
    to get it wrong, and the core raises ``EmbeddingProviderContractError`` the
    moment it has to read the member.
    """

    def __init__(self, dimension: int = 3) -> None:
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        """Return the embedding model name."""
        return "private-dimension"

    async def embed(self, text: str) -> list[float]:
        """Return a fixed-length vector for the given text."""
        return [float(len(text) % 3)] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one fixed-length vector per input text."""
        return [await self.embed(text) for text in texts]


@sqlite_vec_required
class TestVec0DimensionTakesPrecedenceOverTheProvider(TestSqliteVecRealConnection):
    """A configured vec0 index answers for the dimension, so the provider is not asked.

    The upgrade note and the known-limitations page both say a store with a
    ``sqlite-vec`` backend takes the dimension from its dimension-typed ``vec0``
    table and therefore never asks the provider *for its dimension* on the
    vector-search path. That is a documented guarantee, so it is exercised
    against a real loaded extension rather than read off the branch order.

    Both stores are built the way ``from_config`` builds one — the provider is
    passed to the constructor, and the backend is selected through
    ``_configure_vector_backend`` — so the guarantee is checked under real
    initialisation rather than by assigning to store internals afterwards.
    """

    async def _store_with_private_dimension_provider(
        self,
        tmp_path: Path,
        *,
        backend: str,
    ) -> SqliteEngravaCore:
        """Build a store whose provider omits ``dimension``, on the given backend."""
        db = await aiosqlite.connect(str(tmp_path / f"{backend}-precedence.db"))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        store = SqliteEngravaCore(db, embedding_provider=_PrivateDimensionProvider())  # type: ignore[arg-type]  # the non-conformant shape is the subject
        store._owns_connection = True
        await store.ensure_schema()
        await store._configure_vector_backend(backend_name=backend, embedding_dimension=3)
        return store

    async def test_vector_search_succeeds_with_a_non_conformant_provider(
        self,
        tmp_path: Path,
    ) -> None:
        """The search runs to a result even though the provider has no ``dimension``."""
        store = await self._store_with_private_dimension_provider(tmp_path, backend="sqlite-vec")
        try:
            await _make_thought(store, "vec-precedence-1")
            await store.store_embedding("vec-precedence-1", [1.0, 0.0, 0.0], model_name="m")

            results = await store.search_similar([1.0, 0.0, 0.0], top_k=5)

            assert [tid for tid, _ in results] == ["vec-precedence-1"]
        finally:
            await store.close()

    async def test_from_config_produces_the_state_this_precedence_relies_on(
        self,
        tmp_path: Path,
    ) -> None:
        """``from_config`` really leaves a store whose vec0 index declares the dimension.

        The two tests around this one build the store the way ``from_config``
        does; this one closes the gap between "the way it does" and what it
        actually does. A ``from_config`` that chose another backend, or left the
        dimension unresolved, would make the guarantee those tests establish a
        statement about a store no user has.
        """
        from engrava.extensions.vector_sqlite_vec import SqliteVecSearchBackend

        db_path = tmp_path / "from_config_vec.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\n"
            f"extensions:\n  vector:\n    backend: sqlite-vec\n    dimension: 3\n",
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            assert isinstance(store._vector_backend, SqliteVecSearchBackend)
            assert store._declared_embedding_dimension() == 3

    async def test_the_same_provider_is_asked_on_the_numpy_backend(
        self,
        tmp_path: Path,
    ) -> None:
        """The counter-state: the store that declares no vec0 dimension raises.

        Same construction, same provider, the supported ``numpy`` backend — which
        declares no dimension of its own, so the provider is asked. Without this
        the test above would pass just as well against a store that never
        consults the provider under any configuration.
        """
        store = await self._store_with_private_dimension_provider(tmp_path, backend="numpy")
        try:
            await _make_thought(store, "vec-precedence-2")
            await store.store_embedding("vec-precedence-2", [1.0, 0.0, 0.0], model_name="m")
            before = await store.count_thoughts()

            with pytest.raises(EmbeddingProviderContractError) as excinfo:
                await store.search_similar([1.0, 0.0, 0.0], top_k=5)

            assert await store.count_thoughts() == before
            assert (await store.get_thought("vec-precedence-2")) is not None
            assert store.vector_arm_degradation_count == 0
            assert excinfo.value.provider_class == "_PrivateDimensionProvider"
            assert excinfo.value.member == "dimension"
        finally:
            await store.close()


@sqlite_vec_required
class TestVec0DeleteRemovesVector(TestSqliteVecRealConnection):
    """Deleting a thought must also drop its vec0 vector (no ghost row)."""

    async def test_delete_thought_removes_vec_row(self, tmp_path: Path) -> None:
        """delete_thought purges the vec0 vector; a later search never sees it.

        On the pre-fix code the ``embedding`` row is FK-cascaded away but the
        vec0 vector lingers, so the rowid stays in ``embedding_vec`` and can
        still occupy a KNN slot.
        """
        store = await self._build_store(tmp_path, backend="sqlite-vec", dimension=3)
        try:
            await _make_thought(store, "t-keep")
            await _make_thought(store, "t-drop")
            await store.store_embedding(
                thought_id="t-keep", vector=[1.0, 0.0, 0.0], model_name=_PARITY_MODEL
            )
            await store.store_embedding(
                thought_id="t-drop", vector=[0.9, 0.1, 0.0], model_name=_PARITY_MODEL
            )
            drop_rowid = await _embedding_rowid(store, "t-drop")
            assert drop_rowid is not None
            assert drop_rowid in await _vec_rowids(store)

            # Confirm it is a live search hit before deletion.
            before = await store.search_similar([0.9, 0.1, 0.0], top_k=2)
            assert "t-drop" in [r[0] for r in before]

            deleted = await store.delete_thought("t-drop")
            assert deleted is True

            # The vec0 vector must be gone — no ghost, no reserved slot.
            assert drop_rowid not in await _vec_rowids(store)
            after = await store.search_similar([0.9, 0.1, 0.0], top_k=2)
            assert "t-drop" not in [r[0] for r in after]
            assert "t-keep" in [r[0] for r in after]
        finally:
            await store.close()

    async def test_ttl_delete_sweep_purges_only_the_expired_vector(
        self,
        tmp_path: Path,
    ) -> None:
        """The delete-strategy TTL sweep drops the expired vectors and no others.

        ``cleanup_expired`` reaches the same orphan-vector purge as
        ``delete_thought``, and ``embedding_vec`` is the one store a widened
        purge could empty without any foreign key or ``embedding`` read-back
        noticing. Both halves are therefore read from the index itself: the
        expired rowids are gone, and what remains is **exactly** the surviving
        set, still returned by a search.

        The corpus is interleaved survivor · **doomed** · survivor · **doomed**
        · survivor on the two ordering axes it controls: the thought id
        (``t-anchor`` < ``t-drop`` < ``t-keep`` < ``t-purge`` < ``t-vault``) and
        ``rowid``, the key the vec0 table is actually addressed by. A purge that
        picks a row by **rowid** position — lowest, highest, any other row —
        instead of by the rowid it was handed therefore cannot land on a doomed
        one by luck, and **two** doomed vectors make a sweep that purges only
        the first distinguishable from one that purges both. Rowids follow
        insertion order, and the interleaving is asserted, not assumed. The
        stored vectors themselves are chosen for a deterministic search order,
        not to bracket anything.
        """
        store = await self._build_store(
            tmp_path,
            backend="sqlite-vec",
            dimension=3,
            ttl_strategy="delete",
        )
        try:
            expired_at = _past_iso()
            for thought_id, expires_at in (
                ("t-anchor", None),
                ("t-drop", expired_at),
                ("t-keep", None),
                ("t-purge", expired_at),
                ("t-vault", None),
            ):
                await _make_thought(store, thought_id, expires_at=expires_at)
            for thought_id, vector in (
                ("t-anchor", [0.8, 0.2, 0.0]),
                ("t-drop", [0.9, 0.1, 0.0]),
                ("t-keep", [1.0, 0.0, 0.0]),
                ("t-purge", [0.95, 0.05, 0.0]),
                ("t-vault", [0.6, 0.4, 0.0]),
            ):
                await store.store_embedding(
                    thought_id=thought_id, vector=vector, model_name=_PARITY_MODEL
                )
            order = ("t-anchor", "t-drop", "t-keep", "t-purge", "t-vault")
            rowids_by_thought: dict[str, int] = {}
            for tid in order:
                rowid = await _embedding_rowid(store, tid)
                assert rowid is not None, f"no embedding stored for {tid}"
                rowids_by_thought[tid] = rowid
            surviving = {rowids_by_thought[t] for t in ("t-anchor", "t-keep", "t-vault")}
            doomed = {rowids_by_thought[t] for t in ("t-drop", "t-purge")}
            # Corpus precondition: survivors sit either side of each doomed
            # rowid in the index's own key space, and the doomed pair is not
            # adjacent by rowid, so no rowid-positional purge selects exactly it.
            assert [rowids_by_thought[t] for t in order] == sorted(
                rowids_by_thought[t] for t in order
            )
            assert await _vec_rowids(store) == surviving | doomed

            result = await store.cleanup_expired()

            rowids = await _vec_rowids(store)
            assert not doomed & rowids, "an expired vector survived the sweep"
            # Set equality, not containment: containment would hold just as well
            # if the sweep had left extra rows behind in the index.
            assert rowids == surviving
            hits = [r[0] for r in await store.search_similar([1.0, 0.0, 0.0], top_k=5)]
            assert hits == ["t-keep", "t-anchor", "t-vault"]
            # Only once the index is settled does the reported count matter.
            assert result.expired_count == len(doomed)
        finally:
            await store.close()

    async def test_delete_thought_numpy_backend_unaffected(self, tmp_path: Path) -> None:
        """On the numpy backend delete_thought behaves exactly as before.

        Guards the ``_vector_backend is None`` path: no ``embedding_vec`` table
        exists, and the purge helper must no-op rather than raise.
        """
        store = await self._build_store(tmp_path, backend="numpy", dimension=3)
        try:
            assert store._vector_backend is None
            await _make_thought(store, "t-keep")
            await _make_thought(store, "t-drop")
            await store.store_embedding(
                thought_id="t-keep", vector=[1.0, 0.0, 0.0], model_name=_PARITY_MODEL
            )
            await store.store_embedding(
                thought_id="t-drop", vector=[0.9, 0.1, 0.0], model_name=_PARITY_MODEL
            )

            deleted = await store.delete_thought("t-drop")
            assert deleted is True
            assert await _embedding_rowid(store, "t-drop") is None

            results = await store.search_similar([0.9, 0.1, 0.0], top_k=2)
            assert "t-drop" not in [r[0] for r in results]
            assert "t-keep" in [r[0] for r in results]
        finally:
            await store.close()

    async def test_sync_embeddings_prunes_preexisting_ghost(self, tmp_path: Path) -> None:
        """sync_embeddings sweeps out an orphan vec row (self-healing reconcile).

        Simulates a store that already accumulated a ghost (a vec row whose
        ``embedding`` row was deleted directly, e.g. by the CLI ``gc`` path).
        Re-running ``sync_embeddings`` must delete the orphan vector.
        """
        store = await self._build_store(tmp_path, backend="sqlite-vec", dimension=3)
        try:
            await _make_thought(store, "t-live")
            await _make_thought(store, "t-ghost")
            await store.store_embedding(
                thought_id="t-live", vector=[1.0, 0.0, 0.0], model_name=_PARITY_MODEL
            )
            await store.store_embedding(
                thought_id="t-ghost", vector=[0.0, 1.0, 0.0], model_name=_PARITY_MODEL
            )
            ghost_rowid = await _embedding_rowid(store, "t-ghost")
            assert ghost_rowid is not None

            # Delete the embedding row directly, leaving the vec row orphaned —
            # exactly the ghost state a non-core delete path produces.
            await store._db.execute(
                "DELETE FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
                ("t-ghost",),
            )
            await store._db.commit()
            assert ghost_rowid in await _vec_rowids(store)

            assert store._vector_backend is not None
            await store._vector_backend.sync_embeddings(store._db)

            assert ghost_rowid not in await _vec_rowids(store)
            # The live vector is untouched.
            live_rowid = await _embedding_rowid(store, "t-live")
            assert live_rowid in await _vec_rowids(store)
        finally:
            await store.close()


# ------------------------------------------------------------------
# R3 — vec0 over-fetch fills the top-k live window
# ------------------------------------------------------------------


@sqlite_vec_required
class TestVec0OverfetchFillsTopK(TestSqliteVecRealConnection):
    """The vec0 arm over-fetches so the live-row filter can still fill top_k."""

    async def _seed_expired_heavy(
        self, store: SqliteEngravaCore, *, live: int, expired: int
    ) -> list[float]:
        """Seed a store where the nearest neighbours are mostly non-live.

        The expired/retired rows are placed *closer* to the query than the
        live rows, so a naive top-k fetch would surface them first and the
        post-filter would then drop them — the exact under-fill R3 fixes.

        Returns the query vector to use.
        """
        query = [1.0, 0.0]
        # Non-live rows sit right on the query axis (closest). Half are expired
        # OBSERVATIONs, half are retired (ARCHIVED) REFLECTIONs.
        for i in range(expired):
            tid = f"t-exp-{i}"
            if i % 2 == 0:
                await _make_thought(store, tid, expires_at=_past_iso())
            else:
                await _make_thought(
                    store,
                    tid,
                    thought_type=ThoughtType.REFLECTION,
                    lifecycle_status=LifecycleStatus.ARCHIVED,
                )
            # Slightly rotated but still very close to [1, 0].
            angle = 0.001 * (i + 1)
            await store.store_embedding(
                thought_id=tid, vector=[1.0 - angle, angle], model_name=_PARITY_MODEL
            )
        # Live rows are a little further out but still clearly relevant.
        for i in range(live):
            tid = f"t-live-{i}"
            await _make_thought(store, tid)
            angle = 0.05 + 0.001 * i
            await store.store_embedding(
                thought_id=tid, vector=[1.0 - angle, angle], model_name=_PARITY_MODEL
            )
        return query

    async def test_search_similar_fills_topk_despite_expired(self, tmp_path: Path) -> None:
        """search_similar returns top_k live rows even when nearer rows are dead.

        On the pre-fix code only ``top_k`` neighbours are fetched; the closest
        ``top_k`` are the non-live rows, which the post-filter removes, leaving
        fewer than ``top_k`` (often zero) results.
        """
        store = await self._build_store(tmp_path, backend="sqlite-vec", dimension=2)
        try:
            query = await self._seed_expired_heavy(store, live=5, expired=5)
            results = await store.search_similar(query, top_k=3)
            ids = [r[0] for r in results]
            # Filled to top_k, all live.
            assert len(results) == 3
            assert all(tid.startswith("t-live-") for tid in ids)
            # Descending by similarity (deterministic total order preserved).
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            await store.close()

    async def test_returns_all_available_when_fewer_than_topk(self, tmp_path: Path) -> None:
        """When live rows < top_k the arm returns exactly the available live set."""
        store = await self._build_store(tmp_path, backend="sqlite-vec", dimension=2)
        try:
            query = await self._seed_expired_heavy(store, live=2, expired=4)
            results = await store.search_similar(query, top_k=5)
            ids = [r[0] for r in results]
            assert len(results) == 2
            assert all(tid.startswith("t-live-") for tid in ids)
        finally:
            await store.close()

    async def test_numpy_vec_parity_expired_heavy(self, tmp_path: Path) -> None:
        """numpy and vec0 return the same live result set on an expired-heavy store.

        Both arms must exclude the non-live rows and surface the same live ids
        (order/count identical, scores within float tolerance).
        """

        async def collect(backend: str) -> list[tuple[str, float]]:
            store = await self._build_store(
                tmp_path, backend=backend, dimension=2, db_name=f"parity-{backend}"
            )
            try:
                query = await self._seed_expired_heavy(store, live=4, expired=6)
                return await store.search_similar(query, top_k=3)
            finally:
                await store.close()

        numpy_results = await collect("numpy")
        vec_results = await collect("sqlite-vec")

        assert [r[0] for r in vec_results] == [r[0] for r in numpy_results]
        assert len(vec_results) == 3
        for (vid, vscore), (nid, nscore) in zip(vec_results, numpy_results, strict=True):
            assert vid == nid
            assert abs(vscore - nscore) < 1e-5

    async def test_search_hybrid_fills_pool_on_expired_store(self, tmp_path: Path) -> None:
        """search_hybrid's vector arm yields the expected fused top-k live pool.

        Weighted purely on the vector arm (``fts_weight=0``) so the fused
        result depends on the vec pool alone: the deeper live pool now feeds
        hybrid fusion, and the fused top-k fills with live rows while non-live
        rows never appear. (Positive integration check of the fused path;
        the strict under-fill regressions live in the ``search_similar`` tests
        above, where ``top_k`` is the binding fetch bound.)
        """
        store = await self._build_store(tmp_path, backend="sqlite-vec", dimension=2)
        try:
            query = await self._seed_expired_heavy(store, live=5, expired=5)
            result = await store.search_hybrid(
                query_text="essence",
                query_vector=query,
                top_k=3,
                fts_weight=0.0,
                vector_weight=1.0,
                recency_weight=0.0,
                priority_weight=0.0,
                graph_weight=0.0,
            )
            ids = [tid for tid, _ in result.results]
            assert "vector" in result.backends_used
            # Fused top-k fills entirely from the (now-complete) live vec pool.
            assert len(result.results) == 3
            assert all(tid.startswith("t-live-") for tid in ids)
        finally:
            await store.close()

    async def test_clean_store_unchanged_and_cap_respected(self, tmp_path: Path) -> None:
        """A no-expiry store returns the same results regardless of over-fetch.

        Over-fetch only changes behaviour when non-live rows exist; on a clean
        store the trimmed-to-top_k result is identical for factor 1 and 8, and
        the ABSOLUTE_CAP bounds the effective fetch.
        """
        from engrava.config import SearchConfig
        from engrava.infrastructure.sqlite.engrava_core import _VEC0_OVERFETCH_CAP

        fixture = [
            ("t-a", [1.0, 0.0]),
            ("t-b", [0.9, 0.1]),
            ("t-c", [0.7, 0.3]),
            ("t-d", [0.0, 1.0]),
        ]
        query = [1.0, 0.0]

        async def collect(factor: int) -> list[tuple[str, float]]:
            store = await self._build_store(
                tmp_path,
                backend="sqlite-vec",
                dimension=2,
                search_config=SearchConfig(vec0_overfetch_factor=factor),
                db_name=f"clean-{factor}",
            )
            try:
                for tid, vec in fixture:
                    await _make_thought(store, tid)
                    await store.store_embedding(
                        thought_id=tid, vector=vec, model_name=_PARITY_MODEL
                    )
                return await store.search_similar(query, top_k=3)
            finally:
                await store.close()

        low = await collect(1)
        high = await collect(8)
        assert [r[0] for r in low] == [r[0] for r in high]
        assert len(low) == 3

        # The cap is a sane, positive bound comfortably above realistic top_k.
        assert _VEC0_OVERFETCH_CAP >= 100
        # effective_fetch = min(top_k * factor, cap); with a large factor the
        # cap is what binds — assert the cap clamps a pathological product.
        assert min(3 * 10_000, _VEC0_OVERFETCH_CAP) == _VEC0_OVERFETCH_CAP


# ------------------------------------------------------------------
# Config: vec0_overfetch_factor parsing + threading
# ------------------------------------------------------------------


class TestVec0OverfetchConfig:
    def test_default_factor(self) -> None:
        from engrava.config import _parse_search

        assert _parse_search(None).vec0_overfetch_factor == 4

    def test_explicit_factor(self) -> None:
        from engrava.config import _parse_search

        assert _parse_search({"vec0_overfetch_factor": 6}).vec0_overfetch_factor == 6

    def test_non_positive_factor_raises(self) -> None:
        from engrava.config import ConfigError, _parse_search

        with pytest.raises(ConfigError, match=r"vec0_overfetch_factor.*positive integer"):
            _parse_search({"vec0_overfetch_factor": 0})

    def test_non_int_factor_raises(self) -> None:
        from engrava.config import ConfigError, _parse_search

        with pytest.raises(ConfigError, match=r"vec0_overfetch_factor.*positive integer"):
            _parse_search({"vec0_overfetch_factor": "four"})

    def test_threads_through_load_config(self, tmp_path: Path) -> None:
        from engrava.config import load_config

        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            "database:\n  path: test.db\nsearch:\n  vec0_overfetch_factor: 7\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.search.vec0_overfetch_factor == 7


# ------------------------------------------------------------------
# The dimension in the DDL is the dimension that was validated
# ------------------------------------------------------------------


class _LyingDimension(int):
    """A dimension whose numeric value and whose rendering disagree.

    Every numeric check — ``isinstance``, ``>= 1``, comparison against a stored
    vector length — reads the real value 384. ``__format__`` is what the DDL
    f-string calls, and it answers with schema text of its own.
    """

    def __format__(self, format_spec: str) -> str:
        del format_spec
        return "1] distance_metric=L2, smuggled float[1"


class TestVectorTableDeclarationUsesTheValidatedDimension:
    """``vec0(...)`` is DDL: it cannot be parameterised, so the value must be owned.

    The dimension is the only caller-supplied value in the declaration, and the
    declaration decides the vector length, the distance metric and the column
    list of the index every later search runs against.
    """

    def test_the_backend_stores_an_exact_int(self) -> None:
        backend = SqliteVecSearchBackend(_LyingDimension(384))
        assert type(backend.dimension) is int
        assert backend.dimension == 384
        assert f"float[{backend.dimension}]" == "float[384]"

    @pytest.mark.parametrize("dimension", [0, -1, True, 3.5, "384", None])
    def test_a_dimension_that_is_not_a_positive_int_is_refused(self, dimension: object) -> None:
        """The class validates at its own boundary; it is a public export."""
        with pytest.raises(ConfigError, match="must be a positive integer"):
            SqliteVecSearchBackend(dimension)  # type: ignore[arg-type]  # passing the wrong type is the behaviour under test

    @sqlite_vec_required
    async def test_the_created_table_declares_the_validated_dimension(self) -> None:
        db = await aiosqlite.connect(":memory:")
        try:
            assert await load_sqlite_vec(db)
            backend = SqliteVecSearchBackend(_LyingDimension(384))
            await backend.ensure_index(db)

            cursor = await db.execute("SELECT sql FROM sqlite_master WHERE name = 'embedding_vec'")
            row = await cursor.fetchone()
            assert row is not None
            declaration = str(row[0])
            assert "float[384]" in declaration
            assert "distance_metric=cosine" in declaration
            assert "smuggled" not in declaration
            assert "L2" not in declaration
        finally:
            await db.close()

    @sqlite_vec_required
    async def test_a_manager_configured_index_declares_its_own_dimension(
        self,
        tmp_path: Path,
    ) -> None:
        """The manager takes a dimension straight from its caller — check it there too."""
        manager = EngravaManager(
            data_dir=tmp_path / "data",
            vector_backend="sqlite-vec",
            embedding_dimension=_LyingDimension(384),
        )
        try:
            store = await manager.get_store("svc")
            cursor = await store._db.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'embedding_vec'"
            )
            row = await cursor.fetchone()
            assert row is not None
            declaration = str(row[0])
            assert "float[384]" in declaration
            assert "smuggled" not in declaration
        finally:
            await manager.close_all()

    def test_a_config_stores_an_exact_int(self, tmp_path: Path) -> None:
        config = EngravaConfig(
            database_path=tmp_path / "t.db",
            embedding_dimension=_LyingDimension(384),
        )
        assert type(config.embedding_dimension) is int
        assert config.embedding_dimension == 384

    @sqlite_vec_required
    async def test_a_legitimate_dimension_still_builds_a_working_index(self) -> None:
        db = await aiosqlite.connect(":memory:")
        try:
            assert await load_sqlite_vec(db)
            backend = SqliteVecSearchBackend(dimension=4)
            await backend.ensure_index(db)
            await db.execute(
                "INSERT INTO embedding_vec(rowid, embedding) VALUES (?, ?)",
                (1, "[1.0,0.0,0.0,0.0]"),
            )
            cursor = await db.execute("SELECT COUNT(*) FROM embedding_vec")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1
        finally:
            await db.close()


class _BackendNameThatComparesAsAnother(str):
    """Reads as its real text; hashes and compares as ``sqlite-vec``."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash("sqlite-vec")

    def __eq__(self, other: object) -> bool:
        return other == "sqlite-vec"

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class TestVectorBackendSelectionUsesTheValidatedName:
    """The backend named in the configuration is the backend that gets wired."""

    def test_a_config_stores_an_exact_backend_name(self, tmp_path: Path) -> None:
        config = EngravaConfig(
            database_path=tmp_path / "t.db",
            vector_backend=_BackendNameThatComparesAsAnother("numpy"),
        )
        assert type(config.vector_backend) is str
        assert config.vector_backend == "numpy"

    async def test_a_numpy_backend_builds_no_vector_table(self, tmp_path: Path) -> None:
        manager = EngravaManager(
            data_dir=tmp_path / "data",
            vector_backend=_BackendNameThatComparesAsAnother("numpy"),
        )
        try:
            store = await manager.get_store("svc")
            cursor = await store._db.execute(
                "SELECT name FROM sqlite_master WHERE name = 'embedding_vec'"
            )
            assert await cursor.fetchone() is None
        finally:
            await manager.close_all()

    @sqlite_vec_required
    async def test_an_explicit_sqlite_vec_backend_still_builds_one(self, tmp_path: Path) -> None:
        manager = EngravaManager(data_dir=tmp_path / "data", vector_backend="sqlite-vec")
        try:
            store = await manager.get_store("svc")
            cursor = await store._db.execute(
                "SELECT name FROM sqlite_master WHERE name = 'embedding_vec'"
            )
            assert await cursor.fetchone() is not None
        finally:
            await manager.close_all()

    async def test_a_non_string_backend_name_is_a_configuration_error(
        self,
        tmp_path: Path,
    ) -> None:
        manager = EngravaManager(data_dir=tmp_path / "data", vector_backend=object())  # type: ignore[arg-type]  # passing the wrong type is the behaviour under test
        try:
            with pytest.raises(ConfigError, match="vector_backend must be a string"):
                await manager.get_store("svc")
        finally:
            await manager.close_all()

"""Tests for engrava.extensions.vector_sqlite_vec."""

from __future__ import annotations

import importlib.util
import sqlite3
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiosqlite
import pytest

from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.vector_sqlite_vec import (
    SqliteVecSearchBackend,
    _load_sqlite_vec_sync,
    load_sqlite_vec,
)
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


async def _make_thought(store: SqliteEngravaCore, thought_id: str) -> None:
    """Create a minimal thought so embeddings satisfy the FK to ``thought``."""
    thought = ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"essence {thought_id}",
        content=f"content {thought_id}",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )
    await store.create_thought(thought)


@sqlite_vec_required
class TestSqliteVecRealConnection:
    """End-to-end tests against a real loaded sqlite-vec extension."""

    async def _build_store(
        self,
        tmp_path: Path,
        *,
        backend: str,
        dimension: int,
    ) -> SqliteEngravaCore:
        """Construct a store with a real connection and the given backend.

        Mirrors what ``from_config`` does internally (schema bootstrap then
        ``_configure_vector_backend``) but lets the test pick the backend.
        """
        db_path = tmp_path / f"{backend}.db"
        db = await aiosqlite.connect(str(db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        store = SqliteEngravaCore(db)
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

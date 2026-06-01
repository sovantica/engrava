"""Tests for engrava.extensions.vector_sqlite_vec."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engrava.extensions.vector_sqlite_vec import (
    SqliteVecSearchBackend,
    load_sqlite_vec,
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
    def test_import_error_returns_false(self) -> None:
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            conn = MagicMock()
            result = load_sqlite_vec(conn)
            assert result is False

    def test_enable_load_extension_error_returns_false(self) -> None:
        mock_vec = MagicMock()
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            conn = MagicMock()
            conn.enable_load_extension.side_effect = AttributeError("no attr")
            result = load_sqlite_vec(conn)
            assert result is False

    def test_successful_load(self) -> None:
        mock_vec = MagicMock()
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            conn = MagicMock()
            result = load_sqlite_vec(conn)
            assert result is True
            conn.enable_load_extension.assert_called()
            mock_vec.load.assert_called_once_with(conn)


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

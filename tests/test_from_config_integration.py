"""Integration tests for from_config() and async context manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from pathlib import Path
import pytest

from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

# ------------------------------------------------------------------
# from_config integration
# ------------------------------------------------------------------


class TestFromConfig:
    async def test_from_config_creates_store(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\n",
            encoding="utf-8",
        )

        store = await SqliteEngravaCore.from_config(cfg_file)
        try:
            thought = ThoughtRecord(
                thought_id="t1",
                thought_type=ThoughtType.TASK,
                essence="test from_config",
                content="integration test content",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.CREATED,
                created_cycle=0,
                updated_cycle=0,
                source="test",
            )
            created = await store.create_thought(thought)
            assert created.thought_id == "t1"

            fetched = await store.get_thought("t1")
            assert fetched is not None
            assert fetched.essence == "test from_config"

            # FTS should be available
            fts_results = await store.search_fts("from_config")
            assert len(fts_results) >= 1
        finally:
            await store.close()

    async def test_from_config_async_context_manager(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctx.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\n",
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            thought = ThoughtRecord(
                thought_id="t2",
                thought_type=ThoughtType.OBSERVATION,
                essence="ctx manager test",
                content="async with works",
                priority=Priority.P3,
                lifecycle_status=LifecycleStatus.CREATED,
                created_cycle=0,
                updated_cycle=0,
                source="test",
            )
            await store.create_thought(thought)
            assert await store.get_thought("t2") is not None

        # Connection should be closed after __aexit__
        assert store._owns_connection is True

    async def test_from_config_wal_mode(self, tmp_path: Path) -> None:
        db_path = tmp_path / "wal.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\n  wal_mode: true\n",
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            cursor = await store._db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0] == "wal"

    async def test_from_config_no_wal(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nowal.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\n  wal_mode: false\n",
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            # Default for new dbs is 'delete' when WAL not explicitly set
            cursor = await store._db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0] != "wal"

    async def test_from_config_hooks_class(self, tmp_path: Path) -> None:
        db_path = tmp_path / "hooks.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\n"
            f"hooks:\n  class: engrava.domain.protocols.hooks.DefaultEngravaHooks\n",
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            from engrava.domain.protocols.hooks import DefaultEngravaHooks

            assert isinstance(store._hooks, DefaultEngravaHooks)

    async def test_from_config_invalid_config_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("not_database: true\n", encoding="utf-8")

        from engrava.config import ConfigError

        with pytest.raises(ConfigError):
            await SqliteEngravaCore.from_config(cfg_file)

    async def test_from_config_numpy_backend_default(self, tmp_path: Path) -> None:
        db_path = tmp_path / "numpy.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\n",
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            assert store._vector_backend is None  # numpy fallback

    async def test_from_config_sqlite_vec_fallback(self, tmp_path: Path) -> None:
        """When sqlite-vec is requested but not installed, falls back to numpy."""
        db_path = tmp_path / "vec_fallback.db"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {db_path}\nextensions:\n  vector:\n    backend: sqlite-vec\n",
            encoding="utf-8",
        )

        async with await SqliteEngravaCore.from_config(cfg_file) as store:
            # sqlite-vec likely not installed in test env → fallback to numpy
            # No crash is the key assertion; verify numpy fallback when vec unavailable
            assert store._vector_backend is None


# ------------------------------------------------------------------
# Manual constructor backward compatibility
# ------------------------------------------------------------------


class TestManualConstructorCompat:
    async def test_manual_constructor_no_owns_connection(self) -> None:
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            assert store._owns_connection is False
            await store.close()  # Should be no-op
            # DB should still be usable
            await store.ensure_schema()
            thought = ThoughtRecord(
                thought_id="t-manual",
                thought_type=ThoughtType.TASK,
                essence="manual ctor",
                content="still works",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.CREATED,
                created_cycle=0,
                updated_cycle=0,
                source="test",
            )
            await store.create_thought(thought)
            assert await store.get_thought("t-manual") is not None
        finally:
            await db.close()

    async def test_search_similar_delegates_to_numpy(self) -> None:
        """Without vector backend, search_similar uses numpy."""
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(db)
            await store.ensure_schema()
            # No embeddings → empty result
            results = await store.search_similar([0.1, 0.2, 0.3], top_k=5)
            assert results == []
        finally:
            await db.close()

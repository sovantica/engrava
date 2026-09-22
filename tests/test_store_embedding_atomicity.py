"""Regression tests: a ``store_embedding()`` write and its vector-index
update are one failure-atomic unit.

Before the fix, ``store_embedding`` wrote the base ``embedding`` row and
then, when a vector backend was configured, called its ``upsert_embedding``
in a second, separate statement. A failure in that second step (a
wrong-dimension vector rejected by ``vec0``, or any other vec0-layer error)
left the base row's write pending in the connection's open transaction —
``self._db.in_transaction`` stayed ``True`` — so a *later, unrelated* write
on the same connection (e.g. ``create_thought``) would commit it as a side
effect, corrupting the ``embedding`` table with a row that has no matching
vector.

These tests use a real, file-backed temporary database (never ``:memory:``)
because the defect is specifically about state surviving to a later
connection-level commit.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from pathlib import Path

sqlite_vec_required = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec package not installed",
)


async def _build_store(
    tmp_path: Path,
    *,
    backend: str,
    dimension: int,
    db_name: str | None = None,
) -> SqliteEngravaCore:
    """Construct a store with a real file-backed connection and given backend."""
    db_path = tmp_path / f"{db_name or backend}.db"
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    store = SqliteEngravaCore(db)
    store._owns_connection = True
    await store.ensure_schema()
    await store._configure_vector_backend(backend_name=backend, embedding_dimension=dimension)
    return store


async def _make_thought(store: SqliteEngravaCore, thought_id: str) -> None:
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


async def _committed_dimensions(db: aiosqlite.Connection) -> list[int]:
    cursor = await db.execute("SELECT dimension FROM embedding ORDER BY rowid")
    return [int(row["dimension"]) for row in await cursor.fetchall()]


async def _embedding_row_count(db: aiosqlite.Connection) -> int:
    cursor = await db.execute("SELECT COUNT(*) AS c FROM embedding")
    row = await cursor.fetchone()
    assert row is not None
    return int(row["c"])


@sqlite_vec_required
class TestStoreEmbeddingSqliteVecAtomicity:
    """A vec0-layer rejection must not leave the base row committable."""

    async def test_vec0_layer_failure_leaves_no_orphan_after_a_later_commit(
        self, tmp_path: Path
    ) -> None:
        """Force a vec0-layer failure independent of the model/dimension lock.

        The model-identity re-check (the second fix in this change) now
        catches an ordinary "second call with a different vector length"
        before it ever reaches the vec0 upsert, so it alone would no longer
        exercise this savepoint. To test the transaction boundary in
        isolation, the vec0 index table is dropped out from under a
        same-model, same-dimension call, so the model check passes and the
        failure originates purely in the vec0 upsert step — exactly the
        class of failure the savepoint protects against.
        """
        store = await _build_store(tmp_path, backend="sqlite-vec", dimension=3)
        db = store._db
        try:
            await _make_thought(store, "t-p")
            await _make_thought(store, "t-q")

            await store.store_embedding(thought_id="t-p", vector=[1.0, 0.0, 0.0], model_name="m")
            assert await _committed_dimensions(db) == [3]

            await db.execute("DROP TABLE embedding_vec")

            with pytest.raises(Exception, match="embedding_vec"):
                await store.store_embedding(
                    thought_id="t-q", vector=[0.0, 1.0, 0.0], model_name="m"
                )

            # Nothing of the failed call survives, and the connection is not
            # left with a dangling open transaction.
            assert await _committed_dimensions(db) == [3]
            assert db.in_transaction is False

            # Recreate the index so the later, unrelated write below does not
            # itself fail for an unrelated reason.
            await db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS embedding_vec "
                "USING vec0(embedding float[3] distance_metric=cosine)"
            )
            await db.commit()

            # A later, unrelated write on the same connection must not
            # smuggle in the orphaned row from the failed call above.
            await _make_thought(store, "t-r")
            assert await _committed_dimensions(db) == [3]
        finally:
            await store.close()

    async def test_wrong_dimension_second_call_now_caught_before_vec0(self, tmp_path: Path) -> None:
        """The original bug reproduction stays closed end-to-end.

        A genuinely wrong-dimension vector on a second call is now refused
        by the model-identity re-check before it ever reaches the base row
        write or the vec0 upsert — belt and suspenders with the savepoint
        above, and pinned here so a future change to either fix cannot
        silently reopen the original ``[3] -> [3, 4]`` orphan.
        """
        from engrava import EmbeddingModelMismatchError

        store = await _build_store(tmp_path, backend="sqlite-vec", dimension=3)
        db = store._db
        try:
            await _make_thought(store, "t-x")
            await _make_thought(store, "t-z")

            await store.store_embedding(thought_id="t-x", vector=[1.0, 0.0, 0.0], model_name="m")
            assert await _committed_dimensions(db) == [3]

            with pytest.raises(EmbeddingModelMismatchError):
                await store.store_embedding(
                    thought_id="t-z", vector=[1.0, 0.0, 0.0, 0.0], model_name="m"
                )

            assert await _committed_dimensions(db) == [3]
            assert db.in_transaction is False

            await _make_thought(store, "t-w")
            assert await _committed_dimensions(db) == [3]
        finally:
            await store.close()


class TestStoreEmbeddingNumpyBackendFailure:
    """The numpy backend has no second write step, but is wrapped too."""

    async def test_fk_violation_leaves_no_orphan_after_a_later_commit(self, tmp_path: Path) -> None:
        """A base-row failure (FK violation) is self-contained either way.

        The numpy backend never reaches the ``if self._vector_backend is not
        None:`` branch (``_configure_numpy_vector_backend`` leaves
        ``self._vector_backend`` as ``None``), so it cannot fail there — the
        only way ``store_embedding`` fails for numpy is inside the base-row
        INSERT/UPDATE itself, e.g. a foreign-key violation from an
        ``owner_id`` that names no thought. A single failing statement never
        leaves a *partial* write of its own pending, but this pins that the
        savepoint wrapper does not change that, and that the connection is
        not left in a dangling open-transaction state afterward.
        """
        store = await _build_store(tmp_path, backend="numpy", dimension=3)
        db = store._db
        try:
            assert store._vector_backend is None

            await _make_thought(store, "t-real")
            await store.store_embedding(thought_id="t-real", vector=[1.0, 0.0, 0.0], model_name="m")
            assert await _embedding_row_count(db) == 1

            with pytest.raises(Exception, match="FOREIGN KEY"):
                await store.store_embedding(
                    thought_id="does-not-exist", vector=[1.0, 0.0, 0.0], model_name="m"
                )

            assert await _embedding_row_count(db) == 1
            assert db.in_transaction is False

            await _make_thought(store, "t-real2")
            assert await _embedding_row_count(db) == 1
        finally:
            await store.close()

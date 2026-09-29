"""Regression tests: a ``store_embedding()`` write and its vector-index
update are one failure-atomic unit.

``store_embedding`` writes the base ``embedding`` row and then, when a vector
backend is configured, calls its ``upsert_embedding``. A failure in that
second step (a wrong-dimension vector rejected by ``vec0``, or any other
vec0-layer error) must not leave the base row's write pending in the
connection's open transaction. When the caller had no transaction open,
``self._db.in_transaction`` is ``False`` afterwards, so a *later, unrelated*
write on the same connection (e.g. ``create_thought``) cannot commit it as a
side effect and leave the ``embedding`` table with a row that has no matching
vector.

These tests use a real, file-backed temporary database (never ``:memory:``)
because the property is about state surviving to a later connection-level
commit.
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


async def _metadata_value(db: aiosqlite.Connection, key: str) -> str | None:
    """Read a raw ``_metadata`` value, or ``None`` when the key is absent."""
    cursor = await db.execute("SELECT value FROM _metadata WHERE key = ?", (key,))
    row = await cursor.fetchone()
    return None if row is None else str(row["value"])


@sqlite_vec_required
class TestStoreEmbeddingSqliteVecAtomicity:
    """A vec0-layer rejection must not leave the base row committable."""

    async def test_vec0_layer_failure_leaves_no_orphan_after_a_later_commit(
        self, tmp_path: Path
    ) -> None:
        """Force a vec0-layer failure independent of the model/dimension lock.

        The model-identity re-check catches an ordinary "second call with a
        different vector length" before it ever reaches the vec0 upsert, so
        it alone would not exercise this savepoint. To test the transaction
        boundary in isolation, the vec0 index table is dropped out from under
        a same-model, same-dimension call, so the model check passes and the
        failure originates purely in the vec0 upsert step — exactly the class
        of failure the savepoint protects against.
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
        """A wrong-dimension second call is refused before it can leave an orphan.

        A genuinely wrong-dimension vector on a second call is refused by
        the model-identity re-check before it ever reaches the base row
        write or the vec0 upsert — belt and suspenders with the savepoint
        above, and pinned here so a future change to either guard cannot
        silently allow a ``[3] -> [3, 4]`` orphan.
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


class TestFirstEmbeddingIdentityAtomicity:
    """A rejected *first* ``store_embedding()`` must not durably lock identity.

    ``store_embedding`` calls ``_ensure_embedding_model_lock`` from inside its
    own ``_write_readback_savepoint``. On an empty corpus that helper writes
    the ``_metadata`` identity rows, so a first call whose base/vec0 write then
    fails unwinds that identity write with it, and a corrected retry can
    succeed. This is distinct from ``TestStoreEmbeddingSqliteVecAtomicity`` /
    ``TestStoreEmbeddingNumpyBackendFailure`` above, which cover a
    *second* call: these tests are about the identity lock outliving the very
    first write it is meant to gate.
    """

    @sqlite_vec_required
    async def test_dimension_mismatch_on_first_call_leaves_no_identity_committed(
        self, tmp_path: Path
    ) -> None:
        """vec0's width is fixed at store-configuration time, not derived from the first vector.

        A first vector whose length disagrees with that configured width
        fails the vec0 upsert. The identity lock is written inside the same
        savepoint, so the rejected dimension is not left committed and a
        corrected retry using the store's actual configured dimension
        succeeds.
        """
        store = await _build_store(tmp_path, backend="sqlite-vec", dimension=3)
        db = store._db
        try:
            await _make_thought(store, "t-first")
            await _make_thought(store, "t-second")

            with pytest.raises(Exception, match="Dimension mismatch"):
                await store.store_embedding(
                    thought_id="t-first", vector=[1.0, 0.0, 0.0, 0.0], model_name="m"
                )

            # The failed call left no base row, no open transaction, and no
            # identity lock either.
            assert await _embedding_row_count(db) == 0
            assert db.in_transaction is False
            assert await _metadata_value(db, "embedding_model_name") is None

            # A corrected retry, using the dimension the store was actually
            # configured for, succeeds on the same connection.
            record = await store.store_embedding(
                thought_id="t-second", vector=[1.0, 0.0, 0.0], model_name="m"
            )
            assert await _embedding_row_count(db) == 1
            assert record.dimension == 3
            assert await _metadata_value(db, "embedding_model_name") == "m"
            assert await _metadata_value(db, "embedding_dimension") == "3"
        finally:
            await store.close()

        # And after reopening the file on a fresh connection, a further
        # write against the corrected identity still succeeds.
        reopened = await _build_store(tmp_path, backend="sqlite-vec", dimension=3)
        try:
            await _make_thought(reopened, "t-third")
            await reopened.store_embedding(
                thought_id="t-third", vector=[0.0, 1.0, 0.0], model_name="m"
            )
            assert await _embedding_row_count(reopened._db) == 2
        finally:
            await reopened.close()

    async def test_fk_violation_on_first_call_leaves_no_identity_committed(
        self, tmp_path: Path
    ) -> None:
        """A first write against a nonexistent ``thought_id`` fails the owner FK.

        The numpy backend has no vec0 width to violate, so the only way this
        first call fails is the ``embedding.owner_id`` foreign key. The
        failed attempt uses a 4-dimensional placeholder vector; the corrected
        retry uses the real 3-dimensional one. The placeholder's dimension
        must not lock the corpus identity, so a retry with the right
        ``thought_id`` *and* the right dimension for the real provider
        succeeds.
        """
        store = await _build_store(tmp_path, backend="numpy", dimension=3, db_name="numpy-fk")
        db = store._db
        try:
            await _make_thought(store, "t-real")

            with pytest.raises(Exception, match="FOREIGN KEY"):
                await store.store_embedding(
                    thought_id="does-not-exist",
                    vector=[1.0, 0.0, 0.0, 0.0],
                    model_name="m",
                )

            assert await _embedding_row_count(db) == 0
            assert db.in_transaction is False
            assert await _metadata_value(db, "embedding_model_name") is None

            record = await store.store_embedding(
                thought_id="t-real", vector=[1.0, 0.0, 0.0], model_name="m"
            )
            assert await _embedding_row_count(db) == 1
            assert record.dimension == 3
            assert await _metadata_value(db, "embedding_model_name") == "m"
            assert await _metadata_value(db, "embedding_dimension") == "3"
        finally:
            await store.close()

        reopened = await _build_store(tmp_path, backend="numpy", dimension=3, db_name="numpy-fk")
        try:
            await _make_thought(reopened, "t-real2")
            await reopened.store_embedding(
                thought_id="t-real2", vector=[0.0, 1.0, 0.0], model_name="m"
            )
            assert await _embedding_row_count(reopened._db) == 2
        finally:
            await reopened.close()

    async def test_centroid_first_call_that_fails_locks_nothing(self, tmp_path: Path) -> None:
        """The centroid sentinel's lock exemption must survive this fix.

        A REFLECTION centroid write that happens to be the very first
        ``store_embedding()`` call ever made is exempt from the identity lock
        in both directions: it must never be compared against a locked
        identity, and it must never lock one itself. That must hold even
        when that first call's own base-row write fails — the sentinel
        short-circuits before the identity-lock machinery runs at all, so
        moving that machinery inside the write's own savepoint (this fix)
        must not change that a rejected centroid write locks nothing,
        before or after.
        """
        from engrava.domain.dreaming import CENTROID_MODEL_NAME

        store = await _build_store(
            tmp_path, backend="numpy", dimension=3, db_name="centroid-first-fail"
        )
        db = store._db
        try:
            with pytest.raises(Exception, match="FOREIGN KEY"):
                await store.store_embedding(
                    thought_id="does-not-exist",
                    vector=[0.1, 0.2, 0.3],
                    model_name=CENTROID_MODEL_NAME,
                )

            assert await _embedding_row_count(db) == 0
            assert await _metadata_value(db, "embedding_model_name") is None

            await _make_thought(store, "t-real")
            record = await store.store_embedding(
                thought_id="t-real", vector=[0.4, 0.5, 0.6], model_name="real-model"
            )
            assert record.dimension == 3
            assert await _metadata_value(db, "embedding_model_name") == "real-model"
        finally:
            await store.close()

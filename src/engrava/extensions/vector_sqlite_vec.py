"""SqliteVecSearchBackend — KNN vector search via sqlite-vec.

Drop-in replacement for the brute-force numpy cosine similarity search
in ``SqliteEngravaCore``.  When ``sqlite-vec`` is installed and its
extension is loaded, ``search_similar()`` delegates to the ``vec0``
virtual table for k-nearest-neighbour queries.  In the pinned
``sqlite-vec`` 0.1.x line ``vec0`` performs an exhaustive scan over a
compact, chunked columnar store of the vectors — faster and more
memory-efficient than the Python brute-force path, but **not** an
approximate / sub-linear index (no ANN guarantee at this version).

If sqlite-vec is unavailable at runtime the store falls back to the
existing numpy implementation — no crash, just a warning log.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from typing import TYPE_CHECKING

from engrava.config_validation import require_positive_int

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SqliteVecSearchBackend:
    """KNN vector search backend backed by a ``vec0`` virtual table.

    Lifecycle:
        1. ``ensure_index(db, dimension)`` — creates the virtual table.
        2. ``sync_embeddings(db)`` — backfills existing rows.
        3. ``search(db, query_vector, ...)`` — runs k-nearest-neighbour queries.

    All state is kept in SQLite; this class is stateless aside from
    the cached ``dimension``.

    Args:
        dimension: Embedding vector length (e.g. 384 for MiniLM). Must be a
            positive integer.

    Raises:
        ConfigError: If *dimension* is not a positive integer.

    """

    def __init__(self, dimension: int) -> None:
        # The dimension is interpolated into the ``vec0`` table declaration.
        # DDL cannot be parameterised, ``int`` is subclassable, and
        # ``__format__`` is overridable — so a caller's object reaching the
        # f-string writes its own text into the schema. Decode it here into an
        # exact ``int`` this object owns: the class is a public export and is
        # constructed directly as well as from a config, so it validates at its
        # own boundary instead of assuming an earlier one did.
        self._dimension = require_positive_int(dimension, "SqliteVecSearchBackend.dimension")

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension.

        Returns:
            The vector dimension as an integer.

        """
        return self._dimension

    async def ensure_index(self, db: aiosqlite.Connection) -> None:
        """Create the ``embedding_vec`` virtual table if it does not exist.

        Uses cosine distance metric so that results are consistent with
        the numpy cosine-similarity backend.

        Args:
            db: Active database connection with sqlite-vec loaded.

        """
        await db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS embedding_vec "
            f"USING vec0(embedding float[{self._dimension}] distance_metric=cosine)"
        )
        await db.commit()

    async def sync_embeddings(self, db: aiosqlite.Connection) -> int:
        """Backfill existing embeddings into the ``vec0`` index.

        Copies rows from the ``embedding`` table that are not yet
        present in ``embedding_vec``.  Uses ``INSERT OR IGNORE`` to
        be idempotent.

        Args:
            db: Active database connection with sqlite-vec loaded.

        Returns:
            Number of embeddings synced.

        """
        cursor = await db.execute(
            "SELECT e.rowid, e.dimension, e.vector_blob "
            "FROM embedding e "
            "WHERE e.owner_type = 'THOUGHT' "
            "  AND e.rowid NOT IN (SELECT rowid FROM embedding_vec)"
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            dimension: int = row["dimension"]
            blob: bytes = row["vector_blob"]
            floats = list(struct.unpack(f"{dimension}f", blob))
            vec_json = "[" + ",".join(str(f) for f in floats) + "]"
            await db.execute(
                "INSERT OR IGNORE INTO embedding_vec(rowid, embedding) VALUES (?, ?)",
                (row["rowid"], vec_json),
            )
            count += 1
        pruned = await purge_orphan_vectors(db)
        if count or pruned:
            await db.commit()
        return count

    async def search(
        self,
        db: aiosqlite.Connection,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """k-nearest-neighbour search via the sqlite-vec ``vec0`` virtual table.

        The ``vec0`` table uses cosine distance (``1 - cosine_similarity``).
        Results are converted to cosine similarity via ``1 - distance``
        so higher is better, consistent with the numpy backend.

        Args:
            db: Active database connection with sqlite-vec loaded.
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum similarity score (after conversion).

        Returns:
            List of ``(thought_id, similarity_score)`` sorted descending.

        """
        vec_json = "[" + ",".join(str(f) for f in query_vector) + "]"
        sql = (
            "SELECT ev.rowid, ev.distance "
            "FROM embedding_vec ev "
            "WHERE ev.embedding MATCH ? "
            "ORDER BY ev.distance "
            "LIMIT ?"
        )
        cursor = await db.execute(sql, (vec_json, top_k))
        vec_rows = await cursor.fetchall()

        if not vec_rows:
            return []

        # Map vec rowids back to thought_ids via the embedding table.
        rowids = [row["rowid"] for row in vec_rows]
        distances = {row["rowid"]: float(row["distance"]) for row in vec_rows}
        placeholders = ",".join("?" * len(rowids))
        cursor2 = await db.execute(
            f"SELECT rowid, owner_id FROM embedding "  # noqa: S608
            f"WHERE owner_type = 'THOUGHT' AND rowid IN ({placeholders})",
            rowids,
        )
        id_rows = await cursor2.fetchall()
        owner_map = {row["rowid"]: row["owner_id"] for row in id_rows}

        results: list[tuple[str, float]] = []
        for rowid in rowids:
            owner_id = owner_map.get(rowid)
            if owner_id is None:
                continue
            dist = distances[rowid]
            similarity = 1.0 - dist
            if similarity >= threshold:
                results.append((owner_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    async def upsert_embedding(
        self,
        db: aiosqlite.Connection,
        *,
        rowid: int,
        vector: list[float],
    ) -> None:
        """Insert or replace a single embedding in the ``vec0`` index.

        Used by ``store_embedding()`` to keep the ``vec0`` vector table in
        sync after each write to the ``embedding`` table.

        Args:
            db: Active database connection with sqlite-vec loaded.
            rowid: The rowid from the ``embedding`` table.
            vector: Embedding vector as a list of floats.

        """
        vec_json = "[" + ",".join(str(f) for f in vector) + "]"
        # DELETE + INSERT instead of INSERT OR REPLACE because vec0
        # virtual tables may not support ON CONFLICT semantics.
        await self.delete_embedding(db, rowid=rowid)
        await db.execute(
            "INSERT INTO embedding_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, vec_json),
        )

    async def delete_embedding(
        self,
        db: aiosqlite.Connection,
        *,
        rowid: int,
    ) -> None:
        """Delete a single vector from the ``vec0`` index by rowid.

        The ``embedding_vec`` virtual table is not reachable by the
        ``embedding`` table's ``ON DELETE CASCADE`` foreign key, so deleting
        a thought (which cascades to its ``embedding`` row) does not remove
        the corresponding vector. Callers on a thought-delete path must
        invoke this explicitly to avoid leaving a ghost vector that would
        otherwise keep occupying a KNN result slot forever.

        Args:
            db: Active database connection with sqlite-vec loaded.
            rowid: The rowid (shared with the ``embedding`` table) to remove.

        """
        await db.execute(
            "DELETE FROM embedding_vec WHERE rowid = ?",
            (rowid,),
        )


async def purge_orphan_vectors(db: aiosqlite.Connection) -> int:
    """Delete ``embedding_vec`` rows whose rowid is absent from ``embedding``.

    The ``embedding_vec`` virtual table is not reachable by the ``embedding``
    table's ``ON DELETE CASCADE`` foreign key, so a thought delete (or any
    direct ``DELETE FROM embedding``) can leave a vector behind as a ghost.
    Two callers reconcile the index by this predicate: the backend's own
    :meth:`SqliteVecSearchBackend.sync_embeddings`, which makes it self-healing
    on every sqlite-vec-enabled open, and the ``gc`` command, which cleans up
    after a collection over a connection that owns no backend instance. It
    lives here, as one statement, so the two cannot come to disagree about what
    an orphan is.

    Idempotent and additive — a clean store deletes nothing — and it can never
    remove the vector of an embedding that is still stored, whatever the caller
    believes it deleted.

    Ownership here is the ``embedding`` row, not the thought behind it. On a
    schema whose foreign keys cascade — every schema at or above the version
    that introduced them — the two are the same question, because deleting a
    thought takes its embedding with it. On an older one they are not: a
    delete leaves the embedding row behind, so its vector is still owned and
    stays. Nor would removing that vector by rowid help while
    :meth:`SqliteVecSearchBackend.sync_embeddings` treats a dangling embedding
    row as a valid source to backfill from: the same row puts the vector back
    at the next open. Both halves read ownership as the embedding row rather
    than the thought behind it, and changing that is a decision about the
    backend's semantics, not about any one caller.

    Args:
        db: Active database connection with sqlite-vec loaded.

    Returns:
        Number of orphan vector rows deleted.

    """
    cursor = await db.execute(
        "DELETE FROM embedding_vec WHERE rowid NOT IN (SELECT rowid FROM embedding)"
    )
    return cursor.rowcount if cursor.rowcount is not None and cursor.rowcount > 0 else 0


def _load_sqlite_vec_sync(raw_conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into a raw sqlite3 connection.

    This runs the ``enable_load_extension`` / ``sqlite_vec.load`` /
    ``enable_load_extension(False)`` sequence synchronously.  It **must**
    execute on the thread that owns ``raw_conn`` (sqlite3 enforces a
    same-thread guard); callers route it onto aiosqlite's worker thread
    via :func:`load_sqlite_vec`.

    Extension loading is re-disabled in a ``finally`` block: once
    ``enable_load_extension(True)`` has succeeded, the connection must not
    be left with extension loading enabled even if ``sqlite_vec.load``
    raises — leaving it on would be an unintended, elevated-capability
    state on a connection that the caller will keep using.

    Args:
        raw_conn: The underlying ``sqlite3.Connection`` to load into.

    """
    import sqlite_vec  # noqa: PLC0415

    raw_conn.enable_load_extension(True)  # noqa: FBT003
    try:
        sqlite_vec.load(raw_conn)
    finally:
        raw_conn.enable_load_extension(False)  # noqa: FBT003


async def load_sqlite_vec(db: aiosqlite.Connection) -> bool:
    """Attempt to load the sqlite-vec extension on the connection's worker thread.

    aiosqlite creates its underlying ``sqlite3.Connection`` on a dedicated
    worker thread and marshals every query to it.  The extension load must
    run on that same thread, otherwise sqlite3's same-thread guard rejects
    it with :class:`sqlite3.ProgrammingError`.  This coroutine routes the
    synchronous load through aiosqlite's worker-thread execution primitive
    (``Connection._execute``), so the load shares the thread that later
    serves all index and search queries.

    Any failure to load (missing package, unsupported build, threading or
    OS error) degrades gracefully: a warning is logged and ``False`` is
    returned so the caller can fall back to the numpy brute-force backend.

    Args:
        db: The aiosqlite ``Connection`` that owns the target database.

    Returns:
        ``True`` if sqlite-vec was loaded successfully, ``False`` otherwise.

    """
    try:
        import sqlite_vec  # noqa: F401, PLC0415
    except ImportError:
        logger.warning("sqlite-vec not installed — falling back to numpy brute-force search")
        return False

    # Run the load on aiosqlite's worker thread (the thread that owns the
    # raw connection), not the calling coroutine's thread.  ``_execute``
    # queues ``fn(*args)`` onto that worker thread and awaits the result.
    # aiosqlite ships ``py.typed`` but leaves ``_execute`` unannotated, so
    # --strict flags the call as untyped; the loader itself is fully typed.
    try:
        # aiosqlite's private _execute is untyped; the loader passed to it is not.
        await db._execute(_load_sqlite_vec_sync, db._conn)  # type: ignore[no-untyped-call]  # noqa: SLF001
    except (AttributeError, OSError, sqlite3.Error) as exc:
        logger.warning("Cannot load sqlite-vec extension: %s — falling back to numpy", exc)
        return False
    else:
        return True

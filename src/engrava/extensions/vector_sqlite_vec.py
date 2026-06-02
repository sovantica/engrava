"""SqliteVecSearchBackend — ANN vector search via sqlite-vec.

Drop-in replacement for the brute-force numpy cosine similarity search
in ``SqliteEngravaCore``.  When ``sqlite-vec`` is installed and its
extension is loaded, ``search_similar()`` delegates to the ``vec0``
virtual table for O(log n) approximate nearest-neighbor queries.

If sqlite-vec is unavailable at runtime the store falls back to the
existing numpy implementation — no crash, just a warning log.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SqliteVecSearchBackend:
    """ANN vector search backend backed by a ``vec0`` virtual table.

    Lifecycle:
        1. ``ensure_index(db, dimension)`` — creates the virtual table.
        2. ``sync_embeddings(db)`` — backfills existing rows.
        3. ``search(db, query_vector, ...)`` — runs ANN queries.

    All state is kept in SQLite; this class is stateless aside from
    the cached ``dimension``.

    Args:
        dimension: Embedding vector length (e.g. 384 for MiniLM).

    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

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
        if count:
            await db.commit()
        return count

    async def search(
        self,
        db: aiosqlite.Connection,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """ANN search via sqlite-vec ``vec0`` virtual table.

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

        Used by ``store_embedding()`` to keep the ANN index in sync
        after each write to the ``embedding`` table.

        Args:
            db: Active database connection with sqlite-vec loaded.
            rowid: The rowid from the ``embedding`` table.
            vector: Embedding vector as a list of floats.

        """
        vec_json = "[" + ",".join(str(f) for f in vector) + "]"
        # DELETE + INSERT instead of INSERT OR REPLACE because vec0
        # virtual tables may not support ON CONFLICT semantics.
        await db.execute(
            "DELETE FROM embedding_vec WHERE rowid = ?",
            (rowid,),
        )
        await db.execute(
            "INSERT INTO embedding_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, vec_json),
        )


def _load_sqlite_vec_sync(raw_conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into a raw sqlite3 connection.

    This runs the ``enable_load_extension`` / ``sqlite_vec.load`` /
    ``enable_load_extension(False)`` sequence synchronously.  It **must**
    execute on the thread that owns ``raw_conn`` (sqlite3 enforces a
    same-thread guard); callers route it onto aiosqlite's worker thread
    via :func:`load_sqlite_vec`.

    Args:
        raw_conn: The underlying ``sqlite3.Connection`` to load into.

    """
    import sqlite_vec  # noqa: PLC0415

    raw_conn.enable_load_extension(True)  # noqa: FBT003
    sqlite_vec.load(raw_conn)
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
        await db._execute(_load_sqlite_vec_sync, db._conn)  # type: ignore[no-untyped-call]  # noqa: SLF001
    except (AttributeError, OSError, sqlite3.Error) as exc:
        logger.warning("Cannot load sqlite-vec extension: %s — falling back to numpy", exc)
        return False
    else:
        return True

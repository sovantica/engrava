"""JournalWriter — append-only hash-linked mutation log.

Provides SHA-256 hash-chained recording of thought-graph mutations
with delta capture.  Designed for tamper-evident audit trails.

The writer operates on a single ``aiosqlite.Connection`` and expects
the ``journal_entry`` table to already exist (created by schema
migration core-6).
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import uuid as _uuid
from sqlite3 import IntegrityError
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakKeyDictionary

from engrava.domain.models.journal import JournalEntry, JournalIntegrityResult

if TYPE_CHECKING:
    import aiosqlite

    from engrava.infrastructure.sqlite.connection_revocation import ConnectionRevocationToken


class JournalWriter:
    """Append-only writer for hash-linked journal entries.

    Each call to :meth:`append` inserts a new row into the
    ``journal_entry`` table, computing the SHA-256 hash from the
    canonical representation and linking it to the previous entry's
    hash.  The sequence number is monotonically increasing and gapless.

    Args:
        db: An open aiosqlite connection with the ``journal_entry``
            table already created.
        revocation: Optional shared revocation token. When the owning store
            quarantines the connection it revokes this token, and every
            connection-touching method here then fails hard with
            :class:`~engrava.domain.exceptions.ConnectionQuarantinedError` —
            so this holder of the real connection cannot bypass the store's
            terminal quarantine. ``None`` (standalone use) disables the check.

    Examples:
        >>> writer = JournalWriter(db)  # doctest: +SKIP
        >>> entry = await writer.append(
        ...     mutation_type="INSERT_THOUGHT",
        ...     target_id="thought-001",
        ...     delta={"before": None, "after": {"essence": "hello"}},
        ... )

    """

    # Process-global registry that serialises appends across every writer
    # sharing one connection (aiosqlite exposes no row-level locking, so two
    # writers on the same connection must contend on the same lock or they race
    # the gapless-sequence read/insert). Keyed **by the connection object** via a
    # weak-key map so each entry is reclaimed automatically when its connection is
    # garbage-collected: the registry cannot grow without bound as long-lived
    # processes open and drop transient connections, and — because keys are object
    # identity, not ``id(connection)`` — a fresh connection can never inherit a
    # dead connection's stale lock through address reuse.
    _connection_locks: ClassVar[WeakKeyDictionary[aiosqlite.Connection, asyncio.Lock]] = (
        WeakKeyDictionary()
    )

    def __init__(
        self,
        db: aiosqlite.Connection,
        *,
        revocation: ConnectionRevocationToken | None = None,
    ) -> None:
        self._db = db
        self._revocation = revocation
        self._lock = self._connection_locks.setdefault(db, asyncio.Lock())

    def _check_revoked(self) -> None:
        """Fail hard if the shared connection has been revoked (quarantined).

        Raises:
            ConnectionQuarantinedError: When the revocation token is revoked.

        """
        if self._revocation is not None:
            self._revocation.check()

    async def _get_latest_entry_state(self) -> tuple[int, str | None]:
        """Read the current chain tail from the database.

        Returns:
            Tuple of ``(last_sequence_number, last_entry_hash)``.

        """
        cursor = await self._db.execute(
            "SELECT sequence_number, entry_hash "
            "FROM journal_entry ORDER BY sequence_number DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is not None:
            return int(row["sequence_number"]), row["entry_hash"]
        return 0, None

    @staticmethod
    def compute_hash(
        sequence_number: int,
        mutation_type: str,
        target_id: str | None,
        delta: dict[str, object],
        parent_hash: str | None,
    ) -> str:
        """Compute the SHA-256 hash for a journal entry.

        The canonical representation is::

            "{sequence_number}|{mutation_type}|{target_id or ''}|
             {json.dumps(delta, sort_keys=True)}|{parent_hash or ''}"

        The preimage therefore binds an entry's **ordering** (``sequence_number``,
        ``parent_hash``) and its **content** (``mutation_type``, ``target_id``,
        ``delta``). It does **not** include ``created_at`` or ``entry_id``: a
        journal whose timestamps have all been rewritten still verifies as
        ``valid=True``, and moving a timestamp across a :meth:`get_entries`
        ``since`` bound changes that window either way — downward hides the entry,
        upward plants it — because the filter reads the same uncovered column.
        Timestamps are informative, not tamper-evident.

        Args:
            sequence_number: Monotonic sequence number of this entry.
            mutation_type: The mutation classification string.
            target_id: Target entity ID, or ``None``.
            delta: The before/after diff dictionary.
            parent_hash: Hash of the previous entry, or ``None``.

        Returns:
            Lowercase hex SHA-256 digest.

        Examples:
            >>> JournalWriter.compute_hash(
            ...     1, "INSERT_THOUGHT", "t-001",
            ...     {"before": None, "after": {}}, None,
            ... )[:8]
            '...'

        """
        canonical = (
            f"{sequence_number}|{mutation_type}|{target_id or ''}|"
            f"{json.dumps(delta, sort_keys=True)}|{parent_hash or ''}"
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def append(
        self,
        mutation_type: str,
        target_id: str | None,
        delta: dict[str, object],
    ) -> JournalEntry:
        """Record a mutation as a new hash-linked journal entry.

        Computes the SHA-256 hash, links to the previous entry, and
        inserts atomically.  The caller is responsible for committing
        the transaction (or relying on the store's ``_maybe_commit``).

        Args:
            mutation_type: One of INSERT_THOUGHT, UPDATE_THOUGHT,
                DELETE_THOUGHT, INSERT_EDGE, UPDATE_EDGE, DELETE_EDGE, or
                UPDATE_ACTION (an action ``status``/``verification_status``
                state-transition). Action *creation* is not journaled.
            target_id: The ``thought_id``, ``edge_id``, or ``action_id``
                affected.
            delta: JSON-serializable diff
                ``{"before": {...}, "after": {...}}``.

        Returns:
            The persisted ``JournalEntry`` with computed hash.

        Raises:
            ConnectionQuarantinedError: When the shared connection has been
                revoked (the owning store quarantined it).

        """
        self._check_revoked()
        for _attempt in range(5):
            async with self._lock:
                last_sequence, parent_hash = await self._get_latest_entry_state()
                sequence_number = last_sequence + 1
                entry_hash = self.compute_hash(
                    sequence_number,
                    mutation_type,
                    target_id,
                    delta,
                    parent_hash,
                )

                entry_id = str(_uuid.uuid4())
                now = datetime.datetime.now(datetime.UTC).isoformat()

                try:
                    await self._db.execute(
                        "INSERT INTO journal_entry "
                        "(entry_id, sequence_number, mutation_type, target_id, "
                        " delta, parent_hash, entry_hash, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            entry_id,
                            sequence_number,
                            mutation_type,
                            target_id,
                            json.dumps(delta, sort_keys=True),
                            parent_hash,
                            entry_hash,
                            now,
                        ),
                    )
                except IntegrityError as exc:
                    if "journal_entry.sequence_number" not in str(exc):
                        raise
                    continue

                return JournalEntry(
                    entry_id=entry_id,
                    sequence_number=sequence_number,
                    mutation_type=mutation_type,
                    target_id=target_id,
                    delta=delta,
                    parent_hash=parent_hash,
                    entry_hash=entry_hash,
                    created_at=now,
                )

        msg = "Failed to append journal entry after 5 retries due to sequence contention"
        raise RuntimeError(msg)

    async def verify_integrity(self) -> JournalIntegrityResult:
        """Verify the entire hash chain of the journal.

        Iterates every entry in sequence order, recomputes each hash,
        and validates parent-hash linkage.

        The walk proves the **ordering and content** of the rows it read — it
        proves nothing about their ``created_at`` values, which are outside the
        hash preimage (see :meth:`compute_hash`), nor about the chain's length
        (a removed tail leaves a verifying prefix).

        Returns:
            A ``JournalIntegrityResult`` describing chain validity.

        Raises:
            ConnectionQuarantinedError: When the shared connection has been
                revoked (the owning store quarantined it).

        """
        self._check_revoked()
        cursor = await self._db.execute(
            "SELECT entry_id, sequence_number, mutation_type, target_id, "
            "       delta, parent_hash, entry_hash, created_at "
            "FROM journal_entry ORDER BY sequence_number ASC"
        )
        rows = await cursor.fetchall()

        if not rows:
            return JournalIntegrityResult(valid=True, entries_checked=0)

        prev_hash: str | None = None
        checked = 0

        for row in rows:
            checked += 1
            seq = int(row["sequence_number"])
            mutation_type: str = row["mutation_type"]
            target_id: str | None = row["target_id"]
            delta: dict[str, object] = json.loads(row["delta"])
            parent_hash: str | None = row["parent_hash"]
            stored_hash: str = row["entry_hash"]

            # Verify parent linkage.
            if parent_hash != prev_hash:
                return JournalIntegrityResult(
                    valid=False,
                    entries_checked=checked,
                    first_invalid_sequence=seq,
                    error_message=(
                        f"Parent hash mismatch at sequence {seq}: "
                        f"expected {prev_hash!r}, got {parent_hash!r}"
                    ),
                )

            # Recompute and verify entry hash.
            expected_hash = self.compute_hash(
                seq,
                mutation_type,
                target_id,
                delta,
                parent_hash,
            )
            if stored_hash != expected_hash:
                return JournalIntegrityResult(
                    valid=False,
                    entries_checked=checked,
                    first_invalid_sequence=seq,
                    error_message=(
                        f"Hash mismatch at sequence {seq}: "
                        f"stored {stored_hash!r}, computed {expected_hash!r}"
                    ),
                )

            prev_hash = stored_hash

        return JournalIntegrityResult(valid=True, entries_checked=checked)

    async def get_entries(
        self,
        target_id: str | None = None,
        mutation_type: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[JournalEntry]:
        """Query journal entries with optional filters.

        ``since`` is a convenience filter, **not** an audit boundary: it compares
        ``created_at >= since``, and ``created_at`` is outside the hash preimage
        (as is ``entry_id`` — see :meth:`compute_hash`). Rewriting an entry's
        ``created_at`` across ``since`` drops it out of the window or adds it in,
        and either way leaves the chain verifying: an empty result does not prove
        that nothing happened in that period, and a returned entry does not prove
        that it happened within it.

        Args:
            target_id: Filter by target entity ID.
            mutation_type: Filter by mutation type string.
            since: ISO-8601 timestamp lower bound (inclusive) on the
                chain-uncovered ``created_at`` column.
            limit: Maximum number of entries to return.

        Returns:
            List of matching ``JournalEntry`` objects ordered by
            ``sequence_number`` ascending.

        Raises:
            ConnectionQuarantinedError: When the shared connection has been
                revoked (the owning store quarantined it).

        """
        self._check_revoked()
        clauses: list[str] = []
        params: list[object] = []

        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        if mutation_type is not None:
            clauses.append("mutation_type = ?")
            params.append(mutation_type)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT entry_id, sequence_number, mutation_type, target_id, "  # noqa: S608
            "       delta, parent_hash, entry_hash, created_at "
            f"FROM journal_entry{where} "
            "ORDER BY sequence_number ASC LIMIT ?"
        )
        params.append(limit)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()

        return [
            JournalEntry(
                entry_id=row["entry_id"],
                sequence_number=int(row["sequence_number"]),
                mutation_type=row["mutation_type"],
                target_id=row["target_id"],
                delta=json.loads(row["delta"]),
                parent_hash=row["parent_hash"],
                entry_hash=row["entry_hash"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

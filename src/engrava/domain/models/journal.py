"""Journal value objects for the hash-chain audit log.

Immutable data structures representing journal entries and integrity
verification results.  Used by ``JournalWriter`` to record and verify
mutation history.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JournalEntry:
    """A single entry in the hash-linked mutation journal.

    Each entry records one mutation (INSERT/UPDATE/DELETE) and is
    cryptographically linked to the previous entry via SHA-256.

    Attributes:
        entry_id: Stable UUID identity for this entry.
        sequence_number: Monotonic, gapless sequence (starts at 1).
        mutation_type: One of INSERT_THOUGHT, UPDATE_THOUGHT,
            DELETE_THOUGHT, INSERT_EDGE, UPDATE_EDGE, DELETE_EDGE.
        target_id: The ``thought_id`` or ``edge_id`` affected (nullable
            for bulk operations, but always set in current impl).
        delta: JSON-serializable diff ``{"before": {...}, "after": {...}}``.
        parent_hash: SHA-256 hex digest of the previous entry, or ``None``
            for the very first entry in the chain.
        entry_hash: SHA-256 hex digest of this entry's canonical content.
        created_at: ISO-8601 UTC timestamp of creation.

    Examples:
        >>> entry = JournalEntry(
        ...     entry_id="abc-123",
        ...     sequence_number=1,
        ...     mutation_type="INSERT_THOUGHT",
        ...     target_id="thought-001",
        ...     delta={"before": None, "after": {"essence": "hello"}},
        ...     parent_hash=None,
        ...     entry_hash="deadbeef...",
        ...     created_at="2026-04-12T10:00:00+00:00",
        ... )
        >>> entry.sequence_number
        1

    """

    entry_id: str
    sequence_number: int
    mutation_type: str
    target_id: str | None
    delta: dict[str, object]
    parent_hash: str | None
    entry_hash: str
    created_at: str


@dataclass(frozen=True)
class JournalIntegrityResult:
    """Result of a full journal hash-chain verification.

    Attributes:
        valid: ``True`` if every entry's hash matches the recomputed value
            and the parent-hash linkage is intact.
        entries_checked: Total number of journal entries verified.
        first_invalid_sequence: Sequence number of the first broken entry,
            or ``None`` when the chain is fully valid.
        error_message: Human-readable description of the first error found,
            or ``None`` when valid.

    Examples:
        >>> result = JournalIntegrityResult(valid=True, entries_checked=42)
        >>> result.valid
        True

    """

    valid: bool
    entries_checked: int
    first_invalid_sequence: int | None = None
    error_message: str | None = None

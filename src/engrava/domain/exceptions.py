"""Core domain exceptions for engrava.

Typed exception hierarchy for thought-graph operations. Every exception
carries structured context for logging and debugging.
"""

from __future__ import annotations


class EngravaError(Exception):
    """Base exception for all engrava domain errors.

    Args:
        message: Human-readable error description.

    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidTransitionError(EngravaError):
    """Raised when an invalid state transition is attempted.

    Args:
        entity_type: Name of the entity type (e.g., 'LifecycleStatus').
        current_state: The current state value.
        target_state: The attempted target state value.

    """

    def __init__(self, entity_type: str, current_state: str, target_state: str) -> None:
        self.entity_type = entity_type
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(f"Invalid {entity_type} transition: {current_state} -> {target_state}")


class ThoughtNotFoundError(EngravaError):
    """Raised when a thought record is not found.

    Args:
        thought_id: The ID that was not found.

    """

    def __init__(self, thought_id: str) -> None:
        self.thought_id = thought_id
        super().__init__(f"Thought not found: {thought_id}")


class ActionNotFoundError(EngravaError):
    """Raised when an action record is not found.

    Args:
        action_id: The ID that was not found.

    """

    def __init__(self, action_id: str) -> None:
        self.action_id = action_id
        super().__init__(f"Action not found: {action_id}")


class StaleDataError(EngravaError):
    """Raised when a guarded write matches no row.

    Two situations produce that, and this error does not tell them apart:
    another writer stamped a new ``updated_cycle`` between this operation's read
    and its write, or the row was deleted in that window. Either way nothing of
    the operation was applied.

    It is narrower than "the row changed" — nothing in engrava advances
    ``updated_cycle`` on its own, so an ordinary competing edit passes the guard
    and overwrites rather than raising — and also broader, because a delete
    raises it too. Recover by re-reading the record (which may now be gone) and
    recomputing the change.

    Args:
        entity_type: Type of entity (e.g., 'ThoughtRecord').
        entity_id: Identifier of the entity.
        expected_version: The ``updated_cycle`` the caller expected.

    """

    def __init__(self, entity_type: str, entity_id: str, expected_version: int) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        super().__init__(
            f"Stale data: {entity_type} {entity_id} was modified since version {expected_version}"
        )


class ReadOnlyViolationError(EngravaError):
    """Raised when a write operation is attempted on a read-only store.

    Used by ``ReadOnlyEngrava`` to enforce read-only access in
    read-only contexts that must not mutate the thought graph.

    Args:
        operation: Name of the blocked write operation.

    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"Write operation blocked on read-only store: {operation}")


class EmbeddingModelMismatchError(EngravaError):
    """Raised when the configured embedding model differs from the stored one.

    Once a database has embeddings for a specific model, opening it with
    a different model would produce incompatible vectors. This error
    enforces model immutability at the database level.

    Args:
        stored_model: Model name recorded in ``_metadata``.
        configured_model: Model name from the current provider.
        stored_dimension: Dimension recorded in ``_metadata``.
        configured_dimension: Dimension from the current provider.

    """

    def __init__(
        self,
        stored_model: str,
        configured_model: str,
        stored_dimension: int,
        configured_dimension: int,
    ) -> None:
        self.stored_model = stored_model
        self.configured_model = configured_model
        self.stored_dimension = stored_dimension
        self.configured_dimension = configured_dimension
        super().__init__(
            f"Embedding model mismatch: database has '{stored_model}' "
            f"(dim={stored_dimension}), but provider offers '{configured_model}' "
            f"(dim={configured_dimension}). Use a new database or matching provider."
        )


class VectorDimensionMismatchError(EngravaError):
    """Raised when a query vector's length differs from the store's dimension.

    :meth:`SqliteEngravaCore.search_similar` computes cosine similarity between
    the query vector and every stored embedding, an operation that is only
    defined when the two share a dimension. A query vector of the wrong length
    is a structural caller-contract violation — not a benign query that happens
    to match nothing — so it is rejected loudly with this typed error instead of
    silently returning an empty result (which would hide the mistake and let a
    caller believe the corpus simply had no neighbours). The check is dimension
    only: it fires regardless of the vector's magnitude, so a wrong-length
    all-zero vector is still a dimension error rather than a degenerate-vector
    degradation.

    Args:
        expected: The embedding dimension the store expects (from its configured
            vector backend, embedding provider, or stored embeddings).
        actual: The length of the offending query vector.

    Examples:
        >>> raise VectorDimensionMismatchError(expected=384, actual=383)
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.VectorDimensionMismatchError: \
query vector dimension mismatch: store expects 384, got 383

    """

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"query vector dimension mismatch: store expects {expected}, got {actual}")


class EmbeddingQueryPrefixMismatchError(EngravaError):
    """Raised when the active query prefix diverges from the corpus pairing.

    For an asymmetric embedding model, a query is encoded with a
    ``query_prefix`` (e.g. ``"query: "``) that must pair with the
    ``document_prefix`` (e.g. ``"passage: "``) the stored corpus was
    embedded with. Changing only the query prefix does not alter any stored
    vector, so it does not require re-embedding — but querying with a prefix
    that no longer matches the corpus silently degrades ranking. This error
    surfaces that divergence loudly at search time. The fix is to restore the
    query prefix the corpus was built to pair with, or to deliberately
    re-embed the corpus with a new document prefix.

    Args:
        stored_query_prefix: Query prefix the corpus was built to pair with.
        configured_query_prefix: Query prefix the current provider offers.

    """

    def __init__(
        self,
        stored_query_prefix: str,
        configured_query_prefix: str,
    ) -> None:
        self.stored_query_prefix = stored_query_prefix
        self.configured_query_prefix = configured_query_prefix
        super().__init__(
            f"Embedding query prefix mismatch: corpus was built to pair with "
            f"query prefix {stored_query_prefix!r}, but the provider offers "
            f"{configured_query_prefix!r}. Restore the matching query prefix, "
            f"or deliberately re-embed the corpus with a new document prefix."
        )


class EmbeddingGenerationError(EngravaError):
    """Raised when auto-embedding a thought fails under strict mode.

    Auto-embed runs *after* ``create_thought`` (and the batch path)
    has already committed the thought, so a provider failure would
    otherwise leave the thought persisted without an embedding — a
    silent, torn write invisible to vector search. By default the store
    logs a ``WARNING`` naming the thought and re-raises the provider's
    own exception (behaviour is unchanged for existing callers). When
    the operator opts in via ``embeddings.require_embedding = true`` (or
    ``require_embedding=True`` on the store), that failure is instead
    normalised into this typed error — an explicit fail-fast signal that
    the thought is persisted but unembedded.

    Args:
        thought_id: UUID of the thought whose embedding failed.
        message: Human-readable description of the underlying failure,
            typically derived from the provider's own exception.

    """

    def __init__(self, thought_id: str, message: str) -> None:
        self.thought_id = thought_id
        super().__init__(
            f"Failed to auto-embed thought {thought_id}: {message}. "
            f"The thought is persisted but has no embedding and is not "
            f"reachable by vector search."
        )


class ReferentialIntegrityError(EngravaError):
    """Raised when a write would create or leave an orphan reference.

    The store enforces ON DELETE CASCADE between thought and its child
    tables (edge, embedding, action) at the SQLite level (core-12+).
    Inserts that name a non-existent parent thought are surfaced as
    this domain exception so callers do not have to interpret the
    raw ``sqlite3.IntegrityError`` text.

    Args:
        entity_type: Child entity that violated the FK (e.g., ``"edge"``).
        column: Column carrying the dangling reference (e.g.,
            ``"from_thought_id"``).
        referenced_id: Identifier value that was supposed to resolve
            to a parent thought but did not.

    Examples:
        >>> raise ReferentialIntegrityError("edge", "from_thought_id", "ghost")
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.ReferentialIntegrityError: \
referential integrity violation: edge.from_thought_id='ghost' \
does not reference an existing thought

    """

    def __init__(self, entity_type: str, column: str, referenced_id: str) -> None:
        self.entity_type = entity_type
        self.column = column
        self.referenced_id = referenced_id
        super().__init__(
            f"referential integrity violation: {entity_type}.{column}="
            f"{referenced_id!r} does not reference an existing thought",
        )


class DuplicateEdgeError(EngravaError):
    """Raised when a directed, typed edge relationship already exists.

    Edge identity is constrained by ``(from_thought_id, to_thought_id,
    edge_type)`` independently of the caller-supplied ``edge_id``. This typed
    boundary lets callers handle idempotent graph writes without depending on
    SQLite error text.

    Args:
        from_thought_id: Source endpoint of the duplicate relationship.
        to_thought_id: Target endpoint of the duplicate relationship.
        edge_type: Edge type value participating in the uniqueness key.

    """

    def __init__(self, from_thought_id: str, to_thought_id: str, edge_type: str) -> None:
        self.from_thought_id = from_thought_id
        self.to_thought_id = to_thought_id
        self.edge_type = edge_type
        super().__init__(
            "edge relationship already exists: "
            f"{from_thought_id!r} -[{edge_type}]-> {to_thought_id!r}",
        )


class InvalidFilterError(EngravaError):
    """Raised when a metadata/visibility filter is invalid at construction.

    Covers value-domain violations (a non-finite float, an out-of-range
    integer, a scalar of an unsupported type), an empty
    :class:`~engrava.domain.models.filters.VisibilityQueryFilter`, an
    unsupported operator, and a predicate count beyond the documented
    maximum. The error is always raised at *filter construction* — never
    mid-query — so a compiled filter that reaches the store is known-good.

    Args:
        message: Human-readable description of why the filter is invalid.

    """


class InvalidFilterPathError(EngravaError):
    r"""Raised when a filter path does not match the allowed grammar.

    A predicate path must be a JSONPath of the restricted shape
    ``$``, ``$.key`` or ``$[0]`` (dot-identifiers and bracketed array
    indices only), validated against
    ``^\$(\.[A-Za-z0-9_]+|\[[0-9]+\])*$``. Anything else (a bare key
    with no ``$`` root, a wildcard, a quoted segment, an injection
    attempt) is rejected at filter construction.

    Args:
        path: The offending path string.

    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(
            f"invalid filter path {path!r}: paths must match "
            r"^\$(\.[A-Za-z0-9_]+|\[[0-9]+\])*$ "
            "(e.g. '$', '$.session_id', '$.tags[0]')",
        )


class JournalIntegrityError(EngravaError):
    """Raised when an on-open journal integrity check finds a broken chain.

    Surfaced only when ``journal.verify_on_open`` is enabled: opening a
    store via :meth:`SqliteEngravaCore.from_config` re-walks the persisted
    hash chain after the schema is ensured, and raises this error instead
    of returning a store when the chain does not verify. Carries the same
    diagnostics as :class:`~engrava.domain.models.journal.JournalIntegrityResult`
    so callers can report exactly where the chain first broke.

    Args:
        first_invalid_sequence: Sequence number of the first broken entry,
            or ``None`` when the break is not attributable to a single row.
        error_message: Human-readable description of the first inconsistency.

    Examples:
        >>> raise JournalIntegrityError(3, "Hash mismatch at sequence 3")
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.JournalIntegrityError: \
journal integrity check failed at sequence 3: Hash mismatch at sequence 3

    """

    def __init__(
        self,
        first_invalid_sequence: int | None,
        error_message: str | None,
    ) -> None:
        self.first_invalid_sequence = first_invalid_sequence
        self.error_message = error_message
        location = (
            f"at sequence {first_invalid_sequence}"
            if first_invalid_sequence is not None
            else "(sequence unknown)"
        )
        detail = error_message or "chain verification failed"
        super().__init__(f"journal integrity check failed {location}: {detail}")


class ExtensionMigrationError(EngravaError):
    """Raised when an extension schema migration fails.

    Covers SQL execution failures, extension downgrade detection, and
    path-resolution errors.

    Args:
        extension_name: Name of the extension (from ``ExtensionManifest.name``).
        message: Human-readable description of the failure.
        migration_file: Basename of the migration file that triggered the error,
            or ``None`` when the error is not file-specific.

    Examples:
        >>> raise ExtensionMigrationError("my-plugin", "table already exists")
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.ExtensionMigrationError: \
[extension='my-plugin'] table already exists

    """

    def __init__(
        self,
        extension_name: str,
        message: str,
        migration_file: str | None = None,
    ) -> None:
        self.extension_name = extension_name
        self.migration_file = migration_file
        prefix = f"[extension={extension_name!r}"
        if migration_file:
            prefix += f", file={migration_file!r}"
        prefix += "]"
        super().__init__(f"{prefix} {message}")


class CoreMigrationError(EngravaError):
    """Raised when a core schema migration step fails its postcondition.

    A migration step that returns without leaving its target schema structure
    (a column, table, index, or foreign key) in place would otherwise let the
    upgrade loop stamp a higher ``user_version`` over a schema that does not
    actually carry the migrated structure — a permanently "false-current"
    database that skips the migration on every future open. Raising here leaves
    the version at the last fully-applied step, so the next open retries the
    remaining steps.

    Args:
        target_version: The core schema version the failed step targets.
        message: Human-readable description of the unmet postcondition.

    Examples:
        >>> raise CoreMigrationError(8, "idx_edge_type_from missing after create")
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.CoreMigrationError: \
[core schema v8] idx_edge_type_from missing after create

    """

    def __init__(self, target_version: int, message: str) -> None:
        self.target_version = target_version
        super().__init__(f"[core schema v{target_version}] {message}")


class DerivedRecordError(EngravaError):
    """Raised when the derived-records extension seam rejects a producer result.

    Covers the deterministic, content-independent failure modes of the seam:
    a returned sequence that exceeds ``DeriveGates.max_derived_per_source``,
    or a derived record whose core-assigned identity would collide with its
    own source thought (a self-referential ``DERIVED_FROM`` edge). It is
    **not** used for producer-internal exceptions (those propagate or are
    logged verbatim per ``DeriveGates.on_error``).

    Args:
        source_thought_id: Identity of the source thought whose derivation
            was rejected.
        message: Human-readable description of the rejection.

    Examples:
        >>> raise DerivedRecordError("t-1", "over cap")
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.DerivedRecordError: [source='t-1'] over cap

    """

    def __init__(self, source_thought_id: str, message: str) -> None:
        self.source_thought_id = source_thought_id
        super().__init__(f"[source={source_thought_id!r}] {message}")


class SourceThoughtNotFoundError(EngravaError):
    """Raised when an explicit backfill targets a source thought that is absent.

    Surfaced by the derived-records backfill entry point when the requested
    source ``thought_id`` does not exist in the store, so there is nothing to
    derive from. It is a precondition/contract failure distinct from the clean,
    empty result the backfill returns for an *ineligible* (already-derived)
    source — a missing id is an error, an ineligible source is a no-op.

    Args:
        thought_id: The source ID that was not found.

    Examples:
        >>> raise SourceThoughtNotFoundError("t-404")
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.SourceThoughtNotFoundError: \
source thought not found: 't-404'

    """

    def __init__(self, thought_id: str) -> None:
        self.thought_id = thought_id
        super().__init__(f"source thought not found: {thought_id!r}")


class CycleProviderError(EngravaError):
    """Raised when a configured cycle provider returns an invalid value.

    A ``runtime_checkable``
    :class:`~engrava.domain.protocols.cycle_provider.CycleProvider` protocol
    verifies only that an object *has* a ``current_cycle()`` method — never that
    the value it returns is a usable cognitive cycle. The store therefore
    validates the pulled value at the resolution boundary: it must be a real
    ``int`` (a ``bool`` is rejected even though it subclasses ``int``) and it
    must be non-negative (matching the ``created_cycle`` / ``updated_cycle``
    ``ge=0`` invariant). A value failing either check is a provider-contract
    violation surfaced as this typed error, rather than a bare ``TypeError`` /
    ``ValueError`` leaking out of downstream recency arithmetic.

    Engrava can only enforce value *validity* here — a provider's *purity*
    (that its value is not wall-clock-derived and does not conflate the
    operation-count / cognitive-cycle / wall-time axes) is the configuring
    consumer's contract, not something the core can check.

    Args:
        reason: Human-readable description of why the pulled value is invalid.

    Examples:
        >>> str(CycleProviderError("expected int, got str"))
        'invalid cycle provider value: expected int, got str'

    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"invalid cycle provider value: {reason}")


class RecencyModeConflictError(EngravaError):
    """Raised when a query explicitly supplies both recency reference axes.

    Engrava ranks recency along two separately-typed axes, and a single query
    selects **exactly one** reference:

    * **cognitive-cycle recency** — chosen by an explicit ``current_cycle``, or
      (when neither reference is passed) pulled from a configured cycle provider;
    * **transaction-time recency** — chosen by a caller-supplied ``recency_now``
      instant, which **takes precedence over a passive cycle provider**.

    The two axes measure age against incomparable clocks (a cognitive tick count
    versus wall-clock time) and are **never silently combined**. The conflict
    fires only when the caller **explicitly** supplies **both** ``current_cycle``
    and ``recency_now``; a configured cycle provider is passive and yields to an
    explicit ``recency_now`` (it is not consulted), so a provider-configured
    store *can* still use transaction-time recency. To do so, pass ``recency_now``
    and omit an explicit ``current_cycle`` — the provider is then not consulted.

    Args:
        reason: Human-readable description of the conflicting request.

    Examples:
        >>> str(RecencyModeConflictError("current_cycle and recency_now both set"))
        'conflicting recency references: current_cycle and recency_now both set'

    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"conflicting recency references: {reason}")


class InvalidRecencyArgumentError(EngravaError):
    """Raised when a transaction-time recency argument is malformed.

    Covers a ``recency_now`` that is not a valid ISO-8601 timestamp and a
    non-positive ``recency_now_half_life``. It is raised at the ``search_hybrid``
    / ``recall`` call boundary (never mid-ranking), a typed sibling of
    :class:`RecencyModeConflictError` so callers can catch a specific engrava
    error rather than a bare ``ValueError``. For a malformed timestamp the
    underlying parser error is chained as the ``__cause__``.

    Args:
        message: Human-readable description of the invalid argument.

    Examples:
        >>> raise InvalidRecencyArgumentError("recency_now must be ISO-8601")
        Traceback (most recent call last):
            ...
        engrava.domain.exceptions.InvalidRecencyArgumentError: \
recency_now must be ISO-8601

    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ConnectionQuarantinedError(EngravaError):
    """Raised when the store's connection has been quarantined and is unusable.

    A long-lived SQLite connection is quarantined when a compensating rollback
    could not be guaranteed to complete — most critically when a cancellation
    interrupted a per-child rollback in the derived-records seam and the
    rollback ultimately failed, so the connection may still hold an open
    transaction. Continuing to use such a connection could flush an orphaned
    partial write or run later operations on an indeterminate transaction, so
    every public operation fails fast with this error instead. The condition is
    terminal for the store instance: a new store over a fresh connection must
    be constructed to recover.

    Scope: quarantine revokes *admission* — every NEW operation on the store or
    its journal fails fast with this error, so no write/commit can flush an
    orphaned transaction. It does not retract an operation already admitted
    before revocation: a reader admitted just before revocation may complete its
    in-flight read on the pre-revocation connection (a possibly-stale read,
    never a commit).

    Args:
        reason: Human-readable description of why the connection was quarantined.

    Examples:
        >>> str(ConnectionQuarantinedError("open transaction"))
        'connection quarantined: open transaction'

    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"connection quarantined: {reason}")

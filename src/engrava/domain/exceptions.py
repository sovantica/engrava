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


class StaleDataError(EngravaError):
    """Raised when an optimistic-concurrency check fails.

    The row was modified by another writer between read and write.

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

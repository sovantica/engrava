"""SqliteEngravaCore — core SQLite thought-graph persistence.

Provides async CRUD for thoughts, edges, actions, and brute-force
embedding similarity search.  All SQL uses parameterized queries.

The ``_row_to_thought`` method is an overridable template method —
subclasses can override it to produce richer model types while
reusing all core SQL logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import hashlib
import inspect
import json
import logging
import math
import re
import sqlite3
import struct
import unicodedata
import uuid as _uuid
from dataclasses import dataclass
from importlib import resources
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, Self

import aiosqlite
import numpy as np
import numpy.typing as npt

from engrava.config_validation import (
    ConfigError,
    own_str,
    require_exact_type,
    require_exact_type_or_none,
    require_int,
    require_int_or_none,
)
from engrava.domain.dreaming import (
    CENTROID_MODEL_NAME,
    ConsolidationResult,
    DreamingContext,
    compute_centroid,
)
from engrava.domain.enums import (
    ActionStatus,
    ActionType,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
    VerificationStatus,
)
from engrava.domain.exceptions import (
    ActionNotFoundError,
    ConnectionQuarantinedError,
    CoreMigrationError,
    CycleProviderError,
    DedupLockReentryError,
    DerivedRecordError,
    DuplicateEdgeError,
    EmbeddingGenerationError,
    EmbeddingModelMismatchError,
    EmbeddingProviderContractError,
    EmbeddingQueryPrefixMismatchError,
    InvalidRecencyArgumentError,
    InvalidTransitionError,
    JournalIntegrityError,
    RecencyModeConflictError,
    ReferentialIntegrityError,
    SchemaVersionError,
    SourceThoughtNotFoundError,
    StaleDataError,
    ThoughtNotFoundError,
    VectorDimensionMismatchError,
    WriteContentionError,
    WriteLockTimeoutError,
)
from engrava.domain.models._temporal import (
    canonical_timestamp,
    canonical_timestamp_or_none,
    parse_iso8601_to_utc,
    validate_interval_ordering,
    validate_iso8601_nullable,
)
from engrava.domain.models.action import ActionRecord
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.embedding import EmbeddingRecord
from engrava.domain.models.filters import _validate_path, compile_effective_predicate
from engrava.domain.models.journal import JournalIntegrityResult
from engrava.domain.models.provenance import ProvenanceContext
from engrava.domain.models.thought import MetadataValue, ThoughtRecord
from engrava.domain.models.ttl import CleanupResult, CleanupStrategy
from engrava.domain.protocols.derived_records import (
    DeriveContext,
    DerivedRecord,
    DerivedRecordProducerProtocol,
    DeriveGates,
    DeriveResult,
)
from engrava.domain.protocols.dreaming import DreamingConsolidatorProtocol
from engrava.domain.protocols.embedding_provider import RoleAwareEmbeddingProvider
from engrava.domain.protocols.hooks import DefaultEngravaHooks, EngravaHooksProtocol
from engrava.infrastructure.sqlite.connection_revocation import ConnectionRevocationToken
from engrava.infrastructure.sqlite.hygiene import (
    EvictionReason,
    HygieneResult,
    compute_active_hygiene_weights,
    compute_keep_score,
    has_active_usage_signal,
)
from engrava.infrastructure.sqlite.journal_writer import JournalWriter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable, Sequence

    from engrava.config import HygienePolicyConfig, MetricsConfig, SearchConfig
    from engrava.domain.manifest import ExtensionManifest
    from engrava.domain.models.filters import MetadataFilter, VisibilityQueryFilter
    from engrava.domain.models.metrics import EngravaMetrics, LatencyHistogram
    from engrava.domain.models.search import HybridSearchResult
    from engrava.domain.protocols.cycle_provider import CycleProvider
    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol
    from engrava.domain.protocols.hooks import MindQLExtension
    from engrava.infrastructure.sqlite.vector_sqlite_vec import SqliteVecSearchBackend
    from engrava.mindql.executor import MindQLResult
    from engrava.mindql.parser import MindQLQuery

logger = logging.getLogger(__name__)

#: Recursion guard for the derived-records extension seam. Set for the duration
#: of a ``derive_records`` dispatch and its per-child inserts; every write entry
#: point consults it so that a write issued *during* derivation (including one a
#: contract-violating producer performs) never dispatches a nested derivation.
#: Depth is thereby bounded to at most one. A ``ContextVar`` (not a plain
#: attribute) so the flag is task-local and safe under concurrent stores.
_IN_DERIVATION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "engrava_in_derivation",
    default=False,
)

#: Informational label naming the write operation that triggered derivation.
#: Purely descriptive (surfaced on ``DeriveContext.origin``); never consulted
#: for recursion control or authorization.
_DERIVATION_ORIGIN: contextvars.ContextVar[str] = contextvars.ContextVar(
    "engrava_derivation_origin",
    default="create_thought",
)

#: Informational ``DeriveContext.origin`` label for the explicit backfill entry
#: point (``derive_existing``), distinguishing a retroactive backfill from the
#: automatic on-store write operations. Purely descriptive — never consulted for
#: recursion control or gating.
_ORIGIN_DERIVE_EXISTING = "derive_existing"

#: Databases below this ``user_version`` predate the incremental migration
#: ladder and are bootstrapped from the full ``schema_core.sql`` (which stamps
#: the head version itself) rather than stepped. A database at or above it is
#: upgraded through the ordered core-migration registry.
_CORE_SCHEMA_BOOTSTRAP_FLOOR = 2

#: The core schema version ``ensure_schema`` upgrades a database to. Must
#: equal the target of the last entry in :meth:`SqliteEngravaCore._core_migration_steps`
#: — kept as its own constant (rather than read off the registry, which needs
#: an instance) so a caller that has not opened a store yet — the CLI's
#: schema-state gate, in particular — can compare a database's stamped
#: ``user_version`` against head without one.
CORE_SCHEMA_HEAD_VERSION = 21

#: Target version of the one core-migration step that manages its own
#: transaction boundaries rather than running inside the single explicit
#: transaction :meth:`SqliteEngravaCore._run_pending_core_migrations` opens for
#: every other step (see :meth:`SqliteEngravaCore._migrate_core_v11_to_v12` and
#: :meth:`SqliteEngravaCore._recreate_child_tables_with_fk_atomically`).
#: ``PRAGMA foreign_keys`` is a documented no-op while a transaction is open —
#: this step must toggle it to rebuild ``edge`` / ``embedding`` / ``action``
#: with their foreign keys — and that no-op behaviour was verified empirically
#: against SQLite 3.31.1 (2020, compiled from the upstream amalgamation) and
#: 3.53.1 (current) before writing this constant: it is a stable, long-standing
#: engine constraint, not a stale assumption inherited without checking.
_FK_RECREATE_TARGET_VERSION: Final = 12

#: Core tables whose presence on a sub-floor database means it is a real,
#: populated store rather than an empty file — see
#: :meth:`SqliteEngravaCore._has_any_core_table`.
_CORE_TABLE_NAMES = (
    "thought",
    "edge",
    "embedding",
    "action",
    "_metadata",
    "journal_entry",
    "extension_schema_versions",
)

#: The columns :func:`~engrava.domain.models._temporal.validate_iso8601_nullable`
#: covers, per table, as they stand at core-21: the ones the ``v20 -> v21`` step
#: rewrites once into the canonical UTC form (see
#: :meth:`SqliteEngravaCore._normalise_stored_timestamps`). Pinned to that
#: schema version on purpose — a migration must keep doing what it did when it
#: shipped, so a column validated in some later version is not added here.
_CANONICAL_TIMESTAMP_COLUMNS: Final = (
    (
        "thought",
        (
            "created_at",
            "updated_at",
            "last_accessed_at",
            "expires_at",
            "valid_from",
            "valid_until",
            "archived_at",
        ),
    ),
    ("edge", ("valid_from", "valid_until")),
)

#: SQLite ``GLOB`` patterns for the canonical UTC form: what
#: ``datetime.isoformat()`` writes for an aware UTC instant — a whole second, or
#: a six-digit fraction that is never ``.000000`` (``isoformat`` omits a zero
#: fraction). A stored value matching them already has the canonical shape and is
#: not read back, even if it names an impossible date; everything else in a
#: migrated column is.
_CANONICAL_WHOLE_SECOND_GLOB: Final = (
    "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00"
)
_CANONICAL_FRACTION_GLOB: Final = (
    "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]"
    ".[0-9][0-9][0-9][0-9][0-9][0-9]+00:00"
)
_ZERO_FRACTION_GLOB: Final = "*.000000+00:00"

#: Rows read per page while the ``v20 -> v21`` step rewrites values without the
#: canonical shape, so a large store is never held in memory at once.
_TIMESTAMP_NORMALISATION_BATCH_SIZE: Final = 500


@dataclass(frozen=True)
class _DerivationOutcome:
    """Per-source tally of derived-child persistence outcomes for one dispatch.

    Returned by the shared per-child dispatch loop so the explicit backfill
    entry point can report counts; the automatic on-store path computes the same
    tally but discards it (its callers observe derivation only through the store
    state). Private to this module — the public counterpart is
    :class:`~engrava.domain.protocols.derived_records.DeriveResult`.

    Attributes:
        created: Children newly inserted this dispatch.
        reused: Children that already existed and were reused (conflict-as-reuse).
        skipped: Children whose persistence failed under ``on_error="log"`` and
            were left for a later re-run (the source stays durable).

    """

    created: int = 0
    reused: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class _DeleteAtomicResult:
    """Outcome of :meth:`SqliteEngravaCore._delete_thought_atomic`.

    Two independent questions, not one: whether the *parent* thought was
    removed, and whether the call wrote *anything at all*. They can diverge —
    a ``thought_id`` that never matched a row still triggers an unconditional
    orphan sweep of any child rows a schema without FK enforcement is
    carrying, which is a real write even though ``deleted`` is ``False``.
    ``delete_thought`` / ``cleanup_expired`` need ``wrote_anything`` to decide
    whether they have anything of their own to commit; a per-branch boolean
    that only tracked ``deleted`` would silently roll back that sweep instead.

    Attributes:
        deleted: Whether the parent thought row was actually removed.
        wrote_anything: Whether this call's own return leaves any row's
            change intact — the parent, a swept orphan child, or both. A
            change a trigger made and this call then rolled back inside its
            own savepoint (a vetoed parent delete, see
            :meth:`SqliteEngravaCore._delete_thought_atomic`) does not count,
            even though a row was written for part of the call's execution.

    """

    deleted: bool
    wrote_anything: bool


@dataclass(frozen=True)
class _HygieneGcOutcome:
    """Outcome of :meth:`SqliteEngravaCore._hygiene_gc`, for ``run_hygiene``'s finalization.

    ``run_hygiene`` decides whether its archive+GC unit has a surviving write
    to commit from signals the stages report themselves — never from
    ``self._db.total_changes``, which is monotonic and still counts a write a
    savepoint later undid (see ``run_hygiene``'s own docstring). This carries
    the two signals :meth:`_hygiene_gc` alone can report, beyond the plain
    count already in :class:`~engrava.infrastructure.sqlite.hygiene.HygieneResult`.

    Attributes:
        gc_count: The number of thoughts physically deleted this stage —
            identical to the value ``run_hygiene`` puts on the public
            ``HygieneResult``.
        retired_count: How many orphan REFLECTIONs
            ``retire_orphan_reflections`` retired before any delete ran. A
            retirement is its own surviving write even when every GC
            candidate that follows it is vetoed or absent (``gc_count == 0``).
        wrote_anything: Whether any GC candidate's own
            :meth:`SqliteEngravaCore._delete_thought_atomic` call reported
            ``wrote_anything`` — its orphan-sweep case, which can be ``True``
            even for a candidate this stage does not count in ``gc_count``
            (``deleted`` is ``False``). Tracked the same way
            ``cleanup_expired`` already tracks it across its own delete loop.

    """

    gc_count: int
    retired_count: int
    wrote_anything: bool


#: Fixed namespaces for the deterministic identities the derived-records seam
#: assigns. A derived thought's ``thought_id`` is ``uuid5`` over its content, so
#: byte-identical derived content maps to one stored thought and re-running
#: derivation is idempotent; the provenance edge id is ``uuid5`` over its
#: endpoints + type, so a re-run reuses the same edge row.
_DERIVED_THOUGHT_NAMESPACE = _uuid.UUID("d1f5e6a2-3b4c-5d6e-8f90-a1b2c3d4e5f6")
_DERIVED_EDGE_NAMESPACE = _uuid.UUID("e2a6f7b3-4c5d-6e7f-9a01-b2c3d4e5f6a7")

#: Upper bound on a derived thought's ``essence`` (matches the
#: ``ThoughtRecord.essence`` field constraint).
_DERIVED_ESSENCE_MAX_CHARS = 200


def _essence_from_content(content: str) -> str:
    """Derive a compact ``essence`` preview from a derived record's content.

    The persisted thought stores the full ``content`` verbatim; the ``essence``
    is only a short preview (the ``ThoughtRecord.essence`` bound is
    :data:`_DERIVED_ESSENCE_MAX_CHARS` characters), so no information is lost by
    truncating it. Truncation is **best-effort combining-mark-aware**, not full
    Unicode grapheme-cluster segmentation: when the truncation boundary falls on
    a combining mark, it backs off past the base+mark run so a base character is
    not left without its mark. Multi-code-point graphemes beyond simple
    base+combining sequences (e.g. emoji ZWJ sequences, regional-indicator pairs)
    are not specially handled. ``content`` is guaranteed non-empty by
    ``DerivedRecord`` validation, so the result is always a valid non-empty
    essence.

    Args:
        content: The derived record's (non-empty) content.

    Returns:
        A non-empty essence preview of at most ``_DERIVED_ESSENCE_MAX_CHARS``
        code points.

    """
    if len(content) <= _DERIVED_ESSENCE_MAX_CHARS:
        # Short enough to preview verbatim — no truncation, nothing to sever.
        return content
    end = _DERIVED_ESSENCE_MAX_CHARS
    while end > 0 and unicodedata.combining(content[end]):
        end -= 1
    if end == 0:
        # Degenerate: a run of combining marks spans the whole boundary window,
        # so there is no base character to cut after. Fall back to a raw
        # code-point truncation — non-empty and no worse than the input's own
        # structure — rather than emitting a single detached combining mark.
        return content[:_DERIVED_ESSENCE_MAX_CHARS]
    return content[:end]


def _derived_thought_id(content: str) -> str:
    """Return the deterministic identity of a derived thought for *content*.

    A ``uuid5`` over the content, so byte-identical derived content maps to a
    single stored thought (intra-family duplicates collapse; a re-run reuses the
    same row).

    Args:
        content: The derived thought's full content.

    Returns:
        A canonical UUID string usable as a ``thought_id``.

    """
    return str(_uuid.uuid5(_DERIVED_THOUGHT_NAMESPACE, content))


def _derived_edge_id(from_thought_id: str, to_thought_id: str) -> str:
    """Return the deterministic identity of a ``DERIVED_FROM`` provenance edge.

    A ``uuid5`` over the endpoints and edge type, so re-running derivation
    reuses the same edge row (conflict-safe on both the primary key and the
    ``(from, to, type)`` unique constraint).

    Args:
        from_thought_id: The derived (source-of-edge) thought id.
        to_thought_id: The originating source thought id.

    Returns:
        A canonical UUID string usable as an ``edge_id``.

    """
    key = f"{from_thought_id}|{to_thought_id}|{EdgeType.DERIVED_FROM.value}"
    return str(_uuid.uuid5(_DERIVED_EDGE_NAMESPACE, key))


#: SQLite extended result codes that identify an identity collision the
#: derived-records seam treats as conflict-as-reuse: a ``UNIQUE`` constraint
#: (2067) or a ``PRIMARY KEY`` constraint (1555). Classified structurally via
#: :attr:`sqlite3.Error.sqlite_errorcode` rather than by inspecting the message
#: text, so the check is locale/driver-independent and never misclassifies a
#: differently-worded constraint (e.g. a ``CHECK`` or ``FOREIGN KEY`` failure).
_UNIQUE_CONSTRAINT_ERRORCODES: frozenset[int] = frozenset(
    {
        getattr(sqlite3, "SQLITE_CONSTRAINT_UNIQUE", 2067),
        getattr(sqlite3, "SQLITE_CONSTRAINT_PRIMARYKEY", 1555),
    },
)


def _is_unique_violation(exc: aiosqlite.IntegrityError) -> bool:
    """Return ``True`` when *exc* is a UNIQUE / PRIMARY KEY constraint violation.

    Used by the derived-records seam to treat an identity collision as reuse
    (conflict-as-reuse) rather than an error, while letting other integrity
    failures (e.g. FOREIGN KEY, CHECK) propagate. Classification uses SQLite's
    extended result code (:attr:`sqlite3.Error.sqlite_errorcode`, available on
    Python 3.11+) — a UNIQUE (2067) or PRIMARY KEY (1555) code — instead of
    matching the message text, which is locale/driver-fragile and could
    misclassify (e.g. a ``CHECK`` constraint whose name contains ``"unique"``).

    Args:
        exc: The raised SQLite integrity error.

    Returns:
        ``True`` for a UNIQUE/PK violation, ``False`` otherwise (a FOREIGN KEY,
        CHECK, or other integrity failure is *not* treated as a unique
        violation and re-raises upstream).

    """
    errorcode = getattr(exc, "sqlite_errorcode", None)
    if not isinstance(errorcode, int):
        # Unreachable on the supported floor (Python >= 3.11 always exposes
        # ``sqlite_errorcode``). Fail safe rather than guessing from fragile
        # message text: "not a recognised unique violation" lets the caller
        # propagate the original error instead of misclassifying it.
        return False  # pragma: no cover
    return errorcode in _UNIQUE_CONSTRAINT_ERRORCODES


#: The extended result code for a ``FOREIGN KEY`` constraint violation (787).
#: Classified structurally via :attr:`sqlite3.Error.sqlite_errorcode` so a
#: user-defined trigger that merely mentions "foreign key" in its ``RAISE(ABORT,
#: ...)`` message (a ``SQLITE_CONSTRAINT_TRIGGER``, 1811) is never misread as a
#: real referential-integrity failure.
_FOREIGN_KEY_CONSTRAINT_ERRORCODE: int = getattr(sqlite3, "SQLITE_CONSTRAINT_FOREIGNKEY", 787)


def _is_foreign_key_violation(exc: aiosqlite.IntegrityError) -> bool:
    """Return ``True`` when *exc* is a genuine ``FOREIGN KEY`` constraint failure.

    Uses SQLite's extended result code (Python 3.11+) rather than a message
    substring, so a differently-sourced abort whose text happens to contain
    "foreign key" (e.g. a trigger ``RAISE(ABORT, ...)``) is not misclassified.

    Args:
        exc: The raised SQLite integrity error.

    Returns:
        ``True`` for a FOREIGN KEY violation, ``False`` otherwise.

    """
    errorcode = getattr(exc, "sqlite_errorcode", None)
    if not isinstance(errorcode, int):
        # Unreachable on the supported floor (Python >= 3.11 always exposes
        # ``sqlite_errorcode``). Fail safe rather than guessing from fragile
        # message text: "not a recognised FK violation" lets the caller
        # propagate the original error instead of misclassifying it.
        return False  # pragma: no cover
    return errorcode == _FOREIGN_KEY_CONSTRAINT_ERRORCODE


#: SQLite result codes that identify lock contention: the primary ``SQLITE_BUSY``
#: (5) plus its extended forms — a busy signal raised specifically during hot
#: journal recovery (261) or when a snapshot cannot be maintained (517), and
#: the equivalent of the primary code once extended result codes are in force
#: (773). Classified structurally via :attr:`sqlite3.Error.sqlite_errorcode`,
#: matching :data:`_UNIQUE_CONSTRAINT_ERRORCODES` above, so contention is never
#: confused with an unrelated ``OperationalError`` (a locked schema, a missing
#: table) that happens to share the generic "database is locked" wording.
_BUSY_ERRORCODES: frozenset[int] = frozenset(
    {
        getattr(sqlite3, "SQLITE_BUSY", 5),
        getattr(sqlite3, "SQLITE_BUSY_RECOVERY", 261),
        getattr(sqlite3, "SQLITE_BUSY_SNAPSHOT", 517),
        getattr(sqlite3, "SQLITE_BUSY_TIMEOUT", 773),
    },
)


def _is_busy_error(exc: sqlite3.OperationalError) -> bool:
    """Return ``True`` when *exc* is ``SQLITE_BUSY`` (or an extended busy code).

    Used to decide whether a failed ``BEGIN IMMEDIATE`` is lock contention —
    worth retrying — or some other ``OperationalError`` that retrying cannot
    fix. Classification uses the extended result code rather than matching the
    message text, which is locale/driver-fragile.

    Args:
        exc: The raised SQLite operational error.

    Returns:
        ``True`` for a busy/lock-contention error, ``False`` otherwise.

    """
    errorcode = getattr(exc, "sqlite_errorcode", None)
    if not isinstance(errorcode, int):
        # Unreachable on the supported floor (Python >= 3.11 always exposes
        # ``sqlite_errorcode``). Fail safe rather than guessing from fragile
        # message text: "not a recognised busy error" lets the caller
        # propagate the original error instead of misclassifying it.
        return False  # pragma: no cover
    return errorcode in _BUSY_ERRORCODES


#: Maximum number of ``BEGIN IMMEDIATE`` attempts the dedup probe-and-insert
#: window makes before giving up and raising :class:`WriteContentionError`.
#: ``PRAGMA busy_timeout`` already makes each individual attempt wait for the
#: lock; this bounds how many times the store tries again *after* an attempt
#: has exhausted that wait, for the case where contention outlasts it.
_DEDUP_BEGIN_MAX_ATTEMPTS: Final = 3

#: Delay, in seconds, before each retry of a busy ``BEGIN IMMEDIATE``, doubling
#: per attempt (0.05s before the 2nd attempt, 0.1s before the 3rd). Kept short
#: relative to the default 5s ``busy_timeout``: each attempt has already waited
#: out normal contention by the time it fails, so the retry delay only needs to
#: let a sibling transaction that is *between* statements finish, not to
#: substitute for the busy wait itself.
_DEDUP_BEGIN_RETRY_BASE_SECONDS: Final = 0.05


class _QuarantinedConnection:
    """Terminal stand-in installed on a quarantined store's ``_db`` slot.

    Once a store is quarantined its real connection is *detached* and this proxy
    takes the ``_db`` slot. Every attribute access other than an idempotent
    ``close`` raises :class:`ConnectionQuarantinedError`, so a quarantined store
    fails hard on **any** core-initiated DB operation — ``commit``, ``execute``,
    ``cursor``, a read, or one of the direct-commit sites that bypass the
    :meth:`SqliteEngravaCore._maybe_commit` flag guard — **independent of whether
    the physical connection actually closed**. This is what makes quarantine
    terminal by construction rather than by a best-effort ``close()`` succeeding.

    ``close`` is a no-op so store shutdown stays graceful after quarantine.
    Private to this module; never part of the public API.

    Args:
        reason: Human-readable cause, surfaced on every raised error.

    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def close(self) -> None:
        """Idempotent no-op — the real connection is already detached."""

    def __getattr__(self, name: str) -> NoReturn:
        """Reject every other attribute access on a quarantined connection.

        Args:
            name: The attribute being accessed (e.g. ``execute``/``commit``).

        Raises:
            ConnectionQuarantinedError: Always.

        """
        raise ConnectionQuarantinedError(self._reason)


#: Page size for the full-table paginated scans that must inspect *every*
#: matching row rather than relying on a single capped page: the
#: orphan-REFLECTION sweep (:meth:`SqliteEngravaCore.retire_orphan_reflections`,
#: contract "for each ACTIVE REFLECTION") and the Memory Hygiene candidate scan
#: (:meth:`SqliteEngravaCore._hygiene_candidates`, which must score the whole
#: ACTIVE/CREATED pool so the coldest thoughts — not an arbitrary page — are the
#: ones selected under the per-run cap). Exposed as a module constant so tests
#: can shrink it to exercise the multi-page path on small synthetic inputs.
_ORPHAN_SWEEP_PAGE_SIZE = 500

#: Terminal action statuses — the only statuses that contribute to a thought's
#: ``action_outcome_score`` aggregate. Non-terminal statuses (PLANNED,
#: EXECUTING, BLOCKED) are excluded because their outcome is not yet decided.
_TERMINAL_ACTION_STATUSES: frozenset[ActionStatus] = frozenset(
    {ActionStatus.CONFIRMED, ActionStatus.FAILED}
)

#: Outcome value contributed by a CONFIRMED action, keyed by its verification
#: status. A CONFIRMED action that verification later contradicts (FAILED)
#: scores ``0.0``; a fully-verified success scores ``1.0``; every intermediate
#: or not-yet-verified state is neutral (``0.5``) — succeeded-but-unverified is
#: deliberately not rewarded as a full success. These numbers are the documented
#: mapping; they live here as a single named table so they stay tunable and
#: directly testable.
_CONFIRMED_VERIFICATION_OUTCOME: dict[VerificationStatus, float] = {
    VerificationStatus.CONFIRMED: 1.0,
    VerificationStatus.PARTIAL: 0.5,
    VerificationStatus.PENDING: 0.5,
    VerificationStatus.UNVERIFIABLE: 0.5,
    VerificationStatus.FAILED: 0.0,
}


def _action_outcome_value(action: ActionRecord) -> float | None:
    """Return the outcome value of a single action, or ``None`` when undecided.

    The value is defined only for a **terminal** action; a non-terminal
    status (PLANNED, EXECUTING, BLOCKED) returns ``None`` and is excluded
    from the aggregate.

    For a terminal action:

    * ``FAILED`` scores ``0.0`` regardless of verification.
    * ``CONFIRMED`` is adjusted by verification via
      :data:`_CONFIRMED_VERIFICATION_OUTCOME` — ``CONFIRMED`` verification
      scores ``1.0``, ``FAILED`` verification (a contradiction) scores
      ``0.0``, and every other verification state is a neutral ``0.5``.

    Args:
        action: The action to score.

    Returns:
        A float in ``[0.0, 1.0]`` for a terminal action, or ``None`` when
        the action is non-terminal.

    """
    if action.status not in _TERMINAL_ACTION_STATUSES:
        return None
    if action.status is ActionStatus.FAILED:
        return 0.0
    # CONFIRMED status — adjusted by verification.
    return _CONFIRMED_VERIFICATION_OUTCOME[action.verification_status]


def _aggregate_action_outcome(actions: list[ActionRecord]) -> float | None:
    """Return the mean outcome value over the terminal actions, or ``None``.

    The aggregate is the arithmetic mean of :func:`_action_outcome_value`
    over the actions whose status is terminal. A thought with no terminal
    actions has no defined outcome and yields ``None`` (an all-non-terminal
    or empty action set).

    Args:
        actions: All actions linked to one thought.

    Returns:
        The mean terminal outcome value in ``[0.0, 1.0]``, or ``None`` when
        there are no terminal actions.

    """
    values = [v for v in (_action_outcome_value(a) for a in actions) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_embed_input(essence: str, content: str) -> str:
    r"""Build the text payload to embed for a thought, avoiding duplication.

    A common client (and benchmark) convention is to derive ``essence`` from
    the opening of ``content`` (e.g. ``essence = content[:200]``). Naively
    embedding ``f"{essence}\\n{content}"`` then encodes the turn's opening
    twice, letting it dominate the vector and dilute the discriminative tail.

    The rule is deliberately conservative: when the stripped ``essence`` is a
    leading *prefix* of the stripped ``content`` it carries no new information,
    so ``content`` is embedded alone. In every other case — including partial
    overlaps that are not a clean prefix — the joined ``essence`` + ``content``
    form is preserved, because a distinct essence is signal worth encoding.

    Args:
        essence: The thought's short summary / essence field.
        content: The thought's full body text.

    Returns:
        ``content`` alone when ``essence`` is a prefix of it; otherwise the
        newline-joined ``f"{essence}\\n{content}"`` payload.

    """
    if content.strip().startswith(essence.strip()):
        return content
    return f"{essence}\n{content}"


#: ``_metadata`` key recording the fingerprint of the ``document_prefix`` the
#: corpus was embedded with. Present only when a non-empty document prefix is
#: active — an unprefixed corpus (the default) never writes it, so the
#: ``_metadata`` shape is byte-identical to the legacy one and a pre-existing
#: store never false-trips the lock.
_METADATA_DOCUMENT_PREFIX_FINGERPRINT = "embedding_document_prefix_fingerprint"

#: ``_metadata`` key recording the literal ``query_prefix`` the corpus was
#: built to pair with. Present only when a non-empty query prefix is active.
#: A divergent active query prefix raises a loud search-time mismatch; the
#: stored document vectors are unaffected, so this never forces a re-embed.
_METADATA_QUERY_PREFIX = "embedding_query_prefix"


def _role_prefixes(provider: object) -> tuple[str, str]:
    """Return the ``(query_prefix, document_prefix)`` a provider declares.

    The role capability is treated as **all-or-nothing**: only a provider
    that satisfies the full :class:`RoleAwareEmbeddingProvider` capability
    (both prefixes *and* every role method) declares prefixes. A provider
    without it — a user callback, a third-party class, the symmetric OpenAI
    provider, or one that implements the capability only partially — reports
    empty prefixes, the legacy, byte-identical behaviour. Detecting the
    prefixes and dispatching the role methods (see :func:`_embed_document` /
    :func:`_embed_query`) therefore key off the *same* capability check, so a
    partial provider can never be prefixed on one path yet recorded as
    unprefixed in ``_metadata``.

    Args:
        provider: The embedding provider to inspect.

    Returns:
        The ``(query_prefix, document_prefix)`` pair, each ``""`` when the
        provider does not fully declare the role-aware capability.

    """
    if isinstance(provider, RoleAwareEmbeddingProvider):
        return provider.query_prefix, provider.document_prefix
    return "", ""


def _document_prefix_fingerprint(document_prefix: str) -> str | None:
    """Return a deterministic fingerprint of a non-empty document prefix.

    An empty prefix maps to ``None`` — the legacy corpus identity — so the
    lock records nothing extra and an existing unprefixed store is untouched.
    A non-empty prefix hashes to a stable hex digest that changes whenever
    the prefix changes, which is exactly when every stored vector would
    change and the corpus needs re-embedding.

    Args:
        document_prefix: The active document-role prefix.

    Returns:
        A hex SHA-256 digest of the prefix, or ``None`` when the prefix is
        empty.

    """
    if not document_prefix:
        return None
    return hashlib.sha256(document_prefix.encode("utf-8")).hexdigest()


async def _embed_document(provider: object, text: str) -> list[float]:
    """Embed a document, using the role-aware path when the provider has it.

    Dispatches by the full :class:`RoleAwareEmbeddingProvider` capability: a
    provider that satisfies it encodes ``text`` with its document-role
    prefix; any other provider (including one that implements the capability
    only partially) falls back to plain ``embed`` — byte-identical to before
    this capability existed. Using the whole-capability check keeps dispatch
    consistent with :func:`_role_prefixes`, so a provider can never be
    prefixed here yet reported as unprefixed to the model lock.

    Args:
        provider: The embedding provider.
        text: The document text to embed.

    Returns:
        The embedding vector.

    """
    if isinstance(provider, RoleAwareEmbeddingProvider):
        return await provider.embed_document(text)
    return await provider.embed(text)  # type: ignore[attr-defined,no-any-return]  # the non-role-aware branch: an untyped provider protocol


async def _embed_documents_batch(provider: object, texts: list[str]) -> list[list[float]]:
    """Embed several documents in one provider call, role-aware when available.

    The batch analogue of :func:`_embed_document`: a provider satisfying the
    full :class:`RoleAwareEmbeddingProvider` capability encodes every text with
    its document-role prefix via ``embed_document_batch``; any other provider
    (including one that implements the capability only partially) falls back to
    plain ``embed_batch`` — byte-identical to the per-document path. Dispatch
    keys off the same whole-capability check as :func:`_embed_document`, so the
    single-item and batch paths can never disagree about whether a provider is
    prefixed, and the produced vectors match per-document embedding exactly.

    Args:
        provider: The embedding provider.
        texts: The document texts to embed, in order.

    Returns:
        One embedding vector per input text, in the same order.

    """
    if isinstance(provider, RoleAwareEmbeddingProvider):
        return await provider.embed_document_batch(texts)
    return await provider.embed_batch(texts)  # type: ignore[attr-defined,no-any-return]  # the non-role-aware branch: an untyped provider protocol


async def _embed_query(provider: object, text: str) -> list[float]:
    """Embed a query, using the role-aware path when the provider has it.

    Dispatches by the full :class:`RoleAwareEmbeddingProvider` capability: a
    provider that satisfies it encodes ``text`` with its query-role prefix;
    any other provider (including one that implements the capability only
    partially) falls back to plain ``embed`` — byte-identical to before this
    capability existed. Using the whole-capability check keeps dispatch
    consistent with :func:`_role_prefixes` and the recorded query-prefix
    pairing.

    Args:
        provider: The embedding provider.
        text: The query text to embed.

    Returns:
        The embedding vector.

    """
    if isinstance(provider, RoleAwareEmbeddingProvider):
        return await provider.embed_query(text)
    return await provider.embed(text)  # type: ignore[attr-defined,no-any-return]  # the non-role-aware branch: an untyped provider protocol


# Sentinel for :func:`_provider_dimension`'s absence check. ``None`` will not do:
# a provider may legitimately hold ``dimension = None`` in its class dictionary.
_MEMBER_ABSENT: Final = object()


def _provider_dimension(provider: EmbeddingProviderProtocol) -> int:
    """Read a provider's ``dimension``, or say which member it is missing.

    ``dimension`` is a required member of
    :class:`~engrava.domain.protocols.embedding_provider.EmbeddingProviderProtocol`,
    but the protocol is structural and nothing enforces it at construction: a
    provider holding the value privately (``self._dimension``) with no public
    property is accepted and then fails when the core reads it — on the search
    path, from library internals, as a bare ``AttributeError``. Translate that
    into the typed error: a catchable engrava type, structured fields, and a
    message that says the attribute is a required protocol member and what to
    add.

    A conformant provider is read **exactly once**, as before — the member is
    accessed in the ``try`` and everything else lives in the ``except``.
    ``hasattr(provider, ...)`` would be the shorter spelling and is deliberately
    not used: it evaluates the property to answer, so the read would happen twice
    and a provider whose ``dimension`` is stateful, expensive, or single-use
    would behave differently than it does today.

    Only a *missing* member is translated. A provider that declares ``dimension``
    and whose own property raises ``AttributeError`` has not violated the
    protocol — it has failed at something else, possibly transiently — so its
    exception propagates unchanged rather than being relabelled as a contract
    error and handed remediation advice that does not apply. The two are told
    apart with :func:`inspect.getattr_static`, which resolves the attribute
    through the class dictionaries without invoking any descriptor, so the
    already-failed property is not entered a second time.

    Args:
        provider: The store's configured embedding provider.

    Returns:
        The provider's declared embedding dimension.

    Raises:
        EmbeddingProviderContractError: When the provider exposes no public
            ``dimension``.
        AttributeError: Unchanged, when the provider declares ``dimension`` and
            its own property raised.

    """
    try:
        return provider.dimension
    except AttributeError as exc:
        if inspect.getattr_static(provider, "dimension", _MEMBER_ABSENT) is not _MEMBER_ABSENT:
            raise
        raise EmbeddingProviderContractError(
            provider_class=type(provider).__name__,
            member="dimension",
        ) from exc


def _query_vector_is_degenerate(query_vector: list[float]) -> bool:
    """Return whether a query vector has no usable cosine direction.

    Cosine similarity is only defined for a vector with a positive, finite
    magnitude. Three shapes have none, and every one of them would otherwise
    make the vector arm *silently* return an empty result (an empty match is
    indistinguishable from "the corpus had no neighbours"):

    * an **empty** vector (no components at all);
    * an **all-zero** vector — a zero magnitude has no direction, and the
      canonical way one arises is auto-embedding empty/stop-word-only text;
    * a vector carrying a **non-finite** component (``NaN``/``±inf``), which a
      provider should never emit but which a caller can pass directly and which
      poisons the whole dot product into ``NaN``.

    These are surfaced through
    :attr:`SqliteEngravaCore.vector_arm_degradation_count` rather than raised,
    because — unlike a wrong *dimension* — a degenerate vector is a run-time
    query-quality condition, not a structural contract violation.

    Args:
        query_vector: The query embedding to inspect.

    Returns:
        ``True`` when the vector is empty, all-zero, or non-finite; ``False``
        for any vector with at least one finite non-zero component.

    """
    if not query_vector:
        return True
    saw_nonzero = False
    for value in query_vector:
        if not math.isfinite(value):
            return True
        if value != 0.0:
            saw_nonzero = True
    return not saw_nonzero


def _archived_exclusion_sql(*, column: str, include_archived: bool) -> str:
    """Return an ``AND``-prefixed clause excluding archived rows, or empty string.

    Archived thoughts (``lifecycle_status = 'ARCHIVED'``) are removed from the
    default retrieval candidate set — the same eligibility class as expired rows
    and retired REFLECTIONs — so a forgotten thought stops surfacing without
    being deleted. The exclusion is reversible: ``restore_thought`` flips the row
    back to ``ACTIVE`` (eligible again), and an ``include_archived`` query
    re-admits archived rows for this call without restoring them.

    The clause is deliberately narrow — it drops only ``ARCHIVED`` rows and never
    touches the independent retired-REFLECTION freshness floor (a retired
    REFLECTION stays excluded even under ``include_archived=True``, because its
    ``!= 'ACTIVE'`` guard is a separate ``AND``-ed condition).

    Args:
        column: The ``lifecycle_status`` column reference to gate — e.g.
            ``"t.lifecycle_status"`` for a query that aliases ``thought`` as
            ``t``, or ``"lifecycle_status"`` for an unaliased table.
        include_archived: When ``True`` the escape hatch is engaged and this
            returns the empty string (archived rows stay eligible); when
            ``False`` (the default retrieval behaviour) it returns the exclusion
            fragment.

    Returns:
        ``" AND {column} != 'ARCHIVED'"`` when excluding archived rows, otherwise
        the empty string.

    """
    if include_archived:
        return ""
    return f" AND {column} != '{LifecycleStatus.ARCHIVED.value}'"


#: A token is treated as an FTS5 column filter only when it targets a real
#: indexed column. ``thought_fts`` indexes exactly ``essence`` and ``content``
#: (see :meth:`SqliteEngravaCore.ensure_schema`); any other ``word:rest`` token
#: (URLs like ``http://...``, timestamps like ``12:30``) would make FTS5 read a
#: non-existent column and raise, so it is sanitized as a bare token instead.
_FTS_FIELD_FILTER_RE = re.compile(r"^(?:essence|content):.+", re.IGNORECASE)
_FTS_UNSAFE_CHAR_RE = re.compile(r"[^\w\-*]")
#: Standalone uppercase boolean operators that switch a query into expert mode.
#: Lowercase ``and``/``or``/``not`` are ordinary words, not operators.
_FTS_BOOLEAN_OPERATORS = frozenset({"AND", "OR", "NOT"})
#: Thought-count above which :meth:`SqliteEngravaCore.recall` emits a one-time
#: DEBUG nudge when called without ``current_cycle`` (so the recency signal is
#: silently inactive). Below this the omission is unremarkable; past it, a store
#: large enough to benefit from recency that never receives a cycle is worth a
#: single diagnostic breadcrumb (never a warning, never repeated).
_RECENCY_NUDGE_THRESHOLD = 25
#: Default transaction-time recency half-life, in wall-clock seconds (7 days) —
#: the fallback used only when no :class:`~engrava.config.SearchConfig` is wired.
#: A reasonable agent-memory freshness scale; override per call via
#: ``recency_now_half_life`` or store-wide via
#: ``SearchConfig.recency_now_half_life_seconds``.
_DEFAULT_RECENCY_NOW_HALF_LIFE_SECONDS = 604800
#: Deterministic minimum transaction-time recency score. A row whose transaction
#: timestamp is missing or malformed (legacy / imported data) is treated as
#: maximally old and scores this — never a crash and never a host-clock read.
_MIN_RECENCY_SCORE = 0.0
#: Absolute upper bound on how many neighbours the sqlite-vec (vec0) arm may
#: over-fetch before the live-row post-filter runs. The vec0 backend can only
#: filter expired/retired rows *after* its ``LIMIT``, so it over-fetches
#: ``top_k * vec0_overfetch_factor`` to give the filter a deeper pool to survive
#: from. This cap keeps that fetch bounded: without it, when ``search_hybrid``
#: has already widened ``vector_top_k`` via ``collapse_pool_factor``, the effect
#: would compound into ``top_k * collapse_factor * overfetch_factor`` — an
#: unbounded product. The cap turns the combined widening into a bounded maximum
#: rather than a multiplicative blow-up. 500 comfortably exceeds realistic
#: ``top_k`` values while capping worst-case scan/join work per query.
_VEC0_OVERFETCH_CAP = 500
#: Maximum host parameters bound into a single ``... IN (?, ?, …)`` statement.
#: SQLite's historical compile-time default for ``SQLITE_MAX_VARIABLE_NUMBER``
#: is 999; batched ``IN`` fetches chunk their id lists to this size so a large
#: input set never exceeds the limit (newer SQLite raises the default, but
#: staying at 999 is safe on every supported build).
_SQLITE_MAX_VARS = 999
_SUPPRESS_SEARCH_METRICS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "engrava_suppress_search_metrics",
    default=False,
)

#: Task-local suppression for a nested write's own auto-commit, read only by
#: :meth:`SqliteEngravaCore._maybe_commit`. ``run_hygiene`` sets this for the
#: duration of its archive+GC unit so a nested public write it makes itself
#: (``retire_orphan_reflections`` -> ``update_thought``) does not end that
#: unit early with its own commit. Deliberately **not** the instance-wide
#: ``_skip_auto_commit_depth``: that counter is shared by every task on the
#: store, so raising it for the length of a hygiene pass would make an
#: unrelated task's guarded write see a foreign window as its own — the same
#: shape of bug ``_dispatch_derivation`` used to have before it gained its own
#: task-local marker (see ``_current_auto_commit_window``). A ``ContextVar``
#: keeps the suppression visible only to the task running the pass, exactly
#: like ``_IN_DERIVATION`` / ``_SUPPRESS_SEARCH_METRICS`` above.
_SUPPRESS_NESTED_AUTO_COMMIT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "engrava_suppress_nested_auto_commit",
    default=False,
)


class _LatencyRingBuffer:
    """Fixed-size async-safe ring buffer of recent search latencies."""

    def __init__(self, window_size: int = 1000) -> None:
        self._window_size = window_size
        self._buf: list[float] = []
        self._lock = asyncio.Lock()

    async def record(self, latency_ms: float) -> None:
        """Append a latency sample, evicting the oldest entry if needed."""
        async with self._lock:
            if len(self._buf) >= self._window_size:
                self._buf.pop(0)
            self._buf.append(latency_ms)

    async def snapshot(self) -> LatencyHistogram:
        """Return percentile statistics for the current window."""
        from engrava.domain.models.metrics import LatencyHistogram  # noqa: PLC0415

        async with self._lock:
            samples = list(self._buf)

        if not samples:
            return LatencyHistogram()

        arr = np.asarray(samples, dtype=np.float64)
        return LatencyHistogram(
            sample_count=len(samples),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            mean_ms=float(np.mean(arr)),
        )


#: Default hard cap on the number of *distinct* thought ids the in-process
#: access buffer holds before it evicts. The buffer is deliberately small — it
#: only bridges reads to the next consolidation-cycle flush, and the counts are
#: regenerable telemetry — so a modest cap bounds memory without losing
#: material signal (a genuinely hot thought is re-accessed and re-buffered after
#: an eviction). Deterministic FIFO eviction keeps behaviour reproducible.
_ACCESS_BUFFER_DEFAULT_CAP = 10_000


class _AccessBuffer:
    """Bounded, instance-scoped buffer of pending thought-access deltas.

    Retrieval paths append an access event here instead of issuing a
    per-result ``UPDATE`` on the hot read path (which would turn a read into a
    write). The buffer coalesces repeated accesses of the same thought into a
    single ``(count_delta, last_seen_ts)`` entry and is drained by a single
    batched ``UPDATE`` at the consolidation-cycle boundary (and on an explicit
    flush or store close).

    **Bounded with deterministic eviction.** At most ``cap`` distinct thought
    ids are held. When a *new* id would exceed the cap, the oldest-inserted id
    is evicted (FIFO) and the eviction is logged. Coalescing an access into an
    id already in the buffer never triggers eviction.

    **No lock, and none needed.** Every mutation of the buffer happens inside a
    synchronous method with no ``await`` in it, so no other task can interleave
    part-way through one. Concurrent tasks on the same store record accesses
    safely; the buffer does not rely on a single writer.

    Access counts are high-volume regenerable telemetry: they are **not**
    journaled, and a crash before a flush simply undercounts — the signal
    self-heals as access continues.

    Args:
        cap: Maximum number of distinct thought ids retained before eviction.

    """

    def __init__(self, cap: int = _ACCESS_BUFFER_DEFAULT_CAP) -> None:
        self._cap = max(1, cap)
        # Insertion-ordered so eviction is a deterministic FIFO pop.
        self._pending: dict[str, tuple[int, str]] = {}
        self._evicted_total = 0

    def __len__(self) -> int:
        return len(self._pending)

    def record(self, thought_id: str, *, now: str) -> None:
        """Buffer one access to ``thought_id`` seen at ``now``.

        Coalesces into an existing entry (incrementing its delta and advancing
        the last-seen timestamp) or inserts a new entry, evicting the
        oldest-inserted id first when the cap would be exceeded.

        Args:
            thought_id: The retrieved thought's id.
            now: ISO-8601 timestamp of this access.

        """
        existing = self._pending.get(thought_id)
        if existing is not None:
            self._pending[thought_id] = (existing[0] + 1, now)
            return
        if len(self._pending) >= self._cap:
            evicted_id, _ = next(iter(self._pending.items()))
            del self._pending[evicted_id]
            self._evicted_total += 1
            logger.warning(
                "access buffer full (cap=%d); evicted pending access for thought %s "
                "(%d evicted since open) — access counts are best-effort telemetry",
                self._cap,
                evicted_id,
                self._evicted_total,
            )
        self._pending[thought_id] = (1, now)

    def drain(self) -> list[tuple[str, int, str]]:
        """Empty the buffer, returning ``(thought_id, count_delta, last_seen)``.

        Returns:
            The pending deltas as a list; the buffer is cleared. An empty list
            when nothing was buffered.

        """
        drained = [(tid, delta, ts) for tid, (delta, ts) in self._pending.items()]
        self._pending.clear()
        return drained


async def _close_quietly(conn: aiosqlite.Connection) -> None:
    """Close *conn*, logging rather than raising if the close itself fails.

    Cleanup code that closes a connection while another exception -- or a
    cancellation -- is already propagating must not let a failure in the
    close itself replace what the caller actually needs to see: a bare
    ``raise`` after an unconditional ``await conn.close()`` only re-raises
    the original error when that close *succeeds*. If the close itself
    raises, its exception becomes the one that propagates and the original
    -- a ``ConfigError``, a ``JournalIntegrityError``, an
    ``asyncio.CancelledError`` -- is lost. A close failure is real
    information, but it belongs logged underneath the original error, not
    raised in front of it. **Use this for cleanup that closes a connection
    while another exception -- or a cancellation -- is already
    propagating.** A close on the ordinary success path, where a close
    failure is the only thing there is to report, should propagate
    normally instead of coming through here -- routing it through this
    helper would silently turn a genuine close failure into a
    successful-looking outcome. ``service_manager.py`` imports this same
    function rather than defining its own. The CLI layer has the same rule
    under the same name in :mod:`engrava.cli.main` -- not shared as one
    function across the CLI/infrastructure boundary, but copied rather
    than re-derived.

    ``await conn.close()`` is itself a suspension point, so a bare
    ``try/except Exception`` around it has the identical gap this whole
    helper exists to close: a cancellation arriving while the close is
    in flight is a ``BaseException``, skips that handler, and can leave
    the close abandoned mid-way with aiosqlite's non-daemon worker thread
    still alive. The close is run as its own task and shielded so that
    cancelling *this* coroutine does not also cancel the close itself;
    the shield alone would not be enough, though, since it only stops the
    cancellation from reaching the close, not from being re-thrown into
    this coroutine before the close finishes running. So on cancellation
    this explicitly awaits the same task again -- now cancellation-proof,
    since a second throw only happens on an explicit second
    ``cancel()`` -- to hold this coroutine (and so whatever awaits it,
    keeping the event loop alive) open until the real close has actually
    completed, before letting the cancellation propagate.

    Args:
        conn: The aiosqlite connection to close.

    """
    try:
        close_task = asyncio.ensure_future(conn.close())
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        try:
            await close_task
        except Exception:
            logger.warning("Error closing connection during cleanup", exc_info=True)
        raise
    except Exception:
        logger.warning("Error closing connection during cleanup", exc_info=True)


async def _run_cleanup_step_quietly(
    step: Callable[[], Coroutine[Any, Any, object]],
    description: str,
    *,
    log_failure: Callable[[Exception], None] | None = None,
) -> asyncio.CancelledError | None:
    """Run one cleanup statement, logging rather than raising an ordinary failure.

    Generalises :func:`_close_quietly`'s shield-then-redraw technique (see
    that function's docstring for the full rationale) to an arbitrary
    cleanup statement -- a transaction rollback, a pragma restore -- rather
    than specifically a connection close. A migration cleanup that must run
    more than one such step in sequence (rollback, then restore a pragma)
    needs the same "log a failure here, don't let it replace what is already
    propagating" treatment for each step, and this is that treatment,
    written once -- shared across the infrastructure and CLI layers rather
    than copied, unlike :func:`_close_quietly` and its CLI counterpart
    (:mod:`engrava.cli.main`'s own ``_close_quietly``), which predate this
    helper and already drifted from each other once (the CLI copy gained a
    hostile-``__str__``-safe render and a cancellation/SIGINT-delivery fix
    the infrastructure copy still lacks). ``step`` is a zero-argument
    callable rather than an already-created coroutine so that a synchronous
    failure -- raised by calling it, before anything is scheduled -- is also
    caught here rather than escaping before the shield is even in place.

    An ordinary failure (not a cancellation) is logged and swallowed: the
    caller already knows an exception is propagating through it, and this
    step's own failure is secondary information, not the error to report. A
    cancellation delivered to *this* call while the step is in flight is
    different -- a control-flow signal, not an ordinary error -- so it is
    never swallowed. It is returned rather than raised so a caller running
    several steps in sequence can still finish the remaining ones before
    deciding which exception ultimately propagates.

    The default log line is a plain ``exc_info=True`` warning, matching this
    module's own ``_close_quietly``. A caller whose logging needs to differ
    -- a different logger, or a message shape like the CLI's own
    ``_close_quietly``/``_rollback_quietly``, which reads the cleanup
    exception once through a guarded, non-absorbing description instead of
    letting the standard traceback formatter render it a second, unguarded
    way -- passes ``log_failure`` rather than this being forked into a
    second copy of the whole function.

    Args:
        step: Zero-argument callable returning the awaitable to run, e.g.
            ``self._db.rollback`` or ``lambda: self._db.execute("PRAGMA ...")``.
        description: Human-readable description of the step, used only in
            the default warning logged on an ordinary failure (ignored when
            ``log_failure`` is given).
        log_failure: Called with the cleanup step's own exception instead of
            the default warning, when the step fails with an ordinary
            (non-cancellation) exception. Never called for a cancellation.

    Returns:
        The ``CancelledError`` delivered to this call, if any. ``None`` when
        the step ran to completion -- successfully or not -- without this
        call itself being cancelled.

    """
    try:
        task = asyncio.ensure_future(step())
        await asyncio.shield(task)
    except asyncio.CancelledError as exc:
        try:
            await task
        except Exception as cleanup_exc:
            if log_failure is not None:
                log_failure(cleanup_exc)
            else:
                logger.warning("Error %s during cleanup", description, exc_info=True)
        return exc
    except Exception as cleanup_exc:
        if log_failure is not None:
            log_failure(cleanup_exc)
        else:
            logger.warning("Error %s during cleanup", description, exc_info=True)
        # See ``_close_quietly`` for why this matters: without a real
        # suspension point after the synchronous logging above, a
        # cancellation or SIGINT requested while this coroutine was running
        # synchronously would be silently dropped instead of reaching the
        # caller.
        await asyncio.sleep(0)
    return None


def _validate_provider_cycle(value: object) -> int:
    """Validate a value pulled from a ``CycleProvider`` at the trust boundary.

    A ``runtime_checkable`` protocol verifies only that a provider *has* a
    ``current_cycle`` method — never that the value it returns is a usable
    cognitive cycle. So the store validates the pulled value here, at the
    resolution boundary: it must be a real ``int`` (``bool`` is rejected even
    though it subclasses ``int``) and non-negative (matching the
    ``created_cycle`` / ``updated_cycle`` ``ge=0`` invariant).

    Args:
        value: The raw value returned by ``cycle_provider.current_cycle()``.

    Returns:
        The validated cycle as an ``int``.

    Raises:
        CycleProviderError: When ``value`` is not an ``int`` (including a
            ``bool``) or is negative.

    """
    # ``type(value) is int`` — deliberately not ``isinstance`` — so a ``bool``
    # (a subclass of ``int``) is rejected rather than silently coerced.
    if type(value) is not int:
        msg = f"expected int, got {type(value).__name__}"
        raise CycleProviderError(msg)
    if value < 0:
        msg = f"expected a non-negative cycle, got {value}"
        raise CycleProviderError(msg)
    return value


#: Bound on how long a task waits to acquire an in-process write lock before
#: :class:`~engrava.domain.exceptions.WriteLockTimeoutError` is raised (see
#: :class:`_TaskReentrantLock`). **Not** a claim that nothing under the lock
#: does network I/O — ``bulk_store``'s batch embedding call runs inside
#: :meth:`SqliteEngravaCore.suspend_auto_commit`, which holds this lock for
#: its whole duration, and that call reaches a real embedding provider over
#: the network. A legitimate hold can genuinely take minutes, so this bound
#: is a backstop that converts an *unrecoverable* deadlock (a task spawned
#: and awaited from inside another task's own window — see the class
#: docstring) into an attributable, catchable error — not a policy against
#: slow legitimate work, though it **can** still misfire on one if this
#: value is too small for your own provider or batch sizes, which is exactly
#: why it is a per-instance parameter rather than a hardcoded constant. Must
#: never be tightened without re-deriving this number.
#:
#: Derived, not guessed, from the shipped default provider
#: (:class:`~engrava.embeddings.openai_compatible.OpenAICompatibleProvider`):
#: one ``embed_batch``/``embed_document_batch`` call is one HTTP round trip
#: for the *whole* batch (no internal chunking), made with a 60s client
#: timeout, up to 3 attempts total, with backoff of ``1 * attempt`` seconds
#: between them. Worst case before that provider itself gives up:
#: ``3 * 60 + (1 + 2) = 183`` seconds — and that is only the shipped
#: default's own retry-exhaustion ceiling, not an upper bound on every
#: provider a caller can configure (a self-hosted or custom provider may
#: have its own, larger timeout and retry budget the lock cannot see) or on
#: how long a large batch can legitimately take to process even on a single
#: successful attempt. The default below is a small integer multiple of that
#: 183s figure for headroom, not a round number picked for the sake of
#: looking generous — tune it via ``SqliteEngravaCore(...,
#: write_lock_acquire_timeout_seconds=...)`` for a slower configured
#: provider or a larger batch ceiling than this covers.
_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS = 600.0

#: How long :meth:`SqliteEngravaCore.close` waits for the aiosqlite worker
#: thread to answer the physical close before giving up on it. Unbounded
#: before this: a worker that never answers — busy with something
#: legitimately slow, or genuinely wedged inside a call nothing can
#: interrupt — left ``close()`` queued behind it forever. See ``close``'s own
#: docstring for what happens once this bound expires.
#:
#: **Not a per-call budget.** ``close()`` applies this same value a second
#: time, independently, to the access-buffer flush it runs before the
#: physical close — so one call can wait up to *twice* this figure (60s at
#: the default) when the worker never answers at all, not this figure as a
#: ceiling on the whole call. See :meth:`SqliteEngravaCore.close` for why
#: the flush needs its own bound rather than sharing one with the close.
#:
#: Derived from a number already in this file, not an unrelated one:
#: :meth:`SqliteEngravaCore.from_config` sets ``PRAGMA busy_timeout=5000`` on
#: every connection it opens — five seconds is already this product's own
#: line between "the database is legitimately busy" and "something is
#: wrong", at the level SQLite itself can see. ``close()`` is waiting on the
#: same worker thread from one layer above that PRAGMA, so its bound uses a
#: small integer multiple of that figure for headroom (the close's own WAL
#: bookkeeping, plus whatever the worker was already doing when the stop
#: request was queued behind it) rather than inventing an unrelated number:
#: **30 seconds, six times the busy-timeout floor.**
#:
#: This is deliberately much shorter than
#: :data:`_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS` (600s) — that bound covers a
#: *different* task waiting through a legitimate, network-bound embedding
#: round trip that never touches the worker thread at all (the wait happens
#: on the Python/asyncio side, while the connection itself is idle). This
#: bound covers a caller waiting on the worker thread itself, where
#: measurement behind this bound found only two shapes: a slow statement
#: that finishes in seconds, or a worker that will never answer at any
#: bound, however large — so a large default buys nothing for the first
#: case and only makes the second case's caller wait longer than it has to.
#: That caller is plausibly a short-lived command-line invocation through
#: the ``--config`` tier (``SqliteEngravaCore.from_config`` — the case that
#: motivated this): the CLI's other, bare-connection tier calls
#: ``aiosqlite.Connection.close()`` directly and does not go through this
#: method at all, so it does not inherit this bound regardless of the
#: default chosen here.
#: Tune it via ``SqliteEngravaCore(..., close_timeout_seconds=...)`` /
#: ``from_config(..., close_timeout_seconds=...)`` for a deployment whose
#: legitimate closes run slower than this covers.
_CLOSE_TIMEOUT_SECONDS: Final = 30.0


class _TaskReentrantLock:
    """An ``asyncio``-compatible lock that the *same task* may re-acquire freely.

    A plain :class:`asyncio.Lock` is not reentrant: a task that calls
    ``acquire`` twice — e.g. a write issued from inside its own
    :meth:`SqliteEngravaCore.suspend_auto_commit` window — blocks on itself and
    hangs forever. This wrapper tracks the *owning task* and a nesting depth so
    the owning task's own nested acquisitions are free, while every other task
    still blocks on the underlying lock exactly as before. Getting this wrong
    turns a silent data loss into a hang, which is not an improvement — so the
    identity check is on the task object itself (``asyncio.current_task()``),
    never on anything the caller could pass or forge.

    Deliberately task-scoped, not store-scoped: two genuinely concurrent tasks
    still serialise against each other (the second call to ``acquire`` blocks
    until the first task's matching, outermost ``release``), which is exactly
    what makes a read-modify-write critical section atomic across tasks. This
    mirrors the shape of :attr:`SqliteEngravaCore._dedup_lock` (:class:`_DedupLock`,
    itself wrapping an ``asyncio.Lock`` to guard one instance-wide critical
    section — see that class for why a *same-task* second acquisition raises
    there instead of blocking, unlike this lock's free same-task recursion)
    plus the
    task-local ``ContextVar``s already in this file
    (:attr:`SqliteEngravaCore._suppress_access_tracking`) rather than inventing
    a new primitive: an ``asyncio.Lock`` for the store-wide serialisation, task
    identity for the reentrancy.

    **Re-entrancy is keyed on the task, and a task boundary is a hard edge.**
    A task spawned from inside a :meth:`SqliteEngravaCore.suspend_auto_commit`
    window — via ``asyncio.create_task``, ``gather``, ``wait_for``, or one
    spawned inside an ``on_store`` hook or embedding provider callback — is a
    *different* task, however it was created. If the window's own task then
    awaits that spawned task (directly, or transitively) before its window
    closes, the spawned task can never get the lock the window's task is
    holding, and the window's task can never release it while still awaiting
    the spawned task: an unrecoverable deadlock this store cannot resolve.
    This is already out of the documented contract — every guarded write on a
    given store instance while a ``suspend_auto_commit`` window is open must
    come from the one task that opened it — so ``acquire`` bounds a
    *different* task's wait at :data:`_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS` and
    raises :class:`~engrava.domain.exceptions.WriteLockTimeoutError` past it,
    trading a silent, unattributable hang for a typed, catchable failure that
    also ends the deadlock (the spawned task's failure lets whatever awaited
    it unwind, freeing the window's task to finish and release the lock).

    **That bound is not "ordinary contention never gets close to it" —
    ``bulk_store``'s batch embedding call runs inside its own
    ``suspend_auto_commit`` window, holding this lock for the whole,
    genuinely network-bound embedding round trip**, which can legitimately
    take minutes for a large batch. The bound has to clear that legitimate
    case with real margin, not merely clear "ordinary" contention — see
    :data:`_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS` for how its default is
    derived and why a caller with a slower provider or larger batches should
    raise it via ``SqliteEngravaCore(...,
    write_lock_acquire_timeout_seconds=...)`` rather than accept a false
    timeout on correct usage.
    """

    def __init__(
        self, *, acquire_timeout_seconds: float = _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS
    ) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._depth = 0
        self._acquire_timeout_seconds = acquire_timeout_seconds

    async def acquire(self) -> None:
        """Acquire the lock, or recurse for free if this task already holds it.

        Raises:
            WriteLockTimeoutError: When a *different* task cannot acquire the
                lock within :attr:`_acquire_timeout_seconds` — see the class
                docstring for why this bound exists and what tripping it means.

        """
        current = asyncio.current_task()
        if current is not None and current is self._owner:
            self._depth += 1
            return
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._acquire_timeout_seconds)
        except TimeoutError:
            raise WriteLockTimeoutError(timeout_seconds=self._acquire_timeout_seconds) from None
        self._owner = current
        self._depth = 1

    def release(self) -> None:
        """Release one level of nesting; only the outermost release unblocks others.

        Raises:
            RuntimeError: When called by a task that is not the current holder
                (including a task that never acquired it) — the same misuse
                :meth:`asyncio.Lock.release` itself guards against.

        """
        if self._depth <= 0 or asyncio.current_task() is not self._owner:
            msg = "_TaskReentrantLock.release() called by a task that does not hold it"
            raise RuntimeError(msg)
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self) -> None:
        """Support ``async with self._write_lock:``, matching ``asyncio.Lock``."""
        await self.acquire()

    async def __aexit__(self, *exc_info: object) -> None:
        """Release on block exit, including on exception."""
        self.release()


class _DedupLock:
    """A dedup-window lock that raises on same-task re-entry instead of hanging.

    ``_dedup_lock`` guards the dedup probe-and-insert window
    (``create_thought(deduplicate=True)``, ``get_or_create``,
    ``upsert_by_hash``) and, unlike :class:`_TaskReentrantLock`, has no
    legitimate reentrant use: nothing this store does needs the *same* task
    to hold it twice. A second acquisition by the same task is always a bug
    — the concrete shape is ``upsert_by_hash``'s hit branch calling the
    overridable ``update_thought`` while still holding this lock (see
    ``docs/extension-hooks.md`` §1B.3); an ``update_thought`` override that
    calls back into ``create_thought(deduplicate=True)`` / ``get_or_create``
    / ``upsert_by_hash`` / ``bulk_store(deduplicate=True)`` on that same
    task tries to acquire this lock again while its own outer acquisition
    has not yet released it.

    A plain :class:`asyncio.Lock` blocks that second acquisition on itself
    forever, with nothing in any log to point at why — exactly the silent,
    unattributable hang ``docs/concurrency.md`` ("A deadlock this store
    cannot resolve raises, it does not hang") forbids for the write lock's
    own different-task case. This wrapper extends that same policy here:
    whether the current task already owns the lock is known synchronously,
    with no race to bound, so the second acquisition raises
    :class:`~engrava.domain.exceptions.DedupLockReentryError` immediately
    instead of awaiting the underlying lock at all.

    A *different* task still waits on the underlying :class:`asyncio.Lock`
    exactly as it would with no wrapper at all — only a same-task second
    acquisition changes behaviour.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None

    async def acquire(self) -> None:
        """Acquire the lock, or raise if this task already holds it.

        Raises:
            DedupLockReentryError: When the current task already holds this
                lock — waiting for the underlying :class:`asyncio.Lock` would
                block on itself forever, since only this same task's own
                matching :meth:`release` can ever free it.

        """
        current = asyncio.current_task()
        if current is not None and current is self._owner:
            raise DedupLockReentryError
        await self._lock.acquire()
        self._owner = current

    def release(self) -> None:
        """Release the lock so a waiting task (or a later call) can acquire it."""
        self._owner = None
        self._lock.release()

    async def __aenter__(self) -> None:
        """Support ``async with self._dedup_lock:``, matching ``asyncio.Lock``."""
        await self.acquire()

    async def __aexit__(self, *exc_info: object) -> None:
        """Release on block exit, including on exception."""
        self.release()


class SqliteEngravaCore:
    """Core SQLite persistence backend for thought-graph CRUD.

    Uses manual SQL with parameterized queries — no ORM.

    Supports transaction-aware commit control: when ``_skip_auto_commit``
    is ``True`` (set via :meth:`suspend_auto_commit`), individual methods
    skip ``db.commit()`` so the caller manages the commit boundary.

    Subclasses can override ``_row_to_thought`` to produce extended model
    types (template method pattern), and ``prepare_thought_for_insert`` to
    validate or enrich a candidate thought before the decisive probe for a
    duplicate, or — when ``deduplicate=False`` skips that probe — before the
    unconditional write instead; that choice hinges on ``deduplicate``, not on
    which entry point was called, since ``create_thought``, ``bulk_store``,
    and ``remember`` each default to ``deduplicate=False`` — the pre-insert
    seam every path that can create a new row (``create_thought``,
    ``get_or_create``, ``upsert_by_hash``, ``bulk_store``, ``remember``) calls
    through, when it runs at all (two of those skip it entirely on a stable
    hit); see that method's docstring for its exact invocation-count and
    locking contract per entry point.

    Args:
        db: An open aiosqlite connection (WAL mode, FK enabled).
        hooks: Optional extension hooks; defaults to ``DefaultEngravaHooks``.
        embedding_provider: Optional async embedding provider for auto-embed.
        auto_embed: Whether to auto-embed on ``create_thought``/``update_thought``.
        require_embedding: When ``False`` (default), an auto-embed provider
            failure logs a ``WARNING`` naming the thought and re-raises the
            provider's own exception (byte-identical to prior behaviour). When
            ``True``, that failure is normalised into a typed
            :class:`~engrava.domain.exceptions.EmbeddingGenerationError` — the
            opt-in fail-fast. This flag decides the exception *type*, not
            whether the call that raised it already committed its own row.
            When a single-item call (``create_thought``/``update_thought``)
            owns its own transaction, that row is already durable by the
            time auto-embed runs regardless of this flag, and a standalone
            :meth:`bulk_store` rolls its whole batch back regardless of this
            flag too (see that exception's docstring for the full per-path
            outcome, including the nested case, where nothing is durable
            yet on either flag setting until the caller's own outermost
            window exits). But the exception type this flag picks *can*
            affect durability when the failure happens nested inside a
            caller's own ``suspend_auto_commit()`` window — only when the
            caller's own exception handling distinguishes the two types: an
            ``except EmbeddingGenerationError`` clause around the failing
            call catches the strict-mode error and lets that outer window
            exit cleanly (so it commits), while the same clause does not
            catch the untyped provider exception this flag raises by
            default, which escapes the window and rolls it back — same
            provider failure, same caller code, opposite durable outcome. A
            caller whose ``except`` clause instead catches both types (a
            bare ``except Exception``) or neither sees the same outcome
            regardless of this flag; only type-discriminating handling makes
            the flag's choice of exception type observable in durability. No
            effect unless ``auto_embed`` is on.
        search_config: Optional default hybrid-search weights from config.
        journal_enabled: Whether to record mutations in the hash-chain
            journal.  Defaults to ``False``.
        ttl_strategy: Cleanup strategy for expired thoughts.
            ``"archive"`` (default) or ``"delete"``.
        ttl_check_every_n: Auto-cleanup cadence.  ``0`` disables.
        ttl_default_seconds: Default TTL for new thoughts.  ``None``
            means thoughts do not expire unless explicitly set.
        manifests: Extension manifests whose ``schema_migrations`` will be
            applied after core schema bootstrap.  Pass an empty
            sequence (default) to skip extension migrations entirely.
        cycle_provider: Optional, **runtime-only** opt-in cognitive-cycle
            source (a live
            :class:`~engrava.domain.protocols.cycle_provider.CycleProvider`
            object, never serialized config). When configured, the read /
            eligibility paths (``search_hybrid`` / ``recall`` recency,
            ``consolidate``, ``run_hygiene``) pull ``current_cycle`` from it
            **only** when the caller did not pass one explicitly — an explicit
            ``current_cycle`` (including ``0``) always wins. ``None`` (default)
            preserves today's behaviour byte-for-byte: no cycle is pulled and
            recency / age-gating stay off unless a cycle is passed per call. The
            provider is **read-time only** — it never stamps ``created_cycle`` /
            ``updated_cycle`` on writes.
        write_lock_acquire_timeout_seconds: How long a *different* task may
            wait to acquire this instance's in-process write lock before
            :class:`~engrava.domain.exceptions.WriteLockTimeoutError` is
            raised. A backstop that converts an unrecoverable deadlock (a
            task spawned and awaited from inside another task's own
            ``suspend_auto_commit`` window — see
            :class:`_TaskReentrantLock`) into an attributable, catchable
            error — not a policy against ordinary contention or slow
            legitimate work, though it **can** still fire on a legitimate
            hold longer than this value, which is exactly why it is
            configurable rather than fixed. The default covers
            ``bulk_store``'s batch-embedding call through the shipped
            default provider with margin (see
            :data:`_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS` for the derivation).
            Raise this if your embedding provider's own worst-case latency
            for the batches you actually send is close to or above the
            default; lower it only for a deployment that never runs
            ``bulk_store`` with network-bound embedding under load, since a
            legitimate hold longer than this value now fails loudly instead
            of completing.
        close_timeout_seconds: How long :meth:`close` waits for the aiosqlite
            worker thread to answer the physical close before giving up on
            it and quarantining the store — see :meth:`close` and
            :data:`_CLOSE_TIMEOUT_SECONDS` for what expiry does and how the
            default is derived. Raise this for a deployment whose legitimate
            closes (a large WAL checkpoint, a busy worker finishing real
            work) routinely run slower than the default covers; lower it to
            get control back sooner when the worker is wedged and will
            never answer. The same shorter bound is all a merely-slow close
            gets to prove itself in, though:
            :meth:`_finish_close_wait` cannot tell a slow worker from a wedged
            one, so a lower value also raises the odds of quarantining a store
            whose close was healthy and would have finished. The close itself
            runs to completion either way — the bound stops this call's
            observation of it, never the close.

    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        hooks: EngravaHooksProtocol | None = None,
        *,
        embedding_provider: EmbeddingProviderProtocol | None = None,
        auto_embed: bool = False,
        require_embedding: bool = False,
        search_config: SearchConfig | None = None,
        journal_enabled: bool = False,
        ttl_strategy: str = "archive",
        ttl_check_every_n: int = 0,
        ttl_default_seconds: int | None = None,
        metrics_config: MetricsConfig | None = None,
        manifests: Sequence[ExtensionManifest] = (),
        access_tracking_enabled: bool = False,
        hygiene_policy: HygienePolicyConfig | None = None,
        derive_gates: DeriveGates | None = None,
        cycle_provider: CycleProvider | None = None,
        write_lock_acquire_timeout_seconds: float = _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS,
        close_timeout_seconds: float = _CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        self._db = db
        self._hooks: EngravaHooksProtocol = hooks or DefaultEngravaHooks()
        # Nesting depth of open `suspend_auto_commit` windows on this task's
        # current critical section — an ``int``, not a ``bool``, precisely so a
        # *nested* window cannot commit the *outer* one early or clear the flag
        # out from under it (see `suspend_auto_commit` and `_skip_auto_commit`
        # below). Only the outermost `suspend_auto_commit` call ever commits,
        # rolls back, or brings this back to 0.
        self._skip_auto_commit_depth: int = 0
        # Every open `suspend_auto_commit` window (nested or not) registers a
        # fresh identity here for its duration — see `suspend_auto_commit` and
        # `_dispatch_derivation`. Paired with `_current_auto_commit_window`
        # below, this lets the derivation gate ask "is *this task's own*
        # window still open?" instead of "is *some* window open on this
        # store?", which `_skip_auto_commit_depth` alone cannot answer: that
        # counter is instance-wide, so a task with no window of its own would
        # otherwise read another task's open window as if it were its own.
        self._open_auto_commit_windows: set[object] = set()
        # Task-local marker naming the innermost `suspend_auto_commit` window
        # THIS task currently has open on THIS store, or `None` if it has none
        # open. A `contextvars.ContextVar`, created per-instance rather than
        # at module level — the same choice already made for
        # `_suppress_access_tracking` below — so a task holding windows on two
        # different store instances at once keeps each store's marker
        # independent: opening a window on store B must not overwrite store
        # A's own still-open window's identity for a task that holds both.
        # Set with a token at each window's entry and reset in its `finally`,
        # where the identity is also unregistered from
        # `_open_auto_commit_windows` — see `suspend_auto_commit`.
        self._current_auto_commit_window: contextvars.ContextVar[object | None] = (
            contextvars.ContextVar("engrava_current_auto_commit_window", default=None)
        )
        # Terminal quarantine state. Set only when a guarded write's own unit
        # (e.g. ``_write_readback_savepoint``, including a derived child's row
        # or edge insert) could not unwind its savepoint after it failed
        # (raised or was cancelled), so the long-lived connection may still
        # hold an open transaction. Quarantine is terminal by construction:
        #   * the flag makes guarded entry points + ``_maybe_commit`` fail fast;
        #   * ``_db`` is swapped for a ``_QuarantinedConnection`` proxy so every
        #     core-initiated op raises regardless of physical close; and
        #   * the shared revocation token (below) makes every *other* holder of
        #     the real connection (the JournalWriter) fail hard too.
        # Never cleared — recovery requires a fresh connection + store.
        self._connection_quarantined: bool = False
        self._quarantine_reason: str | None = None
        # Shared with the JournalWriter (and any future direct-connection holder)
        # so quarantine revokes them all synchronously, independent of the
        # best-effort physical close.
        self._revocation = ConnectionRevocationToken()
        # The task performing the real connection's *physical* close, however
        # it started -- quarantine's own detached best-effort close, or an
        # ordinary close() call that got there first. Doubles as the "a
        # physical close is already in progress" marker close() and
        # _quarantine_connection each check before starting a second,
        # independent one on the same underlying connection (see close()).
        # Also retains quarantine's own close task so it is neither GC'd
        # while pending nor reported as an unretrieved-exception task.
        self._quarantine_close_task: asyncio.Task[None] | None = None
        # Bound on how long close() waits for the worker to answer the
        # physical close before quarantining the store instead -- see
        # close() and _CLOSE_TIMEOUT_SECONDS for the derivation and what
        # expiry does.
        self._close_timeout_seconds: float = close_timeout_seconds
        self._fts_available: bool = False
        self._fts_probed: bool = False
        # Count of primary FTS5 ``MATCH`` executions that raised an
        # ``OperationalError`` (a malformed MATCH expression) before the
        # bare-mode fallback retry ran. Surfaced read-only via
        # :attr:`fts_match_failure_count` so an operator can detect that any
        # query is silently taking the sanitizing fallback path rather than
        # matching the expression as written.
        self._fts_match_failure_count: int = 0
        # Count of vector-arm searches that degraded to an empty result because
        # the query vector had no usable cosine direction (empty, all-zero, or
        # non-finite — see :func:`_query_vector_is_degenerate`). Surfaced
        # read-only via :attr:`vector_arm_degradation_count` so an operator can
        # detect that some queries are silently returning nothing because of a
        # bad query embedding rather than a genuinely empty neighbourhood. A
        # wrong-*dimension* vector is NOT counted here — it is a structural
        # contract violation raised as :class:`VectorDimensionMismatchError`.
        self._vector_arm_degradation_count: int = 0
        self._vector_backend: SqliteVecSearchBackend | None = None
        self._owns_connection: bool = False
        self._embedding_provider: EmbeddingProviderProtocol | None = embedding_provider
        self._auto_embed: bool = auto_embed and embedding_provider is not None
        self._require_embedding: bool = require_embedding
        # Scoped to a single ``bulk_store`` call: suppresses ``create_thought``'s
        # per-thought auto-embed so the batch path can embed all rows in one
        # provider call after the insert loop. False for every other caller.
        self._suppress_auto_embed: bool = False
        # Configuration objects are required to be *exactly* their class, not
        # instances of it. A subclass passes ``isinstance`` and is still free to
        # report one set of settings while it is validated and another every
        # time it is read afterwards - which puts a lying hygiene policy back on
        # the deletion path that owning its fields was meant to close. The
        # classes are imported here rather than at module scope because the
        # config module imports this one.
        from engrava.config import (  # noqa: PLC0415 -- deferred: the config module imports this one
            HygienePolicyConfig as _HygienePolicyConfig,
        )
        from engrava.config import (  # noqa: PLC0415 -- deferred: the config module imports this one
            MetricsConfig as _MetricsConfig,
        )
        from engrava.config import (  # noqa: PLC0415 -- deferred: the config module imports this one
            SearchConfig as _SearchConfig,
        )

        self._search_config: SearchConfig | None = require_exact_type_or_none(
            search_config, _SearchConfig, "SqliteEngravaCore.search_config"
        )
        self._journal_enabled: bool = journal_enabled
        self._journal: JournalWriter | None = (
            JournalWriter(db, revocation=self._revocation) if journal_enabled else None
        )
        # The TTL strategy decides archive-versus-physical-delete, and it is
        # resolved by value lookup against ``CleanupStrategy``, which consults
        # the value's own ``__hash__`` / ``__eq__``. A ``str`` subclass can
        # therefore read as ``archive`` everywhere it is inspected and still
        # select ``DELETE`` here, so the store owns the text it was handed.
        if not isinstance(ttl_strategy, str):
            msg = "ttl_strategy must be a string"
            raise ConfigError(msg)
        self._ttl_strategy: str = own_str(ttl_strategy)
        # These arrive as raw constructor arguments, not through a config
        # object, so nothing has decoded them. The cadence decides whether an
        # automatic cleanup runs at all, and under the ``delete`` strategy that
        # cleanup destroys rows: an ``int`` subclass whose real value is ``0``
        # ("off") but which answers ``< 1`` as false turns the feature on.
        self._ttl_check_every_n: int = require_int(
            ttl_check_every_n, "SqliteEngravaCore.ttl_check_every_n"
        )
        self._ttl_default_seconds: int | None = require_int_or_none(
            ttl_default_seconds, "SqliteEngravaCore.ttl_default_seconds"
        )
        if metrics_config is not None:
            self._metrics_config = require_exact_type(
                metrics_config, _MetricsConfig, "SqliteEngravaCore.metrics_config"
            )
        else:
            self._metrics_config = _MetricsConfig()
        self._latency_buffer = _LatencyRingBuffer(self._metrics_config.window_size)
        self._operation_count: int = 0
        self._manifests: tuple[ExtensionManifest, ...] = tuple(manifests)
        # Serialises the dedup ``check existing -> INSERT or UPDATE``
        # sequence so concurrent ``create_thought(deduplicate=True)`` calls
        # converge on a single row even though aiosqlite does not expose
        # row-level locking.  Acquired only on the dedup branch — the
        # legacy ``deduplicate=False`` path stays lock-free.
        self._dedup_lock: _DedupLock = _DedupLock()
        # Task-reentrant lock around every write path on the instance. Held
        # for the duration of each read-validate-write critical section —
        # for a single-item call (create_thought, update_thought, ...) not
        # nested in the same task's own suspend_auto_commit() window, never
        # across a slow/arbitrary step such as an embedding provider
        # request, an `on_store` hook, or a derived-records producer,
        # matching the discipline `_serialize_dedup_probe` already
        # established (see its own docstring). `bulk_store` is one
        # exception, by design: its batch embedding call runs *inside*
        # `suspend_auto_commit`'s window (the batch's atomicity spans
        # insert + embed + commit as one unit), so this lock is held
        # for that whole, genuinely network-bound round trip too. A
        # single-item call nested in an outer, already-open
        # suspend_auto_commit() window on the same task is a second
        # exception, incidentally rather than by design: this lock is
        # task-reentrant, so the outer window's hold never actually drops
        # while this call's own follow-up work (auto-embed, on_store,
        # derivation) runs — see `_finish_create_thought` — and a
        # *different* task's write still waits behind it for the whole
        # window, same as `bulk_store`'s case. See `_TaskReentrantLock` and
        # `_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS` for what that means for the
        # acquisition bound. Blocks a *different*
        # task's write for the duration; the *same* task re-enters for free
        # (see `_TaskReentrantLock`), which is what lets a write issued from
        # inside `suspend_auto_commit` complete instead of deadlocking on
        # itself.
        self._write_lock: _TaskReentrantLock = _TaskReentrantLock(
            acquire_timeout_seconds=write_lock_acquire_timeout_seconds
        )
        # Fires the recency-off nudge in ``recall`` at most once per instance.
        self._recency_nudge_emitted: bool = False
        # Live access substrate (feeds the dreaming ``frequency`` signal).
        # When enabled, retrieval paths buffer access events here — O(1), no DB
        # write on the read path — and the buffer is flushed in one batched
        # UPDATE at the consolidation-cycle boundary (and on explicit flush /
        # close). Off by default; ``from_config`` turns it on when
        # ``dreaming.enabled`` and ``dreaming.access_tracking_enabled``.
        self._access_tracking_enabled: bool = access_tracking_enabled
        self._access_buffer: _AccessBuffer = _AccessBuffer()
        # Suppresses access buffering for reads issued *by* consolidation
        # itself (its candidate scans / reflection-member resolution) and by a
        # read-only view. Those are not caller retrievals, so they must not feed
        # the frequency signal. Task-local (a ``ContextVar``, not a plain bool)
        # so overlapping suppressed reads on this store cannot clobber each
        # other's flag: each async task carries its own value and nesting is
        # token-scoped. Set only inside ``suppress_access_tracking``.
        self._suppress_access_tracking: contextvars.ContextVar[bool] = contextvars.ContextVar(
            "engrava_suppress_access_tracking",
            default=False,
        )
        # The backend-independent dreaming consolidator, supplied by the
        # composition root when enabled. ``None`` for a manually built store or
        # dreaming-off configuration. The legacy private attribute name is kept
        # to avoid disrupting existing diagnostic integrations. This is the one
        # slot both wiring routes target: ``from_config`` still writes it
        # directly, and every other caller has ``attach_dreaming_extension``
        # (below) as the supported door onto the same state.
        self._dreaming_extension: DreamingConsolidatorProtocol | None = None
        # Memory Hygiene (deterministic forgetting) policy. ``None`` (default)
        # or ``enabled=False`` ⇒ the forgetting loop never runs and no existing
        # read/write path changes. ``run_hygiene`` and the ``consolidate()``
        # convenience invocation both no-op when this is ``None``/disabled.
        self._hygiene_policy: HygienePolicyConfig | None = require_exact_type_or_none(
            hygiene_policy, _HygienePolicyConfig, "SqliteEngravaCore.hygiene_policy"
        )
        # Derived-records extension seam. ``enabled=False`` (the default) ⇒ the
        # seam is inert and every write path is byte-identical to a store
        # without it. When enabled *and* the hooks object implements
        # ``DerivedRecordProducerProtocol``, a successful source store is
        # followed by a core-controlled, guarded, per-child persistence of the
        # producer's derived records (see ``_dispatch_derivation``).
        self._derive_gates: DeriveGates = (
            require_exact_type_or_none(derive_gates, DeriveGates, "SqliteEngravaCore.derive_gates")
            or DeriveGates()
        )
        # Opt-in, runtime-only cognitive-cycle source. When set, the read /
        # eligibility paths pull ``current_cycle`` from it only when the caller
        # omitted one (an explicit ``current_cycle`` — including ``0`` — wins);
        # the pulled value is validated (``_validate_provider_cycle``). It is
        # never consulted for write-side cycle stamping and is never serialized
        # into config (a live object). ``None`` ⇒ today's behaviour unchanged.
        self._cycle_provider: CycleProvider | None = cycle_provider

    @property
    def fts_match_failure_count(self) -> int:
        """Return how many primary FTS5 ``MATCH`` executions have failed.

        Incremented once each time :meth:`search_fts` runs a normalized query
        whose ``MATCH`` raises an ``OperationalError`` (a malformed FTS5
        expression), *before* the bare-mode fallback retry. A non-zero, growing
        value signals that some queries are silently degrading to the
        sanitizing fallback path instead of matching the expression as written
        — useful as an operational health signal. The fallback still serves the
        query, so a non-zero count never means results were lost.

        Returns:
            The cumulative primary-``MATCH`` failure count for this store
            instance (monotonically non-decreasing, reset only by
            constructing a new store).

        """
        return self._fts_match_failure_count

    @property
    def vector_arm_degradation_count(self) -> int:
        """Return how many vector-arm searches degraded to an empty result.

        Incremented once each time :meth:`search_similar` is called with a
        *degenerate* query vector — one with no usable cosine direction: empty,
        all-zero (the canonical shape produced by auto-embedding empty or
        stop-word-only text), or carrying a non-finite (``NaN``/``inf``)
        component. Such a query cannot rank anything, so the arm returns ``[]``;
        this counter surfaces that silent degradation as an operational health
        signal, exactly mirroring :attr:`fts_match_failure_count` for the FTS
        arm. A non-zero, growing value means some queries are producing bad
        embeddings, not that the corpus is empty.

        A wrong-*dimension* query vector is deliberately **not** counted here: it
        is a structural caller-contract violation and is raised loudly as
        :class:`~engrava.domain.exceptions.VectorDimensionMismatchError` rather
        than degraded.

        Returns:
            The cumulative degenerate-query-vector count for this store instance
            (monotonically non-decreasing, reset only by constructing a new
            store).

        """
        return self._vector_arm_degradation_count

    @property
    def journal(self) -> JournalWriter | None:
        """Return the ``JournalWriter`` if journaling is enabled, else ``None``.

        **A direct call to the returned writer's own ``append()`` is not a
        guarded write path.** Every mutation this store itself journals runs
        the append under the same in-process write lock as the write it
        describes (see the concurrency documentation). ``JournalWriter`` does
        not hold that lock itself — it only serialises its own
        ``sequence_number`` allocation against other appends on the same
        connection — so a caller appending directly through this property,
        outside of any guarded write this store performs, can still land
        inside another task's open transaction-deferral window and be rolled
        back with it. Prefer letting the store journal its own mutations;
        treat a direct ``append()`` call the same as any other unmediated use
        of the underlying connection.

        Returns:
            The active journal writer, or ``None``.

        """
        return self._journal

    @property
    def _skip_auto_commit(self) -> bool:
        """Return whether a ``suspend_auto_commit`` window is currently open.

        Backed by :attr:`_skip_auto_commit_depth` rather than a bare ``bool`` so
        a *nested* ``suspend_auto_commit`` call cannot make this ``False`` while
        an enclosing one is still open — see that method's docstring for the two
        failures a plain ``bool`` produced under nesting.

        Returns:
            ``True`` while at least one ``suspend_auto_commit`` window — nested
            or not — is open on this task's current critical section.

        """
        return self._skip_auto_commit_depth > 0

    async def verify_journal(self) -> JournalIntegrityResult:
        """Verify the persisted hash-chain journal on disk.

        Walks every ``journal_entry`` row in ``sequence_number`` order,
        recomputes each SHA-256 hash, and checks the parent-hash linkage,
        delegating to :meth:`JournalWriter.verify_integrity`.

        The check reads the recorded chain **independent of whether
        journaling is currently enabled**. Entries may have been written in
        an earlier session with journaling on and the store reopened with it
        off (:attr:`journal` is then ``None``); those recorded entries must
        still be auditable, so when there is no active writer this constructs
        a transient, read-only :class:`JournalWriter` over the same
        connection to run the walk. An absent or empty chain verifies as
        ``valid=True`` with ``entries_checked=0``.

        The walk verifies **linkage, not length**: a hash chain cannot detect a
        truncated *tail* (the newest entries removed, or a crash before the
        final flush), because the remaining prefix stays internally consistent.
        Detecting a missing tail needs an external high-water-mark and is out of
        scope here. Mid-chain tampering, deletion, and reordering are all caught.

        It also verifies **ordering and content, not timestamps**: neither
        ``created_at`` nor ``entry_id`` is in the hash preimage, so a journal with
        every timestamp rewritten verifies exactly like an untouched one — and a
        timestamp moved across a ``journal.get_entries(since=...)`` bound leaves
        or enters that window, which filters on the same uncovered column.

        Returns:
            A :class:`JournalIntegrityResult` describing chain validity —
            ``valid`` plus ``entries_checked``, and on a break the
            ``first_invalid_sequence`` and ``error_message``.

        Examples:
            >>> result = await store.verify_journal()  # doctest: +SKIP
            >>> result.valid  # doctest: +SKIP
            True

        """
        journal = (
            self._journal
            if self._journal is not None
            else JournalWriter(self._db, revocation=self._revocation)
        )
        return await journal.verify_integrity()

    async def _record_search_latency(self, latency_ms: float) -> None:
        """Record a completed public-search latency when metrics are enabled."""
        if self._metrics_config.enabled and not _SUPPRESS_SEARCH_METRICS.get():
            await self._latency_buffer.record(latency_ms)

    async def _main_db_path(self) -> Path | None:
        """Resolve the main SQLite file path from the active connection."""
        cursor = await self._db.execute("PRAGMA database_list")
        rows = await cursor.fetchall()
        for row in rows:
            if str(row[1]) == "main" and row[2]:
                return Path(str(row[2]))
        return None

    async def _storage_footprint(self) -> tuple[int, int, int, int]:
        """Return ``(db_bytes, wal_bytes, vec_index_bytes, total_bytes)``."""
        db_path = await self._main_db_path()
        if db_path is None:
            return (0, 0, 0, 0)

        db_bytes = db_path.stat().st_size if db_path.exists() else 0
        wal_path = Path(f"{db_path}-wal")
        wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0  # noqa: ASYNC240
        vec_index_bytes = 0
        total_bytes = db_bytes + wal_bytes + vec_index_bytes
        return (db_bytes, wal_bytes, vec_index_bytes, total_bytes)

    async def metrics(self) -> EngravaMetrics:
        """Return a point-in-time snapshot of store health and workload metrics."""
        import time  # noqa: PLC0415

        from engrava.domain.models.metrics import (  # noqa: PLC0415
            EdgeCounts,
            EngravaMetrics,
            StorageFootprint,
            ThoughtCounts,
        )

        snapshot_ts = time.time()
        if not self._metrics_config.enabled:
            return EngravaMetrics(snapshot_timestamp=snapshot_ts)

        thought_by_type_cursor = await self._db.execute(
            "SELECT thought_type, COUNT(*) FROM thought GROUP BY thought_type"
        )
        thought_by_type_rows = await thought_by_type_cursor.fetchall()
        thought_by_type = {str(row[0]): int(row[1]) for row in thought_by_type_rows}

        thought_by_status_cursor = await self._db.execute(
            "SELECT lifecycle_status, COUNT(*) FROM thought GROUP BY lifecycle_status"
        )
        thought_by_status_rows = await thought_by_status_cursor.fetchall()
        thought_by_status = {str(row[0]): int(row[1]) for row in thought_by_status_rows}

        thought_total_cursor = await self._db.execute("SELECT COUNT(*) FROM thought")
        thought_total_row = await thought_total_cursor.fetchone()
        thought_total = int(thought_total_row[0]) if thought_total_row is not None else 0

        edge_by_type_cursor = await self._db.execute(
            "SELECT edge_type, COUNT(*) FROM edge GROUP BY edge_type"
        )
        edge_by_type_rows = await edge_by_type_cursor.fetchall()
        edge_by_type = {str(row[0]): int(row[1]) for row in edge_by_type_rows}

        edge_total_cursor = await self._db.execute("SELECT COUNT(*) FROM edge")
        edge_total_row = await edge_total_cursor.fetchone()
        edge_total = int(edge_total_row[0]) if edge_total_row is not None else 0

        db_bytes, wal_bytes, vec_index_bytes, total_bytes = await self._storage_footprint()
        latency_snapshot = await self._latency_buffer.snapshot()

        return EngravaMetrics(
            snapshot_timestamp=snapshot_ts,
            measured=True,
            thoughts=ThoughtCounts(
                by_type=thought_by_type,
                by_status=thought_by_status,
                total=thought_total,
            ),
            edges=EdgeCounts(by_type=edge_by_type, total=edge_total),
            storage=StorageFootprint(
                db_bytes=db_bytes,
                wal_bytes=wal_bytes,
                vec_index_bytes=vec_index_bytes,
                total_bytes=total_bytes,
            ),
            search_latency=latency_snapshot,
        )

    async def max_cycle(self) -> int:
        """Return the store's cognitive-cycle high-water mark.

        The maximum cognitive cycle across **every** cycle-bearing record —
        ``MAX(thought.updated_cycle)`` unioned with ``MAX(edge.created_cycle)``
        — i.e. the true store high-water mark. It is *not* thought-only: an edge
        created at a higher cycle than any thought would otherwise under-report
        the mark, so both record kinds are unioned.

        A read-only recovery accessor: a consumer that advances its own
        cognitive cycle can resume its counter from this value across process
        restarts (and it backs
        :class:`~engrava.cycle_providers.MaxCycleProvider`). On an empty store —
        or one where every record is stamped cycle ``0`` (the chicken-and-egg
        case: a writer that never advances the cycle recovers ``0``) — it
        returns ``0``.

        Returns:
            The maximum cognitive cycle stored, or ``0`` when the store holds no
            cycle-bearing records.

        """
        # COALESCE folds the all-NULL empty-store case (and a store with no
        # edges, whose MAX(created_cycle) is NULL) to 0. Fixed SQL, no params.
        cursor = await self._db.execute(
            "SELECT COALESCE(MAX(high), 0) FROM ("
            "  SELECT MAX(updated_cycle) AS high FROM thought"
            "  UNION ALL"
            "  SELECT MAX(created_cycle) AS high FROM edge"
            ")"
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    def _resolve_current_cycle(self, current_cycle: int | None) -> int | None:
        """Resolve the effective cognitive cycle for a read / eligibility path.

        The **single** resolution point for the cycle-consuming paths, so the
        rule lives in one place. Order (never truthiness — an explicit ``0`` is
        a valid cycle and must win, never fall through):

        1. ``current_cycle is not None`` → use it as passed (unchanged).
        2. else, a ``cycle_provider`` is configured → pull and **validate** its
           value (:func:`_validate_provider_cycle`).
        3. else ``None`` — today's behaviour (recency signal off; no
           age-gating). No invented default.

        Args:
            current_cycle: The cycle the caller passed for this call, or
                ``None`` to defer to the configured provider (if any).

        Returns:
            The resolved cycle, or ``None`` when no cycle was passed and no
            provider is configured.

        Raises:
            CycleProviderError: When a configured provider returns an invalid
                value (not an ``int``, a ``bool``, or negative).

        """
        if current_cycle is not None:
            return current_cycle
        if self._cycle_provider is None:
            return None
        return _validate_provider_cycle(self._cycle_provider.current_cycle())

    def _require_current_cycle(self, current_cycle: int | None, *, operation: str) -> int:
        """Resolve a cycle for a path that cannot run without one.

        Wraps :meth:`_resolve_current_cycle` for ``consolidate`` / ``run_hygiene``,
        whose age-gating arithmetic genuinely needs a cycle. When neither an
        explicit ``current_cycle`` nor a configured provider yields one, it
        raises rather than inventing a default (``0`` would silently make every
        record look equally fresh).

        Args:
            current_cycle: The cycle the caller passed, or ``None``.
            operation: The public method name, for the error message.

        Returns:
            The resolved cycle as an ``int``.

        Raises:
            ValueError: When no cycle is available (no explicit argument and no
                configured provider).
            CycleProviderError: When a configured provider returns an invalid
                value.

        """
        resolved = self._resolve_current_cycle(current_cycle)
        if resolved is None:
            msg = (
                f"{operation} requires a cognitive cycle: pass current_cycle=... "
                f"or configure a cycle_provider on the store."
            )
            raise ValueError(msg)
        return resolved

    # ------------------------------------------------------------------
    # Factory + async context manager
    # ------------------------------------------------------------------

    @classmethod
    async def from_config(
        cls,
        config_path: str | Path,
        *,
        cycle_provider: CycleProvider | None = None,
        write_lock_acquire_timeout_seconds: float = _WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS,
        close_timeout_seconds: float = _CLOSE_TIMEOUT_SECONDS,
    ) -> Self:
        """Create a fully configured instance from a YAML config file.

        The returned instance **owns** the database connection and should
        be used as an async context manager to ensure proper cleanup::

            async with await SqliteEngravaCore.from_config("engrava.yaml") as store:
                thought = await store.get_thought("abc")

        The manual constructor ``SqliteEngravaCore(db, hooks=...)``
        still works unchanged — the caller owns the connection in that case.

        Args:
            config_path: Filesystem path to ``engrava.yaml``.
            cycle_provider: Optional, **runtime-only** opt-in cognitive-cycle
                source, forwarded verbatim to the constructor. A provider is a
                live object, so it is **never** read from (or written to) the
                config file — it is supplied here as a runtime keyword. ``None``
                (default) preserves today's behaviour. See the constructor's
                ``cycle_provider`` for the resolution and read-time-only rules.
            write_lock_acquire_timeout_seconds: Forwarded verbatim to the
                constructor — see its own docstring. Not read from the config
                file (it is a runtime tuning knob, not corpus-affecting
                configuration).
            close_timeout_seconds: Forwarded verbatim to the constructor —
                see its own and :meth:`close`'s docstrings. Not read from the
                config file, for the same reason as
                ``write_lock_acquire_timeout_seconds`` above.

        Returns:
            A configured ``SqliteEngravaCore`` with schema applied.

        Raises:
            ConfigError: If the config file is invalid.
            JournalIntegrityError: If ``journal.verify_on_open`` is enabled
                and the persisted hash chain fails verification.

        """
        from engrava.config import load_config, resolve_hooks  # noqa: PLC0415

        config = load_config(config_path)
        db = await aiosqlite.connect(str(config.database_path))
        try:
            if config.wal_mode:
                await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            # synchronous=NORMAL is the documented-safe companion to WAL: the
            # database stays durable across an application crash and is only at
            # risk of losing the most recent transactions on an OS crash or
            # power loss, which is the standard recommendation for WAL.
            await db.execute("PRAGMA synchronous=NORMAL")
            # busy_timeout makes a second connection wait (up to 5s) for a lock
            # instead of failing immediately with SQLITE_BUSY.
            await db.execute("PRAGMA busy_timeout=5000")
            db.row_factory = aiosqlite.Row

            hooks = resolve_hooks(config.hooks_class)

            # Resolve embedding provider from config.
            from engrava.config import (  # noqa: PLC0415
                resolve_embedding_provider,
                resolve_manifests,
            )

            emb_provider = resolve_embedding_provider(config.embeddings)
            auto_embed = config.embeddings.auto_embed if config.embeddings else False
            require_embedding = config.embeddings.require_embedding if config.embeddings else False

            manifests = resolve_manifests(
                config.extension_manifest_paths,
                discover=config.extension_discover,
            )

            # Access tracking feeds the dreaming ``frequency`` signal. It is on
            # only when dreaming is enabled AND its ``access_tracking_enabled``
            # flag is set (the default). With dreaming off, tracking stays off,
            # so the retrieval and scoring paths are byte-identical to today.
            access_tracking_enabled = (
                config.dreaming is not None
                and config.dreaming.enabled
                and config.dreaming.access_tracking_enabled
            )

            store = cls(
                db,
                hooks=hooks,
                embedding_provider=emb_provider,
                auto_embed=auto_embed,
                require_embedding=require_embedding,
                search_config=config.search,
                journal_enabled=config.journal.enabled,
                ttl_strategy=config.ttl.strategy,
                ttl_check_every_n=config.ttl.check_every_n_operations,
                ttl_default_seconds=config.ttl.default_ttl_seconds,
                metrics_config=config.metrics,
                manifests=manifests,
                access_tracking_enabled=access_tracking_enabled,
                hygiene_policy=config.hygiene_policy,
                derive_gates=config.derive,
                cycle_provider=cycle_provider,
                write_lock_acquire_timeout_seconds=write_lock_acquire_timeout_seconds,
                close_timeout_seconds=close_timeout_seconds,
            )
            store._owns_connection = True

            # The composition root owns concrete optional implementations; the
            # SQLite facade retains only the inward consolidator contract. Keep
            # construction after the store, matching the established error and
            # cleanup ordering, and avoid importing Dreaming when it is disabled.
            if config.dreaming is not None and config.dreaming.enabled:
                from engrava._composition import (  # noqa: PLC0415
                    compose_dreaming_consolidator,
                )

                store._dreaming_extension = compose_dreaming_consolidator(config.dreaming)

            await store.ensure_schema()

            # Opt-in on-open integrity check. Runs only when explicitly
            # enabled and only after the schema is ensured, so the
            # ``journal_entry`` table is guaranteed to exist. Default-off ⇒
            # the open path is byte-identical to before when disabled.
            if config.journal.verify_on_open:
                integrity = await store.verify_journal()
                if not integrity.valid:
                    # Raised inside the try so the enclosing handler closes the
                    # connection before propagating — a leaked handle on a
                    # rejected open would otherwise pin the WAL.
                    raise JournalIntegrityError(  # noqa: TRY301
                        integrity.first_invalid_sequence,
                        integrity.error_message,
                    )

            await store._configure_vector_backend(
                backend_name=config.vector_backend,
                embedding_dimension=config.embedding_dimension,
            )
        except BaseException:
            # Not ``except Exception``: ``asyncio.CancelledError`` derives
            # from ``BaseException``, and a cancellation during any await
            # above must close ``db`` exactly like an ordinary failure does
            # — otherwise it leaks aiosqlite's non-daemon connection worker
            # thread just as an uncaught error during open would. Routed
            # through ``_close_quietly`` rather than a direct
            # ``await db.close()`` so a failure in the close itself cannot
            # replace this exception -- see that function's docstring.
            await _close_quietly(db)
            raise

        return store

    async def close(self) -> None:
        """Close the database connection if owned by this instance.

        Flushes any pending access-buffer events first (best-effort — a flush
        failure never blocks the close, and neither does a cancellation
        arriving while the flush is in flight, nor does the flush's own
        bound expiring — see the bounded-wait paragraph below), then closes
        the connection when this instance owns it. No-op on the connection
        when it is caller-managed (i.e. created via the manual constructor).

        **Coordinates with a concurrent or prior quarantine.**
        :meth:`_quarantine_connection` schedules its own physical close of
        the real connection, detached and un-awaited, so that it can return
        promptly even if that close hangs. If this call finds
        :attr:`_quarantine_close_task` already set — quarantine got here
        first, or does so while this call is in flight — it awaits that
        same task instead of issuing a second, independent ``close()`` on
        the same underlying connection: two concurrent closes on the pinned
        aiosqlite version can each enqueue their own stop sentinel to the
        worker thread, which exits on the first and can leave the other
        caller's future unresolved forever. Only one physical close is ever
        entered, whichever caller (this one, or quarantine) gets there
        first.

        **Both branches drain the task under :meth:`_drain_shielded`,
        symmetrically** — the branch that creates the task is not exempt
        just because it is the one that "owns" it. A bare ``await`` on the
        task would propagate *this call's own* cancellation into the task
        (unlike a shielded await), which can cancel the physical close
        before ``_db.close()`` has meaningfully run at all — and that
        half-run, cancelled task would still be ``done()``, so it would sit
        in :attr:`_quarantine_close_task` looking exactly like a completed
        close to every later caller that drains it, none of them any wiser
        that the real connection was never actually closed. Draining under
        ``_drain_shielded`` closes that off structurally rather than
        detecting it afterwards: it re-shields on every repeated
        cancellation of the awaiting coroutine, so this call's own
        cancellation — however many times it happens — never reaches the
        task itself. It returns once the task is genuinely ``done()``, or
        once the bound below expires with the task still pending, whichever
        comes first; either way, draining never cancels the task, so the
        physical close keeps running toward real completion — success or a
        real failure — in the background.

        **What differs between the two branches is what happens next, not
        how safely they wait.** When this call is the one that created the
        task (the ordinary, non-quarantined close), its outcome is
        actionable: a genuine close failure propagates via
        :meth:`asyncio.Task.result`, exactly as an unshielded direct
        ``await`` would have surfaced it. When this call is piggybacking on
        a task quarantine already started, the outcome is discarded instead
        — the connection is already terminally unusable via the quarantine
        proxy, so nothing about how its close turned out is actionable to
        this caller. Either way, a cancellation of *this specific await* —
        whoever is awaiting ``close()`` being itself cancelled — is never
        discarded and always propagates first, ahead of whatever the task
        itself resolved to; the shared task keeps running to completion
        regardless, so nothing is left half-closed by letting the
        cancellation through.

        **The wait on the worker is bounded, not indefinite -- applied
        twice in sequence, once to the flush above and once to the physical
        close below, not once as a shared budget for the whole call.** The
        access-buffer flush awaits the same worker directly (through
        ``_write_lock`` and a raw ``executemany``), so it is wrapped in its
        own ``asyncio.wait_for(..., timeout=self._close_timeout_seconds)``
        immediately above; an unresponsive worker would otherwise hang
        there first, before the physical-close bound described next is
        ever reached, silently defeating it for every store with access
        tracking on. A flush timeout is just another flush failure to this
        call -- caught the same way, for the same reason (the buffered
        counts are documented best-effort telemetry, self-healing after a
        lost flush). **Because each wait gets the full
        :attr:`_close_timeout_seconds` independently, one call to this
        method can take up to twice that value** (60s at the default) when
        the worker never answers at all -- up to the full bound stuck in
        the flush, then up to the full bound again stuck in the close --
        not :attr:`_close_timeout_seconds` itself as a per-call ceiling.

        The physical close is bounded the same way: both branches drain
        the task under :attr:`_close_timeout_seconds` (default
        :data:`_CLOSE_TIMEOUT_SECONDS`, configurable via the constructor /
        :meth:`from_config`) rather than forever. A worker that is merely
        slow still gets that whole window to answer — the task itself is
        never cancelled by the bound, only *this call's observation of it*
        stops, so a slow-but-finite close keeps running to completion behind
        the scenes exactly as it would without a bound. A worker that never
        answers at all leaves the task not ``done()`` when the bound
        expires; this call then hands off to :meth:`_quarantine_connection`
        with that same task already installed in
        :attr:`_quarantine_close_task`, so quarantine's own coordination
        (above) sees a physical close already in flight and defers to it
        rather than starting a second one — an expired bound never causes a
        second physical close, whether this is the first ``close()`` on this
        store or a later one piggybacking on an earlier expiry. The store is
        left terminally unusable either way (the same
        :class:`~engrava.domain.exceptions.ConnectionQuarantinedError` every
        other quarantine path raises, not a new sibling state), because the
        worker's last operation never reported and the connection's true
        state is unknown from here. This call itself raises that same error
        once the hand-off completes — unless a cancellation of this call's
        own await was already pending (from the flush above, or from
        draining the task), which outranks a *discovered* expiry exactly as
        it outranks a discovered close failure elsewhere in this method: the
        expiry is logged instead, and the pending cancellation is what
        propagates.

        **Bounding this wait does not bound the ordinary ~20s residual delay
        measured behind this bound, and for that delay the reason is not
        the worker thread.** A daemon and a non-daemon worker took the same
        ~20s to exit, and by the time that residual wait is even observed
        the worker thread has already finished — that delay lives inside
        the interpreter's own async-runtime shutdown sequence, which runs
        *after* this method (and the rest of your code) has already
        returned control, not in anything ``close()`` is waiting on. That
        measurement did not test, and does not rule out, a worker that
        never finishes at all: aiosqlite creates its connection's worker
        thread non-daemon, so a genuinely wedged worker leaves that thread
        running indefinitely even after this method has abandoned the wait
        and raised :class:`ConnectionQuarantinedError` — and a live
        non-daemon thread is what keeps a Python process from exiting,
        regardless of the async-runtime delay above. Neither delay is
        something this method — or any bound it applies — can fix; see
        ``docs/deployment.md`` for what a caller can do about it.

        Raises:
            ConnectionQuarantinedError: When this call's own wait for the
                worker exceeds :attr:`_close_timeout_seconds` before a
                cancellation of this call does. The store is unusable from
                this point on regardless of whether the worker eventually
                does answer.

        """
        # A cancellation during the flush must not skip the close below --
        # that would strand the connection's non-daemon worker exactly like
        # an uncaught error would, contradicting this method's own "a flush
        # failure never blocks the close" promise (a cancellation is a kind
        # of failure too). Caught here and re-raised only once the close
        # below has actually run, mirroring
        # :meth:`EngravaManager.close_all`: every required cleanup step
        # still runs, and the cancellation is deferred, never discarded.
        pending_cancellation: asyncio.CancelledError | None = None
        if self._access_tracking_enabled:
            try:
                # Bounded on the same budget as the physical close below --
                # this also awaits the worker directly (via ``_write_lock``
                # + ``self._db.executemany``), so an unresponsive worker
                # would otherwise hang here before the close-task bound is
                # ever reached, defeating it for every store with access
                # tracking on. A timeout is just another flush failure to
                # this call: caught by the same ``except Exception`` below,
                # exactly as "a flush failure never blocks the close"
                # already promises, since the buffered counts are
                # documented best-effort telemetry a lost flush self-heals
                # from -- there is nothing here worth preserving in the
                # background the way the physical close is preserved.
                await asyncio.wait_for(
                    self.flush_access_buffer(), timeout=self._close_timeout_seconds
                )
            except asyncio.CancelledError as exc:
                pending_cancellation = exc
            except Exception:  # noqa: BLE001
                logger.debug("access-buffer flush on close failed; counts are best-effort")
        if self._owns_connection:
            task, is_new_task = self._resolve_close_task()
            cancel_error = await self._drain_shielded(
                task, timeout_seconds=self._close_timeout_seconds
            )
            pending_cancellation = await self._finish_close_wait(
                task,
                is_new_task=is_new_task,
                cancel_error=cancel_error,
                pending_cancellation=pending_cancellation,
            )
        if pending_cancellation is not None:
            raise pending_cancellation

    def _resolve_close_task(self) -> tuple[asyncio.Task[None], bool]:
        """Return the physical-close task for this call to drain.

        Reuses :attr:`_quarantine_close_task` when a physical close is
        already in flight -- started by a concurrent quarantine, or by a
        prior ``close()`` call this one is piggybacking on -- rather than
        ever starting a second, independent close on the same connection
        (see :meth:`close` for why two concurrent closes on the pinned
        aiosqlite version can leave a caller's future unresolved forever).
        Only when nothing is in flight yet does this start the real,
        non-quarantined close and install it.

        Returns:
            The task to drain, and whether this call is the one that just
            created it (``True``) versus piggybacking on one that already
            existed (``False``) -- :meth:`_finish_close_wait` uses this to
            decide whether the task's outcome is actionable to this caller.

        """
        existing_task = self._quarantine_close_task
        if existing_task is not None:
            return existing_task, False
        task = asyncio.ensure_future(self._db.close())
        self._quarantine_close_task = task
        return task, True

    async def _finish_close_wait(
        self,
        task: asyncio.Task[None],
        *,
        is_new_task: bool,
        cancel_error: asyncio.CancelledError | None,
        pending_cancellation: asyncio.CancelledError | None,
    ) -> asyncio.CancelledError | None:
        """Interpret one drain of the physical-close task and act on it.

        The state transition in outcome 1 always happens when the bound has
        expired, regardless of any cancellation -- cancellation precedence
        (outcomes 1 and 2) only decides which exception ultimately
        propagates, never whether the store gets quarantined. Three
        outcomes, most-authoritative first for that reason:

        1. **The bound expired before the worker answered** (``task`` still
           not ``done()``): quarantines via :meth:`_abandon_expired_close`
           (never a second physical close -- see that method) unconditionally,
           whether or not a cancellation is also pending -- including a
           cancellation of *this call's own await while draining the task*
           (``cancel_error``), which does not short-circuit the quarantine.
           Then either raises
           :class:`~engrava.domain.exceptions.ConnectionQuarantinedError`, or,
           if a cancellation was already pending -- ``cancel_error`` from
           draining just now, or ``pending_cancellation`` deferred earlier in
           :meth:`close` (the access-buffer flush), ``cancel_error`` preferred
           when both are set -- only logs the expiry and returns that
           cancellation instead.
        2. **The task finished within the bound, but this call's own await
           was cancelled while draining it** (``cancel_error`` set): returned
           as the new ``pending_cancellation`` -- it outranks whatever the
           task itself resolved to.
        3. **The task finished within the bound, with no cancellation of this
           call's own drain**: a genuine outcome is surfaced only when this
           call is the one that created the task (``is_new_task``) -- a
           piggybacked task's outcome is never actionable to this caller,
           exactly as before this bound existed.

        Args:
            task: The physical-close task that was just drained.
            is_new_task: Whether this call created ``task`` itself, per
                :meth:`_resolve_close_task`.
            cancel_error: Whatever :meth:`_drain_shielded` returned for this
                drain.
            pending_cancellation: Any cancellation already deferred earlier
                in :meth:`close` (the access-buffer flush).

        Returns:
            The cancellation that should ultimately propagate from
            :meth:`close`, if any.

        Raises:
            ConnectionQuarantinedError: Per outcome 1 above.

        """
        if not task.done():
            # The bound expired before the worker answered -- slow or
            # genuinely wedged, this call cannot tell which and does not
            # need to. See close()'s own docstring for why this always
            # quarantines (never a second physical close) and raises the
            # same error every other quarantined path already raises.
            # This runs even when `cancel_error` is set: `_drain_shielded`
            # re-shields and keeps waiting out its own bound on a
            # cancellation of our await, so a cancelled drain and a genuine
            # expiry can coincide -- cancellation only decides which
            # exception propagates next, never whether this transition
            # happens.
            await self._abandon_expired_close(task)
            propagating_cancellation = (
                cancel_error if cancel_error is not None else pending_cancellation
            )
            if propagating_cancellation is None:
                raise ConnectionQuarantinedError(
                    self._quarantine_reason
                    or f"close() did not complete within {self._close_timeout_seconds:.1f}s"
                )
            logger.warning(
                "close() exceeded its %.1fs bound while a cancellation was "
                "already pending; the store is now quarantined",
                self._close_timeout_seconds,
            )
            return propagating_cancellation
        if cancel_error is not None:
            return cancel_error
        if is_new_task:
            # This is the real, non-quarantined close -- unlike the
            # piggyback case, its own outcome is actionable, so it is
            # surfaced (not discarded): a clean close returns None, an
            # ordinary failure or an independent cancellation of the task
            # itself both raise via result().
            if pending_cancellation is None:
                task.result()
            else:
                # A cancellation was already deferred above (from the
                # flush) -- see :meth:`_log_close_failure_over_pending_cancellation`
                # for the rule this follows.
                self._log_close_failure_over_pending_cancellation(task)
        # else: piggybacking on a task that finished within this call's own
        # bound -- outcome discarded, exactly as before this bound existed.
        return pending_cancellation

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        Returns:
            This store instance.

        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit the async context manager and close if owned.

        When the ``async with`` body already raised, that is what the
        caller needs to see, so a failure from :meth:`close` here is
        secondary: logged rather than allowed to replace it, the same rule
        :func:`_close_quietly` applies to a bare ``await conn.close()``
        during cleanup, applied here to :meth:`close` itself since this
        call has no raw connection of its own to route through that
        helper. A cancellation reaching this ``__aexit__`` call is a
        distinct signal, not a mere close failure, and is never swallowed.
        On a clean exit (no exception from the body), a close failure is
        not secondary to anything -- it is the only error there is, so it
        propagates normally.

        Args:
            *exc: Exception info (type, value, traceback).

        """
        if exc[1] is None:
            await self.close()
            return
        try:
            await self.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Error closing connection in __aexit__ while the context "
                "body's exception was propagating",
                exc_info=True,
            )

    async def _configure_vector_backend(
        self,
        *,
        backend_name: str,
        embedding_dimension: int,
    ) -> None:
        """Configure the requested vector backend with graceful fallback.

        Args:
            backend_name: Backend identifier from config.
            embedding_dimension: Expected embedding vector dimension.

        """
        backend_handlers = {
            "numpy": self._configure_numpy_vector_backend,
            "sqlite-vec": self._configure_sqlite_vec_vector_backend,
        }
        # Which handler runs is decided by ``__hash__`` / ``__eq__`` on the name,
        # so a ``str`` subclass can read as one backend everywhere it is checked
        # and select the other here. Both entry points reach this line — the
        # config, which validates the name, and the manager, which does not — so
        # the choke point owns the text rather than either of them.
        if not isinstance(backend_name, str):
            msg = "vector_backend must be a string"
            raise ConfigError(msg)
        await backend_handlers[own_str(backend_name)](embedding_dimension)

    async def _configure_numpy_vector_backend(self, embedding_dimension: int) -> None:
        """Use the default numpy brute-force vector search backend.

        Args:
            embedding_dimension: Expected embedding vector dimension.

        """
        del embedding_dimension
        self._vector_backend = None

    async def _configure_sqlite_vec_vector_backend(self, embedding_dimension: int) -> None:
        """Configure the sqlite-vec backend when the extension is available.

        Falls back to numpy when the extension cannot be loaded.

        Args:
            embedding_dimension: Expected embedding vector dimension.

        """
        from engrava.infrastructure.sqlite.vector_sqlite_vec import (  # noqa: PLC0415
            SqliteVecSearchBackend,
            load_sqlite_vec,
        )

        loaded = await load_sqlite_vec(self._db)
        if not loaded:
            logger.warning("sqlite-vec requested but unavailable — using numpy fallback")
            self._vector_backend = None
            return

        backend = SqliteVecSearchBackend(embedding_dimension)
        await backend.ensure_index(self._db)
        await backend.sync_embeddings(self._db)
        self._vector_backend = backend

    # ------------------------------------------------------------------
    # Schema bootstrap (standalone usage)
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create core tables if they don't already exist.

        **Write-lock classification: deliberately outside `_write_lock`,
        bucket 2 — schema bootstrap, not a guarded runtime write path.**
        Every write this method performs directly, or reaches through
        ``_run_pending_core_migrations`` / ``_core_migration_steps`` /
        each ``_migrate_core_v*_to_v*`` step / ``_rebuild_fts_index`` /
        ``_recreate_child_tables_with_fk_atomically`` and the three
        ``_recreate_*_with_fk`` helpers it calls / ``_purge_orphan_children``
        / the ``PRAGMA user_version`` stamps, is schema DDL (plus, in a
        handful of migrations, a one-time data backfill tied to that DDL) run
        while the store is being opened — before ``ensure_schema`` returns,
        the schema this store depends on for every other guarded write does
        not yet reliably exist, so no concurrent guarded write can be
        meaningfully in flight yet. This is a property of *when* these calls
        happen (once, at open, awaited to completion before the store is
        handed to any other caller), not a habit — engrava does not support
        calling ``ensure_schema`` concurrently with itself or with guarded
        writes on the same instance, and nothing about ``_write_lock`` would
        make that safe even if it held it (schema DDL under a data-row lock
        is a different problem this lock does not solve). See
        ``docs/deployment.md`` for the "open once, then share" contract this
        relies on.

        Applies the full ``schema_core.sql`` (including the FTS5 virtual table
        and sync triggers) only when the database predates the
        migration-ladder floor **and no core table holds a row yet** (see
        :meth:`_has_any_core_table` — this is row presence, not mere table
        existence, so an empty table an interrupted bootstrap left behind
        does not itself count as "populated") — a sub-floor database that
        already has a populated core table refuses instead (see
        :class:`SchemaVersionError`). Before that script runs, whichever core
        tables already exist there (necessarily still empty, or the refusal
        above already fired) are also checked against it (see
        :meth:`_existing_core_tables_match_bootstrap_shape`): one already
        carrying an older shape the script's ``CREATE ... IF NOT EXISTS``
        statements would leave untouched is refused up front, before
        anything is written, rather than only discovered after a stamp the
        script's own last statement already committed. A database at or
        above the floor is upgraded incrementally through the ordered
        core-migration registry (see :meth:`_core_migration_steps`) up to the
        head version (:data:`CORE_SCHEMA_HEAD_VERSION`). A database stamped
        **above** head also refuses rather than silently skipping every
        migration step and opening as though it were current.

        After core schema creation or upgrade, probes for the ``thought_fts``
        table and then runs any pending extension schema migrations for each
        manifest supplied via the ``manifests`` constructor parameter.

        Raises:
            SchemaVersionError: When the database is a sub-floor schema with
                a populated core table this build cannot bootstrap, is a
                sub-floor schema whose zero-row core tables already exist
                under an older shape the bootstrap script would leave
                untouched, or is stamped newer than this build's head
                version.

        """
        cursor = await self._db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current_version = int(row[0]) if row else 0

        if current_version < _CORE_SCHEMA_BOOTSTRAP_FLOOR:
            # A sub-floor database is only safe to bootstrap when it is truly
            # empty. The bootstrap script is built entirely of
            # ``CREATE ... IF NOT EXISTS`` statements ending in an
            # unconditional ``PRAGMA user_version`` stamp — run against a file
            # that already carries a core table (written by something older
            # than this migration ladder's public history), it would leave
            # whatever that file actually contains silently mislabelled as a
            # fresh, fully-migrated schema.
            if await self._has_any_core_table():
                raise SchemaVersionError.populated_sub_floor(
                    current_version, _CORE_SCHEMA_BOOTSTRAP_FLOOR
                )
            # Fresh bootstrap: ``schema_core.sql`` already carries the head DDL
            # and stamps ``user_version`` itself, so no incremental step runs
            # for a brand-new database.
            schema_sql = (
                resources.files("engrava.infrastructure.sqlite")
                .joinpath("schema_core.sql")
                .read_text(encoding="utf-8")
            )
            if not await self._existing_core_tables_match_bootstrap_shape(schema_sql):
                # A core table already exists here (with zero rows, or
                # ``_has_any_core_table`` above would already have refused),
                # but not under this script's own shape -- it predates this
                # build. Refusing *before* running anything means there is no
                # stamp for a crash between the write and an undo to strand:
                # nothing was written, so there is nothing to undo.
                raise SchemaVersionError.stale_shape_sub_floor(
                    current_version, _CORE_SCHEMA_BOOTSTRAP_FLOOR
                )
            await self._db.executescript(schema_sql)
        elif current_version > CORE_SCHEMA_HEAD_VERSION:
            # The migration registry has no step targeting a version this high
            # — the incremental loop would simply skip every entry and leave
            # ``user_version`` untouched, opening a file written by a newer
            # engrava as though it were current and understood.
            raise SchemaVersionError.newer_than_head(current_version, CORE_SCHEMA_HEAD_VERSION)
        else:
            await self._run_pending_core_migrations(current_version)

        # Ensure referential integrity is enforced for the lifetime of this
        # connection. SQLite ships with foreign_keys=OFF by default, so any
        # caller that constructs SqliteEngravaCore directly (rather than via
        # the from_config factory) would otherwise miss FK enforcement even
        # though the schema declares the constraints (core-12).
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._probe_fts()

        # Apply extension schema migrations.
        if self._manifests:
            from engrava.infrastructure.sqlite.extension_migrations import (  # noqa: PLC0415
                ExtensionMigrationRunner,
            )

            runner = ExtensionMigrationRunner()
            for _manifest in self._manifests:
                await runner.apply_pending(_manifest, self._db)

    def _core_migration_steps(
        self,
    ) -> tuple[tuple[int, Callable[[], Awaitable[None]]], ...]:
        """Return the ordered core-schema migration registry.

        This is the **single source of truth** for the core upgrade order:
        each entry maps the ``user_version`` a step reaches to the coroutine
        that applies it. :meth:`_run_pending_core_migrations` walks it from the
        database's current version, so adding a future migration is one new
        entry here plus its ``_migrate_core_*`` method — never an edit to every
        historical path.

        The registry is rebuilt per call from bound method references so a
        monkeypatched migration (used by the schema-drift regression test)
        resolves to the patched attribute.

        Returns:
            Entries ordered by ascending target version, contiguous from the
            first post-bootstrap step (``v2 -> v3`` rebuilds the FTS index) up
            to the head version (``v20 -> v21``).

        """
        return (
            (3, self._rebuild_fts_index),
            (4, self._migrate_core_v3_to_v4),
            (5, self._migrate_core_v4_to_v5),
            (6, self._migrate_core_v5_to_v6),
            (7, self._migrate_core_v6_to_v7),
            (8, self._migrate_core_v7_to_v8),
            (9, self._migrate_core_v8_to_v9),
            (10, self._migrate_core_v9_to_v10),
            (11, self._migrate_core_v10_to_v11),
            (12, self._migrate_core_v11_to_v12),
            (13, self._migrate_core_v12_to_v13),
            (14, self._migrate_core_v13_to_v14),
            (15, self._migrate_core_v14_to_v15),
            (16, self._migrate_core_v15_to_v16),
            (17, self._migrate_core_v16_to_v17),
            (18, self._migrate_core_v17_to_v18),
            (19, self._migrate_core_v18_to_v19),
            (20, self._migrate_core_v19_to_v20),
            (21, self._migrate_core_v20_to_v21),
        )

    async def _run_pending_core_migrations(self, current_version: int) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Apply every pending core migration in registry order.

        Walks the ordered registry from :meth:`_core_migration_steps` and runs
        only the steps whose target version exceeds ``current_version``. Each
        step is idempotent and **verifies its own postcondition**, raising
        :class:`CoreMigrationError` (or the underlying SQLite error) before it
        returns if the migrated structure is absent.

        **Atomicity.** Every step here except
        :data:`_FK_RECREATE_TARGET_VERSION` runs its DDL/DML, the
        ``PRAGMA user_version`` stamp, and the ``COMMIT`` itself all inside
        one ``try``: SQLite's schema-modifying statements (``ALTER TABLE``,
        ``CREATE TABLE|INDEX|TRIGGER``, ``DROP``, the FTS5 rebuild) are fully
        transactional, so a step whose DDL/DML or version stamp raises
        partway rolls back to exactly the database state before it began —
        nothing it already executed survives the failure — and the version
        is never stamped over a partial change. A failing ``COMMIT`` itself
        (for example SQLite reporting the database locked because a
        concurrent reader still holds a transaction) is covered the same
        way: the rollback that follows returns the connection to that same
        pre-step state, with no transaction left open on it and the durable
        ``user_version`` unchanged, so a retry does not read back an
        uncommitted stamp on this connection and mistake the step for
        already applied. That guarantee holds when the compensating
        rollback itself succeeds. If it instead raises an ``Exception``,
        that failure is logged and the original migration failure is still
        what propagates (see the ``except`` block below) — but if the
        rollback, or the logging call right after it, raises something that
        is not an ``Exception`` — a cancellation delivered while this
        coroutine is suspended on that ``await`` is the case that matters
        here, and a logging handler that itself raises from ``emit()``
        behaves the same way — the inner ``except Exception`` does not
        catch it: that exception propagates in the original failure's
        place, with the original kept only as its ``__context__``. Either
        way, once the
        compensating rollback has failed, this connection's transaction
        state is no longer known and it should not be reused. When the
        rollback does succeed, a retry
        re-enters the same step from that unchanged state, and because
        every step is *also* independently idempotent (checks its own
        postcondition, guards its own ``ADD COLUMN`` / uses
        ``IF NOT EXISTS``), the retry converges whether or not this outer
        transaction is what rolled the previous attempt back.

        The one exception is the foreign-key recreate step
        (``_FK_RECREATE_TARGET_VERSION`` / :meth:`_migrate_core_v11_to_v12`),
        which is **not** wrapped here: it must toggle
        ``PRAGMA foreign_keys`` off and back on around its own table swap, and
        that pragma is a documented no-op while any transaction — including
        the one this method would open — is already active. When the swap is
        actually needed (some child table still lacks its FK),
        :meth:`_migrate_core_v11_to_v12` calls
        :meth:`_recreate_child_tables_with_fk_atomically`, whose own leading
        ``commit()`` closes any transaction left open by an earlier step
        before toggling the pragma — from that point this step runs in
        autocommit and provides its own atomicity for the swap via an
        internal ``SAVEPOINT`` (see that method's docstring for exactly what
        it can leave behind and why re-running is still safe). When no
        *existing* table needs the swap (each one either already carries
        its FK or was never there to begin with — a partial database is
        "nothing to migrate" here, same as the comment above this method's
        ``migration_needed`` computation says), that call — and its leading
        ``commit()`` — never happens: the trailing
        ``CREATE INDEX`` and the ``PRAGMA user_version`` stamp this step
        still issues then run in whatever transaction state the connection
        was already in when this step started, not necessarily autocommit.

        **Why the per-step transaction opens with ``BEGIN IMMEDIATE``.** Most
        steps' first statement is a postcondition-style presence check
        (:meth:`_table_exists`, :meth:`_column_exists` via
        :meth:`_add_column_if_absent`, :meth:`_index_exists`) before their
        first DDL/DML — a read, inside the transaction, ahead of the write.
        Under a deferred ``BEGIN`` that read takes a WAL snapshot the later
        write then has to upgrade; if another connection holds the write
        lock, SQLite refuses the upgrade with ``SQLITE_BUSY`` and never
        invokes the busy handler, so a contending migration would fail at
        once instead of waiting out ``PRAGMA busy_timeout`` like every other
        write unit in this module. ``BEGIN IMMEDIATE`` takes the write lock
        up front, through the busy handler, before any step gets to read.

        Args:
            current_version: The database's current ``user_version``. It is at
                or above the bootstrap floor; the fresh-bootstrap path is
                handled by :meth:`ensure_schema` before this method is called.

        """
        for target_version, migrate in self._core_migration_steps():
            if target_version <= current_version:
                continue
            if target_version == _FK_RECREATE_TARGET_VERSION:
                await migrate()
                await self._db.execute(f"PRAGMA user_version = {target_version}")
                await self._db.commit()
                continue
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                await migrate()
                await self._db.execute(f"PRAGMA user_version = {target_version}")
                await self._db.commit()
            except BaseException:
                # ``BaseException``, not ``Exception``: a cancellation delivered
                # mid-step must also roll back rather than leave a half-applied
                # schema change committed by a later, unrelated statement.
                # ``commit()`` is inside this ``try`` (not after it) for the
                # same reason: a ``COMMIT`` that itself fails -- e.g. SQLite
                # reporting the database locked because a concurrent reader
                # still holds a transaction -- otherwise left the transaction
                # open on this connection, which then reads back the
                # *uncommitted* new ``user_version`` and can make a retried
                # ``ensure_schema`` on this same connection believe the step
                # already applied when nothing durable happened at all.
                try:
                    await self._db.rollback()
                except Exception:
                    # A rollback failure here does not change what needs to
                    # propagate -- the migration failure above is still the
                    # real error. Logged rather than raised, the same
                    # convention ``_close_quietly`` uses for a cleanup
                    # failure while another exception is already in flight
                    # (see its docstring), applied here to a rollback
                    # instead of a close.
                    logger.warning("Error rolling back failed core migration step", exc_info=True)
                raise

    async def _has_any_core_table(self) -> bool:
        """Return whether the database already carries user data in a core table.

        Used by :meth:`ensure_schema` to tell a genuinely empty file (safe to
        bootstrap) apart from a **populated** sub-floor database (refused —
        see :class:`SchemaVersionError`). Checked as row presence, not mere
        table existence: the bootstrap script (``schema_core.sql``) is pure
        DDL, so a bootstrap interrupted partway through (a mid-script DDL
        failure) can leave empty table / index / trigger definitions behind
        without ever writing a row, and a retry has to be able to re-run the
        same idempotent ``CREATE ... IF NOT EXISTS`` script against that
        partial, data-free state rather than being told it looks like a real
        legacy database (see the failure-injection coverage in
        ``TestBootstrapAtomicity``). A database that was actually used, by
        contrast, carries at least one row somewhere — that is what
        "populated" means here.

        The ``sqlite_master.name`` lookup is matched ``COLLATE NOCASE``:
        SQLite itself resolves table identifiers case-insensitively (a
        ``CREATE TABLE IF NOT EXISTS thought`` matches an existing ``THOUGHT``
        just as ``SELECT ... FROM thought`` reads from it), so a plain
        case-sensitive string comparison against the stored name could miss a
        core table written under a different case and report an empty file
        that is not one.

        Returns:
            ``True`` if any table in :data:`_CORE_TABLE_NAMES` exists **and**
            holds at least one row.

        """
        for table in _CORE_TABLE_NAMES:
            cursor = await self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? COLLATE NOCASE",
                (table,),
            )
            if await cursor.fetchone() is None:
                continue
            # `table` is drawn only from the fixed _CORE_TABLE_NAMES tuple
            # above, never from caller input, so this interpolation cannot
            # carry anything the allow-list did not already name.
            row_cursor = await self._db.execute(f"SELECT 1 FROM {table} LIMIT 1")  # noqa: S608
            if await row_cursor.fetchone() is not None:
                return True
        return False

    async def _existing_core_tables_match_bootstrap_shape(self, schema_sql: str) -> bool:
        """Return whether every already-existing core table matches ``schema_sql``.

        Checked **before** ``schema_sql`` is run, not after: the script is
        pure ``CREATE ... IF NOT EXISTS``, so it can only skip an existing
        table under an older shape, never fix it — while still stamping
        ``user_version`` current at its own trailing statement regardless.
        Checking first means a mismatch is refused with nothing yet written:
        no stamp has landed, so there is nothing to undo and no window
        between a stamp and an undo for a crash to split apart.

        A table that does not exist yet is not a mismatch — it is exactly
        what the script is about to create, whether this is a genuinely
        fresh file or a retry of *this build's own* bootstrap interrupted
        partway through. ``CREATE TABLE`` is atomic, so any table an earlier,
        interrupted run of this exact script already finished already carries
        the full head shape and matches; tables it had not reached yet are
        simply absent here and the now-completing script creates them fresh.

        A genuinely fresh file — no core table exists at all — skips the
        comparison entirely rather than opening a disposable reference
        connection it would never need: that keeps the common case (every
        brand-new database) exactly as cheap, and as free of incidental
        connection churn, as before this check existed.

        When at least one core table does already exist, this bootstraps a
        disposable, provably-fresh reference database from ``schema_sql``
        itself — rather than naming one column the current head happens to
        add last, which would stop being a witness the moment a *future*
        migration adds another — and compares each such table's column set
        against it. The comparison is self-updating: whatever the script
        declares next release, the reference picks it up automatically, with
        no separate constant for a future column to fall out of sync with.

        **What this does not check.** The comparison is column *names* only —
        it says nothing about types, defaults, constraints, collations,
        column order, or generated/hidden columns (some of which
        ``PRAGMA table_info`` does not even surface). A hand-built table
        carrying the right column names under a different type or a hostile
        constraint would pass. No build this project has ever shipped
        produces such a table, which is why this is a documented limit of
        the check rather than something it is fixed to catch — it
        establishes equal column names across the core tables, nothing more.

        **What this does not serialize.** This is a single connection's own
        pre-check, not a cross-process lock. Two different engrava builds
        bootstrapping the same fresh file at the same time can still
        interleave: each can pass this check while no core table yet exists,
        then each runs its own ``schema_core.sql`` and stamps its own head
        version, and whichever runs last wins. Detecting or preventing that
        race is outside what this method — or any check confined to a single
        connection — can do.

        Args:
            schema_sql: The ``schema_core.sql`` text about to be run, if this
                check passes.

        Returns:
            ``True`` if every core table already present on ``self._db`` has
            exactly the column set a fresh bootstrap of ``schema_sql`` gives
            it. Vacuously ``True`` when no core table exists yet.

        """
        existing_tables = [table for table in _CORE_TABLE_NAMES if await self._table_exists(table)]
        if not existing_tables:
            return True
        async with aiosqlite.connect(":memory:") as reference:
            await reference.executescript(schema_sql)
            for table in existing_tables:
                actual = await self._table_column_names(self._db, table)
                expected = await self._table_column_names(reference, table)
                if actual != expected:
                    return False
        return True

    @staticmethod
    async def _table_column_names(conn: aiosqlite.Connection, table: str) -> frozenset[str]:
        """Return the set of column names ``table`` carries on ``conn``.

        Reads each ``PRAGMA table_info`` row **by position**
        (``cid, name, type, notnull, dflt_value, pk`` — ``name`` is index 1),
        never by key. ``conn`` here is not always a connection this class
        configured itself: the disposable reference database this is also
        called against is plain ``aiosqlite.connect(...)`` with no
        ``row_factory`` set, and a caller could hand this a connection with
        any row factory at all. Indexing by key only works when the row
        factory happens to be ``aiosqlite.Row`` (or another mapping); reading
        by position works whether the row comes back as that or as a plain
        tuple, so this makes no assumption about the connection it is given.

        Args:
            conn: Open connection to inspect.
            table: The table name. Always drawn from the fixed
                :data:`_CORE_TABLE_NAMES` tuple, never caller input, so the
                f-string interpolation below cannot carry anything the
                allow-list did not already name.

        Returns:
            The table's column names, order-independent. Empty if the table
            does not exist on ``conn``.

        """
        cursor = await conn.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        return frozenset(row[1] for row in rows)

    async def _probe_fts(self) -> None:
        """Detect whether the ``thought_fts`` FTS5 table exists.

        Sets ``_fts_available`` to ``True`` when the virtual table is
        present, ``False`` otherwise.  Called once during schema bootstrap
        to avoid repeated introspection on every ``search_fts()`` call.
        """
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='thought_fts'"
        )
        self._fts_available = (await cursor.fetchone()) is not None
        self._fts_probed = True

    async def _rebuild_fts_index(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Rebuild the FTS5 index with the hyphen-aware tokenizer.

        This upgrade path is used for existing core schema version 2
        databases whose original FTS5 configuration treated ``-`` as an
        operator, breaking prefix searches for identifier-like terms.
        """
        await self._db.execute("DROP TRIGGER IF EXISTS thought_fts_insert")
        await self._db.execute("DROP TRIGGER IF EXISTS thought_fts_delete")
        await self._db.execute("DROP TRIGGER IF EXISTS thought_fts_update")
        await self._db.execute("DROP TABLE IF EXISTS thought_fts")
        await self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS thought_fts USING fts5("
            "  essence, content,"
            "  tokenize = \"unicode61 tokenchars '-_'\","
            "  content='thought', content_rowid='rowid'"
            ")"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS thought_fts_insert "
            "AFTER INSERT ON thought BEGIN "
            "  INSERT INTO thought_fts(rowid, essence, content) "
            "  VALUES (new.rowid, new.essence, new.content); "
            "END"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS thought_fts_delete "
            "AFTER DELETE ON thought BEGIN "
            "  INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
            "  VALUES ('delete', old.rowid, old.essence, old.content); "
            "END"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS thought_fts_update "
            "AFTER UPDATE OF essence, content ON thought BEGIN "
            "  INSERT INTO thought_fts(thought_fts, rowid, essence, content) "
            "  VALUES ('delete', old.rowid, old.essence, old.content); "
            "  INSERT INTO thought_fts(rowid, essence, content) "
            "  VALUES (new.rowid, new.essence, new.content); "
            "END"
        )
        await self._db.execute(
            "INSERT OR IGNORE INTO thought_fts(rowid, essence, content) "
            "SELECT rowid, essence, content FROM thought"
        )
        # Postcondition: the rebuilt FTS table AND its three sync triggers must
        # all exist before the loop bumps the version, so a v3 database always
        # carries a fully wired, queryable index (not a table without triggers).
        await self._require_table(3, "thought_fts")
        for trigger in (
            "thought_fts_insert",
            "thought_fts_delete",
            "thought_fts_update",
        ):
            await self._require_trigger(3, trigger)

    async def _migrate_core_v3_to_v4(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Add access tracking and datetime timestamp columns (core-4).

        Idempotent — safe to run on a database that already has the columns.
        Backfills ``created_at`` and ``updated_at`` with the current UTC
        time for existing rows that lack timestamps.
        """
        new_columns = (
            ("access_count", "INTEGER NOT NULL DEFAULT 0"),
            ("last_accessed_at", "TEXT"),
            ("created_at", "TEXT"),
            ("updated_at", "TEXT"),
        )
        for column, column_type in new_columns:
            await self._add_column_if_absent("thought", column, column_type)

        now = datetime.datetime.now(datetime.UTC).isoformat()
        await self._db.execute(
            "UPDATE thought SET created_at = ?, updated_at = ? WHERE created_at IS NULL",
            (now, now),
        )
        await self._db.execute(
            "UPDATE thought SET updated_at = ? WHERE updated_at IS NULL",
            (now,),
        )
        # Postcondition: all four columns must exist before the loop bumps the
        # version. ``access_count`` / ``last_accessed_at`` are not read by the
        # backfill above, so a silently-swallowed ``ALTER`` would otherwise be
        # recorded as migrated.
        for column in ("access_count", "last_accessed_at", "created_at", "updated_at"):
            await self._require_column(4, "thought", column)

    async def _migrate_core_v4_to_v5(self) -> None:
        """Add the ``_metadata`` key/value table (core-5).

        Idempotent — uses ``CREATE TABLE IF NOT EXISTS``.
        """
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS _metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        await self._require_table(5, "_metadata")

    async def _migrate_core_v5_to_v6(self) -> None:
        """Add the ``journal_entry`` table and indexes (core-6).

        Idempotent — uses ``CREATE TABLE IF NOT EXISTS`` and
        ``CREATE INDEX IF NOT EXISTS``.
        """
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS journal_entry ("
            "  entry_id         TEXT PRIMARY KEY,"
            "  sequence_number  INTEGER NOT NULL UNIQUE,"
            "  mutation_type    TEXT NOT NULL,"
            "  target_id        TEXT,"
            "  delta            TEXT NOT NULL,"
            "  parent_hash      TEXT,"
            "  entry_hash       TEXT NOT NULL,"
            "  created_at       TEXT NOT NULL"
            ")"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_target "
            "ON journal_entry(target_id, sequence_number)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entry(mutation_type)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_seq ON journal_entry(sequence_number)"
        )
        await self._require_table(6, "journal_entry")
        for index_name in ("idx_journal_target", "idx_journal_type", "idx_journal_seq"):
            await self._require_index(6, index_name)

    async def _migrate_core_v6_to_v7(self) -> None:
        """Add ``expires_at`` column and partial index (core-7).

        Idempotent — the ``ADD COLUMN`` is guarded so a database already
        carrying the column is left unchanged, and any non-duplicate DDL error
        propagates rather than being swallowed.
        """
        await self._add_column_if_absent("thought", "expires_at", "TEXT")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_expires "
            "ON thought(expires_at) WHERE expires_at IS NOT NULL"
        )
        await self._require_column(7, "thought", "expires_at")
        await self._require_index(7, "idx_thought_expires")

    async def _migrate_core_v7_to_v8(self) -> None:
        """Add composite edge index for candidate expansion queries (core-8).

        Supports ``_expand_via_consolidated_from`` which queries::

            SELECT ... FROM edge
            WHERE edge_type = 'CONSOLIDATED_FROM'
            AND from_thought_id IN (...)

        Without this index SQLite performs a full table scan on the edge
        table, which breaks the p95 < 50 ms latency requirement at scale.
        The same index also accelerates the existing ``_load_graph_signal``
        COUNT query backing the giant-cluster guard.

        Idempotent — uses ``CREATE INDEX IF NOT EXISTS``. The ``edge`` table may
        be absent in a partial bootstrap (it is created lazily / by the fresh
        DDL), so the create is guarded by ``_table_exists`` exactly as the later
        edge-touching migrations guard theirs — a thought-only database has no
        edge index to build, and the fresh DDL already carries it. Any *other*
        DDL failure propagates rather than being swallowed, so an isolated
        index-creation error can no longer be recorded as a completed migration.
        A postcondition assertion confirms the index is present (when the
        ``edge`` table exists) before the loop bumps the version.
        """
        if await self._table_exists("edge"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id)"
            )
            # Postcondition: the index must exist before the loop bumps the
            # version, so a v8 database that carries the ``edge`` table is never
            # marked current without the candidate-expansion index.
            await self._require_index(8, "idx_edge_type_from")

    async def _migrate_core_v8_to_v9(self) -> None:
        """Add extension_schema_versions table (core-9).

        Tracks which SQL migration files have been applied for each
        installed extension.  The table is created with
        ``CREATE TABLE IF NOT EXISTS``, so this migration is fully
        idempotent.
        """
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS extension_schema_versions (
                extension_name    TEXT PRIMARY KEY,
                version           INTEGER NOT NULL DEFAULT 0,
                applied_at        REAL NOT NULL,
                migration_file    TEXT,
                extension_version TEXT
            )
            """
        )
        await self._require_table(9, "extension_schema_versions")

    async def _migrate_core_v9_to_v10(self) -> None:
        """Add ``content_hash`` column + index to ``thought`` table (core-10).

        Adds a nullable ``content_hash TEXT`` column and the
        ``idx_thought_content_hash`` index used by opt-in ingest
        deduplication (``create_thought(..., deduplicate=True)``).

        Idempotent: ``ALTER TABLE ADD COLUMN`` is wrapped in
        duplicate-column tolerance and ``CREATE INDEX`` uses
        ``IF NOT EXISTS``, so re-running the migration after a partial
        crash converges on the fully-applied state.

        Existing rows are left with ``content_hash = NULL`` until the
        bundled backfill utility populates them; new ``create_thought``
        calls compute the hash at insert time regardless of the
        ``deduplicate`` flag.
        """
        await self._add_column_if_absent("thought", "content_hash", "TEXT")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash)"
        )
        await self._require_column(10, "thought", "content_hash")
        await self._require_index(10, "idx_thought_content_hash")

    async def _migrate_core_v10_to_v11(self) -> None:
        """Add ``metadata_json`` column to ``thought`` table (core-11).

        Adds a NOT NULL ``metadata_json TEXT`` column with default
        ``'{}'`` to support structured metadata (role, lang,
        content_type, session_id, ...).  Existing rows get the
        empty-dict default — no data migration required.

        Idempotent: the guarded ``ADD COLUMN`` tolerates only a duplicate
        column (matches ``_migrate_core_v9_to_v10`` precedent), so re-running
        the migration after a partial crash converges on the fully-applied
        state.
        """
        await self._add_column_if_absent("thought", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        await self._require_column(11, "thought", "metadata_json")

    async def _recreate_child_tables_with_fk_atomically(
        self,
        *,
        edge_needs: bool,
        embedding_needs: bool,
        action_needs: bool,
    ) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Recreate the child tables that still lack their FK, atomically.

        :meth:`_run_pending_core_migrations` wraps every other core-migration
        step in one outer transaction but deliberately excludes this one's
        target version (see ``_FK_RECREATE_TARGET_VERSION``): that pragma
        toggle below is a no-op inside a transaction, so an outer ``BEGIN``
        would silently defeat it. This method is where the atomicity for the
        excluded step actually lives instead.

        All three recreations run inside ONE explicit SAVEPOINT so the whole
        swap is atomic. Under sqlite3 legacy isolation (aiosqlite's default) the
        driver does not implicitly begin a transaction for DDL, so a bare
        ``rollback()`` may NOT undo a mid-recreate ``DROP`` — the child table
        would be permanently gone and a later ``ensure_schema`` retry (which
        recomputes ``*_exists`` as ``False``) could then stamp v12 over the
        missing table. An explicit savepoint enrols every statement, so
        ``ROLLBACK TO`` undoes the DROP.

        The contract on failure is **all-or-nothing, not "always the original"**:
        the swap is either fully rolled back (the original table survives, ready
        for a clean retry) or fully applied — never half-applied. Both outcomes
        are self-consistent. If ``RELEASE recreate_fk`` reaches SQLite but
        cancellation lands before this coroutine resumes, the reconstruction is
        already committed and the later cleanup ``rollback()`` is a no-op; that
        state is the migration's intended end state anyway, so a retry sees the
        foreign keys present and converges on v12.

        Foreign-key enforcement is per-connection, and ``PRAGMA foreign_keys``
        is silently ignored inside an open transaction, so a leaked "off" state
        would make the rest of the session accept orphans and skip
        ``ON DELETE CASCADE`` without ever raising. Both the disable and the
        savepoint therefore live inside the outer ``try``, whose ``except``
        (failure or cancellation) and ``else`` (success) branches both close
        any still-open transaction and re-enable enforcement — a plain
        ``finally`` cannot draw that distinction, and the two branches need
        different treatment of a cleanup failure (see below). That covers
        every transaction-control statement except the leading ``commit()``,
        which precedes the ``try`` and runs while enforcement is still on.

        On the ``except`` path, a failure rolling back or restoring the
        pragma must not replace the exception (or cancellation) already
        propagating from the body above — an ordinary failure in either
        step is logged underneath instead, via ``_run_cleanup_step_quietly``,
        and both steps are attempted even if the first one fails; a
        cancellation of the cleanup itself is the one thing that still
        outranks the original exception's plain ``raise``. On the ``else``
        path there is no original exception to protect, so an ordinary
        cleanup failure is not merely logged there — it is the error, and
        propagates. The rollback there is usually a no-op with nothing open
        to undo, but a no-op rollback can still raise: not only an ordinary
        ``Exception`` (e.g. against a connection that has already gone
        away), but a ``CancelledError`` or other ``BaseException`` raised
        directly by the rollback call, which ``_run_cleanup_step_quietly``'s
        own shield-then-redraw does not catch and would otherwise let escape
        before the pragma restore is even attempted — so that path wraps the
        pragma restore in a plain ``finally`` rather than relying on the
        helper alone. See :meth:`_restore_after_successful_recreate` for
        exactly which of the two exceptions is the one actually raised in
        each case, and which is instead only chained onto it as
        ``__cause__``.

        This is a strong best effort, not an absolute postcondition: if the
        cleanup ``rollback()`` itself fails while leaving a transaction active,
        the following pragma is ignored, and the restore is itself an ``await``
        that a further cancellation could interrupt. A caller that swallows a
        migration failure and keeps using the connection cannot assume
        enforcement is back on — re-asserting ``PRAGMA foreign_keys=ON`` is not
        sufficient either, since it is ignored while a transaction is still
        open. The only reliable boundary after an uncertain cleanup is to
        discard (close) the connection.

        Args:
            edge_needs: Whether ``edge`` must be recreated with its FK.
            embedding_needs: Whether ``embedding`` must be recreated with its FK.
            action_needs: Whether ``action`` must be recreated with its FK.

        """
        # Close any implicit transaction opened by prior migration steps so
        # PRAGMA foreign_keys=OFF takes effect (it is ignored inside a txn).
        await self._db.commit()
        try:
            # Inside the try: a failure (or cancellation) delivered on this await
            # may still leave the pragma applied on the connection, so the
            # ``except``/``else`` below must already cover it.
            await self._db.execute("PRAGMA foreign_keys=OFF")
            await self._db.execute("SAVEPOINT recreate_fk")
            try:
                await self._purge_orphan_children()
                if edge_needs:
                    await self._recreate_edge_with_fk()
                if embedding_needs:
                    await self._recreate_embedding_with_fk()
                if action_needs:
                    await self._recreate_action_with_fk()
            # ``except BaseException`` (not ``Exception``) so a cancellation
            # during recreate also rolls the swap back before it propagates.
            except BaseException:
                # Undo every statement back to the savepoint so a half-completed
                # swap never leaves a dropped table (or a leftover ``*_new``
                # table) for the next attempt to inherit, then release it.
                #
                # If one of these control statements itself fails, it propagates
                # and the outer ``except`` below discards the whole transaction —
                # the safe outcome, and deliberately NOT a further ``RELEASE``:
                # releasing the OUTERMOST savepoint COMMITS, so a release after a
                # failed ``ROLLBACK TO`` would durably commit the half-swap.
                await self._db.execute("ROLLBACK TO recreate_fk")
                await self._db.execute("RELEASE recreate_fk")
                raise
            else:
                # Success: release the savepoint (commits the swap) and land the
                # commit outside any transaction so re-enabling FK below — a
                # no-op inside a transaction — actually takes effect.
                await self._db.execute("RELEASE recreate_fk")
                await self._db.commit()
        except BaseException as exc:
            # The recreate above (or its own unwind) raised, or was cancelled —
            # that is what the caller needs to see. Rolling back and restoring
            # the pragma are real cleanup, but a failure in either one must not
            # replace it -- logged underneath instead, via
            # ``_run_cleanup_step_quietly``, exactly as ``_close_quietly``'s
            # docstring states for a connection close. Both steps are attempted
            # even if the first one fails, since leaving the pragma unrestored
            # would silently disable FK enforcement for the whole session --
            # matching the previous nested-``finally`` shape's guarantee.
            # ``PRAGMA foreign_keys`` is ignored inside a transaction, so the
            # rollback runs first, exactly as it did before. A cancellation
            # delivered to *this* cleanup, though, is a fresh control-flow
            # signal, not an ordinary failure: it wins over whatever was
            # propagating before it -- the same precedence ``_close_quietly``
            # gives a cancellation over the error it is cleaning up after --
            # but ``from exc`` still records what that error was, rather than
            # discarding it, since it is exactly the information a caller who
            # catches the cancellation would want to see.
            rollback_cancelled = await _run_cleanup_step_quietly(
                self._db.rollback,
                "rolling back the migration transaction",
            )
            pragma_cancelled = await _run_cleanup_step_quietly(
                lambda: self._db.execute("PRAGMA foreign_keys=ON"),
                "restoring PRAGMA foreign_keys",
            )
            if pragma_cancelled is not None:
                raise pragma_cancelled from exc
            if rollback_cancelled is not None:
                raise rollback_cancelled from exc
            raise
        else:
            # Clean path: the rollback is usually a no-op (nothing is left
            # open to undo), but it is still a real await against a real
            # connection and can itself fail -- e.g. against a connection
            # that has already gone away. There is no original exception to
            # protect here, so unlike the ``except`` branch above, a failure
            # in either cleanup step must reach the caller rather than being
            # logged and swallowed -- see
            # :meth:`_restore_after_successful_recreate` for the shape.
            await self._restore_after_successful_recreate()

    async def _restore_after_successful_recreate(self) -> None:
        """Roll back the (normally no-op) transaction and restore the pragma.

        The cleanup half of the ``else`` (success) branch of
        :meth:`_recreate_child_tables_with_fk_atomically`, split out to keep
        that method's branch count down. Issued UNCONDITIONALLY: the caller
        does not gate this on ``in_transaction``, since that flag is read
        straight off the connection and can be stale relative to a statement
        still queued in aiosqlite's worker, which could skip a rollback that
        is in fact still needed.

        Unlike that method's ``except`` branch, there is no original
        exception to protect here, so a failure in either step below is not
        merely logged -- it IS an error and must reach the caller, exactly
        as ``_close_quietly``'s docstring states for a close on the success
        path. But losing the pragma restore because the rollback ahead of it
        raised would silently leave FK enforcement off for the rest of the
        connection's life, so the pragma restore is wrapped in a plain
        ``finally`` under the rollback -- the same guarantee the deleted
        nested-``finally`` shape this whole cleanup replaced used to give --
        rather than relying only on ``_run_cleanup_step_quietly``'s own
        return-a-``CancelledError``-instead-of-raising convention: that
        helper's own shield-then-redraw re-awaits an already-finished task
        through an ``except Exception``, so a rollback that raises a
        ``BaseException`` directly -- a synchronous ``CancelledError`` not
        delivered by the outer task's own ``cancel()``, a ``SystemExit``, a
        ``KeyboardInterrupt``, or a custom ``BaseException`` -- escapes that
        call uncaught instead of coming back as its documented return value.
        The ``except BaseException`` below exists to catch exactly that
        escape (as well as anything else that could in principle escape the
        helper) so the ``finally`` beneath it still runs either way.

        Both steps are attempted no matter what either one raises -- that
        much holds unconditionally, guaranteed by the ``finally`` above, not
        merely by ``_run_cleanup_step_quietly``'s own convention. What
        happens to the two exceptions differs by case, and not every case
        gets the same treatment:

        * Neither step fails: nothing is raised.
        * Exactly one step fails: that exception is re-raised bare, with no
          ``from`` -- genuinely unchanged, including its own
          ``__cause__``/``__context__``, whatever they already were.
        * Both fail: the pragma restore's exception is the one actually
          raised (its loss is the silent, lasting one -- FK enforcement left
          off) -- via ``raise ... from`` the rollback's, so the rollback's is
          not discarded, but it is then visible only as that exception's
          ``__cause__`` in the traceback, not as the exception a caller's
          ``except SomeType:`` would itself catch.

        A cancellation is not special-cased against an ordinary exception in
        the "both fail" case above: whichever of the two steps failed
        *last* (the pragma restore, always, since it runs second) is the one
        raised, cancellation or not. That differs from the ``except``
        branch's own cleanup, where an ordinary cleanup-step failure is
        always only logged and never propagates at all, and the one thing
        that can still outrank the original body exception's plain ``raise``
        is a cancellation of the cleanup itself. There is no original body
        exception here for a cancellation to need to outrank, and an
        ordinary cleanup failure is not merely logged here -- it is the
        error, so it propagates rather than being swallowed.
        """
        rollback_error: BaseException | None = None
        pragma_error: BaseException | None = None

        def _capture_rollback_exc(exc: Exception) -> None:
            nonlocal rollback_error
            rollback_error = exc

        def _capture_pragma_exc(exc: Exception) -> None:
            nonlocal pragma_error
            pragma_error = exc

        try:
            rollback_cancelled = await _run_cleanup_step_quietly(
                self._db.rollback,
                "rolling back the (normally no-op) migration transaction",
                log_failure=_capture_rollback_exc,
            )
            if rollback_cancelled is not None:
                rollback_error = rollback_cancelled
        except BaseException as exc:  # noqa: BLE001 -- see docstring above
            rollback_error = exc
        finally:
            # Attempted no matter what happened above -- an ordinary
            # exception captured into ``rollback_error``, a cancellation
            # returned as one, or a raw ``BaseException`` caught just above.
            try:
                pragma_cancelled = await _run_cleanup_step_quietly(
                    lambda: self._db.execute("PRAGMA foreign_keys=ON"),
                    "restoring PRAGMA foreign_keys",
                    log_failure=_capture_pragma_exc,
                )
                if pragma_cancelled is not None:
                    pragma_error = pragma_cancelled
            except BaseException as exc:  # noqa: BLE001 -- see docstring above
                pragma_error = exc

        # Both steps have now been attempted, whatever either one raised.
        # Exactly which of the two propagates, and how, differs by case --
        # see the docstring above for the three cases this covers and the
        # two it does not.
        if pragma_error is not None:
            if rollback_error is not None:
                raise pragma_error from rollback_error
            raise pragma_error
        if rollback_error is not None:
            raise rollback_error

    async def _migrate_core_v11_to_v12(self) -> None:
        """Add referential integrity (FK + ON DELETE CASCADE) to child tables.

        SQLite does not support ``ALTER TABLE ADD CONSTRAINT`` so the
        FK clauses on ``edge``, ``embedding`` and ``action`` are
        introduced via the recreate-table pattern: build a new table
        with the FK declaration, copy rows over, drop the old table,
        rename the new one, and rebuild any indexes that the schema
        declared on it.

        ``PRAGMA foreign_keys=OFF`` is a documented no-op while a
        transaction is open. This helper therefore commits any pending
        work *before* toggling the pragma, runs the recreate steps,
        commits the recreations, and only then re-enables enforcement.
        The leading commit is defensive: the migration loop commits after
        every step, but a caller that reaches this migration with an open
        implicit transaction (from an earlier write on the same connection)
        would otherwise leave FK enforcement on during the swap, and the
        recreated tables would fail their first ``INSERT … SELECT *`` if any
        unpurged orphan remained.

        Pre-existing orphan rows in any of the three child tables are
        purged before the constraint is enabled. Orphans are already
        invalid against the documented invariant (``ON DELETE CASCADE``
        on the parent) and the documented contract on
        ``delete_thought`` — keeping them would block enabling the FK.
        The purge is unconditional on the FK column (no
        ``owner_type='THOUGHT'`` gate on the embedding side) because
        the FK does not look at ``owner_type``; a stray lowercase or
        non-THOUGHT owner that does not resolve to a thought would
        otherwise survive the purge and break the recreate.

        Idempotent: the helper detects whether each table already
        carries the FK declaration via ``PRAGMA foreign_key_list`` and
        skips the recreation when the constraint is already present.
        Re-running ``ensure_schema`` on a freshly migrated database is
        therefore a no-op for this step. A partial earlier run that
        upgraded some tables and not others resumes per-table on the
        next call.
        """
        edge_exists = await self._table_exists("edge")
        embedding_exists = await self._table_exists("embedding")
        action_exists = await self._table_exists("action")
        edge_done = (
            edge_exists
            and await self._fk_present("edge", "from_thought_id")
            and await self._fk_present("edge", "to_thought_id")
        )
        embedding_done = embedding_exists and await self._fk_present("embedding", "owner_id")
        action_done = action_exists and await self._fk_present("action", "source_thought_id")
        # Tables absent from a partial bootstrap (only `thought` present) are
        # treated as "nothing to migrate" — fresh installs receive the FK
        # directly from ``schema_core.sql``.
        migration_needed = not (
            (edge_done or not edge_exists)
            and (embedding_done or not embedding_exists)
            and (action_done or not action_exists)
        )

        if migration_needed:
            await self._recreate_child_tables_with_fk_atomically(
                edge_needs=edge_exists and not edge_done,
                embedding_needs=embedding_exists and not embedding_done,
                action_needs=action_exists and not action_done,
            )

        # The edge recreation drops its indexes. Ensure the required v8 index
        # also when the FK was already present, so a partial legacy schema is
        # repaired rather than failing the same postcondition on every retry.
        if edge_exists:
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id)"
            )

        # Postcondition: every child table that EXISTED AT ENTRY must still
        # exist and now carry its FK, and the edge recreate re-created
        # ``idx_edge_type_from`` (dropped with the old table). Keying off the
        # entry-time ``*_exists`` flags — not a fresh existence probe — means a
        # table present at entry but vanished mid-migration fails ``_require_table``
        # here rather than being silently skipped and stamped v12 without
        # referential integrity. A table absent at entry (partial bootstrap with
        # only ``thought``) has nothing to migrate and is not required.
        for existed_at_entry, table, column in (
            (edge_exists, "edge", "from_thought_id"),
            (edge_exists, "edge", "to_thought_id"),
            (embedding_exists, "embedding", "owner_id"),
            (action_exists, "action", "source_thought_id"),
        ):
            if not existed_at_entry:
                continue
            await self._require_table(12, table)
            await self._require_fk(12, table, column)
        if edge_exists:
            await self._require_index(12, "idx_edge_type_from")

        # Retry model: the per-step registry resumes a failed upgrade at the
        # failed step (an improvement over the old bump-once-at-the-end ladder,
        # which re-ran the whole tail). The table swaps are atomic within the
        # SAVEPOINT above and rolled back to it on failure, so a mid-recreate
        # drop never persists; a legacy state that already carries the exact FK
        # is skipped and the required edge index is recreated independently, so
        # a resumed upgrade is convergent.

    async def _migrate_core_v12_to_v13(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Add nullable valid-time columns + indexes to thought and edge (core-13).

        Introduces a second time axis ("valid time") alongside the
        existing transaction time. ``created_at`` records *when a fact was
        stored*; ``valid_from`` / ``valid_until`` record *when a fact is
        true in the world*. Both new columns are nullable ISO-8601 TEXT.

        Backfill is intentionally asymmetric:

        * ``thought.valid_from`` is seeded from ``created_at`` for rows
          that have a transaction timestamp, giving existing thoughts a
          sensible default lower bound. ``valid_until`` is left ``NULL``
          (open upper bound). Rows whose ``created_at`` is ``NULL``
          (pre-timestamp legacy rows) keep ``valid_from = NULL`` — no
          date is fabricated.
        * ``edge`` rows are **not** backfilled. The edge table has no
          ``created_at`` column; its only temporal field is
          ``created_cycle``, which is an internal cognitive-cycle counter,
          not a calendar timestamp. Synthesising a valid-time date from a
          cycle number would invent information, so edges keep both
          valid-time fields ``NULL`` (an open lower bound).

        Idempotent: each ``ALTER TABLE ... ADD COLUMN`` is guarded so a re-run
        after the column already exists is a no-op and any non-duplicate DDL
        error propagates, and every index uses ``CREATE INDEX IF NOT EXISTS``.
        Re-running the migration leaves the schema unchanged.
        """
        # Only touch tables that exist. A partial bootstrap may carry just
        # ``thought`` (the ``edge`` table is created lazily); operating on an
        # absent ``edge`` would raise ``no such table``. ``thought`` is always
        # present at this point. This mirrors the table-existence guards used
        # by the earlier edge-touching migrations and ``_purge_orphan_children``.
        tables = ["thought"]
        if await self._table_exists("edge"):
            tables.append("edge")

        for table in tables:
            for column in ("valid_from", "valid_until"):
                await self._add_column_if_absent(table, column, "TEXT")

        # Asymmetric backfill — thought only, sourced from transaction time.
        # Rows with NULL created_at (legacy, pre-timestamp) keep NULL
        # valid_from; no calendar date is fabricated for them.
        await self._db.execute(
            "UPDATE thought SET valid_from = created_at "
            "WHERE created_at IS NOT NULL AND valid_from IS NULL"
        )
        # Edge has no created_at; created_cycle is internal cognitive time,
        # not calendar time, so edges are deliberately left with NULL
        # valid_from / valid_until (an open lower bound).

        for table in tables:
            await self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_valid_from ON {table}(valid_from)"
            )
            await self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_valid_until "
                f"ON {table}(valid_until) WHERE valid_until IS NOT NULL"
            )
            await self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_valid_range "
                f"ON {table}(valid_from, valid_until)"
            )
        # Postcondition: every touched table (``thought`` always, ``edge`` when
        # present) must carry both valid-time columns and their three indexes
        # before the loop bumps the version.
        for table in tables:
            for column in ("valid_from", "valid_until"):
                await self._require_column(13, table, column)
            for suffix in ("valid_from", "valid_until", "valid_range"):
                await self._require_index(13, f"idx_{table}_{suffix}")

    async def _migrate_core_v13_to_v14(self) -> None:
        """Add hot-path indexes for the core read queries (core-14).

        Purely additive: creates four indexes that back the equality
        filters and the sort column hit on every common read, without
        touching any row or column. The targets were chosen from the
        actual ``WHERE`` / ``ORDER BY`` clauses in this module:

        * ``idx_edge_to_thought`` on ``edge(to_thought_id)`` — ``get_edges``
          (the inbound and both-direction modes) and the
          reflection-consolidation scan filter the edge table on
          ``to_thought_id``.
        * ``idx_embedding_owner`` on ``embedding(owner_id)`` —
          ``get_embedding`` looks an embedding up by its owner thought;
          without this index the lookup is a full table scan, and it runs
          inside three dreaming loops.
        * ``idx_thought_updated_cycle`` on ``thought(updated_cycle)`` —
          ``list_thoughts`` orders by ``updated_cycle`` on every call.
        * ``idx_thought_type`` on ``thought(thought_type)`` —
          ``thought_type`` equality is used by the reflection-id scan on
          every search and by ``list_thoughts`` filtering.

        Idempotent: every statement uses ``CREATE INDEX IF NOT EXISTS``, so
        re-running the migration leaves the schema unchanged. The ``edge``
        and ``embedding`` tables may be absent in a partial bootstrap (they
        are created lazily), so each is guarded by ``_table_exists`` exactly
        as ``_migrate_core_v12_to_v13`` guards ``edge``. The ``thought``
        table is always present, but each indexed column is additionally
        guarded by ``_column_exists`` so a minimal or hand-rolled legacy
        schema that has not yet grown a column (for example a very old
        database whose ``thought`` table predates ``updated_cycle``) skips
        that single index instead of raising ``no such column``.
        """
        # ``thought`` is always present, but a minimal legacy schema may lack
        # an indexed column; index only the columns that exist.
        if await self._column_exists("thought", "updated_cycle"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_thought_updated_cycle ON thought(updated_cycle)"
            )
        if await self._column_exists("thought", "thought_type"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_thought_type ON thought(thought_type)"
            )
        # ``edge`` / ``embedding`` may be absent in a partial bootstrap;
        # creating an index on a missing table would raise ``no such table``.
        if await self._table_exists("edge"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_to_thought ON edge(to_thought_id)"
            )
        if await self._table_exists("embedding"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_embedding_owner ON embedding(owner_id)"
            )
        # Postcondition: every hot-path index whose guarded target is present
        # must exist before the loop bumps the version.
        expected_indexes = (
            (await self._column_exists("thought", "updated_cycle"), "idx_thought_updated_cycle"),
            (await self._column_exists("thought", "thought_type"), "idx_thought_type"),
            (await self._table_exists("edge"), "idx_edge_to_thought"),
            (await self._table_exists("embedding"), "idx_embedding_owner"),
        )
        for guarded_present, index_name in expected_indexes:
            if guarded_present:
                await self._require_index(14, index_name)

    async def _migrate_core_v14_to_v15(self) -> None:
        """Add the ``(edge_type, to_thought_id)`` composite edge index (core-15).

        Purely additive. The inbound edge-type lookups
        (``WHERE to_thought_id = ? AND edge_type = ?`` — the
        ``CONSOLIDATED_FROM`` source-resolution scan) can only use the
        single-column ``idx_edge_to_thought`` from core-14 to seek
        ``to_thought_id`` and must then test ``edge_type`` as a residual per
        matched row. This composite index mirrors ``idx_edge_type_from`` on the
        destination side so both predicates are satisfied by one index seek —
        ``EXPLAIN QUERY PLAN`` reports ``idx_edge_type_to (edge_type=? AND
        to_thought_id=?)`` rather than a residual filter. No row, column, or
        query changes; results are unaffected.

        Idempotent — uses ``CREATE INDEX IF NOT EXISTS``. The ``edge`` table may
        be absent in a partial bootstrap (it is created lazily), so the create
        is guarded by ``_table_exists`` exactly as ``_migrate_core_v13_to_v14``
        guards its ``edge`` index.
        """
        if await self._table_exists("edge"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_type_to ON edge(edge_type, to_thought_id)"
            )
            await self._require_index(15, "idx_edge_type_to")

    async def _migrate_core_v15_to_v16(self) -> None:
        """Add the action-outcome aggregate column and its seek index (core-16).

        Purely additive. Two independent changes back the action-outcome
        feedback loop:

        * ``thought.action_outcome_score`` (nullable ``REAL``) — the
          denormalised mean outcome value over a thought's terminal linked
          actions, or ``NULL`` when it has none. Added via
          ``ALTER TABLE ... ADD COLUMN``; an ``OperationalError`` naming a
          duplicate column is swallowed so a database already carrying the
          column (a partial or re-run migration) is left unchanged.
        * ``idx_action_source_thought`` on ``action(source_thought_id)`` — the
          recompute resolves a thought's actions with
          ``WHERE source_thought_id = ?``; without this index that lookup is a
          full scan of the ``action`` table, and it runs on every
          outcome-affecting write. ``EXPLAIN QUERY PLAN`` then reports
          ``SEARCH action USING INDEX idx_action_source_thought
          (source_thought_id=?)`` rather than a full scan.

        Idempotent. The column add is guarded against the duplicate-column
        error exactly as ``_migrate_core_v9_to_v10`` guards its own
        ``ADD COLUMN``; the index create uses ``CREATE INDEX IF NOT EXISTS``.
        The ``action`` table may be absent in a partial bootstrap (it is
        created by the fresh DDL), so the index create is guarded by
        ``_table_exists`` exactly as ``_migrate_core_v14_to_v15`` guards its
        ``edge`` index.
        """
        await self._add_column_if_absent("thought", "action_outcome_score", "REAL")
        if await self._table_exists("action"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_source_thought ON action(source_thought_id)"
            )
            await self._require_index(16, "idx_action_source_thought")
        await self._require_column(16, "thought", "action_outcome_score")

    async def _migrate_core_v16_to_v17(self) -> None:
        """Add the opt-in provenance column and its identity indexes (core-17).

        Purely additive. Backs write-time provenance capture:

        * ``thought.provenance`` (nullable ``TEXT``) — a JSON document holding
          the opt-in :class:`~engrava.domain.models.provenance.ProvenanceContext`
          sub-model, or ``NULL`` when a thought carries no provenance. Added via
          the guarded ``ADD COLUMN`` helper, which tolerates only a duplicate
          column so a database already carrying it (a partial or re-run
          migration) is left unchanged.
        * ``idx_thought_prov_session`` / ``idx_thought_prov_actor`` — JSON
          expression indexes on the two identity fields
          (``json_extract(provenance, '$.session_id')`` and ``'$.actor_id'``).
          These resolve the DEC on first-class session / actor lookup: a
          ``WHERE json_extract(provenance,'$.session_id')=?`` query then reports
          ``SEARCH thought USING INDEX idx_thought_prov_session (<expr>=?)``
          rather than a full scan. The descriptive provenance fields
          (``retrieval_query`` / ``instruction_context`` /
          ``retrieval_context_ids``) are queryable through the same
          ``json_extract`` filter machinery but are deliberately not indexed.

        Provenance is captured and made queryable only — it feeds no ranking,
        dreaming / consolidation, or edge-creation path, and is an untrusted
        hint that the engine grants zero authority (see
        :class:`~engrava.domain.models.provenance.ProvenanceContext`).

        Idempotent. The column add is guarded against the duplicate-column error
        exactly as ``_migrate_core_v15_to_v16`` guards its own ``ADD COLUMN``;
        the index creates use ``CREATE INDEX IF NOT EXISTS``. The ``thought``
        table is always present by this point (it is the first table created by
        the fresh DDL and by every earlier migration path), so the expression
        indexes need no table-existence guard — the column guard above ensures
        the indexed expression resolves.
        """
        await self._add_column_if_absent("thought", "provenance", "TEXT")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_prov_session "
            "ON thought(json_extract(provenance, '$.session_id'))"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_prov_actor "
            "ON thought(json_extract(provenance, '$.actor_id'))"
        )
        await self._require_column(17, "thought", "provenance")
        for index_name in ("idx_thought_prov_session", "idx_thought_prov_actor"):
            await self._require_index(17, index_name)

    async def _migrate_core_v17_to_v18(self) -> None:
        """Add the Memory Hygiene forgetting-loop columns (core-18).

        Purely additive. Two nullable/defaulted columns back the deterministic
        forgetting loop:

        * ``thought.pinned`` (``INTEGER NOT NULL DEFAULT 0``) — the durable
          never-forget marker. A pinned thought is never auto-archived or
          auto-GC'd by the hygiene loop. The ``DEFAULT 0`` means every existing
          row reads back as ``pinned=False`` unchanged.
        * ``thought.archived_at_cycle`` (nullable ``INTEGER``) — the cycle at
          which the hygiene loop archived a thought, or ``NULL`` when it was not
          archived by hygiene (a restore clears it back to ``NULL``). It backs
          the GC restore window; a thought archived by any other path
          (TTL / manual) keeps ``NULL`` and is never reaped by hygiene GC.

        Both adds are guarded against the duplicate-column error exactly as
        ``_migrate_core_v16_to_v17`` guards its own ``ADD COLUMN``, so a database
        already carrying a column (a partial or re-run migration) is left
        unchanged. No index is added — the hygiene loop scans the
        already-indexed ``lifecycle_status`` / ``updated_cycle`` candidate set
        and filters ``archived_at_cycle`` in Python, so no new expression index
        is warranted. While hygiene stays disabled these columns are never read
        on any existing path, so this is "no behavioural change while disabled",
        not literally byte-identical bytes on disk.
        """
        await self._add_column_if_absent("thought", "pinned", "INTEGER NOT NULL DEFAULT 0")
        await self._add_column_if_absent("thought", "archived_at_cycle", "INTEGER")
        # Postcondition: both hygiene columns must exist before the loop bumps
        # the version.
        for column in ("pinned", "archived_at_cycle"):
            await self._require_column(18, "thought", column)

    async def _migrate_core_v18_to_v19(self) -> None:
        """Add the generic ``metadata_json`` column to the ``edge`` table (core-19).

        Purely additive. Mirrors the thought-side ``metadata_json`` column
        (core-11): a NOT NULL ``TEXT`` column defaulting to ``'{}'`` so every
        existing edge reads back an empty metadata mapping with no backfill. The
        column gives edges the same generic structured-attribute carrier that
        thoughts already have; keys carry no reserved meaning, and no secondary
        index is added (parity with thought metadata — filtering is a full
        ``json_extract`` scan). Appended last, matching the fresh ``edge`` DDL
        column order (``ALTER ... ADD COLUMN`` can only append).

        The add is guarded against the duplicate-column error exactly as
        ``_migrate_core_v17_to_v18`` guards its own ``ADD COLUMN``, so a database
        already carrying the column (a partial or re-run migration) is left
        unchanged — this makes the "column added but ``user_version`` not yet
        bumped" state re-entrant.

        The ``ALTER`` is followed by a postcondition assertion that the column
        is present before the function returns. The migration loop bumps
        ``user_version`` only *after* this function returns, so for any database
        that **carries the** ``edge`` **table** the version can never be trusted
        while the column is absent: a migrated ``edge`` table at
        ``user_version = 19`` therefore has ``edge.metadata_json`` by
        construction, closing the "version bumped without the column" hole an
        interrupt could otherwise open.

        The one shape the assertion cannot speak to is a partial bootstrap with
        **no** ``edge`` table at all (a thought-only database): the early return
        below lets the loop stamp v19 without touching a table that does not
        exist — exactly as every earlier edge migration guards its edge work
        with ``_table_exists`` and still advances the version. This is not a
        hole, because the ``edge`` table is only ever created from nothing by the
        base DDL (``schema_core.sql``), which at v19 already carries
        ``metadata_json``; any ``edge`` table that later comes into existence is
        therefore self-healing. No database can reach a state with an ``edge``
        table that lacks ``metadata_json``.
        """
        # The ``edge`` table may be absent in a partial bootstrap (it is created
        # lazily / by the fresh DDL), so guard exactly as the earlier
        # edge-touching migrations (``_migrate_core_v12_to_v13`` /
        # ``_migrate_core_v13_to_v14``) do: a thought-only database has no edge
        # column to add, and the fresh DDL already carries the column.
        if not await self._table_exists("edge"):
            return
        await self._add_column_if_absent("edge", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        # Postcondition: the column must exist before the loop bumps the
        # version, closing the "version bumped without the column" hole.
        await self._require_column(19, "edge", "metadata_json")

    async def _migrate_core_v19_to_v20(self) -> None:
        """Add the wall-clock archival-instant column ``thought.archived_at`` (core-20).

        Purely additive. A single nullable ``TEXT`` column holding the
        UTC-normalised ISO-8601 instant at which the Memory Hygiene loop archived
        a thought, or ``NULL`` when it was not archived by hygiene (a restore
        clears it back to ``NULL``, exactly like ``archived_at_cycle``). It backs
        the wall-clock restore window: the irreversible GC stage may reap a
        hygiene-archived thought only once **both** the cycle window
        (``archived_at_cycle``) and this real-time window have elapsed, so a
        fast-cycling store can no longer permanently delete a just-archived
        thought before any real-time chance to restore it.

        The add is guarded against the duplicate-column error exactly as
        ``_migrate_core_v17_to_v18`` guards its own ``ADD COLUMN``, so a database
        already carrying the column (a partial or re-run migration) is left
        unchanged. No index is added — the GC stage scans the already-narrow
        hygiene-archived candidate set and filters ``archived_at`` with a
        lexicographic ISO-8601 comparison, so no expression index is warranted.
        A row archived by hygiene **before** this column existed reads back
        ``archived_at IS NULL``: it has no real-time stamp and is therefore never
        GC-eligible while the wall-clock window is active — the irreversible stage
        fails closed rather than delete a row it cannot time.

        The ``thought`` table is always present by this point (it is the first
        table created by the fresh DDL and by every earlier migration path), so
        the ``ALTER`` needs no table-existence guard. A postcondition assertion
        confirms the column is present before the migration loop bumps
        ``user_version``, closing the "version bumped without the column" hole an
        interrupt could otherwise open.
        """
        await self._add_column_if_absent("thought", "archived_at", "TEXT")
        # Postcondition: the column must exist before the loop bumps the
        # version, closing the "version bumped without the column" hole.
        await self._require_column(20, "thought", "archived_at")

    async def _migrate_core_v20_to_v21(self) -> None:
        """Add the row-version guard column and canonicalise stored timestamps (core-21).

        Two parts, run in the one transaction the migration loop opens for the
        step. First, ``thought``, ``edge`` and ``action`` each gain
        ``revision INTEGER NOT NULL DEFAULT 0``. Not ``embedding`` — it is a
        carrier owned by its thought, not an independently updatable entity.
        Second, the stored values in the timestamp columns the shared validator
        covers that are not already in the canonical shape are read once, and
        each one that can be read as an instant is rewritten into the canonical
        UTC form that validator now returns — see
        :meth:`_normalise_stored_timestamps`. That second part
        changes row values only, never a table's shape, so it does not need a
        schema version of its own.

        ``NOT NULL DEFAULT 0`` makes a separate backfill step unnecessary: every
        pre-existing row reads back ``revision = 0`` the instant the column
        exists, identically to a freshly bootstrapped row that has never been
        written. There is no NULL state for a write-time guard to special-case.

        This column is the enforcement primitive the guarded ``UPDATE``
        statements on ``update_thought`` / ``restore_thought`` / ``update_edge``
        / ``update_action`` compare and increment atomically (``revision =
        revision + 1 WHERE id = ? AND revision = ?``) — it replaces
        ``updated_cycle`` as the guard column, since ``updated_cycle`` is a
        cognitive-recency signal nothing in this store advances on its own and
        is not safe to overload as a write counter.

        The add is guarded against the duplicate-column error exactly as every
        earlier ``ADD COLUMN`` rung guards its own, so a database already
        carrying the column (a partial or re-run migration) is left unchanged.
        A postcondition assertion per table confirms each added column is
        present before the migration loop bumps ``user_version``, closing the
        "version bumped without the column" hole an interrupt could otherwise
        open — the same pattern every prior rung in this ladder uses.

        ``thought`` is always present by this point (the first table created
        by the fresh DDL and by every earlier migration path), so its
        ``ALTER`` needs no table-existence guard, exactly as
        ``_migrate_core_v19_to_v20`` reasons about the same table. ``edge``
        and ``action`` may each be absent in a partial bootstrap (a
        thought-only database) — guarded by ``_table_exists`` exactly as
        ``_migrate_core_v18_to_v19`` guards its own ``edge`` work and
        ``_migrate_core_v15_to_v16`` guards its own ``action`` work. Either
        table is only ever created from nothing by the base DDL, which at v21
        already carries ``revision``, so a table that later comes into
        existence is self-healing — no database can reach a state with an
        ``edge`` or ``action`` table that lacks ``revision``.
        """
        await self._add_column_if_absent("thought", "revision", "INTEGER NOT NULL DEFAULT 0")
        # Postcondition: the column must exist before the loop bumps the
        # version, closing the "version bumped without the column" hole.
        await self._require_column(21, "thought", "revision")

        if await self._table_exists("edge"):
            await self._add_column_if_absent("edge", "revision", "INTEGER NOT NULL DEFAULT 0")
            await self._require_column(21, "edge", "revision")

        if await self._table_exists("action"):
            await self._add_column_if_absent("action", "revision", "INTEGER NOT NULL DEFAULT 0")
            await self._require_column(21, "action", "revision")

        await self._normalise_stored_timestamps()

    async def _normalise_stored_timestamps(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Rewrite stored timestamps without the canonical shape into the canonical UTC form.

        Before core-21 the shared validator returned a naive timestamp exactly
        as the caller wrote it, so a v20 database can hold values such as
        ``2026-01-02 03:04:05``, ``20260102T030405`` or ``2026-W01-5`` in a
        column the store compares as TEXT — against
        ``datetime.now(UTC).isoformat()`` for expiry, against MindQL literals for
        valid time. Such a value sorts in the wrong place: a space separator
        makes a still-live row read as expired, basic format or a week date
        makes an expired one read as live. This rewrites each of them that can
        be read as an instant into the form the validator returns now (a naive
        value is read as UTC), so it orders by its instant.

        Covers the columns in :data:`_CANONICAL_TIMESTAMP_COLUMNS`. Only rows
        whose value does not already have the canonical shape are read back
        (selected in SQL with ``GLOB``), a page at a time, so a large store that
        is already canonical is scanned but not rewritten. A value that already
        has the canonical shape is not re-read, even if it names an impossible
        date. The rewrite changes no instant and is not an edit: ``revision`` is
        not bumped, ``updated_at`` is not re-stamped (its own value is only put
        into canonical form, like every covered column), and no journal entry is
        written — the journal keeps its own copies of what was written, so its
        hash chain still verifies. Of the values read back, one that cannot be
        read as an ISO-8601 instant is left untouched; the count of such values
        is logged, never the values themselves.

        An ``edge`` table absent in a partial bootstrap (a thought-only
        database) is skipped, as the ``revision`` part of this step skips it.

        Raises:
            CoreMigrationError: If a rewritten column still holds a value
                without the canonical shape that is not one of the values left
                untouched.

        """
        left_untouched: dict[str, int] = {}
        for table, columns in _CANONICAL_TIMESTAMP_COLUMNS:
            if not await self._table_exists(table):
                continue
            for column in columns:
                untouched = await self._normalise_timestamp_column(table, column)
                if untouched:
                    left_untouched[f"{table}.{column}"] = untouched
        if left_untouched:
            logger.warning(
                "Upgrading the core schema to version 21 left %d stored timestamp "
                "value(s) unchanged because they cannot be read as an ISO-8601 "
                "instant (%s)",
                sum(left_untouched.values()),
                ", ".join(f"{name}: {count}" for name, count in left_untouched.items()),
            )

    async def _normalise_timestamp_column(self, table: str, column: str) -> int:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Rewrite one column's values that lack the canonical shape.

        See :meth:`_normalise_stored_timestamps` for what is rewritten and why.

        Pages through the rows whose value lacks the canonical shape by
        ``rowid`` (keyset pagination), so a value left untouched is never read
        twice and a rewritten one never comes back: after its rewrite it matches
        the canonical ``GLOB`` and is no longer selected. The ``SELECT`` of each
        page is fully read before that page's ``UPDATE`` runs, so no statement
        reads a table another one on this connection is changing. Then checks
        its own postcondition.

        Args:
            table: A table from :data:`_CANONICAL_TIMESTAMP_COLUMNS`.
            column: One of that table's columns there. Both are drawn from that
                fixed constant, never caller input, so the f-string
                interpolation below cannot carry anything it did not name.

        Returns:
            How many of the values read back (those without the canonical
            shape) were left untouched because they cannot be read as an
            ISO-8601 instant. A value with the canonical shape is never read, so
            it is never counted.

        Raises:
            CoreMigrationError: If, after the rewrite, the number of values
                without the canonical shape is not exactly the number left
                untouched.

        """
        not_canonical = (
            f"{column} IS NOT NULL AND NOT ({column} GLOB ? "
            f"OR ({column} GLOB ? AND {column} NOT GLOB ?))"
        )
        globs = (_CANONICAL_WHOLE_SECOND_GLOB, _CANONICAL_FRACTION_GLOB, _ZERO_FRACTION_GLOB)
        first_page = (
            f"SELECT rowid, {column} FROM {table} "  # noqa: S608 - fixed identifiers, see Args
            f"WHERE {not_canonical} ORDER BY rowid LIMIT ?"
        )
        next_page = (
            f"SELECT rowid, {column} FROM {table} "  # noqa: S608 - fixed identifiers, see Args
            f"WHERE rowid > ? AND {not_canonical} ORDER BY rowid LIMIT ?"
        )
        rewrite = f"UPDATE {table} SET {column} = ? WHERE rowid = ?"  # noqa: S608 - fixed identifiers

        untouched = 0
        cursor = await self._db.execute(first_page, (*globs, _TIMESTAMP_NORMALISATION_BATCH_SIZE))
        while True:
            rows = list(await cursor.fetchall())
            rewrites: list[tuple[str, int]] = []
            for row in rows:
                canonical = canonical_timestamp_or_none(row[1])
                if canonical is None:
                    untouched += 1
                else:
                    rewrites.append((canonical, int(row[0])))
            if rewrites:
                await self._db.executemany(rewrite, rewrites)
            if len(rows) < _TIMESTAMP_NORMALISATION_BATCH_SIZE:
                break
            cursor = await self._db.execute(
                next_page, (int(rows[-1][0]), *globs, _TIMESTAMP_NORMALISATION_BATCH_SIZE)
            )

        # Postcondition: only the values left untouched may still lack the
        # canonical shape.
        cursor = await self._db.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {not_canonical}",  # noqa: S608 - fixed identifiers
            globs,
        )
        remaining = await cursor.fetchone()
        if remaining is None or int(remaining[0]) != untouched:
            raise CoreMigrationError(
                21,
                f"{table}.{column} still holds values without the canonical shape "
                "after the rewrite",
            )
        return untouched

    async def _fk_present(self, table: str, column: str) -> bool:
        """Return whether ``column`` has the required thought-cascade foreign key.

        Args:
            table: Child table to inspect.
            column: Child column that must reference ``thought.thought_id``.

        Returns:
            ``True`` only for the core contract's exact reference and
            ``ON DELETE CASCADE`` action.

        """
        cursor = await self._db.execute(f"PRAGMA foreign_key_list({table})")
        rows = await cursor.fetchall()
        return any(
            row["from"] == column
            and row["table"] == "thought"
            and row["to"] == "thought_id"
            and str(row["on_delete"]).upper() == "CASCADE"
            for row in rows
        )

    async def _require_fk(self, target_version: int, table: str, column: str) -> None:
        """Raise :class:`CoreMigrationError` when a required FK is absent.

        Args:
            target_version: The core schema version the calling step targets.
            table: Child table expected to carry the foreign key.
            column: Child column expected to reference ``thought.thought_id``
                with ``ON DELETE CASCADE``.

        Raises:
            CoreMigrationError: If the exact foreign-key contract is absent.

        """
        if not await self._fk_present(table, column):
            raise CoreMigrationError(
                target_version,
                f"{table}.{column} missing thought FK with ON DELETE CASCADE",
            )

    async def _table_exists(self, table: str) -> bool:
        """Return ``True`` when ``table`` is registered in ``sqlite_master``.

        Matched ``COLLATE NOCASE``, the same as :meth:`_has_any_core_table`:
        SQLite resolves table identifiers case-insensitively, so a
        case-sensitive comparison here could report a table absent when it
        exists under a different case and every real DDL/DML statement
        against it already resolves fine.
        """
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? COLLATE NOCASE",
            (table,),
        )
        return await cursor.fetchone() is not None

    async def _index_exists(self, index: str) -> bool:
        """Return ``True`` when ``index`` is registered in ``sqlite_master``.

        Presence (registration by name) is the right granularity for a
        migration postcondition: the migrations own these index names and
        create each from a fixed ``CREATE INDEX`` statement, so a registered
        name means our DDL took effect. The exact index *definition* (columns,
        predicate, expression) is pinned separately by the fresh-vs-migrated
        schema-parity test suite, which compares normalised index DDL.

        Args:
            index: The index name to look for.

        Returns:
            ``True`` if an ``index``-typed entry with that name exists.

        """
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        )
        return await cursor.fetchone() is not None

    async def _require_index(self, target_version: int, index: str) -> None:
        """Raise :class:`CoreMigrationError` when ``index`` is not registered.

        A postcondition helper (see :meth:`_require_table` for the shared
        existence-based-gate contract these ``_require_*`` helpers implement)
        confirming a step's index was created before the migration loop stamps
        the version.

        Args:
            target_version: The core schema version the calling step targets.
            index: The index name that must be present.

        Raises:
            CoreMigrationError: If no index with that name is registered.

        """
        if not await self._index_exists(index):
            raise CoreMigrationError(target_version, f"{index} missing after create")

    async def _trigger_exists(self, trigger: str) -> bool:
        """Return ``True`` when ``trigger`` is registered in ``sqlite_master``.

        Args:
            trigger: The trigger name to look for.

        Returns:
            ``True`` if a ``trigger``-typed entry with that name exists.

        """
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (trigger,),
        )
        return await cursor.fetchone() is not None

    async def _require_trigger(self, target_version: int, trigger: str) -> None:
        """Raise :class:`CoreMigrationError` when ``trigger`` is not registered.

        A postcondition helper (see :meth:`_require_table` for the shared
        existence-based-gate contract) confirming a step's trigger was created
        before the migration loop stamps the version.

        Args:
            target_version: The core schema version the calling step targets.
            trigger: The trigger name that must be present.

        Raises:
            CoreMigrationError: If no trigger with that name is registered.

        """
        if not await self._trigger_exists(trigger):
            raise CoreMigrationError(target_version, f"{trigger} missing after create")

    async def _column_exists(self, table: str, column: str) -> bool:
        """Return ``True`` when ``table`` has a column named ``column``.

        Args:
            table: The table to inspect. Must already exist.
            column: The column name to look for.

        Returns:
            ``True`` if the column is present in ``PRAGMA table_info``.

        """
        cursor = await self._db.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        return any(row["name"] == column for row in rows)

    async def _require_table(self, target_version: int, table: str) -> None:
        """Raise :class:`CoreMigrationError` when ``table`` is not registered.

        Shared contract of the ``_require_*`` postcondition helpers
        (:meth:`_require_column`, :meth:`_require_index`, :meth:`_require_trigger`
        and this one): the runtime gate is **existence-based**. It detects a
        *failed or absent* migration — its object was not created — and raises so
        the migration loop leaves ``user_version`` retryable rather than stamping
        it over a partial schema. It deliberately does **not** re-verify object
        *definitions* (FTS tokenizer or index/trigger bodies): those are pinned
        by the fresh-vs-migrated schema-parity test suite in dev/CI. Foreign keys
        are the exception: :meth:`_require_fk` verifies the referenced table,
        referenced column, and delete action because all three are available via
        ``PRAGMA foreign_key_list``. A name collision with a pre-existing object
        of the wrong definition is a corruption/tampering case that the parity
        suite catches, outside this gate's scope (the migrations own these names
        and create each from a fixed statement).

        Args:
            target_version: The core schema version the calling step targets.
            table: The table name that must be present.

        Raises:
            CoreMigrationError: If no table with that name is registered.

        """
        if not await self._table_exists(table):
            raise CoreMigrationError(target_version, f"{table} table missing after create")

    async def _require_column(self, target_version: int, table: str, column: str) -> None:
        """Raise :class:`CoreMigrationError` when ``table.column`` is absent.

        A postcondition helper (see :meth:`_require_table` for the shared
        existence-based-gate contract) confirming a step's column was added
        before the migration loop stamps the version.

        Args:
            target_version: The core schema version the calling step targets.
            table: The table expected to carry the column.
            column: The column name that must be present.

        Raises:
            CoreMigrationError: If the column is not present on the table.

        """
        if not await self._column_exists(table, column):
            raise CoreMigrationError(target_version, f"{table}.{column} missing after migration")

    async def _add_column_if_absent(self, table: str, column: str, column_type: str) -> None:
        """Idempotently add ``column`` to ``table`` when it is not already present.

        Guards on current presence and tolerates only the duplicate-column error
        (the idempotent re-run signal); any other DDL failure propagates so a
        genuine failure is never silently recorded as a completed migration.

        Args:
            table: The table to alter. Must already exist.
            column: The column name to add.
            column_type: The SQLite column type and constraints, e.g. ``"TEXT"``
                or ``"INTEGER NOT NULL DEFAULT 0"``.

        """
        if await self._column_exists(table, column):
            return
        try:
            await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
        except aiosqlite.OperationalError as exc:
            # Tolerate ONLY the exact "duplicate column name: <column>" signal for
            # THIS column (a concurrent add after the presence guard passed). The
            # SQLite message ends with the column name, so an exact (whole-message)
            # match avoids a prefix collision — a duplicate-column error for a
            # different column (e.g. ``<column>_extra``) is NOT a substring match
            # and propagates, so a genuine DDL failure is never silently recorded
            # as a completed migration.
            expected = f"duplicate column name: {column}".lower()
            if str(exc).strip().lower() != expected:
                raise

    async def _purge_orphan_children(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Delete orphan rows whose parent thought no longer exists.

        Runs before FK enablement so the constraint can be added
        without rejecting existing-but-invalid data. Counts of removed
        rows are intentionally not surfaced: orphans are bugs by
        definition and the migration runs once in pre-publish. Tables
        absent from a partial bootstrap are skipped.
        """
        if await self._table_exists("edge"):
            await self._db.execute(
                "DELETE FROM edge "
                "WHERE from_thought_id NOT IN (SELECT thought_id FROM thought) "
                "   OR to_thought_id   NOT IN (SELECT thought_id FROM thought)",
            )
        if await self._table_exists("embedding"):
            # Unconditional on owner_id: the FK does not branch on
            # owner_type, so any owner_id that fails to resolve to a
            # thought is an orphan against the new constraint
            # regardless of the (case-variant) owner_type value.
            await self._db.execute(
                "DELETE FROM embedding WHERE owner_id NOT IN (SELECT thought_id FROM thought)",
            )
        if await self._table_exists("action"):
            await self._db.execute(
                "DELETE FROM action "
                "WHERE source_thought_id NOT IN (SELECT thought_id FROM thought)",
            )

    async def _recreate_edge_with_fk(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Recreate ``edge`` with FK + CASCADE on both endpoints."""
        await self._db.execute(
            "CREATE TABLE edge_new ("
            "  edge_id           TEXT PRIMARY KEY,"
            "  from_thought_id   TEXT NOT NULL,"
            "  to_thought_id     TEXT NOT NULL,"
            "  edge_type         TEXT NOT NULL,"
            "  weight            REAL NOT NULL DEFAULT 0.5,"
            "  created_cycle     INTEGER NOT NULL DEFAULT 0,"
            "  source            TEXT NOT NULL DEFAULT 'EXPERIENCE',"
            "  decay_multiplier  REAL NOT NULL DEFAULT 1.0,"
            "  UNIQUE(from_thought_id, to_thought_id, edge_type),"
            "  FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE,"
            "  FOREIGN KEY (to_thought_id)   REFERENCES thought(thought_id) ON DELETE CASCADE"
            ")",
        )
        await self._db.execute(
            "INSERT INTO edge_new SELECT "
            "  edge_id, from_thought_id, to_thought_id, edge_type, weight, "
            "  created_cycle, source, decay_multiplier "
            "FROM edge",
        )
        await self._db.execute("DROP TABLE edge")
        await self._db.execute("ALTER TABLE edge_new RENAME TO edge")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id)",
        )

    async def _recreate_embedding_with_fk(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Recreate ``embedding`` with FK + CASCADE on ``owner_id``.

        The polymorphic ``owner_type`` column is preserved for forward
        compatibility, but every persisted embedding currently uses
        ``owner_type='THOUGHT'`` so the FK targets ``thought`` directly.
        """
        await self._db.execute(
            "CREATE TABLE embedding_new ("
            "  embedding_id TEXT PRIMARY KEY,"
            "  owner_type   TEXT    NOT NULL,"
            "  owner_id     TEXT    NOT NULL,"
            "  model_name   TEXT    NOT NULL,"
            "  dimension    INTEGER NOT NULL,"
            "  vector_blob  BLOB    NOT NULL,"
            "  created_at   TEXT    NOT NULL,"
            "  FOREIGN KEY (owner_id) REFERENCES thought(thought_id) ON DELETE CASCADE"
            ")",
        )
        await self._db.execute("INSERT INTO embedding_new SELECT * FROM embedding")
        await self._db.execute("DROP TABLE embedding")
        await self._db.execute("ALTER TABLE embedding_new RENAME TO embedding")

    async def _identify_orphan_endpoint(self, edge: EdgeRecord) -> tuple[str, str]:
        """Return the offending ``(column, referenced_id)`` pair after a FK reject.

        SQLite reports a generic "FOREIGN KEY constraint failed" without
        naming the column or value. The endpoints are checked in left-
        to-right order (``from_thought_id`` first); when both endpoints
        are missing the function reports the first one — sufficient
        signal for callers, and consistent with how SQLite itself
        reports a single constraint violation per row.
        """
        if not await self._thought_exists(edge.from_thought_id):
            return "from_thought_id", edge.from_thought_id
        return "to_thought_id", edge.to_thought_id

    async def _thought_exists(self, thought_id: str) -> bool:
        """Return ``True`` when a thought with ``thought_id`` is persisted."""
        cursor = await self._db.execute(
            "SELECT 1 FROM thought WHERE thought_id = ? LIMIT 1",
            (thought_id,),
        )
        return await cursor.fetchone() is not None

    async def _recreate_action_with_fk(self) -> None:
        # Write-lock classification: bucket 2 (schema bootstrap) -- see ensure_schema.
        """Recreate ``action`` with FK + CASCADE on ``source_thought_id``."""
        await self._db.execute(
            "CREATE TABLE action_new ("
            "  action_id           TEXT PRIMARY KEY,"
            "  source_thought_id   TEXT NOT NULL,"
            "  action_type         TEXT NOT NULL,"
            "  intent              TEXT NOT NULL,"
            "  status              TEXT NOT NULL DEFAULT 'PLANNED',"
            "  verification_status TEXT NOT NULL DEFAULT 'PENDING',"
            "  raw_metrics_json    TEXT,"
            "  FOREIGN KEY (source_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE"
            ")",
        )
        await self._db.execute("INSERT INTO action_new SELECT * FROM action")
        await self._db.execute("DROP TABLE action")
        await self._db.execute("ALTER TABLE action_new RENAME TO action")

    # ------------------------------------------------------------------
    # Embedding model immutability
    # ------------------------------------------------------------------

    async def _ensure_embedding_model_lock(
        self, model_name: str, dimension: int, *, commit: bool = True
    ) -> None:
        """Lock the embedding model on first ``store_embedding()``, verify on every call.

        On first call (no ``embedding_model_name`` in ``_metadata``), writes
        the model name, dimension, and — only when the active provider
        applies a non-empty ``document_prefix`` — the deterministic
        fingerprint of that prefix and the ``query_prefix`` the corpus is
        built to pair with. On this and every later call, the stored values
        are read back and compared against the arguments — there is no
        instance-level cache that lets a later call skip the comparison, so a
        store instance that is handed a different model, dimension, or
        document prefix on a later call is refused just as reliably as one
        constructed fresh with the mismatched provider.

        The document-prefix fingerprint is part of the *corpus identity*:
        changing the ``document_prefix`` changes what every stored vector
        would be, so it must trigger a re-embed via
        :class:`EmbeddingModelMismatchError`. The ``query_prefix`` does not
        change stored vectors, so it is recorded for pairing but is verified
        separately at search time (see :meth:`_ensure_query_prefix_pairs`),
        never here.

        Empty prefixes (the default) write nothing extra, so the
        ``_metadata`` shape is byte-identical to the legacy one and a
        pre-existing unprefixed store never false-trips the lock.

        ``model_name == CENTROID_MODEL_NAME`` is exempt from all of the
        above: a REFLECTION centroid is a computed mean of member vectors,
        not something a configured embedding provider produced, so its
        sentinel tag is bookkeeping rather than a corpus identity. It must
        never be compared against the locked identity, and it must never
        itself lock the corpus identity either — including the edge case
        where a centroid write happens to be the very first
        ``store_embedding()`` call ever made on a store.

        **The first-call identity write is committed here only when
        ``commit`` is true.** ``verify_embedding_model`` wants exactly that:
        it establishes identity on an empty corpus on its own account,
        independent of any write, so its call keeps the default and this
        method ends with :meth:`_maybe_commit`. That call commits
        immediately unless the caller already has a
        :meth:`suspend_auto_commit` window open on this task, in which case
        the write joins that window's transaction and only becomes durable
        when the outer window's own exit commits it. ``store_embedding``
        wants the opposite — its call passes
        ``commit=False`` and makes the identity write itself, from inside
        the same :meth:`_write_readback_savepoint` span as the base/vec0
        write it exists to gate: a rejected first vector (wrong dimension,
        an invalid owner) then unwinds the identity write along with it,
        instead of leaving a wrongly-locked identity behind for a corrected
        retry to fail against.

        Args:
            model_name: Model identifier from the current provider.
            dimension: Vector dimensionality from the current provider.
            commit: Whether the first-call identity write commits on its own
                account. ``True`` (the default, used by
                ``verify_embedding_model``) calls :meth:`_maybe_commit`,
                which commits immediately unless a caller's own
                :meth:`suspend_auto_commit` window is already open, in
                which case the commit is deferred to that window's exit.
                ``False`` (used by
                ``store_embedding``) leaves the write pending for the
                caller's own enclosing savepoint/commit to resolve, so it
                rolls back together with a failed write it was meant to
                gate rather than surviving it.

        Raises:
            EmbeddingModelMismatchError: When the configured model, its
                dimension, or its ``document_prefix`` fingerprint differs
                from the one stored in ``_metadata``.

        """
        if model_name == CENTROID_MODEL_NAME:
            # Bookkeeping tag, not a provider identity: skip the comparison
            # and the metadata write in both directions.
            return

        # Held for this whole method, not just by callers that happen to
        # already have it: ``store_embedding`` holds it for its entire body
        # (a free re-entrant no-op here), but ``verify_embedding_model`` — a
        # public entry point — used to call this with no lock at all, so two
        # concurrent first-callers could both see no stored model and both
        # write the lock row. Acquiring it here, in the shared helper, covers
        # every caller by construction rather than relying on each one to
        # remember to wrap it.
        async with self._write_lock:
            # Ensure _metadata table exists (idempotent).
            await self._migrate_core_v4_to_v5()

            _query_prefix, document_prefix = _role_prefixes(self._embedding_provider)
            active_fingerprint = _document_prefix_fingerprint(document_prefix)

            cursor = await self._db.execute(
                "SELECT value FROM _metadata WHERE key = 'embedding_model_name'"
            )
            row = await cursor.fetchone()

            if row is None:
                # First embedding — lock the model.
                await self._db.execute(
                    "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                    ("embedding_model_name", model_name),
                )
                await self._db.execute(
                    "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                    ("embedding_dimension", str(dimension)),
                )
                # Only a non-empty document prefix records a fingerprint — an
                # unprefixed corpus keeps the legacy _metadata shape untouched.
                if active_fingerprint is not None:
                    await self._db.execute(
                        "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                        (_METADATA_DOCUMENT_PREFIX_FINGERPRINT, active_fingerprint),
                    )
                if _query_prefix:
                    await self._db.execute(
                        "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                        (_METADATA_QUERY_PREFIX, _query_prefix),
                    )
                if commit:
                    await self._maybe_commit()
            else:
                stored_model = row["value"]
                dim_cursor = await self._db.execute(
                    "SELECT value FROM _metadata WHERE key = 'embedding_dimension'"
                )
                dim_row = await dim_cursor.fetchone()
                stored_dimension = int(dim_row["value"]) if dim_row else 0

                fp_cursor = await self._db.execute(
                    "SELECT value FROM _metadata WHERE key = ?",
                    (_METADATA_DOCUMENT_PREFIX_FINGERPRINT,),
                )
                fp_row = await fp_cursor.fetchone()
                stored_fingerprint = fp_row["value"] if fp_row else None

                if (
                    stored_model != model_name
                    or stored_dimension != dimension
                    or stored_fingerprint != active_fingerprint
                ):
                    raise EmbeddingModelMismatchError(
                        stored_model=self._describe_corpus_model(stored_model, stored_fingerprint),
                        configured_model=self._describe_corpus_model(
                            model_name, active_fingerprint
                        ),
                        stored_dimension=stored_dimension,
                        configured_dimension=dimension,
                    )

    @staticmethod
    def _describe_corpus_model(model_name: str, fingerprint: str | None) -> str:
        """Render a model identity that includes any document-prefix fingerprint.

        Keeps the plain model name for an unprefixed corpus (legacy, matching
        the value stored in ``_metadata``) and appends a short fingerprint tag
        when a document prefix is active, so a prefix-only mismatch produces a
        self-explanatory error rather than two identical model names.

        Args:
            model_name: The embedding model identifier.
            fingerprint: The document-prefix fingerprint, or ``None`` when no
                document prefix is active.

        Returns:
            The model name, suffixed with the document-prefix fingerprint when
            one is present.

        """
        if fingerprint is None:
            return model_name
        return f"{model_name}+doc_prefix:{fingerprint[:12]}"

    async def _ensure_query_prefix_pairs(self) -> None:
        """Verify the active query prefix pairs with the stored corpus.

        For an asymmetric model the query must be embedded with the
        ``query_prefix`` the corpus was built to pair with. Because the query
        prefix does not change any stored vector, a query-only change never
        forces a re-embed — but a *divergent* active query prefix would
        silently degrade ranking, so it is surfaced loudly here at search
        time. Empty prefixes map to the legacy identity (no stored key), so a
        pre-existing store or a symmetric provider never trips this check.
        Until the corpus has locked an embedding model (its first stored
        embedding), there is nothing to pair against, so the check is a
        no-op — searching a not-yet-populated store with a query prefix
        configured never trips.

        Raises:
            EmbeddingQueryPrefixMismatchError: When the provider's active
                ``query_prefix`` differs from the one the corpus records.

        """
        active_query_prefix, _document_prefix = _role_prefixes(self._embedding_provider)

        # No corpus locked yet (no embedding ever stored) → there is no
        # recorded pairing to diverge from, so a configured query prefix on an
        # empty store must not false-trip. Pairing is meaningful only once a
        # corpus exists.
        lock_cursor = await self._db.execute(
            "SELECT value FROM _metadata WHERE key = 'embedding_model_name'"
        )
        if await lock_cursor.fetchone() is None:
            return

        cursor = await self._db.execute(
            "SELECT value FROM _metadata WHERE key = ?",
            (_METADATA_QUERY_PREFIX,),
        )
        row = await cursor.fetchone()
        stored_query_prefix = row["value"] if row else ""

        if stored_query_prefix != active_query_prefix:
            raise EmbeddingQueryPrefixMismatchError(
                stored_query_prefix=stored_query_prefix,
                configured_query_prefix=active_query_prefix,
            )

    async def verify_embedding_model(self) -> None:
        """Explicit eager check for embedding model compatibility.

        Callers that want fail-fast behaviour at startup can invoke this
        after construction.  When no ``embedding_provider`` is set, this
        is a no-op.

        Raises:
            EmbeddingModelMismatchError: When the configured model differs
                from the one stored in ``_metadata``.
            EmbeddingProviderContractError: When the configured provider
                exposes no public ``dimension``.

        """
        if self._embedding_provider is None:
            return
        # Read in the order this method has always read: ``model_name`` first,
        # then the dimension. Only ``dimension`` is translated into a typed
        # error, so a provider missing ``model_name`` as well still fails on that
        # member — as it did before — rather than being told about a different
        # one. Reversing the order to catch that case would change what a
        # conformant provider with stateful properties observes, which this
        # change is required not to do.
        model_name = self._embedding_provider.model_name
        await self._ensure_embedding_model_lock(
            model_name,
            _provider_dimension(self._embedding_provider),
        )

    # ------------------------------------------------------------------
    # Transaction control
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def suppress_access_tracking(self) -> AsyncIterator[None]:
        """Context manager that suppresses access buffering for reads inside it.

        Reads that a component issues as internal machinery — dreaming's own
        candidate scans and reflection-member resolution — or reads routed
        through a read-only view are not caller retrievals and must not feed the
        ``frequency`` signal. Wrapping those reads in this block keeps them out
        of the access buffer. No effect when access tracking is disabled.

        The suppression flag is a task-local ``ContextVar`` and this block scopes
        it with a reset token, so the guarantee holds under **overlapping**
        suppressed reads: two concurrent async tasks each carry their own value
        (neither task's exit clears the other's), and nested suppression on one
        task restores exactly the enclosing state on exit — even on error.

        Yields:
            None — access buffering is suppressed for the duration of the block.

        """
        token = self._suppress_access_tracking.set(True)
        try:
            yield
        finally:
            self._suppress_access_tracking.reset(token)

    @contextlib.asynccontextmanager
    async def suspend_auto_commit(self) -> AsyncIterator[None]:
        """Context manager that disables per-method auto-commit.

        Batches every write in the block into one transaction: the block
        commits once on clean exit and rolls back entirely on any exception.

        **One writer for the duration of the window — now enforced, not just
        documented.** This method takes
        :attr:`_write_lock` for its entire duration, so a *different* task
        calling any guarded write path on this instance blocks until the
        window closes instead of joining its transaction. The *same* task may
        still write freely inside its own window — the lock is task-reentrant
        (see :class:`_TaskReentrantLock`) — which is what lets ``bulk_store``'s
        insert loop, and a caller's own writes issued inside its own
        ``suspend_auto_commit`` block, complete rather than deadlock on
        themselves. Note that opening a *second* store on the same database
        file is not the way out: only one store may write a given file (see
        the concurrency documentation) — the lock is per-instance, not
        cross-connection.

        **Reentrant nesting is supported and safe.** A nested
        ``suspend_auto_commit`` call — on the same task, which is the only way
        to reach one, since the lock above blocks any other task before it
        could nest — shares the *same* transaction as the outermost call:
        :attr:`_skip_auto_commit_depth` counts the nesting, and only the
        **outermost** call's clean exit commits or its exception rolls back;
        an inner call's own exit does neither, and its ``finally`` only
        decrements the depth rather than clearing it outright. Two failures a
        plain ``bool`` produced here, now closed: a clean inner block used to
        commit the outer transaction early (its ``else`` branch committed
        unconditionally, whatever the nesting), and its ``finally`` used to
        reset the flag outright, so per-call autocommit resumed for the rest
        of the outer block even though the outer window was still open.

        Note on the derived-records seam: a create issued inside this block does
        **not** auto-derive (the source is not yet durable and this block owns
        the transaction). ``bulk_store`` dispatches derivation itself, locally,
        after its batch commits; a caller writing inside its own
        ``suspend_auto_commit`` window triggers derivation via an explicit
        re-run/backfill (see :meth:`_dispatch_derivation`).

        **Cancellation is caught alongside every other exception**
        (``except BaseException``, not ``except Exception``):
        :class:`asyncio.CancelledError` derives from ``BaseException``, and this
        block is the *only* thing that can close the transaction while
        ``_skip_auto_commit`` is set — a nested guard such as
        :meth:`_serialize_dedup_probe` deliberately declines to commit or roll
        back anything itself here, leaving that decision to this method (see its
        own docstring). If this method only caught ``Exception``, a
        cancellation landing inside a ``bulk_store`` batch (or any other
        ``suspend_auto_commit`` caller) would skip the rollback entirely, the
        ``finally`` below would still clear ``_skip_auto_commit`` (at the
        outermost level), and the RESERVED lock would be left stranded on the
        connection — ``in_transaction`` stuck ``True`` — blocking every other
        writer on the file until something else eventually commits, rolls
        back, or closes it.

        Both the commit and the rollback are also guarded on
        ``self._db.in_transaction`` — read fresh at that point rather than
        assumed — so a window that happened to do no writes at all (or one
        cancelled before its first write) does not issue a pointless commit or
        rollback call on a connection with nothing open. That guard is skipped
        entirely when :attr:`_connection_quarantined` is already set: a guarded
        write inside the window (e.g. :meth:`_write_readback_savepoint`) can
        itself quarantine the connection while unwinding its own unit, and
        `self._db` is by then a terminal proxy on which even reading
        ``.in_transaction`` raises — checking the flag first, before touching
        `self._db` again, is what keeps that from replacing the original error
        or cancellation with :class:`ConnectionQuarantinedError`.

        **A failed commit ends its own transaction, one way or another.** The
        clean-exit commit above goes through :meth:`_commit_or_recover`, not a
        bare ``self._db.commit()``: a ``COMMIT`` can itself fail (e.g. a
        concurrent reader still holding a lock when ``busy_timeout`` expires)
        while leaving the write transaction open on the connection. Left
        alone, that open transaction would sit there until some later,
        unrelated write on the same connection committed it too — publishing
        this window's work despite the reported failure. See
        :meth:`_commit_or_recover` for the recovery this now performs.

        Yields:
            None — the store operates in deferred-commit mode.

        """
        async with self._write_lock:
            self._skip_auto_commit_depth += 1
            is_outermost = self._skip_auto_commit_depth == 1
            # Fresh identity for THIS window (nested or not) — see
            # `_open_auto_commit_windows` / `_current_auto_commit_window` for
            # why this is separate from `_skip_auto_commit_depth`.
            window_id = object()
            self._open_auto_commit_windows.add(window_id)
            window_token = self._current_auto_commit_window.set(window_id)
            try:
                yield
            except BaseException:
                if self._connection_quarantined:
                    # A guarded write inside this window already quarantined
                    # the connection while unwinding its own unit and left
                    # `self._db` a terminal proxy — touching it again here
                    # (even just reading `.in_transaction`) would replace the
                    # in-flight exception or cancellation with
                    # ConnectionQuarantinedError instead of letting it
                    # propagate. See _write_readback_savepoint for the same
                    # guard.
                    raise
                if is_outermost and self._db.in_transaction:
                    await self._db.rollback()
                raise
            else:
                if is_outermost and self._db.in_transaction:
                    await self._commit_or_recover()
            finally:
                self._skip_auto_commit_depth -= 1
                # Unregister this window's identity and restore the marker
                # this task had before this window opened (the enclosing
                # window's identity, if nested, else `None`) — on every exit
                # path, including cancellation: see `_dispatch_derivation`.
                self._open_auto_commit_windows.discard(window_id)
                self._current_auto_commit_window.reset(window_token)

    async def _open_write_unit(
        self, name: str, *, begin: Literal["DEFERRED", "IMMEDIATE"], opened_transaction: bool
    ) -> None:
        """Open :meth:`_write_readback_savepoint`'s own transaction and ``SAVEPOINT``.

        Split out of that method purely to keep its own cyclomatic
        complexity in check — no behavior lives here that method's callers
        need to know about separately.

        **What this recovers from, precisely.** ``BEGIN`` and ``SAVEPOINT``
        are each awaited separately. This recovers from a cancellation that
        surfaces *after* SQLite has already executed one of those two
        statements for real — on a worker thread this coroutine's own
        cancellation cannot reach — but *before* this call's own ``await``
        returns control here, and only when this call is also the one that
        opened the transaction (``opened_transaction`` is ``True``). In
        that case, on any failure, this checks live connection state
        (``self._db.in_transaction``), never whether the awaited call was
        observed to complete: only a plain ``rollback()`` — never
        ``ROLLBACK TO {name}``, since this call cannot prove the
        ``SAVEPOINT`` itself was ever created, whichever of the two
        statements actually failed. Already-quarantined is a no-op here,
        the same guard :meth:`_write_readback_savepoint`'s own unwind
        applies to itself: touching ``self._db`` again would replace the
        original exception with :class:`ConnectionQuarantinedError` instead
        of letting it propagate. A failed rollback quarantines the
        connection, mirroring every other compensating-rollback failure in
        this class.

        **Two related gaps exist and are deliberately not covered here.**
        Both belong to aiosqlite's own execution model — the same class of
        gap exists at every other call site in this class that awaits a
        single ``self._db.execute(...)`` and predates this method — so
        neither is patched locally; both are tracked as a separate,
        codebase-wide backlog item instead of being half-fixed in one call
        site:

        (a) If the statement is still *queued* on aiosqlite's worker thread
            — not yet actually executed — when the cancellation lands,
            ``self._db.in_transaction`` can read ``False`` at the moment
            this checks it, correctly reflecting that nothing has run yet.
            But the worker thread does not know its queued job was
            cancelled at the Python level, and executes it anyway once it
            gets to it — opening a real transaction on the connection after
            this method has already concluded there was nothing to roll
            back, and returned. Nothing here can observe that after the
            fact.
        (b) If this call enters with a transaction *already* open
            (``opened_transaction`` is ``False``) and the ``SAVEPOINT``
            itself executes before the cancellation surfaces, that
            savepoint is never released: the rollback above is gated on
            ``opened_transaction`` precisely because a caller-held
            transaction must not be ended by this call's own failure, so
            nothing here ends it, or the savepoint nested inside it,
            either. A transaction a caller already held is **not**
            guaranteed to be left exactly as it was in this case — only
            that this call never ends it and never touches any write the
            caller made before or after this call's own attempt.

        A compensating ``RELEASE {name}`` for (b) — issued only when the
        ``SAVEPOINT`` is known to have executed before a later failure —
        was considered and left out: the exception raised from
        ``await self._db.execute(f"SAVEPOINT {name}")`` looks identical
        whether that statement genuinely created the savepoint and then
        raced with a cancellation, or never created anything because the
        statement itself failed outright, and nothing observable here
        distinguishes the two. Issuing ``RELEASE {name}`` on the strength
        of a guess is not the trivially-safe addition it would need to be:
        ``RELEASE`` targets the innermost, most-recently-created savepoint
        with that literal name, so if this call's own ``SAVEPOINT`` never
        actually existed, that statement would instead release some
        unrelated, older, still-legitimate savepoint the caller's own
        transaction happens to carry under the same name — a worse outcome
        than the leak it would be trying to close. Left undone; this is
        that "otherwise leave it and say so".

        Args:
            name: Savepoint name — see :meth:`_write_readback_savepoint`.
            begin: ``"IMMEDIATE"`` or ``"DEFERRED"`` — see
                :meth:`_write_readback_savepoint`.
            opened_transaction: Whether this call is the one opening the
                transaction, sampled by the caller before either statement
                below ran.

        Raises:
            BaseException: The original failure (``asyncio.CancelledError``
                included), or a cancellation raised by the rollback itself,
                which always outranks it.

        """
        try:
            if opened_transaction:
                await self._db.execute(f"BEGIN {begin}")
            await self._db.execute(f"SAVEPOINT {name}")
        except BaseException as exc:
            if self._connection_quarantined:
                raise
            if opened_transaction and self._db.in_transaction:
                try:
                    await self._db.rollback()
                except BaseException as unwind_exc:
                    await self._quarantine_connection(
                        f"{name} could not roll back after failing to open its own "
                        f"unit: {exc!r}: {unwind_exc!r}"
                    )
                    if isinstance(unwind_exc, asyncio.CancelledError):
                        raise
                    raise exc from unwind_exc
            raise

    @contextlib.asynccontextmanager
    async def _write_readback_savepoint(
        self, name: str, *, begin: Literal["DEFERRED", "IMMEDIATE"]
    ) -> AsyncIterator[None]:
        """Make a journaled write one failure-atomic unit, with its journal entry.

        The store's journaled insert, update and delete paths use this to
        make their own row write(s) and their own journal append recover
        together. ``update_thought``, ``restore_thought``, ``update_edge``
        and ``update_action`` additionally re-read the row to report and
        journal the state actually stored, so for those the unit also covers
        that read-back. Without this wrapper, a failure inside the unit (a
        vanished row, a driver error, the row mapper rejecting a stored
        value, a failed or cancelled journal append) propagated while the
        write itself stayed pending in the connection's transaction — a
        later, unrelated commit on the same connection would then publish a
        mutation whose own operation had reported failure.

        The journal append runs inside this same block, not after it, for
        the identical reason: ``JournalWriter.append`` awaits a chain-tail
        read before its own ``INSERT``, and a failure or cancellation in
        that await must unwind the row write too — a mutation the journal
        never recorded must never be the one thing that survives. Releasing
        the savepoint before the append ran (the original shape of every
        call site above) left that same failure window with no savepoint
        protecting the row write any more, guarded only by
        :attr:`_write_lock` — a lock, not a transaction guard — so a later,
        unrelated commit on the same connection could publish the row write
        with no matching journal entry.

        A bare ``self._db.rollback()`` on that failure is the wrong
        instrument: a caller may already hold this connection's transaction
        open (see :meth:`suspend_auto_commit`), with its own writes pending
        that must not be discarded by a sibling call's unrelated failure.
        This uses a ``SAVEPOINT`` instead, mirroring the established pattern
        in :meth:`_delete_thought_children_explicit` /
        :meth:`_delete_thought_atomic`: a transaction is opened first only
        when :attr:`self._db.in_transaction <aiosqlite.Connection.in_transaction>`
        is not already ``True`` (tracked as ``opened_transaction``).

        **``begin`` is required, with no default, precisely because the two
        modes serve different, incompatible contracts, and a call site must
        say which one it means rather than inherit a silent house default:**

        * ``"IMMEDIATE"`` — for a body that reads before it writes (an FTS5
          trigger reading its own config on insert, an existence check
          before a delete, a before-image or a vector rowid a caller
          deliberately reads after taking this lock). A deferred
          transaction's read takes a WAL snapshot that the later write must
          then upgrade; SQLite refuses that upgrade while another connection
          holds the write lock and returns ``SQLITE_BUSY`` *without ever
          invoking the busy handler*, so the unit would fail at once under
          contention instead of waiting out ``PRAGMA busy_timeout`` like an
          ordinary write. ``BEGIN IMMEDIATE`` takes the write lock up front,
          through the busy handler, before anything in the body gets to
          read — correctness, not a documented retry contract, is what this
          buys. Used by every create and delete path, and by the two delete
          paths that read their before-image / vector rowid after taking
          this same lock.
        * ``"DEFERRED"`` — for the four ``revision``-guarded update paths
          (:meth:`update_thought`, :meth:`restore_thought`,
          :meth:`update_edge`, :meth:`update_action`), which read *before*
          opening this unit. Their guarded ``UPDATE`` is always typed via
          :meth:`_execute_revision_guarded_write`, but **not all four settle
          the same way, and neither settles literally "at once" in
          general** — that shape is narrower than it looks:

          * :meth:`update_thought` alone can fail within milliseconds,
            without the busy handler ever running: a content-changing
            update fires the FTS5 sync trigger, which reads its own config
            as part of the *same* ``UPDATE`` statement, so the write-lock
            upgrade that read forces is refused at once
            (``SQLITE_BUSY``, no wait) while another connection holds the
            lock.
          * :meth:`restore_thought`, :meth:`update_edge` and
            :meth:`update_action` carry no such trigger. Their guarded
            ``UPDATE`` contends through the *ordinary* busy handler and
            waits up to ``PRAGMA busy_timeout`` like any other write. If the
            other writer is still holding the lock once that wait is
            exhausted, this surfaces as typed
            :class:`WriteContentionError`, same as the first case. But if
            the other writer instead commits *while this call is still
            waiting*, the guarded ``UPDATE`` proceeds once the lock frees,
            and finds a genuinely changed ``revision`` — raising
            ``StaleDataError``, correctly, exactly as it would have before
            a write-opening unit ever used ``BEGIN IMMEDIATE`` at all. That
            is not a regression this fix removes; it is the documented
            optimistic-concurrency contract these three have always had.

          What ``DEFERRED`` actually buys, for all four uniformly, is
          narrower and does not depend on which of the two shapes above
          applies: the read that determines ``expected_revision`` happens
          *before* this unit ever competes for the write lock, never after
          waiting for it. Reading only after such a wait — which an
          ``IMMEDIATE`` unit here would do — lets a write guard a revision
          it read *after* another process's edit had already landed,
          rejecting a disjoint-column edit as falsely stale instead of the
          lost update the guard exists to catch. That was measured as a
          regression in engrava-validation's full multiprocess suite when
          this unit read under the lock for these four paths.

        The body then runs inside a named ``SAVEPOINT``, and on any failure
        it is unwound with ``ROLLBACK TO`` + ``RELEASE`` — undoing only what
        this call itself wrote — before the original exception propagates.
        Only when this call is also the one that opened the transaction does
        it end that transaction (with a rollback, never a commit); a
        transaction a caller already held stays open, with only this call's
        own write undone.

        Cancellation is handled the same way: caught alongside every other
        exception (``except BaseException``, not ``except Exception``) so a
        cancellation delivered mid-body still unwinds the savepoint rather
        than leaving it — and the write it guards — dangling on the
        connection.

        **Entry itself is guarded too, not just the body — within limits.**
        ``BEGIN`` and ``SAVEPOINT`` are each awaited separately by
        :meth:`_open_write_unit`, and a cancellation can be delivered after
        SQLite has already executed one of them but before that ``await``
        returns control here — the statement still ran, on a worker thread
        this coroutine's own cancellation cannot reach. When this call is
        the one that opened the transaction, a failure at either point
        rolls that transaction back (checked via live connection state,
        never via whether the awaited call was seen to complete) with a
        plain ``rollback()`` — never ``ROLLBACK TO`` a savepoint this call
        cannot prove was ever created, whichever of the two statements
        actually failed — and the original exception always propagates
        unchanged. When a caller already held the transaction, this never
        ends it or touches the caller's own writes either way, but a
        ``SAVEPOINT`` that raced a cancellation right after really
        executing can be left behind, unreleased, on that caller's
        transaction: see :meth:`_open_write_unit`'s own docstring for
        exactly which two gaps entry does not close, and why.

        **When the unwind itself cannot be trusted, this refuses rather than
        guesses.** If a ``RAISE(ROLLBACK)`` trigger already ended the whole
        transaction (``self._db.in_transaction`` is already ``False`` on the
        failure path), there is nothing left to unwind and the original
        exception propagates unchanged. If the unwind's own statements raise,
        recovery cannot be proven, so the connection is quarantined via
        :meth:`_quarantine_connection` — every later write then fails fast
        with :class:`ConnectionQuarantinedError` instead of ever reaching a
        commit that could make the dangling write durable. A cancellation
        raised by the unwind itself always wins over the error the unwind was
        trying to recover from.

        Args:
            name: Savepoint name, unique among the guarded update paths (a
                fixed literal at every call site, never caller-controlled —
                interpolated directly into the SQL since SQLite does not
                accept savepoint names as bound parameters).
            begin: ``"IMMEDIATE"`` or ``"DEFERRED"`` — see above. Required,
                with no default, so every call site states its own contract
                rather than inheriting one.

        Yields:
            None. The caller performs its write, read-back, and journal
            append inside the block; a failure raised anywhere in it is
            unwound as described above and then re-raised unchanged.

        Raises:
            ConnectionQuarantinedError: When an unwind attempt itself fails
                and a consistent state could not be proven.

        """
        opened_transaction = not self._db.in_transaction
        await self._open_write_unit(name, begin=begin, opened_transaction=opened_transaction)
        try:
            yield
            # The release lives inside this guarded region, deliberately —
            # see _delete_thought_children_explicit's docstring for why: a
            # cancellation delivered while awaiting this specific call can
            # still see the RELEASE complete on the connection, and the
            # unwind below has to be able to recognise that as "already
            # released", not "the write never happened".
            await self._db.execute(f"RELEASE {name}")
        except BaseException as exc:
            if self._connection_quarantined:
                # A unit this block wraps (e.g. _delete_thought_atomic) already
                # quarantined the connection while unwinding its own nested
                # savepoint and left `self._db` a terminal proxy — see that
                # method's docstring. Touching `self._db` again here — even
                # just reading `.in_transaction` — would replace `exc` with
                # ConnectionQuarantinedError instead of letting it propagate.
                raise
            if not self._db.in_transaction:
                # A RAISE(ROLLBACK) trigger already ended the whole
                # transaction (savepoint included). Nothing is left open to
                # unwind, and nothing this call wrote can outlive it.
                raise
            try:
                await self._db.execute(f"ROLLBACK TO {name}")
                await self._db.execute(f"RELEASE {name}")
                if opened_transaction:
                    await self._db.rollback()
            except BaseException as unwind_exc:
                await self._quarantine_connection(
                    f"{name} could not unwind its savepoint after {exc!r}: {unwind_exc!r}"
                )
                if isinstance(unwind_exc, asyncio.CancelledError):
                    raise
                raise exc from unwind_exc
            raise

    async def _execute_revision_guarded_write(
        self,
        sql: str,
        params: tuple[object, ...],
        *,
        operation: str,
    ) -> aiosqlite.Cursor:
        """Execute a ``revision``-guarded ``UPDATE``, typing lock contention.

        ``update_thought``, ``restore_thought``, ``update_edge`` and
        ``update_action`` all execute their guarded write through this, so a
        lock timeout on any of them surfaces as :class:`WriteContentionError`
        rather than a raw :class:`sqlite3.OperationalError` — the same
        conversion :meth:`_begin_dedup_write_lock` already performs for the
        dedup window's own ``BEGIN IMMEDIATE``. An ordinary ``UPDATE`` from a
        second connection or process has always been able to hit the write
        lock and time out — a ``revision`` predicate in the ``WHERE`` clause
        does not, by itself, introduce SQL-level contention that was not
        already there. What changed is how often that contention now surfaces
        as a *typed* error: before these four paths ran through this method, a
        lock timeout on any of them propagated as a raw
        :class:`sqlite3.OperationalError`, and only
        ``_begin_dedup_write_lock``'s cross-connection window converted its
        own — leaving these four untyped would now be a gap this method
        exists to close, not a property they merely inherit.

        No application-level retry is added: ``PRAGMA busy_timeout`` already
        makes SQLite wait out ordinary contention inside this single
        ``execute()`` call, and a caller is free to retry the whole operation
        (a safe thing to do, since nothing was written before this call
        raised). Only a busy/lock error converts; any other
        :class:`sqlite3.OperationalError` (a genuine I/O failure, a schema
        problem) is never mistaken for contention and propagates unchanged.

        Args:
            sql: The guarded ``UPDATE`` statement, as built by
                :func:`_build_update_sql`.
            params: The statement's bound parameters, in order.
            operation: Name of the calling public method, carried onto
                :class:`WriteContentionError` for a caller that logs or
                branches on it.

        Returns:
            The cursor from the executed statement.

        Raises:
            WriteContentionError: The write could not proceed because the
                connection reported lock contention.
            sqlite3.OperationalError: Some other, non-busy failure.

        """
        try:
            return await self._db.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc):
                raise
            raise WriteContentionError(operation=operation, attempts=1) from exc

    async def _rollback_self_opened_transaction(
        self, *, opened_transaction: bool, exc: BaseException
    ) -> None:
        """Undo a transaction this call itself opened, on a failure before any write.

        ``delete_thought`` and ``delete_edge`` each open their own ``BEGIN
        IMMEDIATE`` *before* reading their journal before-image (and, for
        ``delete_thought``, the embedding rowid the vector purge needs) — see
        those methods' docstrings for why the write lock is taken before that
        read. A failure raised while performing that read strikes before
        :meth:`_write_readback_savepoint`'s own ``SAVEPOINT`` is ever
        created, so there is nothing to unwind with ``ROLLBACK TO`` — only
        the transaction itself, and only when this call is the one that
        opened it. A transaction a caller already held when this call
        started (``opened_transaction`` is ``False``) is left exactly as it
        was.

        Deliberately a plain ``rollback()``, not the savepoint-aware unwind
        :meth:`_write_readback_savepoint` uses: at this point nothing has
        been written, so undoing the whole transaction and undoing "only
        this call's own write" are the same thing. Mirrors that method's own
        compensating-rollback failure handling: when the rollback itself
        fails, the connection's state cannot be trusted, so it is
        quarantined via :meth:`_quarantine_connection` instead of leaving an
        indeterminate transaction for a later, unrelated commit to publish.
        Already-quarantined is also a no-op here — :meth:`_write_readback_savepoint`
        may have quarantined the connection on its own unwind failure before
        this ever runs, and touching ``self._db`` again would replace ``exc``
        with :class:`ConnectionQuarantinedError` instead of letting it
        propagate, the same guard that method's own handler applies to
        itself.

        Args:
            opened_transaction: Whether this call is the one that opened the
                active transaction, sampled before this call touched the
                connection.
            exc: The exception that triggered this cleanup — chained onto a
                rollback failure, or left untouched when the rollback
                succeeds (the caller re-raises it unchanged either way).

        Raises:
            BaseException: A cancellation raised by the rollback itself,
                which always outranks ``exc``.

        """
        if self._connection_quarantined:
            return
        if not opened_transaction or not self._db.in_transaction:
            return
        try:
            await self._db.rollback()
        except BaseException as rollback_exc:
            await self._quarantine_connection(
                f"rollback after {exc!r} also failed: {rollback_exc!r}"
            )
            if isinstance(rollback_exc, asyncio.CancelledError):
                raise
            raise exc from rollback_exc

    def _ensure_connection_usable(self) -> None:
        """Fail fast when the connection has been quarantined.

        Called at the start of public operations so a caller cannot run against
        a connection whose transaction state is indeterminate (see
        :attr:`_connection_quarantined`). It is also the universal write
        backstop: :meth:`_maybe_commit` calls it before every commit, so no code
        path — guarded entry point or not — can flush an orphaned transaction on
        a quarantined connection.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        if self._connection_quarantined:
            raise ConnectionQuarantinedError(self._quarantine_reason or "connection unusable")

    @staticmethod
    async def _drain_shielded(
        task: asyncio.Task[None], *, timeout_seconds: float | None = None
    ) -> asyncio.CancelledError | None:
        """Await a shielded task, up to an optional bound, returning any cancellation of us.

        The task is observed via a **single** :func:`asyncio.wait` waiter that is
        awaited under :func:`asyncio.shield` and reused across every cancellation
        of our await (no new waiter is spawned per cancellation). ``asyncio.wait``
        surfaces the task's outcome without raising it, and the shielded waiter
        keeps observing the task to the end no matter how many times *our* await
        is cancelled. A cancellation of our await is captured and returned (never
        swallowed) so the caller can honor it once the task is safely complete;
        the task's own success/failure is left on the task for the caller.

        ``timeout_seconds`` bounds *our own observation*, never the task: it is
        forwarded straight to the single ``asyncio.wait`` call above, so the
        deadline is set once, when this is first called, and does not restart on
        a repeated cancellation of our await (the same "one waiter, reused"
        property the unbounded case already relies on). ``task`` is never
        cancelled by a timeout — the caller decides what an unanswered task means
        by checking ``task.done()`` once this returns; a task that has not
        answered keeps running afterwards exactly as it would have without a
        bound, which is what lets a later drain (a piggybacking ``close()``, or
        :meth:`_quarantine_connection`) keep observing the very same task rather
        than a cancelled one.

        Args:
            task: The already-scheduled task to drain.
            timeout_seconds: Maximum time to wait for ``task``. ``None`` (the
                default) waits without a bound, exactly as before this parameter
                existed.

        Returns:
            The last ``CancelledError`` raised into our await, or ``None``. Check
            ``task.done()`` after this returns to tell a real bound expiry
            (``False``) apart from the task actually finishing (``True``).

        """
        waiter = asyncio.ensure_future(asyncio.wait({task}, timeout=timeout_seconds))
        cancelled: asyncio.CancelledError | None = None
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError as cancel_exc:
                cancelled = cancel_exc
        # Consume the waiter's result so it is never an unretrieved exception.
        with contextlib.suppress(BaseException):
            waiter.result()
        return cancelled

    @staticmethod
    def _log_close_failure_over_pending_cancellation(task: asyncio.Task[None]) -> None:
        """Consume a close task's result when a cancellation already outranks it.

        The rule, stated generally so the next cleanup path here can reuse
        it rather than re-derive it: **the caller's own cancellation
        outranks anything the cleanup discovers about itself.** When
        something earlier in ``close()`` (the access-buffer flush) has
        already deferred a cancellation, the connection's physical close
        still runs to completion -- but its own outcome must not replace
        the cancellation the caller actually needs to see. ``task.result()``
        is still called so a close failure is never an unretrieved task
        exception, and a genuine failure is logged with the exception
        itself -- the only trace of it that will exist -- rather than
        raised in front of the cancellation.

        Args:
            task: The completed close task.

        """
        try:
            task.result()
        except BaseException:
            logger.warning(
                "close() failed while a cancellation was already pending",
                exc_info=True,
            )

    async def _abandon_expired_close(self, task: asyncio.Task[None]) -> None:
        """Quarantine the store after :meth:`close`'s own bound expires unanswered.

        Called only when a caller of :meth:`close` stops waiting on the
        physical close task because :attr:`_close_timeout_seconds` elapsed
        with the task still not ``done()``. The task itself is never
        cancelled here or anywhere upstream — it keeps running (or not) in
        the background exactly as :meth:`_quarantine_connection`'s own
        detached close already does; this only stops *this call* from
        waiting on it any longer.

        Delegates to :meth:`_quarantine_connection`, which is idempotent, so
        a concurrent quarantine (a prior compensating-rollback failure, a
        different caller's own ``close()`` also timing out on this same
        task, or a second ``close()`` after this one already gave up)
        collapses onto whichever caller gets there first. Because ``task``
        is already installed in :attr:`_quarantine_close_task` by the time
        this runs, :meth:`_quarantine_connection`'s own physical-close step
        finds it non-``None`` and defers to it rather than starting a second
        one — the same coordination :meth:`close` already relies on for a
        concurrent quarantine, reused here so an expired bound can never
        cause a second physical close on the pinned connection.

        That deferral also means :meth:`_quarantine_connection` will *not*
        attach its own done-callback to ``task`` (it only does that for a
        task it creates itself), so this method attaches
        :meth:`_consume_quarantine_close` directly — otherwise the task's
        eventual outcome, whenever the worker finally answers, if ever,
        would be reported as an unretrieved task exception instead of
        quietly discarded. Attaching it more than once (a second caller
        hitting this same path for the same task) is harmless: the callback
        only reads the outcome, which is safe to read repeatedly.

        Args:
            task: The in-flight physical-close task this call gave up
                waiting on.

        """
        task.add_done_callback(self._consume_quarantine_close)
        await self._quarantine_connection(
            f"close() did not complete within {self._close_timeout_seconds:.1f}s "
            "-- the connection worker has not answered"
        )

    @staticmethod
    def _consume_quarantine_close(task: asyncio.Task[None]) -> None:
        """Done-callback that consumes the detached best-effort close outcome.

        Retrieves any exception so the task is never reported as an
        unretrieved-exception; a *cancelled* close task carries no exception to
        retrieve and is left alone (``exception()`` would raise on it). The
        outcome is irrelevant to correctness — the proxy + token already
        guarantee terminality — so it is never re-raised.

        Args:
            task: The completed detached close task.

        """
        if not task.cancelled():
            # Retrieve (and discard) any close failure so it is not logged as an
            # unretrieved task exception.
            task.exception()

    async def _quarantine_connection(self, reason: str) -> None:
        """Make the store terminally unusable, by construction and independent of close.

        Kept ``async`` for call-site symmetry with the compensating-rollback flow
        and so the "returns promptly even if close hangs" liveness contract is
        awaitable in tests; it intentionally **awaits nothing** — every step is
        synchronous and the physical close is *detached*.

        Terminal-by-construction, in three synchronous steps (nothing here can be
        cancelled or blocked, so a caller-frame cancellation is never swallowed —
        it is simply delivered at the caller's next await and propagates):

        1. Set the flag — guarded entry points and :meth:`_maybe_commit` fail fast
           with a typed :class:`ConnectionQuarantinedError`.
        2. Revoke the shared :class:`ConnectionRevocationToken` — every *other*
           holder of the real connection (the :class:`JournalWriter`) fails hard
           on its next connection-touching method, so it cannot bypass the proxy.
        3. Detach the real connection, swap in a :class:`_QuarantinedConnection`
           proxy (so every core-initiated op raises), and schedule a **bounded,
           detached** best-effort close for resource cleanup — unless
           :meth:`close` already started (or is starting) that same physical
           close, in which case this step defers to it entirely rather than
           entering ``real_conn.close()`` a second time (see :meth:`close`
           for the race two concurrent closes on the same connection can
           hit). Quarantine returns promptly even if that close hangs
           forever — safety never depends on it. A done-callback consumes
           the close outcome so a failure/cancellation is never an
           unretrieved-task warning, and the task is retained so it is not
           GC'd while pending.

        Idempotent: a second call is a no-op (already quarantined).

        Guarantee / limitation: quarantine synchronously revokes *admission* —
        every NEW operation on the store or its journal fails fast with
        :class:`ConnectionQuarantinedError`, and direct core connection access is
        terminal via the proxy — so no write/commit can flush an orphaned
        transaction, regardless of whether the physical close succeeds. It does
        NOT retract an operation already admitted before revocation: a reader
        admitted just before revocation may complete its in-flight read on the
        pre-revocation connection — a possibly-stale read, never a commit.

        Args:
            reason: Human-readable cause, surfaced on every raised error.

        """
        if self._connection_quarantined:
            return
        self._connection_quarantined = True
        self._quarantine_reason = reason
        self._revocation.revoke(reason)
        real_conn = self._db
        self._db = _QuarantinedConnection(reason)  # type: ignore[assignment]  # terminal proxy: every later use must fail
        # Detached best-effort close: schedule and return; do NOT await it, so a
        # hung close can never block quarantine (safety is already guaranteed by
        # the proxy + token). Retain the task and consume its result via callback.
        # But only schedule it if nothing has already initiated the physical
        # close of this same real connection -- self._quarantine_close_task
        # doubles as that shared "already in progress" marker, checked and set
        # here with no await in between, which is what makes this race-free
        # against a concurrent close() under cooperative scheduling: only one
        # of the two ever wins the check.
        if self._quarantine_close_task is None:
            close_task = asyncio.get_running_loop().create_task(real_conn.close())
            self._quarantine_close_task = close_task
            close_task.add_done_callback(self._consume_quarantine_close)

    async def _commit_or_recover(self) -> None:
        """Commit the current transaction; unwind or quarantine when the commit itself fails.

        Both runtime commit call sites — this one (via :meth:`_maybe_commit`)
        and :meth:`suspend_auto_commit`'s own clean-exit commit — go through
        this method rather than a bare ``self._db.commit()``. A ``COMMIT`` can
        fail on its own account (most concretely: a concurrent connection
        still holding a read lock when ``busy_timeout`` expires, reported as
        ``SQLITE_BUSY``) while the write transaction stays open on the
        connection — SQLite does not roll a transaction back just because its
        ``COMMIT`` failed. Left alone, that open transaction would sit there
        until some later, unrelated write on the *same* connection committed
        it too, silently publishing the failed operation's changes alongside
        its own.

        On a commit failure, a rollback of the now-known-bad transaction is
        attempted:

        * If the transaction already closed on its own (``self._db.in_transaction``
          is already ``False`` — some commit failures do end it, e.g. a
          trigger-raised ``RAISE(ROLLBACK)`` surfacing at commit time), there is
          nothing left to unwind and the commit failure propagates unchanged.
        * If the rollback succeeds, the transaction is gone and the commit
          failure propagates — nothing from this window can become durable via
          a later commit.
        * If the rollback *also* fails, the connection's state cannot be
          trusted, so it is quarantined via :meth:`_quarantine_connection` —
          mirroring :meth:`_write_readback_savepoint`'s own unwind-failure
          handling — so every later write fails fast with
          :class:`ConnectionQuarantinedError` instead of ever reaching a
          commit that could flush the orphaned transaction. A cancellation
          raised by the rollback itself always outranks the commit failure it
          was trying to recover from, exactly as in
          :meth:`_write_readback_savepoint`.

        Not weakened, and deliberately not retried: no application-level retry
        is added here, so a caller sees the real failure instead of it being
        silently absorbed — ``PRAGMA busy_timeout`` has already given SQLite
        every chance to succeed before this runs at all.

        Raises:
            BaseException: The original commit failure (chained to the
                rollback failure, via ``__cause__``, when the rollback also
                fails), or a cancellation raised by the rollback itself.

        """
        try:
            await self._db.commit()
        except BaseException as exc:
            if not self._db.in_transaction:
                raise
            try:
                await self._db.rollback()
            except BaseException as rollback_exc:
                await self._quarantine_connection(
                    f"commit failed and the compensating rollback also failed: "
                    f"{exc!r}; rollback error: {rollback_exc!r}"
                )
                if isinstance(rollback_exc, asyncio.CancelledError):
                    raise
                raise exc from rollback_exc
            raise

    async def _maybe_commit(self) -> None:
        """Commit if auto-commit is not suspended.

        Fails fast with a typed error on a quarantined connection. This flag
        check is the fast path for the common commit; the hard backstop is the
        ``_QuarantinedConnection`` proxy on ``self._db`` — even a commit that
        skipped this check would raise on ``self._db.commit()``.

        The commit itself goes through :meth:`_commit_or_recover`, so a commit
        that fails on its own account (rather than the transaction body having
        already raised) is unwound — rolled back, or the connection quarantined
        if the rollback also fails — instead of leaving an open transaction for
        a later, unrelated commit on this connection to publish by accident.

        **Write-lock classification: relies on every caller, not on its own
        body.** Every one of this method's call sites is the last step of a
        guarded write already running under ``_write_lock`` — its own
        acquisition (``_insert_new_thought_row``, ``update_thought``,
        ``restore_thought``, ``delete_thought``, ``create_edge``,
        ``update_edge``, ``delete_edge``, ``store_embedding``,
        ``record_access``, ``flush_access_buffer``, ``create_action``,
        ``update_action``, ``cleanup_expired``, ``_insert_derived_row``,
        ``_insert_derived_edge``, ``_ensure_embedding_model_lock``) or a named
        caller's (``_increment_confirmation`` via ``_create_thought_with_dedup``
        / ``get_or_create``; ``run_hygiene`` holds it directly). ``upsert_by_hash``
        reaches this method only via ``update_thought``'s own call on its
        matched-and-changed branch — its unchanged-match branch writes nothing
        and does not call this method at all. This method never acquires the
        write lock itself.

        **Nested-commit suppression, task-local.** ``run_hygiene`` sets
        :data:`_SUPPRESS_NESTED_AUTO_COMMIT` for the duration of its
        archive+GC unit so a nested public write it makes itself
        (``retire_orphan_reflections`` -> ``update_thought``) does not end
        that unit early with its own commit here — see that ContextVar's
        module-level docstring for why it is task-local rather than the
        instance-wide :attr:`_skip_auto_commit_depth`.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined
                (already, or by this call's own failed-commit recovery).

        """
        self._ensure_connection_usable()
        if not self._skip_auto_commit and not _SUPPRESS_NESTED_AUTO_COMMIT.get():
            await self._commit_or_recover()

    # ------------------------------------------------------------------
    # Row -> Domain mappers (template methods — override in subclasses)
    # ------------------------------------------------------------------

    def _row_to_thought(self, row: aiosqlite.Row) -> ThoughtRecord:
        """Map a SQLite row to a ThoughtRecord.

        Override in subclasses to produce extended model types.

        Args:
            row: A row from the thought table.

        Returns:
            A ThoughtRecord domain model.

        """
        keys = row.keys()
        source_type_raw = row["source_type"] if "source_type" in keys else None
        confirmation_raw = row["confirmation_count"] if "confirmation_count" in keys else 0
        consolidated_raw = row["consolidated_from"] if "consolidated_from" in keys else None
        visibility_raw = row["visibility"] if "visibility" in keys else "selective"
        access_count_raw = row["access_count"] if "access_count" in keys else 0
        action_outcome_raw = row["action_outcome_score"] if "action_outcome_score" in keys else None
        last_accessed_at_raw = row["last_accessed_at"] if "last_accessed_at" in keys else None
        created_at_raw = row["created_at"] if "created_at" in keys else None
        updated_at_raw = row["updated_at"] if "updated_at" in keys else None
        expires_at_raw = row["expires_at"] if "expires_at" in keys else None
        valid_from_raw = row["valid_from"] if "valid_from" in keys else None
        valid_until_raw = row["valid_until"] if "valid_until" in keys else None
        metadata_json_raw = row["metadata_json"] if "metadata_json" in keys else "{}"
        metadata_decoded: dict[str, MetadataValue] = (
            json.loads(metadata_json_raw) if metadata_json_raw else {}
        )
        provenance_raw = row["provenance"] if "provenance" in keys else None
        pinned_raw = row["pinned"] if "pinned" in keys else 0
        archived_at_cycle_raw = row["archived_at_cycle"] if "archived_at_cycle" in keys else None
        archived_at_raw = row["archived_at"] if "archived_at" in keys else None
        return ThoughtRecord(
            thought_id=row["thought_id"],
            thought_type=ThoughtType(row["thought_type"]),
            essence=row["essence"],
            content=row["content"],
            priority=Priority(row["priority"]),
            lifecycle_status=LifecycleStatus(row["lifecycle_status"]),
            created_cycle=row["created_cycle"],
            updated_cycle=row["updated_cycle"],
            source=row["source"],
            confidence=row["confidence"],
            embedding_ref=row["embedding_ref"],
            source_type=(
                KnowledgeSource(source_type_raw) if source_type_raw else KnowledgeSource.EXPERIENCE
            ),
            confirmation_count=int(confirmation_raw) if confirmation_raw else 0,
            consolidated_from=_decode_consolidated(consolidated_raw),
            visibility=(
                ThoughtVisibility(visibility_raw) if visibility_raw else ThoughtVisibility.SELECTIVE
            ),
            access_count=int(access_count_raw) if access_count_raw else 0,
            action_outcome_score=(
                float(action_outcome_raw) if action_outcome_raw is not None else None
            ),
            last_accessed_at=last_accessed_at_raw,
            created_at=created_at_raw,
            updated_at=updated_at_raw,
            expires_at=expires_at_raw,
            valid_from=valid_from_raw,
            valid_until=valid_until_raw,
            metadata=metadata_decoded,
            provenance=_decode_provenance(provenance_raw),
            pinned=bool(pinned_raw),
            archived_at_cycle=(
                int(archived_at_cycle_raw) if archived_at_cycle_raw is not None else None
            ),
            archived_at=archived_at_raw,
        )

    async def _get_thought_row(self, thought_id: str) -> aiosqlite.Row | None:
        """Fetch a raw thought row without applying retrieval hooks.

        Args:
            thought_id: UUID of the thought.

        Returns:
            Raw SQLite row, or ``None`` if not found.

        """
        cursor = await self._db.execute("SELECT * FROM thought WHERE thought_id = ?", (thought_id,))
        return await cursor.fetchone()

    async def _get_edge_row(self, edge_id: str) -> aiosqlite.Row | None:
        # Write-lock classification: not a write path at all -- a plain SELECT.
        """Fetch a raw edge row without applying transformations.

        Args:
            edge_id: UUID of the edge.

        Returns:
            Raw SQLite row, or ``None`` if not found.

        """
        cursor = await self._db.execute("SELECT * FROM edge WHERE edge_id = ?", (edge_id,))
        return await cursor.fetchone()

    # ------------------------------------------------------------------
    # ThoughtRecord CRUD
    # ------------------------------------------------------------------

    _CORE_INSERT_SQL = (
        "INSERT INTO thought "
        "(thought_id, thought_type, essence, content, content_hash, priority, "
        " lifecycle_status, created_cycle, updated_cycle, source, "
        " confidence, embedding_ref, source_type, confirmation_count, "
        " consolidated_from, visibility, access_count, action_outcome_score, "
        " last_accessed_at, created_at, updated_at, expires_at, "
        " valid_from, valid_until, "
        " metadata_json, provenance, pinned, archived_at_cycle, archived_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?)"
    )

    def _thought_to_core_params(self, thought: ThoughtRecord) -> tuple[object, ...]:
        """Extract core SQL parameters from a ThoughtRecord.

        Computes ``content_hash`` deterministically from
        ``thought.content`` (SHA-256 of the UTF-8 bytes, no normalization),
        so duplicate detection is always based on byte-exact content.

        Serializes ``thought.metadata`` with ``ensure_ascii=False`` so
        non-ASCII attribute values (speaker names, language strings,
        ...) survive a write/read round trip byte-exact.

        Serializes ``thought.provenance`` via ``model_dump_json`` when
        present, or ``None`` (a SQL NULL) when absent — so a thought with no
        provenance writes a NULL column and is byte-identical to a pre-feature
        row.

        Args:
            thought: The thought record.

        Returns:
            Tuple of parameter values for ``_CORE_INSERT_SQL``.

        """
        return (
            thought.thought_id,
            thought.thought_type.value,
            thought.essence,
            thought.content,
            _compute_content_hash(thought.content),
            thought.priority.value,
            thought.lifecycle_status.value,
            thought.created_cycle,
            thought.updated_cycle,
            thought.source,
            thought.confidence,
            thought.embedding_ref,
            thought.source_type.value,
            thought.confirmation_count,
            _encode_consolidated(thought.consolidated_from),
            thought.visibility.value,
            thought.access_count,
            thought.action_outcome_score,
            thought.last_accessed_at,
            thought.created_at,
            thought.updated_at,
            thought.expires_at,
            thought.valid_from,
            thought.valid_until,
            json.dumps(thought.metadata, ensure_ascii=False),
            _encode_provenance(thought.provenance),
            int(thought.pinned),
            thought.archived_at_cycle,
            thought.archived_at,
        )

    #: Guard clause every core thought UPDATE carries: the row identity plus the
    #: ``revision`` this call read. The engine itself increments ``revision`` by
    #: one on every guarded write (``revision = revision + 1`` in the same
    #: statement that checks it — see :func:`_build_update_sql`'s
    #: ``bump_column``), so a write is rejected whenever *any* other guarded
    #: write landed since this call's own read, or the row was deleted — the two
    #: cases ``rowcount == 0`` cannot tell apart. Being on *every* update, it
    #: also rejects an edit that shares no column with the row-version-moving
    #: one; the public docstrings say both rather than implying a general
    #: staleness check. ``updated_cycle`` is not this guard's column: it is a
    #: cognitive-recency signal nothing in this store advances on its own, and
    #: is not safe to overload as a write counter (see the module history for
    #: why the two were once, incorrectly, the same column).
    _CORE_UPDATE_GUARD = "thought_id = ? AND revision = ?"

    def _thought_to_core_columns(self, thought: ThoughtRecord) -> dict[str, object]:
        """Map a ThoughtRecord to the column values an UPDATE may write.

        Mirrors :py:meth:`_thought_to_core_params` for the value encoding of
        every column, but keyed by column name rather than positional so an
        update can write a **subset**. That is what keeps an edit from
        rewriting columns it does not own: high-volume telemetry
        (``access_count``, ``last_accessed_at``) and ``confirmation_count`` are
        maintained by other operations, and a whole-record write would silently
        roll back whatever they stored since the row was read.

        Two columns are absent, for different reasons. ``thought_id``
        identifies the row being updated, so it is correctly not assignable
        here. ``content_hash`` is a **known defect being preserved, not a
        design choice**: no update has ever written it, so editing ``content``
        leaves the stored hash pointing at the superseded text and content-hash
        deduplication then matches the row on content it no longer holds.
        Repairing it changes deduplication behaviour and belongs to a change of
        its own; it is carried unchanged here so this rewrite stays
        behaviour-preserving.

        Args:
            thought: The thought record to encode.

        Returns:
            Mapping of column name to the SQL value for that column.

        """
        return {
            "thought_type": thought.thought_type.value,
            "essence": thought.essence,
            "content": thought.content,
            "priority": thought.priority.value,
            "lifecycle_status": thought.lifecycle_status.value,
            "created_cycle": thought.created_cycle,
            "updated_cycle": thought.updated_cycle,
            "source": thought.source,
            "confidence": thought.confidence,
            "embedding_ref": thought.embedding_ref,
            "source_type": thought.source_type.value,
            "confirmation_count": thought.confirmation_count,
            "consolidated_from": _encode_consolidated(thought.consolidated_from),
            "visibility": thought.visibility.value,
            "access_count": thought.access_count,
            "action_outcome_score": thought.action_outcome_score,
            "last_accessed_at": thought.last_accessed_at,
            "created_at": thought.created_at,
            "updated_at": thought.updated_at,
            "expires_at": thought.expires_at,
            "valid_from": thought.valid_from,
            "valid_until": thought.valid_until,
            "metadata_json": json.dumps(thought.metadata, ensure_ascii=False),
            "provenance": _encode_provenance(thought.provenance),
            "pinned": int(thought.pinned),
            "archived_at_cycle": thought.archived_at_cycle,
            "archived_at": thought.archived_at,
        }

    def _thought_update_columns(
        self,
        current: ThoughtRecord,
        updated: ThoughtRecord,
    ) -> dict[str, object]:
        """Return the columns an edit owns: what it changed, plus the stamp.

        A column is owned when the operation gave it a new value; everything
        else keeps whatever is in storage, including values written by another
        writer since ``current`` was read. ``updated_at`` is always owned
        because :py:meth:`ThoughtRecord.evolve` restamps it on every edit.

        Args:
            current: The record as it was read at the start of the operation.
            updated: The record the operation wants to persist.

        Returns:
            Mapping of column name to SQL value, never empty.

        """
        before = self._thought_to_core_columns(current)
        after = self._thought_to_core_columns(updated)
        columns = {name: value for name, value in after.items() if before[name] != value}
        columns["updated_at"] = after["updated_at"]
        return columns

    async def _read_back_thought(self, thought_id: str) -> ThoughtRecord:
        """Re-read a thought a write just landed on.

        The record handed back to the caller — and the ``after`` image written
        to the journal — must be the row that is actually stored, not the
        in-memory picture the operation intended to store. Only a read after
        the write can tell the two apart once updates are partial.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The thought as it is stored now.

        Raises:
            ThoughtNotFoundError: If the row no longer exists, so the write
                cannot be confirmed and no record may be reported for it.

        """
        row = await self._get_thought_row(thought_id)
        if row is None:
            raise ThoughtNotFoundError(thought_id)
        return self._row_to_thought(row)

    async def _get_thought_by_content_hash(
        self,
        content_hash: str,
    ) -> ThoughtRecord | None:
        """Return the first thought whose ``content_hash`` matches.

        Used by opt-in ingest deduplication to detect existing rows
        with identical content.  Pre-core-10 thoughts whose hash has
        not been backfilled have ``content_hash IS NULL`` and are
        therefore not eligible for deduplication.

        Args:
            content_hash: Lowercase hex SHA-256 digest of the candidate
                thought's content.

        Returns:
            The matching ``ThoughtRecord`` or ``None`` if no row
            matches.  When more than one row shares the hash (only
            possible for older data ingested before this fix
            was deployed) the first match in B-tree order is returned.

        """
        cursor = await self._db.execute(
            "SELECT * FROM thought WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_thought(row)

    async def _increment_confirmation(
        self,
        existing: ThoughtRecord,
    ) -> ThoughtRecord:
        """Bump ``confirmation_count`` + ``updated_at`` for an existing thought.

        **Write-lock classification: under the lock via callers, not its own
        body.** Called only from ``_create_thought_with_dedup`` and
        ``get_or_create``, both of which hold ``_write_lock`` for their whole
        dedup window.

        Implements the dedup-hit branch of ``create_thought``.  The bump is
        **relative** — ``confirmation_count = confirmation_count + 1``
        evaluated by SQLite against the stored row — so a confirmation
        recorded by another writer since ``existing`` was read is counted
        too, instead of being overwritten by an absolute value derived from
        a stale read.  ``updated_at`` is refreshed to the current UTC time.
        The row is read back so the returned record and the journal ``after``
        image carry the count that is actually stored.

        **Commits once, after the journal append, not before it.**
        ``self._journal.append()`` does not commit on its own (see its
        docstring: the caller is responsible, via ``_maybe_commit``) — writing
        the UPDATE, committing, and only then appending to the journal would
        leave that append's own INSERT permanently uncommitted whenever this
        call runs inside :meth:`_serialize_dedup_probe`'s window, since that
        guard takes no commit responsibility of its own (see its docstring).
        Ordering the write, the read-back and the journal append *before* the
        single ``_maybe_commit()`` call makes this method — the thing actually
        doing the writing — responsible for making all of it durable in one
        step, rather than depending on a caller several frames up to notice a
        write it cannot see.

        **The write, its confirming read-back, and its journal entry are one
        failure-atomic unit**, via :meth:`_write_readback_savepoint` — see
        :meth:`update_thought` for what that protects against. A failed or
        cancelled journal append here unwinds this call's own ``UPDATE`` too.

        Args:
            existing: The thought already in the database.

        Returns:
            The stored thought, with the incremented ``confirmation_count``
            and refreshed ``updated_at``.

        Raises:
            ThoughtNotFoundError: If the row disappeared between the
                content-hash probe and this bump, so no confirmation was
                recorded.

        """
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()

        async with self._write_readback_savepoint("increment_confirmation", begin="IMMEDIATE"):
            cursor = await self._db.execute(
                "UPDATE thought SET confirmation_count = confirmation_count + 1, "
                "updated_at = ? WHERE thought_id = ?",
                (now_iso, existing.thought_id),
            )
            if cursor.rowcount == 0:
                raise ThoughtNotFoundError(existing.thought_id)

            after = await self._read_back_thought(existing.thought_id)

            if self._journal is not None:
                await self._journal.append(
                    mutation_type="UPDATE_THOUGHT",
                    target_id=existing.thought_id,
                    delta={
                        "before": existing.model_dump(mode="json"),
                        "after": after.model_dump(mode="json"),
                    },
                )

        await self._maybe_commit()
        return after

    async def _begin_dedup_write_lock(self, *, operation: str) -> None:
        # Write-lock classification: under _write_lock via callers, not its
        # own body -- reached only from _serialize_dedup_probe, itself only
        # called from _create_thought_with_dedup / get_or_create /
        # upsert_by_hash, all three of which hold _write_lock for their whole
        # dedup window. (This method's own name coincidentally contains the
        # substring "_write_lock" -- it does not touch that attribute; it
        # implements the *separate*, cross-connection BEGIN IMMEDIATE lock.)
        """Open the dedup probe-and-insert window with ``BEGIN IMMEDIATE``.

        Issued in place of letting the probe's ``SELECT`` open an implicit
        *deferred* transaction, so the write lock is acquired **before** the
        probe runs rather than only when the eventual ``INSERT``/``UPDATE``
        needs it. A deferred transaction lets two connections both pass the
        probe unlocked and then discover the conflict only at the write —
        ``SQLITE_BUSY`` mid-transaction, the classic embedded-SQLite deadlock
        shape. Immediate mode surfaces contention at the boundary a caller can
        actually act on: before anything has been read for this call.

        ``PRAGMA busy_timeout`` already makes SQLite wait for the lock inside
        a single ``BEGIN IMMEDIATE`` call. This adds a bounded number of
        further attempts, with a short backoff between them, for contention
        that outlasts that wait — and converts the final failure into
        :class:`WriteContentionError` instead of leaking the raw
        :class:`sqlite3.OperationalError`. An ``OperationalError`` that is
        *not* busy/lock contention (e.g. a genuine I/O failure) is never
        retried and propagates unchanged — retrying could not fix it, and
        mislabelling it as contention would hide the real cause.

        Args:
            operation: Name of the calling public method, carried onto
                :class:`WriteContentionError` for a caller that logs or
                branches on it.

        Raises:
            WriteContentionError: The write lock could not be acquired after
                :data:`_DEDUP_BEGIN_MAX_ATTEMPTS` attempts.
            sqlite3.OperationalError: Some other, non-busy failure opening the
                transaction.

        """
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._db.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if not _is_busy_error(exc):
                    raise
                if attempt >= _DEDUP_BEGIN_MAX_ATTEMPTS:
                    raise WriteContentionError(operation=operation, attempts=attempt) from exc
                await asyncio.sleep(_DEDUP_BEGIN_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
            else:
                return

    @contextlib.asynccontextmanager
    async def _serialize_dedup_probe(self, *, operation: str) -> AsyncIterator[None]:
        # Write-lock classification: under _write_lock via callers, not its
        # own body -- see _begin_dedup_write_lock's note; the same three
        # callers hold _write_lock around this method's whole window too.
        """Take the cross-connection write lock for a probe-and-row window.

        Opens ``BEGIN IMMEDIATE`` (via :meth:`_begin_dedup_write_lock`) when
        ``not self._db.in_transaction`` — and *only* that; ``self._skip_auto_commit``
        (set by :meth:`suspend_auto_commit`) is not consulted here. This is
        what protects :meth:`bulk_store` too: its insert loop runs entirely
        inside one ``suspend_auto_commit`` window, so ``_skip_auto_commit`` is
        ``True`` for every row, but only the *first* row that reaches this
        method finds ``in_transaction`` still ``False`` — that row opens the
        lock up front, before its own hash probe runs, closing the same
        probe-then-insert race on the bulk path. Every later row in the same
        batch sees ``in_transaction`` already ``True`` and correctly takes no
        further lock of its own — it is already covered by the one still open.

        **This method commits nothing, on any path.** Inferring, at this
        guard's own exit, whether a transaction it saw open was still the
        one it started produced a new defect each time it was tried: a
        frozen flag went stale the moment a nested write committed early; a
        live ``in_transaction`` read could not tell this call's own leftover
        work apart from a different, genuinely concurrent task's; a per-call
        token protecting that read then made the guard skip a commit some
        *other* code still depended on it for.
        The common cause was not any one of those mechanisms — it was asking
        the guard to know something a shared connection does not expose:
        which open transaction is "ours". The fix is to stop asking. Whoever
        actually does the write — :meth:`_insert_new_thought_row`,
        :meth:`_increment_confirmation`, :meth:`update_thought` (via
        ``upsert_by_hash``'s matched-row path) — commits its own work, in the
        same call, after everything that call itself needs durable (including
        its own journal append — see each of those methods for why the
        ordering matters there). This method never decides that on their
        behalf. The no-op branch of ``upsert_by_hash`` writes nothing despite
        having opened this lock, and correspondingly commits nothing — but it
        does not leave the window open either: see
        :meth:`_end_exploratory_probe`, which that branch calls to roll back
        its own, still-empty transaction (never a caller's) before returning.

        **A rollback is kept, for the exception path only**, gated on having
        taken the lock, on ``self._db.in_transaction`` being ``True`` at that
        point, and on ``self._skip_auto_commit`` being ``False`` (inside a
        ``suspend_auto_commit`` window, that caller's own block owns the
        rollback decision, exactly as it owns the commit decision on the
        normal path). Cancellation is caught alongside every other exception
        (``except BaseException``, not ``except Exception``):
        :class:`asyncio.CancelledError` derives from ``BaseException``, and
        without catching it here a cancellation landing while this call still
        holds an open transaction would strand the RESERVED lock —
        ``in_transaction`` stuck ``True`` — blocking every other writer on the
        connection until something else eventually commits, rolls back, or
        closes it.

        **This rollback used to be able to discard a different, genuinely
        concurrent task's uncommitted write — closed in-process by an
        in-process task-reentrant write lock.** A connection has exactly one
        transaction, so while this call's is open, any other task's write on
        the same store instance used to join it rather than opening its own,
        and a rollback at that point took both down together, whether or not
        this call's own row
        ever committed first. No amount of bookkeeping inside this guard
        could change that, because the information needed to tell the two
        apart — whose write is whose — does not exist below the level of "is
        a transaction currently open on this connection". The fix is not
        bookkeeping inside this guard; it is that every caller of this method
        (:meth:`_create_thought_with_dedup`, :meth:`get_or_create`,
        :meth:`upsert_by_hash`) now holds :attr:`_write_lock` — the same
        task-reentrant lock every other guarded write path on this instance
        holds — for the whole probe-and-row window, so a different task's
        write can no longer even start until this window has closed. See
        ``docs/concurrency.md`` ("What is guaranteed" under "Many async
        tasks, one store") for the caller-facing statement of the fix, and
        its "Busy timeout" section for the "riding along" exposure this
        closed. **This closes it in-process only** — across *connections*
        the same exposure is unchanged (see
        ``docs/concurrency.md#multiple-stores-one-database-file``): that
        needs a transaction-level mechanism this store does not have yet.

        **Known residual gap, separate from the one above.** If a caller
        already holds an open transaction on this connection through means
        this store does not control — a raw ``BEGIN`` issued directly against
        it, with nothing written yet — ``self._db.in_transaction`` is already
        ``True`` when this method is entered, so no ``BEGIN IMMEDIATE`` is
        taken: SQLite refuses a ``BEGIN`` inside an already-open transaction,
        so there is no way to acquire the *cross-connection* write lock up
        front in that case. The window then falls back to whatever atomicity
        the caller's own transaction provides — it still gets the in-process
        :attr:`_write_lock` and ``_dedup_lock`` (the caller of this method
        holds both already), so a *different task on this instance* still
        cannot land inside the window; only the cross-connection ordering is
        unavailable here. This is not something ``bulk_store`` or
        ``suspend_auto_commit`` trigger (neither issues a bare ``BEGIN`` with
        no write): it can only happen through direct, unmediated use of the
        underlying connection, which is outside what this store manages.

        Args:
            operation: Forwarded to :meth:`_begin_dedup_write_lock` for
                :class:`WriteContentionError`.

        Yields:
            None — the probe-and-row body runs inside the method and is
            responsible for committing its own write, if any.

        """
        took_lock = not self._db.in_transaction
        try:
            if took_lock:
                await self._begin_dedup_write_lock(operation=operation)
            yield
        except BaseException:
            if self._connection_quarantined:
                # The dedup write inside this window (e.g. _insert_new_thought_row,
                # _increment_confirmation) already quarantined the connection
                # while unwinding its own unit and left `self._db` a terminal
                # proxy — touching it again here would replace the in-flight
                # exception with ConnectionQuarantinedError instead of letting
                # it propagate. See _write_readback_savepoint for the same guard.
                raise
            if took_lock and not self._skip_auto_commit and self._db.in_transaction:
                await self._db.rollback()
            raise

    async def _end_exploratory_probe(self, *, opened_transaction: bool) -> None:
        """Close a probe's own transaction without touching a caller's.

        Two callers reach this, and both leave a read-only ``BEGIN IMMEDIATE``
        that nothing else would close:

        * the miss branch of :meth:`get_or_create` / :meth:`upsert_by_hash`'s
          **exploratory** probe — the first of their two-phase miss path (see
          either method's docstring) — before the preparation seam runs; and
        * :meth:`upsert_by_hash`'s no-change branch, where a matched row whose
          mutable fields already equal the incoming record's writes nothing at
          all: no ``UPDATE``, no journal entry.

        Both call it from *inside* the ``async with self._write_lock,
        self._dedup_lock:`` block, so both locks are still held while this
        runs; they release only afterward, when that block exits following
        this call's return. What this closes is the cross-connection
        ``BEGIN IMMEDIATE`` :meth:`_serialize_dedup_probe` may have opened,
        which nothing else would ever close — those paths only read, so the
        "whoever writes, commits" rule that normally ends this window (see
        :meth:`_serialize_dedup_probe`) never fires on them.

        **Ownership test: `opened_transaction` alone.** It is computed by the
        caller as ``not self._db.in_transaction``, read *before* this probe's
        own window began — so when it is ``True``, no transaction of any
        kind, the caller's or anyone else's, was open at that point, and the
        ``BEGIN IMMEDIATE`` that may now be open can only be the one this
        probe itself opened. There is nothing ambiguous left to protect: an
        additional test on ``self._skip_auto_commit`` (whether this call
        happens to be nested inside a caller's own
        :meth:`suspend_auto_commit` window) used to gate the rollback here
        too, on the theory that "nested" implies "someone else's
        transaction". It does not — a caller's window can be open with
        *nothing yet written to it* (e.g. before its first guarded write), in
        which case ``opened_transaction`` is still ``True`` and the
        transaction is still this probe's own. Requiring
        ``not self._skip_auto_commit`` on top of that left this probe's
        read-only ``BEGIN IMMEDIATE`` — a cross-connection write reservation —
        open across the seam call in exactly that case: a second connection's
        own write blocks, and a same-call callback that writes can deadlock
        or time out on it. When ``opened_transaction`` is ``False`` instead
        (a transaction was already open before this probe began — e.g. this
        call is itself nested inside the *same* task's own
        :meth:`suspend_auto_commit` window, where an earlier guarded write
        already opened one), the still-open transaction is
        unambiguously the *caller's*: rolling it back would discard writes
        this call knows nothing about, and committing it early would end the
        batch's atomicity mid-flight — that case is excluded by
        ``opened_transaction`` on its own, with no need for a second test.

        Always a rollback, never a commit, when it does act: the probe only
        read, so there is nothing of its own to make durable — rollback is
        also what :meth:`_serialize_dedup_probe`'s own exception path uses to
        end an aborted probe.

        Args:
            opened_transaction: Whether *this* call's own window opened the
                ``BEGIN IMMEDIATE`` that may still be open. Computed by the
                caller as ``not self._db.in_transaction``, read at the same
                point in the call — immediately after acquiring
                ``_write_lock`` and ``_dedup_lock``, before entering
                :meth:`_serialize_dedup_probe` — that method reads it for its
                own ``took_lock``, so the two can never disagree.

        """
        if opened_transaction and self._db.in_transaction:
            await self._db.rollback()

    async def prepare_thought_for_insert(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Override point: prepare or validate a candidate before it is stored.

        **The pre-insert seam**: override this in a subclass to run
        validation or persisted enrichment on every candidate that might
        become a new row, before this store commits to inserting it or treats
        it as a duplicate sighting. The default implementation is a
        pass-through that returns ``thought`` unchanged.

        This exists because ``on_store`` cannot fill this role — it runs
        *after* the row is (usually) durable, so a rejection there cannot
        stop the write, and an enrichment there reaches the caller's returned
        record but never the stored one (see ``docs/upgrade.md``, "0.6 -> 0.7",
        for the measured failure modes). Overriding ``create_thought`` itself
        used to be the only way to run code before every insert; it silently
        stopped covering ``get_or_create`` / ``upsert_by_hash`` once their miss
        branch was rewritten to call the internal insertion step directly
        instead of the public, overridable method. This seam is the
        restored, and now uniform, replacement for that override point.

        **Invocation count — pinned per entry point, not "once per row":**

        * :meth:`create_thought` — exactly once per call, *before* the dedup
          branch, unconditionally: whether the call goes on to insert, to bump
          an existing row's ``confirmation_count`` (``deduplicate=True`` hit),
          or (``deduplicate=False``) always inserts. A direct call has no way
          to know in advance which of those it will be, so — matching what a
          subclass override of the whole method could always do before this
          seam existed — it always runs, exactly once.
        * :meth:`get_or_create` / :meth:`upsert_by_hash` — **zero** times on a
          stable, pre-existing hit (their exploratory probe resolves the call
          before this method is ever reached — an idempotent "ensure it
          exists" call that turns out to be a hit costs nothing extra), and
          exactly **once** after an observed miss. That includes the race
          where a second writer wins between the exploratory probe and the
          mandatory decisive re-probe, turning the miss into a hit: the seam
          already ran once for this call and does not run again just because
          the outcome changed underneath it.
        * :meth:`bulk_store` — once per item, run for the *whole batch*,
          holding no lock this call itself acquires, before ``bulk_store``
          takes any lock for its insert
          transaction — not "through" a per-item :meth:`create_thought` call
          the way the other counts might suggest. A dedup hit inside the
          batch still costs one seam call, the same "still runs once" rule
          a direct call follows.
        * :meth:`remember` — once, through its own :meth:`create_thought` call.

        **Never one this call itself acquires — on any entry point — but not
        "never any lock at all".** Neither ``_dedup_lock`` nor ``_write_lock``
        nor a ``BEGIN IMMEDIATE`` this call's own dedup machinery opened is
        held while this method is awaited. :meth:`create_thought` calls this
        before taking any lock at all; :meth:`get_or_create` /
        :meth:`upsert_by_hash` release ``_write_lock``, ``_dedup_lock``, and
        (via :meth:`_end_exploratory_probe`) their own ``BEGIN IMMEDIATE``
        *before* calling this, and only reacquire everything afterward, from
        scratch, for the decisive probe. This is deliberate: restoring a call
        to the public ``create_thought()`` from *inside* an already-locked
        dedup window (a naive, rejected shape for this fix) runs auto-embed,
        cleanup, ``on_store`` and derivation while still holding
        ``_dedup_lock`` — a plain lock with no legitimate reentrant use — so a
        hook that calls back into a dedup entry point on the *same task*
        raises :class:`~engrava.domain.exceptions.DedupLockReentryError`
        rather than blocking on it. Because this seam never runs with
        ``_dedup_lock`` held, on any entry point, regardless of nesting, a
        recursive call *from inside this method* never contends for it in the
        first place — the same lock's reentry guard exists for a different
        call site entirely: ``upsert_by_hash``'s hit branch, which never
        reaches this method at all (see ``docs/extension-hooks.md`` §1B.3),
        calls the separately overridable ``update_thought`` while still
        holding ``_dedup_lock``, and it is *that* call, if overridden to
        recurse on the same task, the reentry guard protects.

        **That is a statement about locks this call itself takes, not about
        every lock that can be held while it runs.** If you call
        :meth:`create_thought` / :meth:`get_or_create` / :meth:`upsert_by_hash`
        / :meth:`bulk_store` from *inside your own*
        :meth:`suspend_auto_commit` window, that window's ``_write_lock`` is
        still held while this seam runs underneath it — it is *your* lock,
        acquired before you ever reached this call, not one this call took
        for itself. A plain in-process ``asyncio.Lock`` cannot distinguish
        "the caller's own reentrant hold" from "a lock this call should
        release", so a pre-insert override cannot close this on its own; it
        is not new either — it applies identically to a raw
        :meth:`create_thought` call inside your own
        :meth:`suspend_auto_commit` block, seam or no seam.

        **One residual exposure this does not close, because it is out of
        this seam's scope: a *different, spawned* task calling back in from
        inside a caller's own** :meth:`suspend_auto_commit` **window** (for
        example, from inside an override invoked by ``bulk_store``'s insert
        loop). That window's pre-existing contract already requires every
        guarded write on the instance to come from the one task that opened
        it; a spawned task's write times out on ``_write_lock`` there exactly
        as it would for any other code violating that same contract — not a
        new gap this seam introduces, and not one a pre-insert hook could
        close without breaking that contract's own atomicity guarantee.

        **The returned record is revalidated, and its content is decisive.**
        The caller re-runs metadata and provenance validation on whatever this
        method returns (not just the original candidate) — so an override that
        introduces invalid ``metadata`` or ``provenance`` makes the call raise
        here, before the decisive probe or any row write (for
        :meth:`get_or_create` / :meth:`upsert_by_hash`, this revalidation runs
        *after* their own exploratory probe already came up empty, not before
        any probe at all). For :meth:`get_or_create`
        and :meth:`upsert_by_hash`, the *returned* record's ``content`` — not
        the original candidate's — supplies the hash for the decisive probe
        that follows, so an override that changes ``content`` changes what
        counts as a duplicate for this call.

        **Raising aborts the create.** No row is inserted, and — on every path
        this seam covers — no journal entry is appended either: a rejection
        here leaves no trace.

        **Out of the ``revision`` contract, on the path where it matters
        most.** This seam postdates the revision guard's design and is not
        part of it. On :meth:`bulk_store`'s path in particular, this call runs
        for the whole batch *before* ``bulk_store`` takes its own write lock
        (see the invocation-count note above) — so a subclass override that
        persists a write of its own here does so **outside** ``_write_lock``
        and outside any ``revision`` bookkeeping. Such a write is a
        third-party mutation this store neither guards nor bumps; treat it as
        unmediated use of the connection, in the same sense the concurrency
        documentation uses that phrase for a caller's own raw transaction.

        Args:
            thought: The candidate record, already validated (metadata,
                provenance) in its pre-seam form — on every entry point
                *except* :meth:`bulk_store`, which runs this call, batch-wide,
                before its own per-item validation (see
                :meth:`_bulk_store_inner`'s docstring for why); a
                ``bulk_store`` override sees whatever was passed to it,
                unvalidated. Its ``created_at`` / ``updated_at`` /
                ``expires_at`` are not yet resolved — that happens
                afterward, in :meth:`_insert_new_thought_row`, only if this
                call ends up inserting.

        Returns:
            The record to use from here on for the probe and (on a miss) the
            insert. The default implementation returns ``thought`` unchanged.

        """
        return thought

    async def _create_thought_with_dedup(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None,
    ) -> ThoughtRecord:
        """Lock-protected dedup branch of ``create_thought``.

        Called only after the pre-insert seam has already run on ``thought``
        — via :meth:`_prepare_for_create` for a direct :meth:`create_thought`
        call, or via :meth:`_bulk_store_inner`'s own batch-wide phase 1 for a
        ``bulk_store(deduplicate=True)`` item reaching this through
        :meth:`_create_thought_from_prepared` — unlike :meth:`get_or_create` /
        :meth:`upsert_by_hash`, this branch takes no two-phase detour of its
        own: the seam already ran, exactly once, before this method was ever
        entered, whichever of the hit/miss branches below it resolves to.

        Acquires ``self._dedup_lock`` for the entire ``check existing
        → INSERT or UPDATE`` window so concurrent calls **on this store
        instance** with identical ``content`` never race past the existence
        probe, and additionally serialises that same window across
        *connections* with :meth:`_serialize_dedup_probe` (``BEGIN
        IMMEDIATE``), so a second store on the same database file can no
        longer insert between this call's probe and its insert — see that
        method's docstring for its one remaining residual gap (a caller
        holding a raw, unmediated transaction on the connection). The other
        gap that docstring documents — a genuinely concurrent, unrelated
        task's write riding along with this call's own commit — is closed
        here by :attr:`_write_lock` (below). On a miss, the row is inserted
        directly via
        :meth:`_insert_new_thought_row` (not by recursing into
        ``create_thought``) so only the probe and the insert run inside the
        window; when the content has been seen, ``confirmation_count`` is
        bumped and the existing record returned without any additional
        INSERT.

        **The write lock is released, and the row made durable, before
        auto-embed / ``on_store`` run — unless this call is itself nested in
        the caller's own** :meth:`suspend_auto_commit` **window.** On a miss,
        only the probe and the row insert happen inside
        :meth:`_serialize_dedup_probe`'s window; :meth:`_insert_new_thought_row`
        calls :meth:`_maybe_commit`, which — at the top level — ends the
        ``BEGIN IMMEDIATE`` transaction there and then, so
        :meth:`_finish_create_thought` (auto-embed, hygiene cleanup,
        ``on_store``, derivation dispatch) runs *after* the ``async with``
        block has already exited, with the row already durable and no
        transaction or write lock left open — so a slow or hanging embedding
        call cannot stall every other writer on the file. Nested inside the
        caller's own :meth:`suspend_auto_commit` window, :meth:`_maybe_commit`
        is a no-op by design (the caller owns the commit boundary), so the
        row is *not yet durable* and :meth:`_finish_create_thought` runs
        while the outer transaction — and, below, the outer ``_write_lock``
        hold — are still open: an ``on_store`` override reached this way must
        not assume the row survives a crash yet, and a slow follow-up step
        can block other writers for that whole extra span (see
        ``docs/concurrency.md``'s nesting note for the general rule this is
        an instance of).

        **Also under the in-process task-reentrant write lock, at the top
        level:** the probe-then-insert/bump window is held under
        :attr:`_write_lock` in
        addition to ``_dedup_lock`` — the latter blocks a *second* call to one
        of the three dedup methods, the former now blocks *every* other
        guarded write on this instance, closing the gap
        :meth:`_serialize_dedup_probe` documents ("a genuinely concurrent,
        unrelated task's write riding along with this call's own commit"): that
        other task's write cannot even start until this window's lock is
        released, so it can no longer join this window's transaction. Released
        at the same point as before — before :meth:`_finish_create_thought` —
        except when this call is nested in the caller's own
        :meth:`suspend_auto_commit` window, where ``_write_lock`` is
        task-reentrant and the outer hold never actually drops, so it stays
        held through :meth:`_finish_create_thought` too.
        """
        async with (
            self._write_lock,
            self._dedup_lock,
            self._serialize_dedup_probe(operation="create_thought"),
        ):
            existing = await self._get_thought_by_content_hash(
                _compute_content_hash(thought.content),
            )
            if existing is not None:
                return await self._increment_confirmation(existing)
            persisted = await self._insert_new_thought_row(
                thought,
                expires_after_seconds=expires_after_seconds,
            )
        return await self._finish_create_thought(persisted)

    async def create_thought(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
        deduplicate: bool = False,
    ) -> ThoughtRecord:
        # Write-lock classification: under _write_lock via callees, not its
        # own body -- the dedup branch delegates to _create_thought_with_dedup
        # (holds it) and the plain-insert branch to _insert_new_thought_row
        # (holds it); this method does no SQL of its own.
        """Persist a new thought record.

        Automatically sets ``created_at`` and ``updated_at`` to the
        current UTC time when they are ``None``.  When
        ``expires_after_seconds`` is given (or a default TTL is
        configured), ``expires_at`` is computed as an absolute ISO-8601
        timestamp.

        Content-hash deduplication is opt-in via ``deduplicate=True``:
        when an existing thought with the same SHA-256 hash of
        ``content`` is found, its ``confirmation_count`` is incremented
        and the existing record is returned instead of creating a
        duplicate.  The default ``deduplicate=False`` preserves the
        legacy create-on-every-call behaviour.

        Args:
            thought: The thought record to create.
            expires_after_seconds: Optional relative TTL in seconds.
                Overrides the store-level default when provided.
            deduplicate: When True, check for an existing thought with
                identical ``content`` (via the indexed SHA-256
                ``content_hash`` column).  If a match is found, that
                thought's ``confirmation_count`` is incremented, its
                ``updated_at`` is refreshed, and the updated record is
                returned without inserting a new row.  When False
                (default) a new thought is always inserted, exactly
                matching the pre-deduplication behaviour.

        Returns:
            The persisted thought record (with timestamps populated).
            When ``deduplicate=True`` collides with an existing thought,
            returns the existing record with bumped
            ``confirmation_count`` and ``updated_at``.

        Provenance capture (opt-in, untrusted hint):
            When ``thought.provenance`` is set, its typed, bounded fields are
            persisted alongside the thought and journaled with it — enabling
            provenance *stores* these values (there is no silent capture).
            **Provenance is an untrusted hint — never identity, authentication,
            or authorization.** The engine grants it zero authority: it is
            captured verbatim and consulted for no access, ranking, or
            consolidation decision.  ``actor_id`` is *not* a tenant boundary
            (tenant isolation is the store's file boundary), and the engine
            never infers provenance — the caller passes it explicitly.  A
            thought with ``provenance=None`` (the default) writes a NULL column
            and is byte-identical to a pre-feature row.

        Raises:
            ValueError: If a thought with the same ID already exists, if
                ``thought.metadata`` violates the metadata-shape or size
                invariants enforced by :func:`_validate_metadata`, or if
                ``thought.provenance`` is not a
                :class:`~engrava.domain.models.provenance.ProvenanceContext`
                (per :func:`_validate_provenance`).
            ThoughtNotFoundError: When ``deduplicate=True`` matched an existing
                thought that was then deleted before its ``confirmation_count``
                could be bumped — the sighting was not recorded, so no record
                is returned for it.
            WriteContentionError: When ``deduplicate=True`` and the
                cross-connection write lock could not be acquired after
                retrying (see :meth:`_begin_dedup_write_lock`).
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        # Pre-insert seam, then insert/dedup -- split into two helpers so
        # bulk_store can run every item's seam call holding no lock this
        # call itself acquires, for the whole batch, before taking
        # _write_lock for its one insert transaction
        # (see _prepare_for_create / _create_thought_from_prepared). A direct
        # call here just runs both halves back to back, in the same order as
        # before that split.
        thought = await self._prepare_for_create(thought)
        return await self._create_thought_from_prepared(
            thought,
            expires_after_seconds=expires_after_seconds,
            deduplicate=deduplicate,
        )

    async def _prepare_for_create(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Pre-seam validation, the seam itself, and its revalidation.

        The first of :meth:`create_thought`'s two halves (see
        :meth:`_create_thought_from_prepared` for the second): validates the
        candidate, runs :meth:`prepare_thought_for_insert` on it — exactly
        once per call, before the dedup branch, unconditionally, since a
        direct call cannot know in advance whether it will hit or miss — and
        revalidates whatever the seam returns. **Not called by**
        :meth:`_bulk_store_inner`: that method calls
        :meth:`prepare_thought_for_insert` directly, batch-wide, holding no
        lock this call itself acquires, and runs ordinary validation
        separately, per item, in its own insert phase — see its docstring for
        why bundling this helper's validate-seam-revalidate sequence across
        the whole batch would let a later item's validation failure pre-empt
        an earlier item's own outcome. A direct :meth:`create_thought` call
        runs this helper exactly as before that split existed. See
        :meth:`prepare_thought_for_insert`'s docstring for the full
        per-entry-point invocation-count contract.

        Args:
            thought: The candidate record to validate and run through the
                seam.

        Returns:
            The prepared, revalidated record — ready for
            :meth:`_create_thought_from_prepared`.

        """
        self._ensure_connection_usable()
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)

        thought = await self.prepare_thought_for_insert(thought)
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)
        return thought

    async def _create_thought_from_prepared(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None,
        deduplicate: bool,
    ) -> ThoughtRecord:
        """Insert (or dedup-resolve) a record the seam has already run on.

        The second of :meth:`create_thought`'s two halves (see
        :meth:`_prepare_for_create` for the first). Must never call
        :meth:`prepare_thought_for_insert` itself: :meth:`_bulk_store_inner`
        also calls this method directly, per item, once its own per-item
        validation has run — not through :meth:`_prepare_for_create`, which
        it never calls (see that method's docstring); the seam already ran,
        holding no lock this call itself acquires, over the whole batch in
        :meth:`_bulk_store_inner`'s own
        phase 1. Either caller relies on this method not running the seam a
        second time for an already-prepared item.

        Args:
            thought: An already-prepared, already-(re)validated record — for
                a direct :meth:`create_thought` call, the value
                :meth:`_prepare_for_create` returned; for a
                :meth:`_bulk_store_inner` item, the seam's output, validated
                inline by that method's own per-item loop instead.
            expires_after_seconds: Forwarded to :meth:`_insert_new_thought_row`.
            deduplicate: Forwarded to :meth:`_create_thought_with_dedup`.

        Returns:
            The persisted record, or the existing record on a dedup hit.

        """
        if deduplicate:
            return await self._create_thought_with_dedup(
                thought,
                expires_after_seconds=expires_after_seconds,
            )

        persisted = await self._insert_new_thought_row(
            thought,
            expires_after_seconds=expires_after_seconds,
        )
        return await self._finish_create_thought(persisted)

    async def _insert_new_thought_row(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None,
    ) -> ThoughtRecord:
        """Insert-only half of ``create_thought``: row, journal, commit.

        Split out of ``create_thought`` so the dedup entry points
        (:meth:`_create_thought_with_dedup`, :meth:`get_or_create`,
        :meth:`upsert_by_hash`) can run **only this** inside
        :meth:`_serialize_dedup_probe`'s ``BEGIN IMMEDIATE`` window, and run
        :meth:`_finish_create_thought` (auto-embed, cleanup, ``on_store``,
        derivation) afterwards, outside it. This method's own
        :meth:`_maybe_commit` call is what ends that window on the normal
        path — deliberately before any of the slower, I/O-bound follow-up work
        the split keeps out of it. The journal append runs *before* that
        commit, not after: :meth:`_serialize_dedup_probe` takes no commit
        responsibility of its own, so whoever writes here is the one that
        must make the journal entry durable too.

        The existence check, the insert, and the commit run under
        :attr:`_write_lock`: a different task's guarded write
        cannot land between the existence probe and the insert, or between the
        insert and its commit.

        **The insert and its journal entry are one failure-atomic unit**, via
        :meth:`_write_readback_savepoint` — see :meth:`update_thought` for what
        that protects against. A failed or cancelled journal append here
        unwinds the insert too, rather than leaving it pending for a later,
        unrelated commit to publish without the journal entry that documents
        it.

        Args:
            thought: The thought record to create.
            expires_after_seconds: Optional relative TTL in seconds; overrides
                the store-level default when given.

        Returns:
            The persisted thought record, with timestamps and expiry resolved
            — the same value ``create_thought`` used to pass into what is now
            :meth:`_finish_create_thought`.

        Raises:
            ValueError: If a thought with the same ID already exists.

        """
        self._ensure_connection_usable()
        async with self._write_lock:
            existing_row = await self._get_thought_row(thought.thought_id)
            if existing_row is not None:
                msg = f"Thought already exists: {thought.thought_id}"
                raise ValueError(msg)

            now = datetime.datetime.now(datetime.UTC)
            now_iso = now.isoformat()
            updates: dict[str, object] = {}
            if thought.created_at is None:
                updates["created_at"] = now_iso
            if thought.updated_at is None:
                updates["updated_at"] = now_iso

            # Resolve expiry: explicit param > thought field > store default.
            if expires_after_seconds is not None:
                updates["expires_at"] = (
                    now + datetime.timedelta(seconds=expires_after_seconds)
                ).isoformat()
            elif thought.expires_at is None and self._ttl_default_seconds is not None:
                updates["expires_at"] = (
                    now + datetime.timedelta(seconds=self._ttl_default_seconds)
                ).isoformat()

            if updates:
                thought = type(thought).model_validate(
                    {**thought.model_dump(), **updates},
                )

            async with self._write_readback_savepoint("insert_new_thought_row", begin="IMMEDIATE"):
                await self._db.execute(self._CORE_INSERT_SQL, self._thought_to_core_params(thought))

                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="INSERT_THOUGHT",
                        target_id=thought.thought_id,
                        delta={"before": None, "after": thought.model_dump(mode="json")},
                    )

            await self._maybe_commit()
        return thought

    async def _finish_create_thought(self, thought: ThoughtRecord) -> ThoughtRecord:
        """Follow-up half of ``create_thought``: auto-embed through derivation.

        Runs everything ``create_thought`` does *after* the row is inserted:
        auto-embed, hygiene cleanup, the ``on_store`` hook and derived-record
        dispatch. Split out so the dedup entry points can call it once
        :meth:`_serialize_dedup_probe`'s window has already closed — see
        :meth:`_insert_new_thought_row`.

        **"Post-commit" only at the top level.** :meth:`_insert_new_thought_row`
        calls :meth:`_maybe_commit`, which actually commits — making
        ``thought`` durable before this method ever runs — only when this
        call is not itself nested inside the caller's own
        :meth:`suspend_auto_commit` window. Nested, ``_maybe_commit`` is a
        no-op by design (the caller owns the commit boundary), so this
        method runs on a row that is inserted but not yet durable, inside
        the outer transaction, and — because ``_write_lock`` is
        task-reentrant — still under the outer window's hold of it too. Two
        consequences follow for an override reached from here: an
        ``on_store`` implementation must not assume the row survives a
        crash yet in that case, and a slow or hanging step (an embedding
        provider call, a derived-records producer) can block a *different*
        task's guarded write for this whole extra span, not just for the
        insert itself. :meth:`bulk_store` is a second, by-design instance of
        the same shape: it calls this method from *inside* its own batch
        transaction deliberately, so its auto-embed and derivation dispatch
        also run before that batch commits (see its own docstring). See
        ``docs/concurrency.md``'s nesting note for the general rule this is
        one instance of.

        Args:
            thought: The just-inserted, but not necessarily yet durable,
                thought record, as returned by :meth:`_insert_new_thought_row`.

        Returns:
            The enriched record returned by the ``on_store`` hook.

        """
        # Auto-embed when a provider is configured and auto_embed is on.
        # ``_suppress_auto_embed`` lets ``bulk_store`` defer embedding to a
        # single batch call after the insert loop without changing this path
        # for any other caller (the flag is False in every non-bulk call).
        if (
            self._auto_embed
            and self._embedding_provider is not None
            and not self._suppress_auto_embed
        ):
            await self._auto_embed_thought(thought)

        await self._maybe_auto_cleanup(exclude_id=thought.thought_id)
        enriched = await self._hooks.on_store(thought)
        # Derived-records seam. Dispatched inline only after the source's own
        # commit and ``on_store`` have completed; the committed source
        # (``thought``, the input to ``on_store`` — not its possibly-different
        # return value) is what the producer derives from.
        # ``_dispatch_derivation`` returns early (after at most a cheap
        # enabled/capability check) unless the seam is enabled, the source's own
        # commit actually happened (it does not dispatch inside a suspended-commit
        # window — ``bulk_store`` dispatches after its batch commits), and the
        # hooks object is a producer; so the disabled/absent path above yields
        # byte-identical persisted results (DB + journal). If ``on_store`` raised,
        # we never reach here and derivation does not run.
        await self._dispatch_derivation(thought)
        return enriched

    async def get_or_create(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
    ) -> tuple[ThoughtRecord, bool]:
        """Fetch an existing thought by content hash, or create it.

        A thin convenience over the existing content-hash deduplication that
        removes the check-then-create round trip (and its TOCTOU window)
        callers otherwise write by hand. That window is closed against other
        callers of **this store instance** by the in-process deduplication
        lock, and additionally against a second store on the same database
        file by ``BEGIN IMMEDIATE`` around the probe and the row insert (see
        :meth:`_serialize_dedup_probe`) — see that method's docstring for its
        one remaining residual gap (a caller holding a raw, unmediated
        transaction on this connection). The other gap that docstring
        documents — a genuinely concurrent, unrelated task's write riding
        along with this call's own commit — cannot happen here either: this
        call holds ``_write_lock`` for the same probe-and-row window (see
        :meth:`_create_thought_with_dedup`'s docstring for why that closes
        it). At the top level, the write lock is released as
        soon as the row is written, before auto-embed or the ``on_store`` hook
        run; nested in the caller's own :meth:`suspend_auto_commit` window,
        that hold stays through the follow-up work instead — see
        :meth:`_insert_new_thought_row` / :meth:`_finish_create_thought`'s
        docstring for both cases.
        The content hash is the same byte-exact SHA-256 of ``content`` used by
        ``create_thought(deduplicate=True)``:

        * **Hit** — a thought with that hash already exists: it is returned
          with ``created=False``. No new row is inserted; its
          ``confirmation_count`` is bumped and ``updated_at`` refreshed,
          identical to ``create_thought(deduplicate=True)`` so the two APIs
          stay consistent. **The pre-insert seam
          (:meth:`prepare_thought_for_insert`) does not run on this branch**
          — a stable hit costs zero seam calls.
        * **Miss** — no such thought exists: a **two-phase** path runs from
          here. The exploratory probe above found nothing, so this call
          releases ``_write_lock``, ``_dedup_lock`` and any ``BEGIN IMMEDIATE``
          it opened for that probe, then calls
          :meth:`prepare_thought_for_insert` — holding no lock this call
          itself acquires (a lock an *enclosing* caller took is a different
          matter; see :meth:`prepare_thought_for_insert`'s own docstring) —
          and
          revalidates what it returns. It then **reacquires** everything for a
          **decisive** probe keyed on the *prepared* record's content: if that
          still misses, the row is inserted (running the regular journal /
          auto-embed / cleanup pipeline) and returned with ``created=True``;
          if a second writer won the race in between, this call takes the hit
          branch instead — the seam still ran exactly once. See
          :meth:`prepare_thought_for_insert` for why the seam runs holding no
          lock this call itself acquires, on either probe.

        The returned boolean is the value ``create_thought(deduplicate=True)``
        cannot give back: it tells the caller *whether it created*, so an
        idempotent "ensure this thought exists" call needs no follow-up query.

        This does not modify the matched row's mutable fields from ``thought``
        (metadata, priority, essence): a hit returns the stored record
        unchanged apart from the confirmation bump. Use :meth:`upsert_by_hash`
        when a match should adopt the incoming record's fields.

        Args:
            thought: The candidate thought. On a miss it is passed through
                :meth:`prepare_thought_for_insert` and inserted verbatim (with
                timestamps populated); on a hit only its ``content`` was used,
                via the hash.
            expires_after_seconds: Optional relative TTL applied only when the
                thought is created (a hit never re-arms TTL). Mirrors
                ``create_thought``'s parameter.

        Returns:
            A ``(record, created)`` tuple. ``created`` is ``True`` when a new
            row was inserted, ``False`` when an existing thought was returned.

        Raises:
            ValueError: If ``thought.metadata`` violates the metadata-shape or
                size invariants (validated up front on both the hit and miss
                paths, matching ``create_thought(deduplicate=True)`` which
                validates before it branches) — or if
                :meth:`prepare_thought_for_insert` returns a record that
                fails that same revalidation on a miss.
            ThoughtNotFoundError: If a matched thought was deleted before its
                ``confirmation_count`` could be bumped — the sighting was not
                recorded, so no record is returned for it.
            WriteContentionError: The cross-connection write lock could not be
                acquired after retrying (see :meth:`_begin_dedup_write_lock`).
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        # Validate up front — before the hash probe — so an invalid-metadata
        # candidate raises on a hit too, exactly as ``create_thought`` does
        # (it validates at the top, ahead of the dedup branch). ``create_thought``
        # re-validates on the miss/insert path; that is cheap and harmless.
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)

        # Phase 1 -- exploratory probe: a stable hit resolves right here, at
        # zero seam calls, and never reaches phase 2 below.
        async with self._write_lock, self._dedup_lock:
            opened_transaction = not self._db.in_transaction
            async with self._serialize_dedup_probe(operation="get_or_create"):
                existing = await self._get_thought_by_content_hash(
                    _compute_content_hash(thought.content),
                )
                if existing is not None:
                    return await self._increment_confirmation(existing), False
                await self._end_exploratory_probe(opened_transaction=opened_transaction)
        # _write_lock, _dedup_lock and any BEGIN IMMEDIATE this call's own
        # probe opened are all released above -- the seam below runs holding
        # no lock this call itself acquires (an enclosing caller's own lock
        # is a different matter -- see prepare_thought_for_insert's docstring).

        thought = await self.prepare_thought_for_insert(thought)
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)

        # Phase 2 -- decisive probe: reacquired from scratch, keyed on the
        # prepared record's content. Insert only if this still misses.
        async with (
            self._write_lock,
            self._dedup_lock,
            self._serialize_dedup_probe(operation="get_or_create"),
        ):
            existing = await self._get_thought_by_content_hash(
                _compute_content_hash(thought.content),
            )
            if existing is not None:
                # Another writer won the race between the two probes -- the
                # seam already ran once for this call; take the hit branch
                # instead of inserting.
                return await self._increment_confirmation(existing), False
            persisted = await self._insert_new_thought_row(
                thought,
                expires_after_seconds=expires_after_seconds,
            )
        # Write lock released above, at the top level; nested in the
        # caller's own suspend_auto_commit() window, the outer hold stays
        # through auto-embed / on_store / derivation below too -- see
        # _finish_create_thought's docstring for the durability and
        # lock-hold implications of that case.
        origin_token = _DERIVATION_ORIGIN.set("get_or_create")
        try:
            created = await self._finish_create_thought(persisted)
        finally:
            _DERIVATION_ORIGIN.reset(origin_token)
        return created, True

    #: Mutable ``ThoughtRecord`` fields a content-hash upsert copies from the
    #: incoming record onto a matched row. ``content`` is deliberately excluded
    #: — it is the hash key, so a match already has byte-identical content —
    #: as are identity/system-managed fields (ids, cycles, timestamps,
    #: ``confirmation_count``, ``access_count``, valid-time bounds).
    _UPSERT_MUTABLE_FIELDS = (
        "thought_type",
        "essence",
        "priority",
        "lifecycle_status",
        "source",
        "confidence",
        "source_type",
        "visibility",
        "metadata",
    )

    async def upsert_by_hash(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
    ) -> ThoughtRecord:
        """Insert a thought, or update the matching row's mutable fields in place.

        A content-hash upsert with genuine **update-on-match** semantics,
        deliberately distinct from ``create_thought(deduplicate=True)``:

        * ``create_thought(deduplicate=True)`` treats a hash hit as a *sighting*
          of already-known content — it returns the stored record unchanged
          apart from bumping ``confirmation_count`` (and ``updated_at``), and
          discards the incoming record's other fields.
        * ``upsert_by_hash`` treats a hash hit as a *newer version of the same
          content* — it overwrites the stored row's mutable fields
          (``essence``, ``priority``, ``metadata``, ``visibility``,
          ``lifecycle_status``, ``source``, ``confidence``, ``source_type``,
          ``thought_type``) from ``thought`` and returns the updated record. It
          does **not** bump ``confirmation_count``: the call expresses "make the
          stored thought look like this", not "I saw this again".

        ``content`` itself is never written on the match branch — it is the hash
        key, so a match already has byte-identical content. Identity and
        system-managed fields (ids, cycles, timestamps, ``access_count``,
        valid-time bounds) are also preserved. Only fields that *differ* from
        the stored row are written, so an upsert whose mutable fields already
        match the stored thought is a no-op that returns it untouched (and, in
        particular, an unchanged ``lifecycle_status`` is never re-asserted,
        which would otherwise be rejected as a same-state transition). The
        update reuses :meth:`update_thought`, so it carries the same
        ``updated_cycle`` guard — and the same limits on what that guard catches
        — and re-embeds when ``essence`` changed, exactly like any other edit.
        A miss runs the same **two-phase** path as :meth:`get_or_create`: the
        exploratory probe above found nothing, so this call releases
        ``_write_lock``, ``_dedup_lock`` and any ``BEGIN IMMEDIATE`` it opened,
        calls :meth:`prepare_thought_for_insert` holding no lock this call
        itself acquires,
        revalidates what it returns, then reacquires everything for a
        decisive probe keyed on the *prepared* record's content. If that
        still misses, the row is inserted (regular journal pipeline, via
        :meth:`_insert_new_thought_row`) and auto-embed / ``on_store`` /
        derivation run afterwards — at the top level, once the write lock is
        released; nested in the caller's own :meth:`suspend_auto_commit`
        window, while that outer hold stays in place instead (see
        :meth:`_finish_create_thought`'s docstring for both cases); if a
        second writer won the race in
        between, this call takes the hit (update-on-match) branch instead —
        the seam still ran exactly once. **A stable hit costs zero seam
        calls**, exactly like :meth:`get_or_create`. See
        :meth:`prepare_thought_for_insert` for why the seam runs holding no
        lock this call itself acquires, on either probe.

        Choose :meth:`get_or_create` for "ensure it exists, don't touch it if it
        does"; choose ``upsert_by_hash`` for "make the stored thought match this
        record"; choose ``create_thought(deduplicate=True)`` for
        confirmation-counting of repeated sightings.

        Args:
            thought: The desired thought state. On a miss it is passed through
                :meth:`prepare_thought_for_insert` and inserted verbatim; on a
                hit its mutable fields are copied onto the existing row (keyed
                by ``content``).
            expires_after_seconds: Optional relative TTL applied only when the
                thought is created. A hit does not re-arm TTL (``expires_at`` is
                a system-managed field left untouched), matching
                :meth:`get_or_create`.

        Returns:
            The persisted thought record: the freshly inserted row on a miss, or
            the updated existing row on a hit.

        Raises:
            ValueError: If ``thought.metadata`` violates the metadata-shape or
                size invariants (validated up front on both the hit and miss
                paths) — or if :meth:`prepare_thought_for_insert` returns a
                record that fails that same revalidation on a miss.
            StaleDataError: If the matched row's guarded update writes no row —
                another writer stamped a new ``updated_cycle`` between the hash
                probe and the in-place update, or deleted the row. The probe and
                the update are **not** one atomic step, and the guard is
                :meth:`update_thought`'s: an ordinary competing edit to a field
                this upsert also writes is overwritten silently rather than
                reported here.
            WriteContentionError: The cross-connection write lock could not be
                acquired after retrying (see :meth:`_begin_dedup_write_lock`).
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        # Validate up front so an invalid-metadata candidate raises consistently
        # on both the hit (update) and miss (insert) branches.
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)

        # Phase 1 -- exploratory probe: a stable hit resolves right here, at
        # zero seam calls, and never reaches phase 2 below.
        async with self._write_lock, self._dedup_lock:
            opened_transaction = not self._db.in_transaction
            async with self._serialize_dedup_probe(operation="upsert_by_hash"):
                existing = await self._get_thought_by_content_hash(
                    _compute_content_hash(thought.content),
                )
                if existing is not None:
                    return await self._upsert_matched_row(
                        thought, existing, opened_transaction=opened_transaction
                    )
                await self._end_exploratory_probe(opened_transaction=opened_transaction)
        # _write_lock, _dedup_lock and any BEGIN IMMEDIATE this call's own
        # probe opened are all released above -- the seam below runs holding
        # no lock this call itself acquires (an enclosing caller's own lock
        # is a different matter -- see prepare_thought_for_insert's docstring).

        thought = await self.prepare_thought_for_insert(thought)
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)

        # Phase 2 -- decisive probe: reacquired from scratch, keyed on the
        # prepared record's content. Insert only if this still misses.
        async with self._write_lock, self._dedup_lock:
            # Sampled again, not carried over from phase 1: the seam (or
            # anything else on this connection) may have left a transaction
            # open in between, and this window opened nothing then.
            opened_transaction = not self._db.in_transaction
            async with self._serialize_dedup_probe(operation="upsert_by_hash"):
                existing = await self._get_thought_by_content_hash(
                    _compute_content_hash(thought.content),
                )
                if existing is None:
                    # Insert-only, inside the window; auto-embed / on_store /
                    # derivation run afterwards -- at the top level, once the
                    # write lock is released; nested in the caller's own
                    # suspend_auto_commit() window, while that outer hold stays
                    # in place instead (see _insert_new_thought_row /
                    # _finish_create_thought's docstring for both cases).
                    persisted = await self._insert_new_thought_row(
                        thought,
                        expires_after_seconds=expires_after_seconds,
                    )
                else:
                    # Another writer won the race between the two probes -- the
                    # seam already ran once for this call; take the hit branch
                    # instead of inserting.
                    return await self._upsert_matched_row(
                        thought, existing, opened_transaction=opened_transaction
                    )
        origin_token = _DERIVATION_ORIGIN.set("upsert_by_hash")
        try:
            return await self._finish_create_thought(persisted)
        finally:
            _DERIVATION_ORIGIN.reset(origin_token)

    async def _upsert_matched_row(
        self,
        thought: ThoughtRecord,
        existing: ThoughtRecord,
        *,
        opened_transaction: bool,
    ) -> ThoughtRecord:
        """Hit branch shared by both of ``upsert_by_hash``'s probes.

        Applies whichever of ``thought``'s :attr:`_UPSERT_MUTABLE_FIELDS`
        differ from ``existing`` onto the stored row, or leaves the row
        untouched if none differ. Used identically whether the match was found
        on the first (exploratory) probe or the second (decisive) probe of
        :meth:`upsert_by_hash`'s two-phase miss path — the two probes share
        this one implementation so they cannot drift.

        Only the fields that actually differ are forwarded to
        :meth:`update_thought`. This keeps the update minimal (no spurious OCC
        churn or re-embed when a field is unchanged) and, critically, never
        re-asserts an identical ``lifecycle_status`` — ``evolve`` rejects
        same-state lifecycle transitions, so passing the stored value back
        verbatim would raise ``InvalidTransitionError``. ``update_thought``
        commits its own write (after its own journal append), the same
        "whoever writes, commits" rule :meth:`_insert_new_thought_row` and
        :meth:`_increment_confirmation` follow — this method needs no special
        handling for that case.

        Args:
            thought: The candidate whose mutable fields may overwrite ``existing``.
            existing: The stored row the content-hash probe matched.
            opened_transaction: Whether *this* call's own window opened the
                ``BEGIN IMMEDIATE`` that may still be open, forwarded from the
                caller so the no-change branch can end its own transaction
                without touching a caller's. See
                :meth:`_end_exploratory_probe`.

        Returns:
            ``existing`` unchanged when no field differs, or the record
            :meth:`update_thought` returns otherwise.

        """
        changes = {
            field: getattr(thought, field)
            for field in self._UPSERT_MUTABLE_FIELDS
            if getattr(thought, field) != getattr(existing, field)
        }
        if not changes:
            # No write happens on this branch at all, but the window that
            # found this match still holds the transaction that was open
            # before the probe ran -- its own `BEGIN IMMEDIATE` when this
            # call opened one, or an already-open outer transaction when it
            # reused one instead (it cannot know in advance that the match
            # needs no change) -- and the guard itself no longer commits on a
            # clean exit. Closing
            # what nothing else here will, *without committing*: a
            # `_maybe_commit()` here would commit whatever the caller already
            # had pending, so their own later `rollback()` would find nothing
            # to undo -- a call that writes nothing must not commit somebody
            # else's work. `_end_exploratory_probe` rolls back instead, and
            # only the transaction this call itself opened, so the write
            # reservation is released without touching a caller's.
            await self._end_exploratory_probe(opened_transaction=opened_transaction)
            return existing
        return await self.update_thought(existing.thought_id, **changes)

    async def bulk_store(
        self,
        thoughts: list[ThoughtRecord],
        *,
        deduplicate: bool = False,
    ) -> list[ThoughtRecord]:
        # Write-lock classification: under _write_lock via _bulk_store_inner's
        # own suspend_auto_commit() window, which holds the lock for its whole
        # duration (own body has no direct SQL).
        """Persist many thoughts in a single transaction, all-or-nothing when this call owns it.

        The batch analogue of :meth:`create_thought` for ingest paths that
        would otherwise loop ``create_thought`` (one commit — and, under
        auto-embed, one embedding round trip — per thought). The whole loop
        runs under :meth:`suspend_auto_commit`, so:

        * **One commit** — every row commits together when the batch finishes,
          not once per row.
        * **All-or-nothing when this call owns the transaction** — if any row
          raises (duplicate id, metadata violation, an embedding failure under
          ``require_embedding=True``, …) the entire transaction is rolled back
          and *nothing* is persisted; the exception propagates. Partial
          batches never land this way — **provided this call's own**
          :meth:`suspend_auto_commit` **window is the outermost one.** Nested
          inside a caller's own :meth:`suspend_auto_commit` window, only the
          outermost window's exit decides commit or rollback (see that
          method's docstring): this call's own raise does not roll anything
          back at the inner level, so a caller that catches the error and lets
          its own window exit cleanly commits the batch's successful prefix
          along with the rest of its work.
        * **Order preserved** — the returned list is in input order, element
          *i* corresponding to ``thoughts[i]`` (or, under ``deduplicate=True``,
          the existing record that ``thoughts[i]`` collapsed onto).

        When auto-embed is active, the per-thought embed is suppressed during
        the insert loop and all thoughts are embedded in **one** batch provider
        call afterwards (role-aware ``embed_document_batch`` when the provider
        exposes it, else ``embed_batch``; dispatched exactly like the
        single-item path), still inside the same transaction. The resulting
        vectors are byte-identical to embedding each thought individually. A
        deduplication hit is not re-embedded (its content is unchanged), so only
        the genuinely-inserted thoughts are batch-embedded. "Genuinely inserted"
        is decided by row existence (a snapshot of the stored ids taken before
        the batch, plus the ids inserted earlier in the same batch), so a dedup
        hit is skipped even when the submitted record reuses an existing row's
        id.

        When the derived-records seam is active, derivation is dispatched
        **locally after the batch commits**, once per genuinely newly-created
        record (a dedup / hash hit never derives — it returns before the
        dispatch), so derivation runs only on durably-committed inserts, off the
        batch transaction.

        Like :meth:`suspend_auto_commit`, this call briefly toggles
        store-instance state (the deferred-commit flag, and a flag that defers
        per-thought embedding to the batch). The store owns a single
        connection, but a *second* writer no longer needs to be kept off it
        by convention: :attr:`_write_lock` covers this whole call the same
        way it covers :meth:`suspend_auto_commit`, so a concurrent task's
        guarded write on this instance simply waits for the batch to finish
        (or times out, per :attr:`_WRITE_LOCK_ACQUIRE_TIMEOUT_SECONDS`)
        instead of racing its toggled state. Submitting ``bulk_store`` from
        one task at a time is still the simplest mental model, not a
        requirement for correctness.

        Args:
            thoughts: The thoughts to persist, in order. An empty list is a
                no-op returning ``[]`` (no transaction is opened).
            deduplicate: Applied per row exactly like
                ``create_thought(deduplicate=True)`` — a row whose ``content``
                hash already exists bumps that record's ``confirmation_count``
                and yields the existing record instead of inserting.

        Returns:
            The persisted records in input order.

        Raises:
            ValueError: If any thought has a duplicate id or metadata that
                violates the shape/size invariants. Rolls the whole batch
                back when this call's own :meth:`suspend_auto_commit` window
                is the outermost one; nested inside a caller's own window,
                only that outermost window's exit decides commit or rollback
                (see above).
            EmbeddingGenerationError: If batch auto-embed fails and
                ``require_embedding`` is ``True``. Same outermost-window
                caveat as above.
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        if not thoughts:
            return []

        embed_active = self._auto_embed and self._embedding_provider is not None
        derivation_active = self._derive_gates.enabled and isinstance(
            self._hooks,
            DerivedRecordProducerProtocol,
        )

        # The existing-ids snapshot used to be taken here, before any lock was
        # acquired: a concurrent task's insert landing between this read and
        # the batch's own lock acquisition would be invisible to it, so this
        # call's own row could be misclassified as newly-created when it was
        # actually a pre-existing row a dedup hit resolved to — re-embedding
        # and, worse, re-deriving from it, against the guarantee that a
        # dedup hit never derives. It is now taken inside
        # ``_bulk_store_inner``'s own ``suspend_auto_commit`` window, which
        # holds ``_write_lock`` for the window's whole duration, so no
        # concurrent task's write can land between the snapshot and this
        # batch's own insert loop.
        origin_token = _DERIVATION_ORIGIN.set("bulk_store")
        try:
            return await self._bulk_store_inner(
                thoughts,
                deduplicate=deduplicate,
                embed_active=embed_active,
                derivation_active=derivation_active,
            )
        finally:
            _DERIVATION_ORIGIN.reset(origin_token)

    async def _bulk_store_inner(
        self,
        thoughts: list[ThoughtRecord],
        *,
        deduplicate: bool,
        embed_active: bool,
        derivation_active: bool,
    ) -> list[ThoughtRecord]:
        """Run the ``bulk_store`` insert loop under the derivation-origin label.

        Extracted so the ``bulk_store`` public method stays a thin wrapper that
        sets the informational ``DeriveContext.origin`` for the batch.

        **Two phases: only the seam runs holding no lock this call itself
        acquires, and batch-wide; ordinary
        validation stays per-item, in the insert phase.** A batch has every
        one of its records up front, so phase 1 runs :meth:`prepare_thought_for_insert`
        — the seam itself, nothing else — over the *whole* batch before
        phase 2 ever takes ``_write_lock``. Phase 2 then runs under
        ``suspend_auto_commit`` (one transaction) and, for each
        already-seamed record in order, revalidates it (metadata,
        provenance) and inserts it (or dedup-resolves it) via
        :meth:`_create_thought_from_prepared`, which does not call the seam
        again.

        **Why the split is this narrow.** An earlier version of this method
        ran :meth:`_prepare_for_create` — validate, seam, revalidate — for
        the whole batch in phase 1, matching how :meth:`create_thought` does
        it for one item. That satisfied constraint 1 (the seam never runs
        under ``_write_lock``) but over-corrected the caller-visible failure
        ordering: because *every* item's validation, not only its seam call,
        ran before *any* item's insert, a later item's ordinary
        (non-seam) validation error — invalid ``metadata`` or ``provenance``,
        either on the original candidate or on what the seam returned —
        could raise *before* an earlier item's own duplicate-id failure or
        ``on_store`` call was ever reached, silently replacing that earlier
        outcome and suppressing that earlier ``on_store``. A bare loop of
        :meth:`create_thought` calls never did this: item *n* is fully
        finished — validated, seamed, inserted, ``on_store`` dispatched —
        before item *n + 1* is even looked at. Moving *only* the seam call to
        phase 1 and revalidating in phase 2, per item, immediately before
        that item's own insert, restores that exact ordering: item *n*'s
        ordinary validation, insert, and ``on_store`` all happen within the
        same iteration of phase 2's loop, before item *n + 1*'s iteration
        begins, and a later item's validation failure can no longer pre-empt
        an earlier one's insert-time or ``on_store`` outcome.

        **The one thing this does not restore: the seam itself, in phase 1,
        no longer receives pre-validated input for the bulk path.** Every
        other entry point (:meth:`create_thought`, :meth:`get_or_create`,
        :meth:`upsert_by_hash`) validates the candidate's ``metadata`` /
        ``provenance`` immediately before calling the seam, so an override
        can rely on that precondition — see :meth:`prepare_thought_for_insert`'s
        ``Args`` — but here the seam call in phase 1 runs before phase 2's
        revalidation, on whatever ``thoughts`` was given, unvalidated. A
        seam call for item *m* raising anywhere in phase 1 (a validated
        precondition or not) still stops the whole batch there, with no lock
        taken and no transaction opened, so nothing is persisted — the same
        all-or-nothing outcome the batch has always produced when an item
        fails (see :meth:`bulk_store`'s docstring), reached without ever
        opening a transaction that was never going to complete. Only
        genuinely-inserted records — those whose id was absent before the
        batch and not inserted earlier in it, i.e. dedup / hash hits excluded
        — are collected as ``newly_created``. **After** the batch
        commits and is durable (the ``async with`` has exited), derivation is
        dispatched locally, per newly-created record, off the batch
        transaction, each child its own guarded durable unit; a
        producer/child failure there can never roll back a committed source
        or child. There is no shared instance buffer —
        ``newly_created`` is a local variable of this call.

        Args:
            thoughts: The thoughts to persist, in order.
            deduplicate: Per-row content-hash deduplication toggle.
            embed_active: Whether auto-embed is active for this batch.
            derivation_active: Whether the derived-records seam is active (used to
                collect the newly-created records for post-commit dispatch).

        Returns:
            The persisted records in input order.

        """
        # Phase 1 -- *only* the pre-insert seam itself, over the whole batch,
        # holding no lock this call itself acquires. Deliberately not
        # _prepare_for_create: that helper
        # also validates (before and after the seam), and running validation
        # here, batched across the whole input ahead of phase 2, is exactly
        # what let a later item's ordinary validation error pre-empt an
        # earlier item's duplicate-id or on_store outcome (see this method's
        # docstring). Only the seam -- the one piece of overridable code
        # constraint 1 requires out from under the lock -- runs here; ordinary
        # validation now runs in phase 2, per item, at the point the base
        # loop of create_thought() calls would have run it.
        prepared_thoughts = [await self.prepare_thought_for_insert(thought) for thought in thoughts]

        newly_created: list[ThoughtRecord] = []
        async with self.suspend_auto_commit():
            # Phase 2 -- ``_write_lock`` held for this whole window. Snapshot
            # the ids that already exist so genuine inserts can be told apart
            # from dedup hits deterministically — by *row existence*, never
            # by instance identity. (Instance identity is unreliable:
            # ``create_thought`` rebuilds the record to populate timestamps,
            # and a dedup hit can return a row whose id coincides with the
            # submitted one.) Taken here, inside this ``suspend_auto_commit``
            # window — which holds ``_write_lock`` for its whole duration —
            # so no concurrent task's insert can land between this read and
            # the loop below that relies on it. Only rows whose id is absent
            # here — and not yet inserted earlier in this same batch — are
            # freshly inserted, and thus the ones that need embedding and are
            # eligible for derivation. Taken whenever embedding OR the
            # derived-records seam is active, so a dedup hit never derives
            # even with auto-embed off.
            pre_existing_ids: set[str] = set()
            if embed_active or derivation_active:
                pre_existing_ids = await self._existing_thought_ids()
            self._suppress_auto_embed = embed_active
            try:
                persisted: list[ThoughtRecord] = []
                inserted: list[ThoughtRecord] = []
                seen_before: set[str] = set(pre_existing_ids)
                for thought in prepared_thoughts:
                    # Ordinary validation, per item, at the point the base
                    # per-item create_thought() loop would have run it --
                    # right before *this* item's own insert, not batched
                    # ahead of the whole loop (see this method's docstring).
                    # Validates what the seam returned; the seam already ran,
                    # holding no lock this call itself acquires, in phase 1
                    # above.
                    _validate_metadata(thought.metadata)
                    _validate_provenance(thought.provenance)
                    record = await self._create_thought_from_prepared(
                        thought,
                        expires_after_seconds=None,
                        deduplicate=deduplicate,
                    )
                    persisted.append(record)
                    if record.thought_id not in seen_before:
                        if embed_active:
                            inserted.append(record)
                        if derivation_active:
                            newly_created.append(record)
                    seen_before.add(record.thought_id)
                if embed_active and inserted:
                    await self._batch_embed_thoughts(inserted)
            finally:
                self._suppress_auto_embed = False
        # Batch committed and durable now — dispatch derivation locally, off the
        # transaction, for the records this call genuinely newly-created.
        for record in newly_created:
            await self._dispatch_derivation(record)
        return persisted

    async def _existing_thought_ids(self) -> set[str]:
        """Return the set of ``thought_id`` values currently in the table.

        Used by :meth:`bulk_store` to snapshot row existence before a batch so
        genuine inserts are distinguished from dedup hits by whether the row
        already existed, independent of Pydantic instance identity.

        Returns:
            Every ``thought_id`` currently stored.

        """
        cursor = await self._db.execute("SELECT thought_id FROM thought")
        rows = await cursor.fetchall()
        return {str(row[0]) for row in rows}

    # ------------------------------------------------------------------
    # Derived-records extension seam
    # ------------------------------------------------------------------

    async def derive_existing(self, thought_id: str) -> DeriveResult:
        """Run the registered derived-records producer over a stored thought.

        The explicit backfill counterpart of the automatic on-store derived-
        records trigger (:meth:`_dispatch_derivation`): for an already-stored
        source thought it invokes the configured producer capability and persists
        every returned child through the **same** core-owned per-child lifecycle
        the on-store path uses (:meth:`_derive_and_persist` →
        :meth:`_persist_derived_child`). Because backfilled children share the
        on-store path's exact content-addressed identity, guarded lifecycle, and
        ``DERIVED_FROM`` edge, a backfill **converges** with the on-store path:
        its output is byte-identical to what an on-store write would have produced
        for the same content, so backfilled and auto-derived records dedup against
        one another. This convergence holds for producers that respect the
        informational-``origin`` contract: ``DeriveContext.origin`` is **not part
        of the content-hash identity** and a producer must not derive a child's
        content or identity from it. ``origin`` already varies across the on-store
        entry points (``create_thought`` / ``bulk_store`` / ``get_or_create`` /
        ``upsert_by_hash``) and again here, so a producer that keyed its output off
        ``origin`` would already diverge between two on-store writes — this is the
        pre-existing seam contract, not a new limitation of backfill. Re-running it
        is idempotent — already-present children are reused, missing ones filled.

        Gating (independent of ``DeriveGates.enabled``): this runs whenever a
        producer capability is present, honouring ``DeriveGates.on_error`` and
        ``max_derived_per_source`` — but **not** ``DeriveGates.enabled``, which
        governs only the automatic on-store trigger. So an existing base can be
        backfilled once without committing to automatic derivation on every future
        write. With no producer capability registered it is a clean no-op.

        Recursion guard: it consults and sets the same :data:`_IN_DERIVATION`
        guard as the on-store path, so a producer's own nested public write
        (including a nested ``derive_existing``) never re-dispatches — depth stays
        at most one — and a ``derive_existing`` invoked from within a derivation is
        a no-op. A source that is itself a derived record (it carries an outgoing
        ``DERIVED_FROM`` edge) is never re-derived.

        Unlike :meth:`_dispatch_derivation` it does **not** early-return inside a
        caller-held ``suspend_auto_commit`` window (or a caller-held raw
        ``BEGIN``): the source is already durable (stored by a prior committed
        call), so there is no source-durability reason to defer. If a caller
        wraps it in an open transaction, the children simply join that
        transaction like any other write — but who then decides *when* that
        becomes durable depends on which kind of transaction it is. A
        ``suspend_auto_commit`` window suppresses every write's own
        auto-commit for its whole duration, so the caller genuinely owns the
        children's durability: nothing commits until the window's own single,
        final commit. A raw ``BEGIN`` does **not** suppress it: a child's own
        successful step still calls its own ``_maybe_commit()``, which — since
        nothing told it to defer — actually commits the shared transaction
        (the caller's own pending write included) as soon as that step
        succeeds, well before the caller's own explicit commit. See
        :meth:`_insert_derived_row` and :meth:`_insert_derived_edge` for the
        two steps that can trigger this.

        **A failed child undoes only itself, in every transaction context** — see
        :meth:`_persist_derived_child` for exactly what a failing step leaves
        behind. Under ``DeriveGates.on_error="log"`` the dispatch logs the
        failure and continues with the next child; the caller's other pending
        writes in the same transaction are never touched. Under
        ``on_error="raise"`` the error propagates after that per-step undo: if
        the caller lets it escape an open ``suspend_auto_commit`` window
        uncaught, the window rolls back everything it holds — its own normal
        atomicity, not a behaviour unique to backfill — but if the caller
        catches it inside the window, the window's other writes survive. The
        already-committed **source thought is unaffected either way**. Outside a
        suspend window each child commits as its own durable unit (per-child
        isolation), unchanged from before.

        Args:
            thought_id: The already-stored source thought to derive from.

        Returns:
            A :class:`~engrava.domain.protocols.derived_records.DeriveResult`
            tallying children created / reused / skipped for this run (all zero
            for a clean skip or no-op).

        Raises:
            SourceThoughtNotFoundError: If ``thought_id`` does not exist.
            DerivedRecordError: If the producer's return violates the seam's
                deterministic contract (over cap, or an identity collision) and
                ``DeriveGates.on_error="raise"``.
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        row = await self._get_thought_row(thought_id)
        if row is None:
            # A missing source is a precondition failure (an error), distinct from
            # the clean empty result returned for an ineligible source below.
            raise SourceThoughtNotFoundError(thought_id)
        # Nested no-op: a derive_existing issued from within an active derivation
        # (e.g. by a contract-violating producer) must not re-dispatch (depth ≤ 1).
        if _IN_DERIVATION.get():
            return DeriveResult(thought_id=thought_id)
        # Capability-present gate — deliberately independent of DeriveGates.enabled
        # (that master switch governs only the automatic on-store trigger).
        if not isinstance(self._hooks, DerivedRecordProducerProtocol):
            return DeriveResult(thought_id=thought_id)
        producer: DerivedRecordProducerProtocol = self._hooks
        # A source that is itself a derived record is never re-derived: an outgoing
        # DERIVED_FROM edge is the structural marker of a derived child.
        if await self._has_outgoing_derived_edge(thought_id):
            return DeriveResult(thought_id=thought_id)
        # Derive from the raw stored row (never on_retrieve-transformed), so the
        # content — and thus the derived children — match the on-store path, which
        # derives from the record as written. This also avoids buffering an access.
        source = self._row_to_thought(row)
        outcome = await self._derive_and_persist(producer, source, _ORIGIN_DERIVE_EXISTING)
        return DeriveResult(
            thought_id=thought_id,
            created=outcome.created,
            reused=outcome.reused,
            skipped=outcome.skipped,
        )

    async def _has_outgoing_derived_edge(self, thought_id: str) -> bool:
        """Return whether *thought_id* is itself a derived record.

        A derived child is linked to its source by an outgoing ``DERIVED_FROM``
        edge (derived → source), so an outgoing edge of that type is the
        structural marker that a thought was produced by the derived-records seam.
        The explicit backfill entry point consults this to skip a source that is
        itself a derived record. The query is index-backed (``idx_edge_type_from``)
        and short-circuits on the first match.

        Args:
            thought_id: The candidate source thought id.

        Returns:
            ``True`` when at least one outgoing ``DERIVED_FROM`` edge exists.

        """
        cursor = await self._db.execute(
            "SELECT 1 FROM edge WHERE from_thought_id = ? AND edge_type = ? LIMIT 1",
            (thought_id, EdgeType.DERIVED_FROM.value),
        )
        return await cursor.fetchone() is not None

    async def _dispatch_derivation(self, source: ThoughtRecord) -> None:
        """Persist an extension's derived records for a committed source thought.

        Runs only when the seam is enabled, the source is durable (auto-commit is
        not suspended), the recursion guard is clear, and the configured hooks
        object implements
        :class:`~engrava.domain.protocols.derived_records.DerivedRecordProducerProtocol`.
        When any of those does not hold it returns without touching the store, so
        the disabled/absent path produces byte-identical persisted results (DB +
        journal) — it does at most a single cheap capability/enabled check before
        returning, not zero extra work.

        When called while **the current task's own** auto-commit window is still
        open it returns without dispatching: that source is not yet durable and
        derivation must never run inside a transaction. This is asked of the
        current task only — via a task-local marker, not the instance-wide
        ``_skip_auto_commit_depth`` — so a *different* task's open window (which
        holds no transaction this source's insert belongs to) is invisible here;
        see ``_current_auto_commit_window`` and ``_open_auto_commit_windows``.
        ``bulk_store`` instead dispatches derivation locally, per newly-created
        record, *after* its batch commits; a caller writing inside its own
        ``suspend_auto_commit`` window triggers derivation via an explicit
        re-run/backfill -- recoverability, not automatic recovery. A
        dedup / hash hit never reaches this method (those return before the
        dispatch call in ``create_thought``), so only genuine inserts derive.

        The recursion guard (:data:`_IN_DERIVATION`) is set for the whole
        dispatch — including any nested public write a (contract-violating)
        producer might issue — so derivation depth never exceeds one.

        Args:
            source: The committed source thought to derive from (the record that
                was persisted, i.e. the input to ``on_store``).

        """
        if not self._derive_gates.enabled:
            return
        if _IN_DERIVATION.get():
            return
        window_id = self._current_auto_commit_window.get()
        if window_id is not None and window_id in self._open_auto_commit_windows:
            # Not yet durable inside THIS TASK's own suspended-commit window —
            # asked of the current task only (see `_current_auto_commit_window`),
            # so another task's open window never trips this. Do not dispatch
            # and do not buffer — bulk_store dispatches locally post-commit, and
            # a caller-held transaction triggers derivation via explicit backfill.
            return
        if not isinstance(self._hooks, DerivedRecordProducerProtocol):
            return
        # Share the exact per-child dispatch path with the explicit backfill
        # entry point (:meth:`derive_existing`). The returned tally is only
        # meaningful to that caller, so the on-store trigger discards it (callers
        # observe on-store derivation through the store state, not a return).
        await self._derive_and_persist(self._hooks, source, _DERIVATION_ORIGIN.get())

    async def _derive_and_persist(
        self,
        producer: DerivedRecordProducerProtocol,
        source: ThoughtRecord,
        origin: str,
    ) -> _DerivationOutcome:
        """Build the context, set the recursion guard, and run derivation.

        The single code path shared by the automatic on-store trigger
        (:meth:`_dispatch_derivation`) and the explicit backfill entry point
        (:meth:`derive_existing`), so both produce byte-identical children and
        edges: the source content-hash identity, the guarded per-child lifecycle,
        and the recursion guard are all computed here in exactly one place. The
        two callers differ only in their *gating* (the on-store trigger honours
        ``DeriveGates.enabled``; backfill runs on capability-present alone) and in
        the informational ``origin`` label — never in how a child is persisted.

        The context's ``cycle_at_derivation`` is the source's own
        ``updated_cycle`` (the cycle observed on the source thought), so a
        backfilled child is stamped with exactly the cycle its on-store
        counterpart would receive — the property that makes backfill converge
        byte-identically with the on-store path.

        The caller MUST have already confirmed the producer capability and that
        the source is eligible; this method unconditionally dispatches.

        Args:
            producer: The derived-record producer capability.
            source: The durable source thought to derive from.
            origin: Informational ``DeriveContext.origin`` label for this path.

        Returns:
            The per-source tally of created / reused / skipped children.

        """
        ctx = DeriveContext(
            source_thought_id=source.thought_id,
            source_content_hash=_compute_content_hash(source.content),
            cycle_at_derivation=source.updated_cycle,
            origin=origin,
        )
        token = _IN_DERIVATION.set(True)
        try:
            return await self._run_derivation(producer, source, ctx)
        finally:
            _IN_DERIVATION.reset(token)

    async def _run_derivation(
        self,
        producer: DerivedRecordProducerProtocol,
        source: ThoughtRecord,
        ctx: DeriveContext,
    ) -> _DerivationOutcome:
        """Invoke the producer and persist its derived records per-child.

        Fail-open: the source is already durable, so any failure here never
        rolls it back. ``CancelledError`` always propagates (it is not an
        ``on_error`` case). Under ``on_error="log"`` a producer failure is
        logged and skipped, and a per-child failure is logged and the remaining
        children continue; under ``on_error="raise"`` the error re-raises after
        the source is safe, aborting the remaining children.

        **A quarantined connection is non-continuable under either policy.**
        Persisting a child uses failure-atomic units (:meth:`_write_readback_savepoint`
        via :meth:`_insert_derived_row`, :meth:`_insert_derived_edge` and
        :meth:`store_embedding`); when a unit's own unwind cannot be trusted it
        quarantines the connection and re-raises the child's original error. Seeing
        the connection quarantined after a child failure means every later write
        would fail fast anyway, so the remaining children are never attempted and
        the original error always propagates — logged first under ``"log"``, since
        that policy would otherwise swallow it.

        Args:
            producer: The derived-record producer capability.
            source: The committed source thought.
            ctx: The derivation context.

        Returns:
            The per-source tally of children created / reused / skipped. A child
            is *created* when its content-addressed row is newly inserted,
            *reused* when it collided with an existing row (conflict-as-reuse),
            and *skipped* when its persistence failed under ``on_error="log"``.

        """
        on_error = self._derive_gates.on_error
        records = await self._collect_derived(producer, source, ctx, on_error)
        if records is None:
            return _DerivationOutcome()
        created = 0
        reused = 0
        skipped = 0
        for record in records:
            try:
                inserted = await self._persist_derived_child(source, record, ctx)
            except asyncio.CancelledError:
                raise
            except Exception:
                if self._connection_quarantined:
                    # Non-continuable regardless of on_error: a unit's own
                    # unwind failed while persisting this child (see
                    # _write_readback_savepoint), so the connection can no
                    # longer be trusted and every later write would fail fast
                    # anyway. Abort the remaining children and let the
                    # original error through -- logged first under "log",
                    # which would otherwise swallow it.
                    if on_error == "log":
                        logger.exception(
                            "derived-record persistence quarantined the connection "
                            "for source %s; aborting remaining children",
                            ctx.source_thought_id,
                        )
                    raise
                if on_error == "raise":
                    raise
                logger.warning(
                    "derived-record persistence failed for source %s; continuing",
                    ctx.source_thought_id,
                    exc_info=True,
                )
                skipped += 1
            else:
                if inserted:
                    created += 1
                else:
                    reused += 1
        return _DerivationOutcome(created=created, reused=reused, skipped=skipped)

    async def _collect_derived(
        self,
        producer: DerivedRecordProducerProtocol,
        source: ThoughtRecord,
        ctx: DeriveContext,
        on_error: str,
    ) -> list[DerivedRecord] | None:
        """Invoke the producer, consume its sequence, and enforce the cap.

        Both the ``derive_records`` call **and** the consumption of its returned
        sequence run inside one fail-open guard: an exception raised while
        producing *or while iterating* the result (e.g. a lazy sequence that
        raises mid-iteration) is, under ``on_error="log"``, logged and swallowed
        with the source left durable; under ``on_error="raise"`` it re-raises
        after the source is safe. ``CancelledError`` always propagates. At most
        ``max_derived_per_source + 1`` items are pulled, so a lazy or unbounded
        sequence cannot flood the store; an over-cap return is rejected — before
        any child is written — per ``on_error``.

        Args:
            producer: The derived-record producer capability.
            source: The committed source thought.
            ctx: The derivation context.
            on_error: The active failure policy (``"raise"`` / ``"log"``).

        Returns:
            The bounded list of derived records, or ``None`` when the producer
            failed, iteration failed, or an over-cap return was rejected under
            ``on_error="log"``.

        Raises:
            DerivedRecordError: When the return is over-cap and
                ``on_error="raise"``.

        """
        cap = self._derive_gates.max_derived_per_source
        try:
            raw = await producer.derive_records(source, ctx)
            collected = list(islice(raw, cap + 1))
        except asyncio.CancelledError:
            raise
        except Exception:
            if on_error == "raise":
                raise
            logger.warning(
                "derive_records failed for source %s; skipping derivation",
                ctx.source_thought_id,
                exc_info=True,
            )
            return None
        if len(collected) > cap:
            msg = f"producer returned more than max_derived_per_source={cap} records"
            if on_error == "raise":
                raise DerivedRecordError(source.thought_id, msg)
            logger.warning("%s for source %s; skipping derivation", msg, ctx.source_thought_id)
            return None
        return collected

    async def _persist_derived_child(
        self,
        source: ThoughtRecord,
        record: DerivedRecord,
        ctx: DeriveContext,
    ) -> bool:
        """Persist a single derived record as an ordinary thought, per-child.

        Runs the same lifecycle an ordinary thought gets — insert →
        ``_maybe_commit`` → auto-embed → (optional) ``DERIVED_FROM`` edge — but
        without re-entering ``on_store`` (derived records are core-persisted).
        Insertion and the edge are conflict-safe at the DB level, so a child
        colliding with an existing row is reused, not re-inserted.

        Enrichment (embedding + edge) is completion-driven, not insert-driven:
        the embedding is generated whenever the persisted row has none yet, and
        the edge insert is conflict-safe. So a re-run over a child that committed
        but never got enriched — e.g. a crash or cancellation between the child's
        commit and its post-commit embedding/edge — completes the enrichment
        idempotently -- recoverability, not atomic enrichment.

        Enrichment always targets the **stored** row's own content, never the
        producer's content. On a conflict-as-reuse hit the stored row may differ
        from the producer's record (a caller can pre-create a thought whose id
        equals a derived child's deterministic id but with different content), so
        the embedding is computed from the re-read stored row — a producer-content
        vector is never attached to a row whose content differs.

        Per-child transaction isolation: a child's **row** commits as its own
        durable unit; its enrichment (embedding, ``DERIVED_FROM`` edge) completes
        afterward, so a child may be durably present yet not-yet-enriched — a
        recoverable partial state, not atomic enrichment. **A failed step
        undoes only itself, in every transaction context — including inside a
        caller's own ``suspend_auto_commit`` window or raw ``BEGIN``.** The row
        insert and its journal append (:meth:`_insert_derived_row`) are one
        :meth:`_write_readback_savepoint` unit; the ``DERIVED_FROM`` edge insert
        and its append (:meth:`_insert_derived_edge`) are another. The embed step
        needs no unit of its own: a provider failure writes nothing, and a failed
        :meth:`store_embedding` unwinds through that method's own unit. Either
        way, a failure there never touches the row step 1 already wrote. No step
        ever rolls back more than its own pending write — never an earlier
        child's work, and never a caller's other writes in the same transaction.

        Args:
            source: The committed source thought.
            record: The producer-owned derived record.
            ctx: The derivation context.

        Returns:
            ``True`` when the child's row was newly inserted (created), ``False``
            when an existing row with the same content-addressed identity was
            reused (conflict-as-reuse). A skipped child never returns — it raises
            (surfaced per ``on_error`` by the caller).

        Raises:
            DerivedRecordError: When the derived identity would collide with the
                source thought itself, or when a conflict-as-reuse hit lands on a
                pre-existing row whose stored content differs from the derived
                record (a foreign-identity collision — no provenance edge is
                attached and the collision is surfaced per ``on_error``).
            ConnectionQuarantinedError: When a step's own unit could not unwind
                its failure and quarantined the connection (see
                :meth:`_write_readback_savepoint`).

        """
        child_id = _derived_thought_id(record.content)
        if child_id == source.thought_id:
            # Pure pre-check — no database work has happened yet, nothing to undo.
            raise DerivedRecordError(
                source.thought_id,
                "derived record identity collides with its source thought",
            )
        child = self._build_derived_thought(record, child_id, ctx, source)
        reused_foreign = False
        # Each step below undoes only itself on failure — no compensating
        # rollback here at all. `_insert_derived_row` and `_insert_derived_edge`
        # each wrap their own insert + journal append in their own
        # `_write_readback_savepoint` unit (see their docstrings), so a failure
        # unwinds exactly that step's pending write, in every transaction
        # context, including inside a caller's own `suspend_auto_commit` window
        # or raw `BEGIN`. The embed step below needs no unit of its own: a
        # provider failure raises before writing anything, and a failed
        # `store_embedding` unwinds through that method's own unit. Every
        # `async with self._write_lock:` here is a re-entrant no-op once
        # acquired (`_insert_derived_row` / `_insert_derived_edge` /
        # `store_embedding` each take it themselves too), so a failing step's
        # own unwind always runs under the same lock hold its write did.
        async with self._write_lock:
            inserted = await self._insert_derived_row(child)
        async with self._write_lock:
            # Re-read the stored row once: it is both the enrichment
            # target (its own content, never the producer's) and the
            # basis for the provenance identity-collision check below. On
            # a conflict-as-reuse hit it may be a foreign row a caller
            # pre-created at this deterministic id.
            stored_row = await self._get_thought_row(child_id)
            if (
                self._auto_embed
                and self._embedding_provider is not None
                and not self._suppress_auto_embed
                and stored_row is not None
                and await self.get_embedding(child_id) is None
            ):
                # Embed the persisted row's actual content — not the
                # producer's — so a reused foreign row never receives a
                # producer-content vector.
                await self._auto_embed_thought(self._row_to_thought(stored_row))
        if record.attach_provenance_edge:
            # Provenance guard: only attach the ``DERIVED_FROM`` edge when the
            # stored row's content actually matches the derived record. A
            # caller can pre-create a thought whose id equals
            # ``uuid5(record.content)`` but with DIFFERENT content; reusing
            # that row and still attaching the edge would assert a false
            # "derived from source" provenance. On a mismatch treat it as an
            # identity collision: skip the edge and surface it per
            # ``on_error`` (mirroring the source-id collision above).
            if stored_row is None or stored_row["content"] != record.content:
                reused_foreign = True
            else:
                async with self._write_lock:
                    await self._insert_derived_edge(
                        child_id,
                        source.thought_id,
                        ctx.cycle_at_derivation,
                    )
        if reused_foreign:
            # Foreign-identity collision: the conflict-as-reuse hit landed on a
            # pre-existing row whose content differs from this derived record, so
            # no provenance edge was attached. Surface it outside any unwind path
            # — no uncommitted mutation is pending (the reuse insert aborted
            # cleanly and any stored-row embedding already committed) — so
            # ``_run_derivation`` applies ``on_error`` (log→skip this child /
            # raise→abort remaining), exactly like the source-id collision.
            raise DerivedRecordError(
                source.thought_id,
                "derived record identity collides with an unrelated stored thought",
            )
        return inserted

    def _build_derived_thought(
        self,
        record: DerivedRecord,
        child_id: str,
        ctx: DeriveContext,
        source: ThoughtRecord,
    ) -> ThoughtRecord:
        """Assemble the core ``ThoughtRecord`` for a derived child.

        Core owns every system-managed field: identity (the deterministic
        content hash), the ``essence`` (derived from content), timestamps, cycle
        (from ``ctx``), and a ``CREATED`` lifecycle status. Provenance origin
        (``source``/``source_type``) is inherited from the source thought. The
        producer contributes only content, type, priority, and the metadata
        payload.

        Args:
            record: The producer-owned derived record.
            child_id: The deterministic child identity.
            ctx: The derivation context (supplies the cycle).
            source: The source thought (supplies provenance origin fields).

        Returns:
            A fully-populated core :class:`ThoughtRecord`.

        """
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        return ThoughtRecord(
            thought_id=child_id,
            thought_type=record.thought_type,
            essence=_essence_from_content(record.content),
            content=record.content,
            priority=record.priority,
            lifecycle_status=LifecycleStatus.CREATED,
            created_cycle=ctx.cycle_at_derivation,
            updated_cycle=ctx.cycle_at_derivation,
            source=source.source,
            source_type=source.source_type,
            metadata=dict(record.metadata),
            created_at=now_iso,
            updated_at=now_iso,
        )

    async def _insert_derived_row(self, child: ThoughtRecord) -> bool:
        """Insert a derived child row conflict-safely (conflict-as-reuse).

        A child whose deterministic identity already exists (a pre-existing row
        or a concurrent/repeat derivation) is reused, not re-inserted — the
        ``UNIQUE`` / primary-key violation is caught and treated as reuse.
        Enrichment of a reused row is handled by the caller against the stored
        row's own content. The conflicting ``INSERT`` statement is aborted by
        SQLite itself (its own changes rolled back, the transaction preserved),
        so the reuse early-return leaves no pending uncommitted mutation behind:
        it still runs inside the savepoint unit below (the ``INSERT`` is what
        raises), but exits that unit normally — a clean ``RELEASE``, never an
        unwind — because nothing of this attempt is left for the unit to undo.

        **The insert and its journal entry are one failure-atomic unit**, via
        :meth:`_write_readback_savepoint` — see :meth:`update_thought` for what
        that protects against. A failed or cancelled journal append here unwinds
        the insert too, in every transaction context (including inside a
        caller's ``suspend_auto_commit`` window or raw ``BEGIN``), instead of
        leaving it pending for a later, unrelated commit to publish without the
        journal entry that documents it.

        Args:
            child: The derived thought to persist.

        Returns:
            ``True`` when a new row was inserted, ``False`` when an existing row
            with the same content-addressed identity was reused (conflict-as-
            reuse). The caller uses this to tally created vs reused children.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        async with self._write_lock:
            async with self._write_readback_savepoint("insert_derived_row", begin="IMMEDIATE"):
                try:
                    await self._db.execute(
                        self._CORE_INSERT_SQL,
                        self._thought_to_core_params(child),
                    )
                except aiosqlite.IntegrityError as exc:
                    if not _is_unique_violation(exc):
                        raise
                    return False
                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="INSERT_THOUGHT",
                        target_id=child.thought_id,
                        delta={"before": None, "after": child.model_dump(mode="json")},
                    )
            await self._maybe_commit()
            return True

    async def _insert_derived_edge(
        self,
        from_thought_id: str,
        to_thought_id: str,
        cycle: int,
    ) -> None:
        """Attach the single ``DERIVED_FROM`` provenance edge, conflict-safely.

        Records content-level provenance (derived → source). The edge is
        conflict-safe on both its deterministic id and the ``(from, to, type)``
        unique constraint, so a re-run or a concurrent derivation reuses the
        existing edge rather than failing; SQLite itself aborts the conflicting
        ``INSERT`` (rolling back only its own changes), so the reuse early-return
        leaves no pending uncommitted mutation: it still runs inside the
        savepoint unit below (the ``INSERT`` is what raises), but exits that
        unit normally — a clean ``RELEASE``, never an unwind — because nothing
        of this attempt is left for the unit to undo.

        **The insert and its journal entry are one failure-atomic unit**, via
        :meth:`_write_readback_savepoint` — see :meth:`update_thought` for what
        that protects against. A failed or cancelled journal append here unwinds
        the edge insert too, in every transaction context (including inside a
        caller's ``suspend_auto_commit`` window or raw ``BEGIN``): the child's
        row (and any embedding) survives untouched, per-child isolation.

        Args:
            from_thought_id: The derived child id (edge origin).
            to_thought_id: The source thought id (edge target).
            cycle: The cycle to stamp on the edge.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        edge = EdgeRecord(
            edge_id=_derived_edge_id(from_thought_id, to_thought_id),
            from_thought_id=from_thought_id,
            to_thought_id=to_thought_id,
            edge_type=EdgeType.DERIVED_FROM,
            weight=1.0,
            created_cycle=cycle,
            source=KnowledgeSource.EXPERIENCE,
        )
        async with self._write_lock:
            async with self._write_readback_savepoint("insert_derived_edge", begin="IMMEDIATE"):
                try:
                    await self._db.execute(
                        "INSERT INTO edge "
                        "(edge_id, from_thought_id, to_thought_id, edge_type, weight, "
                        " created_cycle, source, decay_multiplier, valid_from, valid_until, "
                        " metadata_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            edge.edge_id,
                            edge.from_thought_id,
                            edge.to_thought_id,
                            edge.edge_type.value,
                            edge.weight,
                            edge.created_cycle,
                            edge.source.value,
                            edge.decay_multiplier,
                            edge.valid_from,
                            edge.valid_until,
                            # Derived edges never carry caller metadata, so bind the
                            # empty ``'{}'`` object literal rather than serializing
                            # the in-memory record. This provenance-only path
                            # therefore cannot smuggle unvalidated (e.g. non-finite)
                            # metadata into the column — it bypasses
                            # ``_validate_metadata`` by writing a trivially valid
                            # empty object, matching the fresh-DDL / ALTER
                            # ``DEFAULT '{}'``.
                            "{}",
                        ),
                    )
                except aiosqlite.IntegrityError as exc:
                    if not _is_unique_violation(exc):
                        raise
                    return
                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="INSERT_EDGE",
                        target_id=edge.edge_id,
                        delta={"before": None, "after": edge.model_dump(mode="json")},
                    )
            await self._maybe_commit()

    async def _batch_embed_thoughts(self, inserted: list[ThoughtRecord]) -> None:
        """Embed the freshly-inserted thoughts of a batch in one provider call.

        Called from :meth:`bulk_store` after the insert loop, inside the same
        transaction, with exactly the rows that were genuinely inserted (dedup
        hits — which keep their stored embedding — are already excluded by the
        caller's row-existence check).

        The embed payloads are built with :func:`_build_embed_input` (same as
        the single-item path) and encoded with :func:`_embed_documents_batch`
        (one round trip, role-aware when the provider supports it). A provider
        failure is routed through :meth:`_on_auto_embed_failure` so it is logged
        and either re-raised or converted to :class:`EmbeddingGenerationError`
        under ``require_embedding=True``. Because this runs inside
        ``suspend_auto_commit``, that raise rolls the whole batch back
        **provided this call's own window is the outermost one** — nested
        inside a caller's own ``suspend_auto_commit()`` window, the raise
        does not roll anything back at this level, and the outer window's
        clean exit commits the batch's successful prefix instead.

        Args:
            inserted: The freshly-inserted records to embed, in input order.

        """
        provider = self._embedding_provider
        if provider is None:
            return  # pragma: no cover
        if not inserted:
            return  # pragma: no cover -- caller already guards on emptiness
        texts = [_build_embed_input(t.essence, t.content) for t in inserted]
        try:
            vectors = await _embed_documents_batch(provider, texts)
        except Exception as exc:  # noqa: BLE001 -- provider may raise any type; re-raised in handler
            # Attribute the whole-batch failure to the first inserted id — a
            # representative, valid lookup key, not a promise that the row
            # is gone. The raise rolls the entire batch back only when this
            # call's own suspend_auto_commit window is outermost; nested in
            # a caller's own window, that window's clean exit can still
            # commit every row named here.
            self._on_auto_embed_failure(inserted[0].thought_id, exc)
        for record, vector in zip(inserted, vectors, strict=True):
            await self.store_embedding(
                record.thought_id,
                vector,
                model_name=provider.model_name,
            )

    async def remember(
        self,
        text: str,
        *,
        metadata: dict[str, MetadataValue] | None = None,
        deduplicate: bool = False,
    ) -> ThoughtRecord:
        """Store a string as a thought with one call.

        Ergonomic shorthand over :meth:`create_thought` for the common case
        of persisting a bare string. A :class:`ThoughtRecord` is built with a
        fresh UUID, ``content=text`` and ``essence=text[:200]`` (the compact
        canonical prefix used in prompts), then handed to ``create_thought``.

        The thought is created at the store's default cognitive cycle
        (``created_cycle == updated_cycle == 0``); callers that track cognitive
        cycles should build a :class:`ThoughtRecord` explicitly and call
        ``create_thought`` so the cycle is recorded.

        Args:
            text: The content to remember. Becomes the thought's ``content``;
                its opening (capped at 200 characters) becomes the ``essence``.
            metadata: Optional structured attributes (e.g. ``speaker``,
                ``lang``, ``session_id``). Defaults to an empty mapping.
            deduplicate: When ``True`` and a thought with byte-identical
                ``content`` already exists, its ``confirmation_count`` is
                incremented and the existing record is returned instead of
                inserting a duplicate (forwarded to
                ``create_thought(deduplicate=True)``). Default ``False``
                inserts a new row on every call.

        Returns:
            The persisted thought record (or the existing record with a bumped
            ``confirmation_count`` when deduplication hits).

        """
        thought = ThoughtRecord(
            thought_id=str(_uuid.uuid4()),
            thought_type=ThoughtType.NOTE,
            essence=text[:200],
            content=text,
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            source="remember",
            metadata=metadata or {},
        )
        return await self.create_thought(thought, deduplicate=deduplicate)

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        current_cycle: int | None = None,
        recency_now: str | None = None,
        recency_now_half_life: int | None = None,
        filters: MetadataFilter | None = None,
        visibility: VisibilityQueryFilter | None = None,
        collapse_key: str | Sequence[str] | None = None,
        collapse_max_per_unit: int | None = None,
        include_archived: bool = False,
    ) -> HybridSearchResult:
        """Retrieve thoughts relevant to a query with one call.

        Ergonomic shorthand over :meth:`search_hybrid` for the common
        retrieval case: the query text is passed straight through with the
        given ``top_k`` and recency reference.

        When ``current_cycle`` is ``None`` the recency signal is inactive
        (see ``search_hybrid``) — **unless** a ``cycle_provider`` is configured
        on the store, in which case ``search_hybrid`` pulls the cycle from it. A
        store that holds more than ``_RECENCY_NUDGE_THRESHOLD`` thoughts and
        recalls without a cycle *and without a provider* emits a single
        DEBUG-level breadcrumb on the module logger — once per store instance —
        pointing out that passing ``current_cycle`` would let recent thoughts
        rank higher. It is never a warning, never repeats, and is suppressed when
        a provider is configured (recency is already active through it).

        Args:
            query: Natural-language text to search for.
            top_k: Maximum number of results to return.
            current_cycle: Current cognitive cycle for **cognitive-cycle**
                recency. When provided, the recency signal is blended into
                ranking; when ``None`` (and no ``recency_now``), cycle recency is
                skipped — unless a ``cycle_provider`` is configured, which then
                supplies the cycle. Mutually exclusive with ``recency_now``:
                passing an **explicit** ``current_cycle`` together with
                ``recency_now`` ⇒ :class:`RecencyModeConflictError`.
            recency_now: Optional caller-supplied "now" instant (ISO-8601)
                selecting **transaction-time** recency (age by ``updated_at`` /
                ``created_at`` in wall-clock seconds); delegated to
                :meth:`search_hybrid`. Takes precedence over a passive
                ``cycle_provider`` (when supplied with no explicit
                ``current_cycle``, the provider is not consulted). Because
                ``recall`` carries no per-call recency weight, this axis only
                affects ranking when the store's ``default_recency_weight`` is
                ``> 0`` (exactly like the cycle case). The store reads no host
                clock — omitting it leaves the axis off. ``None`` (default) is
                byte-identical to before.
            recency_now_half_life: Optional per-call transaction-time half-life
                override, **in seconds** (default
                ``SearchConfig.recency_now_half_life_seconds`` = 604800);
                consulted only with ``recency_now``. Delegated to
                :meth:`search_hybrid`.
            filters: Optional :class:`~engrava.domain.models.filters.MetadataFilter`
                — an ``AND`` of typed field predicates over ``metadata``;
                delegated to :meth:`search_hybrid`. ``None`` (or an empty
                filter) leaves the candidate set unchanged.
            visibility: Optional
                :class:`~engrava.domain.models.filters.VisibilityQueryFilter`
                for the "public-or-mine" pattern; delegated to
                :meth:`search_hybrid`. **This is a query filter, not access
                control** — it performs no authentication, authorization,
                ownership validation, or write enforcement; the caller can
                forge ``owner``; it is bypassable by passing
                ``visibility=None``, by using another API, or by issuing raw
                SQL; it must not be used to protect tenant data.
            collapse_key: Optional de-fragmentation unit key (a single
                metadata path or an ordered sequence forming a composite key);
                delegated to :meth:`search_hybrid`. When set, only the single
                best-ranked row per caller-defined unit reaches the result and
                the freed slots are backfilled by deeper distinct units. This
                is a **presentation / de-dup convenience, not a filter and not
                isolation** — it does not change which rows are *eligible*, and
                the collapse step itself mutates no score (it only drops
                lower-ranked same-unit members). Note that *setting*
                ``collapse_key`` also widens the internal candidate pool, which
                — because the keyword arm is min-max normalized over the
                candidate set — can rescale normalized fusion scores and shift
                order among units; only ``collapse_key=None`` is byte-identical
                to the unfiltered path. It is only as meaningful as the unit
                metadata the application writes.
            collapse_max_per_unit: Optional intra-unit retention depth for
                ``collapse_key``; delegated to :meth:`search_hybrid`. ``None``
                (the default) keeps one best row per unit; an integer ``>= 1``
                keeps up to that many of a unit's highest-ranked rows and lets
                the freed slots backfill deeper distinct units. Only takes
                effect together with ``collapse_key``; a value ``< 1`` is
                rejected.
            include_archived: When ``False`` (the default) archived thoughts are
                excluded from every retrieval path; delegated to
                :meth:`search_hybrid`. When ``True`` archived rows are re-admitted
                for this call (the "recall something I forgot" escape hatch)
                without restoring them.

        Returns:
            A ``HybridSearchResult`` with the ranked matches and the set of
            backends that contributed.

        Raises:
            RecencyModeConflictError: If both an **explicit** ``current_cycle``
                and ``recency_now`` are supplied.
            InvalidRecencyArgumentError: If ``recency_now`` is not a valid
                ISO-8601 timestamp, or ``recency_now_half_life`` is not ``> 0``.

        """
        # The nudge fires only when there is genuinely no recency source: no
        # explicit cycle, no configured provider, AND no transaction-time
        # ``recency_now``. With any of those set, recency is (or can be) active
        # via ``search_hybrid``, so the "you forgot current_cycle" breadcrumb
        # would mislead. With none of them (the default), this condition is
        # byte-identical to before.
        if (
            current_cycle is None
            and recency_now is None
            and self._cycle_provider is None
            and not self._recency_nudge_emitted
        ):
            count_cursor = await self._db.execute("SELECT COUNT(*) FROM thought")
            count_row = await count_cursor.fetchone()
            total = int(count_row[0]) if count_row is not None else 0
            if total > _RECENCY_NUDGE_THRESHOLD:
                self._recency_nudge_emitted = True
                logger.debug(
                    "recall() called without current_cycle on a store of %d thoughts; "
                    "passing current_cycle enables the recency signal so recent thoughts "
                    "rank higher",
                    total,
                )
        return await self.search_hybrid(
            query_text=query,
            top_k=top_k,
            current_cycle=current_cycle,
            recency_now=recency_now,
            recency_now_half_life=recency_now_half_life,
            filters=filters,
            visibility=visibility,
            collapse_key=collapse_key,
            collapse_max_per_unit=collapse_max_per_unit,
            include_archived=include_archived,
        )

    async def cleanup_expired(
        self,
        now: str | None = None,
        *,
        exclude_id: str | None = None,
    ) -> CleanupResult:
        """Remove or archive thoughts whose ``expires_at`` is in the past.

        The strategy used (``archive`` or ``delete``) is determined by
        the store's ``ttl_strategy`` setting.

        * **archive**: Sets ``lifecycle_status`` to ``ARCHIVED`` and clears
          ``expires_at`` so the thought is no longer subject to TTL. It also
          clears the hygiene-archival markers (``archived_at_cycle`` /
          ``archived_at``) — a TTL archival is *not* a hygiene archival, so the
          markers (which mean "archived by hygiene at this cycle/instant" and back
          the GC restore windows) must be ``NULL``. This keeps TTL-archived rows
          out of hygiene GC and prevents a stale marker from an earlier hygiene
          episode (left behind by a low-level un-archive) from making a
          later TTL re-archival GC-eligible on the earlier, already-elapsed
          restore windows. This write also bumps ``revision`` — unconditionally,
          not enforced against a caller's read — for the same reason
          :meth:`_hygiene_archive` does: the lifecycle just changed underneath
          any caller-held token, so that token must not survive it.
        * **delete**: Physically deletes the expired thought rows (cascading
          to edges, embeddings, and actions via ON DELETE CASCADE).

        Mutations are recorded in the journal when journaling is enabled.

        Args:
            now: Optional ISO-8601 timestamp to use as "current time", in any
                form the timestamp validator accepts; a value without an offset
                is read as UTC. It is compared in the canonical UTC form, so a
                row is taken exactly when its ``expires_at`` instant is at or
                before this one. Defaults to ``datetime.now(UTC).isoformat()``
                when omitted. Useful for deterministic testing.
            exclude_id: Optional thought ID to skip during cleanup.
                Used by auto-cleanup to protect a just-written thought.

        Returns:
            A ``CleanupResult`` with the count of processed thoughts, the
            strategy that was applied, and the canonical UTC form of the
            instant it cleaned up to.

        Raises:
            ValueError: If ``now`` is not a valid ISO-8601 timestamp, or has no
                UTC form within the supported ``datetime`` range. Nothing is
                read or written in that case.

        """
        # A caller's ``now`` is compared as TEXT against canonical stored
        # ``expires_at`` values, so it is put into the same canonical UTC form
        # first -- otherwise a space separator, basic format, a week date or an
        # offset would select rows by string order instead of by instant.
        now = (
            datetime.datetime.now(datetime.UTC).isoformat()
            if now is None
            else canonical_timestamp(now)
        )

        strategy = CleanupStrategy(self._ttl_strategy)

        async with self._write_lock:
            # Sampled before anything below touches the connection, mirroring
            # delete_thought / _delete_thought_atomic: whether this call is the
            # one that opened the transaction it may need to close below.
            opened_transaction = not self._db.in_transaction
            # Whether *anything* in this batch actually wrote — not merely
            # whether a candidate existed, and not just whether
            # ``_delete_thought_atomic`` reported ``deleted``: its "id never
            # existed" branch can still sweep real orphaned children (see
            # _DeleteAtomicResult), which is a write this batch must still
            # commit even though that one candidate reports ``deleted=False``.
            wrote_anything = False

            # The candidate read is now inside the same critical section as the
            # writes below (it used to run before the lock was even acquired):
            # a concurrent task extending a thought's `expires_at`, or a
            # suspended transaction exposing a value it then rolls back,
            # could otherwise land between this read and the write that acts
            # on it, and this call would archive or delete a row that is no
            # longer actually expired -- caller data loss, not merely a stale
            # read.
            cursor = await self._db.execute(
                "SELECT thought_id FROM thought WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            expired_rows = await cursor.fetchall()
            expired_ids = [
                row["thought_id"] for row in expired_rows if row["thought_id"] != exclude_id
            ]

            # The whole batch is one failure-atomic unit, via
            # _write_readback_savepoint (see update_thought for what that
            # protects against): a failed or cancelled journal append on any
            # one candidate unwinds every write this call has made so far in
            # the batch, not just that candidate's own -- a partial batch
            # commit on a later, unrelated write would otherwise publish some
            # candidates' mutations without the journal entries that document
            # them. This loop makes no nested public write of its own (each
            # candidate's own commit is deferred to the single
            # ``wrote_anything`` / ``opened_transaction`` finalization below,
            # unchanged), so nothing inside it can end the unit early.
            # ``_delete_thought_atomic`` nests inside this unit unchanged: this
            # block already opens the transaction (when one is not already
            # open) before the first candidate runs, so that call's own
            # `opened_transaction` sample sees one already open and never ends
            # it on the veto path itself.
            async with self._write_readback_savepoint("cleanup_expired", begin="IMMEDIATE"):
                for tid in expired_ids:
                    if strategy is CleanupStrategy.ARCHIVE:
                        before_row = (
                            await self._get_thought_row(tid) if self._journal is not None else None
                        )
                        changes_before = self._db.total_changes
                        await self._db.execute(
                            "UPDATE thought SET lifecycle_status = ?, expires_at = NULL, "
                            "archived_at_cycle = NULL, archived_at = NULL, "
                            "revision = revision + 1 "
                            "WHERE thought_id = ?",
                            (LifecycleStatus.ARCHIVED.value, tid),
                        )
                        if self._journal is not None and before_row is not None:
                            before = self._row_to_thought(before_row)
                            after = before.evolve(
                                lifecycle_status=LifecycleStatus.ARCHIVED.value,
                                expires_at=None,
                                archived_at_cycle=None,
                                archived_at=None,
                            )
                            await self._journal.append(
                                mutation_type="UPDATE_THOUGHT",
                                target_id=tid,
                                delta={
                                    "before": before.model_dump(mode="json"),
                                    "after": after.model_dump(mode="json"),
                                },
                            )
                        # Sampled last, after the journal append -- not right
                        # after the UPDATE. Not unconditionally ``True`` either:
                        # a ``BEFORE UPDATE`` trigger can veto the archive with
                        # ``RAISE(IGNORE)``, which leaves its own rowcount (and
                        # any naive "the UPDATE ran" assumption) saying a write
                        # happened when the row was left untouched -- and a
                        # comparison taken right there would also miss that the
                        # journal append below it is a real row insert of its
                        # own, on a genuinely vetoed archive, that a comparison
                        # taken before it can never see. Reading `total_changes`
                        # only now, after everything this iteration could have
                        # written, is what closes both gaps at once.
                        wrote_anything = (self._db.total_changes > changes_before) or wrote_anything
                    else:
                        # DELETE strategy.
                        before_row = (
                            await self._get_thought_row(tid) if self._journal is not None else None
                        )
                        # Capture the embedding rowid before the delete drops the
                        # embedding row; the vec0 vector is not FK-reachable and
                        # would otherwise linger as a ghost.
                        vec_rowid = await self._embedding_rowid_for_thought(tid)
                        # Parent delete and explicit child deletes as one atomic
                        # unit — see _delete_thought_atomic for why. Its return
                        # value must be honoured, not ignored: a RAISE(IGNORE)
                        # trigger (or any future silent veto) reports False with
                        # the row still there, and a purge or journal append for
                        # a parent that still exists would purge a live vector
                        # and record false history.
                        result = await self._delete_thought_atomic(tid)
                        wrote_anything = result.wrote_anything or wrote_anything
                        if not result.deleted:
                            continue
                        await self._purge_orphan_vector(vec_rowid)
                        if self._journal is not None and before_row is not None:
                            await self._journal.append(
                                mutation_type="DELETE_THOUGHT",
                                target_id=tid,
                                delta={
                                    "before": self._row_to_thought(before_row).model_dump(
                                        mode="json",
                                    ),
                                    "after": None,
                                },
                            )

            if wrote_anything:
                await self._maybe_commit()
            elif opened_transaction and self._db.in_transaction:
                # Every DELETE-strategy candidate was vetoed (or there simply
                # were none) — nothing in this batch was actually written, so
                # there is nothing of this call's own to commit. Close only a
                # transaction this call itself opened, never one a caller
                # already held — see delete_thought for the same reasoning.
                await self._db.rollback()

        return CleanupResult(
            expired_count=len(expired_ids),
            strategy_applied=strategy.value,
            timestamp=now,
        )

    async def _maybe_auto_cleanup(self, *, exclude_id: str | None = None) -> None:
        """Run auto-cleanup of expired thoughts if cadence threshold is met.

        Args:
            exclude_id: Optional thought ID to exclude from cleanup.
                Prevents archiving/deleting a thought that was just
                created or updated in the current operation.

        """
        if self._ttl_check_every_n < 1:
            return
        self._operation_count += 1
        if self._operation_count >= self._ttl_check_every_n:
            self._operation_count = 0
            await self.cleanup_expired(exclude_id=exclude_id)

    async def get_thought(self, thought_id: str) -> ThoughtRecord | None:
        """Retrieve a thought by its ID, or None if not found.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The thought record, or None if not found.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        row = await self._get_thought_row(thought_id)
        if row is None:
            return None
        self._buffer_accesses([thought_id])
        return await self._hooks.on_retrieve(self._row_to_thought(row))

    async def update_thought(self, thought_id: str, **changes: object) -> ThoughtRecord:
        """Update a thought's fields in place.

        Writes **only the columns this edit owns** — the fields ``changes``
        gives a new value to, plus the ``updated_at`` stamp ``evolve`` always
        refreshes. Columns the caller did not touch keep whatever is in
        storage, so an access recorded by :meth:`record_access` or a
        confirmation counted since the row was read is not rolled back.

        The record returned is read back from storage after the write, so it is
        the row that exists rather than the one the call intended to write; the
        journal ``after`` image is the same read-back.

        **What the version guard does and does not catch.** The write carries a
        guard on ``revision`` as read at the start of the call, and the engine
        itself increments ``revision`` by one on every guarded write to this
        row — atomically, in the same ``UPDATE`` that checks it — so no caller
        action is needed to arm it. ``StaleDataError`` therefore means the
        guarded ``UPDATE`` matched no row, which happens when **any** other
        guarded write landed on this row since it was read here **or the row
        was deleted**; it does not distinguish the two.

        **The read, the validation, and the write are now one critical section
        with respect to every other task on this instance:** the whole span
        from the initial read to the commit runs
        under :attr:`_write_lock`, a task-reentrant lock, so a second task
        calling any guarded write path on this store blocks until this call's
        own write has committed — it can no longer land *between* this call's
        read and its write. Concretely: a competing edit to the **same** field
        can no longer be silently discarded by an interleaved write racing this
        one (each such call now runs to completion before the next one's own
        read), a competing cycle stamp can no longer spuriously reject an
        unrelated edit that merely happened to straddle it, and a
        ``lifecycle_status`` transition is validated against a row that no
        concurrent call on this instance can move out from under it mid-check.
        This governs concurrent calls **on this store instance** only — a
        second store on the same database file, or a caller mutating the row
        through a raw connection this store does not mediate, is outside what
        any in-process lock can reach (see the concurrency documentation).

        **The write, its confirming read-back, and its journal entry are one
        failure-atomic unit.** All three run inside
        :meth:`_write_readback_savepoint`: if the read-back raises — the row
        vanished, or the mapper rejects a stored value — or the journal
        append itself fails or is cancelled, this call's own ``UPDATE`` is
        unwound before the exception propagates, so it can never be
        published by a later, unrelated commit on this connection without
        the journal entry that documents it. A caller-owned transaction (a
        :meth:`suspend_auto_commit` window already in progress) is
        unaffected: only this call's write is undone, not the caller's
        earlier ones.

        Args:
            thought_id: UUID of the thought to update.
            **changes: Fields to update.

        Returns:
            The stored thought record, as persisted by this update.

        Raises:
            ThoughtNotFoundError: If the thought does not exist when the call
                starts, or if the row was deleted before the write could be read
                back.
            StaleDataError: If the guarded write matches no row — another
                guarded write landed on this row since it was read here, or it
                was deleted. Nothing of this update is written when it is
                raised.
            ValueError: If the post-``evolve`` metadata violates the
                metadata-shape or size invariants enforced by
                :func:`_validate_metadata`, or if the post-``evolve``
                provenance is not a
                :class:`~engrava.domain.models.provenance.ProvenanceContext`
                (per :func:`_validate_provenance`).
            WriteContentionError: The guarded write could not proceed because
                the connection reported lock contention.
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        async with self._write_lock:
            current_row = await self._get_thought_row(thought_id)
            if current_row is None:
                raise ThoughtNotFoundError(thought_id)

            current = self._row_to_thought(current_row)

            expected_revision = int(current_row["revision"])
            updated = current.evolve(**changes)

            _validate_metadata(updated.metadata)
            _validate_provenance(updated.provenance)

            columns = self._thought_update_columns(current, updated)
            async with self._write_readback_savepoint("update_thought_readback", begin="DEFERRED"):
                cursor = await self._execute_revision_guarded_write(
                    _build_update_sql(
                        "thought", columns, self._CORE_UPDATE_GUARD, bump_column="revision"
                    ),
                    (*columns.values(), thought_id, expected_revision),
                    operation="update_thought",
                )
                if cursor.rowcount == 0:
                    raise StaleDataError(
                        entity_type="ThoughtRecord",
                        entity_id=thought_id,
                        expected_version=expected_revision,
                    )

                persisted = await self._read_back_thought(thought_id)

                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="UPDATE_THOUGHT",
                        target_id=thought_id,
                        delta={
                            "before": current.model_dump(mode="json"),
                            "after": persisted.model_dump(mode="json"),
                        },
                    )

            await self._maybe_commit()

        # Re-embed when essence or content changed. Deliberately outside the
        # write lock (released above): auto-embed is a slow, arbitrary provider
        # call, and holding a per-instance lock across it would turn every
        # other task's unrelated write into a bottleneck on this one's network
        # round trip — exactly what this section rules out (the lock protects a
        # critical section, not this call's entire lifetime).
        if (
            self._auto_embed
            and self._embedding_provider is not None
            and (persisted.essence != current.essence or persisted.content != current.content)
        ):
            await self._auto_embed_thought(persisted)
            # The member's vector moved, so any REFLECTION that summarizes it
            # must re-bind to the current cluster instead of scoring on a
            # frozen centroid. Strictly on the essence/content path — a
            # metadata-only edit never reaches here, so it cannot re-bind.
            await self._rebind_consolidated_reflections(thought_id)

        await self._maybe_auto_cleanup(exclude_id=thought_id)
        return persisted

    async def restore_thought(
        self, thought_id: str, *, current_cycle: int | None = None
    ) -> ThoughtRecord:
        """Restore an archived thought to ``ACTIVE``, clearing its archive stamp.

        The reversible counterpart to archival — whether the thought was
        archived by the memory-hygiene loop (:meth:`run_hygiene`), TTL cleanup,
        or a manual lifecycle change: an ``ARCHIVED`` thought transitions back to
        ``ACTIVE`` through the lifecycle state machine and **both** hygiene
        archival markers (``archived_at_cycle`` and the wall-clock ``archived_at``)
        are cleared, so an archive round-trips with no data loss. The move is
        journaled as an ``UPDATE_THOUGHT`` when journaling is enabled.

        This is the **canonical** un-archive path — the only one that clears the
        archival markers. The ``ARCHIVED -> ACTIVE`` edge is also reachable
        through a raw ``update_thought(lifecycle_status=ACTIVE)``, but that
        low-level write leaves ``archived_at_cycle`` / ``archived_at`` set. That
        is harmless while the thought stays ``ACTIVE`` (the markers are only
        consulted for ``ARCHIVED`` rows). The hygiene archive path and TTL
        archival both refresh or clear the markers, so a normal re-archival is
        safe; only a *raw* ``update_thought(lifecycle_status=ARCHIVED)`` that
        bypasses both would carry the stale markers into a new archival episode —
        another reason to prefer this method (and the hygiene / TTL flows) over
        low-level lifecycle writes.

        Like :meth:`update_thought`, this writes only the columns the restore
        owns and returns the row read back from storage after the write. It
        shares the same task-reentrant :attr:`_write_lock` critical section,
        so the read, the transition check, and the write are atomic with
        respect to every other guarded write on this instance. It also shares
        :meth:`update_thought`'s :meth:`_write_readback_savepoint` protection:
        a read-back failure, or a failed or cancelled journal append, unwinds
        this call's own write instead of leaving it pending for a later,
        unrelated commit to publish without the journal entry that documents
        it, without disturbing a caller-owned transaction already in
        progress.

        Args:
            thought_id: UUID of the archived thought to restore.
            current_cycle: Optional cycle to stamp as the new ``updated_cycle``;
                when omitted the ``updated_cycle`` is left unchanged.

        Returns:
            The restored thought record (``lifecycle_status`` is ``ACTIVE``).

        Raises:
            ThoughtNotFoundError: If the thought does not exist, or if the row
                was deleted before the write could be read back.
            InvalidTransitionError: If the thought is not currently ``ARCHIVED``.
            StaleDataError: If the guarded write matches no row — another
                guarded write landed on this row since it was read here, or it
                was deleted (see :meth:`update_thought` for what that guard
                does and does not catch). Nothing of the restore is written
                when it is raised.
            WriteContentionError: The guarded write could not proceed because
                the connection reported lock contention.

        """
        async with self._write_lock:
            current_row = await self._get_thought_row(thought_id)
            if current_row is None:
                raise ThoughtNotFoundError(thought_id)
            current = self._row_to_thought(current_row)

            if current.lifecycle_status is not LifecycleStatus.ARCHIVED:
                raise InvalidTransitionError(
                    entity_type="LifecycleStatus",
                    current_state=current.lifecycle_status.value,
                    target_state=LifecycleStatus.ACTIVE.value,
                )

            expected_revision = int(current_row["revision"])
            # Pass the enum (not its value) so ``evolve`` runs the state-machine
            # transition check — the ARCHIVED -> ACTIVE edge is what makes the
            # archive reversible.
            changes: dict[str, object] = {
                "lifecycle_status": LifecycleStatus.ACTIVE,
                "archived_at_cycle": None,
                "archived_at": None,
            }
            if current_cycle is not None:
                changes["updated_cycle"] = current_cycle
            updated = current.evolve(**changes)

            columns = self._thought_update_columns(current, updated)
            async with self._write_readback_savepoint("restore_thought_readback", begin="DEFERRED"):
                cursor = await self._execute_revision_guarded_write(
                    _build_update_sql(
                        "thought", columns, self._CORE_UPDATE_GUARD, bump_column="revision"
                    ),
                    (*columns.values(), thought_id, expected_revision),
                    operation="restore_thought",
                )
                if cursor.rowcount == 0:
                    raise StaleDataError(
                        entity_type="ThoughtRecord",
                        entity_id=thought_id,
                        expected_version=expected_revision,
                    )

                persisted = await self._read_back_thought(thought_id)

                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="UPDATE_THOUGHT",
                        target_id=thought_id,
                        delta={
                            "before": current.model_dump(mode="json"),
                            "after": persisted.model_dump(mode="json"),
                        },
                    )
            await self._maybe_commit()
        return persisted

    async def invalidate_thought(
        self,
        thought_id: str,
        valid_until: str,
    ) -> ThoughtRecord:
        """Close a thought's valid-time interval at the given instant.

        Sets the thought's ``valid_until`` to ``valid_until``, marking the
        end of the window during which the fact is considered true in the
        world. This is a deterministic, valid-time-only operation:

        * It is **not** a delete — the row and all of its history remain
          stored and retrievable; only the valid-time upper bound changes.
        * It performs **no** similarity search, automatic invalidation, or
          model inference of any kind.
        * It does **not** cascade to the thought's edges — invalidating a
          thought leaves every connected edge's valid-time interval
          untouched.
        * It is **idempotent**: invalidating with the same ``valid_until``
          twice converges to the same stored value.

        Args:
            thought_id: UUID of the thought to invalidate.
            valid_until: ISO-8601 instant at which the fact stops being
                valid. Stored as the thought's ``valid_until`` bound.

        Returns:
            The updated thought record.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.
            StaleDataError: If the guarded write matches no row — a competing
                cycle stamp or a delete (see :meth:`update_thought`).
            ValueError: If ``valid_until`` is not a valid ISO-8601 timestamp,
                or is earlier than the thought's existing ``valid_from`` (an
                inverted validity interval), or either bound has no UTC form
                within the supported ``datetime`` range.

        """
        normalized = validate_iso8601_nullable(valid_until)
        existing_row = await self._get_thought_row(thought_id)
        if existing_row is None:
            raise ThoughtNotFoundError(thought_id)
        # Guard the mutation path explicitly: the invalidate write closes an
        # existing interval, so a caller cannot depend on the model validator
        # firing only at construction time. Reject a ``valid_until`` that would
        # invert the stored interval before the row is updated.
        validate_interval_ordering(existing_row["valid_from"], normalized)
        return await self.update_thought(thought_id, valid_until=normalized)

    async def list_thoughts(
        self,
        *,
        priority: str | None = None,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        min_cycle: int | None = None,
        max_cycle: int | None = None,
        visibility: str | None = None,
        exclude_visibility: str | None = None,
        include_expired: bool = False,
        provenance_filter: MetadataFilter | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThoughtRecord]:
        """List thoughts matching the given filters.

        Provenance querying reuses the same typed
        :class:`~engrava.domain.models.filters.MetadataFilter` machinery as
        metadata filtering, pointed at the ``provenance`` column instead of
        ``metadata_json`` — so provenance is queryable **read-only** with no new
        verb. A ``session_id`` / ``actor_id`` predicate is served by the
        provenance identity index; the descriptive provenance paths
        (``$.retrieval_query`` etc.) are queryable but not indexed. This is a
        **query capability, not a security boundary** — provenance is an
        untrusted hint (see
        :class:`~engrava.domain.models.provenance.ProvenanceContext`) and is
        consulted for no access decision.

        Args:
            priority: Filter by priority level.
            lifecycle_status: Filter by lifecycle status.
            thought_type: Filter by thought type.
            min_cycle: Minimum updated_cycle (inclusive).
            max_cycle: Maximum updated_cycle (inclusive).
            visibility: Include only thoughts with this visibility.
            exclude_visibility: Exclude thoughts with this visibility.
            include_expired: If True, include expired thoughts. Defaults to False.
            provenance_filter: Optional
                :class:`~engrava.domain.models.filters.MetadataFilter` — an
                ``AND`` of typed field predicates over the ``provenance`` JSON
                column (e.g. ``FieldPredicate("$.session_id", FieldOp.EQ,
                "sess-1")``). ``None`` (or an empty filter) leaves the result
                unchanged; a predicate on ``$.session_id`` / ``$.actor_id`` uses
                the provenance identity index. Rows whose ``provenance`` is NULL
                or malformed JSON never match a non-empty filter (the predicate
                is ``json_valid``-guarded).
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of matching thought records.

        """
        clauses: list[str] = []
        params: list[object] = []

        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.datetime.now(datetime.UTC).isoformat())

        if priority is not None:
            clauses.append("priority = ?")
            params.append(priority)
        if lifecycle_status is not None:
            clauses.append("lifecycle_status = ?")
            params.append(lifecycle_status)
        if thought_type is not None:
            clauses.append("thought_type = ?")
            params.append(thought_type)
        if min_cycle is not None:
            clauses.append("updated_cycle >= ?")
            params.append(min_cycle)
        if max_cycle is not None:
            clauses.append("updated_cycle <= ?")
            params.append(max_cycle)
        if visibility is not None:
            clauses.append("visibility = ?")
            params.append(visibility)
        if exclude_visibility is not None:
            clauses.append("visibility != ?")
            params.append(exclude_visibility)

        # Provenance filtering reuses the generic json_extract predicate
        # machinery, pointed at the ``provenance`` column. ``None`` / empty
        # filter contributes nothing, leaving the query path unchanged; a
        # session_id / actor_id predicate is served by the provenance identity
        # index. The whole predicate is json_valid-guarded, so a NULL or
        # malformed provenance row is non-matching for a non-empty filter.
        provenance_clause = compile_effective_predicate(
            provenance_filter, None, column="provenance"
        )
        if provenance_clause is not None:
            fragment, provenance_params = provenance_clause
            clauses.append(fragment)
            params.extend(provenance_params)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM thought{where} ORDER BY updated_cycle DESC LIMIT ? OFFSET ?"  # noqa: S608
        params.extend([limit, offset])

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        thoughts = [self._row_to_thought(r) for r in rows]
        return [await self._hooks.on_retrieve(t) for t in thoughts]

    async def count_thoughts(
        self,
        *,
        lifecycle_status: str | None = None,
        thought_type: str | None = None,
        priority: str | None = None,
        include_expired: bool = False,
    ) -> int:
        """Count thoughts matching the given filters.

        A lightweight alternative to ``list_thoughts`` when only the
        total count is needed (e.g. for a consolidator's early-stop clustering
        guard).

        Args:
            lifecycle_status: Filter by lifecycle status.
            thought_type: Filter by thought type.
            priority: Filter by priority level (e.g. ``"P1"``).
            include_expired: If True, include expired thoughts. Defaults to False.

        Returns:
            Number of thoughts matching the filters.

        """
        clauses: list[str] = []
        params: list[object] = []

        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.datetime.now(datetime.UTC).isoformat())

        if lifecycle_status is not None:
            clauses.append("lifecycle_status = ?")
            params.append(lifecycle_status)
        if thought_type is not None:
            clauses.append("thought_type = ?")
            params.append(thought_type)
        if priority is not None:
            clauses.append("priority = ?")
            params.append(priority)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT COUNT(*) FROM thought{where}"  # noqa: S608

        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought by its ID.

        **The write lock is taken before the journal before-image and the
        embedding rowid are read, not after.** This call opens its own
        ``BEGIN IMMEDIATE`` first (unless a transaction is already open — a
        caller's :meth:`suspend_auto_commit` window, or a raw ``BEGIN``), so
        a concurrent holder of the write lock is waited out first, and only
        then does this call read the before-image and the rowid the delete
        depends on — instead of a snapshot taken before that wait, which
        could carry a since-superseded before-image or target a since-freed
        rowid. There is no ``revision`` guard here to preserve either way: a
        delete has nothing to compare a stale read against, only a row to
        remove, so reading fresh is a pure correctness improvement, not a
        contract change. A failure while reading either value rolls back a
        transaction this call itself opened, never a caller's, via
        :meth:`_rollback_self_opened_transaction`.

        Args:
            thought_id: UUID of the thought to delete.

        Returns:
            True if the thought row was deleted. False if it was not found,
            or if it was found but a trigger silently vetoed the delete of
            the thought row itself (``RAISE(IGNORE)``) — this call reports
            both cases identically, since either way there is nothing to
            purge or journal. A trigger that instead silently vetoes only
            one of the three child deletes (edge / embedding / action) does
            **not** change this return value: the thought row is still gone
            and this reports ``True``, even though that one child row is
            left behind, orphaned, rather than removed with it.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        async with self._write_lock:
            # Sampled before anything below touches the connection — see
            # _delete_thought_atomic's docstring for the ownership test this
            # mirrors. Nothing between this line and the BEGIN IMMEDIATE
            # right after it is anything but a read, so this and that
            # method's own sample cannot disagree.
            opened_transaction = not self._db.in_transaction
            try:
                # Taken before either read below, not after — see this
                # method's docstring. "Already open" also covers a
                # same-task call already inside its own unit (the
                # task-reentrant write lock allows that) — this never
                # begins a second transaction there. Inside the try, not
                # before it: a cancellation delivered after SQLite has
                # executed this BEGIN but before the ``await`` itself
                # returns must still reach the ``except`` below and roll
                # this transaction back, not leave it open.
                if opened_transaction:
                    await self._db.execute("BEGIN IMMEDIATE")

                before_row = (
                    await self._get_thought_row(thought_id) if self._journal is not None else None
                )

                # Capture the embedding rowid *before* the delete removes the
                # row: the vec0 vector table is not reachable by the
                # embedding FK's ON DELETE CASCADE, so the vector must be
                # purged explicitly to avoid a ghost.
                vec_rowid = await self._embedding_rowid_for_thought(thought_id)

                # Parent delete and explicit child deletes as one atomic
                # unit — see _delete_thought_atomic for why the parent goes
                # first (a user's own BEFORE DELETE trigger must still see
                # the children when it checks for them) and why the child
                # deletes still run explicitly rather than trusting ON
                # DELETE CASCADE (a store on a pre-core-12 schema, or a
                # connection with enforcement off, has no cascade to trust).
                #
                # The delete, the vector purge and the journal entry are one
                # further failure-atomic unit on top of the reads above, via
                # _write_readback_savepoint (see update_thought for what
                # that protects against): a failed or cancelled journal
                # append here unwinds the delete too, rather than leaving it
                # pending in the open transaction with no savepoint left to
                # protect it — _delete_thought_atomic's own savepoint has
                # already released by the time this call reaches its
                # append. _delete_thought_atomic nests inside this unit
                # unchanged: since this block already opened the
                # transaction (when one was not already open) before that
                # call runs, its own `opened_transaction` sample sees one
                # already open and never ends it on the veto path itself,
                # leaving that to this call's own cleanup below as usual.
                # A failure here — including inside the savepoint unit —
                # is caught below, which is why that unit's own unwind does
                # not also need to close this call's outer transaction: it
                # sees one already open (`opened_transaction` is `False`
                # there) and correctly leaves that to this `except`.
                async with self._write_readback_savepoint("delete_thought", begin="IMMEDIATE"):
                    result = await self._delete_thought_atomic(thought_id)
                    deleted = result.deleted

                    if deleted:
                        await self._purge_orphan_vector(vec_rowid)

                    if deleted and self._journal is not None and before_row is not None:
                        await self._journal.append(
                            mutation_type="DELETE_THOUGHT",
                            target_id=thought_id,
                            delta={
                                "before": self._row_to_thought(before_row).model_dump(mode="json"),
                                "after": None,
                            },
                        )
            except BaseException as exc:
                await self._rollback_self_opened_transaction(
                    opened_transaction=opened_transaction, exc=exc
                )
                raise

            if result.wrote_anything:
                # A real write happened — the thought itself, an orphan sweep
                # on a never-existed id, or both — commit it, exactly as
                # before. ``deleted`` alone would miss the orphan-sweep-only
                # case (see _DeleteAtomicResult).
                await self._maybe_commit()
            elif opened_transaction and self._db.in_transaction:
                # Nothing was written at all — the id never matched a row and
                # there was nothing orphaned to sweep, or a trigger vetoed the
                # delete — so this call has nothing of its own to make
                # durable. `_delete_thought_atomic` already ends a transaction
                # *it* opened on the veto path; this closes the "never
                # existed, nothing to sweep" path, where that method leaves
                # its own self-opened, still-empty transaction open for this
                # call to close. Either way, only a transaction this call
                # itself opened is ended here, and always with a rollback — a
                # transaction the caller already held when this call started
                # (``opened_transaction`` is ``False``) is never touched: a
                # commit would durably apply the caller's own unrelated
                # pending work, which this call was never asked to do.
                await self._db.rollback()
        return deleted

    # ------------------------------------------------------------------
    # EdgeRecord CRUD
    # ------------------------------------------------------------------

    async def create_edge(self, edge: EdgeRecord) -> EdgeRecord:
        """Persist a new edge record.

        The schema (core-12+) enforces an ``ON DELETE CASCADE`` foreign
        key on both edge endpoints. Inserting an edge whose endpoints
        do not resolve to existing thoughts raises
        :class:`ReferentialIntegrityError` — the raw
        ``sqlite3.IntegrityError`` is intentionally not surfaced.

        Args:
            edge: The edge record to create.

        Returns:
            The persisted edge record.

        Raises:
            DuplicateEdgeError: When the same directed endpoints and edge type
                already identify a persisted relationship.
            ReferentialIntegrityError: When ``from_thought_id`` or
                ``to_thought_id`` does not match any persisted thought.
            ValueError: When ``edge.metadata`` violates the shared metadata
                contract (a non-scalar / list value, a non-finite float, or a
                serialized size over the 64 KiB hard limit).

        """
        _validate_metadata(edge.metadata)
        async with self._write_lock:
            # The insert and its journal entry are one failure-atomic unit,
            # via _write_readback_savepoint (see update_thought for what that
            # protects against): a failed or cancelled journal append here
            # unwinds the insert too, instead of leaving it pending with no
            # savepoint of its own protecting it.
            async with self._write_readback_savepoint("create_edge", begin="IMMEDIATE"):
                try:
                    await self._db.execute(
                        "INSERT INTO edge "
                        "(edge_id, from_thought_id, to_thought_id, edge_type, weight, "
                        " created_cycle, source, decay_multiplier, valid_from, valid_until, "
                        " metadata_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            edge.edge_id,
                            edge.from_thought_id,
                            edge.to_thought_id,
                            edge.edge_type.value,
                            edge.weight,
                            edge.created_cycle,
                            edge.source.value,
                            edge.decay_multiplier,
                            edge.valid_from,
                            edge.valid_until,
                            json.dumps(edge.metadata, ensure_ascii=False),
                        ),
                    )
                except aiosqlite.IntegrityError as exc:
                    # Classify structurally by the extended result code BEFORE any
                    # existence probe: a FOREIGN KEY failure maps to the domain
                    # wrapper, and only a UNIQUE / PRIMARY KEY failure is a
                    # candidate duplicate. A CHECK / NOT NULL / trigger abort (even
                    # one whose message mentions "foreign key") is neither and
                    # propagates unchanged.
                    if _is_foreign_key_violation(exc):
                        column, referenced = await self._identify_orphan_endpoint(edge)
                        raise ReferentialIntegrityError(
                            entity_type="edge",
                            column=column,
                            referenced_id=referenced,
                        ) from exc
                    if _is_unique_violation(exc):
                        # Confirm the collision is the directed-endpoint + type
                        # identity (the conflict-as-reuse case) rather than another
                        # UNIQUE constraint, such as a caller-supplied duplicate
                        # ``edge_id``, which keeps its own contract and propagates.
                        duplicate_cursor = await self._db.execute(
                            "SELECT 1 FROM edge "
                            "WHERE from_thought_id = ? AND to_thought_id = ? AND edge_type = ? "
                            "LIMIT 1",
                            (edge.from_thought_id, edge.to_thought_id, edge.edge_type.value),
                        )
                        if await duplicate_cursor.fetchone() is not None:
                            raise DuplicateEdgeError(
                                edge.from_thought_id,
                                edge.to_thought_id,
                                edge.edge_type.value,
                            ) from exc
                    # Preserve every other integrity failure (a non-duplicate
                    # UNIQUE, a CHECK, NOT NULL, or trigger abort) for its own
                    # contract.
                    raise

                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="INSERT_EDGE",
                        target_id=edge.edge_id,
                        delta={"before": None, "after": edge.model_dump(mode="json")},
                    )

            await self._maybe_commit()
        return edge

    async def update_edge(self, edge_id: str, **changes: object) -> EdgeRecord:
        """Update an edge by its ID.

        Writes **only the columns whose value this edit changes**, so a field
        another writer set since the row was read is not rolled back. An edit
        that changes nothing writes no column of its own — though an attempt
        is still made to journal it (as a before == after entry) when
        journaling is enabled, since journaling is not conditioned on there
        being a column change; like the column write above, a trigger on the
        journal table can silently veto that insert too (``RAISE(IGNORE)``),
        in which case nothing was journaled either. Either way, this call
        commits only when the connection's own change counter shows
        something of its own actually landed — a column, a journal entry, or
        both, not merely attempted — and otherwise leaves a caller's own
        open transaction exactly as it found it; see :meth:`delete_thought`
        for the same rule applied to a write-free outcome. The record
        returned is read back from storage after the write — the row that exists, not the
        one the call intended to write — and the journal ``after`` image is the
        same read-back.

        **The write (when there is one) carries a ``revision`` guard**, exactly
        like :meth:`update_thought`'s: the row's ``revision`` at the start of
        this call is checked and incremented atomically by the same ``UPDATE``,
        so any other guarded write that landed on this row since it was read
        here — or a delete — makes the ``UPDATE`` match no row, and this call
        raises ``StaleDataError`` rather than silently overwriting. An edit
        that changes nothing issues no ``UPDATE`` at all (see below), so it
        cannot go stale — there is nothing for it to be stale against. It is a
        read-modify-write like :meth:`update_thought`, and shares the same
        task-reentrant :attr:`_write_lock` critical section: the
        read, the merge, and the write are atomic with respect to every other
        guarded write on this instance, so a competing edit to a field this
        call also writes can no longer land between this call's own read and
        write.

        **The write (when there is one), its confirming read-back, and its
        journal entry are one failure-atomic unit**, via
        :meth:`_write_readback_savepoint` — see :meth:`update_thought` for
        what that protects against and how it treats a caller-owned
        transaction. The ``UPDATE`` also now captures
        its cursor and rejects a zero-row match immediately, rather than
        trusting the read-back alone: without that check, a row deleted
        after the initial read (so the ``UPDATE`` matches nothing) and
        re-created under the same ``edge_id`` before the read-back runs would
        be reported as though this call had updated it, when it had written
        nothing at all.

        Args:
            edge_id: UUID of the edge to update.
            **changes: Fields to update.

        Returns:
            The stored edge record, as persisted by this update.

        Raises:
            ValueError: If the edge does not exist at the initial read, or if
                the merged ``metadata`` violates the shared metadata contract
                (a non-scalar / list value, a non-finite float, or a
                serialized size over the 64 KiB hard limit).
            StaleDataError: If a real change's guarded ``UPDATE`` matches no
                row — another guarded write landed on this row since it was
                read here (including a delete, and including a delete
                followed by a different row recreated under the same
                ``edge_id`` before the read-back could run). Nothing of this
                update is written when it is raised.
            WriteContentionError: The guarded write could not proceed because
                the connection reported lock contention.

        """
        async with self._write_lock:
            # Sampled before anything below touches the connection — same
            # ownership test as delete_thought / cleanup_expired.
            opened_transaction = not self._db.in_transaction

            current_row = await self._get_edge_row(edge_id)
            if current_row is None:
                msg = f"Edge not found: {edge_id}"
                raise ValueError(msg)

            current = _row_to_edge(current_row)
            expected_revision = int(current_row["revision"])
            updated = type(current).model_validate({**current.model_dump(mode="json"), **changes})
            _validate_metadata(updated.metadata)

            before = _edge_to_core_columns(current)
            columns = {
                name: value
                for name, value in _edge_to_core_columns(updated).items()
                if before[name] != value
            }
            async with self._write_readback_savepoint("update_edge_readback", begin="DEFERRED"):
                # Sampled *inside* the savepoint, immediately after its own
                # ``SAVEPOINT`` statement -- not before it. A caller-owned
                # transaction with its own pending write to an FTS-indexed
                # column (``thought.essence`` / ``thought.content``) can
                # leave some of that write's own accounting against
                # ``total_changes`` until the next SAVEPOINT/BEGIN executes
                # on this connection (an FTS5 shadow-table quirk, not
                # anything this call did); sampling after this savepoint's
                # own ``SAVEPOINT`` lets that settle on the caller's side of
                # the line instead of being folded into this call's delta.
                changes_before = self._db.total_changes
                if columns:
                    cursor = await self._execute_revision_guarded_write(
                        _build_update_sql(
                            "edge",
                            columns,
                            "edge_id = ? AND revision = ?",
                            bump_column="revision",
                        ),
                        (*columns.values(), edge_id, expected_revision),
                        operation="update_edge",
                    )
                    if cursor.rowcount == 0:
                        raise StaleDataError(
                            entity_type="EdgeRecord",
                            entity_id=edge_id,
                            expected_version=expected_revision,
                        )

                persisted = await self._read_back_edge(edge_id)

                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="UPDATE_EDGE",
                        target_id=edge_id,
                        delta={
                            "before": current.model_dump(mode="json"),
                            "after": persisted.model_dump(mode="json"),
                        },
                    )
            # An edit that changes nothing (``columns`` empty) writes no
            # column of its own *unless* journaling is enabled: the journal
            # entry above is itself a real row insert (JournalWriter.append
            # docstring: "The caller is responsible for committing... or
            # relying on the store's _maybe_commit") that needs the same
            # commit a real column change would need. So this tracks whether
            # *anything* this call did needs to be made durable, not just
            # whether ``columns`` was non-empty. It is not simply "``columns``
            # was non-empty, or the journal was called" either: ``append``
            # does not check whether its own INSERT actually inserted, so a
            # trigger on the journal table can veto it with ``RAISE(IGNORE)``
            # while the call still happened. ``total_changes`` reflects
            # whether the column write and/or the journal insert actually
            # landed, not merely whether either was attempted.
            wrote_anything = self._db.total_changes > changes_before

            if wrote_anything:
                await self._maybe_commit()
            elif opened_transaction and self._db.in_transaction:
                # Truly nothing was written (no column changed, no journal
                # entry) — the read-back savepoint above may still have opened
                # an otherwise-empty transaction. Close only that one, and
                # only with a rollback: a transaction the caller already held
                # is left exactly as it was.
                await self._db.rollback()
        return persisted

    async def _read_back_edge(self, edge_id: str) -> EdgeRecord:
        """Re-read an edge a write just landed on.

        The counterpart of :py:meth:`_read_back_thought` for edges: once an
        update writes only the columns it owns, the stored row is the only
        thing that can be reported to the caller or journaled.

        Args:
            edge_id: UUID of the edge.

        Returns:
            The edge as it is stored now.

        Raises:
            ValueError: If the row no longer exists, so the write cannot be
                confirmed and no record may be reported for it.

        """
        row = await self._get_edge_row(edge_id)
        if row is None:
            msg = f"Edge not found: {edge_id}"
            raise ValueError(msg)
        return _row_to_edge(row)

    async def invalidate_edge(
        self,
        edge_id: str,
        valid_until: str,
    ) -> EdgeRecord:
        """Close an edge's valid-time interval at the given instant.

        Sets the edge's ``valid_until`` to ``valid_until``, marking the end
        of the window during which the relation is considered true in the
        world. Like :meth:`invalidate_thought`, this is a deterministic,
        valid-time-only operation:

        * It is **not** a delete — the edge row remains stored and
          retrievable; only the valid-time upper bound changes.
        * It performs **no** similarity search, automatic invalidation, or
          model inference of any kind.
        * It is **idempotent**: invalidating with the same ``valid_until``
          twice converges to the same stored value.

        Args:
            edge_id: UUID of the edge to invalidate.
            valid_until: ISO-8601 instant at which the relation stops being
                valid. Stored as the edge's ``valid_until`` bound.

        Returns:
            The updated edge record.

        Raises:
            ValueError: If the edge does not exist, ``valid_until`` is not a
                valid ISO-8601 timestamp, or ``valid_until`` is earlier than the
                edge's existing ``valid_from`` (an inverted validity interval),
                or either bound has no UTC form within the supported
                ``datetime`` range.
            StaleDataError: If the guarded write matches no row — see
                :meth:`update_edge`, which this delegates to.
            WriteContentionError: The guarded write could not proceed because
                the connection reported lock contention.

        """
        normalized = validate_iso8601_nullable(valid_until)
        existing_row = await self._get_edge_row(edge_id)
        if existing_row is None:
            msg = f"Edge not found: {edge_id}"
            raise ValueError(msg)
        # Guard the mutation path explicitly: the invalidate write closes an
        # existing interval, so a caller cannot depend on the model validator
        # firing only at construction time. Reject a ``valid_until`` that would
        # invert the stored interval before the row is updated.
        validate_interval_ordering(existing_row["valid_from"], normalized)
        return await self.update_edge(edge_id, valid_until=normalized)

    async def get_edges(
        self,
        thought_id: str,
        *,
        direction: str = "BOTH",
        limit: int | None = None,
    ) -> list[EdgeRecord]:
        """Retrieve edges connected to a thought.

        Args:
            thought_id: UUID of the thought.
            direction: 'IN', 'OUT', or 'BOTH'.
            limit: If given, bound the result to this many edges at the SQL
                layer, keeping the highest-``weight`` ones first. Must be
                ``0`` or a positive integer; ``0`` returns an empty list.
                ``None`` (the default) returns every matching edge, unordered,
                exactly as before this parameter existed — callers that need
                the complete adjacency (e.g. checking every existing
                connection before creating a new one) must keep passing
                ``None``; only pass a bound where the caller's own contract is
                already "the top-N most relevant neighbours", not "all of
                them".

        Returns:
            List of matching edge records. Ordered by ``weight`` descending
            when ``limit`` is given, otherwise in storage order.

        Raises:
            ValueError: If ``limit`` is a ``bool`` or is negative. ``bool``
                is rejected even though it subclasses ``int``: it is never a
                meaningful edge count, only a type mismatch a type checker
                would miss.

        """
        if limit is not None and (isinstance(limit, bool) or limit < 0):
            msg = f"limit must be a non-negative integer, got {limit!r}"
            raise ValueError(msg)

        if direction == "OUT":
            sql = "SELECT * FROM edge WHERE from_thought_id = ?"
            params: list[object] = [thought_id]
        elif direction == "IN":
            sql = "SELECT * FROM edge WHERE to_thought_id = ?"
            params = [thought_id]
        else:
            sql = "SELECT * FROM edge WHERE from_thought_id = ? OR to_thought_id = ?"
            params = [thought_id, thought_id]

        if limit is not None:
            sql += " ORDER BY weight DESC LIMIT ?"
            params.append(limit)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by its ID.

        **The write lock is taken before the before-image is read, not
        after.** This call opens its own ``BEGIN IMMEDIATE`` first (unless a
        transaction is already open), so a concurrent holder of the write
        lock is waited out first, and only then does this call read the
        before-image the journal records — instead of a snapshot taken
        before that wait, which could carry a since-superseded before-image.
        There is no ``revision`` guard here to preserve: a delete has
        nothing to compare a stale read against, only a row to remove, so
        reading fresh is a pure correctness improvement. A failure while
        reading the before-image rolls back a transaction this call itself
        opened, never a caller's, via
        :meth:`_rollback_self_opened_transaction`.

        **The delete and its journal entry are then a further failure-atomic
        unit**, via :meth:`_write_readback_savepoint`. This call commits
        only when its own ``DELETE`` actually removed a row; on a missing
        id, or on a failure reading the before-image, it rolls back only a
        transaction it opened itself, exactly like :meth:`delete_thought` —
        it never commits a caller's already-open transaction for a call that
        deleted nothing.

        Args:
            edge_id: UUID of the edge to delete.

        Returns:
            True if the edge was deleted, False if not found.

        """
        async with self._write_lock:
            # Sampled before anything below touches the connection — same
            # ownership test as delete_thought.
            opened_transaction = not self._db.in_transaction
            try:
                # Taken before the read, not after it — see this method's
                # docstring. Inside the try, not before it: a cancellation
                # delivered after SQLite has executed this BEGIN but before
                # the ``await`` itself returns must still reach the
                # ``except`` below and roll this transaction back, not
                # leave it open.
                if opened_transaction:
                    await self._db.execute("BEGIN IMMEDIATE")

                before_row = (
                    await self._get_edge_row(edge_id) if self._journal is not None else None
                )

                async with self._write_readback_savepoint("delete_edge", begin="IMMEDIATE"):
                    cursor = await self._db.execute(
                        "DELETE FROM edge WHERE edge_id = ?", (edge_id,)
                    )
                    deleted = cursor.rowcount > 0

                    if deleted and self._journal is not None and before_row is not None:
                        await self._journal.append(
                            mutation_type="DELETE_EDGE",
                            target_id=edge_id,
                            delta={
                                "before": dict(before_row),
                                "after": None,
                            },
                        )
            except BaseException as exc:
                await self._rollback_self_opened_transaction(
                    opened_transaction=opened_transaction, exc=exc
                )
                raise

            if deleted:
                await self._maybe_commit()
            elif opened_transaction and self._db.in_transaction:
                # Nothing was written — the id never matched a row — so this
                # call has nothing of its own to make durable. Close only a
                # transaction this call itself opened, never one a caller
                # already held: see delete_thought for the same reasoning.
                await self._db.rollback()
        return deleted

    async def list_edges(
        self,
        *,
        edge_type: EdgeType | None = None,
        source: KnowledgeSource | None = None,
        filters: MetadataFilter | None = None,
        limit: int = 5000,
    ) -> list[EdgeRecord]:
        """List edges matching optional filters.

        Edge-metadata filtering reuses the same typed
        :class:`~engrava.domain.models.filters.MetadataFilter` machinery as
        thought-metadata filtering, pointed at the edge ``metadata_json`` column.
        It is a **query capability, not a security boundary** — it enforces
        nothing and is bypassable.

        Args:
            edge_type: If given, restrict to this edge type.
            source: If given, restrict to this knowledge source.
            filters: Optional
                :class:`~engrava.domain.models.filters.MetadataFilter` — an
                ``AND`` of typed field predicates over the edge ``metadata_json``
                column (e.g. ``FieldPredicate("$.subtype", FieldOp.EQ,
                "supports")``). ``None`` (or an empty filter) leaves the result
                unchanged. Inherits the shipped semantics verbatim: JSONPath
                ``$`` / ``$.key`` / ``$[0]`` only, operators EQ and IN only,
                AND-conjunction, a 250-predicate cap. Edges whose
                ``metadata_json`` is malformed JSON never match a non-empty
                filter (the predicate is ``json_valid``-guarded).
            limit: Maximum number of edges to return.

        Returns:
            List of matching edge records, ordered by ``created_cycle`` DESC.

        """
        clauses: list[str] = []
        params: list[object] = []

        if edge_type is not None:
            clauses.append("edge_type = ?")
            params.append(str(edge_type))
        if source is not None:
            clauses.append("source = ?")
            params.append(str(source))

        # Metadata filtering reuses the generic json_extract predicate
        # machinery, pointed at the edge ``metadata_json`` column. Edges have no
        # visibility axis, so ``visibility=None``. A None / empty filter
        # contributes nothing, leaving the query path unchanged; the whole
        # predicate is json_valid-guarded, so a malformed metadata row is
        # non-matching for a non-empty filter.
        metadata_clause = compile_effective_predicate(filters, None, column="metadata_json")
        if metadata_clause is not None:
            fragment, metadata_params = metadata_clause
            clauses.append(fragment)
            params.extend(metadata_params)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM edge{where} ORDER BY created_cycle DESC LIMIT ?"  # noqa: S608
        params.append(limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def thought_exists_by_source(
        self,
        *,
        source: str,
        thought_type_value: str,
    ) -> bool:
        """Check whether any thought with an exact source and thought_type exists.

        Not an O(1) index lookup: the schema indexes ``thought(thought_type)``
        but not ``thought(source)``, so this scans every row of the matched
        ``thought_type`` for the ``source`` filter — O(number of rows of that
        type), not O(1). Called per cluster in ``_create_reflections``, so
        the cost scales with the REFLECTION count on a store with many of
        them.

        Args:
            source: Exact ``source`` field value to match.
            thought_type_value: ``thought_type`` enum string (e.g.
                ``"REFLECTION"``).

        Returns:
            ``True`` if at least one matching thought exists.

        Examples:
            >>> exists = await store.thought_exists_by_source(
            ...     source="dreaming:abc123",
            ...     thought_type_value="REFLECTION",
            ... )  # doctest: +SKIP

        """
        cursor = await self._db.execute(
            "SELECT thought_id FROM thought WHERE thought_type = ? AND source = ? LIMIT 1",
            (thought_type_value, source),
        )
        return await cursor.fetchone() is not None

    async def consolidated_source_statuses(self, reflection_id: str) -> list[str]:
        """Return the lifecycle statuses of a REFLECTION's source thoughts.

        Resolves the ``CONSOLIDATED_FROM`` edges leaving ``reflection_id``
        and returns the ``lifecycle_status`` of each source thought, using a
        single indexed join rather than a per-edge lookup. The result is the
        liveness picture an orphan sweep needs: a REFLECTION is orphaned when
        this list is non-empty and contains no ``ACTIVE`` entry.

        Args:
            reflection_id: UUID of the REFLECTION whose sources to inspect.

        Returns:
            One lifecycle-status string per resolvable source thought, in no
            particular order. Empty when the REFLECTION has no
            ``CONSOLIDATED_FROM`` edges (or none resolve to a live thought
            row).

        """
        cursor = await self._db.execute(
            "SELECT t.lifecycle_status AS lifecycle_status "
            "FROM edge e "
            "JOIN thought t ON e.to_thought_id = t.thought_id "
            "WHERE e.from_thought_id = ? AND e.edge_type = 'CONSOLIDATED_FROM'",
            (reflection_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["lifecycle_status"]) for row in rows]

    async def reflections_consolidated_from(self, source_id: str) -> list[str]:
        """Return REFLECTION ids that were consolidated from a source thought.

        Resolves the inbound ``CONSOLIDATED_FROM`` edges of ``source_id`` and
        keeps only the parents whose ``thought_type`` is ``REFLECTION``. Used
        by the re-bind path to find the syntheses that must be refreshed when
        a member's embedding changes.

        Args:
            source_id: UUID of the source thought.

        Returns:
            Distinct REFLECTION ids that consolidated ``source_id`` as a
            member. Empty when the source belongs to no REFLECTION.

        """
        cursor = await self._db.execute(
            "SELECT DISTINCT t.thought_id AS thought_id "
            "FROM edge e "
            "JOIN thought t ON e.from_thought_id = t.thought_id "
            "WHERE e.to_thought_id = ? AND e.edge_type = 'CONSOLIDATED_FROM' "
            "AND t.thought_type = 'REFLECTION'",
            (source_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["thought_id"]) for row in rows]

    async def consolidated_member_ids(self, reflection_id: str) -> list[str]:
        """Return the source-thought ids a REFLECTION was consolidated from.

        Args:
            reflection_id: UUID of the REFLECTION.

        Returns:
            The ``to_thought_id`` of each ``CONSOLIDATED_FROM`` edge leaving
            ``reflection_id``. Empty when the REFLECTION has no such edges.

        """
        cursor = await self._db.execute(
            "SELECT to_thought_id FROM edge "
            "WHERE from_thought_id = ? AND edge_type = 'CONSOLIDATED_FROM'",
            (reflection_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["to_thought_id"]) for row in rows]

    # ------------------------------------------------------------------
    # Auto-embed helper
    # ------------------------------------------------------------------

    def _on_auto_embed_failure(self, thought_id: str, exc: Exception) -> NoReturn:
        """Surface an auto-embed provider failure, never silently.

        What is certain regardless of caller: the embedding was not
        produced. What happens to the thought row itself is two
        independent questions.

        **Is the row durable yet?** If this call does not own the
        outermost transaction — nested inside a caller's own
        ``suspend_auto_commit()`` window — nothing is durable yet, on any
        path. The outermost window's exit decides — if the caller catches
        this and that outer window exits cleanly, the rows commit (for a
        batch, every row it inserted, see :meth:`bulk_store`); uncaught,
        they roll back. This holds for every path, single-item and batch
        alike.

        If this call does own the outermost transaction, the path
        decides:

        * ``create_thought`` and ``update_thought`` have already
          committed by the time this runs, so the failure cannot undo
          them.
        * Called from :meth:`_batch_embed_thoughts` on its own (a
          standalone ``bulk_store``), the insert loop has already
          finished, but the whole batch (inserts plus the trailing embed
          call) shares one transaction, so this failure rolls the
          *entire* batch back instead — every row in it, this one
          included — and none of them persist.

        **What is left behind?** Determined by the path, and only
        meaningful for whatever actually committed: ``create_thought``
        leaves no embedding row at all. ``update_thought`` (only reached
        here when ``essence``/``content`` changed) leaves any embedding
        the row already had in place — if the update committed, that
        embedding is now stale against the new content, and the row is
        still findable by vector search against that outdated vector; if
        the row had no embedding before, it still has none, and remains
        unfindable by vector search. If the update instead rolled back —
        only possible when this call is nested inside a caller's own window
        and the caller lets the failure escape it — the durable state
        reverts to whatever existed when that *outermost* window opened,
        not merely to what this call itself started from: an earlier write
        to the same thought inside the same window is undone right along
        with it. If the thought was created inside that same window, it no
        longer exists at all afterward — there is nothing to be "left
        behind". If it already existed before the window opened, it
        reverts to that pre-window state, and the retained embedding
        matches it only if it already did: an earlier update on the same
        thought, before this window ever opened, whose own re-embed failed
        can already have left that pre-window state stale, and this
        rollback neither detects nor repairs that. A standalone
        ``bulk_store``'s rollback leaves nothing behind at all.

        See ``docs/api-reference.md``'s ``bulk_store`` and
        ``EmbeddingGenerationError`` entries for the fuller treatment. This
        handler makes the certain half of that outcome visible either way:
        it always emits a ``WARNING`` naming the thought id and the
        provider error's type — never its message — then either re-raises
        the provider's own exception (default, byte-identical to prior
        behaviour) or, under ``require_embedding=True``, raises a typed
        :class:`EmbeddingGenerationError` — the opt-in fail-fast.

        Args:
            thought_id: UUID of the thought whose embedding failed.
            exc: The exception raised by the embedding provider.

        Raises:
            EmbeddingGenerationError: When ``require_embedding`` is ``True``.
            Exception: The provider's original exception otherwise.

        """
        logger.warning(
            "Auto-embed failed for thought %s: %s. The embedding was not "
            "produced. Whether the thought row survives, and in what "
            "state, depends on the call that raised this and its "
            "surrounding transaction — see docs/api-reference.md for the "
            "specific outcomes.",
            thought_id,
            type(exc).__name__,
        )
        if self._require_embedding:
            raise EmbeddingGenerationError(thought_id, str(exc)) from exc
        raise exc

    async def _auto_embed_thought(self, thought: ThoughtRecord) -> None:
        """Generate and store an embedding for a thought via the provider.

        Builds the embed payload via :func:`_build_embed_input` (which drops a
        prefix-redundant ``essence`` to avoid double-counting the opening),
        embeds it via the configured provider, and persists the vector —
        but only if the thought's stored ``essence``/``content`` still match
        what was just embedded (see "Staleness" below).

        An ``Exception`` that escapes the guarded call to
        :func:`_embed_document` below is never silent: it is routed through
        :meth:`_on_auto_embed_failure`, which logs a ``WARNING`` naming the
        thought and the provider error's type, then re-raises the provider
        error (default) or a typed :class:`EmbeddingGenerationError` (when
        ``require_embedding=True``).
        That guarantee is scoped to this one call, not to provider failures
        in general: ``provider.model_name`` is read afterward, outside the
        ``try``/``except``, to pass to :meth:`store_embedding`, so a
        provider whose ``model_name`` property raises skips the routing
        entirely regardless of ``require_embedding`` or exception type — no
        ``WARNING``, no :class:`EmbeddingGenerationError`. A
        ``BaseException`` that is not an ``Exception`` raised *from inside*
        the guarded call — :class:`asyncio.CancelledError` is the one that
        matters in practice — is also not caught by the ``except Exception``
        here, so it skips the routing too: no ``WARNING`` is logged and it
        is never normalised into :class:`EmbeddingGenerationError`. Either
        way it still propagates, so it is not swallowed, only unlogged and
        untyped.
        Whether the thought row is durable yet depends on whether this
        call owns the outermost transaction: on its own, the insert or
        update has already committed by the time this runs, so the
        failure cannot undo it; nested inside the caller's own
        ``suspend_auto_commit()`` window, nothing is durable yet — that
        window's exit decides. What is left behind is path-specific: a
        fresh ``create_thought`` leaves no embedding row at all, so the
        row stays unfindable by vector search; an ``update_thought``
        leaves any embedding the row already had in place — stale and
        still findable by vector search only if the row had one before
        and the update committed.

        **Staleness.** The provider call above is a slow, arbitrary network
        round trip made *outside* :attr:`_write_lock` (see the call sites in
        :meth:`update_thought` and :meth:`_finish_create_thought`), so by the
        time it returns, an unrelated later write may already have changed
        this same thought's ``essence``/``content`` — or deleted the row
        outright — and installed its own, newer vector. Installing this
        call's vector unconditionally would silently overwrite that newer
        vector with one computed from now-superseded content, even though
        the durable text and FTS index already reflect the later write.
        Guarded against here, atomically with the install: immediately
        before calling :meth:`store_embedding`, the thought's *current*
        stored ``essence``/``content`` are re-read under :attr:`_write_lock`
        and compared against what was actually embedded above. A mismatch —
        including the row no longer existing — means this completion is
        stale: it is dropped silently rather than installed, on the
        assumption that either the later write already triggered its own,
        current auto-embed, or (if it did not, e.g. a metadata-only write)
        the vector already in place is the one that write left there, which
        is correct for its content. Comparing content rather than a revision
        counter also covers thought-id reuse after a delete: a revision
        counter restarts at zero on the recreated row, which could
        coincidentally equal a revision captured before the deletion, but
        the *content* only matches when it is genuinely the same content —
        installing in that case is correct, not stale, because the vector
        this call computed is valid for whatever row currently holds that
        content.

        Args:
            thought: The thought to embed.

        Raises:
            EmbeddingGenerationError: When embedding fails and
                ``require_embedding`` is ``True``.

        """
        provider = self._embedding_provider
        if provider is None:
            return  # pragma: no cover
        text = _build_embed_input(thought.essence, thought.content)

        try:
            vector = await _embed_document(provider, text)
        except Exception as exc:  # noqa: BLE001 -- provider may raise any type; re-raised in handler
            self._on_auto_embed_failure(thought.thought_id, exc)

        model_name = provider.model_name
        async with self._write_lock:
            current_row = await self._get_thought_row(thought.thought_id)
            if (
                current_row is None
                or current_row["essence"] != thought.essence
                or current_row["content"] != thought.content
            ):
                logger.debug(
                    "Dropping stale auto-embed completion for %s: essence/content "
                    "changed (or the row was deleted) since this embed was scheduled.",
                    thought.thought_id,
                )
                return
            await self.store_embedding(
                thought.thought_id,
                vector,
                model_name=model_name,
            )

    async def _rebind_consolidated_reflections(self, source_id: str) -> int:
        """Recompute the centroids of REFLECTIONs that summarize a source.

        Called after a source thought is re-embedded (essence/content
        evolve). A REFLECTION is a synthesis bound to the live state of its
        cluster, so when a member's vector changes the REFLECTION's centroid
        must be recomputed from the current member vectors rather than stay
        frozen at its creation-time value. The recompute reuses the same
        deterministic L2-normalized mean as REFLECTION creation
        (:func:`compute_centroid`) and overwrites the centroid in place via
        the ``store_embedding`` upsert — no schema change, no model call.

        This is intentionally *not* invoked on metadata-only edits: it is
        called only from the essence/content re-embed branch of
        :meth:`update_thought`, so metadata/priority churn leaves dependent
        REFLECTION centroids untouched.

        **Same asynchronous-completion shape as the auto-embed above, closed
        differently.** Like :meth:`_auto_embed_thought`, this runs outside
        :attr:`_write_lock` (:meth:`update_thought` releases it before
        either call), so two members of the same REFLECTION updated close
        together can each trigger a rebind of that one REFLECTION
        concurrently. Unlike the provider call above, though, nothing here
        is slow, arbitrary network I/O — every step reading a member vector
        is a local, in-process database read — so each reflection's whole
        read-recompute-store span is wrapped in one :attr:`_write_lock`
        acquisition instead of using a captured-then-compared identity. That
        makes the span atomic with respect to every other guarded write,
        including a concurrent rebind of the same REFLECTION: whichever
        rebind actually runs always reads the member vectors as they stand
        at that moment, not a snapshot captured earlier, so there is no
        window in which an earlier-read, now-superseded centroid can land
        after a later one already installed a fresher vector.

        Args:
            source_id: UUID of the source thought that was just re-embedded.

        Returns:
            Number of REFLECTION centroids recomputed.

        """
        reflection_ids = await self.reflections_consolidated_from(source_id)
        if not reflection_ids:
            return 0

        rebound = 0
        for reflection_id in reflection_ids:
            async with self._write_lock:
                member_ids = await self.consolidated_member_ids(reflection_id)
                member_vectors: list[list[float]] = []
                for member_id in member_ids:
                    embedding = await self.get_embedding(member_id)
                    if embedding is None:
                        continue
                    member_vectors.append(
                        list(struct.unpack(f"{embedding.dimension}f", embedding.vector_blob)),
                    )
                if not member_vectors:
                    continue
                centroid = compute_centroid(member_vectors)
                await self.store_embedding(
                    reflection_id,
                    centroid,
                    model_name=CENTROID_MODEL_NAME,
                )
                rebound += 1
        return rebound

    # ------------------------------------------------------------------
    # EmbeddingRecord CRUD
    # ------------------------------------------------------------------

    async def store_embedding(
        self,
        thought_id: str,
        vector: list[float],
        *,
        model_name: str = "all-MiniLM-L12-v2",
        embedding_id: str | None = None,
    ) -> EmbeddingRecord:
        """Persist an embedding vector for a thought.

        On first call, locks the embedding model in ``_metadata`` — atomically
        with this call's own write, so a first call that fails (wrong
        dimension, an invalid ``thought_id``) locks nothing. Subsequent calls
        verify the model matches the stored one.

        Args:
            thought_id: UUID of the thought that owns this embedding.
            vector: Embedding vector as a list of floats.
            model_name: Embedding model identifier.
            embedding_id: Optional explicit ID; generated if omitted.

        Returns:
            The persisted EmbeddingRecord.

        Raises:
            EmbeddingModelMismatchError: When model_name, its dimension, or
                its document-prefix fingerprint does not match the one
                already stored in ``_metadata`` — checked on every call, not
                only the first.
            ConnectionQuarantinedError: When the base row write succeeds but
                the vec0 upsert fails and the resulting unwind cannot itself
                be trusted to have undone it cleanly (see
                :meth:`_write_readback_savepoint`).

        """
        async with self._write_lock:
            eid = embedding_id or f"emb-{_uuid.uuid5(_uuid.NAMESPACE_URL, thought_id)}"
            dimension = len(vector)
            blob = struct.pack(f"{dimension}f", *vector)
            created_at = datetime.datetime.now(datetime.UTC).isoformat()

            # The identity lock, the base ``embedding`` row and the ``vec0``
            # upsert are one failure-atomic unit: without this, a first call
            # on an empty corpus that locked the model identity but then
            # failed its own base/vec0 write (a wrong-dimension vector, an
            # invalid owner) left that identity committed regardless — a
            # corrected retry would then fail verification against a corpus
            # identity no write had actually survived to justify. Passing
            # ``commit=False`` keeps the identity write itself inside this
            # same savepoint span, so it unwinds together with a rejected
            # base/vec0 write instead of outliving it; a wrong-dimension
            # vector or vec0 rejection after the base row already succeeded
            # unwinds the same way, leaving nothing pending in the
            # connection's open transaction for a later, unrelated commit to
            # publish. See :meth:`_write_readback_savepoint`.
            async with self._write_readback_savepoint("store_embedding", begin="IMMEDIATE"):
                await self._ensure_embedding_model_lock(model_name, dimension, commit=False)
                cursor = await self._db.execute(
                    "SELECT rowid FROM embedding WHERE embedding_id = ?",
                    (eid,),
                )
                existing_row = await cursor.fetchone()

                rowid: int
                if existing_row is None:
                    await self._db.execute(
                        "INSERT INTO embedding "
                        "(embedding_id, owner_type, owner_id, model_name, "
                        "dimension, vector_blob, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (eid, "THOUGHT", thought_id, model_name, dimension, blob, created_at),
                    )
                    cursor = await self._db.execute(
                        "SELECT rowid FROM embedding WHERE embedding_id = ?",
                        (eid,),
                    )
                    inserted_row = await cursor.fetchone()
                    if inserted_row is None:
                        msg = f"Embedding row missing after insert: {eid}"
                        raise RuntimeError(msg)
                    rowid = int(inserted_row["rowid"])
                else:
                    rowid = int(existing_row["rowid"])
                    await self._db.execute(
                        "UPDATE embedding SET "
                        "owner_type = ?, owner_id = ?, model_name = ?, dimension = ?, "
                        "vector_blob = ?, created_at = ? "
                        "WHERE embedding_id = ?",
                        ("THOUGHT", thought_id, model_name, dimension, blob, created_at, eid),
                    )

                # Keep the vec0 vector table in sync when a vector backend is
                # active. A rejection here (e.g. a wrong-dimension vector) is
                # unwound together with the base row write above, by the
                # savepoint this block is nested in.
                if self._vector_backend is not None:
                    await self._vector_backend.upsert_embedding(
                        self._db,
                        rowid=rowid,
                        vector=vector,
                    )

            await self._maybe_commit()
        return EmbeddingRecord(
            embedding_id=eid,
            owner_type="THOUGHT",
            owner_id=thought_id,
            model_name=model_name,
            dimension=dimension,
            vector_blob=blob,
            created_at=created_at,
        )

    async def _delete_thought_children_explicit(self, thought_id: str) -> bool:
        """Delete a thought's edge / embedding / action rows without a cascade.

        ``ON DELETE CASCADE`` on these three tables only exists from the
        core-12 schema onward (``_migrate_core_v11_to_v12``); on an older
        schema a plain ``DELETE FROM thought`` leaves them behind, orphaned
        but intact. A dangling ``embedding`` row in particular is what let a
        later reconciliation pass (``sync_embeddings``) treat the thought as
        still live and restore its vector — the resurrection this deletion
        rule exists to close off. Every physical thought-delete path in the
        core (``delete_thought``, the TTL ``delete`` strategy, and hygiene GC)
        reaches this only through :meth:`_delete_thought_atomic`, which calls
        it right after the parent row is already gone, so none of them
        depends on a cascade the database it is running against may not
        have. On a core-12+ schema the parent delete's own cascade already
        removed the same rows — issuing these deletes again here is
        redundant, not incorrect.

        **The three deletes are one indivisible unit, not three independent
        statements.** A rejection on any of them (e.g. an extension-installed
        trigger vetoing the ``action`` delete) must not leave the earlier
        ones sitting in an open transaction: with nothing bracketing them,
        that half-applied state is exactly what a *later, unrelated* write on
        this connection would then commit as a side effect — the thought
        survives, but has silently lost its edges and its embedding. Wrapped
        in a ``SAVEPOINT`` covering the deletes *and* their own release, so a
        failure anywhere in that span — including one delivered while
        awaiting the final ``RELEASE`` itself — ``ROLLBACK TO``s all three at
        once before propagating.

        The savepoint is kept **nested**, never the outermost one: releasing
        the outermost savepoint commits the whole transaction immediately
        (SQLite's rule, not a choice made here), which would force an early,
        partial commit ahead of :meth:`_delete_thought_atomic`'s own
        ``RELEASE``, the vector purge, and the journal entry the original
        caller still has to write — breaking the single deferred commit
        :meth:`_maybe_commit` (or an active :meth:`suspend_auto_commit`
        window) is responsible for. So a
        transaction is opened first when :attr:`self._db.in_transaction
        <aiosqlite.Connection.in_transaction>` is not already ``True`` — the
        same check :meth:`_serialize_dedup_probe` uses for the same reason —
        and, on failure, closed again only if this call is the one that
        opened it; a transaction the caller already held stays exactly as
        open as it was, with only this method's own three deletes undone.
        When this call is the one opening it, it opens with ``BEGIN
        IMMEDIATE`` rather than a deferred ``BEGIN``, mirroring
        :meth:`_write_readback_savepoint` and :meth:`_delete_thought_atomic`:
        a write unit that could still end up reading before it writes must
        never rely on a deferred transaction's snapshot, which a later write
        would have to upgrade and SQLite refuses to upgrade while another
        connection holds the write lock — that refusal surfaces as
        ``SQLITE_BUSY`` without ever invoking the busy handler, so the unit
        fails at once under contention instead of waiting out ``PRAGMA
        busy_timeout``. ``BEGIN IMMEDIATE`` takes the write lock up front,
        through the busy handler, closing that gap regardless of what order
        this call's own body happens to read and write in.

        **A trigger using ``RAISE(ROLLBACK, ...)`` is a real limitation of
        installing one, not a defect this method can close.**
        ``RAISE(ABORT, ...)`` — the ordinary case, and the only form the
        rollback-and-reraise above needs — undoes only the failing
        statement, leaving the transaction (and this savepoint) intact.
        ``RAISE(ROLLBACK, ...)`` instead ends the *entire* transaction,
        taking the savepoint down with it and, if the caller already held a
        transaction of its own, discarding whatever else that caller had
        written before ever calling this method — not only this method's
        three deletes. This method cannot prevent that, and does not try to:
        when ``self._db.in_transaction`` is already ``False`` on entry to the
        failure path there is nothing left to unwind, and the trigger's own
        exception (or a cancellation that raced it) propagates unchanged —
        loudly, to :meth:`_delete_thought_atomic` and, from there, to
        whoever called ``delete_thought`` / ``cleanup_expired`` / hygiene GC,
        never silently.

        **When the transaction survives but the unwind itself cannot be
        proven to have worked, this method refuses rather than guesses.**
        Earlier revisions kept trying to *recover* a consistent state after
        an unwind failure — checking one more condition, swallowing one more
        secondary error — and each attempt closed one ordering while leaving
        another: a savepoint the ``RELEASE`` above already consumed despite
        this coroutine observing a cancellation instead of that success (the
        same aiosqlite worker-thread quirk noted below) makes ``ROLLBACK TO``
        fail with ``"no such savepoint"`` even though the transaction is
        still open — and swallowing *that* left the three deletes sitting
        uncommitted-but-applied for a later, unrelated write to commit as a
        side effect, the exact defect this method exists to close. There is
        no bounded number of special cases that makes "always recover" true
        on a connection this method does not exclusively own. So when the
        unwind itself raises — for any reason, including that race — this
        method stops trying to restore consistency and instead calls
        :meth:`_quarantine_connection`, which makes the store terminally
        unusable *by construction*: every later write, on this instance,
        fails fast with :class:`ConnectionQuarantinedError` instead of ever
        reaching a commit that could make the dangling deletes durable.
        Erring toward a quarantined instance costs one store; erring toward
        "probably fine" costs the data silently, at whatever commit happens
        to come next. A cancellation raised *by the unwind itself* always
        wins over whatever error the unwind was trying to recover from — a
        cleanup that can defeat a cancellation would make shutdown and
        timeout both unreliable, which is worse than the error it hid.

        Args:
            thought_id: UUID of the thought whose children are being removed.

        Returns:
            ``True`` if the connection's own change counter is higher after
            the three deletes than it was before them, ``False`` otherwise.
            This is not the same as "one of the three rowcounts is nonzero":
            a ``BEFORE DELETE`` trigger on any of the three tables can insert
            a row of its own (an audit entry, say) and then veto its own
            statement with ``RAISE(IGNORE)``, leaving every rowcount at zero
            while that insert is still applied. It is also independent of
            whether the *parent* thought existed — a schema (or connection)
            without FK enforcement can carry an orphaned child for a
            ``thought_id`` that was never a live thought at all, and sweeping
            that orphan is itself a real write :meth:`_delete_thought_atomic`
            must account for even when it reports the parent as not deleted.

        Raises:
            aiosqlite.Error: Propagated from any of the three deletes (e.g. a
                trigger veto) when the savepoint they were made under is
                still intact and the unwind completes cleanly.
            asyncio.CancelledError: Propagated when this call is cancelled,
                or when a cancellation lands during a failed unwind's own
                cleanup attempt — always in preference to the error that
                unwind was trying to recover from.
            ConnectionQuarantinedError: On this store's *next* guarded call,
                after this method quarantined the connection because an
                unwind attempt itself failed and a consistent state could
                not be proven. This call itself still raises the original
                failure (or the cancellation that pre-empted its cleanup),
                not this error — quarantine changes what happens *after*.

        """
        opened_transaction = not self._db.in_transaction
        if opened_transaction:
            await self._db.execute("BEGIN IMMEDIATE")
        await self._db.execute("SAVEPOINT delete_thought_children")
        try:
            # `total_changes` (not each cursor's own `rowcount`) is what
            # actually answers "did this write anything": a `BEFORE DELETE`
            # trigger on any of these three tables can insert an audit row
            # of its own and then veto its own statement with
            # `RAISE(IGNORE)`, which leaves that insert applied while the
            # vetoed delete's rowcount is zero. `rowcount` cannot see the
            # trigger's write; `total_changes` counts it.
            changes_before = self._db.total_changes
            await self._db.execute(
                "DELETE FROM edge WHERE from_thought_id = ? OR to_thought_id = ?",
                (thought_id, thought_id),
            )
            await self._db.execute(
                "DELETE FROM embedding WHERE owner_id = ?",
                (thought_id,),
            )
            await self._db.execute(
                "DELETE FROM action WHERE source_thought_id = ?",
                (thought_id,),
            )
            wrote_anything = self._db.total_changes > changes_before
            # The release lives INSIDE this guarded region, deliberately —
            # not after it. With aiosqlite, cancelling the awaiting future
            # does not cancel a statement already queued on the worker
            # thread: a cancellation delivered while awaiting this specific
            # call can still see the RELEASE complete on the connection. A
            # cancellation landing here is exactly the shape the unwind
            # below has to be able to recognise as "already released", not
            # just "the deletes never happened".
            await self._db.execute("RELEASE delete_thought_children")
        # ``except BaseException`` (not ``Exception``) so a cancellation
        # landing anywhere in the block above — mid-delete or during the
        # release itself — also reaches the unwind below before it
        # propagates.
        except BaseException as exc:
            if not self._db.in_transaction:
                # A RAISE(ROLLBACK) trigger already ended the whole
                # transaction (savepoint included) — see the docstring.
                # Nothing is left open for a later write to inherit, so
                # there is nothing to unwind and nothing to quarantine.
                raise
            try:
                await self._db.execute("ROLLBACK TO delete_thought_children")
                await self._db.execute("RELEASE delete_thought_children")
                if opened_transaction:
                    await self._db.rollback()
            except BaseException as unwind_exc:
                # The unwind itself failed: recovery cannot be proven, so
                # this connection is no longer trusted to decide anything
                # about the transaction it might still be holding open —
                # quarantine it rather than guess. `_quarantine_connection`
                # awaits nothing internally (every step is synchronous), so
                # this call itself cannot be interrupted partway.
                await self._quarantine_connection(
                    f"delete_thought_children could not unwind its savepoint "
                    f"after {exc!r}: {unwind_exc!r}"
                )
                if isinstance(unwind_exc, asyncio.CancelledError):
                    # A cancellation arriving during the unwind always wins
                    # over the error the unwind was trying to recover from —
                    # never swallowed, or shutdown/timeout stop working.
                    raise
                # Any other unwind failure: the caller still sees the
                # original error, not this one. `unwind_exc` is chained as
                # the cause for diagnosis, never as the raised type.
                raise exc from unwind_exc
            raise
        # No ``else`` branch: the release is the last statement inside the
        # ``try`` above, and reaching here means it already succeeded. Never
        # a commit on any path — the parent row is already gone by the time
        # this runs (see ``_delete_thought_atomic``), which still has its own
        # ``RELEASE`` to issue, and the original caller (``delete_thought`` /
        # ``cleanup_expired`` / hygiene GC) still has the vector purge and
        # the journal entry to write before its own ``_maybe_commit()``
        # decides when any of it becomes durable.
        return wrote_anything

    async def _delete_thought_atomic(self, thought_id: str) -> _DeleteAtomicResult:
        """Delete a thought and its children as one indivisible unit.

        Reverses ``6e4ed41``'s ordering, which deleted the edge / embedding /
        action rows *before* the parent row and released their savepoint the
        moment those three deletes succeeded — before the parent delete had
        even run. Two independent defects followed from that ordering, not
        one:

        1. **Lost atomicity.** Once the children's savepoint released,
           nothing bracketed the parent delete that came after it. Anything
           that then prevented, diverted or skipped that delete — a veto, a
           bug, a caller that stopped short — left the children gone and the
           parent still there, sitting in the open transaction until any
           later, unrelated write on the same connection committed it. No
           error, no quarantine: the loss became durable as a side effect of
           an ordinary later write.
        2. **A defeated guard.** A ``BEFORE DELETE ON thought`` trigger
           written to veto deleting a thought that still has live children
           (``WHEN EXISTS (SELECT 1 FROM edge/embedding/action WHERE ...)``)
           never saw them: by the time the parent delete ran, the explicit
           child deletes had already removed the rows the predicate tests
           for. The trigger did not misfire — it never fired, and the delete
           it existed to block **succeeded**.

        The fix is the parent delete **first**, inside the same savepoint
        that then covers the explicit child deletes, released only once both
        stages are done:

        * A ``BEFORE DELETE`` trigger on ``thought`` now runs while the
          children are still present, exactly as it did before ``6e4ed41`` —
          restoring case 2 without giving up case 1's ability to veto or
          divert the delete outright.
        * ``6e4ed41``'s reason for deleting the children explicitly is
          untouched: this method still issues the same three ``DELETE``
          statements, via :meth:`_delete_thought_children_explicit`, so a
          store on a pre-core-12 schema (no cascade) or a connection with
          ``PRAGMA foreign_keys`` off (the default, and a documented no-op
          mid-transaction — see that method's docstring) still has its
          embedding row removed and cannot resurrect a vector the way the
          original bug did. On a core-12+ schema with enforcement on, the
          parent delete's own cascade removes the same rows first; the
          explicit deletes that follow then affect zero rows — redundant,
          not incorrect, exactly as documented there.
        * If the explicit child deletes are rejected by a *raising* trigger
          (``RAISE(ABORT)`` / ``RAISE(FAIL)`` / a ``WHEN EXISTS`` guard, e.g.
          on the ``action`` delete) after the parent row is already gone,
          this savepoint's own ``ROLLBACK TO`` undoes the parent delete
          along with whatever :meth:`_delete_thought_children_explicit`
          already undid of its own — the parent and the children succeed or
          fail together in that case, which is the property ``6e4ed41``
          broke. **This does not cover a silent ``RAISE(IGNORE)`` veto on a
          child delete**, because ``RAISE(IGNORE)`` never raises at all: the
          ``except`` branch below — the only place this savepoint is rolled
          back — is never reached. :meth:`_delete_thought_children_explicit`
          sees the vetoed delete's own rowcount stay at zero, finds nothing
          to raise, and returns normally; this savepoint's own ``RELEASE``
          then runs instead of a rollback, keeping the parent's deletion. A
          child a ``RAISE(IGNORE)`` trigger quietly leaves in place is
          therefore still there, still referencing the now-gone parent,
          while this method reports the parent as deleted and
          :meth:`delete_thought` returns ``True``. Closing that gap is a
          separate, pre-existing behaviour question, not something this fix
          changes.
        * The explicit child deletes run **whether or not a parent row
          existed** — matching the pre-fix call sites exactly, which issued
          the same three (harmless, zero-row) deletes unconditionally — so a
          delete of a nonexistent ``thought_id`` still sweeps any orphaned
          children a schema without a cascade could be carrying.

        **A zero-row parent delete is not, by itself, "nonexistent."**
        ``RAISE(ABORT)``, ``RAISE(FAIL)`` and a ``WHEN EXISTS`` guard all
        raise, so the ``except`` branch below already unwinds them. But
        ``RAISE(IGNORE)`` does not raise — it aborts only the triggering
        statement itself. Any statement the trigger's own body already ran
        before reaching the ``RAISE`` (an audit-table insert, say) is *not*
        undone and stays applied in the open transaction, so the parent
        ``DELETE`` matches zero rows while the row (and its children) are
        still there — exactly as if ``thought_id`` had never existed, except
        for whatever that trigger already wrote. Rowcount alone cannot tell
        the two apart, so this method establishes the row's existence with a
        ``SELECT`` **inside this savepoint, immediately before** the parent
        ``DELETE`` — the same instant several callers already fetch a
        ``before_row`` for their journal entry, but done here, unconditionally
        and independently of whether a journal is attached, so every caller
        gets the discrimination regardless. That ``SELECT`` is exactly why,
        when this call is the one opening the transaction, it opens with
        ``BEGIN IMMEDIATE`` rather than a deferred ``BEGIN``: a read inside a
        deferred transaction takes a WAL snapshot that the ``DELETE`` right
        after it would then have to upgrade, and SQLite refuses that upgrade
        while another connection holds the write lock — surfacing
        ``SQLITE_BUSY`` without ever invoking the busy handler, so this unit
        would fail at once under contention instead of waiting out
        ``PRAGMA busy_timeout`` like an ordinary write. ``BEGIN IMMEDIATE``
        takes the write lock up front, through the busy handler, before this
        existence check ever runs. Reliable under a concurrent
        writer for the case this fix targets: both statements run back to
        back on this connection, inside one still-open transaction, with
        ``_write_lock`` already held for the whole call, so nothing on *this*
        connection can intervene between them. A genuinely different writer
        racing the same row from another connection can only align with our
        ``DELETE`` after committing first (SQLite allows only one writer's
        transaction to commit at a time), at which point that writer's own
        atomic delete already owns the purge and the journal entry for that
        row — this call correctly sees zero rows, correctly does not treat
        the row as newly gone, and correctly still performs no purge or
        journal append of its own either way, so the misclassification (if
        any) is inert:

        * ``existed_before`` and the parent delete succeeded — the ordinary
          case, children swept as always.
        * **not** ``existed_before`` — nothing to protect; sweep any
          orphaned children as before and report ``False``.
        * ``existed_before`` **and** the parent delete still matched zero
          rows — a silent veto. The children are *not* swept: instead this
          savepoint is rolled back (undoing nothing but the no-op delete
          itself) so the row and its still-attached children are exactly as
          they were, and ``False`` is reported. If this call is also the one
          that opened the outer transaction (``opened_transaction``, sampled
          before either was touched), it ends that transaction here too —
          with a rollback, never a commit, since a transaction this call
          opened cannot hold a caller's pending work. This must not depend on
          a caller closing it instead: ``run_hygiene`` only commits ``if
          archived_count or gc_count``, both zero when every candidate in a
          GC batch is vetoed, so nothing else would ever end it — leaving it
          open here would strand the write reservation on this connection
          indefinitely.

        Quarantine and cancellation handling wrap
        :meth:`_delete_thought_children_explicit`'s rather than duplicating
        it: if that call already quarantined the connection while unwinding
        its own (nested) savepoint, ``self._db`` is now the terminal
        ``_QuarantinedConnection`` proxy, and touching it again — even just
        reading ``.in_transaction`` — raises :class:`ConnectionQuarantinedError`
        from *every* attribute access, which would silently replace the
        real error (an ``asyncio.CancelledError``, in the race that provokes
        this) with that one. So ``self._connection_quarantined`` — a plain
        flag on ``self``, never proxied — is checked *first*, before
        ``self._db`` is touched at all, and the original error is left to
        propagate unchanged when it is already set.

        Args:
            thought_id: UUID of the thought to delete, along with its
                edge / embedding / action rows.

        Returns:
            A :class:`_DeleteAtomicResult`. Its ``deleted`` is ``True`` if a
            thought row was deleted, ``False`` if no row matched
            ``thought_id`` **or** a row matched but a trigger silently
            suppressed the delete (``RAISE(IGNORE)``) — in both cases the
            caller must treat this exactly like "nothing was deleted": no
            vector purge, no journal append. Its ``wrote_anything`` is
            ``True`` whenever this call's own return leaves a row change
            intact — the parent, an orphan swept on a never-existed id, or
            both — which callers need to decide whether they have anything
            of their own left to commit; ``deleted`` alone is not that
            signal (see the class docstring). A silently vetoed parent
            delete rolls its own savepoint all the way back, so a trigger's
            write that happened before the veto does not make this ``True``.

        Raises:
            aiosqlite.Error: Propagated from the parent delete or any of the
                three child deletes (e.g. a trigger veto), once this
                savepoint's own unwind (if one was needed) has completed.
            asyncio.CancelledError: Propagated on cancellation, in
                preference to any error an unwind it interrupted was trying
                to recover from.
            ConnectionQuarantinedError: On this store's *next* guarded call,
                after an unwind attempt (here, or inside
                :meth:`_delete_thought_children_explicit`) itself failed and
                a consistent state could not be proven.

        """
        opened_transaction = not self._db.in_transaction
        if opened_transaction:
            await self._db.execute("BEGIN IMMEDIATE")
        await self._db.execute("SAVEPOINT delete_thought_atomic")
        try:
            # Established inside this savepoint, immediately before the
            # parent DELETE, so a RAISE(IGNORE) veto (which leaves the row in
            # place but the DELETE's own rowcount at zero, indistinguishable
            # from "never existed" by rowcount alone) can be told apart from
            # a genuinely nonexistent ``thought_id`` — see the docstring.
            existence_cursor = await self._db.execute(
                "SELECT 1 FROM thought WHERE thought_id = ?", (thought_id,)
            )
            existed_before = await existence_cursor.fetchone() is not None

            changes_before = self._db.total_changes
            cursor = await self._db.execute(
                "DELETE FROM thought WHERE thought_id = ?", (thought_id,)
            )
            deleted = cursor.rowcount > 0

            if deleted or not existed_before:
                # Either the parent delete actually succeeded, or
                # ``thought_id`` never matched a row at all — the pre-fix
                # behaviour for a nonexistent id, unchanged: sweep any
                # orphaned edge / embedding / action rows a schema without a
                # cascade could still be carrying. That sweep can itself
                # write real rows even when ``deleted`` is ``False`` — see
                # _DeleteAtomicResult — so its own report is folded in here
                # rather than assumed away. Folded in via ``total_changes``
                # over both statements together, not the child call's return
                # OR'd onto ``deleted``: a trigger anywhere in this span
                # (the parent delete or any of the three child deletes) can
                # write a real row and still veto its own statement with
                # ``RAISE(IGNORE)``, which leaves every rowcount involved at
                # zero. ``total_changes`` counts that write; rowcount cannot.
                await self._delete_thought_children_explicit(thought_id)
                await self._db.execute("RELEASE delete_thought_atomic")
                # Sampled last, after the ``RELEASE`` -- not before it. See
                # ``update_edge`` for why: this connection's own
                # ``total_changes`` can lag a write by one further
                # SAVEPOINT-class statement on an FTS-backed table, and
                # ``RELEASE`` is one such statement. Reading it only now
                # means nothing this branch did (the parent delete, the
                # child sweep, and this release) can land uncounted.
                wrote_anything = self._db.total_changes > changes_before
            else:
                # existed_before and not deleted: the row was there and the
                # DELETE still matched zero rows, so a trigger silently
                # suppressed it (RAISE(IGNORE) is the only form that does —
                # RAISE(ABORT)/RAISE(FAIL)/a WHEN EXISTS guard all raise and
                # are handled by the except clause below instead). Roll the
                # savepoint back instead of sweeping the children: this
                # discards the no-op DELETE and, along with it, any earlier
                # write the vetoing trigger's own body already made (e.g. an
                # audit insert) inside this savepoint — none of that is kept,
                # so ``wrote_anything`` is unconditionally ``False`` here,
                # and skipping the sweep is what keeps the still-live
                # parent's children attached to it.
                wrote_anything = False
                await self._db.execute("ROLLBACK TO delete_thought_atomic")
                await self._db.execute("RELEASE delete_thought_atomic")
                if opened_transaction:
                    # This call opened the outer transaction (sampled via
                    # ``opened_transaction`` before anything below it ran),
                    # and the veto means nothing durable happened inside it
                    # — the parent row is unchanged and the children were
                    # never swept. Ending it here — rather than trusting a
                    # caller to do it — is what keeps a vetoed delete from
                    # being indistinguishable, to a second connection, from a
                    # still-open write reservation: ``delete_thought`` and
                    # ``cleanup_expired`` both gate their own call to
                    # ``_maybe_commit()`` on whether anything was actually
                    # written and roll back a self-opened, still-empty
                    # transaction otherwise, and ``run_hygiene`` commits only
                    # ``if archived_count or gc_count`` — all three leave a
                    # vetoed batch's reservation to be closed here, since none
                    # of them would otherwise. This call must not depend on
                    # which of the three its caller happens to be. A rollback,
                    # not a commit — there is nothing of ours to preserve, and
                    # a transaction we opened ourselves cannot be carrying a
                    # caller's pending work for a commit to risk instead.
                    await self._db.rollback()
        # ``except BaseException`` for the same reason
        # ``_delete_thought_children_explicit`` uses it: a cancellation must
        # reach the unwind below too, not just an ordinary ``Exception``.
        except BaseException as exc:
            if self._connection_quarantined:
                # The nested call already quarantined the connection and
                # left `self._db` a terminal proxy — see the docstring.
                # Nothing here can be trusted to roll back or release
                # anything, and touching `self._db` would replace `exc`
                # with ConnectionQuarantinedError instead of propagating it.
                raise
            if not self._db.in_transaction:
                # A RAISE(ROLLBACK) trigger — on `thought` itself, or
                # propagated up from the nested savepoint — already ended
                # the whole transaction. Nothing is left open to unwind.
                raise
            try:
                await self._db.execute("ROLLBACK TO delete_thought_atomic")
                await self._db.execute("RELEASE delete_thought_atomic")
                if opened_transaction:
                    await self._db.rollback()
            except BaseException as unwind_exc:
                # The unwind itself failed: recovery cannot be proven, so
                # this connection is no longer trusted to decide anything
                # about the transaction it might still be holding open —
                # quarantine it rather than guess, mirroring
                # _delete_thought_children_explicit's own handling.
                await self._quarantine_connection(
                    f"delete_thought_atomic could not unwind its savepoint "
                    f"after {exc!r}: {unwind_exc!r}"
                )
                if isinstance(unwind_exc, asyncio.CancelledError):
                    raise
                raise exc from unwind_exc
            raise
        return _DeleteAtomicResult(deleted=deleted, wrote_anything=wrote_anything)

    async def _embedding_rowid_for_thought(self, thought_id: str) -> int | None:
        """Resolve the ``embedding`` rowid backing a thought's vector, if any.

        Must be called *before* a thought delete cascades the ``embedding``
        row away, so the caller can subsequently purge the matching vec0
        vector (which the FK cascade cannot reach). Returns ``None`` when no
        vector backend is active (the numpy path needs no purge) or when the
        thought has no embedding, so the caller can skip the purge entirely.

        Args:
            thought_id: UUID of the thought whose embedding rowid to resolve.

        Returns:
            The ``embedding`` rowid, or ``None`` if there is no vector backend
            or no embedding row for the thought.

        """
        if self._vector_backend is None:
            return None
        cursor = await self._db.execute(
            "SELECT rowid FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
            (thought_id,),
        )
        row = await cursor.fetchone()
        return int(row["rowid"]) if row is not None else None

    async def _purge_orphan_vector(self, rowid: int | None) -> None:
        """Remove a now-orphaned vec0 vector left behind by a thought delete.

        Paired with :meth:`_embedding_rowid_for_thought`: the numpy backend
        yields ``None`` (nothing to do — byte-identical to the pre-fix path),
        while an active sqlite-vec backend deletes the vector whose FK-cascaded
        ``embedding`` row has just been removed.

        Args:
            rowid: The vec0 rowid to delete, or ``None`` to no-op.

        """
        if self._vector_backend is None or rowid is None:
            return
        await self._vector_backend.delete_embedding(self._db, rowid=rowid)

    async def get_embedding(self, thought_id: str) -> EmbeddingRecord | None:
        """Retrieve the embedding for a thought, or None if not found.

        Args:
            thought_id: UUID of the thought.

        Returns:
            The EmbeddingRecord, or None if not found.

        """
        cursor = await self._db.execute(
            "SELECT * FROM embedding WHERE owner_type = 'THOUGHT' AND owner_id = ?",
            (thought_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_embedding(row)

    # ------------------------------------------------------------------
    # Embedding similarity search (brute-force cosine)
    # ------------------------------------------------------------------

    def _declared_embedding_dimension(self) -> int | None:
        """Return the embedding dimension the store declares, if any.

        The dimension a query vector must match, resolved from the store's
        configuration without touching the database: the configured vector
        backend takes precedence (its ``vec0`` table is dimension-typed), then
        the embedding provider. ``None`` when neither is configured — a store
        that only ever received raw vectors via ``store_embedding`` has no
        declared dimension at this level, and the numpy arm validates such a
        vector against the *stored* embedding dimension instead.

        Returns:
            The declared embedding dimension, or ``None`` when the store
            declares none.

        Raises:
            EmbeddingProviderContractError: When the configured embedding
                provider exposes no public ``dimension``.

        """
        if self._vector_backend is not None:
            return self._vector_backend.dimension
        if self._embedding_provider is not None:
            return _provider_dimension(self._embedding_provider)
        return None

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
        *,
        include_archived: bool = False,
        _filter_clause: tuple[str, list[object]] | None = None,
    ) -> list[tuple[str, float]]:
        """Cosine similarity search — delegates to sqlite-vec if available.

        When a ``SqliteVecSearchBackend`` is configured (via
        ``from_config`` with ``vector_backend: "sqlite-vec"``), the
        ``vec0`` vector table serves the query.  Otherwise falls back to
        brute-force numpy cosine similarity.

        Result completeness (sqlite-vec arm): vec0 applies its ``k``/``LIMIT``
        before expired thoughts and retired REFLECTIONs can be filtered out
        (that filter is a post-``MATCH`` join). To avoid returning fewer than
        ``top_k`` live rows, the vec0 arm over-fetches a **bounded** multiple
        of ``top_k`` (``search.vec0_overfetch_factor``, capped by
        ``_VEC0_OVERFETCH_CAP``), applies the live-row filter, then trims to
        ``top_k``. This is **best-effort, not a guarantee** — under-fill can
        still occur in two cases: (1) a store where almost all of the nearest
        ``vec0_overfetch_factor * top_k`` neighbours are expired/retired; and
        (2) when ``top_k * vec0_overfetch_factor`` exceeds ``_VEC0_OVERFETCH_CAP``
        (a large ``top_k``), the fetch is limited to the cap, so even a
        moderate expiry rate among the nearest ``_VEC0_OVERFETCH_CAP`` neighbours
        can leave fewer than ``top_k`` live rows. An exact filter-before-k for
        vec0 is a separately-gated future change. The numpy arm already filters
        eligibility inside the SQL ``WHERE`` before top-k and so does not need
        this.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity score.
            include_archived: When ``False`` (the default) archived thoughts
                (``lifecycle_status = 'ARCHIVED'``) are excluded from the
                candidate set on both the ``vec0`` and the numpy arm — the same
                eligibility class as expired rows. When ``True`` archived rows
                are re-admitted for this call (the "search my archive" escape
                hatch), without restoring them.
            _filter_clause: Internal. A compiled
                ``(sql_fragment, params)`` metadata predicate (referencing
                ``t.metadata_json``). When supplied the exhaustive numpy path
                is used unconditionally — even when a ``vec0`` backend is
                configured. This is specific to how the predicate is shaped,
                not a blanket ``vec0`` limitation: it is an arbitrary
                ``metadata_json`` expression that can only run as a
                post-``MATCH`` join, and the ``vec0`` table declares no
                metadata columns, so the join would land *after* ``vec0``
                applies its ``k``/``LIMIT`` — yielding wrong neighbours
                (filtering eligible rows must precede cosine and top-k, never
                follow a ``LIMIT``). (``vec0`` *can* apply ``k`` after a filter
                on a *declared*, typed metadata column; this table declares
                none.) Supplied by :meth:`search_hybrid`; not part of the
                public contract.

        Returns:
            List of ``(thought_id, similarity_score)`` sorted descending
            (ties broken by ``thought_id`` ascending for a deterministic
            total order).

        Raises:
            VectorDimensionMismatchError: When ``query_vector`` is not the
                dimension the store declares.
            EmbeddingProviderContractError: When the store's embedding provider
                exposes no public ``dimension``, so the store cannot say what
                dimension it declares.

        """
        import time as _time  # noqa: PLC0415

        _t_start = _time.perf_counter()

        # --- Query-vector contract guard (backend-agnostic, pre-dispatch) ---
        # Enforced once here so both the vec0 and the numpy arm share identical
        # semantics. Order matters: a wrong dimension is checked first, so a
        # structurally invalid vector is rejected regardless of its magnitude (a
        # wrong-length all-zero vector is a dimension error, not a degeneracy).
        # A degenerate vector (empty/all-zero/non-finite) then degrades to an
        # empty result surfaced via the read-only degradation counter, rather
        # than silently returning [] as an ordinary "no neighbours" answer.
        expected_dim = self._declared_embedding_dimension()
        if expected_dim is not None and len(query_vector) != expected_dim:
            raise VectorDimensionMismatchError(expected=expected_dim, actual=len(query_vector))
        if _query_vector_is_degenerate(query_vector):
            self._vector_arm_degradation_count += 1
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return []

        if self._vector_backend is not None and _filter_clause is None:
            # Bounded over-fetch: vec0 applies its k/LIMIT *before* we can drop
            # expired/retired rows (the live-row filter is a post-MATCH join),
            # so fetching only ``top_k`` would under-fill whenever any of the
            # nearest ``top_k`` neighbours turn out to be non-live. Fetch a
            # bounded multiple instead, filter, sort, then trim to ``top_k`` so
            # the trim keeps the highest-similarity *live* rows. The deeper live
            # pool also now feeds the hybrid-fusion vector arm more completely
            # (previously under-fed); for stores containing expired/retired rows
            # this can shift the fused order — a more-correct pool, disclosed.
            overfetch_factor = (
                self._search_config.vec0_overfetch_factor if self._search_config is not None else 4
            )
            effective_fetch = min(top_k * overfetch_factor, _VEC0_OVERFETCH_CAP)
            results = await self._vector_backend.search(
                self._db,
                query_vector,
                effective_fetch,
                threshold,
            )
            filtered = await self._filter_expired_results(
                results,
                include_archived=include_archived,
            )
            filtered = _sort_scored_descending(filtered)[:top_k]
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return filtered
        results = await self._search_similar_numpy(
            query_vector,
            top_k,
            threshold,
            include_archived=include_archived,
            _filter_clause=_filter_clause,
        )
        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
        return results

    async def _filter_expired_results(
        self,
        results: list[tuple[str, float]],
        *,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Keep only results that positively resolve to a live, eligible thought.

        Used as a post-filter for search backends (e.g. sqlite-vec) that
        cannot natively exclude ineligible rows in their queries. Under the
        rule that a vector is owned by a live thought, a vec0 hit is only
        ever as trustworthy as the ``embedding`` row it was resolved through,
        and that row can outlive its thought on a schema predating the
        core-12 cascade — so this filter does not compute what to *remove*
        from an otherwise-trusted list. It computes what it can **positively
        confirm** — a ``thought`` row that exists, is not expired, is not a
        retired REFLECTION, and (unless ``include_archived``) is not
        archived — and keeps only that. An id absent from ``thought``
        entirely, including a deleted thought whose ``embedding`` row was
        never cleaned up, resolves no confirming row and is dropped by
        construction rather than by a case this filter has to remember to
        add: **absence is a positive exclusion, not an incidental miss.**

        Args:
            results: List of ``(thought_id, similarity_score)`` pairs.
            include_archived: When ``False`` (the default) archived thoughts
                are never confirmed eligible; when ``True`` they may be (the
                retired-REFLECTION, expiry, and existence gates still apply).

        Returns:
            The subset of ``results`` whose thought positively confirms as
            live and eligible under the current gates.

        """
        if not results:
            return results
        now = datetime.datetime.now(datetime.UTC).isoformat()
        ids = [r[0] for r in results]
        placeholders = ",".join("?" * len(ids))
        # Positive form: keep-form archived gate (``!= 'ARCHIVED'``), matching
        # the arm WHERE clauses — the opposite polarity of the old
        # exclusion-form query this replaces.
        archived_clause = (
            ""
            if include_archived
            else f" AND lifecycle_status != '{LifecycleStatus.ARCHIVED.value}'"
        )
        cursor = await self._db.execute(
            f"SELECT thought_id FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders}) "
            f"AND (expires_at IS NULL OR expires_at > ?) "
            f"AND NOT (thought_type = 'REFLECTION' AND lifecycle_status != 'ACTIVE')"
            f"{archived_clause}",
            [*ids, now],
        )
        eligible_ids = {row["thought_id"] for row in await cursor.fetchall()}
        if not eligible_ids:
            return []
        return [(tid, score) for tid, score in results if tid in eligible_ids]

    async def _search_similar_numpy(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
        *,
        include_archived: bool = False,
        _filter_clause: tuple[str, list[object]] | None = None,
    ) -> list[tuple[str, float]]:
        """Brute-force cosine similarity search (numpy-batched).

        The arm order is mandatory (and the reason a metadata filter forces
        this exhaustive path): SQL-filter the eligible rows, compute cosine
        over **all** of them, then apply top-k. A ``LIMIT`` before cosine
        would surface wrong neighbours.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity score.
            include_archived: When ``False`` (the default) archived thoughts are
                excluded from the candidate rows before cosine; when ``True``
                they remain eligible.
            _filter_clause: Internal. A compiled ``(sql_fragment, params)``
                metadata predicate (referencing ``t.metadata_json``) injected
                into the ``WHERE`` so cosine runs only over eligible rows.

        Returns:
            List of ``(thought_id, similarity_score)`` sorted descending
            (ties broken by ``thought_id`` ascending).

        """
        filter_sql = ""
        filter_params: list[object] = []
        if _filter_clause is not None:
            filter_fragment, filter_params = _filter_clause
            filter_sql = f"AND {filter_fragment} "
        archived_sql = _archived_exclusion_sql(
            column="t.lifecycle_status",
            include_archived=include_archived,
        )

        cursor = await self._db.execute(
            "SELECT e.owner_id, e.dimension, e.vector_blob "  # noqa: S608
            "FROM embedding e "
            "JOIN thought t ON e.owner_id = t.thought_id "
            "WHERE e.owner_type = 'THOUGHT' "
            "AND (t.expires_at IS NULL OR t.expires_at > ?) "
            # Freshness floor: a retired REFLECTION (an orphan archived once
            # its cluster left the active set) must not over-recall on its
            # now-stale centroid. Only REFLECTIONs are gated on lifecycle
            # here; other thought types keep their existing recall behaviour.
            "AND NOT (t.thought_type = 'REFLECTION' AND t.lifecycle_status != 'ACTIVE')"
            # Archived-exclusion: forgotten (archived) thoughts leave the default
            # candidate set unless the caller opts in via include_archived.
            f"{archived_sql} "
            f"{filter_sql}",
            (datetime.datetime.now(datetime.UTC).isoformat(), *filter_params),
        )
        rows = list(await cursor.fetchall())
        if not rows:
            return []

        query_arr = np.asarray(query_vector, dtype=np.float64)
        q_norm = float(np.linalg.norm(query_arr))
        if q_norm == 0.0:
            # Defense-in-depth: an all-zero (zero-norm) query vector has no
            # cosine direction. ``search_similar`` already intercepts every
            # degenerate vector at its boundary and increments the degradation
            # counter, so this branch is not reached on that path; it guards a
            # direct/internal call from dividing by a zero norm below.
            return []

        owner_ids = [str(row["owner_id"]) for row in rows]
        # Batch-decode every blob with a single ``np.frombuffer`` instead of a
        # per-row ``struct.unpack`` + ``list()``. The dtype is **native-endian**
        # ``np.float32`` — the exact byte layout ``store_embedding`` writes with
        # native ``struct.pack(f"{dim}f", …)`` — so the decode is bit-identical
        # to the old ``struct.unpack(f"{dim}f", …)`` on every platform (both use
        # the host byte order). Every embedding in a store shares one
        # ``dimension`` (enforced by the model lock), so the blobs are joined and
        # viewed as one ``(n, dimension)`` matrix in a single decode. If any blob
        # is missing/short (a corrupt or truncated row), the fast path is
        # abandoned for the original per-row decode so the exact prior
        # skip/error behaviour is preserved.
        first_dimension = int(rows[0]["dimension"])
        if len(query_vector) != first_dimension:
            # Typed rejection instead of an opaque numpy ``matmul`` ValueError.
            # Reached whenever no vector backend is configured, whether or not
            # an embedding provider is: ``search_similar``'s boundary guard
            # compares ``query_vector`` against ``_declared_embedding_dimension()``
            # — the provider's *current* dimension, not what a corpus's rows
            # actually stored — so it lets a query through here whenever it
            # matches the provider's current setting even if the stored rows
            # were embedded at a different dimension. This check compares
            # against the *stored* dimension instead, so it is what actually
            # catches a corpus written under one dimension being queried after
            # the provider's own has since changed.
            raise VectorDimensionMismatchError(expected=first_dimension, actual=len(query_vector))
        expected_bytes = first_dimension * 4
        blobs = [row["vector_blob"] for row in rows]
        uniform = first_dimension > 0 and all(
            int(row["dimension"]) == first_dimension and len(blob) == expected_bytes
            for row, blob in zip(rows, blobs, strict=True)
        )
        matrix: npt.NDArray[np.float64]
        if uniform:
            # Decode as native float32 (bit-identical to the stored bytes), then
            # widen to float64 so the norm/dot arithmetic below runs at exactly
            # the same precision as the original per-row path (which built a
            # float64 matrix). ``frombuffer`` returns a read-only view;
            # ``astype`` produces the writable float64 copy the reduction expects.
            matrix = (
                np.frombuffer(b"".join(blobs), dtype=np.float32)
                .reshape(len(rows), first_dimension)
                .astype(np.float64)
            )
        else:
            vectors: list[list[float]] = [
                list(struct.unpack(f"{int(row['dimension'])}f", blob))
                for row, blob in zip(rows, blobs, strict=True)
            ]
            matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        dot_products = matrix @ query_arr
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        scores = np.where(norms > 0.0, dot_products / (safe_norms * q_norm), 0.0)

        results: list[tuple[str, float]] = [
            (owner_ids[i], float(scores[i]))
            for i in range(len(owner_ids))
            if float(scores[i]) >= threshold
        ]
        # Cosine over all eligible rows is complete; apply the deterministic
        # total order, then top-k.
        results = _sort_scored_descending(results)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Full-text search (FTS5 + BM25)
    # ------------------------------------------------------------------

    async def search_fts(
        self,
        query: str,
        top_k: int = 10,
        *,
        include_archived: bool = False,
        _filter_clause: tuple[str, list[object]] | None = None,
    ) -> list[tuple[str, float]]:
        """Full-text search via SQLite FTS5 with BM25 ranking.

        Bare natural-language queries are matched with ``OR`` semantics: a
        document is returned when it shares *any* content word with the query,
        and BM25 IDF weighting ranks documents that share the most distinctive
        words first. Function words ("what", "was", "my") therefore never block
        a match. Expert syntax — quoted phrases, uppercase ``AND``/``OR``/
        ``NOT``, and the ``essence:``/``content:`` column filters — is preserved
        and matched exactly as written.

        Returns an empty list when the FTS5 index is unavailable (backward
        compat for databases that predate the migration), when the query
        normalizes to no usable term, or when a malformed FTS5 expression slips
        through; such errors are logged and degraded rather than propagated, so
        a caller's other search arms can still serve the query.

        Args:
            query: User-facing query string. Bare questions are OR-matched;
                quoted phrases, uppercase ``AND``/``OR``/``NOT`` and
                ``essence:``/``content:`` column filters invoke expert syntax.
            top_k: Maximum number of results.
            include_archived: When ``False`` (the default) archived thoughts
                (``lifecycle_status = 'ARCHIVED'``) are excluded from matches
                before the ``LIMIT``; when ``True`` they are re-admitted for this
                call without restoring them.
            _filter_clause: Internal. A compiled
                ``(sql_fragment, params)`` metadata predicate (referencing
                ``t.metadata_json``) injected into the ``WHERE`` *before* the
                ``LIMIT`` so out-of-filter rows never consume the FTS arm's
                budget. Supplied by :meth:`search_hybrid`; not part of the
                public contract.

        Returns:
            List of ``(thought_id, bm25_score)`` sorted by relevance
            (higher = more relevant; ties broken by ``thought_id`` ascending
            for a deterministic total order).

        """
        import time as _time  # noqa: PLC0415
        from sqlite3 import OperationalError  # noqa: PLC0415

        _t_start = _time.perf_counter()
        if not query or not query.strip():
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return []

        if not self._fts_probed:
            await self._probe_fts()

        if not self._fts_available:
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return []

        normalized_query = _normalize_fts_query(query)
        if not normalized_query:
            # The query held no indexable term (e.g. only punctuation); an empty
            # MATCH string is a syntax error in FTS5, so short-circuit to empty.
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return []

        filter_sql = ""
        filter_params: list[object] = []
        if _filter_clause is not None:
            filter_fragment, filter_params = _filter_clause
            # Injected before LIMIT: out-of-filter rows never consume the
            # arm's budget. ``filter_fragment`` is its own json_valid-guarded
            # CASE expression, safe to AND in directly.
            filter_sql = f"AND {filter_fragment} "

        archived_sql = _archived_exclusion_sql(
            column="t.lifecycle_status",
            include_archived=include_archived,
        )
        # bm25() returns negative values; negate so higher = more relevant.
        sql = (
            "SELECT t.thought_id, -bm25(thought_fts) AS score "  # noqa: S608
            "FROM thought_fts "
            "JOIN thought t ON t.rowid = thought_fts.rowid "
            "WHERE thought_fts MATCH ? "
            "AND (t.expires_at IS NULL OR t.expires_at > ?) "
            # Freshness floor: retired REFLECTIONs are excluded so a stale
            # synthesis does not out-rank fresh relevant thoughts.
            "AND NOT (t.thought_type = 'REFLECTION' AND t.lifecycle_status != 'ACTIVE')"
            # Archived-exclusion: forgotten (archived) thoughts leave the default
            # candidate set unless the caller opts in via include_archived.
            f"{archived_sql} "
            f"{filter_sql}"
            # Deterministic total order: BM25 first, then canonical thought_id.
            "ORDER BY score DESC, t.thought_id ASC "
            "LIMIT ?"
        )
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        try:
            cursor = await self._db.execute(
                sql,
                (normalized_query, now_iso, *filter_params, top_k),
            )
            rows = await cursor.fetchall()
        except OperationalError as exc:
            # The primary MATCH is invalid FTS5. This branch is REACHABLE by
            # real input, by design: a *balanced* quoted phrase carrying
            # adjacent hazardous punctuation (e.g. ``"forum"?``) classifies as
            # expert, so its primary normalization is a deliberate — but
            # invalid — expert expression. The counter therefore *does* increment
            # for real queries; it is a designed, surfaced recovery, not a
            # "never happens" guard. Surface the failure via the counter, then
            # recover instead of silently degrading: re-normalize the *original*
            # query through the bare (sanitizing) path — whose output is always
            # a syntactically valid MATCH (unsafe characters dropped, wildcards
            # collapsed to legal prefix markers, and any exposed uppercase
            # AND/OR/NOT phrase-quoted so FTS5 cannot read it as an operator) —
            # and retry the MATCH once with it.
            self._fts_match_failure_count += 1
            # No query text and no ``exc_info``: SQLite's own FTS5 syntax error
            # names the offending token (e.g. ``fts5: syntax error near
            # "AND"``), which can quote content straight out of the query. A
            # digest was considered and rejected: an unsalted hash of a short,
            # low-entropy search query (often one or two words) is a
            # dictionary lookup away from the plaintext, not a one-way
            # fingerprint. Log only non-content facts -- length, the
            # exception's type, and SQLite's own error name -- so an operator
            # still sees that a fallback happened, how often, and what kind of
            # error caused it.
            logger.warning(
                "FTS MATCH failed for a query of length %d; retrying via "
                "sanitized bare-mode fallback [error type=%s, sqlite error=%s]",
                len(normalized_query),
                type(exc).__name__,
                getattr(exc, "sqlite_errorname", None),
            )
            fallback_query = _normalize_fts_query_bare(query)
            if not fallback_query:
                await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
                return []
            try:
                cursor = await self._db.execute(
                    sql,
                    (fallback_query, now_iso, *filter_params, top_k),
                )
                rows = await cursor.fetchall()
            except OperationalError as exc:  # pragma: no cover - unreachable for real input
                # Effectively unreachable for real input — unlike the primary
                # failure above, which is a designed, counted recovery. The bare
                # path emits only sanitized, wildcard-collapsed, operator-quoted
                # OR-terms, so its MATCH is always valid FTS5 (an 80k-string
                # star-dense + punctuation/quote/operator fuzz finds no input
                # that reaches here). This is defense-in-depth against an
                # unforeseen residual only: degrade to no FTS hits rather than
                # propagate.
                #
                # Same content discipline as the primary failure above: no
                # query text, no ``exc_info`` (SQLite's own error message can
                # quote the offending token), and no digest either -- an
                # unsalted hash of a short query is reversible by dictionary
                # lookup. Length, exception type and SQLite's error name only.
                logger.warning(
                    "FTS bare-mode fallback also failed for a query of length %d; "
                    "returning no FTS results [error type=%s, sqlite error=%s]",
                    len(fallback_query),
                    type(exc).__name__,
                    getattr(exc, "sqlite_errorname", None),
                )
                await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
                return []
        results = [(row["thought_id"], float(row["score"])) for row in rows]
        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
        return results

    # ------------------------------------------------------------------
    # MindQL execution
    # ------------------------------------------------------------------

    async def execute_mindql(
        self,
        query: MindQLQuery,
        *,
        extensions: dict[str, MindQLExtension] | None = None,
    ) -> MindQLResult:
        """Execute an already-parsed MindQL query against this store's connection.

        This is the store-level entry point for the MindQL execution
        contract: it lets a caller whose connection is owned by this store
        run MindQL without reaching into store internals. Parse the query
        first with :func:`engrava.mindql.parse`.

        This method performs **no command-level policy filtering** — it will
        execute whatever command the parsed query carries (``FIND``,
        ``COUNT``, ``SELECT``, or a registered extension command). Callers
        that need to restrict the command set (for example an
        over-the-wire consumer exposing ``FIND`` only) **must** validate
        ``query.command`` *before* calling this method.

        Args:
            query: A parsed MindQL query.
            extensions: Optional registered MindQL extension commands. When
                omitted, no extension commands are available. Callers that
                expose extension commands supply their own map (the store
                holds no extension-command registry of its own).

        Returns:
            The ``MindQLResult`` produced by the executor, carrying
            ``columns``, ``rows``, ``count``, and the executed ``command``.

        Raises:
            MindQLParseError: If the executor rejects the query at
                execution time (for example a ``SELECT`` whose raw SQL is
                not a ``SELECT`` statement, or a ``FIND`` naming a table,
                column, sort direction or ``LIMIT``/``OFFSET`` outside what
                the executor will interpolate). A directly constructed
                ``MindQLQuery`` is validated here just as a parsed one is.

        """
        from engrava.mindql.executor import MindQLExecutor  # noqa: PLC0415

        executor = MindQLExecutor(self._db, extensions=extensions or {})
        return await executor.execute(query)

    # ------------------------------------------------------------------
    # Hybrid search (FTS5 + vector + recency fusion)
    # ------------------------------------------------------------------

    def _resolve_hybrid_defaults(
        self,
        *,
        fts_weight: float | None,
        vector_weight: float | None,
        recency_weight: float | None,
        recency_half_life: int | None,
        priority_weight: float | None = None,
        graph_weight: float | None = None,
    ) -> tuple[float, float, float, int, float, float]:
        """Resolve per-call hybrid settings against configured defaults.

        Args:
            fts_weight: Optional per-call FTS weight override.
            vector_weight: Optional per-call vector weight override.
            recency_weight: Optional per-call recency weight override.
            recency_half_life: Optional per-call recency half-life override.
            priority_weight: Optional per-call priority weight override.
            graph_weight: Optional per-call graph signal weight override.

        Returns:
            Resolved ``(fts_weight, vector_weight, recency_weight,
            recency_half_life, priority_weight, graph_weight)``.

        Raises:
            ValueError: If any weight is negative or half-life is invalid.

        """
        search_config = self._search_config

        resolved_fts_weight = (
            fts_weight
            if fts_weight is not None
            else (search_config.default_fts_weight if search_config is not None else 0.3)
        )
        resolved_vector_weight = (
            vector_weight
            if vector_weight is not None
            else (search_config.default_vector_weight if search_config is not None else 0.55)
        )
        resolved_recency_weight = (
            recency_weight
            if recency_weight is not None
            else (search_config.default_recency_weight if search_config is not None else 0.0)
        )
        resolved_recency_half_life = (
            recency_half_life
            if recency_half_life is not None
            else (search_config.recency_half_life if search_config is not None else 50)
        )
        resolved_priority_weight = (
            priority_weight
            if priority_weight is not None
            else (search_config.default_priority_weight if search_config is not None else 0.05)
        )
        resolved_graph_weight = (
            graph_weight
            if graph_weight is not None
            else (search_config.default_graph_weight if search_config is not None else 0.0)
        )

        if resolved_fts_weight < 0.0:
            msg = "fts_weight must be non-negative"
            raise ValueError(msg)
        if resolved_vector_weight < 0.0:
            msg = "vector_weight must be non-negative"
            raise ValueError(msg)
        if resolved_recency_weight < 0.0:
            msg = "recency_weight must be non-negative"
            raise ValueError(msg)
        if resolved_recency_half_life < 1:
            msg = "recency_half_life must be a positive integer"
            raise ValueError(msg)
        if resolved_priority_weight < 0.0:
            msg = "priority_weight must be non-negative"
            raise ValueError(msg)
        if resolved_graph_weight < 0.0:
            msg = "graph_weight must be non-negative"
            raise ValueError(msg)

        return (
            resolved_fts_weight,
            resolved_vector_weight,
            resolved_recency_weight,
            resolved_recency_half_life,
            resolved_priority_weight,
            resolved_graph_weight,
        )

    @staticmethod
    def _redistribute_hybrid_weights(
        *,
        fts_active: bool,
        vector_active: bool,
        recency_active: bool,
        priority_active: bool = False,
        graph_active: bool = False,
        fts_weight: float,
        vector_weight: float,
        recency_weight: float,
        priority_weight: float = 0.0,
        graph_weight: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        """Redistribute disabled-signal weights across active components."""
        active_weight = 0.0
        if fts_active:
            active_weight += fts_weight
        if vector_active:
            active_weight += vector_weight
        if recency_active:
            active_weight += recency_weight
        if priority_active:
            active_weight += priority_weight
        if graph_active:
            active_weight += graph_weight

        if active_weight == 0.0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            (fts_weight / active_weight) if fts_active else 0.0,
            (vector_weight / active_weight) if vector_active else 0.0,
            (recency_weight / active_weight) if recency_active else 0.0,
            (priority_weight / active_weight) if priority_active else 0.0,
            (graph_weight / active_weight) if graph_active else 0.0,
        )

    async def _load_recency_scores(
        self,
        *,
        thought_ids: set[str],
        current_cycle: int,
        recency_half_life: int,
    ) -> dict[str, float]:
        """Load recency scores for a candidate set of thought IDs."""
        if not thought_ids:
            return {}

        decay_rate = math.log(2) / recency_half_life
        placeholders = ", ".join("?" for _ in thought_ids)
        sql = (
            f"SELECT thought_id, updated_cycle FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders})"
        )
        cursor = await self._db.execute(sql, list(thought_ids))
        rows = await cursor.fetchall()

        scores: dict[str, float] = {}
        for row in rows:
            thought_id = row["thought_id"]
            updated_cycle = int(row["updated_cycle"])
            age = max(current_cycle - updated_cycle, 0)
            scores[thought_id] = math.exp(-decay_rate * age)
        return scores

    async def _load_transaction_recency_scores(
        self,
        *,
        thought_ids: set[str],
        now: datetime.datetime,
        half_life_seconds: float,
    ) -> dict[str, float]:
        """Load transaction-time recency scores for a candidate set of thought IDs.

        The transaction-time analogue of :meth:`_load_recency_scores`: each row's
        freshness is measured from its ``updated_at`` (falling back to
        ``created_at`` when ``updated_at`` is ``NULL``) against the
        caller-supplied ``now`` instant, in wall-clock seconds — the store reads
        no host clock. The score is
        ``exp(-ln2 * age_seconds / half_life_seconds)`` with
        ``age_seconds = max((now - ts), 0)`` (a future-dated row clamps to age
        ``0``), the same exponential-half-life shape the cycle scorer uses.

        A row whose timestamp is missing or malformed (legacy / imported data)
        scores the deterministic minimum (:data:`_MIN_RECENCY_SCORE`) — treated
        as maximally old, never a crash.

        Args:
            thought_ids: Candidate thought IDs to score.
            now: The caller-supplied "now" instant, already UTC-normalised.
            half_life_seconds: Positive wall-clock half-life, in seconds.

        Returns:
            Mapping of ``thought_id`` to a recency score in ``[0.0, 1.0]``.

        """
        if not thought_ids:
            return {}

        decay_rate = math.log(2) / half_life_seconds
        placeholders = ", ".join("?" for _ in thought_ids)
        sql = (
            f"SELECT thought_id, updated_at, created_at FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders})"
        )
        cursor = await self._db.execute(sql, list(thought_ids))
        rows = await cursor.fetchall()

        scores: dict[str, float] = {}
        for row in rows:
            thought_id = str(row["thought_id"])
            raw_ts = row["updated_at"] if row["updated_at"] is not None else row["created_at"]
            ts = _parse_row_timestamp(raw_ts)
            if ts is None:
                scores[thought_id] = _MIN_RECENCY_SCORE
                continue
            age_seconds = max((now - ts).total_seconds(), 0.0)
            scores[thought_id] = math.exp(-decay_rate * age_seconds)
        return scores

    async def _load_priority_scores(
        self,
        *,
        thought_ids: set[str],
    ) -> dict[str, float]:
        """Load priority-boost scores for a candidate set of thought IDs.

        Maps each thought's ``priority`` enum value to a boost
        multiplier defined in ``SearchConfig``.  Thoughts whose
        priority is ``NULL`` or unknown receive a neutral score of
        ``0.0``.

        Args:
            thought_ids: Candidate thought IDs to score.

        Returns:
            Mapping of ``thought_id`` → priority boost score in
            ``[0.0, 1.0]``.

        """
        if not thought_ids:
            return {}

        search_config = self._search_config
        boost_map: dict[str, float] = {
            Priority.P1: search_config.priority_boost_p1 if search_config else 1.0,
            Priority.P2: search_config.priority_boost_p2 if search_config else 0.6,
            Priority.P3: search_config.priority_boost_p3 if search_config else 0.3,
            Priority.P4: search_config.priority_boost_p4 if search_config else 0.0,
        }

        placeholders = ", ".join("?" for _ in thought_ids)
        sql = (
            f"SELECT thought_id, priority FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders})"
        )
        cursor = await self._db.execute(sql, list(thought_ids))
        rows = await cursor.fetchall()

        scores: dict[str, float] = {}
        for row in rows:
            thought_id = row["thought_id"]
            priority_val = row["priority"]
            scores[thought_id] = boost_map.get(priority_val, 0.0)
        return scores

    async def _fetch_candidate_adjacency(
        self,
        all_ids: list[str],
        *,
        max_neighbors: int,
    ) -> dict[str, list[tuple[str, float]]]:
        """Fetch each candidate's top-``max_neighbors`` 1-hop neighbours, bounded in SQL.

        The candidate-ID list is queried in 450-wide chunks (SQLite's bound
        parameter ceiling makes one query per chunk necessary once the
        candidate pool is large). Each chunk's query:

        1. Builds every ``(candidate, neighbour, weight)`` triple touching a
           candidate in that chunk, from either edge direction.
        2. Ranks each candidate's triples by ``weight`` descending (ties
           broken by ``edge_id`` for a deterministic order) with a single
           ``ROW_NUMBER() OVER (PARTITION BY candidate_id ...)`` — one window
           per candidate across *both* directions combined, not one per
           direction, so a candidate with strong edges on both sides isn't
           handed up to ``2 * max_neighbors`` rows.
        3. Keeps only rows within the per-candidate rank cutoff, so no more
           than ``max_neighbors`` rows per candidate ever cross back into
           Python — the SQL layer enforces the bound, not a Python-side
           ``sorted(...)[:max_neighbors]`` slice applied after everything
           was already fetched.

        Chunking also fixes cross-chunk double-counting as a structural
        property rather than a post-hoc deduplication step: a candidate ID
        belongs to exactly one chunk (the chunks are a strict partition of
        ``all_ids``), and each chunk's query only ranks/returns rows for
        *its own* candidates — a candidate's neighbour list is therefore
        computed by exactly one chunk's query, however many other chunks'
        `IN` predicates an edge touching it also happens to satisfy.

        Args:
            all_ids: Candidate thought IDs to fetch adjacent edges for.
            max_neighbors: Maximum neighbours kept per candidate, combined
                across both edge directions.

        Returns:
            Mapping of candidate thought_id to its bounded, weight-ordered
            ``(neighbour_id, edge_weight)`` list.

        """
        chunk_size = 450
        adjacency: dict[str, list[tuple[str, float]]] = {}
        for i in range(0, len(all_ids), chunk_size):
            chunk = all_ids[i : i + chunk_size]
            placeholders = ", ".join("?" for _ in chunk)
            sql = (
                "WITH matched AS ("  # noqa: S608
                "  SELECT from_thought_id AS candidate_id, to_thought_id AS neighbour_id, "
                "         weight, edge_id "
                f"  FROM edge WHERE from_thought_id IN ({placeholders}) "
                "  UNION ALL "
                "  SELECT to_thought_id AS candidate_id, from_thought_id AS neighbour_id, "
                "         weight, edge_id "
                f"  FROM edge WHERE to_thought_id IN ({placeholders}) "
                ") "
                "SELECT candidate_id, neighbour_id, weight FROM ("
                "  SELECT candidate_id, neighbour_id, weight, "
                "         ROW_NUMBER() OVER ("
                "             PARTITION BY candidate_id ORDER BY weight DESC, edge_id"
                "         ) AS rn "
                "  FROM matched "
                ") WHERE rn <= ? "
                "ORDER BY candidate_id, weight DESC"
            )
            params: list[object] = [*chunk, *chunk, max_neighbors]
            cursor = await self._db.execute(sql, params)
            rows = await cursor.fetchall()
            for r in rows:
                candidate_id = str(r["candidate_id"])
                neighbour_id = str(r["neighbour_id"])
                weight = float(r["weight"])
                adjacency.setdefault(candidate_id, []).append((neighbour_id, weight))
        return adjacency

    async def _load_graph_signal(
        self,
        *,
        candidate_scores: dict[str, float],
        graph_edge_decay: float,
        max_neighbors: int,
    ) -> dict[str, float]:
        """Compute 1-hop-weighted graph boost for candidate thoughts.

        For each candidate, look up its 1-hop neighbours via edges.
        If a neighbour is also in the candidate pool, propagate its
        semantic base score (``max(fts, vector)``) weighted by
        ``edge.weight * graph_edge_decay``.

        Only content signals (FTS and vector) propagate
        through the graph.  Priority, recency, and graph scores are
        excluded to prevent hub-cascade effects.

        Args:
            candidate_scores: Semantic-only base scores
                (``max(fts, vector)`` per thought) for each candidate.
            graph_edge_decay: Decay factor applied to neighbour boost.
            max_neighbors: Maximum neighbours to consider per candidate.

        Returns:
            Mapping of thought_id to graph boost value.

        """
        if not candidate_scores:
            return {}

        all_ids = list(candidate_scores.keys())
        adjacency = await self._fetch_candidate_adjacency(all_ids, max_neighbors=max_neighbors)

        # Compute boost per candidate — the adjacency fetched above is
        # already bounded to `max_neighbors` per candidate and ordered by
        # weight, so no further Python-side sorting or slicing is needed.
        boosts: dict[str, float] = {}
        for cid in all_ids:
            boost = 0.0
            for neighbour_id, edge_weight in adjacency.get(cid, []):
                neighbour_base = candidate_scores.get(neighbour_id, 0.0)
                boost += edge_weight * neighbour_base * graph_edge_decay
            if boost > 0.0:
                boosts[cid] = boost

        return boosts

    async def _find_healthy_reflection_seeds(
        self,
        *,
        combined: dict[str, float],
        expansion_top_n: int,
        reflection_source_ceiling: int,
    ) -> list[str]:
        """Return top-ranked REFLECTION IDs from ``combined`` that pass the ceiling guard.

        Preserves score order from ``combined`` (SQL ``IN (…)`` does not guarantee
        row order, so the result is re-ranked against the original scores).
        REFLECTIONs with more ``CONSOLIDATED_FROM`` sources than
        ``reflection_source_ceiling`` are excluded to guard against giant-cluster
        pathology.

        Args:
            combined: Current score mapping (thought_id → score).
            expansion_top_n: Maximum number of healthy REFLECTION seeds to return.
            reflection_source_ceiling: Skip REFLECTIONs with strictly more
                ``CONSOLIDATED_FROM`` sources than this threshold.

        Returns:
            Ordered list of healthy REFLECTION IDs (best-scored first),
            up to ``expansion_top_n`` entries. Empty list when none qualify.

        """
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        window = ranked[: expansion_top_n * 5]
        candidate_ids = [tid for tid, _ in window]

        placeholders = ", ".join("?" for _ in candidate_ids)
        cursor = await self._db.execute(
            f"SELECT thought_id, thought_type FROM thought"  # noqa: S608
            f" WHERE thought_id IN ({placeholders})",
            candidate_ids,
        )
        rows = await cursor.fetchall()
        type_lookup: dict[str, str] = {str(r["thought_id"]): str(r["thought_type"]) for r in rows}
        reflection_candidates = [tid for tid, _ in window if type_lookup.get(tid) == "REFLECTION"][
            :expansion_top_n
        ]

        if not reflection_candidates:
            return []

        rc_placeholders = ", ".join("?" for _ in reflection_candidates)
        cursor = await self._db.execute(
            f"SELECT from_thought_id, COUNT(*) AS n FROM edge"  # noqa: S608
            f" WHERE edge_type = 'CONSOLIDATED_FROM'"
            f" AND from_thought_id IN ({rc_placeholders})"
            f" GROUP BY from_thought_id",
            reflection_candidates,
        )
        count_rows = await cursor.fetchall()
        source_counts: dict[str, int] = {str(r["from_thought_id"]): int(r["n"]) for r in count_rows}
        healthy = [
            rid
            for rid in reflection_candidates
            if source_counts.get(rid, 0) <= reflection_source_ceiling
        ]
        skipped = len(reflection_candidates) - len(healthy)
        if skipped > 0:
            logger.info(
                "expansion guard: skipped %d reflection(s) exceeding "
                "source_ceiling=%d (counts: %s)",
                skipped,
                reflection_source_ceiling,
                {rid: source_counts[rid] for rid in reflection_candidates if rid not in healthy},
            )
        return healthy

    async def _expand_via_consolidated_from(  # noqa: C901
        self,
        *,
        combined: dict[str, float],
        expansion_top_n: int,
        propagation_factor: float,
        max_sources_per_reflection: int,
        reflection_source_ceiling: int,
        expansion_sources: dict[str, str] | None = None,
        include_archived: bool = False,
        _filter_clause: tuple[str, list[object]] | None = None,
    ) -> int:
        """Expand candidate pool by pulling source OBSERVATIONs from top REFLECTIONs.

        Identifies the top-ranked REFLECTIONs in ``combined`` (preserving their
        score-rank), traverses their outgoing ``CONSOLIDATED_FROM`` edges, and
        adds each **OBSERVATION**-type source to ``combined`` with::

            propagated_score = parent_score * propagation_factor * edge_weight

        If a source is already in ``combined`` the higher score wins.
        Non-OBSERVATION targets (REFLECTION, TASK, …) are silently skipped —
        only factual observations should be pulled through.
        REFLECTIONs with more than ``reflection_source_ceiling`` sources
        are skipped to guard against single-link chaining pathology
        (giant clusters whose centroid is too generic to be useful as an
        expansion seed).

        Args:
            combined: Current score mapping (thought_id → score). Modified
                in place.
            expansion_top_n: How many top-ranked REFLECTIONs to use as seeds.
            propagation_factor: Scalar applied to parent score during
                propagation (< 1.0 keeps sources below the REFLECTION).
            max_sources_per_reflection: At most this many source OBSs are
                pulled per REFLECTION, ordered by descending edge weight.
            reflection_source_ceiling: REFLECTIONs with strictly more sources
                than this value are skipped entirely.
            expansion_sources: Optional output mapping populated with
                ``source_id -> parent_reflection_id`` for candidates that
                were newly introduced by graph expansion.
            include_archived: When ``False`` (the default) archived source
                OBSERVATIONs are never injected into ``combined`` — forwarded to
                :meth:`_filter_observation_ids` so an archived observation cannot
                leak back into the fused set via graph expansion even though the
                seed REFLECTION is ACTIVE. When ``True`` archived sources are
                re-admitted, consistent with the arms' escape hatch.
            _filter_clause: Internal. A compiled ``(sql_fragment, params)``
                metadata predicate forwarded to :meth:`_filter_observation_ids`
                so expansion-pulled OBSERVATIONs that fall outside the active
                filter are never injected into ``combined``.

        Returns:
            Number of new or updated entries written into ``combined``.
            Zero means no expansion occurred (no-op).

        """
        if not combined:
            return 0

        healthy = await self._find_healthy_reflection_seeds(
            combined=combined,
            expansion_top_n=expansion_top_n,
            reflection_source_ceiling=reflection_source_ceiling,
        )
        if not healthy:
            return 0

        # Traverse CONSOLIDATED_FROM edges (top sources by weight).
        h_placeholders = ", ".join("?" for _ in healthy)
        cursor = await self._db.execute(
            f"SELECT from_thought_id, to_thought_id, weight FROM edge"  # noqa: S608
            f" WHERE edge_type = 'CONSOLIDATED_FROM'"
            f" AND from_thought_id IN ({h_placeholders})"
            f" ORDER BY from_thought_id, weight DESC",
            healthy,
        )
        edge_rows = await cursor.fetchall()

        obs_ids = await self._filter_observation_ids(
            [str(r["to_thought_id"]) for r in edge_rows],
            include_archived=include_archived,
            _filter_clause=_filter_clause,
        )
        if not obs_ids:
            return 0

        # Group by parent and respect per-reflection cap; skip non-OBSERVATION targets.
        per_reflection: dict[str, list[tuple[str, float]]] = {}
        for row in edge_rows:
            parent_id = str(row["from_thought_id"])
            source_id = str(row["to_thought_id"])
            if source_id not in obs_ids:
                continue
            w = float(row["weight"])
            bucket = per_reflection.setdefault(parent_id, [])
            if len(bucket) < max_sources_per_reflection:
                bucket.append((source_id, w))

        # Propagate scores into combined; count newly written entries.
        added = 0
        for parent_id, sources in per_reflection.items():
            parent_score = combined.get(parent_id, 0.0)
            for source_id, edge_weight in sources:
                propagated = parent_score * propagation_factor * edge_weight
                existing = combined.get(source_id, 0.0)
                was_present = source_id in combined
                if propagated > existing:
                    combined[source_id] = propagated
                    added += 1
                    if expansion_sources is not None and not was_present:
                        expansion_sources[source_id] = parent_id

        return added

    async def _filter_observation_ids(
        self,
        candidate_ids: list[str],
        *,
        include_archived: bool = False,
        _filter_clause: tuple[str, list[object]] | None = None,
    ) -> frozenset[str]:
        """Return the subset of ``candidate_ids`` whose thought_type is OBSERVATION.

        Used by ``_expand_via_consolidated_from`` to strip non-factual
        targets (TASK, REFLECTION, …) from the expansion pool before
        propagating scores.

        The CONSOLIDATED_FROM expansion pulls brand-new OBSERVATION rows that
        never passed an arm's ``WHERE``, so the same eligibility gates the arms
        apply must be re-applied here or an ineligible row would be injected into
        the result set:

        * **Expiry** — a source OBSERVATION whose ``expires_at`` has passed is
          dropped, exactly as the FTS and vector arms drop expired rows; the
          "now" instant is read the same way the arms read it.
        * **Archived-exclusion** — an ACTIVE REFLECTION may still point (via
          ``CONSOLIDATED_FROM``) at a source OBSERVATION that has since been
          archived, so without this gate graph expansion would re-inject a
          forgotten observation the arms already excluded (unless
          ``include_archived`` opts it back in).
        * **Metadata predicate** — the effective ``filters`` / ``visibility``
          predicate, re-applied so an out-of-filter OBSERVATION is not injected.

        The retired-REFLECTION freshness floor needs no separate clause here: the
        query already restricts to ``thought_type = 'OBSERVATION'``, so no
        REFLECTION (retired or otherwise) can pass.

        Args:
            candidate_ids: Unfiltered list of target thought IDs.
            include_archived: When ``False`` (the default) archived source
                OBSERVATIONs are excluded from the expansion pool; when ``True``
                they are re-admitted (consistent with the arms' escape hatch).
                Expiry is always enforced regardless of this flag, matching the
                arms.
            _filter_clause: Internal. A compiled ``(sql_fragment, params)``
                metadata predicate (referencing the bare ``metadata_json``
                column) injected into the ``WHERE``.

        Returns:
            Frozenset containing only IDs of OBSERVATION-type thoughts that are
            unexpired and satisfy the active filter (and the archived-exclusion
            unless ``include_archived`` is set). Empty frozenset when
            ``candidate_ids`` is empty.

        """
        unique = list(dict.fromkeys(candidate_ids))  # deduplicate, preserve insertion order
        if not unique:
            return frozenset()
        filter_sql = ""
        filter_params: list[object] = []
        if _filter_clause is not None:
            filter_fragment, filter_params = _filter_clause
            filter_sql = f" AND {filter_fragment}"
        archived_sql = _archived_exclusion_sql(
            column="lifecycle_status",
            include_archived=include_archived,
        )
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        placeholders = ", ".join("?" for _ in unique)
        cursor = await self._db.execute(
            f"SELECT thought_id FROM thought"  # noqa: S608
            f" WHERE thought_type = 'OBSERVATION'"
            f" AND thought_id IN ({placeholders})"
            # Expiry gate: an expired source must not be re-injected via
            # expansion any more than the arms would surface it.
            f" AND (expires_at IS NULL OR expires_at > ?)"
            f"{archived_sql}"
            f"{filter_sql}",
            [*unique, now_iso, *filter_params],
        )
        rows = await cursor.fetchall()
        return frozenset(str(r["thought_id"]) for r in rows)

    async def _fallback_hybrid_results(
        self,
        *,
        top_k: int,
        current_cycle: int | None,
        recency_half_life: int,
        transaction_now: datetime.datetime | None = None,
        transaction_half_life_seconds: float = 0.0,
        filter_clause: tuple[str, list[object]] | None = None,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Fallback results when neither FTS nor vector search is usable.

        Ranks the candidate window by whichever recency axis is active:
        transaction time (``transaction_now`` supplied — the row window is
        pre-ordered by ``COALESCE(updated_at, created_at)`` so the truncation
        keeps the most-recently-written rows), cognitive cycle
        (``current_cycle`` supplied — pre-ordered by ``updated_cycle``), or
        neither (a flat ``0.0`` score). The two axes are mutually exclusive by
        the time this runs (``search_hybrid`` rejects both references upfront).

        ``filter_clause`` is the compiled ``filters=`` / ``visibility=``
        predicate; it is applied in-query so this query-less path enforces the
        same eligibility as the FTS and vector arms (an out-of-filter row never
        enters the result set). The predicate is threaded here directly — not
        through the public ``list_thoughts`` — so raw-SQL fragments stay off the
        public API surface.

        ``include_archived`` mirrors the arms' escape hatch: unless it is set,
        archived thoughts are excluded from this query-less window too, so the
        all-signals-off fallback honours the archived-exclusion invariant.
        """
        clauses = ["(expires_at IS NULL OR expires_at > ?)"]
        params: list[object] = [datetime.datetime.now(datetime.UTC).isoformat()]
        if filter_clause is not None:
            fragment, filter_params = filter_clause
            clauses.append(fragment)
            params.extend(filter_params)
        if not include_archived:
            clauses.append(f"lifecycle_status != '{LifecycleStatus.ARCHIVED.value}'")
        where = " AND ".join(clauses)
        params.append(top_k)
        # Pre-order the truncation window by the active recency axis so the
        # ``LIMIT`` keeps the freshest rows: transaction time orders by the
        # write timestamp (NULLs sort last under DESC), cycle time by the
        # cognitive cycle. The clause is one of two fixed literals — never
        # caller input — so it carries no injection surface.
        order_by = (
            "COALESCE(updated_at, created_at) DESC"
            if transaction_now is not None
            else "updated_cycle DESC"
        )
        cursor = await self._db.execute(
            "SELECT thought_id, thought_type, lifecycle_status, updated_cycle, "  # noqa: S608
            f"updated_at, created_at FROM thought WHERE {where} ORDER BY {order_by} LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        # Apply the REFLECTION freshness floor consistently with the FTS and
        # vector paths: a retired REFLECTION must not surface here either.
        live = [
            row
            for row in rows
            if not (
                str(row["thought_type"]) == ThoughtType.REFLECTION.value
                and str(row["lifecycle_status"]) != LifecycleStatus.ACTIVE.value
            )
        ]
        if transaction_now is not None:
            decay_rate = math.log(2) / transaction_half_life_seconds
            ranked = []
            for row in live:
                raw_ts = row["updated_at"] if row["updated_at"] is not None else row["created_at"]
                ts = _parse_row_timestamp(raw_ts)
                score = (
                    _MIN_RECENCY_SCORE
                    if ts is None
                    else math.exp(-decay_rate * max((transaction_now - ts).total_seconds(), 0.0))
                )
                ranked.append((str(row["thought_id"]), score))
            ranked.sort(key=lambda item: item[1], reverse=True)
            return ranked
        if current_cycle is None:
            return [(str(row["thought_id"]), 0.0) for row in live]

        decay_rate = math.log(2) / recency_half_life
        ranked = [
            (
                str(row["thought_id"]),
                math.exp(-decay_rate * max(current_cycle - int(row["updated_cycle"]), 0)),
            )
            for row in live
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    async def _resolve_hybrid_state(
        self,
        *,
        query_text: str,
        query_vector: list[float] | None,
        current_cycle: int | None,
        transaction_now: datetime.datetime | None,
        recency_weight: float,
    ) -> tuple[bool, list[float] | None, bool]:
        """Resolve active hybrid-search components for the current query.

        Recency is active when a reference for **either** axis is present — the
        cognitive ``current_cycle`` or the transaction-time ``transaction_now``
        (never both; the two are mutually exclusive by the time this runs) — and
        the recency weight is positive.
        """
        if not self._fts_probed:
            await self._probe_fts()

        fts_active = bool(self._fts_available and query_text and query_text.strip())

        effective_vector = query_vector
        if effective_vector is None and self._embedding_provider is not None and query_text.strip():
            await self._ensure_query_prefix_pairs()
            effective_vector = await _embed_query(self._embedding_provider, query_text)

        recency_reference_present = current_cycle is not None or transaction_now is not None
        recency_active = recency_reference_present and recency_weight > 0.0
        return (fts_active, effective_vector, recency_active)

    async def _fetch_collapse_unit_keys(
        self,
        *,
        thought_ids: list[str],
        paths: tuple[str, ...],
    ) -> dict[str, tuple[object, ...] | None]:
        """Fetch the de-fragmentation unit key per candidate id.

        Issues one ``SELECT thought_id, json_extract(metadata_json, ?)[, …]``
        over the candidate ids (same shape as the existing REFLECTION id
        lookup). Each component is ``json_valid``-guarded, so a row holding
        malformed ``metadata_json`` yields all-NULL components and is treated
        as key-less. A unit key is ``None`` (key-less → its own unit, never
        collapsed) when any component is NULL — a partial composite key is not
        a shared identity.

        Args:
            thought_ids: Candidate ids to look up (the fused candidate set).
            paths: The validated, ordered unit-key paths.

        Returns:
            Map from ``thought_id`` to its unit-key tuple, or ``None`` for a
            key-less / partial-key / malformed-metadata row. Ids absent from
            the table are simply omitted (callers treat missing as ``None``).

        """
        if not thought_ids:
            return {}
        placeholders = ", ".join("?" for _ in thought_ids)
        # Each path projects ``CASE WHEN json_valid(metadata_json) THEN
        # json_extract(metadata_json, ?) ELSE NULL END`` so malformed JSON
        # never aborts the query and folds to a NULL component. ``paths`` and
        # ``thought_ids`` bind as parameters; the column is a fixed literal.
        projections = ", ".join(
            "CASE WHEN json_valid(metadata_json) THEN json_extract(metadata_json, ?) ELSE NULL END"
            for _ in paths
        )
        params: list[object] = [*paths, *thought_ids]
        cursor = await self._db.execute(
            f"SELECT thought_id, {projections} FROM thought"  # noqa: S608
            f" WHERE thought_id IN ({placeholders})",
            params,
        )
        rows = await cursor.fetchall()
        unit_keys: dict[str, tuple[object, ...] | None] = {}
        n = len(paths)
        for row in rows:
            thought_id = str(row[0])
            components = tuple(row[i + 1] for i in range(n))
            # Any NULL component ⇒ key-less (own unit, never collapsed).
            unit_keys[thought_id] = None if any(c is None for c in components) else components
        return unit_keys

    async def _apply_reflection_filter_boost(
        self,
        *,
        combined: dict[str, float],
        include_reflections: bool,
        resolved_reflection_boost: float,
        needs_reflection_ids: bool,
    ) -> set[str]:
        """Resolve REFLECTION ids for ``combined`` and apply filter/boost.

        Shared by the FTS/vector-active fusion path and the query-less
        fallback: both need the same REFLECTION id lookup, the same
        ``include_reflections=False`` hard filter, and the same
        ``reflection_boost`` scaling before either path computes its own
        (path-specific) ranked order.

        Args:
            combined: Candidate ``thought_id -> score`` map, mutated in
                place — REFLECTION ids are popped (``include_reflections``
                ``False``) or score-scaled (``resolved_reflection_boost !=
                1.0``).
            include_reflections: When ``False``, REFLECTION ids are removed
                from ``combined``.
            resolved_reflection_boost: The already-resolved boost factor
                (see ``search_hybrid``'s ``reflection_boost`` argument).
            needs_reflection_ids: Whether a REFLECTION id lookup is needed
                at all (``False`` short-circuits the query — neither the
                filter, the boost, nor the caller's ``reflection_topk_cap``
                check requires it).

        Returns:
            The resolved REFLECTION id set (empty when none matched or
            ``needs_reflection_ids`` was ``False``).

        """
        reflection_ids: set[str] = set()
        if needs_reflection_ids and combined:
            candidate_ids = list(combined)
            placeholders = ", ".join("?" for _ in candidate_ids)
            cursor = await self._db.execute(
                f"SELECT thought_id FROM thought"  # noqa: S608
                f" WHERE thought_type = 'REFLECTION'"
                f" AND thought_id IN ({placeholders})",
                candidate_ids,
            )
            rows = await cursor.fetchall()
            reflection_ids = {str(r["thought_id"]) for r in rows}

        if not include_reflections:
            for rid in reflection_ids:
                combined.pop(rid, None)
        elif resolved_reflection_boost != 1.0 and reflection_ids:
            for rid in reflection_ids:
                if rid in combined:
                    combined[rid] = combined[rid] * resolved_reflection_boost
        return reflection_ids

    async def _apply_collapse_and_topk_cap(
        self,
        *,
        ranked: list[tuple[str, float]],
        collapse_paths: tuple[str, ...] | None,
        collapse_max_per_unit: int | None,
        reflection_ids: set[str],
        resolved_reflection_topk_cap: float,
        include_reflections: bool,
        top_k: int,
    ) -> tuple[list[tuple[str, float]], int]:
        """Apply collapse-by-unit retention, then ``reflection_topk_cap``.

        ``ranked`` must already be in the caller's deterministic total
        order (score descending, under whatever tie-break rule that path
        uses). Shared by the FTS/vector-active fusion path and the
        query-less fallback, so a caller-requested ``collapse_key`` /
        ``reflection_topk_cap`` is honoured identically regardless of which
        arms produced the candidates.

        Args:
            ranked: Candidates already sorted into the caller's total
                order.
            collapse_paths: Validated de-fragmentation unit-key paths, or
                ``None`` to leave ``ranked`` unchanged.
            collapse_max_per_unit: Intra-unit retention depth (see
                ``search_hybrid``'s docstring); inert unless
                ``collapse_paths`` is also set.
            reflection_ids: REFLECTION ids among ``ranked`` (from
                :meth:`_apply_reflection_filter_boost`).
            resolved_reflection_topk_cap: The already-resolved cap fraction.
            include_reflections: When ``False`` the cap step is skipped —
                REFLECTIONs were already removed upstream.
            top_k: Maximum number of final results.

        Returns:
            The ``(final_results, reflections_evicted)`` pair.

        """
        # --- De-fragmentation retention-by-unit + backfill ---
        # Runs AFTER fusion/ranking and REFLECTION boost, BEFORE the
        # ``[:top_k]`` truncation and BEFORE reflection_topk_cap — the same
        # locus and shape as the cap's evict-and-backfill. It touches no
        # score and no candidate set: it only removes surplus lower-ranked
        # members of the same caller-defined unit so deeper distinct units
        # in ``ranked[top_k:]`` flow up into the window. ``collapse_max_per_unit``
        # sets how many members of a unit are kept: ``None`` => 1
        # (single-keeper collapse), an integer keeps that many.
        if collapse_paths is not None and ranked:
            unit_keys = await self._fetch_collapse_unit_keys(
                thought_ids=[tid for tid, _ in ranked],
                paths=collapse_paths,
            )
            ranked = _retain_ranked_by_unit(
                ranked,
                unit_keys,
                max_per_unit=1 if collapse_max_per_unit is None else collapse_max_per_unit,
            )
        final = ranked[:top_k]

        # --- reflection_topk_cap enforcement ---
        # Runs on the (possibly collapsed) ``ranked`` so the single backfill
        # source is the collapsed off-list pool — no unit is double-counted.
        reflections_evicted = 0
        if include_reflections and resolved_reflection_topk_cap < 1.0 and reflection_ids:
            _max_ref_slots = max(0, int(top_k * resolved_reflection_topk_cap))
            _ref_in_final = [
                (i, tid, s) for i, (tid, s) in enumerate(final) if tid in reflection_ids
            ]
            if len(_ref_in_final) > _max_ref_slots:
                _excess = len(_ref_in_final) - _max_ref_slots
                _to_evict = {
                    tid for _, tid, _ in sorted(_ref_in_final, key=lambda x: x[2])[:_excess]
                }
                _off_list_obs = [(tid, s) for tid, s in ranked[top_k:] if tid not in reflection_ids]
                if len(_off_list_obs) < _excess:
                    logger.warning(
                        "reflection_topk_cap: %d excess REFLECTION(s) to evict but only %d "
                        "off-list non-REFLECTION candidates available — partial enforcement",
                        _excess,
                        len(_off_list_obs),
                    )
                _fill = _off_list_obs[:_excess]
                _survivor_ids = {tid for tid, _ in final if tid not in _to_evict}
                _survivor_ids.update(tid for tid, _ in _fill)
                # Rebuild from ``ranked`` — the caller's own total order,
                # already collapse-retained above — instead of re-sorting the
                # survivors by ``(score, id)``. For the FTS/vector arms
                # ``ranked`` already *is* that ``(score desc, id asc)`` order
                # (built by ``_sort_scored_descending`` at the call site
                # below), so filtering it reproduces exactly what the re-sort
                # produced. The query-less fallback's ``ranked`` carries a
                # different order — recency- or priority-ranked, ties broken
                # by the DB's own pre-order — which a ``(score, id)`` re-sort
                # would silently discard in favour of id order. Filtering
                # ``ranked`` in place keeps whichever order it already
                # carries, for either caller.
                final = [item for item in ranked if item[0] in _survivor_ids][:top_k]
                # ``_to_evict`` REFLECTIONs are removed from the window
                # unconditionally (independent of how many backfill candidates
                # were available), so the evicted count is the excess.
                reflections_evicted = len(_to_evict)
                logger.info(
                    "reflection_topk_cap: evicted %d REFLECTION(s) from the top-%d window "
                    "(cap=%.3f, max reflection slots=%d)",
                    reflections_evicted,
                    top_k,
                    resolved_reflection_topk_cap,
                    _max_ref_slots,
                )

        return final, reflections_evicted

    async def search_hybrid(  # noqa: C901, PLR0912, PLR0915
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        *,
        top_k: int = 10,
        fts_weight: float | None = None,
        vector_weight: float | None = None,
        recency_weight: float | None = None,
        recency_half_life: int | None = None,
        current_cycle: int | None = None,
        recency_now: str | None = None,
        recency_now_half_life: int | None = None,
        fts_top_k: int = 50,
        vector_top_k: int = 50,
        priority_weight: float | None = None,
        graph_weight: float | None = None,
        graph_edge_decay: float | None = None,
        include_reflections: bool = True,
        reflection_boost: float | None = None,
        filters: MetadataFilter | None = None,
        visibility: VisibilityQueryFilter | None = None,
        collapse_key: str | Sequence[str] | None = None,
        collapse_max_per_unit: int | None = None,
        include_archived: bool = False,
    ) -> HybridSearchResult:
        """Hybrid search combining FTS5 + vector + recency + priority + graph signals.

        Calls ``search_fts()`` and ``search_similar()`` independently,
        normalizes BM25 scores to ``[0, 1]`` via min-max, computes
        exponential recency decay, applies priority boost, then adds
        1-hop-weighted graph boost, and returns merged results
        sorted by combined score.

        Optional ``filters`` / ``visibility`` scope the ranked query to rows
        whose ``metadata`` satisfies a typed predicate. The predicate is
        applied **in-arm, before each arm's limit** (and re-applied on the
        consolidation-expansion path), so an out-of-filter row never enters
        the candidate set, consumes an arm's budget, or contributes a signal
        — and a narrow filter is not starved by out-of-filter candidates.
        This is a **query capability, not a security boundary** (see
        ``visibility`` below).

        Recency has two separately-typed axes; a query selects **exactly one**
        reference (supplying both raises :class:`RecencyModeConflictError`):

            - **Cognitive-cycle recency** — ``current_cycle`` (explicit or
              resolved from a configured ``cycle_provider``); ages rows by
              ``updated_cycle``.
            - **Transaction-time recency** — ``recency_now`` (a caller-supplied
              ISO-8601 instant); ages rows by ``updated_at`` (falling back to
              ``created_at``) in wall-clock seconds. The store reads **no** host
              clock — a missing ``recency_now`` simply leaves this axis off.

        Graceful degradation:
            - If FTS5 unavailable or ``query_text`` empty → FTS skipped.
            - If ``query_vector`` is ``None`` and no provider → vector skipped.
            - If neither recency reference is present (no ``current_cycle`` /
              ``cycle_provider`` and no ``recency_now``) → recency skipped.
            - If ``priority_weight`` is ``0.0`` → priority skipped.
            - If ``graph_weight`` is ``0.0`` → graph skipped.
            - Disabled weights redistributed proportionally to active signals.
            - If all signals disabled → fallback to its own query over
              ``thought`` (see :meth:`_fallback_hybrid_results`), not a call
              to ``list_thoughts``.

        Args:
            query_text: Text query for FTS5 keyword search.
            query_vector: Embedding vector for similarity search.
                When ``None`` and an embedding provider is configured,
                the query text is auto-embedded.
            top_k: Maximum number of final merged results.
            fts_weight: Optional FTS5 fusion-weight override.
            vector_weight: Optional vector fusion-weight override.
            recency_weight: Optional recency fusion-weight override.
            recency_half_life: Optional recency half-life override.
            current_cycle: Current cycle number for **cognitive-cycle** recency.
                When omitted (``None``) *and* no ``recency_now`` is passed, it is
                pulled from a configured ``cycle_provider`` if one is set — an
                explicit value (including ``0``) always wins, and with neither a
                value nor a provider cycle recency is skipped. Mutually exclusive
                with ``recency_now``: passing an **explicit** ``current_cycle``
                together with ``recency_now`` raises
                :class:`RecencyModeConflictError`.
            recency_now: Caller-supplied "now" instant (ISO-8601) selecting
                **transaction-time** recency, which ages rows by ``updated_at``
                (falling back to ``created_at``) in wall-clock seconds. It takes
                precedence over a **passive** ``cycle_provider``: when
                ``recency_now`` is supplied and no explicit ``current_cycle`` was
                passed, the provider's ``current_cycle()`` is **not** called and
                cycle recency is off. Parsed and UTC-normalised via the shared
                temporal helper (a naive value is interpreted as UTC; the host
                timezone is never consulted); a malformed value raises
                :class:`InvalidRecencyArgumentError`. The store reads no host
                clock: omitting ``recency_now`` simply leaves this axis off (there
                is no "use current time" fallback). A row with a missing or
                malformed timestamp scores the deterministic minimum (treated as
                maximally old). ``None`` (default) keeps the existing behaviour
                byte-for-byte.
            recency_now_half_life: Optional per-call override for the
                transaction-time half-life, **in wall-clock seconds** (distinct
                from ``recency_half_life``, which is in cycles). Consulted only
                when ``recency_now`` is supplied; ``None`` uses
                ``SearchConfig.recency_now_half_life_seconds`` (default 604800 =
                7 days). Must be ``> 0``.
            fts_top_k: Max candidates from FTS5 before fusion.
            vector_top_k: Max candidates from vector search before fusion.
            priority_weight: Optional priority fusion-weight override.
            graph_weight: Optional graph signal fusion-weight override.
            graph_edge_decay: Optional graph edge decay override.
            include_reflections: When ``False``, REFLECTION thoughts are
                excluded from results.
            reflection_boost: Multiplier applied to REFLECTION thought
                scores. ``None`` uses the value from ``SearchConfig``
                (default ``1.0``).
            filters: Optional :class:`~engrava.domain.models.filters.MetadataFilter`
                — an ``AND`` of typed field predicates over ``metadata``.
                ``None`` (or an empty filter) leaves the candidate set
                unchanged. The predicate is applied in-arm before each arm's
                limit, so it never starves ``top_k``.
            visibility: Optional
                :class:`~engrava.domain.models.filters.VisibilityQueryFilter`
                — the bounded ``(visibility IN … [OR owner = …])`` shape for
                the "public-or-mine" pattern. **This is a query filter, not
                access control.** It performs no authentication,
                authorization, ownership validation, or write enforcement;
                the caller supplies (and can forge) ``owner``; it is
                bypassable by passing ``visibility=None``, by using another
                API, or by issuing raw SQL. It must **not** be used to protect
                tenant data — use a store per tenant (``EngravaManager``) for
                isolation and the commercial RBAC tier for shared-corpus
                access control.
            collapse_key: Optional de-fragmentation unit key — a single
                metadata path (``"$.session_turn"``) or an ordered sequence of
                paths forming a composite key (``["$.session_id",
                "$.turn_index"]``). When set, among the already-ranked
                candidates only the single highest-ranked row per unit reaches
                the result, and the slots that frees are backfilled by deeper
                *distinct* units — so the prompt sees one best row per
                caller-defined unit plus more distinct units, instead of many
                fragments of the same unit. This is a **presentation / de-dup
                convenience, not a filter and not isolation**: it does not
                change which rows are *eligible* (use ``filters`` /
                ``visibility`` for that). The collapse step itself mutates no
                score and only drops lower-ranked members of the same unit.
                Note, however, that *setting* ``collapse_key`` also widens the
                internal candidate pool (akin to a larger internal ``top_k``)
                to give backfill more depth; because the keyword arm's scores
                are min-max normalized over the candidate set, a wider pool can
                rescale the normalized fusion scores and shift the order among
                units. Only ``collapse_key=None`` leaves the candidate, score,
                and order path byte-identical to the unfiltered query. It is
                only as meaningful as the unit metadata the application writes —
                a row whose key is missing or holds malformed metadata is
                treated as its own unit and is never collapsed with another.
                Each path is validated at call time.
            collapse_max_per_unit: Optional cap on how many rows of a single
                ``collapse_key`` unit may reach the result — the intra-unit
                retention depth. Only takes effect **together with**
                ``collapse_key`` (there is no unit to retain-by otherwise).
                ``None`` (the default) keeps the single-keeper behaviour: at
                most one best row per unit, identical to passing only
                ``collapse_key``. An integer ``>= 1`` admits up to that many of
                a unit's highest-ranked rows and lets the remaining slots
                backfill deeper *distinct* units from the widened pool — so a
                long fragmented unit can keep more than its single top row while
                still surfacing more distinct units. This only relaxes the
                intra-unit retention count; it never adds a row an arm did not
                produce, never mutates a score, and never merges or drops a
                *distinct* unit as a side effect (ordinary ``top_k`` truncation
                still applies unchanged). Key-less rows are unaffected (each is
                already its own unit). Validated at call time; a value ``< 1``
                is rejected.
            include_archived: When ``False`` (the default) archived thoughts
                (``lifecycle_status = 'ARCHIVED'`` — forgotten by the hygiene
                loop or TTL-archived) are excluded from **every** candidate path:
                the FTS arm, the vector arm (``vec0`` post-filter and numpy
                fallback), the query-less fallback, and the ``CONSOLIDATED_FROM``
                graph expansion (so an archived source OBSERVATION cannot leak
                back in via an ACTIVE seed REFLECTION). When ``True`` archived
                rows are re-admitted across all of those paths for this call (the
                "search my archive" / "recall something I forgot" escape hatch),
                without restoring them — use :meth:`restore_thought` to make a
                thought eligible again permanently. The independent
                retired-REFLECTION freshness floor is unaffected either way: a
                retired REFLECTION stays excluded even under
                ``include_archived=True``.

        Returns:
            ``HybridSearchResult`` with ranked results and diagnostics. Tied
            scores are ordered by canonical ``thought_id`` ascending, giving
            a deterministic total order regardless of ``filters`` — so equal
            recency scores (identical timestamps, or future-dated rows both
            clamped to age ``0``) resolve deterministically.

        Raises:
            RecencyModeConflictError: If both an **explicit** ``current_cycle``
                and ``recency_now`` are supplied.
            InvalidRecencyArgumentError: If ``recency_now`` is not a valid
                ISO-8601 timestamp, or ``recency_now_half_life`` is not ``> 0``.
            ValueError: If a fusion weight is negative or ``recency_half_life``
                (the cognitive-cycle half-life) is not a positive integer.

        """
        import time as _time  # noqa: PLC0415

        from engrava.domain.models.search import HybridSearchResult  # noqa: PLC0415

        _t_start = _time.perf_counter()

        # --- Recency axis selection (cognitive cycle XOR transaction time) ---
        # Two separately-typed recency references; a query selects exactly one.
        # Precedence is explicit-wins, and a configured cycle_provider is PASSIVE:
        #   * explicit recency_now  -> transaction-time recency; the provider is
        #     NOT consulted and current_cycle stays None;
        #   * explicit current_cycle -> cognitive-cycle recency;
        #   * neither explicit reference -> a configured provider supplies the
        #     cycle (else None, recency off — unchanged).
        # Supplying BOTH explicit references is a conflicting request rejected
        # with a stable typed error — the axes measure age against incomparable
        # clocks and are never silently combined. The conflict is checked on the
        # RAW arguments (before provider resolution), so a store that configures a
        # provider can still opt into transaction recency by passing recency_now.
        if current_cycle is not None and recency_now is not None:
            conflict = (
                "pass current_cycle for cognitive-cycle recency or recency_now for "
                "transaction-time recency, never both"
            )
            raise RecencyModeConflictError(conflict)
        transaction_now: datetime.datetime | None = None
        resolved_recency_now_half_life = 0.0
        if recency_now is not None:
            # Transaction recency wins; the passive cycle_provider is NOT consulted
            # (current_cycle stays None). Parse + UTC-normalise the caller's "now"
            # at the boundary (naive => UTC; host tz never read); a malformed value
            # is a bad API argument — the store never invents a clock for this axis.
            transaction_now = _parse_recency_now(recency_now)
            resolved_recency_now_half_life = float(
                recency_now_half_life
                if recency_now_half_life is not None
                else (
                    self._search_config.recency_now_half_life_seconds
                    if self._search_config is not None
                    else _DEFAULT_RECENCY_NOW_HALF_LIFE_SECONDS
                )
            )
            if resolved_recency_now_half_life <= 0.0:
                msg = "recency_now_half_life must be a positive number of seconds"
                raise InvalidRecencyArgumentError(msg)
        else:
            # No transaction reference: resolve the cognitive cycle once (an
            # explicit current_cycle wins even at 0; else a configured provider;
            # else None). Reassigning the local here means every downstream
            # consumer (_resolve_hybrid_state, _fallback_hybrid_results,
            # _load_recency_scores) sees the resolved value.
            current_cycle = self._resolve_current_cycle(current_cycle)

        # Compile the effective metadata predicate once per column alias:
        # the arms join ``thought t`` (t.metadata_json); the expansion stage
        # queries ``thought`` unaliased (metadata_json). ``None`` when neither
        # argument constrains anything, so the unfiltered query path is
        # unchanged (apart from the always-on deterministic tie-break).
        filter_clause_t = compile_effective_predicate(filters, visibility, column="t.metadata_json")
        filter_clause_plain = compile_effective_predicate(
            filters, visibility, column="metadata_json"
        )

        # Validate the intra-unit retention depth at argument time (never
        # mid-query), mirroring the collapse-key path-validation contract. A
        # value below 1 has no meaning as a retention count, so reject it with a
        # typed error. ``None`` keeps the single-keeper behaviour. The param is
        # inert unless ``collapse_key`` is also set (no unit to retain-by), so
        # it does not perturb the ``collapse_key=None`` byte-identical path.
        if collapse_max_per_unit is not None and collapse_max_per_unit < 1:
            from engrava.domain.exceptions import InvalidFilterError  # noqa: PLC0415

            msg = f"collapse_max_per_unit must be >= 1, got {collapse_max_per_unit}"
            raise InvalidFilterError(msg)

        # Validate the de-fragmentation unit key (if any) at argument time —
        # never mid-query (reuses the shared metadata path grammar). ``None``
        # keeps the entire candidate/score/order path byte-identical to today's.
        collapse_paths: tuple[str, ...] | None = None
        collapse_pool_factor = (
            self._search_config.collapse_pool_factor if self._search_config is not None else 4
        )
        if collapse_key is not None:
            collapse_paths = _normalize_collapse_key(collapse_key)
            # Bounded candidate-pool widening: when collapsing, fragments of
            # few units can dominate the per-arm budgets, so widen each arm by
            # a small, config-backed factor to give backfill a deeper distinct
            # -unit pool. Bounded (small int) — never unbounded over-fetch.
            fts_top_k = fts_top_k * collapse_pool_factor
            vector_top_k = vector_top_k * collapse_pool_factor

        # Resolve the REFLECTION cap/boost once, up front, so the query-less
        # fallback below can size its own row window with the same backfill
        # headroom the FTS/vector arms already get for free from their much
        # larger ``fts_top_k`` / ``vector_top_k`` defaults.
        resolved_reflection_boost = (
            reflection_boost
            if reflection_boost is not None
            else (self._search_config.reflection_boost if self._search_config is not None else 1.0)
        )
        resolved_reflection_topk_cap = (
            self._search_config.reflection_topk_cap if self._search_config is not None else 0.3
        )
        needs_reflection_ids = (
            not include_reflections
            or resolved_reflection_boost != 1.0
            or resolved_reflection_topk_cap < 1.0
        )

        # The query-less fallback has no arm to over-fetch from, so give its
        # own row window the same bounded, config-backed headroom that
        # collapse-by-unit backfill and reflection_topk_cap eviction-backfill
        # both need — both draw replacement candidates from beyond ``top_k``.
        # Only widen when finalization can actually use the extra depth: with
        # ``collapse_key=None`` and the cap disabled (``>= 1.0``) the fallback
        # still fetches exactly ``top_k`` rows, byte-identical to before this
        # widening existed.
        fallback_fetch_top_k = top_k
        if collapse_paths is not None or resolved_reflection_topk_cap < 1.0:
            fallback_fetch_top_k = top_k * collapse_pool_factor

        backends_used: set[str] = set()
        (
            resolved_fts_weight,
            resolved_vector_weight,
            resolved_recency_weight,
            resolved_recency_half_life,
            resolved_priority_weight,
            resolved_graph_weight,
        ) = self._resolve_hybrid_defaults(
            fts_weight=fts_weight,
            vector_weight=vector_weight,
            recency_weight=recency_weight,
            recency_half_life=recency_half_life,
            priority_weight=priority_weight,
            graph_weight=graph_weight,
        )

        resolved_graph_edge_decay = (
            graph_edge_decay
            if graph_edge_decay is not None
            else (self._search_config.graph_edge_decay if self._search_config is not None else 0.5)
        )
        resolved_max_neighbors = (
            self._search_config.max_neighbors_per_candidate
            if self._search_config is not None
            else 5
        )

        # --- Determine active signals and redistribute weights ---
        fts_active, effective_vector, recency_active = await self._resolve_hybrid_state(
            query_text=query_text,
            query_vector=query_vector,
            current_cycle=current_cycle,
            transaction_now=transaction_now,
            recency_weight=resolved_recency_weight,
        )
        vector_active = effective_vector is not None
        priority_active = resolved_priority_weight > 0.0
        graph_active = resolved_graph_weight > 0.0

        (
            eff_fts_w,
            eff_vec_w,
            eff_rec_w,
            eff_pri_w,
            eff_gra_w,
        ) = self._redistribute_hybrid_weights(
            fts_active=fts_active,
            vector_active=vector_active,
            recency_active=recency_active,
            priority_active=priority_active,
            graph_active=graph_active,
            fts_weight=resolved_fts_weight,
            vector_weight=resolved_vector_weight,
            recency_weight=resolved_recency_weight,
            priority_weight=resolved_priority_weight,
            graph_weight=resolved_graph_weight,
        )

        if not fts_active and not vector_active:
            if recency_active:
                backends_used.add("recency")
            # Gate BOTH recency references on ``recency_active`` so a weight-0
            # reference stays inert on this query-less path too, on either
            # axis: passing ``current_cycle=None`` / ``transaction_now=None``
            # when recency is inactive falls the fallback through to its
            # neutral (flat-score, updated_cycle-ordered) branch, byte-identical
            # to a query with no recency reference.
            fallback = await self._fallback_hybrid_results(
                top_k=fallback_fetch_top_k,
                current_cycle=current_cycle if recency_active else None,
                recency_half_life=resolved_recency_half_life,
                transaction_now=transaction_now if recency_active else None,
                transaction_half_life_seconds=resolved_recency_now_half_life,
                filter_clause=filter_clause_plain,
                include_archived=include_archived,
            )
            if priority_active and fallback:
                backends_used.add("priority")
                priority_scores = await self._load_priority_scores(
                    thought_ids={tid for tid, _ in fallback},
                )
                fallback = [
                    (tid, score + priority_scores.get(tid, 0.0) * eff_pri_w)
                    for tid, score in fallback
                ]
                fallback.sort(key=lambda x: x[1], reverse=True)

            # Route the fallback's raw (tid, score) pairs through the same
            # REFLECTION filter/boost, collapse-by-unit, and
            # reflection_topk_cap finalization the FTS/vector-active path
            # applies below — a fallback result is still a result, and this
            # path enforces the same per-call limits.
            fallback_combined = dict(fallback)
            reflection_ids = await self._apply_reflection_filter_boost(
                combined=fallback_combined,
                include_reflections=include_reflections,
                resolved_reflection_boost=resolved_reflection_boost,
                needs_reflection_ids=needs_reflection_ids,
            )
            # Preserve the fallback's own deterministic order (recency- and
            # priority-scored, DB pre-order on ties) instead of re-deriving
            # one from dict/scan order — only the filter/boost step above
            # needed dict semantics. A stable sort by score alone keeps that
            # relative order for every tie the boost left untouched, and only
            # re-positions ids whose score the boost actually changed.
            ranked = sorted(
                ((tid, fallback_combined[tid]) for tid, _ in fallback if tid in fallback_combined),
                key=lambda item: -item[1],
            )
            final, reflections_evicted = await self._apply_collapse_and_topk_cap(
                ranked=ranked,
                collapse_paths=collapse_paths,
                collapse_max_per_unit=collapse_max_per_unit,
                reflection_ids=reflection_ids,
                resolved_reflection_topk_cap=resolved_reflection_topk_cap,
                include_reflections=include_reflections,
                top_k=top_k,
            )

            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)

            self._buffer_accesses([tid for tid, _ in final])
            return HybridSearchResult(
                results=final,
                backends_used=frozenset(backends_used),
                reflections_evicted=reflections_evicted,
            )

        token = _SUPPRESS_SEARCH_METRICS.set(True)
        try:
            # --- Gather FTS results ---
            if fts_active:
                backends_used.add("fts5")
                fts_results = await self.search_fts(
                    query_text,
                    top_k=fts_top_k,
                    include_archived=include_archived,
                    _filter_clause=filter_clause_t,
                )
            else:
                fts_results = []

            # --- Gather vector results ---
            vec_results: list[tuple[str, float]] = []
            if effective_vector is not None:
                vec_results = await self.search_similar(
                    effective_vector,
                    top_k=vector_top_k,
                    include_archived=include_archived,
                    _filter_clause=filter_clause_t,
                )
                backends_used.add("vector")
        finally:
            _SUPPRESS_SEARCH_METRICS.reset(token)

        # --- Fuse scores ---
        # Intentional, accepted arm-scale asymmetry: the FTS arm is min-max
        # normalized to [0, 1] per query, while the vector arm is blended at its
        # raw cosine scale (naturally [0, 1] with a non-negative threshold). The
        # arms are deliberately NOT put on a common scale here: min-maxing the
        # vector arm too would be a per-query re-scale that destroys the
        # cross-query comparability of cosine (a strong 0.92 and a weak 0.40
        # would both stretch to span [0, 1] within their own result set), and a
        # global arm-weight recalibration was measured to move end-to-end
        # accuracy by noise (and to regress multi-session). Symmetric re-scaling
        # is therefore a deferred, separately-gated change, not done here.
        fts_normalized = _normalize_min_max(fts_results)

        # Semantic-only base scores for graph signal (max(fts, vector))
        semantic_base: dict[str, float] = {}
        for tid, score in fts_normalized:
            semantic_base[tid] = max(semantic_base.get(tid, 0.0), score)
        for tid, score in vec_results:
            semantic_base[tid] = max(semantic_base.get(tid, 0.0), score)

        combined: dict[str, float] = {}
        for tid, score in fts_normalized:
            combined[tid] = combined.get(tid, 0.0) + score * eff_fts_w
        for tid, score in vec_results:
            combined[tid] = combined.get(tid, 0.0) + score * eff_vec_w

        # --- Recency signal (the one active axis: transaction time or cycle) ---
        if recency_active:
            backends_used.add("recency")
            if transaction_now is not None:
                recency_scores = await self._load_transaction_recency_scores(
                    thought_ids=set(combined.keys()),
                    now=transaction_now,
                    half_life_seconds=resolved_recency_now_half_life,
                )
            else:
                recency_scores = await self._load_recency_scores(
                    thought_ids=set(combined.keys()),
                    current_cycle=current_cycle if current_cycle is not None else 0,
                    recency_half_life=resolved_recency_half_life,
                )
            for thought_id, recency_score in recency_scores.items():
                combined[thought_id] = combined.get(thought_id, 0.0) + recency_score * eff_rec_w

        # --- Priority signal ---
        if priority_active and combined:
            backends_used.add("priority")
            priority_scores = await self._load_priority_scores(
                thought_ids=set(combined.keys()),
            )
            for thought_id, priority_score in priority_scores.items():
                combined[thought_id] = combined.get(thought_id, 0.0) + priority_score * eff_pri_w

        # --- Graph signal (1-hop-weighted, semantic base only) ---
        if graph_active and combined:
            graph_boosts = await self._load_graph_signal(
                candidate_scores=semantic_base,
                graph_edge_decay=resolved_graph_edge_decay,
                max_neighbors=resolved_max_neighbors,
            )
            if graph_boosts:
                backends_used.add("graph")
                for thought_id, graph_boost in graph_boosts.items():
                    combined[thought_id] = combined.get(thought_id, 0.0) + graph_boost * eff_gra_w

        # --- Candidate expansion via CONSOLIDATED_FROM ---
        expansion_cfg = self._search_config
        expansion_enabled = (
            expansion_cfg.graph_expansion_enabled if expansion_cfg is not None else True
        )
        if expansion_enabled and combined:
            _added = await self._expand_via_consolidated_from(
                combined=combined,
                expansion_top_n=(
                    expansion_cfg.graph_expansion_top_n if expansion_cfg is not None else 5
                ),
                propagation_factor=(
                    expansion_cfg.graph_expansion_propagation_factor
                    if expansion_cfg is not None
                    else 0.7
                ),
                max_sources_per_reflection=(
                    expansion_cfg.graph_expansion_max_sources_per_reflection
                    if expansion_cfg is not None
                    else 20
                ),
                reflection_source_ceiling=(
                    expansion_cfg.graph_expansion_reflection_source_ceiling
                    if expansion_cfg is not None
                    else 50
                ),
                expansion_sources=None,
                include_archived=include_archived,
                _filter_clause=filter_clause_plain,
            )
            if _added > 0:
                backends_used.add("graph_expansion")

        # --- REFLECTION filter + boost + collapse-by-unit + top-K cap ---
        # Shared with the query-less fallback path above, so both paths
        # enforce the same per-call limits regardless of which arms produced
        # ``combined``.
        reflection_ids = await self._apply_reflection_filter_boost(
            combined=combined,
            include_reflections=include_reflections,
            resolved_reflection_boost=resolved_reflection_boost,
            needs_reflection_ids=needs_reflection_ids,
        )
        # Deterministic total order: score descending, canonical thought_id
        # ascending — invariant to dict/scan order.
        ranked = _sort_scored_descending(list(combined.items()))
        final, reflections_evicted = await self._apply_collapse_and_topk_cap(
            ranked=ranked,
            collapse_paths=collapse_paths,
            collapse_max_per_unit=collapse_max_per_unit,
            reflection_ids=reflection_ids,
            resolved_reflection_topk_cap=resolved_reflection_topk_cap,
            include_reflections=include_reflections,
            top_k=top_k,
        )

        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)

        self._buffer_accesses([tid for tid, _ in final])
        return HybridSearchResult(
            results=final,
            backends_used=frozenset(backends_used),
            reflections_evicted=reflections_evicted,
        )

    async def _batch_fetch_embedding_blobs(
        self, thought_ids: list[str]
    ) -> dict[str, tuple[int, bytes]]:
        """Fetch ``(dimension, vector_blob)`` for many thoughts in one pass.

        Replaces a per-id ``get_embedding`` loop with a single ``... IN (…)``
        query per chunk. SQLite caps host parameters at ``_SQLITE_MAX_VARS`` per
        statement, so the id list is chunked to stay within that limit for large
        inputs. Ids with no embedding row are simply absent from the result —
        the caller maps them to a zero score exactly as the per-row ``None``
        branch did.

        The default ``embedding_id`` is a deterministic function of the owner
        (``uuid5(thought_id)``), so a thought has at most one embedding row and
        the mapping is exact. To stay faithful even to the pathological case of
        a caller writing several rows for one owner under explicit distinct
        ``embedding_id`` values, rows are ordered by ``rowid`` and the first per
        owner is kept — the same lowest-``rowid`` row a bare ``get_embedding``
        ``fetchone()`` returns.

        Args:
            thought_ids: Thought ids whose embeddings to fetch.

        Returns:
            Mapping of ``thought_id`` to ``(dimension, vector_blob)`` for every
            id that has a ``THOUGHT`` embedding row.

        """
        embeddings_by_id: dict[str, tuple[int, bytes]] = {}
        for chunk_start in range(0, len(thought_ids), _SQLITE_MAX_VARS):
            id_chunk = thought_ids[chunk_start : chunk_start + _SQLITE_MAX_VARS]
            placeholders = ", ".join("?" for _ in id_chunk)
            cursor = await self._db.execute(
                f"SELECT owner_id, dimension, vector_blob FROM embedding"  # noqa: S608
                f" WHERE owner_type = 'THOUGHT' AND owner_id IN ({placeholders})"
                f" ORDER BY rowid",
                id_chunk,
            )
            for row in await cursor.fetchall():
                # ``setdefault`` keeps the first (lowest-rowid) row per owner,
                # matching ``get_embedding``'s ``fetchone()`` under duplicates.
                embeddings_by_id.setdefault(
                    str(row["owner_id"]),
                    (int(row["dimension"]), row["vector_blob"]),
                )
        return embeddings_by_id

    async def search_reflections_only(
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        *,
        top_k: int = 10,
        current_cycle: int | None = None,
    ) -> HybridSearchResult:
        """Return only REFLECTION thoughts ranked by cosine similarity.

        Directly fetches all ``ThoughtType.REFLECTION`` thoughts and
        scores them against the query vector — guarantees completeness
        regardless of how many regular thoughts are in the store (no
        pagination gap like an over-fetch approach would have).

        When ``current_cycle`` is provided, a recency blend is applied
        alongside cosine similarity using the configured
        ``default_recency_weight``.

        When no query vector is available and no embedding provider is
        configured, all eligible REFLECTION thoughts are returned unranked.

        REFLECTIONs whose ``expires_at`` is at or before the single UTC instant
        captured for this call are excluded, matching the general ranked
        retrieval paths.

        Args:
            query_text: Text used for auto-embedding when no
                ``query_vector`` is supplied and a provider is configured.
            query_vector: Embedding vector for cosine similarity ranking.
            top_k: Maximum number of results to return.
            current_cycle: Current cycle for optional recency blending.

        Returns:
            ``HybridSearchResult`` containing only REFLECTION thoughts,
            sorted by cosine similarity (and optionally recency) descending.

        """
        import math  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        from engrava.domain.models.search import HybridSearchResult  # noqa: PLC0415

        _t_start = _time.perf_counter()
        # Pin the expiry boundary before the optional provider await so a slow
        # embedding call cannot change eligibility during this search.
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()

        # Resolve effective query vector (auto-embed if provider available)
        effective_vector = query_vector
        if effective_vector is None and self._embedding_provider is not None and query_text.strip():
            await self._ensure_query_prefix_pairs()
            effective_vector = await _embed_query(self._embedding_provider, query_text)

        # Fetch all eligible REFLECTION thought IDs directly — complete, no
        # pagination gap. Capture the wall-clock boundary once so every row is
        # evaluated against the same instant. Retired REFLECTIONs and expired
        # rows are excluded by the same freshness floors the general ranked
        # paths apply.
        cursor = await self._db.execute(
            "SELECT thought_id FROM thought "
            "WHERE thought_type = 'REFLECTION' AND lifecycle_status = 'ACTIVE' "
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY thought_id ASC",
            (now_iso,),
        )
        rows = await cursor.fetchall()
        reflection_ids = [str(r["thought_id"]) for r in rows]

        if not reflection_ids:
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return HybridSearchResult(results=[], backends_used=frozenset())

        if effective_vector is None:
            # No scoring available — return unranked, capped at top_k
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            self._buffer_accesses(reflection_ids[:top_k])
            return HybridSearchResult(
                results=[(rid, 0.0) for rid in reflection_ids[:top_k]],
                backends_used=frozenset(),
            )

        # Score each REFLECTION by cosine similarity to the query vector
        q_norm = math.sqrt(sum(x * x for x in effective_vector))
        if q_norm == 0.0:
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            self._buffer_accesses(reflection_ids[:top_k])
            return HybridSearchResult(
                results=[(rid, 0.0) for rid in reflection_ids[:top_k]],
                backends_used=frozenset({"vector"}),
            )

        backends_used_set: set[str] = {"vector"}
        # Batch-fetch every REFLECTION embedding in one pass instead of a
        # per-id ``get_embedding`` round trip (was O(N) queries). The result is
        # identical: an id with no embedding row is scored 0.0 exactly as the
        # per-row ``emb is None`` branch did.
        embeddings_by_id = await self._batch_fetch_embedding_blobs(reflection_ids)

        scores = self._cosine_score_reflections(
            reflection_ids, effective_vector, q_norm, embeddings_by_id
        )

        # Optional recency blend when current_cycle is provided
        if current_cycle is not None:
            backends_used_set.add("recency")
            search_config = self._search_config
            recency_weight = search_config.default_recency_weight if search_config else 0.1
            recency_half_life = search_config.recency_half_life if search_config else 50
            if recency_weight > 0.0:
                recency_scores = await self._load_recency_scores(
                    thought_ids={rid for rid, _ in scores},
                    current_cycle=current_cycle,
                    recency_half_life=recency_half_life,
                )
                vec_w = 1.0 - recency_weight
                scores = [
                    (rid, sim * vec_w + recency_scores.get(rid, 0.0) * recency_weight)
                    for rid, sim in scores
                ]

        scores = _sort_scored_descending(scores)
        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
        final_scores = scores[:top_k]
        self._buffer_accesses([rid for rid, _ in final_scores])
        return HybridSearchResult(
            results=final_scores,
            backends_used=frozenset(backends_used_set),
        )

    @staticmethod
    def _cosine_score_reflections(
        reflection_ids: list[str],
        query_vector: list[float],
        query_norm: float,
        embeddings_by_id: dict[str, tuple[int, bytes]],
    ) -> list[tuple[str, float]]:
        """Score each reflection id by cosine similarity to the query vector.

        A reflection with no stored embedding, or a zero-norm embedding, scores
        ``0.0`` — matching the per-row behaviour of the general ranked
        retrieval paths.

        Args:
            reflection_ids: REFLECTION thought ids to score, in output order.
            query_vector: The effective (auto-embedded or supplied) query vector.
            query_norm: Precomputed L2 norm of ``query_vector`` (non-zero).
            embeddings_by_id: Batch-fetched ``(dimension, blob)`` per id.

        Returns:
            ``(reflection_id, cosine_score)`` pairs in ``reflection_ids`` order.

        """
        scores: list[tuple[str, float]] = []
        for rid in reflection_ids:
            emb = embeddings_by_id.get(rid)
            if emb is None:
                scores.append((rid, 0.0))
                continue
            dimension, blob = emb
            vec = list(struct.unpack(f"{dimension}f", blob))
            v_norm = math.sqrt(sum(x * x for x in vec))
            if v_norm == 0.0:
                scores.append((rid, 0.0))
                continue
            dot = sum(a * b for a, b in zip(query_vector, vec, strict=False))
            scores.append((rid, dot / (query_norm * v_norm)))
        return scores

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    async def record_access(self, thought_id: str) -> None:
        """Record an explicit access to a thought.

        Increments ``access_count`` by 1 and sets ``last_accessed_at``
        to the current UTC time.

        Args:
            thought_id: UUID of the thought to mark as accessed.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.

        """
        async with self._write_lock:
            now = datetime.datetime.now(datetime.UTC).isoformat()
            cursor = await self._db.execute(
                "UPDATE thought SET access_count = access_count + 1, "
                "last_accessed_at = ? WHERE thought_id = ?",
                (now, thought_id),
            )
            if cursor.rowcount == 0:
                raise ThoughtNotFoundError(thought_id)
            await self._maybe_commit()

    def _buffer_accesses(self, thought_ids: list[str]) -> None:
        """Buffer access events for retrieved thoughts (no DB write).

        Called from the retrieval paths (search / recall / reflection search /
        explicit ``get_thought``) with the ids a caller actually retrieved.
        No-op unless access tracking is enabled. This never touches the
        database — events accumulate in the bounded in-process buffer and are
        applied in one batched ``UPDATE`` by :meth:`flush_access_buffer` at the
        consolidation-cycle boundary (or an explicit flush / store close). The
        existing per-id :meth:`record_access` is intentionally *not* called
        here: batching is what keeps the read path free of per-result writes.

        Args:
            thought_ids: Ids just returned to the caller. Duplicates and empty
                lists are handled by the buffer (coalesced / ignored).

        """
        if (
            not self._access_tracking_enabled
            or self._suppress_access_tracking.get()
            or not thought_ids
        ):
            return
        now = datetime.datetime.now(datetime.UTC).isoformat()
        for thought_id in thought_ids:
            self._access_buffer.record(thought_id, now=now)

    async def flush_access_buffer(self) -> int:
        """Apply buffered access events in a single batched ``UPDATE``.

        Drains the in-process access buffer and folds every pending
        ``(count_delta, last_seen)`` into the ``thought`` table with one
        ``executemany`` — the batched write the read path deferred. Ids whose
        thought no longer exists are silently skipped (the row may have been
        deleted since it was buffered; access counts are best-effort).

        Access counts are high-volume regenerable telemetry, so these updates
        are **not** written to the hash-chain journal — a deliberate exception
        to the journal-every-mutation rule. A crash before a flush undercounts,
        which self-heals as access continues.

        Called automatically at the start of a dreaming consolidation cycle
        (see :meth:`consolidate`) and on :meth:`close`; also safe to call
        explicitly. A no-op returning ``0`` when tracking is disabled or the
        buffer is empty.

        Returns:
            The number of buffered access **entries flushed** — the distinct
            thought ids drained from the buffer. This counts entries submitted
            to the batched ``UPDATE``, which is not necessarily the number of
            rows actually updated: an id whose thought was deleted since it was
            buffered matches no row, so it is flushed but updates nothing (the
            counts are best-effort telemetry, so this is not reconciled). If
            *none* of the batch matched a row, this call wrote nothing at all,
            and — like :meth:`delete_thought` — does not commit a caller's own
            open transaction on the strength of that empty batch.

        """
        if not self._access_tracking_enabled:
            return 0
        pending = self._access_buffer.drain()
        if not pending:
            return 0
        # (delta, last_seen, thought_id) — matches the UPDATE parameter order.
        params = [(delta, ts, tid) for tid, delta, ts in pending]
        async with self._write_lock:
            # Sampled before anything below touches the connection — same
            # ownership test as delete_thought.
            opened_transaction = not self._db.in_transaction
            cursor = await self._db.executemany(
                "UPDATE thought SET access_count = access_count + ?, "
                "last_accessed_at = ? WHERE thought_id = ?",
                params,
            )
            # sqlite3 (and aiosqlite atop it) sums per-statement row counts
            # into a single ``executemany`` rowcount, so this is exact, not an
            # approximation: 0 here means not one entry in the batch matched a
            # still-existing thought, i.e. this call wrote nothing.
            if cursor.rowcount > 0:
                await self._maybe_commit()
            elif opened_transaction and self._db.in_transaction:
                # Every id in the batch was stale — nothing was written, so
                # there is nothing of this call's own to commit. Close only a
                # transaction this call itself opened; a caller's own open
                # transaction is left untouched.
                await self._db.rollback()
        logger.debug(
            "flushed access buffer: %d entries drained in one batch "
            "(not necessarily the number of rows updated — a deleted "
            "thought's entry still counts here)",
            len(params),
        )
        return len(params)

    # ------------------------------------------------------------------
    # Memory Hygiene — deterministic forgetting loop
    # ------------------------------------------------------------------

    async def run_hygiene(
        self,
        *,
        current_cycle: int | None = None,
        now: datetime.datetime | None = None,
    ) -> HygieneResult:
        """Run one Memory Hygiene pass — archive cold/low-value thoughts.

        A standalone, deterministic, no-LLM forgetting pass: it scores every
        eligible thought with a keep-score (the dreaming signal library under
        the hygiene weight vector, with active-signal redistribution), multiplies
        by the ``decay_function`` hook, and **archives** — reversibly — the
        thoughts whose eviction-score falls below ``eviction_threshold`` and that
        are not protected. When ``auto_gc_enabled`` it then physically
        garbage-collects previously hygiene-archived thoughts once **both** the
        cycle restore window and the wall-clock restore window have elapsed.

        This is the store's primary forgetting entry point; it runs immediately
        and **bypasses** ``check_every_n_cycles`` (that cadence gates only the
        convenience invocation from :meth:`consolidate`). One run performs at most
        one archive stage and at most one GC stage, each independently bounded by
        ``max_evictions_per_run``.

        Safety by construction (mirrors the config defaults):

        * **Archive-not-delete.** The default action flips ``lifecycle_status``
          to ``ARCHIVED`` (reversible, no data loss) via the same mechanism TTL
          archival uses, and stamps ``archived_at_cycle = current_cycle``.
        * **Protection.** A thought is never archived or GC'd when it is
          ``pinned`` or its priority is in ``protected_priorities`` (default
          ``P1``). ``confidence`` is *not* protection.
        * **All-flat fallback.** When no keep-signal is active (e.g. a brand-new
          store with no access history / confirmations / cycle span), the
          keep-score is uninformative, so the pass archives **nothing**.
        * **Cold-start guards.** Two run-safe additions that only ever *add*
          protection: a per-thought **minimum-inactivity-age gate**
          (``min_inactivity_age_seconds`` — a thought must be untouched for at
          least that many wall-clock seconds before it is archivable) and a
          run-level **access-gate** (nothing is archived unless a usage-history
          signal — ``frequency`` / ``confirmation`` / ``action_outcome`` — is
          active across the pool). Together they stop a fresh or bulk-imported
          store, where cycle-recency degenerates into ingest order, from
          archiving its earliest-ingested rows.
        * **Decay clamp.** The ``decay_function`` return is clamped to
          ``[0.0, 1.0]`` and a non-finite value is treated as ``1.0`` (no decay)
          — decay can only lower a score toward archive, never resurrect one, and
          a misbehaving custom hook can never cause a spurious eviction.
        * **Deterministic capped selection.** Same store + config + cycle ⇒ the
          identical eviction set; archive orders by ``eviction_score ASC,
          updated_cycle ASC, thought_id ASC`` and GC by ``archived_at_cycle ASC,
          thought_id ASC`` before the per-stage cap.
        * **GC keys off hygiene's own bookkeeping.** Only thoughts with a
          non-NULL ``archived_at_cycle`` (i.e. archived *by hygiene*) are ever
          auto-GC'd — a TTL/manually-archived thought (``archived_at_cycle`` is
          ``None``) is left alone.
        * **Two restore windows, both required.** GC reaps a hygiene-archived
          thought only once it is old enough *cognitively* (the cycle window,
          ``gc_min_archive_age_cycles``) **and** in *real time* (the wall-clock
          window, ``gc_restore_window_seconds``, measured against ``now``), so a
          fast-cycling store cannot permanently delete a just-archived thought
          before a real-time chance to restore it. A hygiene-archived row that
          predates the ``archived_at`` column (``archived_at`` is ``None``) has
          no real-time stamp and is never auto-GC'd while the wall-clock window
          is active — the irreversible stage fails closed. Setting
          ``gc_restore_window_seconds = 0`` disables the wall-clock window
          (cycle-only, backward-compatible).
        * **Dry run.** When ``dry_run`` is set nothing is mutated and nothing is
          journaled; the would-evict set is returned for preview.

        This is cognitive hygiene, not compliance deletion: GC is best-effort,
        window-gated, and opt-in — it offers no deletion guarantee, legal hold,
        or erasure receipt. **GC is not erasure:** a GC'd thought's content
        survives in the append-only journal (the ``DELETE_THOUGHT`` entry keeps a
        full ``before`` snapshot); GC reclaims the live/queryable working set, it
        does not purge history.

        Args:
            current_cycle: The current cognitive cycle number, driving the
                cycle-based recency / staleness keep-signals and the GC restore
                window. Optional: when omitted (``None``), it is pulled from a
                configured ``cycle_provider`` (an explicit value — including
                ``0`` — always wins). A disabled / absent policy is a no-op that
                needs no cycle, so the value is only required once a real pass is
                about to run.
            now: The wall-clock instant the minimum-inactivity-age gate (archive
                stage) and the wall-clock restore window (GC stage) both measure
                against. Computed **once per run** and threaded into selection,
                the archive re-check (and ``archived_at`` stamp), and the GC
                eligibility cutoff so a run is internally consistent. Optional:
                defaults to ``datetime.now(UTC)``; inject a fixed timezone-aware
                instant to pin both boundaries deterministically in tests /
                benchmarks.

        Returns:
            A :class:`~engrava.infrastructure.sqlite.hygiene.HygieneResult` with
            the archived / GC'd counts, the number of candidates evaluated, the
            ``dry_run`` flag, the would-evict preview (under ``dry_run``), and the
            signals that were flat this run.

        Raises:
            RuntimeError: When no hygiene policy is configured on this store
                (built without ``hygiene_policy`` / ``hygiene_policy`` is
                ``None``) — there is nothing to run.
            ValueError: When a pass is due but no cycle is available — neither an
                explicit ``current_cycle`` nor a configured ``cycle_provider``.
            CycleProviderError: When a configured provider returns an invalid
                value (not an ``int``, a ``bool``, or negative).
            ConnectionQuarantinedError: When the connection has been
                quarantined (already, or by this call's own failed-unwind
                recovery inside its archive+GC unit).

        """
        policy = self._hygiene_policy
        if policy is None:
            msg = (
                "run_hygiene() requires a hygiene policy: build the store via "
                "from_config with a hygiene_policy section (or pass hygiene_policy=...)."
            )
            raise RuntimeError(msg)

        if not policy.enabled:
            # ``enabled`` is a hard master switch: a disabled policy never forgets,
            # even on an explicit ``run_hygiene()`` call — the fail-safe direction
            # for a data-deleting loop. To preview or run, set ``enabled=True``
            # (and ``dry_run=True`` for a non-mutating preview).
            return HygieneResult()

        # A real pass is about to run, so a cycle is now required. Resolved after
        # the no-op guards above (a disabled/absent policy needs no cycle) and
        # never invented — ``0`` would make every record look equally fresh.
        current_cycle = self._require_current_cycle(current_cycle, operation="run_hygiene()")

        # The minimum-inactivity-age gate is measured against a single wall-clock
        # instant for the whole run (never ``datetime.now`` per thought) so the
        # archive set is internally consistent and, when ``now`` is injected,
        # deterministic. An injected ``now`` is normalised to UTC — a naive value
        # is treated as UTC (the domain's naive-as-UTC convention) and an aware
        # non-UTC value is converted — so both the Python age subtraction and the
        # write-time SQL cutoff (a lexicographic compare against UTC-normalised
        # timestamps) stay correct.
        now = datetime.datetime.now(datetime.UTC) if now is None else _ensure_utc(now)

        candidates = await self._hygiene_candidates()
        ctx = DreamingContext(current_cycle=current_cycle, total_thoughts=len(candidates))
        active_weights, flat_signals = compute_active_hygiene_weights(
            policy.signal_weights,
            candidates,
            current_cycle=current_cycle,
            access_tracking_enabled=self._access_tracking_enabled,
        )
        has_active_signal = any(weight > 0.0 for weight in active_weights.values())
        # Access-gate (cold-start guard): without any usage-history signal in the
        # pool, "cold" is indistinguishable from "ingested early", so recency of
        # cycle must not drive eviction alone — archive nothing this run.
        has_usage_signal = has_active_usage_signal(
            candidates,
            current_cycle=current_cycle,
            access_tracking_enabled=self._access_tracking_enabled,
        )

        # All-flat fail-safe: an uninformative keep-score must never drive
        # eviction, so archive nothing (but a GC stage may still reap already
        # hygiene-archived thoughts whose restore window has elapsed).
        would_evict: list[EvictionReason] = []
        if has_active_signal and has_usage_signal:
            would_evict = self._select_archive_candidates(
                candidates,
                ctx=ctx,
                active_weights=active_weights,
                policy=policy,
                now=now,
                decay_multipliers=await self._hygiene_decay_multipliers(
                    candidates,
                    current_cycle=current_cycle,
                ),
            )

        if policy.dry_run:
            return HygieneResult(
                archived_count=0,
                gc_count=0,
                candidates_evaluated=len(candidates),
                dry_run=True,
                would_evict=would_evict,
                flat_signals=flat_signals,
            )

        # Archive + GC share one failure-atomic unit, not just one critical
        # section: both stages' guarded writes run under `_write_lock` (so a
        # different task's guarded write cannot land between the archive
        # stage's last write and the GC stage's first), and both run inside
        # one `_write_readback_savepoint` unit, so a failed or cancelled
        # journal append anywhere in either stage unwinds every write the
        # pass has made so far -- not just the write it failed on. A nested
        # public write inside that unit -- `retire_orphan_reflections` writes
        # through `update_thought`, whose own `_maybe_commit()` commits --
        # would otherwise release the unit's savepoint right along with it;
        # suppressing nested commits for the unit's duration (task-locally,
        # via `_SUPPRESS_NESTED_AUTO_COMMIT` -- see that ContextVar's
        # docstring for why not the instance-wide `_skip_auto_commit_depth`)
        # keeps the whole pass inside the one savepoint instead.
        async with self._write_lock:
            # Sampled before the unit below touches the connection, mirroring
            # delete_thought / cleanup_expired / delete_edge: whether this
            # call is the one that opened the transaction it may need to
            # close once the unit exits cleanly with nothing to commit.
            opened_transaction = not self._db.in_transaction

            gc_count = 0
            retired_count = 0
            gc_wrote_anything = False
            async with self._write_readback_savepoint("run_hygiene", begin="IMMEDIATE"):
                suppress_token = _SUPPRESS_NESTED_AUTO_COMMIT.set(True)
                try:
                    archived_count = await self._hygiene_archive(
                        would_evict, policy=policy, current_cycle=current_cycle, now=now
                    )

                    if policy.auto_gc_enabled:
                        gc_outcome = await self._hygiene_gc(
                            policy=policy, current_cycle=current_cycle, now=now
                        )
                        gc_count = gc_outcome.gc_count
                        retired_count = gc_outcome.retired_count
                        gc_wrote_anything = gc_outcome.wrote_anything
                finally:
                    # Reset before the exception (if any) leaves this block,
                    # so a nested write made by the *unwind* itself (there is
                    # none today, but the invariant is the unit's, not this
                    # call's) never sees a stale suppression left behind.
                    _SUPPRESS_NESTED_AUTO_COMMIT.reset(suppress_token)

            # Finalization is decided by what survived, reported by the
            # stages themselves -- never by `self._db.total_changes`, which
            # is monotonic and still counts a write a savepoint later undid.
            # Concretely: a GC candidate silently vetoed by a `BEFORE DELETE`
            # trigger that writes an audit row and then `RAISE(IGNORE)`s
            # advances `total_changes` for that audit insert even though
            # `_delete_thought_atomic` rolls it back and reports
            # `wrote_anything=False` -- a `total_changes`-keyed finalization
            # would misread that as a surviving write and commit a caller's
            # unrelated pending work along with it.
            wrote_anything = (
                archived_count > 0 or gc_count > 0 or retired_count > 0 or gc_wrote_anything
            )
            if wrote_anything:
                await self._maybe_commit()
            elif opened_transaction and self._db.in_transaction:
                # Nothing this pass made survived -- close only a transaction
                # this call itself opened, never one a caller already held:
                # see delete_thought for the same reasoning.
                await self._db.rollback()

        return HygieneResult(
            archived_count=archived_count,
            gc_count=gc_count,
            candidates_evaluated=len(candidates),
            dry_run=False,
            would_evict=[],
            flat_signals=flat_signals,
        )

    async def _hygiene_candidates(self) -> list[ThoughtRecord]:
        """Collect the eviction candidate pool: every ACTIVE and CREATED thought.

        Already-ARCHIVED and DONE thoughts are outside the candidate set (they
        are not re-processed). The **whole** eligible pool must be scored — the
        per-run cap bounds the *archived* set, not the *considered* set, so that
        the coldest thoughts (not an arbitrary page) are the ones selected.
        Both eligible lifecycles are walked one page at a time (mirroring the
        orphan-reflection sweep) so no candidate is missed regardless of store
        size. The frequency signal is not fed by these internal scans (access
        tracking is suppressed for hygiene's own reads by the caller when
        appropriate).

        Returns:
            The candidate thoughts (ACTIVE then CREATED), each pool fully
            enumerated.

        """
        candidates: list[ThoughtRecord] = []
        for lifecycle in (LifecycleStatus.ACTIVE, LifecycleStatus.CREATED):
            offset = 0
            while True:
                page = await self.list_thoughts(
                    lifecycle_status=lifecycle.value,
                    limit=_ORPHAN_SWEEP_PAGE_SIZE,
                    offset=offset,
                )
                candidates.extend(page)
                if len(page) < _ORPHAN_SWEEP_PAGE_SIZE:
                    break
                offset += _ORPHAN_SWEEP_PAGE_SIZE
        return candidates

    async def _hygiene_decay_multipliers(
        self,
        candidates: list[ThoughtRecord],
        *,
        current_cycle: int,
    ) -> dict[str, float]:
        """Resolve the clamped decay multiplier for each candidate.

        Unlike ``on_store``/``on_retrieve``, which have always had call-sites
        elsewhere, ``decay_function`` was dead code until this call-site was
        added: the hygiene eviction score is its **only** call-site (it is
        never wired into search / ranking / promotion). Its return is clamped
        to ``[0.0, 1.0]`` and a
        non-finite value (``NaN`` / ``±inf``) is treated as ``1.0`` — the
        fail-safe direction, since decay can then only lower a score toward
        archive, never resurrect one above threshold or over-evict.

        Args:
            candidates: The candidate pool.
            current_cycle: The current cycle (elapsed cycles are measured from
                each thought's ``updated_cycle``).

        Returns:
            A mapping of ``thought_id`` to its clamped decay multiplier.

        """
        multipliers: dict[str, float] = {}
        for thought in candidates:
            elapsed = max(0, current_cycle - thought.updated_cycle)
            raw = await self._hooks.decay_function(thought, elapsed)
            multipliers[thought.thought_id] = _clamp_decay(raw)
        return multipliers

    def _select_archive_candidates(
        self,
        candidates: list[ThoughtRecord],
        *,
        ctx: DreamingContext,
        active_weights: dict[str, float],
        policy: HygienePolicyConfig,
        now: datetime.datetime,
        decay_multipliers: dict[str, float],
    ) -> list[EvictionReason]:
        """Score candidates and pick the deterministic, capped archive set.

        For each unprotected candidate, computes ``keep_score`` (weighted average
        over the active signals) and ``eviction_score = keep_score * decay``, and
        keeps those strictly below ``eviction_threshold``. The survivors are
        ordered ``eviction_score ASC, updated_cycle ASC, thought_id ASC``
        (lowest-value, oldest, id tiebreak) and truncated to
        ``max_evictions_per_run`` — a stable set for a given store + config +
        cycle.

        Protected thoughts (``pinned`` or a priority in ``protected_priorities``)
        and thoughts inside the minimum-inactivity-age window
        (:func:`_hygiene_inactive_enough`) are excluded up front and never scored
        into the archive set.

        Args:
            candidates: The candidate pool.
            ctx: The scoring context.
            active_weights: The redistributed per-signal weights for this run.
            policy: The active hygiene policy.
            now: The run's wall-clock instant for the minimum-inactivity-age gate.
            decay_multipliers: Per-thought clamped decay multipliers.

        Returns:
            The ordered, capped list of :class:`EvictionReason` for the thoughts
            to archive.

        """
        scored: list[tuple[float, int, str, EvictionReason]] = []
        for thought in candidates:
            if _hygiene_protected(thought, policy):
                continue
            if not _hygiene_inactive_enough(thought, policy, now):
                # Minimum-inactivity-age gate: a thought contacted within the last
                # ``min_inactivity_age_seconds`` (or with no known last-contact
                # time) is protected, exactly like a pinned / protected-priority row.
                continue
            keep_score, per_signal = compute_keep_score(thought, ctx, active_weights)
            decay = decay_multipliers[thought.thought_id]
            eviction_score = keep_score * decay
            if eviction_score >= policy.eviction_threshold:
                continue
            reason = EvictionReason(
                thought_id=thought.thought_id,
                keep_score=keep_score,
                eviction_score=eviction_score,
                decay_multiplier=decay,
                threshold=policy.eviction_threshold,
                signals=per_signal,
            )
            scored.append((eviction_score, thought.updated_cycle, thought.thought_id, reason))

        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [reason for *_, reason in scored[: policy.max_evictions_per_run]]

    async def _hygiene_archive(
        self,
        to_archive: list[EvictionReason],
        *,
        policy: HygienePolicyConfig,
        current_cycle: int,
        now: datetime.datetime,
    ) -> int:
        """Archive the selected thoughts (Stage 1 — reversible, journaled).

        **Write-lock classification: under the lock via its caller.** Called
        only from ``run_hygiene``, which holds ``_write_lock`` around both
        this call and ``_hygiene_gc``'s.

        Flips each thought ``* -> ARCHIVED`` via the existing archive mechanism
        — a **direct lifecycle write**, exactly as TTL archival does in
        :meth:`cleanup_expired` (an ``UPDATE`` of ``lifecycle_status`` /
        ``expires_at``, not an ``evolve`` transition). Using the direct write is
        deliberate: it lets a ``CREATED`` thought be archived even though the
        lifecycle state machine only permits ``CREATED -> ACTIVE`` (hygiene's
        eligible candidate set is ACTIVE **and** CREATED), matching how TTL archival flips
        any expired row regardless of its current state. The write also stamps
        ``archived_at_cycle = current_cycle`` and the wall-clock
        ``archived_at = now`` (the two hygiene-archival markers, cleared together
        on restore) and clears ``expires_at`` so the thought is no longer subject
        to TTL. The hygiene loop is the only archival flow that *stamps*
        ``archived_at`` (and ``archived_at_cycle``): TTL archival
        (:meth:`cleanup_expired`) actively clears both back to ``NULL``, and a
        never-hygiene-archived row keeps them ``NULL``, so ``archived_at_cycle IS
        NOT NULL`` marks exactly a row whose *current* archival was performed by
        hygiene. Like every model field, both markers are still writable through a
        raw :meth:`update_thought`; that low-level path does not manage them, so
        prefer :meth:`restore_thought` / the hygiene and TTL flows.

        **The write also bumps ``revision``, but does not enforce it.** Hygiene
        must not be defeated by a caller holding a stale read — it archives
        unconditionally (subject only to the predicate guard above, which is
        about *protection*, not staleness) — but a caller's own token for this
        row must not survive an archival it did not cause, since the row's
        lifecycle just changed underneath it. Bumping (without checking)
        ``revision`` achieves both at once.

        The mutation is recorded as an ordinary ``UPDATE_THOUGHT`` journal entry
        — **no new mutation type** — with the forgetting rationale nested in the
        delta under ``eviction_reason`` so the decision is reconstructable and
        stays ``verify_journal``-covered. The journal ``after`` is reconstructed
        with the string lifecycle value (as TTL archival does) so its ``evolve``
        skips the state-machine check.

        Args:
            to_archive: The eviction reasons chosen by
                :meth:`_select_archive_candidates`, already ordered and capped.
            policy: The active hygiene policy — used for the write-time protection
                and minimum-inactivity-age re-checks (a thought pinned /
                re-prioritised / read after selection).
            current_cycle: The cycle stamped into ``archived_at_cycle``.
            now: The run's wall-clock instant — used both for the
                minimum-inactivity-age re-check (the same instant selection used)
                and as the value stamped into ``archived_at`` (``now.isoformat()``,
                a UTC-normalised ISO-8601 string) so the GC stage can compare it
                lexicographically against its cutoff.

        Returns:
            The number of thoughts actually archived.

        """
        # Wall-clock cutoff for the atomic write-time inactivity guard: a row is
        # archivable only if its last contact (COALESCE ladder) is at or before
        # this instant. Computed once — ``now`` and the policy are fixed for the
        # run. ``None`` when the gate is disabled (``min_inactivity_age_seconds``
        # of ``0``), mirroring :func:`_hygiene_inactive_enough`.
        inactivity_cutoff_iso: str | None = None
        if policy.min_inactivity_age_seconds > 0:
            inactivity_cutoff_iso = (
                now - datetime.timedelta(seconds=policy.min_inactivity_age_seconds)
            ).isoformat()

        # The wall-clock archival stamp, written once for the whole run: a
        # UTC-normalised ISO-8601 string (``now`` is UTC-aware) so the GC stage
        # can compare ``archived_at`` lexicographically against its cutoff.
        archived_at_iso = now.isoformat()

        archived = 0
        for reason in to_archive:
            before_row = await self._get_thought_row(reason.thought_id)
            if before_row is None:
                continue
            before = self._row_to_thought(before_row)
            if (
                _hygiene_protected(before, policy)
                or not _hygiene_inactive_enough(before, policy, now)
                or before.lifecycle_status
                not in (
                    LifecycleStatus.ACTIVE,
                    LifecycleStatus.CREATED,
                )
            ):
                # Time-of-check re-check on the freshly re-fetched row: a thought
                # pinned, raised to a protected priority, *read* between selection
                # and here (its ``last_accessed_at`` bumped back inside the
                # inactivity window), or already transitioned is skipped. The
                # UPDATE below re-asserts the protection/lifecycle predicate
                # atomically; the inactivity guard is enforced here on the fresh row.
                continue
            # Predicate-guarded write: the WHERE re-checks candidate lifecycle +
            # unprotected + inactive-enough at write time, so even a pin /
            # re-prioritise / read (``last_accessed_at`` bump) landing between the
            # check above and this UPDATE cannot archive a now-protected thought
            # (closes the TOCTOU fully; ``rowcount == 0`` ⇒ raced, skip).
            update_params: list[object] = [
                LifecycleStatus.ARCHIVED.value,
                current_cycle,
                archived_at_iso,
                reason.thought_id,
                LifecycleStatus.ACTIVE.value,
                LifecycleStatus.CREATED.value,
            ]
            priority_guard = ""
            if policy.protected_priorities:
                placeholders = ", ".join("?" for _ in policy.protected_priorities)
                priority_guard = f" AND priority NOT IN ({placeholders})"
                update_params.extend(policy.protected_priorities)
            inactivity_guard = ""
            if inactivity_cutoff_iso is not None:
                # Same lexicographic ordering of UTC-normalised ISO-8601 the model
                # relies on for TEXT time comparisons. All-NULL COALESCE is NULL,
                # so ``NULL <= ?`` is untrue and the row is skipped — the fail-closed
                # branch, consistent with the Python re-check above.
                inactivity_guard = " AND COALESCE(last_accessed_at, updated_at, created_at) <= ?"
                update_params.append(inactivity_cutoff_iso)
            cursor = await self._db.execute(
                "UPDATE thought SET lifecycle_status = ?, "  # noqa: S608 - interpolation is only ``?`` placeholders
                "expires_at = NULL, archived_at_cycle = ?, archived_at = ?, "
                "revision = revision + 1 "
                "WHERE thought_id = ? AND lifecycle_status IN (?, ?) AND pinned = 0"
                + priority_guard
                + inactivity_guard,
                update_params,
            )
            if cursor.rowcount <= 0:
                continue
            archived += 1
            if self._journal is not None:
                after = before.evolve(
                    lifecycle_status=LifecycleStatus.ARCHIVED.value,
                    expires_at=None,
                    archived_at_cycle=current_cycle,
                    archived_at=archived_at_iso,
                )
                await self._journal.append(
                    mutation_type="UPDATE_THOUGHT",
                    target_id=reason.thought_id,
                    delta={
                        "before": before.model_dump(mode="json"),
                        "after": after.model_dump(mode="json"),
                        "eviction_reason": reason.to_delta(),
                    },
                )
        return archived

    async def _hygiene_gc(
        self,
        *,
        policy: HygienePolicyConfig,
        current_cycle: int,
        now: datetime.datetime,
    ) -> _HygieneGcOutcome:
        """Physically delete hygiene-archived thoughts past both restore windows.

        **Write-lock classification: under the lock via its caller.** Called
        only from ``run_hygiene``, which holds ``_write_lock`` around both
        ``_hygiene_archive``'s call and this one. Its own call to
        ``retire_orphan_reflections`` (which writes via ``update_thought``)
        is reentrant-safe on the same task for the same reason.

        **Reports what it did, not just the delete count.** ``run_hygiene``
        decides whether its unit has anything to commit from what actually
        survived, not from a per-call count alone — a retirement can be the
        only surviving write in an otherwise-empty pass, and a vetoed
        delete's own trigger-driven write must not count even though
        ``self._db.total_changes`` would still show it. The returned
        :class:`_HygieneGcOutcome` carries the retirement count and the
        per-candidate ``wrote_anything`` signal alongside the delete count for
        exactly that decision; ``run_hygiene`` still reports only the delete
        count on the public :class:`~engrava.infrastructure.sqlite.hygiene.HygieneResult`.

        Stage 2 — runs only when ``auto_gc_enabled``. A thought is GC-eligible
        only when it was archived **by hygiene** (``archived_at_cycle IS NOT
        NULL``), **both** restore windows have elapsed — the cycle window
        (``current_cycle - archived_at_cycle >= gc_min_archive_age_cycles``)
        **and** the wall-clock window
        (``archived_at <= now - gc_restore_window_seconds``) — and it is not
        protected (``pinned`` or a protected priority). The eligible set is
        ordered ``archived_at_cycle ASC, thought_id ASC`` (oldest-archived first)
        and truncated to ``max_evictions_per_run``.

        Deletion order per thought is **orphan-reflection sweep -> cascade delete
        -> vec0 vector purge**: the sweep retires any REFLECTION whose entire
        source cluster would become non-live so no dangling ``CONSOLIDATED_FROM``
        synthesis is left, the cascade drops FK-reachable edges / embeddings /
        actions, and the vec0 vector (outside the FK) is purged explicitly. The
        delete is recorded as an ordinary ``DELETE_THOUGHT`` journal entry with a
        full ``before`` snapshot — GC reclaims the live working set, it does not
        erase the content from the append-only journal.

        Args:
            policy: The active hygiene policy (windows, cap, protected priorities).
            current_cycle: The current cycle (drives the cycle-window check).
            now: The run's wall-clock instant (drives the wall-clock-window
                cutoff), injected once per run so the eligible set is
                deterministic.

        Returns:
            A :class:`_HygieneGcOutcome` with the number of thoughts
            physically deleted, the number of orphan REFLECTIONs retired, and
            whether any candidate's own delete wrote something even where it
            did not count toward the delete total.

        """
        eligible = await self._hygiene_gc_eligible(
            policy=policy, current_cycle=current_cycle, now=now
        )
        if not eligible:
            return _HygieneGcOutcome(gc_count=0, retired_count=0, wrote_anything=False)

        # Retire orphan REFLECTIONs *before* any delete so a synthesis never
        # outlives its whole source cluster with a dangling edge.
        retired_count = await self.retire_orphan_reflections()

        gc_count = 0
        wrote_anything = False
        for thought in eligible:
            before_row = await self._get_thought_row(thought.thought_id)
            if before_row is None:
                continue
            vec_rowid = await self._embedding_rowid_for_thought(thought.thought_id)
            # Parent delete and explicit child deletes as one atomic unit —
            # see _delete_thought_atomic for why. Its own ``wrote_anything``
            # is folded in unconditionally (mirroring cleanup_expired's
            # DELETE-strategy loop), not gated on ``deleted``: a candidate a
            # trigger silently vetoes reports ``wrote_anything=False`` here
            # (its own savepoint rolled the veto's own writes back too), but
            # a future orphan-sweep-only case reporting ``True`` here must
            # still count as a surviving write for run_hygiene's own
            # finalization, even though it does not advance ``gc_count``.
            result = await self._delete_thought_atomic(thought.thought_id)
            wrote_anything = result.wrote_anything or wrote_anything
            if not result.deleted:
                continue
            await self._purge_orphan_vector(vec_rowid)
            gc_count += 1
            if self._journal is not None:
                await self._journal.append(
                    mutation_type="DELETE_THOUGHT",
                    target_id=thought.thought_id,
                    delta={
                        "before": self._row_to_thought(before_row).model_dump(mode="json"),
                        "after": None,
                        "eviction_reason": {
                            "mechanism": "hygiene",
                            "stage": "gc",
                            "archived_at_cycle": thought.archived_at_cycle,
                            "gc_min_archive_age_cycles": policy.gc_min_archive_age_cycles,
                            "archived_at": thought.archived_at,
                            "gc_restore_window_seconds": policy.gc_restore_window_seconds,
                        },
                    },
                )
        return _HygieneGcOutcome(
            gc_count=gc_count, retired_count=retired_count, wrote_anything=wrote_anything
        )

    async def _hygiene_gc_eligible(
        self,
        *,
        policy: HygienePolicyConfig,
        current_cycle: int,
        now: datetime.datetime,
    ) -> list[ThoughtRecord]:
        """Resolve the deterministic, capped GC-eligible set.

        Selects ARCHIVED thoughts that hygiene archived (``archived_at_cycle IS
        NOT NULL``) for which **both** restore windows have elapsed, excluding
        protected thoughts, ordered ``archived_at_cycle ASC, thought_id ASC`` and
        capped at ``max_evictions_per_run``:

        * **Cycle window** — ``archived_at_cycle <= current_cycle -
          gc_min_archive_age_cycles`` (computed off the explicit
          ``archived_at_cycle`` column, so a thought archived by any other path,
          whose ``archived_at_cycle`` is ``NULL``, is structurally excluded).
        * **Wall-clock window** — when ``gc_restore_window_seconds > 0``, the row
          must additionally satisfy ``archived_at IS NOT NULL AND archived_at <=
          now - gc_restore_window_seconds`` (a lexicographic ISO-8601 compare,
          valid on the UTC-normalised timestamps this module writes). This
          predicate **fails closed** for a hygiene-archived row with
          ``archived_at IS NULL`` (archived before the column existed): its
          real-time age is unknowable, so the irreversible stage never reaps it.
          Setting ``gc_restore_window_seconds = 0`` omits this predicate entirely
          (cycle-only, backward-compatible with the pre-wall-clock behaviour).

        Requiring the **additional** window can only ever *shrink* the eligible
        **candidate** pool (the monotone-safe property): before the cap, the set
        of rows passing both windows is a subset of the ``gc_restore_window_seconds
        = 0`` (cycle-only) candidate set. When ``max_evictions_per_run`` does not
        bind, the returned set is likewise a subset. Under a **binding** cap the
        deterministic ``ORDER BY … LIMIT`` top-N may instead reap a *different*
        genuinely-eligible row (one that a freed young/legacy slot lets surface) —
        a benign rate-limit reshuffle, never a row that fails either window. The
        per-candidate safety invariant (nothing is reaped that is not past both
        windows) always holds; the whole-set subset relation holds when the cap is
        non-binding, exactly mirroring the archive-stage minimum-inactivity gate.

        Args:
            policy: The active hygiene policy.
            current_cycle: The current cycle (drives the cycle window).
            now: The run's wall-clock instant (drives the wall-clock window
                cutoff). A timezone-aware UTC ``datetime``.

        Returns:
            The GC-eligible thoughts in delete order.

        """
        max_archived_cycle = current_cycle - policy.gc_min_archive_age_cycles
        # Exclude protected rows in SQL so the LIMIT is spent on genuinely
        # reap-eligible thoughts — a protected hygiene-archived row (archived,
        # then later pinned / raised to a protected priority) must not consume a
        # cap slot and starve younger eligible rows. The Python re-check below
        # stays as defence-in-depth.
        params: list[object] = [LifecycleStatus.ARCHIVED.value, max_archived_cycle]
        # Cycle-axis restore window. ``archived_at_cycle`` is set by the hygiene
        # archive stage only, so ``archived_at_cycle IS NOT NULL`` is redundant
        # against ordinary data — the comparison beside it already rejects a
        # NULL row. It stays explicit because it is what keeps a TTL- or
        # manually-archived row (``archived_at_cycle IS NULL``) excluded if the
        # comparison is ever rewritten into a NULL-tolerant form (e.g.
        # ``COALESCE(archived_at_cycle, 0) <= ?``), which the method's contract
        # forbids reaping.
        # Wall-clock restore window (in addition to the cycle window). When
        # disabled (``gc_restore_window_seconds == 0``) the predicate is omitted,
        # so a hygiene-archived row with ``archived_at IS NULL`` stays cycle-only
        # eligible (the pre-wall-clock behaviour the operator opted back into);
        # when active, ``archived_at IS NOT NULL`` makes a NULL-stamped legacy row
        # fail closed.
        wall_clock_clause = ""
        if policy.gc_restore_window_seconds > 0:
            max_archived_at_iso = (
                now - datetime.timedelta(seconds=policy.gc_restore_window_seconds)
            ).isoformat()
            wall_clock_clause = "  AND archived_at IS NOT NULL AND archived_at <= ? "
            params.append(max_archived_at_iso)
        priority_clause = ""
        if policy.protected_priorities:
            placeholders = ", ".join("?" for _ in policy.protected_priorities)
            priority_clause = f"  AND priority NOT IN ({placeholders}) "
            params.extend(policy.protected_priorities)
        params.append(policy.max_evictions_per_run)
        cursor = await self._db.execute(
            "SELECT * FROM thought "  # noqa: S608 - interpolation is only ``?`` placeholders
            "WHERE lifecycle_status = ? "
            "  AND archived_at_cycle IS NOT NULL "
            "  AND archived_at_cycle <= ? "
            "  AND pinned = 0 "
            f"{wall_clock_clause}"
            f"{priority_clause}"
            "ORDER BY archived_at_cycle ASC, thought_id ASC "
            "LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        eligible: list[ThoughtRecord] = []
        for row in rows:
            thought = self._row_to_thought(row)
            if _hygiene_protected(thought, policy):
                continue
            eligible.append(thought)
        return eligible

    async def retire_orphan_reflections(self) -> int:
        """Retire REFLECTIONs whose entire source cluster has left ACTIVE.

        A REFLECTION is a derived synthesis of a live cluster. Once **every**
        thought it was consolidated from is no longer ``ACTIVE`` (all
        ``ARCHIVED`` / ``DONE`` — i.e. the synthesis now summarises nothing
        live), the REFLECTION is retired ``ACTIVE -> ARCHIVED`` so ordinary GC
        can reclaim it (cascading its centroid embedding and
        ``CONSOLIDATED_FROM`` edges). This is the shared store-owned
        implementation used both by dreaming consolidation and by the Memory
        Hygiene GC stage (run there **before** any delete so no REFLECTION is
        left summarising a cluster the delete would empty).

        **Full coverage.** The sweep inspects *every* ACTIVE REFLECTION, not just
        the first page. ``list_thoughts`` orders by ``updated_cycle DESC`` and is
        capped per call, so a long-untouched orphan (low ``updated_cycle``) can
        fall beyond a single capped page and never be seen. To honour the
        "for each ACTIVE REFLECTION" contract regardless of how many REFLECTIONs
        exist, the candidate set is collected by walking successive pages
        (``limit`` / ``offset``) until a short page is returned.

        **Collect-then-retire ordering.** All candidate ids are gathered into a
        list *first*, and only then retired in a second pass. Retiring flips a
        REFLECTION ``ACTIVE -> ARCHIVED``, which drops it out of the
        ``lifecycle_status="ACTIVE"`` filter; mutating during pagination would
        shift every later page's offset and silently skip rows. Collecting the
        full set against a stable filter before any mutation avoids that offset
        drift. Ids are de-duplicated defensively against ties in the non-total
        ``updated_cycle`` ordering crossing a page boundary.

        Guards:

        * **100% threshold** — a REFLECTION with at least one still-ACTIVE source
          is kept; the synthesis still summarises live members.
        * **At least one source** — a REFLECTION with zero ``CONSOLIDATED_FROM``
          edges (defensive: malformed / legacy) is never retired by an
          all-non-ACTIVE rule firing over an empty set.

        The check is a deterministic set query over each candidate's source
        lifecycle statuses — no model call.

        Returns:
            The number of REFLECTIONs retired during this sweep.

        """
        # Phase 1 — collect EVERY ACTIVE REFLECTION id by paginating the full
        # set. Done before any mutation so the ACTIVE filter stays stable and
        # offsets do not drift (see the collect-then-retire note above).
        candidate_ids: list[str] = []
        seen: set[str] = set()
        offset = 0
        while True:
            page = await self.list_thoughts(
                thought_type=ThoughtType.REFLECTION.value,
                lifecycle_status=LifecycleStatus.ACTIVE.value,
                limit=_ORPHAN_SWEEP_PAGE_SIZE,
                offset=offset,
            )
            for reflection in page:
                if reflection.thought_id not in seen:
                    seen.add(reflection.thought_id)
                    candidate_ids.append(reflection.thought_id)
            if len(page) < _ORPHAN_SWEEP_PAGE_SIZE:
                # Short (or empty) page -> the full set has been read.
                break
            offset += _ORPHAN_SWEEP_PAGE_SIZE

        # Phase 2 — retire orphans. Safe to mutate now that the full candidate
        # set is materialised.
        retired = 0
        for reflection_id in candidate_ids:
            source_statuses = await self.consolidated_source_statuses(reflection_id)
            # Require >= 1 source AND 100% of them non-ACTIVE.
            if not source_statuses:
                continue
            if any(status == LifecycleStatus.ACTIVE.value for status in source_statuses):
                continue
            await self.update_thought(
                reflection_id,
                lifecycle_status=LifecycleStatus.ARCHIVED,
            )
            retired += 1

        return retired

    def attach_dreaming_extension(self, extension: DreamingConsolidatorProtocol) -> None:
        """Wire a Dreaming consolidator onto this store.

        This is the supported alternative to writing the private
        ``_dreaming_extension`` attribute directly — the two are the same
        underlying slot, so whichever one last ran wins and :meth:`consolidate`
        cannot tell them apart. :meth:`from_config` itself still uses the
        private write internally; this method exists for every other caller
        (a hand-built store, a downstream integration, third-party code) that
        today has no supported way to do what it is already doing.

        Calling this again **replaces** whatever was attached before,
        including one installed by :meth:`from_config` — the same behaviour
        as re-assigning the private attribute has always had. There is no
        separate "already attached" error: a second call is a deliberate
        re-wiring, not a mistake this method can distinguish from one.

        Detaching is deliberately **not** part of this seam. There is no
        supported way to remove an attached extension in this workstream;
        that gap is not an oversight, it is simply not yet built.

        Args:
            extension: A consolidator satisfying
                :class:`~engrava.domain.protocols.dreaming.DreamingConsolidatorProtocol`.
                This method checks ``extension`` against that
                ``runtime_checkable`` protocol (which looks for a
                ``run_consolidation`` member) and raises below if that check
                fails. The check does not verify the method's parameters or
                that it is a coroutine function, so a same-named but
                wrong-shaped ``run_consolidation`` — sync instead of async, or
                a different signature — passes this check and only fails
                once :meth:`consolidate` actually calls it.

        Raises:
            TypeError: When ``extension`` fails the ``runtime_checkable``
                protocol check above.

        """
        if not isinstance(extension, DreamingConsolidatorProtocol):
            msg = (
                "attach_dreaming_extension() requires a DreamingConsolidatorProtocol "
                f"implementation (an async run_consolidation(store, current_cycle) "
                f"method); got {type(extension).__name__!r}"
            )
            raise TypeError(msg)
        self._dreaming_extension = extension

    async def consolidate(self, *, current_cycle: int | None = None) -> ConsolidationResult:
        """Run one dreaming consolidation cycle on this store.

        The invocable entry point for a store built via :meth:`from_config`
        with ``dreaming.enabled`` — it lets a YAML-only caller run consolidation
        without constructing a consolidator by hand. It first flushes
        any pending access-buffer events (so the ``frequency`` signal sees the
        latest access counts this cycle), then runs the wired extension's
        consolidation with access tracking suppressed for the extension's own
        internal reads (its candidate scans / member resolution are machinery,
        not caller retrievals, and must not feed the frequency signal).

        As an operator convenience, when a hygiene policy is configured with
        ``enabled=True`` **and** this cycle satisfies the cadence
        (``current_cycle % check_every_n_cycles == 0``), one Memory Hygiene pass
        runs at the **end** of the cycle — after promotion and the
        orphan-reflection sweep. The hygiene decision logic is independent of
        whether promotion produced anything; its result is not folded into the
        returned :class:`ConsolidationResult` (a hygiene-off store is unchanged).
        An explicit :meth:`run_hygiene` call bypasses this cadence.

        Args:
            current_cycle: The current cognitive cycle number, driving the
                cycle-based recency / staleness signals and promotion age gates.
                Optional: when omitted (``None``), it is pulled from a configured
                ``cycle_provider`` (an explicit value — including ``0`` — always
                wins).

        Returns:
            The :class:`ConsolidationResult` for the run.

        Raises:
            RuntimeError: When dreaming is not enabled/wired on this store
                (built manually, or ``dreaming.enabled`` is false) — there is
                no extension to run.
            ValueError: When no cycle is available — neither an explicit
                ``current_cycle`` nor a configured ``cycle_provider``.
            CycleProviderError: When a configured provider returns an invalid
                value (not an ``int``, a ``bool``, or negative).

        """
        if self._dreaming_extension is None:
            msg = (
                "consolidate() requires dreaming to be enabled: build the store via "
                "from_config with extensions.dreaming.enabled = true"
            )
            raise RuntimeError(msg)
        # Resolve after the wiring guard (nothing to run without an extension) and
        # never invent a default. The resolved int is passed explicitly to the
        # inner run + run_hygiene, so the provider is pulled at most once.
        current_cycle = self._require_current_cycle(current_cycle, operation="consolidate()")
        await self.flush_access_buffer()
        async with self.suppress_access_tracking():
            result = await self._dreaming_extension.run_consolidation(self, current_cycle)
            if self._hygiene_due(current_cycle):
                await self.run_hygiene(current_cycle=current_cycle)
        return result

    def _hygiene_due(self, current_cycle: int) -> bool:
        """Report whether the ``consolidate()`` convenience hygiene pass runs.

        The convenience invocation runs only when a hygiene policy is configured
        and enabled and this cycle satisfies ``check_every_n_cycles`` (the cadence
        gate applies **only** to this ``consolidate()``-driven invocation — an
        explicit :meth:`run_hygiene` bypasses it).

        Args:
            current_cycle: The current cognitive cycle number.

        Returns:
            ``True`` when the cadence-gated hygiene pass should run this cycle.

        """
        policy = self._hygiene_policy
        return (
            policy is not None
            and policy.enabled
            and current_cycle % policy.check_every_n_cycles == 0
        )

    # ------------------------------------------------------------------
    # ActionRecord CRUD
    # ------------------------------------------------------------------

    async def create_action(self, action: ActionRecord) -> ActionRecord:
        """Persist a new action record.

        When the created action already has a **terminal** status
        (``CONFIRMED`` / ``FAILED``) it is outcome-affecting, so the source
        thought's ``action_outcome_score`` is recomputed in the same
        transaction (see :meth:`_recompute_action_outcome`). A non-terminal
        create leaves the score untouched, so an action-free — or
        only-in-flight — store never writes an outcome score and stays
        byte-identical to one built before this feature.

        **The action write and the recompute's write + journal entry are one
        failure-atomic unit**, via :meth:`_write_readback_savepoint` — see
        :meth:`update_thought` for what that protects against. The action's
        own ``INSERT`` is not itself journaled (a separate, tracked gap), but
        a failed or cancelled recompute append still unwinds it: wrapping only
        the recompute would leave the action row pending after a failed call.

        Args:
            action: The action record to create.

        Returns:
            The persisted action record.

        """
        async with self._write_lock:
            async with self._write_readback_savepoint("create_action", begin="IMMEDIATE"):
                await self._db.execute(
                    "INSERT INTO action "
                    "(action_id, source_thought_id, action_type, intent, "
                    " status, verification_status, raw_metrics_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        action.action_id,
                        action.source_thought_id,
                        action.action_type.value,
                        action.intent,
                        action.status.value,
                        action.verification_status.value,
                        action.raw_metrics_json,
                    ),
                )
                if action.status in _TERMINAL_ACTION_STATUSES:
                    await self._recompute_action_outcome(action.source_thought_id)
            await self._maybe_commit()
        return action

    async def update_action(
        self,
        action_id: str,
        *,
        status: ActionStatus | None = None,
        verification_status: VerificationStatus | None = None,
    ) -> ActionRecord:
        """Advance a stored action's status and/or verification status.

        The action's own state machine governs ``status`` changes: a change
        is validated via :meth:`ActionRecord.evolve` (which calls
        ``can_transition_to``), so an illegal jump (e.g. ``PLANNED`` →
        ``CONFIRMED``) raises :class:`InvalidTransitionError`. Transition
        validation applies **only when ``status`` actually changes** — a
        verification-only update never touches the status machine and is
        therefore permitted in **any** status, including a terminal
        ``CONFIRMED`` / ``FAILED`` action (verification legitimately advances
        while the status stays terminal). The transition is validated against the
        record *this* call read. On a real change the write carries a
        ``revision`` guard exactly like :meth:`update_thought`'s: the row's
        ``revision`` at the start of this call is checked and incremented
        atomically by the same ``UPDATE``. A no-op (below) issues no ``UPDATE``
        at all, so it cannot go stale. The
        read, the transition check, and the write share the same
        task-reentrant :attr:`_write_lock` critical section every other guarded
        write on this instance uses, so a competing move issued by
        a *different task on this instance* can no longer land in that window —
        it is delayed until this call's write has committed, and then it
        validates against the state this call actually left. ``verification_status``
        is not gated
        by the lifecycle state: it may be set on a non-terminal action too, but
        such an action contributes nothing to ``action_outcome_score`` (the
        aggregate counts only terminal actions), so a premature verification
        mark is harmless — it takes effect only once the action is terminal.

        A **no-op** update — every supplied field already equals the stored
        value — returns the unchanged record and is **fully side-effect-free**:
        no persisted write, no journal entry, no feedback recompute, and (since
        it writes nothing) no commit — it never flushes unrelated pending
        writes on the connection.

        On a real change **only the column that changed** is written — a
        status-only update does not re-assert the verification column it read a
        moment ago, so a verification recorded by another writer meanwhile is
        not rolled back. The mutation is journaled as ``UPDATE_ACTION`` (only
        when journaling is enabled), and the source thought's
        ``action_outcome_score`` is recomputed when the change is
        **outcome-affecting** — that is, when it lands a terminal status, or
        changes ``verification_status`` on an already-terminal action. A purely
        non-terminal move (e.g. ``PLANNED`` → ``EXECUTING``) is journaled but
        triggers no recompute. The record returned is read back from storage
        after the write, and the journal ``after`` image is the same read-back.

        **The write, its confirming read-back, and its journal entry are one
        failure-atomic unit**, via :meth:`_write_readback_savepoint` — see
        :meth:`update_thought` for what that protects against and how it
        treats a caller-owned transaction. The ``UPDATE`` also captures its
        cursor and rejects a zero-row match immediately: without that check,
        a row deleted after the initial read (so the ``UPDATE`` matches
        nothing) and re-created under the same ``action_id`` before the
        read-back runs would be reported as though this call had updated it,
        when it had written nothing at all.

        Args:
            action_id: UUID of the action to update.
            status: New status, or ``None`` to leave the status unchanged.
            verification_status: New verification status, or ``None`` to
                leave it unchanged.

        Returns:
            The stored action record (or, for a no-op, the unchanged record).

        Raises:
            ActionNotFoundError: If the action does not exist at the initial
                read.
            StaleDataError: If a real change's guarded ``UPDATE`` matches no
                row — another guarded write landed on this row since it was
                read here (including a delete, and including a delete
                followed by a different row recreated under the same
                ``action_id`` before the read-back could run). Nothing of
                this update is written when it is raised.
            InvalidTransitionError: If a real ``status`` change is illegal
                per the action state machine.
            WriteContentionError: The guarded write could not proceed because
                the connection reported lock contention.

        """
        async with self._write_lock:
            action_row = await self._get_action_row(action_id)
            if action_row is None:
                raise ActionNotFoundError(action_id)
            current = _row_to_action(action_row)
            expected_revision = int(action_row["revision"])

            status_changes = status is not None and status != current.status
            verification_changes = (
                verification_status is not None
                and verification_status != current.verification_status
            )
            if not status_changes and not verification_changes:
                # No-op: nothing to persist, journal, or recompute.
                return current

            changes: dict[str, object] = {}
            if status_changes:
                changes["status"] = status
            if verification_changes:
                changes["verification_status"] = verification_status
            # ``evolve`` validates the status transition when ``status``
            # changes and is a no-op validation-wise for a verification-only
            # change.
            updated = current.evolve(**changes)

            # Write only the column(s) this call moves: a status-only update
            # must not re-assert the verification column it read a moment
            # ago, which would roll back a verification recorded by another
            # writer meanwhile.
            columns: dict[str, object] = {}
            if status_changes:
                columns["status"] = updated.status.value
            if verification_changes:
                columns["verification_status"] = updated.verification_status.value

            async with self._write_readback_savepoint("update_action_readback", begin="DEFERRED"):
                cursor = await self._execute_revision_guarded_write(
                    _build_update_sql(
                        "action",
                        columns,
                        "action_id = ? AND revision = ?",
                        bump_column="revision",
                    ),
                    (*columns.values(), action_id, expected_revision),
                    operation="update_action",
                )
                if cursor.rowcount == 0:
                    raise StaleDataError(
                        entity_type="ActionRecord",
                        entity_id=action_id,
                        expected_version=expected_revision,
                    )

                persisted = await self._read_back_action(action_id)

                if self._journal is not None:
                    await self._journal.append(
                        mutation_type="UPDATE_ACTION",
                        target_id=action_id,
                        delta={
                            "before": {
                                "status": current.status.value,
                                "verification_status": current.verification_status.value,
                            },
                            "after": {
                                "status": persisted.status.value,
                                "verification_status": persisted.verification_status.value,
                            },
                        },
                    )

                # Outcome-affecting iff the change lands a terminal status, or
                # changes verification on an already-terminal action. Because
                # the aggregate reads both status and verification, a
                # verification change on a terminal action IS
                # outcome-affecting. Run inside the same unit as the action's
                # own write and journal entry above: a failed or cancelled
                # recompute append must unwind the action write too, not just
                # its own score write, so wrapping only the recompute is not
                # enough on its own.
                lands_terminal = status_changes and persisted.status in _TERMINAL_ACTION_STATUSES
                verifies_terminal = (
                    verification_changes and persisted.status in _TERMINAL_ACTION_STATUSES
                )
                if lands_terminal or verifies_terminal:
                    await self._recompute_action_outcome(persisted.source_thought_id)

            await self._maybe_commit()
        return persisted

    async def _read_back_action(self, action_id: str) -> ActionRecord:
        """Re-read an action a write just landed on.

        The counterpart of :py:meth:`_read_back_thought` for actions.

        Args:
            action_id: UUID of the action.

        Returns:
            The action as it is stored now.

        Raises:
            ActionNotFoundError: If the row no longer exists, so the write
                cannot be confirmed and no record may be reported for it.

        """
        action = await self._get_action(action_id)
        if action is None:
            raise ActionNotFoundError(action_id)
        return action

    async def _get_action(self, action_id: str) -> ActionRecord | None:
        """Fetch a single action by its ID, or ``None`` when absent.

        Args:
            action_id: UUID of the action.

        Returns:
            The action record, or ``None`` if not found.

        """
        row = await self._get_action_row(action_id)
        return _row_to_action(row) if row is not None else None

    async def _get_action_row(self, action_id: str) -> aiosqlite.Row | None:
        """Fetch the raw ``action`` row by id, or ``None`` when absent.

        The counterpart of :py:meth:`_get_thought_row` / :py:meth:`_get_edge_row`
        for actions: :meth:`update_action` needs the raw ``revision`` column
        for its guard, which :meth:`_get_action`'s mapped
        :class:`~engrava.domain.models.action.ActionRecord` does not carry (it
        is not a domain-model field — see :meth:`update_thought` for why).

        Args:
            action_id: UUID of the action.

        Returns:
            The raw row, or ``None`` if not found.

        """
        cursor = await self._db.execute("SELECT * FROM action WHERE action_id = ?", (action_id,))
        return await cursor.fetchone()

    async def _recompute_action_outcome(self, thought_id: str) -> None:
        """Recompute and persist a thought's denormalised ``action_outcome_score``.

        **Write-lock classification: under the lock via callers.** Called
        only from ``create_action`` and ``update_action``, both of which hold
        ``_write_lock`` around their entire body, including this call.

        Full, idempotent recompute: reads **all** of the thought's actions
        via :meth:`get_actions` (a seek on ``idx_action_source_thought``),
        takes the mean outcome value over its terminal actions (see
        :func:`_aggregate_action_outcome`), and writes the result directly to
        ``thought.action_outcome_score``. Because the new value is a pure
        function of the current action set, running it twice with no
        intervening action change converges to the same score.

        The write is a direct column update (it deliberately does not touch
        ``updated_cycle`` or ``updated_at``, so it is not an optimistic-
        concurrency mutation). When journaling is enabled the change is
        journaled as an ``UPDATE_THOUGHT`` before/after delta over the single
        column. A missing thought (already cascade-deleted) is a silent
        no-op: the ``UPDATE`` simply matches no row.

        Args:
            thought_id: UUID of the thought whose score to recompute.

        """
        before_row = await self._get_thought_row(thought_id) if self._journal is not None else None

        actions = await self.get_actions(thought_id)
        new_score = _aggregate_action_outcome(actions)

        cursor = await self._db.execute(
            "UPDATE thought SET action_outcome_score = ? WHERE thought_id = ?",
            (new_score, thought_id),
        )
        if cursor.rowcount == 0:
            # Thought is gone (cascade-deleted) — nothing to journal.
            return

        if self._journal is not None and before_row is not None:
            before = self._row_to_thought(before_row)
            after = before.model_copy(update={"action_outcome_score": new_score})
            await self._journal.append(
                mutation_type="UPDATE_THOUGHT",
                target_id=thought_id,
                delta={
                    "before": before.model_dump(mode="json"),
                    "after": after.model_dump(mode="json"),
                },
            )

    async def get_actions(self, thought_id: str) -> list[ActionRecord]:
        """Retrieve actions linked to a thought.

        Args:
            thought_id: UUID of the thought.

        Returns:
            List of action records.

        """
        cursor = await self._db.execute(
            "SELECT * FROM action WHERE source_thought_id = ?", (thought_id,)
        )
        rows = await cursor.fetchall()
        return [_row_to_action(r) for r in rows]


# ------------------------------------------------------------------
# Row -> Domain mapper functions (private, module-level)
# ------------------------------------------------------------------


def _row_to_edge(row: aiosqlite.Row) -> EdgeRecord:
    """Map a SQLite row to an EdgeRecord domain model.

    Args:
        row: A row from the edge table.

    Returns:
        An EdgeRecord domain model.

    """
    keys = row.keys()
    source_raw = row["source"] if "source" in keys else None
    decay_raw = row["decay_multiplier"] if "decay_multiplier" in keys else 1.0
    valid_from_raw = row["valid_from"] if "valid_from" in keys else None
    valid_until_raw = row["valid_until"] if "valid_until" in keys else None
    # Read + decode metadata mirroring ``_row_to_thought``. The read side is
    # coupled to the ``update_edge`` write: an un-patched reader would yield
    # ``metadata={}`` for ``current``, so ``update_edge``'s merge would silently
    # wipe stored edge metadata on every partial update.
    metadata_json_raw = row["metadata_json"] if "metadata_json" in keys else "{}"
    metadata_decoded: dict[str, MetadataValue] = (
        json.loads(metadata_json_raw) if metadata_json_raw else {}
    )
    return EdgeRecord(
        edge_id=row["edge_id"],
        from_thought_id=row["from_thought_id"],
        to_thought_id=row["to_thought_id"],
        edge_type=EdgeType(row["edge_type"]),
        weight=row["weight"],
        created_cycle=row["created_cycle"],
        source=KnowledgeSource(source_raw) if source_raw else KnowledgeSource.EXPERIENCE,
        # ``is not None``, never truthiness: ``0.0`` is a valid decay (the field
        # is bounded ``ge=0.0``, not ``gt=0.0``) and it is falsy, so a truthiness
        # fallback reports the ``1.0`` default for a row the database says is
        # ``0.0``. The fallback exists only for a row that predates the column.
        decay_multiplier=float(decay_raw) if decay_raw is not None else 1.0,
        valid_from=valid_from_raw,
        valid_until=valid_until_raw,
        metadata=metadata_decoded,
    )


def _edge_to_core_columns(edge: EdgeRecord) -> dict[str, object]:
    """Map an EdgeRecord to the column values an UPDATE may write.

    The edge counterpart of
    :py:meth:`SqliteEngravaCore._thought_to_core_columns`, keyed by column name
    so an update writes only the columns it owns. ``edge_id`` is absent — it
    identifies the row being updated.

    Args:
        edge: The edge record to encode.

    Returns:
        Mapping of column name to the SQL value for that column.

    """
    return {
        "from_thought_id": edge.from_thought_id,
        "to_thought_id": edge.to_thought_id,
        "edge_type": edge.edge_type.value,
        "weight": edge.weight,
        "created_cycle": edge.created_cycle,
        "source": edge.source.value,
        "decay_multiplier": edge.decay_multiplier,
        "valid_from": edge.valid_from,
        "valid_until": edge.valid_until,
        "metadata_json": json.dumps(edge.metadata, ensure_ascii=False),
    }


def _build_update_sql(
    table: str,
    columns: Iterable[str],
    guard: str,
    *,
    bump_column: str | None = None,
) -> str:
    """Compose an UPDATE that assigns exactly ``columns``, optionally bumping a counter.

    Every update in the store writes a subset of its table's columns — the ones
    the operation owns — so the statement is shaped per call instead of being a
    frozen whole-record string. The table name, the column names and the guard
    are internal literals; every *value* is bound as a parameter.

    Args:
        table: Target table name.
        columns: Column names to assign, each bound to a ``?`` placeholder, in
            the order their values are passed.
        guard: The WHERE clause, its own placeholders included.
        bump_column: When given, an additional ``{bump_column} = {bump_column}
            + 1`` assignment is appended — a self-referential increment, not a
            bound value, which is why it is a separate parameter rather than
            one more entry in ``columns`` (which always binds a caller-supplied
            value). Used for the ``revision`` guard: the row's own current
            value is incremented in the same atomic statement that checks it,
            never read-then-written as two steps.

    Returns:
        The composed UPDATE statement.

    """
    assignments = ", ".join(f"{name} = ?" for name in columns)
    if bump_column is not None:
        bump_assignment = f"{bump_column} = {bump_column} + 1"
        assignments = f"{assignments}, {bump_assignment}" if assignments else bump_assignment
    return f"UPDATE {table} SET {assignments} WHERE {guard}"  # noqa: S608 -- table/columns/guard are internal literals; all values are bound


def _query_is_expert_syntax(query: str) -> bool:
    """Return ``True`` when a query should be parsed as expert FTS5 syntax.

    A query is expert syntax when it holds a *deliberate* FTS5 construct:

    * a **balanced** double-quoted phrase (an even number of ``"``) that wraps
      at least one token,
    * a standalone uppercase boolean operator (``AND``/``OR``/``NOT``), or
    * a whitelisted column filter (``essence:``/``content:``).

    An **odd/unbalanced** number of ``"`` is always bare: it can never form a
    deliberate phrase and would only yield an invalid MATCH. Incidental
    scare-quotes in a natural-language sentence (``he said "run"`` embedded in
    prose, an unterminated ``"quote``) therefore take the bare, sanitizing path
    rather than being misread as expert phrase syntax.

    Expert queries are normalized token-by-token and joined with spaces,
    preserving FTS5's native operators, phrase matching, column filters and
    implicit-AND semantics.

    Bare natural-language queries (none of the above) are instead OR-joined so
    function words cannot block a match; BM25's IDF weighting handles
    uninformative tokens at ranking time.

    Args:
        query: The raw user-facing query string.

    Returns:
        ``True`` for expert syntax, ``False`` for a bare natural-language query.

    """
    if query.count('"') % 2 == 1:
        # An unbalanced quote is never a deliberate phrase; take the bare path
        # so it is sanitized rather than passed through as broken expert syntax.
        return False
    if _has_balanced_quoted_phrase(query):
        return True
    for token in query.split():
        if token in _FTS_BOOLEAN_OPERATORS:
            return True
        if _FTS_FIELD_FILTER_RE.match(token.lstrip("(")):
            return True
    return False


def _has_balanced_quoted_phrase(query: str) -> bool:
    """Return ``True`` when ``query`` holds a balanced quoted phrase with content.

    A balanced quoted phrase is an even, non-zero number of ``"`` where at least
    one quoted span holds a non-whitespace token (so ``""`` or ``" "`` alone
    does not qualify). Splitting on ``"`` places quoted spans at the odd indices
    of the resulting list; the caller guarantees an even quote count, so those
    indices are exactly the inside-quote spans.

    Args:
        query: The raw user-facing query string (assumed to have an even ``"``
            count when a positive result is meaningful).

    Returns:
        ``True`` when at least one quoted span wraps a non-whitespace token.

    """
    # ``len(parts) - 1`` == quote count; an even quote count leaves the
    # inside-quote spans at the odd indices of the split. With no quotes the
    # range is empty and ``any`` is ``False``.
    parts = query.split('"')
    return any(parts[index].strip() for index in range(1, len(parts), 2))


def _normalize_fts_query(query: str) -> str:
    """Normalize a user-facing FTS query to SQLite FTS5-compatible syntax.

    Two query classes are handled:

    * **Expert syntax** (contains a quoted phrase or a standalone uppercase
      ``AND``/``OR``/``NOT``): each token is normalized in place and the tokens
      are joined with spaces, so FTS5's phrase matching, implicit AND, hyphen
      handling and boolean operators all behave exactly as the caller wrote
      them. Hyphenated identifiers such as ``REQ-FUNC*`` are still rewritten to
      the accepted form ``"REQ-FUNC"*``.

    * **Bare natural-language query** (no quotes, no uppercase operators): each
      token expands to zero or more sanitized terms and the terms are joined
      with ``OR``. This lets a question match any document sharing a content
      word, instead of requiring every function word ("what", "was", "my") to
      appear. BM25 IDF weighting keeps uninformative tokens from dominating the
      ranking, so no stopword list or stemmer is needed in any language.

    Unsafe characters (apostrophes, slashes, colons, ...) act as token
    boundaries rather than being deleted, so contractions and clitics like
    ``sister's`` or ``l'école`` split into matchable terms (``sister OR s``)
    instead of becoming an unindexed merged token.

    Args:
        query: The raw user-facing query string.

    Returns:
        An FTS5 MATCH expression. May be empty when no usable term remains.

    """
    expert = _query_is_expert_syntax(query)
    terms: list[str] = []
    for token in query.split():
        terms.extend(_normalize_fts_token(token, expert=expert))
    joiner = " " if expert else " OR "
    return joiner.join(terms)


def _normalize_fts_query_bare(query: str) -> str:
    """Normalize ``query`` through the bare (sanitizing) path unconditionally.

    Every token is sanitized into safe FTS5 fragments -- unsafe characters are
    dropped and wildcards are reduced to valid prefix markers
    (:func:`_collapse_fts_wildcards`) -- and the fragments are OR-joined. For any
    input this yields a syntactically valid FTS5 MATCH expression (or the empty
    string when no indexable term remains). This is the
    execution-time fallback :meth:`SqliteEngravaCore.search_fts` retries with
    when the primary — possibly expert — normalization produced an expression
    that FTS5 rejected, so a stray hazardous character in an expert-looking
    query degrades to a valid bare match instead of silently returning nothing.

    Args:
        query: The raw user-facing query string.

    Returns:
        An always-valid FTS5 MATCH expression, or the empty string when the
        query holds no indexable term.

    """
    terms: list[str] = []
    for token in query.split():
        terms.extend(_normalize_fts_token(token, expert=False))
    return " OR ".join(terms)


def _strip_fts_boundary_punctuation(raw: str) -> str:
    """Strip unsupported leading and trailing punctuation from a bare token.

    Args:
        raw: A single unquoted token.

    Returns:
        The token with leading/trailing characters that FTS5 cannot start or
        end a bare term with removed.

    """
    while raw and not (raw[0].isalnum() or raw[0] in {"_", '"'}):
        raw = raw[1:]

    while raw and not (raw[-1].isalnum() or raw[-1] in {"_", "*"}):
        raw = raw[:-1]

    return raw


def _collapse_fts_wildcards(fragment: str) -> str:
    """Reduce ``*`` wildcards in a bare fragment to FTS5-valid positions.

    FTS5 accepts ``*`` only as a prefix marker attached to a preceding term
    character (``foo*``, ``x*y*z``). A leading ``*``, a standalone ``*``, or a
    run of consecutive ``*`` (``foo**``, ``foo***bar``) is a syntax error. This
    keeps a ``*`` only when it directly follows a non-``*`` character and
    collapses each run to a single marker, so the fragment is always a
    syntactically valid FTS5 term while genuine prefix search (``foo*``) is
    preserved.

    Args:
        fragment: A safe fragment containing only word characters, ``-`` and
            ``*`` (as produced by the unsafe-character split).

    Returns:
        The fragment with every ``*`` reduced to a valid single prefix marker.
        May be the empty string when the fragment was nothing but wildcards.

    """
    collapsed: list[str] = []
    for char in fragment:
        if char == "*":
            # Keep a wildcard only when it attaches to a real term character;
            # this drops leading wildcards and every wildcard after the first in
            # a consecutive run.
            if collapsed and collapsed[-1] != "*":
                collapsed.append(char)
        else:
            collapsed.append(char)
    return "".join(collapsed)


def _sanitize_fts_bare_token(raw: str) -> list[str]:
    """Split an unquoted bare token into safe FTS5 fragments.

    Unsafe characters become fragment boundaries rather than being deleted, so
    a contraction or clitic such as ``sister's`` splits into ``["sister", "s"]``
    (which the ``unicode61`` tokenizer also produced at index time) instead of
    merging into an unindexed ``sisters``. Each fragment's wildcards are then
    reduced to FTS5-valid positions (see :func:`_collapse_fts_wildcards`) so a
    consecutive- or leading-``*`` shape such as ``foo**`` can never reach the
    ``MATCH`` as an invalid term.

    Args:
        raw: A single unquoted token, already paren-stripped.

    Returns:
        A list of non-empty safe fragments, in order. May be empty when the
        token holds no indexable characters.

    """
    stripped = _strip_fts_boundary_punctuation(raw)
    split = _FTS_UNSAFE_CHAR_RE.sub(" ", stripped)
    collapsed = (_collapse_fts_wildcards(fragment) for fragment in split.split())
    return [fragment for fragment in collapsed if fragment]


def _normalize_fts_token(token: str, *, expert: bool) -> list[str]:
    """Normalize a single token into zero or more FTS5 terms.

    Args:
        token: A whitespace-delimited token from the raw query.
        expert: ``True`` when the surrounding query is expert syntax. In expert
            mode quoted phrases and uppercase operators pass through unchanged;
            in bare mode every token is sanitized into plain OR-terms.

    Returns:
        The FTS5 terms this token contributes. A bare contraction may yield
        several terms (``sister's`` -> ``["sister", "s"]``); an empty or
        all-punctuation token yields ``[]``.

    """
    if not token:
        return []
    if expert and '"' in token:
        return [token]
    if expert and token in _FTS_BOOLEAN_OPERATORS:
        return [token]

    leading = ""
    trailing = ""
    raw = token
    while raw.startswith("("):
        leading += "("
        raw = raw[1:]
    while raw.endswith(")"):
        trailing = ")" + trailing
        raw = raw[:-1]

    if expert and _FTS_FIELD_FILTER_RE.match(raw):
        return [f"{leading}{raw}{trailing}"]

    fragments = _sanitize_fts_bare_token(raw)
    if not fragments:
        return []

    terms = [
        _format_fts_bare_fragment(fragment, in_bare_query=not expert) for fragment in fragments
    ]
    if expert:
        # Expert mode keeps each original token as one term, re-attaching any
        # parentheses the caller used for grouping.
        terms[0] = f"{leading}{terms[0]}"
        terms[-1] = f"{terms[-1]}{trailing}"
    return terms


def _fragment_exposes_fts_operator(fragment: str) -> bool:
    """Report whether a bare fragment exposes an uppercase FTS5 boolean operator.

    In a bare, OR-joined query FTS5 reads an uppercase ``AND``/``OR``/``NOT`` as
    a boolean *operator*, never a term, so emitting one as a bareword yields an
    invalid ``MATCH`` (``forum OR NOT OR body`` and ``field*NOT`` both raise). A
    keyword is exposed when it forms a whole ``*``-delimited segment of the
    fragment: the entire fragment (``NOT``), the segment after a prefix marker
    (``field*NOT``) or before one (``NOT*field``). A keyword merely glued into a
    larger token (``NOTbar``) is an ordinary term and is *not* exposed. ``*`` is
    the only intra-fragment boundary to consider, because a hyphen already
    forces the fragment to be phrase-quoted upstream.

    Args:
        fragment: A safe fragment (word characters, ``-`` and ``*`` only) with
            any trailing prefix marker already stripped by the caller.

    Returns:
        ``True`` when a ``*``-delimited segment equals an uppercase FTS5 boolean
        operator, so the fragment must be phrase-quoted to parse as a literal.

    """
    return any(segment in _FTS_BOOLEAN_OPERATORS for segment in fragment.split("*"))


def _format_fts_bare_fragment(fragment: str, *, in_bare_query: bool) -> str:
    """Format a single sanitized fragment as an FTS5 term.

    Preserves a trailing ``*`` prefix marker and phrase-quotes a fragment that a
    bare term would otherwise misparse: a hyphenated identifier always (FTS5
    would read the hyphen as a column/operator), and — only in a bare, OR-joined
    query (``in_bare_query``) — a fragment that exposes an uppercase
    ``AND``/``OR``/``NOT`` (see :func:`_fragment_exposes_fts_operator`), which
    FTS5 would otherwise read as a boolean operator and reject. Phrase-quoting
    forces literal-term parsing while still matching the same case-folded
    documents. Expert-mode callers pass ``in_bare_query=False`` so a deliberate
    operator token is left byte-for-byte as the caller wrote it.

    Args:
        fragment: A safe fragment containing only word characters, ``-`` or a
            trailing ``*``.
        in_bare_query: ``True`` when the fragment belongs to a bare, OR-joined
            query, so an exposed uppercase boolean operator must be neutralized.

    Returns:
        The fragment rewritten as a valid FTS5 term.

    """
    suffix = ""
    if fragment.endswith("*"):
        fragment = fragment[:-1]
        suffix = "*"
    if "-" in fragment or (in_bare_query and _fragment_exposes_fts_operator(fragment)):
        return f'"{fragment}"{suffix}'
    return f"{fragment}{suffix}"


def _row_to_action(row: aiosqlite.Row) -> ActionRecord:
    """Map a SQLite row to an ActionRecord domain model.

    Args:
        row: A row from the action table.

    Returns:
        An ActionRecord domain model.

    """
    return ActionRecord(
        action_id=row["action_id"],
        source_thought_id=row["source_thought_id"],
        action_type=ActionType(row["action_type"]),
        intent=row["intent"],
        status=ActionStatus(row["status"]),
        verification_status=VerificationStatus(row["verification_status"]),
        raw_metrics_json=row["raw_metrics_json"],
    )


def _row_to_embedding(row: aiosqlite.Row) -> EmbeddingRecord:
    """Map a SQLite row to an EmbeddingRecord domain model.

    Args:
        row: A row from the embedding table.

    Returns:
        An EmbeddingRecord domain model.

    """
    return EmbeddingRecord(
        embedding_id=row["embedding_id"],
        owner_type=row["owner_type"],
        owner_id=row["owner_id"],
        model_name=row["model_name"],
        dimension=row["dimension"],
        vector_blob=row["vector_blob"],
        created_at=row["created_at"],
    )


def _encode_consolidated(value: list[str] | None) -> str | None:
    """Encode consolidated_from list as JSON string for storage.

    Args:
        value: List of source thought IDs, or None.

    Returns:
        JSON string, or None.

    """
    if value is None:
        return None
    return json.dumps(value)


def _decode_consolidated(raw: str | None) -> list[str] | None:
    """Decode consolidated_from JSON string from storage.

    Args:
        raw: JSON string from database, or None.

    Returns:
        List of source thought IDs, or None.

    """
    if raw is None:
        return None
    result: list[str] = json.loads(raw)
    return result


def _clamp_decay(raw: float) -> float:
    """Clamp a ``decay_function`` return into the fail-safe ``[0.0, 1.0]`` range.

    A non-finite value (``NaN`` / ``±inf``) maps to ``1.0`` (no decay) — the
    fail-safe direction, since decay can then only lower an eviction-score toward
    archive, never resurrect one above threshold or cause a spurious eviction. A
    finite value is clamped into ``[0.0, 1.0]``.

    Args:
        raw: The raw ``decay_function`` hook result.

    Returns:
        A decay multiplier in ``[0.0, 1.0]``.

    """
    if not math.isfinite(raw):
        return 1.0
    return max(0.0, min(1.0, raw))


def _hygiene_protected(thought: ThoughtRecord, policy: HygienePolicyConfig) -> bool:
    """Report whether a thought is protected from hygiene archival / GC.

    A thought is protected when it is ``pinned`` (the durable never-forget
    marker) or its priority is listed in ``protected_priorities`` (default
    ``P1``). ``confidence`` is deliberately **not** consulted — a model-confidence
    estimate is not a user keep-decision.

    Args:
        thought: The thought to test.
        policy: The active hygiene policy (for ``protected_priorities``).

    Returns:
        ``True`` when the thought must never be auto-archived or auto-GC'd.

    """
    return thought.pinned or thought.priority.value in policy.protected_priorities


def _ensure_utc(moment: datetime.datetime) -> datetime.datetime:
    """Return ``moment`` as a timezone-aware UTC ``datetime``.

    A naive input is interpreted as UTC (the domain's naive-as-UTC convention,
    mirroring :func:`~engrava.domain.models._temporal.parse_iso8601_to_utc`); an
    aware input in any other offset is converted to UTC. Normalising to UTC keeps
    both aware ``datetime`` arithmetic and lexicographic comparison against the
    UTC-normalised ISO-8601 timestamp columns correct.

    Args:
        moment: The instant to normalise (naive or aware).

    Returns:
        The same instant as a timezone-aware UTC ``datetime``.

    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=datetime.UTC)
    return moment.astimezone(datetime.UTC)


def _hygiene_inactive_enough(
    thought: ThoughtRecord,
    policy: HygienePolicyConfig,
    now: datetime.datetime,
) -> bool:
    """Report whether a thought has been untouched long enough to be archivable.

    The minimum-inactivity-age gate: a thought is eligible for hygiene archival
    only once the wall-clock time since its last contact reaches
    ``policy.min_inactivity_age_seconds``. Last contact is the first present of
    ``last_accessed_at`` (last read), ``updated_at`` (last write), then
    ``created_at`` (creation) — the ``COALESCE`` ladder that realises the
    "time since last read *or* creation" baseline. Below the threshold the
    thought is protected, exactly like ``pinned`` / ``protected_priorities``;
    this only ever *adds* protection (it never causes an archival that the
    keep-score alone would not).

    Fails **closed**: when all three timestamps are ``None`` (a legacy row with
    no transaction times) the age is indeterminate and the thought is protected.
    A ``min_inactivity_age_seconds`` of ``0`` disables the gate — every thought
    passes, restoring the pre-gate behaviour.

    Args:
        thought: The candidate thought.
        policy: The active hygiene policy (for ``min_inactivity_age_seconds``).
        now: The run's wall-clock instant, injected once per run so the age
            boundary is deterministic. A timezone-aware ``datetime`` (UTC).

    Returns:
        ``True`` when the thought is inactive for at least
        ``min_inactivity_age_seconds`` (or the gate is disabled); ``False`` when
        it was contacted too recently or its last-contact time is indeterminate.

    """
    if policy.min_inactivity_age_seconds == 0:
        return True
    # COALESCE ladder: a valid timestamp is a non-empty ISO-8601 string (empty
    # strings are rejected by the model validator), so ``or`` selects the first
    # present bound exactly as SQL COALESCE would.
    last_contact = thought.last_accessed_at or thought.updated_at or thought.created_at
    if last_contact is None:
        return False
    age = now - parse_iso8601_to_utc(last_contact)
    return age.total_seconds() >= policy.min_inactivity_age_seconds


def _encode_provenance(value: ProvenanceContext | None) -> str | None:
    """Encode the optional provenance sub-model as a JSON string for storage.

    ``None`` maps to ``None`` (a SQL NULL) so a thought with no provenance
    writes a NULL ``provenance`` column and is byte-identical to a pre-feature
    row.  When present, ``model_dump_json`` produces a compact JSON document
    with the ``json_extract`` identity paths (``$.session_id`` / ``$.actor_id``)
    the expression indexes read.

    Args:
        value: The provenance sub-model, or ``None``.

    Returns:
        The JSON-serialised provenance document, or ``None``.

    """
    if value is None:
        return None
    return value.model_dump_json()


def _decode_provenance(raw: str | None) -> ProvenanceContext | None:
    """Decode a stored provenance JSON string back into the sub-model.

    A NULL column (``raw is None``) round-trips to ``None`` — the byte-identical
    default for a thought created without provenance.

    Args:
        raw: JSON string from the ``provenance`` column, or ``None``.

    Returns:
        The reconstructed :class:`~engrava.domain.models.provenance.ProvenanceContext`,
        or ``None``.

    """
    if raw is None:
        return None
    return ProvenanceContext.model_validate_json(raw)


def _compute_content_hash(content: str) -> str:
    """Compute the SHA-256 hex digest of *content* for ingest deduplication.

    The hash is computed over the UTF-8 encoded bytes of *content* with
    no normalization (no whitespace, casing, or unicode-form folding) so
    that "exact same content" is the well-defined semantic, and any
    deliberate formatting difference is treated as a distinct thought.

    Args:
        content: The thought content string.

    Returns:
        Lowercase hex digest of ``sha256(content.encode("utf-8"))``.

    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Metadata validation
# ----------------------------------------------------------------------

#: Soft warning threshold (bytes) for serialized ``ThoughtRecord.metadata``.
#:
#: Crossing this size emits a ``logger.warning`` to flag callers that may
#: be smuggling structured payloads through metadata where the ``content``
#: field would be the appropriate place.
_METADATA_WARN_BYTES = 4 * 1024

#: Hard rejection threshold (bytes) for serialized ``ThoughtRecord.metadata``.
#:
#: Crossing this size raises ``ValueError`` outright.  SQLite TEXT can
#: store much larger values, but query latency on the JSON1 ``json_extract``
#: paths used by downstream filtering degrades meaningfully past this
#: scale, so the limit is set well below SQLite's hard cap.
_METADATA_REJECT_BYTES = 64 * 1024


def _validate_metadata_value(value: MetadataValue, key_path: str) -> None:
    """Recursively validate a metadata value's structure.

    Walks nested ``dict[str, MetadataValue]`` namespaces and rejects
    anything that is not a scalar leaf (``str``, ``int``, ``float``,
    ``bool``, ``None``) or another mapping with string keys.  The
    ``key_path`` argument is dotted so error messages point at the
    offending location inside nested namespaces (e.g.
    ``source.tags``).

    Args:
        value: Candidate value at the current key path.
        key_path: Dot-joined key path from the root for error messages.

    Raises:
        ValueError: If a non-string key, a non-scalar leaf, or a
            list/tuple/set/custom container is encountered at any depth.

    """
    if value is None or isinstance(value, (str, int, bool)):
        # None / str / int / bool (bool is an int subclass) are always valid
        # scalar leaves.
        return
    if isinstance(value, float):
        # A real float must be finite so it round-trips through JSON. NaN and
        # ±Infinity serialise (``json.dumps`` defaults to ``allow_nan=True``) to
        # the bare tokens ``NaN`` / ``Infinity`` / ``-Infinity``, which are
        # invalid JSON: SQLite's ``json_valid()`` then returns 0 and the row
        # becomes silently unmatchable by every metadata filter. Reject them at
        # the write boundary — the same finite-only rule the filter value domain
        # already enforces on the read side.
        if not math.isfinite(value):
            msg = f"metadata value at {key_path} must be a finite number, got {value!r}"
            raise ValueError(msg)
        return
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                msg = f"metadata key at {key_path} must be str, got {type(nested_key).__name__}"
                raise ValueError(msg)  # noqa: TRY004
            _validate_metadata_value(nested_value, f"{key_path}.{nested_key}")
        return
    # Lists, tuples, sets, custom objects -> reject.
    msg = (
        f"metadata value at {key_path} type {type(value).__name__} not allowed; "
        f"allowed: str, int, float, bool, None, dict[str, MetadataValue]"
    )
    raise ValueError(msg)


def _validate_metadata(metadata: dict[str, MetadataValue]) -> None:
    """Validate metadata dict structure and serialized size.

    Caller-supplied metadata must be a ``dict`` keyed by ``str`` with
    leaf values restricted to ``str | int | float | bool | None`` so the
    column can be queried directly via SQLite's JSON1 functions without
    secondary parsing.  Nested ``dict[str, MetadataValue]`` values are
    accepted (structured namespaces per ``ThoughtSource``).
    Lists, tuples, sets and custom objects are rejected at every depth.

    Note that ``bool`` is a subclass of ``int`` in Python — both are
    accepted as scalar values, and the deserialized round trip preserves
    the original type because :func:`json.dumps` and :func:`json.loads`
    distinguish them.

    Size rules:

    * Serialized size > 4 KiB  -> ``logger.warning`` (soft signal).
    * Serialized size > 64 KiB -> :class:`ValueError` (hard rejection).

    Args:
        metadata: Caller-supplied attributes to validate.

    Raises:
        ValueError: If the structure or size invariants are violated.

    """
    # ValueError is the contractual surface for every metadata-validation
    # failure — callers (and tests) catch a single exception type for both
    # shape and size violations.  TRY004 is silenced to preserve that API.
    if not isinstance(metadata, dict):
        msg = f"metadata must be dict, got {type(metadata).__name__}"
        raise ValueError(msg)  # noqa: TRY004
    for key, value in metadata.items():
        if not isinstance(key, str):
            msg = f"metadata key must be str, got {type(key).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        _validate_metadata_value(value, key)
    serialized = json.dumps(metadata, ensure_ascii=False)
    size_bytes = len(serialized.encode("utf-8"))
    if size_bytes > _METADATA_REJECT_BYTES:
        msg = (
            f"metadata serialized size {size_bytes} bytes exceeds maximum "
            f"{_METADATA_REJECT_BYTES} bytes; consider storing large "
            "payloads as `content` or external references"
        )
        raise ValueError(msg)
    if size_bytes > _METADATA_WARN_BYTES:
        logger.warning(
            "metadata size %d bytes exceeds soft limit %d bytes — consider "
            "whether structured data should be in `content` field instead",
            size_bytes,
            _METADATA_WARN_BYTES,
        )


def _validate_provenance(provenance: ProvenanceContext | None) -> None:
    """Validate a thought's optional provenance sub-model.

    Provenance is opt-in: ``None`` is the common case and passes trivially
    (the write path is byte-identical to a thought with no provenance).  When
    present, the per-field character caps and the id-list length cap are
    enforced by :class:`~engrava.domain.models.provenance.ProvenanceContext`
    itself at construction; this hook re-asserts the type contract on the
    create / update boundary, mirroring :func:`_validate_metadata` so callers
    catch a single ``ValueError`` for a malformed provenance argument.

    Provenance is an **untrusted hint** — it is captured verbatim and consulted
    for no access, ranking, or consolidation decision (see
    :class:`~engrava.domain.models.provenance.ProvenanceContext`).

    Args:
        provenance: The candidate provenance sub-model, or ``None``.

    Raises:
        ValueError: If ``provenance`` is neither ``ProvenanceContext`` nor
            ``None``.

    """
    if provenance is None:
        return
    if not isinstance(provenance, ProvenanceContext):
        msg = f"provenance must be ProvenanceContext or None, got {type(provenance).__name__}"
        raise ValueError(msg)  # noqa: TRY004


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (numpy-accelerated).

    Returns 0.0 for zero-magnitude vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0], or 0.0 if either vector has zero norm.

    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _parse_recency_now(value: str) -> datetime.datetime:
    """Parse the caller-supplied transaction-recency ``now`` instant.

    Normalises via the shared temporal helper: UTC-normalised, a naive value
    interpreted as UTC, and the **host timezone is never consulted**. A value
    that is not a valid ISO-8601 timestamp is a malformed API argument and
    raises :class:`InvalidRecencyArgumentError` at the call boundary — the
    transaction-time recency axis never falls back to a host clock.

    Args:
        value: The caller's ``recency_now`` argument (an ISO-8601 string).

    Returns:
        The parsed instant as a timezone-aware ``datetime`` in UTC.

    Raises:
        InvalidRecencyArgumentError: If ``value`` is not a valid ISO-8601
            timestamp, or is one whose instant has no UTC form within the
            supported ``datetime`` range (an aware value at the range limits,
            such as ``0001-01-01T00:00:00+01:00``). The underlying error is
            chained as ``__cause__``.

    """
    try:
        return parse_iso8601_to_utc(value)
    except (ValueError, TypeError) as exc:
        msg = f"recency_now must be an ISO-8601 timestamp, got {value!r}"
        raise InvalidRecencyArgumentError(msg) from exc
    except OverflowError as exc:
        msg = f"recency_now {value!r} has no UTC form within the supported datetime range"
        raise InvalidRecencyArgumentError(msg) from exc


def _parse_row_timestamp(value: object) -> datetime.datetime | None:
    """Parse a stored transaction-time row timestamp, tolerating bad data.

    Returns the UTC-normalised instant, or ``None`` when the stored value is
    missing (SQL ``NULL``), malformed (a legacy / imported row), or valid
    ISO-8601 with no UTC form within the supported ``datetime`` range — a value
    the core-21 upgrade leaves untouched because it cannot put it into the
    canonical UTC form. Callers map a ``None`` result to the deterministic
    minimum recency score — the row is treated as maximally old — so bad row
    data never crashes the ranking path and never triggers a host-clock read.

    Args:
        value: The raw ``updated_at`` / ``created_at`` column value.

    Returns:
        The parsed instant, or ``None`` when the value is missing, malformed or
        has no UTC form.

    """
    if not isinstance(value, str):
        return None
    try:
        return parse_iso8601_to_utc(value)
    except (ValueError, OverflowError):
        return None


def _sort_scored_descending(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Sort ``(thought_id, score)`` pairs into a deterministic total order.

    Primary key is score descending; ties are broken by canonical
    ``thought_id`` ascending. This makes the order invariant to the
    physical scan order of the underlying query (the determinism guarantee
    for the ranked retrieval path).

    Args:
        results: ``(thought_id, score)`` pairs.

    Returns:
        A new list sorted by score descending, then ``thought_id`` ascending.

    """
    return sorted(results, key=lambda item: (-item[1], item[0]))


def _normalize_collapse_key(collapse_key: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize a ``collapse_key`` argument to a validated path tuple.

    A single ``str`` becomes a one-element composite key; a sequence of
    paths is kept in order. Every path is validated against the restricted
    JSONPath grammar at **argument time** (never mid-query), reusing the
    shared path validator, so a malformed path raises before any SQL runs.

    Args:
        collapse_key: A single metadata path (``"$.session_turn"``) or an
            ordered sequence of paths forming a composite unit key
            (``["$.session_id", "$.turn_index"]``).

    Returns:
        The ordered tuple of validated paths (length ``>= 1``).

    Raises:
        InvalidFilterPathError: If any path violates the path grammar, or
            ``collapse_key`` is an empty sequence (no key to collapse on).

    """
    from engrava.domain.exceptions import InvalidFilterPathError  # noqa: PLC0415

    paths: tuple[str, ...]
    if isinstance(collapse_key, str):
        paths = (collapse_key,)
    else:
        paths = tuple(collapse_key)
        if not paths:
            # An empty composite key has no grouping identity; reject it at
            # argument time rather than silently behaving like collapse off.
            msg = "<empty collapse_key sequence>"
            raise InvalidFilterPathError(msg)
    for path in paths:
        _validate_path(path)
    return paths


def _retain_ranked_by_unit(
    ranked: list[tuple[str, float]],
    unit_keys: dict[str, tuple[object, ...] | None],
    max_per_unit: int,
) -> list[tuple[str, float]]:
    """Retain up to ``max_per_unit`` best rows per unit on an already-ranked list.

    Walks ``ranked`` top-down (it is already in the caller's deterministic
    ranking order, so the members of each unit are visited highest-ranked
    first). A row is admitted
    unless its unit key has already reached ``max_per_unit`` admitted members,
    in which case the surplus lower-ranked member is dropped. A row whose unit
    key is ``None`` (missing / malformed metadata, or a composite with any-NULL
    component) is its OWN unit and always passes through — never grouped with
    another key-less row, which would silently drop distinct rows.

    ``max_per_unit == 1`` is the single-keeper collapse (exactly the
    highest-ranked member per unit survives); ``max_per_unit > 1`` is a strict
    relaxation that keeps a unit's deeper members too — the intra-unit
    retention count is the only thing that changes, never which distinct units
    are eligible. The relative order of the surviving rows is preserved from
    ``ranked`` (already in that order), so no re-sort with a new rule is
    introduced; retention and final order both derive from that single
    ranking order.

    Args:
        ranked: ``(thought_id, score)`` pairs already in the caller's
            deterministic ranking order.
        unit_keys: Map from ``thought_id`` to its unit-key tuple, or ``None``
            for a key-less row. Missing ids are treated as ``None``.
        max_per_unit: Maximum admitted members per non-None unit (``>= 1``).

    Returns:
        The retained ``(thought_id, score)`` list, with that ranking order
        preserved.

    """
    unit_counts: dict[tuple[object, ...], int] = {}
    retained: list[tuple[str, float]] = []
    for thought_id, score in ranked:
        unit = unit_keys.get(thought_id)
        if unit is None:
            # Key-less row: its own unit, never grouped with another.
            retained.append((thought_id, score))
            continue
        admitted = unit_counts.get(unit, 0)
        if admitted >= max_per_unit:
            # Surplus lower-ranked member of an already-full unit: drop.
            continue
        unit_counts[unit] = admitted + 1
        retained.append((thought_id, score))
    return retained


def _collapse_ranked_by_unit(
    ranked: list[tuple[str, float]],
    unit_keys: dict[str, tuple[object, ...] | None],
) -> list[tuple[str, float]]:
    """Collapse an already-ranked candidate list to one best row per unit.

    The single-keeper special case of :func:`_retain_ranked_by_unit`
    (``max_per_unit=1``): the first (highest-ranked) member of each **non-None**
    unit key is the keeper; subsequent members of the same unit are dropped.
    Key-less rows (``None`` unit key) always pass through as their own unit.

    Args:
        ranked: ``(thought_id, score)`` pairs already in the caller's
            deterministic ranking order.
        unit_keys: Map from ``thought_id`` to its unit-key tuple, or ``None``
            for a key-less row. Missing ids are treated as ``None``.

    Returns:
        The collapsed ``(thought_id, score)`` list, with that ranking order
        preserved.

    """
    return _retain_ranked_by_unit(ranked, unit_keys, max_per_unit=1)


def _normalize_min_max(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize scores to ``[0, 1]`` via min-max scaling.

    Min-max encodes each score's *relative position* within this arm's score
    distribution. When all scores are identical (``hi == lo``) there is no
    distribution — and therefore no information about relative quality — so
    every score maps to the neutral midpoint ``0.5`` rather than being asserted
    at maximum confidence. A neutral midpoint (a) removes the unjustified
    top-of-arm boost a lone or all-tied match would otherwise receive in the
    fused blend, and (b) keeps a non-zero contribution, so a match found *only*
    by this arm is not demoted below the other arm's hits. ``0.5`` also avoids
    the division-by-zero the ``hi == lo`` branch guards against.

    Args:
        results: ``(thought_id, raw_score)`` pairs.

    Returns:
        ``(thought_id, normalized_score)`` pairs.

    """
    if not results:
        return []
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(tid, 0.5) for tid, _ in results]
    return [(tid, (s - lo) / (hi - lo)) for tid, s in results]

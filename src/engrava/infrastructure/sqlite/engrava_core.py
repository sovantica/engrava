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
from typing import TYPE_CHECKING, NoReturn, Self

import aiosqlite
import numpy as np
import numpy.typing as npt

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
    DerivedRecordError,
    DuplicateEdgeError,
    EmbeddingGenerationError,
    EmbeddingModelMismatchError,
    EmbeddingQueryPrefixMismatchError,
    InvalidRecencyArgumentError,
    InvalidTransitionError,
    JournalIntegrityError,
    RecencyModeConflictError,
    ReferentialIntegrityError,
    SourceThoughtNotFoundError,
    StaleDataError,
    ThoughtNotFoundError,
    VectorDimensionMismatchError,
)
from engrava.domain.models._temporal import (
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
from engrava.domain.protocols.embedding_provider import RoleAwareEmbeddingProvider
from engrava.domain.protocols.hooks import DefaultEngravaHooks, EngravaHooksProtocol
from engrava.extensions.dreaming_signals import DreamingContext
from engrava.infrastructure.sqlite.centroid import CENTROID_MODEL_NAME, compute_centroid
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
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from engrava.config import HygienePolicyConfig, MetricsConfig, SearchConfig
    from engrava.domain.manifest import ExtensionManifest
    from engrava.domain.models.filters import MetadataFilter, VisibilityQueryFilter
    from engrava.domain.models.metrics import EngravaMetrics, LatencyHistogram
    from engrava.domain.models.search import HybridSearchResult
    from engrava.domain.protocols.cycle_provider import CycleProvider
    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol
    from engrava.domain.protocols.hooks import MindQLExtension
    from engrava.extensions.dreaming import ConsolidationResult, DreamingExtension
    from engrava.extensions.vector_sqlite_vec import SqliteVecSearchBackend
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
    A text match is used only as a fallback if the extended code is unavailable.

    Args:
        exc: The raised SQLite integrity error.

    Returns:
        ``True`` for a UNIQUE/PK violation, ``False`` otherwise (a FOREIGN KEY,
        CHECK, or other integrity failure is *not* treated as a unique
        violation and re-raises upstream).

    """
    errorcode = getattr(exc, "sqlite_errorcode", None)
    if isinstance(errorcode, int):
        return errorcode in _UNIQUE_CONSTRAINT_ERRORCODES
    # Fallback only when the extended result code is unavailable. Unreachable on
    # the supported floor (Python >= 3.11 always exposes ``sqlite_errorcode``).
    return "UNIQUE" in str(exc).upper()  # pragma: no cover


class _DerivationRollbackError(Exception):
    """Internal: a per-child rollback itself failed during derivation.

    Signals that after a child persistence failure the compensating
    ``rollback()`` also raised, leaving the transaction state indeterminate (the
    failed child's pending insert/edge may still be open and could be flushed by
    a later child's commit). The derivation dispatch must therefore abort
    immediately — this exception is **non-continuable** and is never swallowed by
    the ``on_error="log"`` continue branch. The original child failure is chained
    as ``__cause__``. Private to this module; not part of the public API.

    Args:
        rollback_error: The exception raised by the failed ``rollback()``.

    """

    def __init__(self, rollback_error: BaseException) -> None:
        super().__init__(f"rollback failed after a derived-child error: {rollback_error}")


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
    return await provider.embed(text)  # type: ignore[attr-defined,no-any-return]


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
    return await provider.embed_batch(texts)  # type: ignore[attr-defined,no-any-return]


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
    return await provider.embed(text)  # type: ignore[attr-defined,no-any-return]


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

    **Single-writer scoped.** The buffer holds no lock: it is owned by one
    store instance and, like every deferred-write path in the store, assumes a
    single writer drives that instance (the documented concurrency contract).

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


class SqliteEngravaCore:
    """Core SQLite persistence backend for thought-graph CRUD.

    Uses manual SQL with parameterized queries — no ORM.

    Supports transaction-aware commit control: when ``_skip_auto_commit``
    is ``True`` (set via :meth:`suspend_auto_commit`), individual methods
    skip ``db.commit()`` so the caller manages the commit boundary.

    Subclasses can override ``_row_to_thought`` to produce extended model
    types (template method pattern).

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
            opt-in fail-fast. The thought is already committed either way, so
            this governs how loudly the missing embedding is surfaced, not
            whether the row is persisted. No effect unless ``auto_embed`` is on.
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
    ) -> None:
        self._db = db
        self._hooks: EngravaHooksProtocol = hooks or DefaultEngravaHooks()
        self._skip_auto_commit: bool = False
        # Terminal quarantine state. Set only when a per-child compensating
        # rollback in the derived-records seam did not cleanly complete (raised
        # or was cancelled), so the long-lived connection may still hold an open
        # transaction. Quarantine is terminal by construction:
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
        # Retains the detached best-effort close task so it is neither GC'd while
        # pending nor reported as an unretrieved-exception task.
        self._quarantine_close_task: asyncio.Task[None] | None = None
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
        self._embedding_model_verified: bool = False
        self._search_config: SearchConfig | None = search_config
        self._journal_enabled: bool = journal_enabled
        self._journal: JournalWriter | None = (
            JournalWriter(db, revocation=self._revocation) if journal_enabled else None
        )
        self._ttl_strategy: str = ttl_strategy
        self._ttl_check_every_n: int = ttl_check_every_n
        self._ttl_default_seconds: int | None = ttl_default_seconds
        if metrics_config is not None:
            self._metrics_config = metrics_config
        else:
            from engrava.config import MetricsConfig  # noqa: PLC0415

            self._metrics_config = MetricsConfig()
        self._latency_buffer = _LatencyRingBuffer(self._metrics_config.window_size)
        self._operation_count: int = 0
        self._manifests: tuple[ExtensionManifest, ...] = tuple(manifests)
        # Serialises the dedup ``check existing -> INSERT or UPDATE``
        # sequence so concurrent ``create_thought(deduplicate=True)`` calls
        # converge on a single row even though aiosqlite does not expose
        # row-level locking.  Acquired only on the dedup branch — the
        # legacy ``deduplicate=False`` path stays lock-free.
        self._dedup_lock: asyncio.Lock = asyncio.Lock()
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
        # The dreaming extension, wired by ``from_config`` when dreaming is
        # enabled so ``consolidate()`` can run a cycle without the caller
        # constructing it. ``None`` for a manually-built store or dreaming-off.
        self._dreaming_extension: DreamingExtension | None = None
        # Memory Hygiene (deterministic forgetting) policy. ``None`` (default)
        # or ``enabled=False`` ⇒ the forgetting loop never runs and no existing
        # read/write path changes. ``run_hygiene`` and the ``consolidate()``
        # convenience invocation both no-op when this is ``None``/disabled.
        self._hygiene_policy: HygienePolicyConfig | None = hygiene_policy
        # Derived-records extension seam. ``enabled=False`` (the default) ⇒ the
        # seam is inert and every write path is byte-identical to a store
        # without it. When enabled *and* the hooks object implements
        # ``DerivedRecordProducerProtocol``, a successful source store is
        # followed by a core-controlled, guarded, per-child persistence of the
        # producer's derived records (see ``_dispatch_derivation``).
        self._derive_gates: DeriveGates = derive_gates or DeriveGates()
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

        Returns:
            The active journal writer, or ``None``.

        """
        return self._journal

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
            )
            store._owns_connection = True

            # Wire the dreaming extension when enabled so a YAML-only user can
            # run a consolidation cycle via ``store.consolidate(...)`` without
            # constructing ``DreamingExtension`` by hand. Off by default ⇒ no
            # extension is built and dreaming never runs.
            if config.dreaming is not None and config.dreaming.enabled:
                from engrava.extensions.dreaming import (  # noqa: PLC0415
                    DreamingExtension,
                )

                store._dreaming_extension = DreamingExtension(config=config.dreaming)

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
        except Exception:
            await db.close()
            raise

        return store

    async def close(self) -> None:
        """Close the database connection if owned by this instance.

        Flushes any pending access-buffer events first (best-effort — a flush
        failure never blocks the close), then closes the connection when this
        instance owns it. No-op on the connection when it is caller-managed
        (i.e. created via the manual constructor).
        """
        if self._access_tracking_enabled:
            try:
                await self.flush_access_buffer()
            except Exception:  # noqa: BLE001
                logger.debug("access-buffer flush on close failed; counts are best-effort")
        if self._owns_connection:
            await self._db.close()

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        Returns:
            This store instance.

        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit the async context manager and close if owned.

        Args:
            *exc: Exception info (type, value, traceback).

        """
        await self.close()

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
        await backend_handlers[backend_name](embedding_dimension)

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
        from engrava.extensions.vector_sqlite_vec import (  # noqa: PLC0415
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

        Applies the full ``schema_core.sql`` (including the FTS5 virtual table
        and sync triggers) only when the database predates the migration-ladder
        floor. A database at or above the floor is upgraded incrementally
        through the ordered core-migration registry (see
        :meth:`_core_migration_steps`) up to the head version (20).

        After core schema creation or upgrade, probes for the ``thought_fts``
        table and then runs any pending extension schema migrations for each
        manifest supplied via the ``manifests`` constructor parameter.
        """
        cursor = await self._db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current_version = int(row[0]) if row else 0

        if current_version < _CORE_SCHEMA_BOOTSTRAP_FLOOR:
            # Fresh bootstrap: ``schema_core.sql`` already carries the head DDL
            # and stamps ``user_version`` itself, so no incremental step runs
            # for a brand-new database.
            schema_sql = (
                resources.files("engrava.infrastructure.sqlite")
                .joinpath("schema_core.sql")
                .read_text(encoding="utf-8")
            )
            await self._db.executescript(schema_sql)
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
            to the head version (``v19 -> v20``).

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
        )

    async def _run_pending_core_migrations(self, current_version: int) -> None:
        """Apply every pending core migration in registry order.

        Walks the ordered registry from :meth:`_core_migration_steps` and runs
        only the steps whose target version exceeds ``current_version``. Each
        step is idempotent and **verifies its own postcondition**, raising
        :class:`CoreMigrationError` (or the underlying SQLite error) before it
        returns if the migrated structure is absent. The ``user_version`` is
        stamped and committed only *after* the step returns successfully, so a
        failed or interrupted migration leaves the version at the last
        fully-applied step and the next ``ensure_schema`` retries the remaining
        tail — a failure can never mark the database current over a
        partially-migrated schema.

        Args:
            current_version: The database's current ``user_version``. It is at
                or above the bootstrap floor; the fresh-bootstrap path is
                handled by :meth:`ensure_schema` before this method is called.

        """
        for target_version, migrate in self._core_migration_steps():
            if target_version <= current_version:
                continue
            await migrate()
            await self._db.execute(f"PRAGMA user_version = {target_version}")
            await self._db.commit()

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
            # Close any implicit transaction opened by prior migration steps
            # so PRAGMA foreign_keys=OFF actually takes effect (the pragma
            # is silently ignored while a transaction is open).
            await self._db.commit()
            await self._db.execute("PRAGMA foreign_keys=OFF")
            try:
                await self._purge_orphan_children()
                if edge_exists and not edge_done:
                    await self._recreate_edge_with_fk()
                if embedding_exists and not embedding_done:
                    await self._recreate_embedding_with_fk()
                if action_exists and not action_done:
                    await self._recreate_action_with_fk()
                # Commit the recreate steps before re-enabling FK so the
                # ON pragma also lands outside a transaction.
                await self._db.commit()
            except BaseException:  # noqa: BLE001 - rollback must also cover cancellation
                # Keep the failed step retryable on this same connection. In
                # particular, never leave a committed ``*_new`` table or a
                # half-completed table swap for the next attempt to inherit.
                await self._db.rollback()
                raise
            finally:
                await self._db.execute("PRAGMA foreign_keys=ON")

        # The edge recreation drops its indexes. Ensure the required v8 index
        # also when the FK was already present, so a partial legacy schema is
        # repaired rather than failing the same postcondition on every retry.
        if edge_exists:
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_type_from "
                "ON edge(edge_type, from_thought_id)"
            )

        # Postcondition: every child table that exists now carries its FK, and
        # the edge recreate re-created ``idx_edge_type_from`` (dropped with the
        # old table), so the loop never marks a v12 database current without
        # referential integrity or the candidate-expansion index.
        for table, column in (
            ("edge", "from_thought_id"),
            ("edge", "to_thought_id"),
            ("embedding", "owner_id"),
            ("action", "source_thought_id"),
        ):
            if await self._table_exists(table):
                await self._require_fk(12, table, column)
        if await self._table_exists("edge"):
            await self._require_index(12, "idx_edge_type_from")

        # Retry model: the per-step registry resumes a failed upgrade at the
        # failed step (an improvement over the old bump-once-at-the-end ladder,
        # which re-ran the whole tail). The table swaps are atomic within the
        # recreate transaction and explicitly rolled back on failure; a
        # persisted partial legacy state is still convergent because tables
        # already carrying the exact FK are skipped and the required edge index
        # is recreated independently above.

    async def _migrate_core_v12_to_v13(self) -> None:
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
        """Return ``True`` when ``table`` is registered in ``sqlite_master``."""
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
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
        except aiosqlite.OperationalError as exc:  # pragma: no cover - defensive race guard
            if "duplicate column" not in str(exc).lower():
                raise

    async def _purge_orphan_children(self) -> None:
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

    async def _ensure_embedding_model_lock(self, model_name: str, dimension: int) -> None:
        """Lazy-lock the embedding model on first ``store_embedding()``.

        On first call (no ``embedding_model_name`` in ``_metadata``), writes
        the model name, dimension, and — only when the active provider
        applies a non-empty ``document_prefix`` — the deterministic
        fingerprint of that prefix and the ``query_prefix`` the corpus is
        built to pair with. On subsequent calls, verifies the provider
        matches the stored values.

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

        Args:
            model_name: Model identifier from the current provider.
            dimension: Vector dimensionality from the current provider.

        Raises:
            EmbeddingModelMismatchError: When the configured model, its
                dimension, or its ``document_prefix`` fingerprint differs
                from the one stored in ``_metadata``.

        """
        if self._embedding_model_verified:
            return

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
                    configured_model=self._describe_corpus_model(model_name, active_fingerprint),
                    stored_dimension=stored_dimension,
                    configured_dimension=dimension,
                )

        self._embedding_model_verified = True

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

        """
        if self._embedding_provider is None:
            return
        await self._ensure_embedding_model_lock(
            self._embedding_provider.model_name,
            self._embedding_provider.dimension,
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

        **Single-writer contract.** The store owns one connection and holds the
        deferred-commit state on the instance, so a suspended-commit window is
        not safe for a *second* writer running on the same store instance
        concurrently: another coroutine's writes would interleave into this
        block's transaction (and, under auto-embed, be affected by the batch's
        embedding deferral). Drive writes on a given store instance from one
        task at a time. This is the established contract that ``bulk_store``
        and every deferred-commit caller rely on; concurrent same-instance
        writers are unsupported (a separate connection per writer is the
        supported concurrency model).

        Note on the derived-records seam: a create issued inside this block does
        **not** auto-derive (the source is not yet durable and this block owns
        the transaction). ``bulk_store`` dispatches derivation itself, locally,
        after its batch commits; a caller writing inside its own
        ``suspend_auto_commit`` window triggers derivation via an explicit
        re-run/backfill (see :meth:`_dispatch_derivation`).

        Yields:
            None — the store operates in deferred-commit mode.

        """
        self._skip_auto_commit = True
        try:
            yield
        except Exception:
            await self._db.rollback()
            raise
        else:
            await self._db.commit()
        finally:
            self._skip_auto_commit = False

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
    async def _drain_shielded(task: asyncio.Task[None]) -> asyncio.CancelledError | None:
        """Await a shielded task to completion, returning any cancellation of us.

        The task is observed via a **single** :func:`asyncio.wait` waiter that is
        awaited under :func:`asyncio.shield` and reused across every cancellation
        of our await (no new waiter is spawned per cancellation). ``asyncio.wait``
        surfaces the task's outcome without raising it, and the shielded waiter
        keeps observing the task to the end no matter how many times *our* await
        is cancelled. A cancellation of our await is captured and returned (never
        swallowed) so the caller can honor it once the task is safely complete;
        the task's own success/failure is left on the task for the caller.

        Args:
            task: The already-scheduled task to drain to completion.

        Returns:
            The last ``CancelledError`` raised into our await, or ``None``.

        """
        waiter = asyncio.ensure_future(asyncio.wait({task}))
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
           detached** best-effort close for resource cleanup. Quarantine returns
           promptly even if that close hangs forever — safety never depends on it.
           A done-callback consumes the close outcome so a failure/cancellation
           is never an unretrieved-task warning, and the task is retained so it is
           not GC'd while pending.

        Idempotent: a second call is a no-op (already quarantined).

        Guarantee / limitation: quarantine synchronously revokes *admission* —
        every NEW operation on the store or its journal fails fast with
        :class:`ConnectionQuarantinedError`, and direct core connection access is
        terminal via the proxy — so no write/commit can flush an orphaned
        transaction, regardless of whether the physical close succeeds. It does
        NOT retract an operation already admitted before revocation: overlapping
        writes on one store are unsupported (single-writer contract), but a
        concurrent reader admitted just before revocation may complete its
        in-flight read on the pre-revocation connection — a possibly-stale read,
        never a commit.

        Args:
            reason: Human-readable cause, surfaced on every raised error.

        """
        if self._connection_quarantined:
            return
        self._connection_quarantined = True
        self._quarantine_reason = reason
        self._revocation.revoke(reason)
        real_conn = self._db
        self._db = _QuarantinedConnection(reason)  # type: ignore[assignment]  # terminal proxy
        # Detached best-effort close: schedule and return; do NOT await it, so a
        # hung close can never block quarantine (safety is already guaranteed by
        # the proxy + token). Retain the task and consume its result via callback.
        close_task = asyncio.get_running_loop().create_task(real_conn.close())
        self._quarantine_close_task = close_task
        close_task.add_done_callback(self._consume_quarantine_close)

    async def _maybe_commit(self) -> None:
        """Commit if auto-commit is not suspended.

        Fails fast with a typed error on a quarantined connection. This flag
        check is the fast path for the common commit; the hard backstop is the
        ``_QuarantinedConnection`` proxy on ``self._db`` — even a commit that
        skipped this check would raise on ``self._db.commit()``.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        if not self._skip_auto_commit:
            await self._db.commit()

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

    _CORE_UPDATE_SQL = (
        "UPDATE thought SET "
        " thought_type = ?, essence = ?, content = ?, priority = ?,"
        " lifecycle_status = ?, created_cycle = ?, updated_cycle = ?,"
        " source = ?, confidence = ?, embedding_ref = ?,"
        " source_type = ?, confirmation_count = ?,"
        " consolidated_from = ?, visibility = ?,"
        " access_count = ?, action_outcome_score = ?, last_accessed_at = ?,"
        " created_at = ?, updated_at = ?, expires_at = ?,"
        " valid_from = ?, valid_until = ?,"
        " metadata_json = ?, provenance = ?, pinned = ?, archived_at_cycle = ?,"
        " archived_at = ? "
        "WHERE thought_id = ? AND updated_cycle = ?"
    )

    def _thought_to_core_update_params(
        self,
        updated: ThoughtRecord,
        thought_id: str,
        expected_cycle: int,
    ) -> tuple[object, ...]:
        """Extract core SQL parameters for UPDATE from a ThoughtRecord.

        Mirrors :py:meth:`_thought_to_core_params` for the SET column
        ordering, including the trailing ``metadata_json`` value, so a
        round trip through ``update_thought`` preserves caller-supplied
        metadata instead of silently reverting it to the column default.

        Args:
            updated: The updated thought record.
            thought_id: UUID of the thought to update.
            expected_cycle: The OCC version guard.

        Returns:
            Tuple of parameter values for ``_CORE_UPDATE_SQL`` — SET
            columns first (in declaration order), then the WHERE
            ``thought_id`` and ``expected_cycle`` guard.

        """
        return (
            updated.thought_type.value,
            updated.essence,
            updated.content,
            updated.priority.value,
            updated.lifecycle_status.value,
            updated.created_cycle,
            updated.updated_cycle,
            updated.source,
            updated.confidence,
            updated.embedding_ref,
            updated.source_type.value,
            updated.confirmation_count,
            _encode_consolidated(updated.consolidated_from),
            updated.visibility.value,
            updated.access_count,
            updated.action_outcome_score,
            updated.last_accessed_at,
            updated.created_at,
            updated.updated_at,
            updated.expires_at,
            updated.valid_from,
            updated.valid_until,
            json.dumps(updated.metadata, ensure_ascii=False),
            _encode_provenance(updated.provenance),
            int(updated.pinned),
            updated.archived_at_cycle,
            updated.archived_at,
            thought_id,
            expected_cycle,
        )

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

        Implements the dedup-hit branch of ``create_thought``.  Persists
        the bump in SQLite, refreshes ``updated_at`` to the current UTC
        time, and returns a rebuilt frozen ``ThoughtRecord`` reflecting
        the new state (Pydantic ``model_copy`` because the model is
        ``frozen=True`` and forbids in-place mutation).

        Args:
            existing: The thought already in the database.

        Returns:
            A new ``ThoughtRecord`` instance with
            ``confirmation_count = existing.confirmation_count + 1``
            and a refreshed ``updated_at``.

        """
        new_count = existing.confirmation_count + 1
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()

        await self._db.execute(
            "UPDATE thought SET confirmation_count = ?, updated_at = ? WHERE thought_id = ?",
            (new_count, now_iso, existing.thought_id),
        )
        await self._maybe_commit()

        if self._journal is not None:
            after = existing.model_copy(
                update={
                    "confirmation_count": new_count,
                    "updated_at": now_iso,
                },
            )
            await self._journal.append(
                mutation_type="UPDATE_THOUGHT",
                target_id=existing.thought_id,
                delta={
                    "before": existing.model_dump(mode="json"),
                    "after": after.model_dump(mode="json"),
                },
            )
            return after

        return existing.model_copy(
            update={
                "confirmation_count": new_count,
                "updated_at": now_iso,
            },
        )

    async def _create_thought_with_dedup(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None,
    ) -> ThoughtRecord:
        """Lock-protected dedup branch of ``create_thought``.

        Acquires ``self._dedup_lock`` for the entire ``check existing
        → INSERT or UPDATE`` window so concurrent calls with identical
        ``content`` never race past the existence probe.  When the
        content has not been seen the call delegates back to
        ``create_thought(..., deduplicate=False)`` so the regular
        insert / journal / auto-embed pipeline runs unchanged; when it
        has been seen ``confirmation_count`` is bumped and the existing
        record returned without any additional INSERT.
        """
        async with self._dedup_lock:
            existing = await self._get_thought_by_content_hash(
                _compute_content_hash(thought.content),
            )
            if existing is not None:
                return await self._increment_confirmation(existing)
            return await self.create_thought(
                thought,
                expires_after_seconds=expires_after_seconds,
                deduplicate=False,
            )

    async def create_thought(
        self,
        thought: ThoughtRecord,
        *,
        expires_after_seconds: int | None = None,
        deduplicate: bool = False,
    ) -> ThoughtRecord:
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
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)

        if deduplicate:
            return await self._create_thought_with_dedup(
                thought,
                expires_after_seconds=expires_after_seconds,
            )

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

        await self._db.execute(self._CORE_INSERT_SQL, self._thought_to_core_params(thought))

        if self._journal is not None:
            await self._journal.append(
                mutation_type="INSERT_THOUGHT",
                target_id=thought.thought_id,
                delta={"before": None, "after": thought.model_dump(mode="json")},
            )

        await self._maybe_commit()

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
        callers otherwise write by hand. The content hash is the same
        byte-exact SHA-256 of ``content`` used by
        ``create_thought(deduplicate=True)``:

        * **Hit** — a thought with that hash already exists: it is returned
          with ``created=False``. No new row is inserted; its
          ``confirmation_count`` is bumped and ``updated_at`` refreshed,
          identical to ``create_thought(deduplicate=True)`` so the two APIs
          stay consistent.
        * **Miss** — no such thought exists: it is inserted (running the
          regular journal / auto-embed / cleanup pipeline) and returned with
          ``created=True``.

        The returned boolean is the value ``create_thought(deduplicate=True)``
        cannot give back: it tells the caller *whether it created*, so an
        idempotent "ensure this thought exists" call needs no follow-up query.

        This does not modify the matched row's mutable fields from ``thought``
        (metadata, priority, essence): a hit returns the stored record
        unchanged apart from the confirmation bump. Use :meth:`upsert_by_hash`
        when a match should adopt the incoming record's fields.

        Args:
            thought: The candidate thought. On a miss it is inserted verbatim
                (with timestamps populated); on a hit only its ``content`` was
                used, via the hash.
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
                validates before it branches).
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        # Validate up front — before the hash probe — so an invalid-metadata
        # candidate raises on a hit too, exactly as ``create_thought`` does
        # (it validates at the top, ahead of the dedup branch). ``create_thought``
        # re-validates on the miss/insert path; that is cheap and harmless.
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)
        async with self._dedup_lock:
            existing = await self._get_thought_by_content_hash(
                _compute_content_hash(thought.content),
            )
            if existing is not None:
                return await self._increment_confirmation(existing), False
            origin_token = _DERIVATION_ORIGIN.set("get_or_create")
            try:
                created = await self.create_thought(
                    thought,
                    expires_after_seconds=expires_after_seconds,
                    deduplicate=False,
                )
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
        update reuses :meth:`update_thought`, so it participates in
        optimistic-concurrency control and re-embeds when ``essence`` changed,
        exactly like any other edit. A miss delegates to :meth:`create_thought`
        (regular insert / journal / auto-embed pipeline).

        Choose :meth:`get_or_create` for "ensure it exists, don't touch it if it
        does"; choose ``upsert_by_hash`` for "make the stored thought match this
        record"; choose ``create_thought(deduplicate=True)`` for
        confirmation-counting of repeated sightings.

        Args:
            thought: The desired thought state. On a miss it is inserted
                verbatim; on a hit its mutable fields are copied onto the
                existing row (keyed by ``content``).
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
                paths).
            StaleDataError: If the matched row is modified concurrently between
                the hash probe and the in-place update.
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        # Validate up front so an invalid-metadata candidate raises consistently
        # on both the hit (update) and miss (insert) branches.
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)
        async with self._dedup_lock:
            existing = await self._get_thought_by_content_hash(
                _compute_content_hash(thought.content),
            )
            if existing is None:
                origin_token = _DERIVATION_ORIGIN.set("upsert_by_hash")
                try:
                    return await self.create_thought(
                        thought,
                        expires_after_seconds=expires_after_seconds,
                        deduplicate=False,
                    )
                finally:
                    _DERIVATION_ORIGIN.reset(origin_token)
            # Only the fields that actually differ are forwarded to
            # ``update_thought``. This keeps the update minimal (no spurious OCC
            # churn or re-embed when a field is unchanged) and, critically,
            # never re-asserts an identical ``lifecycle_status`` — ``evolve``
            # rejects same-state lifecycle transitions, so passing the stored
            # value back verbatim would raise ``InvalidTransitionError``.
            changes = {
                field: getattr(thought, field)
                for field in self._UPSERT_MUTABLE_FIELDS
                if getattr(thought, field) != getattr(existing, field)
            }
            if not changes:
                return existing
            return await self.update_thought(existing.thought_id, **changes)

    async def bulk_store(
        self,
        thoughts: list[ThoughtRecord],
        *,
        deduplicate: bool = False,
    ) -> list[ThoughtRecord]:
        """Persist many thoughts in a single all-or-nothing transaction.

        The batch analogue of :meth:`create_thought` for ingest paths that
        would otherwise loop ``create_thought`` (one commit — and, under
        auto-embed, one embedding round trip — per thought). The whole loop
        runs under :meth:`suspend_auto_commit`, so:

        * **One commit** — every row commits together when the batch finishes,
          not once per row.
        * **All-or-nothing** — if any row raises (duplicate id, metadata
          violation, an embedding failure under ``require_embedding=True``, …)
          the entire transaction is rolled back and *nothing* is persisted; the
          exception propagates. Partial batches never land.
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
        per-thought embedding to the batch). The store owns a single connection
        and is not built for a *second* writer to run on the same instance
        concurrently with a batch; issue ``bulk_store`` from one task at a time
        (matching the existing bulk-write contract).

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
                violates the shape/size invariants (whole batch rolled back).
            EmbeddingGenerationError: If batch auto-embed fails and
                ``require_embedding`` is ``True`` (whole batch rolled back).
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

        # Snapshot the ids that already exist so genuine inserts can be told
        # apart from dedup hits deterministically — by *row existence*, never by
        # instance identity. (Instance identity is unreliable: ``create_thought``
        # rebuilds the record to populate timestamps, and a dedup hit can return
        # a row whose id coincides with the submitted one.) Only rows whose id is
        # absent here — and not yet inserted earlier in this same batch — are
        # freshly inserted, and thus the ones that need embedding and are eligible
        # for derivation. Taken whenever embedding OR the derived-records seam is
        # active, so a dedup hit never derives even with auto-embed off (D5).
        pre_existing_ids: set[str] = set()
        if embed_active or derivation_active:
            pre_existing_ids = await self._existing_thought_ids()

        origin_token = _DERIVATION_ORIGIN.set("bulk_store")
        try:
            return await self._bulk_store_inner(
                thoughts,
                deduplicate=deduplicate,
                embed_active=embed_active,
                derivation_active=derivation_active,
                pre_existing_ids=pre_existing_ids,
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
        pre_existing_ids: set[str],
    ) -> list[ThoughtRecord]:
        """Run the ``bulk_store`` insert loop under the derivation-origin label.

        Extracted so the ``bulk_store`` public method stays a thin wrapper that
        sets the informational ``DeriveContext.origin`` for the batch.

        The insert loop runs under ``suspend_auto_commit`` (one transaction). Only
        genuinely-inserted records — those whose id was absent before the batch
        and not inserted earlier in it, i.e. dedup / hash hits excluded (D5) — are
        collected as ``newly_created``. **After** the batch commits and is durable
        (the ``async with`` has exited), derivation is dispatched locally, per
        newly-created record, off the batch transaction, each child its own
        guarded durable unit; a producer/child failure there can never roll back a
        committed source or child (D3/D10). There is no shared instance buffer —
        ``newly_created`` is a local variable of this call.

        Args:
            thoughts: The thoughts to persist, in order.
            deduplicate: Per-row content-hash deduplication toggle.
            embed_active: Whether auto-embed is active for this batch.
            derivation_active: Whether the derived-records seam is active (used to
                collect the newly-created records for post-commit dispatch).
            pre_existing_ids: Ids present before the batch (for embed / derivation
                new-insert selection).

        Returns:
            The persisted records in input order.

        """
        newly_created: list[ThoughtRecord] = []
        async with self.suspend_auto_commit():
            self._suppress_auto_embed = embed_active
            try:
                persisted: list[ThoughtRecord] = []
                inserted: list[ThoughtRecord] = []
                seen_before: set[str] = set(pre_existing_ids)
                for thought in thoughts:
                    record = await self.create_thought(thought, deduplicate=deduplicate)
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
        caller-held ``suspend_auto_commit`` window: the source is already durable
        (stored by a prior committed call), so there is no source-durability reason
        to defer. If a caller wraps it in an open transaction, the children simply
        join that transaction like any other write and the caller owns their
        durability. Consequently, inside such a window with
        ``DeriveGates.on_error="raise"`` a derived-child failure rolls back that
        **whole** transaction — the caller's unrelated writes included — which is
        simply ``suspend_auto_commit``'s normal atomicity (the caller who opens the
        window owns its rollback semantics), not a behaviour unique to backfill;
        the already-committed **source thought is unaffected**. Outside a suspend
        window each child commits as its own durable unit (per-child isolation).

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
        # (that master switch governs only the automatic on-store trigger, D4).
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

        When called while auto-commit is suspended it returns without dispatching:
        the source is not yet durable and derivation must never run inside a
        transaction. ``bulk_store`` instead dispatches derivation locally, per
        newly-created record, *after* its batch commits; a caller writing inside
        its own ``suspend_auto_commit`` window triggers derivation via an explicit
        re-run/backfill (ADR D8 — recoverability, not automatic recovery). A
        dedup / hash hit never reaches this method (those return before the
        dispatch call in ``create_thought``), so only genuine inserts derive (D5).

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
        if self._skip_auto_commit:
            # Not yet durable (inside a suspended-commit window). Do not dispatch
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
            except _DerivationRollbackError:
                # Non-continuable: a per-child rollback failed, so the
                # transaction state is indeterminate. Aborting the remaining
                # children is mandatory regardless of ``on_error`` — a later
                # child's commit could flush the failed child's pending work.
                # But the source is already durably committed, so this must not
                # escape a ``"log"`` policy as a caller-visible raise (fail-open,
                # ADR D10): under ``"raise"`` propagate; under ``"log"`` log at
                # error level and stop without re-raising. ``CancelledError`` is
                # handled by its own branch above and always propagates.
                if on_error == "raise":
                    raise
                logger.exception(
                    "derived-record rollback failed for source %s; aborting remaining children",
                    ctx.source_thought_id,
                )
                return _DerivationOutcome(created=created, reused=reused, skipped=skipped)
            except Exception:
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
        idempotently (D8/D10 recoverability).

        Enrichment always targets the **stored** row's own content, never the
        producer's content. On a conflict-as-reuse hit the stored row may differ
        from the producer's record (a caller can pre-create a thought whose id
        equals a derived child's deterministic id but with different content), so
        the embedding is computed from the re-read stored row — a producer-content
        vector is never attached to a row whose content differs.

        Per-child transaction isolation: a child's **row** commits as its own
        durable unit; its enrichment (embedding, ``DERIVED_FROM`` edge) completes
        afterward, so a child may be durably present yet not-yet-enriched — a
        recoverable partial state (D10), not atomic enrichment. If any step
        (insert, its journal append, embed, or edge insert) fails after writing
        but before that step's own commit, the child's uncommitted mutations are
        rolled back before the error propagates — so no half-written row/edge
        (e.g. a row whose journal append failed) can be flushed by a later
        child's commit, and the journal chain stays consistent. Earlier children
        and the source are already committed, so the rollback discards only this
        child's pending work.

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
        try:
            inserted = await self._insert_derived_row(child)
            # Re-read the stored row once: it is both the enrichment target (its
            # own content, never the producer's) and the basis for the provenance
            # identity-collision check below. On a conflict-as-reuse hit it may be
            # a foreign row a caller pre-created at this deterministic id.
            stored_row = await self._get_thought_row(child_id)
            if (
                self._auto_embed
                and self._embedding_provider is not None
                and not self._suppress_auto_embed
                and stored_row is not None
                and await self.get_embedding(child_id) is None
            ):
                # Embed the persisted row's actual content — not the producer's —
                # so a reused foreign row never receives a producer-content vector.
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
                    await self._insert_derived_edge(
                        child_id,
                        source.thought_id,
                        ctx.cycle_at_derivation,
                    )
        except BaseException as original:
            # Roll back this child's uncommitted partial (a written-but-not-yet-
            # committed insert/edge, e.g. one whose journal append raised) so it
            # cannot be flushed by a later child's commit. Earlier children and
            # the source are committed, so a clean rollback discards only this
            # child's pending work. On a clean rollback the helper returns and we
            # re-raise the original so ``_run_derivation`` applies ``on_error``
            # (log→continue / raise→abort); otherwise the helper raises.
            await self._compensate_child_rollback(original)
            raise
        if reused_foreign:
            # Foreign-identity collision: the conflict-as-reuse hit landed on a
            # pre-existing row whose content differs from this derived record, so
            # no provenance edge was attached. Surface it outside the
            # compensating-rollback path — no uncommitted mutation is pending (the
            # reuse insert aborted cleanly and any stored-row embedding already
            # committed) — so ``_run_derivation`` applies ``on_error``
            # (log→skip this child / raise→abort remaining), exactly like the
            # source-id collision.
            raise DerivedRecordError(
                source.thought_id,
                "derived record identity collides with an unrelated stored thought",
            )
        return inserted

    async def _compensate_child_rollback(self, original: BaseException) -> None:
        """Roll back a failed derived child's uncommitted partial, cancel-safely.

        Runs the compensating ``rollback`` as an independent task and awaits it
        under :func:`asyncio.shield`, so a cancellation of *our* awaiting frame
        never aborts the rollback itself. A half-completed rollback would leave
        the long-lived connection mid-transaction — an orphaned partial that a
        later operation could flush, or run atop as indeterminate state. If the
        caller is cancelled during it, the still-running shielded task is awaited
        to completion before the cancellation is honored; the cancellation is
        never swallowed and always wins over ``original``.

        The task's outcome is inspected structurally — :meth:`asyncio.Task.cancelled`
        is checked *before* :meth:`asyncio.Task.exception` (which raises on a
        cancelled task) — so a rollback failure is always detected, never let to
        throw and bypass the quarantine/precedence logic.

        Quarantine on *any* non-clean rollback: whenever the compensating
        rollback does not cleanly complete (raised **or** cancelled), the
        transaction is indeterminate regardless of whether the caller was
        cancelled, so the connection is quarantined and hard-invalidated
        (:meth:`_quarantine_connection`) before anything is surfaced. A clean
        rollback never quarantines. This closes the hole where a non-cancelled
        rollback failure (esp. under ``on_error="log"``, which logs and aborts)
        would otherwise leave the store usable for a later, orphan-flushing
        commit.

        Args:
            original: The child failure that triggered the compensating rollback.

        Raises:
            asyncio.CancelledError: When a cancellation (the caller's, or the
                rollback task's own) is the outcome. The connection is
                quarantined first if the rollback did not cleanly complete.
            _DerivationRollbackError: When, on the non-cancelled path, the
                rollback itself failed and ``original`` was not a cancellation —
                a non-continuable abort of the whole dispatch (post-quarantine).

        """
        # Run the rollback as an independent, shielded task and drain it to
        # completion, capturing any cancellation of our await.
        rollback_task: asyncio.Task[None] = asyncio.ensure_future(self._db.rollback())
        cancel_error = await self._drain_shielded(rollback_task)
        # (#2) A cancelled rollback task would make ``exception()`` raise, so
        # check ``cancelled()`` first and treat it as non-clean completion.
        rollback_cancelled = rollback_task.cancelled()
        rollback_exc = None if rollback_cancelled else rollback_task.exception()

        # (#1) Any non-clean rollback → the transaction is indeterminate →
        # quarantine before surfacing, whether or not the caller was cancelled. A
        # clean rollback never quarantines. ``_quarantine_connection`` is
        # synchronous-effect (detached close) so it cannot swallow ``cancel_error``.
        if rollback_cancelled or rollback_exc is not None:
            await self._quarantine_connection(
                f"compensating rollback did not cleanly complete: "
                f"{rollback_exc if rollback_exc is not None else 'cancelled'}",
            )

        if cancel_error is not None:
            # The caller's cancellation is the visible outcome and takes precedence.
            raise cancel_error from original
        if rollback_cancelled:
            # The rollback task itself was cancelled (its coroutine raised
            # ``CancelledError``) → propagate a cancellation, never a
            # ``_DerivationRollbackError``; ``original`` is chained as context.
            raise asyncio.CancelledError from original
        if rollback_exc is not None:
            # Non-cancelled path: the rollback itself failed → abort the dispatch
            # non-continuably (F2). A CancelledError ``original`` still wins.
            if isinstance(original, asyncio.CancelledError):
                raise original from rollback_exc
            raise _DerivationRollbackError(rollback_exc) from original
        # Clean rollback, no cancellation → the caller re-raises ``original``.

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
        SQLite (its own changes rolled back, the transaction preserved), so the
        reuse early-return leaves no pending uncommitted mutation behind.

        Args:
            child: The derived thought to persist.

        Returns:
            ``True`` when a new row was inserted, ``False`` when an existing row
            with the same content-addressed identity was reused (conflict-as-
            reuse). The caller uses this to tally created vs reused children.

        """
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
        existing edge rather than failing; SQLite aborts the conflicting
        ``INSERT`` (rolling back only its own changes), so the reuse early-return
        leaves no pending uncommitted mutation. A failure of the journal append
        after the edge insert propagates to the caller, which rolls back this
        child's pending edge insert (per-child isolation).

        Args:
            from_thought_id: The derived child id (edge origin).
            to_thought_id: The source thought id (edge target).
            cycle: The cycle to stamp on the edge.

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
                    # Derived edges never carry caller metadata, so bind the empty
                    # ``'{}'`` object literal rather than serializing the in-memory
                    # record. This provenance-only path therefore cannot smuggle
                    # unvalidated (e.g. non-finite) metadata into the column — it
                    # bypasses ``_validate_metadata`` by writing a trivially valid
                    # empty object, matching the fresh-DDL / ALTER ``DEFAULT '{}'``.
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
        under ``require_embedding=True`` — and, because this runs inside
        ``suspend_auto_commit``, that raise rolls the whole batch back.

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
            # representative, valid lookup key (not a silent drop of the others).
            # The raise rolls the entire batch back under suspend_auto_commit, so
            # every inserted row is un-done regardless of which id is named.
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
          restore windows.
        * **delete**: Physically deletes the expired thought rows (cascading
          to edges, embeddings, and actions via ON DELETE CASCADE).

        Mutations are recorded in the journal when journaling is enabled.

        Args:
            now: Optional ISO-8601 UTC timestamp to use as "current time".
                Defaults to ``datetime.now(UTC).isoformat()`` when omitted.
                Useful for deterministic testing.
            exclude_id: Optional thought ID to skip during cleanup.
                Used by auto-cleanup to protect a just-written thought.

        Returns:
            A ``CleanupResult`` with the count of processed thoughts, the
            strategy that was applied, and a UTC timestamp.

        """
        if now is None:
            now = datetime.datetime.now(datetime.UTC).isoformat()

        cursor = await self._db.execute(
            "SELECT thought_id FROM thought WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        expired_rows = await cursor.fetchall()
        expired_ids = [row["thought_id"] for row in expired_rows if row["thought_id"] != exclude_id]

        strategy = CleanupStrategy(self._ttl_strategy)

        for tid in expired_ids:
            if strategy is CleanupStrategy.ARCHIVE:
                before_row = await self._get_thought_row(tid) if self._journal is not None else None
                await self._db.execute(
                    "UPDATE thought SET lifecycle_status = ?, expires_at = NULL, "
                    "archived_at_cycle = NULL, archived_at = NULL "
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
            else:
                # DELETE strategy.
                before_row = await self._get_thought_row(tid) if self._journal is not None else None
                # Capture the embedding rowid before the cascade drops the
                # embedding row; the vec0 vector is not FK-reachable and would
                # otherwise linger as a ghost.
                vec_rowid = await self._embedding_rowid_for_thought(tid)
                await self._db.execute(
                    "DELETE FROM thought WHERE thought_id = ?",
                    (tid,),
                )
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

        if expired_ids:
            await self._maybe_commit()

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
        """Update a thought with optimistic concurrency.

        Uses ``updated_cycle`` as a version guard.

        Args:
            thought_id: UUID of the thought to update.
            **changes: Fields to update.

        Returns:
            The updated thought record.

        Raises:
            ThoughtNotFoundError: If the thought does not exist.
            StaleDataError: If the row was modified since it was read.
            ValueError: If the post-``evolve`` metadata violates the
                metadata-shape or size invariants enforced by
                :func:`_validate_metadata`, or if the post-``evolve``
                provenance is not a
                :class:`~engrava.domain.models.provenance.ProvenanceContext`
                (per :func:`_validate_provenance`).
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        current_row = await self._get_thought_row(thought_id)
        if current_row is None:
            raise ThoughtNotFoundError(thought_id)

        current = self._row_to_thought(current_row)

        expected_cycle = current.updated_cycle
        updated = current.evolve(**changes)

        _validate_metadata(updated.metadata)
        _validate_provenance(updated.provenance)

        cursor = await self._db.execute(
            self._CORE_UPDATE_SQL,
            self._thought_to_core_update_params(updated, thought_id, expected_cycle),
        )
        if cursor.rowcount == 0:
            raise StaleDataError(
                entity_type="ThoughtRecord",
                entity_id=thought_id,
                expected_version=expected_cycle,
            )

        if self._journal is not None:
            await self._journal.append(
                mutation_type="UPDATE_THOUGHT",
                target_id=thought_id,
                delta={
                    "before": current.model_dump(mode="json"),
                    "after": updated.model_dump(mode="json"),
                },
            )

        await self._maybe_commit()

        # Re-embed when essence or content changed.
        if (
            self._auto_embed
            and self._embedding_provider is not None
            and (updated.essence != current.essence or updated.content != current.content)
        ):
            await self._auto_embed_thought(updated)
            # The member's vector moved, so any REFLECTION that summarizes it
            # must re-bind to the current cluster instead of scoring on a
            # frozen centroid. Strictly on the essence/content path — a
            # metadata-only edit never reaches here, so it cannot re-bind.
            await self._rebind_consolidated_reflections(thought_id)

        await self._maybe_auto_cleanup(exclude_id=thought_id)
        return updated

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

        Args:
            thought_id: UUID of the archived thought to restore.
            current_cycle: Optional cycle to stamp as the new ``updated_cycle``;
                when omitted the ``updated_cycle`` is left unchanged.

        Returns:
            The restored thought record (``lifecycle_status`` is ``ACTIVE``).

        Raises:
            ThoughtNotFoundError: If the thought does not exist.
            InvalidTransitionError: If the thought is not currently ``ARCHIVED``.
            StaleDataError: If the row was modified since it was read.

        """
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

        expected_cycle = current.updated_cycle
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

        cursor = await self._db.execute(
            self._CORE_UPDATE_SQL,
            self._thought_to_core_update_params(updated, thought_id, expected_cycle),
        )
        if cursor.rowcount == 0:
            raise StaleDataError(
                entity_type="ThoughtRecord",
                entity_id=thought_id,
                expected_version=expected_cycle,
            )

        if self._journal is not None:
            await self._journal.append(
                mutation_type="UPDATE_THOUGHT",
                target_id=thought_id,
                delta={
                    "before": current.model_dump(mode="json"),
                    "after": updated.model_dump(mode="json"),
                },
            )
        await self._maybe_commit()
        return updated

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
            StaleDataError: If the row was modified since it was read.
            ValueError: If ``valid_until`` is not a valid ISO-8601 timestamp,
                or is earlier than the thought's existing ``valid_from`` (an
                inverted validity interval).

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
        total count is needed (e.g. for the early-stop clustering
        guard in ``DreamingExtension``).

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

        Args:
            thought_id: UUID of the thought to delete.

        Returns:
            True if the thought was deleted, False if not found.

        Raises:
            ConnectionQuarantinedError: When the connection has been quarantined.

        """
        self._ensure_connection_usable()
        before_row = await self._get_thought_row(thought_id) if self._journal is not None else None

        # Capture the embedding rowid *before* the cascade removes the row: the
        # vec0 vector table is not reachable by the embedding FK's ON DELETE
        # CASCADE, so the vector must be deleted explicitly to avoid a ghost.
        vec_rowid = await self._embedding_rowid_for_thought(thought_id)

        cursor = await self._db.execute("DELETE FROM thought WHERE thought_id = ?", (thought_id,))
        deleted = cursor.rowcount > 0

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

        await self._maybe_commit()
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
            error_text = str(exc).upper()
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
            if "FOREIGN KEY" not in error_text:
                # Preserve unrelated integrity failures, such as a duplicate
                # caller-supplied edge_id, for their own future domain contract.
                raise
            column, referenced = await self._identify_orphan_endpoint(edge)
            raise ReferentialIntegrityError(
                entity_type="edge",
                column=column,
                referenced_id=referenced,
            ) from exc

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

        Args:
            edge_id: UUID of the edge to update.
            **changes: Fields to update.

        Returns:
            The updated edge record.

        Raises:
            ValueError: If the edge does not exist, or if the merged
                ``metadata`` violates the shared metadata contract (a
                non-scalar / list value, a non-finite float, or a serialized
                size over the 64 KiB hard limit).

        """
        current_row = await self._get_edge_row(edge_id)
        if current_row is None:
            msg = f"Edge not found: {edge_id}"
            raise ValueError(msg)

        current = _row_to_edge(current_row)
        updated = type(current).model_validate({**current.model_dump(mode="json"), **changes})
        _validate_metadata(updated.metadata)

        await self._db.execute(
            "UPDATE edge SET from_thought_id = ?, to_thought_id = ?, edge_type = ?, "
            "weight = ?, created_cycle = ?, source = ?, decay_multiplier = ?, "
            "valid_from = ?, valid_until = ?, metadata_json = ? "
            "WHERE edge_id = ?",
            (
                updated.from_thought_id,
                updated.to_thought_id,
                updated.edge_type.value,
                updated.weight,
                updated.created_cycle,
                updated.source.value,
                updated.decay_multiplier,
                updated.valid_from,
                updated.valid_until,
                json.dumps(updated.metadata, ensure_ascii=False),
                edge_id,
            ),
        )

        if self._journal is not None:
            await self._journal.append(
                mutation_type="UPDATE_EDGE",
                target_id=edge_id,
                delta={
                    "before": current.model_dump(mode="json"),
                    "after": updated.model_dump(mode="json"),
                },
            )

        await self._maybe_commit()
        return updated

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
                edge's existing ``valid_from`` (an inverted validity interval).

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
    ) -> list[EdgeRecord]:
        """Retrieve edges connected to a thought.

        Args:
            thought_id: UUID of the thought.
            direction: 'IN', 'OUT', or 'BOTH'.

        Returns:
            List of matching edge records.

        """
        if direction == "OUT":
            sql = "SELECT * FROM edge WHERE from_thought_id = ?"
            params: tuple[str, ...] = (thought_id,)
        elif direction == "IN":
            sql = "SELECT * FROM edge WHERE to_thought_id = ?"
            params = (thought_id,)
        else:
            sql = "SELECT * FROM edge WHERE from_thought_id = ? OR to_thought_id = ?"
            params = (thought_id, thought_id)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by its ID.

        Args:
            edge_id: UUID of the edge to delete.

        Returns:
            True if the edge was deleted, False if not found.

        """
        before_row = await self._get_edge_row(edge_id) if self._journal is not None else None

        cursor = await self._db.execute("DELETE FROM edge WHERE edge_id = ?", (edge_id,))
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

        await self._maybe_commit()
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

        Performs a single O(1) index lookup — safe to call per cluster in
        ``_create_reflections`` regardless of store size.

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

        Auto-embed runs *after* the thought (or batch) has already committed,
        so a provider failure leaves the thought persisted but unembedded and
        invisible to vector search. This handler makes that torn write visible:
        it always emits a ``WARNING`` naming the thought id and the provider
        error, then either re-raises the provider's own exception (default,
        byte-identical to prior behaviour) or, under ``require_embedding=True``,
        raises a typed :class:`EmbeddingGenerationError` — the opt-in fail-fast.

        Args:
            thought_id: UUID of the thought whose embedding failed.
            exc: The exception raised by the embedding provider.

        Raises:
            EmbeddingGenerationError: When ``require_embedding`` is ``True``.
            Exception: The provider's original exception otherwise.

        """
        logger.warning(
            "Auto-embed failed for thought %s: %s. The thought is persisted "
            "but has no embedding and is not reachable by vector search.",
            thought_id,
            exc,
        )
        if self._require_embedding:
            raise EmbeddingGenerationError(thought_id, str(exc)) from exc
        raise exc

    async def _auto_embed_thought(self, thought: ThoughtRecord) -> None:
        """Generate and store an embedding for a thought via the provider.

        Builds the embed payload via :func:`_build_embed_input` (which drops a
        prefix-redundant ``essence`` to avoid double-counting the opening),
        embeds it via the configured provider, and persists the vector.

        A provider failure is never silent: it is routed through
        :meth:`_on_auto_embed_failure`, which logs a ``WARNING`` naming the
        thought and then re-raises the provider error (default) or a typed
        :class:`EmbeddingGenerationError` (when ``require_embedding=True``).
        The commit ordering of the caller is unchanged — the thought is already
        persisted when this runs, so on failure it remains stored but
        unembedded.

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

        await self.store_embedding(
            thought.thought_id,
            vector,
            model_name=provider.model_name,
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

        On first call, locks the embedding model in ``_metadata``.
        Subsequent calls verify the model matches the stored one.

        Args:
            thought_id: UUID of the thought that owns this embedding.
            vector: Embedding vector as a list of floats.
            model_name: Embedding model identifier.
            embedding_id: Optional explicit ID; generated if omitted.

        Returns:
            The persisted EmbeddingRecord.

        Raises:
            EmbeddingModelMismatchError: When model_name does not match
                the model already stored in ``_metadata``.

        """
        await self._ensure_embedding_model_lock(model_name, len(vector))
        eid = embedding_id or f"emb-{_uuid.uuid5(_uuid.NAMESPACE_URL, thought_id)}"
        dimension = len(vector)
        blob = struct.pack(f"{dimension}f", *vector)
        created_at = datetime.datetime.now(datetime.UTC).isoformat()

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

        # Keep the vec0 vector table in sync when a vector backend is active.
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

        """
        if self._vector_backend is not None:
            return self._vector_backend.dimension
        if self._embedding_provider is not None:
            return self._embedding_provider.dimension
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
        """Remove expired thoughts, retired REFLECTIONs, and archived rows.

        Used as a post-filter for search backends (e.g. sqlite-vec)
        that cannot natively exclude expired rows in their queries. It also
        applies the REFLECTION freshness floor: a retired REFLECTION (an
        orphan archived once its cluster left the active set) must not
        over-recall on its now-stale centroid, so any REFLECTION whose
        ``lifecycle_status`` is not ``ACTIVE`` is dropped. Other thought
        types are gated on expiry — and, unless ``include_archived`` is set,
        on the archived-exclusion rule.

        Args:
            results: List of ``(thought_id, similarity_score)`` pairs.
            include_archived: When ``False`` (the default) archived thoughts are
                added to the excluded set; when ``True`` they are retained (the
                retired-REFLECTION and expiry gates still apply).

        Returns:
            Filtered list with expired thoughts, retired REFLECTIONs, and — when
            ``include_archived`` is ``False`` — archived rows removed.

        """
        if not results:
            return results
        now = datetime.datetime.now(datetime.UTC).isoformat()
        ids = [r[0] for r in results]
        placeholders = ",".join("?" * len(ids))
        # Inverted form: this SELECT collects the rows to *exclude*, so the
        # archived rule is an ``OR`` here (drop where status IS archived), not
        # the ``!= 'ARCHIVED'`` keep-form used in the arm WHERE clauses.
        archived_exclusion = (
            "" if include_archived else f" OR lifecycle_status = '{LifecycleStatus.ARCHIVED.value}'"
        )
        cursor = await self._db.execute(
            f"SELECT thought_id FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders}) "
            f"AND ((expires_at IS NOT NULL AND expires_at <= ?) "
            f"OR (thought_type = 'REFLECTION' AND lifecycle_status != 'ACTIVE')"
            f"{archived_exclusion})",
            [*ids, now],
        )
        excluded_ids = {row["thought_id"] for row in await cursor.fetchall()}
        if not excluded_ids:
            return results
        return [(tid, score) for tid, score in results if tid not in excluded_ids]

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
            # Reached only when the store declares no dimension at the boundary
            # (no vector backend and no embedding provider), so ``search_similar``
            # cannot check the length up front and the mismatch would otherwise
            # surface as an untyped shape error from the dot product below.
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
        except OperationalError:
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
            logger.warning(
                "FTS MATCH failed for normalized query %r; retrying via "
                "sanitized bare-mode fallback",
                normalized_query,
                exc_info=True,
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
            except OperationalError:  # pragma: no cover - unreachable for real input
                # Effectively unreachable for real input — unlike the primary
                # failure above, which is a designed, counted recovery. The bare
                # path emits only sanitized, wildcard-collapsed, operator-quoted
                # OR-terms, so its MATCH is always valid FTS5 (an 80k-string
                # star-dense + punctuation/quote/operator fuzz finds no input
                # that reaches here). This is defense-in-depth against an
                # unforeseen residual only: degrade to no FTS hits rather than
                # propagate.
                logger.warning(
                    "FTS bare-mode fallback also failed for %r; returning no FTS results",
                    fallback_query,
                    exc_info=True,
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
                not a ``SELECT`` statement, or a ``FIND`` referencing an
                invalid column).

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
        # Batch query — fetch all edges touching any candidate
        chunk_size = 450
        edge_rows: list[dict[str, object]] = []
        for i in range(0, len(all_ids), chunk_size):
            chunk = all_ids[i : i + chunk_size]
            placeholders = ", ".join("?" for _ in chunk)
            sql = (
                f"SELECT from_thought_id, to_thought_id, weight "  # noqa: S608
                f"FROM edge WHERE from_thought_id IN ({placeholders}) "
                f"OR to_thought_id IN ({placeholders}) "
                f"ORDER BY weight DESC"
            )
            params = [*chunk, *chunk]
            cursor = await self._db.execute(sql, params)
            rows = await cursor.fetchall()
            edge_rows.extend(
                {"from": r["from_thought_id"], "to": r["to_thought_id"], "weight": r["weight"]}
                for r in rows
            )

        # Build adjacency: candidate → list of (neighbour_id, edge_weight)
        adjacency: dict[str, list[tuple[str, float]]] = {}
        candidate_set = set(all_ids)
        for edge in edge_rows:
            from_id = str(edge["from"])
            to_id = str(edge["to"])
            w = float(edge["weight"])  # type: ignore[arg-type]
            if from_id in candidate_set:
                adjacency.setdefault(from_id, []).append((to_id, w))
            if to_id in candidate_set:
                adjacency.setdefault(to_id, []).append((from_id, w))

        # Compute boost per candidate (prefer highest-weight neighbours)
        boosts: dict[str, float] = {}
        for cid in all_ids:
            neighbors = sorted(
                adjacency.get(cid, []),
                key=lambda x: x[1],
                reverse=True,
            )[:max_neighbors]
            boost = 0.0
            for neighbour_id, edge_weight in neighbors:
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
            - If all signals disabled → fallback to ``list_thoughts(LIMIT top_k)``.

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
        if collapse_key is not None:
            collapse_paths = _normalize_collapse_key(collapse_key)
            # Bounded candidate-pool widening: when collapsing, fragments of
            # few units can dominate the per-arm budgets, so widen each arm by
            # a small, config-backed factor to give backfill a deeper distinct
            # -unit pool. Bounded (small int) — never unbounded over-fetch.
            collapse_pool_factor = (
                self._search_config.collapse_pool_factor if self._search_config is not None else 4
            )
            fts_top_k = fts_top_k * collapse_pool_factor
            vector_top_k = vector_top_k * collapse_pool_factor

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
            # Gate the transaction axis on ``recency_active`` so a weight-0
            # ``recency_now`` stays inert on this query-less path too: passing
            # ``transaction_now=None`` when recency is inactive falls the fallback
            # through to its neutral (flat-score, updated_cycle-ordered) branch,
            # byte-identical to a query with no recency reference.
            fallback = await self._fallback_hybrid_results(
                top_k=top_k,
                current_cycle=current_cycle,
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
            # --- REFLECTION filter in fallback path ---
            if not include_reflections and fallback:
                fb_ids = [tid for tid, _ in fallback]
                placeholders = ", ".join("?" for _ in fb_ids)
                cursor = await self._db.execute(
                    f"SELECT thought_id FROM thought"  # noqa: S608
                    f" WHERE thought_type = 'REFLECTION'"
                    f" AND thought_id IN ({placeholders})",
                    fb_ids,
                )
                rows = await cursor.fetchall()
                ref_set = {str(r["thought_id"]) for r in rows}
                fallback = [(tid, s) for tid, s in fallback if tid not in ref_set]

            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)

            self._buffer_accesses([tid for tid, _ in fallback])
            return HybridSearchResult(
                results=fallback,
                backends_used=frozenset(backends_used),
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

        # --- REFLECTION filter + boost + top-K cap ---
        resolved_reflection_boost = (
            reflection_boost
            if reflection_boost is not None
            else (self._search_config.reflection_boost if self._search_config is not None else 1.0)
        )
        resolved_reflection_topk_cap = (
            self._search_config.reflection_topk_cap if self._search_config is not None else 0.3
        )
        _needs_reflection_ids = (
            not include_reflections
            or resolved_reflection_boost != 1.0
            or resolved_reflection_topk_cap < 1.0
        )
        reflection_ids: set[str] = set()
        if _needs_reflection_ids and combined:
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

        # Deterministic total order: score descending, canonical thought_id
        # ascending — invariant to dict/scan order.
        ranked = _sort_scored_descending(list(combined.items()))

        # --- De-fragmentation retention-by-unit + backfill ---
        # Runs AFTER fusion + recency/priority/graph scoring + the
        # CONSOLIDATED_FROM expansion and reflection boost, BEFORE the
        # ``[:top_k]`` truncation and BEFORE reflection_topk_cap — the same
        # locus and shape as the cap's evict-and-backfill. It touches neither
        # arm's WHERE, no score, and no candidate set: it only removes surplus
        # lower-ranked members of the same caller-defined unit so deeper
        # distinct units in ``ranked[top_k:]`` flow up into the window.
        # ``collapse_max_per_unit`` sets how many members of a unit are kept:
        # ``None`` => 1 (single-keeper collapse, byte-identical to before), an
        # integer keeps that many (deeper same-unit rows survive while the freed
        # slots still backfill distinct units).
        if collapse_paths is not None and combined:
            unit_keys = await self._fetch_collapse_unit_keys(
                thought_ids=list(combined),
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
                _kept = [(tid, s) for tid, s in final if tid not in _to_evict]
                final = _sort_scored_descending(_kept + _fill)[:top_k]
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
        import struct  # noqa: PLC0415
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

        scores: list[tuple[str, float]] = []
        for rid in reflection_ids:
            emb = embeddings_by_id.get(rid)
            if emb is None:
                scores.append((rid, 0.0))
                continue
            _dimension, _blob = emb
            vec = list(struct.unpack(f"{_dimension}f", _blob))
            v_norm = math.sqrt(sum(x * x for x in vec))
            if v_norm == 0.0:
                scores.append((rid, 0.0))
                continue
            dot = sum(a * b for a, b in zip(effective_vector, vec, strict=False))
            scores.append((rid, dot / (q_norm * v_norm)))

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
            counts are best-effort telemetry, so this is not reconciled).

        """
        if not self._access_tracking_enabled:
            return 0
        pending = self._access_buffer.drain()
        if not pending:
            return 0
        # (delta, last_seen, thought_id) — matches the UPDATE parameter order.
        params = [(delta, ts, tid) for tid, delta, ts in pending]
        await self._db.executemany(
            "UPDATE thought SET access_count = access_count + ?, "
            "last_accessed_at = ? WHERE thought_id = ?",
            params,
        )
        await self._maybe_commit()
        logger.debug(
            "flushed access buffer: %d thought(s) updated in one batch",
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

        archived_count = await self._hygiene_archive(
            would_evict, policy=policy, current_cycle=current_cycle, now=now
        )

        gc_count = 0
        if policy.auto_gc_enabled:
            gc_count = await self._hygiene_gc(policy=policy, current_cycle=current_cycle, now=now)

        if archived_count or gc_count:
            await self._maybe_commit()

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

        The ``decay_function`` hook is the third otherwise-dead hook; the hygiene
        eviction score is its **only** call-site (it is never wired into search /
        ranking / promotion). Its return is clamped to ``[0.0, 1.0]`` and a
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

        Flips each thought ``* -> ARCHIVED`` via the existing archive mechanism
        — a **direct lifecycle write**, exactly as TTL archival does in
        :meth:`cleanup_expired` (an ``UPDATE`` of ``lifecycle_status`` /
        ``expires_at``, not an ``evolve`` transition). Using the direct write is
        deliberate: it lets a ``CREATED`` thought be archived even though the
        lifecycle state machine only permits ``CREATED -> ACTIVE`` (the ADR's
        candidate set is ACTIVE **and** CREATED), matching how TTL archival flips
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
                "expires_at = NULL, archived_at_cycle = ?, archived_at = ? "
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
    ) -> int:
        """Physically delete hygiene-archived thoughts past both restore windows.

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
            The number of thoughts physically deleted.

        """
        eligible = await self._hygiene_gc_eligible(
            policy=policy, current_cycle=current_cycle, now=now
        )
        if not eligible:
            return 0

        # Retire orphan REFLECTIONs *before* any delete so a synthesis never
        # outlives its whole source cluster with a dangling edge.
        await self.retire_orphan_reflections()

        gc_count = 0
        for thought in eligible:
            before_row = await self._get_thought_row(thought.thought_id)
            if before_row is None:
                continue
            vec_rowid = await self._embedding_rowid_for_thought(thought.thought_id)
            cursor = await self._db.execute(
                "DELETE FROM thought WHERE thought_id = ?",
                (thought.thought_id,),
            )
            if cursor.rowcount <= 0:
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
        return gc_count

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

    async def consolidate(self, *, current_cycle: int | None = None) -> ConsolidationResult:
        """Run one dreaming consolidation cycle on this store.

        The invocable entry point for a store built via :meth:`from_config`
        with ``dreaming.enabled`` — it lets a YAML-only caller run consolidation
        without constructing a ``DreamingExtension`` by hand. It first flushes
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

        Args:
            action: The action record to create.

        Returns:
            The persisted action record.

        """
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
        while the status stays terminal). ``verification_status`` is not gated
        by the lifecycle state: it may be set on a non-terminal action too, but
        such an action contributes nothing to ``action_outcome_score`` (the
        aggregate counts only terminal actions), so a premature verification
        mark is harmless — it takes effect only once the action is terminal.

        A **no-op** update — every supplied field already equals the stored
        value — returns the unchanged record and is **fully side-effect-free**:
        no persisted write, no journal entry, no feedback recompute, and (since
        it writes nothing) no commit — it never flushes unrelated pending
        writes on the connection.

        On a real change the new status/verification is persisted, the
        mutation is journaled as ``UPDATE_ACTION`` (only when journaling is
        enabled), and the source thought's ``action_outcome_score`` is
        recomputed when the change is **outcome-affecting** — that is, when
        it lands a terminal status, or changes ``verification_status`` on an
        already-terminal action. A purely non-terminal move (e.g.
        ``PLANNED`` → ``EXECUTING``) is journaled but triggers no recompute.

        Args:
            action_id: UUID of the action to update.
            status: New status, or ``None`` to leave the status unchanged.
            verification_status: New verification status, or ``None`` to
                leave it unchanged.

        Returns:
            The updated (or, for a no-op, the unchanged) action record.

        Raises:
            ActionNotFoundError: If the action does not exist.
            InvalidTransitionError: If a real ``status`` change is illegal
                per the action state machine.

        """
        current = await self._get_action(action_id)
        if current is None:
            raise ActionNotFoundError(action_id)

        status_changes = status is not None and status != current.status
        verification_changes = (
            verification_status is not None and verification_status != current.verification_status
        )
        if not status_changes and not verification_changes:
            # No-op: nothing to persist, journal, or recompute.
            return current

        changes: dict[str, object] = {}
        if status_changes:
            changes["status"] = status
        if verification_changes:
            changes["verification_status"] = verification_status
        # ``evolve`` validates the status transition when ``status`` changes and
        # is a no-op validation-wise for a verification-only change.
        updated = current.evolve(**changes)

        await self._db.execute(
            "UPDATE action SET status = ?, verification_status = ? WHERE action_id = ?",
            (updated.status.value, updated.verification_status.value, action_id),
        )

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
                        "status": updated.status.value,
                        "verification_status": updated.verification_status.value,
                    },
                },
            )

        # Outcome-affecting iff the change lands a terminal status, or changes
        # verification on an already-terminal action. Because the aggregate
        # reads both status and verification, a verification change on a
        # terminal action IS outcome-affecting.
        lands_terminal = status_changes and updated.status in _TERMINAL_ACTION_STATUSES
        verifies_terminal = verification_changes and updated.status in _TERMINAL_ACTION_STATUSES
        if lands_terminal or verifies_terminal:
            await self._recompute_action_outcome(updated.source_thought_id)

        await self._maybe_commit()
        return updated

    async def _get_action(self, action_id: str) -> ActionRecord | None:
        """Fetch a single action by its ID, or ``None`` when absent.

        Args:
            action_id: UUID of the action.

        Returns:
            The action record, or ``None`` if not found.

        """
        cursor = await self._db.execute("SELECT * FROM action WHERE action_id = ?", (action_id,))
        row = await cursor.fetchone()
        return _row_to_action(row) if row is not None else None

    async def _recompute_action_outcome(self, thought_id: str) -> None:
        """Recompute and persist a thought's denormalised ``action_outcome_score``.

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
        decay_multiplier=float(decay_raw) if decay_raw else 1.0,
        valid_from=valid_from_raw,
        valid_until=valid_until_raw,
        metadata=metadata_decoded,
    )


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
            timestamp (the underlying parser error is chained as ``__cause__``).

    """
    try:
        return parse_iso8601_to_utc(value)
    except (ValueError, TypeError) as exc:
        msg = f"recency_now must be an ISO-8601 timestamp, got {value!r}"
        raise InvalidRecencyArgumentError(msg) from exc


def _parse_row_timestamp(value: object) -> datetime.datetime | None:
    """Parse a stored transaction-time row timestamp, tolerating bad data.

    Returns the UTC-normalised instant, or ``None`` when the stored value is
    missing (SQL ``NULL``) or malformed (a legacy / imported row). Callers map a
    ``None`` result to the deterministic minimum recency score — the row is
    treated as maximally old — so bad row data never crashes the ranking path
    and never triggers a host-clock read.

    Args:
        value: The raw ``updated_at`` / ``created_at`` column value.

    Returns:
        The parsed instant, or ``None`` when the value is missing or malformed.

    """
    if not isinstance(value, str):
        return None
    try:
        return parse_iso8601_to_utc(value)
    except ValueError:
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
    """Retain up to ``max_per_unit`` best rows per unit on a D8-ranked list.

    Walks ``ranked`` top-down (it is already in the D8 total order, so the
    members of each unit are visited highest-ranked first). A row is admitted
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
    ``ranked`` (already the D8 order), so no re-sort with a new rule is
    introduced; retention and final order both derive from the single D8 order.

    Args:
        ranked: ``(thought_id, score)`` pairs in D8 total order.
        unit_keys: Map from ``thought_id`` to its unit-key tuple, or ``None``
            for a key-less row. Missing ids are treated as ``None``.
        max_per_unit: Maximum admitted members per non-None unit (``>= 1``).

    Returns:
        The retained ``(thought_id, score)`` list, D8 order preserved.

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
    """Collapse a D8-ranked candidate list to one best row per unit.

    The single-keeper special case of :func:`_retain_ranked_by_unit`
    (``max_per_unit=1``): the first (highest-ranked) member of each **non-None**
    unit key is the keeper; subsequent members of the same unit are dropped.
    Key-less rows (``None`` unit key) always pass through as their own unit.

    Args:
        ranked: ``(thought_id, score)`` pairs in D8 total order.
        unit_keys: Map from ``thought_id`` to its unit-key tuple, or ``None``
            for a key-less row. Missing ids are treated as ``None``.

    Returns:
        The collapsed ``(thought_id, score)`` list, D8 order preserved.

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

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
import struct
import uuid as _uuid
from importlib import resources
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
    EmbeddingGenerationError,
    EmbeddingModelMismatchError,
    EmbeddingQueryPrefixMismatchError,
    JournalIntegrityError,
    ReferentialIntegrityError,
    StaleDataError,
    ThoughtNotFoundError,
)
from engrava.domain.models._temporal import validate_iso8601_nullable
from engrava.domain.models.action import ActionRecord
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.embedding import EmbeddingRecord
from engrava.domain.models.filters import _validate_path, compile_effective_predicate
from engrava.domain.models.journal import JournalIntegrityResult
from engrava.domain.models.provenance import ProvenanceContext
from engrava.domain.models.thought import MetadataValue, ThoughtRecord
from engrava.domain.models.ttl import CleanupResult, CleanupStrategy
from engrava.domain.protocols.embedding_provider import RoleAwareEmbeddingProvider
from engrava.domain.protocols.hooks import DefaultEngravaHooks, EngravaHooksProtocol
from engrava.extensions.dreaming_signals import DreamingContext
from engrava.infrastructure.sqlite.centroid import CENTROID_MODEL_NAME, compute_centroid
from engrava.infrastructure.sqlite.hygiene import (
    EvictionReason,
    HygieneResult,
    compute_active_hygiene_weights,
    compute_keep_score,
)
from engrava.infrastructure.sqlite.journal_writer import JournalWriter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from engrava.config import HygienePolicyConfig, MetricsConfig, SearchConfig
    from engrava.domain.manifest import ExtensionManifest
    from engrava.domain.models.filters import MetadataFilter, VisibilityQueryFilter
    from engrava.domain.models.metrics import EngravaMetrics, LatencyHistogram
    from engrava.domain.models.search import HybridSearchResult
    from engrava.domain.protocols.embedding_provider import EmbeddingProviderProtocol
    from engrava.domain.protocols.hooks import MindQLExtension
    from engrava.extensions.dreaming import ConsolidationResult, DreamingExtension
    from engrava.extensions.vector_sqlite_vec import SqliteVecSearchBackend
    from engrava.mindql.executor import MindQLResult
    from engrava.mindql.parser import MindQLQuery

logger = logging.getLogger(__name__)

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
    ) -> None:
        self._db = db
        self._hooks: EngravaHooksProtocol = hooks or DefaultEngravaHooks()
        self._skip_auto_commit: bool = False
        self._fts_available: bool = False
        self._fts_probed: bool = False
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
        self._journal: JournalWriter | None = JournalWriter(db) if journal_enabled else None
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
        # itself (its candidate scans / reflection-member resolution). Those
        # are internal machinery, not caller retrievals, so they must not feed
        # the frequency signal. Set only inside ``suppress_access_tracking``.
        self._suppress_access_tracking: bool = False
        # The dreaming extension, wired by ``from_config`` when dreaming is
        # enabled so ``consolidate()`` can run a cycle without the caller
        # constructing it. ``None`` for a manually-built store or dreaming-off.
        self._dreaming_extension: DreamingExtension | None = None
        # Memory Hygiene (deterministic forgetting) policy. ``None`` (default)
        # or ``enabled=False`` ⇒ the forgetting loop never runs and no existing
        # read/write path changes. ``run_hygiene`` and the ``consolidate()``
        # convenience invocation both no-op when this is ``None``/disabled.
        self._hygiene_policy: HygienePolicyConfig | None = hygiene_policy

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

        Returns:
            A :class:`JournalIntegrityResult` describing chain validity —
            ``valid`` plus ``entries_checked``, and on a break the
            ``first_invalid_sequence`` and ``error_message``.

        Examples:
            >>> result = await store.verify_journal()  # doctest: +SKIP
            >>> result.valid  # doctest: +SKIP
            True

        """
        journal = self._journal if self._journal is not None else JournalWriter(self._db)
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

    # ------------------------------------------------------------------
    # Factory + async context manager
    # ------------------------------------------------------------------

    @classmethod
    async def from_config(
        cls,
        config_path: str | Path,
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

    async def ensure_schema(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Create core tables if they don't already exist.

        Applies the full ``schema_core.sql`` (including FTS5 virtual
        table and sync triggers) only when the database has not already
        been bootstrapped to schema version 3+.  Databases at older
        versions are upgraded incrementally up to the current version (18).

        After core schema creation or upgrade, probes for the ``thought_fts``
        table and then runs any pending extension schema migrations for each
        manifest supplied via the ``manifests`` constructor parameter.
        """
        cursor = await self._db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current_version = int(row[0]) if row else 0

        if current_version < 2:  # noqa: PLR2004
            schema_sql = (
                resources.files("engrava.infrastructure.sqlite")
                .joinpath("schema_core.sql")
                .read_text(encoding="utf-8")
            )
            await self._db.executescript(schema_sql)
        elif current_version < 3:  # noqa: PLR2004
            await self._rebuild_fts_index()
            await self._migrate_core_v3_to_v4()
            await self._migrate_core_v4_to_v5()
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 4:  # noqa: PLR2004
            await self._migrate_core_v3_to_v4()
            await self._migrate_core_v4_to_v5()
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 5:  # noqa: PLR2004
            await self._migrate_core_v4_to_v5()
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 6:  # noqa: PLR2004
            await self._migrate_core_v5_to_v6()
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 7:  # noqa: PLR2004
            await self._migrate_core_v6_to_v7()
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 8:  # noqa: PLR2004
            await self._migrate_core_v7_to_v8()
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 9:  # noqa: PLR2004
            await self._migrate_core_v8_to_v9()
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 10:  # noqa: PLR2004
            await self._migrate_core_v9_to_v10()
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 11:  # noqa: PLR2004
            await self._migrate_core_v10_to_v11()
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 12:  # noqa: PLR2004
            await self._migrate_core_v11_to_v12()
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 13:  # noqa: PLR2004
            await self._migrate_core_v12_to_v13()
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 14:  # noqa: PLR2004
            await self._migrate_core_v13_to_v14()
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 15:  # noqa: PLR2004
            await self._migrate_core_v14_to_v15()
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 16:  # noqa: PLR2004
            await self._migrate_core_v15_to_v16()
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 17:  # noqa: PLR2004
            await self._migrate_core_v16_to_v17()
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()
        elif current_version < 18:  # noqa: PLR2004
            await self._migrate_core_v17_to_v18()
            await self._db.execute("PRAGMA user_version = 18")
            await self._db.commit()

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

    async def _migrate_core_v3_to_v4(self) -> None:
        """Add access tracking and datetime timestamp columns (core-4).

        Idempotent — safe to run on a database that already has the columns.
        Backfills ``created_at`` and ``updated_at`` with the current UTC
        time for existing rows that lack timestamps.
        """
        from sqlite3 import OperationalError  # noqa: PLC0415

        alter_statements = [
            "ALTER TABLE thought ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE thought ADD COLUMN last_accessed_at TEXT",
            "ALTER TABLE thought ADD COLUMN created_at TEXT",
            "ALTER TABLE thought ADD COLUMN updated_at TEXT",
        ]
        for stmt in alter_statements:
            with contextlib.suppress(OperationalError):
                await self._db.execute(stmt)

        now = datetime.datetime.now(datetime.UTC).isoformat()
        await self._db.execute(
            "UPDATE thought SET created_at = ?, updated_at = ? WHERE created_at IS NULL",
            (now, now),
        )
        await self._db.execute(
            "UPDATE thought SET updated_at = ? WHERE updated_at IS NULL",
            (now,),
        )

    async def _migrate_core_v4_to_v5(self) -> None:
        """Add the ``_metadata`` key/value table (core-5).

        Idempotent — uses ``CREATE TABLE IF NOT EXISTS``.
        """
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS _metadata (key TEXT PRIMARY KEY, value TEXT)"
        )

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

    async def _migrate_core_v6_to_v7(self) -> None:
        """Add ``expires_at`` column and partial index (core-7).

        Idempotent — silently skips when the column already exists.
        """
        with contextlib.suppress(Exception):  # Column may already exist.
            await self._db.execute("ALTER TABLE thought ADD COLUMN expires_at TEXT")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_expires "
            "ON thought(expires_at) WHERE expires_at IS NOT NULL"
        )

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

        Idempotent — uses ``CREATE INDEX IF NOT EXISTS``.  Silently skips
        when the ``edge`` table does not yet exist (partial-schema test
        environments or future migrations that reorder DDL).
        """
        with contextlib.suppress(Exception):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id)"
            )

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
        try:
            await self._db.execute("ALTER TABLE thought ADD COLUMN content_hash TEXT")
        except aiosqlite.OperationalError as exc:
            # SQLite emits "duplicate column name: content_hash" when the
            # column already exists; that is the idempotent re-run signal.
            if "duplicate column" not in str(exc).lower():
                raise

        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash)"
        )

    async def _migrate_core_v10_to_v11(self) -> None:
        """Add ``metadata_json`` column to ``thought`` table (core-11).

        Adds a NOT NULL ``metadata_json TEXT`` column with default
        ``'{}'`` to support structured metadata (role, lang,
        content_type, session_id, ...).  Existing rows get the
        empty-dict default — no data migration required.

        Idempotent: ``ALTER TABLE ADD COLUMN`` is wrapped in
        duplicate-column tolerance (matches ``_migrate_core_v9_to_v10``
        precedent), so re-running the migration after a partial crash
        converges on the fully-applied state.
        """
        try:
            await self._db.execute(
                "ALTER TABLE thought ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
            )
        except aiosqlite.OperationalError as exc:
            # SQLite emits "duplicate column name: metadata_json" when
            # the column already exists; that is the idempotent re-run
            # signal.
            if "duplicate column" not in str(exc).lower():
                raise

    async def _migrate_core_v11_to_v12(self) -> None:
        """Add referential integrity (FK + ON DELETE CASCADE) to child tables.

        SQLite does not support ``ALTER TABLE ADD CONSTRAINT`` so the
        FK clauses on ``edge``, ``embedding`` and ``action`` are
        introduced via the recreate-table pattern: build a new table
        with the FK declaration, copy rows over, drop the old table,
        rename the new one, and rebuild any indexes that the schema
        declared on it.

        ``PRAGMA foreign_keys=OFF`` is a documented no-op while a
        transaction is open. The dispatch chain in ``ensure_schema``
        may arrive here with prior migration writes still in an open
        implicit transaction; this helper therefore commits any pending
        work *before* toggling the pragma, runs the recreate steps,
        commits the recreations, and only then re-enables enforcement.
        Without this, the ladder path (older schema → … → v12) leaves
        FK enforcement on during the swap and the recreated tables
        fail their first ``INSERT … SELECT *`` if any unpurged orphan
        remains.

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
        edge_done = edge_exists and await self._fk_present("edge", "from_thought_id")
        embedding_done = embedding_exists and await self._fk_present("embedding", "owner_id")
        action_done = action_exists and await self._fk_present("action", "source_thought_id")
        # Tables absent from a partial bootstrap (only `thought` present) are
        # treated as "nothing to migrate" — fresh installs receive the FK
        # directly from ``schema_core.sql``.
        if (
            (edge_done or not edge_exists)
            and (embedding_done or not embedding_exists)
            and (action_done or not action_exists)
        ):
            return

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
        finally:
            await self._db.execute("PRAGMA foreign_keys=ON")

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

        Idempotent: each ``ALTER TABLE ... ADD COLUMN`` is wrapped in
        ``contextlib.suppress(Exception)`` so a re-run after the column
        already exists is a no-op, and every index uses
        ``CREATE INDEX IF NOT EXISTS``. Re-running the migration leaves
        the schema unchanged.
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
                with contextlib.suppress(Exception):  # Column may already exist.
                    await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

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
        from sqlite3 import OperationalError  # noqa: PLC0415

        if not await self._column_exists("thought", "action_outcome_score"):
            try:
                await self._db.execute("ALTER TABLE thought ADD COLUMN action_outcome_score REAL")
            except OperationalError as exc:  # pragma: no cover - defensive race guard
                if "duplicate column" not in str(exc).lower():
                    raise
        if await self._table_exists("action"):
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_source_thought ON action(source_thought_id)"
            )

    async def _migrate_core_v16_to_v17(self) -> None:
        """Add the opt-in provenance column and its identity indexes (core-17).

        Purely additive. Backs write-time provenance capture:

        * ``thought.provenance`` (nullable ``TEXT``) — a JSON document holding
          the opt-in :class:`~engrava.domain.models.provenance.ProvenanceContext`
          sub-model, or ``NULL`` when a thought carries no provenance. Added via
          ``ALTER TABLE ... ADD COLUMN``; an ``OperationalError`` naming a
          duplicate column is swallowed so a database already carrying the
          column (a partial or re-run migration) is left unchanged.
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
        from sqlite3 import OperationalError  # noqa: PLC0415

        if not await self._column_exists("thought", "provenance"):
            try:
                await self._db.execute("ALTER TABLE thought ADD COLUMN provenance TEXT")
            except OperationalError as exc:  # pragma: no cover - defensive race guard
                if "duplicate column" not in str(exc).lower():
                    raise
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_prov_session "
            "ON thought(json_extract(provenance, '$.session_id'))"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_prov_actor "
            "ON thought(json_extract(provenance, '$.actor_id'))"
        )

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
        from sqlite3 import OperationalError  # noqa: PLC0415

        if not await self._column_exists("thought", "pinned"):
            try:
                await self._db.execute(
                    "ALTER TABLE thought ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                )
            except OperationalError as exc:  # pragma: no cover - defensive race guard
                if "duplicate column" not in str(exc).lower():
                    raise
        if not await self._column_exists("thought", "archived_at_cycle"):
            try:
                await self._db.execute("ALTER TABLE thought ADD COLUMN archived_at_cycle INTEGER")
            except OperationalError as exc:  # pragma: no cover - defensive race guard
                if "duplicate column" not in str(exc).lower():
                    raise

    async def _fk_present(self, table: str, column: str) -> bool:
        """Return ``True`` when ``table`` carries an FK on ``column``."""
        cursor = await self._db.execute(f"PRAGMA foreign_key_list({table})")
        rows = await cursor.fetchall()
        return any(row["from"] == column for row in rows)

    async def _table_exists(self, table: str) -> bool:
        """Return ``True`` when ``table`` is registered in ``sqlite_master``."""
        cursor = await self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        )
        return await cursor.fetchone() is not None

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
        candidate scans and reflection-member resolution — are not caller
        retrievals and must not feed the ``frequency`` signal. Wrapping those
        reads in this block keeps them out of the access buffer. No effect when
        access tracking is disabled; restores the prior state on exit even on
        error.

        Yields:
            None — access buffering is suppressed for the duration of the block.

        """
        previous = self._suppress_access_tracking
        self._suppress_access_tracking = True
        try:
            yield
        finally:
            self._suppress_access_tracking = previous

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

    async def _maybe_commit(self) -> None:
        """Commit if auto-commit is not suspended."""
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
        " metadata_json, provenance, pinned, archived_at_cycle) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?)"
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
        " metadata_json = ?, provenance = ?, pinned = ?, archived_at_cycle = ? "
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

        """
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
        return await self._hooks.on_store(thought)

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

        """
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
            created = await self.create_thought(
                thought,
                expires_after_seconds=expires_after_seconds,
                deduplicate=False,
            )
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

        """
        # Validate up front so an invalid-metadata candidate raises consistently
        # on both the hit (update) and miss (insert) branches.
        _validate_metadata(thought.metadata)
        _validate_provenance(thought.provenance)
        async with self._dedup_lock:
            existing = await self._get_thought_by_content_hash(
                _compute_content_hash(thought.content),
            )
            if existing is None:
                return await self.create_thought(
                    thought,
                    expires_after_seconds=expires_after_seconds,
                    deduplicate=False,
                )
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

        """
        if not thoughts:
            return []

        embed_active = self._auto_embed and self._embedding_provider is not None

        # Snapshot the ids that already exist so genuine inserts can be told
        # apart from dedup hits deterministically — by *row existence*, never by
        # instance identity. (Instance identity is unreliable: ``create_thought``
        # rebuilds the record to populate timestamps, and a dedup hit can return
        # a row whose id coincides with the submitted one.) Only rows whose id is
        # absent here — and not yet inserted earlier in this same batch — are
        # freshly inserted and thus need embedding.
        pre_existing_ids: set[str] = set()
        if embed_active:
            pre_existing_ids = await self._existing_thought_ids()

        async with self.suspend_auto_commit():
            self._suppress_auto_embed = embed_active
            try:
                persisted: list[ThoughtRecord] = []
                inserted: list[ThoughtRecord] = []
                seen_before: set[str] = set(pre_existing_ids)
                for thought in thoughts:
                    record = await self.create_thought(thought, deduplicate=deduplicate)
                    persisted.append(record)
                    if embed_active and record.thought_id not in seen_before:
                        inserted.append(record)
                    seen_before.add(record.thought_id)
                if embed_active and inserted:
                    await self._batch_embed_thoughts(inserted)
            finally:
                self._suppress_auto_embed = False
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
        filters: MetadataFilter | None = None,
        visibility: VisibilityQueryFilter | None = None,
        collapse_key: str | Sequence[str] | None = None,
        collapse_max_per_unit: int | None = None,
    ) -> HybridSearchResult:
        """Retrieve thoughts relevant to a query with one call.

        Ergonomic shorthand over :meth:`search_hybrid` for the common
        retrieval case: the query text is passed straight through with the
        given ``top_k`` and ``current_cycle``.

        When ``current_cycle`` is ``None`` the recency signal is inactive
        (see ``search_hybrid``). A store that holds more than
        ``_RECENCY_NUDGE_THRESHOLD`` thoughts and recalls without a cycle emits
        a single DEBUG-level breadcrumb on the module logger — once per store
        instance — pointing out that passing ``current_cycle`` would let recent
        thoughts rank higher. It is never a warning and never repeats.

        Args:
            query: Natural-language text to search for.
            top_k: Maximum number of results to return.
            current_cycle: Current cognitive cycle. When provided, the recency
                signal is blended into ranking; when ``None``, recency is
                skipped.
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

        Returns:
            A ``HybridSearchResult`` with the ranked matches and the set of
            backends that contributed.

        """
        if current_cycle is None and not self._recency_nudge_emitted:
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
            filters=filters,
            visibility=visibility,
            collapse_key=collapse_key,
            collapse_max_per_unit=collapse_max_per_unit,
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
          ``expires_at`` so the thought is no longer subject to TTL.
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
                    "UPDATE thought SET lifecycle_status = ?, expires_at = NULL "
                    "WHERE thought_id = ?",
                    (LifecycleStatus.ARCHIVED.value, tid),
                )
                if self._journal is not None and before_row is not None:
                    before = self._row_to_thought(before_row)
                    after = before.evolve(
                        lifecycle_status=LifecycleStatus.ARCHIVED.value,
                        expires_at=None,
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

        """
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

        """
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
            ValueError: If ``valid_until`` is not a valid ISO-8601 timestamp.

        """
        normalized = validate_iso8601_nullable(valid_until)
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

        """
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
            ReferentialIntegrityError: When ``from_thought_id`` or
                ``to_thought_id`` does not match any persisted thought.

        """
        try:
            await self._db.execute(
                "INSERT INTO edge "
                "(edge_id, from_thought_id, to_thought_id, edge_type, weight, "
                " created_cycle, source, decay_multiplier, valid_from, valid_until) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )
        except aiosqlite.IntegrityError as exc:
            if "FOREIGN KEY" not in str(exc).upper():
                # Surface UNIQUE / NOT NULL violations unchanged — only
                # FK violations get the domain wrapper.
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
            ValueError: If the edge does not exist.

        """
        current_row = await self._get_edge_row(edge_id)
        if current_row is None:
            msg = f"Edge not found: {edge_id}"
            raise ValueError(msg)

        current = _row_to_edge(current_row)
        updated = type(current).model_validate({**current.model_dump(mode="json"), **changes})

        await self._db.execute(
            "UPDATE edge SET from_thought_id = ?, to_thought_id = ?, edge_type = ?, "
            "weight = ?, created_cycle = ?, source = ?, decay_multiplier = ?, "
            "valid_from = ?, valid_until = ? "
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
            ValueError: If the edge does not exist, or ``valid_until`` is not
                a valid ISO-8601 timestamp.

        """
        normalized = validate_iso8601_nullable(valid_until)
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
        limit: int = 5000,
    ) -> list[EdgeRecord]:
        """List edges matching optional filters.

        Args:
            edge_type: If given, restrict to this edge type.
            source: If given, restrict to this knowledge source.
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

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
        *,
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
            filtered = await self._filter_expired_results(results)
            filtered = _sort_scored_descending(filtered)[:top_k]
            await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
            return filtered
        results = await self._search_similar_numpy(
            query_vector,
            top_k,
            threshold,
            _filter_clause=_filter_clause,
        )
        await self._record_search_latency((_time.perf_counter() - _t_start) * 1000)
        return results

    async def _filter_expired_results(
        self,
        results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Remove expired thoughts and retired REFLECTIONs from results.

        Used as a post-filter for search backends (e.g. sqlite-vec)
        that cannot natively exclude expired rows in their queries. It also
        applies the REFLECTION freshness floor: a retired REFLECTION (an
        orphan archived once its cluster left the active set) must not
        over-recall on its now-stale centroid, so any REFLECTION whose
        ``lifecycle_status`` is not ``ACTIVE`` is dropped. Other thought
        types are gated on expiry only, exactly as before.

        Args:
            results: List of ``(thought_id, similarity_score)`` pairs.

        Returns:
            Filtered list with expired thoughts and retired REFLECTIONs
            removed.

        """
        if not results:
            return results
        now = datetime.datetime.now(datetime.UTC).isoformat()
        ids = [r[0] for r in results]
        placeholders = ",".join("?" * len(ids))
        cursor = await self._db.execute(
            f"SELECT thought_id FROM thought "  # noqa: S608
            f"WHERE thought_id IN ({placeholders}) "
            f"AND ((expires_at IS NOT NULL AND expires_at <= ?) "
            f"OR (thought_type = 'REFLECTION' AND lifecycle_status != 'ACTIVE'))",
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
            "AND NOT (t.thought_type = 'REFLECTION' AND t.lifecycle_status != 'ACTIVE') "
            f"{filter_sql}",
            (datetime.datetime.now(datetime.UTC).isoformat(), *filter_params),
        )
        rows = list(await cursor.fetchall())
        if not rows:
            return []

        query_arr = np.asarray(query_vector, dtype=np.float64)
        q_norm = float(np.linalg.norm(query_arr))
        if q_norm == 0.0:
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

        # bm25() returns negative values; negate so higher = more relevant.
        sql = (
            "SELECT t.thought_id, -bm25(thought_fts) AS score "  # noqa: S608
            "FROM thought_fts "
            "JOIN thought t ON t.rowid = thought_fts.rowid "
            "WHERE thought_fts MATCH ? "
            "AND (t.expires_at IS NULL OR t.expires_at > ?) "
            # Freshness floor: retired REFLECTIONs are excluded so a stale
            # synthesis does not out-rank fresh relevant thoughts.
            "AND NOT (t.thought_type = 'REFLECTION' AND t.lifecycle_status != 'ACTIVE') "
            f"{filter_sql}"
            # Deterministic total order: BM25 first, then canonical thought_id.
            "ORDER BY score DESC, t.thought_id ASC "
            "LIMIT ?"
        )
        try:
            cursor = await self._db.execute(
                sql,
                (
                    normalized_query,
                    datetime.datetime.now(datetime.UTC).isoformat(),
                    *filter_params,
                    top_k,
                ),
            )
            rows = await cursor.fetchall()
        except OperationalError:
            # Defense in depth: a residual malformed FTS5 expression must never
            # propagate to the caller and break an otherwise-serviceable search
            # (e.g. the vector arm of a hybrid query). Degrade to no FTS hits.
            logger.warning(
                "FTS MATCH failed for normalized query %r; returning no FTS results",
                normalized_query,
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
        _filter_clause: tuple[str, list[object]] | None = None,
    ) -> frozenset[str]:
        """Return the subset of ``candidate_ids`` whose thought_type is OBSERVATION.

        Used by ``_expand_via_consolidated_from`` to strip non-factual
        targets (TASK, REFLECTION, …) from the expansion pool before
        propagating scores.

        When a metadata predicate is active it is re-applied here too: the
        CONSOLIDATED_FROM expansion pulls brand-new OBSERVATION rows that
        never passed an arm's ``WHERE``, so without this an out-of-filter
        OBSERVATION could be injected into the result set. Re-applying the
        same effective predicate keeps the eligibility invariant on the
        expansion path.

        Args:
            candidate_ids: Unfiltered list of target thought IDs.
            _filter_clause: Internal. A compiled ``(sql_fragment, params)``
                metadata predicate (referencing the bare ``metadata_json``
                column) injected into the ``WHERE``.

        Returns:
            Frozenset containing only IDs of OBSERVATION-type thoughts that
            also satisfy the active filter. Empty frozenset when
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
        placeholders = ", ".join("?" for _ in unique)
        cursor = await self._db.execute(
            f"SELECT thought_id FROM thought"  # noqa: S608
            f" WHERE thought_type = 'OBSERVATION'"
            f" AND thought_id IN ({placeholders})"
            f"{filter_sql}",
            [*unique, *filter_params],
        )
        rows = await cursor.fetchall()
        return frozenset(str(r["thought_id"]) for r in rows)

    async def _fallback_hybrid_results(
        self,
        *,
        top_k: int,
        current_cycle: int | None,
        recency_half_life: int,
    ) -> list[tuple[str, float]]:
        """Fallback results when neither FTS nor vector search is usable."""
        thoughts = await self.list_thoughts(limit=top_k)
        # Apply the REFLECTION freshness floor consistently with the FTS and
        # vector paths: a retired REFLECTION must not surface here either.
        thoughts = [
            thought
            for thought in thoughts
            if not (
                thought.thought_type == ThoughtType.REFLECTION
                and thought.lifecycle_status != LifecycleStatus.ACTIVE
            )
        ]
        if current_cycle is None:
            return [(thought.thought_id, 0.0) for thought in thoughts]

        decay_rate = math.log(2) / recency_half_life
        ranked = [
            (
                thought.thought_id,
                math.exp(-decay_rate * max(current_cycle - thought.updated_cycle, 0)),
            )
            for thought in thoughts
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    async def _resolve_hybrid_state(
        self,
        *,
        query_text: str,
        query_vector: list[float] | None,
        current_cycle: int | None,
        recency_weight: float,
    ) -> tuple[bool, list[float] | None, bool]:
        """Resolve active hybrid-search components for the current query."""
        if not self._fts_probed:
            await self._probe_fts()

        fts_active = bool(self._fts_available and query_text and query_text.strip())

        effective_vector = query_vector
        if effective_vector is None and self._embedding_provider is not None and query_text.strip():
            await self._ensure_query_prefix_pairs()
            effective_vector = await _embed_query(self._embedding_provider, query_text)

        recency_active = current_cycle is not None and recency_weight > 0.0
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

        Graceful degradation:
            - If FTS5 unavailable or ``query_text`` empty → FTS skipped.
            - If ``query_vector`` is ``None`` and no provider → vector skipped.
            - If ``current_cycle`` is ``None`` → recency skipped.
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
            current_cycle: Current cycle number for recency calculation.
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

        Returns:
            ``HybridSearchResult`` with ranked results and diagnostics. Tied
            scores are ordered by canonical ``thought_id`` ascending, giving
            a deterministic total order regardless of ``filters``.

        """
        import time as _time  # noqa: PLC0415

        from engrava.domain.models.search import HybridSearchResult  # noqa: PLC0415

        _t_start = _time.perf_counter()

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
            fallback = await self._fallback_hybrid_results(
                top_k=top_k,
                current_cycle=current_cycle,
                recency_half_life=resolved_recency_half_life,
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

        # --- Recency signal ---
        if recency_active:
            backends_used.add("recency")
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
        configured, all REFLECTION thoughts are returned unranked.

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

        # Resolve effective query vector (auto-embed if provider available)
        effective_vector = query_vector
        if effective_vector is None and self._embedding_provider is not None and query_text.strip():
            await self._ensure_query_prefix_pairs()
            effective_vector = await _embed_query(self._embedding_provider, query_text)

        # Fetch all REFLECTION thought IDs directly — complete, no pagination
        # gap. Retired REFLECTIONs (lifecycle != ACTIVE) are excluded by the
        # same freshness floor the similarity/hybrid paths apply.
        cursor = await self._db.execute(
            "SELECT thought_id FROM thought "
            "WHERE thought_type = 'REFLECTION' AND lifecycle_status = 'ACTIVE'"
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

        scores.sort(key=lambda x: x[1], reverse=True)
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
        if not self._access_tracking_enabled or self._suppress_access_tracking or not thought_ids:
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

    async def run_hygiene(self, *, current_cycle: int) -> HygieneResult:
        """Run one Memory Hygiene pass — archive cold/low-value thoughts.

        A standalone, deterministic, no-LLM forgetting pass: it scores every
        eligible thought with a keep-score (the dreaming signal library under
        the hygiene weight vector, with active-signal redistribution), multiplies
        by the ``decay_function`` hook, and **archives** — reversibly — the
        thoughts whose eviction-score falls below ``eviction_threshold`` and that
        are not protected. When ``auto_gc_enabled`` it then physically
        garbage-collects previously-archived thoughts past the restore window.

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
        * **Dry run.** When ``dry_run`` is set nothing is mutated and nothing is
          journaled; the would-evict set is returned for preview.

        This is cognitive hygiene, not compliance deletion: GC is best-effort,
        cycle-based, and opt-in — it offers no deletion guarantee, legal hold,
        or erasure receipt.

        Args:
            current_cycle: The current cognitive cycle number, driving the
                cycle-based recency / staleness keep-signals and the GC restore
                window.

        Returns:
            A :class:`~engrava.infrastructure.sqlite.hygiene.HygieneResult` with
            the archived / GC'd counts, the number of candidates evaluated, the
            ``dry_run`` flag, the would-evict preview (under ``dry_run``), and the
            signals that were flat this run.

        Raises:
            RuntimeError: When no hygiene policy is configured on this store
                (built without ``hygiene_policy`` / ``hygiene_policy`` is
                ``None``) — there is nothing to run.

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

        candidates = await self._hygiene_candidates()
        ctx = DreamingContext(current_cycle=current_cycle, total_thoughts=len(candidates))
        active_weights, flat_signals = compute_active_hygiene_weights(
            policy.signal_weights,
            candidates,
            current_cycle=current_cycle,
            access_tracking_enabled=self._access_tracking_enabled,
        )
        has_active_signal = any(weight > 0.0 for weight in active_weights.values())

        # All-flat fail-safe: an uninformative keep-score must never drive
        # eviction, so archive nothing (but a GC stage may still reap already
        # hygiene-archived thoughts whose restore window has elapsed).
        would_evict: list[EvictionReason] = []
        if has_active_signal:
            would_evict = self._select_archive_candidates(
                candidates,
                ctx=ctx,
                active_weights=active_weights,
                policy=policy,
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
            would_evict, policy=policy, current_cycle=current_cycle
        )

        gc_count = 0
        if policy.auto_gc_enabled:
            gc_count = await self._hygiene_gc(policy=policy, current_cycle=current_cycle)

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

        Already-ARCHIVED / terminal thoughts are outside the candidate set (they
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

        Protected thoughts (``pinned`` or a priority in
        ``protected_priorities``) are excluded up front and never scored into the
        archive set.

        Args:
            candidates: The candidate pool.
            ctx: The scoring context.
            active_weights: The redistributed per-signal weights for this run.
            policy: The active hygiene policy.
            decay_multipliers: Per-thought clamped decay multipliers.

        Returns:
            The ordered, capped list of :class:`EvictionReason` for the thoughts
            to archive.

        """
        scored: list[tuple[float, int, str, EvictionReason]] = []
        for thought in candidates:
            if _hygiene_protected(thought, policy):
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
        ``archived_at_cycle = current_cycle`` and clears ``expires_at`` so the
        thought is no longer subject to TTL.

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
                re-check (a thought pinned / re-prioritised after selection).
            current_cycle: The cycle stamped into ``archived_at_cycle``.

        Returns:
            The number of thoughts actually archived.

        """
        archived = 0
        for reason in to_archive:
            before_row = await self._get_thought_row(reason.thought_id)
            if before_row is None:
                continue
            before = self._row_to_thought(before_row)
            if _hygiene_protected(before, policy) or before.lifecycle_status not in (
                LifecycleStatus.ACTIVE,
                LifecycleStatus.CREATED,
            ):
                # Early time-of-check guard: a thought pinned, raised to a protected
                # priority, or already transitioned between selection and here is
                # skipped. The UPDATE below re-asserts the same predicate atomically.
                continue
            # Predicate-guarded write: the WHERE re-checks candidate lifecycle +
            # unprotected at write time, so even a pin / re-prioritise landing
            # between the check above and this UPDATE cannot archive a now-protected
            # thought (closes the TOCTOU fully; ``rowcount == 0`` ⇒ raced, skip).
            update_params: list[object] = [
                LifecycleStatus.ARCHIVED.value,
                current_cycle,
                reason.thought_id,
                LifecycleStatus.ACTIVE.value,
                LifecycleStatus.CREATED.value,
            ]
            priority_guard = ""
            if policy.protected_priorities:
                placeholders = ", ".join("?" for _ in policy.protected_priorities)
                priority_guard = f" AND priority NOT IN ({placeholders})"
                update_params.extend(policy.protected_priorities)
            cursor = await self._db.execute(
                "UPDATE thought SET lifecycle_status = ?, "  # noqa: S608 - interpolation is only ``?`` placeholders
                "expires_at = NULL, archived_at_cycle = ? "
                "WHERE thought_id = ? AND lifecycle_status IN (?, ?) AND pinned = 0"
                + priority_guard,
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
    ) -> int:
        """Physically delete hygiene-archived thoughts past the restore window.

        Stage 2 — runs only when ``auto_gc_enabled``. A thought is GC-eligible
        only when it was archived **by hygiene** (``archived_at_cycle IS NOT
        NULL``), its restore window has elapsed
        (``current_cycle - archived_at_cycle >= gc_min_archive_age_cycles``), and
        it is not protected (``pinned`` or a protected priority). The eligible
        set is ordered ``archived_at_cycle ASC, thought_id ASC`` (oldest-archived
        first) and truncated to ``max_evictions_per_run``.

        Deletion order per thought is **orphan-reflection sweep -> cascade delete
        -> vec0 vector purge**: the sweep retires any REFLECTION whose entire
        source cluster would become non-live so no dangling ``CONSOLIDATED_FROM``
        synthesis is left, the cascade drops FK-reachable edges / embeddings /
        actions, and the vec0 vector (outside the FK) is purged explicitly. The
        delete is recorded as an ordinary ``DELETE_THOUGHT`` journal entry.

        Args:
            policy: The active hygiene policy (window, cap, protected priorities).
            current_cycle: The current cycle (drives the window check).

        Returns:
            The number of thoughts physically deleted.

        """
        eligible = await self._hygiene_gc_eligible(policy=policy, current_cycle=current_cycle)
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
                        },
                    },
                )
        return gc_count

    async def _hygiene_gc_eligible(
        self,
        *,
        policy: HygienePolicyConfig,
        current_cycle: int,
    ) -> list[ThoughtRecord]:
        """Resolve the deterministic, capped GC-eligible set.

        Selects ARCHIVED thoughts that hygiene archived (``archived_at_cycle IS
        NOT NULL``) whose restore window has elapsed, excluding protected
        thoughts, ordered ``archived_at_cycle ASC, thought_id ASC`` and capped at
        ``max_evictions_per_run``. The window and ordering are computed in SQL off
        the explicit ``archived_at_cycle`` column so a thought archived by any
        other path (its ``archived_at_cycle`` is ``NULL``) is structurally
        excluded.

        Args:
            policy: The active hygiene policy.
            current_cycle: The current cycle.

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

    async def consolidate(self, *, current_cycle: int) -> ConsolidationResult:
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

        Returns:
            The :class:`ConsolidationResult` for the run.

        Raises:
            RuntimeError: When dreaming is not enabled/wired on this store
                (built manually, or ``dreaming.enabled`` is false) — there is
                no extension to run.

        """
        if self._dreaming_extension is None:
            msg = (
                "consolidate() requires dreaming to be enabled: build the store via "
                "from_config with extensions.dreaming.enabled = true"
            )
            raise RuntimeError(msg)
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
    )


def _query_is_expert_syntax(query: str) -> bool:
    """Return ``True`` when a query should be parsed as expert FTS5 syntax.

    A query is expert syntax when it contains any of:

    * a quoted phrase (any ``"``),
    * a standalone uppercase boolean operator (``AND``/``OR``/``NOT``), or
    * a whitelisted column filter (``essence:``/``content:``).

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
    if '"' in query:
        return True
    for token in query.split():
        if token in _FTS_BOOLEAN_OPERATORS:
            return True
        if _FTS_FIELD_FILTER_RE.match(token.lstrip("(")):
            return True
    return False


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


def _sanitize_fts_bare_token(raw: str) -> list[str]:
    """Split an unquoted bare token into safe FTS5 fragments.

    Unsafe characters become fragment boundaries rather than being deleted, so
    a contraction or clitic such as ``sister's`` splits into ``["sister", "s"]``
    (which the ``unicode61`` tokenizer also produced at index time) instead of
    merging into an unindexed ``sisters``.

    Args:
        raw: A single unquoted token, already paren-stripped.

    Returns:
        A list of non-empty safe fragments, in order. May be empty when the
        token holds no indexable characters.

    """
    stripped = _strip_fts_boundary_punctuation(raw)
    split = _FTS_UNSAFE_CHAR_RE.sub(" ", stripped)
    return [fragment for fragment in split.split() if fragment]


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

    terms = [_format_fts_bare_fragment(fragment) for fragment in fragments]
    if expert:
        # Expert mode keeps each original token as one term, re-attaching any
        # parentheses the caller used for grouping.
        terms[0] = f"{leading}{terms[0]}"
        terms[-1] = f"{terms[-1]}{trailing}"
    return terms


def _format_fts_bare_fragment(fragment: str) -> str:
    """Format a single sanitized fragment as an FTS5 term.

    Preserves a trailing ``*`` prefix marker and quotes hyphenated identifiers
    so FTS5 does not read the hyphen as a column/operator.

    Args:
        fragment: A safe fragment containing only word characters, ``-`` or a
            trailing ``*``.

    Returns:
        The fragment rewritten as a valid FTS5 term.

    """
    suffix = ""
    if fragment.endswith("*"):
        fragment = fragment[:-1]
        suffix = "*"
    if "-" in fragment:
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
    if value is None or isinstance(value, (str, int, float, bool)):
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

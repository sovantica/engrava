"""OFF / ON dreaming evaluator for the synthetic benchmark.

The evaluator builds a fresh :class:`engrava.SqliteEngravaCore`
backed by an in-memory SQLite database, ingests every turn of every
conversation as a :class:`engrava.ThoughtRecord` (carrying
self-anchored ``perspective`` / ``source.is_self`` metadata),
optionally invokes :meth:`engrava.DreamingExtension.run_consolidation`
once per conversation, then issues the recall queries via
:meth:`engrava.SqliteEngravaCore.search_hybrid`.  Scoring is pure
deterministic — no LLM judge:

* ``recall@K`` — a question hits when any of the
  ``expected_fact_ids`` resolve to a thought that appears in the
  top-K retrieved IDs.
* ``substring_match`` — a question hits when any of the
  ``expected_substrings`` appears in the content of any top-K
  retrieved thought.

The aggregator reports a global ``recall@K`` plus per-scenario and
per-difficulty breakdowns so the OFF vs ON delta can be inspected
across the dimensions the scenario library exposes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiosqlite

from engrava.benchmarks.synthetic.generate import (
    SyntheticConversation,
    SyntheticQuestion,
    SyntheticTurn,
)
from engrava.config import DreamingConfig, DreamingGates, SearchConfig
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from engrava.domain.protocols.embedding_provider import (
        EmbeddingProviderProtocol,
    )

__all__ = [
    "EvaluationResult",
    "PerBreakdown",
    "evaluate_run",
    "measure_synthesis_coverage",
    "resolve_embedding_provider_or_exit",
    "run_evaluation",
]


_DEFAULT_TOP_K = 5
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerBreakdown:
    """Aggregated metrics for a single bucket (scenario or difficulty).

    Attributes:
        name: Bucket key (scenario name or difficulty tag).
        question_count: How many questions fall into this bucket.
        recall_at_k: Fraction of bucket questions whose
            ``expected_fact_ids`` intersect the top-K retrieved IDs.
        substring_match_rate: Fraction of bucket questions where any
            ``expected_substring`` appears in any retrieved
            thought's content.

    """

    name: str
    question_count: int
    recall_at_k: float
    substring_match_rate: float


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation outcome with per-bucket breakdowns.

    Attributes:
        dreaming_enabled: Whether the run had ``DreamingExtension``
            wired in.  Mirrors the input flag; consumers can pair an
            OFF result with an ON result for delta reporting.
        top_k: The retrieval top-K used.  Reported alongside the
            recall metric so downstream consumers do not have to
            re-derive it from the runner's CLI args.
        total_questions: Number of questions evaluated across the
            entire dataset.
        aggregate_recall_at_k: Dataset-wide recall@K.
        aggregate_substring_match_rate: Dataset-wide substring-match
            rate.
        per_scenario: Per-scenario breakdown keyed by scenario name.
        per_difficulty: Per-difficulty breakdown keyed by difficulty
            tag (``"easy"`` / ``"medium"`` / ``"hard"``).

    """

    dreaming_enabled: bool
    top_k: int
    total_questions: int
    aggregate_recall_at_k: float
    aggregate_substring_match_rate: float
    per_scenario: dict[str, PerBreakdown] = field(default_factory=dict)
    per_difficulty: dict[str, PerBreakdown] = field(default_factory=dict)


@dataclass(frozen=True)
class _QuestionRecord:
    """Internal per-question scoring outcome, aggregated at the end."""

    question_id: str
    scenario_name: str
    difficulty: str
    recalled: bool
    substring_hit: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_embedding_provider_or_exit(
    *,
    model_name: str = _DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingProviderProtocol:
    """Return a configured ``SentenceTransformerProvider`` or exit cleanly.

    The synthetic benchmark requires a real embedding provider; the
    sentence-transformers extras are part of the ``[dev]`` set but
    not the default install.  Surface a clear actionable message to
    end users who try to run the suite from a vanilla
    ``pip install engrava`` rather than letting the
    :class:`ImportError` bubble through the asyncio stack.

    Returns:
        A :class:`engrava.SentenceTransformerProvider` instance.

    """
    # Probe the third-party ``sentence_transformers`` package directly:
    # the ``engrava.embeddings.sentence_transformer`` *wrapper* module
    # imports fine without the extras (it only ``raise``-s ``ImportError``
    # later, from inside ``_load_model``), so checking the wrapper does
    # not catch the missing-extras case.  An ``importlib.util.find_spec``
    # probe surfaces the real condition before we hand a half-broken
    # provider to the async stack.
    if importlib.util.find_spec("sentence_transformers") is None:
        sys.stderr.write(
            "engrava synthetic benchmark requires the embeddings extras:\n"
            "    pip install 'engrava[embeddings-local]'\n",
        )
        sys.exit(2)
    from engrava.embeddings.sentence_transformer import (  # noqa: PLC0415
        SentenceTransformerProvider,
    )

    return SentenceTransformerProvider(model_name=model_name)


async def evaluate_run(
    conversations: Iterable[SyntheticConversation],
    *,
    dreaming_enabled: bool,
    embedding_provider: EmbeddingProviderProtocol,
    retrieval_top_k: int = _DEFAULT_TOP_K,
    db_path: Path | None = None,
    search_config: SearchConfig | None = None,
) -> EvaluationResult:
    """Run the OFF / ON benchmark against a fresh engrava store.

    A new ``SqliteEngravaCore`` is constructed per call (either
    ``:memory:`` or a fresh on-disk path) so state never carries
    over between OFF and ON invocations.  Embeddings are computed
    by the supplied provider — the caller MUST share one provider
    between OFF and ON runs in the same process to avoid paying the
    cold-load cost twice.

    Args:
        conversations: Output of
            :func:`engrava.benchmarks.synthetic.generate.generate_dataset`,
            already in memory.  An iterable is sufficient; the
            evaluator does not iterate twice.
        dreaming_enabled: When ``True`` a :class:`DreamingExtension`
            is constructed and ``run_consolidation`` is invoked once
            after every conversation's turns have been ingested.
            When ``False`` no ``DreamingExtension`` is constructed —
            strictly stricter than building one with ``enabled=False``
            so no accidental side effect leaks into the OFF arm.
        embedding_provider: The provider that backs auto-embed on
            the core.  Re-use a single instance for OFF + ON.
        retrieval_top_k: Top-K passed to ``search_hybrid``.  Defaults
            to ``5`` (the metric reported in the runner's summary).
        db_path: Optional on-disk SQLite path.  ``None`` (default)
            uses ``:memory:``.
        search_config: Optional override for the store's
            :class:`SearchConfig`.  When ``None``, defaults to the
            benchmark's binding configuration with
            ``reflection_boost=1.0`` (per AC-8b).  Tests that exercise
            sanity neutrality MUST pass the binding config explicitly
            so the benchmark behaviour matches what the CLI runner
            uses in production.

    Returns:
        :class:`EvaluationResult` with aggregate and per-bucket
        breakdowns.

    """
    materialised = tuple(conversations)
    db_uri = str(db_path) if db_path is not None else ":memory:"

    # When dreaming is enabled, build the binding (DreamingConfig,
    # SearchConfig) pair from a single source of truth.  When the
    # caller passes an explicit ``search_config`` use it instead so
    # tests can probe the impact of ranking knobs independently from
    # the dreaming side.
    binding_dreaming, binding_search = _build_dreaming_config()
    effective_search = search_config if search_config is not None else binding_search

    async with aiosqlite.connect(db_uri) as db:
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            db=db,
            embedding_provider=embedding_provider,
            auto_embed=True,
            search_config=effective_search,
        )
        await store.ensure_schema()

        dream_ext = DreamingExtension(config=binding_dreaming) if dreaming_enabled else None

        # Local fact_id -> thought_id map, scoped to this call so OFF
        # and ON arms never see each other's IDs.
        fact_to_thoughts: dict[str, list[str]] = {}
        per_question_records: list[_QuestionRecord] = []
        cycle = 0

        for conv in materialised:
            cycle = await _ingest_conversation(
                store=store,
                conv=conv,
                cycle=cycle,
                fact_to_thoughts=fact_to_thoughts,
            )
            if dream_ext is not None:
                await dream_ext.run_consolidation(store, current_cycle=cycle)
                cycle += 1
            for question in conv.questions:
                record = await _score_question(
                    store=store,
                    question=question,
                    fact_to_thoughts=fact_to_thoughts,
                    top_k=retrieval_top_k,
                )
                per_question_records.append(record)

    return _aggregate(
        records=per_question_records,
        dreaming_enabled=dreaming_enabled,
        top_k=retrieval_top_k,
    )


# ---------------------------------------------------------------------------
# Internals — ingestion + scoring + aggregation
# ---------------------------------------------------------------------------


def _build_dreaming_config() -> tuple[DreamingConfig, SearchConfig]:
    """Return the binding (DreamingConfig, SearchConfig) pair for the benchmark.

    Two coupled configurations, both committed to git as the single
    canonical setup for the synthetic benchmark — operators wanting
    custom sweeps should write their own runner.

    DreamingConfig: tuned for the synthetic corpus (~2k thoughts,
    multi-conversation, 8-12 facets per synthesis scenario) so the
    dreaming mechanism has the inputs it needs to form REFLECTION
    cluster summaries:

    * ``candidates_limit`` raised above the corpus size so the early
      consolidation pass evaluates every planted turn (the default
      ``200`` truncates and starves the clustering phase).
    * ``cluster_algorithm="agglomerative"`` works directly on
      embedding similarity, independent of the ASSOCIATED-edge graph.
    * ``cluster_similarity_threshold=0.78`` is tight enough that only
      genuine paraphrases / theme facets cluster together; looser
      thresholds let distractions slip into centroids and produce
      noise REFLECTIONs that displace correct OBS at retrieval time.
    * ``cluster_quality_external_homogeneity_threshold=0.0`` because
      the synthetic corpus is entirely user utterances; the gate's
      default ``0.95`` would reject 100 % of clusters with
      ``external_source_mixed``.

    SearchConfig: ``reflection_boost=1.0`` (disable boost).  The
    engrava core ships REFLECTIONs at the same retrieval weight as
    OBSERVATIONs in the synthetic benchmark — boost > 1.0 displaced
    correct direct-retrieval hits on the AC-8 sanity scenarios in the
    earlier v1.2 calibration; AC-8b binds this gate so the
    boost-disabled behaviour is exercised on every benchmark run.
    """
    dreaming = DreamingConfig(
        enabled=True,
        promote_threshold=0.3,
        max_p1_fraction=0.15,
        promote_targets="OBS_ONLY",
        candidates_limit=3000,
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            max_promoted_per_run=200,
            enable_reflections=True,
            min_cluster_size=2,
            cluster_algorithm="agglomerative",
            cluster_similarity_threshold=0.78,
            cluster_quality_gating_enabled=True,
            # Synthetic corpus is 100 % user utterances; the external-
            # source-homogeneity gate would otherwise reject every
            # cluster as ``external_source_mixed``.
            cluster_quality_external_homogeneity_threshold=0.0,
        ),
    )
    # ``reflection_boost=1.0`` disables the REFLECTION score multiplier;
    # the v1.2 calibration confirmed boost > 1.0 displaces direct OBS
    # hits on the AC-8 sanity subset.  AC-8b binds this configuration.
    search = SearchConfig(reflection_boost=1.0)
    return dreaming, search


def _build_thought(
    turn: SyntheticTurn,
    *,
    conversation_id: str,
    cycle: int,
) -> ThoughtRecord:
    """Construct one ``ThoughtRecord`` from a synthetic turn.

    The turn's ``perspective`` / ``source_is_self`` flags populate
    the self-anchored metadata shape the metadata-aware dreaming
    filter consumes.  Memorable turns carry their stable
    ``synthetic_fact_id`` so the evaluator can map back from
    question.expected_fact_ids to the persisted thought_id.
    """
    return ThoughtRecord(
        thought_id=f"synth-{conversation_id}-turn{turn.turn_index:04d}",
        thought_type=ThoughtType.OBSERVATION,
        essence=turn.text[:200],
        content=turn.text,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source="synthetic-benchmark",
        confirmation_count=0,
        confidence=0.9,
        metadata={
            "perspective": turn.perspective,
            "source": {
                "is_self": turn.source_is_self,
                "confidence": "high",
            },
            "session_id": conversation_id,
            "turn_index": turn.turn_index,
            "lang": "en",
            "content_type": "natural_language",
            # Benchmark-only key — ignored by engrava core, surfaced
            # via ``get_thought`` only when the evaluator inspects it.
            "synthetic_fact_id": turn.fact_id or "",
        },
    )


async def _ingest_conversation(
    *,
    store: SqliteEngravaCore,
    conv: SyntheticConversation,
    cycle: int,
    fact_to_thoughts: dict[str, list[str]],
) -> int:
    """Persist every turn; populate the fact_id -> thought_id index.

    Returns the cycle counter advanced past the conversation's
    turns so subsequent conversations and consolidation passes use
    fresh values.
    """
    next_cycle = cycle
    for turn in conv.turns:
        thought = _build_thought(
            turn,
            conversation_id=conv.conversation_id,
            cycle=next_cycle,
        )
        persisted = await store.create_thought(thought)
        if turn.fact_id is not None:
            fact_to_thoughts.setdefault(turn.fact_id, []).append(persisted.thought_id)
        next_cycle += 1
    return next_cycle


async def _score_retrieval(
    *,
    store: SqliteEngravaCore,
    retrieved: list[tuple[str, float]],
    expected_thought_ids: set[str],
) -> bool:
    """Return ``True`` iff at least one retrieved entry resolves the question.

    Two semantics share this helper:

    (a) **Direct OBSERVATION match** — a retrieved ``(thought_id,
        score)`` whose ``thought_id`` is in ``expected_thought_ids``.
        Used by every scenario where the planted fact is itself the
        answer.

    (b) **REFLECTION-as-answer-carrier match** — a retrieved entry
        that resolves to a ``REFLECTION`` whose ``consolidated_from``
        list intersects ``expected_thought_ids``.  Used by the
        synthesis-requiring scenarios where the answer does NOT exist
        in any single planted OBSERVATION; it only exists in the
        cluster summary that dreaming materialises.

    Without (b) the benchmark would only measure direct retrieval and
    dreaming would mechanically be unable to demonstrate gain on
    synthesis scenarios.

    Args:
        store: The store the retrieved IDs come from.  Looked up via
            ``get_thought`` to inspect ``thought_type`` and
            ``consolidated_from``.
        retrieved: ``(thought_id, score)`` pairs from
            :meth:`SqliteEngravaCore.search_hybrid`.  Ordered
            descending by score; this helper does not care about the
            score itself, only the membership.
        expected_thought_ids: Persisted-store IDs of the planted
            memorable facts for the question being scored.

    Returns:
        ``True`` if any retrieved entry satisfies (a) or (b),
        ``False`` otherwise.

    """
    if not expected_thought_ids:
        return False
    for thought_id, _score in retrieved:
        # (a) Direct OBS match — cheapest check, no DB hit needed.
        if thought_id in expected_thought_ids:
            return True
        # (b) REFL match — pull the full record to inspect type +
        # consolidated_from.  A retrieved REFLECTION with
        # ``consolidated_from=None`` (legacy or non-dreaming origin)
        # cannot match because ``set() & x == set()`` — safe default.
        record = await store.get_thought(thought_id)
        if record is None:
            continue
        if record.thought_type != ThoughtType.REFLECTION:
            continue
        consolidated_from = record.consolidated_from or []
        if expected_thought_ids.intersection(consolidated_from):
            return True
    return False


async def _score_question(
    *,
    store: SqliteEngravaCore,
    question: SyntheticQuestion,
    fact_to_thoughts: dict[str, list[str]],
    top_k: int,
) -> _QuestionRecord:
    """Issue a hybrid-search query and score the result for one question."""
    hsr = await store.search_hybrid(question.question_text, top_k=top_k)

    expected_thought_ids = {
        tid for fid in question.expected_fact_ids for tid in fact_to_thoughts.get(fid, [])
    }
    recalled = await _score_retrieval(
        store=store,
        retrieved=hsr.results,
        expected_thought_ids=expected_thought_ids,
    )

    substring_hit = False
    if question.expected_substrings:
        for thought_id, _score in hsr.results:
            persisted = await store.get_thought(thought_id)
            if persisted is None:
                continue
            if any(sub in persisted.content for sub in question.expected_substrings):
                substring_hit = True
                break

    return _QuestionRecord(
        question_id=question.question_id,
        scenario_name=question.scenario_name,
        difficulty=question.difficulty,
        recalled=recalled,
        substring_hit=substring_hit,
    )


def _aggregate(
    *,
    records: list[_QuestionRecord],
    dreaming_enabled: bool,
    top_k: int,
) -> EvaluationResult:
    """Aggregate per-question records into the public result shape."""
    total = len(records)
    if total == 0:
        return EvaluationResult(
            dreaming_enabled=dreaming_enabled,
            top_k=top_k,
            total_questions=0,
            aggregate_recall_at_k=0.0,
            aggregate_substring_match_rate=0.0,
        )

    aggregate_recall = sum(1 for r in records if r.recalled) / total
    aggregate_substring = sum(1 for r in records if r.substring_hit) / total

    per_scenario = _bucket(records, key=lambda r: r.scenario_name)
    per_difficulty = _bucket(records, key=lambda r: r.difficulty)

    return EvaluationResult(
        dreaming_enabled=dreaming_enabled,
        top_k=top_k,
        total_questions=total,
        aggregate_recall_at_k=aggregate_recall,
        aggregate_substring_match_rate=aggregate_substring,
        per_scenario=per_scenario,
        per_difficulty=per_difficulty,
    )


def _bucket(
    records: list[_QuestionRecord],
    *,
    key: Callable[[_QuestionRecord], str],
) -> dict[str, PerBreakdown]:
    """Group per-question records by ``key`` and roll up the metrics."""
    grouped: dict[str, list[_QuestionRecord]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)

    out: dict[str, PerBreakdown] = {}
    for bucket_key, bucket_records in grouped.items():
        count = len(bucket_records)
        out[bucket_key] = PerBreakdown(
            name=bucket_key,
            question_count=count,
            recall_at_k=sum(1 for r in bucket_records if r.recalled) / count,
            substring_match_rate=(sum(1 for r in bucket_records if r.substring_hit) / count),
        )
    return out


# ---------------------------------------------------------------------------
# Synchronous convenience wrapper — used by runner + tests
# ---------------------------------------------------------------------------


async def measure_synthesis_coverage(
    conversations: Iterable[SyntheticConversation],
    *,
    embedding_provider: EmbeddingProviderProtocol,
    db_path: Path | None = None,
) -> float:
    """Return the synthesis-subset coverage rate at the data layer.

    For every question whose ``expected_fact_ids`` resolve to at least
    one persisted thought, count the question as "covered" if the
    post-dreaming store contains a REFLECTION whose cluster membership
    (``ThoughtRecord.consolidated_from`` field OR ``CONSOLIDATED_FROM``
    outgoing edges from the REFLECTION) intersects the expected set.

    This is the binding AC-9a v1.3 metric — it measures the dreaming
    mechanism at the data layer (clustering, REFLECTION creation,
    cluster-membership wiring) without depending on retrieval
    ranking.  Retrieval-layer surfacing is the deferred AC-9c gate
    that lands in a follow-up workstream.

    The helper builds its own store + DreamingExtension from the
    benchmark's binding ``_build_dreaming_config`` pair, ingests every
    conversation, runs consolidation per-conversation (matching the
    evaluator's flow), and finally walks every REFLECTION in the
    store enumerating cluster members.

    Args:
        conversations: Output of
            :func:`engrava.benchmarks.synthetic.generate.generate_dataset`,
            already in memory.  Coverage is computed only on
            conversations whose questions resolve to persisted
            thoughts — synthesis or otherwise; callers filter the
            input to the synthesis subset when targeting AC-9a.
        embedding_provider: Embedding provider for auto-embed on
            ingestion.  Re-use across calls to amortise the
            sentence-transformers cold-load cost.
        db_path: Optional on-disk SQLite path.  ``None`` (default)
            uses ``:memory:``.

    Returns:
        Coverage rate in ``[0.0, 1.0]``.  Returns ``0.0`` when the
        input has no questions with resolvable expected thoughts
        (vacuously empty subset — the caller should pre-filter).

    """
    materialised = tuple(conversations)
    if not materialised:
        return 0.0

    binding_dreaming, binding_search = _build_dreaming_config()

    db_uri = str(db_path) if db_path is not None else ":memory:"
    async with aiosqlite.connect(db_uri) as db:
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            db=db,
            embedding_provider=embedding_provider,
            auto_embed=True,
            search_config=binding_search,
        )
        await store.ensure_schema()

        dream_ext = DreamingExtension(config=binding_dreaming)
        fact_to_thoughts: dict[str, list[str]] = {}
        await _ingest_and_consolidate(
            store=store,
            dream_ext=dream_ext,
            conversations=materialised,
            fact_to_thoughts=fact_to_thoughts,
        )

        question_records = _build_coverage_question_records(
            conversations=materialised,
            fact_to_thoughts=fact_to_thoughts,
        )
        if not question_records:
            return 0.0

        reflection_memberships = await _collect_reflection_memberships(store)
        covered = sum(
            1
            for _qid, expected in question_records
            if any(expected & members for members in reflection_memberships)
        )
        return covered / len(question_records)


async def _ingest_and_consolidate(
    *,
    store: SqliteEngravaCore,
    dream_ext: DreamingExtension,
    conversations: tuple[SyntheticConversation, ...],
    fact_to_thoughts: dict[str, list[str]],
) -> None:
    """Ingest every conversation and run dreaming after each one."""
    cycle = 0
    for conv in conversations:
        cycle = await _ingest_conversation(
            store=store,
            conv=conv,
            cycle=cycle,
            fact_to_thoughts=fact_to_thoughts,
        )
        await dream_ext.run_consolidation(store, current_cycle=cycle)
        cycle += 1


def _build_coverage_question_records(
    *,
    conversations: tuple[SyntheticConversation, ...],
    fact_to_thoughts: dict[str, list[str]],
) -> list[tuple[str, set[str]]]:
    """Return ``(question_id, expected_thought_ids)`` for resolvable questions.

    Questions whose ``expected_fact_ids`` do not resolve to any
    persisted thought are dropped — they cannot be covered by
    construction and would otherwise deflate the coverage rate.
    """
    records: list[tuple[str, set[str]]] = []
    for conv in conversations:
        for question in conv.questions:
            expected = {
                tid for fid in question.expected_fact_ids for tid in fact_to_thoughts.get(fid, [])
            }
            if expected:
                records.append((question.question_id, expected))
    return records


async def _collect_reflection_memberships(
    store: SqliteEngravaCore,
) -> list[set[str]]:
    """Walk every REFLECTION in the store and return cluster memberships.

    Two representations are unioned for each REFLECTION: the
    ``ThoughtRecord.consolidated_from`` field (forward-compat) and
    ``CONSOLIDATED_FROM`` outgoing edges (the representation the
    current dreaming pipeline emits at REFLECTION creation).
    """
    from engrava.domain.enums import EdgeType  # noqa: PLC0415

    reflections = await store.list_thoughts(
        thought_type="REFLECTION",
        limit=10_000,
    )
    memberships: list[set[str]] = []
    for refl in reflections:
        members: set[str] = set(refl.consolidated_from or [])
        edges = await store.get_edges(refl.thought_id, direction="OUT")
        for edge in edges:
            if edge.edge_type == EdgeType.CONSOLIDATED_FROM:
                members.add(edge.to_thought_id)
        memberships.append(members)
    return memberships


def run_evaluation(
    conversations: Iterable[SyntheticConversation],
    *,
    dreaming_enabled: bool,
    embedding_provider: EmbeddingProviderProtocol,
    retrieval_top_k: int = _DEFAULT_TOP_K,
    db_path: Path | None = None,
    search_config: SearchConfig | None = None,
) -> EvaluationResult:
    """Drive :func:`evaluate_run` synchronously from a non-async caller.

    The runner CLI is synchronous; this wrapper drives the asyncio
    loop so callers do not have to.  Tests can still ``await`` the
    underlying coroutine directly.
    """
    return asyncio.run(
        evaluate_run(
            conversations,
            dreaming_enabled=dreaming_enabled,
            embedding_provider=embedding_provider,
            retrieval_top_k=retrieval_top_k,
            db_path=db_path,
            search_config=search_config,
        ),
    )

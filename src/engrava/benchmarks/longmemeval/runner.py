"""LongMemEval harness — per-question fresh-store evaluation loop.

The runner consumes the typed ``LongMemEvalQuestion`` value-objects
from :mod:`engrava.benchmarks.longmemeval.dataset_loader` and produces
an aggregated :class:`LongMemEvalResults` payload.

Semantics:

* **Fresh store per question** — every question carries its own
  ``haystack_sessions``; LongMemEval explicitly forbids cross-question
  state, so we drop the store after each question rather than rely on
  a reset method.
* **Ingest path** — each haystack turn becomes a ``ThoughtRecord`` with
  the self-anchored metadata shape (``perspective`` derived from the
  turn's ``role`` — user → ``percept``, assistant → ``utterance`` /
  ``source.is_self`` accordingly).
* **Dreaming** — one consolidation cycle after ingestion when
  ``dreaming_enabled=True``; the configuration comes from
  :func:`_build_longmemeval_config` (the binding values are pinned in
  this module; calibration runs may refine them).
* **Hygiene (forgetting)** — one Memory Hygiene pass after ingestion (and
  after dreaming, when both are on) when ``hygiene_enabled=True``, so cold
  thoughts are *archived* before the question is answered. The pass is
  conservative: GC is OFF (archive-only, reversible), P1/pins protected, and a
  low ``eviction_threshold`` so only genuinely cold thoughts are forgotten. The
  per-question ``archived_count`` (and the archived thought ids) are reported so
  a no-recall-regression study can confirm forgetting actually happened and
  later check overlap with the answer-bearing thoughts.
* **Retrieval** — single ``search_hybrid`` query with the natural-
  language question; ``top_k`` defaults to 5 (matches the synthetic
  benchmark).
* **Scoring** — substring (default) or cosine via the
  :mod:`evaluate` module. The LLM mode is wired in C3.

The asynchronous core is :func:`run_longmemeval`; the synchronous
convenience wrapper :func:`run_longmemeval_sync` mirrors the synthetic
benchmark's pattern.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import aiosqlite

from engrava.benchmarks.longmemeval.dataset_loader import (
    LongMemEvalQuestion,
    LongMemEvalSession,
    LongMemEvalTurn,
)
from engrava.benchmarks.longmemeval.evaluate import (
    EvaluationOutcome,
    LLMJudgeClient,
    evaluate_cosine,
    evaluate_llm,
    evaluate_substring,
)
from engrava.config import (
    DreamingConfig,
    DreamingGates,
    HygienePolicyConfig,
    SearchConfig,
)
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from engrava.domain.protocols.embedding_provider import (
        EmbeddingProviderProtocol,
    )

__all__ = [
    "DEFAULT_HYGIENE_EVICTION_THRESHOLD",
    "DEFAULT_TOP_K",
    "EvalMode",
    "LongMemEvalResults",
    "QuestionResult",
    "run_longmemeval",
    "run_longmemeval_sync",
]


DEFAULT_TOP_K = 5

#: Conservative default eviction-score cutoff for the benchmark hygiene arm.
#: Mirrors :attr:`HygienePolicyConfig.eviction_threshold` (``0.20``) — a
#: deliberately low bar so only clearly cold, low-value thoughts are archived,
#: keeping the "ON" arm a genuine no-recall-regression probe rather than a
#: blunt mass-forget. Exposed as a CLI knob so calibration runs can tune it.
DEFAULT_HYGIENE_EVICTION_THRESHOLD = 0.20

EvalMode = Literal["substring", "cosine", "llm"]


@dataclass(frozen=True)
class QuestionResult:
    """Per-question evaluation outcome.

    Attributes:
        question_id: Stable identifier carried over from the dataset.
        question_type: Taxonomy label (e.g. ``single-session-recall``).
        eval_mode: Mode used to produce ``outcome``.
        outcome: The :class:`EvaluationOutcome` returned by the mode.
        archived_count: Number of thoughts the hygiene pass archived for this
            question (``0`` when hygiene is off). The predict-before-spend
            signal that forgetting actually happened.
        archived_thought_ids: The thought ids archived by the hygiene pass for
            this question, in deterministic order. Empty when hygiene is off.
            Retained so a study can later check overlap with the answer-bearing
            thoughts.

    """

    question_id: str
    question_type: str
    eval_mode: EvalMode
    outcome: EvaluationOutcome
    archived_count: int = 0
    archived_thought_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LongMemEvalResults:
    """Aggregated harness result for one run.

    Attributes:
        dreaming_enabled: ``True`` when the run executed a
            consolidation cycle per question.
        hygiene_enabled: ``True`` when the run executed a Memory Hygiene
            (forgetting) pass per question.
        top_k: Retrieval ``top_k`` used.
        eval_mode: Evaluation mode used.
        total_questions: Number of questions scored.
        aggregate_score: Mean ``score`` across all per-question
            outcomes (``0.0`` for an empty input).
        aggregate_raw_signal: Mean ``raw_signal`` across all per-
            question outcomes.
        archived_count: Total thoughts archived by the hygiene pass across every
            question (``0`` when hygiene is off) — the run-level forgetting
            volume.
        per_type: Aggregate score keyed by ``question_type``.
        question_results: Per-question outcomes in input order; tests
            can use this for deterministic comparisons.

    """

    dreaming_enabled: bool
    hygiene_enabled: bool
    top_k: int
    eval_mode: EvalMode
    total_questions: int
    aggregate_score: float
    aggregate_raw_signal: float
    archived_count: int = 0
    per_type: dict[str, float] = field(default_factory=dict)
    question_results: tuple[QuestionResult, ...] = field(default_factory=tuple)


def _build_longmemeval_config() -> tuple[DreamingConfig, SearchConfig]:
    """Return the binding (DreamingConfig, SearchConfig) pair for the harness.

    These defaults are smaller than the synthetic benchmark's tune —
    each LongMemEval question has a self-contained haystack of roughly
    50-200 turns, well below the synthetic corpus size of ~2k thoughts.
    Final values will be re-calibrated after the first real-data run.
    """
    dreaming = DreamingConfig(
        enabled=True,
        promote_threshold=0.3,
        max_p1_fraction=0.15,
        promote_targets="OBS_ONLY",
        candidates_limit=500,
        gates=DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            max_promoted_per_run=200,
            enable_reflections=True,
            min_cluster_size=2,
            cluster_algorithm="agglomerative",
            cluster_similarity_threshold=0.78,
            cluster_quality_gating_enabled=True,
            cluster_quality_external_homogeneity_threshold=0.0,
        ),
    )
    search = SearchConfig(reflection_boost=1.0)
    return dreaming, search


def _build_hygiene_policy(
    *,
    enabled: bool,
    eviction_threshold: float,
) -> HygienePolicyConfig | None:
    """Return the conservative benchmark hygiene policy, or ``None`` when off.

    The policy is intentionally cautious so the forgetting arm never becomes a
    blunt mass-delete that would trivially regress recall:

    * **Archive-only** — ``auto_gc_enabled=False``: the pass only flips cold
      thoughts to ``ARCHIVED`` (reversible, no physical delete), so every
      forgotten thought remains restorable and inspectable after the run.
    * **P1 / pins protected** — the :class:`HygienePolicyConfig` defaults
      (``protected_priorities=("P1",)`` plus the always-on pin invariant) are
      kept, so promoted / high-value thoughts are never forgotten.
    * **Low threshold** — ``eviction_threshold`` defaults to the conservative
      ``0.20`` bar; only genuinely cold, low-value thoughts fall beneath it.

    Args:
        enabled: When ``False`` returns ``None`` so the store is built without a
            hygiene policy and the run performs no forgetting (the OFF arm).
        eviction_threshold: Eviction-score cutoff forwarded to the policy;
            exposed as a CLI knob for calibration.

    Returns:
        A conservative :class:`HygienePolicyConfig` when ``enabled``; otherwise
        ``None``.

    """
    if not enabled:
        return None
    return HygienePolicyConfig(
        enabled=True,
        eviction_threshold=eviction_threshold,
        auto_gc_enabled=False,
    )


def _perspective_for_role(role: str) -> str:
    """Map LongMemEval turn role onto the self-anchored perspective tag."""
    if role.lower() == "assistant":
        return "utterance"
    return "percept"


def _build_thought_from_lme_turn(
    turn: LongMemEvalTurn,
    *,
    question_id: str,
    session: LongMemEvalSession,
    turn_index: int,
    cycle: int,
) -> ThoughtRecord:
    """Construct a ``ThoughtRecord`` from one LongMemEval haystack turn.

    Metadata shape mirrors the synthetic benchmark's ``_build_thought``
    so the same self-anchored namespace is exposed to the dreaming
    metadata filter:

    * ``perspective`` — derived from ``role``.
    * ``source.is_self`` — ``True`` when the turn is from the
      assistant (the speaker is the agent), ``False`` for user input.
    * ``session_id`` — the haystack session this turn belongs to.
    * ``turn_index`` — index within the session.
    * ``lang`` / ``content_type`` — defaults documented for the
      filter contract.

    Args:
        turn: One turn of one haystack session.
        question_id: Owning question's identifier (scopes the
            thought_id namespace; same id on different questions never
            collides because each runs in its own store).
        session: Owning session value-object.
        turn_index: Position within the session, used to build a
            stable thought_id.
        cycle: Monotonic cycle counter the runner advances per turn.

    Returns:
        A constructed :class:`ThoughtRecord` ready for
        ``store.create_thought``.

    """
    role = turn.role.lower()
    is_self = role == "assistant"
    perspective = _perspective_for_role(role)
    return ThoughtRecord(
        thought_id=f"lme-{question_id}-{session.session_id}-t{turn_index:04d}",
        thought_type=ThoughtType.OBSERVATION,
        essence=turn.content[:200],
        content=turn.content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=cycle,
        updated_cycle=cycle,
        source="longmemeval-benchmark",
        confirmation_count=0,
        confidence=0.9,
        metadata={
            "perspective": perspective,
            "source": {
                "is_self": is_self,
                "confidence": "high",
            },
            "session_id": session.session_id,
            "turn_index": turn_index,
            "lang": "en",
            "content_type": "natural_language",
            "lme_role": role,
        },
    )


async def _resolve_contents(
    store: SqliteEngravaCore,
    retrieved: list[tuple[str, float]],
) -> list[str]:
    """Fetch the persisted content for each retrieved ``(thought_id, score)``.

    Returns chunks in the same order as the input. Thoughts that fail
    to resolve (e.g. evicted between search and fetch — should never
    happen in the harness flow but defended against for safety) are
    skipped silently.
    """
    contents: list[str] = []
    for thought_id, _score in retrieved:
        record = await store.get_thought(thought_id)
        if record is None:
            continue
        contents.append(record.content)
    return contents


async def _score_question(
    *,
    question: LongMemEvalQuestion,
    retrieved_contents: list[str],
    eval_mode: EvalMode,
    embedding_provider: EmbeddingProviderProtocol,
    llm_judge: LLMJudgeClient | None,
) -> EvaluationOutcome:
    """Dispatch to the configured evaluator and return its outcome."""
    if eval_mode == "substring":
        return evaluate_substring(question.answer, retrieved_contents)
    if eval_mode == "cosine":
        return await evaluate_cosine(
            question.answer,
            retrieved_contents,
            embedding_provider=embedding_provider,
        )
    if eval_mode == "llm":
        if llm_judge is None:
            msg = (
                "eval_mode='llm' requires an llm_judge client; supply one via "
                "run_longmemeval(..., llm_judge=<your client>)"
            )
            raise ValueError(msg)
        return await evaluate_llm(
            question.answer,
            retrieved_contents,
            question.question,
            judge=llm_judge,
        )
    msg = f"unknown eval_mode {eval_mode!r}"
    raise ValueError(msg)


async def _ingest_question(
    store: SqliteEngravaCore,
    question: LongMemEvalQuestion,
) -> int:
    """Persist every haystack turn; return the next cycle counter."""
    cycle = 0
    for session in question.haystack_sessions:
        for turn_index, turn in enumerate(session.turns):
            thought = _build_thought_from_lme_turn(
                turn,
                question_id=question.question_id,
                session=session,
                turn_index=turn_index,
                cycle=cycle,
            )
            await store.create_thought(thought)
            cycle += 1
    return cycle


async def _archived_thought_ids(store: SqliteEngravaCore) -> frozenset[str]:
    """Return the ids of every ARCHIVED thought currently in the store.

    Used to attribute a hygiene pass's effect: snapshot the archived set before
    and after :meth:`run_hygiene` and diff, so the reported archived ids reflect
    exactly what the pass forgot (never a reflection retired by dreaming, nor a
    TTL/manual archive). The limit is set well above any single LongMemEval
    haystack (tens to low hundreds of turns) so the snapshot is complete.

    Args:
        store: The per-question store to inspect.

    Returns:
        The frozenset of archived thought ids.

    """
    archived = await store.list_thoughts(
        lifecycle_status=LifecycleStatus.ARCHIVED.value,
        include_expired=True,
        limit=1_000_000,
    )
    return frozenset(record.thought_id for record in archived)


def _db_uri_for_question(
    question_id: str,
    db_dir: Path | None,
) -> str:
    """Return a fresh per-question SQLite URI.

    LongMemEval forbids cross-question state. When ``db_dir`` is
    ``None`` every question runs in its own ``:memory:`` database. When
    ``db_dir`` is set, each question gets its own file inside that
    directory (one DB file per question) so on-disk runs cannot leak
    haystack content across questions either.

    Filenames combine a human-readable sanitised prefix with a short
    SHA-256 hash of the original ``question_id``. The hash guarantees
    bijective mapping even when the sanitiser collapses distinct IDs
    to the same readable prefix (e.g. ``"a/b"`` and ``"a:b"`` would
    both sanitise to ``a_b`` but their hashes differ).

    Args:
        question_id: Stable identifier from the dataset.
        db_dir: Optional directory for on-disk DB files. The directory
            must exist or be createable; per-question file naming is
            deterministic.

    Returns:
        SQLite URI suitable for ``aiosqlite.connect``.

    """
    if db_dir is None:
        return ":memory:"
    db_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in question_id)
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:12]
    return str(db_dir / f"lme-{safe_name}-{digest}.sqlite")


async def _process_question(
    *,
    question: LongMemEvalQuestion,
    dreaming_enabled: bool,
    embedding_provider: EmbeddingProviderProtocol,
    eval_mode: EvalMode,
    retrieval_top_k: int,
    binding_dreaming: DreamingConfig,
    binding_search: SearchConfig,
    hygiene_policy: HygienePolicyConfig | None,
    db_dir: Path | None,
    llm_judge: LLMJudgeClient | None,
) -> QuestionResult:
    """Run the ingest/dream/forget/query/score loop for one question.

    When ``hygiene_policy`` is supplied the store forgets cold thoughts (one
    :meth:`run_hygiene` pass) after ingestion and after any dreaming cycle, so
    the archival is reflected in what retrieval can see. The ids archived by that
    pass are captured (a before/after diff of the ARCHIVED set) and returned.
    """
    db_uri = _db_uri_for_question(question.question_id, db_dir)
    async with aiosqlite.connect(db_uri) as db:
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            db=db,
            embedding_provider=embedding_provider,
            auto_embed=True,
            search_config=binding_search,
            hygiene_policy=hygiene_policy,
        )
        await store.ensure_schema()
        cycle = await _ingest_question(store, question)
        if dreaming_enabled:
            extension = DreamingExtension(config=binding_dreaming)
            await extension.run_consolidation(store, current_cycle=cycle)
        archived_ids: tuple[str, ...] = ()
        if hygiene_policy is not None:
            # Forget cold thoughts before the question is answered. Diff the
            # ARCHIVED set around the pass so only ids this pass archived are
            # attributed (dreaming-retired reflections, if any, are pre-existing
            # and excluded). The ingest cycle model gives older turns a colder
            # recency/staleness signal than recent ones, so a conservative
            # threshold archives a genuine subset rather than all-or-nothing.
            before = await _archived_thought_ids(store)
            await store.run_hygiene(current_cycle=cycle)
            after = await _archived_thought_ids(store)
            archived_ids = tuple(sorted(after - before))
        hsr = await store.search_hybrid(question.question, top_k=retrieval_top_k)
        contents = await _resolve_contents(store, hsr.results)
        outcome = await _score_question(
            question=question,
            retrieved_contents=contents,
            eval_mode=eval_mode,
            embedding_provider=embedding_provider,
            llm_judge=llm_judge,
        )
    return QuestionResult(
        question_id=question.question_id,
        question_type=question.question_type,
        eval_mode=eval_mode,
        outcome=outcome,
        archived_count=len(archived_ids),
        archived_thought_ids=archived_ids,
    )


async def run_longmemeval(
    questions: Iterable[LongMemEvalQuestion],
    *,
    dreaming_enabled: bool,
    embedding_provider: EmbeddingProviderProtocol,
    eval_mode: EvalMode = "substring",
    retrieval_top_k: int = DEFAULT_TOP_K,
    hygiene_enabled: bool = False,
    hygiene_eviction_threshold: float = DEFAULT_HYGIENE_EVICTION_THRESHOLD,
    db_path: Path | None = None,
    llm_judge: LLMJudgeClient | None = None,
) -> LongMemEvalResults:
    """Run the LongMemEval harness over the provided questions.

    Args:
        questions: Typed value-objects yielded by ``load_dataset``.
        dreaming_enabled: When ``True`` invoke ``run_consolidation``
            once per question between ingest and retrieval.
        embedding_provider: Provider that backs ``auto_embed`` on the
            store and (for cosine mode) the answer/chunk embedding.
            Share ONE provider across OFF and ON arms to amortise the
            cold-load cost.
        eval_mode: Evaluator dispatch — ``substring`` (default),
            ``cosine``, or ``llm`` (the LLM mode requires
            ``llm_judge``).
        retrieval_top_k: ``top_k`` passed to ``search_hybrid``.
        hygiene_enabled: When ``True`` run one conservative Memory Hygiene
            (forgetting) pass per question after ingestion (and after dreaming,
            when both are on), so cold thoughts are archived before retrieval.
            The pass is archive-only (no GC) and P1/pin-protected. When
            ``False`` (default) no forgetting occurs — the baseline arm.
        hygiene_eviction_threshold: Eviction-score cutoff for the hygiene pass;
            consulted only when ``hygiene_enabled``. Lower forgets less. Default
            :data:`DEFAULT_HYGIENE_EVICTION_THRESHOLD`.
        db_path: Optional directory for on-disk SQLite databases. When
            ``None`` (default) every question runs in its own
            ``:memory:`` database. When set, the runner creates one DB
            file per ``question_id`` inside this directory so the
            "fresh store per question" semantics hold on-disk too —
            cross-question contamination is impossible.
        llm_judge: Optional user-supplied LLM judge implementing
            :class:`LLMJudgeClient`. Required when ``eval_mode="llm"``.

    Returns:
        Aggregated :class:`LongMemEvalResults`.

    """
    materialised = tuple(questions)
    binding_dreaming, binding_search = _build_longmemeval_config()
    hygiene_policy = _build_hygiene_policy(
        enabled=hygiene_enabled,
        eviction_threshold=hygiene_eviction_threshold,
    )

    per_question: list[QuestionResult] = [
        await _process_question(
            question=question,
            dreaming_enabled=dreaming_enabled,
            embedding_provider=embedding_provider,
            eval_mode=eval_mode,
            retrieval_top_k=retrieval_top_k,
            binding_dreaming=binding_dreaming,
            binding_search=binding_search,
            hygiene_policy=hygiene_policy,
            db_dir=db_path,
            llm_judge=llm_judge,
        )
        for question in materialised
    ]

    return _aggregate(
        records=per_question,
        dreaming_enabled=dreaming_enabled,
        hygiene_enabled=hygiene_enabled,
        top_k=retrieval_top_k,
        eval_mode=eval_mode,
    )


def run_longmemeval_sync(
    questions: Iterable[LongMemEvalQuestion],
    *,
    dreaming_enabled: bool,
    embedding_provider: EmbeddingProviderProtocol,
    eval_mode: EvalMode = "substring",
    retrieval_top_k: int = DEFAULT_TOP_K,
    hygiene_enabled: bool = False,
    hygiene_eviction_threshold: float = DEFAULT_HYGIENE_EVICTION_THRESHOLD,
    db_path: Path | None = None,
    llm_judge: LLMJudgeClient | None = None,
) -> LongMemEvalResults:
    """Drive :func:`run_longmemeval` from a non-async caller.

    Mirrors the sync wrapper pattern in the synthetic benchmark so
    tests and the CLI runner do not have to manage the asyncio loop
    themselves.
    """
    return asyncio.run(
        run_longmemeval(
            questions,
            dreaming_enabled=dreaming_enabled,
            embedding_provider=embedding_provider,
            eval_mode=eval_mode,
            retrieval_top_k=retrieval_top_k,
            hygiene_enabled=hygiene_enabled,
            hygiene_eviction_threshold=hygiene_eviction_threshold,
            db_path=db_path,
            llm_judge=llm_judge,
        ),
    )


def _aggregate(
    *,
    records: list[QuestionResult],
    dreaming_enabled: bool,
    top_k: int,
    eval_mode: EvalMode,
    hygiene_enabled: bool = False,
) -> LongMemEvalResults:
    total = len(records)
    if total == 0:
        return LongMemEvalResults(
            dreaming_enabled=dreaming_enabled,
            hygiene_enabled=hygiene_enabled,
            top_k=top_k,
            eval_mode=eval_mode,
            total_questions=0,
            aggregate_score=0.0,
            aggregate_raw_signal=0.0,
            archived_count=0,
        )
    aggregate_score = sum(r.outcome.score for r in records) / total
    aggregate_raw = sum(r.outcome.raw_signal for r in records) / total
    per_type: dict[str, list[float]] = {}
    for r in records:
        per_type.setdefault(r.question_type, []).append(r.outcome.score)
    per_type_aggregated = {qt: sum(scores) / len(scores) for qt, scores in per_type.items()}
    return LongMemEvalResults(
        dreaming_enabled=dreaming_enabled,
        hygiene_enabled=hygiene_enabled,
        top_k=top_k,
        eval_mode=eval_mode,
        total_questions=total,
        aggregate_score=aggregate_score,
        aggregate_raw_signal=aggregate_raw,
        archived_count=sum(r.archived_count for r in records),
        per_type=per_type_aggregated,
        question_results=tuple(records),
    )

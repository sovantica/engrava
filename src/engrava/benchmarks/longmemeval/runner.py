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
from engrava.config import DreamingConfig, DreamingGates, SearchConfig
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
    "DEFAULT_TOP_K",
    "EvalMode",
    "LongMemEvalResults",
    "QuestionResult",
    "run_longmemeval",
    "run_longmemeval_sync",
]


DEFAULT_TOP_K = 5

EvalMode = Literal["substring", "cosine", "llm"]


@dataclass(frozen=True)
class QuestionResult:
    """Per-question evaluation outcome.

    Attributes:
        question_id: Stable identifier carried over from the dataset.
        question_type: Taxonomy label (e.g. ``single-session-recall``).
        eval_mode: Mode used to produce ``outcome``.
        outcome: The :class:`EvaluationOutcome` returned by the mode.

    """

    question_id: str
    question_type: str
    eval_mode: EvalMode
    outcome: EvaluationOutcome


@dataclass(frozen=True)
class LongMemEvalResults:
    """Aggregated harness result for one run.

    Attributes:
        dreaming_enabled: ``True`` when the run executed a
            consolidation cycle per question.
        top_k: Retrieval ``top_k`` used.
        eval_mode: Evaluation mode used.
        total_questions: Number of questions scored.
        aggregate_score: Mean ``score`` across all per-question
            outcomes (``0.0`` for an empty input).
        aggregate_raw_signal: Mean ``raw_signal`` across all per-
            question outcomes.
        per_type: Aggregate score keyed by ``question_type``.
        question_results: Per-question outcomes in input order; tests
            can use this for deterministic comparisons.

    """

    dreaming_enabled: bool
    top_k: int
    eval_mode: EvalMode
    total_questions: int
    aggregate_score: float
    aggregate_raw_signal: float
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
    db_dir: Path | None,
    llm_judge: LLMJudgeClient | None,
) -> QuestionResult:
    """Run the ingest/dream/query/score loop for one question."""
    db_uri = _db_uri_for_question(question.question_id, db_dir)
    async with aiosqlite.connect(db_uri) as db:
        db.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(
            db=db,
            embedding_provider=embedding_provider,
            auto_embed=True,
            search_config=binding_search,
        )
        await store.ensure_schema()
        cycle = await _ingest_question(store, question)
        if dreaming_enabled:
            extension = DreamingExtension(config=binding_dreaming)
            await extension.run_consolidation(store, current_cycle=cycle)
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
    )


async def run_longmemeval(
    questions: Iterable[LongMemEvalQuestion],
    *,
    dreaming_enabled: bool,
    embedding_provider: EmbeddingProviderProtocol,
    eval_mode: EvalMode = "substring",
    retrieval_top_k: int = DEFAULT_TOP_K,
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

    per_question: list[QuestionResult] = [
        await _process_question(
            question=question,
            dreaming_enabled=dreaming_enabled,
            embedding_provider=embedding_provider,
            eval_mode=eval_mode,
            retrieval_top_k=retrieval_top_k,
            binding_dreaming=binding_dreaming,
            binding_search=binding_search,
            db_dir=db_path,
            llm_judge=llm_judge,
        )
        for question in materialised
    ]

    return _aggregate(
        records=per_question,
        dreaming_enabled=dreaming_enabled,
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
) -> LongMemEvalResults:
    total = len(records)
    if total == 0:
        return LongMemEvalResults(
            dreaming_enabled=dreaming_enabled,
            top_k=top_k,
            eval_mode=eval_mode,
            total_questions=0,
            aggregate_score=0.0,
            aggregate_raw_signal=0.0,
        )
    aggregate_score = sum(r.outcome.score for r in records) / total
    aggregate_raw = sum(r.outcome.raw_signal for r in records) / total
    per_type: dict[str, list[float]] = {}
    for r in records:
        per_type.setdefault(r.question_type, []).append(r.outcome.score)
    per_type_aggregated = {qt: sum(scores) / len(scores) for qt, scores in per_type.items()}
    return LongMemEvalResults(
        dreaming_enabled=dreaming_enabled,
        top_k=top_k,
        eval_mode=eval_mode,
        total_questions=total,
        aggregate_score=aggregate_score,
        aggregate_raw_signal=aggregate_raw,
        per_type=per_type_aggregated,
        question_results=tuple(records),
    )

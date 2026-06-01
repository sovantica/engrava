"""Evaluation modes for the LongMemEval harness.

Three modes are provided:

* :func:`evaluate_substring` — deterministic, no external dependency:
  the question's ``answer`` must appear in at least one retrieved
  chunk (case-insensitive by default).
* :func:`evaluate_cosine` — deterministic given a fixed embedding
  provider: maximum cosine similarity between the embedded expected
  answer and embedded retrieved chunks, scored against a threshold.
* :func:`evaluate_llm` — opt-in LLM judge for open-ended scoring;
  requires a user-supplied client implementing :class:`LLMJudgeClient`.
  NOT required for the Free quality gate.

The first two modes never invoke a language model and form the
binding default. The LLM mode is informational.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engrava.domain.protocols.embedding_provider import (
        EmbeddingProviderProtocol,
    )

__all__ = [
    "DEFAULT_COSINE_THRESHOLD",
    "EvaluationOutcome",
    "LLMJudgeClient",
    "LLMJudgeError",
    "evaluate_cosine",
    "evaluate_llm",
    "evaluate_substring",
]


DEFAULT_COSINE_THRESHOLD = 0.7


@dataclass(frozen=True)
class EvaluationOutcome:
    """One question's evaluation result for a single mode.

    Attributes:
        score: Binary hit indicator in ``[0.0, 1.0]``. ``1.0`` when the
            mode considers the question answered by the retrieved
            chunks, ``0.0`` otherwise. Modes that report a graded
            similarity may also surface a fractional score; the runner
            aggregates the mean.
        raw_signal: Mode-specific underlying signal. For ``substring``
            this is ``1.0`` / ``0.0`` (same as ``score``). For
            ``cosine`` this is the maximum cosine similarity observed
            across the retrieved chunks, irrespective of the threshold.

    """

    score: float
    raw_signal: float


def evaluate_substring(
    expected_answer: str,
    retrieved_contents: Sequence[str],
    *,
    case_sensitive: bool = False,
) -> EvaluationOutcome:
    """Return ``1.0`` when ``expected_answer`` appears in any chunk.

    Args:
        expected_answer: The question's ground-truth answer.
        retrieved_contents: Text of each retrieved thought, ordered by
            descending search score.
        case_sensitive: When ``False`` (default) both sides are
            lower-cased before the containment check.

    Returns:
        :class:`EvaluationOutcome` with ``score`` in ``{0.0, 1.0}``.

    """
    if not expected_answer.strip():
        return EvaluationOutcome(score=0.0, raw_signal=0.0)
    needle = expected_answer if case_sensitive else expected_answer.lower()
    for chunk in retrieved_contents:
        haystack = chunk if case_sensitive else chunk.lower()
        if needle in haystack:
            return EvaluationOutcome(score=1.0, raw_signal=1.0)
    return EvaluationOutcome(score=0.0, raw_signal=0.0)


async def evaluate_cosine(
    expected_answer: str,
    retrieved_contents: Sequence[str],
    *,
    embedding_provider: EmbeddingProviderProtocol,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> EvaluationOutcome:
    """Score retrieval by max cosine similarity of embedded answer vs chunks.

    Args:
        expected_answer: Ground-truth answer.
        retrieved_contents: Text of each retrieved thought.
        embedding_provider: Provider used to embed both the expected
            answer and each retrieved chunk. The same provider that
            powers the engrava store should be used so the embedding
            space matches.
        threshold: Cosine-similarity floor for a hit. Defaults to 0.7
            per WS spec §2.3.

    Returns:
        :class:`EvaluationOutcome` whose ``score`` is ``1.0`` when the
        max similarity meets ``threshold``, ``0.0`` otherwise.
        ``raw_signal`` always carries the maximum similarity observed
        (or ``0.0`` when no chunks were retrieved).

    """
    if not expected_answer.strip() or not retrieved_contents:
        return EvaluationOutcome(score=0.0, raw_signal=0.0)
    expected_vec = await _embed_one(embedding_provider, expected_answer)
    if expected_vec is None:
        return EvaluationOutcome(score=0.0, raw_signal=0.0)
    best = 0.0
    for chunk in retrieved_contents:
        chunk_vec = await _embed_one(embedding_provider, chunk)
        if chunk_vec is None:
            continue
        sim = _cosine_similarity(expected_vec, chunk_vec)
        best = max(best, sim)
    score = 1.0 if best >= threshold else 0.0
    return EvaluationOutcome(score=score, raw_signal=best)


class LLMJudgeError(RuntimeError):
    """Raised when the user-supplied LLM judge cannot return a verdict."""


@runtime_checkable
class LLMJudgeClient(Protocol):
    """User-supplied LLM judge surface.

    The harness never imports a specific provider SDK. Callers wire any
    client that implements this single async method — Anthropic,
    OpenAI, a local model, a fake-for-tests, anything goes.

    The method receives a system prompt + user prompt and returns a
    free-form string. The evaluator parses ``YES`` / ``NO`` (case-
    insensitive) at the start of the response.
    """

    async def judge(self, *, system: str, user: str) -> str:
        """Return the LLM's verdict on the user prompt under the system rules."""


async def evaluate_llm(
    expected_answer: str,
    retrieved_contents: Sequence[str],
    question_text: str,
    *,
    judge: LLMJudgeClient,
) -> EvaluationOutcome:
    """Score retrieval by asking a user-supplied LLM judge.

    Args:
        expected_answer: Ground-truth answer text.
        retrieved_contents: Text of each retrieved thought.
        question_text: Original question (some judges need this to
            disambiguate paraphrase from contradiction).
        judge: A client implementing :class:`LLMJudgeClient`.

    Returns:
        :class:`EvaluationOutcome` with ``score`` ``1.0`` when the
        judge starts its response with ``YES`` (case-insensitive),
        ``0.0`` otherwise. ``raw_signal`` mirrors ``score`` for the
        LLM mode (no graded similarity is exposed).

    Raises:
        LLMJudgeError: When the judge raises or returns an empty
            response. Callers can wrap this in their own retry policy.

    """
    if not expected_answer.strip() or not retrieved_contents:
        return EvaluationOutcome(score=0.0, raw_signal=0.0)
    chunk_block = "\n---\n".join(retrieved_contents)
    system_prompt = (
        "You are evaluating whether a memory-retrieval system surfaced "
        "the correct answer to a question. Reply YES or NO on the first "
        "line, then a one-sentence justification. YES means the retrieved "
        "context contains the expected answer (paraphrases acceptable). "
        "NO means it does not."
    )
    user_prompt = (
        f"Question: {question_text}\n"
        f"Expected answer: {expected_answer}\n\n"
        f"Retrieved context:\n{chunk_block}\n"
    )
    try:
        response = await judge.judge(system=system_prompt, user=user_prompt)
    except Exception as exc:
        msg = f"LLM judge raised: {exc!r}"
        raise LLMJudgeError(msg) from exc
    if not response.strip():
        msg = "LLM judge returned an empty response"
        raise LLMJudgeError(msg)
    leading = response.strip().split(maxsplit=1)[0].lower().rstrip(".,:;!")
    hit = leading == "yes"
    score = 1.0 if hit else 0.0
    return EvaluationOutcome(score=score, raw_signal=score)


async def _embed_one(
    provider: EmbeddingProviderProtocol,
    text: str,
) -> tuple[float, ...] | None:
    vectors = await provider.embed_batch([text])
    if not vectors:
        return None
    return tuple(vectors[0])


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b, strict=True):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 0.0:
        return 0.0
    return dot / denom

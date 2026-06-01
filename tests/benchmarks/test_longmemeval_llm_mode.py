"""Tests for the LLM-judge evaluation mode.

The harness defines a thin :class:`LLMJudgeClient` protocol so callers
can wire any LLM provider SDK (Anthropic, OpenAI, local). These tests
exercise the parsing layer with a fake judge that returns scripted
strings; no network is ever touched.
"""

from __future__ import annotations

import pytest

from engrava.benchmarks.longmemeval.evaluate import (
    EvaluationOutcome,
    LLMJudgeError,
    evaluate_llm,
)


class _ScriptedJudge:
    """LLM judge stub returning a pre-set response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_system: str | None = None
        self.last_user: str | None = None

    async def judge(self, *, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self._response


class _RaisingJudge:
    """LLM judge stub that always raises."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def judge(self, *, system: str, user: str) -> str:
        del system, user
        raise self._exc


class TestEvaluateLLM:
    """Parser + error handling for the LLM judge mode."""

    async def test_yes_response_hits(self) -> None:
        judge = _ScriptedJudge("YES — the retrieved context contains the expected answer.")
        outcome = await evaluate_llm(
            "tabby cat named Pepper",
            ["I adopted a tabby cat named Pepper yesterday."],
            "What pet did the user adopt?",
            judge=judge,
        )
        assert outcome == EvaluationOutcome(score=1.0, raw_signal=1.0)

    async def test_no_response_misses(self) -> None:
        judge = _ScriptedJudge("NO. The retrieved chunks do not contain the answer.")
        outcome = await evaluate_llm(
            "tabby cat named Pepper",
            ["The new coffee shop opens at seven."],
            "What pet did the user adopt?",
            judge=judge,
        )
        assert outcome == EvaluationOutcome(score=0.0, raw_signal=0.0)

    async def test_case_insensitive_yes(self) -> None:
        judge = _ScriptedJudge("yes — paraphrased but correct.")
        outcome = await evaluate_llm(
            "Pepper",
            ["I named my cat Pepper."],
            "Pet name?",
            judge=judge,
        )
        assert outcome.score == 1.0

    async def test_yes_followed_by_punctuation_hits(self) -> None:
        judge = _ScriptedJudge("YES! Clear match.")
        outcome = await evaluate_llm(
            "Pepper",
            ["Pepper"],
            "Pet?",
            judge=judge,
        )
        assert outcome.score == 1.0

    async def test_empty_response_raises(self) -> None:
        judge = _ScriptedJudge("   \n")
        with pytest.raises(LLMJudgeError, match="empty response"):
            await evaluate_llm(
                "Pepper",
                ["chunk"],
                "Pet?",
                judge=judge,
            )

    async def test_judge_exception_wrapped_in_domain_error(self) -> None:
        original = RuntimeError("upstream provider timed out")
        judge = _RaisingJudge(original)
        with pytest.raises(LLMJudgeError, match="LLM judge raised"):
            await evaluate_llm(
                "Pepper",
                ["chunk"],
                "Pet?",
                judge=judge,
            )

    async def test_empty_inputs_short_circuit_without_calling_judge(self) -> None:
        judge = _ScriptedJudge("YES")
        outcome = await evaluate_llm(
            "",
            ["chunk"],
            "Q",
            judge=judge,
        )
        assert outcome.score == 0.0
        assert judge.last_user is None
        outcome_no_chunks = await evaluate_llm(
            "Pepper",
            [],
            "Q",
            judge=judge,
        )
        assert outcome_no_chunks.score == 0.0
        assert judge.last_user is None

    async def test_prompt_carries_question_expected_and_chunks(self) -> None:
        judge = _ScriptedJudge("YES")
        await evaluate_llm(
            "Earl Grey with orange peel",
            ["I drink Earl Grey with orange peel every morning.", "filler"],
            "What is the favourite tea?",
            judge=judge,
        )
        assert judge.last_user is not None
        assert "Earl Grey with orange peel" in judge.last_user
        assert "What is the favourite tea?" in judge.last_user
        assert "filler" in judge.last_user

"""Hygiene (forgetting) arm of the LongMemEval harness — offline, deterministic.

These tests wire and exercise the ``--hygiene`` / ``hygiene_enabled`` arm with a
network-free bag-of-words embedder, so they run without any model download or
paid call. They pin three things:

* **Plumbing** — ``_build_hygiene_policy`` yields a conservative archive-only
  policy (GC off, P1 protected) or ``None`` when off; the flag threads through
  ``run_longmemeval`` into the per-question store and back out as
  ``archived_count`` / ``archived_thought_ids``.
* **It actually forgets a SUBSET** — at a threshold in the informative band the
  pass archives some-but-not-all haystack turns, and the archived set changes
  what retrieval surfaces (an archived answer turn drops the score).
* **The methodology signal** — at the *conservative default* threshold the pass
  archives **nothing** on a LongMemEval-shaped haystack, because per-turn cycles
  plus a uniform confidence floor keep every keep-score well above ``0.20``.
  This is pinned deliberately so a future change that makes the default forget
  (or stops it forgetting at the higher band) is caught.

The exact keep-score model is engrava-internal; the assertions use ranges and
ordering (``0 < archived < n``), not magic counts, so they survive score-formula
tuning while still proving the arm is informative.
"""

from __future__ import annotations

import hashlib

import pytest

from engrava import CallbackProvider
from engrava.benchmarks.longmemeval.dataset_loader import (
    LongMemEvalQuestion,
    LongMemEvalSession,
    LongMemEvalTurn,
)
from engrava.benchmarks.longmemeval.runner import (
    DEFAULT_HYGIENE_EVICTION_THRESHOLD,
    _build_hygiene_policy,
    run_longmemeval,
)
from engrava.config import HygienePolicyConfig

_EMBED_DIM = 64

# A threshold inside the informative band (empirically archives a strict subset
# of a LongMemEval-shaped haystack — neither 0 nor all — under the current
# keep-score model). The conservative default (0.20) archives nothing on this
# shape; see ``test_conservative_default_archives_nothing``.
_SUBSET_THRESHOLD = 0.72

_ANSWER = "XYLOPHONE99"
_QUESTION = "what is the secret code"


def _embed(text: str) -> list[float]:
    """Embed text as an L2-normalized bag-of-words hashing vector.

    Args:
        text: Input text to embed.

    Returns:
        An ``_EMBED_DIM``-length unit vector (all-zero only for empty text).
    """
    vector = [0.0] * _EMBED_DIM
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()  # noqa: S324
        vector[int.from_bytes(digest[:4], "big") % _EMBED_DIM] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _provider() -> CallbackProvider:
    """Return the deterministic, network-free embedding provider."""
    return CallbackProvider(
        callback=_embed,
        dimension=_EMBED_DIM,
        model_name="bag-of-words-hygiene",
    )


def _question(*, n_turns: int, answer_turn: int, question_id: str) -> LongMemEvalQuestion:
    """Build a single-session question whose answer lives at one turn.

    Every non-answer turn is generic filler with a distinct topic word; the
    answer turn carries the distinctive ``_ANSWER`` token. Cycles advance one
    per turn in the runner, so a lower ``answer_turn`` index is colder.

    Args:
        n_turns: Number of haystack turns in the single session.
        answer_turn: Index of the turn carrying the answer.
        question_id: Stable id (also namespaces the thought ids).

    Returns:
        A fully populated :class:`LongMemEvalQuestion`.
    """
    turns = []
    for i in range(n_turns):
        content = (
            f"turn {i} the secret code is {_ANSWER} please keep it"
            if i == answer_turn
            else f"turn {i} ordinary chatter about topic number {i} nothing notable"
        )
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(LongMemEvalTurn(role=role, content=content))
    return LongMemEvalQuestion(
        question_id=question_id,
        question_type="single-session-recall",
        question=_QUESTION,
        answer=_ANSWER,
        question_date="2026-01-01",
        haystack_sessions=(LongMemEvalSession(session_id="s0", turns=tuple(turns)),),
    )


def _answer_thought_id(question_id: str, answer_turn: int) -> str:
    """Return the thought id the runner assigns to the answer-bearing turn."""
    return f"lme-{question_id}-s0-t{answer_turn:04d}"


class TestBuildHygienePolicy:
    """The conservative benchmark hygiene-policy builder."""

    def test_disabled_returns_none(self) -> None:
        """``enabled=False`` builds no policy, so the store never forgets."""
        assert _build_hygiene_policy(enabled=False, eviction_threshold=0.2) is None

    def test_enabled_is_conservative_archive_only(self) -> None:
        """``enabled=True`` yields an archive-only, P1-protected policy."""
        policy = _build_hygiene_policy(enabled=True, eviction_threshold=0.33)
        assert isinstance(policy, HygienePolicyConfig)
        assert policy.enabled is True
        assert policy.auto_gc_enabled is False, "GC must stay off — archive-only"
        assert "P1" in policy.protected_priorities
        assert policy.eviction_threshold == pytest.approx(0.33)

    def test_default_threshold_constant_is_conservative(self) -> None:
        """The exported default mirrors the config's conservative bar."""
        assert (
            pytest.approx(HygienePolicyConfig().eviction_threshold)
            == DEFAULT_HYGIENE_EVICTION_THRESHOLD
        )


class TestHygieneArmOfflineSmoke:
    """End-to-end offline smoke: the arm forgets a subset and reports it."""

    async def test_off_arm_archives_nothing(self) -> None:
        """With hygiene off, nothing is archived and the flag is echoed."""
        question = _question(n_turns=16, answer_turn=14, question_id="q-off")
        results = await run_longmemeval(
            [question],
            dreaming_enabled=False,
            embedding_provider=_provider(),
            eval_mode="substring",
            hygiene_enabled=False,
        )
        assert results.hygiene_enabled is False
        assert results.archived_count == 0
        assert results.question_results[0].archived_count == 0
        assert results.question_results[0].archived_thought_ids == ()

    async def test_conservative_default_archives_nothing(self) -> None:
        """METHODOLOGY PIN: the conservative default (0.20) forgets nothing here.

        On a LongMemEval-shaped haystack (one cycle per turn, uniform confidence,
        no access/confirmation history) every keep-score sits well above the
        ``0.20`` bar, so the default hygiene arm is a no-op. A benchmark run at
        this threshold would be a false-green (ON == OFF because nothing is
        forgotten); the informative band is a much higher threshold. This test
        fails loudly if that ever silently changes.
        """
        question = _question(n_turns=16, answer_turn=14, question_id="q-default")
        results = await run_longmemeval(
            [question],
            dreaming_enabled=False,
            embedding_provider=_provider(),
            eval_mode="substring",
            hygiene_enabled=True,
            hygiene_eviction_threshold=DEFAULT_HYGIENE_EVICTION_THRESHOLD,
        )
        assert results.hygiene_enabled is True
        assert results.archived_count == 0, (
            "conservative default unexpectedly archived thoughts — the informative "
            "band changed; re-check the benchmark threshold"
        )

    async def test_informative_threshold_archives_a_strict_subset(self) -> None:
        """At the informative threshold the pass archives some-but-not-all turns."""
        n_turns = 16
        question = _question(n_turns=n_turns, answer_turn=14, question_id="q-subset")
        results = await run_longmemeval(
            [question],
            dreaming_enabled=False,
            embedding_provider=_provider(),
            eval_mode="substring",
            hygiene_enabled=True,
            hygiene_eviction_threshold=_SUBSET_THRESHOLD,
        )
        record = results.question_results[0]
        assert 0 < record.archived_count < n_turns, (
            "hygiene must forget a genuine subset (not zero, not all) for the "
            f"benchmark to be informative; archived {record.archived_count}/{n_turns}"
        )
        # Run-level count aggregates the per-question counts.
        assert results.archived_count == record.archived_count
        # The reported ids match the count and are real haystack turns.
        assert len(record.archived_thought_ids) == record.archived_count
        assert all(tid.startswith("lme-q-subset-s0-t") for tid in record.archived_thought_ids)

    async def test_forgetting_a_recent_answer_leaves_recall_intact(self) -> None:
        """A recent (warm) answer turn is not archived, so recall is unaffected."""
        question = _question(n_turns=16, answer_turn=14, question_id="q-warm")
        provider = _provider()
        off = await run_longmemeval(
            [question],
            dreaming_enabled=False,
            embedding_provider=provider,
            eval_mode="substring",
            hygiene_enabled=False,
        )
        on = await run_longmemeval(
            [question],
            dreaming_enabled=False,
            embedding_provider=provider,
            eval_mode="substring",
            hygiene_enabled=True,
            hygiene_eviction_threshold=_SUBSET_THRESHOLD,
        )
        answer_id = _answer_thought_id("q-warm", 14)
        assert answer_id not in on.question_results[0].archived_thought_ids
        assert on.question_results[0].archived_count > 0, "the arm still forgot cold turns"
        # The answer survived, so retrieval — and the score — are unchanged.
        assert off.aggregate_score == pytest.approx(1.0)
        assert on.aggregate_score == pytest.approx(off.aggregate_score)

    async def test_forgetting_a_cold_answer_changes_retrieval(self) -> None:
        """Archiving the (cold) answer turn removes it from retrieval — the diff.

        This is the load-bearing end-to-end check that the exclusion feature is
        actually exercised through the harness: once the answer-bearing thought
        is archived it leaves the default candidate set, so the substring score
        collapses relative to the OFF arm.
        """
        question = _question(n_turns=16, answer_turn=1, question_id="q-cold")
        provider = _provider()
        off = await run_longmemeval(
            [question],
            dreaming_enabled=False,
            embedding_provider=provider,
            eval_mode="substring",
            hygiene_enabled=False,
        )
        on = await run_longmemeval(
            [question],
            dreaming_enabled=False,
            embedding_provider=provider,
            eval_mode="substring",
            hygiene_enabled=True,
            hygiene_eviction_threshold=_SUBSET_THRESHOLD,
        )
        answer_id = _answer_thought_id("q-cold", 1)
        assert answer_id in on.question_results[0].archived_thought_ids, (
            "the cold answer turn should be among the forgotten thoughts"
        )
        assert off.aggregate_score == pytest.approx(1.0)
        assert on.aggregate_score < off.aggregate_score, (
            "forgetting the answer-bearing thought must change (lower) retrieval"
        )

    async def test_archived_set_is_deterministic(self) -> None:
        """Same input + threshold ⇒ identical archived set across runs."""
        question = _question(n_turns=16, answer_turn=14, question_id="q-det")
        runs = [
            await run_longmemeval(
                [question],
                dreaming_enabled=False,
                embedding_provider=_provider(),
                eval_mode="substring",
                hygiene_enabled=True,
                hygiene_eviction_threshold=_SUBSET_THRESHOLD,
            )
            for _ in range(2)
        ]
        first = runs[0].question_results[0].archived_thought_ids
        assert first == runs[1].question_results[0].archived_thought_ids
        assert len(first) > 0

"""Tests for the LongMemEval runner.

Two layers:

* **Unit** — exercise per-helper behaviour without touching engrava
  core (config builder, metadata builder, perspective mapper,
  aggregator).
* **End-to-end** — drive :func:`run_longmemeval` over the synthesised
  tiny fixture with a real ``SentenceTransformerProvider``. These
  tests are skipped when the ``embeddings-local`` extras are not
  available, so the suite still passes on a vanilla ``pip install
  engrava`` checkout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from engrava.benchmarks.longmemeval import LongMemEvalQuestion, load_dataset
from engrava.benchmarks.longmemeval.dataset_loader import (
    LongMemEvalSession,
    LongMemEvalTurn,
)
from engrava.benchmarks.longmemeval.evaluate import EvaluationOutcome
from engrava.benchmarks.longmemeval.runner import (
    DEFAULT_TOP_K,
    LongMemEvalResults,
    QuestionResult,
    _aggregate,
    _build_longmemeval_config,
    _build_thought_from_lme_turn,
    _db_uri_for_question,
    _perspective_for_role,
    run_longmemeval,
    run_longmemeval_sync,
)
from engrava.config import DreamingConfig, SearchConfig
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "longmemeval_tiny.json"


_EMBEDDINGS_LOCAL = importlib.util.find_spec("sentence_transformers") is not None
_skip_no_embeddings = pytest.mark.skipif(
    not _EMBEDDINGS_LOCAL,
    reason="sentence_transformers extras not installed; install with 'engrava[embeddings-local]'",
)


@pytest.fixture
def tiny_questions(tmp_path: Path) -> list[LongMemEvalQuestion]:
    """Load the synthesised tiny fixture via the public loader API."""
    cache = tmp_path / "lme-cache"
    cache.mkdir()
    (cache / "longmemeval_oracle.json").write_text(
        FIXTURE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return load_dataset(variant="oracle", cache_dir=cache)


class TestPerspectiveMapping:
    """LongMemEval ``role`` → self-anchored ``perspective`` derivation."""

    def test_user_role_maps_to_percept(self) -> None:
        assert _perspective_for_role("user") == "percept"

    def test_assistant_role_maps_to_utterance(self) -> None:
        assert _perspective_for_role("assistant") == "utterance"

    def test_role_matching_is_case_insensitive(self) -> None:
        assert _perspective_for_role("ASSISTANT") == "utterance"
        assert _perspective_for_role("User") == "percept"

    def test_unknown_role_defaults_to_percept(self) -> None:
        """Defensive — unknown roles get treated as user input."""
        assert _perspective_for_role("system") == "percept"


class TestBuildThoughtFromTurn:
    """Self-anchored metadata shape."""

    def _session(self) -> LongMemEvalSession:
        return LongMemEvalSession(
            session_id="session-0001",
            turns=(
                LongMemEvalTurn(role="user", content="My favourite tea is Earl Grey."),
                LongMemEvalTurn(role="assistant", content="Noted."),
            ),
        )

    def test_user_turn_metadata_is_external(self) -> None:
        """User turn → ``perspective=percept`` + ``source.is_self=False``."""
        session = self._session()
        thought = _build_thought_from_lme_turn(
            session.turns[0],
            question_id="q",
            session=session,
            turn_index=0,
            cycle=0,
        )
        assert thought.metadata["perspective"] == "percept"
        source = thought.metadata["source"]
        assert isinstance(source, dict)
        assert source["is_self"] is False

    def test_assistant_turn_metadata_is_self(self) -> None:
        """Assistant turn → ``perspective=utterance`` + ``source.is_self=True``."""
        session = self._session()
        thought = _build_thought_from_lme_turn(
            session.turns[1],
            question_id="q",
            session=session,
            turn_index=1,
            cycle=1,
        )
        assert thought.metadata["perspective"] == "utterance"
        source = thought.metadata["source"]
        assert isinstance(source, dict)
        assert source["is_self"] is True

    def test_thought_id_namespaces_per_session_and_turn(self) -> None:
        """thought_id must encode question_id + session_id + turn_index."""
        session = self._session()
        thought = _build_thought_from_lme_turn(
            session.turns[0],
            question_id="q-42",
            session=session,
            turn_index=7,
            cycle=0,
        )
        assert thought.thought_id == "lme-q-42-session-0001-t0007"

    def test_thought_carries_default_priority_and_status(self) -> None:
        session = self._session()
        thought = _build_thought_from_lme_turn(
            session.turns[0],
            question_id="q",
            session=session,
            turn_index=0,
            cycle=0,
        )
        assert thought.priority == Priority.P3
        assert thought.lifecycle_status == LifecycleStatus.ACTIVE
        assert thought.thought_type == ThoughtType.OBSERVATION

    def test_metadata_carries_session_turn_lang_content_type(self) -> None:
        session = self._session()
        thought = _build_thought_from_lme_turn(
            session.turns[0],
            question_id="q",
            session=session,
            turn_index=3,
            cycle=2,
        )
        assert thought.metadata["session_id"] == "session-0001"
        assert thought.metadata["turn_index"] == 3
        assert thought.metadata["lang"] == "en"
        assert thought.metadata["content_type"] == "natural_language"

    def test_essence_truncated_to_two_hundred_chars(self) -> None:
        """``essence`` carries at most the first 200 chars of content."""
        long_turn = LongMemEvalTurn(role="user", content="x" * 500)
        session = LongMemEvalSession(session_id="s", turns=(long_turn,))
        thought = _build_thought_from_lme_turn(
            long_turn,
            question_id="q",
            session=session,
            turn_index=0,
            cycle=0,
        )
        assert len(thought.essence) == 200


class TestBuildConfig:
    """Binding (DreamingConfig, SearchConfig) pair for the harness."""

    def test_returns_typed_pair(self) -> None:
        dreaming, search = _build_longmemeval_config()
        assert isinstance(dreaming, DreamingConfig)
        assert isinstance(search, SearchConfig)

    def test_search_disables_reflection_boost(self) -> None:
        """Reflection boost is pinned to 1.0 (matches synthetic AC-8b)."""
        _, search = _build_longmemeval_config()
        assert search.reflection_boost == pytest.approx(1.0)

    def test_dreaming_enables_reflections_and_uses_agglomerative(self) -> None:
        """Mirrors synthetic's clustering choice for small per-question haystacks."""
        dreaming, _ = _build_longmemeval_config()
        assert dreaming.enabled is True
        assert dreaming.gates.enable_reflections is True
        assert dreaming.gates.cluster_algorithm == "agglomerative"


class TestAggregate:
    """Aggregator math + per-type rollup."""

    def test_empty_records_yield_zero(self) -> None:
        result = _aggregate(
            records=[],
            dreaming_enabled=False,
            top_k=5,
            eval_mode="substring",
        )
        assert result.total_questions == 0
        assert result.aggregate_score == 0.0
        assert result.aggregate_raw_signal == 0.0
        assert result.per_type == {}

    def test_mixed_outcomes_average_correctly(self) -> None:
        records = [
            QuestionResult(
                question_id="q1",
                question_type="single-session-recall",
                eval_mode="substring",
                outcome=EvaluationOutcome(score=1.0, raw_signal=1.0),
            ),
            QuestionResult(
                question_id="q2",
                question_type="single-session-recall",
                eval_mode="substring",
                outcome=EvaluationOutcome(score=0.0, raw_signal=0.0),
            ),
            QuestionResult(
                question_id="q3",
                question_type="multi-session",
                eval_mode="substring",
                outcome=EvaluationOutcome(score=1.0, raw_signal=1.0),
            ),
        ]
        result = _aggregate(
            records=records,
            dreaming_enabled=True,
            top_k=5,
            eval_mode="substring",
        )
        assert result.total_questions == 3
        assert result.aggregate_score == pytest.approx(2 / 3)
        assert result.per_type["single-session-recall"] == pytest.approx(0.5)
        assert result.per_type["multi-session"] == pytest.approx(1.0)

    def test_question_results_preserve_input_order(self) -> None:
        records = [
            QuestionResult(
                question_id=f"q{i}",
                question_type="t",
                eval_mode="substring",
                outcome=EvaluationOutcome(score=1.0, raw_signal=1.0),
            )
            for i in range(5)
        ]
        result = _aggregate(
            records=records,
            dreaming_enabled=False,
            top_k=5,
            eval_mode="substring",
        )
        assert [q.question_id for q in result.question_results] == [f"q{i}" for i in range(5)]


class TestRunnerSurface:
    """Surface-level runner contract: error paths + sync wrapper."""

    @_skip_no_embeddings
    async def test_llm_mode_without_judge_raises(
        self,
        tiny_questions: list[LongMemEvalQuestion],
    ) -> None:
        """Passing ``llm`` mode without an ``llm_judge`` argument is a contract error."""
        from engrava.benchmarks.synthetic.evaluate import (
            resolve_embedding_provider_or_exit,
        )

        provider = resolve_embedding_provider_or_exit()
        with pytest.raises(ValueError, match="requires an llm_judge"):
            await run_longmemeval(
                tiny_questions[:1],
                dreaming_enabled=False,
                embedding_provider=provider,
                eval_mode="llm",
            )


@_skip_no_embeddings
class TestRunnerEndToEnd:
    """End-to-end: ingest + retrieve + score over the synthesised fixture."""

    def test_substring_mode_smoke_and_determinism(
        self,
        tiny_questions: list[LongMemEvalQuestion],
    ) -> None:
        """Substring mode runs to completion AND is deterministic across 3 runs.

        Re-running the same input through the same engrava pipeline
        with the same embedding provider must produce byte-identical
        aggregate metrics.
        """
        from engrava.benchmarks.synthetic.evaluate import (
            resolve_embedding_provider_or_exit,
        )

        provider = resolve_embedding_provider_or_exit()
        runs = [
            run_longmemeval_sync(
                tiny_questions,
                dreaming_enabled=False,
                embedding_provider=provider,
                eval_mode="substring",
            )
            for _ in range(3)
        ]
        assert {r.total_questions for r in runs} == {len(tiny_questions)}
        first = runs[0]
        for r in runs[1:]:
            assert r.aggregate_score == first.aggregate_score
            assert r.aggregate_raw_signal == first.aggregate_raw_signal
            assert [qr.outcome for qr in r.question_results] == [
                qr.outcome for qr in first.question_results
            ]

    def test_result_top_k_and_eval_mode_propagate(
        self,
        tiny_questions: list[LongMemEvalQuestion],
    ) -> None:
        """Constructor arguments echo through to the aggregate payload."""
        from engrava.benchmarks.synthetic.evaluate import (
            resolve_embedding_provider_or_exit,
        )

        provider = resolve_embedding_provider_or_exit()
        result = run_longmemeval_sync(
            tiny_questions[:1],
            dreaming_enabled=False,
            embedding_provider=provider,
            eval_mode="substring",
            retrieval_top_k=3,
        )
        assert isinstance(result, LongMemEvalResults)
        assert result.top_k == 3
        assert result.eval_mode == "substring"
        assert result.dreaming_enabled is False


def test_default_top_k_is_documented_value() -> None:
    """Public ``DEFAULT_TOP_K`` must stay aligned with the runner default."""
    assert DEFAULT_TOP_K == 5


class TestDbUriIsolation:
    """``_db_uri_for_question`` guarantees fresh-store-per-question on disk."""

    def test_in_memory_when_dir_is_none(self) -> None:
        assert _db_uri_for_question("anything", None) == ":memory:"

    def test_distinct_uris_per_question(self, tmp_path: Path) -> None:
        uri_a = _db_uri_for_question("q-001", tmp_path)
        uri_b = _db_uri_for_question("q-002", tmp_path)
        assert uri_a != uri_b
        assert "lme-q-001-" in uri_a
        assert "lme-q-002-" in uri_b

    def test_question_id_special_chars_are_sanitised(self, tmp_path: Path) -> None:
        """Slashes / colons / spaces in question_id never escape the directory."""
        uri = _db_uri_for_question("a/b:c d", tmp_path)
        filename = Path(uri).name
        # Sanitiser replaces forbidden chars with underscores; final filename
        # must live exactly under the requested directory.
        assert "/" not in filename
        assert ":" not in filename
        assert " " not in filename
        assert filename.startswith("lme-")
        assert filename.endswith(".sqlite")
        assert Path(uri).parent == tmp_path

    def test_sanitiser_collisions_resolved_by_hash_suffix(
        self,
        tmp_path: Path,
    ) -> None:
        """Distinct question_ids that sanitise to the same prefix get distinct files.

        ``"a/b"``, ``"a:b"`` and ``"a b"`` all sanitise to ``"a_b"``;
        without the hash suffix they would collide on disk and a later
        question would overwrite an earlier one's haystack.
        """
        colliding_ids = ["a/b", "a:b", "a b"]
        uris = {qid: _db_uri_for_question(qid, tmp_path) for qid in colliding_ids}
        # All three URIs must be distinct strings.
        assert len(set(uris.values())) == len(colliding_ids)
        # All three must share the same sanitised prefix but differ in
        # the hash suffix component.
        filenames = {qid: Path(uri).name for qid, uri in uris.items()}
        prefixes = {name.split("-", 2)[1] for name in filenames.values()}
        assert prefixes == {"a_b"}
        digests = {name.removesuffix(".sqlite").split("-")[-1] for name in filenames.values()}
        assert len(digests) == len(colliding_ids)

    def test_uri_is_deterministic_for_the_same_question_id(
        self,
        tmp_path: Path,
    ) -> None:
        """Same id under the same directory yields the same URI (idempotence)."""
        first = _db_uri_for_question("question-7", tmp_path)
        second = _db_uri_for_question("question-7", tmp_path)
        assert first == second

    def test_directory_is_created_lazily(self, tmp_path: Path) -> None:
        nested = tmp_path / "lme-store" / "deeper"
        assert not nested.exists()
        _db_uri_for_question("q", nested)
        assert nested.exists()

    @_skip_no_embeddings
    def test_on_disk_run_uses_separate_files_per_question(
        self,
        tiny_questions: list[LongMemEvalQuestion],
        tmp_path: Path,
    ) -> None:
        """End-to-end check: each question writes to its own SQLite file."""
        from engrava.benchmarks.synthetic.evaluate import (
            resolve_embedding_provider_or_exit,
        )

        provider = resolve_embedding_provider_or_exit()
        db_dir = tmp_path / "lme-isolation"
        run_longmemeval_sync(
            tiny_questions,
            dreaming_enabled=False,
            embedding_provider=provider,
            eval_mode="substring",
            db_path=db_dir,
        )
        files = sorted(f.name for f in db_dir.glob("*.sqlite"))
        # One sqlite file per question; filenames carry the sanitised
        # id prefix plus the hash suffix added by ``_db_uri_for_question``.
        assert len(files) == len(tiny_questions)
        for question in tiny_questions:
            matching = [f for f in files if f.startswith(f"lme-{question.question_id}-")]
            assert len(matching) == 1, (
                f"expected exactly one DB file for {question.question_id!r}, got {matching!r}"
            )

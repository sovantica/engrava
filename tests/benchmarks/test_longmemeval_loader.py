"""Tests for LongMemEval dataset loader (parser + cache + download)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engrava.benchmarks.longmemeval import (
    LongMemEvalQuestion,
    LongMemEvalSession,
    LongMemEvalTurn,
    load_dataset,
)
from engrava.benchmarks.longmemeval.dataset_loader import (
    DatasetDownloadError,
    parse_questions,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "longmemeval_tiny.json"


@pytest.fixture
def fixture_payload() -> object:
    """Return the parsed JSON payload of the synthetic fixture."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Provide an isolated cache directory for loader tests."""
    target = tmp_path / "engrava-cache" / "longmemeval"
    target.mkdir(parents=True)
    return target


@pytest.fixture
def seeded_cache(cache_dir: Path) -> Path:
    """Seed the cache with the synthetic fixture under the oracle filename."""
    target = cache_dir / "longmemeval_oracle.json"
    target.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return cache_dir


class TestParseQuestions:
    """Schema parsing for the upstream JSON payload."""

    def test_parses_top_level_list(self, fixture_payload: object) -> None:
        """Upstream layout: top-level JSON list of questions."""
        questions = list(parse_questions(fixture_payload))
        assert len(questions) == 3
        assert all(isinstance(q, LongMemEvalQuestion) for q in questions)

    def test_question_fields_populated(self, fixture_payload: object) -> None:
        """Every documented schema field round-trips into the value-object."""
        questions = list(parse_questions(fixture_payload))
        first = questions[0]
        assert first.question_id == "fixture-q-001"
        assert first.question_type == "single-session-recall"
        assert first.question == "What pet did the user adopt?"
        assert first.answer == "tabby cat named Pepper"
        assert first.question_date == "2026-01-15"
        assert first.answer_session_ids == ("session-0001",)

    def test_haystack_sessions_are_typed(self, fixture_payload: object) -> None:
        """Sessions are parsed as ``LongMemEvalSession`` with typed turns."""
        questions = list(parse_questions(fixture_payload))
        first_session = questions[0].haystack_sessions[0]
        assert isinstance(first_session, LongMemEvalSession)
        assert first_session.session_id == "session-0001"
        assert len(first_session.turns) == 3
        assert all(isinstance(t, LongMemEvalTurn) for t in first_session.turns)
        assert first_session.turns[0].role == "user"
        assert "Pepper" in first_session.turns[0].content

    def test_parses_questions_under_questions_key(self) -> None:
        """Legacy layout: ``{"questions": [...]}`` wrapper."""
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        wrapped = {"questions": payload}
        questions = list(parse_questions(wrapped))
        assert len(questions) == 3

    def test_session_list_without_session_id_falls_back_to_index(self) -> None:
        """Plain-list session payload synthesises ``session-<index>`` id."""
        payload = [
            {
                "question_id": "q",
                "question_type": "t",
                "question": "?",
                "answer": "a",
                "question_date": "2026-01-01",
                "haystack_sessions": [[{"role": "user", "content": "hi"}]],
                "answer_session_ids": [],
            },
        ]
        question = next(iter(parse_questions(payload)))
        assert question.haystack_sessions[0].session_id == "session-0"

    def test_rejects_unrecognised_top_level_shape(self) -> None:
        """A bare integer at the top level fails fast with a clear message."""
        with pytest.raises(DatasetDownloadError, match="unrecognised LongMemEval payload"):
            list(parse_questions(42))

    def test_rejects_non_list_questions_key(self) -> None:
        """``{"questions": "not-a-list"}`` raises ``DatasetDownloadError``."""
        with pytest.raises(DatasetDownloadError, match="under 'questions' key"):
            list(parse_questions({"questions": "not-a-list"}))

    def test_rejects_non_list_haystack_sessions(self) -> None:
        """``haystack_sessions`` MUST be a list at parse time."""
        payload = [
            {
                "question_id": "q",
                "question_type": "t",
                "question": "?",
                "answer": "a",
                "question_date": "2026-01-01",
                "haystack_sessions": "not-a-list",
                "answer_session_ids": [],
            },
        ]
        with pytest.raises(DatasetDownloadError, match="haystack_sessions must be a list"):
            list(parse_questions(payload))


class TestLoadDataset:
    """``load_dataset`` end-to-end caching behaviour."""

    def test_loads_from_seeded_cache_without_network(self, seeded_cache: Path) -> None:
        """When the cache file is present, no download is attempted."""
        questions = load_dataset(variant="oracle", cache_dir=seeded_cache)
        assert len(questions) == 3
        assert questions[0].question_id == "fixture-q-001"

    def test_repeated_load_reuses_cache(self, seeded_cache: Path) -> None:
        """Second invocation returns identical structured content."""
        first = load_dataset(variant="oracle", cache_dir=seeded_cache)
        second = load_dataset(variant="oracle", cache_dir=seeded_cache)
        assert [q.question_id for q in first] == [q.question_id for q in second]

    def test_rejects_unknown_variant(self, cache_dir: Path) -> None:
        """Unknown variants fail fast without touching the network."""
        with pytest.raises(DatasetDownloadError, match="unknown LongMemEval variant"):
            load_dataset(variant="huge", cache_dir=cache_dir)

    def test_rejects_corrupt_cache(self, cache_dir: Path) -> None:
        """A cached file that is not valid JSON raises with a clear message."""
        bad = cache_dir / "longmemeval_oracle.json"
        bad.write_text("{ not json", encoding="utf-8")
        with pytest.raises(DatasetDownloadError, match="failed to parse"):
            load_dataset(variant="oracle", cache_dir=cache_dir)

    def test_force_download_attempts_network(
        self,
        seeded_cache: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``force_download=True`` calls the download helper even with cache present."""
        called: list[str] = []

        def fake_download(url: str, destination: Path) -> None:
            called.append(url)
            destination.write_text(
                FIXTURE_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        monkeypatch.setattr(
            "engrava.benchmarks.longmemeval.dataset_loader._download_to",
            fake_download,
        )
        load_dataset(variant="oracle", force_download=True, cache_dir=seeded_cache)
        assert len(called) == 1
        assert "longmemeval_oracle.json" in called[0]

    def test_download_failure_raises_actionable_error(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Download errors surface a hint to the manual-download instructions."""
        import urllib.error

        unreachable = urllib.error.URLError("network unreachable")

        def failing_urlopen(*_args: object, **_kwargs: object) -> object:
            raise unreachable

        monkeypatch.setattr(
            "engrava.benchmarks.longmemeval.dataset_loader.urllib.request.urlopen",
            failing_urlopen,
        )
        with pytest.raises(DatasetDownloadError, match=r"README\.md for manual"):
            load_dataset(variant="oracle", cache_dir=cache_dir)

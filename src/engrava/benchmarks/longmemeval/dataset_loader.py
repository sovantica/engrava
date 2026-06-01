"""LongMemEval dataset loader — user-download from upstream HuggingFace.

The dataset is NOT bundled with engrava. On first invocation the
loader downloads the requested variant from the upstream HuggingFace
mirror to a local cache directory (``~/.engrava/benchmarks/longmemeval/``)
and parses it into typed value-objects. Subsequent invocations re-use
the cached copy.

Dataset schema (per upstream documentation):

* ``question_id``      — unique question identifier (str)
* ``question_type``    — taxonomy label (``single-session-recall``,
                         ``multi-session``, etc.)
* ``question``         — the natural-language question text
* ``answer``           — the expected answer text (substring-checkable)
* ``question_date``    — ISO-8601 date string the question was asked
* ``haystack_sessions``— ordered list of past conversation sessions; each
                         session is a list of turns; each turn carries
                         ``role`` (``user`` / ``assistant``) and
                         ``content``
* ``answer_session_ids``— indices (or session_id strings) marking which
                         haystack sessions actually contain the answer

This loader normalises the upstream JSON into immutable Pydantic value-
objects. Test fixtures use the SAME schema but are synthesised in-repo
(see ``tests/benchmarks/fixtures/longmemeval_tiny.json``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "LOCAL_CACHE_DIR",
    "LONGMEMEVAL_BASE",
    "LONGMEMEVAL_FILES",
    "DatasetDownloadError",
    "LongMemEvalQuestion",
    "LongMemEvalSession",
    "LongMemEvalTurn",
    "load_dataset",
    "parse_questions",
]


LONGMEMEVAL_BASE: Final = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
)
LONGMEMEVAL_FILES: Final[dict[str, str]] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
LOCAL_CACHE_DIR: Final = Path.home() / ".engrava" / "benchmarks" / "longmemeval"


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded or parsed."""


class LongMemEvalTurn(BaseModel):
    """One conversational turn inside a haystack session.

    Attributes:
        role: Either ``user`` or ``assistant``.
        content: Natural-language content of the turn.

    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    role: str
    content: str


class LongMemEvalSession(BaseModel):
    """An ordered list of turns from a single past conversation.

    Attributes:
        session_id: Stable identifier for the session within the
            haystack (often a date string or numeric index).
        turns: Ordered conversational turns.

    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    session_id: str
    turns: tuple[LongMemEvalTurn, ...]


class LongMemEvalQuestion(BaseModel):
    """One LongMemEval evaluation instance.

    Attributes:
        question_id: Unique identifier for this question.
        question_type: Taxonomy label (e.g. ``single-session-recall``).
        question: Natural-language question text.
        answer: Expected answer text; the substring evaluator checks
            whether retrieved content contains this string.
        question_date: ISO-8601 date string the question was asked
            (used to model temporal recall scenarios).
        haystack_sessions: All past conversation sessions available
            for retrieval, in chronological order.
        answer_session_ids: Subset of ``haystack_sessions`` session_ids
            that actually carry the answer (oracle ground truth).

    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_sessions: tuple[LongMemEvalSession, ...]
    answer_session_ids: tuple[str, ...] = Field(default_factory=tuple)


def load_dataset(
    variant: str = "oracle",
    *,
    force_download: bool = False,
    cache_dir: Path | None = None,
) -> list[LongMemEvalQuestion]:
    """Load (and cache) the requested LongMemEval variant.

    Args:
        variant: One of ``oracle`` / ``s`` / ``m``. ``oracle`` is the
            smallest variant and the default for smoke runs.
        force_download: When ``True`` re-download even if a cached
            copy exists. Useful when upstream re-cleans the dataset.
        cache_dir: Override the cache root. Defaults to
            ``~/.engrava/benchmarks/longmemeval/``. Tests inject a
            ``tmp_path`` here so they never touch the user's real
            home directory.

    Returns:
        A list of ``LongMemEvalQuestion`` value-objects.

    Raises:
        DatasetDownloadError: When the requested variant is unknown,
            the upstream download fails, or the file is not valid JSON.

    """
    if variant not in LONGMEMEVAL_FILES:
        msg = (
            f"unknown LongMemEval variant {variant!r}; expected one of {sorted(LONGMEMEVAL_FILES)}"
        )
        raise DatasetDownloadError(msg)

    root = cache_dir if cache_dir is not None else LOCAL_CACHE_DIR
    cache = root / LONGMEMEVAL_FILES[variant]
    if not cache.exists() or force_download:
        url = f"{LONGMEMEVAL_BASE}/{LONGMEMEVAL_FILES[variant]}"
        _download_to(url, cache)
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"failed to parse cached LongMemEval file {cache}: {exc}"
        raise DatasetDownloadError(msg) from exc
    return list(parse_questions(payload))


def parse_questions(payload: object) -> Iterable[LongMemEvalQuestion]:
    """Yield ``LongMemEvalQuestion`` instances from a parsed JSON payload.

    Accepts the upstream layout (top-level list of question dicts)
    plus the slightly older variant where the questions live under a
    ``"questions"`` key. Anything else raises ``DatasetDownloadError``.

    Args:
        payload: Already-parsed JSON (list or dict).

    Returns:
        Iterable of typed ``LongMemEvalQuestion`` instances.

    Raises:
        DatasetDownloadError: When the payload shape is not recognised.

    """
    if isinstance(payload, list):
        raw_items: list[object] = payload
    elif isinstance(payload, dict) and "questions" in payload:
        nested = payload["questions"]
        if not isinstance(nested, list):
            msg = "expected a JSON list under 'questions' key"
            raise DatasetDownloadError(msg)
        raw_items = nested
    else:
        msg = (
            "unrecognised LongMemEval payload shape — expected top-level list "
            "or dict with 'questions' key"
        )
        raise DatasetDownloadError(msg)

    for item in raw_items:
        yield _parse_question(item)


def _parse_question(item: object) -> LongMemEvalQuestion:
    if not isinstance(item, dict):
        msg = f"expected JSON object per question, got {type(item).__name__}"
        raise DatasetDownloadError(msg)
    raw_sessions = item.get("haystack_sessions", [])
    if not isinstance(raw_sessions, list):
        msg = "haystack_sessions must be a list"
        raise DatasetDownloadError(msg)
    sessions = tuple(_parse_session(idx, sess) for idx, sess in enumerate(raw_sessions))
    answer_session_ids_raw = item.get("answer_session_ids", [])
    if not isinstance(answer_session_ids_raw, list):
        msg = "answer_session_ids must be a list"
        raise DatasetDownloadError(msg)
    return LongMemEvalQuestion(
        question_id=str(item.get("question_id", "")),
        question_type=str(item.get("question_type", "")),
        question=str(item.get("question", "")),
        answer=str(item.get("answer", "")),
        question_date=str(item.get("question_date", "")),
        haystack_sessions=sessions,
        answer_session_ids=tuple(str(s) for s in answer_session_ids_raw),
    )


def _parse_session(index: int, raw: object) -> LongMemEvalSession:
    """Parse one haystack session.

    Upstream emits sessions either as plain lists of turns (in which
    case the position-index becomes the session_id) or as dicts with
    explicit ``session_id`` + ``turns`` keys. Accept both.
    """
    if isinstance(raw, list):
        return LongMemEvalSession(
            session_id=f"session-{index}",
            turns=tuple(_parse_turn(t) for t in raw),
        )
    if isinstance(raw, dict):
        turns_raw = raw.get("turns", [])
        if not isinstance(turns_raw, list):
            msg = "session.turns must be a list"
            raise DatasetDownloadError(msg)
        return LongMemEvalSession(
            session_id=str(raw.get("session_id", f"session-{index}")),
            turns=tuple(_parse_turn(t) for t in turns_raw),
        )
    msg = f"expected JSON list or object per session, got {type(raw).__name__}"
    raise DatasetDownloadError(msg)


def _parse_turn(raw: object) -> LongMemEvalTurn:
    if not isinstance(raw, dict):
        msg = f"expected JSON object per turn, got {type(raw).__name__}"
        raise DatasetDownloadError(msg)
    return LongMemEvalTurn(
        role=str(raw.get("role", "user")),
        content=str(raw.get("content", "")),
    )


def _download_to(url: str, destination: Path) -> None:
    """Stream ``url`` into ``destination`` (parents created as needed)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 — pinned trusted HF URL
            data = response.read()
    except (urllib.error.URLError, OSError) as exc:
        msg = (
            f"failed to download LongMemEval dataset from {url}: {exc}. "
            f"See benchmarks/longmemeval/README.md for manual download instructions."
        )
        raise DatasetDownloadError(msg) from exc
    destination.write_bytes(data)

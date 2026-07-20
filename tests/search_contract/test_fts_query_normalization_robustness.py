"""Regression guard for FTS query-normalization robustness.

This suite locks two invariants of the bare (natural-language) FTS5
normalization path so a future edit cannot silently reintroduce an invalid
``MATCH`` for ordinary user input:

* **Bare trailing-punctuation is stripped, not passed through.** A bare
  natural-language question that merely *contains* a literal ``?`` (or ``.``,
  ``!``, ``;``, ``,``, ``:``) — the everyday shape ``where can I learn to
  drive?`` — is normalized on the *primary* path: boundary punctuation is
  stripped before the term reaches ``MATCH``, so the query never raises and the
  read-only :attr:`SqliteEngravaCore.fts_match_failure_count` stays ``0``. This
  is distinct from the curated *quoted* ``"x"?`` case (a balanced quoted phrase
  that legitimately classifies as expert and is rescued by the bare retry with
  ``count += 1``); here the whole point is that the bare class never touches the
  fallback at all.

* **The bare path is always valid FTS5, even for exposed operators.** FTS5
  reads an uppercase ``AND``/``OR``/``NOT`` as a boolean *operator*, never a
  term. When such a keyword is exposed as a whole ``*``-delimited segment of a
  sanitized fragment (``NOT`` alone, ``field*NOT``, or a dangling operator at
  the tail of an expert query), the bare normalizer now phrase-quotes it
  (``"NOT"``) so the emitted ``MATCH`` is valid and still matches the same
  case-folded documents. This closes the one class that could otherwise make
  *both* the primary and the sanitizing bare fallback raise — the branch
  :meth:`SqliteEngravaCore.search_fts` documents as effectively unreachable.

Everything here runs against a real in-memory ``thought_fts`` index; there is
no network, model, or dataset dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import aiosqlite
import pytest

from engrava import SqliteEngravaCore
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import (
    _normalize_fts_query,
    _normalize_fts_query_bare,
    _query_is_expert_syntax,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


# ---------------------------------------------------------------------------
# A store whose corpus shares vocabulary with every content-bearing input
# ---------------------------------------------------------------------------
# The single stored thought carries every content word used by the bare
# natural-language corpus (so those queries return real BM25 hits) and also the
# standalone words ``and`` / ``or`` / ``not`` / ``field`` (so a phrase-quoted
# operator such as ``"NOT"`` still matches a document). A topical decoy keeps
# retrieval non-trivial.

_STORE_CONTENT = (
    "a new sister learns to drive her car while drivers learn to drive and check "
    "the status ok what happened then tell me what you will not do or skip about "
    "the field name body forum mark final answer"
)


def _thought(thought_id: str, essence: str, content: str) -> ThoughtRecord:
    """Build a stored thought for this suite.

    Args:
        thought_id: Stable identifier used to assert retrieval.
        essence: Short summary line, indexed by FTS5.
        content: Full text, indexed by FTS5.

    Returns:
        A fully populated :class:`ThoughtRecord` ready for ``create_thought``.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


@pytest.fixture
async def bare_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return an FTS store whose corpus shares the input vocabulary.

    Yields:
        A :class:`SqliteEngravaCore` holding one thought that shares every
        content word (and the literal words ``and``/``or``/``not``) with the
        corpora below, plus one topical decoy.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    await store.create_thought(_thought("t-main", "sister field forum", _STORE_CONTENT))
    await store.create_thought(
        _thought("t-decoy", "cooking note", "the lasagna recipe uses three cheeses")
    )
    yield store
    await conn.close()


@pytest.fixture
async def fts5_probe() -> AsyncIterator[Callable[[str], Awaitable[None]]]:
    """Return a callable that executes a raw FTS5 ``MATCH`` expression.

    The callable raises :class:`sqlite3.OperationalError` when the expression is
    invalid FTS5, so a test can assert a normalized query parses without relying
    on any store internals or on whether it happens to match a document.

    Yields:
        An awaitable ``run(expr)`` that executes ``probe MATCH expr`` against a
        throwaway in-memory FTS5 table.
    """
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE VIRTUAL TABLE probe USING fts5(body)")

    async def run(expr: str) -> None:
        cursor = await conn.execute("SELECT rowid FROM probe WHERE probe MATCH ?", (expr,))
        await cursor.fetchall()

    yield run
    await conn.close()


# ---------------------------------------------------------------------------
# Corpus 1 — bare natural-language queries with boundary punctuation
# ---------------------------------------------------------------------------


class BareCase(NamedTuple):
    """A bare natural-language query and whether it can return hits.

    Args:
        query: The raw user-facing query driven through ``search_fts``.
        expect_hits: ``True`` when the query holds at least one indexable term
            shared with the corpus, so a valid ``MATCH`` must return a result.
            ``False`` only when the query is pure boundary punctuation /
            whitespace and therefore legitimately normalizes to no term.
    """

    query: str
    expect_hits: bool


_BARE_NL_CORPUS: tuple[BareCase, ...] = (
    # The canonical bare inputs: a literal '?' (or ':'-then-'?') is stripped as
    # boundary punctuation before the term reaches MATCH.
    BareCase("where can new drivers learn to drive?", expect_hits=True),
    BareCase("sister's car", expect_hits=True),
    BareCase("REQ-FUNC-12: status?", expect_hits=True),
    # Siblings: an inline comma before the trailing '?', and a bare 'word:' that
    # is not a whitelisted column filter.
    BareCase("what happened, then?", expect_hits=True),
    BareCase("status: ok?", expect_hits=True),
    # The same content phrase closed by each trailing sentence mark in turn.
    BareCase("new drivers learn to drive.", expect_hits=True),
    BareCase("new drivers learn to drive!", expect_hits=True),
    BareCase("new drivers learn to drive;", expect_hits=True),
    # Degenerate: nothing but boundary punctuation / whitespace, so no indexable
    # term survives and an empty result is the correct, non-erroring outcome.
    BareCase("?", expect_hits=False),
    BareCase("??", expect_hits=False),
    BareCase("  ?  ", expect_hits=False),
)


class TestBareTrailingPunctuationRegression:
    """Bare '?'/punctuation NL queries stay safe on the primary path."""

    @pytest.mark.parametrize("case", _BARE_NL_CORPUS, ids=lambda case: case.query)
    def test_bare_nl_stays_on_the_primary_path(self, case: BareCase) -> None:
        """A bare punctuation-bearing NL query never classifies as expert syntax."""
        assert _query_is_expert_syntax(case.query) is False

    @pytest.mark.parametrize("case", _BARE_NL_CORPUS, ids=lambda case: case.query)
    async def test_bare_nl_search_is_safe_and_non_degenerate(
        self,
        case: BareCase,
        bare_store: SqliteEngravaCore,
    ) -> None:
        """Each bare NL query (a) never raises, (b) leaves the counter at 0, (c) is non-degenerate.

        (a) ``search_fts`` returns a list rather than propagating an
        ``OperationalError``; (b) the primary-``MATCH`` failure counter does not
        move, proving the query took the primary path and never the fallback;
        (c) a query with an indexable term returns at least one hit, while an
        empty result is asserted *only* when the query genuinely normalizes to
        no term (the pure ``?`` cases).
        """
        before = bare_store.fts_match_failure_count
        results = await bare_store.search_fts(case.query, top_k=10)  # (a) must not raise

        assert isinstance(results, list)
        assert bare_store.fts_match_failure_count == before  # (b) counter unchanged

        if case.expect_hits:  # (c) valid, non-degenerate MATCH
            assert results, f"expected hits for bare NL query {case.query!r}"
        else:
            assert results == []
            assert _normalize_fts_query(case.query) == ""


class TestBareCorpusFailureCounterStaysZero:
    """The failure counter never increments across the whole bare/NL corpus."""

    async def test_counter_stays_zero_across_bare_corpus(
        self,
        bare_store: SqliteEngravaCore,
    ) -> None:
        """No bare/NL query pushes work onto the sanitizing fallback.

        This is the standing guard: a future change that routes a bare
        natural-language query onto the fallback (for instance by widening the
        expert classifier) would move the counter off ``0`` and fail here.
        """
        assert bare_store.fts_match_failure_count == 0
        for case in _BARE_NL_CORPUS:
            await bare_store.search_fts(case.query, top_k=10)
        assert bare_store.fts_match_failure_count == 0


# ---------------------------------------------------------------------------
# Corpus 2 — bare fragments that expose an uppercase FTS5 boolean operator
# ---------------------------------------------------------------------------


class OperatorCase(NamedTuple):
    """A query whose bare normalization exposes an uppercase FTS5 operator.

    Args:
        query: The raw user-facing query driven through ``search_fts``.
        bare_normalized: The exact bare normalization; the exposed uppercase
            ``AND``/``OR``/``NOT`` is phrase-quoted so FTS5 reads it as a term.
        primary_recovers: ``True`` when the *primary* ``MATCH`` is invalid (a
            standalone or dangling expert operator), so the bare fallback runs
            and the counter increments by one. ``False`` when the query is
            bare-classified and its (already phrase-quoted) primary ``MATCH`` is
            valid, so the counter stays put.
        expect_hits: ``True`` when the phrase-quoted term matches the corpus.
    """

    query: str
    bare_normalized: str
    primary_recovers: bool
    expect_hits: bool


_EXPOSED_OPERATOR_CORPUS: tuple[OperatorCase, ...] = (
    # Standalone uppercase operators: expert-classified, invalid primary, then
    # rescued by the phrase-quoted bare fallback.
    OperatorCase("NOT", '"NOT"', primary_recovers=True, expect_hits=True),
    OperatorCase("AND", '"AND"', primary_recovers=True, expect_hits=True),
    OperatorCase("OR", '"OR"', primary_recovers=True, expect_hits=True),
    # A dangling operator at the tail of an otherwise natural sentence.
    OperatorCase(
        "tell me what you will NOT",
        'tell OR me OR what OR you OR will OR "NOT"',
        primary_recovers=True,
        expect_hits=True,
    ),
    # An operator exposed by a '*' prefix boundary inside a single token: the
    # token is bare-classified, so its phrase-quoted normalization is already
    # the valid primary MATCH (no fallback, no counter movement).
    OperatorCase("field*NOT", '"field*NOT"', primary_recovers=False, expect_hits=False),
    OperatorCase("not*NOT", '"not*NOT"', primary_recovers=False, expect_hits=False),
)


class TestExposedBooleanOperatorNormalization:
    """An exposed uppercase AND/OR/NOT is phrase-quoted into a valid bare term."""

    @pytest.mark.parametrize("case", _EXPOSED_OPERATOR_CORPUS, ids=lambda case: case.query)
    def test_bare_path_quotes_the_exposed_operator(self, case: OperatorCase) -> None:
        """The bare normalizer emits the operator phrase-quoted, not as a keyword."""
        assert _normalize_fts_query_bare(case.query) == case.bare_normalized

    @pytest.mark.parametrize("case", _EXPOSED_OPERATOR_CORPUS, ids=lambda case: case.query)
    async def test_bare_normalization_is_valid_fts5(
        self,
        case: OperatorCase,
        fts5_probe: Callable[[str], Awaitable[None]],
    ) -> None:
        """The real bare-normalizer output parses as valid FTS5 (never a syntax error).

        This is the discriminating check for the always-valid bare-path
        invariant: without operator neutralization the output would be a bare
        ``NOT`` / ``field*NOT`` and this ``MATCH`` would raise.
        """
        await fts5_probe(_normalize_fts_query_bare(case.query))

    @pytest.mark.parametrize("case", _EXPOSED_OPERATOR_CORPUS, ids=lambda case: case.query)
    async def test_search_recovers_without_reaching_the_second_branch(
        self,
        case: OperatorCase,
        bare_store: SqliteEngravaCore,
    ) -> None:
        """``search_fts`` recovers to a valid MATCH; the doubly-degenerate branch never fires.

        A ``primary_recovers`` case increments the counter exactly once (the
        by-design first-branch recovery) and, crucially, still returns real hits
        — proving the bare fallback executed a *valid* ``MATCH`` rather than
        falling through the second ``OperationalError`` guard to an empty list.
        A non-recovering case keeps the counter at its prior value because its
        phrase-quoted primary ``MATCH`` is already valid.
        """
        before = bare_store.fts_match_failure_count
        results = await bare_store.search_fts(case.query, top_k=10)
        delta = bare_store.fts_match_failure_count - before

        assert isinstance(results, list)
        assert delta == (1 if case.primary_recovers else 0)
        if case.expect_hits:
            assert results, f"phrase-quoted operator returned no hits for {case.query!r}"

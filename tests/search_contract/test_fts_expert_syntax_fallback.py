"""Safety-invariant suite for FTS5 expert-syntax classification and fallback.

This suite pins the contract introduced to close a silent hybrid-search
degradation: a natural-language query that merely *contains* double quotes
(scare-quotes around a UI label) used to be misclassified as expert phrase
syntax, so hazardous punctuation adjacent to a quote (``"forum"?``) survived
into the FTS5 ``MATCH`` and raised an ``OperationalError``. The guard then
silently returned no FTS results and hybrid search degraded to vector-only with
no signal to the caller.

The fix has three parts, each pinned here:

* **Classification tightening** (:func:`_query_is_expert_syntax`) — a query is
  expert only for a *deliberate* construct (a balanced quoted phrase wrapping a
  token, a standalone ``AND``/``OR``/``NOT``, or an ``essence:``/``content:``
  filter). An odd/unbalanced quote count is always bare.
* **Validate-and-fallback** (:meth:`SqliteEngravaCore.search_fts`) — when the
  primary ``MATCH`` raises, the *original* query is re-normalized through the
  always-valid bare path and the ``MATCH`` is retried once, instead of silently
  degrading. The bare path is genuinely always-valid: its sanitizer collapses
  wildcards to FTS5-legal prefix markers, so a consecutive/leading-``*`` shape
  such as ``foo**`` can never make *both* the primary and the fallback raise.
* **Failure surfacing** — every primary-``MATCH`` failure increments the
  read-only :attr:`SqliteEngravaCore.fts_match_failure_count`.

The standing safety invariant (:class:`TestSafetyInvariant`) drives a broad
adversarial corpus through the real ``thought_fts`` index and asserts no query
ever raises or silently degrades due to an invalid ``MATCH``. Its discriminating
power is verified by :class:`TestDiscriminatingPower`: reverting the
classification tightening flips a scare-quote anchor's failure delta, and the
bare fallback is what turns an invalid expert ``MATCH`` into real hits.
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
    _query_is_expert_syntax,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


# ---------------------------------------------------------------------------
# A store whose corpus contains the reported-input vocabulary
# ---------------------------------------------------------------------------
# The reported natural-language query mixes function words with the content
# words "field", "name", "body", "forum", "mark", "final", "answer". This store
# holds a thought carrying that vocabulary so the sanitizing fallback produces
# real BM25 hits (AC3), plus a couple of decoys so retrieval is non-trivial.

_REPORTED_CONTENT = (
    "field name between body and forum mark your final answer about machine learning"
)


def _thought(thought_id: str, essence: str, content: str) -> ThoughtRecord:
    """Build a stored thought for the fallback suite.

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
async def reported_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return an FTS store whose corpus matches the reported-input vocabulary.

    Yields:
        A :class:`SqliteEngravaCore` holding one thought that shares the
        reported query's content words plus two topical decoys.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    await store.create_thought(_thought("t-reported", "field name forum", _REPORTED_CONTENT))
    await store.create_thought(
        _thought("t-decoy-x", "unrelated phrase", "x a b update attributes here")
    )
    await store.create_thought(
        _thought("t-decoy-cook", "cooking note", "the lasagna recipe uses three cheeses")
    )
    yield store
    await conn.close()


# ---------------------------------------------------------------------------
# The concrete reported input
# ---------------------------------------------------------------------------

_REPORTED_QUERY = 'field name between "body" and "forum"? Mark your final answer'


# ---------------------------------------------------------------------------
# AC5 — genuine-expert parity golden set
# ---------------------------------------------------------------------------
# For each real expert query the classification MUST be expert and the
# normalized output MUST be byte-identical to release/v0.6.0 (which normalized
# expert queries token-by-token, joined with spaces, unchanged by this fix).
# INCLUDES the exact column-filter + phrase queries whose column filter a prior
# rewrite dropped — the regression this fix must not cause.

_EXPERT_PARITY: dict[str, str] = {
    'essence:"a b"': 'essence:"a b"',
    'content:"machine learning"': 'content:"machine learning"',
    "content:foo AND essence:bar": "content:foo AND essence:bar",
    "cats AND dogs": "cats AND dogs",
    '"machine learning" AND relevant': '"machine learning" AND relevant',
}


# ---------------------------------------------------------------------------
# AC2 — curated, human-readable adversarial table
# ---------------------------------------------------------------------------
# (query, expected_expert). Every row must also drive a valid MATCH (directly or
# via the bare fallback), asserted in TestCuratedAdversarialTable.

_CURATED_TABLE: tuple[tuple[str, bool], ...] = (
    ('"x"?', True),  # balanced phrase + trailing hazard -> expert, fallback rescues
    ('"a" and "b".', True),  # two balanced phrases + trailing hazard -> expert
    ('("Update Attributes")', True),  # parenthesised balanced phrase -> expert
    ("l'école", False),  # French elision, no deliberate construct -> bare
    ("REQ-FUNC*", False),  # hyphenated wildcard identifier -> bare
    ('he said "just one quote', False),  # odd/unbalanced quote -> always bare
    ('"machine learning"', True),  # genuine phrase query -> expert
    ("cats AND dogs", True),  # genuine boolean query -> expert
    ("content:memory", True),  # genuine field-filter query -> expert
)


# ---------------------------------------------------------------------------
# AC1 — broad adversarial fuzz corpus (deterministic, enumerated generator)
# ---------------------------------------------------------------------------


class FuzzCase(NamedTuple):
    """One adversarial query and its expected fallback behaviour.

    Args:
        query: The raw user-facing query string driven through ``search_fts``.
        expect_failure_delta: The expected increment of
            ``fts_match_failure_count`` for this case: ``0`` when the primary
            ``MATCH`` is valid, ``1`` when the primary ``MATCH`` is invalid and
            the bare fallback executes.
        expect_hits: When ``True``, the case shares vocabulary with
            :data:`_REPORTED_CONTENT`, so a correctly-executed query (primary or
            fallback) must return at least one hit. Used as the fallback anchor:
            without the fallback these invalid-primary cases return nothing.
    """

    query: str
    expect_failure_delta: int
    expect_hits: bool


# Content words present in the reported store (so some cases produce real hits)
# interleaved with function words. Sentence punctuation is appended per token in
# the decorated variants below.
_NL_WORDS = ("field", "name", "forum", "body", "answer", "mark", "final")
_FUNCTION_WORDS = ("what", "did", "about", "the", "your", "between")
_SENTENCE_PUNCT = (".", "?", "!", ",", ":", ";")


def _build_group_a() -> list[FuzzCase]:
    """Build the valid-primary partition of the fuzz corpus (delta 0).

    Every case here classifies to a valid FTS5 ``MATCH`` under the fix — either
    a bare, fully-sanitized query (odd/zero quote count, hazardous punctuation,
    wildcards, hyphens, unicode, parentheses) or a well-formed genuine-expert
    query (balanced clean phrase, boolean, field filter). None should ever
    exercise the fallback, so each expects a failure delta of ``0``.

    Returns:
        The Group-A fuzz cases.
    """
    cases: list[FuzzCase] = []

    def add(query: str) -> None:
        cases.append(FuzzCase(query, expect_failure_delta=0, expect_hits=False))

    # Plain natural language (zero quotes).
    for content in _NL_WORDS:
        add(f"what did I say about the {content}")
    add(" ".join((*_FUNCTION_WORDS, *_NL_WORDS)))

    # Sentence punctuation appended to a content token (each mark exercised).
    for punct in _SENTENCE_PUNCT:
        add(f"the {_NL_WORDS[0]}{punct} and {_NL_WORDS[1]}{punct} here")

    # Odd/unbalanced scare-quotes (1 or 3 quotes) inside natural-language prose.
    add('he said "run away and never come back')
    add('the field is "forum without a closing mark')
    add('"unterminated leading phrase about the body')
    add('a "b "c three loose quotes about the forum')
    add('"""triple quoted noise')

    # Leading / trailing sentence punctuation on the whole query.
    add(". forum body answer ?")
    add("??? mark your final answer")
    add("...field name...")

    # Wildcards, hyphens, apostrophes, unicode, parentheses (all bare).
    add("foru* mark*")
    add("REQ-FUNC well-known field-name")
    add("l'école sister's don't")
    add("café déjà vu naïve 😀🚀 中文 查询")
    add("(forum body) grouped tokens")
    add("$5 ##tag @handle 12:30 http://example.com/path?q=1")

    # Degenerate wildcard shapes (the class WS-170 AC1 promised to cover). Each
    # is bare and, under the wildcard-collapsing sanitizer, produces a valid
    # primary MATCH (delta 0) even though a raw ``**``/leading-``*`` fragment is
    # an FTS5 syntax error: consecutive trailing stars, token-internal doubled
    # and tripled stars, multiple internal runs, a leading star, a mix of
    # trailing- and leading-star tokens, a standalone star, and a star pressed
    # against a stray quote / sentence punctuation.
    add("forum** body**")
    add("foo***bar internal")
    add("a**b**c crossing")
    add("*forum leading")
    add("field* *name mixed")
    add("just * a standalone")
    add('foo**" forum body')
    add("mark**? final answer")
    add("...field**...")

    # Empty / whitespace-only.
    add("")
    add("   ")
    add("\t\n  \t")

    # Genuine, well-formed expert queries (balanced clean phrase / boolean /
    # field filter) — expert classification but a valid primary MATCH.
    add('"machine learning"')
    add('"final answer"')
    add("forum AND body")
    add("forum OR body")
    add("forum NOT missing")
    add("content:forum")
    add("essence:body")
    add("content:forum AND essence:body")
    add('"final answer" AND forum')
    add('("machine learning")')
    # Lowercase and/or/not are ordinary words, not operators -> bare.
    add("forum and body or answer not missing")

    return cases


def _build_group_b() -> list[FuzzCase]:
    """Build the invalid-primary partition of the fuzz corpus (delta 1).

    Every case classifies as expert (a balanced quoted phrase) yet carries a
    hazardous character adjacent to the closing quote, so the primary ``MATCH``
    is invalid FTS5 and the bare fallback executes exactly once (delta ``1``).
    The ``expect_hits`` cases share the reported vocabulary and therefore must
    return real hits *through the fallback* — the fallback anchor.

    Returns:
        The Group-B fuzz cases.
    """
    return [
        FuzzCase('"x"?', expect_failure_delta=1, expect_hits=False),
        FuzzCase('"forum"?', expect_failure_delta=1, expect_hits=True),
        FuzzCase('"body"!', expect_failure_delta=1, expect_hits=True),
        FuzzCase('"a" and "b".', expect_failure_delta=1, expect_hits=False),
        FuzzCase('content:"x"?', expect_failure_delta=1, expect_hits=False),
        FuzzCase('essence:"a"?', expect_failure_delta=1, expect_hits=False),
        # Doubly-degenerate: a balanced quote span of pure punctuation classifies
        # expert, the primary MATCH is invalid, AND the bare fallback sanitizes to
        # the empty string — so the fallback short-circuits to no hits (still
        # counted, never raising).
        FuzzCase('"?"?', expect_failure_delta=1, expect_hits=False),
        # A balanced quoted phrase wrapping a consecutive-wildcard token
        # classifies expert, its primary MATCH is invalid FTS5, and the bare
        # fallback re-normalizes the original to the valid prefix ``forum*`` —
        # which matches the reported vocabulary, so the fallback must return
        # real hits (crosses ``*`` adjacent to quotes on the fallback arm).
        FuzzCase('"forum**"?', expect_failure_delta=1, expect_hits=True),
        FuzzCase(_REPORTED_QUERY, expect_failure_delta=1, expect_hits=True),
    ]


_FUZZ_CORPUS: tuple[FuzzCase, ...] = tuple(_build_group_a() + _build_group_b())


# ---------------------------------------------------------------------------
# AC5 — genuine-expert parity
# ---------------------------------------------------------------------------


class TestGenuineExpertParity:
    """Genuine expert queries keep their classification and normalized output.

    This is the regression the rejected rewrite caused: ``essence:"a b"`` and
    ``content:"machine learning"`` lost their column filter. The fix touches
    only classification and execution-time fallback, never the expert
    normalizer, so these outputs stay byte-identical to release/v0.6.0.
    """

    @pytest.mark.parametrize(("query", "expected_normalized"), _EXPERT_PARITY.items())
    def test_expert_normalization_is_byte_identical(
        self,
        query: str,
        expected_normalized: str,
    ) -> None:
        """Each genuine expert query classifies expert and normalizes unchanged."""
        assert _query_is_expert_syntax(query) is True
        assert _normalize_fts_query(query) == expected_normalized

    async def test_expert_queries_execute_without_raising(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """Every genuine expert query drives a valid MATCH (no fallback needed)."""
        before = reported_store.fts_match_failure_count
        for query in _EXPERT_PARITY:
            assert isinstance(await reported_store.search_fts(query), list)
        # Well-formed expert queries never hit the fallback.
        assert reported_store.fts_match_failure_count == before


# ---------------------------------------------------------------------------
# AC2 — curated adversarial table
# ---------------------------------------------------------------------------


class TestCuratedAdversarialTable:
    """Explicit, human-readable adversarial rows: classification + valid MATCH."""

    @pytest.mark.parametrize(("query", "expected_expert"), _CURATED_TABLE)
    def test_classification(self, query: str, expected_expert: bool) -> None:
        """Each curated row classifies bare vs expert as documented."""
        assert _query_is_expert_syntax(query) is expected_expert

    @pytest.mark.parametrize(("query", "expected_expert"), _CURATED_TABLE)
    async def test_valid_match(
        self,
        query: str,
        expected_expert: bool,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """Each curated row drives a valid MATCH (directly or via fallback)."""
        del expected_expert  # shared table with the classification test
        assert isinstance(await reported_store.search_fts(query), list)


# ---------------------------------------------------------------------------
# AC3 — the concrete reported input now returns hits
# ---------------------------------------------------------------------------


class TestReportedInputRegression:
    """The concrete reported input returns non-empty FTS/BM25 hits after the fix."""

    async def test_reported_input_returns_fts_hits(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """The scare-quote NL query yields BM25 hits instead of silent zero."""
        results = await reported_store.search_fts(_REPORTED_QUERY, top_k=10)
        assert results, "reported input returned no FTS hits"
        assert "t-reported" in {thought_id for thought_id, _ in results}

    async def test_reported_input_hybrid_keeps_fts_arm_live(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """The reported input drives the primary MATCH failure exactly once."""
        before = reported_store.fts_match_failure_count
        await reported_store.search_fts(_REPORTED_QUERY, top_k=10)
        assert reported_store.fts_match_failure_count == before + 1


# ---------------------------------------------------------------------------
# AC4 — failure counter semantics
# ---------------------------------------------------------------------------


class TestFailureCounter:
    """The read-only failure counter surfaces primary-MATCH failures.

    "0 across the AC1 corpus" is precise: the counter stays 0 across the
    *Group-A* (valid-primary) partition, whose queries never reach the
    fallback. The Group-B partition deliberately exercises the fallback, so its
    per-case delta is exactly 1 by design — that is asserted separately in
    :class:`TestSafetyInvariant`, not folded into the "stays 0" claim.
    """

    async def test_counter_increments_on_invalid_primary_match(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """A deliberately-invalid primary MATCH increments the counter by one."""
        assert reported_store.fts_match_failure_count == 0
        await reported_store.search_fts('"x"?')
        assert reported_store.fts_match_failure_count == 1

    async def test_counter_stays_zero_across_group_a(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """No Group-A (valid-primary) query ever touches the fallback."""
        for case in _FUZZ_CORPUS:
            if case.expect_failure_delta != 0:
                continue
            await reported_store.search_fts(case.query, top_k=10)
        assert reported_store.fts_match_failure_count == 0

    async def test_empty_bare_fallback_returns_no_hits_and_still_counts(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """An invalid primary whose bare fallback is empty degrades to no hits.

        ``"?"?`` classifies expert, its primary MATCH is invalid, and its bare
        normalization sanitizes to the empty string, so the fallback cannot run.
        The query still returns an empty list (never raises) and the failure is
        still counted — the guarded, doubly-degenerate branch.
        """
        results = await reported_store.search_fts('"?"?')
        assert results == []
        assert reported_store.fts_match_failure_count == 1

    async def test_counter_property_is_read_only(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """``fts_match_failure_count`` is a read-only property (no setter)."""
        prop = type(reported_store).fts_match_failure_count
        assert isinstance(prop, property)
        assert prop.fset is None
        with pytest.raises(AttributeError):
            reported_store.fts_match_failure_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC1 — standing safety invariant over the broad adversarial corpus
# ---------------------------------------------------------------------------


class TestSafetyInvariant:
    """No adversarial query raises or silently degrades due to an invalid MATCH.

    For every case: ``search_fts`` returns a list, never raising
    ``OperationalError``; and the primary-``MATCH`` failure counter moves
    exactly as the case's partition predicts (0 for valid-primary, 1 for
    fallback). A fallback case that shares the corpus vocabulary must still
    return real hits — i.e. the fallback genuinely *executed*, not merely
    swallowed the error.
    """

    async def test_no_query_raises_or_silently_degrades(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """Drive the whole adversarial corpus through the real thought_fts."""
        for case in _FUZZ_CORPUS:
            before = reported_store.fts_match_failure_count
            results = await reported_store.search_fts(case.query, top_k=10)
            after = reported_store.fts_match_failure_count

            assert isinstance(results, list), f"non-list result for {case.query!r}"
            delta = after - before
            assert delta == case.expect_failure_delta, (
                f"failure delta {delta} != expected {case.expect_failure_delta} for {case.query!r}"
            )
            if case.expect_hits:
                # The fallback anchor: without the retry these invalid-primary
                # cases would return an empty list rather than real hits.
                assert results, f"expected hits (fallback did not execute) for {case.query!r}"

    def test_corpus_crosses_the_adversarial_space(self) -> None:
        """Meta-test: the corpus genuinely spans the documented feature space.

        Guards against a corpus that silently collapses to a single token
        shape (a renamed token table). Each predicate below must be witnessed
        by at least one case.
        """
        queries = [case.query for case in _FUZZ_CORPUS]

        def any_case(predicate: Callable[[str], bool]) -> bool:
            return any(predicate(q) for q in queries)

        # Quote parity: both odd/unbalanced and balanced (>0) quote counts.
        assert any_case(lambda q: q.count('"') % 2 == 1)
        assert any_case(lambda q: q.count('"') > 0 and q.count('"') % 2 == 0)

        # Every sentence-punctuation mark appears somewhere.
        for punct in _SENTENCE_PUNCT:
            assert any_case(lambda q, p=punct: p in q), f"missing punctuation {punct!r}"

        # Booleans: uppercase operators and lowercase word forms.
        assert any_case(lambda q: " AND " in q or " OR " in q or " NOT " in q)
        assert any_case(lambda q: " and " in q or " or " in q or " not " in q)

        # Field filters, wildcards, hyphens, apostrophes, unicode, parentheses.
        assert any_case(lambda q: "essence:" in q)
        assert any_case(lambda q: "content:" in q)
        assert any_case(lambda q: "*" in q)
        assert any_case(lambda q: "-" in q)

        # Wildcard hazards (the WS-170 AC1 gap): consecutive ``**``, a triple-or-
        # more run, a leading-``*`` token, a standalone ``*`` token, and a ``*``
        # pressed against a quote — each must be witnessed so the invariant
        # genuinely exercises the wildcard-collapsing sanitizer.
        assert any_case(lambda q: "**" in q)
        assert any_case(lambda q: q.count("*") >= 3)
        assert any_case(lambda q: any(tok.startswith("*") for tok in q.split()))
        assert any_case(lambda q: "*" in q.split())
        assert any_case(lambda q: '*"' in q or '"*' in q)
        assert any_case(lambda q: "'" in q)
        assert any_case(lambda q: any(ord(ch) > 127 for ch in q))
        assert any_case(lambda q: "(" in q and ")" in q)

        # Leading / trailing punctuation and empty / whitespace-only inputs.
        assert any_case(lambda q: bool(q) and not q[0].isalnum() and q[0] not in {'"', "'"})
        assert any_case(lambda q: bool(q) and not q.strip()[-1:].isalnum() if q.strip() else False)
        assert "" in queries
        assert any_case(lambda q: bool(q) and not q.strip())

        # Both partitions are non-trivially populated.
        assert sum(1 for c in _FUZZ_CORPUS if c.expect_failure_delta == 0) >= 25
        assert sum(1 for c in _FUZZ_CORPUS if c.expect_failure_delta == 1) >= 5


# ---------------------------------------------------------------------------
# AC1 discriminating power — the invariant fails when a fix is removed
# ---------------------------------------------------------------------------


class TestDiscriminatingPower:
    """The safety invariant is only meaningful if removing a fix breaks it.

    These tests reproduce, in-process, the two reverts the WS spec requires and
    assert the invariant's per-case predictions change — proving the corpus has
    real discriminating power rather than passing vacuously.
    """

    async def test_reverting_classification_flips_scare_quote_anchor(
        self,
        reported_store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the old (over-triggering) classifier a scare-quote case fails.

        The odd/scare-quote anchor ``he said "run away`` is a Group-A case
        (expected delta 0). Under the pre-fix rule ``'"' in query -> expert`` it
        normalizes to an unbalanced phrase, the primary MATCH is invalid, and
        the fallback fires — so its failure delta becomes 1, breaking the
        invariant's Group-A prediction.
        """
        anchor = 'he said "run away'
        # Baseline: valid primary, no failure.
        before = reported_store.fts_match_failure_count
        await reported_store.search_fts(anchor)
        assert reported_store.fts_match_failure_count == before

        # Revert Fix 2: restore the over-triggering classifier.
        import engrava.infrastructure.sqlite.engrava_core as core_mod

        def _old_is_expert(query: str) -> bool:
            if '"' in query:
                return True
            for token in query.split():
                if token in core_mod._FTS_BOOLEAN_OPERATORS:
                    return True
                if core_mod._FTS_FIELD_FILTER_RE.match(token.lstrip("(")):
                    return True
            return False

        monkeypatch.setattr(core_mod, "_query_is_expert_syntax", _old_is_expert)

        before = reported_store.fts_match_failure_count
        await reported_store.search_fts(anchor)
        assert reported_store.fts_match_failure_count == before + 1, (
            "reverting the classification tightening must make the scare-quote "
            "anchor take the fallback (delta 1), breaking the Group-A invariant"
        )

    async def test_fallback_is_what_turns_invalid_expert_into_hits(
        self,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """The bare fallback, not the primary MATCH, produces the reported hits.

        The genuine-invalid-MATCH anchor ``"forum"?`` classifies as expert and
        its primary MATCH is invalid FTS5; the hits come solely from the bare
        fallback. Confirming the primary is invalid (counter increments) while
        results are non-empty proves that removing the fallback — which would
        return an empty list on the same ``OperationalError`` — breaks the
        invariant's ``expect_hits`` assertion.
        """
        anchor = '"forum"?'
        before = reported_store.fts_match_failure_count
        results = await reported_store.search_fts(anchor, top_k=10)
        after = reported_store.fts_match_failure_count

        assert after == before + 1, "primary MATCH for the anchor must be invalid"
        assert results, "the bare fallback must supply the anchor's hits"
        assert "t-reported" in {thought_id for thought_id, _ in results}

    async def test_reverting_wildcard_collapse_breaks_a_consecutive_star_case(
        self,
        reported_store: SqliteEngravaCore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Removing the wildcard sanitizer flips a bare consecutive-``*`` case.

        The anchor ``forum** body`` is a bare Group-A shape: under the
        wildcard-collapsing sanitizer it normalizes to the valid
        ``forum* OR body`` (delta 0) and returns real hits. With the collapse
        reverted to an identity — the pre-fix behaviour — the ``forum**`` token
        survives as an invalid ``**`` term, so the *primary* MATCH raises AND
        the bare fallback (which re-uses the same sanitizer) raises too: the
        query silently degrades to no hits with a failure delta of 1. That is
        exactly the WS-170 bug the safety invariant must catch, proving the
        wildcard cases carry real discriminating power rather than passing
        vacuously.
        """
        import engrava.infrastructure.sqlite.engrava_core as core_mod

        anchor = "forum** body"

        # Baseline: the collapsing sanitizer keeps the primary MATCH valid.
        before = reported_store.fts_match_failure_count
        baseline = await reported_store.search_fts(anchor, top_k=10)
        assert reported_store.fts_match_failure_count == before, (
            "the wildcard-collapsing sanitizer must keep the primary MATCH valid"
        )
        assert baseline, "the collapsed primary MATCH must return real hits"

        # Revert the wildcard fix: an identity collapse restores the raw ``**``.
        monkeypatch.setattr(core_mod, "_collapse_fts_wildcards", lambda fragment: fragment)

        before = reported_store.fts_match_failure_count
        reverted = await reported_store.search_fts(anchor, top_k=10)
        assert reported_store.fts_match_failure_count == before + 1, (
            "reverting the wildcard collapse must make the consecutive-star "
            "primary MATCH invalid (delta 1)"
        )
        assert reverted == [], (
            "with the collapse reverted the bare fallback is also invalid, so "
            "the query silently degrades to no hits — the WS-170 defect"
        )


# ---------------------------------------------------------------------------
# G4 — empty quoted spans are a deliberate, pinned bare classification
# ---------------------------------------------------------------------------


class TestEmptyQuotedSpanIsBare:
    """An empty ``""`` span is not a deliberate phrase, so it stays bare.

    ``foo "" bar`` and ``foo "  " bar`` carry a balanced quote count but wrap no
    token, so there is no deliberate phrase construct: they classify *bare* and
    OR-expand to a valid ``foo OR bar`` MATCH rather than being treated as
    expert phrase syntax. This pins that as an intentional, tested choice (not
    an accident of the quote-parity rule).
    """

    _EMPTY_QUOTE_QUERIES: tuple[str, ...] = ('foo "" bar', 'foo "  " bar')

    @pytest.mark.parametrize("query", _EMPTY_QUOTE_QUERIES)
    def test_empty_quoted_span_classifies_bare(self, query: str) -> None:
        """An empty quoted span never trips the expert classifier."""
        assert _query_is_expert_syntax(query) is False
        assert _normalize_fts_query(query) == "foo OR bar"

    @pytest.mark.parametrize("query", _EMPTY_QUOTE_QUERIES)
    async def test_empty_quoted_span_drives_a_valid_match(
        self,
        query: str,
        reported_store: SqliteEngravaCore,
    ) -> None:
        """The bare normalization of an empty quoted span is a valid MATCH."""
        before = reported_store.fts_match_failure_count
        assert isinstance(await reported_store.search_fts(query), list)
        assert reported_store.fts_match_failure_count == before

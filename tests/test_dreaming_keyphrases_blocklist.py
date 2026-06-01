"""Tests for the sentence-starter blocklist + role-marker stripping helpers.

Covers the two pure helpers introduced as the v2 quality amendment:

* ``_strip_role_markers`` — substitutes ``[USER] User: ...``,
  ``[ASSISTANT] Assistant: ...`` and ``[SYSTEM] ...`` prefixes with a
  single space.  Verified independently here and as a black-box
  effect on ``extract_simple_keywords`` / ``top_keyphrases_tfidf``.
* ``SENTENCE_STARTER_BLOCKLIST`` — the public ``frozenset`` of
  capitalised non-entity words.  We verify the membership semantics
  used by the named-entity filter, plus a smoke test that real
  proper nouns are not collateral.
"""

from __future__ import annotations

from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming_keyphrases import (
    SENTENCE_STARTER_BLOCKLIST,
    _strip_role_markers,
    extract_simple_keywords,
    top_keyphrases_tfidf,
)


def _thought(thought_id: str, content: str) -> ThoughtRecord:
    """Minimal ``ThoughtRecord`` fixture for keyphrase tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        created_at="2026-04-29T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# _strip_role_markers — pure-function unit tests
# ---------------------------------------------------------------------------


class TestStripRoleMarkers:
    """Behaviour of the role-marker stripping helper in isolation."""

    def test_strip_user_marker_with_literal_role(self) -> None:
        """``[USER] User: ...`` → ``" ..."`` (marker + role literal removed)."""
        text = "[USER] User: hello there"
        result = _strip_role_markers(text)
        assert "User:" not in result
        assert "[USER]" not in result
        assert "hello there" in result

    def test_strip_assistant_marker_with_literal_role(self) -> None:
        text = "[ASSISTANT] Assistant: response body"
        result = _strip_role_markers(text)
        assert "Assistant:" not in result
        assert "[ASSISTANT]" not in result
        assert "response body" in result

    def test_strip_system_marker_consumes_to_newline(self) -> None:
        """``[SYSTEM] persona stuff\\n`` → blank line; trailing body survives."""
        text = "[SYSTEM] You are a helpful assistant.\nactual body here"
        result = _strip_role_markers(text)
        assert "[SYSTEM]" not in result
        assert "You are a helpful assistant" not in result
        assert "actual body here" in result

    def test_strip_bare_user_bracket_only(self) -> None:
        """Bare ``[USER]`` (no ``User:`` literal) is also stripped."""
        text = "[USER] message without role literal"
        result = _strip_role_markers(text)
        assert "[USER]" not in result
        assert "message without role literal" in result

    def test_strip_is_case_insensitive(self) -> None:
        """Stripping handles upper / lower / mixed case markers."""
        for variant in ("[user] user: hi", "[User] User: hi", "[USER] User: hi"):
            result = _strip_role_markers(variant)
            assert "user:" not in result.lower()
            assert "[user]" not in result.lower()

    def test_strip_preserves_non_marker_brackets(self) -> None:
        """Non-marker bracketed text passes through unchanged."""
        text = "Notes [TODO] follow up"
        result = _strip_role_markers(text)
        assert "[TODO]" in result
        assert "follow up" in result

    def test_strip_idempotent(self) -> None:
        """Calling the helper twice yields the same output as once."""
        text = "[USER] User: hello\n[ASSISTANT] Assistant: hi"
        once = _strip_role_markers(text)
        twice = _strip_role_markers(once)
        assert once == twice

    def test_strip_empty_text(self) -> None:
        assert _strip_role_markers("") == ""


# ---------------------------------------------------------------------------
# SENTENCE_STARTER_BLOCKLIST — membership semantics
# ---------------------------------------------------------------------------


class TestSentenceStarterBlocklist:
    """The public blocklist holds non-entity capitalised words."""

    def test_contains_canonical_sentence_openers(self) -> None:
        for token in (
            "Absolutely",
            "Additionally",
            "However",
            "Therefore",
            "Furthermore",
            "Moreover",
        ):
            assert token in SENTENCE_STARTER_BLOCKLIST

    def test_contains_role_marker_capitalisations(self) -> None:
        """Bare ``User`` / ``Assistant`` / ``System`` capitalisations are blocked."""
        for token in (
            "User",
            "USER",
            "Assistant",
            "ASSISTANT",
            "System",
            "SYSTEM",
        ):
            assert token in SENTENCE_STARTER_BLOCKLIST

    def test_does_not_contain_real_proper_nouns(self) -> None:
        """Genuine proper nouns must survive the blocklist filter."""
        for token in (
            "Cornell",
            "Maria",
            "Jordan",
            "Berlin",
            "Tokyo",
            "Anthropic",
            "Tan",
        ):
            assert token not in SENTENCE_STARTER_BLOCKLIST


# ---------------------------------------------------------------------------
# extract_simple_keywords — strip applied before tokenisation
# ---------------------------------------------------------------------------


class TestExtractSimpleKeywordsRoleMarkerStrip:
    """Role markers stripped before ``_tokenize`` so they cannot rank as keywords."""

    def test_user_user_artifact_does_not_appear(self) -> None:
        # Repeated turns with ``[USER] User: ...`` would naturally
        # produce ``"user"`` as a top frequency token.  After the strip
        # both the bracket marker and the role literal are gone.
        texts = [
            "[USER] User: Discuss study techniques regularly",
            "[USER] User: Discuss study habits during the week",
            "[USER] User: Discuss study sessions before exams",
        ]
        keywords = extract_simple_keywords(texts, top_n=10)
        assert "user" not in keywords
        # And the genuine content words still rank high.
        assert "study" in keywords
        assert "discuss" in keywords

    def test_assistant_marker_artifact_suppressed(self) -> None:
        texts = [
            "[ASSISTANT] Assistant: explanation about goals planning sessions",
            "[ASSISTANT] Assistant: explanation about goals quarterly notes",
        ]
        keywords = extract_simple_keywords(texts, top_n=10)
        assert "assistant" not in keywords


# ---------------------------------------------------------------------------
# top_keyphrases_tfidf — strip applied before n-gram extraction
# ---------------------------------------------------------------------------


class TestTopKeyphrasesTfidfRoleMarkerStrip:
    """N-gram extractor must not surface ``"user user"`` etc. artifacts."""

    def test_no_user_user_artifact_in_top_keyphrases(self) -> None:
        cluster = [
            _thought("t-1", "[USER] User: study techniques discussion ongoing"),
            _thought("t-2", "[USER] User: study techniques review meeting today"),
        ]
        corpus = ["unrelated content one", "another unrelated topic two"]
        result = top_keyphrases_tfidf(cluster, corpus=corpus, top_n=10)
        phrases = {entry["phrase"] for entry in result}
        # Critical regression guard: no ``user user`` appears as a keyphrase.
        assert "user user" not in phrases
        assert "assistant assistant" not in phrases

    def test_legitimate_keyphrases_unaffected(self) -> None:
        """The strip preserves substantive bigrams from the surrounding content."""
        cluster = [
            _thought(
                "t-1",
                "[USER] User: I am studying neural networks for my thesis project",
            ),
            _thought(
                "t-2",
                "[USER] User: Reading papers on neural networks weekly now",
            ),
        ]
        corpus = ["unrelated content one", "another unrelated topic two"]
        result = top_keyphrases_tfidf(cluster, corpus=corpus, top_n=10)
        phrases = {entry["phrase"] for entry in result}
        # ``neural networks`` survives — strip only removed the markers.
        assert "neural networks" in phrases


# ---------------------------------------------------------------------------
# short07 NE top-15 audit regression tests
# ---------------------------------------------------------------------------


class TestBlocklistExtensionShort07Audit:
    """Regression tests for the blocklist extension.

    11 words from the short07 NE top-15 audit (2026-05-04) were found
    among the most frequent named_entities in 47 REFLECTION thoughts;
    all must be present in ``SENTENCE_STARTER_BLOCKLIST``.
    """

    def test_blocklist_includes_short07_audit_findings(self) -> None:
        """Verify 11 NE top-15 sentence-starters from short07 are blocked."""
        short07_findings = {
            "How",
            "For",
            "Ultimately",
            "Have",
            "Instead",
            "Lastly",
            "Also",
            "Not",
            "Did",
            "Embracing",
            "Reflecting",
        }
        missing = short07_findings - SENTENCE_STARTER_BLOCKLIST
        assert not missing, f"Missing from blocklist: {missing}"

    def test_legitimate_nes_not_filtered(self) -> None:
        """Verify proper nouns and year literals pass the blocklist."""
        legit = ["Alex", "Cornell", "Maria", "Hispanic", "Q3", "1974", "1966"]
        blocked = [w for w in legit if w in SENTENCE_STARTER_BLOCKLIST]
        assert not blocked, f"Legitimate NEs incorrectly blocked: {blocked}"

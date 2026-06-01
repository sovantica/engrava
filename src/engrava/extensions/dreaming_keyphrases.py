"""Deterministic keyphrase extraction for structural REFLECTION content.

Two pure functions land here, both used by the v2 REFLECTION content
builder in :mod:`engrava.extensions.dreaming_reflection_content`:

* :func:`extract_simple_keywords` — frequency-ranked single tokens.
  This is the migrated body of the previously-private ``_extract_keywords``
  helper in :mod:`engrava.extensions.dreaming` (renamed and made public
  so the v2 content builder can call it without crossing private-API
  boundaries).
* :func:`top_keyphrases_tfidf` — TF-IDF-scored 2-3 word n-grams over
  a cluster's content with the corpus baseline coming from a caller-
  supplied list of all corpus content strings.  Produces richer
  semantic surface than single tokens for downstream readers (LLM
  judges, semantic search, agent retrieval).

Both functions are intentionally LLM-free per the cognitive-boundary
guard test at ``tests/test_tier_boundary_guard.py`` — they rely only on
``numpy``, ``re``, and ``math`` from the standard library.  The
stopword list is an inline tuple constant; no ``nltk`` / ``spaCy``
dependency is required.

Determinism guarantees:

* Identical input ``cluster`` produces byte-identical output across
  process invocations, Python versions, and operating systems.
* No randomness, no system-time inputs, no hash-randomisation
  exposure.  ``re.split`` and ``str.lower`` are deterministic; tie-
  breaking uses lexicographic order on the phrase string itself.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engrava.domain.models.thought import ThoughtRecord

__all__ = (
    "SENTENCE_STARTER_BLOCKLIST",
    "compute_cluster_phrase_frequency",
    "extract_simple_keywords",
    "is_boilerplate_phrase",
    "top_keyphrases_tfidf",
)


# Common sentence-starter words and conversation-marker artifacts that
# the lightweight regex-based named-entity extractor would otherwise
# capture as proper nouns.  Public so the v2 REFLECTION content
# builder can apply the same blocklist when filtering its
# ``named_entities`` field.
#
# Empirically derived from cluster-content sampling — extend
# conservatively (prefer false negatives over false positives so real
# proper nouns survive).
SENTENCE_STARTER_BLOCKLIST: frozenset[str] = frozenset(
    {
        # Sentence-starter adverbs / connectives / determiners.
        "Absolutely",
        "Additionally",
        "Alongside",
        "Although",
        "Among",
        "And",
        "Anyway",
        "Around",
        "As",
        "Attending",
        "Beautifully",
        "Before",
        "Being",
        "Beyond",
        "Building",
        "But",
        "Certainly",
        "Collaboration",
        "Community",
        "Considering",
        "Creating",
        "Depending",
        "Despite",
        "Doing",
        "During",
        "Each",
        "Either",
        "Engaging",
        "Enjoying",
        "Especially",
        "Even",
        "Every",
        "Exploring",
        "Finally",
        "First",
        "Following",
        "Furthermore",
        "Generally",
        "Given",
        "Going",
        "Having",
        "Hello",
        "Here",
        "However",
        "Importantly",
        "Indeed",
        "Initially",
        "Initiatives",
        "Interesting",
        "Just",
        "Knowing",
        "Last",
        "Letting",
        "Likewise",
        "Looking",
        "Making",
        "Many",
        "Maybe",
        "Meanwhile",
        "Moreover",
        "Most",
        "Much",
        "Nevertheless",
        "New",
        "Next",
        "Note",
        "Now",
        "Often",
        "Once",
        "One",
        "Only",
        "Originally",
        "Other",
        "Otherwise",
        "Our",
        "Overall",
        "Particularly",
        "People",
        "Perhaps",
        "Please",
        "Plus",
        "Practicing",
        "Probably",
        "Putting",
        "Recently",
        "Regarding",
        "Remember",
        "Same",
        "Seeing",
        "Several",
        "Similarly",
        "Since",
        "Some",
        "Sometimes",
        "Soon",
        "Specifically",
        "Starting",
        "Still",
        "Such",
        "Sure",
        "Taking",
        "Than",
        "Thanks",
        "That",
        "The",
        "Their",
        "Then",
        "There",
        "Therefore",
        "These",
        "They",
        "This",
        "Those",
        "Though",
        "Through",
        "Throughout",
        "Thus",
        "Together",
        "Towards",
        "Trying",
        "Understanding",
        "Unfortunately",
        "Unless",
        "Until",
        "Upon",
        "Usually",
        "Very",
        "Watching",
        "Well",
        "What",
        "When",
        "Whenever",
        "Where",
        "Whether",
        "Which",
        "While",
        "Who",
        "Why",
        "With",
        "Within",
        "Without",
        "Working",
        "Yes",
        "Yet",
        "You",
        "Your",
        # Empirical additions from short07 NE top-15 audit (2026-05-04).
        # These words appeared among the 15 most common named_entities in 47
        # REFLECTION thoughts; 11 of 15 were sentence-starters absent from the
        # Blocklist.
        "Also",
        "Did",
        "Embracing",
        "For",
        "Have",
        "How",
        "Instead",
        "Lastly",
        "Not",
        "Reflecting",
        "Ultimately",
        # Preventive gerund coverage — common gerund/participle
        # sentence-starters not yet in the base list.
        "Conducting",
        "Connecting",
        "Continuing",
        "Discovering",
        "Discussing",
        "Focusing",
        "Implementing",
        "Integrating",
        "Learning",
        "Living",
        "Managing",
        "Preparing",
        "Pursuing",
        "Sharing",
        "Studying",
        # Conversation / role markers — both the ``[USER]`` literal and the
        # bare ``User``/``Assistant`` capitalisation that survives the
        # ``_strip_role_markers`` pass when the marker prefix has already
        # been removed by the chunker.
        "User",
        "USER",
        "Assistant",
        "ASSISTANT",
        "System",
        "SYSTEM",
    }
)

# Role-marker prefixes the benchmark adapter inserts in front of every
# turn (``[USER] User: ...``, ``[ASSISTANT] Assistant: ...``,
# ``[SYSTEM] You are ...``).  ``_strip_role_markers`` substitutes them
# with a single space *before* tokenisation so n-gram extraction does
# not produce the ``"user user"`` / ``"assistant assistant"``
# artifacts that earlier deep-analysis flagged as
# high-TF-IDF noise.
#
# Order matters: the longer ``[USER] User:`` / ``[ASSISTANT] Assistant:``
# patterns must match before the bare ``[USER]`` / ``[ASSISTANT]``
# fallbacks so the colon and ``User``/``Assistant`` literal are also
# consumed.
_ROLE_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[USER\]\s*User:?\s*", re.IGNORECASE),
    re.compile(r"\[ASSISTANT\]\s*Assistant:?\s*", re.IGNORECASE),
    re.compile(r"\[SYSTEM\][^\n]*\n?", re.IGNORECASE),
    re.compile(r"\[USER\]", re.IGNORECASE),
    re.compile(r"\[ASSISTANT\]", re.IGNORECASE),
    re.compile(r"\[SYSTEM\]", re.IGNORECASE),
)


def _strip_role_markers(text: str) -> str:
    """Remove ``[USER]`` / ``[ASSISTANT]`` / ``[SYSTEM]`` markers from *text*.

    Each pattern is replaced with a single space so the surrounding
    tokens remain word-boundary separated for downstream regex /
    tokeniser callers.  The function is pure and stateless — safe to
    call on cluster contents and corpus documents.

    Args:
        text: Raw content string, typically a thought's ``content``.

    Returns:
        The same string with every recognised role-marker prefix
        replaced by a single space.

    """
    for pattern in _ROLE_MARKER_PATTERNS:
        text = pattern.sub(" ", text)
    return text


# Stopword list inlined per the zero-dep policy (no nltk / spaCy).
# Ordered to mirror the canonical English short-list; tuple keeps the
# membership check O(1) on a frozenset built at import time.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "its",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "this",
        "that",
        "with",
        "from",
        "they",
        "them",
        "have",
        "been",
        "were",
        "will",
        "into",
        "your",
        "what",
        "when",
        "make",
        "like",
        "just",
        "than",
        "then",
        "look",
        "only",
        "come",
        "over",
        "also",
        "back",
        "after",
        "work",
        "first",
        "well",
        "even",
        "want",
        "give",
        "most",
        "very",
        "such",
        "here",
        "there",
        "would",
        "could",
        "should",
    }
)

_MIN_TOKEN_LENGTH = 3
_TOKENIZER = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    """Lower-case, split on non-word characters, drop short / stopword tokens.

    Args:
        text: Raw content string.

    Returns:
        Ordered list of meaningful tokens; empty when *text* contains
        only stopwords or short tokens.

    """
    return [
        token
        for token in _TOKENIZER.split(text.lower())
        if len(token) >= _MIN_TOKEN_LENGTH and token not in _STOPWORDS
    ]


def extract_simple_keywords(texts: list[str], *, top_n: int = 10) -> list[str]:
    """Extract the top-N tokens by frequency over *texts* (legacy v1 producer).

    Migrated verbatim from the former ``_extract_keywords`` helper in
    :mod:`engrava.extensions.dreaming` so the v2 REFLECTION content
    builder can keep producing the legacy ``keywords`` field while
    living in a sibling module.  Behaviourally identical: lowercase,
    split on non-word characters, drop tokens shorter than three
    characters, return the top-N tokens sorted by frequency
    descending.

    The implementation is intentionally LLM-free — pure
    string / regex / dict operations only, validated by the cognitive-
    boundary guard test at ``tests/test_tier_boundary_guard.py``.

    Args:
        texts: List of raw content strings (one per cluster member).
        top_n: Maximum number of keywords to return.  Defaults to
            ``10`` to match the legacy callsite contract.

    Returns:
        List of keywords sorted by frequency descending and trimmed
        to *top_n*.  Ties are broken by insertion order (Python dicts
        preserve insertion order), so the output is fully
        deterministic for a given input.

    """
    freq: dict[str, int] = {}
    for text in texts:
        for token in _tokenize(_strip_role_markers(text)):
            freq[token] = freq.get(token, 0) + 1

    sorted_words = sorted(freq, key=lambda w: freq[w], reverse=True)
    return sorted_words[:top_n]


def _build_ngrams(tokens: list[str], *, n: int) -> list[str]:
    """Return all *n*-token contiguous n-grams from *tokens*."""
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def top_keyphrases_tfidf(
    cluster: list[ThoughtRecord],
    *,
    corpus: list[str],
    top_n: int = 3,
) -> list[dict[str, float | str]]:
    """Score 2-3 word n-grams via TF-IDF and return the top-N with their scores.

    Term frequency (TF) is computed over the cluster: how often each
    n-gram appears across the cluster members' content.  Inverse
    document frequency (IDF) is computed over the supplied *corpus*,
    treating every corpus entry as a separate document.  The score is
    ``tf * log((1 + N) / (1 + df))`` with ``+1`` smoothing on both
    numerator and denominator so that an n-gram appearing in every
    corpus document still receives a small positive score (avoids
    NaN / log(0)).

    The function is fully deterministic: identical *cluster* + *corpus*
    produces byte-identical output across runs.  Tie-breaking on the
    score uses lexicographic order on the phrase string.

    Args:
        cluster: Member ``ThoughtRecord`` instances (post-clustering).
            Only the ``content`` attribute is read.
        corpus: Flat list of all content strings in the parent
            thought corpus (the broader collection from which
            *cluster* was drawn).  Used as the IDF document set.
            Empty corpus is allowed and degenerates to a TF-only
            ranking with all IDFs equal to ``log(1.0)`` after
            smoothing.
        top_n: Maximum number of keyphrases to return.  Defaults to
            ``3`` to match ``DreamingConfig.top_keyphrases_count``.

    Returns:
        List of ``{"phrase": str, "score": float}`` dictionaries
        sorted by score descending, length ≤ *top_n*.  Scores are
        rounded to 4 decimal places for stable JSON serialisation.

    """
    # --- Tokenise cluster contents and build candidate n-grams ---------
    # Role markers are stripped before tokenisation so the n-gram
    # extractor does not produce ``"user user"`` / ``"assistant
    # assistant"`` artifacts from the synthetic ``[USER] User: ...``
    # prefixes the benchmark adapter inserts.
    cluster_tokens_per_member: list[list[str]] = [
        _tokenize(_strip_role_markers(member.content)) for member in cluster
    ]
    cluster_phrases: list[str] = []
    for tokens in cluster_tokens_per_member:
        cluster_phrases.extend(_build_ngrams(tokens, n=2))
        cluster_phrases.extend(_build_ngrams(tokens, n=3))

    if not cluster_phrases:
        return []

    # --- Term frequency over the cluster ------------------------------
    tf_counts: dict[str, int] = {}
    for phrase in cluster_phrases:
        tf_counts[phrase] = tf_counts.get(phrase, 0) + 1

    unique_phrases = sorted(tf_counts.keys())  # deterministic order

    # --- Document frequency over the corpus ---------------------------
    # Pre-tokenise the corpus once and reuse for every candidate phrase.
    # Role markers are stripped from corpus documents too, so IDF
    # computed across the corpus does not see the synthetic role-marker
    # n-grams either.
    corpus_token_lists: list[list[str]] = [_tokenize(_strip_role_markers(doc)) for doc in corpus]
    corpus_phrase_sets: list[set[str]] = []
    for tokens in corpus_token_lists:
        bigrams = set(_build_ngrams(tokens, n=2))
        trigrams = set(_build_ngrams(tokens, n=3))
        corpus_phrase_sets.append(bigrams | trigrams)

    n_docs = len(corpus_phrase_sets)
    df_counts: dict[str, int] = dict.fromkeys(unique_phrases, 0)
    for doc_phrases in corpus_phrase_sets:
        for phrase in unique_phrases:
            if phrase in doc_phrases:
                df_counts[phrase] += 1

    # --- TF-IDF score (with +1 smoothing on both numerator + denominator) ---
    tf_array = np.array([tf_counts[p] for p in unique_phrases], dtype=np.float64)
    df_array = np.array([df_counts[p] for p in unique_phrases], dtype=np.float64)
    idf_array = np.log((1.0 + n_docs) / (1.0 + df_array))
    score_array = tf_array * idf_array

    # --- Sort: score descending, phrase ascending for tie-break -------
    indexed = sorted(
        enumerate(unique_phrases),
        key=lambda pair: (-score_array[pair[0]], pair[1]),
    )
    return [
        {"phrase": phrase, "score": round(float(score_array[idx]), 4)}
        for idx, phrase in indexed[:top_n]
    ]


# ------------------------------------------------------------------
# Cross-cluster boilerplate detection (language-agnostic, statistical)
# ------------------------------------------------------------------


def compute_cluster_phrase_frequency(
    cluster_keyphrases: list[list[dict[str, float | str]]],
) -> dict[str, int]:
    """Count in how many clusters each keyphrase appears.

    Takes the per-cluster top-N keyphrases (as produced by
    :func:`top_keyphrases_tfidf`) and returns a mapping from the
    lowercased phrase to the number of distinct clusters in which it
    occurs.  Each cluster contributes at most ``1`` per phrase even if
    the same phrase is repeated inside the cluster's own list — the
    metric is "document frequency across clusters", not raw count.

    The phrase is normalised via :py:meth:`str.lower` so that callers
    using mixed-case scoring (e.g. ``"Wonderful"`` vs ``"wonderful"``)
    share a single bucket.  Non-string ``phrase`` values are ignored
    defensively.

    Args:
        cluster_keyphrases: Outer list = clusters in the current
            dreaming run; inner list = top-N keyphrases per cluster
            (``{"phrase": str, "score": float}`` dicts as returned by
            :func:`top_keyphrases_tfidf`).

    Returns:
        Mapping ``lowercased_phrase -> cluster_count``.  Phrases that
        appear in zero clusters are absent from the mapping.

    Examples:
        >>> compute_cluster_phrase_frequency([
        ...     [{"phrase": "wonderful", "score": 0.5}],
        ...     [{"phrase": "wonderful", "score": 0.6}],
        ...     [{"phrase": "piano", "score": 0.4}],
        ... ])
        {'wonderful': 2, 'piano': 1}

    """
    document_frequency: dict[str, int] = {}
    for cluster_kps in cluster_keyphrases:
        seen: set[str] = set()
        for kp in cluster_kps:
            raw_phrase = kp.get("phrase")
            if not isinstance(raw_phrase, str):
                continue
            phrase = raw_phrase.lower()
            if phrase in seen:
                continue
            seen.add(phrase)
            document_frequency[phrase] = document_frequency.get(phrase, 0) + 1
    return document_frequency


def is_boilerplate_phrase(
    phrase: str,
    cluster_doc_frequency: dict[str, int],
    total_clusters: int,
    *,
    threshold: float,
    min_corpus_size: int,
) -> bool:
    """Return ``True`` when *phrase* is corpus-wide boilerplate.

    A phrase counts as boilerplate when its share of clusters exceeds
    ``threshold`` — for example, with ``threshold=0.30`` and 10
    clusters, a phrase appearing in 4 or more clusters (4/10 = 0.40 >
    0.30) is flagged.  The filter is intentionally statistical and
    language-agnostic: any phrase that floods the corpus, in any
    language, becomes a candidate for exclusion.

    The check is bypassed for small corpora (``total_clusters <
    min_corpus_size``).  Phrase-frequency statistics on a handful of
    clusters are too noisy to support a binary filter, and a strict
    filter under those conditions would erase most signal rather than
    just the boilerplate.

    The lookup is case-insensitive: the same normalisation
    (:py:meth:`str.lower`) used by
    :func:`compute_cluster_phrase_frequency` is applied to the
    candidate phrase before the dictionary lookup.

    Args:
        phrase: Candidate keyphrase from a cluster's top-N list.
        cluster_doc_frequency: Mapping produced by
            :func:`compute_cluster_phrase_frequency`.
        total_clusters: Total number of clusters scanned during the
            same dreaming run that produced ``cluster_doc_frequency``.
        threshold: Cluster-share ratio strictly above which a phrase
            is treated as boilerplate.  ``0.30`` flags phrases that
            appear in more than 30 % of clusters; ``1.0`` disables
            the filter (no phrase can exceed 100 %).
        min_corpus_size: Minimum ``total_clusters`` before the filter
            engages.  Smaller corpora always return ``False`` to
            avoid stripping the entire keyphrase list.

    Returns:
        ``True`` when *phrase* should be dropped as boilerplate,
        ``False`` otherwise.

    Examples:
        >>> df = {"wonderful": 5}
        >>> is_boilerplate_phrase(
        ...     "wonderful", df, total_clusters=10,
        ...     threshold=0.30, min_corpus_size=5,
        ... )
        True
        >>> is_boilerplate_phrase(
        ...     "wonderful", df, total_clusters=10,
        ...     threshold=0.60, min_corpus_size=5,
        ... )
        False

    """
    if total_clusters < min_corpus_size:
        return False
    cluster_count = cluster_doc_frequency.get(phrase.lower(), 0)
    return cluster_count / total_clusters > threshold

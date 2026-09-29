"""Cluster quality gates for v2 REFLECTION creation.

Each gate is a pure deterministic algorithm — no LLM, no state, no
side-effects.  The gates are designed to be called by the dreaming
consolidation loop on a resolved cluster (``list[ThoughtRecord]`` plus
its member embedding vectors) *before* materialising the REFLECTION, so
that clusters failing any quality check are skipped.  This module
provides the gate functions only; the call-site wiring lives in
``engrava.extensions.dreaming``.

Gate inventory (priority order — first failure rejects the cluster):

1. :func:`has_duplicate_content_members` — byte-identical member content
   (a dedup escape; duplicates inflate one persona description into a
   "cluster").
2. :func:`is_persona_only_cluster` — members are system-side persona
   descriptions rather than user-authored facts.
3. :func:`has_contradictory_members` — members contain sentiment-opposite
   tokens (English-only lexicon; clusters in other languages are not
   flagged by this gate — see the limitation note on ``_CONTRADICTION_PAIRS``).
4. :func:`is_low_cohesion` — mean pairwise cosine of member embeddings
   below the configured threshold (mixed-topic cluster). Computed as
   ``dot / (||a|| * ||b||)`` rather than the previous raw dot product,
   which removes that formula's magnitude sensitivity in the common
   case but is not an exact transform — it changes which side of the
   threshold several non-finite and overflowing input shapes land on,
   in both directions, relative to the pre-0.7 formula — see
   :func:`cluster_cohesion_score` for exactly which, executed, and the
   0.6 -> 0.7 upgrade note for the gate-level effect.
5. :func:`is_external_source_homogeneous` — at least the configured
   fraction of members come from external sources per the self-anchored
   ``metadata.source.is_self`` semantic.  Belt-and-suspenders over the
   upstream eligibility filter; legacy members without a populated
   ``metadata.source`` are treated as external (safe fallback).
6. :func:`has_consistent_entities` — at least the configured fraction of
   members share a named entity with the first member.  Delegates the
   per-member named-entity extraction to
   :func:`engrava.extensions.dreaming_reflection_content.extract_named_entities_per_member`
   so the extraction logic lives in exactly one place.
8. :func:`has_meaningful_keyphrases` — the cluster's ``top_keyphrases``
   list (post the cross-cluster TF-IDF boilerplate filter) is not entirely
   generic determiner-noun phrases.

The numbering follows the design document's intent: there is no Gate 7
in this module — the REFLECTION-default-priority knob already exists as
``DreamingConfig.reflection_default_priority`` (defaulting to ``"P2"``)
so there was no mechanism to add.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from engrava.extensions.dreaming_reflection_content import (
    extract_named_entities_per_member,
)

if TYPE_CHECKING:
    from engrava.domain.models.thought import ThoughtRecord

__all__ = (
    "has_consistent_entities",
    "has_contradictory_members",
    "has_duplicate_content_members",
    "has_meaningful_keyphrases",
    "is_external_source_homogeneous",
    "is_low_cohesion",
    "is_persona_only_cluster",
)


# ---------------------------------------------------------------------------
# Gate 1 — duplicate-member content
# ---------------------------------------------------------------------------

#: Minimum number of byte-identical members that flags a cluster.  Two or
#: more identical member contents almost always means the upstream
#: content-hash dedup was bypassed (whitespace / encoding differences) and
#: one persona description got inflated into a pseudo-cluster.
_DUPLICATE_MEMBER_MINIMUM = 2


def has_duplicate_content_members(
    cluster: list[ThoughtRecord],
) -> tuple[bool, int]:
    """Detect a cluster with two or more members of byte-identical content.

    Args:
        cluster: Resolved cluster members.

    Returns:
        ``(is_duplicate, duplicate_count)`` — ``is_duplicate`` is ``True``
        when at least :data:`_DUPLICATE_MEMBER_MINIMUM` members collide on
        content hash; ``duplicate_count`` is the total number of members
        that participate in any such collision (e.g. three identical
        members → ``3``).

    Examples:
        >>> from engrava.domain.models.thought import ThoughtRecord
        >>> from engrava.domain.enums import (
        ...     LifecycleStatus, Priority, ThoughtType,
        ... )
        >>> def _t(tid: str, content: str) -> ThoughtRecord:
        ...     return ThoughtRecord(
        ...         thought_id=tid, thought_type=ThoughtType.OBSERVATION,
        ...         essence="e", content=content, priority=Priority.P3,
        ...         lifecycle_status=LifecycleStatus.ACTIVE,
        ...         created_cycle=0, updated_cycle=0, source="t",
        ...     )
        >>> has_duplicate_content_members([_t("a", "x"), _t("b", "x")])
        (True, 2)
        >>> has_duplicate_content_members([_t("a", "x"), _t("b", "y")])
        (False, 0)

    """
    content_hashes = Counter(
        hashlib.sha256(member.content.encode("utf-8")).hexdigest()[:16] for member in cluster
    )
    duplicates = sum(
        count for count in content_hashes.values() if count >= _DUPLICATE_MEMBER_MINIMUM
    )
    return duplicates >= _DUPLICATE_MEMBER_MINIMUM, duplicates


# ---------------------------------------------------------------------------
# Gate 2 — persona-only cluster
# ---------------------------------------------------------------------------

#: Substrings that, in the absence of conversation markers, mark a member
#: as a system-side persona description rather than a user-authored fact.
#: Some entries (``"embracing their"``) are benchmark-specific persona
#: phrasing; the gate is intentionally heuristic.
_PERSONA_INDICATORS: tuple[str, ...] = (
    "born in 19",
    "born in 20",
    "is a ",
    "is the ",
    "is an ",
    "embracing their",
)

#: Markers that indicate a member is conversational turn content rather
#: than a persona description.
_CONVERSATION_MARKERS: tuple[str, ...] = ("[USER]", "[ASSISTANT]", "[SYSTEM]")


def is_persona_only_cluster(
    cluster: list[ThoughtRecord],
    *,
    persona_threshold: float = 0.75,
) -> tuple[bool, float]:
    """Detect a cluster dominated by system-side persona descriptions.

    A member is counted as a persona description when it carries no
    conversation marker (``[USER]`` / ``[ASSISTANT]`` / ``[SYSTEM]``) *and*
    contains at least one persona indicator substring.

    Args:
        cluster: Resolved cluster members.
        persona_threshold: Fraction of members that must look like persona
            descriptions for the cluster to be flagged.  Defaults to
            ``0.75``.

    Returns:
        ``(is_persona, persona_ratio)`` — ``is_persona`` is ``True`` when
        the ratio of persona-looking members meets ``persona_threshold``;
        ``persona_ratio`` is the observed ratio (``0.0`` for an empty
        cluster).

    """
    if not cluster:
        return False, 0.0
    persona_count = 0
    for member in cluster:
        content_lower = member.content.lower()
        has_conversation_marker = any(marker in member.content for marker in _CONVERSATION_MARKERS)
        has_persona_indicator = any(indicator in content_lower for indicator in _PERSONA_INDICATORS)
        if not has_conversation_marker and has_persona_indicator:
            persona_count += 1
    persona_ratio = persona_count / len(cluster)
    return persona_ratio >= persona_threshold, persona_ratio


# ---------------------------------------------------------------------------
# Gate 3 — contradictory members (sentiment lexicon)
# ---------------------------------------------------------------------------

#: Sentiment-opposite token pairs used to flag a cluster whose members
#: directly contradict each other (e.g. "stopped making videos" alongside
#: "created a video, positive feedback").
#:
#: **Limitation:** this lexicon is English-only, so a contradiction
#: expressed only in other languages' words is not detected.
_CONTRADICTION_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset({"stopped", "quit", "abandoned", "dropped", "ended"}),
        frozenset(
            {"started", "began", "made", "created", "produced", "presented"},
        ),
    ),
    (
        frozenset(
            {"hate", "dislike", "disliked", "regret", "regretted", "annoying"},
        ),
        frozenset(
            {
                "love",
                "loved",
                "enjoy",
                "enjoyed",
                "favorite",
                "favourite",
                "amazing",
            },
        ),
    ),
    (
        frozenset({"can't", "couldn't", "unable", "incapable", "failed"}),
        frozenset({"can", "could", "able", "succeeded", "managed"}),
    ),
)

#: Token pattern for the contradiction lexicon — lowercase words including
#: an apostrophe so contractions like ``can't`` match.
_WORD_TOKEN = re.compile(r"\b[a-z']+\b")


def has_contradictory_members(
    cluster: list[ThoughtRecord],
) -> tuple[bool, list[str]]:
    """Detect a cluster whose members contain sentiment-opposite tokens.

    A cluster is flagged when, for some pair in :data:`_CONTRADICTION_PAIRS`,
    one member contains a token from the "negative" set *and* another (or
    the same) member contains a token from the "positive" set.  The check
    is intentionally cluster-wide rather than per-member — a single member
    discussing both sides ("I couldn't do X, but later I managed") will
    still trip the gate, which is the conservative choice for a quality
    filter.

    **Limitation:** English-only — see :data:`_CONTRADICTION_PAIRS`.

    Args:
        cluster: Resolved cluster members.

    Returns:
        ``(is_contradictory, reasons)`` — ``reasons`` lists, for each
        triggered pair, the actually-present negative and positive tokens
        in the form ``"contradiction:<neg>↔<pos>"`` (empty list when the
        cluster is not flagged).

    """
    member_tokens: list[frozenset[str]] = [
        frozenset(_WORD_TOKEN.findall(member.content.lower())) for member in cluster
    ]
    all_tokens: set[str] = set().union(*member_tokens) if member_tokens else set()

    reasons: list[str] = []
    for negative, positive in _CONTRADICTION_PAIRS:
        has_negative = any(tokens & negative for tokens in member_tokens)
        has_positive = any(tokens & positive for tokens in member_tokens)
        if has_negative and has_positive:
            present_neg = sorted(negative & all_tokens)
            present_pos = sorted(positive & all_tokens)
            reasons.append(
                f"contradiction:{','.join(present_neg)}↔{','.join(present_pos)}",
            )
    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# Gate 4 — low cohesion (embedding variance)
# ---------------------------------------------------------------------------


def _pairwise_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the true cosine similarity between two embedding vectors.

    **Design decision — normalisation lives here, at the gate, not at the
    provider boundary.** ``EmbeddingProviderProtocol.embed()`` is documented
    to return an "L2-normalized" vector, but only
    :class:`~engrava.embeddings.sentence_transformer.SentenceTransformerProvider`
    actually does; ``HuggingFaceProvider``, ``OllamaProvider``,
    ``OpenAICompatibleProvider`` and ``CallbackProvider`` all pass through
    whatever magnitude the remote model or user callback returns. Normalising
    at the provider boundary instead would change what *every* consumer of
    those vectors sees (vector-index storage, dedup similarity, any future
    caller doing its own dot product) for a fix whose scope is this one gate.
    Normalising here keeps the blast radius to the cohesion computation:
    every other consumer of a raw provider vector is unaffected, and the
    gate now approximates the quantity its name and the documentation
    promise for every provider — see the Returns section below for where
    floating-point arithmetic still keeps that from being exact.

    **Design decision — a zero-length vector has no direction, so its
    cosine with anything is undefined; this returns ``0.0`` for that case,
    deliberately** rather than raising or silently treating it as either
    "identical" or "orthogonal" by accident. ``0.0`` is the conservative
    choice for a *cohesion* gate: a zero vector normally means an embedding
    failure or empty input, and treating it as similar to anything would
    let a broken vector silently prop up a cluster's cohesion score. This
    happens to match what the pre-0.7 raw-dot-product formula already
    returned for a zero-vector pair (a dot product against the zero vector
    is trivially zero for any finite other vector), so it is not an
    observable behaviour change for that case — what changes is that it is
    now a stated policy rather than an accident of the arithmetic.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        An approximation of the cosine similarity, which is mathematically
        bounded to ``[-1.0, 1.0]`` — but this function's floating-point
        arithmetic is not guaranteed to respect that bound even for
        ordinary, finite input: two identical copies of
        ``[1.0, 2.0**-26]`` compute ``1.0000000000000002``, and scaling
        that same direction by ``3`` computes ``0.9999999999999998``
        instead — not bit-identical, despite representing the same pair
        of directions (see :func:`cluster_cohesion_score` for the full
        executed comparison). Returns ``0.0`` if either vector's norm
        computes to zero — including a non-zero vector whose squared
        components all underflow to zero in ``float64`` (e.g. every
        component near ``1e-200``). This zero-norm check also fires when
        the *other* vector contains ``nan`` or ``inf``: ``[nan]`` paired
        with ``[0.0]`` returns ``0.0``, not ``nan``, because ``vec_b``'s
        norm is genuinely zero and that check is evaluated before
        ``vec_a``'s ``nan`` can propagate through the division. A pair
        that does not hit the zero-norm branch but whose components
        overflow ``float64`` when squared and summed (e.g. ``[1e200]``
        against ``[-1e200]``) returns ``nan`` instead. See
        :func:`cluster_cohesion_score` for which of these executed cases
        change a gate decision relative to the pre-0.7 raw-dot-product
        formula, and in which direction.

    """
    dot = sum(x * y for x, y in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cluster_cohesion_score(cluster_embeddings: list[list[float]]) -> float:
    """Compute the cluster cohesion as mean pairwise cosine similarity.

    Computes an approximation of the true cosine similarity of each pair —
    ``dot(a, b) / (||a|| * ||b||)`` — rather than assuming member
    embeddings are already L2-normalised. Only one shipped embedding
    provider
    (:class:`~engrava.embeddings.sentence_transformer.SentenceTransformerProvider`)
    returns vectors close to unit length; for a provider that does not, a
    raw dot product varies with vector magnitude as well as direction,
    which is not the cosine this gate is named after and documented to
    compute. Dividing by the norms here removes that magnitude sensitivity
    in the common case, but this is floating-point arithmetic, not an
    exact transform: it is not bit-identical even for a provider that
    already normalises. Executed: two identical copies of
    ``[1.0, 2.0**-26]`` score ``1.0000000000000002`` (marginally over the
    ``1.0`` a true self-similarity cannot exceed); the same direction
    scaled by ``3`` scores ``0.9999999999999998`` instead of the same
    value. A pair whose components were originally normalised in single
    (``float32``) precision before being widened to Python floats — a
    realistic shape for a vector already normalised at the source — can
    shift by around ``1e-8`` relative to the raw dot product, not merely
    by a rounding step of the ``float64`` this function computes in; see
    the upgrade notes for the executed pair and the exact figures. See
    :func:`_pairwise_cosine_similarity` for the zero-vector and
    non-finite-input policy. A cluster with fewer than two embeddings is
    trivially cohesive (returns ``1.0``).

    **This changes the gate's output for most pairs whose norm product
    (``||a|| * ||b||``) is not ``1.0`` — but not for every such pair.**
    Two counter-examples, both executed: an orthogonal pair such as
    ``[2.0, 0.0]`` against ``[0.0, 3.0]`` has norm product ``6.0`` yet
    scores ``0.0`` under both the old and the new formula, because the
    dot product itself is ``0.0`` and zero divided by any nonzero norm
    product is still zero; and a pair whose norm product happens to equal
    ``1.0`` (``[2.0, 0.0]`` against ``[0.5, 0.0]``) scores ``1.0`` under
    both formulas too, by coincidence. A ``cluster_quality_cohesion_threshold``
    tuned against the old (magnitude-sensitive) score is not automatically
    tuned against a different function just because a provider returns
    non-unit vectors — see the upgrade notes for the executed cases that
    do move, and in which direction (it depends on the sign of each
    pair's cosine as well as its norm product, so pairs in the same
    cluster can shift oppositely).

    **Non-finite and overflowing input.** If any pairwise cosine is
    ``nan`` and neither vector in that pair hit the zero-norm branch (see
    :func:`_pairwise_cosine_similarity`), the mean that averages it in is
    ``nan`` too — ``nan`` propagates through the sum and division
    unconditionally, regardless of how many other pairs in the cluster
    are well-behaved. This function does not detect or reject that case;
    it returns ``nan`` as computed. :func:`is_low_cohesion` compares that
    ``nan`` against its threshold with ``<``, which Python (and IEEE 754)
    defines as ``False`` for any comparison involving ``nan``, so a
    cluster with a ``nan`` cohesion score is reported as *not* low-cohesion
    and passes this gate. That admit-on-``nan`` comparison predates this
    change and is not addressed by this function — but which *inputs*
    reach it, and which instead reach the zero-norm branch and score
    ``0.0``, changed here, in both directions, executed:

    * ``[nan]`` paired with ``[0.0]`` scored ``nan`` under the pre-0.7
      raw-dot-product formula (``nan * 0.0`` is ``nan``) and scores
      ``0.0`` here (:func:`_pairwise_cosine_similarity`'s zero-norm check
      on the genuinely-zero vector fires first). ``nan`` compares as "not
      low cohesion" (admitted); ``0.0`` compares as low cohesion for any
      positive threshold (rejected) — so this pair's cluster flips from
      admitted to rejected relative to the pre-0.7 formula.
    * ``[1e200]`` paired with ``[-1e200]`` scored ``-inf`` under the
      pre-0.7 raw-dot-product formula (correctly rejected, since ``-inf``
      is below any threshold) and scores ``nan`` here, because both norms
      overflow to ``inf`` and ``-inf / inf`` is ``nan`` — so this pair's
      cluster flips from rejected to admitted, the opposite direction.

    Neither outcome was designed; both are accidents of floating-point
    arithmetic, on the old formula and this one alike. See the upgrade
    notes for the full executed comparison, including a case that does
    *not* change (an overflowing orthogonal pair stays rejected under
    both formulas).

    Args:
        cluster_embeddings: One embedding vector per cluster member.

    Returns:
        An approximation of the mean pairwise cosine, mathematically
        bounded to ``[-1.0, 1.0]`` (practically ``[0.0, 1.0]`` for
        sentence-transformer embeddings) but not guaranteed to respect
        that bound in floating-point arithmetic even for ordinary finite
        input — see the executed ``1.0000000000000002`` case above.
        Higher means tighter cluster. Returns ``nan`` if any pairwise
        cosine is ``nan``, and ``0.0`` — not ``nan`` — for a pairwise
        comparison that hit the zero-norm branch despite one side
        containing ``nan``/``inf``; see the non-finite-and-overflowing
        note above for both directions this can move a gate decision.

    """
    n = len(cluster_embeddings)
    if n < 2:  # noqa: PLR2004 -- "two members" is the natural lower bound
        return 1.0
    total_similarity = 0.0
    pair_count = 0
    for i in range(n):
        vec_a = cluster_embeddings[i]
        for j in range(i + 1, n):
            vec_b = cluster_embeddings[j]
            total_similarity += _pairwise_cosine_similarity(vec_a, vec_b)
            pair_count += 1
    return total_similarity / pair_count if pair_count else 1.0


def is_low_cohesion(
    cluster_embeddings: list[list[float]],
    *,
    cohesion_threshold: float = 0.40,
) -> tuple[bool, float]:
    """Detect a cluster whose mean pairwise cosine is below a threshold.

    Args:
        cluster_embeddings: One embedding vector per cluster member.
        cohesion_threshold: Cohesion strictly below this value flags the
            cluster.  Defaults to ``0.40`` (calibrated on short07 evidence;
            sentence-transformer member cosines cluster around 0.3-0.7).

    Returns:
        ``(is_loose, cohesion)`` — ``is_loose`` is ``True`` when
        ``cohesion < cohesion_threshold``; ``cohesion`` is the observed
        mean pairwise cosine, which is ``nan`` when
        :func:`cluster_cohesion_score` computed a non-finite pairwise
        value (see its docstring). ``nan < cohesion_threshold`` is
        ``False`` for any threshold, so a ``nan`` cohesion reports
        ``is_loose=False`` — the cluster is treated as *not* low-cohesion
        and admitted. That comparison itself is pre-existing behaviour,
        unchanged by this function — but as of the 0.6 -> 0.7 change,
        which inputs actually produce ``nan`` (and are therefore admitted
        by this comparison) versus ``0.0`` (and are therefore rejected)
        is not the same as before: see :func:`cluster_cohesion_score`'s
        non-finite-and-overflowing note for two executed input shapes
        that now land on the opposite side of this gate from where the
        pre-0.7 formula put them.

    """
    score = cluster_cohesion_score(cluster_embeddings)
    return score < cohesion_threshold, score


# ---------------------------------------------------------------------------
# Gate 5 — external-source homogeneity (self-anchored provenance semantic)
# ---------------------------------------------------------------------------


def is_external_source_homogeneous(
    cluster: list[ThoughtRecord],
    *,
    min_external_fraction: float = 0.95,
) -> tuple[bool, float]:
    """Detect a cluster whose members come predominantly from external sources.

    The binding provenance signal is
    ``metadata["source"]["is_self"]`` (a ``bool``), *not* the legacy
    ``metadata["role"]`` string (kept only as a debug-only
    ``role_hint``).  A member with ``is_self`` ``True`` is the
    agent's own thought; ``False`` is external; anything else — missing
    ``source``, missing ``is_self``, or a malformed (non-mapping)
    ``source`` — is treated as **external** under the safe-fallback rule
    so that legacy data ingested before the role-aware path was deployed
    passes through unchanged.

    This gate is belt-and-suspenders: the upstream eligibility filter
    already keeps the dreaming pool to external thoughts, so under normal
    operation every member is external and the gate is a no-op.  It exists
    to catch clusters that slip through if the filter is bypassed or if a
    later ingest path forgets to populate ``metadata.source``.

    Args:
        cluster: Resolved cluster members.
        min_external_fraction: Minimum fraction of external-source members
            required for the cluster to pass.  Defaults to ``0.95``.

    Returns:
        ``(is_homogeneous, external_fraction)`` — ``is_homogeneous`` is
        ``True`` when ``external_fraction >= min_external_fraction``;
        ``external_fraction`` is the observed ratio (``1.0`` for an empty
        cluster — vacuously homogeneous).

    """
    if not cluster:
        return True, 1.0
    external_count = 0
    for member in cluster:
        source = member.metadata.get("source")
        if not isinstance(source, dict):
            # Missing or malformed source → safe-fallback: treat as external.
            external_count += 1
            continue
        is_self = source.get("is_self")
        if is_self is not True:
            # False or missing is_self → external (safe-fallback for missing).
            external_count += 1
    external_fraction = external_count / len(cluster)
    return external_fraction >= min_external_fraction, external_fraction


# ---------------------------------------------------------------------------
# Gate 6 — named-entity consistency
# ---------------------------------------------------------------------------


def has_consistent_entities(
    cluster: list[ThoughtRecord],
    *,
    min_shared_ratio: float = 0.60,
) -> tuple[bool, float]:
    """Detect whether cluster members share named entities with the first member.

    Named entities are extracted per member via
    :func:`engrava.extensions.dreaming_reflection_content.extract_named_entities_per_member`
    — the same lightweight regex extractor that feeds the v2 content
    builder's ``named_entities`` field, so the two consumers agree on
    what counts as an entity.

    A cluster is considered consistent when, counting the first member
    itself, at least ``min_shared_ratio`` of members share at least one
    named entity with the first member.  Two design choices to be aware
    of:

    * **Anchored on the first member.**  The first member's entity set
      is the reference; other members are scored by whether they
      intersect it.  This is asymmetric on purpose — a "cluster about
      Alice" should not pass just because the second half of members
      happen to share an entity with each other but not with Alice.
    * **Extraction-failure escape.**  If the first member yields no
      named entities at all the gate passes unconditionally (returning
      ``(True, 1.0)``).  Extraction failure must not block REFLECTION
      creation; this matches the broader safe-fallback discipline used
      by the other gates.

    Args:
        cluster: Resolved cluster members.
        min_shared_ratio: Minimum fraction of members that must overlap
            on named entities with the first member.  Defaults to
            ``0.60``.

    Returns:
        ``(is_consistent, shared_ratio)`` — ``is_consistent`` is ``True``
        when the observed ratio meets ``min_shared_ratio``;
        ``shared_ratio`` is the observed overlap ratio (``1.0`` for an
        empty cluster, a single-member cluster, or a cluster where the
        first member has no named entities).

    """
    if len(cluster) < 2:  # noqa: PLR2004 -- single-member cluster is vacuously consistent
        return True, 1.0
    members_entities: list[set[str]] = [
        set(extract_named_entities_per_member(member)) for member in cluster
    ]
    first_entities = members_entities[0]
    if not first_entities:
        return True, 1.0
    overlap_count = sum(1 for entities in members_entities[1:] if first_entities & entities)
    shared_ratio = (overlap_count + 1) / len(cluster)
    return shared_ratio >= min_shared_ratio, shared_ratio


# ---------------------------------------------------------------------------
# Gate 8 — meaningful keyphrases
# ---------------------------------------------------------------------------

#: Determiner / vague-quantifier words that, when they lead a keyphrase,
#: make that phrase generic ("these moments", "specific projects").
_GENERIC_KEYPHRASE_LEADERS: tuple[str, ...] = (
    "these",
    "those",
    "this",
    "that",
    "specific",
    "various",
    "certain",
    "some",
    "many",
    "few",
    "several",
    "their",
    "its",
    "other",
    "new",
    "old",
    "recent",
    "past",
    "future",
)

#: Pre-compiled pattern: keyphrase that starts with a generic leader word
#: followed by at least one more word.  ``re.IGNORECASE`` so capitalisation
#: does not matter.
_GENERIC_KEYPHRASE_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(_GENERIC_KEYPHRASE_LEADERS) + r")\s+\w+",
    flags=re.IGNORECASE,
)


def has_meaningful_keyphrases(
    cluster_keyphrases: list[dict[str, float | str]],
) -> bool:
    """Detect whether a cluster's top keyphrases include at least one non-generic phrase.

    A keyphrase is "generic" when it starts with a determiner / vague
    quantifier (``these``, ``specific``, ``their``, …) followed by another
    word.  A cluster passes this gate when at least one of its
    ``top_keyphrases`` entries is *not* generic.  An empty keyphrase list
    fails the gate — a cluster with no keyphrases at all has nothing
    meaningful to anchor a REFLECTION on.

    Non-string ``phrase`` values are treated defensively as generic (they
    cannot be a meaningful phrase).

    Args:
        cluster_keyphrases: The ``top_keyphrases`` list from the v2 content
            builder — ``{"phrase": str, "score": float}`` dicts, already
            run through the cross-cluster TF-IDF boilerplate filter.

    Returns:
        ``True`` when at least one keyphrase is non-generic; ``False`` for
        an empty list or an all-generic list.

    """
    for keyphrase in cluster_keyphrases:
        phrase = keyphrase.get("phrase")
        if not isinstance(phrase, str):
            continue
        if not _GENERIC_KEYPHRASE_PATTERN.match(phrase):
            return True
    return False

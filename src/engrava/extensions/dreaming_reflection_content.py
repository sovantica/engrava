"""Build structural REFLECTION content (schema v2) from a clustered thought set.

Public surface is :func:`build_reflection_content_v2`, a pure
deterministic function the dreaming extension calls once per cluster
on its way to creating the corresponding REFLECTION thought.

The v2 schema is additive over the legacy v1 layout (3 fields:
``member_ids``, ``keywords``, ``cluster_hash``):

* Legacy v1 fields are preserved verbatim so v1-aware readers keep
  working without changes.
* All seven mandated fields from the cognitive-boundary REFLECTION
  schema (``type``, ``version``, ``member_ids``, ``keywords``,
  ``member_count``, ``cluster_algorithm``, ``created_at``) are
  produced — closing a long-standing gap where the legacy emitter
  populated only three of the seven.
* Four new structural enrichments are added (``top_keyphrases``,
  ``member_excerpts``, ``temporal_span``, ``named_entities``) that
  give downstream readers (LLM judges, semantic search, agent
  retrieval) substantive surface area without crossing the
  cognitive boundary into LLM territory.

Backward-compat dispatch lives in :func:`parse_reflection_content`,
which handles both legacy v1 (no ``version`` field) and v2 via
``"version" not in content_dict`` detection.

All functions in this module are LLM-free (validated by the
cognitive-boundary guard test).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar

from engrava.extensions.dreaming_keyphrases import (
    SENTENCE_STARTER_BLOCKLIST,
    extract_simple_keywords,
    is_boilerplate_phrase,
    top_keyphrases_tfidf,
)

if TYPE_CHECKING:
    from engrava.config import DreamingConfig
    from engrava.domain.models.thought import ThoughtRecord

__all__ = (
    "MemberExcerpt",
    "ReflectionContentV2",
    "TemporalSpan",
    "build_reflection_content_v2",
    "extract_named_entities_per_member",
    "parse_reflection_content",
)


class MemberExcerpt(TypedDict):
    """A cluster member's id paired with a bounded content excerpt.

    Attributes:
        thought_id: The member thought's stable identifier.
        excerpt: The member's ``content`` truncated at a word boundary to
            the configured maximum length.

    """

    thought_id: str
    excerpt: str


class TemporalSpan(TypedDict):
    """Creation-time bounds and span across a cluster's members.

    Attributes:
        min_created_at: Earliest member ``created_at`` (ISO-8601), or the
            fallback timestamp when no member carries one.
        max_created_at: Latest member ``created_at`` (ISO-8601), or the
            fallback timestamp when no member carries one.
        span_days: Day span between the bounds, rounded to two decimals.

    """

    min_created_at: str
    max_created_at: str
    span_days: float


class ReflectionContentV2(TypedDict):
    """Structural REFLECTION content payload (schema v2).

    The stable, documented schema produced by
    :func:`build_reflection_content_v2`. Additive over legacy v1 (which
    carried only ``member_ids``, ``keywords`` and ``cluster_hash``): all
    seven mandated cognitive-boundary fields plus four structural
    enrichments are present. See the module docstring for the field-by-field
    rationale.

    Attributes:
        type: Discriminator, always ``"reflection"``.
        version: Schema version, always ``2``.
        member_ids: Sorted member thought identifiers.
        keywords: Simple frequency keywords (<=10) across member content.
        cluster_hash: Legacy 16-char SHA-256 prefix over ``member_ids``.
        member_count: Number of members in the cluster.
        cluster_algorithm: Clustering algorithm name.
        created_at: ISO-8601 build timestamp.
        top_keyphrases: TF-IDF keyphrase dicts (``{"phrase", "score"}``).
        member_excerpts: Top members with bounded content excerpts.
        temporal_span: Creation-time bounds and span days.
        named_entities: Sorted unique entities drawn from every member.

    """

    type: Literal["reflection"]
    version: Literal[2]
    member_ids: list[str]
    keywords: list[str]
    cluster_hash: str
    member_count: int
    cluster_algorithm: str
    created_at: str
    top_keyphrases: list[dict[str, float | str]]
    member_excerpts: list[MemberExcerpt]
    temporal_span: TemporalSpan
    named_entities: list[str]


_EXCERPT_LENGTH = 80
_EXCERPT_TRUNCATION_SUFFIX = "..."
_NAMED_ENTITY_MIN_LENGTH = 3
_CAPITALIZED_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
_YEAR_OR_MEASUREMENT = re.compile(r"\b(?:\d{4}|\d+(?:km|min|s|hr|h|m|kg)|\$\d+)\b")
_FALLBACK_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def _truncate_at_word_boundary(text: str, *, max_length: int) -> str:
    """Truncate *text* to <= *max_length* characters, falling back on a word break.

    Appends an ellipsis suffix when truncation occurs.  Returns *text*
    unchanged if it already fits.

    Args:
        text: Source string.
        max_length: Hard upper bound on returned length, **including**
            the trailing ``"..."`` when truncation happens.

    Returns:
        The truncated string, or *text* itself when no truncation is
        necessary.

    """
    if len(text) <= max_length:
        return text

    budget = max_length - len(_EXCERPT_TRUNCATION_SUFFIX)
    if budget <= 0:
        return text[:max_length]

    head = text[:budget]
    last_space = head.rfind(" ")
    if last_space > 0:
        head = head[:last_space]
    return head.rstrip() + _EXCERPT_TRUNCATION_SUFFIX


def _build_member_excerpts(
    cluster: list[ThoughtRecord],
    *,
    top_n: int,
    max_length: int = _EXCERPT_LENGTH,
) -> list[MemberExcerpt]:
    """Top-N members by priority + recency, each with a bounded excerpt.

    Sort key: ``(Priority.__lt__, created_at-or-fallback descending)``.
    ``ThoughtRecord.created_at`` is ``str | None``; the fallback string
    keeps ``sorted`` stable when timestamps are missing.

    Args:
        cluster: Member ThoughtRecords.
        top_n: Maximum number of members to include.
        max_length: Hard upper bound on every excerpt length
            (including the trailing ellipsis).  Defaults to
            ``_EXCERPT_LENGTH`` so existing callers keep their old
            behaviour; the v2 builder now threads
            ``DreamingConfig.member_excerpt_max_chars`` through to
            give operators a single config knob over excerpt size.

    Returns:
        List of :class:`MemberExcerpt` mappings.

    """
    ordered = sorted(
        cluster,
        key=lambda t: (t.priority, _negative_iso(t.created_at)),
    )
    return [
        {
            "thought_id": member.thought_id,
            "excerpt": _truncate_at_word_boundary(member.content, max_length=max_length),
        }
        for member in ordered[:top_n]
    ]


def _negative_iso(value: str | None) -> str:
    """Return a sort key that orders None last, then ISO timestamps descending.

    The trick: ISO-8601 strings sort lexicographically the same way they
    sort temporally.  We invert each character to flip ascending into
    descending while preserving the lexicographic invariant.

    Args:
        value: ``ThoughtRecord.created_at`` or ``None``.

    Returns:
        A surrogate string that, when sorted ascending, yields the
        equivalent of "newest first" with ``None`` pushed to the end.

    """
    if value is None:
        return "\xff" * 32  # comes after every printable ISO
    return "".join(chr(0xFFFF - ord(c)) for c in value)


def _build_temporal_span(cluster: list[ThoughtRecord]) -> TemporalSpan:
    """Compute min/max ``created_at`` and span days across cluster members.

    Members with ``created_at is None`` are skipped.  When every member
    has ``None`` we fall back to ``_FALLBACK_TIMESTAMP`` for both bounds
    and ``0.0`` span days.

    Args:
        cluster: Member ThoughtRecords.

    Returns:
        A :class:`TemporalSpan` with ``min_created_at`` (str),
        ``max_created_at`` (str), and ``span_days`` (float, 2dp).

    """
    timestamps = [m.created_at for m in cluster if m.created_at is not None]
    if not timestamps:
        return {
            "min_created_at": _FALLBACK_TIMESTAMP,
            "max_created_at": _FALLBACK_TIMESTAMP,
            "span_days": 0.0,
        }
    parsed = [datetime.datetime.fromisoformat(ts) for ts in timestamps]
    min_dt = min(parsed)
    max_dt = max(parsed)
    span = (max_dt - min_dt).total_seconds() / 86400.0
    return {
        "min_created_at": min_dt.isoformat(),
        "max_created_at": max_dt.isoformat(),
        "span_days": round(span, 2),
    }


def extract_named_entities_per_member(thought: ThoughtRecord) -> list[str]:
    """Extract named entities from a single thought's content.

    Lightweight regex-based extraction — NOT a proper NER.  Picks up
    capitalised tokens (filtered against the shared sentence-starter
    blocklist and a minimum length) plus year-or-measurement patterns
    (e.g. ``"1974"``, ``"5kg"``).  The result is sorted-unique so callers
    that compare entity sets across members get byte-stable input.

    This is the per-member primitive used by both the v2 content
    builder (aggregated across the cluster via
    :func:`_extract_named_entities`) and the cluster-quality gate that
    checks named-entity consistency across members.  Keeping the
    extraction in one place ensures the two consumers agree on what
    counts as an entity.

    Args:
        thought: The thought whose ``content`` to scan.

    Returns:
        Sorted list of unique entity strings extracted from
        ``thought.content``.  Empty list when the content yields no
        entities.

    """
    entities: set[str] = set()
    for match in _CAPITALIZED_TOKEN.finditer(thought.content):
        token = match.group(0)
        # Drop common sentence-starters / connectives / role-marker
        # capitalisations — they slip through the regex but are never
        # proper nouns.  See ``SENTENCE_STARTER_BLOCKLIST`` in
        # ``dreaming_keyphrases`` for the canonical list.
        if token in SENTENCE_STARTER_BLOCKLIST:
            continue
        if len(token) >= _NAMED_ENTITY_MIN_LENGTH:
            entities.add(token)
    for match in _YEAR_OR_MEASUREMENT.finditer(thought.content):
        entities.add(match.group(0))
    return sorted(entities)


def _extract_named_entities(cluster: list[ThoughtRecord]) -> list[str]:
    """Aggregate named entities across the whole cluster.

    Delegates per-member extraction to
    :func:`extract_named_entities_per_member` and returns the
    sorted-unique union, keeping the v2 content dict byte-stable
    across runs.

    Args:
        cluster: Member ThoughtRecords.

    Returns:
        Sorted list of unique entity strings drawn from every member.

    """
    aggregated: set[str] = set()
    for member in cluster:
        aggregated.update(extract_named_entities_per_member(member))
    return sorted(aggregated)


def _legacy_cluster_hash(member_ids: list[str]) -> str:
    """Reproduce the legacy ``cluster_hash`` derivation for v2 emit.

    The dreaming extension uses ``sha256(json.dumps(member_ids))[:16]``
    when stamping its ``source`` field; the v2 builder mirrors the same
    derivation so the legacy ``cluster_hash`` field stays meaningful for
    downstream consumers that key on it.

    Args:
        member_ids: Stable-ordered list of member thought IDs.

    Returns:
        Hex-encoded 16-char prefix of the SHA-256 over the JSON
        encoding of *member_ids*.

    """
    return hashlib.sha256(json.dumps(member_ids).encode()).hexdigest()[:16]


def _apply_boilerplate_filter(
    raw_keyphrases: list[dict[str, float | str]],
    *,
    cluster_phrase_df: dict[str, int] | None,
    total_clusters: int | None,
    config: DreamingConfig,
) -> list[dict[str, float | str]]:
    """Strip corpus-wide boilerplate from a cluster's raw keyphrase list.

    The filter engages only when *both* ``cluster_phrase_df`` and
    ``total_clusters`` are supplied — passing either side alone is
    treated as "filter disabled" so the helper stays trivially
    backward-compatible with callers that have not adopted the
    two-pass orchestration yet.  Filtering is delegated to
    :func:`is_boilerplate_phrase` from
    ``engrava.extensions.dreaming_keyphrases`` with the configured
    threshold and minimum-corpus knobs.

    A fallback guard preserves the raw list when filtering would
    strip too many entries: if fewer than
    ``config.boilerplate_min_keyphrases_per_refl`` keyphrases survive,
    the unfiltered list is returned.  This prevents the filter from
    emptying ``top_keyphrases`` on REFLECTIONs whose every phrase
    happens to look like boilerplate (degenerate corpora, very tight
    clusters), at the cost of letting the boilerplate through in that
    edge case.

    Args:
        raw_keyphrases: TF-IDF keyphrases as returned by
            ``top_keyphrases_tfidf``.
        cluster_phrase_df: Cross-cluster phrase document frequency.
        total_clusters: Total number of clusters in the same run.
        config: Dreaming config carrying the boilerplate knobs.

    Returns:
        Filtered keyphrase list, or the raw list when filtering is
        disabled or when the fallback guard fires.

    """
    if cluster_phrase_df is None or total_clusters is None:
        return raw_keyphrases

    filtered = [
        kp
        for kp in raw_keyphrases
        if isinstance(kp.get("phrase"), str)
        and not is_boilerplate_phrase(
            str(kp["phrase"]),
            cluster_phrase_df,
            total_clusters,
            threshold=config.boilerplate_threshold,
            min_corpus_size=config.boilerplate_min_corpus_size,
        )
    ]
    if len(filtered) < config.boilerplate_min_keyphrases_per_refl:
        return raw_keyphrases
    return filtered


def build_reflection_content_v2(
    cluster: list[ThoughtRecord],
    *,
    algorithm: str,
    config: DreamingConfig,
    corpus: list[str] | None = None,
    cluster_phrase_df: dict[str, int] | None = None,
    total_clusters: int | None = None,
    now: datetime.datetime | None = None,
) -> ReflectionContentV2:
    """Build a v2 REFLECTION content dict from a cluster.

    Pure function: identical inputs produce byte-identical output (the
    *now* parameter must be supplied for deterministic tests; in
    production the dreaming extension passes the current cycle's UTC
    timestamp).

    Args:
        cluster: Member ThoughtRecords (post-clustering).  Real class
            is ``ThoughtRecord`` per the engrava domain layer (Pydantic
            ``frozen=True``).
        algorithm: Cluster algorithm name resolved at the dreaming
            callsite from ``gates.cluster_algorithm`` — passed
            explicitly because the dreaming extension does not retain
            that value as instance state.
        config: ``DreamingConfig`` carrying ``top_keyphrases_count``,
            ``top_member_excerpts_count`` and (when boilerplate
            filtering is requested) the
            ``boilerplate_threshold`` / ``boilerplate_min_corpus_size``
            / ``boilerplate_min_keyphrases_per_refl`` knobs.
        corpus: Flat list of all content strings in the parent thought
            corpus, used as the IDF document set for keyphrase scoring.
            Defaults to ``[]`` (degenerates to TF-only ranking).
        cluster_phrase_df: Optional cross-cluster phrase document
            frequency, as produced by
            :func:`compute_cluster_phrase_frequency` over every
            cluster's raw keyphrases from the same dreaming run.  When
            paired with ``total_clusters`` (also non-``None``) the
            keyphrase list is filtered via
            :func:`is_boilerplate_phrase` against the configured
            threshold.  ``None`` (default) disables the filter — the
            raw TF-IDF keyphrase list is returned unchanged, matching
            the pre-extension behaviour.  Supplying one side of the
            pair without the other is treated as "filter disabled" for
            the same backward-compat reason.
        total_clusters: Total number of clusters scanned in the
            dreaming run that produced ``cluster_phrase_df``.  Must be
            supplied together with ``cluster_phrase_df`` for the
            filter to engage.
        now: UTC timestamp stamped into ``created_at``.  Defaults to
            ``datetime.now(UTC)``; tests should always pass a fixed
            value to keep assertions stable.

    Returns:
        A :class:`ReflectionContentV2` payload ready to ``json.dumps`` and
        persist.

    """
    if now is None:
        now = datetime.datetime.now(datetime.UTC)
    effective_corpus = corpus if corpus is not None else []

    member_ids = sorted(member.thought_id for member in cluster)
    member_contents = [member.content for member in cluster]

    keywords = extract_simple_keywords(member_contents, top_n=10)
    raw_keyphrases = top_keyphrases_tfidf(
        cluster,
        corpus=effective_corpus,
        top_n=config.top_keyphrases_count,
    )
    keyphrases = _apply_boilerplate_filter(
        raw_keyphrases,
        cluster_phrase_df=cluster_phrase_df,
        total_clusters=total_clusters,
        config=config,
    )
    excerpts = _build_member_excerpts(
        cluster,
        top_n=config.top_member_excerpts_count,
        max_length=config.member_excerpt_max_chars,
    )
    temporal_span = _build_temporal_span(cluster)
    named_entities = _extract_named_entities(cluster)

    return {
        "type": "reflection",
        "version": 2,
        "member_ids": member_ids,
        "keywords": keywords,
        "cluster_hash": _legacy_cluster_hash(member_ids),
        "member_count": len(member_ids),
        "cluster_algorithm": algorithm,
        "created_at": now.isoformat(),
        "top_keyphrases": keyphrases,
        "member_excerpts": excerpts,
        "temporal_span": temporal_span,
        "named_entities": named_entities,
    }


_ReflectionContentT = TypeVar("_ReflectionContentT", bound=Mapping[str, object])
"""A REFLECTION content mapping: a v1/v2 ``dict`` or a :class:`ReflectionContentV2`.

Bounding to a read-only :class:`~collections.abc.Mapping` lets the dispatch parser
accept a builder's ``TypedDict`` result (or a JSON-decoded ``dict``) and echo back
the caller's *exact* type — it only reads keys and returns its argument unchanged.
"""


def parse_reflection_content(content_dict: _ReflectionContentT) -> _ReflectionContentT:
    """Dispatch parser handling both legacy v1 and v2 REFLECTION content.

    Legacy v1 has no ``version`` field at all; this function detects it
    via key absence and returns the dict as-is (callers that read only
    legacy fields keep working).  v2 is detected via
    ``content_dict["version"] == 2`` and returned verbatim.

    Args:
        content_dict: Parsed JSON object from ``thought.content``.

    Returns:
        The same dict (this function is a *dispatch* contract — it
        validates the shape and lets callers consume known fields).

    Raises:
        ValueError: When ``version`` is present but neither ``2`` nor
            a future version this dispatcher knows about.

    """
    if "version" not in content_dict:
        return content_dict
    version = content_dict["version"]
    if version == 2:  # noqa: PLR2004
        return content_dict
    msg = (
        f"Unsupported reflection content version: {version}. "
        "Supported: legacy v1 (no version field), v2."
    )
    raise ValueError(msg)

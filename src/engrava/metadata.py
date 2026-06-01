"""Convenience helpers for constructing self-anchored thought metadata.

The thought-graph distinguishes three perspectives on every stored
piece of content:

* ``percept`` — input arriving at the agent (a user message, a sensor
  reading, a retrieved document). Anchored to an external source.
* ``utterance`` — output the agent sent to the world (a reply, a
  rendered response). Always self-anchored.
* ``thought`` — the agent's own internal cognition (a reflection, a
  plan, an internal signal). Always self-anchored.

These functions build the small structured ``metadata`` dictionary the
persistence layer recognises for each perspective. They are pure: the
same arguments always return the same dictionary, and the returned
value carries no shared state. Callers are free to pass a literal
dictionary instead; the helpers simply remove a class of typo-driven
shape mismatches at the call site.

Examples:
    >>> percept(is_self=False, source_id="user-42")["perspective"]
    'percept'
    >>> utterance()["source"]["is_self"]
    True
    >>> thought()["perspective"]
    'thought'

"""

from __future__ import annotations

from typing import Literal

from engrava.domain.models.thought import MetadataValue

__all__ = ["percept", "thought", "utterance"]


ConfidenceLevel = Literal["high", "medium", "low"]


def percept(
    *,
    is_self: bool = False,
    source_id: str | None = None,
    label: str | None = None,
    confidence: ConfidenceLevel = "high",
    lang: str = "en",
) -> dict[str, MetadataValue]:
    """Build metadata for content arriving at the agent from the outside.

    Use this for user messages, retrieved documents, sensor readings, or
    any other observation the agent did not author itself. The returned
    dictionary carries the ``"percept"`` perspective and a structured
    ``source`` block pinning the origin.

    Args:
        is_self: ``True`` only when the percept is the agent observing
            its own state (rare; ``False`` is the common case for
            external input).
        source_id: Stable identifier of the external source (e.g. user
            identifier, document URI). Omitted when unknown.
        label: Human-readable label for the source (e.g. ``"user"``,
            ``"transcript"``). Omitted when unset.
        confidence: Reliability rating attached to the source. Defaults
            to ``"high"``; lower the value for noisy channels.
        lang: BCP-47 language tag of the content (``"en"`` by default).

    Returns:
        A metadata dictionary suitable for the ``metadata=`` argument of
        ``ThoughtRecord`` and persistence-layer writes.

    """
    source: dict[str, MetadataValue] = {
        "is_self": is_self,
        "confidence": confidence,
    }
    if source_id is not None:
        source["id"] = source_id
    if label is not None:
        source["label"] = label
    return {
        "perspective": "percept",
        "source": source,
        "lang": lang,
        "content_type": "natural_language",
    }


def utterance(*, lang: str = "en") -> dict[str, MetadataValue]:
    """Build metadata for content the agent itself produced and sent out.

    Use this for the agent's outgoing replies, generated artefacts, or
    any content the agent authored and surfaced externally. The
    perspective is always ``"utterance"`` and the source is always
    self-anchored.

    Args:
        lang: BCP-47 language tag of the content (``"en"`` by default).

    Returns:
        A metadata dictionary suitable for the ``metadata=`` argument of
        ``ThoughtRecord`` and persistence-layer writes.

    """
    return {
        "perspective": "utterance",
        "source": {"is_self": True, "confidence": "high"},
        "lang": lang,
        "content_type": "natural_language",
    }


def thought(*, lang: str = "en") -> dict[str, MetadataValue]:
    """Build metadata for the agent's own internal cognition.

    Use this for reflections, plans, internal signals, or any content
    the agent generated for itself rather than for the outside world.
    The perspective is always ``"thought"`` and the source is always
    self-anchored.

    Args:
        lang: BCP-47 language tag of the content (``"en"`` by default).

    Returns:
        A metadata dictionary suitable for the ``metadata=`` argument of
        ``ThoughtRecord`` and persistence-layer writes.

    """
    return {
        "perspective": "thought",
        "source": {"is_self": True, "confidence": "high"},
        "lang": lang,
        "content_type": "natural_language",
    }

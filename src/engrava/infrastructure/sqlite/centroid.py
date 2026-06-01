"""Centroid math for REFLECTION cluster summaries.

A REFLECTION's centroid embedding is the L2-normalized arithmetic mean of
its member thoughts' embedding vectors. It is the single semantic anchor a
REFLECTION is recalled by, so the same deterministic computation must be
used both when the REFLECTION is first created and whenever it is re-bound
to its (now-evolved) members. This module holds that one computation so the
creation path and the re-bind path cannot drift apart.

The function is pure (no I/O, no model call): it is a plain numeric
reduction over vectors already held in memory, keeping the no-LLM boundary
intact.
"""

from __future__ import annotations

import math

#: Model identifier under which a REFLECTION centroid embedding is stored.
#: Both the creation path and the re-bind path write the centroid under this
#: name so the upsert in ``store_embedding`` overwrites the same row in place.
CENTROID_MODEL_NAME = "dreaming-centroid"


def compute_centroid(member_vectors: list[list[float]]) -> list[float]:
    """Compute the L2-normalized mean of a cluster's member vectors.

    The centroid is the arithmetic mean of the member vectors, normalized to
    unit length. When the mean is the zero vector (e.g. members cancel out),
    it is returned un-normalized (all zeros) rather than dividing by zero.

    Args:
        member_vectors: Non-empty list of equal-length embedding vectors.

    Returns:
        The centroid vector, normalized to unit length when its magnitude is
        positive.

    Raises:
        ValueError: If ``member_vectors`` is empty.

    Examples:
        >>> compute_centroid([[1.0, 0.0], [0.0, 1.0]])
        [0.7071067811865475, 0.7071067811865475]

    """
    if not member_vectors:
        msg = "compute_centroid requires at least one member vector"
        raise ValueError(msg)

    dim = len(member_vectors[0])
    count = len(member_vectors)
    centroid = [sum(v[i] for v in member_vectors) / count for i in range(dim)]

    norm = math.sqrt(sum(x * x for x in centroid))
    if norm > 0.0:
        centroid = [x / norm for x in centroid]
    return centroid

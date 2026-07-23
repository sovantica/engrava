"""Compatibility exports for the legacy SQLite centroid import path.

The deterministic implementation lives in :mod:`engrava.domain.dreaming`.
"""

from engrava.domain.dreaming import CENTROID_MODEL_NAME, compute_centroid

__all__ = ["CENTROID_MODEL_NAME", "compute_centroid"]

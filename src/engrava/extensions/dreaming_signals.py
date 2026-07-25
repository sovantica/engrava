"""Compatibility exports for the legacy Dreaming signal import path.

The backend-independent implementations live in :mod:`engrava.domain.dreaming`.
This module remains as an import-compatible facade for existing consumers.
"""

from engrava.domain.dreaming import (
    DEFAULT_SIGNALS,
    ActionOutcomeSignal,
    ConfidenceSignal,
    ConfirmationSignal,
    DreamingContext,
    DreamingSignalProtocol,
    FrequencySignal,
    RecencySignal,
    StalenessSignal,
    default_signal_active,
)

__all__ = [
    "DEFAULT_SIGNALS",
    "ActionOutcomeSignal",
    "ConfidenceSignal",
    "ConfirmationSignal",
    "DreamingContext",
    "DreamingSignalProtocol",
    "FrequencySignal",
    "RecencySignal",
    "StalenessSignal",
    "default_signal_active",
]

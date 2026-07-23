"""Internal composition root for optional Engrava components."""

from __future__ import annotations

from engrava.config import DreamingConfig
from engrava.domain.protocols.dreaming import DreamingConsolidatorProtocol
from engrava.extensions.dreaming import DreamingExtension


def compose_dreaming_consolidator(config: DreamingConfig) -> DreamingConsolidatorProtocol:
    """Build the configured Dreaming implementation at the application boundary.

    Args:
        config: Parsed, enabled Dreaming configuration.

    Returns:
        A backend-independent consolidator implemented by Dreaming.

    """
    return DreamingExtension(config=config)

"""Extension manifest discovery helper.

Provides an **opt-in** utility to discover installed extension packages by
scanning the ``engrava.extensions`` entry-point group.  The result is
intended to be passed explicitly to ``SqliteEngravaCore``::

    from engrava.extensions.discovery import discover_manifests
    from engrava import SqliteEngravaCore

    store = SqliteEngravaCore(db, manifests=discover_manifests())
    await store.ensure_schema()

Automatic discovery is intentionally not performed by the store itself —
schema migrations have side-effects (ALTER TABLE, CREATE TABLE) and should
only run when the caller opts in.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engrava.domain.manifest import ExtensionManifest

logger = logging.getLogger(__name__)

_DEFAULT_GROUP = "engrava.extensions"


def discover_manifests(group: str = _DEFAULT_GROUP) -> list[ExtensionManifest]:
    """Return ``ExtensionManifest`` objects from installed packages.

    Scans *group* for entry points whose loaded value is either an
    ``ExtensionManifest`` instance or a no-argument callable that returns one.
    Entry points that fail to load are skipped with a ``WARNING`` log.

    This function is **opt-in**: the caller decides whether to enable
    discovery and passes the result explicitly to the store constructor.

    Args:
        group: Entry-point group name to scan.
            Defaults to ``"engrava.extensions"``.

    Returns:
        Ordered list of discovered ``ExtensionManifest`` instances.

    Examples:
        >>> from engrava.extensions.discovery import discover_manifests
        >>> manifests = discover_manifests()  # doctest: +SKIP
        >>> store = SqliteEngravaCore(db, manifests=manifests)  # doctest: +SKIP

    """
    from engrava.domain.manifest import ExtensionManifest  # noqa: PLC0415

    result: list[ExtensionManifest] = []
    for ep in entry_points(group=group):
        try:
            obj = ep.load()
            if callable(obj) and not isinstance(obj, ExtensionManifest):
                obj = obj()
            if isinstance(obj, ExtensionManifest):
                result.append(obj)
            else:
                logger.warning(
                    "Entry point %r in group %r yielded %r instead of an "
                    "ExtensionManifest; skipping.",
                    ep.name,
                    group,
                    type(obj).__name__,
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load extension manifest %r from group %r.",
                ep.name,
                group,
                exc_info=True,
            )
    return result

"""ExtensionManifest — metadata for an engrava extension package.

Used by the CLI and runtime to discover and register extension hooks,
MindQL commands, and schema migrations provided by extension packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from engrava.domain.protocols.hooks import EngravaHooksProtocol, MindQLExtension


@dataclass(frozen=True)
class ExtensionManifest:
    """Metadata for an engrava extension package.

    Attributes:
        name: Extension package name (e.g. ``"my-engrava-plugin"``).
        version: Semantic version of the extension.
        hooks_class: The class implementing ``EngravaHooksProtocol``.
        mindql_extensions: MindQL extension commands provided.
        schema_migrations: Paths to SQL migration scripts (relative or absolute).
            Relative paths are resolved via ``importlib.resources`` against
            the top-level package of ``hooks_class`` unless ``package_root``
            is set.
        package_root: Optional base directory for resolving relative
            ``schema_migrations`` paths.  Useful for test fixtures and
            non-installable manifests where ``importlib.resources`` lookup
            would fail.

    """

    name: str
    version: str
    hooks_class: type[EngravaHooksProtocol]
    mindql_extensions: list[MindQLExtension] = field(default_factory=list)
    schema_migrations: list[Path] = field(default_factory=list)
    package_root: Path | None = None

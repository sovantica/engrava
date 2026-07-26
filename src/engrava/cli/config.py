"""CLI configuration for engrava.

Resolves the database path from CLI args / environment / defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

from engrava.config_validation import forbid_subclassing, own_config_fields, own_str

_DEFAULT_DB_PATH = "engrava.db"
"""Default database file when no override is provided."""


@final
@forbid_subclassing
@dataclass(frozen=True)
class EngravaCLIConfig:
    """CLI configuration resolved from env / args.

    Attributes:
        db_path: Path to the SQLite database file.
        output_format: Output format for query results.
        verbose: Enable verbose logging.
        config_path: Optional path to ``engrava.yaml`` for services config.
        extensions_enabled: Whether installed CLI and MindQL entry points may load.

    """

    db_path: Path
    output_format: Literal["json", "table", "csv"] = "table"
    verbose: bool = False
    config_path: Path | None = None
    extensions_enabled: bool = True

    def __post_init__(self) -> None:
        """Own every field, so the config retains what it was constructed with.

        ``output_format`` selects a renderer by membership and by equality, and
        both consult the value's own methods; the flags gate whether extensions
        load. This class is a public entry point reached straight from the
        command line, so it owns its values here rather than assuming the
        caller passed built-ins.
        """
        own_config_fields(self)

    @classmethod
    def resolve(
        cls,
        *,
        db_path: str | None = None,
        output_format: str = "table",
        verbose: bool = False,
        config_path: str | None = None,
        disable_extensions: bool = False,
    ) -> EngravaCLIConfig:
        """Resolve config from explicit args, then env, then defaults.

        Priority: CLI arg > env var > default.

        - ``db_path``: ``--db`` > ``ENGRAVA_DB`` > ``./engrava.db``
        - ``config_path``: ``--config`` > ``ENGRAVA_CONFIG`` > ``None``

        Args:
            db_path: Explicit database path from CLI.
            output_format: Output format (json/table/csv).
            verbose: Enable verbose output.
            config_path: Explicit path to engrava.yaml.
            disable_extensions: Prevent installed CLI and MindQL entry-point loading.

        Returns:
            Resolved CLI configuration.

        """
        # Own each string *before* it is tested, chosen between, or turned into
        # a path. Every step here consults a method the value may define: the
        # ``or`` chain runs ``__bool__``, so a subclass could suppress a path
        # the user supplied explicitly and hand the environment's or the
        # default's instead; the membership test runs ``__hash__`` / ``__eq__``;
        # and a path is built by joining text a subclass could render
        # differently from the text that was checked.
        owned_db = own_str(db_path) if isinstance(db_path, str) else None
        resolved_path = owned_db or os.environ.get("ENGRAVA_DB") or _DEFAULT_DB_PATH
        owned_format = own_str(output_format) if isinstance(output_format, str) else ""
        fmt = owned_format if owned_format in {"json", "table", "csv"} else "table"
        owned_config = own_str(config_path) if isinstance(config_path, str) else None
        resolved_config = owned_config or os.environ.get("ENGRAVA_CONFIG") or None
        return cls(
            db_path=Path(own_str(resolved_path)),
            output_format=fmt,  # type: ignore[arg-type]  # narrowed to the Literal by the membership test above
            verbose=verbose,
            config_path=Path(resolved_config) if resolved_config else None,
            extensions_enabled=not disable_extensions,
        )

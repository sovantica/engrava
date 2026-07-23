"""CLI configuration for engrava.

Resolves the database path from CLI args / environment / defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_DEFAULT_DB_PATH = "engrava.db"
"""Default database file when no override is provided."""


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
        resolved_path = db_path or os.environ.get("ENGRAVA_DB") or _DEFAULT_DB_PATH
        fmt = output_format if output_format in {"json", "table", "csv"} else "table"
        resolved_config = config_path or os.environ.get("ENGRAVA_CONFIG") or None
        return cls(
            db_path=Path(resolved_path),
            output_format=fmt,  # type: ignore[arg-type]
            verbose=verbose,
            config_path=Path(resolved_config) if resolved_config else None,
            extensions_enabled=not disable_extensions,
        )

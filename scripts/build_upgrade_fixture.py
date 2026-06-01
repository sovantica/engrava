"""Build a representative Engrava fixture database via an isolated install."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.upgrade.fixtures import create_venv, install_package, populate_fixture_db


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-spec", required=True, help="Package spec used to build the fixture.")
    parser.add_argument("--out", required=True, help="Output SQLite database path.")
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Install the package spec in editable mode.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="engrava-upgrade-fixture-") as temp_dir:
        venv_dir = Path(temp_dir) / "venv"
        python_executable = create_venv(venv_dir)
        install_package(
            python_executable,
            args.from_spec,
            editable=args.editable,
            cwd=REPOSITORY_ROOT,
        )
        populate_fixture_db(python_executable, output_path)

    print(f"Upgrade fixture created: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Verify that critical package-data files are bundled in the wheel and sdist.

The packaging pipeline reads ``schema_core.sql`` and ``synthetic-v1.json``
from inside the installed distribution at runtime via
``importlib.resources``. If either file is missing from the built wheel
or sdist, fresh installs of ``engrava`` break with ``FileNotFoundError``
on the first ``ensure_schema()`` call (and the synthetic benchmark fails
to load its frozen dataset).

This script builds both artifacts via ``python -m build`` and then asserts
that the required data files appear in both of them **and** that
development-only files (e.g. the dependency lockfile) do not. It exits
``0`` when everything is bundled correctly and ``1`` otherwise.

Usage:
    python scripts/verify_wheel_data.py

The script is standalone (no test harness, no Makefile dependency) so it
can be run on its own before tagging a release.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"

# Files that MUST appear inside every published wheel and sdist.
REQUIRED_DATA_FILES = (
    "engrava/infrastructure/sqlite/schema_core.sql",
    "engrava/benchmarks/synthetic/datasets/synthetic-v1.json",
)

# Repo-root development files that MUST NOT ship inside the published wheel
# or sdist. The lockfile pins exact dependency versions for a reproducible
# developer / CI environment; consumers installing via ``pip install engrava``
# resolve their own versions from the ``pyproject.toml`` constraints, so the
# lockfile would be dead weight (and a misleading signal) inside the package.
# A MANIFEST.in change that started shipping it would be a silent regression;
# this guard fails the build instead.
FORBIDDEN_PACKAGE_FILES = ("uv.lock",)


def _build_artifacts() -> tuple[Path, Path]:
    """Run ``python -m build`` and return the wheel + sdist paths.

    Removes any previous ``dist/`` and ``build/`` so the assertion below
    only inspects freshly produced artifacts.
    """
    for stale in (REPO_ROOT / "dist", REPO_ROOT / "build"):
        if stale.exists():
            _rmtree(stale)
    subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=REPO_ROOT,
        check=True,
    )
    wheels = sorted(DIST_DIR.glob("engrava-*.whl"))
    sdists = sorted(DIST_DIR.glob("engrava-*.tar.gz"))
    if not wheels:
        msg = "no wheel produced by python -m build"
        raise RuntimeError(msg)
    if not sdists:
        msg = "no sdist produced by python -m build"
        raise RuntimeError(msg)
    return wheels[-1], sdists[-1]


def _wheel_contents(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as z:
        return z.namelist()


def _sdist_contents(sdist: Path) -> list[str]:
    with tarfile.open(sdist) as t:
        return t.getnames()


def _missing(names: list[str], required: tuple[str, ...]) -> list[str]:
    """Return the subset of ``required`` whose path tail is not in ``names``."""
    return [r for r in required if not any(n.endswith(r) for n in names)]


def _present(names: list[str], forbidden: tuple[str, ...]) -> list[str]:
    """Return the subset of ``forbidden`` whose basename appears in ``names``.

    A file is considered present if any archive entry's path tail equals
    the forbidden name (matching the basename, e.g. ``uv.lock`` regardless
    of the leading ``engrava-0.3.0/`` sdist prefix).
    """
    return [f for f in forbidden if any(n.rsplit("/", 1)[-1] == f for n in names)]


def _rmtree(path: Path) -> None:
    """Remove ``path`` recursively without depending on shutil semantics."""
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()


def main() -> int:
    """Build artifacts and return ``0`` iff every required file is bundled."""
    try:
        wheel, sdist = _build_artifacts()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"verify_wheel_data: build failed: {exc}\n")
        return 1

    wheel_names = _wheel_contents(wheel)
    sdist_names = _sdist_contents(sdist)

    sys.stdout.write(f"verify_wheel_data: inspecting {wheel.name}\n")
    sys.stdout.write(f"verify_wheel_data: inspecting {sdist.name}\n")

    # (description, offending entries) for each of the four checks.
    violations = (
        ("required files missing from wheel", _missing(wheel_names, REQUIRED_DATA_FILES)),
        ("required files missing from sdist", _missing(sdist_names, REQUIRED_DATA_FILES)),
        (
            "development-only files leaked into wheel",
            _present(wheel_names, FORBIDDEN_PACKAGE_FILES),
        ),
        (
            "development-only files leaked into sdist",
            _present(sdist_names, FORBIDDEN_PACKAGE_FILES),
        ),
    )

    failed = False
    for description, entries in violations:
        if entries:
            failed = True
            sys.stderr.write(f"verify_wheel_data: ERROR — {description}:\n")
            for entry in entries:
                sys.stderr.write(f"  - {entry}\n")

    if failed:
        return 1

    sys.stdout.write(
        f"verify_wheel_data: OK — {len(REQUIRED_DATA_FILES)} required file(s) present "
        f"and {len(FORBIDDEN_PACKAGE_FILES)} development-only file(s) absent "
        f"in both wheel and sdist.\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

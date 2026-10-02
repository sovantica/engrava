#!/usr/bin/env python3
"""Assert the built wheel and sdist embed the version that is about to be tagged.

Run from scripts/verify_release_artifacts.sh, after scripts/verify_wheel_data.py
has produced dist/*.whl and dist/*.tar.gz, with the version semantic-release
computed (``${nextRelease.version}``) as the sole argument.

Comparing against the *filename* would not catch a pyproject.toml bump that
silently failed to write to disk: ``python -m build`` derives both the
sdist/wheel filenames and their internal metadata from whatever
pyproject.toml says at build time, so a stale pyproject.toml produces a
wheel/sdist pair that is internally consistent with itself and just happens
to be the wrong version throughout — the filename would not disagree with
the metadata, because both come from the same (stale) source. This instead
reads the version actually embedded in each artifact — the wheel's
``*.dist-info/METADATA`` and the sdist's ``PKG-INFO`` — and compares it
against the version bump_pyproject_version.py was told to write, so a
pyproject.toml that silently kept its old version is caught here before the
tag is created, not after the wrong bytes have shipped.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"
VERSION_RE = re.compile(r"(?m)^Version:\s*(\S+)\s*$")
# Argv is [script_name, expected_version]: exactly one positional argument.
_EXPECTED_ARGC = 2


def _wheel_metadata_version(wheel: Path) -> str | None:
    with zipfile.ZipFile(wheel) as z:
        metadata_names = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
        if not metadata_names:
            return None
        text = z.read(metadata_names[0]).decode("utf-8", errors="replace")
    match = VERSION_RE.search(text)
    return match.group(1) if match else None


def _sdist_metadata_version(sdist: Path) -> str | None:
    with tarfile.open(sdist) as t:
        pkginfo_names = [n for n in t.getnames() if n == "PKG-INFO" or n.endswith("/PKG-INFO")]
        if not pkginfo_names:
            return None
        member = t.getmember(pkginfo_names[0])
        extracted = t.extractfile(member)
        if extracted is None:
            return None
        text = extracted.read().decode("utf-8", errors="replace")
    match = VERSION_RE.search(text)
    return match.group(1) if match else None


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) != _EXPECTED_ARGC:
        sys.stderr.write("usage: verify_artifact_version.py <expected-version>\n")
        return 2
    expected = sys.argv[1]

    wheels = sorted(DIST_DIR.glob("engrava-*.whl"))
    sdists = sorted(DIST_DIR.glob("engrava-*.tar.gz"))
    if not wheels:
        sys.stderr.write("verify_artifact_version: no wheel found in dist/\n")
        return 1
    if not sdists:
        sys.stderr.write("verify_artifact_version: no sdist found in dist/\n")
        return 1

    wheel, sdist = wheels[-1], sdists[-1]
    checks = (
        ("wheel", wheel, _wheel_metadata_version(wheel)),
        ("sdist", sdist, _sdist_metadata_version(sdist)),
    )

    failed = False
    for label, path, actual in checks:
        if actual is None:
            sys.stderr.write(
                f"verify_artifact_version: could not read a Version from {path.name}\n",
            )
            failed = True
        elif actual != expected:
            sys.stderr.write(
                f"verify_artifact_version: {label} {path.name} embeds version "
                f"{actual!r}, but {expected!r} is the version being tagged\n",
            )
            failed = True

    if failed:
        return 1

    sys.stdout.write(
        f"verify_artifact_version: OK — wheel and sdist both embed version {expected}\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

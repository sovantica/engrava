#!/usr/bin/env python3
"""Assert dist/ holds exactly the two release artifacts, and nothing else at all.

Run from scripts/verify_release_artifacts.sh immediately after the build,
before anything downstream picks "the" wheel or "the" sdist out of dist/.

scripts/verify_wheel_data.py and scripts/verify_artifact_version.py both
select a single wheel and a single sdist out of dist/ via a sorted glob's
last element. That selection only says something about what ships if dist/
holds exactly one of each — and today it does, but only because
verify_wheel_data.py happens to wipe dist/ before building. That is a
property of a *different* script, not something either verification script
enforces itself. Everything that actually reaches PyPI — `twine check
dist/*`, the sha256 manifest, `upload-artifact` with `path: dist/`, and
pypa's own publish action — operates on the whole directory, not on the
pair the version gate looked at. Without an explicit count here, an extra
or unexpected entry in dist/ would mean the gate verified one pair while an
unbounded number of things ship.

Classification checks ``is_symlink()`` first, before anything else, and
rejects on it unconditionally — a symlink is never the wheel or the sdist,
regardless of what it points at. This is deliberate, not incidental:
``Path.is_file()`` follows symlinks, so classifying by ``is_file()`` first
would accept a symlink named ``engrava-x.y.z-py3-none-any.whl`` that points
at an ordinary file, which is exactly the entry a symlink-based swap would
use. Only a non-symlink is then considered for wheel/sdist status: it is
the wheel only if it is a regular file whose final suffix is,
case-sensitively, exactly ``.whl``; it is the sdist only if it is a regular
file whose name ends, case-sensitively, exactly in ``.tar.gz``. Every other
entry — any symlink, a directory, a differently-cased extension, or a
regular file matching neither pattern — is "other" and fails the gate;
nothing is silently excluded from the count to make a stricter check pass.
This deliberately rejects a ``.zip`` sdist even though Twine itself accepts
that format: engrava's build backend (setuptools) only ever produces
``.tar.gz``, and a build that started producing something else is exactly
the kind of drift this gate exists to catch, not a format to wave through
because Twine happens to allow it.

Exits non-zero, naming every entry it found and what kind of thing each
one is, unless dist/ contains exactly one wheel, exactly one sdist, and no
third entry of any kind.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


def _describe(path: Path) -> str:
    """Name what kind of filesystem entry ``path`` is, for the failure log."""
    if path.is_symlink():
        if not path.exists():
            return "broken symlink"
        return "symlink -> directory" if path.is_dir() else "symlink -> file"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "unknown entry type"


def _classify(path: Path) -> str:
    """Return "wheel", "sdist", or "other" — checking is_symlink() first.

    A symlink must never fall through to the ``is_file()`` branch: that
    call follows the link and would classify a symlink to a regular file
    named like a wheel or sdist as one, which is exactly the entry this
    gate exists to catch.
    """
    if path.is_symlink():
        return "other"
    if path.is_file() and path.suffix == ".whl":
        return "wheel"
    if path.is_file() and path.name.endswith(".tar.gz"):
        return "sdist"
    return "other"


def main() -> int:
    """CLI entry point."""
    if not DIST_DIR.is_dir():
        sys.stderr.write(f"verify_dist_cardinality: {DIST_DIR} does not exist\n")
        return 1

    entries = sorted(DIST_DIR.iterdir())
    classified = [(p, _classify(p)) for p in entries]
    wheels = [p for p, kind in classified if kind == "wheel"]
    sdists = [p for p, kind in classified if kind == "sdist"]
    other = [p for p, kind in classified if kind == "other"]

    sys.stdout.write(
        f"verify_dist_cardinality: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
        f"in {DIST_DIR}: {len(wheels)} wheel(s), {len(sdists)} sdist(s), "
        f"{len(other)} other\n",
    )
    for path in entries:
        sys.stdout.write(f"  - {path.name} ({_describe(path)})\n")

    problems = []
    if len(wheels) != 1:
        problems.append(
            f"expected exactly 1 wheel, found {len(wheels)}: {[p.name for p in wheels]}",
        )
    if len(sdists) != 1:
        problems.append(
            f"expected exactly 1 sdist, found {len(sdists)}: {[p.name for p in sdists]}",
        )
    if other:
        problems.append(
            "unexpected entr"
            + ("y" if len(other) == 1 else "ies")
            + " in dist/: "
            + ", ".join(f"{p.name} ({_describe(p)})" for p in other),
        )

    if problems:
        sys.stderr.write("verify_dist_cardinality: FAILED\n")
        for problem in problems:
            sys.stderr.write(f"  - {problem}\n")
        return 1

    sys.stdout.write(
        "verify_dist_cardinality: OK — exactly one wheel and one sdist, nothing else\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Write the version semantic-release is about to tag into pyproject.toml.

Invoked by .releaserc.json's first `@semantic-release/exec` prepareCmd, with
the version semantic-release computed (``${nextRelease.version}``) as the
sole argument.

Counts matches of the version-line pattern *before* touching the file, with
no cap on the count, and asserts there is exactly one before performing any
replacement. A count capped at 1 (as a plain ``re.subn(..., count=1)`` would
give) can only ever come back 0 or 1, so it cannot see a second match — a
stray ``version = "..."`` line under a later ``[tool.*]`` table would be
invisible to it, and if such a line preceded ``[project]`` it would be the
one silently rewritten instead. Counting unbounded first, then replacing
only the single confirmed match, closes both cases.

This also still refuses a silent no-op: if the version line's formatting
ever changes (single quotes, different whitespace) so the pattern matches
zero times, the OLD version would otherwise stay on disk to be built and
gated while semantic-release goes on to tag the NEW version — the artifact
would only be *named* after the release, not actually be it. A match count
other than exactly 1, in either direction, fails the release instead of
shipping it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(r'(?m)^version\s*=\s*"[^"]+"')


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: bump_pyproject_version.py <version>\n")
        return 2

    version = sys.argv[1]
    path = Path("pyproject.toml")
    text = path.read_text()
    match_count = len(PATTERN.findall(text))

    if match_count != 1:
        sys.stderr.write(
            f"bump_pyproject_version: pattern {PATTERN.pattern!r} matched "
            f"{match_count} time(s) in {path} (expected exactly 1) — refusing "
            "to tag a version that was not unambiguously written to disk.\n",
        )
        return 1

    new_text = PATTERN.sub(f'version = "{version}"', text, count=1)
    path.write_text(new_text)
    sys.stdout.write(f"bump_pyproject_version: wrote version {version} to {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

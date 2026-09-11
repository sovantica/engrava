"""Release gate: refuse a release whose computed version disagrees with the declared target.

Nothing in this pipeline used to name the version it intends to publish.
``.releaserc.json``'s commit-analyzer *derives* a version from whatever the
commit range on ``dev`` happens to type as (``feat`` -> minor, ``fix``/``perf``
-> patch, everything else -> no release, this being a ``0.x`` repository).
The release branch name (``release/v0.7.0``), the release record
(``docs/upgrade.md``'s in-progress compatibility notes), and the GitHub
milestone all name a version too, but none of them was ever compared to what
semantic-release actually computes -- reword or drop the one commit a bump
rests on, and the same branch, with the same content, silently publishes a
different version.

``release-target.json`` at the repository root is the fix: a single
git-tracked file naming the version this release train intends to publish,
independent of the branch name that carried the work to ``dev``. The branch
name itself cannot serve this role -- the release job runs on a push to
``dev``, and by the time it runs, the source ``release/vX.Y.Z`` branch has
already been merged and (per ``BRANCHING.md``) deleted; there is no branch
name left to read.

This script is the F1 half of closing that gap: the wrong-version failure
mode, where semantic-release computes a release that disagrees with what
this release train intends. Invoked as a ``@semantic-release/exec``
``prepareCmd`` in ``.releaserc.json``, positioned as the *first* plugin with
a ``prepare`` hook -- ahead of ``@semantic-release/changelog``,
``bump_pyproject_version.py``, ``verify_release_artifacts.sh``, and
``@semantic-release/git`` -- so a mismatch is caught before any of those run,
not just before the tag. semantic-release aborts the whole ``prepare``
lifecycle step on the first plugin that exits non-zero and only tags, pushes,
and publishes a GitHub Release after every ``prepare`` plugin has succeeded
(the exact reasoning ``.github/workflows/release.yml``'s header documents for
``verify_release_artifacts.sh``, reused here for the same placement).

This script does not, and cannot, catch the sibling failure mode: no version
computed at all, when every commit in the range types as no-release and
semantic-release's ``prepare`` step never runs because there is no release to
prepare. ``scripts/check_release_target_was_published.py`` covers that one,
from outside the plugin lifecycle entirely -- see that script's docstring.

What a PASS from this script establishes, and only this: the version
semantic-release is about to tag is textually equal, component by
component, to the version declared in ``release-target.json`` at this
commit. It does not prove ``release-target.json`` declares the *right*
version -- that the declared target reflects what the release branch was
actually meant to ship, that the person or process which last edited
``release-target.json`` got it right, or that the commit history computing
the matching version is itself correctly typed. A wrong declared target
that happens to agree with a wrongly-computed version passes this gate
without complaint.

Usage::

    python scripts/check_computed_version_matches_target.py 0.7.0

``<computed-version>`` is semantic-release's own ``${nextRelease.version}``,
substituted by semantic-release before this command runs -- a bare
``MAJOR.MINOR.PATCH`` string, no leading ``v``.

Exit codes:
* ``0`` -- the computed version equals the declared target.
* ``1`` -- they disagree, or the target file could not be read or parsed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_TARGET_PATH = REPO_ROOT / "release-target.json"

# Spelled out explicitly rather than '\d+': '\d' is Unicode-aware and matches
# far more than the ten ASCII digits (e.g. U+0662 ARABIC-INDIC DIGIT TWO),
# and int() accepts what it matches. The leading-zero exclusion additionally
# stops a malformed spelling ("01") from parsing to the same value as its
# canonical form ("1") and silently standing in for it in the comparison
# below. See scripts/check_main_carries_the_released_tag.py's TAG_RE, which
# this mirrors -- that module's docstring has the fuller history of both
# pitfalls being found by execution, not by inspection.
_NUMERIC_COMPONENT = r"(?:0|[1-9][0-9]*)"
VERSION_RE = re.compile(
    rf"^({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})$"
)

EXIT_OK = 0
EXIT_FAIL = 1


class GateInputError(RuntimeError):
    """Raised when the declared target or the computed version cannot be resolved."""


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a bare ``MAJOR.MINOR.PATCH`` string (no leading ``v``, no surrounding whitespace).

    No stripping is applied before matching: a value that differs from a
    canonical version only by leading or trailing whitespace is not that
    version and must not be silently treated as though it were.
    """
    match = VERSION_RE.match(version)
    if match is None:
        msg = f"{version!r} is not a bare MAJOR.MINOR.PATCH version"
        raise GateInputError(msg)
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def read_declared_target(path: Path | None = None) -> str:
    """Read and validate the declared target version from ``release-target.json``.

    ``path`` defaults to the module-level :data:`RELEASE_TARGET_PATH`, read
    at call time rather than bound as the parameter's default value -- a
    default value is evaluated once, at function-definition time, which
    would make it immune to a test monkeypatching :data:`RELEASE_TARGET_PATH`
    on the module after import.

    Returns the version exactly as written in the file (a string), after
    confirming it parses as a bare ``MAJOR.MINOR.PATCH`` value. Raises
    :class:`GateInputError` for every way the declaration itself can be
    broken: the file missing, unreadable, not valid JSON, not a JSON object,
    missing the ``version`` key, that key not being a string, or that string
    not being a well-formed version.
    """
    if path is None:
        path = RELEASE_TARGET_PATH
    try:
        text = path.read_text()
    except OSError as exc:
        msg = f"could not read {path}: {exc}"
        raise GateInputError(msg) from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"{path} is not valid JSON: {exc}"
        raise GateInputError(msg) from exc

    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object, got {type(data).__name__}"
        raise GateInputError(msg)

    if "version" not in data:
        msg = f"{path} does not declare a 'version' key"
        raise GateInputError(msg)

    version = data["version"]
    if not isinstance(version, str):
        msg = f"{path}'s 'version' value must be a string, got {type(version).__name__}"
        raise GateInputError(msg)

    parse_version(version)  # validated for its shape; raises GateInputError if malformed
    return version


def run_gate(*, computed_version: str, declared_target: str) -> tuple[bool, list[str]]:
    """Compare the computed release version against the declared target and report."""
    computed_parsed = parse_version(computed_version)
    declared_parsed = parse_version(declared_target)

    messages = [
        f"declared target (release-target.json): {declared_target}",
        f"computed version (nextRelease.version): {computed_version}",
    ]

    if computed_parsed == declared_parsed:
        messages.append(
            f"PASS: the computed release version matches the declared target ({declared_target}).",
        )
        return True, messages

    messages.append(
        f"FAIL: semantic-release computed {computed_version}, but release-target.json "
        f"declares {declared_target}. Publish blocked -- either the commit history on "
        "this branch does not compute the version this release train intends (reword "
        "or drop the commit(s) responsible), or release-target.json itself needs "
        "updating to match what this branch actually ships. This gate cannot tell "
        "you which one is wrong, only that they disagree.",
    )
    return False, messages


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the release-target gate."""
    parser = argparse.ArgumentParser(
        prog="check_computed_version_matches_target.py",
        description=(
            "Release gate. Fails when semantic-release's computed version "
            "disagrees with the version declared in release-target.json."
        ),
    )
    parser.add_argument(
        "computed_version",
        help="the version semantic-release computed, e.g. 0.7.0 (no leading 'v')",
    )
    args = parser.parse_args(argv)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Release-target gate\n")
    sys.stdout.write("=" * 60 + "\n")

    try:
        declared_target = read_declared_target()
        passed, messages = run_gate(
            computed_version=args.computed_version,
            declared_target=declared_target,
        )
    except GateInputError as exc:
        sys.stderr.write(f"release target gate: {exc}\n")
        return EXIT_FAIL

    for line in messages:
        sys.stdout.write(line + "\n")
    sys.stdout.write("=" * 60 + "\n")
    verdict = "PASS" if passed else "FAIL -- publish blocked"
    sys.stdout.write(f"RELEASE TARGET GATE: {verdict}\n")
    sys.stdout.write("=" * 60 + "\n")

    return EXIT_OK if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())

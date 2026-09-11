"""Check that a declared release target was actually published, not silently skipped.

``release-target.json`` names the version this release train intends to
publish (see ``scripts/check_computed_version_matches_target.py`` for the
full motivation). That script closes F1 -- semantic-release computing the
*wrong* version -- by running inside semantic-release's ``prepare``
lifecycle step, ahead of ``@semantic-release/git``.

It cannot close the other failure mode, F2: semantic-release computing *no*
version at all. If every commit in the range between the last release and
``HEAD`` types as no-release (``docs``, ``style``, ``refactor``, ``test``,
``build``, ``ci``, ``chore`` -- see ``.releaserc.json``'s
``commit-analyzer`` rules), semantic-release's own ``prepare`` step never
runs, because there is no release to prepare. ``release.yml``'s own
``detect`` step already anticipates exactly this: it writes
"No release-triggering commits -- nothing published" as a ``::notice::``
and lets the job succeed. A gate for this has to sit outside the plugin
lifecycle entirely, because there is no lifecycle step to hook into on the
run that needs catching.

This script is that outside gate. It is invoked as its own step in
``release.yml``, unconditionally -- after semantic-release has run and
after the workflow's own "detect whether a release was published" step,
but with no ``if:`` guard tying it to that step's result, since the whole
point is to fail a run the rest of the job considers a clean no-op.

The check itself does not need to know whether *this* run was the one that
published the target, only whether the target has been published by the
time this step runs: it resolves the declared target's tag
(``refs/tags/v<version>``) and passes if that tag exists at all, in this
run's history or an earlier one. That is deliberate, not an oversight --
once a target has genuinely shipped, every later push to ``dev`` before
someone advances ``release-target.json`` to the next target would otherwise
fail this gate for a version that was never meant to ship again. On this
repository's own branching model (``BRANCHING.md``: ``feature/* ->
release/vX.Y.Z -> dev -> main``, semantic-release firing on the push to
``dev``), every push to ``dev`` is itself a release-branch merge that was
supposed to trigger a real release, so treating "the target's tag already
exists" as sufficient is not a loophole for skipping releases quietly --
it just avoids re-flagging a target this pipeline already met.

If ``release-target.json`` itself does not exist, this script passes with a
notice rather than failing: a repository (or an older commit in this one's
own history) that declares no target at all has made no promise this gate
can check.

What a PASS from this script establishes, and only this: a git tag named
``v<version>``, where ``<version>`` is release-target.json's declared
target, exists in this checkout. It does not prove that tag was created by
a *correct* release (only ``check_computed_version_matches_target.py``'s
F1 check does that, and only for the run that created it), that the tagged
commit's tree matches what stabilization on the release branch intended,
that the corresponding PyPI upload succeeded or even ran (that gap belongs
to ``check_main_carries_the_released_tag.py``'s own boundary note, not this
script), or that ``release-target.json``'s declared version is the version
this release train *should* have shipped -- a target that was wrong from
the start, once met, satisfies this gate exactly as cleanly as a correct
one would.

Usage::

    python scripts/check_release_target_was_published.py

Exit codes:
* ``0`` -- no target is declared, or the declared target's tag exists.
* ``1`` -- a target is declared and its tag does not exist, or the
  declaration itself could not be parsed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_TARGET_FILENAME = "release-target.json"

# See check_computed_version_matches_target.py's VERSION_RE for why the
# numeric component is spelled out rather than using '\d+'.
_NUMERIC_COMPONENT = r"(?:0|[1-9][0-9]*)"
VERSION_RE = re.compile(
    rf"^({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})$"
)

EXIT_OK = 0
EXIT_FAIL = 1


class GateInputError(RuntimeError):
    """Raised when a declared target exists but cannot be parsed, or a git call fails."""


def read_declared_target() -> str | None:
    """Return the declared target version, or ``None`` if no target is declared.

    ``None`` is a legitimate, passing outcome (see the module docstring): a
    repository state that declares no target has made no promise this gate
    can check. Any other problem with a *present* file -- unreadable,
    malformed JSON, wrong shape, an unparsable version -- is a hard failure,
    not treated the same as "no target declared".
    """
    path = REPO_ROOT / RELEASE_TARGET_FILENAME
    if not path.exists():
        return None

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

    match = VERSION_RE.match(version)
    if match is None:
        msg = f"{path}'s 'version' value {version!r} is not a bare MAJOR.MINOR.PATCH version"
        raise GateInputError(msg)

    return version


def tag_exists(version: str) -> bool:
    """Return whether ``refs/tags/v<version>`` exists as an object in this repository.

    The ``^{object}`` suffix forces git to actually resolve an object rather
    than merely accept the ref name as syntactically plausible -- without
    it, a well-formed-but-nonexistent ref does not reliably fail the same
    way across every ref shape. Mirrors
    ``check_main_carries_the_released_tag.py``'s ``_ref_path_exists``.
    """
    completed = subprocess.run(  # noqa: S603 -- trusted internal git invocation
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/v{version}^{{object}}"],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def run_gate() -> tuple[bool, list[str]]:
    """Apply the F2 rule and return ``(passed, report_lines)``."""
    declared_target = read_declared_target()

    if declared_target is None:
        return True, [
            f"no {RELEASE_TARGET_FILENAME} found at this ref -- no target declared, "
            "nothing for this gate to check.",
            "PASS: no declared target to verify.",
        ]

    published = tag_exists(declared_target)
    messages = [f"declared target ({RELEASE_TARGET_FILENAME}): {declared_target}"]

    if published:
        messages.append(
            f"PASS: tag v{declared_target} exists -- the declared target has been published.",
        )
        return True, messages

    messages.append(
        f"FAIL: release-target.json declares {declared_target}, but no "
        f"v{declared_target} tag exists in this repository. Every commit in this "
        "run's range typed as no-release (or the release otherwise failed to "
        "publish), so nothing was tagged, announced, or shipped to PyPI -- and "
        "without this check, the workflow would report that as a clean, green "
        "no-op instead of the gap it actually is.",
    )
    return False, messages


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the F2 check."""
    parser = argparse.ArgumentParser(
        prog="check_release_target_was_published.py",
        description=(
            "Fails when release-target.json declares a version whose tag "
            "does not exist in this repository -- a declared release target "
            "that was silently never published."
        ),
    )
    parser.parse_args(argv)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Release-target-was-published gate\n")
    sys.stdout.write("=" * 60 + "\n")

    try:
        passed, messages = run_gate()
    except GateInputError as exc:
        sys.stderr.write(f"release target was-published gate: {exc}\n")
        return EXIT_FAIL

    for line in messages:
        sys.stdout.write(line + "\n")
    sys.stdout.write("=" * 60 + "\n")
    verdict = "PASS" if passed else "FAIL"
    sys.stdout.write(f"RELEASE TARGET WAS-PUBLISHED GATE: {verdict}\n")
    sys.stdout.write("=" * 60 + "\n")

    return EXIT_OK if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())

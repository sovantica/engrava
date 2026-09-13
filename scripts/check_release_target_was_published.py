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

If ``release-target.json`` itself does not exist, this script fails rather
than passing. A prior revision treated an absent file as a clean pass,
reasoning that a repository -- or an older commit in this one's own
history -- that declares no target at all has made no promise this gate
can check. That reasoning does not hold for this repository: the file and
this script were introduced one commit apart on the same branch (the
commit that added ``release-target.json`` never shipped without this
script, and this script has never shipped without the file already
present), so there is no point in this repository's history where a
checkout legitimately has the script but not the file. The only way to
reach that state on a real run is for something -- an accidental deletion,
a broken checkout step, a misconfigured sparse checkout -- to remove a
file that is supposed to be there, which is exactly the kind of silent
gap this gate exists to catch, not a state it should wave through.

What a PASS from this script establishes, and only this: a tag named
``v<version>``, where ``<version>`` is release-target.json's declared
target, resolves to a commit (not a blob, tree, or non-commit object) that
is reachable from ``HEAD``. It does not prove that tag was created by a
*correct* release (only ``check_computed_version_matches_target.py``'s
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
* ``0`` -- the declared target's tag resolves to a commit reachable from
  ``HEAD``.
* ``1`` -- ``release-target.json`` does not exist or could not be parsed,
  the declared target's tag does not exist or does not resolve to a
  reachable commit, or git itself could not answer.
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


def read_declared_target() -> str:
    """Return the declared target version.

    An absent file is a hard failure, not a passing "no target declared"
    outcome -- see the module docstring for why this repository has no
    legitimate state where the file is missing but this script still runs.
    Every other problem with the file -- unreadable, malformed JSON, wrong
    shape, an unparsable version -- is also a hard failure.
    """
    path = REPO_ROOT / RELEASE_TARGET_FILENAME
    if not path.exists():
        msg = (
            f"{path} does not exist. This gate has no legitimate case where "
            "the file is absent but the gate still runs -- see this script's "
            "module docstring -- so a missing file is treated as a broken "
            "checkout or an accidental deletion, not as 'no target declared'."
        )
        raise GateInputError(msg)

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


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand, raising :class:`GateInputError` if git cannot even be invoked.

    ``subprocess.run`` raises ``OSError`` (typically ``FileNotFoundError``)
    when the executable itself cannot be found or started -- that is a
    problem with this environment, not an answer about the repository, and
    previously escaped this module as a bare traceback instead of the
    clean, fail-closed diagnostic every other input problem here gets.
    """
    try:
        return subprocess.run(  # noqa: S603 -- trusted internal git invocation
            ["git", *args],  # noqa: S607
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        msg = f"could not run 'git {' '.join(args)}': {exc}"
        raise GateInputError(msg) from exc


def resolve_tag_commit(version: str) -> str | None:
    """Return the commit ``refs/tags/v<version>`` resolves to, or ``None`` if it does not name one.

    The ``^{commit}`` peel is deliberate, not ``^{object}``: it requires the
    ref to resolve to a commit specifically, so a tag that exists but names
    a blob, a tree, or an annotated tag pointing at either of those fails
    here -- rather than being accepted merely because *something* exists
    under that ref name, which is all ``^{object}`` would have confirmed.
    ``git rev-parse --verify --quiet`` exits ``1`` both when the ref does
    not exist at all and when it exists but is the wrong type (confirmed by
    execution against a tag pointing at a blob); either way, "not a
    published commit" is the correct answer here, so both are folded into
    the same ``None`` result. Any other exit code means git could not
    answer the question at all (a broken repository, an unreadable object
    database) and is raised as :class:`GateInputError` instead of being
    silently treated as "no such tag".
    """
    ref = f"refs/tags/v{version}^{{commit}}"
    completed = _run_git(["rev-parse", "--verify", "--quiet", ref])
    if completed.returncode == 0:
        return completed.stdout.strip()
    if completed.returncode == 1:
        return None
    msg = (
        f"'git rev-parse --verify --quiet {ref}' failed unexpectedly "
        f"(exit {completed.returncode}): {completed.stderr.strip()}"
    )
    raise GateInputError(msg)


def is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` is reachable from ``descendant``.

    Mirrors ``check_main_carries_the_released_tag.py``'s function of the
    same name: ``git merge-base --is-ancestor`` exits ``0`` when it is,
    ``1`` when it is not, and anything else means git could not resolve one
    of the two refs at all -- raised as :class:`GateInputError` rather than
    folded into "not an ancestor", which would report an unrelated failure
    (a bad ref, a corrupt repository) as if it were a normal, negative
    answer about reachability.
    """
    completed = _run_git(["merge-base", "--is-ancestor", ancestor, descendant])
    if completed.returncode in (0, 1):
        return completed.returncode == 0
    msg = (
        f"'git merge-base --is-ancestor {ancestor} {descendant}' failed "
        f"unexpectedly (exit {completed.returncode}): {completed.stderr.strip()}"
    )
    raise GateInputError(msg)


def target_is_published(version: str) -> bool:
    """Return whether ``refs/tags/v<version>`` names a commit reachable from ``HEAD``.

    Both conditions are required: a tag that exists but names a non-commit
    object, or names a commit that is not an ancestor of ``HEAD`` (an
    unrelated or since-orphaned branch, for instance), has not published
    this target -- it has merely put a tag with the right name somewhere in
    the object database.
    """
    commit = resolve_tag_commit(version)
    if commit is None:
        return False
    return is_ancestor(commit, "HEAD")


def run_gate() -> tuple[bool, list[str]]:
    """Apply the F2 rule and return ``(passed, report_lines)``."""
    declared_target = read_declared_target()

    published = target_is_published(declared_target)
    messages = [f"declared target ({RELEASE_TARGET_FILENAME}): {declared_target}"]

    if published:
        messages.append(
            f"PASS: tag v{declared_target} resolves to a commit reachable from HEAD "
            "-- the declared target has been published.",
        )
        return True, messages

    messages.append(
        f"FAIL: release-target.json declares {declared_target}, but no "
        f"v{declared_target} tag resolving to a commit reachable from HEAD exists "
        "in this repository. Either nothing was tagged, announced, or shipped to "
        "PyPI for this target (every commit in this run's range typed as "
        "no-release, or the release otherwise failed to publish), or a tag named "
        f"v{declared_target} exists but does not name a published commit for this "
        "branch (it points at a non-commit object, or at a commit this branch "
        "does not contain) -- and without this check, the workflow would report "
        "that as a clean, green no-op instead of the gap it actually is.",
    )
    return False, messages


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the F2 check."""
    parser = argparse.ArgumentParser(
        prog="check_release_target_was_published.py",
        description=(
            "Fails when release-target.json is missing, or declares a "
            "version whose tag does not resolve to a commit reachable from "
            "HEAD in this repository -- a declared release target that was "
            "silently never published."
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

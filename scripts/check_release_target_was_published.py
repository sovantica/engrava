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
(``refs/tags/v<version>``) and passes if that tag resolves to a commit git
reports as reachable from ``HEAD``, whether that happened in this run or an
earlier one. Tag existence alone is not enough -- a tag naming a blob, a
tree, or a commit this branch does not contain all fail this check the same
as no tag at all (see :func:`classify_target`). Treating an *earlier* run's
tag as sufficient is deliberate, not an oversight, though -- once a target
has genuinely shipped, every later push to ``dev`` before someone advances
``release-target.json`` to the next target would otherwise fail this gate
for a version that was never meant to ship again. On this repository's own
branching model (``BRANCHING.md``: ``feature/* -> release/vX.Y.Z -> dev ->
main``, semantic-release firing on the push to ``dev``), every push to
``dev`` is itself a release-branch merge that was supposed to trigger a
real release, so treating "the target's tag already resolves to a reachable
commit" as sufficient is not a loophole for skipping releases quietly -- it
just avoids re-flagging a target this pipeline already met.

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
target, resolves to a commit that git reports as reachable from ``HEAD``,
in this repository, as git sees it. That door -- a repository whose git
metadata has been altered locally -- is left open on purpose: anyone able
to write inside ``.git`` on the machine running this gate can make git
misreport its own history to every check that asks it, including this one.
A PASS proves a tag exists, not that this version reached PyPI.

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
import enum
import json
import re
import stat
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
#
# The pattern ends in '\Z', not '$': without 're.MULTILINE', '$' matches at
# the end of the string *or* just before a trailing newline, so a value
# like "0.7.0\n" satisfies "^...$" at the position before the newline and
# is reported as a match, even though it is not the version it appears to
# be. '\Z' matches only the true end of the string, with no such exception,
# so the same value fails to match regardless of whether the call site uses
# 'match()' or 'fullmatch()' -- confirmed by execution against both. The
# call site below still uses 'fullmatch()' rather than 'match()' as a
# second, independent line of defence: 'match()' does not require the
# pattern to consume the whole string at all, so a future edit that widens
# this pattern (e.g. to allow a suffix) could reopen the same class of bug
# even with '\Z' in place, whereas 'fullmatch()' fails closed regardless of
# how the pattern itself is written.
_NUMERIC_COMPONENT = r"(?:0|[1-9][0-9]*)"
VERSION_RE = re.compile(
    rf"^({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})\Z"
)

EXIT_OK = 0
EXIT_FAIL = 1


class GateInputError(RuntimeError):
    """Raised when a declared target exists but cannot be parsed, or a git call fails."""


def _assert_is_a_regular_declaration_file(path: Path) -> None:
    """Raise unless ``path`` is a regular file belonging to this checkout, not a symlink or absent.

    An absent file is a hard failure, not a passing "no target declared"
    outcome -- see the module docstring for why this repository has no
    legitimate state where the file is missing but this script still runs.
    'exists()' follows a symlink to its target and reports False for one
    that is broken; 'is_symlink()' does not follow it at all. The
    combination is deliberate: something is only truly absent here if
    neither check finds it. A symlink -- broken or not -- is caught below
    instead, as a distinct problem from "the file is missing".

    'lstat()', not 'stat()': the latter follows a symlink and reports on
    whatever it points at, which is exactly the fact this check needs to
    see through, not past. Confirmed by execution: pointing
    release-target.json at an unrelated, well-formed release-target.json
    elsewhere on the filesystem -- with a matching old tag already in this
    repository -- made this gate exit 0 before this check existed.
    'release-target.json' is a git-tracked file; a real checkout never
    produces it as a symlink, so requiring a regular file rejects nothing
    legitimate.
    """
    if not path.exists() and not path.is_symlink():
        msg = (
            f"{path} does not exist. This gate has no legitimate case where "
            "the file is absent but the gate still runs -- see this script's "
            "module docstring -- so a missing file is treated as a broken "
            "checkout or an accidental deletion, not as 'no target declared'."
        )
        raise GateInputError(msg)

    try:
        file_stat = path.lstat()
    except OSError as exc:
        msg = f"could not stat {path}: {exc}"
        raise GateInputError(msg) from exc

    if stat.S_ISLNK(file_stat.st_mode):
        msg = (
            f"{path} is a symlink, not a regular file. This gate refuses to "
            "follow it: a symlink can point release-target.json's declared "
            "contents at anything else readable on the filesystem, entirely "
            "outside this repository checkout, and there is no way from "
            "inside this check to tell that apart from a legitimate, "
            "git-tracked declaration. Replace it with a regular file."
        )
        raise GateInputError(msg)

    if not stat.S_ISREG(file_stat.st_mode):
        msg = f"{path} is not a regular file (mode {stat.filemode(file_stat.st_mode)})."
        raise GateInputError(msg)


def read_declared_target() -> str:
    """Return the declared target version.

    This is the one boundary for the whole declaration-reading path -- from
    opening ``path`` through producing a validated version string. Everything
    :func:`_read_declared_target` raises deliberately, as :class:`GateInputError`
    with a diagnostic already specific to what went wrong (absent, a
    symlink, not a regular file, not valid JSON, wrong shape, an unparsable
    version -- see :func:`_assert_is_a_regular_declaration_file` and the
    module docstring for why an absent file in particular is not treated as
    "no target declared"), passes straight through unchanged below --
    wrapping it here would only make it vaguer. Everything else -- any
    exception nobody anticipated -- is caught once, by the trailing
    ``except Exception``, and converted to the same clean, fail-closed
    diagnostic, naming this file and the exception that hit it.

    This replaces what used to be a short, hand-picked list of anticipated
    exception types on this path (``OSError`` and ``MemoryError`` around
    ``read_text()``, ``json.JSONDecodeError`` and a second ``MemoryError``
    around ``json.loads()``): each entry closed one specific, previously
    found gap and left every other kind of unreadable or malformed file to
    escape as a bare traceback -- confirmed by execution against a file
    containing invalid UTF-8 (``UnicodeDecodeError`` out of ``read_text()``)
    and a 10,000-level nested JSON document (``RecursionError`` out of
    ``json.loads()``), neither of which is any of the four types the old
    list named. A single boundary around the whole path has no fifth type to
    miss, because it does not enumerate types at all.
    """
    path = REPO_ROOT / RELEASE_TARGET_FILENAME
    try:
        return _read_declared_target(path)
    except GateInputError:
        raise
    except Exception as exc:  # this *is* the boundary -- see the docstring above.
        msg = f"could not read {path}: {type(exc).__name__}: {exc}"
        raise GateInputError(msg) from exc


def _read_declared_target(path: Path) -> str:
    """Do the actual reading, parsing and validating, with no safety net of its own.

    :func:`read_declared_target` above is the sole caller and supplies the
    one boundary that turns any failure here -- anticipated or not -- into a
    clean :class:`GateInputError`. This function raises whatever the
    underlying call raises, whether that is a deliberate ``GateInputError``
    (the checks below) or something unanticipated (a decode error, a
    recursion limit) -- it makes no attempt to catch or classify the latter
    itself.
    """
    _assert_is_a_regular_declaration_file(path)
    text = path.read_text()
    data = json.loads(text)

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

    # 'fullmatch()', not 'match()': with the previous pattern (ending in
    # '$'), 'match()' let "0.7.0\n" through, because '$' matches just before
    # a trailing newline -- see VERSION_RE's own comment. Changing the
    # pattern to end in '\Z' already closes that specific case even under
    # 'match()', but 'match()' never required the pattern to consume the
    # whole string in the first place, only to match starting at position 0
    # -- a future change to this pattern (e.g. widening it to allow a
    # suffix) could reopen the same class of bug under 'match()' without
    # touching '\Z' at all. 'fullmatch()' fails closed regardless of how the
    # pattern is written, so the two fixes are independent, not redundant.
    match = VERSION_RE.fullmatch(version)
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


def _is_symbolic_ref(ref: str) -> bool:
    """Return whether ``ref`` is itself a symbolic ref rather than a real tag object.

    ``git symbolic-ref --quiet <ref>`` exits ``0`` and prints the ref's
    target when ``ref`` is symbolic, and exits ``1`` both when ``ref`` does
    not exist at all and when it exists but is an ordinary (annotated or
    lightweight) tag -- confirmed by execution against all three cases.
    Both non-zero cases collapse to ``False`` here: this function only
    answers "is this specifically a symbolic ref", and the caller separately
    handles "does not exist" and "names a non-commit object".

    This exists because ``git symbolic-ref refs/tags/v0.7.0
    refs/heads/some-branch`` is accepted by git and creates something that
    answers to the name ``refs/tags/v0.7.0`` without ever creating a tag
    object -- confirmed by execution: with such a symbolic ref in place and
    ``refs/heads/some-branch`` pointing at ``HEAD``, ``git rev-parse
    --verify --quiet 'refs/tags/v0.7.0^{commit}'`` resolved and printed
    ``HEAD``'s own commit, exiting ``0``, so :func:`resolve_tag_commit`
    would previously have reported the target as published with no real tag
    anywhere in the repository. Any other exit code means git could not
    answer at all and is raised as :class:`GateInputError` rather than
    silently treated as "not symbolic".
    """
    completed = _run_git(["symbolic-ref", "--quiet", ref])
    if completed.returncode in (0, 1):
        return completed.returncode == 0
    msg = (
        f"'git symbolic-ref --quiet {ref}' failed unexpectedly "
        f"(exit {completed.returncode}): {completed.stderr.strip()}"
    )
    raise GateInputError(msg)


def _resolves_to_any_object(ref: str) -> bool:
    """Return whether ``ref`` resolves to any object at all (commit, tree, blob, or tag).

    Used only to tell apart, for reporting, a tag that does not exist from
    one that exists but does not name a commit -- :func:`resolve_tag_commit`
    itself deliberately folds both into the same ``None`` result, because
    the pass/fail decision does not need to tell them apart, only the
    message a reader sees does.
    """
    completed = _run_git(["rev-parse", "--verify", "--quiet", f"{ref}^{{object}}"])
    if completed.returncode in (0, 1):
        return completed.returncode == 0
    msg = (
        f"'git rev-parse --verify --quiet {ref}^{{object}}' failed unexpectedly "
        f"(exit {completed.returncode}): {completed.stderr.strip()}"
    )
    raise GateInputError(msg)


def resolve_tag_commit(version: str, *, is_symbolic_tag_ref: bool | None = None) -> str | None:
    """Return the commit ``refs/tags/v<version>`` resolves to, or ``None`` if it does not name one.

    A symbolic ref of that name is rejected before it is ever peeled: it is
    not a tag this gate created or this repository's release tooling would
    ever produce, only something able to imitate one for this one check --
    see :func:`_is_symbolic_ref` for the executed proof. It is treated the
    same as a missing tag, not resolved through to whatever it happens to
    point at.

    ``is_symbolic_tag_ref``, when given, is the caller's own already-known
    answer to "is ``refs/tags/v<version>`` a symbolic ref", and this
    function trusts it instead of asking git again. :func:`classify_target`
    passes its own answer here, because it has to compute it anyway before
    deciding whether to call this function at all -- without this
    parameter, a passing run asked git the identical
    ``git symbolic-ref --quiet refs/tags/v<version>`` question twice,
    confirmed by execution (instrumenting ``_run_git`` on a passing
    ``classify_target()`` call showed two identical invocations before this
    parameter existed, one after). Every other caller -- including this
    module's own direct tests of this function -- omits it, and the
    ``None`` default makes this function compute the answer itself exactly
    as it always has.

    Past that, the ``^{commit}`` peel is deliberate, not ``^{object}``: it
    requires the ref to resolve to a commit specifically, so a tag that
    exists but names a blob, a tree, or an annotated tag pointing at either
    of those fails here -- rather than being accepted merely because
    *something* exists under that ref name, which is all ``^{object}``
    would have confirmed. ``git rev-parse --verify --quiet`` exits ``1``
    both when the ref does not exist at all and when it exists but is the
    wrong type (confirmed by execution against a tag pointing at a blob);
    either way, "not a published commit" is the correct answer here, so
    both are folded into the same ``None`` result. Any other exit code
    means git could not answer the question at all (a broken repository, an
    unreadable object database) and is raised as :class:`GateInputError`
    instead of being silently treated as "no such tag".
    """
    tag_ref = f"refs/tags/v{version}"
    if is_symbolic_tag_ref is None:
        is_symbolic_tag_ref = _is_symbolic_ref(tag_ref)
    if is_symbolic_tag_ref:
        return None
    ref = f"{tag_ref}^{{commit}}"
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


class TagState(enum.Enum):
    """Which of the distinct reasons a declared target's tag can fail to establish a PASS."""

    NO_TAG = "no_tag"
    NON_COMMIT = "non_commit"
    UNREACHABLE = "unreachable"
    PUBLISHED = "published"


def classify_target(version: str) -> tuple[TagState, str | None]:
    """Return which of :class:`TagState` applies to ``version``, and the resolved commit if any.

    This is the single source of truth for both the pass/fail decision and
    the FAIL message a reader sees -- there is deliberately no separate
    boolean function that re-derives "published or not" from its own git
    calls. Two independent lookups of the same fact can only ever agree by
    coincidence; keeping one means there is nothing left to disagree.

    That single-source-of-truth intent is about the pass/fail decision, not
    about how many times git itself gets asked: this function's own
    ``git symbolic-ref --quiet`` probe and :func:`resolve_tag_commit`'s used
    to ask the identical question independently, on every passing
    classification -- not a second opinion able to disagree with the first,
    just the same subprocess call made twice for no reason. This function
    computes the answer once and passes it into :func:`resolve_tag_commit`
    via ``is_symbolic_tag_ref`` instead of letting that function re-derive
    it.

    Both a tag naming a non-commit object and a tag naming a commit outside
    ``HEAD``'s history fail to establish a PASS, distinctly from each other
    and from no tag existing at all -- :attr:`TagState.NO_TAG`,
    :attr:`TagState.NON_COMMIT`, and :attr:`TagState.UNREACHABLE`
    respectively; only :attr:`TagState.PUBLISHED` passes.

    A symbolic ref standing in for the tag (see :func:`resolve_tag_commit`)
    is reported as :attr:`TagState.NO_TAG`: it is not a real tag object by
    any of this repository's release tooling, so "no tag" is the accurate
    description of what actually exists under that name, even though
    ``^{object}`` can still resolve through it to whatever it points at.
    """
    tag_ref = f"refs/tags/v{version}"
    is_symbolic = _is_symbolic_ref(tag_ref)
    if is_symbolic:
        return TagState.NO_TAG, None

    commit = resolve_tag_commit(version, is_symbolic_tag_ref=is_symbolic)
    if commit is None:
        if _resolves_to_any_object(tag_ref):
            return TagState.NON_COMMIT, None
        return TagState.NO_TAG, None

    if is_ancestor(commit, "HEAD"):
        return TagState.PUBLISHED, commit
    return TagState.UNREACHABLE, commit


_WOULD_BE_A_CLEAN_NO_OP = (
    "Without this check, the workflow would report that as a clean, "
    "green no-op instead of the gap it actually is."
)


def run_gate() -> tuple[bool, list[str]]:
    """Apply the F2 rule and return ``(passed, report_lines)``."""
    declared_target = read_declared_target()

    state, commit = classify_target(declared_target)
    messages = [f"declared target ({RELEASE_TARGET_FILENAME}): {declared_target}"]

    if state is TagState.PUBLISHED:
        messages.append(
            f"PASS: tag v{declared_target} resolves to a commit that git reports as "
            "reachable from HEAD in this repository. That is what this check reads "
            "and no more: git's own view of this repository's history, not a defense "
            "against locally altered git metadata, and proof that a tag exists, not "
            "that this version reached PyPI.",
        )
        return True, messages

    if state is TagState.NON_COMMIT:
        messages.append(
            f"FAIL: tag v{declared_target} exists but does not name a commit -- "
            f"refs/tags/v{declared_target} resolves to a non-commit object (a blob, "
            "a tree, or an annotated tag pointing at one), not something this branch "
            f"could contain. {_WOULD_BE_A_CLEAN_NO_OP}",
        )
    elif state is TagState.UNREACHABLE:
        messages.append(
            f"FAIL: tag v{declared_target} resolves to commit {commit}, but that "
            f"commit is not reachable from HEAD in this repository -- it is on an "
            f"unrelated or since-orphaned branch. {_WOULD_BE_A_CLEAN_NO_OP}",
        )
    else:  # state is TagState.NO_TAG -- the only member not handled above or returned early
        messages.append(
            f"FAIL: no tag named v{declared_target} exists in this repository -- "
            f"refs/tags/v{declared_target} does not resolve to any object here. "
            f"{_WOULD_BE_A_CLEAN_NO_OP}",
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

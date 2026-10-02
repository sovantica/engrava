"""Check that the declared release target's tag is reachable from ``HEAD``.

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
but with no ``if:`` guard tying it to that step's result, since the run it is
aimed at is one the rest of the job considers a clean no-op.

The check itself reads only what git says about the declared target: it
resolves the declared target's tag (``refs/tags/v<version>``) and passes if
that tag resolves to a commit git reports as reachable from ``HEAD``. It does
not establish that this run published anything. The tag may have been
created by an earlier run, and once the target's tag is reachable a docs-only
merge that releases nothing still passes. Tag existence alone is not enough
-- a tag naming a blob, a tree, or a commit this branch does not contain all
fail this check the same as no tag at all (see :func:`classify_target`).
Accepting an *earlier* run's tag is deliberate, not an oversight -- once a
target has genuinely shipped, every later push to ``dev`` before someone
advances ``release-target.json`` to the next target would otherwise fail this
gate for a version that was never meant to ship again.

If ``release-target.json`` itself does not exist, this script fails rather
than passing. A checkout that has this script but not the file is not a
legitimate state: this repository tracks both, so the only way to reach it
on a real run is for something -- an accidental deletion, a broken
checkout step, a misconfigured sparse checkout -- to remove a file that is
supposed to be there, which is exactly the kind of silent gap this gate
exists to catch, not a state it should wave through.

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
# the end of the string *or* just before a trailing newline, so under
# 'match()' a value like "0.7.0\n" satisfies "^...$" at the position before
# the newline and is reported as a match, even though it is not the version
# it appears to be. '\Z' matches only the true end of the string, with no
# such exception, so the same value fails to match regardless of whether
# the call site uses 'match()' or 'fullmatch()'. The call site below uses
# 'fullmatch()', which requires the pattern to consume the whole string;
# 'match()' only requires a match starting at position 0.
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
    whatever it points at, so it could not tell that ``path`` is a link.
    Without this check, 'read_text()' would follow the link and the gate
    could pass on declared contents that live outside this checkout.
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

    The trailing ``except Exception`` is deliberately a single boundary
    rather than a list of anticipated exception types, so that an exception
    nobody anticipated cannot escape as a bare traceback. Reading and parsing
    can raise well beyond ``OSError`` and ``json.JSONDecodeError``:
    ``UnicodeDecodeError`` out of ``read_text()`` for a file whose bytes are
    not valid in the locale's text encoding, and ``RecursionError`` out of
    ``json.loads()`` for a deeply nested JSON document.
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

    # 'fullmatch()', not 'match()': 'fullmatch()' requires the pattern to
    # consume the whole string, while 'match()' only requires a match
    # starting at position 0. VERSION_RE also ends in '\Z', which rejects
    # "0.7.0\n" under either -- see VERSION_RE's own comment.
    match = VERSION_RE.fullmatch(version)
    if match is None:
        msg = f"{path}'s 'version' value {version!r} is not a bare MAJOR.MINOR.PATCH version"
        raise GateInputError(msg)

    return version


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand, raising :class:`GateInputError` if git cannot even be invoked.

    ``subprocess.run`` raises ``OSError`` (typically ``FileNotFoundError``)
    when the executable itself cannot be found or started -- that is a
    problem with this environment, not an answer about the repository, so
    it is reported as a clean, fail-closed diagnostic like every other input
    problem here rather than escaping as a bare traceback.
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
    lightweight) tag. Both non-zero cases collapse to ``False`` here: this
    function only answers "is this specifically a symbolic ref", and the
    caller separately handles "does not exist" and "names a non-commit
    object".

    This exists because ``git symbolic-ref refs/tags/v0.7.0
    refs/heads/some-branch`` is accepted by git and creates something that
    answers to the name ``refs/tags/v0.7.0`` without ever creating a tag
    object: with ``refs/heads/some-branch`` pointing at ``HEAD``, ``git
    rev-parse --verify --quiet 'refs/tags/v0.7.0^{commit}'`` resolves and
    prints ``HEAD``'s own commit, exiting ``0``. Without this check,
    :func:`resolve_tag_commit` would accept the target with no real tag
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
    see :func:`_is_symbolic_ref` for how it can pass for a tag. It is
    treated the same as a missing tag, not resolved through to whatever it
    happens to point at.

    ``is_symbolic_tag_ref``, when given, is the caller's own already-known
    answer to "is ``refs/tags/v<version>`` a symbolic ref", and this
    function trusts it instead of asking git again. :func:`classify_target`
    passes its own answer here, because it has to compute it anyway before
    deciding whether to call this function at all, so a classification asks
    git the ``git symbolic-ref --quiet refs/tags/v<version>`` question once.
    Every other caller -- including this module's own direct tests of this
    function -- omits it, and the ``None`` default makes this function
    compute the answer itself.

    Past that, the ``^{commit}`` peel is deliberate, not ``^{object}``: it
    requires the ref to resolve to a commit specifically, so a tag that
    exists but names a blob, a tree, or an annotated tag pointing at either
    of those fails here -- rather than being accepted merely because
    *something* exists under that ref name, which is all ``^{object}``
    would have confirmed. ``git rev-parse --verify --quiet`` exits ``1``
    both when the ref does not exist at all and when it exists but is the
    wrong type (for example a tag pointing at a blob); either way, "not a
    commit" is the correct answer here, so both are folded into the same
    ``None`` result. Any other exit code means git could not answer the
    question at all (a broken repository, an unreadable object database)
    and is raised as :class:`GateInputError` instead of being silently
    treated as "no such tag".
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
    REACHABLE = "reachable"


def classify_target(version: str) -> tuple[TagState, str | None]:
    """Return which of :class:`TagState` applies to ``version``, and the resolved commit if any.

    This is the single source of truth for both the pass/fail decision and
    the FAIL message a reader sees -- there is deliberately no separate
    boolean function that re-derives "reachable or not" from its own git
    calls. Two independent lookups of the same fact can only ever agree by
    coincidence; keeping one means there is nothing left to disagree.

    That single-source-of-truth intent is about the pass/fail decision, not
    about how many times git itself gets asked. This function computes
    whether the tag is a symbolic ref once and passes the answer into
    :func:`resolve_tag_commit` via ``is_symbolic_tag_ref`` instead of
    letting that function re-derive it, so a classification runs
    ``git symbolic-ref --quiet`` once.

    Both a tag naming a non-commit object and a tag naming a commit outside
    ``HEAD``'s history fail to establish a PASS, distinctly from each other
    and from no tag existing at all -- :attr:`TagState.NO_TAG`,
    :attr:`TagState.NON_COMMIT`, and :attr:`TagState.UNREACHABLE`
    respectively; only :attr:`TagState.REACHABLE` passes.

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
        return TagState.REACHABLE, commit
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

    if state is TagState.REACHABLE:
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
            "HEAD in this repository."
        ),
    )
    parser.parse_args(argv)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Release-target tag gate\n")
    sys.stdout.write("=" * 60 + "\n")

    try:
        passed, messages = run_gate()
    except GateInputError as exc:
        sys.stderr.write(f"release target tag gate: {exc}\n")
        return EXIT_FAIL

    for line in messages:
        sys.stdout.write(line + "\n")
    sys.stdout.write("=" * 60 + "\n")
    verdict = "PASS" if passed else "FAIL"
    sys.stdout.write(f"RELEASE TARGET TAG GATE: {verdict}\n")
    sys.stdout.write("=" * 60 + "\n")

    return EXIT_OK if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())

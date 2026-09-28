"""Release gate: refuse a release whose computed version disagrees with the declared target.

Nothing in this pipeline used to name the version it intends to publish.
``.releaserc.json``'s commit-analyzer *derives* a version from whatever the
commit range on ``dev`` happens to type as (``feat`` -> minor, ``fix``/``perf``
-> patch, everything else -> no release, this being a ``0.x`` repository).
The release branch name (``release/v0.7.0``), the release record
(``docs/upgrade.md``'s in-progress compatibility notes), and the GitHub
milestone all name a version too, but none of them was ever compared to what
semantic-release actually computes.

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
prepare. ``scripts/check_release_target_was_published.py`` is the gate that
runs for that one, from outside the plugin lifecycle entirely; what it does
and does not establish is in that script's docstring.

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
import stat
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
#
# The pattern ends in '\Z', not '$': without 're.MULTILINE', '$' matches at
# the end of the string *or* just before a trailing newline, so
# "0.7.0\n" satisfied "^...$" up to the position before the newline and
# was reported as equal to the clean "0.7.0" -- confirmed by execution:
# 'run_gate(computed_version="0.7.0\n", declared_target="0.7.0")' passed
# before this pattern used '\Z'. '\Z' matches only the true end of the
# string, closing that case regardless of which method the call site uses.
#
# Each numeric component is additionally bounded to 18 digits ('[0-9]{0,17}'
# after the leading digit), not left as '[0-9]*'. The bound sits comfortably
# above anything this pipeline can produce: the computed version comes from
# semantic-release, whose semver implementation caps a component at
# JavaScript's MAX_SAFE_INTEGER -- 16 digits -- so nothing it can emit is
# rejected here. (An earlier version of this comment justified the bound by
# saying 18 digits exceeds a 64-bit integer's range. It does not: the largest
# 18-digit number is about 1.0e18 and a signed 64-bit integer reaches roughly
# 9.2e18. The bound is fine; that reason for it was wrong.) It exists to stop a
# component with thousands of digits from ever reaching parse_version()'s
# int() call below, which raises a bare ValueError once a component exceeds
# Python's own int-string conversion ceiling (4300 digits by default,
# CPython's 'sys.get_int_max_str_digits()') -- confirmed by execution: a
# 5,000-digit component matched the previous, unbounded pattern and then
# raised an uncaught ValueError out of int(), rather than the clean
# GateInputError every other malformed version gets here. The bound reduces
# how often that path is taken; it does not replace the boundary in
# read_declared_target() below, which still converts a ValueError from
# int() -- reachable if this limit is ever raised or lowered independently
# of this pattern -- into the same clean diagnostic as everything else.
_NUMERIC_COMPONENT = r"(?:0|[1-9][0-9]{0,17})"
VERSION_RE = re.compile(
    rf"^({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})\Z"
)

EXIT_OK = 0
EXIT_FAIL = 1


class GateInputError(RuntimeError):
    """Raised when the declared target or the computed version cannot be resolved."""


def _assert_is_a_regular_declaration_file(path: Path) -> None:
    """Raise if ``path`` exists but is a symlink or another non-regular file.

    Duplicated from ``check_release_target_was_published.py``'s guard of
    the same name and for the same reason, rather than factored into a
    shared module: the two scripts already duplicate ``VERSION_RE`` and its
    surrounding reasoning outright (see this module's own comment above
    that pattern), and each is invoked standalone, from a different point
    in the release pipeline, by a different caller
    (``@semantic-release/exec``'s ``prepareCmd`` here,
    ``release.yml`` directly there) -- neither imports the other or any
    third module today. Introducing one for two call sites this small would
    add an import edge between two scripts that currently have none, for a
    ten-line guard neither is likely to change independently of the other;
    the existing duplication in this file is the precedent for keeping it
    that way here too.

    Unlike that script's version, a *missing* file is not raised here: this
    function only runs before ``read_text()`` in :func:`read_declared_target`
    below, and that call already turns a missing file into a clean
    ``"could not read {path}: ..."`` :class:`GateInputError` on its own --
    this script's docstring makes no claim, unlike the sibling gate's, that
    there is no legitimate history where the file is briefly absent, so
    there is nothing to add for that case. ``lstat()``, not ``stat()``: the
    latter follows a symlink and reports on whatever it points at, which is
    exactly the fact this check needs to see through, not past -- confirmed
    by execution: pointing ``release-target.json`` at an unrelated,
    well-formed ``release-target.json`` elsewhere on the filesystem made
    this gate exit 0, reporting PASS against ambient JSON this checkout
    never declared, before this check existed.
    """
    try:
        file_stat = path.lstat()
    except OSError:
        # Missing (or otherwise unstattable, e.g. an unreadable parent
        # directory) -- read_text() below raises its own OSError for this,
        # with the message this script has always given for it.
        return

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


def parse_version(version: str) -> tuple[int, int, int]:
    r"""Parse a bare ``MAJOR.MINOR.PATCH`` string (no leading ``v``, no surrounding whitespace).

    No stripping is applied before matching: a value that differs from a
    canonical version only by leading or trailing whitespace is not that
    version and must not be silently treated as though it were.

    ``fullmatch()``, not ``match()``: changing ``VERSION_RE`` to end in
    ``'\\Z'`` already closes the specific trailing-newline case this method
    used to let through (see the pattern's own comment), but ``match()``
    never required the pattern to consume the whole string to begin with,
    only to match starting at position 0 -- a future change widening this
    pattern could reopen the same class of bug under ``match()`` without
    touching the anchor at all. ``fullmatch()`` fails closed regardless of
    how the pattern itself is written.
    """
    match = VERSION_RE.fullmatch(version)
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
    confirming it parses as a bare ``MAJOR.MINOR.PATCH`` value.

    This is the one boundary for the whole declaration-reading path -- from
    opening ``path`` through producing a validated version string. Everything
    :func:`_read_declared_target` raises deliberately, as :class:`GateInputError`
    with a diagnostic already specific to what went wrong (the file missing,
    a symlink, not valid JSON, not a JSON object, missing the ``version``
    key, that key not being a string, that string not being a well-formed
    version), passes straight through unchanged below -- wrapping it here
    would only make it vaguer. Everything else -- any exception nobody
    anticipated -- is caught once, by the trailing ``except Exception``, and
    converted to the same clean, fail-closed diagnostic, naming this file and
    the exception that hit it.

    This replaces what used to be a short, hand-picked list of anticipated
    exception types on this path (``OSError`` and ``MemoryError`` around
    ``read_text()``, ``json.JSONDecodeError`` and a second ``MemoryError``
    around ``json.loads()``): each entry closed one specific, previously
    found gap and left every other kind of unreadable or malformed file to
    escape as a bare traceback -- confirmed by execution against a file
    containing invalid UTF-8 (``UnicodeDecodeError`` out of ``read_text()``),
    a 10,000-level nested JSON document (``RecursionError`` out of
    ``json.loads()``), and a version component of 5,000 digits (``ValueError``
    out of ``int()`` inside :func:`parse_version`) -- none of which is any of
    the four types the old list named. A single boundary around the whole
    path has no fifth type to miss, because it does not enumerate types at
    all.
    """
    if path is None:
        path = RELEASE_TARGET_PATH
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
    (the four checks below) or something unanticipated (a decode error, a
    recursion limit, an integer-conversion limit) -- it makes no attempt to
    catch or classify the latter itself.
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

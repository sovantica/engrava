"""Refuse a release branch that carries a merge commit instead of a squash.

``BRANCHING.md`` (and the private release protocol it is generated to agree
with) states one rule for how work reaches a release branch: every
``<type>/* -> release/vX.Y.Z`` change is squash-merged, one commit per
change. The only merge commit this flow ever expects is
``release/vX.Y.Z -> dev`` itself, and that merge lives *outside* the range
this script inspects -- ``dev`` is the base, so a proper release branch's
own history, from that base to its tip, should contain no merge commits at
all. If someone runs a plain ``git merge`` instead of squashing, the
merge's second parent drags that entire feature branch's raw commits into
the release history intact: non-conventional subjects, scopes absent from
``commit-scopes.json``, and a commit-lint failure per offending commit --
none of it visible from the merge commit's own message, which looks like
routine housekeeping (``Merge branch 'fix/x' into release/v0.7.0``).

This script makes that visible: run before merging anything into a release
branch, and again before merging that branch into ``dev``, it reports
whether ``base..branch`` contains a merge commit.

What a PASS from this script establishes, and only this: no merge commit
(a commit with two or more parents) exists in ``base..branch``. It does not
check squash quality (a squash with a poor commit message still passes),
does not check the ``release/vX.Y.Z -> dev`` merge itself (deliberately
excluded from the range, since it is expected to be one), and does not
prevent a future merge from being added *after* a clean run -- it is a
point-in-time check, not a standing branch protection.

Usage::

    python scripts/check_release_branch_merge_discipline.py
    python scripts/check_release_branch_merge_discipline.py --branch release/v0.7.0

``--branch`` defaults to the current branch (``git rev-parse --abbrev-ref
HEAD``) and ``--base`` defaults to ``origin/dev``, matching how this flow's
release branches are always cut and always merged.

Exit codes:
* ``0`` -- no merge commit exists between ``base`` and ``branch``.
* ``1`` -- at least one does, or ``base``/``branch`` could not be resolved.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

# A control character used to split the hash from the subject on one line.
# The hash is hexadecimal, so the first separator on a line always follows
# it, and str.partition() keeps a subject that contains spaces (or this
# character itself) whole.
_FIELD_SEP = "\x1f"

EXIT_OK = 0
EXIT_FAIL = 1


class GateInputError(RuntimeError):
    """Raised when ``base`` or ``branch`` cannot be resolved by git."""


@dataclass(frozen=True)
class MergeCommit:
    """One merge commit found in the inspected range."""

    sha: str
    subject: str

    @property
    def short_sha(self) -> str:
        """The first 12 characters of :attr:`sha`, git's own default abbreviation length."""
        return self.sha[:12]


def current_branch(*, repo_root: Path) -> str:
    """Return the checked-out branch name, for use as the default ``--branch``.

    Raises :class:`GateInputError` on a detached HEAD or any other state
    where git cannot name a branch -- this gate has nothing sensible to
    default to in that case, and guessing would be worse than asking.
    """
    completed = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    branch = completed.stdout.strip()
    if completed.returncode != 0 or not branch:
        msg = "could not determine the current branch (detached HEAD?); pass --branch explicitly"
        raise GateInputError(msg)
    return branch


def list_merge_commits(base: str, branch: str, *, repo_root: Path = REPO_ROOT) -> list[MergeCommit]:
    """Return every merge commit in ``base..branch``, tip-first.

    A merge commit here means exactly what git means by one: a commit
    recorded with two or more parents, per ``git log --merges``. This does
    not distinguish a legitimate merge from a protocol-violating one --
    every caller in this flow expects zero, so the distinction has no case
    to serve.
    """
    completed = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "log",
            "--merges",
            f"--format=%H{_FIELD_SEP}%s",
            f"{base}..{branch}",
            "--",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        msg = (
            f"git log --merges {base}..{branch} failed (exit {completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
        raise GateInputError(msg)

    commits: list[MergeCommit] = []
    for line in completed.stdout.splitlines():
        if not line:
            continue
        sha, sep, subject = line.partition(_FIELD_SEP)
        if not sep:
            msg = f"unparseable git log line (no field separator found): {line!r}"
            raise GateInputError(msg)
        commits.append(MergeCommit(sha=sha, subject=subject))
    return commits


def run_gate(
    merge_commits: Sequence[MergeCommit], *, base: str, branch: str
) -> tuple[bool, list[str]]:
    """Decide PASS/FAIL from an already-collected list of merge commits and report why."""
    messages = [f"range: {base}..{branch}"]

    if not merge_commits:
        messages.append(
            "PASS: no merge commit found -- every change reached "
            f"{branch!r} as a squash, as the flow requires.",
        )
        return True, messages

    messages.append(
        f"FAIL: {len(merge_commits)} merge commit(s) found in {base}..{branch}. "
        "Every '<type>/* -> release/vX.Y.Z' change must be squash-merged; a merge "
        "commit here drags its entire source branch's raw commits into the release "
        "history, breaking commit-lint and the changelog it feeds. Squash each one "
        "instead (git commit-tree '<merge>^{tree}' -p '<merge>^1' -m \"<message>\", "
        "then rebase --onto):",
    )
    messages.extend(f"  {commit.short_sha}  {commit.subject}" for commit in merge_commits)
    return False, messages


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the release-branch merge-discipline gate."""
    parser = argparse.ArgumentParser(
        prog="check_release_branch_merge_discipline.py",
        description=(
            "Fails when a release branch's history contains a merge commit "
            "instead of a squash between the given base and branch."
        ),
    )
    parser.add_argument(
        "--base",
        default="origin/dev",
        help="the base ref the release branch was cut from (default: origin/dev)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="the release branch to check (default: the current branch)",
    )
    args = parser.parse_args(argv)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Release-branch merge-discipline gate\n")
    sys.stdout.write("=" * 60 + "\n")

    try:
        branch = args.branch or current_branch(repo_root=REPO_ROOT)
        merge_commits = list_merge_commits(args.base, branch, repo_root=REPO_ROOT)
        passed, messages = run_gate(merge_commits, base=args.base, branch=branch)
    except GateInputError as exc:
        sys.stderr.write(f"merge-discipline gate: {exc}\n")
        return EXIT_FAIL

    for line in messages:
        sys.stdout.write(line + "\n")
    sys.stdout.write("=" * 60 + "\n")
    verdict = "PASS" if passed else "FAIL"
    sys.stdout.write(f"MERGE DISCIPLINE GATE: {verdict}\n")
    sys.stdout.write("=" * 60 + "\n")

    return EXIT_OK if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())

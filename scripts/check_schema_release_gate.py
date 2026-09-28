"""Release gate: refuse a patch release that moved the core schema version.

``docs/upgrade.md`` tells operators that a patch upgrade never changes the
core schema, so they can decide whether an upgrade is safe to roll purely
from its version number. Nothing enforced that until this script: a release
that moved the schema and was tagged as a patch would previously pass every
other gate in this pipeline.

Two independent stamps carry the schema version, and this gate reads both
rather than trusting either alone:

* ``CORE_SCHEMA_HEAD_VERSION`` — the module constant the rest of the
  codebase reasons about (``src/engrava/infrastructure/sqlite/engrava_core.py``).
* the ``PRAGMA user_version = N`` line at the end of
  ``src/engrava/infrastructure/sqlite/schema_core.sql`` — the value a fresh
  bootstrap actually stamps into a database.

``tests/test_schema_version_gate.py`` already pins the two together for a
fresh bootstrap on whatever commit is checked out; it does not compare
across a release boundary. A gate that read only the constant (or only the
SQL stamp) here would be one refactor away from measuring nothing if the two
ever drifted apart again — so either stamp moving between the last released
tag and the candidate release counts as the schema having moved, and the two
stamps disagreeing at the candidate release is itself a hard failure.

The module constant did not exist for this project's earliest releases (it
was introduced well after several tags were already published), so it is
read as best-effort on *either* ref: its absence is expected history, not
an error, and the SQL stamp -- present since the first published release --
is the fallback whenever it is missing. Where the constant is present at
the candidate release, though, it must agree with the SQL stamp there, or
the gate refuses to judge the release at all.

Usage::

    python scripts/check_schema_release_gate.py --old-tag v0.6.0 --new-version 0.6.1

``--new-ref`` is optional and defaults to ``HEAD``.

``--old-tag`` is the last released tag (e.g. resolved in CI via
``git describe --tags --abbrev=0 HEAD``, since this script runs inside the
`prepare` lifecycle step, before semantic-release commits, tags, or pushes
anything, so no new tag sits at ``HEAD`` yet).
``--new-version`` is the version about to be published. ``--new-ref``
defaults to ``HEAD`` and exists mainly so this gate's own tests can point it
at an arbitrary historical commit without touching the working tree.

Exit codes:
* ``0`` — the release does not violate the rule (schema unchanged, or the
  bump is minor/major).
* ``1`` — the release violates the rule, or the inputs could not be
  resolved (bad ref, malformed version, disagreeing stamps at the candidate
  release).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGRAVA_CORE_PATH = "src/engrava/infrastructure/sqlite/engrava_core.py"
SCHEMA_SQL_PATH = "src/engrava/infrastructure/sqlite/schema_core.sql"

CONSTANT_RE = re.compile(r"(?m)^CORE_SCHEMA_HEAD_VERSION\s*=\s*(\d+)\s*$")
PRAGMA_RE = re.compile(r"(?m)^\s*PRAGMA\s+user_version\s*=\s*(\d+)\s*;")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

EXIT_OK = 0
EXIT_FAIL = 1


class GateInputError(RuntimeError):
    """Raised when a ref, path, or version string cannot be resolved."""


def _git_show(ref: str, path: str) -> str:
    """Return the text content of ``path`` as committed at ``ref``."""
    completed = subprocess.run(  # noqa: S603 -- trusted internal git invocation
        ["git", "show", f"{ref}:{path}"],  # noqa: S607 -- git resolved via PATH, not user input
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        msg = f"'git show {ref}:{path}' failed: {completed.stderr.strip()}"
        raise GateInputError(msg)
    return completed.stdout


def read_core_constant(ref: str) -> int | None:
    """Return ``CORE_SCHEMA_HEAD_VERSION`` as committed at ``ref``.

    Returns ``None`` if the file exists at ``ref`` but does not define the
    constant — true of every tag from before it was introduced.
    """
    text = _git_show(ref, ENGRAVA_CORE_PATH)
    match = CONSTANT_RE.search(text)
    return int(match.group(1)) if match else None


def read_sql_stamp(ref: str) -> int:
    """Return the ``PRAGMA user_version`` stamp committed in schema_core.sql at ``ref``.

    Raises :class:`GateInputError` if the file has no matching line — this
    stamp has been present since the first published release, so its
    absence means this gate no longer understands the file's shape rather
    than that the schema predates it.
    """
    text = _git_show(ref, SCHEMA_SQL_PATH)
    match = PRAGMA_RE.search(text)
    if match is None:
        msg = f"no 'PRAGMA user_version = N' stamp found in {SCHEMA_SQL_PATH} at {ref!r}"
        raise GateInputError(msg)
    return int(match.group(1))


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a bare ``MAJOR.MINOR.PATCH`` string (no leading ``v``)."""
    match = VERSION_RE.match(version.strip())
    if match is None:
        msg = f"{version!r} is not a MAJOR.MINOR.PATCH version"
        raise GateInputError(msg)
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_patch_only_bump(old_version: str, new_version: str) -> bool:
    """Return whether ``new_version`` differs from ``old_version`` only in the patch component."""
    old_major, old_minor, old_patch = parse_version(old_version)
    new_major, new_minor, new_patch = parse_version(new_version)
    return (old_major, old_minor) == (new_major, new_minor) and old_patch != new_patch


def schema_moved(
    *,
    old_sql: int,
    new_sql: int,
    old_constant: int | None,
    new_constant: int | None,
) -> bool:
    """Return whether either stamp shows the schema moved between the two refs.

    The constant is compared only when both refs define it. Older release
    tags predate the constant entirely (see the module docstring), so its
    absence on one or both sides is expected history, not evidence either
    way -- the SQL stamp, present since the first published release, is
    always the fallback.
    """
    if old_constant is not None and new_constant is not None and old_constant != new_constant:
        return True
    return old_sql != new_sql


def _constant_text(constant: int | None) -> str:
    if constant is None:
        return "CORE_SCHEMA_HEAD_VERSION not yet introduced at this ref"
    return f"CORE_SCHEMA_HEAD_VERSION = {constant}"


def run_gate(*, old_tag: str, new_version: str, new_ref: str) -> tuple[bool, list[str]]:
    """Apply the release rule and return ``(passed, report_lines)``."""
    old_version = old_tag.removeprefix("v")
    old_sql = read_sql_stamp(old_tag)
    new_sql = read_sql_stamp(new_ref)
    old_constant = read_core_constant(old_tag)
    new_constant = read_core_constant(new_ref)

    messages = [
        f"last released tag:  {old_tag}  (version {old_version})",
        f"  schema_core.sql PRAGMA user_version = {old_sql}, {_constant_text(old_constant)}",
        f"candidate release:  {new_ref}  (version {new_version})",
        f"  schema_core.sql PRAGMA user_version = {new_sql}, {_constant_text(new_constant)}",
    ]

    if new_constant is not None and new_constant != new_sql:
        messages.append(
            "FAIL: the two schema stamps disagree at the candidate release -- "
            f"CORE_SCHEMA_HEAD_VERSION = {new_constant} but "
            f"schema_core.sql PRAGMA user_version = {new_sql}. A release cannot ship "
            "while its own two schema stamps contradict each other.",
        )
        return False, messages

    moved = schema_moved(
        old_sql=old_sql,
        new_sql=new_sql,
        old_constant=old_constant,
        new_constant=new_constant,
    )
    patch_only = is_patch_only_bump(old_version, new_version)

    if moved and patch_only:
        messages.append(
            f"FAIL: the core schema version moved ({old_sql} -> {new_sql}) but "
            f"{old_version} -> {new_version} is a patch-only bump. A schema change "
            "must ship with at least a minor version bump.",
        )
        return False, messages

    if moved:
        messages.append(
            f"PASS: the core schema version moved ({old_sql} -> {new_sql}) and "
            f"{old_version} -> {new_version} is not a patch-only bump.",
        )
    else:
        messages.append(
            f"PASS: the core schema version did not move ({old_sql}); "
            f"{old_version} -> {new_version} may safely be a patch.",
        )
    return True, messages


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the schema-version release gate."""
    parser = argparse.ArgumentParser(
        prog="check_schema_release_gate.py",
        description=(
            "Release gate. Fails when the core schema version moved since "
            "the last released tag and the candidate release's version "
            "bump is patch-only."
        ),
    )
    parser.add_argument(
        "--old-tag",
        required=True,
        help="the last released tag, e.g. v0.6.0",
    )
    parser.add_argument(
        "--new-version",
        required=True,
        help="the version about to be released, e.g. 0.6.1 (no leading 'v')",
    )
    parser.add_argument(
        "--new-ref",
        default="HEAD",
        help="git ref to read the candidate release's schema stamps from (default: HEAD)",
    )
    args = parser.parse_args(argv)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Schema-version release gate\n")
    sys.stdout.write("=" * 60 + "\n")

    try:
        passed, messages = run_gate(
            old_tag=args.old_tag,
            new_version=args.new_version,
            new_ref=args.new_ref,
        )
    except GateInputError as exc:
        sys.stderr.write(f"schema release gate: {exc}\n")
        return EXIT_FAIL

    for line in messages:
        sys.stdout.write(line + "\n")
    sys.stdout.write("=" * 60 + "\n")
    verdict = "PASS" if passed else "FAIL -- publish blocked"
    sys.stdout.write(f"SCHEMA RELEASE GATE: {verdict}\n")
    sys.stdout.write("=" * 60 + "\n")

    return EXIT_OK if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())

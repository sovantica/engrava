"""Census logic comparing ``examples/README.md`` against what ``examples/`` ships.

``examples/README.md`` is not part of the documentation-example suite's scope
(``tests/docs/_md_blocks.py:markdown_files()`` is ``README.md`` plus every
``*.md`` under ``docs/`` — by design, see that module's docstring). Adding it
there would pull it through every layer built for narrative documentation at
once, including the ``bash``-invocation layer
(``tests/docs/test_docs_shell_examples.py``), which checks a fenced ``bash``
line's ``engrava ...`` invocation against the real CLI command tree — a
question this page's three ``bash`` blocks (``pip install ...``,
``python examples/<x>.py`` lines, ``python -m engrava.benchmarks.synthetic``)
never ask, so widening the shared scope would only produce exemption
registrations for blocks that were never in that layer's problem domain.

This module answers the narrower, actual question instead: does the page's
inventory match the directory on disk, in both directions? It is deliberately
self-contained — no dependency on ``tests/docs/_md_blocks.py`` — so a change to
the narrative-docs pipeline can never silently affect this census, and vice
versa.

A markdown *link* is the unit of "indexed", not a filename's occurrence in
running prose: a file can be named in a sentence without being a real entry a
reader can click through to, and counting substrings would miss that
distinction (and would also miss a stale link to a file that no longer
exists, which names a *different* filename than the one prompting the
sentence). Every inline link ``[text](target)`` in the page is extracted by
its markdown link structure; a target is resolved relative to the
``examples/`` directory itself (not necessarily the checked README's own
location — see ``compute_census``), and classified as an ``examples/``-local
inventory entry only when it names a bare filename (no ``/`` in it) — a
cross-reference up into ``docs/`` or out to another repository is still
checked for resolving, but it is not part of the ``examples/`` inventory
question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# tests/examples/_examples_census.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES_README = EXAMPLES_DIR / "README.md"

# A markdown inline link: `[text](target)`. Good enough for this page's own
# authoring style (no reference-style `[text][ref]` links today); this module
# is not a general markdown parser, just this one page's link structure.
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

# A Markdown table row: a line that both opens and closes with `|`. Matches a
# table's own header-separator row (`|---|---|`) too, which is excluded
# separately below rather than relied on to simply carry no link.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

# `python examples/<file>` printed as a runnable invocation line, on a line by
# itself inside a fenced block (the page's own convention for these).
_INVOCATION_RE = re.compile(r"(?m)^\s*python\s+examples/(\S+\.py)\s*$")

# Files that live in examples/ but are not part of the shipped inventory the
# page indexes. The page itself is the only current member; if a non-shipped
# helper file (`__init__.py`, `.gitignore`, ...) is ever added to the
# directory, register it here explicitly rather than excluding it by a
# guessed pattern.
_NOT_INVENTORY: frozenset[str] = frozenset({"README.md"})

# A shipped file's index entry is a row of one of the page's tables (`Script
# | What it shows`, `File | Profile | What you get`, `File | For`) -- a link
# named in running prose is not the same claim, and counting it the same way
# would let a file quietly drop out of its table (demoted to a passing
# mention) without the census noticing. `config.yaml` is the one deliberate
# exception: the paragraph right after the profiles table narrates it by
# name precisely because it is *not* a fourth profile, so it is indexed in
# prose on purpose rather than by omission.
_PROSE_INDEXED_EXCEPTIONS: frozenset[str] = frozenset({"config.yaml"})


def _local_link_targets(text: str) -> list[str]:
    """Every non-``http(s)``/``mailto`` link target in ``text``, fragment stripped."""
    targets: list[str] = []
    for raw_target in _LINK_RE.findall(text):
        if raw_target.startswith(("http://", "https://", "mailto:")):
            continue
        target = raw_target.split("#", 1)[0]
        if target:
            targets.append(target)
    return targets


def _table_row_link_targets(text: str) -> list[str]:
    """Every non-``http(s)``/``mailto`` link target inside a Markdown table row."""
    row_lines = [
        line
        for line in text.splitlines()
        if _TABLE_ROW_RE.match(line) and not _TABLE_SEPARATOR_RE.match(line)
    ]
    return _local_link_targets("\n".join(row_lines))


@dataclass(frozen=True)
class ExamplesCensus:
    """The result of comparing one README's index against one directory's files."""

    shipped_files: frozenset[str]
    indexed_files: dict[str, int]
    unresolved_relative_links: tuple[str, ...]
    dangling_invocations: tuple[str, ...]

    @property
    def missing_from_index(self) -> frozenset[str]:
        """Shipped files with zero index entries."""
        return self.shipped_files - self.indexed_files.keys()

    @property
    def missing_from_disk(self) -> frozenset[str]:
        """Indexed same-directory files that do not exist on disk."""
        return frozenset(self.indexed_files) - self.shipped_files

    @property
    def duplicated_in_index(self) -> dict[str, int]:
        """Shipped files linked more than once from the index."""
        return {
            name: count
            for name, count in self.indexed_files.items()
            if count > 1 and name in self.shipped_files
        }


def compute_census(examples_dir: Path, readme_path: Path) -> ExamplesCensus:
    """Compare ``readme_path``'s links against the files ``examples_dir`` ships.

    Args:
        examples_dir: The directory whose files are the shipped inventory
            (excludes the README itself). Every relative link in the index
            resolves against this directory too, deliberately **not** against
            ``readme_path``'s own parent: the two coincide for the real
            ``examples/README.md``, but a caller checking a README pulled out
            of another tree state (e.g. via ``git show <rev>:examples/README.md``
            into a temp file) still wants links resolved against the real,
            current directory content, not against the temp file's location.
        readme_path: The index page to check.

    Returns:
        An :class:`ExamplesCensus` capturing both census directions, every
        unresolved relative link, and every documented invocation naming a
        file that does not exist.

    """
    shipped = frozenset(
        entry.name
        for entry in examples_dir.iterdir()
        if entry.is_file() and entry.name not in _NOT_INVENTORY
    )
    text = readme_path.read_text(encoding="utf-8")

    unresolved = [
        target for target in _local_link_targets(text) if not (examples_dir / target).exists()
    ]

    indexed: dict[str, int] = {}
    for target in _table_row_link_targets(text):
        if "/" not in target:
            indexed[target] = indexed.get(target, 0) + 1
    for target in _local_link_targets(text):
        if target in _PROSE_INDEXED_EXCEPTIONS and "/" not in target:
            indexed[target] = indexed.get(target, 0) + 1

    dangling = [
        fname
        for fname in dict.fromkeys(_INVOCATION_RE.findall(text))
        if not (examples_dir / fname).is_file()
    ]

    return ExamplesCensus(
        shipped_files=shipped,
        indexed_files=indexed,
        unresolved_relative_links=tuple(sorted(set(unresolved))),
        dangling_invocations=tuple(dangling),
    )

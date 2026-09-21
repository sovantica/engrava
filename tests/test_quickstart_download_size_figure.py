"""Cross-file agreement on the ``embeddings-local`` one-time download size.

``docs/quickstart.md``'s installation table is the designated source of
truth for the ``torch`` + ``sentence-transformers`` one-time download size
figure (``~NNN+ MB``). The same figure is restated three more times: a
second mention later in ``docs/quickstart.md`` itself, once in
``examples/README.md``, and once in ``examples/quickstart.py``'s module
docstring. Each restatement is read back from the source of truth and
compared, rather than encoded as a separate literal -- a figure written
independently in four places is wrong in three of them within a milestone
of the real number changing, and this test exists to make that impossible
to miss.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUICKSTART_DOC = REPO_ROOT / "docs" / "quickstart.md"
EXAMPLES_README = REPO_ROOT / "examples" / "README.md"
EXAMPLES_QUICKSTART_PY = REPO_ROOT / "examples" / "quickstart.py"

_FIGURE = re.compile(r"~\d+\+ MB")


def _figures_in(path: Path) -> list[str]:
    return _FIGURE.findall(path.read_text(encoding="utf-8"))


def _source_of_truth() -> str:
    """Return the download-size figure from quickstart.md's installation table.

    The table row is the first ``~NNN+ MB`` mention in the file.
    """
    figures = _figures_in(QUICKSTART_DOC)
    assert figures, f"no '~NNN+ MB' figure found in {QUICKSTART_DOC.name} at all"
    return figures[0]


def test_quickstart_doc_second_mention_matches_its_own_table() -> None:
    figures = _figures_in(QUICKSTART_DOC)
    assert len(figures) == 2, (
        f"expected exactly two '~NNN+ MB' mentions in {QUICKSTART_DOC.name}, found {figures!r}"
    )
    table_figure, second_mention = figures
    assert second_mention == table_figure, (
        f"{QUICKSTART_DOC.name}'s second download-size mention ({second_mention!r}) "
        f"disagrees with its own installation-table figure ({table_figure!r})"
    )


def test_examples_readme_matches_quickstart_doc_figure() -> None:
    source_of_truth = _source_of_truth()
    figures = _figures_in(EXAMPLES_README)
    assert figures == [source_of_truth], (
        f"examples/README.md states {figures!r}, but the source of truth in "
        f"{QUICKSTART_DOC.name} is {source_of_truth!r}"
    )


def test_examples_quickstart_py_matches_quickstart_doc_figure() -> None:
    source_of_truth = _source_of_truth()
    figures = _figures_in(EXAMPLES_QUICKSTART_PY)
    assert figures == [source_of_truth], (
        f"examples/quickstart.py states {figures!r}, but the source of truth in "
        f"{QUICKSTART_DOC.name} is {source_of_truth!r}"
    )

"""Couples the exception classes in the domain module to ``docs/error-handling.md``.

``docs/error-handling.md`` is the page a reader follows for operational retry,
repair and store-replacement guidance, so an exception class it never names is
one the reader gets no recovery guidance for. This module does not keep a second
list of exception names that could itself drift from the code: it parses
``src/engrava/domain/exceptions.py`` with :mod:`ast`, collects every class it
defines, and asserts each one is named on the page as a backtick-quoted token.

A backtick-quoted whole token is required, not a substring match, so that
``ThoughtNotFoundError`` cannot be satisfied by a page that only names
``SourceThoughtNotFoundError``.
"""

from __future__ import annotations

import ast
import re

from tests.docs._md_blocks import REPO_ROOT

EXCEPTIONS_MODULE = REPO_ROOT / "src" / "engrava" / "domain" / "exceptions.py"
ERROR_HANDLING = REPO_ROOT / "docs" / "error-handling.md"


def _defined_exception_classes() -> list[str]:
    """Return every class name defined at module level in the exceptions module.

    Derived by parsing the source, not hardcoded, so the list cannot drift from
    the module it describes.
    """
    tree = ast.parse(EXCEPTIONS_MODULE.read_text(encoding="utf-8"))
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _is_named(name: str, text: str) -> bool:
    """Return whether ``name`` appears in ``text`` as a backtick-quoted whole token."""
    return re.search(rf"`{re.escape(name)}`", text) is not None


def test_registry_is_nonempty() -> None:
    """Guard against the ast-derived class list silently being empty."""
    names = _defined_exception_classes()
    assert "EngravaError" in names, (
        "parsing exceptions.py did not yield EngravaError; the derivation in this "
        "module (or the module layout) likely broke."
    )
    assert len(names) >= 28, f"expected at least 28 exception classes, parsed {len(names)}"


def test_every_domain_exception_is_named_on_the_error_handling_page() -> None:
    """Every exception class in ``exceptions.py`` is named in ``error-handling.md``."""
    document_text = ERROR_HANDLING.read_text(encoding="utf-8")

    missing = [name for name in _defined_exception_classes() if not _is_named(name, document_text)]
    assert not missing, (
        f"these exception classes are not named (backtick-quoted) anywhere in "
        f"{ERROR_HANDLING.relative_to(REPO_ROOT)}: {missing}. Add each to the decision "
        "table, with the recovery guidance its raise site and docstring support."
    )


def test_name_check_is_a_whole_token_match() -> None:
    """A longer name that merely contains a class name does not count as naming it."""
    assert not _is_named("ThoughtNotFoundError", "`SourceThoughtNotFoundError`")
    assert not _is_named("ThoughtNotFoundError", "ThoughtNotFoundError without quotes")
    assert _is_named("ThoughtNotFoundError", "`ThoughtNotFoundError`, `ActionNotFoundError`")

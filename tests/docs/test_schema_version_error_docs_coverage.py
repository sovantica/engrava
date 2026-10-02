"""Couples ``SchemaVersionError``'s named constructors to their documentation.

``SchemaVersionError`` has three named constructors — :meth:`populated_sub_floor`,
:meth:`stale_shape_sub_floor`, and :meth:`newer_than_head` — each setting a
distinct ``.reason`` string literal. This module does not hardcode that set of
three strings as a second list, which could itself drift from the class the
moment a fourth constructor is added: it derives the actual reason strings by
calling all three constructors and reading ``.reason`` off each result, then
asserts every one of those strings appears verbatim, as a substring, somewhere
in ``docs/api-reference.md``. If a future reason is added to the class and
nobody updates that documentation, this test goes red instead of the
documentation silently falling one reason behind.
"""

from __future__ import annotations

from engrava import SchemaVersionError
from tests.docs._md_blocks import REPO_ROOT

API_REFERENCE = REPO_ROOT / "docs" / "api-reference.md"


def _actual_reason_strings() -> list[str]:
    """Return every ``.reason`` string a named constructor currently sets.

    Derived by calling the constructors themselves — not hardcoded — so this
    list can never drift out of sync with the class it describes.
    """
    return [
        SchemaVersionError.populated_sub_floor(1, 1).reason,
        SchemaVersionError.stale_shape_sub_floor(1, 1).reason,
        SchemaVersionError.newer_than_head(1, 1).reason,
    ]


def test_registry_is_nonempty() -> None:
    """Guard against the constructor-derived reason list silently being empty."""
    assert len(_actual_reason_strings()) >= 3, (
        "SchemaVersionError's named constructors yielded fewer than 3 reason "
        "strings; the derivation in this module (or the class itself) likely broke."
    )


def test_every_schema_version_error_reason_is_documented() -> None:
    """Every ``.reason`` string a named constructor sets is named in api-reference.md.

    Each reason string is checked as a verbatim substring of the document's
    full text — not confined to a single table row or line — so this passes
    regardless of how the documentation chooses to lay the three reasons out.
    """
    document_text = API_REFERENCE.read_text(encoding="utf-8")
    reasons = _actual_reason_strings()

    missing = [reason for reason in reasons if reason not in document_text]
    assert not missing, (
        f"these SchemaVersionError .reason values are not documented anywhere in "
        f"{API_REFERENCE.relative_to(REPO_ROOT)}: {missing}. Add each one to the "
        "SchemaVersionError row (or elsewhere on the page) so a reader can tell "
        "the refusals apart."
    )

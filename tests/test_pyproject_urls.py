"""Pins ``pyproject.toml``'s package-metadata Documentation URL.

This is a pinned-value test only: it catches an accidental revert of the
URL string (e.g. back to a stale, redirecting address) between releases. It
does **not** confirm the URL resolves or that the redirect chain behind an
old value still works -- that needs a person running ``curl`` by hand
whenever the value is deliberately changed. No network call belongs in this
suite.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

EXPECTED_DOCUMENTATION_URL = "https://docs.engrava.ai/"


def test_documentation_url_is_pinned_to_the_docs_subdomain() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    documentation_url = data["project"]["urls"]["Documentation"]
    assert documentation_url == EXPECTED_DOCUMENTATION_URL, (
        f"pyproject.toml [project.urls].Documentation is {documentation_url!r}, "
        f"expected {EXPECTED_DOCUMENTATION_URL!r} -- if this change is "
        "intentional, confirm the new URL is live before updating this test"
    )

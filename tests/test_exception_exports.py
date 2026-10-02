"""Every public exception class in ``engrava.domain.exceptions`` is exported from ``engrava``.

The rule: every public class (no leading underscore) defined in
``engrava.domain.exceptions`` that subclasses ``Exception`` is importable from
``engrava`` and listed in ``engrava.__all__``. The classes are discovered from
the module rather than named here, so a class added later without an export
fails this module. Each name must resolve from the package root to the exact
class the raising code raises.
"""

from __future__ import annotations

import inspect

import pytest

import engrava
from engrava.domain import exceptions as domain_exceptions


def _public_exception_classes() -> dict[str, type[Exception]]:
    """Return every public exception class defined in ``engrava.domain.exceptions``.

    "Public" means no leading underscore; "defined" means the class's own
    ``__module__`` is the exceptions module, so a class merely imported into it
    from elsewhere is not claimed as one of its own.
    """
    return {
        name: cls
        for name, cls in inspect.getmembers(domain_exceptions, inspect.isclass)
        if not name.startswith("_")
        and cls.__module__ == domain_exceptions.__name__
        and issubclass(cls, Exception)
    }


PUBLIC_EXCEPTIONS = _public_exception_classes()


def test_discovery_finds_the_public_exceptions() -> None:
    """Guard the discovery: an empty or partial scan would make every export check vacuous."""
    assert "EngravaError" in PUBLIC_EXCEPTIONS
    assert len(PUBLIC_EXCEPTIONS) > 1


class TestTopLevelExceptionExports:
    """``from engrava import <name>`` must yield the raising code's own class."""

    @pytest.mark.parametrize("name", sorted(PUBLIC_EXCEPTIONS))
    def test_exception_is_importable_and_identical(self, name: str) -> None:
        assert hasattr(engrava, name), (
            f"engrava.domain.exceptions.{name} is not importable from the package root; "
            f"import it in engrava/__init__.py and add {name!r} to __all__"
        )
        assert getattr(engrava, name) is PUBLIC_EXCEPTIONS[name]

    @pytest.mark.parametrize("name", sorted(PUBLIC_EXCEPTIONS))
    def test_exception_is_listed_in_all(self, name: str) -> None:
        assert name in engrava.__all__, f"{name} is not listed in engrava.__all__"

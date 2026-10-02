"""The eight pre-rename (``MindStore*``) aliases are kept, not removed.

The policy recorded in ``engrava/__init__.py`` is to keep these names —
removing them would be a breaking change this project's versioning would
turn into a major release, which nobody has asked for. Committing to the
names is not the same as blessing them: each one must still resolve to the
*exact same object* as its current-name counterpart (so a caller who mixes
old and new names gets one class, one function of `isinstance` and
equality checks), and each one must still raise ``DeprecationWarning`` on
access, because that warning is the entire value of keeping them and
nothing else in the module pins it — a refactor of ``__getattr__`` could
silently drop it and nothing would fail.
"""

from __future__ import annotations

import warnings

import pytest

import engrava

_ALIAS_TO_CURRENT_NAME = {
    "SqliteMindStoreCore": "SqliteEngravaCore",
    "MindStoreManager": "EngravaManager",
    "MindStoreConfig": "EngravaConfig",
    "MindStoreError": "EngravaError",
    "MindStoreCoreProtocol": "EngravaCoreProtocol",
    "MindStoreHooksProtocol": "EngravaHooksProtocol",
    "DefaultMindStoreHooks": "DefaultEngravaHooks",
    "ReadOnlyMindStore": "ReadOnlyEngrava",
}


@pytest.mark.parametrize(
    ("alias_name", "current_name"),
    sorted(_ALIAS_TO_CURRENT_NAME.items()),
)
def test_alias_warns_and_resolves_to_the_current_object(alias_name: str, current_name: str) -> None:
    """Accessing the alias warns, and yields the current name's exact object."""
    current_object = getattr(engrava, current_name)

    with pytest.warns(DeprecationWarning, match=alias_name):
        alias_object = getattr(engrava, alias_name)

    assert alias_object is current_object


def test_every_alias_is_covered_by_this_module() -> None:
    """Guards the table above against silently losing or gaining a name.

    Compares the table's keys directly against ``engrava._DEPRECATED_ALIASES``
    — the actual source of truth ``__getattr__`` reads from — rather than a
    count pinned in this module. If an alias is added to or removed from that
    module-level constant without a matching change here, this fails naming
    the mismatched keys, instead of passing by coincidence.
    """
    assert set(_ALIAS_TO_CURRENT_NAME) == set(engrava._DEPRECATED_ALIASES)


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """``__getattr__`` must not swallow genuinely missing names."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(AttributeError, match="has no attribute"):
            engrava.__getattr__("NotARealExport")

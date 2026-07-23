"""Architecture contracts for the Dreaming extension boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import engrava
import engrava._composition as composition_root
import engrava.domain.dreaming as inward_dreaming
import engrava.domain.protocols.dreaming as dreaming_protocols
import engrava.extensions.dreaming as dreaming_extension
import engrava.infrastructure.sqlite.engrava_core as sqlite_core
import engrava.infrastructure.sqlite.hygiene as sqlite_hygiene
from engrava.config import DreamingConfig
from engrava.extensions import dreaming_signals as legacy_signals
from engrava.infrastructure.sqlite import centroid as legacy_centroid

if TYPE_CHECKING:
    from types import ModuleType


_DREAMING_STORE_CAPABILITIES = {
    "count_thoughts",
    "create_edge",
    "create_thought",
    "get_edges",
    "get_embedding",
    "get_thought",
    "list_edges",
    "list_thoughts",
    "retire_orphan_reflections",
    "search_similar",
    "store_embedding",
    "suspend_auto_commit",
    "thought_exists_by_source",
    "update_thought",
}


def _module_path(module: ModuleType) -> Path:
    module_file = module.__file__
    assert module_file is not None
    return Path(module_file)


def _import_targets(module: ModuleType) -> set[str]:
    tree = ast.parse(_module_path(module).read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.add(node.module)
    return targets


def test_inward_dreaming_modules_have_no_outward_dependencies() -> None:
    for module in (inward_dreaming, dreaming_protocols):
        imports = _import_targets(module)
        assert not any(
            target.startswith(("engrava.extensions", "engrava.infrastructure"))
            for target in imports
        )


def test_dreaming_extension_has_no_sqlite_dependency() -> None:
    imports = _import_targets(dreaming_extension)
    assert not any(target.startswith("engrava.infrastructure.sqlite") for target in imports)
    source = _module_path(dreaming_extension).read_text(encoding="utf-8")
    assert "SqliteEngravaCore" not in source


def test_sqlite_modules_do_not_import_dreaming_implementation() -> None:
    for module in (sqlite_core, sqlite_hygiene):
        imports = _import_targets(module)
        assert not any(target.startswith("engrava.extensions.dreaming") for target in imports)
    core_source = _module_path(sqlite_core).read_text(encoding="utf-8")
    assert "DreamingExtension" not in core_source
    assert "DreamingConsolidatorProtocol" in core_source


def test_composition_root_owns_optional_dreaming_construction() -> None:
    imports = _import_targets(composition_root)
    assert "engrava.extensions.dreaming" in imports

    consolidator = composition_root.compose_dreaming_consolidator(DreamingConfig(enabled=True))
    assert isinstance(consolidator, dreaming_extension.DreamingExtension)


def test_dreaming_store_protocol_exposes_exact_capability_set() -> None:
    protocol_members = {
        name
        for name, value in dreaming_protocols.DreamingStoreProtocol.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert protocol_members == _DREAMING_STORE_CAPABILITIES


def test_legacy_and_top_level_imports_reexport_inward_objects() -> None:
    assert legacy_signals.DreamingContext is inward_dreaming.DreamingContext
    assert legacy_signals.DreamingSignalProtocol is inward_dreaming.DreamingSignalProtocol
    assert legacy_signals.DEFAULT_SIGNALS is inward_dreaming.DEFAULT_SIGNALS
    assert legacy_centroid.compute_centroid is inward_dreaming.compute_centroid
    assert legacy_centroid.CENTROID_MODEL_NAME == inward_dreaming.CENTROID_MODEL_NAME
    assert dreaming_extension.ConsolidationResult is inward_dreaming.ConsolidationResult
    top_level_exports = (
        "ActionOutcomeSignal",
        "ConfidenceSignal",
        "ConfirmationSignal",
        "ConsolidationResult",
        "DreamingContext",
        "DreamingSignalProtocol",
        "FrequencySignal",
        "RecencySignal",
        "StalenessSignal",
    )
    for name in top_level_exports:
        assert getattr(engrava, name) is getattr(inward_dreaming, name)


class _ConfiguredSubclass(sqlite_core.SqliteEngravaCore):
    """Concrete subclass used to pin the classmethod's ``Self`` behavior."""


async def test_from_config_preserves_subclass_construction(tmp_path: Path) -> None:
    config_path = tmp_path / "engrava.yaml"
    config_path.write_text(
        f"database:\n  path: {tmp_path / 'subclass.db'}\n",
        encoding="utf-8",
    )

    store = await _ConfiguredSubclass.from_config(config_path)
    try:
        assert type(store) is _ConfiguredSubclass
    finally:
        await store.close()

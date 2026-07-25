"""Architecture contracts for the Dreaming extension boundary."""

from __future__ import annotations

import ast
import inspect
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import engrava
import engrava._composition as composition_root
import engrava.domain.dreaming as inward_dreaming
import engrava.domain.protocols.dreaming as dreaming_protocols
import engrava.extensions.dreaming as dreaming_extension
import engrava.infrastructure.sqlite.engrava_core as sqlite_core
import engrava.infrastructure.sqlite.hygiene as sqlite_hygiene
from engrava.config import DreamingConfig
from engrava.domain.dreaming import (
    CENTROID_MODEL_NAME,
    DEFAULT_SIGNALS,
    ActionOutcomeSignal,
    ConfidenceSignal,
    ConfirmationSignal,
    ConsolidationResult,
    DreamingContext,
    FrequencySignal,
    RecencySignal,
    StalenessSignal,
    compute_centroid,
    default_signal_active,
)
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
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


# ------------------------------------------------------------------
# Transitive import boundary
# ------------------------------------------------------------------

_PACKAGE_ROOT = Path(engrava.__file__).parent
_SRC_ROOT = _PACKAGE_ROOT.parent

#: The one deliberate route by which infrastructure reaches an extension
#: implementation, spelled out hop by hop.  ``SqliteEngravaCore.from_config()``
#: is documented public API and must keep wiring the configured Dreaming
#: extension, so the SQLite facade defers construction to the composition root
#: instead of importing Dreaming itself.  Moving that factory to a
#: package-level composition root would change public API, so the hop is
#: deferred beyond this release and pinned here rather than hidden.
_SANCTIONED_DREAMING_PATH = (
    "engrava.infrastructure.sqlite.engrava_core",
    "engrava._composition",
    "engrava.extensions.dreaming",
)

#: Crossings that predate the Dreaming inversion and are unrelated to it: the
#: optional sqlite-vec vector backend, and configuration's optional extension
#: manifest discovery helper.  They are enumerated so the walk below can be an
#: exact allow-list instead of a "known-bad prefix" filter -- extending this
#: tuple is an architecture decision, not a test fix.
_PRE_EXISTING_EXTENSION_CROSSINGS = (
    ("engrava.config", "engrava.extensions.discovery"),
    (
        "engrava.infrastructure.sqlite.engrava_core",
        "engrava.extensions.vector_sqlite_vec",
    ),
)

#: Every edge from a module reachable out of ``engrava.infrastructure`` into
#: ``engrava.extensions``.  Any other such edge -- direct or transitive, new or
#: widened -- fails the walk below.
_ALLOWED_EXTENSION_CROSSINGS = frozenset(
    {
        (_SANCTIONED_DREAMING_PATH[1], _SANCTIONED_DREAMING_PATH[2]),
        *_PRE_EXISTING_EXTENSION_CROSSINGS,
    }
)


def _package_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        parts = list(path.relative_to(_SRC_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


def _resolve_relative_import(module: str, path: Path, node: ast.ImportFrom) -> str:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _module_edges(module: str, path: Path, known: set[str]) -> set[str]:
    """Collect the ``engrava`` modules one source file imports.

    Every import statement counts, including ones deferred into a function body
    and ones guarded by ``TYPE_CHECKING``: both couple the two modules at the
    source level, so neither may be used to slip a forbidden dependency past
    this contract.  Implicit parent-package imports are deliberately not
    modelled -- ``engrava/__init__.py`` re-exports the whole public surface, so
    charging every module with its ancestors' imports would make the graph
    say nothing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.update(alias.name for alias in node.names if alias.name in known)
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative_import(module, path, node) if node.level else node.module
            if target is None:
                continue
            if target in known:
                edges.add(target)
            edges.update(
                f"{target}.{alias.name}"
                for alias in node.names
                if f"{target}.{alias.name}" in known
            )
    return edges


def _import_graph() -> dict[str, set[str]]:
    modules = _package_modules()
    known = set(modules)
    return {name: _module_edges(name, path, known) for name, path in modules.items()}


def _is_extension(module: str) -> bool:
    return module == "engrava.extensions" or module.startswith("engrava.extensions.")


def _infrastructure_modules(graph: dict[str, set[str]]) -> list[str]:
    return sorted(
        name
        for name in graph
        if name == "engrava.infrastructure" or name.startswith("engrava.infrastructure.")
    )


def _reachable_from_infrastructure(graph: dict[str, set[str]]) -> set[str]:
    """Walk outward from every infrastructure module, stopping at extensions.

    Traversal does not continue *through* an extension module: once the
    boundary is crossed the edge has already been recorded, and whatever an
    extension imports afterwards is not an infrastructure dependency.
    """
    seeds = _infrastructure_modules(graph)
    seen = set(seeds)
    queue = deque(seeds)
    while queue:
        current = queue.popleft()
        if _is_extension(current):
            continue
        for target in graph[current]:
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def _route_from_infrastructure(graph: dict[str, set[str]], target: str) -> list[str]:
    parents: dict[str, str] = {}
    seeds = _infrastructure_modules(graph)
    seen = set(seeds)
    queue = deque(seeds)
    while queue:
        current = queue.popleft()
        if current == target:
            route = [current]
            while route[-1] in parents:
                route.append(parents[route[-1]])
            return list(reversed(route))
        if _is_extension(current):
            continue
        for neighbour in sorted(graph[current]):
            if neighbour not in seen:
                seen.add(neighbour)
                parents[neighbour] = current
                queue.append(neighbour)
    return [target]


def _extension_crossings(graph: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {
        (source, target)
        for source in _reachable_from_infrastructure(graph)
        if not _is_extension(source)
        for target in graph[source]
        if _is_extension(target)
    }


def _describe_boundary_drift(
    graph: dict[str, set[str]],
    unexpected: set[tuple[str, str]],
    stale: set[tuple[str, str]],
) -> str:
    lines = [
        f"new infrastructure -> extension edge: {source} -> {target}"
        f"\n    reached as: {' -> '.join([*_route_from_infrastructure(graph, source), target])}"
        for source, target in sorted(unexpected)
    ]
    lines += [
        f"allow-listed edge no longer exists, delete it from the allow-list: {source} -> {target}"
        for source, target in sorted(stale)
    ]
    return "\n".join(lines)


def test_infrastructure_reaches_extensions_only_through_allowed_edges() -> None:
    graph = _import_graph()
    crossings = _extension_crossings(graph)
    drift = _describe_boundary_drift(
        graph,
        crossings - _ALLOWED_EXTENSION_CROSSINGS,
        _ALLOWED_EXTENSION_CROSSINGS - crossings,
    )
    assert not drift, f"the infrastructure -> extension boundary moved:\n{drift}"


def test_sanctioned_dreaming_path_is_the_only_bridge_to_the_extension() -> None:
    graph = _import_graph()
    entry, bridge, extension = _SANCTIONED_DREAMING_PATH
    assert bridge in graph[entry]
    assert extension in graph[bridge]

    bridge_importers = sorted(name for name, targets in graph.items() if bridge in targets)
    assert bridge_importers == [entry]

    bridged_extensions = sorted(target for target in graph[bridge] if _is_extension(target))
    assert bridged_extensions == [extension]

    dreaming_importers = sorted(
        name
        for name in _reachable_from_infrastructure(graph)
        if not _is_extension(name)
        and name != bridge
        and any(target.startswith("engrava.extensions.dreaming") for target in graph[name])
    )
    assert dreaming_importers == []


# ------------------------------------------------------------------
# Protocol signature compatibility
# ------------------------------------------------------------------

#: ``SqliteEngravaCore.suspend_auto_commit`` is an ``@asynccontextmanager``
#: generator, so ``functools.wraps`` keeps the *generator's* annotation while
#: the call returns the async context manager the protocol declares.  Both
#: spellings are pinned, so a change to either side still fails.
_RETURN_ANNOTATION_EXCEPTIONS = {
    "suspend_auto_commit": ("AbstractAsyncContextManager[None]", "AsyncIterator[None]"),
}

_VARIADIC_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _declared_parameters(signature: inspect.Signature) -> list[inspect.Parameter]:
    """Return the parameters a caller supplies, i.e. everything after ``self``."""
    return list(signature.parameters.values())[1:]


def _positional_parameters(signature: inspect.Signature) -> list[inspect.Parameter]:
    return [
        parameter
        for parameter in _declared_parameters(signature)
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


def _parameter_mismatches(
    protocol_signature: inspect.Signature,
    impl_signature: inspect.Signature,
) -> list[str]:
    mismatches: list[str] = []
    impl_positional = _positional_parameters(impl_signature)
    for index, expected in enumerate(_positional_parameters(protocol_signature)):
        actual_name = impl_positional[index].name if index < len(impl_positional) else None
        if actual_name != expected.name:
            mismatches.append(
                f"positional slot {index} is {actual_name!r}, protocol declares {expected.name!r}"
            )

    impl_parameters = {
        parameter.name: parameter for parameter in _declared_parameters(impl_signature)
    }
    for expected in _declared_parameters(protocol_signature):
        actual = impl_parameters.get(expected.name)
        if actual is None:
            mismatches.append(f"parameter {expected.name!r} is not implemented")
            continue
        if actual.kind is not expected.kind:
            mismatches.append(
                f"parameter {expected.name!r} is {actual.kind!s},"
                f" protocol declares {expected.kind!s}"
            )
        if actual.default != expected.default:
            mismatches.append(
                f"parameter {expected.name!r} defaults to {actual.default!r},"
                f" protocol declares {expected.default!r}"
            )
        if str(actual.annotation) != str(expected.annotation):
            mismatches.append(
                f"parameter {expected.name!r} is annotated {actual.annotation!r},"
                f" protocol declares {expected.annotation!r}"
            )

    declared = {parameter.name for parameter in _declared_parameters(protocol_signature)}
    mismatches += [
        f"implementation adds required parameter {parameter.name!r}"
        for parameter in impl_parameters.values()
        if parameter.name not in declared
        and parameter.default is inspect.Parameter.empty
        and parameter.kind not in _VARIADIC_KINDS
    ]
    return mismatches


def _return_annotation_mismatches(
    member: str,
    protocol_signature: inspect.Signature,
    impl_signature: inspect.Signature,
) -> list[str]:
    expected = protocol_signature.return_annotation
    actual = impl_signature.return_annotation
    exception = _RETURN_ANNOTATION_EXCEPTIONS.get(member)
    if exception is not None:
        if (str(expected), str(actual)) != exception:
            return [
                f"documented return-annotation exception is stale:"
                f" got ({str(expected)!r}, {str(actual)!r}), expected {exception!r}"
            ]
        return []
    if inspect.Signature.empty in (expected, actual):
        return []
    if str(expected) != str(actual):
        return [f"returns {actual!r}, protocol declares {expected!r}"]
    return []


def test_dreaming_store_protocol_signatures_match_the_sqlite_implementation() -> None:
    failures: dict[str, list[str]] = {}
    for member in sorted(_DREAMING_STORE_CAPABILITIES):
        implementation = getattr(sqlite_core.SqliteEngravaCore, member, None)
        if implementation is None:
            failures[member] = ["SqliteEngravaCore does not implement this capability"]
            continue
        protocol_signature = inspect.signature(
            getattr(dreaming_protocols.DreamingStoreProtocol, member)
        )
        impl_signature = inspect.signature(implementation)
        mismatches = _parameter_mismatches(protocol_signature, impl_signature)
        mismatches += _return_annotation_mismatches(member, protocol_signature, impl_signature)
        if mismatches:
            failures[member] = mismatches
    report = "\n".join(f"{member}: {'; '.join(reasons)}" for member, reasons in failures.items())
    assert not report, f"SqliteEngravaCore drifted from DreamingStoreProtocol:\n{report}"


# ------------------------------------------------------------------
# Behavioural parity of the relocated logic
# ------------------------------------------------------------------

_PARITY_CTX = DreamingContext(current_cycle=100, total_thoughts=50)


def _signal_thought(
    *,
    created_cycle: int = 0,
    updated_cycle: int = 0,
    confirmation_count: int = 0,
    confidence: float | None = None,
    access_count: int = 0,
    action_outcome_score: float | None = None,
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id="parity",
        thought_type=ThoughtType.OBSERVATION,
        essence="parity candidate",
        content="parity candidate content",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=created_cycle,
        updated_cycle=updated_cycle,
        source="test",
        confirmation_count=confirmation_count,
        confidence=confidence,
        access_count=access_count,
        action_outcome_score=action_outcome_score,
    )


def test_recency_signal_scores() -> None:
    signal = RecencySignal()
    assert signal(_signal_thought(updated_cycle=100), _PARITY_CTX) == 1.0
    # A thought updated after the run cycle clamps to age zero, never above 1.0.
    assert signal(_signal_thought(updated_cycle=140), _PARITY_CTX) == 1.0
    assert signal(_signal_thought(updated_cycle=0), _PARITY_CTX) == pytest.approx(
        0.36787944117144233
    )
    assert RecencySignal(decay_rate=0.5)(
        _signal_thought(updated_cycle=98), _PARITY_CTX
    ) == pytest.approx(0.36787944117144233)


def test_staleness_signal_scores() -> None:
    signal = StalenessSignal()
    assert signal(_signal_thought(created_cycle=0, updated_cycle=0), _PARITY_CTX) == 0.0
    assert signal(_signal_thought(created_cycle=10, updated_cycle=35), _PARITY_CTX) == 0.25
    assert signal(_signal_thought(created_cycle=0, updated_cycle=100), _PARITY_CTX) == 1.0
    assert signal(_signal_thought(created_cycle=0, updated_cycle=500), _PARITY_CTX) == 1.0
    # A non-positive span is coerced to 1, so it saturates instead of dividing
    # by zero or inverting the score.
    assert StalenessSignal(max_span=0)(_signal_thought(updated_cycle=1), _PARITY_CTX) == 1.0
    assert StalenessSignal(max_span=-10)(_signal_thought(updated_cycle=1), _PARITY_CTX) == 1.0


def test_confirmation_signal_scores() -> None:
    signal = ConfirmationSignal()
    assert signal(_signal_thought(confirmation_count=0), _PARITY_CTX) == 0.0
    assert signal(_signal_thought(confirmation_count=1), _PARITY_CTX) == pytest.approx(0.2)
    assert signal(_signal_thought(confirmation_count=5), _PARITY_CTX) == 1.0
    assert signal(_signal_thought(confirmation_count=9), _PARITY_CTX) == 1.0
    for max_count in (0, -10):
        saturated = ConfirmationSignal(max_count=max_count)
        assert saturated(_signal_thought(confirmation_count=1), _PARITY_CTX) == 1.0


def test_confidence_signal_scores() -> None:
    signal = ConfidenceSignal()
    # A missing confidence is the neutral 0.5; a stored 0.0 stays 0.0.
    assert signal(_signal_thought(confidence=None), _PARITY_CTX) == 0.5
    assert signal(_signal_thought(confidence=0.0), _PARITY_CTX) == 0.0
    assert signal(_signal_thought(confidence=0.25), _PARITY_CTX) == 0.25
    assert signal(_signal_thought(confidence=1.0), _PARITY_CTX) == 1.0


def test_frequency_signal_scores() -> None:
    signal = FrequencySignal()
    assert signal(_signal_thought(access_count=0), _PARITY_CTX) == 0.0
    assert signal(_signal_thought(access_count=4), _PARITY_CTX) == pytest.approx(0.4)
    assert signal(_signal_thought(access_count=10), _PARITY_CTX) == 1.0
    assert signal(_signal_thought(access_count=25), _PARITY_CTX) == 1.0
    for max_accesses in (0, -10):
        saturated = FrequencySignal(max_accesses=max_accesses)
        assert saturated(_signal_thought(access_count=1), _PARITY_CTX) == 1.0


def test_action_outcome_signal_scores() -> None:
    signal = ActionOutcomeSignal()
    # No terminal action means no evidence, which contributes 0.0 rather than 0.5.
    assert signal(_signal_thought(action_outcome_score=None), _PARITY_CTX) == 0.0
    assert signal(_signal_thought(action_outcome_score=0.0), _PARITY_CTX) == 0.0
    assert signal(_signal_thought(action_outcome_score=0.75), _PARITY_CTX) == 0.75
    assert signal(_signal_thought(action_outcome_score=1.0), _PARITY_CTX) == 1.0


def test_cycle_signals_are_active_exactly_when_a_cycle_is_supplied() -> None:
    for name in ("recency", "staleness"):
        assert (
            default_signal_active(name, [], current_cycle=0, access_tracking_enabled=False) is True
        )
        assert (
            default_signal_active(name, [], current_cycle=None, access_tracking_enabled=True)
            is False
        )


def test_data_signals_are_active_when_any_candidate_carries_the_field() -> None:
    plain = [_signal_thought()]
    carriers = {
        "confirmation": _signal_thought(confirmation_count=1),
        # A stored 0.0 is data, so it activates the signal just like any other value.
        "confidence": _signal_thought(confidence=0.0),
        "action_outcome": _signal_thought(action_outcome_score=0.0),
    }
    for name, carrier in carriers.items():
        assert (
            default_signal_active(name, plain, current_cycle=1, access_tracking_enabled=True)
            is False
        )
        # The predicate is pool-relative: one carrier activates the whole run.
        assert (
            default_signal_active(
                name, [*plain, carrier], current_cycle=1, access_tracking_enabled=True
            )
            is True
        )


def test_frequency_signal_needs_both_tracking_and_a_recorded_access() -> None:
    accessed = [_signal_thought(access_count=1)]
    assert (
        default_signal_active("frequency", accessed, current_cycle=1, access_tracking_enabled=False)
        is False
    )
    assert (
        default_signal_active(
            "frequency", [_signal_thought()], current_cycle=1, access_tracking_enabled=True
        )
        is False
    )
    assert (
        default_signal_active("frequency", accessed, current_cycle=1, access_tracking_enabled=True)
        is True
    )


def test_unknown_signal_name_is_rejected() -> None:
    with pytest.raises(KeyError, match="not a default signal"):
        default_signal_active("nope", [], current_cycle=1, access_tracking_enabled=True)


def test_centroid_edge_cases() -> None:
    with pytest.raises(ValueError, match="at least one member vector"):
        compute_centroid([])
    assert compute_centroid([[3.0, 4.0]]) == pytest.approx([0.6, 0.8])
    # Any positive magnitude is rescaled to unit length, shorter ones included.
    assert compute_centroid([[0.3, 0.4]]) == pytest.approx([0.6, 0.8])
    # A zero mean is returned unchanged instead of dividing by zero.
    assert compute_centroid([[0.0, 0.0, 0.0]]) == [0.0, 0.0, 0.0]
    assert compute_centroid([[1.0, 0.0], [-1.0, 0.0]]) == [0.0, 0.0]
    assert compute_centroid([[1.0, 0.0], [1.0, 1.0]]) == pytest.approx(
        [0.8944271909999159, 0.4472135954999579]
    )
    assert CENTROID_MODEL_NAME == "dreaming-centroid"


def test_default_signals_registry_contents_and_order() -> None:
    # Order is part of the contract: it fixes the weighting and reporting order.
    assert list(DEFAULT_SIGNALS) == [
        "recency",
        "staleness",
        "confirmation",
        "confidence",
        "frequency",
        "action_outcome",
    ]
    assert dict(DEFAULT_SIGNALS) == {
        "recency": RecencySignal,
        "staleness": StalenessSignal,
        "confirmation": ConfirmationSignal,
        "confidence": ConfidenceSignal,
        "frequency": FrequencySignal,
        "action_outcome": ActionOutcomeSignal,
    }


def test_consolidation_result_field_defaults() -> None:
    result = ConsolidationResult(candidates_evaluated=7, promoted_count=2)
    assert result.candidates_evaluated == 7
    assert result.promoted_count == 2
    assert result.promoted_ids == []
    assert result.skipped_gate_count == 0
    assert result.scores == {}
    assert result.edges_created == 0
    assert result.reflections_created == 0
    assert result.promotion_capped is False
    assert result.p1_fraction_after == 0.0
    assert result.orphans_retired == 0
    assert result.active_signal_weights == {}
    assert result.flat_signals == []

    # Each run gets its own containers; a shared default would leak across runs.
    other = ConsolidationResult(candidates_evaluated=0, promoted_count=0)
    assert result.promoted_ids is not other.promoted_ids
    assert result.scores is not other.scores
    assert result.active_signal_weights is not other.active_signal_weights
    assert result.flat_signals is not other.flat_signals

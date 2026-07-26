"""Cross-section configuration validation parity.

Equivalent invalid values must be rejected **consistently** across every public
configuration dataclass and across both construction paths — direct dataclass
construction and the YAML loader. Every rejection raises the one typed error
(:class:`engrava.config.ConfigError`), so a ``bool`` in a numeric field, a float
where an integer is required, an out-of-range value, and a malformed (non-mapping)
section all fail identically regardless of how the config is built.

The matrix also pins loader-parity (direct == YAML) and a golden round-trip proving
the tightened validation never rejects a genuinely valid configuration.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest
import yaml

import engrava
import engrava.config
from engrava.config import (
    ConfigError,
    DreamingConfig,
    DreamingGates,
    EdgeCreationConfig,
    EmbeddingConfig,
    EngravaConfig,
    HygienePolicyConfig,
    IngestConfig,
    JournalConfig,
    MetricsConfig,
    SearchConfig,
    ServiceConfig,
    ServicesConfig,
    TTLConfig,
    load_config,
)
from engrava.config_validation import (
    own_config_fields,
    require_mapping,
    require_nonneg_int,
    require_positive_int,
    require_str_collection,
    require_unit_float,
)
from engrava.domain.protocols.derived_records import DeriveGates

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, Sequence


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


#: The arguments a config dataclass cannot be constructed without, so that the
#: single field under test is the only variable. A class with a required
#: argument missing from here fails loudly at construction rather than silently
#: dropping out of any enumeration built on it.
_REQUIRED_ARGS: dict[str, dict[str, object]] = {
    "EngravaConfig": {"database_path": Path("./t.db")},
    "ServicesConfig": {"data_dir": Path("./data")},
    "EngravaCLIConfig": {"db_path": Path("./engrava.db")},
}


def _construct(cls: type, field: str, value: object) -> object:
    """Construct ``cls`` with ``field`` set to ``value`` (all else defaulted)."""
    kwargs: dict[str, object] = dict(_REQUIRED_ARGS.get(cls.__name__, {}))
    kwargs[field] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def _build_doc(path: Sequence[str], value: object) -> dict[str, object]:
    """Return a full config document with ``value`` placed at nested ``path``."""
    doc: dict[str, object] = {"database": {"path": "./t.db"}}
    node: dict[str, object] = doc
    for key in path[:-1]:
        child = node.setdefault(key, {})
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = value
    return doc


def _load(tmp_path: Path, doc: dict[str, object]) -> EngravaConfig:
    """Serialise ``doc`` to YAML on disk and load it through the public loader."""
    cfg_file = tmp_path / "engrava.yaml"
    cfg_file.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return load_config(cfg_file)


class NumericFieldCase(NamedTuple):
    """One numeric field addressed by both construction paths."""

    id: str
    cls: type
    direct_field: str
    yaml_path: tuple[str, ...]
    is_int: bool
    out_of_range: float


# Representative numeric fields across every public dataclass that owns one. Each
# names the same concept on both paths (direct dataclass field + YAML key path).
NUMERIC_CASES: list[NumericFieldCase] = [
    NumericFieldCase(
        "engrava.embedding_dimension",
        EngravaConfig,
        "embedding_dimension",
        ("extensions", "vector", "dimension"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "search.recency_half_life",
        SearchConfig,
        "recency_half_life",
        ("search", "recency_half_life"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "search.default_fts_weight",
        SearchConfig,
        "default_fts_weight",
        ("search", "default_fts_weight"),
        is_int=False,
        out_of_range=-0.1,
    ),
    NumericFieldCase(
        "ttl.check_every_n_operations",
        TTLConfig,
        "check_every_n_operations",
        ("ttl", "check_every_n_operations"),
        is_int=True,
        out_of_range=-1,
    ),
    NumericFieldCase(
        "ttl.default_ttl_seconds",
        TTLConfig,
        "default_ttl_seconds",
        ("ttl", "default_ttl_seconds"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "metrics.window_size",
        MetricsConfig,
        "window_size",
        ("metrics", "window_size"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "embeddings.batch_size",
        EmbeddingConfig,
        "batch_size",
        ("embeddings", "batch_size"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "hygiene.check_every_n_cycles",
        HygienePolicyConfig,
        "check_every_n_cycles",
        ("hygiene_policy", "check_every_n_cycles"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "hygiene.gc_restore_window_seconds",
        HygienePolicyConfig,
        "gc_restore_window_seconds",
        ("hygiene_policy", "gc_restore_window_seconds"),
        is_int=True,
        out_of_range=-1,
    ),
    NumericFieldCase(
        "hygiene.eviction_threshold",
        HygienePolicyConfig,
        "eviction_threshold",
        ("hygiene_policy", "eviction_threshold"),
        is_int=False,
        out_of_range=2.0,
    ),
    NumericFieldCase(
        "gates.max_promoted_per_run",
        DreamingGates,
        "max_promoted_per_run",
        ("extensions", "dreaming", "gates", "max_promoted_per_run"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "gates.cluster_similarity_threshold",
        DreamingGates,
        "cluster_similarity_threshold",
        ("extensions", "dreaming", "gates", "cluster_similarity_threshold"),
        is_int=False,
        out_of_range=2.0,
    ),
    NumericFieldCase(
        "dreaming.schedule_every_n_cycles",
        DreamingConfig,
        "schedule_every_n_cycles",
        ("extensions", "dreaming", "schedule_every_n_cycles"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "dreaming.promote_threshold",
        DreamingConfig,
        "promote_threshold",
        ("extensions", "dreaming", "promote_threshold"),
        is_int=False,
        out_of_range=2.0,
    ),
    NumericFieldCase(
        "edges.top_k",
        EdgeCreationConfig,
        "top_k",
        ("extensions", "dreaming", "edges", "top_k"),
        is_int=True,
        out_of_range=0,
    ),
    NumericFieldCase(
        "edges.min_similarity",
        EdgeCreationConfig,
        "min_similarity",
        ("extensions", "dreaming", "edges", "min_similarity"),
        is_int=False,
        out_of_range=2.0,
    ),
    # ``reflection_topk_cap`` is a [0.0, 1.0] fraction; 1.5 must be rejected
    # (it was wrongly accepted while validated as a plain non-negative float).
    NumericFieldCase(
        "search.reflection_topk_cap",
        SearchConfig,
        "reflection_topk_cap",
        ("search", "reflection_topk_cap"),
        is_int=False,
        out_of_range=1.5,
    ),
]

_NUMERIC_IDS = [case.id for case in NUMERIC_CASES]


class TestNumericFieldParity:
    """Every numeric field rejects the same invalid values on both paths."""

    @pytest.mark.parametrize("case", NUMERIC_CASES, ids=_NUMERIC_IDS)
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_bool_rejected_direct(self, case: NumericFieldCase, *, bool_value: bool) -> None:
        # ``bool`` is an ``int`` subclass; it must never impersonate 1/0 in a
        # numeric field.
        with pytest.raises(ConfigError):
            _construct(case.cls, case.direct_field, bool_value)

    @pytest.mark.parametrize("case", NUMERIC_CASES, ids=_NUMERIC_IDS)
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_bool_rejected_yaml(
        self,
        tmp_path: Path,
        case: NumericFieldCase,
        *,
        bool_value: bool,
    ) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(case.yaml_path, bool_value))

    @pytest.mark.parametrize("case", NUMERIC_CASES, ids=_NUMERIC_IDS)
    def test_out_of_range_rejected_direct(self, case: NumericFieldCase) -> None:
        with pytest.raises(ConfigError):
            _construct(case.cls, case.direct_field, case.out_of_range)

    @pytest.mark.parametrize("case", NUMERIC_CASES, ids=_NUMERIC_IDS)
    def test_out_of_range_rejected_yaml(self, tmp_path: Path, case: NumericFieldCase) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(case.yaml_path, case.out_of_range))

    @pytest.mark.parametrize(
        "case",
        [c for c in NUMERIC_CASES if c.is_int],
        ids=[c.id for c in NUMERIC_CASES if c.is_int],
    )
    def test_float_in_int_rejected_direct(self, case: NumericFieldCase) -> None:
        with pytest.raises(ConfigError):
            _construct(case.cls, case.direct_field, 1.5)

    @pytest.mark.parametrize(
        "case",
        [c for c in NUMERIC_CASES if c.is_int],
        ids=[c.id for c in NUMERIC_CASES if c.is_int],
    )
    def test_float_in_int_rejected_yaml(self, tmp_path: Path, case: NumericFieldCase) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(case.yaml_path, 1.5))


class NonMappingCase(NamedTuple):
    """A section that must reject a non-mapping value."""

    id: str
    yaml_path: tuple[str, ...]
    match: str


# Every YAML section must fail loudly when it is not a mapping — including
# ``extensions.vector`` and ``hooks``, which previously retained defaults silently.
NON_MAPPING_CASES: list[NonMappingCase] = [
    NonMappingCase("database", ("database",), r"database.*mapping"),
    NonMappingCase("extensions", ("extensions",), r"extensions.*mapping"),
    NonMappingCase("extensions.vector", ("extensions", "vector"), r"vector.*mapping"),
    NonMappingCase("hooks", ("hooks",), r"hooks.*mapping"),
    NonMappingCase("search", ("search",), r"search.*mapping"),
    NonMappingCase("ttl", ("ttl",), r"ttl.*mapping"),
    NonMappingCase("metrics", ("metrics",), r"metrics.*mapping"),
    NonMappingCase("embeddings", ("embeddings",), r"embeddings.*mapping"),
    NonMappingCase("ingest", ("ingest",), r"ingest.*mapping"),
    NonMappingCase("derive", ("derive",), r"derive.*mapping"),
    NonMappingCase("journal", ("journal",), r"journal.*mapping"),
    NonMappingCase("services", ("services",), r"services.*mapping"),
    NonMappingCase("hygiene_policy", ("hygiene_policy",), r"hygiene_policy.*mapping"),
    NonMappingCase("dreaming", ("extensions", "dreaming"), r"dreaming.*mapping"),
    NonMappingCase("gates", ("extensions", "dreaming", "gates"), r"gates.*mapping"),
    NonMappingCase("edges", ("extensions", "dreaming", "edges"), r"edges.*mapping"),
    NonMappingCase("manifests", ("manifests",), r"manifests"),
]

_NON_MAPPING_IDS = [case.id for case in NON_MAPPING_CASES]


class TestNonMappingSectionParity:
    """A non-mapping section raises ``ConfigError`` — no silent default retention."""

    @pytest.mark.parametrize("case", NON_MAPPING_CASES, ids=_NON_MAPPING_IDS)
    def test_non_mapping_section_rejected(self, tmp_path: Path, case: NonMappingCase) -> None:
        doc = _build_doc(case.yaml_path, "not-a-mapping")
        with pytest.raises(ConfigError, match=case.match):
            _load(tmp_path, doc)


class TestBooleanFieldParity:
    """Boolean-only dataclasses reject a non-boolean on both paths."""

    def test_journal_direct(self) -> None:
        with pytest.raises(ConfigError, match="enabled"):
            JournalConfig(enabled="yes")  # type: ignore[arg-type]

    def test_journal_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="enabled"):
            _load(tmp_path, _build_doc(("journal", "enabled"), "yes"))

    def test_ingest_direct(self) -> None:
        with pytest.raises(ConfigError, match="deduplication_enabled"):
            IngestConfig(deduplication_enabled=1)  # type: ignore[arg-type]

    def test_ingest_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="deduplication_enabled"):
            _load(tmp_path, _build_doc(("ingest", "deduplication_enabled"), 1))


class TestConfigErrorIsValueError:
    """``ConfigError`` is a ``ValueError`` so the historical catch still works."""

    def test_subclass(self) -> None:
        assert issubclass(ConfigError, ValueError)

    def test_direct_construction_caught_as_value_error(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            MetricsConfig(window_size=0)

    def test_yaml_caught_as_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="window_size"):
            _load(tmp_path, _build_doc(("metrics", "window_size"), 0))


class CollectionCase(NamedTuple):
    """A string-collection or mapping field addressed by both construction paths."""

    id: str
    cls: type
    direct_field: str
    yaml_path: tuple[str, ...]


# Fields that take a list/tuple/set of strings. A bare ``str`` iterates into
# characters and must be rejected on both paths (the direct path previously
# accepted it silently).
STR_COLLECTION_CASES: list[CollectionCase] = [
    CollectionCase(
        "gates.cluster_allowed_types",
        DreamingGates,
        "cluster_allowed_types",
        ("extensions", "dreaming", "gates", "cluster_allowed_types"),
    ),
    CollectionCase(
        "hygiene.protected_priorities",
        HygienePolicyConfig,
        "protected_priorities",
        ("hygiene_policy", "protected_priorities"),
    ),
    CollectionCase(
        "dreaming.excluded_content_types",
        DreamingConfig,
        "excluded_content_types",
        ("extensions", "dreaming", "excluded_content_types"),
    ),
    CollectionCase(
        "dreaming.eligible_content_types",
        DreamingConfig,
        "eligible_content_types",
        ("extensions", "dreaming", "eligible_content_types"),
    ),
]

# Mapping fields. A non-mapping container must raise ``ConfigError``, never an
# ``AttributeError`` from calling ``.items()`` on a non-mapping.
MAPPING_CONTAINER_CASES: list[CollectionCase] = [
    CollectionCase(
        "dreaming.signals",
        DreamingConfig,
        "signals",
        ("extensions", "dreaming", "signals"),
    ),
    CollectionCase(
        "hygiene.signal_weights",
        HygienePolicyConfig,
        "signal_weights",
        ("hygiene_policy", "signal_weights"),
    ),
]


class TestCollectionShapeParity:
    """String-collection and mapping fields reject bad container shapes on both paths."""

    @pytest.mark.parametrize(
        "case",
        STR_COLLECTION_CASES,
        ids=[c.id for c in STR_COLLECTION_CASES],
    )
    def test_bare_string_rejected_direct(self, case: CollectionCase) -> None:
        with pytest.raises(ConfigError):
            _construct(case.cls, case.direct_field, "single")

    @pytest.mark.parametrize(
        "case",
        STR_COLLECTION_CASES,
        ids=[c.id for c in STR_COLLECTION_CASES],
    )
    def test_bare_string_rejected_yaml(self, tmp_path: Path, case: CollectionCase) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(case.yaml_path, "single"))

    @pytest.mark.parametrize(
        "case",
        MAPPING_CONTAINER_CASES,
        ids=[c.id for c in MAPPING_CONTAINER_CASES],
    )
    def test_non_mapping_container_rejected_direct(self, case: CollectionCase) -> None:
        # Must be a typed ConfigError, not an AttributeError from ``.items()``.
        with pytest.raises(ConfigError):
            _construct(case.cls, case.direct_field, 123)

    @pytest.mark.parametrize(
        "case",
        MAPPING_CONTAINER_CASES,
        ids=[c.id for c in MAPPING_CONTAINER_CASES],
    )
    def test_non_mapping_container_rejected_yaml(
        self,
        tmp_path: Path,
        case: CollectionCase,
    ) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(case.yaml_path, 123))


class TestFiniteFloatParity:
    """Non-finite floats (``NaN`` / ``inf``) are rejected on both paths."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonneg_float_direct(self, bad: float) -> None:
        with pytest.raises(ConfigError):
            SearchConfig(default_fts_weight=bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonneg_float_yaml(self, tmp_path: Path, bad: float) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(("search", "default_fts_weight"), bad))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_unit_float_direct(self, bad: float) -> None:
        with pytest.raises(ConfigError):
            HygienePolicyConfig(eviction_threshold=bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_unit_float_yaml(self, tmp_path: Path, bad: float) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(("hygiene_policy", "eviction_threshold"), bad))


class TestNonConfigErrorEscapes:
    """Paths that once escaped as ``TypeError`` / ``AttributeError`` now raise ``ConfigError``."""

    def test_derive_gates_int_enabled(self) -> None:
        with pytest.raises(ConfigError, match="enabled"):
            DeriveGates(enabled=1)  # type: ignore[arg-type]

    def test_embedding_unhashable_provider(self) -> None:
        # A list is unhashable; ``x in frozenset`` would raise TypeError without
        # the isinstance short-circuit.
        with pytest.raises(ConfigError, match="provider"):
            EmbeddingConfig(provider=[])  # type: ignore[arg-type]

    def test_services_non_string_default_service(self) -> None:
        # A non-string would reach ``re.match`` and raise TypeError without the guard.
        with pytest.raises(ConfigError, match="default_service"):
            ServicesConfig(data_dir=Path("./d"), default_service=1)  # type: ignore[arg-type]

    def test_dreaming_non_mapping_signals(self) -> None:
        with pytest.raises(ConfigError, match="signals"):
            DreamingConfig(signals=123)  # type: ignore[arg-type]


class TestNewlyValidatedDataclasses:
    """Dataclasses that gained direct-construction validation reject bad values.

    The nested-type and ``Path`` checks have no YAML analogue — the loader
    structurally produces the right types — so these assert the direct path alone.
    """

    def test_service_config_bad_embeddings_direct(self) -> None:
        with pytest.raises(ConfigError, match="embeddings"):
            ServiceConfig(embeddings=123)  # type: ignore[arg-type]

    def test_service_config_bad_embeddings_yaml(self, tmp_path: Path) -> None:
        doc = {
            "database": {"path": "./t.db"},
            "services": {"data_dir": "./d", "configs": {"svc": {"embeddings": 123}}},
        }
        with pytest.raises(ConfigError):
            _load(tmp_path, doc)

    def test_services_non_path_data_dir(self) -> None:
        with pytest.raises(ConfigError, match="data_dir"):
            ServicesConfig(data_dir="./d")  # type: ignore[arg-type]

    def test_services_bad_config_value(self) -> None:
        with pytest.raises(ConfigError, match="configs"):
            ServicesConfig(data_dir=Path("./d"), configs={"main": 123})  # type: ignore[dict-item]

    def test_engrava_non_path_database(self) -> None:
        with pytest.raises(ConfigError, match="database_path"):
            EngravaConfig(database_path="./t.db")  # type: ignore[arg-type]

    def test_engrava_bad_nested_type(self) -> None:
        with pytest.raises(ConfigError, match="search"):
            EngravaConfig(database_path=Path("./t.db"), search=123)  # type: ignore[arg-type]

    def test_engrava_bad_manifest_paths(self) -> None:
        with pytest.raises(ConfigError, match="extension_manifest_paths"):
            EngravaConfig(
                database_path=Path("./t.db"),
                extension_manifest_paths="mod:ATTR",  # type: ignore[arg-type]
            )


class TestSignalWeightFiniteness:
    """Custom signal weights must be finite numbers on both paths."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10**1000])
    def test_dreaming_signal_weight_direct(self, bad: float) -> None:
        with pytest.raises(ConfigError):
            DreamingConfig(signals={"recency": bad})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_dreaming_signal_weight_yaml(self, tmp_path: Path, bad: float) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(("extensions", "dreaming", "signals", "recency"), bad))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10**1000])
    def test_hygiene_signal_weight_direct(self, bad: float) -> None:
        with pytest.raises(ConfigError):
            HygienePolicyConfig(signal_weights={"recency": bad})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_hygiene_signal_weight_yaml(self, tmp_path: Path, bad: float) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(("hygiene_policy", "signal_weights", "recency"), bad))

    # ``1e308`` (float) sum overflows to +inf; ``10**308`` (int) sum overflows the
    # float range so ``math.isfinite`` itself would raise OverflowError — both must
    # normalise to ConfigError rather than silently zero the weights downstream.
    @pytest.mark.parametrize("weight", [1e308, 10**308])
    def test_dreaming_signal_sum_overflow_direct(self, weight: float) -> None:
        with pytest.raises(ConfigError):
            DreamingConfig(signals={"a": weight, "b": weight})

    @pytest.mark.parametrize("weight", [1e308, 10**308])
    def test_dreaming_signal_sum_overflow_yaml(self, tmp_path: Path, weight: float) -> None:
        doc = _build_doc(("extensions", "dreaming", "signals"), {"a": weight, "b": weight})
        with pytest.raises(ConfigError):
            _load(tmp_path, doc)

    @pytest.mark.parametrize("weight", [1e308, 10**308])
    def test_hygiene_signal_sum_overflow_direct(self, weight: float) -> None:
        with pytest.raises(ConfigError):
            HygienePolicyConfig(signal_weights={"a": weight, "b": weight})

    @pytest.mark.parametrize("weight", [1e308, 10**308])
    def test_hygiene_signal_sum_overflow_yaml(self, tmp_path: Path, weight: float) -> None:
        doc = _build_doc(("hygiene_policy", "signal_weights"), {"a": weight, "b": weight})
        with pytest.raises(ConfigError):
            _load(tmp_path, doc)


class TestHugeIntParity:
    """An integer too large to be a finite float raises ConfigError (not OverflowError)."""

    _HUGE = 10**1000

    def test_unit_float_direct(self) -> None:
        with pytest.raises(ConfigError):
            HygienePolicyConfig(eviction_threshold=self._HUGE)

    def test_unit_float_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(("hygiene_policy", "eviction_threshold"), self._HUGE))

    def test_nonneg_float_direct(self) -> None:
        with pytest.raises(ConfigError):
            SearchConfig(default_fts_weight=self._HUGE)

    def test_nonneg_float_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            _load(tmp_path, _build_doc(("search", "default_fts_weight"), self._HUGE))


class TestDreamingNestedTypeParity:
    """DreamingConfig rejects ill-typed nested configs (direct path; YAML is always typed)."""

    def test_bad_gates(self) -> None:
        with pytest.raises(ConfigError, match="gates"):
            DreamingConfig(gates=1)  # type: ignore[arg-type]

    def test_bad_edges(self) -> None:
        with pytest.raises(ConfigError, match="edges"):
            DreamingConfig(edges=object())  # type: ignore[arg-type]


class TestYamlTypeErrorNormalized:
    """Malformed YAML that once leaked ``TypeError`` now raises ``ConfigError``."""

    def test_embeddings_provider_list(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="provider"):
            _load(tmp_path, _build_doc(("embeddings", "provider"), []))

    def test_ttl_strategy_list(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="strategy"):
            _load(tmp_path, _build_doc(("ttl", "strategy"), []))

    def test_eligible_perspectives_nested_list(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="eligible_perspectives"):
            _load(
                tmp_path,
                _build_doc(("extensions", "dreaming", "eligible_perspectives"), [[]]),
            )

    def test_database_path_list(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"database\.path"):
            _load(tmp_path, _build_doc(("database", "path"), ["x"]))


_GOLDEN_DOC: dict[str, object] = {
    "database": {"path": "./golden.db", "wal_mode": False},
    "extensions": {
        "vector": {"backend": "sqlite-vec", "dimension": 512},
        "dreaming": {
            "enabled": True,
            "schedule_every_n_cycles": 50,
            "promote_threshold": 0.5,
            "candidates_limit": 100,
            "gates": {
                "min_confirmations": 3,
                "max_promoted_per_run": 10,
                "cluster_similarity_threshold": 0.8,
            },
            "edges": {"top_k": 2, "min_similarity": 0.6, "edge_weight_factor": 0.4},
        },
    },
    "hygiene_policy": {
        "enabled": True,
        "eviction_threshold": 0.15,
        "check_every_n_cycles": 2,
        "gc_restore_window_seconds": 3600,
    },
    "search": {"default_fts_weight": 0.4, "recency_half_life": 60},
    "ttl": {"strategy": "delete", "check_every_n_operations": 5},
    "metrics": {"window_size": 500},
    "journal": {"enabled": True},
}


class TestGoldenRoundTrip:
    """A valid configuration is accepted unchanged on both paths (no false reject)."""

    def test_yaml_golden(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, _GOLDEN_DOC)
        assert cfg.database_path == Path("./golden.db")
        assert cfg.wal_mode is False
        assert cfg.vector_backend == "sqlite-vec"
        assert cfg.embedding_dimension == 512
        assert cfg.dreaming is not None
        assert cfg.dreaming.enabled is True
        assert cfg.dreaming.schedule_every_n_cycles == 50
        assert cfg.dreaming.promote_threshold == 0.5
        assert cfg.dreaming.gates.max_promoted_per_run == 10
        assert cfg.dreaming.gates.cluster_similarity_threshold == 0.8
        assert cfg.dreaming.edges.top_k == 2
        assert cfg.hygiene_policy is not None
        assert cfg.hygiene_policy.eviction_threshold == 0.15
        assert cfg.hygiene_policy.gc_restore_window_seconds == 3600
        assert cfg.search.default_fts_weight == 0.4
        assert cfg.search.recency_half_life == 60
        assert cfg.ttl.strategy == "delete"
        assert cfg.ttl.check_every_n_operations == 5
        assert cfg.metrics.window_size == 500
        assert cfg.journal.enabled is True

    def test_direct_golden(self) -> None:
        # The same valid values construct cleanly by hand — direct construction
        # enforces the invariants without rejecting anything the loader accepts.
        cfg = EngravaConfig(
            database_path=Path("./golden.db"),
            wal_mode=False,
            vector_backend="sqlite-vec",
            embedding_dimension=512,
            dreaming=DreamingConfig(
                enabled=True,
                schedule_every_n_cycles=50,
                promote_threshold=0.5,
                candidates_limit=100,
                gates=DreamingGates(
                    min_confirmations=3,
                    max_promoted_per_run=10,
                    cluster_similarity_threshold=0.8,
                ),
                edges=EdgeCreationConfig(top_k=2, min_similarity=0.6, edge_weight_factor=0.4),
            ),
            hygiene_policy=HygienePolicyConfig(
                enabled=True,
                eviction_threshold=0.15,
                check_every_n_cycles=2,
                gc_restore_window_seconds=3600,
            ),
            search=SearchConfig(default_fts_weight=0.4, recency_half_life=60),
            ttl=TTLConfig(strategy="delete", check_every_n_operations=5),
            metrics=MetricsConfig(window_size=500),
            journal=JournalConfig(enabled=True),
        )
        assert cfg.embedding_dimension == 512
        assert cfg.dreaming is not None
        assert cfg.dreaming.gates.max_promoted_per_run == 10
        assert cfg.hygiene_policy is not None
        assert cfg.hygiene_policy.gc_restore_window_seconds == 3600


# ---------------------------------------------------------------------------
# Every decoded value is a value the module owns
# ---------------------------------------------------------------------------


class _Entry(str):
    """A ``str`` subclass — genuinely a string, and free to answer for itself."""

    __slots__ = ()


class _Number(int):
    """An ``int`` subclass — genuinely an integer, and free to answer for itself."""


class TestDecodedValuesAreOwned:
    """A decoder hands back a value with no behaviour of the caller's in it.

    ``isinstance`` is the wrong tool for this and cannot be made into the right
    one: a subclass genuinely *is* an instance. It establishes that a value has
    real text or a real number behind it; what the value will answer when it is
    later formatted into a path, interpolated into DDL, or asked ``in`` is a
    separate question, and the only answer that holds is to stop consulting the
    caller's object once it has been checked.
    """

    @pytest.mark.parametrize(
        "case",
        STR_COLLECTION_CASES,
        ids=[c.id for c in STR_COLLECTION_CASES],
    )
    def test_string_collection_fields_hold_exact_strings(self, case: CollectionCase) -> None:
        """The container type each of these fields normalises to, entry by entry.

        This shares the hand-written matrix used by the shape checks above, and
        a hand-written matrix cannot be trusted to be complete — it omitted two
        string-collection fields when it was written. Completeness is the job of
        ``TestEveryConfigFieldIsOwned`` below, which enumerates the dataclasses
        themselves; what this adds on top is the *declared container type*,
        which that test deliberately does not assert.
        """
        config = _construct(case.cls, case.direct_field, [_Entry("code")])
        stored = getattr(config, case.direct_field)
        assert type(stored) in (tuple, frozenset)
        assert [type(entry) for entry in stored] == [str]
        assert set(stored) == {"code"}

    def test_a_collection_is_read_from_its_real_storage(self) -> None:
        """The caller's ``__iter__`` is not consulted at all, not merely once.

        Reading it once would still let a container present entries that are
        not in it. Reading the built-in's own iterator over the real storage
        means the entries decoded are the entries the container holds.
        """

        class _EntriesThatIterateDifferently(list):  # type: ignore[type-arg]  # the adversary is the subclassing itself
            def __init__(self) -> None:
                super().__init__(["real"])
                self.iter_calls = 0

            def __iter__(self) -> Iterator[str]:
                self.iter_calls += 1
                return iter(["smuggled"])

        entries = _EntriesThatIterateDifferently()
        assert require_str_collection(entries, "label") == ("real",)
        assert entries.iter_calls == 0

    def test_positive_int_decodes_to_an_exact_int(self) -> None:
        decoded = require_positive_int(_Number(384), "label")
        assert type(decoded) is int
        assert decoded == 384

    def test_nonneg_int_decodes_to_an_exact_int(self) -> None:
        decoded = require_nonneg_int(_Number(0), "label")
        assert type(decoded) is int
        assert decoded == 0

    def test_the_embedding_dimension_field_holds_an_exact_int(self) -> None:
        """The one integer field whose value reaches a string it cannot be bound into.

        Every other integer in the configuration ends up as a ``?`` parameter or
        in arithmetic, both of which read the real machine value. This one is
        interpolated into the ``vec0`` virtual-table declaration, where the
        rendering is whatever ``__format__`` decides to return.
        """
        config = _construct(EngravaConfig, "embedding_dimension", _Number(512))
        assert type(config.embedding_dimension) is int  # type: ignore[attr-defined]  # attribute set on a module built for this test
        assert config.embedding_dimension == 512  # type: ignore[attr-defined]  # attribute set on a module built for this test


# ---------------------------------------------------------------------------
# Every field of every configuration object, enumerated from the dataclasses
# ---------------------------------------------------------------------------

#: Types a configuration field may hold after construction. Every one of them is
#: a built-in whose behaviour the caller cannot redefine, which is the whole
#: property: once a field holds one of these, no later ``__format__`` /
#: ``__contains__`` / ``rsplit`` / ``__eq__`` on it can run the caller's code.
_OWNED_TYPES: frozenset[type] = frozenset(
    {bool, int, float, str, tuple, list, set, frozenset, dict, type(None)}
)

#: Fields whose value is not a built-in scalar or container, and what each one
#: must be instead. A field is either decoded into an owned value or it is named
#: here with the category it belongs to; there is no third option, so a field
#: added to any config in future fails this test until someone decides which.
#: Nothing is skipped - every entry here is still asserted, just differently.
_NESTED_CONFIG = "nested-config: it owns its own fields in its own __post_init__"
_UNOWNED_PATH = (
    "path: no validated property exists for a subclass to contradict, because any "
    "path is a legal path. A caller that lies about the directory it declared is "
    "lying to itself about a value it was free to set to that directory outright"
)
_NON_BUILTIN_FIELDS: dict[tuple[str, str], str] = {
    ("EngravaConfig", "database_path"): _UNOWNED_PATH,
    ("ServicesConfig", "data_dir"): _UNOWNED_PATH,
    ("EngravaCLIConfig", "db_path"): _UNOWNED_PATH,
    ("EngravaCLIConfig", "config_path"): _UNOWNED_PATH,
    ("EngravaConfig", "dreaming"): _NESTED_CONFIG,
    ("EngravaConfig", "hygiene_policy"): _NESTED_CONFIG,
    ("EngravaConfig", "embeddings"): _NESTED_CONFIG,
    ("EngravaConfig", "services"): _NESTED_CONFIG,
    ("EngravaConfig", "search"): _NESTED_CONFIG,
    ("EngravaConfig", "journal"): _NESTED_CONFIG,
    ("EngravaConfig", "ttl"): _NESTED_CONFIG,
    ("EngravaConfig", "metrics"): _NESTED_CONFIG,
    ("EngravaConfig", "ingest"): _NESTED_CONFIG,
    ("EngravaConfig", "derive"): _NESTED_CONFIG,
    ("DreamingConfig", "gates"): _NESTED_CONFIG,
    ("DreamingConfig", "edges"): _NESTED_CONFIG,
    ("ServiceConfig", "embeddings"): _NESTED_CONFIG,
}

#: A valid sample for every field whose default is ``None``, so that the field
#: can be probed at all. Completeness is asserted, not assumed.
_NONE_DEFAULT_SAMPLES: dict[tuple[str, str], object] = {
    ("DreamingConfig", "eligible_perspectives"): frozenset({"percept"}),
    ("DreamingConfig", "eligible_content_types"): frozenset({"prose"}),
    ("DreamingGates", "max_cluster_size"): 4,
    ("EmbeddingConfig", "provider"): "ollama",
    ("EmbeddingConfig", "model"): "a-model",
    ("EmbeddingConfig", "base_url"): "http://localhost:1234",
    ("EmbeddingConfig", "api_key"): "a-key",
    ("EmbeddingConfig", "query_prefix"): "q: ",
    ("EmbeddingConfig", "document_prefix"): "d: ",
    ("EngravaConfig", "hooks_class"): "engrava.domain.protocols.hooks.DefaultEngravaHooks",
    ("TTLConfig", "default_ttl_seconds"): 60,
}


class _AdversarialInt(int):
    __slots__ = ()


class _AdversarialFloat(float):
    __slots__ = ()


class _AdversarialStr(str):
    __slots__ = ()


class _AdversarialTuple(tuple):  # type: ignore[type-arg]  # a bare built-in base is exactly what the adversary subclasses
    __slots__ = ()


class _AdversarialList(list):  # type: ignore[type-arg]  # a bare built-in base is exactly what the adversary subclasses
    pass


class _AdversarialSet(set):  # type: ignore[type-arg]  # a bare built-in base is exactly what the adversary subclasses
    pass


class _AdversarialFrozenSet(frozenset):  # type: ignore[type-arg]  # a bare built-in base is exactly what the adversary subclasses
    __slots__ = ()


class _AdversarialDict(dict):  # type: ignore[type-arg]  # a bare built-in base is exactly what the adversary subclasses
    pass


#: The subclass to wrap a value in, by the built-in it is an instance of. Order
#: matters: ``bool`` is an ``int``, and CPython forbids subclassing ``bool``, so
#: a boolean is deliberately absent and left alone.
_ADVERSARIAL_SCALARS: tuple[tuple[type, type], ...] = (
    (int, _AdversarialInt),
    (float, _AdversarialFloat),
    (str, _AdversarialStr),
)
_ADVERSARIAL_CONTAINERS: tuple[tuple[type, type], ...] = (
    (tuple, _AdversarialTuple),
    (list, _AdversarialList),
    (frozenset, _AdversarialFrozenSet),
    (set, _AdversarialSet),
)


def _subclass_of(value: object) -> object:
    """Return *value* re-wrapped in a subclass of its own type, one level deep.

    A subclass is the whole difficulty: it passes every ``isinstance`` check the
    configuration performs while remaining free to answer for itself afterwards.
    """
    if isinstance(value, bool):
        return value
    for builtin, adversarial in _ADVERSARIAL_SCALARS:
        if isinstance(value, builtin):
            return adversarial(value)
    if isinstance(value, dict):
        return _AdversarialDict(
            {_subclass_of(key): _subclass_of(entry) for key, entry in value.items()}
        )
    for builtin, adversarial in _ADVERSARIAL_CONTAINERS:
        if isinstance(value, builtin):
            return adversarial(_subclass_of(entry) for entry in value)
    return value


#: Module prefixes whose dataclasses are not configuration, with the reason.
#:
#: **These auto-classify.** A configuration dataclass added under one of these
#: prefixes is silently accepted as non-configuration and never reviewed, so
#: the guarantee below is narrower than "a config class added anywhere is
#: caught": it is "a config class added anywhere *except* under these five
#: prefixes is caught". They were chosen because each names a namespace whose
#: contents are records by definition — benchmark inputs and results, domain
#: records, parsed query nodes, snapshot rows — where a settings object would
#: be misplaced on its own terms. ``engrava.infrastructure.*`` was deliberately
#: *not* left as a prefix, because retained configuration could plausibly live
#: there; its five dataclasses are listed individually below instead.
_NON_CONFIG_PREFIXES: tuple[tuple[str, str], ...] = (
    ("engrava.benchmarks.", "benchmark inputs and results, never retained to steer a store"),
    ("engrava.domain.models.", "domain records: what the store returns, not what configures it"),
    ("engrava.domain.dreaming", "consolidation inputs and results, not settings"),
    ("engrava.mindql.", "parsed query nodes, guarded where they execute"),
    ("engrava.cli.snapshot_records.", "snapshot rows: a file format, not configuration"),
)

#: Dataclasses outside those prefixes that are still not configuration.
_NON_CONFIG_CLASSES: dict[str, str] = {
    "engrava.domain.manifest.ExtensionManifest": (
        "an extension's own declaration, type-checked where it is loaded"
    ),
    "engrava.domain.protocols.derived_records.DeriveContext": "per-call producer context",
    "engrava.domain.protocols.derived_records.DeriveResult": "the outcome of a derivation",
    "engrava.domain.protocols.derived_records.DerivedRecord": "a record a producer emits",
    "engrava.domain.protocols.hooks.MindQLExtension": "an extension registration",
    "engrava.domain.protocols.hooks.ScoringContext": "per-query scoring inputs",
    "engrava.infrastructure.sqlite.engrava_core._DerivationOutcome": "one derivation's outcome",
    "engrava.infrastructure.sqlite.extension_migrations._AppliedMigration": (
        "a migration ledger row"
    ),
    "engrava.infrastructure.sqlite.extension_migrations._PreparedMigration": "a migration to run",
    "engrava.infrastructure.sqlite.hygiene.EvictionReason": "why one thought was evicted",
    "engrava.infrastructure.sqlite.hygiene.HygieneResult": "the outcome of a hygiene run",
}

#: The configuration classes: constructed from user-supplied settings and
#: retained to steer later behaviour. Every one owns its fields on construction.
_CONFIG_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "engrava.cli.config.EngravaCLIConfig",
        "engrava.config.DreamingConfig",
        "engrava.config.DreamingGates",
        "engrava.config.EdgeCreationConfig",
        "engrava.config.EmbeddingConfig",
        "engrava.config.EngravaConfig",
        "engrava.config.HygienePolicyConfig",
        "engrava.config.IngestConfig",
        "engrava.config.JournalConfig",
        "engrava.config.MetricsConfig",
        "engrava.config.SearchConfig",
        "engrava.config.ServiceConfig",
        "engrava.config.ServicesConfig",
        "engrava.config.TTLConfig",
        "engrava.domain.protocols.derived_records.DeriveGates",
    }
)


def _discovered_dataclasses() -> dict[str, type]:
    """Every dataclass in the installed package, found by walking it.

    Enumerating one module is how the previous version of this test missed two
    configuration classes that live elsewhere. This walks the whole package, so
    a configuration class added in any module is discovered the moment it
    exists — and then has to be classified, because the partition below is
    asserted to be exact.

    ``walk_packages`` enumerates a package's *sub*modules and never the package
    module itself, so ``engrava`` is inspected explicitly: a dataclass defined
    in ``engrava/__init__.py`` sits outside every exclusion prefix and would
    otherwise be invisible to the whole partition.

    Returns:
        Fully-qualified name to class, for every dataclass under ``engrava``.

    """
    found: dict[str, type] = {}
    modules = [engrava]
    modules += [
        importlib.import_module(module.name)
        for module in pkgutil.walk_packages(engrava.__path__, prefix="engrava.")
    ]
    for imported in modules:
        for obj in vars(imported).values():
            if (
                inspect.isclass(obj)
                and dataclasses.is_dataclass(obj)
                and obj.__module__.startswith("engrava")
            ):
                found.setdefault(f"{obj.__module__}.{obj.__qualname__}", obj)
    return found


def _classified_non_config(name: str) -> str | None:
    """Return why *name* is not configuration, or ``None`` if it is unclassified."""
    for prefix, reason in _NON_CONFIG_PREFIXES:
        if name.startswith(prefix):
            return reason
    return _NON_CONFIG_CLASSES.get(name)


_DISCOVERED: dict[str, type] = _discovered_dataclasses()


def _default_instance(cls: type) -> object:
    """Construct *cls* with every field defaulted."""
    return cls(**_REQUIRED_ARGS.get(cls.__name__, {}))  # type: ignore[arg-type]  # every config class accepts its required args by name


_FIELD_CASES: list[tuple[str, type, str]] = sorted(
    (cls.__name__, cls, field.name)
    for name, cls in _DISCOVERED.items()
    if name in _CONFIG_CLASS_NAMES
    for field in dataclasses.fields(cls)  # type: ignore[arg-type]  # guarded by is_dataclass at discovery
)


class TestEveryConfigFieldIsOwned:
    """Every field of every config object holds a value the caller cannot redefine.

    Enumerated from ``dataclasses.fields()`` rather than from a hand-written
    list, because a hand-written list is exactly what fails silently: the
    string-collection matrix in this file omitted two fields for as long as it
    existed, and nothing said so. A field added to any configuration in future
    joins this test automatically and fails until it is either decoded or
    recorded in ``_NON_BUILTIN_FIELDS`` with a reason.
    """

    def test_every_dataclass_in_the_package_is_classified(self) -> None:
        """No discovered dataclass is left unaccounted for.

        The classification rule is human — "is this constructed from user
        settings and retained to steer behaviour?" — because no structural
        property distinguishes a settings object from a result object. What is
        mechanical, and what is checked here, is that every discovered class has
        been put on one side of that line.

        **The scope of that, stated exactly.** A configuration class added
        outside the five namespaces in ``_NON_CONFIG_PREFIXES`` fails this test
        until someone classifies it. One added *inside* them is auto-classified
        and does not. See the note on that table for why those five, and why
        ``engrava.infrastructure.*`` is not among them.
        """
        unclassified: list[str] = []
        for name in _DISCOVERED:
            if name in _CONFIG_CLASS_NAMES:
                continue
            if _classified_non_config(name) is None:
                unclassified.append(name)
        assert sorted(unclassified) == []

    def test_the_classification_has_no_stale_or_overlapping_entries(self) -> None:
        """A class named on both sides, or on neither list any more, is a defect."""
        assert set(_DISCOVERED) >= _CONFIG_CLASS_NAMES
        assert set(_NON_CONFIG_CLASSES) <= set(_DISCOVERED)
        both = sorted(name for name in _CONFIG_CLASS_NAMES if _classified_non_config(name))
        assert both == []

    def test_the_walk_reaches_the_whole_package(self) -> None:
        """A discovery that silently found nothing would pass every test below."""
        assert len(_DISCOVERED) >= 60
        assert len(_FIELD_CASES) >= 100
        assert "engrava.cli.config.EngravaCLIConfig" in _DISCOVERED
        assert "engrava.domain.protocols.derived_records.DeriveGates" in _DISCOVERED

    def test_the_root_module_is_inspected_too(self) -> None:
        """``walk_packages`` skips the package module; a class there must not hide.

        A dataclass defined in ``engrava/__init__.py`` would carry the module
        name ``engrava``, which no exclusion prefix matches, so it must reach
        the partition rather than never being seen.
        """
        discovered = _discovered_dataclasses()
        for name in vars(engrava):
            obj = getattr(engrava, name)
            if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
                assert f"{obj.__module__}.{obj.__qualname__}" in discovered

    def test_the_none_default_sample_table_is_complete(self) -> None:
        """Every field that defaults to ``None`` is probeable or otherwise classified."""
        missing = [
            (cls_name, field_name)
            for cls_name, cls, field_name in _FIELD_CASES
            if getattr(_default_instance(cls), field_name) is None
            and (cls_name, field_name) not in _NONE_DEFAULT_SAMPLES
            and (cls_name, field_name) not in _NON_BUILTIN_FIELDS
        ]
        assert missing == []

    def test_the_field_tables_name_only_real_fields(self) -> None:
        """An entry for a field that no longer exists is stale and must go."""
        known = {(cls_name, field_name) for cls_name, _cls, field_name in _FIELD_CASES}
        assert set(_NON_BUILTIN_FIELDS) <= known
        assert set(_NONE_DEFAULT_SAMPLES) <= known
        assert set(_REQUIRED_ARGS) <= {cls_name for cls_name, _cls, _field in _FIELD_CASES}

    @pytest.mark.parametrize(
        ("cls_name", "cls", "field_name"),
        _FIELD_CASES,
        ids=[f"{cls_name}.{field_name}" for cls_name, _cls, field_name in _FIELD_CASES],
    )
    def test_field_holds_an_owned_value(self, cls_name: str, cls: type, field_name: str) -> None:
        default = getattr(_default_instance(cls), field_name)
        sample = _NONE_DEFAULT_SAMPLES.get((cls_name, field_name), default)
        config = _construct(cls, field_name, _subclass_of(sample))
        stored = getattr(config, field_name)

        category = _NON_BUILTIN_FIELDS.get((cls_name, field_name))
        if category is _NESTED_CONFIG:
            assert stored is None or dataclasses.is_dataclass(stored)
            return
        if category is _UNOWNED_PATH:
            assert stored is None or isinstance(stored, Path)
            return

        assert type(stored) in _OWNED_TYPES
        for entry in stored if isinstance(stored, (tuple, list, set, frozenset)) else ():
            assert type(entry) in _OWNED_TYPES
        if isinstance(stored, dict):
            for key, entry in stored.items():
                assert type(key) in _OWNED_TYPES
                assert type(entry) in _OWNED_TYPES


# ---------------------------------------------------------------------------
# The value that was range-checked is the value that was owned
# ---------------------------------------------------------------------------


class _IntThatIsNeverTooSmall(int):
    """An integer that answers ``<`` for itself, so no lower bound holds."""

    __slots__ = ()

    def __lt__(self, other: object) -> bool:
        del other
        return False


class _FloatAlwaysInRange(float):
    """A float that answers both ends of ``0.0 <= x <= 1.0`` for itself.

    A ``float`` subclass gets the *reflected* comparison first, so overriding
    ``__ge__`` decides ``0.0 <= x`` and overriding ``__le__`` decides
    ``x <= 1.0``. Neither has to agree with the number stored underneath.
    """

    __slots__ = ()

    def __le__(self, other: object) -> bool:
        del other
        return True

    def __ge__(self, other: object) -> bool:
        del other
        return True


class _ShiftingMapping(dict):  # type: ignore[type-arg]  # a bare built-in base is exactly what the adversary subclasses
    """A mapping that reports different entries each time it is read."""

    def __init__(self) -> None:
        super().__init__({"recency": 1.0})
        self._reads = 0

    def items(self) -> ItemsView[str, float]:
        self._reads += 1
        return dict.items({"recency": 1.0} if self._reads == 1 else {"recency": 99.0})


class TestRangeChecksRunOnTheOwnedValue:
    """Own first, then compare — the other order checks one value and keeps another.

    A range check is a comparison, and a comparison is a method on the value
    being compared. Performing it on the caller's object and only afterwards
    taking a copy means the copy was never the thing that satisfied the bound.
    """

    def test_a_lower_bound_cannot_be_answered_by_the_value(self) -> None:
        with pytest.raises(ConfigError, match="must be a positive integer"):
            require_positive_int(_IntThatIsNeverTooSmall(-5), "label")

    def test_a_non_negative_bound_cannot_be_answered_by_the_value(self) -> None:
        with pytest.raises(ConfigError, match="must be a non-negative integer"):
            require_nonneg_int(_IntThatIsNeverTooSmall(-5), "label")

    def test_a_unit_range_cannot_be_answered_by_the_value(self) -> None:
        with pytest.raises(ConfigError, match=r"must be a float in \[0.0, 1.0\]"):
            require_unit_float(_FloatAlwaysInRange(9.9), "label")

    def test_a_float_decodes_through_the_unbound_builtin(self) -> None:
        """``float(x)`` asks the value what it is worth; ``float.__float__`` reads it."""

        class _MisreportingFloat(float):
            __slots__ = ()

            def __float__(self) -> float:
                return 9.9

        assert require_unit_float(_MisreportingFloat(0.5), "label") == 0.5

    def test_a_field_cannot_be_stored_outside_its_documented_range(self) -> None:
        """The end of the same story, at the boundary a user actually touches."""
        with pytest.raises(ConfigError, match=r"must be a float in \[0.0, 1.0\]"):
            HygienePolicyConfig(eviction_threshold=_FloatAlwaysInRange(9.9))

    def test_legitimate_bounds_still_pass(self) -> None:
        assert require_positive_int(1, "label") == 1
        assert require_nonneg_int(0, "label") == 0
        assert require_unit_float(0.0, "label") == 0.0
        assert require_unit_float(1.0, "label") == 1.0
        assert require_unit_float(1, "label") == 1


class TestMappingsAreOwned:
    """A validated mapping cannot present different entries afterwards."""

    def test_a_mapping_is_read_once(self) -> None:
        """Two reads of the decoded mapping agree; two reads of the caller's need not.

        Equality alone would not show this — a ``dict`` subclass compares by its
        real contents however it answers ``items()`` — so the test reads the
        result twice, which is what the scoring pass does.
        """
        decoded = require_mapping(_ShiftingMapping(), "label")
        assert dict(decoded.items()) == {"recency": 1.0}
        assert dict(decoded.items()) == {"recency": 1.0}

    def test_a_mapping_decodes_to_a_plain_dict_of_plain_entries(self) -> None:
        decoded = require_mapping(_AdversarialDict({_Entry("a"): _Number(1)}), "label")
        assert type(decoded) is dict
        assert [type(key) for key in decoded] == [str]
        assert [type(entry) for entry in decoded.values()] == [int]

    def test_a_weight_map_field_holds_a_plain_dict(self) -> None:
        policy = HygienePolicyConfig(
            signal_weights=_AdversarialDict({_Entry("recency"): _Number(1)})
        )
        assert type(policy.signal_weights) is dict
        assert [type(key) for key in policy.signal_weights] == [str]
        assert [type(entry) for entry in policy.signal_weights.values()] == [int]


# ---------------------------------------------------------------------------
# The sweep runs before validation, so it must be unable to fail
# ---------------------------------------------------------------------------


class _EntriesThatRefuseToBeRead(list):  # type: ignore[type-arg]  # the adversary is the subclassing itself
    """A sequence whose ``__iter__`` raises."""

    def __iter__(self) -> Iterator[object]:
        msg = "iteration ran caller-controlled code"
        raise RuntimeError(msg)


class _EntriesThatNeverEnd(list):  # type: ignore[type-arg]  # the adversary is the subclassing itself
    """A sequence whose ``__iter__`` never stops."""

    def __iter__(self) -> Iterator[str]:
        while True:
            yield "forever"


class _MappingThatRefusesToBeRead(dict):  # type: ignore[type-arg]  # the adversary is the subclassing itself
    """A mapping whose ``items`` raises."""

    def items(self) -> ItemsView[object, object]:
        msg = "items ran caller-controlled code"
        raise RuntimeError(msg)


class _EntryThatRefusesToBeHashed:
    """A container entry whose ``__hash__`` raises on the second call."""

    def __init__(self) -> None:
        self._hashes = 0

    def __hash__(self) -> int:
        self._hashes += 1
        if self._hashes > 1:
            msg = "hashing ran caller-controlled code"
            raise RuntimeError(msg)
        return 0


class TestHostileContainersAreRefusedNotObeyed:
    """A container that fights being read is refused with the typed error.

    This does **not** assert that no caller code runs — that property was
    claimed twice on this branch and was false both times, and claiming it is
    the exact defect this work exists to remove. What is asserted is narrower
    and true: the dispatches the sweep *can* avoid it does avoid (containers are
    read through the unbound built-in iterators over their real storage, and one
    is rebuilt only when its entries have decoded to exact built-ins whose
    ``__hash__`` is the built-in one), and anything that still escapes leaves as
    the refusal validation would have produced.
    """

    def test_a_container_that_refuses_to_be_iterated_is_still_rejected_cleanly(self) -> None:
        with pytest.raises(ConfigError):
            HygienePolicyConfig(protected_priorities=_EntriesThatRefuseToBeRead([1]))

    def test_a_mapping_that_refuses_to_be_read_is_still_rejected_cleanly(self) -> None:
        with pytest.raises(ConfigError):
            HygienePolicyConfig(signal_weights=_MappingThatRefusesToBeRead({"recency": "x"}))

    def test_an_entry_that_refuses_to_be_hashed_does_not_escape(self) -> None:
        """Rebuilding never hashes anything but a built-in, so this cannot fire."""
        hostile = _EntryThatRefusesToBeHashed()
        with pytest.raises(ConfigError):
            DreamingConfig(excluded_content_types=frozenset({hostile}))

    def test_a_container_that_never_stops_iterating_still_terminates(self) -> None:
        """Read through the built-in, an endless ``__iter__`` is never entered."""
        with pytest.raises(ConfigError):
            HygienePolicyConfig(protected_priorities=_EntriesThatNeverEnd([1]))

    def test_the_sweep_reads_what_an_object_holds_not_what_it_reports(self) -> None:
        """``getattr`` would run a caller-defined ``__getattribute__``; the sweep does not.

        The sweep is exercised directly, on a dataclass that is not a
        configuration class — configuration classes refuse to be subclassed at
        all now, which is what removes this attack from the product rather than
        merely surviving it. The behaviour of the sweep itself is still worth
        pinning, because it is what any future caller of it inherits.
        """

        @dataclasses.dataclass
        class _RefusingHolder:
            entries: object = ()

            def __getattribute__(self, name: str) -> object:
                msg = "attribute read ran caller-controlled code"
                raise RuntimeError(msg)

        holder = object.__new__(_RefusingHolder)
        object.__setattr__(holder, "entries", _AdversarialTuple([_Entry("P1")]))

        own_config_fields(holder)

        stored = object.__getattribute__(holder, "entries")
        assert type(stored) is tuple
        assert [type(entry) for entry in stored] == [str]


class TestConfigClassesRefuseSubclassing:
    """A configuration class cannot be subclassed, and says so at the definition.

    This is the fix for the attack no reading discipline can survive. A **data
    descriptor** installed over a field answers benign values while the object
    is decoded and validated, then different ones for ever after; nothing
    raises, so no fail-closed guard fires, and the sweep's write-back is fed to
    the descriptor's own ``__set__`` and discarded. Installing one requires a
    subclass to put it on, so the subclass is what gets refused.

    Enumerated from the same discovered set as everything else here, so a
    configuration class added later joins this test without being listed.
    """

    @pytest.mark.parametrize("name", sorted(_CONFIG_CLASS_NAMES))
    def test_the_class_cannot_be_subclassed(self, name: str) -> None:
        cls = _DISCOVERED[name]
        with pytest.raises(TypeError, match="may not be subclassed"):
            type(f"_{cls.__name__}Subclass", (cls,), {})

    def test_a_class_statement_is_refused_as_well_as_a_dynamic_type(self) -> None:
        """Both routes to a subclass go through the same hook."""
        with pytest.raises(TypeError, match="may not be subclassed"):
            exec(  # noqa: S102 -- a class statement is the thing under test
                "class _Sub(HygienePolicyConfig): pass",
                {"HygienePolicyConfig": HygienePolicyConfig},
            )

    def test_the_refusal_names_the_configuration_class_not_the_offender(self) -> None:
        """The message reads only names this package owns."""
        with pytest.raises(TypeError, match="HygienePolicyConfig may not be subclassed"):
            type("_WhateverTheAttackerCalledIt", (HygienePolicyConfig,), {})

    def test_a_stateful_descriptor_can_no_longer_be_installed(self) -> None:
        """The concrete attack, refused where it has to start."""

        class _Sneaky:
            def __get__(self, obj: object, objtype: type | None = None) -> object:
                return () if obj is not None else self

            def __set__(self, obj: object, value: object) -> None:
                del obj, value

        with pytest.raises(TypeError, match="may not be subclassed"):
            type("_Sneaky2Faced", (HygienePolicyConfig,), {"protected_priorities": _Sneaky()})


def _look_alike_of(config: object) -> object:
    """Return *config* with its ``__class__`` reassigned to a non-subclass.

    Configuration classes refuse to be subclassed, but ``__class__`` can still
    be pointed at a layout-compatible class that was never a subclass at all.
    That is the vector the exact-type checks at the boundaries exist for, and
    the reason forbidding subclassing does not make them redundant.
    """

    class _LookAlike:
        pass

    object.__setattr__(config, "__class__", _LookAlike)
    return config


class TestConfigObjectsAcceptedAtABoundaryMustBeExactlyTheirClass:
    """Being an instance is not being the thing, and neither is looking like it.

    Subclassing is refused at the source now, so the object that reaches a
    boundary pretending to be a configuration arrives another way: with its
    ``__class__`` reassigned to a look-alike, or as something that was never a
    configuration at all. Only a check on the exact type at the point of
    acceptance sees either.
    """

    def test_a_look_alike_is_refused_where_a_nested_config_is_expected(self) -> None:
        policy = _look_alike_of(HygienePolicyConfig())
        with pytest.raises(ConfigError, match="must be exactly a HygienePolicyConfig"):
            EngravaConfig(database_path=Path("./t.db"), hygiene_policy=policy)

    @pytest.mark.parametrize(
        ("field_name", "cls"),
        [
            ("dreaming", DreamingConfig),
            ("hygiene_policy", HygienePolicyConfig),
            ("embeddings", EmbeddingConfig),
            ("search", SearchConfig),
            ("journal", JournalConfig),
            ("ttl", TTLConfig),
            ("metrics", MetricsConfig),
            ("ingest", IngestConfig),
            ("derive", DeriveGates),
        ],
    )
    def test_every_nested_config_field_requires_its_exact_class(
        self,
        field_name: str,
        cls: type,
    ) -> None:
        with pytest.raises(ConfigError, match=f"must be exactly a {cls.__name__}"):
            _construct(EngravaConfig, field_name, object())

    def test_the_services_field_requires_its_exact_class(self) -> None:
        """``services`` needs a required argument, so it is covered separately."""
        services = _look_alike_of(ServicesConfig(data_dir=Path("./data")))
        with pytest.raises(ConfigError, match="must be exactly a ServicesConfig"):
            EngravaConfig(database_path=Path("./t.db"), services=services)

    @pytest.mark.parametrize(
        ("field_name", "cls"),
        [("gates", DreamingGates), ("edges", EdgeCreationConfig)],
    )
    def test_a_dreaming_configs_own_nested_fields_require_their_exact_class(
        self,
        field_name: str,
        cls: type,
    ) -> None:
        with pytest.raises(ConfigError, match=f"must be exactly a {cls.__name__}"):
            _construct(DreamingConfig, field_name, object())

    def test_a_service_configs_embeddings_require_their_exact_class(self) -> None:
        with pytest.raises(ConfigError, match="must be exactly a EmbeddingConfig"):
            ServiceConfig(embeddings=_look_alike_of(EmbeddingConfig()))

    def test_a_services_configs_entries_require_their_exact_class(self) -> None:
        with pytest.raises(ConfigError, match="must be exactly a ServiceConfig"):
            ServicesConfig(
                data_dir=Path("./data"),
                configs={"main": _look_alike_of(ServiceConfig())},
            )

    def test_the_exact_class_is_still_accepted(self) -> None:
        config = EngravaConfig(
            database_path=Path("./t.db"),
            hygiene_policy=HygienePolicyConfig(),
            dreaming=DreamingConfig(),
            services=ServicesConfig(data_dir=Path("./data"), configs={"main": ServiceConfig()}),
        )
        assert config.hygiene_policy == HygienePolicyConfig()
        assert config.dreaming == DreamingConfig()
        assert config.services is not None
        assert config.services.configs["main"] == ServiceConfig()


class _ExplodingDescriptor:
    """A data descriptor that stores what it is given and refuses to give it back."""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: object, objtype: type | None = None) -> object:
        if obj is None:
            return self
        msg = "descriptor read ran caller-controlled code"
        raise RuntimeError(msg)

    def __set__(self, obj: object, value: object) -> None:
        object.__getattribute__(obj, "__dict__")[self._name] = value


def _holder_whose_field_cannot_be_read(descriptor: object) -> object:
    """Build a dataclass instance whose only field is intercepted by *descriptor*.

    Deliberately not a configuration class: those refuse to be subclassed, which
    is what removes this attack from the product. What is pinned here is the
    behaviour of the sweep itself, for anything else that ever calls it.
    """
    holder_cls = dataclasses.make_dataclass("_Holder", [("entries", object)])
    holder_cls.entries = descriptor  # type: ignore[attr-defined]  # installing the descriptor is the point
    return object.__new__(holder_cls)


class TestTheSweepFailsClosed:
    """Caller code the sweep reaches may cause a refusal, never an acceptance.

    Reading an arbitrary object's state in Python cannot be done with zero
    dispatch — a metaclass, a data descriptor and ``isinstance`` itself all
    consult something the caller can define — so "no caller code runs" is not a
    property this could hold, and it is not claimed anywhere any more.

    Fail-closed is the guard behind the real fix, not the fix. On a
    configuration class the attack cannot start, because such a class refuses to
    be subclassed and a descriptor needs a subclass to be installed on. What is
    pinned here is the behaviour of the sweep itself for anything else that
    calls it: whatever runs, the outcome is a typed refusal and never an object
    that should not have been built.
    """

    def test_a_raising_data_descriptor_produces_a_configuration_error(self) -> None:
        """A data descriptor intercepts the field read the sweep performs."""
        holder = _holder_whose_field_cannot_be_read(_ExplodingDescriptor())
        with pytest.raises(ConfigError, match="could not be decoded"):
            own_config_fields(holder)

    def test_a_base_exception_does_not_escape_untyped(self) -> None:
        """``Exception`` was too narrow: a refusal must not leave by another door."""

        class _InterruptingDescriptor:
            def __get__(self, obj: object, objtype: type | None = None) -> object:
                if obj is None:
                    return self
                raise KeyboardInterrupt

            def __set__(self, obj: object, value: object) -> None:
                del obj, value

        holder = _holder_whose_field_cannot_be_read(_InterruptingDescriptor())
        with pytest.raises(ConfigError, match="could not be decoded"):
            own_config_fields(holder)

    def test_the_refusal_is_a_value_error_like_every_other(self) -> None:
        """Callers catching ``ValueError`` still catch it."""
        holder = _holder_whose_field_cannot_be_read(_ExplodingDescriptor())
        with pytest.raises(ValueError, match="could not be decoded"):
            own_config_fields(holder)

    def test_a_well_behaved_configuration_is_unaffected(self) -> None:
        assert HygienePolicyConfig().protected_priorities == ("P1",)

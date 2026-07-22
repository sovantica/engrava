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

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest
import yaml

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
from engrava.domain.protocols.derived_records import DeriveGates

if TYPE_CHECKING:
    from collections.abc import Sequence


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _construct(cls: type, field: str, value: object) -> object:
    """Construct ``cls`` with ``field`` set to ``value`` (all else defaulted).

    ``EngravaConfig`` alone has a required ``database_path``; it is supplied so the
    single field under test is the only variable.
    """
    kwargs: dict[str, object] = {field: value}
    if cls is EngravaConfig:
        kwargs.setdefault("database_path", Path("./t.db"))
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

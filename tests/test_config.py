"""Tests for engrava.config — YAML loader and value objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from engrava.config import (
    ConfigError,
    DreamingConfig,
    DreamingGates,
    EngravaConfig,
    IngestConfig,
    load_config,
    resolve_hooks,
)
from engrava.domain.protocols.hooks import DefaultEngravaHooks

# ------------------------------------------------------------------
# Value object defaults
# ------------------------------------------------------------------


class TestDreamingGates:
    def test_defaults(self) -> None:
        gates = DreamingGates()
        assert gates.min_confirmations == 2
        assert gates.min_age_cycles == 1
        assert gates.max_promoted_per_run == 20
        assert gates.allow_zero_confirmation is True

    def test_frozen(self) -> None:
        gates = DreamingGates()
        with pytest.raises(AttributeError):
            gates.min_confirmations = 5  # type: ignore[misc]

    def test_cold_start_clustering_defaults_off(self) -> None:
        # The cold-start fallback is strictly opt-in; the shipped default
        # keeps the LPA path byte-identical to today.
        gates = DreamingGates()
        assert gates.cold_start_clustering is False

    @pytest.mark.parametrize(
        "field_name",
        [
            "cluster_similarity_threshold",
            "cluster_quality_persona_threshold",
            "cluster_quality_cohesion_threshold",
            "cluster_quality_external_homogeneity_threshold",
            "cluster_quality_ne_consistency_threshold",
        ],
    )
    def test_threshold_above_one_rejected(self, field_name: str) -> None:
        with pytest.raises(ValueError, match=field_name) as excinfo:
            DreamingGates(**{field_name: 2.0})  # type: ignore[arg-type]
        assert "[0.0, 1.0]" in str(excinfo.value)

    @pytest.mark.parametrize(
        "field_name",
        [
            "cluster_similarity_threshold",
            "cluster_quality_persona_threshold",
            "cluster_quality_cohesion_threshold",
            "cluster_quality_external_homogeneity_threshold",
            "cluster_quality_ne_consistency_threshold",
        ],
    )
    def test_threshold_below_zero_rejected(self, field_name: str) -> None:
        with pytest.raises(ValueError, match=field_name):
            DreamingGates(**{field_name: -0.1})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field_name",
        [
            "cluster_similarity_threshold",
            "cluster_quality_persona_threshold",
            "cluster_quality_cohesion_threshold",
            "cluster_quality_external_homogeneity_threshold",
            "cluster_quality_ne_consistency_threshold",
        ],
    )
    def test_threshold_boundaries_accepted(self, field_name: str) -> None:
        # Closed interval [0.0, 1.0] — both ends inclusive.
        for boundary in (0.0, 1.0):
            DreamingGates(**{field_name: boundary})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field_name",
        [
            "cluster_similarity_threshold",
            "cluster_quality_persona_threshold",
            "cluster_quality_cohesion_threshold",
            "cluster_quality_external_homogeneity_threshold",
            "cluster_quality_ne_consistency_threshold",
        ],
    )
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_threshold_rejects_bool(self, field_name: str, bool_value: bool) -> None:
        # ``bool`` is a subclass of ``int`` in Python; the contract is "float in
        # [0.0, 1.0]", so ``True``/``False`` must be rejected explicitly rather
        # than silently coerced to ``1.0``/``0.0`` and disable a gate. Both
        # construction paths raise the same ``ConfigError`` for this.
        with pytest.raises(ConfigError, match=field_name):
            DreamingGates(**{field_name: bool_value})  # type: ignore[arg-type]


class TestDreamingConfig:
    def test_defaults(self) -> None:
        cfg = DreamingConfig()
        assert cfg.enabled is False
        assert cfg.schedule_every_n_cycles == 100
        assert cfg.promote_threshold == 0.7
        assert len(cfg.signals) == 6
        assert cfg.signals["recency"] == 0.25
        assert cfg.signals["frequency"] == 0.20
        assert cfg.signals["action_outcome"] == 0.15
        assert cfg.candidates_limit == 200
        assert cfg.top_keyphrases_count == 3
        assert cfg.top_member_excerpts_count == 5

    def test_frozen(self) -> None:
        cfg = DreamingConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]


class TestIngestConfig:
    def test_defaults(self) -> None:
        cfg = IngestConfig()
        assert cfg.deduplication_enabled is True

    def test_frozen(self) -> None:
        cfg = IngestConfig()
        with pytest.raises(AttributeError):
            cfg.deduplication_enabled = False  # type: ignore[misc]


class TestEngravaConfig:
    def test_defaults(self) -> None:
        cfg = EngravaConfig(database_path=Path("./test.db"))
        assert cfg.wal_mode is True
        assert cfg.hooks_class is None
        assert cfg.vector_backend == "numpy"
        assert cfg.embedding_dimension == 384
        assert cfg.dreaming is None
        assert cfg.ingest.deduplication_enabled is True

    def test_frozen(self) -> None:
        cfg = EngravaConfig(database_path=Path("./test.db"))
        with pytest.raises(AttributeError):
            cfg.wal_mode = False  # type: ignore[misc]


class TestIngestParser:
    _BASE_YAML = "database:\n  path: ./engrava.sqlite3\n"

    def test_minimal_yaml_uses_defaults(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(self._BASE_YAML, encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.ingest.deduplication_enabled is True

    def test_explicit_disable(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            self._BASE_YAML + "ingest:\n  deduplication_enabled: false\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.ingest.deduplication_enabled is False

    def test_non_mapping_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            self._BASE_YAML + "ingest: not-a-mapping\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="'ingest' must be a mapping"):
            load_config(cfg_file)

    def test_non_bool_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            self._BASE_YAML + "ingest:\n  deduplication_enabled: maybe\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="must be a boolean"):
            load_config(cfg_file)


# ------------------------------------------------------------------
# load_config
# ------------------------------------------------------------------


class TestUnknownKeys:
    @pytest.mark.parametrize(
        ("yaml_suffix", "path", "key"),
        [
            ("unexpected: true\n", "<root>", "unexpected"),
            ("database:\n  path: ./test.db\n  wal_mod: true\n", "database", "wal_mod"),
            (
                "extensions:\n  dreaming:\n    promote_treshold: 0.8\n",
                "extensions.dreaming",
                "promote_treshold",
            ),
            (
                "extensions:\n  dreaming:\n    gates:\n      min_confrmations: 2\n",
                "extensions.dreaming.gates",
                "min_confrmations",
            ),
        ],
    )
    def test_unknown_static_key_rejected(
        self,
        tmp_path: Path,
        yaml_suffix: str,
        path: str,
        key: str,
    ) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        base = "database:\n  path: ./test.db\n"
        if yaml_suffix.startswith("database:"):
            base = ""
        cfg_file.write_text(base + yaml_suffix, encoding="utf-8")

        with pytest.raises(ConfigError, match=rf"{path}.*{key}"):
            load_config(cfg_file)

    def test_dynamic_service_and_signal_names_remain_supported(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            """database:
  path: ./test.db
services:
  data_dir: ./services
  configs:
    custom-service:
      embeddings: null
extensions:
  dreaming:
    signals:
      caller_signal: 0.4
""",
            encoding="utf-8",
        )

        config = load_config(cfg_file)
        assert config.services is not None
        assert "custom-service" in config.services.configs
        assert config.dreaming is not None
        assert config.dreaming.signals["caller_signal"] == 0.4


class TestLoadConfig:
    def test_minimal_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text("database:\n  path: ./test.db\n", encoding="utf-8")
        config = load_config(cfg_file)
        assert config.database_path == Path("./test.db")
        assert config.wal_mode is True
        assert config.vector_backend == "numpy"
        assert config.dreaming is None

    def test_full_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            """
database:
  path: ./data.db
  wal_mode: false

extensions:
  vector:
    backend: sqlite-vec
  dreaming:
    enabled: true
    schedule_every_n_cycles: 50
    promote_threshold: 0.5
    signals:
      recency: 0.5
      confidence: 0.5
    gates:
      min_confirmations: 3
      min_age_cycles: 20
      max_promoted_per_run: 10

hooks:
  class: "my_package.MyHooks"
""",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.database_path == Path("./data.db")
        assert config.wal_mode is False
        assert config.vector_backend == "sqlite-vec"
        assert config.hooks_class == "my_package.MyHooks"
        assert config.dreaming is not None
        assert config.dreaming.enabled is True
        assert config.dreaming.schedule_every_n_cycles == 50
        assert config.dreaming.promote_threshold == 0.5
        # A partial ``signals:`` mapping MERGES onto the defaults (overriding
        # only recency + confidence), so the other four keep their defaults
        # rather than being zeroed out.
        assert config.dreaming.signals == {
            "recency": 0.5,
            "confidence": 0.5,
            "staleness": 0.20,
            "confirmation": 0.20,
            "frequency": 0.20,
            "action_outcome": 0.15,
        }
        assert config.dreaming.gates.min_confirmations == 3
        assert config.dreaming.gates.min_age_cycles == 20

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("{{invalid yaml", encoding="utf-8")
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config(cfg_file)

    def test_non_dict_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a YAML mapping"):
            load_config(cfg_file)

    def test_missing_database_path(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "no_path.yaml"
        cfg_file.write_text("database:\n  wal_mode: true\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=r"database.path.*required"):
            load_config(cfg_file)

    def test_invalid_vector_backend(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_vec.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  vector:\n    backend: faiss\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"numpy.*sqlite-vec"):
            load_config(cfg_file)

    def test_invalid_dreaming_threshold(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_threshold.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n    promote_threshold: 2.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="promote_threshold"):
            load_config(cfg_file)

    def test_invalid_dreaming_schedule(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_sched.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\n"
            "extensions:\n  dreaming:\n    schedule_every_n_cycles: -1\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="schedule_every_n_cycles"):
            load_config(cfg_file)

    def test_invalid_gates_type(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_gates.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n    gates: 42\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"gates.*mapping"):
            load_config(cfg_file)

    def test_gates_threshold_rejects_yaml_bool(self, tmp_path: Path) -> None:
        # YAML ``true`` / ``false`` parse to Python ``bool``, which is a
        # subclass of ``int``; the loader must reject it instead of silently
        # coercing to ``1.0`` / ``0.0`` and disabling a gate.
        cfg_file = tmp_path / "bool_threshold.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\n"
            "extensions:\n  dreaming:\n"
            "    gates:\n      cluster_quality_persona_threshold: true\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"cluster_quality_persona_threshold"):
            load_config(cfg_file)

    def test_cold_start_clustering_yaml_override(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "cold_start.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n"
            "    enabled: true\n    gates:\n      cold_start_clustering: true\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.dreaming is not None
        assert config.dreaming.gates.cold_start_clustering is True

    def test_cold_start_clustering_defaults_false_when_absent(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "cold_start_absent.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n    enabled: true\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.dreaming is not None
        assert config.dreaming.gates.cold_start_clustering is False

    def test_cold_start_clustering_rejects_non_bool(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "cold_start_bad.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n"
            "    gates:\n      cold_start_clustering: 3\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"cold_start_clustering.*boolean"):
            load_config(cfg_file)

    def test_dreaming_defaults_when_empty_section(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "dreaming_empty.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n    enabled: true\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.dreaming is not None
        assert config.dreaming.enabled is True
        assert config.dreaming.schedule_every_n_cycles == 100
        assert len(config.dreaming.signals) == 6
        assert config.dreaming.top_keyphrases_count == 3
        assert config.dreaming.top_member_excerpts_count == 5

    def test_dreaming_top_keyphrases_count_yaml_override(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "dreaming_keyphrases.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n"
            "    enabled: true\n    top_keyphrases_count: 7\n"
            "    top_member_excerpts_count: 10\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.dreaming is not None
        assert config.dreaming.top_keyphrases_count == 7
        assert config.dreaming.top_member_excerpts_count == 10

    def test_dreaming_top_keyphrases_count_must_be_positive(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_top_keyphrases.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n"
            "    enabled: true\n    top_keyphrases_count: 0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"top_keyphrases_count.*positive integer"):
            load_config(cfg_file)

    def test_dreaming_member_excerpt_max_chars_default_and_override(self, tmp_path: Path) -> None:
        # Default omitted in YAML → falls back to 150.
        cfg_default = tmp_path / "default.yaml"
        cfg_default.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n    enabled: true\n",
            encoding="utf-8",
        )
        config_default = load_config(cfg_default)
        assert config_default.dreaming is not None
        assert config_default.dreaming.member_excerpt_max_chars == 150

        # YAML override is honoured.
        cfg_override = tmp_path / "override.yaml"
        cfg_override.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n"
            "    enabled: true\n    member_excerpt_max_chars: 300\n",
            encoding="utf-8",
        )
        config_override = load_config(cfg_override)
        assert config_override.dreaming is not None
        assert config_override.dreaming.member_excerpt_max_chars == 300

    def test_dreaming_member_excerpt_max_chars_must_be_positive(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_max_chars.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n"
            "    enabled: true\n    member_excerpt_max_chars: 0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"member_excerpt_max_chars.*positive integer"):
            load_config(cfg_file)

    def test_dreaming_top_member_excerpts_count_must_be_positive(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_top_excerpts.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions:\n  dreaming:\n"
            "    enabled: true\n    top_member_excerpts_count: -1\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"top_member_excerpts_count.*positive integer"):
            load_config(cfg_file)

    def test_wal_mode_non_bool_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad_wal.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\n  wal_mode: 'yes'\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"wal_mode.*boolean"):
            load_config(cfg_file)

    def test_empty_extensions_section(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "empty_ext.yaml"
        cfg_file.write_text(
            "database:\n  path: ./t.db\nextensions: {}\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.vector_backend == "numpy"
        assert config.dreaming is None


# ------------------------------------------------------------------
# Manifest YAML section parser and resolver
# ------------------------------------------------------------------


class TestParseManifests:
    """Unit tests for the ``manifests:`` YAML section parser."""

    def _write_cfg(self, tmp_path: Path, manifests_yaml: str) -> Path:
        cfg = tmp_path / "engrava.yaml"
        cfg.write_text(
            f"database:\n  path: ./t.db\n{manifests_yaml}",
            encoding="utf-8",
        )
        return cfg

    def test_no_manifests_section_returns_empty(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(tmp_path, "")
        config = load_config(cfg_file)
        assert config.extension_manifest_paths == []
        assert config.extension_discover is False

    def test_list_form_parses_paths(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(
            tmp_path,
            "manifests:\n  - 'my_plugin.manifest:MANIFEST'\n  - 'other.ext:MY_EXT'\n",
        )
        config = load_config(cfg_file)
        assert config.extension_manifest_paths == [
            "my_plugin.manifest:MANIFEST",
            "other.ext:MY_EXT",
        ]
        assert config.extension_discover is False

    def test_dict_discover_only(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(tmp_path, "manifests:\n  discover: true\n")
        config = load_config(cfg_file)
        assert config.extension_manifest_paths == []
        assert config.extension_discover is True

    def test_dict_with_paths_and_discover(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(
            tmp_path,
            "manifests:\n  discover: true\n  paths:\n    - 'my_plugin.manifest:MANIFEST'\n",
        )
        config = load_config(cfg_file)
        assert config.extension_manifest_paths == ["my_plugin.manifest:MANIFEST"]
        assert config.extension_discover is True

    def test_dict_paths_without_discover_defaults_to_false(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(
            tmp_path,
            "manifests:\n  paths:\n    - 'a.b:C'\n",
        )
        config = load_config(cfg_file)
        assert config.extension_manifest_paths == ["a.b:C"]
        assert config.extension_discover is False

    def test_invalid_path_format_raises(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(tmp_path, "manifests:\n  - 'no_colon_separator'\n")
        with pytest.raises(ConfigError, match=r"Invalid manifest path"):
            load_config(cfg_file)

    def test_non_string_path_entry_raises(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(tmp_path, "manifests:\n  - 42\n")
        with pytest.raises(ConfigError, match=r"string"):
            load_config(cfg_file)

    def test_discover_non_bool_raises(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(tmp_path, "manifests:\n  discover: 'yes'\n")
        with pytest.raises(ConfigError, match=r"boolean"):
            load_config(cfg_file)

    def test_paths_non_list_raises(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(tmp_path, "manifests:\n  paths: 'not_a_list'\n")
        with pytest.raises(ConfigError, match=r"list"):
            load_config(cfg_file)

    def test_non_mapping_non_list_raises(self, tmp_path: Path) -> None:
        cfg_file = self._write_cfg(tmp_path, "manifests: 99\n")
        with pytest.raises(ConfigError, match=r"manifests.*must be"):
            load_config(cfg_file)


# ------------------------------------------------------------------
# resolve_manifests helper
# ------------------------------------------------------------------


class TestResolveManifests:
    """Unit tests for ``resolve_manifests()``."""

    def test_empty_list_returns_empty(self) -> None:
        from engrava.config import resolve_manifests

        assert resolve_manifests([]) == []

    def test_loads_known_manifest(self) -> None:
        import sys
        import types

        from engrava.config import resolve_manifests
        from engrava.domain.manifest import ExtensionManifest
        from engrava.domain.protocols.hooks import DefaultEngravaHooks

        mod = types.ModuleType("_engrava_test_manifest_mod")
        mod.FAKE_MANIFEST = ExtensionManifest(  # type: ignore[attr-defined]
            name="fake", version="0.0.1", hooks_class=DefaultEngravaHooks
        )
        sys.modules["_engrava_test_manifest_mod"] = mod
        try:
            result = resolve_manifests(["_engrava_test_manifest_mod:FAKE_MANIFEST"])
            assert len(result) == 1
            assert isinstance(result[0], ExtensionManifest)
            assert result[0].name == "fake"
        finally:
            del sys.modules["_engrava_test_manifest_mod"]

    def test_non_existent_module_raises_config_error(self) -> None:
        from engrava.config import ConfigError, resolve_manifests

        with pytest.raises(ConfigError, match=r"Cannot import"):
            resolve_manifests(["no_such_module_xyz.manifest:MANIFEST"])

    def test_missing_attribute_raises_config_error(self) -> None:
        from engrava.config import ConfigError, resolve_manifests

        with pytest.raises(ConfigError, match=r"not found"):
            resolve_manifests(["engrava.domain.manifest:DOES_NOT_EXIST"])

    def test_wrong_type_attribute_raises_config_error(self) -> None:
        import sys
        import types

        from engrava.config import ConfigError, resolve_manifests

        # Put a plain string (not callable, not an ExtensionManifest) at a
        # well-known module path so the code reaches the final isinstance check.
        mod = types.ModuleType("_engrava_wrong_type_mod")
        mod.BAD = "not-a-manifest"  # type: ignore[attr-defined]
        sys.modules["_engrava_wrong_type_mod"] = mod
        try:
            with pytest.raises(ConfigError, match=r"expected ExtensionManifest"):
                resolve_manifests(["_engrava_wrong_type_mod:BAD"])
        finally:
            del sys.modules["_engrava_wrong_type_mod"]

    def test_discover_false_skips_entry_point_scan(self) -> None:
        from engrava.config import resolve_manifests

        # No installed 'engrava.extensions' EPs in test env — no-op is correct
        result = resolve_manifests([], discover=False)
        assert result == []


# ------------------------------------------------------------------
# resolve_hooks
# ------------------------------------------------------------------


class TestResolveHooks:
    def test_none_returns_default(self) -> None:
        hooks = resolve_hooks(None)
        assert isinstance(hooks, DefaultEngravaHooks)

    def test_valid_class_path(self) -> None:
        hooks = resolve_hooks("engrava.domain.protocols.hooks.DefaultEngravaHooks")
        assert isinstance(hooks, DefaultEngravaHooks)

    def test_bad_path_no_dot(self) -> None:
        with pytest.raises(ConfigError, match="dotted path"):
            resolve_hooks("NoDots")

    def test_missing_module(self) -> None:
        with pytest.raises(ConfigError, match="Cannot import"):
            resolve_hooks("nonexistent_module.Foo")

    def test_missing_class(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            resolve_hooks("engrava.domain.protocols.hooks.NonexistentClass")


# ------------------------------------------------------------------
# Signal weight sum validation
# ------------------------------------------------------------------


class TestSignalWeightSumValidation:
    def test_default_weights_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Default signal weights sum to 1.0 → no warning."""
        with caplog.at_level("WARNING", logger="engrava.config"):
            DreamingConfig()
        assert "signal weights sum" not in caplog.text.lower()

    def test_bad_sum_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Weights summing far from 1.0 emit a warning."""
        with caplog.at_level("WARNING", logger="engrava.config"):
            DreamingConfig(signals={"a": 0.1, "b": 0.1})
        assert "signal weights sum" in caplog.text.lower()


# ------------------------------------------------------------------
# Derived-records extension-seam section
# ------------------------------------------------------------------


class TestDeriveConfig:
    """Parsing of the ``derive:`` section into ``DeriveGates``."""

    _BASE_YAML = "database:\n  path: ./engrava.sqlite3\n"

    def test_absent_section_defaults_to_disabled(self, tmp_path: Path) -> None:
        """No ``derive`` section ⇒ the seam is disabled by default."""
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(self._BASE_YAML, encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.derive.enabled is False
        assert cfg.derive.on_error == "log"
        assert cfg.derive.max_derived_per_source == 32

    def test_full_section_parsed(self, tmp_path: Path) -> None:
        """An explicit ``derive`` section populates every gate."""
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            self._BASE_YAML
            + "derive:\n  enabled: true\n  on_error: raise\n  max_derived_per_source: 8\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.derive.enabled is True
        assert cfg.derive.on_error == "raise"
        assert cfg.derive.max_derived_per_source == 8

    def test_non_mapping_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(self._BASE_YAML + "derive: nope\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="'derive' must be a mapping"):
            load_config(cfg_file)

    def test_bad_on_error_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            self._BASE_YAML + "derive:\n  on_error: explode\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"derive\.on_error"):
            load_config(cfg_file)

    def test_bad_cap_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            self._BASE_YAML + "derive:\n  max_derived_per_source: 0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="max_derived_per_source"):
            load_config(cfg_file)

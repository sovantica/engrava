"""Tests for the release-target gate (F1: the computed version disagrees with the target).

Unlike ``tests/test_schema_release_gate.py`` and
``tests/test_main_carries_the_released_tag.py``, none of this module's logic
reads this repository's own git history -- ``check_computed_version_matches_
target.py`` only ever compares a CLI argument against a file's contents, so
every case below is unconditional coverage: it runs the same way on a
shallow, depth-1 checkout as it does on a full clone. There is deliberately
no ``TestFailabilityOnRealHistory``-style class here, because there is no
history-dependent code path to separate out.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_computed_version_matches_target.py"


@pytest.fixture
def gate_module() -> object:
    """Load ``scripts/check_computed_version_matches_target.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location(
        "check_computed_version_matches_target", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        msg = f"could not load release-target gate module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestParseVersion:
    def test_parses_bare_semver(self, gate_module: object) -> None:
        assert gate_module.parse_version("1.2.3") == (1, 2, 3)  # type: ignore[attr-defined]

    def test_rejects_a_leading_v(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("v1.2.3")  # type: ignore[attr-defined]

    def test_rejects_a_prerelease_suffix(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("1.2.3-rc.1")  # type: ignore[attr-defined]

    def test_rejects_a_leading_zero_component(self, gate_module: object) -> None:
        # "01" is not valid semantic versioning -- it must not parse to the
        # same tuple as the canonical "1" and silently stand in for it.
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("0.7.01")  # type: ignore[attr-defined]

    def test_accepts_a_legitimate_zero_component(self, gate_module: object) -> None:
        assert gate_module.parse_version("0.7.0") == (0, 7, 0)  # type: ignore[attr-defined]

    def test_rejects_a_non_ascii_digit(self, gate_module: object) -> None:
        # '\d' in a Python regex is Unicode-aware; U+0662 ARABIC-INDIC DIGIT
        # TWO must not be accepted as a spelling of the ASCII digit "2".
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("0.٢2.0")  # type: ignore[attr-defined]

    def test_rejects_surrounding_whitespace(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version(" 0.7.0")  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("0.7.0 ")  # type: ignore[attr-defined]


class TestReadDeclaredTarget:
    def test_reads_a_well_formed_file(self, gate_module: object, tmp_path: Path) -> None:
        path = tmp_path / "release-target.json"
        path.write_text('{"version": "0.7.0"}\n')
        assert gate_module.read_declared_target(path) == "0.7.0"  # type: ignore[attr-defined]

    def test_missing_file_raises(self, gate_module: object, tmp_path: Path) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(tmp_path / "does-not-exist.json")  # type: ignore[attr-defined]

    def test_malformed_json_raises(self, gate_module: object, tmp_path: Path) -> None:
        path = tmp_path / "release-target.json"
        path.write_text("{not valid json")
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]

    def test_non_object_json_raises(self, gate_module: object, tmp_path: Path) -> None:
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps(["0.7.0"]))
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]

    def test_missing_version_key_raises(self, gate_module: object, tmp_path: Path) -> None:
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps({"not_version": "0.7.0"}))
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]

    def test_non_string_version_raises(self, gate_module: object, tmp_path: Path) -> None:
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps({"version": 7}))
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]

    def test_malformed_version_string_raises(self, gate_module: object, tmp_path: Path) -> None:
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps({"version": "not-a-version"}))
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]


class TestRunGate:
    def test_matching_versions_pass(self, gate_module: object) -> None:
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            computed_version="0.7.0",
            declared_target="0.7.0",
        )
        assert passed is True
        assert any(line.startswith("PASS") for line in messages)

    def test_a_minor_mismatch_fails(self, gate_module: object) -> None:
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            computed_version="0.6.1",
            declared_target="0.7.0",
        )
        assert passed is False
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "0.6.1" in fail_lines[0]
        assert "0.7.0" in fail_lines[0]

    def test_a_patch_only_mismatch_fails(self, gate_module: object) -> None:
        # The two versions agree on major and minor -- only the patch
        # component differs. This must still be caught, not treated as
        # "close enough".
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            computed_version="0.7.0",
            declared_target="0.7.1",
        )
        assert passed is False
        assert any(line.startswith("FAIL") for line in messages)

    def test_a_major_mismatch_fails(self, gate_module: object) -> None:
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            computed_version="1.0.0",
            declared_target="0.7.0",
        )
        assert passed is False
        assert any(line.startswith("FAIL") for line in messages)


class TestMain:
    def test_matching_versions_exit_zero(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target_path = tmp_path / "release-target.json"
        target_path.write_text(json.dumps({"version": "0.7.0"}))
        monkeypatch.setattr(gate_module, "RELEASE_TARGET_PATH", target_path)  # type: ignore[attr-defined]
        exit_code = gate_module.main(["0.7.0"])  # type: ignore[attr-defined]
        assert exit_code == 0

    def test_mismatched_versions_exit_one(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target_path = tmp_path / "release-target.json"
        target_path.write_text(json.dumps({"version": "0.7.0"}))
        monkeypatch.setattr(gate_module, "RELEASE_TARGET_PATH", target_path)  # type: ignore[attr-defined]
        exit_code = gate_module.main(["0.6.1"])  # type: ignore[attr-defined]
        assert exit_code == 1

    def test_missing_target_file_exits_one_with_a_clean_message(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(gate_module, "RELEASE_TARGET_PATH", tmp_path / "missing.json")  # type: ignore[attr-defined]
        exit_code = gate_module.main(["0.7.0"])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "could not read" in captured.err

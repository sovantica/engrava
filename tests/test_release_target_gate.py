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

    def test_rejects_a_trailing_newline(self, gate_module: object) -> None:
        # Regression for the finding that "0.7.0\n" satisfied the previous
        # '^...$'-anchored VERSION_RE under match(): '$' matches just
        # before a trailing newline, so a computed version corrupted with
        # one was silently treated as equal to the clean version.
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("0.7.0\n")  # type: ignore[attr-defined]


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

    def test_a_memory_error_while_reading_raises_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression for the finding that this script caught MemoryError at
        # neither the read nor the decode stage -- unlike
        # check_release_target_was_published.py, which already caught it at
        # the read stage. Confirmed by execution: a real ~15 MB
        # release-target.json (a flat array of five million elements) read
        # under a 40 MB 'ulimit -v' raised an uncaught MemoryError out of
        # 'read_text()' before this except clause covered it. MemoryError is
        # simulated directly here rather than reproducing that memory
        # pressure in a unit test.
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps({"version": "0.7.0"}))

        def _raise_memory_error(self: Path, *args: object, **kwargs: object) -> str:
            msg = "simulated: file too large to read"
            raise MemoryError(msg)

        monkeypatch.setattr(Path, "read_text", _raise_memory_error)
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]

    def test_a_memory_error_while_decoding_raises_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Sibling of the read-stage test above: 'json.loads()' builds a
        # tree of Python objects that costs far more memory than the raw
        # text it parses, so it can exhaust memory even after a successful
        # read. Confirmed by execution against the real file described
        # above under a 60 MB 'ulimit -v': the read succeeded and
        # 'json.loads()' then raised an uncaught MemoryError.
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps({"version": "0.7.0"}))

        def _raise_memory_error(*args: object, **kwargs: object) -> object:
            msg = "simulated: document too large to decode"
            raise MemoryError(msg)

        monkeypatch.setattr(json, "loads", _raise_memory_error)
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]

    def test_a_symlinked_declaration_file_raises(self, gate_module: object, tmp_path: Path) -> None:
        # Regression for the finding that this script's post-publication
        # sibling (check_release_target_was_published.py) refused a
        # symlinked release-target.json, but this pre-publication gate
        # still followed one through a bare 'read_text()' -- letting it
        # pass, before anything is tagged, against ambient JSON elsewhere on
        # the filesystem. Confirmed by execution: pointing
        # release-target.json at an unrelated file declaring a version
        # equal to the CLI-supplied computed version made this gate exit 0
        # before this guard existed.
        ambient_dir = tmp_path / "ambient-outside-the-checkout"
        ambient_dir.mkdir()
        ambient_file = ambient_dir / "elsewhere.json"
        ambient_file.write_text(json.dumps({"version": "9.9.9"}))

        repo = tmp_path / "repo"
        repo.mkdir()
        declared_path = repo / "release-target.json"
        declared_path.symlink_to(ambient_file)

        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(declared_path)  # type: ignore[attr-defined]

    def test_a_non_regular_declaration_file_raises(
        self, gate_module: object, tmp_path: Path
    ) -> None:
        # A directory named release-target.json is not a symlink, but it is
        # also not a regular file -- the same "require a regular file"
        # guard must reject it too, not just the symlink shape. Both before
        # and after the guard, a directory raises GateInputError (read_text()
        # already turns 'IsADirectoryError' into one via the generic OSError
        # clause) -- so this asserts the *message* changes to the guard's
        # own wording, which is what actually distinguishes "the guard ran"
        # from "read_text() merely failed for an unrelated reason".
        path = tmp_path / "release-target.json"
        path.mkdir()
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert "is not a regular file" in str(excinfo.value)

    def test_a_missing_file_still_reports_could_not_read(
        self, gate_module: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The regular-file guard must not change this script's existing
        # behaviour for the ordinary "file absent" case: it still falls
        # through to read_text()'s own OSError, not a new "does not exist"
        # message the guard could have introduced.
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(tmp_path / "does-not-exist.json")  # type: ignore[attr-defined]
        assert "could not read" in str(excinfo.value)


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

    def test_a_computed_version_with_a_trailing_newline_exits_one(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression for the finding that a computed version corrupted with
        # a trailing newline ("0.7.0\n") was reported equal to a clean
        # declared target of "0.7.0" and exited 0 -- confirmed by execution
        # against the unfixed script in a disposable clone.
        target_path = tmp_path / "release-target.json"
        target_path.write_text(json.dumps({"version": "0.7.0"}))
        monkeypatch.setattr(gate_module, "RELEASE_TARGET_PATH", target_path)  # type: ignore[attr-defined]
        exit_code = gate_module.main(["0.7.0\n"])  # type: ignore[attr-defined]
        assert exit_code == 1

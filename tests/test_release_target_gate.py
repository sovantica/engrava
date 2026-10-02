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
import re
from pathlib import Path

import pytest

from tests._shell_command import INTERPRETERS, script_argv

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_computed_version_matches_target.py"
RELEASERC_PATH = REPO_ROOT / ".releaserc.json"
SCRIPT_RELATIVE = "scripts/check_computed_version_matches_target.py"
COMPUTED_VERSION_TEMPLATE = "${nextRelease.version}"
RELEASERC_KEYS = {"branches", "plugins"}


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
        # A trailing newline must be rejected. '$' matches just before a
        # trailing newline, so a '^...$'-anchored pattern under match() would
        # accept "0.7.0\n" and treat a computed version corrupted with one as
        # equal to the clean version. VERSION_RE ends in '\Z' instead.
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
        # A MemoryError raised while reading the file must surface as
        # GateInputError. It is simulated directly here rather than by
        # reproducing real memory pressure in a unit test.
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
        # tree of Python objects, so it can exhaust memory even after a
        # successful read. The MemoryError is simulated directly.
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps({"version": "0.7.0"}))

        def _raise_memory_error(*args: object, **kwargs: object) -> object:
            msg = "simulated: document too large to decode"
            raise MemoryError(msg)

        monkeypatch.setattr(json, "loads", _raise_memory_error)
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]

    def test_a_symlinked_declaration_file_raises(self, gate_module: object, tmp_path: Path) -> None:
        # A symlinked release-target.json must be refused. Followed through
        # 'read_text()', it could declare a version equal to the
        # CLI-supplied computed version from a file outside the checkout.
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
        # also not a regular file, so the "require a regular file" guard
        # must reject it too. read_text() alone would also end in a
        # GateInputError ("could not read ... IsADirectoryError"), so this
        # asserts the guard's own wording, which shows the guard ran.
        path = tmp_path / "release-target.json"
        path.mkdir()
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert "is not a regular file" in str(excinfo.value)

    def test_a_missing_file_still_reports_could_not_read(
        self, gate_module: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # For the ordinary "file absent" case the regular-file guard stays
        # silent and read_text()'s own failure is reported as "could not
        # read", not as a separate "does not exist" message.
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(tmp_path / "does-not-exist.json")  # type: ignore[attr-defined]
        assert "could not read" in str(excinfo.value)


def _read_text_rejecting_the_bytes(self: Path, *args: object, **kwargs: object) -> str:
    return b"\xff".decode("utf-8")


class TestReadDeclaredTargetBoundary:
    """``read_declared_target()`` has one catch-all boundary, not an enumerated exception list."""

    def test_a_unicode_decode_error_raises_a_clean_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A UnicodeDecodeError out of read_text() is neither OSError nor
        # MemoryError. Whether a real file with invalid bytes raises one
        # depends on the locale's default codec, so read_text() is made to
        # raise it.
        path = tmp_path / "release-target.json"
        path.write_bytes(b'{"version": "0.7.0"}')
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", _read_text_rejecting_the_bytes)
            with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
                gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)
        assert f"could not read {path}: UnicodeDecodeError: " in str(excinfo.value)

    def test_a_deeply_nested_json_document_raises_a_clean_gate_input_error(
        self, gate_module: object, tmp_path: Path
    ) -> None:
        # A 10,000-level nested JSON document. A RecursionError out of
        # json.loads() is not a json.JSONDecodeError.
        path = tmp_path / "release-target.json"
        nested = "[" * 10_000 + "]" * 10_000
        path.write_text('{"version": ' + nested + "}")
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert str(path) in str(excinfo.value)
        assert "RecursionError" in str(excinfo.value)

    def test_an_unrelated_oversized_integer_field_raises_a_clean_gate_input_error(
        self, gate_module: object, tmp_path: Path
    ) -> None:
        # A field this script never reads (not "version") holding a
        # 5,000-digit integer literal. json.loads() itself calls int() on
        # every JSON integer literal in the document, and int() refuses more
        # than 4300 digits by default, so ValueError is raised from *inside*
        # json.loads(), before this script's own code runs. That is a
        # different trigger site than the version-component overflow below.
        path = tmp_path / "release-target.json"
        huge_literal = "9" * 5000
        path.write_text('{"version": "0.7.0", "unrelated_field": ' + huge_literal + "}")
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert str(path) in str(excinfo.value)
        assert "ValueError" in str(excinfo.value)

    def test_a_5000_digit_version_component_is_rejected_by_the_regex_bound(
        self, gate_module: object, tmp_path: Path
    ) -> None:
        # A version component of 5,000 digits must be rejected by the
        # VERSION_RE bound (18 digits per component) before int() runs, via
        # parse_version()'s own diagnostic, not the generic boundary message.
        path = tmp_path / "release-target.json"
        huge_component = "9" * 5000
        path.write_text(json.dumps({"version": f"0.7.{huge_component}"}))
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert "is not a bare MAJOR.MINOR.PATCH version" in str(excinfo.value)
        assert "could not read" not in str(excinfo.value)

    def test_an_int_conversion_limit_hit_via_parse_version_is_still_caught_by_the_boundary(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defence in depth: the regex bound (18 digits) is what stops a
        # pathological component from reaching int() in the first place, but
        # the boundary in read_declared_target() must independently survive
        # a ValueError out of int() inside parse_version() too -- not rely on
        # the bound being the only thing standing between a malformed
        # component and a bare traceback. VERSION_RE is widened to an
        # unbounded shape here to simulate a future change
        # removing the bound, so this exercises the real
        # int() call inside parse_version() with a 5,000-digit component,
        # proving the boundary alone -- not the bound -- is what makes this
        # fail closed rather than traceback. (sys.set_int_max_str_digits()
        # cannot simulate this instead: its own minimum is 640, well above
        # the 18-digit bound, so no legitimate process-wide setting can ever
        # make an in-bound component overflow int() -- the bound alone
        # already closes that door for any real deployment.)
        unbounded_version_re = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)\Z")
        monkeypatch.setattr(gate_module, "VERSION_RE", unbounded_version_re)  # type: ignore[attr-defined]
        path = tmp_path / "release-target.json"
        huge_component = "9" * 5000
        path.write_text(json.dumps({"version": f"0.7.{huge_component}"}))
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert str(path) in str(excinfo.value)
        assert "ValueError" in str(excinfo.value)

    def test_a_deliberate_gate_input_error_is_not_rewrapped(
        self, gate_module: object, tmp_path: Path
    ) -> None:
        # The boundary must re-raise a deliberate GateInputError exactly as
        # raised, not fold it into the generic "could not read ..." message
        # -- that would trade a specific diagnostic for a vaguer one.
        path = tmp_path / "release-target.json"
        path.write_text(json.dumps({"not_version": "0.7.0"}))
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target(path)  # type: ignore[attr-defined]
        assert str(excinfo.value) == f"{path} does not declare a 'version' key"


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
        # A computed version with a trailing newline ("0.7.0\n") must not be
        # reported equal to a clean declared target of "0.7.0"; main() exits 1.
        target_path = tmp_path / "release-target.json"
        target_path.write_text(json.dumps({"version": "0.7.0"}))
        monkeypatch.setattr(gate_module, "RELEASE_TARGET_PATH", target_path)  # type: ignore[attr-defined]
        exit_code = gate_module.main(["0.7.0\n"])  # type: ignore[attr-defined]
        assert exit_code == 1


def _plugin_name(entry: object) -> str:
    """Return the plugin name of one ``plugins`` entry (a bare string or ``[name, config]``)."""
    if isinstance(entry, str):
        return entry
    assert isinstance(entry, list)
    assert isinstance(entry[0], str)
    return entry[0]


def _plugin_config(entry: object) -> dict[str, object]:
    """Return the configuration object of one ``plugins`` entry, empty for a bare string."""
    if isinstance(entry, list) and len(entry) > 1:
        assert isinstance(entry[1], dict)
        return entry[1]
    return {}


class TestSemanticReleaseWiring:
    """The tests above hand the gate a version directly; these pin how the release passes it one.

    ``.releaserc.json`` says how the version semantic-release computed
    reaches ``check_computed_version_matches_target.py``. A literal in place
    of the template leaves the cases above green while the gate compares the
    target against a fixed string. The configuration is pinned to the
    top-level keys it carries today and the gate's entry to its command, so
    a new key at either level fails here until the pin is updated.
    """

    @staticmethod
    def _config() -> dict[str, object]:
        config = json.loads(RELEASERC_PATH.read_text(encoding="utf-8"))
        assert isinstance(config, dict)
        return config

    @classmethod
    def _plugins(cls) -> list[object]:
        plugins = cls._config()["plugins"]
        assert isinstance(plugins, list)
        return plugins

    @staticmethod
    def _index_of(plugins: list[object], name: str) -> int:
        matching = [i for i, entry in enumerate(plugins) if _plugin_name(entry) == name]
        assert len(matching) == 1, f"expected exactly one {name} entry, found {len(matching)}"
        return matching[0]

    @classmethod
    def _gate_entry(cls, plugins: list[object]) -> int:
        matching = [
            i
            for i, entry in enumerate(plugins)
            if _plugin_name(entry) == "@semantic-release/exec"
            and script_argv(_plugin_config(entry).get("prepareCmd"), SCRIPT_RELATIVE) is not None
        ]
        assert len(matching) == 1, (
            "expected exactly one @semantic-release/exec prepareCmd to run "
            f"{SCRIPT_RELATIVE} under an interpreter, found {len(matching)}"
        )
        return matching[0]

    @classmethod
    def _gate_command(cls, plugins: list[object]) -> list[str]:
        command = _plugin_config(plugins[cls._gate_entry(plugins)]).get("prepareCmd")
        argv = script_argv(command, SCRIPT_RELATIVE)
        assert argv is not None
        return argv

    def test_the_gate_is_the_script_an_interpreter_runs_from_an_exec_prepare_command(
        self,
    ) -> None:
        argv = self._gate_command(self._plugins())

        assert argv[0] in INTERPRETERS
        assert argv[1] == SCRIPT_RELATIVE

    def test_the_gate_command_ends_with_the_computed_version_template(self) -> None:
        argv = self._gate_command(self._plugins())

        assert argv[2:] == [COMPUTED_VERSION_TEMPLATE]

    def test_the_release_configuration_carries_no_key_it_does_not_carry_today(self) -> None:
        config = self._config()

        assert set(config) <= RELEASERC_KEYS, (
            "a top-level key in .releaserc.json can change whether the gate runs; check "
            f"{sorted(set(config) - RELEASERC_KEYS)} against it, then add it here"
        )

    def test_the_gate_entry_configures_nothing_but_its_command(self) -> None:
        plugins = self._plugins()

        config = _plugin_config(plugins[self._gate_entry(plugins)])

        assert set(config) == {"prepareCmd"}, (
            "a key beside prepareCmd on the gate's entry can change whether, or where, the "
            f"command runs; found {sorted(set(config) - {'prepareCmd'})}"
        )

    def test_the_gate_runs_after_the_analyzer_and_notes_plugins_and_before_the_git_plugin(
        self,
    ) -> None:
        plugins = self._plugins()
        gate = self._gate_entry(plugins)

        assert self._index_of(plugins, "@semantic-release/commit-analyzer") < gate
        assert self._index_of(plugins, "@semantic-release/release-notes-generator") < gate
        assert gate < self._index_of(plugins, "@semantic-release/git")

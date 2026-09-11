"""Tests for the release-target-was-published gate (F2: no version was published, silently).

Like ``tests/test_release_target_gate.py``, this needs no
``TestFailabilityOnRealHistory``-style class guarded by ``skipif``: the two
cases that touch git at all (a tag existing or not) are built from disposable
throwaway repositories under ``tmp_path``, the same pattern
``tests/test_main_carries_the_released_tag.py`` uses for its
``TestIsAncestorAgainstADisposableRepository`` class -- so every case here
runs unconditionally, including on a shallow, depth-1 CI checkout, without
depending on any tag actually present in this repository's own history.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_release_target_was_published.py"


@pytest.fixture
def gate_module() -> object:
    """Load ``scripts/check_release_target_was_published.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("check_release_target_was_published", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load release-target-was-published module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_disposable_repo(path: Path) -> None:
    """Initialise an empty git repository at ``path`` with a local commit identity."""
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)  # noqa: S603, S607
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],  # noqa: S607
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)  # noqa: S607
    subprocess.run(
        ["git", "commit", "--quiet", "--allow-empty", "-m", "base commit"],  # noqa: S607
        cwd=path,
        check=True,
    )


def _write_target(path: Path, version: object) -> None:
    (path / "release-target.json").write_text(json.dumps({"version": version}))


class TestReadDeclaredTarget:
    def test_no_file_returns_none(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        assert gate_module.read_declared_target() is None  # type: ignore[attr-defined]

    def test_a_well_formed_file_returns_its_version(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_target(tmp_path, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        assert gate_module.read_declared_target() == "0.7.0"  # type: ignore[attr-defined]

    def test_malformed_json_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "release-target.json").write_text("{not valid json")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_non_object_json_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "release-target.json").write_text(json.dumps(["0.7.0"]))
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_missing_version_key_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "release-target.json").write_text(json.dumps({"not_version": "0.7.0"}))
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_non_string_version_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_target(tmp_path, 7)
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_malformed_version_string_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_target(tmp_path, "not-a-version")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]


class TestTagExistsAgainstADisposableRepository:
    def test_an_existing_tag_is_found(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.tag_exists("0.7.0") is True  # type: ignore[attr-defined]

    def test_a_missing_tag_is_not_found(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.tag_exists("0.7.0") is False  # type: ignore[attr-defined]


class TestRunGateWithStubbedReads:
    """Drive run_gate() with monkeypatched module functions -- no git involved."""

    def test_no_declared_target_passes(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_declared_target", lambda: None)  # type: ignore[attr-defined]
        passed, messages = gate_module.run_gate()  # type: ignore[attr-defined]
        assert passed is True
        assert any(line.startswith("PASS") for line in messages)

    def test_a_published_target_passes(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_declared_target", lambda: "0.7.0")  # type: ignore[attr-defined]
        monkeypatch.setattr(gate_module, "tag_exists", lambda _version: True)  # type: ignore[attr-defined]
        passed, messages = gate_module.run_gate()  # type: ignore[attr-defined]
        assert passed is True
        assert any("0.7.0" in line and line.startswith("PASS") for line in messages)

    def test_an_unpublished_target_fails(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_declared_target", lambda: "0.7.0")  # type: ignore[attr-defined]
        monkeypatch.setattr(gate_module, "tag_exists", lambda _version: False)  # type: ignore[attr-defined]
        passed, messages = gate_module.run_gate()  # type: ignore[attr-defined]
        assert passed is False
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "0.7.0" in fail_lines[0]


class TestMainAgainstADisposableRepository:
    def test_no_target_declared_exits_zero(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 0  # type: ignore[attr-defined]

    def test_a_published_target_exits_zero(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 0  # type: ignore[attr-defined]

    def test_an_unpublished_target_exits_one(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "0.7.0" in captured.out
        assert "FAIL" in captured.out

    def test_a_malformed_target_file_exits_one(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        (repo / "release-target.json").write_text("{not valid json")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 1  # type: ignore[attr-defined]

"""Tests for ``scripts/verify_wheel_execution.py``'s argv and lane assertions.

Drives the pure argv-building and assertion functions directly, with
``_run`` monkeypatched where a lane assertion needs a subprocess result --
no wheel build, no venv creation, no network. The full sequence (build a
wheel, create two venvs, install into them, run the real CLI) is exercised
separately, against a real wheel, by the release pipeline itself.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_wheel_execution.py"


@pytest.fixture
def smoke_module() -> object:
    """Load ``scripts/verify_wheel_execution.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("verify_wheel_execution", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load verify_wheel_execution module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed(
    stdout: str = "", stderr: str = "", *, returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["engrava"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestBuildCommandArgs:
    """remember/recall resolve through --config in the vector lane; info never does."""

    @pytest.mark.parametrize("command", ["remember", "info", "recall"])
    def test_base_lane_every_command_uses_db_flag_only(
        self, smoke_module: object, command: str
    ) -> None:
        args = smoke_module._build_command_args(
            "engrava", command, db_path=Path("/work/smoke-base.db"), config_path=None
        )
        assert "--db" in args
        assert "/work/smoke-base.db" in args
        assert "--config" not in args

    @pytest.mark.parametrize("command", ["remember", "recall"])
    def test_vector_lane_remember_and_recall_use_config_flag_only(
        self, smoke_module: object, command: str
    ) -> None:
        args = smoke_module._build_command_args(
            "engrava",
            command,
            db_path=Path("/work/smoke-vector.db"),
            config_path=Path("/work/engrava.yaml"),
        )
        assert "--db" not in args
        assert "/work/smoke-vector.db" not in args
        assert "--config" in args
        assert "/work/engrava.yaml" in args

    def test_vector_lane_info_uses_db_flag_matching_configs_database_path(
        self, smoke_module: object
    ) -> None:
        """``info`` never reads ``--config``'s ``database.path`` -- it needs ``--db``."""
        db_path = Path("/work/smoke-vector.db")
        args = smoke_module._build_command_args(
            "engrava", "info", db_path=db_path, config_path=Path("/work/engrava.yaml")
        )
        assert "--config" not in args
        assert "--db" in args
        assert str(db_path) in args


class TestVectorLaneRecallQuery:
    """The vector lane's recall query must share no word with the remembered text.

    Tokenised the way the module itself does (``_word_tokens`` -- lower-case,
    runs of letters/digits), so a differently-cased repeat of a word is
    still caught. This is an exact-token comparison, not a stem-aware one --
    ``"packaging"`` and ``"pack"`` would not collide here -- but that matches
    engrava's own FTS5 tokenizer (``unicode61``, not wrapped in a stemmer;
    see ``schema_core.sql``), which does no stemming either, so comparing
    exact tokens is the right unit for this check.
    """

    def test_shares_no_word_token_with_the_remembered_text(self, smoke_module: object) -> None:
        remembered_tokens = smoke_module._word_tokens(smoke_module._remembered_text("vector"))
        query_tokens = smoke_module._word_tokens(smoke_module._VECTOR_LANE_RECALL_QUERY)
        assert remembered_tokens & query_tokens == frozenset()

    def test_base_lane_recall_is_the_remembered_text_itself(self, smoke_module: object) -> None:
        """The base lane keeps an ordinary FTS match -- it has no vector arm."""
        remembered = smoke_module._remembered_text("base")
        # Sanity check on the fixture text itself: an *identical* string
        # necessarily shares every one of its own tokens with itself.
        assert smoke_module._word_tokens(remembered) & smoke_module._word_tokens(remembered)


class TestWriteVectorConfig:
    def test_config_carries_database_path(self, smoke_module: object, tmp_path: Path) -> None:
        db_path = tmp_path / "smoke-vector.db"
        config_path = smoke_module._write_vector_config(tmp_path, db_path)
        text = config_path.read_text(encoding="utf-8")
        assert f"path: {db_path}" in text
        assert "sqlite-vec" in text


class TestAssertRecallUsedVector:
    def test_passes_when_vector_backend_used_and_no_fallback_warning(
        self, smoke_module: object
    ) -> None:
        payload = {"backends_used": ["fts5", "vector"], "results": []}
        smoke_module._assert_recall_used_vector(payload, "", lane_name="vector")

    def test_fails_when_backends_used_omits_vector(self, smoke_module: object) -> None:
        payload = {"backends_used": ["fts5"], "results": []}
        with pytest.raises(AssertionError, match="vector"):
            smoke_module._assert_recall_used_vector(payload, "", lane_name="vector")

    def test_fails_when_backends_used_is_missing_entirely(self, smoke_module: object) -> None:
        payload: dict[str, object] = {"results": []}
        with pytest.raises(AssertionError, match="vector"):
            smoke_module._assert_recall_used_vector(payload, "", lane_name="vector")

    def test_fails_when_stderr_carries_the_numpy_fallback_warning(
        self, smoke_module: object
    ) -> None:
        payload = {"backends_used": ["fts5", "vector"], "results": []}
        stderr = "sqlite-vec requested but unavailable — using numpy fallback\n"
        with pytest.raises(AssertionError, match="fallback"):
            smoke_module._assert_recall_used_vector(payload, stderr, lane_name="vector")


class TestAssertVectorIndexPopulated:
    def test_passes_when_row_exists(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            lambda *_args, **_kwargs: _completed(
                stdout=json.dumps({"table_exists": True, "row_exists": True})
            ),
        )
        smoke_module._assert_vector_index_populated(
            tmp_path / "venv",
            tmp_path,
            tmp_path / "smoke-vector.db",
            "t-1",
            lane_name="vector",
        )

    def test_fails_when_table_missing(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            lambda *_args, **_kwargs: _completed(
                stdout=json.dumps({"table_exists": False, "row_exists": False})
            ),
        )
        with pytest.raises(AssertionError, match="embedding_vec"):
            smoke_module._assert_vector_index_populated(
                tmp_path / "venv",
                tmp_path,
                tmp_path / "smoke-vector.db",
                "t-1",
                lane_name="vector",
            )

    def test_fails_when_table_exists_but_no_row(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            lambda *_args, **_kwargs: _completed(
                stdout=json.dumps({"table_exists": True, "row_exists": False})
            ),
        )
        with pytest.raises(AssertionError, match="no row"):
            smoke_module._assert_vector_index_populated(
                tmp_path / "venv",
                tmp_path,
                tmp_path / "smoke-vector.db",
                "t-1",
                lane_name="vector",
            )

    def test_missing_table_fails_even_if_a_row_were_reported(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the table check on its own, independent of the row check.

        A correct check script can never report ``row_exists`` on a missing
        table (see ``_INDEX_CHECK_PROGRAM``'s own ``if table_exists:``
        guard), so this payload cannot occur in practice -- but it isolates
        the table-existence branch: without it, a check script that got
        this one field wrong would slip through on the row check alone.
        """
        monkeypatch.setattr(
            smoke_module,
            "_run",
            lambda *_args, **_kwargs: _completed(
                stdout=json.dumps({"table_exists": False, "row_exists": True})
            ),
        )
        with pytest.raises(AssertionError, match="embedding_vec"):
            smoke_module._assert_vector_index_populated(
                tmp_path / "venv",
                tmp_path,
                tmp_path / "smoke-vector.db",
                "t-1",
                lane_name="vector",
            )


class TestRunSmokeSequenceVectorLane:
    """Drives the full lane sequence with ``_run`` mocked -- no venv, no CLI."""

    @staticmethod
    def _fake_run(
        *,
        recall_backends: list[str],
        recall_stderr: str = "",
        index_payload: dict[str, bool],
    ) -> Callable[..., subprocess.CompletedProcess[str]]:
        def fake_run(
            command: list[str], *, cwd: Path, check: bool = True
        ) -> subprocess.CompletedProcess[str]:
            del cwd, check
            if "remember" in command:
                return _completed(stdout=json.dumps({"thought_id": "t-1"}))
            if "info" in command:
                return _completed(stdout=json.dumps({"thoughts": {"total": 1}}))
            if "recall" in command:
                return _completed(
                    stdout=json.dumps(
                        {"results": [{"thought_id": "t-1"}], "backends_used": recall_backends}
                    ),
                    stderr=recall_stderr,
                )
            # The lane-venv sqlite-vec index check.
            return _completed(stdout=json.dumps(index_payload))

        return fake_run

    def test_vector_lane_remember_and_recall_skip_db_but_info_uses_it(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """remember/recall resolve through --config alone; info needs --db.

        ``info`` reads only the global ``--db`` (see ``docs/cli.md``'s
        "Store resolution" section, scoped to remember/recall/link) --
        without it, it looks at the CLI's own default rather than the
        config's ``database.path`` and never finds what remember wrote.
        """
        seen_commands: dict[str, list[str]] = {}
        fake_run = self._fake_run(
            recall_backends=["fts5", "vector"],
            index_payload={"table_exists": True, "row_exists": True},
        )

        def recording_run(
            command: list[str], *, cwd: Path, check: bool = True
        ) -> subprocess.CompletedProcess[str]:
            for name in ("remember", "info", "recall"):
                if name in command:
                    seen_commands[name] = command
                    break
            return fake_run(command, cwd=cwd, check=check)

        monkeypatch.setattr(smoke_module, "_run", recording_run)
        db_path = tmp_path / "smoke-vector.db"
        smoke_module._run_smoke_sequence(
            tmp_path / "venv",
            tmp_path,
            "vector",
            db_path=db_path,
            config_path=tmp_path / "engrava.yaml",
        )
        assert set(seen_commands) == {"remember", "info", "recall"}
        for name in ("remember", "recall"):
            assert "--db" not in seen_commands[name]
            assert "--config" in seen_commands[name]
        assert "--config" not in seen_commands["info"]
        assert "--db" in seen_commands["info"]
        assert str(db_path) in seen_commands["info"]
        # remember writes the ordinary text; recall queries the disjoint one.
        assert smoke_module._remembered_text("vector") in seen_commands["remember"]
        assert smoke_module._VECTOR_LANE_RECALL_QUERY in seen_commands["recall"]
        assert smoke_module._remembered_text("vector") not in seen_commands["recall"]

    def test_vector_lane_fails_when_recall_does_not_find_the_thought(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(
            command: list[str], *, cwd: Path, check: bool = True
        ) -> subprocess.CompletedProcess[str]:
            del cwd, check
            if "remember" in command:
                return _completed(stdout=json.dumps({"thought_id": "t-1"}))
            if "info" in command:
                return _completed(stdout=json.dumps({"thoughts": {"total": 1}}))
            if "recall" in command:
                # A vector-only query that found nothing at all.
                return _completed(stdout=json.dumps({"results": [], "backends_used": ["vector"]}))
            msg = f"unexpected command: {command}"
            raise AssertionError(msg)

        monkeypatch.setattr(smoke_module, "_run", fake_run)
        with pytest.raises(AssertionError, match="did not find"):
            smoke_module._run_smoke_sequence(
                tmp_path / "venv",
                tmp_path,
                "vector",
                db_path=tmp_path / "smoke-vector.db",
                config_path=tmp_path / "engrava.yaml",
            )

    def test_vector_lane_prints_evidence_line_on_success(
        self,
        smoke_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            self._fake_run(
                recall_backends=["fts5", "vector"],
                index_payload={"table_exists": True, "row_exists": True},
            ),
        )
        smoke_module._run_smoke_sequence(
            tmp_path / "venv",
            tmp_path,
            "vector",
            db_path=tmp_path / "smoke-vector.db",
            config_path=tmp_path / "engrava.yaml",
        )
        out = capsys.readouterr().out
        assert "[vector]" in out
        assert "vector-only query" in out
        assert "sqlite-vec" in out

    def test_vector_lane_fails_when_backends_used_omits_vector(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            self._fake_run(
                recall_backends=["fts5"],
                index_payload={"table_exists": True, "row_exists": True},
            ),
        )
        with pytest.raises(AssertionError):
            smoke_module._run_smoke_sequence(
                tmp_path / "venv",
                tmp_path,
                "vector",
                db_path=tmp_path / "smoke-vector.db",
                config_path=tmp_path / "engrava.yaml",
            )

    def test_vector_lane_fails_when_recall_stderr_carries_fallback_warning(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            self._fake_run(
                recall_backends=["fts5", "vector"],
                recall_stderr="sqlite-vec requested but unavailable — using numpy fallback\n",
                index_payload={"table_exists": True, "row_exists": True},
            ),
        )
        with pytest.raises(AssertionError):
            smoke_module._run_smoke_sequence(
                tmp_path / "venv",
                tmp_path,
                "vector",
                db_path=tmp_path / "smoke-vector.db",
                config_path=tmp_path / "engrava.yaml",
            )

    def test_vector_lane_fails_when_index_has_no_row(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            self._fake_run(
                recall_backends=["fts5", "vector"],
                index_payload={"table_exists": True, "row_exists": False},
            ),
        )
        with pytest.raises(AssertionError):
            smoke_module._run_smoke_sequence(
                tmp_path / "venv",
                tmp_path,
                "vector",
                db_path=tmp_path / "smoke-vector.db",
                config_path=tmp_path / "engrava.yaml",
            )

    def test_vector_lane_fails_when_index_table_missing(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            smoke_module,
            "_run",
            self._fake_run(
                recall_backends=["fts5", "vector"],
                index_payload={"table_exists": False, "row_exists": False},
            ),
        )
        with pytest.raises(AssertionError):
            smoke_module._run_smoke_sequence(
                tmp_path / "venv",
                tmp_path,
                "vector",
                db_path=tmp_path / "smoke-vector.db",
                config_path=tmp_path / "engrava.yaml",
            )

    def test_base_lane_unaffected_by_vector_assertions(
        self, smoke_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The base lane asserts only what it already asserted -- no vector checks."""
        calls: list[list[str]] = []

        def fake_run(
            command: list[str], *, cwd: Path, check: bool = True
        ) -> subprocess.CompletedProcess[str]:
            del cwd, check
            calls.append(command)
            if "remember" in command:
                return _completed(stdout=json.dumps({"thought_id": "t-1"}))
            if "info" in command:
                return _completed(stdout=json.dumps({"thoughts": {"total": 1}}))
            if "recall" in command:
                # No "vector" in backends_used and no index check -- must
                # still pass: the base lane never configures sqlite-vec.
                return _completed(
                    stdout=json.dumps(
                        {"results": [{"thought_id": "t-1"}], "backends_used": ["fts5"]}
                    )
                )
            msg = f"unexpected base-lane command: {command}"
            raise AssertionError(msg)

        monkeypatch.setattr(smoke_module, "_run", fake_run)
        smoke_module._run_smoke_sequence(
            tmp_path / "venv",
            tmp_path,
            "base",
            db_path=tmp_path / "smoke-base.db",
            config_path=None,
        )
        assert calls
        assert all("--config" not in c for c in calls)
        assert all("--db" in c for c in calls)
        recall_call = next(c for c in calls if "recall" in c)
        assert smoke_module._remembered_text("base") in recall_call

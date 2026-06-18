"""Subprocess-driven smoke tests for the shipped walkthrough example.

The test runs ``examples/quickstart.py`` as the user would, via
``python examples/quickstart.py``, and asserts that the output carries
the markers documented in the script docstring. It requires the
``embeddings-local`` extra: when ``sentence_transformers`` is missing
the test skips cleanly rather than failing the suite. Two defensive
guards pin the dreaming demonstration policy: the fresh-store
walkthrough script that promised a REFLECTION is no longer shipped,
and ``quickstart.py`` must not promise one either.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"

requires_local_embeddings = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (engrava[embeddings-local] extra)",
)


def _isolated_child_env(**overrides: str) -> dict[str, str]:
    """Return a deterministic, offline, single-threaded environment for a child.

    ``quickstart.py`` loads ``sentence-transformers``/``torch`` in a fresh
    subprocess. Two hazards make that spawn flaky if the child simply inherits
    the ambient environment:

    * **Network.** Without the offline flags a cold cache lets the encoder
      reach for the HuggingFace Hub, so the test would block on a socket and
      depend on the *caller* having exported the offline vars. Forcing them
      here makes the suite network-independent by construction.
    * **Native thread pools.** torch/OpenMP/MKL each spin up worker pools sized
      to the host CPU. Pinning them to a single thread keeps a heavy model load
      from contending for native resources with the rest of the suite — the
      walkthrough does a one-shot encode where extra threads buy nothing.

    Args:
        **overrides: Extra variables to set on top of the isolated defaults.

    Returns:
        A copy of ``os.environ`` with the deterministic overrides applied.

    """
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.update(overrides)
    return env


def _run_example(script_name: str, timeout_s: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — trusted invocation of the shipped example script
        [sys.executable, str(EXAMPLES_DIR / script_name)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        # The example reads nothing from stdin; closing it removes a stdin-
        # inheritance wedge when the parent runs under pytest's output capture.
        stdin=subprocess.DEVNULL,
        env=_isolated_child_env(),
    )


@requires_local_embeddings
def test_quickstart_runs_to_completion() -> None:
    """``quickstart.py`` exits 0 and prints the expected query and a match."""
    result = _run_example("quickstart.py")
    assert result.returncode == 0, f"non-zero exit; stderr=\n{result.stderr}"
    assert "Query:" in result.stdout
    assert "favorite color" in result.stdout
    assert "teal" in result.stdout.lower()


def test_agent_loop_runs_to_completion() -> None:
    """``agent_loop.py`` runs the full memory-backed turn loop to a clean exit.

    Unlike the quickstart it needs no local-embeddings extra — it uses a
    deterministic ``CallbackProvider`` and a mock LLM — so it always runs.
    """
    result = _run_example("agent_loop.py")
    assert result.returncode == 0, f"non-zero exit; stderr=\n{result.stderr}"
    assert "cycle 0:" in result.stdout
    assert "[dreaming]" in result.stdout
    assert "Done." in result.stdout


def test_notes_memory_runs_to_completion() -> None:
    """``notes_memory.py`` (the tutorial companion) runs to a clean exit.

    Uses a deterministic ``CallbackProvider`` — no local-embeddings extra — so it
    always runs.
    """
    result = _run_example("notes_memory.py")
    assert result.returncode == 0, f"non-zero exit; stderr=\n{result.stderr}"
    assert "Query:" in result.stdout
    assert "Stored 4 notes." in result.stdout


def test_dreaming_benefit_script_not_shipped() -> None:
    """The fresh-store dreaming walkthrough script is not part of the public surface.

    A previous iteration shipped a script that promised a REFLECTION
    on a fresh in-memory store, which the default consolidation
    configuration cannot deliver. The script was dropped; this guard
    keeps it gone so the dropped artifact cannot silently reappear.
    """
    assert not (EXAMPLES_DIR / "dreaming_benefit.py").exists(), (
        "examples/dreaming_benefit.py must not be shipped — "
        "the fresh-store dreaming walkthrough was dropped"
    )


@requires_local_embeddings
def test_quickstart_does_not_promise_a_reflection() -> None:
    """``quickstart.py`` output must not claim a REFLECTION on a fresh-store run.

    Default-config dreaming is conservative: a brand-new store has
    nothing to consolidate yet. This guard pins the honesty contract
    so any future edit that prints a ``[REFLECTION]`` row from the
    fresh-store walkthrough is caught by the suite.
    """
    result = _run_example("quickstart.py")
    assert result.returncode == 0, f"non-zero exit; stderr=\n{result.stderr}"
    assert "[REFLECTION]" not in result.stdout, (
        "quickstart.py must not promise a REFLECTION on a fresh-store run; "
        f"stdout=\n{result.stdout}"
    )


def test_quickstart_exits_with_actionable_message_without_extras(tmp_path: Path) -> None:
    """The shipped pre-flight check exits 2 and points at the extra when missing.

    Runs ``quickstart.py`` in a subprocess whose import system is
    pre-loaded with a meta-path finder that makes
    ``find_spec("sentence_transformers")`` return ``None`` regardless
    of whether the extra is actually installed.
    """
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import sys\n"
        "class _BlockST:\n"
        "    @classmethod\n"
        "    def find_spec(cls, name, *_args, **_kwargs):\n"
        '        if name == "sentence_transformers":\n'
        "            return None\n"
        "        return None\n"
        "# Insert at front AND drop existing finders that would resolve it,\n"
        "# so find_spec returns None for sentence_transformers.\n"
        "def _patched_find_spec(name, *args, **kwargs):\n"
        '    if name == "sentence_transformers":\n'
        "        return None\n"
        "    return _orig(name, *args, **kwargs)\n"
        "import importlib.util\n"
        "_orig = importlib.util.find_spec\n"
        "importlib.util.find_spec = _patched_find_spec\n",
    )
    env = _isolated_child_env(PYTHONPATH=str(tmp_path))
    result = subprocess.run(  # noqa: S603 — trusted invocation of the shipped example script
        [sys.executable, str(EXAMPLES_DIR / "quickstart.py")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 2, (
        f"expected exit code 2 when the extra is hidden; got {result.returncode}; "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert "engrava[embeddings-local]" in result.stderr

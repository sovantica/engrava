#!/usr/bin/env python3
"""Install the exact built wheel and run it, in two dependency lanes.

Everything else in the release gate (``verify_wheel_data.py``,
``verify_dist_cardinality.py``, ``verify_artifact_version.py``, ``twine
check``) inspects the built wheel/sdist's *archive metadata* -- what files
and what version string they contain -- never whether the thing actually
runs once installed. A packaging defect that only shows up on install
(a file the package-data list forgot, a broken console-script entry point,
an extras dependency set that fails to resolve) would sail through every
one of those checks and still ship. This script closes that gap: it
installs the exact wheel ``verify_wheel_data.py`` already built -- it does
not build a second one -- into a fresh virtual environment outside this
checkout, and runs a real smoke sequence against it through the installed
``engrava`` console script (not ``python -m engrava.cli.main``, so a broken
entry point is caught too).

Two lanes, each its own venv, because a wheel that works with one
dependency set can still have a packaging defect specific to the other:

* ``base`` -- no extras. Exercises the console script most users get from
  a plain ``pip install engrava``.
* ``vector`` -- ``engrava[embeddings-local,vec]``. Exercises the
  sentence-transformers embedding path and the sqlite-vec index, the two
  optional-dependency surfaces most likely to have their own,
  extras-specific packaging defect.

Each lane runs the same sequence against a persistent file database
(never ``:memory:``, which a broken on-disk data file would not catch):
``remember`` (create), ``info`` (reopen), ``recall`` (reopen + search).

Run from ``scripts/verify_release_artifacts.sh`` after
``verify_wheel_data.py``'s one build, before ``twine check`` / upload.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"

# (lane name, extras to install alongside the wheel).
_LANES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("base", ()),
    ("vector", ("embeddings-local", "vec")),
)


def _select_wheel() -> Path:
    """Return the wheel ``verify_wheel_data.py`` already built.

    Never builds one -- see this module's docstring on why a second build
    here would defeat the point of a single, gated build-once chain.
    """
    wheels = sorted(DIST_DIR.glob("engrava-*.whl"))
    if not wheels:
        msg = f"no wheel found in {DIST_DIR} -- run verify_wheel_data.py first"
        raise RuntimeError(msg)
    return wheels[-1]


def _venv_bin(venv_dir: Path, name: str) -> Path:
    """Return the path to a console script or interpreter inside a venv."""
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _isolated_environment() -> dict[str, str]:
    """Environment with the interpreter-path variables removed.

    An ambient ``PYTHONPATH`` pointed at this checkout -- normal when
    testing from a git worktree -- is inherited by every subprocess below
    and outranks the venv's own site-packages. Left in place, the smoke
    commands would silently import ``engrava`` from the source tree
    instead of from the wheel this script just installed, testing the
    wrong thing while still reporting success. ``PYTHONHOME`` is stripped
    for the same class of reason: it relocates the standard library
    wholesale.
    """
    environment = dict(os.environ)
    for variable in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(variable, None)
    return environment


def _run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — trusted internal invocation, fixed argv shape
        command,
        cwd=str(cwd),
        env=_isolated_environment(),
        capture_output=True,
        text=True,
        check=check,
    )


def _create_lane_venv(work_dir: Path, lane_name: str) -> Path:
    """Create a fresh virtual environment for one lane, outside the checkout.

    ``work_dir`` is a ``tempfile.TemporaryDirectory()`` result, never a path
    under ``REPO_ROOT`` -- running from inside the checkout is exactly the
    trap this script's docstring warns about (a stray relative import
    reaching the source tree instead of the installed wheel).
    """
    venv_dir = work_dir / f"venv-{lane_name}"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    return venv_dir


def _install_wheel(venv_dir: Path, wheel: Path, extras: tuple[str, ...]) -> None:
    target = f"{wheel}[{','.join(extras)}]" if extras else str(wheel)
    pip = _venv_bin(venv_dir, "pip")
    _run([str(pip), "install", "--quiet", target], cwd=venv_dir)


def _write_vector_config(work_dir: Path) -> Path:
    """Write a config enabling the sentence-transformer + sqlite-vec lane.

    Mirrors ``examples/profile-local.yaml``, with the vector backend
    switched from its numpy default to ``sqlite-vec`` -- the extra under
    test -- and the model set to ``all-MiniLM-L6-v2``, the one this
    pipeline's own HuggingFace cache step already warms.
    """
    config_path = work_dir / "engrava.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            embeddings:
              provider: sentence-transformer
              model: all-MiniLM-L6-v2
              auto_embed: true

            extensions:
              vector:
                backend: sqlite-vec
                dimension: 384
            """
        ),
        encoding="utf-8",
    )
    return config_path


def _run_smoke_sequence(
    venv_dir: Path, work_dir: Path, lane_name: str, *, config_path: Path | None
) -> None:
    """Create, reopen, and search a persistent file database via the installed CLI.

    Uses the installed ``engrava`` console script directly (not
    ``python -m engrava.cli.main``) so a broken entry point fails this gate
    instead of shipping unnoticed.
    """
    engrava = str(_venv_bin(venv_dir, "engrava"))
    db_path = work_dir / f"smoke-{lane_name}.db"
    text = f"release wheel smoke thought ({lane_name} lane)"
    # --format is a *global* option (must precede the subcommand); remember/
    # recall control their own JSON shape via their own --json flag instead
    # (see below) and ignore --format, so including it unconditionally here
    # only affects info.
    base_args = [engrava, "--db", str(db_path), "--format", "json"]
    if config_path is not None:
        base_args = [*base_args, "--config", str(config_path)]

    remember = _run([*base_args, "remember", text, "--json"], cwd=work_dir)
    remembered = json.loads(remember.stdout)
    thought_id = remembered["thought_id"]

    # Reopen: a fresh process, the same on-disk database file.
    info = _run([*base_args, "info"], cwd=work_dir)
    info_payload = json.loads(info.stdout)
    if info_payload.get("thoughts", {}).get("total", 0) < 1:
        msg = f"{lane_name}: info reports no thoughts after remember: {info.stdout!r}"
        raise AssertionError(msg)

    # Reopen again: search must find what the first process wrote.
    recall = _run([*base_args, "recall", text, "--json"], cwd=work_dir)
    recall_payload = json.loads(recall.stdout)
    found_ids = {row["thought_id"] for row in recall_payload["results"]}
    if thought_id not in found_ids:
        msg = (
            f"{lane_name}: recall did not find the thought remember just created "
            f"in {db_path}: {recall.stdout!r}"
        )
        raise AssertionError(msg)


def _run_lane(wheel: Path, lane_name: str, extras: tuple[str, ...]) -> None:
    sys.stdout.write(f"verify_wheel_execution: [{lane_name}] extras={extras or '(none)'}\n")
    with tempfile.TemporaryDirectory(prefix=f"engrava-wheel-smoke-{lane_name}-") as raw_work_dir:
        work_dir = Path(raw_work_dir)
        venv_dir = _create_lane_venv(work_dir, lane_name)
        _install_wheel(venv_dir, wheel, extras)
        config_path = _write_vector_config(work_dir) if extras else None
        _run_smoke_sequence(venv_dir, work_dir, lane_name, config_path=config_path)
    sys.stdout.write(f"verify_wheel_execution: [{lane_name}] OK\n")


def main() -> int:
    """CLI entry point."""
    try:
        wheel = _select_wheel()
    except RuntimeError as exc:
        sys.stderr.write(f"verify_wheel_execution: {exc}\n")
        return 1

    sys.stdout.write(f"verify_wheel_execution: installing and exercising {wheel.name}\n")

    for lane_name, extras in _LANES:
        try:
            _run_lane(wheel, lane_name, extras)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(
                f"verify_wheel_execution: [{lane_name}] FAILED — "
                f"{exc.cmd} exited {exc.returncode}\n"
                f"--- stdout ---\n{exc.stdout}\n--- stderr ---\n{exc.stderr}\n"
            )
            return 1
        except AssertionError as exc:
            sys.stderr.write(f"verify_wheel_execution: [{lane_name}] FAILED — {exc}\n")
            return 1

    sys.stdout.write(
        f"verify_wheel_execution: OK — {wheel.name} installs and runs in "
        f"{len(_LANES)} dependency lane(s)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

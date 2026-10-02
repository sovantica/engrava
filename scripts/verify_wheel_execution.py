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
  a plain ``pip install engrava``, resolving its database through ``--db``
  for every command.
* ``vector`` -- ``engrava[embeddings-local,vec]``. ``remember`` and
  ``recall`` resolve their database through their own config file's
  ``database.path`` instead (``--config``, never alongside ``--db`` -- an
  explicit ``--db`` outranks ``--config`` for those two commands, per
  ``engrava.cli.store_resolution``, so the two together would silently
  skip the config's ``embeddings`` / ``extensions.vector`` sections).
  ``info`` never reads a ``--config`` file at all (see ``docs/cli.md``'s
  "Store resolution" section) and always gets ``--db`` instead, pointed at
  the same file. Proves the sentence-transformers embedding path and the
  sqlite-vec index were both actually exercised, not merely installed --
  see ``_run_smoke_sequence``.

Each lane runs the same sequence against a persistent file database
(never ``:memory:``, which a broken on-disk data file would not catch):
``remember`` (create), ``info`` (reopen), ``recall`` (reopen + search).

Run from ``scripts/verify_release_artifacts.sh`` after
``verify_wheel_data.py``'s one build, before ``twine check`` / upload.
"""

from __future__ import annotations

import json
import os
import re
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


def _write_vector_config(work_dir: Path, db_path: Path) -> Path:
    """Write a config enabling the sentence-transformer + sqlite-vec lane.

    Mirrors ``examples/profile-local.yaml``, with the vector backend
    switched from its numpy default to ``sqlite-vec`` -- the extra under
    test -- and the model set to ``all-MiniLM-L6-v2``, the one this
    pipeline's own HuggingFace cache step already warms. ``database.path``
    is set to ``db_path`` so ``remember`` and ``recall`` -- run with
    ``--config`` and no ``--db`` -- resolve their store from this file
    alone (see ``_build_command_args``). ``info`` does not read this
    section at all and is pointed at the same file through ``--db``
    instead.

    Args:
        work_dir: Directory the config file is written into.
        db_path: The lane's SQLite database file, recorded in the config's
            ``database.path``.

    Returns:
        The path to the written config file.

    """
    config_path = work_dir / "engrava.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            database:
              path: {db_path}

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


#: Commands whose store resolution reads ``--config``'s ``database.path``
#: when no explicit ``--db`` is given (see ``docs/cli.md``'s "Store
#: resolution" section). ``info`` is deliberately absent: it reads only the
#: global ``--db`` / ``ENGRAVA_DB`` / the CLI's own ``./engrava.db`` default,
#: never a ``--config`` file, so it always needs ``--db`` to see a
#: non-default database at all.
_CONFIG_AWARE_COMMANDS = frozenset({"remember", "recall"})


def _build_command_args(
    engrava: str, command: str, *, db_path: Path, config_path: Path | None
) -> list[str]:
    """Build the argv prefix for one CLI command in one lane.

    A lane with no ``config_path`` (the base lane) points every command at
    ``db_path`` through ``--db``. A lane with a ``config_path`` (the vector
    lane) passes ``--config`` **instead of** ``--db`` -- but only for
    ``command in _CONFIG_AWARE_COMMANDS``: an explicit ``--db`` outranks
    ``--config`` for those commands (see ``engrava.cli.store_resolution``),
    so passing both would silently ignore the config's ``embeddings`` /
    ``extensions.vector`` sections, exactly the defect this script exists to
    catch. ``info`` is not one of those commands -- it never reads
    ``--config`` at all -- so it always gets ``--db`` in every lane, pointed
    at the same file the vector lane's config names in its own
    ``database.path``.

    Args:
        engrava: Path to the lane's installed ``engrava`` console script.
        command: The subcommand this argv prefix is built for (``"remember"``,
            ``"info"``, or ``"recall"``).
        db_path: The lane's SQLite database file. Always used for ``info``,
            and for every command in the base lane; for a config-aware
            command in a lane with a ``config_path``, the database path
            travels inside that file's own ``database.path`` instead, so
            ``db_path`` here is not referenced on that branch.
        config_path: The lane's config file, or ``None`` for the base lane.

    Returns:
        The argv prefix -- ``[engrava, "--format", "json", ...]`` -- that
        ``command``'s own arguments are appended to.

    """
    # --format is a *global* option (must precede the subcommand); remember/
    # recall control their own JSON shape via their own --json flag instead
    # and ignore --format, so including it unconditionally here only affects
    # info.
    args = [engrava, "--format", "json"]
    if config_path is not None and command in _CONFIG_AWARE_COMMANDS:
        return [*args, "--config", str(config_path)]
    return [*args, "--db", str(db_path)]


def _remembered_text(lane_name: str) -> str:
    """Return the text ``remember`` writes for one lane.

    The base lane's own ``recall`` searches for this same text (an
    ordinary FTS match -- that lane has no vector arm to prove anything
    about). The vector lane's ``recall`` deliberately does not: see
    ``_VECTOR_LANE_RECALL_QUERY``.
    """
    return f"release wheel smoke thought ({lane_name} lane)"


def _word_tokens(text: str) -> frozenset[str]:
    """Lower-cased word tokens of ``text``, for a token-overlap check.

    Lower-case runs of ASCII letters and digits: a plain word split, not
    SQLite FTS5's tokenizer.
    """
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


#: The vector lane's ``recall`` query. Picked to share no word token with
#: ``_remembered_text("vector")`` (see ``_word_tokens`` -- checked directly
#: in this script's tests, not just asserted here), so FTS5 cannot match it:
#: a hit ``recall`` still returns cannot have come from a shared word.
_VECTOR_LANE_RECALL_QUERY = "packaging verification memo"


#: Logged by ``_configure_sqlite_vec_vector_backend`` in
#: ``engrava.infrastructure.sqlite.engrava_core`` when sqlite-vec cannot be
#: loaded and the store silently falls back to the numpy backend. Checked
#: against ``recall``'s own stderr below because ``backends_used`` alone
#: cannot distinguish the two backends (see ``_assert_recall_used_vector``).
_SQLITE_VEC_FALLBACK_WARNING = "sqlite-vec requested but unavailable"


def _assert_recall_used_vector(
    recall_payload: dict[str, object], recall_stderr: str, *, lane_name: str
) -> None:
    """Assert the vector arm was live in ``recall``'s own process.

    Neither check here, alone or together, proves the vector arm actually
    *found* the thought -- ``search_hybrid`` runs FTS and vector together
    and fuses their scores, so ``backends_used`` names every arm that ran,
    not which one's result won, and reports ``"vector"`` identically for
    sqlite-vec and for the numpy fallback. This function only rules out the
    numpy fallback (via the absence of its warning on ``recall``'s own
    stderr) and confirms the vector arm ran at all. Combined with a
    token-disjoint query (``_VECTOR_LANE_RECALL_QUERY``, which rules out
    FTS) and the ``embedding_vec`` row check (``_assert_vector_index_populated``),
    the four together are what let ``_run_smoke_sequence`` say sqlite-vec
    served the match -- see its docstring for the full chain.

    Args:
        recall_payload: The parsed JSON ``recall --json`` printed.
        recall_stderr: ``recall``'s captured stderr.
        lane_name: The lane's name, for the failure message.

    Raises:
        AssertionError: ``"vector"`` is missing from ``backends_used``, or
            the sqlite-vec fallback warning appears in ``recall_stderr``.

    """
    backends_used = recall_payload.get("backends_used", [])
    if not isinstance(backends_used, list) or "vector" not in backends_used:
        msg = (
            f"{lane_name}: recall's backends_used {backends_used!r} does not include "
            f"'vector' -- the sqlite-vec index was not searched"
        )
        raise AssertionError(msg)
    if _SQLITE_VEC_FALLBACK_WARNING in recall_stderr:
        msg = (
            f"{lane_name}: recall's stderr carries the sqlite-vec-unavailable fallback "
            f"warning -- 'vector' in backends_used came from the numpy fallback, not the "
            f"sqlite-vec index: {recall_stderr!r}"
        )
        raise AssertionError(msg)


#: Runs with the lane venv's own ``python`` (never this script's) so the
#: loaded ``sqlite_vec`` wheel matches the SQLite build that interpreter was
#: compiled against. Joins ``embedding_vec`` back to ``embedding`` through
#: their shared SQLite ``rowid`` -- the same join
#: ``vector_sqlite_vec.py`` uses to resolve a vec0 hit back to a thought --
#: to prove a row for *this* thought's embedding is actually indexed, not
#: just that the table exists.
_INDEX_CHECK_PROGRAM = textwrap.dedent(
    """\
    import json
    import sqlite3
    import sys

    import sqlite_vec

    db_path, thought_id = sys.argv[1], sys.argv[2]
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)

    table_exists = (
        conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'embedding_vec'"
        ).fetchone()[0]
        == 1
    )

    row_exists = False
    if table_exists:
        row_exists = (
            conn.execute(
                "SELECT count(*) FROM embedding_vec ev "
                "JOIN embedding e ON e.rowid = ev.rowid "
                "WHERE e.owner_id = ? AND e.owner_type = 'THOUGHT'",
                (thought_id,),
            ).fetchone()[0]
            > 0
        )

    conn.close()
    json.dump({"table_exists": table_exists, "row_exists": row_exists}, sys.stdout)
    """
)


def _build_index_check_args(python: str, db_path: Path, thought_id: str) -> list[str]:
    """Build the argv for the lane-venv sqlite-vec index check.

    Args:
        python: Path to the lane venv's own interpreter.
        db_path: The lane's SQLite database file.
        thought_id: The id ``remember`` returned for the thought under test.

    Returns:
        The argv running ``_INDEX_CHECK_PROGRAM`` under ``python``.

    """
    return [python, "-c", _INDEX_CHECK_PROGRAM, str(db_path), thought_id]


def _assert_vector_index_populated(
    venv_dir: Path, work_dir: Path, db_path: Path, thought_id: str, *, lane_name: str
) -> None:
    """Assert the sqlite-vec ``embedding_vec`` index actually holds the thought's vector.

    Args:
        venv_dir: The lane's virtual environment (its ``python`` runs the check).
        work_dir: Working directory for the check subprocess.
        db_path: The lane's SQLite database file.
        thought_id: The id ``remember`` returned for the thought under test.
        lane_name: The lane's name, for the failure message.

    Raises:
        AssertionError: The ``embedding_vec`` table does not exist, or holds
            no row for ``thought_id``.

    """
    python = str(_venv_bin(venv_dir, "python"))
    result = _run(_build_index_check_args(python, db_path, thought_id), cwd=work_dir)
    payload = json.loads(result.stdout)
    if not payload["table_exists"]:
        msg = f"{lane_name}: sqlite-vec's embedding_vec table does not exist in {db_path}"
        raise AssertionError(msg)
    if not payload["row_exists"]:
        msg = (
            f"{lane_name}: embedding_vec holds no row for thought {thought_id!r} in "
            f"{db_path} -- the sqlite-vec index was never populated"
        )
        raise AssertionError(msg)


def _run_smoke_sequence(
    venv_dir: Path,
    work_dir: Path,
    lane_name: str,
    *,
    db_path: Path,
    config_path: Path | None,
) -> None:
    """Create, reopen, and search a persistent file database via the installed CLI.

    Uses the installed ``engrava`` console script directly (not
    ``python -m engrava.cli.main``) so a broken entry point fails this gate
    instead of shipping unnoticed.

    The base lane (``config_path is None``) proves only what it already
    did: the console script installs and runs, a fresh process's ``info``
    sees what ``remember`` wrote, and a fresh process's ``recall`` finds it
    by an ordinary FTS match on the same text -- all three pointed at
    ``db_path`` through ``--db``. This lane has no vector arm, so there is
    nothing more to prove about how the match was found.

    The vector lane (``config_path`` given) additionally proves the
    sqlite-vec extra was actually exercised, not merely installed.
    ``remember`` and ``recall`` run with ``--config`` and no ``--db``, so
    the config's ``embeddings`` / ``extensions.vector`` sections are the
    only thing that can configure their store -- that is what proves the
    config gets read at all. ``info`` cannot join them: it resolves only
    the global ``--db`` and never a ``--config`` file's ``database.path``
    (see ``docs/cli.md``'s "Store resolution" section, scoped to
    ``remember`` / ``recall`` / ``link``), so it runs with ``--db`` pointed
    at the same file the config names, purely to confirm ``remember``
    actually persisted to disk -- it proves nothing about the sqlite-vec
    path itself.

    What proves sqlite-vec actually served ``recall``'s match is a chain,
    not any single check: ``search_hybrid`` runs FTS and vector together
    and fuses their scores, so neither ``backends_used`` (which names every
    arm that ran, not the one whose result won) nor a query that merely
    repeats the remembered text can tell them apart. So this lane instead:

    * queries with ``_VECTOR_LANE_RECALL_QUERY``, which shares no word
      token with the remembered text (see ``_word_tokens``) -- FTS cannot
      match it, so a hit did not come from FTS;
    * passes no metadata filter -- ``search_hybrid`` takes the sqlite-vec
      ``vec0`` branch precisely when the backend is loaded and no filter is
      given (see ``engrava_core.py``'s ``_filter_clause is None`` guard);
    * checks (``_assert_recall_used_vector``) that ``recall``'s own stderr
      carries no sqlite-vec-fallback warning, so sqlite-vec -- not the
      numpy fallback -- is what is loaded in *this* process;
    * checks (``_assert_vector_index_populated``, with the lane venv's own
      interpreter) that the sqlite-vec ``embedding_vec`` index actually
      holds a row for the thought's embedding.

    Sqlite-vec loaded, no filter, a token-disjoint query still finding the
    thought, and the index holding the row: together these say sqlite-vec
    served the match. Passing a filter, or a query overlapping the
    remembered text, would silently stop proving any of it.

    Args:
        venv_dir: The lane's virtual environment.
        work_dir: Working directory the smoke sequence runs in.
        lane_name: The lane's name, used in failure messages.
        db_path: The lane's SQLite database file.
        config_path: The lane's config file, or ``None`` for the base lane
            (see ``_build_command_args``).

    Raises:
        AssertionError: Any of the checks above fails.

    """
    engrava = str(_venv_bin(venv_dir, "engrava"))
    remembered_text = _remembered_text(lane_name)

    def _args(command: str) -> list[str]:
        return _build_command_args(engrava, command, db_path=db_path, config_path=config_path)

    remember = _run([*_args("remember"), "remember", remembered_text, "--json"], cwd=work_dir)
    remembered = json.loads(remember.stdout)
    thought_id = remembered["thought_id"]

    # Reopen: a fresh process, the same on-disk database file. Always --db
    # (see _build_command_args's docstring on why info never takes --config).
    info = _run([*_args("info"), "info"], cwd=work_dir)
    info_payload = json.loads(info.stdout)
    if info_payload.get("thoughts", {}).get("total", 0) < 1:
        msg = f"{lane_name}: info reports no thoughts after remember: {info.stdout!r}"
        raise AssertionError(msg)

    # Reopen again: search must find what the first process wrote. The
    # vector lane searches for a query sharing no word with the remembered
    # text (see _VECTOR_LANE_RECALL_QUERY) -- FTS cannot match it, so a hit
    # here cannot be an FTS match. The base lane keeps the ordinary FTS
    # match: it has no vector arm to prove anything about.
    recall_query = _VECTOR_LANE_RECALL_QUERY if config_path is not None else remembered_text
    recall = _run([*_args("recall"), "recall", recall_query, "--json"], cwd=work_dir)
    recall_payload = json.loads(recall.stdout)
    found_ids = {row["thought_id"] for row in recall_payload["results"]}
    if thought_id not in found_ids:
        msg = (
            f"{lane_name}: recall (query {recall_query!r}) did not find the thought "
            f"remember just created in {db_path}: {recall.stdout!r}"
        )
        raise AssertionError(msg)

    if config_path is not None:
        _assert_recall_used_vector(recall_payload, recall.stderr, lane_name=lane_name)
        _assert_vector_index_populated(venv_dir, work_dir, db_path, thought_id, lane_name=lane_name)
        sys.stdout.write(
            f"verify_wheel_execution: [{lane_name}] recall found the thought by a "
            "vector-only query through sqlite-vec (no fallback, embedding_vec row present)\n"
        )


def _run_lane(wheel: Path, lane_name: str, extras: tuple[str, ...]) -> None:
    sys.stdout.write(f"verify_wheel_execution: [{lane_name}] extras={extras or '(none)'}\n")
    with tempfile.TemporaryDirectory(prefix=f"engrava-wheel-smoke-{lane_name}-") as raw_work_dir:
        work_dir = Path(raw_work_dir)
        venv_dir = _create_lane_venv(work_dir, lane_name)
        _install_wheel(venv_dir, wheel, extras)
        db_path = work_dir / f"smoke-{lane_name}.db"
        config_path = _write_vector_config(work_dir, db_path) if extras else None
        _run_smoke_sequence(venv_dir, work_dir, lane_name, db_path=db_path, config_path=config_path)
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

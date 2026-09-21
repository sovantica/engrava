"""``remember`` / ``recall`` / ``link`` never traceback -- one boundary, not an enumeration.

Two review rounds each found more exception types escaping a hand-enumerated
``except`` list in these three commands (see ``engrava.cli.memory_commands``):
a malformed ``--filter`` path, a directory or corrupt file given as ``--db``,
an unreadable / non-UTF-8 / directory ``--config``, an uninitialised database.
``_error_boundary`` replaces the enumeration with a single ``except
Exception`` around each command's entire body, so *every* exception that
reaches it -- not just the ones this test module happens to name -- becomes
the documented ``engrava.cli.error.v1`` object under ``kind:
"unexpected_error"``, exit ``1``, never a bare traceback.

The classes below are organised in two groups:

* :class:`TestPreviouslyTracebackingCasesNowProduceTheDocumentedObject`
  reproduces every concrete case the second review round found tracebacking,
  now against the fixed code, proving each one individually.
* :class:`TestGenericPathCoversAnyException` proves the boundary is generic
  rather than merely a longer enumeration, by injecting an exception type
  this module has never seen and could not have named in an ``except``
  clause -- the entire point of replacing enumeration with a boundary.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path as _Path
from typing import TYPE_CHECKING, NoReturn

import pytest
from click.testing import CliRunner

from engrava import ConfigError, ReferentialIntegrityError, SqliteEngravaCore
from engrava.cli import memory_commands
from engrava.cli.main import cli

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _last_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line]
    return lines[-1] if lines else ""


class TestPreviouslyTracebackingCasesNowProduceTheDocumentedObject:
    def test_recall_malformed_filter_key_is_the_documented_object(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        runner.invoke(cli, ["--db", str(db), "remember", "seed"])

        result = runner.invoke(
            cli, ["--db", str(db), "recall", "seed", "--filter", "bad[=x", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "unexpected_error"
        assert "InvalidFilterPathError" in payload["message"]
        assert "bad[" in payload["message"]

    def test_recall_malformed_filter_key_plain_text_is_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        runner.invoke(cli, ["--db", str(db), "remember", "seed"])

        result = runner.invoke(cli, ["--db", str(db), "recall", "seed", "--filter", "bad[=x"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        assert "InvalidFilterPathError" in result.output

    def test_directory_as_db_is_the_documented_object(self, tmp_path: Path) -> None:
        target = tmp_path / "a_directory"
        target.mkdir()
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(target), "remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        assert str(target) in payload["message"]

    def test_corrupt_database_file_is_the_documented_object(self, tmp_path: Path) -> None:
        db = tmp_path / "corrupt.db"
        db.write_bytes(b"not a real sqlite file")
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "recall", "x", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        # Stronger than "mentions the word database": that survives a
        # regression to a message that no longer names *which* database --
        # exactly the gap an operator running against several stores, or
        # with the path coming from --config/ENGRAVA_DB, hits.
        assert str(db) in payload["message"]

    def test_uninitialised_empty_database_is_the_documented_object(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.db"
        db.touch()
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "recall", "x", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        assert str(db) in payload["message"]

    def test_directory_as_config_is_the_documented_object(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "a_directory.yaml"
        config_dir.mkdir()
        runner = CliRunner()

        result = runner.invoke(cli, ["--config", str(config_dir), "remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        assert str(config_dir) in payload["message"]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
    def test_unreadable_config_is_the_documented_object(self, tmp_path: Path) -> None:
        config_path = tmp_path / "noperm.yaml"
        config_path.write_text("database:\n  path: x.db\n", encoding="utf-8")
        config_path.chmod(0)
        runner = CliRunner()
        try:
            result = runner.invoke(cli, ["--config", str(config_path), "remember", "hi", "--json"])
        finally:
            config_path.chmod(0o644)

        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"

    def test_non_utf8_config_is_the_documented_object(self, tmp_path: Path) -> None:
        config_path = tmp_path / "badutf8.yaml"
        config_path.write_bytes(b"database:\n  path: \xff\xfe.db\n")
        runner = CliRunner()

        result = runner.invoke(cli, ["--config", str(config_path), "remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        assert "UnicodeDecodeError" in payload["message"]

    def test_link_corrupt_database_file_is_the_documented_object(self, tmp_path: Path) -> None:
        db = tmp_path / "corrupt.db"
        db.write_bytes(b"not a real sqlite file")
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "a", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        assert str(db) in payload["message"]


class _NobodyAnticipatedError(Exception):
    """An exception type none of ``memory_commands``'s specific catches name.

    Deliberately arbitrary: it exists only to prove the boundary converts
    *any* exception reaching it, not merely the finite set this test module
    also happens to exercise through a real failing code path. That is the
    entire premise of replacing an enumerated ``except`` list with one
    ``except Exception`` boundary -- an enumeration is only ever as good as
    the last review round that extended it.
    """


_UNANTICIPATED_MESSAGE = "nobody saw this coming"


def _raise_unanticipated(*_args: object, **_kwargs: object) -> None:
    raise _NobodyAnticipatedError(_UNANTICIPATED_MESSAGE)


async def _raise_unanticipated_async(*_args: object, **_kwargs: object) -> None:
    raise _NobodyAnticipatedError(_UNANTICIPATED_MESSAGE)


class TestGenericPathCoversAnyException:
    def test_remember_converts_an_arbitrary_unanticipated_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_unanticipated)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "unexpected_error"
        assert "_NobodyAnticipatedError" in payload["message"]
        assert "nobody saw this coming" in payload["message"]

    def test_link_converts_an_arbitrary_unanticipated_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_unanticipated)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "a", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        assert "_NobodyAnticipatedError" in payload["message"]

    def test_recall_converts_an_arbitrary_unanticipated_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(SqliteEngravaCore, "recall", _raise_unanticipated_async)
        db = tmp_path / "m.db"
        runner = CliRunner()
        runner.invoke(cli, ["--db", str(db), "remember", "seed"])

        result = runner.invoke(cli, ["--db", str(db), "recall", "seed", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "unexpected_error"
        assert "_NobodyAnticipatedError" in payload["message"]


_ORIGINAL_STORE_CLOSE = SqliteEngravaCore.close


async def _raise_after_closing(store: SqliteEngravaCore) -> None:
    """Complete the real ``close()``, then fail anyway -- a close that succeeds but still raises.

    Matches the review's own repro ("I made connection close complete and
    then raise"), and avoids leaking the underlying connection: a
    monkeypatch that raises *without* running the real close would leave a
    schema/connection resource open for the rest of the test session. Calls
    the *original*, captured before any test monkeypatches this method --
    calling ``SqliteEngravaCore.close`` here instead would recurse into
    this very function for the duration of the patch.
    """
    await _ORIGINAL_STORE_CLOSE(store)
    message = "boom: store close failed"
    raise RuntimeError(message)


_LINK_WITH_BROKEN_BARE_CLOSE_SCRIPT = (
    "import aiosqlite\n"
    "_orig_close = aiosqlite.Connection.close\n"
    "async def _boom(self):\n"
    "    # Complete the real close first -- an aiosqlite connection's worker\n"
    "    # thread is not a daemon, so a monkeypatch that only raises, without\n"
    "    # ever letting the real close run, would leave the subprocess itself\n"
    "    # hanging at interpreter shutdown instead of exercising the failure\n"
    "    # this test wants (the real close *succeeding* and then a separate,\n"
    "    # unrelated failure raising anyway).\n"
    "    await _orig_close(self)\n"
    "    raise RuntimeError('boom: close failed')\n"
    "aiosqlite.Connection.close = _boom\n"
    "from engrava.cli.main import main\n"
    "main()\n"
)


def _run_subprocess(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``python *argv*`` as a real, separate process against this worktree's ``src``.

    Needed instead of ``CliRunner`` for the bare-store close-failure case
    below: pytest's own log-capturing handler attaches to the root logger
    during a test, so ``logging.lastResort`` (which is what would otherwise
    print an unhandled ``WARNING`` to ``stderr``) never fires -- the exact
    ordering this test exists to prove would be invisible through
    ``CliRunner`` inside pytest, not merely unaffected by it.

    Args:
        argv: Full argv after the interpreter, e.g.
            ``["-m", "engrava.cli.main", "--db", str(db_path), "info"]``.

    Returns:
        The completed process.

    """
    repo_src = str(_Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": repo_src}
    return subprocess.run(  # noqa: S603 -- fixed argv, no shell, our own source
        [sys.executable, *argv],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=20.0,
    )


def _run_subprocess_with_external_sigint(
    argv: list[str], marker: str, *, sigint_delay: float = 0.05
) -> subprocess.CompletedProcess[str]:
    """Run ``python *argv*`` as a real subprocess and deliver a genuine external ``SIGINT``.

    A later verification round found a real, externally delivered ``SIGINT``
    absorbed during the close-exception description on both cleanup tiers:
    ``asyncio.run()``'s own ``SIGINT`` handling (``asyncio.runners.Runner``)
    does not raise anything into the running coroutine on the first Ctrl-C --
    it only calls the main task's ``cancel()``, which is delivered as
    ``CancelledError`` at the coroutine's *next* suspension point. With no
    ``await`` between building the cleanup warning and returning or
    re-raising, that request was simply dropped. This is not a self-delivered
    ``os.kill()`` inside the child's own call stack (an earlier round found
    that timing unreliable for this exact window -- see
    ``TestCleanupLogNoLongerReformatsThePropagatingException``'s own
    ``test_keyboard_interrupt_from_close_str_reaches_clicks_own_abort_handling``
    docstring): the signal genuinely originates in *this* process and is
    delivered to the child via ``Popen.send_signal()``, exactly like a
    user's Ctrl-C or another process's ``kill``, after the child has printed
    *marker* to prove it has reached the exact synchronous window this test
    targets, with no suspension point ahead of it.

    Args:
        argv: Full argv after the interpreter, as for :func:`_run_subprocess`.
        marker: Line the child prints (then blocks briefly) once it has
            reached the vulnerable synchronous window.
        sigint_delay: Extra real time to wait, after the marker appears,
            before sending the signal -- gives the child's own stdout flush
            and this process's readline a moment to settle.

    Returns:
        The completed process, with combined stdout captured whether it
        arrived before or after the signal was sent.

    """
    repo_src = str(_Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": repo_src}
    proc = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell, our own source
        [sys.executable, *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    stdout_so_far = ""
    marker_seen = False
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        stdout_so_far += line
        if marker in line:
            marker_seen = True
            break
    if not marker_seen:
        proc.kill()
        remaining_out, stderr = proc.communicate(timeout=5.0)
        msg = (
            f"marker {marker!r} never appeared; "
            f"stdout={stdout_so_far + remaining_out!r} stderr={stderr!r}"
        )
        raise AssertionError(msg)

    time.sleep(sigint_delay)
    proc.send_signal(signal.SIGINT)
    try:
        remaining_out, stderr = proc.communicate(timeout=15.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        remaining_out, stderr = proc.communicate()
        msg = (
            "subprocess did not exit after a real, externally delivered SIGINT; "
            f"stdout={stdout_so_far + remaining_out!r} stderr={stderr!r}"
        )
        raise AssertionError(msg) from None

    return subprocess.CompletedProcess(
        proc.args, proc.returncode, stdout_so_far + remaining_out, stderr
    )


class TestFinalOutputIsGenuinelyLast:
    """A cleanup failure must not follow, or replace, an already-decided outcome.

    A third review round made a store's own ``close()`` fail after
    ``link --json`` had already decided a ``missing_thought`` failure (exit
    ``4``). Before ``_error_boundary`` became the single place that writes
    and exits (see its docstring), ``_fail`` wrote the JSON object and
    exited from *inside* the still-open ``async with``: on the bare tier, a
    close failure logged during unwind printed after that JSON, breaking
    the documented "last line is JSON" contract; on the ``--config`` tier,
    an unconditional ``finally: await store.close()`` let a close failure
    there *replace* the already-decided ``SystemExit(4)`` outright, so the
    boundary caught the close failure instead and printed a second,
    contradicting ``unexpected_error`` object at exit ``1``.
    """

    def test_bare_store_close_failure_does_not_follow_the_final_json(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        seed = _run_subprocess(["-m", "engrava.cli.main", "--db", str(db), "remember", "seed"])
        assert seed.returncode == 0, seed.stderr

        result = _run_subprocess(
            [
                "-c",
                _LINK_WITH_BROKEN_BARE_CLOSE_SCRIPT,
                "--db",
                str(db),
                "link",
                "missing-from",
                "missing-to",
                "--type",
                "ASSOCIATED",
                "--json",
            ]
        )
        assert result.returncode == 4, result.stderr
        lines = [line for line in result.stderr.split("\n") if line]
        payload = json.loads(lines[-1])
        assert payload["error"] == "missing_thought"

    def test_configured_store_close_failure_does_not_replace_the_decided_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "m.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(f"database:\n  path: {db}\n", encoding="utf-8")
        runner = CliRunner()
        seed = runner.invoke(cli, ["--config", str(config_path), "remember", "seed"])
        assert seed.exit_code == 0, seed.output

        monkeypatch.setattr(SqliteEngravaCore, "close", _raise_after_closing)
        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "link",
                "missing-from",
                "missing-to",
                "--type",
                "ASSOCIATED",
                "--json",
            ],
        )
        assert result.exit_code == 4, result.output
        lines = [line for line in result.output.split("\n") if line]
        payload = json.loads(lines[-1])
        assert payload["error"] == "missing_thought"


class _BrokenStrError(Exception):
    """An exception whose own ``__str__`` raises instead of returning text."""

    def __str__(self) -> str:
        message = "str() itself is broken"
        raise RuntimeError(message)


def _raise_broken_str(*_args: object, **_kwargs: object) -> None:
    message = "irrelevant -- __str__ never gets this far"
    raise _BrokenStrError(message)


class TestBoundaryFormattingSurvivesABrokenException:
    """The boundary's own message-building must not itself be the thing that fails.

    A third review round injected an ordinary exception whose ``__str__``
    raises and found it escaped as a bare, undocumented ``RuntimeError``
    with no error object at all: the boundary's fallback message built
    ``f"...{exc}..."``, which calls that broken ``__str__`` directly.
    :func:`~engrava.cli.memory_commands._safe_str` closes this by falling
    back to a fixed placeholder instead of letting the formatting itself
    raise a second, unrelated exception.
    """

    def test_remember_survives_an_exception_whose_str_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_broken_str)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "unexpected_error"
        assert "_BrokenStrError" in payload["message"]


class _RaisingNameMeta(type):
    """A metaclass whose ``__name__`` read raises instead of returning a string."""

    @property
    def __name__(cls) -> str:
        message = "type name unavailable"
        raise RuntimeError(message)


class _UnnameableError(Exception, metaclass=_RaisingNameMeta):
    """An exception whose *type name* cannot be read, not just its ``str()``."""


def _raise_unnameable(*_args: object, **_kwargs: object) -> None:
    message = "boom"
    raise _UnnameableError(message)


class TestBoundarySurvivesAnExceptionWhoseTypeNameRaises:
    """``type(exc).__name__`` is exactly as arbitrary as ``str(exc)``.

    A fourth review round found the type-name half of the boundary's
    fallback description read unprotected at both of its call sites (the
    debug log line and the message text itself) even though the text half
    was already guarded by the earlier ``_safe_str``. An exception whose
    *metaclass* raises on a plain ``__name__`` read escaped as a bare,
    undocumented ``RuntimeError`` with no error object at all --
    :func:`~engrava.cli.memory_commands._describe_exception` closes this the
    same way its predecessor closed the ``__str__`` half.
    """

    def test_remember_survives_an_exception_whose_type_name_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_unnameable)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "unexpected_error"
        assert "boom" in payload["message"]


class _HostileFormatStr(str):
    """A ``str`` subclass that is a valid string until something tries to format it.

    Distinct from the two escapes an earlier review round named (a plain
    ``__str__`` that raises, and a metaclass whose ``__name__`` raises):
    this value comes back from ``str(exc)`` *without* raising -- it
    genuinely is a ``str`` instance -- and only misbehaves later, when
    something interpolates it. That is exactly the gap a guarded-read/
    unguarded-interpolation shape leaves open.
    """

    __slots__ = ()

    def __format__(self, format_spec: str) -> str:
        message = "hostile: __format__ raises a BaseException, not an Exception"
        raise GeneratorExit(message)


class _CompoundHostileError(Exception):
    """An exception whose ``__str__`` succeeds but returns a hostile ``str`` subclass.

    Combines both halves of the escape in one object: the value is
    obtained without raising (so a guard around the *read* alone would
    never fire), and when it eventually misbehaves, it raises a
    ``BaseException`` that is not an ``Exception`` (so a guard written as
    ``except Exception`` would not catch it either). Deliberately
    different from this module's own ``_BrokenStrError`` (raises
    ``Exception`` directly from ``__str__``) and ``_UnnameableError``
    (raises from a metaclass ``__name__`` property).
    """

    def __str__(self) -> str:
        return _HostileFormatStr("visible text that must never reach an unguarded format")


def _raise_compound_hostile(*_args: object, **_kwargs: object) -> None:
    message = "irrelevant ctor argument"
    raise _CompoundHostileError(message)


class TestDescribeExceptionNormalizesEveryValueToAnExactStr:
    """The final f-string must never see anything but a plain, exact ``str``.

    A fifth review round found that guarding the *read* of ``str(exc)`` and
    ``type(exc).__name__`` was not enough: a value that comes back without
    raising can still be unsafe to interpolate, if interpolating it -- not
    just obtaining it -- is what runs arbitrary code.
    :func:`~engrava.cli.memory_commands._describe_exception` now normalizes
    each half with :func:`~engrava.config_validation.own_str` inside the
    same guard that obtains it, so nothing but an exact ``str``, with a
    plain, non-overridden ``__format__``, ever reaches the final f-string.
    """

    def test_describe_exception_returns_an_exact_str_for_a_compound_hostile_value(self) -> None:
        exc = _CompoundHostileError("irrelevant ctor argument")

        result = memory_commands._describe_exception(exc)

        assert type(result) is str
        assert result == (
            "_CompoundHostileError: visible text that must never reach an unguarded format"
        )

    def test_remember_survives_a_compound_hostile_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_compound_hostile)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 1
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "unexpected_error"
        assert "_CompoundHostileError" in payload["message"]


class _ExpandingIteratorStr(str):
    """A one-byte ``str`` subclass whose ``__iter__`` yields 200,000 characters.

    Distinct from ``_HostileFormatStr``, which attacks *interpolation*
    (``__format__``): this class attacks *normalization* itself, and only if
    normalization is implemented as something that walks the value through
    its own ``__iter__`` -- as ``"".join(...)`` did before a sixth review
    round replaced it with :func:`~engrava.config_validation.own_str` -- as
    opposed to reading the underlying buffer directly.
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[str]:
        for _ in range(200_000):
            yield "Z"


class _ExpandingIteratorError(Exception):
    """An exception whose ``str()`` is a one-byte value with an unbounded ``__iter__``."""

    def __str__(self) -> str:
        return _ExpandingIteratorStr("a")


class TestDescribeExceptionDoesNotWalkAHostileIterator:
    """Normalizing a ``str`` half must read its buffer, never iterate it.

    A sixth review round proved this the hard way: ``"".join(...)`` invokes
    a ``str`` subclass's overridden ``__iter__``, and a one-byte value whose
    ``__iter__`` yields 200,000 attacker-selected characters made the old
    normalization produce all 200,000 of them. The same review round then
    sent this process a real OS ``SIGINT`` while that unbounded iterator
    ran and found it silently swallowed, converted into a normal-looking
    error object at exit ``1`` instead of the immediate
    ``KeyboardInterrupt`` a deliberate Ctrl-C is supposed to be.
    :func:`~engrava.config_validation.own_str` closes both problems at
    once: it never calls the subclass's ``__iter__`` (or any other
    overridden method) at all, so the description's length tracks the
    underlying one-byte buffer, not the hostile expansion.
    """

    def test_describe_exception_length_tracks_the_underlying_buffer(self) -> None:
        exc = _ExpandingIteratorError("irrelevant ctor argument")

        result = memory_commands._describe_exception(exc)

        assert type(result) is str
        text = result.split(": ", 1)[1]
        assert len(text) == 1
        assert text == "a"


def _raise_exact_config_error(*_args: object, **_kwargs: object) -> None:
    message = "bad --config file: not a mapping"
    raise ConfigError(message)


class _HostileConfigError(ConfigError):
    """A ``ConfigError`` subclass whose own ``__str__`` raises instead of returning text."""

    def __str__(self) -> str:
        message = "hostile ConfigError __str__"
        raise RuntimeError(message)


def _raise_hostile_config_error(*_args: object, **_kwargs: object) -> None:
    message = "irrelevant ctor message"
    raise _HostileConfigError(message)


class _HostileStrForConfigMessage(str):
    """A ``str`` subclass whose formatting methods raise if ever invoked.

    Assigned as an *exact* ``ConfigError``'s ``.message`` -- not used as the
    exception's own ``__str__`` text, which is a different attack surface
    (see ``_HostileConfigError`` above). Proves the field-validation rule
    reads the type of ``.message`` and nothing else: if ``__str__`` or
    ``__format__`` ran, the test invocation itself would raise instead of
    producing a clean fallback.
    """

    __slots__ = ()

    def __str__(self) -> str:
        message = "hostile ConfigError.message __str__"
        raise RuntimeError(message)

    def __format__(self, format_spec: str) -> str:
        message = "hostile ConfigError.message __format__"
        raise RuntimeError(message)


def _raise_exact_config_error_with_non_str_message(*_args: object, **_kwargs: object) -> None:
    raise ConfigError(_HostileStrForConfigMessage("looks like an ordinary message"))


class TestConfigErrorPreservesKindAndCodeAcrossSubclasses:
    """A ``ConfigError`` subclass keeps ``invalid_config`` / exit ``2`` too.

    ``ConfigError`` is a public library class (``engrava.config_validation``),
    so a third-party subclass is realistic, not theoretical, and reading
    ``str(exc)`` unprotected for one is not safe. An earlier fix re-raised a
    subclass into the generic boundary instead, which stopped it from
    corrupting the failure path but also silently downgraded it to
    ``unexpected_error`` / exit ``1`` -- breaking the categorical exit-``2``
    promise ``docs/cli.md`` makes for an invalid ``--config``. The class
    hierarchy is trustworthy even when a subclass's own attributes are not:
    :func:`~engrava.cli.memory_commands._resolve_for_command` now keeps
    ``invalid_config`` / exit ``2`` for *any* ``ConfigError``, exact or
    subclass, and only changes *how the message is built* -- an exact
    instance has its ``.message`` field read once and used only if
    ``type(message) is str``; a subclass, or an exact instance whose
    ``.message`` fails that check, gets a fixed literal message instead,
    with nothing read off it at all -- not even through the hardened
    :func:`~engrava.cli.memory_commands._describe_exception`, since that
    still calls ``str(exc)``, which a subclass fully controls.
    """

    def test_exact_config_error_still_produces_invalid_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands, "resolve_store_target", _raise_exact_config_error)
        runner = CliRunner()

        result = runner.invoke(cli, ["remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "invalid_config"
        assert "bad --config file" in payload["message"]

    def test_config_error_subclass_with_hostile_str_keeps_invalid_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands, "resolve_store_target", _raise_hostile_config_error)
        runner = CliRunner()

        result = runner.invoke(cli, ["remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "invalid_config"
        # A seventh review round found that even the hardened, generic
        # _describe_exception was not enough here: it still includes
        # str(exc), which the subclass fully controls, and a subclass built
        # to return a believable-looking fabricated diagnosis had that
        # fabrication reported as if this CLI had produced it. The message
        # is now a fixed literal -- nothing derived from the exception at
        # all, not even its type name -- so the hostile __str__'s own text
        # can never appear here, whether or not it happens to run. It says
        # the detail is *omitted*, not that a read of it failed: this path
        # never reads .message off a subclass instance at all.
        assert payload["message"] == memory_commands._CONFIG_ERROR_DETAILS_OMITTED_MESSAGE
        assert "_HostileConfigError" not in payload["message"]
        assert "hostile ConfigError __str__" not in payload["message"]

    def test_exact_config_error_with_non_str_message_falls_back_without_formatting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An *exact* ``ConfigError`` is not automatically safe: ``.message`` can

        still be a hostile value (H3 in the intake). ``type(message) is str``
        rejects the ``str`` subclass here before anything reads its text, so
        the fixed fallback comes back cleanly -- if ``__str__`` or
        ``__format__`` had run instead, this test invocation would raise.
        """
        monkeypatch.setattr(
            memory_commands,
            "resolve_store_target",
            _raise_exact_config_error_with_non_str_message,
        )
        runner = CliRunner()

        result = runner.invoke(cli, ["remember", "hi", "--json"])
        assert "Traceback" not in result.output
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "invalid_config"
        assert payload["message"] == memory_commands._CONFIG_ERROR_DETAILS_OMITTED_MESSAGE


async def _raise_exact_referential_integrity_error(*_args: object, **_kwargs: object) -> None:
    # `referenced_id` matches the `FROM` argument the tests below invoke
    # `link` with ("ghost-id") -- reflecting what the real store actually
    # does: a `ReferentialIntegrityError` it raises always carries back the
    # very id `create_edge()` was given, never an arbitrary third value.
    entity_type, column, referenced_id = "edge", "from_thought_id", "ghost-id"
    raise ReferentialIntegrityError(entity_type, column, referenced_id)


async def _raise_referential_integrity_error_with_mismatched_id(
    *_args: object, **_kwargs: object
) -> None:
    # A `.referenced_id` that does not match either endpoint this invocation
    # was given -- unrealistic for the real store, but exactly the shape a
    # hostile or buggy raiser could produce (H2 in the intake).
    entity_type, column, referenced_id = "edge", "from_thought_id", "some-other-id"
    raise ReferentialIntegrityError(entity_type, column, referenced_id)


class _StrSubclassColumn(str):
    """A ``str`` subclass used as ``.column`` -- fails ``type(x) is str``."""

    __slots__ = ()


async def _raise_referential_integrity_error_with_str_subclass_column(
    *_args: object, **_kwargs: object
) -> None:
    entity_type, referenced_id = "edge", "ghost-id"
    column = _StrSubclassColumn("from_thought_id")
    raise ReferentialIntegrityError(entity_type, column, referenced_id)


async def _raise_referential_integrity_error_with_unexpected_column(
    *_args: object, **_kwargs: object
) -> None:
    # A real `str`, but not one of the two column names this command could
    # actually violate.
    entity_type, column, referenced_id = "edge", "entity_type", "ghost-id"
    raise ReferentialIntegrityError(entity_type, column, referenced_id)


class _HostileReferentialIntegrityError(ReferentialIntegrityError):
    """A ``ReferentialIntegrityError`` subclass whose ``.column`` accessor raises."""

    def __init__(self) -> None:
        super().__init__("edge", "from_thought_id", "ghost-id")

    @property
    def column(self) -> str:
        message = "hostile .column accessor"
        raise RuntimeError(message)

    @column.setter
    def column(self, value: str) -> None:
        # Swallow the base __init__'s plain attribute assignment -- the
        # getter above is what makes this subclass hostile.
        del value


async def _raise_hostile_referential_integrity_error(*_args: object, **_kwargs: object) -> None:
    raise _HostileReferentialIntegrityError


class _StrRaisingReferentialIntegrityError(ReferentialIntegrityError):
    """A ``ReferentialIntegrityError`` subclass whose ``__str__`` itself raises.

    Distinct from ``_HostileReferentialIntegrityError`` below: that class's
    ``.column`` accessor is hostile, but its ``__str__`` (inherited,
    unoverridden) still succeeds, because the base class's constructor
    bakes the full message into ``Exception.args`` from its *local*
    constructor parameters rather than from ``self.column`` /
    ``self.referenced_id``. This class instead makes ``str()`` itself
    fail, so nothing about the missing reference can be read safely at
    all -- proving the resulting message does not fabricate a column or id
    it never actually obtained.
    """

    def __init__(self) -> None:
        super().__init__("edge", "from_thought_id", "ghost-id")

    def __str__(self) -> str:
        message = "hostile ReferentialIntegrityError __str__"
        raise RuntimeError(message)


async def _raise_str_raising_referential_integrity_error(*_args: object, **_kwargs: object) -> None:
    raise _StrRaisingReferentialIntegrityError


class TestReferentialIntegrityErrorPreservesKindAndCodeAcrossSubclasses:
    """A ``ReferentialIntegrityError`` subclass keeps ``missing_thought`` / exit ``4`` too.

    ``ReferentialIntegrityError`` is a public library class
    (``engrava.domain.exceptions``), so a third-party subclass is
    realistic, not theoretical, and reading ``.column`` / ``.referenced_id``
    unprotected for one is not safe. An earlier fix re-raised a subclass
    into the generic boundary instead, which stopped it from corrupting the
    failure path but also silently downgraded it to ``unexpected_error`` /
    exit ``1`` -- breaking the categorical exit-``4`` promise ``docs/cli.md``
    makes for a missing ``FROM`` / ``TO``. The class hierarchy is
    trustworthy even when a subclass's own attributes are not: ``link`` now
    keeps ``missing_thought`` / exit ``4`` for *any*
    ``ReferentialIntegrityError``, exact or subclass, and only changes *how
    the message is built* -- an exact instance has ``.column`` /
    ``.referenced_id`` read once and validated (exact ``str`` types, a real
    column name, and an id matching this invocation's own ``FROM``/``TO``)
    before the *invocation's own* endpoint value is used to name which one
    is missing; a subclass, or an exact instance that fails that
    validation, gets a fixed literal message instead, with nothing read off
    it at all -- not even through the hardened
    :func:`~engrava.cli.memory_commands._describe_exception`, since that
    still calls ``str(exc)``, which a subclass fully controls -- and its
    message is phrased so it stays true without claiming to know which
    endpoint, since that cannot be read safely off a subclass.
    """

    def test_exact_referential_integrity_error_still_produces_missing_thought(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            SqliteEngravaCore, "create_edge", _raise_exact_referential_integrity_error
        )
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "ghost-id", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 4
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "missing_thought"
        assert "ghost-id" in payload["message"]

    def test_referential_integrity_error_with_mismatched_id_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``.referenced_id`` that matches neither ``FROM`` nor ``TO`` falls back.

        Realistic for a hostile or buggy raiser (H2), not for the real
        store. Proves the equality rule, not just the type/column checks.
        """
        monkeypatch.setattr(
            SqliteEngravaCore,
            "create_edge",
            _raise_referential_integrity_error_with_mismatched_id,
        )
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "a", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 4
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "missing_thought"
        assert payload["message"] == memory_commands._MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE
        assert "some-other-id" not in payload["message"]

    def test_referential_integrity_error_with_str_subclass_column_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``.column`` that is a ``str`` *subclass* -- not exactly ``str`` -- falls back.

        Proves the ``type(column) is str`` gate: the value reads back as
        ``"from_thought_id"`` under ``==``, but is rejected on type alone,
        before any comparison or interpolation would touch it.
        """
        monkeypatch.setattr(
            SqliteEngravaCore,
            "create_edge",
            _raise_referential_integrity_error_with_str_subclass_column,
        )
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "ghost-id", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 4
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "missing_thought"
        assert payload["message"] == memory_commands._MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE

    def test_referential_integrity_error_with_unexpected_column_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``.column`` that is a real ``str`` but not a real edge column name falls back."""
        monkeypatch.setattr(
            SqliteEngravaCore,
            "create_edge",
            _raise_referential_integrity_error_with_unexpected_column,
        )
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "ghost-id", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 4
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "missing_thought"
        assert payload["message"] == memory_commands._MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE
        assert "entity_type" not in payload["message"]

    def test_referential_integrity_error_subclass_with_hostile_column_keeps_missing_thought(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            SqliteEngravaCore, "create_edge", _raise_hostile_referential_integrity_error
        )
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "a", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 4
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "missing_thought"
        # The message is a fixed literal -- nothing read off the subclass at
        # all, not even its type name -- so it does not matter that this
        # particular subclass's hostile half is .column rather than __str__.
        assert payload["message"] == memory_commands._MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE
        assert "_HostileReferentialIntegrityError" not in payload["message"]
        assert "hostile .column accessor" not in payload["message"]

    def test_referential_integrity_error_subclass_with_hostile_str_keeps_missing_thought(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            SqliteEngravaCore, "create_edge", _raise_str_raising_referential_integrity_error
        )
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "link", "a", "b", "--type", "ASSOCIATED", "--json"]
        )
        assert "Traceback" not in result.output
        assert result.exit_code == 4
        payload = json.loads(_last_line(result.output))
        assert payload["error"] == "missing_thought"
        # The message is a fixed literal -- neither .column, .referenced_id,
        # str(exc), nor the type name is read -- so it must not claim to
        # know which endpoint is missing, nor surface the hostile __str__'s
        # own chosen text.
        assert payload["message"] == memory_commands._MISSING_THOUGHT_ENDPOINT_OMITTED_MESSAGE
        assert "_StrRaisingReferentialIntegrityError" not in payload["message"]
        assert "ghost-id" not in payload["message"]
        assert "from_thought_id" not in payload["message"]
        assert "hostile ReferentialIntegrityError __str__" not in payload["message"]


class TestErrorJsonEscapesUnicodeLineSeparators:
    """The error object must not embed a literal Unicode line separator.

    ``json.dumps(..., ensure_ascii=False)`` leaves U+0085/U+2028/U+2029
    literal in its output -- they are not JSON control characters, so
    nothing about the JSON grammar itself escapes them. A ``--db`` path
    containing one produced a ``database_not_found`` object whose last
    fragment, after a consumer's ``str.splitlines()`` (which treats those
    three code points as line breaks, unlike a strict ``"\\n"`` split), was
    not valid JSON. ``ensure_ascii=True`` (see
    ``engrava.cli.memory_commands._emit_and_exit``) removes the hazard by
    escaping them to ``\\uXXXX`` instead of leaving them literal.
    """

    def test_a_line_separator_in_the_db_path_does_not_split_the_json(self, tmp_path: Path) -> None:
        db = tmp_path / f"line{chr(0x2028)}sep.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "recall", "anything", "--json"])
        assert result.exit_code == 3, result.output

        raw = result.output
        assert chr(0x2028) not in raw, "the raw U+2028 byte must not survive into the JSON"

        strict_lines = [line for line in raw.split("\n") if line]
        payload = json.loads(strict_lines[-1])
        assert payload["error"] == "database_not_found"

        splitlines_lines = [line for line in raw.splitlines() if line]
        assert json.loads(splitlines_lines[-1]) == payload


class TestMutationSurvivorsFromTheReviewAreNowClosedGaps:
    """Three "advertised protections" a review round found were mutation survivors.

    Independently re-run and confirmed as real, pre-existing coverage gaps
    (not an artefact of a failed patch apply): nothing in this test module
    checked the command-name prefix on an ``unexpected_error`` message, the
    stack a ``--verbose`` debug log record carries, or the ``--verbose``
    resolution line memory verbs share with every other CLI command. Each
    is closed here directly rather than left as a reported-but-unaddressed
    finding.
    """

    def test_unexpected_error_message_is_prefixed_with_the_command_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_unanticipated)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "remember", "hi", "--json"])
        payload = json.loads(_last_line(result.output))
        # The resolved database sits between the command name and
        # "unexpected" -- naming *which* database is what this whole
        # boundary now exists to add over a bare "remember: unexpected ...".
        assert payload["message"].startswith(f"remember: {db}: unexpected ")

    def test_unexpected_error_is_logged_with_a_frame_stack_under_verbose(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``--verbose`` logs the caught exception's description and frame stack.

        Not ``exc_info=True`` any more (see ``memory_commands._error_boundary``):
        that made CPython render the exception a second time, and a real
        Ctrl-C during that second render was swallowed by the standard
        library's own traceback formatter. The record's own message now
        carries filename:line-in-function frames built from
        ``exc.__traceback__`` via ``traceback.walk_tb`` -- this test asserts
        the frame where the exception was actually raised is present, and
        that the record does not carry stdlib ``exc_info`` (which would
        indicate the old, second-formatting shape).
        """
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_unanticipated)
        db = tmp_path / "m.db"
        runner = CliRunner()

        # Attached directly to this module's own logger rather than relying
        # on propagation to caplog's root-level handler: `--verbose`
        # deliberately sets the `engrava` package logger's `propagate` to
        # `False` for the lifetime of one invocation (see
        # `_configure_verbose_logging` in `main.py`), so a record from
        # `engrava.cli.memory_commands` never reaches a handler on the root
        # logger while `--verbose` is in effect.
        target_logger = logging.getLogger("engrava.cli.memory_commands")
        target_logger.addHandler(caplog.handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            result = runner.invoke(cli, ["--db", str(db), "--verbose", "remember", "hi", "--json"])
        finally:
            target_logger.removeHandler(caplog.handler)

        assert result.exit_code == 1
        matching = [r for r in caplog.records if "_NobodyAnticipatedError" in r.getMessage()]
        assert matching, "expected the boundary's own debug log record"
        record = matching[0]
        assert record.exc_info is None, (
            "the debug record must not carry exc_info -- that would mean "
            "logging formats the exception a second time"
        )
        message = record.getMessage()
        # The frame where the exception was actually raised is on the stack.
        assert "in _raise_unanticipated" in message
        assert "test_cli_memory_verbs_error_boundary.py:" in message
        assert "caught exception's stack" in message
        assert "not a full exception-chain rendering" in message


class TestVerboseReportsTheResolvedDatabase:
    """``--verbose`` echoes the resolution decision -- no test checked this at all.

    :func:`engrava.cli.memory_commands._report_resolution` is only ever
    called under ``cfg.verbose``; nothing in this module (or
    ``test_cli_memory_verbs.py``) previously asserted its output existed,
    which is exactly why disabling it in review left every relevant test
    green.
    """

    def test_remember_verbose_reports_the_resolved_database(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "--verbose", "remember", "hello"])
        assert result.exit_code == 0, result.output
        assert f"Resolved database: {db} (source: --db)" in result.output


class _KeyboardInterruptOnStrError(Exception):
    """An exception whose ``__str__`` raises ``KeyboardInterrupt``, like a real Ctrl-C mid-read."""

    def __str__(self) -> str:
        raise KeyboardInterrupt


class _SystemExitOnStrError(Exception):
    """An exception whose ``__str__`` raises ``SystemExit``."""

    def __str__(self) -> str:
        raise SystemExit(1)


class _KeyboardInterruptNameMeta(type):
    """A metaclass whose ``__name__`` read raises ``KeyboardInterrupt``, not just ``str()``."""

    @property
    def __name__(cls) -> str:
        raise KeyboardInterrupt


class _KeyboardInterruptOnTypeNameError(Exception, metaclass=_KeyboardInterruptNameMeta):
    """An exception whose *type name* read raises ``KeyboardInterrupt``."""


class _OrdinaryRuntimeErrorOnStrError(Exception):
    """An exception whose ``__str__`` raises a plain ``RuntimeError``, not a control-flow signal."""

    def __str__(self) -> str:
        message = "ordinary failure raised from __str__"
        raise RuntimeError(message)


def _raise_keyboard_interrupt_on_str(*_args: object, **_kwargs: object) -> None:
    message = "irrelevant ctor argument"
    raise _KeyboardInterruptOnStrError(message)


class TestDescribeExceptionReraisesControlFlowSignals:
    """A seventh review round found a hostile ``__str__`` raising ``KeyboardInterrupt`` swallowed.

    The round delivered a real OS ``SIGINT`` to the process during an
    endless hostile ``__str__`` and found it converted into an
    ordinary-looking error object at exit ``1`` instead of the immediate
    ``KeyboardInterrupt`` a deliberate Ctrl-C is supposed to be --
    :func:`~engrava.cli.memory_commands._describe_exception`'s two guards
    were both ``except BaseException``, which converts a real interrupt
    exactly like any ordinary exception. Both guards now re-raise
    ``KeyboardInterrupt`` and ``SystemExit`` immediately instead of
    converting them; every ordinary ``Exception`` -- including one raised
    from ``__str__`` or a hostile metaclass's ``__name__`` -- still converts
    to the documented placeholder exactly as before.
    """

    def test_describe_exception_reraises_a_keyboard_interrupt_from_str(self) -> None:
        exc = _KeyboardInterruptOnStrError("irrelevant")

        with pytest.raises(KeyboardInterrupt):
            memory_commands._describe_exception(exc)

    def test_describe_exception_reraises_a_system_exit_from_str(self) -> None:
        exc = _SystemExitOnStrError("irrelevant")

        with pytest.raises(SystemExit):
            memory_commands._describe_exception(exc)

    def test_describe_exception_reraises_a_keyboard_interrupt_from_type_name(self) -> None:
        exc = _KeyboardInterruptOnTypeNameError("irrelevant")

        with pytest.raises(KeyboardInterrupt):
            memory_commands._describe_exception(exc)

    def test_describe_exception_still_converts_an_ordinary_exception_from_str(self) -> None:
        exc = _OrdinaryRuntimeErrorOnStrError("irrelevant")

        result = memory_commands._describe_exception(exc)

        assert result == "_OrdinaryRuntimeErrorOnStrError: <str() raised>"

    def test_remember_lets_a_keyboard_interrupt_reach_clicks_own_abort_handling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raised ``KeyboardInterrupt`` now reaches Click's own interrupt handling, not ours.

        ``_error_boundary`` catches only ``Exception``, and (after this fix)
        ``_describe_exception`` re-raises rather than converts a
        ``KeyboardInterrupt`` too, so it propagates all the way up through
        ``asyncio.run()`` and the command callback to Click's own
        ``BaseCommand.main()`` -- the same top-level ``except (EOFError,
        KeyboardInterrupt)`` handling that turns a real terminal Ctrl-C into
        ``Abort`` -- rather than being disguised as this CLI's own
        ``unexpected_error`` JSON object first. Confirmed against the
        pre-fix guard (plain ``except BaseException``) that the same
        scenario instead produces exactly that JSON object, with the
        exception's type name and ``<str() raised>`` placeholder baked into
        ``message`` as if it had been safely diagnosed.
        """
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_keyboard_interrupt_on_str)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "remember", "hi", "--json"])

        # Click's own Abort handling still exits non-zero via a fresh
        # SystemExit(1) -- the exit code alone does not distinguish the fix
        # from the bug it replaces. What distinguishes them is the output:
        # no JSON error object at all, and Click's own "Aborted!" message
        # instead of a fabricated diagnosis of the exception.
        assert isinstance(result.exception, SystemExit)
        assert "Aborted!" in result.output
        assert "schema" not in result.output
        assert "unexpected_error" not in result.output
        assert "_KeyboardInterruptOnStrError" not in result.output


_VERBOSE_REAL_SIGINT_SCRIPT = """
import os
import signal
import time

import engrava.cli.memory_commands as memory_commands


class _SelfInterruptingStrError(Exception):
    \"\"\"Behaves like an ordinary exception the *first* time __str__ runs --

    that call is memory_commands._describe_exception's own, already-guarded
    read, used to build the unexpected_error message either way. Only a
    *second* call -- which, before this fix, was Python's own traceback
    formatter calling str(exc) again to render `exc_info=True` -- prints a
    marker this test looks for, then sends this process a real OS SIGINT
    and spins so the interpreter has every chance to deliver it before this
    call returns.
    \"\"\"

    _calls = 0

    def __str__(self) -> str:
        type(self)._calls += 1
        if type(self)._calls == 2:
            print("REACHED_SECOND_STR_CALL", flush=True)
            os.kill(os.getpid(), signal.SIGINT)
            for _ in range(300):
                time.sleep(0.01)
        return "irrelevant text"


def _raise_it(*_args, **_kwargs):
    raise _SelfInterruptingStrError


memory_commands.uuid.uuid4 = _raise_it

from engrava.cli.main import main

main()
"""


class TestVerboseNoLongerCallsTheExceptionASecondTime:
    """The specific window a real OS ``SIGINT`` used to be swallowed in is now gone.

    Before this fix, the debug log call passed ``exc_info=True``, which made
    CPython's own traceback formatter call ``str(exc)`` a *second* time (via
    ``traceback._safe_string``'s bare ``except:`` in the standard library)
    to build the traceback's exception-message line, entirely outside
    :func:`~engrava.cli.memory_commands._describe_exception`'s guards. A
    real interrupt landing during that second call was swallowed by that
    bare ``except``, and the command finished as an ordinary
    ``unexpected_error`` JSON object at exit ``1`` instead of aborting --
    confirmed live against ``0da258c`` (see the delivery report for that
    run's output; reproducing it here would require checking out that
    commit inside this test, which this module does not do).

    This test's hostile exception succeeds normally on its *first* call --
    the one read ``_describe_exception`` performs either way, needed for
    the message regardless of ``--verbose`` -- and only self-signals on a
    *second* call. Against the fixed code that second call never happens at
    all: the ``--verbose`` stack is built from ``exc.__traceback__`` frame
    metadata, not by calling ``str(exc)`` again, so there is no longer a
    second read for a real interrupt to land inside. The marker the second
    call would print never appears, and the command completes as an
    ordinary, correctly classified error -- not because an interrupt was
    survived, but because the vulnerable second read no longer exists to
    receive one.
    """

    def test_second_str_call_marker_never_appears_under_verbose(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"

        result = _run_subprocess(
            [
                "-c",
                _VERBOSE_REAL_SIGINT_SCRIPT,
                "--db",
                str(db),
                "--verbose",
                "remember",
                "hi",
                "--json",
            ]
        )

        # If a future change reintroduced a second, unguarded read (e.g. an
        # `exc_info=True` regression), the marker below would print, this
        # exception would self-signal, and this assertion would catch the
        # regression immediately regardless of how the signal then behaved.
        assert "REACHED_SECOND_STR_CALL" not in result.stdout
        assert result.returncode == 1, result.stderr
        assert "Aborted!" not in result.stderr
        payload = json.loads(_last_line(result.stderr))
        assert payload["error"] == "unexpected_error"
        assert "_SelfInterruptingStrError" in payload["message"]


_FORMATTER_CALL_LOG: list[tuple[str, str]] = []


class _RecordingFormatterError(Exception):
    """Records every call to its own ``__str__`` / ``__format__``.

    Used, chained via ``__cause__`` / ``__context__``, to prove ``--verbose``
    adds no further call to the original exception's formatter or to any
    cause/context formatter -- not just that a real interrupt would survive
    one, but that the call is not made at all. A manual comparison against
    ``0da258c`` recorded the counts this closes: three total calls
    (``main`` twice, ``cause`` once) before this fix, one (``main`` once)
    after, with or without ``--verbose``.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        super().__init__(f"{label} ctor text")

    def __str__(self) -> str:
        _FORMATTER_CALL_LOG.append(("str", self.label))
        return f"{self.label}-str"

    def __format__(self, format_spec: str) -> str:
        _FORMATTER_CALL_LOG.append(("format", self.label))
        return f"{self.label}-format"


def _raise_chained_recording_error(*_args: object, **_kwargs: object) -> None:
    cause = _RecordingFormatterError("cause")
    context = _RecordingFormatterError("context")
    main = _RecordingFormatterError("main")
    main.__cause__ = cause
    main.__context__ = context
    raise main


class TestVerboseCallsNoFormatterMoreThanOnceTotal:
    """Turning on ``--verbose`` must not add a single extra formatter call.

    Proof property 3 from the direction review: no further call to the
    original exception's formatter, nor to any cause/context/group-child
    formatter, and stack rendering inspects no local values. The one call
    that does happen either way -- :func:`~engrava.cli.memory_commands
    ._describe_exception`'s own guarded ``str(exc)``, needed to build the
    message regardless of ``--verbose`` -- is the baseline; this asserts
    nothing beyond it happens, with ``--verbose`` on or off, and that
    ``__cause__`` / ``__context__`` are never touched at all.
    """

    def test_verbose_on_calls_the_formatter_exactly_once_total(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FORMATTER_CALL_LOG.clear()
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_chained_recording_error)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "--verbose", "remember", "hi", "--json"])

        assert result.exit_code == 1
        assert _FORMATTER_CALL_LOG == [("str", "main")]

    def test_verbose_off_calls_the_formatter_exactly_once_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same exception, ``--verbose`` omitted: the call count must not differ.

        Confirms the one remaining call is the message-building read that
        always happens, not something ``--verbose`` newly triggers.
        """
        _FORMATTER_CALL_LOG.clear()
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_chained_recording_error)
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(cli, ["--db", str(db), "remember", "hi", "--json"])

        assert result.exit_code == 1
        assert _FORMATTER_CALL_LOG == [("str", "main")]


_CLEANUP_LOG_FORMATTER_CALL_COUNTS_SCRIPT = """
import json

_counts = {}


def _record(label):
    _counts[label] = _counts.get(label, 0) + 1


class _CountingChild(Exception):
    def __str__(self):
        _record("group_child")
        return "child-str"


class _CountingGroup(ExceptionGroup):
    def __str__(self):
        _record("exception_group")
        return super().__str__()


class _CountingOriginal(Exception):
    def __str__(self):
        _record("original")
        return "original-str"


class _CountingClose(Exception):
    def __str__(self):
        _record("close")
        return "close-str"


import atexit


def _dump():
    print("COUNTS:" + json.dumps(_counts), flush=True)


atexit.register(_dump)

import engrava.cli.memory_commands as memory_commands
from engrava import SqliteEngravaCore

_orig_close = SqliteEngravaCore.close


async def _boom_close(self):
    # Complete the real close first -- see _LINK_WITH_BROKEN_BARE_CLOSE_SCRIPT
    # above for why a monkeypatch that only raises, without ever letting the
    # real close run, would leak the connection.
    await _orig_close(self)
    raise _CountingClose("close")


SqliteEngravaCore.close = _boom_close


def _raise_original(*_args, **_kwargs):
    # A secondary close failure while this exception is still propagating --
    # the review round's exact repro -- with a __cause__ chain deep enough to
    # prove nothing beyond the close exception's own frames is ever touched:
    # an exception group and a group child, each independently instrumented.
    original = _CountingOriginal("orig")
    original.__cause__ = _CountingGroup("grp", [_CountingChild("c")])
    raise original


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


_CLEANUP_LOG_CLOSE_KEYBOARD_INTERRUPT_SCRIPT = """
import engrava.cli.memory_commands as memory_commands
from engrava import SqliteEngravaCore


class _KeyboardInterruptOnCloseStrError(Exception):
    \"\"\"Its own __str__ raises KeyboardInterrupt, like a real Ctrl-C mid-read.

    Before this fix, the cleanup log's ``exc_info=True`` made Python's own
    traceback formatter call this exception's ``__str__`` to render it, and
    that formatter wraps its own rendering in a bare ``except`` -- a
    ``KeyboardInterrupt`` raised there was absorbed instead of propagating.
    ``_describe_exception`` reads this exception's ``__str__`` exactly once
    (the marker printed below), under a guard that re-raises
    ``KeyboardInterrupt``/``SystemExit`` immediately rather than converting
    them -- see its own docstring.
    \"\"\"

    def __str__(self) -> str:
        print("REACHED_CLOSE_STR_CALL", flush=True)
        raise KeyboardInterrupt


_orig_close = SqliteEngravaCore.close


async def _boom_close(self):
    await _orig_close(self)
    raise _KeyboardInterruptOnCloseStrError("close")


SqliteEngravaCore.close = _boom_close


def _raise_original(*_args, **_kwargs):
    raise RuntimeError("original body failure")


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


_CLEANUP_LOG_SYSTEM_EXIT_SCRIPT = """
import engrava.cli.memory_commands as memory_commands
from engrava import SqliteEngravaCore


class _SystemExitingCloseError(Exception):
    \"\"\"Its own __str__ raises SystemExit(37) instead of returning text.

    Before this fix, a formatter raising ``SystemExit`` while rendering this
    exception for ``exc_info=True`` was absorbed by the standard library's
    own traceback-formatting code (it substitutes a fixed placeholder rather
    than letting the SystemExit propagate), so the command still exited `1`
    instead of `37`.
    \"\"\"

    def __str__(self) -> str:
        print("REACHED_CLOSE_STR_CALL", flush=True)
        raise SystemExit(37)


_orig_close = SqliteEngravaCore.close


async def _boom_close(self):
    await _orig_close(self)
    raise _SystemExitingCloseError("close")


SqliteEngravaCore.close = _boom_close


def _raise_original(*_args, **_kwargs):
    raise RuntimeError("original body failure")


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


_BARE_CLEANUP_LOG_FORMATTER_CALL_COUNTS_SCRIPT = """
import json

_counts = {}


def _record(label):
    _counts[label] = _counts.get(label, 0) + 1


class _CountingChild(Exception):
    def __str__(self):
        _record("group_child")
        return "child-str"


class _CountingGroup(ExceptionGroup):
    def __str__(self):
        _record("exception_group")
        return super().__str__()


class _CountingOriginal(Exception):
    def __str__(self):
        _record("original")
        return "original-str"


class _CountingClose(Exception):
    def __str__(self):
        _record("close")
        return "close-str"


import atexit


def _dump():
    print("COUNTS:" + json.dumps(_counts), flush=True)


atexit.register(_dump)

import aiosqlite
import engrava.cli.memory_commands as memory_commands

_orig_close = aiosqlite.Connection.close


async def _boom_close(self):
    await _orig_close(self)
    raise _CountingClose("close")


aiosqlite.Connection.close = _boom_close


def _raise_original(*_args, **_kwargs):
    # Chained the same way as the --config-tier script above, to prove the
    # group and its child stay at zero here too -- see
    # TestBareTierCleanupLogNowSharesTheSameFix's own docstring for why this
    # tier's "original" count does not double the way the --config tier's
    # once did: the close here runs as a separately scheduled, shielded
    # task (main._close_quietly), so it never inherits the original
    # exception as its own __context__.
    original = _CountingOriginal("orig")
    original.__cause__ = _CountingGroup("grp", [_CountingChild("c")])
    raise original


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


_BARE_CLEANUP_LOG_CLOSE_KEYBOARD_INTERRUPT_SCRIPT = """
import aiosqlite
import engrava.cli.memory_commands as memory_commands


class _KeyboardInterruptOnCloseStrError(Exception):
    \"\"\"Its own __str__ raises KeyboardInterrupt, like a real Ctrl-C mid-read.

    ``main._close_quietly`` still passed ``exc_info=True`` here before this
    fix -- the bare/default store tier's own copy of the same defect
    already fixed at the --config tier (see the scripts above).
    \"\"\"

    def __str__(self) -> str:
        print("REACHED_CLOSE_STR_CALL", flush=True)
        raise KeyboardInterrupt


_orig_close = aiosqlite.Connection.close


async def _boom_close(self):
    await _orig_close(self)
    raise _KeyboardInterruptOnCloseStrError("close")


aiosqlite.Connection.close = _boom_close


def _raise_original(*_args, **_kwargs):
    raise RuntimeError("original body failure")


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


_BARE_CLEANUP_LOG_CLOSE_SYSTEM_EXIT_SCRIPT = """
import aiosqlite
import engrava.cli.memory_commands as memory_commands


class _SystemExitOnCloseStrError(Exception):
    \"\"\"Its own __str__ raises SystemExit(37) instead of returning text.\"\"\"

    def __str__(self) -> str:
        print("REACHED_CLOSE_STR_CALL", flush=True)
        raise SystemExit(37)


_orig_close = aiosqlite.Connection.close


async def _boom_close(self):
    await _orig_close(self)
    raise _SystemExitOnCloseStrError("close")


aiosqlite.Connection.close = _boom_close


def _raise_original(*_args, **_kwargs):
    raise RuntimeError("original body failure")


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


_BARE_CLOSE_PERMISSION_ERROR_SCRIPT = """
import aiosqlite
import engrava.cli.memory_commands as memory_commands

_orig_close = aiosqlite.Connection.close


async def _boom_close(self):
    await _orig_close(self)
    raise PermissionError(13, "Permission denied")


aiosqlite.Connection.close = _boom_close


def _raise_original(*_args, **_kwargs):
    raise RuntimeError("original body failure")


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


_CONFIG_CLOSE_PERMISSION_ERROR_SCRIPT = """
import engrava.cli.memory_commands as memory_commands
from engrava import SqliteEngravaCore

_orig_close = SqliteEngravaCore.close


async def _boom_close(self):
    await _orig_close(self)
    raise PermissionError(13, "Permission denied")


SqliteEngravaCore.close = _boom_close


def _raise_original(*_args, **_kwargs):
    raise RuntimeError("original body failure")


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""


# A template used by both real-SIGINT scripts below, delivered via
# _run_subprocess_with_external_sigint (a genuine external SIGINT, not a
# self-delivered one and not a hostile __str__ raising a control-flow signal
# directly). The close exception's own __str__ is an entirely ordinary
# description that merely prints a marker once reached and then blocks for
# real wall-clock time on an ordinary time.sleep (not a signal-raising
# trick), purely so the parent process has a well-defined window in which to
# deliver the real SIGINT before this synchronous call returns. The two
# format placeholders below let the same script body drive both cleanup
# tiers; only which callable gets monkeypatched differs between them.
_REAL_SIGINT_DURING_CLOSE_DESCRIPTION_SCRIPT_TEMPLATE = """
import time
import engrava.cli.memory_commands as memory_commands
{close_patch_imports}


class _SlowDescribingCloseError(Exception):
    \"\"\"An ordinary close failure whose own __str__ takes real wall-clock time.

    Not hostile -- it never raises anything itself. The sleep below exists
    only to give the parent test process (which is watching this process's
    stdout for the marker) a real window to deliver an external SIGINT
    before this synchronous call returns, with no suspension point anywhere
    in sight until the caller's own cleanup checkpoint.
    \"\"\"

    def __str__(self):
        print("REACHED_CLOSE_STR_CALL", flush=True)
        time.sleep(1.0)
        return "close failed: slow description"


{close_patch_body}


def _raise_original(*_args, **_kwargs):
    raise RuntimeError("original body failure")


memory_commands.uuid.uuid4 = _raise_original

from engrava.cli.main import main

main()
"""

_BARE_REAL_SIGINT_DURING_CLOSE_DESCRIPTION_SCRIPT = (
    _REAL_SIGINT_DURING_CLOSE_DESCRIPTION_SCRIPT_TEMPLATE.format(
        close_patch_imports="import aiosqlite",
        close_patch_body=(
            "_orig_close = aiosqlite.Connection.close\n\n\n"
            "async def _boom_close(self):\n"
            "    await _orig_close(self)\n"
            "    raise _SlowDescribingCloseError('close')\n\n\n"
            "aiosqlite.Connection.close = _boom_close"
        ),
    )
)

_CONFIG_REAL_SIGINT_DURING_CLOSE_DESCRIPTION_SCRIPT = (
    _REAL_SIGINT_DURING_CLOSE_DESCRIPTION_SCRIPT_TEMPLATE.format(
        close_patch_imports="from engrava import SqliteEngravaCore",
        close_patch_body=(
            "_orig_close = SqliteEngravaCore.close\n\n\n"
            "async def _boom_close(self):\n"
            "    await _orig_close(self)\n"
            "    raise _SlowDescribingCloseError('close')\n\n\n"
            "SqliteEngravaCore.close = _boom_close"
        ),
    )
)


class TestCleanupLogNoLongerReformatsThePropagatingException:
    """A secondary close failure must not re-render the exception it is cleaning up after.

    A later review round found ``_opened_full_store``'s cleanup-failure log
    (``logger.warning("Error closing store during cleanup", exc_info=True)``)
    was not the harmless, unrelated second exception it looked like: the body
    exception it is cleaning up after is still the *active* exception at that
    point (this runs inside ``except BaseException:``), so Python attaches it
    as the close exception's own ``__context__``, and ``exc_info=True`` asked
    the standard library's traceback formatter to render that whole chain --
    calling ``__str__`` on the close exception, on the original exception a
    *second* time (:func:`~engrava.cli.memory_commands._describe_exception`
    already reads it once, downstream, to build the final message), and on
    any exception-group children attached to either. Measured live against
    this module's own predecessor (commit ``bdf78a9``) with a secondary close
    failure whose original exception carried a one-child exception group:
    ``{"close": 1, "original": 2, "exception_group": 1, "group_child": 1}``,
    identical with and without ``--verbose`` (the warning was never gated
    behind ``--verbose`` in the first place). The fix, ``_frame_only_stack``
    shared with :func:`~engrava.cli.memory_commands._error_boundary`, reads
    only frame metadata, so the group and its child stay at zero calls and
    the original exception's own formatter is read only the one time the
    boundary always reads it downstream.

    **A still later round found dropping the close exception's own count to
    zero was itself a regression** -- frame metadata says *where* closing
    failed, never *why*. The fix now calls
    :func:`~engrava.cli.exception_reporting._describe_exception` once on the
    close exception too, the same single, guarded, non-absorbing attempt
    already made for the original exception, so ``"close"`` is ``1`` again
    here -- not a second render of anything, the *first* and only read of
    the close exception's own formatting.
    """

    def test_close_original_group_and_child_formatter_call_counts(self, tmp_path: Path) -> None:
        for index, extra_args in enumerate(([], ["--verbose"])):
            db = tmp_path / f"m{index}.db"
            config_path = tmp_path / f"engrava{index}.yaml"
            config_path.write_text(f"database:\n  path: {db}\n", encoding="utf-8")

            result = _run_subprocess(
                [
                    "-c",
                    _CLEANUP_LOG_FORMATTER_CALL_COUNTS_SCRIPT,
                    "--config",
                    str(config_path),
                    *extra_args,
                    "remember",
                    "hi",
                ]
            )

            assert result.returncode == 1, result.stderr
            counts_line = next(
                line for line in result.stdout.splitlines() if line.startswith("COUNTS:")
            )
            counts = json.loads(counts_line[len("COUNTS:") :])
            assert counts == {"original": 1, "close": 1}, (extra_args, counts, result.stderr)

    def test_keyboard_interrupt_from_close_str_reaches_clicks_own_abort_handling(
        self, tmp_path: Path
    ) -> None:
        """A ``KeyboardInterrupt`` raised while describing the close exception must escape.

        A real OS ``SIGINT`` delivered during this exact window was also
        verified live against this fix (see the delivery report for this
        change), but ``asyncio.run()``'s own signal-delivery timing is not
        reliable enough on this platform to assert on in an automated test
        -- a ``time.sleep`` polling loop woken by a self-delivered
        ``os.kill(..., SIGINT)`` inside this exact call stack was observed
        to defer the interrupt past process exit in both the pre-fix and
        post-fix code, which is a CPython/asyncio scheduling property, not
        something either version of this code controls. This exercises the
        same guarded read deterministically instead: ``__str__`` raising
        ``KeyboardInterrupt`` directly, exactly like
        :class:`TestDescribeExceptionReraisesControlFlowSignals`'s existing
        coverage of the *original* exception's own describe call, applied
        here to the *close* exception's.
        """
        db = tmp_path / "m.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(f"database:\n  path: {db}\n", encoding="utf-8")

        result = _run_subprocess(
            [
                "-c",
                _CLEANUP_LOG_CLOSE_KEYBOARD_INTERRUPT_SCRIPT,
                "--config",
                str(config_path),
                "remember",
                "hi",
            ]
        )

        assert "REACHED_CLOSE_STR_CALL" in result.stdout
        assert result.returncode == 1, result.stderr
        assert "Aborted!" in result.stderr
        assert "unexpected_error" not in result.stderr
        assert "unexpected RuntimeError" not in result.stderr

    def test_system_exit_from_close_formatter_now_reaches_the_caller(self, tmp_path: Path) -> None:
        """A formatter raising ``SystemExit`` during the close description must escape too.

        Confirmed live against ``bdf78a9``: the standard library's traceback
        formatter absorbed the ``SystemExit(37)`` and substituted a fixed
        placeholder, so the command exited ``1`` rather than ``37``. Against
        the fixed code, ``_describe_exception`` reads the close exception's
        ``__str__`` once (the marker appears), and this time that raised
        ``SystemExit(37)`` propagates all the way out instead of being
        substituted for anything.
        """
        db = tmp_path / "m.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(f"database:\n  path: {db}\n", encoding="utf-8")

        result = _run_subprocess(
            [
                "-c",
                _CLEANUP_LOG_SYSTEM_EXIT_SCRIPT,
                "--config",
                str(config_path),
                "remember",
                "hi",
            ]
        )

        assert "REACHED_CLOSE_STR_CALL" in result.stdout
        assert result.returncode == 37, result.stderr
        assert "unexpected RuntimeError" not in result.stderr

    def test_an_ordinary_close_failure_now_shows_why_not_just_where(self, tmp_path: Path) -> None:
        """The ``--config`` tier had only ``__str__``-hostile proofs of this, never a plain one.

        Mirrors the bare tier's own
        ``TestBareTierCleanupLogNowSharesTheSameFix``
        ``.test_an_ordinary_close_failure_now_shows_why_not_just_where``:
        an ordinary ``PermissionError`` closing the store must still show
        *why* (its own type and text), not just *where* (the frame-only
        stack), and the pending-cancellation checkpoint added for the real
        ``SIGINT`` fix must not cost that diagnostic.
        """
        db = tmp_path / "m.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(f"database:\n  path: {db}\n", encoding="utf-8")

        result = _run_subprocess(
            [
                "-c",
                _CONFIG_CLOSE_PERMISSION_ERROR_SCRIPT,
                "--config",
                str(config_path),
                "remember",
                "hi",
            ]
        )

        assert result.returncode == 1, result.stderr
        assert "PermissionError: [Errno 13] Permission denied" in result.stderr
        assert "Error closing store during cleanup" in result.stderr

    @pytest.mark.parametrize("extra_args", [[], ["--verbose"]], ids=["default", "verbose"])
    def test_real_external_sigint_during_close_description_now_aborts(
        self, tmp_path: Path, extra_args: list[str]
    ) -> None:
        """A real, externally delivered ``SIGINT`` during this exact window must now abort.

        A later verification round found this absorbed: with no suspension
        point between building the close-exception warning and re-raising
        the original exception, ``asyncio.run()``'s pending task-cancellation
        request (see :func:`_run_subprocess_with_external_sigint`'s own
        docstring) was simply dropped, and the process finished the warning,
        emitted the original ``unexpected_error`` JSON, and exited ``1`` --
        confirmed live against this exact commit before the fix, for both
        ``--verbose`` on and off. The fix adds a genuine ``await
        asyncio.sleep(0)`` checkpoint in
        :func:`~engrava.cli.memory_commands._opened_full_store`, after the
        warning is logged and before the original exception is re-raised, so
        a cancellation requested during that synchronous stretch is now
        delivered there instead of being lost.
        """
        db = tmp_path / "m.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(f"database:\n  path: {db}\n", encoding="utf-8")

        result = _run_subprocess_with_external_sigint(
            [
                "-c",
                _CONFIG_REAL_SIGINT_DURING_CLOSE_DESCRIPTION_SCRIPT,
                "--config",
                str(config_path),
                *extra_args,
                "remember",
                "hi",
            ],
            marker="REACHED_CLOSE_STR_CALL",
        )

        assert "REACHED_CLOSE_STR_CALL" in result.stdout
        assert "Aborted!" in result.stderr, result.stderr
        assert "unexpected_error" not in result.stderr
        assert "unexpected RuntimeError" not in result.stderr


class TestBareTierCleanupLogNowSharesTheSameFix:
    """``main._close_quietly`` (the bare/default store tier) needed the identical fix.

    A verification round found this function still passing
    ``exc_info=True`` after the ``--config`` tier's own cleanup site
    (:func:`~engrava.cli.memory_commands._opened_full_store`, covered by
    :class:`TestCleanupLogNoLongerReformatsThePropagatingException` above)
    had already been fixed: ``remember`` / ``recall`` / ``link`` with no
    ``--config`` reach ``_close_quietly`` through
    ``_opened_full_store``'s bare branch and
    :func:`~engrava.cli.main._opened_db`, not through the ``--config``
    tier's own inline ``await store.close()`` -- a call site the earlier
    fix never touched. These mirror the ``--config``-tier tests above,
    against the bare tier instead: a plain ``--db`` (or no ``--db`` at all)
    invocation, patching ``aiosqlite.Connection.close`` rather than
    ``SqliteEngravaCore.close``.

    Because ``_close_quietly`` runs the close as a separately scheduled,
    shielded task (see its own docstring), the close exception's
    ``__context__`` is never linked back to whatever this coroutine was
    cleaning up after -- unlike the ``--config`` tier's inline close. That
    means the original exception's own formatter is never invoked by this
    tier's cleanup log at all (measured ``{"close": 1, "original": 1}``
    both before and after this fix, since the boundary's own downstream
    read of the original exception is unaffected by this tier's cleanup
    log either way): the defect here was the absorbed control-flow signal
    and the missing "why", not an inflated call count.
    """

    def test_close_and_original_formatter_call_counts(self, tmp_path: Path) -> None:
        for index, extra_args in enumerate(([], ["--verbose"])):
            db = tmp_path / f"m{index}.db"

            result = _run_subprocess(
                [
                    "-c",
                    _BARE_CLEANUP_LOG_FORMATTER_CALL_COUNTS_SCRIPT,
                    "--db",
                    str(db),
                    *extra_args,
                    "remember",
                    "hi",
                ]
            )

            assert result.returncode == 1, result.stderr
            counts_line = next(
                line for line in result.stdout.splitlines() if line.startswith("COUNTS:")
            )
            counts = json.loads(counts_line[len("COUNTS:") :])
            assert counts == {"original": 1, "close": 1}, (extra_args, counts, result.stderr)

    def test_keyboard_interrupt_from_close_str_reaches_clicks_own_abort_handling(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"

        result = _run_subprocess(
            [
                "-c",
                _BARE_CLEANUP_LOG_CLOSE_KEYBOARD_INTERRUPT_SCRIPT,
                "--db",
                str(db),
                "remember",
                "hi",
            ]
        )

        assert "REACHED_CLOSE_STR_CALL" in result.stdout
        assert result.returncode == 1, result.stderr
        assert "Aborted!" in result.stderr
        assert "unexpected_error" not in result.stderr
        assert "unexpected RuntimeError" not in result.stderr

    def test_system_exit_from_close_str_now_reaches_the_caller(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"

        result = _run_subprocess(
            [
                "-c",
                _BARE_CLEANUP_LOG_CLOSE_SYSTEM_EXIT_SCRIPT,
                "--db",
                str(db),
                "remember",
                "hi",
            ]
        )

        assert "REACHED_CLOSE_STR_CALL" in result.stdout
        assert result.returncode == 37, result.stderr
        assert "unexpected RuntimeError" not in result.stderr

    def test_an_ordinary_close_failure_now_shows_why_not_just_where(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"

        result = _run_subprocess(
            [
                "-c",
                _BARE_CLOSE_PERMISSION_ERROR_SCRIPT,
                "--db",
                str(db),
                "remember",
                "hi",
            ]
        )

        assert result.returncode == 1, result.stderr
        assert "PermissionError: [Errno 13] Permission denied" in result.stderr
        assert "Error closing connection during cleanup" in result.stderr

    @pytest.mark.parametrize("extra_args", [[], ["--verbose"]], ids=["default", "verbose"])
    def test_real_external_sigint_during_close_description_now_aborts(
        self, tmp_path: Path, extra_args: list[str]
    ) -> None:
        """A real, externally delivered ``SIGINT`` during this exact window must now abort.

        Same defect as the ``--config`` tier's own
        ``TestCleanupLogNoLongerReformatsThePropagatingException.test_real_external_sigint_during_close_description_now_aborts``,
        at this tier's own cleanup site (``main._close_quietly``): confirmed
        live against this exact commit before the fix, absorbed for both
        ``--verbose`` on and off -- the process finished the warning below,
        emitted the original ``unexpected_error`` JSON, and exited ``1``
        instead of aborting. The fix adds the identical ``await
        asyncio.sleep(0)`` checkpoint here, after the warning is logged and
        before this function returns (letting the caller's own ``raise``
        re-raise the original exception), so a cancellation requested during
        that synchronous stretch is now delivered inside this function
        instead of being lost once it returns.
        """
        db = tmp_path / "m.db"

        result = _run_subprocess_with_external_sigint(
            [
                "-c",
                _BARE_REAL_SIGINT_DURING_CLOSE_DESCRIPTION_SCRIPT,
                "--db",
                str(db),
                *extra_args,
                "remember",
                "hi",
            ],
            marker="REACHED_CLOSE_STR_CALL",
        )

        assert "REACHED_CLOSE_STR_CALL" in result.stdout
        assert "Aborted!" in result.stderr, result.stderr
        assert "unexpected_error" not in result.stderr
        assert "unexpected RuntimeError" not in result.stderr


class _WatchedTracebackReadError(Exception):
    """Records every ``__traceback__`` read that goes through ``__getattribute__``.

    A plain attribute read (``exc.__traceback__``) is ordinary instance
    attribute lookup, so it runs this override; the built-in descriptor form
    :func:`~engrava.cli.memory_commands._frame_only_stack` actually uses
    (``BaseException.__traceback__.__get__(exc)``) bypasses it entirely.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label

    def __getattribute__(self, name: str) -> object:
        if name == "__traceback__":
            label = object.__getattribute__(self, "label")
            _TRACEBACK_READ_COUNTS[label] = _TRACEBACK_READ_COUNTS.get(label, 0) + 1
        return object.__getattribute__(self, name)


_TRACEBACK_READ_COUNTS: dict[str, int] = {}


def _raise_watched_traceback_read(label: str) -> NoReturn:
    raise _WatchedTracebackReadError(label)


def _raise_watched_boundary(*_args: object, **_kwargs: object) -> NoReturn:
    _raise_watched_traceback_read("boundary")


async def _raise_watched_close(_self: SqliteEngravaCore) -> None:
    _raise_watched_traceback_read("cleanup")


class TestFrameOnlyStackReadsTheTracebackThroughTheDescriptor:
    """``_frame_only_stack`` must not trigger an overridden ``__getattribute__``.

    The direction review instrumented an exception with an overridden
    ``__getattribute__`` and required the traceback read to record zero
    calls through it, at both call sites that now share
    :func:`~engrava.cli.memory_commands._frame_only_stack`:
    :func:`~engrava.cli.memory_commands._error_boundary`'s ``--verbose``
    stack log, and :func:`~engrava.cli.memory_commands._opened_full_store`'s
    cleanup-failure warning. The plain attribute form the reviewer compared
    against records exactly one call.
    """

    def test_helper_itself_records_zero_calls(self) -> None:
        _TRACEBACK_READ_COUNTS.clear()
        try:
            _raise_watched_traceback_read("direct")
        except _WatchedTracebackReadError as exc:
            memory_commands._frame_only_stack(exc)

        assert _TRACEBACK_READ_COUNTS.get("direct", 0) == 0
        # The plain attribute form this helper deliberately avoids: shown
        # here, not to exercise the helper, but to prove the instrumentation
        # itself is sound -- it does record a call through that path.
        try:
            _raise_watched_traceback_read("plain-attribute-baseline")
        except _WatchedTracebackReadError as exc:
            _ = exc.__traceback__
        assert _TRACEBACK_READ_COUNTS.get("plain-attribute-baseline", 0) == 1

    def test_error_boundary_verbose_stack_log_records_zero_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _TRACEBACK_READ_COUNTS.clear()
        monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_watched_boundary)
        db = tmp_path / "m.db"
        runner = CliRunner()

        target_logger = logging.getLogger("engrava.cli.memory_commands")
        target_logger.addHandler(caplog.handler)
        target_logger.setLevel(logging.DEBUG)
        try:
            result = runner.invoke(cli, ["--db", str(db), "--verbose", "remember", "hi", "--json"])
        finally:
            target_logger.removeHandler(caplog.handler)

        assert result.exit_code == 1
        assert _TRACEBACK_READ_COUNTS.get("boundary", 0) == 0

    def test_cleanup_warning_records_zero_calls(self, tmp_path: Path) -> None:
        _TRACEBACK_READ_COUNTS.clear()
        db = tmp_path / "m.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(f"database:\n  path: {db}\n", encoding="utf-8")
        runner = CliRunner()
        seed = runner.invoke(cli, ["--config", str(config_path), "remember", "seed"])
        assert seed.exit_code == 0, seed.output

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(SqliteEngravaCore, "close", _raise_watched_close)
            monkeypatch.setattr(memory_commands.uuid, "uuid4", _raise_unanticipated)
            result = runner.invoke(
                cli,
                ["--config", str(config_path), "remember", "hi", "--json"],
            )

        assert result.exit_code == 1
        assert _TRACEBACK_READ_COUNTS.get("cleanup", 0) == 0

"""Tests for ``tests/_shell_command.py``, which the release wiring tests trust.

The cases are command lines ``script_argv`` recognises as running the script and
command lines it does not.
"""

from __future__ import annotations

import pytest

from tests._shell_command import script_argv

SCRIPT = "scripts/gate.py"
VERSION = "${nextRelease.version}"


class TestACommandThatRunsTheScript:
    @pytest.mark.parametrize(
        ("command", "arguments"),
        [
            (f"python {SCRIPT} {VERSION}", [VERSION]),
            (f"python3 {SCRIPT} 0.7.0", ["0.7.0"]),
            (f"python {SCRIPT}", []),
            (f"python {SCRIPT} {VERSION}\n", [VERSION]),
            (f"\npython {SCRIPT}\n\n", []),
        ],
        ids=["python", "python3", "no-arguments", "trailing-newline", "surrounding-blank-lines"],
    )
    def test_is_recognised_with_the_tokens_after_the_script(
        self, command: str, arguments: list[str]
    ) -> None:
        argv = script_argv(command, SCRIPT)

        assert argv is not None
        assert argv[:2] == [argv[0], SCRIPT]
        assert argv[2:] == arguments


class TestAnyOtherCommandLine:
    @pytest.mark.parametrize(
        "command",
        [
            f"echo {SCRIPT} {VERSION}",
            f"echo python {SCRIPT} {VERSION}",
            f"echo '&&' python {SCRIPT} {VERSION}",
            f"bash {SCRIPT} {VERSION}",
            f"python -c pass {SCRIPT} {VERSION}",
            f"python {SCRIPT}x {VERSION}",
            f"python {SCRIPT}#x {VERSION}",
            f"python scripts/other.py {VERSION}",
            f"python {SCRIPT} {VERSION} || true",
            f"true || python {SCRIPT} {VERSION}",
            f"exit 0 && python {SCRIPT} {VERSION}",
            f"false && python {SCRIPT} {VERSION}",
            f"true && python {SCRIPT} {VERSION}",
            f"python {SCRIPT} {VERSION} && true",
            f"python {SCRIPT} 0.7.0&&true",
            f"python {SCRIPT} {VERSION} && true\nexit 0",
            f"python {SCRIPT} {VERSION}\nexit 0",
            f"python {SCRIPT} {VERSION}; true",
            f"python {SCRIPT} {VERSION} | cat",
            f"python {SCRIPT} {VERSION} &",
            f"echo $(python {SCRIPT} {VERSION})",
            f"# python {SCRIPT}",
            "",
        ],
        ids=[
            "echo",
            "echo-python",
            "quoted-and",
            "wrong-interpreter",
            "script-is-not-the-first-argument",
            "another-script-with-a-longer-name",
            "hash-in-the-script-word",
            "another-script",
            "or-true",
            "skipped-by-or",
            "shell-exits-first",
            "false-before-it",
            "not-the-first-segment",
            "and-after-it",
            "and-after-it-without-spaces",
            "second-line-swallows-the-status",
            "second-line",
            "semicolon",
            "pipe",
            "background",
            "command-substitution",
            "comment",
            "empty",
        ],
    )
    def test_is_not_recognised(self, command: str) -> None:
        assert script_argv(command, SCRIPT) is None

    @pytest.mark.parametrize("command", [None, 3, ["python", SCRIPT]], ids=["none", "int", "list"])
    def test_a_value_that_is_not_a_string_is_not_recognised(self, command: object) -> None:
        assert script_argv(command, SCRIPT) is None

"""Read a shell command line for the one question a wiring test asks of it: does it run this script?

A workflow step or a semantic-release ``prepareCmd`` is a string handed to a
shell. A string that contains the path of a script does not run it:
``echo scripts/gate.py`` names the script and runs nothing, and
``python scripts/gate.py || true`` runs it and discards its exit status. The
wiring tests call :func:`script_argv` so that a gate counts as wired only when
it would actually run and could actually fail the step.
"""

from __future__ import annotations

import shlex

INTERPRETERS = frozenset({"python", "python3"})


def script_argv(command: object, script: str) -> list[str] | None:
    """Return the tokens of a command line that is one simple command running ``script``, else None.

    The line is recognised when it is one simple command on one line: an
    interpreter, the script, then any further tokens. Any shell operator
    (``&&``, ``||``, ``;``, ``|``, ``&``, a redirection, a subshell) or a
    second line makes the line something else, and it is not recognised.
    Surrounding blank lines, such as the trailing newline of a YAML
    ``run: |`` block, are ignored.

    Quotes stay in the tokens (``posix=False``), so a quoted operator is a
    token and not an operator. A ``#`` stays in its token, so ``gate.py#x``
    is a different word from ``gate.py``.
    """
    if not isinstance(command, str):
        return None
    command = command.strip()
    if "\n" in command:
        return None
    lexer = shlex.shlex(command, posix=False, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    argv: list[str] = []
    for word in lexer:
        if set(word) <= set(lexer.punctuation_chars):
            return None
        argv.append(word)
    if len(argv) >= 2 and argv[0] in INTERPRETERS and argv[1] == script:
        return argv
    return None

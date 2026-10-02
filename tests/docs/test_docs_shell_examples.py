"""Layer 6 of the documentation-example tests — ``bash`` invocations vs the real CLI.

The ``engrava ...`` invocations on a ``bash`` block's lines are checked,
**without executing anything**, against the real ``click`` command tree the
CLI ships, except in the blocks registered in ``EXEMPT_BASH_BLOCKS``, so a
command that does not exist, or a long option that does not exist on the
command it is attached to, fails the suite.

Checked, not executed
----------------------
Running the documented commands would need a real database, network access, or
destructive side effects (``gc``, ``restore --clear``, ...). The claim worth
defending for free is narrower and cheaper: *the command and its long options
exist*. Each logical line is tokenised by ``_tokenize_shell_line``, a
hand-written scan that tracks single-/double-quote state character by
character and uses that state for exactly two decisions: whether a
character is a real (unquoted) operator (``;``, a single or doubled ``|``,
a single or doubled ``&``), and whether a ``#`` starts a comment (only at
the start of a word, unquoted). Each token is tagged as an operator or not
*while quote state is known* -- a quoted value equal to a separator's text
(a filename literally
named ``|``) can never be reinterpreted as one later.
Escaping, command substitution, subshells, heredocs, and redirection are out
of scope; see ``_tokenize_shell_line``'s docstring for the exact boundary. An
unquoted redirection character (``>``, ``<``, and their relatives, including
the ``&`` inside ``2>&1``) is reported, naming the line, rather than modelled
or silently ignored.

The resulting token stream is walked left to right for ``engrava`` invocations: a
fresh segment begins at the start of the line and immediately after any
``;``/``|``/``&``/``&&``/``||`` boundary (leading ``VAR=value`` assignments at
that position are skipped first), and the command word is read wherever such a
segment begins — not only when it is the very first word of the whole line,
so ``true && engrava reindex`` still checks the second invocation. Each
``engrava``-led segment is walked against ``engrava.cli.main.cli``: the
command name must be one of the eight built-in commands, and every
``--long-option`` token must belong to the object that actually owns it —
**group-level** options (declared on ``engrava`` itself, e.g. ``--db``) are a
different set from **command-level** options (declared on the subcommand,
e.g. ``gc --dry-run``), and a line commonly carries both. Click's own ``--``
option terminator is recognised, not mistaken for an unknown option, and
``--help`` is validated like any other flag rather than short-circuiting the
rest of the line — real click does not let it bypass parsing: ``engrava
--help --nonexistent`` exits ``2`` with ``No such option``. An option needing
a value never consumes a real (unquoted) operator token as that value; the
operator is left for the walk to find as a boundary instead.

A line whose tokenising fails (an unterminated quote) is unconditionally a
violation, named by the line it occurs on -- there is no second, cruder
model deciding which failures are "worth" reporting. A bash block in this documentation that
cannot be tokenised is either wrong or belongs in ``EXEMPT_BASH_BLOCKS`` with
a reason. Each logical line is tokenised independently, so a quote a real
shell would let span multiple physical lines on a bare newline (as opposed
to an explicit backslash continuation, which ``_command_line_candidates`` does join)
is not supported; see ``_tokenize_shell_line`` for that boundary. No current
documented example needs it.

One documented form is exempt from "every line is a command": a **shell
session transcript**, which shows typed commands interleaved with the
program output they produce (a ``$ `` prompt marks each command; e.g. ``$
engrava ... gc`` followed by the text it prints, possibly containing an
apostrophe or other character that is not shell input at all and must never
be tokenised as if it were). A block is recognised as this form, as a whole,
by containing any ``$ ``-prefixed line -- see ``_is_session_transcript`` and
``_command_line_candidates``. Only its ``$ ``-prefixed lines are checked;
every other line is treated as output and skipped entirely, never
tokenised, regardless of content. A block with no ``$ `` line keeps the
ordinary behaviour: every line is a command.

A ``bash`` block this module cannot check is registered in
``EXEMPT_BASH_BLOCKS`` with a reason from the closed
:class:`~tests.docs._md_blocks.ExemptionReason` vocabulary, never left silently
uncovered. The registration also carries a digest of the block's text, and the
exemption applies only to the block's text (line endings normalised): edit an
exempt block and it is checked like any other, and
``test_exempt_bash_registry_digests_match`` names it. Most such blocks name no
``engrava`` invocation at all (``pip install``,
``make``, ``sqlite3``, ``python -m ...``); one — the CLI's own
``engrava [GLOBAL OPTIONS] COMMAND [ARGS]...`` usage-grammar line — genuinely
is shaped like an invocation and is filed under a distinct reason for exactly
that reason (see ``ExemptionReason.USAGE_GRAMMAR_PLACEHOLDER``), rather than
under "not an invocation," which would be false. Every other ``bash`` block
must contain at least one invocation line this module can check.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

import click
import pytest

from engrava.cli.main import cli
from tests.docs._md_blocks import (
    REPO_ROOT,
    CodeBlock,
    ExemptionReason,
    block_digest,
    exemption_digest_problems,
    extract_exact_fenced_blocks,
    markdown_files,
)

_PROMPT_RE = re.compile(r"^\$\s*")

# A whole token that is an env-var assignment, e.g. "ENGRAVA_DB=x.db" --
# checked against a real token once a line tokenises.
_ENV_ASSIGNMENT_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _all_bash_blocks() -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    for path in markdown_files():
        blocks.extend(extract_exact_fenced_blocks(path, "bash"))
    return blocks


_ALL_BASH_BLOCKS = _all_bash_blocks()

# Bash blocks that name no `engrava` invocation at all, each with a reason from
# the closed ExemptionReason vocabulary. Every bash block not listed here must
# contain at least one line this module can check (see
# test_every_bash_block_is_covered).
#
# Entries are (markdown_path, anchor, reason, digest). The anchor finds the
# block; the digest (see `block_digest`) binds the exemption to the block's text
# (line endings normalised). An exemption applies only while the block still
# hashes to its digest: an edited block is checked like any other (so a command
# appended to an exempt block reports itself), and
# test_exempt_bash_registry_digests_match names it and prints the digest to
# register if the edit is deliberate.
EXEMPT_BASH_BLOCKS: tuple[tuple[str, str, ExemptionReason, str], ...] = (
    (
        "README.md",
        "pip install engrava",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "5402b543a860c012",
    ),
    (
        "README.md",
        "pip install 'engrava[vec]'",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "2c31b4fce4174f49",
    ),
    ("README.md", "uvx engrava-mcp", ExemptionReason.NOT_AN_ENGRAVA_INVOCATION, "c78476e34cb67f22"),
    ("README.md", "make install", ExemptionReason.NOT_AN_ENGRAVA_INVOCATION, "c0230d95890ff568"),
    (
        "docs/backup-and-recovery.md",
        "VACUUM INTO 'engrava-backup.db'",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "4e3cb5488560033b",
    ),
    (
        "docs/backup-and-recovery.md",
        "wal_checkpoint(TRUNCATE)",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "fc844eb00d129556",
    ),
    (
        "docs/benchmarks.md",
        "pip install 'engrava[embeddings-local]'",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "dba79ad4af250b02",
    ),
    (
        "docs/benchmarks.md",
        "--with-reproducibility",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "007859d1d6ab408b",
    ),
    (
        "docs/benchmarks.md",
        "python -m engrava.benchmarks.longmemeval",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "f0286eae04210cfb",
    ),
    (
        "docs/cli.md",
        "engrava [GLOBAL OPTIONS] COMMAND [ARGS]...",
        ExemptionReason.USAGE_GRAMMAR_PLACEHOLDER,
        "f062fe7d84352d8b",
    ),
    (
        "docs/guides/agent-memory.md",
        "python examples/agent_loop.py",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "6c022454bb3101a5",
    ),
    (
        "docs/guides/embeddings.md",
        'pip install "engrava[embeddings-local]"',
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "e18807c71720cd8e",
    ),
    (
        "docs/guides/embeddings.md",
        'pip install "engrava[embeddings-openai]"',
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "b06d22a879be0b47",
    ),
    (
        "docs/guides/embeddings.md",
        'pip install "engrava[embeddings-ollama]"',
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "4e543da9ca0ec9f9",
    ),
    (
        "docs/guides/embeddings.md",
        'pip install "engrava[embeddings-hf]"',
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "03777e7d057bfae8",
    ),
    (
        "docs/known-limitations.md",
        "brew install python@3.12",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "0c3806b5ef557a62",
    ),
    (
        "docs/performance.md",
        "pip install 'engrava[vec]'",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "1e0959f1ef869a29",
    ),
    (
        "docs/quickstart.md",
        "pip install engrava",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "5402b543a860c012",
    ),
    (
        "docs/quickstart.md",
        "pip install 'engrava[embeddings-local]'",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "6ac2ca5a345cd808",
    ),
    (
        "docs/quickstart.md",
        "python examples/quickstart.py",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "3e0a8cf522cfe0df",
    ),
    (
        "docs/quickstart.md",
        "python -m engrava.benchmarks.synthetic",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "022e31d633869a6a",
    ),
    (
        "docs/troubleshooting.md",
        "sqlite3 engrava.db",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "090cf395f71ce64e",
    ),
    (
        "docs/upgrade.md",
        'sqlite3 my-data.db ".backup my-data.db.bak"',
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "b1709edea76d249d",
    ),
    (
        "docs/upgrade.md",
        "your app's existing ensure_schema()",
        ExemptionReason.NOT_AN_ENGRAVA_INVOCATION,
        "fd0f24efc8778771",
    ),
)


def _anchor_matches(anchor: str, body: str) -> bool:
    """Whether ``anchor`` appears in ``body`` at a word boundary.

    Plain substring matching lets a short anchor like ``"pip install engrava"``
    also match ``"pip install engrava-mcp"`` (a different package). Requiring
    that the character after the match is not a word/hyphen character avoids
    that false collision without needing artificially padded anchors.
    """
    return re.search(re.escape(anchor) + r"(?![\w-])", body) is not None


def _unique_block(rel: str, anchor: str) -> CodeBlock:
    """Return the single bash block in ``rel`` whose body contains ``anchor``."""
    matches = [b for b in _ALL_BASH_BLOCKS if b.rel == rel and _anchor_matches(anchor, b.body)]
    if len(matches) != 1:
        pytest.fail(
            f"anchor {anchor!r} matched {len(matches)} bash blocks in {rel} (want "
            f"exactly 1); update EXEMPT_BASH_BLOCKS in {__file__}.",
        )
    return matches[0]


def _registered_exemptions() -> list[tuple[CodeBlock, ExemptionReason, str]]:
    """Resolve every registry entry to ``(block, reason, registered_digest)``."""
    return [
        (_unique_block(rel, anchor), reason, digest)
        for rel, anchor, reason, digest in EXEMPT_BASH_BLOCKS
    ]


def _exempt_locations() -> dict[str, ExemptionReason]:
    """Map location to reason for each registered block whose text still matches its digest.

    A block edited since it was registered is deliberately absent: the
    exemption was granted for the text that was reviewed, not for whatever the
    block says now, so an edited block goes back to being checked.
    """
    return {
        block.location: reason
        for block, reason, digest in _registered_exemptions()
        if block_digest(block.body) == digest
    }


# ---------------------------------------------------------------------------
# Tokenising and walking a shell line against the real click tree
# ---------------------------------------------------------------------------


def _line_has_invocation_word(tokens: list[_Token]) -> bool:
    """Whether some ``engrava`` word begins a fresh segment anywhere in a tokenised line.

    Mirrors ``_scan_tokens_for_invocations``'s segment-start detection (start
    of line, or right after a boundary, a leading run of ``!`` and then
    leading env assignments skipped) but answers only "is there one", for
    ``block_has_invocation`` -- which needs to know a block is checkable,
    not to validate it.
    """
    i = 0
    n = len(tokens)
    at_segment_start = True
    while i < n:
        if at_segment_start:
            j = i
            while j < n and tokens[j].text == "!" and not tokens[j].is_operator:
                j += 1
            while j < n and _ENV_ASSIGNMENT_TOKEN_RE.match(tokens[j].text):
                j += 1
            if j < n and tokens[j].text == "engrava" and not tokens[j].is_operator:
                return True
            at_segment_start = False
            i = j
            continue
        if tokens[i].is_operator:
            at_segment_start = True
        i += 1
    return False


def _tokenize_shell_line(text: str) -> list[_Token]:  # noqa: C901, PLR0915
    """Tokenise one shell line with a hand-written, quote-state character scan.

    This is not a shell parser: it tracks exactly one thing -- whether the
    scan position is inside a single- or double-quoted region -- and uses
    that state for exactly two decisions: whether a character is a real
    (unquoted) operator (``;``, a single or doubled ``|``, a single or
    doubled ``&``), and whether a ``#`` starts a comment (only at the start
    of a word, i.e. preceded by whitespace, an operator, or the start of the
    line, and unquoted). Tagging each token with whether it *is* an operator,
    decided while quote state is still known, is what a later
    ``token.text in {...}`` comparison cannot recover.

    Escaping, command substitution, subshells, and heredocs are out of
    scope; a line needing any of those is not documentation-example shell
    input this suite generates or expects. Quote state is scoped to one
    call, and ``_command_line_candidates`` only joins physical lines that end with an
    explicit backslash continuation -- a single-quoted string that a real
    shell would let span multiple physical lines on a bare (un-escaped)
    newline is tokenised here as two independent, and therefore each
    apparently unterminated, logical lines. No current documented example
    needs that.

    Redirection (``>``, ``>>``, ``<``, ``<<``, ``&>``, ``2>&1``, and the
    like) is not modelled either: an unquoted ``>``/``<`` is not a command
    boundary the way ``;``/``|``/``&`` are, and the ``&`` inside ``2>&1`` is
    not a real ``&&``-style separator -- treating it as one silently drops
    everything after it from being checked. Any unquoted occurrence raises,
    naming the line, rather than being mis-split or silently ignored.

    Raises:
        ValueError: If a quote is opened and never closed, or an unquoted
            redirection character is found.

    """
    tokens: list[_Token] = []
    current: list[str] = []
    have_token = False
    quote: str | None = None
    at_word_start = True
    i = 0
    n = len(text)

    def flush() -> None:
        nonlocal have_token
        if have_token:
            tokens.append(_Token("".join(current), is_operator=False))
            current.clear()
            have_token = False

    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
            have_token = True
            at_word_start = False
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            have_token = True
            at_word_start = False
            i += 1
            continue
        if ch == "#" and at_word_start:
            break  # Comment: the rest of the line is ignored.
        if ch.isspace():
            flush()
            at_word_start = True
            i += 1
            continue
        if ch in _REDIRECTION_CHARS:
            msg = f"contains a {ch!r} redirection, which this checker does not model"
            raise ValueError(msg)
        if ch in _OPERATOR_CHARS:
            flush()
            if i + 1 < n and text[i + 1] == ch and ch in ("&", "|"):
                tokens.append(_Token(ch * 2, is_operator=True))
                i += 2
            else:
                tokens.append(_Token(ch, is_operator=True))
                i += 1
            at_word_start = True
            continue
        current.append(ch)
        have_token = True
        at_word_start = False
        i += 1

    if quote is not None:
        msg = f"unterminated {quote!r} quote"
        raise ValueError(msg)
    flush()
    return tokens


_OPERATOR_CHARS = frozenset({"|", ";", "&"})

# `>`, `>>`, `<`, `<<`, `&>`, `2>&1`, and their relatives are redirection, not
# a command boundary -- modelling what a redirected engrava invocation still
# means to check would mean parsing shell redirection, which is out of
# scope. Any unquoted occurrence is reported (see _tokenize_shell_line)
# rather than silently mis-split (the `&` in `2>&1` is not a real `&&`-style
# separator) or silently ignored (redirected output can carry a phrase that
# is not shell input at all).
_REDIRECTION_CHARS = frozenset({">", "<"})


def _opt_map(params: list[click.Parameter]) -> dict[str, bool]:
    """Map every option string (short and long) on ``params`` to its is_flag."""
    mapping: dict[str, bool] = {}
    for param in params:
        if isinstance(param, click.Option):
            for opt in param.opts:
                mapping[opt] = param.is_flag
    return mapping


def _with_help(opts: dict[str, bool]) -> dict[str, bool]:
    """Add click's auto-generated ``--help`` flag to an option map.

    ``click`` adds ``--help`` to every group and command automatically; it is
    not part of ``cli.params``/``cmd.params``. It must be validated like any
    other flag: real click does not let ``--help`` bypass parsing of the rest
    of the line (``engrava --help --nonexistent`` exits ``2`` with ``No such
    option``), so this module must not short-circuit on it either.
    """
    return {**opts, "--help": True}


_GROUP_OPTS = _with_help(_opt_map(cli.params))
_COMMAND_NAMES = sorted(cli.commands)


@dataclass(frozen=True)
class _Token:
    """One shell token, tagged with whether it is a real (unquoted) operator.

    The tag is decided during tokenising, while quote state is still known
    -- never re-derived later from ``text`` alone, which cannot distinguish
    a quoted ``"|"`` from a real pipe.
    """

    text: str
    is_operator: bool


@dataclass(frozen=True)
class InvocationError:
    """A single ``engrava ...`` invocation that does not match the real CLI tree."""

    line: str
    message: str


def _step_over_option(tokens: list[_Token], i: int, opts: dict[str, bool]) -> int | None:
    """If ``tokens[i]`` is a known option, return the index just past it (and its value).

    ``None`` means ``tokens[i].text`` is not a recognised option string at
    all (a positional argument, an unrecognised short option, or an unknown
    ``--long-option`` -- the caller distinguishes the last case itself). An
    operator token is never consumed as a value: an option needing one that
    is immediately followed by a real ``;``/``|``/``&`` is treated as having
    no value here, leaving that operator for the caller to find as a
    boundary.
    """
    tok = tokens[i]
    name = tok.text.split("=", 1)[0]
    if name not in opts:
        return None
    if opts[name] or "=" in tok.text:
        return i + 1
    if i + 1 < len(tokens) and not tokens[i + 1].is_operator:
        return i + 2
    return i + 1


def _scan_global_options(tokens: list[_Token], start: int) -> tuple[int, InvocationError | None]:
    """Walk group-level option tokens starting right after ``tokens[start]`` (``"engrava"``).

    Returns the index of the command name (a segment-boundary token's
    position, or ``len(tokens)`` if neither follows) and the first violation
    met, if any -- scanning continues past a violation so the caller can
    still find where this invocation's own tokens end.
    """
    line = " ".join(t.text for t in tokens[start:])
    i = start + 1
    error: InvocationError | None = None
    after_terminator = False
    while i < len(tokens):
        tok = tokens[i]
        if tok.is_operator:
            return i, error
        if after_terminator:
            break  # First post-`--` token is the command name.
        if tok.text == "--":
            after_terminator = True
            i += 1
            continue
        stepped = _step_over_option(tokens, i, _GROUP_OPTS)
        if stepped is not None:
            i = stepped
            continue
        if tok.text.startswith("--"):
            if error is None:
                name = tok.text.split("=", 1)[0]
                error = InvocationError(line, f"unknown group-level option {name!r}")
            i += 1
            continue
        break  # First non-option token: the command name.
    return i, error


def _scan_command_options(
    tokens: list[_Token],
    start: int,
    command: str,
    opts: dict[str, bool],
) -> tuple[int, InvocationError | None]:
    """Walk the remaining tokens after the command name, validating long options."""
    line = " ".join(t.text for t in tokens)
    i = start
    error: InvocationError | None = None
    after_terminator = False
    while i < len(tokens):
        tok = tokens[i]
        if tok.is_operator:
            return i, error
        if after_terminator:
            i += 1
            continue
        if tok.text == "--":
            after_terminator = True
            i += 1
            continue
        stepped = _step_over_option(tokens, i, opts)
        if stepped is not None:
            i = stepped
            continue
        if tok.text.startswith("--"):
            if error is None:
                name = tok.text.split("=", 1)[0]
                error = InvocationError(line, f"unknown option {name!r} for command {command!r}")
            i += 1
            continue
        i += 1  # Positional argument, or an unrecognised short option/value.
    return i, error


def _walk_one_invocation(tokens: list[_Token], start: int) -> tuple[int, InvocationError | None]:
    """Validate one ``engrava`` invocation beginning at ``tokens[start]``.

    Only existence is checked: the command must be one of the built-in eight,
    and every ``--long-option`` token must belong to the object (group or
    command) that owns it at that point in the line. Positional arguments and
    unrecognised short options are consumed without being validated — the
    task is proving a real command/flag exists, not replicating click's own
    argument arity checking.

    Returns the index immediately after this invocation's own tokens (a
    segment-boundary token's position, or ``len(tokens)``) and the first
    violation found, if any. A token consumed as an option's value is never
    reinterpreted as a boundary, even if its text is exactly ``|``, because
    it is tagged ``is_operator=False`` at tokenising time (quoted) and the
    walk advances past it while consuming the preceding option regardless.
    """
    line = " ".join(t.text for t in tokens[start:])
    i, error = _scan_global_options(tokens, start)
    if i >= len(tokens) or tokens[i].is_operator:
        return i, error  # `--help`-only, or a bare `engrava` before a boundary.

    command = tokens[i].text
    if command not in cli.commands:
        cmd_error = error or InvocationError(
            line,
            f"{command!r} is not a known engrava command (known: {_COMMAND_NAMES})",
        )
        end = i + 1
        while end < len(tokens) and not tokens[end].is_operator:
            end += 1
        return end, cmd_error

    cmd_opts = _with_help(_opt_map(cli.commands[command].params))
    end, cmd_level_error = _scan_command_options(tokens, i + 1, command, cmd_opts)
    return end, error or cmd_level_error


def _check_invocation(tokens: list[_Token]) -> InvocationError | None:
    """Validate a single, already-isolated ``engrava`` invocation (``tokens[0].text``)."""
    _, error = _walk_one_invocation(tokens, 0)
    return error


def _scan_tokens_for_invocations(tokens: list[_Token]) -> list[InvocationError]:
    """Find and validate every ``engrava`` invocation in a fully tokenised line.

    A fresh segment begins at the start of the token list and immediately
    after any ``;``/``|``/``&``/``&&``/``||`` boundary token. A leading
    **run** of unquoted ``!`` there -- bash's pipeline-negation operator,
    unambiguous as the first, unquoted word of a segment, and one bash
    itself allows repeated (``! ! echo hi`` still runs ``echo hi``) -- is
    skipped in full, not just one, then leading ``VAR=value`` assignments
    are skipped (so ``! ! ENV=1 engrava ...`` after a boundary is still an
    invocation). The command word is looked for wherever a segment begins,
    not only when it is the first word of the whole line, so ``true &&
    engrava reindex`` still checks the second invocation.

    Known limitation: tokenising strips quotes, so a *quoted* ``"!"`` is
    indistinguishable here from the operator -- ``"!" engrava reindex``
    reads as negation, though bash treats a quoted ``!`` as an ordinary
    word (e.g. a command name) and never as negation. No documented example
    does this.
    """
    errors: list[InvocationError] = []
    i = 0
    n = len(tokens)
    at_segment_start = True
    while i < n:
        if at_segment_start:
            j = i
            while j < n and tokens[j].text == "!" and not tokens[j].is_operator:
                j += 1
            while j < n and _ENV_ASSIGNMENT_TOKEN_RE.match(tokens[j].text):
                j += 1
            if j < n and tokens[j].text == "engrava" and not tokens[j].is_operator:
                end, error = _walk_one_invocation(tokens, j)
                if error is not None:
                    errors.append(error)
                i = end
                at_segment_start = False
                continue
            at_segment_start = False
            i = j
            continue
        if tokens[i].is_operator:
            at_segment_start = True
        i += 1
    return errors


def _is_session_transcript(body: str) -> bool:
    """Whether a bash block uses the shell-session-transcript convention.

    A block mixing typed commands with their interleaved program output
    marks each command with a literal ``$ `` prompt (e.g. a page showing
    ``$ engrava ... gc`` followed by the text it prints). Detected from any
    **physical** line starting with the prompt, before any continuation
    joining -- so a trailing backslash on an output line can never hide
    (or fabricate) this decision.
    """
    return any(raw.strip().startswith("$ ") for raw in body.splitlines())


def _unquoted_comment_start(text: str) -> int | None:
    """Return the index where an unquoted, word-initial ``#`` starts a comment, if any.

    Tracks single-/double-quote state the same way ``_tokenize_shell_line``
    does, narrowed to the one question a continuation decision needs: does a
    ``#`` here begin a real comment? Kept separate from the main tokeniser
    (which discards comment text rather than reporting where it starts)
    instead of changing that function's contract for every other caller.
    Word-start resets after an operator character (``;``/``|``/``&``), the
    same way it resets after whitespace -- ``engrava info;# comment`` starts
    a comment right after the ``;`` in real bash, which a tracker that only
    reset on whitespace would miss.
    """
    quote: str | None = None
    at_word_start = True
    for index, ch in enumerate(text):
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            at_word_start = False
            continue
        if ch == "#" and at_word_start:
            return index
        if ch.isspace() or ch in _OPERATOR_CHARS:
            at_word_start = True
            continue
        at_word_start = False
    return None


def _backslash_run_length(text: str) -> int:
    """Count the consecutive backslash characters ending ``text``."""
    count = 0
    index = len(text) - 1
    while index >= 0 and text[index] == "\\":
        count += 1
        index -= 1
    return count


def _is_inside_quotes_at_end(text: str) -> bool:
    """Whether scanning ``text`` alone leaves an open (unterminated) quote.

    Narrowed, like ``_unquoted_comment_start``, to one question: a trailing
    backslash reached while still inside a quote (``engrava 'in\\`` on one
    physical line, closed by ``fo'`` on the next) is not a reconstructable
    continuation -- the quote never closed on this physical line, and this
    checker does not model a quoted string spanning the join point.
    """
    quote: str | None = None
    for ch in text:
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
    return quote is not None


def _continuation_decision(text: str, *, has_next_line: bool) -> str:
    """Classify what ``text``'s trailing character means for continuation.

    A trailing backslash continues the line only when it is the line's
    literal final character (nothing, not even whitespace, after it --
    checked by ``text.endswith("\\\\")`` on the raw, un-stripped text), an
    **even** number of other backslashes precede it -- equivalently, the
    total trailing run is **odd** -- so it is itself unescaped
    (``engrava info\\`` has zero backslashes before the final one, and zero
    is even, so it continues; ``\\\\\\\\`` is an even run of two, the
    second escaped by the first into one literal, escaped backslash, not a
    marker), it is outside any unquoted comment, and it is outside any
    (unterminated-on-this-line) quote.

    Returns one of:

    * ``"final"`` -- no reconstructable continuation; use ``text`` as the
      finished command (this also covers a comment's trailing backslash,
      and an escaped trailing space/backslash pair a real shell would keep
      as literal content rather than continue past).
    * ``"continue"`` -- a genuine marker, and a following physical line
      exists to join with.
    * ``"final_strip"`` -- a genuine marker, but this is the block's last
      physical line: a real shell reads the same backslash-newline-EOF and
      simply ends the command there, so the marker is removed and nothing
      is appended.
    * ``"refused"`` -- a trailing backslash whose shape (escaped, or inside
      a quote this checker cannot see close) it will not attempt to
      reconstruct; the caller must report this, naming the line.
    """
    if not text.endswith("\\"):
        return "final"
    if _unquoted_comment_start(text) is not None:
        return "final"
    if _backslash_run_length(text) % 2 == 0:
        return "refused"
    if _is_inside_quotes_at_end(text):
        return "refused"
    return "continue" if has_next_line else "final_strip"


@dataclass(frozen=True)
class _CommandCandidate:
    """One command's display text and either its checkable text or a refusal reason.

    ``text`` is ``None`` exactly when ``refusal`` names why this command's
    trailing backslash could not be reconstructed (see
    ``_continuation_decision``) -- the caller reports ``refusal`` as a
    violation instead of tokenising anything.
    """

    display: str
    text: str | None
    refusal: str | None = None


def _command_line_candidates(body: str) -> list[_CommandCandidate]:
    """Return one ``_CommandCandidate`` per command line in a bash block.

    Classification runs on **physical** lines, before any continuation
    joining. In a session-transcript block (see ``_is_session_transcript``),
    a physical line is a command only if it starts with ``$ ``; every other
    physical line is program output. In an ordinary block, a physical line
    is a command unless it is a bare comment (starts with ``#``). Only once
    a line is classified as the *start* of a command is its own trailing
    backslash examined for continuation (see ``_continuation_decision``): an
    output line's, or a comment line's, trailing backslash is never treated
    as continuing into the next physical line. Classifying first and
    joining second (rather than the reverse) is what keeps output from
    swallowing the command that follows it, and a comment from swallowing
    whatever follows it, while a genuine multi-line command
    (``docs/troubleshooting.md`` has two, each an ``engrava restore``
    command split across two lines with a trailing backslash) still joins
    correctly, because the join happens *within* an already-started
    command.

    Joining removes the backslash and inserts **nothing** in its place --
    exactly what a real shell does with a backslash-newline pair -- so a
    line ending in ``engra`` followed by a continuation starting ``va
    reindex`` becomes ``engrava reindex``, not ``engra va reindex``. Inside
    a transcript, a continuation's own physical line is never a fresh
    prompt: a leading ``$ `` there (with any leading whitespace) is stripped
    before concatenation, the same as on any command's first line, rather
    than treated as literal content or as an error. A trailing backslash
    this checker cannot safely reconstruct (escaped, or inside an
    unterminated quote) is never guessed at or silently dropped -- see
    ``_continuation_decision``'s ``"refused"`` outcome.
    """
    transcript = _is_session_transcript(body)
    raw_lines = body.splitlines()
    n = len(raw_lines)
    candidates: list[_CommandCandidate] = []
    i = 0
    while i < n:
        raw = raw_lines[i]
        lstripped = raw.lstrip()
        stripped = raw.strip()
        is_command_start = (
            lstripped.startswith("$ ") if transcript else not stripped.startswith("#")
        )
        if not is_command_start:
            i += 1
            continue  # Program output (transcript) or a comment (ordinary): never joins.
        # Only the leading prompt/indent is stripped here -- trailing
        # whitespace is kept exactly as written, since a trailing backslash
        # is a continuation marker only when it is the line's literal last
        # character (see _continuation_decision).
        text = lstripped[len("$ ") :] if transcript else _PROMPT_RE.sub("", lstripped)
        display_lines = [raw]
        while True:
            decision = _continuation_decision(text, has_next_line=i + 1 < n)
            if decision == "refused":
                refusal = (
                    "trailing backslash cannot be reconstructed (escaped, or inside "
                    "a quote that does not close on this line)"
                )
                candidates.append(_CommandCandidate("\n".join(display_lines), None, refusal))
                break
            if decision == "final_strip":
                candidates.append(_CommandCandidate("\n".join(display_lines), text[:-1]))
                break
            if decision == "final":
                candidates.append(_CommandCandidate("\n".join(display_lines), text))
                break
            # decision == "continue": a following physical line exists.
            i += 1
            continuation = raw_lines[i]
            if transcript:
                lstripped_continuation = continuation.lstrip()
                if lstripped_continuation.startswith("$ "):
                    continuation = lstripped_continuation[len("$ ") :]
            text = text[:-1] + continuation
            display_lines.append(raw_lines[i])
        i += 1
    return candidates


def block_invocation_errors(block: CodeBlock) -> list[InvocationError]:
    """Return every CLI-tree violation among a bash block's ``engrava`` invocations.

    Every command line (see ``_command_line_candidates``) is tokenised; there
    is no pre-filter deciding which lines are "worth" tokenising, and no
    fallback deciding which tokenising failures are "worth" reporting -- a
    second, cruder model is exactly what this checker does not have. A
    command whose trailing backslash could not be reconstructed, or that
    fails to tokenise once reconstructed (e.g. an unterminated quote
    unrelated to continuation), is unconditionally a violation, named by the
    line it occurs on: a bash block in this documentation that cannot be
    read this way is either wrong or needs registering with a reason in
    ``EXEMPT_BASH_BLOCKS``, and a human should see either outcome, not have
    it silently decided away.
    """
    errors: list[InvocationError] = []
    for candidate in _command_line_candidates(block.body):
        if candidate.refusal is not None:
            errors.append(InvocationError(candidate.display, candidate.refusal))
            continue
        if not candidate.text:
            continue
        try:
            tokens = _tokenize_shell_line(candidate.text)
        except ValueError as exc:
            message = f"does not tokenise as shell input ({exc})"
            errors.append(InvocationError(candidate.display, message))
            continue
        errors.extend(_scan_tokens_for_invocations(tokens))
    return errors


def block_has_invocation(block: CodeBlock) -> bool:
    """Whether a bash block contains at least one checkable ``engrava`` line.

    A candidate that could not be reconstructed, or that fails to tokenise,
    counts as "has an invocation" too: it is not exempt, it fails loudly in
    ``block_invocation_errors`` instead.
    """
    for candidate in _command_line_candidates(block.body):
        if candidate.refusal is not None:
            return True
        if not candidate.text:
            continue
        try:
            tokens = _tokenize_shell_line(candidate.text)
        except ValueError:
            return True
        if _line_has_invocation_word(tokens):
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _block_id(block: CodeBlock) -> str:
    return block.location


def test_bash_extractor_found_blocks() -> None:
    """Sanity check against a silently-empty extractor."""
    assert len(_ALL_BASH_BLOCKS) > 20, (
        f"expected many bash blocks, found {len(_ALL_BASH_BLOCKS)}; the "
        f"extractor may be misconfigured."
    )


def test_exempt_bash_registry_anchors_are_unique() -> None:
    """Every EXEMPT_BASH_BLOCKS anchor binds exactly one bash block."""
    _registered_exemptions()


def test_exempt_bash_registry_digests_match() -> None:
    """Every exempt bash block still has the text it was exempted for.

    An exemption that keyed on location alone kept skipping a block after an
    unchecked command was appended to it. Here each entry also carries a digest
    of the block's text (line endings normalised), so the edit is reported by
    block, with the digest to register if the edit is deliberate.
    """
    problems = exemption_digest_problems(
        "EXEMPT_BASH_BLOCKS",
        [(block, digest) for block, _reason, digest in _registered_exemptions()],
    )
    assert not problems, "exempt bash blocks were edited:\n" + "\n".join(problems)


def test_every_bash_block_is_covered() -> None:
    """Every bash block either names a checkable invocation or is exempt with a reason."""
    exempt = _exempt_locations()
    uncovered = [
        block.location
        for block in _ALL_BASH_BLOCKS
        if block.location not in exempt and not block_has_invocation(block)
    ]
    assert not uncovered, (
        "these bash blocks name no `engrava` invocation this module can check, "
        f"and are not registered in EXEMPT_BASH_BLOCKS with a reason: {uncovered}"
    )
    checked = [b for b in _ALL_BASH_BLOCKS if b.location not in exempt]
    print(  # noqa: T201 — intentional census summary for the -s report
        f"\nBash-example census: total={len(_ALL_BASH_BLOCKS)} "
        f"checked={len(checked)} exempt={len(exempt)}"
    )
    reason_tally: dict[str, int] = {}
    for reason in exempt.values():
        reason_tally[reason.value] = reason_tally.get(reason.value, 0) + 1
    for reason_value, count in sorted(reason_tally.items()):
        print(f"  exempt[{reason_value}] = {count}")  # noqa: T201


@pytest.mark.parametrize("block", _ALL_BASH_BLOCKS, ids=[_block_id(b) for b in _ALL_BASH_BLOCKS])
def test_engrava_invocations_match_the_real_cli(block: CodeBlock) -> None:
    """Every `engrava ...` invocation in a bash block names a real command/option.

    Exempt blocks are skipped, including one filed as
    ``ExemptionReason.USAGE_GRAMMAR_PLACEHOLDER`` (the CLI's own usage-grammar
    line, ``engrava [GLOBAL OPTIONS] COMMAND [ARGS]...``): it genuinely has the
    shape of an invocation, it just names no real command or option, so
    checking it would fail for a reason unrelated to documentation drift. An
    exemption covers only the block's text (line endings normalised) it was
    registered for (see ``EXEMPT_BASH_BLOCKS``): a block edited since then is not
    skipped.
    """
    if block.location in _exempt_locations():
        pytest.skip("exempt bash block: filed under its ExemptionReason, see EXEMPT_BASH_BLOCKS")
    errors = block_invocation_errors(block)
    assert not errors, (
        f"Documentation bash block at {block.location} names a command/option "
        f"that does not exist on the real engrava CLI:\n"
        + "\n".join(f"  {e.line!r}: {e.message}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Failability + controls (not part of the doc census — direct unit checks of
# the checker logic itself, using its own exact mutation strings)
# ---------------------------------------------------------------------------


def test_checker_rejects_a_nonexistent_command() -> None:
    """The exact mutation: `reindex` is not a real engrava command."""
    invocation = "engrava --db mydata.db reindex --parallel 8 --strategy aggressive"
    tokens = _tokenize_shell_line(invocation)
    error = _check_invocation(tokens)
    assert error is not None
    assert "reindex" in error.message
    assert "is not a known engrava command" in error.message


def test_checker_rejects_a_nonexistent_option_on_a_real_command() -> None:
    """The exact mutation: `--purge-everything` is not a real `gc` option."""
    tokens = _tokenize_shell_line("engrava --db mydata.db gc --purge-everything")
    error = _check_invocation(tokens)
    assert error is not None
    assert "--purge-everything" in error.message
    assert "'gc'" in error.message


def test_checker_accepts_group_and_command_level_options_together() -> None:
    """Control: a real, valid line mixing a group-level and a command-level option.

    `--db` lives on the `engrava` group; `--dry-run` lives on `gc`. Getting the
    group/command distinction wrong is the most likely way to false-fire here.
    """
    tokens = _tokenize_shell_line("engrava --db mydata.db gc --dry-run")
    assert _check_invocation(tokens) is None


def test_checker_accepts_group_level_option_with_a_positional_argument() -> None:
    """Control: a real, valid line mixing a group-level option and a positional.

    `query` is the only built-in command that takes a positional argument (its
    MQL string); none of the built-ins combine a positional with a
    command-level long option, so this and the test above together cover the
    group/command/positional space no single real command exercises alone.
    """
    mql = "FIND thoughts WHERE thought_type = 'OBSERVATION' LIMIT 5"
    tokens = _tokenize_shell_line(f'engrava --db my_thoughts.db query "{mql}"')
    assert _check_invocation(tokens) is None


def test_checker_accepts_help_alone() -> None:
    """`--help` alone (with no other bogus option) is a valid line."""
    tokens = _tokenize_shell_line("engrava --no-extensions --help")
    assert _check_invocation(tokens) is None


def _synthetic_block(body: str) -> CodeBlock:
    """A CodeBlock for testing checker logic directly, not read from any real file."""
    return CodeBlock(path=REPO_ROOT / "synthetic.md", rel="synthetic.md", start_line=1, body=body)


def test_checker_accepts_the_option_terminator() -> None:
    """Click's `--` option terminator is not an unknown option.

    `--` before the command name means "stop parsing global options"; the
    next token is still the command, not something `--` itself must be found
    among the recognised options.
    """
    tokens = _tokenize_shell_line("engrava --db demo.db -- info")
    assert _check_invocation(tokens) is None


def test_checker_does_not_attribute_a_piped_commands_flags_to_engrava() -> None:
    """A `|` starts a new segment; the piped command's own flags are not engrava's.

    `--lines=5` is a flag on `head`, not on `info`, so it is not reported as
    an unknown option on `info`.
    """
    block = _synthetic_block("engrava --db demo.db info | head --lines=5")
    assert block_invocation_errors(block) == []


def test_checker_catches_a_second_invocation_after_a_compound_operator() -> None:
    """`a && b` checks `b` too, not just `a`.

    `&&` starts a new segment, so the second invocation's `reindex` is
    checked and reported.
    """
    block = _synthetic_block("engrava --db demo.db info && engrava reindex")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message
    assert "is not a known engrava command" in errors[0].message


def test_checker_fails_loudly_on_an_engrava_line_that_will_not_tokenise() -> None:
    """A line that plausibly invokes `engrava` but is not valid shell input is a violation.

    Silently skipping it would let a malformed example evade every check here.
    """
    block = _synthetic_block("engrava --db demo.db query 'unterminated")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "does not tokenise as shell input" in errors[0].message


def test_checker_does_not_treat_a_quoted_separator_as_a_boundary() -> None:
    """A quoted value equal to a separator's text hides no command.

    The quoted filename `'|'` is `--db`'s value, not a boundary, so `reindex`
    -- not a real command -- is read as the command name and reported.
    """
    block = _synthetic_block("engrava --db '|' reindex")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message
    assert "is not a known engrava command" in errors[0].message


def test_checker_splits_a_semicolon_with_no_surrounding_whitespace() -> None:
    """`;` is a boundary even glued directly to the previous word.

    A library lexer glues `info;` into one word by default; the quote-state
    scan recognises `;` as its own token regardless of adjacent whitespace.
    """
    block = _synthetic_block("engrava info ;engrava reindex")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_does_not_glue_a_command_name_to_a_following_semicolon() -> None:
    """`info;` must not become the (invalid) command name `info;`."""
    block = _synthetic_block("engrava info; engrava info")
    assert block_invocation_errors(block) == []


def test_checker_strips_an_env_assignment_after_a_compound_operator() -> None:
    """A second segment's own leading env assignment is not part of the command word."""
    block = _synthetic_block(
        "engrava info && ENGRAVA_DISABLE_EXTENSIONS=1 engrava reindex",
    )
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_finds_an_invocation_that_is_not_the_lines_first_word() -> None:
    """`engrava` need not be the first word of the line to be checked.

    The command word is detected per segment, so `reindex` is reported
    although `engrava` is not the first word of the line.
    """
    block = _synthetic_block("true && engrava reindex")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_rejects_an_unknown_option_after_help() -> None:
    """`--help` does not bypass validation of later options.

    Real click parses the whole line before acting on an eager option like
    `--help`: `engrava --help --nonexistent` exits 2 with `No such option`.
    """
    tokens = _tokenize_shell_line("engrava --help --nonexistent")
    error = _check_invocation(tokens)
    assert error is not None
    assert "--nonexistent" in error.message


def test_checker_rejects_an_unknown_command_option_after_help() -> None:
    """Same as above, at the command level."""
    tokens = _tokenize_shell_line("engrava info --help --nonexistent")
    error = _check_invocation(tokens)
    assert error is not None
    assert "--nonexistent" in error.message


def test_checker_does_not_let_a_quoted_positional_hide_a_later_option() -> None:
    """A quoted positional equal to an operator's text is not a boundary.

    `query`'s MQL positional argument has no option consuming it by count.
    Each token is tagged as an operator (or not) at tokenising time, while
    quote state is known, so the quoted `'|'` is never taken for a boundary
    and the `--nonexistent` after it is still reached.
    """
    tokens = _tokenize_shell_line("engrava query '|' --nonexistent")
    error = _check_invocation(tokens)
    assert error is not None
    assert "--nonexistent" in error.message


def test_checker_recognises_a_bare_ampersand_as_a_separator() -> None:
    """`&` (not just `&&`) is a real shell control operator too."""
    block = _synthetic_block("engrava info & engrava reindex")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_never_consumes_a_real_operator_as_an_options_value() -> None:
    """`--output ;` must not swallow `;` as `--output`'s value.

    A genuine (unquoted) operator token immediately after an option needing
    a value is never eligible to be that value.
    """
    block = _synthetic_block("engrava export --output ; engrava reindex")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_does_not_treat_a_mid_word_hash_as_a_comment() -> None:
    """`#` only starts a comment at the start of a word."""
    tokens = _tokenize_shell_line("engrava --db demo#tag.db info")
    assert [t.text for t in tokens] == ["engrava", "--db", "demo#tag.db", "info"]


def test_checker_still_treats_a_real_comment_as_a_comment() -> None:
    """Control: a genuine `# comment` at the start of a word is still dropped."""
    tokens = _tokenize_shell_line("engrava info # a real comment")
    assert [t.text for t in tokens] == ["engrava", "info"]


def test_checker_does_not_lose_an_invocation_to_a_quoted_env_value() -> None:
    """A quoted character inside an env-assignment value must not hide the line.

    The `ENGRAVA_DB=a'b'c engrava reindex` line reaches the real,
    quote-aware scanner, so its `reindex` invocation is checked.
    """
    block = _synthetic_block("engrava info\nENGRAVA_DB=a'b'c engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_still_passes_engrava_mentioned_only_as_an_echoed_argument() -> None:
    """Control (decoy): `echo '|' engrava reindex` is not a boundary restart.

    After the `;`, the segment starting with `echo` runs to the end of the
    line (no further unquoted operator), so the `engrava reindex` inside it
    is an argument to `echo`, not a fresh invocation -- this must stay green.
    """
    block = _synthetic_block("engrava info; echo '|' engrava reindex")
    assert block_invocation_errors(block) == []


def test_checker_still_fails_a_mid_word_hash_prefixed_argument() -> None:
    """Control (decoy): `demo#tag.db` is one word, and `reindex` is invalid."""
    block = _synthetic_block("engrava --db demo#tag.db reindex")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_reports_a_malformed_line_even_when_first_word_is_not_engrava() -> None:
    """A malformed line is reported even when its first word is not `engrava`.

    `true && engrava reindex 'unterminated` starts with `true`, yet the line
    is reported, not discarded.
    """
    block = _synthetic_block("engrava info\ntrue && engrava reindex 'unterminated\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "does not tokenise as shell input" in errors[0].message


def test_checker_reports_a_malformed_line_behind_an_env_assignment() -> None:
    """A malformed line behind an env assignment with a quoted space is reported.

    `ENGRAVA_DB='demo db' engrava reindex 'unterminated` is untokenisable, so
    it fails instead of passing.
    """
    block = _synthetic_block("engrava info\nENGRAVA_DB='demo db' engrava reindex 'unterminated\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "does not tokenise as shell input" in errors[0].message


def test_checker_treats_a_dollar_prefixed_line_as_a_command_in_a_transcript() -> None:
    """A session transcript's `$ `-prefixed command is checked as an invocation."""
    block = _synthetic_block("$ engrava --db old.db reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_never_tokenises_a_transcripts_output_line() -> None:
    """Output in a session transcript is not shell input, ever.

    The apostrophe in this line would make it fail to tokenise if treated as
    a command; because the block also has a `$ `-prefixed line, this one is
    recognised as output and skipped entirely.
    """
    block = _synthetic_block(
        "$ engrava --db old.db gc\n"
        "Database schema is at version 11; this engrava build's head "
        "version is 20. Run 'engrava migrate' before running 'gc' on it.\n"
        "$ echo $?\n"
        "1\n",
    )
    assert block_invocation_errors(block) == []


def test_checker_still_reports_a_non_transcript_untokenisable_line() -> None:
    """Control: the tokenise-failure rule still fires outside the transcript convention."""
    block = _synthetic_block("engrava info\ntrue && engrava reindex 'unterminated\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "does not tokenise as shell input" in errors[0].message


def test_checker_ignores_non_engrava_commands_in_a_transcript() -> None:
    """Control: a `$ `-prefixed command that is not engrava is simply not an invocation."""
    block = _synthetic_block("$ echo $?\n1\n")
    assert block_invocation_errors(block) == []


def test_checker_does_not_let_transcript_output_swallow_the_next_command() -> None:
    """An output line's trailing backslash must never join to the next command.

    Lines are classified as commands or output before continuations are
    joined, so a backslash at the end of an output line does not absorb the
    next physical line -- a fresh `$ `-prefixed command is still checked.
    """
    block = _synthetic_block("$ engrava info\noutput \\\n$ engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_does_not_let_a_comment_swallow_the_next_line() -> None:
    """A comment line's trailing backslash must never join to the next line.

    Real bash comments run to end of line regardless of a trailing
    backslash; classifying lines before joining continuations keeps a
    comment from absorbing whatever follows it.
    """
    block = _synthetic_block("# comment \\\nengrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_still_joins_a_genuine_multi_line_command() -> None:
    """Control: a real multi-line command (as in docs/troubleshooting.md) still joins.

    Classifying first must not break the case it was designed around:
    continuation joining still happens, just scoped to an already-started
    command's own lines.
    """
    block = _synthetic_block(
        "engrava --db restored.db --config engrava.yaml restore \\\n"
        "  -i backup.jsonl --clear --re-embed\n",
    )
    assert block_invocation_errors(block) == []


def test_checker_reports_a_redirection_instead_of_mis_splitting_on_it() -> None:
    """`2>&1`'s `&` is not a real `&&`-style separator.

    Redirection is out of scope, so the line is reported as a redirection
    instead of being split at the `&`.
    """
    block = _synthetic_block("engrava info 2>&1 --nonexistent\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "redirection" in errors[0].message


def test_checker_does_not_mistake_a_comment_arrow_for_redirection() -> None:
    """Control: `# ->` in a comment is not a redirection operator; it is never reached."""
    block = _synthetic_block("engrava info # -> prints database stats\n")
    assert block_invocation_errors(block) == []


def test_checker_does_not_continue_a_line_from_inside_a_comment() -> None:
    """A trailing backslash inside a comment does not continue the line.

    Real bash comments run to end of line regardless of a trailing
    backslash, so `reindex` on the next line is checked as its own command.
    """
    block = _synthetic_block("engrava info # comment \\\nengrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_does_not_insert_whitespace_when_joining_a_continuation() -> None:
    """Backslash-newline removal inserts no character, not even a space.

    A real shell removes the backslash-newline pair with nothing in its
    place, so `engra` + `va reindex` produces `engrava reindex`, whose
    `reindex` is reported.
    """
    block = _synthetic_block("engrava info\nengra\\\nva reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_passes_the_no_space_join_mirror() -> None:
    """Control (the mirror of the no-space join above): joining inserts no space.

    `engrava in\\` followed by `fo` must become `engrava info`, a valid
    invocation -- not `engrava in fo`, which a space-inserting join would
    produce and which names no real command.
    """
    block = _synthetic_block("engrava in\\\nfo\n")
    assert block_invocation_errors(block) == []


def test_checker_strips_a_repeated_prompt_on_a_continuation_line() -> None:
    """A `$ ` on a continuation line is stripped, not literal or an error.

    Inside a transcript, a continued command's next physical line is a
    continuation of that command, never a fresh prompt -- a `$ ` there is
    documentation convention (or a formatting slip), not something a real
    shell would see as input.
    """
    block = _synthetic_block("$ engrava \\\n$ info\n")
    assert block_invocation_errors(block) == []


def test_checker_resets_comment_detection_after_an_operator() -> None:
    """A comment right after ``;``/``|``/``&`` is still a comment.

    The comment tracker resets word-start after an operator as well as after
    whitespace, so ``engrava info;# comment`` recognises the ``#`` as
    starting a comment right after the ``;`` the way real bash does.
    """
    block = _synthetic_block("engrava info;# comment \\\nengrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_refuses_an_escaped_trailing_backslash() -> None:
    """An even run of backslashes is an escaped, literal one.

    Bash reads a trailing ``\\\\`` as one escaped backslash ending the
    command, not a continuation marker -- this checker does not model
    escaping, so it reports the line instead of guessing at it.
    """
    block = _synthetic_block("engrava info \\\\\nengrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 2
    assert "cannot be reconstructed" in errors[0].message
    assert "reindex" in errors[1].message


def test_checker_refuses_a_backslash_inside_an_open_quote() -> None:
    """A quote spanning the join point is not reconstructed.

    Bash keeps both the backslash and the newline as literal content inside
    a single-quoted string; this checker does not model a quote spanning a
    join point, so it reports the line instead.
    """
    block = _synthetic_block("engrava 'in\\\nfo'\n")
    errors = block_invocation_errors(block)
    assert errors
    assert "cannot be reconstructed" in errors[0].message


def test_checker_treats_an_escaped_trailing_space_as_a_real_boundary() -> None:
    """A backslash followed by trailing whitespace does not continue.

    Checked on the raw, un-stripped line: the backslash is not the literal
    last character, so this is not a continuation at all -- the physical
    line ends normally, and the next line is its own, separate command.
    """
    block = _synthetic_block("engrava info \\ \nengrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_strips_a_dangling_continuation_on_the_last_line() -> None:
    """A block's final line ending in a real marker is not `info\\`.

    There is no next physical line to join with; a real shell reads the
    same backslash-newline-EOF and simply ends the command there.
    """
    block = _synthetic_block("engrava info\\\n")
    assert block_invocation_errors(block) == []


def test_checker_still_joins_and_fails_the_no_space_mirror() -> None:
    """Control: the ordinary, reconstructable case must still work."""
    block = _synthetic_block("engra\\\nva reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_strips_pipeline_negation_before_the_command() -> None:
    """A leading, unquoted ``!`` does not hide the invocation it negates.

    An unquoted ``!`` is bash's pipeline-negation operator when it is the
    first word of a segment -- unambiguous there, no context needed -- so it
    is stripped and the invocation that follows is found.
    """
    block = _synthetic_block("! engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_accepts_a_negated_valid_command() -> None:
    """Control: negating a real, valid invocation must not itself be an error."""
    assert block_invocation_errors(_synthetic_block("! engrava info\n")) == []


def test_checker_strips_negation_after_a_compound_operator() -> None:
    """Negation composes with an existing segment boundary (``;``)."""
    block = _synthetic_block("true; ! engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_strips_negation_before_an_env_assignment() -> None:
    """Negation composes with a following env assignment, in that order."""
    block = _synthetic_block("! ENGRAVA_DISABLE_EXTENSIONS=1 engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_strips_a_run_of_pipeline_negations() -> None:
    """Bash allows repeated ``!``, not just one.

    ``bash -c '! ! echo hi'`` prints ``hi`` and exits 0 -- a double negation
    still runs the command, so ``! ! engrava reindex`` genuinely invokes
    ``engrava reindex``. The whole leading run of ``!`` is skipped, so the
    invocation is found.
    """
    block = _synthetic_block("! ! engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_strips_a_negation_run_after_a_compound_operator() -> None:
    """A negation run composes with an existing segment boundary."""
    block = _synthetic_block("true && ! ! engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_strips_a_negation_run_before_an_env_assignment() -> None:
    """A negation run composes with a following env assignment, in that order."""
    block = _synthetic_block("! ! ENGRAVA_X=1 engrava reindex\n")
    errors = block_invocation_errors(block)
    assert len(errors) == 1
    assert "reindex" in errors[0].message


def test_checker_accepts_a_doubly_negated_valid_command() -> None:
    """Control: a negation run over a real, valid invocation must not itself be an error."""
    assert block_invocation_errors(_synthetic_block("! ! engrava info\n")) == []


def _edit_registered_block(
    monkeypatch: pytest.MonkeyPatch,
    rel: str,
    anchor: str,
    suffix: str,
) -> CodeBlock:
    """Swap one registered exempt block for a copy with ``suffix`` appended to its text.

    The replacement is made in this module's block list, so the registry, the
    census and the per-block test all see the edited block exactly as they
    would see an edited documentation page.
    """
    original = _unique_block(rel, anchor)
    edited = dataclasses.replace(original, body=original.body + suffix)
    monkeypatch.setitem(
        globals(),
        "_ALL_BASH_BLOCKS",
        [edited if block is original else block for block in _ALL_BASH_BLOCKS],
    )
    return edited


def test_an_edited_exempt_block_is_checked_again_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command appended to an exempt block is neither hidden nor still exempt.

    ``reindex`` is not a real command, so the raw checker flags it. The
    exemption was granted for the block's original text, so it no longer
    applies: the per-block test runs (rather than skips) and fails on the
    appended command, and the digest census names the block and prints the
    digest to register if the edit was deliberate.
    """
    edited = _edit_registered_block(
        monkeypatch, "README.md", "pip install engrava", "\nengrava reindex"
    )

    assert any("reindex" in error.message for error in block_invocation_errors(edited))
    assert edited.location not in _exempt_locations()

    with pytest.raises(AssertionError, match="reindex"):
        test_engrava_invocations_match_the_real_cli(edited)

    with pytest.raises(AssertionError, match=re.escape(edited.location)) as census:
        test_exempt_bash_registry_digests_match()
    message = str(census.value)
    assert block_digest(edited.body) in message
    assert "EXEMPT_BASH_BLOCKS" in message


def test_an_edit_that_adds_no_command_still_fails_the_digest_census(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a harmless edit to an exempt block is reported, not silently absorbed.

    The block is checked again, and passes (nothing in it is an ``engrava``
    invocation), but the registered digest no longer matches, so a reviewer has
    to re-register the block deliberately.
    """
    edited = _edit_registered_block(monkeypatch, "README.md", "pip install engrava", " --upgrade")

    assert block_invocation_errors(edited) == []
    assert edited.location not in _exempt_locations()
    with pytest.raises(AssertionError, match=re.escape(edited.location)):
        test_exempt_bash_registry_digests_match()

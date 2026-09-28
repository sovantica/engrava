"""Shared helpers for the documentation-example test suite.

The documentation tests treat the project's Markdown files (``README.md``
and everything under ``docs/``) as a source of executable truth. These
helpers locate the Markdown files and extract their fenced ``python``
code blocks so the individual test modules can compile, scan, or execute
them.

A fenced block follows CommonMark's fenced-code rule: an opener is a run of
three or more backticks or tildes (optionally followed by an info string,
e.g. ``python``); the closer is a run of the same character at least as long
as the opener, with only trailing whitespace after it. A leading Markdown
blockquote marker (``>``, with an optional following space) is stripped
**only when the fence's own opening line carries one** -- decided once per
block, from its opener, and then applied consistently to that block's closer
and body. An ordinary (non-blockquoted) fence is parsed exactly as written,
so a literal ``>`` at the start of a body line (real content, not a
container marker) is preserved rather than silently discarded. Indented
blocks are supported; the captured body is dedented to the fence's
indentation. A fence that is opened but never closed before end of file --
including one whose container prefix this scan cannot resolve consistently
between opener and closer -- raises ``ValueError`` rather than silently
dropping or misreading the block.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# tests/docs/_md_blocks.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"

# Markdown lets a fence use either backticks or tildes, three or more of
# them; a fence opened with one character type is closed only by the same
# type, by a run at least as long as the opener's.
_FENCE_CHARS = ("`", "~")
_MIN_FENCE_LENGTH = 3


@dataclass(frozen=True)
class CodeBlock:
    """A fenced ``python`` code block extracted from a Markdown file.

    Attributes:
        path: Absolute path to the source Markdown file.
        rel: Path relative to the repository root (for messages).
        start_line: 1-based line number of the first body line.
        body: The dedented block body (without the fences).

    """

    path: Path
    rel: str
    start_line: int
    body: str

    @property
    def location(self) -> str:
        """Return a human-readable ``file:line`` locator."""
        return f"{self.rel}:{self.start_line}"


def markdown_files() -> list[Path]:
    """Return all documentation Markdown files in scope, sorted.

    Returns:
        ``README.md`` plus every ``*.md`` under ``docs/`` (recursive),
        in a stable sorted order so test parametrisation is
        deterministic.

    """
    files = [README, *sorted(DOCS_DIR.rglob("*.md"))]
    return [f for f in files if f.is_file()]


def extract_python_blocks(path: Path) -> list[CodeBlock]:
    """Extract every fenced ``python`` code block from one Markdown file.

    Args:
        path: The Markdown file to scan.

    Returns:
        A list of :class:`CodeBlock` in document order. Empty when the
        file contains no ``python`` fences.

    """
    return extract_fenced_blocks(path, "python")


def _strip_blockquote_prefix(line: str) -> str:
    """Strip one leading Markdown blockquote marker (``>``) from a line.

    The marker may be preceded by up to three spaces and is optionally
    followed by a single space, per CommonMark; only one level is stripped
    (a fence nested inside more than one blockquote level is not a case any
    current document needs).
    """
    lstripped = line.lstrip(" ")
    if len(line) - len(lstripped) > 3 or not lstripped.startswith(">"):
        return line
    after_marker = lstripped[1:]
    return after_marker.removeprefix(" ")


def _match_opening_fence(stripped_line: str) -> tuple[str, int, str] | None:
    """Return ``(fence_char, run_length, info_string)`` if the line opens a fence."""
    if not stripped_line or stripped_line[0] not in _FENCE_CHARS:
        return None
    char = stripped_line[0]
    length = len(stripped_line) - len(stripped_line.lstrip(char))
    if length < _MIN_FENCE_LENGTH:
        return None
    return char, length, stripped_line[length:].strip()


def _matches_closing_fence(stripped_line: str, fence_char: str, fence_length: int) -> bool:
    """Whether the line closes a fence opened with ``fence_char`` repeated ``fence_length`` times.

    Per CommonMark, the closer is a run of the same character at least as
    long as the opener, followed by nothing but trailing whitespace.
    """
    if not stripped_line or stripped_line[0] != fence_char:
        return False
    run = len(stripped_line) - len(stripped_line.lstrip(fence_char))
    return run >= fence_length and stripped_line[run:].strip() == ""


def _iter_fenced_blocks(path: Path) -> list[tuple[str, int, str]]:
    """Yield ``(info_string, body_start_line, body)`` for every fenced block in a file.

    The shared scan both public extractors below build on -- implements
    CommonMark's fenced-code rule (see the module docstring) once, so a fix
    to fence detection (tildes, blockquotes, closer length) never needs to
    be made twice.

    Whether a blockquote prefix is stripped is decided **once per block**,
    from its opener: a plain fence (the overwhelming common case) is parsed
    exactly as written, so a line starting with a literal ``>`` inside its
    body -- an ordinary character, not a container marker -- is never
    touched. Only a fence whose *own opening line* carries a ``>`` prefix has
    that prefix stripped, consistently, from its closer and its body too.

    Raises:
        ValueError: If a fence is opened but never closed before end of
            file -- Markdown lets a fence run to the end of the document,
            which would otherwise silently drop the block (and everything
            after it) instead of failing loudly. The same applies to a
            fence whose opener carries a container prefix (blockquote,
            optionally wrapping a list item) this scan cannot resolve
            consistently between the opener and the closer: it fails the
            same way, rather than silently mis-reading the block.

    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()

    results: list[tuple[str, int, str]] = []
    in_block = False
    in_blockquote = False
    fence_char = ""
    fence_length = 0
    indent = 0
    info = ""
    body_lines: list[str] = []
    body_start = 0

    for index, raw in enumerate(lines):
        if not in_block:
            plain = raw.lstrip()
            opened = _match_opening_fence(plain)
            blockquoted = False
            if opened is None:
                bq_line = _strip_blockquote_prefix(raw)
                if bq_line != raw:
                    opened = _match_opening_fence(bq_line.lstrip())
                    blockquoted = opened is not None
            if opened is not None:
                fence_char, fence_length, info = opened
                in_block = True
                in_blockquote = blockquoted
                effective = _strip_blockquote_prefix(raw) if blockquoted else raw
                stripped = effective.lstrip()
                indent = len(effective) - len(stripped)
                body_lines = []
                body_start = index + 2  # 1-based, first line after the fence
            continue
        effective = _strip_blockquote_prefix(raw) if in_blockquote else raw
        stripped = effective.lstrip()
        if _matches_closing_fence(stripped, fence_char, fence_length):
            body = "\n".join(_dedent(line, indent) for line in body_lines)
            results.append((info, body_start, body))
            in_block = False
            continue
        body_lines.append(effective)

    if in_block:
        container = "blockquoted " if in_blockquote else ""
        msg = (
            f"{rel}: a {container}fenced block opened at line {body_start - 1} "
            f"({fence_char * fence_length}{info}) is never closed before end of file "
            f"(or its closer's container prefix does not match its opener's)"
        )
        raise ValueError(msg)

    return results


def extract_fenced_blocks(path: Path, language: str) -> list[CodeBlock]:
    """Extract every fenced block of one info-string language from a Markdown file.

    ``python`` blocks are the executable-example surface; other languages carry
    documented *output* (a ``text`` block showing what an example prints), which
    a test can compare against a real run so the page cannot promise a result it
    does not produce.

    Args:
        path: The Markdown file to scan.
        language: The fence info-string language, e.g. ``"python"`` or ``"text"``.
            Matched by *prefix*: ``"python"`` also matches a hypothetical
            ``python3`` fence. Use :func:`extract_exact_fenced_blocks` for an
            exact match.

    Returns:
        A list of :class:`CodeBlock` in document order. Empty when the file
        contains no fence of that language.

    Raises:
        ValueError: If a fence is opened but never closed before end of file.

    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    return [
        CodeBlock(path=path, rel=rel, start_line=start, body=body)
        for info, start, body in _iter_fenced_blocks(path)
        if info.startswith(language)
    ]


def all_python_blocks() -> list[CodeBlock]:
    """Return every ``python`` code block across all documentation files."""
    blocks: list[CodeBlock] = []
    for path in markdown_files():
        blocks.extend(extract_python_blocks(path))
    return blocks


def extract_all_fenced_blocks(path: Path) -> list[tuple[str, CodeBlock]]:
    """Extract every fenced code block in a Markdown file, with its exact info string.

    ``extract_fenced_blocks`` matches an info string by *prefix*: asking it for
    ``"python"`` also collects a hypothetical ```python3`` block, and asking it
    for the empty string collects **every** fenced block regardless of
    language (every fence "starts with" the empty string). That is fine for a
    single known language, but a census that must partition *every* fenced
    block -- including a bare fence carrying no language at all -- needs the
    info string matched exactly instead. This function is the one place that
    distinction is made.

    Args:
        path: The Markdown file to scan.

    Returns:
        ``(info_string, block)`` pairs in document order. ``info_string`` is
        ``""`` for a bare fence.

    Raises:
        ValueError: If a fence is opened but never closed before end of file.

    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    return [
        (info, CodeBlock(path=path, rel=rel, start_line=start, body=body))
        for info, start, body in _iter_fenced_blocks(path)
    ]


def extract_exact_fenced_blocks(path: Path, language: str) -> list[CodeBlock]:
    """Extract fenced blocks whose info string equals ``language`` exactly.

    Use this instead of :func:`extract_fenced_blocks` whenever the empty
    string must mean "a bare fence, no language" rather than "every fence" --
    see :func:`extract_all_fenced_blocks` for why the two disagree.

    Args:
        path: The Markdown file to scan.
        language: The exact info string to match, e.g. ``"bash"`` or ``""``
            for a bare fence.

    Returns:
        A list of :class:`CodeBlock` in document order.

    """
    return [block for info, block in extract_all_fenced_blocks(path) if info == language]


def all_blocks_by_exact_language() -> dict[str, list[CodeBlock]]:
    """Group every fenced block across all documentation files by exact info string.

    The keys are exact fence info strings (``""`` for a bare fence). This is
    the source of truth for a census over every language, not just ``python``.
    """
    grouped: dict[str, list[CodeBlock]] = defaultdict(list)
    for path in markdown_files():
        for info, block in extract_all_fenced_blocks(path):
            grouped[info].append(block)
    return grouped


class ExemptionReason(Enum):
    """Closed vocabulary for why a non-``python`` fenced block is not checked.

    A free-text reason field lets exemptions multiply invisibly -- the
    ``python`` census's 122 compile-only blocks hide behind 113 distinct
    free-text reasons, so nobody can count how many blocks are exempt for
    which cause, or notice one category quietly growing. Every non-``python``
    exemption in this suite cites one of these members instead, so a census
    test can print an exact tally per reason.

    Extend this enum deliberately when a genuinely new category of
    unaddressable block appears -- do not reach for a free-text string.
    """

    #: A ``bash`` line/block that does not invoke ``engrava`` at all (``pip
    #: install``, ``make``, ``sqlite3``, ``python -m ...``, ...).
    NOT_AN_ENGRAVA_INVOCATION = "not-an-engrava-invocation"
    #: A syntax-grammar template with placeholder tokens (``COMMAND``,
    #: ``<table>``, ``[WHERE <bool-expr>]``, ...), not a concrete example --
    #: it genuinely has the shape of an invocation/query, it just names no
    #: real command, option, or literal value.
    USAGE_GRAMMAR_PLACEHOLDER = "usage-grammar-placeholder"
    #: An architecture diagram, pipeline sketch, directory tree, or a
    #: transcript of real program output / an error message -- prose
    #: illustration, not a checkable claim about a command, flag, or key.
    DIAGRAM_OR_TRANSCRIPT = "diagram-or-transcript"
    #: A scoring formula or an algorithm sketch (loop/pseudocode) describing
    #: how a value is computed -- notation, not a diagram, transcript, or
    #: query.
    FORMULA_OR_PSEUDOCODE = "formula-or-pseudocode"
    #: A concrete, bare MindQL query example shown as language syntax, not
    #: wrapped in the ``engrava query "..."`` CLI invocation that would make
    #: it checkable.
    MINDQL_QUERY = "mindql-query"
    #: Raw SQL run directly against the SQLite file (a maintenance command or
    #: an escape-hatch query), not through any engrava-owned surface.
    RAW_SQL_ILLUSTRATION = "raw-sql-illustration"
    #: A block that shows the same top-level key twice, side by side, to
    #: illustrate alternate forms -- the real duplicate-key check correctly
    #: reports this as a duplicate key (it is one, by construction), so
    #: checking the whole block as one document can only ever report that,
    #: never validate either form on its own.
    DUPLICATE_KEY_ALTERNATE_FORMS = "duplicate-key-alternate-forms"


class CompileOnlyReason(Enum):
    """Closed vocabulary for why a compile-only ``python`` documentation block is not
    executed or behaviour-asserted.

    Not every member names something that genuinely *cannot* be executed --
    ``NO_ASSERTABLE_CLAIM``'s own definition says the block would run cleanly exactly
    as written, and ``HARNESS_SHAPE_MISMATCH`` names a limitation of the execute-layer
    harness, not a property of the example. "Not executed" is the accurate claim; "can
    never be executed" is not, and members should not be extended on the assumption
    that it is.

    Mirrors :class:`ExemptionReason` for the ``python`` fence's own, larger, previously
    unclassified tier: the 122 blocks in ``COMPILE_ONLY`` (see
    ``test_docs_examples_coverage.py``) used to carry 113 distinct free-text reasons --
    122 against 113 is nine duplicate uses, not a one-sentence-per-nine-blocks ratio --
    so nobody could count how many blocks were exempt for which cause, or notice one
    cause quietly growing. Every ``COMPILE_ONLY``
    entry now cites exactly one of these members in addition to its free-text note; the
    note may still say something the member cannot (which exact API, which specific
    test mirrors it) -- the member is what gets counted, the note is what gets read.

    Extend this enum deliberately when a genuinely new reason a block is not executed
    or behaviour-asserted appears -- never by grepping the old free-text sentence for
    a keyword.
    """

    #: A plain, default-configured store or connection is the block's only obstacle,
    #: and literal arguments (a string, an int, a default-valued keyword) cover
    #: everything else it needs. Concretely: building
    #: ``aiosqlite.connect(":memory:")`` with ``row_factory = aiosqlite.Row``, wrapping
    #: it in a bare ``SqliteEngravaCore(conn)``, and calling ``ensure_schema()`` -- with
    #: no extra constructor keyword, no wrapper class, and no other collaborator --
    #: reproduces everything the block's own undefined names need. A bare fragment
    #: that calls straight into ``store``/``conn`` (``await store.recall(...)``), or a
    #: helper function that takes one as its only out-of-the-ordinary parameter (its
    #: other parameters, if any, being plain literals such as a query string or a
    #: cycle number) and is never invoked in the block, both qualify. A block needing
    #: a *differently built* store (extra constructor keywords, a wrapper class) or
    #: any non-literal collaborator does not -- see ``REQUIRES_SPECIALLY_CONFIGURED_STORE``
    #: and ``UNDEFINED_DOMAIN_VALUE`` respectively.
    ASSUMES_STORE_OR_CONNECTION = "assumes-store-or-connection"

    #: The block needs more than a plain store/connection: an undefined value or
    #: callable standing in for domain-specific data a generic fixture cannot
    #: manufacture as a literal -- a specific id (``src_id``, ``some_thought_id``), a
    #: record or collection of records (``fact``, ``transient_thought``, the
    #: ``items``/``first``/``second``/``link`` arguments of a helper that builds or
    #: links them), a vector (``embedding``), a caller-supplied callback
    #: (``my_embed_fn``, ``my_llm``), a loop-control flag (``running``), a name only
    #: defined or imported in a sibling block on the same page (``observation``,
    #: ``RecencyBoostHooks``, ``RECENT_COMMAND``, an unimported ``ThoughtRecord``), or a
    #: second collaborator beyond the store itself (a caller-owned lock, an embedding
    #: provider instance).
    UNDEFINED_DOMAIN_VALUE = "undefined-domain-value"

    #: The block needs a store built differently from the plain default -- an extra
    #: constructor keyword that changes what the store can do (``journal_enabled=True``
    #: before ``store.journal`` is usable), or the store wrapped in a class that changes
    #: its behaviour (``ReadOnlyEngrava(store)`` before a write raises
    #: ``ReadOnlyViolationError``, which is the block's whole point). A plain
    #: ``SqliteEngravaCore(conn)`` with schema applied does not exhibit what the block
    #: demonstrates; a differently-configured one would.
    REQUIRES_SPECIALLY_CONFIGURED_STORE = "requires-specially-configured-store"

    #: The block's entire content is a definition -- a class, a ``Protocol``, a
    #: subclass, a bare method/function signature stub (body is ``...``), or a
    #: plugin/extension object -- never invoked or asserted within the block, so no
    #: collaborator would make it prove anything by itself.
    DEFINITION_ONLY = "definition-only"

    #: The block's own point is bound to a real on-disk path in a way a disposable,
    #: single-process fixture cannot cheaply substitute for -- not necessarily because
    #: the file must already exist (a fresh ``connect("engrava.db")`` call creates its
    #: own file just fine; that alone is not this reason). Concretely: an
    #: ``engrava.yaml`` config file ``from_config`` must read with real content, a data
    #: directory ``EngravaManager`` manages across several stores, on-disk migration
    #: SQL files referenced by a package-relative path, a corrupted on-disk journal to
    #: exercise a startup failure, or an example whose claim is that a named ``.db``
    #: file's data survives **across separate process invocations** -- a property one
    #: subprocess run cannot demonstrate no matter how cheaply it creates the file.
    REQUIRES_ON_DISK_ARTIFACT = "requires-on-disk-artifact"

    #: The block needs a live external system a test cannot cheaply fake: real model
    #: weights (``SentenceTransformerProvider``), a real network endpoint and API key
    #: (``OpenAICompatibleProvider``, ``HuggingFaceProvider``), or a running local
    #: service (``OllamaProvider``). An optional Python dependency that merely is not
    #: guaranteed installed (e.g. ``prometheus_client``) is not a live external system
    #: by itself -- no service, endpoint, credential, or network is involved, only a
    #: package that may be absent. A current entry citing this reason for that alone
    #: is a known open question, not a member of what this reason actually names.
    REQUIRES_LIVE_EXTERNAL_SERVICE = "requires-live-external-service"

    #: The block would run cleanly exactly as written -- no undefined collaborator, no
    #: disk, no network -- but asserts or prints nothing, so there is no claim for a
    #: test to check even though nothing prevents running it: a standalone capability
    #: probe against stdlib ``sqlite3``, a bare logging-configuration statement, or an
    #: object construction with no observable outcome.
    NO_ASSERTABLE_CLAIM = "no-assertable-claim"

    #: The block is self-contained and would behave correctly if run, but its shape
    #: does not fit the execute layer's required entrypoint convention (an
    #: ``asyncio.run(main())``-wrapped script) -- e.g. a synchronous class definition
    #: plus wiring that never awaits anything.
    HARNESS_SHAPE_MISMATCH = "harness-shape-mismatch"


def _dedent(line: str, indent: int) -> str:
    """Strip up to ``indent`` leading spaces from a captured body line."""
    stripped = line[:indent]
    if stripped.strip() == "":
        return line[indent:]
    return line.lstrip()

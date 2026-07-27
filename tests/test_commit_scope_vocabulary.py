"""Guard: the commit-scope vocabulary has one source, and the docs must match it.

``commit-scopes.json`` is the single machine-readable source of the vocabulary, in
two tiers (canonical, and deprecated spellings kept only so that unrewritable
history lints clean).  ``.commitlintrc.js`` reads that file and spreads both tiers
into its ``scope-enum``; ``docs/scopes.md`` describes the same vocabulary in prose
for contributors.  Until this module existed the vocabulary was written out twice,
in two grammars, agreeing by discipline alone.

What this module checks:

1. **The documented tiers equal the JSON tiers**, in both directions.  Both sides
   are parsed, never restated here — a list of expected scopes in this file would
   be a third hand-maintained copy and would reintroduce the defect it closes.
2. **The config contains the pinned wiring text** — the two lines of
   ``.commitlintrc.js`` that load the JSON file and spread it into the enum are
   present somewhere in the file.
3. **The word "canonical" occurs in every deprecated description**, so a discouraged
   spelling cannot be listed while ignoring the question of what to write instead.

What this module does **not** check.  Property 2 asserts **presence, never absence**.
It catches the realistic accidental edit — someone replacing the wiring with a
literal list — and it cannot show that no second, contradicting line exists: a
``require`` with a ``.push()``, a computed rule key, a rule rebuilt after the config
object exists, or an alternate rc file all leave the pinned text sitting there
untouched.  An earlier version tried to bound that with occurrence counts over a
comment-stripped copy, and the strip was unsound in both directions — a ``//``
inside a string URL swallowed live code, and a comment between two tokens joined
them into the pinned text — so a real second mutation counted as clean.  Proving
absence over text needs a parser; this is a tripwire.  Absence is the job of the
``Verify commitlint admits the same scope set`` step in
``.github/workflows/commitlint.yml``,
which asks commitlint for the effective rule.  That step checks the rule resolved by
its own default-root invocation; it detects static drift visible through that
resolution path, but does not prove that later lint commands use the same
configuration when their flags, working directory, environment, parser or ignore
behaviour differ.  ``.commitlintrc.js`` also exempts whole classes of message from
linting — subjects starting with merge or revert, and any message containing the
substring ``Signed-off-by: dependabot[bot]`` anywhere in it, from any author and
unauthenticated — and for those the enum is never consulted at all.

Parsing JavaScript is deliberately not attempted.  An earlier version of this guard
did, and a single-quoted entry defeated it: the parser read fewer scopes than the
file declared, and a short read compares equal.  JSON has one string grammar, so
that read is total.  Every parse failure raises rather than returning a smaller
answer, and each parse result is asserted non-empty: two empty sets compare equal,
so a silent under-read is the one failure mode a set comparison cannot see.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SCOPES_PATH: Final = REPO_ROOT / "commit-scopes.json"
CONFIG_PATH: Final = REPO_ROOT / ".commitlintrc.js"
DOCS_PATH: Final = REPO_ROOT / "docs" / "scopes.md"

#: Tier name in commit-scopes.json -> heading prefix in docs/scopes.md.
TIERS: Final[dict[str, str]] = {
    "canonical": "### Canonical",
    "deprecated": "### Deprecated",
}

#: The two lines of .commitlintrc.js that load the JSON source and spread it into
#: the enum.  Their *presence* is asserted, in the file as written, with whitespace
#: runs collapsed so that re-indenting or rewrapping them passes.  Nothing about
#: their uniqueness or about the absence of other code is asserted — see this
#: module's docstring for why text cannot carry that.
REQUIRE_LINE: Final = 'const SCOPES = require("./commit-scopes.json");'
RULE_LINE: Final = '"scope-enum": [2, "always", [...SCOPES.canonical, ...SCOPES.deprecated]],'

_SCOPES_HEADING: Final = "## Scopes"
_SCOPE_NAME: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BACKTICKED: Final = re.compile(r"`([^`]+)`")
_ANY_LIST_MARKER: Final = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
_EM_DASH: Final = "—"


class VocabularyParseError(AssertionError):
    """A vocabulary source could not be parsed the way this guard requires."""


def json_tiers() -> dict[str, list[str]]:
    """Load the vocabulary from ``commit-scopes.json``.

    Returns:
        Mapping from tier name to the scope names it declares, in file order.

    Raises:
        VocabularyParseError: If the file is not an object carrying exactly the
            expected tiers, each a list of strings.
    """
    loaded = json.loads(SCOPES_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or set(loaded) != set(TIERS):
        msg = f"{SCOPES_PATH} must be an object with exactly the tiers {sorted(TIERS)}"
        raise VocabularyParseError(msg)
    tiers: dict[str, list[str]] = {}
    for tier, names in loaded.items():
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            msg = f"{SCOPES_PATH} tier {tier!r} must be a list of strings"
            raise VocabularyParseError(msg)
        tiers[tier] = names
    return tiers


def _normalised_config() -> str:
    """Return ``.commitlintrc.js`` with every whitespace run collapsed to a space.

    The file is read as written — comments included.  Nothing here tries to tell
    code from comment: a previous version stripped ``//`` comments to make a token
    count meaningful, and the strip was unsound in both directions (a ``//`` inside
    a string URL swallowed live code; a comment between two tokens joined them into
    the pinned text).  Presence of a substring needs no such distinction.

    Returns:
        The config source as a single normalised line, so that re-indenting or
        rewrapping the pinned lines passes while a change to their tokens does not.
    """
    return " ".join(CONFIG_PATH.read_text(encoding="utf-8").split())


def _scopes_section() -> list[str]:
    """Return the lines of the ``## Scopes`` section of ``docs/scopes.md``.

    Returns:
        Every line after the heading, up to the next ``##`` heading.

    Raises:
        VocabularyParseError: If the section is missing.
    """
    lines = DOCS_PATH.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != _SCOPES_HEADING:
            continue
        end = next(
            (later for later in range(index + 1, len(lines)) if lines[later].startswith("## ")),
            len(lines),
        )
        return lines[index + 1 : end]
    msg = f'no "{_SCOPES_HEADING}" section in {DOCS_PATH}'
    raise VocabularyParseError(msg)


def _tier_bullets(tier: str) -> list[str]:
    """Return the bullet texts under one tier heading, wrapped lines joined.

    Every list-like line in the section must be a top-level ``- `` bullet: an
    alternative marker (``* ``, ``+ ``, a nested bullet) would otherwise read as
    prose and the scope it declares would go unseen while this test stayed green.

    Args:
        tier: Key of :data:`TIERS`.

    Returns:
        One string per bullet, without the ``- `` marker.

    Raises:
        VocabularyParseError: If a list-like line is not a ``- `` bullet, or the
            tier heading carries no bullets at all.
    """
    heading = TIERS[tier]
    bullets: list[str] = []
    in_tier = False
    for line in _scopes_section():
        if line.startswith("### "):
            in_tier = line.startswith(heading)
            continue
        if _ANY_LIST_MARKER.match(line) and not line.startswith("- "):
            msg = f"unrecognised list marker in the scope tables of {DOCS_PATH}: {line!r}"
            raise VocabularyParseError(msg)
        if not in_tier:
            continue
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif bullets and line.strip() and line[:1].isspace():
            bullets[-1] = f"{bullets[-1]} {line.strip()}"
    if not bullets:
        msg = f"no bullets under {heading!r} in {DOCS_PATH}"
        raise VocabularyParseError(msg)
    return bullets


def _split_bullet(bullet: str) -> tuple[list[str], str]:
    """Split one scope bullet into its declared names and its description.

    A bullet declares its scope name(s) as backticked tokens before the em dash;
    anything after the em dash is prose and may cite other scopes freely.

    Args:
        bullet: Bullet text without the ``- `` marker.

    Returns:
        The declared scope names and the description that follows them.

    Raises:
        VocabularyParseError: If the bullet lacks an em dash or a backticked name.
    """
    if _EM_DASH not in bullet:
        msg = f"bullet without an em-dash description: {bullet!r}"
        raise VocabularyParseError(msg)
    head, _, description = bullet.partition(_EM_DASH)
    declared = _BACKTICKED.findall(head)
    if not declared:
        msg = f"bullet declares no backticked scope: {bullet!r}"
        raise VocabularyParseError(msg)
    return declared, description.strip()


def docs_tier(tier: str) -> list[str]:
    """Parse one tier of the scope vocabulary out of ``docs/scopes.md``.

    Args:
        tier: Key of :data:`TIERS`.

    Returns:
        The scope names in document order.
    """
    names: list[str] = []
    for bullet in _tier_bullets(tier):
        names.extend(_split_bullet(bullet)[0])
    return names


def docs_tier_descriptions(tier: str) -> dict[str, str]:
    """Return ``scope -> description`` for one documented tier.

    Args:
        tier: Key of :data:`TIERS`.

    Returns:
        Mapping from each declared scope name to the prose after its em dash.
    """
    described: dict[str, str] = {}
    for bullet in _tier_bullets(tier):
        declared, description = _split_bullet(bullet)
        for name in declared:
            described[name] = description
    return described


def _assert_wellformed(source: str, tier: str, names: list[str]) -> None:
    """Assert a parsed tier is non-empty, duplicate-free and syntactically valid.

    Args:
        source: Human-readable name of the file the names came from.
        tier: Tier key the names belong to.
        names: Parsed scope names, in source order.
    """
    assert names, f"{source} tier {tier!r} parsed as empty — the parser or the file is broken"
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"{source} tier {tier!r} repeats {duplicates}"
    malformed = sorted(name for name in names if not _SCOPE_NAME.match(name))
    assert not malformed, f"{source} tier {tier!r} has malformed scope names {malformed}"


def test_json_source_is_wellformed() -> None:
    """The single source parses to non-empty, duplicate-free, disjoint tiers."""
    tiers = json_tiers()
    for tier, names in tiers.items():
        _assert_wellformed(SCOPES_PATH.name, tier, names)
    overlap = set(tiers["canonical"]) & set(tiers["deprecated"])
    assert not overlap, f"{SCOPES_PATH.name} has {sorted(overlap)} in both tiers"


def test_docs_tiers_are_wellformed() -> None:
    """Every tier parsed from the scopes page is usable, non-empty input."""
    for tier in TIERS:
        _assert_wellformed(DOCS_PATH.name, tier, docs_tier(tier))
    overlap = set(docs_tier("canonical")) & set(docs_tier("deprecated"))
    assert not overlap, f"{DOCS_PATH.name} has {sorted(overlap)} in both tiers"


def test_docs_and_json_declare_the_same_vocabulary() -> None:
    """The page and the source agree, per tier, in both directions."""
    tiers = json_tiers()
    for tier in TIERS:
        in_source = set(tiers[tier])
        in_docs = set(docs_tier(tier))
        assert in_source, f"tier {tier!r} parsed empty from the source — comparison is vacuous"
        assert in_docs, f"tier {tier!r} parsed empty from the docs — comparison is vacuous"
        assert in_source == in_docs, (
            f"tier {tier!r} has drifted: "
            f"in {SCOPES_PATH.name} only {sorted(in_source - in_docs)}, "
            f"in {DOCS_PATH.name} only {sorted(in_docs - in_source)}"
        )


def test_config_contains_the_pinned_json_wiring_text() -> None:
    """Both wiring lines are somewhere in ``.commitlintrc.js``.

    Presence, and nothing more.  It catches the realistic accidental edit — someone
    replacing the wiring with a literal list of scopes — and it does not attempt to
    show that no *second*, contradicting line exists: proving absence over text needs
    a parser, and this is a tripwire.  Absence is the workflow step's job.
    """
    config = _normalised_config()
    assert REQUIRE_LINE in config, (
        f"{CONFIG_PATH.name} must load the vocabulary with this line: {REQUIRE_LINE}"
    )
    assert RULE_LINE in config, (
        f"{CONFIG_PATH.name} must build its scope-enum with this line: {RULE_LINE}"
    )


def test_every_deprecated_scope_description_mentions_canonical_status() -> None:
    """The word "canonical" occurs in every deprecated entry's description.

    A keyword check, not a review of the guidance: "canonical is ``infra``", "no
    canonical of its own" and "nothing canonical here" all satisfy it equally.  It
    exists so that a deprecated spelling cannot be added with a description that
    ignores the question; whether the answer is useful is for a reader to judge.
    """
    described = docs_tier_descriptions("deprecated")
    for scope in docs_tier("deprecated"):
        description = described.get(scope, "")
        assert "canonical" in description.lower(), (
            f"deprecated scope {scope!r} says nothing about its canonical status: {description!r}"
        )

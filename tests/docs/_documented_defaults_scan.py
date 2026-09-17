"""Scanner that derives documented-default claims from Markdown, not a hand-kept list.

``test_docs_config_defaults.py`` used to carry a hand-picked list of five
``(config class, field)`` pairs and check each against the field it names in
the docs. That list only ever grew by someone remembering to add a row after
an incident — the five entries were exactly the fields involved in the
incidents that prompted the check. This module inverts the direction: it
reads the documentation and finds every place it states a default for one of
the fields on ``SearchConfig``, ``DreamingGates``, ``HygienePolicyConfig`` or
``TTLConfig``, then resolves each claim against the shipped dataclass. The
set of fields checked is a function of what the docs say, not of what a
maintainer remembered to list.

Two recognised forms
=====================

The documentation states a default in two shapes this scanner recognises:

1. **A Markdown table with a ``Default`` column.** The row's key cell (the
   column headed ``Key`` / ``Field`` / ``Gate`` / ``Option`` / ``Parameter``)
   names the field; the ``Default`` cell is the claim.
2. **Inline prose**, in one clause (see ``_CLAUSE_SPLIT_RE``): a backtick
   field name (bare, ``ClassName.field``, or ``section.field``) near the
   literal word "default"/"defaults", paired with a backtick value token —
   either a separate token (``` `field` ... (default `value`) ```) or a
   single ``field = value`` token.

A third form the design brief for this scanner calls out — inline
``field = value`` — is handled as a sub-case of (2): a backtick token that
itself contains ``=`` is parsed as its own self-contained claim.

Recognition is loud, not clever
================================

The rule for "does this field name mention count as a claim" is deliberately
simple: word-boundary equality against a field name harvested from
``dataclasses.fields()`` on one of the four target classes (never a
hand-typed list — see ``derive_field_owner``). A handful of field names
(currently just ``enabled``) are *also* used as a field name on some other,
out-of-scope ``engrava.config`` dataclass (``DreamingConfig``,
``EdgeCreationConfig``, ``JournalConfig``, ...). A bare mention of one of
those ambiguous names is only accepted as a claim about a target class when
it is either class-qualified (``HygienePolicyConfig.enabled``) or sits under
a Markdown heading this scanner can resolve to that class via
``derive_section_aliases`` (itself derived by walking ``EngravaConfig``'s
own field types, never hand-typed either). Outside a resolvable scope, an
ambiguous bare name is not silently matched to the wrong class — it is
recorded in ``ScanResult.ambiguous_skips`` and not turned into a claim.

A class-qualified mention (``ClassName.field``) whose field does not exist
on that class is *never* skipped — it is recorded in
``ScanResult.nonexistent`` and the caller is expected to fail the build on
it. Documentation describing a removed or renamed option is a real defect
class other layers in this test suite watch for elsewhere; this scanner
catches its own door into that same class.

When a clause names N known fields and the scanner can find exactly N
plausible value tokens, it pairs them positionally in the order they appear
(covering constructs like "``min_cluster_size`` / ``max_cluster_size``
(defaults `3` / `200`)"). A count that does not line up -- 0 values or a
mismatched count -- is **not guessed**: it is recorded in
``ScanResult.unparseable_clauses`` rather than silently dropped or wrongly
paired. A pair whose value is "none"/"null" is a separate, narrower case
(``ScanResult.null_skips``): almost every occurrence of that shape describes
a *per-call* parameter's default (e.g. ``reflection_boost=None -> uses
config``), not the dataclass field's own default, so it is excluded from the
pairing rather than compared -- but counted, never silently dropped.

Resolution is separate from recognition
========================================

Once a claim exists (a field, a class, and a raw value token), resolving it
to a Python value is a second, independent step (``_resolve_literal`` /
``_compare``). A raw value that does not parse as a Python literal (and is
not a bare identifier, e.g. an enum-like ``"lpa"`` written unquoted) or whose
field's shipped value is a ``dict`` (relative signal weights are not a
single literal — comparing them would require guessing which sub-mention of
"0.30" pairs with which key) is recorded in ``ScanResult.unresolved`` rather
than compared. Every field in scope is still expected to show up in at least
one of ``resolved`` or ``unresolved`` — see
``test_every_target_field_is_reached_or_explained`` in the test module.
"""

from __future__ import annotations

import ast
import re
import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_KEY_HEADERS = frozenset({"key", "field", "gate", "option", "parameter"})
_KV_RE = re.compile(r"^([A-Za-z_][\w.]*)\s*=\s*(.+)$")
# A clause boundary is a sentence-ending period (not a decimal point -- the
# lookbehind requires a letter, closing paren, or backtick, never a digit)
# or a semicolon. Long documentation sentences routinely chain several
# independent default-bearing clauses with semicolons.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[a-zA-Z)`])\.\s+(?=[A-Z0-9\"`(])|;\s*")
_LIST_MARKER_RE = re.compile(r"^\s*([-*+]|\d+\.)\s")
_DEFAULT_WORD_RE = re.compile(r"\bdefaults?\b", re.IGNORECASE)
_PIPE_PLACEHOLDER = "\x00"
_NULLISH = frozenset({"none", "null"})


@dataclass(frozen=True)
class Claim:
    """One place the documentation states a default for a target-class field."""

    doc: str
    line: int
    field: str
    cls: type
    raw_value: str
    form: str
    context: str


@dataclass(frozen=True)
class ResolvedClaim:
    """A :class:`Claim` whose raw value was parsed and compared to the shipped default."""

    claim: Claim
    shipped: object
    doc_value: object
    matches: bool


@dataclass(frozen=True)
class UnresolvedClaim:
    """A :class:`Claim` whose raw value could not be compared to the shipped default."""

    claim: Claim
    reason: str


@dataclass(frozen=True)
class UnparseableClause:
    """A default-bearing clause that named field(s) but could not be paired with value(s)."""

    doc: str
    line: int
    fields: tuple[str, ...]
    value_count: int
    context: str


@dataclass(frozen=True)
class AmbiguousSkip:
    """A bare mention of a cross-class-ambiguous field name outside a resolvable scope."""

    doc: str
    line: int
    field: str
    context: str


@dataclass(frozen=True)
class NonexistentField:
    """A ``ClassName.field`` mention naming a field the class does not have."""

    doc: str
    line: int
    cls_name: str
    field_name: str
    context: str


@dataclass(frozen=True)
class NullValueSkip:
    """A field paired positionally with a ``none``/``null`` value token in prose.

    Almost every occurrence of this shape in the documentation describes a
    *per-call* parameter's default ("``reflection_boost=None`` -> uses
    config"), not the dataclass field's own default -- a field that
    genuinely defaults to ``None`` (e.g. ``TTLConfig.default_ttl_seconds``)
    is stated in a table's ``Default`` column instead, which is unambiguous
    and not subject to this exclusion. Recorded rather than silently
    dropped or wrongly compared.
    """

    doc: str
    line: int
    field: str
    context: str


@dataclass(frozen=True)
class ScanResult:
    """Everything the scanner found across the documentation tree."""

    resolved: tuple[ResolvedClaim, ...]
    unresolved: tuple[UnresolvedClaim, ...]
    unparseable_clauses: tuple[UnparseableClause, ...]
    ambiguous_skips: tuple[AmbiguousSkip, ...]
    nonexistent: tuple[NonexistentField, ...]
    null_skips: tuple[NullValueSkip, ...]


def derive_field_owner(target_classes: dict[str, type]) -> dict[str, type]:
    """Map every field name on a target class to the class that owns it.

    Read from ``dataclasses.fields()`` at import time -- there is no
    hand-typed field list anywhere in this module.
    """
    owner: dict[str, type] = {}
    for cls in target_classes.values():
        for f in fields(cls):
            owner[f.name] = cls
    return owner


def derive_ambiguous_names(
    target_classes: dict[str, type], all_dataclasses: list[type]
) -> set[str]:
    """Return field names owned by a target class that some *other* dataclass also uses.

    A bare mention of one of these names is not, on its own, evidence about
    which class it describes.
    """
    owned = set(derive_field_owner(target_classes))
    ambiguous: set[str] = set()
    for cls in all_dataclasses:
        if cls in target_classes.values():
            continue
        for f in fields(cls):
            if f.name in owned:
                ambiguous.add(f.name)
    return ambiguous


def _unwrap_optional(tp: object) -> object:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def derive_section_aliases(root_cls: type, target_classes: dict[str, type]) -> dict[str, type]:
    """Derive ``"section.subsection"`` heading names for each target class.

    Walks ``root_cls``'s own field *types* (e.g. ``EngravaConfig.search:
    SearchConfig`` and ``EngravaConfig.dreaming: DreamingConfig`` ->
    ``DreamingConfig.gates: DreamingGates``) so a heading like
    ``### `hygiene_policy``` or ``#### `dreaming.gates``` resolves to the
    class it actually configures -- derived from the same config module the
    docs describe, not a second hand-typed mapping of section name to class.
    """
    aliases: dict[str, type] = {}
    seen: set[type] = set()
    target_set = set(target_classes.values())

    def walk(cls: type, prefix: str) -> None:
        if cls in seen:
            return
        seen.add(cls)
        hints = typing.get_type_hints(cls)
        for f in fields(cls):
            tp = _unwrap_optional(hints.get(f.name, f.type))
            if isinstance(tp, type) and is_dataclass(tp):
                path = f"{prefix}{f.name}" if prefix else f.name
                if tp in target_set:
                    aliases[path] = tp
                    aliases.setdefault(f.name, tp)
                walk(tp, path + ".")

    walk(root_cls, "")
    return aliases


def _split_row(row: str) -> list[str]:
    """Split a Markdown table row on unescaped ``|``."""
    protected = row.replace("\\|", _PIPE_PLACEHOLDER)
    cells = [c.strip() for c in protected.strip().strip("|").split("|")]
    return [c.replace(_PIPE_PLACEHOLDER, "\\|") for c in cells]


def _is_table_header(lines: list[str], i: int, n: int) -> bool:
    """Whether line ``i`` opens a Markdown pipe table (a separator row follows it)."""
    return (
        lines[i].count("|") >= 2
        and i + 1 < n
        and _TABLE_SEPARATOR_RE.match(lines[i + 1]) is not None
        and "-" in lines[i + 1]
    )


def _table_columns(header_line: str) -> tuple[int | None, int | None]:
    """Return ``(key column index, Default column index)`` for a table header row."""
    key_idx = default_idx = None
    for idx, h in enumerate(_split_row(header_line)):
        hl = h.strip("* ").lower()
        if hl in _KEY_HEADERS and key_idx is None:
            key_idx = idx
        if hl == "default":
            default_idx = idx
    return key_idx, default_idx


def _skip_table_block(lines: list[str], start: int, n: int) -> int:
    """Return the index just past a run of pipe-table rows starting at ``start``."""
    j = start
    while j < n and lines[j].count("|") >= 2 and not _FENCE_RE.match(lines[j]):
        j += 1
    return j


def _clean_heading_text(text: str) -> str:
    text = text.strip()
    m = _BACKTICK_RE.search(text)
    return m.group(1).strip() if m else text.strip("* ")


def _resolve_scope(heading_text: str, section_aliases: dict[str, type]) -> type | None:
    key = _clean_heading_text(heading_text)
    if key in section_aliases:
        return section_aliases[key]
    return section_aliases.get(key.split(".")[-1])


def _bare_field(token: str, field_owner: dict[str, type]) -> str | None:
    seg = token.strip().strip("*").split(".")[-1]
    return seg if seg in field_owner else None


def _qualified_class(token: str, target_classes: dict[str, type]) -> tuple[str, str] | None:
    parts = token.strip().strip("*").split(".")
    if len(parts) >= 2 and parts[0] in target_classes:
        return parts[0], parts[1]
    return None


def _mentions_a_field(
    token: str, target_classes: dict[str, type], field_owner: dict[str, type]
) -> bool:
    """Whether a backtick token names a target-class field at all (qualified or bare).

    Does not judge whether the field exists on the named class or is in
    scope here -- only whether the token is a field mention rather than a
    candidate value token. See ``_Scanner._classify_field_token`` for the
    existence/ambiguity handling.
    """
    if _qualified_class(token, target_classes) is not None:
        return True
    return _bare_field(token, field_owner) is not None


def _looks_like_value(tok: str, target_classes: dict[str, type]) -> bool:
    """Whether a non-field backtick token is plausibly a literal value.

    Excludes bare class-name mentions (``SearchConfig``), anything
    containing parentheses (method calls / signatures), and multi-word text
    that is not a bracketed list/tuple literal.
    """
    cleaned = tok.strip()
    if not cleaned or cleaned in target_classes:
        return False
    if "(" in cleaned or ")" in cleaned:
        return False
    return not (" " in cleaned and not (cleaned[0] in "[{" and cleaned[-1] in "]}"))


class _ParagraphBuffer:
    """Accumulates consecutive prose lines into one paragraph for clause splitting.

    The caller decides paragraph boundaries (blank line, heading, table,
    fence, or a fresh list item -- see ``_scan_prose``); this class only
    joins whatever lines it is given between two calls to :meth:`take`.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._start: int | None = None

    def add(self, line: str, line_idx: int) -> None:
        if not self._lines:
            self._start = line_idx + 1
        self._lines.append(line.strip())

    def take(self) -> tuple[int, str] | None:
        """Return and clear the buffered ``(start line, joined text)``, or ``None`` if empty."""
        if not self._lines or self._start is None:
            self._lines = []
            self._start = None
            return None
        result = (self._start, " ".join(self._lines))
        self._lines = []
        self._start = None
        return result


class _Scanner:
    """Stateful per-run scan; not part of the public API."""

    def __init__(
        self,
        field_owner: dict[str, type],
        target_classes: dict[str, type],
        ambiguous_names: set[str],
        section_aliases: dict[str, type],
    ) -> None:
        self.field_owner = field_owner
        self.target_classes = target_classes
        self.ambiguous_names = ambiguous_names
        self.section_aliases = section_aliases
        self.table_claims: list[Claim] = []
        self.prose_claims: list[Claim] = []
        self.unparseable: list[UnparseableClause] = []
        self.ambiguous_skips: list[AmbiguousSkip] = []
        self.nonexistent: list[NonexistentField] = []
        self.null_skips: list[NullValueSkip] = []
        self._table_claimed_lines: dict[str, set[int]] = {}

    def _owned_fields(self, cls_name: str) -> set[str]:
        return {f.name for f in fields(self.target_classes[cls_name])}

    def scan_file(self, path: Path, repo_root: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(repo_root).as_posix()
        self._scan_tables(rel, lines)
        self._scan_prose(rel, lines)

    def _scan_tables(self, rel: str, lines: list[str]) -> None:
        n = len(lines)
        in_fence = False
        scope: type | None = None
        i = 0
        while i < n:
            line = lines[i]
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                i += 1
                continue
            if in_fence:
                i += 1
                continue
            hm = _HEADING_RE.match(line)
            if hm:
                scope = _resolve_scope(hm.group(2), self.section_aliases)
                i += 1
                continue
            if _is_table_header(lines, i, n):
                i = self._scan_one_table(rel, lines, i, scope)
                continue
            i += 1

    def _scan_one_table(
        self, rel: str, lines: list[str], header_idx: int, scope: type | None
    ) -> int:
        n = len(lines)
        key_idx, default_idx = _table_columns(lines[header_idx])
        if key_idx is None or default_idx is None:
            return _skip_table_block(lines, header_idx + 2, n)
        j = header_idx + 2
        while j < n and lines[j].count("|") >= 2 and not _FENCE_RE.match(lines[j]):
            cells = _split_row(lines[j])
            if len(cells) > max(key_idx, default_idx):
                key_cell = cells[key_idx]
                toks = _BACKTICK_RE.findall(key_cell) or [key_cell]
                for tok in toks:
                    self._handle_key_token(rel, j, tok, cells[default_idx], lines[j].strip(), scope)
            j += 1
        return j

    def _handle_key_token(
        self, rel: str, line_idx: int, tok: str, default_cell: str, context: str, scope: type | None
    ) -> None:
        qual = _qualified_class(tok, self.target_classes)
        if qual:
            cname, fname = qual
            if fname not in self._owned_fields(cname):
                self.nonexistent.append(NonexistentField(rel, line_idx + 1, cname, fname, context))
                return
            self._claim_table(
                rel, line_idx, fname, self.target_classes[cname], default_cell, context
            )
            return
        fld = _bare_field(tok, self.field_owner)
        if fld is None:
            return
        if fld in self.ambiguous_names and self.field_owner[fld] is not scope:
            self.ambiguous_skips.append(AmbiguousSkip(rel, line_idx + 1, fld, context))
            return
        self._claim_table(rel, line_idx, fld, self.field_owner[fld], default_cell, context)

    def _claim_table(
        self, rel: str, line_idx: int, field: str, cls: type, raw_value: str, context: str
    ) -> None:
        self.table_claims.append(Claim(rel, line_idx + 1, field, cls, raw_value, "table", context))
        self._table_claimed_lines.setdefault(rel, set()).add(line_idx)

    def _scan_prose(self, rel: str, lines: list[str]) -> None:
        n = len(lines)
        already = self._table_claimed_lines.get(rel, set())
        buf = _ParagraphBuffer()
        in_fence = False
        scope: type | None = None
        i = 0
        while i < n:
            line = lines[i]
            if _FENCE_RE.match(line):
                self._flush_paragraph(rel, buf, scope)
                in_fence = not in_fence
                i += 1
                continue
            if in_fence:
                i += 1
                continue
            hm = _HEADING_RE.match(line)
            if hm:
                self._flush_paragraph(rel, buf, scope)
                scope = _resolve_scope(hm.group(2), self.section_aliases)
                i += 1
                continue
            if not line.strip() or i in already:
                self._flush_paragraph(rel, buf, scope)
                i += 1
                continue
            if _is_table_header(lines, i, n):
                self._flush_paragraph(rel, buf, scope)
                i = _skip_table_block(lines, i + 2, n)
                continue
            if _LIST_MARKER_RE.match(line):
                self._flush_paragraph(rel, buf, scope)
            buf.add(line, i)
            i += 1
        self._flush_paragraph(rel, buf, scope)

    def _flush_paragraph(self, rel: str, buf: _ParagraphBuffer, scope: type | None) -> None:
        text = buf.take()
        if text is not None:
            self._process_paragraph(rel, text[0], text[1], scope)

    def _process_paragraph(self, rel: str, start_line: int, text: str, scope: type | None) -> None:
        for clause in _CLAUSE_SPLIT_RE.split(text):
            self._process_clause(rel, start_line, clause, scope)

    def _process_clause(self, rel: str, start_line: int, text: str, scope: type | None) -> None:
        if not _DEFAULT_WORD_RE.search(text):
            return
        toks = _BACKTICK_RE.findall(text)
        if not toks:
            return
        field_toks, value_toks = self._collect_clause_tokens(rel, start_line, text, scope, toks)
        self._pair_clause_fields_and_values(rel, start_line, text, field_toks, value_toks)

    def _collect_clause_tokens(
        self, rel: str, start_line: int, text: str, scope: type | None, toks: list[str]
    ) -> tuple[list[tuple[str, type]], list[str]]:
        """Sort one clause's backtick tokens into field mentions and candidate values.

        An inline ``field = value`` token is resolved and emitted directly
        (it is self-contained, not part of the positional pairing below).
        """
        field_toks: list[tuple[str, type]] = []
        value_toks: list[str] = []
        for tok in toks:
            kv = _KV_RE.match(tok.strip())
            if kv:
                self._handle_inline_kv(rel, start_line, kv.group(1), kv.group(2), text, scope)
                continue
            if _mentions_a_field(tok, self.target_classes, self.field_owner):
                field = self._classify_field_token(rel, start_line, tok, text, scope)
                if field is not None:
                    field_toks.append(field)
                continue
            if _looks_like_value(tok, self.target_classes):
                value_toks.append(tok)
        return field_toks, value_toks

    def _pair_clause_fields_and_values(
        self,
        rel: str,
        start_line: int,
        text: str,
        field_toks: list[tuple[str, type]],
        value_toks: list[str],
    ) -> None:
        """Pair fields to values positionally when the counts line up; else record the mismatch."""
        if not field_toks:
            return
        if len(field_toks) != len(value_toks):
            self.unparseable.append(
                UnparseableClause(
                    rel, start_line, tuple(f for f, _ in field_toks), len(value_toks), text
                )
            )
            return
        for (fname, cls), raw in zip(field_toks, value_toks, strict=True):
            if raw.strip().lower() in _NULLISH:
                self.null_skips.append(NullValueSkip(rel, start_line, fname, text))
            else:
                self.prose_claims.append(Claim(rel, start_line, fname, cls, raw, "prose", text))

    def _classify_field_token(
        self, rel: str, start_line: int, tok: str, context: str, scope: type | None
    ) -> tuple[str, type] | None:
        """Resolve a token already known to mention a field (see ``_mentions_a_field``).

        Records a nonexistent-field or ambiguous-scope finding as a side
        effect and returns ``None`` in both cases -- the caller must not
        also treat the token as a candidate value.
        """
        qual = _qualified_class(tok, self.target_classes)
        if qual:
            cname, fname = qual
            if fname not in self._owned_fields(cname):
                self.nonexistent.append(NonexistentField(rel, start_line, cname, fname, context))
                return None
            return fname, self.target_classes[cname]
        fld = _bare_field(tok, self.field_owner)
        assert fld is not None  # guaranteed by _mentions_a_field
        if fld in self.ambiguous_names and self.field_owner[fld] is not scope:
            self.ambiguous_skips.append(AmbiguousSkip(rel, start_line, fld, context))
            return None
        return fld, self.field_owner[fld]

    def _handle_inline_kv(
        self, rel: str, start_line: int, lhs: str, rhs: str, context: str, scope: type | None
    ) -> None:
        """Resolve one self-contained ``field = value`` backtick token and record its claim."""
        qual = _qualified_class(lhs, self.target_classes)
        if qual:
            cname, fname = qual
            if fname not in self._owned_fields(cname):
                self.nonexistent.append(NonexistentField(rel, start_line, cname, fname, context))
            else:
                self._add_prose_claim(
                    rel, start_line, fname, self.target_classes[cname], rhs, context
                )
            return
        seg = lhs.rsplit(".", maxsplit=1)[-1]
        if seg not in self.field_owner:
            return
        if seg in self.ambiguous_names and self.field_owner[seg] is not scope:
            self.ambiguous_skips.append(AmbiguousSkip(rel, start_line, seg, context))
            return
        self._add_prose_claim(rel, start_line, seg, self.field_owner[seg], rhs, context)

    def _add_prose_claim(
        self, rel: str, start_line: int, field: str, cls: type, raw_value: str, context: str
    ) -> None:
        self.prose_claims.append(
            Claim(rel, start_line, field, cls, raw_value, "inline-kv", context)
        )


def _resolve_literal(raw: str) -> tuple[bool, object, str | None]:
    """Parse a raw doc token into a Python value, or report why it cannot be parsed."""
    s = raw.strip()
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()
    low = s.lower()
    if low in ("true", "false", *_NULLISH):
        return True, {"true": True, "false": False, "none": None, "null": None}[low], None
    try:
        return True, ast.literal_eval(s), None
    except (ValueError, SyntaxError):
        pass
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        return True, s, None
    return False, None, f"doc value {raw!r} does not parse as a Python literal or a bare identifier"


def _compare_sequence(
    shipped: tuple[object, ...] | list[object], parsed: object
) -> tuple[bool, str | None]:
    if isinstance(parsed, (tuple, list)):
        return list(shipped) == list(parsed), None
    if len(shipped) == 1:
        return shipped[0] == parsed, None
    return False, "doc value is a bare scalar but the shipped sequence has more than one element"


def _compare_scalar(shipped: object, parsed: object) -> tuple[bool, str | None]:
    if isinstance(shipped, bool):
        if isinstance(parsed, bool):
            return shipped == parsed, None
        return False, "shipped value is bool but the doc value is not"
    if (
        isinstance(shipped, (int, float))
        and isinstance(parsed, (int, float))
        and not isinstance(parsed, bool)
    ):
        return shipped == parsed, None
    if shipped is None:
        return parsed is None, None
    return str(shipped) == str(parsed), None


def _compare(shipped: object, parsed: object) -> tuple[bool | None, str | None]:
    """Compare a shipped default to a parsed doc value; ``None`` means "cannot compare"."""
    if isinstance(shipped, dict):
        return None, "field's shipped default is a dict; not a single literal to compare"
    if isinstance(shipped, (tuple, list)):
        return _compare_sequence(shipped, parsed)
    return _compare_scalar(shipped, parsed)


def _resolve_claim(claim: Claim) -> ResolvedClaim | UnresolvedClaim:
    shipped = getattr(claim.cls(), claim.field)
    ok, parsed, reason = _resolve_literal(claim.raw_value)
    if not ok:
        return UnresolvedClaim(claim, reason or "unparseable")
    matches, cmp_reason = _compare(shipped, parsed)
    if matches is None:
        return UnresolvedClaim(claim, cmp_reason or "not comparable")
    return ResolvedClaim(claim, shipped, parsed, matches)


def scan_documented_defaults(
    doc_files: list[Path],
    repo_root: Path,
    field_owner: dict[str, type],
    target_classes: dict[str, type],
    ambiguous_names: set[str],
    section_aliases: dict[str, type],
) -> ScanResult:
    """Scan ``doc_files`` for documented defaults and resolve each against the shipped code.

    Args:
        doc_files: Markdown files to scan, in document order.
        repo_root: Root every file in ``doc_files`` is reported relative to.
        field_owner: ``field name -> owning class``, from :func:`derive_field_owner`.
        target_classes: ``"ClassName" -> class`` for the in-scope config classes.
        ambiguous_names: Field names also used by an out-of-scope dataclass,
            from :func:`derive_ambiguous_names`.
        section_aliases: Heading text -> class, from :func:`derive_section_aliases`.

    Returns:
        A :class:`ScanResult` covering every claim found, resolved or not.

    """
    scanner = _Scanner(field_owner, target_classes, ambiguous_names, section_aliases)
    for path in doc_files:
        scanner.scan_file(path, repo_root)

    resolved: list[ResolvedClaim] = []
    unresolved: list[UnresolvedClaim] = []
    for claim in (*scanner.table_claims, *scanner.prose_claims):
        outcome = _resolve_claim(claim)
        if isinstance(outcome, ResolvedClaim):
            resolved.append(outcome)
        else:
            unresolved.append(outcome)

    return ScanResult(
        resolved=tuple(resolved),
        unresolved=tuple(unresolved),
        unparseable_clauses=tuple(scanner.unparseable),
        ambiguous_skips=tuple(scanner.ambiguous_skips),
        nonexistent=tuple(scanner.nonexistent),
        null_skips=tuple(scanner.null_skips),
    )

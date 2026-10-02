"""Census guard: exactly one construction site claims a real ``EngravaMetrics`` measurement.

``EngravaMetrics.measured`` defaults to ``False`` precisely so a forgotten
``False`` costs nothing — any construction site this test has never heard of
already reports "not measured" by construction. The risk that default cannot
cover is the opposite one: a ``measured=True`` copy-pasted into a path that
does not actually measure anything, which would silently bless a fabricated
snapshot.

This is deliberately **not** a prohibition on a second measuring path. A
second, legitimate ``measured=True`` site may exist later (a different
backend's own aggregate-query path, for instance); a bare "only one call site
allowed" rule would just get deleted by whoever needs the second one. Pinning
the *count* instead means a genuine new measuring path costs one deliberate
edit to the number below plus review, while a careless copy of ``True`` into a
non-measuring path fails this test immediately.

**What the scan actually matches.** This is a syntactic scan over *spellings*,
not a resolved-constructor analysis — it does not know what a call actually
returns at runtime. It matches a bare name that is literally ``EngravaMetrics``
or a local alias introduced by ``from ... import EngravaMetrics as alias``
anywhere in the file, and it matches **any** qualified
``<anything>.EngravaMetrics(...)`` regardless of what ``<anything>`` resolves
to. That means `Unrelated().EngravaMetrics(measured=True)` is counted even if
that method returns ``None``, and a local variable that
happens to be named the same as an aliased import (shadowing it) is still
counted as a call to the alias. **Counting too much is the safe direction**
for this guard — a false positive costs a moment of review, a false negative
would let a fabricated measurement through — so this is deliberate, not a bug
to fix.

Within a matched call, a literal ``True`` is recognised as ``measured=True``
either as the ``measured`` keyword or positionally at ``measured``'s current
slot in the dataclass's own field order (resolved from
:data:`dataclasses.fields`, not a hardcoded index, so this survives a future
field reorder — though `EngravaMetrics`'s own docstring asks people not to
reorder it).

**A starred positional argument (``*args``) makes a call unclassifiable, and
that fails the census loudly rather than passing it — but only when no
explicit ``measured=`` keyword is also present.** A keyword pins the value
outright: a real call combining `*args` with `measured=...` cannot also have
a positional argument land on that same slot without Python raising
`TypeError` first, so the keyword is authoritative whenever it is there
(`EngravaMetrics(*args, measured=True)` is correctly read as ``True``). It is
only once there is **no** `measured=` keyword that a starred argument makes
the position genuinely unknowable — `EngravaMetrics(*some_iterable, True)`
might bind that `True` to `measured` or to nothing at all, depending on
`some_iterable`'s length at runtime, which this static scan cannot know.
Treating that as "not `measured=True`" would be exactly the failure mode this
census exists to prevent, so instead the census refuses to classify it and
fails the whole test, naming the call site, so a human has to look at it.

**What is deliberately out of reach and silently not counted** (as opposed to
the starred case above, which is loud): a non-literal value
(``measured=some_flag``), an unpacked keyword dict (``**{"measured": True}``),
and a call built through a genuinely dynamic constructor (e.g.
``getattr(module, "EngravaMetrics")`` or a name bound by a plain ``=``
assignment rather than an import). Those need a human reviewer; this census
does not see them at all, in either direction.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from engrava.domain.models.metrics import EngravaMetrics

_SRC_DIR = Path(__file__).parent.parent / "src"

# The number of `EngravaMetrics(...)` construction sites, across all of `src/`,
# that pass `measured=True`. Bump this deliberately — with review — if a
# genuine second measuring path is added; do not bump it to silence a failure
# without checking that the new site actually queries the store.
_EXPECTED_MEASURED_TRUE_SITES = 1

# Resolved from the dataclass's own field order rather than hardcoded, so a
# future reorder of `EngravaMetrics` (which its own docstring asks people not
# to do casually — `measured` was appended on purpose) cannot silently make
# this census stop recognising a positional `measured=True`.
_MEASURED_FIELD_INDEX = [f.name for f in dataclasses.fields(EngravaMetrics)].index("measured")


def _real_name_bound_by_import(node: ast.ImportFrom) -> dict[str, str]:
    """Return ``{local_name: "EngravaMetrics"}`` for each matching import in *node*.

    Covers both ``from ... import EngravaMetrics`` (local name is the real
    name) and ``from ... import EngravaMetrics as alias`` (local name is the
    alias) — without this, a call through an aliased import escapes the scan
    entirely, since it is spelled with a name that is not literally
    ``EngravaMetrics`` anywhere in the file.
    """
    return {
        (alias.asname or alias.name): "EngravaMetrics"
        for alias in node.names
        if alias.name == "EngravaMetrics"
    }


def _resolve_local_metrics_names(tree: ast.Module) -> set[str]:
    """Return every bare name in *tree* that this scan treats as ``EngravaMetrics``.

    This is a spelling match, not constructor resolution: it does not know
    what any name actually refers to at runtime. Always includes the literal
    name itself, plus any local alias introduced by a
    ``from ... import EngravaMetrics as alias`` anywhere in the module
    (regardless of which module it was imported from — the class is
    re-exported from more than one place, and this scan does not need to
    care which one a given file used).
    """
    names = {"EngravaMetrics"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(_real_name_bound_by_import(node))
    return names


def _is_engrava_metrics_call(node: ast.Call, *, local_names: set[str]) -> bool:
    """Return whether *node*'s callee is spelled like ``EngravaMetrics``.

    This does not resolve what the callee actually is at runtime — it matches
    a bare name found in *local_names* (covers a direct ``EngravaMetrics(...)``
    and an aliased import) and **any** qualified ``<anything>.EngravaMetrics(...)``
    regardless of what ``<anything>`` is (covers a module alias, since the
    attribute name itself is never renamed by ``import ... as``, at the cost of
    also matching an unrelated object that merely has a same-named attribute).
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in local_names
    if isinstance(func, ast.Attribute):
        return func.attr == "EngravaMetrics"
    return False


class UnanalysablePositionalArgumentError(Exception):
    """A construction call cannot be classified for ``measured`` at all.

    Raised when no explicit ``measured=`` keyword is present *and* any
    positional argument is a starred unpack (``*args``-style): once anything
    is unpacked, the AST argument index is no longer the runtime parameter
    index, so this static scan cannot know which parameter a later positional
    argument — including a literal ``True`` — actually binds to. Silently
    deciding such a call does not claim ``measured=True`` would be exactly the
    failure this census exists to prevent, so it refuses to classify the call
    instead, and the caller must fail loudly on this.
    """


def _passes_measured_true(node: ast.Call) -> bool:
    """Return whether *node* sets ``measured=True`` by keyword or by position.

    See the module docstring for exactly what this does and does not see.

    Raises:
        UnanalysablePositionalArgumentError: If there is no explicit
            ``measured=`` keyword and a positional argument is a starred
            unpack, making the positional mapping unknowable.

    """
    for kw in node.keywords:
        if kw.arg == "measured":
            # An explicit `measured=` keyword pins the value regardless of any
            # positional arguments: a real call mixing the two would be a
            # `TypeError` (multiple values for the same parameter) rather
            # than a silent value, so a call that runs at all cannot also
            # have a positional argument reach this slot. That makes the
            # keyword authoritative even when `node.args` also contains a
            # starred unpack -- there is no ambiguity left to fail on.
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        message = (
            f"line {node.lineno}: a starred positional argument (`*...`) makes it "
            f"impossible to know which parameter each positional argument binds to, "
            f"including whether any of them is `measured`. Rewrite this call with "
            f"explicit keyword arguments so it can be classified."
        )
        raise UnanalysablePositionalArgumentError(message)
    if len(node.args) > _MEASURED_FIELD_INDEX:
        positional_value = node.args[_MEASURED_FIELD_INDEX]
        if isinstance(positional_value, ast.Constant) and positional_value.value is True:
            return True
    return False


def _find_engrava_metrics_calls(source: str) -> list[ast.Call]:
    """Return every call node in *source* spelled like ``EngravaMetrics(...)``."""
    tree = ast.parse(source)
    local_names = _resolve_local_metrics_names(tree)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_engrava_metrics_call(node, local_names=local_names)
    ]


def _all_engrava_metrics_calls() -> list[tuple[Path, ast.Call]]:
    """Return every ``EngravaMetrics(...)`` call under ``src/``, paired with its file."""
    found: list[tuple[Path, ast.Call]] = []
    for path in sorted(_SRC_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        found.extend((path, call) for call in _find_engrava_metrics_calls(source))
    return found


class TestMeasuredTrueCensus:
    """Pin the number of construction sites that claim a real measurement."""

    def test_src_directory_is_discovered(self) -> None:
        """Precondition: pins that discovery targets a real, existing directory.

        This is not what stands between a broken path and a vacuous pass —
        ``test_exactly_one_construction_site_claims_measured``'s own
        ``assert calls`` already catches that case (a missing directory
        yields no ``.py`` files, so ``calls`` comes back empty and that
        assertion fails with its own message). This test simply states the
        precondition explicitly rather than leaving it implicit.
        """
        assert _SRC_DIR.is_dir(), f"src directory not found at {_SRC_DIR}"

    def test_exactly_one_construction_site_claims_measured(self) -> None:
        calls = _all_engrava_metrics_calls()
        assert calls, "found no `EngravaMetrics(...)` construction sites at all under src/"

        measured_true_sites: list[str] = []
        for path, call in calls:
            site = f"{path.relative_to(_SRC_DIR)}:{call.lineno}"
            try:
                claims_measured = _passes_measured_true(call)
            except UnanalysablePositionalArgumentError as exc:
                pytest.fail(f"{site} cannot be classified by the measured=True census: {exc}")
            if claims_measured:
                measured_true_sites.append(site)

        assert len(measured_true_sites) == _EXPECTED_MEASURED_TRUE_SITES, (
            f"expected exactly {_EXPECTED_MEASURED_TRUE_SITES} `EngravaMetrics(...)` "
            f"construction site(s) with `measured=True`, found "
            f"{len(measured_true_sites)}: {measured_true_sites}. A new site needs a "
            f"deliberate bump of _EXPECTED_MEASURED_TRUE_SITES plus review that it "
            f"genuinely measures the store; a site that does not measure anything "
            f"must not pass `measured=True` at all."
        )


class TestCensusDetectors:
    """The detectors the census relies on must actually discriminate.

    Coverage here matches exactly what the module docstring advertises: a
    literal ``True`` by keyword or by resolved position, reached through a
    bare name, a qualified attribute, or an aliased ``from ... import``. Any
    case documented above as out of reach gets its own counter-case test
    showing it stays undetected, so the claimed gap cannot silently close (or
    silently widen) without a test noticing.
    """

    def test_bare_call_is_matched(self) -> None:
        calls = _find_engrava_metrics_calls("EngravaMetrics(snapshot_timestamp=1.0)\n")
        assert len(calls) == 1

    def test_qualified_call_is_matched(self) -> None:
        calls = _find_engrava_metrics_calls("metrics.EngravaMetrics(snapshot_timestamp=1.0)\n")
        assert len(calls) == 1

    def test_unrelated_call_is_not_matched(self) -> None:
        calls = _find_engrava_metrics_calls("ThoughtCounts(total=0)\n")
        assert calls == []

    def test_measured_true_keyword_is_detected(self) -> None:
        (call,) = _find_engrava_metrics_calls("EngravaMetrics(measured=True)\n")
        assert _passes_measured_true(call)

    def test_measured_false_keyword_is_not_a_true_site(self) -> None:
        (call,) = _find_engrava_metrics_calls("EngravaMetrics(measured=False)\n")
        assert not _passes_measured_true(call)

    def test_call_with_no_measured_keyword_is_not_a_true_site(self) -> None:
        (call,) = _find_engrava_metrics_calls("EngravaMetrics(snapshot_timestamp=1.0)\n")
        assert not _passes_measured_true(call)

    def test_a_variable_passed_as_measured_is_not_a_literal_true_site(self) -> None:
        """A non-literal value is deliberately not credited as ``measured=True``.

        The census exists to catch a copy-pasted literal ``True``, so it only
        recognises the literal. A computed value at a construction site would
        need its own review, not a pass granted by this detector.
        """
        (call,) = _find_engrava_metrics_calls("EngravaMetrics(measured=some_flag)\n")
        assert not _passes_measured_true(call)

    def test_positional_true_at_the_measured_slot_is_detected(self) -> None:
        """A positional call claiming measured=True is not invisible to the scan.

        Built from the real field order so this test does not itself hardcode
        the position ``measured`` happens to occupy.
        """
        field_names = [f.name for f in dataclasses.fields(EngravaMetrics)]
        args = ["0" for _ in field_names]
        args[_MEASURED_FIELD_INDEX] = "True"
        source = f"EngravaMetrics({', '.join(args)})\n"

        (call,) = _find_engrava_metrics_calls(source)

        assert _passes_measured_true(call), (
            f"positional `True` at index {_MEASURED_FIELD_INDEX} (the `measured` slot) "
            f"in {source!r} was not detected"
        )

    def test_positional_true_at_a_different_slot_is_not_a_true_site(self) -> None:
        """The counter-case: a `True` positional elsewhere is not `measured`.

        Without this, a detector that just checked "any positional arg is
        `True`" would look identical to one checking the right slot.
        """
        field_names = [f.name for f in dataclasses.fields(EngravaMetrics)]
        assert len(field_names) > 1, "need at least two fields to place True at the wrong one"
        wrong_index = 0 if _MEASURED_FIELD_INDEX != 0 else 1
        args = ["0" for _ in field_names]
        args[wrong_index] = "True"
        source = f"EngravaMetrics({', '.join(args)})\n"

        (call,) = _find_engrava_metrics_calls(source)

        assert not _passes_measured_true(call)

    def test_aliased_from_import_call_is_matched(self) -> None:
        """`from ... import EngravaMetrics as M; M(measured=True)` does not escape the scan."""
        source = "from engrava.domain.models.metrics import EngravaMetrics as M\nM(measured=True)\n"

        calls = _find_engrava_metrics_calls(source)

        assert len(calls) == 1
        assert _passes_measured_true(calls[0])

    def test_unaliased_from_import_call_is_matched(self) -> None:
        """`from ... import EngravaMetrics` followed by a bare call is still matched."""
        source = "from engrava.domain.models.metrics import EngravaMetrics\nEngravaMetrics()\n"

        calls = _find_engrava_metrics_calls(source)

        assert len(calls) == 1

    def test_an_unrelated_aliased_import_does_not_create_a_false_match(self) -> None:
        """Aliasing an unrelated import must not make its calls look like `EngravaMetrics`."""
        source = "from engrava.domain.models.metrics import ThoughtCounts as M\nM(total=1)\n"

        assert _find_engrava_metrics_calls(source) == []

    def test_a_keyword_dict_unpack_is_deliberately_not_a_true_site(self) -> None:
        """Documented residue: `**{"measured": True}` is not recognised.

        This pins the *advertised* gap so the module docstring's claim about
        what is out of reach stays true rather than aspirational.
        """
        (call,) = _find_engrava_metrics_calls('EngravaMetrics(**{"measured": True})\n')
        assert not _passes_measured_true(call)

    def test_a_reassigned_name_is_deliberately_not_matched(self) -> None:
        """Documented residue: a plain-assignment alias (not an import) is not matched.

        Only ``from ... import EngravaMetrics as alias`` is recognised;
        `Other = EngravaMetrics` followed by `Other(...)` is out of reach,
        same as a dynamically obtained constructor.
        """
        source = "Other = EngravaMetrics\nOther(measured=True)\n"

        assert _find_engrava_metrics_calls(source) == []

    def test_a_starred_positional_argument_is_refused_not_silently_passed(self) -> None:
        """A `*args` unpack with no `measured=` keyword must fail loudly, not pass silently.

        This is the counter-case that matters most: without it, a detector
        that just fell through to "no positional match" on a starred call
        would look identical to a correctly-working one, and a real
        `measured=True` hidden behind an unpack would pass the census.
        """
        (call,) = _find_engrava_metrics_calls("EngravaMetrics(*some_iterable, True)\n")

        with pytest.raises(UnanalysablePositionalArgumentError):
            _passes_measured_true(call)

    def test_an_explicit_measured_true_keyword_wins_over_a_starred_argument(self) -> None:
        """`EngravaMetrics(*args, measured=True)` is read as True, not refused.

        A real call combining `*args` with an explicit `measured=` keyword
        cannot also bind a positional argument to that same slot without
        Python raising `TypeError` first, so the keyword is unambiguous even
        though a starred argument is also present.
        """
        (call,) = _find_engrava_metrics_calls("EngravaMetrics(*args, measured=True)\n")

        assert _passes_measured_true(call)

    def test_an_explicit_measured_false_keyword_wins_over_a_starred_argument(self) -> None:
        """`EngravaMetrics(*args, measured=False)` is read as False, not refused.

        The counter-case to the True keyword above: the keyword still settles
        the question on its own, so this must not raise either.
        """
        (call,) = _find_engrava_metrics_calls("EngravaMetrics(*args, measured=False)\n")

        assert not _passes_measured_true(call)

    def test_a_call_matched_only_by_spelling_is_still_counted(self) -> None:
        """The scan matches the spelling `<anything>.EngravaMetrics(...)`, not a resolved class.

        `Unrelated().EngravaMetrics(measured=True)` is counted even though
        nothing here proves `Unrelated().EngravaMetrics` is the real
        constructor -- documented in the module docstring as the deliberate,
        safe-direction over-match this scan makes.
        """
        (call,) = _find_engrava_metrics_calls("Unrelated().EngravaMetrics(measured=True)\n")

        assert _passes_measured_true(call)

    def test_a_parameter_shadowing_an_aliased_import_is_still_counted(self) -> None:
        """A local name shadowing an aliased import is matched by spelling, not by scope.

        After `from ... import EngravaMetrics as M`, any later `M(...)` in the
        file is counted -- including one where `M` is actually a function
        parameter that shadows the import, since this scan does not model
        Python scoping. Documented as the same deliberate over-match as the
        qualified-call case above.
        """
        source = (
            "from engrava.domain.models.metrics import EngravaMetrics as M\n"
            "def f(M):\n"
            "    return M(measured=True)\n"
        )

        (call,) = _find_engrava_metrics_calls(source)

        assert _passes_measured_true(call)

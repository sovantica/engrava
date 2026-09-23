"""A static guard that notices a new, unprotected journal-append call site.

This does not verify that any given append is *correctly* wrapped — that is
what the behavioural atomicity tests are for (``test_journal_append_atomicity.py``,
``test_create_delete_journal_append_atomicity.py``,
``test_hygiene_journal_append_atomicity.py``). It proves a narrower, purely
structural obligation over *every* ``await self._journal.append(...)`` call
node in ``engrava_core.py``: each one is lexically inside a
``_write_readback_savepoint`` block in its own function, or its function is a
registered helper whose *every* call site is itself inside such a block, or
its function is a registered exemption with the exact call-node count on
record. It certifies nothing about safety — a call site can satisfy this
guard and still be wrapped incorrectly (the wrong savepoint left open, a
finalization bug); this only proves the obligation was not silently dropped
as the module grows a new append.

Functions are identified by their **fully qualified** scope path (e.g.
``SqliteEngravaCore._hygiene_gc``), not by bare name: a same-named method on
a different class, or a same-named nested function, is a different identity
to this guard and is never mistaken for a registered helper or exemption.

**Known blind spot.** This is a static, syntactic guard, not a dataflow
analysis: it recognises a direct call by name -- an attribute access with a
matching attribute name (any receiver), or a matching bare name -- and does
not follow a helper reached indirectly (stored in a variable and called
through it, passed as an argument, or looked up with ``getattr``). It proves
an obligation for the call shapes it recognises and certifies nothing about
safety, here as everywhere else in this module.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engrava.infrastructure.sqlite import engrava_core

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

# The wrapper every append site must nest inside, directly or transitively
# through a registered helper. Matches ``self._write_readback_savepoint(...)``
# used as an ``async with`` context expression.
_SAVEPOINT_HELPER_NAME = "_write_readback_savepoint"

# Functions that make the append call themselves but are never wrapped in
# their *own* body -- every one of their own call sites is wrapped instead.
# A helper registered here whose call sites are not *all* wrapped fails the
# guard; a helper that is not actually called anywhere also fails it (a
# vestigial entry hides a real gap just as well as a missing one would).
# Keyed by *fully qualified* name (``<enclosing class>.<method>``) so a
# same-named method on an unrelated class is never mistaken for one of these.
_HELPER_FUNCTIONS = frozenset(
    {
        # Every call site is inside run_hygiene's own
        # _write_readback_savepoint("run_hygiene") block.
        "SqliteEngravaCore._hygiene_archive",
        "SqliteEngravaCore._hygiene_gc",
        # Every call site (create_action, update_action) is inside that
        # caller's own savepoint block around the action's write.
        "SqliteEngravaCore._recompute_action_outcome",
    }
)

# Functions exempt from the wrapping obligation, with the reason and the
# exact number of `self._journal.append(...)` call nodes the function
# contains -- a count mismatch (a second, unrelated append added beside the
# recorded one) fails the guard just as an entirely new, unregistered site
# would. Keyed by fully qualified name, same reasoning as `_HELPER_FUNCTIONS`.
#
# Empty on the real tree: `_insert_derived_row` and `_insert_derived_edge`
# used to be registered here (their appends recovered through a full-
# transaction compensating rollback instead of their own savepoint unit).
# Both are now wrapped in their own `_write_readback_savepoint` block like
# every other journaled insert, so neither needs an exemption any more --
# the guard protects them lexically instead. Kept as a real, empty registry
# (not deleted) so the exemption machinery below still has a home; the
# scratch mutations further down register their own synthetic entries via
# `monkeypatch` to keep exercising that machinery without a real function on
# the exemption table.
_EXEMPTIONS: dict[str, tuple[str, int]] = {}


@dataclass(frozen=True)
class _CallSite:
    """One matched call node: which function it lexically sits in, and where."""

    function: str
    lineno: int
    wrapped: bool


class _FunctionScopedCallFinder(ast.NodeVisitor):
    """Find every call node a predicate matches, and whether each sits inside a savepoint block.

    "Inside a savepoint block" is scoped **per function**: entering a
    ``def`` / ``async def`` pushes a fresh, zero savepoint-depth for that
    function's own body, so a call is only ever considered wrapped by an
    ``async with`` in its *own* enclosing function, never one belonging to an
    outer function it happens to be lexically nested under (not a shape this
    module uses, but the scoping rule the work item's wording implies: "in
    its own function").

    Functions are attributed by their **fully qualified** scope path --
    ``<class>.<method>``, or ``<class>.<method>.<nested function>`` for a
    function nested inside another -- built from a scope stack that pushes on
    every ``class`` and every ``def`` / ``async def`` alike. A same-named
    method on an unrelated class, or a same-named nested function, therefore
    gets a distinct identity and is never conflated with a registered helper
    or exemption that happens to share its bare name.
    """

    def __init__(self, predicate: Callable[[ast.Call], bool]) -> None:
        self._predicate = predicate
        self._scope_stack: list[str] = []
        self._function_stack: list[str] = ["<module>"]
        self._savepoint_depth_stack: list[int] = [0]
        self.sites: list[_CallSite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def _visit_function(self, node: ast.AST) -> None:
        self._scope_stack.append(node.name)  # type: ignore[attr-defined]
        self._function_stack.append(".".join(self._scope_stack))
        self._savepoint_depth_stack.append(0)
        self.generic_visit(node)
        self._savepoint_depth_stack.pop()
        self._function_stack.pop()
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        # `is_savepoint` already requires the item's context expression to be
        # an actual call to the savepoint helper (see `_is_savepoint_context`)
        # -- an unrelated `async with` never raises the depth.
        is_savepoint = any(_is_savepoint_context(item.context_expr) for item in node.items)
        # The with-items (context expressions, `as` targets) are visited at
        # the *outer*, not-yet-raised depth: an append made inside the
        # savepoint helper's own call expression -- as one of its arguments,
        # say -- sits lexically inside this node but is evaluated *before*
        # the block it is about to open exists, so it must not count as
        # protected. Only `node.body` is visited at the raised depth.
        for item in node.items:
            self.visit(item)
        if is_savepoint:
            self._savepoint_depth_stack[-1] += 1
        for stmt in node.body:
            self.visit(stmt)
        if is_savepoint:
            self._savepoint_depth_stack[-1] -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if self._predicate(node):
            self.sites.append(
                _CallSite(
                    function=self._function_stack[-1],
                    lineno=node.lineno,
                    wrapped=self._savepoint_depth_stack[-1] > 0,
                )
            )
        self.generic_visit(node)


def _is_savepoint_context(expr: ast.expr) -> bool:
    """Whether an ``async with`` item's context expression is the savepoint helper."""
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == _SAVEPOINT_HELPER_NAME
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id == "self"
    )


def _is_journal_append_call(node: ast.Call) -> bool:
    f = node.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "append"
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "_journal"
        and isinstance(f.value.value, ast.Name)
        and f.value.value.id == "self"
    )


def _is_call_to_bare_name(node: ast.Call, bare_name: str) -> bool:
    """Whether ``node`` calls something named ``bare_name``, whatever the receiver.

    Deliberately over-inclusive: it matches an attribute call regardless of
    what the receiver expression is (``self._hygiene_gc(...)``,
    ``SqliteEngravaCore._hygiene_gc(self, ...)``, ``store._hygiene_gc(...)``),
    and a bare-name call with no receiver at all
    (``_hygiene_gc(...)``). A registered helper is a distinctive enough
    private name that a genuine, unrelated collision is not a real risk here
    -- missing a real call site because of an unusual receiver expression is
    the failure mode this guards against, not a false positive on a
    coincidence.
    """
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr == bare_name
    if isinstance(f, ast.Name):
        return f.id == bare_name
    return False


def _bare_name(qualified_name: str) -> str:
    """The last dotted segment of a fully qualified function name."""
    return qualified_name.rsplit(".", 1)[-1]


def _find_calls(tree: ast.Module, predicate: Callable[[ast.Call], bool]) -> list[_CallSite]:
    finder = _FunctionScopedCallFinder(predicate)
    finder.visit(tree)
    return finder.sites


class _QualifiedFunctionNameCollector(ast.NodeVisitor):
    """Collect the fully qualified scope path of every ``def`` / ``async def`` in the module.

    Mirrors `_FunctionScopedCallFinder`'s own scope-stack construction (class
    and function nesting both push a segment), kept as its own small,
    single-purpose walker rather than folded into the call finder, which is
    built around a call-matching predicate this has no use for.
    """

    def __init__(self) -> None:
        self._scope_stack: list[str] = []
        self.qualified_names: set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def _visit_function(self, node: ast.AST) -> None:
        self._scope_stack.append(node.name)  # type: ignore[attr-defined]
        self.qualified_names.add(".".join(self._scope_stack))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def _all_defined_function_names(tree: ast.Module) -> set[str]:
    """Every ``def`` / ``async def``'s fully qualified scope path in the module.

    Used to catch a stale registration directly: a rename (or deletion) of a
    registered exempt or helper function leaves its old qualified name out of
    this set entirely, distinct from -- and checked before -- whether any
    append call node or call site was found under that name.
    """
    collector = _QualifiedFunctionNameCollector()
    collector.visit(tree)
    return collector.qualified_names


def _check_exempt_function(function_name: str, sites: list[_CallSite]) -> list[str]:
    """A registered exemption's append count must match the recorded one exactly.

    Called with ``sites`` already looked up from the exemption table's own
    name, not from whatever ``by_function`` happens to contain -- a mutation
    that removes every append from an exempt function leaves it out of
    ``by_function`` altogether, and ``sites`` is then simply empty, which
    still trips this check (``0 != expected_count``) rather than silently
    skipping it.
    """
    _reason, expected_count = _EXEMPTIONS[function_name]
    if len(sites) == expected_count:
        return []
    return [
        (
            f"{function_name}: exempt with a recorded count of {expected_count} "
            f"append call node(s), but {len(sites)} found "
            f"(lines {[s.lineno for s in sites]})"
        )
    ]


def _check_unregistered_function(function_name: str, sites: list[_CallSite]) -> list[str]:
    """Every append in a plain (non-helper, non-exempt) function must be wrapped directly."""
    return [
        f"{function_name}:{site.lineno}: append is not inside a "
        f"{_SAVEPOINT_HELPER_NAME} block in its own function, and "
        f"{function_name} is neither a registered helper nor an exemption"
        for site in sites
        if not site.wrapped
    ]


def _check_helper_call_sites(tree: ast.Module, qualified_helper_name: str) -> list[str]:
    """Every call site of a registered helper must itself be wrapped.

    Matched by the helper's *bare* name against any call whose attribute (or
    bare-name) matches it, regardless of receiver -- see
    `_is_call_to_bare_name` -- since a call site's own receiver expression
    (``self``, the class itself, an instance held in another variable) is not
    what decides whether the call needs to be wrapped.
    """
    bare_name = _bare_name(qualified_helper_name)

    def _calls_helper(node: ast.Call, name: str = bare_name) -> bool:
        return _is_call_to_bare_name(node, name)

    call_sites = _find_calls(tree, _calls_helper)
    if not call_sites:
        return [
            (
                f"{qualified_helper_name}: registered as a helper but never "
                "called anywhere in the module"
            )
        ]
    return [
        f"{qualified_helper_name}: called from {site.function}:{site.lineno}, which is "
        f"not inside a {_SAVEPOINT_HELPER_NAME} block"
        for site in call_sites
        if not site.wrapped
    ]


def check_recurrence_guard(source: str) -> list[str]:
    """Verify every ``self._journal.append(...)`` call node in ``source``.

    Returns:
        A list of human-readable violation strings; empty when every append
        call node is accounted for (directly wrapped, wrapped transitively
        through a registered helper, or a registered exemption with a
        matching count). Non-empty means the guard fails.

    """
    tree = ast.parse(source)
    defined_functions = _all_defined_function_names(tree)
    append_sites = _find_calls(tree, _is_journal_append_call)

    by_function: dict[str, list[_CallSite]] = {}
    for site in append_sites:
        by_function.setdefault(site.function, []).append(site)

    violations: list[str] = []

    # Exemptions and helpers are driven from their own registration tables,
    # not from `by_function` -- a mutation that removes every append from a
    # registered function leaves it out of `by_function` entirely, and a
    # lookup keyed off `by_function` would then never visit that name at all.
    # Existence is checked first and explicitly: a renamed or deleted
    # function is a stale-table failure in its own right, worded distinctly
    # from "found 0 of the expected N", not merely folded into it.
    for function_name in _EXEMPTIONS:
        if function_name not in defined_functions:
            violations.append(
                f"{function_name}: registered as exempt, but no function by "
                "that name is defined anywhere in the module (renamed or "
                "deleted?)"
            )
            continue
        violations.extend(_check_exempt_function(function_name, by_function.get(function_name, [])))

    for helper_name in _HELPER_FUNCTIONS:
        if helper_name not in defined_functions:
            violations.append(
                f"{helper_name}: registered as a helper, but no function by "
                "that name is defined anywhere in the module (renamed or "
                "deleted?)"
            )
            continue
        violations.extend(_check_helper_call_sites(tree, helper_name))

    for function_name, sites in by_function.items():
        if function_name in _EXEMPTIONS or function_name in _HELPER_FUNCTIONS:
            # Already checked above, from the registration table's own name
            # rather than from this dict.
            continue
        violations.extend(_check_unregistered_function(function_name, sites))

    return violations


def _engrava_core_source() -> str:
    return inspect.getsource(engrava_core)


class TestRecurrenceGuardOnTheRealTree:
    def test_every_append_call_site_is_accounted_for(self) -> None:
        violations = check_recurrence_guard(_engrava_core_source())
        assert violations == []


# ---------------------------------------------------------------------------
# The guard must fail, not just pass: three scratch mutations, each adding one
# unprotected append the guard has to notice. Demonstrated on an in-memory
# scratch copy of the real source text -- nothing here touches the file on
# disk.
# ---------------------------------------------------------------------------

_UNPROTECTED_APPEND_STATEMENT = (
    "        await self._journal.append(\n"
    '            mutation_type="UPDATE_THOUGHT",\n'
    '            target_id="scratch-mutation-marker",\n'
    '            delta={"before": None, "after": None},\n'
    "        )\n"
)


def _insert_after_first_match(source: str, needle: str, insertion: str) -> str:
    """Insert ``insertion`` on the line right after the first line containing ``needle``."""
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if needle in line:
            return "".join([*lines[: index + 1], insertion, *lines[index + 1 :]])
    msg = f"needle not found: {needle!r}"
    raise AssertionError(msg)


def _insert_before_first_match(source: str, needle: str, insertion: str) -> str:
    """Insert ``insertion`` right before the first line containing ``needle``.

    Used where the insertion must land at the *same* indentation as an
    existing statement (so the result stays syntactically valid) rather than
    inside whatever block the needle line opens.
    """
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if needle in line:
            return "".join([*lines[:index], insertion, *lines[index:]])
    msg = f"needle not found: {needle!r}"
    raise AssertionError(msg)


class TestRecurrenceGuardCatchesAnUnprotectedAppend:
    def test_a_new_unprotected_append_on_an_already_registered_method(self) -> None:
        """(a) A second, unguarded append added inside an already-wrapped method.

        Inserted into ``delete_thought`` right before its own ``return
        deleted`` -- inside the method (so it is attributed to
        ``delete_thought``, a function already registered by rule 1), but
        outside its ``_write_readback_savepoint`` block, which has already
        closed a few lines earlier.
        """
        source = _engrava_core_source()
        mutated = _insert_before_first_match(
            source, "        return deleted\n", _UNPROTECTED_APPEND_STATEMENT
        )
        violations = check_recurrence_guard(mutated)
        assert any("delete_thought" in v for v in violations), violations

    def test_a_new_unprotected_append_on_a_brand_new_method(self) -> None:
        """(b) A whole new method with its own unguarded append, never registered anywhere."""
        source = _engrava_core_source()
        new_method = (
            "\n"
            "    async def _scratch_mutation_new_unguarded_append(self) -> None:\n"
            '        """Scratch mutation (b): a brand-new method the guard has never seen."""\n'
            + _UNPROTECTED_APPEND_STATEMENT
        )
        mutated = _insert_after_first_match(source, "class SqliteEngravaCore:", new_method)
        violations = check_recurrence_guard(mutated)
        assert any("_scratch_mutation_new_unguarded_append" in v for v in violations), violations

    def test_a_new_unprotected_append_next_to_an_exempt_functions_existing_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(c) A second append added beside the one an exemption's recorded count expects.

        A registered exemption's append count must be exact: a second append
        call node added beside the one it was recorded for must trip the count
        check even though the function is still exempt, not merely be waved
        through because the function's name is on the exemption table at all.
        No function on the real tree is exempt any more (``_insert_derived_row``
        and ``_insert_derived_edge`` moved to their own savepoint units), so
        this registers a scratch function as a synthetic exemption instead of
        depending on one of the real tree's own functions to stay exempt.
        """
        source = _engrava_core_source()
        scratch_method = (
            "\n"
            "    async def _scratch_mutation_exempt_with_extra_append(self) -> None:\n"
            '        """Scratch mutation (c): a registered exemption gains a second append."""\n'
            + _UNPROTECTED_APPEND_STATEMENT
            + _UNPROTECTED_APPEND_STATEMENT
        )
        mutated = _insert_after_first_match(source, "class SqliteEngravaCore:", scratch_method)
        monkeypatch.setitem(
            _EXEMPTIONS,
            "SqliteEngravaCore._scratch_mutation_exempt_with_extra_append",
            ("scratch exemption for this test", 1),
        )
        violations = check_recurrence_guard(mutated)
        assert any(
            "_scratch_mutation_exempt_with_extra_append" in v and "exempt" in v for v in violations
        ), violations


# ---------------------------------------------------------------------------
# Three more scratch mutations, one per rule the guard was found not to
# enforce on review: an exempt function whose only append is deleted (so it
# has zero call nodes and drops out of the by-function scan entirely), an
# exempt function renamed so its registered name no longer exists, and an
# append placed inside the savepoint helper's own context expression (so it
# is evaluated *before* the block it sits under opens).
# ---------------------------------------------------------------------------


class TestRecurrenceGuardCatchesAStaleOrMisscopedRegistration:
    def test_exempt_function_with_its_only_append_removed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(i) An exempt function whose only append is gone must still be a count mismatch.

        Before this was fixed, the guard only ever looked at functions that
        still had at least one append call node (keyed off `by_function`), so
        a function registered as exempt with count 1 that dropped to 0 simply
        never got checked at all -- it fell out of the scan, not just out of
        the count. No function on the real tree is exempt any more, so this
        registers a scratch function -- with no append at all -- as a
        synthetic exemption instead.
        """
        source = _engrava_core_source()
        scratch_method = (
            "\n"
            "    async def _scratch_mutation_exempt_now_empty(self) -> None:\n"
            '        """Scratch mutation (i): a registered exemption\'s append is gone."""\n'
            "        pass\n"
        )
        mutated = _insert_after_first_match(source, "class SqliteEngravaCore:", scratch_method)
        monkeypatch.setitem(
            _EXEMPTIONS,
            "SqliteEngravaCore._scratch_mutation_exempt_now_empty",
            ("scratch exemption for this test", 1),
        )
        violations = check_recurrence_guard(mutated)
        assert any(
            "_scratch_mutation_exempt_now_empty" in v and "0 found" in v for v in violations
        ), violations

    def test_renamed_exempt_function_no_longer_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(ii) A stale registration pointing at a function nobody defines must be caught.

        No function on the real tree is exempt any more, so this registers a
        synthetic exemption for a name that is never defined anywhere in the
        (unmodified) source -- the same stale-registration shape a rename
        would produce, without depending on a real function to rename.
        """
        source = _engrava_core_source()
        monkeypatch.setitem(
            _EXEMPTIONS,
            "SqliteEngravaCore._scratch_mutation_exempt_renamed_away",
            ("scratch exemption for this test", 1),
        )
        violations = check_recurrence_guard(source)
        assert any(
            "_scratch_mutation_exempt_renamed_away" in v
            and "no function by that name is defined" in v
            for v in violations
        ), violations

    def test_append_inside_the_savepoints_own_context_expression(self) -> None:
        """(iii) An append evaluated as part of the savepoint call itself is not protected.

        The savepoint block it sits beside does not exist yet while its own
        context expression is being evaluated, so an append there must be
        judged exactly as if no savepoint were present at all.
        """
        source = _engrava_core_source()
        new_method = (
            "\n"
            "    async def _scratch_mutation_append_in_context_expr(self) -> None:\n"
            '        """Scratch mutation (iii): append inside the context expression."""\n'
            "        async with self._write_readback_savepoint(\n"
            "            str(\n"
            "                await self._journal.append(\n"
            '                    mutation_type="UPDATE_THOUGHT",\n'
            '                    target_id="scratch-mutation-marker",\n'
            '                    delta={"before": None, "after": None},\n'
            "                )\n"
            "            )\n"
            "        ):\n"
            "            pass\n"
        )
        mutated = _insert_after_first_match(source, "class SqliteEngravaCore:", new_method)
        violations = check_recurrence_guard(mutated)
        assert any("_scratch_mutation_append_in_context_expr" in v for v in violations), violations


# ---------------------------------------------------------------------------
# Two more scratch mutations for a fourth gap found on review: a registered
# helper or exemption identified only by bare name would take a same-named
# method on an unrelated class for the real one, and would miss a real,
# unwrapped call site of the real one written with an unusual receiver.
# ---------------------------------------------------------------------------


class TestRecurrenceGuardDistinguishesFunctionsByQualifiedScope:
    def test_a_same_named_method_on_a_different_class_is_not_the_registered_helper(
        self,
    ) -> None:
        """(i) A decoy class's own `_hygiene_gc` must not borrow the real one's registration.

        Bare-name matching would see a method named `_hygiene_gc` with an
        unwrapped append and accept it as covered by
        `SqliteEngravaCore._hygiene_gc`'s helper registration -- the decoy
        never runs under `run_hygiene`'s savepoint at all.
        """
        source = _engrava_core_source()
        decoy_class = (
            "class _ScratchMutationDecoyClass:\n"
            '    """A same-named method on an unrelated class."""\n'
            "\n"
            "    async def _hygiene_gc(self) -> None:\n" + _UNPROTECTED_APPEND_STATEMENT + "\n\n"
        )
        mutated = decoy_class + source
        violations = check_recurrence_guard(mutated)
        assert any(
            "_ScratchMutationDecoyClass._hygiene_gc" in v
            and "neither a registered helper nor an exemption" in v
            for v in violations
        ), violations

    def test_an_unwrapped_call_with_an_unusual_receiver_is_still_a_call_site(self) -> None:
        """(ii) `SqliteEngravaCore._hygiene_gc(self, ...)` outside any savepoint still counts.

        Matching only ``self.<name>(...)`` receivers would miss this call
        entirely, silently treating the helper as fully covered by its one
        legitimate call site inside ``run_hygiene``.
        """
        source = _engrava_core_source()
        extra_call = (
            "        await SqliteEngravaCore._hygiene_gc(\n"
            "            self, policy=self._hygiene_policy, current_cycle=0, now=None\n"
            "        )\n"
        )
        mutated = _insert_before_first_match(source, "        return deleted\n", extra_call)
        violations = check_recurrence_guard(mutated)
        assert any(
            "SqliteEngravaCore._hygiene_gc" in v
            and "SqliteEngravaCore.delete_thought" in v
            and "not inside a" in v
            for v in violations
        ), violations

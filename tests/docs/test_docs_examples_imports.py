"""Layer 9 of the documentation-example tests — every import names something real.

The compile layer (Layer 2) proves every fenced ``python`` block is valid
Python; its phantom-API guard there catches a small, hand-maintained denylist
of *specific* known-bad tokens from past drift. This module raises the floor
for one whole *class* of that defect, independent of that denylist: it
resolves every ``import`` and ``from ... import`` statement a block makes
against the real, installed ``engrava`` package, and fails on any that name a
module or attribute that does not exist.

See ``tests/docs/_import_resolution.py`` for the resolution rule itself and,
importantly, for what it does **not** cover: an import that resolves cleanly
tells you nothing about whether the block later calls a method that does not
exist on the object that import produced. That is attribute resolution, a
different and considerably harder problem this module does not attempt.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from tests.docs._import_resolution import (
    _STATEMENT_BODY_CONTAINERS,
    DeadImport,
    resolve_imports,
)
from tests.docs._md_blocks import REPO_ROOT, CodeBlock, all_python_blocks

_ALL_BLOCKS = all_python_blocks()


def _block_id(block: CodeBlock) -> str:
    return block.location


@pytest.mark.parametrize("block", _ALL_BLOCKS, ids=[_block_id(b) for b in _ALL_BLOCKS])
def test_doc_block_imports_resolve(block: CodeBlock) -> None:
    """Every ``engrava``-owned import in a documentation block names something real.

    This says nothing about attribute access reached through a resolved
    import -- see the module docstring and ``_import_resolution.py``.
    """
    resolution = resolve_imports(block)
    if resolution.has_dead_imports:
        offenders = [f"from {dead.source} import {dead.name}" for dead in resolution.dead]
        pytest.fail(
            f"Documentation block at {block.location} imports a non-existent "
            f"engrava symbol: {', '.join(offenders)}. Fix the snippet in the "
            f"source Markdown file or the real API, whichever is wrong.",
        )


def test_documentation_has_python_examples_to_resolve() -> None:
    """Sanity check mirroring the compile layer's: the extractor found blocks.

    Guards against the parametrized test above vacuously passing because a
    path change made ``all_python_blocks()`` return nothing.
    """
    assert len(_ALL_BLOCKS) > 20, (
        f"expected the docs to contain many python blocks, found {len(_ALL_BLOCKS)}; "
        f"the Markdown extractor may be misconfigured."
    )


def test_import_resolution_examines_a_nonzero_number_of_imports() -> None:
    """The no-vacuous-pass guarantee for this layer, over the real corpus.

    A check that silently examines zero import statements would pass on
    every run regardless of what the documentation actually says -- exactly
    the failure mode this item exists to avoid. Also reports the census: how
    many blocks and how many import statements were actually examined.
    """
    total_checked = 0
    total_dead = 0
    blocks_with_checked_imports = 0
    for block in _ALL_BLOCKS:
        resolution = resolve_imports(block)
        if resolution.checked:
            blocks_with_checked_imports += 1
        total_checked += len(resolution.checked)
        total_dead += len(resolution.dead)

    assert total_checked > 0, (
        "the import walker examined zero import statements across the whole "
        "documentation corpus -- it is almost certainly misconfigured, not "
        "reporting a genuinely import-free set of docs."
    )
    print(  # noqa: T201 - intentional census summary for the -s report
        f"\nImport-resolution census: blocks={len(_ALL_BLOCKS)} "
        f"blocks-with-checked-imports={blocks_with_checked_imports} "
        f"imports-checked={total_checked} dead={total_dead}",
    )


# ---------------------------------------------------------------------------
# Unit-level demonstrations of the resolution rule itself, independent of the
# current state of the documentation: failability, the control in the other
# direction, and the shapes the module docstring calls out by name.
# ---------------------------------------------------------------------------


def _synthetic_block(body: str) -> CodeBlock:
    return CodeBlock(path=REPO_ROOT / "README.md", rel="README.md", start_line=1, body=body)


def test_rule_flags_a_from_import_naming_a_nonexistent_symbol() -> None:
    """The motivating defect: ``from engrava import DoesNotExistAtAll`` is unambiguous."""
    block = _synthetic_block("from engrava import DoesNotExistAtAll\n")
    resolution = resolve_imports(block)
    assert resolution.has_dead_imports
    assert resolution.dead == (DeadImport(source="engrava", name="DoesNotExistAtAll"),)


def test_rule_does_not_flag_a_real_from_import() -> None:
    """The control: importing a real symbol from a real engrava module passes."""
    block = _synthetic_block("from engrava import SqliteEngravaCore\n")
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports
    assert resolution.checked == (("engrava", "SqliteEngravaCore"),)


def test_rule_does_not_flag_a_real_but_unusual_symbol() -> None:
    """The control in the other direction: a real, less-common symbol must not be flagged.

    ``ThoughtType`` is an enum, not one of the handful of CRUD classes a
    reader sees most often -- exactly the "unusual but real" shape a checker
    that merely pattern-matches common imports would miss or, worse, flag.
    """
    block = _synthetic_block("from engrava import ThoughtType\n")
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports


def test_rule_resolves_a_documented_lazy_module_export() -> None:
    """A name served only by ``engrava.__getattr__`` must not be flagged dead.

    ``SqliteMindStoreCore`` is a real, documented deprecated alias resolved
    dynamically by ``engrava``'s own ``__getattr__``, invisible to a plain
    ``dir()`` snapshot -- ``hasattr`` still resolves it correctly.
    """
    block = _synthetic_block("from engrava import SqliteMindStoreCore\n")
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports


def test_rule_flags_a_plain_import_of_a_nonexistent_engrava_module() -> None:
    block = _synthetic_block("import engrava.this_does_not_exist\n")
    resolution = resolve_imports(block)
    assert resolution.has_dead_imports
    assert resolution.dead == (
        DeadImport(source="engrava.this_does_not_exist", name="engrava.this_does_not_exist"),
    )


def test_rule_does_not_flag_a_real_dotted_submodule_import() -> None:
    block = _synthetic_block("import engrava.config\n")
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports


def test_rule_ignores_a_non_engrava_import() -> None:
    """Out of scope by design: only ``engrava``-owned imports are checked.

    A block importing a package that genuinely does not exist anywhere is
    not this module's concern -- see the module docstring.
    """
    block = _synthetic_block("import this_package_does_not_exist_anywhere\n")
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports
    assert resolution.checked == ()


def test_rule_does_not_attempt_to_resolve_a_star_import() -> None:
    """A star import names no specific symbol -- nothing to check, and not a failure."""
    block = _synthetic_block("from engrava import *\n")
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports
    assert resolution.checked == ()


def test_rule_flags_a_star_import_from_a_nonexistent_module() -> None:
    """A star import names no specific symbol, but the module it names must still exist.

    ``from engrava import *`` (the module above) rightly skips enumerating
    members. That must not extend to skipping the module itself: the
    control below shows the very same nonexistent module is already caught
    when a *named* symbol is imported from it, so a star import must not be
    a way to launder a dead module past this check.
    """
    block = _synthetic_block("from engrava.this_does_not_exist import *\n")
    resolution = resolve_imports(block)
    assert resolution.has_dead_imports
    assert resolution.dead == (DeadImport(source="engrava.this_does_not_exist", name="*"),)

    # CONTROL, same dead module, a named symbol instead of `*`.
    named = _synthetic_block("from engrava.this_does_not_exist import Thing\n")
    named_resolution = resolve_imports(named)
    assert named_resolution.has_dead_imports
    assert named_resolution.dead == (
        DeadImport(source="engrava.this_does_not_exist", name="Thing"),
    )

    # CONTROL, a real module: star import must stay clean.
    real = _synthetic_block("from engrava import *\n")
    assert not resolve_imports(real).has_dead_imports


def test_rule_flags_a_relative_import_of_an_engrava_named_module() -> None:
    """A relative import can never resolve against the installed absolute package.

    ``from .engrava import SqliteEngravaCore`` cannot run in a standalone
    documentation block -- there is no enclosing package for the leading dot
    to be relative TO, so Python raises ``ImportError: attempted relative
    import with no known parent package`` before it ever looks at what
    ``engrava`` provides. Probing the installed top-level ``engrava``
    package as if the dot were not there (the previous behaviour) reports a
    broken statement as clean.
    """
    block = _synthetic_block("from .engrava import SqliteEngravaCore\n")
    resolution = resolve_imports(block)
    assert resolution.has_dead_imports
    assert resolution.dead == (DeadImport(source=".engrava", name="SqliteEngravaCore"),)

    # CONTROL, the absolute equivalent of the exact same symbol stays clean.
    absolute = _synthetic_block("from engrava import SqliteEngravaCore\n")
    assert not resolve_imports(absolute).has_dead_imports


def test_rule_flags_a_two_level_relative_import_too() -> None:
    """``level`` can be greater than 1 (``from ..engrava import X``); both must be caught."""
    block = _synthetic_block("from ..engrava import SqliteEngravaCore\n")
    resolution = resolve_imports(block)
    assert resolution.has_dead_imports
    assert resolution.dead == (DeadImport(source="..engrava", name="SqliteEngravaCore"),)


def test_rule_resolves_an_unimported_submodule_named_via_from_import() -> None:
    """``from engrava import cli`` is valid even if nothing has imported ``cli`` yet.

    ``from package import name`` is not equivalent to ``hasattr(package,
    name)``: when ``name`` is not already bound as an attribute of
    ``package``, CPython's own import machinery retries by importing
    ``package.name`` as a submodule before giving up. ``engrava.cli`` is a
    real submodule that is not eagerly imported by ``engrava/__init__.py``,
    so a plain ``hasattr`` check reports it dead even though the statement
    runs fine. The controls show the other two directions still work: a
    truly nonexistent attribute of a real module, and a named symbol from a
    module that does not exist at all, are still reported.
    """
    block = _synthetic_block("from engrava import cli\n")
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports
    assert resolution.checked == (("engrava", "cli"),)

    # CONTROL, a genuinely nonexistent attribute of a real module.
    missing = _synthetic_block("from engrava import NoSuchSymbol\n")
    assert resolve_imports(missing).has_dead_imports

    # CONTROL, a named symbol from a module that does not exist at all.
    dead_module = _synthetic_block("from engrava.this_does_not_exist import Thing\n")
    assert resolve_imports(dead_module).has_dead_imports


def test_rule_resolving_an_unimported_submodule_does_not_depend_on_import_order() -> None:
    """The verdict for ``from engrava import cli`` must not depend on session history.

    A pure ``hasattr`` check does not just miss an unimported submodule --
    it is order-dependent: CPython's import machinery binds an imported
    submodule onto its parent package as a side effect, so if something
    *else* earlier in the same process already imported ``engrava.cli``,
    ``hasattr(engrava, "cli")`` silently flips to ``True`` and the same
    check that failed a moment ago now passes, for a reason that has
    nothing to do with the documentation block being checked.

    This runs the resolution twice in a single, otherwise-fresh subprocess:
    once before anything has touched ``engrava.cli`` (this is the shape a
    real pytest session hits on some import orders and not others), and
    once again immediately after explicitly importing ``engrava.cli``
    directly (reproducing the side effect that used to change the answer).
    Both must report the same, correct verdict for both a real submodule
    and a genuinely nonexistent one -- proving the fix consults the actual
    submodule, not whatever a prior import happened to leave bound on the
    parent package.
    """
    script = """
import sys
sys.path.insert(0, {src!r})
sys.path.insert(0, {repo_root!r})

from pathlib import Path
from tests.docs._md_blocks import CodeBlock
from tests.docs._import_resolution import resolve_imports

def resolves_clean(src_text):
    block = CodeBlock(path=Path("docs/probe.md"), rel="docs/probe.md", start_line=1, body=src_text)
    return not resolve_imports(block).has_dead_imports

# Before anything else has imported engrava.cli.
assert "engrava.cli" not in sys.modules
assert resolves_clean("from engrava import cli\\n"), "cli should resolve before any prior import"
assert not resolves_clean("from engrava import NoSuchSymbol\\n"), "NoSuchSymbol must stay dead"

# Now force the side effect a real submodule import leaves on the parent
# package, and prove the verdict for both names is unchanged.
import engrava.cli  # noqa: F401
assert hasattr(__import__("engrava"), "cli")
assert resolves_clean("from engrava import cli\\n"), "cli should still resolve after a prior import"
assert not resolves_clean("from engrava import NoSuchSymbol\\n"), (
    "NoSuchSymbol must not be affected by an unrelated prior import"
)
print("ORDER_INDEPENDENCE_OK")
""".format(src=str(REPO_ROOT / "src"), repo_root=str(REPO_ROOT))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
    result = subprocess.run(  # noqa: S603 -- fixed argv, no shell, our own probe source
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"order-independence subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "ORDER_INDEPENDENCE_OK" in result.stdout


def test_rule_does_not_infer_anything_about_attribute_access() -> None:
    """The explicit boundary: a real import followed by a dead method call passes here.

    This is not a gap in this test -- it is the documented scope boundary
    (see the module and ``_import_resolution.py`` docstrings). Attribute
    resolution is a separate, harder problem this module does not attempt.
    """
    block = _synthetic_block(
        "from engrava import SqliteEngravaCore\nstore = SqliteEngravaCore(conn)\nstore.info()\n",
    )
    resolution = resolve_imports(block)
    assert not resolution.has_dead_imports


# ---------------------------------------------------------------------------
# Structural coverage: a dead import must be found no matter which kind of
# statement-body construct it is nested inside. This is the "AST node-type
# classification" half of this layer -- see the module docstring and
# ``_STATEMENT_BODY_CONTAINERS`` in ``_import_resolution.py`` for why
# import resolution does not need the scope-tracking machinery the (out of
# scope) attribute resolver required to get this right.
# ---------------------------------------------------------------------------

_NESTING_CONSTRUCTS: tuple[tuple[str, str], ...] = (
    ("module level", "from engrava import DoesNotExistAtAll\n"),
    ("function body", "def f():\n    from engrava import DoesNotExistAtAll\n"),
    ("async function body", "async def f():\n    from engrava import DoesNotExistAtAll\n"),
    ("class body", "class C:\n    from engrava import DoesNotExistAtAll\n"),
    ("if body", "if True:\n    from engrava import DoesNotExistAtAll\n"),
    ("if orelse", "if True:\n    pass\nelse:\n    from engrava import DoesNotExistAtAll\n"),
    ("for body", "for _ in []:\n    from engrava import DoesNotExistAtAll\n"),
    (
        "async for body",
        "async def f():\n    async for _ in x:\n        from engrava import DoesNotExistAtAll\n",
    ),
    ("while body", "while False:\n    from engrava import DoesNotExistAtAll\n"),
    ("with body", "with open('x') as fh:\n    from engrava import DoesNotExistAtAll\n"),
    (
        "async with body",
        "async def f():\n    async with x:\n        from engrava import DoesNotExistAtAll\n",
    ),
    ("try body", "try:\n    from engrava import DoesNotExistAtAll\nexcept Exception:\n    pass\n"),
    (
        "except handler body",
        "try:\n    pass\nexcept Exception:\n    from engrava import DoesNotExistAtAll\n",
    ),
    (
        "try finally body",
        "try:\n    pass\nfinally:\n    from engrava import DoesNotExistAtAll\n",
    ),
    (
        "match case body",
        "match 1:\n    case _:\n        from engrava import DoesNotExistAtAll\n",
    ),
    (
        "deeply nested (function in class in function)",
        (
            "def outer():\n"
            "    class C:\n"
            "        def m(self):\n"
            "            from engrava import DoesNotExistAtAll\n"
        ),
    ),
)


@pytest.mark.parametrize(
    ("label", "body"),
    _NESTING_CONSTRUCTS,
    ids=[label for label, _ in _NESTING_CONSTRUCTS],
)
def test_walker_finds_a_dead_import_in_every_statement_body_construct(
    label: str,
    body: str,
) -> None:
    resolution = resolve_imports(_synthetic_block(body))
    assert resolution.has_dead_imports, f"a dead import nested inside a {label} was not found"


def _leaf_ast_node_types() -> list[type[ast.AST]]:
    """Every ast node type that can actually be a real node's exact class.

    Filters out the abstract grouping bases (``ast.expr``, ``ast.stmt``,
    ``ast.operator``, ...), which are never a node's own ``type()`` -- only
    one of their concrete subclasses ever is.
    """
    all_types = {
        obj for obj in vars(ast).values() if isinstance(obj, type) and issubclass(obj, ast.AST)
    }
    has_subclass = {base for cls in all_types for base in cls.__mro__[1:] if base in all_types}
    return sorted(all_types - has_subclass, key=lambda cls: cls.__name__)


# Every leaf ast node type that is neither `Import`/`ImportFrom` (the two
# types this layer explicitly checks) nor a reviewed statement-body container
# (`_STATEMENT_BODY_CONTAINERS`) -- reviewed once, by hand, as a type whose
# own shape cannot hold a nested statement at all (an operator, a context
# marker, an expression, a pattern-matching node, a parameter/argument shape,
# or a legacy eval-mode-only wrapper `ast.parse(mode="exec")` never
# produces). This list exists so a future Python grammar addition that
# introduces a genuinely new statement-body construct is forced onto
# `_STATEMENT_BODY_CONTAINERS` by a failing test, rather than silently
# falling into "presumably harmless" the way an unlisted construct hid a real
# hole in the (out of scope) attribute resolver's history.
_CANNOT_HOLD_A_NESTED_STATEMENT: frozenset[str] = frozenset(
    {
        "Add",
        "And",
        "AnnAssign",
        "Assert",
        "Assign",
        "Attribute",
        "AugAssign",
        "AugLoad",
        "AugStore",
        "Await",
        "BinOp",
        "BitAnd",
        "BitOr",
        "BitXor",
        "BoolOp",
        "Break",
        "Call",
        "Compare",
        "Continue",
        "Del",
        "Delete",
        "Dict",
        "DictComp",
        "Div",
        "Ellipsis",
        "Eq",
        "Expr",
        "Expression",
        "ExtSlice",
        "FloorDiv",
        "FormattedValue",
        "FunctionType",
        "GeneratorExp",
        "Global",
        "Gt",
        "GtE",
        "IfExp",
        "In",
        "Index",
        "Invert",
        "Is",
        "IsNot",
        "JoinedStr",
        "LShift",
        "Lambda",
        "List",
        "ListComp",
        "Load",
        "Lt",
        "LtE",
        "MatMult",
        "Match",
        "MatchAs",
        "MatchClass",
        "MatchMapping",
        "MatchOr",
        "MatchSequence",
        "MatchSingleton",
        "MatchStar",
        "MatchValue",
        "Mod",
        "Mult",
        "Name",
        "NamedExpr",
        "Nonlocal",
        "Not",
        "NotEq",
        "NotIn",
        "Or",
        "Param",
        "ParamSpec",
        "Pass",
        "Pow",
        "RShift",
        "Raise",
        "Return",
        "Set",
        "SetComp",
        "Slice",
        "Starred",
        "Store",
        "Sub",
        "Subscript",
        "Suite",
        "Tuple",
        "TypeAlias",
        "TypeIgnore",
        "TypeVar",
        "TypeVarTuple",
        "UAdd",
        "USub",
        "UnaryOp",
        "Yield",
        "YieldFrom",
        "alias",
        "arg",
        "arguments",
        "comprehension",
        "keyword",
        "withitem",
    },
)


def test_every_ast_node_type_is_reviewed_for_holding_a_nested_import() -> None:
    """Every concrete ast node type falls into exactly one reviewed bucket.

    The buckets: explicitly checked (``Import``/``ImportFrom``), a reviewed
    statement-body container that could hold one nested inside it
    (``_STATEMENT_BODY_CONTAINERS``), or reviewed as structurally incapable
    of holding a statement at all (``_CANNOT_HOLD_A_NESTED_STATEMENT``). A
    node type in none of the three -- most likely a future Python grammar
    addition -- fails here rather than silently defaulting to "presumably
    fine", and a node type in more than one hides which rule actually
    governs it.
    """
    container_names = {node_type.__name__ for node_type in _STATEMENT_BODY_CONTAINERS}
    explicit_names = {"Import", "ImportFrom"}

    for node_type in _leaf_ast_node_types():
        name = node_type.__name__
        is_explicit = name in explicit_names
        is_container = name in container_names
        is_inert = name in _CANNOT_HOLD_A_NESTED_STATEMENT
        bucket_count = sum((is_explicit, is_container, is_inert))
        assert bucket_count >= 1, (
            f"ast.{name} is not explicitly checked, not a reviewed statement-body "
            f"container, and not on the reviewed 'cannot hold a statement' list -- "
            f"most likely a node type a newer Python grammar added. Review it: does "
            f"this node type's OWN shape carry a list of statements (add it to "
            f"_STATEMENT_BODY_CONTAINERS) or not (add it to "
            f"_CANNOT_HOLD_A_NESTED_STATEMENT)?"
        )
        assert bucket_count == 1, (
            f"ast.{name} is classified in more than one bucket (explicit={is_explicit}, "
            f"container={is_container}, inert={is_inert}) -- pick exactly one."
        )

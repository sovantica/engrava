"""Static import resolution for documentation code blocks.

The compile layer (``test_docs_examples_compile.py``) proves a block is
syntactically valid Python. It says nothing about whether the modules and
names a block *imports* actually exist: ``from engrava import DoesNotExist``
compiles just as cleanly as ``from engrava import SqliteEngravaCore``. This
module closes that specific gap: it walks every ``import`` and
``from ... import`` statement in a block, and for every absolute import that
names an ``engrava``-owned module, resolves it against the real, installed
package. Every relative import is reported unresolvable (see below).

What this module deliberately does NOT do
------------------------------------------
This is import resolution, not attribute resolution. Once a name is bound by
a resolved import, whatever the *rest of the block* does with that name --
``store.recall(...)``, ``store.foo(...)``, a chain of attribute access several
lines later -- is entirely outside what this module checks. A block can
import only real symbols and still call a method that has never existed on
any of them; this module reports that block as clean, because it is, by the
one question this module asks. Verifying attribute access against the type a
name is bound to is a different, considerably harder problem -- an earlier
attempt at it (see the version-controlled history this module's sibling
replaced) kept finding a real Python binding/scoping rule it had not
modelled, including cases where it stayed silent about a genuine dead
symbol. That problem is deliberately out of scope here. Nothing in this
module infers a name's type from a constructor call, an annotation, or
anything else -- it looks at import statements and nothing else.

Only ``engrava``-owned imports are resolved. A block's ``import asyncio`` or
``from pathlib import Path`` is out of scope: this check protects the surface
the documentation is actually about.

A relative import (``ast.ImportFrom.level > 0``, e.g. ``from . import X`` or
``from ..engrava.config import Y``) resolves against an enclosing package. A
documentation block read as a standalone script has none, so a relative
import in it can never resolve. This module reports every relative import as
unresolvable, whatever it names, without looking up any package.

Scope, not flow
-----------------
Whether an import target exists has no dependence on control flow, branching,
or binding history -- the class of problem that made attribute resolution
need a restricted-grammar trust model (see the sibling module's history).
``from engrava import DoesNotExist`` is dead whether it sits at module level,
inside a function, a class, a conditional, a loop, a ``try``/``except``, a
``with``, or a ``match`` case: it is checked once, unconditionally, wherever
it appears. That is why :class:`_ImportWalker` below does not override
``generic_visit`` and tracks no bindings or scopes at all -- the stock
``ast.NodeVisitor.generic_visit`` already recurses into every child field of
every node type, so completeness does not depend on this module reviewing
every node type by hand. ``test_docs_examples_imports.py`` demonstrates that
directly (a dead import nested inside every kind of statement-body construct
is still found) and separately proves, from ``ast``'s own grammar, exactly
which node types could ever legally carry a nested import in the first place
(``_STATEMENT_BODY_CONTAINERS`` below) -- a tripwire that fails loudly
if a future Python version adds a new statement-body construct nobody has
looked at yet, not a mechanism this module's own correctness depends on.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.docs._md_blocks import CodeBlock

_ENGRAVA_PREFIX = "engrava"


def _is_engrava_module(name: str) -> bool:
    return name == _ENGRAVA_PREFIX or name.startswith(f"{_ENGRAVA_PREFIX}.")


@cache
def _import_engrava_module(name: str) -> object | None:
    """Import an ``engrava``-owned module by dotted name, or ``None`` on failure.

    Cached because the same modules (``engrava``, ``engrava.config``, ...) are
    imported repeatedly across hundreds of blocks.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _safe_hasattr(obj: object, name: str) -> bool:
    """``hasattr`` with warnings from a real ``__getattr__`` muted.

    ``engrava.__getattr__`` raises ``DeprecationWarning`` for legacy aliases
    (e.g. ``SqliteMindStoreCore``) -- real runtime behaviour a reader would
    also see, but this module is doing *static* analysis, so probing whether
    an attribute exists must not emit the same warning a genuine import
    would.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return hasattr(obj, name)


def _from_import_target_exists(module: object, module_name: str, symbol_name: str) -> bool:
    """Decide whether ``from module_name import symbol_name`` would succeed.

    A plain ``hasattr(module, symbol_name)`` is not what CPython actually
    does for ``from package import name``: if ``name`` is not already bound
    as an attribute of ``package``, the import machinery retries by
    importing ``package.name`` as a submodule (``importlib._bootstrap``'s
    fromlist handling) before giving up. A submodule that has never been
    imported yet -- ``engrava.cli``, say -- is a real, resolvable target
    that plain ``hasattr`` reports as dead; worse, if some *other*, unrelated
    import earlier in the same process already imported that submodule,
    Python's own import system binds it onto the parent package as a side
    effect, and ``hasattr`` alone would then flip to ``True`` -- making the
    verdict depend on import order rather than on whether the submodule
    exists. Attempting the submodule import directly, through the same
    cached :func:`_import_engrava_module` every other check in this module
    goes through, gives an answer that depends only on whether
    ``module_name.symbol_name`` exists on disk, never on what happened to
    run earlier in the session.
    """
    if _safe_hasattr(module, symbol_name):
        return True
    return _import_engrava_module(f"{module_name}.{symbol_name}") is not None


@dataclass(frozen=True)
class DeadImport:
    """An import that cannot resolve.

    Either an absolute import naming an ``engrava`` module or attribute that
    does not exist, or any relative import, which can never resolve in a
    block read as a standalone script.
    """

    source: str
    name: str


@dataclass(frozen=True)
class ImportResolution:
    """The result of resolving one code block's ``engrava``-owned imports.

    Attributes:
        checked: Every ``(source, name)`` pair this module actually looked up
            against the real package -- every ``engrava``-owned import
            statement, minus any ``*`` member (see the module docstring: a
            star import names no specific symbol to check) and minus any
            relative import (its target is never looked up against the
            installed package at all -- see the module docstring). This is
            the count a caller reports to show the check examined
            something, rather than silently passing over an empty block.
        dead: Every checked pair that does not exist in the installed
            package, plus two shapes that are reported dead without ever
            appearing in ``checked``: a star import (``from x import *``)
            whose named module itself does not exist, and any relative
            import, whatever it names -- see the module docstring.

    """

    checked: tuple[tuple[str, str], ...]
    dead: tuple[DeadImport, ...]

    @property
    def has_dead_imports(self) -> bool:
        return bool(self.dead)


# Node types whose OWN shape carries a body of statements -- a list literally
# named `body`, `orelse`, `finalbody`, or a per-branch body reached through
# `handlers`/`cases` -- and could therefore legally hold a nested
# `ast.Import`/`ast.ImportFrom` directly inside it. Reviewed once, by hand,
# against Python's grammar: an import is an `ast.stmt`, and a statement can
# only occur inside one of these containers' own statement-list field(s),
# never inside an expression, a pattern, a comprehension, a parameter list, or
# any other node type reachable only through an `expr`-shaped field --
# `ast.Lambda` and `ast.IfExp` both have a field literally named `body`, but
# each holds exactly one `expr`, never a list of `stmt`, so neither belongs
# here. `ast.Match` holds its per-case bodies indirectly through `match_case`
# (already listed) rather than a `body` field of its own.
_STATEMENT_BODY_CONTAINERS: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.Interactive,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.ExceptHandler,
    ast.match_case,
)


class _ImportWalker(ast.NodeVisitor):
    """Finds every ``Import``/``ImportFrom`` node in a block, however nested.

    Deliberately does not override ``generic_visit``: see the module
    docstring for why import resolution needs no scope or binding model at
    all, unlike the (out of scope) problem of resolving attribute access.
    """

    def __init__(self) -> None:
        self.checked: list[tuple[str, str]] = []
        self.dead: list[DeadImport] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _is_engrava_module(alias.name):
                continue
            self.checked.append((alias.name, alias.name))
            if _import_engrava_module(alias.name) is None:
                self.dead.append(DeadImport(source=alias.name, name=alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level > 0:
            # A relative import can never resolve when a documentation block
            # is run standalone -- see the module docstring. Report each
            # imported name, or `*` for a star import, as dead, whatever
            # module it names: being relative already rules out resolving it
            # here, so the engrava-only filter below does not apply to it.
            source = f"{'.' * node.level}{node.module or ''}"
            for alias in node.names:
                self.dead.append(DeadImport(source=source, name=alias.name))
            self.generic_visit(node)
            return
        module_name = node.module
        if module_name is None or not _is_engrava_module(module_name):
            self.generic_visit(node)
            return
        module = _import_engrava_module(module_name)
        for alias in node.names:
            if alias.name == "*":
                # A star import names no specific symbol to check -- see the
                # module docstring -- but the module it names must still
                # exist: a star import from a nonexistent module is just as
                # dead as a named one, even though there is no per-symbol
                # name to add to `checked`.
                if module is None:
                    self.dead.append(DeadImport(source=module_name, name="*"))
                continue
            self.checked.append((module_name, alias.name))
            if module is None or not _from_import_target_exists(module, module_name, alias.name):
                self.dead.append(DeadImport(source=module_name, name=alias.name))
        self.generic_visit(node)


def resolve_imports(block: CodeBlock) -> ImportResolution:
    """Resolve every ``engrava``-owned import in one documentation block.

    Args:
        block: The extracted fenced-``python`` block to check.

    Returns:
        An :class:`ImportResolution` naming any dead imports (hard failures)
        and every ``(source, name)`` pair actually checked.

    """
    tree = ast.parse(block.body, filename=block.location)
    walker = _ImportWalker()
    walker.visit(tree)
    return ImportResolution(checked=tuple(walker.checked), dead=tuple(walker.dead))

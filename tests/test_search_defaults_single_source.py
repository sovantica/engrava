"""Hybrid-search defaults have exactly one source: ``SearchConfig``.

A store built with the direct constructor and no ``search_config`` used to
carry its own set of literal fallbacks in ``_resolve_hybrid_defaults`` and
``search_reflections_only`` — a second, separately maintained copy of
``SearchConfig``'s own field defaults that could (and did, for recency) drift
from it. ``config._parse_search`` carried a third copy, for the YAML loader's
own fallback when a key is omitted.

Covers:
- A direct-constructor store's resolved hybrid defaults equal
  ``SearchConfig()``'s own field defaults, field by field — including
  ``default_recency_weight`` (``0.1``), not a stale ``0.0``.
- ``search_reflections_only``'s recency blend resolves the same recency
  weight and half-life as ``search_hybrid`` on the same direct-constructor
  store.
- ``_parse_search(None)`` and ``_parse_search({})`` resolve the same values
  as ``SearchConfig()`` — an omitted ``search:`` section and an empty one are
  indistinguishable.
- Neither ``_resolve_hybrid_defaults``, ``search_reflections_only``'s recency
  blend, nor ``_parse_search``'s six weight/half-life fallbacks carry a
  duplicated numeric literal: each reads it from ``SearchConfig``.
- An explicit per-call weight, and an explicit ``SearchConfig``, still win
  over the shared default (see ``tests/test_hybrid_search_enhanced.py::
  TestRecencyScoring::test_recency_exp_decay_formula`` for the direct-
  constructor case already covered elsewhere).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import SearchConfig, SqliteEngravaCore
from engrava.config import _parse_search
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore as _CoreClass

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


def _reflection(thought_id: str) -> ThoughtRecord:
    """Minimal REFLECTION thought, eligible for search_reflections_only."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.REFLECTION,
        essence="reflection",
        content="reflection content",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
    )


@pytest.fixture
async def direct_store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """A store built with the direct constructor and no ``SearchConfig``."""
    conn = await aiosqlite.connect(str(tmp_path / "direct.db"))
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield store
    await conn.close()


# ---------------------------------------------------------------------------
# _resolve_hybrid_defaults() — equality with SearchConfig()
# ---------------------------------------------------------------------------


class TestResolveHybridDefaultsSingleSource:
    """A direct-constructor store's resolved defaults equal SearchConfig()'s."""

    async def test_matches_search_config_field_by_field(
        self, direct_store: SqliteEngravaCore
    ) -> None:
        """Red at the base: recency resolved 0.0 there, SearchConfig() is 0.1."""
        assert direct_store._search_config is None
        expected = SearchConfig()
        resolved = direct_store._resolve_hybrid_defaults(
            fts_weight=None,
            vector_weight=None,
            recency_weight=None,
            recency_half_life=None,
            priority_weight=None,
            graph_weight=None,
        )
        assert resolved == (
            expected.default_fts_weight,
            expected.default_vector_weight,
            expected.default_recency_weight,
            expected.recency_half_life,
            expected.default_priority_weight,
            expected.default_graph_weight,
        )

    def test_recency_default_is_active_not_zero(self) -> None:
        """The specific defect this module guards: recency defaults to 0.1."""
        assert SearchConfig().default_recency_weight == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# search_reflections_only() — same recency weight as search_hybrid
# ---------------------------------------------------------------------------


class TestSearchReflectionsOnlyMatchesSearchHybridRecency:
    """search_reflections_only's recency blend agrees with search_hybrid's.

    Both methods have their own fallback when the store has no
    ``SearchConfig``; this proves they resolve to the identical weight and
    half-life on the identical store, by comparing the ranked output of a
    direct-constructor store against an otherwise-identical store built with
    an explicit ``SearchConfig()`` — two stores that must rank identically
    once both resolve the same shared default.
    """

    @staticmethod
    async def _build(
        tmp_path: Path, *, with_config: bool, name: str
    ) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
        conn = await aiosqlite.connect(str(tmp_path / f"{name}.db"))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn, search_config=SearchConfig() if with_config else None)
        await store.ensure_schema()
        return store, conn

    async def test_recency_blend_identical_between_direct_and_explicit_config(
        self, tmp_path: Path
    ) -> None:
        direct, direct_conn = await self._build(tmp_path, with_config=False, name="direct")
        explicit, explicit_conn = await self._build(tmp_path, with_config=True, name="explicit")
        try:
            for store in (direct, explicit):
                assert direct._search_config is None
                await store.create_thought(_reflection("r-1"))
                await store.create_thought(_reflection("r-2"))
                await store.store_embedding("r-1", [1.0, 0.0], model_name="test")
                await store.store_embedding("r-2", [0.0, 1.0], model_name="test")

            direct_result = await direct.search_reflections_only("", [0.8, 0.2], current_cycle=10)
            explicit_result = await explicit.search_reflections_only(
                "", [0.8, 0.2], current_cycle=10
            )
            assert direct_result.results == explicit_result.results
            assert direct_result.backends_used == explicit_result.backends_used
            assert "recency" in direct_result.backends_used
        finally:
            await direct_conn.close()
            await explicit_conn.close()


# ---------------------------------------------------------------------------
# _parse_search() — an omitted section and an empty one resolve identically
# ---------------------------------------------------------------------------


class TestParseSearchSingleSource:
    """_parse_search(None) and _parse_search({}) equal SearchConfig()."""

    def test_omitted_section_matches_search_config(self) -> None:
        assert _parse_search(None) == SearchConfig()

    def test_empty_section_matches_search_config(self) -> None:
        assert _parse_search({}) == SearchConfig()

    def test_omitted_and_empty_agree_with_each_other(self) -> None:
        assert _parse_search(None) == _parse_search({})


# ---------------------------------------------------------------------------
# No duplicated numeric literal remains in any of the three fallbacks
# ---------------------------------------------------------------------------


class TestNoDuplicatedLiteralFallbackRemains:
    """Every default-weight fallback reads SearchConfig; none repeats a value.

    The check is structural, not a search for the old text: any numeric
    literal used as a fallback, of whatever value, means that field's default
    has drifted back to a second, independently maintained copy instead of
    SearchConfig's own.
    """

    @staticmethod
    def _numeric_else_branches(func: object) -> list[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))  # type: ignore[arg-type]
        return [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.IfExp)
            and isinstance(node.orelse, ast.Constant)
            and isinstance(node.orelse.value, int | float)
            and not isinstance(node.orelse.value, bool)
        ]

    def test_resolve_hybrid_defaults_has_no_literal_fallback(self) -> None:
        offenders = self._numeric_else_branches(_CoreClass._resolve_hybrid_defaults)
        assert offenders == [], (
            f"_resolve_hybrid_defaults falls back to a numeric literal: {offenders}"
        )

    def test_search_reflections_only_has_no_literal_fallback(self) -> None:
        offenders = self._numeric_else_branches(_CoreClass.search_reflections_only)
        assert offenders == [], (
            f"search_reflections_only falls back to a numeric literal: {offenders}"
        )

    def test_parse_search_has_no_literal_fallback(self) -> None:
        import engrava.config as config_module

        keys = {
            "default_fts_weight",
            "default_vector_weight",
            "default_recency_weight",
            "default_priority_weight",
            "recency_half_life",
            "default_graph_weight",
        }
        tree = ast.parse(textwrap.dedent(inspect.getsource(config_module._parse_search)))
        seen: set[str] = set()
        offenders: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in keys
            ):
                seen.add(node.args[0].value)
                if isinstance(node.args[1], ast.Constant):
                    offenders.append(ast.unparse(node))
        assert seen == keys, f"_parse_search no longer reads every weight key: {keys - seen}"
        assert offenders == [], f"_parse_search falls back to a numeric literal: {offenders}"

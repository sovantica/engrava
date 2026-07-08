"""Unit tests for the metadata / visibility filter value objects.

Covers construction-time validation (the "never mid-query" contract) and
the SQL-compilation contract of ``FieldPredicate`` / ``MetadataFilter`` /
``VisibilityQueryFilter`` / ``compile_effective_predicate``.
"""

from __future__ import annotations

import math

import pytest

from engrava.domain.exceptions import InvalidFilterError, InvalidFilterPathError
from engrava.domain.models.filters import (
    MAX_PREDICATE_COUNT,
    FieldOp,
    FieldPredicate,
    MetadataFilter,
    VisibilityQueryFilter,
    compile_effective_predicate,
)

_COL = "t.metadata_json"


class TestFieldPredicateConstruction:
    """Construction-time validation for FieldPredicate (the error contract)."""

    @pytest.mark.parametrize(
        "path",
        ["$", "$.session_id", "$.a.b.c", "$.tags[0]", "$[3]", "$.x[10].y"],
    )
    def test_valid_paths_accepted(self, path: str) -> None:
        """Paths matching the grammar construct without error."""
        assert FieldPredicate(path, FieldOp.EQ, "v").path == path

    @pytest.mark.parametrize(
        "path",
        [
            "session_id",  # no $ root
            "$.",  # trailing dot, empty segment
            "$.bad-key",  # hyphen not allowed
            "$['quoted']",  # quoted segment
            "$.*",  # wildcard
            "$.a..b",  # double dot
            "$ .a",  # space
            "$.a; DROP TABLE thought",  # injection attempt
            "",  # empty
            "$.key\n",  # trailing newline ($ would admit it; fullmatch must not)
            "$.key\nmalicious",  # embedded newline
        ],
    )
    def test_bad_paths_rejected(self, path: str) -> None:
        """Paths violating the grammar raise InvalidFilterPathError."""
        with pytest.raises(InvalidFilterPathError):
            FieldPredicate(path, FieldOp.EQ, "v")

    def test_non_string_path_rejected_as_typed_error(self) -> None:
        """A non-string path raises InvalidFilterPathError, not a bare TypeError."""
        with pytest.raises(InvalidFilterPathError):
            FieldPredicate(123, FieldOp.EQ, "v")  # type: ignore[arg-type]

    def test_unsupported_operator_rejected(self) -> None:
        """A non-FieldOp operator raises InvalidFilterError at construction."""
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", "GT", 1)  # type: ignore[arg-type]

    def test_eq_out_of_range_int_rejected(self) -> None:
        """An int beyond SQLite's signed 64-bit range is rejected."""
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", FieldOp.EQ, 2**63)
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", FieldOp.EQ, -(2**63) - 1)

    def test_eq_int64_boundaries_accepted(self) -> None:
        """The exact signed 64-bit boundaries are accepted."""
        assert FieldPredicate("$.k", FieldOp.EQ, 2**63 - 1).value == 2**63 - 1
        assert FieldPredicate("$.k", FieldOp.EQ, -(2**63)).value == -(2**63)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_eq_non_finite_float_rejected(self, bad: float) -> None:
        """NaN / ±inf floats are rejected at construction."""
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", FieldOp.EQ, bad)

    def test_eq_finite_float_accepted(self) -> None:
        """A finite float is a valid EQ value."""
        assert FieldPredicate("$.k", FieldOp.EQ, 3.14).value == 3.14

    def test_eq_unsupported_type_rejected(self) -> None:
        """A non-scalar EQ value (e.g. a dict) is rejected."""
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", FieldOp.EQ, {"nested": 1})  # type: ignore[arg-type]

    def test_eq_unsupported_non_iterable_type_rejected(self) -> None:
        """A non-iterable scalar of an unsupported type (e.g. complex) is rejected.

        ``complex`` is not iterable, so it bypasses the IN/collection guard and
        reaches the scalar-domain check, exercising the final rejection branch.
        """
        with pytest.raises(InvalidFilterError, match="must be one of"):
            FieldPredicate("$.k", FieldOp.EQ, 1 + 2j)  # type: ignore[arg-type]

    def test_in_element_unsupported_type_rejected(self) -> None:
        """An IN element of an unsupported type is rejected at construction."""
        with pytest.raises(InvalidFilterError, match="must be one of"):
            FieldPredicate("$.k", FieldOp.IN, [1, 2 + 3j])  # type: ignore[list-item]

    def test_eq_collection_rejected(self) -> None:
        """An iterable passed to EQ (which expects a single scalar) is rejected."""
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", FieldOp.EQ, [1, 2, 3])  # type: ignore[arg-type]

    def test_in_non_collection_rejected(self) -> None:
        """A bare scalar passed to IN is rejected."""
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", FieldOp.IN, 5)  # type: ignore[arg-type]

    def test_in_bad_element_rejected(self) -> None:
        """A non-finite float element inside an IN collection is rejected."""
        with pytest.raises(InvalidFilterError):
            FieldPredicate("$.k", FieldOp.IN, [1, math.inf])

    def test_in_normalises_to_tuple(self) -> None:
        """IN collections are normalised to a tuple so the predicate is frozen."""
        pred = FieldPredicate("$.k", FieldOp.IN, ["a", "b"])
        assert pred.value == ("a", "b")
        assert isinstance(pred.value, tuple)

    def test_string_is_not_treated_as_collection(self) -> None:
        """A str EQ value is a scalar, not an iterable of characters."""
        assert FieldPredicate("$.k", FieldOp.EQ, "abc").value == "abc"

    def test_frozen(self) -> None:
        """FieldPredicate is immutable."""
        pred = FieldPredicate("$.k", FieldOp.EQ, "v")
        with pytest.raises(AttributeError):
            pred.path = "$.other"  # type: ignore[misc]


class TestFieldPredicateCompile:
    """SQL-compilation contract for FieldPredicate."""

    def test_eq_scalar(self) -> None:
        """EQ of a scalar compiles to an equality with bound path + value."""
        sql, params = FieldPredicate("$.k", FieldOp.EQ, "v").compile(column=_COL)
        assert sql == "json_extract(t.metadata_json, ?) = ?"
        assert params == ["$.k", "v"]

    def test_eq_none_is_null(self) -> None:
        """EQ None compiles to IS NULL (matches missing path and JSON null)."""
        sql, params = FieldPredicate("$.k", FieldOp.EQ, None).compile(column=_COL)
        assert sql == "json_extract(t.metadata_json, ?) IS NULL"
        assert params == ["$.k"]

    def test_in_binds_collection_as_one_param(self) -> None:
        """IN binds the whole collection as a single JSON-array parameter."""
        sql, params = FieldPredicate("$.k", FieldOp.IN, ["a", "b", "c"]).compile(column=_COL)
        assert "json_each(?)" in sql
        # path + one JSON-array param only — no per-element placeholders.
        assert params == ["$.k", '["a", "b", "c"]']

    def test_in_empty_matches_nothing(self) -> None:
        """An empty IN compiles to a contradiction (matches nothing), not a syntax error."""
        sql, params = FieldPredicate("$.k", FieldOp.IN, []).compile(column=_COL)
        assert sql == "0 = 1"
        assert params == []

    def test_in_with_null_member_folds_into_is_null(self) -> None:
        """A NULL member of IN adds an IS NULL branch."""
        sql, params = FieldPredicate("$.k", FieldOp.IN, ["a", None]).compile(column=_COL)
        assert "IS NULL" in sql
        assert "json_each(?)" in sql
        assert params == ["$.k", '["a"]', "$.k"]

    def test_in_only_null(self) -> None:
        """IN of only None compiles to a single IS NULL branch."""
        sql, params = FieldPredicate("$.k", FieldOp.IN, [None]).compile(column=_COL)
        assert sql == "(json_extract(t.metadata_json, ?) IS NULL)"
        assert params == ["$.k"]


class TestMetadataFilter:
    """MetadataFilter conjunction + empty no-op semantics."""

    def test_empty_is_match_all(self) -> None:
        """An empty MetadataFilter is the match-all no-op."""
        mf = MetadataFilter()
        assert mf.is_empty()
        sql, params = mf.compile(column=_COL)
        assert sql == "1 = 1"
        assert params == []

    def test_conjunction_ands_predicates(self) -> None:
        """Multiple predicates compile to an AND-joined fragment."""
        mf = MetadataFilter(
            [
                FieldPredicate("$.a", FieldOp.EQ, 1),
                FieldPredicate("$.b", FieldOp.EQ, "x"),
            ]
        )
        sql, params = mf.compile(column=_COL)
        assert " AND " in sql
        assert params == ["$.a", 1, "$.b", "x"]

    def test_predicate_count_cap(self) -> None:
        """A predicate count beyond the documented max is rejected at construction."""
        preds = [FieldPredicate(f"$.k{i}", FieldOp.EQ, i) for i in range(MAX_PREDICATE_COUNT + 1)]
        with pytest.raises(InvalidFilterError):
            MetadataFilter(preds)

    def test_predicate_count_at_cap_ok(self) -> None:
        """Exactly the documented max constructs without error."""
        preds = [FieldPredicate(f"$.k{i}", FieldOp.EQ, i) for i in range(MAX_PREDICATE_COUNT)]
        assert len(MetadataFilter(preds).predicates) == MAX_PREDICATE_COUNT

    def test_non_predicate_element_rejected(self) -> None:
        """A non-FieldPredicate element is rejected at construction, not at compile."""
        with pytest.raises(InvalidFilterError, match="FieldPredicate"):
            MetadataFilter(["not a predicate"])  # type: ignore[list-item]

    def test_non_iterable_predicates_rejected_as_typed_error(self) -> None:
        """A non-iterable ``predicates`` raises InvalidFilterError, not TypeError."""
        with pytest.raises(InvalidFilterError):
            MetadataFilter(None)  # type: ignore[arg-type]


class TestVisibilityQueryFilter:
    """VisibilityQueryFilter bounded OR + empty rejection."""

    def test_empty_rejected(self) -> None:
        """The all-empty form is rejected (matches nothing → loud error)."""
        with pytest.raises(InvalidFilterError):
            VisibilityQueryFilter(frozenset(), owner=None)

    def test_bare_string_allowed_rejected(self) -> None:
        """A bare str for ``allowed`` is rejected (would silently iterate chars)."""
        with pytest.raises(InvalidFilterError, match="not a bare string"):
            VisibilityQueryFilter("public")  # type: ignore[arg-type]

    def test_non_string_allowed_member_rejected(self) -> None:
        """A non-string member of ``allowed`` is rejected at construction."""
        with pytest.raises(InvalidFilterError, match="members must be str"):
            VisibilityQueryFilter({"public", 5})  # type: ignore[arg-type]

    def test_non_string_owner_rejected(self) -> None:
        """A non-string ``owner`` is rejected at construction."""
        with pytest.raises(InvalidFilterError, match="owner must be str"):
            VisibilityQueryFilter({"public"}, owner=42)  # type: ignore[arg-type]

    def test_non_iterable_allowed_rejected_as_typed_error(self) -> None:
        """A non-iterable ``allowed`` raises InvalidFilterError, not TypeError."""
        with pytest.raises(InvalidFilterError):
            VisibilityQueryFilter(None)  # type: ignore[arg-type]

    def test_unhashable_allowed_member_rejected_as_typed_error(self) -> None:
        """An unhashable member of ``allowed`` raises InvalidFilterError, not TypeError."""
        with pytest.raises(InvalidFilterError):
            VisibilityQueryFilter([[]])  # type: ignore[list-item]

    def test_allowed_only_omits_owner_branch(self) -> None:
        """With owner=None only the visibility IN branch is emitted."""
        sql, params = VisibilityQueryFilter({"public"}).compile(column=_COL)
        assert "$.owner" not in sql
        assert "json_each(?)" in sql
        assert params == ['["public"]']

    def test_public_or_mine_is_parenthesised(self) -> None:
        """allowed + owner emits a parenthesised OR group (precedence-safe)."""
        sql, params = VisibilityQueryFilter({"public"}, owner="alice").compile(column=_COL)
        assert sql.startswith("(")
        assert sql.endswith(")")
        assert " OR " in sql
        assert "$.owner" in sql
        assert params == ['["public"]', "alice"]

    def test_owner_only(self) -> None:
        """owner with empty allowed emits only the owner branch."""
        sql, params = VisibilityQueryFilter(frozenset(), owner="bob").compile(column=_COL)
        assert "$.visibility" not in sql
        assert "$.owner" in sql
        assert params == ["bob"]

    def test_allowed_is_sorted_for_stability(self) -> None:
        """The bound allowed-set is sorted so the param is deterministic."""
        _, params = VisibilityQueryFilter({"c", "a", "b"}).compile(column=_COL)
        assert params == ['["a", "b", "c"]']


class TestCompileEffectivePredicate:
    """The combined, json_valid-guarded effective predicate."""

    def test_none_none_returns_none(self) -> None:
        """No filter and no visibility => None (the unfiltered query path)."""
        assert compile_effective_predicate(None, None, column=_COL) is None

    def test_empty_filter_returns_none(self) -> None:
        """An empty MetadataFilter with no visibility => None."""
        assert compile_effective_predicate(MetadataFilter(), None, column=_COL) is None

    def test_json_valid_guard_wraps_whole_predicate(self) -> None:
        """The guard wraps the entire predicate (filters AND visibility)."""
        result = compile_effective_predicate(
            MetadataFilter([FieldPredicate("$.a", FieldOp.EQ, 1)]),
            VisibilityQueryFilter({"public"}, owner="alice"),
            column=_COL,
        )
        assert result is not None
        sql, params = result
        assert sql.startswith("CASE WHEN json_valid(t.metadata_json) THEN (")
        assert sql.endswith(") ELSE 0 END")
        # The visibility OR group is inside the guarded predicate.
        assert "$.owner" in sql
        assert " AND " in sql
        assert params == ["$.a", 1, '["public"]', "alice"]

    def test_visibility_only(self) -> None:
        """Visibility alone still produces a guarded predicate."""
        result = compile_effective_predicate(None, VisibilityQueryFilter({"public"}), column=_COL)
        assert result is not None
        sql, _ = result
        assert "json_valid" in sql
        assert "$.visibility" in sql

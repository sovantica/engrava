"""Domain + serde tests for ``ThoughtRecord.metadata``.

Covers:

* default-empty + frozen invariants on the Pydantic model;
* INSERT + UPDATE serde round trips through ``SqliteEngravaCore``;
* ``_validate_metadata`` shape and serialized-size rules
  (incl. the bool/int Python edge case);
* unicode preservation via ``ensure_ascii=False``;
* the F12 contract — UPDATE persists modified metadata instead of
  silently dropping it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import aiosqlite
import pytest
from pydantic import ValidationError

from engrava import SqliteEngravaCore
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import MetadataValue, ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import _validate_metadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_thought(
    thought_id: str = "t-1",
    *,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Build a minimal ThoughtRecord for direct domain-level checks."""
    metadata_value: dict[str, MetadataValue] = dict(metadata) if metadata is not None else {}
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence="essence",
        content="content",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        metadata=metadata_value,
    )


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """In-memory ``SqliteEngravaCore`` with the head schema applied."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    core = SqliteEngravaCore(conn)
    await core.ensure_schema()
    try:
        yield core
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Domain-model invariants
# ---------------------------------------------------------------------------


class TestThoughtMetadataDomain:
    """Pure-Pydantic invariants — no SQLite involved."""

    def test_default_metadata_is_empty_dict(self) -> None:
        thought = _make_thought()
        assert thought.metadata == {}

    def test_frozen_metadata_assignment_raises(self) -> None:
        """Frozen Pydantic model rejects attribute reassignment."""
        thought = _make_thought(metadata={"role": "user"})
        with pytest.raises(ValidationError):
            thought.metadata = {"role": "assistant"}

    def test_evolve_with_new_metadata_returns_new_instance(self) -> None:
        original = _make_thought(metadata={"role": "user", "lang": "en"})
        evolved = original.evolve(metadata={"role": "assistant", "lang": "en"})
        assert evolved.metadata == {"role": "assistant", "lang": "en"}
        assert original.metadata == {"role": "user", "lang": "en"}


# ---------------------------------------------------------------------------
# Persistence round trips
# ---------------------------------------------------------------------------


class TestThoughtMetadataPersistence:
    """End-to-end INSERT/UPDATE/READ round trips."""

    async def test_default_empty_metadata_round_trip(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        thought = _make_thought("t-default")
        await store.create_thought(thought)
        fetched = await store.get_thought("t-default")
        assert fetched is not None
        assert fetched.metadata == {}

    async def test_metadata_round_trip_mixed_types(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        meta: dict[str, MetadataValue] = {
            "role": "user",
            "lang": "en",
            "turn_index": 5,
            "confidence_override": 0.85,
            "is_first": True,
            "speaker": None,
        }
        await store.create_thought(_make_thought("t-mixed", metadata=meta))
        fetched = await store.get_thought("t-mixed")
        assert fetched is not None
        assert fetched.metadata == meta

    async def test_metadata_bool_preserved_distinct_from_int(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """``bool`` ⊂ ``int`` in Python — JSON keeps them distinguishable."""
        meta = {"is_first": True, "is_last": False, "turn_index": 1}
        await store.create_thought(_make_thought("t-bool", metadata=meta))
        fetched = await store.get_thought("t-bool")
        assert fetched is not None
        assert fetched.metadata["is_first"] is True
        assert fetched.metadata["is_last"] is False
        # turn_index round-trips as int (1), not silently coerced to bool.
        assert fetched.metadata["turn_index"] == 1
        assert not isinstance(fetched.metadata["turn_index"], bool)

    async def test_metadata_unicode_preserved(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """``ensure_ascii=False`` preserves non-ASCII attribute values."""
        meta = {"speaker": "Łukasz", "lang": "pl", "city": "Kraków"}
        await store.create_thought(_make_thought("t-unicode", metadata=meta))
        fetched = await store.get_thought("t-unicode")
        assert fetched is not None
        assert fetched.metadata["speaker"] == "Łukasz"
        assert fetched.metadata["city"] == "Kraków"

    async def test_update_persists_modified_metadata(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """F12 contract: UPDATE writes metadata via _CORE_UPDATE_SQL."""
        await store.create_thought(
            _make_thought("t-update", metadata={"role": "user", "lang": "en"}),
        )
        await store.update_thought("t-update", metadata={"role": "assistant"})
        refetched = await store.get_thought("t-update")
        assert refetched is not None
        assert refetched.metadata == {"role": "assistant"}

    async def test_update_unrelated_field_does_not_drop_metadata(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Updating a non-metadata field must not zero metadata."""
        await store.create_thought(
            _make_thought("t-untouched", metadata={"role": "user"}),
        )
        # Update something other than metadata; metadata should survive.
        await store.update_thought("t-untouched", essence="new essence")
        refetched = await store.get_thought("t-untouched")
        assert refetched is not None
        assert refetched.metadata == {"role": "user"}
        assert refetched.essence == "new essence"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateMetadataHelper:
    """Direct unit tests for the ``_validate_metadata`` helper.

    These tests exercise the runtime validator that ``create_thought``
    and ``update_thought`` invoke before persistence.  The helper is
    asserted directly because Pydantic's field type
    (``dict[str, MetadataValue]``) already rejects malformed inputs at
    model-construction time, so the round-trip path through
    ``ThoughtRecord(...)`` would never reach ``_validate_metadata``
    for the negative cases.  Direct calls preserve coverage of the
    validator's per-key error messages.
    """

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ValueError, match="must be dict"):
            _validate_metadata(cast("dict[str, MetadataValue]", "not-a-dict"))

    def test_rejects_non_str_key(self) -> None:
        bad = cast("dict[str, MetadataValue]", {42: "value"})
        with pytest.raises(ValueError, match="key must be str"):
            _validate_metadata(bad)

    def test_rejects_list_value(self) -> None:
        bad = cast("dict[str, MetadataValue]", {"tags": ["a", "b"]})
        with pytest.raises(ValueError, match="not allowed"):
            _validate_metadata(bad)

    @pytest.mark.parametrize("bad_float", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_float(self, bad_float: float) -> None:
        """NaN / ±Infinity are rejected — they serialize to invalid JSON tokens.

        Without this guard the row would serialize to ``{"score": NaN}`` (not
        valid JSON), so SQLite's ``json_valid()`` returns 0 and the row becomes
        silently unmatchable by every metadata filter. A latent defect on the
        thought path that the finite-float guard fixes for both records.
        """
        bad: dict[str, MetadataValue] = {"score": bad_float}
        with pytest.raises(ValueError, match="finite number"):
            _validate_metadata(bad)

    def test_rejects_non_finite_float_nested(self) -> None:
        """The finite-float rule holds at depth, like the list rule."""
        bad: dict[str, MetadataValue] = {"outer": {"inner": float("inf")}}
        with pytest.raises(ValueError, match=r"at outer\.inner must be a finite number"):
            _validate_metadata(bad)

    def test_accepts_finite_float(self) -> None:
        """A finite float passes (the positive control for the guard)."""
        _validate_metadata({"score": 0.75, "neg": -3.5, "big": 1e300})

    def test_size_under_warn_threshold_passes_silently(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """≈3 KiB metadata stays well below the 4 KiB soft warn threshold."""
        meta: dict[str, MetadataValue] = {"big_text": "x" * 3000}
        with caplog.at_level(logging.WARNING):
            _validate_metadata(meta)
        warns = [r for r in caplog.records if "metadata size" in r.getMessage()]
        assert warns == []

    def test_size_above_warn_threshold_emits_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """≈5 KiB metadata trips the 4 KiB soft warning."""
        meta: dict[str, MetadataValue] = {"big_text": "x" * 5000}
        with caplog.at_level(logging.WARNING):
            _validate_metadata(meta)
        warns = [r for r in caplog.records if "metadata size" in r.getMessage()]
        assert len(warns) == 1
        assert "exceeds soft limit" in warns[0].getMessage()

    def test_size_above_reject_threshold_raises(self) -> None:
        """≈70 KiB metadata exceeds the 64 KiB hard limit and is rejected."""
        meta: dict[str, MetadataValue] = {"huge_text": "x" * 70_000}
        with pytest.raises(ValueError, match="exceeds maximum"):
            _validate_metadata(meta)


class TestValidateMetadataAtAPIBoundaries:
    """Integration: the API entries call ``_validate_metadata``."""

    async def test_create_thought_runs_size_check(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Persisted size limits fire from the create_thought API surface."""
        meta: dict[str, MetadataValue] = {"huge_text": "x" * 70_000}
        bad = _make_thought("t-too-big", metadata=meta)
        with pytest.raises(ValueError, match="exceeds maximum"):
            await store.create_thought(bad)

    async def test_update_thought_runs_size_check(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Validation parity: update_thought enforces the same size limits."""
        await store.create_thought(
            _make_thought("t-update-size", metadata={"role": "user"}),
        )
        bad: dict[str, MetadataValue] = {"huge_text": "x" * 70_000}
        with pytest.raises(ValueError, match="exceeds maximum"):
            await store.update_thought("t-update-size", metadata=bad)

    async def test_create_thought_rejects_non_finite_float(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """The finite-float guard fires from the create_thought API surface."""
        bad = _make_thought("t-nan", metadata={"score": float("nan")})
        with pytest.raises(ValueError, match="finite number"):
            await store.create_thought(bad)


# ---------------------------------------------------------------------------
# nested dict support
# ---------------------------------------------------------------------------


class TestThoughtMetadataNested:
    """Nested structured namespaces (``ThoughtSource``)."""

    async def test_nested_source_roundtrip(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """``ThoughtSource`` nested structure survives write -> read."""
        meta: dict[str, MetadataValue] = {
            "perspective": "utterance",
            "source": {
                "is_self": False,
                "id": "user_123",
                "label": "Alice",
                "confidence": "high",
                "role_hint": "user",
            },
            "session_id": "chat_abc",
        }
        thought = _make_thought("t-nested-source", metadata=meta)
        await store.create_thought(thought)
        fetched = await store.get_thought("t-nested-source")
        assert fetched is not None
        assert fetched.metadata == meta
        source = cast("dict[str, MetadataValue]", fetched.metadata["source"])
        assert source["is_self"] is False
        assert source["confidence"] == "high"

    def test_nested_validation_rejects_list_at_depth(self) -> None:
        """List inside a nested dict is rejected with a key-path message."""
        bad = cast("dict[str, MetadataValue]", {"source": {"tags": ["a", "b"]}})
        with pytest.raises(ValueError, match=r"at source\.tags type list not allowed"):
            _validate_metadata(bad)

    def test_nested_validation_rejects_non_str_nested_key(self) -> None:
        """Non-string key inside a nested dict is rejected."""
        bad = cast("dict[str, MetadataValue]", {"source": {42: "value"}})
        with pytest.raises(ValueError, match=r"at source must be str"):
            _validate_metadata(bad)

    def test_nested_validation_rejects_deeply_nested_list(self) -> None:
        """Recursive validator walks past depth 1 — list at depth 2 still rejected."""
        bad = cast(
            "dict[str, MetadataValue]",
            {"outer": {"inner": {"tags": ["a", "b"]}}},
        )
        with pytest.raises(
            ValueError,
            match=r"at outer\.inner\.tags type list not allowed",
        ):
            _validate_metadata(bad)

    def test_nested_size_limit_enforced(self) -> None:
        """Nested metadata still subject to the 64 KiB hard rejection threshold."""
        huge = "x" * 70_000
        bad: dict[str, MetadataValue] = {"source": {"label": huge}}
        with pytest.raises(ValueError, match="exceeds maximum"):
            _validate_metadata(bad)

    async def test_nested_evolve_roundtrip(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """``evolve(metadata=nested)`` preserves the nested structure on UPDATE."""
        original = _make_thought(
            "t-evolve-nested",
            metadata={"source": {"is_self": False}},
        )
        await store.create_thought(original)
        evolved = original.evolve(
            metadata={"source": {"is_self": False, "confidence": "high"}},
        )
        await store.update_thought(
            "t-evolve-nested",
            metadata=evolved.metadata,
        )
        refreshed = await store.get_thought("t-evolve-nested")
        assert refreshed is not None
        assert refreshed.metadata == {
            "source": {"is_self": False, "confidence": "high"},
        }

    async def test_json1_nested_extract_query(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """SQLite JSON1 ``json_extract`` supports the nested ``$.source.is_self`` path."""
        await store.create_thought(
            _make_thought("t-self", metadata={"source": {"is_self": True}}),
        )
        await store.create_thought(
            _make_thought("t-other", metadata={"source": {"is_self": False}}),
        )
        cursor = await store._db.execute(
            "SELECT thought_id FROM thought "
            "WHERE json_extract(metadata_json, '$.source.is_self') = 1",
        )
        rows = await cursor.fetchall()
        thought_ids = {row["thought_id"] for row in rows}
        assert thought_ids == {"t-self"}

    async def test_thought_source_full_schema(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """``ThoughtSource`` with all five fields round-trips."""
        meta: dict[str, MetadataValue] = {
            "perspective": "thought",
            "source": {
                "is_self": True,
                "id": None,
                "label": None,
                "confidence": "medium",
                "role_hint": None,
            },
        }
        thought = _make_thought(
            "t-thought-source-full",
            metadata=meta,
        )
        await store.create_thought(thought)
        fetched = await store.get_thought("t-thought-source-full")
        assert fetched is not None
        assert fetched.metadata == meta

    async def test_backward_compat_flat_only_caller_unchanged(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        """Legacy flat-only callers see no behaviour change."""
        meta: dict[str, MetadataValue] = {
            "role": "user",
            "lang": "en",
            "turn_index": 5,
        }
        thought = _make_thought("t-flat-compat", metadata=meta)
        await store.create_thought(thought)
        fetched = await store.get_thought("t-flat-compat")
        assert fetched is not None
        assert fetched.metadata == meta

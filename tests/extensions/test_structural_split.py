"""Tests for the ``StructuralSplitProducer`` zero-dependency split modes.

Two layers:

* **Unit** — drive :meth:`StructuralSplitProducer.derive_records` directly (no
  store) to pin the segmentation of each mode: the byte-identical paragraph
  default (a golden), fixed-window tiling / overlap / spans, the ``min_chars``
  gate, the child-metadata contract, and constructor validation.
* **Integration** — drive the producer through the real ``SqliteEngravaCore``
  seam, both on-store and via the explicit ``derive_existing`` backfill, to prove
  a non-LLM ``fixed_window`` split persists linked children and converges across
  the two trigger paths.

Every mode here is deterministic and dependency-free — no model, no network.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CoreThoughtRecord,
    DeriveContext,
    DeriveGates,
    DeriveResult,
    EdgeType,
    LifecycleStatus,
    Priority,
    SplitMode,
    SqliteEngravaCore,
    StructuralSplitProducer,
    ThoughtType,
)
from engrava.infrastructure.sqlite.engrava_core import _derived_thought_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from engrava.domain.protocols.derived_records import DerivedRecord

# The shipped paragraph boundary, re-declared here independently so the golden
# proves byte-identity against the *reference* algorithm, not the refactor.
_REFERENCE_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thought(content: str, *, thought_id: str = "src-1") -> CoreThoughtRecord:
    """A minimal committed source thought carrying ``content``."""
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.NOTE,
        essence="source essence",
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.CREATED,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
    )


def _ctx(thought_id: str = "src-1") -> DeriveContext:
    """A stable derivation context (the producer ignores it)."""
    return DeriveContext(
        source_thought_id=thought_id,
        source_content_hash="deadbeef",
        cycle_at_derivation=0,
        origin="create_thought",
    )


async def _derive(producer: StructuralSplitProducer, content: str) -> Sequence[DerivedRecord]:
    """Run the producer over ``content`` and return the derived records."""
    return await producer.derive_records(_thought(content), _ctx())


def _reference_paragraphs(content: str) -> list[str]:
    """Today's shipped paragraph algorithm — the golden reference."""
    pieces = [piece.strip() for piece in _REFERENCE_PARAGRAPH_BOUNDARY.split(content)]
    return [piece for piece in pieces if piece]


def _assert_span_reassembles(content: str, records: Sequence[DerivedRecord]) -> None:
    """Every child's recorded span must slice back to its own content."""
    for record in records:
        start = record.metadata["char_start"]
        end = record.metadata["char_end"]
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert content[start:end] == record.content


# ---------------------------------------------------------------------------
# SplitMode enum + construction defaults
# ---------------------------------------------------------------------------


def test_split_mode_values_are_stable() -> None:
    """The public enum values are the documented, JSON-safe strings."""
    assert SplitMode.PARAGRAPH.value == "paragraph"
    assert SplitMode.FIXED_WINDOW.value == "fixed_window"
    assert SplitMode("fixed_window") is SplitMode.FIXED_WINDOW


def test_default_construction_is_paragraph() -> None:
    """A default producer splits on paragraphs, unchanged from the shipped one."""
    producer = StructuralSplitProducer()
    assert producer.split_mode is SplitMode.PARAGRAPH
    assert producer.thought_type is ThoughtType.OBSERVATION
    assert producer.priority is Priority.P3
    assert producer.attach_edges is True
    assert producer.min_chars == 0


# ---------------------------------------------------------------------------
# Paragraph — byte-identical default (golden) + additive metadata
# ---------------------------------------------------------------------------


async def test_paragraph_default_matches_shipped_algorithm_golden() -> None:
    """GOLDEN: the default mode's content + child shape equal today's output."""
    content = "First paragraph.\n\nSecond paragraph.\n\n  Third with pad.  "
    records = await _derive(StructuralSplitProducer(), content)

    # Content + order are byte-identical to the shipped paragraph algorithm.
    assert [r.content for r in records] == _reference_paragraphs(content)
    assert [r.content for r in records] == [
        "First paragraph.",
        "Second paragraph.",
        "Third with pad.",
    ]
    # Child shape is unchanged: OBSERVATION / P3 / provenance edge on.
    for index, record in enumerate(records):
        assert record.thought_type is ThoughtType.OBSERVATION
        assert record.priority is Priority.P3
        assert record.attach_provenance_edge is True
        # The original segment_index key is preserved (additive metadata).
        assert record.metadata["segment_index"] == index
        assert record.metadata["split_mode"] == "paragraph"
    _assert_span_reassembles(content, records)


async def test_paragraph_metadata_is_additive_superset() -> None:
    """Paragraph metadata keeps segment_index and adds mode + span, nothing else."""
    records = await _derive(StructuralSplitProducer(), "Alpha.\n\nBeta.")
    assert set(records[0].metadata) == {
        "split_mode",
        "segment_index",
        "char_start",
        "char_end",
    }


async def test_paragraph_drops_blank_pieces_around_boundaries() -> None:
    """Leading / trailing blank pieces are dropped, matching the shipped filter."""
    content = "\n\nAlpha.\n\nBeta.\n\n"
    records = await _derive(StructuralSplitProducer(), content)
    assert [r.content for r in records] == _reference_paragraphs(content)
    assert [r.content for r in records] == ["Alpha.", "Beta."]
    _assert_span_reassembles(content, records)


async def test_paragraph_single_segment_derives_nothing() -> None:
    """A single-segment source derives nothing (byte-identical to shipped)."""
    records = await _derive(StructuralSplitProducer(), "Just one paragraph, nothing to split.")
    assert list(records) == []


async def test_paragraph_custom_boundary_escape_hatch_still_works() -> None:
    """The ``boundary`` regex escape hatch still applies in paragraph mode."""
    producer = StructuralSplitProducer(boundary=re.compile(r"\|"))
    records = await _derive(producer, "a|b|c")
    assert [r.content for r in records] == ["a", "b", "c"]
    _assert_span_reassembles("a|b|c", records)


async def test_paragraph_capturing_boundary_preserves_re_split_semantics() -> None:
    """A custom boundary with a capturing group keeps ``re.split`` semantics.

    ``re.split`` keeps captured delimiters as segments, and the shipped producer
    split on ``boundary.split``; this preserves that byte-identically — a captured
    ``|`` stays a segment instead of being dropped (the interim finditer refactor
    had dropped it, a delimiter-dropping regression).
    """
    producer = StructuralSplitProducer(boundary=re.compile(r"(\|)"))
    content = "a|b|c"
    records = await _derive(producer, content)
    assert [r.content for r in records] == ["a", "|", "b", "|", "c"]
    _assert_span_reassembles(content, records)


async def test_paragraph_offsets_are_structural_not_a_content_search() -> None:
    """Spans come from the match positions, not a global content search.

    A segment whose text also occurs *inside* the delimiter must still be located
    at its true position. Here the 2nd ``"b"`` is at offset 3 (after the ``"Xb"``
    delimiter), not offset 2 (the ``"b"`` inside the delimiter) — a content-search
    heuristic would drift to 2.
    """
    producer = StructuralSplitProducer(boundary=re.compile(r"Xb"))
    content = "aXbb"  # split on "Xb" -> ["a", "b"]
    records = await _derive(producer, content)
    assert [r.content for r in records] == ["a", "b"]
    assert (records[1].metadata["char_start"], records[1].metadata["char_end"]) == (3, 4)
    _assert_span_reassembles(content, records)


def test_invalid_split_mode_raises() -> None:
    """An unknown ``split_mode`` is rejected at construction with a clean ValueError."""
    with pytest.raises(ValueError, match="split_mode must be one of"):
        StructuralSplitProducer(split_mode="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fixed_window — character unit
# ---------------------------------------------------------------------------


async def test_fixed_window_char_no_overlap_tiles_exactly() -> None:
    """Non-overlapping char windows tile the content with no gap or duplication."""
    content = "abcdefghij"  # 10 chars
    producer = StructuralSplitProducer(split_mode=SplitMode.FIXED_WINDOW, window_size=4)
    records = await _derive(producer, content)

    assert [r.content for r in records] == ["abcd", "efgh", "ij"]
    # Perfect tiling: concatenating the windows reconstructs the original.
    assert "".join(r.content for r in records) == content
    for record in records:
        assert record.metadata["split_mode"] == "fixed_window"
    _assert_span_reassembles(content, records)


async def test_fixed_window_char_overlap_shares_exactly_overlap_chars() -> None:
    """Overlapping char windows share exactly ``window_overlap`` chars, lossless."""
    content = "abcdefghij"  # 10 chars
    producer = StructuralSplitProducer(
        split_mode=SplitMode.FIXED_WINDOW,
        window_size=4,
        window_overlap=2,
    )
    records = await _derive(producer, content)

    assert [r.content for r in records] == ["abcd", "cdef", "efgh", "ghij"]
    # Consecutive windows share exactly the overlap; the trailing (size-overlap)
    # slice of each window is fresh content, so reassembly loses nothing.
    reassembled = records[0].content + "".join(r.content[2:] for r in records[1:])
    assert reassembled == content
    # Each neighbour pair shares exactly two characters.
    for earlier, later in pairwise(records):
        assert earlier.content[-2:] == later.content[:2]
    _assert_span_reassembles(content, records)


async def test_fixed_window_char_final_window_may_be_short_never_dropped() -> None:
    """The final char window is kept even when shorter than ``window_size``."""
    content = "abcdefg"  # 7 chars
    producer = StructuralSplitProducer(split_mode=SplitMode.FIXED_WINDOW, window_size=3)
    records = await _derive(producer, content)
    assert [r.content for r in records] == ["abc", "def", "g"]


async def test_fixed_window_content_shorter_than_window_derives_nothing() -> None:
    """Content fitting in one window yields a single segment ⇒ derives nothing."""
    producer = StructuralSplitProducer(split_mode=SplitMode.FIXED_WINDOW, window_size=100)
    assert list(await _derive(producer, "short")) == []


def test_fixed_window_empty_content_segments_to_nothing() -> None:
    """The char-window helper yields no segments for empty content (defensive).

    A stored thought can never carry empty content (the model rejects it), so this
    guards the segmentation helper's empty branch directly.
    """
    producer = StructuralSplitProducer(split_mode=SplitMode.FIXED_WINDOW, window_size=4)
    assert producer._fixed_window_char_segments("") == []


# ---------------------------------------------------------------------------
# fixed_window — word unit
# ---------------------------------------------------------------------------


async def test_fixed_window_word_no_overlap_groups_words() -> None:
    """Word windows group ``window_size`` whitespace-delimited words each."""
    content = "one two three four five"
    producer = StructuralSplitProducer(
        split_mode=SplitMode.FIXED_WINDOW,
        window_unit="word",
        window_size=2,
    )
    records = await _derive(producer, content)
    assert [r.content for r in records] == ["one two", "three four", "five"]
    _assert_span_reassembles(content, records)


async def test_fixed_window_word_overlap_shares_words() -> None:
    """Overlapping word windows share exactly ``window_overlap`` words."""
    content = "one two three four five"
    producer = StructuralSplitProducer(
        split_mode=SplitMode.FIXED_WINDOW,
        window_unit="word",
        window_size=2,
        window_overlap=1,
    )
    records = await _derive(producer, content)
    assert [r.content for r in records] == [
        "one two",
        "two three",
        "three four",
        "four five",
    ]
    _assert_span_reassembles(content, records)


async def test_fixed_window_word_ignores_surrounding_whitespace() -> None:
    """Word spans exclude surrounding whitespace but keep internal whitespace."""
    content = "  alpha   beta  gamma  "
    producer = StructuralSplitProducer(
        split_mode=SplitMode.FIXED_WINDOW,
        window_unit="word",
        window_size=2,
    )
    records = await _derive(producer, content)
    assert [r.content for r in records] == ["alpha   beta", "gamma"]
    _assert_span_reassembles(content, records)


async def test_fixed_window_word_whitespace_only_content_derives_nothing() -> None:
    """Whitespace-only content has no words ⇒ derives nothing."""
    producer = StructuralSplitProducer(
        split_mode=SplitMode.FIXED_WINDOW,
        window_unit="word",
        window_size=2,
    )
    assert list(await _derive(producer, "   \n\t  ")) == []


# ---------------------------------------------------------------------------
# min_chars gate
# ---------------------------------------------------------------------------


async def test_min_chars_blocks_short_source() -> None:
    """A source whose stripped length is below ``min_chars`` derives nothing."""
    producer = StructuralSplitProducer(min_chars=100)
    assert list(await _derive(producer, "Alpha.\n\nBeta.")) == []


async def test_min_chars_allows_long_enough_source() -> None:
    """A source at or above ``min_chars`` splits normally."""
    producer = StructuralSplitProducer(min_chars=5)
    records = await _derive(producer, "Alpha.\n\nBeta.")
    assert [r.content for r in records] == ["Alpha.", "Beta."]


async def test_min_chars_zero_is_no_gate() -> None:
    """The default ``min_chars=0`` never gates (byte-identical default)."""
    producer = StructuralSplitProducer(min_chars=0)
    records = await _derive(producer, "A.\n\nB.")
    assert [r.content for r in records] == ["A.", "B."]


# ---------------------------------------------------------------------------
# child type / priority remain configurable
# ---------------------------------------------------------------------------


async def test_child_type_and_priority_are_configurable() -> None:
    """``thought_type`` / ``priority`` still flow onto every derived child."""
    producer = StructuralSplitProducer(
        thought_type=ThoughtType.BELIEF,
        priority=Priority.P1,
    )
    records = await _derive(producer, "A.\n\nB.")
    assert all(r.thought_type is ThoughtType.BELIEF for r in records)
    assert all(r.priority is Priority.P1 for r in records)


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------


def test_window_size_below_one_raises() -> None:
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        StructuralSplitProducer(window_size=0)


def test_negative_window_overlap_raises() -> None:
    with pytest.raises(ValueError, match="window_overlap must be >= 0"):
        StructuralSplitProducer(window_overlap=-1)


def test_overlap_not_less_than_size_raises() -> None:
    with pytest.raises(ValueError, match="window_overlap must be < window_size"):
        StructuralSplitProducer(window_size=4, window_overlap=4)


def test_negative_min_chars_raises() -> None:
    with pytest.raises(ValueError, match="min_chars must be >= 0"):
        StructuralSplitProducer(min_chars=-1)


def test_invalid_window_unit_raises() -> None:
    with pytest.raises(ValueError, match="window_unit must be"):
        StructuralSplitProducer(window_unit="bytes")  # type: ignore[arg-type]


# ===========================================================================
# Integration — the seam (on-store) + the explicit derive_existing backfill
# ===========================================================================


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """Fresh in-memory SQLite with the core schema bootstrapped."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    boot = SqliteEngravaCore(conn)
    await boot.ensure_schema()
    yield conn
    await conn.close()


async def _count(db: aiosqlite.Connection, sql: str, *params: object) -> int:
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


# The fixed-window demo content: 26 letters, size-10 windows ⇒ three children.
_DEMO_CONTENT = "abcdefghijklmnopqrstuvwxyz"
_DEMO_SEGMENTS = ("abcdefghij", "klmnopqrst", "uvwxyz")


def _fixed_window_producer() -> StructuralSplitProducer:
    return StructuralSplitProducer(split_mode=SplitMode.FIXED_WINDOW, window_size=10)


async def test_fixed_window_demo_through_on_store_and_backfill(
    db: aiosqlite.Connection,
) -> None:
    """A non-LLM ``fixed_window`` split persists via BOTH seam trigger paths.

    (1) On-store: an enabled producer derives one linked child per window when a
    source is created. (2) Backfill: the same producer, seam disabled, derives
    the identical children on an explicit ``derive_existing`` call. Both paths
    converge on the same content-addressed child rows.
    """
    # (1) On-store trigger.
    on_store = SqliteEngravaCore(
        db,
        hooks=_fixed_window_producer(),
        derive_gates=DeriveGates(enabled=True),
    )
    await on_store.create_thought(_thought(_DEMO_CONTENT))

    assert await _count(db, "SELECT COUNT(*) FROM thought") == 1 + len(_DEMO_SEGMENTS)
    edges = await on_store.get_edges("src-1", direction="IN")
    assert len(edges) == len(_DEMO_SEGMENTS)
    assert all(e.edge_type == EdgeType.DERIVED_FROM for e in edges)
    # Every window landed as its own content-addressed child.
    for segment in _DEMO_SEGMENTS:
        assert await on_store.get_thought(_derived_thought_id(segment)) is not None

    # (2) Explicit backfill trigger on a separate, seam-disabled store.
    conn = await aiosqlite.connect(":memory:")
    try:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        backfill = SqliteEngravaCore(
            conn,
            hooks=_fixed_window_producer(),
            derive_gates=DeriveGates(enabled=False),
        )
        await backfill.ensure_schema()
        await backfill.create_thought(_thought(_DEMO_CONTENT))
        # enabled=False ⇒ nothing derived on store.
        assert await _count(conn, "SELECT COUNT(*) FROM thought") == 1

        result = await backfill.derive_existing("src-1")
        assert result == DeriveResult(
            thought_id="src-1",
            created=len(_DEMO_SEGMENTS),
            reused=0,
            skipped=0,
        )
        # Same content-addressed children as the on-store path (convergence).
        for segment in _DEMO_SEGMENTS:
            assert await backfill.get_thought(_derived_thought_id(segment)) is not None
        assert len(await backfill.get_edges("src-1", direction="IN")) == len(_DEMO_SEGMENTS)
    finally:
        await conn.close()


async def test_fixed_window_children_carry_window_metadata_when_persisted(
    db: aiosqlite.Connection,
) -> None:
    """Persisted children carry the mode + source span in their metadata."""
    store = SqliteEngravaCore(
        db,
        hooks=_fixed_window_producer(),
        derive_gates=DeriveGates(enabled=True),
    )
    await store.create_thought(_thought(_DEMO_CONTENT))

    child = await store.get_thought(_derived_thought_id(_DEMO_SEGMENTS[0]))
    assert child is not None
    assert child.metadata["split_mode"] == "fixed_window"
    assert child.metadata["segment_index"] == 0
    assert child.metadata["char_start"] == 0
    assert child.metadata["char_end"] == len(_DEMO_SEGMENTS[0])
    # The recorded span slices the source back to the child content.
    assert _DEMO_CONTENT[0 : len(_DEMO_SEGMENTS[0])] == _DEMO_SEGMENTS[0]


# ---------------------------------------------------------------------------
# The window numbers the producer keeps are the numbers it checked
# ---------------------------------------------------------------------------


class _NumberThatIsNeverTooSmall(int):
    """An integer that answers ``<`` for itself, so no lower bound holds."""

    __slots__ = ()

    def __lt__(self, other: object) -> bool:
        del other
        return False


class _UnitThatComparesAsAnother(str):
    """Reads as its real text; compares equal to ``word``."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash("word")

    def __eq__(self, other: object) -> bool:
        return other == "word"

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class TestConstructorKeepsTheValuesItValidated:
    """This producer is a public entry point and validates its own inputs.

    It sits outside ``engrava.config``, so nothing else validates for it. The
    numbers it stores drive the window arithmetic over caller content, and the
    unit selects which arithmetic runs — both compared with methods the value
    itself may define.
    """

    def test_a_lower_bound_cannot_be_answered_by_the_value(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            StructuralSplitProducer(window_size=_NumberThatIsNeverTooSmall(0))

    def test_a_negative_overlap_cannot_be_answered_by_the_value(self) -> None:
        with pytest.raises(ValueError, match="window_overlap must be >= 0"):
            StructuralSplitProducer(window_overlap=_NumberThatIsNeverTooSmall(-1))

    def test_a_negative_minimum_cannot_be_answered_by_the_value(self) -> None:
        with pytest.raises(ValueError, match="min_chars must be >= 0"):
            StructuralSplitProducer(min_chars=_NumberThatIsNeverTooSmall(-1))

    def test_the_stored_numbers_are_exact_ints(self) -> None:
        producer = StructuralSplitProducer(
            window_size=_NumberThatIsNeverTooSmall(10),
            window_overlap=_NumberThatIsNeverTooSmall(2),
            min_chars=_NumberThatIsNeverTooSmall(1),
        )
        assert type(producer.window_size) is int
        assert type(producer.window_overlap) is int
        assert type(producer.min_chars) is int

    def test_the_stored_unit_is_an_exact_str(self) -> None:
        producer = StructuralSplitProducer(window_unit=_UnitThatComparesAsAnother("char"))
        assert type(producer.window_unit) is str
        assert producer.window_unit == "char"

    def test_a_non_integer_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            StructuralSplitProducer(window_size="10")  # type: ignore[arg-type]  # the rejection is the behaviour under test

    def test_legitimate_values_are_unchanged(self) -> None:
        producer = StructuralSplitProducer(window_size=42, window_overlap=7, min_chars=3)
        assert (producer.window_size, producer.window_overlap, producer.min_chars) == (42, 7, 3)
        assert producer.window_unit == "char"

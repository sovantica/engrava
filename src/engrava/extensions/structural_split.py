"""Structural-split producer — a deterministic, dependency-free seam consumer.

A reference consumer of the derived-records extension seam
(:class:`~engrava.domain.protocols.derived_records.DerivedRecordProducerProtocol`)
that runs purely on the stored text — no model, no network, no external service:
it splits a stored thought's ``content`` into structural segments and derives one
child thought per segment, each linked back to the source with a
``DERIVED_FROM`` edge.

Two zero-dependency split modes ship (:class:`SplitMode`):

* :attr:`SplitMode.PARAGRAPH` (the default) — split on a configurable blank-line
  boundary. This is the byte-identical original behaviour.
* :attr:`SplitMode.FIXED_WINDOW` — fixed-size windows measured in characters or
  words, with an optional overlap. Bounds chunk size for embedding robustness on
  long content, with no dependence on natural boundaries.

Every mode is deterministic, pure-text, and adds no third-party dependency (regex
and character/word counting only): a model-tokenizer window is deliberately
excluded because it would couple this producer to an embedding model — that is
the model-coupled lane, not this producer's.

It doubles as the canonical example of *how to implement the seam*: subclass the
default hooks (so the object is a valid
:class:`~engrava.domain.protocols.hooks.EngravaHooksProtocol` implementation)
and add :meth:`derive_records`. Because the output is a pure function of the
source content, re-running derivation is idempotent.
"""

from __future__ import annotations

import re
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from engrava.config_validation import own_int, own_str
from engrava.domain.enums import Priority, ThoughtType
from engrava.domain.protocols.derived_records import DerivedRecord
from engrava.domain.protocols.hooks import DefaultEngravaHooks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from engrava.domain.models.thought import ThoughtRecord
    from engrava.domain.protocols.derived_records import DeriveContext

#: Splits content on one-or-more blank lines (a run of newlines separated only
#: by horizontal whitespace), the conventional paragraph boundary.
_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n+")

#: Matches a run of non-whitespace characters — a "word" for word-unit windows.
_WORD = re.compile(r"\S+")

#: Content must yield at least this many segments before anything is derived; a
#: single-segment source has nothing structural to extract and derives nothing.
_MIN_SEGMENTS_TO_SPLIT = 2

#: The unit a fixed window's size and overlap are measured in.
WindowUnit = Literal["char", "word"]


@unique
class SplitMode(StrEnum):
    """The strategy used to split source content into structural segments.

    Every mode is deterministic and dependency-free (regex plus character/word
    counting); none couples the producer to a model or the network.

    Examples:
        >>> SplitMode.FIXED_WINDOW
        <SplitMode.FIXED_WINDOW: 'fixed_window'>
        >>> SplitMode("paragraph") is SplitMode.PARAGRAPH
        True

    """

    #: Split on a blank-line (paragraph) boundary — the byte-identical default.
    PARAGRAPH = "paragraph"
    #: Split into fixed-size windows (characters or words) with optional overlap.
    FIXED_WINDOW = "fixed_window"


class _Segment(NamedTuple):
    """One structural segment plus its source span.

    The span is a half-open offset range into the *original* source content, so
    ``content[char_start:char_end] == segment_content`` holds for every segment
    (provenance + reassembly). ``content`` is never empty.
    """

    content: str
    char_start: int
    char_end: int


def _own_bounded_int(value: int, minimum: int, message: str) -> int:
    """Return *value* as an exact ``int`` at or above *minimum*, else raise.

    The order is the point: the bound is checked on the owned copy, because
    ``<`` is a method on the value being compared and an ``int`` subclass may
    answer it for itself.

    Args:
        value: Candidate number.
        minimum: Inclusive lower bound.
        message: The rejection message, owned by this module.

    Returns:
        The validated number as an exact ``int``.

    Raises:
        ValueError: When *value* is not an integer, or falls below *minimum*.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(message)  # noqa: TRY004 -- this constructor's documented rejection type
    owned = own_int(value)
    if owned < minimum:
        raise ValueError(message)
    return owned


class StructuralSplitProducer(DefaultEngravaHooks):
    """Derive one child thought per structural segment of the source content.

    Deterministic and dependency-free. The source content is split according to
    :attr:`split_mode` and each segment becomes a :class:`DerivedRecord` whose
    ``metadata`` records the mode and the segment's source span. Splitting runs
    only when the content yields at least two segments, so a single-segment
    thought derives nothing (there is nothing structural to extract).

    Args:
        thought_type: Classification assigned to every derived child. Defaults
            to :attr:`~engrava.domain.enums.ThoughtType.OBSERVATION`.
        priority: Priority assigned to every derived child. Defaults to
            :attr:`~engrava.domain.enums.Priority.P3`.
        split_mode: The segmentation strategy. Defaults to
            :attr:`SplitMode.PARAGRAPH` (the byte-identical original behaviour).
        window_size: For :attr:`SplitMode.FIXED_WINDOW`, the window length in
            ``window_unit`` units (characters or words). Must be ``>= 1``.
            Ignored for :attr:`SplitMode.PARAGRAPH`. Defaults to ``1000``.
        window_unit: For :attr:`SplitMode.FIXED_WINDOW`, whether
            ``window_size`` / ``window_overlap`` count characters (``"char"``)
            or whitespace-delimited words (``"word"``). Defaults to ``"char"``.
        window_overlap: For :attr:`SplitMode.FIXED_WINDOW`, how many
            ``window_unit`` units consecutive windows share. Must satisfy
            ``0 <= window_overlap < window_size`` (windows must advance).
            Defaults to ``0`` (no overlap).
        min_chars: Minimum source length (measured on the stripped content) to
            split at all. A source shorter than this derives nothing. Must be
            ``>= 0``. Defaults to ``0`` (no minimum).
        boundary: For :attr:`SplitMode.PARAGRAPH`, the compiled regular
            expression marking a segment boundary. Ignored by other modes.
            Defaults to a blank-line (paragraph) boundary.
        attach_edges: Whether each child links back to the source with a
            ``DERIVED_FROM`` edge. Defaults to ``True``.

    Raises:
        ValueError: When ``window_size < 1``, ``window_overlap`` is negative or
            not strictly less than ``window_size``, ``min_chars`` is negative, or
            ``window_unit`` is neither ``"char"`` nor ``"word"``.

    Examples:
        >>> producer = StructuralSplitProducer()
        >>> producer.thought_type
        <ThoughtType.OBSERVATION: 'OBSERVATION'>
        >>> producer.split_mode
        <SplitMode.PARAGRAPH: 'paragraph'>
        >>> windowed = StructuralSplitProducer(
        ...     split_mode=SplitMode.FIXED_WINDOW,
        ...     window_size=200,
        ...     window_overlap=20,
        ... )
        >>> windowed.window_unit
        'char'

    """

    def __init__(
        self,
        *,
        thought_type: ThoughtType = ThoughtType.OBSERVATION,
        priority: Priority = Priority.P3,
        split_mode: SplitMode = SplitMode.PARAGRAPH,
        window_size: int = 1000,
        window_unit: WindowUnit = "char",
        window_overlap: int = 0,
        min_chars: int = 0,
        boundary: re.Pattern[str] = _PARAGRAPH_BOUNDARY,
        attach_edges: bool = True,
    ) -> None:
        # Own before comparing, and keep what was owned. These numbers drive the
        # window arithmetic over caller content and the unit selects which
        # arithmetic runs, so a subclass answering ``<`` for itself would clear
        # a bound the retained value does not satisfy - and the retained value
        # is what every later window is computed from. This producer is a public
        # entry point outside the configuration module, so nothing else does
        # this for it.
        window_size = _own_bounded_int(window_size, 1, "window_size must be >= 1")
        window_overlap = _own_bounded_int(window_overlap, 0, "window_overlap must be >= 0")
        min_chars = _own_bounded_int(min_chars, 0, "min_chars must be >= 0")
        if window_overlap >= window_size:
            msg = "window_overlap must be < window_size (windows must advance)"
            raise ValueError(msg)
        unit_msg = "window_unit must be 'char' or 'word'"
        if not isinstance(window_unit, str):
            raise ValueError(unit_msg)  # noqa: TRY004 -- this constructor's documented rejection type
        window_unit = cast("WindowUnit", own_str(window_unit))
        if window_unit not in ("char", "word"):
            raise ValueError(unit_msg)
        try:
            split_mode = SplitMode(split_mode)
        except ValueError as exc:
            valid = ", ".join(m.value for m in SplitMode)
            msg = f"split_mode must be one of: {valid}"
            raise ValueError(msg) from exc
        self.thought_type = thought_type
        self.priority = priority
        self.split_mode = split_mode
        self.window_size = window_size
        self.window_unit: WindowUnit = window_unit
        self.window_overlap = window_overlap
        self.min_chars = min_chars
        self.boundary = boundary
        self.attach_edges = attach_edges

    async def derive_records(
        self,
        thought: ThoughtRecord,
        ctx: DeriveContext,  # noqa: ARG002 -- seam contract; this producer needs no context
    ) -> Sequence[DerivedRecord]:
        """Split the source content into segments and derive one child each.

        Args:
            thought: The committed source thought.
            ctx: Stable derivation context (unused by this producer).

        Returns:
            One :class:`DerivedRecord` per structural segment, in document order,
            or an empty sequence when the content is shorter than ``min_chars``
            or yields fewer than two segments.

        """
        content = thought.content
        if self.min_chars > 0 and len(content.strip()) < self.min_chars:
            return []
        segments = self._segment(content)
        if len(segments) < _MIN_SEGMENTS_TO_SPLIT:
            return []
        mode = self.split_mode.value
        return [
            DerivedRecord(
                content=segment.content,
                thought_type=self.thought_type,
                priority=self.priority,
                metadata={
                    "split_mode": mode,
                    "segment_index": index,
                    "char_start": segment.char_start,
                    "char_end": segment.char_end,
                },
                attach_provenance_edge=self.attach_edges,
            )
            for index, segment in enumerate(segments)
        ]

    def _segment(self, content: str) -> list[_Segment]:
        """Dispatch to the configured mode's segmentation."""
        if self.split_mode is SplitMode.PARAGRAPH:
            return self._paragraph_segments(content)
        if self.window_unit == "word":
            return self._fixed_window_word_segments(content)
        return self._fixed_window_char_segments(content)

    def _paragraph_segments(self, content: str) -> list[_Segment]:
        """Split on the configured ``boundary`` and record each segment's span.

        Emits the same segments as the shipped ``[p.strip() for p in
        boundary.split(content)]`` — including, for a custom ``boundary`` carrying
        capturing groups, the captured delimiters (``re.split`` keeps them) — while
        taking each segment's offsets **structurally from the match spans** (not by
        a content search), so ``content[char_start:char_end] == segment.content``
        holds even when a segment's text also occurs inside a delimiter. Blank
        pieces are dropped.
        """
        segments: list[_Segment] = []
        cursor = 0
        group_count = self.boundary.groups
        for match in self.boundary.finditer(content):
            segments.extend(_strip_span(content, cursor, match.start()))
            for group_index in range(1, group_count + 1):
                if match.group(group_index) is not None:
                    group_start, group_end = match.span(group_index)
                    segments.extend(_strip_span(content, group_start, group_end))
            cursor = match.end()
        segments.extend(_strip_span(content, cursor, len(content)))
        return segments

    def _fixed_window_char_segments(self, content: str) -> list[_Segment]:
        """Tile the content with fixed-length character windows.

        Windows advance by ``window_size - window_overlap`` characters and fully
        cover the content: consecutive windows share exactly ``window_overlap``
        characters and no character is dropped. The final window may be shorter
        than ``window_size``. Slices are verbatim (never stripped), so the
        overlap and coverage are exact.
        """
        stride = self.window_size - self.window_overlap
        length = len(content)
        segments: list[_Segment] = []
        start = 0
        while start < length:
            end = min(start + self.window_size, length)
            segments.append(_Segment(content[start:end], start, end))
            if end == length:
                break
            start += stride
        return segments

    def _fixed_window_word_segments(self, content: str) -> list[_Segment]:
        """Tile the content with fixed-count word windows.

        Words are whitespace-delimited runs. Windows advance by
        ``window_size - window_overlap`` words: consecutive windows share exactly
        ``window_overlap`` words and no word is dropped. Each segment spans from
        its first word's start to its last word's end (internal whitespace kept,
        surrounding whitespace excluded). The final window may hold fewer than
        ``window_size`` words.
        """
        words = [match.span() for match in _WORD.finditer(content)]
        if not words:
            return []
        stride = self.window_size - self.window_overlap
        total = len(words)
        segments: list[_Segment] = []
        index = 0
        while index < total:
            last = min(index + self.window_size, total) - 1
            start = words[index][0]
            end = words[last][1]
            segments.append(_Segment(content[start:end], start, end))
            if last == total - 1:
                break
            index += stride
        return segments


def _strip_span(content: str, raw_start: int, raw_end: int) -> list[_Segment]:
    """Return the stripped segment for ``content[raw_start:raw_end]``, if any.

    A single-element list carrying the stripped content and its true offsets in the
    original ``content`` (so ``content[char_start:char_end]`` equals the stripped
    content), or an empty list when the slice is blank.
    """
    raw = content[raw_start:raw_end]
    stripped = raw.strip()
    if not stripped:
        return []
    lead = len(raw) - len(raw.lstrip())
    start = raw_start + lead
    return [_Segment(stripped, start, start + len(stripped))]

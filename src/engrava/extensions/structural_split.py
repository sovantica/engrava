"""Structural-split producer — a deterministic, dependency-free seam consumer.

A reference consumer of the derived-records extension seam
(:class:`~engrava.domain.protocols.derived_records.DerivedRecordProducerProtocol`)
that runs purely on the stored text — no model, no network, no external service:
it splits a stored thought's ``content`` into structural segments (paragraphs by
default) and derives one child thought per segment, each linked back to the
source with a ``DERIVED_FROM`` edge.

It doubles as the canonical example of *how to implement the seam*: subclass the
default hooks (so the object is a valid
:class:`~engrava.domain.protocols.hooks.EngravaHooksProtocol` implementation)
and add :meth:`derive_records`. Because the output is a pure function of the
source content, re-running derivation is idempotent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

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


class StructuralSplitProducer(DefaultEngravaHooks):
    """Derive one child thought per structural segment of the source content.

    Deterministic and dependency-free: the source content is split into
    segments on a configurable boundary (blank lines by default), and each
    non-empty segment becomes a :class:`DerivedRecord`. Splitting runs only when
    the content yields at least two segments, so a single-segment thought
    derives nothing (there is nothing structural to extract).

    Args:
        thought_type: Classification assigned to every derived child. Defaults
            to :attr:`~engrava.domain.enums.ThoughtType.OBSERVATION`.
        priority: Priority assigned to every derived child. Defaults to
            :attr:`~engrava.domain.enums.Priority.P3`.
        boundary: Compiled regular expression marking a segment boundary.
            Defaults to a blank-line (paragraph) boundary.
        attach_edges: Whether each child links back to the source with a
            ``DERIVED_FROM`` edge. Defaults to ``True``.

    Examples:
        >>> producer = StructuralSplitProducer()
        >>> producer.thought_type
        <ThoughtType.OBSERVATION: 'OBSERVATION'>

    """

    def __init__(
        self,
        *,
        thought_type: ThoughtType = ThoughtType.OBSERVATION,
        priority: Priority = Priority.P3,
        boundary: re.Pattern[str] = _PARAGRAPH_BOUNDARY,
        attach_edges: bool = True,
    ) -> None:
        self.thought_type = thought_type
        self.priority = priority
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
            One :class:`DerivedRecord` per non-empty structural segment, in
            document order, or an empty sequence when the content has fewer than
            two segments.

        """
        segments = [seg.strip() for seg in self.boundary.split(thought.content)]
        segments = [seg for seg in segments if seg]
        min_segments_to_split = 2
        if len(segments) < min_segments_to_split:
            return []
        return [
            DerivedRecord(
                content=segment,
                thought_type=self.thought_type,
                priority=self.priority,
                metadata={"segment_index": index},
                attach_provenance_edge=self.attach_edges,
            )
            for index, segment in enumerate(segments)
        ]

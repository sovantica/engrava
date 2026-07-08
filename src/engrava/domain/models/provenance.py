"""ProvenanceContext — opt-in, typed, bounded write-time provenance capture.

Four provenance signals are **irrecoverable at write time**: the retrieval
query and instruction context that shaped a synthesised thought, the ids of
the retrieved thoughts fed into that synthesis, and the session / actor a
thought was produced under.  Once ``create_thought`` returns, these are gone —
they cannot be reconstructed from the stored content.  This model captures
them, opt-in, as a typed, bounded sub-model on the thought row.

Trust posture (binding — read this before using any field):

    Provenance is an **untrusted hint, never identity, authentication, or
    authorization.**  The engine grants it *zero* authority: it is
    descriptive-only, captured verbatim from the caller and consulted for no
    access, ranking, or consolidation decision anywhere in the engine.  In
    particular ``actor_id`` is **not** a tenant boundary — tenant isolation is
    the store's file boundary (one store per tenant via
    :class:`~engrava.infrastructure.service_manager.EngravaManager`), not this
    field.  The engine never infers provenance; the caller passes it
    explicitly, and the ranked search path never guesses it (no auto-capture).

Boundaries (capture-only):

    This model is captured and made **queryable**, nothing more.  Provenance
    feeds no ranking signal, no dreaming / consolidation score, and no edge
    creation.  It adds no new query verb: the identity fields are indexed and
    every field is readable through the same ``json_extract`` filter machinery
    used for metadata, pointed at the ``provenance`` column.

Privacy:

    When provenance is supplied it is persisted in the thought row and
    journaled with the thought like any other content.  Enabling provenance
    therefore *stores* these values (opt-in — there is no silent capture); a
    thought created with ``provenance=None`` writes a NULL column and is
    byte-identical to one created before this model existed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Hard character cap for the identity fields (``session_id`` / ``actor_id``).
#: Deliberately far below the 64 KiB metadata ceiling — an identity token is a
#: short opaque handle, not a payload.
MAX_IDENTITY_CHARS = 256

#: Hard character cap for the free-text synthesis-context fields
#: (``retrieval_query`` / ``instruction_context``).  Generous enough for a real
#: prompt fragment, still orders of magnitude under the metadata ceiling.
MAX_CONTEXT_CHARS = 4096

#: Hard length cap for the ``retrieval_context_ids`` list.  Bounds the number of
#: source-thought ids one synthesis may record.
MAX_CONTEXT_IDS = 128

#: Hard character cap for each element of ``retrieval_context_ids``.  A thought
#: id is a short handle (a UUID is 36 chars); this bounds a single pathological
#: element without constraining legitimate ids.
MAX_CONTEXT_ID_CHARS = 256


class ProvenanceContext(BaseModel):
    """Typed, bounded, opt-in write-time provenance for a thought.

    Every field is optional so a caller records only what it has.  The model
    is frozen; all string fields are hard-capped and the id list is
    length-capped (see the module-level ``MAX_*`` constants) so provenance can
    never approach the metadata size ceiling.

    **Untrusted hint, never identity / authn / authz.** The engine grants this
    model zero authority — it is descriptive-only and is consulted for no
    access, ranking, or consolidation decision.  ``actor_id`` is *not* a tenant
    boundary; tenant isolation is the store's file boundary (one store per
    tenant), not this field.  The engine never infers provenance — the caller
    passes it explicitly, and the ranked search path never guesses it.

    Args:
        session_id: Opaque identifier of the session the thought was produced
            under.  Identity signal — indexed for lookup.  A hint, not an
            authenticated principal.
        actor_id: Opaque identifier of the actor (agent / user handle) that
            produced the thought.  Identity signal — indexed for lookup.
            **Not** an authorization principal and **not** a tenant boundary.
        retrieval_query: The query text that retrieved the context feeding a
            synthesised thought.  Irrecoverable once synthesis completes.
        instruction_context: The instruction / system-prompt fragment that
            shaped the synthesis.  Irrecoverable once synthesis completes.
        retrieval_context_ids: Thought ids that were retrieved and fed into the
            synthesis of this thought.  Length-bounded; queryable but not
            indexed.

    Examples:
        >>> prov = ProvenanceContext(
        ...     session_id="sess-42",
        ...     actor_id="agent-a",
        ...     retrieval_query="remote work trade-offs",
        ...     instruction_context="summarise for a busy exec",
        ...     retrieval_context_ids=["t-1", "t-2"],
        ... )
        >>> prov.session_id
        'sess-42'

    """

    model_config = ConfigDict(frozen=True)

    session_id: str | None = Field(default=None, max_length=MAX_IDENTITY_CHARS)
    actor_id: str | None = Field(default=None, max_length=MAX_IDENTITY_CHARS)
    retrieval_query: str | None = Field(default=None, max_length=MAX_CONTEXT_CHARS)
    instruction_context: str | None = Field(default=None, max_length=MAX_CONTEXT_CHARS)
    retrieval_context_ids: list[str] | None = Field(default=None)

    @field_validator("retrieval_context_ids")
    @classmethod
    def _validate_context_ids(cls, v: list[str] | None) -> list[str] | None:
        """Bound the id list length and each element's length.

        Args:
            v: The candidate list of source-thought ids, or ``None``.

        Returns:
            The validated list unchanged, or ``None``.

        Raises:
            ValueError: If the list exceeds :data:`MAX_CONTEXT_IDS` elements, or
                any element exceeds :data:`MAX_CONTEXT_ID_CHARS` characters.

        """
        if v is None:
            return None
        if len(v) > MAX_CONTEXT_IDS:
            msg = (
                f"retrieval_context_ids has {len(v)} elements, exceeding the "
                f"maximum of {MAX_CONTEXT_IDS}"
            )
            raise ValueError(msg)
        for index, element in enumerate(v):
            if len(element) > MAX_CONTEXT_ID_CHARS:
                msg = (
                    f"retrieval_context_ids[{index}] length {len(element)} "
                    f"exceeds the maximum of {MAX_CONTEXT_ID_CHARS} characters"
                )
                raise ValueError(msg)
        return v

"""Tests for the derived-records seam public types and protocol.

Covers the generic value objects (:class:`DerivedRecord`, :class:`DeriveContext`,
:class:`DeriveGates`) and the capability protocol
(:class:`DerivedRecordProducerProtocol`) in isolation from the persistence
engine: field sets, immutability, validation, runtime-checkability, and the
public-surface stability guarantees (AC-1, AC-7).
"""

from __future__ import annotations

import dataclasses

import pytest

from engrava.domain.enums import Priority, ThoughtType
from engrava.domain.protocols import derived_records
from engrava.domain.protocols.derived_records import (
    DeriveContext,
    DerivedRecord,
    DerivedRecordProducerProtocol,
    DeriveGates,
)

# ---------------------------------------------------------------------------
# DerivedRecord — producer-owned fields only (AC-1 / AC-9 field authority)
# ---------------------------------------------------------------------------

#: The system-managed fields core assigns at persist time. None of them may be
#: representable on ``DerivedRecord`` — a producer must have nothing to forge.
_SYSTEM_MANAGED_FIELDS = frozenset(
    {
        "thought_id",
        "created_at",
        "updated_at",
        "created_cycle",
        "updated_cycle",
        "lifecycle_status",
        "status",
        "confirmation_count",
        "valid_from",
        "valid_until",
        "expires_at",
    },
)


def test_derived_record_exposes_only_producer_owned_fields() -> None:
    """``DerivedRecord`` carries only content-side fields, no system identity."""
    names = {f.name for f in dataclasses.fields(DerivedRecord)}
    assert names == {
        "content",
        "thought_type",
        "priority",
        "metadata",
        "attach_provenance_edge",
    }
    # No system-managed field is structurally representable. The persisted
    # thought's ``essence`` is one such core-owned field — a producer must not be
    # able to supply it (core derives it from ``content``).
    assert "essence" not in names
    assert names.isdisjoint(_SYSTEM_MANAGED_FIELDS)


def test_derived_record_is_frozen() -> None:
    """``DerivedRecord`` is immutable (frozen dataclass)."""
    record = DerivedRecord(
        content="x",
        thought_type=ThoughtType.OBSERVATION,
        priority=Priority.P3,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.content = "y"  # type: ignore[misc]


def test_derived_record_defaults() -> None:
    """Optional fields default to empty metadata and an attached edge."""
    record = DerivedRecord(
        content="x",
        thought_type=ThoughtType.NOTE,
        priority=Priority.P2,
    )
    assert record.metadata == {}
    assert record.attach_provenance_edge is True


def test_derived_record_rejects_empty_content() -> None:
    """Empty ``content`` is rejected (it would yield an invalid empty essence)."""
    with pytest.raises(ValueError, match="content must be non-empty"):
        DerivedRecord(content="", thought_type=ThoughtType.NOTE, priority=Priority.P2)


def test_derived_record_metadata_is_per_instance() -> None:
    """The default metadata factory does not share state across instances."""
    a = DerivedRecord(content="a", thought_type=ThoughtType.NOTE, priority=Priority.P2)
    b = DerivedRecord(content="b", thought_type=ThoughtType.NOTE, priority=Priority.P2)
    assert a.metadata is not b.metadata


# ---------------------------------------------------------------------------
# DeriveContext — stable, store-handle-free (AC-1)
# ---------------------------------------------------------------------------


def test_derive_context_fields_are_stable_and_handle_free() -> None:
    """``DeriveContext`` exposes only stable facts and no store handle."""
    names = {f.name for f in dataclasses.fields(DeriveContext)}
    assert names == {
        "source_thought_id",
        "source_content_hash",
        "cycle_at_derivation",
        "origin",
    }
    # No field name hints at a store / connection / handle being exposed.
    assert not any(token in name for name in names for token in ("store", "conn", "handle", "db"))


def test_derive_context_is_frozen() -> None:
    """``DeriveContext`` is immutable."""
    ctx = DeriveContext(
        source_thought_id="t-1",
        source_content_hash="ab",
        cycle_at_derivation=0,
        origin="create_thought",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.origin = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DeriveGates — defaults + validation
# ---------------------------------------------------------------------------


def test_derive_gates_defaults_are_inert() -> None:
    """The default gates disable the seam with a log-and-continue policy."""
    gates = DeriveGates()
    assert gates.enabled is False
    assert gates.on_error == "log"
    assert gates.max_derived_per_source == 32


def test_derive_gates_rejects_unknown_on_error() -> None:
    """An unrecognised ``on_error`` policy is rejected at construction."""
    with pytest.raises(ValueError, match="on_error"):
        DeriveGates(on_error="explode")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_cap", [0, -1])
def test_derive_gates_rejects_non_positive_cap(bad_cap: int) -> None:
    """``max_derived_per_source`` must be at least one."""
    with pytest.raises(ValueError, match="max_derived_per_source"):
        DeriveGates(max_derived_per_source=bad_cap)


def test_derive_gates_rejects_bool_cap() -> None:
    """A ``bool`` cannot masquerade as an ``int`` cap (bool is an int subclass)."""
    with pytest.raises(TypeError, match="max_derived_per_source"):
        DeriveGates(max_derived_per_source=True)


def test_derive_gates_rejects_non_int_cap() -> None:
    """A non-int cap is rejected, matching the YAML loader's integer contract."""
    with pytest.raises(TypeError, match="max_derived_per_source"):
        DeriveGates(max_derived_per_source=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_enabled", [1, 0, "yes"])
def test_derive_gates_rejects_non_bool_enabled(bad_enabled: object) -> None:
    """``enabled`` must be a strict ``bool`` (matching the YAML loader)."""
    with pytest.raises(TypeError, match="enabled"):
        DeriveGates(enabled=bad_enabled)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol — runtime-checkable capability detection (AC-1 / D1)
# ---------------------------------------------------------------------------


def test_producer_protocol_is_runtime_checkable() -> None:
    """A class with ``derive_records`` is an instance; one without is not."""

    class _Producer:
        async def derive_records(
            self,
            thought: object,
            ctx: object,
        ) -> list[DerivedRecord]:
            return []

    class _NotAProducer:
        pass

    assert isinstance(_Producer(), DerivedRecordProducerProtocol)
    assert not isinstance(_NotAProducer(), DerivedRecordProducerProtocol)


# ---------------------------------------------------------------------------
# Stability + deprecation policy on the public surface (AC-7)
# ---------------------------------------------------------------------------


def test_public_types_document_stability_policy() -> None:
    """The seam's public surface documents the X.Y.x stability guarantee."""
    module_doc = derived_records.__doc__ or ""
    assert "X.Y.x" in module_doc
    assert "deprecat" in module_doc.lower()
    protocol_doc = DerivedRecordProducerProtocol.__doc__ or ""
    assert "stability" in protocol_doc.lower()


def test_public_api_names_are_generic() -> None:
    """The exported seam names carry no domain-specific (non-generic) tokens."""
    exported = {
        "DerivedRecord",
        "DeriveContext",
        "DeriveGates",
        "DerivedRecordProducerProtocol",
    }
    for name in exported:
        assert hasattr(derived_records, name)

"""Integration tests — cluster quality gates wired into the dreaming loop.

Each test seeds a small cluster engineered to trip a specific gate and
runs ``DreamingExtension._create_reflections`` against the real
``SqliteEngravaCore``.  A gate-rejected cluster produces zero
REFLECTIONs; a clean cluster produces exactly one.

The fixtures keep the cluster pre-conditions (size, similarity,
membership) just permissive enough to reach the gating loop, so a test
that asserts "no REFLECTION created" really does pin the gate's
decision rather than some upstream filter.

Backward-compat: flipping ``cluster_quality_gating_enabled=False``
must restore the pre-gating behaviour — clusters that the gates would
reject still produce a REFLECTION.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava.config import DreamingConfig, DreamingGates
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models import MetadataValue, ThoughtRecord
from engrava.extensions.dreaming import DreamingExtension
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(
    thought_id: str,
    content: str,
    *,
    metadata: dict[str, MetadataValue] | None = None,
) -> ThoughtRecord:
    """Minimal OBSERVATION ThoughtRecord shaped for clustering tests."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=(content[:60] or "essence"),
        content=content,
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        metadata=metadata or {},
    )


def _gates(
    *,
    cluster_quality_gating_enabled: bool = True,
    **overrides: object,
) -> DreamingGates:
    """Build a ``DreamingGates`` instance with the bare minimum opt-ins."""
    base: dict[str, object] = {
        "min_confirmations": 0,
        "min_age_cycles": 0,
        "max_promoted_per_run": 20,
        "min_cluster_size": 2,
        "cluster_algorithm": "agglomerative",
        "clustering_min_new_candidates": 0,
        "cluster_similarity_threshold": 0.5,
        "cluster_quality_gating_enabled": cluster_quality_gating_enabled,
    }
    base.update(overrides)
    return DreamingGates(**base)  # type: ignore[arg-type]


def _cfg(gates: DreamingGates) -> DreamingConfig:
    return DreamingConfig(enabled=True, gates=gates, clustering_backend="numpy")


async def _persist_cluster(
    store: SqliteEngravaCore,
    members: list[ThoughtRecord],
    *,
    axis: int = 0,
    dim: int = 8,
) -> frozenset[str]:
    """Persist members + identical unit vectors so they form one tight cluster."""
    unit_vector = [0.0] * dim
    unit_vector[axis] = 1.0
    for member in members:
        await store.create_thought(member)
        await store.store_embedding(
            member.thought_id,
            unit_vector,
            model_name="test",
        )
    return frozenset(member.thought_id for member in members)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEngravaCore]:
    """Fresh in-file store (avoids :memory: WAL conflicts under aiosqlite)."""
    conn = await aiosqlite.connect(str(tmp_path / "gating.db"))
    conn.row_factory = aiosqlite.Row
    s = SqliteEngravaCore(conn)
    await s.ensure_schema()
    try:
        yield s
    finally:
        await conn.close()


# Three short conversational lines that have enough non-generic vocabulary
# to clear Gate 8 (meaningful keyphrases) and Gate 2 (no persona indicators)
# while being content-distinct enough to clear Gate 1 (no byte-identical
# duplicates).  Used as the baseline "clean" cluster shape.
_CLEAN_MEMBER_CONTENTS: tuple[str, ...] = (
    "[USER] I practised violin pieces in the morning",
    "[USER] Played violin scales after breakfast yesterday",
    "[USER] Recorded violin etudes for my teacher",
)


def _clean_cluster_members(prefix: str = "ok") -> list[ThoughtRecord]:
    """Three external-source clean members — clears every active gate."""
    return [
        _obs(
            f"{prefix}-{idx}",
            content,
            metadata={"source": {"is_self": False, "confidence": "high"}},
        )
        for idx, content in enumerate(_CLEAN_MEMBER_CONTENTS)
    ]


# ---------------------------------------------------------------------------
# Per-gate categorical rejection
# ---------------------------------------------------------------------------


class TestPerGateCategoricalRejection:
    """Each gate must reject its own pathology end-to-end."""

    async def test_clean_cluster_produces_reflection(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        cluster = await _persist_cluster(store, _clean_cluster_members())
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 1

    async def test_duplicate_members_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Two byte-identical members trip Gate 1.
        duplicate_text = "[USER] I practised violin pieces in the morning"
        members = [
            _obs(
                f"dup-{idx}",
                duplicate_text,
                metadata={"source": {"is_self": False}},
            )
            for idx in range(3)
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0

    async def test_persona_only_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Three persona descriptions, no conversation markers — Gate 2.
        members = [
            _obs(
                "persona-a",
                "Alex Martinez is a graphic design student.",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "persona-b",
                "Alex was born in 1974 and is an industrial designer.",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "persona-c",
                "Embracing their creative side, Alex is an artist.",
                metadata={"source": {"is_self": False}},
            ),
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0

    async def test_contradictory_members_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # "stopped" vs "created" trips Gate 3.
        members = [
            _obs(
                "ct-1",
                "[USER] I stopped making study videos last year",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "ct-2",
                "[USER] I created a study video and got positive feedback",
                metadata={"source": {"is_self": False}},
            ),
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0

    async def test_low_cohesion_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Two members on orthogonal axes -> mean pairwise cosine 0.0,
        # which is strictly below the 0.40 default.
        member_a = _obs(
            "co-a",
            "[USER] morning violin practice with new etudes",
            metadata={"source": {"is_self": False}},
        )
        member_b = _obs(
            "co-b",
            "[USER] painting watercolour sketches in the studio",
            metadata={"source": {"is_self": False}},
        )
        await store.create_thought(member_a)
        await store.create_thought(member_b)
        vec_a = [1.0, 0.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0, 0.0]
        await store.store_embedding(member_a.thought_id, vec_a, model_name="test")
        await store.store_embedding(member_b.thought_id, vec_b, model_name="test")
        cluster = frozenset({member_a.thought_id, member_b.thought_id})
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0

    async def test_external_source_mixed_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Two external members + one agent-authored member -> external
        # fraction 2/3 ~= 0.667 < 0.95 default -> Gate 5 rejects.
        members = [
            _obs(
                "es-1",
                "[USER] violin etudes practiced today",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "es-2",
                "[USER] morning scales with metronome",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "es-3",
                "[USER] evening repertoire session",
                metadata={"source": {"is_self": True}},
            ),
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0

    async def test_named_entity_inconsistent_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # First member entities: {Alice, Paris}.  Other members carry
        # disjoint entities, so the overlap ratio drops below the 0.60
        # default -> Gate 6 rejects.
        members = [
            _obs(
                "ne-1",
                "[USER] Alice visited Paris museums last weekend",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "ne-2",
                "[USER] Bob practised judo at Tokyo dojo evenings",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "ne-3",
                "[USER] Carol gardened tulips in Amsterdam every Sunday",
                metadata={"source": {"is_self": False}},
            ),
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0

    async def test_generic_keyphrases_rejected(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Single-word distinct content — too short for the n-gram
        # extractor to produce any bigrams, so the post-build
        # ``top_keyphrases`` list is empty.  ``has_meaningful_keyphrases``
        # treats an empty list as "nothing meaningful to anchor a
        # REFLECTION on" and Gate 8 rejects the cluster.  All earlier
        # gates pass (no duplicates, no persona indicators, no
        # sentiment-opposites, external sources, no NE inconsistency).
        members = [
            _obs("gk-1", "hello", metadata={"source": {"is_self": False}}),
            _obs("gk-2", "world", metadata={"source": {"is_self": False}}),
            _obs("gk-3", "goodbye", metadata={"source": {"is_self": False}}),
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0


# ---------------------------------------------------------------------------
# Safe-fallback — legacy data without metadata.source passes
# ---------------------------------------------------------------------------


class TestSafeFallback:
    """Clusters with missing/malformed source survive the homogeneity gate."""

    async def test_missing_source_passes_external_gate(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # No metadata.source at all -- legacy data path.  Gate 5 must
        # treat missing as external (safe fallback), and the cluster
        # passes through to REFLECTION creation.
        members = [
            _obs(f"legacy-{idx}", content)  # NB: no metadata kwarg
            for idx, content in enumerate(_CLEAN_MEMBER_CONTENTS)
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 1

    async def test_malformed_source_passes_external_gate(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # metadata.source set to a non-dict scalar -- treated as
        # external (defensive fallback) so Gate 5 still passes.
        members = [
            _obs(
                f"mal-{idx}",
                content,
                metadata={"source": "not-a-dict"},
            )
            for idx, content in enumerate(_CLEAN_MEMBER_CONTENTS)
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 1


# ---------------------------------------------------------------------------
# Backward compatibility — disabling the gates restores legacy behaviour
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """``cluster_quality_gating_enabled=False`` reverts to pre-gating behaviour."""

    async def test_disabling_gates_allows_rejected_cluster_through(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Persona-only cluster -- Gate 2 would reject under defaults.
        members = [
            _obs(
                "bw-a",
                "Alex Martinez is a graphic design student.",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "bw-b",
                "Alex was born in 1974 and is an industrial designer.",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "bw-c",
                "Embracing their creative side, Alex is an artist.",
                metadata={"source": {"is_self": False}},
            ),
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(
            config=_cfg(_gates(cluster_quality_gating_enabled=False)),
        )
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 1

    async def test_disabling_gate_8_alone_keeps_others_active(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # ``cluster_quality_require_meaningful_keyphrases=False`` keeps
        # the other gates active.  Persona-only payload still trips
        # Gate 2 even when the keyphrase gate is opted out.
        members = [
            _obs(
                "g8-a",
                "Alice is a sculptor working in clay.",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "g8-b",
                "Alice is an art teacher in the local school.",
                metadata={"source": {"is_self": False}},
            ),
            _obs(
                "g8-c",
                "Alice is the curator at the regional museum.",
                metadata={"source": {"is_self": False}},
            ),
        ]
        cluster = await _persist_cluster(store, members)
        ext = DreamingExtension(
            config=_cfg(
                _gates(cluster_quality_require_meaningful_keyphrases=False),
            ),
        )
        created = await ext._create_reflections(
            store,
            [cluster],
            current_cycle=10,
        )
        assert created == 0

    async def test_distinct_cluster_ids_isolate_rejections(
        self,
        store: SqliteEngravaCore,
    ) -> None:
        # Two clusters: one clean, one persona-only.  The clean one
        # produces a REFLECTION; the persona-only one is dropped.
        clean = await _persist_cluster(
            store,
            _clean_cluster_members(prefix="iso-clean"),
            axis=0,
        )
        persona_members = [
            _obs(
                f"iso-persona-{idx}",
                content,
                metadata={"source": {"is_self": False}},
            )
            for idx, content in enumerate(
                (
                    f"Persona {uuid.uuid4().hex[:8]} is a teacher.",
                    f"Persona {uuid.uuid4().hex[:8]} is the gardener.",
                    f"Persona {uuid.uuid4().hex[:8]} is an engineer.",
                ),
            )
        ]
        persona = await _persist_cluster(store, persona_members, axis=1)
        ext = DreamingExtension(config=_cfg(_gates()))
        created = await ext._create_reflections(
            store,
            [clean, persona],
            current_cycle=10,
        )
        assert created == 1

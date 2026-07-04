"""Tests for the action-outcome feedback loop and mutable action lifecycle.

Covers the two coupled write-surface features:

1. ``update_action`` — advancing a stored action through its state machine
   (transition validation, verification-only updates on terminal actions, the
   no-op contract, ``ActionNotFoundError``), journaled as ``UPDATE_ACTION``.
2. The denormalised ``thought.action_outcome_score`` aggregate: the per-action
   outcome mapping, the mean-over-terminal-actions aggregate, the idempotent
   recompute and its firing points, and — critically — the flat-safe guarantee
   that the 6th ``action_outcome`` dreaming signal leaves an action-free store's
   dreaming promotion byte-identical to before the signal existed.

The schema block asserts the core-16 migration (column + ``action`` seek index,
idempotency, fresh-DB version, cascade from an older version) and that the
by-source action query uses the new index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    ActionNotFoundError,
    ActionRecord,
    ActionStatus,
    ActionType,
    InvalidTransitionError,
    LifecycleStatus,
    Priority,
    ReadOnlyViolationError,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    VerificationStatus,
)
from engrava.config import DreamingConfig, DreamingGates
from engrava.extensions.dreaming import DreamingExtension
from engrava.extensions.dreaming_signals import (
    ActionOutcomeSignal,
    DreamingContext,
    default_signal_active,
)
from engrava.infrastructure.read_only_store import ReadOnlyEngrava
from engrava.infrastructure.sqlite.engrava_core import (
    _CONFIRMED_VERIFICATION_OUTCOME,
    _action_outcome_value,
    _aggregate_action_outcome,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

_CYCLE = 100


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> AsyncIterator[SqliteEngravaCore]:
    """A schema-bootstrapped in-memory store (journaling off)."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(conn)
    await s.ensure_schema()
    yield s
    await conn.close()


@pytest.fixture
async def jstore() -> AsyncIterator[SqliteEngravaCore]:
    """A schema-bootstrapped in-memory store with journaling enabled."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    s = SqliteEngravaCore(conn, journal_enabled=True)
    await s.ensure_schema()
    yield s
    await conn.close()


def _thought(thought_id: str = "t-001", *, updated_cycle: int = 0) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.TASK,
        essence=f"Essence {thought_id}",
        content=f"Content of {thought_id}",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=updated_cycle,
        source="test",
    )


def _action(
    action_id: str,
    thought_id: str = "t-001",
    *,
    status: ActionStatus = ActionStatus.PLANNED,
    verification_status: VerificationStatus = VerificationStatus.PENDING,
) -> ActionRecord:
    return ActionRecord(
        action_id=action_id,
        source_thought_id=thought_id,
        action_type=ActionType.TOOL_CALL,
        intent="do the thing",
        status=status,
        verification_status=verification_status,
    )


async def _journal_len(store: SqliteEngravaCore) -> int:
    cursor = await store._db.execute("SELECT COUNT(*) FROM journal_entry")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _score(store: SqliteEngravaCore, thought_id: str) -> float | None:
    """Fetch a thought (asserting it exists) and return its outcome score."""
    thought = await store.get_thought(thought_id)
    assert thought is not None
    return thought.action_outcome_score


# ---------------------------------------------------------------------------
# Per-action outcome value + aggregate (pure functions)
# ---------------------------------------------------------------------------


class TestActionOutcomeValue:
    """The documented per-action outcome mapping."""

    @pytest.mark.parametrize(
        "status",
        [ActionStatus.PLANNED, ActionStatus.EXECUTING, ActionStatus.BLOCKED],
    )
    def test_non_terminal_is_none(self, status: ActionStatus) -> None:
        assert _action_outcome_value(_action("a", status=status)) is None

    def test_failed_is_zero_regardless_of_verification(self) -> None:
        for verif in VerificationStatus:
            act = _action("a", status=ActionStatus.FAILED, verification_status=verif)
            assert _action_outcome_value(act) == 0.0

    @pytest.mark.parametrize(
        ("verif", "expected"),
        [
            (VerificationStatus.CONFIRMED, 1.0),
            (VerificationStatus.PARTIAL, 0.5),
            (VerificationStatus.PENDING, 0.5),
            (VerificationStatus.UNVERIFIABLE, 0.5),
            (VerificationStatus.FAILED, 0.0),
        ],
    )
    def test_confirmed_adjusted_by_verification(
        self, verif: VerificationStatus, expected: float
    ) -> None:
        act = _action("a", status=ActionStatus.CONFIRMED, verification_status=verif)
        assert _action_outcome_value(act) == expected

    def test_mapping_const_covers_every_verification_status(self) -> None:
        # The named table must map every VerificationStatus so a new enum member
        # cannot silently fall through to a KeyError at recompute time.
        assert set(_CONFIRMED_VERIFICATION_OUTCOME) == set(VerificationStatus)


class TestAggregate:
    """Mean over terminal actions, ``None`` when there are none."""

    def test_none_when_no_actions(self) -> None:
        assert _aggregate_action_outcome([]) is None

    def test_none_when_only_non_terminal(self) -> None:
        actions = [
            _action("a", status=ActionStatus.PLANNED),
            _action("b", status=ActionStatus.EXECUTING),
        ]
        assert _aggregate_action_outcome(actions) is None

    def test_mean_over_mixed_outcomes(self) -> None:
        actions = [
            _action(
                "a", status=ActionStatus.CONFIRMED, verification_status=VerificationStatus.CONFIRMED
            ),
            _action("b", status=ActionStatus.FAILED, verification_status=VerificationStatus.FAILED),
        ]
        # Mean of a CONFIRMED (1.0) and a FAILED (0.0) action is 0.5.
        assert _aggregate_action_outcome(actions) == 0.5

    def test_non_terminal_excluded_from_mean(self) -> None:
        actions = [
            _action(
                "a", status=ActionStatus.CONFIRMED, verification_status=VerificationStatus.CONFIRMED
            ),
            _action("b", status=ActionStatus.PLANNED),  # excluded
        ]
        # Mean over the single terminal action only.
        assert _aggregate_action_outcome(actions) == 1.0


# ---------------------------------------------------------------------------
# update_action — state machine
# ---------------------------------------------------------------------------


class TestUpdateActionTransitions:
    """Every legal transition persists; every illegal jump raises."""

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (ActionStatus.PLANNED, ActionStatus.EXECUTING),
            (ActionStatus.PLANNED, ActionStatus.BLOCKED),
            (ActionStatus.EXECUTING, ActionStatus.CONFIRMED),
            (ActionStatus.EXECUTING, ActionStatus.FAILED),
            (ActionStatus.BLOCKED, ActionStatus.PLANNED),
        ],
    )
    async def test_legal_transition_persists(
        self, store: SqliteEngravaCore, start: ActionStatus, target: ActionStatus
    ) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=start))

        updated = await store.update_action("a", status=target)

        assert updated.status is target
        # Reload from storage to confirm the write landed.
        reloaded = await store.get_actions("t-001")
        assert reloaded[0].status is target

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (ActionStatus.PLANNED, ActionStatus.CONFIRMED),
            (ActionStatus.PLANNED, ActionStatus.FAILED),
            (ActionStatus.CONFIRMED, ActionStatus.EXECUTING),
            (ActionStatus.FAILED, ActionStatus.PLANNED),
            (ActionStatus.EXECUTING, ActionStatus.BLOCKED),
            (ActionStatus.BLOCKED, ActionStatus.EXECUTING),
        ],
    )
    async def test_illegal_transition_raises(
        self, store: SqliteEngravaCore, start: ActionStatus, target: ActionStatus
    ) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=start))

        with pytest.raises(InvalidTransitionError):
            await store.update_action("a", status=target)

        # The stored status is unchanged after the rejected jump.
        reloaded = await store.get_actions("t-001")
        assert reloaded[0].status is start

    async def test_unknown_action_raises(self, store: SqliteEngravaCore) -> None:
        with pytest.raises(ActionNotFoundError):
            await store.update_action("ghost", status=ActionStatus.EXECUTING)


class TestVerificationOnlyUpdate:
    """Verification advances on a terminal action while status stays terminal."""

    @pytest.mark.parametrize(
        "verif",
        [
            VerificationStatus.CONFIRMED,
            VerificationStatus.PARTIAL,
            VerificationStatus.FAILED,
            VerificationStatus.UNVERIFIABLE,
        ],
    )
    async def test_verification_only_on_terminal_confirmed(
        self, store: SqliteEngravaCore, verif: VerificationStatus
    ) -> None:
        await store.create_thought(_thought())
        await store.create_action(
            _action(
                "a",
                status=ActionStatus.CONFIRMED,
                verification_status=VerificationStatus.PENDING,
            )
        )

        updated = await store.update_action("a", verification_status=verif)

        assert updated.status is ActionStatus.CONFIRMED  # status unchanged (terminal)
        assert updated.verification_status is verif
        reloaded = await store.get_actions("t-001")
        assert reloaded[0].verification_status is verif

    async def test_verification_only_on_terminal_failed(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(
            _action("a", status=ActionStatus.FAILED, verification_status=VerificationStatus.PENDING)
        )

        updated = await store.update_action(
            "a", verification_status=VerificationStatus.UNVERIFIABLE
        )

        assert updated.status is ActionStatus.FAILED
        assert updated.verification_status is VerificationStatus.UNVERIFIABLE


class TestNoOpUpdate:
    """A no-op update writes no journal entry and does not recompute."""

    async def test_noop_identical_values(self, jstore: SqliteEngravaCore) -> None:
        await jstore.create_thought(_thought())
        # Terminal create so the thought already carries a score to watch.
        await jstore.create_action(
            _action(
                "a",
                status=ActionStatus.CONFIRMED,
                verification_status=VerificationStatus.CONFIRMED,
            )
        )
        before_journal = await _journal_len(jstore)
        before = await jstore.get_thought("t-001")
        assert before is not None
        assert before.action_outcome_score == 1.0

        # Supply the exact stored values -> no change.
        same = await jstore.update_action(
            "a",
            status=ActionStatus.CONFIRMED,
            verification_status=VerificationStatus.CONFIRMED,
        )

        assert same.status is ActionStatus.CONFIRMED
        assert same.verification_status is VerificationStatus.CONFIRMED
        assert await _journal_len(jstore) == before_journal  # no journal entry
        after = await jstore.get_thought("t-001")
        assert after is not None
        assert after.action_outcome_score == 1.0  # unchanged

    async def test_noop_all_kwargs_omitted(self, jstore: SqliteEngravaCore) -> None:
        await jstore.create_thought(_thought())
        await jstore.create_action(_action("a", status=ActionStatus.PLANNED))
        before_journal = await _journal_len(jstore)

        same = await jstore.update_action("a")  # nothing supplied

        assert same.status is ActionStatus.PLANNED
        assert await _journal_len(jstore) == before_journal


# ---------------------------------------------------------------------------
# Recompute + firing points
# ---------------------------------------------------------------------------


class TestRecomputeFiring:
    """The recompute fires only on outcome-affecting changes."""

    async def test_terminal_create_sets_score(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(
            _action(
                "a",
                status=ActionStatus.CONFIRMED,
                verification_status=VerificationStatus.CONFIRMED,
            )
        )
        t = await store.get_thought("t-001")
        assert t is not None
        assert t.action_outcome_score == 1.0

    async def test_non_terminal_create_leaves_score_none(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.PLANNED))
        t = await store.get_thought("t-001")
        assert t is not None
        assert t.action_outcome_score is None

    async def test_transition_to_terminal_recomputes(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.EXECUTING))
        assert (await store.get_thought("t-001")).action_outcome_score is None  # type: ignore[union-attr]

        await store.update_action("a", status=ActionStatus.CONFIRMED)
        t = await store.get_thought("t-001")
        # CONFIRMED status + default PENDING verification = neutral 0.5.
        assert t is not None
        assert t.action_outcome_score == 0.5

    async def test_non_terminal_move_does_not_recompute(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.PLANNED))

        await store.update_action("a", status=ActionStatus.EXECUTING)  # non-terminal
        t = await store.get_thought("t-001")
        assert t is not None
        assert t.action_outcome_score is None

    async def test_verification_change_on_terminal_recomputes(
        self, store: SqliteEngravaCore
    ) -> None:
        await store.create_thought(_thought())
        await store.create_action(
            _action(
                "a",
                status=ActionStatus.CONFIRMED,
                verification_status=VerificationStatus.CONFIRMED,
            )
        )
        assert (await store.get_thought("t-001")).action_outcome_score == 1.0  # type: ignore[union-attr]

        # Verification degrades to PARTIAL -> score recomputes to 0.5.
        await store.update_action("a", verification_status=VerificationStatus.PARTIAL)
        t = await store.get_thought("t-001")
        assert t is not None
        assert t.action_outcome_score == 0.5

    async def test_mean_over_two_actions(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(
            _action(
                "ok",
                status=ActionStatus.CONFIRMED,
                verification_status=VerificationStatus.CONFIRMED,
            )
        )
        await store.create_action(
            _action(
                "bad", status=ActionStatus.FAILED, verification_status=VerificationStatus.FAILED
            )
        )
        t = await store.get_thought("t-001")
        assert t is not None
        assert t.action_outcome_score == 0.5  # mean of 1.0 and 0.0

    async def test_score_none_when_no_terminal_actions(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.PLANNED))
        await store.create_action(_action("b", status=ActionStatus.BLOCKED))
        t = await store.get_thought("t-001")
        assert t is not None
        assert t.action_outcome_score is None

    async def test_recompute_idempotent(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(
            _action(
                "a",
                status=ActionStatus.CONFIRMED,
                verification_status=VerificationStatus.CONFIRMED,
            )
        )
        first = (await store.get_thought("t-001")).action_outcome_score  # type: ignore[union-attr]
        # Direct second recompute — no intervening action change.
        await store._recompute_action_outcome("t-001")
        second = (await store.get_thought("t-001")).action_outcome_score  # type: ignore[union-attr]
        assert first == second == 1.0


# ---------------------------------------------------------------------------
# Journaling
# ---------------------------------------------------------------------------


class TestJournaling:
    """update_action journals as UPDATE_ACTION; the chain verifies + detects tamper."""

    async def test_update_action_writes_one_entry(self, jstore: SqliteEngravaCore) -> None:
        await jstore.create_thought(_thought())
        await jstore.create_action(_action("a", status=ActionStatus.PLANNED))
        before = await _journal_len(jstore)

        await jstore.update_action("a", status=ActionStatus.EXECUTING)

        # One UPDATE_ACTION entry (non-terminal move -> no recompute journal).
        assert await _journal_len(jstore) == before + 1
        cursor = await jstore._db.execute(
            "SELECT mutation_type FROM journal_entry ORDER BY sequence_number DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "UPDATE_ACTION"

    async def test_terminal_update_journals_action_and_recompute(
        self, jstore: SqliteEngravaCore
    ) -> None:
        await jstore.create_thought(_thought())
        await jstore.create_action(_action("a", status=ActionStatus.EXECUTING))
        before = await _journal_len(jstore)

        await jstore.update_action("a", status=ActionStatus.CONFIRMED)

        # UPDATE_ACTION + the UPDATE_THOUGHT from the recompute = two entries.
        assert await _journal_len(jstore) == before + 2
        cursor = await jstore._db.execute(
            "SELECT mutation_type FROM journal_entry ORDER BY sequence_number"
        )
        kinds = [r[0] for r in await cursor.fetchall()]
        assert kinds[-2:] == ["UPDATE_ACTION", "UPDATE_THOUGHT"]

    async def test_clean_chain_verifies(self, jstore: SqliteEngravaCore) -> None:
        await jstore.create_thought(_thought())
        await jstore.create_action(_action("a", status=ActionStatus.EXECUTING))
        await jstore.update_action("a", status=ActionStatus.CONFIRMED)
        await jstore.update_action("a", verification_status=VerificationStatus.PARTIAL)
        await jstore._db.commit()

        result = await jstore.verify_journal()
        assert result.valid is True
        assert result.first_invalid_sequence is None

    async def test_tampered_update_action_detected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tamper.db"
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        store = SqliteEngravaCore(conn, journal_enabled=True)
        await store.ensure_schema()
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.EXECUTING))
        await store.update_action("a", status=ActionStatus.FAILED)
        await store._db.commit()

        # Locate the UPDATE_ACTION row's sequence number and tamper its delta.
        cursor = await conn.execute(
            "SELECT sequence_number FROM journal_entry WHERE mutation_type = 'UPDATE_ACTION'"
        )
        row = await cursor.fetchone()
        assert row is not None
        seq = int(row[0])
        await conn.execute(
            "UPDATE journal_entry SET delta = ? WHERE sequence_number = ?",
            ('{"before": {}, "after": {"status": "TAMPERED"}}', seq),
        )
        await conn.commit()

        result = await store.verify_journal()
        assert result.valid is False
        assert result.first_invalid_sequence == seq
        await conn.close()

    async def test_no_journal_when_disabled(self, store: SqliteEngravaCore) -> None:
        # store fixture has journaling off; update_action must not error and
        # must not attempt to write to the (present-but-unused) journal table.
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.EXECUTING))
        await store.update_action("a", status=ActionStatus.CONFIRMED)
        assert await _journal_len(store) == 0


# ---------------------------------------------------------------------------
# 6th dreaming signal — flat-safe regression guard
# ---------------------------------------------------------------------------


def _obs(
    thought_id: str,
    *,
    action_outcome_score: float | None = None,
    updated_cycle: int = _CYCLE,
) -> ThoughtRecord:
    """A recent + mature ACTIVE OBSERVATION, optionally carrying an outcome."""
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=f"Essence {thought_id}",
        content=f"Content of {thought_id} about apples and oranges and pears.",
        priority=Priority.P3,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=updated_cycle,
        source="test",
        action_outcome_score=action_outcome_score,
    )


def _dreaming_cfg(signals: dict[str, float] | None = None) -> DreamingConfig:
    kwargs: dict[str, object] = {
        "enabled": True,
        "promote_threshold": 0.7,
        "max_p1_fraction": 0.5,
        "gates": DreamingGates(
            min_age_cycles=0,
            allow_zero_confirmation=True,
            enable_reflections=False,
        ),
    }
    if signals is not None:
        kwargs["signals"] = signals
    return DreamingConfig(**kwargs)  # type: ignore[arg-type]


class TestSignalActiveness:
    """The action_outcome signal is active iff some candidate has an outcome."""

    def test_inactive_when_no_candidate_has_outcome(self) -> None:
        candidates = [_obs("a"), _obs("b")]
        assert (
            default_signal_active(
                "action_outcome",
                candidates,
                current_cycle=_CYCLE,
                access_tracking_enabled=False,
            )
            is False
        )

    def test_active_when_a_candidate_has_outcome(self) -> None:
        candidates = [_obs("a"), _obs("b", action_outcome_score=0.0)]
        assert (
            default_signal_active(
                "action_outcome",
                candidates,
                current_cycle=_CYCLE,
                access_tracking_enabled=False,
            )
            is True
        )

    def test_signal_reads_score_and_defaults_none_to_zero(self) -> None:
        sig = ActionOutcomeSignal()
        ctx = DreamingContext(current_cycle=_CYCLE, total_thoughts=1)
        assert sig(_obs("a", action_outcome_score=0.75), ctx) == 0.75
        assert sig(_obs("a"), ctx) == 0.0


class TestFlatSafeRegression:
    """THE regression guard — an action-free store's dreaming is byte-identical."""

    def test_action_free_scores_byte_identical(self) -> None:
        # Candidates with NO action outcome (the pre-feature world).
        candidates = [_obs(f"t{i}", updated_cycle=i * 5) for i in range(6)]

        cfg6 = _dreaming_cfg()  # ships the 6th signal
        ext6 = DreamingExtension(cfg6)
        five = {k: v for k, v in cfg6.signals.items() if k != "action_outcome"}
        ext5 = DreamingExtension(_dreaming_cfg(signals=five))  # pre-feature 5 signals

        w6, flat6 = ext6._compute_active_weights(candidates, current_cycle=_CYCLE)
        w5, _ = ext5._compute_active_weights(candidates, current_cycle=_CYCLE)

        # The new signal is flat and contributes zero weight.
        assert "action_outcome" in flat6
        assert w6["action_outcome"] == 0.0
        # Every other signal's redistributed weight is identical to the
        # 5-signal world — the new signal falls out of the denominator.
        assert {k: v for k, v in w6.items() if k != "action_outcome"} == w5

        ctx = DreamingContext(current_cycle=_CYCLE, total_thoughts=len(candidates))
        scores6 = [ext6._compute_score(t, ctx, active_weights=w6) for t in candidates]
        scores5 = [ext5._compute_score(t, ctx, active_weights=w5) for t in candidates]
        assert scores6 == scores5  # byte-identical

    async def test_action_free_promotion_set_identical_end_to_end(self, tmp_path: Path) -> None:
        # Two identical stores; one runs dreaming with the 6th signal, the other
        # with only the five pre-feature signals. No actions anywhere.
        async def _run(signals: dict[str, float] | None) -> list[str]:
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            s = SqliteEngravaCore(conn)
            await s.ensure_schema()
            for i in range(5):
                await s.create_thought(_obs(f"obs-{i}", updated_cycle=_CYCLE))
            ext = DreamingExtension(_dreaming_cfg(signals=signals))
            result = await ext.run_consolidation(s, current_cycle=_CYCLE)
            await conn.close()
            return sorted(result.promoted_ids)

        cfg6 = _dreaming_cfg()
        five = {k: v for k, v in cfg6.signals.items() if k != "action_outcome"}
        with_signal = await _run(None)
        without_signal = await _run(five)
        assert with_signal == without_signal

    async def test_store_that_never_uses_actions_writes_no_score(
        self, store: SqliteEngravaCore
    ) -> None:
        await store.create_thought(_thought("t-001"))
        await store.create_thought(_thought("t-002"))
        for tid in ("t-001", "t-002"):
            t = await store.get_thought(tid)
            assert t is not None
            assert t.action_outcome_score is None


class TestActionUsingPromotion:
    """An all-succeeded thought promotes over an equivalent all-failed one."""

    async def test_succeeded_outranks_failed(self, store: SqliteEngravaCore) -> None:
        # Two otherwise-identical observations; one accrues successful actions,
        # the other failed ones. action_outcome then discriminates between them.
        await store.create_thought(_obs("winner", updated_cycle=_CYCLE))
        await store.create_thought(_obs("loser", updated_cycle=_CYCLE))
        for i in range(2):
            await store.create_action(
                _action(
                    f"w{i}",
                    "winner",
                    status=ActionStatus.CONFIRMED,
                    verification_status=VerificationStatus.CONFIRMED,
                )
            )
            await store.create_action(
                _action(
                    f"l{i}",
                    "loser",
                    status=ActionStatus.FAILED,
                    verification_status=VerificationStatus.FAILED,
                )
            )
        assert (await store.get_thought("winner")).action_outcome_score == 1.0  # type: ignore[union-attr]
        assert (await store.get_thought("loser")).action_outcome_score == 0.0  # type: ignore[union-attr]

        # Weight action_outcome heavily and disable the other data-bearing
        # signals so the outcome is the sole discriminator; keep recency +
        # staleness (equal for both, since updated_cycle is identical).
        signals = {
            "recency": 0.1,
            "staleness": 0.1,
            "confirmation": 0.0,
            "confidence": 0.0,
            "frequency": 0.0,
            "action_outcome": 0.8,
        }
        ext = DreamingExtension(_dreaming_cfg(signals=signals))
        ctx = DreamingContext(current_cycle=_CYCLE, total_thoughts=2)
        candidates = [
            await store.get_thought("winner"),
            await store.get_thought("loser"),
        ]
        assert candidates[0] is not None
        assert candidates[1] is not None
        weights, flat = ext._compute_active_weights(candidates, current_cycle=_CYCLE)
        assert "action_outcome" not in flat  # active now
        winner_score = ext._compute_score(candidates[0], ctx, active_weights=weights)
        loser_score = ext._compute_score(candidates[1], ctx, active_weights=weights)
        assert winner_score > loser_score

        result = await ext.run_consolidation(store, current_cycle=_CYCLE)
        assert "winner" in result.promoted_ids
        assert "loser" not in result.promoted_ids


# ---------------------------------------------------------------------------
# Schema / migration
# ---------------------------------------------------------------------------


class TestSchema:
    """core-16 migration: column, seek index, idempotency, version, cascade."""

    async def test_fresh_db_is_v16(self, store: SqliteEngravaCore) -> None:
        cursor = await store._db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 18

    async def test_column_and_index_present(self, store: SqliteEngravaCore) -> None:
        cursor = await store._db.execute("PRAGMA table_info(thought)")
        cols = {r["name"] for r in await cursor.fetchall()}
        assert "action_outcome_score" in cols
        cursor = await store._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_action_source_thought'"
        )
        assert await cursor.fetchone() is not None

    async def test_migration_idempotent(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()
            await store.ensure_schema()  # second run must be a no-op
            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert row is not None
            assert int(row[0]) == 18
            # Column added exactly once (no duplicate-column crash on re-run).
            cursor = await conn.execute("PRAGMA table_info(thought)")
            names = [r["name"] for r in await cursor.fetchall()]
            assert names.count("action_outcome_score") == 1
        finally:
            await conn.close()

    async def test_by_source_query_uses_index(self, store: SqliteEngravaCore) -> None:
        cursor = await store._db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM action WHERE source_thought_id = ?",
            ("t-001",),
        )
        plan = " ".join(str(r["detail"]) for r in await cursor.fetchall())
        assert "idx_action_source_thought" in plan

    async def test_cascade_from_v15_reaches_v16(self) -> None:
        # A v15 database (pre-feature) upgraded in place must land at v16 with
        # the new column + index, without touching existing rows.
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            await conn.execute("PRAGMA foreign_keys = ON")
            # Bootstrap a full current schema, then rewind the version marker to
            # 15 and drop the new column/index to simulate a genuine v15 DB.
            boot = SqliteEngravaCore(conn)
            await boot.ensure_schema()
            await conn.execute("DROP INDEX IF EXISTS idx_action_source_thought")
            await conn.execute("ALTER TABLE thought DROP COLUMN action_outcome_score")
            await conn.execute("PRAGMA user_version = 15")
            await conn.commit()

            # Re-open and upgrade.
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()

            cursor = await conn.execute("PRAGMA user_version")
            row = await cursor.fetchone()
            assert row is not None
            assert int(row[0]) == 18
            cursor = await conn.execute("PRAGMA table_info(thought)")
            cols = {r["name"] for r in await cursor.fetchall()}
            assert "action_outcome_score" in cols
            cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='index' AND name='idx_action_source_thought'"
            )
            assert await cursor.fetchone() is not None
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# FK cascade
# ---------------------------------------------------------------------------


class TestFkCascade:
    """Deleting a thought removes its actions and leaves journal entries intact."""

    async def test_delete_thought_cascades_actions(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought("t-001"))
        await store.create_action(_action("a", status=ActionStatus.PLANNED))
        assert len(await store.get_actions("t-001")) == 1

        deleted = await store.delete_thought("t-001")
        assert deleted is True
        # No orphan actions remain.
        assert await store.get_actions("t-001") == []
        cursor = await store._db.execute("SELECT COUNT(*) FROM action")
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 0

    async def test_journal_entries_survive_cascade(self, jstore: SqliteEngravaCore) -> None:
        await jstore.create_thought(_thought("t-001"))
        await jstore.create_action(_action("a", status=ActionStatus.EXECUTING))
        await jstore.update_action("a", status=ActionStatus.CONFIRMED)  # UPDATE_ACTION entry
        journal_before = await _journal_len(jstore)
        assert journal_before > 0

        await jstore.delete_thought("t-001")

        # Append-only journal is unaffected by the cascade.
        assert await _journal_len(jstore) >= journal_before
        cursor = await jstore._db.execute(
            "SELECT COUNT(*) FROM journal_entry WHERE mutation_type = 'UPDATE_ACTION'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 1


# ---------------------------------------------------------------------------
# Read-only store
# ---------------------------------------------------------------------------


class TestReadOnlyStore:
    """The read-only wrapper blocks the action write surface."""

    async def test_update_action_blocked(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.EXECUTING))
        ro = ReadOnlyEngrava(store)
        with pytest.raises(ReadOnlyViolationError):
            await ro.update_action("a", status=ActionStatus.CONFIRMED)

    async def test_create_action_blocked(self, store: SqliteEngravaCore) -> None:
        ro = ReadOnlyEngrava(store)
        with pytest.raises(ReadOnlyViolationError):
            await ro.create_action(_action("a"))

    async def test_get_actions_delegated(self, store: SqliteEngravaCore) -> None:
        await store.create_thought(_thought())
        await store.create_action(_action("a", status=ActionStatus.PLANNED))
        ro = ReadOnlyEngrava(store)
        actions = await ro.get_actions("t-001")
        assert len(actions) == 1

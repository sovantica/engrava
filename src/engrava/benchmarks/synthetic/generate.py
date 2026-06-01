"""Deterministic conversation + question generator for the synthetic benchmark.

The generator is a pure function of its inputs: same ``seed`` plus
same generation parameters produce byte-identical output across
machines, Python versions and OS line-ending settings.  This is the
foundation of the public reproducibility commitment — the
committed ``synthetic-v1.json`` MUST equal what
``generate_dataset(seed=20260508, ...)`` produces at any later point
in time, and the ``test_frozen_synthetic_v1_unchanged`` guard
enforces that invariant on every CI run.

Determinism strategy
--------------------

The top-level :class:`random.Random` instance picks scenarios and
seeds the per-conversation sub-RNGs.  Each conversation then runs on
its own :class:`random.Random` derived from a 64-bit integer the
top-level RNG produces.  This isolates parameter choices inside a
single conversation from the global stream, so adding or removing
distractions inside one conversation does not perturb the slot
values picked for the next conversation.

Scenario weighting
------------------

When ``scenario_mix`` is ``None`` (the default) the generator picks
scenarios uniformly at random, then truncates to ``n_conversations``.
For the v0.3.x line we rely on that uniform draw to give roughly
proportional representation; the empirical mix is recorded in
``EvaluationResult.per_scenario`` so we can monitor any drift across
release rebuilds.  Passing an explicit ``scenario_mix`` dict turns
the picker into a weighted choice — used in tests to build
anti-cherry-pick subsets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from random import Random
from typing import TYPE_CHECKING, Literal

from engrava.benchmarks.synthetic.scenarios import (
    SCENARIO_LIBRARY,
    Scenario,
    ScenarioDifficulty,
    ThemeBundle,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "Perspective",
    "SyntheticConversation",
    "SyntheticQuestion",
    "SyntheticTurn",
    "dataset_to_json",
    "generate_dataset",
]


Perspective = Literal["percept", "utterance", "thought"]


# ---------------------------------------------------------------------------
# Data shapes — frozen dataclasses, serialised as plain dicts with tuples
# normalised to lists so JSON round-trips byte-stable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticTurn:
    """One turn in a synthetic conversation.

    Attributes:
        turn_index: Zero-based position of the turn within its
            conversation.
        perspective: One of ``"percept"``,
            ``"utterance"`` or ``"thought"``.  Memorable facts are
            always ``"utterance"`` (user statements); distractions
            split between perspectives to vary the corpus.
        source_is_self: ``True`` for the agent's
            own thoughts, ``False`` for external content.  Memorable
            facts and most distractions are user utterances
            (``True``); a fraction of distractions are external
            percepts (``False``).
        text: The plain-text content of the turn.
        is_memorable: ``True`` when this turn carries one of the
            scenario's memorable facts.  Distractions are always
            ``False``.
        fact_id: Stable identifier for the memorable fact this turn
            carries.  ``None`` for distractions.  The evaluator maps
            ``SyntheticQuestion.expected_fact_ids`` to the
            ``ThoughtRecord`` IDs created when this turn is ingested.

    """

    turn_index: int
    perspective: Perspective
    source_is_self: bool
    text: str
    is_memorable: bool
    fact_id: str | None


@dataclass(frozen=True)
class SyntheticQuestion:
    """A recall question keyed against a conversation's memorable facts.

    Attributes:
        question_id: Stable identifier, unique across the dataset.
        scenario_name: Name of the originating scenario.  Used by the
            evaluator's per-scenario breakdown.
        difficulty: Difficulty tag copied from the source scenario at
            generation time.  Carried on the question itself so the
            frozen JSON dataset and the downstream evaluator both
            remain self-sufficient without re-keying through the
            scenario library at read time.
        asked_at_turn: The conversation turn after which the question
            is asked.  Higher than every memorable fact's turn_index
            by at least the scenario's ``question_offset_min_turns``.
        question_text: The natural-language question.
        expected_fact_ids: Tuple of ``fact_id`` values whose
            corresponding thoughts should be retrieved.  A retrieval
            counts as ``recall@K`` hit when any expected fact_id maps
            to a thought returned in the top-K.
        expected_substrings: Substrings any retrieved thought's
            content should contain for a positive substring-match.
            Orthogonal scoring axis from ``expected_fact_ids``.

    """

    question_id: str
    scenario_name: str
    difficulty: ScenarioDifficulty
    asked_at_turn: int
    question_text: str
    expected_fact_ids: tuple[str, ...]
    expected_substrings: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticConversation:
    """A full synthetic conversation with its recall questions.

    Attributes:
        conversation_id: Stable identifier — ``synth-conv-NNNN`` with
            zero-padded index.
        scenario_name: Name of the source scenario.
        turns: Ordered turns; ``turn_index`` matches the position in
            this tuple.
        questions: Recall questions for this conversation.  Each
            question's ``asked_at_turn`` is greater than any
            memorable turn's index.

    """

    conversation_id: str
    scenario_name: str
    turns: tuple[SyntheticTurn, ...]
    questions: tuple[SyntheticQuestion, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dataset(
    *,
    seed: int,
    n_conversations: int,
    avg_turns_per_conversation: int,
    distraction_density: float,
    scenario_mix: Mapping[str, float] | None = None,
) -> tuple[SyntheticConversation, ...]:
    """Deterministically generate a synthetic benchmark dataset.

    Args:
        seed: Master seed for the top-level RNG.  Same ``seed`` and
            same parameters reproduce the dataset byte-for-byte.
        n_conversations: Number of conversations to generate.
            Must be positive.
        avg_turns_per_conversation: Target conversation length.  The
            generator places memorable facts and the recall question
            so the total turn count lands at this value; if the
            scenario's question-offset window is incompatible the
            conversation grows past the target rather than the
            scenario being silently dropped.
        distraction_density: Probability that any given non-memorable
            slot emits a distraction turn.  At ``1.0`` every gap is
            filled (the densest possible corpus); at ``0.0`` only
            memorable turns plus the forced tail-fill leading up to
            the question land in the conversation.  Applied as a
            per-slot Bernoulli gate seeded by the conversation RNG,
            so the same seed and density produce byte-identical
            output.  Validated to lie in ``[0.0, 1.0]``.
        scenario_mix: Optional mapping from scenario name to weight.
            When ``None`` the generator picks scenarios uniformly.
            Unknown names raise :class:`ValueError`; the weights are
            normalised internally so callers can pass raw counts.

    Returns:
        Tuple of generated conversations, length ``n_conversations``.

    Raises:
        ValueError: If any parameter is out of range or
            ``scenario_mix`` references an unknown scenario name.

    """
    _validate_inputs(
        n_conversations=n_conversations,
        avg_turns_per_conversation=avg_turns_per_conversation,
        distraction_density=distraction_density,
    )
    scenario_picker = _build_scenario_picker(scenario_mix)

    # ``Random(seed)`` is the right tool here — we explicitly want
    # reproducible, non-cryptographic pseudo-randomness as the
    # foundation of the frozen-dataset commitment.  Cryptographic RNGs
    # would defeat the byte-identity contract.
    rng = Random(seed)  # noqa: S311 -- determinism is the goal, not security.
    conversations: list[SyntheticConversation] = []
    for conv_idx in range(n_conversations):
        scenario = scenario_picker(rng)
        sub_seed = rng.getrandbits(64)
        conv_rng = Random(sub_seed)  # noqa: S311 -- same reason as the top-level RNG.
        conv = _generate_conversation(
            seed=seed,
            conv_idx=conv_idx,
            scenario=scenario,
            target_turns=avg_turns_per_conversation,
            distraction_density=distraction_density,
            rng=conv_rng,
        )
        conversations.append(conv)

    return tuple(conversations)


def _turn_marker(seed: int, conv_idx: int, turn_idx: int) -> str:
    """Return a deterministic, naturalistic per-turn suffix.

    The marker breaks byte-identical content collisions across
    conversations that emit the same distraction or facet template,
    without poisoning the embedding-space semantics with random hex
    blobs.  The shape ``[note <6-hex>]`` is human-readable enough
    that the cosine similarity of two distraction lines drops below
    the cluster-similarity threshold for the *content body* alone,
    while their actual textual semantics still cluster naturally
    inside one conversation.

    Derivation: SHA-256 of ``"{seed}:{conv_idx}:{turn_idx}"`` ->
    first 6 hex chars.  Pure function of ``(seed, conv_idx,
    turn_idx)`` so the frozen dataset stays byte-identical across
    re-generations from the same seed.

    Args:
        seed: Master seed for the dataset.  Threaded through from
            ``generate_dataset(seed=...)``.
        conv_idx: Zero-based conversation index.
        turn_idx: Pre-density turn position (the position the slot
            occupies in the conversation's provisional layout before
            distraction density skips re-number the surviving
            turns).  Using the pre-density index keeps the marker
            stable for memorable turns whose position is determined
            before density resolves.

    Returns:
        ``"[note <6-hex>]"``.

    """
    payload = f"{seed}:{conv_idx}:{turn_idx}"
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    return f"[note {digest[:6]}]"


def dataset_to_json(
    conversations: tuple[SyntheticConversation, ...],
    *,
    indent: int = 2,
) -> str:
    r"""Serialise a dataset to a deterministic JSON string.

    Tuples are normalised to lists (JSON has no tuple type) and
    ``None`` is preserved.  The output is sorted by key, ASCII-only
    and uses ``\n`` line endings — same content on Windows, macOS
    and Linux.  Callers writing to a file MUST open the file with
    ``newline=""`` (or ``newline="\n"``) to avoid OS-level CRLF
    translation, which would defeat byte-identity.

    Args:
        conversations: Output of :func:`generate_dataset`.
        indent: JSON indent width.  Defaults to ``2``.

    Returns:
        Deterministic JSON string.

    """
    payload = [asdict(c) for c in conversations]
    return json.dumps(
        payload,
        sort_keys=True,
        indent=indent,
        ensure_ascii=True,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_inputs(
    *,
    n_conversations: int,
    avg_turns_per_conversation: int,
    distraction_density: float,
) -> None:
    if n_conversations <= 0:
        msg = f"n_conversations must be positive; got {n_conversations}"
        raise ValueError(msg)
    if avg_turns_per_conversation <= 0:
        msg = f"avg_turns_per_conversation must be positive; got {avg_turns_per_conversation}"
        raise ValueError(msg)
    if not 0.0 <= distraction_density <= 1.0:
        msg = f"distraction_density must lie in [0.0, 1.0]; got {distraction_density}"
        raise ValueError(msg)


@dataclass(frozen=True)
class _ScenarioPicker:
    """Callable wrapper bundling weighted-choice population + weights."""

    population: tuple[Scenario, ...]
    weights: tuple[float, ...] = field(default_factory=tuple)

    def __call__(self, rng: Random) -> Scenario:
        if not self.weights:
            return rng.choice(self.population)
        return rng.choices(self.population, weights=list(self.weights), k=1)[0]


def _build_scenario_picker(
    scenario_mix: Mapping[str, float] | None,
) -> _ScenarioPicker:
    if scenario_mix is None:
        return _ScenarioPicker(population=SCENARIO_LIBRARY)

    by_name = {s.name: s for s in SCENARIO_LIBRARY}
    unknown = set(scenario_mix.keys()) - by_name.keys()
    if unknown:
        msg = f"scenario_mix references unknown scenarios: {sorted(unknown)}"
        raise ValueError(msg)
    if not scenario_mix:
        msg = "scenario_mix must be non-empty when provided"
        raise ValueError(msg)

    # Stable iteration order: SCENARIO_LIBRARY order, filtered to mix keys.
    ordered = tuple(s for s in SCENARIO_LIBRARY if s.name in scenario_mix)
    weights = tuple(float(scenario_mix[s.name]) for s in ordered)
    if any(w < 0 for w in weights):
        msg = f"scenario_mix weights must be non-negative; got {weights}"
        raise ValueError(msg)
    if sum(weights) == 0.0:
        msg = "scenario_mix weights sum to zero"
        raise ValueError(msg)
    return _ScenarioPicker(population=ordered, weights=weights)


def _generate_conversation(
    *,
    seed: int,
    conv_idx: int,
    scenario: Scenario,
    target_turns: int,
    distraction_density: float,
    rng: Random,
) -> SyntheticConversation:
    """Build one conversation by laying down memorable facts then filling gaps."""
    conversation_id = f"synth-conv-{conv_idx:04d}"

    # Synthesis-path branch — pick one bundle per conversation and
    # emit every facet as a planted turn.  Distinguished from the
    # direct path by ``scenario.theme_bundles`` being non-None.
    bundle: ThemeBundle | None = None
    if scenario.theme_bundles is not None:
        bundle = rng.choice(list(scenario.theme_bundles))
        # The number of planted turns equals the bundle's facet count
        # (overrides ``scenario.n_memorable_facts`` which is the direct
        # path's default).
        n_facts = len(bundle.facets)
    else:
        n_facts = scenario.n_memorable_facts

    # Step 1 — pick one concrete value per slot for this conversation.
    # Direct path uses these to render templates; synthesis path keeps
    # the slot values around only for distraction rendering (memorable
    # facets are taken verbatim from the bundle).
    slot_values: dict[str, str] = {
        slot_name: rng.choice(list(candidates))
        for slot_name, candidates in scenario.slot_vocabulary.items()
    }

    # Step 2 — place memorable facts.  First memorable lives somewhere
    # in the first quarter of the conversation; subsequent memorable
    # facts (multi-fact or synthesis scenarios) follow at small gaps.
    memorable_turns = _allocate_memorable_turns(
        n_facts=n_facts,
        target_turns=target_turns,
        rng=rng,
    )

    # Step 3 — render memorable-fact turns.  Synthesis path takes
    # facets verbatim; direct path substitutes into templates.
    memorable_records: list[tuple[int, SyntheticTurn]] = []
    if bundle is not None:
        fact_texts: tuple[str, ...] = bundle.facets
    else:
        fact_texts = tuple(
            _render(template, slot_values) for template in scenario.memorable_templates
        )
    for fact_idx, (turn_position, base_text) in enumerate(
        zip(memorable_turns, fact_texts, strict=True),
    ):
        fact_id = f"{scenario.name}-conv{conv_idx:04d}-fact{fact_idx}"
        # Append the deterministic per-turn marker.  Two conversations
        # that pick the same theme bundle (synthesis path) or the same
        # slot values (direct path) would otherwise emit byte-identical
        # memorable text, which trips the cluster-quality
        # ``has_duplicate_content_members`` gate and rejects the whole
        # cross-conversation synthesis cluster.  The marker is
        # naturalistic enough not to dominate the embedding-space
        # similarity that drives clustering of related facets.
        text = f"{base_text} {_turn_marker(seed, conv_idx, turn_position)}"
        memorable_records.append(
            (
                turn_position,
                SyntheticTurn(
                    turn_index=turn_position,
                    perspective="utterance",
                    source_is_self=True,
                    text=text,
                    is_memorable=True,
                    fact_id=fact_id,
                ),
            ),
        )

    # Step 4 — draw the question offset in dense (post-density) turn
    # coordinates.  The question sits within the scenario's offset
    # window past the LAST memorable turn, measured after distraction
    # density has been applied.
    offset = rng.randint(
        scenario.question_offset_min_turns,
        scenario.question_offset_max_turns,
    )

    # Step 5 — provisional pre-density length.  This is the upper
    # bound on the number of positions we sample from; the density
    # gate may emit fewer turns, in which case the conversation gets
    # extended with forced distractions below to make room for the
    # question.
    last_memorable_position = memorable_turns[-1]
    provisional_total = max(target_turns, last_memorable_position + offset + 1)

    # Step 6 — fill positions in pre-density coordinates, applying
    # the density gate to every non-memorable slot.  Memorable turns
    # ALWAYS emit (they carry the planted facts); non-memorable slots
    # emit with probability ``distraction_density``.  Crucially the
    # RNG is consumed even when the slot is skipped so the question
    # offset draw above stays uncorrelated with density.
    memorable_by_position = dict(memorable_records)
    turns: list[SyntheticTurn] = []
    last_memorable_dense_index = -1
    for position in range(provisional_total):
        if position in memorable_by_position:
            mem_turn = memorable_by_position[position]
            last_memorable_dense_index = len(turns)
            turns.append(
                # Re-stamp the dense index — the original turn was
                # built with the pre-density position which is no
                # longer meaningful after density skips.
                SyntheticTurn(
                    turn_index=last_memorable_dense_index,
                    perspective=mem_turn.perspective,
                    source_is_self=mem_turn.source_is_self,
                    text=mem_turn.text,
                    is_memorable=mem_turn.is_memorable,
                    fact_id=mem_turn.fact_id,
                ),
            )
            continue
        # Always consume one RNG draw so the stream is stable wrt
        # density value (a later code path may want to add a second
        # draw inside ``_render_distraction``; the density branch
        # owns the first one).
        keep = rng.random() < distraction_density
        if not keep:
            continue
        turns.append(
            _render_distraction(
                position=len(turns),
                scenario=scenario,
                slot_values=slot_values,
                seed=seed,
                conv_idx=conv_idx,
                rng=rng,
            ),
        )

    # Step 7 — extend if the question would otherwise land past the
    # end.  When density is low the loop above may have skipped most
    # tail distractions, leaving the conversation shorter than
    # ``last_memorable_dense_index + offset + 1``.  Force-fill the
    # gap with distractions (no density gate, no new RNG draws for
    # the gate itself) so the question always has room.
    asked_at_turn = last_memorable_dense_index + offset
    while len(turns) <= asked_at_turn:
        turns.append(
            _render_distraction(
                position=len(turns),
                scenario=scenario,
                slot_values=slot_values,
                seed=seed,
                conv_idx=conv_idx,
                rng=rng,
            ),
        )

    # Step 7 — build questions (one per question_template, sharing
    # ``asked_at_turn`` and the multi-fact's expected_fact_ids list).
    questions = _build_questions(
        scenario=scenario,
        bundle=bundle,
        conversation_id=conversation_id,
        asked_at_turn=asked_at_turn,
        slot_values=slot_values,
        memorable_records=memorable_records,
    )

    return SyntheticConversation(
        conversation_id=conversation_id,
        scenario_name=scenario.name,
        turns=tuple(turns),
        questions=questions,
    )


def _allocate_memorable_turns(
    *,
    n_facts: int,
    target_turns: int,
    rng: Random,
) -> tuple[int, ...]:
    """Choose turn indices for the memorable facts.

    The first memorable lives in the first quarter of the
    conversation; subsequent memorable turns follow at 1-3 turn
    gaps.  The caller ensures the total conversation length grows to
    accommodate the question turn, so the per-fact placement here
    does not need to bound-check against the question offset.
    """
    first_quarter_end = max(1, target_turns // 4)
    first_position = rng.randint(0, first_quarter_end - 1)
    positions = [first_position]
    for _ in range(n_facts - 1):
        gap = rng.randint(1, 3)
        positions.append(positions[-1] + gap)
    return tuple(positions)


def _render(template: str, slot_values: Mapping[str, str]) -> str:
    """Substitute ``{slot}`` placeholders.

    Uses :meth:`str.format_map` rather than ``str.format`` so the
    full ``Mapping`` protocol is exercised; an unknown placeholder
    surfaces as :class:`KeyError` and bubbles to the caller, which
    is what we want — the scenario-shape tests guarantee every
    placeholder has a vocabulary entry.
    """
    return template.format_map(slot_values)


def _render_distraction(
    *,
    position: int,
    scenario: Scenario,
    slot_values: Mapping[str, str],
    seed: int,
    conv_idx: int,
    rng: Random,
) -> SyntheticTurn:
    """Render one distraction turn, deterministic in the conv RNG.

    The rendered template gets a deterministic per-turn marker
    suffix (see :func:`_turn_marker`) so distractions from different
    conversations that pick the same template + slot values do not
    end up byte-identical and trip the cluster-quality
    ``has_duplicate_content_members`` gate when dreaming tries to
    cluster them across conversations.
    """
    template = rng.choice(list(scenario.distraction_templates))
    base_text = _render(template, slot_values)
    text = f"{base_text} {_turn_marker(seed, conv_idx, position)}"
    # Vary perspective + provenance to keep the corpus realistic.
    # Roughly 70% utterance / 30% percept; perspective and is_self
    # correlate (percepts are external observations).
    if rng.random() < 0.7:  # noqa: PLR2004 -- intentional inline split.
        perspective: Perspective = "utterance"
        source_is_self = True
    else:
        perspective = "percept"
        source_is_self = False
    return SyntheticTurn(
        turn_index=position,
        perspective=perspective,
        source_is_self=source_is_self,
        text=text,
        is_memorable=False,
        fact_id=None,
    )


def _build_questions(
    *,
    scenario: Scenario,
    bundle: ThemeBundle | None,
    conversation_id: str,
    asked_at_turn: int,
    slot_values: Mapping[str, str],
    memorable_records: list[tuple[int, SyntheticTurn]],
) -> tuple[SyntheticQuestion, ...]:
    """Render the recall questions for a conversation.

    Each scenario has one or more question templates with a matching
    substring tuple at the same index.  When the scenario plants
    multiple memorable facts (multi-fact or synthesis), every question
    references *all* of them via ``expected_fact_ids`` — the evaluator
    counts recall@K against the union (plus the REFLECTION
    ``consolidated_from`` intersection on the synthesis path).

    The synthesis path uses ``bundle.question_text`` and
    ``bundle.expected_substrings`` verbatim — exactly one question per
    conversation, no slot rendering.  The direct path uses the
    scenario's template tuples.
    """
    all_fact_ids = tuple(turn.fact_id for _, turn in memorable_records if turn.fact_id)

    if bundle is not None:
        return (
            SyntheticQuestion(
                question_id=f"{conversation_id}-q0",
                scenario_name=scenario.name,
                difficulty=scenario.difficulty,
                asked_at_turn=asked_at_turn,
                question_text=bundle.question_text,
                expected_fact_ids=all_fact_ids,
                expected_substrings=bundle.expected_substrings,
            ),
        )

    questions: list[SyntheticQuestion] = []
    for q_idx, (q_template, substring_tuple) in enumerate(
        zip(
            scenario.question_templates,
            scenario.answer_substring_templates,
            strict=True,
        ),
    ):
        question_text = _render(q_template, slot_values)
        expected_substrings = tuple(_render(s, slot_values) for s in substring_tuple)
        questions.append(
            SyntheticQuestion(
                question_id=f"{conversation_id}-q{q_idx}",
                scenario_name=scenario.name,
                difficulty=scenario.difficulty,
                asked_at_turn=asked_at_turn,
                question_text=question_text,
                expected_fact_ids=all_fact_ids,
                expected_substrings=expected_substrings,
            ),
        )
    return tuple(questions)

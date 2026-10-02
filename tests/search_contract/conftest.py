"""Fixtures for the search functional-contract suite.

This module hand-authors a small, synthetic, conversational corpus and a set
of natural-language questions labelled with their gold-answer thought, then
exposes both through pytest fixtures together with a populated store.

Everything here is deterministic and network-free:

* The corpus is written by hand (no benchmark dataset is read), so it is safe
  to ship in a public repository.
* Query embeddings come from a deterministic bag-of-words hashing provider
  (:class:`BagOfWordsProvider`), so ``search_hybrid`` exercises a real vector
  arm without loading a model or reaching the network.

The corpus deliberately includes the inputs that purely line-coverage-driven
tests miss: long turns whose distinctive fact lives in the tail, contractions
and non-English clitics, a pasted URL, bare numbers/timestamps, and clusters
of near-duplicate same-topic turns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import CallbackProvider, SqliteEngravaCore
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Corpus data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusTurn:
    """A single synthetic conversational turn stored as a thought.

    Args:
        thought_id: Stable identifier used to assert retrieval.
        essence: Short summary line, indexed by FTS5.
        content: Full turn text, indexed by FTS5.
        distinctive_terms: One to three content words unique enough to find
            this turn. A findability query is built from these plus arbitrary
            function words.
        priority: Fusion priority signal. Defaults to ``P2`` — most turns keep
            the default; a subset below is given a different level so the
            priority signal actually varies across the fixture (see the comment
            above the discriminator pairs in :data:`_CORPUS`).
        created_cycle: Cognitive-cycle write time. Defaults to ``0``.
        updated_cycle: Cognitive-cycle last-touch time. Defaults to ``0``;
            staggered across most turns below so the recency signal has
            something to discriminate.
    """

    thought_id: str
    essence: str
    content: str
    distinctive_terms: tuple[str, ...] = field(default_factory=tuple)
    priority: Priority = Priority.P2
    created_cycle: int = 0
    updated_cycle: int = 0


@dataclass(frozen=True)
class CorpusEdge:
    """A directed graph edge between two corpus turns.

    Args:
        from_thought_id: Source turn id.
        to_thought_id: Target turn id.
        weight: Edge strength, ``[0.0, 1.0]``.
        edge_type: Relationship classification.
    """

    from_thought_id: str
    to_thought_id: str
    weight: float
    edge_type: EdgeType = EdgeType.ASSOCIATED


@dataclass(frozen=True)
class GoldQuestion:
    """A natural-language question paired with its gold-answer turn.

    Args:
        question: A user-style natural-language question, including function
            words ("what", "did", "my") that must not block a match.
        gold_thought_id: The ``thought_id`` of the turn that answers it.
    """

    question: str
    gold_thought_id: str


# ---------------------------------------------------------------------------
# The hand-authored corpus
# ---------------------------------------------------------------------------
# ~40 conversational turns (varied length, contractions, a URL, numbers/names,
# non-English samples, near-duplicate same-topic clusters) plus a handful of
# dedicated ranking-signal pairs at the end (see the comment above them). Each
# turn lists the distinctive content terms that should retrieve it.

_CORPUS: tuple[CorpusTurn, ...] = (
    CorpusTurn(
        "turn-job-marketing",
        "Career background",
        "Before this job I worked as a marketing specialist at a small startup downtown.",
        ("marketing", "specialist", "startup"),
        updated_cycle=50,
    ),
    CorpusTurn(
        "turn-sister-dog",
        "Family pet note",
        "My sister's dog is a golden retriever named Biscuit who hates thunderstorms.",
        ("retriever", "Biscuit", "thunderstorms"),
        updated_cycle=20,
    ),
    CorpusTurn(
        "turn-coffee-creamer",
        "Grocery coupon",
        "I redeemed a coupon on hazelnut coffee creamer at the corner store yesterday.",
        ("hazelnut", "creamer", "coupon"),
        updated_cycle=70,
    ),
    CorpusTurn(
        "turn-paris-trip",
        "Travel plan",
        "We are flying to Paris in October and staying near the Montmartre district.",
        ("Paris", "Montmartre", "October"),
        updated_cycle=10,
    ),
    CorpusTurn(
        "turn-guitar-lessons",
        "Hobby update",
        "I finally started weekly guitar lessons and I am learning fingerpicking now.",
        ("guitar", "fingerpicking", "lessons"),
        updated_cycle=90,
    ),
    CorpusTurn(
        "turn-docs-link",
        "Shared reference",
        "Here is the onboarding guide at https://docs.example.com/onboarding for new hires.",
        ("onboarding", "hires"),
        priority=Priority.P3,
        updated_cycle=15,
    ),
    CorpusTurn(
        "turn-budget-spreadsheet",
        "Finance task",
        "I updated the quarterly budget spreadsheet and the travel line is over by 1200 dollars.",
        ("budget", "spreadsheet", "quarterly"),
        priority=Priority.P1,
        updated_cycle=40,
    ),
    CorpusTurn(
        "turn-marathon-training",
        "Running goal",
        "My marathon training peaks next month with a brutal twenty-two mile long run.",
        ("marathon", "training"),
        priority=Priority.P3,
        updated_cycle=70,
    ),
    CorpusTurn(
        "turn-dentist-appointment",
        "Health reminder",
        "The dentist appointment got moved to Thursday because the hygienist was out sick.",
        ("dentist", "hygienist"),
        priority=Priority.P4,
        updated_cycle=5,
    ),
    CorpusTurn(
        "turn-recipe-lasagna",
        "Cooking note",
        "Grandma's lasagna recipe uses three cheeses and a slow simmered tomato ragu.",
        ("lasagna", "ragu", "cheeses"),
        priority=Priority.P1,
        updated_cycle=90,
    ),
    CorpusTurn(
        "turn-car-repair",
        "Vehicle issue",
        "The mechanic said the alternator is failing and the timing belt is due soon.",
        ("alternator", "mechanic"),
        updated_cycle=35,
    ),
    CorpusTurn(
        "turn-book-club",
        "Reading group",
        "Our book club picked a sprawling science fiction novel about generation ships.",
        ("generation", "ships", "novel"),
        priority=Priority.P4,
        updated_cycle=60,
    ),
    CorpusTurn(
        "turn-garden-tomatoes",
        "Gardening",
        "The heirloom tomatoes in the raised beds finally ripened after the heat wave.",
        ("heirloom", "tomatoes"),
        priority=Priority.P3,
        updated_cycle=25,
    ),
    CorpusTurn(
        "turn-flight-delay",
        "Travel mishap",
        "My connecting flight was delayed three hours so I missed the riverside dinner booking.",
        ("delayed", "riverside", "booking"),
        priority=Priority.P1,
        updated_cycle=80,
    ),
    CorpusTurn(
        "turn-new-laptop",
        "Purchase",
        "I bought a refurbished laptop with a mechanical keyboard and a matte display.",
        ("refurbished", "mechanical", "keyboard"),
        priority=Priority.P4,
        updated_cycle=10,
    ),
    CorpusTurn(
        "turn-yoga-class",
        "Wellness",
        "The new vinyasa yoga instructor pushes a punishing pace on Tuesday evenings.",
        ("vinyasa", "instructor"),
        updated_cycle=55,
    ),
    CorpusTurn(
        "turn-spanish-greeting",
        "Language practice",
        "Mi hermano vive en Sevilla y trabaja como arquitecto cerca del río.",
        ("hermano", "Sevilla", "arquitecto"),
        priority=Priority.P3,
        updated_cycle=35,
    ),
    CorpusTurn(
        "turn-french-school",
        "Language practice",
        "L'école française du quartier ferme ses portes pendant les vacances d'été.",
        ("française", "quartier"),
        priority=Priority.P1,
        updated_cycle=95,
    ),
    CorpusTurn(
        "turn-german-train",
        "Language practice",
        "Der Zug nach München war pünktlich und überraschend leer am Sonntagmorgen.",
        ("München", "Sonntagmorgen"),
        priority=Priority.P4,
        updated_cycle=20,
    ),
    CorpusTurn(
        "turn-promotion",
        "Work milestone",
        "I got promoted to staff engineer and now I lead the payments reliability squad.",
        ("promoted", "payments", "reliability"),
        updated_cycle=45,
    ),
    CorpusTurn(
        "turn-long-conference",
        "Conference recap",
        (
            "The three day conference opened with a sleepy keynote and an endless hallway "
            "of vendor booths handing out the usual stickers and stress balls, and most of "
            "the morning talks rehashed material everyone already knew, but the very last "
            "lightning talk of the final afternoon was given by a researcher named "
            "Okonkwo who quietly demonstrated a lossless compression trick for vector "
            "indexes that nobody in the room had seen before."
        ),
        ("Okonkwo", "compression"),
        updated_cycle=60,
    ),
    CorpusTurn(
        "turn-long-roadtrip",
        "Road trip diary",
        (
            "We left before dawn and the first six hours were nothing but flat farmland and "
            "gas station coffee, then a long stretch of construction near the state line "
            "that crawled for ages, and we almost gave up on the detour, but right at "
            "sunset we crested a ridge and found a tiny roadside diner called the "
            "Larkspur whose blueberry pie turned the entire miserable drive into the best "
            "day of the trip."
        ),
        ("Larkspur", "blueberry"),
        updated_cycle=25,
    ),
    CorpusTurn(
        "turn-long-meeting",
        "Standup overflow",
        (
            "Standup ran long again because everyone relitigated the deployment incident "
            "from last week and then drifted into a tangent about whether to switch issue "
            "trackers, and after twenty minutes of circular debate that nobody wrote down, "
            "the only real decision was buried at the end when Priya volunteered to own "
            "the flaky integration test that has blocked the release pipeline for days."
        ),
        ("Priya", "flaky"),
        updated_cycle=80,
    ),
    # Near-duplicate cluster: the office plant, three slightly different tellings.
    # Also carries an edge (turn-plant-a -> turn-plant-b, see _CORPUS_EDGES) so
    # the graph signal has a second, non-isolated case beyond the dedicated pair.
    CorpusTurn(
        "turn-plant-a",
        "Office plant",
        "The office fiddle leaf fig is dropping leaves again near the drafty window.",
        ("fiddle", "fig"),
        priority=Priority.P2,
        updated_cycle=30,
    ),
    CorpusTurn(
        "turn-plant-b",
        "Office plant note",
        "Someone overwatered the office fiddle leaf fig and now its leaves are yellowing.",
        ("fiddle", "overwatered"),
        priority=Priority.P3,
        updated_cycle=65,
    ),
    CorpusTurn(
        "turn-plant-c",
        "Office plant update",
        "We moved the office fiddle leaf fig away from the window and it perked up.",
        ("fiddle", "perked"),
        priority=Priority.P1,
        updated_cycle=45,
    ),
    # Near-duplicate cluster: the standing desk, two tellings. Also carries an
    # edge (turn-desk-a -> turn-desk-b, see _CORPUS_EDGES).
    CorpusTurn(
        "turn-desk-a",
        "Ergonomics",
        "My new standing desk wobbles slightly when it is raised to the tallest setting.",
        ("standing", "wobbles"),
        updated_cycle=5,
    ),
    CorpusTurn(
        "turn-desk-b",
        "Ergonomics follow-up",
        "I added felt pads under the standing desk feet and the wobble is mostly gone.",
        ("felt", "pads"),
        priority=Priority.P4,
        updated_cycle=85,
    ),
    CorpusTurn(
        "turn-podcast",
        "Media recommendation",
        "A friend recommended a history podcast about the cartography of medieval trade routes.",
        ("cartography", "medieval"),
        updated_cycle=50,
    ),
    CorpusTurn(
        "turn-allergy",
        "Health note",
        "My seasonal ragweed allergy flared up so I switched to a non drowsy antihistamine.",
        ("ragweed", "antihistamine"),
        priority=Priority.P3,
        updated_cycle=5,
    ),
    CorpusTurn(
        "turn-camera",
        "Photography",
        "I rented a wide angle lens for the canyon shoot and the dynamic range was stunning.",
        ("canyon", "lens"),
        updated_cycle=65,
    ),
    CorpusTurn(
        "turn-volunteer",
        "Community",
        "On Saturdays I volunteer at the riverbank cleanup and we filled forty trash bags.",
        ("riverbank", "cleanup"),
        priority=Priority.P1,
        updated_cycle=75,
    ),
    CorpusTurn(
        "turn-keyboard-don't",
        "Typing habit",
        "I don't use the number pad much so I switched to a compact tenkeyless keyboard.",
        ("tenkeyless",),
        priority=Priority.P4,
        updated_cycle=15,
    ),
    CorpusTurn(
        "turn-numbers-invoice",
        "Billing",
        "Invoice 4471 is still unpaid and the late fee kicks in after thirty days.",
        ("4471", "invoice"),
        updated_cycle=15,
    ),
    CorpusTurn(
        "turn-timestamp-meeting",
        "Calendar",
        "The retro is locked in for half past noon so block out that slot on the calendar.",
        ("retro",),
        updated_cycle=100,
    ),
    CorpusTurn(
        "turn-names-people",
        "Introductions",
        "At the offsite I finally met Nakamura from design and Olafsson from infrastructure.",
        ("Nakamura", "Olafsson"),
        updated_cycle=55,
    ),
    CorpusTurn(
        "turn-coffee-shop",
        "Routine",
        "The barista at the Wexford cafe remembers my oat milk cortado without me asking.",
        ("Wexford", "cortado"),
        priority=Priority.P3,
        updated_cycle=40,
    ),
    CorpusTurn(
        "turn-puzzle",
        "Leisure",
        "I am stuck on a thousand piece jigsaw of a lighthouse swallowed by fog.",
        ("jigsaw", "lighthouse"),
        priority=Priority.P1,
        updated_cycle=60,
    ),
    CorpusTurn(
        "turn-bike-commute",
        "Commute",
        "My bike commute got faster after they finally painted the protected lane on Birch Street.",
        ("Birch", "lane"),
        priority=Priority.P4,
        updated_cycle=30,
    ),
    CorpusTurn(
        "turn-houseplant-tip",
        "Advice received",
        "A neighbor told me bottom watering keeps the succulents from rotting at the crown.",
        ("succulents", "crown"),
        updated_cycle=90,
    ),
    # -----------------------------------------------------------------------
    # Dedicated ranking-signal discriminator pairs (these widen the frozen
    # baseline corpus so priority, cycle and graph each have a pair close
    # enough in fused score that perturbing that signal's own parameter
    # reorders them — see tests/search_contract/golden_fixtures.py and
    # scripts/regenerate_search_goldens.py).
    # -----------------------------------------------------------------------
    # Priority discriminator: byte-identical content (so FTS/vector/recency
    # tie exactly), differing ONLY in priority (P1 vs P4). At the shipped
    # priority weight the P1 turn outranks the P4 turn; zeroing the priority
    # weight reties them and the deterministic thought_id tiebreak flips the
    # order (turn-priority-control < turn-priority-target ascending).
    CorpusTurn(
        "turn-priority-control",
        "Aurora sighting",
        "The lighthouse keeper logged a rare aurora sighting over Kelso Sound at midnight.",
        ("aurora", "Kelso", "lighthouse"),
        priority=Priority.P4,
    ),
    CorpusTurn(
        "turn-priority-target",
        "Aurora sighting",
        "The lighthouse keeper logged a rare aurora sighting over Kelso Sound at midnight.",
        ("aurora", "Kelso", "lighthouse"),
        priority=Priority.P1,
    ),
    # Cycle discriminator: byte-identical content, but the fresher turn also
    # carries the LOWER priority (P4) and the staler turn the HIGHER priority
    # (P2) — a deliberate cycle x priority interaction, since a pure
    # recency-only difference can never be reordered by a half-life change
    # (exponential decay is monotonic in age for any positive half-life). At
    # the shipped half-life (50) freshness wins; widening the half-life
    # shrinks the recency gap until priority dominates and the order flips.
    CorpusTurn(
        "turn-cycle-control",
        "Quail migration survey",
        "Researchers began the quail migration route survey near the delta this spring.",
        ("quail", "migration", "delta"),
        priority=Priority.P2,
        updated_cycle=0,
    ),
    CorpusTurn(
        "turn-cycle-target",
        "Quail migration survey",
        "Researchers began the quail migration route survey near the delta this spring.",
        ("quail", "migration", "delta"),
        priority=Priority.P4,
        updated_cycle=100,
    ),
    # Graph discriminator: byte-identical content between control and target;
    # only the target is connected (see _CORPUS_EDGES) to a neighbour that
    # also matches the query. At the shipped edge decay the connected target
    # outranks the isolated control; zeroing the edge decay removes the boost
    # entirely, reties them, and the thought_id tiebreak flips the order
    # (turn-graph-control < turn-graph-target ascending).
    CorpusTurn(
        "turn-graph-control",
        "Night shift inspection",
        "The apprentice welder sparked a small fire drill during the night shift inspection.",
        ("welder", "apprentice", "inspection"),
    ),
    CorpusTurn(
        "turn-graph-target",
        "Night shift inspection",
        "The apprentice welder sparked a small fire drill during the night shift inspection.",
        ("welder", "apprentice", "inspection"),
    ),
    CorpusTurn(
        "turn-graph-neighbor",
        "Night shift follow-up",
        "During the night shift inspection the apprentice welder also flagged a faulty gauge.",
        ("gauge", "flagged", "faulty"),
    ),
)


# ---------------------------------------------------------------------------
# A few graph edges over the corpus above
# ---------------------------------------------------------------------------
# Inert by default: the graph signal only activates at a non-zero
# ``graph_weight`` (product default is ``0.0``, opt-in), so these edges cost
# every other test in this suite nothing. They exist so the frozen ranked
# golden — the one place that does opt in via an explicit ``graph_weight``
# override — has more than one graph case to freeze.

_CORPUS_EDGES: tuple[CorpusEdge, ...] = (
    # Near-duplicate office-plant retellings: a real thematic association.
    CorpusEdge("turn-plant-a", "turn-plant-b", weight=0.8),
    # Near-duplicate standing-desk retellings: same idea.
    CorpusEdge("turn-desk-a", "turn-desk-b", weight=0.8),
    # The dedicated graph-signal discriminator pair (see _CORPUS above): only
    # the target is connected to the neighbour, so it alone gets a boost.
    CorpusEdge("turn-graph-target", "turn-graph-neighbor", weight=1.0),
)


# ---------------------------------------------------------------------------
# Gold-labelled natural-language questions
# ---------------------------------------------------------------------------
# Each question is a realistic user query (with function words) whose answer is
# a single distinctive turn above.

_GOLD_QUESTIONS: tuple[GoldQuestion, ...] = (
    GoldQuestion("what did I say about the marketing specialist job", "turn-job-marketing"),
    GoldQuestion("what was the thing about my sister's dog", "turn-sister-dog"),
    GoldQuestion("did I mention the hazelnut coffee creamer coupon", "turn-coffee-creamer"),
    GoldQuestion("where are we staying on the Paris trip", "turn-paris-trip"),
    GoldQuestion("what kind of guitar lessons did I start", "turn-guitar-lessons"),
    GoldQuestion("what did the mechanic say about the alternator", "turn-car-repair"),
    GoldQuestion("who gave the compression talk at the conference", "turn-long-conference"),
    GoldQuestion("which diner had the blueberry pie on our road trip", "turn-long-roadtrip"),
    GoldQuestion("who volunteered to own the flaky integration test", "turn-long-meeting"),
    GoldQuestion("what role did I get promoted to", "turn-promotion"),
    GoldQuestion("which invoice is still unpaid", "turn-numbers-invoice"),
    GoldQuestion("who did I meet from design at the offsite", "turn-names-people"),
    GoldQuestion("what is wrong with my new standing desk", "turn-desk-a"),
    GoldQuestion("what lens did I rent for the canyon shoot", "turn-camera"),
)


# ---------------------------------------------------------------------------
# Deterministic embedding provider
# ---------------------------------------------------------------------------

_EMBED_DIM = 256


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric word tokens.

    Args:
        text: Arbitrary input text.

    Returns:
        Lowercase word tokens, with punctuation stripped.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _bag_of_words_embed(text: str) -> list[float]:
    """Embed text as an L2-normalized bag-of-words hashing vector.

    Each token is hashed to a single dimension and contributes a unit count
    there; the resulting vector is L2-normalized. Cosine similarity between two
    such vectors therefore grows with the fraction of shared vocabulary, which
    gives ``search_hybrid`` a deterministic, network-free semantic signal whose
    ranking is fully predictable from the words two texts share.

    Args:
        text: Input text to embed.

    Returns:
        An ``_EMBED_DIM``-length unit vector (all-zero only for empty text).
    """
    vector = [0.0] * _EMBED_DIM
    for token in _tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()  # noqa: S324
        index = int.from_bytes(digest[:4], "big") % _EMBED_DIM
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus() -> tuple[CorpusTurn, ...]:
    """Return the hand-authored synthetic conversational corpus.

    Returns:
        The immutable tuple of corpus turns.
    """
    return _CORPUS


@pytest.fixture
def gold_questions() -> tuple[GoldQuestion, ...]:
    """Return the gold-labelled natural-language questions.

    Returns:
        The immutable tuple of gold questions.
    """
    return _GOLD_QUESTIONS


def make_embedding_provider() -> CallbackProvider:
    """Build the deterministic bag-of-words embedding provider.

    Factored out of the ``embedding_provider`` fixture so the golden
    regeneration path (which runs outside pytest fixture scope) constructs a
    byte-identical provider — a golden is only trustworthy if it is produced by
    exactly the wiring the tests read it back against.

    Returns:
        A :class:`CallbackProvider` wrapping the network-free hashing embedder.
    """
    return CallbackProvider(
        callback=_bag_of_words_embed,
        dimension=_EMBED_DIM,
        model_name="bag-of-words-contract",
    )


@pytest.fixture
def embedding_provider() -> CallbackProvider:
    """Return a deterministic bag-of-words embedding provider.

    Returns:
        A :class:`CallbackProvider` wrapping the network-free hashing embedder.
    """
    return make_embedding_provider()


def _to_thought(turn: CorpusTurn) -> ThoughtRecord:
    """Build a stored thought from a corpus turn.

    Args:
        turn: The synthetic corpus turn.

    Returns:
        A fully populated :class:`ThoughtRecord` ready for ``create_thought``.
    """
    return ThoughtRecord(
        thought_id=turn.thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=turn.essence,
        content=turn.content,
        priority=turn.priority,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=turn.created_cycle,
        updated_cycle=turn.updated_cycle,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


def _to_edge(edge: CorpusEdge) -> EdgeRecord:
    """Build a stored edge from a corpus edge.

    Args:
        edge: The synthetic corpus edge.

    Returns:
        A fully populated :class:`EdgeRecord` ready for ``create_edge``.
    """
    return EdgeRecord(
        edge_id=f"edge-{edge.from_thought_id}-{edge.to_thought_id}",
        from_thought_id=edge.from_thought_id,
        to_thought_id=edge.to_thought_id,
        edge_type=edge.edge_type,
        weight=edge.weight,
        created_cycle=0,
        source=KnowledgeSource.EXPERIENCE,
    )


async def populate_corpus(store: SqliteEngravaCore) -> None:
    """Write the synthetic corpus (thoughts, then edges) into a store.

    Factored out of :func:`open_populated_store` so any store-construction
    path — direct construction here, or ``SqliteEngravaCore.from_config`` in
    the golden regeneration entry point — writes byte-identical data. Edges
    are created after every thought so the foreign-key-backed endpoints
    already exist.

    Args:
        store: An already-schema'd, otherwise-empty store.
    """
    for turn in _CORPUS:
        await store.create_thought(_to_thought(turn))
    for edge in _CORPUS_EDGES:
        await store.create_edge(_to_edge(edge))


async def open_populated_store(
    *,
    embedding_provider: CallbackProvider | None = None,
    auto_embed: bool = False,
) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
    """Open an in-memory store populated with the synthetic corpus.

    Single store-construction path shared by the ``fts_store`` / ``hybrid_store``
    fixtures and the golden regeneration entry point, so a regenerated golden is
    produced from byte-identical setup to what the tests read it back against
    (no drift between the fixture and the generator).

    Args:
        embedding_provider: Optional deterministic provider. ``None`` leaves the
            vector arm dormant (FTS-only store).
        auto_embed: When ``True``, every stored thought is embedded on write so
            the vector arm is live for ``search_hybrid``.

    Returns:
        The populated store together with its owning connection; the caller is
        responsible for closing the connection.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(
        conn,
        embedding_provider=embedding_provider,
        auto_embed=auto_embed,
    )
    await store.ensure_schema()
    await populate_corpus(store)
    return store, conn


@pytest.fixture
async def fts_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return a store populated with the corpus, FTS-only (no embeddings).

    Yields:
        A :class:`SqliteEngravaCore` whose FTS5 index holds every corpus turn.
    """
    store, conn = await open_populated_store()
    yield store
    await conn.close()


@pytest.fixture
async def hybrid_store(
    embedding_provider: CallbackProvider,
) -> AsyncIterator[SqliteEngravaCore]:
    """Return a store populated with the corpus and a deterministic vector arm.

    Args:
        embedding_provider: The network-free bag-of-words provider.

    Yields:
        A :class:`SqliteEngravaCore` with ``auto_embed`` enabled so both the
        FTS arm and the vector arm are live for ``search_hybrid``.
    """
    store, conn = await open_populated_store(
        embedding_provider=embedding_provider,
        auto_embed=True,
    )
    yield store
    await conn.close()

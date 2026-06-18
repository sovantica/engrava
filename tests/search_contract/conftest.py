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
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
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
    """

    thought_id: str
    essence: str
    content: str
    distinctive_terms: tuple[str, ...] = field(default_factory=tuple)


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
# ~40 turns: varied length, contractions, a URL, numbers/names, non-English
# samples, and near-duplicate same-topic clusters. Each turn lists the
# distinctive content terms that should retrieve it.

_CORPUS: tuple[CorpusTurn, ...] = (
    CorpusTurn(
        "turn-job-marketing",
        "Career background",
        "Before this job I worked as a marketing specialist at a small startup downtown.",
        ("marketing", "specialist", "startup"),
    ),
    CorpusTurn(
        "turn-sister-dog",
        "Family pet note",
        "My sister's dog is a golden retriever named Biscuit who hates thunderstorms.",
        ("retriever", "Biscuit", "thunderstorms"),
    ),
    CorpusTurn(
        "turn-coffee-creamer",
        "Grocery coupon",
        "I redeemed a coupon on hazelnut coffee creamer at the corner store yesterday.",
        ("hazelnut", "creamer", "coupon"),
    ),
    CorpusTurn(
        "turn-paris-trip",
        "Travel plan",
        "We are flying to Paris in October and staying near the Montmartre district.",
        ("Paris", "Montmartre", "October"),
    ),
    CorpusTurn(
        "turn-guitar-lessons",
        "Hobby update",
        "I finally started weekly guitar lessons and I am learning fingerpicking now.",
        ("guitar", "fingerpicking", "lessons"),
    ),
    CorpusTurn(
        "turn-docs-link",
        "Shared reference",
        "Here is the onboarding guide at https://docs.example.com/onboarding for new hires.",
        ("onboarding", "hires"),
    ),
    CorpusTurn(
        "turn-budget-spreadsheet",
        "Finance task",
        "I updated the quarterly budget spreadsheet and the travel line is over by 1200 dollars.",
        ("budget", "spreadsheet", "quarterly"),
    ),
    CorpusTurn(
        "turn-marathon-training",
        "Running goal",
        "My marathon training peaks next month with a brutal twenty-two mile long run.",
        ("marathon", "training"),
    ),
    CorpusTurn(
        "turn-dentist-appointment",
        "Health reminder",
        "The dentist appointment got moved to Thursday because the hygienist was out sick.",
        ("dentist", "hygienist"),
    ),
    CorpusTurn(
        "turn-recipe-lasagna",
        "Cooking note",
        "Grandma's lasagna recipe uses three cheeses and a slow simmered tomato ragu.",
        ("lasagna", "ragu", "cheeses"),
    ),
    CorpusTurn(
        "turn-car-repair",
        "Vehicle issue",
        "The mechanic said the alternator is failing and the timing belt is due soon.",
        ("alternator", "mechanic"),
    ),
    CorpusTurn(
        "turn-book-club",
        "Reading group",
        "Our book club picked a sprawling science fiction novel about generation ships.",
        ("generation", "ships", "novel"),
    ),
    CorpusTurn(
        "turn-garden-tomatoes",
        "Gardening",
        "The heirloom tomatoes in the raised beds finally ripened after the heat wave.",
        ("heirloom", "tomatoes"),
    ),
    CorpusTurn(
        "turn-flight-delay",
        "Travel mishap",
        "My connecting flight was delayed three hours so I missed the riverside dinner booking.",
        ("delayed", "riverside", "booking"),
    ),
    CorpusTurn(
        "turn-new-laptop",
        "Purchase",
        "I bought a refurbished laptop with a mechanical keyboard and a matte display.",
        ("refurbished", "mechanical", "keyboard"),
    ),
    CorpusTurn(
        "turn-yoga-class",
        "Wellness",
        "The new vinyasa yoga instructor pushes a punishing pace on Tuesday evenings.",
        ("vinyasa", "instructor"),
    ),
    CorpusTurn(
        "turn-spanish-greeting",
        "Language practice",
        "Mi hermano vive en Sevilla y trabaja como arquitecto cerca del río.",
        ("hermano", "Sevilla", "arquitecto"),
    ),
    CorpusTurn(
        "turn-french-school",
        "Language practice",
        "L'école française du quartier ferme ses portes pendant les vacances d'été.",
        ("française", "quartier"),
    ),
    CorpusTurn(
        "turn-german-train",
        "Language practice",
        "Der Zug nach München war pünktlich und überraschend leer am Sonntagmorgen.",
        ("München", "Sonntagmorgen"),
    ),
    CorpusTurn(
        "turn-promotion",
        "Work milestone",
        "I got promoted to staff engineer and now I lead the payments reliability squad.",
        ("promoted", "payments", "reliability"),
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
    ),
    # Near-duplicate cluster: the office plant, three slightly different tellings.
    CorpusTurn(
        "turn-plant-a",
        "Office plant",
        "The office fiddle leaf fig is dropping leaves again near the drafty window.",
        ("fiddle", "fig"),
    ),
    CorpusTurn(
        "turn-plant-b",
        "Office plant note",
        "Someone overwatered the office fiddle leaf fig and now its leaves are yellowing.",
        ("fiddle", "overwatered"),
    ),
    CorpusTurn(
        "turn-plant-c",
        "Office plant update",
        "We moved the office fiddle leaf fig away from the window and it perked up.",
        ("fiddle", "perked"),
    ),
    # Near-duplicate cluster: the standing desk, two tellings.
    CorpusTurn(
        "turn-desk-a",
        "Ergonomics",
        "My new standing desk wobbles slightly when it is raised to the tallest setting.",
        ("standing", "wobbles"),
    ),
    CorpusTurn(
        "turn-desk-b",
        "Ergonomics follow-up",
        "I added felt pads under the standing desk feet and the wobble is mostly gone.",
        ("felt", "pads"),
    ),
    CorpusTurn(
        "turn-podcast",
        "Media recommendation",
        "A friend recommended a history podcast about the cartography of medieval trade routes.",
        ("cartography", "medieval"),
    ),
    CorpusTurn(
        "turn-allergy",
        "Health note",
        "My seasonal ragweed allergy flared up so I switched to a non drowsy antihistamine.",
        ("ragweed", "antihistamine"),
    ),
    CorpusTurn(
        "turn-camera",
        "Photography",
        "I rented a wide angle lens for the canyon shoot and the dynamic range was stunning.",
        ("canyon", "lens"),
    ),
    CorpusTurn(
        "turn-volunteer",
        "Community",
        "On Saturdays I volunteer at the riverbank cleanup and we filled forty trash bags.",
        ("riverbank", "cleanup"),
    ),
    CorpusTurn(
        "turn-keyboard-don't",
        "Typing habit",
        "I don't use the number pad much so I switched to a compact tenkeyless keyboard.",
        ("tenkeyless",),
    ),
    CorpusTurn(
        "turn-numbers-invoice",
        "Billing",
        "Invoice 4471 is still unpaid and the late fee kicks in after thirty days.",
        ("4471", "invoice"),
    ),
    CorpusTurn(
        "turn-timestamp-meeting",
        "Calendar",
        "The retro is locked in for half past noon so block out that slot on the calendar.",
        ("retro",),
    ),
    CorpusTurn(
        "turn-names-people",
        "Introductions",
        "At the offsite I finally met Nakamura from design and Olafsson from infrastructure.",
        ("Nakamura", "Olafsson"),
    ),
    CorpusTurn(
        "turn-coffee-shop",
        "Routine",
        "The barista at the Wexford cafe remembers my oat milk cortado without me asking.",
        ("Wexford", "cortado"),
    ),
    CorpusTurn(
        "turn-puzzle",
        "Leisure",
        "I am stuck on a thousand piece jigsaw of a lighthouse swallowed by fog.",
        ("jigsaw", "lighthouse"),
    ),
    CorpusTurn(
        "turn-bike-commute",
        "Commute",
        "My bike commute got faster after they finally painted the protected lane on Birch Street.",
        ("Birch", "lane"),
    ),
    CorpusTurn(
        "turn-houseplant-tip",
        "Advice received",
        "A neighbor told me bottom watering keeps the succulents from rotting at the crown.",
        ("succulents", "crown"),
    ),
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


@pytest.fixture
def embedding_provider() -> CallbackProvider:
    """Return a deterministic bag-of-words embedding provider.

    Returns:
        A :class:`CallbackProvider` wrapping the network-free hashing embedder.
    """
    return CallbackProvider(
        callback=_bag_of_words_embed,
        dimension=_EMBED_DIM,
        model_name="bag-of-words-contract",
    )


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
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
    )


@pytest.fixture
async def fts_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return a store populated with the corpus, FTS-only (no embeddings).

    Yields:
        A :class:`SqliteEngravaCore` whose FTS5 index holds every corpus turn.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    for turn in _CORPUS:
        await store.create_thought(_to_thought(turn))
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
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(
        conn,
        embedding_provider=embedding_provider,
        auto_embed=True,
    )
    await store.ensure_schema()
    for turn in _CORPUS:
        await store.create_thought(_to_thought(turn))
    yield store
    await conn.close()

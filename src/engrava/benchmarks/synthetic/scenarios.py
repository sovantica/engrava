"""Scenario templates for the synthetic benchmark.

Each :class:`Scenario` describes a conversational pattern: how
memorable facts are introduced, what distraction-line templates fill
the gaps, and how the recall question is phrased.  The generator in
:mod:`engrava.benchmarks.synthetic.generate` instantiates one
conversation per scenario by drawing concrete slot values from the
seeded :class:`random.Random`.

Pre-registration discipline
---------------------------

The scenario library MUST contain at least two scenarios
where dreaming is *not* expected to help.  Those scenarios are
marked ``expected_dreaming_effect="neutral"`` in this file and that
flag is committed to git history **before** any calibration run.
The runtime evaluator then enforces a tight |ON - OFF| <= 0.02 band
on the sanity subset (see :mod:`engrava.benchmarks.synthetic.evaluate`
and ``tests/benchmarks/test_synthetic_e2e.py``); silently lowering
that band would invalidate the whole benchmark.

The non-neutral scenarios are intentionally biased toward the
recall pathologies dreaming targets — long-distance recall,
multi-fact composition, thematic clusters, contradiction resolution
and distraction-heavy retrieval.  This is fair because the benchmark
exists to surface whether those gains materialise on a dataset where
the structure makes them *possible*, not to prove that dreaming
helps everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "SCENARIO_LIBRARY",
    "Scenario",
    "ScenarioDifficulty",
    "ScenarioDreamingEffect",
    "ThemeBundle",
]


def _freeze_vocab(
    raw: dict[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    """Wrap a slot-vocabulary dict in a read-only ``MappingProxyType``.

    Pre-registered scenarios must be effectively immutable so a test
    or downstream caller cannot mutate ``SCENARIO_LIBRARY[i].
    slot_vocabulary`` and silently change the corpus on a future run.
    """
    return MappingProxyType(dict(raw))


ScenarioDifficulty = Literal["easy", "medium", "hard"]
# Three categories aligned with the AC-9a / AC-9b / AC-8 grouping:
#
# * ``"gain"`` — synthesis-requiring; AC-9a binding ≥5pp ON-vs-OFF gain.
#   The answer to the recall question exists in a REFLECTION cluster
#   summary (consolidated_from semantics), not in any single planted
#   OBSERVATION.  Dreaming is the mechanism under test.
# * ``"neutral_or_minor"`` — direct-retrieval scenarios; AC-9b binding
#   ±2pp neutrality.  FTS / vector retrieval finds the planted facts on
#   its own; dreaming must not degrade direct lookup.
# * ``"neutral"`` — anti-cherry-pick sanity subset; AC-8 binding ±2pp.
#   Pre-registered scenarios where dreaming should not help at all.
ScenarioDreamingEffect = Literal["gain", "neutral_or_minor", "neutral"]


@dataclass(frozen=True)
class ThemeBundle:
    """A pre-baked theme for the synthesis-requiring scenarios.

    Direct-retrieval scenarios pick slot values from a vocabulary and
    render parameterised templates.  Synthesis scenarios (where the
    theme name MUST NOT appear in any single planted turn) need a
    different shape: a fixed bundle of literal facets / paraphrases
    plus a fixed question that does not match any facet lexically.

    Attributes:
        theme_id: Stable identifier (e.g. ``"lisbon"``,
            ``"black-coffee"``).  Used in fact-id construction and
            substring-match scoring.
        facets: Literal memorable-turn texts.  Each facet is emitted
            once per conversation.  For ``abstract_theme_recall``
            facets are lexically diverse aspects of one theme and
            never mention the theme name itself.  For
            ``repeated_paraphrase_compression`` facets are lexically
            diverse paraphrases of one claim.
        question_text: Literal question phrasing.  Stays abstract
            for the theme-recall scenario so direct lexical match on
            any single facet is unlikely; for paraphrase-compression
            asks for the underlying claim without echoing any single
            paraphrase.
        expected_substrings: Substring(s) any retrieved record's
            content should contain for a positive substring-match.
            For ``abstract_theme_recall`` this is the theme name
            (which appears in REFLECTION ``top_keyphrases`` once
            dreaming has aggregated facets); for
            ``repeated_paraphrase_compression`` it is a stable
            phrase fragment shared across paraphrases.

    """

    theme_id: str
    facets: tuple[str, ...]
    question_text: str
    expected_substrings: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    """Template describing one synthetic conversational pattern.

    A scenario carries every parameter the generator needs to produce
    a single conversation: the memorable-fact templates (with named
    slots filled from the scenario's slot vocabulary), the distraction
    templates, the question phrasing, and the placement rules for the
    memorable turn(s) and the question turn.

    Two generation paths share the dataclass:

    * The **direct-retrieval** path uses ``memorable_templates`` /
      ``question_templates`` / ``slot_vocabulary`` — each conversation
      picks one concrete value per slot and renders a small number of
      planted turns whose answer matches the question lexically.
    * The **synthesis** path uses ``theme_bundles`` instead — each
      conversation picks one ``ThemeBundle`` and emits every facet as
      a planted turn.  The question is the bundle's ``question_text``
      verbatim and never matches any facet lexically.  This is the
      path that exercises dreaming as the answer carrier (the
      ``consolidated_from`` REFLECTION branch in
      :func:`engrava.benchmarks.synthetic.evaluate._score_retrieval`).

    Exactly one path is active per scenario.  ``theme_bundles`` is
    ``None`` for direct-retrieval scenarios and non-empty for synthesis
    scenarios; ``memorable_templates`` / ``slot_vocabulary`` are unused
    when ``theme_bundles`` is set.

    Attributes:
        name: Short stable identifier, used as a dict key and as the
            ``scenario`` tag on every generated question.
        description: One-sentence summary of the recall pathology
            the scenario exercises.
        difficulty: Coarse difficulty tag for the summary table.
        expected_dreaming_effect: Pre-registered expectation:
            ``"gain"`` for synthesis-requiring scenarios (AC-9a
            ≥5pp ON-vs-OFF gain MUST materialise),
            ``"neutral_or_minor"`` for direct-retrieval scenarios
            (AC-9b ±2pp neutrality MUST hold), ``"neutral"`` for
            anti-cherry-pick sanity scenarios (AC-8 ±2pp).  Never
            flip a flag to chase a target metric.
        memorable_templates: Sentence templates for the memorable
            fact(s) on the direct path.  Slots are simple
            ``{name}``-style placeholders that the generator fills
            from ``slot_vocabulary``.  Ignored when ``theme_bundles``
            is set.
        question_templates: Question templates referring back to the
            memorable fact(s).  Slots match ``memorable_templates``.
            Ignored when ``theme_bundles`` is set.
        answer_substring_templates: Per-question substrings that any
            retrieved thought content should contain for a positive
            substring-match score (orthogonal to recall@K).  Ignored
            when ``theme_bundles`` is set — bundles carry their own
            ``expected_substrings`` per theme.
        distraction_templates: Templates for filler / off-topic
            turns.  These never carry a memorable fact.
        slot_vocabulary: Mapping of slot name to candidate values
            (direct path only).  The generator picks one value per
            slot per conversation.
        n_memorable_facts: Number of memorable fact turns the
            scenario plants per conversation on the direct path.
            For synthesis scenarios the count is derived from the
            chosen bundle's ``facets`` length and this field is
            ignored.
        question_offset_min_turns: Lower bound on the gap (in turns)
            between the last memorable turn and the recall question.
        question_offset_max_turns: Upper bound on that gap.  Together
            with the corpus-level conversation length these two
            numbers control "how long ago was the fact mentioned".
        theme_bundles: When non-None, switches the scenario to the
            synthesis generation path.  Each conversation picks one
            bundle and emits every facet as a planted turn.

    """

    name: str
    description: str
    difficulty: ScenarioDifficulty
    expected_dreaming_effect: ScenarioDreamingEffect
    memorable_templates: tuple[str, ...]
    question_templates: tuple[str, ...]
    answer_substring_templates: tuple[tuple[str, ...], ...]
    distraction_templates: tuple[str, ...]
    slot_vocabulary: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    n_memorable_facts: int = 1
    question_offset_min_turns: int = 30
    question_offset_max_turns: int = 50
    theme_bundles: tuple[ThemeBundle, ...] | None = None


# ---------------------------------------------------------------------------
# Shared distraction pool — re-used across most scenarios so the dataset
# does not telegraph which conversation belongs to which scenario via
# distraction wording alone.
# ---------------------------------------------------------------------------

_SHARED_DISTRACTIONS: tuple[str, ...] = (
    "I went to {place} earlier today.",
    "The weather was {weather} this morning.",
    "I read a chapter from {book}.",
    "Someone mentioned {topic} on the radio.",
    "I cooked {meal} for dinner.",
    "There was a small queue at the {shop}.",
    "I listened to a podcast about {topic}.",
    "The neighbours were playing {music_genre} again.",
    "I had a short call with a colleague about {topic}.",
    "I took a walk through {place}.",
)

_SHARED_SLOT_VOCAB: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "place": ("the park", "downtown", "the harbour", "the library", "the riverside"),
        "weather": ("rainy", "sunny", "windy", "overcast", "foggy"),
        "book": (
            "a travel memoir",
            "a programming guide",
            "a short-story collection",
            "a history of cartography",
            "a science magazine",
        ),
        "topic": (
            "renewable energy",
            "city planning",
            "ocean currents",
            "language learning",
            "ancient pottery",
        ),
        "meal": ("pasta", "lentil soup", "a stir-fry", "fish tacos", "vegetable curry"),
        "shop": ("bakery", "post office", "hardware store", "florist", "grocer"),
        "music_genre": ("jazz", "indie folk", "classical", "ambient", "synthwave"),
    },
)


# ---------------------------------------------------------------------------
# Pre-registered scenario library — order is stable; tests pin the count.
# ---------------------------------------------------------------------------

SCENARIO_LIBRARY: tuple[Scenario, ...] = (
    Scenario(
        name="long_recall_simple",
        description=(
            "One memorable fact early in the conversation, recall "
            "question after ~40 turns of unrelated distractions."
        ),
        difficulty="easy",
        expected_dreaming_effect="neutral_or_minor",
        memorable_templates=("My favourite colour is {colour}.",),
        question_templates=("What is my favourite colour?",),
        answer_substring_templates=(("{colour}",),),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(
            {
                "colour": (
                    "deep emerald green",
                    "burnt orange",
                    "midnight blue",
                    "soft lavender",
                    "warm terracotta",
                ),
                **_SHARED_SLOT_VOCAB,
            },
        ),
        n_memorable_facts=1,
        question_offset_min_turns=35,
        question_offset_max_turns=55,
    ),
    Scenario(
        name="multi_fact_recall",
        description=(
            "Two related memorable facts planted in the first third; "
            "the question requires composing both."
        ),
        difficulty="medium",
        expected_dreaming_effect="neutral_or_minor",
        memorable_templates=(
            "I started learning {language_a} last month.",
            "I also picked up {language_b} as a second language to practise.",
        ),
        question_templates=("Which two languages did I say I was learning?",),
        answer_substring_templates=(("{language_a}", "{language_b}"),),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(
            {
                "language_a": ("Portuguese", "Mandarin", "Finnish", "Greek", "Korean"),
                "language_b": ("Japanese", "Swahili", "Hungarian", "Turkish", "Welsh"),
                **_SHARED_SLOT_VOCAB,
            },
        ),
        n_memorable_facts=2,
        question_offset_min_turns=30,
        question_offset_max_turns=50,
    ),
    Scenario(
        name="thematic_cluster",
        description=(
            "Four related memorable facts spread across the early "
            "and middle turns; the question asks about the theme."
        ),
        difficulty="medium",
        # Synthesis subset (AC-9a) — the destination name is mentioned
        # in every facet so the cluster centroid is dominated by it;
        # the question is abstract enough that REFLECTION ranking helps.
        expected_dreaming_effect="gain",
        memorable_templates=(
            "I'm planning a trip to {destination} next month.",
            "I've been reading about the {destination} food scene.",
            "I booked a small guesthouse in {destination}.",
            "A friend recommended a walking route around {destination}.",
        ),
        question_templates=("Where am I planning to travel and why?",),
        answer_substring_templates=(("{destination}",),),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(
            {
                "destination": ("Lisbon", "Kyoto", "Reykjavik", "Marrakech", "Tallinn"),
                **_SHARED_SLOT_VOCAB,
            },
        ),
        n_memorable_facts=4,
        question_offset_min_turns=20,
        question_offset_max_turns=40,
    ),
    Scenario(
        name="contradiction_resolution",
        description=(
            "An initial fact is later contradicted by a corrected "
            "version; the question asks for the current state."
        ),
        difficulty="hard",
        expected_dreaming_effect="neutral_or_minor",
        memorable_templates=(
            "I've decided to adopt a {pet_a}.",
            "Actually I changed my mind and adopted a {pet_b} instead.",
        ),
        question_templates=("Which pet did I end up adopting?",),
        answer_substring_templates=(("{pet_b}",),),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(
            {
                "pet_a": (
                    "tabby kitten",
                    "labrador puppy",
                    "rescue greyhound",
                    "rabbit",
                    "cockatiel",
                ),
                "pet_b": (
                    "border collie",
                    "siamese cat",
                    "ferret",
                    "lovebird",
                    "shiba inu",
                ),
                **_SHARED_SLOT_VOCAB,
            },
        ),
        n_memorable_facts=2,
        question_offset_min_turns=25,
        question_offset_max_turns=45,
    ),
    Scenario(
        name="distraction_heavy",
        description=(
            "One memorable fact buried under a very long run of "
            "distractions; tests recall under noise."
        ),
        difficulty="medium",
        expected_dreaming_effect="neutral_or_minor",
        memorable_templates=("My passport number ends in {digits}.",),
        question_templates=("What were the last four digits of my passport?",),
        answer_substring_templates=(("{digits}",),),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(
            {
                "digits": ("4729", "1083", "5614", "9302", "7148"),
                **_SHARED_SLOT_VOCAB,
            },
        ),
        n_memorable_facts=1,
        question_offset_min_turns=50,
        question_offset_max_turns=65,
    ),
    # -----------------------------------------------------------------
    # Anti-cherry-pick scenarios — pre-registered as DREAMING-NEUTRAL.
    # The evaluator enforces |ON - OFF| <= 0.02 on the subset built
    # from these two scenarios; the assertion lives in
    # ``tests/benchmarks/test_synthetic_e2e.py``.
    # -----------------------------------------------------------------
    Scenario(
        name="single_unique_fact",
        description=(
            "One isolated unique fact with no related context.  "
            "Dreaming has nothing to cluster against, so OFF and ON "
            "should perform within a couple of percentage points."
        ),
        difficulty="easy",
        expected_dreaming_effect="neutral",
        memorable_templates=("My favourite obscure word is {rare_word}.",),
        question_templates=("What unusual word did I say was my favourite?",),
        answer_substring_templates=(("{rare_word}",),),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(
            {
                "rare_word": (
                    "petrichor",
                    "sonder",
                    "limerence",
                    "apricity",
                    "susurrus",
                ),
                **_SHARED_SLOT_VOCAB,
            },
        ),
        n_memorable_facts=1,
        question_offset_min_turns=25,
        question_offset_max_turns=45,
    ),
    Scenario(
        name="recent_fact_recall",
        description=(
            "Memorable fact mentioned only a handful of turns before "
            "the question; recency dominates retrieval, dreaming is "
            "irrelevant.  Sanity check against ``always wins`` artifacts."
        ),
        difficulty="easy",
        expected_dreaming_effect="neutral",
        memorable_templates=("My new phone case is {colour}.",),
        question_templates=("What colour is my new phone case?",),
        answer_substring_templates=(("{colour}",),),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(
            {
                "colour": (
                    "deep emerald green",
                    "burnt orange",
                    "midnight blue",
                    "soft lavender",
                    "warm terracotta",
                ),
                **_SHARED_SLOT_VOCAB,
            },
        ),
        n_memorable_facts=1,
        question_offset_min_turns=3,
        question_offset_max_turns=6,
    ),
    # -----------------------------------------------------------------
    # Synthesis-requiring scenarios (AC-9a binding ≥5pp gain).  The
    # answer to the recall question exists in the REFLECTION cluster
    # summary, not in any single planted observation — so dreaming is
    # mechanically required to score.
    # -----------------------------------------------------------------
    Scenario(
        name="abstract_theme_recall",
        description=(
            "Eight lexically-diverse facets of a single theme are planted "
            "across the conversation.  The theme name itself never appears "
            "in any single turn.  The recall question asks for the theme "
            "abstractly (``what city has been recurring in our recent "
            "chats?``) — direct OBSERVATION retrieval cannot answer it; "
            "only a REFLECTION clustering the facets surfaces the theme."
        ),
        difficulty="hard",
        expected_dreaming_effect="gain",
        memorable_templates=(),
        question_templates=(),
        answer_substring_templates=(),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(dict(_SHARED_SLOT_VOCAB)),
        n_memorable_facts=8,
        question_offset_min_turns=10,
        question_offset_max_turns=20,
        theme_bundles=(
            ThemeBundle(
                theme_id="lisbon",
                facets=(
                    "I tried pastel de nata at a small bakery yesterday afternoon.",
                    "The number 28 tram squeals as it climbs through Alfama.",
                    "Fado music spilled out of a tavern near Bairro Alto.",
                    "I climbed the Belém Tower stairs and the view was breathtaking.",
                    "The azulejo tile work at the National Museum is mesmerising.",
                    "I queued for grilled sardines at a tiny tasca near the harbour.",
                    "The yellow funicular up to Lavra creaks louder than I expected.",
                    "I read about Saramago in a bookshop with creaky wooden floors.",
                ),
                question_text=(
                    "What city has been the recurring backdrop of my recent "
                    "stories — the one I keep dropping fragments about?"
                ),
                expected_substrings=("Lisbon",),
            ),
            ThemeBundle(
                theme_id="kyoto",
                facets=(
                    "I walked through a bamboo forest in the early morning mist.",
                    "The torii gates of Fushimi Inari climb up the mountain endlessly.",
                    "I watched maiko cross the lantern-lit streets of Gion at dusk.",
                    "Matcha so vivid it almost glows arrived in a chawan.",
                    "Cherry blossoms drifted down on the Philosopher's Path canal.",
                    "I sat in seiza posture during a tea ceremony for an hour.",
                    "The dry-stone garden at Ryoan-ji has fifteen rocks you can never see at once.",
                    "Geta clattered against stone steps near a small Shinto shrine.",
                ),
                question_text=(
                    "What city has been the recurring backdrop of my recent "
                    "stories — the one I keep dropping fragments about?"
                ),
                expected_substrings=("Kyoto",),
            ),
            ThemeBundle(
                theme_id="reykjavik",
                facets=(
                    "I soaked in a geothermal pool while sleet drifted overhead.",
                    "The northern lights flickered green above the harbour at midnight.",
                    "Skyr with rye crumbs is denser than any yoghurt back home.",
                    "I drove past a black-sand beach with basalt columns at sunset.",
                    "A fishing trawler unloaded cod at the rainbow-coloured pier.",
                    "Volcanic steam rose from cracks in the moss-covered lava field.",
                    "I tasted hákarl on a dare and immediately regretted it.",
                    "Hallgrímskirkja's organ rumbled through the whole nave.",
                ),
                question_text=(
                    "What city has been the recurring backdrop of my recent "
                    "stories — the one I keep dropping fragments about?"
                ),
                expected_substrings=("Reykjavik",),
            ),
            ThemeBundle(
                theme_id="marrakech",
                facets=(
                    "The medina's spice stalls hit me with cumin and dried rose.",
                    "Snake charmers and storytellers shared Jemaa el-Fnaa at dusk.",
                    "I haggled over a hand-knotted rug for nearly an hour.",
                    "Mint tea poured from a great height foams perfectly.",
                    "The Koutoubia minaret towers over the date-palm-lined avenues.",
                    "I lost my bearings three times in the souk's twisting alleys.",
                    "A donkey-cart squeezed past me carrying mountains of mint.",
                    "Tagine simmered slowly over coals at a riad rooftop.",
                ),
                question_text=(
                    "What city has been the recurring backdrop of my recent "
                    "stories — the one I keep dropping fragments about?"
                ),
                expected_substrings=("Marrakech",),
            ),
            ThemeBundle(
                theme_id="tallinn",
                facets=(
                    "I walked the medieval ramparts under a low grey sky.",
                    "Marzipan figurines fill the windows of a Raekoja plats shop.",
                    "The cobbled streets of Vanalinn glistened after the morning drizzle.",
                    "A brass band played in the square below St Olaf's church.",
                    "I had black bread soup at a cellar restaurant inside the old wall.",
                    "Wooden Kalamaja houses lean against each other along narrow lanes.",
                    "The Toompea Castle tower flies a flag taller than the keep.",
                    "Glühwein steamed in clay mugs at the winter market booths.",
                ),
                question_text=(
                    "What city has been the recurring backdrop of my recent "
                    "stories — the one I keep dropping fragments about?"
                ),
                expected_substrings=("Tallinn",),
            ),
        ),
    ),
    Scenario(
        name="repeated_paraphrase_compression",
        description=(
            "Twelve lexically-distinct paraphrases of one underlying claim "
            "are planted across the conversation.  Each paraphrase ranks "
            "moderately against the recall question alone; a REFLECTION "
            "centroid compressing all twelve into one high-confidence "
            "summary should rank above any single paraphrase."
        ),
        difficulty="medium",
        expected_dreaming_effect="gain",
        memorable_templates=(),
        question_templates=(),
        answer_substring_templates=(),
        distraction_templates=_SHARED_DISTRACTIONS,
        slot_vocabulary=_freeze_vocab(dict(_SHARED_SLOT_VOCAB)),
        n_memorable_facts=12,
        question_offset_min_turns=8,
        question_offset_max_turns=18,
        theme_bundles=(
            ThemeBundle(
                theme_id="black-coffee",
                facets=(
                    "I take my coffee black, no sugar, no milk.",
                    "Just a plain americano for me, nothing added.",
                    "I drink my espresso straight, the way it leaves the machine.",
                    "No cream and definitely no syrup in my cup, thanks.",
                    "Black coffee is the only way I order it.",
                    "Skip the milk — straight coffee is what I want.",
                    "I never add sugar to my morning brew.",
                    "Plain, unsweetened, no dairy — that's how I drink it.",
                    "I'd rather have it bitter than ruin it with cream.",
                    "Hot, black, nothing else — same as always.",
                    "I order it without any of the usual additions.",
                    "Pour-over, black, no sweetener — my standing order.",
                ),
                question_text=(
                    "Looking across what I've mentioned about my coffee, "
                    "how do I prefer it prepared?"
                ),
                expected_substrings=("black",),
            ),
            ThemeBundle(
                theme_id="early-riser",
                facets=(
                    "I'm usually up before five in the morning.",
                    "My alarm goes off at half past four most days.",
                    "I tend to start my day while it's still dark outside.",
                    "Mornings before sunrise are when I get my best work done.",
                    "I wake up well before anyone else in the household.",
                    "Pre-dawn hours suit me — I'm at my desk early.",
                    "Most days I've already finished breakfast by six.",
                    "I rarely sleep past five, even on weekends.",
                    "The world is quietest at four in the morning, and I love it.",
                    "I'm an early riser through and through.",
                    "My productive window opens before the sun comes up.",
                    "I head to bed around nine because I'm up so early.",
                ),
                question_text=(
                    "Across everything I've said about my mornings, what is "
                    "my typical waking pattern?"
                ),
                expected_substrings=("early",),
            ),
            ThemeBundle(
                theme_id="vegetarian",
                facets=(
                    "I haven't eaten meat for several years now.",
                    "My diet is plant-based and has been for a while.",
                    "I order the vegetarian option whenever it's available.",
                    "Beans, lentils and tofu make up most of my protein.",
                    "I gave up chicken and beef back in university.",
                    "Restaurants without a meat-free dish are tough for me.",
                    "I cook entirely with vegetables, grains and legumes at home.",
                    "Meat hasn't been on my plate in years.",
                    "When I travel I always check for vegetarian menus first.",
                    "I'd describe myself as fully plant-based.",
                    "Eggs and dairy are fine but I never order meat.",
                    "My pantry has no animal protein in it at all.",
                ),
                question_text=(
                    "Across the things I've mentioned about food and meals, "
                    "what would you say my dietary pattern is?"
                ),
                expected_substrings=("vegetarian",),
            ),
            ThemeBundle(
                theme_id="cyclist",
                facets=(
                    "I commute to work on two wheels every weekday.",
                    "My bike is how I get around the city, rain or shine.",
                    "I haven't owned a car in over six years.",
                    "Weekend rides through the countryside are my main hobby.",
                    "I rode 40 kilometres yesterday before lunch.",
                    "My pannier bags carry everything I need for the day.",
                    "I service my own chain and brakes in the garage.",
                    "Cycle lanes determine which routes I take to meetings.",
                    "I belong to a small Sunday peloton that meets at dawn.",
                    "Drop-bars and clipless pedals have been my setup for years.",
                    "I've cycled across three countries on tour.",
                    "Everywhere I go in town, I go on a bicycle.",
                ),
                question_text=(
                    "Looking at everything I've said about getting around, "
                    "what is my primary mode of transport?"
                ),
                expected_substrings=("cycl",),
            ),
            ThemeBundle(
                theme_id="night-owl",
                facets=(
                    "I rarely fall asleep before two in the morning.",
                    "My best thinking happens after midnight, always has.",
                    "The house is quietest at one and I love working then.",
                    "I'm at my sharpest in the small hours, not at sunrise.",
                    "Sleep before midnight feels alien to me.",
                    "I keep my desk lamp on long after everyone else turns in.",
                    "Late-night writing sessions are where most of my work gets done.",
                    "I read until two or three most nights without realising.",
                    "Going to bed at eleven would feel impossibly early for me.",
                    "Midnight is when my second wind kicks in.",
                    "I describe myself as nocturnal more than anything.",
                    "My energy peaks at exactly the wrong time of day for most people.",
                ),
                question_text=(
                    "Considering what I've said about my hours and energy, "
                    "what is my sleep pattern?"
                ),
                expected_substrings=("night",),
            ),
        ),
    ),
)

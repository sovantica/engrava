# The Bi-temporal Model

Engrava can track **two independent time axes** for a fact: *when you stored it*
and *when it is true in the world*. This page explains the second axis —
**valid time** — what it is, how it differs from the clocks Engrava already has,
and the small set of opt-in tools that work with it.

> Valid time is **entirely optional**. If you only ever care about "what is true
> now", you can ignore everything on this page — the defaults already behave the
> way you want (see [When you don't need valid time](#when-you-dont-need-valid-time)).

## Two clocks (three, actually)

A thought carries more than one notion of "time", and they answer different
questions. Keep them apart:

| Field | Axis | Answers | Who sets it |
|---|---|---|---|
| `created_at` / `updated_at` | **transaction time** | "When did we *record* (or last change) this?" | Engrava, automatically |
| `valid_from` / `valid_until` | **valid time** | "During what real-world period is this fact *true*?" | **you** (optional) |
| `created_cycle` / `updated_cycle` | **logical clock** | "At which agent *tick* did this appear?" | you (your cycle counter) |

- **Transaction time** is bookkeeping: it never moves backwards and you don't
  manage it. It tells you the order in which your system learned things.
- **Valid time** is about the world, not your database. "The user lived in
  Berlin from January to June" is a statement about reality — it is true for a
  window that has nothing to do with when you happened to write it down. You set
  it; Engrava only stores and queries it.

### Don't confuse the cycle with valid time

The **cycle** (`created_cycle`) is *not* a calendar. It is a monotonically
increasing integer **you** advance once per agent turn (see
[Core Concepts → Cycle](concepts.md#cycle-the-agent-clock)). It drives recency
and dreaming math; it is deliberately wall-clock-independent.

Valid time, by contrast, is an **ISO-8601 calendar timestamp** describing the
real world. A fact can have a low `created_cycle` (you learned it early in the
agent's life) yet a `valid_from` far in the future, or vice versa. Never reach
for the cycle when you mean a date — they are different tools for different jobs.

## `valid_from` / `valid_until`: an interval, with open ends

Both `valid_from` and `valid_until` are **nullable ISO-8601 strings** on
`ThoughtRecord` and `EdgeRecord`. Together they describe the half-open interval
during which the fact is considered true:

- **`valid_from = None`** — *open lower bound*. The fact is treated as valid
  **from the beginning of time** (−∞). "We don't know (or don't care) when this
  started; it has always been true as far as we're concerned."
- **`valid_until = None`** — *open upper bound*. The fact is treated as
  **still valid, with no known end** (+∞). This is the common case for a fact
  that is currently true.
- Both `None` (the default for every newly created record) — the fact is valid
  **for all time** and matches every "is it valid?" query.

> **NULL = open, not unknown-and-excluded.** This is the single most important
> rule on this page. A NULL bound means the interval extends to infinity on that
> side, so a row with open bounds *stays visible* to point-in-time queries — it
> is not filtered out. That is what makes adopting valid time incremental: facts
> you never annotate keep showing up exactly as before.

## The four query predicates

Valid time is queried through four **opt-in** `WHERE` predicates in
[MindQL](mindql.md). They are only valid against the `thoughts` and `edges`
tables (the two record types that carry valid-time columns). A query that uses
**no** temporal predicate behaves exactly as it did before this feature existed.

| Predicate | Arguments | Matches a row when… | NULL bounds |
|---|---|---|---|
| `valid_now` | none | the interval contains the current instant | tolerant (open bound = always in range) |
| `valid_at <ts>` | one timestamp | the interval contains `<ts>` | tolerant |
| `valid_within <start> <end>` | two timestamps | the interval **overlaps** `[<start>, <end>]` | tolerant |
| `valid_between <start> <end>` | two timestamps | the interval is **fully contained** in `[<start>, <end>]` | **strict** — open-bound rows are excluded |

### Worked semantics

The first three predicates treat a NULL bound as ±∞, so open-ended facts stay in
the result. `valid_between` is the deliberate exception: "fully contained"
cannot be true of an interval that runs to infinity, so it requires **real
bounds on both ends** and drops any row with a NULL `valid_from` or
`valid_until`.

The upper bound is **exclusive** (`valid_until` is the first instant the fact is
*no longer* true). Concretely, for a fact valid `[2026-01-01, 2026-07-01)`:

| Query | Result | Why |
|---|---|---|
| `valid_at '2026-03-15...'` | match | `2026-03-15` is inside `[Jan 1, Jul 1)` |
| `valid_at '2026-07-01...'` | no match | upper bound is exclusive — `Jul 1` is already out |
| `valid_at '2025-12-01...'` | no match | before `valid_from` |
| `valid_within '2026-06-01...' '2026-12-01...'` | match | the intervals overlap (Jun–Jul) |
| `valid_between '2025-01-01...' '2026-12-31...'` | match | `[Jan, Jul)` is fully inside the range |
| `valid_between '2026-02-01...' '2026-12-31...'` | no match | starts before the range's lower bound |

And for a fact with an **open** upper bound — valid `[2026-01-01, ∞)`:

| Query | Result | Why |
|---|---|---|
| `valid_now` | match (if now ≥ Jan 2026) | open upper bound = still valid |
| `valid_at '2030-01-01...'` | match | open upper bound reaches any future instant |
| `valid_between '2026-01-01...' '2026-12-31...'` | **no match** | `valid_between` rejects the open `valid_until` |

## `invalidate` vs `delete`

Engrava gives you two very different ways to retire a fact, and choosing the
right one is a modelling decision, not a performance one.

| | `invalidate_thought` / `invalidate_edge` | `delete_thought` |
|---|---|---|
| Meaning | "This **was** true, and is now superseded." | "This should never have existed." |
| Effect | Sets `valid_until`; the row stays on file | Removes the row entirely |
| History | **Preserved** — fully auditable, still retrievable | Gone |
| Past queries | A `valid_at` in the still-valid window **still finds it** | Finds nothing |
| LLM / search | **None** — deterministic, valid-time only | n/a |

`invalidate_thought(id, valid_until)` simply closes the valid-time interval at
the instant you pass. It is:

- **deterministic** — it performs no similarity search, no model inference, no
  automatic discovery of related facts;
- **idempotent** — calling it twice with the same `valid_until` converges to the
  same stored value;
- **non-cascading** — invalidating a thought leaves every connected edge
  untouched (invalidate the edges separately with `invalidate_edge` if you need
  to);
- **not a delete** — the row and its history remain. A point-in-time query for
  an instant *before* the new `valid_until` still returns it.

Reach for `invalidate` when reality changed (the user moved cities, a price was
updated, a status closed). Reach for `delete` only when a fact was an outright
mistake and you want it gone, history and all.

## Reflections inherit their members' extent

A `REFLECTION` is created by [dreaming](dreaming.md) from a cluster of member
thoughts. When dreaming builds one, it derives the reflection's valid-time
extent from its members rather than leaving it blank:

- `valid_from` becomes the **earliest** member `valid_from` — **unless any
  member has an open (NULL) lower bound**, in which case the reflection's
  `valid_from` is also open (NULL). An interval that summarises something
  open-ended is itself open-ended.
- `valid_until` becomes the **latest** member `valid_until` — but **only if
  every member has a closed upper bound**. If any member is still open, the
  reflection's `valid_until` is open (NULL) too.

In short: the reflection's interval is the union of its members' intervals, and
"open on either side" is contagious. A summary of facts that are still true is
itself still true.

## Worked examples

The three snippets below are complete, runnable scripts. Each opens an in-memory
database, so you can paste any one of them into a file and run it directly.

### (a) Set valid time on a new fact

Pass `valid_from` (and optionally `valid_until`) when you build the
`ThoughtRecord`. Here we record a fact known to be true from the start of 2026,
with no known end (`valid_until` left open):

```python
import asyncio
import uuid

import aiosqlite

from engrava import (
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)


async def main() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        fact = ThoughtRecord(
            thought_id=str(uuid.uuid4()),
            thought_type=ThoughtType.BELIEF,
            essence="The user lives in Berlin",
            content="Stated during the 2026 onboarding call.",
            priority=Priority.P2,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=10,
            updated_cycle=10,
            source="onboarding",
            valid_from="2026-01-01T00:00:00+00:00",  # true from the start of 2026
            # valid_until omitted -> open upper bound -> still valid
        )
        stored = await store.create_thought(fact)

        fetched = await store.get_thought(stored.thought_id)
        assert fetched is not None
        assert fetched.valid_from == "2026-01-01T00:00:00+00:00"
        assert fetched.valid_until is None  # open upper bound
        print("valid_from:", fetched.valid_from, "valid_until:", fetched.valid_until)


asyncio.run(main())
```

### (b) Time-travel: query a past instant

`valid_at <timestamp>` returns the facts that were true at that instant. Here a
belief is true only for the first half of 2026; a query inside that window finds
it, one after it does not:

```python
import asyncio
import uuid

import aiosqlite

from engrava import (
    LifecycleStatus,
    MindQLExecutor,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    parse,
)


async def main() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        await store.create_thought(
            ThoughtRecord(
                thought_id=str(uuid.uuid4()),
                thought_type=ThoughtType.BELIEF,
                essence="The user lives in Berlin",
                content="True for the first half of 2026.",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=10,
                updated_cycle=10,
                source="onboarding",
                valid_from="2026-01-01T00:00:00+00:00",
                valid_until="2026-07-01T00:00:00+00:00",  # closed in mid-2026
            ),
        )

        executor = MindQLExecutor(conn)

        march = await executor.execute(
            parse("FIND thoughts WHERE valid_at '2026-03-15T00:00:00+00:00'"),
        )
        september = await executor.execute(
            parse("FIND thoughts WHERE valid_at '2026-09-15T00:00:00+00:00'"),
        )

        assert len(march.rows) == 1  # inside the valid window
        assert len(september.rows) == 0  # after valid_until
        print("March match:", len(march.rows), "September match:", len(september.rows))


asyncio.run(main())
```

### (c) Invalidate a fact, then watch it drop out of `valid_now`

`invalidate_thought` closes the valid-time interval. After invalidation the fact
no longer matches `valid_now`, but it is **not deleted** — it remains on file and
a query for an instant before the cut-off still finds it:

```python
import asyncio
import uuid

import aiosqlite

from engrava import (
    LifecycleStatus,
    MindQLExecutor,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
    parse,
)


async def main() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        fact = await store.create_thought(
            ThoughtRecord(
                thought_id=str(uuid.uuid4()),
                thought_type=ThoughtType.BELIEF,
                essence="The user lives in Berlin",
                content="Open-ended until superseded.",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=10,
                updated_cycle=10,
                source="onboarding",
                valid_from="2026-01-01T00:00:00+00:00",
            ),
        )

        executor = MindQLExecutor(conn)

        before = await executor.execute(parse("FIND thoughts WHERE valid_now"))
        assert len(before.rows) == 1  # currently valid

        # Reality changed: the fact stopped being true on 2026-06-01.
        await store.invalidate_thought(
            fact.thought_id,
            valid_until="2026-06-01T00:00:00+00:00",
        )

        after = await executor.execute(parse("FIND thoughts WHERE valid_now"))
        assert len(after.rows) == 0  # no longer valid "now"

        # Not a delete: the row is still on file and auditable.
        still_there = await store.get_thought(fact.thought_id)
        assert still_there is not None
        assert still_there.valid_until == "2026-06-01T00:00:00+00:00"
        print("valid_now before:", len(before.rows), "after:", len(after.rows))


asyncio.run(main())
```

## When you don't need valid time

If your application only ever asks "what is true *now*", you do not need to do
anything. Every record is created with `valid_from = None` and
`valid_until = None`, which means "valid for all time", so:

- you never have to set a timestamp,
- queries that use no temporal predicate are unchanged, and
- if you *do* later run a `valid_now` / `valid_at` query, your never-annotated
  facts still match (open bounds are ±∞).

Valid time is a tool for the cases where history matters — auditing what an agent
believed at some past moment, or modelling facts with a real-world lifespan.
When that is not your problem, ignore it; it imposes no cost and changes no
existing behaviour.

## Next

- [MindQL](mindql.md) — the full query language the temporal predicates live in.
- [Upgrade Guide](upgrade.md) — how an existing database gains the valid-time
  columns automatically.
- [Core Concepts](concepts.md) — thoughts, edges, cycles, and the rest of the
  model.
- [Dreaming](dreaming.md) — how reflections (which inherit valid-time extent) are
  made.

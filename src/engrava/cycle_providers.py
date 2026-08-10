"""Reference :class:`~engrava.domain.protocols.cycle_provider.CycleProvider` implementations.

Three minimal, ready-to-use providers for the opt-in cycle-provider seam:

* :class:`StaticCycleProvider` — a fixed value; pure by construction.
* :class:`CallableCycleProvider` — a thin adapter over a consumer callable; its
  purity is the consumer's contract, not this adapter's.
* :class:`MaxCycleProvider` — an explicitly-cached, possibly-stale snapshot of a
  store's cognitive-cycle high-water mark, for restart-recovery.

None of these advances a cycle on its own: they either return a value they were
handed or, for :class:`MaxCycleProvider`, a snapshot the consumer explicitly
refreshes. The store validates every pulled value (a real, non-negative
``int``); a provider's *purity* — that its value is not wall-clock-derived and
does not conflate the operation-count / cognitive-cycle / wall-time axes —
remains the configuring consumer's contract (see
:class:`~engrava.domain.protocols.cycle_provider.CycleProvider`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from engrava.domain.protocols.engrava_core import EngravaCoreProtocol


class StaticCycleProvider:
    """A cycle provider that always returns a single fixed value.

    Pure by construction — it holds one configured integer and never mutates it.
    Useful when a consumer wants to pin recency/eligibility to a known cycle
    (for example a fixed replay cycle in a test or an offline batch job).

    The value is returned verbatim; the store validates it on pull, so an
    out-of-range value (a negative, or a ``bool``) surfaces as a
    :class:`~engrava.domain.exceptions.CycleProviderError` when the store first
    consults the provider, not at construction.

    Args:
        value: The fixed cognitive cycle to return from every
            :meth:`current_cycle` call.

    Examples:
        >>> provider = StaticCycleProvider(7)
        >>> provider.current_cycle()
        7

    """

    def __init__(self, value: int) -> None:
        self._value = value

    def current_cycle(self) -> int:
        """Return the fixed configured cycle.

        Returns:
            The value passed to the constructor.

        """
        return self._value


class CallableCycleProvider:
    """A thin adapter that turns a zero-argument callable into a cycle provider.

    Each :meth:`current_cycle` call simply invokes the wrapped callable and
    returns its result. This is a **non-mutating adapter**: it holds the
    callable reference but never mutates it.

    Purity is **the consumer's contract, not this adapter's**: whether the
    wrapped callable is free of wall-clock reads, I/O, and side effects — and
    whether it returns a genuine cognitive cycle rather than, say, a timestamp —
    depends entirely on the callable the consumer supplies. The store validates
    that each returned *value* is a real, non-negative ``int``; it cannot
    validate the callable's *behaviour*.

    Args:
        fn: A zero-argument callable returning the current cognitive cycle.

    Examples:
        >>> counter = {"cycle": 3}
        >>> provider = CallableCycleProvider(lambda: counter["cycle"])
        >>> provider.current_cycle()
        3
        >>> counter["cycle"] = 4
        >>> provider.current_cycle()
        4

    """

    def __init__(self, fn: Callable[[], int]) -> None:
        self._fn = fn

    def current_cycle(self) -> int:
        """Return the current cycle by invoking the wrapped callable.

        Returns:
            Whatever the wrapped callable returns for this call.

        """
        return self._fn()


class MaxCycleProvider:
    """A cached snapshot of a store's cognitive-cycle high-water mark.

    Because a store's high-water mark is read with the **async**
    :meth:`~engrava.domain.protocols.engrava_core.EngravaCoreProtocol.max_cycle`
    while :meth:`current_cycle` is a **sync** pull, this provider is explicitly
    cached: :meth:`create` snapshots the initial value, :meth:`current_cycle`
    returns that **cached** value, and :meth:`refresh` re-reads the store.

    The cache is **explicitly mutable and may be stale** — it is the last
    refreshed snapshot, never a live value. A consumer that advances its own
    cadence should call :meth:`refresh` when it wants the provider to catch up;
    between refreshes :meth:`current_cycle` keeps returning the previous
    snapshot. This suits restart-recovery (resume a counter from the store's
    high-water mark) far more than moment-to-moment cadence tracking.

    Construct it with the async :meth:`create` factory rather than the
    constructor, so the initial snapshot is taken from the store.

    Args:
        value: The initial cached high-water mark.
        source: The store to re-read on :meth:`refresh`.

    """

    def __init__(self, value: int, source: EngravaCoreProtocol) -> None:
        self._value = value
        self._source = source

    @classmethod
    async def create(cls, store: EngravaCoreProtocol) -> MaxCycleProvider:
        """Create a provider seeded with the store's current high-water mark.

        Args:
            store: The store to snapshot now and re-read on :meth:`refresh`.

        Returns:
            A :class:`MaxCycleProvider` whose cached value is
            ``await store.max_cycle()`` at construction time.

        """
        return cls(await store.max_cycle(), store)

    def current_cycle(self) -> int:
        """Return the last refreshed high-water snapshot.

        Returns:
            The cached value — the store's high-water mark as of the last
            :meth:`create` / :meth:`refresh`. This may be **stale**: it does not
            re-read the store.

        """
        return self._value

    async def refresh(self) -> int:
        """Re-read the store's high-water mark and update the cached value.

        Returns:
            The freshly read high-water mark, which also becomes the new cached
            value returned by subsequent :meth:`current_cycle` calls.

        """
        self._value = await self._source.max_cycle()
        return self._value

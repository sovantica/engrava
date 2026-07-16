"""CycleProvider — an opt-in, pull-only cognitive-cycle injection point.

Engrava's cognitive cycle is **consumer-owned**: ``current_cycle`` is a
parameter on the read/eligibility paths (``search_hybrid`` / ``recall`` recency,
``consolidate``, ``run_hygiene``); the core never advances it and runs no
background process. A consumer that already has a cadence can thread that value
through every call — or it can plug it **once** by configuring a
``CycleProvider`` on the store, which the resolution logic then pulls on demand.

This protocol is deliberately a **pull** interface with a single synchronous
method: the store asks the provider for a value when (and only when) a cycle is
needed and the caller did not pass one explicitly. It is *not* a push /
advancement mechanism — configuring a provider never causes the core to tick a
counter, run a daemon, or stamp writes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CycleProvider(Protocol):
    """Pull a cognitive-cycle value on demand.

    A store configured with a ``CycleProvider`` consults it to resolve the
    effective ``current_cycle`` for a read/eligibility path when the caller did
    not pass one explicitly (an explicit ``current_cycle`` — including ``0`` —
    always wins). Because the protocol is ``runtime_checkable`` it verifies only
    that an implementation *has* ``current_cycle`` — never that the returned
    value is a valid, pure cognitive cycle; the store validates the *value* on
    pull (a real, non-negative ``int``), while the provider's *purity* remains
    the configuring consumer's contract.

    **Normative provider contract.** A value returned to be used as a cognitive
    cycle **must not** be wall-clock-derived, nor otherwise conflate the three
    axes engrava keeps distinct (operation count, cognitive cycle, wall-time
    recency). The core cannot enforce this — it is the consumer's obligation.

    Implementations must be synchronous: ``current_cycle`` is called from the
    store's read paths and must return without awaiting.
    """

    def current_cycle(self) -> int:
        """Return the current cognitive cycle.

        Returns:
            The current cognitive cycle as a non-negative ``int``. The store
            validates this value when it is pulled and raises
            :class:`~engrava.domain.exceptions.CycleProviderError` if it is not
            a real, non-negative ``int``.

        """
        ...

"""Shared connection-revocation token for terminal store quarantine.

A single :class:`ConnectionRevocationToken` is created per store and shared with
every object that retains a *direct* reference to the real ``aiosqlite``
connection — the core and its :class:`~engrava.infrastructure.sqlite.journal_writer.JournalWriter`.
When the store quarantines the connection it revokes the token **synchronously**,
so a holder that bypasses the core's ``_db`` proxy still fails hard on its next
connection-touching method. This makes quarantine terminal by construction:
it no longer relies on the argued "core is the only caller" invariant, nor on a
physical ``close()`` succeeding.
"""

from __future__ import annotations

from engrava.domain.exceptions import ConnectionQuarantinedError


class ConnectionRevocationToken:
    """A shared, one-way revocation flag guarding a real connection.

    Created once per store; shared by every holder of the real connection. Once
    :meth:`revoke` is called the token stays revoked for its lifetime (quarantine
    is terminal — recovery requires a fresh connection + store).
    """

    __slots__ = ("reason", "revoked")

    def __init__(self) -> None:
        self.revoked: bool = False
        self.reason: str = "connection unusable"

    def revoke(self, reason: str) -> None:
        """Revoke the token; every subsequent :meth:`check` then raises.

        Args:
            reason: Human-readable cause, surfaced on :meth:`check`.

        """
        self.revoked = True
        self.reason = reason

    def check(self) -> None:
        """Raise if the connection has been revoked.

        Raises:
            ConnectionQuarantinedError: When the token has been revoked.

        """
        if self.revoked:
            raise ConnectionQuarantinedError(self.reason)

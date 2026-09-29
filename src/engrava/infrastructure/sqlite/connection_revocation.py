"""Shared connection-revocation token for terminal store quarantine.

A single :class:`ConnectionRevocationToken` is created per store and shared with
the core and its :class:`~engrava.infrastructure.sqlite.journal_writer.JournalWriter`,
which retain a *direct* reference to the real ``aiosqlite`` connection.
When the store quarantines the connection it revokes the token **synchronously**,
so a holder that bypasses the core's ``_db`` proxy fails hard when it checks the
token before using its connection.
"""

from __future__ import annotations

from engrava.domain.exceptions import ConnectionQuarantinedError


class ConnectionRevocationToken:
    """A shared, one-way revocation flag guarding a real connection.

    Created once per store; shared by the core and its journal writer. Once
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

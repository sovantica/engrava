"""Shared database-target resolution for the one-shot memory verbs.

``remember``, ``recall``, and ``link`` (see :mod:`engrava.cli.memory_commands`)
each need the *same* answer to "which database, opened how?" before they can
do anything else. The existing built-ins answer a narrower version of that
question three different ways in three different places:

* ``info`` / ``verify`` / ``query`` / ``gc`` open a bare ``aiosqlite``
  connection at ``cfg.db_path`` and wrap it in ``SqliteEngravaCore(conn)``
  directly — no embedding provider, no journal/search configuration, because
  none of those commands need one.
* ``snapshot`` / ``restore`` additionally resolve a ``--service`` name
  through :class:`~engrava.infrastructure.service_manager.EngravaManager`.
* ``remember`` / ``recall`` / ``link`` are the first CLI commands to open a
  store via
  :meth:`~engrava.infrastructure.sqlite.engrava_core.SqliteEngravaCore.from_config`
  — see ``_opened_full_store`` in :mod:`engrava.cli.memory_commands`, which
  dispatches to it whenever this module resolves the ``config`` tier below.
  Before that, a ``--config`` file's ``database.path`` and its ``embeddings``
  section were never consulted by any built-in command.

Copying the first (bare) pattern for ``recall`` would silently ignore
embedding, search, and journal configuration — a configured library
``recall()`` and a CLI ``recall`` would stop agreeing with each other. This
module is the single place that decides *which* database a memory-verb
invocation means, so that decision is made once and is reported identically
by every command that asks it.

**Precedence (highest wins):**

1. A non-empty ``--db`` value (or ``ENGRAVA_DB`` — :class:`EngravaCLIConfig`
   already folds the two together as "explicit"). This tier returns
   immediately, before ``cfg.config_path`` is even looked at: a ``--config``
   file, however broken, is never read for an invocation that gave a
   non-empty ``--db``. ``db_explicit`` is computed by truthiness (``bool(db_path)``
   in the ``cli()`` group callback in ``main.py``), matching
   :meth:`EngravaCLIConfig.resolve`'s own ``owned_db or ...`` fallback chain
   for consistency between the two — so ``--db ""`` is **not** "explicit" by
   this tier's own test, and falls through exactly as if ``--db`` had never
   been given at all (to ``ENGRAVA_DB``, then this module's tier 3). This is
   a deliberate mirror of shell convention (an empty string is "absence", not
   "a value"), not an oversight — but it does mean "an explicit ``--db`` flag"
   is imprecise: it is a *non-empty* one.
2. A ``--config`` file's own ``database.path`` — loaded via
   :func:`engrava.config.load_config`, so a configured embedding provider,
   search weights, and journal settings travel with it. A non-empty
   ``--config`` the caller named is validated here unconditionally: a missing
   or malformed file raises :class:`~engrava.config_validation.ConfigError`
   rather than being treated as though ``--config`` had never been given (see
   :func:`resolve_store_target`'s own ``Raises`` section). ``--config ""``
   is, by the same truthiness rule as tier 1, indistinguishable from omitting
   ``--config`` altogether — it is *not* validated, and this tier does not
   fire for it. Whatever the CLI's own default resolves to in tier 3 below is
   *not* a fallback for a non-empty ``--config`` that does not hold up.

Otherwise (``db_explicit`` is ``False`` and ``cfg.config_path`` is ``None``),
the CLI's own default (``./engrava.db``). ``ENGRAVA_DB`` never reaches this
tier: like ``--db``, it always sets ``db_explicit`` (see the ``cli()`` group
callback in ``main.py``), so it is folded into tier 1 above, not this one.

**No service tier.** None of ``remember`` / ``recall`` / ``link`` exposes a
``--service`` option (unlike ``snapshot`` / ``restore``), so a service
selected via ``services.default_service`` has no invocation that could ever
reach it here — the CLI group only ever loads ``services_cfg`` for the
``snapshot`` and ``restore`` subcommands themselves (see
``_SERVICES_CONFIG_COMMANDS`` in ``main.py``), which these three verbs are
not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from engrava.cli.config import EngravaCLIConfig

#: Which of the precedence tiers produced a :class:`ResolvedStore`.
StoreSource = Literal["db_flag", "config", "default"]


@dataclass(frozen=True)
class ResolvedStore:
    """The single database an invocation resolved to, and why.

    Attributes:
        db_path: The resolved SQLite file path.
        source: Which precedence tier produced ``db_path``.
        config_path: The ``--config`` file to build a full store from, when
            ``source == "config"``. ``None`` for every other source, since a
            bare (``db_flag`` / ``default``) store is not built from an
            ``engrava.yaml``.

    """

    db_path: Path
    source: StoreSource
    config_path: Path | None = None

    def describe(self) -> str:
        """Render the one-line ``--verbose`` resolution message.

        Returns:
            A human-readable line naming both the path and why it was
            chosen, for a command to echo to stderr under ``--verbose``.

        """
        reason = {
            "db_flag": "--db",
            "config": f"--config database.path ({self.config_path})",
            "default": "default",
        }[self.source]
        return f"Resolved database: {self.db_path} (source: {reason})"


def resolve_store_target(
    cfg: EngravaCLIConfig,
    *,
    db_explicit: bool,
) -> ResolvedStore:
    """Pick exactly one database for a memory-verb invocation.

    Args:
        cfg: The resolved CLI config (``--db`` / ``--config`` / defaults
            already folded in by :meth:`EngravaCLIConfig.resolve`).
        db_explicit: Whether ``--db`` (or ``ENGRAVA_DB``) was actually
            supplied, as opposed to ``cfg.db_path`` holding the CLI's own
            hardcoded default. Computed by the caller from the raw
            ``--db`` option value, since :class:`EngravaCLIConfig` itself no
            longer distinguishes "explicit" from "defaulted" once resolved.

    Returns:
        The resolved target, tagged with which tier produced it.

    Raises:
        engrava.config_validation.ConfigError: ``cfg.config_path`` was given
            (tier 2 is in play — ``db_explicit`` is ``False``) and it does
            not exist or fails to parse. A ``--config`` the caller named
            explicitly is never silently skipped in favour of the CLI's own
            default: the caller (see
            :func:`engrava.cli.memory_commands._resolve_for_command`) is
            responsible for turning this into a clean, exit-``2`` CLI
            failure — never a traceback.

    """
    if db_explicit:
        return ResolvedStore(db_path=cfg.db_path, source="db_flag")

    if cfg.config_path is not None:
        from engrava.config import load_config  # noqa: PLC0415

        parsed = load_config(cfg.config_path)
        return ResolvedStore(
            db_path=parsed.database_path,
            source="config",
            config_path=cfg.config_path,
        )

    return ResolvedStore(db_path=cfg.db_path, source="default")

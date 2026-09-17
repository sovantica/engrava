"""``--config`` really reaches the memory verbs' opened store, not just its path.

:mod:`engrava.cli.store_resolution` exists specifically so a ``--config``
file's ``search`` and ``journal`` sections travel with the database it names
— ``_opened_full_store`` (see :mod:`engrava.cli.memory_commands`) dispatches
the ``config`` tier through ``SqliteEngravaCore.from_config()`` rather than a
bare connection precisely so those sections are not silently dropped. Both
tests here need that positively demonstrated, not merely "a bare and a
configured call happen to agree" (two routes that both ignore the config
would agree with each other too):

* **Search weights.** A bare connection and a ``from_config`` connection would
  produce the *same* ranking whenever neither one activates a second signal
  (e.g. no vector arm, since nothing here needs the optional embeddings
  extra) — configured weights only become observable once something big
  enough to reorder results rides on them. ``search.default_priority_weight``
  is exactly that lever: at its always-on default (``0.05``) a strong lexical
  match wins comfortably, but a large configured value flips the ranking
  entirely in favour of a weak lexical match at higher priority. This needs
  no embedding provider at all.
* **Journaling.** ``journal.enabled`` is off by default (the journal table
  exists but stays empty). A ``--config`` that turns it on must produce
  journal rows through the CLI exactly as it would through a direct library
  call — checked directly against the on-disk ``journal_entry`` table, not
  through ``engrava verify`` (which would still report "valid" over an empty
  journal, so it cannot tell "journaling honoured" from "journaling silently
  dropped").
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from click.testing import CliRunner

from engrava.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


def _journal_entry_count(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM journal_entry").fetchone()
    finally:
        conn.close()
    return int(count)


class TestConfiguredSearchWeightsReachRecall:
    def test_configured_priority_weight_flips_the_ranking_lexical_search_would_pick(
        self, tmp_path: Path
    ) -> None:
        """A --config with a dominant priority weight outranks a strong lexical match.

        Two thoughts share the term "printer": one repeats it and is a strong
        FTS match at P4 (low priority), the other mentions it once and is a
        weak FTS match at P1 (high priority). At the CLI's own defaults
        (``default_fts_weight=0.3``, ``default_priority_weight=0.05``) the
        strong lexical match wins comfortably. A ``--config`` that sets
        ``default_fts_weight`` near zero and ``default_priority_weight`` far
        above it must flip the winner to the P1 thought — proving the CLI
        actually opened this database through the configured ``SearchConfig``
        rather than a bare connection that would keep the lexical-match
        winner regardless of what the file says.
        """
        db = tmp_path / "weighted.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(
            f"database:\n  path: {db}\n"
            "search:\n"
            "  default_fts_weight: 0.01\n"
            "  default_priority_weight: 50.0\n",
            encoding="utf-8",
        )
        runner = CliRunner()

        lexical_match = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "remember",
                "printer jam printer jam recurring issue printer",
                "--priority",
                "P4",
                "--json",
            ],
        )
        priority_match = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "remember",
                "printer maintenance schedule",
                "--priority",
                "P1",
                "--json",
            ],
        )
        assert lexical_match.exit_code == 0, lexical_match.output
        assert priority_match.exit_code == 0, priority_match.output

        configured = runner.invoke(
            cli, ["--config", str(config_path), "recall", "printer jam", "--json"]
        )
        assert configured.exit_code == 0, configured.output
        configured_payload = json.loads(configured.output.strip().splitlines()[-1])
        assert configured_payload["results"][0]["essence"] == "printer maintenance schedule", (
            "the configured, priority-dominant weights did not reach recall() -- "
            f"got: {configured_payload}"
        )

        # Control: the identical database read bare (no --config, so the
        # CLI's own hardcoded defaults apply) must pick the *other* winner --
        # otherwise this test would pass by coincidence regardless of whether
        # the config's weights were honoured.
        bare = runner.invoke(cli, ["--db", str(db), "recall", "printer jam", "--json"])
        assert bare.exit_code == 0, bare.output
        bare_payload = json.loads(bare.output.strip().splitlines()[-1])
        assert (
            bare_payload["results"][0]["essence"]
            == "printer jam printer jam recurring issue printer"
        )


class TestConfiguredJournalingReachesTheStore:
    def test_configured_journal_enabled_records_a_journal_entry(self, tmp_path: Path) -> None:
        db = tmp_path / "journaled.db"
        config_path = tmp_path / "engrava.yaml"
        config_path.write_text(
            f"database:\n  path: {db}\njournal:\n  enabled: true\n", encoding="utf-8"
        )
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--config", str(config_path), "remember", "a journaled thought"]
        )
        assert result.exit_code == 0, result.output
        assert _journal_entry_count(db) > 0, (
            "remember through a --config with journal.enabled: true left the "
            "journal_entry table empty -- configured journaling was not honoured"
        )

    def test_bare_remember_with_no_config_never_journals(self, tmp_path: Path) -> None:
        """Control: journaling is off by default, so a bare call writes no entries.

        Without this, a resolver that journaled unconditionally would still
        pass the positive assertion above by coincidence.
        """
        db = tmp_path / "unjournaled.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(db), "remember", "an unjournaled thought"])
        assert result.exit_code == 0, result.output
        assert _journal_entry_count(db) == 0

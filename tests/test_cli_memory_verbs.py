"""Tests for the one-shot memory verbs: ``remember``, ``recall``, ``link``.

Covers the acceptance surface for the new commands: the headline
store-then-search round trip, ``--type``/``--priority`` on ``remember``,
``--filter`` narrowing on ``recall``, edge creation and its two failure
exits, the malformed-``--meta``/``--filter`` usage errors (plain and
``--json``), stdin input, ``--dedup``, and the absent-database exit codes
(``3`` for a read, ``0`` **and** file creation for a write) -- plus the exact
``--json`` schemas (no undocumented fields), nested-parent-directory
creation, ``--top-k``/``--weight`` range validation ahead of any database
side effect, and the ``--db``-vs-``--config`` precedence and error-reporting
rules (an explicit ``--db`` never reads ``--config`` at all; a ``--config``
named without ``--db`` is validated unconditionally). The config-driven
embedding-provider criterion for ``backends_used`` lives in
:mod:`test_cli_memory_verbs_embedding`, and the configured-search-weights /
configured-journaling criteria live in :mod:`test_cli_memory_verbs_config`,
since both need more elaborate ``--config`` fixtures than the rest of this
file.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from click.testing import CliRunner

from engrava.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


def _last_line(output: str) -> str:
    """Return the final non-empty output line.

    ``CliRunner`` combines stdout and stderr into one ``.output`` string in
    this Click version. ``remember`` / ``link`` against a not-yet-existing
    database print a "Created database: ..." notice to stderr *before* the
    id they print to stdout, so extracting the id from a first call against a
    fresh path needs the last line, not the whole (stripped) blob.

    """
    lines = [line for line in output.splitlines() if line]
    return lines[-1] if lines else ""


def _stored_thought_row(db_path: Path, thought_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM thought WHERE thought_id = ?", (thought_id,)).fetchone()
    finally:
        conn.close()
    assert row is not None, f"thought {thought_id!r} not found in {db_path}"
    return row


def _stored_edge_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM edge").fetchall()
    finally:
        conn.close()


class TestRememberRecallRoundTrip:
    def test_remember_then_recall_returns_the_stored_thought(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()

        remember_result = runner.invoke(
            cli, ["--db", str(db), "remember", "The launch is scheduled for Tuesday"]
        )
        assert remember_result.exit_code == 0, remember_result.output
        thought_id = _last_line(remember_result.output)
        assert thought_id

        recall_result = runner.invoke(cli, ["--db", str(db), "recall", "launch schedule", "--json"])
        assert recall_result.exit_code == 0, recall_result.output
        payload = json.loads(_last_line(recall_result.output))
        assert payload["schema"] == "engrava.cli.recall.v1"
        returned_ids = {r["thought_id"] for r in payload["results"]}
        assert thought_id in returned_ids

    def test_remember_type_and_priority_are_the_stored_values(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "--db",
                str(db),
                "remember",
                "Escalate the outage to on-call",
                "--type",
                "REFLECTION",
                "--priority",
                "P1",
            ],
        )
        assert result.exit_code == 0, result.output
        thought_id = _last_line(result.output)

        row = _stored_thought_row(db, thought_id)
        assert row["thought_type"] == "REFLECTION"
        assert row["priority"] == "P1"

    def test_recall_filter_narrows_strictly(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()

        runner.invoke(
            cli,
            ["--db", str(db), "remember", "alpha note about weather", "--meta", "topic=weather"],
        )
        runner.invoke(
            cli,
            ["--db", str(db), "remember", "beta note about finance", "--meta", "topic=finance"],
        )

        unfiltered = json.loads(
            runner.invoke(cli, ["--db", str(db), "recall", "note", "--json"]).output
        )
        filtered = json.loads(
            runner.invoke(
                cli, ["--db", str(db), "recall", "note", "--filter", "topic=weather", "--json"]
            ).output
        )

        assert len(filtered["results"]) < len(unfiltered["results"])
        assert len(filtered["results"]) >= 1
        for row in filtered["results"]:
            stored = _stored_thought_row(db, row["thought_id"])
            assert json.loads(stored["metadata_json"])["topic"] == "weather"

    def test_remember_reads_text_from_stdin(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "remember", "-"], input="content piped over stdin\n"
        )
        assert result.exit_code == 0, result.output
        thought_id = _last_line(result.output)
        row = _stored_thought_row(db, thought_id)
        assert row["content"] == "content piped over stdin\n"

    def test_remember_dedup_returns_the_same_existing_id_twice(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()

        first = runner.invoke(
            cli, ["--db", str(db), "remember", "identical content for dedup", "--dedup"]
        )
        second = runner.invoke(
            cli, ["--db", str(db), "remember", "identical content for dedup", "--dedup"]
        )
        assert first.exit_code == 0
        assert second.exit_code == 0
        assert _last_line(first.output) == _last_line(second.output)

        conn = sqlite3.connect(db)
        try:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM thought WHERE content = ?",
                ("identical content for dedup",),
            ).fetchone()
        finally:
            conn.close()
        assert count == 1


class TestAbsentDatabase:
    def test_recall_against_absent_db_exits_3_naming_the_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(missing), "recall", "anything"])
        assert result.exit_code == 3
        assert str(missing) in result.output
        assert not missing.exists()

    def test_remember_against_absent_db_creates_it_and_reports_the_path(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "fresh.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(missing), "remember", "first thought here"])
        assert result.exit_code == 0
        assert missing.exists()
        # click's CliRunner mixes stdout/stderr by default; the created-path
        # notice and the printed id both land in .output.
        assert str(missing) in result.output

    def test_remember_creates_every_missing_nested_parent_directory(self, tmp_path: Path) -> None:
        """``remember`` must create a multi-level missing parent, not just a leaf.

        ``_opened_full_store`` calls ``mkdir(parents=True, ...)`` rather than
        a bare ``mkdir()`` -- a single-level check (a database directly under
        an already-existing ``tmp_path``, as in the sibling test above) cannot
        tell the two apart, since both create the one missing level the same
        way.
        """
        missing = tmp_path / "a" / "b" / "c" / "deep.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(missing), "remember", "buried thought"])
        assert result.exit_code == 0, result.output
        assert missing.exists()
        assert missing.parent.is_dir()


class TestLink:
    def test_link_persists_type_and_weight_read_back_through_get_edges(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        from_id = _last_line(
            runner.invoke(cli, ["--db", str(db), "remember", "source thought"]).output
        )
        to_id = _last_line(
            runner.invoke(cli, ["--db", str(db), "remember", "target thought"]).output
        )

        result = runner.invoke(
            cli,
            ["--db", str(db), "link", from_id, to_id, "--type", "DEPENDS_ON", "--weight", "0.42"],
        )
        assert result.exit_code == 0, result.output

        rows = _stored_edge_rows(db)
        assert len(rows) == 1
        assert rows[0]["from_thought_id"] == from_id
        assert rows[0]["to_thought_id"] == to_id
        assert rows[0]["edge_type"] == "DEPENDS_ON"
        assert rows[0]["weight"] == 0.42

    def test_link_invalid_edge_type_exits_2_naming_value_and_every_valid_member(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        from_id = _last_line(runner.invoke(cli, ["--db", str(db), "remember", "a"]).output)
        to_id = _last_line(runner.invoke(cli, ["--db", str(db), "remember", "b"]).output)

        result = runner.invoke(
            cli, ["--db", str(db), "link", from_id, to_id, "--type", "NOT_A_REAL_TYPE"]
        )
        assert result.exit_code == 2
        assert "NOT_A_REAL_TYPE" in result.output
        for valid in (
            "ASSOCIATED",
            "DEPENDS_ON",
            "DERIVED_FROM",
            "MESSAGE_OF",
            "BRIDGE",
            "CONSOLIDATED_FROM",
            "CONTESTED_BY",
        ):
            assert valid in result.output

    def test_link_missing_source_exits_4_naming_the_source(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        to_id = _last_line(runner.invoke(cli, ["--db", str(db), "remember", "target"]).output)

        result = runner.invoke(
            cli, ["--db", str(db), "link", "ghost-source", to_id, "--type", "ASSOCIATED"]
        )
        assert result.exit_code == 4
        assert "ghost-source" in result.output
        assert "from_thought_id" in result.output

    def test_link_missing_target_exits_4_naming_the_target(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        from_id = _last_line(runner.invoke(cli, ["--db", str(db), "remember", "source"]).output)

        result = runner.invoke(
            cli, ["--db", str(db), "link", from_id, "ghost-target", "--type", "ASSOCIATED"]
        )
        assert result.exit_code == 4
        assert "ghost-target" in result.output
        assert "to_thought_id" in result.output

    def test_link_missing_source_json_emits_error_object(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        to_id = _last_line(runner.invoke(cli, ["--db", str(db), "remember", "target"]).output)

        result = runner.invoke(
            cli,
            ["--db", str(db), "link", "ghost-source", to_id, "--type", "ASSOCIATED", "--json"],
        )
        assert result.exit_code == 4
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "missing_thought"
        assert "ghost-source" in payload["message"]

    def test_link_creates_the_database_when_absent(self, tmp_path: Path) -> None:
        # link's two thoughts cannot exist yet on a database link itself just
        # created, so this always ends in the exit-4 missing-source refusal —
        # what this test actually checks is that the database file exists
        # afterwards (created before the referential check ran), not that
        # the edge was written.
        missing = tmp_path / "fresh-link.db"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(missing), "link", "a", "b", "--type", "ASSOCIATED"]
        )
        assert missing.exists()
        assert str(missing) in result.output
        assert result.exit_code == 4


class TestMalformedOptions:
    def test_remember_malformed_meta_exits_2_naming_the_token(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(db), "remember", "x", "--meta", "not-a-pair"])
        assert result.exit_code == 2
        assert "not-a-pair" in result.output

    def test_remember_malformed_meta_json_emits_error_object(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "remember", "x", "--meta", "not-a-pair", "--json"]
        )
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert "not-a-pair" in payload["message"]

    def test_recall_malformed_filter_exits_2_naming_the_token(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        runner.invoke(cli, ["--db", str(db), "remember", "seed thought"])
        result = runner.invoke(cli, ["--db", str(db), "recall", "seed", "--filter", "not-a-pair"])
        assert result.exit_code == 2
        assert "not-a-pair" in result.output

    def test_recall_malformed_filter_json_emits_error_object(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        runner.invoke(cli, ["--db", str(db), "remember", "seed thought"])
        result = runner.invoke(
            cli, ["--db", str(db), "recall", "seed", "--filter", "not-a-pair", "--json"]
        )
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert "not-a-pair" in payload["message"]


class TestJsonSchemas:
    def test_remember_json_schema(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(db), "remember", "x", "--json"])
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.remember.v1"
        assert set(payload) == {"schema", "thought_id", "deduplicated"}
        assert payload["deduplicated"] is False

    def test_remember_dedup_json_reports_deduplicated_true_on_the_second_call(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        content = "identical content for dedup json"

        first = runner.invoke(cli, ["--db", str(db), "remember", content, "--dedup", "--json"])
        second = runner.invoke(cli, ["--db", str(db), "remember", content, "--dedup", "--json"])
        assert first.exit_code == 0, first.output
        assert second.exit_code == 0, second.output

        first_payload = json.loads(_last_line(first.output))
        second_payload = json.loads(_last_line(second.output))
        assert first_payload["deduplicated"] is False
        assert second_payload["deduplicated"] is True
        assert second_payload["thought_id"] == first_payload["thought_id"]

    def test_recall_json_schema_has_no_undocumented_fields(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        runner.invoke(cli, ["--db", str(db), "remember", "a thought to find"])
        result = runner.invoke(cli, ["--db", str(db), "recall", "thought", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.recall.v1"
        assert set(payload) == {"schema", "query", "top_k", "backends_used", "results"}
        assert set(payload["results"][0]) == {"thought_id", "score", "essence"}

    def test_link_json_schema(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        from_id = _last_line(runner.invoke(cli, ["--db", str(db), "remember", "a"]).output)
        to_id = _last_line(runner.invoke(cli, ["--db", str(db), "remember", "b"]).output)
        result = runner.invoke(
            cli, ["--db", str(db), "link", from_id, to_id, "--type", "ASSOCIATED", "--json"]
        )
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.link.v1"
        assert set(payload) == {
            "schema",
            "edge_id",
            "from_thought_id",
            "to_thought_id",
            "edge_type",
            "weight",
        }


class TestConfigPrecedence:
    """An explicit ``--db`` beats even a broken ``--config``; a named ``--config`` must hold up.

    Covers the two-sided fix: ``--db`` wins outright and never causes
    ``--config`` to be read at all (so a malformed file next to it is
    irrelevant), while a ``--config`` given *without* an explicit ``--db``
    is validated unconditionally -- a missing or malformed file is always
    reported, never silently swapped for the CLI's own default database.
    """

    def test_explicit_db_wins_over_a_malformed_config(self, tmp_path: Path) -> None:
        db = tmp_path / "target.db"
        bad_config = tmp_path / "broken.yaml"
        bad_config.write_text("not: valid: yaml: [", encoding="utf-8")
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "--config", str(bad_config), "remember", "hello"]
        )
        assert result.exit_code == 0, result.output
        assert db.exists()

    def test_explicit_db_wins_over_a_missing_config(self, tmp_path: Path) -> None:
        db = tmp_path / "target.db"
        missing_config = tmp_path / "does-not-exist.yaml"
        runner = CliRunner()

        result = runner.invoke(
            cli, ["--db", str(db), "--config", str(missing_config), "remember", "hello"]
        )
        assert result.exit_code == 0, result.output
        assert db.exists()

    def test_missing_explicit_config_without_db_is_an_error_not_a_silent_default(
        self, tmp_path: Path
    ) -> None:
        missing_config = tmp_path / "does-not-exist.yaml"
        default_db = tmp_path / "engrava.db"
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["--config", str(missing_config), "remember", "hello"])
        assert result.exit_code == 2
        assert str(missing_config) in result.output
        assert not default_db.exists()

    def test_malformed_config_without_db_is_a_clean_error_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        bad_config = tmp_path / "broken.yaml"
        bad_config.write_text("not: valid: yaml: [", encoding="utf-8")
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["--config", str(bad_config), "remember", "hello"])
        assert result.exit_code == 2
        assert "Traceback" not in result.output

    def test_malformed_config_without_db_json_emits_error_object(self, tmp_path: Path) -> None:
        bad_config = tmp_path / "broken.yaml"
        bad_config.write_text("not: valid: yaml: [", encoding="utf-8")
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["--config", str(bad_config), "remember", "hello", "--json"]
            )
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "invalid_config"


class TestOptionShapedValueIsRejected:
    """``--db``/``--config`` refuse a value that looks like another flag.

    Click (like ``argparse``) binds the token immediately following a
    string option to that option, even when the token itself starts with
    ``-`` and spells out another known flag. Left unchecked, ``--db --json
    remember TEXT`` creates a database literally named ``--json``, stores
    the thought there, and exits ``0`` -- a silent write to the wrong place
    that also defeats the caller's requested ``--json`` output, since the
    real ``--json`` flag was never seen. This must be rejected as a usage
    error instead, before any command body (and so before any database is
    created).
    """

    def test_db_flag_shaped_value_is_rejected_before_any_database_is_created(
        self, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["--db", "--json", "remember", "hello world"])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
        assert "--json" in result.output
        assert not (tmp_path / "--json").exists()

    def test_config_flag_shaped_value_is_rejected(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["--config", "--verbose", "remember", "hello"])
        assert result.exit_code == 2
        assert "Traceback" not in result.output

    def test_dot_slash_prefixed_dash_path_still_works(self, tmp_path: Path) -> None:
        """The documented escape hatch (a leading ``./``) is not itself rejected."""
        import pathlib

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            result = runner.invoke(cli, ["--db", "./--not-really-a-flag", "remember", "hello"])
            assert result.exit_code == 0, result.output
            assert (pathlib.Path(cwd) / "--not-really-a-flag").exists()


class TestNumericValidation:
    """``--top-k`` / ``--weight`` are range-checked before anything opens or is created.

    Both checks run ahead of ``_resolve_for_command`` / ``_opened_full_store``
    in their respective commands, so a rejected value must leave no database
    (and no parent directory) behind -- proving the check ran *before* the
    side effect, not merely that the side effect was later rolled back (it
    never is; nothing here uses a transaction that could be).
    """

    def test_recall_rejects_negative_top_k_before_any_lookup(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "missing.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(db), "recall", "anything", "--top-k", "-1"])
        assert result.exit_code == 2
        assert "-1" in result.output
        assert not db.parent.exists()

    def test_recall_rejects_zero_top_k(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        runner.invoke(cli, ["--db", str(db), "remember", "seed"])
        result = runner.invoke(cli, ["--db", str(db), "recall", "seed", "--top-k", "0"])
        assert result.exit_code == 2

    def test_recall_negative_top_k_json_emits_error_object(self, tmp_path: Path) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "recall", "anything", "--top-k", "-1", "--json"]
        )
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "invalid_top_k"
        assert "-1" in payload["message"]

    def test_link_rejects_out_of_range_weight_before_creating_the_database(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "deep" / "nested" / "missing.db"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--db", str(db), "link", "a", "b", "--type", "ASSOCIATED", "--weight", "1.5"]
        )
        assert result.exit_code == 2
        assert "1.5" in result.output
        assert not db.exists()
        assert not db.parent.exists()

    def test_link_negative_weight_json_emits_error_object_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--db",
                str(db),
                "link",
                "a",
                "b",
                "--type",
                "ASSOCIATED",
                "--weight",
                "-0.1",
                "--json",
            ],
        )
        assert result.exit_code == 2
        payload = json.loads(_last_line(result.output))
        assert payload["schema"] == "engrava.cli.error.v1"
        assert payload["error"] == "invalid_weight"
        assert "-0.1" in payload["message"]
        assert not db.exists()


class TestEmptyTextMessage:
    def test_whitespace_only_text_message_reproduces_the_offending_value(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "m.db"
        runner = CliRunner()
        result = runner.invoke(cli, ["--db", str(db), "remember", "   "])
        assert result.exit_code == 2
        assert repr("   ") in result.output

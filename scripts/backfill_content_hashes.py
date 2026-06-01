"""Backfill ``thought.content_hash`` for databases upgraded from core-9.

Engrava's core-10 schema adds a nullable ``content_hash`` column to the
``thought`` table.  Rows ingested before the upgrade leave that column
``NULL`` until this utility (or a deliberate downstream pass) populates
them, which means opt-in deduplication via
``SqliteEngravaCore.create_thought(..., deduplicate=True)`` cannot detect
duplicates of pre-upgrade content until the backfill has run.

The script is fully idempotent — it only touches rows where
``content_hash IS NULL``, so re-running it after a partial pass converges
on the fully populated state.

Usage::

    python -m scripts.backfill_content_hashes --db-path ./engrava.sqlite3
    python -m scripts.backfill_content_hashes --db-path ./engrava.sqlite3 --batch-size 5000
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

import aiosqlite


def _compute_hash(content: str) -> str:
    """Return the SHA-256 hex digest of *content* (UTF-8, no normalization).

    Mirrors ``engrava.infrastructure.sqlite.engrava_core._compute_content_hash``
    so backfilled hashes are byte-identical to those produced at insert
    time after the upgrade.

    Args:
        content: The thought content string.

    Returns:
        Lowercase hex digest of ``sha256(content.encode("utf-8"))``.

    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def backfill(db_path: Path, batch_size: int = 1000) -> int:
    """Populate ``content_hash`` for rows where it is currently ``NULL``.

    Runs in batches to keep memory usage bounded for large databases and
    commits per batch so a long-running backfill can be interrupted and
    resumed without losing already-applied progress.

    Args:
        db_path: Path to the SQLite database file.
        batch_size: Number of rows to update per transaction (must be
            positive).

    Returns:
        The total number of rows updated.

    Raises:
        FileNotFoundError: If *db_path* does not exist.
        ValueError: If *batch_size* is not positive.

    """
    if not db_path.exists():
        msg = f"Database file does not exist: {db_path}"
        raise FileNotFoundError(msg)
    if batch_size < 1:
        msg = "batch_size must be a positive integer"
        raise ValueError(msg)

    total_updated = 0
    async with aiosqlite.connect(str(db_path)) as db:
        while True:
            cursor = await db.execute(
                "SELECT thought_id, content FROM thought WHERE content_hash IS NULL LIMIT ?",
                (batch_size,),
            )
            rows = list(await cursor.fetchall())
            if not rows:
                break

            for thought_id, content in rows:
                await db.execute(
                    "UPDATE thought SET content_hash = ? WHERE thought_id = ?",
                    (_compute_hash(content), thought_id),
                )
            await db.commit()
            total_updated += len(rows)

            # When the final batch is short, no further rows remain.
            if len(rows) < batch_size:
                break

    return total_updated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the engrava SQLite database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows to update per transaction (default: 1000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Always returns ``0`` on a successful backfill — exit codes are kept
    binary (``0`` success / non-zero failure) so shell scripts can chain
    the backfill into pipelines without parsing.  The precise number of
    rows that were updated is written to stdout as a single human-
    readable line so operators and CI logs can inspect the run.

    Failures inside ``backfill`` (missing DB file, bad ``--batch-size``,
    SQLite errors) propagate as exceptions and translate to a non-zero
    exit code via ``raise SystemExit(...)`` at the entry point — they
    are not swallowed here.
    """
    args = parse_args(argv)
    updated = asyncio.run(backfill(args.db_path, args.batch_size))
    sys.stdout.write(f"Backfilled content_hash for {updated} thought(s).\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

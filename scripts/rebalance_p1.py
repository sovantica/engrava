"""Rebalance excess P1 thoughts back to P2 to meet the ``max_p1_fraction`` cap.

The configurable ``DreamingConfig.max_p1_fraction`` cap (default 5 %) prevents
``dreaming.promote`` from flooding the corpus with P1 thoughts. Existing
databases populated before this cap was in place may contain many more P1
thoughts than the target fraction allows.

This utility performs a one-off rebalance: it counts the current P1 total,
computes the excess against the requested cap, and demotes the **oldest**
P1 thoughts (by ``created_at`` ascending) back to P2.  Demoting the oldest
first preserves the ranking signal for the most recently promoted thoughts,
which are likely to be the most relevant.

The script is **fully idempotent** — running it multiple times on the same
database converges to the same final state.

Usage::

    python -m scripts.rebalance_p1 --db-path ./engrava.sqlite3
    python -m scripts.rebalance_p1 --db-path ./engrava.sqlite3 --max-p1-fraction 0.10
    python -m scripts.rebalance_p1 --db-path ./engrava.sqlite3 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


async def rebalance(
    db_path: Path,
    max_p1_fraction: float = 0.05,
    *,
    dry_run: bool = False,
) -> int:
    """Demote excess P1 thoughts to P2 to meet the fraction cap.

    Computes the allowed P1 count as
    ``max(1, int(total_thoughts * max_p1_fraction))``, then demotes the
    oldest ``(current_p1 - allowed_p1)`` P1 thoughts back to P2.

    Args:
        db_path: Path to the SQLite database file.
        max_p1_fraction: Maximum allowed fraction of P1 thoughts
            (0.0 to 1.0, inclusive).  Defaults to ``0.05``.
        dry_run: When ``True``, report what *would* be demoted without
            writing any changes.

    Returns:
        Number of thoughts demoted (or that *would* be demoted in
        ``dry_run`` mode).

    Raises:
        ValueError: If ``max_p1_fraction`` is outside ``[0.0, 1.0]``.

    """
    if not 0.0 <= max_p1_fraction <= 1.0:
        msg = f"max_p1_fraction must be in [0.0, 1.0], got {max_p1_fraction}"
        raise ValueError(msg)

    async with aiosqlite.connect(str(db_path)) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM thought")
        row = await cursor.fetchone()
        total = int(row[0]) if row else 0

        cursor = await db.execute("SELECT COUNT(*) FROM thought WHERE priority = 'P1'")
        row = await cursor.fetchone()
        current_p1 = int(row[0]) if row else 0

        max_p1 = max(1, int(total * max_p1_fraction))

        if current_p1 <= max_p1:
            logger.info(
                "rebalance_p1: P1 fraction within cap (%d/%d = %.1f%% ≤ %.1f%%), "
                "no rebalance needed",
                current_p1,
                total,
                current_p1 * 100.0 / total if total else 0.0,
                max_p1_fraction * 100.0,
            )
            return 0

        excess = current_p1 - max_p1
        logger.info(
            "rebalance_p1: demoting %d excess P1 thoughts to P2 "
            "(current %d, allowed %d, total %d)%s",
            excess,
            current_p1,
            max_p1,
            total,
            " [DRY RUN]" if dry_run else "",
        )

        if dry_run:
            return excess

        # Demote oldest P1 thoughts first (newest are most likely relevant).
        await db.execute(
            """
            UPDATE thought
            SET priority = 'P2'
            WHERE thought_id IN (
                SELECT thought_id
                FROM thought
                WHERE priority = 'P1'
                ORDER BY created_at ASC
                LIMIT ?
            )
            """,
            (excess,),
        )
        await db.commit()
        return excess


def main(argv: list[str] | None = None) -> int:
    """Entry point for command-line invocation.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 on success, non-zero on error).

    """
    parser = argparse.ArgumentParser(
        description="Demote excess P1 thoughts to P2 to meet the P1 fraction cap.",
    )
    parser.add_argument("--db-path", required=True, type=Path, help="Path to the SQLite DB.")
    parser.add_argument(
        "--max-p1-fraction",
        type=float,
        default=0.05,
        help="Maximum allowed P1 fraction (default: 0.05).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be demoted without writing changes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    args = parser.parse_args(argv)

    if args.verbose or args.dry_run:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        demoted = asyncio.run(
            rebalance(
                db_path=args.db_path,
                max_p1_fraction=args.max_p1_fraction,
                dry_run=args.dry_run,
            )
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    prefix = "[DRY RUN] Would demote" if args.dry_run else "Demoted"
    print(f"{prefix} {demoted} thoughts from P1 to P2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

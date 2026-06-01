"""Re-enrich legacy v1 REFLECTION content to schema v2.

Engrava's structural REFLECTION schema bumped from v1 (3 fields:
``member_ids``, ``keywords``, ``cluster_hash``) to v2 (additive — 12
fields total) once the dreaming extension started emitting the full
schema mandated by the cognitive-boundary spec.  Existing v1 rows
keep working because the v2 reader detects them via *absence* of the
``version`` field and returns the dict as-is — but they miss the
enrichment fields (``top_keyphrases``, ``member_excerpts``,
``temporal_span``, ``named_entities``) that downstream readers (LLM
judges, semantic search, agent retrieval) benefit from.

This utility walks ``thought`` rows of type ``REFLECTION``, parses
the legacy v1 content, hydrates the cluster's member ``ThoughtRecord``
instances from the same database, calls the same v2 builder the
dreaming extension uses, and writes the result back via ``UPDATE``.

The script is fully idempotent — only rows where the content lacks a
``version`` field (i.e. legacy v1) are touched.  Re-running on a
fully migrated DB is a no-op.

Usage::

    python -m scripts.reenrich_reflections_to_v2 --db-path ./engrava.sqlite3
    python -m scripts.reenrich_reflections_to_v2 --db-path ./engrava.sqlite3 --batch-size 50
    python -m scripts.reenrich_reflections_to_v2 --db-path ./engrava.sqlite3 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

import aiosqlite

from engrava.config import DreamingConfig
from engrava.domain.enums import (
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.thought import ThoughtRecord
from engrava.extensions.dreaming_reflection_content import build_reflection_content_v2

_REFLECTION_TYPE = "REFLECTION"


async def _fetch_legacy_reflection_batch(
    db: aiosqlite.Connection,
    *,
    batch_size: int,
    after_thought_id: str | None,
) -> list[tuple[str, str]]:
    """Return ``(thought_id, content)`` pairs for legacy v1 REFLECTION rows.

    A row counts as legacy v1 when ``json_valid(content)`` is true and
    ``json_extract(content, '$.version')`` is ``NULL``.  The dreaming
    extension never wrote ``version`` for v1, so this filter is the
    canonical legacy detector.  Rows with malformed JSON in ``content``
    are silently excluded — the script leaves them in place rather
    than crashing the entire run on a single corrupt row.

    Pagination is keyed off the lexicographically-greatest thought_id
    seen so far, supplied by the caller as *after_thought_id*.  This
    keeps the loop terminating in dry-run mode (where the legacy filter
    keeps matching the same rows because no UPDATE is issued) and is
    cheaper than a global ``OFFSET`` scan because the ``thought_id``
    PRIMARY KEY index gives O(log N) seeks.
    """
    # ``json_valid(content)`` is evaluated *before* ``json_extract`` so a
    # row with malformed JSON does not raise ``OperationalError: malformed
    # JSON`` mid-SELECT — SQLite short-circuits the AND chain.  Without
    # this guard a single corrupt REFLECTION blocks the whole re-enrichment
    # run; with it, malformed rows are simply not picked up by the filter
    # and stay in place untouched.
    if after_thought_id is None:
        cursor = await db.execute(
            """
            SELECT thought_id, content
            FROM thought
            WHERE thought_type = ?
              AND json_valid(content)
              AND json_extract(content, '$.version') IS NULL
            ORDER BY thought_id
            LIMIT ?
            """,
            (_REFLECTION_TYPE, batch_size),
        )
    else:
        cursor = await db.execute(
            """
            SELECT thought_id, content
            FROM thought
            WHERE thought_type = ?
              AND json_valid(content)
              AND json_extract(content, '$.version') IS NULL
              AND thought_id > ?
            ORDER BY thought_id
            LIMIT ?
            """,
            (_REFLECTION_TYPE, after_thought_id, batch_size),
        )
    rows = await cursor.fetchall()
    return [(row[0], row[1]) for row in rows]


async def _fetch_member_thought(db: aiosqlite.Connection, thought_id: str) -> ThoughtRecord | None:
    """Hydrate a single ``ThoughtRecord`` from the database (or ``None``)."""
    cursor = await db.execute(
        """
        SELECT thought_id, thought_type, essence, content, priority,
               lifecycle_status, created_cycle, updated_cycle, source,
               confidence, embedding_ref, source_type, confirmation_count,
               consolidated_from, visibility, access_count,
               last_accessed_at, created_at, updated_at, expires_at
        FROM thought
        WHERE thought_id = ?
        """,
        (thought_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return ThoughtRecord(
        thought_id=row[0],
        thought_type=ThoughtType(row[1]),
        essence=row[2],
        content=row[3],
        priority=Priority(row[4]),
        lifecycle_status=LifecycleStatus(row[5]),
        created_cycle=row[6],
        updated_cycle=row[7],
        source=row[8],
        confidence=row[9],
        embedding_ref=row[10],
        source_type=KnowledgeSource(row[11]) if row[11] else KnowledgeSource.EXPERIENCE,
        confirmation_count=row[12] or 0,
        consolidated_from=json.loads(row[13]) if row[13] else None,
        visibility=ThoughtVisibility(row[14]) if row[14] else ThoughtVisibility.SELECTIVE,
        access_count=row[15] or 0,
        last_accessed_at=row[16],
        created_at=row[17],
        updated_at=row[18],
        expires_at=row[19],
    )


async def _hydrate_cluster(db: aiosqlite.Connection, member_ids: list[str]) -> list[ThoughtRecord]:
    """Return cluster members that still exist in the database."""
    members: list[ThoughtRecord] = []
    for tid in member_ids:
        thought = await _fetch_member_thought(db, tid)
        if thought is not None:
            members.append(thought)
    return members


async def _reenrich_one(
    db: aiosqlite.Connection,
    *,
    thought_id: str,
    legacy_content: str,
    config: DreamingConfig,
    now: datetime.datetime,
    dry_run: bool,
) -> bool:
    """Build and persist v2 content for a single legacy reflection.

    Returns ``True`` when the row qualifies for enrichment (and, in
    non-dry-run mode, an UPDATE was issued).  Returns ``False`` when
    the row cannot be enriched (legacy content malformed, every
    member deleted, etc.) and is left in place.

    In ``dry_run`` mode the v2 content is built and discarded — no
    UPDATE is issued.  The caller can therefore drive the same row
    set forward by ``thought_id`` without the legacy filter
    re-matching the same rows on the next batch.

    """
    try:
        legacy = json.loads(legacy_content)
    except json.JSONDecodeError:
        return False

    member_ids_raw = legacy.get("member_ids")
    if not isinstance(member_ids_raw, list) or not member_ids_raw:
        return False

    members = await _hydrate_cluster(db, [str(m) for m in member_ids_raw])
    if not members:
        return False

    legacy_algorithm = legacy.get("cluster_algorithm")
    algorithm = (
        legacy_algorithm
        if isinstance(legacy_algorithm, str) and legacy_algorithm
        else "agglomerative"
    )

    # Build the v2 content in both modes — exercising the builder
    # surfaces malformed-cluster issues (missing members, bad
    # legacy fields) before the operator commits to a full run.
    build_reflection_content_v2(
        members,
        algorithm=algorithm,
        config=config,
        now=now,
    )

    if dry_run:
        return True

    v2_content = build_reflection_content_v2(
        members,
        algorithm=algorithm,
        config=config,
        now=now,
    )
    new_content_str = json.dumps(v2_content, ensure_ascii=False)

    await db.execute(
        "UPDATE thought SET content = ? WHERE thought_id = ?",
        (new_content_str, thought_id),
    )
    return True


async def reenrich(
    db_path: Path,
    *,
    batch_size: int = 100,
    dry_run: bool = False,
    config: DreamingConfig | None = None,
) -> int:
    """Walk legacy v1 REFLECTIONs and write v2 content back.

    Args:
        db_path: Path to the SQLite database file.
        batch_size: Number of rows to fetch per pass (must be positive).
        dry_run: When ``True``, log what would be updated without
            issuing any ``UPDATE`` statement.
        config: Optional ``DreamingConfig`` carrying the
            ``top_keyphrases_count`` and ``top_member_excerpts_count``
            fields used by the builder.  Defaults to a fresh
            ``DreamingConfig()`` instance (defaults: 3 / 5).

    Returns:
        Total number of rows enriched (or that would be enriched, in
        dry-run mode).

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

    effective_config = config if config is not None else DreamingConfig()
    now = datetime.datetime.now(datetime.UTC)

    total = 0
    after_thought_id: str | None = None
    async with aiosqlite.connect(str(db_path)) as db:
        while True:
            batch = await _fetch_legacy_reflection_batch(
                db,
                batch_size=batch_size,
                after_thought_id=after_thought_id,
            )
            if not batch:
                break

            for thought_id, legacy_content in batch:
                ok = await _reenrich_one(
                    db,
                    thought_id=thought_id,
                    legacy_content=legacy_content,
                    config=effective_config,
                    now=now,
                    dry_run=dry_run,
                )
                if ok:
                    total += 1

            # Advance the pagination cursor regardless of mode — the
            # batch is sorted by thought_id ascending, so the last row
            # is the greatest id we have seen.  In production mode an
            # UPDATE flips the row out of the legacy filter and the
            # plain ``LIMIT`` would be enough; in dry-run mode the
            # cursor is what guarantees forward progress because no
            # UPDATE is issued.
            after_thought_id = batch[-1][0]

            if not dry_run:
                await db.commit()

            # When a short batch comes back, no further rows remain.
            if len(batch) < batch_size:
                break

    return total


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
        default=100,
        help="Rows to enrich per transaction (default: 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be updated without writing anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns 0 on success, 1 on failure."""
    args = parse_args(argv)
    try:
        total = asyncio.run(
            reenrich(
                args.db_path,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"reenrich failed: {exc}\n")
        return 1

    suffix = " (dry run — no rows written)" if args.dry_run else ""
    sys.stdout.write(f"Re-enriched {total} legacy v1 reflection(s) to v2{suffix}.\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

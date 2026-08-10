"""Archived thoughts leave the default retrieval set — reversibly, and via every path.

Archiving a thought (the hygiene forgetting loop, or a TTL ``archive`` sweep)
sets ``lifecycle_status = 'ARCHIVED'`` while keeping the FTS row and the
embedding. This module pins the contract that such a thought:

* is excluded from the default candidate set on **every** retrieval path — the
  FTS and vector arms, the query-less fallback, ``recall``, and the
  ``CONSOLIDATED_FROM`` graph expansion (the path that could otherwise re-inject
  an archived source OBSERVATION through an ACTIVE seed REFLECTION);
* is never silently lost — it stays fetchable by id, reachable via
  ``include_archived=True``, and fully restorable to the default set through
  :meth:`SqliteEngravaCore.restore_thought`;
* leaves the default ranked result **byte-identical** when nothing is archived
  (the exclusion is inert on an all-active corpus); and
* is excluded consistently regardless of *how* it became archived — a
  TTL-``archive`` sweep produces the same exclusion as the hygiene loop.

The graph-expansion coverage is the load-bearing one: applying the archived
clause only to the arm ``WHERE`` clauses is insufficient, because expansion
pulls brand-new source OBSERVATIONs that never passed an arm. A discriminating
revert proves the expansion gate specifically.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import CallbackProvider, SqliteEngravaCore
from engrava.domain.enums import (
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
)
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.thought import ThoughtRecord
from engrava.infrastructure.sqlite.engrava_core import _archived_exclusion_sql

if TYPE_CHECKING:
    from tests.search_contract.conftest import GoldQuestion


_EMBED_DIM = 64
_SCORE_ATOL = 1e-9


def _embed(text: str) -> list[float]:
    """Embed text as an L2-normalized bag-of-words hashing vector.

    Args:
        text: Input text to embed.

    Returns:
        An ``_EMBED_DIM``-length unit vector (all-zero only for empty text).
    """
    vector = [0.0] * _EMBED_DIM
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()  # noqa: S324
        vector[int.from_bytes(digest[:4], "big") % _EMBED_DIM] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _thought(
    thought_id: str,
    *,
    essence: str,
    content: str,
    thought_type: ThoughtType = ThoughtType.OBSERVATION,
    expires_at: str | None = None,
) -> ThoughtRecord:
    """Build an ACTIVE stored thought.

    Args:
        thought_id: Stable identifier used to assert retrieval.
        essence: Short FTS-indexed summary line.
        content: Full FTS-indexed body.
        thought_type: Classification (OBSERVATION by default; REFLECTION for a
            graph-expansion seed).
        expires_at: Optional ISO-8601 expiry (used only by the TTL-archive test).

    Returns:
        A fully populated :class:`ThoughtRecord` ready for ``create_thought``.
    """
    return ThoughtRecord(
        thought_id=thought_id,
        thought_type=thought_type,
        essence=essence,
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test",
        confidence=0.8,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        expires_at=expires_at,
    )


async def _open_store(
    *,
    auto_embed: bool = True,
    ttl_strategy: str = "archive",
) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
    """Open an in-memory store with a deterministic vector arm.

    Args:
        auto_embed: When ``True`` every stored thought is embedded on write.
        ttl_strategy: TTL cleanup strategy (``"archive"`` by default).

    Returns:
        The store together with its owning connection (caller closes it).
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    provider = CallbackProvider(
        callback=_embed,
        dimension=_EMBED_DIM,
        model_name="bag-of-words-archived",
    )
    store = SqliteEngravaCore(
        conn,
        embedding_provider=provider,
        auto_embed=auto_embed,
        ttl_strategy=ttl_strategy,
    )
    await store.ensure_schema()
    return store, conn


def _ids(results: list[tuple[str, float]]) -> set[str]:
    """Collect thought ids from scored results.

    Args:
        results: ``(thought_id, score)`` pairs.

    Returns:
        The set of returned thought ids.
    """
    return {thought_id for thought_id, _ in results}


class TestArchivedRestoreRoundTrip:
    """Archive → excluded from the default path → restore → retrieved again."""

    async def test_archive_then_restore_reenters_default_search(self) -> None:
        """A thought archived then restored is excluded, then eligible again."""
        store, conn = await _open_store()
        try:
            query = "alpha beta gamma distinctive"
            await store.create_thought(
                _thought("target", essence=query, content=query),
            )
            # Present in the default path before archiving.
            assert "target" in _ids((await store.search_hybrid(query, top_k=10)).results)

            await store.update_thought("target", lifecycle_status=LifecycleStatus.ARCHIVED)
            # Excluded from both the full hybrid and the ergonomic recall shorthand.
            assert "target" not in _ids((await store.search_hybrid(query, top_k=10)).results)
            assert "target" not in _ids((await store.recall(query, top_k=10)).results)

            restored = await store.restore_thought("target")
            assert restored.lifecycle_status is LifecycleStatus.ACTIVE
            # Eligible again on the default path.
            assert "target" in _ids((await store.search_hybrid(query, top_k=10)).results)
        finally:
            await conn.close()


class TestArchivedNoSilentDataLoss:
    """An archived thought is filtered, never lost: fetchable, opt-in reachable, restorable."""

    async def test_archived_thought_is_only_filtered_not_deleted(self) -> None:
        """Archived → absent from default search, yet fully recoverable three ways."""
        store, conn = await _open_store()
        try:
            query = "alpha beta gamma distinctive"
            await store.create_thought(_thought("target", essence=query, content=query))
            await store.update_thought("target", lifecycle_status=LifecycleStatus.ARCHIVED)

            # (1) Row still exists — no permanent delete.
            fetched = await store.get_thought("target")
            assert fetched is not None
            assert fetched.lifecycle_status is LifecycleStatus.ARCHIVED

            # (2) Absent from the default retrieval path...
            assert "target" not in _ids((await store.search_hybrid(query, top_k=10)).results)

            # (3) ...but reachable via the opt-in escape hatch...
            assert "target" in _ids(
                (await store.search_hybrid(query, top_k=10, include_archived=True)).results
            )

            # (4) ...and restorable back into the default path.
            await store.restore_thought("target")
            assert "target" in _ids((await store.search_hybrid(query, top_k=10)).results)
        finally:
            await conn.close()


class TestArchivedGraphExpansionExclusion:
    """Graph expansion must not re-inject an archived source OBSERVATION.

    Setup: an ACTIVE REFLECTION matches the query (so it becomes an expansion
    seed) and consolidates a single source OBSERVATION whose own text does *not*
    match the query — so the source can reach the result set only through
    ``CONSOLIDATED_FROM`` expansion, never through an arm. Archiving that source
    must keep it out of the default fused set, while ``include_archived=True``
    (or a reverted expansion gate) lets it back in.
    """

    _QUERY = "alpha beta gamma reflection"

    async def _build(self) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
        """Build the seed-REFLECTION + archived-source store.

        Returns:
            The store and its owning connection.
        """
        store, conn = await _open_store()
        # ACTIVE REFLECTION that matches the query — becomes the expansion seed.
        await store.create_thought(
            _thought(
                "refl",
                essence="alpha beta gamma reflection",
                content="alpha beta gamma reflection summary",
                thought_type=ThoughtType.REFLECTION,
            )
        )
        # Source OBSERVATION whose vocabulary is disjoint from the query, so it
        # can only enter the result set via graph expansion of the seed.
        await store.create_thought(
            _thought(
                "src-archived",
                essence="zeta eta theta unrelated",
                content="zeta eta theta unrelated vocabulary",
            )
        )
        await store.create_edge(
            EdgeRecord(
                edge_id=str(uuid.uuid4()),
                from_thought_id="refl",
                to_thought_id="src-archived",
                edge_type=EdgeType.CONSOLIDATED_FROM,
                weight=1.0,
                created_cycle=0,
                source=KnowledgeSource.DREAMING,
            )
        )
        await store.update_thought("src-archived", lifecycle_status=LifecycleStatus.ARCHIVED)
        return store, conn

    async def test_archived_source_does_not_leak_via_expansion(self) -> None:
        """The archived source is absent by default and present under include_archived."""
        store, conn = await self._build()
        try:
            # Sanity: the source does not match the query directly, so any
            # appearance is graph-expansion, not an arm.
            assert "src-archived" not in _ids(await store.search_fts(self._QUERY, top_k=20))

            default_ids = _ids((await store.search_hybrid(self._QUERY, top_k=20)).results)
            assert "refl" in default_ids, "the seed REFLECTION must surface (expansion runs)"
            assert "src-archived" not in default_ids

            opt_in = await store.search_hybrid(self._QUERY, top_k=20, include_archived=True)
            assert "src-archived" in _ids(opt_in.results), (
                "include_archived=True must let the archived source back in via expansion"
            )
            assert "graph_expansion" in opt_in.backends_used
        finally:
            await conn.close()

    async def test_expansion_unit_gate_honours_include_archived(self) -> None:
        """``_expand_via_consolidated_from`` gates the archived source on the flag."""
        store, conn = await self._build()
        try:
            # Default: the archived source is not propagated into combined.
            combined_default: dict[str, float] = {"refl": 0.8}
            await store._expand_via_consolidated_from(
                combined=combined_default,
                expansion_top_n=5,
                propagation_factor=0.7,
                max_sources_per_reflection=20,
                reflection_source_ceiling=50,
            )
            assert "src-archived" not in combined_default

            # Opt-in: the archived source is propagated.
            combined_opt_in: dict[str, float] = {"refl": 0.8}
            await store._expand_via_consolidated_from(
                combined=combined_opt_in,
                expansion_top_n=5,
                propagation_factor=0.7,
                max_sources_per_reflection=20,
                reflection_source_ceiling=50,
                include_archived=True,
            )
            assert "src-archived" in combined_opt_in
            assert combined_opt_in["src-archived"] > 0.0
        finally:
            await conn.close()

    async def test_reverting_expansion_gate_leaks_archived_via_expansion_only(self) -> None:
        """Forcing the expansion gate to skip its archived clause leaks the source.

        This is the discriminating revert for the graph-expansion path
        specifically: the arms still exclude the archived source (it never
        matched them anyway), so its appearance proves the leak arrives through
        expansion — exactly the path a filter applied only to the arm ``WHERE``
        clauses would have missed.
        """
        store, conn = await self._build()
        try:
            original = store._filter_observation_ids

            async def _leaky_filter(
                candidate_ids: list[str],
                *,
                include_archived: bool = False,
                _filter_clause: object = None,
            ) -> frozenset[str]:
                return await original(
                    candidate_ids,
                    include_archived=True,
                    _filter_clause=_filter_clause,  # type: ignore[arg-type]
                )

            # Baseline (unpatched): no leak.
            assert "src-archived" not in _ids(
                (await store.search_hybrid(self._QUERY, top_k=20)).results
            )

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(store, "_filter_observation_ids", _leaky_filter)
                leaked = _ids((await store.search_hybrid(self._QUERY, top_k=20)).results)

            assert "src-archived" in leaked, (
                "dropping the expansion archived clause must leak the archived source "
                "back into the fused set via graph expansion"
            )
        finally:
            await conn.close()


class TestExpiredSourceGraphExpansionExclusion:
    """Graph expansion must not re-inject an EXPIRED source OBSERVATION either.

    The expansion eligibility must match the arms in full: the arms drop expired
    rows (``expires_at IS NULL OR expires_at > now``), so the expansion gate must
    too. As with the archived case, the source's own vocabulary is disjoint from
    the query, so it can reach the fused set only through ``CONSOLIDATED_FROM``
    expansion — any appearance is a genuine expansion leak. Unlike the archived
    exclusion, expiry is *not* relaxed by ``include_archived``.
    """

    _QUERY = "alpha beta gamma reflection"
    _PAST = "2000-01-01T00:00:00+00:00"

    async def _build(self) -> tuple[SqliteEngravaCore, aiosqlite.Connection]:
        """Build a seed REFLECTION consolidating a single expired source OBSERVATION.

        Returns:
            The store and its owning connection.
        """
        store, conn = await _open_store()
        await store.create_thought(
            _thought(
                "refl",
                essence="alpha beta gamma reflection",
                content="alpha beta gamma reflection summary",
                thought_type=ThoughtType.REFLECTION,
            )
        )
        # ACTIVE but expired source, vocabulary disjoint from the query.
        await store.create_thought(
            _thought(
                "src-expired",
                essence="zeta eta theta unrelated",
                content="zeta eta theta unrelated vocabulary",
                expires_at=self._PAST,
            )
        )
        await store.create_edge(
            EdgeRecord(
                edge_id=str(uuid.uuid4()),
                from_thought_id="refl",
                to_thought_id="src-expired",
                edge_type=EdgeType.CONSOLIDATED_FROM,
                weight=1.0,
                created_cycle=0,
                source=KnowledgeSource.DREAMING,
            )
        )
        return store, conn

    async def test_expired_source_does_not_leak_via_expansion(self) -> None:
        """The expired source is excluded by default AND under include_archived=True."""
        store, conn = await self._build()
        try:
            # It never matches the query directly.
            assert "src-expired" not in _ids(await store.search_fts(self._QUERY, top_k=20))

            default_ids = _ids((await store.search_hybrid(self._QUERY, top_k=20)).results)
            assert "refl" in default_ids, "the seed REFLECTION must surface (expansion runs)"
            assert "src-expired" not in default_ids

            # include_archived re-admits *archived* rows, never *expired* ones —
            # expiry is an always-on gate, matching the arms.
            opt_in_ids = _ids(
                (await store.search_hybrid(self._QUERY, top_k=20, include_archived=True)).results
            )
            assert "src-expired" not in opt_in_ids
        finally:
            await conn.close()

    async def test_expansion_unit_gate_excludes_expired_regardless_of_flag(self) -> None:
        """``_expand_via_consolidated_from`` drops the expired source for both flag values."""
        store, conn = await self._build()
        try:
            for include_archived in (False, True):
                combined: dict[str, float] = {"refl": 0.8}
                await store._expand_via_consolidated_from(
                    combined=combined,
                    expansion_top_n=5,
                    propagation_factor=0.7,
                    max_sources_per_reflection=20,
                    reflection_source_ceiling=50,
                    include_archived=include_archived,
                )
                assert "src-expired" not in combined, (
                    f"expired source leaked with include_archived={include_archived}"
                )
        finally:
            await conn.close()

    async def test_reverting_expiry_gate_leaks_expired_via_expansion(self) -> None:
        """Dropping ONLY the expiry clause from the expansion gate leaks the expired source.

        The stub reproduces the pre-fix ``_filter_observation_ids`` — type +
        archived + metadata predicate, but no ``expires_at`` gate — proving the
        expiry clause specifically is what keeps an expired source out of the
        fused set on the expansion path.
        """
        store, conn = await self._build()
        try:

            async def _no_expiry_filter(
                candidate_ids: list[str],
                *,
                include_archived: bool = False,
                _filter_clause: tuple[str, list[object]] | None = None,
            ) -> frozenset[str]:
                unique = list(dict.fromkeys(candidate_ids))
                if not unique:
                    return frozenset()
                filter_sql = ""
                filter_params: list[object] = []
                if _filter_clause is not None:
                    fragment, filter_params = _filter_clause
                    filter_sql = f" AND {fragment}"
                archived_sql = _archived_exclusion_sql(
                    column="lifecycle_status",
                    include_archived=include_archived,
                )
                placeholders = ", ".join("?" for _ in unique)
                cursor = await store._db.execute(
                    f"SELECT thought_id FROM thought"  # noqa: S608 - expiry clause deliberately dropped
                    f" WHERE thought_type = 'OBSERVATION'"
                    f" AND thought_id IN ({placeholders})"
                    f"{archived_sql}"
                    f"{filter_sql}",
                    [*unique, *filter_params],
                )
                rows = await cursor.fetchall()
                return frozenset(str(r["thought_id"]) for r in rows)

            # Baseline (unpatched): no leak.
            assert "src-expired" not in _ids(
                (await store.search_hybrid(self._QUERY, top_k=20)).results
            )

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(store, "_filter_observation_ids", _no_expiry_filter)
                leaked = _ids((await store.search_hybrid(self._QUERY, top_k=20)).results)

            assert "src-expired" in leaked, (
                "dropping the expansion expiry clause must leak the expired source "
                "back into the fused set via graph expansion"
            )
        finally:
            await conn.close()


class TestArchivedExclusionDefaultOff:
    """With nothing archived, the exclusion is inert — default ranked results are unchanged."""

    async def test_toggle_is_byte_identical_on_all_active_corpus(
        self,
        hybrid_store: SqliteEngravaCore,
        gold_questions: tuple[GoldQuestion, ...],
    ) -> None:
        """On the all-active contract corpus, include_archived changes nothing.

        The order is identical and every paired score matches within ``atol`` —
        proving the archived clause adds no rows, drops no rows, and perturbs no
        score when the store holds no archived thoughts.
        """
        for question in gold_questions:
            default = (await hybrid_store.search_hybrid(question.question, top_k=10)).results
            opted = (
                await hybrid_store.search_hybrid(question.question, top_k=10, include_archived=True)
            ).results
            assert [tid for tid, _ in default] == [tid for tid, _ in opted], (
                f"archived toggle reordered results for {question.question!r}"
            )
            for (default_id, default_score), (opted_id, opted_score) in zip(
                default, opted, strict=True
            ):
                assert default_id == opted_id
                assert default_score == pytest.approx(opted_score, abs=_SCORE_ATOL)


class TestTtlArchivedRowExcluded:
    """A TTL-``archive`` sweep produces the same exclusion as the hygiene loop.

    Consistent semantics (the ADR C4 caveat, intended): a row archived by
    ``cleanup_expired`` under ``ttl_strategy="archive"`` — whose ``expires_at``
    is then cleared — is excluded from the default path by the archived clause,
    not the (now-absent) expiry clause.
    """

    async def test_ttl_archived_row_leaves_default_search(self) -> None:
        """A TTL-archived row is excluded by default and re-admitted on opt-in."""
        store, conn = await _open_store(ttl_strategy="archive")
        try:
            query = "alpha beta gamma distinctive"
            past = "2000-01-01T00:00:00+00:00"
            await store.create_thought(
                _thought("ttl-target", essence=query, content=query, expires_at=past),
            )

            cleanup = await store.cleanup_expired()
            assert cleanup.strategy_applied == "archive"
            archived = await store.get_thought("ttl-target")
            assert archived is not None
            assert archived.lifecycle_status is LifecycleStatus.ARCHIVED
            # The expiry gate no longer applies (expires_at was cleared on archive)...
            assert archived.expires_at is None

            # ...so the archived clause is the sole thing keeping it out by default.
            assert "ttl-target" not in _ids((await store.search_hybrid(query, top_k=10)).results)
            assert "ttl-target" in _ids(
                (await store.search_hybrid(query, top_k=10, include_archived=True)).results
            )
        finally:
            await conn.close()

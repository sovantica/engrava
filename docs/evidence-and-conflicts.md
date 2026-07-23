# Evidence, Claims, and Conflicts

Engrava persists thoughts and directional edges. It gives an application a
durable graph in which incompatible statements can coexist and remain
traceable, but it is not an entity resolver, claim extractor, contradiction
classifier, truth engine, or clarification workflow.

In particular, Engrava does **not** automatically:

- identify that two names refer to the same real-world entity;
- turn prose into typed subject/predicate/object claims;
- decide that two thoughts are semantically incompatible;
- choose which conflicting thought is true, current, or authoritative;
- create a `CONTESTED_BY` edge; or
- open a clarification task.

Those decisions belong to the caller. **The caller establishes that a conflict
exists and creates the `CONTESTED_BY` edge**; Engrava Free does not detect
contradictions automatically. Engrava stores the resulting thoughts, edges,
valid-time bounds, metadata, and provenance without silently resolving the
conflict.

## What happens to incompatible writes

`create_thought()` creates a new row by default. Two differently worded or
otherwise byte-distinct claims therefore coexist, even when they concern the
same entity and cannot both be true.

Content-hash deduplication is opt-in with `deduplicate=True`, and only
byte-identical `content` matches. A match returns the existing thought and
increments its `confirmation_count`; it does not compare meaning. Likewise,
`upsert_by_hash()` only updates a row with the same content hash. Engrava never
uses either operation to overwrite a semantic conflict.

A caller can still deliberately update, archive, invalidate, or delete a
thought. A high-ranked search result is also not a truth decision: ranking and
conflict resolution are separate concerns. See [Hybrid Search](search.md) and
[Data Lifecycle](data-lifecycle.md).

## A practical graph convention

Engrava does not prescribe an ontology. The following convention keeps entity,
claim, and evidence responsibilities explicit while using only the v0.6 public
API.

| Concept | Recommended representation |
|---|---|
| Entity | A stable caller-owned identifier in nested thought metadata, for example `metadata["entity"] = {"type": "customer", "id": "customer-42"}`. Engrava does not canonicalize it. |
| Claim | A thought whose metadata contains a stable entity id, predicate, and object. Keep the complete human-readable proposition in `content`. |
| Evidence | A separate thought containing or locating the observed source material. Put source-specific scalar fields in metadata. |
| Computational lineage | `ProvenanceContext`, especially `retrieval_context_ids`, when retrieved thoughts actually shaped a synthesized thought. This is caller-declared write-time context, not proof that the claim is true. |
| Semantic support | A caller-owned convention. For example, use a claim-to-evidence `DERIVED_FROM` edge only when that accurately describes the derivation, and document the convention in edge metadata. Engrava has no native `SUPPORTED_BY` inference rule. |
| Conflict | One directional `CONTESTED_BY` edge for a claim pair, with the detector, rule version, and workflow state in edge metadata. The caller detects the tension. |

Thought and edge metadata accepts scalar leaves and nested dictionaries, but
not lists. Use `ProvenanceContext.retrieval_context_ids` for a bounded list of
retrieved thought ids when its lineage meaning is accurate; do not use it as a
generic bag of citations.

### Time and reliability

Use `valid_from` and `valid_until` for the real-world interval during which a
claim applies. This prevents a legitimate change over time from being mistaken
for a simultaneous conflict. Open bounds are represented by `None`; Engrava
validates interval ordering but does not calculate whether two claims are
incompatible. See [The Bi-temporal Model](bitemporal.md).

Keep these reliability signals distinct:

- `confidence` is the caller's nullable `0.0-1.0` estimate for one thought;
- `confirmation_count` records independent re-encounters of byte-identical
  content when the caller opts into deduplication, or confirmations maintained
  by caller logic; and
- `EdgeRecord.weight` is the caller's strength for the relationship, not a
  probability that either endpoint is true.

None of these fields resolves a conflict automatically. A high-confidence
thought can be contested by another high-confidence thought.

## Direction and edge semantics

All `EdgeType` values are persisted labels. They carry no automatic graph
reasoning, symmetry, transitivity, conflict propagation, ranking effect, or
workflow behavior.

For `CONTESTED_BY`, choose and document one direction. This guide uses:

```text
existing claim --CONTESTED_BY--> challenging claim
```

The edge means only that the caller recorded the challenging claim as a
contest to the existing claim. Engrava neither evaluates the relationship nor
prefers the target. `get_edges(..., direction="BOTH")` can traverse the stored
edge from either endpoint; store a reciprocal edge only when your own graph
contract requires one.

The database permits only one edge for a given
`(from_thought_id, to_thought_id, edge_type)` triple. A second
`CONTESTED_BY` edge in the same direction between the same claims is therefore
not a second conflict event; update the existing edge metadata or create a new
claim node when your application needs a distinct tension. Deleting either
claim cascades to this edge row. That cascade does not produce a separate
`DELETE_EDGE` journal entry. When journaling is enabled and the endpoint is
deleted through a journaled store path, the journal records its
`DELETE_THOUGHT` mutation instead.

## Complete example

The following script uses an explicit single-value domain rule. It stores two
source observations, two incompatible claims, a `CONTESTED_BY` edge, and a
caller-created clarification task. No LLM is involved.

```python
import asyncio
import uuid

import aiosqlite

from engrava import (
    EdgeRecord,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ProvenanceContext,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)


def new_id() -> str:
    return str(uuid.uuid4())


def incompatible_single_value_claims(
    left: ThoughtRecord,
    right: ThoughtRecord,
) -> bool:
    """Caller-owned rule: one entity/predicate cannot have two values."""
    left_entity = left.metadata.get("entity")
    right_entity = right.metadata.get("entity")
    left_claim = left.metadata.get("claim")
    right_claim = right.metadata.get("claim")

    if not all(
        isinstance(value, dict)
        for value in (left_entity, right_entity, left_claim, right_claim)
    ):
        return False

    return (
        left_entity.get("type") == right_entity.get("type")
        and left_entity.get("id") == right_entity.get("id")
        and left_claim.get("predicate") == right_claim.get("predicate")
        and left_claim.get("object") != right_claim.get("object")
    )


async def main() -> None:
    async with aiosqlite.connect(":memory:") as connection:
        connection.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(connection, journal_enabled=True)
        await store.ensure_schema()

        crm_evidence = await store.create_thought(
            ThoughtRecord(
                thought_id=new_id(),
                thought_type=ThoughtType.OBSERVATION,
                essence="CRM lists billing country as Poland",
                content="CRM export row 831: billing_country=PL",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=41,
                updated_cycle=41,
                source="crm-import",
                source_type=KnowledgeSource.EXPERIENCE,
                confidence=0.95,
                metadata={
                    "record_kind": "evidence",
                    "entity": {"type": "customer", "id": "customer-42"},
                    "source_ref": {"system": "crm", "record": "831"},
                },
            )
        )

        invoice_evidence = await store.create_thought(
            ThoughtRecord(
                thought_id=new_id(),
                thought_type=ThoughtType.OBSERVATION,
                essence="Invoice lists billing country as Germany",
                content="Invoice 2026-104: billing_country=DE",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=42,
                updated_cycle=42,
                source="invoice-import",
                source_type=KnowledgeSource.EXPERIENCE,
                confidence=0.98,
                metadata={
                    "record_kind": "evidence",
                    "entity": {"type": "customer", "id": "customer-42"},
                    "source_ref": {"system": "billing", "record": "2026-104"},
                },
            )
        )

        claim_pl = await store.create_thought(
            ThoughtRecord(
                thought_id=new_id(),
                thought_type=ThoughtType.BELIEF,
                essence="Customer 42 billing country is Poland",
                content="The billing country for customer 42 is Poland.",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=41,
                updated_cycle=41,
                source="profile-reconciler",
                source_type=KnowledgeSource.EXPERIENCE,
                confidence=0.95,
                valid_from="2026-01-01T00:00:00+00:00",
                metadata={
                    "record_kind": "claim",
                    "entity": {"type": "customer", "id": "customer-42"},
                    "claim": {"predicate": "billing_country", "object": "PL"},
                },
                provenance=ProvenanceContext(
                    session_id="reconcile-2026-07-23",
                    actor_id="profile-reconciler",
                    instruction_context="Map the CRM billing country to the profile.",
                    retrieval_context_ids=[crm_evidence.thought_id],
                ),
            )
        )

        claim_de = await store.create_thought(
            ThoughtRecord(
                thought_id=new_id(),
                thought_type=ThoughtType.BELIEF,
                essence="Customer 42 billing country is Germany",
                content="The billing country for customer 42 is Germany.",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=42,
                updated_cycle=42,
                source="profile-reconciler",
                source_type=KnowledgeSource.EXPERIENCE,
                confidence=0.98,
                valid_from="2026-01-01T00:00:00+00:00",
                metadata={
                    "record_kind": "claim",
                    "entity": {"type": "customer", "id": "customer-42"},
                    "claim": {"predicate": "billing_country", "object": "DE"},
                },
                provenance=ProvenanceContext(
                    session_id="reconcile-2026-07-23",
                    actor_id="profile-reconciler",
                    instruction_context="Map the invoice billing country to the profile.",
                    retrieval_context_ids=[invoice_evidence.thought_id],
                ),
            )
        )

        if not incompatible_single_value_claims(claim_pl, claim_de):
            raise RuntimeError("The caller's domain rule found no conflict")

        conflict = await store.create_edge(
            EdgeRecord(
                edge_id=new_id(),
                from_thought_id=claim_pl.thought_id,
                to_thought_id=claim_de.thought_id,
                edge_type=EdgeType.CONTESTED_BY,
                weight=1.0,
                created_cycle=42,
                source=KnowledgeSource.EXPERIENCE,
                metadata={
                    "conflict": {
                        "detector": "single-value-slot",
                        "rule_version": "1",
                        "status": "open",
                    }
                },
            )
        )

        # Read the stored edge and the write-time lineage on both endpoints.
        for edge in await store.get_edges(claim_pl.thought_id, direction="OUT"):
            if edge.edge_type is not EdgeType.CONTESTED_BY:
                continue
            existing = await store.get_thought(edge.from_thought_id)
            challenger = await store.get_thought(edge.to_thought_id)
            if existing is None or challenger is None:
                continue
            print(existing.content)
            print(challenger.content)
            print(existing.provenance)
            print(challenger.provenance)

        # This task is proposed and worded by the caller. Engrava only stores it.
        clarification = await store.create_thought(
            ThoughtRecord(
                thought_id=new_id(),
                thought_type=ThoughtType.TASK,
                essence="Clarify customer 42 billing country",
                content=(
                    "Clarify whether customer 42's billing country is Poland or "
                    "Germany, and record the effective date of the confirmed value."
                ),
                priority=Priority.P1,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=43,
                updated_cycle=43,
                source="conflict-workflow",
                source_type=KnowledgeSource.EXPERIENCE,
                metadata={
                    "record_kind": "clarification_task",
                    "conflict": {"edge_id": conflict.edge_id, "status": "open"},
                },
                provenance=ProvenanceContext(
                    session_id="reconcile-2026-07-23",
                    actor_id="conflict-workflow",
                    instruction_context="Open a task for an unresolved single-value slot.",
                    retrieval_context_ids=[claim_pl.thought_id, claim_de.thought_id],
                ),
            )
        )
        print(clarification.thought_id)


asyncio.run(main())
```

The two claims are deliberately inserted without semantic deduplication. The
rule, edge direction, edge metadata, and task wording are application policy,
not hidden engine behavior. In a production detector, also test whether the
claims' valid-time intervals overlap before declaring a conflict.

## Detection without and with an LLM

An LLM is not required when the caller already has comparable, typed values.
Examples include:

- one customer and one single-valued `billing_country` slot with two different
  normalized country codes;
- overlapping employment intervals that violate a domain constraint;
- incompatible version, status, or ownership values from structured systems;
  and
- a source-specific precedence rule that always requires human confirmation
  when two authorities disagree.

These checks are ordinary deterministic application code. Record the detector
name and version on the conflict edge so the decision can be reproduced.

An LLM can be useful for open-ended prose where contradiction depends on
paraphrase, implication, negation, or missing context. That is an optional
caller-side classifier, not an Engrava dependency. Treat its output as a
proposal: retain both claims and their evidence, record the model/rule context,
and let policy or a human resolve consequential cases. Embedding similarity
alone indicates relatedness, not contradiction.

## Three kinds of provenance

The word *provenance* can refer to different guarantees. Keep them separate:

1. **Write-time computational context.** `ThoughtRecord.provenance` stores the
   caller-supplied session, actor, retrieval query, instruction context, and
   retrieved thought ids that shaped a write. It is an untrusted query hint,
   not authentication, authorization, or a truth signal. See
   [API Reference: provenance capture](api-reference.md#provenance-capture).
2. **Semantic evidence provenance.** Evidence thoughts, source references in
   metadata, and caller-defined lineage edges explain why a domain claim was
   made. Their meaning is established by the application's schema and trust
   policy; Engrava does not verify the external source.
3. **Mutation provenance.** With journaling enabled, the hash-chain journal
   records thought and edge inserts, updates, and deletes as before/after
   deltas. It can show that a conflict edge was added and detect inconsistent
   retained journal rows within its documented threat model. It does not
   reconcile live claim/edge rows against the journal, prove that either claim
   is true, authenticate an external document, or replace semantic evidence.
   See [Audit Trail](audit-trail.md).

The journal is tamper-evident, not tamper-proof: it is a keyless chain stored in
the same database. A `ProvenanceContext` included in a thought is itself
journaled as part of that thought when journaling is enabled, but the chain only
protects the recorded mutation history; it does not promote caller-supplied
provenance into verified identity or evidence.

## Operational recommendations

- Define stable entity ids, claim predicates, normalization rules, and edge
  direction before ingesting production data.
- Preserve conflicting claims until a deliberate policy closes, invalidates,
  archives, or supersedes one of them.
- Use valid time to distinguish simultaneous incompatibility from legitimate
  historical change.
- Record detector/rule versions and resolution state on edge metadata.
- Keep conflict detection separate from ranking. Retrieve all relevant claims
  for the entity/slot before applying resolution policy.
- Treat metadata and provenance filters as query refinements, not access-control
  boundaries. See [Known Limitations](known-limitations.md).

For exact record fields and method signatures, see the [API Reference](api-reference.md).

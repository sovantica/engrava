-- engrava: Core thought-graph schema (free-tier boundary — no internal-cognitive columns).
-- Version: core-18 (Memory Hygiene forgetting-loop columns thought.pinned +
--          thought.archived_at_cycle; both nullable/defaulted so a store that
--          never enables hygiene reads back unchanged — pinned is the durable
--          never-forget marker, archived_at_cycle records the cycle hygiene
--          archived a thought and backs the GC restore window;
--          core-17 added the opt-in thought.provenance capture column + the two
--          JSON expression indexes idx_thought_prov_session /
--          idx_thought_prov_actor on its identity fields; provenance is captured
--          and queryable only — it feeds no ranking / dreaming / edge path;
--          core-16 added the denormalised thought.action_outcome_score aggregate
--          + the idx_action_source_thought seek index that backs its recompute;
--          core-15 added the composite inbound edge index
--          edge(edge_type, to_thought_id) for edge-type-scoped inbound lookups;
--          core-14 added the hot-path indexes edge.to_thought_id,
--          embedding.owner_id, thought.updated_cycle, thought.thought_type)

PRAGMA user_version = 18;

CREATE TABLE IF NOT EXISTS thought (
    thought_id        TEXT    PRIMARY KEY,
    thought_type      TEXT    NOT NULL,
    essence           TEXT    NOT NULL,
    content           TEXT    NOT NULL,
    content_hash      TEXT,
    priority          TEXT    NOT NULL,
    lifecycle_status  TEXT    NOT NULL DEFAULT 'CREATED',
    created_cycle     INTEGER NOT NULL DEFAULT 0,
    updated_cycle     INTEGER NOT NULL DEFAULT 0,
    source            TEXT    NOT NULL DEFAULT 'human',
    confidence        REAL,
    embedding_ref     TEXT,
    source_type       TEXT    NOT NULL DEFAULT 'EXPERIENCE',
    confirmation_count INTEGER NOT NULL DEFAULT 0,
    consolidated_from TEXT,
    visibility        TEXT    NOT NULL DEFAULT 'selective',
    access_count      INTEGER NOT NULL DEFAULT 0,
    last_accessed_at  TEXT,
    created_at        TEXT,
    updated_at        TEXT,
    expires_at        TEXT,
    metadata_json     TEXT    NOT NULL DEFAULT '{}',
    -- Valid-time (world-time) axis. Declared last so a freshly created
    -- database matches the column order of one upgraded in place, where
    -- ``ALTER TABLE ADD COLUMN`` can only append.
    valid_from        TEXT,
    valid_until       TEXT,
    -- Denormalised action-outcome aggregate (core-16). Mean outcome value over
    -- the thought's terminal linked actions, or NULL when it has none. Appended
    -- last for the same column-order parity as the valid-time columns above.
    action_outcome_score REAL,
    -- Opt-in write-time provenance capture (core-17). A JSON document holding
    -- the ProvenanceContext sub-model (session_id / actor_id identity +
    -- retrieval_query / instruction_context / retrieval_context_ids synthesis
    -- context), or NULL when a thought carries no provenance — a NULL column is
    -- byte-identical to a pre-core-17 row. Appended last for the same
    -- column-order parity as the columns above. Provenance is an untrusted hint
    -- granted zero authority: it is captured and made queryable only and feeds
    -- no ranking / dreaming / edge-creation path.
    provenance        TEXT,
    -- Memory Hygiene forgetting-loop columns (core-18). Both nullable/defaulted
    -- and appended last for the same column-order parity as the columns above,
    -- so a database upgraded in place (ALTER ... ADD COLUMN can only append)
    -- matches a freshly created one. ``pinned`` is the durable never-forget
    -- marker: a pinned thought is never auto-archived or auto-GC'd by the
    -- hygiene loop (default 0 = not pinned). ``archived_at_cycle`` records the
    -- cycle at which the hygiene loop archived a thought (NULL when it was not
    -- archived by hygiene — a restore clears it back to NULL); it backs the
    -- GC restore window, so a thought archived by any other path (TTL / manual)
    -- keeps NULL and is never reaped by hygiene GC. A store that never enables
    -- hygiene leaves both at their defaults and reads back unchanged.
    pinned            INTEGER NOT NULL DEFAULT 0,
    archived_at_cycle INTEGER
);

CREATE TABLE IF NOT EXISTS edge (
    edge_id           TEXT PRIMARY KEY,
    from_thought_id   TEXT NOT NULL,
    to_thought_id     TEXT NOT NULL,
    edge_type         TEXT NOT NULL,
    weight            REAL NOT NULL DEFAULT 0.5,
    created_cycle     INTEGER NOT NULL DEFAULT 0,
    source            TEXT NOT NULL DEFAULT 'EXPERIENCE',
    decay_multiplier  REAL NOT NULL DEFAULT 1.0,
    valid_from        TEXT,
    valid_until       TEXT,
    UNIQUE(from_thought_id, to_thought_id, edge_type),
    FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE,
    FOREIGN KEY (to_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE
);

-- ``embedding.owner_id`` references ``thought(thought_id)`` because every
-- embedding ingested by this codebase is keyed to a thought
-- (``owner_type='THOUGHT'`` on every insert). The polymorphic-looking
-- ``owner_type`` column is reserved for forward-compatibility but is not
-- currently exercised by any non-thought owner; the FK + CASCADE assumption
-- here is safe today and any future non-thought owner would have to revisit
-- this constraint.
CREATE TABLE IF NOT EXISTS embedding (
    embedding_id TEXT PRIMARY KEY,
    owner_type   TEXT    NOT NULL,
    owner_id     TEXT    NOT NULL,
    model_name   TEXT    NOT NULL,
    dimension    INTEGER NOT NULL,
    vector_blob  BLOB    NOT NULL,
    created_at   TEXT    NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES thought(thought_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS action (
    action_id           TEXT PRIMARY KEY,
    source_thought_id   TEXT NOT NULL,
    action_type         TEXT NOT NULL,
    intent              TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'PLANNED',
    verification_status TEXT NOT NULL DEFAULT 'PENDING',
    raw_metrics_json    TEXT,
    FOREIGN KEY (source_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE
);

-- -------------------------------------------------------------------
-- Generic key/value metadata
-- -------------------------------------------------------------------
-- Used for embedding model lock (embedding_model_name, embedding_dimension)
-- and future migration metadata (schema_version, created_at, etc.).

CREATE TABLE IF NOT EXISTS _metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- -------------------------------------------------------------------
-- FTS5 full-text search index for topic-based retrieval
-- -------------------------------------------------------------------
-- content= syncs with the thought table via triggers.
-- content_rowid='rowid' uses the implicit rowid (thought is NOT
-- WITHOUT ROWID, so the implicit rowid column exists).
-- Only essence and content are indexed — sufficient for keyword recall.

CREATE VIRTUAL TABLE IF NOT EXISTS thought_fts USING fts5(
    essence,
    content,
    tokenize = "unicode61 tokenchars '-_'",
    content='thought',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS thought_fts_insert
AFTER INSERT ON thought BEGIN
    INSERT INTO thought_fts(rowid, essence, content)
    VALUES (new.rowid, new.essence, new.content);
END;

CREATE TRIGGER IF NOT EXISTS thought_fts_delete
AFTER DELETE ON thought BEGIN
    INSERT INTO thought_fts(thought_fts, rowid, essence, content)
    VALUES ('delete', old.rowid, old.essence, old.content);
END;

CREATE TRIGGER IF NOT EXISTS thought_fts_update
AFTER UPDATE OF essence, content ON thought BEGIN
    INSERT INTO thought_fts(thought_fts, rowid, essence, content)
    VALUES ('delete', old.rowid, old.essence, old.content);
    INSERT INTO thought_fts(rowid, essence, content)
    VALUES (new.rowid, new.essence, new.content);
END;

-- -------------------------------------------------------------------
-- CognitiveJournal: hash-linked mutation log
-- -------------------------------------------------------------------
-- Append-only audit trail for thought-graph mutations.  Each entry is
-- SHA-256 linked to the previous entry (parent_hash) for tamper evidence.

CREATE TABLE IF NOT EXISTS journal_entry (
    entry_id         TEXT    PRIMARY KEY,
    sequence_number  INTEGER NOT NULL UNIQUE,
    mutation_type    TEXT    NOT NULL,
    target_id        TEXT,
    delta            TEXT    NOT NULL,
    parent_hash      TEXT,
    entry_hash       TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_target ON journal_entry(target_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entry(mutation_type);
CREATE INDEX IF NOT EXISTS idx_journal_seq ON journal_entry(sequence_number);

-- -------------------------------------------------------------------
-- Partial index for TTL / auto-expiry
-- -------------------------------------------------------------------
-- Only thoughts with a non-NULL expires_at are indexed.  Thoughts that
-- never expire (the vast majority) incur zero index overhead.

CREATE INDEX IF NOT EXISTS idx_thought_expires ON thought(expires_at)
    WHERE expires_at IS NOT NULL;

-- -------------------------------------------------------------------
-- Content-hash index for ingest deduplication (core-10)
-- -------------------------------------------------------------------
-- Enables O(log N) lookup of existing thoughts by SHA-256 content hash
-- when callers opt in to deduplication via
-- ``SqliteEngravaCore.create_thought(..., deduplicate=True)``.
-- Nullable column: pre-core-10 thoughts may have ``content_hash IS NULL``
-- until the explicit backfill utility script populates them.

CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash);

-- -------------------------------------------------------------------
-- Composite edge index for candidate expansion queries
-- -------------------------------------------------------------------
-- Supports _expand_via_consolidated_from and the giant-cluster guard:
--   SELECT ... FROM edge
--   WHERE edge_type = 'CONSOLIDATED_FROM' AND from_thought_id IN (...)
-- Also benefits _load_graph_signal and any future edge_type lookups.

CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id);

-- -------------------------------------------------------------------
-- Per-extension schema version tracking
-- -------------------------------------------------------------------
-- Tracks which SQL migration files have been applied for each extension.
-- extension_name matches ExtensionManifest.name.
-- version is the count of applied migration files (1-indexed).

CREATE TABLE IF NOT EXISTS extension_schema_versions (
    extension_name    TEXT PRIMARY KEY,
    version           INTEGER NOT NULL DEFAULT 0,
    applied_at        REAL NOT NULL,
    migration_file    TEXT,
    extension_version TEXT
);

-- -------------------------------------------------------------------
-- Valid-time (world-time) indexes for thought and edge
-- -------------------------------------------------------------------
-- The valid_from / valid_until columns form a second time axis ("valid
-- time" — when a fact is true in the world) alongside the transaction
-- time recorded by created_at. These indexes back range scans over that
-- axis. valid_until is partial (only non-NULL upper bounds are indexed)
-- because an open upper bound is the common case and incurs no overhead.
-- A fresh-bootstrap database must carry the same indexes as one upgraded
-- in place, so they are declared here as well as in the migration helper.

CREATE INDEX IF NOT EXISTS idx_thought_valid_from ON thought(valid_from);
CREATE INDEX IF NOT EXISTS idx_thought_valid_until ON thought(valid_until)
    WHERE valid_until IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_thought_valid_range ON thought(valid_from, valid_until);

CREATE INDEX IF NOT EXISTS idx_edge_valid_from ON edge(valid_from);
CREATE INDEX IF NOT EXISTS idx_edge_valid_until ON edge(valid_until)
    WHERE valid_until IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_edge_valid_range ON edge(valid_from, valid_until);

-- -------------------------------------------------------------------
-- Hot-path indexes for the core read queries (core-14)
-- -------------------------------------------------------------------
-- These back the equality and sort columns hit on every common read:
--   * idx_edge_to_thought   — get_edges (IN / BOTH direction) and the
--     reflection-consolidation scan filter edge on to_thought_id.
--   * idx_embedding_owner   — get_embedding looks up an embedding by its
--     owner thought; without an index this is a full table scan, and it
--     runs inside three dreaming loops.
--   * idx_thought_updated_cycle — list_thoughts orders by updated_cycle on
--     every call.
--   * idx_thought_type      — thought_type equality is used by the
--     reflection-id scan on every search and by list_thoughts filtering.
-- A fresh-bootstrap database must carry the same indexes as one upgraded
-- in place, so they are declared here as well as in the migration helper.

CREATE INDEX IF NOT EXISTS idx_edge_to_thought ON edge(to_thought_id);
CREATE INDEX IF NOT EXISTS idx_embedding_owner ON embedding(owner_id);
CREATE INDEX IF NOT EXISTS idx_thought_updated_cycle ON thought(updated_cycle);
CREATE INDEX IF NOT EXISTS idx_thought_type ON thought(thought_type);

-- -------------------------------------------------------------------
-- Composite inbound edge index for edge-type-scoped lookups (core-15)
-- -------------------------------------------------------------------
-- Mirrors idx_edge_type_from on the destination side. Inbound scans that
-- filter on both edge_type and to_thought_id (the CONSOLIDATED_FROM
-- source-resolution query) otherwise seek to_thought_id via
-- idx_edge_to_thought and test edge_type as a per-row residual; this
-- composite lets one seek satisfy both predicates:
--   SELECT ... FROM edge WHERE to_thought_id = ? AND edge_type = ?
-- EXPLAIN QUERY PLAN then reports
--   idx_edge_type_to (edge_type=? AND to_thought_id=?).

CREATE INDEX IF NOT EXISTS idx_edge_type_to ON edge(edge_type, to_thought_id);

-- -------------------------------------------------------------------
-- Action-by-source index for the outcome-score recompute (core-16)
-- -------------------------------------------------------------------
-- The action-outcome recompute resolves a thought's linked actions with
--   SELECT ... FROM action WHERE source_thought_id = ?
-- which, without an index, is a full scan of the action table. This index
-- turns that lookup into a seek so the recompute stays cheap even as the
-- action table grows. EXPLAIN QUERY PLAN then reports
--   SEARCH action USING INDEX idx_action_source_thought (source_thought_id=?).
-- A fresh-bootstrap database must carry the same index as one upgraded in
-- place, so it is declared here as well as in the migration helper.

CREATE INDEX IF NOT EXISTS idx_action_source_thought ON action(source_thought_id);

-- -------------------------------------------------------------------
-- Provenance identity indexes (core-17)
-- -------------------------------------------------------------------
-- JSON expression indexes on the two first-class identity fields of the opt-in
-- provenance sub-model. They make session / actor lookup a seek rather than a
-- full scan:
--   SELECT ... FROM thought WHERE json_extract(provenance,'$.session_id') = ?
-- EXPLAIN QUERY PLAN then reports
--   SEARCH thought USING INDEX idx_thought_prov_session (<expr>=?).
-- The descriptive provenance fields (retrieval_query / instruction_context /
-- retrieval_context_ids) are queryable through the same json_extract filter
-- machinery but are deliberately not indexed. A fresh-bootstrap database must
-- carry the same indexes as one upgraded in place, so they are declared here as
-- well as in the migration helper.

CREATE INDEX IF NOT EXISTS idx_thought_prov_session
    ON thought(json_extract(provenance, '$.session_id'));
CREATE INDEX IF NOT EXISTS idx_thought_prov_actor
    ON thought(json_extract(provenance, '$.actor_id'));

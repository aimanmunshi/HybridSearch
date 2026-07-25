-- Schema for the semantic search engine.
--
-- {embedding_dim} is substituted at migration time from the active embedding
-- provider (384 for all-MiniLM-L6-v2, 1536 for text-embedding-3-small).

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per recipe: the unit the user actually searches for, and the unit
-- results are returned in.
CREATE TABLE IF NOT EXISTS recipes (
    id            TEXT PRIMARY KEY,
    title         TEXT   NOT NULL,
    category      TEXT   NOT NULL DEFAULT '',
    cuisine       TEXT   NOT NULL DEFAULT '',
    tags          TEXT[] NOT NULL DEFAULT '{}',
    -- Space-joined copy of `tags`, maintained by the application on write.
    -- array_to_string() is STABLE rather than IMMUTABLE in Postgres, so it
    -- cannot appear inside a generated column below; this column exists
    -- purely to work around that restriction.
    tags_text     TEXT   NOT NULL DEFAULT '',
    ingredients   JSONB  NOT NULL DEFAULT '[]',
    instructions  TEXT   NOT NULL DEFAULT '',
    thumbnail_url TEXT   NOT NULL DEFAULT '',
    source_url    TEXT   NOT NULL DEFAULT '',

    -- Denormalised full text, kept so the tsvector below can be a generated
    -- column and so snippets can be built without re-joining chunks.
    full_text     TEXT   NOT NULL DEFAULT '',

    -- Weighted lexical index. Postgres ranks A > B > C > D, so a query word
    -- matching the title outranks the same word buried in the method. This is
    -- what makes ts_rank a useful signal rather than a flat term count.
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english',
            coalesce(cuisine, '') || ' ' ||
            coalesce(category, '') || ' ' ||
            coalesce(tags_text, '')
        ), 'B') ||
        setweight(to_tsvector('english', coalesce(full_text, '')), 'C')
    ) STORED
);

-- One row per embeddable slice of a recipe. Retrieval happens here; results are
-- aggregated back up to recipes by taking each recipe's best-scoring chunk.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    recipe_id   TEXT NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({embedding_dim}) NOT NULL,

    -- Makes re-ingestion idempotent: ON CONFLICT can update a chunk in place
    -- rather than accumulating duplicates.
    UNIQUE (recipe_id, chunk_index)
);

-- HNSW over cosine distance.
--
-- HNSW rather than IVFFlat: IVFFlat must be built *after* the data is loaded
-- (it trains centroids on existing rows) and degrades if the corpus later
-- shifts, whereas HNSW can be created on an empty table and stays correct as
-- rows arrive -- which keeps the re-index script simple. HNSW also gives
-- better recall at equal latency; its cost is a slower build and more memory,
-- neither of which matters at this corpus size.
--
-- Honest caveat: with ~1.7k vectors Postgres will often prefer a sequential
-- scan anyway, and exact search is already sub-millisecond. The index is here
-- because it is what makes the design scale, not because it wins today.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN is the standard index for tsvector: it stores one posting list per
-- lexeme, which is what full-text lookup needs.
CREATE INDEX IF NOT EXISTS recipes_search_vector_gin
    ON recipes USING gin (search_vector);

CREATE INDEX IF NOT EXISTS chunks_recipe_id ON chunks (recipe_id);

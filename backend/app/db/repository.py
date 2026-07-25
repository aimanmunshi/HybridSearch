"""Write and read the corpus.

Writes are idempotent: every insert is an UPSERT keyed on the recipe's upstream
ID, so re-running the indexer over a changed corpus converges to the right
state instead of duplicating rows. That is what makes `index_corpus.py` safe to
re-run at any time.
"""
from __future__ import annotations

import json
import logging

from psycopg.types.json import Jsonb

from app.db.connection import get_pool
from app.models.recipe import Chunk, Recipe

logger = logging.getLogger(__name__)

UPSERT_RECIPE = """
    INSERT INTO recipes (
        id, title, category, cuisine, tags, tags_text, ingredients,
        instructions, thumbnail_url, source_url, full_text
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        title         = EXCLUDED.title,
        category      = EXCLUDED.category,
        cuisine       = EXCLUDED.cuisine,
        tags          = EXCLUDED.tags,
        tags_text     = EXCLUDED.tags_text,
        ingredients   = EXCLUDED.ingredients,
        instructions  = EXCLUDED.instructions,
        thumbnail_url = EXCLUDED.thumbnail_url,
        source_url    = EXCLUDED.source_url,
        full_text     = EXCLUDED.full_text
"""

UPSERT_CHUNK = """
    INSERT INTO chunks (recipe_id, chunk_index, text, embedding)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (recipe_id, chunk_index) DO UPDATE SET
        text      = EXCLUDED.text,
        embedding = EXCLUDED.embedding
"""


def upsert_recipes(recipes: list[Recipe], batch_size: int = 500) -> int:
    """Insert or update recipes. Returns the number written."""
    if not recipes:
        return 0

    rows = [
        (
            r.id,
            r.title,
            r.category,
            r.cuisine,
            r.tags,
            " ".join(r.tags),
            Jsonb([i.model_dump() for i in r.ingredients]),
            r.instructions,
            r.thumbnail_url,
            r.source_url,
            r.full_text(),
        )
        for r in recipes
    ]

    with get_pool().connection() as conn, conn.cursor() as cur:
        for start in range(0, len(rows), batch_size):
            cur.executemany(UPSERT_RECIPE, rows[start : start + batch_size])

    logger.info("upserted %d recipes", len(rows))
    return len(rows)


def upsert_chunks(
    chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500
) -> int:
    """Insert or update chunks with their embeddings. Returns rows written."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"got {len(chunks)} chunks but {len(embeddings)} embeddings"
        )
    if not chunks:
        return 0

    rows = [
        (chunk.recipe_id, chunk.chunk_index, chunk.text, embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]

    with get_pool().connection() as conn, conn.cursor() as cur:
        for start in range(0, len(rows), batch_size):
            cur.executemany(UPSERT_CHUNK, rows[start : start + batch_size])

    logger.info("upserted %d chunks", len(rows))
    return len(rows)


def delete_orphaned_chunks(chunks: list[Chunk]) -> int:
    """Remove chunks that no longer exist in the freshly built corpus.

    Needed because a recipe that gets *shorter* between runs would otherwise
    keep its now-stale trailing chunks: the UPSERT rewrites indices 0..n but
    never touches n+1 and beyond.
    """
    if not chunks:
        return 0

    highest: dict[str, int] = {}
    for chunk in chunks:
        highest[chunk.recipe_id] = max(
            highest.get(chunk.recipe_id, -1), chunk.chunk_index
        )

    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "DELETE FROM chunks WHERE recipe_id = %s AND chunk_index > %s",
            list(highest.items()),
        )
        removed = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    if removed:
        logger.info("removed %d stale chunks", removed)
    return removed


def corpus_stats() -> dict[str, int]:
    """Return row counts, for the health endpoint and the indexer summary."""
    with get_pool().connection() as conn:
        recipes = conn.execute("SELECT count(*) FROM recipes").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"recipes": recipes, "chunks": chunks}

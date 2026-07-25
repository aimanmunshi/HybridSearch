"""Pure vector (cosine similarity) search over recipe chunks.

Retrieval happens at chunk level -- that is the unit the embedding index is
built over -- but results are returned at recipe level, since that is the unit
the user is searching for. The strategy is "best chunk wins": a recipe's score
is its single closest-matching chunk, not an average, so a recipe with one
perfect-match paragraph outranks one with several mediocre ones.

To get top-K distinct *recipes* from a chunk-level index, we over-fetch
chunk-level candidates (more than K) and then deduplicate by recipe in Python,
keeping the best-scoring chunk per recipe. `CANDIDATE_MULTIPLIER` is generous
relative to the corpus's ~2.16 chunks/recipe average, so exhausting the
candidate pool before finding K distinct recipes is not expected in practice.
"""
from __future__ import annotations

import time

from app.db.connection import get_pool
from app.ingestion.embed import get_embedding_provider
from app.models.search import SearchResult

CANDIDATE_MULTIPLIER = 8
MIN_CANDIDATES = 50
SNIPPET_MAX_CHARS = 240

CHUNK_CANDIDATES_SQL = """
    SELECT
        r.id, r.title, r.cuisine, r.category, r.tags, r.thumbnail_url,
        c.text,
        c.embedding <=> %(query_vector)s::vector AS distance
    FROM chunks c
    JOIN recipes r ON r.id = c.recipe_id
    ORDER BY distance ASC
    LIMIT %(candidate_pool)s
"""


def _snippet_from_chunk(chunk_text: str) -> str:
    """Strip the recipe header off a chunk and truncate for display.

    Every chunk is stored as "{header}\\n{body}" (see `app.ingestion.chunk`);
    the header is redundant once the title is already shown alongside it.
    """
    body = chunk_text.split("\n", 1)[-1].strip()
    if len(body) <= SNIPPET_MAX_CHARS:
        return body
    # Cut at the last word boundary before the limit rather than mid-word.
    truncated = body[:SNIPPET_MAX_CHARS].rsplit(" ", 1)[0]
    return f"{truncated}..."


def semantic_search(query: str, top_k: int = 10) -> tuple[list[SearchResult], float]:
    """Embed `query` and return the top-K distinct recipes by cosine similarity.

    Returns a (results, took_ms) tuple so callers can report latency without a
    second timer.
    """
    started = time.perf_counter()

    provider = get_embedding_provider()
    query_vector = provider.embed_query(query)

    candidate_pool = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    with get_pool().connection() as conn:
        rows = conn.execute(
            CHUNK_CANDIDATES_SQL,
            {"query_vector": query_vector, "candidate_pool": candidate_pool},
        ).fetchall()

    best_per_recipe: dict[str, SearchResult] = {}
    for recipe_id, title, cuisine, category, tags, thumbnail_url, chunk_text, distance in rows:
        if recipe_id in best_per_recipe:
            continue  # rows are distance-ordered, so the first hit is the best
        # pgvector's <=> is cosine *distance* in [0, 2]; 1 - distance is cosine
        # similarity, which reads more naturally as a relevance score.
        best_per_recipe[recipe_id] = SearchResult(
            recipe_id=recipe_id,
            title=title,
            cuisine=cuisine,
            category=category,
            tags=list(tags),
            thumbnail_url=thumbnail_url,
            snippet=_snippet_from_chunk(chunk_text),
            score=1.0 - distance,
        )
        if len(best_per_recipe) >= top_k:
            break

    took_ms = (time.perf_counter() - started) * 1000
    return list(best_per_recipe.values()), took_ms

"""Keyword (lexical) search over recipes, using Postgres full-text search.

Chosen over `rank_bm25` in Python: the corpus already lives in Postgres, and
`tsvector`/`tsquery` push ranking down to the database with a GIN index behind
it, so there's no separate in-memory index to build, keep warm, and re-sync
whenever the corpus changes. The tradeoff is a coarser ranking function
(ts_rank_cd approximates BM25's term-frequency saturation but isn't identical)
in exchange for one less moving part -- worth it at this scale, and the kind
of tradeoff worth being able to explain rather than avoid.

Unlike semantic search, this operates directly on `recipes.search_vector`
rather than on chunks: full-text ranking already accounts for document length
and term position across the whole document, so there's no truncation problem
here that would motivate chunking.
"""
from __future__ import annotations

import time

from app.db.connection import get_pool
from app.models.search import SearchResult

SNIPPET_MAX_CHARS = 240

# websearch_to_tsquery understands plain natural-language input (quotes for
# phrases, "-" to exclude, "or" for alternation) and never raises on malformed
# input -- unlike plainto_tsquery/to_tsquery, a query of only stopwords just
# yields zero matches rather than an error. That leniency matters here since
# the query box takes free-form user text, not a query-language expert's input.
KEYWORD_SEARCH_SQL = """
    SELECT
        id, title, cuisine, category, tags, thumbnail_url,
        ts_rank_cd(search_vector, query) AS rank,
        -- `instructions`, not `full_text`: full_text is header-prefixed
        -- (title | cuisine | category | tags), which would duplicate the
        -- title and cuisine already shown alongside the snippet in the UI.
        left(instructions, 500) AS snippet_source
    FROM recipes, websearch_to_tsquery('english', %(query)s) AS query
    WHERE search_vector @@ query
    ORDER BY rank DESC
    LIMIT %(top_k)s
"""


def _snippet(text: str) -> str:
    text = text.strip()
    if len(text) <= SNIPPET_MAX_CHARS:
        return text
    return text[:SNIPPET_MAX_CHARS].rsplit(" ", 1)[0] + "..."


def keyword_search(query: str, top_k: int = 10) -> tuple[list[SearchResult], float]:
    """Full-text search ranked by `ts_rank_cd`. Returns (results, took_ms).

    The score is a raw `ts_rank_cd` value: unbounded and only meaningful
    relative to other results from the same query, unlike semantic search's
    0-2 cosine distance. Callers that need to compare across modes (see
    `hybrid.py`) must normalize it first.
    """
    started = time.perf_counter()

    with get_pool().connection() as conn:
        rows = conn.execute(KEYWORD_SEARCH_SQL, {"query": query, "top_k": top_k}).fetchall()

    results = [
        SearchResult(
            recipe_id=recipe_id,
            title=title,
            cuisine=cuisine,
            category=category,
            tags=list(tags),
            thumbnail_url=thumbnail_url,
            snippet=_snippet(snippet_source),
            score=rank,
        )
        for recipe_id, title, cuisine, category, tags, thumbnail_url, rank, snippet_source in rows
    ]

    took_ms = (time.perf_counter() - started) * 1000
    return results, took_ms

"""Blend semantic and keyword search into a single ranked list.

The two branches score on incomparable scales -- cosine similarity in roughly
[0, 1] versus an unbounded `ts_rank_cd` -- so neither can be added to the other
directly. This blends them with **query-relative min-max normalization**:
within this query's candidate pool, the best semantic match becomes 1.0 and
the worst becomes 0.0, and likewise for keyword; then

    hybrid_score = alpha * semantic_norm + (1 - alpha) * keyword_norm

A recipe missing from one branch's candidates (it didn't semantically
resemble the query at all, or shares no lexemes with it) scores 0 on that
side rather than being excluded outright -- a recipe that one method loves and
the other ignores can still surface, just ranked lower than one both agree on.

Why min-max and not, say, a fixed calibration constant: it needs no
corpus-wide tuning and adapts per query, at the cost of being sensitive to
the candidate pool's composition (a query with one overwhelming keyword match
compresses every other keyword score toward 0). That's an accepted tradeoff
for a project this size, flagged here rather than hidden.

`alpha` defaults to 0.5 -- an even bet between "meaning" and "exact terms" --
and is exposed as a query param so the tradeoff is something a reader can
actually go feel for themselves rather than take on faith.
"""
from __future__ import annotations

import time

from app.models.search import SearchResult
from app.search.keyword import keyword_search
from app.search.semantic import CANDIDATE_MULTIPLIER, MIN_CANDIDATES, semantic_search

DEFAULT_ALPHA = 0.5


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max scale a score map to [0, 1]. A single/tied score maps to 1.0."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    spread = high - low
    if spread == 0:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / spread for key, value in scores.items()}


def hybrid_search(
    query: str, top_k: int = 10, alpha: float = DEFAULT_ALPHA
) -> tuple[list[SearchResult], float]:
    """Blend semantic and keyword candidates. Returns (results, took_ms)."""
    started = time.perf_counter()

    candidate_pool = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    semantic_results, _ = semantic_search(query, top_k=candidate_pool)
    keyword_results, _ = keyword_search(query, top_k=candidate_pool)

    semantic_by_id = {r.recipe_id: r for r in semantic_results}
    keyword_by_id = {r.recipe_id: r for r in keyword_results}
    semantic_norm = _normalize({r.recipe_id: r.score for r in semantic_results})
    keyword_norm = _normalize({r.recipe_id: r.score for r in keyword_results})

    all_ids = semantic_by_id.keys() | keyword_by_id.keys()
    blended: list[SearchResult] = []
    for recipe_id in all_ids:
        base = semantic_by_id.get(recipe_id) or keyword_by_id.get(recipe_id)
        sem_score = semantic_norm.get(recipe_id, 0.0)
        kw_score = keyword_norm.get(recipe_id, 0.0)
        # Prefer the semantic snippet: it's a specific matching passage picked
        # out of the recipe's body, whereas the keyword snippet is just the
        # opening of `full_text` and carries less signal about *why* it matched.
        snippet_source = semantic_by_id.get(recipe_id) or keyword_by_id[recipe_id]
        blended.append(
            SearchResult(
                recipe_id=recipe_id,
                title=base.title,
                cuisine=base.cuisine,
                category=base.category,
                tags=base.tags,
                thumbnail_url=base.thumbnail_url,
                snippet=snippet_source.snippet,
                score=alpha * sem_score + (1 - alpha) * kw_score,
            )
        )

    blended.sort(key=lambda r: r.score, reverse=True)
    took_ms = (time.perf_counter() - started) * 1000
    return blended[:top_k], took_ms

"""API route definitions."""
import time

from fastapi import APIRouter, Query

from app.models.search import SearchResponse
from app.search.hybrid import DEFAULT_ALPHA, hybrid_search
from app.search.keyword import keyword_search
from app.search.rerank import rerank as rerank_candidates
from app.search.semantic import semantic_search

# How many hybrid candidates to pull before cutting down to top_k with the
# cross-encoder. Wider than top_k so reranking has real room to reorder --
# capped well below the corpus size since cross-encoder cost is per-candidate.
RERANK_CANDIDATE_MULTIPLIER = 3
MIN_RERANK_CANDIDATES = 20

router = APIRouter(prefix="/search", tags=["search"])

QueryParam = Query(..., min_length=1, max_length=300, description="Natural-language query")
TopKParam = Query(10, ge=1, le=50, description="Number of recipes to return")


@router.get("/semantic", response_model=SearchResponse)
def search_semantic(q: str = QueryParam, top_k: int = TopKParam):
    """Pure vector similarity search -- no keyword matching involved."""
    results, took_ms = semantic_search(q, top_k=top_k)
    return SearchResponse(query=q, mode="semantic", results=results, took_ms=took_ms)


@router.get("/keyword", response_model=SearchResponse)
def search_keyword(q: str = QueryParam, top_k: int = TopKParam):
    """Pure Postgres full-text search -- no embeddings involved."""
    results, took_ms = keyword_search(q, top_k=top_k)
    return SearchResponse(query=q, mode="keyword", results=results, took_ms=took_ms)


@router.get("/hybrid", response_model=SearchResponse)
def search_hybrid(
    q: str = QueryParam,
    top_k: int = TopKParam,
    alpha: float = Query(
        DEFAULT_ALPHA, ge=0.0, le=1.0,
        description="Weight toward semantic (1.0) vs keyword (0.0) score",
    ),
    rerank: bool = Query(
        False, description="Apply cross-encoder reranking to the hybrid candidates"
    ),
):
    """Blend semantic and keyword results, optionally cross-encoder reranked."""
    if rerank:
        candidate_k = max(top_k * RERANK_CANDIDATE_MULTIPLIER, MIN_RERANK_CANDIDATES)
        candidates, retrieval_ms = hybrid_search(q, top_k=candidate_k, alpha=alpha)
        started = time.perf_counter()
        results = rerank_candidates(q, candidates, top_k=top_k)
        took_ms = retrieval_ms + (time.perf_counter() - started) * 1000
        mode = "hybrid+rerank"
    else:
        results, took_ms = hybrid_search(q, top_k=top_k, alpha=alpha)
        mode = "hybrid"

    return SearchResponse(query=q, mode=mode, results=results, took_ms=took_ms)

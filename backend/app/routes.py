"""API route definitions."""
from fastapi import APIRouter, Query

from app.models.search import SearchResponse
from app.search.hybrid import DEFAULT_ALPHA, hybrid_search
from app.search.keyword import keyword_search
from app.search.semantic import semantic_search

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
):
    """Blend semantic and keyword results into one ranked list."""
    results, took_ms = hybrid_search(q, top_k=top_k, alpha=alpha)
    return SearchResponse(query=q, mode="hybrid", results=results, took_ms=took_ms)

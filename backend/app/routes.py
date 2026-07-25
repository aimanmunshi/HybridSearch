"""API route definitions."""
from fastapi import APIRouter, Query

from app.models.search import SearchResponse
from app.search.semantic import semantic_search

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/semantic", response_model=SearchResponse)
def search_semantic(
    q: str = Query(..., min_length=1, max_length=300, description="Natural-language query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recipes to return"),
):
    """Pure vector similarity search -- no keyword matching involved."""
    results, took_ms = semantic_search(q, top_k=top_k)
    return SearchResponse(query=q, mode="semantic", results=results, took_ms=took_ms)

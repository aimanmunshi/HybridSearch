"""API response schemas for the search endpoints."""
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One recipe in a search response, with the snippet that matched."""

    recipe_id: str
    title: str
    cuisine: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    thumbnail_url: str = ""
    snippet: str = Field(description="The best-matching passage, for display")
    score: float = Field(description="Higher is more relevant. Scale depends on `mode`.")


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]
    took_ms: float

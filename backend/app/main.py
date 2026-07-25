"""FastAPI application entrypoint for the semantic search engine."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(
    title="Semantic Search Engine",
    description="Hybrid (vector + keyword) semantic search over a recipe dataset.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Semantic Search Engine API", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}

"""FastAPI application entrypoint for the semantic search engine."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.connection import close_pool, healthcheck
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open resources on startup and release them on shutdown.

    The embedding model is deliberately *not* warmed here -- it is loaded lazily
    on first search so the container becomes healthy quickly.
    """
    yield
    close_pool()


app = FastAPI(
    title="Semantic Search Engine",
    description="Hybrid (vector + keyword) semantic search over a recipe dataset.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Semantic Search Engine API", "status": "ok"}


@app.get("/health")
def health():
    """Liveness plus a database round-trip."""
    database_ok = healthcheck()
    return {
        "status": "healthy" if database_ok else "degraded",
        "database": "up" if database_ok else "down",
    }

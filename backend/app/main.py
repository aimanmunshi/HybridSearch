"""FastAPI application entrypoint for the semantic search engine."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.db.connection import close_pool, healthcheck
from app.rate_limit import limiter
from app.routes import router

logger = logging.getLogger(__name__)


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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the real error server-side, never leak internals to the client.

    Without this, an unexpected failure (e.g. the DB connection dropping
    mid-query) would bubble up as FastAPI's default 500 page, which in debug
    contexts can include a stack trace -- fine for local development, not for
    something meant to look production-shaped.
    """
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


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

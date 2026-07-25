"""Build the search index from scratch, idempotently.

    python -m scripts.index_corpus              # incremental; reuses caches
    python -m scripts.index_corpus --recreate   # drop and rebuild the schema
    python -m scripts.index_corpus --refresh    # re-download the corpus first

Safe to re-run: recipes and chunks are upserted on their natural keys, so a
second run over an unchanged corpus is a no-op rather than a duplication.
Embedding is the slow step (~1700 chunks on CPU), so it happens once per run
and in batches.
"""
from __future__ import annotations

import argparse
import logging
import time

from app.db.connection import close_pool, run_migrations
from app.db.repository import (
    corpus_stats,
    delete_orphaned_chunks,
    upsert_chunks,
    upsert_recipes,
)
from app.ingestion.embed import get_embedding_provider
from app.ingestion.pipeline import build_corpus

logger = logging.getLogger(__name__)


def index_corpus(recreate: bool = False, refresh: bool = False) -> dict[str, int]:
    """Run the full ingest -> embed -> store cycle."""
    started = time.perf_counter()

    recipes, chunks = build_corpus(force_refresh=refresh)

    provider = get_embedding_provider()
    logger.info("embedding %d chunks with %s", len(chunks), provider.model_name)
    embed_started = time.perf_counter()
    embeddings = provider.embed_documents([c.text for c in chunks])
    embed_seconds = time.perf_counter() - embed_started
    logger.info(
        "embedded %d chunks in %.1fs (%.0f chunks/s)",
        len(chunks),
        embed_seconds,
        len(chunks) / max(embed_seconds, 1e-6),
    )

    # Migrate only after embedding succeeds, so a model-loading failure does not
    # leave a freshly dropped, empty schema behind.
    run_migrations(embedding_dim=provider.dimensions, recreate=recreate)

    upsert_recipes(recipes)
    upsert_chunks(chunks, embeddings)
    delete_orphaned_chunks(chunks)

    stats = corpus_stats()
    logger.info(
        "index ready: %d recipes, %d chunks in %.1fs",
        stats["recipes"],
        stats["chunks"],
        time.perf_counter() - started,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild tables (required when changing embedding model)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-download the corpus first"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        index_corpus(recreate=args.recreate, refresh=args.refresh)
    finally:
        close_pool()


if __name__ == "__main__":
    main()

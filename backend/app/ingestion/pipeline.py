"""Orchestrates fetch -> clean -> chunk, and reports corpus statistics.

Run directly to inspect the pipeline without touching the database:

    python -m app.ingestion.pipeline --stats
    python -m app.ingestion.pipeline --refresh   # bypass the download cache
"""
from __future__ import annotations

import argparse
import logging
import statistics

from app.ingestion.chunk import chunk_recipes
from app.ingestion.clean import clean_recipes
from app.ingestion.fetch import fetch_raw_recipes
from app.models.recipe import Chunk, Recipe

logger = logging.getLogger(__name__)


def build_corpus(force_refresh: bool = False) -> tuple[list[Recipe], list[Chunk]]:
    """Run the full pipeline and return the cleaned recipes and their chunks."""
    raw = fetch_raw_recipes(force_refresh=force_refresh)
    recipes = clean_recipes(raw)
    chunks = chunk_recipes(recipes)
    return recipes, chunks


def _describe(label: str, values: list[int]) -> str:
    if not values:
        return f"{label}: (none)"
    ordered = sorted(values)
    return (
        f"{label}: min={ordered[0]} "
        f"median={int(statistics.median(ordered))} "
        f"p95={ordered[int(len(ordered) * 0.95)]} "
        f"max={ordered[-1]}"
    )


def print_stats(recipes: list[Recipe], chunks: list[Chunk]) -> None:
    """Print corpus statistics; useful for sanity-checking chunk sizing."""
    recipe_words = [len(r.full_text().split()) for r in recipes]
    chunk_words = [len(c.text.split()) for c in chunks]
    per_recipe: dict[str, int] = {}
    for chunk in chunks:
        per_recipe[chunk.recipe_id] = per_recipe.get(chunk.recipe_id, 0) + 1

    print(f"\nRecipes:            {len(recipes)}")
    print(f"Chunks:             {len(chunks)}")
    print(f"Chunks per recipe:  {len(chunks) / max(len(recipes), 1):.2f} avg, "
          f"max {max(per_recipe.values(), default=0)}")
    print(_describe("Recipe words      ", recipe_words))
    print(_describe("Chunk words       ", chunk_words))

    oversized = sum(1 for w in chunk_words if w > 190)
    print(f"Chunks over ~190 words (model limit): {oversized}")

    truncated = sum(1 for w in recipe_words if w > 190)
    print(
        f"Recipes that would be truncated without chunking: "
        f"{truncated}/{len(recipes)} ({truncated / max(len(recipes), 1):.0%})"
    )

    cuisines = {r.cuisine for r in recipes if r.cuisine}
    categories = {r.category for r in recipes if r.category}
    print(f"\nDistinct cuisines:  {len(cuisines)}")
    print(f"Distinct categories: {len(categories)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recipe ingestion pipeline.")
    parser.add_argument(
        "--refresh", action="store_true", help="re-download instead of using the cache"
    )
    parser.add_argument(
        "--stats", action="store_true", help="print corpus statistics"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    recipes, chunks = build_corpus(force_refresh=args.refresh)

    if args.stats:
        print_stats(recipes, chunks)


if __name__ == "__main__":
    main()

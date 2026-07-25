"""Normalise raw TheMealDB records into the internal `Recipe` model.

The upstream data is community-contributed and messy in predictable ways:

- Ingredients are flattened across 20 numbered field pairs
  (`strIngredient1`/`strMeasure1` ... `strIngredient20`/`strMeasure20`), with
  unused slots holding `""`, `" "`, or `null`.
- Instructions use Windows line endings, occasional double-encoded HTML
  entities, and inconsistent "STEP 1" / "step 1" markers.
- Cuisine lives in `strArea` (adjectival, e.g. "Syrian") on older records and
  `strCountry` (e.g. "Syria") on newer ones.

Cleaning matters more than usual here because the same text is used for both
embedding and full-text search: stray markup becomes tokens that dilute the
embedding and pollute the `tsvector`.
"""
from __future__ import annotations

import html
import logging
import re

from app.models.recipe import Ingredient, Recipe

logger = logging.getLogger(__name__)

MAX_INGREDIENT_SLOTS = 20

# "STEP 1", "step 1.", "Step 1)" at the start of a line -- these are structural
# markers, not content, and they add nothing but noise to the embedding.
_STEP_MARKER = re.compile(r"^\s*step\s*\d+\s*[.):]?\s*", re.IGNORECASE | re.MULTILINE)
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# Fold typographic punctuation to ASCII. The corpus uses curly quotes ("don’t"),
# but users type straight ones ("don't"). Postgres' full-text tokeniser treats
# the two as different characters, so without this the keyword half of hybrid
# search misses otherwise-exact matches.
_PUNCTUATION_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ",
})


def _clean_text(value: str | None) -> str:
    """Unescape entities, fold punctuation, and normalise whitespace."""
    if not value:
        return ""
    # Some records are double-escaped (e.g. "&amp;amp;"), so unescape twice.
    text = html.unescape(html.unescape(value))
    text = text.translate(_PUNCTUATION_FOLD)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _extract_ingredients(raw: dict) -> list[Ingredient]:
    """Flatten the 20 numbered ingredient/measure slots into a list."""
    ingredients: list[Ingredient] = []
    for slot in range(1, MAX_INGREDIENT_SLOTS + 1):
        name = _clean_text(raw.get(f"strIngredient{slot}"))
        if not name:
            continue
        measure = _clean_text(raw.get(f"strMeasure{slot}"))
        ingredients.append(Ingredient(name=name, measure=measure))
    return ingredients


def _extract_tags(raw: dict) -> list[str]:
    """Split the comma-separated `strTags` field, dropping blanks."""
    tags = _clean_text(raw.get("strTags"))
    if not tags:
        return []
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def clean_recipe(raw: dict) -> Recipe | None:
    """Convert one raw meal dict into a `Recipe`, or None if unusable.

    A record is unusable if it has no ID, no title, or no instructions --
    without instructions there is nothing meaningful to search over.
    """
    recipe_id = _clean_text(raw.get("idMeal"))
    title = _clean_text(raw.get("strMeal"))
    instructions = _STEP_MARKER.sub("", _clean_text(raw.get("strInstructions")))

    if not recipe_id or not title or not instructions:
        logger.debug("skipping unusable record: id=%r title=%r", recipe_id, title)
        return None

    return Recipe(
        id=recipe_id,
        title=title,
        category=_clean_text(raw.get("strCategory")),
        # Prefer the adjectival cuisine ("Syrian") over the country ("Syria"):
        # it reads more naturally in a query like "spicy Syrian dish".
        cuisine=_clean_text(raw.get("strArea")) or _clean_text(raw.get("strCountry")),
        tags=_extract_tags(raw),
        ingredients=_extract_ingredients(raw),
        instructions=instructions,
        thumbnail_url=_clean_text(raw.get("strMealThumb")),
        source_url=_clean_text(raw.get("strSource")),
    )


def clean_recipes(raw_recipes: list[dict]) -> list[Recipe]:
    """Clean a batch, logging how many records were dropped."""
    cleaned = [recipe for raw in raw_recipes if (recipe := clean_recipe(raw))]
    dropped = len(raw_recipes) - len(cleaned)
    if dropped:
        logger.warning("dropped %d/%d unusable records", dropped, len(raw_recipes))
    logger.info("cleaned %d recipes", len(cleaned))
    return cleaned

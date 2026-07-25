"""Download the raw recipe corpus from TheMealDB.

TheMealDB's free tier has no API key requirement and no documented rate limit,
but it also has no "list everything" endpoint. The standard workaround is to
page through the search-by-first-letter endpoint (a-z), which between them
cover the whole public catalogue (~790 recipes).

The raw JSON is cached to disk so that re-running the pipeline (re-chunking,
re-embedding) never re-hits the network. Delete the cache file to force a
refresh.
"""
from __future__ import annotations

import json
import logging
import string
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.themealdb.com/api/json/v1/1/search.php"
DATA_DIR = Path(__file__).parent / "data"
RAW_CACHE = DATA_DIR / "themealdb_raw.json"

REQUEST_TIMEOUT = 30
RETRIES = 3
RETRY_BACKOFF = 2.0


def _fetch_letter(letter: str, session: requests.Session) -> list[dict]:
    """Fetch every recipe whose title starts with `letter`, with retries."""
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(
                BASE_URL, params={"f": letter}, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            # TheMealDB returns {"meals": null} rather than an empty list.
            return response.json().get("meals") or []
        except (requests.RequestException, ValueError) as exc:
            if attempt == RETRIES:
                logger.error("letter '%s' failed after %d attempts: %s", letter, RETRIES, exc)
                return []
            wait = RETRY_BACKOFF * attempt
            logger.warning("letter '%s' attempt %d failed (%s); retrying in %.0fs", letter, attempt, exc, wait)
            time.sleep(wait)
    return []


def fetch_raw_recipes(force_refresh: bool = False) -> list[dict]:
    """Return the raw meal dicts, reading from the on-disk cache when possible.

    Args:
        force_refresh: Ignore the cache and re-download from the API.
    """
    if RAW_CACHE.exists() and not force_refresh:
        logger.info("loading cached corpus from %s", RAW_CACHE)
        return json.loads(RAW_CACHE.read_text(encoding="utf-8"))

    logger.info("downloading corpus from TheMealDB")
    seen: dict[str, dict] = {}
    with requests.Session() as session:
        for letter in string.ascii_lowercase:
            meals = _fetch_letter(letter, session)
            for meal in meals:
                # A recipe can surface under multiple queries; de-duplicate by ID.
                seen[meal["idMeal"]] = meal
            logger.info("letter '%s': %d recipes (%d unique so far)", letter, len(meals), len(seen))

    recipes = list(seen.values())
    if not recipes:
        raise RuntimeError(
            "Downloaded 0 recipes from TheMealDB. Check network connectivity."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.write_text(json.dumps(recipes, indent=2), encoding="utf-8")
    logger.info("cached %d recipes to %s", len(recipes), RAW_CACHE)
    return recipes

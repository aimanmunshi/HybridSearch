"""Tests for the cleaning stage."""
from app.ingestion.clean import clean_recipe, clean_recipes


def make_raw(**overrides) -> dict:
    """A minimal valid raw record, with per-test overrides."""
    raw = {
        "idMeal": "1001",
        "strMeal": "Test Curry",
        "strCategory": "Chicken",
        "strArea": "Thai",
        "strCountry": "Thailand",
        "strTags": "Spicy,Dinner",
        "strInstructions": "Heat the oil. Add the paste.",
        "strMealThumb": "https://example.com/a.jpg",
        "strSource": "https://example.com/recipe",
    }
    raw.update(overrides)
    return raw


def test_extracts_core_fields():
    recipe = clean_recipe(make_raw(strIngredient1="Chicken", strMeasure1="500 g"))

    assert recipe is not None
    assert recipe.id == "1001"
    assert recipe.title == "Test Curry"
    assert recipe.tags == ["Spicy", "Dinner"]
    assert recipe.ingredients[0].as_text() == "500 g Chicken"


def test_prefers_area_over_country_for_cuisine():
    assert clean_recipe(make_raw()).cuisine == "Thai"


def test_falls_back_to_country_when_area_missing():
    assert clean_recipe(make_raw(strArea=None)).cuisine == "Thailand"


def test_skips_empty_ingredient_slots():
    recipe = clean_recipe(
        make_raw(
            strIngredient1="Chicken", strMeasure1="500 g",
            strIngredient2="  ", strMeasure2="1 cup",   # blank name -> dropped
            strIngredient3=None, strMeasure3=None,
            strIngredient4="Basil", strMeasure4="",     # no measure is fine
        )
    )

    assert [i.name for i in recipe.ingredients] == ["Chicken", "Basil"]
    assert recipe.ingredients[1].as_text() == "Basil"


def test_strips_step_markers():
    recipe = clean_recipe(
        make_raw(strInstructions="STEP 1\r\nHeat oil.\r\nstep 2. Add paste.")
    )

    assert "STEP" not in recipe.instructions
    assert "step 2" not in recipe.instructions
    assert "Heat oil." in recipe.instructions
    assert "Add paste." in recipe.instructions


def test_folds_typographic_punctuation():
    recipe = clean_recipe(
        make_raw(strInstructions="Don’t stir — it’s “done”.")
    )

    assert recipe.instructions == 'Don\'t stir - it\'s "done".'


def test_unescapes_double_encoded_entities():
    recipe = clean_recipe(make_raw(strMeal="Salt &amp;amp; Pepper Squid"))

    assert recipe.title == "Salt & Pepper Squid"


def test_rejects_records_missing_required_fields():
    assert clean_recipe(make_raw(strInstructions="")) is None
    assert clean_recipe(make_raw(strMeal=None)) is None
    assert clean_recipe(make_raw(idMeal="")) is None


def test_clean_recipes_drops_unusable_records():
    cleaned = clean_recipes([make_raw(), make_raw(idMeal="2", strInstructions="")])

    assert [r.id for r in cleaned] == ["1001"]

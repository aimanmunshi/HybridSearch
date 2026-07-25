"""Pydantic schemas for recipes and their embeddable chunks.

These are the internal domain models used by the ingestion pipeline and the
database layer. API response models live in `app.models.search`.
"""
from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    """A single ingredient line, e.g. measure="2 cups", name="flour"."""

    name: str
    measure: str = ""

    def as_text(self) -> str:
        """Render as a human-readable line for embedding/display."""
        return f"{self.measure} {self.name}".strip()


class Recipe(BaseModel):
    """A cleaned recipe, one row in the `recipes` table."""

    id: str = Field(description="Stable upstream ID (TheMealDB idMeal)")
    title: str
    category: str = ""
    cuisine: str = ""
    tags: list[str] = Field(default_factory=list)
    ingredients: list[Ingredient] = Field(default_factory=list)
    instructions: str = ""
    thumbnail_url: str = ""
    source_url: str = ""

    def header_text(self) -> str:
        """Short topical context prepended to every chunk of this recipe.

        Keeping the title/cuisine/tags on each chunk means an isolated chunk of
        instructions is still recognisably "a Thai curry" to the embedding model,
        rather than an anonymous list of steps.
        """
        parts = [self.title]
        if self.cuisine:
            parts.append(f"{self.cuisine} cuisine")
        if self.category:
            parts.append(self.category)
        if self.tags:
            parts.append(", ".join(self.tags))
        return " | ".join(parts)

    def body_text(self) -> str:
        """The full searchable body: ingredients followed by instructions."""
        ingredients = "; ".join(i.as_text() for i in self.ingredients)
        sections = []
        if ingredients:
            sections.append(f"Ingredients: {ingredients}.")
        if self.instructions:
            sections.append(self.instructions)
        return "\n".join(sections)

    def full_text(self) -> str:
        """Header + body, used for keyword (full-text) search and snippets."""
        return f"{self.header_text()}\n{self.body_text()}".strip()


class Chunk(BaseModel):
    """An embeddable slice of a recipe, one row in the `chunks` table."""

    recipe_id: str
    chunk_index: int = Field(description="0-based position within the recipe")
    text: str = Field(description="Header + slice of body; what gets embedded")

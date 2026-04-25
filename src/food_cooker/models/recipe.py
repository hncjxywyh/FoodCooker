from pydantic import BaseModel, Field
from typing import Optional


class RecipeStep(BaseModel):
    step_number: int
    instruction: str
    duration_minutes: Optional[int] = None


class Ingredient(BaseModel):
    name: str
    amount: str  # e.g., "150g", "2 tbsp"
    unit: Optional[str] = None


class AdaptedRecipe(BaseModel):
    name: str
    cuisine: str
    tags: list[str]
    ingredients: list[Ingredient]
    steps: list[RecipeStep]
    estimated_calories: int
    protein_grams: float
    carbs_grams: float
    fat_grams: float
    prep_time_minutes: int
    cook_time_minutes: int
    adaptation_notes: list[str] = Field(default_factory=list)
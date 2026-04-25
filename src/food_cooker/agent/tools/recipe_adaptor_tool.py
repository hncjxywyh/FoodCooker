import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from food_cooker.llm import get_llm

ADAPTOR_PROMPT = """You are a professional chef adapting a recipe for a user with specific constraints.
Base recipe:
{name}
Ingredients: {ingredients}
Steps: {steps}
Tags: {tags}

User profile constraints:
- Allergies: {allergies}
- Diet: {diet}
- Cuisine preference: {cuisine_preference}
- Dislikes: {dislikes}
- Equipment constraints: {equipment_constraints}
- Spice tolerance: {spice_tolerance}

Adapt the recipe to satisfy all constraints. Replace or remove allergenic ingredients.
Return a JSON object with the adapted recipe following this schema:
{{
  "name": "...",
  "cuisine": "...",
  "tags": [...],
  "ingredients": [{{"name": "...", "amount": "..."}}],
  "steps": [{{"step_number": 1, "instruction": "..."}}],
  "estimated_calories": ...,
  "protein_grams": ...,
  "carbs_grams": ...,
  "fat_grams": ...,
  "prep_time_minutes": ...,
  "cook_time_minutes": ...,
  "adaptation_notes": [...]
}}"""


@tool
def recipe_adaptor_tool(base_recipe: str, user_profile: str) -> dict:
    """Adapt a base recipe according to user preferences and constraints.
    base_recipe: JSON string of retrieved recipe metadata.
    user_profile: JSON string of the user profile."""
    try:
        recipe_data = json.loads(base_recipe)
        profile_data = json.loads(user_profile)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON input: {e}", "adapted_recipe": None}

    prompt = ChatPromptTemplate.from_template(ADAPTOR_PROMPT)
    chain = prompt | get_llm(temperature=0.7) | JsonOutputParser()

    try:
        adapted = chain.invoke(
            {
                "name": recipe_data.get("name", ""),
                "ingredients": ", ".join(recipe_data.get("ingredients", [])),
                "steps": " | ".join(str(s) for s in recipe_data.get("steps", [])),
                "tags": ", ".join(recipe_data.get("tags", [])),
                "allergies": ", ".join(profile_data.get("allergies", [])) or "none",
                "diet": profile_data.get("diet", "none"),
                "cuisine_preference": profile_data.get("cuisine_preference", "any"),
                "dislikes": ", ".join(profile_data.get("dislikes", [])) or "none",
                "equipment_constraints": ", ".join(profile_data.get("equipment_constraints", [])) or "none",
                "spice_tolerance": profile_data.get("spice_tolerance", "medium"),
            }
        )
        return adapted
    except Exception as e:
        return {"error": str(e), "adapted_recipe": None}
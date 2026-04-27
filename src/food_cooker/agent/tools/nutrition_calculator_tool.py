import json
import logging
import re
from pathlib import Path
from langchain_core.tools import tool
from food_cooker.settings import settings

logger = logging.getLogger(__name__)


def _load_nutrition_db() -> dict[str, dict]:
    """Load nutrition database from external JSON file."""
    path = settings.nutrition_db_path
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


NUTRITION_DB: dict[str, dict] = _load_nutrition_db()


def _parse_grams(amount_str: str) -> int:
    """Extract gram value from amount string like '150g', '2 tbsp'."""
    digits = re.findall(r"[\d.]+", amount_str)
    if not digits:
        return 100
    val = float(digits[0])
    if "tbsp" in amount_str.lower():
        val *= 15
    elif "tsp" in amount_str.lower():
        val *= 5
    elif "cup" in amount_str.lower():
        val *= 240
    elif "ml" in amount_str.lower():
        val = val
    return int(val)


def _resolve_ingredient_name(raw_name: str) -> str:
    """Resolve ingredient name to a key in NUTRITION_DB using synonym matching.
    Strips quantity suffixes (e.g., '鸡胸肉 150g' -> '鸡胸肉').
    Returns original if no match found."""
    cleaned = re.sub(r"[\d.]+\s*(g|ml|kg|毫克|克|千克|tbsp|tsp|cup|杯|汤匙|茶匙|个|块|片)?\s*$",
                     "", raw_name.strip()).strip()

    if cleaned in NUTRITION_DB:
        return cleaned
    if cleaned.lower() in NUTRITION_DB:
        return cleaned.lower()

    from food_cooker.agent.tools.shopping_list_tool import _get_canonical_name
    canonical = _get_canonical_name(cleaned)
    if canonical in NUTRITION_DB:
        return canonical

    return cleaned


@tool
def nutrition_calculator_tool(ingredients: list[dict]) -> dict:
    """Calculate estimated nutrition for a list of ingredients.
    Each ingredient: {"name": "...", "amount": "..."}.
    Returns total calories, protein, carbs, fat."""
    logger.debug(f"nutrition_calculator_tool ingredients_count={len(ingredients)}")
    totals = {"calories": 0, "protein_grams": 0, "carbs_grams": 0, "fat_grams": 0}
    for item in ingredients:
        raw_name = item.get("name", "")
        amount_str = item.get("amount", "100g")
        grams = _parse_grams(amount_str)
        resolved_name = _resolve_ingredient_name(raw_name)
        nutrient = NUTRITION_DB.get(resolved_name, {"cal": 50, "protein": 2, "carbs": 5, "fat": 2})
        scale = grams / 100.0
        totals["calories"] += nutrient["cal"] * scale
        totals["protein_grams"] += nutrient["protein"] * scale
        totals["carbs_grams"] += nutrient["carbs"] * scale
        totals["fat_grams"] += nutrient["fat"] * scale

    result = {k: round(v, 1) for k, v in totals.items()}
    logger.info(f"nutrition_calculator_tool result calories={result['calories']} protein={result['protein_grams']} carbs={result['carbs_grams']} fat={result['fat_grams']}")
    return result

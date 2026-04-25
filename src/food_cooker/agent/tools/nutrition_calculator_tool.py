import re
from langchain_core.tools import tool

# Hardcoded per 100g / common unit for POC
NUTRITION_DB: dict[str, dict] = {
    "鸡胸肉": {"cal": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    "chicken breast": {"cal": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    "西兰花": {"cal": 34, "protein": 2.8, "carbs": 7, "fat": 0.4},
    "broccoli": {"cal": 34, "protein": 2.8, "carbs": 7, "fat": 0.4},
    "酱油": {"cal": 53, "protein": 8, "carbs": 4, "fat": 0.6},
    "soy sauce": {"cal": 53, "protein": 8, "carbs": 4, "fat": 0.6},
    "橄榄油": {"cal": 884, "protein": 0, "carbs": 0, "fat": 100},
    "olive oil": {"cal": 884, "protein": 0, "carbs": 0, "fat": 100},
    "鸡蛋": {"cal": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    "egg": {"cal": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    "花生": {"cal": 567, "protein": 25, "carbs": 16, "fat": 49},
    "peanuts": {"cal": 567, "protein": 25, "carbs": 16, "fat": 49},
    "鱼片": {"cal": 90, "protein": 18, "carbs": 0, "fat": 2},
    "虾仁": {"cal": 60, "protein": 12, "carbs": 0.5, "fat": 0.7},
    "shrimp": {"cal": 60, "protein": 12, "carbs": 0.5, "fat": 0.7},
    "土豆": {"cal": 76, "protein": 2, "carbs": 17, "fat": 0.1},
    "potato": {"cal": 76, "protein": 2, "carbs": 17, "fat": 0.1},
    "番茄": {"cal": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
    "tomato": {"cal": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
    "黄瓜": {"cal": 15, "protein": 0.7, "carbs": 3.6, "fat": 0.1},
    "cucumber": {"cal": 15, "protein": 0.7, "carbs": 3.6, "fat": 0.1},
    "紫菜": {"cal": 35, "protein": 5, "carbs": 5, "fat": 0.5},
    "生菜": {"cal": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2},
    "lettuce": {"cal": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2},
    "醋": {"cal": 21, "protein": 0, "carbs": 0.9, "fat": 0},
    "vinegar": {"cal": 21, "protein": 0, "carbs": 0.9, "fat": 0},
    "糖": {"cal": 387, "protein": 0, "carbs": 100, "fat": 0},
    "sugar": {"cal": 387, "protein": 0, "carbs": 100, "fat": 0},
    "大蒜": {"cal": 126, "protein": 6, "carbs": 27, "fat": 0.5},
    "garlic": {"cal": 126, "protein": 6, "carbs": 27, "fat": 0.5},
    "葱": {"cal": 30, "protein": 1.6, "carbs": 7, "fat": 0.4},
    "scallion": {"cal": 30, "protein": 1.6, "carbs": 7, "fat": 0.4},
    "姜": {"cal": 41, "protein": 1.9, "carbs": 9, "fat": 0.5},
    "ginger": {"cal": 41, "protein": 1.9, "carbs": 9, "fat": 0.5},
    "芝麻油": {"cal": 884, "protein": 0, "carbs": 0, "fat": 100},
    "sesame oil": {"cal": 884, "protein": 0, "carbs": 0, "fat": 100},
    "干辣椒": {"cal": 40, "protein": 1.5, "carbs": 8, "fat": 0.5},
    "花椒": {"cal": 316, "protein": 11, "carbs": 66, "fat": 8},
    "蚝油": {"cal": 50, "protein": 5, "carbs": 6, "fat": 0},
}


def _parse_grams(amount_str: str) -> int:
    """Extract gram value from amount string like '150g', '2 tbsp'."""
    digits = re.findall(r"[\d.]+", amount_str)
    if not digits:
        return 100
    val = float(digits[0])
    # Rough volume-to-weight for common units
    if "tbsp" in amount_str.lower():
        val *= 15
    elif "tsp" in amount_str.lower():
        val *= 5
    elif "cup" in amount_str.lower():
        val *= 240
    elif "ml" in amount_str.lower():
        val = val  # roughly 1:1 with grams for water-based
    return int(val)


@tool
def nutrition_calculator_tool(ingredients: list[dict]) -> dict:
    """Calculate estimated nutrition for a list of ingredients.
    Each ingredient: {"name": "...", "amount": "..."}.
    Returns total calories, protein, carbs, fat."""
    totals = {"calories": 0, "protein_grams": 0, "carbs_grams": 0, "fat_grams": 0}
    for item in ingredients:
        name = item.get("name", "").lower().strip()
        amount_str = item.get("amount", "100g")
        grams = _parse_grams(amount_str)
        nutrient = NUTRITION_DB.get(name, {"cal": 50, "protein": 2, "carbs": 5, "fat": 2})
        scale = grams / 100.0
        totals["calories"] += nutrient["cal"] * scale
        totals["protein_grams"] += nutrient["protein"] * scale
        totals["carbs_grams"] += nutrient["carbs"] * scale
        totals["fat_grams"] += nutrient["fat"] * scale
    return {k: round(v, 1) for k, v in totals.items()}
from langchain_core.tools import tool

PROTEIN_KEYWORDS = ["鸡", "肉", "鱼", "虾", "蛋", "豆腐", "肉", "肉", "肉"]
SEASONING_KEYWORDS = ["酱油", "盐", "糖", "醋", "料酒", "酱", "蚝油"]
VEG_KEYWORDS = ["菜", "西兰花", "白菜", "萝卜", "葱", "姜", "蒜", "番茄", "黄瓜", "生菜", "土豆"]


def _categorize(name: str) -> str:
    n = name.lower()
    if any(k in n for k in SEASONING_KEYWORDS):
        return "调味品"
    if any(k in n for k in PROTEIN_KEYWORDS):
        return "蛋白质"
    if any(k in n for k in VEG_KEYWORDS):
        return "蔬菜"
    return "其他"


@tool
def shopping_list_tool(
    recipe_ingredients: list[dict],
    user_inventory: list[str],
) -> dict:
    """Compare recipe ingredients against user's existing inventory.
    Return categorized shopping list of missing items."""
    inventory_lower = {i.lower().strip() for i in user_inventory}
    missing = []
    for ing in recipe_ingredients:
        name = ing.get("name", "").lower().strip()
        amount = ing.get("amount", "")
        if not any(name in inv or inv in name for inv in inventory_lower):
            missing.append({"name": ing.get("name", ""), "amount": amount})

    categories: dict[str, list] = {"调味品": [], "蛋白质": [], "蔬菜": [], "其他": []}
    for item in missing:
        categories[_categorize(item["name"])].append(item)

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    return {"shopping_list": categories, "total_items": len(missing)}
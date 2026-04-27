import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 同义词映射：标准名 -> 别名列表
SYNONYM_MAP: dict[str, list[str]] = {
    "鸡蛋": ["蛋", "鸡蛋", "禽蛋"],
    "鸡胸肉": ["鸡胸", "鸡胸肉", "鸡脯肉"],
    "鸡腿肉": ["鸡腿", "鸡腿肉"],
    "猪肉": ["猪肉", "五花肉", "里脊肉", "瘦肉"],
    "牛肉": ["牛肉", "牛腩", "牛腱子"],
    "羊肉": ["羊肉", "羊排"],
    "鱼肉": ["鱼", "鱼肉", "鱼片"],
    "虾仁": ["虾", "虾仁", "大虾"],
    "豆腐": ["豆腐", "嫩豆腐", "老豆腐", "豆干"],
    "土豆": ["土豆", "马铃薯", "洋芋"],
    "番茄": ["番茄", "西红柿", "洋番茄"],
    "黄瓜": ["黄瓜", "青瓜"],
    "生菜": ["生菜", "叶生菜", "球生菜"],
    "白菜": ["白菜", "大白菜", "小白菜"],
    "菠菜": ["菠菜", "青菠菜"],
    "胡萝卜": ["胡萝卜", "红萝卜"],
    "洋葱": ["洋葱", "葱头", "圆葱"],
    "大米": ["米饭", "大米", "白米"],
    "白糖": ["糖", "白砂糖", "砂糖"],
    "牛奶": ["奶", "鲜奶", "纯牛奶"],
    "橄榄油": ["油", "食用油", "植物油"],
}

# 已知同义词集合（用于匹配）
_ALL_SYNONYMS: set[str] = set()
for canonical, synonyms in SYNONYM_MAP.items():
    _ALL_SYNONYMS.add(canonical)
    _ALL_SYNONYMS.update(synonyms)

PROTEIN_KEYWORDS = ["鸡", "肉", "鱼", "虾", "蛋", "豆腐", "牛", "羊", "猪", "蟹", "贝"]
SEASONING_KEYWORDS = ["酱油", "盐", "糖", "醋", "料酒", "酱", "蚝油", "豆瓣", "芥末", "番茄酱"]
VEG_KEYWORDS = ["菜", "西兰花", "白菜", "萝卜", "葱", "姜", "蒜", "番茄", "黄瓜", "生菜", "土豆", "洋葱", "青椒", "茄子", "菠菜"]


def _get_canonical_name(name: str) -> str:
    """Find canonical name for a given ingredient."""
    name_lower = name.lower()
    for canonical, synonyms in SYNONYM_MAP.items():
        if name_lower in [s.lower() for s in synonyms] or name_lower == canonical.lower():
            return canonical
    return name


def _is_match(ingredient_name: str, inventory_name: str) -> bool:
    """Check if ingredient matches inventory item (with synonym support)."""
    ing_lower = ingredient_name.lower().strip()
    inv_lower = inventory_name.lower().strip()

    # Exact match
    if ing_lower == inv_lower:
        return True

    # Substring match (ingredient in inventory or vice versa)
    if ing_lower in inv_lower or inv_lower in ing_lower:
        return True

    # Synonym match: check if they share the same canonical name
    ing_canonical = _get_canonical_name(ingredient_name)
    inv_canonical = _get_canonical_name(inventory_name)
    if ing_canonical and inv_canonical and ing_canonical == inv_canonical:
        return True

    return False


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
    logger.debug(f"shopping_list_tool recipe_ingredients={len(recipe_ingredients)} inventory={len(user_inventory)}")
    if not user_inventory:
        user_inventory = []

    missing = []
    for ing in recipe_ingredients:
        name = ing.get("name", "").strip()
        amount = ing.get("amount", "")

        # Check if any inventory item matches this ingredient
        matched = any(_is_match(name, inv) for inv in user_inventory)
        if not matched:
            missing.append({"name": name, "amount": amount})

    categories: dict[str, list] = {"调味品": [], "蛋白质": [], "蔬菜": [], "其他": []}
    for item in missing:
        categories[_categorize(item["name"])].append(item)

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    result = {"shopping_list": categories, "total_items": len(missing)}
    logger.info(f"shopping_list_tool missing_count={len(missing)} categories={list(categories.keys())}")
    return result

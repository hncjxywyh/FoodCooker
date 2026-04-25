import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from food_cooker.agent.tools.shopping_list_tool import shopping_list_tool


def test_identifies_missing_ingredients():
    recipe_ingredients = [
        {"name": "鸡胸肉", "amount": "200g"},
        {"name": "西兰花", "amount": "150g"},
        {"name": "酱油", "amount": "1 tbsp"},
    ]
    user_inventory = ["鸡胸肉"]  # only has chicken
    result = shopping_list_tool.invoke({
        "recipe_ingredients": recipe_ingredients,
        "user_inventory": user_inventory,
    })
    assert result["total_items"] == 2  # broccoli and soy sauce missing


def test_categorizes_ingredients():
    recipe_ingredients = [
        {"name": "鸡胸肉", "amount": "200g"},
        {"name": "酱油", "amount": "1 tbsp"},
        {"name": "西兰花", "amount": "150g"},
    ]
    user_inventory = []
    result = shopping_list_tool.invoke({
        "recipe_ingredients": recipe_ingredients,
        "user_inventory": user_inventory,
    })
    categories = result["shopping_list"]
    assert "蛋白质" in categories
    assert "调味品" in categories
    assert "蔬菜" in categories


def test_partial_match_in_inventory():
    recipe_ingredients = [{"name": "西兰花", "amount": "150g"}]
    user_inventory = ["西兰花"]  # exact match
    result = shopping_list_tool.invoke({
        "recipe_ingredients": recipe_ingredients,
        "user_inventory": user_inventory,
    })
    assert result["total_items"] == 0


def test_empty_inventory_shows_all():
    recipe_ingredients = [{"name": "鸡蛋", "amount": "2个"}]
    result = shopping_list_tool.invoke({
        "recipe_ingredients": recipe_ingredients,
        "user_inventory": [],
    })
    assert result["total_items"] == 1
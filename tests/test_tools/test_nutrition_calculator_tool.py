import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from food_cooker.agent.tools.nutrition_calculator_tool import nutrition_calculator_tool


def test_calculates_chicken_broccoli():
    ingredients = [
        {"name": "chicken breast", "amount": "150g"},
        {"name": "broccoli", "amount": "100g"},
    ]
    result = nutrition_calculator_tool.invoke({"ingredients": ingredients})
    assert result["calories"] > 0
    assert result["protein_grams"] > 20


def test_handles_unknown_ingredient():
    ingredients = [{"name": "unknown ingredient", "amount": "100g"}]
    result = nutrition_calculator_tool.invoke({"ingredients": ingredients})
    assert result["calories"] > 0  # Uses fallback values


def test_parses_tbsp_amount():
    ingredients = [{"name": "olive oil", "amount": "2 tbsp"}]
    result = nutrition_calculator_tool.invoke({"ingredients": ingredients})
    # 2 tbsp ~ 30g, olive oil 884 cal/100g -> ~265 cal
    assert result["calories"] > 200


def test_empty_list_returns_zeros():
    result = nutrition_calculator_tool.invoke({"ingredients": []})
    assert result["calories"] == 0
    assert result["protein_grams"] == 0
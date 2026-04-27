import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from food_cooker.agent.tools.recipe_retriever_tool import recipe_retriever_tool


def test_retriever_returns_list_of_recipes():
    mock_results = [{
        "name": "西兰花鸡胸肉",
        "cuisine": "Chinese",
        "tags": ["low-carb", "high-protein"],
        "ingredients": [{"name": "鸡胸肉", "amount": "200g"}],
        "steps": ["炒熟"],
        "nutrition": {"calories": 320},
    }]

    with patch("food_cooker.agent.tools.recipe_retriever_tool.hybrid_search", return_value=mock_results):
        result = recipe_retriever_tool.invoke({
            "query": "low-carb high-protein Chinese dinner",
            "tags_filter": ["low-carb"],
            "cuisine_filter": "Chinese",
            "k": 3,
        })

    assert result["count"] == 1
    assert result["recipes"][0]["name"] == "西兰花鸡胸肉"


def test_retriever_with_no_filters():
    with patch("food_cooker.agent.tools.recipe_retriever_tool.hybrid_search", return_value=[]):
        result = recipe_retriever_tool.invoke({"query": "chicken", "k": 3})
    assert result["count"] == 0

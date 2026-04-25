import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from food_cooker.agent.tools.recipe_adaptor_tool import recipe_adaptor_tool


def test_adaptor_handles_invalid_json():
    result = recipe_adaptor_tool.invoke({
        "base_recipe": "not valid json",
        "user_profile": '{"allergies": []}',
    })
    assert "error" in result


def test_adaptor_handles_missing_fields():
    """Test with empty but valid JSON objects."""
    result = recipe_adaptor_tool.invoke({
        "base_recipe": "{}",
        "user_profile": "{}",
    })
    # Should not crash - either returns adapted recipe or error from LLM call
    assert isinstance(result, dict)
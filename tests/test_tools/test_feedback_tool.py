import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from food_cooker.agent.tools.feedback_tool import feedback_tool


@patch("food_cooker.agent.tools.feedback_tool.user_profile_tool")
def test_feedback_spicy_sets_mild(mock_profile_tool):
    mock_profile_tool.invoke.return_value = {"spice_tolerance": "mild", "dislikes": []}
    result = feedback_tool.invoke({
        "session_id": "test_session",
        "feedback_text": "too spicy",
    })
    assert result["status"] == "profile_updated"
    assert result["regenerate"] is True


@patch("food_cooker.agent.tools.feedback_tool.user_profile_tool")
def test_feedback_oily_adds_dislike(mock_profile_tool):
    mock_profile_tool.invoke.return_value = {"dislikes": ["oily_food"]}
    result = feedback_tool.invoke({
        "session_id": "test_session",
        "feedback_text": "太油了",
    })
    assert "oily_food" in result["updated_profile"].get("dislikes", [])


@patch("food_cooker.agent.tools.feedback_tool.user_profile_tool")
def test_feedback_unknown_adds_to_history(mock_profile_tool):
    mock_profile_tool.invoke.return_value = {"feedback_history": ["love it"]}
    result = feedback_tool.invoke({
        "session_id": "test_session",
        "feedback_text": "love it",
    })
    # Unknown feedback just gets added to history
    assert result["status"] == "profile_updated"
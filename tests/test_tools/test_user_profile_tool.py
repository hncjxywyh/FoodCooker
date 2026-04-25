import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from food_cooker.agent.tools.user_profile_tool import user_profile_tool


def test_get_returns_default_profile_when_empty(temp_profiles_path, mock_settings):
    result = user_profile_tool.invoke({
        "action": "get",
        "session_id": "new_session",
        "preferences": None,
    })
    assert result["session_id"] == "new_session"


def test_update_saves_profile(temp_profiles_path, mock_settings):
    updates = {"allergies": ["peanuts"], "diet": "low_carb"}
    result = user_profile_tool.invoke({
        "action": "update",
        "session_id": "session_abc",
        "preferences": updates,
    })
    assert result["allergies"] == ["peanuts"]
    assert result["diet"] == "low_carb"


def test_update_requires_preferences(temp_profiles_path, mock_settings):
    result = user_profile_tool.invoke({
        "action": "update",
        "session_id": "session_abc",
        "preferences": None,
    })
    assert "error" in result
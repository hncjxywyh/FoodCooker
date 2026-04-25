import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture
def temp_profiles_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({}, f)
        path = f.name
    yield Path(path)
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def mock_settings(temp_profiles_path):
    with patch("food_cooker.settings.settings") as mock:
        mock.user_profiles_path = temp_profiles_path
        yield mock


@pytest.fixture
def sample_user_profile():
    return {
        "session_id": "test_session",
        "allergies": ["peanuts"],
        "diet": "low_carb",
        "cuisine_preference": "Chinese",
        "dislikes": ["cilantro"],
        "calorie_target": 500,
        "spice_tolerance": "medium",
        "equipment_constraints": [],
        "user_inventory": ["鸡胸肉", "西兰花", "鸡蛋"],
        "feedback_history": [],
    }
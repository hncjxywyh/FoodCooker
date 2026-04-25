import json
from pathlib import Path
from typing import Literal
from langchain_core.tools import tool
from food_cooker.settings import settings


def _load_profiles() -> dict:
    path = settings.user_profiles_path
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_profiles(profiles: dict) -> None:
    path = settings.user_profiles_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


@tool
def user_profile_tool(
    action: Literal["get", "update"],
    session_id: str,
    preferences: dict | None = None,
) -> dict:
    """Read or update the user profile for the current session.
    Use 'get' to retrieve the profile, 'update' to modify it.
    Returns the full profile after any update."""
    profiles = _load_profiles()
    if action == "get":
        profile_data = profiles.get(session_id, {"session_id": session_id})
        return profile_data
    elif action == "update":
        if preferences is None:
            return {"error": "preferences required for update action"}
        current = profiles.get(session_id, {"session_id": session_id})
        current.update(preferences)
        profiles[session_id] = current
        _save_profiles(profiles)
        return current
    return {"error": "invalid action"}
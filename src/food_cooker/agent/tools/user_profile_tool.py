import json
import logging
from pathlib import Path
from typing import Literal
from langchain_core.tools import tool
from filelock import FileLock
from food_cooker.settings import settings

logger = logging.getLogger(__name__)

LOCK_TIMEOUT = 10  # seconds


def _load_profiles() -> dict:
    path = settings.user_profiles_path
    lock_path = str(path) + ".lock"
    lock = FileLock(lock_path, timeout=LOCK_TIMEOUT)

    with lock:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def _save_profiles(profiles: dict) -> None:
    path = settings.user_profiles_path
    lock_path = str(path) + ".lock"
    lock = FileLock(lock_path, timeout=LOCK_TIMEOUT)

    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)


@tool
def user_profile_tool(
    action: Literal["get", "update", "merge_inventory"],
    session_id: str,
    preferences: dict | None = None,
) -> dict:
    """Read or update the user profile for the current session.
    - 'get': retrieve the profile
    - 'update': replace specified fields
    - 'merge_inventory': add new ingredients to existing inventory (deduplicated)"""
    profiles = _load_profiles()
    logger.debug(f"user_profile_tool action={action} session_id={session_id}")
    if action == "get":
        profile_data = profiles.get(session_id, {"session_id": session_id})
        logger.debug(f"Returning profile for session {session_id}: keys={list(profile_data.keys())}")
        return profile_data
    elif action == "update":
        if preferences is None:
            return {"error": "preferences required for update action"}
        current = profiles.get(session_id, {"session_id": session_id})
        current.update(preferences)
        profiles[session_id] = current
        _save_profiles(profiles)
        logger.info(f"Updated profile for session {session_id}: updated_fields={list(preferences.keys())}")
        return current
    elif action == "merge_inventory":
        if preferences is None:
            return {"error": "preferences required for merge_inventory action"}
        new_items = preferences.get("user_inventory", [])
        current = profiles.get(session_id, {"session_id": session_id})
        existing = set(current.get("user_inventory", []))
        for item in new_items:
            if item.strip():
                existing.add(item.strip())
        current["user_inventory"] = sorted(list(existing))
        profiles[session_id] = current
        _save_profiles(profiles)
        logger.info(f"Merged inventory for session {session_id}: added={len(new_items)}, total={len(current['user_inventory'])}")
        return {"inventory": current["user_inventory"], "total": len(current["user_inventory"])}
    return {"error": "invalid action"}
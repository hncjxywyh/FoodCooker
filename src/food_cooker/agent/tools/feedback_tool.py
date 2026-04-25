from langchain_core.tools import tool
from food_cooker.agent.tools.user_profile_tool import user_profile_tool


@tool
def feedback_tool(session_id: str, feedback_text: str) -> dict:
    """Parse user feedback text and update user profile accordingly.
    Examples: 'too oily' → add 'oily' to dislikes; 'too spicy' → adjust spice_tolerance.
    Returns updated profile and instructs agent to regenerate."""
    feedback_lower = feedback_text.lower()

    updates: dict = {}
    if any(w in feedback_lower for w in ["太油", "oily", "greasy"]):
        updates.setdefault("dislikes", []).append("oily_food")
    if any(w in feedback_lower for w in ["太辣", "spicy", "hot"]):
        updates["spice_tolerance"] = "mild"
    if any(w in feedback_lower for w in ["太淡", "bland", "not flavorful"]):
        updates.setdefault("dislikes", []).append("bland_food")

    if not updates:
        updates["feedback_history"] = [feedback_text]
    else:
        updates["feedback_history"] = [f"User feedback: {feedback_text}"]

    updated_profile = user_profile_tool.invoke({
        "action": "update",
        "session_id": session_id,
        "preferences": updates,
    })

    return {
        "status": "profile_updated",
        "updated_profile": updated_profile,
        "regenerate": True,
        "message": "Preferences updated. Please regenerate the recipe considering the new feedback.",
    }
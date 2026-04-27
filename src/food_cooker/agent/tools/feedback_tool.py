import logging
from langchain_core.tools import tool
from food_cooker.agent.tools.user_profile_tool import user_profile_tool

logger = logging.getLogger(__name__)


@tool
def feedback_tool(session_id: str, feedback_text: str) -> dict:
    """解析用户反馈文本并更新用户档案。
    示例：'太油' → dislikes 添加 '油腻食物'；'太辣' → spice_tolerance 设为 'mild'。
    返回更新后的档案并指示 Agent 重新生成。"""
    logger.debug(f"feedback_tool session_id={session_id} feedback={feedback_text!r}")
    updates: dict = {}
    if any(w in feedback_text for w in ["太油", "油腻", "太腻"]):
        updates.setdefault("dislikes", []).append("油腻食物")
    if any(w in feedback_text for w in ["太辣", "好辣", "太辛辣"]):
        updates["spice_tolerance"] = "mild"
    if any(w in feedback_text for w in ["太淡", "太无聊", "没味道"]):
        updates.setdefault("dislikes", []).append("清淡食物")

    if not updates:
        updates["feedback_history"] = [feedback_text]
    else:
        updates["feedback_history"] = [f"用户反馈: {feedback_text}"]

    logger.info(f"feedback_tool parsed updates={list(updates.keys())} session_id={session_id}")

    updated_profile = user_profile_tool.invoke({
        "action": "update",
        "session_id": session_id,
        "preferences": updates,
    })

    return {
        "status": "profile_updated",
        "updated_profile": updated_profile,
        "regenerate": True,
        "message": "已更新偏好设置，请根据新反馈重新生成食谱。",
    }
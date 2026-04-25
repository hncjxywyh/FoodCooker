import json
from pathlib import Path
from food_cooker.ui.history_store import load_history, get_categories


def get_sessions_by_category() -> dict:
    """Returns {category_name: [(session_id, first_user_message, timestamp, category), ...]}."""
    categories = get_categories()
    result = {}
    for cat in categories:
        result[cat["name"]] = []
        for sid in cat["session_ids"]:
            history = load_history(sid)
            if history.get("entries"):
                first_msg = history["entries"][0]["user"]
                ts = history.get("created_at", history["entries"][0].get("timestamp", ""))
            else:
                first_msg = ""
                ts = history.get("created_at", "")
            result[cat["name"]].append((sid, first_msg, ts))
    return result


def build_category_sidebar_content(categories_dict: dict) -> str:
    """Build markdown for category-organized sidebar."""
    if not categories_dict:
        return "**暂无历史记录**\n\n输入 /sessions 查看并切换历史会话"
    lines = ["## 📋 历史对话\n", "> 输入 `/sessions` 切换会话\n"]
    for cat_name, sessions in categories_dict.items():
        if not sessions:
            continue
        lines.append(f"\n### {cat_name}\n")
        for sid, first_msg, ts in sessions:
            preview = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
            lines.append(f"- `{sid}` {ts}\n  > {preview}")
    return "\n".join(lines)

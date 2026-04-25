import json
from pathlib import Path
from datetime import datetime


CATEGORIES = ["用户身份", "食谱推荐", "营养计算", "购物清单", "其他"]


def get_history_dir() -> Path:
    path = Path("data/history")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_categories_path() -> Path:
    path = Path("data/categories.json")
    return path


def load_history(session_id: str) -> dict:
    """Load history for a session, returns empty structure if none exists."""
    path = get_history_dir() / f"{session_id}.json"
    if not path.exists():
        return {"session_id": session_id, "category": "其他", "entries": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: dict) -> None:
    """Save entire history dict to file."""
    path = get_history_dir() / f"{history['session_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_categories() -> list:
    """Load categories index from categories.json."""
    path = get_categories_path()
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_categories(categories: list) -> None:
    """Save categories index to categories.json."""
    path = get_categories_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)


def update_categories_index(session_id: str, category: str) -> None:
    """Add session_id to the category's list in categories.json."""
    categories = get_categories()
    for cat in categories:
        if cat["name"] == category:
            if session_id not in cat["session_ids"]:
                cat["session_ids"].append(session_id)
            break
    else:
        categories.append({"name": category, "description": "", "session_ids": [session_id]})
    save_categories(categories)


def categorize_conversation(first_user_message: str) -> str:
    """Use LLM to determine category from first user message."""
    from food_cooker.llm import get_llm
    llm = get_llm(temperature=0)
    prompt = f"""你是一个对话分类器。根据用户的第一条消息，判断这个对话属于哪个分类。
可选分类：{CATEGORIES}
只返回一个分类名称，不要其他内容。
用户首条消息：{first_user_message}"""
    try:
        result = llm.invoke([{"role": "user", "content": prompt}])
        category = result.content.strip()
        if category not in CATEGORIES:
            category = "其他"
    except Exception:
        category = "其他"
    return category


def append_entry(session_id: str, user: str, assistant: str, assistant_summary: str = "") -> None:
    """Append a new entry to the session's history file."""
    history = load_history(session_id)

    is_first_entry = len(history["entries"]) == 0
    if is_first_entry:
        category = categorize_conversation(user)
        history["category"] = category
        history["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_categories_index(session_id, category)

    if not assistant_summary:
        assistant_summary = assistant[:100] + "..." if len(assistant) > 100 else assistant
    history["entries"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "assistant_summary": assistant_summary,
        "assistant_full": assistant,
    })
    save_history(history)


def build_history_content(history: dict) -> str:
    """Build markdown content for the history panel from history dict."""
    if not history.get("entries"):
        return "**暂无历史记录**"
    lines = ["## 📋 历史对话\n"]
    for i, entry in enumerate(history["entries"], 1):
        ts = entry.get("timestamp", "")
        user = entry.get("user", "")
        summary = entry.get("assistant_summary", "")
        # Truncate user message for display
        if len(user) > 60:
            user_display = user[:57] + "..."
        else:
            user_display = user
        lines.append(f"**{i}.** {ts}\n> {user_display}\n   → {summary}\n")
    return "\n".join(lines)
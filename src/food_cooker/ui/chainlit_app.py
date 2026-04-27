import chainlit as cl
from chainlit.types import ThreadDict
import uuid
import logging
from typing import Optional
from food_cooker.llm import get_llm
from food_cooker.agent.tools import user_profile_tool
from food_cooker.agent.supervisor import build_agent
from food_cooker.settings import settings
from food_cooker.logging_config import setup_logging

setup_logging()

# LangSmith observability — auto-traces all LLM calls and tool executions
import os as _os
if settings.langsmith_api_key:
    _os.environ["LANGCHAIN_TRACING_V2"] = "true"
    _os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    _os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个个性化的食谱助手（中文回答）。
重要规则：
- 同一轮对话中，每个工具只允许调用一次，禁止重复调用相同工具
- 如果用户询问"还有什么推荐"或类似问题，不需要再次检索，直接基于已有结果回复
- 不要重复检索已经推荐过的食谱
- 用户说"继续推荐"时，直接推荐其他可用食材的菜谱，不要重新检索
- 如果检索结果为空（count=0），不要继续调用下游工具，直接告诉用户没有找到匹配的食谱，并建议用户尝试其他关键词或调整描述
- 当用户说"我冰箱里有X,Y,Z"或"推荐个菜"或"今天吃什么"或类似表达时，先调用 user_profile_tool(action="get") 获取用户已有食材，再据此检索推荐

用户每次说话时的食材库存可能不同，请根据当前用户输入的食材进行推荐。

session_id 由 Chainlit 提供（在 input 的 [session_id=xxx] 中），直接使用此值，不要询问用户。
"""

# Fixed key for cross-session conversation memory (shared across all sessions)
_MEMORY_KEY = "__global_user__"


def _load_memory_context() -> str:
    """Load cross-session conversation memory and return as context string."""
    try:
        profile = user_profile_tool.invoke({"action": "get", "session_id": _MEMORY_KEY})
        memory = profile.get("conversation_memory", "")
        if memory and memory.strip():
            return f"\n\n[你对这个用户的已知信息（来自历史对话）]\n{memory}\n"
    except Exception:
        logger.debug("Failed to load memory context", exc_info=True)
    return ""


def _save_conversation_memory(user_input: str, assistant_response: str) -> None:
    """Generate and persist a brief summary of key user facts from this turn."""
    try:
        summary_llm = get_llm(temperature=0)
        prompt = f"""Based on the conversation below, extract key facts about the user's dietary preferences, allergies, dislikes, favorite cuisines, commonly available ingredients, and any other important dietary information. Write in Chinese, 2-4 sentences max. Only include confirmed facts, not speculations.

User message: {user_input}
Assistant response: {assistant_response[:800]}

Key facts about this user:"""

        summary = summary_llm.invoke(prompt).content
        if summary and summary.strip():
            # Merge with existing memory
            profile = user_profile_tool.invoke({"action": "get", "session_id": _MEMORY_KEY})
            existing = profile.get("conversation_memory", "")
            merged = f"{existing}\n{summary.strip()}".strip() if existing else summary.strip()
            # Cap total memory length to prevent unbounded growth
            if len(merged) > 2000:
                merged = merged[-2000:]
            user_profile_tool.invoke({
                "action": "update",
                "session_id": _MEMORY_KEY,
                "preferences": {"conversation_memory": merged}
            })
            logger.debug(f"Saved conversation memory ({len(merged)} chars)")
    except Exception:
        logger.debug("Failed to save conversation memory", exc_info=True)


# Global agent instance (built once, reused across requests)
agent = build_agent()


@cl.data_layer
def get_data_layer():
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
    return SQLAlchemyDataLayer(conninfo=settings.database_url)


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username == "admin" and password == "admin123":
        return cl.User(identifier="admin", metadata={"role": "admin"})
    return None


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    logger.info(f"on_chat_resume called, thread id={thread.get('id')}, steps count={len(thread.get('steps', []))}")
    session_id = thread.get("id", str(uuid.uuid4())[:8])
    cl.user_session.set("session_id", session_id)

    memory_context = _load_memory_context()
    messages = [{"role": "system", "content": SYSTEM_PROMPT + memory_context}]
    recommended_recipes = []
    import json
    for step in thread.get("steps", []):
        step_type = step.get("type", "")
        if step_type == "user_message":
            content = step.get("input", "") or step.get("content", "")
            if content:
                messages.append({"role": "user", "content": content})
        elif step_type == "assistant_message":
            output = step.get("output", "") or step.get("content", "")
            tool_calls = step.get("toolCalls", [])
            if tool_calls:
                messages.append({"role": "assistant", "content": output, "tool_calls": tool_calls})
            elif output:
                messages.append({"role": "assistant", "content": output})
        elif step_type == "tool":  # 恢复工具消息
            content = step.get("output", "") or step.get("content", "")
            tool_call_id = step.get("toolCallId", "")
            name = step.get("name", "")
            if content:
                # Extract recommended recipes from tool results for deduplication
                if name == "recipe_retriever_tool":
                    try:
                        data = json.loads(content)
                        if "recipes" in data:
                            for recipe in data["recipes"]:
                                r_name = recipe.get("name", "")
                                if r_name and r_name not in recommended_recipes:
                                    recommended_recipes.append(r_name)
                    except (json.JSONDecodeError, ValueError):
                        pass
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": content,
                })

    logger.info(f"Restored {len(messages)} messages for session {session_id}")
    logger.info(f"Restored {len(recommended_recipes)} recommended recipes")
    cl.user_session.set("messages", messages)
    cl.user_session.set("recommended_recipes", recommended_recipes)


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())[:8]
    cl.user_session.set("session_id", session_id)

    # Inject cross-session memory into system prompt
    memory_context = _load_memory_context()
    system_content = SYSTEM_PROMPT + memory_context
    cl.user_session.set("messages", [{"role": "system", "content": system_content}])
    cl.user_session.set("recommended_recipes", [])  # 用于去重

    welcome = "👋 欢迎使用个性化食谱助手！请告诉我你想做什么菜，或者分享你现有的食材。"
    if memory_context.strip():
        welcome += "\n\n> 我已经记住了一些你的偏好信息 🧠"
    await cl.Message(content=welcome).send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
        cl.user_session.set("session_id", session_id)

    messages = cl.user_session.get("messages")
    if messages is None:
        memory_context = _load_memory_context()
        messages = [{"role": "system", "content": SYSTEM_PROMPT + memory_context}]

    user_content = message.content
    full_input = f"[session_id={session_id}] {user_content}"
    messages.append({"role": "user", "content": full_input})

    langchain_messages = _convert_to_langchain_messages(messages)

    msg = cl.Message(content="")
    await msg.send()

    # Stream LLM tokens and tool calls in real-time via astream_events
    streamed_tokens: list[str] = []
    tool_names_seen: set[str] = set()
    final_state = None

    try:
        async for event in agent.astream_events(
            {"messages": langchain_messages},
            config={"configurable": {"thread_id": session_id}},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content
                if content:
                    streamed_tokens.append(content)
                    await msg.stream_token(content)

            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                if tool_name and tool_name not in tool_names_seen:
                    tool_names_seen.add(tool_name)
                    logger.debug(f"Tool started: {tool_name}")
    except Exception:
        logger.exception("Agent streaming failed")
        await msg.stream_token("\n\n抱歉，处理请求时遇到错误，请重试。")
    finally:
        await msg.update()

    # Retrieve final state for conversation continuity
    try:
        final_state = agent.get_state(config={"configurable": {"thread_id": session_id}})
        final_messages = final_state.values.get("messages", langchain_messages)
    except Exception:
        logger.exception("Failed to retrieve agent state, using minimal fallback")
        if streamed_tokens:
            from langchain_core.messages import AIMessage
            final_messages = langchain_messages + [AIMessage(content="".join(streamed_tokens))]
        else:
            final_messages = langchain_messages

    stored_messages = _convert_to_dict_messages(final_messages)

    # Dedup: extract recommended recipe names from tool results
    recommended = cl.user_session.get("recommended_recipes") or []
    for m in stored_messages:
        if m.get("role") == "tool":
            try:
                content = m.get("content", "")
                if isinstance(content, str):
                    import json
                    data = json.loads(content)
                    if "recipes" in data:
                        for recipe in data["recipes"]:
                            name = recipe.get("name", "")
                            if name and name not in recommended:
                                recommended.append(name)
            except (json.JSONDecodeError, ValueError):
                pass
    cl.user_session.set("recommended_recipes", recommended)
    cl.user_session.set("messages", stored_messages)

    # Persist cross-session memory (async, non-blocking)
    assistant_content = ""
    for m in reversed(stored_messages):
        if m.get("role") == "assistant" and m.get("content"):
            assistant_content = m["content"]
            break
    if assistant_content:
        import asyncio as _asyncio
        _asyncio.ensure_future(
            _asyncio.to_thread(_save_conversation_memory, user_content, assistant_content)
        )


def _convert_to_langchain_messages(messages: list[dict]) -> list:
    """Convert dict messages to LangChain message objects."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, ToolCall
    result = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Convert dict tool_calls to LangChain ToolCall objects
                lc_tool_calls = [
                    ToolCall(
                        name=tc.get("name", ""),
                        args=tc.get("args", tc.get("input", {})),
                        id=tc.get("id", ""),
                    )
                    for tc in tool_calls
                ]
                result.append(AIMessage(content=content, tool_calls=lc_tool_calls))
            else:
                result.append(AIMessage(content=content))
        elif role == "tool":
            result.append(ToolMessage(
                content=content,
                tool_call_id=msg.get("tool_call_id", ""),
                name=msg.get("name", ""),
            ))
    return result


def _convert_to_dict_messages(messages: list) -> list[dict]:
    """Convert LangChain message objects back to dict format."""
    result = []
    for msg in messages:
        d = {"role": msg.type, "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
        result.append(d)
    return result

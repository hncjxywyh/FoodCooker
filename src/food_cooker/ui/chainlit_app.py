import chainlit as cl
from chainlit.types import ThreadDict
import uuid
import json
import asyncio
import logging
import time
from typing import Optional
from food_cooker.llm import get_llm
from food_cooker.agent.tools import (
    user_profile_tool,
    recipe_retriever_tool,
    recipe_adaptor_tool,
    nutrition_calculator_tool,
    shopping_list_tool,
    feedback_tool,
)
from food_cooker.settings import settings

logger = logging.getLogger(__name__)

ALL_TOOLS = [
    user_profile_tool,
    recipe_retriever_tool,
    recipe_adaptor_tool,
    nutrition_calculator_tool,
    shopping_list_tool,
    feedback_tool,
]

SYSTEM_PROMPT = """你是一个个性化的食谱助手（中文回答）。
重要规则：
- 同一轮对话中，每个工具只允许调用一次，禁止重复调用相同工具
- 如果用户询问"还有什么推荐"或类似问题，不需要再次检索，直接基于已有结果回复
- 不要重复检索已经推荐过的食谱
- 用户说"继续推荐"时，直接推荐其他可用食材的菜谱，不要重新检索

用户每次说话时的食材库存可能不同，请根据当前用户输入的食材进行推荐。

session_id 由 Chainlit 提供（在 input 的 [session_id=xxx] 中），直接使用此值，不要询问用户。
"""


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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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

    logger.info(f"Restored {len(messages)} messages for session {session_id}")
    cl.user_session.set("messages", messages)


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())[:8]
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("messages", [{"role": "system", "content": SYSTEM_PROMPT}])

    await cl.Message(
        content="👋 欢迎使用个性化食谱助手！请告诉我你想做什么菜，或者分享你现有的食材。"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
        cl.user_session.set("session_id", session_id)

    messages = cl.user_session.get("messages")
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    full_input = f"[session_id={session_id}] {message.content}"

    llm = get_llm(temperature=0.7)
    chain = llm.bind_tools(ALL_TOOLS)

    msg = cl.Message(content="正在思考中...")
    await msg.send()

    messages.append({"role": "user", "content": full_input})

    max_turns = 20
    for turn in range(max_turns):
        turn_start = time.time()
        logger.info(f"Turn {turn + 1}/{max_turns} - invoking LLM...")

        response = await asyncio.to_thread(chain.invoke, messages)
        turn_time = time.time() - turn_start
        logger.info(f"Turn {turn + 1} LLM call done in {turn_time:.1f}s")

        messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})

        if not response.tool_calls:
            logger.info(f"Turn {turn + 1}: No more tool calls, finishing")
            await msg.remove()
            final_msg = cl.Message(content=response.content)
            await final_msg.send()
            cl.user_session.set("messages", messages)
            return

        logger.info(f"Turn {turn + 1}: {len(response.tool_calls)} tool call(s): {[tc['name'] for tc in response.tool_calls]}")

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_obj = next((t for t in ALL_TOOLS if t.name == tool_name), None)
            if tool_obj is None:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_result = await asyncio.to_thread(tool_obj.invoke, tool_args)
                except Exception as e:
                    tool_result = {"error": str(e)}

            logger.info(f"Turn {turn + 1}: {tool_name} -> {list(tool_result.keys()) if isinstance(tool_result, dict) else type(tool_result).__name__}")

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
            messages.append(tool_msg)

    logger.warning("Max turns reached")
    await msg.remove()
    final_msg = cl.Message(content="抱歉，执行次数超限，请重试。")
    await final_msg.send()
    cl.user_session.set("messages", messages)

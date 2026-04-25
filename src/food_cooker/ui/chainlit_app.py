import chainlit as cl
import uuid
import json
import re
import asyncio
import logging
import time
from food_cooker.llm import get_llm
from food_cooker.agent.tools import (
    user_profile_tool,
    recipe_retriever_tool,
    recipe_adaptor_tool,
    nutrition_calculator_tool,
    shopping_list_tool,
    feedback_tool,
)

logger = logging.getLogger(__name__)

# All tools in execution order
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


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
        cl.user_session.set("session_id", session_id)

    # Load conversation history from session, or start fresh with system prompt
    messages = cl.user_session.get("messages")
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    full_input = f"[session_id={session_id}] {message.content}"

    llm = get_llm(temperature=0.7)
    chain = llm.bind_tools(ALL_TOOLS)

    msg = cl.Message(content="正在思考中...")
    await msg.send()

    # Append user message to conversation history
    messages.append({"role": "user", "content": full_input})

    max_turns = 20
    for turn in range(max_turns):
        turn_start = time.time()
        logger.info(f"Turn {turn + 1}/{max_turns} - invoking LLM...")

        # Run synchronous chain.invoke() in thread pool to avoid blocking event loop
        response = await asyncio.to_thread(chain.invoke, messages)
        turn_time = time.time() - turn_start
        logger.info(f"Turn {turn + 1} LLM call done in {turn_time:.1f}s")

        # Add assistant message to history
        messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})

        # If no tool calls, we're done
        if not response.tool_calls:
            logger.info(f"Turn {turn + 1}: No more tool calls, finishing")
            msg.content = response.content
            cl.user_session.set("messages", messages)
            await msg.update()
            return

        logger.info(f"Turn {turn + 1}: {len(response.tool_calls)} tool call(s): {[tc['name'] for tc in response.tool_calls]}")

        # Execute each tool call (these are CPU-bound/local, run in thread for safety)
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_obj = next((t for t in ALL_TOOLS if t.name == tool_name), None)
            if tool_obj is None:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_result = tool_obj.invoke(tool_args)
                except Exception as e:
                    tool_result = {"error": str(e)}

            logger.info(f"Turn {turn + 1}: {tool_name} -> {list(tool_result.keys()) if isinstance(tool_result, dict) else type(tool_result).__name__}")

            # Add tool message
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
            messages.append(tool_msg)

        # Loop continues - next LLM call will process tool results

    # Max turns reached
    logger.warning("Max turns reached")
    msg.content = "抱歉，执行次数超限，请重试。"
    cl.user_session.set("messages", messages)
    await msg.update()
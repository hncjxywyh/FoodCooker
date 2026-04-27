import json
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from food_cooker.api.schemas import ChatRequest
from food_cooker.api.deps import get_agent, get_current_user
from food_cooker.api.db import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

SYSTEM_PROMPT = """你是一个个性化的食谱助手（中文回答）。
重要规则：
- 同一轮对话中，每个工具只允许调用一次，禁止重复调用相同工具
- 如果用户询问"还有什么推荐"或类似问题，不需要再次检索，直接基于已有结果回复
- 不要重复检索已经推荐过的食谱
- 用户说"继续推荐"时，直接推荐其他可用食材的菜谱，不要重新检索
- 如果检索结果为空（count=0），不要继续调用下游工具，直接告诉用户没有找到匹配的食谱，并建议用户尝试其他关键词或调整描述
- 当用户说"我冰箱里有X,Y,Z"或"推荐个菜"或"今天吃什么"或类似表达时，先调用 user_profile_tool(action="get") 获取用户已有食材，再据此检索推荐

用户每次说话时的食材库存可能不同，请根据当前用户输入的食材进行推荐。

session_id 由系统提供（在 input 的 [session_id=xxx] 中），直接使用此值，不要询问用户。
"""


async def _stream_agent_response(message: str, session_id: str):
    """Generator that yields SSE-formatted strings from agent streaming."""
    agent = get_agent()
    langchain_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"[session_id={session_id}] {message}"),
    ]

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
                    yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                if tool_name:
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': tool_name}, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                if tool_name:
                    yield f"data: {json.dumps({'type': 'tool_end', 'name': tool_name}, ensure_ascii=False)}\n\n"

    except Exception:
        logger.exception("Agent streaming failed")
        yield f"data: {json.dumps({'type': 'error', 'content': '处理请求时遇到错误，请重试'}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, user: User = Depends(get_current_user)):
    """Stream agent response via Server-Sent Events."""
    session_id = body.session_id or str(uuid.uuid4())[:8]
    logger.info(f"Streaming chat session_id={session_id} user={user.username}")

    return StreamingResponse(
        _stream_agent_response(body.message, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(body: ChatRequest, user: User = Depends(get_current_user)):
    """Non-streaming chat: return complete agent response."""
    import asyncio

    session_id = body.session_id or str(uuid.uuid4())[:8]
    logger.info(f"Chat session_id={session_id} user={user.username}")

    agent = get_agent()
    langchain_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"[session_id={session_id}] {body.message}"),
    ]

    try:
        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": langchain_messages},
            config={"configurable": {"thread_id": session_id}},
        )
        messages = result.get("messages", [])
        response_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                response_text = msg.content
                break

        return {
            "session_id": session_id,
            "response": response_text or "未能生成回复",
        }
    except Exception:
        logger.exception("Chat invoke failed")
        raise HTTPException(status_code=500, detail="处理请求时遇到错误")

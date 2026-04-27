"""Supervisor-Worker multi-agent architecture for FoodCooker.

Supervisor routes user requests to specialized workers:
- recipe: recipe retrieval + adaptation
- nutrition: nutrition calculation
- shopping: shopping list generation
- general: profile management + feedback
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from food_cooker.llm import get_llm
from food_cooker.agent.tools import (
    user_profile_tool,
    recipe_retriever_tool,
    recipe_adaptor_tool,
    nutrition_calculator_tool,
    shopping_list_tool,
    feedback_tool,
    vision_identify_ingredients_tool,
    image_generation_tool,
)

logger = logging.getLogger(__name__)

# ── Worker tool sets ──────────────────────────────────────────────
RECIPE_TOOLS = [recipe_retriever_tool, recipe_adaptor_tool, image_generation_tool]
NUTRITION_TOOLS = [nutrition_calculator_tool]
SHOPPING_TOOLS = [shopping_list_tool]
PROFILE_TOOLS = [user_profile_tool, feedback_tool, vision_identify_ingredients_tool]

ALL_WORKER_TOOLS = RECIPE_TOOLS + NUTRITION_TOOLS + SHOPPING_TOOLS + PROFILE_TOOLS

# ── Supervisor routing tools ──────────────────────────────────────


@tool
def transfer_to_recipe_worker(query: str) -> str:
    """Transfer to recipe expert. Call when user asks about recipes, meal ideas,
    cooking instructions, recipe recommendations based on ingredients,
    or wants to generate food images for recipes."""
    return f"[recipe_worker] {query}"


@tool
def transfer_to_nutrition_worker(query: str) -> str:
    """Transfer to nutrition expert. Call when user asks about calories, protein,
    carbs, fat, or any nutrition-related questions about specific ingredients."""
    return f"[nutrition_worker] {query}"


@tool
def transfer_to_shopping_worker(query: str) -> str:
    """Transfer to shopping expert. Call when user wants to generate a shopping list,
    compare ingredients against inventory, or organize grocery items."""
    return f"[shopping_worker] {query}"


@tool
def transfer_to_general_worker(query: str) -> str:
    """Transfer to general worker for profile management (get/update preferences,
    merge inventory), feedback processing (record dislikes, spice tolerance),
    or identifying ingredients from food photos using vision AI."""
    return f"[general_worker] {query}"


SUPERVISOR_ROUTER_TOOLS = [
    transfer_to_recipe_worker,
    transfer_to_nutrition_worker,
    transfer_to_shopping_worker,
    transfer_to_general_worker,
]

SUPERVISOR_SYSTEM = """你是主管 Agent，负责协调专家团队回答用户的食谱相关问题。

可用专家：
- recipe_worker: 食谱检索与改编（搜索菜谱、根据约束调整菜谱）
- nutrition_worker: 营养计算与分析（卡路里、蛋白质、碳水、脂肪）
- shopping_worker: 购物清单生成（对比食材库存、分类整理）
- general_worker: 用户档案与反馈（偏好管理、食材库存、用户反馈）

规则：
1. 分析用户请求，判断需要调用哪些专家
2. 简单问候或闲聊直接回复，不调用专家
3. 复杂请求可以依次调用多个专家（如：先搜菜谱 → 再算营养 → 最后生成购物清单）
4. 每个专家最多调用一次
5. 汇总所有专家结果后，用中文给用户一个完整的回复
6. 如果专家返回的结果不足以回答用户问题，如实告知
"""

# ── Worker nodes ──────────────────────────────────────────────────


def _make_worker_node(name: str, tools: list, worker_prompt: str):
    """Create a worker node that uses LLM with its specialized tools."""
    llm = get_llm(temperature=0.5).bind_tools(tools)

    def worker_node(state: MessagesState) -> dict:
        system_msg = SystemMessage(content=worker_prompt)
        response = llm.invoke([system_msg] + list(state["messages"]))
        logger.debug(f"[{name}] worker produced response")
        return {"messages": [response]}

    return worker_node


RECIPE_WORKER_PROMPT = """你是食谱专家。根据用户需求检索并改编菜谱。
- 使用 recipe_retriever_tool 搜索菜谱（支持混合检索：BM25关键词+语义向量）
- 使用 recipe_adaptor_tool 根据用户约束调整菜谱
- 使用 image_generation_tool 为菜谱生成精美食物图片
- 提供清晰的步骤和食材清单
- 用中文回复"""

NUTRITION_WORKER_PROMPT = """你是营养专家。分析食材或菜谱的营养成分。
- 使用 nutrition_calculator_tool 计算热量和营养素
- 用中文回复，列出每种营养素的数据"""

SHOPPING_WORKER_PROMPT = """你是购物专家。对比菜谱食材和用户库存，生成购物清单。
- 使用 shopping_list_tool 找出缺失食材
- 按类别整理（调味品、蛋白质、蔬菜、其他）
- 用中文回复"""

GENERAL_WORKER_PROMPT = """你是用户档案专家。管理用户偏好和反馈。
- 使用 user_profile_tool 读取/更新用户档案
- 使用 feedback_tool 处理用户反馈
- 使用 vision_identify_ingredients_tool 识别用户上传的食物照片中的食材
- 用中文回复"""

# ── Graph builder ─────────────────────────────────────────────────


def build_supervisor_graph():
    """Build the Supervisor-Worker LangGraph graph.

    Returns a compiled graph with checkpointer=None (caller provides checkpointer).
    """

    # Workers
    recipe_worker = _make_worker_node("recipe", RECIPE_TOOLS, RECIPE_WORKER_PROMPT)
    nutrition_worker = _make_worker_node("nutrition", NUTRITION_TOOLS, NUTRITION_WORKER_PROMPT)
    shopping_worker = _make_worker_node("shopping", SHOPPING_TOOLS, SHOPPING_WORKER_PROMPT)
    general_worker = _make_worker_node("general", PROFILE_TOOLS, GENERAL_WORKER_PROMPT)

    # Tool execution nodes (one per worker)
    recipe_tools_node = ToolNode(RECIPE_TOOLS, name="recipe_tools")
    nutrition_tools_node = ToolNode(NUTRITION_TOOLS, name="nutrition_tools")
    shopping_tools_node = ToolNode(SHOPPING_TOOLS, name="shopping_tools")
    general_tools_node = ToolNode(PROFILE_TOOLS, name="general_tools")

    # Supervisor LLM
    supervisor_llm = get_llm(temperature=0.3).bind_tools(SUPERVISOR_ROUTER_TOOLS)

    def supervisor_node(state: MessagesState) -> dict:
        system_msg = SystemMessage(content=SUPERVISOR_SYSTEM)
        response = supervisor_llm.invoke([system_msg] + list(state["messages"]))
        logger.debug(f"Supervisor produced response tool_calls={bool(response.tool_calls)}")
        return {"messages": [response]}

    # Routing helpers
    def _resolve_transfer(last_msg) -> str | None:
        """Parse transfer_to_* tool calls to determine target worker."""
        if not last_msg.tool_calls:
            return None
        for tc in last_msg.tool_calls:
            name = tc.get("name", "")
            if name == "transfer_to_recipe_worker":
                return "recipe_worker"
            elif name == "transfer_to_nutrition_worker":
                return "nutrition_worker"
            elif name == "transfer_to_shopping_worker":
                return "shopping_worker"
            elif name == "transfer_to_general_worker":
                return "general_worker"
        return None

    def supervisor_router(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        target = _resolve_transfer(last_msg)
        if target:
            return target
        return END

    def recipe_router(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "recipe_tools"
        return "supervisor"

    def nutrition_router(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "nutrition_tools"
        return "supervisor"

    def shopping_router(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "shopping_tools"
        return "supervisor"

    def general_router(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "general_tools"
        return "supervisor"

    # Build graph
    workflow = StateGraph(MessagesState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("recipe_worker", recipe_worker)
    workflow.add_node("nutrition_worker", nutrition_worker)
    workflow.add_node("shopping_worker", shopping_worker)
    workflow.add_node("general_worker", general_worker)
    workflow.add_node("recipe_tools", recipe_tools_node)
    workflow.add_node("nutrition_tools", nutrition_tools_node)
    workflow.add_node("shopping_tools", shopping_tools_node)
    workflow.add_node("general_tools", general_tools_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges("supervisor", supervisor_router, {
        "recipe_worker": "recipe_worker",
        "nutrition_worker": "nutrition_worker",
        "shopping_worker": "shopping_worker",
        "general_worker": "general_worker",
        END: END,
    })

    workflow.add_conditional_edges("recipe_worker", recipe_router, {
        "recipe_tools": "recipe_tools",
        "supervisor": "supervisor",
    })
    workflow.add_edge("recipe_tools", "recipe_worker")

    workflow.add_conditional_edges("nutrition_worker", nutrition_router, {
        "nutrition_tools": "nutrition_tools",
        "supervisor": "supervisor",
    })
    workflow.add_edge("nutrition_tools", "nutrition_worker")

    workflow.add_conditional_edges("shopping_worker", shopping_router, {
        "shopping_tools": "shopping_tools",
        "supervisor": "supervisor",
    })
    workflow.add_edge("shopping_tools", "shopping_worker")

    workflow.add_conditional_edges("general_worker", general_router, {
        "general_tools": "general_tools",
        "supervisor": "supervisor",
    })
    workflow.add_edge("general_tools", "general_worker")

    return workflow


def build_agent():
    """Build the Supervisor-Worker agent graph with in-memory checkpointer."""
    workflow = build_supervisor_graph()
    memory_saver = MemorySaver()
    return workflow.compile(checkpointer=memory_saver)

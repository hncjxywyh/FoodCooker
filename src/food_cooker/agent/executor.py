from langchain.agents import create_agent
from food_cooker.agent.tools import (
    user_profile_tool,
    recipe_retriever_tool,
    recipe_adaptor_tool,
    nutrition_calculator_tool,
    shopping_list_tool,
    feedback_tool,
)
from food_cooker.agent.prompts import SYSTEM_PROMPT
from food_cooker.llm import get_llm


def build_agent():
    llm = get_llm(temperature=0.7)

    tools = [
        user_profile_tool,
        recipe_retriever_tool,
        recipe_adaptor_tool,
        nutrition_calculator_tool,
        shopping_list_tool,
        feedback_tool,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def run_agent(query: str, session_id: str = "default"):
    agent = build_agent()
    # Embed session_id into the input so the agent can use it
    input_with_sid = f"[session_id={session_id}] {query}"
    result = agent.invoke({"input": input_with_sid})
    return result
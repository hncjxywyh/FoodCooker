import json
import logging
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from food_cooker.llm import get_llm

logger = logging.getLogger(__name__)

ADAPTOR_PROMPT = """你是一位专业厨师，根据用户的特定约束来调整菜谱。
基础菜谱：
名称：{name}
食材：{ingredients}
步骤：{steps}
标签：{tags}
份量：{servings}人份（请根据此人数调整食材用量）

用户档案约束：
- 过敏原：{allergies}
- 饮食类型：{diet}
- 菜系偏好：{cuisine_preference}
- 不喜欢的食材：{dislikes}
- 设备限制：{equipment_constraints}
- 辛辣耐受度：{spice_tolerance}

根据所有约束调整菜谱。替换或移除含过敏原的食材。
返回符合以下 schema 的 JSON 对象：
{{
  "name": "...",
  "cuisine": "...",
  "tags": [...],
  "ingredients": [{{"name": "...", "amount": "..."}}],
  "steps": [{{"step_number": 1, "instruction": "..."}}],
  "estimated_calories": ...,
  "protein_grams": ...,
  "carbs_grams": ...,
  "fat_grams": ...,
  "prep_time_minutes": ...,
  "cook_time_minutes": ...,
  "servings": ...,
  "adaptation_notes": [...]
}}

重要：你的最终回复必须包含以下信息（如果提供了的话）：
- 改编后的菜名（name）
- 总热量（estimated_calories）和主要营养素（protein_grams, carbs_grams, fat_grams）
- 准备时间（prep_time_minutes）和烹饪时间（cook_time_minutes）
- 份量（servings）
- 关键改编说明（adaptation_notes 中的要点）

如果任何字段为空或不可用，在回复中说明"未提供"。"""


def _format_steps_for_prompt(steps) -> str:
    """Format steps for prompt — handles both string list and object list formats."""
    if not steps:
        return ""
    if isinstance(steps[0], str):
        return " | ".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return " | ".join(f"{s.get('step_number', i+1)}. {s.get('instruction', '')}" for i, s in enumerate(steps))


@tool
def recipe_adaptor_tool(
    base_recipe: str,
    user_profile: str,
    servings: int = 2,
) -> dict:
    """Adapt a base recipe according to user preferences and constraints.
    base_recipe: JSON string of retrieved recipe metadata.
    user_profile: JSON string of the user profile.
    servings: number of people to cook for (default 2)."""
    try:
        recipe_data = json.loads(base_recipe)
        profile_data = json.loads(user_profile)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON input: {e}", "adapted_recipe": None}

    if not recipe_data.get("name"):
        return {"error": "Empty recipe (no name)", "adapted_recipe": None}

    logger.debug(f"recipe_adaptor_tool base_recipe={recipe_data.get('name')!r} servings={servings}")

    prompt = ChatPromptTemplate.from_template(ADAPTOR_PROMPT)
    chain = prompt | get_llm(temperature=0.7) | JsonOutputParser()

    try:
        adapted = chain.invoke(
            {
                "name": recipe_data.get("name", ""),
                "ingredients": ", ".join(recipe_data.get("ingredients", [])),
                "steps": _format_steps_for_prompt(recipe_data.get("steps", [])),
                "tags": ", ".join(recipe_data.get("tags", [])),
                "servings": servings,
                "allergies": ", ".join(profile_data.get("allergies", [])) or "none",
                "diet": profile_data.get("diet", "none"),
                "cuisine_preference": profile_data.get("cuisine_preference", "any"),
                "dislikes": ", ".join(profile_data.get("dislikes", [])) or "none",
                "equipment_constraints": ", ".join(profile_data.get("equipment_constraints", [])) or "none",
                "spice_tolerance": profile_data.get("spice_tolerance", "medium"),
            }
        )
        logger.info(f"recipe_adaptor_tool adapted {recipe_data.get('name')!r} -> {adapted.get('name')!r}")
        return adapted
    except Exception as e:
        logger.error(f"recipe_adaptor_tool failed for {recipe_data.get('name')!r}: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "adapted_recipe": None}

"""Multimodal tool: identify ingredients from food photos using GPT-4V."""

import base64
import logging
from langchain_core.tools import tool
from openai import OpenAI
from food_cooker.settings import settings

logger = logging.getLogger(__name__)


def _get_openai_client() -> OpenAI:
    api_key = settings.openai_api_key_for_images or settings.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key required for vision features")
    return OpenAI(api_key=api_key)


@tool
def vision_identify_ingredients_tool(image_base64: str) -> dict:
    """Analyze a food/fridge photo and identify visible ingredients.
    image_base64: base64-encoded JPEG/PNG image data (without data URI prefix)."""
    if not image_base64:
        return {"error": "No image provided", "ingredients": []}

    logger.debug("vision_identify_ingredients_tool called")
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "识别这张照片中可见的食材。请用中文列出所有可见的食材名称，"
                            "每行一个。只列出明确的食材，不要猜测。"
                            "格式：食材名（如：番茄、鸡蛋、鸡胸肉）"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }],
            max_tokens=300,
        )
        text = response.choices[0].message.content or ""
        ingredients = [line.strip("- 1234567890. ") for line in text.split("\n") if line.strip() and not line.startswith("格式")]
        logger.info(f"vision_identify_ingredients_tool found {len(ingredients)} ingredients")
        return {"ingredients": ingredients, "raw_response": text[:500]}
    except Exception as e:
        logger.error(f"vision_identify_ingredients_tool failed: {e}")
        return {"error": str(e), "ingredients": []}

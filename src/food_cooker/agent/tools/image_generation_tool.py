"""Generate food images using DALL-E 3."""

import logging
from langchain_core.tools import tool
from openai import OpenAI
from food_cooker.settings import settings

logger = logging.getLogger(__name__)


def _get_openai_client() -> OpenAI:
    api_key = settings.openai_api_key_for_images or settings.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key required for image generation")
    return OpenAI(api_key=api_key)


@tool
def image_generation_tool(recipe_name: str, cuisine: str = "Chinese") -> dict:
    """Generate an appetizing food photo for a recipe using DALL-E.
    recipe_name: name of the dish (in Chinese or English).
    cuisine: cuisine type for styling context."""
    logger.debug(f"image_generation_tool recipe={recipe_name!r} cuisine={cuisine!r}")
    try:
        client = _get_openai_client()
        prompt = (
            f"A professional food photography shot of '{recipe_name}', "
            f"{cuisine} cuisine, beautifully plated, warm natural lighting, "
            "overhead angle, high resolution, appetizing and vibrant colors, "
            "restaurant quality"
        )
        response = client.images.generate(
            model=settings.image_generation_model,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = response.data[0].url
        logger.info(f"image_generation_tool generated: {url}")
        return {"image_url": url, "recipe_name": recipe_name, "success": True}
    except Exception as e:
        logger.error(f"image_generation_tool failed: {e}")
        return {"error": str(e), "success": False}

import base64
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from food_cooker.api.deps import get_current_user
from food_cooker.api.db import User
from food_cooker.agent.tools.vision_tool import vision_identify_ingredients_tool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision"])


@router.post("/vision/identify-ingredients")
async def identify_ingredients(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    """Upload a food photo and get identified ingredients list."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    image_b64 = base64.b64encode(contents).decode("utf-8")
    result = vision_identify_ingredients_tool.invoke({"image_base64": image_b64})
    return result

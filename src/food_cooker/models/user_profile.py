from pydantic import BaseModel, Field
from typing import Optional


class UserProfile(BaseModel):
    session_id: str
    allergies: list[str] = Field(default_factory=list)
    diet: Optional[str] = None  # e.g., "low_carb", "keto", "vegetarian"
    cuisine_preference: Optional[str] = None  # e.g., "Chinese", "Italian"
    dislikes: list[str] = Field(default_factory=list)
    calorie_target: Optional[int] = None  # per meal
    spice_tolerance: str = "medium"  # "mild", "medium", "hot"
    equipment_constraints: list[str] = Field(default_factory=list)  # e.g., ["no_oven"]
    user_inventory: list[str] = Field(default_factory=list)  # ingredients user already has
    feedback_history: list[str] = Field(default_factory=list)  # recent feedback strings
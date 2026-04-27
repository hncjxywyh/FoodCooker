import logging
from fastapi import APIRouter, HTTPException, status
from food_cooker.api.schemas import UserCreate, UserLogin, TokenResponse, UserResponse
from food_cooker.api.auth import hash_password, verify_password, create_access_token
from food_cooker.api.db import init_db, get_user_by_username, create_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate):
    """Register a new user."""
    await init_db()
    existing = await get_user_by_username(body.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    user = await create_user(body.username, hash_password(body.password))
    logger.info(f"User registered: {body.username}")
    return UserResponse(id=user.id, username=user.username)


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: UserLogin):
    """Authenticate and return a JWT access token."""
    await init_db()
    user = await get_user_by_username(body.username)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": str(user.id), "username": user.username})
    logger.info(f"User logged in: {body.username}")
    return TokenResponse(access_token=token)

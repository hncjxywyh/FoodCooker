from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from food_cooker.agent.supervisor import build_agent
from food_cooker.api.auth import decode_access_token
from food_cooker.api.db import init_db, get_user_by_username

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

_agent = None


def get_agent():
    """Return the global agent instance (lazy-init, shared across requests)."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate JWT and return the current user."""
    await init_db()
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub"))
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = await get_user_by_username(payload.get("username", ""))
    if user is None or user.id != user_id:
        raise HTTPException(status_code=401, detail="User not found")
    return user

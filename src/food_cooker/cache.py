"""Redis cache layer for recipe retrieval and LLM responses."""

import hashlib
import json
import logging
from functools import wraps
from typing import Optional

import redis
from food_cooker.settings import settings

logger = logging.getLogger(__name__)

_pool: Optional[redis.ConnectionPool] = None


def get_redis() -> Optional[redis.Redis]:
    """Get Redis connection (lazy-init with connection pool). Returns None if unavailable."""
    global _pool
    if _pool is None:
        try:
            _pool = redis.ConnectionPool.from_url(settings.redis_url, socket_connect_timeout=2)
            client = redis.Redis(connection_pool=_pool)
            client.ping()
            logger.info(f"Redis connected: {settings.redis_url}")
        except Exception:
            logger.warning(f"Redis unavailable at {settings.redis_url}, caching disabled")
            _pool = False
            return None
    if _pool is False:
        return None
    return redis.Redis(connection_pool=_pool)


def _make_key(prefix: str, *args, **kwargs) -> str:
    """Create a deterministic cache key from arguments."""
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"foodcooker:{prefix}:{digest}"


def cached(prefix: str, ttl: int = 3600):
    """Decorator: cache function return value in Redis.

    Usage:
        @cached("recipe", ttl=3600)
        def search_recipes(query, k=3):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            r = get_redis()
            if r is None:
                return func(*args, **kwargs)

            key = _make_key(prefix, *args, **kwargs)
            cached_val = r.get(key)
            if cached_val is not None:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(cached_val)

            result = func(*args, **kwargs)
            try:
                r.setex(key, ttl, json.dumps(result, ensure_ascii=False, default=str))
                logger.debug(f"Cache SET: {key} ttl={ttl}s")
            except Exception:
                logger.debug("Failed to cache result", exc_info=True)
            return result
        return wrapper
    return decorator

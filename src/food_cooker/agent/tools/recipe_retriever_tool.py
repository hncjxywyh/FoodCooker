import logging
from typing import Optional
from langchain_core.tools import tool
from food_cooker.vectorstore.hybrid_retriever import hybrid_search

logger = logging.getLogger(__name__)


@tool
def recipe_retriever_tool(
    query: str,
    tags_filter: Optional[list[str]] = None,
    cuisine_filter: Optional[str] = None,
    k: int = 3,
    exclude_recipes: Optional[list[str]] = None,
) -> dict:
    """Retrieve the top-k most relevant recipes using hybrid search (BM25 + vector).
    Supports optional metadata filtering by tags and cuisine.
    exclude_recipes: list of recipe names to exclude from results (for deduplication)."""
    logger.debug(
        f"recipe_retriever_tool query={query!r} k={k} "
        f"tags_filter={tags_filter} cuisine_filter={cuisine_filter} exclude={exclude_recipes}"
    )
    results = hybrid_search(
        query=query,
        k=k,
        tags_filter=tags_filter,
        cuisine_filter=cuisine_filter,
        exclude_recipes=exclude_recipes,
    )
    logger.info(f"recipe_retriever_tool query={query!r} returned {len(results)} recipes")
    return {"recipes": results, "count": len(results)}

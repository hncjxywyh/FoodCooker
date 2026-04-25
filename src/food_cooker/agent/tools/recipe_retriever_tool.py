from typing import Optional
from langchain_core.tools import tool
from food_cooker.vectorstore.chroma_client import get_chroma_client


@tool
def recipe_retriever_tool(
    query: str,
    tags_filter: Optional[list[str]] = None,
    cuisine_filter: Optional[str] = None,
    k: int = 3,
) -> dict:
    """Retrieve the top-k most relevant recipes from the vector store.
    Supports optional metadata filtering by tags and cuisine."""
    db = get_chroma_client()
    filter_dict = {}
    if tags_filter:
        filter_dict["tags"] = {"$in": tags_filter}
    if cuisine_filter:
        filter_dict["cuisine"] = cuisine_filter

    results = db.similarity_search(
        query, k=k, filter=filter_dict if filter_dict else None
    )
    recipes = [
        {
            "name": r.metadata.get("name", ""),
            "cuisine": r.metadata.get("cuisine", ""),
            "tags": r.metadata.get("tags", []),
            "ingredients": r.metadata.get("ingredients", []),
            "steps": r.metadata.get("steps", []),
            "nutrition": r.metadata.get("nutrition", {}),
        }
        for r in results
    ]
    return {"recipes": recipes, "count": len(recipes)}
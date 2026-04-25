"""
Standalone script: python scripts/ingest_recipes.py
Loads recipes from data/recipes_raw.json, embeds them, upserts to Chroma.
"""
import json
from pathlib import Path
from langchain_core.documents import Document
from food_cooker.vectorstore.chroma_client import get_chroma_client
from food_cooker.settings import settings


def load_recipes(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(recipes: list[dict]) -> list[Document]:
    docs = []
    for r in recipes:
        text = (
            f"Dish: {r['name']}, "
            f"Tags: {', '.join(r.get('tags', []))}, "
            f"Cuisine: {r.get('cuisine', 'unknown')}, "
            f"Ingredients: {', '.join(i['name'] for i in r.get('ingredients', []))}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "name": r["name"],
                    "ingredients": json.dumps(r.get("ingredients", []), ensure_ascii=False),
                    "steps": json.dumps(r.get("steps", []), ensure_ascii=False),
                    "tags": r.get("tags", []),
                    "cuisine": r.get("cuisine", "unknown"),
                    "nutrition": json.dumps(r.get("nutrition", {}), ensure_ascii=False),
                },
            )
        )
    return docs


def ingest():
    recipes = load_recipes(settings.recipes_data_path)
    docs = build_documents(recipes)
    db = get_chroma_client()
    db.add_documents(docs)
    print(f"Ingested {len(docs)} recipes into Chroma.")


if __name__ == "__main__":
    ingest()

    
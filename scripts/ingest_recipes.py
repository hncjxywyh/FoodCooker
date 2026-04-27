"""
Standalone script: python scripts/ingest_recipes.py
Loads recipes from data/recipes_raw.json, embeds them, upserts to Chroma.
"""
import json
import logging
from pathlib import Path
from langchain_core.documents import Document
from food_cooker.vectorstore.chroma_client import get_chroma_client
from food_cooker.vectorstore.hybrid_retriever import save_bm25_index
from food_cooker.settings import settings

logger = logging.getLogger(__name__)


def load_recipes(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(recipes: list[dict]) -> list[Document]:
    docs = []
    for r in recipes:
        raw_steps = r.get("steps", [])
        if raw_steps and isinstance(raw_steps[0], str):
            steps = [{"step_number": i + 1, "instruction": s} for i, s in enumerate(raw_steps)]
        else:
            steps = raw_steps

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
                    "steps": json.dumps(steps, ensure_ascii=False),
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
    logger.info(f"Ingested {len(docs)} recipes into Chroma.")

    # Build BM25 index for hybrid retrieval
    save_bm25_index(recipes)
    logger.info(f"Built BM25 index for {len(recipes)} recipes.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ingest()

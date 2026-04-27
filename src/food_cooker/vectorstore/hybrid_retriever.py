"""Hybrid retrieval: BM25 (keyword) + Chroma (semantic) with RRF fusion
and optional cross-encoder reranking."""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi
from food_cooker.vectorstore.chroma_client import get_chroma_client
from food_cooker.settings import settings

logger = logging.getLogger(__name__)

BM25_INDEX_PATH = settings.chroma_db_path.parent / "bm25_index.pkl"
RRF_K = 60

# ── BM25 index ──────────────────────────────────────────────────────

_bm25: Optional[BM25Okapi] = None
_bm25_docs: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Simple character-level tokenization for Chinese text."""
    import re
    tokens = re.findall(r"[一-鿿]+|[a-zA-Z]+|\d+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _build_bm25(recipes: list[dict]) -> BM25Okapi:
    """Build BM25 index from recipe documents."""
    corpus = []
    docs = []
    for r in recipes:
        text = (
            f"Dish: {r['name']}, "
            f"Tags: {', '.join(r.get('tags', []))}, "
            f"Cuisine: {r.get('cuisine', 'unknown')}, "
            f"Ingredients: {', '.join(i['name'] for i in r.get('ingredients', []))}"
        )
        corpus.append(_tokenize(text))
        docs.append({
            "name": r["name"],
            "cuisine": r.get("cuisine", ""),
            "tags": r.get("tags", []),
            "ingredients": r.get("ingredients", []),
            "steps": r.get("steps", []),
            "nutrition": r.get("nutrition", {}),
        })
    return BM25Okapi(corpus), docs


def save_bm25_index(recipes: list[dict]) -> None:
    """Build and persist BM25 index to disk."""
    bm25, docs = _build_bm25(recipes)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    logger.info(f"BM25 index saved: {len(docs)} docs -> {BM25_INDEX_PATH}")


def _load_bm25():
    """Lazy-load BM25 index from disk."""
    global _bm25, _bm25_docs
    if _bm25 is not None:
        return _bm25, _bm25_docs
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _bm25_docs = data["docs"]
        logger.debug(f"BM25 index loaded: {len(_bm25_docs)} docs")
    return _bm25, _bm25_docs


# ── Reranker ────────────────────────────────────────────────────────

_reranker = None


def _get_reranker():
    """Lazy-load cross-encoder reranker."""
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        _reranker = CrossEncoder(model_name)
        logger.info(f"Reranker loaded: {model_name}")
    except Exception:
        logger.warning("Failed to load cross-encoder, reranking disabled", exc_info=True)
        _reranker = False
    return _reranker if _reranker is not False else None


def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates using cross-encoder."""
    reranker = _get_reranker()
    if reranker is None or len(candidates) <= 1:
        return candidates[:top_k]

    pairs = [(query, c["name"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


# ── Hybrid search ───────────────────────────────────────────────────


def hybrid_search(
    query: str,
    k: int = 5,
    tags_filter: Optional[list[str]] = None,
    cuisine_filter: Optional[str] = None,
    exclude_recipes: Optional[list[str]] = None,
    use_rerank: bool = False,
) -> list[dict]:
    """Hybrid search combining BM25 + vector similarity with RRF fusion.

    Returns list of recipe dicts sorted by relevance.
    """
    exclude_set = set(exclude_recipes) if exclude_recipes else set()

    # Build Chroma filter
    chroma_filter = {}
    if tags_filter:
        chroma_filter["tags"] = {"$in": tags_filter}
    if cuisine_filter:
        chroma_filter["cuisine"] = cuisine_filter

    # Fetch more candidates than needed for fusion
    fetch_k = max(k * 3, 10)

    # ── Vector search (Chroma) ──
    db = get_chroma_client()
    vec_results = db.similarity_search(
        query, k=fetch_k, filter=chroma_filter if chroma_filter else None
    )
    vec_rank: dict[str, int] = {}
    vec_docs: dict[str, dict] = {}
    rank = 1
    for r in vec_results:
        name = r.metadata.get("name", "")
        if name not in exclude_set and name not in vec_rank:
            vec_rank[name] = rank
            rank += 1
            vec_docs[name] = {
                "name": name,
                "cuisine": r.metadata.get("cuisine", ""),
                "tags": r.metadata.get("tags", []),
                "ingredients": r.metadata.get("ingredients", []),
                "steps": r.metadata.get("steps", []),
                "nutrition": r.metadata.get("nutrition", {}),
            }

    # ── BM25 keyword search ──
    bm25, bm25_docs = _load_bm25()
    bm25_rank: dict[str, int] = {}
    bm25_docs_map: dict[str, dict] = {}
    if bm25 is not None:
        tokenized = _tokenize(query)
        scores = bm25.get_scores(tokenized)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank = 1
        for idx in ranked_indices:
            doc = bm25_docs[idx]
            name = doc["name"]
            # Apply filters
            if cuisine_filter and doc.get("cuisine", "") != cuisine_filter:
                continue
            if tags_filter and not any(t in doc.get("tags", []) for t in tags_filter):
                continue
            if name not in exclude_set and name not in bm25_rank:
                bm25_rank[name] = rank
                rank += 1
                bm25_docs_map[name] = doc

    # ── Reciprocal Rank Fusion ──
    fused_scores: dict[str, float] = {}
    all_docs: dict[str, dict] = {}

    for name, rank in vec_rank.items():
        fused_scores[name] = 1.0 / (RRF_K + rank)
        all_docs[name] = vec_docs[name]

    for name, rank in bm25_rank.items():
        all_docs[name] = bm25_docs_map.get(name, all_docs.get(name, {}))
        if name in fused_scores:
            fused_scores[name] += 1.0 / (RRF_K + rank)
        else:
            fused_scores[name] = 1.0 / (RRF_K + rank)

    # Sort by fused score
    sorted_names = sorted(fused_scores, key=lambda n: fused_scores[n], reverse=True)

    candidates = []
    for name in sorted_names:
        doc = all_docs.get(name, {})
        doc["score"] = round(fused_scores[name], 4)
        candidates.append(doc)

    # ── Rerank (optional) ──
    if use_rerank and len(candidates) > k:
        candidates = _rerank(query, candidates, k)

    result = candidates[:k]
    logger.info(
        f"hybrid_search query={query!r} k={k} vec_results={len(vec_rank)} "
        f"bm25_results={len(bm25_rank)} fused={len(fused_scores)} final={len(result)}"
    )
    return result

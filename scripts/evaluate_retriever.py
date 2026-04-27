"""RAG retrieval evaluation script.

Computes hit@K, MRR, and precision@K metrics against ground-truth QA pairs.
Run: python scripts/evaluate_retriever.py
"""

import json
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from food_cooker.vectorstore.chroma_client import get_chroma_client
from food_cooker.settings import settings, BASE_DIR

logger = logging.getLogger(__name__)


def load_ground_truth(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_k(
    db,
    qa_pairs: list[dict],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Compute hit@K, MRR, and precision@K for each K value."""
    results: dict[str, float] = {}
    mrr_scores: list[float] = []
    hits: dict[int, int] = {k: 0 for k in k_values}
    precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
    total = len(qa_pairs)

    for qa in qa_pairs:
        query = qa["query"]
        relevant_set = set(qa["relevant"])
        retrieved = db.similarity_search(query, k=max(k_values))
        retrieved_names = [r.metadata.get("name", "") for r in retrieved]

        # MRR
        for rank, name in enumerate(retrieved_names, start=1):
            if name in relevant_set:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

        for k in k_values:
            top_k = retrieved_names[:k]
            matched = sum(1 for name in top_k if name in relevant_set)
            if matched > 0:
                hits[k] += 1
            precision_sums[k] += matched / k

    results["mrr"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    for k in k_values:
        results[f"hit_at_{k}"] = hits[k] / total if total > 0 else 0.0
        results[f"precision_at_{k}"] = precision_sums[k] / total if total > 0 else 0.0

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gt_path = BASE_DIR / "data" / "eval_ground_truth.json"
    if not gt_path.exists():
        logger.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)

    qa_pairs = load_ground_truth(gt_path)
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), gt_path)

    db = get_chroma_client()
    metrics = evaluate_k(db, qa_pairs, k_values=(1, 3, 5))

    # Output results
    print("\n" + "=" * 50)
    print("  RAG Retrieval Evaluation Results")
    print("=" * 50)
    print(f"  Queries evaluated: {len(qa_pairs)}")
    print(f"  Collection count:  {db._collection.count()}")
    print("-" * 50)
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<20} {value:.3f}")
    print("=" * 50)

    # Persist results
    out_dir = BASE_DIR / "data" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"num_queries": len(qa_pairs), **metrics}, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()

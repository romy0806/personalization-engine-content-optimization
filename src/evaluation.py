"""
Recommendation evaluation utilities.
"""

import numpy as np


def precision_at_k(recommended_items, relevant_items, k: int = 10) -> float:
    recommended_k = recommended_items[:k]
    if not recommended_k:
        return 0.0
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / k


def recall_at_k(recommended_items, relevant_items, k: int = 10) -> float:
    if not relevant_items:
        return 0.0
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / len(relevant_items)


def ndcg_at_k(recommended_items, relevant_items, k: int = 10) -> float:
    recommended_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

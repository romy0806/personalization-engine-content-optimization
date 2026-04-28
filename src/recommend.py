"""
Recommendation engine utilities.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def content_based_recommendations(item_id, item_index, item_feature_matrix, top_n: int = 10):
    """Return top-N similar items based on item features."""
    if item_id not in item_index:
        raise ValueError("item_id not found in item_index")

    idx = item_index[item_id]
    similarities = cosine_similarity(item_feature_matrix[idx], item_feature_matrix).ravel()
    top_indices = np.argsort(similarities)[::-1][1:top_n + 1]
    return top_indices, similarities[top_indices]


def rank_items_by_score(items_df, score_col: str = "predicted_engagement", top_n: int = 10):
    """Rank items by predicted engagement score."""
    return items_df.sort_values(score_col, ascending=False).head(top_n)

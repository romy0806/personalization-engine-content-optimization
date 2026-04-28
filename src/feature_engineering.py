"""
Feature engineering utilities.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(text_series: pd.Series, max_features: int = 5000):
    """Convert article text into TF-IDF features."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2
    )
    matrix = vectorizer.fit_transform(text_series.fillna(""))
    return matrix, vectorizer


def create_interaction_score(event_type: str) -> int:
    """Map engagement events to weighted scores."""
    mapping = {
        "view": 1,
        "click": 2,
        "save": 3,
        "share": 4,
        "conversion": 5
    }
    return mapping.get(event_type, 0)

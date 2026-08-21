"""Realistic, deterministic behavioral data for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_PARAMETERS = {
    "High-intent researchers": {
        "recency_days": (2, 1.5),
        "sessions_30d": (17, 4),
        "articles_viewed_30d": (45, 9),
        "click_through_rate": (0.31, 0.05),
        "category_diversity": (5, 1),
        "repeat_visit_rate": (0.72, 0.08),
        "avg_session_depth": (5.8, 0.8),
        "high_intent_actions_30d": (8, 2),
    },
    "Loyal category specialists": {
        "recency_days": (3, 2),
        "sessions_30d": (20, 4),
        "articles_viewed_30d": (38, 8),
        "click_through_rate": (0.24, 0.04),
        "category_diversity": (2, 0.6),
        "repeat_visit_rate": (0.84, 0.06),
        "avg_session_depth": (4.9, 0.7),
        "high_intent_actions_30d": (5, 1.5),
    },
    "Broad explorers": {
        "recency_days": (8, 3),
        "sessions_30d": (10, 3),
        "articles_viewed_30d": (28, 7),
        "click_through_rate": (0.18, 0.04),
        "category_diversity": (8, 1.2),
        "repeat_visit_rate": (0.45, 0.09),
        "avg_session_depth": (3.4, 0.7),
        "high_intent_actions_30d": (2, 1),
    },
    "Casual visitors": {
        "recency_days": (16, 5),
        "sessions_30d": (5, 2),
        "articles_viewed_30d": (10, 4),
        "click_through_rate": (0.11, 0.03),
        "category_diversity": (4, 1),
        "repeat_visit_rate": (0.23, 0.07),
        "avg_session_depth": (2.0, 0.5),
        "high_intent_actions_30d": (0.7, 0.6),
    },
    "At-risk users": {
        "recency_days": (27, 4),
        "sessions_30d": (2.5, 1.2),
        "articles_viewed_30d": (5, 2.5),
        "click_through_rate": (0.07, 0.025),
        "category_diversity": (2.5, 0.8),
        "repeat_visit_rate": (0.10, 0.04),
        "avg_session_depth": (1.4, 0.35),
        "high_intent_actions_30d": (0.2, 0.3),
    },
}


def generate_behavioral_users(
    users_per_segment: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate privacy-safe user behavior with known latent segments."""
    if users_per_segment < 20:
        raise ValueError("users_per_segment must be at least 20")

    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    user_number = 1

    for segment, parameters in SEGMENT_PARAMETERS.items():
        values: dict[str, np.ndarray] = {}
        for feature, (mean, std) in parameters.items():
            sample = rng.normal(mean, std, users_per_segment)
            if feature in {"click_through_rate", "repeat_visit_rate"}:
                sample = np.clip(sample, 0, 1)
            else:
                sample = np.clip(sample, 0, None)
            values[feature] = sample

        frame = pd.DataFrame(values)
        frame.insert(
            0,
            "user_id",
            [f"U{number:06d}" for number in range(user_number, user_number + users_per_segment)],
        )
        frame["known_segment"] = segment
        frames.append(frame)
        user_number += users_per_segment

    return pd.concat(frames, ignore_index=True)

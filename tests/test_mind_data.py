import numpy as np
import pandas as pd

from src.mind_data import build_mind_user_features


def test_mind_feature_engineering_is_user_level(mind_behaviors, mind_news):
    features = build_mind_user_features(mind_behaviors, mind_news)
    assert len(features) == mind_behaviors["user_id"].nunique()
    assert features["user_id"].is_unique
    assert features["click_through_rate"].between(0, 1).all()
    assert np.isfinite(features.select_dtypes("number")).all().all()


def test_mind_features_preserve_behavioral_differences(mind_behaviors, mind_news):
    features = build_mind_user_features(mind_behaviors, mind_news).set_index("user_id")
    assert features.loc["U000", "sessions"] > features.loc["U002", "sessions"]
    assert features.loc["U000", "click_through_rate"] > features.loc["U002", "click_through_rate"]


def test_history_features_use_latest_session_without_double_counting(mind_news):
    behaviors = pd.DataFrame(
        [
            {
                "impression_id": "I1",
                "user_id": "U1",
                "timestamp": "2019-11-01 08:00:00",
                "history": "N1",
                "impressions": "N2-0",
            },
            {
                "impression_id": "I2",
                "user_id": "U1",
                "timestamp": "2019-11-02 08:00:00",
                "history": "N1 N2 N3",
                "impressions": "N4-0",
            },
        ]
    )

    features = build_mind_user_features(behaviors, mind_news).iloc[0]

    assert features["history_length"] == 3
    assert features["category_diversity"] == 3

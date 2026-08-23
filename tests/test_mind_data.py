import numpy as np

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

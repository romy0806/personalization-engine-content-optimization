import numpy as np

from src.mind_data import build_mind_user_features
from src.segmentation import SegmentationConfig, fit_segmentation


def test_mind_segmentation_pipeline(mind_behaviors, mind_news):
    users = build_mind_user_features(mind_behaviors, mind_news)
    result = fit_segmentation(
        users,
        SegmentationConfig(cluster_range=(2, 3), bootstrap_iterations=3),
    )
    assert len(result.assignments) == len(users)
    assert result.assignments["user_id"].is_unique
    assert result.assignments["assignment_strength"].between(0, 1).all()
    assert result.profiles["users"].sum() == len(users)
    assert np.isclose(result.profiles["user_share"].sum(), 1.0)
    assert bool(result.candidates.iloc[0]["business_viable"])

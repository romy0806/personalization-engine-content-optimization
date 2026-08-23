from pathlib import Path

import pandas as pd

from src.mind_data import build_mind_user_features
from src.segmentation import SegmentationConfig, fit_segmentation


def test_export_contract(tmp_path: Path, mind_behaviors, mind_news):
    users = build_mind_user_features(mind_behaviors, mind_news)
    result = fit_segmentation(
        users,
        SegmentationConfig(cluster_range=(2, 3), bootstrap_iterations=2),
    )
    assignments_path = tmp_path / "segment_assignments.csv"
    profiles_path = tmp_path / "segment_profiles.csv"
    result.assignments.to_csv(assignments_path, index=False)
    result.profiles.to_csv(profiles_path, index=False)
    assignments = pd.read_csv(assignments_path)
    profiles = pd.read_csv(profiles_path)
    assert {"user_id", "segment_id", "segment_name", "assignment_strength"}.issubset(
        assignments.columns
    )
    assert {"segment_id", "segment_name", "users", "user_share"}.issubset(profiles.columns)

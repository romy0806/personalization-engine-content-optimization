import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src.segmentation import SegmentationConfig, fit_segmentation
from src.synthetic_data import generate_behavioral_users


@pytest.fixture(scope="module")
def users() -> pd.DataFrame:
    return generate_behavioral_users(users_per_segment=80, random_state=7)


@pytest.fixture(scope="module")
def result(users: pd.DataFrame):
    config = SegmentationConfig(
        cluster_range=(3, 4, 5, 6),
        bootstrap_iterations=5,
        random_state=7,
    )
    return fit_segmentation(users, config)


def test_pipeline_selects_a_viable_candidate(result):
    selected = result.candidates.iloc[0]
    assert selected["business_viable"]
    assert selected["silhouette"] > 0.20
    assert selected["stability_ari"] > 0.50
    assert result.selected_clusters in {3, 4, 5, 6}


def test_assignments_are_complete_and_have_strength(users, result):
    assert len(result.assignments) == len(users)
    assert result.assignments["user_id"].is_unique
    assert result.assignments["segment_name"].notna().all()
    assert result.assignments["assignment_strength"].between(0, 1).all()
    assert "membership_confidence" not in result.assignments.columns


def test_profiles_reconcile_to_population(users, result):
    assert result.profiles["users"].sum() == len(users)
    assert np.isclose(result.profiles["user_share"].sum(), 1.0)
    assert result.profiles["segment_name"].str.len().gt(0).all()


def test_segments_recover_known_behavioral_structure(users, result):
    truth = pd.Categorical(users["known_segment"]).codes
    discovered = result.assignments["segment_id"].to_numpy()
    # Automated selection may merge closely related latent personas when the
    # evidence favors a smaller, more stable solution.
    assert adjusted_rand_score(truth, discovered) > 0.55


def test_missing_features_fail_with_helpful_message(users):
    with pytest.raises(ValueError, match="Missing required segmentation features"):
        fit_segmentation(users.drop(columns=["recency_days"]))


def test_too_few_users_are_rejected(users):
    with pytest.raises(ValueError, match="at least 60 users"):
        fit_segmentation(users.head(20))

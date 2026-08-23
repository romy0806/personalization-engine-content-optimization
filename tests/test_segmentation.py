import numpy as np

from src.mind_data import build_mind_user_features
from src.segmentation import SegmentationConfig, fit_segmentation, validate_segmentation


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


def test_validation_uses_fitted_train_model(
    mind_behaviors,
    mind_validation_behaviors,
    mind_news,
):
    train_users = build_mind_user_features(mind_behaviors, mind_news)
    validation_users = build_mind_user_features(mind_validation_behaviors, mind_news)
    config = SegmentationConfig(cluster_range=(2, 3), bootstrap_iterations=2)
    fitted = fit_segmentation(train_users, config)
    medians_before = fitted.preprocessor.medians_.copy()
    centers_before = (
        fitted.model.means_.copy()
        if hasattr(fitted.model, "means_")
        else fitted.model.cluster_centers_.copy()
    )

    validation = validate_segmentation(fitted, train_users, validation_users)

    assert fitted.preprocessor.medians_.equals(medians_before)
    centers_after = (
        fitted.model.means_ if hasattr(fitted.model, "means_") else fitted.model.cluster_centers_
    )
    assert np.array_equal(centers_before, centers_after)
    assert len(validation.assignments) == len(validation_users)
    assert validation.assignments["user_id"].str.startswith("V").all()
    assert validation.assignments["assignment_strength"].between(0, 1).all()
    assert np.isclose(validation.segment_comparison["train_share"].sum(), 1.0)
    assert np.isclose(validation.segment_comparison["validation_share"].sum(), 1.0)
    assert set(validation.metrics["metric"]) >= {
        "silhouette",
        "segment_distribution_psi",
        "mean_assignment_strength",
    }
    assert set(validation.feature_drift["feature"]) == set(config.features)

import sys
from pathlib import Path

import pandas as pd

from scripts import run_segmentation
from src.mind_data import BEHAVIOR_COLUMNS, NEWS_COLUMNS, build_mind_user_features
from src.segmentation import SegmentationConfig, fit_segmentation, validate_segmentation


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


def test_validation_export_contract(
    tmp_path: Path,
    mind_behaviors,
    mind_validation_behaviors,
    mind_news,
):
    train_users = build_mind_user_features(mind_behaviors, mind_news)
    validation_users = build_mind_user_features(mind_validation_behaviors, mind_news)
    config = SegmentationConfig(cluster_range=(2, 3), bootstrap_iterations=2)
    fitted = fit_segmentation(train_users, config)
    validation = validate_segmentation(fitted, train_users, validation_users)
    comparison_path = tmp_path / "train_validation_segment_comparison.csv"
    metrics_path = tmp_path / "validation_metrics.csv"
    drift_path = tmp_path / "validation_feature_drift.csv"
    validation.segment_comparison.to_csv(comparison_path, index=False)
    validation.metrics.to_csv(metrics_path, index=False)
    validation.feature_drift.to_csv(drift_path, index=False)
    comparison = pd.read_csv(comparison_path)
    metrics = pd.read_csv(metrics_path)
    drift = pd.read_csv(drift_path)
    assert {"train_share", "validation_share", "share_delta"}.issubset(comparison.columns)
    assert {"metric", "value"}.issubset(metrics.columns)
    assert {"feature", "psi", "scaled_mean_delta"}.issubset(drift.columns)


def test_runner_uses_train_and_validation_splits(
    tmp_path: Path,
    monkeypatch,
    mind_behaviors,
    mind_validation_behaviors,
    mind_news,
):
    train_behaviors_path = tmp_path / "train_behaviors.tsv"
    validation_behaviors_path = tmp_path / "validation_behaviors.tsv"
    train_news_path = tmp_path / "train_news.tsv"
    validation_news_path = tmp_path / "validation_news.tsv"
    output_path = tmp_path / "outputs"
    mind_behaviors.loc[:, BEHAVIOR_COLUMNS].to_csv(
        train_behaviors_path, sep="\t", header=False, index=False
    )
    mind_validation_behaviors.loc[:, BEHAVIOR_COLUMNS].to_csv(
        validation_behaviors_path, sep="\t", header=False, index=False
    )
    mind_news.loc[:, NEWS_COLUMNS].to_csv(train_news_path, sep="\t", header=False, index=False)
    mind_news.loc[:, NEWS_COLUMNS].to_csv(validation_news_path, sep="\t", header=False, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_segmentation",
            "--train-behaviors",
            str(train_behaviors_path),
            "--train-news",
            str(train_news_path),
            "--validation-behaviors",
            str(validation_behaviors_path),
            "--validation-news",
            str(validation_news_path),
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        run_segmentation,
        "SegmentationConfig",
        lambda: SegmentationConfig(cluster_range=(2, 3), bootstrap_iterations=2),
    )

    run_segmentation.main()

    expected_outputs = {
        "train_user_features.csv",
        "validation_user_features.csv",
        "train_segment_assignments.csv",
        "train_segment_profiles.csv",
        "segmentation_model_comparison.csv",
        "validation_segment_assignments.csv",
        "validation_segment_profiles.csv",
        "train_validation_segment_comparison.csv",
        "validation_metrics.csv",
        "validation_feature_drift.csv",
    }
    assert {path.name for path in output_path.iterdir()} == expected_outputs

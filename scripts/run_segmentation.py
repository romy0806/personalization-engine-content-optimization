"""Build Microsoft MIND user features, fit segmentation, and export evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.mind_data import load_mind_user_features
from src.segmentation import SegmentationConfig, fit_segmentation, validate_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-behaviors", type=Path, required=True)
    parser.add_argument("--train-news", type=Path, required=True)
    parser.add_argument("--validation-behaviors", type=Path, required=True)
    parser.add_argument("--validation-news", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_users = load_mind_user_features(args.train_behaviors, args.train_news)
    validation_users = load_mind_user_features(
        args.validation_behaviors,
        args.validation_news,
    )
    config = SegmentationConfig()
    result = fit_segmentation(train_users, config)
    validation = validate_segmentation(result, train_users, validation_users)
    args.output.mkdir(parents=True, exist_ok=True)
    train_users.to_csv(args.output / "train_user_features.csv", index=False)
    validation_users.to_csv(args.output / "validation_user_features.csv", index=False)
    result.assignments.to_csv(args.output / "train_segment_assignments.csv", index=False)
    result.profiles.to_csv(args.output / "train_segment_profiles.csv", index=False)
    result.candidates.to_csv(args.output / "segmentation_model_comparison.csv", index=False)
    validation.assignments.to_csv(args.output / "validation_segment_assignments.csv", index=False)
    validation.profiles.to_csv(args.output / "validation_segment_profiles.csv", index=False)
    validation.segment_comparison.to_csv(
        args.output / "train_validation_segment_comparison.csv", index=False
    )
    validation.metrics.to_csv(args.output / "validation_metrics.csv", index=False)
    validation.feature_drift.to_csv(args.output / "validation_feature_drift.csv", index=False)
    selected = result.candidates.iloc[0]
    validation_metric_map = validation.metrics.set_index("metric")["value"]
    print(
        f"Selected {result.selected_algorithm} with {result.selected_clusters} segments; "
        f"train silhouette={selected['silhouette']:.3f}, "
        f"train stability_ari={selected['stability_ari']:.3f}, "
        f"validation silhouette={validation_metric_map['silhouette']:.3f}, "
        f"segment PSI={validation_metric_map['segment_distribution_psi']:.3f}."
    )


if __name__ == "__main__":
    main()
